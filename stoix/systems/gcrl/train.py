import navix as nx
import distrax
from navix.entities import Entities
from stoix.wrappers.navix import NavixWrapper, NavixGoalWrapper
from stoix.systems.gcrl.envs.random_goals import RandomGoals
import jax
import jax.numpy as jnp
from jumanji.env import Environment
import chex
import copy
import time
from typing import Any, Dict, Tuple, Callable

import flashbax as fbx
import flax
import hydra
import optax
import rlax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from stoix.base_types import (
    ActorApply,
    ActorCriticOptStates,
    ActorCriticParams,
    AnakinExperimentOutput,
    CriticApply,
    LearnerFn,
    OffPolicyLearnerState,
)
from stoix.evaluator import evaluator_setup, get_distribution_act_fn
from stoix.networks.base import ContrastiveNet as Net
from stoix.networks.torso import ContrastiveTorso as Torso
from stoix.networks.inputs import ObservationActionInput, EmbeddingInput
from stoix.systems.gcrl.gcrl_types import Transition, ContrastiveOptState, ContrastiveParams
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.multistep import batch_discounted_returns
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate
from stoix.wrappers.episode_metrics import get_final_step_metrics
from stoix.wrappers.transforms import FlattenObservationWrapper
from stoix.wrappers.episode_metrics import RecordEpisodeMetrics
from stoix.systems.gcrl.buffer import flatten_crl_fn

from stoix.wrappers.gcrl import TrajectoryIdWrapper


def log_softmax(logits, axis, resubs):
    if not resubs:
        I = jnp.eye(logits.shape[0])
        big = 100
        eps = 1e-6
        return logits, -jax.nn.logsumexp(logits - big * I + eps, axis=axis, keepdims=True)
    else:
        return logits, -jax.nn.logsumexp(logits, axis=axis, keepdims=True)


def compute_energy(energy_fn, sa_repr, g_repr):
    if energy_fn == "l2":
        logits = - \
            jnp.sqrt(
                jnp.sum((sa_repr - g_repr) ** 2, axis=-1))
    elif energy_fn == "l1":
        logits = - \
            jnp.sum(jnp.abs(sa_repr - g_repr), axis=-1)
    elif energy_fn == "dot":
        logits = jnp.einsum("bjk,bjk->bj", sa_repr, g_repr)
    else:
        raise ValueError(f"Unknown energy function: {energy_fn}")
    # jax.debug.print("post energy {x}", x=logits.shape)
    print(f"post energy {logits.shape}")
    return logits


def compute_loss(contrastive_loss_fn, logits, resubs):
    # if contrastive_loss_fn == "symmetric_infonce":
    print(f"logits {logits.shape}")
    l_align, l_unif = log_softmax(logits, axis=0, resubs=resubs)
    loss = -jnp.mean(jnp.diag(l_align + l_unif))
    print(f"l_align: {l_align.shape}")
    print(f"l_unif: {l_unif.shape}")
    return loss, l_align, l_unif


def get_learner_fn(
    config: DictConfig,
    env: Environment,
    apply_fns: Tuple[CriticApply, CriticApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    buffer_fns: Tuple[Callable, Callable]
) -> LearnerFn[OffPolicyLearnerState]:
    """Get the learner function."""

    # Get apply and update functions for actor and critic networks.
    sa_apply_fn, g_apply_fn, actor_apply_fn = apply_fns
    critic_update_fn = update_fns
    buffer_add_fn, buffer_sample_fn, vmap_flatten_crl_fn = buffer_fns

    def _update_step(
        learner_state: OffPolicyLearnerState, _: Any
    ) -> Tuple[OffPolicyLearnerState, Tuple]:
        def _env_step(
            learner_state: OffPolicyLearnerState, _: Any
        ) -> Tuple[OffPolicyLearnerState, Transition]:
            """Step the environment."""
            params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

            # SELECT ACTION
            key, policy_key = jax.random.split(key)

            print("goal ", last_timestep.extras["goal"])
            action = actor_apply_fn(
                params, last_timestep.observation, last_timestep.extras["goal"], policy_key)
            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(
                env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = timestep.last().reshape(-1)
            goal = timestep.extras["goal"]
            player_pos = timestep.extras["player"]
            traj_id = timestep.extras["traj_id"]
            info = timestep.extras["episode_metrics"]

            transition = Transition(
                done, action,
                timestep.reward, timestep.observation,
                player_pos, goal, traj_id, info
            )

            learner_state = OffPolicyLearnerState(
                params, opt_states, buffer_state, key, env_state, timestep)
            return learner_state, transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, 32
        )

        params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

        traj_batch = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)
        buffer_state = buffer_add_fn(buffer_state, traj_batch)

        def _critic_loss_fn(params, transitions, energy_fn_name):
            """Calculate the critic loss."""
            # RERUN NETWORK
            sa_params, g_params = params
            sa_repr = sa_apply_fn(
                sa_params, transitions.obs, transitions.action)
            g_repr = g_apply_fn(
                g_params, transitions.goal)

            print(f"g dim: {g_repr.shape}")
            print(f"sa dim: {sa_repr.shape}")
            contrastive_loss_fn_name = ""
            resubs = True

            # Compute energy and loss
            logits = compute_energy(energy_fn_name, sa_repr, g_repr)
            loss, l_align, l_unif = compute_loss(
                contrastive_loss_fn_name, logits, resubs)

            logsumexp_penalty = 0
            l2_penalty = 0

            # Modify loss (logsumexp, L2 penalty)
            if logsumexp_penalty > 0:
                # For backward we can check jax.nn.logsumexp(logits, axis=0)
                # VM: we could also try removing the diagonal here when using logsumexp penalty + resubs=False
                logits_ = logits
                big = 100
                I = jnp.eye(logits.shape[0])

                if not resubs:
                    logits_ = logits - big * I

                eps = 1e-6
                logsumexp = jax.nn.logsumexp(logits_ + eps, axis=1)
                loss += logsumexp_penalty * jnp.mean(logsumexp**2)

            if l2_penalty > 0:
                l2_loss = l2_penalty * \
                    (jnp.mean(sa_repr**2) + jnp.mean(g_repr**2))
                loss += l2_loss
            else:
                l2_loss = 0

            loss_info = {"loss": loss}
            return loss, loss_info

        key, sample_key, flatten_key = jax.random.split(key, 3)
        traj_batch = buffer_sample_fn(buffer_state, sample_key)

        flatten_keys = jax.random.split(
            flatten_key, config.system.total_batch_size)
        traj_batch = vmap_flatten_crl_fn(
            config, env, traj_batch.experience, flatten_keys)

        # CALCULATE CRITIC LOSS
        critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
        critic_grads, critic_loss_info = critic_grad_fn(
            params, traj_batch, "dot"
        )

        # Compute the parallel mean (pmean) over the batch.
        # This calculation is inspired by the Anakin architecture demo notebook.
        # available at https://tinyurl.com/26tdzs5x
        # This pmean could be a regular mean as the batch axis is on the same device.
        # pmean over devices.

        critic_grads, critic_loss_info = jax.lax.pmean(
            (critic_grads, critic_loss_info), axis_name="batch"
        )
        # pmean over devices.
        critic_grads, critic_loss_info = jax.lax.pmean(
            (critic_grads, critic_loss_info), axis_name="device"
        )

        # UPDATE CRITIC PARAMS AND OPTIMISER STATE
        critic_updates, critic_new_opt_state = critic_update_fn(
            critic_grads, opt_states)
        critic_new_params = optax.apply_updates(
            params, critic_updates)

        # PACK LOSS INFO
        loss_info = {
            **critic_loss_info,
        }

        metric = traj_batch.info
        learner_state = OffPolicyLearnerState(
            critic_new_params, critic_new_opt_state, buffer_state, key, env_state, last_timestep
        )
        # metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: OffPolicyLearnerState,
    ) -> AnakinExperimentOutput[OffPolicyLearnerState]:

        batched_update_step = jax.vmap(
            _update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.arch.num_updates_per_eval
        )
        return AnakinExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: Environment, keys: chex.Array, config: DictConfig
):
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number/dimension of actions.
    num_actions = int(env.action_spec().num_values)
    init_action = env.action_spec().generate_value()[None]
    # print(f"action shape: {init_action.shape}")

    # PRNG keys.
    key, sa_key, g_key = keys

    sa_network = Net(torso=Torso(
        64), input_layer=ObservationActionInput())
    g_network = Net(torso=Torso(64), input_layer=EmbeddingInput())

    lr = make_learning_rate(config.system.critic_lr, config, 1, 1)

    optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )

    # Initialise observation
    init_obs = env.observation_spec().generate_value()
    init_obs = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)
    # print(f"obs shape: {init_obs.agent_view.shape}")

    init_goal = env.goal_spec().generate_value()
    init_goal = jax.tree_util.tree_map(lambda x: x[None, ...], init_goal)
    # print(f"goal shape: {init_goal.shape}")
    #
    # Create replay buffer
    dummy_transition = Transition(
        obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_obs),
        player_pos=jnp.zeros(2, dtype=int),
        goal=jnp.zeros(2, dtype=int),
        action=jnp.zeros((), dtype=int),
        reward=jnp.zeros((), dtype=float),
        traj_id=jnp.zeros((), dtype=int),
        done=jnp.zeros((), dtype=bool),
        info={"episode_return": 0.0, "episode_length": 0,
              "is_terminal_step": False},
    )
    buffer_fn = fbx.make_trajectory_buffer(
        add_batch_size=config.arch.total_num_envs,
        sample_batch_size=config.system.total_batch_size,
        sample_sequence_length=config.system.rollout_length,
        max_length_time_axis=config.system.rollout_length,
        min_length_time_axis=config.system.rollout_length,
        period=config.system.rollout_length,
    )

    vmap_flatten_crl_fn = jax.vmap(flatten_crl_fn, in_axes=(None, None, 0, 0))

    buffer_fns = (buffer_fn.add, buffer_fn.sample, vmap_flatten_crl_fn)
    buffer_states = buffer_fn.init(dummy_transition)

    # Initialise actor params and optimiser state.
    sa_params = sa_network.init(sa_key, init_obs, init_action)

    # Initialise critic params and optimiser state.
    g_params = g_network.init(g_key, init_goal)

    params = ContrastiveParams(sa_params, g_params)
    opt_state = optim.init(params)

    # Pack params.

    sa_network_apply_fn = sa_network.apply
    g_network_apply_fn = g_network.apply

    actions = jnp.arange(num_actions)[..., None]
    actions = jnp.broadcast_to(
        actions, (num_actions, config.arch.num_envs))
    # print(f"broadcast actions: {actions.shape}")

    @jax.jit
    def policy_apply_fn(params, observation, goal, seed):
        sa_params, g_params = params
        vmap_sa = jax.vmap(sa_network_apply_fn, in_axes=(
            None, None, 0))
        sa_repr = vmap_sa(sa_params, observation, actions)
        g_repr = g_network_apply_fn(g_params, goal)
        print(f"sa output repr {sa_repr}")
        print(f"goal output repr {g_repr}")
        logits = jnp.einsum("ijk,jk->ji", sa_repr, g_repr)
        print(f"logits: {logits}")
        return distrax.EpsilonGreedy(preferences=logits, epsilon=.9).sample(seed=seed)

    # Pack apply and update functions.
    apply_fns = (sa_network_apply_fn, g_network_apply_fn, policy_apply_fn)
    update_fns = optim.update

    learn = get_learner_fn(config, env, apply_fns, update_fns, buffer_fns)
    learn = jax.pmap(learn, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )
    #

    def reshape_states(x): return x.reshape(
        (n_devices, config.arch.update_batch_size,
         config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)
    #
    # # Load model from checkpoint if specified.
    # if config.logger.checkpointing.load_model:
    #     loaded_checkpoint = Checkpointer(
    #         model_name=config.system.system_name,
    #         **config.logger.checkpointing.load_args,  # Other checkpoint args
    #     )
    #     # Restore the learner state from the checkpoint
    #     restored_params, _ = loaded_checkpoint.restore_params()
    #     # Update the params
    #     params = restored_params
    #
    # Define params to be replicated across devices and batches.
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(
        step_key, n_devices * config.arch.update_batch_size)
    #

    def reshape_keys(x): return x.reshape(
        (n_devices, config.arch.update_batch_size) + x.shape[1:])
    step_keys = reshape_keys(jnp.stack(step_keys))
    #
    replicate_learner = (params, opt_state, buffer_states)

    # Duplicate learner for update_batch_size.
    def broadcast(x): return jnp.broadcast_to(
        x, (config.arch.update_batch_size,) + x.shape)
    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(
        replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states, buffer_states = replicate_learner
    init_learner_state = OffPolicyLearnerState(
        params, opt_states, buffer_states, step_keys, env_states, timesteps)

    return learn, init_learner_state, policy_apply_fn


def train(_config):
    config = copy.deepcopy(_config)
    # Calculate total timesteps.
    n_devices = len(jax.devices())
    config.num_devices = n_devices
    config = check_total_timesteps(config)
    assert (
        config.arch.num_updates >= config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    def make_gcrl_env(name):
        env = nx.make(name)
        env = NavixWrapper(env)
        env = FlattenObservationWrapper(env)
        env = NavixGoalWrapper(env)
        env = RecordEpisodeMetrics(env)
        env = TrajectoryIdWrapper(env)
        return env

    env = make_gcrl_env("Navix-Empty-5x5-v0")
    eval_env = make_gcrl_env("Navix-Empty-5x5-v0")

    key = jax.random.PRNGKey(0)
    key_e, *keys = jax.random.split(key, 4)
    learn, learner_state, policy_fn = learner_setup(env, keys, config)

    # Setup evaluator.
    # evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
    #     eval_env=eval_env,
    #     key_e=key_e,
    #     eval_act_fn=get_distribution_act_fn(config, policy_fn),
    #     params=learner_state.params.actor_params,
    #     config=config,
    # )
    config.arch.num_updates_per_eval = config.arch.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.arch.num_updates_per_eval
        * config.system.rollout_length
        * config.arch.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup
    logger = StoixLogger(config)
    cfg: Dict = OmegaConf.to_container(config, resolve=True)
    cfg["arch"]["devices"] = jax.devices()
    pprint(cfg)

    # for eval_step in range(1):
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(32 * (eval_step + 1))

        episode_metrics, ep_completed = get_final_step_metrics(
            learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = 32 / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        train_metrics = learner_output.train_metrics
        # Calculate the number of optimiser steps per second. Since gradients are aggregated
        # across the device and batch axis, we don't consider updates per device/batch as part of
        # the SPS for the learner.
        opt_steps_per_eval = config.arch.num_updates_per_eval
        train_metrics["steps_per_second"] = opt_steps_per_eval / elapsed_time
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        start_time = time.time()
        trained_sa_params = unreplicate_batch_dim(
            learner_output.learner_state.params.sa_params
        )  # Select only actor params
        trained_g_params = unreplicate_batch_dim(
            learner_output.learner_state.params.g_params
        )  # Select only actor params
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        # Evaluate.
        # evaluator_output = evaluator(trained_params, eval_keys)
        # jax.block_until_ready(evaluator_output)

        # Separately log timesteps, actoring metrics and training metrics.
        learner_state = learner_output.learner_state


@hydra.main(
    config_path="../../configs/default/anakin",
    config_name="default_ff_gcrl.yaml",
    version_base="1.2",
)
def main(config: DictConfig) -> float:

    OmegaConf.set_struct(config, False)
    train(config)
    # state, timestep = env.reset(key)
    # _, _, policy = apply_fns
    # action = policy(params, timestep.observation, timestep.extras["goal"])
    # print(action)
    # get_learner_fn(env, apply_fns, )
    # for i in range(10):
    #     _key, key = jax.random.split(key)
    #     state, timestep = env.reset(_key)
    #     print(jnp.squeeze(
    #         state.navix_state.state.entities[Entities.GOAL].position))
    #     print(jnp.squeeze(timestep.extras["goal"]))
    #     print(env.goal_spec().generate_value())

    return 0


if __name__ == "__main__":
    main()
