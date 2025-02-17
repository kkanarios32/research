import navix as nx
from navix.entities import Entities
from stoix.wrappers.navix import NavixWrapper, NavixGoalWrapper
from stoix.systems.gcrl.envs.random_goals import RandomGoals
import jax
import jax.numpy as jnp
from jumanji.env import Environment
import chex
import copy
import time
from typing import Any, Dict, Tuple

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
    OnPolicyLearnerState,
)
from stoix.evaluator import evaluator_setup, get_distribution_act_fn
from stoix.networks.base import ContrastiveNet as Net
from stoix.networks.torso import ContrastiveTorso as Torso
from stoix.networks.inputs import ObservationActionInput, EmbeddingInput
from stoix.systems.gcrl.gcrl_types import Transition
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.multistep import batch_discounted_returns
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate
from stoix.wrappers.episode_metrics import get_final_step_metrics
from stoix.wrappers.transforms import FlattenObservationWrapper


def get_learner_fn(
    env: Environment,
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[OnPolicyLearnerState]:
    """Get the learner function."""

    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(
        learner_state: OnPolicyLearnerState, _: Any
    ) -> Tuple[OnPolicyLearnerState, Tuple]:
        def _env_step(
            learner_state: OnPolicyLearnerState, _: Any
        ) -> Tuple[OnPolicyLearnerState, Transition]:
            """Step the environment."""
            params, opt_states, key, env_state, last_timestep = learner_state

            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy = actor_apply_fn(
                params.actor_params, last_timestep.observation)
            value = critic_apply_fn(
                params.critic_params, last_timestep.observation)
            action = actor_policy.sample(seed=policy_key)

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(
                env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = timestep.last().reshape(-1)
            goal = timestep.extras["goal"]
            info = timestep.extras["episode_metrics"]

            transition = Transition(
                done, action, value, timestep.reward, timestep.observation, goal, info
            )

            learner_state = OnPolicyLearnerState(
                params, opt_states, key, env_state, timestep)
            return learner_state, transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # CALCULATE ADVANTAGE
        params, opt_states, key, env_state, last_timestep = learner_state
        last_val = critic_apply_fn(
            params.critic_params, last_timestep.observation)
        # Swap the batch and time axes.
        traj_batch = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), traj_batch)

        r_t = traj_batch.reward
        v_t = jnp.concatenate(
            [traj_batch.value, last_val[..., jnp.newaxis]], axis=-1)[:, 1:]
        d_t = 1.0 - traj_batch.done.astype(jnp.float32)
        d_t = (d_t * config.system.gamma).astype(jnp.float32)
        monte_carlo_returns = batch_discounted_returns(
            r_t, d_t, v_t, True, False)

        def _actor_loss_fn(
            actor_params: FrozenDict,
            observations: chex.Array,
            actions: chex.Array,
            monte_carlo_returns: chex.Array,
            value_predictions: chex.Array,
        ) -> Tuple:
            """Calculate the actor loss."""
            # RERUN NETWORK
            actor_policy = actor_apply_fn(actor_params, observations)
            log_prob = actor_policy.log_prob(actions)
            advantage = monte_carlo_returns - value_predictions
            # CALCULATE ACTOR LOSS
            loss_actor = -advantage * log_prob
            entropy = actor_policy.entropy().mean()

            total_loss_actor = loss_actor.mean() - config.system.ent_coef * entropy
            loss_info = {
                "actor_loss": loss_actor,
                "entropy": entropy,
            }
            return total_loss_actor, loss_info

        def _critic_loss_fn(
            critic_params: FrozenDict,
            observations: chex.Array,
            targets: chex.Array,
        ) -> Tuple:
            """Calculate the critic loss."""
            # RERUN NETWORK
            value = critic_apply_fn(critic_params, observations)

            # CALCULATE VALUE LOSS
            value_loss = rlax.l2_loss(value, targets).mean()

            critic_total_loss = config.system.vf_coef * value_loss
            loss_info = {
                "value_loss": value_loss,
            }
            return critic_total_loss, loss_info

        # CALCULATE ACTOR LOSS
        actor_grad_fn = jax.grad(_actor_loss_fn, has_aux=True)
        actor_grads, actor_loss_info = actor_grad_fn(
            params.actor_params,
            traj_batch.obs,
            traj_batch.action,
            monte_carlo_returns,
            traj_batch.value,
        )

        # CALCULATE CRITIC LOSS
        critic_grad_fn = jax.grad(_critic_loss_fn, has_aux=True)
        critic_grads, critic_loss_info = critic_grad_fn(
            params.critic_params, traj_batch.obs, monte_carlo_returns
        )

        # Compute the parallel mean (pmean) over the batch.
        # This calculation is inspired by the Anakin architecture demo notebook.
        # available at https://tinyurl.com/26tdzs5x
        # This pmean could be a regular mean as the batch axis is on the same device.
        actor_grads, actor_loss_info = jax.lax.pmean(
            (actor_grads, actor_loss_info), axis_name="batch"
        )
        # pmean over devices.
        actor_grads, actor_loss_info = jax.lax.pmean(
            (actor_grads, actor_loss_info), axis_name="device"
        )

        critic_grads, critic_loss_info = jax.lax.pmean(
            (critic_grads, critic_loss_info), axis_name="batch"
        )
        # pmean over devices.
        critic_grads, critic_loss_info = jax.lax.pmean(
            (critic_grads, critic_loss_info), axis_name="device"
        )

        # UPDATE ACTOR PARAMS AND OPTIMISER STATE
        actor_updates, actor_new_opt_state = actor_update_fn(
            actor_grads, opt_states.actor_opt_state
        )
        actor_new_params = optax.apply_updates(
            params.actor_params, actor_updates)

        # UPDATE CRITIC PARAMS AND OPTIMISER STATE
        critic_updates, critic_new_opt_state = critic_update_fn(
            critic_grads, opt_states.critic_opt_state
        )
        critic_new_params = optax.apply_updates(
            params.critic_params, critic_updates)

        # PACK NEW PARAMS AND OPTIMISER STATE
        new_params = ActorCriticParams(actor_new_params, critic_new_params)
        new_opt_state = ActorCriticOptStates(
            actor_new_opt_state, critic_new_opt_state)

        # PACK LOSS INFO
        loss_info = {
            **actor_loss_info,
            **critic_loss_info,
        }

        learner_state = OnPolicyLearnerState(
            new_params, new_opt_state, key, env_state, last_timestep
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: OnPolicyLearnerState,
    ) -> AnakinExperimentOutput[OnPolicyLearnerState]:

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
    init_action = jnp.zeros((1, num_actions))

    # PRNG keys.
    key, sa_key, g_key = keys

    sa_network = Net(torso=Torso(
        16), input_layer=ObservationActionInput())
    g_network = Net(torso=Torso(16), input_layer=EmbeddingInput())

    sa_lr = make_learning_rate(config.system.actor_lr, config, 1, 1)
    g_lr = make_learning_rate(config.system.critic_lr, config, 1, 1)

    sa_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(sa_lr, eps=1e-5),
    )
    g_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(g_lr, eps=1e-5),
    )

    # Initialise observation
    init_obs = env.observation_spec().generate_value()
    init_obs = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)

    init_goal = env.goal_spec().generate_value()
    init_goal = jax.tree_util.tree_map(lambda x: x[None, ...], init_goal)

    # Initialise actor params and optimiser state.
    sa_params = sa_network.init(sa_key, *(init_obs, init_action))
    sa_opt_state = sa_optim.init(sa_params)

    # Initialise critic params and optimiser state.
    g_params = g_network.init(g_key, init_goal)
    g_opt_state = g_optim.init(g_params)

    # Pack params.
    params = (sa_params, g_params)

    sa_network_apply_fn = sa_network.apply
    g_network_apply_fn = g_network.apply

    # Pack apply and update functions.
    apply_fns = (sa_network_apply_fn, g_network_apply_fn)

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )

    def reshape_states(x): return x.reshape(
        (n_devices, config.arch.update_batch_size,
         config.arch.num_envs) + x.shape[1:]
    )
    # # (devices, update batch size, num_envs, ...)
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params()
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(
        step_key, n_devices * config.arch.update_batch_size)

    def reshape_keys(x): return x.reshape(
        (n_devices, config.arch.update_batch_size) + x.shape[1:])
    step_keys = reshape_keys(jnp.stack(step_keys))

    opt_states = (sa_opt_state, g_opt_state)
    replicate_learner = (params, opt_states)

    # Duplicate learner for update_batch_size.
    def broadcast(x): return jnp.broadcast_to(
        x, (config.arch.update_batch_size,) + x.shape)
    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(
        replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states = replicate_learner
    init_learner_state = OnPolicyLearnerState(
        params, opt_states, step_keys, env_states, timesteps)

    return apply_fns, params


def test() -> float:
    """Runs experiment."""

    # Create the environments for train and eval.
    env = nx.make("Navix-Random-Goals-6x6-v0")
    env = NavixWrapper(env)
    env = FlattenObservationWrapper(env)
    env = NavixGoalWrapper(env)
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 3)
    apply_fns, params = learner_setup(env, keys)
    state, timestep = env.reset(key)
    for i in range(10):
        _key, key = jax.random.split(key)
        state, timestep = env.reset(_key)
        print(jnp.squeeze(
            state.navix_state.state.entities[Entities.GOAL].position))
        print(jnp.squeeze(timestep.extras["goal"]))
        print(env.goal_spec().generate_value())

    return 0


if __name__ == "__main__":
    test()
