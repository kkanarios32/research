import copy
import functools
import time
from typing import Any, Callable, Dict, Tuple

import chex
import flashbax as fbx
import flax
import hydra
import jax
import jax.numpy as jnp
import mctx
import navix as nx
import optax
from colorama import Fore, Style
from flashbax.buffers.trajectory_buffer import BufferState
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment
from jumanji.types import TimeStep
from jumanji.wrappers import AutoResetWrapper
from navix import observations, rewards
from navix.environments import DynamicObstacles
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from stoix.base_types import (
    ActorApply,
    AnakinExperimentOutput,
    GoalActFn,
    LearnerFn,
    LogEnvState,
    OffPolicyLearnerState,
    OnlineAndTarget,
)
from stoix.networks.base import FeedForwardActor as Actor
from stoix.networks.inputs import ObservationGoalInput
from stoix.systems.gcrl.gcrl_types import QLearningTransition as Transition
from stoix.systems.gcrl.gcrl_types import SimState
from stoix.systems.gcrl.search.evaluator import search_evaluator_setup
from stoix.systems.gcrl.search.mcts.base import RecurrentFnOutput, RootFnOutput
from stoix.systems.search.search_types import EnvironmentStep, RootFnApply, SearchApply
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.loss import q_learning
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate
from stoix.wrappers.episode_metrics import RecordEpisodeMetrics, get_final_step_metrics
from stoix.wrappers.navix import NavixGoalWrapper, NavixWrapper
from stoix.wrappers.transforms import FlattenObservationWrapper


def make_root_fn(q_apply_fn: GoalActFn) -> RootFnApply:
    def root_fn(
        params: FrozenDict,
        observation: chex.ArrayTree,
        goal: chex.ArrayTree,
        state_embedding: SimState,
        seed: chex.PRNGKey,
    ) -> mctx.RootFnOutput:
        values = q_apply_fn(params, observation, goal).preferences
        value = jnp.max(values, axis=-1)

        # root_fn_output = RootFnOutput(
        #     prior_logits=values,
        #     obstacle_logits=jnp.zeros_like(values),
        #     value=value,
        #     embedding=state_embedding,
        # )

        root_fn_output = RootFnOutput(
            prior_logits=jnp.ones_like(values),
            obstacle_logits=jnp.zeros_like(values),
            value=jnp.zeros(1),
            embedding=state_embedding,
        )

        return root_fn_output

    return root_fn


def make_recurrent_fn(
    environment_step: EnvironmentStep,
    q_apply_fn: ActorApply,
    config: DictConfig,
) -> mctx.RecurrentFn:
    vmap_obs_q = jax.vmap(q_apply_fn, in_axes=(None, None, 1))

    def recurrent_fn(
        params: FrozenDict,
        _: chex.PRNGKey,  # Unused key
        action: chex.Array,
        state_embedding: SimState,
    ) -> Tuple[RecurrentFnOutput, Any]:
        next_state_embedding, next_timestep = environment_step(state_embedding, action)
        goal = next_timestep.extras["goal"]
        obstacles = next_timestep.extras["obstacles"]
        obs = next_timestep.observation

        values = q_apply_fn(params, next_timestep.observation, goal).preferences
        value = jnp.max(values, axis=-1)
        o_values = vmap_obs_q(params, obs, obstacles).preferences[action]
        o_values = jnp.max(o_values, axis=0)

        # recurrent_fn_output = RecurrentFnOutput(
        #     reward=next_timestep.reward,
        #     discount=next_timestep.discount * config.system.gamma,
        #     prior_logits=values,
        #     obstacle_logits=o_values,
        #     value=value,
        # )

        recurrent_fn_output = RecurrentFnOutput(
            reward=next_timestep.reward,
            discount=next_timestep.discount * config.system.gamma,
            prior_logits=jnp.ones_like(values),
            obstacle_logits=jnp.zeros_like(values),
            value=jnp.zeros(1),
        )

        return recurrent_fn_output, next_state_embedding

    return recurrent_fn


def get_warmup_fn(
    env: Environment,
    q_params: FrozenDict,
    q_apply_fn: ActorApply,
    buffer_add_fn: Callable,
    config: DictConfig,
) -> Callable:
    def warmup(
        env_states: LogEnvState,
        timesteps: TimeStep,
        buffer_states: BufferState,
        keys: chex.PRNGKey,
    ) -> Tuple[LogEnvState, TimeStep, BufferState, chex.PRNGKey]:
        def _env_step(
            carry: Tuple[LogEnvState, TimeStep, chex.PRNGKey], _: Any
        ) -> Tuple[Tuple[LogEnvState, TimeStep, chex.PRNGKey], Transition]:
            """Step the environment."""

            env_state, last_timestep, key = carry
            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy = q_apply_fn(
                q_params.online,
                last_timestep.observation,
                last_timestep.extras["goal"],
            )
            action = actor_policy.sample(seed=policy_key)

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = timestep.last().reshape(-1)
            info = timestep.extras["episode_metrics"]
            next_obs = timestep.extras["next_obs"]
            goal = timestep.extras["goal"]

            transition = Transition(
                done,
                action,
                timestep.reward,
                last_timestep.observation,
                next_obs,
                goal,
                info,
            )

            return (env_state, timestep, key), transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        (env_states, timesteps, keys), traj_batch = jax.lax.scan(
            _env_step, (env_states, timesteps, keys), None, config.system.warmup_steps
        )

        # Add the trajectory to the buffer.
        buffer_states = buffer_add_fn(buffer_states, traj_batch)

        return env_states, timesteps, keys, buffer_states

    batched_warmup_step: Callable = jax.vmap(
        warmup, in_axes=(0, 0, 0, 0), out_axes=(0, 0, 0, 0), axis_name="batch"
    )

    return batched_warmup_step


def get_learner_fn(
    env: Environment,
    q_apply_fn: ActorApply,
    q_update_fn: optax.TransformUpdateFn,
    buffer_fns: Tuple[Callable, Callable],
    config: DictConfig,
) -> LearnerFn[OffPolicyLearnerState]:
    """Get the learner function."""

    buffer_add_fn, buffer_sample_fn = buffer_fns

    def _update_step(
        learner_state: OffPolicyLearnerState, _: Any
    ) -> Tuple[OffPolicyLearnerState, Tuple]:
        def _env_step(
            learner_state: OffPolicyLearnerState, _: Any
        ) -> Tuple[OffPolicyLearnerState, Transition]:
            """Step the environment."""
            q_params, opt_states, buffer_state, key, env_state, last_timestep = (
                learner_state
            )

            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy = q_apply_fn(
                q_params.online, last_timestep.observation, last_timestep.extras["goal"]
            )
            action = actor_policy.sample(seed=policy_key)

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done = timestep.last().reshape(-1)
            info = timestep.extras["episode_metrics"]
            next_obs = timestep.extras["next_obs"]
            goal = timestep.extras["goal"]

            transition = Transition(
                done,
                action,
                timestep.reward,
                last_timestep.observation,
                next_obs,
                goal,
                info,
            )

            learner_state = OffPolicyLearnerState(
                q_params, opt_states, buffer_state, key, env_state, timestep
            )
            return learner_state, transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

        # Add the trajectory to the buffer.
        buffer_state = buffer_add_fn(buffer_state, traj_batch)

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _q_loss_fn(
                q_params: FrozenDict,
                target_q_params: FrozenDict,
                transitions: Transition,
            ) -> jnp.ndarray:
                q_tm1 = q_apply_fn(
                    q_params, transitions.obs, transitions.goal
                ).preferences
                q_t = q_apply_fn(
                    target_q_params, transitions.next_obs, transitions.goal
                ).preferences

                # Cast and clip rewards.
                discount = 1.0 - transitions.done.astype(jnp.float32)
                d_t = (discount * config.system.gamma).astype(jnp.float32)
                r_t = jnp.clip(
                    transitions.reward,
                    -config.system.max_abs_reward,
                    config.system.max_abs_reward,
                ).astype(jnp.float32)
                a_tm1 = transitions.action

                # Compute Q-learning loss.
                batch_loss = q_learning(
                    q_tm1,
                    a_tm1,
                    r_t,
                    d_t,
                    q_t,
                    config.system.huber_loss_parameter,
                )

                loss_info = {
                    "q_loss": batch_loss,
                }

                return batch_loss, loss_info

            params, opt_states, buffer_state, key = update_state

            key, sample_key = jax.random.split(key)

            # SAMPLE TRANSITIONS
            transition_sample = buffer_sample_fn(buffer_state, sample_key)
            transitions: Transition = transition_sample.experience

            # CALCULATE Q LOSS
            q_grad_fn = jax.grad(_q_loss_fn, has_aux=True)
            q_grads, q_loss_info = q_grad_fn(
                params.online,
                params.target,
                transitions,
            )

            # Compute the parallel mean (pmean) over the batch.
            # This calculation is inspired by the Anakin architecture demo notebook.
            # available at https://tinyurl.com/26tdzs5x
            # This pmean could be a regular mean as the batch axis is on the same device.
            q_grads, q_loss_info = jax.lax.pmean(
                (q_grads, q_loss_info), axis_name="batch"
            )
            q_grads, q_loss_info = jax.lax.pmean(
                (q_grads, q_loss_info), axis_name="device"
            )

            # UPDATE Q PARAMS AND OPTIMISER STATE
            q_updates, q_new_opt_state = q_update_fn(q_grads, opt_states)
            q_new_online_params = optax.apply_updates(params.online, q_updates)
            # Target network polyak update.
            new_target_q_params = optax.incremental_update(
                q_new_online_params, params.target, config.system.tau
            )
            q_new_params = OnlineAndTarget(q_new_online_params, new_target_q_params)

            # PACK NEW PARAMS AND OPTIMISER STATE
            new_params = q_new_params
            new_opt_state = q_new_opt_state

            # PACK LOSS INFO
            loss_info = {
                **q_loss_info,
            }
            return (new_params, new_opt_state, buffer_state, key), loss_info

        update_state = (params, opt_states, buffer_state, key)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        params, opt_states, buffer_state, key = update_state
        learner_state = OffPolicyLearnerState(
            params, opt_states, buffer_state, key, env_state, last_timestep
        )
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: OffPolicyLearnerState,
    ) -> AnakinExperimentOutput[OffPolicyLearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        """

        batched_update_step = jax.vmap(
            _update_step, in_axes=(0, None), axis_name="batch"
        )

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
    env: Environment, keys: chex.Array, config: DictConfig, model_env: Environment
) -> Tuple[LearnerFn[Any], RootFnApply, SearchApply, Any]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of actions.
    action_dim = int(env.action_spec().num_values)
    config.system.action_dim = action_dim

    # PRNG keys.
    key, q_net_key = keys

    # Define networks and optimiser.
    q_network_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    q_network_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        action_dim=action_dim,
        epsilon=config.system.training_epsilon,
    )

    q_network = Actor(
        torso=q_network_torso,
        action_head=q_network_action_head,
        input_layer=ObservationGoalInput(),
    )

    eval_q_network_action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head,
        action_dim=action_dim,
        epsilon=config.system.evaluation_epsilon,
    )
    eval_q_network = Actor(
        torso=q_network_torso,
        action_head=eval_q_network_action_head,
        input_layer=ObservationGoalInput(),
    )

    q_lr = make_learning_rate(config.system.q_lr, config, config.system.epochs)
    q_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(q_lr, eps=1e-5),
    )

    # Initialise observation
    init_x = env.observation_spec().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

    init_goal = env.goal_spec().generate_value()
    init_goal = jax.tree_util.tree_map(lambda x: x[None, ...], init_goal)

    # Initialise q params and optimiser state.
    q_online_params = q_network.init(q_net_key, init_x, init_goal)
    q_target_params = q_online_params
    q_opt_state = q_optim.init(q_online_params)

    params = OnlineAndTarget(q_online_params, q_target_params)
    opt_states = q_opt_state

    q_network_apply_fn = q_network.apply

    root_fn = make_root_fn(q_network_apply_fn)
    environment_model_step = jax.vmap(model_env.step)
    model_recurrent_fn = make_recurrent_fn(
        environment_model_step, q_network_apply_fn, config
    )
    # search_method = safe_policy
    search_method = mctx.gumbel_muzero_policy
    # search_method = mctx.muzero_policy
    search_apply_fn = functools.partial(
        search_method,
        recurrent_fn=model_recurrent_fn,
        num_simulations=config.system.num_simulations,
        max_depth=config.system.max_depth,
        **config.system.search_method_kwargs,
    )

    # Pack apply and update functions.
    apply_fns = q_network_apply_fn
    update_fns = q_optim.update

    # Create replay buffer
    dummy_transition = Transition(
        obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
        action=jnp.zeros((), dtype=int),
        reward=jnp.zeros((), dtype=float),
        done=jnp.zeros((), dtype=bool),
        next_obs=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_x),
        goal=jax.tree_util.tree_map(lambda x: x.squeeze(0), init_goal),
        info={"episode_return": 0.0, "episode_length": 0, "is_terminal_step": False},
    )
    assert config.system.total_buffer_size % n_devices == 0, (
        f"{Fore.RED}{Style.BRIGHT}The total buffer size should be divisible "
        + "by the number of devices!{Style.RESET_ALL}"
    )
    assert config.system.total_batch_size % n_devices == 0, (
        f"{Fore.RED}{Style.BRIGHT}The total batch size should be divisible "
        + "by the number of devices!{Style.RESET_ALL}"
    )
    config.system.buffer_size = config.system.total_buffer_size // (
        n_devices * config.arch.update_batch_size
    )
    config.system.batch_size = config.system.total_batch_size // (
        n_devices * config.arch.update_batch_size
    )
    buffer_fn = fbx.make_item_buffer(
        max_length=config.system.buffer_size,
        min_length=config.system.batch_size,
        sample_batch_size=config.system.batch_size,
        add_batches=True,
        add_sequences=True,
    )
    buffer_fns = (buffer_fn.add, buffer_fn.sample)
    buffer_states = buffer_fn.init(dummy_transition)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, buffer_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    warmup = get_warmup_fn(env, params, q_network_apply_fn, buffer_fn.add, config)
    warmup = jax.pmap(warmup, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )

    def reshape_states(x: chex.Array) -> chex.Array:
        return x.reshape(
            (n_devices, config.arch.update_batch_size, config.arch.num_envs)
            + x.shape[1:]
        )

    # (devices, update batch size, num_envs, ...)
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(TParams=OnlineAndTarget)
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    key, step_key, warmup_key = jax.random.split(key, num=3)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    warmup_keys = jax.random.split(
        warmup_key, n_devices * config.arch.update_batch_size
    )

    def reshape_keys(x: chex.Array) -> chex.Array:
        return x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])

    step_keys = reshape_keys(jnp.stack(step_keys))
    warmup_keys = reshape_keys(jnp.stack(warmup_keys))

    replicate_learner = (params, opt_states, buffer_states)

    # Duplicate learner for update_batch_size.
    def broadcast(x: chex.Array) -> chex.Array:
        return jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)

    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(
        replicate_learner, devices=jax.devices()
    )

    # Initialise learner state.
    params, opt_states, buffer_states = replicate_learner
    # Warmup the buffer.
    env_states, timesteps, keys, buffer_states = warmup(
        env_states, timesteps, buffer_states, warmup_keys
    )
    init_learner_state = OffPolicyLearnerState(
        params, opt_states, buffer_states, step_keys, env_states, timesteps
    )

    return learn, root_fn, search_apply_fn, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    # Calculate total timesteps.
    n_devices = len(jax.devices())
    config.num_devices = n_devices
    config = check_total_timesteps(config)
    assert config.arch.num_updates >= config.arch.num_evaluation, (
        "Number of updates per evaluation must be less than total number of updates."
    )

    # Create the environments for train and eval.
    def make_gcrl_env(name):
        env = nx.make(name)
        eval_env = DynamicObstacles
        eval_env = DynamicObstacles.create(
            height=8,
            width=8,
            n_obstacles=4,
            random_start=False,
            observation_fn=observations.symbolic,
            reward_fn=rewards.on_goal_reached,
        )

        env = NavixWrapper(env)
        eval_env = NavixWrapper(eval_env)

        env = NavixGoalWrapper(env)
        eval_env = NavixGoalWrapper(eval_env)

        env = AutoResetWrapper(env, next_obs_in_extras=True)
        env = RecordEpisodeMetrics(env)

        env = FlattenObservationWrapper(env)
        eval_env = FlattenObservationWrapper(eval_env)

        return env, eval_env

    env, eval_env = make_gcrl_env("Navix-Empty-8x8-v0")

    # PRNG keys.
    key, key_e, q_net_key = jax.random.split(
        jax.random.PRNGKey(config.arch.seed), num=3
    )

    # Setup learner.
    learn, root_fn, search_apply_fn, learner_state = learner_setup(
        env, (key, q_net_key), config, eval_env
    )

    # Setup evaluator.
    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = (
        search_evaluator_setup(
            eval_env=eval_env,
            key_e=key_e,
            search_apply_fn=search_apply_fn,
            root_fn=root_fn,
            params=learner_state.params,
            config=config,
        )
    )

    # Calculate number of updates per evaluation.
    config.arch.num_updates_per_eval = (
        config.arch.num_updates // config.arch.num_evaluation
    )
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

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.system.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Run experiment for a total number of evaluations.
    max_episode_return = jnp.float32(-1e6)
    best_params = unreplicate_batch_dim(learner_state.params.online)
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(
            learner_output.episode_metrics
        )
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if (
            ep_completed
        ):  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        train_metrics = learner_output.train_metrics
        # Calculate the number of optimiser steps per second. Since gradients are aggregated
        # across the device and batch axis, we don't consider updates per device/batch as part of
        # the SPS for the learner.
        opt_steps_per_eval = config.arch.num_updates_per_eval * (config.system.epochs)
        train_metrics["steps_per_second"] = opt_steps_per_eval / elapsed_time
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        start_time = time.time()
        trained_params = unreplicate_batch_dim(
            learner_output.learner_state.params.online
        )  # Select only actor params
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        # Evaluate.
        evaluator_output = evaluator(trained_params, eval_keys)
        jax.block_until_ready(evaluator_output)

        # Log the results of the evaluation.
        elapsed_time = time.time() - start_time
        episode_return = jnp.mean(evaluator_output.episode_metrics["episode_return"])
        episode_return = 0

        steps_per_eval = int(
            jnp.sum(evaluator_output.episode_metrics["episode_length"])
        )
        evaluator_output.episode_metrics["steps_per_second"] = (
            steps_per_eval / elapsed_time
        )
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL)

        if save_checkpoint:
            checkpointer.save(
                timestep=int(steps_per_rollout * (eval_step + 1)),
                unreplicated_learner_state=unreplicate_n_dims(
                    learner_output.learner_state
                ),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Measure absolute metric.
    # if config.arch.absolute_metric:
    #     start_time = time.time()
    #
    #     key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
    #     eval_keys = jnp.stack(eval_keys)
    #     eval_keys = eval_keys.reshape(n_devices, -1)
    #
    #     evaluator_output = absolute_metric_evaluator(best_params, eval_keys)
    #     jax.block_until_ready(evaluator_output)
    #
    #     elapsed_time = time.time() - start_time
    #     t = int(steps_per_rollout * (eval_step + 1))
    #     steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
    #     evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
    #     logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()
    # Record the performance for the final evaluation run. If the absolute metric is not
    # calculated, this will be the final evaluation run.
    # eval_performance = float(jnp.mean(evaluator_output.episode_metrics[config.env.eval_metric]))
    eval_performance = 0
    return eval_performance


@hydra.main(
    config_path="../../../configs/default/anakin/",
    config_name="default_mcts_gcrl.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "arch.total_num_envs", 256)
    OmegaConf.update(cfg, "system.total_batch_size", 256)

    # Run experiment.
    eval_performance = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}DQN experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
