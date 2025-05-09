import copy
import functools
import time
from typing import Any, Dict, Tuple

import chex
import hydra
import jax
import jax.numpy as jnp
import mctx
import navix as nx
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment
from jumanji.wrappers import AutoResetWrapper
from navix import observations, rewards
from navix.environments import DynamicObstacles
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from stoix.base_types import (
    LearnerFn,
    OnlineAndTarget,
)
from stoix.networks.base import FeedForwardActor as Actor
from stoix.networks.inputs import ObservationGoalInput
from stoix.systems.gcrl.gcrl_types import SimState
from stoix.systems.gcrl.search.evaluator import search_evaluator_setup
from stoix.systems.gcrl.search.mcts.base import RecurrentFnOutput, RootFnOutput
from stoix.systems.search.search_types import EnvironmentStep, RootFnApply, SearchApply
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.wrappers.episode_metrics import RecordEpisodeMetrics, get_final_step_metrics
from stoix.wrappers.navix import NavixGoalWrapper, NavixWrapper
from stoix.wrappers.transforms import FlattenObservationWrapper


def make_root_fn() -> RootFnApply:
    def root_fn(
        observation: chex.ArrayTree,
        goal: chex.ArrayTree,
        state_embedding: SimState,
        seed: chex.PRNGKey,
    ) -> mctx.RootFnOutput:
        root_fn_output = RootFnOutput(
            prior_logits=jnp.zeros(),
            obstacle_logits=jnp.zeros_like(values),
            value=jnp.zeros(1),
            embedding=state_embedding,
        )

        return root_fn_output

    return root_fn


def make_recurrent_fn(
    environment_step: EnvironmentStep,
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

        recurrent_fn_output = RecurrentFnOutput(
            reward=next_timestep.reward,
            discount=next_timestep.discount * config.system.gamma,
            prior_logits=jnp.zeros(num_actions),
            obstacle_logits=jnp.zeros(num_actions),
            value=jnp.zeros(1),
        )

        return recurrent_fn_output, next_state_embedding

    return recurrent_fn


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

    # Initialise observation
    init_x = env.observation_spec().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

    init_goal = env.goal_spec().generate_value()
    init_goal = jax.tree_util.tree_map(lambda x: x[None, ...], init_goal)

    # Initialise q params and optimiser state.
    q_online_params = q_network.init(q_net_key, init_x, init_goal)
    q_target_params = q_online_params

    params = OnlineAndTarget(q_online_params, q_target_params)

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

    return root_fn, search_apply_fn


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
