from typing import Any, Dict, Optional, Tuple, Union

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment
from omegaconf import DictConfig

from stoix.base_types import (
    ActFn,
    ActorApply,
    EvalFn,
    EvalState,
    EvaluationOutput,
    GoalActFn,
    ObstActFn,
    RecActFn,
)
from stoix.utils.env_factory import EnvFactory
from stoix.utils.jax_utils import unreplicate_batch_dim


def get_penalty(logits: chex.Array):
    max = jnp.max(logits)

    def penalty(val: chex.Array):
        return jnp.log(1 - val / max)

    vmap_penalty = jax.vmap(penalty, in_axes=(0))
    return vmap_penalty(logits)


def obstacle_act_fn(
    config: DictConfig,
    actor_apply: ActorApply,
    rngs: Optional[Dict[str, chex.PRNGKey]] = None,
) -> ObstActFn:
    """Get the act_fn for a network that returns a distribution."""

    def act_fn(
        params: FrozenDict,
        observation: chex.Array,
        goal: chex.Array,
        obstacles: chex.Array,
        key: chex.PRNGKey,
    ) -> chex.Array:
        """Get the action from the distribution."""
        # TODO: config.penalty_fn
        vmap_obst = jax.vmap(actor_apply, in_axes=(None, None, 0))
        obstacle_logits = vmap_obst(
            params,
            observation,
            obstacles,
        ).preferences
        vmap_penalty = jax.vmap(get_penalty, in_axes=(0))
        penalties = vmap_penalty(obstacle_logits)
        penalty = jnp.mean(penalties, axis=0)

        logits = actor_apply(params, observation, goal).preferences - 0.01 * penalty
        # logits = actor_apply(params, observation, goal).preferences
        action = jnp.argmax(logits)
        return action

    return act_fn


def get_obst_evaluator_fn(
    env: Environment,
    act_fn: GoalActFn,
    config: DictConfig,
    log_solve_rate: bool = False,
    eval_multiplier: int = 1,
) -> EvalFn:
    """Get the evaluator function for feedforward networks.

    Args:
        env (Environment): An environment instance for evaluation.
        act_fn (callable): The act_fn that returns the action taken by the agent.
        config (dict): Experiment configuration.
        eval_multiplier (int): A scalar that will increase the number of evaluation
            episodes by a fixed factor. The reason for the increase is to enable the
            computation of the `absolute metric` which is a metric computed and the end
            of training by rolling out the policy which obtained the greatest evaluation
            performance during training for 10 times more episodes than were used at a
            single evaluation step.
    """

    def eval_one_episode(params: FrozenDict, init_eval_state: EvalState) -> Dict:
        """Evaluate one episode. It is vectorized over the number of evaluation episodes."""

        def not_done(carry: Tuple) -> bool:
            """Check if the episode is done."""
            timestep = carry[2]
            is_not_done: bool = ~timestep.last()
            return is_not_done

        def _obst_env_step(eval_state: EvalState) -> EvalState:
            """Step the environment."""
            # PRNG keys.
            key, env_state, last_timestep, step_count, episode_return = eval_state

            # Select action.
            key, policy_key = jax.random.split(key)

            action = act_fn(
                params,
                jax.tree_util.tree_map(
                    lambda x: x[jnp.newaxis, ...],
                    last_timestep.observation,
                ),
                last_timestep.extras["goal"][jnp.newaxis, ...],
                last_timestep.extras["obstacles"][:, jnp.newaxis],
                policy_key,
            )

            # Step environment.
            env_state, timestep = env.step(env_state, action.squeeze())

            # Log episode metrics.
            episode_return += timestep.reward
            step_count += 1
            eval_state = EvalState(key, env_state, timestep, step_count, episode_return)
            return eval_state

        final_state = jax.lax.while_loop(not_done, _obst_env_step, init_eval_state)

        eval_metrics = {
            "episode_return": final_state.episode_return,
            "episode_length": final_state.step_count,
        }
        # Log solve episode if solve rate is required.
        if log_solve_rate:
            eval_metrics["solve_episode"] = jnp.all(
                final_state.episode_return >= config.env.solved_return_threshold
            ).astype(int)

        return eval_metrics

    def evaluator_fn(trained_params: FrozenDict, key: chex.PRNGKey) -> EvaluationOutput[EvalState]:
        """Evaluator function."""

        # Initialise environment states and timesteps.
        n_devices = len(jax.devices())

        eval_batch = (config.arch.num_eval_episodes // n_devices) * eval_multiplier

        key, *env_keys = jax.random.split(key, eval_batch + 1)
        env_states, timesteps = jax.vmap(env.reset)(
            jnp.stack(env_keys),
        )
        # Split keys for each core.
        key, *step_keys = jax.random.split(key, eval_batch + 1)
        # Add dimension to pmap over.
        step_keys = jnp.stack(step_keys).reshape(eval_batch, -1)

        eval_state = EvalState(
            key=step_keys,
            env_state=env_states,
            timestep=timesteps,
            step_count=jnp.zeros((eval_batch, 1)),
            episode_return=jnp.zeros_like(timesteps.reward),
        )

        eval_metrics = jax.vmap(
            eval_one_episode,
            in_axes=(None, 0),
            axis_name="eval_batch",
        )(trained_params, eval_state)

        return EvaluationOutput(
            learner_state=eval_state,
            episode_metrics=eval_metrics,
        )

    return evaluator_fn


def evaluator_setup(
    eval_env: Environment,
    key_e: chex.PRNGKey,
    eval_act_fn: Union[ActFn, RecActFn, GoalActFn, ObstActFn],
    params: FrozenDict,
    config: DictConfig,
) -> Tuple[EvalFn, EvalFn, Tuple[FrozenDict, chex.Array]]:
    """Initialise evaluator_fn."""
    # Get available TPU cores.
    n_devices = len(jax.devices())
    # Check if solve rate is required for evaluation.
    if hasattr(config.env, "solved_return_threshold"):
        log_solve_rate = True
    else:
        log_solve_rate = False

    get_evaluator_fn = get_obst_evaluator_fn
    evaluator = get_evaluator_fn(eval_env, eval_act_fn, config, log_solve_rate)  # type: ignore

    absolute_metric_evaluator = get_evaluator_fn(
        eval_env,
        eval_act_fn,  # type: ignore
        config,
        log_solve_rate,
        10,
    )

    evaluator = jax.pmap(evaluator, axis_name="device")
    absolute_metric_evaluator = jax.pmap(absolute_metric_evaluator, axis_name="device")

    # Broadcast trained params to cores and split keys for each core.
    trained_params = unreplicate_batch_dim(params)
    key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
    eval_keys = jnp.stack(eval_keys).reshape(n_devices, -1)

    return evaluator, absolute_metric_evaluator, (trained_params, eval_keys)
