from typing import TYPE_CHECKING, Tuple

import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.specs import Array, DiscreteArray, Spec
from jumanji.types import StepType, TimeStep, restart
from jumanji.wrappers import Wrapper
from navix.entities import Entities, Goal, Player
from navix.environments.environment import Environment
from navix.environments.environment import Timestep as NavixState
from navix.grid import random_positions, room

from stoix.base_types import Observation

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass


@dataclass
class NavixEnvState:
    key: chex.PRNGKey
    navix_state: NavixState


class NavixWrapper(Wrapper):
    def __init__(self, env: Environment):
        self._env = env
        self._n_actions = len(self._env.action_set)

    def reset(self, key: chex.PRNGKey) -> Tuple[NavixEnvState, TimeStep]:
        key, key_reset = jax.random.split(key)
        navix_state = self._env.reset(key_reset)
        agent_view = navix_state.observation.astype(float)
        legal_action_mask = jnp.ones((self._n_actions,), dtype=float)
        step_count = navix_state.t.astype(int)
        obs = Observation(agent_view, legal_action_mask, step_count)
        timestep = restart(obs, extras={})
        state = NavixEnvState(key=key, navix_state=navix_state)
        return state, timestep

    def step(self, state: NavixEnvState, action: chex.Array) -> Tuple[NavixEnvState, TimeStep]:
        key, key_step = jax.random.split(state.key)

        navix_state = self._env.step(state.navix_state, action)

        agent_view = navix_state.observation.astype(float)
        legal_action_mask = jnp.ones((self._n_actions,), dtype=float)
        step_count = navix_state.t.astype(int)
        next_obs = Observation(agent_view, legal_action_mask, step_count)

        reward = navix_state.reward.astype(float)
        terminal = navix_state.is_termination()
        truncated = navix_state.is_truncation()

        discount = jnp.array(1.0 - terminal, dtype=float)
        final_step = jnp.logical_or(terminal, truncated)

        timestep = TimeStep(
            observation=next_obs,
            reward=reward,
            discount=discount,
            step_type=jax.lax.select(final_step, StepType.LAST, StepType.MID),
            extras={},
        )
        next_state = NavixEnvState(key=key_step, navix_state=navix_state)
        return next_state, timestep

    def reward_spec(self) -> specs.Array:
        return specs.Array(shape=(), dtype=float, name="reward")

    def discount_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(shape=(), dtype=float, minimum=0.0, maximum=1.0, name="discount")

    def action_spec(self) -> Spec:
        return DiscreteArray(num_values=self._n_actions)

    def observation_spec(self) -> Spec:
        agent_view_shape = self._env.observation_space.shape
        agent_view_min = self._env.observation_space.minimum
        agent_view_max = self._env.observation_space.maximum
        agent_view_spec = specs.BoundedArray(
            shape=agent_view_shape,
            dtype=float,
            minimum=agent_view_min,
            maximum=agent_view_max,
        )
        action_mask_spec = Array(shape=(self._n_actions,), dtype=float)

        return specs.Spec(
            Observation,
            "ObservationSpec",
            agent_view=agent_view_spec,
            action_mask=action_mask_spec,
            step_count=Array(shape=(), dtype=int),
        )


# add goal to extras
class NavixGoalWrapper(Wrapper):
    def __init__(self, env: Environment):
        self._env = env
        self._n_actions = len(self._env.action_set)

    def reset(self, key: chex.PRNGKey) -> Tuple[NavixEnvState, TimeStep]:
        key_1, key_2, key_3 = jax.random.split(key, 3)
        state, timestep = self._env.reset(key_1)

        player_pos = state.navix_state.state.get_player().position
        grid = room(height=self.height, width=self.width)
        goal_pos = random_positions(key_2, grid, n=1, exclude=jnp.array([player_pos]))
        # goal
        goal = Goal.create(position=goal_pos, probability=jnp.asarray(1.0))
        # goal = Goal.create(position=jnp.asarray([2, 2]), probability=jnp.asarray(1.0))
        state.navix_state.state.set_goals(goal[None])

        timestep.extras["goal"] = goal.position
        timestep.extras["player"] = player_pos
        if Entities.BALL in state.navix_state.state.entities.keys():
            timestep.extras["obstacles"] = state.navix_state.state.get_balls().position

        return state, timestep

    def step(self, state: NavixEnvState, action: chex.Array) -> Tuple[NavixEnvState, TimeStep]:
        next_state, timestep = self._env.step(state, action)

        timestep.extras["goal"] = next_state.navix_state.state.get_goals().position[0]
        timestep.extras["player"] = next_state.navix_state.state.get_player().position

        if Entities.BALL in state.navix_state.state.entities.keys():
            timestep.extras["obstacles"] = next_state.navix_state.state.get_balls().position
        return next_state, timestep

    def reward_spec(self) -> specs.Array:
        return self._env.reward_spec()

    def discount_spec(self) -> specs.BoundedArray:
        return self._env.discount_spec()

    def action_spec(self) -> Spec:
        return self._env.action_spec()

    def observation_spec(self) -> Spec:
        return self._env.observation_spec()

    def goal_spec(self) -> Spec:
        goal_min = [0, 0]
        goal_max = [self._env.height, self._env.width]
        goal_spec = specs.BoundedArray(
            shape=(2,),
            dtype=int,
            minimum=goal_min,
            maximum=goal_max,
        )
        return goal_spec
