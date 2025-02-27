from typing import TYPE_CHECKING, Tuple, Any

import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.specs import Array, DiscreteArray, Spec
from jumanji.types import StepType, TimeStep, restart
from jumanji.wrappers import Wrapper
from navix.environments.environment import Environment
from navix.environments.environment import Timestep as NavixState
from navix.entities import Entities

from stoix.base_types import Observation

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass


State = TimeStep


@dataclass
class IdState:
    env_state: State
    traj_id: chex.Array


# NOTE: requires env to be wrapped in episode metric wrapper first.
class TrajectoryIdWrapper(Wrapper):
    def __init__(self, env: Environment):
        self._env = env

    def reset(self, rng: jax.Array) -> State:
        env_state, timestep = self._env.reset(rng)
        timestep.extras["traj_id"] = jnp.zeros(rng.shape[:-1], dtype=int)
        new_state = IdState(env_state=env_state,
                            traj_id=jnp.zeros(rng.shape[:-1], dtype=int))
        return new_state, timestep

    def step(self, state: IdState, action: jax.Array) -> State:
        traj_id = state.traj_id + \
            jnp.where(state.env_state.running_count_episode_length, 0, 1)
        env_state, timestep = self._env.step(state.env_state, action)
        next_state = IdState(traj_id=traj_id, env_state=env_state)
        timestep.extras["traj_id"] = traj_id
        return next_state, timestep
