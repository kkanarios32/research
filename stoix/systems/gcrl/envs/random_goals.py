# Copyright 2023 The Navix Authors.

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


from __future__ import annotations
from typing import Union, Callable

import jax
import jax.numpy as jnp
from jax import Array
from flax import struct

from navix import observations, rewards

from navix.components import EMPTY_POCKET_ID
from navix.entities import Entities, Goal, Player
from navix.states import State
from navix.grid import random_positions, random_directions, room
from navix.rendering.cache import RenderingCache
from navix.environments.environment import Environment, Timestep
from navix.environments.registry import register_env
from navix.rewards import on_goal_reached


class RandomGoals(Environment):
    random_start: bool = struct.field(pytree_node=False, default=False)
    n_obstacles: int = struct.field(pytree_node=False, default=2)
    reward_fn: Callable[[State, Array, State], Array] = struct.field(
        pytree_node=False, default=on_goal_reached
    )

    def _reset(self, key: Array, cache: Union[RenderingCache, None] = None) -> Timestep:
        key, k1, k2, k3 = jax.random.split(key, 4)

        # map
        grid = room(height=self.height, width=self.width)

        # goal and player
        if self.random_start:
            player_pos = random_positions(k1, grid)
            direction = random_directions(k2, n=1)
        else:
            player_pos = jnp.asarray([1, 1])
            direction = jnp.asarray(0)
        # player
        player = Player.create(
            position=player_pos,
            direction=direction,
            pocket=EMPTY_POCKET_ID,
        )
        goal_pos = random_positions(
            k3, grid, n=1, exclude=jnp.array([player_pos]))
        # goal
        goal = Goal.create(position=goal_pos, probability=jnp.asarray(1.0))

        entities = {
            Entities.PLAYER: player[None],
            Entities.GOAL: goal[None],
        }

        # systems
        state = State(
            key=key,
            grid=grid,
            cache=cache or RenderingCache.init(grid),
            entities=entities,
        )

        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation_fn(state),
            action=jnp.asarray(0, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )


register_env(
    "Navix-Random-Goals-5x5-v0",
    lambda *args, **kwargs: RandomGoals.create(
        height=5,
        width=5,
        n_obstacles=2,
        random_start=False,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-Random-Goals-5x5-Random-v0",
    lambda *args, **kwargs: RandomGoals.create(
        height=5,
        width=5,
        n_obstacles=2,
        random_start=True,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-Random-Goals-6x6-v0",
    lambda *args, **kwargs: RandomGoals.create(
        height=6,
        width=6,
        n_obstacles=3,
        random_start=False,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-Random-Goals-6x6-Random-v0",
    lambda *args, **kwargs: RandomGoals.create(
        height=6,
        width=6,
        n_obstacles=3,
        random_start=True,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-Random-Goals-8x8-v0",
    lambda *args, **kwargs: RandomGoals.create(
        height=8,
        width=8,
        n_obstacles=4,
        random_start=False,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        *args,
        **kwargs,
    ),
)
register_env(
    "Navix-Random-Goals-16x16-v0",
    lambda *args, **kwargs: RandomGoals.create(
        height=16,
        width=16,
        n_obstacles=8,
        random_start=False,
        observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        reward_fn=kwargs.pop("reward_fn", rewards.on_goal_reached),
        *args,
        **kwargs,
    ),
)
