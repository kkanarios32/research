from typing import Dict
from flax.core.frozen_dict import FrozenDict
from optax import OptState

import chex
from typing_extensions import NamedTuple

from stoix.base_types import Action, Done


class Transition(NamedTuple):
    done: Done
    action: Action
    reward: chex.Array
    obs: chex.Array
    player_pos: chex.Array
    goal: chex.Array
    traj_id: chex.Array
    info: Dict


class ContrastiveParams(NamedTuple):
    """Parameters of an actor critic network."""

    sa_params: FrozenDict
    g_params: FrozenDict


class ContrastiveOptState(NamedTuple):
    """OptStates of actor critic learner."""

    sa_opt_state: OptState
    g_opt_state: OptState
