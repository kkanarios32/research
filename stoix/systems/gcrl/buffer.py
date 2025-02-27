import jax
import jax.numpy as jnp
from jax.random import PRNGKey

from stoix.systems.gcrl.gcrl_types import Transition


# @functools.partial(jax.jit, static_argnames=["config", "env"])
def flatten_crl_fn(config, env, transition: Transition, sample_key: PRNGKey) -> Transition:
    goal_key, transition_key = jax.random.split(sample_key)

    # Because it's vmaped transition obs.shape is of shape (transitions,obs_dim)
    seq_len = transition.obs.agent_view.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(
        arrangement[:, None] < arrangement[None], dtype=jnp.float32)
    discount = 0.9 ** jnp.array(
        arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount
    single_trajectories = jnp.concatenate(
        [transition.traj_id[:, jnp.newaxis].T] * seq_len, axis=0
    )
    probs = probs * jnp.equal(single_trajectories,
                              single_trajectories.T) + jnp.eye(seq_len) * 1e-5
    goal_index = jax.random.categorical(goal_key, jnp.log(probs))

    # TODO: deal with last state
    goals = jnp.take(transition.player_pos,
                     goal_index, axis=0)
    future_action = jnp.take(transition.action, goal_index, axis=0)
    # __import__('pdb').set_trace()
    obs = jax.tree.map(lambda x: x[:-1], transition.obs)

    return transition._replace(
        goal=goals
    )
