import functools
from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp

import stoix.systems.gcrl.search.mcts.base as base
import stoix.systems.gcrl.search.mcts.qtransform as qtransforms
from stoix.systems.gcrl.search.mcts import tree as tree_lib
from stoix.systems.gcrl.search.mcts.base import PolicyOutput, QTransform
from stoix.systems.gcrl.search.mcts.search import search

Params = Any


def _get_logits_from_probs(probs):
    tiny = jnp.finfo(probs.dtype).tiny
    return jnp.log(jnp.maximum(probs, tiny))


def _apply_temperature(logits, temperature):
    """Returns `logits / temperature`, supporting also temperature=0."""
    # The max subtraction prevents +inf after dividing by a small temperature.
    logits = logits - jnp.max(logits, keepdims=True, axis=-1)
    tiny = jnp.finfo(logits.dtype).tiny
    return logits / jnp.maximum(tiny, temperature)


def action_selection(
    rng_key: chex.PRNGKey,
    tree: tree_lib.Tree,
    node_index: chex.Numeric,
    depth: chex.Numeric,
    pb_c_init: float = 1.25,
    pb_c_base: float = 19652.0,
    *,
    qtransform: QTransform = qtransforms.qtransform_by_parent_and_siblings,
) -> chex.Array:
    visit_counts = tree.children_visits[node_index]
    node_visit = tree.node_visits[node_index]
    pb_c = pb_c_init + jnp.log((node_visit + pb_c_base + 1.0) / pb_c_base)
    prior_logits = tree.children_prior_logits[node_index]
    obstacle_logits = tree.children_obstacle_logits[node_index]
    prior_probs = jax.nn.softmax(prior_logits / obstacle_logits)
    policy_score = jnp.sqrt(node_visit) * pb_c * prior_probs / (visit_counts + 1)
    chex.assert_shape([node_index, node_visit], ())
    chex.assert_equal_shape([prior_probs, visit_counts, policy_score])
    value_score = qtransform(tree, node_index)
    obstacle_probs = jax.nn.softmax(obstacle_logits)
    obstacle_penalty = normalized_obstacles * (1 - jnp.exp(visit_counts**2 / 1000))

    # Add tiny bit of randomness for tie break
    node_noise_score = 1e-7 * jax.random.uniform(rng_key, (tree.num_actions,))
    # to_argmax = value_score + policy_score + node_noise_score
    to_argmax = value_score + policy_score + node_noise_score - obstacle_penalty

    # Masking the invalid actions at the root.
    return jnp.argmax(to_argmax)


def safe_policy(
    params: Params,
    rng_key: chex.PRNGKey,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    num_simulations: int,
    invalid_actions: Optional[chex.Array] = None,
    max_depth: Optional[int] = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
    *,
    qtransform: QTransform = qtransforms.qtransform_by_parent_and_siblings,
    temperature: chex.Numeric = 1.0,
) -> base.PolicyOutput[None]:
    """Runs MuZero search and returns the `PolicyOutput`.

    In the shape descriptions, `B` denotes the batch dimension.

    Args:
      params: params to be forwarded to root and recurrent functions.
      rng_key: random number generator state, the key is consumed.
      root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
        `prior_logits` are from a policy network. The shapes are
        `([B, num_actions], [B], [B, ...])`, respectively.
      recurrent_fn: a callable to be called on the leaf nodes and unvisited
        actions retrieved by the simulation step, which takes as args
        `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
        and the new state embedding. The `rng_key` argument is consumed.
      num_simulations: the number of simulations.
      invalid_actions: a mask with invalid actions. Invalid actions
        have ones, valid actions have zeros in the mask. Shape `[B, num_actions]`.
      max_depth: maximum search tree depth allowed during simulation.
      loop_fn: Function used to run the simulations. It may be required to pass
        hk.fori_loop if using this function inside a Haiku module.
      qtransform: function to obtain completed Q-values for a node.
      temperature: temperature for acting proportionally to
        `visit_counts**(1 / temperature)`.

    Returns:
      `PolicyOutput` containing the proposed action, action_weights and the used
      search tree.
    """
    rng_key, dirichlet_rng_key, search_rng_key = jax.random.split(rng_key, 3)

    # Adding Dirichlet noise.
    prior_logits = jax.nn.softmax(root.prior_logits)
    # root = root.replace(prior_logits=_mask_invalid_actions(noisy_logits, invalid_actions))
    root = root.replace(prior_logits=prior_logits)

    # Running the search.
    interior_action_selection_fn = functools.partial(
        action_selection,
        qtransform=qtransform,
    )
    root_action_selection_fn = functools.partial(interior_action_selection_fn, depth=0)

    search_tree = search(
        params=params,
        rng_key=search_rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        root_action_selection_fn=root_action_selection_fn,
        interior_action_selection_fn=interior_action_selection_fn,
        num_simulations=num_simulations,
        max_depth=max_depth,
        invalid_actions=invalid_actions,
        loop_fn=loop_fn,
    )

    # Sampling the proposed action proportionally to the visit counts.
    summary = search_tree.summary()
    action_weights = summary.visit_probs
    action_logits = _apply_temperature(
        _get_logits_from_probs(action_weights), temperature
    )
    action = jax.random.categorical(rng_key, action_logits)
    return PolicyOutput(
        action=action, action_weights=action_weights, search_tree=search_tree
    )
