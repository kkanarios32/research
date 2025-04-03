# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for `policies.py`."""


import jax
import jax.numpy as jnp
import mctx
import numpy as np
from absl.testing import absltest

from stoix.systems.gcrl.search.mcts_policy import safe_policy


def _make_bandit_recurrent_fn(rewards, dummy_embedding=()):
    """Returns a recurrent_fn with discount=0."""

    def recurrent_fn(params, rng_key, action, embedding):
        del params, rng_key, embedding
        reward = rewards[jnp.arange(action.shape[0]), action]
        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=jnp.zeros_like(reward),
                prior_logits=jnp.zeros_like(rewards),
                value=jnp.zeros_like(reward),
            ),
            dummy_embedding,
        )

    return recurrent_fn


class PoliciesTest(absltest.TestCase):
    def test_muzero_policy(self):
        root = mctx.RootFnOutput(
            prior_logits=jnp.array(
                [
                    [-1.0, 0.0, 2.0, 3.0],
                ]
            ),
            value=jnp.array([0.0]),
            embedding=(),
        )
        rewards = jnp.zeros_like(root.prior_logits)
        invalid_actions = jnp.array(
            [
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        policy_output = safe_policy(
            params=(),
            rng_key=jax.random.PRNGKey(0),
            root=root,
            recurrent_fn=_make_bandit_recurrent_fn(rewards),
            num_simulations=1,
            invalid_actions=invalid_actions,
        )

        expected_action = jnp.array([2], dtype=jnp.int32)
        np.testing.assert_array_equal(expected_action, policy_output.action)
        # expected_action_weights = jnp.array(
        #     [
        #         [0.0, 0.0, 1.0, 0.0],
        #     ]
        # )
        # np.testing.assert_allclose(expected_action_weights, policy_output.action_weights)


if __name__ == "__main__":
    absltest.main()
