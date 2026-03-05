"""
modulus.tests.test_groups

Basic tests for group labeling and multi_transform builder.
"""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import optax

from modulus.optim.groups import make_grouped_hyperball_tx, make_llm_default_labels
from modulus.optim.schedules import constant


class GroupsTests(unittest.TestCase):
    def test_labels_fn_shapes(self):
        params = {
            "block_0": {"attn": {"kernel": jnp.zeros((4, 4))}},
            "mlp_fc1": {"kernel": jnp.zeros((4, 4))},
            "embed": {"embedding": jnp.zeros((10, 4))},
            "LayerNorm_0": {"scale": jnp.ones((4,))},
        }
        labels_fn = make_llm_default_labels()
        labels = labels_fn(params)

        # ensure same structure
        self.assertEqual(set(labels.keys()), set(params.keys()))
        # ensure path regexes classify key groups
        self.assertEqual(labels["block_0"]["attn"]["kernel"], "attn")
        self.assertEqual(labels["mlp_fc1"]["kernel"], "mlp")
        self.assertEqual(labels["embed"]["embedding"], "embed")
        self.assertEqual(labels["LayerNorm_0"]["scale"], "norm")

    def test_multi_transform_build(self):
        params = {"W": jnp.ones((4, 4))}

        def labels_fn(_params):
            return {"W": "attn"}

        base = optax.sgd(1e-1)
        tx_builder = make_grouped_hyperball_tx(
            base_by_group={"attn": base, "other": base},
            hyperball_kwargs_by_group={
                "attn": {
                    "radius": 1.0,
                    "mode": "sphere",
                    "granularity": "leaf",
                    "target_angle": constant(0.1),
                }
            },
            labels_fn=labels_fn,
            default_group="other",
        )
        tx = tx_builder(params)
        st = tx.init(params)
        grads = {"W": jnp.ones((4, 4))}
        upd, st2 = tx.update(grads, st, params)
        self.assertIn("W", upd)


if __name__ == "__main__":
    unittest.main()
