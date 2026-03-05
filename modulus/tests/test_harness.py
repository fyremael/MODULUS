"""
modulus.tests.test_harness

Regression tests for SimpleTrainState + make_train_step JIT behavior.
"""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import optax

from modulus.training.harness import SimpleTrainState, make_train_step


class HarnessTests(unittest.TestCase):
    def test_simple_train_state_jit_step(self):
        def apply_fn(params, x):
            return x @ params["W"]

        def loss_fn(apply_fn, params, batch):
            x, y = batch
            pred = apply_fn(params, x)
            return jnp.mean((pred - y) ** 2), pred

        params = {"W": jnp.eye(4, dtype=jnp.float32)}
        tx = optax.sgd(1e-2)
        state = SimpleTrainState.create(params=params, tx=tx)
        step_fn = make_train_step(apply_fn=apply_fn, loss_fn=loss_fn)

        x = jnp.ones((2, 4), dtype=jnp.float32)
        y = jnp.zeros((2, 4), dtype=jnp.float32)
        new_state, metrics = step_fn(state, (x, y))

        self.assertIn("loss", metrics)
        self.assertEqual(new_state.step.shape, ())
        self.assertTrue(bool(new_state.step == 1))


if __name__ == "__main__":
    unittest.main()
