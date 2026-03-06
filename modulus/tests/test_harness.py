"""
modulus.tests.test_harness

Regression tests for SimpleTrainState + make_train_step JIT behavior.
"""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import optax

from modulus.training.harness import (
    SimpleTrainState,
    make_eval_step,
    make_train_step,
    run_train_loop,
)


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

    def test_run_loop_emits_periodic_validation_metrics(self):
        def apply_fn(params, x):
            return x @ params["W"]

        def loss_fn(apply_fn, params, batch):
            x, y = batch
            pred = apply_fn(params, x)
            return jnp.mean((pred - y) ** 2), pred

        params = {"W": jnp.eye(4, dtype=jnp.float32)}
        tx = optax.sgd(1e-2)
        state = SimpleTrainState.create(params=params, tx=tx)
        train_step = make_train_step(apply_fn=apply_fn, loss_fn=loss_fn)
        eval_step = make_eval_step(apply_fn=apply_fn, loss_fn=loss_fn)

        x = jnp.ones((2, 4), dtype=jnp.float32)
        y = jnp.zeros((2, 4), dtype=jnp.float32)
        train_batches = [(x, y)] * 5
        eval_batches = [(x, y)]

        final_state, history = run_train_loop(
            state=state,
            train_step_fn=train_step,
            train_batches=train_batches,
            num_steps=5,
            eval_step_fn=eval_step,
            eval_batches=eval_batches,
            eval_interval=2,
        )

        self.assertEqual(len(history), 5)
        self.assertIn("train/loss", history[0])
        self.assertNotIn("val/loss", history[0])
        self.assertIn("val/loss", history[1])  # step 2
        self.assertNotIn("val/loss", history[2])
        self.assertIn("val/loss", history[3])  # step 4
        self.assertIn("val/loss", history[4])  # final step always
        self.assertTrue(bool(final_state.step == 5))


if __name__ == "__main__":
    unittest.main()
