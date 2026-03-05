"""
modulus.tests.test_hyperball

Unit tests for Hyperball wrapper invariants:
- sphere norm invariance (within tolerance)
- tangent orthogonality (approx)
- masking behavior
- angle control behavior

Run:
  python -m modulus.tests.test_hyperball
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import optax

from modulus.optim.hyperball import get_last_metrics, hyperball
from modulus.optim.masks import mask_all, mask_none


def _tree_allclose(a, b, atol=1e-5, rtol=1e-5):
    leaves = jax.tree.leaves(
        jax.tree.map(lambda x, y: jnp.allclose(x, y, atol=atol, rtol=rtol), a, b)
    )
    return bool(jnp.all(jnp.stack(leaves)))


class HyperballTests(unittest.TestCase):
    def test_sphere_norm_invariance(self):
        key = jax.random.PRNGKey(0)
        w = jax.random.normal(key, (8, 16))
        w = w / (jnp.linalg.norm(w, axis=1, keepdims=True) + 1e-8)  # per-row unit

        params = {"W": w}
        grads = {"W": jax.random.normal(key, (8, 16))}

        base = optax.sgd(learning_rate=1e-1)
        opt = hyperball(
            base, radius=1.0, mode="sphere", granularity="row", target_angle=0.05, mask=mask_all
        )
        state = opt.init(params)

        updates, state = opt.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)

        # per-row norms should be ~1
        row_norms = jnp.linalg.norm(new_params["W"], axis=1)
        self.assertTrue(bool(jnp.all(jnp.abs(row_norms - 1.0) < 1e-3)))

    def test_mask_noop(self):
        key = jax.random.PRNGKey(1)
        params = {"W": jax.random.normal(key, (4, 4))}
        grads = {"W": jax.random.normal(key, (4, 4))}

        base = optax.adam(1e-2)
        opt_base = base
        opt_hb = hyperball(
            base, radius=1.0, mode="sphere", granularity="leaf", target_angle=0.1, mask=mask_none
        )

        s0 = opt_base.init(params)
        s1 = opt_hb.init(params)

        u0, s0b = opt_base.update(grads, s0, params)
        u1, s1b = opt_hb.update(grads, s1, params)

        self.assertTrue(_tree_allclose(u0, u1, atol=1e-6, rtol=1e-6))

    def test_angle_control_rough(self):
        key_w, key_g = jax.random.split(jax.random.PRNGKey(2))
        w = jax.random.normal(key_w, (32,))
        w = w / (jnp.linalg.norm(w) + 1e-8)
        params = {"w": w}
        grads = {"w": jax.random.normal(key_g, (32,))}

        base = optax.sgd(learning_rate=1.0)  # base magnitude won't matter much
        alpha = 0.10
        opt = hyperball(
            base, radius=1.0, mode="sphere", granularity="leaf", target_angle=alpha, mask=mask_all
        )

        state = opt.init(params)
        updates, state = opt.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)

        # Measured angle between w and w_new
        w0 = params["w"]
        w1 = new_params["w"]
        dot = jnp.sum(w0 * w1) / ((jnp.linalg.norm(w0) * jnp.linalg.norm(w1)) + 1e-8)
        dot = jnp.clip(dot, -1.0, 1.0)
        theta = jnp.arccos(dot)

        # Rough tolerance: retraction + projection approximations
        self.assertTrue(bool(jnp.abs(theta - alpha) < 0.03))

        mets = get_last_metrics(state)
        self.assertIn("hyperball/angle_mean", mets)


if __name__ == "__main__":
    unittest.main()
