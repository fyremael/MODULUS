"""
modulus.tests.test_lora_grad

Tests the LoRA gradient orthogonalization hook.
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp

from modulus.peft.lora import orth_lora_grad_jax


class LoRAGradTests(unittest.TestCase):
    def test_orth_projection_reduces_component(self):
        key = jax.random.PRNGKey(0)
        A = jax.random.normal(key, (16, 4))
        B = jax.random.normal(key, (12, 4))
        Ag = jax.random.normal(key, (16, 4))
        Bg = jax.random.normal(key, (12, 4))

        Ag2, Bg2 = orth_lora_grad_jax(A, B, Ag, Bg, eps=1e-5)

        # Check that A^T Ag2 is smaller in Frobenius norm than A^T Ag
        lhs_before = jnp.linalg.norm(A.T @ Ag)
        lhs_after = jnp.linalg.norm(A.T @ Ag2)
        self.assertTrue(bool(lhs_after <= lhs_before + 1e-5))

        rhs_before = jnp.linalg.norm(B.T @ Bg)
        rhs_after = jnp.linalg.norm(B.T @ Bg2)
        self.assertTrue(bool(rhs_after <= rhs_before + 1e-5))


if __name__ == "__main__":
    unittest.main()
