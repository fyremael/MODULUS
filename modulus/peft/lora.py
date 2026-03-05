"""
modulus.peft.lora

A minimal Flax-compatible LoRA implementation + gradient steering hooks.

This module provides:
- LoRADense: drop-in replacement for nn.Dense with low-rank adapters
- orth_lora_grad_jax: gradient orthogonalization / tangent steering for LoRA factors
- apply_lora_grad_hook: apply the orthogonalization to a param/grads pytree

Design philosophy:
- Base weights may be Hyperball constrained (sphere)
- LoRA acts as tangent steering (low-rank directional adjustment)
- Orthogonalizing LoRA gradients reduces degeneracy and improves conditioning

Note: We do not assume LoRA factors are orthonormal; we project gradients to be
orthogonal to the current column spaces (span-orthogonalization).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import jax.numpy as jnp

try:
    import flax.linen as nn
except Exception:
    nn = None  # type: ignore

PyTree = Any


def _safe_solve(mat: jnp.ndarray, rhs: jnp.ndarray, eps: float) -> jnp.ndarray:
    """
    Solve (mat + eps I) x = rhs with float32 accumulation.
    mat: [..., r, r]
    rhs: [..., r, k]
    """
    r = mat.shape[-1]
    eye = jnp.eye(r, dtype=jnp.float32)
    mat32 = mat.astype(jnp.float32) + eps * eye
    rhs32 = rhs.astype(jnp.float32)
    sol = jnp.linalg.solve(mat32, rhs32)
    return sol.astype(rhs.dtype)


def orth_lora_grad_jax(
    A: jnp.ndarray,
    B: jnp.ndarray,
    A_grad: jnp.ndarray,
    B_grad: jnp.ndarray,
    *,
    eps: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Orthogonalize LoRA gradients by removing components that lie in the column span of A and B.

    For A (shape [in_dim, r]):
      A_grad <- A_grad - A @ (A^T A + eps I)^{-1} @ (A^T A_grad)
    For B (shape [out_dim, r]):
      B_grad <- B_grad - B @ (B^T B + eps I)^{-1} @ (B^T B_grad)

    This is a practical tangent-steering proxy:
    - reduces redundant scaling directions
    - stabilizes low-rank updates
    - plays well with Hyperball-constrained base weights

    Returns:
      (A_grad_new, B_grad_new)
    """
    # A projection
    AtA = A.T @ A
    AtG = A.T @ A_grad
    proj_A = A @ _safe_solve(AtA, AtG, eps)

    # B projection
    BtB = B.T @ B
    BtG = B.T @ B_grad
    proj_B = B @ _safe_solve(BtB, BtG, eps)

    return A_grad - proj_A, B_grad - proj_B


def apply_lora_grad_hook(
    params: PyTree,
    grads: PyTree,
    *,
    a_name: str = "lora_A",
    b_name: str = "lora_B",
    eps: float = 1e-6,
) -> PyTree:
    """
    Apply orth_lora_grad_jax to all LoRA parameter pairs discovered in the pytree.

    Convention:
      each LoRA module stores params:
        {"kernel": W, "lora_A": A, "lora_B": B}

    We walk each dict node; when both A and B exist, we adjust their grads.
    """

    def rec(pnode, gnode):
        if isinstance(pnode, dict) and isinstance(gnode, dict):
            out = {}
            # First recurse children
            for k in gnode.keys():
                out[k] = rec(pnode.get(k, None), gnode[k])

            if a_name in pnode and b_name in pnode and a_name in gnode and b_name in gnode:
                A = pnode[a_name]
                B = pnode[b_name]
                Ag = gnode[a_name]
                Bg = gnode[b_name]
                if (
                    isinstance(A, jnp.ndarray)
                    and isinstance(B, jnp.ndarray)
                    and isinstance(Ag, jnp.ndarray)
                    and isinstance(Bg, jnp.ndarray)
                ):
                    Ag2, Bg2 = orth_lora_grad_jax(A, B, Ag, Bg, eps=eps)
                    out[a_name] = Ag2
                    out[b_name] = Bg2
            return out
        # Non-dict: pass through grad
        return gnode

    return rec(params, grads)


# ----------------------
# Flax LoRA Dense module
# ----------------------


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 8.0
    dropout: float = 0.0
    merge_weights: bool = False  # optional experimental


if nn is not None:

    class LoRADense(nn.Module):
        """A Dense layer with LoRA adapters: y = xW + scale*(xA)B."""

        features: int
        use_bias: bool = False
        lora: Optional[LoRAConfig] = None
        kernel_init: Callable = nn.initializers.lecun_normal()

        @nn.compact
        def __call__(self, x, deterministic: bool = True):
            y = nn.Dense(
                self.features, use_bias=self.use_bias, kernel_init=self.kernel_init, name="base"
            )(x)

            if self.lora is None or self.lora.rank <= 0:
                return y

            r = self.lora.rank
            scale = self.lora.alpha / float(r)

            in_dim = x.shape[-1]
            A = self.param("lora_A", nn.initializers.normal(stddev=0.02), (in_dim, r))
            B = self.param("lora_B", nn.initializers.zeros, (r, self.features))

            if self.lora.dropout > 0.0 and not deterministic:
                x_in = nn.Dropout(rate=self.lora.dropout)(x, deterministic=deterministic)
            else:
                x_in = x

            lora_out = (x_in @ A) @ B
            return y + scale * lora_out
