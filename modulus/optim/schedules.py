"""
modulus.optim.schedules

Simple schedule utilities for Hyperball:
- constant schedules
- warmup + cosine decay schedules for angular step α

These are intentionally minimal and dependency-free beyond jax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import jax.numpy as jnp

Schedule = Union[float, Callable[[jnp.ndarray], jnp.ndarray]]


def constant(value: float) -> Callable[[jnp.ndarray], jnp.ndarray]:
    v = float(value)

    def f(step: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(v, dtype=jnp.float32)

    return f


@dataclass(frozen=True)
class WarmupCosine:
    """
    Warm up linearly to peak, then cosine decay.

    Parameters
    ----------
    warmup_steps:
        Number of warmup steps.
    total_steps:
        Total steps for the schedule. For steps >= total_steps, value is end_value.
    peak_value:
        Value at end of warmup.
    end_value:
        Final value at total_steps.
    """

    warmup_steps: int
    total_steps: int
    peak_value: float
    end_value: float = 0.0

    def __call__(self, step: jnp.ndarray) -> jnp.ndarray:
        s = jnp.asarray(step, dtype=jnp.float32)
        w = jnp.asarray(float(self.warmup_steps), dtype=jnp.float32)
        T = jnp.asarray(float(self.total_steps), dtype=jnp.float32)
        peak = jnp.asarray(float(self.peak_value), dtype=jnp.float32)
        end = jnp.asarray(float(self.end_value), dtype=jnp.float32)

        # warmup
        warm = jnp.clip(s / jnp.maximum(w, 1.0), 0.0, 1.0)
        warm_val = warm * peak

        # cosine decay
        # t in [0, 1] after warmup, clamped
        t = jnp.clip((s - w) / jnp.maximum(T - w, 1.0), 0.0, 1.0)
        cos = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
        cos_val = end + (peak - end) * cos

        return jnp.where(s < w, warm_val, cos_val)
