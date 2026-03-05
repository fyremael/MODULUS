"""
modulus.optim.masks

Path-aware mask utilities for selecting which parameters are constrained by Hyperball.

We keep this conservative and LLM-friendly by default:
- Exclude biases
- Exclude LayerNorm/RMSNorm gains & biases (often 1D scale vectors)
- Exclude small scalars
- Optionally exclude embeddings or LoRA weights

These masks are intended to be used as `mask=params->mask_tree` callables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

PyTree = Any
Path = Tuple[Any, ...]


def _path_to_str(path: Path) -> str:
    # Flax uses strings for dict keys; sometimes paths include integers.
    def _part(p: Any) -> str:
        if hasattr(p, "key"):
            return str(p.key)
        if hasattr(p, "idx"):
            return str(p.idx)
        if hasattr(p, "name"):
            return str(p.name)
        return str(p)

    return "/".join(_part(p) for p in path)


def mask_all(params: PyTree) -> PyTree:
    return jax.tree.map(lambda _: True, params)


def mask_none(params: PyTree) -> PyTree:
    return jax.tree.map(lambda _: False, params)


@dataclass(frozen=True)
class RegexMaskSpec:
    include: Optional[Sequence[str]] = None
    exclude: Optional[Sequence[str]] = None

    def compile(self):
        inc = [re.compile(p) for p in (self.include or [])]
        exc = [re.compile(p) for p in (self.exclude or [])]
        return inc, exc


def make_regex_mask(spec: RegexMaskSpec) -> Callable[[PyTree], PyTree]:
    """
    Build a path-aware mask based on regex patterns.
    - If include patterns exist, only paths matching at least one include are True.
    - Exclude patterns always win.
    """
    inc, exc = spec.compile()

    def mask_fn(params: PyTree) -> PyTree:
        def f(path, leaf):
            p = _path_to_str(path)
            ok = True
            if inc:
                ok = any(r.search(p) for r in inc)
            if exc and any(r.search(p) for r in exc):
                ok = False
            return bool(ok)

        return jax.tree_util.tree_map_with_path(f, params)

    return mask_fn


def default_llm_hyperball_mask(
    *,
    include_embeddings: bool = False,
    exclude_lora: bool = True,
    exclude_1d: bool = True,
    min_ndim: int = 2,
) -> Callable[[PyTree], PyTree]:
    """
    A strong default mask for LLM pretraining:
    - Include: weight matrices and conv kernels (ndim>=2)
    - Exclude: biases, norm gains, (optionally) embeddings, and (optionally) LoRA weights.
    """
    # Common naming patterns across Flax/HF-like param trees
    bias_pat = re.compile(r"(?:^|/)(bias|b)(?:$|/)", re.IGNORECASE)
    norm_pat = re.compile(r"(layernorm|rmsnorm|norm)(?:$|/)", re.IGNORECASE)
    scale_pat = re.compile(r"(?:^|/)(scale|gamma|gain)(?:$|/)", re.IGNORECASE)
    embed_pat = re.compile(r"(embed|embedding)(?:$|/)", re.IGNORECASE)
    lora_pat = re.compile(r"(lora|adapter)(?:$|/)", re.IGNORECASE)

    def mask_fn(params: PyTree) -> PyTree:
        def f(path, leaf):
            p = _path_to_str(path)
            if not isinstance(leaf, jnp.ndarray):
                return False

            # Exclude trivial small tensors
            if leaf.size == 0:
                return False

            if exclude_1d and leaf.ndim == 1:
                return False

            if leaf.ndim < min_ndim:
                return False

            if bias_pat.search(p):
                return False
            if norm_pat.search(p) or scale_pat.search(p):
                return False
            if exclude_lora and lora_pat.search(p):
                return False
            if (not include_embeddings) and embed_pat.search(p):
                return False

            return True

        return jax.tree_util.tree_map_with_path(f, params)

    return mask_fn
