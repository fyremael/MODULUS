"""
modulus.optim.groups

Parameter-group labeling and Optax multi_transform helpers.

Goal:
- Allow different Hyperball (and base optimizer) configs per parameter group,
  e.g. attention vs MLP vs embeddings.
- Keep this path-aware and Flax-friendly: labels derived from pytree paths.

This module provides:
- make_param_labels_regex: assign group labels via regex patterns.
- make_llm_default_labels: a reasonable LLM preset.
- make_grouped_hyperball_tx: build optax.multi_transform with per-group Hyperball wrappers.

All functions are designed to be robust and ablatable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import jax
import optax

from .hyperball import hyperball

PyTree = Any
Path = Tuple[Any, ...]

LabelTree = PyTree  # pytree of strings


def _path_to_str(path: Path) -> str:
    def _part(p: Any) -> str:
        # jax.tree_util path elements are often DictKey/SequenceKey/GetAttrKey wrappers.
        if hasattr(p, "key"):
            return str(p.key)
        if hasattr(p, "idx"):
            return str(p.idx)
        if hasattr(p, "name"):
            return str(p.name)
        return str(p)

    return "/".join(_part(p) for p in path)


@dataclass(frozen=True)
class LabelRule:
    """Assign label if regex matches the parameter path."""

    label: str
    pattern: str


def make_param_labels_regex(
    rules: Sequence[LabelRule],
    *,
    default_label: str = "other",
    leaf_predicate: Optional[Callable[[Any], bool]] = None,
) -> Callable[[PyTree], LabelTree]:
    """
    Build a labeling function mapping params pytree -> pytree of labels.

    Rules are evaluated in order; first match wins.
    If no rule matches -> default_label.

    leaf_predicate can be used to assign "other" to non-arrays.
    """
    compiled = [(r.label, re.compile(r.pattern, flags=re.IGNORECASE)) for r in rules]

    def labels_fn(params: PyTree) -> LabelTree:
        def f(path, leaf):
            if leaf_predicate is not None and not leaf_predicate(leaf):
                return default_label
            s = _path_to_str(path)
            for lab, pat in compiled:
                if pat.search(s):
                    return lab
            return default_label

        return jax.tree_util.tree_map_with_path(f, params)

    return labels_fn


def make_llm_default_labels(
    *,
    attn_label: str = "attn",
    mlp_label: str = "mlp",
    embed_label: str = "embed",
    norm_label: str = "norm",
    bias_label: str = "bias",
    default_label: str = "other",
) -> Callable[[PyTree], LabelTree]:
    """
    Default labeler for typical Flax/HF-style transformer param trees.

    Heuristics:
    - attention modules: "SelfAttention", "attention", "attn"
    - MLP / feedforward: "Dense", "mlp", "ffn"
    - embeddings: "Embed", "embedding"
    - norm params: LayerNorm/RMSNorm
    - bias params: keys containing "bias"
    """
    rules = [
        LabelRule(bias_label, r"(?:^|/)(bias|b)(?:$|/|_)"),
        LabelRule(norm_label, r"(layernorm|rmsnorm|norm)"),
        LabelRule(embed_label, r"(embed|embedding)"),
        LabelRule(attn_label, r"(selfattention|attention|attn)"),
        LabelRule(mlp_label, r"(mlp|ffn|feedforward|dense)"),
    ]

    import jax.numpy as jnp

    def leaf_pred(leaf):
        return isinstance(leaf, jnp.ndarray)

    return make_param_labels_regex(rules, default_label=default_label, leaf_predicate=leaf_pred)


def make_grouped_hyperball_tx(
    *,
    base_by_group: Mapping[str, optax.GradientTransformation],
    hyperball_kwargs_by_group: Mapping[str, Mapping[str, Any]],
    labels_fn: Callable[[PyTree], LabelTree],
    default_group: str = "other",
) -> Callable[[PyTree], optax.GradientTransformation]:
    """
    Construct a grouped Optax transformation for params using optax.multi_transform.

    For each group label:
      tx_group = hyperball(base_by_group[group], **hyperball_kwargs_by_group[group])

    For groups without explicit hyperball kwargs, we use the base optimizer
    without a Hyperball wrapper.
    """

    def build(params: PyTree) -> optax.GradientTransformation:
        labels = labels_fn(params)

        txs: Dict[str, optax.GradientTransformation] = {}
        for g, base in base_by_group.items():
            hb_kwargs = dict(hyperball_kwargs_by_group.get(g, {}))
            # If no hyperball kwargs, pass-through base optimizer
            if hb_kwargs:
                txs[g] = hyperball(base, **hb_kwargs)
            else:
                txs[g] = base

        # ensure default exists
        if default_group not in txs:
            # fall back: identity
            txs[default_group] = optax.identity()

        return optax.multi_transform(txs, labels)

    return build
