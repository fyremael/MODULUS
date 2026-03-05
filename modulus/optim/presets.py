"""
modulus.optim.presets

Model-specific integration presets for production onboarding.

Current target:
- LLaMA/HF-style decoder parameter trees
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Tuple

import jax
import optax

from .groups import LabelRule, make_grouped_hyperball_tx, make_param_labels_regex
from .masks import default_llm_hyperball_mask

PyTree = Any
Path = Tuple[Any, ...]


def _path_to_str(path: Path) -> str:
    def _part(p: Any) -> str:
        if hasattr(p, "key"):
            return str(p.key)
        if hasattr(p, "idx"):
            return str(p.idx)
        if hasattr(p, "name"):
            return str(p.name)
        return str(p)

    return "/".join(_part(p) for p in path)


def make_llama_like_labels(
    *,
    attn_label: str = "attn",
    mlp_label: str = "mlp",
    embed_label: str = "embed",
    norm_label: str = "norm",
    bias_label: str = "bias",
    default_label: str = "other",
) -> Callable[[PyTree], PyTree]:
    """
    Labeler tuned for common LLaMA/HF decoder naming patterns.
    """
    rules = [
        LabelRule(bias_label, r"(?:^|/)(bias|b)(?:$|/|_)"),
        LabelRule(
            norm_label,
            r"(input_layernorm|post_attention_layernorm|layernorm|rmsnorm|norm)",
        ),
        LabelRule(embed_label, r"(embed_tokens|tok_embeddings|wte|embedding|embed)"),
        LabelRule(attn_label, r"(self_attn|attention|attn|q_proj|k_proj|v_proj|o_proj)"),
        LabelRule(
            mlp_label,
            r"(mlp|feed_forward|feedforward|ffn|gate_proj|up_proj|down_proj|fc_in|fc_out)",
        ),
    ]

    import jax.numpy as jnp

    def leaf_pred(leaf):
        return isinstance(leaf, jnp.ndarray)

    return make_param_labels_regex(rules, default_label=default_label, leaf_predicate=leaf_pred)


def make_llama_like_mask(
    *,
    include_embeddings: bool = False,
    include_lora: bool = False,
    include_lm_head: bool = False,
    exclude_1d: bool = True,
    min_ndim: int = 2,
) -> Callable[[PyTree], PyTree]:
    """
    Mask preset tuned for LLaMA/HF-style decoder trees.
    """
    base_mask_fn = default_llm_hyperball_mask(
        include_embeddings=include_embeddings,
        exclude_lora=not include_lora,
        exclude_1d=exclude_1d,
        min_ndim=min_ndim,
    )
    lm_head_pat = re.compile(r"(?:^|/)(lm_head|output_projection|out_proj)(?:$|/)", re.IGNORECASE)

    def mask_fn(params: PyTree) -> PyTree:
        base_mask = base_mask_fn(params)

        def f(path, m):
            if not bool(m):
                return False
            if include_lm_head:
                return True
            return not lm_head_pat.search(_path_to_str(path))

        return jax.tree_util.tree_map_with_path(f, base_mask)

    return mask_fn


def build_llama_grouped_hyperball_tx(
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    attn_angle: float = 0.03,
    mlp_angle: float = 0.05,
    include_embeddings: bool = False,
    include_lora: bool = False,
    include_lm_head: bool = False,
) -> Callable[[PyTree], optax.GradientTransformation]:
    """
    Build grouped Hyperball optimizer for a LLaMA/HF-style decoder tree.
    """
    base = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    labels_fn = make_llama_like_labels()
    mask_fn = make_llama_like_mask(
        include_embeddings=include_embeddings,
        include_lora=include_lora,
        include_lm_head=include_lm_head,
        exclude_1d=True,
    )

    hb_common = dict(
        radius=1.0,
        mode="sphere",
        proj_tangent=True,
        granularity="row",
        mask=mask_fn,
        emit_metrics=True,
    )
    hb_kwargs_by_group: Mapping[str, Mapping[str, Any]] = {
        "attn": dict(**hb_common, target_angle=attn_angle),
        "mlp": dict(**hb_common, target_angle=mlp_angle),
        "other": {},
        "embed": {},
        "norm": {},
        "bias": {},
    }
    base_by_group = {
        "attn": base,
        "mlp": base,
        "other": base,
        "embed": base,
        "norm": base,
        "bias": base,
    }
    return make_grouped_hyperball_tx(
        base_by_group=base_by_group,
        hyperball_kwargs_by_group=hb_kwargs_by_group,
        labels_fn=labels_fn,
        default_group="other",
    )
