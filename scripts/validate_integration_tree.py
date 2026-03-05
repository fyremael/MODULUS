from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

from modulus.optim.groups import make_llm_default_labels
from modulus.optim.masks import default_llm_hyperball_mask


def build_transformer_like_tree(width: int = 64, vocab: int = 256, rank: int = 8) -> Dict[str, Any]:
    key = jax.random.PRNGKey(0)
    k = jax.random.split(key, 7)
    s = 1.0 / jnp.sqrt(float(width))
    return {
        "embed": {"embedding": jax.random.normal(k[0], (vocab, width)) * 0.01},
        "block_0": {
            "attn": {
                "kernel": jax.random.normal(k[1], (width, width)) * s,
                "bias": jnp.zeros((width,)),
            },
            "mlp_fc1": {"kernel": jax.random.normal(k[2], (width, width * 4)) * s},
            "mlp_fc2": {"kernel": jax.random.normal(k[3], (width * 4, width)) * s},
            "adapter": {
                "lora_A": jax.random.normal(k[4], (width, rank)) * 0.01,
                "lora_B": jnp.zeros((rank, width)),
            },
            "LayerNorm_0": {"scale": jnp.ones((width,))},
        },
        "lm_head": {"kernel": jax.random.normal(k[5], (width, vocab)) * s},
        "misc": {"proj": jax.random.normal(k[6], (width, width)) * s},
    }


def flatten_with_paths(tree: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def rec(path: Tuple[Any, ...], node: Any):
        if isinstance(node, dict):
            for k, v in node.items():
                rec(path + (k,), v)
            return
        out["/".join(str(p) for p in path)] = node

    rec((), tree)
    return out


def main() -> None:
    params = build_transformer_like_tree()
    labels = make_llm_default_labels()(params)
    mask = default_llm_hyperball_mask(
        include_embeddings=False,
        exclude_lora=True,
        exclude_1d=True,
    )(params)

    flat_labels = flatten_with_paths(labels)
    flat_mask = flatten_with_paths(mask)

    label_counts = Counter(str(v) for v in flat_labels.values())
    mask_true = sum(1 for v in flat_mask.values() if bool(v))
    mask_total = max(len(flat_mask), 1)
    mask_frac = mask_true / mask_total

    print("Label counts:")
    for k in sorted(label_counts):
        print(f"- {k}: {label_counts[k]}")

    print("")
    print(f"Mask coverage: {mask_true}/{mask_total} ({mask_frac:.2%})")

    expected = {"attn", "mlp", "embed", "norm"}
    present = set(label_counts.keys())
    missing = sorted(expected - present)
    if missing:
        print(f"WARNING: missing expected groups: {', '.join(missing)}")
    else:
        print("Group check: PASS")


if __name__ == "__main__":
    main()
