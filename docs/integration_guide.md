# Integration Guide

## Goal

Attach MODULUS Hyperball + grouped optimization to an existing JAX/Optax training loop with minimal changes.

## 30-Line Drop-In Recipe

```python
import optax
from modulus.optim.groups import make_llm_default_labels, make_grouped_hyperball_tx
from modulus.optim.masks import default_llm_hyperball_mask

def build_tx(params, lr=1e-3, wd=1e-2):
    labels_fn = make_llm_default_labels()
    base = optax.adamw(lr, weight_decay=wd)
    mask_fn = default_llm_hyperball_mask(
        include_embeddings=False,
        exclude_lora=True,
        exclude_1d=True,
    )
    hb_common = dict(
        radius=1.0, mode="sphere", proj_tangent=True,
        granularity="row", mask=mask_fn, emit_metrics=True
    )
    hb_kwargs = {
        "attn": dict(**hb_common, target_angle=0.03),
        "mlp": dict(**hb_common, target_angle=0.05),
        "other": {}, "embed": {}, "norm": {}, "bias": {},
    }
    base_by_group = {k: base for k in ["attn", "mlp", "other", "embed", "norm", "bias"]}
    return make_grouped_hyperball_tx(
        base_by_group=base_by_group,
        hyperball_kwargs_by_group=hb_kwargs,
        labels_fn=labels_fn,
        default_group="other",
    )(params)
```

Use this `tx` in your existing train state:

```python
tx = build_tx(params)
opt_state = tx.init(params)
updates, opt_state = tx.update(grads, opt_state, params)
params = optax.apply_updates(params, updates)
```

## Validation Before First Real Run

Run:

```bash
python scripts/validate_integration_tree.py
```

This checks a transformer-like param tree and prints:
1. Label counts by group (`attn`, `mlp`, `embed`, `norm`, `bias`, `other`)
2. Hyperball mask coverage percentage
3. Pass/fail warnings for missing expected groups

## Production Preset: LLaMA/HF Decoder

For common decoder trees (`self_attn`, `mlp`, `embed_tokens`, `layernorm`, `lm_head`), use the built-in preset:

```python
from modulus.optim.presets import build_llama_grouped_hyperball_tx

tx_builder = build_llama_grouped_hyperball_tx(
    lr=1e-3,
    weight_decay=1e-2,
    attn_angle=0.03,
    mlp_angle=0.05,
    include_embeddings=False,
    include_lora=False,
    include_lm_head=False,
)
tx = tx_builder(params)
```

Preset behavior:
1. Labels attention/MLP/embedding/norm paths using LLaMA/HF naming heuristics.
2. Excludes embeddings, LoRA adapters, norms, bias, and `lm_head` from Hyperball by default.
3. Applies grouped Hyperball to `attn` and `mlp` with separate target angles.

## Caveats and Fallbacks

1. If most params fall into `other`, customize label regex rules with `make_param_labels_regex`.
2. If training destabilizes, lower `target_angle` first (for example `0.05 -> 0.02`).
3. If embedding drift is desired, set `include_embeddings=True` in the mask.
4. If LoRA adapters should be constrained, set `exclude_lora=False`.
5. If model has atypical names, define project-specific labeling rules instead of defaults.

## Suggested Integration Sequence

1. Start ungrouped Hyperball and validate loss/metrics behavior.
2. Enable grouped Hyperball (`attn`/`mlp`) with conservative angles.
3. Enable LoRA gradient hook after grouped behavior is stable.
4. Benchmark ablations with `scripts/run_benchmarks.py` and compare CSV summaries.
