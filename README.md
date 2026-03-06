# MODULUS Hyperball Next Steps (1–3)

This package contains:

1) **Training Harness** (`modulus/training/harness.py`)
   - JAX-first, JIT-friendly step function builder
   - Hyperball metrics extraction
   - Optional LoRA gradient steering hook

2) **Parameter-group Hyperball** (`modulus/optim/groups.py`)
   - Path-aware labeler for params (attn vs mlp vs embed vs norm vs bias)
   - `optax.multi_transform` builder so each group can have distinct Hyperball configs
   - Designed for clean ablations

3) **LoRA tangent-steering** (`modulus/peft/lora.py`)
   - Minimal Flax LoRA Dense module
   - `orth_lora_grad_jax` + `apply_lora_grad_hook` to orthogonalize LoRA factor gradients
   - Works alongside Hyperball constraints on base weights

## Quick start

Install (development):
```bash
python -m pip install -e ".[dev]"
```

Run unit tests:
```bash
python -m pytest
```

Run the grouped Hyperball + LoRA demo:
```bash
python -m pip install -e ".[examples]"
python -m modulus.examples.train_grouped_hyperball_lora_demo
```

Run ablation benchmarks (CSV artifacts):
```bash
python scripts/run_benchmarks.py
```

Run benchmark on a real-world streamed corpus (SlimPajama / MiniPile-style):
```bash
python -m pip install datasets
python scripts/run_benchmarks.py \
  --data-source hf_stream \
  --dataset-name cerebras/SlimPajama-627B \
  --dataset-train-split train \
  --dataset-eval-split validation
```

Build benchmark report from CSV artifacts:
```bash
python -m pip install -e ".[report]"
python scripts/build_benchmark_report.py --with-plots
```

## Ablation switches

- Turn off Hyperball per group by setting that group's `hb_kwargs_by_group[label] = {}`.
- Turn off LoRA gradient steering via `use_lora=False` in the config (or `use_lora_grad_hook=False`).
- Switch Hyperball granularity: `"leaf"` vs `"row"` vs `"col"` vs `"channel"`.
- Try `"ball"` mode with `radial_decay` and `ball_norm_clamp`.

## IP and compliance

- License (MIT): `LICENSE`
- Provenance record: `docs/ip/provenance.md`
- AI origin evidence: `docs/ip/ai_origin_evidence.md`
- Diligence signoff: `docs/ip/SIGNOFF.md`
- Third-party notices inventory: `THIRD_PARTY_NOTICES.md`
- Diligence report: `IP_DUE_DILIGENCE_REPORT_2026-03-04.md`
- Contribution policy: `CONTRIBUTING.md`
- Security policy: `SECURITY.md`

## Engineering handoff

- GCT handoff plan: `docs/engineering/ENGINEERING_HANDOFF_GCT_2026-03-04.md`
- Install flow: `docs/engineering/INSTALL.md`
- Lint phase plan: `docs/engineering/LINT_PHASE_PLAN.md`
- Integration guide: `docs/integration_guide.md`
- Documentation hub: `docs/DOCUMENTATION.md`
- Colab pedagogy notebook: `notebooks/MODULUS_Pedagogical_Walkthrough.ipynb`
- Artifacts layout: `artifacts/README.md`
- CI workflow: `.github/workflows/ci.yml`

## Presets

- LLaMA/HF decoder presets: `modulus.optim.presets`
