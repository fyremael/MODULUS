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
python scripts/run_benchmarks.py \
  --data-source hf_http \
  --dataset-name HuggingFaceFW/fineweb \
  --dataset-config sample-10BT \
  --dataset-train-split train \
  --dataset-eval-split train \
  --hardware-aware \
  --max-tokens-per-step 8192 \
  --auto-token-pool-by-host-ram \
  --host-ram-token-pool-fraction 0.20 \
  --dataset-http-cache-dir artifacts/datasets/hf_http_cache \
  --dataset-http-cache-read \
  --dataset-http-cache-write \
  --lr 6e-4 \
  --lr-schedule warmup_cosine \
  --lr-warmup-steps 500 \
  --lr-min-ratio 0.10
```
Note: for `hf_http`, keep `--dataset-rows-page-size` at `<=100` (HF API limit).
For higher request budgets, set `HF_TOKEN` in the environment and the runner will
send `Authorization: Bearer ...` to dataset-server.
If rate-limited, increase `--dataset-http-max-retries` and
`--dataset-http-min-interval-sec`.
Use `--log-interval` (for example `10`) to print rich live progress lines.
For long runs, increase `--step-record-interval` (for example `10` or `25`) to
reduce memory and CSV size while preserving eval snapshots.
Hardware-aware mode is on by default and can downshift `batch_size` or
`token_pool_batches` when requested settings exceed device/host limits.
For `hf_stream` mode, set `HF_HOME` / `HF_DATASETS_CACHE` to a persistent path
to reuse downloaded shards across reruns.

If a dataset is unavailable in your environment, switch to another public stream
(for example `--dataset-name cerebras/SlimPajama-627B`) or log in with
`huggingface-cli login` for gated datasets.

Colab TPU substantial baseline (single config, 1B-token budget):
```bash
python scripts/run_benchmarks.py \
  --data-source hf_http \
  --dataset-name HuggingFaceFW/fineweb \
  --dataset-config sample-10BT \
  --configs baseline \
  --target-train-tokens 1000000000 \
  --steps 1000 \
  --max-steps 90000 \
  --lr 6e-4 \
  --lr-schedule warmup_cosine \
  --lr-warmup-steps 2000 \
  --lr-min-ratio 0.10 \
  --lr-total-steps 90000
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
