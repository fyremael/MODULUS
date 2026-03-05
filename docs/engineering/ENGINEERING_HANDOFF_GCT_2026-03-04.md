# Engineering Handoff: GCT Execution

- Date: 2026-03-04
- Project: MODULUS
- Handoff mode: build-and-validate

## What This Is

MODULUS is a JAX/Optax research toolkit focused on:
1. Hyperball-constrained optimization
2. Parameter-grouped optimizer policies
3. LoRA gradient steering utilities
4. A minimal training harness + demo loop

## Current Baseline

1. Runtime/test baseline validated:
- `jax==0.4.38`
- `jaxlib==0.4.38`
- `optax==0.2.5`
- `pytest==8.3.5`

2. Test evidence:
- Command: `python -m pytest`
- Result: `10 passed`

3. Key modules:
- `modulus/optim/hyperball.py`
- `modulus/optim/groups.py`
- `modulus/peft/lora.py`
- `modulus/training/harness.py`
- `modulus/examples/train_grouped_hyperball_lora_demo.py`

## Execution Status (2026-03-04)

Completed now:
1. Added `pyproject.toml` package metadata with extras: `dev`, `examples`, `report`, `all`.
2. Added install runbook: `docs/engineering/INSTALL.md`.
3. Validated editable dev install: `python -m pip install -e ".[dev]"`.
4. Validated default test flow: `python -m pytest` -> `10 passed`.
5. Added benchmark runner: `scripts/run_benchmarks.py`.
6. Added artifact output conventions: `artifacts/README.md`.
7. Verified benchmark artifact generation on Windows.
8. Added integration guide: `docs/integration_guide.md`.
9. Added integration tree validator: `scripts/validate_integration_tree.py`.
10. Verified integration validator output (`Group check: PASS`).
11. Added CI workflow: `.github/workflows/ci.yml` (Python `3.10/3.11/3.12` matrix).
12. Added lint gate via `ruff` in dev toolchain and CI.
13. Added benchmark report builder: `scripts/build_benchmark_report.py`.
14. Added CI benchmark/report job with artifact upload.
15. Added model-specific presets for LLaMA/HF decoder trees (`modulus.optim.presets`).
16. Added staged strict-style lint phase (`scripts/run_style_lint_phase.py`) with CI advisory artifact.
17. Burned down strict-style advisory findings and promoted full-repo strict style lint to blocking in CI.
18. Enabled line-length enforcement (`E501`) and verified full strict lint passes.
19. Added auto-updating API docs pipeline and docs freshness CI gate.

## GCT Engineering Objective

Turn the current toolkit into a reproducible, benchmarked, packageable engineering asset that can be used by model teams with minimal friction.

## Sprint Plan (Recommended)

### Workstream 1: Packaging + DX

1. Add `pyproject.toml` and package metadata.
2. Split dependencies into `core`, `dev`, and `examples`.
3. Add a one-command smoke path:
- `python -m pytest`

Acceptance:
1. Fresh environment setup is deterministic.
2. `pip install -e .[dev]` (or equivalent) works cleanly.

### Workstream 2: Reproducible Benchmarks

1. Add benchmark script for:
- Hyperball on/off
- LoRA grad hook on/off
- grouped vs ungrouped configs
2. Emit CSV metrics:
- loss trajectory
- step time
- angle/radial metrics (if available)

Acceptance:
1. Benchmark runs end-to-end with fixed seed.
2. Artifacts are written under `artifacts/benchmarks/<date>/`.

### Workstream 3: Integration Readiness

1. Add adapter examples for real model parameter trees.
2. Validate masks/labels on at least one transformer-like tree.
3. Document failure modes and fallback configs.

Acceptance:
1. Integration guide demonstrates drop-in usage in < 30 lines.
2. Known caveats are documented with mitigation guidance.

### Workstream 4: CI Hardening

1. Add CI job for tests and lint.
2. Add matrix for Python versions used internally.
3. Persist test/benchmark artifacts on CI runs.

Acceptance:
1. Green CI on default branch.
2. Failed runs preserve diagnostics.

## Immediate Task Queue

1. Continue routine maintenance to keep full strict lint and tests green.
2. Expand pedagogical docs with additional worked examples over real model trees.

## Definition Of Done (GCT Phase)

1. Reproducible setup and tests on a clean machine.
2. Benchmark evidence for key ablations.
3. Integration recipe for external model teams.
4. CI-backed quality gate for ongoing changes.
