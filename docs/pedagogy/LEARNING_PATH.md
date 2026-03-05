# Learning Path

## Goal

Bring a new engineer from zero context to productive experimentation with
Hyperball, grouped optimization, and LoRA steering.

## Phase 1: Orientation (30-60 min)

1. Read [../DOCUMENTATION.md](../DOCUMENTATION.md).
2. Read [../context/MOTIVATION_AND_CONTEXT.md](../context/MOTIVATION_AND_CONTEXT.md).
3. Skim package layout in `modulus/optim`, `modulus/peft`, `modulus/training`.

## Phase 2: Baseline Operation (30 min)

1. Install dev environment:
```bash
python -m pip install -e ".[dev]"
```
2. Run tests:
```bash
python -m pytest
```
3. Run strict lint:
```bash
python -m ruff check modulus scripts
```

## Phase 3: Core Concepts (60-90 min)

1. Hyperball mechanics:
- Read `modulus/optim/hyperball.py`.
- Focus on tangent projection, target angle, retraction.

2. Grouped transforms:
- Read `modulus/optim/groups.py`.
- Validate path-based labeling assumptions.

3. Masking policy:
- Read `modulus/optim/masks.py`.
- Understand which leaves are constrained and why.

4. LoRA steering:
- Read `modulus/peft/lora.py`.
- Understand gradient orthogonalization role and limitations.

## Phase 4: Hands-On Practice (60 min)

1. Run benchmark generator:
```bash
python scripts/run_benchmarks.py
```
2. Build benchmark report:
```bash
python scripts/build_benchmark_report.py
```
3. Validate integration tree:
```bash
python scripts/validate_integration_tree.py
```

## Phase 5: Integration Readiness (60 min)

1. Follow [../integration_guide.md](../integration_guide.md).
2. Start with ungrouped baseline.
3. Enable grouped Hyperball with conservative angles.
4. Add LoRA hook only after grouped behavior is stable.

## Common Pitfalls

1. Over-aggressive `target_angle` causing instability.
2. Mis-labeled parameter paths leading to unintended `other` group usage.
3. Constraining embeddings or `lm_head` without explicit intent.
4. Comparing benchmark variants with inconsistent seeds or shapes.

## Definition of Proficiency

You are productive when you can:
1. Explain why a parameter leaf is or is not constrained.
2. Create and run a new ablation row in benchmark CSV outputs.
3. Interpret benchmark report trade-offs (quality vs speed).
4. Add a model-specific preset with tests and keep CI green.

