# Future Extensions

## Extension Philosophy

Prioritize additions that improve:
1. Experimental validity (clearer ablations, better diagnostics)
2. Integration reliability (safer defaults, better presets)
3. Operational confidence (repeatability, automation, evidence trails)

## Near-Term Extensions

1. Model-specific preset library expansion:
- Add presets for additional architectures beyond LLaMA/HF-style decoder trees.
- Include tests proving label/mask intent on representative parameter trees.

2. Rich benchmark analytics:
- Add trend statistics (variance, confidence intervals).
- Add richer report visualizations and comparative tables.

3. Hyperball policy variants:
- Adaptive angle schedules tied to training stage/metrics.
- Alternative tangent/radial blending policies.

## Mid-Term Extensions

1. Multi-device benchmarking profiles:
- Standardized CPU/GPU profile sets and artifact schemas.

2. Contracted API layers:
- Stabilize public API surface and version compatibility policy.

3. Config-driven experiment runner:
- Declarative experiment matrix definitions with reproducible outputs.

## Long-Term Extensions

1. Automated policy search over grouped Hyperball settings.
2. Cross-model transfer studies for optimizer preset portability.
3. Tight integration with release evidence packs for production governance.

## Guardrails for New Extensions

1. Every new feature ships with at least one regression test.
2. Benchmark/report pipeline must remain runnable in CI.
3. Lint and tests must stay fully green.
4. Documentation updates are required in the same change.

## De-Scope Candidates

1. Features that add heavy complexity without clear measurement strategy.
2. Integrations that cannot be validated with current benchmark/report pipeline.
3. Changes that weaken reproducibility guarantees.

