# Motivation and Context

## Why MODULUS Exists

MODULUS addresses a specific optimization problem in modern model training:
how to constrain and steer updates geometrically (Hyperball/spherical dynamics)
while still preserving practical optimizer workflows and ablation flexibility.

## Core Design Motivation

1. Geometry as a first-class control surface:
- Hyperball wraps standard Optax optimizers to expose angle/norm behavior directly.

2. Group-specific behavior:
- Different parameter regions (attention, MLP, embeddings, norms, bias) often
  benefit from different constraints and update regimes.

3. Adapter-aware training:
- LoRA factors can be handled with additional gradient steering to reduce
  degenerate low-rank update directions.

4. Experimental velocity with operational discipline:
- Small, composable modules for ablation speed.
- Tests, lint, benchmarks, reports, and CI to keep experiments reproducible.

## Problem Framing

The project is best understood as an optimization research toolkit with product-like
engineering controls, not a monolithic framework replacement.

## Intended Audience

1. Optimization researchers testing geometric update hypotheses.
2. ML engineers integrating grouped optimizer policies into existing JAX loops.
3. Platform teams requiring reproducible benchmark/report pipelines.

## Non-Goals

1. Full end-to-end model training framework competing with major ecosystems.
2. Automatic architecture discovery or AutoML.
3. Turnkey model zoo with broad pretrained checkpoints.

