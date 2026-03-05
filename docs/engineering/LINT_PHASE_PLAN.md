# Lint Phase Plan

## Purpose

Introduce stricter style linting without blocking velocity on legacy files.

## Phase 0 (Completed)

- Command: `python -m ruff check modulus scripts`
- Rule profile: correctness-only (`E9`, `F63`, `F7`, `F82`)
- Status: completed

## Phase 1 (Completed)

- Command: `python scripts/run_style_lint_phase.py --phase staged`
- Effective rules: `E,F,I` with `E501` ignored
- Scope:
  - `modulus/optim/presets.py`
  - `scripts/`
- Status: completed

## Phase 2 (Current Blocking)

- Command: `python -m ruff check modulus scripts`
- Effective rules: `E,F,I` including `E501`
- Scope: `modulus/` + `scripts/`
- Status: blocking in CI

## Follow-On Policy

1. Keep full strict lint (`E,F,I` + `E501`) clean on all new changes.
2. Run `python -m ruff format modulus scripts` before committing larger refactors.
