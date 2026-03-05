# MODULUS Documentation Hub

This is the entry point for technical, contextual, and operational documentation.

## 1) API Reference (Auto-Generated)

- API index: [api_generated/INDEX.md](./api_generated/INDEX.md)
- Source of truth: code introspection via `scripts/generate_api_docs.py`

## 2) Context and Motivation

- [context/MOTIVATION_AND_CONTEXT.md](./context/MOTIVATION_AND_CONTEXT.md)

## 3) Pedagogy and Learning Path

- [pedagogy/LEARNING_PATH.md](./pedagogy/LEARNING_PATH.md)
- [../notebooks/MODULUS_Pedagogical_Walkthrough.ipynb](../notebooks/MODULUS_Pedagogical_Walkthrough.ipynb)

## 4) Future Extensions

- [roadmap/FUTURE_EXTENSIONS.md](./roadmap/FUTURE_EXTENSIONS.md)

## 5) Integration and Engineering Operations

- [integration_guide.md](./integration_guide.md)
- [engineering/INSTALL.md](./engineering/INSTALL.md)
- [engineering/ENGINEERING_HANDOFF_GCT_2026-03-04.md](./engineering/ENGINEERING_HANDOFF_GCT_2026-03-04.md)
- [engineering/LINT_PHASE_PLAN.md](./engineering/LINT_PHASE_PLAN.md)

## 6) Compliance and Governance

- [ip/provenance.md](./ip/provenance.md)
- [ip/ai_origin_evidence.md](./ip/ai_origin_evidence.md)
- [ip/SIGNOFF.md](./ip/SIGNOFF.md)

## Auto-Update Workflow

Generate API docs:

```bash
python scripts/generate_api_docs.py
```

Check freshness (CI mode):

```bash
python scripts/generate_api_docs.py --check
```
