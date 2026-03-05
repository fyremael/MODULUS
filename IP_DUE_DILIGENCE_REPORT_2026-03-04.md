# IP Due Diligence Report

- Date: 2026-03-04
- Target asset: `F:\_codex\MODULUS`
- Reviewer: Codex (workspace review)
- Scope: repository-level legal/commercial diligence for newly acquired code IP

## Executive Summary

The codebase is technically coherent and now has baseline governance artifacts in place (MIT license, provenance ledger, contributor policy, third-party notices, dependency baseline, and passing unit tests). Remaining risk is concentrated in legal provenance evidence finalization for AI-generated origin and full transitive OSS notice capture.

Current recommendation: **Proceed with controlled external OSS release under MIT**, while restricting enterprise/commercial assurances until legal provenance signoff is completed.

## Evidence Reviewed

- `README.md`
- `requirements.txt`
- `LICENSE`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `THIRD_PARTY_NOTICES.md`
- `docs/ip/provenance.md`
- `docs/ip/ai_origin_evidence.md`
- `docs/ip/SIGNOFF.md`
- `modulus/optim/hyperball.py`
- `modulus/optim/groups.py`
- `modulus/optim/masks.py`
- `modulus/optim/schedules.py`
- `modulus/peft/lora.py`
- `modulus/training/harness.py`
- `modulus/tests/test_hyperball.py`
- `modulus/tests/test_groups.py`
- `modulus/tests/test_lora_grad.py`
- Environment checks:
  - `git` metadata unavailable (`not a git repository`)
  - dependency baseline installed: `jax==0.4.38`, `jaxlib==0.4.38`, `optax==0.2.5`, `pytest==8.3.5`
  - test evidence: `python -m pytest -q -p no:cacheprovider modulus/tests` -> `6 passed`

## Findings (Severity-Ordered)

1. **High**: Provenance legal evidence is not fully closed.
- Origin declaration is recorded: `ChatGPT 5.2 via Grand Challenge Labs`.
- Terms snapshot, account-control evidence, and legal approval fields are still pending.
- Impact: commercialization representations remain constrained until counsel signoff.

2. **Resolved (was Critical)**: Outbound licensing is declared.
- Repository `LICENSE` is MIT.
- Residual action: legal confirmation that MIT aligns with commercialization and patent posture.

3. **Medium**: Third-party compliance is partially complete.
- First-party dependency inventory exists and baseline versions are verified.
- Transitive dependency notices/SBOM are not yet captured.
- Impact: redistribution notice bundle could be incomplete without SBOM closure.

4. **Resolved (was High)**: Contributor intake controls are now present.
- `CONTRIBUTING.md` defines DCO certification, prohibited sources, and AI-origin disclosure expectations.

5. **Resolved (was Medium)**: Technical reproducibility baseline established.
- Unit tests pass on scoped suite (`6/6`) with pinned dependency versions.
- Additional optional demo path (Flax) remains outside baseline.

6. **Medium**: Patent/trade-secret governance remains informal.
- No explicit internal patent disclosure log or trade-secret handling checklist is present.

## Risk Register

- Overall commercialization risk: **Medium**
- OSS release-readiness: **Go (controlled)**
- Enterprise/commercial assurance readiness: **No-Go** pending legal provenance closure

## Remediation Plan

### P0 (Complete before enterprise assurances / commercial claims)

1. Complete provenance legal package:
- Archive platform terms/version applicable to `ChatGPT 5.2` output rights.
- Attach account/control evidence for generation context.
- Finalize `docs/ip/provenance.md` and `docs/ip/SIGNOFF.md` with legal signoff.

2. Finalize third-party compliance package:
- Generate lockfile/SBOM and include transitive licenses/notices.

### P1 (Governance hardening)

1. Add internal patent/trade-secret decision record.
2. Add release-gate checklist requiring provenance + notices verification at each release.
3. Validate optional Flax demo path in a separate optional dependency profile.

## Go/No-Go Decision

- Decision: **Go for MIT OSS distribution with controlled claims**
- Conditions for full commercial-assurance Go:
  - legal provenance signoff completed
  - transitive third-party compliance package completed

## Suggested Disclosure Language (Current)

"This project is licensed under MIT. Provenance and third-party compliance verification are in progress; enterprise support/commercial assurances remain subject to completion of diligence artifacts."

## Execution Update (2026-03-04)

Completed in repository:
- Added outbound `LICENSE` (MIT).
- Added provenance and AI-origin evidence templates.
- Added `THIRD_PARTY_NOTICES.md` baseline inventory.
- Added `CONTRIBUTING.md` and `SECURITY.md` governance controls.
- Added pinned `requirements.txt` baseline.
- Fixed optional Flax import behavior in `modulus/peft/__init__.py`.
- Fixed test determinism issue in `modulus/tests/test_hyperball.py`.
- Verified tests pass (`6 passed`).

Still pending for closure:
- Legal signoff fields in `docs/ip/SIGNOFF.md`.
- Terms snapshot and account-control evidence in `docs/ip/ai_origin_evidence.md`.
- Transitive dependency/SBOM license closure.
