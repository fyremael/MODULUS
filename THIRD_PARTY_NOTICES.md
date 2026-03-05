# Third-Party Notices

Date: 2026-03-04

This file tracks third-party software used by this repository and associated
license obligations for redistribution.

## Inventory

| Package | Role | Version evidence | License | Obligations summary | Verification status |
|---|---|---|---|---|---|
| jax | Runtime dependency (`import jax`) | Installed locally: `0.4.38` | Apache-2.0 | Preserve copyright/license notice; include license text for redistribution. | Verified locally |
| jaxlib | Runtime dependency (binary backend for JAX) | Installed locally: `0.4.38` | Apache-2.0 | Preserve copyright/license notice; include license text for redistribution. | Verified locally |
| optax | Runtime dependency (`import optax`) | Installed locally: `0.2.5` | Apache-2.0 | Preserve copyright/license notice; include license text for redistribution. | Verified locally |
| flax | Optional dependency for demos (`import flax.linen`) | Not installed in baseline environment | Apache-2.0 (expected) | Include notices if installed/distributed with product artifacts. | Pending optional-path verification |
| pytest | Test dependency | Installed locally: `8.3.5` | MIT | Preserve copyright/license notice; include license text for redistribution where required. | Verified locally |

## Notes

1. Runtime/test baseline verified with `python -m pytest -q -p no:cacheprovider modulus/tests` on 2026-03-04.
2. Add transitive dependencies and their licenses once a lockfile/SBOM is generated.
3. If binaries are distributed, include a complete notices bundle with all required license texts.
