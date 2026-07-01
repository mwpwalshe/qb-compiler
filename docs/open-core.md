# Open core: qb-compiler vs QubitBoost Pro

qb-compiler is **open source and free**. It is the preflight layer for quantum execution — circuit
viability, backend recommendation, cost estimation, calibration-aware compilation, and QEC decoder-input
correctness (**ObservableGate**). You can run all of it, in CI, forever, for free.

**QubitBoost Pro** adds the team/lab/enterprise layer on top: signed and stored receipts, batch reports,
shared dashboards, CI policy bundles, and the broader governance stack (SafetyGate, DriftGate, the QEC
preflight bundle, Result Passport). The goal is simple: prove value for free, charge only once teams
depend on the receipts.

## What's free (qb-compiler, open source)

| Capability | Command |
|---|---|
| Circuit viability preflight | `qbc preflight` / `qbc analyze` / `qbc diff` |
| Calibration-aware compilation + receipt | `qbc compile --receipt` |
| Mirror-circuit verification | `qbc verify` |
| **ObservableGate: QEC DEM correctness audit** | `qbc dem-audit` (text or `--json` receipt) |
| **Observable-preserving DEM canonicalization** | `qbc dem-canonicalize` |
| CI-safe exit codes (0 / 1 / 2) | all of the above |
| Python API, notebooks, docs | — |

The `--json` receipt is an unsigned, machine-readable community-tier artifact you can store and diff
yourself.

## What's QubitBoost Pro (paid)

| Capability | Notes |
|---|---|
| **Signed receipts** | tamper-evident, attributable ObservableGate / fidelity / drift receipts |
| **Batch audit reports** | `dem-audit ./dems/` over a whole directory, rolled into one report |
| **CI policy bundles** | team policy files, GitHub/GitLab gate templates |
| **Receipt dashboard + history** | hosted, shared, searchable receipt storage over time |
| **DriftGate / SafetyGate / QEC-preflight bundle** | the governance stack |
| **Result Passport** | end-to-end provenance for a quantum result |
| **Deep QEC audit (services)** | private DEM/decoder audit + report + recommendations |
| Custom vendor/backend adapters, private support | — |

Pro is in development at <https://qubitboost.io/compiler>.

## Design principle

> Every check becomes a receipt. The check is free; the trusted, signed, stored, team-shared receipt is
> the product.

ObservableGate is the first module in that story (fidelity receipt · drift receipt · ObservableGate
receipt · SafetyGate receipt · QEC preflight receipt · Result Passport). Adopt the free CLI now; the Pro
layer is additive and never blocks the open-source path.
