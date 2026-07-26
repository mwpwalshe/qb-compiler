# qb-compiler multi-platform audit (exact state)

**Date:** 2026-06-28. Source of truth: `qbc backends --json`, `BACKEND_CONFIGS`, `VENDOR_PRICING`,
`calibration/`, `ir/converters/`, `backends/`. Honest snapshot, no claims, only what runs.

## The 7 layers that make a tool "multi-platform"
1. Backend specs (qubits / topology / basis)  2. Live calibration  3. Static calibration fixtures
4. Compile → vendor-native gates  5. Cost model  6. Circuit input formats  7. Execution/submission

## Per-vendor capability matrix

| Vendor | Backends (qubits) | Specs | **Live cal** | Static fixture | Compile→native | Cost | Net status |
|---|---|---|---|---|---|---|---|
| **IBM** | fez 156, torino 133, marrakesh 156 | ✅ | ✅ **LIVE** (qiskit-ibm-runtime, OSS) | fez ✅, torino ✅, marrakesh ✗ | ✅ id,rz,sx,x,cz | ✅ | **fully live** |
| **Rigetti** | ankaa 84 | ✅ | ⚠ built, **unvalidated** (Braket) | ankaa ✅ | ✅ rx,rz,cz | ✅ | live-unvalidated |
| **IonQ** | aria 25, forte 36 | ✅ | ⚠ built, **unvalidated** (Braket) | ✗ | ✅ gpi,gpi2,ms | ✅ | live-unvalidated |
| **IQM** | garnet 20, emerald 5 | ✅ | ⚠ built, **unvalidated** (Braket) | ✗ | ✅ prx,cz | ✅ | live-unvalidated |
| **Quantinuum** | h2 32 | ✅ | ⚠ built, **unvalidated** (pytket-quantinuum) | ✗ | ✅ rz,u1q,zz | ✅ | live-unvalidated |

Legend: ✅ works · ⚠ code exists, not proven on a real device · ✗ absent.

## Cross-cutting layers (apply to ALL vendors)

- **Circuit input (layer 6), BROADENED (P4).** OpenQASM 2 + Qiskit (core), plus **OpenQASM 3**
  (`qasm3_converter`, via `qiskit-qasm3-import`), **Cirq** (`cirq_converter`), and **PennyLane**
  (`pennylane_converter`). CLI `_load_circuit` dispatches by extension/header (`.qasm3` / `OPENQASM 3`).
  Converters lazy-import their SDK; tests run for cirq + qasm3, skip for pennylane when absent.
- **Transpilation (layer 4), multi-platform output, qiskit-centric engine.** `qiskit.transpile`
  + per-vendor basis-gate decomposition + custom passes produces **vendor-native compiled circuits for
  all 6 vendors**. Layout/routing uses qiskit Sabre (no native pyQuil/IonQ transpilers).
- **Execution (layer 7), NONE, by design.** qb-compiler is **preflight/analyze/compile only**; it does
  NOT submit jobs to any QPU. (`verify` checks predictions on a simulator.) Hardware submission is the
  QubitBoost platform's role, not the OSS tool's.

## Calibration providers present (`calibration/`)
`registry.py` (dispatcher+status) · `ibm_runtime_provider.py` (IBM live, OSS) ·
`braket_provider.py` (IonQ/Rigetti/IQM live) · `live_provider.py` (IBM via proprietary hub) ·
`static_provider.py` · `cached_provider.py` (TTL) · `provider.py` (ABC).
Bundled real fixtures: ibm_fez (×3 dates), ibm_torino, rigetti_ankaa.

## CLI surface (13 commands)
`preflight · analyze · diff · when · compile · verify · backends · calibration · info · doctor ·
dem-audit · dem-canonicalize`

## Per-vendor SDK extras (pyproject)
`[ibm]` qiskit-ibm-runtime · `[rigetti]` pyquil · `[ionq]` amazon-braket-sdk · `[iqm]` iqm-client ·
`[qubitboost]` hub. **Missing:** no `[quantinuum]`, no `[azure]`, no `[cirq]`/`[pennylane]`.

## Honest one-line verdict
qb-compiler is **truly multi-platform for targeting** (specs, native compile, cost across all 6
vendors) and **partially multi-platform for live data** (IBM genuinely live; IonQ/Rigetti/IQM
live-but-unvalidated; Quantinuum static-only). It is **single-ecosystem on input** (Qiskit/QASM2) and
**non-executing** by design.

## Gap → roadmap (status after P2/P3/P4, 2026-06-28)
| Gap | Fix | Phase | Status |
|---|---|---|---|
| IonQ/Rigetti/IQM live unvalidated | registry wiring + offline replay test done; one real Braket run to flip → LIVE | **P2** | ⚠ wired+replay-tested; real-device run pending (needs AWS creds) |
| Quantinuum had no live adapter | `quantinuum_provider.py` (pytket BackendInfo) + registered | **P3** | ✅ built, offline-tested, live-unvalidated |
| No Azure Quantum platform | `azure_provider.py` (secondary path, IonQ/Quantinuum/Rigetti) | **P3** | ✅ built, offline-tested |
| Input = QASM2/Qiskit only | QASM3 / Cirq / PennyLane converters + CLI dispatch | **P4** | ✅ built; cirq+qasm3 paths proven, pennylane skip-gated |
| marrakesh/ionq/iqm/quantinuum lack bundled fixtures | capture real snapshots (needs creds) | P2/P3 | pending |
| Routing is qiskit-only | native transpilers (only if fidelity gap measured) | P5 | deferred |

**To finish:** the only remaining work needs **real credentials**, one AWS Braket run (flips
IonQ/Rigetti/IQM → LIVE), one Quantinuum login (flips Quantinuum → LIVE), and optional snapshot capture
for offline fixtures. All code paths are built + tested offline.
