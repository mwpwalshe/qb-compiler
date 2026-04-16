# Changelog

All notable changes to [qb-compiler](https://qubitboost.io/compiler), the open-source quantum circuit compiler by [QubitBoost](https://qubitboost.io), will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Qiskit SDK 2.x compatibility: `qiskit` dependency widened to `>=1.0,<3.0`.
- CI now runs the test suite against both Qiskit 1.4 and Qiskit 2.0 in matrix.
- `QBCalibrationLayoutPlugin` — proper `qiskit.transpiler.layout` stage plugin.
  Invoke via `generate_preset_pass_manager(layout_method="qb_calibration")`
  with the `QB_CALIBRATION_PATH` env var set.  Plugin is now discoverable
  through Qiskit's entry-point system.

### Changed
- `qb_transpile()` now injects `QBCalibrationLayout` into the pass manager's
  `pre_layout` stage instead of `layout`.  On Qiskit 2.x the previous
  approach triggered `ApplyLayout` `KeyError` and silently fell back to
  stock `qiskit.transpile`, bypassing calibration-aware layout.  The
  custom pipeline is now the primary code path on both Qiskit versions.
- `QBTranspilerPlugin` entry-point group corrected from the non-existent
  `qiskit.transpiler.stage` to `qiskit.transpiler.layout`, now pointing at
  `QBCalibrationLayoutPlugin`.  The plugin was previously undiscoverable
  via Qiskit's loader.

### Deprecated
- `QBTranspilerPlugin.get_pass_manager(calibration_data=...)` — emits a
  `DeprecationWarning` and will be removed in 0.4.0.  Migrate to
  `generate_preset_pass_manager(layout_method="qb_calibration")` with
  `QB_CALIBRATION_PATH` set, or call `qb_transpile()` directly.

### Fixed
- `ci.yml` workflow now triggers on `master` as well as `main` (the repo's
  default branch is `master`; the workflow had been dormant).
- Removed the phantom `[qiskit]` optional-dependency extra from CI install
  commands (it did not exist in `pyproject.toml` and was silently ignored).

## [0.1.0] - 2026-03-13

### Added
- Core IR: QBCircuit, QBDag, QBGate, QBMeasure, QBBarrier
- Qiskit and OpenQASM 2.0 converters
- CalibrationMapper: VF2-based calibration-weighted qubit placement
- NoiseAwareRouter: Dijkstra shortest-error-path SWAP routing
- NoiseAwareScheduler: ALAP scheduling with T1/T2 urgency scoring
- GateDecomposition: Native basis decomposition (IBM ECR/RZ/SX/X, Rigetti CZ/RX/RZ, IonQ MS/GPI/GPI2, IQM CZ/PRX)
- ErrorBudgetEstimator: Pre-execution fidelity prediction
- T1 asymmetry awareness: readout-scaled penalty for qubits with high |1> decay
- Temporal correlation detection: Pearson correlation across calibration snapshots
- Calibration subsystem: StaticCalibrationProvider, CachedCalibrationProvider, BackendProperties
- Noise modelling: EmpiricalNoiseModel, FidelityEstimator
- Backend support: IBM Heron, Rigetti Ankaa, IonQ Aria/Forte, IQM Garnet/Emerald
- Cost estimation with vendor pricing
- Qiskit transpiler plugin: QBCalibrationLayout, qb_transpile(), QBPassManager
- CLI: `qbc compile`, `qbc info`, `qbc calibration show`
- Gate cancellation and commutation analysis optimisation passes
- Depth and gate count analysis passes
- ML Phase 2: XGBoost layout predictor (AUC=0.94, 454KB, +5.4% fidelity on GHZ-8)
- ML Phase 3: GNN layout predictor (dual-graph GCN, 42KB, +6.5% fidelity on QAOA-8)
- ML Phase 4: RL SWAP router (PPO actor-critic, 190KB, calibration-aware routing)
- ML training infrastructure: data generator, feature extraction, model training scripts
- 461 tests covering all passes, IR, calibration, backends, ML pipeline
- CI/CD: GitHub Actions for lint, typecheck, test matrix (Python 3.10-3.12)
- 10 example scripts demonstrating key features
- Comprehensive benchmark suite comparing all ML phases
