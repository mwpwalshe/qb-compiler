# Changelog

All notable changes to [qb-compiler](https://qubitboost.io/compiler), the open-source quantum circuit compiler by [QubitBoost](https://qubitboost.io), will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-04-26

Live calibration end-to-end against real backends. The
``LiveCalibrationProvider`` path stops being a stub: it now delegates to
a working ``qubitboost_sdk.calibration.CalibrationHub``.

### Architectural scope of v0.5 — the "in-process tier"

What this release ships and explicitly does not ship:

- **Ships.** A `CalibrationHub` Python class that runs inside whatever
  process imports `qubitboost_sdk.calibration` (qb-compiler, a notebook,
  a script). On-demand fetch from IBM Quantum via
  `qiskit-ibm-runtime`. Per-user disk JSON cache at
  `~/.cache/qubitboost/calibration/` with 30-min TTL. Each consuming
  process holds its own hub instance and its own view of the disk
  cache.
- **Does not ship.** Background polling daemon. Redis or other shared
  cache. Cross-process synchronisation. FastAPI HTTP endpoint. Hardware
  Observatory page consumer. IBM API rate-limit budget shared across
  consumers (today, two parallel processes can both fetch the same
  backend within seconds of each other; for the demo this is fine).

The PM2-managed daemon, Redis cache, FastAPI endpoint, and Observatory
page are scoped at QubitBoost-internal ``sales/CALIBRATION_HUB_DESIGN.md``
and are scheduled for a follow-on release. They are not implied or
required by v0.5.

### Authentication

`LiveCalibrationProvider` (and `CalibrationHub`) authenticate to IBM via
a saved-credential profile resolved by `QiskitRuntimeService(name=...)`.
Default profile name is `"qubitboost_cloud"`. External users must save
their own IBM credentials before invoking the live provider::

    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(
        name="my_account", channel="ibm_quantum", token="...",
    )

then pass `account="my_account"` to either constructor. No tokens are
embedded in the package; no environment variable is read in v0.5; the
profile resolution is identical to plain `QiskitRuntimeService` use.

### Freshness contract (locked)

- **Cache age 0 to 30 min** (default `cache_ttl_minutes=30`):
  `get_latest` serves the cached snapshot. No IBM contact.
- **Cache age > 30 min**: `get_latest` attempts a fresh fetch. On
  success, the new snapshot replaces the cache and is returned. On
  failure (IBM unreachable, 5xx, network error), the previous stale
  cached snapshot is returned with a `UserWarning` of the form
  `"CalibrationHub: fresh fetch for {backend} failed (...); serving
  stale cache (age N min)"`.
- **`fetch()` always bypasses cache** and always performs a fresh vendor
  call. On failure it raises (no stale fallback). Use this for explicit
  pre-experiment freshness guarantees.
- **No upper bound on stale-fallback age inside the hub.** Callers
  needing a hard floor (e.g. demo harnesses that refuse to launch on
  >24h-old calibration) must enforce it themselves by reading
  `provider.timestamp` and comparing to `datetime.now(timezone.utc)`.

The 30-min TTL is chosen as a small fraction of IBM Heron-class devices'
~12h calibration cadence on Fez/Torino/Marrakesh/Kingston: long enough
to amortise IBM API traffic across multiple compilations of the same
circuit, short enough to catch event-driven recalibrations within ~30
min of IBM publishing them.

### Coverage improvement vs v0.4 fixture path

`LiveCalibrationProvider` returns snapshots with the full property
surface IBM exposes. For ibm_fez specifically:

- v0.4 fixture (hand-fetched 2026-03-14): 156 `qubit_properties` (T1,
  T2, readout_error_0to1, readout_error_1to0, frequency=None) +
  **352 `gate_properties`** (2-qubit ECR errors per coupling only).
- v0.5 live fetch: same 156 `qubit_properties` plus IBM-specific fields
  (`prob_meas0_prep1`, `prob_meas1_prep0`, `readout_length`) +
  **1796 `gate_properties`** (~5×: adds per-basis-gate single-qubit
  error rates for `id`, `sx`, `rz`, `x`).

The added 1444 entries are per-basis-gate single-qubit error rates
that v0.4's fixture-based path did not capture. Practical effect on
chain selection is small (single-qubit errors are typically much
smaller than 2-qubit), but the data is now complete.

### Field-format reconciliation

- The disk-cached JSON written by `CalibrationHub` is dual-format:
  legacy `readout_error_0to1` / `readout_error_1to0` aliases AND
  modern `prob_meas1_prep0` / `prob_meas0_prep1` fields are both
  present per qubit; gate parameters are written both flat
  (`gate_error: ...`) and nested under `parameters: {...}`.
- The provider snapshot materialised by `_provider_to_dict()` for
  `QBCalibrationLayout` consumption uses the legacy field-name
  convention internally (because qb-compiler's `BackendProperties`
  dataclass keeps only the legacy fields after parsing).
  `QBCalibrationLayout` consumes via the nested `parameters.gate_error`
  path, which `_provider_to_dict()` emits — verified working.

### Benchmarks

**v0.5 has not been re-benchmarked against Qiskit `optimization_level=3`
on the live (full-coverage) calibration path.** The "matches or exceeds
Qiskit's default on hardware-validated benchmarks" claim in the README
was established on the v0.4 partial-fixture path and remains the
baseline. v0.5's calibration data is a strict superset of v0.4's so the
claim should not regress, but no new benchmark numbers are asserted.
Re-benchmarking on the live path is scheduled for v0.5.1.

### Added
- `qb_transpile(..., calibration_provider=...)` accepts any
  `CalibrationProvider` instance directly, including the live one.
  Previously only `calibration_path` and `calibration_data` were
  supported.
- `_provider_to_dict` helper materialises a provider's snapshot into
  the calibration dict `QBCalibrationLayout` consumes — enables the
  live provider to drive the layout pass without bespoke wiring.
- `LiveCalibrationProvider(..., account="...")` parameter so external
  users can pass their saved-credential profile name. Default
  `"qubitboost_cloud"` for backward compatibility.

### Changed
- `LiveCalibrationProvider.refresh()` calls `hub.fetch()` directly to
  bypass the cache TTL. Previously it called `hub.get_latest()`, which
  would silently serve cached data within the TTL window — contradicting
  the docstring's "force re-fetch" promise.
- `LiveCalibrationProvider`'s `cache_ttl_minutes` is now propagated
  down to the hub it constructs, so both layers agree on freshness.
  v0.5 dev versions briefly defaulted `LiveCalibrationProvider` to 30
  min while the underlying hub defaulted to 60; this is fixed.
- `LiveCalibrationProvider` no longer raises `ImportError` when
  `qubitboost-sdk>=2.6` is installed. Earlier versions pointed at
  `qubitboost_sdk.calibration.CalibrationHub` which did not exist;
  v2.6 of the SDK ships that module.

### Honest disclosure on prior versions
- v0.1-v0.4 "calibration-aware" claims operated on real IBM backend
  properties (audit at QubitBoost-internal
  `sales/QB_COMPILER_FIXTURE_PROVENANCE.md`), but the shipped fixture
  files contained 2-qubit gate errors only and dropped per-basis-gate
  single-qubit error data. From v0.5 onwards the live fetch via
  CalibrationHub provides the full property surface. For chain
  selection the practical impact of the prior partial coverage was
  small (single-qubit errors are typically much smaller than 2-qubit
  errors), but v0.5 closes the gap.
- README claim "Calibration data can be loaded from local JSON files
  or fetched from vendor APIs" was technically incorrect on v0.4 and
  earlier (the vendor-API path raised `ImportError`). It is correct on
  v0.5 with `pip install qb-compiler[qubitboost]`.

## [0.4.0b1] - 2026-04-22

Beta release. Stim-validated only, no hardware runs yet. API may shift.

### Added
- `qb_compiler.ising`, first Qiskit-side integration for NVIDIA's
  `Ising-Decoder-SurfaceCode-1` model family (released 2026-04-14).
  Converts rotated-surface-code memory experiments (Qiskit or stim)
  into the 4-channel `(B, 4, T, D, D)` float32 tensor consumed by the
  pretrained decoder.  Public API: `SurfaceCodePatchSpec`,
  `build_ising_tensor`, `PyMatchingDecoder` (MWPM baseline),
  `IsingDecoderWrapper` (pre-decoder + residual-MWPM chain; users
  bring their own gated-HF weights + NVIDIA's Apache-2.0 model
  definition, qb-compiler does not vendor NVIDIA code or weights),
  `evaluate_logical_error_rate` harness.  Install via
  `pip install qb-compiler[ising]` for the PyMatching baseline,
  `qb-compiler[ising-nvidia]` to add torch + safetensors for the
  NVIDIA pre-decoder.  See `src/qb_compiler/ising/README.md`.
- Benchmark harness `benchmarks/ising/run_pymatching_sweep.py`
  sweeping `(distance, rounds, p_error, basis)` to establish the
  baseline any pre-decoder must beat.
- New optional extras: `ising`, `ising-nvidia`.

## [0.3.0] - 2026-04-16

### Added
- Qiskit SDK 2.x compatibility: `qiskit` dependency widened to `>=1.0,<3.0`.
- CI now runs the test suite against both Qiskit 1.4 and Qiskit 2.0 in matrix.
- `QBCalibrationLayoutPlugin`, proper `qiskit.transpiler.layout` stage plugin.
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
- `QBTranspilerPlugin.get_pass_manager(calibration_data=...)`, emits a
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
