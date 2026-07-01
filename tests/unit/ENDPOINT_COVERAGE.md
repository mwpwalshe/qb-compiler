# qb-compiler Endpoint Unit-Test Coverage (v0.8.0 release gate)

Inventory of every CLI command and every public API symbol in `qb_compiler.__all__`,
cross-referenced against `tests/unit/`. "Test added" = added in this pass (file-owned dir `tests/unit/`).

## CLI commands (`qbc <cmd>`, via `click.testing.CliRunner` on `qb_compiler.cli.main:cli`)

| Endpoint | Had test? | Test added |
|----------|-----------|------------|
| `preflight` | yes (`test_cli.py`, `test_golden_path.py`) | — |
| `analyze` | yes (`test_cli.py`) | — |
| `diff` | yes (`test_cli.py`) | — |
| `compile` | **no** (only library API) | `test_cli_compile.py` — happy path, no-backend, 3 strategies, invalid strategy, `-o` output, `--receipt`, missing-file, `--help` |
| `verify` | yes (`test_golden_path.py`, aer-gated) | — |
| `when` | yes (`test_golden_path.py`) | — |
| `doctor` | yes (`test_cli.py`, `test_golden_path.py`) | — |
| `info` | yes (`test_cli.py`, `test_golden_path.py`) | — |
| `calibration show` | yes (`test_cli.py`) | — |
| `dem-audit` (NEW) | **no** | `test_cli_dem.py` — FAIL→exit2, FAIL+strict→exit2, PASS→exit0, WARN→exit0, WARN+strict→exit1, `-o` canonicalize output, missing-file, `--help` |
| `dem-canonicalize` (NEW) | **no** | `test_cli_dem.py` — writes `-o` output, required-`-o`, PASS roundtrip, `--help` |

## Public API (`qb_compiler.__all__`)

| Endpoint | Had test? | Test added |
|----------|-----------|------------|
| compiler: `QBCompiler`, `QBCircuit`, `CompileResult`, `EnhancedCompileResult`, `GateOp`, `NoiseModel`, `PassManager`, `BasePass`, `PassResult`, `CalibrationProvider`, `CostEstimate`, `CostEstimator` | yes (`test_compiler.py`, `test_cost/`, `test_passes/`) | — |
| config: `CompilerConfig`, `BackendSpec`, `BACKEND_CONFIGS` | yes (`test_backends/`, `test_compiler.py`) | — |
| discovery: `discover_backends`, `rank_discovered`, `check_viability_pub`, `DiscoveredBackend` | yes (`test_discovery.py`) | — |
| `qec_preflight`, `QECPreflightResult` | yes (`test_qec_preflight.py`, `test_golden_path.py`) | — |
| receipts: `make_receipt`, `record_receipt`, `receipt_history`, `regression_check`, `CompilationReceipt`, `RegressionReport` | yes (`test_receipts.py`) | — |
| recommender: `BackendRecommender`, `RecommendationReport` | yes (`test_recommender.py`) | — |
| verify: `build_mirror`, `run_mirror`, `verify_viability`, `accuracy_summary`, `MirrorResult`, `VerifyResult` | yes (`test_verify.py`) | — |
| viability: `check_viability`, `ViabilityResult` | yes (`test_viability.py`) | — |
| windows: `calibration_trend`, `rank_value`, `BackendValue` | yes (`test_windows.py`) | — |
| exceptions (8 classes) | yes (raised across suites) | — |
| `passmanager` (factory) | **no** | `test_passmanager.py` — backend-name string, `optimization_level`, unknown backend, `None` fallback, runs on a circuit (qiskit-gated) |
| `__version__` | yes (`info`/`--version`) | — |

## observable_gate API (`qb_compiler.observable_gate`)

| Endpoint | Had test? | Test added |
|----------|-----------|------------|
| `audit_matrices` | yes (`test_observable_gate.py`) | — |
| `audit_dem` | yes (`test_observable_gate.py`) | — |
| `canonicalize_dem` | yes (`test_observable_gate.py`) | — |
| `preflight_dem_gate` | yes (`test_observable_gate.py`) | — |
| `ObservableAuditResult` | yes (`test_observable_gate.py`) | — |
| `ObservableMaskCollapseError` | yes (`test_observable_gate.py`) | — |
| `dem_to_matrices` (internal) | indirectly (`test_observable_gate.py`) | — |

## Notes
- stim-dependent tests use `pytest.importorskip("stim")`; qiskit-dependent use `pytest.importorskip("qiskit")`.
- WARN fixture for the `dem-audit --strict` (exit 1) path is built inline: a detector-identical/
  logical-distinct mixed group plus a decomposed (separator `^`) mechanism → status WARN.
- No `src/` edits made (file-ownership constraint). No bugs found in the audited endpoints.
