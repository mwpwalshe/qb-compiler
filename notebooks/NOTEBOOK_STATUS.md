# Notebook Execution Status

Release-quality execution pass. Every notebook in `notebooks/` was run headless with

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=200 <nb>
```

Environment: stim 1.15.0, pymatching 2.3.1, qiskit + qiskit-aer, torch, cudaq, and
qubitboost-sdk 0.2.0 all installed (the `[ising]`, `gnn`, and `qubitboost` extras are present).
Notebooks that depend on an optional extra guard the import and print a clear
`skipped: requires …` message instead of crashing when the extra is absent.

| Notebook | Executes clean? | Time | Action taken |
|----------|-----------------|------|--------------|
| 01_preflight_viability.ipynb | PASS (fixed) | ~190s | **Was broken**, the GHZ 2→30 sweep cell exceeded the cell timeout (n=30 routing ≈80s at the default `n_seeds=10`). Added `n_seeds=1` to the batch-screening sweep. GHZ maps to a line so routing is trivial and the fidelity numbers are unchanged; cell now ≈25s. Re-executed in place. |
| 02_compilation_receipts.ipynb | PASS | 61s | none |
| 03_multi_vendor_ranking.ipynb | PASS | 109s | none |
| 04_dynamical_decoupling.ipynb | PASS | 29s | none |
| 05_fidelity_estimation.ipynb | PASS | 10s | none |
| 06_cost_estimation.ipynb | PASS | 13s | none |
| 07_calibration_data.ipynb | PASS | 19s | none |
| 08_qiskit_integration.ipynb | PASS | 18s | none |
| 09_ml_layout_prediction.ipynb | PASS | 59s | none |
| 10_circuit_ir.ipynb | PASS | 13s | none |
| 11_compilation_strategies.ipynb | PASS | 69s | none |
| 12_error_handling.ipynb | PASS | 26s | none |
| 13_qubitboost_integration.ipynb | PASS | 23s | none, `qubitboost` import already guarded |
| 14_backend_deep_dive.ipynb | PASS | 8s | none |
| 15_cli_workflows.ipynb | PASS | 172s | none |
| 16_real_world_pipelines.ipynb | PASS | 188s | none, `qubitboost` import already guarded |
| 17_nvidia_ising_integration.ipynb | PASS | 27s | none, needs `[ising]`; NVIDIA-weights path already guarded (raises/handles `NotImplementedError`) |
| 18_ising_pymatching_baseline_sweep.ipynb | PASS | 17s | none, reads a precomputed benchmark JSON; no extra needed at runtime |
| 19_know_before_you_run.ipynb | PASS | 21s | none, `qec_preflight` needs `[ising]` |
| 20_proof_not_promises.ipynb | PASS | 33s | none |
| 21_observablegate_qec_preflight.ipynb | PASS (new) | ~15s | **New notebook** demoing the v0.8.0 ObservableGate feature. Every stim/CLI cell guards on `HAVE_STIM` / `shutil.which("qbc")` and skips gracefully if the `[ising]` extra or console script is absent. |

**Result: all 21 notebooks execute cleanly** (or skip-gracefully when an optional extra is
absent). The only notebook that was broken was 01 (cell timeout), now fixed.
