# Notebook execution status

Every notebook here was executed headless, in order, one at a time, with

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=400 <notebook>
```

Environment: qiskit with aer, stim, pymatching, torch and the qubitboost SDK all present, so
the optional paths run rather than skip. Notebooks that need an extra guard the import and
print what they skipped instead of failing when it is absent.

The checks are: does it execute without an error output, and does any output contain an
absolute home directory path. Both have to be clean.

| Notebook | Clean | Time | Changed this pass |
|---|---|---|---|
| `01_preflight_viability.ipynb` | yes | 63s | price table warning filtered |
| `02_compilation_receipts.ipynb` | yes | 19s |  |
| `03_multi_vendor_ranking.ipynb` | yes | 27s | price table warning filtered |
| `04_dynamical_decoupling.ipynb` | yes | 10s |  |
| `05_fidelity_estimation.ipynb` | yes | 4s |  |
| `06_cost_estimation.ipynb` | yes | 4s | price table warning filtered |
| `07_calibration_data.ipynb` | yes | 4s |  |
| `08_qiskit_integration.ipynb` | yes | 5s | price table warning filtered |
| `09_ml_layout_prediction.ipynb` | yes | 10s |  |
| `10_circuit_ir.ipynb` | yes | 5s |  |
| `11_compilation_strategies.ipynb` | yes | 8s | price table warning filtered |
| `12_error_handling.ipynb` | yes | 8s | price table warning filtered |
| `13_qubitboost_integration.ipynb` | yes | 9s | price table warning filtered |
| `14_backend_deep_dive.ipynb` | yes | 4s | price table warning filtered |
| `15_cli_workflows.ipynb` | yes | 68s |  |
| `16_real_world_pipelines.ipynb` | yes | 14s | price table warning filtered |
| `17_nvidia_ising_integration.ipynb` | yes | 8s |  |
| `18_ising_pymatching_baseline_sweep.ipynb` | yes | 6s |  |
| `19_know_before_you_run.ipynb` | yes | 7s | price table warning filtered |
| `20_receipts_not_claims.ipynb` | yes | 8s | price table warning filtered |
| `21_observablegate_qec_preflight.ipynb` | yes | 5s |  |
| `22_willow_complementary_gap_validation.ipynb` | yes | 94s |  |
| `23_selection_receipts.ipynb` | yes | 5s |  |
| `24_multi_platform_calibration.ipynb` | yes | 6s |  |
| `25_backend_discovery.ipynb` | yes | 10s | price table warning filtered |
| `26_circuit_interop.ipynb` | yes | 11s | price table warning filtered |
| `27_cross_vendor_advice.ipynb` | yes | 11s | price table warning filtered, in the subprocess environment too |
| `28_signing_and_verification.ipynb` | yes | 5s |  |
| `29_chem_audit.ipynb` | yes | 4s |  |
| `30_qec_corpus_verification.ipynb` | yes | 4s |  |
| `31_record_validation.ipynb` | yes | 8s | new |
| `32_record_residual.ipynb` | yes | 15s | new |

32 of 32 execute clean with no home directory paths in any output.

## What changed and why

**31 and 32 are new.** 31 builds a repetition code memory record in software, runs the nine
construction checks on it, then reads the same shots with their rounds backwards and watches the
round profile, the endpoint agreement and the state profile find it. It also shows what the time
mirror control does and does not catch, checks the model fingerprint against its closed form, and
round trips the record through the `.npz` contract and the command line. 32 measures what a
feature block adds over a decoder's own output, in bits per shot, against a geometry null and a
shot permutation null, and shows a feature that carries something and one that does not. Both have
an optional section that runs against a real record when one is on the machine and skips politely
when it is not. Neither needs credentials or a network.

**Fourteen notebooks had a warning filtered.** The package warns when its price table is more than
90 days past review. The table was last reviewed on 12 June 2026, so the warning began firing on
10 September 2026, and Python prints the emitting module's absolute path along with it. That put a
home directory path into the output of every notebook that estimates a cost. The filter is in the
notebook, in the same setup cell that already filters the urllib3 import warning, and neither the
price table nor the library was changed: the warning is correct and every user sees it until the
table is reviewed again. Notebook 27 needed the filter in `PYTHONWARNINGS` as well, because it
calls `qbc` as a subprocess and a subprocess does not inherit the in process filter.
