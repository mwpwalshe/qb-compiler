# Notebook execution status

Every notebook here was executed headless, in order, with

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
| `01_preflight_viability.ipynb` | yes | 212s | opener trimmed |
| `02_compilation_receipts.ipynb` | yes | 48s | opener trimmed, one printed string reworded |
| `03_multi_vendor_ranking.ipynb` | yes | 79s | opener trimmed |
| `04_dynamical_decoupling.ipynb` | yes | 49s | opener trimmed |
| `05_fidelity_estimation.ipynb` | yes | 19s | prose dashes removed |
| `06_cost_estimation.ipynb` | yes | 21s | prose dashes removed |
| `07_calibration_data.ipynb` | yes | 10s | opener trimmed, prose dashes removed, two printed strings reworded |
| `08_qiskit_integration.ipynb` | yes | 13s | opener trimmed, prose dashes removed |
| `09_ml_layout_prediction.ipynb` | yes | 77s |  |
| `10_circuit_ir.ipynb` | yes | 29s | opener trimmed |
| `11_compilation_strategies.ipynb` | yes | 47s |  |
| `12_error_handling.ipynb` | yes | 46s |  |
| `13_qubitboost_integration.ipynb` | yes | 17s | opener trimmed, gate registry listing re-executed against the current registry |
| `14_backend_deep_dive.ipynb` | yes | 32s |  |
| `15_cli_workflows.ipynb` | yes | 360s |  |
| `16_real_world_pipelines.ipynb` | yes | 42s |  |
| `17_nvidia_ising_integration.ipynb` | yes | 34s |  |
| `18_ising_pymatching_baseline_sweep.ipynb` | yes | 27s |  |
| `19_know_before_you_run.ipynb` | yes | 43s |  |
| `20_receipts_not_claims.ipynb` | yes | 36s |  |
| `21_observablegate_qec_preflight.ipynb` | yes | 9s |  |
| `22_willow_complementary_gap_validation.ipynb` | yes | 174s |  |
| `23_selection_receipts.ipynb` | yes | 10s | rewritten on a real IBM Fez snapshot: executed layout, calibration age, ranked alternatives, Qiskit head to head |
| `24_multi_platform_calibration.ipynb` | yes | 16s |  |
| `25_backend_discovery.ipynb` | yes | 22s |  |
| `26_circuit_interop.ipynb` | yes | 28s |  |
| `27_cross_vendor_advice.ipynb` | yes | 21s |  |
| `28_signing_and_verification.ipynb` | yes | 10s | new |
| `29_chem_audit.ipynb` | yes | 8s | new |
| `30_qec_corpus_verification.ipynb` | yes | 7s | new |

30 of 30 execute clean with no home directory paths in any output.

## What changed and why

**23** was rewritten. It used to assert that the receipt could not disagree with what ran,
which was the exact defect fixed in 0.12.0: the receipt described the recommendation, not the
executed layout. It now runs on a real IBM Fez calibration snapshot from `tests/fixtures`,
shows the override case with its score penalty, prints the calibration age the choice was made
from, and shows the ranked alternatives including the two that were the same physical qubits
with the logical labels permuted. It also scores Qiskit's own calibration aware layout on the
same objective and says plainly that scoring both with our own objective is not evidence about
hardware.

**28, 29 and 30** are new: signing and offline verification end to end, the Hamiltonian audit
on the real H2 fixture with each check watched failing, and corpus digest verification with a
clean pass and a clean failure. None of them needs credentials or a network.

The rest of the pass removed boilerplate openers and dashes from prose, and re-executed
everything so no output is older than the code that produced it.
