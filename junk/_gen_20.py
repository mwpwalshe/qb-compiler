"""Generate notebooks/20_proof_not_promises.ipynb (demo authoring helper)."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# 20. Proof, Not Promises

## 1. The pain

Every fidelity estimate you have ever been shown was a bare number: no error bar, no statement of where it was validated, and no record of whether the last one turned out to be right. This notebook closes that loop three ways: the estimate ships with its measured accuracy band, you can check it against a mirror-circuit run, and every compile leaves a receipt that future compiles are compared against.""")

code("""import os
import tempfile
import warnings

# Keep the demo self-contained: verify records and receipts go to a temp dir,
# not ~/.qb_compiler. Delete this line to keep a real local history.
os.environ["QBC_DATA_DIR"] = tempfile.mkdtemp(prefix="qbc_demo_")

# Silence an unrelated urllib3 version warning from this environment.
warnings.filterwarnings("ignore", message=r"urllib3")

import qb_compiler

print("qb-compiler", qb_compiler.__version__)""")

md("""## 2. The band

Start with a plain preflight on a Bell pair targeting ibm_fez.""")

code("""from qiskit import QuantumCircuit

from qb_compiler import check_viability

bell = QuantumCircuit(2, name="Bell")
bell.h(0)
bell.cx(0, 1)
bell.measure_all()

pre = check_viability(bell, backend="ibm_fez")
print(pre)""")

md("""Look at the fidelity line: the estimate never appears alone. The +-0.05 typical error comes from n=6 predicted-vs-measured hardware pairs (IBM Fez, GHZ family, March 2026), and on that set the model tends to be optimistic. Small validation set, one backend, one circuit family: indicative, not a guarantee. Stating that is the whole point. A number with a measured band and a known bias is something you can act on; a bare number is a promise.""")

md("""## 3. Verify mode

A band is only honest if you can test it. `build_mirror` strips the final measurements, appends the inverse of the circuit, and measures everything: under noiseless execution the mirror returns all zeros with probability 1, so the all-zeros rate is a success proxy related to (not equal to) the squared fidelity of the original circuit.""")

code("""from qb_compiler import build_mirror

mirror = build_mirror(bell)
print(mirror.draw(output="text"))""")

code("""from qb_compiler import verify_viability

result = verify_viability(bell, "aer", backend="ibm_fez", shots=1024)
print(result)""")

md("""Read the comparison carefully. The prediction models ibm_fez hardware, but the runner here is a noiseless Aer simulator, so measured 1.0000 against predicted^2 0.8892 is exactly the right answer: the +0.11 discrepancy is the noise the model expects Fez to add and Aer does not have. Point the same call at real hardware (any SamplerV2-style runner works as the second argument) and the discrepancy becomes a genuine accuracy check on the model. One caveat stays either way: the mirror proxy can be optimistic when coherent errors echo out between the circuit and its inverse, and the docstring says so.""")

md("""## 4. The accuracy record

Every verify call appends a predicted-vs-measured pair to a local log. Run two more circuits, then summarise.""")

code("""def make_ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"GHZ-{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


print(verify_viability(make_ghz(4), "aer", backend="ibm_fez", shots=1024))
print(verify_viability(make_ghz(6), "aer", backend="ibm_fez", shots=1024))""")

code("""from qb_compiler.verify import accuracy_summary

summary = accuracy_summary()
for k, v in summary.items():
    print(f"{k}: {v}")""")

md("""Three records, median absolute discrepancy about 0.23, all positive because every run so far was noiseless: on this simulator the summary measures the hardware noise the model prices in, not the model's error. Feed it hardware runs and the same summary becomes your own accuracy ledger for the fidelity model, built from your circuits on your backends. The record grows with use and stays on your machine: a plain JSONL file under `QBC_DATA_DIR` (default `~/.qb_compiler`), nothing transmitted anywhere.""")

md("""## 5. Receipts

The third loop: every compile can leave a passport. `make_receipt` captures what was compiled, for which backend, what was predicted (band attached), where the error went, how old the calibration was, and which tool versions produced all of it.""")

code("""import json

from qb_compiler import make_receipt, record_receipt

ghz6 = make_ghz(6)
pre6 = check_viability(ghz6, backend="ibm_fez")

receipt = make_receipt(ghz6, pre6, backend="ibm_fez")
print(receipt)
print()
print(json.dumps(receipt.to_json_dict(), indent=2))""")

md("""The structural hash fingerprints the workload, deliberately coarsely: parameter values are excluded, so two iterations of the same ansatz hash identically and can be compared across sessions. Check the receipt against history, then record it.""")

code("""from qb_compiler.receipts import regression_check

report = regression_check(receipt)
print(report.status)
print(report.message)

record_receipt(receipt)""")

md("""## 6. Regression watch

`NO_BASELINE`: first sighting of this workload, so it becomes the baseline. Now simulate next week. We fabricate a second receipt with `dataclasses.replace`, dropping the predicted fidelity by 0.15: this stands in for what a drifted calibration snapshot would produce for the same circuit, without waiting a week for real drift.""")

code("""import dataclasses

degraded = dataclasses.replace(
    receipt,
    predicted_fidelity=receipt.predicted_fidelity - 0.15,
    notes=["fabricated for the demo: simulates a drifted calibration snapshot"],
)

report = regression_check(degraded)
print(report.status)
print(report.message)""")

md("""`REGRESSION`: the 0.15 drop exceeds the combined +-0.10 band of the two estimates, so it is flagged as a real change rather than estimate noise. A drop inside the band would read `STABLE`: the bands from section 2 are what keep this check from crying wolf. And note what did not happen: nothing was blocked, no threshold was applied, no job was cancelled. Your compilation has a memory; you decide what to do with it.""")

md("""## 7. Close

Bands, verify mode, receipts, and regression watch are free signals in `pip install qb-compiler` and stay on your machine. The layer that turns them into signed certificates and execution warranties is the QubitBoost SDK.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python"}
nbf.write(nb, "notebooks/20_proof_not_promises.ipynb")
print("written", len(cells), "cells")
