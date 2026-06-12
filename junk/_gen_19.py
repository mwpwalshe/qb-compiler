"""Generate notebooks/19_know_before_you_run.ipynb (demo authoring helper)."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# 19. Know Before You Run

## 1. The pain

You are about to spend money on a quantum circuit. What do you actually know? Most submissions go out with no fidelity estimate, no error budget, and no shot plan: the first real data point is the invoice. This notebook replaces that with numbers you can read before paying.""")

code("""import os
import tempfile
import warnings

# Keep the demo self-contained: any local records go to a temp dir,
# not ~/.qb_compiler. Delete this line to keep a real history.
os.environ["QBC_DATA_DIR"] = tempfile.mkdtemp(prefix="qbc_demo_")

# Silence an unrelated urllib3 version warning from this environment.
warnings.filterwarnings("ignore", message=r"urllib3")

import qb_compiler

print("qb-compiler", qb_compiler.__version__)""")

md("""## 2. Preflight a GHZ-6 on ibm_fez

`check_viability` transpiles the circuit against the bundled ibm_fez calibration snapshot and prints everything it can know before submission.""")

code("""from qiskit import QuantumCircuit

from qb_compiler import check_viability

ghz = QuantumCircuit(6, name="GHZ-6")
ghz.h(0)
for i in range(5):
    ghz.cx(i, i + 1)
ghz.measure_all()

pre = check_viability(ghz, backend="ibm_fez")
print(pre)""")

md("""Two parts of that block do the real work.

**The error budget says where the fidelity goes.** For this circuit, readout accounts for roughly 94 percent of the predicted loss and two-qubit gates for about 6 percent. If you want this circuit to do better, work on readout (qubit choice, measurement mitigation), not on gate count.

**The estimate carries its own accuracy band.** 0.84 +-0.05 typical error. That band comes from n=6 predicted-vs-measured hardware pairs (IBM Fez, GHZ family, March 2026), and on that set the model tends to be optimistic. Small validation set, one backend, one circuit family: treat it as indicative. Every number downstream of this estimate inherits that band.

Also note the stderr warning above: the price table is dated, and the package says so rather than presenting stale prices as fresh.""")

md("""## 3. Calibration freshness

An estimate is only as current as the calibration behind it, so the result dates its inputs.""")

code("""print(f"Calibration snapshot age: {pre.calibration_age_days:.0f} days")
print()
for s in pre.suggestions:
    print("-", s)""")

md("""The bundled fixture is from March 2026, months old, and the staleness suggestion fires. That is the point: the tool tells you the estimate reflects the snapshot date instead of letting you assume it is live. Point `QBC_CALIBRATION_DIR` at fresh snapshots (or pass `backend_props`) for current numbers.""")

md("""## 4. The shot bill before you pay it

The preflight implies a per-run error rate of about 1 minus the predicted fidelity. How many shots does it take to actually resolve that rate, and what does that cost? Textbook sample-size formulas, exposed where the spending decision happens.""")

code("""from qb_compiler.cost.pricing import get_pricing
from qb_compiler.cost.shot_budget import shots_for_expectation, shots_for_rate

error_rate = 1.0 - pre.estimated_fidelity
per_shot = get_pricing("ibm_fez").cost_per_shot_usd

print(f"Implied per-run error rate: {error_rate:.4f}")
print()
for rel in (0.5, 0.2, 0.1):
    n = shots_for_rate(error_rate, rel_width=rel)
    print(f"  resolve it to +-{int(rel * 100):>2d}% relative: {n:>6,} shots  (~${n * per_shot:.2f})")

n_exp = shots_for_expectation(0.02)
print()
print(f"  expectation value to +-0.02 (worst-case bound): {n_exp:,} shots  (~${n_exp * per_shot:.2f})")""")

md("""Resolving the implied error rate to +-10 percent relative needs about 2,000 shots, roughly 32 cents at the listed Fez rate. The expectation-value figure uses the worst-case variance bound, so real circuits often need fewer: it is the budget-safe upper estimate, not a prediction.""")

md("""## 5. Where is your dollar best spent

`rank_value` runs the same preflight across every backend with a loadable calibration fixture and ranks by predicted fidelity per dollar.""")

code("""from qb_compiler import rank_value
from qb_compiler.windows import format_table

rows = rank_value(ghz)
print(format_table(rows))
print()
for row in rows:
    print(f"{row.backend}: {row.trend_detail}")
    print()""")

md("""Read the `Fid/$` column: on this circuit ibm_torino delivers about 1.5 units of predicted fidelity per dollar against 1.3 for ibm_fez, at the listed prices.

The `Trend` column is deliberately naive: a least-squares slope through the median two-qubit error of the dated calibration snapshots on disk. It does not forecast tomorrow and it does not schedule jobs. ibm_fez reads improving across its three snapshots; ibm_torino has only one snapshot on disk, so the honest answer there is unknown, and the table says so.""")

md("""## 6. Best-of-N with proof

Qiskit's transpiler is stochastic: different seeds give different layouts and different routed circuits. A linear GHZ chain routes the same way under every seed, so for this demo we use a QFT-6, where routing has real choices to make. `qb_transpile(n_seeds=5, return_candidates=True)` runs all five seeds, scores each candidate with the calibration-aware fidelity estimate, and returns the winner plus the evidence.""")

code("""import math
from pathlib import Path

from qb_compiler.qiskit_plugin import qb_transpile

qft = QuantumCircuit(6, name="QFT-6")
for i in range(6):
    qft.h(i)
    for j in range(i + 1, 6):
        qft.cp(math.pi / (2 ** (j - i)), i, j)
qft.measure_all()

# Latest bundled ibm_fez snapshot (resolved from the package location,
# so this works regardless of the notebook's working directory).
repo = Path(qb_compiler.__file__).resolve().parents[2]
cal = repo / "tests" / "fixtures" / "calibration_snapshots" / "ibm_fez_2026_03_14.json"
best, candidates = qb_transpile(
    qft,
    backend="ibm_fez",
    calibration_path=cal,
    n_seeds=5,
    return_candidates=True,
)

print(f"{'seed':>4} {'2q gates':>9} {'depth':>6} {'pred. fidelity':>15}")
for c in candidates:
    print(f"{c['seed']:>4} {c['two_q']:>9} {c['depth']:>6} {c['score']:>15.4f}")

scores = [c["score"] for c in candidates]
winner = max(candidates, key=lambda c: c["score"])
print()
print(f"winner: seed {winner['seed']} at {winner['score']:.4f}")
print(f"spread between best and worst seed: {max(scores) - min(scores):.4f}")""")

md("""The spread between the best and worst seed is about 0.07 in predicted fidelity, on a five-seed run that costs nothing but local compute. Note what won: the seed-2 candidate carries 48 two-qubit gates against 46 for the others, yet scores highest, because the calibration-aware estimate saw it land on better-calibrated edges. Picking by gate count alone would have chosen a worse circuit. Seed luck deleted, evidence attached: the candidates list is exactly what goes into a compilation receipt (notebook 20).""")

md("""## 7. QEC bonus: preflight an error-correction experiment

The same discipline applies to surface-code memory experiments, which this package can preflight before any hardware time: a stim simulation at your physical-error proxy, decoded with PyMatching, summarised as sizing signals.""")

code("""from qb_compiler import qec_preflight

report = qec_preflight(3, 3, physical_error=0.01)
print(report)""")

md("""A d=3, 3-round memory at physical error 0.01 projects a logical error rate of about 4.0e-2, with a Wilson 95 percent band attached. The shots table is the budget conversation: about 370 shots to resolve that LER to +-50 percent relative, about 9,200 to reach +-10 percent. The projection comes from a uniform depolarizing proxy and the notes say so: real devices have structured noise the proxy cannot capture, so this is a sizing aid, not a forecast.""")

md("""## 8. Close

Everything above ran locally, against on-disk calibration data, for zero QPU spend: a dated fidelity estimate with an honest band, an error budget, a shot bill, a per-dollar backend ranking, a seed tournament with evidence, and a QEC sizing report. These are all free signals from `pip install qb-compiler` (the QEC preflight needs the `[ising]` extra). What you do with them, thresholds, gating, policy, is your call. Notebook 20 covers the other half: verifying the predictions and keeping receipts.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python"}
nbf.write(nb, "notebooks/19_know_before_you_run.ipynb")
print("written", len(cells), "cells")
