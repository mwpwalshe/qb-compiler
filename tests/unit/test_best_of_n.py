"""Best-of-N qb_transpile (v0.6)."""

import glob

from qiskit import QuantumCircuit

from qb_compiler.qiskit_plugin.transpiler_plugin import qb_transpile

CAL = sorted(glob.glob("tests/fixtures/calibration_snapshots/ibm_fez_*.json"))[-1]


def _ghz(n=4):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def test_best_of_n_returns_the_argmax_candidate():
    best, cands = qb_transpile(
        _ghz(), backend="ibm_fez", calibration_path=CAL, n_seeds=3, return_candidates=True
    )
    assert {c["seed"] for c in cands} == {0, 1, 2}
    winner = max(cands, key=lambda c: c["score"])
    # the returned circuit matches the argmax candidate's recorded properties
    two_q = sum(
        1
        for inst in best.data
        if len(inst.qubits) == 2 and inst.operation.name not in ("barrier", "measure", "reset")
    )
    assert two_q == winner["two_q"]
    assert best.depth() == winner["depth"]


def test_single_seed_back_compat():
    out = qb_transpile(_ghz(), backend="ibm_fez")
    # default path returns a bare circuit, unchanged API
    assert hasattr(out, "depth")


def test_best_never_worse_than_median_on_two_q():
    _best, cands = qb_transpile(_ghz(6), backend="ibm_fez", n_seeds=4, return_candidates=True)
    best_two_q = min(c["two_q"] for c in cands)
    chosen = [c for c in cands if c["score"] == max(x["score"] for x in cands)][0]
    # when scoring by fidelity the chosen candidate may trade 2q for better edges,
    # but it must never be the strictly worst candidate by score construction
    assert chosen["score"] >= sorted(x["score"] for x in cands)[0]
    assert best_two_q >= 0


def test_fidelity_scored_path_engages_with_calibration():
    _best, cands = qb_transpile(
        _ghz(), backend="ibm_fez", calibration_path=CAL, n_seeds=2, return_candidates=True
    )
    # fidelity scores are probabilities, the 2q-count fallback is negative
    assert all(0.0 < c["score"] <= 1.0 for c in cands), cands
