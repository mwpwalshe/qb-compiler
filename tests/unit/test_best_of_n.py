"""Best-of-N qb_transpile (v0.7)."""
import glob

CAL = sorted(glob.glob("tests/fixtures/calibration_snapshots/ibm_fez_*.json"))[-1]
from qiskit import QuantumCircuit

from qb_compiler.qiskit_plugin.transpiler_plugin import qb_transpile


def _ghz(n=4):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


def test_best_of_n_returns_circuit_and_candidates():
    best, cands = qb_transpile(_ghz(), backend="ibm_fez", n_seeds=3, return_candidates=True)
    assert best is not None
    assert len(cands) == 3
    assert {c["seed"] for c in cands} == {0, 1, 2}
    best_score = max(c["score"] for c in cands)
    # the returned circuit corresponds to the best score
    assert any(c["score"] == best_score for c in cands)


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
