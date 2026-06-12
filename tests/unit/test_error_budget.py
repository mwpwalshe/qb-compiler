"""Error-budget breakdown + fidelity band + pricing staleness (v0.5.3 additions)."""

import warnings

import pytest
from qiskit import QuantumCircuit

from qb_compiler.viability import check_viability


def _bell():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def test_error_budget_present_and_sane():
    res = check_viability(_bell(), backend="ibm_fez", n_seeds=2)
    assert res.error_budget is not None
    assert set(res.error_budget) == {"two_qubit_gates", "readout"}
    for v in res.error_budget.values():
        assert 0.0 <= v <= 1.0
    # the product of per-source survivals reproduces the estimate
    surv = 1.0
    for v in res.error_budget.values():
        surv *= 1.0 - v
    assert abs(surv - res.estimated_fidelity) < 1e-3


def test_fidelity_band_attached_and_rendered():
    res = check_viability(_bell(), backend="ibm_fez", n_seeds=2)
    assert res.fidelity_typical_abs_error is not None
    text = str(res)
    assert "+-" in text and "Error budget" in text


def test_pricing_staleness_warns_once():
    import qb_compiler.cost.pricing as pricing

    pricing._stale_warned = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pricing.get_pricing("ibm_fez")
        pricing.get_pricing("ibm_torino")
    msgs = [x for x in w if "price table" in str(x.message)]
    assert len(msgs) <= 1  # warn at most once per process
    if (
        pytest.importorskip("datetime").date.today()
        - __import__("datetime").date.fromisoformat(pricing.PRICING_AS_OF)
    ).days > 90:
        assert len(msgs) == 1
