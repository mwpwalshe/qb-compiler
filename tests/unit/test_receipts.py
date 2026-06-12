"""Tests for compilation receipts and regression watch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit

from qb_compiler.receipts import (
    CompilationReceipt,
    RegressionReport,
    make_receipt,
    receipt_history,
    record_receipt,
    regression_check,
    structural_hash,
)


@pytest.fixture(autouse=True)
def _isolated_store(monkeypatch, tmp_path):
    """Point the JSONL store at a throwaway directory for every test."""
    monkeypatch.setenv("QBC_DATA_DIR", str(tmp_path))


def _bell() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2, name="Bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(range(2), range(2))
    return qc


def _viability(fidelity: float, band: float | None = 0.05) -> SimpleNamespace:
    return SimpleNamespace(
        estimated_fidelity=fidelity,
        error_budget={"two_qubit_gates": 0.01, "readout": 0.02},
        fidelity_typical_abs_error=band,
        calibration_age_days=1.5,
        two_qubit_gate_count=1,
        depth=4,
    )


# ── structural hash ──────────────────────────────────────────────


class TestStructuralHash:
    def test_stable_across_identical_structures(self):
        h1 = structural_hash(_bell())
        h2 = structural_hash(_bell())
        assert h1 == h2
        assert len(h1) == 16
        int(h1, 16)  # valid hex

    def test_differs_for_different_structure(self):
        qc = QuantumCircuit(3, 3, name="GHZ")
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure(range(3), range(3))
        assert structural_hash(_bell()) != structural_hash(qc)

    def test_ignores_circuit_name(self):
        a = _bell()
        b = _bell()
        b.name = "something_else"
        assert structural_hash(a) == structural_hash(b)


# ── receipt assembly ─────────────────────────────────────────────


class TestMakeReceipt:
    def test_fields_from_circuit_and_viability(self):
        receipt = make_receipt(_bell(), _viability(0.9), backend="ibm_fez", seed=7, layout=[0, 1])
        assert receipt.backend == "ibm_fez"
        assert receipt.circuit_name == "Bell"
        assert receipt.n_qubits == 2
        assert receipt.depth_in == _bell().depth()
        assert receipt.post_2q_count == 1
        assert receipt.post_depth == 4
        assert receipt.predicted_fidelity == 0.9
        assert receipt.fidelity_typical_abs_error == 0.05
        assert receipt.calibration_age_days == 1.5
        assert receipt.seed == 7
        assert receipt.layout == [0, 1]
        assert receipt.structural_hash == structural_hash(_bell())
        assert receipt.qb_compiler_version  # never empty, "dev" fallback

    def test_to_json_dict_round_trips(self):
        receipt = make_receipt(_bell(), _viability(0.9), backend="ibm_fez")
        d = receipt.to_json_dict()
        assert d["predicted_fidelity"] == 0.9
        assert CompilationReceipt(**d) == receipt

    def test_str_is_compact_single_line(self):
        text = str(make_receipt(_bell(), _viability(0.9), backend="ibm_fez"))
        assert "\n" not in text
        assert "Bell" in text
        assert "ibm_fez" in text


# ── regression watch ─────────────────────────────────────────────


class TestRegressionCheck:
    def test_no_baseline_on_first_receipt(self):
        receipt = make_receipt(_bell(), _viability(0.9), backend="ibm_fez")
        report = regression_check(receipt)
        assert isinstance(report, RegressionReport)
        assert report.status == "NO_BASELINE"
        assert report.baseline is None
        assert report.fidelity_delta is None

    def test_regression_when_drop_exceeds_band(self):
        first = make_receipt(_bell(), _viability(0.90), backend="ibm_fez")
        record_receipt(first)
        # Combined band 0.05 + 0.05 = 0.10; drop of 0.30 exceeds it
        second = make_receipt(_bell(), _viability(0.60), backend="ibm_fez")
        record_receipt(second)

        report = regression_check(second)
        assert report.status == "REGRESSION"
        assert report.baseline is not None
        assert report.fidelity_delta == pytest.approx(-0.30)
        assert report.band_used == pytest.approx(0.10)
        assert first.timestamp in report.message

    def test_stable_when_drop_within_band(self):
        first = make_receipt(_bell(), _viability(0.90), backend="ibm_fez")
        record_receipt(first)
        # Drop of 0.05 sits inside the combined 0.10 band
        second = make_receipt(_bell(), _viability(0.85), backend="ibm_fez")
        record_receipt(second)

        report = regression_check(second)
        assert report.status == "STABLE"
        assert report.fidelity_delta == pytest.approx(-0.05)

    def test_improvement_when_rise_exceeds_band(self):
        record_receipt(make_receipt(_bell(), _viability(0.60), backend="ibm_fez"))
        second = make_receipt(_bell(), _viability(0.90), backend="ibm_fez")
        report = regression_check(second)
        assert report.status == "IMPROVEMENT"

    def test_baseline_scoped_to_backend(self):
        record_receipt(make_receipt(_bell(), _viability(0.90), backend="ibm_torino"))
        second = make_receipt(_bell(), _viability(0.60), backend="ibm_fez")
        report = regression_check(second)
        assert report.status == "NO_BASELINE"

    def test_two_q_delta_reported(self):
        record_receipt(make_receipt(_bell(), _viability(0.90), backend="ibm_fez"))
        worse = _viability(0.60)
        worse.two_qubit_gate_count = 5
        report = regression_check(make_receipt(_bell(), worse, backend="ibm_fez"))
        assert report.two_q_delta == 4


# ── history ──────────────────────────────────────────────────────


class TestReceiptHistory:
    def test_empty_when_nothing_recorded(self):
        assert receipt_history() == []

    def test_filters_by_hash_and_backend(self):
        bell = _bell()
        ghz = QuantumCircuit(3, 3, name="GHZ")
        ghz.h(0)
        ghz.cx(0, 1)
        ghz.cx(1, 2)
        ghz.measure(range(3), range(3))

        record_receipt(make_receipt(bell, _viability(0.9), backend="ibm_fez"))
        record_receipt(make_receipt(bell, _viability(0.8), backend="ibm_torino"))
        record_receipt(make_receipt(ghz, _viability(0.7), backend="ibm_fez"))

        assert len(receipt_history()) == 3
        assert len(receipt_history(structural_hash=structural_hash(bell))) == 2
        assert len(receipt_history(backend="ibm_fez")) == 2
        assert len(receipt_history(structural_hash=structural_hash(bell), backend="ibm_fez")) == 1
