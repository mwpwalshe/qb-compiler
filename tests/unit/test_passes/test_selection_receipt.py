# SPDX-License-Identifier: Apache-2.0
"""Selection receipt built from real CalibrationMapper output.

The receipt is derived purely from the mapper's PassResult metadata, so these tests run the actual
mapper and check the receipt faithfully reflects its chosen layout, score and breakdown.

The executed-layout tests exist because of a live defect: a campaign ran layout [6,5,4,3,2] and
got a receipt naming {144,143,136,123,124}, a mapping that never executed. Anything here asserting
that the receipt describes what ran is guarding against that returning.
"""

from __future__ import annotations

import functools

import pytest

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.calibration.models.coupling_properties import GateProperties
from qb_compiler.calibration.models.qubit_properties import QubitProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.mapping import (
    CalibrationMapper,
    calibration_fingerprint,
    calibration_freshness,
    selection_receipt,
)


def _backend(timestamp: str = "2026-03-12T00:00:00") -> BackendProperties:
    qubit_props = [
        QubitProperties(qubit_id=0, t1_us=300.0, t2_us=250.0, readout_error=0.005),
        QubitProperties(qubit_id=1, t1_us=200.0, t2_us=180.0, readout_error=0.010),
        QubitProperties(qubit_id=2, t1_us=150.0, t2_us=120.0, readout_error=0.030),
        QubitProperties(qubit_id=3, t1_us=100.0, t2_us=80.0, readout_error=0.050),
    ]
    gate_props = [
        GateProperties(gate_type="cz", qubits=(0, 1), error_rate=0.002, gate_time_ns=68.0),
        GateProperties(gate_type="cz", qubits=(1, 0), error_rate=0.002, gate_time_ns=68.0),
        GateProperties(gate_type="cz", qubits=(1, 2), error_rate=0.008, gate_time_ns=68.0),
        GateProperties(gate_type="cz", qubits=(2, 1), error_rate=0.008, gate_time_ns=68.0),
        GateProperties(gate_type="cz", qubits=(2, 3), error_rate=0.015, gate_time_ns=68.0),
        GateProperties(gate_type="cz", qubits=(3, 2), error_rate=0.015, gate_time_ns=68.0),
    ]
    coupling = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
    return BackendProperties(
        backend="test_heron",
        provider="test",
        n_qubits=4,
        basis_gates=("cz", "rz", "sx", "x", "id"),
        coupling_map=coupling,
        qubit_properties=qubit_props,
        gate_properties=gate_props,
        timestamp=timestamp,
    )


def _circuit() -> QBCircuit:
    circ = QBCircuit(n_qubits=2, name="rx")
    circ.add_gate(QBGate(name="h", qubits=(0,)))
    circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
    return circ


def _run():
    backend = _backend()
    mapper = CalibrationMapper(backend)
    result = mapper.run(_circuit(), {})
    return backend, result


@pytest.fixture()
def local_key(tmp_path, monkeypatch):
    """Keep signing inside the test's own directory; never touch a real home."""
    monkeypatch.setenv("QBC_SIGNING_KEY", str(tmp_path / "signing_key"))
    monkeypatch.setenv("QBC_TRUSTED_KEYS", str(tmp_path / "trusted_keys"))
    return tmp_path / "signing_key"


def test_receipt_reflects_mapper_choice():
    backend, result = _run()
    receipt = selection_receipt(result, calibration=backend)

    assert receipt["schema"] == "qb.selection_receipt.v1"
    assert "CalibrationMapper" in receipt["objective"]
    # the receipt's layout IS the mapper's chosen layout
    assert receipt["selected_layout"] == {
        str(k): v for k, v in result.metadata["initial_layout"].items()
    }
    assert receipt["selected_score"] == result.metadata["calibration_score"]
    assert receipt["score_breakdown"] == result.metadata["score_breakdown"]
    assert receipt["calibration_hash"] is not None


def test_unsigned_by_default():
    _, result = _run()
    receipt = selection_receipt(result)
    assert receipt["signature"] is None
    assert receipt["signing"] == "unsigned"


def test_no_executed_layout_means_the_recommendation_ran():
    _, result = _run()
    receipt = selection_receipt(result)
    assert receipt["describes_executed_layout"] is True
    # Backward compatible: without executed_layout the extra fields do not appear at all.
    for field in (
        "recommended_layout",
        "recommended_score",
        "score_penalty_vs_recommended",
        "divergence_note",
    ):
        assert field not in receipt


def test_executed_layout_equal_to_the_recommendation_carries_zero_penalty():
    _, result = _run()
    executed = dict(result.metadata["initial_layout"])
    receipt = selection_receipt(result, executed_layout=executed)

    assert receipt["executed_layout_matches_recommendation"] is True
    assert receipt["score_penalty_vs_recommended"] == 0
    assert receipt["selected_score"] == result.metadata["calibration_score"]
    assert "recommended" in receipt["divergence_note"]


def test_executed_layout_that_differs_is_what_the_receipt_describes():
    """The regression for the original defect: the receipt must name what actually ran."""
    backend, result = _run()
    recommended = dict(result.metadata["initial_layout"])
    executed = {0: 3, 1: 2}
    assert executed != recommended

    receipt = selection_receipt(result, calibration=backend, executed_layout=executed)

    assert receipt["selected_layout"] == {"0": 3, "1": 2}
    assert receipt["recommended_layout"] == {str(k): v for k, v in recommended.items()}
    assert receipt["executed_layout_matches_recommendation"] is False
    assert receipt["describes_executed_layout"] is True
    assert receipt["divergence_note"].startswith("OVERRIDDEN")


def test_penalty_is_computed_when_a_scorer_is_supplied():
    backend = _backend()
    mapper = CalibrationMapper(backend)
    circuit = _circuit()
    result = mapper.run(circuit, {})

    receipt = selection_receipt(
        result,
        executed_layout={0: 3, 1: 2},
        scorer=functools.partial(mapper.score_layout, circuit=circuit),
    )
    penalty = receipt["score_penalty_vs_recommended"]
    assert penalty is not None
    assert penalty > 0, "a worse layout than the pass's pick must show a positive penalty"
    assert receipt["selected_score"] == pytest.approx(mapper.score_layout({0: 3, 1: 2}, circuit))


def test_penalty_is_none_and_says_so_when_no_score_is_available():
    _, result = _run()
    receipt = selection_receipt(result, executed_layout={0: 3, 1: 2})
    assert receipt["score_penalty_vs_recommended"] is None
    assert "not computable" in receipt["divergence_note"]


def test_breakdown_is_dropped_when_a_different_layout_ran():
    _, result = _run()
    receipt = selection_receipt(result, executed_layout={0: 3, 1: 2})
    assert receipt["score_breakdown"] == {}
    assert "recommendation only" in receipt["score_breakdown_note"]


def test_signing_uses_a_persistent_key_and_verifies(local_key):
    from qb_compiler.signing import load_signing_key, verify_receipt

    _, result = _run()
    first = selection_receipt(result, sign=True)
    second = selection_receipt(result, sign=True)

    assert first["signature"] is not None
    assert "ed25519" in first["signing"]
    assert first["key_fingerprint"] == second["key_fingerprint"], "key must outlive the call"
    assert "public_key" not in first, "a receipt must not carry the key that signed it"

    key = load_signing_key()
    assert verify_receipt(first, public_key=key.public).ok


def test_signature_covers_the_executed_layout_fields(local_key):
    from qb_compiler.signing import load_signing_key, verify_receipt

    _, result = _run()
    receipt = selection_receipt(result, executed_layout={0: 3, 1: 2}, sign=True)
    key = load_signing_key()
    assert verify_receipt(receipt, public_key=key.public).ok

    receipt["recommended_layout"] = {"0": 99}
    assert not verify_receipt(receipt, public_key=key.public).ok


def test_fingerprint_stable_and_backend_derived():
    backend = _backend()
    fp1 = calibration_fingerprint(backend)
    fp2 = calibration_fingerprint(backend)
    assert fp1 == fp2 and fp1 is not None
    # different calibration timestamp -> different fingerprint
    backend2 = _backend(timestamp="2026-03-13T00:00:00")
    assert calibration_fingerprint(backend2) != fp1


def test_fingerprint_dict_fallback_and_none():
    assert calibration_fingerprint({"qubits": {"0": 0.01}}) is not None
    assert calibration_fingerprint(object()) is None


class TestCalibrationFreshness:
    def test_iso_string_timestamp_is_read(self):
        from datetime import datetime, timezone

        backend = _backend(timestamp="2026-03-12T00:00:00")
        now = datetime(2026, 3, 12, 1, 0, tzinfo=timezone.utc)
        fresh = calibration_freshness(backend, now=now)
        assert fresh["timestamp_status"] == "measured"
        assert fresh["age_minutes"] == pytest.approx(60.0)
        assert fresh["exceeds_default_tolerance"] is True

    def test_within_tolerance(self):
        from datetime import datetime, timezone

        backend = _backend(timestamp="2026-03-12T00:00:00+00:00")
        now = datetime(2026, 3, 12, 0, 10, tzinfo=timezone.utc)
        fresh = calibration_freshness(backend, now=now)
        assert fresh["exceeds_default_tolerance"] is False

    def test_synthetic_snapshot_has_no_age(self):
        fresh = calibration_freshness(_backend(timestamp="synthetic"))
        assert fresh["timestamp_status"] == "synthetic"
        assert fresh["age_minutes"] is None

    def test_unreadable_timestamp_says_so(self):
        fresh = calibration_freshness(_backend(timestamp="last tuesday"))
        assert fresh["timestamp_status"] == "unreadable"
        assert fresh["age_minutes"] is None

    def test_absent_timestamp(self):
        fresh = calibration_freshness(None)
        assert fresh["timestamp_status"] == "absent"
        assert fresh["age_minutes"] is None

    def test_clock_skew_is_reported_not_treated_as_fresh(self):
        from datetime import datetime, timezone

        backend = _backend(timestamp="2026-03-12T00:00:00")
        fresh = calibration_freshness(backend, now=datetime(2026, 3, 11, tzinfo=timezone.utc))
        assert fresh["timestamp_status"] == "clock_skew"
        assert fresh["exceeds_default_tolerance"] is None

    def test_tolerance_always_declares_it_is_a_default(self):
        fresh = calibration_freshness(_backend())
        assert fresh["tolerance_basis"] == "builtin_default_not_measured_for_this_device"

    def test_dict_source_and_epoch_number(self):
        from datetime import datetime, timezone

        stamp = datetime(2026, 3, 12, tzinfo=timezone.utc)
        fresh = calibration_freshness(
            {"last_update_date": stamp.timestamp()},
            now=datetime(2026, 3, 12, 0, 5, tzinfo=timezone.utc),
        )
        assert fresh["age_minutes"] == pytest.approx(5.0)

    def test_freshness_rides_on_every_receipt(self):
        backend, result = _run()
        receipt = selection_receipt(result, calibration=backend)
        assert receipt["calibration_freshness"]["timestamp_status"] == "measured"
