# SPDX-License-Identifier: Apache-2.0
"""Selection receipt built from real CalibrationMapper output.

The receipt is derived purely from the mapper's PassResult metadata, so these
tests run the actual mapper and check the receipt faithfully reflects its
chosen layout, score, and breakdown, plus the unsigned-by-default funnel
behaviour.
"""

from __future__ import annotations

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.calibration.models.coupling_properties import GateProperties
from qb_compiler.calibration.models.qubit_properties import QubitProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.mapping import (
    CalibrationMapper,
    calibration_fingerprint,
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


def test_sign_fallback_unsigned_when_sdk_absent(monkeypatch):
    # qubitboost_sdk is an optional paid dependency; absent in OSS CI. Force the
    # import to fail and assert signing degrades to an unsigned receipt with a
    # pointer, never raises.
    import sys

    monkeypatch.setitem(sys.modules, "qubitboost_sdk", None)
    _, result = _run()
    receipt = selection_receipt(result, sign=True)
    assert receipt["signature"] is None
    assert "unsigned" in receipt["signing"]


def test_sign_produces_signature_when_sdk_present():
    # When the paid layer IS installed, sign=True yields a real Ed25519 signature.
    import pytest

    pytest.importorskip("qubitboost_sdk")
    _, result = _run()
    receipt = selection_receipt(result, sign=True)
    assert receipt["signature"] is not None
    assert "ed25519" in receipt["signing"]


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
