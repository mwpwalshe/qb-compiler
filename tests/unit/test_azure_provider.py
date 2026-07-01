"""Offline tests for the Azure Quantum provider (pure parser + offline factory; no SDK needed)."""

from __future__ import annotations

import pytest

from qb_compiler.calibration.azure_provider import (
    AzureQuantumCalibrationProvider,
    parse_azure_target,
)

_TARGET = {
    "provider_id": "ionq",
    "capabilities": {"qubitCount": 25, "nativeGateSet": ["GPI", "GPI2", "MS"]},
}

_TARGET_WITH_CAL = {
    "provider_id": "quantinuum",
    "capabilities": {"qubitCount": 2, "gateSet": ["RZ", "ZZ"]},
    "calibration": {
        "qubits": {
            "0": {"readout_error": 0.01, "t1_us": 1000.0},
            "1": {"readout_error": 0.02},
        }
    },
}


def test_parse_capabilities_only():
    props = parse_azure_target(_TARGET, backend="ionq_aria")
    assert props.provider == "ionq"
    assert props.n_qubits == 25
    assert "gpi" in props.basis_gates and "ms" in props.basis_gates
    # capability-only target -> structurally valid, calibration-sparse
    assert len(props.qubit_properties) == 25
    assert all(q.readout_error is None for q in props.qubit_properties)


def test_parse_published_calibration():
    props = parse_azure_target(_TARGET_WITH_CAL, backend="quantinuum_h2")
    assert props.provider == "quantinuum"
    by_q = {q.qubit_id: q for q in props.qubit_properties}
    assert by_q[0].readout_error == pytest.approx(0.01)
    assert by_q[0].t1_us == pytest.approx(1000.0)
    assert by_q[1].readout_error == pytest.approx(0.02)


def test_from_target_offline():
    prov = AzureQuantumCalibrationProvider.from_target(_TARGET, backend="ionq_aria")
    assert prov.backend_name == "ionq_aria"
    assert prov.backend_properties.n_qubits == 25
    assert len(prov.get_all_qubit_properties()) == 25
