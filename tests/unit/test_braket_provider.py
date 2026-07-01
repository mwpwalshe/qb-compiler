"""Offline tests for the AWS Braket live-calibration provider.

No AWS credentials/network: we exercise the pure parser and the ``from_device_properties`` path with
realistic Braket-shaped ``.properties`` dicts (IonQ device-average shape + Rigetti per-qubit shape).
"""

from __future__ import annotations

import pytest

from qb_compiler.calibration.braket_provider import (
    BRAKET_DEVICE_ARNS,
    BraketCalibrationProvider,
    parse_braket_properties,
)

# IonQ publishes device-average fidelities + timing, fully-connected.
_IONQ_PROPS = {
    "paradigm": {
        "qubitCount": 25,
        "nativeGateSet": ["GPI", "GPI2", "MS"],
        "connectivity": {"fullyConnected": True},
    },
    "provider": {
        "fidelity": {"1Q": {"mean": 0.9963}, "2Q": {"mean": 0.971}, "spam": {"mean": 0.9947}},
        "timing": {"T1": 10.0, "T2": 1.0},  # seconds
    },
}

# Rigetti publishes per-qubit / per-edge specs over a sparse graph.
_RIGETTI_PROPS = {
    "paradigm": {
        "qubitCount": 3,
        "nativeGateSet": ["rx", "rz", "cz"],
        "connectivity": {"fullyConnected": False, "connectivityGraph": {"0": ["1"], "1": ["2"]}},
    },
    "provider": {
        "specs": {
            "1Q": {
                "0": {"f1QRB": 0.999, "fRO": 0.95, "T1": 20e-6, "T2": 18e-6},
                "1": {"f1QRB": 0.998, "fRO": 0.94, "T1": 22e-6, "T2": 19e-6},
                "2": {"f1QRB": 0.997, "fRO": 0.93, "T1": 21e-6, "T2": 17e-6},
            },
            "2Q": {"0-1": {"fCZ": 0.98}, "1-2": {"fCZ": 0.97}},
        }
    },
}


def test_ionq_device_average_parse():
    props = parse_braket_properties(_IONQ_PROPS, backend="ionq_aria", provider="ionq")
    assert props.n_qubits == 25
    assert "gpi" in props.basis_gates
    assert len(props.qubit_properties) == 25
    q0 = props.qubit_properties[0]
    assert q0.t1_us == pytest.approx(10.0 * 1e6)  # seconds -> microseconds
    assert q0.readout_error == pytest.approx(1.0 - 0.9947)
    one_q = [g for g in props.gate_properties if g.gate_type == "1q"]
    assert one_q and one_q[0].error_rate == pytest.approx(1.0 - 0.9963)
    two_q = [g for g in props.gate_properties if g.gate_type == "2q"]
    assert two_q and two_q[0].error_rate == pytest.approx(1.0 - 0.971)


def test_rigetti_per_qubit_parse():
    props = parse_braket_properties(_RIGETTI_PROPS, backend="rigetti_ankaa", provider="rigetti")
    assert props.n_qubits == 3
    assert sorted(props.coupling_map) == [(0, 1), (1, 2)]
    by_id = {q.qubit_id: q for q in props.qubit_properties}
    assert by_id[0].t1_us == pytest.approx(20.0)  # 20e-6 s -> 20 us
    assert by_id[0].readout_error == pytest.approx(1.0 - 0.95)
    g01 = next(g for g in props.gate_properties if g.gate_type == "2q" and g.qubits == (0, 1))
    assert g01.error_rate == pytest.approx(1.0 - 0.98)


def test_from_device_properties_lookups():
    prov = BraketCalibrationProvider.from_device_properties(
        _RIGETTI_PROPS, backend="rigetti_ankaa", status="ONLINE"
    )
    assert prov.backend_name == "rigetti_ankaa"
    assert prov.device_status == "ONLINE"
    q1 = prov.get_qubit_properties(1)
    assert q1 is not None and q1.t1_us == pytest.approx(22.0)
    assert len(prov.get_all_qubit_properties()) == 3
    g = prov.get_gate_properties("2q", (1, 2))
    assert g is not None and g.error_rate == pytest.approx(1.0 - 0.97)


def test_missing_provider_calibration_is_graceful():
    props = parse_braket_properties(
        {"paradigm": {"qubitCount": 5, "connectivity": {"fullyConnected": True}}},
        backend="x",
        provider="braket",
    )
    assert props.n_qubits == 5
    assert props.qubit_properties == []  # no calibration published -> no crash, empty
    assert len(props.coupling_map) == 5 * 4  # fully connected


def test_known_device_arns_present():
    for name in ("ionq_aria", "ionq_forte", "rigetti_ankaa", "iqm_garnet"):
        assert name in BRAKET_DEVICE_ARNS
        assert BRAKET_DEVICE_ARNS[name].startswith("arn:aws:braket:")
