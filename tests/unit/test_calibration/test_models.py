"""Tests for calibration data models: BackendProperties, QubitProperties, GateProperties."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qb_compiler.calibration.models.backend_properties import (
    BackendProperties,
    _infer_provider,
)
from qb_compiler.calibration.models.coupling_properties import GateProperties
from qb_compiler.calibration.models.qubit_properties import QubitProperties

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"
CALIBRATION_DIR = FIXTURES_DIR / "calibration_snapshots"


# ── QubitProperties ──────────────────────────────────────────────────


class TestQubitProperties:
    """Tests for the QubitProperties frozen dataclass."""

    def test_construction_minimal(self) -> None:
        """A QubitProperties can be created with just qubit_id."""
        qp = QubitProperties(qubit_id=0)
        assert qp.qubit_id == 0
        assert qp.t1_us is None
        assert qp.t2_us is None
        assert qp.readout_error is None
        assert qp.frequency_ghz is None

    def test_construction_full(self) -> None:
        """All fields should be preserved."""
        qp = QubitProperties(
            qubit_id=7,
            t1_us=300.0,
            t2_us=150.0,
            readout_error=0.01,
            frequency_ghz=5.1,
            readout_error_0to1=0.005,
            readout_error_1to0=0.015,
        )
        assert qp.qubit_id == 7
        assert qp.t1_us == 300.0
        assert qp.t2_us == 150.0
        assert qp.readout_error == 0.01
        assert qp.frequency_ghz == 5.1
        assert qp.readout_error_0to1 == 0.005
        assert qp.readout_error_1to0 == 0.015

    def test_is_frozen(self) -> None:
        """QubitProperties should be immutable."""
        qp = QubitProperties(qubit_id=0, t1_us=100.0)
        with pytest.raises(AttributeError):
            qp.t1_us = 200.0  # type: ignore[misc]

    def test_symmetrise_readout_both_present(self) -> None:
        """Symmetrise should average when both errors are given."""
        result = QubitProperties.symmetrise_readout(0.002, 0.006)
        assert result == pytest.approx(0.004)

    def test_symmetrise_readout_one_none(self) -> None:
        """Symmetrise should return the non-None value."""
        assert QubitProperties.symmetrise_readout(0.003, None) == 0.003
        assert QubitProperties.symmetrise_readout(None, 0.007) == 0.007

    def test_symmetrise_readout_both_none(self) -> None:
        """Symmetrise should return None when both are None."""
        assert QubitProperties.symmetrise_readout(None, None) is None

    def test_from_qubitboost_dict(self) -> None:
        """Parsing from a QubitBoost-format dict should populate all fields."""
        d = {
            "qubit": 3,
            "T1": 250.0,
            "T2": 120.0,
            "frequency": 5.08,
            "readout_error_0to1": 0.01,
            "readout_error_1to0": 0.03,
        }
        qp = QubitProperties.from_qubitboost_dict(d)
        assert qp.qubit_id == 3
        assert qp.t1_us == 250.0
        assert qp.t2_us == 120.0
        assert qp.frequency_ghz == 5.08
        assert qp.readout_error == pytest.approx(0.02)
        assert qp.readout_error_0to1 == 0.01
        assert qp.readout_error_1to0 == 0.03

    def test_from_qubitboost_dict_missing_optional(self) -> None:
        """Missing optional keys should result in None values."""
        d = {"qubit": 0}
        qp = QubitProperties.from_qubitboost_dict(d)
        assert qp.qubit_id == 0
        assert qp.t1_us is None
        assert qp.readout_error is None


# ── GateProperties ───────────────────────────────────────────────────


class TestGateProperties:
    """Tests for the GateProperties frozen dataclass."""

    def test_construction(self) -> None:
        gp = GateProperties(
            gate_type="cx",
            qubits=(0, 1),
            error_rate=0.005,
            gate_time_ns=340.0,
        )
        assert gp.gate_type == "cx"
        assert gp.qubits == (0, 1)
        assert gp.error_rate == 0.005
        assert gp.gate_time_ns == 340.0

    def test_construction_minimal(self) -> None:
        """Only gate_type and qubits are required."""
        gp = GateProperties(gate_type="rz", qubits=(0,))
        assert gp.error_rate is None
        assert gp.gate_time_ns is None

    def test_is_frozen(self) -> None:
        gp = GateProperties(gate_type="cx", qubits=(0, 1))
        with pytest.raises(AttributeError):
            gp.error_rate = 0.1  # type: ignore[misc]

    def test_from_qubitboost_dict(self) -> None:
        d = {
            "gate": "cx",
            "qubits": [2, 3],
            "parameters": {
                "gate_error": 0.0073,
                "gate_length": 420.0,
            },
        }
        gp = GateProperties.from_qubitboost_dict(d)
        assert gp.gate_type == "cx"
        assert gp.qubits == (2, 3)
        assert gp.error_rate == 0.0073
        assert gp.gate_time_ns == 420.0

    def test_from_qubitboost_dict_no_params(self) -> None:
        d = {"gate": "id", "qubits": [0]}
        gp = GateProperties.from_qubitboost_dict(d)
        assert gp.error_rate is None
        assert gp.gate_time_ns is None


# ── BackendProperties ────────────────────────────────────────────────


class TestBackendProperties:
    """Tests for the BackendProperties aggregate."""

    @pytest.fixture()
    def sample_props(self) -> BackendProperties:
        """Build a small BackendProperties for testing."""
        qubits = [
            QubitProperties(qubit_id=0, t1_us=200.0, t2_us=100.0, readout_error=0.01),
            QubitProperties(qubit_id=1, t1_us=300.0, t2_us=150.0, readout_error=0.005),
            QubitProperties(qubit_id=2, t1_us=150.0, t2_us=80.0, readout_error=0.02),
        ]
        gates = [
            GateProperties(gate_type="cx", qubits=(0, 1), error_rate=0.004),
            GateProperties(gate_type="cx", qubits=(1, 2), error_rate=0.008),
        ]
        return BackendProperties(
            backend="test_backend",
            provider="ibm",
            n_qubits=3,
            basis_gates=("cx", "rz", "sx", "x", "id"),
            coupling_map=[(0, 1), (1, 0), (1, 2), (2, 1)],
            qubit_properties=qubits,
            gate_properties=gates,
            timestamp="2026-02-15T06:00:00",
        )

    def test_qubit_lookup_found(self, sample_props: BackendProperties) -> None:
        qp = sample_props.qubit(1)
        assert qp is not None
        assert qp.qubit_id == 1
        assert qp.t1_us == 300.0

    def test_qubit_lookup_not_found(self, sample_props: BackendProperties) -> None:
        assert sample_props.qubit(99) is None

    def test_gate_lookup_found(self, sample_props: BackendProperties) -> None:
        gp = sample_props.gate("cx", (0, 1))
        assert gp is not None
        assert gp.error_rate == 0.004

    def test_gate_lookup_not_found(self, sample_props: BackendProperties) -> None:
        assert sample_props.gate("cx", (0, 2)) is None
        assert sample_props.gate("cz", (0, 1)) is None

    def test_from_qubitboost_dict(self) -> None:
        data = {
            "backend_name": "ibm_test",
            "provider": "ibm",
            "n_qubits": 2,
            "basis_gates": ["cx", "rz"],
            "coupling_map": [[0, 1], [1, 0]],
            "timestamp": "2026-01-01T00:00:00",
            "qubit_properties": [
                {"qubit": 0, "T1": 100.0, "T2": 50.0},
                {"qubit": 1, "T1": 120.0, "T2": 60.0},
            ],
            "gate_properties": [
                {"gate": "cx", "qubits": [0, 1], "parameters": {"gate_error": 0.01}},
            ],
        }
        props = BackendProperties.from_qubitboost_dict(data)
        assert props.backend == "ibm_test"
        assert props.provider == "ibm"
        assert props.n_qubits == 2
        assert len(props.qubit_properties) == 2
        assert len(props.gate_properties) == 1
        assert props.coupling_map == [(0, 1), (1, 0)]

    def test_from_qubitboost_json(self) -> None:
        path = CALIBRATION_DIR / "ibm_fez_2026_02_15.json"
        props = BackendProperties.from_qubitboost_json(path)
        assert props.backend == "ibm_fez"
        assert props.n_qubits == 156
        assert len(props.qubit_properties) == 10

    def test_infer_provider_ibm(self) -> None:
        assert _infer_provider("ibm_fez") == "ibm"
        assert _infer_provider("ibm_torino") == "ibm"

    def test_infer_provider_rigetti(self) -> None:
        assert _infer_provider("rigetti_ankaa") == "rigetti"
        assert _infer_provider("ankaa_3") == "rigetti"

    def test_infer_provider_ionq(self) -> None:
        assert _infer_provider("ionq_aria") == "ionq"

    def test_infer_provider_iqm(self) -> None:
        assert _infer_provider("iqm_garnet") == "iqm"

    def test_infer_provider_unknown(self) -> None:
        assert _infer_provider("some_random_backend") == "unknown"

    def test_frozen(self, sample_props: BackendProperties) -> None:
        with pytest.raises(AttributeError):
            sample_props.n_qubits = 999  # type: ignore[misc]
