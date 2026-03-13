"""Tests for the Rigetti backend adapter and calibration parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qb_compiler.backends.base import BackendTarget
from qb_compiler.backends.rigetti.calibration import parse_rigetti_calibration
from qb_compiler.backends.rigetti.native_gates import (
    RIGETTI_ANKAA_BASIS,
    RIGETTI_TYPICAL_1Q_ERROR,
    RIGETTI_TYPICAL_2Q_ERROR,
    RIGETTI_TYPICAL_GATE_TIMES_NS,
    RIGETTI_TYPICAL_READOUT_ERROR,
)

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"
CALIBRATION_DIR = FIXTURES_DIR / "calibration_snapshots"


class TestRigettiNativeGateConstants:
    """Verify Rigetti native gate constants are sensible."""

    def test_ankaa_basis_includes_cz(self) -> None:
        assert "cz" in RIGETTI_ANKAA_BASIS

    def test_ankaa_basis_includes_single_qubit_gates(self) -> None:
        assert "rx" in RIGETTI_ANKAA_BASIS
        assert "rz" in RIGETTI_ANKAA_BASIS

    def test_error_rates_positive(self) -> None:
        assert RIGETTI_TYPICAL_1Q_ERROR > 0
        assert RIGETTI_TYPICAL_2Q_ERROR > 0
        assert RIGETTI_TYPICAL_READOUT_ERROR > 0

    def test_2q_error_higher_than_1q(self) -> None:
        """Two-qubit gate error should be significantly higher than single-qubit."""
        assert RIGETTI_TYPICAL_2Q_ERROR > RIGETTI_TYPICAL_1Q_ERROR

    def test_gate_times_all_nonnegative(self) -> None:
        for gate, time_ns in RIGETTI_TYPICAL_GATE_TIMES_NS.items():
            assert time_ns >= 0.0, f"{gate} has negative gate time"


class TestRigettiCalibrationParsing:
    """Test Rigetti calibration JSON parsing."""

    @pytest.fixture()
    def rigetti_data(self) -> dict:
        path = CALIBRATION_DIR / "rigetti_ankaa_2026_02_15.json"
        with open(path) as fh:
            return json.load(fh)

    def test_parse_basic_fields(self, rigetti_data: dict) -> None:
        props = parse_rigetti_calibration(rigetti_data)
        assert props.backend == "rigetti_ankaa"
        assert props.provider == "rigetti"
        assert props.n_qubits == 84

    def test_basis_gates(self, rigetti_data: dict) -> None:
        props = parse_rigetti_calibration(rigetti_data)
        assert "cz" in props.basis_gates

    def test_qubit_properties_populated(self, rigetti_data: dict) -> None:
        props = parse_rigetti_calibration(rigetti_data)
        assert len(props.qubit_properties) > 0
        q0 = props.qubit(0)
        assert q0 is not None
        assert q0.t1_us is not None
        assert q0.t1_us > 0

    def test_gate_properties_populated(self, rigetti_data: dict) -> None:
        props = parse_rigetti_calibration(rigetti_data)
        assert len(props.gate_properties) > 0
        for gp in props.gate_properties:
            assert gp.gate_type == "cz"

    def test_coupling_map_populated(self, rigetti_data: dict) -> None:
        props = parse_rigetti_calibration(rigetti_data)
        assert len(props.coupling_map) > 0
        for q1, q2 in props.coupling_map:
            assert isinstance(q1, int)
            assert isinstance(q2, int)

    def test_build_backend_target(self, rigetti_data: dict) -> None:
        """Build a BackendTarget from parsed Rigetti calibration data."""
        props = parse_rigetti_calibration(rigetti_data)
        target = BackendTarget.from_backend_properties(props)

        assert target.name == "rigetti_ankaa"
        assert target.n_qubits == 84
        assert target.are_connected(0, 1)


class TestRigettiSyntheticCalibration:
    """Test Rigetti calibration parsing with synthetic data."""

    def test_minimal_calibration(self) -> None:
        data = {
            "backend_name": "rigetti_test",
            "n_qubits": 4,
            "timestamp": "2026-03-01T00:00:00",
            "basis_gates": ["cz", "rx", "rz"],
            "coupling_map": [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]],
            "qubit_properties": [
                {
                    "qubit": i,
                    "T1": 15.0 + i * 2.0,
                    "T2": 8.0 + i,
                    "readout_error_0to1": 0.015,
                    "readout_error_1to0": 0.025,
                }
                for i in range(4)
            ],
            "gate_properties": [
                {
                    "gate": "cz",
                    "qubits": [0, 1],
                    "parameters": {"gate_error": 0.02, "gate_length": 200.0},
                },
            ],
        }
        props = parse_rigetti_calibration(data)
        assert props.backend == "rigetti_test"
        assert props.provider == "rigetti"
        assert props.n_qubits == 4
        assert len(props.qubit_properties) == 4
        assert len(props.gate_properties) == 1

    def test_readout_symmetrisation(self) -> None:
        data = {
            "backend_name": "rigetti_test",
            "n_qubits": 1,
            "timestamp": "",
            "basis_gates": [],
            "coupling_map": [],
            "qubit_properties": [
                {
                    "qubit": 0,
                    "T1": 20.0,
                    "T2": 10.0,
                    "readout_error_0to1": 0.01,
                    "readout_error_1to0": 0.05,
                },
            ],
            "gate_properties": [],
        }
        props = parse_rigetti_calibration(data)
        q0 = props.qubit(0)
        assert q0 is not None
        assert q0.readout_error == pytest.approx(0.03)

    def test_nan_gate_error_sanitised(self) -> None:
        """NaN gate errors should be replaced with 1.0."""
        data = {
            "backend_name": "rigetti_test",
            "n_qubits": 2,
            "timestamp": "",
            "basis_gates": ["cz"],
            "coupling_map": [[0, 1]],
            "qubit_properties": [],
            "gate_properties": [
                {
                    "gate": "cz",
                    "qubits": [0, 1],
                    "parameters": {"gate_error": float("nan"), "gate_length": 200.0},
                },
            ],
        }
        props = parse_rigetti_calibration(data)
        assert props.gate_properties[0].error_rate == 1.0
