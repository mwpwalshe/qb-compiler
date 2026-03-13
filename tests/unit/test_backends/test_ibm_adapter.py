"""Tests for the IBM backend adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qb_compiler.backends.base import BackendTarget
from qb_compiler.backends.ibm.adapter import (
    IBM_DEFAULT_ERROR_BUDGET,
    _infer_processor_family,
    ibm_backend_target,
)
from qb_compiler.backends.ibm.calibration import parse_ibm_calibration
from qb_compiler.backends.ibm.native_gates import (
    IBM_EAGLE_BASIS,
    IBM_HERON_BASIS,
    IBM_TYPICAL_1Q_ERROR,
    IBM_TYPICAL_2Q_ERROR,
    IBM_TYPICAL_READOUT_ERROR,
)

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"
CALIBRATION_DIR = FIXTURES_DIR / "calibration_snapshots"


class TestIBMBackendTarget:
    """Test ibm_backend_target factory function."""

    def test_heron_backend(self) -> None:
        """Heron backends should get CZ-based native gates."""
        target = ibm_backend_target("ibm_fez", 156)
        assert target.name == "ibm_fez"
        assert target.n_qubits == 156
        assert target.basis_gates == IBM_HERON_BASIS
        assert "cz" in target.basis_gates
        assert "ecr" not in target.basis_gates

    def test_eagle_backend(self) -> None:
        """Eagle backends should get ECR-based native gates."""
        target = ibm_backend_target("ibm_sherbrooke", 127, processor_family="eagle")
        assert target.basis_gates == IBM_EAGLE_BASIS
        assert "ecr" in target.basis_gates
        assert "cz" not in target.basis_gates

    def test_explicit_processor_family_overrides_inference(self) -> None:
        """Explicit processor_family should override name-based inference."""
        target = ibm_backend_target("ibm_fez", 156, processor_family="eagle")
        assert "ecr" in target.basis_gates

    def test_coupling_map_default_empty(self) -> None:
        """When no coupling_map is given, it should default to empty list."""
        target = ibm_backend_target("ibm_fez", 156)
        assert target.coupling_map == []
        assert target.is_fully_connected

    def test_coupling_map_provided(self) -> None:
        """When coupling_map is given, it should be stored."""
        coupling = [(0, 1), (1, 0), (1, 2), (2, 1)]
        target = ibm_backend_target("ibm_fez", 156, coupling_map=coupling)
        assert target.coupling_map == coupling
        assert not target.is_fully_connected
        assert target.are_connected(0, 1)
        assert not target.are_connected(0, 2)

    def test_supports_gate(self) -> None:
        """BackendTarget.supports_gate should work for IBM gates."""
        target = ibm_backend_target("ibm_fez", 156)
        assert target.supports_gate("cz")
        assert target.supports_gate("rz")
        assert target.supports_gate("sx")
        assert not target.supports_gate("ecr")
        assert not target.supports_gate("toffoli")


class TestInferProcessorFamily:
    """Test processor family inference from backend name."""

    def test_eagle_backends(self) -> None:
        assert _infer_processor_family("ibm_sherbrooke") == "eagle"
        assert _infer_processor_family("ibm_nazca") == "eagle"
        assert _infer_processor_family("ibm_cusco") == "eagle"

    def test_heron_backends(self) -> None:
        assert _infer_processor_family("ibm_fez") == "heron"
        assert _infer_processor_family("ibm_torino") == "heron"
        assert _infer_processor_family("ibm_marrakesh") == "heron"

    def test_unknown_defaults_to_heron(self) -> None:
        """Unknown IBM backends should default to Heron."""
        assert _infer_processor_family("ibm_unknown_2027") == "heron"


class TestIBMDefaultErrorBudget:
    """Test the IBM_DEFAULT_ERROR_BUDGET constants."""

    def test_contains_expected_keys(self) -> None:
        assert "typical_1q_error" in IBM_DEFAULT_ERROR_BUDGET
        assert "typical_2q_error" in IBM_DEFAULT_ERROR_BUDGET
        assert "typical_readout_error" in IBM_DEFAULT_ERROR_BUDGET

    def test_values_match_constants(self) -> None:
        assert IBM_DEFAULT_ERROR_BUDGET["typical_1q_error"] == IBM_TYPICAL_1Q_ERROR
        assert IBM_DEFAULT_ERROR_BUDGET["typical_2q_error"] == IBM_TYPICAL_2Q_ERROR
        assert IBM_DEFAULT_ERROR_BUDGET["typical_readout_error"] == IBM_TYPICAL_READOUT_ERROR

    def test_error_rates_in_valid_range(self) -> None:
        """All error rates should be between 0 and 1."""
        for key, val in IBM_DEFAULT_ERROR_BUDGET.items():
            assert 0.0 < val < 1.0, f"{key} = {val} out of range"


class TestIBMCalibrationParsing:
    """Test IBM calibration data parsing through the adapter flow."""

    def test_parse_and_build_target(self) -> None:
        """Parse IBM calibration JSON and build a BackendTarget from it."""
        path = CALIBRATION_DIR / "ibm_fez_2026_02_15.json"
        with open(path) as fh:
            data = json.load(fh)

        props = parse_ibm_calibration(data)
        target = BackendTarget.from_backend_properties(props)

        assert target.name == "ibm_fez"
        assert target.n_qubits == 156
        assert target.are_connected(0, 1)
        assert target.qubit_distance(0, 2) == 2

    def test_ibm_nan_gate_error_sanitised(self) -> None:
        """NaN gate errors should be replaced with 1.0."""
        data = {
            "backend_name": "ibm_test",
            "n_qubits": 2,
            "timestamp": "",
            "basis_gates": ["cx"],
            "coupling_map": [[0, 1]],
            "qubit_properties": [
                {"qubit": 0, "T1": 100.0, "T2": 50.0},
            ],
            "gate_properties": [
                {
                    "gate": "cx",
                    "qubits": [0, 1],
                    "parameters": {"gate_error": float("nan"), "gate_length": 340.0},
                },
            ],
        }
        props = parse_ibm_calibration(data)
        assert props.gate_properties[0].error_rate == 1.0

    def test_ibm_readout_symmetrisation(self) -> None:
        """IBM readout errors should be symmetrised."""
        data = {
            "backend_name": "ibm_test",
            "n_qubits": 1,
            "timestamp": "",
            "basis_gates": [],
            "coupling_map": [],
            "qubit_properties": [
                {
                    "qubit": 0,
                    "T1": 100.0,
                    "T2": 50.0,
                    "readout_error_0to1": 0.002,
                    "readout_error_1to0": 0.008,
                },
            ],
            "gate_properties": [],
        }
        props = parse_ibm_calibration(data)
        q0 = props.qubit(0)
        assert q0 is not None
        assert q0.readout_error == pytest.approx(0.005)
