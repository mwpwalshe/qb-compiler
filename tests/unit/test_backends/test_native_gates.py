"""Tests for multi-vendor backend native gate sets and calibration parsers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qb_compiler.backends.base import BackendTarget
from qb_compiler.backends.ibm.adapter import ibm_backend_target
from qb_compiler.backends.ibm.calibration import parse_ibm_calibration
from qb_compiler.backends.ibm.native_gates import (
    IBM_EAGLE_BASIS,
    IBM_HERON_BASIS,
    IBM_TYPICAL_GATE_TIMES_NS,
)
from qb_compiler.backends.ionq.calibration import parse_ionq_calibration
from qb_compiler.backends.ionq.native_gates import (
    IONQ_ARIA_BASIS,
    IONQ_TYPICAL_GATE_TIMES_NS,
)
from qb_compiler.backends.iqm.calibration import parse_iqm_calibration
from qb_compiler.backends.iqm.native_gates import (
    IQM_GARNET_BASIS,
    IQM_TYPICAL_GATE_TIMES_NS,
)
from qb_compiler.backends.rigetti.calibration import parse_rigetti_calibration
from qb_compiler.backends.rigetti.native_gates import (
    RIGETTI_ANKAA_BASIS,
    RIGETTI_TYPICAL_GATE_TIMES_NS,
)

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"
CALIBRATION_DIR = FIXTURES_DIR / "calibration_snapshots"


# ── Native gate set tests ────────────────────────────────────────────


class TestIBMNativeGates:
    """Verify IBM native gate definitions."""

    def test_heron_basis_contains_cz(self) -> None:
        assert "cz" in IBM_HERON_BASIS

    def test_eagle_basis_contains_ecr(self) -> None:
        assert "ecr" in IBM_EAGLE_BASIS

    def test_heron_basis_has_single_qubit_gates(self) -> None:
        for gate in ("rz", "sx", "x", "id"):
            assert gate in IBM_HERON_BASIS

    def test_gate_times_rz_is_virtual(self) -> None:
        """RZ should be a virtual gate with zero duration."""
        assert IBM_TYPICAL_GATE_TIMES_NS["rz"] == 0.0

    def test_gate_times_all_non_negative(self) -> None:
        for gate, time_ns in IBM_TYPICAL_GATE_TIMES_NS.items():
            assert time_ns >= 0.0, f"{gate} has negative gate time"


class TestRigettiNativeGates:
    """Verify Rigetti native gate definitions."""

    def test_ankaa_basis_gates(self) -> None:
        expected = {"cz", "rx", "rz", "id"}
        assert set(RIGETTI_ANKAA_BASIS) == expected

    def test_cz_gate_time_realistic(self) -> None:
        """CZ should be around 200 ns for Rigetti."""
        assert 100.0 <= RIGETTI_TYPICAL_GATE_TIMES_NS["cz"] <= 300.0

    def test_rz_is_virtual(self) -> None:
        assert RIGETTI_TYPICAL_GATE_TIMES_NS["rz"] == 0.0


class TestIonQNativeGates:
    """Verify IonQ native gate definitions."""

    def test_aria_basis_gates(self) -> None:
        expected = {"gpi", "gpi2", "ms", "id"}
        assert set(IONQ_ARIA_BASIS) == expected

    def test_ms_gate_time_microsecond_scale(self) -> None:
        """MS gate should be ~600 us (600,000 ns) for trapped ions."""
        ms_time = IONQ_TYPICAL_GATE_TIMES_NS["ms"]
        assert ms_time >= 100_000.0, "MS gate too fast for trapped ions"
        assert ms_time <= 2_000_000.0, "MS gate too slow"

    def test_single_qubit_much_faster_than_two_qubit(self) -> None:
        gpi_time = IONQ_TYPICAL_GATE_TIMES_NS["gpi"]
        ms_time = IONQ_TYPICAL_GATE_TIMES_NS["ms"]
        assert ms_time > 10 * gpi_time


class TestIQMNativeGates:
    """Verify IQM native gate definitions."""

    def test_garnet_basis_gates(self) -> None:
        expected = {"cz", "prx", "id"}
        assert set(IQM_GARNET_BASIS) == expected

    def test_cz_gate_time_realistic(self) -> None:
        """IQM CZ should be ~60 ns (fast superconducting)."""
        assert 30.0 <= IQM_TYPICAL_GATE_TIMES_NS["cz"] <= 120.0

    def test_prx_gate_time_realistic(self) -> None:
        """PRX should be ~32 ns."""
        assert 10.0 <= IQM_TYPICAL_GATE_TIMES_NS["prx"] <= 80.0


# ── Calibration parser tests ─────────────────────────────────────────


class TestIBMCalibrationParser:
    """Test IBM calibration JSON parsing."""

    def test_parse_ibm_fez(self) -> None:
        path = CALIBRATION_DIR / "ibm_fez_2026_02_15.json"
        with open(path) as fh:
            data = json.load(fh)
        props = parse_ibm_calibration(data)

        assert props.backend == "ibm_fez"
        assert props.provider == "ibm"
        assert props.n_qubits == 156
        assert len(props.qubit_properties) == 10
        assert len(props.gate_properties) == 10

    def test_parse_ibm_torino(self) -> None:
        path = CALIBRATION_DIR / "ibm_torino_2026_02_15.json"
        with open(path) as fh:
            data = json.load(fh)
        props = parse_ibm_calibration(data)

        assert props.backend == "ibm_torino"
        assert props.provider == "ibm"
        assert props.n_qubits == 133
        assert len(props.qubit_properties) == 10

        # Spot check qubit 0
        q0 = props.qubit(0)
        assert q0 is not None
        assert q0.t1_us == pytest.approx(220.15, rel=0.01)

    def test_ibm_gate_error_range(self) -> None:
        path = CALIBRATION_DIR / "ibm_fez_2026_02_15.json"
        with open(path) as fh:
            data = json.load(fh)
        props = parse_ibm_calibration(data)

        for gp in props.gate_properties:
            assert gp.error_rate is not None
            assert 0.0 < gp.error_rate < 0.1


class TestRigettiCalibrationParser:
    """Test Rigetti calibration JSON parsing."""

    def test_parse_rigetti_ankaa(self) -> None:
        path = CALIBRATION_DIR / "rigetti_ankaa_2026_02_15.json"
        with open(path) as fh:
            data = json.load(fh)
        props = parse_rigetti_calibration(data)

        assert props.backend == "rigetti_ankaa"
        assert props.provider == "rigetti"
        assert props.n_qubits == 84
        assert len(props.qubit_properties) == 10
        assert "cz" in props.basis_gates

    def test_rigetti_t1_t2_realistic(self) -> None:
        """Rigetti T1/T2 should be much shorter than IBM (superconducting transmon)."""
        path = CALIBRATION_DIR / "rigetti_ankaa_2026_02_15.json"
        with open(path) as fh:
            data = json.load(fh)
        props = parse_rigetti_calibration(data)

        for qp in props.qubit_properties:
            assert qp.t1_us is not None
            # Rigetti T1 typically 10-30 us
            assert 5.0 <= qp.t1_us <= 50.0

    def test_rigetti_gate_type_is_cz(self) -> None:
        path = CALIBRATION_DIR / "rigetti_ankaa_2026_02_15.json"
        with open(path) as fh:
            data = json.load(fh)
        props = parse_rigetti_calibration(data)

        for gp in props.gate_properties:
            assert gp.gate_type == "cz"


class TestIonQCalibrationParser:
    """Test IonQ calibration parsing from synthetic data."""

    def test_parse_ionq_minimal(self) -> None:
        """Parse a minimal IonQ calibration dict."""
        data = {
            "backend_name": "ionq_aria",
            "n_qubits": 25,
            "timestamp": "2026-02-15T12:00:00",
            "basis_gates": ["gpi", "gpi2", "ms", "id"],
            "coupling_map": [],
            "qubit_properties": [
                {
                    "qubit": i,
                    "T1": 1_000_000.0,
                    "T2": 500_000.0,
                    "frequency": None,
                    "readout_error_0to1": 0.002,
                    "readout_error_1to0": 0.004,
                }
                for i in range(5)
            ],
            "gate_properties": [
                {
                    "gate": "ms",
                    "qubits": [0, 1],
                    "parameters": {
                        "gate_error": 0.003,
                        "gate_length": 600_000.0,
                    },
                }
            ],
        }
        props = parse_ionq_calibration(data)

        assert props.backend == "ionq_aria"
        assert props.provider == "ionq"
        assert props.n_qubits == 25
        assert len(props.qubit_properties) == 5
        # All-to-all: empty coupling map
        assert len(props.coupling_map) == 0

    def test_ionq_readout_symmetrisation(self) -> None:
        data = {
            "backend_name": "ionq_aria",
            "n_qubits": 1,
            "timestamp": "",
            "basis_gates": ["gpi", "gpi2", "ms", "id"],
            "coupling_map": [],
            "qubit_properties": [
                {
                    "qubit": 0,
                    "T1": 1_000_000.0,
                    "T2": 500_000.0,
                    "readout_error_0to1": 0.002,
                    "readout_error_1to0": 0.006,
                }
            ],
            "gate_properties": [],
        }
        props = parse_ionq_calibration(data)
        q0 = props.qubit(0)
        assert q0 is not None
        assert q0.readout_error == pytest.approx(0.004, rel=0.01)


class TestIQMCalibrationParser:
    """Test IQM calibration parsing from synthetic data."""

    def test_parse_iqm_minimal(self) -> None:
        data = {
            "backend_name": "iqm_garnet",
            "n_qubits": 20,
            "timestamp": "2026-02-15T14:00:00",
            "basis_gates": ["cz", "prx", "id"],
            "coupling_map": [[0, 1], [1, 0], [1, 2], [2, 1]],
            "qubit_properties": [
                {
                    "qubit": i,
                    "T1": 30.0 + i * 2.0,
                    "T2": 15.0 + i,
                    "frequency": 5.0 + i * 0.01,
                    "readout_error_0to1": 0.01,
                    "readout_error_1to0": 0.03,
                }
                for i in range(3)
            ],
            "gate_properties": [
                {
                    "gate": "cz",
                    "qubits": [0, 1],
                    "parameters": {
                        "gate_error": 0.012,
                        "gate_length": 58.0,
                    },
                },
                {
                    "gate": "cz",
                    "qubits": [1, 2],
                    "parameters": {
                        "gate_error": 0.018,
                        "gate_length": 62.0,
                    },
                },
            ],
        }
        props = parse_iqm_calibration(data)

        assert props.backend == "iqm_garnet"
        assert props.provider == "iqm"
        assert props.n_qubits == 20
        assert len(props.coupling_map) == 4
        assert "prx" in props.basis_gates


# ── IBM adapter tests ─────────────────────────────────────────────────


class TestIBMAdapter:
    """Test the IBM backend adapter."""

    def test_ibm_adapter_heron(self) -> None:
        target = ibm_backend_target("ibm_fez", 156)
        assert target.name == "ibm_fez"
        assert target.n_qubits == 156
        assert "cz" in target.basis_gates
        assert "ecr" not in target.basis_gates

    def test_ibm_adapter_eagle(self) -> None:
        target = ibm_backend_target("ibm_sherbrooke", 127, processor_family="eagle")
        assert "ecr" in target.basis_gates
        assert "cz" not in target.basis_gates

    def test_ibm_adapter_infers_family(self) -> None:
        target = ibm_backend_target("ibm_torino", 133)
        # ibm_torino is Heron
        assert "cz" in target.basis_gates


# ── BackendTarget from calibration ────────────────────────────────────


class TestBackendTargetFromCalibration:
    """Test building BackendTarget from parsed calibration data."""

    def test_backend_target_from_ibm(self) -> None:
        path = CALIBRATION_DIR / "ibm_fez_2026_02_15.json"
        with open(path) as fh:
            data = json.load(fh)
        props = parse_ibm_calibration(data)
        target = BackendTarget.from_backend_properties(props)

        assert target.name == "ibm_fez"
        assert target.n_qubits == 156
        assert target.are_connected(0, 1)
        assert not target.are_connected(0, 9)

    def test_backend_target_from_rigetti(self) -> None:
        path = CALIBRATION_DIR / "rigetti_ankaa_2026_02_15.json"
        with open(path) as fh:
            data = json.load(fh)
        props = parse_rigetti_calibration(data)
        target = BackendTarget.from_backend_properties(props)

        assert target.name == "rigetti_ankaa"
        assert target.n_qubits == 84
        assert target.are_connected(0, 1)
        assert target.qubit_distance(0, 2) == 2
