"""Shared fixtures for the qb-compiler test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

# ── Import guards for optional dependencies ──────────────────────────

try:
    import qiskit  # noqa: F401

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

try:
    import cirq  # noqa: F401

    HAS_CIRQ = True
except ImportError:
    HAS_CIRQ = False

requires_qiskit = pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
requires_cirq = pytest.mark.skipif(not HAS_CIRQ, reason="cirq not installed")

# ── Paths ────────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CALIBRATION_DIR = FIXTURES_DIR / "calibration_snapshots"
CIRCUITS_DIR = FIXTURES_DIR / "circuits"


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def sample_circuit():
    """4-qubit Bell + GHZ circuit using the compiler's QBCircuit."""
    from qb_compiler.compiler import QBCircuit

    # Bell pair on q0-q1, then extend to GHZ across q0-q3
    circ = QBCircuit(4)
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.cx(2, 3)
    return circ


@pytest.fixture()
def mock_calibration():
    """StaticCalibrationProvider with realistic IBM Fez-like data (10 qubits).

    Builds an in-memory BackendProperties matching the QubitBoost
    calibration_hub/heron/ format and wraps it in a StaticCalibrationProvider.
    """
    from qb_compiler.calibration.models.backend_properties import BackendProperties
    from qb_compiler.calibration.models.coupling_properties import GateProperties
    from qb_compiler.calibration.models.qubit_properties import QubitProperties
    from qb_compiler.calibration.static_provider import StaticCalibrationProvider

    qubit_props = [
        QubitProperties(
            qubit_id=i,
            t1_us=150.0 + i * 15.0,  # 150 - 285 us
            t2_us=100.0 + i * 12.0,  # 100 - 208 us
            readout_error=(0.005 + i * 0.004),  # 0.005 - 0.041
            frequency_ghz=5.0 + i * 0.02,
            readout_error_0to1=0.002 + i * 0.002,
            readout_error_1to0=0.008 + i * 0.006,
        )
        for i in range(10)
    ]
    gate_props = [
        GateProperties(
            gate_type="cx",
            qubits=(i, i + 1),
            error_rate=0.004 + i * 0.001,
            gate_time_ns=340.0 + i * 20.0,
        )
        for i in range(9)
    ]
    coupling = [(i, i + 1) for i in range(9)] + [(i + 1, i) for i in range(9)]

    props = BackendProperties(
        backend="ibm_fez",
        provider="ibm",
        n_qubits=156,
        basis_gates=("id", "rz", "sx", "x", "cx", "reset"),
        coupling_map=coupling,
        qubit_properties=qubit_props,
        gate_properties=gate_props,
        timestamp="2026-02-15T06:00:11.254874",
    )
    return StaticCalibrationProvider(props)


@pytest.fixture()
def ibm_fez_calibration_path() -> Path:
    """Path to the IBM Fez calibration snapshot JSON fixture."""
    return CALIBRATION_DIR / "ibm_fez_2026_02_15.json"
