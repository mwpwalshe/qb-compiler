"""IBM backend native gate definitions."""

from __future__ import annotations

# IBM Heron r1/r2 native basis — the hardware runs CZ natively (Heron),
# but the Qiskit transpiler commonly targets ECR for Eagle.  We include
# both basis sets so the compiler can target either.

IBM_HERON_BASIS: tuple[str, ...] = ("cz", "rz", "sx", "x", "id")
"""Native gates for IBM Heron r1 and r2 processors (ibm_torino, ibm_fez, ibm_marrakesh)."""

IBM_EAGLE_BASIS: tuple[str, ...] = ("ecr", "rz", "sx", "x", "id")
"""Native gates for IBM Eagle r3 processors."""

# Gate durations (typical, in nanoseconds) — used as defaults when
# calibration data is unavailable.
IBM_TYPICAL_GATE_TIMES_NS: dict[str, float] = {
    "id": 24.0,
    "rz": 0.0,  # virtual gate, zero duration
    "sx": 24.0,
    "x": 24.0,
    "cz": 68.0,  # Heron native 2q gate
    "ecr": 660.0,  # Eagle native 2q gate
    "reset": 1000.0,
    "measure": 1200.0,
}

# Typical error rates (approximate, from public IBM calibration data)
IBM_TYPICAL_1Q_ERROR: float = 0.0003
IBM_TYPICAL_2Q_ERROR: float = 0.005
IBM_TYPICAL_READOUT_ERROR: float = 0.01
