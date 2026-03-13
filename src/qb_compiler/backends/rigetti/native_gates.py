"""Rigetti backend native gate definitions."""

from __future__ import annotations

RIGETTI_ANKAA_BASIS: tuple[str, ...] = ("cz", "rx", "rz", "id")
"""Native gates for Rigetti Ankaa-3 processors."""

RIGETTI_TYPICAL_GATE_TIMES_NS: dict[str, float] = {
    "id": 20.0,
    "rx": 40.0,
    "rz": 0.0,      # virtual gate, zero duration
    "cz": 200.0,     # native 2q gate
    "reset": 2000.0,
    "measure": 1500.0,
}

# Typical error rates (approximate, from public Rigetti calibration data)
RIGETTI_TYPICAL_1Q_ERROR: float = 0.003
RIGETTI_TYPICAL_2Q_ERROR: float = 0.02
RIGETTI_TYPICAL_READOUT_ERROR: float = 0.03
