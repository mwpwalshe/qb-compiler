"""IonQ backend native gate definitions."""

from __future__ import annotations

IONQ_ARIA_BASIS: tuple[str, ...] = ("gpi", "gpi2", "ms", "id")
"""Native gates for IonQ Aria-2 trapped-ion processors.

- ``gpi``: single-qubit gate (rotation about axis in XY plane)
- ``gpi2``: single-qubit gate (pi/2 rotation about axis in XY plane)
- ``ms``: Molmer-Sorensen entangling gate (two-qubit)
- ``id``: identity
"""

IONQ_TYPICAL_GATE_TIMES_NS: dict[str, float] = {
    "id": 10_000.0,      # ~10 us
    "gpi": 10_000.0,     # ~10 us single-qubit gate
    "gpi2": 10_000.0,    # ~10 us single-qubit gate
    "ms": 600_000.0,     # ~600 us Molmer-Sorensen gate
    "reset": 100_000.0,
    "measure": 300_000.0,
}

# Typical error rates for IonQ Aria (trapped-ion, much lower than superconducting)
IONQ_TYPICAL_1Q_ERROR: float = 0.0003
IONQ_TYPICAL_2Q_ERROR: float = 0.003
IONQ_TYPICAL_READOUT_ERROR: float = 0.003
