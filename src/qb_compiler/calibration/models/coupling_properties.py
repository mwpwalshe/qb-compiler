"""Per-gate calibration properties."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GateProperties:
    """Calibration snapshot for a single gate on specific qubit(s).

    Parameters
    ----------
    gate_type:
        Gate name (lower-cased, e.g. ``"cx"``, ``"cz"``, ``"sx"``).
    qubits:
        Ordered tuple of physical qubit indices the gate acts on.
    error_rate:
        Gate error rate from randomised benchmarking / IRB, or *None*.
    gate_time_ns:
        Gate duration in nanoseconds, or *None*.
    """

    gate_type: str
    qubits: tuple[int, ...]
    error_rate: float | None = None
    gate_time_ns: float | None = None

    @classmethod
    def from_qubitboost_dict(cls, d: dict) -> GateProperties:
        """Parse a single entry from the QubitBoost ``gate_properties`` list.

        Expected keys: gate, qubits (list[int]),
        parameters.gate_error, parameters.gate_length.
        """
        params = d.get("parameters", {})
        return cls(
            gate_type=d["gate"],
            qubits=tuple(d["qubits"]),
            error_rate=params.get("gate_error"),
            gate_time_ns=params.get("gate_length"),
        )
