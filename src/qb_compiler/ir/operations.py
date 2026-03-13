"""Gate and operation definitions for the qb-compiler IR.

Every quantum operation that flows through the compiler is represented by one of
the dataclasses here.  They are intentionally lightweight (frozen dataclasses)
so that millions of them can live in memory without overhead.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Operations ───────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class QBGate:
    """A single quantum gate application.

    Parameters
    ----------
    name : str
        Lower-cased gate name (e.g. ``"cx"``, ``"rz"``).
    qubits : tuple[int, ...]
        Qubit indices this gate acts on, in order.
    params : tuple[float, ...]
        Continuous parameters (rotation angles, etc.).  Empty for non-parametric
        gates.
    condition : tuple[int, float] | None
        Optional classical condition ``(clbit, value)`` — the gate fires only
        when ``clbit == value``.
    """

    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()
    condition: tuple[int, float] | None = None

    # convenience ──────────────────────────────────────────────────────

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    @property
    def is_parametric(self) -> bool:
        return len(self.params) > 0

    def inverse(self) -> QBGate:
        """Return the inverse gate where trivially known.

        Self-inverse gates (``h``, ``x``, ``y``, ``z``, ``cx``, ``cz``,
        ``swap``, ``id``, ``ecr``) return an identical copy.  For rotation
        gates the parameters are negated.  Raises ``ValueError`` for gates
        whose inverse is not implemented here.
        """
        if self.name in _SELF_INVERSE:
            return self
        if self.name in _ROTATION_GATES:
            return QBGate(
                name=self.name,
                qubits=self.qubits,
                params=tuple(-p for p in self.params),
                condition=self.condition,
            )
        raise ValueError(f"inverse not implemented for gate '{self.name}'")

    def __repr__(self) -> str:
        parts = [self.name, f"q{list(self.qubits)}"]
        if self.params:
            parts.append(f"params={list(self.params)}")
        if self.condition is not None:
            parts.append(f"if c{self.condition[0]}=={self.condition[1]}")
        return f"QBGate({', '.join(parts)})"


@dataclass(frozen=True, slots=True)
class QBMeasure:
    """Measurement of a single qubit into a classical bit."""

    qubit: int
    clbit: int

    def __repr__(self) -> str:
        return f"QBMeasure(q{self.qubit} -> c{self.clbit})"


@dataclass(frozen=True, slots=True)
class QBBarrier:
    """Scheduling barrier — prevents reordering of ops across it."""

    qubits: tuple[int, ...]

    def __repr__(self) -> str:
        return f"QBBarrier(q{list(self.qubits)})"


# ── Gate catalogue ───────────────────────────────────────────────────

# Maps gate name -> number of qubits.  Used for validation and quick lookups.
STANDARD_GATES: dict[str, int] = {
    # 1-qubit
    "id": 1,
    "x": 1,
    "y": 1,
    "z": 1,
    "h": 1,
    "s": 1,
    "sdg": 1,
    "t": 1,
    "tdg": 1,
    "sx": 1,
    "sxdg": 1,
    "rx": 1,
    "ry": 1,
    "rz": 1,
    "p": 1,     # phase gate
    "u": 1,     # U3
    "u1": 1,
    "u2": 1,
    "u3": 1,
    # IonQ native
    "gpi": 1,
    "gpi2": 1,
    # IQM native
    "prx": 1,
    # 2-qubit
    "cx": 2,
    "cy": 2,
    "cz": 2,
    "ch": 2,
    "swap": 2,
    "ecr": 2,
    "rzz": 2,
    "rxx": 2,
    "ryy": 2,
    "ms": 2,    # IonQ Mølmer-Sørensen
    "cp": 2,    # controlled-phase
    "crx": 2,
    "cry": 2,
    "crz": 2,
    # 3-qubit
    "ccx": 3,
    "cswap": 3,
}

# ── Vendor native basis sets ─────────────────────────────────────────

IBM_BASIS: tuple[str, ...] = ("ecr", "rz", "sx", "x", "id")
RIGETTI_BASIS: tuple[str, ...] = ("cz", "rx", "rz")
IONQ_BASIS: tuple[str, ...] = ("gpi", "gpi2", "ms")
IQM_BASIS: tuple[str, ...] = ("cz", "prx")

# ── Internal helpers ─────────────────────────────────────────────────

_SELF_INVERSE: frozenset[str] = frozenset(
    {"h", "x", "y", "z", "cx", "cy", "cz", "swap", "id", "ecr", "ccx", "cswap"}
)

_ROTATION_GATES: frozenset[str] = frozenset(
    {"rx", "ry", "rz", "p", "u1", "crx", "cry", "crz", "cp", "rzz", "rxx", "ryy"}
)
