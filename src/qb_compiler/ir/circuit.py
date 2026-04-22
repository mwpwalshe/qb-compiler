"""Vendor-neutral quantum circuit intermediate representation.

:class:`QBCircuit` is the central data structure that every compiler pass
operates on.  It stores an ordered sequence of :class:`QBGate`,
:class:`QBMeasure`, and :class:`QBBarrier` operations and exposes cheap
summary statistics (depth, gate counts) used by analysis passes.
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import TYPE_CHECKING

from qb_compiler.ir.operations import (
    QBBarrier,
    QBGate,
    QBMeasure,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

Operation = QBGate | QBMeasure | QBBarrier


class QBCircuit:
    """Ordered list of quantum operations — the compiler's main IR.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit register.
    n_clbits : int
        Number of classical bits (defaults to 0).
    name : str
        Optional human-readable label.
    """

    __slots__ = ("_ops", "n_clbits", "n_qubits", "name")

    def __init__(self, n_qubits: int, n_clbits: int = 0, name: str = "") -> None:
        if n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
        if n_clbits < 0:
            raise ValueError(f"n_clbits must be >= 0, got {n_clbits}")
        self.n_qubits = n_qubits
        self.n_clbits = n_clbits
        self.name = name
        self._ops: list[Operation] = []

    # ── mutation ──────────────────────────────────────────────────────

    def add_gate(self, gate: QBGate) -> None:
        """Append a gate.  Validates qubit indices against the register."""
        for q in gate.qubits:
            if q < 0 or q >= self.n_qubits:
                raise IndexError(f"Qubit index {q} out of range for {self.n_qubits}-qubit circuit")
        self._ops.append(gate)

    def add_measurement(self, qubit: int, clbit: int) -> None:
        """Append a measurement operation."""
        if qubit < 0 or qubit >= self.n_qubits:
            raise IndexError(f"Qubit index {qubit} out of range for {self.n_qubits}-qubit circuit")
        if clbit < 0 or clbit >= self.n_clbits:
            raise IndexError(
                f"Classical bit {clbit} out of range for {self.n_clbits}-clbit circuit"
            )
        self._ops.append(QBMeasure(qubit=qubit, clbit=clbit))

    def add_barrier(self, qubits: Sequence[int] | None = None) -> None:
        """Append a barrier over the given qubits (default: all)."""
        qubits = tuple(range(self.n_qubits)) if qubits is None else tuple(qubits)
        for q in qubits:
            if q < 0 or q >= self.n_qubits:
                raise IndexError(f"Qubit index {q} out of range for {self.n_qubits}-qubit circuit")
        self._ops.append(QBBarrier(qubits=qubits))

    # ── read-only views ───────────────────────────────────────────────

    @property
    def operations(self) -> list[Operation]:
        """All operations in insertion order (gates + measurements + barriers)."""
        return list(self._ops)

    @property
    def gates(self) -> list[QBGate]:
        """Only gate operations, preserving order."""
        return [op for op in self._ops if isinstance(op, QBGate)]

    @property
    def measurements(self) -> list[QBMeasure]:
        return [op for op in self._ops if isinstance(op, QBMeasure)]

    def iter_ops(self) -> Iterator[Operation]:
        """Iterate over operations without copying."""
        return iter(self._ops)

    # ── statistics ────────────────────────────────────────────────────

    @property
    def gate_count(self) -> int:
        """Total number of gates (excludes measurements and barriers)."""
        return sum(1 for op in self._ops if isinstance(op, QBGate))

    @property
    def two_qubit_gate_count(self) -> int:
        return sum(1 for op in self._ops if isinstance(op, QBGate) and op.num_qubits >= 2)

    @property
    def gate_counts(self) -> Counter[str]:
        """Gate-name -> count mapping."""
        c: Counter[str] = Counter()
        for op in self._ops:
            if isinstance(op, QBGate):
                c[op.name] += 1
        return c

    @property
    def depth(self) -> int:
        """Circuit depth computed via a per-qubit occupation array.

        Barriers are respected as synchronisation points.
        """
        if not self._ops:
            return 0

        qubit_depth = [0] * self.n_qubits

        for op in self._ops:
            if isinstance(op, QBGate):
                qubits = op.qubits
            elif isinstance(op, QBMeasure):
                qubits = (op.qubit,)
            elif isinstance(op, QBBarrier):
                qubits = op.qubits
            else:
                continue  # pragma: no cover

            if not qubits:
                continue

            max_d = max(qubit_depth[q] for q in qubits)
            new_d = max_d + 1
            for q in qubits:
                qubit_depth[q] = new_d

        return max(qubit_depth) if qubit_depth else 0

    def qubits_used(self) -> set[int]:
        """Set of qubit indices that appear in at least one operation."""
        used: set[int] = set()
        for op in self._ops:
            if isinstance(op, QBGate):
                used.update(op.qubits)
            elif isinstance(op, QBMeasure):
                used.add(op.qubit)
            elif isinstance(op, QBBarrier):
                used.update(op.qubits)
        return used

    # ── copying ───────────────────────────────────────────────────────

    def copy(self) -> QBCircuit:
        """Deep copy of this circuit."""
        return copy.deepcopy(self)

    # ── dunder ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of operations (including barriers and measurements)."""
        return len(self._ops)

    def __repr__(self) -> str:
        return (
            f"QBCircuit(name={self.name!r}, qubits={self.n_qubits}, "
            f"clbits={self.n_clbits}, gates={self.gate_count}, "
            f"depth={self.depth})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QBCircuit):
            return NotImplemented
        return (
            self.n_qubits == other.n_qubits
            and self.n_clbits == other.n_clbits
            and self._ops == other._ops
        )
