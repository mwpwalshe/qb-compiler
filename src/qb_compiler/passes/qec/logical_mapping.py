"""Logical-to-physical QEC qubit mapping pass.

Maps logical QEC qubits (data qubits + ancillas for a given error-correcting
code) to physical hardware qubits, taking into account code distance,
stabiliser structure, and hardware connectivity.

This pass requires the QubitBoost SDK (>= 2.5) for access to code
definitions and hardware-aware placement algorithms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qb_compiler.passes.base import PassResult, TransformationPass

if TYPE_CHECKING:
    from qb_compiler.ir.circuit import QBCircuit


class LogicalQubitMapper(TransformationPass):
    """Map logical QEC qubits to physical hardware qubits.

    Given a quantum error-correcting code (e.g. surface code, repetition code)
    and a hardware coupling map, this pass determines an optimal placement
    of data qubits and ancilla qubits on the physical device.  The mapping
    accounts for:

    - Code distance and stabiliser weight
    - Hardware connectivity constraints
    - Ancilla placement for efficient syndrome extraction
    - Minimisation of cross-talk between logical patches

    Parameters
    ----------
    code_distance : int
        Distance of the error-correcting code.
    code_type : str
        Type of QEC code (e.g. ``"surface"``, ``"repetition"``, ``"steane"``).
    coupling_map : list[tuple[int, int]] | None
        Physical device coupling map.  If None, assumes all-to-all.
    """

    def __init__(
        self,
        code_distance: int = 3,
        code_type: str = "surface",
        coupling_map: list[tuple[int, int]] | None = None,
    ) -> None:
        self._code_distance = code_distance
        self._code_type = code_type
        self._coupling_map = coupling_map

    @property
    def name(self) -> str:
        return "logical_qubit_mapper"

    def transform(self, circuit: QBCircuit, context: dict) -> PassResult:
        raise NotImplementedError("QEC passes require qubitboost-sdk >= 2.5")
