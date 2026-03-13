"""Gate count analysis pass."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qb_compiler.passes.base import AnalysisPass

if TYPE_CHECKING:
    from qb_compiler.ir.circuit import QBCircuit


class GateCountAnalysis(AnalysisPass):
    """Count gates by type and store in ``context["gate_counts"]``.

    Also populates:
    - ``context["total_gates"]`` — total gate count
    - ``context["two_qubit_gates"]`` — number of 2+ qubit gates
    """

    @property
    def name(self) -> str:
        return "gate_count_analysis"

    def analyze(self, circuit: QBCircuit, context: dict) -> None:
        context["gate_counts"] = dict(circuit.gate_counts)
        context["total_gates"] = circuit.gate_count
        context["two_qubit_gates"] = circuit.two_qubit_gate_count
