"""Depth analysis pass."""

from __future__ import annotations

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.passes.base import AnalysisPass


class DepthAnalysis(AnalysisPass):
    """Compute circuit depth and store it in ``context["depth"]``."""

    @property
    def name(self) -> str:
        return "depth_analysis"

    def analyze(self, circuit: QBCircuit, context: dict) -> None:
        context["depth"] = circuit.depth
