"""Depth analysis pass."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qb_compiler.passes.base import AnalysisPass

if TYPE_CHECKING:
    from qb_compiler.ir.circuit import QBCircuit


class DepthAnalysis(AnalysisPass):
    """Compute circuit depth and store it in ``context["depth"]``."""

    @property
    def name(self) -> str:
        return "depth_analysis"

    def analyze(self, circuit: QBCircuit, context: dict) -> None:
        context["depth"] = circuit.depth
