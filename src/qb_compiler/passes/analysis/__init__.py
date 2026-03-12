"""Built-in analysis passes."""

from qb_compiler.passes.analysis.depth_analysis import DepthAnalysis
from qb_compiler.passes.analysis.gate_count import GateCountAnalysis

__all__ = ["DepthAnalysis", "GateCountAnalysis"]
