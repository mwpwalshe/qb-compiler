"""Built-in analysis passes."""

from __future__ import annotations

from qb_compiler.passes.analysis.connectivity_check import ConnectivityCheck
from qb_compiler.passes.analysis.depth_analysis import DepthAnalysis
from qb_compiler.passes.analysis.error_budget_estimator import ErrorBudgetEstimator
from qb_compiler.passes.analysis.gate_count import GateCountAnalysis

__all__ = ["ConnectivityCheck", "DepthAnalysis", "ErrorBudgetEstimator", "GateCountAnalysis"]
