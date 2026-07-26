"""QEC (Quantum Error Correction) compiler passes.

These passes are available only in a QubitBoost commercial build and raise
``NotImplementedError`` otherwise.
"""

from __future__ import annotations

from qb_compiler.passes.qec.correlated_error_avoidance import CorrelatedErrorAvoidance
from qb_compiler.passes.qec.logical_mapping import LogicalQubitMapper
from qb_compiler.passes.qec.syndrome_scheduling import SyndromeScheduler

__all__ = ["CorrelatedErrorAvoidance", "LogicalQubitMapper", "SyndromeScheduler"]
