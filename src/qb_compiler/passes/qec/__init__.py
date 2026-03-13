"""QEC (Quantum Error Correction) compiler passes.

These passes require the QubitBoost SDK (>= 2.5) and will raise
``NotImplementedError`` until that dependency is available.
"""

from qb_compiler.passes.qec.correlated_error_avoidance import CorrelatedErrorAvoidance
from qb_compiler.passes.qec.logical_mapping import LogicalQubitMapper
from qb_compiler.passes.qec.syndrome_scheduling import SyndromeScheduler

__all__ = ["CorrelatedErrorAvoidance", "LogicalQubitMapper", "SyndromeScheduler"]
