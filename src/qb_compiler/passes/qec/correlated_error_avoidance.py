"""Correlated error avoidance pass for QEC circuits.

Uses temporal correlation data from SafetyGate / QubitBoost to identify
qubit pairs that exhibit correlated errors over time and restructures
QEC circuits to avoid scheduling critical operations on those pairs
simultaneously.

Correlated errors are particularly damaging to QEC because decoders
(e.g. MWPM, union-find) assume independent error models.  When errors
are correlated, the effective code distance is reduced.

This pass requires the QubitBoost SDK (>= 2.5) for access to
SafetyGate temporal correlation analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qb_compiler.passes.base import PassResult, TransformationPass

if TYPE_CHECKING:
    from qb_compiler.ir.circuit import QBCircuit


class CorrelatedErrorAvoidance(TransformationPass):
    """Restructure QEC circuits to avoid correlated error pairs.

    Given temporal correlation data (from SafetyGate monitoring), this pass:

    - Identifies stabiliser measurements that touch correlated qubit pairs
    - Reorders syndrome extraction to avoid simultaneous operations on
      correlated pairs
    - Inserts dynamical decoupling sequences on idle qubits near correlated
      pairs to suppress cross-talk
    - Optionally adjusts the decoder graph weights to account for residual
      correlations

    Parameters
    ----------
    correlation_threshold : float
        Minimum correlation coefficient (0-1) to consider a pair as
        "correlated".  Pairs below this threshold are treated as independent.
    correlation_data : dict[tuple[int, int], float] | None
        Pre-loaded correlation data mapping qubit pairs to correlation
        coefficients.  If None, the pass will attempt to fetch live data
        from a QubitBoost session.
    insert_dd_sequences : bool
        If True, insert dynamical decoupling sequences on idle qubits
        adjacent to correlated pairs.
    """

    def __init__(
        self,
        correlation_threshold: float = 0.3,
        correlation_data: dict[tuple[int, int], float] | None = None,
        insert_dd_sequences: bool = False,
    ) -> None:
        self._correlation_threshold = correlation_threshold
        self._correlation_data = correlation_data
        self._insert_dd_sequences = insert_dd_sequences

    @property
    def name(self) -> str:
        return "correlated_error_avoidance"

    def transform(self, circuit: QBCircuit, context: dict) -> PassResult:
        raise NotImplementedError("QEC passes require qubitboost-sdk >= 2.5")
