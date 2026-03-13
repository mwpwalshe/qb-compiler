"""Syndrome extraction scheduling optimisation pass.

Optimises the order and parallelism of syndrome extraction circuits
for quantum error-correcting codes.  Proper scheduling of stabiliser
measurements can reduce circuit depth, minimise idle time (reducing
decoherence), and avoid hook errors that arise from particular gate
orderings.

This pass requires the QubitBoost SDK (>= 2.5) for access to code
definitions, hook-error analysis, and scheduling heuristics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qb_compiler.passes.base import PassResult, TransformationPass

if TYPE_CHECKING:
    from qb_compiler.ir.circuit import QBCircuit


class SyndromeScheduler(TransformationPass):
    """Optimise syndrome extraction circuit scheduling.

    Given a QEC syndrome extraction circuit, this pass reorders CNOT gates
    within each stabiliser measurement round to:

    - Minimise circuit depth (parallelise where possible)
    - Avoid hook errors by enforcing safe gate orderings
    - Reduce idle time on data qubits between stabiliser interactions
    - Balance ancilla reset/measurement overhead

    Parameters
    ----------
    code_distance : int
        Distance of the error-correcting code.
    avoid_hook_errors : bool
        If True, enforce gate orderings that prevent weight-2 hook errors
        from appearing as logical errors.
    max_parallel_measurements : int | None
        Maximum number of ancilla measurements that can run in parallel.
        Hardware-dependent; None means no limit.
    """

    def __init__(
        self,
        code_distance: int = 3,
        avoid_hook_errors: bool = True,
        max_parallel_measurements: int | None = None,
    ) -> None:
        self._code_distance = code_distance
        self._avoid_hook_errors = avoid_hook_errors
        self._max_parallel_measurements = max_parallel_measurements

    @property
    def name(self) -> str:
        return "syndrome_scheduler"

    def transform(self, circuit: QBCircuit, context: dict) -> PassResult:
        raise NotImplementedError("QEC passes require qubitboost-sdk >= 2.5")
