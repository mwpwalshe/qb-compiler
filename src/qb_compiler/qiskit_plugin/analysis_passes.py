"""Qiskit-compatible analysis pass wrappers for qb-compiler.

Wraps qb-compiler analysis functionality as Qiskit
:class:`~qiskit.transpiler.basepasses.AnalysisPass` objects so they can
be inserted into any Qiskit pass manager pipeline.
"""

from __future__ import annotations

from typing import Any

from qiskit.transpiler.basepasses import AnalysisPass


class QBDepthAnalysis(AnalysisPass):
    """Qiskit analysis pass that records circuit depth in the property set.

    After running, sets ``property_set["qb_depth"]`` to the circuit depth
    and ``property_set["qb_depth_by_type"]`` to a breakdown by gate type.
    """

    def run(self, dag: Any) -> None:
        """Analyse circuit depth from the DAG representation."""
        self.property_set["qb_depth"] = dag.depth()

        # Depth breakdown by gate type
        depth_by_type: dict[str, int] = {}
        for node in dag.op_nodes():
            gate_name = node.op.name
            depth_by_type[gate_name] = depth_by_type.get(gate_name, 0) + 1
        self.property_set["qb_depth_by_type"] = depth_by_type


class QBGateCountAnalysis(AnalysisPass):
    """Qiskit analysis pass that records gate counts in the property set.

    After running, sets:
    - ``property_set["qb_gate_count"]``: total gate count
    - ``property_set["qb_gate_counts_by_type"]``: ``{gate_name: count}``
    - ``property_set["qb_two_qubit_count"]``: count of 2-qubit gates
    """

    def run(self, dag: Any) -> None:
        """Count gates in the DAG."""
        counts: dict[str, int] = {}
        total = 0
        two_qubit = 0

        for node in dag.op_nodes():
            gate_name = node.op.name
            counts[gate_name] = counts.get(gate_name, 0) + 1
            total += 1
            if len(node.qargs) >= 2:
                two_qubit += 1

        self.property_set["qb_gate_count"] = total
        self.property_set["qb_gate_counts_by_type"] = counts
        self.property_set["qb_two_qubit_count"] = two_qubit


class QBErrorBudgetAnalysis(AnalysisPass):
    """Qiskit analysis pass that estimates the circuit error budget.

    Uses gate counts and configurable error rates to compute a rough
    error budget.  Sets ``property_set["qb_error_budget"]`` to the
    estimated total error probability.

    Parameters
    ----------
    single_qubit_error:
        Assumed single-qubit gate error rate.
    two_qubit_error:
        Assumed two-qubit gate error rate.
    readout_error:
        Assumed per-qubit readout error rate.
    """

    def __init__(
        self,
        *,
        single_qubit_error: float = 0.001,
        two_qubit_error: float = 0.01,
        readout_error: float = 0.01,
    ) -> None:
        super().__init__()
        self._sq_error = single_qubit_error
        self._tq_error = two_qubit_error
        self._ro_error = readout_error

    def run(self, dag: Any) -> None:
        """Estimate the error budget from the DAG."""
        p_ok = 1.0
        n_measurements = 0

        for node in dag.op_nodes():
            n_qubits = len(node.qargs)
            if node.op.name == "measure":
                n_measurements += 1
            elif n_qubits >= 2:
                p_ok *= (1.0 - self._tq_error)
            else:
                p_ok *= (1.0 - self._sq_error)

        # Readout error
        p_ok *= (1.0 - self._ro_error) ** n_measurements

        self.property_set["qb_error_budget"] = 1.0 - p_ok
        self.property_set["qb_estimated_fidelity"] = p_ok
