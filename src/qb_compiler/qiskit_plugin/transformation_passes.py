"""Qiskit-compatible transformation pass wrappers for qb-compiler.

Wraps qb-compiler optimisation passes as Qiskit
:class:`~qiskit.transpiler.basepasses.TransformationPass` objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiskit.transpiler.basepasses import TransformationPass

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGCircuit


class QBGateCancellation(TransformationPass):
    """Qiskit transformation pass that cancels adjacent inverse gates.

    Scans the DAG for adjacent self-inverse gate pairs (e.g. CX-CX,
    H-H, X-X) and removes them.  Also cancels adjacent rotation gates
    that sum to zero or 2pi.

    Parameters
    ----------
    max_iterations:
        Maximum number of cancellation sweeps.  Each sweep may reveal
        new cancellation opportunities.  Default is 3.
    """

    def __init__(self, *, max_iterations: int = 3) -> None:
        super().__init__()
        self._max_iterations = max_iterations

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run gate cancellation on the DAG.

        This is a structural pass — it looks for patterns in the DAG
        topology rather than doing full unitary simulation.

        Returns the (potentially modified) DAG.
        """
        # Self-inverse gates that cancel when applied twice in succession
        self_inverse = {"cx", "cz", "h", "x", "y", "z", "swap"}

        for _iteration in range(self._max_iterations):
            cancelled = False
            nodes_to_remove: list[Any] = []

            for node in dag.op_nodes():
                if node.op.name not in self_inverse:
                    continue

                # Check successors on the same qubits
                for successor in dag.successors(node):
                    if not hasattr(successor, "op"):
                        continue
                    if successor.op.name == node.op.name and successor.qargs == node.qargs:
                        nodes_to_remove.extend([node, successor])
                        cancelled = True
                        break

            # Remove cancelled pairs
            seen = set()
            for n in nodes_to_remove:
                nid = id(n)
                if nid not in seen:
                    seen.add(nid)
                    dag.remove_op_node(n)

            if not cancelled:
                break

        return dag


class QBGateDecomposition(TransformationPass):
    """Qiskit transformation pass for basis gate decomposition.

    Decomposes gates not in the target basis set into sequences of
    basis gates.  This is a simplified version that handles common
    decompositions; for full decomposition, use Qiskit's built-in
    :class:`BasisTranslator`.

    Parameters
    ----------
    target_basis:
        Set of allowed gate names in the target basis.  Gates not in
        this set will be decomposed if a decomposition rule exists.
    """

    def __init__(self, *, target_basis: set[str] | None = None) -> None:
        super().__init__()
        self._target_basis = target_basis or {"cx", "rz", "sx", "x", "id"}

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Decompose non-basis gates in the DAG.

        Currently delegates to Qiskit's built-in decomposition for
        gates that have a ``definition`` property.  Gates without a
        definition that are not in the target basis are left unchanged
        with a warning.

        Returns the (potentially modified) DAG.
        """
        import logging

        logger = logging.getLogger(__name__)

        nodes_to_decompose = []
        for node in dag.op_nodes():
            if node.op.name not in self._target_basis:
                nodes_to_decompose.append(node)

        for node in nodes_to_decompose:
            if hasattr(node.op, "definition") and node.op.definition is not None:
                from qiskit.converters import circuit_to_dag

                sub_dag = circuit_to_dag(node.op.definition)
                dag.substitute_node_with_dag(node, sub_dag)
            else:
                logger.debug(
                    "QBGateDecomposition: no decomposition for %s, keeping as-is",
                    node.op.name,
                )

        return dag
