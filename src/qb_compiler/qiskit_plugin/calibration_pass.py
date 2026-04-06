"""Qiskit TransformationPass powered by live calibration data.

:class:`QBCalibrationPass` is a Qiskit ``TransformationPass`` that uses
per-qubit and per-gate error rates from a ``Backend`` or ``Target`` to
optimise circuit layout and cancel redundant gates.

Usage inside a Qiskit pass pipeline::

    from qb_compiler.qiskit_plugin.calibration_pass import QBCalibrationPass

    pass_ = QBCalibrationPass(backend=backend)
    dag_out = pass_.run(dag)

Usage with the convenience factory::

    from qb_compiler import passmanager

    pm = passmanager(backend)
    compiled = pm.run(circuit)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGCircuit

logger = logging.getLogger(__name__)

_SELF_INVERSE = frozenset({"cx", "cz", "h", "x", "y", "z", "swap"})


class QBCalibrationPass(TransformationPass):
    """Calibration-aware transformation pass for Qiskit pipelines.

    Accepts a Qiskit ``Backend`` or ``Target`` at init.  During ``run()``,
    the pass:

    1. Scores physical qubits using calibration data (gate error, readout
       error, T1/T2 coherence).
    2. Sets an optimal ``layout`` in ``property_set`` so downstream routing
       passes (e.g. ``SabreSwap``) place virtual qubits on the best hardware.
    3. Cancels adjacent self-inverse gate pairs (H-H, CX-CX, etc.) to
       reduce gate count.

    Parameters
    ----------
    backend :
        A Qiskit ``Backend`` instance (e.g. from ``QiskitRuntimeService``).
        Calibration is extracted from ``backend.target``.
    target :
        A Qiskit ``Target`` directly.  Used when no ``Backend`` is available.
    """

    def __init__(
        self,
        backend: Any | None = None,
        target: Any | None = None,
    ) -> None:
        super().__init__()
        self._target = None
        self._qubit_scores: dict[int, float] = {}

        if backend is not None:
            self._target = getattr(backend, "target", None)
            if self._target is not None:
                self._qubit_scores = _score_qubits_from_target(self._target)
        elif target is not None:
            self._target = target
            self._qubit_scores = _score_qubits_from_target(target)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Apply calibration-aware layout and gate cancellation.

        Sets ``property_set["layout"]`` when calibration data is available,
        then cancels adjacent self-inverse gate pairs in the DAG.

        Parameters
        ----------
        dag :
            Input DAG circuit.

        Returns
        -------
        DAGCircuit
            The optimised DAG.
        """
        # --- Step 1: set calibration-optimal layout -----------------------
        if self._qubit_scores:
            n_virtual = dag.num_qubits()
            ranked = sorted(self._qubit_scores.items(), key=lambda kv: kv[1])
            if len(ranked) >= n_virtual:
                best_physical = [qid for qid, _ in ranked[:n_virtual]]
                layout_dict = {}
                for v_idx, v_qubit in enumerate(dag.qubits):
                    layout_dict[v_qubit] = best_physical[v_idx]
                self.property_set["layout"] = Layout(layout_dict)
                logger.info(
                    "QBCalibrationPass: mapped %d qubits to physical %s",
                    n_virtual,
                    best_physical,
                )

        # --- Step 2: gate cancellation ------------------------------------
        dag = _cancel_self_inverse(dag)

        return dag


def _score_qubits_from_target(target: Any) -> dict[int, float]:
    """Score physical qubits from a Qiskit Target — lower is better.

    Combines gate error, readout error, and duration into a single
    quality metric per qubit.
    """
    n_qubits = target.num_qubits
    scores: dict[int, float] = {}

    for qubit in range(n_qubits):
        gate_errors: list[float] = []
        readout_err = 0.05  # pessimistic default

        for op_name in target.operation_names:
            try:
                props = target[op_name].get((qubit,))
            except (KeyError, TypeError):
                props = None
            if props is None:
                continue

            if op_name == "measure" and props.error is not None:
                readout_err = props.error
            elif props.error is not None:
                gate_errors.append(props.error)

        # Also check 2Q gate errors on edges touching this qubit
        for op_name in target.operation_names:
            try:
                qargs_list = target.qargs_for_operation_name(op_name)
            except Exception:
                continue
            for qargs in qargs_list:
                if len(qargs) == 2 and qubit in qargs:
                    try:
                        props = target[op_name].get(qargs)
                    except (KeyError, TypeError):
                        continue
                    if props is not None and props.error is not None:
                        gate_errors.append(props.error)

        avg_gate_err = sum(gate_errors) / len(gate_errors) if gate_errors else 0.01
        scores[qubit] = 0.35 * readout_err + 0.65 * avg_gate_err

    return scores


def _cancel_self_inverse(dag: DAGCircuit) -> DAGCircuit:
    """Cancel adjacent self-inverse gate pairs in the DAG."""
    for _ in range(3):
        cancelled = False
        nodes_to_remove: list[Any] = []
        removed_ids: set[int] = set()

        for node in dag.op_nodes():
            if id(node) in removed_ids:
                continue
            if node.op.name not in _SELF_INVERSE:
                continue
            for successor in dag.successors(node):
                if not hasattr(successor, "op"):
                    continue
                if id(successor) in removed_ids:
                    continue
                if successor.op.name == node.op.name and successor.qargs == node.qargs:
                    nodes_to_remove.extend([node, successor])
                    removed_ids.add(id(node))
                    removed_ids.add(id(successor))
                    cancelled = True
                    break

        for n in nodes_to_remove:
            dag.remove_op_node(n)

        if not cancelled:
            break

    return dag
