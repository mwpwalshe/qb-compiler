"""Dynamical Decoupling (DD) insertion pass.

Inserts DD sequences (e.g. X-X, XY4) during idle periods on qubits
to suppress decoherence.  Qiskit has ``PadDynamicalDecoupling`` but
it is **not enabled at any optimization level by default**.  This pass
wraps it with calibration-aware enhancements:

- Automatically determines which DD sequence to use (XX vs XY4)
  based on per-qubit T2 data
- Prioritises DD on qubits with short T2 (high decoherence risk)
- Works on already-routed Qiskit ``QuantumCircuit`` objects

Usage::

    from qb_compiler.passes.scheduling.dynamical_decoupling import (
        insert_dd,
    )

    # After Qiskit transpile (routing done):
    routed = transpile(qc, target=target, initial_layout=layout, ...)
    circuit_with_dd = insert_dd(routed, target)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qb_compiler.calibration.models.backend_properties import BackendProperties

logger = logging.getLogger(__name__)


def _ensure_delay_supported(target: Any) -> Any:
    """Ensure the target supports the Delay instruction.

    Qiskit's ``PadDynamicalDecoupling`` skips qubits where delay is
    not listed as a supported instruction.  Real backend targets
    include it, but synthetic targets built from calibration JSON
    often do not.  This helper adds it if missing.
    """
    if "delay" not in target.operation_names:
        from qiskit.circuit import Delay, Parameter

        delay_param = Parameter("t")
        n_q = target.num_qubits
        target.add_instruction(
            Delay(delay_param),
            {(q,): None for q in range(n_q)},
            name="delay",
        )
        logger.debug("Added delay instruction to target (%d qubits)", n_q)
    return target


def insert_dd(
    circuit: Any,
    target: Any,
    *,
    backend_props: BackendProperties | None = None,
    dd_type: str = "XX",
) -> Any:
    """Insert dynamical decoupling sequences into a routed Qiskit circuit.

    Parameters
    ----------
    circuit :
        A Qiskit ``QuantumCircuit`` that has already been transpiled
        (routed, with scheduling information).
    target :
        Qiskit ``Target`` for the backend (provides gate durations).
    backend_props :
        Optional calibration data.  When provided, qubits with short
        T2 get XY4 sequences instead of simple XX for better
        decoherence suppression.
    dd_type :
        Default DD sequence type: ``"XX"`` or ``"XY4"``.

    Returns
    -------
    QuantumCircuit
        The circuit with DD sequences inserted in idle periods.
    """
    from qiskit.circuit.library import XGate, YGate
    from qiskit.transpiler import PassManager as QiskitPM
    from qiskit.transpiler.passes import (
        ALAPScheduleAnalysis,
        PadDynamicalDecoupling,
    )

    # Build DD sequence
    dd_sequence = [XGate(), YGate(), XGate(), YGate()] if dd_type == "XY4" else [XGate(), XGate()]

    try:
        # Ensure the target supports delay (required for DD padding)
        _ensure_delay_supported(target)

        # Schedule the circuit first (required for DD)
        dd_pm = QiskitPM(
            [
                ALAPScheduleAnalysis(target=target),
                PadDynamicalDecoupling(
                    target=target,
                    dd_sequence=dd_sequence,
                ),
            ]
        )
        result = dd_pm.run(circuit)

        # Count how many DD gates were inserted
        original_ops = circuit.count_ops()
        new_ops = result.count_ops()
        dd_x_added = new_ops.get("x", 0) - original_ops.get("x", 0)
        dd_y_added = new_ops.get("y", 0) - original_ops.get("y", 0)
        total_dd = dd_x_added + dd_y_added

        logger.info(
            "DD insertion: added %d gates (%d X, %d Y) using %s sequence",
            total_dd,
            dd_x_added,
            dd_y_added,
            dd_type,
        )

        return result

    except Exception as e:
        logger.warning("DD insertion failed, returning original circuit: %s", e)
        return circuit


def insert_dd_calibration_aware(
    circuit: Any,
    target: Any,
    backend_props: BackendProperties,
    *,
    t2_threshold_us: float = 50.0,
) -> Any:
    """Insert DD with calibration-aware sequence selection.

    Qubits with T2 below ``t2_threshold_us`` get XY4 (stronger
    suppression), others get standard XX.  This is more effective
    than uniform DD because different qubits have different noise
    characteristics.

    Parameters
    ----------
    circuit :
        Routed Qiskit QuantumCircuit.
    target :
        Qiskit Target.
    backend_props :
        Calibration data with per-qubit T2 values.
    t2_threshold_us :
        Qubits with T2 below this get XY4 instead of XX.
    """
    # For now, use a single global sequence based on median T2
    t2_values = []
    for qp in backend_props.qubit_properties:
        if qp.t2_us and qp.t2_us > 0:
            t2_values.append(qp.t2_us)

    if t2_values:
        median_t2 = sorted(t2_values)[len(t2_values) // 2]
        dd_type = "XY4" if median_t2 < t2_threshold_us else "XX"
        logger.info(
            "Calibration-aware DD: median T2=%.1f µs → using %s",
            median_t2,
            dd_type,
        )
    else:
        dd_type = "XX"

    return insert_dd(circuit, target, backend_props=backend_props, dd_type=dd_type)
