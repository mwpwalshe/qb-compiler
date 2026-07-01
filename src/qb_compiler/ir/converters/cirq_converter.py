"""Convert between Cirq ``Circuit`` and :class:`QBCircuit`.

Cirq is an optional dependency.  All public functions raise ``ImportError``
with a helpful message if ``cirq`` is not installed.

A direct mapping is implemented for the common gate set: ``H``, ``X``, ``Y``,
``Z``, ``S``, ``T``, ``RX``, ``RY``, ``RZ``, ``CNOT``/``CX``, ``CZ``, ``SWAP``
and measurement.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate

if TYPE_CHECKING:
    import cirq


def _ensure_cirq() -> None:
    try:
        import cirq  # noqa: F401
    except ImportError:
        raise ImportError(
            "Cirq is required for this converter. Install it with: pip install 'qb-compiler[cirq]'"
        ) from None


def _cirq_gate_to_qb(gate: cirq.Gate) -> tuple[str, tuple[float, ...]]:
    """Map a Cirq gate to a ``(name, params)`` pair for :class:`QBGate`.

    Raises ``ValueError`` for gates outside the supported common set.
    """
    import cirq

    # Parametric rotations first (subclasses of *PowGate*).
    if isinstance(gate, cirq.Rx):
        return "rx", (float(gate.exponent) * math.pi,)
    if isinstance(gate, cirq.Ry):
        return "ry", (float(gate.exponent) * math.pi,)
    if isinstance(gate, cirq.Rz):
        return "rz", (float(gate.exponent) * math.pi,)

    # Non-parametric named gates via value equality.
    if gate == cirq.H:
        return "h", ()
    if gate == cirq.X:
        return "x", ()
    if gate == cirq.Y:
        return "y", ()
    if gate == cirq.Z:
        return "z", ()
    if gate == cirq.S:
        return "s", ()
    if gate == cirq.T:
        return "t", ()
    if gate == cirq.CNOT:
        return "cx", ()
    if gate == cirq.CZ:
        return "cz", ()
    if gate == cirq.SWAP:
        return "swap", ()

    raise ValueError(f"Unsupported Cirq gate for conversion: {gate!r}")


def from_cirq(circuit: cirq.Circuit) -> QBCircuit:
    """Convert a Cirq :class:`~cirq.Circuit` to a :class:`QBCircuit`.

    Qubits are mapped to contiguous integer indices in Cirq's sorted qubit
    order.  Measurements consume one classical bit per measured qubit, in the
    order encountered.

    Parameters
    ----------
    circuit : cirq.Circuit
        Source circuit.

    Returns
    -------
    QBCircuit
        Equivalent vendor-neutral circuit.
    """
    _ensure_cirq()
    import cirq

    qubits = sorted(circuit.all_qubits())
    if not qubits:
        raise ValueError("Cirq circuit has no qubits")
    qubit_map = {q: idx for idx, q in enumerate(qubits)}

    # Count classical bits required by measurements.
    n_clbits = 0
    for op in circuit.all_operations():
        if isinstance(op.gate, cirq.MeasurementGate):
            n_clbits += len(op.qubits)

    circ = QBCircuit(n_qubits=len(qubits), n_clbits=n_clbits)

    clbit_idx = 0
    for op in circuit.all_operations():
        gate = op.gate
        qubit_indices = tuple(qubit_map[q] for q in op.qubits)

        if isinstance(gate, cirq.MeasurementGate):
            for q in qubit_indices:
                circ.add_measurement(q, clbit_idx)
                clbit_idx += 1
            continue

        if gate is None:
            raise ValueError(f"Cirq operation has no gate: {op!r}")

        name, params = _cirq_gate_to_qb(gate)
        circ.add_gate(QBGate(name=name, qubits=qubit_indices, params=params))

    return circ


def to_cirq(circuit: QBCircuit) -> cirq.Circuit:
    """Convert a :class:`QBCircuit` to a Cirq :class:`~cirq.Circuit`.

    Qubit indices become :class:`cirq.LineQubit` objects.  Only the supported
    common gate set is emitted; other gate names raise ``ValueError``.

    Parameters
    ----------
    circuit : QBCircuit
        Source circuit.

    Returns
    -------
    cirq.Circuit
        Equivalent Cirq circuit.
    """
    _ensure_cirq()
    import cirq

    qubits = cirq.LineQubit.range(circuit.n_qubits)

    cirq_circuit = cirq.Circuit()
    clbit_counter = 0

    from qb_compiler.ir.operations import QBBarrier, QBMeasure

    for op in circuit.iter_ops():
        if isinstance(op, QBMeasure):
            cirq_circuit.append(cirq.measure(qubits[op.qubit], key=f"m{clbit_counter}_{op.clbit}"))
            clbit_counter += 1
            continue
        if isinstance(op, QBBarrier):
            # Cirq has no direct barrier; skip (scheduling hint only).
            continue
        if not isinstance(op, QBGate):
            continue  # pragma: no cover

        targets = [qubits[q] for q in op.qubits]
        name = op.name
        if name == "h":
            cirq_circuit.append(cirq.H(*targets))
        elif name == "x":
            cirq_circuit.append(cirq.X(*targets))
        elif name == "y":
            cirq_circuit.append(cirq.Y(*targets))
        elif name == "z":
            cirq_circuit.append(cirq.Z(*targets))
        elif name == "s":
            cirq_circuit.append(cirq.S(*targets))
        elif name == "t":
            cirq_circuit.append(cirq.T(*targets))
        elif name == "rx":
            cirq_circuit.append(cirq.rx(op.params[0])(*targets))
        elif name == "ry":
            cirq_circuit.append(cirq.ry(op.params[0])(*targets))
        elif name == "rz":
            cirq_circuit.append(cirq.rz(op.params[0])(*targets))
        elif name in ("cx", "cnot"):
            cirq_circuit.append(cirq.CNOT(*targets))
        elif name == "cz":
            cirq_circuit.append(cirq.CZ(*targets))
        elif name == "swap":
            cirq_circuit.append(cirq.SWAP(*targets))
        else:
            raise ValueError(f"Unsupported gate for Cirq conversion: {name!r}")

    return cirq_circuit
