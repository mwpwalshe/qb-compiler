"""Convert between PennyLane tapes and :class:`QBCircuit`.

PennyLane is an optional dependency.  All public functions raise
``ImportError`` with a helpful message if ``pennylane`` is not installed.

A direct mapping is implemented for the common gate set: ``Hadamard``,
``PauliX``, ``PauliY``, ``PauliZ``, ``S``, ``T``, ``RX``, ``RY``, ``RZ``,
``CNOT``, ``CZ`` and ``SWAP``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate

if TYPE_CHECKING:
    import pennylane as qml


def _ensure_pennylane() -> None:
    try:
        import pennylane  # noqa: F401
    except ImportError:
        raise ImportError(
            "PennyLane is required for this converter. "
            "Install it with: pip install 'qb-compiler[pennylane]'"
        ) from None


# PennyLane operation name -> (QBGate name, is_parametric)
_PL_NAME_MAP: dict[str, str] = {
    "Hadamard": "h",
    "H": "h",
    "PauliX": "x",
    "X": "x",
    "PauliY": "y",
    "Y": "y",
    "PauliZ": "z",
    "Z": "z",
    "S": "s",
    "T": "t",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "CNOT": "cx",
    "CX": "cx",
    "CZ": "cz",
    "SWAP": "swap",
}

# Reverse map for emission: QBGate name -> PennyLane class name.
_QB_TO_PL: dict[str, str] = {
    "h": "Hadamard",
    "x": "PauliX",
    "y": "PauliY",
    "z": "PauliZ",
    "s": "S",
    "t": "T",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "cx": "CNOT",
    "cnot": "CNOT",
    "cz": "CZ",
    "swap": "SWAP",
}


def _resolve_tape(tape_or_qnode: Any) -> Any:
    """Return an object exposing ``.operations`` (and maybe ``.measurements``).

    Accepts a :class:`~pennylane.tape.QuantumTape`, a QNode (via its
    ``.tape``/``.qtape``), or a plain list of operations.
    """
    if isinstance(tape_or_qnode, (list, tuple)):
        return tape_or_qnode
    for attr in ("tape", "qtape"):
        candidate = getattr(tape_or_qnode, attr, None)
        if candidate is not None and hasattr(candidate, "operations"):
            return candidate
    if hasattr(tape_or_qnode, "operations"):
        return tape_or_qnode
    raise TypeError(
        "from_pennylane expects a QuantumTape, a QNode, or a list of operations; "
        f"got {type(tape_or_qnode).__name__}"
    )


def from_pennylane(tape_or_qnode: Any) -> QBCircuit:
    """Convert a PennyLane tape (or list of operations) to a :class:`QBCircuit`.

    Wire labels are mapped to contiguous integer indices in sorted order.
    Measurement processes on the tape consume one classical bit per measured
    wire.

    Parameters
    ----------
    tape_or_qnode : pennylane.tape.QuantumTape | QNode | list
        Source operations.

    Returns
    -------
    QBCircuit
        Equivalent vendor-neutral circuit.
    """
    _ensure_pennylane()

    source = _resolve_tape(tape_or_qnode)
    if isinstance(source, (list, tuple)):
        operations = list(source)
        measurements: list[Any] = []
    else:
        operations = list(source.operations)
        measurements = list(getattr(source, "measurements", []))

    # Collect wires across ops and measurements, build a stable index map.
    wires: list[Any] = []
    seen: set[Any] = set()
    for op in operations:
        for w in op.wires:
            if w not in seen:
                seen.add(w)
                wires.append(w)
    for meas in measurements:
        for w in meas.wires:
            if w not in seen:
                seen.add(w)
                wires.append(w)

    if not wires:
        raise ValueError("PennyLane tape has no wires")

    try:
        ordered = sorted(wires)
    except TypeError:
        ordered = wires  # heterogeneous labels: preserve discovery order
    wire_map = {w: idx for idx, w in enumerate(ordered)}

    n_clbits = sum(len(meas.wires) for meas in measurements)
    circ = QBCircuit(n_qubits=len(ordered), n_clbits=n_clbits)

    for op in operations:
        name = _PL_NAME_MAP.get(op.name)
        if name is None:
            raise ValueError(f"Unsupported PennyLane operation for conversion: {op.name!r}")
        qubit_indices = tuple(wire_map[w] for w in op.wires)
        params = tuple(float(p) for p in op.parameters) if op.parameters else ()
        circ.add_gate(QBGate(name=name, qubits=qubit_indices, params=params))

    clbit_idx = 0
    for meas in measurements:
        for w in meas.wires:
            circ.add_measurement(wire_map[w], clbit_idx)
            clbit_idx += 1

    return circ


def to_pennylane(circuit: QBCircuit) -> qml.tape.QuantumTape:
    """Convert a :class:`QBCircuit` to a PennyLane :class:`~pennylane.tape.QuantumTape`.

    Qubit indices become integer wire labels.  Only the supported common gate
    set is emitted; other gate names raise ``ValueError``.  Measurements become
    ``qml.sample`` processes on the measured wire.

    Parameters
    ----------
    circuit : QBCircuit
        Source circuit.

    Returns
    -------
    pennylane.tape.QuantumTape
        Equivalent PennyLane tape.
    """
    _ensure_pennylane()
    import pennylane as qml

    from qb_compiler.ir.operations import QBBarrier, QBMeasure

    ops: list[Any] = []
    meas: list[Any] = []

    for op in circuit.iter_ops():
        if isinstance(op, QBMeasure):
            meas.append(qml.sample(wires=[op.qubit]))
            continue
        if isinstance(op, QBBarrier):
            # PennyLane has no scheduling barrier in a plain tape; skip.
            continue
        if not isinstance(op, QBGate):
            continue  # pragma: no cover

        pl_name = _QB_TO_PL.get(op.name)
        if pl_name is None:
            raise ValueError(f"Unsupported gate for PennyLane conversion: {op.name!r}")
        gate_cls = getattr(qml, pl_name)
        wires = list(op.qubits)
        if op.params:
            ops.append(gate_cls(*op.params, wires=wires))
        else:
            ops.append(gate_cls(wires=wires))

    return qml.tape.QuantumTape(ops=ops, measurements=meas)
