"""Convert between Qiskit ``QuantumCircuit`` and :class:`QBCircuit`.

Qiskit is an optional dependency.  All public functions raise
``ImportError`` with a helpful message if ``qiskit`` is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBBarrier, QBGate, QBMeasure

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def _ensure_qiskit() -> None:
    try:
        import qiskit  # noqa: F401
    except ImportError:
        raise ImportError(
            "Qiskit is required for this converter. "
            "Install it with: pip install 'qb-compiler[qiskit]'"
        ) from None


def from_qiskit(qc: QuantumCircuit) -> QBCircuit:
    """Convert a Qiskit :class:`QuantumCircuit` to a :class:`QBCircuit`.

    Parameters
    ----------
    qc : qiskit.circuit.QuantumCircuit
        Source circuit.

    Returns
    -------
    QBCircuit
        Equivalent vendor-neutral circuit.
    """
    _ensure_qiskit()
    from qiskit.circuit import Barrier, Measure

    n_qubits = qc.num_qubits
    n_clbits = qc.num_clbits
    circ = QBCircuit(n_qubits=n_qubits, n_clbits=n_clbits, name=qc.name or "")

    # Build qubit/clbit -> index maps
    qubit_map = {qubit: idx for idx, qubit in enumerate(qc.qubits)}
    clbit_map = {clbit: idx for idx, clbit in enumerate(qc.clbits)}

    for instruction in qc.data:
        op = instruction.operation
        qubits = tuple(qubit_map[q] for q in instruction.qubits)
        clbits = tuple(clbit_map[c] for c in instruction.clbits)

        if isinstance(op, Measure):
            circ.add_measurement(qubits[0], clbits[0])
        elif isinstance(op, Barrier):
            circ.add_barrier(qubits)
        else:
            # Extract condition if present
            condition = None
            if hasattr(op, "condition") and op.condition is not None:
                cond_clbit, cond_val = op.condition
                if hasattr(cond_clbit, "__iter__"):
                    # ClassicalRegister: use first bit index
                    cond_idx = clbit_map.get(next(iter(cond_clbit)), 0)
                else:
                    cond_idx = clbit_map.get(cond_clbit, 0)
                condition = (cond_idx, float(cond_val))

            params = tuple(float(p) for p in op.params) if op.params else ()
            gate = QBGate(
                name=op.name.lower(),
                qubits=qubits,
                params=params,
                condition=condition,
            )
            circ.add_gate(gate)

    return circ


def compiler_circuit_to_ir(circuit: Any) -> QBCircuit:
    """Convert the public :class:`qb_compiler.QBCircuit` into the compiler IR.

    ``qb_compiler.QBCircuit`` (defined in ``qb_compiler.compiler``) is the circuit users build,
    and it is a different, simpler class from the IR of the same name used here: a flat list of
    ``GateOp`` with measurement carried as a gate called ``"measure"`` rather than a distinct
    operation type. Converting into the IR lets both types share one Qiskit conversion instead
    of maintaining two gate tables that can drift apart.

    Measurements map onto clbits in the order encountered, which matches what
    ``QBCircuit.measure_all()`` produces.
    """
    measure_qubits = [op.qubits[0] for op in circuit.ops if op.name == "measure"]
    ir = QBCircuit(circuit.n_qubits, n_clbits=len(measure_qubits), name="")

    next_clbit = 0
    for op in circuit.ops:
        if op.name == "measure":
            ir.add_measurement(op.qubits[0], next_clbit)
            next_clbit += 1
        elif op.name == "barrier":
            ir.add_barrier(op.qubits or None)
        else:
            ir.add_gate(
                QBGate(name=op.name.lower(), qubits=tuple(op.qubits), params=tuple(op.params))
            )
    return ir


def ir_to_compiler_circuit(circuit: QBCircuit) -> Any:
    """Convert the compiler IR into the public :class:`qb_compiler.QBCircuit`.

    The inverse of :func:`compiler_circuit_to_ir`. Measurements collapse back onto a gate named
    ``"measure"``, since the public type carries no separate classical register.
    """
    from qb_compiler.compiler import GateOp
    from qb_compiler.compiler import QBCircuit as CompilerCircuit

    out = CompilerCircuit(circuit.n_qubits)
    for op in circuit.iter_ops():
        if isinstance(op, QBMeasure):
            out.ops.append(GateOp(name="measure", qubits=(op.qubit,)))
        elif isinstance(op, QBBarrier):
            out.ops.append(GateOp(name="barrier", qubits=tuple(op.qubits)))
        elif isinstance(op, QBGate):
            out.ops.append(GateOp(name=op.name, qubits=tuple(op.qubits), params=tuple(op.params)))
    return out


def any_to_compiler_circuit(circuit: Any) -> Any:
    """Convert any circuit this package understands into ``qb_compiler.QBCircuit``.

    Accepts the public ``qb_compiler.QBCircuit`` (returned unchanged), a Qiskit
    ``QuantumCircuit``, or the IR ``QBCircuit``. The counterpart to :func:`any_to_qiskit`, so
    the public entry points accept the same set of inputs in both directions.
    """
    from qb_compiler.compiler import QBCircuit as CompilerCircuit

    if isinstance(circuit, CompilerCircuit):
        return circuit
    if isinstance(circuit, QBCircuit):
        return ir_to_compiler_circuit(circuit)

    _ensure_qiskit()
    from qiskit.circuit import QuantumCircuit as _QuantumCircuit

    if isinstance(circuit, _QuantumCircuit):
        return ir_to_compiler_circuit(from_qiskit(circuit))

    raise TypeError(
        f"Cannot convert {type(circuit).__name__} to a qb_compiler.QBCircuit. Expected a Qiskit "
        f"QuantumCircuit, a qb_compiler.QBCircuit, or a qb_compiler.ir QBCircuit."
    )


def any_to_qiskit(circuit: Any) -> QuantumCircuit:
    """Convert any circuit this package understands into a Qiskit ``QuantumCircuit``.

    Accepts a Qiskit circuit (returned unchanged), the public ``qb_compiler.QBCircuit``, or the
    IR ``QBCircuit``. Public entry points use this so a caller does not have to know which of
    the circuit types a given function was originally written against.
    """
    _ensure_qiskit()
    from qiskit.circuit import QuantumCircuit as _QuantumCircuit

    if isinstance(circuit, _QuantumCircuit):
        return circuit
    if isinstance(circuit, QBCircuit):
        return to_qiskit(circuit)

    from qb_compiler.compiler import QBCircuit as CompilerCircuit

    if isinstance(circuit, CompilerCircuit):
        return to_qiskit(compiler_circuit_to_ir(circuit))

    raise TypeError(
        f"Cannot convert {type(circuit).__name__} to a Qiskit circuit. Expected a Qiskit "
        f"QuantumCircuit, a qb_compiler.QBCircuit, or a qb_compiler.ir QBCircuit."
    )


def to_qiskit(circuit: QBCircuit) -> QuantumCircuit:
    """Convert a :class:`QBCircuit` to a Qiskit :class:`QuantumCircuit`.

    Parameters
    ----------
    circuit : QBCircuit
        Source circuit.

    Returns
    -------
    qiskit.circuit.QuantumCircuit
        Equivalent Qiskit circuit.
    """
    _ensure_qiskit()
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import standard_gates

    qc = QuantumCircuit(circuit.n_qubits, circuit.n_clbits, name=circuit.name)

    # Build a lookup for standard gate classes by lowercase name
    _gate_map: dict[str, type] = {}
    for attr_name in dir(standard_gates):
        cls = getattr(standard_gates, attr_name)
        if isinstance(cls, type) and hasattr(cls, "name"):
            try:
                inst = cls.__new__(cls)  # type: ignore[call-overload]
                if hasattr(inst, "name"):
                    _gate_map[inst.name.lower()] = cls
            except Exception:
                pass

    # Common overrides / aliases that the introspection might miss
    _name_overrides = {
        "cx": "CXGate",
        "cz": "CZGate",
        "cy": "CYGate",
        "ch": "CHGate",
        "ccx": "CCXGate",
        "swap": "SwapGate",
        "cswap": "CSwapGate",
        "ecr": "ECRGate",
        "h": "HGate",
        "x": "XGate",
        "y": "YGate",
        "z": "ZGate",
        "s": "SGate",
        "sdg": "SdgGate",
        "t": "TGate",
        "tdg": "TdgGate",
        "sx": "SXGate",
        "sxdg": "SXdgGate",
        "rx": "RXGate",
        "ry": "RYGate",
        "rz": "RZGate",
        "p": "PhaseGate",
        "cp": "CPhaseGate",
        "crx": "CRXGate",
        "cry": "CRYGate",
        "crz": "CRZGate",
        "rxx": "RXXGate",
        "ryy": "RYYGate",
        "rzz": "RZZGate",
        "u": "UGate",
        "u1": "U1Gate",
        "u2": "U2Gate",
        "u3": "U3Gate",
        "id": "IGate",
    }
    for gname, cls_name in _name_overrides.items():
        cls = getattr(standard_gates, cls_name, None)
        if cls is not None:
            _gate_map[gname] = cls

    for op in circuit.iter_ops():
        if isinstance(op, QBMeasure):
            qc.measure(op.qubit, op.clbit)
        elif isinstance(op, QBBarrier):
            qc.barrier(*op.qubits)
        elif isinstance(op, QBGate):
            gate_cls = _gate_map.get(op.name)
            if gate_cls is not None:
                try:
                    gate_inst = gate_cls(*op.params) if op.params else gate_cls()
                except TypeError:
                    # Some gates need specific arg handling
                    gate_inst = gate_cls(*op.params)
                qc.append(gate_inst, list(op.qubits))
            else:
                # Fallback: use a generic unitary placeholder via append
                from qiskit.circuit import Gate as QiskitGate

                generic = QiskitGate(op.name, len(op.qubits), list(op.params))
                qc.append(generic, list(op.qubits))

    return qc
