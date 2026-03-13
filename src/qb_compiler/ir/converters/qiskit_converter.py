"""Convert between Qiskit ``QuantumCircuit`` and :class:`QBCircuit`.

Qiskit is an optional dependency.  All public functions raise
``ImportError`` with a helpful message if ``qiskit`` is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
                    # ClassicalRegister — use first bit index
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
