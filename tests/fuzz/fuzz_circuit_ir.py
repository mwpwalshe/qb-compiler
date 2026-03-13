"""Fuzz the circuit IR and DAG round-trip.

Targets:
- ``QBCircuit`` with invalid / extreme gate parameters
- ``QBGate`` with NaN/Inf params, negative/huge qubit indices
- ``to_dag()`` / ``from_dag()`` round-trip
- Extremely deep circuits
"""
from __future__ import annotations

import contextlib
import math
import sys

import atheris

with atheris.instrument_imports():
    from qb_compiler.exceptions import CompilationError
    from qb_compiler.ir.circuit import QBCircuit
    from qb_compiler.ir.dag import QBDag
    from qb_compiler.ir.operations import QBGate


_GATE_NAMES = ["h", "x", "y", "z", "cx", "cz", "rz", "rx", "ry", "ecr", "swap", "ccx", "id"]
_SPECIAL_FLOATS = [0.0, 1.0, -1.0, math.pi, -math.pi, math.inf, -math.inf, math.nan, 1e308, -1e308]


def test_one_input(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    choice = fdp.ConsumeIntInRange(0, 4)

    try:
        if choice == 0:
            # GateOp with invalid/negative/huge qubit indices
            n_qubits = fdp.ConsumeIntInRange(1, 100)
            circ = QBCircuit(n_qubits=n_qubits)
            n_ops = fdp.ConsumeIntInRange(0, 50)
            for _ in range(n_ops):
                gate_name = fdp.PickValueInList(_GATE_NAMES)
                n_gate_qubits = fdp.ConsumeIntInRange(1, 4)
                qubits = tuple(
                    fdp.ConsumeIntInRange(-10, n_qubits + 10)
                    for _ in range(n_gate_qubits)
                )
                params = tuple(
                    fdp.PickValueInList(_SPECIAL_FLOATS)
                    if fdp.ConsumeBool()
                    else fdp.ConsumeRegularFloat()
                    for _ in range(fdp.ConsumeIntInRange(0, 3))
                )
                gate = QBGate(name=gate_name, qubits=qubits, params=params)
                circ.add_gate(gate)

        elif choice == 1:
            # GateOp with NaN/Inf params — test inverse, repr
            params = tuple(
                fdp.PickValueInList(_SPECIAL_FLOATS)
                for _ in range(fdp.ConsumeIntInRange(0, 5))
            )
            gate = QBGate(
                name=fdp.PickValueInList(_GATE_NAMES),
                qubits=(0,),
                params=params,
            )
            repr(gate)
            _ = gate.num_qubits
            _ = gate.is_parametric
            with contextlib.suppress(ValueError):
                gate.inverse()

        elif choice == 2:
            # to_dag() / from_dag() round-trip
            n_qubits = fdp.ConsumeIntInRange(1, 20)
            n_clbits = fdp.ConsumeIntInRange(0, 20)
            circ = QBCircuit(n_qubits=n_qubits, n_clbits=n_clbits)
            n_ops = fdp.ConsumeIntInRange(0, 100)
            for _ in range(n_ops):
                op_type = fdp.ConsumeIntInRange(0, 2)
                if op_type == 0:
                    gate_name = fdp.PickValueInList(_GATE_NAMES)
                    n_gate_qubits = fdp.ConsumeIntInRange(1, min(3, n_qubits))
                    qubits = tuple(
                        fdp.ConsumeIntInRange(0, n_qubits - 1)
                        for _ in range(n_gate_qubits)
                    )
                    circ.add_gate(QBGate(name=gate_name, qubits=qubits))
                elif op_type == 1 and n_clbits > 0:
                    q = fdp.ConsumeIntInRange(0, n_qubits - 1)
                    c = fdp.ConsumeIntInRange(0, n_clbits - 1)
                    circ.add_measurement(q, c)
                elif op_type == 2:
                    n_barrier = fdp.ConsumeIntInRange(1, n_qubits)
                    barrier_qubits = tuple(
                        fdp.ConsumeIntInRange(0, n_qubits - 1)
                        for _ in range(n_barrier)
                    )
                    circ.add_barrier(barrier_qubits)

            dag = QBDag.from_circuit(circ)
            rebuilt = dag.to_circuit()
            # Basic sanity: same number of qubits
            assert rebuilt.n_qubits == circ.n_qubits
            assert rebuilt.n_clbits == circ.n_clbits

        elif choice == 3:
            # Extremely deep circuit
            n_qubits = fdp.ConsumeIntInRange(1, 4)
            circ = QBCircuit(n_qubits=n_qubits)
            depth = fdp.ConsumeIntInRange(500, 2000)
            for _ in range(depth):
                q = fdp.ConsumeIntInRange(0, n_qubits - 1)
                circ.add_gate(QBGate(name="h", qubits=(q,)))
            _ = circ.depth
            _ = circ.gate_count

        else:
            # QBCircuit with bad n_qubits
            n = fdp.ConsumeIntInRange(-100, 0)
            QBCircuit(n_qubits=n)

    except (ValueError, IndexError, TypeError, OverflowError, CompilationError):
        pass


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
