"""Tests for IR converters: qiskit_converter and openqasm_converter."""

from __future__ import annotations

import math

import pytest

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.converters.openqasm_converter import from_qasm, to_qasm
from qb_compiler.ir.operations import QBGate
from tests.conftest import requires_qiskit

# ── OpenQASM converter ───────────────────────────────────────────────


class TestOpenQASMFromQasm:
    """Tests for from_qasm (QASM string -> QBCircuit)."""

    def test_simple_bell_state(self) -> None:
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        """
        circ = from_qasm(qasm)
        assert circ.n_qubits == 2
        assert circ.n_clbits == 2
        assert circ.gate_count == 2  # h + cx
        assert len(circ.measurements) == 2

    def test_parametric_gate(self) -> None:
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rz(pi) q[0];
        """
        circ = from_qasm(qasm)
        gates = circ.gates
        assert len(gates) == 1
        assert gates[0].name == "rz"
        assert gates[0].params[0] == pytest.approx(math.pi)

    def test_multiple_registers(self) -> None:
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg a[2];
        qreg b[3];
        creg c[5];
        h a[0];
        cx b[0],b[1];
        """
        circ = from_qasm(qasm)
        assert circ.n_qubits == 5
        assert circ.n_clbits == 5
        # h on a[0] -> qubit 0, cx on b[0],b[1] -> qubits 2,3
        gates = circ.gates
        assert gates[0].qubits == (0,)
        assert gates[1].qubits == (2, 3)

    def test_no_qreg_raises(self) -> None:
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        """
        with pytest.raises(ValueError, match="No qreg"):
            from_qasm(qasm)

    def test_comments_stripped(self) -> None:
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        // This is a comment
        qreg q[1];
        h q[0]; // inline comment
        """
        circ = from_qasm(qasm)
        assert circ.gate_count == 1

    def test_barrier(self) -> None:
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        barrier q[0],q[1];
        cx q[0],q[1];
        """
        circ = from_qasm(qasm)
        ops = circ.operations
        assert len(ops) == 3  # h, barrier, cx


class TestOpenQASMToQasm:
    """Tests for to_qasm (QBCircuit -> QASM string)."""

    def test_simple_circuit(self) -> None:
        circ = QBCircuit(n_qubits=2, n_clbits=2)
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
        circ.add_measurement(0, 0)
        circ.add_measurement(1, 1)

        qasm_str = to_qasm(circ)
        assert "OPENQASM 2.0;" in qasm_str
        assert "qreg q[2];" in qasm_str
        assert "creg c[2];" in qasm_str
        assert "h q[0];" in qasm_str
        assert "cx q[0],q[1];" in qasm_str
        assert "measure q[0] -> c[0];" in qasm_str

    def test_parametric_gate_output(self) -> None:
        circ = QBCircuit(n_qubits=1)
        circ.add_gate(QBGate(name="rz", qubits=(0,), params=(math.pi,)))

        qasm_str = to_qasm(circ)
        assert "rz(pi)" in qasm_str

    def test_no_clbits_no_creg(self) -> None:
        circ = QBCircuit(n_qubits=2, n_clbits=0)
        circ.add_gate(QBGate(name="h", qubits=(0,)))

        qasm_str = to_qasm(circ)
        assert "creg" not in qasm_str


class TestOpenQASMRoundTrip:
    """Roundtrip tests: QBCircuit -> QASM -> QBCircuit."""

    def test_roundtrip_gates_preserved(self) -> None:
        circ = QBCircuit(n_qubits=3, n_clbits=3)
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
        circ.add_gate(QBGate(name="cx", qubits=(1, 2)))
        circ.add_measurement(0, 0)
        circ.add_measurement(1, 1)
        circ.add_measurement(2, 2)

        qasm_str = to_qasm(circ)
        reconstructed = from_qasm(qasm_str)

        assert reconstructed.n_qubits == circ.n_qubits
        assert reconstructed.n_clbits == circ.n_clbits
        assert reconstructed.gate_count == circ.gate_count
        assert len(reconstructed.measurements) == len(circ.measurements)

    def test_roundtrip_parametric_gates(self) -> None:
        circ = QBCircuit(n_qubits=1)
        circ.add_gate(QBGate(name="rz", qubits=(0,), params=(math.pi / 4,)))

        qasm_str = to_qasm(circ)
        reconstructed = from_qasm(qasm_str)

        assert reconstructed.gate_count == 1
        assert reconstructed.gates[0].params[0] == pytest.approx(math.pi / 4, rel=1e-6)


# ── Qiskit converter ────────────────────────────────────────────────


@requires_qiskit
class TestQiskitFromQiskit:
    """Tests for from_qiskit (Qiskit QuantumCircuit -> QBCircuit)."""

    def test_bell_circuit(self) -> None:
        from qiskit.circuit import QuantumCircuit

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        from qb_compiler.ir.converters.qiskit_converter import from_qiskit

        circ = from_qiskit(qc)
        assert circ.n_qubits == 2
        assert circ.n_clbits == 2
        assert circ.gate_count == 2
        assert len(circ.measurements) == 2

    def test_preserves_name(self) -> None:
        from qiskit.circuit import QuantumCircuit

        from qb_compiler.ir.converters.qiskit_converter import from_qiskit

        qc = QuantumCircuit(1, name="my_circuit")
        qc.h(0)
        circ = from_qiskit(qc)
        assert circ.name == "my_circuit"

    def test_parametric_gate(self) -> None:
        from qiskit.circuit import QuantumCircuit

        from qb_compiler.ir.converters.qiskit_converter import from_qiskit

        qc = QuantumCircuit(1)
        qc.rz(math.pi / 3, 0)

        circ = from_qiskit(qc)
        assert circ.gate_count == 1
        assert circ.gates[0].name == "rz"
        assert circ.gates[0].params[0] == pytest.approx(math.pi / 3)

    def test_barrier(self) -> None:
        from qiskit.circuit import QuantumCircuit

        from qb_compiler.ir.converters.qiskit_converter import from_qiskit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)

        circ = from_qiskit(qc)
        assert len(circ.operations) == 3


@requires_qiskit
class TestQiskitToQiskit:
    """Tests for to_qiskit (QBCircuit -> Qiskit QuantumCircuit)."""

    def test_simple_circuit(self) -> None:
        from qiskit.circuit import QuantumCircuit

        from qb_compiler.ir.converters.qiskit_converter import to_qiskit

        circ = QBCircuit(n_qubits=2, n_clbits=2)
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
        circ.add_measurement(0, 0)

        qc = to_qiskit(circ)
        assert isinstance(qc, QuantumCircuit)
        assert qc.num_qubits == 2
        assert qc.num_clbits == 2

    def test_preserves_name(self) -> None:
        from qb_compiler.ir.converters.qiskit_converter import to_qiskit

        circ = QBCircuit(n_qubits=1, name="test_name")
        circ.add_gate(QBGate(name="x", qubits=(0,)))
        qc = to_qiskit(circ)
        assert qc.name == "test_name"


@requires_qiskit
class TestQiskitRoundTrip:
    """Roundtrip: Qiskit -> QBCircuit -> Qiskit."""

    def test_roundtrip_preserves_gate_count(self) -> None:
        from qiskit.circuit import QuantumCircuit

        from qb_compiler.ir.converters.qiskit_converter import from_qiskit, to_qiskit

        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure([0, 1, 2], [0, 1, 2])

        circ = from_qiskit(qc)
        qc2 = to_qiskit(circ)

        assert qc2.num_qubits == qc.num_qubits
        assert qc2.num_clbits == qc.num_clbits
        # Gate count (excluding measurements/barriers)
        orig_ops = sum(1 for inst in qc.data if inst.operation.name not in ("measure", "barrier"))
        new_ops = sum(1 for inst in qc2.data if inst.operation.name not in ("measure", "barrier"))
        assert new_ops == orig_ops

    def test_roundtrip_qb_to_qiskit_to_qb(self) -> None:
        from qb_compiler.ir.converters.qiskit_converter import from_qiskit, to_qiskit

        original = QBCircuit(n_qubits=2, n_clbits=2, name="roundtrip")
        original.add_gate(QBGate(name="h", qubits=(0,)))
        original.add_gate(QBGate(name="cx", qubits=(0, 1)))
        original.add_measurement(0, 0)
        original.add_measurement(1, 1)

        qc = to_qiskit(original)
        reconstructed = from_qiskit(qc)

        assert reconstructed.n_qubits == original.n_qubits
        assert reconstructed.n_clbits == original.n_clbits
        assert reconstructed.gate_count == original.gate_count
        assert len(reconstructed.measurements) == len(original.measurements)
