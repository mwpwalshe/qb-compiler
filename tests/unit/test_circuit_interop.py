"""The public entry points accept the same circuit types.

qb-compiler carries more than one class called ``QBCircuit``: the public one in
``qb_compiler.compiler`` that users build, and the IR in ``qb_compiler.ir.circuit`` that passes
operate on. On top of that, ``check_viability`` was written against Qiskit's ``QuantumCircuit``
while ``QBCompiler.compile`` was written against the public ``QBCircuit``.

The result was that the two headline features did not compose: calling ``check_viability`` and
then ``compile`` on the same object raised, either an ``AttributeError`` thrown from inside
Qiskit's transpiler or an ``InvalidCircuitError`` naming a type the caller never chose. These
tests pin the conversion so the entry points keep accepting the same set of inputs.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qiskit")

from qiskit import QuantumCircuit

from qb_compiler import QBCircuit, QBCompiler, check_viability
from qb_compiler.ir.converters.qiskit_converter import (
    any_to_compiler_circuit,
    any_to_qiskit,
)

BACKEND = "ibm_fez"


def _qb_ghz(n: int) -> QBCircuit:
    circ = QBCircuit(n)
    circ.h(0)
    for i in range(n - 1):
        circ.cx(i, i + 1)
    circ.measure_all()
    return circ


def _qiskit_ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc


class TestConverters:
    def test_public_circuit_to_qiskit_preserves_structure(self) -> None:
        qc = any_to_qiskit(_qb_ghz(4))
        assert qc.num_qubits == 4
        counts = qc.count_ops()
        assert counts.get("h") == 1
        assert counts.get("cx") == 3
        assert counts.get("measure") == 4

    def test_qiskit_to_public_circuit_preserves_structure(self) -> None:
        circ = any_to_compiler_circuit(_qiskit_ghz(4))
        assert isinstance(circ, QBCircuit)
        assert circ.n_qubits == 4
        assert sum(1 for op in circ.ops if op.name == "cx") == 3
        assert sum(1 for op in circ.ops if op.name == "measure") == 4

    def test_round_trip_is_stable(self) -> None:
        original = _qb_ghz(5)
        again = any_to_compiler_circuit(any_to_qiskit(original))
        gates = [op.name for op in original.ops if op.name != "measure"]
        round_tripped = [op.name for op in again.ops if op.name not in ("measure", "barrier")]
        assert gates == round_tripped

    def test_already_correct_type_is_passed_through_untouched(self) -> None:
        qc = _qiskit_ghz(3)
        assert any_to_qiskit(qc) is qc
        circ = _qb_ghz(3)
        assert any_to_compiler_circuit(circ) is circ

    def test_unconvertible_input_names_what_is_accepted(self) -> None:
        for bad in ("not a circuit", 42, None):
            with pytest.raises(TypeError, match="Expected a Qiskit"):
                any_to_qiskit(bad)


class TestEntryPointsCompose:
    """The point of the exercise: one circuit object works through both APIs."""

    def test_check_viability_accepts_the_public_circuit(self) -> None:
        result = check_viability(_qb_ghz(5), backend=BACKEND)
        assert result.status in ("VIABLE", "MARGINAL", "NOT_VIABLE")

    def test_check_viability_agrees_across_circuit_types(self) -> None:
        # Same circuit expressed two ways must not produce different verdicts.
        from_qb = check_viability(_qb_ghz(5), backend=BACKEND)
        from_qiskit = check_viability(_qiskit_ghz(5), backend=BACKEND)
        assert from_qb.status == from_qiskit.status
        assert from_qb.estimated_fidelity == pytest.approx(from_qiskit.estimated_fidelity)

    def test_compile_accepts_a_qiskit_circuit(self) -> None:
        result = QBCompiler.from_backend(BACKEND).compile(_qiskit_ghz(5))
        assert result.compiled_depth > 0

    def test_one_circuit_flows_through_both_entry_points(self) -> None:
        """The workflow the tutorials describe, for each circuit type."""
        compiler = QBCompiler.from_backend(BACKEND)
        for circuit in (_qb_ghz(5), _qiskit_ghz(5)):
            viability = check_viability(circuit, backend=BACKEND)
            compiled = compiler.compile(circuit)
            assert viability.status
            assert compiled.compiled_depth > 0

    def test_compile_still_rejects_a_non_circuit_with_a_useful_message(self) -> None:
        from qb_compiler.exceptions import InvalidCircuitError

        with pytest.raises(InvalidCircuitError, match="Expected a Qiskit"):
            QBCompiler.from_backend(BACKEND).compile("not a circuit")


class TestLayoutModelIsNotUsedOffItsHardware:
    def test_non_ibm_backends_compile_without_the_ibm_layout_model(self) -> None:
        """Only ibm_heron weights ship, so other providers must not load them.

        The predictor narrows the candidate qubit set before scoring, so applying an IBM model
        to trapped ion or other hardware degrades layouts quietly rather than failing loudly.
        """
        for backend in ("ionq_aria", "iqm_garnet"):
            circ = QBCircuit(4)
            circ.h(0)
            for i in range(3):
                circ.cx(i, i + 1)
            result = QBCompiler.from_backend(backend).compile(circ)
            assert result.compiled_depth > 0
