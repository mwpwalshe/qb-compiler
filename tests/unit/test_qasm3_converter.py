"""Tests for the OpenQASM 3 converter.

``to_qasm3`` (emission) works with core Qiskit and is exercised directly.
``from_qasm3`` (parsing) needs the optional ``qiskit_qasm3_import`` package
and is gated behind ``importorskip``.
"""

from __future__ import annotations

import pytest

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from tests.conftest import requires_qiskit


@requires_qiskit
class TestToQasm3:
    """QBCircuit -> OpenQASM 3 string (no extra needed)."""

    def test_emits_openqasm3_header(self) -> None:
        from qb_compiler.ir.converters.qasm3_converter import to_qasm3

        circ = QBCircuit(n_qubits=2, n_clbits=2)
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))

        qasm = to_qasm3(circ)
        assert isinstance(qasm, str)
        assert "OPENQASM 3" in qasm


@requires_qiskit
class TestFromQasm3:
    """OpenQASM 3 string -> QBCircuit (needs qiskit_qasm3_import)."""

    def test_parse_bell(self) -> None:
        pytest.importorskip("qiskit_qasm3_import")
        from qb_compiler.ir.converters.qasm3_converter import from_qasm3

        qasm = (
            "OPENQASM 3;\n"
            'include "stdgates.inc";\n'
            "qubit[2] q;\n"
            "bit[2] c;\n"
            "h q[0];\n"
            "cx q[0], q[1];\n"
            "c[0] = measure q[0];\n"
            "c[1] = measure q[1];\n"
        )
        circ = from_qasm3(qasm)
        assert circ.n_qubits == 2
        assert circ.gate_count == 2

    def test_roundtrip(self) -> None:
        pytest.importorskip("qiskit_qasm3_import")
        from qb_compiler.ir.converters.qasm3_converter import from_qasm3, to_qasm3

        original = QBCircuit(n_qubits=2)
        original.add_gate(QBGate(name="h", qubits=(0,)))
        original.add_gate(QBGate(name="cx", qubits=(0, 1)))

        rebuilt = from_qasm3(to_qasm3(original))
        assert rebuilt.n_qubits == original.n_qubits
        assert rebuilt.gate_count == original.gate_count


@requires_qiskit
def test_from_qasm3_missing_extra_raises_importerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """If qiskit_qasm3_import is absent, from_qasm3 must raise a helpful ImportError."""
    import builtins

    import qb_compiler.ir.converters.qasm3_converter as mod

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "qiskit_qasm3_import":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ImportError, match=r"qb-compiler\[qasm3\]"):
        mod.from_qasm3("OPENQASM 3;\nqubit[1] q;\nh q[0];\n")
