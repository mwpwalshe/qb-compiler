"""Benchmark compilation time for circuits of increasing size.

Uses pytest-benchmark to measure wall-clock time of QBCompiler.compile()
across 4, 8, 16, and 32 qubit GHZ circuits at optimization level 3.
"""

from __future__ import annotations

import pytest

from qb_compiler.compiler import QBCircuit, QBCompiler


def _build_ghz(n_qubits: int) -> QBCircuit:
    """Build an *n*-qubit GHZ circuit: H on q0, then CX chain."""
    circ = QBCircuit(n_qubits)
    circ.h(0)
    for i in range(n_qubits - 1):
        circ.cx(i, i + 1)
    circ.measure_all()
    return circ


def _compile_circuit(n_qubits: int) -> None:
    """Compile a GHZ circuit of *n_qubits* on ibm_fez with max optimisation."""
    circ = _build_ghz(n_qubits)
    compiler = QBCompiler.from_backend("ibm_fez", strategy="fidelity_optimal")
    compiler.compile(circ)


@pytest.mark.benchmark
def test_compile_4q(benchmark):
    """Benchmark compilation of a 4-qubit GHZ circuit."""
    benchmark(_compile_circuit, 4)


@pytest.mark.benchmark
def test_compile_8q(benchmark):
    """Benchmark compilation of an 8-qubit GHZ circuit."""
    benchmark(_compile_circuit, 8)


@pytest.mark.benchmark
def test_compile_16q(benchmark):
    """Benchmark compilation of a 16-qubit GHZ circuit."""
    benchmark(_compile_circuit, 16)


@pytest.mark.benchmark
def test_compile_32q(benchmark):
    """Benchmark compilation of a 32-qubit GHZ circuit."""
    benchmark(_compile_circuit, 32)
