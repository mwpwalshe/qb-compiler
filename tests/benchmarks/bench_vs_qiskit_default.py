"""Benchmark comparing qb-compiler output vs Qiskit default transpile.

Uses pytest-benchmark to measure compilation time and reports estimated
fidelity for both qb-compiler (level 3 / fidelity_optimal) and Qiskit
default transpiler (level 3).

Tests are marked with @pytest.mark.benchmark so they can be deselected
in normal CI runs: ``pytest -m "not benchmark"``.
"""

from __future__ import annotations

import math

import pytest

from qb_compiler.compiler import QBCircuit, QBCompiler
from tests.conftest import requires_qiskit

# ── circuit builders ─────────────────────────────────────────────────


def _build_ghz(n: int) -> QBCircuit:
    """Build an n-qubit GHZ circuit."""
    circ = QBCircuit(n)
    circ.h(0)
    for i in range(n - 1):
        circ.cx(i, i + 1)
    circ.measure_all()
    return circ


def _build_qft(n: int) -> QBCircuit:
    """Approximate n-qubit QFT with CX + Rz."""
    circ = QBCircuit(n)
    for i in range(n):
        circ.h(i)
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            circ.cx(j, i)
            circ.rz(i, -angle / 2)
            circ.cx(j, i)
            circ.rz(i, angle / 2)
    circ.measure_all()
    return circ


def _build_rotation_layer(n: int) -> QBCircuit:
    """Build a circuit with alternating rotation and CX layers."""
    circ = QBCircuit(n)
    for _ in range(3):
        for i in range(n):
            circ.rx(i, math.pi / 4)
            circ.rz(i, math.pi / 3)
        for i in range(0, n - 1, 2):
            circ.cx(i, i + 1)
        for i in range(1, n - 1, 2):
            circ.cx(i, i + 1)
    circ.measure_all()
    return circ


# ── qb-compiler benchmark helpers ────────────────────────────────────


def _qb_compile(circ: QBCircuit) -> float:
    """Compile with qb-compiler and return estimated fidelity."""
    compiler = QBCompiler.from_backend("ibm_fez", strategy="fidelity_optimal")
    result = compiler.compile(circ)
    return result.estimated_fidelity


# ── benchmarks ───────────────────────────────────────────────────────


@pytest.mark.benchmark
def test_qb_vs_baseline_ghz_4(benchmark) -> None:
    """Benchmark qb-compiler on a 4-qubit GHZ circuit."""
    circ = _build_ghz(4)
    fidelity = benchmark(_qb_compile, circ)
    assert 0.0 < fidelity <= 1.0


@pytest.mark.benchmark
def test_qb_vs_baseline_ghz_8(benchmark) -> None:
    """Benchmark qb-compiler on an 8-qubit GHZ circuit."""
    circ = _build_ghz(8)
    fidelity = benchmark(_qb_compile, circ)
    assert 0.0 < fidelity <= 1.0


@pytest.mark.benchmark
def test_qb_vs_baseline_qft_4(benchmark) -> None:
    """Benchmark qb-compiler on a 4-qubit QFT circuit."""
    circ = _build_qft(4)
    fidelity = benchmark(_qb_compile, circ)
    assert 0.0 < fidelity <= 1.0


@pytest.mark.benchmark
def test_qb_vs_baseline_rotation_6(benchmark) -> None:
    """Benchmark qb-compiler on a 6-qubit rotation-layer circuit."""
    circ = _build_rotation_layer(6)
    fidelity = benchmark(_qb_compile, circ)
    assert 0.0 < fidelity <= 1.0


@requires_qiskit
@pytest.mark.benchmark
def test_qiskit_default_ghz_4(benchmark) -> None:
    """Benchmark Qiskit Level 3 transpile on a 4-qubit GHZ circuit.

    This serves as the baseline for comparison with qb-compiler.
    """
    from qiskit import transpile
    from qiskit.circuit import QuantumCircuit

    def _qiskit_compile():
        qc = QuantumCircuit(4, 4)
        qc.h(0)
        for i in range(3):
            qc.cx(i, i + 1)
        qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
        transpiled = transpile(
            qc,
            basis_gates=["id", "rz", "sx", "x", "cx"],
            optimization_level=3,
            seed_transpiler=42,
        )
        return transpiled.depth()

    depth = benchmark(_qiskit_compile)
    assert depth > 0


@requires_qiskit
@pytest.mark.benchmark
def test_qiskit_default_qft_4(benchmark) -> None:
    """Benchmark Qiskit Level 3 transpile on a 4-qubit QFT circuit."""
    from qiskit import transpile
    from qiskit.circuit import QuantumCircuit

    def _qiskit_compile():
        qc = QuantumCircuit(4, 4)
        for i in range(4):
            qc.h(i)
            for j in range(i + 1, 4):
                angle = math.pi / (2 ** (j - i))
                qc.cx(j, i)
                qc.rz(-angle / 2, i)
                qc.cx(j, i)
                qc.rz(angle / 2, i)
        qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
        transpiled = transpile(
            qc,
            basis_gates=["id", "rz", "sx", "x", "cx"],
            optimization_level=3,
            seed_transpiler=42,
        )
        return transpiled.depth()

    depth = benchmark(_qiskit_compile)
    assert depth > 0
