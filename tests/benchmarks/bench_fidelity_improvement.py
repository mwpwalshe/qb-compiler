"""Benchmark estimated fidelity improvement of qb-compiler vs baseline.

Compiles Bell, GHZ, and QFT circuits through the qb-compiler pass pipeline
at optimization level 3 (fidelity_optimal) and reports the estimated fidelity
before and after compilation using the IBM Fez calibration fixture.
"""

from __future__ import annotations

import math

import pytest

from qb_compiler.compiler import QBCircuit, QBCompiler

# ── circuit builders ──────────────────────────────────────────────────


def _build_bell() -> QBCircuit:
    """2-qubit Bell state: H(0) CX(0,1) measure_all."""
    return QBCircuit(2).h(0).cx(0, 1).measure_all()


def _build_ghz(n: int = 8) -> QBCircuit:
    """*n*-qubit GHZ state: H(0) then CX chain."""
    circ = QBCircuit(n)
    circ.h(0)
    for i in range(n - 1):
        circ.cx(i, i + 1)
    circ.measure_all()
    return circ


def _build_qft(n: int = 4) -> QBCircuit:
    """Approximate *n*-qubit QFT using H and controlled-Rz (decomposed to CX+Rz)."""
    circ = QBCircuit(n)
    for i in range(n):
        circ.h(i)
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            # Controlled-Rz decomposed: CX, Rz(-angle/2), CX, Rz(angle/2)
            circ.cx(j, i)
            circ.rz(i, -angle / 2)
            circ.cx(j, i)
            circ.rz(i, angle / 2)
    circ.measure_all()
    return circ


# ── helpers ───────────────────────────────────────────────────────────


def _fidelity_improvement(circ: QBCircuit) -> float:
    """Return the fidelity improvement ratio (compiled / original)."""
    compiler = QBCompiler.from_backend("ibm_fez", strategy="fidelity_optimal")
    original_fidelity = compiler.estimate_fidelity(circ)
    result = compiler.compile(circ)
    compiled_fidelity = result.estimated_fidelity
    if original_fidelity == 0.0:
        return 0.0
    return compiled_fidelity / original_fidelity


# ── benchmarks ────────────────────────────────────────────────────────


@pytest.mark.benchmark
def test_fidelity_bell(benchmark, mock_calibration):
    """Benchmark fidelity improvement for a Bell circuit."""
    circ = _build_bell()
    benchmark(_fidelity_improvement, circ)


@pytest.mark.benchmark
def test_fidelity_ghz(benchmark, mock_calibration):
    """Benchmark fidelity improvement for an 8-qubit GHZ circuit."""
    circ = _build_ghz(8)
    benchmark(_fidelity_improvement, circ)


@pytest.mark.benchmark
def test_fidelity_qft(benchmark, mock_calibration):
    """Benchmark fidelity improvement for a 4-qubit QFT circuit."""
    circ = _build_qft(4)
    benchmark(_fidelity_improvement, circ)
