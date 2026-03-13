#!/usr/bin/env python3
"""Standalone benchmark script for qb-compiler.

Compiles several standard circuits, reports depth / gate count / estimated
fidelity in a Rich-formatted table.

Usage::

    python benchmarks/run_benchmarks.py
"""

from __future__ import annotations

import math
import time

from rich.console import Console
from rich.table import Table

from qb_compiler.compiler import QBCircuit, QBCompiler

console = Console()

# ── circuit builders ──────────────────────────────────────────────────


def bell() -> tuple[str, QBCircuit]:
    return "Bell (2q)", QBCircuit(2).h(0).cx(0, 1).measure_all()


def ghz(n: int = 8) -> tuple[str, QBCircuit]:
    circ = QBCircuit(n)
    circ.h(0)
    for i in range(n - 1):
        circ.cx(i, i + 1)
    circ.measure_all()
    return f"GHZ ({n}q)", circ


def qft(n: int = 4) -> tuple[str, QBCircuit]:
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
    return f"QFT ({n}q)", circ


def qaoa_random(n: int = 6) -> tuple[str, QBCircuit]:
    """Simple QAOA-like ansatz: Rx layer + entangling CX ring."""
    circ = QBCircuit(n)
    for i in range(n):
        circ.rx(i, 0.5)
    for i in range(n):
        circ.cx(i, (i + 1) % n)
    for i in range(n):
        circ.rz(i, 0.3)
    circ.measure_all()
    return f"QAOA-like ({n}q)", circ


def vqe_h2() -> tuple[str, QBCircuit]:
    """Minimal VQE ansatz for H2 (2 qubits, single excitation)."""
    circ = QBCircuit(2)
    circ.rx(0, 0.7)
    circ.rx(1, -0.3)
    circ.cx(0, 1)
    circ.rz(1, 0.5)
    circ.cx(0, 1)
    circ.measure_all()
    return "VQE H2 (2q)", circ


# ── main ──────────────────────────────────────────────────────────────

BACKENDS = ["ibm_fez", "rigetti_ankaa", "ionq_aria"]


def run() -> None:
    circuits = [bell(), ghz(4), ghz(8), ghz(16), qft(4), qft(8), qaoa_random(6), vqe_h2()]

    for backend in BACKENDS:
        table = Table(title=f"qb-compiler benchmarks — {backend}", show_lines=True)
        table.add_column("Circuit", style="cyan", min_width=18)
        table.add_column("Orig Depth", justify="right")
        table.add_column("Compiled Depth", justify="right")
        table.add_column("Depth Reduction", justify="right", style="green")
        table.add_column("Orig Gates", justify="right")
        table.add_column("Compiled Gates", justify="right")
        table.add_column("Est. Fidelity", justify="right", style="bold")
        table.add_column("Time (ms)", justify="right")

        compiler = QBCompiler.from_backend(backend, strategy="fidelity_optimal")

        for name, circ in circuits:
            if circ.n_qubits > 25 and backend == "ionq_aria":
                continue  # skip circuits that exceed backend qubit count

            t0 = time.perf_counter()
            result = compiler.compile(circ)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            table.add_row(
                name,
                str(result.original_depth),
                str(result.compiled_depth),
                f"{result.depth_reduction_pct:.1f}%",
                str(circ.gate_count),
                str(result.compiled_circuit.gate_count),
                f"{result.estimated_fidelity:.4f}",
                f"{elapsed_ms:.2f}",
            )

        console.print(table)
        console.print()


if __name__ == "__main__":
    run()
