#!/usr/bin/env python3
"""Example 16: Cost estimation with explicit assumptions.

Estimate execution cost across backends with full transparency
on pricing assumptions.
"""
from __future__ import annotations

from qb_compiler import QBCompiler, QBCircuit
from qb_compiler.config import BACKEND_CONFIGS

# Build a small GHZ circuit using fluent API
circuit = QBCircuit(4)
circuit.h(0)
for i in range(3):
    circuit.cx(i, i + 1)
circuit.measure_all()

shots = 4096

print("=== Cost Estimation Across Backends ===")
print(f"Circuit: GHZ-4  Qubits: {circuit.n_qubits}")
print(f"Shots:   {shots:,}")
print()

print(f"{'Backend':<18} {'Qubits':>6} {'$/shot':>12} {'Total':>10} {'Fidelity':>10}")
print("-" * 60)

for name in sorted(BACKEND_CONFIGS):
    spec = BACKEND_CONFIGS[name]
    if spec.n_qubits < circuit.n_qubits:
        continue
    compiler = QBCompiler.from_backend(name)
    result = compiler.compile(circuit)
    cost = compiler.estimate_cost(result.compiled_circuit, shots=shots)

    print(
        f"{name:<18} "
        f"{spec.n_qubits:>6} "
        f"${cost.cost_per_shot_usd:>11.6f} "
        f"${cost.total_usd:>9.4f} "
        f"{result.estimated_fidelity:>10.4f}"
    )

print()
print("Note: Prices are estimates based on published vendor pricing.")
print("Actual costs may vary. See vendor documentation for current rates.")
