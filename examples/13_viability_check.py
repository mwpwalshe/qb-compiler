#!/usr/bin/env python3
"""Example 13: Circuit viability checking.

Know before you run — check whether your circuit is worth
executing on a given backend before spending QPU time.
"""
from __future__ import annotations

from qiskit import QuantumCircuit

from qb_compiler import check_viability

# --- Bell state (simple, should be VIABLE) ---
bell = QuantumCircuit(2, 2, name="Bell")
bell.h(0)
bell.cx(0, 1)
bell.measure([0, 1], [0, 1])

result = check_viability(bell, backend="ibm_fez")
print("=== Bell State ===")
print(f"Status:    {result.status}")
print(f"Fidelity:  {result.estimated_fidelity:.4f}")
print(f"Depth:     {result.depth}")
print(f"2Q gates:  {result.two_qubit_gate_count}")
print(f"Cost:      ${result.cost_estimate_usd:.4f}")
print()
for s in result.suggestions:
    print(f"  -> {s}")
print()

# --- Deep circuit (likely NOT VIABLE) ---
deep = QuantumCircuit(4, 4, name="DeepCircuit")
for _ in range(200):
    for i in range(3):
        deep.cx(i, i + 1)
        deep.rz(0.5, i + 1)
deep.measure(range(4), range(4))

result2 = check_viability(deep, backend="ibm_fez")
print("=== Deep Circuit (200 layers) ===")
print(f"Status:    {result2.status}")
print(f"Fidelity:  {result2.estimated_fidelity:.6f}")
print(f"Depth:     {result2.depth}")
print(f"2Q gates:  {result2.two_qubit_gate_count}")
print()
for s in result2.suggestions:
    print(f"  -> {s}")
