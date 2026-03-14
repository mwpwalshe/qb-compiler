#!/usr/bin/env python3
"""Example 14: Backend recommendation.

Compare multiple backends and get a recommendation for where
to run your circuit.
"""
from __future__ import annotations

from qiskit import QuantumCircuit

from qb_compiler.recommender import BackendRecommender

# Build a GHZ-5 circuit
qc = QuantumCircuit(5, 5, name="GHZ-5")
qc.h(0)
for i in range(4):
    qc.cx(i, i + 1)
qc.measure(range(5), range(5))

# Analyze across configured backends
recommender = BackendRecommender()
recommender.add_backend("ibm_fez")
recommender.add_backend("ibm_torino")
report = recommender.analyze(qc)

print("=== Backend Recommendation ===")
print(f"Circuit: {report.circuit_name}  Qubits: {report.n_qubits}")
print()

print(f"{'Backend':<15} {'Status':<12} {'Fidelity':>10} {'Cost/4096':>12}")
print("-" * 52)
for a in report.analyses:
    print(
        f"{a.backend:<15} "
        f"{a.status:<12} "
        f"{a.estimated_fidelity:>10.4f} "
        f"${a.cost_per_4096_shots:>11.4f}"
    )
print()

if report.best_fidelity:
    print(f"Best fidelity:       {report.best_fidelity}")
if report.best_value:
    print(f"Best value ($/fid):  {report.best_value}")
print()

if report.warnings:
    print("Warnings:")
    for w in report.warnings:
        print(f"  - {w}")
