#!/usr/bin/env python3
"""Generate standard benchmark circuits as QASM files.

Usage:
    python scripts/generate_benchmark_circuits.py
"""
from __future__ import annotations

import math
from pathlib import Path


def _qasm_header(name: str, n_qubits: int, n_clbits: int = 0) -> str:
    header = f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{n_qubits}];\n'
    if n_clbits:
        header += f"creg c[{n_clbits}];\n"
    return header


def bell_state() -> str:
    return _qasm_header("bell", 2, 2) + "h q[0];\ncx q[0],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\n"


def ghz_state(n: int) -> str:
    lines = [_qasm_header(f"ghz_{n}", n, n), "h q[0];\n"]
    for i in range(n - 1):
        lines.append(f"cx q[{i}],q[{i + 1}];\n")
    for i in range(n):
        lines.append(f"measure q[{i}] -> c[{i}];\n")
    return "".join(lines)


def qft_circuit(n: int) -> str:
    lines = [_qasm_header(f"qft_{n}", n)]
    for i in range(n):
        lines.append(f"h q[{i}];\n")
        for j in range(i + 1, n):
            k = j - i + 1
            angle = math.pi / (2 ** (k - 1))
            lines.append(f"cp({angle:.6f}) q[{j}],q[{i}];\n")
    # Swap to reverse order
    for i in range(n // 2):
        lines.append(f"swap q[{i}],q[{n - 1 - i}];\n")
    return "".join(lines)


def qaoa_maxcut(n: int, p: int = 1) -> str:
    """QAOA circuit for MaxCut on a random graph."""
    lines = [_qasm_header(f"qaoa_maxcut_{n}", n, n)]
    # Initial superposition
    for i in range(n):
        lines.append(f"h q[{i}];\n")
    # p layers
    for _ in range(p):
        gamma = 0.75
        beta = 0.5
        # Problem unitary (ring graph)
        for i in range(n):
            j = (i + 1) % n
            lines.append(f"cx q[{i}],q[{j}];\n")
            lines.append(f"rz({2 * gamma:.4f}) q[{j}];\n")
            lines.append(f"cx q[{i}],q[{j}];\n")
        # Mixer unitary
        for i in range(n):
            lines.append(f"rx({2 * beta:.4f}) q[{i}];\n")
    for i in range(n):
        lines.append(f"measure q[{i}] -> c[{i}];\n")
    return "".join(lines)


def main() -> None:
    out_dir = Path("benchmarks/circuits")
    out_dir.mkdir(parents=True, exist_ok=True)

    circuits = {
        "bell_state.qasm": bell_state(),
        "ghz_8.qasm": ghz_state(8),
        "ghz_16.qasm": ghz_state(16),
        "qft_4.qasm": qft_circuit(4),
        "qft_8.qasm": qft_circuit(8),
        "qaoa_maxcut_4.qasm": qaoa_maxcut(4, p=1),
        "qaoa_maxcut_8.qasm": qaoa_maxcut(8, p=2),
    }

    for name, qasm in circuits.items():
        path = out_dir / name
        path.write_text(qasm)
        print(f"Wrote {path}")

    print(f"\nGenerated {len(circuits)} benchmark circuits in {out_dir}/")


if __name__ == "__main__":
    main()
