"""Fuzz the OpenQASM 2.0 parser.

Targets ``qb_compiler.ir.converters.openqasm_converter.from_qasm`` with:
- Completely random byte strings
- Valid QASM headers with corrupted gate definitions
- Extremely long circuits
- Unicode / null bytes in gate names
"""
from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    from qb_compiler.ir.converters.openqasm_converter import from_qasm


def _build_qasm_with_random_gates(fdp: atheris.FuzzedDataProvider) -> str:
    """Construct a syntactically plausible QASM string with fuzzed gate body."""
    n_qubits = fdp.ConsumeIntInRange(1, 64)
    n_gates = fdp.ConsumeIntInRange(0, 200)
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{n_qubits}];",
        f"creg c[{n_qubits}];",
    ]
    for _ in range(n_gates):
        gate_name = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(1, 30))
        n_args = fdp.ConsumeIntInRange(1, 4)
        qubits = ",".join(
            f"q[{fdp.ConsumeIntInRange(0, n_qubits - 1)}]" for _ in range(n_args)
        )
        if fdp.ConsumeBool():
            param = fdp.ConsumeRegularFloat()
            lines.append(f"{gate_name}({param}) {qubits};")
        else:
            lines.append(f"{gate_name} {qubits};")
    return "\n".join(lines)


def test_one_input(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    choice = fdp.ConsumeIntInRange(0, 3)

    try:
        if choice == 0:
            # Completely random string
            qasm_str = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 4096))
            from_qasm(qasm_str)

        elif choice == 1:
            # Valid header with corrupted gate definitions
            qasm_str = _build_qasm_with_random_gates(fdp)
            from_qasm(qasm_str)

        elif choice == 2:
            # Extremely long circuit
            n_qubits = fdp.ConsumeIntInRange(1, 16)
            lines = [
                "OPENQASM 2.0;",
                'include "qelib1.inc";',
                f"qreg q[{n_qubits}];",
            ]
            for _ in range(fdp.ConsumeIntInRange(500, 2000)):
                q = fdp.ConsumeIntInRange(0, n_qubits - 1)
                lines.append(f"h q[{q}];")
            from_qasm("\n".join(lines))

        else:
            # Null bytes and unicode in gate names
            header = "OPENQASM 2.0;\nqreg q[4];\n"
            gate_name = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 50))
            body = f"{gate_name} q[0];\n"
            from_qasm(header + body)

    except (ValueError, IndexError, KeyError, TypeError, OverflowError):
        pass


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
