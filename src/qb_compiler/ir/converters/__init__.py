"""Circuit format converters (QASM, QASM3, Qiskit, Cirq, PennyLane, etc.)."""

from __future__ import annotations

from qb_compiler.ir.converters.openqasm_converter import from_qasm, to_qasm

__all__ = [
    "from_cirq",
    "from_pennylane",
    "from_qasm",
    "from_qasm3",
    "from_qiskit",
    "to_cirq",
    "to_pennylane",
    "to_qasm",
    "to_qasm3",
    "to_qiskit",
]

# Map each lazily-exposed name to the submodule that provides it.
_LAZY: dict[str, str] = {
    "from_qiskit": "qiskit_converter",
    "to_qiskit": "qiskit_converter",
    "from_qasm3": "qasm3_converter",
    "to_qasm3": "qasm3_converter",
    "from_cirq": "cirq_converter",
    "to_cirq": "cirq_converter",
    "from_pennylane": "pennylane_converter",
    "to_pennylane": "pennylane_converter",
}


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy-load optional-SDK converters to avoid hard dependencies."""
    module_name = _LAZY.get(name)
    if module_name is not None:
        import importlib

        module = importlib.import_module(f"qb_compiler.ir.converters.{module_name}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
