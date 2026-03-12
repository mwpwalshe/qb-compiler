"""Circuit format converters (QASM, Qiskit, etc.)."""

from __future__ import annotations

from qb_compiler.ir.converters.openqasm_converter import from_qasm, to_qasm

__all__ = [
    "from_qasm",
    "to_qasm",
    "from_qiskit",
    "to_qiskit",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy-load Qiskit converter to avoid hard dependency."""
    if name in ("from_qiskit", "to_qiskit"):
        from qb_compiler.ir.converters import qiskit_converter

        return getattr(qiskit_converter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
