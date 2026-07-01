"""Convert between OpenQASM 3 strings and :class:`QBCircuit`.

OpenQASM 3 support is bridged through Qiskit: parsing uses
``qiskit.qasm3.loads`` (which requires the optional ``qiskit_qasm3_import``
package) followed by :func:`from_qiskit`, and emission uses
``qiskit.qasm3.dumps`` applied to the output of :func:`to_qiskit`.

All public functions raise ``ImportError`` with a helpful message if the
required dependency is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qb_compiler.ir.converters.qiskit_converter import (
    _ensure_qiskit,
    from_qiskit,
    to_qiskit,
)

if TYPE_CHECKING:
    from qb_compiler.ir.circuit import QBCircuit


def _ensure_qasm3_import() -> None:
    try:
        import qiskit_qasm3_import  # noqa: F401
    except ImportError:
        raise ImportError(
            "Parsing OpenQASM 3 requires the 'qiskit_qasm3_import' package. "
            "Install it with: pip install 'qb-compiler[qasm3]'"
        ) from None


def from_qasm3(qasm_str: str) -> QBCircuit:
    """Parse an OpenQASM 3 string into a :class:`QBCircuit`.

    Bridges through Qiskit: ``qiskit.qasm3.loads`` builds a
    :class:`~qiskit.circuit.QuantumCircuit`, which is then converted via
    :func:`from_qiskit`.

    Parameters
    ----------
    qasm_str : str
        OpenQASM 3 program text.

    Returns
    -------
    QBCircuit
        Equivalent vendor-neutral circuit.
    """
    _ensure_qiskit()
    _ensure_qasm3_import()
    from qiskit.qasm3 import loads

    qc = loads(qasm_str)
    return from_qiskit(qc)


def to_qasm3(circuit: QBCircuit) -> str:
    """Emit an OpenQASM 3 string from a :class:`QBCircuit`.

    Bridges through Qiskit: :func:`to_qiskit` builds a
    :class:`~qiskit.circuit.QuantumCircuit`, which ``qiskit.qasm3.dumps``
    serialises.  This path works with core Qiskit and does **not** require
    the ``qiskit_qasm3_import`` extra.

    Parameters
    ----------
    circuit : QBCircuit
        Source circuit.

    Returns
    -------
    str
        OpenQASM 3 program text.
    """
    _ensure_qiskit()
    from qiskit.qasm3 import dumps

    qc = to_qiskit(circuit)
    return str(dumps(qc))
