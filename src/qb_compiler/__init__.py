"""qb-compiler — A quantum circuit compiler with calibration-aware optimisation.

Public API
----------
>>> from qb_compiler import QBCompiler, QBCircuit, CompilerConfig
>>> compiler = QBCompiler.from_backend("ibm_fez")
>>> circ = QBCircuit(3).h(0).cx(0, 1).cx(1, 2).measure_all()
>>> result = compiler.compile(circ)
>>> print(result.compiled_depth, result.estimated_fidelity)
"""

from __future__ import annotations

from qb_compiler._version import __version__

# ── core types (cheap imports from compiler.py) ──────────────────────
from qb_compiler.compiler import (
    BasePass,
    CalibrationProvider,
    CompileResult,
    CostEstimate,
    CostEstimator,
    GateOp,
    NoiseModel,
    PassManager,
    PassResult,
    QBCircuit,
    QBCompiler,
)

# ── eagerly imported (lightweight, always needed) ────────────────────
from qb_compiler.config import BACKEND_CONFIGS, BackendSpec, CompilerConfig
from qb_compiler.exceptions import (
    BackendNotSupportedError,
    BudgetExceededError,
    CalibrationError,
    CalibrationNotFoundError,
    CalibrationStaleError,
    CompilationError,
    InvalidCircuitError,
    QBCompilerError,
)

__all__ = [
    "BACKEND_CONFIGS",
    "BackendNotSupportedError",
    "BackendSpec",
    "BasePass",
    "BudgetExceededError",
    "CalibrationError",
    "CalibrationNotFoundError",
    # protocols
    "CalibrationProvider",
    "CalibrationStaleError",
    "CompilationError",
    "CompileResult",
    "CompilerConfig",
    "CostEstimate",
    "CostEstimator",
    "GateOp",
    "InvalidCircuitError",
    "NoiseModel",
    # passes
    "PassManager",
    "PassResult",
    "QBCircuit",
    # compiler
    "QBCompiler",
    # exceptions
    "QBCompilerError",
    # version
    "__version__",
]
