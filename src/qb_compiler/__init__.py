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

# ── eagerly imported (lightweight, always needed) ────────────────────
from qb_compiler.config import CompilerConfig, BackendSpec, BACKEND_CONFIGS
from qb_compiler.exceptions import (
    QBCompilerError,
    CompilationError,
    CalibrationError,
    CalibrationNotFoundError,
    CalibrationStaleError,
    BackendNotSupportedError,
    BudgetExceededError,
    InvalidCircuitError,
)

# ── core types (cheap imports from compiler.py) ──────────────────────
from qb_compiler.compiler import (
    QBCompiler,
    QBCircuit,
    GateOp,
    CompileResult,
    CostEstimate,
    CostEstimator,
    PassResult,
    BasePass,
    PassManager,
    CalibrationProvider,
    NoiseModel,
)

__all__ = [
    # version
    "__version__",
    # compiler
    "QBCompiler",
    "QBCircuit",
    "GateOp",
    "CompileResult",
    "CostEstimate",
    "CompilerConfig",
    "BackendSpec",
    "BACKEND_CONFIGS",
    # passes
    "PassManager",
    "BasePass",
    "PassResult",
    # protocols
    "CalibrationProvider",
    "NoiseModel",
    "CostEstimator",
    # exceptions
    "QBCompilerError",
    "CompilationError",
    "CalibrationError",
    "CalibrationNotFoundError",
    "CalibrationStaleError",
    "BackendNotSupportedError",
    "BudgetExceededError",
    "InvalidCircuitError",
]
