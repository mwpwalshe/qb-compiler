"""qb-compiler — calibration-aware quantum circuit compiler by QubitBoost.

Built by `QubitBoost <https://qubitboost.io>`_.

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
    "PassManager",
    "PassResult",
    "QBCircuit",
    "QBCompiler",
    "QBCompilerError",
    "__version__",
]


def _lazy_ml_imports() -> None:
    """Populate ML classes into module namespace on first access.

    Usage::

        from qb_compiler.ml import is_available, is_gnn_available
        if is_available():
            from qb_compiler.ml.layout_predictor import MLLayoutPredictor
        if is_gnn_available():
            from qb_compiler.ml.gnn_router import GNNLayoutPredictor
    """
