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
    EnhancedCompileResult,
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
from qb_compiler.recommender import BackendRecommender, RecommendationReport
from qb_compiler.viability import ViabilityResult, check_viability


def passmanager(backend: object = None, *, optimization_level: int = 2) -> object:
    """Return a Qiskit ``PassManager`` configured for *backend*.

    Convenience factory that builds a Qiskit ``StagedPassManager`` with
    :class:`QBCalibrationPass` injected into the layout stage.  Accepts
    a Qiskit ``Backend``, ``Target``, or qb-compiler backend name string.

    Parameters
    ----------
    backend :
        A Qiskit ``Backend`` instance, a Qiskit ``Target``, or a
        qb-compiler backend name (e.g. ``"ibm_fez"``).
    optimization_level :
        Qiskit optimization level (0-3).  Default 2.

    Returns
    -------
    PassManager
        A Qiskit ``StagedPassManager`` ready to ``.run()`` circuits.

    Examples
    --------
    >>> from qb_compiler import passmanager
    >>> pm = passmanager(backend)
    >>> compiled = pm.run(circuit)
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    from qb_compiler.qiskit_plugin.calibration_pass import QBCalibrationPass

    target = None
    basis_gates = None

    # Resolve input type
    if isinstance(backend, str):
        # qb-compiler backend name → use our config for basis gates
        spec = BACKEND_CONFIGS.get(backend)
        if spec is not None:
            basis_gates = list(spec.basis_gates)
        pm = generate_preset_pass_manager(
            optimization_level=optimization_level,
            basis_gates=basis_gates,
        )
        return pm

    if hasattr(backend, "target"):
        # Qiskit Backend
        target = backend.target
        pm = generate_preset_pass_manager(
            optimization_level=optimization_level,
            target=target,
        )
        cal_pass = QBCalibrationPass(backend=backend)
        pm.layout.append(cal_pass)
        return pm

    if hasattr(backend, "num_qubits") and hasattr(backend, "operation_names"):
        # Qiskit Target
        target = backend
        pm = generate_preset_pass_manager(
            optimization_level=optimization_level,
            target=target,
        )
        cal_pass = QBCalibrationPass(target=target)
        pm.layout.append(cal_pass)
        return pm

    # Fallback — no backend
    return generate_preset_pass_manager(optimization_level=optimization_level)


__all__ = [
    "BACKEND_CONFIGS",
    "BackendNotSupportedError",
    "BackendRecommender",
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
    "EnhancedCompileResult",
    "GateOp",
    "InvalidCircuitError",
    "NoiseModel",
    "PassManager",
    "PassResult",
    "QBCircuit",
    "QBCompiler",
    "QBCompilerError",
    "RecommendationReport",
    "ViabilityResult",
    "__version__",
    "check_viability",
    "passmanager",
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
