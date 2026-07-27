"""qb-compiler: calibration-aware quantum circuit compiler by QubitBoost.

Built by `QubitBoost <https://qubitboost.io>`_.

Public API
----------
>>> from qb_compiler import QBCompiler, QBCircuit, CompilerConfig
>>> compiler = QBCompiler.from_backend("ibm_fez")
>>> circ = QBCircuit(3).h(0).cx(0, 1).cx(1, 2).measure_all()
>>> result = compiler.compile(circ)
>>> print(result.compiled_depth, result.estimated_fidelity)

Everything in ``__all__`` is importable straight from this package, including the QEC
decoder-input audit (``audit_dem``, ``canonicalize_dem``), layout selection and its receipts
(``CalibrationMapper``, ``selection_receipt``), the multi-vendor calibration registry
(``get_calibration_provider``, ``all_backend_statuses``) and circuit interop (``any_to_qiskit``).

Names resolve on first use rather than at import, so ``import qb_compiler`` does not pay for
numpy, package metadata or any vendor SDK until something actually needs them. That keeps the CLI
responsive and lets an optional dependency stay genuinely optional: a name whose backing module
needs a package you have not installed only raises when you reach for that name.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Re-stated eagerly for type checkers and IDEs, which cannot follow __getattr__. At runtime
    # the same names come from _EXPORTS below, so this block costs nothing.
    from qb_compiler._version import __version__
    from qb_compiler.calibration.registry import (
        all_backend_statuses,
        get_backend_status,
        get_calibration_provider,
    )
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
    from qb_compiler.config import BACKEND_CONFIGS, BackendSpec, CompilerConfig
    from qb_compiler.discovery import (
        DiscoveredBackend,
        check_viability_pub,
        discover_backends,
        rank_discovered,
    )
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
    from qb_compiler.ir.converters.qiskit_converter import (
        any_to_compiler_circuit,
        any_to_qiskit,
    )
    from qb_compiler.observable_gate import (
        ObservableAuditResult,
        audit_dem,
        audit_matrices,
        canonicalize_dem,
        preflight_dem_gate,
    )
    from qb_compiler.passes.mapping import (
        CalibrationMapper,
        CalibrationMapperConfig,
        calibration_fingerprint,
        selection_receipt,
    )
    from qb_compiler.qec_preflight import QECPreflightResult, qec_preflight
    from qb_compiler.receipts import (
        CompilationReceipt,
        RegressionReport,
        make_receipt,
        receipt_history,
        record_receipt,
        regression_check,
    )
    from qb_compiler.recommender import BackendRecommender, RecommendationReport
    from qb_compiler.verify import (
        MirrorResult,
        VerifyResult,
        accuracy_summary,
        build_mirror,
        run_mirror,
        verify_viability,
    )
    from qb_compiler.viability import ViabilityResult, check_viability
    from qb_compiler.windows import BackendValue, calibration_trend, rank_value


# Public name to the module that defines it. Keep in step with __all__: the test suite asserts
# the two agree and that every entry actually resolves.
_EXPORTS: dict[str, str] = {
    "__version__": "qb_compiler._version",
    # core compiler types
    "BasePass": "qb_compiler.compiler",
    "CalibrationProvider": "qb_compiler.compiler",
    "CompileResult": "qb_compiler.compiler",
    "CostEstimate": "qb_compiler.compiler",
    "CostEstimator": "qb_compiler.compiler",
    "EnhancedCompileResult": "qb_compiler.compiler",
    "GateOp": "qb_compiler.compiler",
    "NoiseModel": "qb_compiler.compiler",
    "PassManager": "qb_compiler.compiler",
    "PassResult": "qb_compiler.compiler",
    "QBCircuit": "qb_compiler.compiler",
    "QBCompiler": "qb_compiler.compiler",
    # configuration
    "BACKEND_CONFIGS": "qb_compiler.config",
    "BackendSpec": "qb_compiler.config",
    "CompilerConfig": "qb_compiler.config",
    # discovery and ranking
    "DiscoveredBackend": "qb_compiler.discovery",
    "check_viability_pub": "qb_compiler.discovery",
    "discover_backends": "qb_compiler.discovery",
    "rank_discovered": "qb_compiler.discovery",
    # errors
    "BackendNotSupportedError": "qb_compiler.exceptions",
    "BudgetExceededError": "qb_compiler.exceptions",
    "CalibrationError": "qb_compiler.exceptions",
    "CalibrationNotFoundError": "qb_compiler.exceptions",
    "CalibrationStaleError": "qb_compiler.exceptions",
    "CompilationError": "qb_compiler.exceptions",
    "InvalidCircuitError": "qb_compiler.exceptions",
    "QBCompilerError": "qb_compiler.exceptions",
    # QEC preflight and decoder-input audit
    "QECPreflightResult": "qb_compiler.qec_preflight",
    "qec_preflight": "qb_compiler.qec_preflight",
    "ObservableAuditResult": "qb_compiler.observable_gate",
    "audit_dem": "qb_compiler.observable_gate",
    "audit_matrices": "qb_compiler.observable_gate",
    "canonicalize_dem": "qb_compiler.observable_gate",
    "preflight_dem_gate": "qb_compiler.observable_gate",
    # layout selection and its receipts
    "CalibrationMapper": "qb_compiler.passes.mapping",
    "CalibrationMapperConfig": "qb_compiler.passes.mapping",
    "calibration_fingerprint": "qb_compiler.passes.mapping",
    "selection_receipt": "qb_compiler.passes.mapping",
    # multi-vendor calibration
    "all_backend_statuses": "qb_compiler.calibration.registry",
    "get_backend_status": "qb_compiler.calibration.registry",
    "get_calibration_provider": "qb_compiler.calibration.registry",
    # circuit interop
    "any_to_compiler_circuit": "qb_compiler.ir.converters.qiskit_converter",
    "any_to_qiskit": "qb_compiler.ir.converters.qiskit_converter",
    # receipts and regression
    "CompilationReceipt": "qb_compiler.receipts",
    "RegressionReport": "qb_compiler.receipts",
    "make_receipt": "qb_compiler.receipts",
    "receipt_history": "qb_compiler.receipts",
    "record_receipt": "qb_compiler.receipts",
    "regression_check": "qb_compiler.receipts",
    # recommendation
    "BackendRecommender": "qb_compiler.recommender",
    "RecommendationReport": "qb_compiler.recommender",
    # verification
    "MirrorResult": "qb_compiler.verify",
    "VerifyResult": "qb_compiler.verify",
    "accuracy_summary": "qb_compiler.verify",
    "build_mirror": "qb_compiler.verify",
    "run_mirror": "qb_compiler.verify",
    "verify_viability": "qb_compiler.verify",
    # viability
    "ViabilityResult": "qb_compiler.viability",
    "check_viability": "qb_compiler.viability",
    # cost and drift windows
    "BackendValue": "qb_compiler.windows",
    "calibration_trend": "qb_compiler.windows",
    "rank_value": "qb_compiler.windows",
}


# qec_preflight is both a submodule and the function it exports. Importing the submodule anywhere
# binds the MODULE as an attribute of this package, which shadows __getattr__ and would make
# qb_compiler.qec_preflight a module rather than the callable, depending on import order. Binding
# the function here settles it once. The submodule no longer imports numpy at module scope, so
# this stays cheap.
from qb_compiler.qec_preflight import qec_preflight as qec_preflight  # noqa: E402


def __getattr__(name: str) -> Any:
    """Resolve a public name on first access (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache, so later access is an ordinary global lookup
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


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

    # Imported here, not read from module scope: PEP 562 __getattr__ serves attribute access on
    # the module, not plain global lookups inside a function defined in it.
    from qb_compiler.config import BACKEND_CONFIGS
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

    # Fallback: no backend
    return generate_preset_pass_manager(optimization_level=optimization_level)


__all__ = [
    "BACKEND_CONFIGS",
    "BackendNotSupportedError",
    "BackendRecommender",
    "BackendSpec",
    "BackendValue",
    "BasePass",
    "BudgetExceededError",
    "CalibrationError",
    "CalibrationMapper",
    "CalibrationMapperConfig",
    "CalibrationNotFoundError",
    "CalibrationProvider",
    "CalibrationStaleError",
    "CompilationError",
    "CompilationReceipt",
    "CompileResult",
    "CompilerConfig",
    "CostEstimate",
    "CostEstimator",
    "DiscoveredBackend",
    "EnhancedCompileResult",
    "GateOp",
    "InvalidCircuitError",
    "MirrorResult",
    "NoiseModel",
    "ObservableAuditResult",
    "PassManager",
    "PassResult",
    "QBCircuit",
    "QBCompiler",
    "QBCompilerError",
    "QECPreflightResult",
    "RecommendationReport",
    "RegressionReport",
    "VerifyResult",
    "ViabilityResult",
    "__version__",
    "accuracy_summary",
    "all_backend_statuses",
    "any_to_compiler_circuit",
    "any_to_qiskit",
    "audit_dem",
    "audit_matrices",
    "build_mirror",
    "calibration_fingerprint",
    "calibration_trend",
    "canonicalize_dem",
    "check_viability",
    "check_viability_pub",
    "discover_backends",
    "get_backend_status",
    "get_calibration_provider",
    "make_receipt",
    "passmanager",
    "preflight_dem_gate",
    "qec_preflight",
    "rank_discovered",
    "rank_value",
    "receipt_history",
    "record_receipt",
    "regression_check",
    "run_mirror",
    "selection_receipt",
    "verify_viability",
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
