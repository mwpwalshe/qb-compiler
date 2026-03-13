"""Qiskit integration for the qb-compiler.

Public API
----------
QBCalibrationLayout
    Qiskit ``AnalysisPass`` that sets layout from calibration-aware scoring.
QBPassManager
    Full pass manager wrapping Qiskit's ``StagedPassManager`` with
    calibration-aware layout.
QBTranspilerPlugin
    Entry-point class for Qiskit's ``qiskit.transpiler.stage`` plugin system.
qb_transpile
    Convenience function — transpile with calibration-aware layout in one call.
"""

from qb_compiler.qiskit_plugin.pass_manager import QBPassManager
from qb_compiler.qiskit_plugin.transpiler_plugin import (
    QBCalibrationLayout,
    QBTranspilerPlugin,
    qb_transpile,
)

__all__ = [
    "QBCalibrationLayout",
    "QBPassManager",
    "QBTranspilerPlugin",
    "qb_transpile",
]
