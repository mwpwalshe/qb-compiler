"""Qiskit integration for the qb-compiler.

Public API
----------
QBCalibrationLayout
    Qiskit ``AnalysisPass`` that sets layout from calibration-aware scoring.
QBCalibrationPass
    Qiskit ``TransformationPass`` that uses live calibration data from a
    ``Backend`` or ``Target`` for layout selection and gate cancellation.
QBPassManager
    Full pass manager wrapping Qiskit's ``StagedPassManager`` with
    calibration-aware layout.
QBTranspilerPlugin
    Entry-point class for Qiskit's ``qiskit.transpiler.stage`` plugin system.
qb_transpile
    Convenience function — transpile with calibration-aware layout in one call.
"""

from __future__ import annotations

from qb_compiler.qiskit_plugin.calibration_pass import QBCalibrationPass
from qb_compiler.qiskit_plugin.pass_manager import QBPassManager
from qb_compiler.qiskit_plugin.transpiler_plugin import (
    QBCalibrationLayout,
    QBTranspilerPlugin,
    qb_transpile,
)

__all__ = [
    "QBCalibrationLayout",
    "QBCalibrationPass",
    "QBPassManager",
    "QBTranspilerPlugin",
    "qb_transpile",
]
