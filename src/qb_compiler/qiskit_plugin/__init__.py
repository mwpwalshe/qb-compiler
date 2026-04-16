"""Qiskit integration for the qb-compiler.

Public API
----------
QBCalibrationLayout
    Qiskit ``AnalysisPass`` that sets layout from calibration-aware scoring.
QBCalibrationLayoutPlugin
    Entry-point class for Qiskit's ``qiskit.transpiler.layout`` stage-plugin
    system.  Invoke via ``generate_preset_pass_manager(layout_method=
    "qb_calibration")`` with the ``QB_CALIBRATION_PATH`` env var set.
QBCalibrationPass
    Qiskit ``TransformationPass`` that uses live calibration data from a
    ``Backend`` or ``Target`` for layout selection and gate cancellation.
QBPassManager
    Full pass manager wrapping Qiskit's ``StagedPassManager`` with
    calibration-aware layout.
QBTranspilerPlugin
    **Deprecated** legacy alias for :class:`QBCalibrationLayoutPlugin` with
    an older ``get_pass_manager(calibration_data=...)`` method.  Scheduled
    for removal in 0.4.0.
qb_transpile
    Convenience function — transpile with calibration-aware layout in one call.
"""

from __future__ import annotations

from qb_compiler.qiskit_plugin.calibration_pass import QBCalibrationPass
from qb_compiler.qiskit_plugin.pass_manager import QBPassManager
from qb_compiler.qiskit_plugin.transpiler_plugin import (
    QBCalibrationLayout,
    QBCalibrationLayoutPlugin,
    QBTranspilerPlugin,
    qb_transpile,
)

__all__ = [
    "QBCalibrationLayout",
    "QBCalibrationLayoutPlugin",
    "QBCalibrationPass",
    "QBPassManager",
    "QBTranspilerPlugin",
    "qb_transpile",
]
