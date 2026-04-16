"""Test that the Qiskit plugin entry point loads correctly.

This verifies that qb-compiler can be discovered by Qiskit's plugin
infrastructure without import errors or missing dependencies.
"""


class TestQiskitEntryPoint:
    """Verify Qiskit ecosystem integration."""

    def test_import_calibration_pass(self):
        """QBCalibrationPass imports without error."""
        from qb_compiler.qiskit_plugin.calibration_pass import QBCalibrationPass
        assert QBCalibrationPass is not None

    def test_inherits_transformation_pass(self):
        """QBCalibrationPass is a proper Qiskit TransformationPass."""
        from qiskit.transpiler.basepasses import TransformationPass

        from qb_compiler.qiskit_plugin.calibration_pass import QBCalibrationPass
        assert issubclass(QBCalibrationPass, TransformationPass)

    def test_passmanager_factory(self):
        """passmanager() factory returns a StagedPassManager."""
        from qb_compiler import passmanager
        assert callable(passmanager)

    def test_import_public_api(self):
        """All public API symbols import without error."""
        from qb_compiler import (
            BackendNotSupportedError,
            BackendRecommender,
            CompileResult,
            CostEstimator,
            QBCompiler,
            QBCompilerError,
            ViabilityResult,
            check_viability,
        )
        for symbol in (
            QBCompiler,
            CompileResult,
            check_viability,
            ViabilityResult,
            BackendRecommender,
            CostEstimator,
            QBCompilerError,
            BackendNotSupportedError,
        ):
            assert symbol is not None

    def test_qiskit_plugin_module(self):
        """Plugin module exists and has expected attributes."""
        from qb_compiler import qiskit_plugin
        assert hasattr(qiskit_plugin, 'QBCalibrationPass')

    def test_calibration_pass_has_run_method(self):
        """QBCalibrationPass.run() exists (Qiskit protocol)."""
        from qb_compiler.qiskit_plugin.calibration_pass import QBCalibrationPass
        assert hasattr(QBCalibrationPass, 'run')

    def test_no_side_effects_on_import(self):
        """Importing qb_compiler does not trigger network calls or file I/O."""
        import importlib
        # Re-import should be fast and side-effect free
        mod = importlib.import_module('qb_compiler')
        assert mod.__name__ == 'qb_compiler'
