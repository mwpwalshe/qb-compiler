"""Integration tests for the qb-compiler Qiskit plugin.

These tests verify that QBCalibrationLayout, qb_transpile, and QBPassManager
work correctly end-to-end with real Qiskit transpilation.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import requires_qiskit

# Fixture path
CALIBRATION_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "calibration_snapshots"
    / "ibm_fez_2026_02_15.json"
)


@requires_qiskit
class TestQBCalibrationLayout:
    """Tests for QBCalibrationLayout as a Qiskit AnalysisPass."""

    def test_qb_calibration_layout_sets_layout(self):
        """Run QBCalibrationLayout on a simple circuit and verify layout is set."""
        from qiskit.circuit import QuantumCircuit
        from qiskit.converters import circuit_to_dag
        from qiskit.transpiler.layout import Layout

        from qb_compiler.qiskit_plugin import QBCalibrationLayout

        # Create a simple 2-qubit Bell circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        dag = circuit_to_dag(qc)
        cal_pass = QBCalibrationLayout(CALIBRATION_FIXTURE)
        cal_pass.run(dag)

        layout = cal_pass.property_set.get("layout")
        assert layout is not None, "Layout should be set by QBCalibrationLayout"
        assert isinstance(layout, Layout)

        # Layout should map both virtual qubits to distinct physical qubits
        physical_qubits = set()
        for vq in qc.qubits:
            pq = layout[vq]
            assert isinstance(pq, int)
            physical_qubits.add(pq)
        assert len(physical_qubits) == 2, "Two virtual qubits must map to two distinct physicals"

    def test_layout_prefers_better_qubits(self):
        """With known calibration data, verify layout picks the best qubits.

        From the fixture, qubit 4 has T1=299.74, T2=222.57, low readout error
        (0.0076 / 0.012) and low gate error (0.0035 on cx[3,4] and 0.0091 on
        cx[4,5]).  It should be among the top-ranked qubits.
        """
        import json

        from qiskit.circuit import QuantumCircuit
        from qiskit.converters import circuit_to_dag

        from qb_compiler.qiskit_plugin import QBCalibrationLayout
        from qb_compiler.qiskit_plugin.transpiler_plugin import _build_qubit_scores

        # Load the fixture and check scores directly
        with open(CALIBRATION_FIXTURE) as f:
            cal_data = json.load(f)

        scores = _build_qubit_scores(cal_data)
        # Qubit 4 should have one of the best (lowest) scores
        ranked = sorted(scores.items(), key=lambda kv: kv[1])
        best_ids = [qid for qid, _ in ranked[:3]]
        assert 4 in best_ids, (
            f"Qubit 4 (best T1/T2/low errors) should be in top 3, "
            f"but top 3 are {best_ids} with scores {dict(ranked[:5])}"
        )

        # Now verify the layout actually uses qubit 4 for a small circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        dag = circuit_to_dag(qc)
        cal_pass = QBCalibrationLayout(cal_data)
        cal_pass.run(dag)

        layout = cal_pass.property_set["layout"]
        assigned = {layout[vq] for vq in qc.qubits}
        assert 4 in assigned, (
            f"Qubit 4 should be assigned for a 2-qubit circuit, "
            f"but got physical qubits {assigned}"
        )


@requires_qiskit
class TestQBTranspile:
    """Tests for the qb_transpile convenience function."""

    def test_qb_transpile_basic(self):
        """Transpile a Bell circuit for ibm_fez with calibration JSON."""
        from qiskit.circuit import QuantumCircuit

        from qb_compiler.qiskit_plugin import qb_transpile

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        compiled = qb_transpile(
            qc,
            backend="ibm_fez",
            calibration_path=CALIBRATION_FIXTURE,
            optimization_level=2,
        )

        assert isinstance(compiled, QuantumCircuit)
        assert compiled.num_qubits >= 2  # may be mapped to larger register

    def test_qb_transpile_without_calibration_falls_back(self):
        """qb_transpile should work even without calibration data."""
        from qiskit.circuit import QuantumCircuit

        from qb_compiler.qiskit_plugin import qb_transpile

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        # No calibration, no backend — should still transpile
        compiled = qb_transpile(qc, optimization_level=1)
        assert isinstance(compiled, QuantumCircuit)

    def test_qb_transpile_reduces_to_native_gates(self):
        """Verify transpiled output uses only IBM native gates for ibm_fez."""
        from qiskit.circuit import QuantumCircuit

        from qb_compiler.qiskit_plugin import qb_transpile

        # Build a circuit with non-native gates (h, ccx, swap, etc.)
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.h(1)
        qc.ccx(0, 1, 2)  # Toffoli — definitely not native
        qc.measure([0, 1, 2], [0, 1, 2])

        compiled = qb_transpile(
            qc,
            backend="ibm_fez",
            calibration_path=CALIBRATION_FIXTURE,
            optimization_level=2,
        )

        # IBM Fez native gates
        ibm_native = {"id", "rz", "sx", "x", "cx", "reset", "measure", "barrier", "delay"}
        ops_in_circuit = compiled.count_ops()
        for gate_name in ops_in_circuit:
            assert gate_name in ibm_native, (
                f"Gate '{gate_name}' is not in IBM native set {ibm_native}. "
                f"Full ops: {dict(ops_in_circuit)}"
            )


@requires_qiskit
class TestQBPassManager:
    """Tests for QBPassManager."""

    def test_pass_manager_from_backend(self):
        """QBPassManager.from_backend should create a working pipeline."""
        from qiskit.circuit import QuantumCircuit

        from qb_compiler.qiskit_plugin import QBPassManager

        pm = QBPassManager.from_backend(
            "ibm_fez",
            calibration_data=CALIBRATION_FIXTURE,
            optimization_level=2,
        )

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        compiled = pm.run(qc)
        assert isinstance(compiled, QuantumCircuit)

    def test_pass_manager_level_0_no_calibration(self):
        """Level 0 should work without calibration (no layout injection)."""
        from qiskit.circuit import QuantumCircuit

        from qb_compiler.qiskit_plugin import QBPassManager

        pm = QBPassManager(
            optimization_level=0,
            basis_gates=["cx", "rz", "sx", "x", "id"],
        )

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        compiled = pm.run(qc)
        assert isinstance(compiled, QuantumCircuit)
