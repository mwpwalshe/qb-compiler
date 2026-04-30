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
        # Qubit 4 should have one of the best (lowest) scores by per-qubit metric
        ranked = sorted(scores.items(), key=lambda kv: kv[1])
        best_ids = [qid for qid, _ in ranked[:3]]
        assert 4 in best_ids, (
            f"Qubit 4 (best T1/T2/low errors) should be in top 3, "
            f"but top 3 are {best_ids} with scores {dict(ranked[:5])}"
        )

        # v0.5.1: layout is connectivity-aware via VF2. The chosen physical
        # qubits must form a connected subgraph on the device coupling map AND
        # have low total per-qubit + per-edge score. For a 2q Bell circuit, the
        # winning pair will be the lowest-total-cost connected edge in the
        # coupling map -> not necessarily the single best-individual qubit if
        # that qubit's only neighbours have high error.
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        dag = circuit_to_dag(qc)
        cal_pass = QBCalibrationLayout(cal_data)
        cal_pass.run(dag)

        layout = cal_pass.property_set["layout"]
        assigned = sorted({layout[vq] for vq in qc.qubits})
        # Must be a connected pair on the coupling map
        coupling_edges = {tuple(sorted(e)) for e in cal_data.get("coupling_map", [])}
        assert tuple(assigned) in coupling_edges, (
            f"v0.5.1 must pick a connected pair; got {assigned} which is not in the coupling map"
        )
        # And the pair's total score should be in the top quartile of all coupling edges
        edge_total = sum(scores.get(q, 1.0) for q in assigned)
        all_edge_totals = sorted(sum(scores.get(q, 1.0) for q in edge) for edge in coupling_edges)
        top_quartile_threshold = all_edge_totals[len(all_edge_totals) // 4]
        assert edge_total <= top_quartile_threshold, (
            f"v0.5.1 chose pair {assigned} with total score {edge_total:.4f}; "
            f"top quartile of coupling edges is <={top_quartile_threshold:.4f}"
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
class TestQBTranspileBackendObject:
    """qb_transpile takes a Qiskit backend object, not only a string (v0.5.2).

    Covers the case where the registry's hardcoded basis_gates lag the
    live device (Heron r2 went from ``cx`` to ``ecr``).
    """

    def _stub_backend(self, basis_gates, name="stub_heron", coupling_map=None):
        """Tiny duck-type backend that mimics BackendV1.configuration()."""

        class _Cfg:
            def __init__(self, bg, cm):
                self.basis_gates = list(bg)
                self.coupling_map = cm

        class _Backend:
            def __init__(self, n, bg, cm):
                self.name = n
                self._cfg = _Cfg(bg, cm)

            def configuration(self):
                return self._cfg

        return _Backend(name, basis_gates, coupling_map)

    def test_qb_transpile_with_backend_object_uses_runtime_basis(self):
        """Backend object's runtime basis_gates wins over the registry."""
        from qiskit.circuit import QuantumCircuit

        from qb_compiler.qiskit_plugin import qb_transpile

        backend_obj = self._stub_backend(
            basis_gates=["id", "rz", "sx", "x", "ecr", "reset"],
            name="stub_heron_r2",
        )

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        compiled = qb_transpile(qc, backend=backend_obj, optimization_level=2)

        # Backend exposed ecr only, cx must be translated away.
        ops = compiled.count_ops()
        assert "cx" not in ops, f"cx leaked through despite ecr-only backend. Ops: {dict(ops)}"

    def test_qb_transpile_string_path_unchanged(self):
        """Legacy string path keeps the registry behaviour."""
        from qiskit.circuit import QuantumCircuit

        from qb_compiler.qiskit_plugin import qb_transpile

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        compiled = qb_transpile(
            qc,
            backend="ibm_fez",  # registry still says cx
            calibration_path=CALIBRATION_FIXTURE,
            optimization_level=2,
        )

        # String path keeps cx, that bug stays opt-in until callers move
        # to passing the backend object.
        assert isinstance(compiled, QuantumCircuit)

    def test_qb_transpile_unknown_object_raises(self):
        """Backend object exposing nothing usable raises TypeError."""
        from qiskit.circuit import QuantumCircuit

        from qb_compiler.qiskit_plugin import qb_transpile

        class _Useless:
            pass

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        try:
            qb_transpile(qc, backend=_Useless())
        except TypeError as exc:
            assert "basis_gates" in str(exc)
            return
        raise AssertionError("expected TypeError on unusable backend object")


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
