"""Integration tests for QBCalibrationPass inside Qiskit StagedPassManager.

These tests verify that the TransformationPass is composable with
standard Qiskit pass pipelines and produces valid output.
"""

from __future__ import annotations

from tests.conftest import requires_qiskit


@requires_qiskit
class TestQBCalibrationPassStandalone:
    """QBCalibrationPass as a standalone TransformationPass."""

    def test_subclasses_transformation_pass(self):
        """QBCalibrationPass must be a Qiskit TransformationPass."""
        from qiskit.transpiler.basepasses import TransformationPass

        from qb_compiler.qiskit_plugin.calibration_pass import QBCalibrationPass

        assert issubclass(QBCalibrationPass, TransformationPass)

    def test_run_returns_dag(self):
        """run() must accept DAGCircuit and return DAGCircuit."""
        from qiskit.circuit import QuantumCircuit
        from qiskit.converters import circuit_to_dag
        from qiskit.dagcircuit import DAGCircuit

        from qb_compiler.qiskit_plugin.calibration_pass import QBCalibrationPass

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        dag = circuit_to_dag(qc)
        cal_pass = QBCalibrationPass()
        result = cal_pass.run(dag)

        assert isinstance(result, DAGCircuit)

    def test_cancels_self_inverse_pairs(self):
        """Adjacent H-H or CX-CX pairs should be cancelled."""
        from qiskit.circuit import QuantumCircuit
        from qiskit.converters import circuit_to_dag, dag_to_circuit

        from qb_compiler.qiskit_plugin.calibration_pass import QBCalibrationPass

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.h(0)  # H-H cancel
        qc.cx(0, 1)

        dag = circuit_to_dag(qc)
        cal_pass = QBCalibrationPass()
        result_dag = cal_pass.run(dag)
        result_qc = dag_to_circuit(result_dag)

        ops = result_qc.count_ops()
        assert ops.get("h", 0) == 0, f"H-H should cancel, got ops: {dict(ops)}"
        assert ops.get("cx", 0) == 1

    def test_accepts_target(self):
        """QBCalibrationPass should accept a Qiskit Target at init."""
        from qiskit.circuit import QuantumCircuit
        from qiskit.converters import circuit_to_dag
        from qiskit.dagcircuit import DAGCircuit

        from qb_compiler.qiskit_plugin.calibration_pass import QBCalibrationPass

        target = _build_simple_target(5)
        cal_pass = QBCalibrationPass(target=target)

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        dag = circuit_to_dag(qc)
        result = cal_pass.run(dag)
        assert isinstance(result, DAGCircuit)

    def test_sets_layout_from_target(self):
        """When target has calibration, pass should set layout in property_set."""
        from qiskit.circuit import QuantumCircuit
        from qiskit.converters import circuit_to_dag

        from qb_compiler.qiskit_plugin.calibration_pass import QBCalibrationPass

        target = _build_simple_target(5)
        cal_pass = QBCalibrationPass(target=target)

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        dag = circuit_to_dag(qc)
        cal_pass.run(dag)

        layout = cal_pass.property_set.get("layout")
        assert layout is not None, "Layout should be set when target has calibration"


@requires_qiskit
class TestQBCalibrationPassInStagedPM:
    """QBCalibrationPass composed inside a Qiskit StagedPassManager."""

    def test_in_pass_manager(self):
        """Pass should work inside a Qiskit PassManager."""
        from qiskit.circuit import QuantumCircuit
        from qiskit.transpiler import PassManager as QiskitPM

        from qb_compiler.qiskit_plugin.calibration_pass import QBCalibrationPass

        pm = QiskitPM([QBCalibrationPass()])
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        result = pm.run(qc)
        assert isinstance(result, QuantumCircuit)

    def test_in_staged_pass_manager(self):
        """Pass should work inside a Qiskit StagedPassManager pipeline."""
        from qiskit.circuit import QuantumCircuit
        from qiskit.transpiler import CouplingMap
        from qiskit.transpiler.preset_passmanagers import (
            generate_preset_pass_manager,
        )

        from qb_compiler.qiskit_plugin.calibration_pass import QBCalibrationPass

        coupling = CouplingMap.from_line(5)
        pm = generate_preset_pass_manager(
            optimization_level=2,
            basis_gates=["cx", "rz", "sx", "x", "id"],
            coupling_map=coupling,
        )
        pm.layout.append(QBCalibrationPass())

        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure([0, 1, 2], [0, 1, 2])

        result = pm.run(qc)
        assert isinstance(result, QuantumCircuit)

    def test_with_target_in_staged_pm(self):
        """Pass with a Target should work in a full StagedPassManager."""
        from qiskit.circuit import QuantumCircuit
        from qiskit.transpiler.preset_passmanagers import (
            generate_preset_pass_manager,
        )

        from qb_compiler.qiskit_plugin.calibration_pass import QBCalibrationPass

        target = _build_simple_target(10)
        pm = generate_preset_pass_manager(
            optimization_level=2,
            target=target,
        )
        pm.layout.append(QBCalibrationPass(target=target))

        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure([0, 1, 2], [0, 1, 2])

        result = pm.run(qc)
        assert isinstance(result, QuantumCircuit)


@requires_qiskit
class TestPassmanagerFactory:
    """Tests for the qb_compiler.passmanager() convenience factory."""

    def test_passmanager_with_string_backend(self):
        """passmanager('ibm_fez') should return a working PassManager."""
        from qiskit.circuit import QuantumCircuit

        from qb_compiler import passmanager

        pm = passmanager("ibm_fez")
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        result = pm.run(qc)
        assert isinstance(result, QuantumCircuit)

    def test_passmanager_without_backend(self):
        """passmanager() with no args should return a basic PassManager."""
        from qiskit.circuit import QuantumCircuit

        from qb_compiler import passmanager

        pm = passmanager()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        result = pm.run(qc)
        assert isinstance(result, QuantumCircuit)

    def test_passmanager_with_target(self):
        """passmanager(target) should accept a Qiskit Target."""
        from qiskit.circuit import QuantumCircuit

        from qb_compiler import passmanager

        target = _build_simple_target(5)
        pm = passmanager(target)

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        result = pm.run(qc)
        assert isinstance(result, QuantumCircuit)


def _build_simple_target(n_qubits: int):
    """Build a simple Qiskit Target for testing."""
    from qiskit.circuit import Measure, Parameter
    from qiskit.circuit.library import CXGate, HGate, RZGate, SXGate, XGate
    from qiskit.transpiler import InstructionProperties, Target

    target = Target(num_qubits=n_qubits)

    # 1Q gates
    theta = Parameter("theta")
    for gate_cls in [SXGate, XGate, HGate]:
        props = {
            (q,): InstructionProperties(error=0.001 * (1 + q * 0.1), duration=25e-9)
            for q in range(n_qubits)
        }
        target.add_instruction(gate_cls(), props)

    target.add_instruction(
        RZGate(theta),
        {(q,): InstructionProperties(duration=0) for q in range(n_qubits)},
    )

    # 2Q gates (linear chain)
    cx_props = {}
    for q in range(n_qubits - 1):
        err = 0.005 * (1 + q * 0.2)
        cx_props[(q, q + 1)] = InstructionProperties(error=err, duration=300e-9)
        cx_props[(q + 1, q)] = InstructionProperties(error=err, duration=300e-9)
    target.add_instruction(CXGate(), cx_props)

    # Measure
    target.add_instruction(
        Measure(),
        {
            (q,): InstructionProperties(error=0.01 * (1 + q * 0.05), duration=1.6e-6)
            for q in range(n_qubits)
        },
    )

    return target
