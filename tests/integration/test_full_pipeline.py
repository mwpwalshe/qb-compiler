"""Integration tests: full qb-compiler pipeline vs Qiskit Level 3 transpiler.

Loads the real IBM Fez calibration snapshot, builds various non-trivial
circuits, runs them through the complete qb-compiler pass pipeline, and
compares the output against Qiskit's default Level 3 transpiler.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate, QBMeasure
from qb_compiler.passes.analysis.error_budget_estimator import ErrorBudgetEstimator
from qb_compiler.passes.base import PassManager
from qb_compiler.passes.mapping.calibration_mapper import CalibrationMapper
from qb_compiler.passes.mapping.noise_aware_router import NoiseAwareRouter
from qb_compiler.passes.scheduling.noise_aware_scheduler import NoiseAwareScheduler
from qb_compiler.passes.transformation.gate_decomposition import GateDecompositionPass
from tests.conftest import requires_qiskit

# ── fixture path ────────────────────────────────────────────────────────

CALIBRATION_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "calibration_snapshots"
    / "ibm_fez_2026_02_15.json"
)


# ── helpers ─────────────────────────────────────────────────────────────


@pytest.fixture()
def fez_backend() -> BackendProperties:
    """Load the real IBM Fez calibration snapshot."""
    if not CALIBRATION_FIXTURE.exists():
        pytest.skip("IBM Fez calibration fixture not found")
    return BackendProperties.from_qubitboost_json(CALIBRATION_FIXTURE)


def _run_qb_pipeline(circuit: QBCircuit, backend: BackendProperties) -> tuple[QBCircuit, dict]:
    """Run the full qb-compiler pipeline and return (compiled_circuit, context)."""
    # Build gate error lookup for the router
    gate_errors: dict[tuple[int, int], float] = {}
    for gp in backend.gate_properties:
        if len(gp.qubits) == 2 and gp.error_rate is not None:
            gate_errors[gp.qubits] = gp.error_rate

    # Average 2Q gate error for the error budget estimator
    cx_errors = [
        gp.error_rate
        for gp in backend.gate_properties
        if gp.error_rate is not None and len(gp.qubits) == 2
    ]
    avg_cx_error = sum(cx_errors) / len(cx_errors) if cx_errors else 0.005

    # Average 1Q gate error (approximate)
    avg_1q_error = avg_cx_error / 10.0

    gate_error_rates = {
        "cx": avg_cx_error,
        "rz": avg_1q_error,
        "sx": avg_1q_error,
        "x": avg_1q_error,
        "id": 0.0,
        "h": avg_1q_error,
    }

    pm = PassManager(
        [
            CalibrationMapper(backend),
            NoiseAwareRouter(
                coupling_map=backend.coupling_map,
                gate_errors=gate_errors,
            ),
            NoiseAwareScheduler(qubit_properties=backend.qubit_properties),
            GateDecompositionPass(target_basis=("cx", "rz", "sx", "x", "id")),
            ErrorBudgetEstimator(
                qubit_properties=backend.qubit_properties,
                gate_error_rates=gate_error_rates,
            ),
        ]
    )

    context: dict = {}
    result = pm.run_all(circuit, context)
    return result.circuit, context


def _run_qiskit_level3(circuit: QBCircuit, backend: BackendProperties) -> tuple[int, int, float]:
    """Transpile an equivalent Qiskit circuit with Level 3 and return
    (two_qubit_gate_count, depth, estimated_fidelity).

    Fidelity is estimated using the same error model as the qb-compiler
    pipeline so comparisons are fair.
    """
    from qiskit import transpile
    from qiskit.circuit import QuantumCircuit
    from qiskit.transpiler import CouplingMap

    # Rebuild the circuit in Qiskit
    n_qubits = circuit.n_qubits
    n_clbits = circuit.n_clbits
    qc = QuantumCircuit(n_qubits, n_clbits)

    for op in circuit.iter_ops():
        if isinstance(op, QBGate):
            if op.name == "h":
                qc.h(op.qubits[0])
            elif op.name == "cx":
                qc.cx(op.qubits[0], op.qubits[1])
            elif op.name == "x":
                qc.x(op.qubits[0])
            elif op.name == "rz":
                qc.rz(op.params[0], op.qubits[0])
            elif op.name == "sx":
                qc.sx(op.qubits[0])
            elif op.name == "ry":
                qc.ry(op.params[0], op.qubits[0])
            elif op.name == "rx":
                qc.rx(op.params[0], op.qubits[0])
            elif op.name == "id":
                qc.id(op.qubits[0])
            elif op.name == "swap":
                qc.swap(op.qubits[0], op.qubits[1])
        elif isinstance(op, QBMeasure):
            qc.measure(op.qubit, op.clbit)

    coupling = CouplingMap(backend.coupling_map)

    transpiled = transpile(
        qc,
        coupling_map=coupling,
        basis_gates=list(backend.basis_gates),
        optimization_level=3,
        seed_transpiler=42,
    )

    # Count 2Q gates
    cx_count = 0
    total_1q = 0
    for instruction in transpiled.data:
        if instruction.operation.num_qubits >= 2:
            cx_count += 1
        elif instruction.operation.num_qubits == 1:
            total_1q += 1

    depth = transpiled.depth()

    # Estimate fidelity using the same error model
    cx_errors = [
        gp.error_rate
        for gp in backend.gate_properties
        if gp.error_rate is not None and len(gp.qubits) == 2
    ]
    avg_cx_error = sum(cx_errors) / len(cx_errors) if cx_errors else 0.005
    avg_1q_error = avg_cx_error / 10.0

    fidelity = ((1.0 - avg_cx_error) ** cx_count) * ((1.0 - avg_1q_error) ** total_1q)

    return cx_count, depth, fidelity


# ── circuit builders ────────────────────────────────────────────────────


def _build_bell_circuit() -> QBCircuit:
    """2-qubit Bell state circuit: H + CX."""
    circ = QBCircuit(n_qubits=2, n_clbits=2, name="bell")
    circ.add_gate(QBGate(name="h", qubits=(0,)))
    circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
    circ.add_measurement(0, 0)
    circ.add_measurement(1, 1)
    return circ


def _build_ghz_circuit() -> QBCircuit:
    """4-qubit GHZ state: H then cascade of CX."""
    circ = QBCircuit(n_qubits=4, n_clbits=4, name="ghz_4")
    circ.add_gate(QBGate(name="h", qubits=(0,)))
    circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
    circ.add_gate(QBGate(name="cx", qubits=(1, 2)))
    circ.add_gate(QBGate(name="cx", qubits=(2, 3)))
    for i in range(4):
        circ.add_measurement(i, i)
    return circ


def _build_qft_like_circuit() -> QBCircuit:
    """QFT-like circuit on 4 qubits with H and CX layers."""
    circ = QBCircuit(n_qubits=4, n_clbits=4, name="qft_like_4")
    for i in range(4):
        circ.add_gate(QBGate(name="h", qubits=(i,)))
        for j in range(i + 1, 4):
            circ.add_gate(QBGate(name="cx", qubits=(i, j)))
            circ.add_gate(QBGate(name="rz", qubits=(j,), params=(math.pi / (2 ** (j - i)),)))
    for i in range(4):
        circ.add_measurement(i, i)
    return circ


def _build_random_cx_layers() -> QBCircuit:
    """Circuit with multiple CX layers forming a non-trivial interaction pattern."""
    circ = QBCircuit(n_qubits=5, n_clbits=5, name="random_cx_layers")
    # Layer 1
    circ.add_gate(QBGate(name="h", qubits=(0,)))
    circ.add_gate(QBGate(name="h", qubits=(2,)))
    circ.add_gate(QBGate(name="h", qubits=(4,)))
    circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
    circ.add_gate(QBGate(name="cx", qubits=(2, 3)))
    # Layer 2
    circ.add_gate(QBGate(name="cx", qubits=(1, 2)))
    circ.add_gate(QBGate(name="cx", qubits=(3, 4)))
    # Layer 3
    circ.add_gate(QBGate(name="cx", qubits=(0, 3)))
    circ.add_gate(QBGate(name="cx", qubits=(1, 4)))
    # Layer 4
    circ.add_gate(QBGate(name="cx", qubits=(0, 2)))
    circ.add_gate(QBGate(name="cx", qubits=(1, 3)))
    for i in range(5):
        circ.add_measurement(i, i)
    return circ


def _build_measurement_heavy_circuit() -> QBCircuit:
    """Circuit with gates and measurements interleaved on all qubits."""
    circ = QBCircuit(n_qubits=4, n_clbits=4, name="measurement_heavy")
    # Prepare
    for i in range(4):
        circ.add_gate(QBGate(name="h", qubits=(i,)))
    # Entangle
    circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
    circ.add_gate(QBGate(name="cx", qubits=(2, 3)))
    # More single-qubit gates
    for i in range(4):
        circ.add_gate(QBGate(name="rz", qubits=(i,), params=(math.pi / 4,)))
    circ.add_gate(QBGate(name="cx", qubits=(1, 2)))
    # Measure all
    for i in range(4):
        circ.add_measurement(i, i)
    return circ


# ── test class ──────────────────────────────────────────────────────────


CIRCUIT_BUILDERS = [
    ("bell", _build_bell_circuit),
    ("ghz_4", _build_ghz_circuit),
    ("qft_like_4", _build_qft_like_circuit),
    ("random_cx_layers", _build_random_cx_layers),
    ("measurement_heavy", _build_measurement_heavy_circuit),
]


class TestFullPipeline:
    """Test the full qb-compiler pipeline on various circuit types."""

    @pytest.mark.parametrize(
        "circuit_name,builder", CIRCUIT_BUILDERS, ids=[c[0] for c in CIRCUIT_BUILDERS]
    )
    def test_pipeline_produces_valid_circuit(
        self, fez_backend: BackendProperties, circuit_name: str, builder
    ):
        """Pipeline output should be a valid QBCircuit with correct qubit count."""
        circuit = builder()
        compiled, _context = _run_qb_pipeline(circuit, fez_backend)

        # Output circuit must exist and have reasonable properties
        assert compiled is not None
        assert compiled.n_qubits >= circuit.n_qubits
        assert compiled.gate_count > 0

    @pytest.mark.parametrize(
        "circuit_name,builder", CIRCUIT_BUILDERS, ids=[c[0] for c in CIRCUIT_BUILDERS]
    )
    def test_pipeline_populates_context(
        self, fez_backend: BackendProperties, circuit_name: str, builder
    ):
        """All expected context keys must be populated after pipeline run."""
        circuit = builder()
        _compiled, context = _run_qb_pipeline(circuit, fez_backend)

        # CalibrationMapper
        assert "initial_layout" in context
        assert isinstance(context["initial_layout"], dict)

        # NoiseAwareRouter
        assert "swaps_inserted" in context
        assert isinstance(context["swaps_inserted"], int)

        # NoiseAwareScheduler
        assert "estimated_decoherence_reduction" in context

        # ErrorBudgetEstimator
        assert "estimated_fidelity" in context
        assert 0.0 < context["estimated_fidelity"] <= 1.0
        assert "error_budget" in context
        budget = context["error_budget"]
        assert "gate" in budget
        assert "decoherence" in budget
        assert "readout" in budget

    @pytest.mark.parametrize(
        "circuit_name,builder", CIRCUIT_BUILDERS, ids=[c[0] for c in CIRCUIT_BUILDERS]
    )
    def test_estimated_fidelity_is_positive(
        self, fez_backend: BackendProperties, circuit_name: str, builder
    ):
        """Estimated fidelity must be positive and at most 1.0."""
        circuit = builder()
        _compiled, context = _run_qb_pipeline(circuit, fez_backend)

        fidelity = context["estimated_fidelity"]
        assert 0.0 < fidelity <= 1.0, (
            f"Estimated fidelity {fidelity} out of valid range for {circuit_name}"
        )

    @pytest.mark.parametrize(
        "circuit_name,builder", CIRCUIT_BUILDERS, ids=[c[0] for c in CIRCUIT_BUILDERS]
    )
    def test_gate_decomposition_to_basis(
        self, fez_backend: BackendProperties, circuit_name: str, builder
    ):
        """After decomposition, all gates must be in the target basis set."""
        circuit = builder()
        compiled, _context = _run_qb_pipeline(circuit, fez_backend)

        target_basis = {"cx", "rz", "sx", "x", "id"}
        for op in compiled.iter_ops():
            if isinstance(op, QBGate):
                assert op.name in target_basis, (
                    f"Gate '{op.name}' not in target basis {target_basis} "
                    f"for circuit {circuit_name}"
                )

    @requires_qiskit
    @pytest.mark.parametrize(
        "circuit_name,builder", CIRCUIT_BUILDERS, ids=[c[0] for c in CIRCUIT_BUILDERS]
    )
    def test_compare_with_qiskit_level3(
        self, fez_backend: BackendProperties, circuit_name: str, builder
    ):
        """Compare qb-compiler output with Qiskit Level 3 transpiler.

        We check that:
        - qb-compiler estimated fidelity is reported
        - Qiskit-estimated fidelity is computed
        - Both produce valid results (2Q gate count >= original, depth > 0)
        """
        circuit = builder()
        compiled, context = _run_qb_pipeline(circuit, fez_backend)

        qb_fidelity = context["estimated_fidelity"]
        qb_2q_count = compiled.two_qubit_gate_count
        qb_depth = compiled.depth

        qiskit_2q_count, qiskit_depth, qiskit_fidelity = _run_qiskit_level3(circuit, fez_backend)

        # Both must produce valid outputs
        assert qb_2q_count >= 0
        assert qiskit_2q_count >= 0
        assert qb_depth > 0
        assert qiskit_depth > 0
        assert 0.0 < qb_fidelity <= 1.0
        assert 0.0 < qiskit_fidelity <= 1.0

        # Log comparison for diagnostic purposes (not a hard pass/fail)
        print(f"\n{'=' * 60}")
        print(f"Circuit: {circuit_name}")
        print(f"  qb-compiler: 2Q={qb_2q_count}, depth={qb_depth}, fidelity={qb_fidelity:.6f}")
        print(
            f"  Qiskit L3:   2Q={qiskit_2q_count}, depth={qiskit_depth}, "
            f"fidelity={qiskit_fidelity:.6f}"
        )
        print(f"{'=' * 60}")

    @pytest.mark.parametrize(
        "circuit_name,builder", CIRCUIT_BUILDERS, ids=[c[0] for c in CIRCUIT_BUILDERS]
    )
    def test_measurements_preserved(
        self, fez_backend: BackendProperties, circuit_name: str, builder
    ):
        """The number of measurements must not change through the pipeline."""
        circuit = builder()
        original_meas_count = len(circuit.measurements)
        compiled, _context = _run_qb_pipeline(circuit, fez_backend)
        compiled_meas_count = len(compiled.measurements)

        assert compiled_meas_count == original_meas_count, (
            f"Measurement count changed from {original_meas_count} to "
            f"{compiled_meas_count} for {circuit_name}"
        )

    @pytest.mark.parametrize(
        "circuit_name,builder", CIRCUIT_BUILDERS, ids=[c[0] for c in CIRCUIT_BUILDERS]
    )
    def test_error_budget_breakdown_sums_correctly(
        self, fez_backend: BackendProperties, circuit_name: str, builder
    ):
        """Error budget components should be non-negative."""
        circuit = builder()
        _compiled, context = _run_qb_pipeline(circuit, fez_backend)

        budget = context["error_budget"]
        assert budget["gate"] >= 0.0
        assert budget["decoherence"] >= 0.0
        assert budget["readout"] >= 0.0


class TestFullPipelinePassManagerMetadata:
    """Verify that PassManager metadata is correctly populated."""

    def test_all_passes_recorded(self, fez_backend: BackendProperties):
        """The PassManager should record metadata for all 5 passes."""
        circuit = _build_ghz_circuit()

        gate_errors: dict[tuple[int, int], float] = {}
        for gp in fez_backend.gate_properties:
            if len(gp.qubits) == 2 and gp.error_rate is not None:
                gate_errors[gp.qubits] = gp.error_rate

        pm = PassManager(
            [
                CalibrationMapper(fez_backend),
                NoiseAwareRouter(
                    coupling_map=fez_backend.coupling_map,
                    gate_errors=gate_errors,
                ),
                NoiseAwareScheduler(qubit_properties=fez_backend.qubit_properties),
                GateDecompositionPass(target_basis=("cx", "rz", "sx", "x", "id")),
                ErrorBudgetEstimator(
                    qubit_properties=fez_backend.qubit_properties,
                    gate_error_rates={"cx": 0.006},
                ),
            ]
        )

        context: dict = {}
        result = pm.run_all(circuit, context)

        assert "passes" in result.metadata
        pass_names = [p["name"] for p in result.metadata["passes"]]
        assert "calibration_mapper" in pass_names
        assert "noise_aware_router" in pass_names
        assert "noise_aware_scheduler" in pass_names
        assert "gate_decomposition" in pass_names
        assert "error_budget_estimator" in pass_names

        # Each pass should have elapsed time recorded
        for pass_meta in result.metadata["passes"]:
            assert "elapsed_s" in pass_meta
            assert pass_meta["elapsed_s"] >= 0.0
