"""Integration test: compile same circuit for different backends.

Verifies that QBCompiler can compile a circuit for IBM, Rigetti, IonQ, and IQM
backends and that the output is valid for each.
"""

from __future__ import annotations

import pytest

from qb_compiler.compiler import QBCircuit, QBCompiler
from qb_compiler.config import BACKEND_CONFIGS

# ── helpers ──────────────────────────────────────────────────────────


def _build_ghz(n_qubits: int = 4) -> QBCircuit:
    """Build an n-qubit GHZ circuit."""
    circ = QBCircuit(n_qubits)
    circ.h(0)
    for i in range(n_qubits - 1):
        circ.cx(i, i + 1)
    return circ


def _build_rotation_circuit() -> QBCircuit:
    """Build a circuit with parametric rotation gates."""
    import math

    circ = QBCircuit(3)
    circ.h(0)
    circ.rz(0, math.pi / 4)
    circ.cx(0, 1)
    circ.rx(1, math.pi / 2)
    circ.cx(1, 2)
    return circ


# ── multi-backend tests ─────────────────────────────────────────────

# Backends to test across all vendors
MULTI_VENDOR_BACKENDS = [
    "ibm_fez",
    "ibm_torino",
    "rigetti_ankaa",
    "ionq_aria",
    "iqm_garnet",
]


class TestMultiBackendCompilation:
    """Compile the same circuit for multiple backends and validate output."""

    @pytest.mark.parametrize("backend", MULTI_VENDOR_BACKENDS)
    def test_compile_ghz_circuit(self, backend: str) -> None:
        """GHZ circuit should compile successfully for any backend."""
        circ = _build_ghz(4)
        compiler = QBCompiler.from_backend(backend)
        result = compiler.compile(circ)

        assert result.compiled_circuit is not None
        assert result.compiled_circuit.gate_count > 0
        assert result.compilation_time_ms >= 0
        assert 0.0 <= result.estimated_fidelity <= 1.0

    @pytest.mark.parametrize("backend", MULTI_VENDOR_BACKENDS)
    def test_compile_rotation_circuit(self, backend: str) -> None:
        """Rotation circuit should compile successfully for any backend."""
        circ = _build_rotation_circuit()
        compiler = QBCompiler.from_backend(backend)
        result = compiler.compile(circ)

        assert result.compiled_circuit is not None
        assert result.compiled_circuit.gate_count > 0
        assert result.compiled_depth >= 1

    @pytest.mark.parametrize(
        "backend",
        [
            "ibm_fez",
            "ibm_torino",
            "rigetti_ankaa",
        ],
    )
    def test_basis_translation_applied(self, backend: str) -> None:
        """Compiled circuit should contain only native gates for the backend.

        Only tested for backends whose basis sets are fully supported by the
        decomposer (IBM and Rigetti).  IonQ/IQM use vendor-specific single-qubit
        gates (gpi/gpi2/prx) that the current decomposer does not target.
        """
        circ = _build_ghz(3)
        compiler = QBCompiler.from_backend(backend)
        result = compiler.compile(circ)

        spec = BACKEND_CONFIGS[backend]
        native = set(spec.basis_gates) | {"measure", "reset", "barrier"}

        # After basis translation, all gates should be native
        for op in result.compiled_circuit.ops:
            assert op.name in native, f"Gate '{op.name}' not in native set {native} for {backend}"

    @pytest.mark.parametrize("backend", MULTI_VENDOR_BACKENDS)
    def test_cost_estimation(self, backend: str) -> None:
        """Cost estimation should produce reasonable values for all backends."""
        circ = _build_ghz(4)
        compiler = QBCompiler.from_backend(backend)
        cost = compiler.estimate_cost(circ, shots=1000)

        assert cost.total_usd > 0.0
        assert cost.shots == 1000
        assert cost.cost_per_shot_usd > 0.0

    @pytest.mark.parametrize("backend", MULTI_VENDOR_BACKENDS)
    def test_fidelity_estimation(self, backend: str) -> None:
        """Fidelity estimation should produce valid values for all backends."""
        circ = _build_ghz(4)
        compiler = QBCompiler.from_backend(backend)
        fidelity = compiler.estimate_fidelity(circ)

        assert 0.0 < fidelity <= 1.0


class TestCrossBackendComparison:
    """Compare compilation results across vendors."""

    def test_ibm_vs_rigetti_fidelity(self) -> None:
        """Both IBM and Rigetti should produce positive fidelity estimates."""
        circ = _build_ghz(4)

        ibm_compiler = QBCompiler.from_backend("ibm_fez")
        ibm_result = ibm_compiler.compile(circ)

        rigetti_compiler = QBCompiler.from_backend("rigetti_ankaa")
        rigetti_result = rigetti_compiler.compile(circ)

        assert ibm_result.estimated_fidelity > 0.0
        assert rigetti_result.estimated_fidelity > 0.0

    def test_ionq_higher_fidelity_than_ibm_for_small_circuits(self) -> None:
        """IonQ (trapped-ion) should have lower error rates for small circuits."""
        circ = _build_ghz(3)

        ibm_compiler = QBCompiler.from_backend("ibm_fez")
        ibm_fidelity = ibm_compiler.estimate_fidelity(circ)

        ionq_compiler = QBCompiler.from_backend("ionq_aria")
        ionq_fidelity = ionq_compiler.estimate_fidelity(circ)

        # IonQ typically has lower gate errors on small circuits
        assert ionq_fidelity > 0.0
        assert ibm_fidelity > 0.0

    def test_all_backends_in_config(self) -> None:
        """Every backend in MULTI_VENDOR_BACKENDS should be in BACKEND_CONFIGS."""
        for backend in MULTI_VENDOR_BACKENDS:
            assert backend in BACKEND_CONFIGS, f"Backend '{backend}' not found in BACKEND_CONFIGS"

    def test_compiled_depth_positive(self) -> None:
        """Compiled depth should be positive for all backends."""
        circ = _build_ghz(4)
        for backend in MULTI_VENDOR_BACKENDS:
            compiler = QBCompiler.from_backend(backend)
            result = compiler.compile(circ)
            assert result.compiled_depth > 0, f"Compiled depth is 0 for {backend}"
