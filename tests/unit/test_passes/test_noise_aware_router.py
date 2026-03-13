"""Unit tests for the noise-aware SWAP router."""

from __future__ import annotations

import math

import pytest

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.mapping.noise_aware_router import NoiseAwareRouter

# ── Helpers ───────────────────────────────────────────────────────────


def _linear_chain(n: int) -> list[tuple[int, int]]:
    """Return coupling map for a linear chain 0-1-2-..-(n-1)."""
    return [(i, i + 1) for i in range(n - 1)]


def _count_cx(circuit: QBCircuit) -> int:
    return sum(1 for g in circuit.gates if g.name == "cx")


# ── Tests ─────────────────────────────────────────────────────────────


class TestAdjacentQubitsNoSwaps:
    """When qubits are already adjacent, no SWAPs should be inserted."""

    def test_cx_on_adjacent_pair(self):
        coupling = [(0, 1), (1, 2)]
        router = NoiseAwareRouter(coupling_map=coupling)

        circ = QBCircuit(n_qubits=3)
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))

        ctx: dict = {}
        result = router.run(circ, ctx)

        assert ctx["swaps_inserted"] == 0
        assert result.modified is False
        # Only the original CX should be present.
        assert result.circuit.gate_count == 1
        assert result.circuit.gates[0].name == "cx"
        assert result.circuit.gates[0].qubits == (0, 1)

    def test_multiple_adjacent_gates(self):
        coupling = [(0, 1), (1, 2), (2, 3)]
        router = NoiseAwareRouter(coupling_map=coupling)

        circ = QBCircuit(n_qubits=4)
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
        circ.add_gate(QBGate(name="cx", qubits=(2, 3)))

        ctx: dict = {}
        result = router.run(circ, ctx)
        assert ctx["swaps_inserted"] == 0
        assert result.circuit.gate_count == 2


class TestNonAdjacentQubitsGetSwaps:
    """Non-adjacent qubits must trigger SWAP insertion."""

    def test_linear_chain_distance_2(self):
        # 0 - 1 - 2:  CX(0,2) needs 1 SWAP.
        coupling = _linear_chain(3)
        router = NoiseAwareRouter(coupling_map=coupling)

        circ = QBCircuit(n_qubits=3)
        circ.add_gate(QBGate(name="cx", qubits=(0, 2)))

        ctx: dict = {}
        result = router.run(circ, ctx)

        assert ctx["swaps_inserted"] == 1
        assert result.modified is True
        # 1 SWAP (3 CX) + 1 original CX = 4 CX total.
        assert _count_cx(result.circuit) == 4

    def test_linear_chain_distance_3(self):
        # 0 - 1 - 2 - 3:  CX(0,3) needs 2 SWAPs.
        coupling = _linear_chain(4)
        router = NoiseAwareRouter(coupling_map=coupling)

        circ = QBCircuit(n_qubits=4)
        circ.add_gate(QBGate(name="cx", qubits=(0, 3)))

        ctx: dict = {}
        result = router.run(circ, ctx)

        assert ctx["swaps_inserted"] == 2
        # 2 SWAPs (6 CX) + 1 CX = 7 CX total.
        assert _count_cx(result.circuit) == 7


class TestPrefersLowErrorPath:
    """The router should choose the lower-error path, even if it is longer.

    Topology:
        0 ---(high error)--- 1
        |                    |
        2 ---(low error)---- 3

    Direct path 0->1: 1 hop but high error.
    Longer path 0->2->3->1: 3 hops but much lower total error.

    For a CX(0,1) we need SWAPs.  If 0-1 is adjacent but has high error,
    we test with 0 and 4 non-adjacent, where two routes exist:
      short route: 0->1->4  (2 hops, 1 SWAP, high error on 0-1)
      long route:  0->2->3->4 (3 hops, 2 SWAPs, low error per edge)
    """

    def test_low_error_path_preferred(self):
        # Diamond + tail topology:
        #   0 --- 1 --- 4
        #   |           |
        #   2 --- 3 ----+
        #
        # Edges: (0,1), (1,4), (0,2), (2,3), (3,4)
        coupling = [(0, 1), (1, 4), (0, 2), (2, 3), (3, 4)]

        # High error on 0-1 and 1-4 edges.
        # Low error on 0-2, 2-3, 3-4 edges.
        gate_errors = {
            (0, 1): 0.20,
            (1, 4): 0.20,
            (0, 2): 0.001,
            (2, 3): 0.001,
            (3, 4): 0.001,
        }

        router = NoiseAwareRouter(coupling_map=coupling, gate_errors=gate_errors)

        circ = QBCircuit(n_qubits=5)
        circ.add_gate(QBGate(name="cx", qubits=(0, 4)))

        ctx: dict = {}
        router.run(circ, ctx)

        # The short path 0->1->4 has weight = -log(0.80) + -log(0.80) ~ 0.446
        # The long path 0->2->3->4 has weight = 3*(-log(0.999)) ~ 0.003
        # Router should pick the long path: 2 SWAPs.
        assert ctx["swaps_inserted"] == 2

        # Verify fidelity cost is low (the low-error path).
        # Each SWAP contributes 3 * edge_weight.  For error=0.001:
        #   weight = -log(0.999) ≈ 0.001
        #   per SWAP cost ≈ 0.003
        #   total for 2 SWAPs ≈ 0.006
        assert ctx["routing_fidelity_cost"] < 0.01


class TestQubitPermutationTracking:
    """Verify that the qubit permutation is correctly maintained across
    multiple routed gates."""

    def test_permutation_after_swap(self):
        # Linear chain: 0 - 1 - 2
        # Gate 1: CX(0, 2) — requires SWAP(0,1), then CX(1,2)
        # After SWAP: logical 0 is at physical 1, logical 1 is at physical 0.
        # Gate 2: CX(0, 1) — logical 0 at phys 1, logical 1 at phys 0 => CX(1,0), adjacent.
        coupling = _linear_chain(3)
        router = NoiseAwareRouter(coupling_map=coupling)

        circ = QBCircuit(n_qubits=3)
        circ.add_gate(QBGate(name="cx", qubits=(0, 2)))
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))

        ctx: dict = {}
        result = router.run(circ, ctx)

        # First gate: 1 SWAP + 1 CX = 4 CX.
        # Second gate: already adjacent after permutation = 1 CX.
        # Total: 5 CX.
        assert ctx["swaps_inserted"] == 1
        assert _count_cx(result.circuit) == 5

    def test_single_qubit_gates_follow_permutation(self):
        # Linear chain: 0 - 1 - 2
        # CX(0,2) causes SWAP(0,1) → logical 0 now at physical 1.
        # H(0) should act on physical 1.
        coupling = _linear_chain(3)
        router = NoiseAwareRouter(coupling_map=coupling)

        circ = QBCircuit(n_qubits=3)
        circ.add_gate(QBGate(name="cx", qubits=(0, 2)))
        circ.add_gate(QBGate(name="h", qubits=(0,)))

        ctx: dict = {}
        result = router.run(circ, ctx)

        # Find the H gate in the output.
        h_gates = [g for g in result.circuit.gates if g.name == "h"]
        assert len(h_gates) == 1
        # Logical qubit 0 was swapped to physical qubit 1.
        assert h_gates[0].qubits == (1,)


class TestPropertySet:
    """Verify that the context/property_set is populated correctly."""

    def test_zero_swaps(self):
        coupling = [(0, 1)]
        router = NoiseAwareRouter(coupling_map=coupling)

        circ = QBCircuit(n_qubits=2)
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))

        ctx: dict = {}
        router.run(circ, ctx)

        assert ctx["swaps_inserted"] == 0
        assert ctx["routing_fidelity_cost"] == 0.0

    def test_nonzero_fidelity_cost(self):
        coupling = _linear_chain(3)
        gate_errors = {(0, 1): 0.05, (1, 2): 0.05}
        router = NoiseAwareRouter(
            coupling_map=coupling, gate_errors=gate_errors
        )

        circ = QBCircuit(n_qubits=3)
        circ.add_gate(QBGate(name="cx", qubits=(0, 2)))

        ctx: dict = {}
        router.run(circ, ctx)

        assert ctx["swaps_inserted"] == 1
        # SWAP on edge (0,1) with error 0.05: cost = 3 * -log(0.95) ≈ 0.1539
        expected = 3 * (-math.log(1.0 - 0.05))
        assert abs(ctx["routing_fidelity_cost"] - expected) < 1e-6

    def test_metadata_matches_context(self):
        coupling = _linear_chain(3)
        router = NoiseAwareRouter(coupling_map=coupling)

        circ = QBCircuit(n_qubits=3)
        circ.add_gate(QBGate(name="cx", qubits=(0, 2)))

        ctx: dict = {}
        result = router.run(circ, ctx)

        assert result.metadata["swaps_inserted"] == ctx["swaps_inserted"]


class TestLinearChainTopology:
    """End-to-end routing on a 5-qubit linear chain."""

    def test_route_cx_across_chain(self):
        coupling = _linear_chain(5)
        router = NoiseAwareRouter(coupling_map=coupling)

        circ = QBCircuit(n_qubits=5)
        circ.add_gate(QBGate(name="cx", qubits=(0, 4)))

        ctx: dict = {}
        result = router.run(circ, ctx)

        # Distance 4 → 3 SWAPs needed.
        assert ctx["swaps_inserted"] == 3
        # 3 SWAPs * 3 CX each + 1 original CX = 10 CX.
        assert _count_cx(result.circuit) == 10

    def test_mixed_gates_on_chain(self):
        coupling = _linear_chain(5)
        router = NoiseAwareRouter(coupling_map=coupling)

        circ = QBCircuit(n_qubits=5)
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        circ.add_gate(QBGate(name="cx", qubits=(0, 4)))
        circ.add_gate(QBGate(name="rz", qubits=(2,), params=(1.57,)))

        ctx: dict = {}
        result = router.run(circ, ctx)

        assert ctx["swaps_inserted"] == 3
        # H + 9 SWAP-CX + 1 CX + RZ = 12 gates total.
        assert result.circuit.gate_count == 12


class TestMeasurementRouting:
    """Measurements should follow the qubit permutation."""

    def test_measurement_after_swap(self):
        coupling = _linear_chain(3)
        router = NoiseAwareRouter(coupling_map=coupling)

        circ = QBCircuit(n_qubits=3, n_clbits=1)
        circ.add_gate(QBGate(name="cx", qubits=(0, 2)))
        circ.add_measurement(qubit=0, clbit=0)

        ctx: dict = {}
        result = router.run(circ, ctx)

        # After SWAP(0,1), logical 0 is at physical 1.
        measurements = result.circuit.measurements
        assert len(measurements) == 1
        assert measurements[0].qubit == 1  # physical qubit
        assert measurements[0].clbit == 0


class TestEdgeCases:
    """Edge cases: single-qubit-only circuits, empty circuits, etc."""

    def test_single_qubit_gates_only(self):
        coupling = _linear_chain(3)
        router = NoiseAwareRouter(coupling_map=coupling)

        circ = QBCircuit(n_qubits=3)
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        circ.add_gate(QBGate(name="x", qubits=(2,)))

        ctx: dict = {}
        result = router.run(circ, ctx)

        assert ctx["swaps_inserted"] == 0
        assert result.modified is False
        assert result.circuit.gate_count == 2

    def test_empty_circuit(self):
        coupling = _linear_chain(3)
        router = NoiseAwareRouter(coupling_map=coupling)

        circ = QBCircuit(n_qubits=3)
        ctx: dict = {}
        result = router.run(circ, ctx)

        assert ctx["swaps_inserted"] == 0
        assert result.circuit.gate_count == 0

    def test_no_path_raises(self):
        # Disconnected graph: 0-1 and 2-3.
        coupling = [(0, 1), (2, 3)]
        router = NoiseAwareRouter(coupling_map=coupling)

        circ = QBCircuit(n_qubits=4)
        circ.add_gate(QBGate(name="cx", qubits=(0, 3)))

        ctx: dict = {}
        with pytest.raises(ValueError, match="No path"):
            router.run(circ, ctx)
