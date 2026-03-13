"""Tests for the topology mapper pass."""

from __future__ import annotations

import pytest

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.mapping.topology_mapper import TopologyMapper


class TestTopologyMapper:
    """Tests for TopologyMapper."""

    def test_identity_mapping_no_2q_gates(self) -> None:
        """With no 2Q gates, identity mapping should be used."""
        coupling = [(0, 1), (1, 2), (2, 3)]
        mapper = TopologyMapper(coupling_map=coupling)

        circ = QBCircuit(n_qubits=3)
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        circ.add_gate(QBGate(name="x", qubits=(1,)))

        ctx: dict = {}
        result = mapper.run(circ, ctx)

        assert "initial_layout" in ctx
        assert result.circuit.gate_count == 2

    def test_maps_2q_gates_to_connected_qubits(self) -> None:
        """2Q gates should be mapped to physically connected qubits."""
        coupling = [(0, 1), (1, 2), (2, 3)]
        mapper = TopologyMapper(coupling_map=coupling)

        circ = QBCircuit(n_qubits=2)
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))

        ctx: dict = {}
        mapper.run(circ, ctx)

        layout = ctx["initial_layout"]
        # The mapped qubits should be connected
        p0, p1 = layout[0], layout[1]
        edge = (min(p0, p1), max(p0, p1))
        coupling_set = {(min(a, b), max(a, b)) for a, b in coupling}
        assert edge in coupling_set

    def test_raises_if_too_few_physical_qubits(self) -> None:
        """Should raise ValueError if circuit needs more qubits than available."""
        coupling = [(0, 1)]
        mapper = TopologyMapper(coupling_map=coupling)

        circ = QBCircuit(n_qubits=5)

        ctx: dict = {}
        with pytest.raises(ValueError, match="requires 5 qubits"):
            mapper.run(circ, ctx)

    def test_preserves_gate_count(self) -> None:
        """Mapping should not add or remove gates."""
        coupling = [(0, 1), (1, 2), (2, 3), (3, 4)]
        mapper = TopologyMapper(coupling_map=coupling)

        circ = QBCircuit(n_qubits=3)
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
        circ.add_gate(QBGate(name="rz", qubits=(2,), params=(1.5,)))

        ctx: dict = {}
        result = mapper.run(circ, ctx)

        assert result.circuit.gate_count == 3
