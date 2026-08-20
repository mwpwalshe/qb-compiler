# SPDX-License-Identifier: Apache-2.0
"""The public trade space: ranked candidate layouts, scores, and the diversity controls.

Before this existed a caller wanting alternatives had to reach into ``_find_top_k_layouts``, which
is what our own campaign did. The other half of that finding was that the top of a raw ranking is
not diverse: rank 0 and rank 1 came back as the same physical qubits with the logical labels
permuted, which is one option presented as two.
"""

from __future__ import annotations

import pytest

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.calibration.models.coupling_properties import GateProperties
from qb_compiler.calibration.models.qubit_properties import QubitProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.mapping import CalibrationMapper, CalibrationMapperConfig, LayoutCandidate

_N = 12


def _line_backend() -> BackendProperties:
    """A 12 qubit line, good at one end and worse along it, so the ranking is not a tie."""
    qubit_props = [
        QubitProperties(
            qubit_id=i,
            t1_us=300.0 - i * 10.0,
            t2_us=250.0 - i * 8.0,
            readout_error=0.005 + i * 0.002,
        )
        for i in range(_N)
    ]
    gate_props = []
    coupling = []
    for i in range(_N - 1):
        error = 0.002 + i * 0.001
        gate_props.append(
            GateProperties(gate_type="cz", qubits=(i, i + 1), error_rate=error, gate_time_ns=68.0)
        )
        gate_props.append(
            GateProperties(gate_type="cz", qubits=(i + 1, i), error_rate=error, gate_time_ns=68.0)
        )
        coupling.extend([(i, i + 1), (i + 1, i)])
    return BackendProperties(
        backend="test_line",
        provider="test",
        n_qubits=_N,
        basis_gates=("cz", "rz", "sx", "x", "id"),
        coupling_map=coupling,
        qubit_properties=qubit_props,
        gate_properties=gate_props,
        timestamp="2026-03-12T00:00:00",
    )


def _pair_circuit() -> QBCircuit:
    circ = QBCircuit(n_qubits=2, name="bell")
    circ.add_gate(QBGate(name="h", qubits=(0,)))
    circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
    return circ


def _chain_circuit() -> QBCircuit:
    circ = QBCircuit(n_qubits=3, name="chain")
    circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
    circ.add_gate(QBGate(name="cx", qubits=(1, 2)))
    return circ


def _mapper() -> CalibrationMapper:
    return CalibrationMapper(_line_backend())


class TestRankLayouts:
    def test_returns_ranked_candidates(self):
        candidates = _mapper().rank_layouts(_chain_circuit(), top_k=5)
        assert candidates
        assert all(isinstance(c, LayoutCandidate) for c in candidates)
        assert [c.rank for c in candidates] == list(range(len(candidates)))
        scores = [c.score for c in candidates]
        assert scores == sorted(scores), "candidates must come back best first"

    def test_top_candidate_matches_what_transform_applies(self):
        mapper = _mapper()
        circuit = _chain_circuit()
        best = mapper.rank_layouts(circuit, top_k=5)[0]
        applied = mapper.run(circuit, {}).metadata["initial_layout"]
        assert best.score == pytest.approx(mapper.score_layout(applied, circuit))

    def test_top_k_is_respected(self):
        candidates = _mapper().rank_layouts(_chain_circuit(), top_k=3)
        assert len(candidates) <= 3

    def test_candidates_are_distinct(self):
        candidates = _mapper().rank_layouts(_chain_circuit(), top_k=10)
        keys = [tuple(sorted(c.layout.items())) for c in candidates]
        assert len(keys) == len(set(keys)), "the same mapping must not appear at two ranks"

    def test_no_two_qubit_gates_still_yields_a_candidate(self):
        circ = QBCircuit(n_qubits=2, name="idle")
        circ.add_gate(QBGate(name="h", qubits=(0,)))
        candidates = _mapper().rank_layouts(circ)
        assert len(candidates) == 1

    def test_circuit_wider_than_the_device_raises(self):
        circ = QBCircuit(n_qubits=_N + 4, name="too wide")
        circ.add_gate(QBGate(name="cx", qubits=(0, 1)))
        with pytest.raises(ValueError, match="only covers"):
            _mapper().rank_layouts(circ)

    def test_bad_overlap_value_raises(self):
        with pytest.raises(ValueError, match="max_overlap"):
            _mapper().rank_layouts(_pair_circuit(), max_overlap=1.5)


class TestDiversity:
    def test_overlap_filter_removes_the_relabelled_duplicate(self):
        """The finding in one test: the same hardware twice is one option, not two."""
        mapper = _mapper()
        circuit = _pair_circuit()
        raw = mapper.rank_layouts(circuit, top_k=8, diversify=False)
        filtered = mapper.rank_layouts(circuit, top_k=8, diversify=False, max_overlap=0.0)

        raw_sets = [c.physical_qubits for c in raw]
        filtered_sets = [c.physical_qubits for c in filtered]
        assert len(set(filtered_sets)) == len(filtered_sets)
        assert all(
            set(a).isdisjoint(set(b))
            for i, a in enumerate(filtered_sets)
            for b in filtered_sets[i + 1 :]
        ), "max_overlap=0 must return hardware that does not share a qubit"
        assert len(filtered) <= len(raw)
        assert len(raw_sets) >= len(filtered_sets)

    def test_partial_overlap_threshold(self):
        mapper = _mapper()
        candidates = mapper.rank_layouts(_chain_circuit(), top_k=6, max_overlap=0.5)
        for i, first in enumerate(candidates):
            for second in candidates[i + 1 :]:
                assert second.overlap(first) <= 0.5

    def test_overlap_is_a_fraction_of_this_candidates_qubits(self):
        one = LayoutCandidate(rank=0, layout={0: 1, 1: 2}, score=1.0, physical_qubits=(1, 2))
        two = LayoutCandidate(rank=1, layout={0: 2, 1: 3}, score=1.1, physical_qubits=(2, 3))
        assert one.overlap(two) == pytest.approx(0.5)
        assert one.overlap(one) == pytest.approx(1.0)

    def test_config_overlap_applies_to_the_pass_itself(self):
        config = CalibrationMapperConfig(max_overlap=0.0)
        mapper = CalibrationMapper(_line_backend(), config=config)
        # The pass still returns a single layout and still works; the knob only narrows the pool.
        result = mapper.run(_chain_circuit(), {})
        assert len(result.metadata["initial_layout"]) == 3

    def test_as_dict_is_json_shaped(self):
        candidate = _mapper().rank_layouts(_pair_circuit(), top_k=1)[0]
        payload = candidate.as_dict()
        assert set(payload) == {"rank", "layout", "score", "physical_qubits"}
        assert isinstance(payload["physical_qubits"], list)


class TestScoreLayout:
    def test_public_scorer_matches_the_internal_one(self):
        mapper = _mapper()
        circuit = _chain_circuit()
        layout = mapper.run(circuit, {}).metadata["initial_layout"]
        assert mapper.score_layout(layout, circuit) == pytest.approx(
            mapper.run(circuit, {}).metadata["calibration_score"]
        )

    def test_a_worse_layout_scores_higher(self):
        mapper = _mapper()
        circuit = _pair_circuit()
        good = mapper.score_layout({0: 0, 1: 1}, circuit)
        bad = mapper.score_layout({0: _N - 2, 1: _N - 1}, circuit)
        assert bad > good, "lower is better, so the noisy end of the line must score higher"
