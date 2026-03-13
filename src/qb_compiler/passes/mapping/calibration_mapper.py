"""Calibration-aware qubit mapping pass.

Maps logical qubits to physical qubits using live calibration data.
Unlike topology-only approaches (e.g. Qiskit SabreLayout), this pass
scores candidate mappings by combining:

- **Gate error rates** — prefer lower-error 2Q links
- **Qubit coherence** — prefer higher T1/T2 qubits
- **Readout error** — prefer lower-error measurement qubits for output qubits
- **T1 asymmetry** — penalise qubits where |1⟩ decays disproportionately
- **Temporal correlation** — penalise qubit pairs with correlated error drift
- **Weighted scoring** combining all factors

The pass uses ``rustworkx.vf2_mapping`` for subgraph isomorphism search
to enumerate candidate layouts, scores each one, and picks the best.

T1 Asymmetry
~~~~~~~~~~~~

On superconducting qubits, the probability of reading ``0`` when the qubit
is in ``|1⟩`` (relaxation, P(0|1)) can be 10-25x higher than reading ``1``
when the qubit is in ``|0⟩`` (thermal excitation, P(1|0)).  This asymmetry
means circuits that hold qubits in ``|1⟩`` — after X gates, as CNOT
targets, or in long-lived entangled states — lose fidelity faster on
high-asymmetry qubits.

Standard transpilers use the *symmetrised* readout error and miss this
effect entirely.  CalibrationMapper uses the raw asymmetric readout data
(``readout_error_0to1`` vs ``readout_error_1to0``) to penalise
high-asymmetry qubits.

Temporal Correlation
~~~~~~~~~~~~~~~~~~~~

When calibration data from multiple time points is available, the mapper
can detect qubit pairs whose error rates co-vary.  Correlated errors
violate the independent-error assumption used by quantum error correction
codes, reducing effective code distance.  Penalising correlated edges
during layout selection reduces exposure to this failure mode.

For real-time correlation monitoring, see QubitBoost SafetyGate
(requires ``qubitboost-sdk``).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import rustworkx as rx

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate, QBMeasure
from qb_compiler.passes.base import PassResult, TransformationPass

if TYPE_CHECKING:
    from qb_compiler.calibration.models.coupling_properties import GateProperties
    from qb_compiler.calibration.models.qubit_properties import QubitProperties
    from qb_compiler.calibration.provider import CalibrationProvider
    from qb_compiler.ml.layout_predictor import MLLayoutPredictor
    from qb_compiler.passes.mapping.temporal_correlation import (
        TemporalCorrelationAnalyzer,
    )

logger = logging.getLogger(__name__)

# ── defaults for missing calibration data ────────────────────────────

_DEFAULT_GATE_ERROR = 0.02
_DEFAULT_READOUT_ERROR = 0.015
_DEFAULT_T1_US = 100.0
_DEFAULT_T2_US = 80.0

# ── scoring weight defaults ──────────────────────────────────────────

_DEFAULT_GATE_ERROR_WEIGHT = 10.0
_DEFAULT_QUBIT_COHERENCE_WEIGHT = 0.3
_DEFAULT_READOUT_WEIGHT = 5.0
_DEFAULT_T1_ASYMMETRY_WEIGHT = 3.0
_DEFAULT_CORRELATION_WEIGHT = 2.0


@dataclass
class CalibrationMapperConfig:
    """Tunable parameters for :class:`CalibrationMapper`.

    Attributes
    ----------
    gate_error_weight:
        Weight of 2Q gate error contribution to the layout score.
    coherence_weight:
        Weight of qubit coherence (1/T1 + 1/T2) contribution.
    readout_weight:
        Weight of readout error contribution.
    t1_asymmetry_weight:
        Weight of T1 asymmetry penalty.  Penalises qubits where
        P(0|1) >> P(1|0), indicating fast |1⟩ state decay.
        Set to 0 to disable.
    correlation_weight:
        Weight of temporal error correlation penalty.  Penalises
        qubit pairs whose errors co-vary across calibration snapshots.
        Requires a :class:`TemporalCorrelationAnalyzer`.  Set to 0 to disable.
    max_candidates:
        Maximum number of VF2 candidate mappings to evaluate.
        Higher values give better solutions but take longer.
    vf2_call_limit:
        Upper bound on the number of VF2 internal states explored.
        ``None`` means no limit.
    """

    gate_error_weight: float = _DEFAULT_GATE_ERROR_WEIGHT
    coherence_weight: float = _DEFAULT_QUBIT_COHERENCE_WEIGHT
    readout_weight: float = _DEFAULT_READOUT_WEIGHT
    t1_asymmetry_weight: float = _DEFAULT_T1_ASYMMETRY_WEIGHT
    correlation_weight: float = _DEFAULT_CORRELATION_WEIGHT
    max_candidates: int = 10_000
    vf2_call_limit: int | None = 100_000


class CalibrationMapper(TransformationPass):
    """Map logical qubits to physical qubits using calibration data.

    This is a :class:`TransformationPass`: it rewrites the qubit indices
    in the circuit and records the chosen layout in ``context``.

    Parameters
    ----------
    calibration:
        Source of per-qubit and per-gate calibration data.  Accepts either
        a :class:`CalibrationProvider` or a :class:`BackendProperties`.
    config:
        Optional scoring and search tuning parameters.
    correlation_analyzer:
        Optional temporal correlation analyzer.  When provided, the mapper
        penalises layouts that place interacting qubits on physically
        correlated edges.
    """

    def __init__(
        self,
        calibration: CalibrationProvider | BackendProperties,
        config: CalibrationMapperConfig | None = None,
        correlation_analyzer: TemporalCorrelationAnalyzer | None = None,
        layout_predictor: MLLayoutPredictor | None = None,
    ) -> None:
        self._config = config or CalibrationMapperConfig()
        self._correlation = correlation_analyzer
        self._layout_predictor = layout_predictor

        # Normalise input into uniform lookup structures
        if isinstance(calibration, BackendProperties):
            self._backend_props = calibration
            self._qubit_map: dict[int, QubitProperties] = {
                qp.qubit_id: qp for qp in calibration.qubit_properties
            }
            self._gate_map: dict[tuple[str, tuple[int, ...]], GateProperties] = {
                (gp.gate_type, gp.qubits): gp for gp in calibration.gate_properties
            }
            self._coupling_map = calibration.coupling_map
            self._n_physical = calibration.n_qubits
        else:
            # CalibrationProvider
            all_qubits = calibration.get_all_qubit_properties()
            all_gates = calibration.get_all_gate_properties()
            self._qubit_map = {qp.qubit_id: qp for qp in all_qubits}
            self._gate_map = {
                (gp.gate_type, gp.qubits): gp for gp in all_gates
            }
            # Derive coupling map from 2Q gates
            coupling_set: set[tuple[int, int]] = set()
            for gp in all_gates:
                if len(gp.qubits) == 2:
                    coupling_set.add(gp.qubits)
            self._coupling_map = sorted(coupling_set)
            self._n_physical = max(self._qubit_map.keys()) + 1 if self._qubit_map else 0
            self._backend_props = None  # type: ignore[assignment]

    # ── BasePass interface ───────────────────────────────────────────

    @property
    def name(self) -> str:
        return "calibration_mapper"

    def transform(self, circuit: QBCircuit, context: dict) -> PassResult:
        """Find the best calibration-aware layout and apply it."""

        n_logical = circuit.n_qubits

        # Edge case: circuit needs more qubits than available
        if n_logical > self._n_physical:
            raise ValueError(
                f"Circuit requires {n_logical} qubits but calibration data "
                f"only covers {self._n_physical} physical qubits"
            )

        # Step 1: extract interaction graph
        interactions = self._extract_interactions(circuit)

        # Step 2: find the best mapping
        if not interactions:
            # No 2Q gates — pick the best individual qubits
            layout = self._best_individual_qubits(circuit)
        elif n_logical <= 2 and len(interactions) == 1:
            # Small circuit fast path: skip VF2, directly pick the
            # best-calibrated edge.  VF2 overhead on trivial circuits
            # is wasteful and the greedy edge-ranked approach is optimal
            # when there's only one interacting pair.
            layout = self._best_edge_direct(circuit, interactions)
        else:
            layout = self._find_best_layout(circuit, interactions)

        # Step 3: compute final score for the chosen layout
        score = self._score_layout(layout, interactions, circuit)

        # Step 4: apply the mapping to produce a new circuit
        mapped_circuit = self._apply_layout(circuit, layout)

        # Build detailed scoring breakdown for diagnostics
        breakdown = self._score_breakdown(layout, interactions, circuit)

        # Record results in context
        context["initial_layout"] = dict(layout)
        context["calibration_score"] = score
        context["score_breakdown"] = breakdown

        logger.info(
            "CalibrationMapper: mapped %d logical -> %d physical qubits, "
            "score=%.6f (gate=%.4f coh=%.4f ro=%.4f asym=%.4f corr=%.4f)",
            n_logical,
            self._n_physical,
            score,
            breakdown.get("gate_error", 0),
            breakdown.get("coherence", 0),
            breakdown.get("readout", 0),
            breakdown.get("t1_asymmetry", 0),
            breakdown.get("correlation", 0),
        )

        return PassResult(
            circuit=mapped_circuit,
            metadata={
                "initial_layout": dict(layout),
                "calibration_score": score,
                "score_breakdown": breakdown,
            },
            modified=True,
        )

    # ── interaction graph extraction ─────────────────────────────────

    @staticmethod
    def _extract_interactions(circuit: QBCircuit) -> dict[tuple[int, int], int]:
        """Build a weighted interaction graph from the circuit.

        Returns a dict mapping ``(q_a, q_b)`` (with ``q_a < q_b``) to the
        number of 2Q gates between them.
        """
        interactions: dict[tuple[int, int], int] = defaultdict(int)
        for op in circuit.iter_ops():
            if isinstance(op, QBGate) and op.num_qubits >= 2:
                # Normalise edge direction for counting
                q_sorted = sorted(op.qubits[:2])
                q_pair: tuple[int, int] = (q_sorted[0], q_sorted[1])
                interactions[q_pair] += 1
        return dict(interactions)

    # ── qubit scoring helpers ────────────────────────────────────────

    def _qubit_score(self, physical_qubit: int) -> float:
        """Lower is better.  Combines coherence, readout error, and T1 asymmetry."""
        qp = self._qubit_map.get(physical_qubit)
        t1 = qp.t1_us if (qp and qp.t1_us) else _DEFAULT_T1_US
        t2 = qp.t2_us if (qp and qp.t2_us) else _DEFAULT_T2_US
        readout = (
            qp.readout_error if (qp and qp.readout_error is not None) else _DEFAULT_READOUT_ERROR
        )

        # Coherence contribution: higher T1/T2 → lower score
        coherence_term = (1.0 / t1 + 1.0 / t2) * 10.0  # scale factor

        score = (
            self._config.coherence_weight * coherence_term
            + self._config.readout_weight * readout
        )

        # T1 asymmetry penalty
        if self._config.t1_asymmetry_weight > 0 and qp is not None:
            score += self._config.t1_asymmetry_weight * qp.t1_asymmetry_penalty

        # Temporal volatility penalty (if available)
        if (
            self._config.correlation_weight > 0
            and self._correlation is not None
        ):
            vol = self._correlation.qubit_volatility(physical_qubit)
            # Volatility is in readout error units (e.g. 0.01 = 1% swing)
            # Scale up to be comparable with other terms
            score += self._config.correlation_weight * vol * 100.0

        return score

    def _edge_score(
        self, phys_a: int, phys_b: int, interaction_count: int
    ) -> float:
        """Score for mapping a logical 2Q interaction to a physical edge."""
        # Look up the gate error for this physical edge
        error = self._get_two_qubit_error(phys_a, phys_b)
        score = self._config.gate_error_weight * error * interaction_count

        # Temporal correlation penalty
        if (
            self._config.correlation_weight > 0
            and self._correlation is not None
        ):
            corr = self._correlation.edge_correlation(phys_a, phys_b)
            # Positive correlation = errors move together = bad for QEC
            # Scale by interaction count: more interactions on a correlated
            # edge means more exposure
            if corr > 0:
                score += (
                    self._config.correlation_weight
                    * corr
                    * interaction_count
                    * 0.01  # scale to be comparable with gate error terms
                )

        return score

    def _get_two_qubit_error(self, phys_a: int, phys_b: int) -> float:
        """Best 2Q gate error between phys_a and phys_b in either direction."""
        best = _DEFAULT_GATE_ERROR
        for (_gtype, gqubits), gp in self._gate_map.items():
            if len(gqubits) == 2 and gp.error_rate is not None and set(gqubits) == {phys_a, phys_b}:
                best = min(best, gp.error_rate)
        return best

    # ── layout scoring ───────────────────────────────────────────────

    def _score_layout(
        self,
        layout: dict[int, int],
        interactions: dict[tuple[int, int], int],
        circuit: QBCircuit,
    ) -> float:
        """Total score for a candidate layout (lower is better).

        Combines:
        - Sum of qubit scores for all mapped logical qubits
        - Sum of edge scores for all 2Q interactions
        """
        score = 0.0

        # Qubit contribution
        for _logical_q, physical_q in layout.items():
            score += self._qubit_score(physical_q)

        # Edge contribution
        for (log_a, log_b), count in interactions.items():
            phys_a = layout[log_a]
            phys_b = layout[log_b]
            score += self._edge_score(phys_a, phys_b, count)

        return score

    def _score_breakdown(
        self,
        layout: dict[int, int],
        interactions: dict[tuple[int, int], int],
        circuit: QBCircuit,
    ) -> dict[str, float]:
        """Detailed per-component score breakdown for diagnostics."""
        gate_err = 0.0
        coherence = 0.0
        readout = 0.0
        t1_asym = 0.0
        correlation = 0.0

        for _logical_q, physical_q in layout.items():
            qp = self._qubit_map.get(physical_q)
            t1 = qp.t1_us if (qp and qp.t1_us) else _DEFAULT_T1_US
            t2 = qp.t2_us if (qp and qp.t2_us) else _DEFAULT_T2_US
            ro = (
                qp.readout_error
                if (qp and qp.readout_error is not None)
                else _DEFAULT_READOUT_ERROR
            )

            coherence_term = (1.0 / t1 + 1.0 / t2) * 10.0
            coherence += self._config.coherence_weight * coherence_term
            readout += self._config.readout_weight * ro

            if self._config.t1_asymmetry_weight > 0 and qp is not None:
                t1_asym += self._config.t1_asymmetry_weight * qp.t1_asymmetry_penalty

            if self._config.correlation_weight > 0 and self._correlation is not None:
                vol = self._correlation.qubit_volatility(physical_q)
                correlation += self._config.correlation_weight * vol * 100.0

        for (log_a, log_b), count in interactions.items():
            phys_a = layout[log_a]
            phys_b = layout[log_b]
            error = self._get_two_qubit_error(phys_a, phys_b)
            gate_err += self._config.gate_error_weight * error * count

            if self._config.correlation_weight > 0 and self._correlation is not None:
                corr = self._correlation.edge_correlation(phys_a, phys_b)
                if corr > 0:
                    correlation += (
                        self._config.correlation_weight
                        * corr * count * 0.01
                    )

        return {
            "gate_error": gate_err,
            "coherence": coherence,
            "readout": readout,
            "t1_asymmetry": t1_asym,
            "correlation": correlation,
            "total": gate_err + coherence + readout + t1_asym + correlation,
        }

    # ── best individual qubits (no 2Q gates) ─────────────────────────

    def _best_individual_qubits(self, circuit: QBCircuit) -> dict[int, int]:
        """Pick the N best physical qubits when there are no 2Q interactions."""
        n_logical = circuit.n_qubits

        # Score all physical qubits and pick the top N
        scored: list[tuple[float, int]] = []
        for phys_q in sorted(self._qubit_map.keys()):
            scored.append((self._qubit_score(phys_q), phys_q))

        # If we have fewer calibrated qubits than needed, also add uncalibrated
        if len(scored) < n_logical:
            calibrated = set(self._qubit_map.keys())
            for q in range(self._n_physical):
                if q not in calibrated:
                    scored.append((self._qubit_score(q), q))

        scored.sort()
        layout = {}
        for logical_q in range(n_logical):
            layout[logical_q] = scored[logical_q][1]
        return layout

    def _best_edge_direct(
        self,
        circuit: QBCircuit,
        interactions: dict[tuple[int, int], int],
    ) -> dict[int, int]:
        """Fast path for small circuits: directly pick the best physical edge.

        For circuits with <= 2 qubits and a single 2Q interaction, VF2
        subgraph search is unnecessary.  Instead, score every physical
        edge and pick the one with the lowest combined error.
        """
        (log_a, log_b), count = next(iter(interactions.items()))

        # Score every physical edge
        best_score = float("inf")
        best_pa, best_pb = 0, 1
        seen: set[tuple[int, int]] = set()
        for q1, q2 in self._coupling_map:
            edge_key = (min(q1, q2), max(q1, q2))
            if edge_key in seen:
                continue
            seen.add(edge_key)
            score = (
                self._edge_score(q1, q2, count)
                + self._qubit_score(q1)
                + self._qubit_score(q2)
            )
            if score < best_score:
                best_score = score
                best_pa, best_pb = q1, q2

        layout = {log_a: best_pa, log_b: best_pb}
        # Complete layout for any other logical qubits (e.g. ancilla)
        return self._complete_layout(circuit, layout)

    # ── VF2-based layout search ──────────────────────────────────────

    def _find_best_layout(
        self,
        circuit: QBCircuit,
        interactions: dict[tuple[int, int], int],
    ) -> dict[int, int]:
        """Use VF2 subgraph isomorphism to find the best layout.

        1. Build the logical interaction graph (undirected)
        2. Build the physical coupling graph (undirected)
        3. Enumerate candidate subgraph isomorphisms
        4. Score each and return the best
        """
        # Build logical interaction graph
        logical_graph = rx.PyGraph()
        logical_nodes: dict[int, int] = {}  # logical qubit -> node index
        for (log_a, log_b) in interactions:
            for q in (log_a, log_b):
                if q not in logical_nodes:
                    logical_nodes[q] = logical_graph.add_node(q)
            logical_graph.add_edge(logical_nodes[log_a], logical_nodes[log_b], None)

        # ML-accelerated candidate filtering: restrict physical graph
        # to the top-K most promising qubits predicted by the ML model.
        ml_candidates: set[int] | None = None
        if self._layout_predictor is not None and self._backend_props is not None:
            try:
                candidate_list = self._layout_predictor.predict_candidate_qubits(
                    circuit, self._backend_props
                )
                ml_candidates = set(candidate_list)
                logger.info(
                    "ML predictor narrowed search to %d/%d physical qubits",
                    len(ml_candidates),
                    self._n_physical,
                )
            except Exception:
                logger.warning(
                    "ML predictor failed, falling back to full search",
                    exc_info=True,
                )

        # Build physical coupling graph (undirected)
        # If ML candidates are available, only include those qubits.
        coupling_edges = self._coupling_map
        if ml_candidates is not None:
            coupling_edges = [
                (q1, q2)
                for q1, q2 in self._coupling_map
                if q1 in ml_candidates and q2 in ml_candidates
            ]

        physical_graph = rx.PyGraph()
        physical_nodes: dict[int, int] = {}  # physical qubit -> node index
        seen_edges: set[tuple[int, int]] = set()
        for (q1, q2) in coupling_edges:
            for q in (q1, q2):
                if q not in physical_nodes:
                    physical_nodes[q] = physical_graph.add_node(q)
            edge_key = (min(q1, q2), max(q1, q2))
            if edge_key not in seen_edges:
                physical_graph.add_edge(
                    physical_nodes[q1], physical_nodes[q2], None
                )
                seen_edges.add(edge_key)

        # Reverse map: node index -> physical qubit id
        phys_node_to_qubit = {nid: q for q, nid in physical_nodes.items()}
        log_node_to_qubit = {nid: q for q, nid in logical_nodes.items()}

        # Find candidate mappings via VF2 subgraph isomorphism
        # physical_graph is "first", logical_graph is "second"
        # mapping: physical_node_idx -> logical_node_idx
        vf2_iter = rx.vf2_mapping(
            physical_graph,
            logical_graph,
            subgraph=True,
            induced=False,
            id_order=False,
            call_limit=self._config.vf2_call_limit,
        )

        best_layout: dict[int, int] | None = None
        best_score = float("inf")
        n_evaluated = 0

        for mapping in vf2_iter:
            if n_evaluated >= self._config.max_candidates:
                break

            # mapping: {physical_node_idx: logical_node_idx}
            # Convert to {logical_qubit: physical_qubit}
            layout: dict[int, int] = {}
            for phys_node, log_node in mapping.items():
                phys_q = phys_node_to_qubit[phys_node]
                log_q = log_node_to_qubit[log_node]
                layout[log_q] = phys_q

            score = self._score_layout(layout, interactions, circuit)
            if score < best_score:
                best_score = score
                best_layout = layout

            n_evaluated += 1

        if best_layout is None and ml_candidates is not None:
            # ML-restricted search found nothing — retry with full graph
            logger.info(
                "CalibrationMapper: ML-restricted VF2 found no mapping, "
                "retrying with full coupling map"
            )
            return self._find_best_layout_full(circuit, interactions)

        if best_layout is None:
            # VF2 found no subgraph isomorphism — fall back to greedy
            logger.warning(
                "CalibrationMapper: VF2 found no mapping, falling back to greedy"
            )
            best_layout = self._greedy_layout(circuit, interactions)
        else:
            # Ensure all logical qubits are in the layout (some may only
            # appear in 1Q gates and not be in the interaction graph)
            best_layout = self._complete_layout(circuit, best_layout)

        logger.debug(
            "CalibrationMapper: evaluated %d candidates, best_score=%.6f",
            n_evaluated,
            best_score,
        )
        return best_layout

    def _find_best_layout_full(
        self,
        circuit: QBCircuit,
        interactions: dict[tuple[int, int], int],
    ) -> dict[int, int]:
        """Run VF2 without ML filtering (fallback)."""
        # Temporarily disable the ML predictor for one call
        predictor = self._layout_predictor
        self._layout_predictor = None
        try:
            return self._find_best_layout(circuit, interactions)
        finally:
            self._layout_predictor = predictor

    def _complete_layout(
        self, circuit: QBCircuit, partial: dict[int, int]
    ) -> dict[int, int]:
        """Fill in any unmapped logical qubits with the best available physical qubits."""
        used_physical = set(partial.values())
        layout = dict(partial)

        # Score available physical qubits
        available = []
        for phys_q in sorted(self._qubit_map.keys()):
            if phys_q not in used_physical:
                available.append((self._qubit_score(phys_q), phys_q))
        available.sort()

        avail_idx = 0
        for log_q in range(circuit.n_qubits):
            if log_q not in layout:
                if avail_idx < len(available):
                    layout[log_q] = available[avail_idx][1]
                    used_physical.add(available[avail_idx][1])
                    avail_idx += 1
                else:
                    # Last resort: find any unused physical qubit
                    for q in range(self._n_physical):
                        if q not in used_physical:
                            layout[log_q] = q
                            used_physical.add(q)
                            break

        return layout

    def _greedy_layout(
        self,
        circuit: QBCircuit,
        interactions: dict[tuple[int, int], int],
    ) -> dict[int, int]:
        """Greedy fallback: place highest-interaction qubits on lowest-error edges."""
        # Sort interactions by count (most interactions first)
        sorted_interactions = sorted(interactions.items(), key=lambda x: -x[1])

        # Score all physical edges
        edge_scores: list[tuple[float, int, int]] = []
        seen = set()
        for (q1, q2) in self._coupling_map:
            edge_key = (min(q1, q2), max(q1, q2))
            if edge_key not in seen:
                seen.add(edge_key)
                err = self._get_two_qubit_error(q1, q2)
                edge_scores.append((err, edge_key[0], edge_key[1]))
        edge_scores.sort()

        layout: dict[int, int] = {}
        used_physical: set[int] = set()

        edge_idx = 0
        for (log_a, log_b), _count in sorted_interactions:
            if log_a in layout and log_b in layout:
                continue

            # Find the best available edge
            while edge_idx < len(edge_scores):
                _, pa, pb = edge_scores[edge_idx]
                if pa not in used_physical and pb not in used_physical:
                    break
                # Allow partial reuse
                if log_a not in layout and pa not in used_physical and log_b in layout:
                    layout[log_a] = pa
                    used_physical.add(pa)
                    break
                if log_b not in layout and pb not in used_physical and log_a in layout:
                    layout[log_b] = pb
                    used_physical.add(pb)
                    break
                edge_idx += 1
            else:
                break

            if edge_idx < len(edge_scores):
                _, pa, pb = edge_scores[edge_idx]
                if log_a not in layout:
                    layout[log_a] = pa
                    used_physical.add(pa)
                if log_b not in layout:
                    layout[log_b] = pb
                    used_physical.add(pb)
                edge_idx += 1

        # Complete unmapped qubits
        return self._complete_layout(circuit, layout)

    # ── apply the layout ─────────────────────────────────────────────

    @staticmethod
    def _apply_layout(circuit: QBCircuit, layout: dict[int, int]) -> QBCircuit:
        """Produce a new circuit with qubit indices remapped according to *layout*.

        The output circuit has ``n_qubits = max(physical qubit ids) + 1``.
        """
        max_phys = max(layout.values()) if layout else 0
        new_n_qubits = max_phys + 1

        mapped = QBCircuit(
            n_qubits=new_n_qubits,
            n_clbits=circuit.n_clbits,
            name=circuit.name,
        )

        for op in circuit.iter_ops():
            if isinstance(op, QBGate):
                new_qubits = tuple(layout[q] for q in op.qubits)
                mapped.add_gate(
                    QBGate(
                        name=op.name,
                        qubits=new_qubits,
                        params=op.params,
                        condition=op.condition,
                    )
                )
            elif isinstance(op, QBMeasure):
                mapped.add_measurement(layout[op.qubit], op.clbit)
            else:
                # QBBarrier
                from qb_compiler.ir.operations import QBBarrier

                if isinstance(op, QBBarrier):
                    new_qubits_b = tuple(layout[q] for q in op.qubits)
                    mapped.add_barrier(new_qubits_b)

        return mapped
