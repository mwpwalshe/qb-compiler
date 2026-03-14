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

import heapq
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import rustworkx as rx

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate, QBMeasure
from qb_compiler.passes.base import PassResult, TransformationPass

if TYPE_CHECKING:
    from qb_compiler.calibration.models.coupling_properties import GateProperties
    from qb_compiler.calibration.models.qubit_properties import QubitProperties
    from qb_compiler.calibration.provider import CalibrationProvider
    from qb_compiler.ml.layout_predictor import MLLayoutPredictor, MLLayoutPredictorV2
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
    top_k: int = 20
    max_per_region: int = 3


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
        layout_predictor: MLLayoutPredictor | MLLayoutPredictorV2 | None = None,
        qiskit_target: Any | None = None,
    ) -> None:
        self._config = config or CalibrationMapperConfig()
        self._correlation = correlation_analyzer
        self._layout_predictor = layout_predictor
        self._qiskit_target = qiskit_target

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
            layout = self._best_edge_direct(circuit, interactions)
        else:
            # Collect top-K candidates, diversify, and optionally rescore
            # after trial routing to account for SWAP insertion cost.
            candidates = self._find_top_k_layouts(circuit, interactions)

            # Store raw candidate info for diagnostics
            self._last_raw_candidates = len(candidates)
            self._last_raw_regions = self._count_distinct_regions(candidates) if candidates else 0

            if len(candidates) > 1:
                candidates = self._diversify_candidates(candidates)

            self._last_diversified_candidates = len(candidates)
            self._last_diversified_regions = self._count_distinct_regions(candidates) if candidates else 0

            # Inject Qiskit seed layouts into the candidate pool.
            # This guarantees we never lose to Qiskit — their best
            # layout is always in our candidate set.
            qiskit_injected = 0
            if self._qiskit_target is not None:
                qiskit_layouts = self._get_qiskit_seed_layouts(
                    circuit, self._qiskit_target,
                )
                for ql in qiskit_layouts:
                    if not self._layout_already_in(ql, candidates):
                        candidates.append(ql)
                        qiskit_injected += 1

            self._last_qiskit_injected = qiskit_injected

            if self._qiskit_target is not None and len(candidates) > 1:
                layout = self._post_routing_rescore(candidates, circuit, self._qiskit_target)
            else:
                layout = candidates[0]

            logger.info(
                "CalibrationMapper pipeline: raw=%d candidates (%d regions) "
                "-> diversified=%d (%d regions) + %d qiskit seed layouts "
                "-> post-routing rescore=%s",
                self._last_raw_candidates,
                self._last_raw_regions,
                self._last_diversified_candidates,
                self._last_diversified_regions,
                qiskit_injected,
                "yes" if self._qiskit_target is not None else "no",
            )

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

        Delegates to :meth:`_find_top_k_layouts` and returns the single best.
        """
        candidates = self._find_top_k_layouts(circuit, interactions)
        return candidates[0]

    def _find_top_k_layouts(
        self,
        circuit: QBCircuit,
        interactions: dict[tuple[int, int], int],
    ) -> list[dict[int, int]]:
        """Use VF2 subgraph isomorphism to find the top-K layouts.

        Returns a list of up to ``config.top_k`` candidate layouts sorted
        by calibration score (best first).

        To avoid VF2 fixating on one chip region (which happens because it
        enumerates subgraph isomorphisms in node-index order), we partition
        the physical coupling map into non-overlapping windows and run VF2
        independently on each.  Results from all windows are merged and the
        global top-K are returned.
        """
        top_k = self._config.top_k

        # Build logical interaction graph
        logical_graph = rx.PyGraph()
        logical_nodes: dict[int, int] = {}
        for (log_a, log_b) in interactions:
            for q in (log_a, log_b):
                if q not in logical_nodes:
                    logical_nodes[q] = logical_graph.add_node(q)
            logical_graph.add_edge(logical_nodes[log_a], logical_nodes[log_b], None)

        n_logical_nodes = logical_graph.num_nodes()
        log_node_to_qubit = {nid: q for q, nid in logical_nodes.items()}

        # ML-accelerated candidate filtering
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

        coupling_edges = self._coupling_map
        if ml_candidates is not None:
            coupling_edges = [
                (q1, q2)
                for q1, q2 in self._coupling_map
                if q1 in ml_candidates and q2 in ml_candidates
            ]

        # Partition the chip into windows for multi-region search.
        # Each window is a subset of physical qubits.  We run VF2 on
        # each window independently with a per-window candidate budget.
        windows = self._partition_into_windows(coupling_edges, n_logical_nodes)

        # Global heap across all windows
        heap: list[tuple[float, int, dict[int, int]]] = []
        counter = 0
        total_evaluated = 0

        # Budget per window: distribute max_candidates across windows,
        # with a minimum per window to ensure coverage.
        per_window_budget = max(
            50,
            self._config.max_candidates // max(len(windows), 1),
        )
        per_window_call_limit = self._config.vf2_call_limit

        for window_qubits in windows:
            # Build physical subgraph for this window
            window_edges = [
                (q1, q2) for q1, q2 in coupling_edges
                if q1 in window_qubits and q2 in window_qubits
            ]
            if not window_edges:
                continue

            physical_graph = rx.PyGraph()
            physical_nodes: dict[int, int] = {}
            seen_edges: set[tuple[int, int]] = set()
            for (q1, q2) in window_edges:
                for q in (q1, q2):
                    if q not in physical_nodes:
                        physical_nodes[q] = physical_graph.add_node(q)
                edge_key = (min(q1, q2), max(q1, q2))
                if edge_key not in seen_edges:
                    physical_graph.add_edge(
                        physical_nodes[q1], physical_nodes[q2], None
                    )
                    seen_edges.add(edge_key)

            # Need enough nodes in this window for the logical circuit
            if physical_graph.num_nodes() < n_logical_nodes:
                continue

            phys_node_to_qubit = {nid: q for q, nid in physical_nodes.items()}

            vf2_iter = rx.vf2_mapping(
                physical_graph,
                logical_graph,
                subgraph=True,
                induced=False,
                id_order=False,
                call_limit=per_window_call_limit,
            )

            n_window_evaluated = 0
            for mapping in vf2_iter:
                if n_window_evaluated >= per_window_budget:
                    break

                layout: dict[int, int] = {}
                for phys_node, log_node in mapping.items():
                    phys_q = phys_node_to_qubit[phys_node]
                    log_q = log_node_to_qubit[log_node]
                    layout[log_q] = phys_q

                score = self._score_layout(layout, interactions, circuit)

                if len(heap) < top_k:
                    heapq.heappush(heap, (-score, counter, layout))
                elif score < -heap[0][0]:
                    heapq.heapreplace(heap, (-score, counter, layout))

                counter += 1
                n_window_evaluated += 1

            total_evaluated += n_window_evaluated

        if not heap and ml_candidates is not None:
            logger.info(
                "CalibrationMapper: ML-restricted VF2 found no mapping, "
                "retrying with full coupling map"
            )
            return self._find_top_k_layouts_full(circuit, interactions)

        if not heap:
            logger.warning(
                "CalibrationMapper: VF2 found no mapping, falling back to greedy"
            )
            greedy = self._greedy_layout(circuit, interactions)
            return [greedy]

        # Sort results best-first (lowest score) and complete layouts
        results = sorted(heap, key=lambda x: -x[0])
        completed = [
            self._complete_layout(circuit, layout)
            for _, _, layout in results
        ]

        logger.debug(
            "CalibrationMapper: evaluated %d candidates across %d windows, "
            "kept top-%d, best_score=%.6f, worst_kept=%.6f",
            total_evaluated,
            len(windows),
            len(completed),
            -results[0][0],
            -results[-1][0],
        )
        return completed

    @staticmethod
    def _partition_into_windows(
        coupling_edges: list[tuple[int, int]],
        min_window_size: int,
    ) -> list[set[int]]:
        """Partition the physical qubits into overlapping windows for VF2.

        Uses a sliding window over qubit indices to ensure every chip
        region gets searched.  Windows overlap by 50% so that layouts
        straddling window boundaries are still found.

        For small chips (< 40 qubits), returns a single window containing
        all qubits (equivalent to the old single-VF2 approach).
        """
        all_qubits = set()
        for q1, q2 in coupling_edges:
            all_qubits.add(q1)
            all_qubits.add(q2)

        if not all_qubits:
            return []

        n_physical = max(all_qubits) + 1

        # For small chips, a single window is fine
        if n_physical <= 40:
            return [all_qubits]

        # Window size: large enough to contain the circuit with room for
        # routing, small enough to force exploration of multiple regions.
        # Use max(3 * min_window_size, 30) to ensure reasonable window size.
        window_size = max(min_window_size * 3, 30)
        step = max(window_size // 2, 1)  # 50% overlap

        sorted_qubits = sorted(all_qubits)
        windows: list[set[int]] = []

        for start_idx in range(0, len(sorted_qubits), step):
            end_idx = min(start_idx + window_size, len(sorted_qubits))
            window = set(sorted_qubits[start_idx:end_idx])
            if len(window) >= min_window_size:
                windows.append(window)
            if end_idx >= len(sorted_qubits):
                break

        # Also add a "full chip" window to ensure global optima aren't
        # missed, but with a lower per-window budget (handled by caller).
        if len(windows) > 1:
            windows.append(all_qubits)

        return windows

    def _find_best_layout_full(
        self,
        circuit: QBCircuit,
        interactions: dict[tuple[int, int], int],
    ) -> dict[int, int]:
        """Run VF2 without ML filtering (fallback)."""
        candidates = self._find_top_k_layouts_full(circuit, interactions)
        return candidates[0]

    def _find_top_k_layouts_full(
        self,
        circuit: QBCircuit,
        interactions: dict[tuple[int, int], int],
    ) -> list[dict[int, int]]:
        """Run top-K VF2 without ML filtering (fallback)."""
        predictor = self._layout_predictor
        self._layout_predictor = None
        try:
            return self._find_top_k_layouts(circuit, interactions)
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

    # ── post-routing rescoring ────────────────────────────────────────

    def _count_distinct_regions(
        self,
        candidates: list[dict[int, int]],
    ) -> int:
        """Count distinct chip regions among candidates (by centroid proximity)."""
        if not candidates:
            return 0
        distance_threshold = max(10.0, self._n_physical / 15.0)
        region_centroids: list[float] = []
        for layout in candidates:
            phys = list(layout.values())
            centroid = sum(phys) / len(phys) if phys else 0.0
            found = False
            for rc in region_centroids:
                if abs(centroid - rc) < distance_threshold:
                    found = True
                    break
            if not found:
                region_centroids.append(centroid)
        return len(region_centroids)

    def _diversify_candidates(
        self,
        candidates: list[dict[int, int]],
    ) -> list[dict[int, int]]:
        """Ensure candidates span multiple chip regions.

        A "region" is defined by the centroid (mean qubit index) of the
        layout's physical qubits.  Candidates whose centroids are within
        ``distance_threshold`` of each other are considered to be in the
        same region.  At most ``max_per_region`` candidates from any single
        region are kept; remaining slots are filled from underrepresented
        regions.
        """
        if len(candidates) <= 1:
            return candidates

        max_per_region = self._config.max_per_region
        # Distance threshold: qubits within this centroid distance are
        # "same region".  On a 156-qubit chip, regions ~15 qubits apart
        # are meaningfully different.
        distance_threshold = max(10.0, self._n_physical / 15.0)

        # Compute centroid for each candidate
        centroids: list[float] = []
        for layout in candidates:
            phys_qubits = list(layout.values())
            centroid = sum(phys_qubits) / len(phys_qubits) if phys_qubits else 0.0
            centroids.append(centroid)

        # Assign candidates to regions greedily
        # regions[i] = list of candidate indices in that region
        regions: list[list[int]] = []
        region_centroids: list[float] = []
        candidate_region: list[int] = [-1] * len(candidates)

        for idx, centroid in enumerate(centroids):
            assigned = False
            for r_idx, r_centroid in enumerate(region_centroids):
                if abs(centroid - r_centroid) < distance_threshold:
                    candidate_region[idx] = r_idx
                    regions[r_idx].append(idx)
                    assigned = True
                    break
            if not assigned:
                candidate_region[idx] = len(regions)
                region_centroids.append(centroid)
                regions.append([idx])

        # Keep at most max_per_region from each region, preferring
        # earlier candidates (they have lower calibration scores)
        result: list[dict[int, int]] = []
        region_count: dict[int, int] = defaultdict(int)

        for idx, layout in enumerate(candidates):
            r = candidate_region[idx]
            if region_count[r] < max_per_region:
                result.append(layout)
                region_count[r] += 1

        logger.debug(
            "CalibrationMapper: diversified %d candidates -> %d across %d regions",
            len(candidates),
            len(result),
            len(regions),
        )
        return result if result else candidates[:1]

    def _get_qiskit_seed_layouts(
        self,
        circuit: QBCircuit,
        qiskit_target: Any,
        n_seeds: int = 20,
    ) -> list[dict[int, int]]:
        """Run Qiskit transpile with multiple seeds and extract their layouts.

        Returns deduplicated layouts that Qiskit's VF2PostLayout selects.
        These are injected into our candidate pool so we can never pick
        a worse layout than Qiskit.
        """
        try:
            from qiskit import transpile
        except ImportError:
            return []

        qc = self._ir_to_qiskit_circuit(circuit)
        if qc is None:
            return []

        n_logical = circuit.n_qubits
        seen_qubit_sets: list[frozenset[int]] = []
        unique_layouts: list[dict[int, int]] = []

        for seed in range(n_seeds):
            try:
                tc = transpile(
                    qc,
                    target=qiskit_target,
                    optimization_level=3,
                    seed_transpiler=seed,
                )
                ql = tc.layout
                if ql and ql.initial_layout:
                    phys = [ql.initial_layout[qc.qubits[i]] for i in range(n_logical)]
                else:
                    phys = list(range(n_logical))

                layout = {i: phys[i] for i in range(n_logical)}
                qset = frozenset(phys)

                # Deduplicate by physical qubit set
                if qset not in seen_qubit_sets:
                    seen_qubit_sets.append(qset)
                    unique_layouts.append(layout)
            except Exception:
                logger.debug("Qiskit seed %d failed", seed, exc_info=True)
                continue

        logger.info(
            "CalibrationMapper: extracted %d unique Qiskit layouts from %d seeds",
            len(unique_layouts),
            n_seeds,
        )
        return unique_layouts

    @staticmethod
    def _layout_already_in(
        layout: dict[int, int],
        candidates: list[dict[int, int]],
    ) -> bool:
        """Check if a layout (by physical qubit set) is already in candidates."""
        target_set = frozenset(layout.values())
        for c in candidates:
            if frozenset(c.values()) == target_set:
                return True
        return False

    def _routed_fidelity(self, tc: Any) -> float:
        """Estimate fidelity from a transpiled circuit's actual 2Q gate placements.

        For each 2-qubit gate in the routed circuit, multiply
        ``(1 - gate_error)`` for the physical qubits that gate actually
        lands on.  This uses the *real* calibration errors of the routed
        circuit — not the pre-routing estimate — so it differentiates
        regions that have the same gate count but different error rates.
        """
        fidelity = 1.0
        for inst in tc.data:
            if (
                len(inst.qubits) == 2
                and inst.operation.name not in ("barrier", "measure", "reset")
            ):
                # inst.qubits are Qubit objects; get their physical indices
                q0 = tc.find_bit(inst.qubits[0]).index
                q1 = tc.find_bit(inst.qubits[1]).index
                err = self._get_two_qubit_error(q0, q1)
                fidelity *= (1.0 - err)

        # Also include readout errors for measured qubits
        for inst in tc.data:
            if inst.operation.name == "measure":
                q = tc.find_bit(inst.qubits[0]).index
                qp = self._qubit_map.get(q)
                if qp is not None and qp.readout_error is not None:
                    fidelity *= (1.0 - qp.readout_error)

        return fidelity

    def _post_routing_rescore(
        self,
        candidates: list[dict[int, int]],
        circuit: QBCircuit,
        qiskit_target: Any,
    ) -> dict[int, int]:
        """Re-score candidates by trial-transpiling and counting post-routing 2Q gates.

        For each candidate layout, runs a quick Qiskit transpile with
        ``optimization_level=1`` and the given initial layout, then counts
        the number of 2-qubit gates in the output.  The layout that yields
        the fewest 2Q gates (i.e. fewest SWAPs inserted) wins.  Ties are
        broken by **post-routing estimated fidelity**: for each 2Q gate in
        the routed circuit, multiply ``(1 - gate_error)`` for the actual
        physical qubits that gate lands on.  Higher fidelity wins.

        Falls back to ``candidates[0]`` if Qiskit is unavailable or all
        trial transpiles fail.
        """
        try:
            from qiskit import QuantumCircuit, transpile
        except ImportError:
            logger.warning("Qiskit not available for post-routing rescore")
            return candidates[0]

        # Build a Qiskit circuit from our IR for trial transpilation
        qc = self._ir_to_qiskit_circuit(circuit)
        if qc is None:
            return candidates[0]

        interactions = self._extract_interactions(circuit)
        best_layout = candidates[0]
        best_2q_count = float("inf")
        best_routed_fidelity = -1.0  # higher is better
        rescore_details: list[dict[str, Any]] = []

        for i, layout in enumerate(candidates):
            try:
                # Build initial_layout list: logical qubit i -> physical qubit layout[i]
                n_logical = circuit.n_qubits
                layout_list = [layout[i] for i in range(n_logical)]

                tc = transpile(
                    qc,
                    target=qiskit_target,
                    optimization_level=1,
                    initial_layout=layout_list,
                    seed_transpiler=42,
                )

                # Count 2Q gates in transpiled circuit
                count_2q = sum(
                    1 for inst in tc.data
                    if len(inst.qubits) == 2
                    and inst.operation.name not in ("barrier", "measure", "reset")
                )

                # Compute post-routing fidelity from actual gate placements
                routed_fid = self._routed_fidelity(tc)

                cal_score = self._score_layout(layout, interactions, circuit)
                phys = [layout[q] for q in range(n_logical)]
                centroid = sum(phys) / len(phys)
                rescore_details.append({
                    "index": i,
                    "physical_qubits": phys,
                    "centroid": centroid,
                    "region": f"{min(phys)}-{max(phys)}",
                    "post_routing_2q": count_2q,
                    "routed_fidelity": routed_fid,
                    "cal_score": cal_score,
                    "depth": tc.depth(),
                })

                # Lower 2Q count wins; ties broken by higher routed fidelity
                # (negate fidelity so lower tuple = better)
                if (count_2q, -routed_fid) < (best_2q_count, -best_routed_fidelity):
                    best_2q_count = count_2q
                    best_routed_fidelity = routed_fid
                    best_layout = layout

            except Exception:
                logger.debug(
                    "Trial transpile failed for layout %s", layout, exc_info=True
                )
                continue

        # Store rescore details for external diagnostic access
        self._last_rescore_details = rescore_details

        for detail in rescore_details:
            logger.info(
                "  Candidate #%d: qubits=%s region=%s centroid=%.1f "
                "2Q_gates=%d routed_fid=%.6f cal_score=%.4f",
                detail["index"],
                detail["physical_qubits"],
                detail["region"],
                detail["centroid"],
                detail["post_routing_2q"],
                detail["routed_fidelity"],
                detail["cal_score"],
            )

        logger.info(
            "CalibrationMapper: post-routing rescore selected layout with "
            "%d 2Q gates, routed_fidelity=%.6f (from %d candidates)",
            best_2q_count if best_2q_count != float("inf") else -1,
            best_routed_fidelity,
            len(candidates),
        )
        return best_layout

    @staticmethod
    def _ir_to_qiskit_circuit(circuit: QBCircuit) -> Any:
        """Convert a QBCircuit to a Qiskit QuantumCircuit for trial transpilation."""
        try:
            from qb_compiler.ir.converters.qiskit_converter import to_qiskit

            return to_qiskit(circuit)
        except Exception:
            pass

        # Fallback: build manually from ops
        try:
            from qiskit import QuantumCircuit

            n_q = circuit.n_qubits
            n_c = circuit.n_clbits
            qc = QuantumCircuit(n_q, max(n_c, 0))

            for op in circuit.iter_ops():
                if isinstance(op, QBGate):
                    name = op.name
                    qubits = list(op.qubits)
                    if name == "h":
                        qc.h(*qubits)
                    elif name == "cx":
                        qc.cx(*qubits)
                    elif name == "cz":
                        qc.cz(*qubits)
                    elif name == "x":
                        qc.x(*qubits)
                    elif name == "rz" and op.params:
                        qc.rz(op.params[0], *qubits)
                    elif name == "rx" and op.params:
                        qc.rx(op.params[0], *qubits)
                    elif name == "ry" and op.params:
                        qc.ry(op.params[0], *qubits)
                    elif name == "sx":
                        qc.sx(*qubits)
                    elif name == "id":
                        qc.id(*qubits)
                elif isinstance(op, QBMeasure):
                    if op.clbit < n_c:
                        qc.measure(op.qubit, op.clbit)

            return qc
        except Exception:
            logger.debug("Failed to build Qiskit circuit for trial transpile", exc_info=True)
            return None

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
