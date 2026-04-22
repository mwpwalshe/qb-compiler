"""Fidelity-optimal compilation strategy.

Builds a pass pipeline that prioritises estimated output fidelity by:
1. Using calibration-aware initial layout (place logical qubits on the
   lowest-error physical qubits).
2. Noise-aware routing (prefer SWAP paths through low-error couplings).
3. Gate cancellation and commutation optimisations.
4. T1/T2-aware scheduling (minimise idle time on short-T2 qubits).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qb_compiler.strategies.base import CompilationStrategy, PassConfig, PassManager

if TYPE_CHECKING:
    from qb_compiler.calibration.provider import CalibrationProvider
    from qb_compiler.config import CompilerConfig
    from qb_compiler.noise.noise_model import NoiseModel


class FidelityOptimalStrategy(CompilationStrategy):
    """Compilation strategy that maximises expected output fidelity.

    This strategy is more expensive than a depth-optimal or
    gate-count-optimal strategy, but produces circuits that run better
    on noisy hardware.
    """

    @property
    def name(self) -> str:
        return "fidelity_optimal"

    def build_pass_manager(
        self,
        config: CompilerConfig,
        calibration: CalibrationProvider | None = None,
        noise_model: NoiseModel | None = None,
    ) -> PassManager:
        pm = PassManager()

        # ── 1. Analysis ──────────────────────────────────────────────
        pm.append(
            PassConfig(
                name="circuit_analysis",
                pass_type="analysis",
                options={"collect_gate_counts": True, "collect_depth": True},
            )
        )

        # ── 2. High-level optimisation (before decomposition) ────────
        opt_level = config.optimization_level
        if opt_level >= 1:
            pm.append(
                PassConfig(
                    name="gate_cancellation",
                    pass_type="optimization",
                    options={"max_iterations": 2 if opt_level < 3 else 5},
                )
            )
            pm.append(
                PassConfig(
                    name="commutation_analysis",
                    pass_type="optimization",
                    options={"enabled": opt_level >= 2},
                )
            )

        # ── 3. Basis decomposition ───────────────────────────────────
        basis = config.effective_basis_gates
        if basis:
            pm.append(
                PassConfig(
                    name="basis_translation",
                    pass_type="decomposition",
                    options={"target_basis": list(basis)},
                )
            )

        # ── 4. Layout ────────────────────────────────────────────────
        if calibration is not None and noise_model is not None:
            # Calibration-aware: rank physical qubits by combined error
            qubit_errors = _rank_qubits_by_error(calibration, noise_model)
            pm.append(
                PassConfig(
                    name="noise_aware_layout",
                    pass_type="routing",
                    options={
                        "qubit_ranking": qubit_errors,
                        "method": "vf2" if opt_level >= 2 else "trivial",
                    },
                )
            )
        else:
            pm.append(
                PassConfig(
                    name="trivial_layout",
                    pass_type="routing",
                    options={},
                )
            )

        # ── 5. Routing ───────────────────────────────────────────────
        coupling = config.coupling_map
        if coupling:
            routing_opts: dict = {"coupling_map": coupling}
            if noise_model is not None:
                # Weight edges by gate error so the router prefers
                # low-error SWAP paths
                edge_weights = _compute_edge_weights(coupling, noise_model)
                routing_opts["edge_weights"] = edge_weights
                routing_opts["method"] = "sabre_noise_aware"
            else:
                routing_opts["method"] = "sabre"

            if config.seed is not None:
                routing_opts["seed"] = config.seed

            pm.append(
                PassConfig(
                    name="swap_routing",
                    pass_type="routing",
                    options=routing_opts,
                )
            )

        # ── 6. Post-routing optimisation ─────────────────────────────
        if opt_level >= 2:
            pm.append(
                PassConfig(
                    name="post_route_cancellation",
                    pass_type="optimization",
                    options={"max_iterations": 3},
                )
            )

        if opt_level >= 3:
            pm.append(
                PassConfig(
                    name="peephole_optimization",
                    pass_type="optimization",
                    options={"window_size": 6},
                )
            )

        # ── 7. Scheduling ────────────────────────────────────────────
        if config.enable_noise_aware_scheduling and noise_model is not None:
            pm.append(
                PassConfig(
                    name="t2_aware_scheduling",
                    pass_type="scheduling",
                    options={"strategy": "alap_noise_aware"},
                )
            )
        else:
            pm.append(
                PassConfig(
                    name="alap_scheduling",
                    pass_type="scheduling",
                    options={"strategy": "alap"},
                )
            )

        # ── 8. Final analysis ────────────────────────────────────────
        pm.append(
            PassConfig(
                name="fidelity_estimation",
                pass_type="analysis",
                options={"estimate_fidelity": True},
            )
        )

        return pm


def _rank_qubits_by_error(
    calibration: CalibrationProvider,
    noise_model: NoiseModel,
) -> list[int]:
    """Return qubit indices sorted from lowest to highest error."""
    qprops = calibration.get_all_qubit_properties()
    scored = []
    for qp in qprops:
        err = noise_model.qubit_error(qp.qubit_id)
        scored.append((err, qp.qubit_id))
    scored.sort()
    return [qid for _, qid in scored]


def _compute_edge_weights(
    coupling_map: list[tuple[int, int]],
    noise_model: NoiseModel,
) -> dict[tuple[int, int], float]:
    """Assign weights to coupling edges based on 2-qubit gate error.

    Lower weight = preferred path.  We use ``-log(1 - error)`` so that
    multiplying fidelities becomes additive in the weight space.
    """
    import math

    weights: dict[tuple[int, int], float] = {}
    for q1, q2 in coupling_map:
        # Try common 2-qubit gate names
        err = None
        for gate in ("cz", "cx", "ecr", "rzz", "ms"):
            e = noise_model.gate_error(gate, (q1, q2))
            # Only use if it looks like real data (not default)
            if e < 0.019:  # below the conservative default of 0.02
                err = e
                break
        if err is None:
            err = noise_model.gate_error("cz", (q1, q2))

        # Clamp to avoid log(0)
        err = min(max(err, 1e-10), 1.0 - 1e-10)
        weights[(q1, q2)] = -math.log(1.0 - err)
    return weights
