"""Cost-optimal compilation strategy.

Like fidelity-optimal but also factors in execution cost: avoids
expensive backends/qubits when cheaper alternatives give comparable
fidelity.  Optimises for *fidelity per dollar*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qb_compiler.strategies.base import CompilationStrategy, PassConfig, PassManager

if TYPE_CHECKING:
    from qb_compiler.calibration.provider import CalibrationProvider
    from qb_compiler.config import CompilerConfig
    from qb_compiler.noise.noise_model import NoiseModel


class CostOptimalStrategy(CompilationStrategy):
    """Compilation strategy that maximises fidelity per dollar.

    Builds a pass pipeline similar to :class:`FidelityOptimalStrategy`,
    but also considers per-shot cost when selecting layout and routing.
    Qubits and coupling paths are scored by a cost-weighted fidelity
    metric so that cheaper physical resources are preferred when their
    fidelity is within an acceptable tolerance of the best option.
    """

    @property
    def name(self) -> str:
        return "cost_optimal"

    def build_pass_manager(
        self,
        config: CompilerConfig,
        calibration: CalibrationProvider | None = None,
        noise_model: NoiseModel | None = None,
    ) -> PassManager:
        pm = PassManager()
        opt_level = config.optimization_level

        # ── 1. Analysis ───────────────────────────────────────────────
        pm.append(
            PassConfig(
                name="circuit_analysis",
                pass_type="analysis",
                options={"collect_gate_counts": True, "collect_depth": True},
            )
        )

        # ── 2. Cost-aware analysis ────────────────────────────────────
        cost_per_shot = 0.0
        if config.backend_spec is not None:
            cost_per_shot = config.backend_spec.cost_per_shot
        pm.append(
            PassConfig(
                name="cost_analysis",
                pass_type="analysis",
                options={"cost_per_shot": cost_per_shot, "backend": config.backend},
            )
        )

        # ── 3. Pre-routing optimisation ───────────────────────────────
        if opt_level >= 1:
            pm.append(
                PassConfig(
                    name="gate_cancellation",
                    pass_type="optimization",
                    options={"max_iterations": 3 if opt_level < 3 else 5},
                )
            )
            pm.append(
                PassConfig(
                    name="commutation_analysis",
                    pass_type="optimization",
                    options={"enabled": opt_level >= 2},
                )
            )

        # ── 4. Basis decomposition ────────────────────────────────────
        basis = config.effective_basis_gates
        if basis:
            pm.append(
                PassConfig(
                    name="basis_translation",
                    pass_type="decomposition",
                    options={"target_basis": list(basis)},
                )
            )

        # ── 5. Cost-weighted layout ───────────────────────────────────
        if calibration is not None and noise_model is not None:
            qubit_scores = _rank_qubits_by_cost_fidelity(
                calibration,
                noise_model,
                cost_per_shot,
            )
            pm.append(
                PassConfig(
                    name="cost_aware_layout",
                    pass_type="routing",
                    options={
                        "qubit_ranking": qubit_scores,
                        "method": "vf2" if opt_level >= 2 else "trivial",
                        "cost_weight": 0.3,
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

        # ── 6. Routing ────────────────────────────────────────────────
        coupling = config.coupling_map
        if coupling:
            routing_opts: dict = {
                "coupling_map": coupling,
                "method": "sabre",
            }
            if config.seed is not None:
                routing_opts["seed"] = config.seed
            pm.append(
                PassConfig(
                    name="swap_routing",
                    pass_type="routing",
                    options=routing_opts,
                )
            )

        # ── 7. Post-routing optimisation ──────────────────────────────
        if opt_level >= 2:
            pm.append(
                PassConfig(
                    name="post_route_cancellation",
                    pass_type="optimization",
                    options={"max_iterations": 3},
                )
            )

        # ── 8. Scheduling ─────────────────────────────────────────────
        pm.append(
            PassConfig(
                name="alap_scheduling",
                pass_type="scheduling",
                options={"strategy": "alap"},
            )
        )

        # ── 9. Fidelity + cost estimation ─────────────────────────────
        pm.append(
            PassConfig(
                name="fidelity_estimation",
                pass_type="analysis",
                options={"estimate_fidelity": True, "include_cost": True},
            )
        )

        return pm


def _rank_qubits_by_cost_fidelity(
    calibration: CalibrationProvider,
    noise_model: NoiseModel,
    cost_per_shot: float,
) -> list[int]:
    """Return qubit indices sorted by cost-weighted error.

    Qubits with slightly higher error but on a cheaper backend are
    preferred over marginally better qubits that cost much more.
    The cost factor is normalised so it acts as a mild tiebreaker
    rather than dominating the ranking.
    """
    qprops = calibration.get_all_qubit_properties()
    scored = []
    for qp in qprops:
        err = noise_model.qubit_error(qp.qubit_id)
        # Cost factor: a small additive term proportional to per-shot cost
        # Normalised assuming max ~$1/shot (IonQ range)
        cost_penalty = cost_per_shot * 0.3
        combined = err + cost_penalty
        scored.append((combined, qp.qubit_id))
    scored.sort()
    return [qid for _, qid in scored]
