"""Depth-optimal compilation strategy.

Builds a pass pipeline that aggressively minimises circuit depth via
gate cancellation, commutation analysis, and circuit simplification.
Focuses on reducing depth rather than maximising fidelity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qb_compiler.strategies.base import CompilationStrategy, PassConfig, PassManager

if TYPE_CHECKING:
    from qb_compiler.calibration.provider import CalibrationProvider
    from qb_compiler.config import CompilerConfig
    from qb_compiler.noise.noise_model import NoiseModel


class DepthOptimalStrategy(CompilationStrategy):
    """Compilation strategy that minimises circuit depth.

    Uses aggressive gate cancellation, commutation-based reordering,
    and circuit simplification to produce the shallowest possible
    circuit.  This is useful when decoherence is the dominant error
    source and depth correlates directly with error.
    """

    @property
    def name(self) -> str:
        return "depth_optimal"

    def build_pass_manager(
        self,
        config: CompilerConfig,
        calibration: CalibrationProvider | None = None,
        noise_model: NoiseModel | None = None,
    ) -> PassManager:
        pm = PassManager()

        # ── 1. Initial analysis ───────────────────────────────────────
        pm.append(PassConfig(
            name="circuit_analysis",
            pass_type="analysis",
            options={"collect_gate_counts": True, "collect_depth": True},
        ))

        # ── 2. Aggressive gate cancellation (pre-routing) ────────────
        pm.append(PassConfig(
            name="gate_cancellation",
            pass_type="optimization",
            options={"max_iterations": 8},
        ))

        # ── 3. Commutation analysis for depth reduction ───────────────
        pm.append(PassConfig(
            name="commutation_analysis",
            pass_type="optimization",
            options={"enabled": True, "objective": "minimize_depth"},
        ))

        # ── 4. Circuit simplification ─────────────────────────────────
        pm.append(PassConfig(
            name="circuit_simplification",
            pass_type="optimization",
            options={"merge_single_qubit_gates": True, "remove_identity": True},
        ))

        # ── 5. Basis decomposition ────────────────────────────────────
        basis = config.effective_basis_gates
        if basis:
            pm.append(PassConfig(
                name="basis_translation",
                pass_type="decomposition",
                options={"target_basis": list(basis)},
            ))

        # ── 6. Layout ─────────────────────────────────────────────────
        pm.append(PassConfig(
            name="trivial_layout",
            pass_type="routing",
            options={},
        ))

        # ── 7. Routing ────────────────────────────────────────────────
        coupling = config.coupling_map
        if coupling:
            routing_opts: dict = {
                "coupling_map": coupling,
                "method": "sabre",
            }
            if config.seed is not None:
                routing_opts["seed"] = config.seed
            pm.append(PassConfig(
                name="swap_routing",
                pass_type="routing",
                options=routing_opts,
            ))

        # ── 8. Post-routing depth optimisation ────────────────────────
        pm.append(PassConfig(
            name="post_route_cancellation",
            pass_type="optimization",
            options={"max_iterations": 5},
        ))
        pm.append(PassConfig(
            name="commutation_analysis",
            pass_type="optimization",
            options={"enabled": True, "objective": "minimize_depth"},
        ))

        # ── 9. ASAP scheduling for minimum depth ─────────────────────
        pm.append(PassConfig(
            name="asap_scheduling",
            pass_type="scheduling",
            options={"strategy": "asap"},
        ))

        # ── 10. Final depth analysis ──────────────────────────────────
        pm.append(PassConfig(
            name="depth_analysis",
            pass_type="analysis",
            options={"collect_depth": True},
        ))

        return pm
