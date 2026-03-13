"""Budget-aware compilation strategy.

Compiles within a cost constraint: if estimated cost exceeds the
budget, reduces shots or suggests a cheaper backend.  Raises
:class:`BudgetExceededError` if it is impossible to stay within budget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qb_compiler.cost.pricing import cost_per_shot
from qb_compiler.exceptions import BudgetExceededError
from qb_compiler.strategies.base import CompilationStrategy, PassConfig, PassManager

if TYPE_CHECKING:
    from qb_compiler.calibration.provider import CalibrationProvider
    from qb_compiler.config import CompilerConfig
    from qb_compiler.noise.noise_model import NoiseModel


class BudgetAwareStrategy(CompilationStrategy):
    """Compilation strategy that respects a USD budget constraint.

    Builds a fidelity-focused pass pipeline, but first validates that
    the target backend and requested shots fit within the budget.  If
    the cost exceeds the budget, the strategy attempts to reduce shots
    to fit.  If even a single shot exceeds the budget, a
    :class:`BudgetExceededError` is raised.

    Parameters
    ----------
    budget_usd:
        Maximum allowed execution cost in US dollars.
    shots:
        Requested number of shots.  Defaults to 1024.
    """

    def __init__(self, budget_usd: float, shots: int = 1024) -> None:
        if budget_usd <= 0:
            raise ValueError(f"budget_usd must be positive, got {budget_usd}")
        self._budget_usd = budget_usd
        self._requested_shots = shots
        self._effective_shots = shots

    @property
    def name(self) -> str:
        return "budget_aware"

    @property
    def budget_usd(self) -> float:
        """The configured budget cap."""
        return self._budget_usd

    @property
    def effective_shots(self) -> int:
        """Number of shots after budget adjustment."""
        return self._effective_shots

    def build_pass_manager(
        self,
        config: CompilerConfig,
        calibration: CalibrationProvider | None = None,
        noise_model: NoiseModel | None = None,
    ) -> PassManager:
        # ── Budget validation ─────────────────────────────────────────
        if config.backend is not None:
            cps = self._resolve_cost_per_shot(config.backend)
            estimated_cost = cps * self._requested_shots

            if estimated_cost > self._budget_usd:
                # Try to reduce shots to fit within budget
                max_affordable = int(self._budget_usd / cps) if cps > 0 else self._requested_shots
                if max_affordable < 1:
                    raise BudgetExceededError(
                        estimated_usd=cps,
                        budget_usd=self._budget_usd,
                        shots=1,
                    )
                self._effective_shots = max_affordable
            else:
                self._effective_shots = self._requested_shots

        # ── Build pass pipeline (similar to fidelity_optimal) ─────────
        pm = PassManager()
        opt_level = config.optimization_level

        # Analysis
        pm.append(PassConfig(
            name="circuit_analysis",
            pass_type="analysis",
            options={"collect_gate_counts": True, "collect_depth": True},
        ))

        # Budget constraint pass
        pm.append(PassConfig(
            name="budget_constraint",
            pass_type="analysis",
            options={
                "budget_usd": self._budget_usd,
                "effective_shots": self._effective_shots,
                "backend": config.backend,
            },
        ))

        # Pre-routing optimisation
        if opt_level >= 1:
            pm.append(PassConfig(
                name="gate_cancellation",
                pass_type="optimization",
                options={"max_iterations": 2 if opt_level < 3 else 5},
            ))

        # Basis decomposition
        basis = config.effective_basis_gates
        if basis:
            pm.append(PassConfig(
                name="basis_translation",
                pass_type="decomposition",
                options={"target_basis": list(basis)},
            ))

        # Layout
        if calibration is not None and noise_model is not None:
            pm.append(PassConfig(
                name="noise_aware_layout",
                pass_type="routing",
                options={"method": "vf2" if opt_level >= 2 else "trivial"},
            ))
        else:
            pm.append(PassConfig(
                name="trivial_layout",
                pass_type="routing",
                options={},
            ))

        # Routing
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

        # Post-routing optimisation
        if opt_level >= 2:
            pm.append(PassConfig(
                name="post_route_cancellation",
                pass_type="optimization",
                options={"max_iterations": 3},
            ))

        # Scheduling
        pm.append(PassConfig(
            name="alap_scheduling",
            pass_type="scheduling",
            options={"strategy": "alap"},
        ))

        return pm

    @staticmethod
    def _resolve_cost_per_shot(backend: str) -> float:
        """Look up cost per shot, returning 0.0 for unknown backends."""
        try:
            return cost_per_shot(backend)
        except KeyError:
            return 0.0
