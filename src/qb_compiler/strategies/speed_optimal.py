"""Speed-optimal compilation strategy.

Builds the minimal pass pipeline for fast compilation:
just topology mapping and basic gate decomposition.
No calibration-aware passes, no noise modelling.
Designed for development, testing, and rapid iteration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qb_compiler.strategies.base import CompilationStrategy, PassConfig, PassManager

if TYPE_CHECKING:
    from qb_compiler.calibration.provider import CalibrationProvider
    from qb_compiler.config import CompilerConfig
    from qb_compiler.noise.noise_model import NoiseModel


class SpeedOptimalStrategy(CompilationStrategy):
    """Compilation strategy that minimises compilation time.

    Uses only the bare minimum passes: trivial layout, basic routing,
    and basis gate decomposition.  Calibration and noise data are
    ignored entirely — this strategy is intended for quick development
    cycles and simulator runs where compilation overhead matters more
    than output fidelity.
    """

    @property
    def name(self) -> str:
        return "speed_optimal"

    def build_pass_manager(
        self,
        config: CompilerConfig,
        calibration: CalibrationProvider | None = None,
        noise_model: NoiseModel | None = None,
    ) -> PassManager:
        pm = PassManager()

        # ── 1. Trivial layout (no calibration scoring) ────────────────
        pm.append(
            PassConfig(
                name="trivial_layout",
                pass_type="routing",
                options={},
            )
        )

        # ── 2. Basic routing if coupling map exists ───────────────────
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

        # ── 3. Basis decomposition ────────────────────────────────────
        basis = config.effective_basis_gates
        if basis:
            pm.append(
                PassConfig(
                    name="basis_translation",
                    pass_type="decomposition",
                    options={"target_basis": list(basis)},
                )
            )

        return pm
