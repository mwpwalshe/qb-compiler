"""Abstract base class for compilation strategies."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qb_compiler.calibration.provider import CalibrationProvider
    from qb_compiler.config import CompilerConfig
    from qb_compiler.noise.noise_model import NoiseModel


@dataclass(frozen=True, slots=True)
class PassConfig:
    """Configuration for a single compiler pass.

    Parameters
    ----------
    name:
        Human-readable pass name.
    pass_type:
        Category: ``"routing"``, ``"optimization"``, ``"scheduling"``,
        ``"decomposition"``, ``"analysis"``.
    options:
        Pass-specific keyword arguments.
    """

    name: str
    pass_type: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class PassManager:
    """Ordered sequence of compiler passes.

    This is a lightweight pass-manager representation that strategies
    populate.  The actual execution engine consumes it.
    """

    passes: list[PassConfig] = field(default_factory=list)

    def append(self, pass_cfg: PassConfig) -> None:
        """Add a pass to the end of the pipeline."""
        self.passes.append(pass_cfg)

    def prepend(self, pass_cfg: PassConfig) -> None:
        """Add a pass to the beginning of the pipeline."""
        self.passes.insert(0, pass_cfg)

    def __len__(self) -> int:
        return len(self.passes)

    def __iter__(self):
        return iter(self.passes)

    def describe(self) -> str:
        """Human-readable summary of the pass pipeline."""
        lines = [f"PassManager ({len(self.passes)} passes):"]
        for i, p in enumerate(self.passes):
            opts = f" {p.options}" if p.options else ""
            lines.append(f"  {i+1}. [{p.pass_type}] {p.name}{opts}")
        return "\n".join(lines)


class CompilationStrategy(abc.ABC):
    """Interface for compilation strategies.

    A strategy decides *which* passes to run and in *what order*, given
    the compiler configuration and (optionally) calibration / noise data.
    """

    @abc.abstractmethod
    def build_pass_manager(
        self,
        config: CompilerConfig,
        calibration: CalibrationProvider | None = None,
        noise_model: NoiseModel | None = None,
    ) -> PassManager:
        """Construct a :class:`PassManager` for the given configuration.

        Parameters
        ----------
        config:
            Compiler settings (optimization level, backend, etc.).
        calibration:
            Live calibration data, if available.
        noise_model:
            Noise model derived from calibration, if available.

        Returns
        -------
        PassManager
            Ordered list of passes to execute.
        """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Strategy identifier for logging / telemetry."""
