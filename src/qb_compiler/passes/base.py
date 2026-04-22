"""Compiler pass infrastructure.

Every pass in qb-compiler inherits from :class:`BasePass`.  Passes are either
*analysis* (read-only, populate ``context``) or *transformation* (may modify
the circuit).  A :class:`PassManager` orchestrates an ordered sequence of
passes.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qb_compiler.ir.circuit import QBCircuit

logger = logging.getLogger(__name__)


# ── result ────────────────────────────────────────────────────────────


@dataclass
class PassResult:
    """Return value of a single pass invocation.

    Attributes
    ----------
    circuit : QBCircuit
        The (potentially modified) circuit.
    metadata : dict
        Arbitrary data produced by the pass (timings, stats, etc.).
    modified : bool
        Whether the pass actually changed the circuit.
    """

    circuit: QBCircuit
    metadata: dict = field(default_factory=dict)
    modified: bool = False


# ── base classes ──────────────────────────────────────────────────────


class BasePass(ABC):
    """Abstract base for all compiler passes."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable pass name (used in logs and metadata)."""
        ...

    @abstractmethod
    def run(self, circuit: QBCircuit, context: dict) -> PassResult:
        """Execute the pass.

        Parameters
        ----------
        circuit : QBCircuit
            Input circuit.  Transformation passes may return a different
            :class:`QBCircuit` instance in the result; analysis passes must
            return the same object.
        context : dict
            Mutable mapping shared across all passes in a :class:`PassManager`
            run.  Analysis passes write results here; transformation passes may
            read from it.

        Returns
        -------
        PassResult
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class AnalysisPass(BasePass):
    """A pass that inspects but does not modify the circuit.

    Subclasses should set ``modified=False`` in their :class:`PassResult`
    and return the *same* circuit object they received.
    """

    def run(self, circuit: QBCircuit, context: dict) -> PassResult:
        self.analyze(circuit, context)
        return PassResult(circuit=circuit, modified=False)

    @abstractmethod
    def analyze(self, circuit: QBCircuit, context: dict) -> None:
        """Perform analysis and write results into *context*."""
        ...


class TransformationPass(BasePass):
    """A pass that may modify the circuit."""

    def run(self, circuit: QBCircuit, context: dict) -> PassResult:
        return self.transform(circuit, context)

    @abstractmethod
    def transform(self, circuit: QBCircuit, context: dict) -> PassResult:
        """Transform *circuit* and return a :class:`PassResult`."""
        ...


# ── pass manager ──────────────────────────────────────────────────────


class PassManager:
    """Ordered pipeline of compiler passes.

    Usage::

        pm = PassManager([DepthAnalysis(), GateCancellationPass()])
        result = pm.run_all(circuit)
    """

    def __init__(self, passes: Sequence[BasePass] | None = None) -> None:
        self._passes: list[BasePass] = list(passes) if passes else []

    # ── mutation ──────────────────────────────────────────────────────

    def add(self, pass_: BasePass) -> PassManager:
        """Append a pass.  Returns *self* for chaining."""
        self._passes.append(pass_)
        return self

    def remove(self, name: str) -> bool:
        """Remove the first pass with the given name.  Returns ``True`` if found."""
        for i, p in enumerate(self._passes):
            if p.name == name:
                self._passes.pop(i)
                return True
        return False

    def insert(self, index: int, pass_: BasePass) -> PassManager:
        """Insert a pass at a specific position.  Returns *self*."""
        self._passes.insert(index, pass_)
        return self

    # ── execution ─────────────────────────────────────────────────────

    def run_all(self, circuit: QBCircuit, context: dict | None = None) -> PassResult:
        """Run all passes in order.

        Parameters
        ----------
        circuit : QBCircuit
            Input circuit (will not be mutated — a copy is made).
        context : dict | None
            Shared context dict.  Created if *None*.

        Returns
        -------
        PassResult
            Final result with the last circuit, combined metadata, and
            ``modified=True`` if *any* transformation pass modified the circuit.
        """
        if context is None:
            context = {}

        current = circuit.copy()
        any_modified = False
        all_metadata: dict = {"passes": []}

        for p in self._passes:
            t0 = time.perf_counter()
            result = p.run(current, context)
            elapsed = time.perf_counter() - t0

            current = result.circuit
            any_modified = any_modified or result.modified

            pass_meta = {
                "name": p.name,
                "elapsed_s": round(elapsed, 6),
                "modified": result.modified,
                **result.metadata,
            }
            all_metadata["passes"].append(pass_meta)
            logger.debug("pass %-30s  modified=%-5s  %.4fs", p.name, result.modified, elapsed)

        return PassResult(
            circuit=current,
            metadata=all_metadata,
            modified=any_modified,
        )

    # ── introspection ─────────────────────────────────────────────────

    @property
    def passes(self) -> list[BasePass]:
        return list(self._passes)

    def __len__(self) -> int:
        return len(self._passes)

    def __repr__(self) -> str:
        names = [p.name for p in self._passes]
        return f"PassManager({names})"
