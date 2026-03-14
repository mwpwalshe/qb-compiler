"""Backend recommendation engine.

Answers the question no other tool answers: **which backend should I
use for this circuit TODAY?**

Analyzes a circuit across all configured backends and returns a ranked
recommendation based on estimated fidelity, cost, and viability.

Usage::

    from qb_compiler.recommender import BackendRecommender

    rec = BackendRecommender()
    rec.add_backend("ibm_fez", calibration="path/to/fez.json")
    rec.add_backend("ibm_torino", calibration="path/to/torino.json")

    report = rec.analyze(circuit)
    print(report)
    print(report.recommendation)
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BackendAnalysis:
    """Analysis of a circuit on a single backend."""

    backend: str
    provider: str
    estimated_fidelity: float
    two_qubit_gate_count: int
    depth: int
    viable: bool
    status: str  # VIABLE / MARGINAL / NOT VIABLE
    cost_per_4096_shots: float | None
    fidelity_per_dollar: float | None
    best_seed: int
    physical_qubits: list[int]
    viable_depth: int
    analysis_time_ms: float


@dataclass
class RecommendationReport:
    """Full recommendation across all analyzed backends.

    Attributes
    ----------
    circuit_name :
        Name of the analyzed circuit.
    n_qubits :
        Number of qubits in the circuit.
    analyses :
        Per-backend analysis results, sorted by fidelity (best first).
    best_fidelity :
        Backend with highest estimated fidelity.
    best_value :
        Backend with best fidelity-per-dollar ratio.
    warnings :
        Global warnings (e.g. no viable backends found).
    total_analysis_time_ms :
        Wall-clock time for the full analysis.
    """

    circuit_name: str
    n_qubits: int
    analyses: list[BackendAnalysis]
    best_fidelity: str | None
    best_value: str | None
    warnings: list[str] = field(default_factory=list)
    total_analysis_time_ms: float = 0.0

    @property
    def recommendation(self) -> str:
        """One-line recommendation."""
        if not self.analyses:
            return "No backends configured."
        if self.best_fidelity is None:
            return "No viable backends found for this circuit."
        best = next(a for a in self.analyses if a.backend == self.best_fidelity)
        line = f"Recommendation: {self.best_fidelity} (fidelity={best.estimated_fidelity:.3f})"
        if self.best_value and self.best_value != self.best_fidelity:
            val = next(a for a in self.analyses if a.backend == self.best_value)
            line += f"\nBudget pick: {self.best_value} (fidelity={val.estimated_fidelity:.3f}"
            if val.cost_per_4096_shots is not None:
                line += f", ${val.cost_per_4096_shots:.2f}/4096 shots"
            line += ")"
        return line

    def __str__(self) -> str:
        if not self.analyses:
            return "No backends configured."

        lines = [
            f"Backend Recommendation Report — {self.circuit_name} ({self.n_qubits}q)",
            "",
        ]

        # Table header
        hdr = (
            f"{'Backend':>14} | {'Est.Fid':>8} | {'2Q Gates':>8} | "
            f"{'Depth':>6} | {'Status':>11} | {'Cost/4096':>10} | {'Fid/$':>8}"
        )
        sep = "-" * len(hdr)
        lines.append(sep)
        lines.append(hdr)
        lines.append(sep)

        for a in self.analyses:
            cost_str = f"${a.cost_per_4096_shots:.4f}" if a.cost_per_4096_shots else "N/A"
            fpd_str = f"{a.fidelity_per_dollar:.1f}" if a.fidelity_per_dollar else "N/A"
            marker = ""
            if a.backend == self.best_fidelity:
                marker = " <-- BEST"
            elif a.backend == self.best_value:
                marker = " <-- VALUE"
            lines.append(
                f"{a.backend:>14} | {a.estimated_fidelity:>8.4f} | "
                f"{a.two_qubit_gate_count:>8} | {a.depth:>6} | "
                f"{a.status:>11} | {cost_str:>10} | {fpd_str:>8}{marker}"
            )

        lines.append(sep)
        lines.append("")
        lines.append(self.recommendation)

        for w in self.warnings:
            lines.append(f"WARNING: {w}")

        lines.append(f"\nAnalysis time: {self.total_analysis_time_ms:.0f}ms")
        return "\n".join(lines)


class BackendRecommender:
    """Recommends the best backend for a given circuit.

    Parameters
    ----------
    n_seeds :
        Number of Qiskit transpiler seeds per backend (default 10).
    shots :
        Shot count for cost estimation (default 4096).
    """

    def __init__(self, *, n_seeds: int = 10, shots: int = 4096) -> None:
        self._backends: dict[str, _BackendEntry] = {}
        self._n_seeds = n_seeds
        self._shots = shots

    def add_backend(
        self,
        name: str,
        *,
        calibration: str | Any | None = None,
        qiskit_target: Any | None = None,
    ) -> BackendRecommender:
        """Register a backend for analysis.

        Parameters
        ----------
        name :
            Backend identifier (e.g. ``"ibm_fez"``).
        calibration :
            Path to calibration JSON, or a :class:`BackendProperties`
            instance.  If ``None``, tries to load from fixtures.
        qiskit_target :
            Qiskit ``Target``.  If ``None``, one is built from
            calibration data.

        Returns
        -------
        BackendRecommender
            Self, for chaining.
        """
        props = None
        if isinstance(calibration, str):
            from qb_compiler.calibration.models.backend_properties import (
                BackendProperties,
            )
            props = BackendProperties.from_qubitboost_json(calibration)
        elif calibration is not None:
            props = calibration

        if props is None:
            from qb_compiler.compiler import _load_calibration_fixture
            props = _load_calibration_fixture(name)

        self._backends[name] = _BackendEntry(
            name=name, props=props, qiskit_target=qiskit_target,
        )
        return self

    def add_backend_live(
        self, name: str, qiskit_backend: Any,
    ) -> BackendRecommender:
        """Register a live IBM backend (uses ``backend.target`` directly).

        Parameters
        ----------
        name :
            Backend identifier.
        qiskit_backend :
            A Qiskit ``IBMBackend`` instance with live calibration.
        """
        self._backends[name] = _BackendEntry(
            name=name,
            props=None,
            qiskit_target=qiskit_backend.target,
        )
        return self

    def analyze(self, circuit: Any) -> RecommendationReport:
        """Analyze *circuit* across all configured backends.

        Parameters
        ----------
        circuit :
            A Qiskit ``QuantumCircuit``.

        Returns
        -------
        RecommendationReport
            Ranked recommendation with per-backend analysis.
        """

        t0 = time.perf_counter()
        analyses: list[BackendAnalysis] = []
        warnings: list[str] = []

        for name, entry in self._backends.items():
            try:
                analysis = self._analyze_one(circuit, entry)
                analyses.append(analysis)
            except Exception as e:
                logger.warning("Failed to analyze %s: %s", name, e)
                warnings.append(f"{name}: analysis failed ({e})")

        # Sort by fidelity (best first)
        analyses.sort(key=lambda a: -a.estimated_fidelity)

        # Find best fidelity (must be viable)
        viable = [a for a in analyses if a.viable]
        best_fidelity = viable[0].backend if viable else None

        # Find best value (fidelity per dollar, must be viable)
        viable_with_cost = [a for a in viable if a.fidelity_per_dollar is not None]
        best_value = None
        if viable_with_cost:
            best_value = max(viable_with_cost, key=lambda a: a.fidelity_per_dollar).backend

        if not viable:
            warnings.append(
                "No backends produce viable results for this circuit. "
                "Consider reducing circuit depth or qubit count."
            )

        # Warn about low-fidelity backends
        for a in analyses:
            if a.status == "NOT VIABLE":
                warnings.append(
                    f"{a.backend}: fidelity {a.estimated_fidelity:.4f} is below "
                    f"noise floor — results will be random noise."
                )

        total_ms = (time.perf_counter() - t0) * 1000

        circuit_name = getattr(circuit, "name", None) or f"{circuit.num_qubits}q circuit"

        return RecommendationReport(
            circuit_name=circuit_name,
            n_qubits=circuit.num_qubits,
            analyses=analyses,
            best_fidelity=best_fidelity,
            best_value=best_value,
            warnings=warnings,
            total_analysis_time_ms=round(total_ms, 1),
        )

    def _analyze_one(
        self, circuit: Any, entry: _BackendEntry,
    ) -> BackendAnalysis:
        """Analyze circuit on a single backend."""
        from qiskit import transpile

        from qb_compiler.config import get_backend_spec
        from qb_compiler.cost.pricing import get_pricing
        from qb_compiler.viability import (
            _build_target_from_props,
            _count_2q,
            _estimate_routed_fidelity,
            _estimate_viable_depth,
        )

        t0 = time.perf_counter()

        # Resolve target
        target = entry.qiskit_target
        if target is None and entry.props is not None:
            target = _build_target_from_props(entry.props)

        if target is None:
            raise ValueError(f"No target or calibration for {entry.name}")

        # Get spec for median errors
        spec = None
        with contextlib.suppress(Exception):
            spec = get_backend_spec(entry.name)
        median_2q = spec.median_cx_error if spec else 0.005
        median_ro = spec.median_readout_error if spec else 0.01

        # Transpile with N seeds
        best_tc = None
        best_2q = float("inf")
        best_seed = -1

        for seed in range(self._n_seeds):
            tc = transpile(
                circuit, target=target,
                optimization_level=3, seed_transpiler=seed,
            )
            c2q = _count_2q(tc)
            if c2q < best_2q:
                best_2q = c2q
                best_tc = tc
                best_seed = seed

        depth = best_tc.depth()
        n_qubits = circuit.num_qubits

        # Fidelity
        fidelity = _estimate_routed_fidelity(
            best_tc, entry.props, median_2q, median_ro,
        )

        # Viability
        noise_floor = 1.0 / (2 ** n_qubits)
        snr = fidelity / noise_floor if noise_floor > 0 else float("inf")
        viable_depth = _estimate_viable_depth(median_2q, median_ro, n_qubits)

        if snr >= 2.0 and fidelity >= 0.01:
            viable = True
            status = "VIABLE" if fidelity >= 0.3 else "MARGINAL"
        else:
            viable = False
            status = "NOT VIABLE"

        # Cost
        pricing = get_pricing(entry.name)
        cost_4096 = pricing.cost_per_shot_usd * self._shots if pricing else None
        fpd = fidelity / cost_4096 if cost_4096 and cost_4096 > 0 else None

        # Physical qubits
        ql = best_tc.layout
        if ql and ql.initial_layout:
            phys = [
                ql.initial_layout[circuit.qubits[i]]
                for i in range(n_qubits)
            ]
        else:
            phys = list(range(n_qubits))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        provider = spec.provider if spec else "unknown"

        return BackendAnalysis(
            backend=entry.name,
            provider=provider,
            estimated_fidelity=round(fidelity, 6),
            two_qubit_gate_count=best_2q,
            depth=depth,
            viable=viable,
            status=status,
            cost_per_4096_shots=round(cost_4096, 4) if cost_4096 else None,
            fidelity_per_dollar=round(fpd, 1) if fpd else None,
            best_seed=best_seed,
            physical_qubits=phys,
            viable_depth=viable_depth,
            analysis_time_ms=round(elapsed_ms, 1),
        )


@dataclass
class _BackendEntry:
    """Internal: tracks a registered backend."""

    name: str
    props: Any  # BackendProperties | None
    qiskit_target: Any  # Qiskit Target | None
