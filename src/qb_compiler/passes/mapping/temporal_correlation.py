"""Temporal correlation analysis from public calibration snapshots.

Detects qubit pairs whose error rates co-vary across calibration cycles.
When errors are correlated, the independent-error assumption used by QEC
codes breaks down, reducing effective code distance.

This module provides a "free sample" of QubitBoost's SafetyGate temporal
correlation monitoring.  SafetyGate tracks correlations in real time with
30-60 minute resolution; this module works with any number of static
calibration snapshots (minimum 2).

Usage::

    from qb_compiler.passes.mapping.temporal_correlation import (
        TemporalCorrelationAnalyzer,
    )

    analyzer = TemporalCorrelationAnalyzer.from_snapshots([snap_feb, snap_mar])

    # Per-qubit volatility (how much does this qubit's error change?)
    vol = analyzer.qubit_volatility(42)

    # Per-edge correlation (do these two qubits' errors move together?)
    corr = analyzer.edge_correlation(42, 43)

    # Feed into CalibrationMapper
    mapper = CalibrationMapper(
        calibration=latest_snapshot,
        config=CalibrationMapperConfig(correlation_weight=2.0),
        correlation_analyzer=analyzer,
    )
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qb_compiler.calibration.models.backend_properties import BackendProperties

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorrelationResult:
    """Temporal correlation analysis results for a device.

    Attributes
    ----------
    qubit_volatility:
        Per-qubit error volatility (standard deviation of readout error
        across snapshots).  Higher = less stable qubit.
    edge_correlation:
        Per-edge Pearson-like correlation coefficient.  Maps
        ``(min(q0,q1), max(q0,q1))`` to a score in ``[-1, 1]``.
        +1 = errors move together, -1 = opposite, 0 = uncorrelated.
    qubit_drift:
        Per-qubit directional drift.  Positive = getting worse,
        negative = improving.
    n_snapshots:
        Number of calibration snapshots used in the analysis.
    """

    qubit_volatility: dict[int, float] = field(default_factory=dict)
    edge_correlation: dict[tuple[int, int], float] = field(default_factory=dict)
    qubit_drift: dict[int, float] = field(default_factory=dict)
    n_snapshots: int = 0


class TemporalCorrelationAnalyzer:
    """Detect temporal error correlations from multiple calibration snapshots.

    Works with public calibration data only — no proprietary API needed.
    For real-time correlation monitoring, see ``LiveCalibrationProvider``
    with QubitBoost SafetyGate (requires ``qubitboost-sdk``).

    Parameters
    ----------
    snapshots:
        Two or more :class:`BackendProperties` snapshots from different
        calibration cycles.  Order does not matter (sorted by timestamp).
    """

    def __init__(self, snapshots: list[BackendProperties]) -> None:
        if len(snapshots) < 2:
            raise ValueError(
                "TemporalCorrelationAnalyzer requires at least 2 calibration "
                f"snapshots, got {len(snapshots)}"
            )
        # Sort by timestamp
        self._snapshots = sorted(snapshots, key=lambda s: s.timestamp or "")
        self._result: CorrelationResult | None = None

    @classmethod
    def from_snapshots(
        cls, snapshots: list[BackendProperties]
    ) -> TemporalCorrelationAnalyzer:
        """Convenience constructor."""
        return cls(snapshots)

    @property
    def result(self) -> CorrelationResult:
        """Lazily compute and return correlation results."""
        if self._result is None:
            self._result = self._analyze()
        return self._result

    def qubit_volatility(self, qubit_id: int) -> float:
        """Error volatility for a specific qubit (0 = perfectly stable)."""
        return self.result.qubit_volatility.get(qubit_id, 0.0)

    def edge_correlation(self, q0: int, q1: int) -> float:
        """Correlation coefficient for a qubit pair (-1 to +1)."""
        key = (min(q0, q1), max(q0, q1))
        return self.result.edge_correlation.get(key, 0.0)

    def qubit_drift(self, qubit_id: int) -> float:
        """Directional drift for a qubit (positive = getting worse)."""
        return self.result.qubit_drift.get(qubit_id, 0.0)

    def _analyze(self) -> CorrelationResult:
        """Run the full temporal correlation analysis."""
        # Collect per-qubit readout error time series
        qubit_series: dict[int, list[float]] = {}
        for snap in self._snapshots:
            for qp in snap.qubit_properties:
                ro = qp.readout_error
                if ro is not None:
                    qubit_series.setdefault(qp.qubit_id, []).append(ro)

        # Collect per-edge gate error time series
        edge_series: dict[tuple[int, int], list[float]] = {}
        for snap in self._snapshots:
            for gp in snap.gate_properties:
                if len(gp.qubits) == 2 and gp.error_rate is not None:
                    key = (min(gp.qubits), max(gp.qubits))
                    edge_series.setdefault(key, []).append(gp.error_rate)

        n_snaps = len(self._snapshots)

        # 1. Per-qubit volatility (std dev of readout error)
        qubit_vol: dict[int, float] = {}
        for qid, series in qubit_series.items():
            if len(series) >= 2:
                mean = sum(series) / len(series)
                variance = sum((x - mean) ** 2 for x in series) / len(series)
                qubit_vol[qid] = math.sqrt(variance)

        # 2. Per-qubit drift (last - first, positive = worse)
        qubit_drift: dict[int, float] = {}
        for qid, series in qubit_series.items():
            if len(series) >= 2:
                qubit_drift[qid] = series[-1] - series[0]

        # 3. Per-edge correlation
        #    For each pair of connected qubits, compute correlation
        #    between their readout error changes.
        edge_corr: dict[tuple[int, int], float] = {}

        # Get all connected qubit pairs from coupling maps
        connected_pairs: set[tuple[int, int]] = set()
        for snap in self._snapshots:
            for q0, q1 in snap.coupling_map:
                connected_pairs.add((min(q0, q1), max(q0, q1)))

        for q0, q1 in connected_pairs:
            s0 = qubit_series.get(q0, [])
            s1 = qubit_series.get(q1, [])
            min_len = min(len(s0), len(s1))
            if min_len >= 2:
                # Compute deltas (changes between consecutive snapshots)
                d0 = [s0[i + 1] - s0[i] for i in range(min_len - 1)]
                d1 = [s1[i + 1] - s1[i] for i in range(min_len - 1)]
                corr = self._pearson(d0, d1)
                if corr is not None:
                    edge_corr[(q0, q1)] = corr

        # Also add gate-error-based correlation for edges
        for edge_key, series in edge_series.items():
            if len(series) >= 2:
                # Gate error volatility contributes to edge instability
                mean = sum(series) / len(series)
                variance = sum((x - mean) ** 2 for x in series) / len(series)
                gate_vol = math.sqrt(variance)
                # If gate error is volatile, penalize the edge even if
                # the qubit readout correlation is low
                if edge_key not in edge_corr:
                    # Use gate volatility as a proxy for correlation
                    # Normalise by mean to get coefficient of variation
                    if mean > 1e-9:
                        edge_corr[edge_key] = min(gate_vol / mean, 1.0)

        logger.info(
            "TemporalCorrelationAnalyzer: %d qubits, %d edges, %d snapshots",
            len(qubit_vol),
            len(edge_corr),
            n_snaps,
        )

        return CorrelationResult(
            qubit_volatility=qubit_vol,
            edge_correlation=edge_corr,
            qubit_drift=qubit_drift,
            n_snapshots=n_snaps,
        )

    @staticmethod
    def _pearson(x: list[float], y: list[float]) -> float | None:
        """Pearson correlation coefficient between two equal-length series.

        Returns None if the series have zero variance (constant).
        """
        n = len(x)
        if n < 1:
            return None

        mx = sum(x) / n
        my = sum(y) / n

        sx = math.sqrt(sum((xi - mx) ** 2 for xi in x) / n)
        sy = math.sqrt(sum((yi - my) ** 2 for yi in y) / n)

        if sx < 1e-15 or sy < 1e-15:
            return None

        cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
        return cov / (sx * sy)
