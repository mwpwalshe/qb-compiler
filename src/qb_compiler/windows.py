"""Fidelity-per-dollar ranking with a naive calibration trend.

Answers two questions before any QPU money is spent: **where** is a
circuit cheapest per unit of predicted fidelity, and is that backend's
recent calibration **trending** better or worse.

The trend is deliberately naive: a least-squares slope through the
median two-qubit gate error of the most recent public calibration
snapshots on disk.  It is a *signal only*.  This module does NOT
forecast future error rates and does NOT schedule jobs; both are
deliberately out of scope for this package.

Usage::

    from qb_compiler.windows import rank_value, format_table

    rows = rank_value(circuit)
    print(format_table(rows))
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SNAPSHOT_RE_TEMPLATE = r"^{backend}_(\d{{4}})_(\d{{2}})_(\d{{2}})\.json$"
_BACKEND_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

# Relative change of the fitted line over the window, as a fraction of
# the mean median error, below which the trend is reported as stable.
_STABLE_BAND = 0.10


@dataclass(frozen=True, slots=True)
class BackendValue:
    """Value assessment of a circuit on a single backend.

    Attributes
    ----------
    backend :
        Backend identifier (e.g. ``"ibm_fez"``).
    predicted_fidelity :
        Multiplicative fidelity estimate from :func:`check_viability`.
    cost_per_run_usd :
        Estimated job cost in USD for the requested shot count, or
        ``None`` when no pricing data exists.
    fidelity_per_dollar :
        ``predicted_fidelity / cost_per_run_usd`` when the cost is
        known and positive, else ``None``.
    trend :
        Naive calibration trend class: ``"degrading"``, ``"stable"``,
        ``"improving"``, or ``"unknown"``.
    trend_detail :
        Human-readable sentence with the numbers behind the trend.
    notes :
        Per-backend caveats (missing pricing, missing snapshots, ...).
    """

    backend: str
    predicted_fidelity: float
    cost_per_run_usd: float | None
    fidelity_per_dollar: float | None
    trend: str
    trend_detail: str
    notes: list[str] = field(default_factory=list)


def _bundled_snapshot_dir() -> Path:
    return Path(__file__).resolve().parent / "calibration" / "snapshots"


def _snapshot_dirs() -> list[str]:
    """Calibration snapshot directories, in the same priority order as
    :func:`qb_compiler.compiler._load_calibration_fixture`."""
    dirs: list[str] = []
    env_dir = os.environ.get("QBC_CALIBRATION_DIR")
    if env_dir and os.path.isdir(env_dir):
        dirs.append(env_dir)
    dirs.append(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "tests",
            "fixtures",
            "calibration_snapshots",
        ),
    )
    bundled = str(_bundled_snapshot_dir())
    if os.path.isdir(bundled):
        dirs.append(bundled)
    return dirs


def _dated_snapshots(backend: str) -> list[tuple[date, str]]:
    """Dated snapshot files for *backend*, sorted oldest first.

    Filenames must match ``<backend>_YYYY_MM_DD.json``.  When the same
    date appears in more than one directory, the earlier (higher
    priority) directory wins.
    """
    pattern = re.compile(_SNAPSHOT_RE_TEMPLATE.format(backend=re.escape(backend)))
    found: dict[date, str] = {}
    for snap_dir in _snapshot_dirs():
        if not os.path.isdir(snap_dir):
            continue
        for fname in os.listdir(snap_dir):
            m = pattern.match(fname)
            if m is None:
                continue
            try:
                when = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                continue
            found.setdefault(when, os.path.join(snap_dir, fname))
    return sorted(found.items())


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _median_2q_error(props: Any) -> float | None:
    """Median two-qubit gate error from a BackendProperties snapshot."""
    errors = [
        gp.error_rate
        for gp in props.gate_properties
        if len(gp.qubits) == 2 and gp.error_rate is not None
    ]
    if not errors:
        return None
    return _median(errors)


def _least_squares_slope(xs: list[float], ys: list[float]) -> float:
    """Slope of the ordinary least-squares line through (xs, ys)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    return sxy / sxx


def calibration_trend(backend: str, *, window: int = 5) -> tuple[str, str]:
    """Naive calibration trend for *backend* from on-disk snapshots.

    Lists dated calibration snapshot files (``<backend>_YYYY_MM_DD.json``)
    in the fixture directories used by the compiler, takes up to
    *window* most recent, computes the median two-qubit gate error per
    snapshot, and fits a least-squares slope versus days.

    Classification: when the fitted change over the window is within
    10 percent of the mean median error the trend is ``"stable"``;
    a larger positive change is ``"degrading"`` and a larger negative
    change is ``"improving"``.  Fewer than two usable snapshots gives
    ``"unknown"``.

    This is a naive slope on public calibration data, a signal only.
    No forecasting, no scheduling.

    Returns
    -------
    tuple[str, str]
        ``(trend_class, trend_detail)``.
    """
    if not _BACKEND_NAME_RE.match(backend):
        return "unknown", f"Invalid backend name {backend!r}; no snapshots searched."

    snapshots = _dated_snapshots(backend)[-window:]
    if len(snapshots) < 2:
        return (
            "unknown",
            f"Found {len(snapshots)} dated calibration snapshot(s) for {backend}; "
            "at least 2 are needed to estimate a trend.",
        )

    from qb_compiler.calibration.models.backend_properties import BackendProperties

    points: list[tuple[date, float]] = []
    for when, path in snapshots:
        try:
            props = BackendProperties.from_qubitboost_json(path)
        except Exception as e:
            logger.debug("Failed to load snapshot %s: %s", path, e)
            continue
        med = _median_2q_error(props)
        if med is not None:
            points.append((when, med))

    if len(points) < 2:
        return (
            "unknown",
            f"Only {len(points)} snapshot(s) for {backend} had usable two-qubit "
            "gate errors; at least 2 are needed to estimate a trend.",
        )

    origin = points[0][0]
    xs = [float((when - origin).days) for when, _ in points]
    ys = [med for _, med in points]
    span_days = xs[-1] - xs[0]
    mean_err = sum(ys) / len(ys)
    slope = _least_squares_slope(xs, ys)
    change = slope * span_days
    relative = change / mean_err if mean_err > 0 else 0.0

    if abs(relative) < _STABLE_BAND:
        trend = "stable"
    elif relative > 0:
        trend = "degrading"
    else:
        trend = "improving"

    detail = (
        f"Median two-qubit error moved from {ys[0]:.5f} to {ys[-1]:.5f} across "
        f"{len(points)} snapshots over {span_days:.0f} days; fitted change "
        f"{change:+.5f} ({100 * relative:+.1f} percent of the mean), so {trend}. "
        "Naive slope on public calibration data, signal only."
    )
    return trend, detail


def _default_backends() -> list[str]:
    """IBM backends from BACKEND_CONFIGS that have a loadable calibration fixture."""
    from qb_compiler.compiler import _load_calibration_fixture
    from qb_compiler.config import BACKEND_CONFIGS

    names: list[str] = []
    for name, spec in BACKEND_CONFIGS.items():
        if spec.provider != "ibm":
            continue
        try:
            props = _load_calibration_fixture(name)
        except Exception as e:
            logger.debug("Calibration probe failed for %s: %s", name, e)
            continue
        if props is not None:
            names.append(name)
    return names


def rank_value(
    circuit: Any,
    backends: list[str] | None = None,
    *,
    shots: int = 4096,
    n_seeds: int = 2,
) -> list[BackendValue]:
    """Rank backends by fidelity per dollar for *circuit*.

    Parameters
    ----------
    circuit :
        A Qiskit ``QuantumCircuit``.
    backends :
        Backend names to assess.  Defaults to the IBM backends in
        :data:`qb_compiler.config.BACKEND_CONFIGS` that have a loadable
        calibration fixture.
    shots :
        Shot count used for the cost estimate.
    n_seeds :
        Transpiler seeds per backend (passed to ``check_viability``).

    Returns
    -------
    list[BackendValue]
        Sorted by ``fidelity_per_dollar`` descending, ``None`` last.
        Backends that fail to assess are skipped (logged), never raised.
    """
    from qb_compiler.viability import check_viability

    if backends is None:
        backends = _default_backends()

    rows: list[BackendValue] = []
    for name in backends:
        notes: list[str] = []
        try:
            result = check_viability(circuit, backend=name, n_seeds=n_seeds, shots=shots)
        except Exception as e:
            logger.warning("Skipping %s: viability check failed (%s)", name, e)
            continue
        predicted = result.estimated_fidelity

        cost: float | None = None
        with contextlib.suppress(Exception):
            from qb_compiler.cost.pricing import get_pricing

            pricing = get_pricing(name)
            if pricing is not None:
                cost = pricing.job_cost(shots)
        if cost is None:
            notes.append("No pricing data; cannot compute fidelity per dollar.")

        fpd = predicted / cost if cost is not None and cost > 0 else None

        try:
            trend, trend_detail = calibration_trend(name)
        except Exception as e:
            logger.debug("Trend computation failed for %s: %s", name, e)
            trend, trend_detail = "unknown", f"Trend computation failed for {name}."
        if trend == "unknown":
            notes.append("Calibration trend unavailable; see trend_detail.")

        rows.append(
            BackendValue(
                backend=name,
                predicted_fidelity=predicted,
                cost_per_run_usd=round(cost, 6) if cost is not None else None,
                fidelity_per_dollar=round(fpd, 4) if fpd is not None else None,
                trend=trend,
                trend_detail=trend_detail,
                notes=notes,
            )
        )

    rows.sort(
        key=lambda r: (
            r.fidelity_per_dollar is None,
            -(r.fidelity_per_dollar or 0.0),
        )
    )
    return rows


def _no_backends_hint() -> str:
    return (
        "No backends assessed. Point QBC_CALIBRATION_DIR at a directory of calibration "
        "snapshots named <backend>_YYYY_MM_DD.json, or pass backends explicitly; a small "
        "bundled snapshot set ships with the package for ibm_fez and ibm_torino."
    )


def format_table(rows: list[BackendValue]) -> str:
    """Render *rows* as an aligned text table for CLI use."""
    if not rows:
        return "No backends assessed."

    headers = ("Backend", "Pred.Fid", "Cost USD", "Fid/$", "Trend")
    cells: list[tuple[str, str, str, str, str]] = []
    for r in rows:
        cells.append(
            (
                r.backend,
                f"{r.predicted_fidelity:.4f}",
                f"{r.cost_per_run_usd:.4f}" if r.cost_per_run_usd is not None else "N/A",
                f"{r.fidelity_per_dollar:.3g}" if r.fidelity_per_dollar is not None else "N/A",
                r.trend,
            )
        )

    widths = [max(len(headers[i]), *(len(row[i]) for row in cells)) for i in range(len(headers))]
    hdr = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "-" * len(hdr)
    lines = [sep, hdr, sep]
    for row in cells:
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    lines.append(sep)

    for r in rows:
        for note in r.notes:
            lines.append(f"{r.backend}: {note}")
    return "\n".join(lines)
