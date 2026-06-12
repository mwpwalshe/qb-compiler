"""Tests for fidelity-per-dollar ranking and the naive calibration trend."""

from __future__ import annotations

import glob
import os
import re

from qiskit import QuantumCircuit

from qb_compiler.windows import (
    BackendValue,
    calibration_trend,
    format_table,
    rank_value,
)

_FIXTURE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tests",
    "fixtures",
    "calibration_snapshots",
)

_TREND_CLASSES = {"degrading", "stable", "improving", "unknown"}


def _bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2, name="bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


# ── fixture sanity ───────────────────────────────────────────────


class TestFixtures:
    def test_ibm_fez_has_at_least_three_dated_snapshots(self):
        matches = [
            f
            for f in glob.glob(os.path.join(_FIXTURE_DIR, "ibm_fez_*.json"))
            if re.search(r"ibm_fez_\d{4}_\d{2}_\d{2}\.json$", f)
        ]
        assert len(matches) >= 3


# ── calibration_trend ────────────────────────────────────────────


class TestCalibrationTrend:
    def test_ibm_fez_returns_valid_class_and_numeric_detail(self):
        trend, detail = calibration_trend("ibm_fez")
        assert trend in _TREND_CLASSES
        # With >= 3 dated fixtures the trend must be computable
        assert trend != "unknown"
        assert re.search(r"\d", detail)

    def test_unknown_for_made_up_backend(self):
        trend, detail = calibration_trend("not_a_real_backend_xyz")
        assert trend == "unknown"
        assert "not_a_real_backend_xyz" in detail

    def test_invalid_backend_name_is_unknown_not_error(self):
        trend, _ = calibration_trend("../../etc/passwd")
        assert trend == "unknown"

    def test_window_one_is_unknown(self):
        trend, detail = calibration_trend("ibm_fez", window=1)
        assert trend == "unknown"
        assert "1" in detail


# ── rank_value ───────────────────────────────────────────────────


class TestRankValue:
    def test_bell_circuit_returns_sorted_consistent_rows(self):
        rows = rank_value(_bell_circuit(), n_seeds=1)
        assert len(rows) >= 1
        assert all(isinstance(r, BackendValue) for r in rows)

        # Sorted by fidelity_per_dollar descending, None last
        fpds = [r.fidelity_per_dollar for r in rows]
        known = [f for f in fpds if f is not None]
        assert known == sorted(known, reverse=True)
        if None in fpds:
            assert fpds.index(None) >= len(known)

        # fidelity_per_dollar consistent with predicted / cost
        for r in rows:
            assert 0.0 <= r.predicted_fidelity <= 1.0
            assert r.trend in _TREND_CLASSES
            if r.cost_per_run_usd is not None and r.cost_per_run_usd > 0:
                assert r.fidelity_per_dollar is not None
                expected = r.predicted_fidelity / r.cost_per_run_usd
                assert abs(r.fidelity_per_dollar - expected) <= 0.05 * expected + 1e-3

    def test_unknown_backend_never_raises(self):
        # check_viability degrades to a rough estimate for unknown backends,
        # so the row may exist, but it must carry no pricing and no trend.
        rows = rank_value(_bell_circuit(), backends=["definitely_not_a_backend"], n_seeds=1)
        for r in rows:
            assert r.cost_per_run_usd is None
            assert r.fidelity_per_dollar is None
            assert r.trend == "unknown"
            assert r.notes


# ── format_table ─────────────────────────────────────────────────


class TestFormatTable:
    def test_contains_headers_and_backend_name(self):
        rows = [
            BackendValue(
                backend="ibm_fez",
                predicted_fidelity=0.95,
                cost_per_run_usd=0.6554,
                fidelity_per_dollar=1.4495,
                trend="stable",
                trend_detail="Median two-qubit error stable across 3 snapshots.",
                notes=["example note"],
            ),
            BackendValue(
                backend="ibm_torino",
                predicted_fidelity=0.90,
                cost_per_run_usd=None,
                fidelity_per_dollar=None,
                trend="unknown",
                trend_detail="Found 1 dated calibration snapshot(s).",
            ),
        ]
        table = format_table(rows)
        for header in ("Backend", "Pred.Fid", "Cost USD", "Fid/$", "Trend"):
            assert header in table
        assert "ibm_fez" in table
        assert "ibm_torino" in table
        assert "N/A" in table
        assert "example note" in table

    def test_empty(self):
        assert "No backends" in format_table([])
