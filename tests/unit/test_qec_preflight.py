"""Tests for the QEC memory-experiment preflight."""

from __future__ import annotations

import pytest

stim = pytest.importorskip("stim")
pytest.importorskip("pymatching")

from qb_compiler.qec_preflight import QECPreflightResult, qec_preflight  # noqa: E402


@pytest.fixture(scope="module")
def d3_result() -> QECPreflightResult:
    return qec_preflight(distance=3, rounds=3, physical_error=0.01, seed=0)


class TestQECPreflightSignals:
    def test_projected_ler_in_open_unit_interval(self, d3_result: QECPreflightResult) -> None:
        assert 0.0 < d3_result.projected_ler < 1.0

    def test_band_is_ordered_and_brackets_ler(self, d3_result: QECPreflightResult) -> None:
        lo, hi = d3_result.projected_ler_band
        assert lo < hi
        assert lo <= d3_result.projected_ler <= hi
        assert lo >= 0.0
        assert hi <= 1.0

    def test_detector_fraction_in_open_unit_interval(self, d3_result: QECPreflightResult) -> None:
        assert 0.0 < d3_result.projected_detector_fraction < 1.0

    def test_shots_for_rel_ci_monotone(self, d3_result: QECPreflightResult) -> None:
        shots = d3_result.shots_for_rel_ci
        assert set(shots) == {"0.5", "0.2", "0.1"}
        # Tighter relative width needs more shots
        assert shots["0.1"] > shots["0.2"] > shots["0.5"] > 0

    def test_str_renders_report(self, d3_result: QECPreflightResult) -> None:
        text = str(d3_result)
        assert "LER" in text
        assert "d=3" in text

    def test_simulation_caveat_always_present(self, d3_result: QECPreflightResult) -> None:
        assert any("uniform depolarizing proxy" in note for note in d3_result.notes)

    def test_metadata_round_trip(self, d3_result: QECPreflightResult) -> None:
        assert d3_result.distance == 3
        assert d3_result.rounds == 3
        assert d3_result.basis == "Z"
        assert d3_result.backend is None
        assert d3_result.physical_error_proxy == pytest.approx(0.01)


class TestQECPreflightGuards:
    def test_distance_above_guard_raises(self) -> None:
        with pytest.raises(ValueError, match="distance"):
            qec_preflight(distance=11, rounds=3, physical_error=0.01)

    def test_rounds_above_guard_raises(self) -> None:
        with pytest.raises(ValueError, match="rounds"):
            qec_preflight(distance=3, rounds=51, physical_error=0.01)

    def test_missing_noise_source_raises(self) -> None:
        with pytest.raises(ValueError, match="physical_error"):
            qec_preflight(distance=3, rounds=3)


class TestQECPreflightBackendPath:
    def test_ibm_fez_backend_derives_proxy(self) -> None:
        result = qec_preflight(distance=3, rounds=3, backend="ibm_fez", shots_sim=5_000, seed=1)
        assert result.backend == "ibm_fez"
        assert result.physical_error_proxy > 0.0
        assert any("median two-qubit gate error" in note for note in result.notes)
        assert 0.0 <= result.projected_ler <= 1.0
