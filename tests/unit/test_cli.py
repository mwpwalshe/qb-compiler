"""Tests for CLI commands."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from qb_compiler.cli.main import cli

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "circuits"
BELL = str(FIXTURES / "bell_state.qasm")


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


# ── qbc doctor ──────────────────────────────────────────────────────


class TestDoctor:
    def test_runs_successfully(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0
        assert "qb-compiler" in result.output
        assert "Python" in result.output

    def test_shows_qiskit(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["doctor"])
        assert "Qiskit" in result.output

    def test_shows_backends(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["doctor"])
        assert "backends configured" in result.output


# ── qbc preflight ───────────────────────────────────────────────────


class TestPreflight:
    def test_bell_state_viable(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["preflight", BELL, "-b", "ibm_fez", "-n", "1"])
        assert result.exit_code == 0
        assert "VIABLE" in result.output
        assert "fidelity" in result.output.lower()

    def test_multi_backend_rejected(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli, ["preflight", BELL, "-b", "ibm_fez", "-b", "ibm_torino"]
        )
        assert result.exit_code != 0
        assert "qubitboost.io" in result.output

    def test_shows_cost(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["preflight", BELL, "-b", "ibm_fez", "-n", "1"])
        assert "$" in result.output

    def test_shows_depth(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["preflight", BELL, "-b", "ibm_fez", "-n", "1"])
        assert "Depth" in result.output
        assert "viable limit" in result.output

    def test_shows_gate_eligibility(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["preflight", BELL, "-b", "ibm_fez", "-n", "1"])
        assert "QubitBoost gate eligibility" in result.output
        assert "TomoGate" in result.output


# ── qbc analyze ─────────────────────────────────────────────────────


class TestAnalyze:
    def test_bell_state(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["analyze", BELL, "-b", "ibm_fez", "-n", "1"])
        assert result.exit_code == 0
        assert "ibm_fez" in result.output
        assert "fidelity" in result.output.lower()

    def test_shows_suggestions(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["analyze", BELL, "-b", "ibm_fez", "-n", "1"])
        assert result.exit_code == 0
        # Bell state is viable, so there should be a suggestion
        assert "Suggest" in result.output or "proceed" in result.output.lower()

    def test_shows_upsell(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["analyze", BELL, "-b", "ibm_fez", "-n", "1"])
        assert "qubitboost" in result.output.lower()

    def test_shows_gate_breakdown(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["analyze", BELL, "-b", "ibm_fez", "-n", "1"])
        assert "Gate breakdown" in result.output

    def test_shows_circuit_type(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["analyze", BELL, "-b", "ibm_fez", "-n", "1"])
        assert "Circuit type" in result.output

    def test_shows_gate_eligibility(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["analyze", BELL, "-b", "ibm_fez", "-n", "1"])
        assert "QubitBoost gate eligibility" in result.output


# ── qbc diff ────────────────────────────────────────────────────────


class TestDiff:
    def test_two_backends(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli, ["diff", BELL, "-b", "ibm_fez", "--vs", "ibm_torino", "-n", "1"]
        )
        assert result.exit_code == 0
        assert "ibm_fez" in result.output
        assert "ibm_torino" in result.output

    def test_shows_recommendation(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli, ["diff", BELL, "-b", "ibm_fez", "--vs", "ibm_torino", "-n", "1"]
        )
        assert "Recommendation" in result.output or "equivalent" in result.output

    def test_shows_upsell(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli, ["diff", BELL, "-b", "ibm_fez", "--vs", "ibm_torino", "-n", "1"]
        )
        assert "qubitboost.io" in result.output


# ── qbc info ────────────────────────────────────────────────────────


class TestInfo:
    def test_shows_version(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["info"])
        assert result.exit_code == 0
        assert "qb-compiler" in result.output

    def test_shows_backends(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["info"])
        assert "ibm_fez" in result.output
        assert "ibm_torino" in result.output


# ── qbc calibration show ────────────────────────────────────────────


class TestCalibrationShow:
    def test_known_backend(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["calibration", "show", "ibm_fez"])
        assert result.exit_code == 0
        assert "ibm_fez" in result.output
        assert "156" in result.output

    def test_unknown_backend(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["calibration", "show", "nonexistent"])
        assert result.exit_code != 0


# ── Multi-backend calibration tests ─────────────────────────────────


_BACKENDS_WITH_FIXTURES = ["ibm_fez", "ibm_torino"]


class TestMultiBackendPreflight:
    """Test preflight across all backends with calibration fixtures."""

    @pytest.mark.parametrize("backend", _BACKENDS_WITH_FIXTURES)
    def test_bell_preflight(self, runner: CliRunner, backend: str) -> None:
        result = runner.invoke(cli, ["preflight", BELL, "-b", backend, "-n", "1"])
        assert result.exit_code == 0
        assert "VIABLE" in result.output
        assert backend in result.output

    @pytest.mark.parametrize("backend", _BACKENDS_WITH_FIXTURES)
    def test_analyze_all(self, runner: CliRunner, backend: str) -> None:
        result = runner.invoke(cli, ["analyze", BELL, "-b", backend, "-n", "1"])
        assert result.exit_code == 0
        assert backend in result.output
        assert "fidelity" in result.output.lower()


class TestMultiBackendDiff:
    """Test diff across backend pairs with calibration fixtures."""

    def test_ibm_fez_vs_torino(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli, ["diff", BELL, "-b", "ibm_fez", "--vs", "ibm_torino", "-n", "1"]
        )
        assert result.exit_code == 0
        assert "ibm_fez" in result.output
        assert "ibm_torino" in result.output
        assert "fidelity" in result.output.lower()

    def test_diff_shows_all_metrics(self, runner: CliRunner) -> None:
        result = runner.invoke(
            cli, ["diff", BELL, "-b", "ibm_fez", "--vs", "ibm_torino", "-n", "1"]
        )
        assert "Status" in result.output
        assert "2Q gates" in result.output
        assert "Depth" in result.output
        assert "Cost" in result.output
