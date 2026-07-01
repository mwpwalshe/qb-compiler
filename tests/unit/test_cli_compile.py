"""CLI-level tests for ``qbc compile`` (previously only exercised via the library API)."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from qb_compiler.cli.main import cli

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "circuits"
BELL = str(FIXTURES / "bell_state.qasm")


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class TestCompile:
    def test_happy_path(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["compile", BELL, "-b", "ibm_fez"])
        assert result.exit_code == 0, (result.output, result.exception)
        assert "Compiled:" in result.output
        assert "Estimated fidelity:" in result.output
        assert "Compilation time:" in result.output

    def test_without_backend(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["compile", BELL])
        assert result.exit_code == 0, (result.output, result.exception)
        assert "Compiled:" in result.output

    @pytest.mark.parametrize("strategy", ["fidelity_optimal", "depth_optimal", "budget_optimal"])
    def test_strategies(self, runner: CliRunner, strategy: str) -> None:
        result = runner.invoke(cli, ["compile", BELL, "-b", "ibm_fez", "-s", strategy])
        assert result.exit_code == 0, (result.output, result.exception)
        assert "Estimated fidelity:" in result.output

    def test_invalid_strategy_rejected(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["compile", BELL, "-s", "not_a_strategy"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "not_a_strategy" in result.output

    def test_output_file_written(self, runner: CliRunner, tmp_path: Path) -> None:
        out = tmp_path / "compiled.qasm"
        result = runner.invoke(cli, ["compile", BELL, "-b", "ibm_fez", "-o", str(out)])
        assert result.exit_code == 0, (result.output, result.exception)
        assert out.exists()
        assert "Compiled by qb-compiler" in out.read_text()
        assert f"Written to {out}" in result.output

    def test_receipt_flag(self, runner: CliRunner, tmp_path: Path) -> None:
        circuit = tmp_path / "bell.qasm"
        circuit.write_text(Path(BELL).read_text())
        result = runner.invoke(cli, ["compile", str(circuit), "-b", "ibm_fez", "--receipt"])
        assert result.exit_code == 0, (result.output, result.exception)
        receipt = circuit.with_suffix(".receipt.json")
        assert receipt.exists()
        import json

        data = json.loads(receipt.read_text())
        assert isinstance(data, dict)
        assert "Receipt saved to" in result.output

    def test_missing_file_usage_error(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(cli, ["compile", str(tmp_path / "absent.qasm")])
        assert result.exit_code != 0

    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["compile", "--help"])
        assert result.exit_code == 0
        assert "--strategy" in result.output
        assert "--receipt" in result.output
