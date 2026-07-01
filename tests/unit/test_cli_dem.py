"""CLI-level tests for the v0.8.0 ObservableGate commands ``dem-audit`` / ``dem-canonicalize``.

These exercise the click command wiring (exit codes 0/1/2, ``--strict``, ``-o`` canonicalize
output, ``--help`` text) on top of the already-unit-tested ``observable_gate`` API. They require
stim (the ``[ising]`` extra) and are skipped otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from qb_compiler.cli.main import cli

# Detector-identical / logical-distinct witness -> intrinsic FAIL.
WITNESS = str(Path(__file__).resolve().parents[1] / "fixtures" / "witness.dem")

PASS_DEM = "error(0.01) D0\nerror(0.01) D1\ndetector D0\ndetector D1\n"
# Mixed group (D0 with/without L0) plus a decomposed (separator) mechanism -> WARN, not FAIL.
WARN_DEM = (
    "error(0.01) D0\n"
    "error(0.01) D0 L0\n"
    "error(0.01) D1 ^ D2\n"
    "detector D0\ndetector D1\ndetector D2\n"
)


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _require_stim() -> None:
    pytest.importorskip("stim")


def _write(tmp_path: Path, name: str, body: str) -> str:
    p = tmp_path / name
    p.write_text(body)
    return str(p)


# ── qbc dem-audit ────────────────────────────────────────────────────


class TestDemAudit:
    def test_fail_exits_2(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["dem-audit", WITNESS])
        assert result.exit_code == 2
        assert "status: FAIL" in result.output
        assert "ObservableGate DEM audit" in result.output

    def test_fail_exits_2_even_with_strict(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["dem-audit", WITNESS, "--strict"])
        assert result.exit_code == 2
        assert "status: FAIL" in result.output

    def test_pass_exits_0(self, runner: CliRunner, tmp_path: Path) -> None:
        dem = _write(tmp_path, "pass.dem", PASS_DEM)
        result = runner.invoke(cli, ["dem-audit", dem])
        assert result.exit_code == 0
        assert "status: PASS" in result.output

    def test_warn_exits_0_by_default(self, runner: CliRunner, tmp_path: Path) -> None:
        dem = _write(tmp_path, "warn.dem", WARN_DEM)
        result = runner.invoke(cli, ["dem-audit", dem])
        assert result.exit_code == 0
        assert "status: WARN" in result.output

    def test_warn_exits_1_with_strict(self, runner: CliRunner, tmp_path: Path) -> None:
        dem = _write(tmp_path, "warn.dem", WARN_DEM)
        result = runner.invoke(cli, ["dem-audit", dem, "--strict"])
        assert result.exit_code == 1
        assert "status: WARN" in result.output

    def test_canonicalize_output_flag(self, runner: CliRunner, tmp_path: Path) -> None:
        out = tmp_path / "safe.dem"
        result = runner.invoke(cli, ["dem-audit", WITNESS, "-o", str(out)])
        # FAIL is intrinsic, so exit stays 2 even though a safe DEM is written.
        assert result.exit_code == 2
        assert out.exists()
        assert "observable-preserving DEM written to" in result.output

        import stim

        safe = stim.DetectorErrorModel.from_file(str(out))
        # Both distinct (detector, mask) mechanisms preserved, none erased.
        assert safe.num_errors == 2

    def test_missing_file_errors(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(cli, ["dem-audit", str(tmp_path / "nope.dem")])
        assert result.exit_code != 0  # click Path(exists=True) usage error == 2

    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["dem-audit", "--help"])
        assert result.exit_code == 0
        assert "observable-mask collapse" in result.output
        assert "--strict" in result.output
        assert "--canonicalize" in result.output or "-o" in result.output


# ── qbc dem-canonicalize ─────────────────────────────────────────────


class TestDemCanonicalize:
    def test_writes_output(self, runner: CliRunner, tmp_path: Path) -> None:
        out = tmp_path / "canon.dem"
        result = runner.invoke(cli, ["dem-canonicalize", WITNESS, "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        assert "observable-preserving DEM written to" in result.output
        assert "mechanisms:" in result.output

        import stim

        safe = stim.DetectorErrorModel.from_file(str(out))
        assert safe.num_errors == 2  # distinct (detector, mask) pairs preserved

    def test_output_is_required(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["dem-canonicalize", WITNESS])
        assert result.exit_code != 0  # missing required -o/--output
        assert "output" in result.output.lower() or "Missing option" in result.output

    def test_pass_dem_roundtrips(self, runner: CliRunner, tmp_path: Path) -> None:
        dem = _write(tmp_path, "pass.dem", PASS_DEM)
        out = tmp_path / "pass_canon.dem"
        result = runner.invoke(cli, ["dem-canonicalize", dem, "-o", str(out)])
        assert result.exit_code == 0
        assert "PASS -> PASS" in result.output

    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["dem-canonicalize", "--help"])
        assert result.exit_code == 0
        assert "observable-preserving canonical" in result.output
        assert "-o" in result.output or "--output" in result.output
