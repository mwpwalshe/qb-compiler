"""Hostile-input tests for the ``dem-audit`` / ``dem-canonicalize`` CLI commands.

These commands take a stim ``.dem`` path and an output path. Hostile inputs:
malformed DEM text, a stim *circuit* masquerading as a DEM, binary/non-UTF8 files,
and output paths that are directories or have missing parents. Each must exit
nonzero with a clean message, never a raw traceback.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from qb_compiler.cli.main import cli

pytest.importorskip("stim")

WITNESS = "tests/fixtures/witness.dem"


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _assert_no_traceback(result) -> None:
    """A graceful failure has a SystemExit (or none), never an uncaught exception."""
    if result.exception is not None:
        assert isinstance(result.exception, SystemExit), (
            f"uncaught {type(result.exception).__name__}: {result.exception}"
        )


class TestDemAuditHostile:
    def test_malformed_dem_text(self, runner: CliRunner, tmp_path) -> None:
        bad = tmp_path / "bad.dem"
        bad.write_text("this is not a dem @@@\n")
        result = runner.invoke(cli, ["dem-audit", str(bad)])
        _assert_no_traceback(result)
        assert result.exit_code == 2
        assert "could not parse" in result.output

    def test_stim_circuit_as_dem(self, runner: CliRunner, tmp_path) -> None:
        circ = tmp_path / "circ.dem"
        circ.write_text("H 0\nCNOT 0 1\nM 0 1\n")
        result = runner.invoke(cli, ["dem-audit", str(circ)])
        _assert_no_traceback(result)
        assert result.exit_code == 2

    def test_binary_non_utf8_dem(self, runner: CliRunner, tmp_path) -> None:
        b = tmp_path / "bin.dem"
        b.write_bytes(b"\xff\xfe\x00\x01malformed")
        result = runner.invoke(cli, ["dem-audit", str(b)])
        _assert_no_traceback(result)
        assert result.exit_code == 2

    def test_empty_dem_file_is_clean_pass(self, runner: CliRunner, tmp_path) -> None:
        e = tmp_path / "empty.dem"
        e.write_text("")
        result = runner.invoke(cli, ["dem-audit", str(e)])
        _assert_no_traceback(result)
        assert result.exit_code == 0

    def test_nonexistent_file(self, runner: CliRunner, tmp_path) -> None:
        result = runner.invoke(cli, ["dem-audit", str(tmp_path / "nope.dem")])
        # click.Path(exists=True) rejects this before our code runs.
        assert result.exit_code == 2

    def test_witness_fail_exits_two(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["dem-audit", WITNESS])
        _assert_no_traceback(result)
        assert result.exit_code == 2  # genuine FAIL

    def test_output_parent_missing(self, runner: CliRunner, tmp_path) -> None:
        out = tmp_path / "nodir" / "sub" / "out.dem"
        result = runner.invoke(cli, ["dem-audit", WITNESS, "-o", str(out)])
        _assert_no_traceback(result)
        assert result.exit_code != 0
        assert "could not write" in result.output

    def test_output_is_a_directory(self, runner: CliRunner, tmp_path) -> None:
        d = tmp_path / "adir"
        d.mkdir()
        result = runner.invoke(cli, ["dem-audit", WITNESS, "-o", str(d)])
        _assert_no_traceback(result)
        assert result.exit_code != 0
        assert "could not write" in result.output


class TestDemCanonicalizeHostile:
    def test_malformed_dem_text(self, runner: CliRunner, tmp_path) -> None:
        bad = tmp_path / "bad.dem"
        bad.write_text("garbage not a dem\n")
        result = runner.invoke(cli, ["dem-canonicalize", str(bad), "-o", str(tmp_path / "o.dem")])
        _assert_no_traceback(result)
        assert result.exit_code == 2

    def test_output_is_a_directory(self, runner: CliRunner, tmp_path) -> None:
        d = tmp_path / "adir"
        d.mkdir()
        result = runner.invoke(cli, ["dem-canonicalize", WITNESS, "-o", str(d)])
        _assert_no_traceback(result)
        assert result.exit_code != 0
        assert "could not write" in result.output

    def test_output_parent_missing(self, runner: CliRunner, tmp_path) -> None:
        out = tmp_path / "nodir" / "x.dem"
        result = runner.invoke(cli, ["dem-canonicalize", WITNESS, "-o", str(out)])
        _assert_no_traceback(result)
        assert result.exit_code != 0

    def test_witness_roundtrip_preserves_masks(self, runner: CliRunner, tmp_path) -> None:
        out = tmp_path / "safe.dem"
        result = runner.invoke(cli, ["dem-canonicalize", WITNESS, "-o", str(out)])
        _assert_no_traceback(result)
        assert result.exit_code == 0
        assert out.exists()
