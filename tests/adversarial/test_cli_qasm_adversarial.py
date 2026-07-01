"""Hostile-input tests for the QASM-consuming CLI commands.

``compile`` uses its own minimal QASM reader/parser (not qiskit); ``preflight`` /
``analyze`` / ``diff`` / ``when`` / ``verify`` go through ``_load_qasm`` (qiskit).
Hostile inputs: binary / non-UTF8 / wrong-type files, empty files, pathological
qubit counts, deeply nested expressions, bad backend names, output-path attacks,
and unicode/emoji args. Every case must exit nonzero cleanly, never traceback or hang.
"""

from __future__ import annotations

import time

import pytest
from click.testing import CliRunner

from qb_compiler.cli.main import cli

pytest.importorskip("qiskit")

BELL = "tests/fixtures/circuits/bell_state.qasm"


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _no_traceback(result) -> None:
    if result.exception is not None:
        assert isinstance(result.exception, SystemExit), (
            f"uncaught {type(result.exception).__name__}: {result.exception}"
        )


# ── compile: file-read / parser attacks ─────────────────────────────


class TestCompileHostile:
    def test_binary_non_utf8_file(self, runner: CliRunner, tmp_path) -> None:
        p = tmp_path / "bin.qasm"
        p.write_bytes(b"\xff\xfe\x00bad")
        result = runner.invoke(cli, ["compile", str(p)])
        _no_traceback(result)
        assert result.exit_code == 1
        assert "could not read" in result.output

    def test_png_disguised_as_qasm(self, runner: CliRunner, tmp_path) -> None:
        p = tmp_path / "img.qasm"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        result = runner.invoke(cli, ["compile", str(p)])
        _no_traceback(result)
        assert result.exit_code == 1

    def test_empty_file(self, runner: CliRunner, tmp_path) -> None:
        p = tmp_path / "empty.qasm"
        p.write_text("")
        result = runner.invoke(cli, ["compile", str(p)])
        _no_traceback(result)
        assert result.exit_code == 1
        assert "qubit count" in result.output

    def test_pathological_qubit_count_does_not_hang(self, runner: CliRunner, tmp_path) -> None:
        # FIX: previously hung/OOMed building a ~1e9-qubit circuit.
        p = tmp_path / "huge.qasm"
        p.write_text("OPENQASM 2.0;\nqreg q[999999999];\nh q[0];\n")
        start = time.monotonic()
        result = runner.invoke(cli, ["compile", str(p)])
        elapsed = time.monotonic() - start
        _no_traceback(result)
        assert result.exit_code == 1
        assert elapsed < 10.0
        assert "exceeds the supported maximum" in result.output

    def test_deeply_nested_parens_no_hang(self, runner: CliRunner, tmp_path) -> None:
        p = tmp_path / "nest.qasm"
        p.write_text("OPENQASM 2.0;\nqreg q[2];\nrx(" + "(" * 500 + "0.1" + ")" * 500 + ") q[0];\n")
        start = time.monotonic()
        result = runner.invoke(cli, ["compile", str(p)])
        _no_traceback(result)
        assert time.monotonic() - start < 10.0

    def test_output_path_is_directory(self, runner: CliRunner, tmp_path) -> None:
        d = tmp_path / "outdir"
        d.mkdir()
        result = runner.invoke(cli, ["compile", BELL, "-o", str(d)])
        _no_traceback(result)
        assert result.exit_code == 1
        assert "could not write" in result.output

    def test_output_parent_missing(self, runner: CliRunner, tmp_path) -> None:
        out = tmp_path / "nodir" / "o.qasm"
        result = runner.invoke(cli, ["compile", BELL, "-o", str(out)])
        _no_traceback(result)
        assert result.exit_code == 1
        assert "could not write" in result.output

    def test_directory_passed_as_circuit(self, runner: CliRunner, tmp_path) -> None:
        result = runner.invoke(cli, ["compile", str(tmp_path)])
        # click.Path(dir_okay=False) rejects this with exit 2.
        assert result.exit_code == 2


# ── viability commands: bad files / backends / args ─────────────────


class TestViabilityCommandsHostile:
    def test_preflight_png(self, runner: CliRunner, tmp_path) -> None:
        p = tmp_path / "img.qasm"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        result = runner.invoke(cli, ["preflight", str(p), "-b", "ibm_fez"])
        _no_traceback(result)
        assert result.exit_code == 1

    def test_analyze_binary(self, runner: CliRunner, tmp_path) -> None:
        p = tmp_path / "bin.qasm"
        p.write_bytes(b"\xff\xfe\x00bad")
        result = runner.invoke(cli, ["analyze", str(p), "-b", "ibm_fez"])
        _no_traceback(result)
        assert result.exit_code == 1

    def test_when_on_garbage(self, runner: CliRunner, tmp_path) -> None:
        p = tmp_path / "img.qasm"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        result = runner.invoke(cli, ["when", str(p)])
        _no_traceback(result)
        assert result.exit_code == 1

    def test_verify_on_garbage(self, runner: CliRunner, tmp_path) -> None:
        p = tmp_path / "img.qasm"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        result = runner.invoke(cli, ["verify", str(p)])
        _no_traceback(result)
        assert result.exit_code == 1

    def test_calibration_show_bad_backend(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["calibration", "show", "nonexistent_xyz"])
        _no_traceback(result)
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_emoji_backend_does_not_crash(self, runner: CliRunner) -> None:
        # Unicode/emoji backend name must degrade gracefully, never traceback.
        result = runner.invoke(cli, ["preflight", BELL, "-b", "\U0001f480\U0001f525"])
        _no_traceback(result)

    def test_diff_bad_backend(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["diff", BELL, "-b", "ibm_fez", "--vs", "\U0001f4a9"])
        _no_traceback(result)


# ── missing / malformed argument handling ───────────────────────────


class TestArgHandling:
    def test_preflight_missing_backend(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["preflight", BELL])
        assert result.exit_code == 2  # click usage error

    def test_preflight_multi_backend_rejected(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["preflight", BELL, "-b", "ibm_fez", "-b", "ibm_torino"])
        _no_traceback(result)
        assert result.exit_code == 1

    def test_unknown_command(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["frobnicate"])
        assert result.exit_code == 2

    def test_compile_missing_circuit_arg(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["compile"])
        assert result.exit_code == 2
