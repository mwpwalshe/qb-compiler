"""Tests for multi-format circuit input on the CLI (.qasm and .qasm3)."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from qb_compiler.cli.main import cli
from tests.conftest import requires_qiskit

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "circuits"
BELL_QASM2 = str(FIXTURES / "bell_state.qasm")

QASM3_BELL = (
    "OPENQASM 3;\n"
    'include "stdgates.inc";\n'
    "qubit[2] q;\n"
    "bit[2] c;\n"
    "h q[0];\n"
    "cx q[0], q[1];\n"
    "c[0] = measure q[0];\n"
    "c[1] = measure q[1];\n"
)


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@requires_qiskit
class TestCliInputFormats:
    def test_qasm2_path_still_works(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["preflight", BELL_QASM2, "-b", "ibm_fez"])
        assert result.exit_code == 0
        assert "Status" in result.output

    def test_qasm3_file_preflight(self, runner: CliRunner, tmp_path: Path) -> None:
        pytest.importorskip("qiskit_qasm3_import")
        qasm3_file = tmp_path / "bell.qasm3"
        qasm3_file.write_text(QASM3_BELL, encoding="utf-8")

        result = runner.invoke(cli, ["preflight", str(qasm3_file), "-b", "ibm_fez"])
        assert result.exit_code == 0, result.output
        assert "Status" in result.output

    def test_qasm3_missing_extra_gives_hint(self, runner: CliRunner, tmp_path: Path) -> None:
        # Only meaningful when the extra is absent.
        try:
            import qiskit_qasm3_import  # noqa: F401

            pytest.skip("qiskit_qasm3_import installed; ImportError path not exercised")
        except ImportError:
            pass

        qasm3_file = tmp_path / "bell.qasm3"
        qasm3_file.write_text(QASM3_BELL, encoding="utf-8")

        result = runner.invoke(cli, ["preflight", str(qasm3_file), "-b", "ibm_fez"])
        assert result.exit_code == 1
        assert "qb-compiler[qasm3]" in result.output
