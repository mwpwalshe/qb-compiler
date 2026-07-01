"""CLI tests for `qbc preflight --braket` (graceful-fallback path, no AWS required)."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

pytest.importorskip("qiskit")

from qb_compiler.cli.main import cli

_BELL = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
    "h q[0];\ncx q[0],q[1];\nmeasure q -> c;\n"
)


def _bell(tmp_path):
    p = tmp_path / "bell.qasm"
    p.write_text(_BELL)
    return str(p)


def test_live_flag_present_in_help():
    # --live is the public flag; --braket is now a hidden deprecated alias.
    res = CliRunner().invoke(cli, ["preflight", "--help"])
    assert res.exit_code == 0
    assert "--live" in res.output
    assert "--braket" not in res.output  # hidden


def test_braket_alias_still_works(tmp_path):
    # --braket is a back-compat alias for --live: routes through the registry, never aborts.
    res = CliRunner().invoke(cli, ["preflight", _bell(tmp_path), "-b", "ionq_aria", "--braket"])
    assert res.exit_code == 0
    assert "Status:" in res.output  # preflight still produced a verdict
