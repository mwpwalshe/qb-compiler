"""CLI tests for `qbc backends` (honest status) and `qbc preflight --live` (graceful fallback)."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

pytest.importorskip("qiskit")

from qb_compiler.cli.main import cli

_BELL = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n'


def test_backends_table():
    res = CliRunner().invoke(cli, ["backends"])
    assert res.exit_code == 0
    assert "LIVE STATUS" in res.output
    assert "ibm_fez" in res.output and "live" in res.output


def test_backends_json():
    res = CliRunner().invoke(cli, ["backends", "--json"])
    assert res.exit_code == 0
    rows = json.loads(res.output)
    by_backend = {r["backend"]: r for r in rows}
    assert by_backend["ibm_fez"]["live_status"] == "live"
    assert by_backend["ionq_aria"]["live_status"] == "live-unvalidated"
    assert by_backend["quantinuum_h2"]["live_status"] == "live-unvalidated"
    for r in rows:
        assert isinstance(r["live_deps_available"], bool)


def test_preflight_live_falls_back_gracefully(tmp_path):
    # --live with no IBM/AWS creds must not abort: registry degrades to static.
    p = tmp_path / "bell.qasm"
    p.write_text(_BELL)
    res = CliRunner().invoke(cli, ["preflight", str(p), "-b", "ionq_aria", "--live"])
    assert res.exit_code == 0
    assert "Status:" in res.output  # produced a verdict despite no live data
