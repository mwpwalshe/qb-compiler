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


@pytest.mark.parametrize("absent", ["pytket", "azure", "braket", "qiskit_ibm_runtime"])
def test_backends_survives_absent_vendor_sdk(monkeypatch, absent):
    """`qbc backends` must work on a bare install, with no vendor SDK present.

    Every vendor SDK is an optional extra, so the common case is that none are installed. The
    dotted probes (``pytket.extensions.quantinuum``, ``azure.quantum``) called
    ``importlib.util.find_spec``, which imports the PARENT package to locate a submodule and so
    RAISES ModuleNotFoundError when the parent is missing rather than returning None. The result
    was that ``qbc backends`` exited 1 with a traceback on any install lacking the quantinuum
    extra, which is the default.

    This is parametrised over each vendor and simulates the absence rather than relying on what
    the test machine happens to have installed, so it holds in CI and on a developer box alike.
    """
    import importlib.util

    real_find_spec = importlib.util.find_spec

    def find_spec_without(name, *args, **kwargs):
        if name.split(".")[0] == absent:
            raise ModuleNotFoundError(f"No module named '{absent}'")
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec_without)

    res = CliRunner().invoke(cli, ["backends"])
    assert res.exit_code == 0, f"crashed with {absent} absent: {res.exception!r}"
    assert "LIVE STATUS" in res.output

    # The backend still has to be listed, reported as unavailable rather than hidden.
    res_json = CliRunner().invoke(cli, ["backends", "--json"])
    assert res_json.exit_code == 0
    rows = {r["backend"]: r for r in json.loads(res_json.output)}
    assert "quantinuum_h2" in rows
    if absent == "pytket":
        assert rows["quantinuum_h2"]["live_deps_available"] is False
