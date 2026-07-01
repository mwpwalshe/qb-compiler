"""CLI tests for `qbc dem-audit --json` (the free community-tier receipt)."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

pytest.importorskip("stim")

from qb_compiler.cli.main import cli

_WITNESS = "error(0.01) D0\nerror(0.01) D0 L0\ndetector D0\n"
_CLEAN = "error(0.01) D0\nerror(0.01) D1\ndetector D0\ndetector D1\n"


def _write(tmp_path, text, name="m.dem"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_json_receipt_is_valid_and_fails(tmp_path):
    dem = _write(tmp_path, _WITNESS)
    res = CliRunner().invoke(cli, ["dem-audit", dem, "--json"])
    assert res.exit_code == 2  # FAIL
    receipt = json.loads(res.output)
    assert receipt["schema"] == "observablegate.receipt/1"
    assert receipt["tool"] == "qb-compiler"
    assert receipt["tier"] == "community"
    assert receipt["status"] == "FAIL"
    assert receipt["audit"]["mixed_groups"] == 1
    assert receipt["audit"]["unique_detector_sigs"] < receipt["audit"]["unique_detector_obs_sigs"]
    assert "tool_version" in receipt


def test_json_receipt_pass(tmp_path):
    dem = _write(tmp_path, _CLEAN)
    res = CliRunner().invoke(cli, ["dem-audit", dem, "--json"])
    assert res.exit_code == 0  # PASS
    receipt = json.loads(res.output)
    assert receipt["status"] == "PASS"
    assert receipt["audit"]["mixed_groups"] == 0
    assert receipt["canonicalized_to"] is None


def test_json_receipt_records_canonicalization(tmp_path):
    dem = _write(tmp_path, _WITNESS)
    out = str(tmp_path / "safe.dem")
    res = CliRunner().invoke(cli, ["dem-audit", dem, "--json", "-o", out])
    receipt = json.loads(res.output)
    assert receipt["canonicalized_to"] == out
