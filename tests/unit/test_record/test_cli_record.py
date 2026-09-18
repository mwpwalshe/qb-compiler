# SPDX-License-Identifier: Apache-2.0
"""``qbc record``: exit codes, JSON on stdout, the human summary on stderr."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
from click.testing import CliRunner

pytest.importorskip("pymatching")

from qb_compiler.cli.main import cli
from qb_compiler.record.loaders import generic_npz

from .conftest import make_spec, simulate_repetition


def _runner() -> CliRunner:
    """A runner that keeps stderr apart from stdout on every supported click.

    click 8.2 removed ``mix_stderr`` and always keeps the two apart. 8.0 and 8.1 mix them unless
    asked not to, and then refuse to hand over ``result.stderr``.
    """
    if "mix_stderr" in inspect.signature(CliRunner.__init__).parameters:
        return CliRunner(mix_stderr=False)
    return CliRunner()


def _record(tmp_path, mirrored=False, name="record.npz"):
    run = simulate_repetition(n_shots=600, seed=21)
    return str(generic_npz.write_npz(tmp_path / name, make_spec(run, mirrored=mirrored)))


class TestRecordValidate:
    def test_a_good_record_exits_zero_with_json_on_stdout(self, tmp_path):
        result = _runner().invoke(cli, ["record", "validate", _record(tmp_path)])
        assert result.exit_code == 0, result.output
        report = json.loads(result.stdout)
        assert report["schema"] == "qb.record_validation.v1"
        assert report["passed"] is True
        assert len(report["checks"]) == 8
        assert "record validation: PASS" in result.stderr

    def test_a_mirrored_record_exits_two(self, tmp_path):
        result = _runner().invoke(cli, ["record", "validate", _record(tmp_path, mirrored=True)])
        assert result.exit_code == 2
        report = json.loads(result.stdout)
        assert report["passed"] is False
        assert any(c["name"] == "round_profile" and c["status"] == "FAIL" for c in report["checks"])

    def test_thresholds_can_be_given_inline(self, tmp_path):
        result = _runner().invoke(
            cli,
            [
                "record",
                "validate",
                _record(tmp_path),
                "--thresholds",
                json.dumps({"event_density": {"high": 0.9}}),
            ],
        )
        assert result.exit_code == 0
        report = json.loads(result.stdout)
        density = next(c for c in report["checks"] if c["name"] == "event_density")
        assert density["threshold"]["high"] == 0.9

    def test_thresholds_can_be_given_as_a_file(self, tmp_path):
        settings = tmp_path / "thresholds.json"
        settings.write_text(json.dumps({"event_density": {"low": 0.0001}}))
        result = _runner().invoke(
            cli, ["record", "validate", _record(tmp_path), "--thresholds", str(settings)]
        )
        assert result.exit_code == 0
        report = json.loads(result.stdout)
        density = next(c for c in report["checks"] if c["name"] == "event_density")
        assert density["threshold"]["low"] == 0.0001

    def test_an_unreadable_record_exits_one(self, tmp_path):
        broken = tmp_path / "broken.npz"
        with broken.open("wb") as handle:
            np.savez_compressed(handle, dets=np.zeros((4, 2, 2), dtype=np.int64))
        result = _runner().invoke(cli, ["record", "validate", str(broken)])
        assert result.exit_code == 1
        assert "Error:" in result.stderr

    def test_an_unknown_decoder_exits_one(self, tmp_path):
        result = _runner().invoke(
            cli, ["record", "validate", _record(tmp_path), "--decoder", "belief"]
        )
        assert result.exit_code == 1
        assert "decoder must be one of" in result.stderr


class TestRecordResidual:
    def _inputs(self, tmp_path):
        pytest.importorskip("sklearn")
        rng = np.random.default_rng(31)
        run = simulate_repetition(n_shots=800, seed=31)
        spec = make_spec(run)
        record = generic_npz.write_npz(tmp_path / "record.npz", spec)
        hidden = rng.normal(size=spec.n_shots)
        failed = (rng.random(spec.n_shots) < 1 / (1 + np.exp(-(hidden - 1.2)))).astype(np.uint8)
        output = tmp_path / "output.npz"
        with output.open("wb") as handle:
            np.savez_compressed(
                handle, prediction=failed ^ np.asarray(spec.labels), weight=rng.normal(size=800)
            )
        features = tmp_path / "features.npz"
        with features.open("wb") as handle:
            np.savez_compressed(handle, features=hidden[:, None], names=np.array(["hidden"]))
        return str(record), str(output), str(features)

    def test_it_reports_json_and_exits_zero(self, tmp_path):
        record, output, features = self._inputs(tmp_path)
        result = _runner().invoke(
            cli,
            [
                "record",
                "residual",
                record,
                "--decoder-output",
                output,
                "--features",
                features,
                "--holdout",
                "shot",
                "--nulls",
                "3",
            ],
        )
        assert result.exit_code == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["schema"] == "qb.record_residual.v1"
        assert report["feature_names"] == ["hidden"]
        assert report["holdout"] == "shot"
        assert report["nulls"]["geometry"]["available"] is False
        assert "residual" in result.stderr

    def test_decoder_output_without_a_prediction_exits_one(self, tmp_path):
        record, _, features = self._inputs(tmp_path)
        empty = tmp_path / "empty.npz"
        with empty.open("wb") as handle:
            np.savez_compressed(handle, weight=np.zeros(800))
        result = _runner().invoke(
            cli,
            [
                "record",
                "residual",
                record,
                "--decoder-output",
                str(empty),
                "--features",
                features,
                "--nulls",
                "0",
            ],
        )
        assert result.exit_code == 1
        assert "prediction" in result.stderr


class TestRecordLoadFez:
    def test_a_directory_with_no_regime_exits_three(self, tmp_path):
        result = _runner().invoke(cli, ["record", "load-fez", str(tmp_path), "-d", "5", "-r", "5"])
        assert result.exit_code == 3
        assert "Cannot run:" in result.stderr
        assert "d5_r5" in result.stderr

    def test_the_help_warns_about_the_stored_order_flag(self):
        result = CliRunner().invoke(cli, ["record", "load-fez", "--help"])
        assert result.exit_code == 0
        assert "--stored-order" in result.output
        assert "backwards" in result.output

    def test_the_group_is_listed(self):
        result = CliRunner().invoke(cli, ["record", "--help"])
        assert result.exit_code == 0
        for command in ("validate", "residual", "load-fez"):
            assert command in result.output
