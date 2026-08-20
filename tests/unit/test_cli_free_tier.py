# SPDX-License-Identifier: Apache-2.0
"""CLI tests for the free tier added in 0.12.0: chem-audit, measure-plan, verify-receipt, corpus.

Exit codes matter more than output here, because these commands are meant to sit in a CI job where
nobody reads the text. Each command's codes are pinned.
"""

from __future__ import annotations

import copy
import json

import pytest
from click.testing import CliRunner

from qb_compiler.cli.main import cli

GOOD = {
    "metadata": {
        "id": "h2_2e2o",
        "name": "Hydrogen",
        "active_space": {"n_electrons": 2, "n_orbitals": 2},
        "n_qubits": 4,
        "mapping": "jordan_wigner",
        "provenance": "PySCF -> OpenFermion -> JW -> sparse_FCI",
        "synthetic": False,
    },
    "pauli_groups": [
        {"pauli_strings": ["I"], "coefficients": [-0.81], "measurable": False},
        {"pauli_strings": ["Z0", "Z1"], "coefficients": [0.17, 0.17], "measurable": True},
        {"pauli_strings": ["X0 X1 Y2 Y3"], "coefficients": [0.045], "measurable": True},
    ],
    "reference_energy_type": "FCI_sparse",
    "reference_rhf_energy": -1.116,
    "reference_exact_energy": -1.137,
}


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def good_file(tmp_path):
    path = tmp_path / "h2.json"
    path.write_text(json.dumps(GOOD))
    return str(path)


@pytest.fixture()
def local_key(tmp_path, monkeypatch):
    monkeypatch.setenv("QBC_SIGNING_KEY", str(tmp_path / "signing_key"))
    monkeypatch.setenv("QBC_TRUSTED_KEYS", str(tmp_path / "trusted_keys"))
    from qb_compiler.signing import load_or_create_signing_key

    return load_or_create_signing_key()


class TestChemAudit:
    def test_accept_exits_zero(self, runner, good_file):
        result = runner.invoke(cli, ["chem-audit", good_file])
        assert result.exit_code == 0
        assert "ACCEPT" in result.output

    def test_refuse_exits_two(self, runner, tmp_path):
        bad = copy.deepcopy(GOOD)
        bad["metadata"]["n_qubits"] = 2
        bad["pauli_groups"] = [
            {"pauli_strings": ["Z0 X1"], "coefficients": [0.17], "measurable": True}
        ]
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        result = runner.invoke(cli, ["chem-audit", str(path)])
        assert result.exit_code == 2
        assert "REFUSE" in result.output

    def test_incomplete_exits_one(self, runner, tmp_path):
        path = tmp_path / "sparse.json"
        path.write_text(json.dumps({"n_qubits": 2, "terms": [["Z0", 1.0]]}))
        result = runner.invoke(cli, ["chem-audit", str(path)])
        assert result.exit_code == 1
        assert "INCOMPLETE" in result.output

    def test_strict_turns_incomplete_into_a_refusal(self, runner, tmp_path):
        path = tmp_path / "sparse.json"
        path.write_text(json.dumps({"n_qubits": 2, "terms": [["Z0", 1.0]]}))
        assert runner.invoke(cli, ["chem-audit", str(path), "--strict"]).exit_code == 2

    def test_json_output_is_a_receipt(self, runner, good_file):
        result = runner.invoke(cli, ["chem-audit", good_file, "--json"])
        payload = json.loads(result.output)
        assert payload["schema"] == "qb.chem_audit.v1"
        assert payload["verdict"] == "ACCEPT"

    def test_unreadable_file_exits_three(self, runner, tmp_path):
        path = tmp_path / "junk.json"
        path.write_text("{nope")
        result = runner.invoke(cli, ["chem-audit", str(path)])
        assert result.exit_code == 3
        assert "Error" in result.output


class TestMeasurePlan:
    def test_prints_the_bill(self, runner, good_file):
        result = runner.invoke(cli, ["measure-plan", good_file])
        assert result.exit_code == 0
        assert "QWC settings" in result.output

    def test_shots_scale_with_the_rate(self, runner, good_file):
        payload = json.loads(
            runner.invoke(
                cli, ["measure-plan", good_file, "--shots-per-setting", "1000", "--json"]
            ).output
        )
        assert payload["schema"] == "qb.measure_plan.v1"
        assert payload["shots_qwc"] == payload["n_settings_qwc"] * 1000

    def test_note_travels_with_the_json(self, runner, good_file):
        payload = json.loads(runner.invoke(cli, ["measure-plan", good_file, "--json"]).output)
        assert "structural" in payload["notes"][0]


class TestVerifyReceipt:
    def _receipt_file(self, tmp_path, receipt):
        path = tmp_path / "receipt.json"
        path.write_text(json.dumps(receipt))
        return str(path)

    def test_verified_exits_zero(self, runner, tmp_path, local_key):
        from qb_compiler.signing import sign_receipt

        signed = sign_receipt({"schema": "qb.selection_receipt.v1", "selected_layout": {"0": 1}})
        path = self._receipt_file(tmp_path, signed)
        result = runner.invoke(cli, ["verify-receipt", path, "--key", local_key.public_b64])
        assert result.exit_code == 0
        assert "VERIFIED" in result.output

    def test_key_can_be_a_file(self, runner, tmp_path, local_key):
        from qb_compiler.signing import export_public_key, sign_receipt

        key_file = export_public_key(tmp_path / "qbc.pub", key=local_key)
        path = self._receipt_file(tmp_path, sign_receipt({"schema": "qb.selection_receipt.v1"}))
        result = runner.invoke(cli, ["verify-receipt", path, "--key", str(key_file)])
        assert result.exit_code == 0

    def test_tampered_receipt_exits_two(self, runner, tmp_path, local_key):
        from qb_compiler.signing import sign_receipt

        signed = sign_receipt({"schema": "qb.selection_receipt.v1", "selected_layout": {"0": 1}})
        signed["selected_layout"] = {"0": 99}
        path = self._receipt_file(tmp_path, signed)
        result = runner.invoke(cli, ["verify-receipt", path, "--key", local_key.public_b64])
        assert result.exit_code == 2
        assert "INVALID_SIGNATURE" in result.output

    def test_no_key_exits_one(self, runner, tmp_path, local_key):
        from qb_compiler.signing import sign_receipt

        path = self._receipt_file(tmp_path, sign_receipt({"schema": "qb.selection_receipt.v1"}))
        result = runner.invoke(cli, ["verify-receipt", path])
        assert result.exit_code == 1
        assert "NO_KEY" in result.output

    def test_unsigned_passes_unless_strict(self, runner, tmp_path, local_key):
        path = self._receipt_file(tmp_path, {"schema": "qb.selection_receipt.v1"})
        assert runner.invoke(cli, ["verify-receipt", path]).exit_code == 0
        assert runner.invoke(cli, ["verify-receipt", path, "--strict"]).exit_code == 1

    def test_legacy_self_signed_exits_two(self, runner, tmp_path, local_key):
        import base64

        legacy = {
            "schema": "qb.selection_receipt.v1",
            "signature": base64.b64encode(bytes(64)).decode(),
            "public_key": base64.b64encode(bytes(32)).decode(),
        }
        path = self._receipt_file(tmp_path, legacy)
        result = runner.invoke(cli, ["verify-receipt", path])
        assert result.exit_code == 2
        assert "LEGACY_SELF_SIGNED" in result.output

    def test_json_verdict(self, runner, tmp_path, local_key):
        from qb_compiler.signing import sign_receipt

        path = self._receipt_file(tmp_path, sign_receipt({"schema": "qb.chem_audit.v1"}))
        result = runner.invoke(
            cli, ["verify-receipt", path, "--key", local_key.public_b64, "--json"]
        )
        payload = json.loads(result.output)
        assert payload["status"] == "VERIFIED"
        assert payload["does_not_claim"]


class TestCorpusCommands:
    def test_list_names_the_datasets(self, runner):
        result = runner.invoke(cli, ["corpus", "list"])
        assert result.exit_code == 0
        assert "willow-105q-d3-d5-d7" in result.output
        assert "Nothing is mirrored here" in result.output

    def test_list_json(self, runner):
        payload = json.loads(runner.invoke(cli, ["corpus", "list", "--json"]).output)
        assert payload and all("sha256" in entry for entry in payload)

    def test_show_prints_the_citation(self, runner):
        result = runner.invoke(cli, ["corpus", "show", "willow-105q-d3-d5-d7"])
        assert result.exit_code == 0
        assert "cite" in result.output
        assert "zenodo" in result.output.lower()

    def test_show_unknown_exits_one(self, runner):
        assert runner.invoke(cli, ["corpus", "show", "nope"]).exit_code == 1

    def test_verify_missing_file_exits_one(self, runner, tmp_path):
        result = runner.invoke(
            cli, ["corpus", "verify", "willow-105q-d3-d5-d7", str(tmp_path / "absent.zip")]
        )
        assert result.exit_code == 1
        assert "MISSING" in result.output

    def test_verify_wrong_bytes_exits_two(self, runner, tmp_path):
        path = tmp_path / "google_105Q_surface_code_d3_d5_d7.zip"
        path.write_bytes(b"not the real archive")
        result = runner.invoke(cli, ["corpus", "verify", "willow-105q-d3-d5-d7", str(path)])
        assert result.exit_code == 2
        assert "MISMATCH" in result.output
        assert "expected:" in result.output


class TestActionDefinition:
    """The composite action is a public entry point, so its shape is pinned like any other."""

    def test_action_yml_is_valid_and_wires_the_free_checks(self):
        yaml = pytest.importorskip("yaml")
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        action = yaml.safe_load((root / "action.yml").read_text())

        assert action["runs"]["using"] == "composite"
        run_steps = " ".join(step.get("run", "") for step in action["runs"]["steps"])
        for command in ("qbc chem-audit", "qbc measure-plan", "qbc verify-receipt"):
            assert command in run_steps
        # Nothing in the action may emit a verdict about a result.
        assert "qbc when" not in run_steps
        for required in ("hamiltonians", "receipts", "public-key", "strict"):
            assert required in action["inputs"]


class TestVerifyReceiptBadKey:
    def test_unreadable_key_is_a_usage_error_not_a_verdict(self, runner, tmp_path, local_key):
        from qb_compiler.signing import sign_receipt

        path = tmp_path / "receipt.json"
        path.write_text(json.dumps(sign_receipt({"schema": "qb.selection_receipt.v1"})))
        result = runner.invoke(cli, ["verify-receipt", str(path), "--key", "not-a-key"])
        assert result.exit_code == 3
        assert "Error" in result.output
