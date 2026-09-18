# SPDX-License-Identifier: Apache-2.0
"""``qbc pricing``, and the pricing fields the other commands now carry.

Exit codes and the four provenance fields are what matter here. Nothing touches the network: the
live path is a local feed file, and every key is a throwaway generated in the test.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from qb_compiler.cli.main import cli
from qb_compiler.cost.pricing import PRICING_AS_OF
from qb_compiler.cost.pricing_feed import build_feed, sign_feed

from .test_cost.test_pricing_feed import a_key, some_entries


@pytest.fixture()
def runner():
    return CliRunner()


@pytest.fixture()
def signed_feed(tmp_path):
    seed, _ = a_key(17)
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps(sign_feed(build_feed(some_entries()), seed)), encoding="utf-8")
    return path


class TestPricingShow:
    def test_it_lists_every_backend_with_its_billing_model(self, runner):
        result = runner.invoke(cli, ["pricing", "show"])
        assert result.exit_code == 0, result.output
        assert "ibm_fez" in result.output
        assert "per_second" in result.output
        assert "per_hqc" in result.output
        assert f"static {PRICING_AS_OF}" in result.output

    def test_json_carries_the_provenance_fields(self, runner):
        result = runner.invoke(cli, ["pricing", "show", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["pricing_status"] == "static"
        assert payload["pricing_as_of"] == PRICING_AS_OF
        assert payload["pricing_signature_verified"] is False
        row = {e["backend"]: e for e in payload["entries"]}["quantinuum_h2"]
        assert row["billing"] == "per_hqc"
        assert row["notes"]

    def test_the_static_table_says_three_vendors_do_not_sell_shots(self, runner):
        result = runner.invoke(cli, ["pricing", "show"])
        assert "do not sell shots" in result.output


class TestPricingVerify:
    def test_a_feed_signed_by_another_key_exits_two(self, runner, signed_feed):
        result = runner.invoke(cli, ["pricing", "verify", str(signed_feed)])
        assert result.exit_code == 2
        assert "REFUSED" in result.output

    def test_it_verifies_against_the_key_it_was_signed_with(self, runner, signed_feed):
        import base64

        _, public = a_key(17)
        result = runner.invoke(
            cli,
            [
                "pricing",
                "verify",
                str(signed_feed),
                "--key",
                base64.b64encode(public).decode("ascii"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "VERIFIED" in result.output
        assert "entries      : 2" in result.output

    def test_a_feed_with_the_wrong_schema_exits_two(self, runner, tmp_path):
        path = tmp_path / "not-a-feed.json"
        path.write_text(json.dumps({"schema": "qb.other.v1", "entries": []}), encoding="utf-8")
        result = runner.invoke(cli, ["pricing", "verify", str(path)])
        assert result.exit_code == 2
        assert "REFUSED" in result.output

    def test_json_verdict(self, runner, signed_feed):
        import base64

        _, public = a_key(17)
        result = runner.invoke(
            cli,
            [
                "pricing",
                "verify",
                str(signed_feed),
                "--key",
                base64.b64encode(public).decode("ascii"),
                "--json",
            ],
        )
        assert result.exit_code == 0
        verdict = json.loads(result.output)
        assert verdict["ok"] is True
        assert verdict["entries"] == 2
        assert verdict["signed_by"] == verdict["checked_against"]

    def test_the_group_lists_its_commands(self, runner):
        result = runner.invoke(cli, ["pricing", "--help"])
        assert result.exit_code == 0
        for command in ("show", "verify"):
            assert command in result.output


class TestDoctor:
    def test_it_reports_the_static_table_without_being_asked_for_live(self, runner):
        pytest.importorskip("rich")
        result = runner.invoke(cli, ["doctor"])
        assert "pricing static" in result.output
        assert PRICING_AS_OF in result.output


HAMILTONIAN = {
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


class TestMeasurePlanPricing:
    @pytest.fixture()
    def hamiltonian(self, tmp_path):
        path = tmp_path / "h2.json"
        path.write_text(json.dumps(HAMILTONIAN), encoding="utf-8")
        return str(path)

    def test_without_a_backend_it_prices_nothing_and_still_works(self, runner, hamiltonian):
        result = runner.invoke(cli, ["measure-plan", hamiltonian, "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "cost" not in payload
        assert payload["schema"] == "qb.measure_plan.v1"

    def test_with_a_backend_it_carries_the_cost_and_its_provenance(self, runner, hamiltonian):
        result = runner.invoke(cli, ["measure-plan", hamiltonian, "-b", "ibm_fez", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["backend"] == "ibm_fez"
        assert payload["pricing_status"] == "static"
        assert payload["pricing_as_of"] == PRICING_AS_OF
        assert payload["cost"]["model"] == "per_second"
        assert payload["cost"]["usd"] == pytest.approx(payload["shots_qwc"] * 0.00016)
        assert payload["cost"]["assumptions"]["assumed_shots_per_second"] == 10_000

    def test_an_unknown_backend_exits_three(self, runner, hamiltonian):
        result = runner.invoke(cli, ["measure-plan", hamiltonian, "-b", "no_such_backend"])
        assert result.exit_code == 3
        # This runner does not separate the streams, so output already holds the error line.
        assert "no pricing" in result.output.lower()

    def test_the_human_output_names_the_assumption(self, runner, hamiltonian):
        result = runner.invoke(cli, ["measure-plan", hamiltonian, "-b", "ibm_fez"])
        assert result.exit_code == 0
        assert "pricing" in result.output
        assert "publishes no per-shot price" in result.output
