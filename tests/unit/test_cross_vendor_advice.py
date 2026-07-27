"""`qbc when` as a neutral cross-vendor advisor.

The ranking spans vendors, which means it will happily put an IonQ or IQM row next to an IBM one.
Only IBM is validated against real hardware today, so an unlabelled table would invite a reader to
treat a model estimate as a measurement. The `validation` field, surfaced as the `Data` column,
exists to stop that, and these tests hold it to it.

The provenance labels come from the calibration registry's live status, so they cannot drift from
what `qbc backends` reports without one of these failing.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

pytest.importorskip("qiskit")

from qb_compiler.cli.main import cli
from qb_compiler.windows import format_table, rank_value

_GHZ_QASM = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nh q[0];\ncx q[0],q[1];\ncx q[1],q[2];\n'
)

# One validated vendor and three that are not, so the distinction has something to bite on.
CROSS_VENDOR = ["ibm_fez", "ionq_aria", "iqm_garnet", "rigetti_ankaa"]

VALID_LABELS = {"validated", "UNVALIDATED", "fixture-only", "no-adapter", "unknown"}


@pytest.fixture
def circuit_file(tmp_path):
    path = tmp_path / "ghz.qasm"
    path.write_text(_GHZ_QASM)
    return str(path)


class TestProvenanceLabelling:
    def test_every_row_declares_its_provenance(self):
        rows = rank_value(_qc(), backends=CROSS_VENDOR, n_seeds=1)
        assert rows, "expected at least one ranked backend"
        for row in rows:
            assert row.validation in VALID_LABELS, f"{row.backend}: {row.validation!r}"

    def test_ibm_is_validated_and_the_others_are_not(self):
        """The whole point: only hardware we have actually validated says so.

        If a vendor becomes genuinely validated this test should be updated deliberately, which is
        the intent. It must never start passing because the label was quietly widened.
        """
        rows = {r.backend: r for r in rank_value(_qc(), backends=CROSS_VENDOR, n_seeds=1)}
        assert rows["ibm_fez"].validation == "validated"
        for backend in ("ionq_aria", "iqm_garnet", "rigetti_ankaa"):
            assert rows[backend].validation != "validated", (
                f"{backend} is labelled validated, but no hardware validation exists for it"
            )

    def test_unvalidated_rows_carry_a_note_saying_so(self):
        rows = rank_value(_qc(), backends=["ionq_aria"], n_seeds=1)
        notes = " ".join(rows[0].notes).lower()
        assert "not validated" in notes
        assert "estimate" in notes

    def test_labels_match_what_the_registry_reports(self):
        """Provenance is derived, not asserted, so it cannot drift from `qbc backends`."""
        from qb_compiler.calibration.registry import all_backend_statuses

        expected = {
            "live": "validated",
            "live-unvalidated": "UNVALIDATED",
            "static": "fixture-only",
            "none": "no-adapter",
        }
        status = {s.backend: s.live_status.value for s in all_backend_statuses()}
        for row in rank_value(_qc(), backends=CROSS_VENDOR, n_seeds=1):
            if row.backend in status:
                assert row.validation == expected.get(status[row.backend], status[row.backend]), (
                    f"{row.backend}: registry says {status[row.backend]}, row says {row.validation}"
                )


class TestTable:
    def test_data_column_is_present_and_populated(self):
        rendered = format_table(rank_value(_qc(), backends=CROSS_VENDOR, n_seeds=1))
        assert "Data" in rendered
        assert "validated" in rendered

    def test_unvalidated_is_shouted_not_whispered(self):
        # Upper case is deliberate: it has to be visible at a glance in a table of numbers.
        rendered = format_table(rank_value(_qc(), backends=["ionq_aria"], n_seeds=1))
        assert "UNVALIDATED" in rendered


class TestCli:
    def test_backend_option_selects_and_spans_vendors(self, circuit_file):
        res = CliRunner().invoke(cli, ["when", circuit_file, "-b", "ibm_fez", "-b", "ionq_aria"])
        assert res.exit_code == 0, res.output
        assert "ibm_fez" in res.output
        assert "ionq_aria" in res.output
        assert "iqm_garnet" not in res.output, "ranked a backend that was not requested"

    def test_json_emits_a_versioned_advice_receipt(self, circuit_file):
        res = CliRunner().invoke(
            cli, ["when", circuit_file, "-b", "ibm_fez", "-b", "ionq_aria", "--json"]
        )
        assert res.exit_code == 0, res.output
        payload = json.loads(res.output)
        assert payload["schema"] == "qb.cross_vendor_advice.v1"
        assert payload["shots"] == 4096
        assert len(payload["ranking"]) == 2

    def test_json_receipt_carries_provenance_per_row(self, circuit_file):
        """A stored receipt has to be readable later without the table's context."""
        res = CliRunner().invoke(
            cli, ["when", circuit_file, "-b", "ibm_fez", "-b", "ionq_aria", "--json"]
        )
        rows = {r["backend"]: r for r in json.loads(res.output)["ranking"]}
        assert rows["ibm_fez"]["validation"] == "validated"
        assert rows["ionq_aria"]["validation"] == "UNVALIDATED"

    def test_table_is_the_default(self, circuit_file):
        res = CliRunner().invoke(cli, ["when", circuit_file, "-b", "ibm_fez"])
        assert res.exit_code == 0
        assert "Backend" in res.output and "Data" in res.output
        assert not res.output.lstrip().startswith("{")

    def test_unknown_backend_does_not_crash_the_scan(self, circuit_file):
        res = CliRunner().invoke(
            cli, ["when", circuit_file, "-b", "ibm_fez", "-b", "not_a_real_backend"]
        )
        assert res.exit_code == 0, res.output


def _qc():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    return qc
