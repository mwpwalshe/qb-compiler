# SPDX-License-Identifier: Apache-2.0
"""Tendrils: declarations, the conformance check, plugin discovery and the two commands."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from click.testing import CliRunner

from qb_compiler.cli.main import cli
from qb_compiler.tendrils import (
    CAPABILITIES,
    CONFORMS,
    DOES_NOT_CONFORM,
    PriceRow,
    Tendril,
    TendrilError,
    ValidationRecord,
    check,
    conformance_receipt,
    discover,
    manifest_sha256,
    verdict,
)
from qb_compiler.tendrils import discovery as disc
from qb_compiler.tendrils.builtin import IBM_VALIDATION, builtin_tendrils, round_trip

TODAY = date(2026, 11, 8)
SHA = "a" * 64


def _hardware(**over: object) -> Tendril:
    base = Tendril(
        name="acme",
        kind="hardware",
        vendor="Acme Quantum",
        maintainer="Acme",
        contact="https://acme.example/support",
        adapter_version="1.0.0",
        capabilities={c: c in ("circuits", "price") for c in CAPABILITIES},
        label="UNVALIDATED",
        backends=("acme_one",),
        price_rows=(
            PriceRow("acme_one", "per_shot", 0.001, "USD", "2026-11-01", "https://acme.example/p"),
        ),
    )
    return replace(base, **over)  # type: ignore[arg-type]


def _record(**over: object) -> ValidationRecord:
    base = ValidationRecord("2026-10-01", "acme_one", "job-1", SHA, "results/acme.json")
    return replace(base, **over)  # type: ignore[arg-type]


def _failed(t: Tendril) -> set[str]:
    return {c.name for c in check(t, today=TODAY) if c.passed is False}


# ── built-ins ───────────────────────────────────────────────────────


class TestBuiltins:
    def test_every_builtin_conforms(self) -> None:
        for t in builtin_tendrils():
            assert verdict(check(t, today=TODAY)) == CONFORMS, (t.name, _failed(t))

    def test_names_unique_and_kinds_present(self) -> None:
        names = [t.name for t in builtin_tendrils()]
        assert len(names) == len(set(names))
        kinds = {t.kind for t in builtin_tendrils()}
        assert {"hardware", "format"} <= kinds

    def test_only_a_named_record_makes_validated(self) -> None:
        for t in builtin_tendrils():
            if t.label == "validated":
                assert t.validation is not None
        ibm = next(t for t in builtin_tendrils() if t.name == "ibm")
        assert ibm.validation == IBM_VALIDATION
        assert ibm.validated_until() == "2026-06-12"
        assert ibm.validation_state(TODAY) == "expired"
        assert ibm.validation_state(date(2026, 5, 1)) == "current"

    def test_ibm_record_hash_matches_the_committed_file(self) -> None:
        import hashlib

        record = Path(__file__).resolve().parents[2] / "results" / "hardware_validation_dd.json"
        if not record.exists():
            pytest.skip("results/ not present in this checkout")
        assert hashlib.sha256(record.read_bytes()).hexdigest() == IBM_VALIDATION.raw_sha256
        assert IBM_VALIDATION.job_id in record.read_text(encoding="utf-8")

    def test_hardware_labels_match_qbc_when(self) -> None:
        pytest.importorskip("qiskit")
        from qiskit import QuantumCircuit

        from qb_compiler.windows import rank_value

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        hardware = {t.name: t for t in builtin_tendrils() if t.kind == "hardware"}
        picks = [t.backends[0] for t in hardware.values()]
        rows = {r.backend: r for r in rank_value(qc, picks, n_seeds=1)}
        for t in hardware.values():
            row = rows.get(t.backends[0])
            if row is not None and row.validation != "unknown":
                assert row.validation == t.label, t.name

    def test_round_trip_openqasm2_needs_nothing_installed(self) -> None:
        c = round_trip("openqasm2")
        assert c.passed is True

    def test_round_trip_unknown_name_only_notes(self) -> None:
        assert round_trip("ibm").passed is None

    def test_tendrils_json_matches_the_builtins(self) -> None:
        listing = Path(__file__).resolve().parents[2] / "TENDRILS.json"
        if not listing.exists():
            pytest.skip("TENDRILS.json not present in this checkout")
        rows = json.loads(listing.read_text(encoding="utf-8"))["tendrils"]
        want = [
            {
                "name": t.name,
                "kind": t.kind,
                "maintainer": t.maintainer,
                "label": t.label,
                "validated_until": t.validated_until(),
                "receipt_sha256": None,
            }
            for t in builtin_tendrils()
        ]
        assert [r for r in rows if r.get("origin") == "builtin"] == [
            {**w, "origin": "builtin"} for w in want
        ]


# ── conformance ─────────────────────────────────────────────────────


class TestConformance:
    def test_good_declaration_conforms(self) -> None:
        assert _failed(_hardware()) == set()

    def test_validated_without_record_fails(self) -> None:
        assert "label" in _failed(_hardware(label="validated"))

    def test_validated_with_good_record_passes(self) -> None:
        t = _hardware(label="validated", validation=_record())
        assert _failed(t) == set()
        assert t.validation_state(TODAY) == "current"

    @pytest.mark.parametrize(
        "bad",
        [
            {"raw_sha256": "xyz"},
            {"raw_sha256": "A" * 64},
            {"date": "2027-01-01"},
            {"date": "1 Oct 2026"},
            {"job_id": "  "},
            {"backend": "someone_else"},
            {"source": ""},
        ],
    )
    def test_bad_record_fails_label(self, bad: dict[str, str]) -> None:
        assert "label" in _failed(_hardware(label="validated", validation=_record(**bad)))

    def test_record_expires_after_ninety_days(self) -> None:
        t = _hardware(label="validated", validation=_record(date="2026-08-01"))
        assert t.validated_until() == "2026-10-30"
        assert t.validation_state(TODAY) == "expired"
        assert _failed(t) == set()

    def test_unknown_label_fails(self) -> None:
        assert "label" in _failed(_hardware(label="trusted"))

    def test_missing_and_unknown_capabilities_fail(self) -> None:
        caps = {c: False for c in CAPABILITIES if c != "execute"}
        assert "capabilities" in _failed(_hardware(capabilities=caps))
        extra = {**{c: False for c in CAPABILITIES}, "teleport": True}
        assert "capabilities" in _failed(_hardware(capabilities=extra))

    def test_hardware_must_list_prices_or_declare_none(self) -> None:
        caps = {c: c == "circuits" for c in CAPABILITIES}
        assert "price" in _failed(_hardware(capabilities=caps, price_rows=()))
        assert "price" not in _failed(
            _hardware(capabilities=caps, price_rows=(), no_public_price=True)
        )

    @pytest.mark.parametrize(
        "row",
        [
            PriceRow("acme_one", "per_shot", -1.0, "USD", "2026-11-01", "src"),
            PriceRow("acme_one", "per_shot", math.nan, "USD", "2026-11-01", "src"),
            PriceRow("acme_one", "per_shot", math.inf, "USD", "2026-11-01", "src"),
            PriceRow("acme_one", "per_shot", None, "USD", "2026-11-01", "src"),
            PriceRow("acme_one", "per_shot", 0.1, "usd", "2026-11-01", "src"),
            PriceRow("acme_one", "per_shot", 0.1, "USD", "2027-11-01", "src"),
            PriceRow("acme_one", "per_shot", 0.1, "USD", "soon", "src"),
            PriceRow("acme_one", "per_shot", 0.1, "USD", "2026-11-01", " "),
            PriceRow("acme_one", "per_gate", 0.1, "USD", "2026-11-01", "src"),
            PriceRow("acme_one", "formula", None, "USD", "2026-11-01", "src"),
            PriceRow("other", "per_shot", 0.1, "USD", "2026-11-01", "src"),
        ],
    )
    def test_bad_price_row_fails(self, row: PriceRow) -> None:
        assert "price" in _failed(_hardware(price_rows=(row,)))

    def test_formula_row_with_formula_passes(self) -> None:
        row = PriceRow("acme_one", "formula", None, "USD", "2026-11-01", "src", note="5 + N/5000")
        assert "price" not in _failed(_hardware(price_rows=(row,)))

    def test_calibration_must_match_capability(self) -> None:
        assert "calibration" in _failed(_hardware(calibration="live"))
        caps = {c: c in ("circuits", "price", "calibration") for c in CAPABILITIES}
        assert "calibration" in _failed(_hardware(capabilities=caps, calibration="none"))
        assert "calibration" in _failed(_hardware(capabilities=caps, calibration="snapshot"))
        ok = _hardware(capabilities=caps, calibration="snapshot", calibration_as_of="2026-10-01")
        assert "calibration" not in _failed(ok)

    @pytest.mark.parametrize("name", ["", "Acme", "acme quantum", "acme/one", "é", "a" * 65])
    def test_bad_names_fail_identity(self, name: str) -> None:
        assert "identity" in _failed(_hardware(name=name))

    def test_hardware_without_backends_fails_identity(self) -> None:
        assert "identity" in _failed(_hardware(backends=(), price_rows=(), no_public_price=True))

    def test_receipt_shape_and_stable_hash(self) -> None:
        t = _hardware()
        receipt = conformance_receipt(t, check(t, today=TODAY))
        assert receipt["schema"] == "qb.tendril_conformance.v1"
        assert receipt["verdict"] == CONFORMS
        assert receipt["manifest_sha256"] == manifest_sha256(replace(t, origin="elsewhere"))
        bad = _hardware(label="validated")
        assert conformance_receipt(bad, check(bad, today=TODAY))["verdict"] == DOES_NOT_CONFORM

    def test_schema_claims_are_registered(self) -> None:
        from qb_compiler.signing import _SCHEMA_CLAIMS

        assert "qb.tendril_conformance.v1" in _SCHEMA_CLAIMS


# ── reading declarations ────────────────────────────────────────────


class TestFromDict:
    def test_round_trips(self) -> None:
        t = _hardware(label="validated", validation=_record())
        back = Tendril.from_dict(json.loads(json.dumps(t.to_dict())))
        assert back == t

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda d: [],
            lambda d: "acme",
            lambda d: {**d, "name": 7},
            lambda d: {**d, "capabilities": ["circuits"]},
            lambda d: {**d, "capabilities": {"circuits": "yes"}},
            lambda d: {**d, "backends": "acme_one"},
            lambda d: {**d, "price_rows": {}},
            lambda d: {**d, "price_rows": [{"amount": True}]},
            lambda d: {**d, "price_rows": [{**d["price_rows"][0], "amount": "0.1"}]},
            lambda d: {**d, "validation": "job-1"},
            lambda d: {**d, "validation": {"date": "2026-10-01"}},
            lambda d: {**d, "no_public_price": "no"},
            lambda d: {**d, "calibration_as_of": 20261001},
        ],
    )
    def test_wrong_shapes_raise(self, mutate: object) -> None:
        d = _hardware().to_dict()
        with pytest.raises(TendrilError):
            Tendril.from_dict(mutate(d))  # type: ignore[operator]


# ── discovery ───────────────────────────────────────────────────────


class _EP:
    def __init__(self, name: str, obj: object, dist: str = "acme-qbc") -> None:
        self.name = name
        self._obj = obj
        self.dist = type("D", (), {"name": dist})()

    def load(self) -> object:
        if isinstance(self._obj, BaseException):
            raise self._obj
        return self._obj


class TestDiscovery:
    def test_plugins_load_and_faults_are_listed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        good = _hardware(name="acme")
        eps = [
            _EP("acme", good),
            _EP("acme_dict", _hardware(name="acme_dict").to_dict()),
            _EP("acme_fn", lambda: _hardware(name="acme_fn")),
            _EP("crashes", ImportError("no module named acme_sdk")),
            _EP("shape", {"name": "shape"}),
            _EP("taken", _hardware(name="ibm")),
        ]
        monkeypatch.setattr(disc, "entry_points", lambda group: eps)
        found, broken = discover()
        names = {t.name for t in found}
        assert {"acme", "acme_dict", "acme_fn"} <= names
        assert {b.name for b in broken} == {"crashes", "shape", "taken"}
        assert next(t for t in found if t.name == "acme").origin == "plugin:acme-qbc"
        assert sum(t.name == "ibm" for t in found) == 1

    def test_no_plugins(self) -> None:
        found, broken = discover(include_plugins=False)
        assert broken == []
        assert [t.name for t in found] == [t.name for t in builtin_tendrils()]


# ── commands ────────────────────────────────────────────────────────


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def manifest(tmp_path: Path) -> Path:
    path = tmp_path / "acme.json"
    path.write_text(json.dumps(_hardware().to_dict()), encoding="utf-8")
    return path


class TestCommands:
    def test_tendrils_lists_builtins(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["tendrils"])
        assert result.exit_code == 0
        assert "ibm" in result.output and "openqasm2" in result.output

    def test_tendrils_json(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["tendrils", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        ibm = next(t for t in payload["tendrils"] if t["name"] == "ibm")
        assert ibm["label"] == "validated" and ibm["validated_until"] == "2026-06-12"

    def test_check_builtin(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["tendril", "check", "openqasm2"])
        assert result.exit_code == 0
        assert "round_trip" in result.output and "CONFORMS" in result.output

    def test_check_file_conforms(self, runner: CliRunner, manifest: Path) -> None:
        result = runner.invoke(cli, ["tendril", "check", "--file", str(manifest)])
        assert result.exit_code == 0, result.output

    def test_check_file_does_not_conform(self, runner: CliRunner, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(_hardware(label="validated").to_dict()), encoding="utf-8")
        result = runner.invoke(cli, ["tendril", "check", "--file", str(bad)])
        assert result.exit_code == 2
        assert "DOES_NOT_CONFORM" in result.output

    @pytest.mark.parametrize(
        "content",
        [b"{not json", b"\xff\xfe\x00garbage", b"[1, 2]", b'{"name": "x"}', b""],
    )
    def test_unreadable_file_exits_3(
        self, runner: CliRunner, tmp_path: Path, content: bytes
    ) -> None:
        path = tmp_path / "hostile.json"
        path.write_bytes(content)
        result = runner.invoke(cli, ["tendril", "check", "--file", str(path)])
        assert result.exit_code == 3
        assert "could not read" in result.output

    def test_name_and_file_together_exit_3(self, runner: CliRunner, manifest: Path) -> None:
        result = runner.invoke(cli, ["tendril", "check", "ibm", "--file", str(manifest)])
        assert result.exit_code == 3

    def test_neither_exits_3(self, runner: CliRunner) -> None:
        assert runner.invoke(cli, ["tendril", "check"]).exit_code == 3

    def test_unknown_name_exits_3(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["tendril", "check", "nosuch"])
        assert result.exit_code == 3
        assert "Known:" in result.output

    def test_missing_key_exits_3(self, runner: CliRunner, manifest: Path, tmp_path: Path) -> None:
        missing = tmp_path / "no_such_key"
        result = runner.invoke(
            cli, ["tendril", "check", "--file", str(manifest), "--sign", "--key", str(missing)]
        )
        assert result.exit_code == 3
        assert not missing.exists()

    def test_signed_receipt_verifies_offline(
        self, runner: CliRunner, manifest: Path, tmp_path: Path
    ) -> None:
        from qb_compiler.signing import (
            VERIFIED,
            load_or_create_signing_key,
            verify_receipt,
        )

        key_file = tmp_path / "submitter_key"
        key = load_or_create_signing_key(key_file)
        out = tmp_path / "receipt.json"
        result = runner.invoke(
            cli,
            [
                "tendril", "check", "--file", str(manifest),
                "--sign", "--key", str(key_file), "--out", str(out),
            ],
        )  # fmt: skip
        assert result.exit_code == 0, result.output
        receipt = json.loads(out.read_text(encoding="utf-8"))
        assert receipt["key_fingerprint"] == key.fingerprint
        assert verify_receipt(receipt, public_key=key.public).status == VERIFIED
        receipt["verdict"] = DOES_NOT_CONFORM
        assert verify_receipt(receipt, public_key=key.public).status != VERIFIED
