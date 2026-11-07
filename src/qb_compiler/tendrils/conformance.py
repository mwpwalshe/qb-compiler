# SPDX-License-Identifier: Apache-2.0
"""The conformance check every tendril runs, built in or not.

It checks what a tendril declares, not whether its hardware is any good. Every check has a name
and says what it found. The receipt it writes can be signed with the submitter's own key, so a
vendor can publish one for its own tendril without sending us any code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from qb_compiler.tendrils.model import (
    CALIBRATION_KINDS,
    CAPABILITIES,
    CURRENCY_RE,
    KINDS,
    LABELS,
    NAME_RE,
    PRICE_UNITS,
    SHA256_RE,
    Tendril,
    finite_non_negative,
    parse_date,
)

SCHEMA = "qb.tendril_conformance.v1"
CONFORMS = "CONFORMS"
DOES_NOT_CONFORM = "DOES_NOT_CONFORM"


@dataclass(frozen=True)
class Check:
    """One named check. ``passed`` is None for a check that only reports."""

    name: str
    passed: bool | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def _identity(t: Tendril) -> Check:
    problems = []
    if not NAME_RE.match(t.name):
        problems.append("name must be lower case letters, digits, '.', '_' or '-', 1 to 64 long")
    if t.kind not in KINDS:
        problems.append(f"kind {t.kind!r} is not one of {', '.join(KINDS)}")
    for key in ("vendor", "maintainer", "contact", "adapter_version"):
        if not getattr(t, key).strip():
            problems.append(f"{key} is empty")
    if t.kind == "hardware" and not t.backends:
        problems.append("a hardware tendril names at least one backend")
    if t.kind == "format" and not t.formats:
        problems.append("a format tendril names at least one format")
    return Check("identity", not problems, "; ".join(problems) or "complete")


def _capabilities(t: Tendril) -> Check:
    missing = [c for c in CAPABILITIES if c not in t.capabilities]
    extra = sorted(k for k in t.capabilities if k not in CAPABILITIES)
    problems = []
    if missing:
        problems.append("not declared: " + ", ".join(missing))
    if extra:
        problems.append("unknown: " + ", ".join(extra))
    supported = [c for c in CAPABILITIES if t.capabilities.get(c)]
    detail = "; ".join(problems) or ("supports " + (", ".join(supported) or "nothing"))
    return Check("capabilities", not problems, detail)


def _label(t: Tendril, today: date) -> Check:
    if t.label not in LABELS:
        return Check("label", False, f"{t.label!r} is not one of {', '.join(LABELS)}")
    if t.label != "validated":
        return Check("label", True, t.label)
    v = t.validation
    if v is None:
        return Check("label", False, "validated, but no validation record is named")
    problems = []
    when = parse_date(v.date)
    if when is None:
        problems.append("record date is not YYYY-MM-DD")
    elif when > today:
        problems.append("record date is in the future")
    if not v.job_id.strip():
        problems.append("record names no job id")
    if not SHA256_RE.match(v.raw_sha256):
        problems.append("record raw_sha256 is not 64 lower-case hex digits")
    if v.backend not in t.backends:
        problems.append(f"record backend {v.backend!r} is not one this tendril lists")
    if not v.source.strip():
        problems.append("record does not say where the raw result is")
    if not (t.capabilities.get("circuits") or t.capabilities.get("qec_results")):
        problems.append("validated, but runs neither circuits nor QEC results")
    return Check("label", not problems, "; ".join(problems) or f"validated on job {v.job_id}")


def _price(t: Tendril, today: date) -> Check:
    if not t.capabilities.get("price"):
        if t.no_public_price or t.kind != "hardware":
            return Check("price", True, "declared: no public price")
        return Check("price", False, "a hardware tendril either lists prices or declares none")
    if not t.price_rows:
        return Check("price", False, "price is supported but no price row is listed")
    problems = []
    for i, row in enumerate(t.price_rows):
        where = f"row {i} ({row.backend or '?'})"
        if row.unit not in PRICE_UNITS:
            problems.append(f"{where}: unit {row.unit!r} is not one of {', '.join(PRICE_UNITS)}")
        if row.unit == "formula":
            if not row.note.strip():
                problems.append(f"{where}: a formula price states its formula in the note")
        elif not finite_non_negative(row.amount):
            problems.append(f"{where}: amount must be a finite number, zero or more")
        if not CURRENCY_RE.match(row.currency):
            problems.append(f"{where}: currency must be a three-letter code")
        read = parse_date(row.as_of)
        if read is None:
            problems.append(f"{where}: as_of is not YYYY-MM-DD")
        elif read > today:
            problems.append(f"{where}: as_of is in the future")
        if not row.source.strip():
            problems.append(f"{where}: no source")
        if t.backends and row.backend not in t.backends:
            problems.append(f"{where}: backend is not one this tendril lists")
    oldest = min((r.as_of for r in t.price_rows if parse_date(r.as_of)), default="?")
    detail = "; ".join(problems) or f"{len(t.price_rows)} rows, oldest read {oldest}"
    return Check("price", not problems, detail)


def _calibration(t: Tendril, today: date) -> Check:
    if t.calibration not in CALIBRATION_KINDS:
        return Check(
            "calibration",
            False,
            f"{t.calibration!r} is not one of {', '.join(CALIBRATION_KINDS)}",
        )
    declared = bool(t.capabilities.get("calibration"))
    if declared != (t.calibration != "none"):
        return Check("calibration", False, "capability and calibration kind disagree")
    if t.calibration == "snapshot":
        taken = parse_date(t.calibration_as_of)
        if taken is None:
            return Check("calibration", False, "a snapshot names the date it was taken")
        if taken > today:
            return Check("calibration", False, "snapshot date is in the future")
        return Check("calibration", True, f"snapshot taken {t.calibration_as_of}")
    return Check("calibration", True, t.calibration)


def _validation_window(t: Tendril, today: date) -> Check:
    state = t.validation_state(today)
    if state == "none":
        return Check("validation_window", None, "no dated validation record")
    return Check(
        "validation_window",
        None,
        f"record of {t.validation.date if t.validation else '?'} is {state}, "
        f"covers to {t.validated_until()}",
    )


def check(
    tendril: Tendril, *, today: date | None = None, extra: list[Check] | None = None
) -> list[Check]:
    """Run every conformance check on *tendril*. *extra* adds checks a caller ran itself."""
    day = today or date.today()
    checks = [
        _identity(tendril),
        _capabilities(tendril),
        _label(tendril, day),
        _price(tendril, day),
        _calibration(tendril, day),
        _validation_window(tendril, day),
    ]
    return checks + list(extra or [])


def verdict(checks: list[Check]) -> str:
    return CONFORMS if all(c.passed is not False for c in checks) else DOES_NOT_CONFORM


def manifest_sha256(tendril: Tendril) -> str:
    payload = json.dumps(tendril.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def conformance_receipt(tendril: Tendril, checks: list[Check]) -> dict[str, Any]:
    """An unsigned receipt. Sign it with :func:`qb_compiler.signing.sign_receipt`."""
    from qb_compiler import __version__

    return {
        "schema": SCHEMA,
        "tendril": tendril.name,
        "origin": tendril.origin,
        "verdict": verdict(checks),
        "checks": [c.to_dict() for c in checks],
        "manifest": tendril.to_dict(),
        "manifest_sha256": manifest_sha256(tendril),
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "qb_compiler_version": __version__,
    }
