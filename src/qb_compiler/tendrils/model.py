# SPDX-License-Identifier: Apache-2.0
"""What a tendril declares about itself.

A tendril is one place an experiment can go, or one format or toolkit it can come from: a
hardware vendor, a circuit format, a QEC toolkit. Each one declares what it supports, where its
prices come from and how far its numbers have been checked against real hardware. Nothing is
inferred: a capability that is not supported is declared as not supported, and a price that is
not public is declared as not public.

The four labels are the same ones ``qbc when`` prints, with the same meaning:

``validated``
    The adapter has run a real job on that vendor's hardware, and the record of that job is named:
    date, backend, job id and the sha256 of the raw result.
``UNVALIDATED``
    An adapter exists and has not been run against real hardware.
``fixture-only``
    Exercised on recorded fixtures only, never against a live vendor.
``no-adapter``
    Listed so the gap is visible. Nothing runs.

A validation record covers :data:`VALIDATION_DAYS` days from its date. After that the label stays
as declared and the listing says the record has expired, so a reader can see both facts.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

CAPABILITIES: tuple[str, ...] = ("circuits", "qec_results", "price", "calibration", "execute")
LABELS: tuple[str, ...] = ("validated", "UNVALIDATED", "fixture-only", "no-adapter")
KINDS: tuple[str, ...] = ("hardware", "format", "toolkit")
PRICE_UNITS: tuple[str, ...] = (
    "per_shot",
    "per_task",
    "per_second",
    "per_minute",
    "formula",
    "subscription",
)
CALIBRATION_KINDS: tuple[str, ...] = ("live", "snapshot", "none")
VALIDATION_DAYS = 90

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


class TendrilError(ValueError):
    """A tendril declaration that cannot be read at all."""


def parse_date(value: object) -> date | None:
    """An ISO date (YYYY-MM-DD), or None when *value* is not one."""
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class PriceRow:
    """One published price, with the date it was read and where from."""

    backend: str
    unit: str
    amount: float | None
    currency: str
    as_of: str
    source: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "unit": self.unit,
            "amount": self.amount,
            "currency": self.currency,
            "as_of": self.as_of,
            "source": self.source,
            "note": self.note,
        }


@dataclass(frozen=True)
class ValidationRecord:
    """The real job a ``validated`` label rests on."""

    date: str
    backend: str
    job_id: str
    raw_sha256: str
    source: str
    invoice_checked: bool = False

    def until(self) -> _dt.date | None:
        start = parse_date(self.date)
        return start + timedelta(days=VALIDATION_DAYS) if start is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "backend": self.backend,
            "job_id": self.job_id,
            "raw_sha256": self.raw_sha256,
            "source": self.source,
            "invoice_checked": self.invoice_checked,
        }


@dataclass(frozen=True)
class Tendril:
    """One tendril's declaration. Build it in code, or read it with :meth:`from_dict`."""

    name: str
    kind: str
    vendor: str
    maintainer: str
    contact: str
    adapter_version: str
    capabilities: Mapping[str, bool]
    label: str
    backends: tuple[str, ...] = ()
    formats: tuple[str, ...] = ()
    calibration: str = "none"
    calibration_as_of: str | None = None
    price_rows: tuple[PriceRow, ...] = ()
    no_public_price: bool = False
    validation: ValidationRecord | None = None
    origin: str = field(default="builtin", compare=False)

    def validated_until(self) -> str | None:
        """Last day the validation record covers, or None without a dated record."""
        if self.validation is None:
            return None
        end = self.validation.until()
        return end.isoformat() if end is not None else None

    def validation_state(self, today: date | None = None) -> str:
        """``current``, ``expired`` or ``none``: whether a dated record covers *today*."""
        end = self.validation.until() if self.validation is not None else None
        if end is None:
            return "none"
        return "current" if (today or date.today()) <= end else "expired"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "vendor": self.vendor,
            "maintainer": self.maintainer,
            "contact": self.contact,
            "adapter_version": self.adapter_version,
            "capabilities": {k: bool(self.capabilities.get(k, False)) for k in CAPABILITIES},
            "label": self.label,
            "backends": list(self.backends),
            "formats": list(self.formats),
            "calibration": self.calibration,
            "calibration_as_of": self.calibration_as_of,
            "price_rows": [p.to_dict() for p in self.price_rows],
            "no_public_price": self.no_public_price,
            "validation": self.validation.to_dict() if self.validation is not None else None,
            "validated_until": self.validated_until(),
        }

    @classmethod
    def from_dict(cls, data: object, origin: str = "file") -> Tendril:
        """Read a declaration. Raises :class:`TendrilError` when the shape is wrong.

        Shape only: whether the values make sense is the conformance check's job, so a
        declaration with an unknown label still loads and then fails that check by name.
        """
        if not isinstance(data, Mapping):
            raise TendrilError("a tendril declaration must be a JSON object")

        def text(key: str, default: str | None = None) -> str:
            value = data.get(key, default)
            if not isinstance(value, str):
                raise TendrilError(f"field {key!r} must be a string")
            return value

        def strings(key: str) -> tuple[str, ...]:
            value = data.get(key, [])
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise TendrilError(f"field {key!r} must be a list of strings")
            return tuple(value)

        caps = data.get("capabilities")
        if not isinstance(caps, Mapping) or not all(isinstance(k, str) for k in caps):
            raise TendrilError("field 'capabilities' must be an object")
        if not all(isinstance(v, bool) for v in caps.values()):
            raise TendrilError("every capability must be true or false")

        rows = data.get("price_rows", [])
        if not isinstance(rows, list):
            raise TendrilError("field 'price_rows' must be a list")
        prices = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise TendrilError("every price row must be an object")
            amount = row.get("amount")
            if amount is not None and (
                isinstance(amount, bool) or not isinstance(amount, int | float)
            ):
                raise TendrilError("a price amount must be a number or null")
            fields = {}
            for key in ("backend", "unit", "currency", "as_of", "source"):
                value = row.get(key)
                if not isinstance(value, str):
                    raise TendrilError(f"price row field {key!r} must be a string")
                fields[key] = value
            note = row.get("note", "")
            if not isinstance(note, str):
                raise TendrilError("price row field 'note' must be a string")
            prices.append(
                PriceRow(amount=float(amount) if amount is not None else None, note=note, **fields)
            )

        record = data.get("validation")
        validation = None
        if record is not None:
            if not isinstance(record, Mapping):
                raise TendrilError("field 'validation' must be an object or null")
            vals = {}
            for key in ("date", "backend", "job_id", "raw_sha256", "source"):
                value = record.get(key)
                if not isinstance(value, str):
                    raise TendrilError(f"validation field {key!r} must be a string")
                vals[key] = value
            invoice = record.get("invoice_checked", False)
            if not isinstance(invoice, bool):
                raise TendrilError("validation field 'invoice_checked' must be true or false")
            validation = ValidationRecord(invoice_checked=invoice, **vals)

        as_of = data.get("calibration_as_of")
        if as_of is not None and not isinstance(as_of, str):
            raise TendrilError("field 'calibration_as_of' must be a string or null")
        no_price = data.get("no_public_price", False)
        if not isinstance(no_price, bool):
            raise TendrilError("field 'no_public_price' must be true or false")

        return cls(
            name=text("name"),
            kind=text("kind"),
            vendor=text("vendor"),
            maintainer=text("maintainer"),
            contact=text("contact"),
            adapter_version=text("adapter_version"),
            capabilities=dict(caps),
            label=text("label"),
            backends=strings("backends"),
            formats=strings("formats"),
            calibration=text("calibration", "none"),
            calibration_as_of=as_of,
            price_rows=tuple(prices),
            no_public_price=no_price,
            validation=validation,
            origin=origin,
        )


def finite_non_negative(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value >= 0
