"""The signed pricing feed: one schema, one signature, and a refusal when it does not check out.

A price that decides a spend is worth as much as its provenance. The feed carries every entry with
the date it was checked and where it came from, and an Ed25519 signature over the canonical JSON of
everything except the signature itself. A feed whose signature does not verify is **refused**, not
used with a warning: a price nobody can vouch for is worse than a stale price that says its age.

Schema ``qb.pricing_feed.v1``::

    {
      "schema": "qb.pricing_feed.v1",
      "generated_at": "2026-09-18T07:00:00+00:00",
      "entries": [ {"backend": ..., "provider": ..., "billing": {...},
                    "as_of": ..., "source": ..., "notes": ...}, ... ],
      "signature": {"alg": "ed25519", "key_id": "<fingerprint>", "sig": "<base64>"}
    }

Canonical means sorted keys, no insignificant whitespace, UTF-8, which is the same rule
:func:`qb_compiler.signing.canonical_payload` uses for receipts. Reordering the entries changes
the bytes and breaks the signature, which is the point.

Nothing here fetches. :mod:`qb_compiler.cost.pricing_provider` decides when a feed is read and
from where.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from qb_compiler.cost.billing import BillingModel, billing_as_dict, billing_from_dict

FEED_SCHEMA = "qb.pricing_feed.v1"
SIGNATURE_ALGORITHM = "ed25519"


class FeedError(ValueError):
    """Raised when a feed is not a feed: wrong schema, missing field, unreadable billing model."""


@dataclass(frozen=True, slots=True)
class PricingEntry:
    """One backend's price, with its billing model and its provenance.

    Attributes
    ----------
    backend :
        Backend identifier, e.g. ``"ibm_fez"``.
    provider :
        Vendor name.
    billing :
        How this vendor bills: see :mod:`qb_compiler.cost.billing`.
    as_of :
        ISO-8601 date the price was last checked against the vendor.
    source :
        Where it was read, in words.
    notes :
        Anything a reader needs before quoting the number.
    """

    backend: str
    provider: str
    billing: BillingModel
    as_of: str
    source: str
    notes: str = ""

    @property
    def cost_per_shot_usd(self) -> float:
        """The per-shot figure this entry implies, for the callers that only want a float.

        For a vendor that does not sell shots this is a conversion or an approximation, and the
        breakdown from :meth:`job_cost` is where the assumptions behind it are written down.
        """
        billing = self.billing
        if hasattr(billing, "cost_per_shot_usd"):
            return float(billing.cost_per_shot_usd)
        fallback = getattr(billing, "fallback", None)
        if fallback is not None:
            return float(fallback.cost_per_shot_usd)
        raise FeedError(
            f"{self.backend} bills {billing.name} and carries no per-shot fallback, so it has no "
            "per-shot number. Use job_cost() with the circuit's counts"
        )

    def job_cost(
        self,
        shots: int,
        *,
        tasks: int = 1,
        one_qubit_gates: int | None = None,
        two_qubit_gates: int | None = None,
        measurements: int | None = None,
        status: str = "static",
    ) -> Any:
        """Price a job on this entry.

        Returns a :class:`~qb_compiler.cost.billing.CostBreakdown`: the number and what it
        assumed, never a bare float.
        """
        return self.billing.job_cost(
            shots,
            tasks=tasks,
            one_qubit_gates=one_qubit_gates,
            two_qubit_gates=two_qubit_gates,
            measurements=measurements,
            as_of=self.as_of,
            source=self.source,
            status=status,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "provider": self.provider,
            "billing": billing_as_dict(self.billing),
            "as_of": self.as_of,
            "source": self.source,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PricingEntry:
        try:
            return cls(
                backend=str(payload["backend"]),
                provider=str(payload["provider"]),
                billing=billing_from_dict(dict(payload["billing"])),
                as_of=str(payload["as_of"]),
                source=str(payload["source"]),
                notes=str(payload.get("notes", "")),
            )
        except KeyError as exc:
            raise FeedError(f"pricing entry is missing {exc.args[0]!r}") from exc
        except (TypeError, ValueError) as exc:
            raise FeedError(f"pricing entry is unreadable: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ParsedFeed:
    """A feed that parsed, with whether its signature checked out and against which key."""

    generated_at: str
    entries: dict[str, PricingEntry]
    signature_verified: bool
    key_id: str = ""
    reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def canonical_feed_bytes(feed: dict[str, Any]) -> bytes:
    """The bytes a feed signature covers: the feed minus its signature, as canonical JSON."""
    body = {key: value for key, value in feed.items() if key != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def build_feed(entries: list[PricingEntry], *, generated_at: str | None = None) -> dict[str, Any]:
    """An unsigned feed body from *entries*, entries sorted by backend so the bytes are stable."""
    stamp = generated_at or datetime.now(timezone.utc).isoformat()
    return {
        "schema": FEED_SCHEMA,
        "generated_at": stamp,
        "entries": [entry.as_dict() for entry in sorted(entries, key=lambda e: e.backend)],
    }


def sign_feed(feed: dict[str, Any], seed: bytes, *, key_id: str | None = None) -> dict[str, Any]:
    """Return a copy of *feed* carrying an Ed25519 signature over its canonical bytes."""
    from qb_compiler.signing import public_key_fingerprint, public_key_from_seed, sign_bytes

    body = {key: value for key, value in feed.items() if key != "signature"}
    signature = sign_bytes(seed, canonical_feed_bytes(body))
    public = public_key_from_seed(seed)
    signed = dict(body)
    signed["signature"] = {
        "alg": SIGNATURE_ALGORITHM,
        "key_id": key_id or public_key_fingerprint(public),
        "sig": base64.b64encode(signature).decode("ascii"),
    }
    return signed


def verify_feed(feed: dict[str, Any], public_key: bytes) -> tuple[bool, str]:
    """``(verified, reason)`` for *feed* against *public_key*. Never raises on a bad feed."""
    from qb_compiler.signing import public_key_fingerprint, verify_bytes

    signature = feed.get("signature")
    if not isinstance(signature, dict):
        return False, "the feed carries no signature block"
    if signature.get("alg") != SIGNATURE_ALGORITHM:
        return False, f"signature algorithm is {signature.get('alg')!r}, expected ed25519"
    raw = signature.get("sig")
    if not raw:
        return False, "the signature block carries no signature"
    try:
        blob = base64.b64decode(str(raw), validate=True)
    except Exception:
        return False, "the signature is not base64"
    key_id = str(signature.get("key_id", ""))
    expected = public_key_fingerprint(public_key)
    if key_id and key_id != expected:
        return False, f"the feed is signed by key {key_id}, and the key here is {expected}"
    if not verify_bytes(public_key, canonical_feed_bytes(feed), blob):
        return False, "the signature does not verify against this key"
    return True, "signature verified"


def parse_feed(payload: dict[str, Any], public_key: bytes | None = None) -> ParsedFeed:
    """Read a feed, checking the schema and, when a key is given, the signature.

    A bad schema or an unreadable entry raises :class:`FeedError`. A signature that does not
    verify does not raise here; it comes back as ``signature_verified=False`` with the reason, and
    the provider is the one that refuses to serve it.
    """
    if not isinstance(payload, dict):
        raise FeedError(f"a feed must be a JSON object, got {type(payload).__name__}")
    schema = payload.get("schema")
    if schema != FEED_SCHEMA:
        raise FeedError(f"feed schema is {schema!r}, expected {FEED_SCHEMA!r}")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise FeedError("a feed needs an 'entries' list")
    entries = {}
    for item in raw_entries:
        if not isinstance(item, dict):
            raise FeedError(f"a feed entry must be an object, got {type(item).__name__}")
        entry = PricingEntry.from_dict(item)
        entries[entry.backend] = entry
    verified, reason = (False, "no public key was supplied to check against")
    if public_key is not None:
        verified, reason = verify_feed(payload, public_key)
    signature = payload.get("signature")
    key_id = str(signature.get("key_id", "")) if isinstance(signature, dict) else ""
    return ParsedFeed(
        generated_at=str(payload.get("generated_at", "")),
        entries=entries,
        signature_verified=verified,
        key_id=key_id,
        reason=reason,
        raw=payload,
    )


def parse_feed_text(text: str, public_key: bytes | None = None) -> ParsedFeed:
    """:func:`parse_feed` on JSON text. Unparseable JSON raises :class:`FeedError`."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FeedError(f"the feed is not readable JSON: {exc}") from exc
    return parse_feed(payload, public_key)


__all__ = [
    "FEED_SCHEMA",
    "SIGNATURE_ALGORITHM",
    "FeedError",
    "ParsedFeed",
    "PricingEntry",
    "build_feed",
    "canonical_feed_bytes",
    "parse_feed",
    "parse_feed_text",
    "sign_feed",
    "verify_feed",
]
