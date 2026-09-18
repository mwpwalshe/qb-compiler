# SPDX-License-Identifier: Apache-2.0
"""The feed schema and its signature.

Every key here is a throwaway generated in the test. The production key is never read, never
needed, and nothing in this file touches the network.
"""

from __future__ import annotations

import base64
import json
import os

import pytest

from qb_compiler.cost.billing import PerSecond, PerShot
from qb_compiler.cost.pricing_feed import (
    FEED_SCHEMA,
    FeedError,
    PricingEntry,
    build_feed,
    canonical_feed_bytes,
    parse_feed,
    parse_feed_text,
    sign_feed,
    verify_feed,
)
from qb_compiler.signing import public_key_from_seed


def a_key(seed_byte: int = 7) -> tuple[bytes, bytes]:
    """A deterministic throwaway keypair. Not a secret, not the production key."""
    seed = bytes([seed_byte]) * 32
    return seed, public_key_from_seed(seed)


def some_entries() -> list[PricingEntry]:
    return [
        PricingEntry(
            backend="ibm_fez",
            provider="ibm",
            billing=PerSecond.from_per_shot(1.60, 0.00016),
            as_of="2026-09-17",
            source="a vendor page",
            notes="a conversion, not a vendor price",
        ),
        PricingEntry(
            backend="iqm_garnet",
            provider="iqm",
            billing=PerShot(0.00145, 0.30),
            as_of="2026-09-17",
            source="a vendor page",
        ),
    ]


class TestSchemaRoundTrip:
    def test_an_entry_survives_a_round_trip(self):
        for entry in some_entries():
            restored = PricingEntry.from_dict(entry.as_dict())
            assert restored == entry
            assert restored.cost_per_shot_usd == entry.cost_per_shot_usd

    def test_a_feed_parses_back_to_its_entries(self):
        seed, public = a_key()
        feed = sign_feed(build_feed(some_entries()), seed)
        parsed = parse_feed(feed, public)
        assert parsed.signature_verified
        assert sorted(parsed.entries) == ["ibm_fez", "iqm_garnet"]
        assert parsed.entries["ibm_fez"].billing.cost_per_shot_usd == 0.00016

    def test_the_schema_is_declared_and_checked(self):
        feed = build_feed(some_entries())
        assert feed["schema"] == FEED_SCHEMA
        with pytest.raises(FeedError, match="feed schema is"):
            parse_feed({"schema": "qb.something_else.v1", "entries": []})

    def test_an_entry_missing_a_field_is_named(self):
        with pytest.raises(FeedError, match="missing 'as_of'"):
            PricingEntry.from_dict(
                {
                    "backend": "x",
                    "provider": "y",
                    "billing": {
                        "model": "per_shot",
                        "cost_per_shot_usd": 1.0,
                        "cost_per_task_usd": 0.0,
                    },
                    "source": "z",
                }
            )

    def test_unreadable_json_is_refused(self):
        with pytest.raises(FeedError, match="not readable JSON"):
            parse_feed_text("{not json")


class TestCanonicalBytes:
    def test_key_order_and_whitespace_do_not_change_the_bytes(self):
        feed = build_feed(some_entries())
        shuffled = json.loads(json.dumps(feed))
        reordered = {key: shuffled[key] for key in sorted(shuffled, reverse=True)}
        spaced = json.loads(json.dumps(feed, indent=4))
        assert canonical_feed_bytes(reordered) == canonical_feed_bytes(feed)
        assert canonical_feed_bytes(spaced) == canonical_feed_bytes(feed)

    def test_the_signature_block_is_not_part_of_the_bytes(self):
        seed, _ = a_key()
        body = build_feed(some_entries())
        signed = sign_feed(body, seed)
        assert canonical_feed_bytes(signed) == canonical_feed_bytes(body)

    def test_entries_are_sorted_so_the_bytes_are_stable(self):
        stamp = "2026-09-18T00:00:00+00:00"
        first = build_feed(some_entries(), generated_at=stamp)
        second = build_feed(list(reversed(some_entries())), generated_at=stamp)
        assert canonical_feed_bytes(first) == canonical_feed_bytes(second)


class TestSignature:
    def test_a_signed_feed_verifies(self):
        seed, public = a_key()
        feed = sign_feed(build_feed(some_entries()), seed)
        ok, reason = verify_feed(feed, public)
        assert ok, reason
        assert feed["signature"]["alg"] == "ed25519"

    def test_a_tampered_byte_is_caught(self):
        seed, public = a_key()
        feed = sign_feed(build_feed(some_entries()), seed)
        feed["entries"][0]["billing"]["usd_per_second"] = 0.01
        ok, reason = verify_feed(feed, public)
        assert not ok
        assert "does not verify" in reason

    def test_the_wrong_key_is_caught(self):
        seed, _ = a_key(7)
        _, other_public = a_key(9)
        feed = sign_feed(build_feed(some_entries()), seed)
        ok, reason = verify_feed(feed, other_public)
        assert not ok
        assert "signed by key" in reason

    def test_a_missing_signature_is_caught(self):
        _, public = a_key()
        ok, reason = verify_feed(build_feed(some_entries()), public)
        assert not ok
        assert "no signature" in reason

    def test_a_signature_that_is_not_base64_is_caught(self):
        seed, public = a_key()
        feed = sign_feed(build_feed(some_entries()), seed)
        feed["signature"]["sig"] = "not base64 at all"
        ok, reason = verify_feed(feed, public)
        assert not ok
        assert "base64" in reason

    def test_another_algorithm_is_refused(self):
        seed, public = a_key()
        feed = sign_feed(build_feed(some_entries()), seed)
        feed["signature"]["alg"] = "rsa"
        ok, reason = verify_feed(feed, public)
        assert not ok
        assert "algorithm" in reason

    def test_parsing_without_a_key_does_not_claim_a_verification(self):
        seed, _ = a_key()
        parsed = parse_feed(sign_feed(build_feed(some_entries()), seed))
        assert not parsed.signature_verified
        assert "no public key" in parsed.reason


class TestShippedKey:
    def test_the_shipped_key_is_a_usable_ed25519_key(self):
        from qb_compiler.cost.pricing_feed_pubkey import (
            FEED_KEY_FINGERPRINT,
            FEED_PUBLIC_KEY_B64,
            feed_public_key,
        )
        from qb_compiler.signing import public_key_fingerprint

        raw = feed_public_key()
        assert len(raw) == 32
        assert base64.b64decode(FEED_PUBLIC_KEY_B64, validate=True) == raw
        assert public_key_fingerprint(raw) == FEED_KEY_FINGERPRINT

    def test_no_private_key_material_is_in_the_package(self):
        """The shipped module holds a public key and nothing else."""
        from qb_compiler.cost import pricing_feed_pubkey

        source = os.fsdecode(pricing_feed_pubkey.__file__)
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        assert "private" not in text.lower().replace("private half", "")
