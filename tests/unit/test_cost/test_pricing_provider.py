# SPDX-License-Identifier: Apache-2.0
"""Providers: static, cached, live, and every way a feed is declined.

No test here touches the network. The live path is exercised through a local file, which is what
``QBC_PRICING_FEED`` accepts, and every key is a throwaway generated in the test.
"""

from __future__ import annotations

import json

import pytest

from qb_compiler.cost.pricing import PRICING_AS_OF, VENDOR_PRICING, cost_per_shot
from qb_compiler.cost.pricing_feed import build_feed, sign_feed
from qb_compiler.cost.pricing_provider import (
    ENV_FEED,
    ENV_LIVE,
    FeedPricingProvider,
    StaticPricingProvider,
    get_pricing_provider,
    live_requested,
    pricing_fields,
    static_entries,
)

from .test_pricing_feed import a_key, some_entries


@pytest.fixture
def feed_file(tmp_path):
    """A signed feed on disk, and the key it was signed with."""
    seed, public = a_key(11)
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps(sign_feed(build_feed(some_entries()), seed)), encoding="utf-8")
    return path, public


class TestStaticProvider:
    def test_every_backend_prices_exactly_as_the_table_does(self):
        """The one number a user has always got must not move because this module exists."""
        provider = StaticPricingProvider()
        for backend in VENDOR_PRICING:
            entry = provider.get(backend)
            assert entry is not None
            assert entry.cost_per_shot_usd == cost_per_shot(backend), backend

    def test_it_says_it_is_static_and_names_its_date(self):
        provider = StaticPricingProvider()
        assert provider.status == "static"
        assert provider.as_of == PRICING_AS_OF
        assert not provider.signature_verified

    def test_an_unknown_backend_is_none(self):
        assert StaticPricingProvider().get("no_such_backend") is None

    def test_the_entries_carry_the_vendor_billing_model(self):
        entries = static_entries()
        assert entries["ibm_fez"].billing.name == "per_second"
        assert entries["ionq_forte"].billing.name == "per_gate_shot"
        assert entries["quantinuum_h2"].billing.name == "per_hqc"
        assert entries["iqm_garnet"].billing.name == "per_shot"

    def test_a_vendor_that_does_not_sell_shots_still_answers_without_counts(self):
        breakdown = static_entries()["quantinuum_h2"].job_cost(10)
        assert breakdown.usd == pytest.approx(80.0)
        assert breakdown.assumptions["fell_back_from"] == "per_hqc"


class TestFeedProvider:
    def test_a_local_feed_is_served_live(self, feed_file, tmp_path):
        path, public = feed_file
        provider = FeedPricingProvider(
            url=str(path),
            public_key=public,
            cache_path=tmp_path / "cache.json",
            prefer_live=True,
        )
        assert provider.status == "live"
        assert provider.signature_verified
        assert provider.get("ibm_fez").source == "a vendor page"
        assert provider.status_of("ibm_fez") == "live"

    def test_a_backend_absent_from_the_feed_falls_back_to_the_table(self, feed_file, tmp_path):
        path, public = feed_file
        provider = FeedPricingProvider(
            url=str(path),
            public_key=public,
            cache_path=tmp_path / "cache.json",
            prefer_live=True,
        )
        entry = provider.get("quantinuum_h2")
        assert entry is not None
        assert provider.status_of("quantinuum_h2") == "static"
        assert "not in the feed" in provider.reason

    def test_a_missing_feed_falls_back_with_a_reason(self, tmp_path):
        _, public = a_key(11)
        provider = FeedPricingProvider(
            url=str(tmp_path / "absent.json"),
            public_key=public,
            cache_path=tmp_path / "cache.json",
            prefer_live=True,
        )
        assert provider.status == "static"
        assert "no feed at" in provider.reason
        assert provider.get("ibm_fez").cost_per_shot_usd == cost_per_shot("ibm_fez")

    def test_a_tampered_feed_is_refused_and_never_served(self, tmp_path):
        seed, public = a_key(11)
        feed = sign_feed(build_feed(some_entries()), seed)
        feed["entries"][0]["billing"]["usd_per_second"] = 99.0
        path = tmp_path / "tampered.json"
        path.write_text(json.dumps(feed), encoding="utf-8")
        provider = FeedPricingProvider(
            url=str(path), public_key=public, cache_path=tmp_path / "c.json", prefer_live=True
        )
        assert provider.status == "static"
        assert "refused" in provider.reason
        assert provider.get("ibm_fez").cost_per_shot_usd == cost_per_shot("ibm_fez")

    def test_a_feed_signed_by_the_wrong_key_is_refused(self, tmp_path):
        seed, _ = a_key(11)
        _, other = a_key(13)
        path = tmp_path / "pricing.json"
        path.write_text(json.dumps(sign_feed(build_feed(some_entries()), seed)), encoding="utf-8")
        provider = FeedPricingProvider(
            url=str(path), public_key=other, cache_path=tmp_path / "c.json", prefer_live=True
        )
        assert provider.status == "static"
        assert "refused" in provider.reason

    def test_a_feed_with_the_wrong_schema_is_refused(self, tmp_path):
        path = tmp_path / "pricing.json"
        path.write_text(json.dumps({"schema": "qb.other.v1", "entries": []}), encoding="utf-8")
        _, public = a_key(11)
        provider = FeedPricingProvider(
            url=str(path), public_key=public, cache_path=tmp_path / "c.json", prefer_live=True
        )
        assert provider.status == "static"
        assert "does not match the schema" in provider.reason

    def test_an_unsupported_scheme_is_refused_without_a_fetch(self, tmp_path):
        _, public = a_key(11)
        provider = FeedPricingProvider(
            url="ftp://example.invalid/pricing.json",
            public_key=public,
            cache_path=tmp_path / "c.json",
            prefer_live=True,
        )
        assert provider.status == "static"
        assert "cannot be read over" in provider.reason

    def test_nothing_is_fetched_unless_live_is_asked_for(self, tmp_path):
        path, public = feed_path_that_would_raise(tmp_path)
        provider = FeedPricingProvider(
            url=str(path), public_key=public, cache_path=tmp_path / "c.json", prefer_live=False
        )
        assert provider.status == "static"
        assert "not requested" in provider.reason


def feed_path_that_would_raise(tmp_path):
    """A path that is not a feed at all, so touching it would show up as a failure."""
    _, public = a_key(11)
    path = tmp_path / "not-a-feed.json"
    path.write_text("{", encoding="utf-8")
    return path, public


class TestCache:
    def _write_cache(self, path, feed, fetched_at):
        path.write_text(
            json.dumps({"schema": "qb.pricing_cache.v1", "fetched_at": fetched_at, "feed": feed}),
            encoding="utf-8",
        )

    def test_a_fresh_cache_is_served_without_a_fetch(self, tmp_path):
        seed, public = a_key(11)
        cache = tmp_path / "cache.json"
        self._write_cache(cache, sign_feed(build_feed(some_entries()), seed), 1000.0)
        provider = FeedPricingProvider(
            url=str(tmp_path / "absent.json"),
            public_key=public,
            cache_path=cache,
            prefer_live=False,
            now=lambda: 1000.0 + 3600.0,
        )
        assert provider.status == "cached"
        assert provider.get("ibm_fez").source == "a vendor page"

    def test_a_stale_cache_is_not_served(self, tmp_path):
        seed, public = a_key(11)
        cache = tmp_path / "cache.json"
        self._write_cache(cache, sign_feed(build_feed(some_entries()), seed), 1000.0)
        provider = FeedPricingProvider(
            url=str(tmp_path / "absent.json"),
            public_key=public,
            cache_path=cache,
            prefer_live=False,
            ttl_hours=24.0,
            now=lambda: 1000.0 + 25 * 3600.0,
        )
        assert provider.status == "static"

    def test_an_absent_cache_is_not_an_error(self, tmp_path):
        _, public = a_key(11)
        provider = FeedPricingProvider(
            url=str(tmp_path / "absent.json"),
            public_key=public,
            cache_path=tmp_path / "nothing-here.json",
            prefer_live=False,
        )
        assert provider.status == "static"

    def test_a_cache_signed_by_another_key_is_not_served(self, tmp_path):
        seed, _ = a_key(11)
        _, other = a_key(13)
        cache = tmp_path / "cache.json"
        self._write_cache(cache, sign_feed(build_feed(some_entries()), seed), 1000.0)
        provider = FeedPricingProvider(
            url=str(tmp_path / "absent.json"),
            public_key=other,
            cache_path=cache,
            prefer_live=False,
            now=lambda: 1000.0,
        )
        assert provider.status == "static"

    def test_a_live_fetch_writes_the_cache(self, feed_file, tmp_path):
        path, public = feed_file
        cache = tmp_path / "cache.json"
        FeedPricingProvider(
            url=str(path),
            public_key=public,
            cache_path=cache,
            prefer_live=True,
            now=lambda: 4242.0,
        )
        written = json.loads(cache.read_text(encoding="utf-8"))
        assert written["schema"] == "qb.pricing_cache.v1"
        assert written["fetched_at"] == 4242.0
        assert written["feed"]["schema"] == "qb.pricing_feed.v1"


class TestSelection:
    def test_the_default_is_the_static_table(self, monkeypatch):
        monkeypatch.delenv(ENV_LIVE, raising=False)
        provider = get_pricing_provider()
        assert isinstance(provider, StaticPricingProvider)

    def test_the_environment_can_ask_for_live(self, monkeypatch, tmp_path, feed_file):
        path, _ = feed_file
        monkeypatch.setenv(ENV_LIVE, "1")
        monkeypatch.setenv(ENV_FEED, str(path))
        assert live_requested() is True
        provider = get_pricing_provider(cache_path=tmp_path / "c.json")
        assert isinstance(provider, FeedPricingProvider)
        # Signed by a throwaway key, so the shipped key refuses it. That is the point.
        assert provider.status == "static"
        assert "refused" in provider.reason

    def test_the_receipt_fields_say_where_the_price_came_from(self):
        fields = pricing_fields(StaticPricingProvider())
        assert fields["pricing_status"] == "static"
        assert fields["pricing_as_of"] == PRICING_AS_OF
        assert fields["pricing_signature_verified"] is False
        assert fields["pricing_source"]
