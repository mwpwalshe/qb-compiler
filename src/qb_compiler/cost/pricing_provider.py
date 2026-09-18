"""Where a price comes from, and what it says about itself when it cannot come from there.

The calibration registry already solves this shape for hardware data: a provider per source, a
declared status, a fallback that never fails, and a column in the output saying which row is live
and which is static. Prices get the same treatment.

* **Nothing fetches at import.** Building a provider opens no socket. Offline behaviour is exactly
  what it was before this module existed, plus a status field.
* **Network only when asked**: ``prefer_live=True``, ``--live`` on the command line, or
  ``QBC_PRICING_LIVE=1``. Timeout 5 seconds, once.
* **A feed that does not verify is refused**, and the reason travels with the fallback, so a
  caller can see that a price was rejected rather than merely missing.

Status on every number: ``live`` fetched in this call, ``cached`` served from a verified cache
younger than the TTL, ``static`` the table that shipped with the package.

Usage::

    from qb_compiler.cost.pricing_provider import get_pricing_provider

    provider = get_pricing_provider()                  # static, no network, no surprises
    entry = provider.get("ibm_fez")
    entry.job_cost(4096).usd

    live = get_pricing_provider(prefer_live=True)      # reads QBC_PRICING_FEED or the default URL
    live.status                                        # 'live', 'cached' or 'static'
    live.reason                                        # why, when it is not live
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from qb_compiler.cost.billing import CACHED, LIVE, STATIC, PerGateShot, PerHQC, PerSecond, PerShot
from qb_compiler.cost.pricing import PRICING_AS_OF, VENDOR_PRICING
from qb_compiler.cost.pricing_feed import FeedError, ParsedFeed, PricingEntry, parse_feed

#: Where the published feed lives. Reading it is opt in; see the module docstring.
DEFAULT_FEED_URL = "https://qubitboost.io/feeds/pricing.json"

#: Environment overrides. ``QBC_PRICING_FEED`` takes a URL or a local path.
ENV_FEED = "QBC_PRICING_FEED"
ENV_LIVE = "QBC_PRICING_LIVE"

#: Seconds to wait for the feed. One attempt, then the static table.
FETCH_TIMEOUT_SECONDS = 5.0

#: How long a verified cached feed is served before another fetch is attempted.
DEFAULT_TTL_HOURS = 24.0

_CACHE_SCHEMA = "qb.pricing_cache.v1"


def default_cache_path() -> Path:
    """``~/.qb-compiler/pricing_cache.json``, beside the signing key."""
    return Path.home() / ".qb-compiler" / "pricing_cache.json"


# ── the shipped table, as entries ────────────────────────────────────


def _static_billing(backend: str) -> Any:
    """The billing model for a shipped row, built so its per-shot number is exactly the table's.

    IBM publishes seconds, Azure publishes gate shots and HQCs. Those rows carry the vendor's own
    model with the shipped per-shot figure as the fallback, so a caller with gate counts gets the
    vendor's arithmetic and a caller without one gets the number this package has always returned.
    """
    row = VENDOR_PRICING[backend]
    per_shot = PerShot(
        cost_per_shot_usd=row.cost_per_shot_usd, cost_per_task_usd=row.cost_per_task_usd
    )
    if row.provider == "ibm":
        # 96 USD per minute, billed per second, is 1.60 per second. The shot rate that implies
        # is derived from the shipped conversion so the two never disagree.
        return PerSecond.from_per_shot(1.60, row.cost_per_shot_usd)
    if backend == "ionq_aria":
        return PerGateShot(0.000220, 0.000975, 12.4166, fallback=per_shot)
    if backend == "ionq_forte":
        return PerGateShot(0.0001645, 0.001121, 25.7899, fallback=per_shot)
    if row.provider == "quantinuum":
        # 125,000 USD a month for 10,000 HQCs on the Standard plan implies 12.50 per HQC.
        return PerHQC(12.50, fallback=per_shot)
    return per_shot


def static_entries() -> dict[str, PricingEntry]:
    """The shipped table as :class:`PricingEntry` rows. No network, no files, no clock."""
    entries: dict[str, PricingEntry] = {}
    for backend, row in VENDOR_PRICING.items():
        entries[backend] = PricingEntry(
            backend=backend,
            provider=row.provider,
            billing=_static_billing(backend),
            as_of=PRICING_AS_OF,
            source="the table shipped with this package, checked against the vendor pages",
            notes=row.notes,
        )
    return entries


# ── providers ────────────────────────────────────────────────────────


@dataclass
class StaticPricingProvider:
    """The table that shipped with the package. Never fails, never fetches, never expires.

    It carries the staleness warning it always did: :func:`qb_compiler.cost.pricing.get_pricing`
    warns once when the table is more than 90 days old, and that warning is not suppressed here.
    """

    status: str = STATIC
    reason: str = "the table shipped with this package"
    as_of: str = PRICING_AS_OF
    signature_verified: bool = False

    def __post_init__(self) -> None:
        self._entries = static_entries()

    def get(self, backend: str) -> PricingEntry | None:
        from qb_compiler.cost.pricing import get_pricing as _warn_and_get

        _warn_and_get(backend)  # keeps the staleness warning on the path callers already use
        return self._entries.get(backend)

    def backends(self) -> list[str]:
        return sorted(self._entries)

    def status_of(self, backend: str) -> str:
        del backend
        return self.status


@dataclass
class FeedPricingProvider:
    """A signed feed, with the shipped table underneath it.

    Parameters
    ----------
    url :
        A URL or a local path. Defaults to ``QBC_PRICING_FEED``, then :data:`DEFAULT_FEED_URL`.
    cache_path :
        Where a verified feed is kept. Defaults to :func:`default_cache_path`.
    public_key :
        The key the signature is checked against. Defaults to the one shipped in the package.
    ttl_hours :
        How long a cached feed is served before another fetch is attempted.
    prefer_live :
        Fetch at all. False means cache then static, and nothing touches the network.
    now :
        Clock, injectable so the TTL can be tested without waiting a day.
    """

    url: str | None = None
    cache_path: Path | None = None
    public_key: bytes | None = None
    ttl_hours: float = DEFAULT_TTL_HOURS
    prefer_live: bool = False
    now: Any = time.time

    def __post_init__(self) -> None:
        self._static = StaticPricingProvider()
        self._feed: ParsedFeed | None = None
        self.status = STATIC
        self.reason = ""
        self.as_of = PRICING_AS_OF
        self.signature_verified = False
        self._resolve()

    # -- resolution ---------------------------------------------------
    def _key(self) -> bytes:
        if self.public_key is not None:
            return self.public_key
        from qb_compiler.cost.pricing_feed_pubkey import feed_public_key

        return feed_public_key()

    def _source(self) -> str:
        return self.url or os.environ.get(ENV_FEED) or DEFAULT_FEED_URL

    def _cache_file(self) -> Path:
        return self.cache_path or default_cache_path()

    def _resolve(self) -> None:
        """Pick a feed once, at construction: live, then cache, then static with a reason."""
        if self.prefer_live:
            feed, reason = self._fetch()
            if feed is not None:
                self._adopt(feed, LIVE, "fetched and verified in this call")
                self._write_cache(feed)
                return
            cached = self._read_cache(ignore_ttl=False)
            if cached is not None:
                self._adopt(cached, CACHED, f"{reason}; served the cached feed instead")
                return
            self.reason = reason
            return
        cached = self._read_cache(ignore_ttl=False)
        if cached is not None:
            self._adopt(cached, CACHED, "served from the cache; live was not requested")
            return
        self.reason = "live pricing was not requested"

    def _adopt(self, feed: ParsedFeed, status: str, reason: str) -> None:
        self._feed = feed
        self.status = status
        self.reason = reason
        self.signature_verified = feed.signature_verified
        stamps = sorted(entry.as_of for entry in feed.entries.values() if entry.as_of)
        self.as_of = stamps[-1] if stamps else feed.generated_at[:10]

    # -- fetching -----------------------------------------------------
    def _fetch(self) -> tuple[ParsedFeed | None, str]:
        source = self._source()
        parsed = urlparse(source)
        try:
            if parsed.scheme in ("http", "https"):
                with urlopen(source, timeout=FETCH_TIMEOUT_SECONDS) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            elif parsed.scheme in ("", "file"):
                path = Path(parsed.path if parsed.scheme == "file" else source).expanduser()
                if not path.is_file():
                    return None, f"no feed at {path}"
                payload = json.loads(path.read_text(encoding="utf-8"))
            else:
                return None, f"a feed cannot be read over {parsed.scheme!r}"
        except (URLError, TimeoutError, OSError) as exc:
            return None, f"the feed at {source} could not be read: {exc}"
        except json.JSONDecodeError as exc:
            return None, f"the feed at {source} is not readable JSON: {exc}"

        try:
            feed = parse_feed(payload, self._key())
        except FeedError as exc:
            return None, f"the feed at {source} does not match the schema: {exc}"
        if not feed.signature_verified:
            return None, f"the feed at {source} was refused: {feed.reason}"
        return feed, "fetched"

    # -- cache --------------------------------------------------------
    def _read_cache(self, *, ignore_ttl: bool) -> ParsedFeed | None:
        path = self._cache_file()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("schema") != _CACHE_SCHEMA:
            return None
        fetched_at = float(payload.get("fetched_at", 0.0))
        if not ignore_ttl and (self.now() - fetched_at) > self.ttl_hours * 3600.0:
            return None
        try:
            feed = parse_feed(payload.get("feed", {}), self._key())
        except FeedError:
            return None
        return feed if feed.signature_verified else None

    def _write_cache(self, feed: ParsedFeed) -> None:
        path = self._cache_file()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {"schema": _CACHE_SCHEMA, "fetched_at": float(self.now()), "feed": feed.raw},
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            # A cache that cannot be written costs a fetch next time and nothing else.
            pass

    # -- the provider interface ---------------------------------------
    def get(self, backend: str) -> PricingEntry | None:
        if self._feed is not None:
            entry = self._feed.entries.get(backend)
            if entry is not None:
                return entry
            self.reason = f"{backend} is not in the feed; served the shipped table for it"
        return self._static.get(backend)

    def backends(self) -> list[str]:
        names = set(self._static.backends())
        if self._feed is not None:
            names |= set(self._feed.entries)
        return sorted(names)

    def status_of(self, backend: str) -> str:
        """The status of this one backend, which is not always the provider's own."""
        if self._feed is not None and backend in self._feed.entries:
            return self.status
        return STATIC


PricingProvider = StaticPricingProvider | FeedPricingProvider


def live_requested(flag: bool = False) -> bool:
    """True when live pricing was asked for, by flag or by ``QBC_PRICING_LIVE=1``."""
    return bool(flag) or os.environ.get(ENV_LIVE, "").strip() in ("1", "true", "yes", "on")


def get_pricing_provider(prefer_live: bool = False, **kwargs: Any) -> PricingProvider:
    """The pricing provider to use, in the registry's own style.

    ``prefer_live=False`` and no ``QBC_PRICING_LIVE`` gives the shipped table and touches nothing
    else, which is what every existing caller gets. Otherwise a :class:`FeedPricingProvider`,
    which falls back to the same table with the reason recorded.
    """
    if not live_requested(prefer_live) and not kwargs:
        return StaticPricingProvider()
    return FeedPricingProvider(prefer_live=live_requested(prefer_live), **kwargs)


def pricing_fields(provider: PricingProvider, backend: str | None = None) -> dict[str, Any]:
    """The four fields a receipt carries about where its prices came from."""
    status = provider.status_of(backend) if backend else provider.status
    return {
        "pricing_status": status,
        "pricing_as_of": provider.as_of if status != STATIC else PRICING_AS_OF,
        "pricing_source": getattr(provider, "reason", "") or "the table shipped with this package",
        "pricing_signature_verified": bool(getattr(provider, "signature_verified", False)),
    }


__all__ = [
    "DEFAULT_FEED_URL",
    "DEFAULT_TTL_HOURS",
    "ENV_FEED",
    "ENV_LIVE",
    "FETCH_TIMEOUT_SECONDS",
    "FeedPricingProvider",
    "PricingProvider",
    "StaticPricingProvider",
    "default_cache_path",
    "get_pricing_provider",
    "live_requested",
    "pricing_fields",
    "static_entries",
]
