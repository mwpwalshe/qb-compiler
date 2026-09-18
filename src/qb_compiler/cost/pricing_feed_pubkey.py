"""The public key a pricing feed is checked against.

One constant, safe to publish, and the only thing this package needs in order to refuse a feed
that somebody else signed. The private half never leaves the machine that issues the feed and is
not in this repository.

Replacing this key invalidates every feed signed under the old one, so it changes with a release,
never quietly.
"""

from __future__ import annotations

#: Ed25519 public key, base64, 32 bytes.
FEED_PUBLIC_KEY_B64 = "CX85hIDhhZiF/iZyM/vf24C7dibfi29KACrjJb+qxkg="

#: First 16 hex characters of the key's sha256, the same fingerprint form receipts carry.
FEED_KEY_FINGERPRINT = "fec00ff18c3867ae"


def feed_public_key() -> bytes:
    """The key as raw bytes."""
    import base64

    return base64.b64decode(FEED_PUBLIC_KEY_B64, validate=True)


__all__ = ["FEED_KEY_FINGERPRINT", "FEED_PUBLIC_KEY_B64", "feed_public_key"]
