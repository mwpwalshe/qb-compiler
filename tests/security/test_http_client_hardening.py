"""Regression tests for the hardened HTTP client defaults.

``create_hardened_http_client`` is the documented way to build an
``httpx.Client`` for talking to the calibration hub.  These tests pin the
security-relevant defaults so a future edit cannot silently weaken them:

* TLS certificate verification stays ON,
* explicit connect/read/write timeouts are set (no hang-forever DoS),
* redirects are bounded.

Note: nothing in this repository currently fetches and parses untrusted
JSON through this client (live fetching is delegated to the proprietary
``qubitboost-sdk``), so there is no SSRF/JSON-bomb sink to exploit here;
these tests lock down the helper's hardened baseline for downstream use.
"""

from __future__ import annotations

import httpx

from qb_compiler.calibration.live_provider import create_hardened_http_client


def test_tls_verification_is_enabled_by_default():
    client = create_hardened_http_client()
    try:
        # httpx stores the verify decision on the transport's SSL context;
        # a verifying context has check_hostname True and CERT_REQUIRED.
        import ssl

        transport = client._transport_for_url(httpx.URL("https://example.com"))
        ssl_ctx = transport._pool._ssl_context  # type: ignore[attr-defined]
        assert isinstance(ssl_ctx, ssl.SSLContext)
        assert ssl_ctx.verify_mode == ssl.CERT_REQUIRED
        assert ssl_ctx.check_hostname is True
    finally:
        client.close()


def test_timeouts_are_bounded():
    client = create_hardened_http_client()
    try:
        timeout = client.timeout
        assert timeout.connect is not None and timeout.connect <= 30
        assert timeout.read is not None and timeout.read <= 60
        assert timeout.write is not None
    finally:
        client.close()


def test_redirects_are_bounded():
    client = create_hardened_http_client()
    try:
        assert client.max_redirects <= 5
    finally:
        client.close()


def test_verify_default_present_in_source():
    """The hardened defaults dict must explicitly pin verify=True."""
    import inspect

    from qb_compiler.calibration import live_provider

    src = inspect.getsource(live_provider.create_hardened_http_client)
    assert '"verify": True' in src or "verify=True" in src
