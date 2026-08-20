# SPDX-License-Identifier: Apache-2.0
"""Signing and offline verification.

The behaviour under test is the correction made in 0.12.0: a signature is worth something only
when the key outlives the receipt and the verifier refuses to trust a key that travelled with the
signature. Every test here sets QBC_SIGNING_KEY to a temporary path, so nothing touches a real
home directory.
"""

from __future__ import annotations

import base64
import json
import os
import stat

import pytest

from qb_compiler import signing
from qb_compiler.signing import (
    INVALID_SIGNATURE,
    LEGACY_SELF_SIGNED,
    MALFORMED,
    NO_KEY,
    UNSIGNED,
    VERIFIED,
    SigningError,
    canonical_payload,
    load_or_create_signing_key,
    load_signing_key,
    public_key_fingerprint,
    sign_receipt,
    verify_receipt,
    verify_receipt_file,
)


@pytest.fixture()
def key_path(tmp_path, monkeypatch):
    """Point the signing key at a temporary location for the duration of a test."""
    path = tmp_path / "keys" / "signing_key"
    monkeypatch.setenv("QBC_SIGNING_KEY", str(path))
    monkeypatch.setenv("QBC_TRUSTED_KEYS", str(tmp_path / "keys" / "trusted_keys"))
    return path


@pytest.fixture()
def receipt():
    return {
        "schema": "qb.selection_receipt.v1",
        "selected_layout": {"0": 6, "1": 5},
        "selected_score": 0.6084,
        "signature": None,
        "signing": "unsigned",
    }


class TestKeyManagement:
    def test_key_is_created_once_and_reused(self, key_path):
        first = load_or_create_signing_key()
        second = load_or_create_signing_key()
        assert first.seed == second.seed
        assert first.fingerprint == second.fingerprint
        assert key_path.exists()

    def test_key_file_is_private(self, key_path):
        load_or_create_signing_key()
        mode = stat.S_IMODE(os.stat(key_path).st_mode)
        assert mode == 0o600, f"private key is mode {oct(mode)}, expected 0o600"

    def test_loading_a_missing_key_fails_rather_than_inventing_one(self, key_path):
        with pytest.raises(SigningError, match="no signing key"):
            load_signing_key()

    def test_key_file_with_a_mismatched_public_half_is_refused(self, key_path):
        load_or_create_signing_key()
        record = json.loads(key_path.read_text())
        other = base64.b64encode(bytes(32)).decode()
        record["public_key_b64"] = other
        key_path.write_text(json.dumps(record))
        with pytest.raises(SigningError, match="inconsistent"):
            load_signing_key()

    def test_fingerprint_is_derived_from_the_public_key(self, key_path):
        key = load_or_create_signing_key()
        assert key.fingerprint == public_key_fingerprint(key.public)
        assert len(key.fingerprint) == 16

    def test_export_public_key_writes_something_publishable(self, key_path, tmp_path):
        key = load_or_create_signing_key()
        out = signing.export_public_key(tmp_path / "qbc.pub", key=key)
        text = out.read_text()
        assert key.public_b64 in text
        assert base64.b64encode(key.seed).decode() not in text, "private key leaked into the export"


class TestSignAndVerify:
    def test_roundtrip(self, key_path, receipt):
        key = load_or_create_signing_key()
        signed = sign_receipt(receipt, key=key)
        result = verify_receipt(signed, public_key=key.public)
        assert result.status == VERIFIED
        assert result.ok
        assert result.key_fingerprint == key.fingerprint

    def test_signature_does_not_carry_the_public_key(self, key_path, receipt):
        signed = sign_receipt(receipt)
        assert "public_key" not in signed
        assert signed["key_fingerprint"]
        assert signed["signature"]

    def test_tampering_breaks_the_signature(self, key_path, receipt):
        key = load_or_create_signing_key()
        signed = sign_receipt(receipt, key=key)
        signed["selected_layout"]["0"] = 99
        result = verify_receipt(signed, public_key=key.public)
        assert result.status == INVALID_SIGNATURE
        assert not result.ok

    def test_tampering_with_the_fingerprint_breaks_it_too(self, key_path, receipt):
        # The fingerprint is inside the signed bytes precisely so it cannot be swapped.
        key = load_or_create_signing_key()
        signed = sign_receipt(receipt, key=key)
        signed["key_fingerprint"] = "0" * 16
        assert verify_receipt(signed, public_key=key.public).status == INVALID_SIGNATURE

    def test_wrong_key_does_not_verify(self, key_path, tmp_path, receipt):
        signed = sign_receipt(receipt)
        other = load_or_create_signing_key(tmp_path / "other_key")
        result = verify_receipt(signed, public_key=other.public)
        assert result.status == INVALID_SIGNATURE
        assert not result.ok

    def test_missing_key_fails_closed(self, key_path, receipt):
        signed = sign_receipt(receipt)
        result = verify_receipt(signed)
        assert result.status == NO_KEY
        assert not result.ok, "a signed receipt with no key to check it against must not pass"

    def test_unsigned_receipt_is_not_a_pass(self, key_path, receipt):
        result = verify_receipt(receipt)
        assert result.status == UNSIGNED
        assert not result.ok

    def test_legacy_self_signed_receipt_is_refused(self, key_path):
        # The shape emitted before 0.12.0: a signature plus the key that made it.
        legacy = {
            "schema": "qb.selection_receipt.v1",
            "selected_layout": {"0": 1},
            "signature": base64.b64encode(bytes(64)).decode(),
            "signing": "ed25519 (qubitboost_sdk)",
            "public_key": base64.b64encode(bytes(32)).decode(),
        }
        result = verify_receipt(legacy)
        assert result.status == LEGACY_SELF_SIGNED
        assert not result.ok
        assert "attest to nothing" in result.reason

    def test_trusted_keys_file_is_read_when_no_key_is_passed(self, key_path, tmp_path, receipt):
        key = load_or_create_signing_key()
        trusted = tmp_path / "keys" / "trusted_keys"
        trusted.write_text(f"# a comment\n{key.public_b64}\n")
        signed = sign_receipt(receipt, key=key)
        assert verify_receipt(signed).status == VERIFIED

    def test_malformed_signature_is_reported_not_raised(self, key_path, receipt):
        receipt["signature"] = "not base64 at all !!"
        assert verify_receipt(receipt).status == MALFORMED

    def test_wrong_length_signature_is_malformed(self, key_path, receipt):
        receipt["signature"] = base64.b64encode(b"short").decode()
        assert verify_receipt(receipt).status == MALFORMED

    def test_unknown_scheme_is_refused(self, key_path, receipt):
        key = load_or_create_signing_key()
        signed = sign_receipt(receipt, key=key)
        signed["signature_scheme"] = "something-else-v9"
        assert verify_receipt(signed, public_key=key.public).status == MALFORMED

    def test_verify_file_roundtrip(self, key_path, tmp_path, receipt):
        key = load_or_create_signing_key()
        path = tmp_path / "receipt.json"
        path.write_text(json.dumps(sign_receipt(receipt, key=key)))
        assert verify_receipt_file(path, public_key=key.public_b64).ok

    def test_verify_file_on_junk_is_malformed(self, tmp_path, key_path):
        path = tmp_path / "junk.json"
        path.write_text("{not json")
        assert verify_receipt_file(path).status == MALFORMED

    def test_verification_states_what_it_does_not_claim(self, key_path, receipt):
        key = load_or_create_signing_key()
        result = verify_receipt(sign_receipt(receipt, key=key), public_key=key.public)
        joined = " ".join(result.does_not_claim)
        assert "correct" in joined or "worth doing" in joined


class TestCanonicalPayload:
    def test_key_order_does_not_change_the_payload(self):
        one = {"b": 2, "a": 1, "signature": "x"}
        two = {"a": 1, "b": 2, "signature": "y"}
        assert canonical_payload(one) == canonical_payload(two)

    def test_every_other_field_is_covered(self):
        base = {"a": 1, "signature": None}
        changed = {"a": 2, "signature": None}
        assert canonical_payload(base) != canonical_payload(changed)


class TestPurePythonBackend:
    """The fallback used when cryptography is not installed has to be right on its own."""

    def test_rfc8032_test_vector_1(self):
        from qb_compiler import _ed25519

        seed = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
        expected_public = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
        assert _ed25519.public_key(seed).hex() == expected_public
        signature = _ed25519.sign(seed, b"")
        assert signature.hex() == (
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
            "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
        )
        assert _ed25519.verify(_ed25519.public_key(seed), b"", signature)

    def test_rfc8032_test_vector_2(self):
        from qb_compiler import _ed25519

        seed = bytes.fromhex("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb")
        message = bytes.fromhex("72")
        assert (
            _ed25519.public_key(seed).hex()
            == "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c"
        )
        assert _ed25519.sign(seed, message).hex() == (
            "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
            "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"
        )

    def test_rejects_a_flipped_bit(self):
        from qb_compiler import _ed25519

        seed = bytes(range(32))
        pub = _ed25519.public_key(seed)
        signature = bytearray(_ed25519.sign(seed, b"receipt"))
        signature[0] ^= 0x01
        assert not _ed25519.verify(pub, b"receipt", bytes(signature))

    def test_rejects_malformed_lengths(self):
        from qb_compiler import _ed25519

        assert not _ed25519.verify(b"short", b"m", bytes(64))
        assert not _ed25519.verify(bytes(32), b"m", b"short")

    def test_agrees_with_cryptography_when_it_is_installed(self, key_path, receipt):
        cryptography_backend = signing._cryptography_backend()
        if cryptography_backend is None:  # pragma: no cover - depends on the environment
            pytest.skip("cryptography not installed")

        from qb_compiler import _ed25519

        key = load_or_create_signing_key()
        signed = sign_receipt(receipt, key=key)
        payload = canonical_payload(signed)
        signature = base64.b64decode(signed["signature"])
        # Signed by whichever backend is live; the pure-Python code must accept it, and the
        # signature it produces must in turn verify through the compiled one.
        assert _ed25519.verify(key.public, payload, signature)
        assert _ed25519.public_key(key.seed) == key.public
        assert signing._raw_verify(key.public, payload, _ed25519.sign(key.seed, payload))
