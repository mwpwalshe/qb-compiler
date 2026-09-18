# SPDX-License-Identifier: Apache-2.0
"""Signing and offline verification for qb-compiler receipts.

Two halves, and the split is deliberate.

**Signing** binds a receipt to a key you keep. The key is created once, stored at a path you
control, and reused for every receipt afterwards, so a reader who has seen one of your receipts
before can tell that a later one came from the same place.

**Verification** is free, offline, and needs nothing from us: a third party who has been handed a
receipt runs :func:`verify_receipt` against a public key they got separately, and gets one of a
small set of verdicts. It never trusts a key carried inside the receipt it is checking, because a
signature that validates against a key travelling with the signature proves only that the two were
made together.

That last sentence is the whole reason this module exists. Before 0.12.0 ``selection_receipt(
sign=True)`` generated a fresh keypair per call and embedded the public half in the receipt it had
just signed. Every such receipt verified, and none of them attested to anything. Receipts in that
shape are reported as ``LEGACY_SELF_SIGNED`` and are never treated as verified.

Key location
------------
``QBC_SIGNING_KEY`` names the private key file; the default is ``~/.qb-compiler/signing_key``. It
is created with mode 0600 on first use and is never written anywhere else. Deliberately not under
``QBC_DATA_DIR``, which is a log directory that users copy around.

``QBC_TRUSTED_KEYS`` names a file of public keys to verify against, one base64 key per line, ``#``
comments allowed; the default is ``~/.qb-compiler/trusted_keys``. A caller can pass keys directly
instead, which is what a CI job should do.

What a signature does and does not say
--------------------------------------
It says these exact bytes were signed by the holder of that private key. It says nothing about
whether the layout was good, the run was worth doing, or the numbers inside are correct. The
verifier prints that distinction rather than leaving a reader to assume the stronger claim.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KEY_SCHEMA = "qb.signing_key.v1"
SIGNATURE_SCHEME = "ed25519-over-canonical-json-v1"

#: Fields stripped before the canonical payload is computed. Everything else in the receipt,
#: including the key fingerprint and the scheme name, is covered by the signature.
_UNSIGNED_FIELDS = ("signature",)

_ENV_SIGNING_KEY = "QBC_SIGNING_KEY"
_ENV_TRUSTED_KEYS = "QBC_TRUSTED_KEYS"


class SigningError(RuntimeError):
    """Raised when a key cannot be loaded, created, or used."""


# ── backend selection ────────────────────────────────────────────────


def _cryptography_backend() -> Any | None:
    """Return the ``cryptography`` ed25519 module, or ``None`` when it is not installed."""
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except Exception:  # pragma: no cover - depends on the environment
        return None
    return ed25519


def _derive_public(seed: bytes) -> bytes:
    ed25519 = _cryptography_backend()
    if ed25519 is not None:
        from cryptography.hazmat.primitives import serialization

        private = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
        return bytes(
            private.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        )
    from qb_compiler._ed25519 import public_key as fallback_public_key

    return fallback_public_key(seed)


def _raw_sign(seed: bytes, payload: bytes) -> bytes:
    ed25519 = _cryptography_backend()
    if ed25519 is not None:
        private = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
        return bytes(private.sign(payload))
    from qb_compiler._ed25519 import sign as fallback_sign

    return fallback_sign(seed, payload)


def _raw_verify(public: bytes, payload: bytes, signature: bytes) -> bool:
    ed25519 = _cryptography_backend()
    if ed25519 is not None:
        try:
            ed25519.Ed25519PublicKey.from_public_bytes(public).verify(signature, payload)
        except Exception:
            return False
        return True
    from qb_compiler._ed25519 import verify as fallback_verify

    return fallback_verify(public, payload, signature)


def public_key_from_seed(seed: bytes) -> bytes:
    """The 32-byte public half of a 32-byte Ed25519 seed."""
    return _derive_public(seed)


def sign_bytes(seed: bytes, payload: bytes) -> bytes:
    """Sign raw bytes with a 32-byte Ed25519 seed. Returns the 64-byte signature.

    For anything that is not a receipt and carries its own envelope, such as the pricing feed.
    Uses ``cryptography`` when it is installed and the pure-python implementation otherwise, so a
    base install can still sign and verify.
    """
    return _raw_sign(seed, payload)


def verify_bytes(public: bytes, payload: bytes, signature: bytes) -> bool:
    """Check a signature over raw bytes. False on any failure, never raises."""
    return _raw_verify(public, payload, signature)


# ── keys ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SigningKey:
    """An Ed25519 keypair loaded from disk.

    Attributes
    ----------
    seed :
        The 32-byte private seed. Never emitted into a receipt.
    public :
        The 32-byte public key.
    key_id :
        Free-form label stored alongside the key, for a holder with more than one.
    path :
        Where the key was loaded from, or ``None`` for an in-memory key.
    """

    seed: bytes = field(repr=False)
    public: bytes
    key_id: str = "qbc"
    path: Path | None = None

    @property
    def fingerprint(self) -> str:
        """Short, stable identifier for the public half: first 16 hex of its sha256."""
        return public_key_fingerprint(self.public)

    @property
    def public_b64(self) -> str:
        return base64.b64encode(self.public).decode("ascii")


def public_key_fingerprint(public: bytes) -> str:
    """Fingerprint a raw 32-byte public key: first 16 hex characters of its sha256."""
    return hashlib.sha256(public).hexdigest()[:16]


def default_key_path() -> Path:
    """Path of the private key: ``QBC_SIGNING_KEY``, else ``~/.qb-compiler/signing_key``."""
    override = os.environ.get(_ENV_SIGNING_KEY)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".qb-compiler" / "signing_key"


def default_trusted_keys_path() -> Path:
    """Path of the trusted key list: ``QBC_TRUSTED_KEYS``, else ``~/.qb-compiler/trusted_keys``."""
    override = os.environ.get(_ENV_TRUSTED_KEYS)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".qb-compiler" / "trusted_keys"


def load_signing_key(path: str | Path | None = None) -> SigningKey:
    """Load the signing key at *path*, raising :class:`SigningError` when it is absent.

    Never creates anything. Use :func:`load_or_create_signing_key` when creation is wanted.
    """
    key_path = Path(path).expanduser() if path is not None else default_key_path()
    if not key_path.exists():
        raise SigningError(
            f"no signing key at {key_path}. Create one with "
            f"qb_compiler.signing.load_or_create_signing_key(), or point "
            f"{_ENV_SIGNING_KEY} at an existing key."
        )
    try:
        record = json.loads(key_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SigningError(f"signing key at {key_path} is not readable JSON: {exc}") from exc

    if record.get("schema") != KEY_SCHEMA:
        raise SigningError(
            f"signing key at {key_path} has schema {record.get('schema')!r}, "
            f"expected {KEY_SCHEMA!r}"
        )
    try:
        seed = base64.b64decode(record["private_key_b64"], validate=True)
    except Exception as exc:
        raise SigningError(f"signing key at {key_path} has an unreadable private key") from exc
    if len(seed) != 32:
        raise SigningError(f"signing key at {key_path} is {len(seed)} bytes, expected 32")

    public = _derive_public(seed)
    stored = record.get("public_key_b64")
    if stored and base64.b64decode(stored) != public:
        raise SigningError(
            f"signing key at {key_path} is inconsistent: the stored public key does not match "
            "the private key. Refusing to sign with it."
        )
    return SigningKey(
        seed=seed,
        public=public,
        key_id=str(record.get("key_id") or "qbc"),
        path=key_path,
    )


def load_or_create_signing_key(
    path: str | Path | None = None,
    *,
    key_id: str = "qbc",
) -> SigningKey:
    """Load the signing key at *path*, creating it once with mode 0600 when it does not exist.

    Creating a key is the only write this module performs. The same key is then used for every
    receipt signed on this machine, which is what makes a signature worth checking: a reader who
    has one of your receipts can tell that a later one came from the same holder.
    """
    key_path = Path(path).expanduser() if path is not None else default_key_path()
    if key_path.exists():
        return load_signing_key(key_path)

    seed = os.urandom(32)
    public = _derive_public(seed)
    record = {
        "schema": KEY_SCHEMA,
        "algorithm": "ed25519",
        "key_id": key_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "private_key_b64": base64.b64encode(seed).decode("ascii"),
        "public_key_b64": base64.b64encode(public).decode("ascii"),
        "fingerprint": public_key_fingerprint(public),
    }
    key_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Create with 0600 from the outset rather than chmod after writing: between the write and
        # the chmod the private key would sit on disk world-readable.
        fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    except FileExistsError:  # pragma: no cover - lost a race with another process
        return load_signing_key(key_path)
    except OSError as exc:
        raise SigningError(f"could not create a signing key at {key_path}: {exc}") from exc
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    # A shared parent directory is the caller's choice, so a refusal here is not fatal.
    with contextlib.suppress(OSError):  # pragma: no cover
        key_path.parent.chmod(stat.S_IRWXU)
    return SigningKey(seed=seed, public=public, key_id=key_id, path=key_path)


def export_public_key(path: str | Path, *, key: SigningKey | None = None) -> Path:
    """Write the public half of *key* to *path*, one base64 line, safe to publish."""
    signing_key = key or load_signing_key()
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"# qb-compiler ed25519 public key, fingerprint {signing_key.fingerprint}\n"
        f"{signing_key.public_b64}\n",
        encoding="utf-8",
    )
    return out


def parse_public_key(value: str | bytes) -> bytes:
    """Read a public key given as raw bytes, base64, or hex. Raises on anything else."""
    if isinstance(value, bytes | bytearray):
        if len(value) == 32:
            return bytes(value)
        value = value.decode("ascii", errors="ignore")
    text = str(value).strip()
    for decoder in (
        lambda t: base64.b64decode(t, validate=True),
        bytes.fromhex,
    ):
        try:
            raw = decoder(text)
        except Exception:
            continue
        if len(raw) == 32:
            return raw
    raise SigningError(f"not an ed25519 public key: {text[:32]!r}")


def load_trusted_keys(path: str | Path | None = None) -> list[bytes]:
    """Read public keys from a trusted-keys file. Missing file means an empty list."""
    key_path = Path(path).expanduser() if path is not None else default_trusted_keys_path()
    if not key_path.exists():
        return []
    keys: list[bytes] = []
    for raw_line in key_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        keys.append(parse_public_key(line))
    return keys


# ── canonical payload ────────────────────────────────────────────────


def canonical_payload(receipt: dict[str, Any]) -> bytes:
    """Bytes a signature covers: the receipt minus its signature field, as canonical JSON.

    Canonical here means sorted keys, no insignificant whitespace, UTF-8. Any change to any other
    field, including the key fingerprint, changes these bytes and breaks the signature.
    """
    body = {k: v for k, v in receipt.items() if k not in _UNSIGNED_FIELDS}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sign_receipt(
    receipt: dict[str, Any],
    *,
    key: SigningKey | None = None,
    key_path: str | Path | None = None,
    create_key: bool = True,
) -> dict[str, Any]:
    """Return a copy of *receipt* carrying an Ed25519 signature.

    Parameters
    ----------
    receipt :
        Any JSON-serialisable receipt dict.
    key :
        A loaded :class:`SigningKey`. When omitted the key at *key_path* is used.
    key_path :
        Where to load the key from. Defaults to :func:`default_key_path`.
    create_key :
        Create the key when it does not exist yet. Set ``False`` in a context where a missing
        key should be an error rather than a new identity.
    """
    signing_key = key
    if signing_key is None:
        signing_key = (
            load_or_create_signing_key(key_path) if create_key else load_signing_key(key_path)
        )

    signed = dict(receipt)
    signed["signature"] = None
    signed["signing"] = f"ed25519 (qb-compiler key {signing_key.fingerprint})"
    signed["signing_algorithm"] = "ed25519"
    signed["signature_scheme"] = SIGNATURE_SCHEME
    signed["key_fingerprint"] = signing_key.fingerprint
    signed["key_id"] = signing_key.key_id
    # A key that travels with the signature attests to nothing; drop it if an older receipt is
    # being re-signed.
    signed.pop("public_key", None)

    signature = _raw_sign(signing_key.seed, canonical_payload(signed))
    signed["signature"] = base64.b64encode(signature).decode("ascii")
    return signed


# ── verification ─────────────────────────────────────────────────────

#: Verdicts, and the one that means the signature checked out.
VERIFIED = "VERIFIED"
UNSIGNED = "UNSIGNED"
NO_KEY = "NO_KEY"
INVALID_SIGNATURE = "INVALID_SIGNATURE"
LEGACY_SELF_SIGNED = "LEGACY_SELF_SIGNED"
MALFORMED = "MALFORMED"


@dataclass(frozen=True)
class ReceiptVerification:
    """Outcome of checking one receipt.

    ``ok`` is ``True`` only for :data:`VERIFIED`. Every other verdict, including "this receipt is
    unsigned" and "no key was supplied", is a refusal to attest, not a pass.
    """

    status: str
    ok: bool
    reason: str
    schema: str | None = None
    key_fingerprint: str | None = None
    claims: list[str] = field(default_factory=list)
    does_not_claim: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ok": self.ok,
            "reason": self.reason,
            "schema": self.schema,
            "key_fingerprint": self.key_fingerprint,
            "claims": list(self.claims),
            "does_not_claim": list(self.does_not_claim),
        }

    def __str__(self) -> str:
        head = f"{self.status}: {self.reason}"
        if self.schema:
            head = f"{head}\n  schema      : {self.schema}"
        if self.key_fingerprint:
            head = f"{head}\n  key         : {self.key_fingerprint}"
        for line in self.claims:
            head = f"{head}\n  claims      : {line}"
        for line in self.does_not_claim:
            head = f"{head}\n  does not say: {line}"
        return head


_GENERIC_CLAIMS = [
    "these exact bytes were signed by the holder of the key named above",
]
_GENERIC_NON_CLAIMS = [
    "nothing about whether the numbers inside are correct",
    "nothing about whether the run described was worth doing",
    "no endorsement by QubitBoost of the party that signed it",
]

_SCHEMA_CLAIMS: dict[str, tuple[list[str], list[str]]] = {
    "qb.selection_receipt.v1": (
        [
            "which layout was selected, and whether that layout is the one that ran",
            "the calibration fingerprint and the measured age of that calibration",
        ],
        [
            "no claim that the selected layout is optimal",
            "the staleness tolerance is a builtin default, not a measurement for that device",
        ],
    ),
    "qb.chem_audit.v1": (
        ["which of the five integrity checks a Hamiltonian file passed or failed"],
        [
            "no claim about the accuracy of the chemistry, only about the file's own declarations",
        ],
    ),
    "qb.measure_plan.v1": (
        ["the term, setting and shot counts implied by the operator, from its structure alone"],
        [
            "no variance-weighted shot allocation and no claim about estimator error",
        ],
    ),
    "observablegate.receipt/1": (
        ["the result of a decoder-input audit on the supplied detector error model"],
        ["no claim that a decode on that model is faithful"],
    ),
}


def _claims_for(schema: str | None) -> tuple[list[str], list[str]]:
    claims, non_claims = _SCHEMA_CLAIMS.get(schema or "", ([], []))
    return list(claims) + _GENERIC_CLAIMS, list(non_claims) + _GENERIC_NON_CLAIMS


def verify_receipt(
    receipt: dict[str, Any],
    *,
    public_key: str | bytes | None = None,
    trusted_keys: list[bytes] | None = None,
    trusted_keys_path: str | Path | None = None,
) -> ReceiptVerification:
    """Check the signature on *receipt* against keys the caller supplies.

    Parameters
    ----------
    receipt :
        A receipt dict, as loaded from JSON.
    public_key :
        One key, as raw bytes, base64, or hex.
    trusted_keys :
        Several keys, already decoded.
    trusted_keys_path :
        A trusted-keys file. When none of the three is given, the default trusted-keys file is
        read, and if that is empty a signed receipt is reported :data:`NO_KEY` rather than passed.

    Returns
    -------
    ReceiptVerification
        ``ok`` is true only when a signature verified against one of the supplied keys.
    """
    if not isinstance(receipt, dict):
        return ReceiptVerification(
            status=MALFORMED, ok=False, reason="receipt is not a JSON object"
        )

    schema = receipt.get("schema")
    schema_str = str(schema) if schema is not None else None
    claims, non_claims = _claims_for(schema_str)
    signature_b64 = receipt.get("signature")

    if not signature_b64:
        return ReceiptVerification(
            status=UNSIGNED,
            ok=False,
            reason=(
                "receipt carries no signature, so its origin cannot be checked. Its contents may "
                "still be read; nothing attests to where they came from."
            ),
            schema=schema_str,
            claims=[],
            does_not_claim=non_claims,
        )

    if "public_key" in receipt:
        return ReceiptVerification(
            status=LEGACY_SELF_SIGNED,
            ok=False,
            reason=(
                "receipt embeds the public key its own signature was made with. Signatures made "
                "before qb-compiler 0.12.0 used a fresh keypair per receipt and carried it inline, "
                "so they attest to nothing. Re-issue the receipt with a persistent key."
            ),
            schema=schema_str,
            key_fingerprint=receipt.get("key_fingerprint"),
            claims=[],
            does_not_claim=non_claims,
        )

    try:
        signature = base64.b64decode(str(signature_b64), validate=True)
    except Exception:
        return ReceiptVerification(
            status=MALFORMED,
            ok=False,
            reason="signature field is not valid base64",
            schema=schema_str,
        )
    if len(signature) != 64:
        return ReceiptVerification(
            status=MALFORMED,
            ok=False,
            reason=f"signature is {len(signature)} bytes, expected 64",
            schema=schema_str,
        )

    scheme = receipt.get("signature_scheme")
    if scheme not in (None, SIGNATURE_SCHEME):
        return ReceiptVerification(
            status=MALFORMED,
            ok=False,
            reason=f"unknown signature scheme {scheme!r}; this verifier reads {SIGNATURE_SCHEME}",
            schema=schema_str,
        )

    candidates: list[bytes] = []
    if public_key is not None:
        candidates.append(parse_public_key(public_key))
    if trusted_keys:
        candidates.extend(trusted_keys)
    if not candidates:
        candidates.extend(load_trusted_keys(trusted_keys_path))

    fingerprint = receipt.get("key_fingerprint")
    fingerprint_str = str(fingerprint) if fingerprint is not None else None

    if not candidates:
        return ReceiptVerification(
            status=NO_KEY,
            ok=False,
            reason=(
                "receipt is signed but no public key was supplied, so the signature cannot be "
                "checked. Pass the signer's public key, or add it to the trusted-keys file."
            ),
            schema=schema_str,
            key_fingerprint=fingerprint_str,
            claims=[],
            does_not_claim=non_claims,
        )

    payload = canonical_payload(receipt)
    for candidate in candidates:
        if _raw_verify(candidate, payload, signature):
            actual = public_key_fingerprint(candidate)
            if fingerprint_str and fingerprint_str != actual:
                # Cannot happen through a normal path: the fingerprint is inside the signed bytes.
                return ReceiptVerification(  # pragma: no cover
                    status=INVALID_SIGNATURE,
                    ok=False,
                    reason=(
                        f"signature verified under key {actual} but the receipt names "
                        f"{fingerprint_str}"
                    ),
                    schema=schema_str,
                    key_fingerprint=actual,
                )
            return ReceiptVerification(
                status=VERIFIED,
                ok=True,
                reason=f"signature verified against key {actual}",
                schema=schema_str,
                key_fingerprint=actual,
                claims=claims,
                does_not_claim=non_claims,
            )

    return ReceiptVerification(
        status=INVALID_SIGNATURE,
        ok=False,
        reason=(
            "signature did not verify against any supplied key. Either the receipt was altered "
            "after signing, or it was signed by a key you do not have."
        ),
        schema=schema_str,
        key_fingerprint=fingerprint_str,
        claims=[],
        does_not_claim=non_claims,
    )


def verify_receipt_file(
    path: str | Path,
    *,
    public_key: str | bytes | None = None,
    trusted_keys_path: str | Path | None = None,
) -> ReceiptVerification:
    """Load a receipt from *path* and verify it. Unreadable files are :data:`MALFORMED`."""
    try:
        receipt = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ReceiptVerification(status=MALFORMED, ok=False, reason=f"cannot read {path}: {exc}")
    return verify_receipt(
        receipt,
        public_key=public_key,
        trusted_keys_path=trusted_keys_path,
    )
