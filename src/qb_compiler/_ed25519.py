# SPDX-License-Identifier: Apache-2.0
"""Pure-Python Ed25519, used only when a compiled implementation is unavailable.

Why this exists. Receipt verification has to work for a third party who was handed a receipt and
has no relationship with us, which means it cannot be gated behind an optional wheel. So
:mod:`qb_compiler.signing` prefers ``cryptography`` when it is importable and falls back to the
code here otherwise, and ``pip install qb-compiler`` alone is enough to check a signature.

The implementation follows the reference code in RFC 8032 section 6, which is public domain. It is
straightforward big-integer arithmetic with no side-channel hardening: it is fine for verifying a
signature and for signing a receipt on your own machine, and it is not appropriate for a key that
faces an attacker who can time it. Signing a handful of receipts costs a few milliseconds, so the
speed difference against ``cryptography`` does not matter at this scale.

Test vectors from RFC 8032 section 7.1 are pinned in the test suite, and every operation is
cross-checked against ``cryptography`` when that package is installed.
"""

from __future__ import annotations

import hashlib

# Curve constants, RFC 8032 section 5.1.
_P = 2**255 - 19
_Q = 2**252 + 27742317777372353535851937790883648493

_SEED_BYTES = 32
_PUBLIC_BYTES = 32
_SIGNATURE_BYTES = 64

Point = tuple[int, int, int, int]


def _sha512(data: bytes) -> bytes:
    return hashlib.sha512(data).digest()


def _sha512_modq(data: bytes) -> int:
    return int.from_bytes(_sha512(data), "little") % _Q


def _modp_inv(x: int) -> int:
    return pow(x, _P - 2, _P)


_D = -121665 * _modp_inv(121666) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)


def _recover_x(y: int, sign: int) -> int | None:
    """Recover the x coordinate of a curve point from y and the sign bit."""
    if y >= _P:
        return None
    x2 = (y * y - 1) * _modp_inv(_D * y * y + 1) % _P
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P != 0:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


_G_Y = 4 * _modp_inv(5) % _P
_G_X = _recover_x(_G_Y, 0)
if _G_X is None:  # pragma: no cover - a fixed curve constant; unreachable on a sane interpreter
    raise RuntimeError("ed25519 base point could not be recovered")
_G: Point = (_G_X, _G_Y, 1, _G_X * _G_Y % _P)


def _point_add(p1: Point, p2: Point) -> Point:
    """Add two points in extended homogeneous coordinates."""
    a = (p1[1] - p1[0]) * (p2[1] - p2[0]) % _P
    b = (p1[1] + p1[0]) * (p2[1] + p2[0]) % _P
    c = 2 * p1[3] * p2[3] * _D % _P
    dd = 2 * p1[2] * p2[2] % _P
    e, f, g, h = b - a, dd - c, dd + c, b + a
    return (e * f % _P, g * h % _P, f * g % _P, e * h % _P)


def _point_mul(scalar: int, point: Point) -> Point:
    """Multiply *point* by *scalar* using double-and-add."""
    out: Point = (0, 1, 1, 0)
    while scalar > 0:
        if scalar & 1:
            out = _point_add(out, point)
        point = _point_add(point, point)
        scalar >>= 1
    return out


def _point_equal(p1: Point, p2: Point) -> bool:
    if (p1[0] * p2[2] - p2[0] * p1[2]) % _P != 0:
        return False
    return (p1[1] * p2[2] - p2[1] * p1[2]) % _P == 0


def _point_compress(point: Point) -> bytes:
    zinv = _modp_inv(point[2])
    x = point[0] * zinv % _P
    y = point[1] * zinv % _P
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _point_decompress(data: bytes) -> Point | None:
    if len(data) != _PUBLIC_BYTES:
        return None
    y = int.from_bytes(data, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, x * y % _P)


def _secret_expand(seed: bytes) -> tuple[int, bytes]:
    if len(seed) != _SEED_BYTES:
        raise ValueError(f"ed25519 seed must be {_SEED_BYTES} bytes, got {len(seed)}")
    h = _sha512(seed)
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def public_key(seed: bytes) -> bytes:
    """Derive the 32-byte public key for a 32-byte private seed."""
    a, _ = _secret_expand(seed)
    return _point_compress(_point_mul(a, _G))


def sign(seed: bytes, message: bytes) -> bytes:
    """Produce the 64-byte Ed25519 signature of *message* under *seed*."""
    a, prefix = _secret_expand(seed)
    pub = _point_compress(_point_mul(a, _G))
    r = _sha512_modq(prefix + message)
    rs = _point_compress(_point_mul(r, _G))
    h = _sha512_modq(rs + pub + message)
    s = (r + h * a) % _Q
    return rs + int.to_bytes(s, 32, "little")


def verify(pub: bytes, message: bytes, signature: bytes) -> bool:
    """Check an Ed25519 *signature*. Returns ``False`` on any malformed input."""
    if len(pub) != _PUBLIC_BYTES or len(signature) != _SIGNATURE_BYTES:
        return False
    point_a = _point_decompress(pub)
    if point_a is None:
        return False
    rs = signature[:32]
    point_r = _point_decompress(rs)
    if point_r is None:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _Q:
        return False
    h = _sha512_modq(rs + pub + message)
    return _point_equal(_point_mul(s, _G), _point_add(point_r, _point_mul(h, point_a)))
