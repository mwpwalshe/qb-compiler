# SPDX-License-Identifier: Apache-2.0
"""Signed-passport-ready selection receipt for calibration-aware layout selection.

The receipt is the open-core funnel primitive: the *check* (running
:class:`~qb_compiler.passes.mapping.calibration_mapper.CalibrationMapper` and
seeing which physical qubits it chose and why) is free; the *signed, stored*
receipt is the product. This module derives a receipt purely from the mapper's
returned :class:`~qb_compiler.passes.base.PassResult` metadata, so it holds no
hidden state and never re-implements the layout objective.

The mapper already scores candidate layouts by gate error, coherence (T1/T2),
readout error, T1 asymmetry, and temporal correlation, then picks the best via
VF2 subgraph isomorphism. The receipt just makes that choice auditable and,
when the paid layer is present, cryptographically signed. No accuracy /
advantage / "beats SOTA" claim is made or implied.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

SCHEMA = "qb.selection_receipt.v1"
OBJECTIVE = (
    "calibration-aware layout (CalibrationMapper: gate error + coherence + "
    "readout + T1 asymmetry + temporal correlation, VF2 subgraph search)"
)


def _metadata(result: Any) -> dict:
    """Accept a PassResult, its metadata dict, or a pipeline ``context`` dict."""
    meta = getattr(result, "metadata", None)
    if isinstance(meta, dict):
        return meta
    if isinstance(result, dict):
        return result
    return {}


def calibration_fingerprint(source: Any) -> str | None:
    """Stable short fingerprint of the calibration a selection was made against.

    Prefers a provider/snapshot's ``backend_name`` + ``timestamp`` (the natural
    identity of a calibration reading); falls back to hashing a plain dict of
    error data; returns ``None`` if neither is available.
    """
    backend = getattr(source, "backend_name", None) or getattr(source, "backend", None)
    ts = getattr(source, "timestamp", None)
    if backend is not None and ts is not None:
        payload = f"{backend}@{ts.isoformat() if hasattr(ts, 'isoformat') else ts}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
    if isinstance(source, dict):
        return hashlib.sha256(json.dumps(source, sort_keys=True, default=str).encode()).hexdigest()[
            :16
        ]
    return None


def selection_receipt(
    result: Any,
    *,
    calibration: Any = None,
    calibration_hash: str | None = None,
    sign: bool = False,
) -> dict:
    """Build a selection receipt from a :class:`CalibrationMapper` result.

    Parameters
    ----------
    result :
        The ``PassResult`` returned by ``CalibrationMapper.run(...)`` (or its
        ``metadata`` dict, or a pipeline ``context`` dict). Must carry
        ``initial_layout``; ``calibration_score`` and ``score_breakdown`` are
        included when present.
    calibration :
        Optional calibration provider/snapshot/dict, hashed for provenance when
        ``calibration_hash`` is not given explicitly.
    calibration_hash :
        Pre-computed calibration fingerprint (overrides ``calibration``).
    sign :
        If ``True`` AND the paid ``qubitboost_sdk`` is importable, sign the
        receipt with Ed25519. Otherwise the receipt is emitted unsigned with a
        one-line pointer. No hard dependency; off by default; no nagging.
    """
    meta = _metadata(result)
    layout = meta.get("initial_layout") or meta.get("layout") or {}
    receipt = {
        "schema": SCHEMA,
        "objective": OBJECTIVE,
        "selected_layout": {str(k): v for k, v in dict(layout).items()},
        "selected_score": meta.get("calibration_score"),
        "score_breakdown": meta.get("score_breakdown", {}),
        "calibration_hash": calibration_hash
        or (calibration_fingerprint(calibration) if calibration is not None else None),
        "signature": None,
        "signing": "unsigned",
    }
    if sign:
        try:  # soft, optional bridge to the paid layer
            from qubitboost_sdk import Ed25519PassportSigner  # type: ignore

            priv, pub = Ed25519PassportSigner.generate_keypair()
            receipt["signature"] = Ed25519PassportSigner(private_key=priv, key_id="qbc").sign(
                receipt
            )
            receipt["signing"] = "ed25519 (qubitboost_sdk)"
            receipt["public_key"] = pub
        except Exception:
            receipt["signing"] = (
                "unsigned -- signed receipts available with the QubitBoost SDK (qubitboost.io)"
            )
    return receipt
