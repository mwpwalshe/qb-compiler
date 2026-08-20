# SPDX-License-Identifier: Apache-2.0
"""Selection receipt for calibration-aware layout selection.

The receipt is the open-core funnel primitive: the *check* (running
:class:`~qb_compiler.passes.mapping.calibration_mapper.CalibrationMapper` and
seeing which physical qubits it chose and why) is free; the *signed, stored*
receipt is the product. This module derives a receipt purely from the mapper's
returned :class:`~qb_compiler.passes.base.PassResult` metadata, so it holds no
hidden state and never re-implements the layout objective.

The mapper already scores candidate layouts by gate error, coherence (T1/T2),
readout error, T1 asymmetry, and temporal correlation, then picks the best via
VF2 subgraph isomorphism. The receipt just makes that choice auditable. No
accuracy, advantage or "beats SOTA" claim is made or implied.

Two things changed in 0.12.0, both corrections rather than features.

**The receipt can now describe the layout that actually ran.** Pass ``executed_layout=`` when you
run something other than the recommendation: the receipt then describes what you ran, keeps the
recommendation in its own field, and states the score you gave up. Before this, a receipt named
the pass's pick whatever the caller executed, which made it an attestation to a run that may never
have happened.

**Signing uses a key that outlives the call.** It previously generated a fresh keypair per receipt
and embedded the public half in the receipt it had just signed, which verifies against itself and
proves nothing about origin. See :mod:`qb_compiler.signing`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

SCHEMA = "qb.selection_receipt.v1"

#: Blunt, static staleness tolerance in minutes, used when nothing better is available.
#:
#: A layout is chosen from calibration data, and that data has an age. Nobody surfaces the age, so
#: nobody can act on it. This constant lets the free receipt say something useful about it without
#: pretending to a per-device measurement it does not have.
#:
#: 30 minutes is deliberately conservative and deliberately one number for every backend. Real
#: tolerance is a property of the specific device and moves with its recalibration schedule;
#: measuring it means comparing what was visible at a moment against what the provider later
#: reports was true at that moment, over weeks. That measurement is not in this package. See
#: ``calibration_freshness.tolerance_basis`` in the emitted receipt, which always says which of the
#: two a reader is looking at.
DEFAULT_STALENESS_TOLERANCE_MINUTES = 30.0

OBJECTIVE = (
    "calibration-aware layout (CalibrationMapper: gate error + coherence + "
    "readout + T1 asymmetry + temporal correlation, VF2 subgraph search)"
)

_FRESHNESS_NOTE = (
    "Age is measured, tolerance is a fixed default. A tolerance measured for this specific "
    "backend, and the publication delay that makes the reported age an over-estimate, are "
    "not derivable from this package."
)


def _metadata(result: Any) -> dict:
    """Accept a PassResult, its metadata dict, or a pipeline ``context`` dict."""
    meta = getattr(result, "metadata", None)
    if isinstance(meta, dict):
        return meta
    if isinstance(result, dict):
        return result
    return {}


def _as_datetime(value: Any) -> datetime | None:
    """Coerce a calibration timestamp to an aware datetime, or ``None`` if it cannot be read.

    Snapshots carry their timestamp in whatever shape their provider used: a ``datetime`` from the
    IBM runtime, an ISO-8601 string in a stored fixture, an epoch number from a cache. Anything
    else, including the literal ``"synthetic"`` this package writes when it is using static specs
    rather than a reading from a device, is not an age and is reported as not being one.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # datetime.fromisoformat on 3.10 does not accept a trailing Z.
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return None


def _raw_timestamp(source: Any) -> Any:
    """Pull the timestamp off a provider, a snapshot, or a plain dict."""
    ts = getattr(source, "timestamp", None)
    if ts is not None:
        return ts
    props = getattr(source, "backend_properties", None)
    if props is not None:
        ts = getattr(props, "timestamp", None)
        if ts is not None:
            return ts
    if isinstance(source, dict):
        for key in ("timestamp", "last_update_date", "calibration_timestamp"):
            if source.get(key) is not None:
                return source[key]
    return None


def calibration_freshness(source: Any, *, now: datetime | None = None) -> dict:
    """Age of the calibration a selection was made against, as a plain signal.

    Emitted with every receipt because it is a fact the caller cannot otherwise see and can
    certainly act on: layout choice is a function of calibration data, and calibration data has an
    age. A compiler that silently uses hour-old numbers and one that uses fresh ones look identical
    from the outside.

    This reports the age and compares it against one blunt constant. It does NOT claim to know the
    tolerance of your specific device, and ``tolerance_basis`` says so on every receipt, because a
    default presented as a measurement is worse than no number at all.

    ``age_minutes`` is ``None`` whenever an age is not knowable: no timestamp, one that cannot be
    read, or a snapshot marked synthetic. ``timestamp_status`` says which of those it was, so a
    reader never has to guess why a number is missing.
    """
    now = now or datetime.now(timezone.utc)
    raw = _raw_timestamp(source)
    out: dict = {
        "calibration_timestamp": None,
        "age_minutes": None,
        "tolerance_minutes": DEFAULT_STALENESS_TOLERANCE_MINUTES,
        "tolerance_basis": "builtin_default_not_measured_for_this_device",
        "exceeds_default_tolerance": None,
        "timestamp_status": "absent",
        "note": _FRESHNESS_NOTE,
    }
    if raw is None:
        return out

    if isinstance(raw, str) and raw.strip().lower() == "synthetic":
        out["timestamp_status"] = "synthetic"
        out["note"] = (
            "calibration is synthetic static specs rather than a reading from a device, so it has "
            "no age"
        )
        return out

    stamp = _as_datetime(raw)
    if stamp is None:
        out["timestamp_status"] = "unreadable"
        out["note"] = f"calibration timestamp {raw!r} could not be read as a time"
        return out

    age = (now - stamp).total_seconds() / 60.0
    out["calibration_timestamp"] = stamp.isoformat()
    out["age_minutes"] = round(age, 2)
    out["timestamp_status"] = "measured"
    # Negative age means the clocks disagree. Reporting that beats asserting freshness from it.
    if age < 0:
        out["timestamp_status"] = "clock_skew"
        out["note"] = "calibration timestamp is in the future; clocks disagree, age not usable"
        return out
    out["exceeds_default_tolerance"] = age > DEFAULT_STALENESS_TOLERANCE_MINUTES
    return out


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


def _normalise_layout(layout: Any) -> dict[str, int]:
    """Layouts are keyed by logical qubit; JSON keys are strings, so settle it in one place."""
    return {str(k): v for k, v in dict(layout or {}).items()}


def _divergence_note(
    executed: dict[str, int],
    recommended: dict[str, int],
    penalty: float | None,
) -> str:
    differing = sum(1 for k, v in executed.items() if recommended.get(k) != v)
    differing += sum(1 for k in recommended if k not in executed)
    total = len(recommended) or len(executed)
    tail = (
        f"the calibration score of what ran is {penalty:+.6f} against the recommendation"
        if penalty is not None
        else "no score was supplied for what ran, so the penalty is not computable"
    )
    return (
        f"OVERRIDDEN: the layout that ran is not the one the pass recommended. "
        f"{differing} of {total} logical qubits map elsewhere; {tail}."
    )


def selection_receipt(
    result: Any,
    *,
    calibration: Any = None,
    calibration_hash: str | None = None,
    executed_layout: Any = None,
    executed_score: float | None = None,
    scorer: Callable[[dict[int, int]], float] | None = None,
    sign: bool = False,
    signing_key: Any = None,
    key_path: str | None = None,
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
        ``calibration_hash`` is not given explicitly, and read for its age.
    calibration_hash :
        Pre-computed calibration fingerprint (overrides ``calibration``).
    executed_layout :
        The layout that actually ran, when it is not the pass's recommendation. Supplying it makes
        ``selected_layout`` the executed one, moves the recommendation into ``recommended_layout``,
        and records the score given up. Omit it and the receipt states, rather than implies, that
        the recommendation is what ran.
    executed_score :
        Calibration score of ``executed_layout``. When omitted and *scorer* is given it is
        computed; when neither is available the score and the penalty are ``None`` and the receipt
        says so.
    scorer :
        Callable taking a layout dict and returning its calibration score. Normally
        ``functools.partial(mapper.score_layout, circuit=circuit)`` or an equivalent binding of
        :meth:`CalibrationMapper.score_layout`.
    sign :
        Sign the receipt with the local Ed25519 key (see :mod:`qb_compiler.signing`). The key is
        created once on first use and reused afterwards. Raises
        :class:`~qb_compiler.signing.SigningError` when a signature was asked for and could not be
        produced: quietly handing back an unsigned receipt is the failure this replaced.
    signing_key :
        A loaded :class:`~qb_compiler.signing.SigningKey` to use instead of the default.
    key_path :
        Private key location, overriding ``QBC_SIGNING_KEY`` and the default path.

    Returns
    -------
    dict
        A JSON-serialisable receipt. ``describes_executed_layout`` is always present, so a
        consumer never has to infer whether ``selected_layout`` ran.
    """
    meta = _metadata(result)
    recommended = _normalise_layout(meta.get("initial_layout") or meta.get("layout") or {})
    recommended_score = meta.get("calibration_score")

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "objective": OBJECTIVE,
        "selected_layout": recommended,
        "selected_score": recommended_score,
        "score_breakdown": meta.get("score_breakdown", {}),
        "describes_executed_layout": True,
        "calibration_hash": calibration_hash
        or (calibration_fingerprint(calibration) if calibration is not None else None),
        "calibration_freshness": calibration_freshness(calibration),
        "signature": None,
        "signing": "unsigned",
    }

    if executed_layout is not None:
        executed = _normalise_layout(executed_layout)
        matches = executed == recommended
        score = executed_score
        if score is None and scorer is not None:
            score = float(scorer({int(k): v for k, v in executed.items()}))
        if score is None and matches:
            score = recommended_score

        penalty: float | None = None
        if score is not None and recommended_score is not None:
            penalty = round(float(score) - float(recommended_score), 12)

        receipt["selected_layout"] = executed
        receipt["selected_score"] = score
        receipt["recommended_layout"] = recommended
        receipt["recommended_score"] = recommended_score
        receipt["executed_layout_matches_recommendation"] = matches
        receipt["score_penalty_vs_recommended"] = penalty
        receipt["divergence_note"] = (
            "the executed layout is the one the pass recommended"
            if matches
            else _divergence_note(executed, recommended, penalty)
        )

        if not matches:
            # The breakdown belongs to the recommendation, and the mapper does not produce one for
            # an arbitrary layout. Dropping it beats leaving numbers that describe another mapping
            # sitting next to the layout that ran.
            receipt["score_breakdown"] = {}
            receipt["score_breakdown_note"] = (
                "the per signal breakdown is computed for the recommendation only, and is omitted "
                "here because a different layout ran"
            )

    if sign:
        from qb_compiler.signing import sign_receipt

        receipt = sign_receipt(receipt, key=signing_key, key_path=key_path)
    return receipt
