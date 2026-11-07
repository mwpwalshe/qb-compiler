# SPDX-License-Identifier: Apache-2.0
"""Tendrils: every vendor, format and toolkit qb-compiler can reach, with what each one declares.

``qbc tendrils`` lists them with their labels. ``qbc tendril check`` runs the conformance check
on one and writes a receipt that can be signed with the submitter's own key. Outside packages add
tendrils through the ``qb_compiler.tendrils`` entry point group, without changing this package.
"""

from __future__ import annotations

from qb_compiler.tendrils.conformance import (
    CONFORMS,
    DOES_NOT_CONFORM,
    SCHEMA,
    Check,
    check,
    conformance_receipt,
    manifest_sha256,
    verdict,
)
from qb_compiler.tendrils.discovery import (
    ENTRY_POINT_GROUP,
    BrokenTendril,
    discover,
    get_tendril,
)
from qb_compiler.tendrils.model import (
    CAPABILITIES,
    LABELS,
    VALIDATION_DAYS,
    PriceRow,
    Tendril,
    TendrilError,
    ValidationRecord,
)

__all__ = [
    "CAPABILITIES",
    "CONFORMS",
    "DOES_NOT_CONFORM",
    "ENTRY_POINT_GROUP",
    "LABELS",
    "SCHEMA",
    "VALIDATION_DAYS",
    "BrokenTendril",
    "Check",
    "PriceRow",
    "Tendril",
    "TendrilError",
    "ValidationRecord",
    "check",
    "conformance_receipt",
    "discover",
    "get_tendril",
    "manifest_sha256",
    "verdict",
]
