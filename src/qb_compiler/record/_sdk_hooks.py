# SPDX-License-Identifier: Apache-2.0
"""Report sinks. External consumers register here."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from qb_compiler.record.types import ResidualReport, ValidationReport


def on_validation_report(report: ValidationReport) -> None:
    """Report sink. External consumers register here."""


def on_residual_report(report: ResidualReport) -> None:
    """Report sink. External consumers register here."""


__all__ = ["on_residual_report", "on_validation_report"]
