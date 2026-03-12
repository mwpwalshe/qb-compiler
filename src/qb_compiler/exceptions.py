"""Exception hierarchy for the qb-compiler package.

All public exceptions inherit from :class:`QBCompilerError` so callers
can catch the base class for blanket handling.
"""

from __future__ import annotations


class QBCompilerError(Exception):
    """Base exception for all qb-compiler errors."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        self.detail = detail
        super().__init__(message)


# ── compilation ──────────────────────────────────────────────────────

class CompilationError(QBCompilerError):
    """Raised when circuit compilation fails."""


class InvalidCircuitError(CompilationError):
    """Raised when the input circuit is malformed or contains unsupported ops."""

    def __init__(self, message: str, *, gate: str | None = None) -> None:
        self.gate = gate
        super().__init__(message, detail=f"unsupported gate: {gate}" if gate else None)


# ── calibration ──────────────────────────────────────────────────────

class CalibrationError(QBCompilerError):
    """Base class for calibration-related errors."""


class CalibrationStaleError(CalibrationError):
    """Raised when cached calibration data exceeds the configured max age."""

    def __init__(self, backend: str, age_hours: float, max_hours: float) -> None:
        self.backend = backend
        self.age_hours = age_hours
        self.max_hours = max_hours
        super().__init__(
            f"Calibration for {backend} is {age_hours:.1f}h old "
            f"(max allowed: {max_hours:.1f}h)",
        )


class CalibrationNotFoundError(CalibrationError):
    """Raised when no calibration data exists for the requested backend."""

    def __init__(self, backend: str) -> None:
        self.backend = backend
        super().__init__(f"No calibration data found for backend '{backend}'")


# ── backend / budget ─────────────────────────────────────────────────

class BackendNotSupportedError(QBCompilerError):
    """Raised when a requested backend is not in the known configuration."""

    def __init__(self, backend: str, available: list[str] | None = None) -> None:
        self.backend = backend
        self.available = available or []
        avail_str = ", ".join(self.available) if self.available else "none"
        super().__init__(
            f"Backend '{backend}' is not supported. Available: {avail_str}"
        )


class BudgetExceededError(QBCompilerError):
    """Raised when estimated execution cost exceeds the caller's budget."""

    def __init__(
        self,
        estimated_usd: float,
        budget_usd: float,
        *,
        shots: int | None = None,
    ) -> None:
        self.estimated_usd = estimated_usd
        self.budget_usd = budget_usd
        self.shots = shots
        super().__init__(
            f"Estimated cost ${estimated_usd:.4f} exceeds budget "
            f"${budget_usd:.4f}"
            + (f" ({shots} shots)" if shots else ""),
        )
