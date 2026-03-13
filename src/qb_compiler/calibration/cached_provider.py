"""Thread-safe caching wrapper around any :class:`CalibrationProvider`."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from qb_compiler.calibration.provider import CalibrationProvider
from qb_compiler.exceptions import CalibrationStaleError

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from qb_compiler.calibration.models.coupling_properties import GateProperties
    from qb_compiler.calibration.models.qubit_properties import QubitProperties


class CachedCalibrationProvider(CalibrationProvider):
    """Wraps another :class:`CalibrationProvider`, caching its results.

    The wrapper checks the cached data's age before every access and
    refreshes transparently when the cache is stale.

    Parameters
    ----------
    provider_factory:
        Callable that returns a *fresh* :class:`CalibrationProvider`.
        Called on first access and whenever the cache expires.
    max_age_seconds:
        Maximum cache lifetime in seconds.  Defaults to 3600 (1 hour).
    hard_limit_hours:
        If the provider factory raises and cached data is older than this
        many hours, a :class:`CalibrationStaleError` is raised instead of
        serving potentially very old data.  Defaults to 24.
    """

    def __init__(
        self,
        provider_factory: Callable[[], CalibrationProvider],
        *,
        max_age_seconds: float = 3600.0,
        hard_limit_hours: float = 24.0,
    ) -> None:
        self._factory = provider_factory
        self._max_age = max_age_seconds
        self._hard_limit_hours = hard_limit_hours

        self._lock = threading.Lock()
        self._inner: CalibrationProvider | None = None
        self._fetched_at: float = 0.0  # monotonic clock

    # ── internal cache management ────────────────────────────────────

    def _ensure_fresh(self) -> CalibrationProvider:
        """Return the inner provider, refreshing if stale.

        Thread-safe: only one thread performs the refresh; others wait.
        """
        now = time.monotonic()
        # Fast path — no lock needed for read if still fresh
        if self._inner is not None and (now - self._fetched_at) < self._max_age:
            return self._inner

        with self._lock:
            # Re-check after acquiring lock (another thread may have refreshed)
            now = time.monotonic()
            if self._inner is not None and (now - self._fetched_at) < self._max_age:
                return self._inner

            try:
                fresh = self._factory()
                self._inner = fresh
                self._fetched_at = time.monotonic()
                return fresh
            except Exception:
                # If we have stale data, check hard limit
                if self._inner is not None:
                    age_h = self._inner.age_hours
                    if age_h <= self._hard_limit_hours:
                        # Serve stale data rather than crash
                        return self._inner
                    raise CalibrationStaleError(
                        backend=self._inner.backend_name,
                        age_hours=age_h,
                        max_hours=self._hard_limit_hours,
                    ) from None
                raise  # No cached data at all — propagate original error

    # ── CalibrationProvider interface ────────────────────────────────

    def get_qubit_properties(self, qubit: int) -> QubitProperties | None:
        return self._ensure_fresh().get_qubit_properties(qubit)

    def get_gate_properties(
        self, gate: str, qubits: tuple[int, ...]
    ) -> GateProperties | None:
        return self._ensure_fresh().get_gate_properties(gate, qubits)

    def get_all_qubit_properties(self) -> list[QubitProperties]:
        return self._ensure_fresh().get_all_qubit_properties()

    def get_all_gate_properties(self) -> list[GateProperties]:
        return self._ensure_fresh().get_all_gate_properties()

    @property
    def backend_name(self) -> str:
        return self._ensure_fresh().backend_name

    @property
    def timestamp(self) -> datetime:
        return self._ensure_fresh().timestamp

    # ── manual control ───────────────────────────────────────────────

    def invalidate(self) -> None:
        """Force the next access to refresh from the factory."""
        with self._lock:
            # Set fetched_at far enough in the past to guarantee staleness
            self._fetched_at = time.monotonic() - self._max_age - 1.0

    def prefetch(self) -> None:
        """Eagerly populate the cache (useful at application startup)."""
        self._ensure_fresh()

    def __repr__(self) -> str:
        inner_repr = repr(self._inner) if self._inner else "not-yet-loaded"
        return f"CachedCalibrationProvider(inner={inner_repr}, max_age={self._max_age}s)"
