"""Opt-in telemetry collector for qb-compiler.

Disabled by default.  When enabled, collects anonymous compilation
statistics (depth reduction, gate counts, compilation time) to help
improve the compiler.  No circuit content is ever transmitted.

Enable via environment variable ``QBC_TELEMETRY=1`` or by calling
:meth:`TelemetryCollector.enable`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CompilationEvent:
    """A single compilation telemetry record."""

    backend: str | None = None
    strategy: str = ""
    original_depth: int = 0
    compiled_depth: int = 0
    gate_count_before: int = 0
    gate_count_after: int = 0
    compilation_time_ms: float = 0.0
    n_qubits: int = 0


class TelemetryCollector:
    """Collects and (optionally) exports compilation telemetry.

    Parameters
    ----------
    enabled:
        Whether to collect telemetry.  Defaults to ``False`` unless the
        ``QBC_TELEMETRY`` environment variable is set to ``"1"``.
    """

    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            enabled = os.environ.get("QBC_TELEMETRY", "0") == "1"
        self._enabled = enabled
        self._events: list[CompilationEvent] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        """Enable telemetry collection."""
        self._enabled = True

    def disable(self) -> None:
        """Disable telemetry collection and clear buffered events."""
        self._enabled = False
        self._events.clear()

    def record(self, event: CompilationEvent) -> None:
        """Record a compilation event (no-op if disabled)."""
        if not self._enabled:
            return
        self._events.append(event)

    def record_compilation(self, **kwargs: Any) -> None:
        """Convenience: record a compilation event from keyword args."""
        if not self._enabled:
            return
        self._events.append(CompilationEvent(**kwargs))

    @property
    def events(self) -> list[CompilationEvent]:
        """Return a copy of collected events."""
        return list(self._events)

    def flush(self) -> list[CompilationEvent]:
        """Return and clear all buffered events."""
        evts = list(self._events)
        self._events.clear()
        return evts

    def __repr__(self) -> str:
        return (
            f"TelemetryCollector(enabled={self._enabled}, "
            f"buffered={len(self._events)})"
        )
