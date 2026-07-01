"""Calibration provider registry — the central dispatcher that makes qb-compiler multi-platform.

Maps a backend name to the best available calibration provider via ``BackendSpec.provider`` and
an ordered per-platform strategy, wraps it in a TTL cache, and **always degrades to static**
(real fixture, else synthetic) so it never raises for a known backend. Also exposes an honest,
machine-readable per-backend status so callers (and the website) can report live-vs-static
truthfully instead of by claim.

Adding a platform = add a factory + a ``STRATEGY_MAP`` entry; nothing else changes.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from qb_compiler.calibration.cached_provider import CachedCalibrationProvider
from qb_compiler.calibration.static_provider import StaticCalibrationProvider
from qb_compiler.config import BACKEND_CONFIGS, get_backend_spec

if TYPE_CHECKING:
    from collections.abc import Callable

    from qb_compiler.calibration.provider import CalibrationProvider

logger = logging.getLogger(__name__)


class LiveStatus(str, Enum):
    """Declared truth about a backend's live-calibration support (updated as platforms validate)."""

    LIVE = "live"  # adapter exists, deps importable, validated on real hardware
    LIVE_UNVALIDATED = "live-unvalidated"  # adapter exists, not yet smoke-tested on real hardware
    STATIC = "static"  # only fixture/synthetic data
    NONE = "none"  # no live adapter at all


# ── live-provider factories (each lazy-imports its SDK inside the provider) ──
def _make_ibm(backend: str) -> CalibrationProvider:
    from qb_compiler.calibration.ibm_runtime_provider import IBMRuntimeCalibrationProvider

    return IBMRuntimeCalibrationProvider(backend)


def _make_braket(backend: str) -> CalibrationProvider:
    from qb_compiler.calibration.braket_provider import BraketCalibrationProvider

    return BraketCalibrationProvider(backend)


def _make_quantinuum(backend: str) -> CalibrationProvider:
    from qb_compiler.calibration.quantinuum_provider import QuantinuumCalibrationProvider

    return QuantinuumCalibrationProvider(backend)


def _make_azure(backend: str) -> CalibrationProvider:
    from qb_compiler.calibration.azure_provider import AzureQuantumCalibrationProvider

    return AzureQuantumCalibrationProvider(backend)


def _ibm_deps() -> bool:
    return importlib.util.find_spec("qiskit_ibm_runtime") is not None


def _braket_deps() -> bool:
    return importlib.util.find_spec("braket") is not None


def _quantinuum_deps() -> bool:
    return importlib.util.find_spec("pytket.extensions.quantinuum") is not None


def _azure_deps() -> bool:
    return importlib.util.find_spec("azure.quantum") is not None


@dataclass(frozen=True)
class ProviderStrategy:
    """Ordered live-provider factories for a platform, its declared status, and a deps probe."""

    live_factories: tuple[Callable[[str], CalibrationProvider], ...]
    declared_status: LiveStatus
    deps_probe: Callable[[], bool]


#: Keyed by ``BackendSpec.provider``. ``declared_status`` is the maintained truth that drives honest
#: reporting; it flips to ``LIVE`` only after a real-device validation pass.
STRATEGY_MAP: dict[str, ProviderStrategy] = {
    "ibm": ProviderStrategy((_make_ibm,), LiveStatus.LIVE, _ibm_deps),
    # IonQ/Rigetti: Braket primary, Azure Quantum as a secondary access path.
    "ionq": ProviderStrategy(
        (_make_braket, _make_azure), LiveStatus.LIVE_UNVALIDATED, _braket_deps
    ),
    "rigetti": ProviderStrategy(
        (_make_braket, _make_azure), LiveStatus.LIVE_UNVALIDATED, _braket_deps
    ),
    "iqm": ProviderStrategy((_make_braket,), LiveStatus.LIVE_UNVALIDATED, _braket_deps),
    # Quantinuum: pytket-quantinuum primary, Azure secondary (was NONE before P3).
    "quantinuum": ProviderStrategy(
        (_make_quantinuum, _make_azure), LiveStatus.LIVE_UNVALIDATED, _quantinuum_deps
    ),
}

_STATIC_ONLY = ProviderStrategy((), LiveStatus.STATIC, lambda: False)


def _real_fixture(backend: str) -> bool:
    """True iff a real (non-synthetic) calibration fixture exists on disk for *backend*."""
    from qb_compiler.compiler import _load_calibration_fixture

    try:
        return _load_calibration_fixture(backend) is not None
    except Exception:
        return False


def _static_provider(backend: str) -> StaticCalibrationProvider:
    """Never-failing static fallback: real fixture if present, else a synthetic snapshot."""
    from qb_compiler.compiler import _build_synthetic_calibration, _load_calibration_fixture

    props = _load_calibration_fixture(backend)
    if props is None:
        spec = get_backend_spec(backend)
        props = _build_synthetic_calibration(spec, backend, spec.n_qubits)
    return StaticCalibrationProvider(props)


def get_calibration_provider(
    backend: str,
    *,
    prefer_live: bool = True,
    max_age_hours: float = 1.0,
    hard_limit_hours: float = 24.0,
) -> CalibrationProvider:
    """Return the best available calibration provider for *backend*.

    Tries the platform's live factories in order (skipping any that raise — missing SDK, no
    credentials, network/device error), then **always** falls back to static (real fixture, else
    synthetic). Wrapped in a :class:`CachedCalibrationProvider` (TTL). Never raises for a known
    backend; raises ``BackendNotSupportedError`` only for an unknown one.
    """
    spec = get_backend_spec(backend)  # validates the backend
    strategy = STRATEGY_MAP.get(spec.provider, _STATIC_ONLY)

    def factory() -> CalibrationProvider:
        if prefer_live:
            for make in strategy.live_factories:
                try:
                    return make(backend)
                except Exception as exc:  # missing SDK / creds / network -> next, then static
                    logger.info("live calibration for %s via %s failed: %s", backend, make, exc)
        return _static_provider(backend)

    return CachedCalibrationProvider(
        factory,
        max_age_seconds=max_age_hours * 3600.0,
        hard_limit_hours=hard_limit_hours,
    )


# ── honest per-backend status ───────────────────────────────────────────
@dataclass(frozen=True)
class BackendStatus:
    """Truthful live-vs-static status for one backend (drives reporting, e.g. website badges)."""

    backend: str
    provider: str
    live_status: LiveStatus  # declared truth
    live_deps_available: bool  # live SDK importable right now (no network)
    static_available: bool  # a real (non-synthetic) calibration fixture is present

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "provider": self.provider,
            "live_status": self.live_status.value,
            "live_deps_available": self.live_deps_available,
            "static_available": self.static_available,
        }


def get_backend_status(backend: str) -> BackendStatus:
    """Truthful status for a single backend."""
    spec = get_backend_spec(backend)
    strategy = STRATEGY_MAP.get(spec.provider, _STATIC_ONLY)
    return BackendStatus(
        backend=backend,
        provider=spec.provider,
        live_status=strategy.declared_status,
        live_deps_available=strategy.deps_probe(),
        static_available=_real_fixture(backend),
    )


def all_backend_statuses() -> list[BackendStatus]:
    """Truthful status for every registered backend."""
    return [get_backend_status(b) for b in BACKEND_CONFIGS]
