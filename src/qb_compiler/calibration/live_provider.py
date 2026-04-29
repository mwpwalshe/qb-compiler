"""Live calibration provider — connects to the QubitBoost calibration_hub API.

This is a **proprietary integration stub**.  Full functionality requires
the ``qubitboost-sdk`` package (``pip install qb-compiler[qubitboost]``).

When ``qubitboost-sdk`` is installed, :class:`LiveCalibrationProvider`
delegates to :class:`qubitboost_sdk.calibration.CalibrationHub` to fetch
real-time calibration snapshots from the QubitBoost calibration service.
This provides:

- Sub-hour calibration freshness (vs. 24h+ from public provider APIs)
- Unified format across IBM, IonQ, IQM, and Rigetti backends
- Automatic caching with configurable TTL
- Pre-computed noise model metadata for faster compilation

Without the SDK, instantiating this class raises a descriptive
:class:`ImportError` with installation instructions.
"""

from __future__ import annotations

import importlib.metadata
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx

from qb_compiler.calibration.provider import CalibrationProvider

if TYPE_CHECKING:
    from qb_compiler.calibration.models.coupling_properties import GateProperties
    from qb_compiler.calibration.models.qubit_properties import QubitProperties

try:
    from qubitboost_sdk.calibration import CalibrationHub

    _HAS_QUBITBOOST = True
except ImportError:
    _HAS_QUBITBOOST = False


def _get_version() -> str:
    """Return the installed qb-compiler version, or 'dev' if unavailable."""
    try:
        return importlib.metadata.version("qb-compiler")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def create_hardened_http_client(**overrides: Any) -> httpx.Client:
    """Create an ``httpx.Client`` with secure default settings.

    Parameters
    ----------
    **overrides:
        Keyword arguments forwarded to :class:`httpx.Client`, overriding
        the hardened defaults where specified.

    Returns
    -------
    httpx.Client
        A configured HTTP client with timeouts, connection limits,
        redirect policy, and a descriptive User-Agent header.
    """
    defaults: dict[str, Any] = {
        "timeout": httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0),
        "limits": httpx.Limits(max_connections=10, max_keepalive_connections=5),
        "follow_redirects": True,
        "max_redirects": 5,
        "headers": {
            "User-Agent": f"qb-compiler/{_get_version()} (https://qubitboost.io)",
        },
    }
    defaults.update(overrides)
    return httpx.Client(**defaults)


class LiveCalibrationProvider(CalibrationProvider):
    """Calibration provider that fetches live data from QubitBoost.

    Connects to the QubitBoost calibration_hub API to retrieve the
    latest calibration snapshots for a given backend.  Snapshots are
    updated every 30-60 minutes by the QubitBoost calibration service,
    which aggregates data from vendor APIs (IBM Quantum, AWS Braket,
    IQM Cloud) into a normalised format.

    Parameters
    ----------
    backend:
        Target backend identifier (e.g. ``"ibm_fez"``).
    api_key:
        QubitBoost API key.  If *None*, reads from the
        ``QUBITBOOST_API_KEY`` environment variable.
    cache_ttl_minutes:
        How long to cache calibration data before re-fetching.
        Default is 30 minutes.

    Raises
    ------
    ImportError
        If ``qubitboost-sdk`` is not installed.
    """

    def __init__(
        self,
        backend: str,
        *,
        api_key: str | None = None,
        account: str = "qubitboost_cloud",
        cache_ttl_minutes: float = 30.0,
    ) -> None:
        """Initialise the live calibration provider.

        Parameters
        ----------
        backend:
            Vendor backend identifier (e.g. ``"ibm_fez"``).
        api_key:
            Reserved for future remote-API mode; not used in v0.5
            (in-process tier authenticates via the saved ``account``
            credential profile).
        account:
            Saved-credential profile name to authenticate with.
            Default ``"qubitboost_cloud"``. External users must first
            run::

                from qiskit_ibm_runtime import QiskitRuntimeService
                QiskitRuntimeService.save_account(name="my_account",
                                                  channel="ibm_quantum",
                                                  token="...")

            then pass ``account="my_account"``.
        cache_ttl_minutes:
            Snapshot TTL in minutes. Default 30. The hub is configured
            with the same value; no inconsistency between them.
        """
        if not _HAS_QUBITBOOST:
            raise ImportError(
                "Live calibration requires qubitboost-sdk. "
                "Install with: pip install qb-compiler[qubitboost]"
            )

        self._backend = backend
        # Pass cache_ttl down to the hub so both layers agree on freshness.
        self._hub = CalibrationHub(
            api_key=api_key,
            account=account,
            max_age_minutes=cache_ttl_minutes,
        )
        self._cache_ttl = cache_ttl_minutes
        self._snapshot = self._hub.get_latest(backend)

    # ── CalibrationProvider interface ─────────────────────────────────

    def get_qubit_properties(self, qubit: int) -> QubitProperties | None:
        """Return calibration data for *qubit* from the latest snapshot."""
        return self._snapshot.get_qubit_properties(qubit)  # type: ignore[no-any-return]

    def get_gate_properties(self, gate: str, qubits: tuple[int, ...]) -> GateProperties | None:
        """Return gate calibration data from the latest snapshot."""
        return self._snapshot.get_gate_properties(gate, qubits)  # type: ignore[no-any-return]

    def get_all_qubit_properties(self) -> list[QubitProperties]:
        """Return all qubit calibrations from the latest snapshot."""
        return self._snapshot.get_all_qubit_properties()  # type: ignore[no-any-return]

    def get_all_gate_properties(self) -> list[GateProperties]:
        """Return all gate calibrations from the latest snapshot."""
        return self._snapshot.get_all_gate_properties()  # type: ignore[no-any-return]

    @property
    def backend_name(self) -> str:
        """The backend this provider is configured for."""
        return self._backend

    @property
    def timestamp(self) -> datetime:
        """UTC timestamp of the calibration snapshot."""
        return self._snapshot.timestamp  # type: ignore[no-any-return]

    # ── Extended API ──────────────────────────────────────────────────

    def refresh(self) -> None:
        """Force a re-fetch of calibration data from the API.

        Bypasses the hub's TTL cache to guarantee a fresh vendor call.
        Use ``cache_ttl_minutes`` (init-time) to control implicit
        re-fetching via :attr:`is_stale`; use this method for explicit
        invalidation (e.g. before a critical experiment).
        """
        if hasattr(self._hub, "fetch"):
            self._snapshot = self._hub.fetch(self._backend)
        else:
            # Older qubitboost-sdk versions only exposed get_latest. Fall
            # back; freshness will then respect the hub's TTL only.
            self._snapshot = self._hub.get_latest(self._backend)

    @property
    def is_stale(self) -> bool:
        """Whether the cached snapshot exceeds the configured TTL."""
        age_min = (datetime.now(timezone.utc) - self.timestamp).total_seconds() / 60.0
        return age_min > self._cache_ttl
