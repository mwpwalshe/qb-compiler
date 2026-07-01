"""Offline tests for the calibration provider registry + honest status API.

No credentials/network: the registry's static fallback (real fixture, else synthetic) always works,
and a raising live factory must degrade to static without raising.
"""

from __future__ import annotations

import pytest

from qb_compiler.calibration import registry
from qb_compiler.calibration.cached_provider import CachedCalibrationProvider
from qb_compiler.calibration.registry import (
    LiveStatus,
    ProviderStrategy,
    all_backend_statuses,
    get_backend_status,
    get_calibration_provider,
)
from qb_compiler.exceptions import BackendNotSupportedError


def test_static_fallback_returns_usable_provider():
    prov = get_calibration_provider("ibm_fez", prefer_live=False)
    assert isinstance(prov, CachedCalibrationProvider)
    props = prov.backend_properties  # triggers the factory -> static snapshot
    assert props is not None
    assert props.n_qubits > 0


def test_live_factory_failure_degrades_to_static(monkeypatch):
    def _boom(_backend):
        raise RuntimeError("no credentials")

    monkeypatch.setitem(
        registry.STRATEGY_MAP,
        "ibm",
        ProviderStrategy((_boom,), LiveStatus.LIVE, lambda: True),
    )
    prov = get_calibration_provider("ibm_fez", prefer_live=True)  # must NOT raise
    assert prov.backend_properties is not None  # fell back to static


def test_unknown_backend_raises():
    with pytest.raises(BackendNotSupportedError):
        get_calibration_provider("not_a_real_backend")


def test_status_truth_table():
    ibm = get_backend_status("ibm_fez")
    assert ibm.provider == "ibm"
    assert ibm.live_status == LiveStatus.LIVE

    ionq = get_backend_status("ionq_aria")
    assert ionq.live_status == LiveStatus.LIVE_UNVALIDATED  # live via Braket, unvalidated

    quant = get_backend_status("quantinuum_h2")
    assert quant.live_status == LiveStatus.LIVE_UNVALIDATED  # pytket adapter added in P3


def test_all_statuses_cover_registry():
    from qb_compiler.config import BACKEND_CONFIGS

    statuses = all_backend_statuses()
    assert {s.backend for s in statuses} == set(BACKEND_CONFIGS)
    for s in statuses:
        d = s.as_dict()
        assert d["live_status"] in {"live", "live-unvalidated", "static", "none"}
        assert isinstance(d["live_deps_available"], bool)
