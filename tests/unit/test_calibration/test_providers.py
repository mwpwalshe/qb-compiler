"""Tests for calibration providers."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.calibration.models.coupling_properties import GateProperties
from qb_compiler.calibration.models.qubit_properties import QubitProperties
from qb_compiler.calibration.static_provider import StaticCalibrationProvider
from qb_compiler.calibration.cached_provider import CachedCalibrationProvider


class TestStaticProvider:
    """Tests for StaticCalibrationProvider."""

    def test_static_provider_qubit_lookup(self, mock_calibration) -> None:
        """Looking up a qubit should return its calibration data."""
        qp = mock_calibration.get_qubit_properties(0)

        assert qp is not None
        assert qp.qubit_id == 0
        assert qp.t1_us is not None
        assert qp.t1_us > 0
        assert qp.t2_us is not None
        assert qp.t2_us > 0
        assert qp.readout_error is not None
        assert 0 < qp.readout_error < 1

    def test_static_provider_qubit_not_found(self, mock_calibration) -> None:
        """Looking up a non-existent qubit should return None."""
        qp = mock_calibration.get_qubit_properties(999)
        assert qp is None

    def test_static_provider_gate_lookup(self, mock_calibration) -> None:
        """Looking up a gate should return its calibration data."""
        gp = mock_calibration.get_gate_properties("cx", (0, 1))

        assert gp is not None
        assert gp.gate_type == "cx"
        assert gp.qubits == (0, 1)
        assert gp.error_rate is not None
        assert gp.error_rate > 0

    def test_static_provider_from_json(self, ibm_fez_calibration_path: Path) -> None:
        """Loading from a JSON fixture should produce a valid provider."""
        provider = StaticCalibrationProvider.from_json(ibm_fez_calibration_path)

        assert provider.backend_name == "ibm_fez"
        # The fixture has 10 qubits
        all_qubits = provider.get_all_qubit_properties()
        assert len(all_qubits) == 10

        # Spot-check qubit 0
        q0 = provider.get_qubit_properties(0)
        assert q0 is not None
        assert q0.t1_us == pytest.approx(156.32, rel=0.01)

        # Spot-check a gate
        gp = provider.get_gate_properties("cx", (0, 1))
        assert gp is not None
        assert gp.error_rate == pytest.approx(0.0042, rel=0.01)

    def test_calibration_age(self, mock_calibration) -> None:
        """age_hours should be positive for a historical timestamp."""
        # The mock's timestamp is 2026-02-15, which is in the past
        assert mock_calibration.age_hours > 0

    def test_static_provider_all_gates(self, mock_calibration) -> None:
        """get_all_gate_properties should return all stored gates."""
        all_gates = mock_calibration.get_all_gate_properties()
        assert len(all_gates) == 9  # 9 CX gates in the fixture


class TestCachedProvider:
    """Tests for CachedCalibrationProvider."""

    def test_cached_provider_caches(self, mock_calibration) -> None:
        """The cached provider should call the factory only once within max_age."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return mock_calibration

        cached = CachedCalibrationProvider(factory, max_age_seconds=60.0)

        # First access triggers factory
        q0 = cached.get_qubit_properties(0)
        assert q0 is not None
        assert call_count == 1

        # Second access should use cache
        q1 = cached.get_qubit_properties(1)
        assert q1 is not None
        assert call_count == 1  # still 1

    def test_cached_provider_refreshes_on_expiry(self, mock_calibration) -> None:
        """After invalidation, the factory should be called again."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return mock_calibration

        cached = CachedCalibrationProvider(factory, max_age_seconds=0.01)

        cached.get_qubit_properties(0)
        assert call_count == 1

        # Wait for cache to expire
        time.sleep(0.02)

        cached.get_qubit_properties(0)
        assert call_count == 2

    def test_cached_provider_invalidate(self, mock_calibration) -> None:
        """Manual invalidation should force a refresh on next access."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return mock_calibration

        cached = CachedCalibrationProvider(factory, max_age_seconds=3600.0)

        cached.get_qubit_properties(0)
        assert call_count == 1

        cached.invalidate()
        cached.get_qubit_properties(0)
        assert call_count == 2
