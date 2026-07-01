"""P2: prove the registry routes IonQ/Rigetti through the Braket provider end-to-end, offline.

Uses ``BraketCalibrationProvider.from_device_properties`` (no AWS) as a replay of a real device
fetch, so we exercise registry -> Braket -> snapshot wiring without credentials. Flipping the
declared status from ``live-unvalidated`` to ``live`` still requires a real-device smoke run.
"""

from __future__ import annotations

from qb_compiler.calibration import registry
from qb_compiler.calibration.braket_provider import BraketCalibrationProvider

# Realistic IonQ-on-Braket device .properties shape (device averages).
_IONQ_PROPS = {
    "paradigm": {"qubitCount": 25, "nativeGateSet": ["GPI", "GPI2", "MS"]},
    "provider": {
        "fidelity": {"1Q": {"mean": 0.9998}, "2Q": {"mean": 0.991}, "spam": {"mean": 0.995}},
        "timing": {"T1": 1e-5, "T2": 1e-6},
    },
}


def test_registry_routes_ionq_through_braket(monkeypatch):
    def _replay(backend: str):
        return BraketCalibrationProvider.from_device_properties(
            _IONQ_PROPS, backend=backend, provider="ionq"
        )

    monkeypatch.setitem(
        registry.STRATEGY_MAP,
        "ionq",
        registry.ProviderStrategy((_replay,), registry.LiveStatus.LIVE_UNVALIDATED, lambda: True),
    )
    prov = registry.get_calibration_provider("ionq_aria", prefer_live=True)
    props = prov.backend_properties
    assert props is not None
    assert props.provider == "ionq"
    assert props.n_qubits == 25
    # device-average fidelity -> per-qubit readout error surfaced
    q0 = prov.get_qubit_properties(0)
    assert q0 is not None and q0.readout_error is not None
