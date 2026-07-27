# SPDX-License-Identifier: Apache-2.0
"""Regression: _provider_to_dict must preserve the coupling map.

A live IBM provider holds its BackendProperties under ``._delegate`` (it wraps a
StaticCalibrationProvider). If _provider_to_dict fails to find it, it falls through to the
abstract-interface path; historically that path dropped the coupling map entirely, so
QBCalibrationLayout had no topology and silently degraded to the default layout -- disabling
calibration-aware placement on every live backend. These tests pin both the delegate-drill and
the fallback topology reconstruction.
"""

from __future__ import annotations

from qb_compiler.calibration.models.backend_properties import BackendProperties
from qb_compiler.calibration.models.coupling_properties import GateProperties
from qb_compiler.calibration.models.qubit_properties import QubitProperties
from qb_compiler.calibration.static_provider import StaticCalibrationProvider
from qb_compiler.qiskit_plugin.transpiler_plugin import _provider_to_dict


def _backend() -> BackendProperties:
    qp = [
        QubitProperties(qubit_id=i, t1_us=100.0 + i, t2_us=80.0 + i, readout_error=0.01 + 0.001 * i)
        for i in range(4)
    ]
    gp = [
        GateProperties(gate_type="cz", qubits=(0, 1), error_rate=0.002, gate_time_ns=68.0),
        GateProperties(gate_type="cz", qubits=(1, 2), error_rate=0.008, gate_time_ns=68.0),
        GateProperties(gate_type="cz", qubits=(2, 3), error_rate=0.015, gate_time_ns=68.0),
    ]
    return BackendProperties(
        backend="test_heron",
        provider="test",
        n_qubits=4,
        basis_gates=("cz", "rz", "sx", "x", "id"),
        coupling_map=[(0, 1), (1, 2), (2, 3)],
        qubit_properties=qp,
        gate_properties=gp,
        timestamp="2026-07-26T00:00:00",
    )


class _FakeLiveProvider:
    """Mimics IBMRuntimeCalibrationProvider: BackendProperties reachable only via _delegate."""

    def __init__(self, props):
        self._delegate = StaticCalibrationProvider(props)


class _FallbackProvider:
    """No _props / _snapshot / _delegate -- only the abstract getter interface."""

    def __init__(self, props):
        self._props = None  # force the fallback path
        self.backend_name = props.backend
        self.timestamp = props.timestamp
        self._q = props.qubit_properties
        self._g = props.gate_properties

    def get_all_qubit_properties(self):
        return self._q

    def get_all_gate_properties(self):
        return self._g


def test_delegate_drill_preserves_coupling_map():
    cal = _provider_to_dict(_FakeLiveProvider(_backend()), "test_heron")
    assert cal.get("coupling_map"), "coupling_map lost -> calibration-aware layout silently dies"
    assert sorted(tuple(e) for e in cal["coupling_map"]) == [(0, 1), (1, 2), (2, 3)]
    assert len(cal["qubit_properties"]) == 4


def test_fallback_reconstructs_coupling_from_2q_gates():
    # getattr(provider, "_props", None) is None -> fallback path taken
    cal = _provider_to_dict(_FallbackProvider(_backend()), "test_heron")
    assert cal.get("coupling_map"), "fallback must reconstruct topology from 2Q gate pairs"
    assert sorted(tuple(e) for e in cal["coupling_map"]) == [(0, 1), (1, 2), (2, 3)]


def test_coupling_feeds_a_nontrivial_layout():
    # end-to-end: with topology present, QBCalibrationLayout picks calibration-good qubits,
    # not the trivial 0..n-1 identity.
    from qiskit import QuantumCircuit
    from qiskit.transpiler import PassManager

    from qb_compiler.qiskit_plugin import QBCalibrationLayout

    cal = _provider_to_dict(_FakeLiveProvider(_backend()), "test_heron")
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    p = QBCalibrationLayout(cal)
    PassManager([p]).run(qc)
    layout = p.property_set.get("layout")
    assert layout is not None
    chosen = sorted(layout[q] for q in qc.qubits)
    # qubits 0,1 are the lowest-readout / lowest-error pair -> should be selected
    assert chosen == [0, 1]
