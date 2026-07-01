"""Offline tests for the Quantinuum provider, using a real hand-built pytket ``BackendInfo``.

No Quantinuum account: we build a ``BackendInfo`` in-process and exercise the pure parser + the
offline ``from_backend_info`` path.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pytket")

from qb_compiler.calibration.quantinuum_provider import (
    QuantinuumCalibrationProvider,
    parse_quantinuum_info,
)


def _build_info():
    from pytket.architecture import Architecture
    from pytket.backends.backendinfo import BackendInfo
    from pytket.circuit import Node, OpType

    n = [Node(0), Node(1), Node(2)]
    arch = Architecture([(n[0], n[1]), (n[1], n[2])])
    return BackendInfo(
        name="QuantinuumBackend",
        device_name="H2-test",
        version="1.0",
        architecture=arch,
        gate_set={OpType.Rz, OpType.PhasedX, OpType.ZZPhase, OpType.Measure},
        all_node_gate_errors={
            n[0]: {OpType.PhasedX: 1e-4},
            n[1]: {OpType.PhasedX: 2e-4},
            n[2]: {OpType.PhasedX: 1.5e-4},
        },
        all_edge_gate_errors={
            (n[0], n[1]): {OpType.ZZPhase: 2e-3},
            (n[1], n[2]): {OpType.ZZPhase: 3e-3},
        },
        all_readout_errors={
            n[0]: [[0.99, 0.01], [0.02, 0.98]],
            n[1]: [[0.98, 0.02], [0.02, 0.98]],
            n[2]: [[0.97, 0.03], [0.03, 0.97]],
        },
    )


def test_parse_structure_and_errors():
    props = parse_quantinuum_info(_build_info(), backend="quantinuum_h2")
    assert props.n_qubits == 3
    assert props.provider == "quantinuum"
    assert "zzphase" in props.basis_gates or "zz" in props.basis_gates
    assert sorted(props.coupling_map) == [(0, 1), (1, 2)]
    # T1/T2 not reported by Quantinuum -> None (honest)
    assert all(q.t1_us is None and q.t2_us is None for q in props.qubit_properties)
    # readout error for q0 = 1 - mean(0.99, 0.98) = 0.015
    q0 = props.qubit_properties[0]
    assert q0.readout_error == pytest.approx(0.015, abs=1e-6)


def test_gate_errors_mapped():
    props = parse_quantinuum_info(_build_info(), backend="quantinuum_h2")
    g2 = [g for g in props.gate_properties if g.gate_type == "2q"]
    by_edge = {g.qubits: g.error_rate for g in g2}
    assert by_edge[(0, 1)] == pytest.approx(2e-3)
    assert by_edge[(1, 2)] == pytest.approx(3e-3)
    g1 = {g.qubits: g.error_rate for g in props.gate_properties if g.gate_type == "1q"}
    assert g1[(1,)] == pytest.approx(2e-4)


def test_from_backend_info_lookups():
    prov = QuantinuumCalibrationProvider.from_backend_info(_build_info(), backend="quantinuum_h2")
    assert prov.backend_name == "quantinuum_h2"
    assert prov.backend_properties.n_qubits == 3
    g = prov.get_gate_properties("2q", (0, 1))
    assert g is not None and g.error_rate == pytest.approx(2e-3)
