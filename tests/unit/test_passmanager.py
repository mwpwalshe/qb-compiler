"""Tests for the public ``qb_compiler.passmanager`` Qiskit-integration factory.

This convenience factory is exported in ``qb_compiler.__all__`` but was otherwise uncovered. It
requires Qiskit and is skipped if Qiskit is unavailable.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qiskit")

from qb_compiler import passmanager


def _is_passmanager(obj: object) -> bool:
    from qiskit.transpiler import PassManager

    return isinstance(obj, PassManager)


class TestPassmanager:
    def test_backend_name_string(self) -> None:
        pm = passmanager("ibm_fez")
        assert _is_passmanager(pm)

    def test_optimization_level_option(self) -> None:
        pm = passmanager("ibm_fez", optimization_level=1)
        assert _is_passmanager(pm)

    def test_unknown_backend_name_still_builds(self) -> None:
        # Unknown name -> no basis_gates, but a usable PassManager is still returned.
        pm = passmanager("not_a_real_backend")
        assert _is_passmanager(pm)

    def test_no_backend_fallback(self) -> None:
        pm = passmanager(None)
        assert _is_passmanager(pm)

    def test_runs_on_a_circuit(self) -> None:
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        pm = passmanager("ibm_fez", optimization_level=1)
        out = pm.run(qc)
        assert out.num_qubits >= 2
