"""Tests for backend auto-discovery and PUB-aware preflight."""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit

import qb_compiler.discovery as discovery
from qb_compiler.discovery import (
    DiscoveredBackend,
    check_viability_pub,
    discover_backends,
    rank_discovered,
)

# ── stubs (no qiskit_ibm_runtime import anywhere) ────────────────


class _StubStatus:
    def __init__(self, operational: bool = True, pending_jobs: int = 0) -> None:
        self.operational = operational
        self.pending_jobs = pending_jobs


class _StubTarget:
    operation_names = ("cz", "rz", "sx", "x", "measure")


class _StubBackend:
    def __init__(
        self,
        name: str,
        num_qubits: int = 5,
        operational: bool = True,
        pending_jobs: int = 0,
        target: Any = None,
        status_raises: bool = False,
    ) -> None:
        self.name = name
        self.num_qubits = num_qubits
        self.target = target
        self._status = _StubStatus(operational, pending_jobs)
        self._status_raises = status_raises

    def status(self) -> _StubStatus:
        if self._status_raises:
            raise RuntimeError("status endpoint down")
        return self._status


class _StubService:
    def __init__(self, backend_list: list[_StubBackend]) -> None:
        self._backends = backend_list

    def backends(self) -> list[_StubBackend]:
        return self._backends


def _bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2, name="Bell")
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(range(2), range(2))
    return qc


# ── discover_backends ────────────────────────────────────────────


class TestDiscoverBackends:
    def test_collects_fields(self):
        service = _StubService(
            [
                _StubBackend("stub_a", num_qubits=7, pending_jobs=3, target=_StubTarget()),
                _StubBackend("stub_b", num_qubits=4, operational=False),
            ]
        )
        result = discover_backends(service)

        assert len(result) == 2
        a, b = result
        assert isinstance(a, DiscoveredBackend)
        assert a.name == "stub_a"
        assert a.num_qubits == 7
        assert a.operational is True
        assert a.pending_jobs == 3
        assert a.has_target is True
        assert "cz" in a.basis_gates

        assert b.name == "stub_b"
        assert b.operational is False
        assert b.has_target is False
        assert b.basis_gates == ()

    def test_survives_status_failure(self):
        service = _StubService(
            [
                _StubBackend("flaky", status_raises=True),
                _StubBackend("ok", pending_jobs=1),
            ]
        )
        result = discover_backends(service)

        assert len(result) == 2
        flaky, ok = result
        assert flaky.name == "flaky"
        assert flaky.operational is False
        assert flaky.pending_jobs == 0
        assert ok.operational is True


# ── check_viability_pub ──────────────────────────────────────────


class TestCheckViabilityPub:
    def test_bell_pub_returns_fidelity(self):
        pub = (_bell_circuit(), None, None)
        result = check_viability_pub(pub, backend="ibm_fez", n_seeds=2)
        assert 0.0 < result.estimated_fidelity <= 1.0
        assert result.backend == "ibm_fez"

    def test_explicit_shots_forwarded(self):
        pub = (_bell_circuit(), None, 1024)
        result = check_viability_pub(pub, backend="ibm_fez", n_seeds=2)
        assert 0.0 < result.estimated_fidelity <= 1.0

    def test_empty_pub_raises(self):
        import pytest

        with pytest.raises(ValueError, match="empty"):
            check_viability_pub(())


# ── rank_discovered ──────────────────────────────────────────────


class _FakeResult:
    def __init__(self, estimated_fidelity: float) -> None:
        self.estimated_fidelity = estimated_fidelity


class TestRankDiscovered:
    def test_skips_backend_without_target(self):
        service = _StubService([_StubBackend("no_target", num_qubits=5, target=None)])
        ranked = rank_discovered(_bell_circuit(), service)
        assert ranked == []

    def test_skips_non_operational_and_too_small(self):
        service = _StubService(
            [
                _StubBackend("down", operational=False, target=_StubTarget()),
                _StubBackend("tiny", num_qubits=1, target=_StubTarget()),
            ]
        )
        ranked = rank_discovered(_bell_circuit(), service)
        assert ranked == []

    def test_ranking_order_and_top(self, monkeypatch):
        fidelities = {"low": 0.4, "high": 0.9, "mid": 0.7}

        def fake_check_viability(circuit: Any, **kwargs: Any) -> _FakeResult:
            return _FakeResult(fidelities[kwargs["backend"]])

        monkeypatch.setattr(discovery, "check_viability", fake_check_viability)

        service = _StubService(
            [
                _StubBackend(name, num_qubits=5, target=_StubTarget())
                for name in ("low", "high", "mid")
            ]
        )

        ranked = rank_discovered(_bell_circuit(), service)
        assert [db.name for db, _ in ranked] == ["high", "mid", "low"]
        assert [r.estimated_fidelity for _, r in ranked] == [0.9, 0.7, 0.4]

        top1 = rank_discovered(_bell_circuit(), service, top=1)
        assert [db.name for db, _ in top1] == ["high"]

    def test_check_viability_failure_skips_backend(self, monkeypatch):
        def fake_check_viability(circuit: Any, **kwargs: Any) -> _FakeResult:
            if kwargs["backend"] == "broken":
                raise RuntimeError("transpile exploded")
            return _FakeResult(0.5)

        monkeypatch.setattr(discovery, "check_viability", fake_check_viability)

        service = _StubService(
            [
                _StubBackend("broken", num_qubits=5, target=_StubTarget()),
                _StubBackend("fine", num_qubits=5, target=_StubTarget()),
            ]
        )

        ranked = rank_discovered(_bell_circuit(), service)
        assert [db.name for db, _ in ranked] == ["fine"]
