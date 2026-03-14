"""Tests for QubitBoost SDK integration layer."""
from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from qb_compiler.integrations.qubitboost import (
    GATE_REGISTRY,
    Confidence,
    detect_circuit_type,
    is_sdk_available,
    recommend_gates,
)

# ── Circuit type detection ──────────────────────────────────────────


class TestDetectCircuitType:
    def test_qaoa_by_name(self) -> None:
        qc = QuantumCircuit(4, 4, name="QAOA-MaxCut")
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(range(4), range(4))
        ct, conf = detect_circuit_type(qc)
        assert ct == "qaoa"
        assert conf == Confidence.HIGH

    def test_vqe_by_name(self) -> None:
        qc = QuantumCircuit(2, 2, name="VQE-H2")
        qc.ry(0.5, 0)
        qc.cx(0, 1)
        qc.measure(range(2), range(2))
        ct, conf = detect_circuit_type(qc)
        assert ct == "vqe"
        assert conf == Confidence.HIGH

    def test_qec_by_name(self) -> None:
        qc = QuantumCircuit(5, 5, name="surface-code-d3")
        qc.h(0)
        qc.measure(range(5), range(5))
        ct, conf = detect_circuit_type(qc)
        assert ct == "qec"
        assert conf == Confidence.HIGH

    def test_qaoa_by_structure(self) -> None:
        """Circuit with Rz + Rx + CX pattern → QAOA at MEDIUM confidence."""
        qc = QuantumCircuit(4, 4, name="test")
        for _ in range(2):
            for i in range(3):
                qc.cx(i, i + 1)
                qc.rz(0.5, i + 1)
                qc.cx(i, i + 1)
            for i in range(4):
                qc.rx(0.3, i)
        qc.measure(range(4), range(4))
        ct, conf = detect_circuit_type(qc)
        assert ct == "qaoa"
        assert conf == Confidence.MEDIUM

    def test_vqe_by_structure(self) -> None:
        """Circuit dominated by parameterised rotations → VQE at MEDIUM."""
        qc = QuantumCircuit(4, 4, name="test")
        for i in range(4):
            qc.ry(0.5, i)
            qc.rz(0.3, i)
        qc.cx(0, 1)
        qc.measure(range(4), range(4))
        ct, conf = detect_circuit_type(qc)
        assert ct == "vqe"
        assert conf == Confidence.MEDIUM

    def test_general_circuit(self) -> None:
        """Bell state → general at LOW confidence."""
        qc = QuantumCircuit(2, 2, name="Bell")
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(range(2), range(2))
        ct, conf = detect_circuit_type(qc)
        assert ct == "general"
        assert conf == Confidence.LOW

    def test_empty_circuit(self) -> None:
        qc = QuantumCircuit(2)
        ct, conf = detect_circuit_type(qc)
        assert ct == "general"
        assert conf == Confidence.LOW


# ── Gate recommendations ────────────────────────────────────────────


class TestRecommendGates:
    def test_qaoa_high_confidence(self) -> None:
        recs = recommend_gates("qaoa", Confidence.HIGH)
        gate_names = [r.gate for r in recs]
        assert "OptGate" in gate_names
        assert "GuardGate" in gate_names
        assert "LiveGate" in gate_names
        assert "ShotValidator" in gate_names
        assert "TomoGate" in gate_names

        opt = next(r for r in recs if r.gate == "OptGate")
        assert opt.status == "Eligible"
        assert opt.validated_claim is not None
        assert "117" in opt.validated_claim

    def test_qaoa_medium_confidence(self) -> None:
        recs = recommend_gates("qaoa", Confidence.MEDIUM)
        gate_names = [r.gate for r in recs]
        assert "OptGate" in gate_names

        opt = next(r for r in recs if r.gate == "OptGate")
        assert opt.status == "May be eligible"

    def test_qaoa_low_confidence_hides_specialised(self) -> None:
        """At LOW confidence, specialised gates like OptGate are hidden."""
        recs = recommend_gates("qaoa", Confidence.LOW)
        gate_names = [r.gate for r in recs]
        assert "OptGate" not in gate_names
        assert "GuardGate" not in gate_names
        # Universal gates still present
        assert "TomoGate" in gate_names
        assert "LiveGate" in gate_names
        assert "ShotValidator" in gate_names

    def test_vqe_high_confidence(self) -> None:
        recs = recommend_gates("vqe", Confidence.HIGH)
        gate_names = [r.gate for r in recs]
        assert "ChemGate" in gate_names
        chem = next(r for r in recs if r.gate == "ChemGate")
        assert chem.status == "Eligible"
        assert "32-42%" in chem.validated_claim

    def test_qec_gates(self) -> None:
        recs = recommend_gates("qec", Confidence.HIGH)
        gate_names = [r.gate for r in recs]
        assert "SafetyGate" in gate_names

    def test_general_only_universal(self) -> None:
        recs = recommend_gates("general", Confidence.LOW)
        gate_names = [r.gate for r in recs]
        assert "TomoGate" in gate_names
        assert "LiveGate" in gate_names
        assert "ShotValidator" in gate_names
        assert "OptGate" not in gate_names
        assert "ChemGate" not in gate_names

    def test_phase_ordering(self) -> None:
        """Recommendations are ordered pre → during → post."""
        recs = recommend_gates("qaoa", Confidence.HIGH)
        phases = [r.phase for r in recs]
        assert phases == sorted(phases, key=lambda p: {"pre": 0, "during": 1, "post": 2}[p])

    def test_recommendation_str(self) -> None:
        recs = recommend_gates("qaoa", Confidence.HIGH)
        for r in recs:
            s = str(r)
            assert r.gate in s
            assert r.headline in s


# ── Gate registry ───────────────────────────────────────────────────


class TestGateRegistry:
    def test_all_seven_gates_present(self) -> None:
        expected = {
            "OptGate", "ChemGate", "TomoGate", "LiveGate",
            "SafetyGate", "GuardGate", "ShotValidator",
        }
        assert set(GATE_REGISTRY.keys()) == expected

    def test_all_gates_have_qualifiers(self) -> None:
        for name, info in GATE_REGISTRY.items():
            assert info.qualifier, f"{name} missing qualifier"
            assert info.validated_claim, f"{name} missing validated_claim"

    def test_all_gates_have_phase(self) -> None:
        for name, info in GATE_REGISTRY.items():
            assert info.phase in ("pre", "during", "post"), f"{name} bad phase"


# ── SDK availability ────────────────────────────────────────────────


class TestSDKAvailability:
    def test_is_sdk_available_returns_bool(self) -> None:
        result = is_sdk_available()
        assert isinstance(result, bool)

    def test_executor_raises_without_sdk(self) -> None:
        """QubitBoostExecutor raises ImportError when SDK not installed."""
        from qb_compiler.integrations.qubitboost import QubitBoostExecutor

        if not is_sdk_available():
            with pytest.raises(ImportError, match="qubitboost"):
                QubitBoostExecutor(backend="ibm_fez")
