"""Tests for backend recommendation engine."""
from __future__ import annotations

from qiskit import QuantumCircuit

from qb_compiler.recommender import (
    BackendRecommender,
    RecommendationReport,
)


class TestBackendRecommender:
    def test_single_backend(self):
        qc = QuantumCircuit(4, 4, name="GHZ-4")
        qc.h(0)
        for i in range(3):
            qc.cx(i, i + 1)
        qc.measure(range(4), range(4))

        rec = BackendRecommender(n_seeds=2)
        rec.add_backend("ibm_fez")
        report = rec.analyze(qc)

        assert isinstance(report, RecommendationReport)
        assert report.circuit_name == "GHZ-4"
        assert report.n_qubits == 4
        assert len(report.analyses) == 1
        assert report.best_fidelity == "ibm_fez"
        assert report.analyses[0].viable is True
        assert report.analyses[0].estimated_fidelity > 0.5

    def test_report_str(self):
        qc = QuantumCircuit(3, 3, name="Bell-3")
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure(range(3), range(3))

        rec = BackendRecommender(n_seeds=2)
        rec.add_backend("ibm_fez")
        report = rec.analyze(qc)

        text = str(report)
        assert "ibm_fez" in text
        assert "Recommendation" in text
        assert "Bell-3" in text

    def test_recommendation_property(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(range(2), range(2))

        rec = BackendRecommender(n_seeds=1)
        rec.add_backend("ibm_fez")
        report = rec.analyze(qc)

        assert "ibm_fez" in report.recommendation
        assert "fidelity" in report.recommendation.lower()

    def test_analysis_has_cost(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(range(2), range(2))

        rec = BackendRecommender(n_seeds=1)
        rec.add_backend("ibm_fez")
        report = rec.analyze(qc)

        a = report.analyses[0]
        assert a.cost_per_4096_shots is not None
        assert a.cost_per_4096_shots > 0
        assert a.fidelity_per_dollar is not None
        assert a.fidelity_per_dollar > 0

    def test_empty_recommender(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(range(2), range(2))

        rec = BackendRecommender()
        report = rec.analyze(qc)

        assert len(report.analyses) == 0
        assert report.best_fidelity is None
        assert "No backends" in report.recommendation

    def test_chaining(self):
        rec = BackendRecommender()
        result = rec.add_backend("ibm_fez")
        assert result is rec  # returns self for chaining

    def test_analysis_has_physical_qubits(self):
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure(range(3), range(3))

        rec = BackendRecommender(n_seeds=1)
        rec.add_backend("ibm_fez")
        report = rec.analyze(qc)

        a = report.analyses[0]
        assert len(a.physical_qubits) == 3
        assert all(isinstance(q, int) for q in a.physical_qubits)

    def test_analysis_time_tracked(self):
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(range(2), range(2))

        rec = BackendRecommender(n_seeds=1)
        rec.add_backend("ibm_fez")
        report = rec.analyze(qc)

        assert report.total_analysis_time_ms > 0
        assert report.analyses[0].analysis_time_ms > 0
