# SPDX-License-Identifier: Apache-2.0
"""The four billing models, on worked examples, with the assumptions asserted.

Three of the four vendors do not sell shots, so the number a model returns is only as good as
what it assumed. Every test here checks the assumption dict as well as the figure, because a
number without its assumptions is what a wrong invoice is made of.
"""

from __future__ import annotations

import pytest

from qb_compiler.cost.billing import (
    CostBreakdown,
    PerGateShot,
    PerHQC,
    PerSecond,
    PerShot,
    billing_as_dict,
    billing_from_dict,
)


class TestPerShot:
    def test_shots_and_the_task_fee(self):
        breakdown = PerShot(0.08, 0.30).job_cost(1000, tasks=2)
        assert breakdown.usd == pytest.approx(0.08 * 1000 + 0.60)
        assert breakdown.model == "per_shot"
        assert breakdown.assumptions["cost_per_shot_usd"] == 0.08
        assert breakdown.assumptions["cost_per_task_usd"] == 0.30
        assert breakdown.assumptions["tasks"] == 2

    def test_it_is_a_breakdown_and_not_a_float(self):
        assert isinstance(PerShot(0.001).job_cost(10), CostBreakdown)

    def test_provenance_travels_with_the_number(self):
        breakdown = PerShot(0.001).job_cost(10, as_of="2026-09-17", source="a page", status="live")
        assert breakdown.as_of == "2026-09-17"
        assert breakdown.source == "a page"
        assert breakdown.status == "live"
        assert "live" in str(breakdown)


class TestPerSecond:
    def test_the_shot_rate_is_named_as_an_assumption(self):
        breakdown = PerSecond(1.60, 10_000).job_cost(4096)
        assert breakdown.usd == pytest.approx(1.60 * 4096 / 10_000)
        assert breakdown.assumptions["usd_per_second"] == 1.60
        assert breakdown.assumptions["assumed_shots_per_second"] == 10_000
        assert breakdown.assumptions["implied_seconds"] == pytest.approx(0.4096)
        assert "not a vendor figure" in breakdown.assumptions["basis"]

    def test_from_per_shot_keeps_that_number_exact(self):
        """The shipped conversion is the authority; the throughput is derived from it."""
        for per_shot in (0.00016, 0.00014):
            model = PerSecond.from_per_shot(1.60, per_shot)
            assert model.cost_per_shot_usd == per_shot
            assert model.job_cost(10_000).usd == pytest.approx(per_shot * 10_000)

    def test_a_slower_machine_costs_more(self):
        fast = PerSecond(1.60, 10_000).job_cost(10_000).usd
        slow = PerSecond(1.60, 1_000).job_cost(10_000).usd
        assert slow == pytest.approx(10 * fast)


class TestPerGateShot:
    def test_gate_counts_price_the_job(self):
        model = PerGateShot(0.000220, 0.000975, minimum_per_job_usd=12.4166)
        breakdown = model.job_cost(1000, one_qubit_gates=100, two_qubit_gates=40)
        assert breakdown.usd == pytest.approx(1000 * (100 * 0.000220 + 40 * 0.000975))
        assert breakdown.assumptions["minimum_applied"] is False
        assert breakdown.model == "per_gate_shot"

    def test_the_per_program_minimum_applies(self):
        model = PerGateShot(0.000220, 0.000975, minimum_per_job_usd=12.4166)
        breakdown = model.job_cost(10, one_qubit_gates=1, two_qubit_gates=1)
        assert breakdown.usd == pytest.approx(12.4166)
        assert breakdown.assumptions["minimum_applied"] is True
        assert breakdown.assumptions["priced_before_minimum_usd"] < 12.4166

    def test_without_counts_it_falls_back_and_says_so(self):
        model = PerGateShot(0.000220, 0.000975, 12.4166, fallback=PerShot(0.03, 0.30))
        breakdown = model.job_cost(100)
        assert breakdown.usd == pytest.approx(0.03 * 100 + 0.30)
        assert breakdown.model == "per_shot"
        assert breakdown.assumptions["fell_back_from"] == "per_gate_shot"
        assert "no gate counts" in breakdown.assumptions["fell_back_because"]

    def test_without_counts_and_without_a_fallback_it_refuses(self):
        with pytest.raises(ValueError, match="needs one_qubit_gates"):
            PerGateShot(0.000220, 0.000975).job_cost(100)


class TestPerHQC:
    def test_the_published_formula(self):
        """HQC = 5 + C(N1q + 10 N2q + 5 Nm)/5000, at 12.50 USD per HQC."""
        model = PerHQC(12.50)
        breakdown = model.job_cost(100, one_qubit_gates=120, two_qubit_gates=280, measurements=56)
        assert breakdown.assumptions["hqc"] == pytest.approx(5 + 100 * 3200 / 5000)
        assert breakdown.usd == pytest.approx(69.0 * 12.50)
        assert "N1q" in breakdown.assumptions["formula"]

    def test_the_fallback_is_the_same_circuit_in_the_large_shot_limit(self):
        """8.00 per shot is what 12.50 per HQC comes to on the circuit the notes name."""
        model = PerHQC(12.50, fallback=PerShot(8.00))
        counted = model.job_cost(100_000, one_qubit_gates=120, two_qubit_gates=280, measurements=56)
        approximated = model.job_cost(100_000)
        assert counted.usd == pytest.approx(approximated.usd, rel=1e-4)
        assert approximated.assumptions["fell_back_from"] == "per_hqc"

    def test_without_counts_and_without_a_fallback_it_refuses(self):
        with pytest.raises(ValueError, match="needs one_qubit_gates"):
            PerHQC(12.50).job_cost(100)


class TestSerialisation:
    @pytest.mark.parametrize(
        "model",
        [
            PerShot(0.08, 0.30),
            PerSecond(1.60, 10_000),
            PerGateShot(0.000220, 0.000975, 12.4166, fallback=PerShot(0.03, 0.30)),
            PerHQC(12.50, fallback=PerShot(8.00)),
        ],
    )
    def test_a_model_round_trips(self, model):
        restored = billing_from_dict(billing_as_dict(model))
        assert restored == model
        assert restored.job_cost(
            1000, one_qubit_gates=10, two_qubit_gates=5, measurements=4
        ).usd == pytest.approx(
            model.job_cost(1000, one_qubit_gates=10, two_qubit_gates=5, measurements=4).usd
        )

    def test_an_unknown_model_is_refused(self):
        with pytest.raises(ValueError, match="unknown billing model"):
            billing_from_dict({"model": "per_furlong", "cost_per_shot_usd": 1.0})
