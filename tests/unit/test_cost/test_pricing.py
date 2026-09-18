# SPDX-License-Identifier: Apache-2.0
"""The price table, and the one thing that keeps it honest.

The per-shot numbers used to live in two places, `cost/pricing.py` and `config.py`, copied by
hand. They drifted: the IonQ rows in `config.py` carried 0.30, which is Amazon Braket's flat
per-task fee, not a per-shot price, so every IonQ cost estimate came out ten times high on Aria
and nearly four times high on Forte. `config.py` now reads the number from the table, and the
first test here fails if anyone ever copies one back.
"""

from __future__ import annotations

import datetime

import pytest

from qb_compiler.config import BACKEND_CONFIGS
from qb_compiler.cost.pricing import (
    PRICING_AS_OF,
    VENDOR_PRICING,
    VendorPricing,
    cost_per_shot,
    get_pricing,
)


class TestSingleSource:
    def test_every_backend_spec_matches_the_price_table(self):
        for name, spec in BACKEND_CONFIGS.items():
            assert name in VENDOR_PRICING, f"{name} has a spec but no price"
            assert spec.cost_per_shot == VENDOR_PRICING[name].cost_per_shot_usd, name

    def test_a_price_row_may_exist_without_a_spec(self):
        """Braket lists devices this package has no hardware spec for."""
        priced_only = set(VENDOR_PRICING) - set(BACKEND_CONFIGS)
        assert priced_only == {"rigetti_cepheus"}


class TestTable:
    def test_the_review_date_is_a_date(self):
        assert datetime.date.fromisoformat(PRICING_AS_OF) <= datetime.date.today()

    def test_every_entry_is_well_formed(self):
        for name, entry in VENDOR_PRICING.items():
            assert isinstance(entry, VendorPricing)
            assert entry.backend == name
            assert entry.provider
            assert entry.cost_per_shot_usd > 0, name
            assert entry.cost_per_task_usd >= 0, name
            assert entry.currency == "USD"
            assert entry.notes, f"{name} has no provenance note"

    @pytest.mark.parametrize(
        ("backend", "per_shot"),
        [
            ("ionq_forte", 0.08),
            ("iqm_garnet", 0.00145),
            ("iqm_emerald", 0.00160),
            ("rigetti_cepheus", 0.000425),
        ],
    )
    def test_braket_published_prices(self, backend, per_shot):
        """The four rows the Amazon Braket pricing page still lists, as published 2026-09-17."""
        entry = VENDOR_PRICING[backend]
        assert entry.cost_per_shot_usd == pytest.approx(per_shot)
        assert entry.cost_per_task_usd == pytest.approx(0.30)

    def test_ankaa_is_the_price_the_aws_price_list_reports(self):
        """0.00035 was the Aspen generation price, carried onto Ankaa-3 when it was renamed.

        The AWS Price List API, which is the source AWS bills from, reports 0.0009 for Ankaa-3 and
        0.00035 for Aspen-10, Aspen-11 and the Aspen-M devices. Read 2026-09-18.
        """
        entry = VENDOR_PRICING["rigetti_ankaa"]
        assert entry.cost_per_shot_usd == pytest.approx(0.0009)
        assert entry.cost_per_task_usd == pytest.approx(0.30)
        assert "Aspen generation price" in entry.notes
        assert "read 2026-09-18" in entry.notes

    @pytest.mark.parametrize("backend", ["ionq_aria", "rigetti_ankaa"])
    def test_rows_off_the_pricing_page_say_exactly_that(self, backend):
        """Off the pricing page is not off Braket, and the notes must not say it is."""
        notes = VENDOR_PRICING[backend].notes
        assert "Not on the Amazon Braket pricing page as of 2026-09-17" in notes
        assert "not listed on amazon braket" not in notes.lower()

    def test_aria_says_the_price_list_still_carries_it(self):
        notes = VENDOR_PRICING["ionq_aria"].notes
        assert "AWS Price List API still carries it at 0.03 per shot" in notes
        assert "read 2026-09-18" in notes

    def test_the_ibm_rows_say_they_are_a_conversion(self):
        for backend in ("ibm_fez", "ibm_torino", "ibm_marrakesh"):
            notes = VENDOR_PRICING[backend].notes
            assert "not an IBM price" in notes
            assert "shots per second" in notes

    def test_the_quantinuum_row_states_its_circuit(self):
        notes = VENDOR_PRICING["quantinuum_h2"].notes
        assert "not a Quantinuum price" in notes
        assert "HQC" in notes
        assert "12.50" in notes

    def test_ibm_conversions_are_consistent_with_the_published_rate(self):
        """1.60 USD per second, divided by the assumed throughput."""
        assert VENDOR_PRICING["ibm_fez"].cost_per_shot_usd == pytest.approx(1.60 / 10_000)
        assert VENDOR_PRICING["ibm_torino"].cost_per_shot_usd == pytest.approx(
            1.60 / 11_400, rel=0.02
        )


class TestLookup:
    def test_a_known_backend_resolves(self):
        assert get_pricing("iqm_garnet").cost_per_shot_usd == pytest.approx(0.00145)
        assert cost_per_shot("iqm_garnet") == pytest.approx(0.00145)

    def test_an_unknown_backend_returns_none_or_raises(self):
        assert get_pricing("no_such_backend") is None
        with pytest.raises(KeyError, match="No pricing data"):
            cost_per_shot("no_such_backend")

    def test_job_cost_adds_the_task_fee(self):
        entry = VENDOR_PRICING["ionq_forte"]
        assert entry.job_cost(shots=1000, tasks=1) == pytest.approx(0.08 * 1000 + 0.30)
        assert entry.job_cost(shots=1000, tasks=3) == pytest.approx(0.08 * 1000 + 0.90)

    def test_a_task_fee_free_vendor_charges_shots_only(self):
        assert VENDOR_PRICING["ibm_fez"].job_cost(shots=1000) == pytest.approx(0.16)
