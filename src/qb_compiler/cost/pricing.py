"""Vendor pricing data for quantum backends.

Prices are in USD per shot, and three of the five vendors below do not sell shots.

* **AWS Braket** publishes a per-shot price and a flat 0.30 USD per task for every QPU. Those
  numbers are quoted here as published, and are the only per-shot prices in this table that a
  vendor actually charges.
* **IBM** publishes time, not shots: Pay-As-You-Go starts at 96 USD per minute billed per second,
  which is 1.60 USD per second. The per-shot numbers here are a **throughput conversion**, and the
  assumed throughput is stated in each entry's notes. They are not an IBM price.
* **Azure** bills IonQ per gate-shot, Quantinuum per HQC, and Rigetti per 10 ms of execution. A
  flat per-shot number for any of those is a model with a stated circuit behind it, or it is
  nothing. Where one is quoted, the circuit is named in the notes.

A cost estimate from this table is indicative. For anything that decides a spend, read the vendor
page. :func:`get_pricing` and :func:`cost_per_shot` warn once the table is more than
``_PRICING_STALE_DAYS`` old.

Last checked against the vendor pages on 2026-09-17: the AWS Braket pricing page, the IBM Quantum
pricing page, and the Azure Quantum pricing documentation (itself last updated 2026-04-23).

The Braket rows were then checked against the AWS Price List API on 2026-09-18, which is the same
source AWS bills from. That read corrected ``rigetti_ankaa`` from 0.00035 to 0.0009 and confirmed
the other four. ``PRICING_AS_OF`` stays at the date of the full page review; the rows that were
re-read carry their own date in their notes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VendorPricing:
    """Pricing entry for a single backend.

    Parameters
    ----------
    backend:
        Backend identifier (e.g. ``"ibm_fez"``).
    provider:
        Vendor name.
    cost_per_shot_usd:
        Cost in US dollars per single shot.
    currency:
        Always ``"USD"`` for now.
    notes:
        Free-text provenance / pricing tier notes.
    """

    backend: str
    provider: str
    cost_per_shot_usd: float
    currency: str = "USD"
    notes: str = ""
    cost_per_task_usd: float = 0.0

    def job_cost(self, shots: int, tasks: int = 1) -> float:
        """Total cost of a job: per-shot volume plus per-task fees (Braket-style)."""
        return self.cost_per_shot_usd * shots + self.cost_per_task_usd * tasks


# ── Master pricing table ────────────────────────────────────────────
# This table is the single source of the per-shot numbers. config.py reads
# BackendSpec.cost_per_shot from here, and a unit test asserts the two agree.
#
# On Amazon Braket and absent from this table, because config.py carries no
# BackendSpec for them: AQT Ibex-Q1 (0.0235 per shot) and QuEra Aquila (0.01 per
# shot), both plus the same 0.30 per task. Add the spec first, then the row.

# Last manual review of the price table. Update this whenever the numbers are re-checked
# against vendor pricing pages; get_pricing() warns when the table has gone stale.
PRICING_AS_OF = "2026-09-17"
_PRICING_STALE_DAYS = 90
_stale_warned = False


def _warn_if_stale() -> None:
    global _stale_warned
    if _stale_warned:
        return
    import datetime
    import warnings

    age = (datetime.date.today() - datetime.date.fromisoformat(PRICING_AS_OF)).days
    if age > _PRICING_STALE_DAYS:
        warnings.warn(
            f"qb-compiler price table last reviewed {PRICING_AS_OF} ({age} days ago); "
            "treat cost estimates as indicative and check vendor pricing pages.",
            stacklevel=3,
        )
        _stale_warned = True


VENDOR_PRICING: dict[str, VendorPricing] = {
    # IBM Heron (Utility tier, ~$1.60/sec)
    "ibm_fez": VendorPricing(
        backend="ibm_fez",
        provider="ibm",
        cost_per_shot_usd=0.00016,
        notes=(
            "Heron r2, 156q. Conversion, not an IBM price: IBM bills 1.60 USD per second "
            "(96 per minute, Pay-As-You-Go), and 0.00016 per shot assumes a throughput of "
            "10,000 shots per second."
        ),
    ),
    "ibm_torino": VendorPricing(
        backend="ibm_torino",
        provider="ibm",
        cost_per_shot_usd=0.00014,
        notes=(
            "Heron r1, 133q. Conversion, not an IBM price: at 1.60 USD per second, 0.00014 "
            "per shot assumes a throughput of about 11,400 shots per second."
        ),
    ),
    "ibm_marrakesh": VendorPricing(
        backend="ibm_marrakesh",
        provider="ibm",
        cost_per_shot_usd=0.00016,
        notes=(
            "Heron r2, 156q. Conversion, not an IBM price: at 1.60 USD per second, 0.00016 "
            "per shot assumes a throughput of 10,000 shots per second."
        ),
    ),
    # IonQ (AWS Braket)
    "ionq_aria": VendorPricing(
        backend="ionq_aria",
        cost_per_task_usd=0.30,
        provider="ionq",
        cost_per_shot_usd=0.03,
        notes=(
            "Aria-2, 25q. Not on the Amazon Braket pricing page as of 2026-09-17, but the AWS "
            "Price List API still carries it at 0.03 per shot, read 2026-09-18. On Azure, IonQ "
            "bills per gate-shot instead: 0.000220 per one-qubit gate-shot and 0.000975 per "
            "two-qubit gate-shot, with a per-program minimum. The Braket per-shot figure is the "
            "one used here."
        ),
    ),
    "ionq_forte": VendorPricing(
        backend="ionq_forte",
        cost_per_task_usd=0.30,
        provider="ionq",
        cost_per_shot_usd=0.08,
        notes=(
            "Forte-1, 36q, Amazon Braket published price, checked 2026-09-17. On Azure, IonQ "
            "bills per gate-shot instead: 0.0001645 per one-qubit gate-shot and 0.001121 per "
            "two-qubit gate-shot. The Braket per-shot figure is the one used here."
        ),
    ),
    # IQM (AWS Braket)
    "iqm_garnet": VendorPricing(
        backend="iqm_garnet",
        cost_per_task_usd=0.30,
        provider="iqm",
        cost_per_shot_usd=0.00145,
        notes="Garnet, 20q, Amazon Braket published price, checked 2026-09-17",
    ),
    "iqm_emerald": VendorPricing(
        backend="iqm_emerald",
        cost_per_task_usd=0.30,
        provider="iqm",
        cost_per_shot_usd=0.00160,
        notes="Emerald, Amazon Braket published price, checked 2026-09-17",
    ),
    # Rigetti (AWS Braket)
    "rigetti_ankaa": VendorPricing(
        backend="rigetti_ankaa",
        cost_per_task_usd=0.30,
        provider="rigetti",
        cost_per_shot_usd=0.0009,
        notes=(
            "Ankaa-3, 84q. 0.0009 per shot from the AWS Price List API, read 2026-09-18. Not on "
            "the Amazon Braket pricing page as of 2026-09-17. The 0.00035 carried here before "
            "that read is the Aspen generation price, which the API still reports for Aspen-10, "
            "Aspen-11 and the Aspen-M devices. On Azure, Rigetti bills 0.02 USD per 10 ms of "
            "execution time instead."
        ),
    ),
    "rigetti_cepheus": VendorPricing(
        backend="rigetti_cepheus",
        cost_per_task_usd=0.30,
        provider="rigetti",
        cost_per_shot_usd=0.000425,
        notes="Cepheus, Amazon Braket published price, checked 2026-09-17",
    ),
    # Quantinuum
    "quantinuum_h2": VendorPricing(
        backend="quantinuum_h2",
        provider="quantinuum",
        cost_per_shot_usd=8.00,
        notes=(
            "H2, 56q. A model, not a Quantinuum price: Quantinuum bills in HQCs, where "
            "HQC = 5 + C(N1q + 10*N2q + 5*Nm)/5000 for C shots. The Standard plan, 125,000 USD "
            "a month for 10,000 HQCs, implies 12.50 USD per HQC, and 8.00 per shot corresponds "
            "in the large-C limit to N1q + 10*N2q + 5*Nm of about 3,200: roughly 280 two-qubit "
            "gates, 120 one-qubit gates and 56 measurements. A shallower circuit costs less, a "
            "deeper one more. Pay-As-You-Go per-HQC pricing is quote-only."
        ),
    ),
}


def get_pricing(backend: str) -> VendorPricing | None:
    """Look up pricing for *backend*, returning *None* if unknown."""
    _warn_if_stale()
    return VENDOR_PRICING.get(backend)


def _cost_per_shot_impl(backend: str) -> float:
    """Return cost-per-shot in USD, raising :class:`KeyError` if unknown."""
    entry = VENDOR_PRICING.get(backend)
    if entry is None:
        raise KeyError(
            f"No pricing data for backend '{backend}'. "
            f"Known backends: {sorted(VENDOR_PRICING.keys())}"
        )
    return entry.cost_per_shot_usd


def cost_per_shot(backend: str) -> float:
    """Per-shot price for *backend* (USD). Emits the staleness warning."""
    _warn_if_stale()
    return _cost_per_shot_impl(backend)
