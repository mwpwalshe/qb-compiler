"""Vendor pricing data for quantum backends.

Prices are in USD per shot.  Sources:
- IBM: Qiskit Runtime Utility tier (~$1.60/sec, converted at typical
  shot throughput).
- IonQ: AWS Braket published pricing.
- IQM: AWS Braket published pricing.
- Rigetti: AWS Braket published pricing.
- Quantinuum: Published H-Series pricing (HQC-based, converted to per-shot).
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
# Kept in sync with config.py BackendSpec.cost_per_shot.

# Last manual review of the price table. Update this whenever the numbers are re-checked
# against vendor pricing pages; get_pricing() warns when the table has gone stale.
PRICING_AS_OF = "2026-06-12"
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
        notes="Heron r2, 156q, Utility tier",
    ),
    "ibm_torino": VendorPricing(
        backend="ibm_torino",
        provider="ibm",
        cost_per_shot_usd=0.00014,
        notes="Heron r1, 133q, Utility tier",
    ),
    "ibm_marrakesh": VendorPricing(
        backend="ibm_marrakesh",
        provider="ibm",
        cost_per_shot_usd=0.00016,
        notes="Heron r2, 156q, Utility tier",
    ),
    # IonQ (AWS Braket)
    "ionq_aria": VendorPricing(
        backend="ionq_aria",
        cost_per_task_usd=0.30,
        provider="ionq",
        cost_per_shot_usd=0.03,
        notes="Aria-2, 25q, Braket per-shot + per-task",
    ),
    "ionq_forte": VendorPricing(
        backend="ionq_forte",
        cost_per_task_usd=0.30,
        provider="ionq",
        cost_per_shot_usd=0.08,
        notes="Forte-1, 36q, Braket pricing",
    ),
    # IQM (AWS Braket)
    "iqm_garnet": VendorPricing(
        backend="iqm_garnet",
        cost_per_task_usd=0.30,
        provider="iqm",
        cost_per_shot_usd=0.00045,
        notes="Garnet, 20q, Braket pricing",
    ),
    "iqm_emerald": VendorPricing(
        backend="iqm_emerald",
        cost_per_task_usd=0.30,
        provider="iqm",
        cost_per_shot_usd=0.00020,
        notes="Emerald, 5q, Braket pricing",
    ),
    # Rigetti (AWS Braket)
    "rigetti_ankaa": VendorPricing(
        backend="rigetti_ankaa",
        cost_per_task_usd=0.30,
        provider="rigetti",
        cost_per_shot_usd=0.00035,
        notes="Ankaa-3, 84q, Braket pricing",
    ),
    # Quantinuum
    "quantinuum_h2": VendorPricing(
        backend="quantinuum_h2",
        provider="quantinuum",
        cost_per_shot_usd=8.00,
        notes="H2, 56q, HQC-based pricing (approximate per-shot)",
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
