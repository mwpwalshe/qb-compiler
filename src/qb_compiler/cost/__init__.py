"""Cost estimation and vendor pricing.

``cost_per_shot`` and ``get_pricing`` read the table that ships with the package and are what they
have always been. ``get_pricing_provider`` is the same table behind a provider that can also read
a signed feed when live pricing is asked for, and every number it returns says whether it is live,
cached or static.
"""

from __future__ import annotations

from qb_compiler.cost.billing import (
    CostBreakdown,
    PerGateShot,
    PerHQC,
    PerSecond,
    PerShot,
)
from qb_compiler.cost.estimator import CostEstimate, CostEstimator
from qb_compiler.cost.pricing import VENDOR_PRICING, VendorPricing, cost_per_shot, get_pricing
from qb_compiler.cost.pricing_feed import PricingEntry
from qb_compiler.cost.pricing_provider import (
    FeedPricingProvider,
    StaticPricingProvider,
    get_pricing_provider,
)

__all__ = [
    "VENDOR_PRICING",
    "CostBreakdown",
    "CostEstimate",
    "CostEstimator",
    "FeedPricingProvider",
    "PerGateShot",
    "PerHQC",
    "PerSecond",
    "PerShot",
    "PricingEntry",
    "StaticPricingProvider",
    "VendorPricing",
    "cost_per_shot",
    "get_pricing",
    "get_pricing_provider",
]
