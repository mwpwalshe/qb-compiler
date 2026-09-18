"""How vendors actually bill, and what a price estimate is allowed to leave unsaid.

Three of the four models here are not per shot. IBM bills per second, Azure bills IonQ per gate
shot and Quantinuum per HQC, and a flat per-shot number for any of those is a model with a stated
circuit behind it or it is nothing. So every model returns a :class:`CostBreakdown` carrying the
number **and the assumptions it stood on**, never a bare float. A caller that wants only the float
reads ``breakdown.usd`` and has to walk past the assumptions to do it.

A model that needs gate counts and is not given them falls back to a per-shot approximation and
says so in the breakdown, rather than refusing or guessing silently.

Usage::

    from qb_compiler.cost.billing import PerShot

    breakdown = PerShot(0.00016).job_cost(4096)
    breakdown.usd            # 0.65536
    breakdown.assumptions    # {'cost_per_shot_usd': 0.00016, 'cost_per_task_usd': 0.0, ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: The three things a number can be: fetched in this call, served from a verified cache, or the
#: table that shipped with the package.
LIVE = "live"
CACHED = "cached"
STATIC = "static"


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """One priced job: the number, the model that produced it, and what it assumed.

    Attributes
    ----------
    usd :
        Total cost in US dollars.
    model :
        Name of the billing model, e.g. ``"per_shot"``.
    assumptions :
        Every input the number stood on, including the ones the caller did not supply and the
        model had to assume. Read this before quoting the number.
    as_of :
        The date the price was last checked against the vendor, ISO-8601.
    source :
        Where the price came from, in words.
    status :
        ``live``, ``cached`` or ``static``.
    """

    usd: float
    model: str
    assumptions: dict[str, Any] = field(default_factory=dict)
    as_of: str = ""
    source: str = ""
    status: str = STATIC

    def as_dict(self) -> dict[str, Any]:
        return {
            "usd": self.usd,
            "model": self.model,
            "assumptions": dict(self.assumptions),
            "as_of": self.as_of,
            "source": self.source,
            "status": self.status,
        }

    def __str__(self) -> str:
        return f"{self.usd:.6f} USD ({self.model}, {self.status}, priced {self.as_of or 'undated'})"


def _finish(
    usd: float,
    model: str,
    assumptions: dict[str, Any],
    as_of: str,
    source: str,
    status: str,
) -> CostBreakdown:
    return CostBreakdown(
        usd=float(usd),
        model=model,
        assumptions=assumptions,
        as_of=as_of,
        source=source,
        status=status,
    )


@dataclass(frozen=True, slots=True)
class PerShot:
    """A published per-shot price, plus a per-task fee. What Braket vendors charge.

    These are the only numbers in this package that a vendor actually invoices per shot.
    """

    cost_per_shot_usd: float
    cost_per_task_usd: float = 0.0
    name: str = "per_shot"

    def job_cost(
        self,
        shots: int,
        *,
        tasks: int = 1,
        one_qubit_gates: int | None = None,
        two_qubit_gates: int | None = None,
        measurements: int | None = None,
        as_of: str = "",
        source: str = "",
        status: str = STATIC,
    ) -> CostBreakdown:
        del one_qubit_gates, two_qubit_gates, measurements  # a per-shot price ignores the circuit
        usd = self.cost_per_shot_usd * shots + self.cost_per_task_usd * tasks
        assumptions = {
            "shots": int(shots),
            "tasks": int(tasks),
            "cost_per_shot_usd": self.cost_per_shot_usd,
            "cost_per_task_usd": self.cost_per_task_usd,
            "basis": "published per-shot price plus per-task fee",
        }
        return _finish(usd, self.name, assumptions, as_of, source, status)


@dataclass(frozen=True, slots=True)
class PerSecond:
    """Billing by wall-clock second, converted to shots by an assumed throughput.

    IBM publishes Pay-As-You-Go at 96 USD per minute billed per second, which is 1.60 per second,
    and publishes no per-shot price at all. The shot rate here is **an assumption of ours**, it is
    named in every breakdown, and a circuit that runs slower than it costs more than the number
    below says.
    """

    usd_per_second: float
    assumed_shots_per_second: float = 10000.0
    name: str = "per_second"

    @property
    def cost_per_shot_usd(self) -> float:
        return self.usd_per_second / self.assumed_shots_per_second

    @classmethod
    def from_per_shot(cls, usd_per_second: float, cost_per_shot_usd: float) -> PerSecond:
        """Build from a published conversion, keeping that per-shot number exact."""
        return cls(
            usd_per_second=usd_per_second,
            assumed_shots_per_second=usd_per_second / cost_per_shot_usd,
        )

    def job_cost(
        self,
        shots: int,
        *,
        tasks: int = 1,
        one_qubit_gates: int | None = None,
        two_qubit_gates: int | None = None,
        measurements: int | None = None,
        as_of: str = "",
        source: str = "",
        status: str = STATIC,
    ) -> CostBreakdown:
        del tasks, one_qubit_gates, two_qubit_gates, measurements
        seconds = shots / self.assumed_shots_per_second
        usd = self.usd_per_second * seconds
        assumptions = {
            "shots": int(shots),
            "usd_per_second": self.usd_per_second,
            "assumed_shots_per_second": self.assumed_shots_per_second,
            "implied_seconds": seconds,
            "implied_cost_per_shot_usd": self.cost_per_shot_usd,
            "basis": (
                "the vendor bills per second and publishes no per-shot price; the shot rate is "
                "an assumption of this package, not a vendor figure"
            ),
        }
        return _finish(usd, self.name, assumptions, as_of, source, status)


@dataclass(frozen=True, slots=True)
class PerGateShot:
    """Billing per gate shot, with a per-program minimum. What Azure charges for IonQ.

    Needs gate counts. Without them it falls back to ``fallback``, the vendor's per-shot price on
    another platform, and the breakdown says which number it used and why.
    """

    one_qubit_usd: float
    two_qubit_usd: float
    minimum_per_job_usd: float = 0.0
    fallback: PerShot | None = None
    name: str = "per_gate_shot"

    def job_cost(
        self,
        shots: int,
        *,
        tasks: int = 1,
        one_qubit_gates: int | None = None,
        two_qubit_gates: int | None = None,
        measurements: int | None = None,
        as_of: str = "",
        source: str = "",
        status: str = STATIC,
    ) -> CostBreakdown:
        del measurements
        if one_qubit_gates is None or two_qubit_gates is None:
            if self.fallback is None:
                raise ValueError(
                    "per-gate-shot billing needs one_qubit_gates and two_qubit_gates, and this "
                    "entry carries no per-shot fallback to price without them"
                )
            breakdown = self.fallback.job_cost(
                shots, tasks=tasks, as_of=as_of, source=source, status=status
            )
            assumptions = dict(breakdown.assumptions)
            assumptions["fell_back_from"] = self.name
            assumptions["fell_back_because"] = (
                "no gate counts were supplied, so the per-gate-shot price could not be applied; "
                "this is the vendor's per-shot price on the other platform"
            )
            return _finish(breakdown.usd, self.fallback.name, assumptions, as_of, source, status)
        priced = shots * (
            one_qubit_gates * self.one_qubit_usd + two_qubit_gates * self.two_qubit_usd
        )
        usd = max(priced, self.minimum_per_job_usd)
        assumptions = {
            "shots": int(shots),
            "one_qubit_gates": int(one_qubit_gates),
            "two_qubit_gates": int(two_qubit_gates),
            "one_qubit_usd_per_gate_shot": self.one_qubit_usd,
            "two_qubit_usd_per_gate_shot": self.two_qubit_usd,
            "priced_before_minimum_usd": priced,
            "minimum_per_job_usd": self.minimum_per_job_usd,
            "minimum_applied": bool(usd > priced),
            "basis": "per gate shot, with the vendor's per-program minimum",
        }
        return _finish(usd, self.name, assumptions, as_of, source, status)


@dataclass(frozen=True, slots=True)
class PerHQC:
    """Billing in Quantinuum HQCs.

    ``HQC = base + C(N1q + one_q_weight... )`` as the vendor publishes it:
    ``HQC = 5 + C (N1q + 10 N2q + 5 Nm) / 5000`` for ``C`` shots. Needs gate and measurement
    counts. Without them it falls back to ``fallback``, a per-shot approximation whose circuit is
    named in the breakdown.
    """

    usd_per_hqc: float
    base: float = 5.0
    one_q: float = 1.0
    two_q: float = 10.0
    meas: float = 5.0
    divisor: float = 5000.0
    fallback: PerShot | None = None
    name: str = "per_hqc"

    def hqc(
        self, shots: int, one_qubit_gates: int, two_qubit_gates: int, measurements: int
    ) -> float:
        weighted = (
            self.one_q * one_qubit_gates + self.two_q * two_qubit_gates + self.meas * measurements
        )
        return self.base + shots * weighted / self.divisor

    def job_cost(
        self,
        shots: int,
        *,
        tasks: int = 1,
        one_qubit_gates: int | None = None,
        two_qubit_gates: int | None = None,
        measurements: int | None = None,
        as_of: str = "",
        source: str = "",
        status: str = STATIC,
    ) -> CostBreakdown:
        if one_qubit_gates is None or two_qubit_gates is None or measurements is None:
            if self.fallback is None:
                raise ValueError(
                    "HQC billing needs one_qubit_gates, two_qubit_gates and measurements, and "
                    "this entry carries no per-shot fallback to price without them"
                )
            breakdown = self.fallback.job_cost(
                shots, tasks=tasks, as_of=as_of, source=source, status=status
            )
            assumptions = dict(breakdown.assumptions)
            assumptions["fell_back_from"] = self.name
            assumptions["fell_back_because"] = (
                "no gate or measurement counts were supplied, so the HQC formula could not be "
                "applied; this is a per-shot approximation and the circuit behind it is in the "
                "entry's notes"
            )
            assumptions["usd_per_hqc"] = self.usd_per_hqc
            return _finish(breakdown.usd, self.fallback.name, assumptions, as_of, source, status)
        hqc = self.hqc(shots, one_qubit_gates, two_qubit_gates, measurements)
        usd = hqc * self.usd_per_hqc
        assumptions = {
            "shots": int(shots),
            "one_qubit_gates": int(one_qubit_gates),
            "two_qubit_gates": int(two_qubit_gates),
            "measurements": int(measurements),
            "hqc": hqc,
            "usd_per_hqc": self.usd_per_hqc,
            "formula": (
                f"HQC = {self.base:g} + C({self.one_q:g} N1q + {self.two_q:g} N2q + "
                f"{self.meas:g} Nm) / {self.divisor:g}"
            ),
            "basis": "the vendor's published HQC formula at the stated USD per HQC",
        }
        return _finish(usd, self.name, assumptions, as_of, source, status)


#: Every billing model, by the name it serialises under.
BILLING_MODELS: dict[str, type] = {
    "per_shot": PerShot,
    "per_second": PerSecond,
    "per_gate_shot": PerGateShot,
    "per_hqc": PerHQC,
}

BillingModel = PerShot | PerSecond | PerGateShot | PerHQC


def billing_as_dict(model: BillingModel) -> dict[str, Any]:
    """Serialise a billing model, flat, with its name under ``model``."""
    payload: dict[str, Any] = {"model": model.name}
    for slot in model.__slots__:  # type: ignore[union-attr]
        if slot == "name":
            continue
        value = getattr(model, slot)
        if slot == "fallback":
            payload[slot] = billing_as_dict(value) if value is not None else None
        else:
            payload[slot] = value
    return payload


def billing_from_dict(payload: dict[str, Any]) -> BillingModel:
    """Rebuild a billing model from :func:`billing_as_dict`. Raises on an unknown model name."""
    data = dict(payload)
    name = data.pop("model", None)
    factory = BILLING_MODELS.get(str(name))
    if factory is None:
        raise ValueError(f"unknown billing model {name!r}. Known: {sorted(BILLING_MODELS)}")
    fallback = data.pop("fallback", None)
    if fallback is not None:
        data["fallback"] = billing_from_dict(fallback)
    built: BillingModel = factory(**data)
    return built


__all__ = [
    "BILLING_MODELS",
    "CACHED",
    "LIVE",
    "STATIC",
    "BillingModel",
    "CostBreakdown",
    "PerGateShot",
    "PerHQC",
    "PerSecond",
    "PerShot",
    "billing_as_dict",
    "billing_from_dict",
]
