# SPDX-License-Identifier: Apache-2.0
"""Find tendrils: the built-in ones, and any package that registers one.

A package registers a tendril with an entry point in the ``qb_compiler.tendrils`` group, pointing
at a :class:`~qb_compiler.tendrils.model.Tendril`, a dict in the same shape, or a function with no
arguments that returns either::

    [project.entry-points."qb_compiler.tendrils"]
    myvendor = "myvendor_qbc:TENDRIL"

A plugin that fails to load is listed as broken with the reason. It never stops the others.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib.metadata import entry_points
from typing import Any

from qb_compiler.tendrils.builtin import builtin_tendrils
from qb_compiler.tendrils.model import Tendril, TendrilError

ENTRY_POINT_GROUP = "qb_compiler.tendrils"


@dataclass(frozen=True)
class BrokenTendril:
    """A registered tendril that could not be read."""

    name: str
    origin: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "origin": self.origin, "reason": self.reason}


def _coerce(obj: Any, origin: str) -> Tendril:
    if callable(obj) and not isinstance(obj, Tendril):
        obj = obj()
    if isinstance(obj, Tendril):
        return replace(obj, origin=origin)
    return Tendril.from_dict(obj, origin=origin)


def discover(include_plugins: bool = True) -> tuple[list[Tendril], list[BrokenTendril]]:
    """Every tendril that loads, and every registered one that does not."""
    found = builtin_tendrils()
    broken: list[BrokenTendril] = []
    if not include_plugins:
        return found, broken
    taken = {t.name for t in found}
    for ep in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda e: e.name):
        dist = getattr(getattr(ep, "dist", None), "name", None) or "unknown package"
        origin = f"plugin:{dist}"
        try:
            tendril = _coerce(ep.load(), origin)
        except TendrilError as exc:
            broken.append(BrokenTendril(ep.name, origin, str(exc)))
            continue
        except Exception as exc:  # a third-party import fault is reported, never raised
            broken.append(BrokenTendril(ep.name, origin, f"{type(exc).__name__}: {exc}"))
            continue
        if tendril.name in taken:
            broken.append(BrokenTendril(ep.name, origin, f"name {tendril.name!r} is already taken"))
            continue
        taken.add(tendril.name)
        found.append(tendril)
    return found, broken


def get_tendril(name: str) -> Tendril | None:
    """The tendril called *name*, or None."""
    found, _ = discover()
    return next((t for t in found if t.name == name), None)
