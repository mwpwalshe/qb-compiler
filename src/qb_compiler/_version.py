"""Version information for qb-compiler."""

from __future__ import annotations

__version__ = "0.4.0b1"

try:
    from importlib.metadata import version as _meta_version

    __version__ = _meta_version("qb-compiler")
except Exception:
    pass
