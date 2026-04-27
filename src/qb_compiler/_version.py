"""Version information for qb-compiler."""

from __future__ import annotations

__version__ = "0.5.1"

try:
    from importlib.metadata import version as _meta_version

    __version__ = _meta_version("qb-compiler")
except Exception:
    pass
