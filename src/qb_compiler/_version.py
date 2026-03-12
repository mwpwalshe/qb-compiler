"""Version information for qb-compiler."""

__version__ = "0.1.0"

try:
    from importlib.metadata import version as _meta_version

    __version__ = _meta_version("qb-compiler")
except Exception:
    pass
