"""ML-powered layout prediction for calibration-aware qubit mapping.

This module provides two layout predictors that narrow the VF2 search
space in :class:`CalibrationMapper`:

- **XGBoost** (Phase 2): flat feature-based, fast, ``pip install "qb-compiler[ml]"``
- **GNN** (Phase 3): graph-structured, captures topology, ``pip install "qb-compiler[gnn]"``

Check availability at runtime::

    from qb_compiler.ml import is_available, is_gnn_available
    if is_available():
        from qb_compiler.ml.layout_predictor import MLLayoutPredictor
    if is_gnn_available():
        from qb_compiler.ml.gnn_router import GNNLayoutPredictor
"""

from __future__ import annotations

try:
    import xgboost as _xgb  # noqa: F401

    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

try:
    import torch as _torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def is_available() -> bool:
    """Return True if XGBoost ML dependencies are installed."""
    return _HAS_XGBOOST


def is_gnn_available() -> bool:
    """Return True if PyTorch (GNN) dependencies are installed."""
    return _HAS_TORCH


__all__ = ["is_available", "is_gnn_available"]
