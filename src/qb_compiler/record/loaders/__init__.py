# SPDX-License-Identifier: Apache-2.0
"""Loaders that build a :class:`~qb_compiler.record.types.RecordSpec` from a stored record.

Three of them:

* :mod:`~qb_compiler.record.loaders.ibm_fez_repetition` reads the layout of IBM Fez
  repetition-code memory runs, handling the stored round order.
* :mod:`~qb_compiler.record.loaders.quera_surface` reads the QuEra surface-code release through
  the decoding framework published with it.
* :mod:`~qb_compiler.record.loaders.generic_npz` reads and writes the ``.npz`` contract, which is
  how a record from anywhere else gets in.

No dataset is redistributed by any of them, and no third-party code is vendored into this package.
"""

from __future__ import annotations

from qb_compiler.record.loaders import generic_npz, ibm_fez_repetition, quera_surface
from qb_compiler.record.loaders.generic_npz import read_npz, write_npz

__all__ = [
    "generic_npz",
    "ibm_fez_repetition",
    "quera_surface",
    "read_npz",
    "write_npz",
]
