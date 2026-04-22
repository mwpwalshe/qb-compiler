"""Full pass manager combining calibration-aware layout with Qiskit optimisation.

:class:`QBPassManager` creates a Qiskit ``StagedPassManager`` with a custom
layout stage powered by :class:`QBCalibrationLayout`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qb_compiler.qiskit_plugin.transpiler_plugin import (
    QBCalibrationLayout,
    _load_calibration_dict,
)

if TYPE_CHECKING:
    from pathlib import Path

    from qiskit.circuit import QuantumCircuit

logger = logging.getLogger(__name__)


class QBPassManager:
    """Calibration-aware pass manager that wraps Qiskit's staged pipeline.

    Creates a Qiskit ``StagedPassManager`` (via ``generate_preset_pass_manager``)
    with the layout stage augmented by :class:`QBCalibrationLayout`.

    Parameters
    ----------
    optimization_level:
        Qiskit optimization level (0-3).
    calibration_data:
        Calibration dict or path — used for layout at level >= 2.
    basis_gates:
        Target basis gate set (e.g. ``["cx", "rz", "sx", "x", "id"]``).
    coupling_map:
        Device coupling map as a list of ``[q_i, q_j]`` edges, or a
        Qiskit :class:`CouplingMap`.
    """

    def __init__(
        self,
        optimization_level: int = 2,
        calibration_data: dict | str | Path | None = None,
        basis_gates: list[str] | None = None,
        coupling_map: CouplingMap | list[list[int]] | None = None,
    ) -> None:
        self._opt_level = optimization_level

        # Parse calibration
        self._cal_dict: dict | None = None
        if calibration_data is not None:
            self._cal_dict = _load_calibration_dict(calibration_data)

        # Resolve coupling map
        if coupling_map is not None and not isinstance(coupling_map, CouplingMap):
            coupling_map = CouplingMap(couplinglist=coupling_map)
        elif (
            coupling_map is None and self._cal_dict is not None and "coupling_map" in self._cal_dict
        ):
            coupling_map = CouplingMap(couplinglist=self._cal_dict["coupling_map"])

        self._coupling_map = coupling_map
        self._basis_gates = basis_gates

        # Build the underlying Qiskit pass manager
        self._pm = self._build()

    def _build(self) -> Any:
        """Construct the Qiskit StagedPassManager."""
        pm = generate_preset_pass_manager(
            optimization_level=self._opt_level,
            basis_gates=self._basis_gates,
            coupling_map=self._coupling_map,
        )

        # Inject calibration-aware layout for levels 2+
        if self._cal_dict is not None and self._opt_level >= 2:
            cal_pass = QBCalibrationLayout(self._cal_dict)
            pm.layout.append(cal_pass)
            logger.info(
                "QBPassManager: injected QBCalibrationLayout at level %d",
                self._opt_level,
            )

        return pm

    def run(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Transpile *circuit* through the full pipeline.

        Parameters
        ----------
        circuit:
            Input Qiskit ``QuantumCircuit``.

        Returns
        -------
        QuantumCircuit
            Transpiled circuit targeting the configured backend.
        """
        return self._pm.run(circuit)

    @classmethod
    def from_backend(
        cls,
        backend: str,
        calibration_data: dict | str | Path | None = None,
        optimization_level: int = 2,
    ) -> QBPassManager:
        """Create a pass manager pre-configured for a qb-compiler backend.

        Pulls basis gates and (if available) coupling map from the
        qb-compiler backend registry.

        Parameters
        ----------
        backend:
            Backend name (e.g. ``"ibm_fez"``).
        calibration_data:
            Optional calibration dict or path.
        optimization_level:
            Qiskit optimization level (0-3).
        """
        from qb_compiler.config import get_backend_spec

        spec = get_backend_spec(backend)

        # Build coupling map from calibration if available
        coupling_map = None
        cal_dict = None
        if calibration_data is not None:
            cal_dict = _load_calibration_dict(calibration_data)
            if "coupling_map" in cal_dict:
                coupling_map = CouplingMap(couplinglist=cal_dict["coupling_map"])

        return cls(
            optimization_level=optimization_level,
            calibration_data=cal_dict,
            basis_gates=list(spec.basis_gates),
            coupling_map=coupling_map,
        )
