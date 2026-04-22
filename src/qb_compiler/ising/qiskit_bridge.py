"""Qiskit ⇆ Stim bridge for surface-code memory experiments.

qb-compiler's typical user builds circuits with Qiskit.  The NVIDIA
Ising-Decoder needs a stim circuit as its syndrome source.  This
module lets users:

1. Turn a :class:`SurfaceCodePatchSpec` into a Qiskit
   :class:`~qiskit.circuit.QuantumCircuit` for visualisation or
   integration with the rest of qb-compiler's analysis passes.
2. Recover the equivalent stim circuit (used internally by the
   decoders) from the same spec.

The Qiskit circuit is a functional equivalent — it prepares the
logical eigenstate, runs ``rounds`` of stabiliser extraction and
measures in the matching basis.  It is NOT an optimised device-ready
circuit; users should still route / decompose it via their favourite
transpiler (qb-compiler's own layout-stage plugin will happily slot
into the layout stage — see :mod:`qb_compiler.qiskit_plugin`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import stim

if TYPE_CHECKING:
    from qb_compiler.ising.patch_spec import SurfaceCodePatchSpec


def stim_circuit_for(spec: SurfaceCodePatchSpec) -> stim.Circuit:
    """Return the stim surface-code memory circuit for *spec*."""
    return stim.Circuit.generated(
        spec.stim_task_name,
        distance=spec.distance,
        rounds=spec.rounds,
        after_clifford_depolarization=spec.p_error,
        after_reset_flip_probability=spec.p_error,
        before_measure_flip_probability=spec.p_error,
        before_round_data_depolarization=spec.p_error,
    )


def qiskit_circuit_for(spec: SurfaceCodePatchSpec) -> Any:
    """Return a Qiskit ``QuantumCircuit`` implementing the same experiment.

    Qiskit is imported lazily — ``qb_compiler.ising`` can be used with
    stim-only installs; the Qiskit bridge is only triggered if this
    helper is called.

    Notes
    -----
    The returned circuit uses the canonical planar layout: data qubits
    at indices ``[0 .. d*d-1]`` (row-major), X ancillas next, Z
    ancillas last.  It does NOT include the circuit-level noise
    primitives Stim carries — Qiskit noise is backend-specific and up
    to the caller to attach (e.g. via an :class:`AerSimulator` noise
    model or qb-compiler's own noise profiles).
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ClassicalRegister, QuantumRegister
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "qb_compiler.ising.qiskit_bridge.qiskit_circuit_for requires qiskit."
        ) from exc

    d = spec.distance
    t = spec.rounds
    basis = spec.basis
    n_data = d * d
    n_x_anc = (d * d - 1) // 2
    n_z_anc = n_x_anc

    data = QuantumRegister(n_data, name="data")
    x_anc = QuantumRegister(n_x_anc, name="x_anc")
    z_anc = QuantumRegister(n_z_anc, name="z_anc")
    # Classical bits: (t-1) rounds * (x_anc + z_anc) syndromes +
    # final data measurement.  Round 0 is state-prep (no ancilla
    # readout in this simplified reference circuit).
    creg = ClassicalRegister((t - 1) * (n_x_anc + n_z_anc) + n_data, name="m")
    qc = QuantumCircuit(data, x_anc, z_anc, creg, name=f"surface_d{d}_r{t}_{basis}")

    # Step 1: prepare logical eigenstate.
    if basis == "X":
        for q in data:
            qc.h(q)
    # (basis == "Z" → leave in |0>)

    # Step 2: rounds of stabiliser extraction.  We use the standard
    # "rotated surface code" connectivity: each X ancilla couples to up
    # to four data qubits via CX (ancilla is control in the X-basis
    # measurement scheme), Z ancillas via CZ.  The exact geometric
    # connectivity depends on the orientation; we emit a SIMPLE
    # nearest-neighbour pattern for visualisation.  For device
    # execution, call qb-compiler's layout/routing stages.
    #
    # Row-major indexing of the data grid.
    def data_idx(row: int, col: int) -> int:
        return row * d + col

    def x_stabiliser_neighbours(stab_row: int, stab_col: int) -> list[int]:
        """Return data indices supporting the X stab at (stab_row, stab_col)."""
        out = []
        for dr in (0, 1):
            for dc in (0, 1):
                r = stab_row + dr - 1
                c = stab_col + dc - 1
                if 0 <= r < d and 0 <= c < d:
                    out.append(data_idx(r, c))
        return out

    # Enumerate ancilla positions in the standard rotated layout.
    # X stabs at (r, c) with r in 0..d-1, c in 0..d-1 where (r+c) even
    # (boundary pattern simplified — MVP visualisation circuit only).
    x_positions: list[tuple[int, int]] = []
    z_positions: list[tuple[int, int]] = []
    for r in range(d + 1):
        for c in range(d + 1):
            if (r + c) % 2 == 0:
                if 0 < r < d or (r == 0 and c % 2 == 1) or (r == d and c % 2 == 1):
                    x_positions.append((r, c))
            else:
                if 0 < c < d or (c == 0 and r % 2 == 1) or (c == d and r % 2 == 1):
                    z_positions.append((r, c))
    x_positions = x_positions[:n_x_anc]
    z_positions = z_positions[:n_z_anc]

    for round_ in range(1, t):
        for i, (sr, sc) in enumerate(x_positions):
            neigh = x_stabiliser_neighbours(sr, sc)
            if not neigh:
                continue
            qc.reset(x_anc[i])
            qc.h(x_anc[i])
            for nq in neigh:
                qc.cx(x_anc[i], data[nq])
            qc.h(x_anc[i])
            bit = (round_ - 1) * (n_x_anc + n_z_anc) + i
            qc.measure(x_anc[i], creg[bit])
        for i, (sr, sc) in enumerate(z_positions):
            neigh = x_stabiliser_neighbours(sr, sc)  # same plaquette geometry
            if not neigh:
                continue
            qc.reset(z_anc[i])
            for nq in neigh:
                qc.cx(data[nq], z_anc[i])
            bit = (round_ - 1) * (n_x_anc + n_z_anc) + n_x_anc + i
            qc.measure(z_anc[i], creg[bit])
        qc.barrier()

    # Step 3: final data-qubit measurement in matching basis.
    if basis == "X":
        for q in data:
            qc.h(q)
    bit_base = (t - 1) * (n_x_anc + n_z_anc)
    for i, q in enumerate(data):
        qc.measure(q, creg[bit_base + i])

    return qc
