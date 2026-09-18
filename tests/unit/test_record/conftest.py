"""Synthetic repetition-code records for the record tests.

No data, no network, no vendor code. The generator walks a chain of ``d`` data qubits through
``rounds`` of stabilizer measurement and a final data readout, accumulating data errors as it
goes, so the record it produces has the time asymmetry a real one has: a first round that
differences against a reset, a last round that differences against the final data parity, and a
stored readout rate that climbs as errors pile up.

``build_detectors`` reconstructs detectors from the stored blocks the way a loader does, with the
same ``reverse_rounds`` switch, so a test can build the same run in the right order and in the
wrong one and compare.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from qb_compiler.record.dem import build_repetition_dem
from qb_compiler.record.types import FIRST_LAYER_PREMISE, RecordSpec

#: The generator resets before the first round and differences it against the prepared value, so
#: the record it writes has the quiet first round the round profile check assumes. Declared here
#: the way a loader declares it for a platform it knows.
SYNTHETIC_PREMISE = {
    "source": "the generator in this file: the first round differences against the prepared state"
}


@dataclass(frozen=True)
class SyntheticRun:
    """The blocks a repetition-code run leaves behind, before any loader touches them."""

    syndromes: np.ndarray
    final_parity: np.ndarray
    data0: np.ndarray
    logical_state: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    d: int
    rounds: int


def simulate_repetition(
    n_shots: int = 2000,
    d: int = 5,
    rounds: int = 5,
    p_data: float = 0.02,
    p_meas: float = 0.02,
    seed: int = 0,
    n_chains: int = 4,
) -> SyntheticRun:
    """Walk a repetition-code memory forward in time and keep what the hardware would store."""
    rng = np.random.default_rng(seed)
    state = np.zeros((n_shots, d), dtype=np.uint8)
    syndromes = np.zeros((n_shots, rounds, d - 1), dtype=np.uint8)
    for t in range(rounds):
        state ^= (rng.random((n_shots, d)) < p_data).astype(np.uint8)
        parity = state[:, :-1] ^ state[:, 1:]
        syndromes[:, t] = parity ^ (rng.random((n_shots, d - 1)) < p_meas).astype(np.uint8)
    state ^= (rng.random((n_shots, d)) < p_data).astype(np.uint8)
    final_parity = (state[:, :-1] ^ state[:, 1:]).astype(np.uint8)
    logical_state = (rng.random(n_shots) < 0.5).astype(np.uint8)
    data0 = (logical_state ^ state[:, 0]).astype(np.uint8)
    return SyntheticRun(
        syndromes=syndromes,
        final_parity=final_parity,
        data0=data0,
        logical_state=logical_state,
        labels=(data0 ^ logical_state).astype(np.uint8),
        groups=np.asarray([f"chain{i % n_chains}" for i in range(n_shots)]),
        d=d,
        rounds=rounds,
    )


def build_detectors(run: SyntheticRun, reverse_rounds: bool) -> tuple[np.ndarray, np.ndarray]:
    """``(detectors, stored_rounds)`` the way a loader builds them, in the order it believes."""
    syndromes = run.syndromes[:, ::-1, :] if reverse_rounds else run.syndromes
    syndromes = np.ascontiguousarray(syndromes)
    n_shots, rounds, sites = syndromes.shape
    detectors = np.empty((n_shots, rounds + 1, sites), dtype=np.uint8)
    detectors[:, 0] = syndromes[:, 0]
    detectors[:, 1:rounds] = syndromes[:, 1:] ^ syndromes[:, :-1]
    detectors[:, rounds] = run.final_parity ^ syndromes[:, rounds - 1]
    return detectors, syndromes


def make_spec(
    run: SyntheticRun,
    *,
    mirrored: bool = False,
    with_dem: bool = True,
    p_data: float = 0.02,
    p_meas: float = 0.02,
    declare_premise: bool = True,
) -> RecordSpec:
    """A :class:`RecordSpec` built from ``run``.

    ``mirrored=False`` reads the stored rounds the way the generator wrote them, which is the
    right way round. ``mirrored=True`` reverses them, which is the storage-order fault.
    ``declare_premise=False`` leaves the first layer premise off, which is a record from a
    platform nothing has measured.
    """
    detectors, stored = build_detectors(run, reverse_rounds=mirrored)
    dem = (
        build_repetition_dem(run.d, run.rounds, p_data=p_data, p_meas=p_meas) if with_dem else None
    )
    meta = {"loader": "synthetic", "d": run.d, "rounds": run.rounds, "mirrored": mirrored}
    if declare_premise:
        meta[FIRST_LAYER_PREMISE] = dict(SYNTHETIC_PREMISE)
    return RecordSpec(
        detectors=detectors,
        labels=run.labels,
        groups=run.groups,
        raw_syndromes=stored,
        final_parity=run.final_parity,
        logical_state=run.logical_state,
        data0=run.data0,
        dem=dem,
        meta=meta,
    )


@pytest.fixture(scope="session")
def synthetic_run() -> SyntheticRun:
    return simulate_repetition()


@pytest.fixture(scope="session")
def good_spec(synthetic_run: SyntheticRun) -> RecordSpec:
    return make_spec(synthetic_run, mirrored=False)


@pytest.fixture(scope="session")
def mirrored_spec(synthetic_run: SyntheticRun) -> RecordSpec:
    return make_spec(synthetic_run, mirrored=True)
