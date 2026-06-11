"""QEC memory-experiment preflight: simulation-derived sizing signals.

Answers "what should I expect if I run a distance-``d``, ``r``-round
rotated surface-code memory experiment on this backend?" before any
hardware time is spent.  The preflight builds a stim memory circuit at
a physical-error proxy for the backend, decodes it with PyMatching, and
reports projection *signals* only:

* a projected logical error rate (LER) with a Wilson 95 percent band,
* the projected detector-event density,
* the shot counts needed to resolve that LER to a given relative
  confidence-interval width.

These are observability signals, not recommendations.  qb-compiler does
not decide whether the experiment is "worth running", apply thresholds,
or gate submission; downstream tooling owns any policy built on top.

The physical-error proxy is deliberately coarse: when derived from a
backend spec it maps the backend's median two-qubit gate error onto
stim's uniform si1000-style circuit-noise parameter.  Real devices have
structured noise (crosstalk, leakage, drift, readout asymmetry) that a
uniform depolarizing model cannot capture, so the projection is a
sizing aid, not a forecast.

Requires the ``[ising]`` extra (``stim`` + ``pymatching``)::

    pip install qb-compiler[ising]

Quick start
-----------

::

    from qb_compiler import qec_preflight

    report = qec_preflight(distance=3, rounds=3, backend="ibm_fez")
    print(report)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: Relative CI half-widths the preflight sizes shot budgets for.
_REL_CI_WIDTHS: tuple[float, ...] = (0.5, 0.2, 0.1)

#: z^2 for a 95 percent two-sided normal interval (1.96 ** 2 rounded).
_Z_SQUARED_95: float = 3.84

#: Compute-cost guards (simulation cost only, not an experiment policy).
_MAX_DISTANCE: int = 9
_MAX_ROUNDS: int = 50

_PROXY_NOTE: str = (
    "physical_error_proxy derived from the backend's median two-qubit gate "
    "error, mapped onto stim's uniform si1000-style noise parameter as a "
    "coarse proxy"
)
_SIM_NOTE: str = (
    "simulation-derived projection from a uniform depolarizing proxy; "
    "real-device LER depends on structured noise the proxy cannot capture"
)
_FLOOR_NOTE: str = (
    "no logical errors at this size in simulation; experiment is below "
    "measurable floor at this budget"
)


@dataclass(frozen=True)
class QECPreflightResult:
    """Sizing signals for a surface-code memory experiment.

    Parameters
    ----------
    backend:
        Backend name the proxy was derived from, or ``None`` when the
        caller supplied ``physical_error`` directly.
    distance:
        Code distance of the projected experiment.
    rounds:
        Number of stabiliser-measurement rounds.
    basis:
        Logical memory basis, ``"X"`` or ``"Z"``.
    physical_error_proxy:
        Uniform circuit-noise parameter used in the stim simulation.
    projected_ler:
        Logical error rate observed in the preflight simulation.
    projected_ler_band:
        Wilson 95 percent confidence interval ``(lo, hi)`` on
        :attr:`projected_ler`.
    projected_detector_fraction:
        Fraction of detectors that fired across all sampled shots
        (detector-event density).
    shots_for_rel_ci:
        Mapping from relative CI half-width (as a string, e.g.
        ``"0.1"``) to the shot count needed to resolve the projected
        LER at that relative width.  Entries are ``0`` when no logical
        errors were observed in simulation.
    notes:
        Free-form caveats describing how the projection was produced.
    """

    backend: str | None
    distance: int
    rounds: int
    basis: str
    physical_error_proxy: float
    projected_ler: float
    projected_ler_band: tuple[float, float]
    projected_detector_fraction: float
    shots_for_rel_ci: dict[str, int]
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lo, hi = self.projected_ler_band
        backend = self.backend or "(physical_error supplied)"
        lines = [
            f"QEC preflight: d={self.distance}, rounds={self.rounds}, "
            f"basis={self.basis}, backend={backend}",
            f"  physical error proxy : {self.physical_error_proxy:.3e}",
            f"  projected LER        : {self.projected_ler:.3e} (95% band {lo:.3e} .. {hi:.3e})",
            f"  detector fraction    : {self.projected_detector_fraction:.4f}",
        ]
        shot_parts = [
            f"rel {width} -> {self.shots_for_rel_ci[width]}" for width in self.shots_for_rel_ci
        ]
        lines.append(f"  shots for rel. CI    : {', '.join(shot_parts)}")
        if self.notes:
            lines.append("  notes:")
            lines.extend(f"    - {note}" for note in self.notes)
        return "\n".join(lines)


def _wilson_interval(successes: int, trials: int, z_squared: float) -> tuple[float, float]:
    """Wilson score interval for a binomial rate ``successes / trials``."""
    if trials == 0:
        return (0.0, 0.0)
    p_hat = successes / trials
    z2_over_n = z_squared / trials
    denom = 1.0 + z2_over_n
    centre = p_hat + z2_over_n / 2.0
    margin = math.sqrt(z_squared * (p_hat * (1.0 - p_hat) / trials + z2_over_n / (4.0 * trials)))
    return ((centre - margin) / denom, (centre + margin) / denom)


def _shots_for_relative_ci(p_l: float, rel_width: float) -> int:
    """Shots needed so the 95 percent CI half-width is ``rel_width * p_l``.

    Uses the normal approximation for a binomial rate: half-width
    ``z * sqrt(p (1 - p) / n)`` equals ``rel_width * p`` at
    ``n = z^2 * (1 - p) / (p * rel_width^2)``.  A sizing signal only;
    it does not say whether that budget should be spent.
    """
    return math.ceil(_Z_SQUARED_95 * (1.0 - p_l) / (p_l * rel_width * rel_width))


def qec_preflight(
    distance: int,
    rounds: int,
    *,
    basis: str = "Z",
    backend: str | None = None,
    physical_error: float | None = None,
    shots_sim: int = 20_000,
    seed: int = 0,
) -> QECPreflightResult:
    """Project sizing signals for a surface-code memory experiment.

    Builds a stim rotated-memory circuit at a uniform physical-error
    proxy, samples ``shots_sim`` detector shots, decodes them with
    PyMatching, and summarises the result as :class:`QECPreflightResult`.

    Parameters
    ----------
    distance:
        Code distance.  Odd, ``3 <= distance <= 9`` (upper bound is a
        simulation compute-cost guard, not an experiment policy).
    rounds:
        Stabiliser-measurement rounds, ``1 <= rounds <= 50`` (same
        compute-cost guard).
    basis:
        Logical memory basis, ``"X"`` or ``"Z"``.
    backend:
        qb-compiler backend name.  Used only to derive the physical
        error proxy from :func:`qb_compiler.config.get_backend_spec`
        when ``physical_error`` is not given; the backend's median
        two-qubit gate error is mapped onto stim's uniform si1000-style
        noise parameter as a coarse proxy.
    physical_error:
        Explicit uniform physical-error parameter.  Takes precedence
        over the backend-derived proxy.
    shots_sim:
        Detector shots to sample in the preflight simulation.
    seed:
        Seed for stim's detector sampler.

    Returns
    -------
    QECPreflightResult
        Projection signals; see the class docstring for fields.

    Raises
    ------
    ValueError
        On out-of-range ``distance`` / ``rounds`` / ``shots_sim``, or
        when neither ``backend`` nor ``physical_error`` is supplied.
    ImportError
        When ``stim`` / ``pymatching`` are not installed.
    """
    try:
        import pymatching
        import stim
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "qec_preflight requires stim and pymatching. "
            "Install via `pip install qb-compiler[ising]`."
        ) from exc

    # Imported lazily: qb_compiler.ising requires the [ising] extra.
    from qb_compiler.ising.patch_spec import SurfaceCodePatchSpec

    if distance > _MAX_DISTANCE:
        raise ValueError(
            f"distance={distance} exceeds the preflight simulation guard "
            f"(distance <= {_MAX_DISTANCE}).  This bounds local compute cost only; "
            f"for larger patches run your own stim study."
        )
    if rounds > _MAX_ROUNDS:
        raise ValueError(
            f"rounds={rounds} exceeds the preflight simulation guard "
            f"(rounds <= {_MAX_ROUNDS}).  This bounds local compute cost only; "
            f"for longer memories run your own stim study."
        )
    if shots_sim < 1:
        raise ValueError(f"shots_sim must be >= 1, got {shots_sim}")

    notes: list[str] = []
    if physical_error is not None:
        p = physical_error
    elif backend is not None:
        from qb_compiler.config import get_backend_spec

        p = get_backend_spec(backend).median_cx_error
        notes.append(_PROXY_NOTE)
    else:
        raise ValueError("supply either backend= or physical_error= to set the noise proxy")

    # Validates distance (odd, >= 3), rounds (>= 1), basis, and p range,
    # and resolves the stim task name for the chosen basis.
    spec = SurfaceCodePatchSpec(distance=distance, rounds=rounds, basis=basis, p_error=p)  # type: ignore[arg-type]

    circuit = stim.Circuit.generated(
        spec.stim_task_name,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        before_round_data_depolarization=p / 10.0,
    )
    sampler = circuit.compile_detector_sampler(seed=seed)
    detector_events, observable_flips = sampler.sample(shots_sim, separate_observables=True)

    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    predictions = matching.decode_batch(detector_events.astype(np.uint8)).astype(bool)

    failures = int(np.any(predictions != observable_flips, axis=1).sum())
    projected_ler = failures / shots_sim
    band = _wilson_interval(failures, shots_sim, _Z_SQUARED_95)
    detector_fraction = float(detector_events.mean())

    shots_for_rel_ci: dict[str, int] = {}
    if failures == 0:
        shots_for_rel_ci = {str(width): 0 for width in _REL_CI_WIDTHS}
        notes.append(_FLOOR_NOTE)
    else:
        p_l = max(projected_ler, 1.0 / shots_sim)
        shots_for_rel_ci = {
            str(width): _shots_for_relative_ci(p_l, width) for width in _REL_CI_WIDTHS
        }

    notes.append(_SIM_NOTE)

    return QECPreflightResult(
        backend=backend,
        distance=distance,
        rounds=rounds,
        basis=spec.basis,
        physical_error_proxy=p,
        projected_ler=projected_ler,
        projected_ler_band=band,
        projected_detector_fraction=detector_fraction,
        shots_for_rel_ci=shots_for_rel_ci,
        notes=notes,
    )
