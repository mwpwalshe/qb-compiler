Architecture
============

Pass pipeline
-------------

qb-compiler transforms circuits through an ordered sequence of **passes**,
managed by :class:`~qb_compiler.compiler.PassManager`. Each pass receives
a :class:`~qb_compiler.compiler.QBCircuit` and a
:class:`~qb_compiler.config.CompilerConfig`, and returns a (possibly
modified) circuit.

The default pipeline at each optimisation level:

- **Level 0** — Basis translation only. Decomposes gates into the target
  backend's native gate set.
- **Level 1** — Adds gate cancellation. Adjacent self-inverse gate pairs
  (H-H, X-X, CX-CX on the same qubits) are eliminated.
- **Level 2** (default) — Adds rotation merging. Consecutive single-qubit
  rotations on the same axis are combined and normalised.
- **Level 3** — Aggressive. Runs cancellation and rotation merge twice
  to catch opportunities exposed by earlier passes.

Pass types
^^^^^^^^^^

Passes are organised into categories under ``qb_compiler.passes``:

- **Transformation** — gate cancellation, rotation merge, decomposition
- **Mapping** — qubit routing and layout selection (coupling-aware)
- **Scheduling** — noise-aware gate ordering
- **Analysis** — depth / fidelity / cost estimation
- **QEC** — ancilla reservation for error correction (future)

Calibration system
------------------

The calibration subsystem provides per-qubit and per-gate error data
to noise-aware passes.

:class:`~qb_compiler.calibration.provider.CalibrationProvider` is the
abstract interface. Concrete implementations include:

- :class:`~qb_compiler.calibration.static_provider.StaticCalibrationProvider`
  — serves data from a JSON snapshot file (offline / testing).
- :class:`~qb_compiler.calibration.cached_provider.CachedProvider`
  — wraps another provider with time-based caching.

Calibration data is modelled by:

- :class:`~qb_compiler.calibration.models.backend_properties.BackendProperties`
  — full device snapshot
- :class:`~qb_compiler.calibration.models.qubit_properties.QubitProperties`
  — per-qubit T1, T2, readout errors
- :class:`~qb_compiler.calibration.models.coupling_properties.GateProperties`
  — per-gate error rate and duration

Fidelity estimation
-------------------

The compiler estimates output-state fidelity using an analytic depolarising
model:

.. math::

   F \approx \prod_{\text{gate } g} (1 - e_g)

where :math:`e_g` is the gate error rate from calibration data (or the
backend's median error as fallback). Readout errors are applied
multiplicatively for measurement operations.

Cost estimation
---------------

:class:`~qb_compiler.compiler.CostEstimator` computes execution cost
in USD using the backend's ``cost_per_shot`` scaled by a depth factor
(proxy for wall-clock time on pay-per-second platforms like IBM Utility
tier).
