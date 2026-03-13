Calibration System Guide
========================

The calibration system is qb-compiler's key differentiator. While standard
transpilers map circuits using only device topology, qb-compiler uses
**today's** per-qubit coherence times, per-gate error rates, and readout
fidelities to make every compilation decision.

This guide covers the calibration data model, providers, and how
calibration feeds into compilation.

Data Model
----------

Calibration data is represented by three core classes:

QubitProperties
^^^^^^^^^^^^^^^

Per-qubit hardware characteristics:

.. code-block:: python

   from qb_compiler.calibration.models.qubit_properties import QubitProperties

   qp = QubitProperties(
       qubit_id=0,
       t1_us=350.0,           # T1 relaxation time (microseconds)
       t2_us=180.0,           # T2 dephasing time (microseconds)
       readout_error=0.008,   # Readout assignment error rate
       frequency_ghz=5.1,     # Qubit drive frequency
   )

Key properties:

- **T1** — energy relaxation time. Limits how long a qubit can hold |1⟩ state.
  Higher is better. IBM Heron: 150–400 μs typical.
- **T2** — dephasing time. Limits how long superposition states survive.
  Higher is better. Always T2 ≤ 2×T1.
- **Readout error** — probability of reading 0 when state is 1 or vice versa.
  Lower is better. IBM Heron: 0.5–9% typical.

GateProperties
^^^^^^^^^^^^^^

Per-gate error and timing for a specific qubit pair:

.. code-block:: python

   from qb_compiler.calibration.models.coupling_properties import GateProperties

   gp = GateProperties(
       gate_type="cz",
       qubits=(0, 1),
       error_rate=0.003,       # Gate error rate
       gate_time_ns=68.0,      # Gate duration in nanoseconds
   )

Key properties:

- **Error rate** — probability that the gate introduces an error.
  Lower is better. On IBM Heron, CZ errors range from 0.15% to 11.3%.
- **Gate time** — duration of the gate operation. Longer gates mean more
  decoherence during execution.

BackendProperties
^^^^^^^^^^^^^^^^^

Aggregates all calibration data for a backend:

.. code-block:: python

   from qb_compiler.calibration.models.backend_properties import BackendProperties

   props = BackendProperties(
       backend="ibm_fez",
       provider="ibm",
       n_qubits=156,
       basis_gates=("cz", "rz", "sx", "x", "id"),
       coupling_map=[(0, 1), (1, 0), ...],
       qubit_properties=qubit_props_list,
       gate_properties=gate_props_list,
       timestamp="2026-03-12T10:00:00Z",
   )

Calibration Providers
---------------------

Providers fetch and serve calibration data. All implement the same interface.

StaticCalibrationProvider
^^^^^^^^^^^^^^^^^^^^^^^^^

Serves calibration from an in-memory ``BackendProperties`` object or a JSON
file. No network access. Ideal for testing, offline work, and reproducible
experiments.

.. code-block:: python

   from qb_compiler.calibration.static_provider import StaticCalibrationProvider

   # From a JSON file
   provider = StaticCalibrationProvider.from_json(
       "tests/fixtures/calibration_snapshots/ibm_fez_2026_03_12.json"
   )

   # From a BackendProperties object
   provider = StaticCalibrationProvider(backend_props)

   # Access data
   qp = provider.get_qubit_properties(0)
   gp = provider.get_gate_properties("cz", (0, 1))
   print(f"Q0 T1: {qp.t1_us}μs, CZ(0,1) error: {gp.error_rate}")

CachedCalibrationProvider
^^^^^^^^^^^^^^^^^^^^^^^^^

Wraps any calibration source and automatically refreshes when data goes stale.
Use this in long-running applications:

.. code-block:: python

   from qb_compiler.calibration.cached_provider import CachedCalibrationProvider

   cached = CachedCalibrationProvider(
       provider_factory=lambda: StaticCalibrationProvider.from_json("latest.json"),
       max_age_seconds=3600,     # Refresh every hour
       hard_limit_hours=24.0,    # Reject data older than 24 hours
   )

   # First access triggers the factory
   qp = cached.get_qubit_properties(0)

   # Force a refresh
   cached.invalidate()

LiveProvider (QubitBoost SDK)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fetches real-time calibration data from the QubitBoost calibration hub.
Requires the proprietary ``qubitboost-sdk`` package:

.. code-block:: python

   # pip install qubitboost-sdk
   from qb_compiler.calibration.live_provider import LiveCalibrationProvider

   provider = LiveCalibrationProvider(
       api_key="your-qubitboost-api-key",
       backend="ibm_fez",
   )

Calibration JSON Format
^^^^^^^^^^^^^^^^^^^^^^^

Calibration snapshots are stored as JSON. The format mirrors the
``BackendProperties`` schema:

.. code-block:: json

   {
     "backend": "ibm_fez",
     "provider": "ibm",
     "n_qubits": 156,
     "basis_gates": ["cz", "rz", "sx", "x", "id"],
     "coupling_map": [[0, 1], [1, 0], ...],
     "qubit_properties": [
       {
         "qubit_id": 0,
         "t1_us": 350.0,
         "t2_us": 180.0,
         "readout_error": 0.008,
         "frequency_ghz": 5.1
       },
       ...
     ],
     "gate_properties": [
       {
         "gate_type": "cz",
         "qubits": [0, 1],
         "error_rate": 0.003,
         "gate_time_ns": 68.0
       },
       ...
     ],
     "timestamp": "2026-03-12T10:00:00Z"
   }

How Calibration Feeds Compilation
---------------------------------

CalibrationMapper
^^^^^^^^^^^^^^^^^

The ``CalibrationMapper`` uses calibration data to find the best physical
qubit placement. It builds a weighted graph where edge weights combine:

- **Gate error** (weight=10.0) — CZ/ECR error for each physical connection
- **Coherence** (weight=0.3) — inverse of T1×T2 product for each qubit
- **Readout error** (weight=5.0) — readout assignment error for each qubit

The mapper uses VF2 subgraph isomorphism (via ``rustworkx``) to enumerate
candidate layouts and scores each one. The lowest-score layout wins.

.. code-block:: python

   from qb_compiler.passes.mapping.calibration_mapper import (
       CalibrationMapper,
       CalibrationMapperConfig,
   )

   mapper = CalibrationMapper(
       calibration=backend_props,
       config=CalibrationMapperConfig(
           gate_error_weight=10.0,
           coherence_weight=0.3,
           readout_weight=5.0,
           max_candidates=10000,
       ),
   )

   ctx = {}
   result = mapper.run(circuit, ctx)
   print(f"Layout: {ctx['initial_layout']}")
   print(f"Score:  {ctx['calibration_score']:.6f}")

NoiseAwareRouter
^^^^^^^^^^^^^^^^

After mapping, if the circuit requires 2-qubit gates between non-adjacent
physical qubits, the ``NoiseAwareRouter`` inserts SWAP gates. Unlike
standard routers that minimise SWAP count, it minimises **accumulated
gate error** using Dijkstra shortest-error-path:

.. code-block:: python

   from qb_compiler.passes.mapping.noise_aware_router import NoiseAwareRouter

   gate_errors = {
       (q0, q1): gp.error_rate
       for gp in backend_props.gate_properties
       if len(gp.qubits) == 2 and gp.error_rate is not None
   }

   router = NoiseAwareRouter(
       coupling_map=backend_props.coupling_map,
       gate_errors=gate_errors,
   )
   routed = router.run(mapped_circuit, ctx)

ErrorBudgetEstimator
^^^^^^^^^^^^^^^^^^^^

Predicts circuit fidelity before execution using calibration data:

.. math::

   F \approx \prod_{\text{gate } g} (1 - e_g)
   \times \prod_{\text{qubit } q} (1 - e_{\text{readout},q})

.. code-block:: python

   from qb_compiler.passes.analysis.error_budget_estimator import ErrorBudgetEstimator

   estimator = ErrorBudgetEstimator(
       qubit_properties=backend_props.qubit_properties,
       gate_error_rates={"h": 0.0003, "cx": 0.005, "cz": 0.003},
   )
   estimator.analyze(circuit, ctx)
   print(f"Estimated fidelity: {ctx['estimated_fidelity']:.4f}")

Best Practices
--------------

1. **Use fresh calibration data.** Calibration drifts over hours. Data older
   than 24 hours may not reflect current device state.

2. **Store snapshots for reproducibility.** Save the calibration JSON used
   for each experiment so you can reproduce results.

3. **Check ``calibration_score``.** The mapper's score indicates layout
   quality. Compare scores across compilation runs to track improvements.

4. **Watch for high-error edges.** If the mapper consistently avoids
   certain physical qubits, those qubits may be degraded. Consider
   filing a support ticket with the hardware vendor.

5. **Use ``CachedCalibrationProvider`` in production.** Automatic refresh
   keeps compilation quality high without manual intervention.
