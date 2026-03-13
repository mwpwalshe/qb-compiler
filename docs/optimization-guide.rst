Optimization Level Guide
========================

qb-compiler provides four optimization levels (0–3) and three compilation
strategies. This guide helps you choose the right settings.

Optimization Levels
-------------------

Each level adds more passes to the compilation pipeline:

.. list-table::
   :header-rows: 1
   :widths: 10 25 35 30

   * - Level
     - Name
     - What It Does
     - When to Use
   * - 0
     - None
     - Gate decomposition to native basis only. No optimization.
     - Debugging, verifying circuit equivalence.
   * - 1
     - Light
     - Decomposition + adjacent gate cancellation (H-H, X-X, CX-CX).
     - Quick compilation, large circuits where compilation time matters.
   * - 2
     - Medium (default)
     - Level 1 + rotation merging + calibration-aware mapping +
       noise-aware routing.
     - **Most workloads.** Good balance of quality and speed.
   * - 3
     - Heavy
     - Level 2 run twice (second pass catches opportunities exposed
       by the first). More VF2 candidates explored.
     - Maximising fidelity for production experiments. Slower compilation.

Setting the optimization level:

.. code-block:: python

   from qb_compiler import QBCompiler

   # Via from_backend
   compiler = QBCompiler.from_backend("ibm_fez", optimization_level=3)

   # Via config
   from qb_compiler.config import CompilerConfig
   config = CompilerConfig(backend="ibm_fez", optimization_level=3)
   compiler = QBCompiler(config)

Compilation Strategies
----------------------

Strategies control **what the compiler optimizes for**, independent of
the optimization level:

``fidelity_optimal`` (default)
   Maximise estimated output fidelity. The CalibrationMapper explores
   more candidate layouts and picks the one with lowest weighted error.
   Use this when circuit quality is paramount.

``depth_optimal``
   Minimise compiled circuit depth. Prioritises parallel gate execution.
   Useful when decoherence during idle time is the dominant error source.

``budget_optimal``
   Minimise execution cost. Uses lighter optimization (level 1 max) and
   recommends the cheapest viable shot count. Best for parameter sweeps
   and iterative algorithms like VQE.

.. code-block:: python

   compiler = QBCompiler.from_backend("ibm_fez", strategy="budget_optimal")
   result = compiler.compile(circuit)

Decision Tree
-------------

.. code-block:: text

   Is this a production experiment?
   ├── YES → optimization_level=3, strategy="fidelity_optimal"
   ├── NO, just testing → optimization_level=0 or 1
   └── Is budget a concern?
       ├── YES → strategy="budget_optimal"
       └── NO
           ├── Circuit depth > 100? → strategy="depth_optimal"
           └── Otherwise → strategy="fidelity_optimal" (default)

Performance vs Quality
----------------------

Rough compilation time for a 20-qubit circuit with 50 CX gates on a
modern laptop:

.. list-table::
   :header-rows: 1

   * - Level
     - Compilation Time
     - Fidelity Improvement
   * - 0
     - < 10 ms
     - Baseline
   * - 1
     - ~ 20 ms
     - +0.5–1%
   * - 2
     - ~ 100 ms
     - +2–4%
   * - 3
     - ~ 500 ms
     - +3–5%

These are approximate. Actual performance depends on circuit structure,
backend size, and available calibration data.

Calibration-Aware vs Topology-Only
----------------------------------

The biggest fidelity gains come from calibration-aware mapping (level 2+).
On IBM Fez (156 qubits), calibration-aware mapping provides +2.4% to +4.5%
improvement over topology-only mapping.

This improvement increases with:

- **More qubits** — larger devices have more variation in qubit quality
- **More 2Q gates** — each CX/CZ gate accumulates error
- **Higher error variance** — devices with large error spreads benefit most

The improvement decreases when:

- **Circuits use most qubits** — fewer layout choices available
- **All qubits are similar quality** — calibration adds less signal
- **Circuits are very shallow** — readout error dominates

Combining Level and Strategy
-----------------------------

.. code-block:: python

   # Maximum quality
   compiler = QBCompiler.from_backend("ibm_fez",
       optimization_level=3,
       strategy="fidelity_optimal",
   )

   # Fast iteration during development
   compiler = QBCompiler.from_backend("ibm_fez",
       optimization_level=1,
       strategy="budget_optimal",
   )

   # Depth-sensitive circuit (long coherence-limited experiment)
   compiler = QBCompiler.from_backend("ibm_fez",
       optimization_level=2,
       strategy="depth_optimal",
   )
