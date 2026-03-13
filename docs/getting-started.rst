Getting Started
===============

Installation
------------

Install from PyPI::

    pip install qb-compiler

Or install from source for development::

    git clone https://github.com/mwpwalshe/qb-compiler.git
    cd qb-compiler
    pip install -e ".[dev]"

Optional backend support
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Rigetti support (requires pyquil)
   pip install "qb-compiler[rigetti]"

   # IonQ support (via Amazon Braket)
   pip install "qb-compiler[ionq]"

   # IQM support
   pip install "qb-compiler[iqm]"

   # All backends
   pip install "qb-compiler[all]"

Quick Start
-----------

5 lines to better circuits
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from qb_compiler import QBCompiler, QBCircuit

   circuit = QBCircuit(3).h(0).cx(0, 1).cx(1, 2).measure_all()
   compiler = QBCompiler.from_backend("ibm_fez")
   result = compiler.compile(circuit)

   print(f"Depth: {result.compiled_depth}")
   print(f"Est. fidelity: {result.estimated_fidelity:.4f}")
   print(f"Depth reduction: {result.depth_reduction_pct:.1f}%")

The ``QBCircuit`` fluent API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Build circuits using method chaining:

.. code-block:: python

   from qb_compiler import QBCircuit
   import math

   circ = (
       QBCircuit(4, name="my_circuit")
       .h(0)
       .cx(0, 1)
       .cx(1, 2)
       .cx(2, 3)
       .rz(0, math.pi / 4)
       .rz(1, math.pi / 3)
       .measure_all()
   )

   print(f"Qubits: {circ.n_qubits}")
   print(f"Depth:  {circ.depth}")
   print(f"Gates:  {circ.gate_count}")

Compilation Strategies
^^^^^^^^^^^^^^^^^^^^^^

qb-compiler supports three compilation strategies:

``fidelity_optimal`` (default)
   Maximise estimated output fidelity. Uses calibration-aware mapping,
   noise-aware routing, and aggressive gate cancellation. Best when
   circuit quality is more important than compilation speed.

``depth_optimal``
   Minimise compiled circuit depth. Useful when decoherence is the
   dominant error source and shorter circuits are critical.

``budget_optimal``
   Minimise estimated execution cost. Uses lighter optimization (level 1)
   and targets the cheapest viable configuration. Best when running many
   circuits on a budget.

.. code-block:: python

   # Compare strategies
   for strategy in ["fidelity_optimal", "depth_optimal", "budget_optimal"]:
       compiler = QBCompiler.from_backend("ibm_fez", strategy=strategy)
       result = compiler.compile(circ)
       print(f"{strategy}: depth={result.compiled_depth}, "
             f"fidelity={result.estimated_fidelity:.4f}")

Using Calibration Data
^^^^^^^^^^^^^^^^^^^^^^

Load a calibration snapshot for noise-aware compilation:

.. code-block:: python

   from qb_compiler.calibration.static_provider import StaticCalibrationProvider

   provider = StaticCalibrationProvider.from_json(
       "tests/fixtures/calibration_snapshots/ibm_fez_2026_03_12.json"
   )
   compiler = QBCompiler.from_backend("ibm_fez", calibration=provider)
   result = compiler.compile(circ)

   print(f"Calibration timestamp: {provider.properties.timestamp}")
   print(f"Qubits available: {provider.properties.n_qubits}")

See :doc:`calibration-guide` for full details on calibration providers.

Budget Enforcement
^^^^^^^^^^^^^^^^^^

Set a maximum budget and the compiler will raise ``BudgetExceededError``
if the estimated cost exceeds it:

.. code-block:: python

   from qb_compiler import BudgetExceededError

   try:
       result = compiler.compile(circ, budget_usd=0.50)
   except BudgetExceededError as e:
       print(f"Too expensive: estimated ${e.estimated_usd:.2f} "
             f"exceeds budget ${e.budget_usd:.2f}")

Cost estimation:

.. code-block:: python

   cost = compiler.estimate_cost(result.compiled_circuit, shots=4096)
   print(f"Estimated cost: ${cost.total_usd:.4f}")
   print(f"Per-shot: ${cost.cost_per_shot_usd:.6f}")

Qiskit Integration
^^^^^^^^^^^^^^^^^^

Drop-in replacement for Qiskit's transpile:

.. code-block:: python

   from qb_compiler.qiskit_plugin import qb_transpile

   compiled = qb_transpile(circuit, backend="ibm_fez")

Or as a pass in an existing Qiskit pipeline:

.. code-block:: python

   from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
   from qb_compiler.qiskit_plugin import QBCalibrationLayout

   pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
   pm.layout.append(QBCalibrationLayout(calibration_data))
   compiled = pm.run(circuit)

CLI Usage
---------

qb-compiler ships a CLI (``qbc``) for quick compilation:

.. code-block:: bash

   # Compile a QASM file
   qbc compile circuit.qasm --backend ibm_fez --strategy fidelity_optimal

   # Show available backends
   qbc info

   # Show calibration data
   qbc calibration show ibm_fez

See :doc:`cli-reference` for the full CLI reference.

Next Steps
----------

- :doc:`optimization-guide` — choosing the right optimization level and strategy
- :doc:`calibration-guide` — deep dive on the calibration system
- :doc:`custom-passes` — writing your own compiler passes
- :doc:`tutorials/01-basic-compilation` — comprehensive interactive tutorial
