Getting started
===============

Installation
------------

Install from source (recommended during alpha)::

    pip install -e ".[dev]"

Or install the package directly::

    pip install qb-compiler

Quick start
-----------

Compile a circuit for IBM Fez:

.. code-block:: python

   from qb_compiler import QBCompiler, QBCircuit

   # Create a GHZ circuit
   circ = QBCircuit(4).h(0).cx(0, 1).cx(1, 2).cx(2, 3).measure_all()

   # Compile for a specific backend
   compiler = QBCompiler.from_backend("ibm_fez")
   result = compiler.compile(circ)

   print(f"Original depth:  {result.original_depth}")
   print(f"Compiled depth:  {result.compiled_depth}")
   print(f"Depth reduction: {result.depth_reduction_pct:.1f}%")
   print(f"Est. fidelity:   {result.estimated_fidelity:.4f}")

Compilation strategies
^^^^^^^^^^^^^^^^^^^^^^

qb-compiler supports three strategies:

- ``fidelity_optimal`` (default) — maximise estimated output fidelity
- ``depth_optimal`` — minimise circuit depth
- ``budget_optimal`` — minimise estimated execution cost

.. code-block:: python

   compiler = QBCompiler.from_backend("ibm_fez", strategy="budget_optimal")
   result = compiler.compile(circ, budget_usd=5.00)

Using calibration data
^^^^^^^^^^^^^^^^^^^^^^

Load a calibration snapshot for noise-aware compilation:

.. code-block:: python

   from qb_compiler.calibration.static_provider import StaticCalibrationProvider

   calib = StaticCalibrationProvider.from_json("calibration_hub/heron/ibm_fez_2026-02-15.json")
   compiler = QBCompiler.from_backend("ibm_fez", calibration=calib)
   result = compiler.compile(circ)

CLI usage
---------

qb-compiler ships a CLI (``qbc``) for quick compilation::

    qbc compile circuit.qasm --backend ibm_fez --strategy fidelity_optimal
