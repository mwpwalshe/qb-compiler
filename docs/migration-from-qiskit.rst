Migrating from Qiskit Transpiler
=================================

If you're already using Qiskit's ``transpile()`` or ``generate_preset_pass_manager()``,
this guide shows how to integrate qb-compiler with minimal changes.

Option 1: Drop-in Replacement
------------------------------

Replace ``transpile()`` with ``qb_transpile()``:

**Before (Qiskit only):**

.. code-block:: python

   from qiskit import QuantumCircuit, transpile

   qc = QuantumCircuit(3)
   qc.h(0)
   qc.cx(0, 1)
   qc.cx(1, 2)
   qc.measure_all()

   compiled = transpile(qc, backend=backend, optimization_level=3)

**After (qb-compiler):**

.. code-block:: python

   from qiskit import QuantumCircuit
   from qb_compiler.qiskit_plugin import qb_transpile

   qc = QuantumCircuit(3)
   qc.h(0)
   qc.cx(0, 1)
   qc.cx(1, 2)
   qc.measure_all()

   compiled = qb_transpile(qc, backend="ibm_fez")

The returned circuit is a standard Qiskit ``QuantumCircuit`` that you can
submit to any Qiskit runtime as before.

Option 2: Add as a Qiskit Pass
------------------------------

Keep your existing pass manager and add qb-compiler's calibration-aware
layout as an additional pass:

.. code-block:: python

   from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
   from qb_compiler.qiskit_plugin import QBCalibrationLayout

   # Your existing pass manager
   pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

   # Add calibration-aware layout
   pm.layout.append(QBCalibrationLayout(calibration_data))

   compiled = pm.run(circuit)

This approach lets you keep all your existing Qiskit passes and add
calibration awareness on top.

Option 3: Full QBCompiler API
-----------------------------

For maximum control, use the QBCompiler API directly and convert back
to Qiskit when needed:

.. code-block:: python

   from qb_compiler import QBCompiler, QBCircuit

   compiler = QBCompiler.from_backend("ibm_fez")
   circ = QBCircuit(3).h(0).cx(0, 1).cx(1, 2).measure_all()

   result = compiler.compile(circ)

   # Get a Qiskit QuantumCircuit for submission
   qiskit_circuit = result.compiled_circuit.to_qiskit()

   # Submit to Qiskit Runtime as usual
   # job = sampler.run(qiskit_circuit)

Key Differences from Qiskit
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - Qiskit ``transpile()``
     - qb-compiler
   * - Layout selection
     - Topology-based (SabreLayout)
     - Calibration-weighted VF2
   * - SWAP routing
     - Minimise SWAP count (SabreSwap)
     - Minimise accumulated error (Dijkstra)
   * - Scheduling
     - ALAP/ASAP (static)
     - T1/T2 urgency-weighted ALAP
   * - Fidelity estimate
     - Not included
     - Included in ``CompileResult``
   * - Cost estimate
     - Not included
     - Included with ``estimate_cost()``
   * - Budget enforcement
     - Not included
     - ``budget_usd`` parameter
   * - Multi-vendor
     - IBM only
     - IBM, Rigetti, IonQ, IQM

What Stays the Same
^^^^^^^^^^^^^^^^^^^

- qb-compiler uses Qiskit's ``QuantumCircuit`` type. No conversion needed.
- Gate decomposition uses the same native basis gates.
- The output circuit is submission-ready for Qiskit Runtime.
- All standard Qiskit gates are supported.

What You Gain
^^^^^^^^^^^^^

- **+2–5% estimated fidelity** from calibration-aware mapping on IBM backends
- **Cost estimation** before execution
- **Budget enforcement** to prevent surprise bills
- **Multi-vendor support** from a single API
- **Pre-execution fidelity prediction** to know if your circuit is viable
