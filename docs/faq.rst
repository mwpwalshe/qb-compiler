FAQ & Troubleshooting
=====================

General Questions
-----------------

**What does qb-compiler do that Qiskit's transpiler doesn't?**

Qiskit's transpiler maps circuits using **topology only** — it knows which
qubits are connected but not how noisy each qubit/gate is. qb-compiler
uses **today's calibration data** (T1, T2, gate error, readout error) to
select the best physical qubits. This typically improves estimated fidelity
by 2–5%.

**Do I need a QubitBoost account?**

No. The open-source core works with calibration snapshots (JSON files).
You can use the shipped test fixtures or capture your own from vendor APIs.
A QubitBoost account is only needed for the ``LiveCalibrationProvider``
which fetches real-time calibration data.

**Which backends are supported?**

IBM (Fez, Torino, Marrakesh), Rigetti (Ankaa-3), IonQ (Aria, Forte),
and IQM (Garnet, Emerald). See ``qbc info`` for the full list.

**What Python versions are supported?**

Python 3.10, 3.11, and 3.12.

**Is qb-compiler a replacement for Qiskit?**

No. qb-compiler sits **on top of** Qiskit. It adds calibration-aware
passes to the compilation pipeline. You still use Qiskit for circuit
construction and job submission.

Calibration Questions
---------------------

**How do I get calibration data?**

Three options:

1. **Test fixtures** — qb-compiler ships with calibration snapshots in
   ``tests/fixtures/calibration_snapshots/``. These are real data captured
   from IBM and Rigetti devices.

2. **Capture your own** — Use your vendor's API to download calibration
   data and save as JSON. Set ``QBC_CALIBRATION_DIR`` to point at your
   snapshot directory.

3. **Live feed** — Install ``qubitboost-sdk`` and use
   ``LiveCalibrationProvider`` for real-time data.

**How old can calibration data be?**

Calibration data is valid for 4–24 hours on superconducting devices
(IBM, Rigetti, IQM). Trapped-ion devices (IonQ) have more stable
calibration. The ``CachedCalibrationProvider`` can enforce staleness
limits.

**What if I don't have calibration data?**

The compiler falls back to built-in device specifications with median
error rates. You'll still get topology-aware mapping but without the
calibration advantage.

Compilation Questions
---------------------

**Why is my compiled circuit deeper than the input?**

SWAP gates are inserted when the circuit requires 2-qubit gates between
non-adjacent physical qubits. Each SWAP decomposes to 3 CX/CZ gates.
This is inherent to the hardware topology and happens with any transpiler.

To reduce depth:

- Use ``strategy="depth_optimal"``
- Design circuits with linear connectivity if the backend is linear
- Use fewer qubits to give the mapper more layout options

**How accurate is the fidelity estimate?**

The estimate uses a depolarising model: F ≈ Π(1 - e_gate). This is a
reasonable approximation for short circuits (< 100 2Q gates) but
overestimates fidelity for deep circuits where correlated errors and
decoherence accumulate non-linearly. Use it for **relative comparisons**
(this layout vs that layout) rather than absolute predictions.

**What's the maximum circuit size?**

Practically limited by compilation time. The VF2 subgraph isomorphism
search is the bottleneck:

- Up to 20 qubits: < 1 second
- 20–50 qubits: 1–10 seconds
- 50+ qubits: may need ``max_candidates`` tuning

**Can I compile for a backend I don't have access to?**

Yes. The compiler only needs calibration data, not hardware access.
Compile for any backend with a calibration snapshot.

Cost & Budget Questions
-----------------------

**How accurate are cost estimates?**

Cost estimates use published per-shot pricing. They're accurate for
pay-per-shot platforms (IBM, IonQ). For pay-per-second platforms,
the estimate includes a depth-based multiplier but actual cost
depends on queue time and runtime overhead.

**What happens if I set a budget and the circuit is too expensive?**

``BudgetExceededError`` is raised before returning the compilation
result. The error includes the estimated cost and budget limit so
you can adjust.

**How do I find the cheapest backend?**

.. code-block:: python

   from qb_compiler.cost.budget_optimizer import BudgetOptimizer

   optimizer = BudgetOptimizer()
   result = optimizer.find_cheapest_backend(
       budget_usd=10.0,
       min_qubits=10,
       target_shots=4096,
   )
   print(f"Use {result.backend}: ${result.estimated_cost_usd:.2f}")

Troubleshooting
---------------

**ImportError: No module named 'rustworkx'**

Install ``rustworkx``:

.. code-block:: bash

   pip install rustworkx>=0.14

This is a core dependency and should be installed automatically.
If not, check your Python environment.

**CalibrationMapper finds no valid layout**

This can happen when:

- The circuit needs more qubits than the backend has
- The circuit's interaction graph can't be embedded in the coupling map
- ``max_candidates`` is too low

Fix: increase ``max_candidates`` or ``vf2_call_limit`` in
``CalibrationMapperConfig``.

**Slow compilation for large circuits**

Increase ``max_candidates`` for better quality at the cost of speed,
or decrease for faster compilation:

.. code-block:: python

   from qb_compiler.passes.mapping.calibration_mapper import CalibrationMapperConfig

   # Faster but potentially lower quality
   config = CalibrationMapperConfig(max_candidates=100, vf2_call_limit=10000)

   # Slower but explores more options
   config = CalibrationMapperConfig(max_candidates=50000, vf2_call_limit=500000)

**Tests fail with "fixture not found"**

Make sure you're running tests from the project root:

.. code-block:: bash

   cd /path/to/qb-compiler
   pytest

Getting Help
------------

- **GitHub Issues**: https://github.com/mwpwalshe/qb-compiler/issues
- **Contributing**: See ``CONTRIBUTING.md`` for development setup
- **API docs**: https://docs.qubitboost.io/compiler
