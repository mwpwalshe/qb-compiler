Compiler Passes
===============

qb-compiler uses a pass-based architecture.  Every pass inherits from
:class:`~qb_compiler.passes.base.BasePass` and is executed through a
:class:`~qb_compiler.passes.base.PassManager`.

Pass Infrastructure
-------------------

.. automodule:: qb_compiler.passes.base
   :members:
   :undoc-members:
   :show-inheritance:

Analysis Passes
---------------

Analysis passes inspect the circuit without modifying it.  They write
results into the shared ``context`` dictionary for downstream passes to
consume.

.. automodule:: qb_compiler.passes.analysis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.passes.analysis.depth_analysis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.passes.analysis.gate_count
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.passes.analysis.connectivity_check
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.passes.analysis.error_budget_estimator
   :members:
   :undoc-members:
   :show-inheritance:

Transformation Passes
---------------------

Transformation passes modify the circuit -- decomposing gates, cancelling
redundant operations, or simplifying sub-circuits.

.. automodule:: qb_compiler.passes.transformation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.passes.transformation.gate_decomposition
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.passes.transformation.gate_cancellation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.passes.transformation.commutation_analysis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.passes.transformation.circuit_simplification
   :members:
   :undoc-members:
   :show-inheritance:

Mapping Passes
--------------

Mapping passes handle logical-to-physical qubit assignment and routing.
The calibration-aware mapper is qb-compiler's key differentiator -- it
uses live device error rates to select the highest-fidelity qubit subset.

.. automodule:: qb_compiler.passes.mapping
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.passes.mapping.topology_mapper
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.passes.mapping.correlated_error_router
   :members:
   :undoc-members:
   :show-inheritance:

Scheduling Passes
-----------------

Scheduling passes determine the time ordering of gates to minimise
decoherence or satisfy hardware timing constraints.

.. automodule:: qb_compiler.passes.scheduling
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.passes.scheduling.asap_scheduler
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.passes.scheduling.alap_scheduler
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.passes.scheduling.noise_aware_scheduler
   :members:
   :undoc-members:
   :show-inheritance:

Qiskit Plugin Passes
--------------------

These passes subclass Qiskit's native ``TransformationPass`` and
``AnalysisPass`` base classes, making them directly composable with any
Qiskit ``PassManager`` or ``StagedPassManager``.

.. automodule:: qb_compiler.qiskit_plugin.calibration_pass
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.qiskit_plugin.transformation_passes
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.qiskit_plugin.analysis_passes
   :members:
   :undoc-members:
   :show-inheritance:

QEC Passes
----------

Quantum error correction passes prepare circuits for fault-tolerant
execution.  These passes require the QubitBoost SDK (>= 2.5).

.. automodule:: qb_compiler.passes.qec
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.passes.qec.logical_mapping
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.passes.qec.syndrome_scheduling
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.passes.qec.correlated_error_avoidance
   :members:
   :undoc-members:
   :show-inheritance:
