Compiler API
============

This page documents the primary user-facing classes for compiling quantum
circuits with qb-compiler.

QBCompiler
----------

The main entry point.  Create an instance via
:meth:`~qb_compiler.compiler.QBCompiler.from_backend` and call
:meth:`~qb_compiler.compiler.QBCompiler.compile` to compile a circuit.

.. autoclass:: qb_compiler.compiler.QBCompiler
   :members:
   :undoc-members:
   :show-inheritance:

Compilation Results
-------------------

.. autoclass:: qb_compiler.compiler.CompileResult
   :members:
   :undoc-members:

.. autoclass:: qb_compiler.compiler.PassResult
   :members:
   :undoc-members:

Cost Estimation
---------------

.. autoclass:: qb_compiler.compiler.CostEstimator
   :members:
   :undoc-members:

.. autoclass:: qb_compiler.compiler.CostEstimate
   :members:
   :undoc-members:

Configuration
-------------

.. autoclass:: qb_compiler.config.CompilerConfig
   :members:
   :undoc-members:

.. autoclass:: qb_compiler.config.BackendSpec
   :members:
   :undoc-members:

.. autodata:: qb_compiler.config.BACKEND_CONFIGS

.. autofunction:: qb_compiler.config.get_backend_spec

Circuit Representation
----------------------

.. autoclass:: qb_compiler.compiler.QBCircuit
   :members:
   :undoc-members:

.. autoclass:: qb_compiler.compiler.GateOp
   :members:
   :undoc-members:

Protocols
---------

These runtime-checkable protocols define the extension points that
calibration providers and noise models must satisfy.

.. autoclass:: qb_compiler.compiler.CalibrationProvider
   :members:

.. autoclass:: qb_compiler.compiler.NoiseModel
   :members:

Exceptions
----------

.. automodule:: qb_compiler.exceptions
   :members:
   :undoc-members:
   :show-inheritance:
