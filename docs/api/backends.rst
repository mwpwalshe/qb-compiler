Backends, Calibration & Noise
=============================

This page covers backend target descriptions, vendor-specific adapters,
calibration data providers, and noise modelling.

Backend Target
--------------

:class:`~qb_compiler.backends.base.BackendTarget` encapsulates the physical
constraints of a quantum device -- qubit count, native gate set, and coupling
topology.

.. automodule:: qb_compiler.backends.base
   :members:
   :undoc-members:
   :show-inheritance:

Vendor Adapters
---------------

Each vendor module provides basis-gate constants, gate-time tables, and
calibration parsers that normalise hardware-specific data into the
qb-compiler common format.

IBM
~~~

.. automodule:: qb_compiler.backends.ibm
   :members:
   :undoc-members:

.. automodule:: qb_compiler.backends.ibm.adapter
   :members:
   :undoc-members:

.. automodule:: qb_compiler.backends.ibm.calibration
   :members:
   :undoc-members:

Rigetti
~~~~~~~

.. automodule:: qb_compiler.backends.rigetti
   :members:
   :undoc-members:

.. automodule:: qb_compiler.backends.rigetti.calibration
   :members:
   :undoc-members:

IonQ
~~~~

.. automodule:: qb_compiler.backends.ionq
   :members:
   :undoc-members:

.. automodule:: qb_compiler.backends.ionq.calibration
   :members:
   :undoc-members:

IQM
~~~

.. automodule:: qb_compiler.backends.iqm
   :members:
   :undoc-members:

.. automodule:: qb_compiler.backends.iqm.calibration
   :members:
   :undoc-members:

Calibration Providers
---------------------

Calibration providers supply per-qubit and per-gate error data to the
compiler.  The abstract base class defines the interface; concrete
implementations fetch data from files, caches, or live APIs.

.. automodule:: qb_compiler.calibration.provider
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.calibration.static_provider
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.calibration.cached_provider
   :members:
   :undoc-members:
   :show-inheritance:

Calibration Models
~~~~~~~~~~~~~~~~~~

Vendor-neutral data classes that represent calibration snapshots.

.. automodule:: qb_compiler.calibration.models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.calibration.models.backend_properties
   :members:
   :undoc-members:
   :show-inheritance:

Noise Modelling
---------------

Noise models provide per-gate depolarizing rates and fidelity estimates
used by noise-aware compilation passes.

.. automodule:: qb_compiler.noise
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.noise.empirical_model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: qb_compiler.noise.fidelity_estimator
   :members:
   :undoc-members:
   :show-inheritance:
