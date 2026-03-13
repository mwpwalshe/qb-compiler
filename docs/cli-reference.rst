CLI Reference
=============

qb-compiler provides the ``qbc`` command-line tool for compiling circuits,
inspecting backends, and viewing calibration data.

Global Options
--------------

.. code-block:: bash

   qbc --version        # Show version
   qbc --help           # Show help

qbc compile
-----------

Compile a quantum circuit file for a target backend.

.. code-block:: bash

   qbc compile <circuit_file> --backend <backend_name> [options]

Arguments:

``circuit_file``
   Path to a QASM 2.0 or QASM 3.0 file.

Options:

``--backend, -b`` (required)
   Target backend name. Examples: ``ibm_fez``, ``ibm_torino``,
   ``rigetti_ankaa``, ``ionq_aria``, ``iqm_garnet``.

``--strategy, -s``
   Compilation strategy. Choices: ``fidelity_optimal`` (default),
   ``depth_optimal``, ``budget_optimal``.

``--output, -o``
   Path to write the compiled QASM output. If omitted, prints to stdout.

``--optimization-level``
   Optimization level 0–3. Default: 2.

``--budget``
   Budget constraint in USD. Raises error if estimated cost exceeds this.

``--compare``
   Compare compiled output with Qiskit default transpilation and show
   metrics side by side.

Examples:

.. code-block:: bash

   # Basic compilation
   qbc compile bell.qasm --backend ibm_fez

   # With strategy and output file
   qbc compile circuit.qasm -b ibm_fez -s depth_optimal -o compiled.qasm

   # With budget constraint
   qbc compile circuit.qasm -b ionq_aria --budget 50.0

   # Compare with Qiskit default
   qbc compile circuit.qasm -b ibm_fez --compare

qbc info
--------

Show available backends and their specifications.

.. code-block:: bash

   qbc info

Output includes:

- Backend name and vendor
- Number of qubits
- Native basis gates
- Connectivity type
- Estimated per-shot cost

qbc calibration show
--------------------

Display calibration data for a backend.

.. code-block:: bash

   qbc calibration show <backend_name>

Shows:

- Calibration timestamp
- Per-qubit T1, T2, readout error
- Gate error rates
- Best and worst qubits
- Coupling map summary

Example output:

.. code-block:: text

   Backend: ibm_fez (IBM Heron, 156 qubits)
   Calibration: 2026-03-12T10:00:00Z

   Top 5 qubits (lowest readout error):
     Q42:  T1=380μs  T2=190μs  readout=0.0055
     Q87:  T1=350μs  T2=175μs  readout=0.0060
     ...

   Top 5 CZ links (lowest error):
     CZ(42,43): 0.0015
     CZ(87,88): 0.0018
     ...

   Worst 5 qubits (highest readout error):
     Q103: T1=120μs  T2=55μs  readout=0.0890
     ...

Environment Variables
---------------------

``QBC_CALIBRATION_DIR``
   Directory to search for calibration JSON snapshots. If set, the compiler
   looks here before falling back to bundled test fixtures.

   .. code-block:: bash

      export QBC_CALIBRATION_DIR=/path/to/calibration/snapshots
      qbc compile circuit.qasm -b ibm_fez
