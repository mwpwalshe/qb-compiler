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
   Optimization level 0-3. Default: 2.

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

qbc chem-audit
--------------

Run the five integrity checks over a qubit Hamiltonian file.

.. code-block:: bash

   qbc chem-audit hamiltonian.json [--strict] [--json]

``--strict``
   Treat an undeclared field as a failure. What CI should use.

``--json``
   Emit a ``qb.chem_audit.v1`` receipt instead of text.

Exit codes: 0 ACCEPT, 1 INCOMPLETE, 2 REFUSE, 3 unreadable file. See
:doc:`chemistry` for what each check means.

qbc measure-plan
----------------

Price the measurement of a Hamiltonian before submitting it.

.. code-block:: bash

   qbc measure-plan hamiltonian.json [--shots-per-setting 4096] [--json]

Reports measurable terms, qubit-wise commuting settings, the grouping factor,
the largest group, and total shots. A structural count: no variance weighting
and no precision claim. See :doc:`chemistry`.

qbc verify-receipt
------------------

Check a receipt offline against a public key you were given.

.. code-block:: bash

   qbc verify-receipt receipt.json [--key KEY] [--trusted-keys FILE] [--strict] [--json]

``--key``
   The signer's public key: base64, hex, or a path to a file holding one.

``--trusted-keys``
   A file of public keys, one per line. Defaults to ``QBC_TRUSTED_KEYS``, then
   ``~/.qb-compiler/trusted_keys``.

``--strict``
   Also fail when the receipt carries no signature.

Exit codes: 0 verified (or unsigned without ``--strict``), 1 cannot be checked,
2 does not verify. See :doc:`receipts`.

qbc corpus
----------

Public QEC datasets and whether your copy of one is intact.

.. code-block:: bash

   qbc corpus list [--json]
   qbc corpus show NAME
   qbc corpus verify NAME PATH [--json]

``verify`` exits 0 when the digest matches, 1 when the file is missing or the
name is unknown, and 2 on a mismatch. Nothing is mirrored by this package. See
:doc:`corpora`.

More Environment Variables
--------------------------

``QBC_SIGNING_KEY``
   Path to the Ed25519 private key used by ``sign=True``. Defaults to
   ``~/.qb-compiler/signing_key``, created once with mode 0600 on first use.

``QBC_TRUSTED_KEYS``
   Path to a file of public keys used by ``qbc verify-receipt`` when no key is
   passed on the command line. Defaults to ``~/.qb-compiler/trusted_keys``.

``QBC_DATA_DIR``
   Where local receipt and verification logs are appended. Defaults to
   ``~/.qb_compiler``.
