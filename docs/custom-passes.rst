Writing Custom Passes
=====================

qb-compiler's pass infrastructure is designed for extensibility. You can
write custom analysis or transformation passes and compose them with
built-in passes.

Pass Contract
-------------

Every pass follows a strict contract:

1. ``__init__`` takes **configuration only** — no circuit data
2. ``run(circuit, context)`` returns a ``PassResult``
3. Analysis passes override ``analyze(circuit, context)`` — read-only
4. Transformation passes override ``transform(circuit, context)`` — may modify

Base Classes
------------

There are two base classes to choose from:

- ``AnalysisPass`` — override ``analyze(circuit, context) -> None``.
  Writes results to ``context`` but does not modify the circuit.
- ``TransformationPass`` — override ``transform(circuit, context) -> PassResult``.
  May return a new circuit.

Example: Analysis Pass — Gate Frequency
----------------------------------------

.. code-block:: python

   from qb_compiler.passes.base import AnalysisPass
   from qb_compiler.ir.circuit import QBCircuit

   class GateFrequencyAnalysis(AnalysisPass):
       """Count how many times each gate type appears."""

       @property
       def name(self) -> str:
           return "GateFrequencyAnalysis"

       def analyze(self, circuit: QBCircuit, context: dict) -> None:
           freq: dict[str, int] = {}
           for gate in circuit.gates:
               freq[gate.name] = freq.get(gate.name, 0) + 1
           context["gate_frequencies"] = freq

Usage:

.. code-block:: python

   from qb_compiler.ir.circuit import QBCircuit
   from qb_compiler.ir.operations import QBGate

   circ = QBCircuit(n_qubits=2, n_clbits=0)
   circ.add_gate(QBGate("h", (0,)))
   circ.add_gate(QBGate("cx", (0, 1)))

   analysis = GateFrequencyAnalysis()
   ctx = {}
   analysis.run(circ, ctx)
   print(ctx["gate_frequencies"])
   # {'h': 1, 'cx': 1}

Example: Transformation Pass — Identity Gate Removal
-----------------------------------------------------

.. code-block:: python

   from qb_compiler.passes.base import TransformationPass, PassResult
   from qb_compiler.ir.circuit import QBCircuit

   class IdentityGateRemoval(TransformationPass):
       """Remove identity (id) gates from the circuit."""

       @property
       def name(self) -> str:
           return "IdentityGateRemoval"

       def transform(self, circuit: QBCircuit, context: dict) -> PassResult:
           non_id_gates = [g for g in circuit.gates if g.name != "id"]
           removed = len(list(circuit.gates)) - len(non_id_gates)

           if removed == 0:
               return PassResult(circuit=circuit, modified=False)

           result = QBCircuit(
               n_qubits=circuit.n_qubits,
               n_clbits=circuit.n_clbits,
               name=circuit.name,
           )
           for gate in non_id_gates:
               result.add_gate(gate)
           for m in circuit.measurements:
               result.add_measurement(m.qubit, m.clbit)

           context["identity_gates_removed"] = removed
           return PassResult(circuit=result, modified=True)

Example: Redundant Gate Removal
-------------------------------

This pass removes adjacent pairs of self-inverse gates (H-H, X-X, CX-CX
on the same qubits):

.. code-block:: python

   from qb_compiler.passes.base import TransformationPass, PassResult
   from qb_compiler.ir.circuit import QBCircuit

   SELF_INVERSE = frozenset({"h", "x", "y", "z", "cx", "cz", "swap"})

   class RedundantGateRemoval(TransformationPass):
       """Remove adjacent pairs of self-inverse gates."""

       @property
       def name(self) -> str:
           return "RedundantGateRemoval"

       def transform(self, circuit: QBCircuit, context: dict) -> PassResult:
           gates = list(circuit.gates)
           new_gates = []
           modified = False
           i = 0

           while i < len(gates):
               if (
                   i + 1 < len(gates)
                   and gates[i].name in SELF_INVERSE
                   and gates[i].name == gates[i + 1].name
                   and gates[i].qubits == gates[i + 1].qubits
               ):
                   i += 2  # Skip both gates
                   modified = True
               else:
                   new_gates.append(gates[i])
                   i += 1

           if not modified:
               return PassResult(circuit=circuit, modified=False)

           result = QBCircuit(
               n_qubits=circuit.n_qubits,
               n_clbits=circuit.n_clbits,
               name=circuit.name,
           )
           for gate in new_gates:
               result.add_gate(gate)
           for m in circuit.measurements:
               result.add_measurement(m.qubit, m.clbit)

           context["redundant_gates_removed"] = len(gates) - len(new_gates)
           return PassResult(circuit=result, modified=True)

Composing Passes with PassManager
----------------------------------

Use ``PassManager`` to chain passes together:

.. code-block:: python

   from qb_compiler.passes.base import PassManager

   pm = PassManager([
       GateFrequencyAnalysis(),
       IdentityGateRemoval(),
       RedundantGateRemoval(),
       GateFrequencyAnalysis(),  # Re-analyze after cleanup
   ])

   result = pm.run_all(circuit)
   # result.circuit is the cleaned circuit
   # result.metadata["passes"] has timing for each pass

Using Custom Passes with QBCompiler
------------------------------------

You can run custom passes on compiled output:

.. code-block:: python

   from qb_compiler import QBCompiler, QBCircuit

   compiler = QBCompiler.from_backend("ibm_fez")
   circ = QBCircuit(2).h(0).cx(0, 1).measure_all()
   result = compiler.compile(circ)

   # Run a custom analysis on the compiled circuit
   analysis = GateFrequencyAnalysis()
   ctx = {}
   analysis.run(result.compiled_circuit, ctx)
   print(ctx["gate_frequencies"])

Testing Custom Passes
---------------------

Test with known circuits and expected outcomes:

.. code-block:: python

   def test_identity_removal():
       from qb_compiler.ir.circuit import QBCircuit
       from qb_compiler.ir.operations import QBGate

       circ = QBCircuit(n_qubits=2, n_clbits=0)
       circ.add_gate(QBGate("h", (0,)))
       circ.add_gate(QBGate("id", (1,)))
       circ.add_gate(QBGate("cx", (0, 1)))

       pass_ = IdentityGateRemoval()
       result = pass_.run(circ, {})

       assert result.modified is True
       assert result.circuit.gate_count == 2  # h + cx

   def test_no_ids_is_noop():
       circ = QBCircuit(n_qubits=1, n_clbits=0)
       circ.add_gate(QBGate("h", (0,)))

       pass_ = IdentityGateRemoval()
       result = pass_.run(circ, {})

       assert result.modified is False

Tips
----

- Keep passes **focused** — one pass, one transformation.
- Use ``AnalysisPass`` for read-only passes, ``TransformationPass`` for mutations.
- Test with edge cases: empty circuits, single-gate circuits, measurements.
- Passes should be **idempotent** — running twice produces the same result.
- The ``context`` dict is shared across all passes in a ``PassManager``.
  Use descriptive keys to avoid collisions.
