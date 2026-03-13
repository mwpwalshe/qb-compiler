"""Example 07: Writing and using a custom compiler pass.

Shows how to extend qb-compiler with your own analysis and transformation
passes using the AnalysisPass and TransformationPass base classes.
"""

from qb_compiler.ir.circuit import QBCircuit
from qb_compiler.ir.operations import QBGate
from qb_compiler.passes.base import AnalysisPass, PassResult, TransformationPass


class GateFrequencyAnalysis(AnalysisPass):
    """Count how many times each gate type appears in the circuit.

    This is an analysis pass -- it does not modify the circuit.
    The results are stored in the context dict under 'gate_frequencies'.
    """

    @property
    def name(self) -> str:
        return "GateFrequencyAnalysis"

    def analyze(self, circuit: QBCircuit, context: dict) -> None:
        freq: dict[str, int] = {}
        for gate in circuit.gates:
            freq[gate.name] = freq.get(gate.name, 0) + 1
        context["gate_frequencies"] = freq


class IdentityGateRemoval(TransformationPass):
    """Remove identity (id) gates from the circuit.

    Identity gates have no effect on the quantum state, but they consume
    gate time on real hardware. This pass removes them.
    """

    @property
    def name(self) -> str:
        return "IdentityGateRemoval"

    def transform(self, circuit: QBCircuit, context: dict) -> PassResult:
        non_id_gates = [g for g in circuit.gates if g.name != "id"]
        removed = len(list(circuit.gates)) - len(non_id_gates)

        if removed == 0:
            context["identity_gates_removed"] = 0
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


def main() -> None:
    # Build a circuit with some identity gates
    circ = QBCircuit(n_qubits=3, n_clbits=3, name="example_with_ids")
    circ.add_gate(QBGate("h", (0,)))
    circ.add_gate(QBGate("id", (1,)))       # Identity -- does nothing
    circ.add_gate(QBGate("cx", (0, 1)))
    circ.add_gate(QBGate("id", (2,)))       # Identity -- does nothing
    circ.add_gate(QBGate("cx", (1, 2)))
    circ.add_gate(QBGate("id", (0,)))       # Identity -- does nothing
    for i in range(3):
        circ.add_measurement(i, i)

    print(f"Original circuit: {circ.gate_count} gates")

    # Step 1: Analyze gate frequencies
    ctx: dict = {}
    analysis = GateFrequencyAnalysis()
    analysis.run(circ, ctx)
    print(f"\nGate frequencies: {ctx['gate_frequencies']}")

    # Step 2: Remove identity gates
    removal = IdentityGateRemoval()
    result = removal.run(circ, ctx)
    print(f"\nRemoved {ctx['identity_gates_removed']} identity gates")
    print(f"Circuit after cleanup: {result.circuit.gate_count} gates")

    # Step 3: Re-analyze
    analysis.run(result.circuit, ctx)
    print(f"Gate frequencies after: {ctx['gate_frequencies']}")


if __name__ == "__main__":
    main()
