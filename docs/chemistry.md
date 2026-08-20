# Chemistry inputs: is the file sound, and what will measuring it cost

Two commands, both of which run in seconds on a laptop and neither of which needs hardware.

`qbc chem-audit` reads a qubit Hamiltonian and refuses it when its own declared metadata
contradicts itself. `qbc measure-plan` counts what measuring that operator costs.

Neither computes a chemistry result. They tell you whether the input is well formed and what the run
costs. Whether the result is worth believing is a different question and this package does not
answer it.

## qbc chem-audit

The file below ships with the repository, so this output is reproducible:

```bash
$ qbc chem-audit tests/fixtures/hamiltonians/h2_2e2o.json
ACCEPT: tests/fixtures/hamiltonians/h2_2e2o.json
  [        pass] reference method is correlated
                 reference_energy_type = 'FCI_sparse', names FCI
  [        pass] exact energy sits below RHF
                 exact -1.136654584383908 vs RHF -1.1174874250696742, correlation -0.019167 Ha
  [        pass] provenance chain reaches a correlated solve
                 provenance 'RDKit -> PySCF -> OpenFermion -> JW -> sparse_FCI' names FCI and the JW mapping
  [        pass] not flagged synthetic
                 metadata.synthetic = False
  [        pass] qubit count matches the active space
                 n_qubits 4 = 2 x 2 orbitals
```

### The five checks

| check | fails when |
|---|---|
| reference method is correlated | `reference_energy_type` names no post Hartree Fock method. A mean field reference measures the basis set, not the algorithm |
| exact energy sits below RHF | the reference energy is not below the mean field one. Equality means no correlation energy is present, which is what a placeholder record looks like |
| provenance chain reaches a correlated solve | the declared chain names no correlated solve, or no qubit mapping. Without a mapping stage, a fermionic operator never became a qubit operator |
| not flagged synthetic | the file flags itself synthetic |
| qubit count matches the active space | `n_qubits` is not `2 x n_orbitals` under a mapping that costs one qubit per spin orbital, or a term reaches past the declared width |

The fifth one is the reason the tool exists. A generator wrote `n_qubits = n_active_orbitals`
instead of `2 * n_active_orbitals`, so a file describing an 8 electron, 8 orbital active space
carried an 8 qubit operator where the space needs 16. Nothing downstream noticed, because nothing
downstream checks, and the filename said the file was real the whole time.

### Three verdicts, not two

| verdict | exit | means |
|---|---|---|
| `ACCEPT` | 0 | every check passed |
| `INCOMPLETE` | 1 | nothing failed, but the file did not declare enough to check |
| `REFUSE` | 2 | at least one check failed |

`INCOMPLETE` is a real answer. A file from somebody else's pipeline usually declares less than ours
does. Calling that a failure would make this a tool for our own files; calling it a pass would be an
assurance nobody measured. `--strict` turns undeclared into a refusal, which is what CI should use.

### File formats it reads

Deliberately generic, because a validator that only reads one project's layout is that project's
internal script.

- grouped JSON, the layout used by the QubitBoost chemistry corpus (`pauli_groups`)
- a flat term list: `{"n_qubits": 4, "terms": [{"pauli": "Z0 Z1", "coefficient": 0.17}]}`, also
  accepting `[["Z0 Z1", 0.17]]` and `{"Z0 Z1": 0.17}`
- OpenFermion `QubitOperator`, or any record whose `terms` are keyed by `((index, letter), ...)`
- Qiskit `SparsePauliOp`, with a metadata block passed alongside

Pauli strings are read sparse (`"Z0 X3"`) or dense (`"ZIIX"`, left to right from qubit 0).

```python
from qb_compiler.chem import audit_hamiltonian, load_hamiltonian

verdict = audit_hamiltonian(load_hamiltonian("your_hamiltonian.json"), strict=True)
print(verdict.verdict, verdict.failures())
```

## qbc measure-plan

```bash
$ qbc measure-plan beh2_6e6o.json --shots-per-setting 4096
Measurement plan: beh2_6e6o.json
  qubits              : 12
  measurable terms    : 366 of 367
  QWC settings        : 99 (grouping factor 3.70x)
  largest group       : 78 terms
  shots per setting   : 4096
  shots, grouped      : 405,504
  shots, term by term : 1,499,136 (1,093,632 more than grouped)
  identity offset     : -9.534814
```

The identity term carries the nuclear and frozen core energy. It is a constant, so it is excluded
from the bill rather than paid for.

### What the number is, and what it is not

It is a **structural** count: how many measurement settings the operator's own commutation structure
implies, and what that costs at your chosen rate. It carries no variance weighting and no precision
claim. Two operators with identical setting counts can need very different budgets to reach the same
error bar, because that depends on coefficients and on state dependent variance, neither of which is
visible here.

The grouping is textbook greedy qubit-wise commuting. Better groupers exist and finding the minimum
is NP-hard, so a good grouper returns fewer settings than this. That makes the number an upper bound
on what a good grouper needs, which is a fair basis for a budget and a poor basis for a claim about
grouping quality. This package does not compete on grouping quality.

### What the spread looks like on a real corpus

Measured across the 14 file corpus this was built against: at 12 qubits the setting count spans
1.79 times between molecules (99 to 177, three files), and at 16 qubits 1.59 times (1196 to 1896,
nine files). Qubit count does not predict where a molecule lands in that range, so two jobs of the
same width on your desk can differ by nearly a factor of two in what they cost to measure.

```python
from qb_compiler.chem import load_hamiltonian, measurement_plan

plan = measurement_plan(load_hamiltonian("beh2_6e6o.json"), shots_per_setting=4096)
print(plan.n_settings_qwc, plan.shots_qwc)
```

## Both in CI

```bash
qbc chem-audit hamiltonians/*.json --strict
```

or use the action described in [github-action.md](github-action.md), which runs the audit, prints
the bill, and fails the build on a malformed input.
