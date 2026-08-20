# qb-compiler 0.12.0

Two things in here are fixes to receipts, and both matter more than the new features, so they go
first. If you have receipts from 0.9.0 through 0.11.0 sitting anywhere, read the first two
sections before you rely on them.

## Receipts now describe the layout that ran

Up to 0.11.0, `selection_receipt()` always described the layout the mapper **recommended**, whatever
the caller actually executed. Override the layout, which is a normal thing to do, and you got a
receipt naming a mapping that never ran. It was found on a live campaign here: the run executed
`[6, 5, 4, 3, 2]` and the receipt said `{144, 143, 136, 123, 124}`. Nothing warned about it, and the
receipt looked fine.

From 0.12.0 you can pass what you ran:

```python
receipt = selection_receipt(
    result,
    calibration=backend,
    executed_layout={0: 6, 1: 5, 2: 4, 3: 3, 4: 2},
    scorer=functools.partial(mapper.score_layout, circuit=circuit),
)
```

The receipt then names the executed layout, keeps the recommendation in `recommended_layout`, and
records `score_penalty_vs_recommended`, which is what the override cost on the mapper's own scale.

**Schema, for anyone reading receipts programmatically.** Still `qb.selection_receipt.v1`, and the
additions are additive.

- `describes_executed_layout` is now on **every** receipt, including ones with no `executed_layout`
  passed, where it is `true` because the recommendation is what ran.
- `recommended_layout`, `recommended_score`, `executed_layout_matches_recommendation`,
  `score_penalty_vs_recommended` and `divergence_note` appear **only** when you pass
  `executed_layout`.
- When the executed layout differs from the recommendation, `score_breakdown` is emitted empty with
  a `score_breakdown_note` saying why. The breakdown is computed for the recommendation, and leaving
  numbers that describe a different mapping next to the layout that ran is how this went wrong in
  the first place.

Old receipts still parse. A consumer that only reads `selected_layout` keeps working.

## Signing before 0.12.0 proved nothing, and is replaced

`sign=True` in earlier releases generated a **fresh keypair on every call** and embedded that
keypair's public half in the receipt it had just signed. Every one of those receipts verifies, and
none of them says anything about who produced it: the key came with the signature, so anybody could
have made both. **Do not treat a receipt with a `public_key` field in it as evidence of origin.**
There is no way to recover origin from such a receipt after the fact, because the private half was
discarded at the end of the call that made it.

What replaces it:

- One Ed25519 key, created once at `~/.qb-compiler/signing_key` with mode 0600, or wherever
  `QBC_SIGNING_KEY` points, and reused for every receipt after that.
- The receipt carries the key's **fingerprint**, not the key.
- `sign=True` now raises if it cannot sign, rather than handing back an unsigned receipt with a
  note.
- The verifier reports old-style receipts as `LEGACY_SELF_SIGNED` and refuses them. It never trusts
  a key carried inside the thing it is checking.

If you signed anything with an older release and it matters, re-issue it.

## The verifier is free, offline, and needs nothing from us

```bash
qbc verify-receipt receipt.json --key their-public-key.txt
```

Verdicts are `VERIFIED`, `UNSIGNED`, `NO_KEY`, `INVALID_SIGNATURE`, `LEGACY_SELF_SIGNED` and
`MALFORMED`, and only the first is a pass. "I could not check it" never reads as "fine". The output
also prints what the receipt does **not** claim, because a signature says these bytes came from that
key holder and nothing about whether the numbers are right.

It works on any receipt this package emits, needs no account and no network, and falls back to a
pure Python Ed25519 implementation when a compiled one is not installed, so a plain
`pip install qb-compiler` is enough to check somebody else's receipt.

## qec_preflight says out loud what it did not check

The projected logical error rate is decoded with PyMatching, and this package has never checked
whether a matching-based decode is faithful on the error model it was handed. It now says so, in a
`matching_faithfulness` field that always reads `not_assessed`, in the printed output, and in the
notes. No new check was added. The gap is named rather than closed, which is the honest version.

## New, all free

**`qbc chem-audit`** runs five integrity checks over a qubit Hamiltonian file: correlated reference
method, exact energy below RHF, a provenance chain that reaches a correlated solve and a qubit
mapping, no synthetic flag, and a qubit count that matches the declared active space. Three verdicts:
ACCEPT, REFUSE, and INCOMPLETE for a file that did not declare enough to check. `--strict` turns
INCOMPLETE into a refusal. Reads grouped JSON, flat term lists, OpenFermion operators and Qiskit
`SparsePauliOp`.

The last check exists because a generator once wrote `n_qubits = n_orbitals` instead of
`2 * n_orbitals`, so a file describing an 8 electron, 8 orbital active space carried an 8 qubit
operator where the space needs 16. Nothing downstream noticed.

**`qbc measure-plan`** prices measuring an operator: measurable terms, qubit-wise commuting
settings, grouping factor, largest group, and shots at your chosen rate. It is a structural count.
No variance weighting, no shot allocation, no precision claim. Measured across the 14 file corpus
this was built against, the setting count spans 1.79 times at 12 qubits (99 to 177, three files) and
1.59 times at 16 (1196 to 1896, nine files), and qubit count does not predict where a molecule lands
in that range.

**`qbc corpus`** lists public QEC datasets with a pinned sha256, the publisher's URL and DOI, and
the citation they ask for, and checks whether your copy is byte for byte the published one. Nothing
is mirrored or redistributed: fetch from the publisher, then verify. Google's 105 qubit surface code
release and QuEra's surface code dataset are in the manifest.

**qbcal-2026-02** is being archived and is named in the manifest as pending, with no digest and no
DOI, because a row that looks checkable and is not is worse than no row. Archive record and DOI to
follow.

**Calibration age rides on every selection receipt.** `calibration_freshness` reports the measured
age of the calibration a layout was chosen from, and states in `tolerance_basis` on every receipt
that the 30 minute tolerance it compares against is a builtin default and not a measurement of your
device. `timestamp_status` distinguishes a measured age from an absent, unreadable, synthetic or
clock-skewed one. It reads ISO strings, datetimes and epoch numbers, where the earlier draft only
handled a datetime and silently reported nothing for the string timestamps most snapshots carry.

**Ranked layouts are public.** `CalibrationMapper.rank_layouts(circuit, top_k=...)` returns
candidates with their scores, and `CalibrationMapper.score_layout(layout, circuit)` scores a layout
of your own on the same scale. Reaching into `_find_top_k_layouts` is no longer necessary.

There are two diversity controls and the docstring is blunt about what each one does. `diversify`
groups candidates by the centroid of their physical qubit **indices**, which is a proxy for chip
locality and only as good as the vendor's numbering. `max_overlap` compares the **sets** of physical
qubits directly, which is exact, and it is the one that removes the near-duplicate at the top of a
ranking: the same hardware with the logical labels permuted, which is one option presented as two.

**A GitHub Action.** `action.yml` runs `chem-audit`, `measure-plan` and `verify-receipt` over the
files in a pull request and fails the build on a malformed input or a bad signature. It runs free
checks only. See `docs/github-action.md`.

**`CITATION.cff`, plus a JOSS paper draft** (`paper.md`, `paper.bib`), so the tool can be cited.
The DOI is not minted yet; that is a release action.

## Hygiene

- The coverage badge no longer claims a hardcoded 60 percent behind an empty link; it names the
  enforced floor and links to where the floor is set. The docs badge points at the docs site rather
  than back at the README.
- `qbc doctor` now names the qiskit 1.x plus qiskit-ibm-runtime 0.40 or newer pairing, where a
  backend advertises a translation plugin the installed qiskit does not ship and a plain transpile
  raises `TranspilerError: Invalid plugin name`. It prints the workaround. The README compatibility
  table says the same thing.
- The unsigned receipt string no longer carries a stray SDK pitch; an unsigned receipt just says
  `unsigned`.
- Issue template config points at Discussions and the security policy; a pull request template
  landed.

## Upgrading

Nothing that reads a receipt breaks. The two things to do:

1. If you signed receipts with 0.9.0 to 0.11.0 and they are load bearing, re-issue them.
2. If you ever run a layout other than the recommendation, start passing `executed_layout=`.
