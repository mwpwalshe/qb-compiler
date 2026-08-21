# qb-compiler 0.12.0

Receipts from 0.9.0 to 0.11.0: read FIXED before you rely on one.

## FIXED

```
- selection_receipt   now records the executed layout. previously recorded the
                      recommendation regardless of what ran. found on a live
                      campaign: the run executed [6, 5, 4, 3, 2], the receipt
                      said {144, 143, 136, 123, 124}, nothing warned.
                      pass executed_layout= and the receipt names what ran,
                      keeps the recommendation in recommended_layout, and
                      records score_penalty_vs_recommended.
- signing             one persistent ed25519 key at ~/.qb-compiler/signing_key
                      (mode 0600, or QBC_SIGNING_KEY), fingerprint in the
                      receipt. previously sign=True minted a fresh keypair per
                      call and embedded its public half in the receipt it had
                      just signed, which proves nothing about origin. those
                      receipts cannot be repaired; re-issue anything load
                      bearing. sign=True now raises when it cannot sign.
- verifier            qbc verify-receipt <file> --key <public-key>. verdicts:
                      VERIFIED, UNSIGNED, NO_KEY, INVALID_SIGNATURE,
                      LEGACY_SELF_SIGNED, MALFORMED. only VERIFIED passes; a
                      key carried inside the receipt is never trusted. offline,
                      no account, pure python ed25519 fallback so a bare
                      pip install qb-compiler can check anyone's receipt.
- qec_preflight       new matching_faithfulness field, always "not_assessed".
                      the projected LER is decoded with PyMatching and nothing
                      here checks whether matching is faithful on your error
                      model. the gap is named, not closed.
```

## SCHEMA (selection receipt, still qb.selection_receipt.v1, additions additive)

```
- describes_executed_layout   on every receipt. true when the recommendation ran.
- recommended_layout, recommended_score, executed_layout_matches_recommendation,
  score_penalty_vs_recommended, divergence_note
                              only when executed_layout= is passed.
- score_breakdown             emitted empty with a note when executed differs
                              from recommended: the breakdown describes the
                              recommendation, and numbers next to a layout they
                              do not describe is how this went wrong.
- old receipts parse. a consumer reading selected_layout keeps working.
```

## NEW (all free)

```
- qbc chem-audit      five integrity checks on a qubit hamiltonian file:
                      correlated reference method, exact energy below RHF,
                      provenance chain, no synthetic flag, qubit count matches
                      the active space. ACCEPT / INCOMPLETE / REFUSE; --strict
                      turns INCOMPLETE into a refusal. reads grouped JSON, flat
                      term lists, OpenFermion, Qiskit SparsePauliOp. the last
                      check exists because a generator once wrote n_qubits =
                      n_orbitals and an 8e8o space shipped an 8 qubit operator.
- qbc measure-plan    measurable terms, QWC settings, grouping factor, largest
                      group, shots at your rate. structural counts only: no
                      variance weighting, no allocation, no precision claim.
                      across the 14 file corpus this was built against the
                      setting count spans 1.79x at 12 qubits and 1.59x at 16;
                      qubit count does not predict where a molecule lands.
- qbc corpus          public QEC datasets with pinned sha256, publisher URL,
                      DOI and requested citation. verifies your copy is byte
                      for byte the published one. nothing mirrored. in the
                      manifest: google willow 105q, quera surface code.
                      qbcal-2026-02 listed as pending with no digest, because
                      a row that looks checkable and is not is worse than none.
- rank_layouts        CalibrationMapper.rank_layouts(circuit, top_k=) and
                      score_layout(layout, circuit), public API. diversify
                      groups by qubit-index centroid (a locality proxy, only as
                      good as the vendor numbering); max_overlap compares qubit
                      sets exactly and drops the near-duplicate that is the
                      same hardware with the labels permuted.
- calibration age     calibration_freshness on every selection receipt, with
                      timestamp_status: measured, absent, unreadable, synthetic
                      or clock-skewed. tolerance_basis states the 30 minute
                      default is a builtin, not a measurement of your device.
                      reads ISO strings, datetimes and epochs; the earlier
                      draft silently dropped string timestamps.
- github action       action.yml runs chem-audit, measure-plan and
                      verify-receipt over PR files, fails on malformed input or
                      bad signature. free checks only. docs/github-action.md.
- citation            CITATION.cff plus a JOSS paper draft. DOI not minted yet.
```

## HYGIENE

```
- coverage badge      names the enforced floor and links to where it is set.
                      previously claimed a hardcoded 60 percent behind an empty
                      link. docs badge points at the docs site.
- qbc doctor          detects qiskit 1.x + qiskit-ibm-runtime >= 0.40, where a
                      backend advertises a translation plugin qiskit does not
                      ship and plain transpile raises TranspilerError. prints
                      the workaround. README compatibility table says the same.
- receipt string      an unsigned receipt says "unsigned". the stray SDK pitch
                      is gone.
- repo                issue template config points at Discussions and the
                      security policy. pull request template added.
```

## UPGRADING

```
- nothing that reads a receipt breaks.
- signed receipts from 0.9.0 to 0.11.0 that are load bearing: re-issue them.
- if you ever run a layout other than the recommendation: pass executed_layout=.
```
