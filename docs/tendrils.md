# Tendrils

A tendril is one place an experiment can go, or one format or toolkit it can come from. qb-compiler
lists every one it can reach, and each one declares the same things: what it supports, where its
prices come from, where its calibration comes from, and how far its numbers have been checked
against real hardware.

```bash
qbc tendrils                         # every tendril and its label
qbc tendrils --json                  # the same, as JSON
qbc tendril check ibm                # the conformance check on one tendril
qbc tendril check --file mine.json   # the check on a declaration that is not installed
```

## Labels

The same four labels `qbc when` prints, with the same meaning.

| label | means |
|---|---|
| `validated` | the adapter ran a real job on that vendor's hardware, and the record of that job is named: date, backend, job id, sha256 of the result |
| `UNVALIDATED` | an adapter exists and has not been run against real hardware |
| `fixture-only` | exercised on recorded fixtures only, never against a live vendor |
| `no-adapter` | listed so the gap is visible; nothing runs |

A validation record covers 90 days from its date. After that the label stays as declared and the
listing shows the record as `expired`, so both facts are in front of the reader. IBM today:

```
NAME         KIND      LABEL         RECORD   COVERS TO   ORIGIN
ibm          hardware  validated     expired  2026-06-12  builtin
```

The IBM record is job `d6qknvfr88ds73dch6r0` on ibm_fez, committed on 2026-03-14 as
`results/hardware_validation_dd.json`. A test checks the sha256 the tendril names against that
file.

## What a tendril declares

| field | |
|---|---|
| `name` | lower case, digits, `.`, `_`, `-`, 1 to 64 characters |
| `kind` | `hardware`, `format` or `toolkit` |
| `vendor`, `maintainer`, `contact`, `adapter_version` | who answers for it |
| `capabilities` | every one of `circuits`, `qec_results`, `price`, `calibration`, `execute`, each true or false. Leaving one out fails the check: unsupported is declared, never silent |
| `label` | one of the four above |
| `backends`, `formats` | what it reaches. A hardware tendril names at least one backend |
| `calibration` | `live`, `snapshot` (with `calibration_as_of`) or `none` |
| `price_rows` | backend, unit (`per_shot`, `per_task`, `per_second`, `per_minute`, `formula`, `subscription`), amount, currency, the date it was read and the source. A `formula` row states its formula in `note` |
| `no_public_price` | true when a hardware tendril has no public price, so the gap is declared |
| `validation` | the record a `validated` label rests on |

`execute` is declared like the rest. qb-compiler itself never submits a job.

## The conformance check

`qbc tendril check` runs these, by name, and says what each found:

- **identity**: name, kind, who answers for it, and backends or formats for the kinds that need them
- **capabilities**: all five declared, nothing unknown
- **label**: a known label, and for `validated` a record with a real date, a job id, a 64-digit
  sha256, a backend the tendril lists and a stated location for the result
- **price**: every row with a known unit, a finite amount of zero or more, a currency code, a read
  date that is not in the future, and a source; or a declared absence of public prices
- **calibration**: the kind matches the capability, and a snapshot names its date
- **validation_window**: reports whether a record covers today. It never fails
- **round_trip** (built-in formats): a three-qubit GHZ circuit goes through the format and back
  and must come back with the same gates and measurements. Skipped, and said so, when the format's
  package is not installed

Exit 0 means the declaration conforms, 2 that it does not, 3 that the check could not run.

The check is about the declaration. It does not say the hardware is any good, and a price is only
as current as the date it was read.

## Adding a tendril

No change to qb-compiler is needed. Publish a package that registers an entry point in the
`qb_compiler.tendrils` group, pointing at a `Tendril`, a dict in the same shape as
`qbc tendrils --json` prints, or a function that returns either:

```toml
[project.entry-points."qb_compiler.tendrils"]
acme = "acme_qbc:TENDRIL"
```

```python
TENDRIL = {
    "name": "acme",
    "kind": "hardware",
    "vendor": "Acme Quantum",
    "maintainer": "Acme",
    "contact": "https://acme.example/support",
    "adapter_version": "1.0.0",
    "capabilities": {"circuits": True, "qec_results": False, "price": True,
                     "calibration": False, "execute": False},
    "label": "UNVALIDATED",
    "backends": ["acme_one"],
    "price_rows": [{"backend": "acme_one", "unit": "per_shot", "amount": 0.001,
                    "currency": "USD", "as_of": "2026-11-01",
                    "source": "https://acme.example/pricing"}],
}
```

Then sign a receipt with your own key, not ours:

```bash
qbc tendril check acme --sign --key ~/.acme/signing_key --out acme.receipt.json
```

and open a pull request adding one row to `TENDRILS.json` with `origin` set to
`plugin:<your package>` and the receipt's sha256. Your code stays in your package. Anyone can check
the receipt offline with `qbc verify-receipt acme.receipt.json --key acme-public-key.txt`.

A plugin that fails to import, or declares a name already taken, is listed by `qbc tendrils` as
broken with the reason, and never stops the others loading.

## Receipt

Schema `qb.tendril_conformance.v1`: the tendril's name and origin, the verdict, every check with
its detail, the full declaration, its sha256, the time it was checked and the qb-compiler version.
It claims which checks the declaration passed. It does not claim the adapter works on hardware
beyond any record it names, or that a price is current after the date it was read.
