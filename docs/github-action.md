# Preflight in CI

The repository ships a composite GitHub Action that runs the free checks against the files in a
pull request. A researcher who runs a check by hand runs it when they remember; a check that fails
a pull request runs every time.

The action does three things, and refuses to do a fourth:

| step | what it does | fails the build |
|---|---|---|
| `chem-audit` | five integrity checks on each Hamiltonian file | yes, on REFUSE, and on INCOMPLETE when `strict` is on |
| `measure-plan` | prints terms, settings and shots for each operator | no, it is a report |
| `verify-receipt` | checks receipt signatures against a key you supply | yes, on a bad signature or, with `strict`, an unsigned receipt |

It never says whether a result is worth believing. It says whether an input is well formed, what a
run costs, and whether a receipt is what it claims to be.

## Use it

```yaml
name: quantum preflight
on: [pull_request]

jobs:
  preflight:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - uses: mwpwalshe/qb-compiler@v0.12.0
        with:
          hamiltonians: "hamiltonians/*.json"
          measure-plan: "hamiltonians/*.json"
          receipts: "receipts/*.json"
          public-key: ${{ vars.QBC_PUBLIC_KEY }}
          strict: "true"
```

## Inputs

| input | default | notes |
|---|---|---|
| `hamiltonians` | empty | glob of files to audit. Empty skips the step |
| `measure-plan` | empty | glob of files to price |
| `shots-per-setting` | `4096` | the rate the bill is computed at |
| `receipts` | empty | glob of receipts to verify |
| `public-key` | empty | base64 key or a path to a key file. Without it, a signed receipt cannot be checked and the step fails |
| `strict` | `true` | undeclared fields and unsigned receipts count as failures |
| `version` | `qb-compiler` | pass a pinned specifier such as `qb-compiler==0.12.0` for a reproducible job |

The public key is not a secret. It is the thing you publish so other people can check your receipts,
so a repository variable is the right home for it, not a repository secret.

## Pinning

Pin the action to a tag and pin the package version in the same commit. A preflight that silently
changes what it accepts between two runs of the same branch is worse than no preflight, because the
build turning red stops meaning the input changed.

## Exit codes, if you would rather call the CLI directly

| command | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `qbc chem-audit` | ACCEPT | INCOMPLETE | REFUSE | file unreadable |
| `qbc verify-receipt` | verified, or unsigned without `--strict` | cannot be checked | does not verify | |
| `qbc corpus verify` | digest matches | file missing or corpus unknown | digest mismatch | |
| `qbc dem-audit` | PASS | WARN with `--strict` | FAIL | stim not installed |

## What is not in it

No verdict on a chemistry result, no drift alarm, no policy bundle, no shared history. Those are
decisions about your data rather than checks on your files, and they sit on the paid side of the
line described in [open-core.md](open-core.md).
