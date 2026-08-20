# What this changes

<!-- One or two sentences. What is different after this lands, from a user's point of view. -->

## Why

<!-- The problem, ideally with the case that hit it. Link an issue or discussion if there is one. -->

## Checks

- [ ] `ruff check src/ tests/` and `ruff format --check src/ tests/` pass
- [ ] `mypy src/qb_compiler/` passes
- [ ] `pytest` passes locally
- [ ] New behaviour has a test, and a bug fix has a test that fails without the fix
- [ ] `CHANGELOG.md` has an entry if anything user visible changed
- [ ] Docs or docstrings updated if the public surface moved

## Claims

<!--
If this adds or changes a number the tool reports, say where the number comes from and what it does
not claim. A measured result and a modelled estimate are different things and the output has to say
which it is.
-->

## Compatibility

- [ ] No breaking change, or a deprecation path is included
- [ ] Optional dependencies stay optional
