# Release Process

## Pre-Release Checklist

Complete every item before tagging a release.

- [ ] All tests pass: `pytest --strict-markers`
- [ ] Linting clean: `ruff check src/ tests/`
- [ ] Type checking clean: `mypy src/`
- [ ] Security scan: `bandit -r src/`
- [ ] CHANGELOG.md updated with release notes under the new version heading
- [ ] Version string confirmed (hatch-vcs derives it from the git tag)
- [ ] SBOM generated: `python sbom/generate_sbom.py --version <version>`
- [ ] Benchmarks run and results reviewed: `pytest tests/benchmarks/`
- [ ] README quick-start example tested manually

## Release Steps

### 1. Prepare the release branch (if not releasing from main)

```bash
git checkout main
git pull origin main
```

### 2. Final validation

```bash
pip install -e ".[dev]"
ruff check src/ tests/
mypy src/
pytest --strict-markers --cov=qb_compiler
python sbom/generate_sbom.py --version <version>
```

### 3. Tag the release

Tags must be GPG-signed (see `docs/SIGNING.md`).

```bash
git tag -s v<version> -m "Release v<version>"
git push origin v<version>
```

### 4. CI publishes automatically

Pushing a signed tag triggers `.github/workflows/release.yml`, which:

1. Runs the full test suite.
2. Builds the sdist and wheel via `hatch build`.
3. Publishes to PyPI using trusted publishing (OIDC).
4. Generates the SBOM and attaches it to the GitHub Release.
5. Creates a GitHub Release with auto-generated notes.

### 5. Post-release verification

- [ ] Package visible on PyPI: `pip install qb-compiler==<version>`
- [ ] GitHub Release page shows the SBOM artefacts
- [ ] Docs site updated (ReadTheDocs webhook triggers on tag)

## Hotfix Process

For critical fixes against a released version:

1. Branch from the release tag:
   ```bash
   git checkout -b hotfix/v<version> v<version>
   ```
2. Apply the fix, add tests, update CHANGELOG.
3. Bump the patch version (e.g., 0.1.0 -> 0.1.1).
4. Follow the standard release steps above (tag, push, CI publishes).
5. Cherry-pick the fix into `main` if applicable:
   ```bash
   git checkout main
   git cherry-pick <hotfix-commit>
   ```

## Yanking a Release

If a release contains a critical defect:

```bash
# Remove from PyPI (users can still install pinned versions)
pip install twine
twine yank qb-compiler <version> --reason "Critical bug in ..."
```

Then publish a hotfix release immediately.
