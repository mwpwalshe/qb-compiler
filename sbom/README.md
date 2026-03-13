# Software Bill of Materials (SBOM)

An SBOM is a machine-readable inventory of all components, libraries, and dependencies shipped with a software product. SBOMs enable downstream consumers to audit supply-chain risk, check for known vulnerabilities (CVEs), and verify licence compliance.

qb-compiler produces SBOMs in **CycloneDX JSON** format (OWASP standard) and a human-readable Markdown dependency list.

## Generating an SBOM

Prerequisites:

```bash
pip install cyclonedx-bom pip-licenses
```

Generate both artefacts for a given release version:

```bash
python sbom/generate_sbom.py --version 0.1.0
```

This produces two files in `sbom/`:

| File | Format | Purpose |
|------|--------|---------|
| `qb-compiler-<version>-cyclonedx.json` | CycloneDX JSON | Machine-readable SBOM for vulnerability scanners |
| `qb-compiler-<version>-dependencies.md` | Markdown | Human-readable dependency and licence listing |

## Verifying an SBOM

Use any CycloneDX-compatible tool to validate or scan the JSON SBOM:

```bash
# Validate schema
cyclonedx validate --input-file sbom/qb-compiler-0.1.0-cyclonedx.json

# Scan for known vulnerabilities (requires grype)
grype sbom:sbom/qb-compiler-0.1.0-cyclonedx.json
```

## CI Integration

The release workflow automatically generates and attaches SBOMs to each GitHub Release. See `docs/RELEASE.md` for the full process.
