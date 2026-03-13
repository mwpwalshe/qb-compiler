#!/usr/bin/env bash
# Verify SLSA provenance for a qb-compiler release.
#
# Usage:
#   ./scripts/verify_provenance.sh 0.1.0
#
# Prerequisites:
#   - gh CLI authenticated
#   - pip installed

set -euo pipefail

VERSION="${1:?Usage: $0 <version>}"
PACKAGE="qb_compiler"
REPO="qubitboost/qb-compiler"
WORKDIR=$(mktemp -d)

echo "==> Verifying provenance for qb-compiler v${VERSION}"
echo "    Working directory: ${WORKDIR}"

# Download wheel from PyPI
echo "==> Downloading wheel from PyPI..."
pip download \
    --no-deps \
    --dest "${WORKDIR}" \
    "qb-compiler==${VERSION}"

WHEEL=$(find "${WORKDIR}" -name "*.whl" | head -1)
if [[ -z "${WHEEL}" ]]; then
    echo "ERROR: No wheel found for qb-compiler==${VERSION}"
    rm -rf "${WORKDIR}"
    exit 1
fi
echo "    Downloaded: $(basename "${WHEEL}")"

# Verify attestation via GitHub CLI
echo "==> Verifying build provenance attestation..."
gh attestation verify "${WHEEL}" \
    --repo "${REPO}"

RESULT=$?

# Cleanup
rm -rf "${WORKDIR}"

if [[ ${RESULT} -eq 0 ]]; then
    echo "==> Provenance verification PASSED for qb-compiler v${VERSION}"
else
    echo "==> Provenance verification FAILED for qb-compiler v${VERSION}"
    exit 1
fi
