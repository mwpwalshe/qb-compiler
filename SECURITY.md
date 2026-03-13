# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | Current release    |

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: **security@qubitboost.io**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### What to Expect

- **Acknowledgement** within 48 hours
- **Assessment** within 7 days
- **Fix timeline** communicated within 14 days
- **Credit** in the advisory (unless you prefer anonymity)

### Scope

The following are in scope:
- Code injection via malicious circuit files (QASM, Qiskit circuits)
- Deserialization vulnerabilities in calibration data parsing
- Command injection via CLI inputs
- Path traversal in file handling
- Data exfiltration via telemetry or calibration modules
- Dependency vulnerabilities

### Out of Scope

- Vulnerabilities in upstream dependencies (report to them directly)
- Social engineering attacks
- Physical attacks
- Denial of service via legitimate API usage

## Security Advisories

Published security advisories will be listed at:
https://github.com/qubitboost/qb-compiler/security/advisories
