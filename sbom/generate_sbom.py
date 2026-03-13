#!/usr/bin/env python3
"""Generate Software Bill of Materials for qb-compiler.

Usage:
    python sbom/generate_sbom.py --version 0.1.0
"""
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def generate_cyclonedx(version: str) -> None:
    output_path = Path(f"sbom/qb-compiler-{version}-cyclonedx.json")
    try:
        subprocess.run([
            "cyclonedx-py", "environment",
            "--output", str(output_path),
            "--format", "json",
        ], check=True)
        print(f"CycloneDX SBOM written to {output_path}")
    except FileNotFoundError:
        print("cyclonedx-bom not installed. Run: pip install cyclonedx-bom")
        sys.exit(1)


def generate_dependency_list(version: str) -> None:
    output_path = Path(f"sbom/qb-compiler-{version}-dependencies.md")
    result = subprocess.run(
        ["pip-licenses", "--from=mixed", "--format=markdown",
         "--with-urls"],
        capture_output=True, text=True
    )
    header = f"# qb-compiler {version} — Dependencies\n\nGenerated: {datetime.now(timezone.utc).isoformat()}\n\n## Direct and Transitive Dependencies\n\n"
    output_path.write_text(header + result.stdout)
    print(f"Dependency list written to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    Path("sbom").mkdir(exist_ok=True)
    generate_cyclonedx(args.version)
    generate_dependency_list(args.version)
