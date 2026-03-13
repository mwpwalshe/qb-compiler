"""Fuzz the CLI entry points.

Targets ``qb_compiler.cli.main`` via click's CliRunner with:
- Random strings as backend names and strategy names
- Random circuit file paths
- Random subcommand arguments
"""
from __future__ import annotations

import os
import sys
import tempfile

import atheris

with atheris.instrument_imports():
    from click.testing import CliRunner

    from qb_compiler.cli.main import cli


_runner = CliRunner(mix_stderr=False)


def test_one_input(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    choice = fdp.ConsumeIntInRange(0, 3)

    try:
        if choice == 0:
            # Fuzz "info" subcommand (should always succeed)
            _runner.invoke(cli, ["info"])

        elif choice == 1:
            # Fuzz "calibration show" with random backend
            backend = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 100))
            _runner.invoke(cli, ["calibration", "show", backend])

        elif choice == 2:
            # Fuzz "compile" with a temp file containing random QASM
            qasm_content = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 2000))
            backend = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 50))
            strategy = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 30))

            # Write fuzzed QASM to a temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".qasm", delete=False
            ) as f:
                f.write(qasm_content)
                temp_path = f.name

            try:
                args = ["compile", temp_path]
                if backend:
                    args.extend(["--backend", backend])
                if strategy:
                    args.extend(["--strategy", strategy])
                _runner.invoke(cli, args)
            finally:
                os.unlink(temp_path)

        else:
            # Fuzz with completely random argv
            n_args = fdp.ConsumeIntInRange(0, 5)
            args = [
                fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 50))
                for _ in range(n_args)
            ]
            _runner.invoke(cli, args)

    except (SystemExit, Exception):
        # CLI may raise SystemExit or various exceptions; that is OK
        pass


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
