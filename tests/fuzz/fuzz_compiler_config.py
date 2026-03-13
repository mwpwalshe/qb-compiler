"""Fuzz the compiler configuration.

Targets ``CompilerConfig`` with:
- Invalid backend names (path traversal, unicode, etc.)
- Invalid optimization levels
- Random field values
"""
from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    from qb_compiler.config import CompilerConfig
    from qb_compiler.exceptions import BackendNotSupportedError


_PATH_TRAVERSAL_NAMES = [
    "../../etc/passwd",
    "../../../etc/shadow",
    "/dev/null",
    "ibm_fez; rm -rf /",
    "ibm_fez\x00injected",
    "${PATH}",
    "$(whoami)",
    "`id`",
    "ibm_fez\nX-Injected: true",
    "<script>alert(1)</script>",
    "A" * 10000,
]


def test_one_input(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    choice = fdp.ConsumeIntInRange(0, 3)

    try:
        if choice == 0:
            # Invalid backend names including path traversal
            backend = fdp.PickValueInList(_PATH_TRAVERSAL_NAMES)
            CompilerConfig(backend=backend)

        elif choice == 1:
            # Random backend name string
            backend = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 200))
            CompilerConfig(backend=backend if backend else None)

        elif choice == 2:
            # Invalid optimization levels
            opt_level = fdp.ConsumeIntInRange(-1000, 1000)
            CompilerConfig(optimization_level=opt_level)

        else:
            # Random combination of all fields
            backend = fdp.ConsumeUnicodeNoSurrogates(fdp.ConsumeIntInRange(0, 50)) or None
            opt_level = fdp.ConsumeIntInRange(-10, 10)
            seed = fdp.ConsumeIntInRange(-2**31, 2**31) if fdp.ConsumeBool() else None
            cal_age = fdp.ConsumeRegularFloat()
            CompilerConfig(
                backend=backend,
                optimization_level=opt_level,
                calibration_max_age_hours=cal_age,
                seed=seed,
                enable_calibration_aware=fdp.ConsumeBool(),
                enable_noise_aware_scheduling=fdp.ConsumeBool(),
            )

    except (ValueError, BackendNotSupportedError, TypeError, OverflowError):
        pass


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
