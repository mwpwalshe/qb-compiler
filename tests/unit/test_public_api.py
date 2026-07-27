"""The package root is the discoverability surface, so it is tested like one.

Names are resolved lazily (PEP 562 ``__getattr__``) rather than imported eagerly, which keeps
``import qb_compiler`` cheap and lets optional dependencies stay optional. The cost of that is a
second place to keep in step: ``_EXPORTS`` maps a public name to the module defining it, and
nothing checks that map at import time. A typo there is invisible until a user reaches for the
name, so it is checked here instead.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import qb_compiler

# Capabilities that were reachable only by full module path before, and are the reason the root
# is worth auditing: someone running dir(qb_compiler) should see what the package actually does.
HEADLINE_CAPABILITIES = [
    "audit_dem",
    "audit_matrices",
    "canonicalize_dem",
    "preflight_dem_gate",
    "ObservableAuditResult",
    "CalibrationMapper",
    "CalibrationMapperConfig",
    "selection_receipt",
    "calibration_fingerprint",
    "get_calibration_provider",
    "all_backend_statuses",
    "get_backend_status",
    "any_to_qiskit",
    "any_to_compiler_circuit",
]


class TestExportMap:
    def test_every_public_name_resolves(self) -> None:
        """The whole point of the map: no entry may be a dead reference."""
        failures = []
        for name in qb_compiler.__all__:
            try:
                getattr(qb_compiler, name)
            except Exception as exc:
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        assert not failures, "names in __all__ that do not resolve:\n" + "\n".join(failures)

    def test_all_and_export_map_agree(self) -> None:
        # passmanager is defined in __init__ itself, so it is the one name not in the map.
        mapped = set(qb_compiler._EXPORTS)
        declared = set(qb_compiler.__all__) - {"passmanager"}
        assert mapped == declared, (
            f"only in _EXPORTS: {sorted(mapped - declared)}\n"
            f"only in __all__: {sorted(declared - mapped)}"
        )

    def test_all_is_sorted_and_unique(self) -> None:
        assert qb_compiler.__all__ == sorted(qb_compiler.__all__)
        assert len(qb_compiler.__all__) == len(set(qb_compiler.__all__))

    def test_dir_matches_all(self) -> None:
        assert dir(qb_compiler) == sorted(qb_compiler.__all__)

    def test_unknown_attribute_still_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError, match="no attribute"):
            qb_compiler.does_not_exist  # noqa: B018

    @pytest.mark.parametrize("name", HEADLINE_CAPABILITIES)
    def test_headline_capability_is_importable_from_the_root(self, name: str) -> None:
        assert name in qb_compiler.__all__
        assert getattr(qb_compiler, name) is not None


class TestLazyImport:
    def test_importing_the_package_does_not_pull_numpy(self) -> None:
        """Import stays cheap, which is the reason the root is lazy at all.

        numpy arrives via qec_preflight and used to be imported at package import, costing about
        a quarter second on every CLI invocation. Run in a subprocess because numpy is certainly
        already imported inside the test session.
        """
        code = "import sys, qb_compiler; print('numpy' in sys.modules)"
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False", "importing qb_compiler eagerly pulled in numpy"

    def test_accessing_a_name_imports_its_module(self) -> None:
        code = (
            "import sys, qb_compiler\n"
            "before = 'qb_compiler.observable_gate' in sys.modules\n"
            "qb_compiler.audit_dem\n"
            "after = 'qb_compiler.observable_gate' in sys.modules\n"
            "print(before, after)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False True"

    def test_resolved_names_are_cached(self) -> None:
        # Second access must be an ordinary global lookup, not another import round trip.
        first = qb_compiler.check_viability
        assert "check_viability" in vars(qb_compiler)
        assert qb_compiler.check_viability is first
