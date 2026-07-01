"""Static guard: no unsafe deserialization primitives in shipped source.

Complements ``test_model_loading_rce.py`` with a cheap, dependency-free
scan of ``src/`` so that any future ``torch.load(..., weights_only=False)``,
bare ``torch.load`` (defaults are version-dependent), ``pickle.load`` or
``yaml.load`` regression fails CI immediately, even on machines without
torch installed.
"""

from __future__ import annotations

import re
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "qb_compiler"


def _python_files() -> list[Path]:
    return sorted(_SRC.rglob("*.py"))


def test_no_weights_only_false():
    """``weights_only=False`` must never appear in shipped source."""
    offenders = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        if re.search(r"weights_only\s*=\s*False", text):
            offenders.append(str(path))
    assert not offenders, f"unsafe torch.load(weights_only=False) in: {offenders}"


def test_every_torch_load_is_weights_only_true():
    """Each ``torch.load(`` call must explicitly pass weights_only=True.

    ``torch.load``'s default flipped across versions; pinning the kwarg at
    every call site removes the ambiguity for the whole supported range.
    """
    offenders = []
    pattern = re.compile(r"torch\.load\s*\(", re.MULTILINE)
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            # Grab the call up to its matching newline region (calls here are
            # single-line); assert weights_only=True is present in the slice.
            tail = text[m.start() : m.start() + 200]
            call = tail.split("\n")[0]
            # Allow multi-line by also checking the next line.
            two_lines = "\n".join(tail.split("\n")[:2])
            if "weights_only=True" not in call and "weights_only=True" not in two_lines:
                offenders.append(f"{path}: {call.strip()}")
    assert not offenders, f"torch.load without weights_only=True: {offenders}"


def test_no_pickle_or_unsafe_yaml_load():
    """No ``pickle.load`` / ``yaml.load`` (without SafeLoader) in source."""
    offenders = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bpickle\.load\b", text):
            offenders.append(f"{path}: pickle.load")
        # yaml.load without an explicit safe loader is an arbitrary-object sink
        for m in re.finditer(r"yaml\.load\s*\(([^)]*)\)", text):
            if "SafeLoader" not in m.group(1) and "Loader=" not in m.group(1):
                offenders.append(f"{path}: unsafe yaml.load")
    assert not offenders, f"unsafe deserialization sinks: {offenders}"
