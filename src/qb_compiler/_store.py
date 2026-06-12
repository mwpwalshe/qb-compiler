"""Local on-disk store for receipts and verification records.

Everything stays on the user's machine (no transmission, no telemetry): a plain JSONL
append log under QBC_DATA_DIR (default ~/.qb_compiler). Shared by the receipts,
regression-watch, and verify modules.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def data_dir() -> Path:
    root = Path(os.environ.get("QBC_DATA_DIR", Path.home() / ".qb_compiler"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def append_jsonl(name: str, record: dict[str, Any]) -> Path:
    path = data_dir() / name
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return path


def read_jsonl(name: str) -> list[dict[str, Any]]:
    path = data_dir() / name
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    skipped = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        import logging

        logging.getLogger(__name__).warning(
            "%s: skipped %d unparseable line(s)", path.name, skipped
        )
    return out
