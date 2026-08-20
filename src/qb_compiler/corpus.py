# SPDX-License-Identifier: Apache-2.0
"""Hash-verified loaders for public QEC datasets.

Every group that works on decoding wrangles the same handful of public releases by hand, and every
one of them writes the same paragraph in their methods section about which files they used. This
module holds a pinned manifest: for each dataset, the archive's sha256, its size, where the
publisher put it, and the citation the publisher asks for. So a reader of your paper can tell that
the bytes you decoded are the bytes the release contains.

**Nothing is redistributed.** These are other people's datasets, published under their own terms.
The loader tells you where the publisher put a file, checks the copy you fetched, and prints the
citation they ask for. It never mirrors, repackages, or serves the data.

**Only other people's public releases are listed here.** Our own calibration series is not a public
dataset and does not belong in a public manifest, whatever else it is useful for.

Usage::

    from qb_compiler.corpus import list_corpora, verify_corpus_file, corpus_citation

    for entry in list_corpora():
        print(entry.name, entry.n_bytes, entry.doi)

    result = verify_corpus_file("willow-105q-d3-d5-d7", "google_105Q_surface_code_d3_d5_d7.zip")
    print(result.status, result.detail)

A mismatch is a hard failure with the two digests printed. A truncated download and an edited file
look identical to every other tool in the stack, and both silently change a decoder benchmark.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "qb.corpus_manifest.v1"

_MANIFEST_PATH = Path(__file__).resolve().parent / "data" / "corpus_manifest.json"

# qbcal-2026-02 is being archived and does not have an archive record yet. When it lands, add a
# manifest entry with its real digest and DOI in one pass: grep DOI_QBCAL_2026_02_PENDING.
# Nothing in this module references it until then, because a manifest entry without a real digest
# is exactly the kind of number this package refuses to print.
DOI_QBCAL_2026_02_PENDING = None

_CHUNK = 8 * 1024 * 1024

VERIFIED = "VERIFIED"
MISMATCH = "MISMATCH"
MISSING = "MISSING"
UNKNOWN_CORPUS = "UNKNOWN_CORPUS"


class CorpusError(KeyError):
    """Raised when a corpus name is not in the manifest."""


@dataclass(frozen=True)
class CorpusEntry:
    """One dataset in the manifest.

    Attributes
    ----------
    name :
        Short handle used on the command line.
    description :
        One line on what the dataset holds.
    publisher :
        Who released it. Not us, in every case, by design.
    doi, url :
        Where the publisher put it. ``url`` is the exact file the digest belongs to.
    filename :
        The archive's own name at the publisher.
    sha256, n_bytes :
        Digest and size of that archive, computed on a copy fetched from ``url``.
    fetched :
        The date our copy was fetched. A publisher can re-release under the same DOI, so a digest
        is a statement about a moment, and this is that moment.
    citation :
        What the publisher asks to be cited.
    licence :
        The publisher's terms, as they state them.
    """

    name: str
    description: str
    publisher: str
    doi: str
    url: str
    filename: str
    sha256: str
    n_bytes: int
    fetched: str
    citation: str
    licence: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "publisher": self.publisher,
            "doi": self.doi,
            "url": self.url,
            "filename": self.filename,
            "sha256": self.sha256,
            "n_bytes": self.n_bytes,
            "fetched": self.fetched,
            "citation": self.citation,
            "licence": self.licence,
        }


@dataclass(frozen=True)
class CorpusVerification:
    """Outcome of checking one file against the manifest."""

    status: str
    ok: bool
    detail: str
    name: str
    path: str | None = None
    expected_sha256: str | None = None
    actual_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "qb.corpus_verification.v1",
            "status": self.status,
            "ok": self.ok,
            "detail": self.detail,
            "corpus": self.name,
            "path": self.path,
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
        }

    def __str__(self) -> str:
        return f"{self.status}: {self.detail}"


def _load_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    return manifest


def list_corpora() -> list[CorpusEntry]:
    """Every dataset in the manifest, ordered by name."""
    manifest = _load_manifest()
    return [CorpusEntry(**entry) for entry in sorted(manifest["corpora"], key=lambda e: e["name"])]


def get_corpus(name: str) -> CorpusEntry:
    """One dataset by name. Raises :class:`CorpusError` when the name is not listed."""
    for entry in list_corpora():
        if entry.name == name:
            return entry
    known = ", ".join(e.name for e in list_corpora())
    raise CorpusError(f"no corpus named {name!r}. Known: {known}")


def corpus_citation(name: str) -> str:
    """The citation the publisher of *name* asks for."""
    return get_corpus(name).citation


def file_sha256(path: str | Path, *, chunk: int = _CHUNK) -> str:
    """sha256 of a file, read in chunks so a multi-gigabyte archive does not go into memory."""
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def verify_corpus_file(name: str, path: str | Path) -> CorpusVerification:
    """Check the file at *path* against the manifest entry for *name*.

    Returns a verdict rather than raising, so a caller can report several files in one pass. ``ok``
    is true only when the digest matches exactly.
    """
    try:
        entry = get_corpus(name)
    except CorpusError as exc:
        return CorpusVerification(
            status=UNKNOWN_CORPUS,
            ok=False,
            detail=str(exc),
            name=name,
            path=str(path),
        )

    file_path = Path(path).expanduser()
    if file_path.is_dir():
        # A directory is a natural thing to point at; look for the archive by its published name.
        candidate = file_path / entry.filename
        if candidate.is_file():
            file_path = candidate
        else:
            return CorpusVerification(
                status=MISSING,
                ok=False,
                detail=(
                    f"{file_path} is a directory and does not contain {entry.filename}. "
                    f"Fetch it from {entry.url} and point this at the file or the directory "
                    f"holding it; nothing is mirrored by this package."
                ),
                name=name,
                path=str(file_path),
                expected_sha256=entry.sha256,
            )
    if not file_path.exists():
        return CorpusVerification(
            status=MISSING,
            ok=False,
            detail=(
                f"{file_path} does not exist. Fetch {entry.filename} from {entry.url} "
                f"and point this at it; nothing is mirrored by this package."
            ),
            name=name,
            path=str(file_path),
            expected_sha256=entry.sha256,
        )

    actual = file_sha256(file_path)
    if actual == entry.sha256:
        size = file_path.stat().st_size
        return CorpusVerification(
            status=VERIFIED,
            ok=True,
            detail=(
                f"{file_path.name} matches the pinned digest for {name} "
                f"({size:,} bytes). Cite: {entry.citation}"
            ),
            name=name,
            path=str(file_path),
            expected_sha256=entry.sha256,
            actual_sha256=actual,
        )

    size = file_path.stat().st_size
    size_note = (
        f"; the file is {size:,} bytes against an expected {entry.n_bytes:,}, "
        "which usually means a truncated download"
        if size != entry.n_bytes
        else "; the size matches, so the contents differ rather than the transfer failing"
    )
    return CorpusVerification(
        status=MISMATCH,
        ok=False,
        detail=(
            f"{file_path.name} does not match the pinned digest for {name}{size_note}. "
            f"Do not benchmark against it without settling why."
        ),
        name=name,
        path=str(file_path),
        expected_sha256=entry.sha256,
        actual_sha256=actual,
    )


def load_corpus(name: str, path: str | Path, *, verify: bool = True) -> Path:
    """Resolve a local copy of corpus *name*, refusing a file whose digest does not match.

    Returns the path so a caller can chain straight into their own reader. Raises ``ValueError``
    on a mismatch, because a decoder benchmark run against unverified bytes is worse than one that
    did not run.
    """
    result = verify_corpus_file(name, path) if verify else None
    if result is not None and not result.ok:
        raise ValueError(result.detail)
    return Path(path).expanduser()
