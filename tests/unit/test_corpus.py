# SPDX-License-Identifier: Apache-2.0
"""The public corpus manifest and its hash check.

The manifest is a claim about other people's files, so the tests check the claim is well formed
(no placeholder digests, real DOIs, sizes present) and that verification is strict: a byte out of
place is a hard refusal, not a warning.
"""

from __future__ import annotations

import json
import re

import pytest

from qb_compiler.corpus import (
    MISMATCH,
    MISSING,
    UNKNOWN_CORPUS,
    VERIFIED,
    CorpusError,
    corpus_citation,
    file_sha256,
    get_corpus,
    list_corpora,
    load_corpus,
    verify_corpus_file,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TestManifest:
    def test_manifest_is_not_empty(self):
        assert list_corpora()

    def test_entries_are_complete_and_plausible(self):
        for entry in list_corpora():
            assert _SHA256.match(entry.sha256), f"{entry.name} has no real digest"
            assert entry.n_bytes > 0
            assert entry.url.startswith("https://")
            assert entry.doi.startswith("10.")
            assert entry.citation
            assert entry.fetched

    def test_no_entry_is_ours(self):
        """Only other people's public releases belong in a public manifest."""
        for entry in list_corpora():
            assert "qubitboost" not in entry.publisher.lower()
            assert "qubitboost" not in entry.url.lower()

    def test_lookup_by_name(self):
        entry = list_corpora()[0]
        assert get_corpus(entry.name) == entry
        assert corpus_citation(entry.name) == entry.citation

    def test_unknown_name_raises_and_lists_the_known_ones(self):
        with pytest.raises(CorpusError, match="Known:"):
            get_corpus("not-a-corpus")

    def test_entry_serialises(self):
        json.dumps([e.as_dict() for e in list_corpora()])


class TestVerification:
    @pytest.fixture()
    def fake_manifest(self, tmp_path, monkeypatch):
        """A one-entry manifest pointing at a file we can actually create."""
        payload = b"detection events, pretend edition\n"
        data_file = tmp_path / "sample.bin"
        data_file.write_bytes(payload)

        import hashlib

        manifest = {
            "schema": "qb.corpus_manifest.v1",
            "corpora": [
                {
                    "name": "test-corpus",
                    "description": "a small file created by the test",
                    "publisher": "Someone Else",
                    "doi": "10.5281/zenodo.0000000",
                    "url": "https://example.invalid/sample.bin",
                    "filename": "sample.bin",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "n_bytes": len(payload),
                    "fetched": "2026-08-15",
                    "citation": "Someone Else, a dataset (2026)",
                    "licence": "as stated by the publisher",
                }
            ],
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        monkeypatch.setattr("qb_compiler.corpus._MANIFEST_PATH", manifest_path)
        return data_file

    def test_matching_file_verifies_and_prints_the_citation(self, fake_manifest):
        result = verify_corpus_file("test-corpus", fake_manifest)
        assert result.status == VERIFIED
        assert result.ok
        assert "Someone Else" in result.detail

    def test_edited_file_is_a_mismatch(self, fake_manifest):
        # Same length, different bytes: the case a size check would wave through.
        fake_manifest.write_bytes(b"detection events, edited editionX\n")
        result = verify_corpus_file("test-corpus", fake_manifest)
        assert result.status == MISMATCH
        assert not result.ok
        assert result.expected_sha256 != result.actual_sha256
        assert "size matches" in result.detail

    def test_truncated_file_says_truncated(self, fake_manifest):
        fake_manifest.write_bytes(b"detection")
        result = verify_corpus_file("test-corpus", fake_manifest)
        assert result.status == MISMATCH
        assert "truncated" in result.detail

    def test_missing_file_points_at_the_publisher(self, fake_manifest, tmp_path):
        result = verify_corpus_file("test-corpus", tmp_path / "absent.bin")
        assert result.status == MISSING
        assert "example.invalid" in result.detail
        assert "nothing is mirrored" in result.detail.lower()

    def test_unknown_corpus_is_a_verdict_not_an_exception(self, fake_manifest):
        result = verify_corpus_file("nope", fake_manifest)
        assert result.status == UNKNOWN_CORPUS
        assert not result.ok

    def test_load_corpus_refuses_a_bad_file(self, fake_manifest):
        fake_manifest.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="does not match"):
            load_corpus("test-corpus", fake_manifest)

    def test_load_corpus_returns_the_path_when_it_verifies(self, fake_manifest):
        assert load_corpus("test-corpus", fake_manifest) == fake_manifest

    def test_verification_serialises(self, fake_manifest):
        json.dumps(verify_corpus_file("test-corpus", fake_manifest).as_dict())

    def test_chunked_hash_matches_a_one_shot_hash(self, tmp_path):
        import hashlib

        payload = b"x" * (3 * 1024 * 1024 + 7)
        path = tmp_path / "big.bin"
        path.write_bytes(payload)
        assert file_sha256(path, chunk=1024) == hashlib.sha256(payload).hexdigest()


def test_verify_directory_containing_archive(tmp_path, monkeypatch):
    """Pointing verify at a directory finds the archive by its published filename."""
    from qb_compiler.corpus import get_corpus, verify_corpus_file

    entry = get_corpus("willow-105q-d3-d5-d7")
    blob = tmp_path / entry.filename
    blob.write_bytes(b"not the real archive")
    verdict = verify_corpus_file("willow-105q-d3-d5-d7", tmp_path)
    assert verdict.status == "MISMATCH"
    assert verdict.path == str(blob)


def test_verify_directory_without_archive_refuses_cleanly(tmp_path):
    """A directory without the archive is a clean MISSING verdict, not a traceback."""
    from qb_compiler.corpus import verify_corpus_file

    verdict = verify_corpus_file("willow-105q-d3-d5-d7", tmp_path)
    assert verdict.ok is False
    assert verdict.status == "MISSING"
    assert "is a directory" in verdict.detail
