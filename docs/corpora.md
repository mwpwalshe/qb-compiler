# Public QEC datasets, hash verified

Every group working on decoding wrangles the same handful of public releases by hand, and every one
of them writes the same paragraph about which files they used. `qbc corpus` holds a pinned manifest
so that paragraph can say something checkable.

Nothing is redistributed here. These are other people's datasets, published under their own terms.
The manifest records where the publisher put the file, the sha256 of a copy fetched from there, the
size, the date it was fetched, and the citation the publisher asks for.

```bash
$ qbc corpus list
NAME                     PUBLISHER                      SIZE  DOI
----------------------------------------------------------------------------------------
quera-surface-code       QuEra Computing         996,799,635  10.5281/zenodo.15685795
willow-105q-d3-d5-d7     Google Quantum AI     5,716,907,033  10.5281/zenodo.13273331

Nothing is mirrored here. Fetch from the publisher, then run qbc corpus verify.
```

```bash
$ qbc corpus show willow-105q-d3-d5-d7
willow-105q-d3-d5-d7
  Surface code memory experiments at distances 3, 5 and 7 on a 105 qubit device, with detection
  events and observable flips per shot.
  publisher : Google Quantum AI
  doi       : 10.5281/zenodo.13273331
  file      : google_105Q_surface_code_d3_d5_d7.zip (5,716,907,033 bytes)
  url       : https://zenodo.org/records/13273331/files/google_105Q_surface_code_d3_d5_d7.zip
  sha256    : 1e8e3b4f5f35ba4fd9b4a8448473f37a7090b404fd6eff0c00188092876070dd
  fetched   : 2026-05-31
  licence   : as stated on the Zenodo record
  cite      : Google Quantum AI and Collaborators, Quantum error correction below the surface
              code threshold, Nature (2025). Dataset: https://doi.org/10.5281/zenodo.13273331
```

## Check your copy

```bash
$ qbc corpus verify willow-105q-d3-d5-d7 ~/data/google_105Q_surface_code_d3_d5_d7.zip
VERIFIED: google_105Q_surface_code_d3_d5_d7.zip matches the pinned digest for
willow-105q-d3-d5-d7 (5,716,907,033 bytes). Cite: Google Quantum AI and Collaborators, Quantum
error correction below the surface code threshold, Nature (2025).
Dataset: https://doi.org/10.5281/zenodo.13273331
  expected: 1e8e3b4f5f35ba4fd9b4a8448473f37a7090b404fd6eff0c00188092876070dd
  actual  : 1e8e3b4f5f35ba4fd9b4a8448473f37a7090b404fd6eff0c00188092876070dd
```

Exit 0 means the bytes match. Exit 2 means they do not, and the two digests are printed. Exit 1
means the file is not there or the name is not in the manifest.

A truncated download and an edited file look identical to every other tool in the stack, and both
change a decoder benchmark silently. The message says which of the two it looks like: a size
mismatch is usually a transfer that stopped, a size match with a different digest is different
content.

```python
from qb_compiler.corpus import load_corpus, corpus_citation

path = load_corpus("willow-105q-d3-d5-d7", "~/data/google_105Q_surface_code_d3_d5_d7.zip")
print(corpus_citation("willow-105q-d3-d5-d7"))
```

`load_corpus` raises rather than returning a path when the digest does not match. A benchmark run on
unverified bytes is worse than one that did not run, because the number it produces looks the same.

## What a digest does and does not tell you

It says your copy is byte for byte the copy that was fetched on the date in the entry. It does not
say the publisher has not re-released since under the same DOI, which happens and is legitimate. A
mismatch is information to chase, not an accusation: check the record's version history first.

## Why this is not a downloader

There is no `qbc corpus fetch`. These archives run to gigabytes, the publishers already serve them
properly, and a half implemented resumable download inside a compiler package is a liability rather
than a feature. Fetch with whatever you already trust, then verify.

## Coming

**qbcal-2026-02** is being archived. Archive record and DOI to follow, at which point it gets a
manifest entry with a real digest like every other row. It is named here rather than listed, because
a manifest entry whose digest and DOI are placeholders is worse than no entry: it looks checkable
and is not. Grep `DOI_QBCAL_2026_02_PENDING` for the one place that changes when the record lands.

## What is not listed

Everything else in the manifest is somebody else's public release. A dataset of ours belongs there
once it has been archived and has a record a reader can fetch it from, and not before.
