"""Locations and identity of the published artifact.

The results, figures and learner artifacts this repository publishes live under `results/`
and `data/`, and `results/CHECKSUMS.txt` carries a SHA-256 for every one of them. This
package does not copy any of that. It resolves paths into the published tree and provides
the checks in `verify.py`, so the repository holds exactly one copy of every published
file and the checksum file stays a complete statement about the tree rather than about one
of two divergent copies.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CHECKSUMS = RESULTS / "CHECKSUMS.txt"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
DATA = ROOT / "data"
LEARNER_ARTIFACTS = DATA / "learner_artifacts"

# What the checksum file covers, relative to ROOT. `scripts/build_checksums.py` writes
# over exactly these and `verify.py` reads them back, so one list decides what "published"
# means instead of two that can drift apart.
COVERED = ("results", "data/learner_artifacts")

__all__ = [
    "ROOT",
    "RESULTS",
    "CHECKSUMS",
    "FIGURES",
    "TABLES",
    "DATA",
    "LEARNER_ARTIFACTS",
    "COVERED",
]
