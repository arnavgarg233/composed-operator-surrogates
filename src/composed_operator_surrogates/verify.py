"""Checks a reader can run against the published artifact, with no network.

Each check returns a list of problems; an empty list means it passed. `scripts/` and
`tests/` both call these, so what the test suite asserts and what `reproduce.sh` reports
are the same code rather than two descriptions of it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import CHECKSUMS, COVERED, FIGURES, LEARNER_ARTIFACTS, ROOT, TABLES

FIGURE_RECEIPT = TABLES / "FIGURE_RECEIPT.json"

FIGURE_STEMS = (
    "figure1_state_support",
    "figure2_signal_by_observable",
    "figure3_supports_both_pairs",
    "figure4_broadening_repair",
    "figure5_shared_share_by_time",
    "figure6_signal_second_pair",
    "figure7_fno_intervention",
)

# The result files the headline numbers are read out of. Naming them here means a claim
# loses its evidence loudly rather than through a path that quietly stops resolving.
CITED_TABLES = (
    "broadening/summary.json",
    "baseline/BASELINE_RESULT.json",
    "baseline/DISSOCIATION_RESULT.json",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recorded() -> dict[str, str]:
    rows = {}
    for line in CHECKSUMS.read_text().splitlines():
        if not line.strip():
            continue
        recorded, name = line.split("  ", 1)
        rows[name] = recorded
    return rows


def _published() -> set[str]:
    """Every published file, by path relative to the repository root."""
    names = set()
    for covered in COVERED:
        for path in (ROOT / covered).rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path == CHECKSUMS:
                continue
            names.add(str(path.relative_to(ROOT)))
    return names


def checksum_digests() -> list[str]:
    """Every file the checksum file names is here and hashes to the value it records."""
    problems = []
    for name, expected in sorted(_recorded().items()):
        path = ROOT / name
        if not path.exists():
            problems.append(f"the checksum file names a missing file: {name}")
        elif digest(path) != expected:
            problems.append(f"{name} does not match its recorded digest")
    return problems


def checksums_cover_tree() -> list[str]:
    """Nothing sits in the published tree that the checksum file does not account for.

    The digest check alone would pass a tree with an extra file in it, which is how an
    unlisted file reaches a public repository.
    """
    return [
        f"present but not in the checksum file: {name}"
        for name in sorted(_published() - set(_recorded()))
    ]


def figures_present() -> list[str]:
    """All seven figures ship as vector PDF."""
    return [
        f"missing figure file: {stem}.pdf"
        for stem in FIGURE_STEMS
        if not (FIGURES / f"{stem}.pdf").exists()
    ]


def figure_receipt_describes_what_ships() -> list[str]:
    """The receipt names the figures this repository contains, and no others.

    A receipt listing a file the repository does not carry is worse than no receipt,
    because a reader checking against it cannot tell an omission from a rename.
    """
    recorded = json.loads(FIGURE_RECEIPT.read_text())["sha256"]
    shipped = {path.name for path in FIGURES.glob("*.pdf")}
    problems = [
        f"the figure receipt names a file that does not ship: {name}"
        for name in sorted(set(recorded) - shipped)
    ]
    problems += [
        f"figure ships but the receipt does not name it: {name}"
        for name in sorted(shipped - set(recorded))
    ]
    problems += [
        f"{name} does not match the digest in the figure receipt"
        for name in sorted(set(recorded) & shipped)
        if digest(FIGURES / name) != recorded[name]
    ]
    return problems


def cited_tables_present() -> list[str]:
    """The result files the headline numbers are quoted from are still here."""
    return [
        f"missing result file: results/tables/{name}"
        for name in CITED_TABLES
        if not (TABLES / name).exists()
    ]


def learner_artifacts_present() -> list[str]:
    """Five seeds, each with a composed-prediction bundle and a weight file."""
    return [
        f"missing learner artifact: {name}"
        for seed in range(5)
        for name in (f"composed_seed{seed}.npz", f"weights_seed{seed}.pt")
        if not (LEARNER_ARTIFACTS / name).exists()
    ]


CHECKS = {
    "recorded digests match": checksum_digests,
    "the checksum file covers the whole published tree": checksums_cover_tree,
    "all seven figures present as PDF": figures_present,
    "the figure receipt describes the figures that ship": (
        figure_receipt_describes_what_ships
    ),
    "results behind the headline numbers present": cited_tables_present,
    "learner artifacts present for five seeds": learner_artifacts_present,
}


def run_all() -> dict[str, list[str]]:
    return {name: check() for name, check in CHECKS.items()}
