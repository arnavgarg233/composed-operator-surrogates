"""The published artifact is complete and matches its own checksum file.

These call the same functions `scripts/verify_package.py` calls, so the suite and the
replay script cannot disagree about what passing means.
"""

from __future__ import annotations

from composed_operator_surrogates import (
    CHECKSUMS,
    FIGURES,
    LEARNER_ARTIFACTS,
    TABLES,
    verify,
)


def test_recorded_digests_match():
    assert verify.checksum_digests() == []


def test_checksums_cover_the_whole_published_tree():
    assert verify.checksums_cover_tree() == []


def test_all_seven_figures_present():
    assert verify.figures_present() == []


def test_figure_receipt_describes_what_ships():
    assert verify.figure_receipt_describes_what_ships() == []


def test_results_behind_the_headline_numbers_present():
    assert verify.cited_tables_present() == []


def test_learner_artifacts_present_for_five_seeds():
    assert verify.learner_artifacts_present() == []


def test_checksum_file_does_not_record_itself():
    assert "results/CHECKSUMS.txt" not in CHECKSUMS.read_text()


def test_paths_resolve():
    for path in (FIGURES, TABLES, LEARNER_ARTIFACTS):
        assert path.is_dir(), path
