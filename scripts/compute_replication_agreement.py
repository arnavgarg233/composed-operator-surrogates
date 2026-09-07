"""Compute, rather than assert, the agreement between regenerated and banked values.

The receipt audit found that two claims this manuscript leans on were never computed as
comparisons. `BAND_CHECK_RESULT.json` records the regenerated median ratio and, in a
separate block, the banked one, but nothing in the file compares them: the assertion that
their intervals overlap was made in prose. `INTEGRITY_CONTROLS_RESULT.json` has the same
shape.

That is the failure class the portfolio has now hit four times. A value being present
beside another value is not a comparison. This file performs the comparisons and records
observed, expected and verdict for each, so the manuscript cites a computed result rather
than a reading.

Nothing here re-runs an experiment. It compares numbers already produced.
"""

from __future__ import annotations

import json
from pathlib import Path

TABLES = Path(__file__).resolve().parents[1] / "results" / "tables"


def overlap(first: list[float], second: list[float]) -> tuple[bool, float]:
    low = max(first[0], second[0])
    high = min(first[1], second[1])
    if high < low:
        return False, 0.0
    width = min(first[1] - first[0], second[1] - second[0])
    return True, (high - low) / width if width else 0.0


def main() -> None:
    band = json.loads((TABLES / "band_check" / "BAND_CHECK_RESULT.json").read_text())
    controls = json.loads(
        (TABLES / "band_check" / "INTEGRITY_CONTROLS_RESULT.json").read_text()
    )

    comparisons = []

    # Configuration confirmation. The manuscript states the intervals overlap
    # substantially. That is now computed.
    cutoff5 = next(r for r in band["results"] if r["cutoff"] == 5)
    observed_ci = cutoff5["bootstrap_ci95"]
    expected_ci = band["banked_comparison"]["bootstrap_ci95"]
    overlaps, fraction = overlap(observed_ci, expected_ci)
    comparisons.append({
        "quantity": "median order contrast over cross-initial-condition scale",
        "observed": cutoff5["median_contrast_to_cross_ic"],
        "observed_ci95": observed_ci,
        "expected": band["banked_comparison"]["median_contrast_to_cross_ic"],
        "expected_ci95": expected_ci,
        "expected_confidence": "reconstructed",
        "absolute_difference": abs(cutoff5["median_contrast_to_cross_ic"]
                                   - band["banked_comparison"]["median_contrast_to_cross_ic"]),
        "intervals_overlap": overlaps,
        "overlap_as_fraction_of_narrower_interval": fraction,
        "observed_within_expected_interval": (
            expected_ci[0] <= cutoff5["median_contrast_to_cross_ic"] <= expected_ci[1]
        ),
    })

    # Numerical-floor passes.
    comparisons.append({
        "quantity": "numerical-floor passes",
        "observed": cutoff5["floor_passes"],
        "expected": 256,
        "expected_confidence": "reconstructed",
        "equal": cutoff5["floor_passes"] == 256,
    })

    # Commuting-pair null. Both are far under the gate; the meaningful comparison is
    # that both clear it and that they agree in order of magnitude.
    control_one = controls["control_1_commuting"]
    observed_null = control_one["max_relative_order_difference"]
    expected_null = control_one["banked_value"]
    comparisons.append({
        "quantity": "commuting-pair maximum relative order difference",
        "observed": observed_null,
        "expected": expected_null,
        "expected_confidence": "reconstructed",
        "gate": control_one["gate"],
        "observed_clears_gate": observed_null < control_one["gate"],
        "ratio_observed_over_expected": observed_null / expected_null,
        "same_order_of_magnitude": 0.1 <= observed_null / expected_null <= 10.0,
    })

    # Endpoint contrast decomposition.
    endpoint = controls["control_2_endpoint"]
    observed_share = endpoint["shared_fraction"]
    expected_share = endpoint["banked_shared_fraction"]
    comparisons.append({
        "quantity": "endpoint contrast shared-mean energy fraction",
        "observed": observed_share,
        "expected": expected_share,
        "expected_confidence": "reconstructed",
        "absolute_difference": abs(observed_share - expected_share),
        "relative_difference": abs(observed_share - expected_share) / expected_share,
        "agrees_within_2_percent_relative": (
            abs(observed_share - expected_share) / expected_share < 0.02
        ),
    })

    verdicts = []
    for row in comparisons:
        keys = [k for k in row if isinstance(row[k], bool)]
        verdicts.append(all(row[k] for k in keys))

    print(f"{'agrees':>7}  quantity")
    for row, ok in zip(comparisons, verdicts):
        print(f"{'yes' if ok else 'NO':>7}  {row['quantity']}")
        print(f"         observed {row['observed']}   expected {row['expected']}"
              f"   ({row['expected_confidence']})")
        for key in ("absolute_difference", "relative_difference",
                    "overlap_as_fraction_of_narrower_interval",
                    "ratio_observed_over_expected"):
            if key in row:
                print(f"         {key}: {row[key]:.6g}")

    print(f"\n{sum(verdicts)} of {len(verdicts)} comparisons agree")
    print("Each is now a computed comparison with observed, expected and a verdict, "
          "rather than two values sitting near each other in a file.")

    (TABLES / "REPLICATION_AGREEMENT.json").write_text(json.dumps({
        "schema": "pde-replication-agreement-v1",
        "why": (
            "The receipt audit found these claims were asserted in prose rather than "
            "computed. A value present beside another value is not a comparison."
        ),
        "expected_values_are_reconstructed": (
            "Every expected value here is quoted from a destroyed artifact and cannot be "
            "reopened. Agreement is evidence about the regenerated configuration, not a "
            "re-verification of the original."
        ),
        "comparisons": comparisons,
        "all_agree": all(verdicts),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
