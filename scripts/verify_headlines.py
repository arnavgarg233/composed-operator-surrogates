#!/usr/bin/env python3
"""Verify the public result summaries against the shipped evidence."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def at(value: Any, path: tuple[str, ...]) -> Any:
    for part in path:
        value = value[part]
    return value


CHECKS = (
    ("results/family1b_pair_screen/RESULT.json", (("counts", "passing"), ("counts", "configurations")), "12 of 25"),
    ("results/family1c_state_support/RESULT.json", (("PRED", "PRED_A_spearman", "rho"), ("REPAIR", "REPAIR_4_coverage", "coverage_achieved")), "-0.257 and coverage 0.828"),
    ("results/family1e_state_broadening_new_pairs/RESULT.json", (("pairs", "S1_fkpp_r0.5", "broadS", "median_R"), ("pairs", "S1_grayscott_1sp", "broadS", "median_R")), "0.033 and 0.035"),
    ("results/family3_baselines/RESULT.json", (("results", "ensemble", "observables", "trajectory", "S_learner", "mean"), ("results", "ensemble", "observables", "trajectory", "baselines", "order_blind", "S_baseline", "mean")), "0.396 versus 1.653"),
    ("results/family5_ic_robustness/RESULT.json", (("results", "family1_spectral_grf_power_law", "median_contrast_to_cross_ic"), ("results", "family2_bump_localized", "median_contrast_to_cross_ic")), "0.241 and 0.179"),
    ("results/family6_serrano_baseline/RESULT_SMOKE.json", (("R", "narrow_oracle", "R", "median"), ("R", "broad_oracle", "R", "median"), ("R", "serrano_narrow_lie_causal/exhaustive", "R", "median")), "376.929, broad oracle 85.684, exhaustive splitting 1376.858"),
    ("results/family4b_unet_broadS/RESULT.json", (("P3_broadening_repairs", "ratio_broadS_to_narrow"),), "0.101, PASS"),
)


def validate_internal_derivations() -> list[str]:
    problems: list[str] = []
    document = load("results/family1_rerun/RESULT.json")
    for pair in ("S1_fkpp_r0.5", "S1_grayscott_1sp"):
        record = document["pairs"][pair]
        for condition in ("narrow", "broad"):
            values = [v["R"] for key, v in record["per_seed"].items() if key.startswith(condition + "_")]
            if statistics.median(values) != record[condition]["median_R"]:
                problems.append(f"results/family1_rerun/RESULT.json:{pair} median mismatch")
    document = load("results/family1e_state_broadening_new_pairs/RESULT.json")
    for pair in ("S1_fkpp_r0.5", "S1_grayscott_1sp"):
        record = document["pairs"][pair]
        values = [row["R"] for row in record["broadS"]["per_seed"].values()]
        if statistics.median(values) != record["broadS"]["median_R"]:
            problems.append(f"results/family1e_state_broadening_new_pairs/RESULT.json:{pair} median mismatch")
    document = load("results/family6_serrano_baseline/RESULT_SMOKE.json")
    for arm in ("narrow_oracle", "broad_oracle", "serrano_narrow_lie_causal/exhaustive"):
        record = document["R"][arm]["R"]
        values = record["per_seed"].values() if isinstance(record["per_seed"], dict) else record["per_seed"]
        if statistics.median(values) != record["median"]:
            problems.append(f"family6 {arm} median mismatch")
    return problems


def verify() -> list[str]:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    problems = validate_internal_derivations()
    for relative, paths, phrase in CHECKS:
        values = [at(load(relative), path) for path in paths]
        if relative.endswith("family1b_pair_screen/RESULT.json") and values != [12, 25]:
            problems.append("pair-screen counts changed")
        if relative.endswith("family1c_state_support/RESULT.json") and [round(values[0], 3), round(values[1], 3)] != [-0.257, 0.828]:
            problems.append("state-support values changed")
        if relative.endswith("family1e_state_broadening_new_pairs/RESULT.json") and [round(values[0], 3), round(values[1], 3)] != [1.948, 3.701]:
            problems.append("new-pair values changed")
        if phrase not in readme or f"`{relative}:{', '.join('.'.join(path) for path in paths)}`" not in readme:
            problems.append(f"missing public result check: {relative}")
    return problems


if __name__ == "__main__":
    problems = verify()
    for problem in problems:
        print(f"FAIL: {problem}")
    raise SystemExit(bool(problems))
