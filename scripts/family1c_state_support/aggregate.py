"""Family 1c aggregation: prediction table, repair table, gates, verdict.

Reads only. Writes RESULT.json and RESULT.md into this directory.
Gates are PREDECLARED.md sections 3 and 4.4, unchanged.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RELEASE_ROOT = Path(os.environ.get("RELEASE_ROOT", HERE.parents[1])).resolve()
RES = RELEASE_ROOT / "results" / "family1c_state_support"
F1 = RELEASE_ROOT / "dependency" / "family1_second_pairs"
PKG = RELEASE_ROOT / "dependency"

SEEDS = list(range(10))
CELLS = [("P1", "narrow"), ("P1", "broad"), ("P2", "narrow"),
         ("P2", "broad"), ("CH", "narrow"), ("CH", "broad")]
LABEL = {"P1": "diffusion / Fisher-KPP", "P2": "diffusion / Allen-Cahn",
         "CH": "diffusion / Cahn-Hilliard"}


def jload(p):
    return json.loads(Path(p).read_text())


def main() -> None:
    support = jload(RES / "SUPPORT.json")
    pkg = jload(PKG / "results" / "tables" / "baseline" / "BASELINE_RESULT.json")
    nc = jload(F1 / "results" / "P3_negative_control" / "RESULT.json")
    nc_gates = jload(F1 / "results" / "P3_negative_control" / "GATES.json")
    p2_narrow = jload(RES / "banked" / "P2_narrow.json")
    p2_broad = jload(RES / "banked" / "P2_broad.json")
    p1_narrow = jload(RES / "banked" / "P1_narrow.json")

    # ---- R per cell, PREDECLARED.md section 3 table -----------------------
    R = {
        ("P1", "narrow"): dict(median=pkg["narrow"]["median_R"],
                               rng=pkg["narrow"]["R_range"], n=5,
                               src="package BASELINE_RESULT.json (5 seeds)"),
        ("P1", "broad"): dict(median=pkg["broad"]["median_R"],
                              rng=pkg["broad"]["R_range"], n=5,
                              src="package BASELINE_RESULT.json (5 seeds)"),
        ("P2", "narrow"): dict(median=p2_narrow["median_R"], rng=p2_narrow["R_range"],
                               n=10, src="computed here from 10 banked checkpoints"),
        ("P2", "broad"): dict(median=p2_broad["median_R"], rng=p2_broad["R_range"],
                              n=10, src="computed here from 10 banked checkpoints"),
        ("CH", "narrow"): dict(median=nc["narrow"]["median_R"],
                               rng=nc["narrow"]["R_range"], n=10,
                               src="family 1 P3_negative_control (10 seeds)"),
        ("CH", "broad"): dict(median=nc["broad"]["median_R"], rng=nc["broad"]["R_range"],
                              n=10, src="family 1 P3_negative_control (10 seeds)"),
    }

    rows = []
    for p, c in CELLS:
        s = support["cells"][f"{p}_{c}"]
        rows.append({
            "cell": f"{p} {c}", "pair": p, "condition": c, "label": LABEL[p],
            "training_offsets": s["training_offsets"],
            "R_median": R[(p, c)]["median"], "R_range": R[(p, c)]["rng"],
            "R_seeds": R[(p, c)]["n"], "R_source": R[(p, c)]["src"],
            "log10_R": float(np.log10(R[(p, c)]["median"])),
            "pca_k_99": s["pca_k_99"],
            "variance_explained_at_k": s["variance_explained_at_k"],
            "train_MD_p99_radius": s["train_MD_p99_radius"],
            "switch_MD_mean": s["switch_MD_mean"],
            "switch_MD_median": s["switch_MD_median"],
            "switch_MD_over_r99_median": s["switch_MD_over_r99_median"],
            "coverage_inside_pca99": s["coverage_inside_pca99"],
            "inside": s["inside"],
            "S2_band_nn_mean": s["S2_band_nn_mean"],
        })

    logR = np.array([r["log10_R"] for r in rows])
    D = np.array([r["switch_MD_mean"] for r in rows])
    E = np.array([r["S2_band_nn_mean"] for r in rows])
    rho_S = stats.spearmanr(logR, D)
    rho_S2 = stats.spearmanr(logR, E)

    ch_narrow = next(r for r in rows if r["cell"] == "CH narrow")
    ch_broad = next(r for r in rows if r["cell"] == "CH broad")
    p1_broad_row = next(r for r in rows if r["cell"] == "P1 broad")
    p2_broad_row = next(r for r in rows if r["cell"] == "P2 broad")

    pred = {
        "PRED_A_spearman": {
            "statistic": "Spearman rho(log10 R, mean switch-state Mahalanobis distance)",
            "cells": 6, "rho": float(rho_S.statistic), "p_value": float(rho_S.pvalue),
            "threshold": 0.8, "pass": bool(rho_S.statistic >= 0.8)},
        "PRED_B_cahn_hilliard_outside": {
            "cell": "CH narrow", "coverage": ch_narrow["coverage_inside_pca99"],
            "threshold": "coverage < 0.5", "pass": bool(
                ch_narrow["coverage_inside_pca99"] < 0.5),
            "reported_not_gated_CH_broad_coverage": ch_broad["coverage_inside_pca99"]},
        "PRED_C_broad_cells_inside": {
            "P1_broad_coverage": p1_broad_row["coverage_inside_pca99"],
            "P2_broad_coverage": p2_broad_row["coverage_inside_pca99"],
            "threshold": "coverage >= 0.5 for both",
            "pass": bool(p1_broad_row["coverage_inside_pca99"] >= 0.5
                         and p2_broad_row["coverage_inside_pca99"] >= 0.5)},
        "S2_robustness_reported_not_gated": {
            "statistic": "Spearman rho(log10 R, mean nearest-training band-profile L2)",
            "rho": float(rho_S2.statistic), "p_value": float(rho_S2.pvalue)},
    }
    # Post-hoc sensitivity, NOT a gate and never substituted for one: P2's ratio is
    # censored in both directions (see caveats), so the same correlation is reported
    # on the four cells whose metric resolves. Declared as post-hoc here in the code.
    keep = [i for i, r in enumerate(rows) if r["pair"] != "P2"]
    pred["POSTHOC_four_cells_excluding_P2_not_a_gate"] = {
        "note": ("post-hoc, reported only; the predeclared gate PRED-A is the six-cell "
                 "number above and is not replaced by this one"),
        "cells": [rows[i]["cell"] for i in keep],
        "rho_S": float(stats.spearmanr(logR[keep], D[keep]).statistic),
        "rho_S2": float(stats.spearmanr(logR[keep], E[keep]).statistic)}
    # POST-HOC, reported only, changes no verdict: the residual (out-of-basis) norm from
    # results/DIAGNOSTIC.json, written after SUPPORT.json was read. Recorded because it
    # is a lead for a future PREDECLARED family, not because it rescues anything here.
    diag_path = RES / "DIAGNOSTIC.json"
    if diag_path.exists():
        diag = jload(diag_path)["cells"]
        q = np.array([diag[f"{p}_{c}"]["switch_residual_mean"] for p, c in CELLS])
        qr = np.array([diag[f"{p}_{c}"]["switch_residual_over_train_p99_median"]
                       for p, c in CELLS])
        pred["POSTHOC_residual_statistic_not_a_gate"] = {
            "note": ("POST-HOC. Written after S and S2 were read. It is NOT a support "
                     "statistic of this family, did not exist in PREDECLARED.md, and "
                     "changes no verdict. Reported as a lead only."),
            "rho_switch_residual_mean": float(
                stats.spearmanr(logR, q).statistic),
            "rho_switch_residual_over_train_p99": float(
                stats.spearmanr(logR, qr).statistic),
            "per_cell_switch_residual_mean": {
                f"{p} {c}": diag[f"{p}_{c}"]["switch_residual_mean"]
                for p, c in CELLS},
            "per_cell_coverage_inside_train_residual_p99": {
                f"{p} {c}": diag[f"{p}_{c}"]["coverage_inside_train_residual_p99"]
                for p, c in CELLS},
        }
    pred["metric_resolution_per_cell"] = {
        r["cell"]: {
            "median_e_off": v, "persistence_off": p,
            "e_off_over_persistence_off": v / p,
            "censored_above_e_off_ge_persistence_off": bool(v >= p)}
        for r, v, p in (
            (rows[2], p2_narrow["median_e_off"], p2_narrow["persistence_off"]),
            (rows[3], p2_broad["median_e_off"], p2_broad["persistence_off"]),
            (rows[4], nc["narrow"]["median_e_off"],
             nc_gates["G2_metric_resolves"]["persistence_off"]),
            (rows[5], nc["broad"]["median_e_off"],
             nc_gates["G2_metric_resolves"]["persistence_off"]),
            (rows[0], pkg["narrow"]["median_e_off"],
             pkg["gates"]["G2_metric_resolves"]["persistence_off"]),
            (rows[1], pkg["broad"]["median_e_off"],
             pkg["gates"]["G2_metric_resolves"]["persistence_off"]))}
    pred["PRED_verdict"] = "PASS" if all(
        pred[k]["pass"] for k in ("PRED_A_spearman", "PRED_B_cahn_hilliard_outside",
                                  "PRED_C_broad_cells_inside")) else "FAIL"

    # ---- repair -----------------------------------------------------------
    gates = jload(RES / "broadS" / "GATES.json")
    repair_rows = {}
    for pair in ("CH", "P1"):
        per = {}
        for s in SEEDS:
            per[s] = jload(RES / "broadS" / f"{pair}_seed{s}.json")
        rb_ = [per[s]["R"] for s in SEEDS]
        if pair == "CH":
            rn = [nc["per_seed"][f"narrow_{s}"]["R"] for s in SEEDS]
            rn_src = "family 1 P3_negative_control banked narrow fits (10 seeds)"
        else:
            rn = [p1_narrow["per_seed"][str(s)]["R"] for s in SEEDS]
            rn_src = ("this harness, the 10 banked narrow checkpoints evaluated on the "
                      "Fisher-KPP switch cohort (10 seeds)")
        repair_rows[pair] = {
            "R_narrow_per_seed": rn, "R_broadS_per_seed": rb_,
            "R_narrow_source": rn_src,
            "e_on_broadS_per_seed": [per[s]["e_on"] for s in SEEDS],
            "e_off_broadS_per_seed": [per[s]["e_off"] for s in SEEDS],
            "median_R_narrow": float(np.median(rn)),
            "median_R_broadS": float(np.median(rb_)),
            "R_narrow_range": [float(min(rn)), float(max(rn))],
            "R_broadS_range": [float(min(rb_)), float(max(rb_))],
            "ratio": float(np.median(rb_) / np.median(rn)),
            "coverage_achieved": support["cells"][f"{pair}_broadS"][
                "coverage_inside_pca99"],
            "broadS_pca_k_99": support["cells"][f"{pair}_broadS"]["pca_k_99"],
            "broadS_switch_MD_mean": support["cells"][f"{pair}_broadS"][
                "switch_MD_mean"],
            "broadS_S2_band_nn_mean": support["cells"][f"{pair}_broadS"][
                "S2_band_nn_mean"],
        }

    repair = {
        "REPAIR_1_CH_ratio": {
            "median_R_narrow": repair_rows["CH"]["median_R_narrow"],
            "median_R_broadS": repair_rows["CH"]["median_R_broadS"],
            "ratio": repair_rows["CH"]["ratio"], "threshold": 0.25,
            "pass": bool(repair_rows["CH"]["ratio"] <= 0.25)},
        "REPAIR_2_G2": gates["REPAIR_2_G2_dynamic_range"],
        "REPAIR_3_P6": gates["REPAIR_3_P6_convergence"],
        "REPAIR_4_coverage": {
            "coverage_achieved": repair_rows["CH"]["coverage_achieved"],
            "threshold": 0.9,
            "pass": bool(repair_rows["CH"]["coverage_achieved"] >= 0.9)},
        "REPAIR_5_P1_positive_control": {
            "median_R_narrow": repair_rows["P1"]["median_R_narrow"],
            "median_R_broadS": repair_rows["P1"]["median_R_broadS"],
            "ratio": repair_rows["P1"]["ratio"], "threshold": 0.25,
            "package_mean_broadening_ratio": float(
                pkg["broad"]["median_R"] / pkg["narrow"]["median_R"]),
            "ratio_against_package_narrow_median": float(
                repair_rows["P1"]["median_R_broadS"] / pkg["narrow"]["median_R"]),
            "coverage_achieved": repair_rows["P1"]["coverage_achieved"],
            "pass": bool(repair_rows["P1"]["ratio"] <= 0.25)},
    }
    repair["REPAIR_verdict"] = "PASS" if all(
        repair[k]["pass"] for k in ("REPAIR_1_CH_ratio", "REPAIR_2_G2", "REPAIR_3_P6",
                                    "REPAIR_4_coverage",
                                    "REPAIR_5_P1_positive_control")) else "FAIL"

    binding = [k for k in ("PRED_A_spearman", "PRED_B_cahn_hilliard_outside",
                           "PRED_C_broad_cells_inside") if not pred[k]["pass"]]
    binding += [k for k in ("REPAIR_1_CH_ratio", "REPAIR_2_G2", "REPAIR_3_P6",
                            "REPAIR_4_coverage", "REPAIR_5_P1_positive_control")
                if not repair[k]["pass"]]
    if pred["PRED_verdict"] == "PASS" and repair["REPAIR_verdict"] == "PASS":
        verdict = "PASS"
    elif pred["PRED_verdict"] == "PASS" or repair["REPAIR_verdict"] == "PASS":
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    payload = {
        "schema": "family1c-result-v1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predeclared_sha256": (RES / "PREDECLARED.sha256").read_text().split()[0],
        "prediction_table": rows,
        "PRED": pred,
        "repair_table": repair_rows,
        "REPAIR": repair,
        "family_verdict": verdict,
        "binding_gates": binding,
        "caveats": {
            "P2_G2": ("P2's R is newly read here from checkpoints family 1 banked. "
                      "Family 1 stopped P2 at G2 (dynamic range 30.8 < 100) and its "
                      "verdict stays FAIL on G2; this R is a poorly-resolved ratio and "
                      "is flagged wherever it appears."),
            "P1_R_seeds": ("P1 narrow/broad R in the prediction table are the package's "
                           "5-seed medians, as predeclared. The repair table's P1 "
                           "R_narrow is a 10-seed median measured with this harness."),
            "nc_gates_dynamic_range": nc_gates["G2_metric_resolves"]["dynamic_range"],
            "P1_harness_agreement": {
                "package_median_R_narrow_5_seeds": pkg["narrow"]["median_R"],
                "this_harness_median_R_narrow_10_seeds": p1_narrow["median_R"],
                "this_harness_R_range": p1_narrow["R_range"]},
        },
    }
    (HERE / "RESULT.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"family_verdict": verdict, "PRED": pred["PRED_verdict"],
                      "REPAIR": repair["REPAIR_verdict"], "binding": binding,
                      "rho_S": float(rho_S.statistic),
                      "rho_S2": float(rho_S2.statistic)}, indent=2))


if __name__ == "__main__":
    main()
