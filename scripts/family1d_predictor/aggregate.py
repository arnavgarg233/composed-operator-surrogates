"""Family 1d aggregation: held-out table, gates PRED-S3-A/B, RESULT.json + RESULT.md.

Every threshold and every cell is fixed by PREDECLARED.md
(sha256 c50bc3eeb6186dd7146085d308f71f08995cd30827491c14bf4f289946f9cdb5).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
RELEASE_ROOT = Path(os.environ.get("RELEASE_ROOT", HERE.parents[1])).resolve()
F1C = RELEASE_ROOT / "results" / "family1c_state_support"
F6 = RELEASE_ROOT / "results" / "family6_serrano_baseline"
RERUN = RELEASE_ROOT / "results" / "family1_rerun"
RESULTS = RELEASE_ROOT / "results" / "family1d_predictor"
HELD = RESULTS / "heldout"

SPEARMAN_MIN = 0.8
MIN_HELDOUT_CELLS = 6


def rho(x, y):
    r = sps.spearmanr(np.asarray(x, float), np.asarray(y, float))
    return float(r.statistic), float(r.pvalue)


def med_R_from_broadS(pair: str):
    rs = []
    for s in range(10):
        p = F1C / "results" / "broadS" / f"{pair}_seed{s}.json"
        rs.append(json.loads(p.read_text())["R"])
    return float(np.median(rs)), [float(min(rs)), float(max(rs))], len(rs)


def main() -> None:
    started = time.perf_counter()
    st = json.loads((RESULTS / "STATISTICS.json").read_text())["cells"]
    f1c = json.loads((F1C / "RESULT.json").read_text())
    smoke = json.loads((F6 / "results" / "RESULT_SMOKE.json").read_text())

    # ---------------------------------------------------------------- R per cell
    # Six selection cells: family 1c's own prediction table.
    sel_R = {}
    for row in f1c["prediction_table"]:
        sel_R[f"{row['pair']}_{row['condition']}"] = (
            float(row["R_median"]), int(row["R_seeds"]), row["R_source"],
            row.get("R_range"))

    cells = {}

    def add(name, R, source, seeds, blind, kind, extra=None):
        s = st[name if name in st else extra["stat_cell"]]
        cells[name] = {
            "R": R, "log10R": float(np.log10(R)), "seeds": seeds,
            "R_source": source, "blind_at_predeclaration": blind, "kind": kind,
            "stat_cell": name if name in st else extra["stat_cell"],
            "D3": s["D3_switch_residual_mean"],
            "C3": s["C3_coverage_inside_train_residual_p99"],
            "D3f": s["D3f_switch_fraction_outside_basis_median"],
            "Dbar": s["Dbar_switch_MD_mean"],
            "coverage_S": s["coverage_S_inside_pca99"],
            "pca_k_99": s["pca_k_99"],
            "train_residual_p99": s["train_residual_p99"],
        }
        if extra:
            cells[name].update({k: v for k, v in extra.items() if k != "stat_cell"})

    # --- the six selection cells (reported, never gated) ---------------------
    for name, (R, seeds, src, rng) in sel_R.items():
        add(name, R, f"family1c_state_support/RESULT.json prediction_table -- {src}",
            seeds, False, "selection", {"R_range": rng})

    # --- held-out (b): family 1c BROAD-S -------------------------------------
    for pair in ("CH", "P1"):
        R, rng, n = med_R_from_broadS(pair)
        add(f"{pair}_broadS", R,
            f"family1c_state_support/results/broadS/{pair}_seed*.json, {n}-seed median",
            n, False, "heldout", {"R_range": rng})

    # --- held-out (c): family 6 smoke ---------------------------------------
    for cond, stat_cell in (("narrow", "P1_narrow"), ("broad", "P1_broad")):
        arm = smoke["R"][f"{cond}_oracle"]
        add(f"F6_P1_{cond}", float(arm["R_frame0_bridge"]["median"]),
            "family6_serrano_baseline/results/RESULT_SMOKE.json "
            f"R.{cond}_oracle.R_frame0_bridge.median (frame-0 on-support bridge: the "
            "package/family-1 e_on convention)",
            len(smoke["seeds"]), False, "heldout",
            {"stat_cell": stat_cell,
             "R_per_seed": arm["R_frame0_bridge"]["per_seed"],
             "R_headline_convention": float(arm["R"]["median"]),
             "duplicate_of": stat_cell,
             "duplicate_note": "training cloud identical to the selection cell "
                               f"{stat_cell}; D3/C3/Dbar are therefore identical and "
                               "carry no independent information about the predictor"})

    # --- held-out (d): the two blind S1_fkpp_r0.5 cells ----------------------
    fk_present = []
    for cond in ("narrow", "broad"):
        p = HELD / f"FK05_{cond}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        add(f"FK05_{cond}", d["median_R"],
            f"family1d, {len(d['seeds'])}-seed median ({d['note'][:60]}...)",
            len(d["seeds"]), True, "heldout",
            {"R_range": d["R_range"], "median_e_on": d["median_e_on"],
             "median_e_off": d["median_e_off"],
             "persistence_off": d["persistence_off"],
             "censored_above": not d["max_e_off_below_persistence_off"]})
        fk_present.append(f"FK05_{cond}")

    # --- held-out (a): the live family-1 rerun, if it has landed -------------
    rerun_status = {"polled": True, "result_files": []}
    for pat in ("RESULT.json", "RESULT_S1_fkpp_r0.5.json",
                "RESULT_S1_grayscott_1sp.json"):
        if (RERUN / pat).exists():
            rerun_status["result_files"].append(pat)
    rerun_ck = sorted((RERUN / "ckpt").glob("*/*.npz"))
    rerun_status["checkpoints_present"] = len(rerun_ck)
    rerun_status["fits_required"] = 40
    rerun_status["note"] = ("family1_rerun is a live detached driver (PID 89748). No "
                            "RESULT file existed inside this family's 4-hour window, so "
                            "held-out group (a) contributes no cells."
                            if not rerun_status["result_files"] else
                            "rerun RESULT files present; see below")

    # ---------------------------------------------------------------- gates
    held = [k for k, v in cells.items() if v["kind"] == "heldout"]
    held.sort(key=lambda k: cells[k]["D3"])
    sel = [k for k, v in cells.items() if v["kind"] == "selection"]

    def corr(names, field):
        if len(names) < 3:
            return {"n": len(names), "rho": None, "p": None,
                    "note": "fewer than 3 cells; a rank correlation is not defined"}
        r, p = rho([cells[n]["log10R"] for n in names],
                   [cells[n][field] for n in names])
        return {"n": len(names), "rho": r, "p": p, "cells": names}

    dedup = [k for k in held if not k.startswith("F6_")]
    blind = [k for k in held if cells[k]["blind_at_predeclaration"]]

    A = corr(held, "D3")
    A_pass = bool(A["rho"] is not None and A["rho"] >= SPEARMAN_MIN
                  and len(held) >= MIN_HELDOUT_CELLS)

    B = {
        "CH": {"C3_narrow": cells["CH_narrow"]["C3"], "C3_broadS": cells["CH_broadS"]["C3"],
               "pass": bool(cells["CH_broadS"]["C3"] > cells["CH_narrow"]["C3"])},
        "P1": {"C3_narrow": cells["P1_narrow"]["C3"], "C3_broadS": cells["P1_broadS"]["C3"],
               "pass": bool(cells["P1_broadS"]["C3"] > cells["P1_narrow"]["C3"])},
    }
    B_pass = bool(B["CH"]["pass"] and B["P1"]["pass"])

    partial = bool(len(held) < MIN_HELDOUT_CELLS or not rerun_status["result_files"])
    verdict = "PASS" if (A_pass and B_pass) else "FAIL"
    if partial:
        verdict += " (PARTIAL)"

    out = {
        "schema": "family1d-result-v1",
        "predeclared_sha256":
            "c50bc3eeb6186dd7146085d308f71f08995cd30827491c14bf4f289946f9cdb5",
        "verdict": verdict,
        "gates": {
            "PRED_S3_A_spearman_heldout": {
                "definition": "Spearman(log10 R, D3) over the held-out cells",
                "threshold": SPEARMAN_MIN, "min_cells": MIN_HELDOUT_CELLS,
                **A, "pass": A_pass},
            "PRED_S3_B_C3_monotone_under_intervention": {
                "definition": "C3(broad-S) > C3(narrow) for both CH and P1",
                **B, "pass": B_pass},
        },
        "reported_not_gated": {
            "six_selection_cells_spearman_D3": corr(sel, "D3"),
            "six_selection_cells_spearman_Dbar": corr(sel, "Dbar"),
            "six_selection_cells_spearman_D3f": corr(sel, "D3f"),
            "heldout_spearman_Dbar_mahalanobis": corr(held, "Dbar"),
            "heldout_spearman_D3f": corr(held, "D3f"),
            "heldout_dedup_spearman_D3": corr(dedup, "D3"),
            "heldout_dedup_spearman_Dbar": corr(dedup, "Dbar"),
            "blind_subset": {"cells": blind, "n": len(blind),
                             "note": "cells whose R and D3 were not visible when "
                                     "PREDECLARED.md was written"},
        },
        "cells": cells,
        "heldout_cells": held,
        "selection_cells": sel,
        "rerun_status": rerun_status,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if (HELD / "FK05_GATES.json").exists():
        out["fk05_gates"] = json.loads((HELD / "FK05_GATES.json").read_text())
    if (HELD / "FK05_GEOMETRY.json").exists():
        out["fk05_geometry"] = json.loads((HELD / "FK05_GEOMETRY.json").read_text())

    (HERE / "RESULT.json").write_text(json.dumps(out, indent=2) + "\n")

    # ------------------------------------------------------------------ print
    print(f"VERDICT {verdict}")
    print(f"PRED-S3-A rho={A['rho']} n={A['n']} -> {'PASS' if A_pass else 'FAIL'}")
    print(f"PRED-S3-B {'PASS' if B_pass else 'FAIL'}  "
          f"CH {B['CH']['C3_narrow']:.4f}->{B['CH']['C3_broadS']:.4f}  "
          f"P1 {B['P1']['C3_narrow']:.4f}->{B['P1']['C3_broadS']:.4f}")
    print(f"Mahalanobis on held-out: rho="
          f"{out['reported_not_gated']['heldout_spearman_Dbar_mahalanobis']['rho']}")
    print(f"de-duplicated (no F6): rho="
          f"{out['reported_not_gated']['heldout_dedup_spearman_D3']['rho']} "
          f"n={out['reported_not_gated']['heldout_dedup_spearman_D3']['n']}")
    print("\nheld-out table (sorted by D3)")
    for k in held:
        v = cells[k]
        print(f"  {k:14s} R {v['R']:10.3f}  log10R {v['log10R']:7.3f}  "
              f"D3 {v['D3']:9.5f}  C3 {v['C3']:.4f}  Dbar {v['Dbar']:7.4f}  "
              f"blind={v['blind_at_predeclaration']}")
    print("\nselection cells")
    for k in sel:
        v = cells[k]
        print(f"  {k:14s} R {v['R']:10.3f}  log10R {v['log10R']:7.3f}  "
              f"D3 {v['D3']:9.5f}  C3 {v['C3']:.4f}  Dbar {v['Dbar']:7.4f}")




# ---------------------------------------------------------------------------
# Markdown tables, generated from RESULT.json so no number is ever transcribed
# by hand into RESULT.md.
# ---------------------------------------------------------------------------
def tables() -> None:
    d = json.loads((HERE / "RESULT.json").read_text())
    cells, L = d["cells"], []
    A = d["gates"]["PRED_S3_A_spearman_heldout"]
    B = d["gates"]["PRED_S3_B_C3_monotone_under_intervention"]
    rn = d["reported_not_gated"]

    def row(k, blindcol=True):
        v = cells[k]
        b = {True: "**yes**", False: "no"}[v["blind_at_predeclaration"]]
        s = (f"| `{k}` | {v['R']:.3f} | {v['seeds']} | {v['log10R']:+.3f} | "
             f"{v['D3']:.5f} | {v['C3']:.4f} | {v['Dbar']:.4f} | {v['coverage_S']:.4f} | "
             f"{v['pca_k_99']} |")
        return s + (f" {b} |" if blindcol else "")

    L.append("### Held-out cells (sorted by D3)\n")
    L.append("| cell | R | seeds | log₁₀R | **D3** | **C3** | D̄ (Mahalanobis) | cov_S | k | blind |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for k in d["heldout_cells"]:
        L.append(row(k))
    L.append("")
    L.append("### The six selection cells (reported, gate nothing)\n")
    L.append("| cell | R | seeds | log₁₀R | **D3** | **C3** | D̄ (Mahalanobis) | cov_S | k |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k in d["selection_cells"]:
        L.append(row(k, blindcol=False))
    L.append("")
    L.append("### Correlations\n")
    L.append("| set | n | statistic | Spearman ρ | p |")
    L.append("|---|---|---|---|---|")

    def crow(label, obj, stat):
        if obj.get("rho") is None:
            L.append(f"| {label} | {obj['n']} | {stat} | — | — |")
        else:
            L.append(f"| {label} | {obj['n']} | {stat} | **{obj['rho']:+.4f}** | "
                     f"{obj['p']:.4f} |")
    crow("**held-out (the gate)**", A, "**S3 = D3**")
    crow("held-out", rn["heldout_spearman_Dbar_mahalanobis"], "PCA-99 Mahalanobis D̄")
    crow("held-out", rn["heldout_spearman_D3f"], "D3f (normalised residual)")
    crow("held-out minus family-6 duplicates", rn["heldout_dedup_spearman_D3"], "S3 = D3")
    crow("held-out minus family-6 duplicates", rn["heldout_dedup_spearman_Dbar"],
         "PCA-99 Mahalanobis D̄")
    crow("six selection cells", rn["six_selection_cells_spearman_D3"], "S3 = D3")
    crow("six selection cells", rn["six_selection_cells_spearman_Dbar"],
         "PCA-99 Mahalanobis D̄")
    crow("six selection cells", rn["six_selection_cells_spearman_D3f"], "D3f")
    L.append("")
    L.append("### Gates\n")
    L.append("| gate | definition | threshold | measured | result |")
    L.append("|---|---|---|---|---|")
    rr = "**PASS**" if A["pass"] else "**FAIL**"
    L.append(f"| **PRED-S3-A** | Spearman ρ(log₁₀R, D3) over held-out cells | ≥ 0.8, "
             f"≥ 6 cells | **{A['rho']:+.4f}** (p = {A['p']:.4f}), n = {A['n']} | {rr} |")
    rr = "**PASS**" if B["pass"] else "**FAIL**"
    L.append(f"| **PRED-S3-B** | C3(broad-S) > C3(narrow), both CH and P1 | strict, both "
             f"| CH {B['CH']['C3_narrow']:.4f} → {B['CH']['C3_broadS']:.4f}; "
             f"P1 {B['P1']['C3_narrow']:.4f} → {B['P1']['C3_broadS']:.4f} | {rr} |")
    L.append("")
    L.append(f"**VERDICT: {d['verdict']}**")
    (RESULTS / "TABLES.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) > 1 and _s.argv[1] == "tables":
        tables()
    else:
        main()
        tables()
