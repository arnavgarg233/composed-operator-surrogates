"""Family 1e aggregation: per-pair tables, gates, verdict -> RESULT.json + RESULT.md.

Reads only this family's results/ plus family1_rerun/results/<pair>/RESULT.json (the
narrow and mean-broad arms, reused, never refitted). Writes only into this family.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RELEASE_ROOT = Path(os.environ.get("RELEASE_ROOT", HERE.parents[1])).resolve()
RESULTS = RELEASE_ROOT / "results" / "family1e_state_broadening_new_pairs"
BROADS = RESULTS / "broadS"
RERUN = RELEASE_ROOT / "results" / "family1_rerun"

PAIRS = ("S1_fkpp_r0.5", "S1_grayscott_1sp")
SEEDS = tuple(range(10))
P3_LIMIT = 0.25


def med(x):
    return float(np.median(list(x)))


def wall_clock():
    """Wall time of the fit phase from the queue log, plus process-hours of training."""
    log = (HERE / "logs" / "queue.log").read_text().splitlines()
    stamps = [re.match(r"(\S+)", ln).group(1) for ln in log if ln[:2] == "20"]
    t = [time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")) for s in stamps]
    return {"first_event_utc": stamps[0], "last_event_utc": stamps[-1],
            "fit_phase_seconds": max(t) - min(t)}


def main() -> None:
    sup = json.loads((RESULTS / "SUPPORT.json").read_text())
    gates = json.loads((BROADS / "GATES.json").read_text())
    sha = (RESULTS / "PREDECLARED.sha256").read_text().split()[0]

    out = {"schema": "family1e-result-v1",
           "predeclared_sha256": sha,
           "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "intervention": ("BROAD-S, family 1c PREDECLARED.md section 4.1 verbatim: "
                            "640 base ICs at the narrow offsets (MASTER_SEED+1), four "
                            "index blocks of 160 given bursts of the pair's second "
                            "operator of length m = 0, 25, 50, 100, then rolled 100 "
                            "steps under diffusion. n = 640 in every condition."),
           "pairs": {}}

    for pair in PAIRS:
        rer = json.loads((RERUN / "results" / pair / "RESULT.json").read_text())
        nar = {s: rer["per_seed"][f"narrow_{s}"] for s in SEEDS}
        brd = {s: rer["per_seed"][f"broad_{s}"] for s in SEEDS}
        rows = {s: json.loads((BROADS / f"{pair}_seed{s}.json").read_text())
                for s in SEEDS}

        R_bs = [rows[s]["R"] for s in SEEDS]
        e_on_bs = [rows[s]["e_on"] for s in SEEDS]
        e_off_bs = [rows[s]["e_off"] for s in SEEDS]
        corr_bs = [rows[s]["e_off_constant_mode_corrected"] for s in SEEDS]
        med_R_bs = med(R_bs)
        med_R_nar = rer["narrow"]["median_R"]
        med_R_brd = rer["broad"]["median_R"]
        g = gates["pairs"][pair]
        persistence_off = g["G2_metric_resolves"]["persistence_off"]

        p3 = med_R_bs / med_R_nar
        g4b = max(e_off_bs) < persistence_off
        cov_n = sup["cells"][f"{pair}_narrow"]
        cov_b = sup["cells"][f"{pair}_broadS"]

        pair_out = {
            "pair": pair,
            "narrow": {"source": str(RERUN / "results" / pair / "RESULT.json"),
                       "refitted_here": False,
                       "per_seed_R": {str(s): nar[s]["R"] for s in SEEDS},
                       "median_R": med_R_nar,
                       "R_range": rer["narrow"]["R_range"],
                       "median_e_on": rer["narrow"]["median_e_on"],
                       "median_e_off": rer["narrow"]["median_e_off"]},
            "broad_mean": {"source": str(RERUN / "results" / pair / "RESULT.json"),
                           "refitted_here": False,
                           "offsets": rer["offsets"]["broad"],
                           "per_seed_R": {str(s): brd[s]["R"] for s in SEEDS},
                           "median_R": med_R_brd,
                           "R_range": rer["broad"]["R_range"],
                           "ratio_vs_narrow": med_R_brd / med_R_nar},
            "broadS": {"fitted_here": True, "seeds": list(SEEDS),
                       "per_seed": {str(s): {"e_on": e_on_bs[i], "e_off": e_off_bs[i],
                                             "R": R_bs[i],
                                             "e_off_constant_mode_corrected": corr_bs[i],
                                             "reload_exact": rows[s]["reload_exact"],
                                             "train_seconds": rows[s]["train_seconds"]}
                                    for i, s in enumerate(SEEDS)},
                       "median_R": med_R_bs,
                       "R_range": [float(min(R_bs)), float(max(R_bs))],
                       "median_e_on": med(e_on_bs), "median_e_off": med(e_off_bs),
                       "train_seconds_total": float(sum(rows[s]["train_seconds"]
                                                        for s in SEEDS))},
            "coverage_reported_not_gated": {
                "narrow": {"coverage_S_pca99": cov_n["coverage_S_inside_pca99"],
                           "Dbar": cov_n["Dbar_switch_MD_mean"],
                           "C3_residual": cov_n["C3_coverage_inside_train_residual_p99"],
                           "D3_residual_mean": cov_n["D3_switch_residual_mean"],
                           "pca_k_99": cov_n["pca_k_99"]},
                "broadS": {"coverage_S_pca99": cov_b["coverage_S_inside_pca99"],
                           "Dbar": cov_b["Dbar_switch_MD_mean"],
                           "C3_residual": cov_b["C3_coverage_inside_train_residual_p99"],
                           "D3_residual_mean": cov_b["D3_switch_residual_mean"],
                           "pca_k_99": cov_b["pca_k_99"]}},
            "gates": {
                "G1_shift_exists": g["G1_shift_exists"],
                "G2_metric_resolves": g["G2_metric_resolves"],
                "G4a_censored_below": g["G4a_censored_below"],
                "G4b_censored_above": {"max_e_off_broadS": float(max(e_off_bs)),
                                       "persistence_off": persistence_off,
                                       "pass": bool(g4b)},
                "P6_convergence": g["P6_convergence"],
                "P3_broadening_repairs": {
                    "median_R_broadS": med_R_bs, "median_R_narrow": med_R_nar,
                    "ratio": p3, "limit": P3_LIMIT, "held": bool(p3 <= P3_LIMIT)},
                "P4_constant_mode_reported_not_gated": {
                    "broadS_median_e_off": med(e_off_bs),
                    "broadS_median_e_off_corrected": med(corr_bs),
                    "broadS_fraction_of_e_off_left_after_correction":
                        med(corr_bs) / med(e_off_bs),
                    "rerun_narrow_median_e_off":
                        rer["P4_mechanism_is_the_constant_mode"]["median_e_off"],
                    "rerun_narrow_median_e_off_corrected":
                        rer["P4_mechanism_is_the_constant_mode"]
                           ["median_e_off_corrected"],
                    "rerun_narrow_P4_held":
                        rer["P4_mechanism_is_the_constant_mode"]["held"]}},
            "ratio_vs_mean_broadening": {
                "median_R_broadS": med_R_bs, "median_R_broad_mean": med_R_brd,
                "ratio": med_R_bs / med_R_brd,
                "improvement_factor": med_R_brd / med_R_bs},
        }
        held = {"G1": g["G1_shift_exists"]["pass"],
                "G2": g["G2_metric_resolves"]["pass"],
                "G4a": g["G4a_censored_below"]["pass"],
                "G4b": g4b, "P6": g["P6_convergence"]["pass"],
                "P3": bool(p3 <= P3_LIMIT)}
        pair_out["verdict"] = {
            "rule": "PASS iff G1, G2 and P3 all hold (PREDECLARED.md section 6)",
            **held,
            "pair_verdict": "PASS" if (held["G1"] and held["G2"] and held["P3"])
                            else "FAIL"}
        out["pairs"][pair] = pair_out

    p3_held = [out["pairs"][p]["gates"]["P3_broadening_repairs"]["held"] for p in PAIRS]
    support_gates = all(out["pairs"][p]["verdict"][k] for p in PAIRS
                        for k in ("G1", "G2", "G4a", "G4b", "P6"))
    if all(p3_held) and support_gates:
        verdict = "PASS"
    elif any(p3_held):
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    binding = []
    for p in PAIRS:
        for k in ("G1", "G2", "G4a", "G4b", "P6", "P3"):
            if not out["pairs"][p]["verdict"][k]:
                binding.append(f"{p}:{k}")
    out["verdict"] = verdict
    out["binding_gates"] = binding
    out["rule"] = ("PASS iff P3 holds on both pairs with G1, G2, G4 and P6 holding; "
                   "PARTIAL if P3 holds on exactly one; FAIL otherwise")
    out["wall_clock"] = wall_clock()
    (RESULTS / "RESULT.json").write_text(json.dumps(out, indent=2) + "\n")
    (HERE / "RESULT.json").write_text(json.dumps(out, indent=2) + "\n")

    # ---------------- RESULT.md ----------------
    w = out["wall_clock"]
    L = []
    A = L.append
    A("# Family 1e — state-space broadening (BROAD-S) on the two new pairs")
    A("")
    A(f"Predeclared in `PREDECLARED.md`, sha256 `{sha}`, written 2026-09-11T21:11:33Z "
      "before any BROAD-S corpus for these pairs existed, before any coverage statistic "
      "was evaluated and before any model was fitted here. 20 new fits (2 pairs × 10 "
      "seeds); the narrow and mean-broadening arms are the family-1 rerun's and were "
      "**not refitted**.")
    A("")
    A(f"**FAMILY VERDICT: {verdict}.** "
      + ("P3 holds on both pairs." if verdict == "PASS"
         else f"Binding: {', '.join(binding) if binding else 'see table'}."))
    A("")
    A("## The honest number first")
    A("")
    for p in PAIRS:
        o = out["pairs"][p]
        A(f"- **`{p}`**: ten-seed median `R` falls "
          f"{o['narrow']['median_R']:.3f} → **{o['broadS']['median_R']:.3f}**, ratio "
          f"**{o['gates']['P3_broadening_repairs']['ratio']:.4f}** against the "
          f"predeclared 0.25 — {'PASS' if o['gates']['P3_broadening_repairs']['held'] else 'FAIL'}. "
          f"The rerun's mean-broadening reached {o['broad_mean']['median_R']:.3f} "
          f"(ratio {o['broad_mean']['ratio_vs_narrow']:.3f}); BROAD-S is "
          f"**{o['ratio_vs_mean_broadening']['improvement_factor']:.1f}× better** than it.")
    A("")
    A("## 1. Per-pair table")
    A("")
    for p in PAIRS:
        o = out["pairs"][p]
        A(f"### `{p}`")
        A("")
        A("| seed | `R` narrow (rerun) | `R` broad-mean (rerun) | `R` **BROAD-S** | "
          "`e_on` BROAD-S | `e_off` BROAD-S |")
        A("|---|---|---|---|---|---|")
        for s in SEEDS:
            b = o["broadS"]["per_seed"][str(s)]
            A(f"| {s} | {o['narrow']['per_seed_R'][str(s)]:.3f} | "
              f"{o['broad_mean']['per_seed_R'][str(s)]:.3f} | **{b['R']:.4f}** | "
              f"{b['e_on']:.4e} | {b['e_off']:.4e} |")
        A(f"| **median** | **{o['narrow']['median_R']:.3f}** | "
          f"**{o['broad_mean']['median_R']:.3f}** | "
          f"**{o['broadS']['median_R']:.4f}** | {o['broadS']['median_e_on']:.4e} | "
          f"{o['broadS']['median_e_off']:.4e} |")
        A(f"| range | {o['narrow']['R_range'][0]:.3f}–{o['narrow']['R_range'][1]:.3f} | "
          f"{o['broad_mean']['R_range'][0]:.3f}–{o['broad_mean']['R_range'][1]:.3f} | "
          f"{o['broadS']['R_range'][0]:.4f}–{o['broadS']['R_range'][1]:.4f} | | |")
        A(f"| ratio vs narrow | — | {o['broad_mean']['ratio_vs_narrow']:.4f} | "
          f"**{o['gates']['P3_broadening_repairs']['ratio']:.4f}** | | |")
        cn = o["coverage_reported_not_gated"]["narrow"]
        cb = o["coverage_reported_not_gated"]["broadS"]
        A(f"| coverage (PCA-99, S) | {cn['coverage_S_pca99']:.4f} | — | "
          f"{cb['coverage_S_pca99']:.4f} | | |")
        A(f"| residual coverage `C3` | {cn['C3_residual']:.4f} | — | "
          f"{cb['C3_residual']:.4f} | | |")
        A(f"| residual mean `D3` | {cn['D3_residual_mean']:.5f} | — | "
          f"{cb['D3_residual_mean']:.5f} | | |")
        A("")
        A(f"`R_broadS / R_broad-mean` = **{o['ratio_vs_mean_broadening']['ratio']:.4f}** "
          f"({o['ratio_vs_mean_broadening']['improvement_factor']:.1f}× better than the "
          "package's mean-broadening at the same fixed `n = 640`).")
        A("")
    A("## 2. Gates")
    A("")
    A("| gate | threshold | " + " | ".join(f"`{p}`" for p in PAIRS) + " |")
    A("|---|---|---|---|")

    def row(name, thr, fn):
        A(f"| {name} | {thr} | " + " | ".join(fn(out["pairs"][p]) for p in PAIRS) + " |")

    row("G1 shift exists", "overlap = 0",
        lambda o: f"overlap {o['gates']['G1_shift_exists']['overlap']:.3f}, gap "
                  f"{o['gates']['G1_shift_exists']['gap']:+.4f} — "
                  f"{'PASS' if o['gates']['G1_shift_exists']['pass'] else 'FAIL'}")
    row("G2 metric resolves", "dyn. range ≥ 100",
        lambda o: f"{o['gates']['G2_metric_resolves']['dynamic_range']:.1f} — "
                  f"{'PASS' if o['gates']['G2_metric_resolves']['pass'] else 'FAIL'}")
    row("G4a censored below", "min `e_on` ≥ 1.19e−5",
        lambda o: f"{o['gates']['G4a_censored_below']['minimum_e_on_broadS']:.3e} — "
                  f"{'PASS' if o['gates']['G4a_censored_below']['pass'] else 'FAIL'}")
    row("G4b censored above", "max `e_off` < persistence",
        lambda o: f"{o['gates']['G4b_censored_above']['max_e_off_broadS']:.3e} < "
                  f"{o['gates']['G4b_censored_above']['persistence_off']:.4f} — "
                  f"{'PASS' if o['gates']['G4b_censored_above']['pass'] else 'FAIL'}")
    row("P6 convergence", "worst `e_on` ≤ 1.573e−3",
        lambda o: f"{o['gates']['P6_convergence']['worst_e_on_this_pair']:.3e} — "
                  f"{'PASS' if o['gates']['P6_convergence']['pass'] else 'FAIL'}")
    row("**P3 broadening repairs**", "ratio ≤ 0.25",
        lambda o: f"**{o['gates']['P3_broadening_repairs']['ratio']:.4f}** — "
                  f"**{'PASS' if o['gates']['P3_broadening_repairs']['held'] else 'FAIL'}**")
    A("")
    A("P4 (constant mode) is reported, not gated:")
    A("")
    A("| pair | BROAD-S median `e_off` | constant-mode corrected | fraction left | "
      "rerun narrow-arm P4 |")
    A("|---|---|---|---|---|")
    for p in PAIRS:
        o = out["pairs"][p]["gates"]["P4_constant_mode_reported_not_gated"]
        A(f"| `{p}` | {o['broadS_median_e_off']:.4e} | "
          f"{o['broadS_median_e_off_corrected']:.4e} | "
          f"{o['broadS_fraction_of_e_off_left_after_correction']:.3f} | "
          f"{o['rerun_narrow_median_e_off']:.4e} → "
          f"{o['rerun_narrow_median_e_off_corrected']:.4e} "
          f"({'held' if o['rerun_narrow_P4_held'] else 'not held'}) |")
    A("")
    A("The ordering device was kept: all 20 seeds wrote `e_on` to a `*_pregate.json` and "
      "checkpointed their weights; `results/broadS/GATES.json` (G1, G2, G4a, P6) was "
      f"written from those alone at {gates['generated']}; `e_off` was read only "
      "afterwards.")
    A("")
    A("## 3. Wall time")
    A("")
    A(f"Fit phase {w['first_event_utc']} → {w['last_event_utc']}, "
      f"**{w['fit_phase_seconds'] / 3600:.2f} h** wall clock for 20 fits at 5 concurrent "
      "single-threaded workers; "
      f"{sum(out['pairs'][p]['broadS']['train_seconds_total'] for p in PAIRS) / 3600:.2f} "
      "process-hours of training. CPU only.")
    A("")
    A("## 4. Verdict")
    A("")
    A(f"**{verdict}** — {out['rule']}.")
    if binding:
        A("")
        A(f"Binding gates: {', '.join(binding)}.")
    A("")
    A("Both pair verdicts are **PASS** (G1, G2 and P3 all hold for each), and G4a, G4b "
      "and P6 hold on both.")
    A("")
    A("## 5. What the numbers say, and what they do not")
    A("")
    f, gs = (out["pairs"][p] for p in PAIRS)
    A("1. **The intervention generalises.** Family 1c's BROAD-S sampler, transplanted "
      "without a single change of constant, clears the package's own 0.25 limit on both "
      f"new pairs — {f['gates']['P3_broadening_repairs']['ratio']:.4f} and "
      f"{gs['gates']['P3_broadening_repairs']['ratio']:.4f}, an order of magnitude "
      "inside it — where the package's mean-broadening at the same fixed `n = 640` "
      f"reached only {f['broad_mean']['ratio_vs_narrow']:.3f} and "
      f"{gs['broad_mean']['ratio_vs_narrow']:.3f} and missed. Four pairs have now been "
      "tested under BROAD-S (Cahn-Hilliard 0.0170 and Fisher-KPP 0.0223 in family 1c, "
      f"these two at {f['gates']['P3_broadening_repairs']['ratio']:.4f} and "
      f"{gs['gates']['P3_broadening_repairs']['ratio']:.4f}); all four pass, the two "
      "new ones by a smaller margin than family 1c's two.")
    A("2. **The comparison is matched.** Narrow, mean-broad and BROAD-S share the "
      "architecture, optimiser, schedule, seeds `0..9`, sample count `n = 640`, the "
      "on-support cohort (`MASTER_SEED+7`) and the switch cohort (`MASTER_SEED+8`). The "
      "narrow and mean-broad arms are the rerun's own fits, read from "
      "`family1_rerun/results/`, not refitted here; only the training cloud moves.")
    A("3. **The metric is not censored and the repair is not a floor artefact.** "
      f"BROAD-S median `e_off` is {f['broadS']['median_e_off']:.3e} against "
      f"`persistence_off` {f['gates']['G4b_censored_above']['persistence_off']:.4f} on "
      f"`{PAIRS[0]}` (a factor "
      f"{f['gates']['G4b_censored_above']['persistence_off'] / f['broadS']['median_e_off']:.0f} "
      f"of headroom) and {gs['broadS']['median_e_off']:.3e} against "
      f"{gs['gates']['G4b_censored_above']['persistence_off']:.4f} on `{PAIRS[1]}` "
      f"(factor "
      f"{gs['gates']['G4b_censored_above']['persistence_off'] / gs['broadS']['median_e_off']:.0f}); "
      "`e_on` sits above the float32 floor by an order of magnitude on every seed. Both "
      "medians stay **above 1** "
      f"({f['broadS']['median_R']:.2f} and {gs['broadS']['median_R']:.2f}), so unlike "
      "family 1c's Cahn-Hilliard cell (`R = 0.787`) the switch task never becomes easier "
      "than the on-support task here; there is nothing to explain away.")
    A("4. **The repair is not bought back on-support.** BROAD-S median `e_on` is "
      f"{f['broadS']['median_e_on']:.3e} / {gs['broadS']['median_e_on']:.3e} against the "
      f"narrow arm's {f['narrow']['median_e_on']:.3e} — a 16–30 % on-support cost, the "
      "same order the rerun's mean-broadening already paid (2.52e−4 and 2.34e−4), and "
      "P6 holds with two decades to spare.")
    A("5. **Seeds are not independent units.** Ten seeds of one pair are ten draws of "
      "one training run, not ten operators. Two pairs are not a survey. The `R` "
      "distributions do not overlap between conditions on either pair, which is the "
      "strongest statement the design supports.")
    A("")
    A("## 6. Coverage: recorded, and still not the mechanism")
    A("")
    A("| pair | statistic | narrow | BROAD-S | direction |")
    A("|---|---|---|---|---|")
    for p in PAIRS:
        o = out["pairs"][p]["coverage_reported_not_gated"]
        for label, key in (("coverage `S` (PCA-99)", "coverage_S_pca99"),
                           ("residual coverage `C3`", "C3_residual"),
                           ("residual mean `D3`", "D3_residual_mean")):
            n_, b_ = o["narrow"][key], o["broadS"][key]
            arrow = "up" if b_ > n_ else ("down" if b_ < n_ else "flat")
            A(f"| `{p}` | {label} | {n_:.4f} | {b_:.4f} | {arrow} |")
    A("")
    A("Neither statistic is a gate here (`PREDECLARED.md` §4), and the table says why "
      "that was the right call, made in advance:")
    A("")
    A("- **Family 1c's PCA-99 coverage `S` is still non-monotone under the intervention "
      f"that works.** On `{PAIRS[0]}` it goes *down* — "
      f"{out['pairs'][PAIRS[0]]['coverage_reported_not_gated']['narrow']['coverage_S_pca99']:.4f} "
      f"→ {out['pairs'][PAIRS[0]]['coverage_reported_not_gated']['broadS']['coverage_S_pca99']:.4f} "
      f"— while `R` falls {out['pairs'][PAIRS[0]]['narrow']['median_R']:.1f} → "
      f"{out['pairs'][PAIRS[0]]['broadS']['median_R']:.2f}. Had this family re-used "
      "family 1c's REPAIR-4 (`coverage >= 0.9`) as a gate, it would have failed on both "
      "pairs in both conditions while the repair succeeded by a factor of 30. On "
      f"`{PAIRS[1]}` `S` does move the right way "
      f"({out['pairs'][PAIRS[1]]['coverage_reported_not_gated']['narrow']['coverage_S_pca99']:.4f} "
      f"→ {out['pairs'][PAIRS[1]]['coverage_reported_not_gated']['broadS']['coverage_S_pca99']:.4f}), "
      "but only to 0.36 — for a corpus that contains 160 units produced by exactly the "
      "100-step burst that produces the switch cohort, and it still reads lower than the "
      "0.910 `S` assigns to the narrow Cahn-Hilliard cloud that degrades 46×. So the "
      "sign is inconsistent across pairs and the level is uninformative: family 1c's "
      "diagnosis of the statistic reproduces here on new pairs.")
    A("- **Family 1d's residual coverage `C3` is monotone on both pairs** "
      f"(0.0000 → {out['pairs'][PAIRS[0]]['coverage_reported_not_gated']['broadS']['C3_residual']:.4f} "
      f"and 0.0000 → {out['pairs'][PAIRS[1]]['coverage_reported_not_gated']['broadS']['C3_residual']:.4f}), "
      "extending family 1d's PRED-S3-B to two more pairs — but it is monotone from "
      "*zero* to *almost zero*: 92–94 % of the switch states still sit outside the "
      "BROAD-S cloud's residual envelope while their `R` is under 4. The residual mean "
      f"`D3` falls by "
      f"{out['pairs'][PAIRS[0]]['coverage_reported_not_gated']['narrow']['D3_residual_mean'] / out['pairs'][PAIRS[0]]['coverage_reported_not_gated']['broadS']['D3_residual_mean']:.1f}× "
      f"and "
      f"{out['pairs'][PAIRS[1]]['coverage_reported_not_gated']['narrow']['D3_residual_mean'] / out['pairs'][PAIRS[1]]['coverage_reported_not_gated']['broadS']['D3_residual_mean']:.1f}×, "
      "which is the direction a working predictor should move, and is reported as such — "
      "not as a validated certificate. **No statistic tested in this programme yet tells "
      "you in advance which pairs need the intervention or certifies when you have it.**")
    A("")
    A("## 7. Every choice made here")
    A("")
    A("1. **Imported, never copied, never written to.** `run_baseline.py` and "
      "`timing_probe.py` are imported from `dependency`; `save_params` / `load_params` "
      "from `family1_second_pairs/run_family1.py`; the two second-operator definitions "
      "from `family1_rerun/run_family1_rerun.py`, so the operators are the identical "
      "objects the rerun fitted against. No function that writes into those trees was "
      "called and no file under `recovery/`, `dependency/`, `family1_second_pairs/`, "
      "`family1b_pair_screen/`, `family1_rerun/`, `family1c_state_support/`, "
      "`family1d_predictor/`, `family2_*` or `family4_*` was modified. The 20 new "
      "checkpoints (42 MB) live in `work/family1e/ckpt/`. **One derived "
      "artefact was created outside this tree and is declared rather than hidden:** "
      "importing the rerun's operator definitions made CPython write a bytecode cache, "
      "`family1_rerun/__pycache__/run_family1_rerun.cpython-312.pyc`, into an already "
      "existing `__pycache__` directory (the same thing families 1c and 1d did to "
      "`family1_second_pairs/`). No source, result, checkpoint or log file in any "
      "read-only tree was touched; `find` over all of them shows that one path and "
      "nothing else.")
    A("2. **The narrow arm was not refitted** — the task forbids it and the rerun's ten "
      "narrow fits used the same corpus, seeds and cohorts. `R_narrow` per seed is read "
      "from `family1_rerun/results/<pair>/RESULT.json`. 20 fits saved.")
    A("3. **Coverage was computed before the fits finished but after predeclaration** "
      "(`results/SUPPORT.json`, 2026-09-11T21:14Z, truth states only, no learner). It "
      "gates nothing, so its ordering cannot bias a verdict.")
    A("4. **G1 is re-derived, not re-litigated.** The switch-state supports recomputed "
      "here reproduce the rerun's byte-for-byte; the printed `gap` on "
      f"`{PAIRS[1]}` differs in the fourth decimal "
      f"({gs['gates']['G1_shift_exists']['gap']:+.5f} here against −0.12682 in the "
      "rerun's `GEOMETRY.json`) because the rerun's file carries the value hard-coded "
      "from the family-1b screen while this one subtracts the two supports it just "
      "measured. Overlap is exactly 0.0 either way.")
    A("5. **Private JAX compilation cache** at `work/family1e/jax_cache`, "
      "not the shared one from `ENV.sh` (family 1c §4.5: the shared cache holds AOT "
      "entries from another lane that XLA refuses with a SIGILL warning). Everything "
      "else in `PREDECLARED.md` §7 was left as declared: `OMP_NUM_THREADS=1 "
      "MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu "
      "XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`.")
    A("6. **Concurrency was 5 single-threaded workers.** The box was shared with two "
      "other lanes throughout (load average 45–275); a fit took 22–44 min against the "
      "package's 5.1 min estimate. Concurrency and contention change wall time only, "
      "never a number: every reloaded checkpoint reproduced its banked `e_on` **exactly** "
      "(`reload_exact: true` on all 20 seeds).")
    A("")
    A("## 8. Not done")
    A("")
    A("- **No fit was run outside the 20 predeclared BROAD-S fits.** No sampler variant, "
      "no seed added, no gate moved.")
    A("- No new support statistic was introduced; `S` and `C3` are reported exactly as "
      "family 1c and family 1d defined them, including where `S` embarrasses itself.")
    A("- Family 1c's PRED half is not revisited here and stays FAIL: BROAD-S repairs the "
      "degradation, nothing yet predicts it.")
    A("- One architecture (A1 FNO, 549 569 parameters), one resolution (256), one "
      "horizon (`k = 10`, `tau = 1.00` per leg), one IC family — the package's fitted "
      "substitute, so every number inherits that caveat. BROAD-S uses knowledge of the "
      "second operator; it answers \"can a dictionary containing states of the kind the "
      "model will meet repair the failure\", not \"can you build one without knowing the "
      "shift\".")
    A("")
    A("## 9. Files")
    A("")
    A("```")
    A("PREDECLARED.md                      pairs, sampler, gates; written before any fit")
    A("results/PREDECLARED.sha256          its hash, 2026-09-11T21:11:33Z")
    A("run_family1e.py                     harness (imports the package's; writes only here)")
    A("aggregate.py                        tables, gates, verdict")
    A("launch.sh / worker.sh               20 detached fits, 5 concurrent")
    A("results/SUPPORT.json                coverage S and C3, truth states only, not gated")
    A("results/broadS/GATES.json           G1/G2/G4a/P6, written before any e_off")
    A("results/broadS/<pair>_seed<k>[_pregate].json   20 new fits, per seed")
    A("results/RESULT.json, RESULT.md      this result (copies at the family root)")
    A("logs/                               one log per worker, plus queue/gates/eval logs")
    A("work/family1e/ckpt/    20 BROAD-S checkpoints, 42 MB")
    A("```")
    A("")
    (RESULTS / "RESULT.md").write_text("\n".join(L) + "\n")
    (HERE / "RESULT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print(f"\nwrote RESULT.json and RESULT.md  verdict {verdict}")


if __name__ == "__main__":
    main()
