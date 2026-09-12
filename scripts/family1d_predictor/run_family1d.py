"""Family 1d -- the out-of-basis residual S3 as a degradation predictor, out of sample.

Everything here is fixed by PREDECLARED.md (sha256 in results/PREDECLARED.sha256, taken
at 2026-09-11T11:46:11Z, before any number in this family was computed).

PROVENANCE. Nothing under recovery/, dependency/, family1_second_pairs/,
family1b_pair_screen/, family1c_state_support/, family6_serrano_baseline/ or
family1_rerun/ is written to. The shipped harness and family 1c's corpora are IMPORTED:

    dependency/scripts/baseline/run_baseline.py   -> rb
    dependency/scripts/baseline/timing_probe.py   -> tp
    family1_second_pairs/run_family1.py            -> f1  (second_operator, save/load)
    family1c_state_support/run_family1c.py         -> c   (train_states_for, eval_cohorts,
                                                           BROAD-S sampler, OFFSETS)

New code here: the S3/C3 statistic (a verbatim re-implementation of family 1c's
diagnostic.py, PREDECLARED.md section 1), the S1_fkpp_r0.5 cells, and the aggregation.

Subcommands:
    geom                    -> results/heldout/FK05_GEOMETRY.json   (truth only, no learner)
    train <seed>            -> results/heldout/FK05_broad_seed<k>_pregate.json + ckpt
    gates                   -> results/heldout/FK05_GATES.json      (before any e_off)
    eval <condition>        -> results/heldout/FK05_<condition>.json
    stats                   -> results/STATISTICS.json              (S3/C3/Mahalanobis)
    aggregate               -> RESULT.json + RESULT.md
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RELEASE_ROOT = Path(os.environ.get("RELEASE_ROOT", HERE.parents[1])).resolve()
PKG = RELEASE_ROOT / "dependency"
F1 = RELEASE_ROOT / "dependency" / "family1_second_pairs"
F1C = RELEASE_ROOT / "results" / "family1c_state_support"
F6 = RELEASE_ROOT / "results" / "family6_serrano_baseline"
RERUN = RELEASE_ROOT / "results" / "family1_rerun"
WORK = RELEASE_ROOT / "work" / "family1d"

sys.path.insert(0, str(PKG / "scripts" / "baseline"))
sys.path.insert(0, str(PKG / "scripts" / "band_check"))
sys.path.insert(0, str(F1))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

_spec = importlib.util.spec_from_file_location("rf1c", F1C / "run_family1c.py")
c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c)                      # read-only use; writes nothing on import

tp = c.tp
rb = c.rb
f1 = c.f1
ex = f1.ex

RESULTS = RELEASE_ROOT / "results" / "family1d_predictor"
HELD = RESULTS / "heldout"
CKPT = WORK / "ckpt"

# ---- PREDECLARED.md section 2(d) -------------------------------------------------
FK05 = "S1_fkpp_r0.5"
FK05_OFFSETS = {"narrow": (0.4865, 0.5171), "broad": (0.48, 0.70)}
FK05_BROAD_SEEDS = (0, 1, 2, 3, 4)
FK05_NARROW_SEEDS = tuple(range(10))

# The six cells S3 was selected on (PREDECLARED.md section 1.2) -- reported, never gated.
SELECTION_CELLS = [("P1", "narrow"), ("P1", "broad"), ("P2", "narrow"),
                   ("P2", "broad"), ("CH", "narrow"), ("CH", "broad")]


def fk05_operator():
    """Verbatim from family1_rerun/run_family1_rerun.py:second_operator (pair A) and
    family1b_pair_screen/screen_pairs.py: FisherKPP(diffusivity=0.0, reactivity=0.5)."""
    return ex.stepper.reaction.FisherKPP(
        tp.NUM_SPATIAL_DIMS, tp.DOMAIN_EXTENT, tp.NUM_POINTS, tp.DT,
        diffusivity=0.0, reactivity=0.5, order=tp.ETDRK_ORDER,
        dealiasing_fraction=tp.DEALIASING_FRACTION,
        num_circle_points=tp.NUM_CIRCLE_POINTS, circle_radius=tp.CIRCLE_RADIUS)


def fk05_train_states(condition: str):
    ic = tp.initial_conditions(tp.NUM_TRAIN_UNITS, FK05_OFFSETS[condition],
                               tp.MASTER_SEED + 1)
    return tp.rollout(c.diffusion(), ic, tp.STEPS_PER_TRAJECTORY)


def fk05_cohorts():
    """Same construction as run_family1c.eval_cohorts, with the r=0.5 second operator."""
    d = c.diffusion()
    on_ic = tp.initial_conditions(rb.EVAL_UNITS, rb.NARROW, tp.MASTER_SEED + 7)
    on_states = tp.rollout(d, on_ic, tp.STEPS_PER_TRAJECTORY)[:, 0]
    on_target = tp.rollout(d, on_states, rb.K)[:, -1]
    sw_ic = tp.initial_conditions(rb.EVAL_UNITS, rb.NARROW, tp.MASTER_SEED + 8)
    switch_states = tp.rollout(fk05_operator(), sw_ic, tp.STEPS_PER_TRAJECTORY)[:, -1]
    switch_target = tp.rollout(d, switch_states, rb.K)[:, -1]
    return dict(on_states=on_states, on_target=on_target,
                switch_states=switch_states, switch_target=switch_target)


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def med_rel(a, b) -> float:
    return float(np.median(rb.relative_l2(np.asarray(a, np.float64),
                                          np.asarray(b, np.float64))))


# ===========================================================================
# geom -- truth states only. No learner. Written before any e_off.
# ===========================================================================
def geom() -> None:
    started = time.perf_counter()
    co = fk05_cohorts()
    train_narrow = fk05_train_states("narrow")
    train_broad = fk05_train_states("broad")

    def support(states):
        m = np.asarray(states.mean(axis=(2, 3)), np.float64).reshape(-1)
        return float(m.min()), float(m.max())

    sup_n = support(train_narrow)
    sup_b = support(train_broad)
    sw_means = np.asarray(co["switch_states"].mean(axis=(1, 2)), np.float64).reshape(-1)
    sw_sup = (float(sw_means.min()), float(sw_means.max()))
    out = {
        "schema": "family1d-fk05-geometry-v1",
        "status": "truth states only; no learner; written before any e_off is read",
        "pair": FK05,
        "second_operator": "exponax FisherKPP(diffusivity=0.0, reactivity=0.5)",
        "offsets": {k: list(v) for k, v in FK05_OFFSETS.items()},
        "narrow_training_support": list(sup_n),
        "broad_training_support": list(sup_b),
        "switch_support": list(sw_sup),
        "narrow_overlap": max(0.0, min(sup_n[1], sw_sup[1]) - max(sup_n[0], sw_sup[0])),
        "broad_overlap": max(0.0, min(sup_b[1], sw_sup[1]) - max(sup_b[0], sw_sup[0])),
        "persistence_on": med_rel(co["on_states"], co["on_target"]),
        "persistence_off": med_rel(co["switch_states"], co["switch_target"]),
        "screen_reference": {
            "source": "family1b_pair_screen/RESULT.json intervals/S1_fkpp_r0.5",
            "switch_support": [0.5941062569618225, 0.6244125366210938],
            "persistence_off": 0.10153809212755065},
        "elapsed_seconds": time.perf_counter() - started,
    }
    HELD.mkdir(parents=True, exist_ok=True)
    (HELD / "FK05_GEOMETRY.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


# ===========================================================================
# train -- FK05 broad, one seed. e_on and the checkpoint first; e_off never here.
# ===========================================================================
def train(seed: int) -> None:
    started = time.perf_counter()
    ts = fk05_train_states("broad")
    pairs_in, pairs_out = rb.make_pairs(ts)
    del ts
    g = c.grid()
    t0 = time.perf_counter()
    params = rb.train(pairs_in, pairs_out, g, seed)          # shipped fit, unchanged
    train_seconds = time.perf_counter() - t0
    del pairs_in, pairs_out

    d = c.diffusion()
    on_ic = tp.initial_conditions(rb.EVAL_UNITS, rb.NARROW, tp.MASTER_SEED + 7)
    on_states = tp.rollout(d, on_ic, tp.STEPS_PER_TRAJECTORY)[:, 0]
    on_target = tp.rollout(d, on_states, rb.K)[:, -1]
    e_on = float(np.median(rb.evaluate(params, on_states, on_target, g)))

    CKPT.mkdir(parents=True, exist_ok=True)
    f1.save_params(CKPT / f"FK05_broad_seed{seed}.npz", params)
    HELD.mkdir(parents=True, exist_ok=True)
    (HELD / f"FK05_broad_seed{seed}_pregate.json").write_text(json.dumps({
        "schema": "family1d-fk05-pregate-v1",
        "pair": FK05, "condition": "broad", "seed": seed,
        "offsets": list(FK05_OFFSETS["broad"]),
        "e_on": e_on, "train_seconds": train_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "steps": rb.STEPS, "batch": rb.BATCH,
        "note": "e_off is deliberately NOT in this file; gates are written first",
        "platform": {"python": sys.version.split()[0], "machine": platform.machine(),
                     "jax": jax.__version__},
    }, indent=2) + "\n")
    print(f"FK05 broad seed {seed}  e_on {e_on:.6e}  {train_seconds / 60:.1f} min train",
          flush=True)


# ===========================================================================
# gates -- written before any off-support error is read, for both FK05 cells.
# ===========================================================================
def gates() -> None:
    geo = json.loads((HELD / "FK05_GEOMETRY.json").read_text())

    # narrow arm: family 1's ten banked narrow checkpoints, sha256-verified identical.
    a = F1 / "ckpt" / "P1_diffusion_fisher_kpp" / "seed0_narrow.npz"
    b = F1 / "ckpt" / "P2_diffusion_allen_cahn" / "seed0_narrow.npz"
    e = F1 / "ckpt" / "P3_negative_control" / "seed0_narrow.npz"
    reuse = {"P1_seed0": sha(a), "P2_seed0": sha(b), "CH_seed0": sha(e)}
    reuse["identical"] = reuse["P1_seed0"] == reuse["P2_seed0"] == reuse["CH_seed0"]
    if not reuse["identical"]:
        raise SystemExit("narrow checkpoints differ across pairs; refusing to reuse")

    co = fk05_cohorts()
    g = c.grid()
    e_on_narrow = {}
    for s in FK05_NARROW_SEEDS:
        p = F1 / "ckpt" / "P2_diffusion_allen_cahn" / f"seed{s}_narrow.npz"
        params = f1.load_params(p, s)
        e_on_narrow[s] = float(np.median(rb.evaluate(params, co["on_states"],
                                                     co["on_target"], g)))
    e_on_broad = {s: json.loads((HELD / f"FK05_broad_seed{s}_pregate.json").read_text())
                  ["e_on"] for s in FK05_BROAD_SEEDS}

    worst = max(list(e_on_narrow.values()) + list(e_on_broad.values()))
    med_narrow = float(np.median(list(e_on_narrow.values())))
    payload = {
        "schema": "family1d-fk05-gates-v1",
        "written_before_any_off_support_number_was_read": True,
        "narrow_reuse_checksum": reuse,
        "per_seed_e_on_narrow": {str(k): v for k, v in e_on_narrow.items()},
        "per_seed_e_on_broad": {str(k): v for k, v in e_on_broad.items()},
        "G1_shift_exists": {"narrow_overlap": geo["narrow_overlap"],
                            "pass": bool(geo["narrow_overlap"] == 0.0)},
        "G2_metric_resolves": {
            "persistence_off": geo["persistence_off"],
            "persistence_on": geo["persistence_on"],
            "median_e_on_narrow": med_narrow,
            "dynamic_range": geo["persistence_off"] / med_narrow,
            "required": rb.G2_MIN_DYNAMIC_RANGE,
            "e_on_below_tenth_of_persistence_on":
                bool(med_narrow <= 0.1 * geo["persistence_on"]),
            "pass": bool(geo["persistence_off"] / med_narrow >= rb.G2_MIN_DYNAMIC_RANGE
                         and med_narrow <= 0.1 * geo["persistence_on"])},
        "P6_convergence": {"worst_e_on": worst, "limit": rb.P6_MAX_E_ON,
                           "pass": bool(worst <= rb.P6_MAX_E_ON)},
    }
    (HELD / "FK05_GATES.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items()
                      if k.startswith(("G", "P6"))}, indent=2))


# ===========================================================================
# eval -- e_off, only after GATES.json exists.
# ===========================================================================
def evaluate_condition(condition: str) -> None:
    if not (HELD / "FK05_GATES.json").exists():
        raise SystemExit("FK05_GATES.json absent: refusing to read the off-support error")
    gt = json.loads((HELD / "FK05_GATES.json").read_text())
    co = fk05_cohorts()
    g = c.grid()
    rows = {}
    if condition == "narrow":
        seeds, key = FK05_NARROW_SEEDS, "per_seed_e_on_narrow"
        paths = {s: F1 / "ckpt" / "P2_diffusion_allen_cahn" / f"seed{s}_narrow.npz"
                 for s in seeds}
        note = ("R read from the ten narrow models family 1 already fitted (the narrow "
                "corpus does not depend on the second operator; sha256-verified "
                "identical across pairs). NO model was fitted for this cell and nothing "
                "was written into family1_second_pairs/.")
    else:
        seeds, key = FK05_BROAD_SEEDS, "per_seed_e_on_broad"
        paths = {s: CKPT / f"FK05_broad_seed{s}.npz" for s in seeds}
        note = "5 new fits, this family, seeds 0-4."
    for s in seeds:
        params = f1.load_params(paths[s], s)
        e_on = gt[key][str(s)]
        e_on_reload = float(np.median(rb.evaluate(params, co["on_states"],
                                                  co["on_target"], g)))
        e_off = float(np.median(rb.evaluate(params, co["switch_states"],
                                            co["switch_target"], g)))
        rows[str(s)] = {"e_on": e_on, "e_on_reloaded": e_on_reload,
                        "reload_exact": bool(e_on_reload == e_on),
                        "e_off": e_off, "R": e_off / e_on,
                        "checkpoint": str(paths[s])}
        print(f"FK05 {condition} seed {s}  e_on {e_on:.6e}  e_off {e_off:.6e}  "
              f"R {e_off / e_on:.3f}", flush=True)
    Rs = [rows[str(s)]["R"] for s in seeds]
    geo = json.loads((HELD / "FK05_GEOMETRY.json").read_text())
    out = {"schema": "family1d-fk05-eval-v1", "pair": FK05, "condition": condition,
           "note": note, "seeds": list(seeds), "per_seed": rows,
           "offsets": list(FK05_OFFSETS[condition]),
           "median_R": float(np.median(Rs)), "R_range": [float(min(Rs)), float(max(Rs))],
           "median_e_on": float(np.median([rows[str(s)]["e_on"] for s in seeds])),
           "median_e_off": float(np.median([rows[str(s)]["e_off"] for s in seeds])),
           "persistence_off": geo["persistence_off"],
           "max_e_off_below_persistence_off":
               bool(max(rows[str(s)]["e_off"] for s in seeds) < geo["persistence_off"])}
    (HELD / f"FK05_{condition}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"median R {out['median_R']:.3f}  range {out['R_range']}", flush=True)


# ===========================================================================
# stats -- S3/C3 (PREDECLARED section 1) and the PCA-99 Mahalanobis contrast.
# ===========================================================================
def _cloud_stats(train_states, switch_states) -> dict:
    x = np.asarray(train_states, np.float64).reshape(-1, tp.NUM_POINTS)
    mu = x.mean(axis=0)
    xc = x - mu
    n = x.shape[0]
    cov = (xc.T @ xc) / (n - 1)
    lam, vec = np.linalg.eigh(cov)
    order = np.argsort(lam)[::-1]
    lam, vec = lam[order], vec[:, order]
    cum = np.cumsum(lam) / lam.sum()
    k = int(np.searchsorted(cum, 0.99) + 1)
    u = vec[:, :k]

    # --- S3 (out-of-basis residual) -------------------------------------------
    res_tr = np.linalg.norm(xc - (xc @ u) @ u.T, axis=1)
    q99 = float(np.percentile(res_tr, 99))
    s = np.asarray(switch_states, np.float64).reshape(-1, tp.NUM_POINTS)
    sc = s - mu
    res_sw = np.linalg.norm(sc - (sc @ u) @ u.T, axis=1)
    tot = np.linalg.norm(sc, axis=1)

    # --- S (PCA-99 Mahalanobis), family 1c's failed statistic, for contrast ----
    def md(z):
        return np.sqrt(((z @ u) ** 2 / lam[:k]).sum(axis=1))
    md_tr = md(xc)
    r99 = float(np.percentile(md_tr, 99))
    md_sw = md(sc)

    return {
        "n_train_states": int(n), "n_switch_states": int(s.shape[0]),
        "pca_k_99": k, "variance_explained_at_k": float(cum[k - 1]),
        "train_residual_p99": q99,
        "D3_switch_residual_mean": float(res_sw.mean()),
        "C3_coverage_inside_train_residual_p99": float((res_sw <= q99).mean()),
        "D3f_switch_fraction_outside_basis_median": float(np.median(res_sw / tot)),
        "switch_residual_over_train_p99_median": float(np.median(res_sw) / q99),
        "switch_residual_median": float(np.median(res_sw)),
        "train_MD_p99_radius": r99,
        "Dbar_switch_MD_mean": float(md_sw.mean()),
        "coverage_S_inside_pca99": float((md_sw <= r99).mean()),
        "switch_MD_median": float(np.median(md_sw)),
    }


def stats() -> None:
    started = time.perf_counter()
    out = {"schema": "family1d-statistics-v1",
           "definition": "PREDECLARED.md section 1 (verbatim re-implementation of "
                         "family1c_state_support/diagnostic.py lines 40-64)",
           "cells": {}}

    # switch cohorts, one per second operator
    switch = {p: c.eval_cohorts(p)["switch_states"] for p in ("P1", "P2", "CH")}
    switch["FK05"] = fk05_cohorts()["switch_states"]

    # (pair-key, condition, cloud-builder, switch-key)
    groups = [
        ("P1", "narrow", lambda: c.train_states_for("P1", "narrow"), ["P1", "P2", "CH"]),
        ("P1", "broad", lambda: c.train_states_for("P1", "broad"), ["P1"]),
        ("P2", "broad", lambda: c.train_states_for("P2", "broad"), ["P2"]),
        ("CH", "broad", lambda: c.train_states_for("CH", "broad"), ["CH"]),
        ("CH", "broadS", lambda: c.train_states_for("CH", "broadS"), ["CH"]),
        ("P1", "broadS", lambda: c.train_states_for("P1", "broadS"), ["P1"]),
        ("FK05", "broad", lambda: fk05_train_states("broad"), ["FK05"]),
    ]
    for _, condition, build, sw_keys in groups:
        ts = build()
        for sk in sw_keys:
            name = f"{sk}_{condition}"
            out["cells"][name] = _cloud_stats(ts, switch[sk])
            r = out["cells"][name]
            print(f"{name:14s} k={r['pca_k_99']:3d}  D3 {r['D3_switch_residual_mean']:9.5f}"
                  f"  C3 {r['C3_coverage_inside_train_residual_p99']:.5f}"
                  f"  Dbar {r['Dbar_switch_MD_mean']:8.4f}"
                  f"  covS {r['coverage_S_inside_pca99']:.4f}", flush=True)
        del ts
    # the narrow cloud is shared by every pair, including FK05
    out["cells"]["FK05_narrow"] = out["cells"]["P1_narrow"].copy()
    ts = c.train_states_for("P1", "narrow")
    out["cells"]["FK05_narrow"] = _cloud_stats(ts, switch["FK05"])
    del ts
    r = out["cells"]["FK05_narrow"]
    print(f"{'FK05_narrow':14s} k={r['pca_k_99']:3d}  D3 {r['D3_switch_residual_mean']:9.5f}"
          f"  C3 {r['C3_coverage_inside_train_residual_p99']:.5f}"
          f"  Dbar {r['Dbar_switch_MD_mean']:8.4f}"
          f"  covS {r['coverage_S_inside_pca99']:.4f}", flush=True)

    out["elapsed_seconds"] = time.perf_counter() - started
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "STATISTICS.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote STATISTICS.json ({out['elapsed_seconds']:.1f} s)")


def main(argv) -> None:
    if not argv:
        raise SystemExit(__doc__)
    cmd = argv[0]
    if cmd == "geom":
        geom()
    elif cmd == "train":
        train(int(argv[1]))
    elif cmd == "gates":
        gates()
    elif cmd == "eval":
        evaluate_condition(argv[1])
    elif cmd == "stats":
        stats()
    elif cmd == "aggregate":
        import aggregate as agg
        agg.main()
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
