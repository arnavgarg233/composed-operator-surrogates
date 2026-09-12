"""Family 1c — support as multi-dimensional state coverage, not spatial mean.

Two halves, both fixed in PREDECLARED.md (sha256 in results/PREDECLARED.sha256) before
any number in this family was computed:

  PRED    no new fits. Compute, per (pair, condition) cell, the mean switch-state
          Mahalanobis distance in the PCA-99 basis of that model's OWN training cloud,
          and the model's R. Gate on the Spearman correlation of log10 R against that
          distance over six cells, plus inside/outside of the PCA-99 radius.
  REPAIR  20 new fits. A BROAD-S dictionary at fixed n=640 whose training-state cloud
          covers the switch states in the PCA basis (sampler in PREDECLARED.md 4.1),
          fitted on diffusion/Cahn-Hilliard (intervention) and on diffusion/Fisher-KPP
          (positive control).

ATTRIBUTION / PROVENANCE
------------------------
Nothing under dependency/, family1_second_pairs/, family1b_pair_screen/,
family1_rerun/ or recovery/ is written to. The shipped harness is IMPORTED so the code
that fits a model here is byte-for-byte the code that fits a model there:

    dependency/scripts/baseline/run_baseline.py  -> rb  (train, evaluate, relative_l2,
        make_pairs, K, STEPS, BATCH, EVAL_UNITS, NARROW, BROAD, gate thresholds)
    dependency/scripts/baseline/timing_probe.py  -> tp  (physical constants, corpus,
        FNO, hand-written Adam)
    family1_second_pairs/run_family1.py           -> f1  (second_operator(),
        save_params/load_params) -- imported read-only; no function that writes to that
        tree is ever called.

New code here: the PCA-99 Mahalanobis / spectral-band statistics, the BROAD-S sampler,
and the aggregation.

Run with:
    source environment setup script
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu \
    XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
    .venv/bin/python run_family1c.py <cmd> ...

Subcommands:
    eval_banked <pair> <cond>   -> results/banked/<pair>_<cond>.json   (10 seeds, no fit)
    support                     -> results/SUPPORT.json                (truth states only)
    broads_train <pair> <seed>  -> results/broadS/<pair>_seed<k>_pregate.json + ckpt
    broads_gates                -> results/broadS/GATES.json           (before any e_off)
    broads_eval  <pair> <seed>  -> results/broadS/<pair>_seed<k>.json
    aggregate                   -> RESULT.json + RESULT.md
"""

from __future__ import annotations

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
WORK = RELEASE_ROOT / "work" / "family1c"

sys.path.insert(0, str(PKG / "scripts" / "baseline"))
sys.path.insert(0, str(PKG / "scripts" / "band_check"))
sys.path.insert(0, str(F1))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import timing_probe as tp  # noqa: E402
import run_baseline as rb  # noqa: E402
import run_family1 as f1  # noqa: E402  (read-only use: second_operator, load/save params)

RESULTS = RELEASE_ROOT / "results" / "family1c_state_support"
CKPT = WORK / "ckpt"

SEEDS = tuple(range(10))

# The three pairs, and the family-1 tree each one's banked artefacts live in.
PAIR_KEY = {
    "P1": "P1_diffusion_fisher_kpp",
    "P2": "P2_diffusion_allen_cahn",
    "CH": "P3_negative_control",
}
# Training offsets per cell -- PREDECLARED.md section 3 table, from
# family1_second_pairs/results/INTERVALS.json and run_baseline.py.
OFFSETS = {
    ("P1", "narrow"): (0.4865, 0.5171),
    ("P1", "broad"): (0.4865, 0.7258),
    ("P2", "narrow"): (0.4865, 0.5171),
    ("P2", "broad"): (0.48, 0.90),
    ("CH", "narrow"): (0.4865, 0.5171),
    ("CH", "broad"): (0.48, 0.60),
}
CELLS = [("P1", "narrow"), ("P1", "broad"), ("P2", "narrow"),
         ("P2", "broad"), ("CH", "narrow"), ("CH", "broad")]

# BROAD-S sampler, PREDECLARED.md section 4.1. Four equal index blocks of 160.
BROADS_BLOCKS = ((0, 160, 0), (160, 320, 25), (320, 480, 50), (480, 640, 100))

# Spectral bands for S2, PREDECLARED.md section 2.
BANDS = ((0, 1), (1, 3), (3, 5), (5, 9), (9, 17), (17, 33), (33, 65), (65, 129))


def second_operator(pair: str):
    return f1.second_operator(PAIR_KEY[pair])


def diffusion():
    return tp.build_steppers(tp.DT)[0]


def grid():
    return jnp.linspace(0.0, tp.DOMAIN_EXTENT, tp.NUM_POINTS, endpoint=False)


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------
def train_states_for(pair: str, condition: str):
    """(640, 101, 1, 256) training states of a cell -- the states the learner sees."""
    d = diffusion()
    if condition == "broadS":
        base = tp.initial_conditions(tp.NUM_TRAIN_UNITS, rb.NARROW, tp.MASTER_SEED + 1)
        traj = tp.rollout(second_operator(pair), base, tp.STEPS_PER_TRAJECTORY)
        ic = jnp.concatenate([traj[lo:hi, m] for lo, hi, m in BROADS_BLOCKS], axis=0)
        del traj
    else:
        ic = tp.initial_conditions(tp.NUM_TRAIN_UNITS, OFFSETS[(pair, condition)],
                                   tp.MASTER_SEED + 1)
    return tp.rollout(d, ic, tp.STEPS_PER_TRAJECTORY)


def eval_cohorts(pair: str):
    """On-support cohort (MASTER_SEED+7) and the pair's switch cohort (MASTER_SEED+8)."""
    d = diffusion()
    on_ic = tp.initial_conditions(rb.EVAL_UNITS, rb.NARROW, tp.MASTER_SEED + 7)
    on_states = tp.rollout(d, on_ic, tp.STEPS_PER_TRAJECTORY)[:, 0]
    on_target = tp.rollout(d, on_states, rb.K)[:, -1]
    sw_ic = tp.initial_conditions(rb.EVAL_UNITS, rb.NARROW, tp.MASTER_SEED + 8)
    switch_states = tp.rollout(second_operator(pair), sw_ic,
                               tp.STEPS_PER_TRAJECTORY)[:, -1]
    switch_target = tp.rollout(d, switch_states, rb.K)[:, -1]
    return dict(on_states=on_states, on_target=on_target,
                switch_states=switch_states, switch_target=switch_target)


# ===========================================================================
# eval_banked -- R for models already fitted by family 1. No fit, no training corpus.
# ===========================================================================
def eval_banked(pair: str, condition: str) -> None:
    started = time.perf_counter()
    ck_pair = PAIR_KEY[pair]
    # The narrow model is the same model for every pair (family 1 reuse_narrow.py);
    # re-verify by checksum before relying on it.
    import hashlib

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    reuse_check = None
    if condition == "narrow":
        a = F1 / "ckpt" / "P1_diffusion_fisher_kpp" / "seed0_narrow.npz"
        b = F1 / "ckpt" / "P2_diffusion_allen_cahn" / "seed0_narrow.npz"
        c = F1 / "ckpt" / "P3_negative_control" / "seed0_narrow.npz"
        reuse_check = {"P1_seed0": sha(a), "P2_seed0": sha(b), "CH_seed0": sha(c),
                       "identical": sha(a) == sha(b) == sha(c)}
        if not reuse_check["identical"]:
            raise SystemExit("narrow checkpoints differ across pairs; refusing to reuse")
        # Fisher-KPP narrow models are only banked under P1 for seeds 0,1; the other
        # eight are the byte-identical files under P2/CH. Use P2's tree for all ten.
        ck_pair = "P2_diffusion_allen_cahn"

    c = eval_cohorts(pair)
    g = grid()
    rows = {}
    for seed in SEEDS:
        path = F1 / "ckpt" / ck_pair / f"seed{seed}_{condition}.npz"
        params = f1.load_params(path, seed)
        e_on = float(np.median(rb.evaluate(params, c["on_states"], c["on_target"], g)))
        e_off = float(np.median(rb.evaluate(params, c["switch_states"],
                                            c["switch_target"], g)))
        rows[str(seed)] = {"e_on": e_on, "e_off": e_off, "R": e_off / e_on,
                           "checkpoint": str(path)}
        print(f"{pair} {condition} seed {seed}  e_on {e_on:.6e}  e_off {e_off:.6e}  "
              f"R {e_off / e_on:.3f}", flush=True)

    Rs = [rows[str(s)]["R"] for s in SEEDS]
    persistence_on = float(np.median(rb.relative_l2(
        np.asarray(c["on_states"], np.float64), np.asarray(c["on_target"], np.float64))))
    persistence_off = float(np.median(rb.relative_l2(
        np.asarray(c["switch_states"], np.float64),
        np.asarray(c["switch_target"], np.float64))))
    out = {
        "schema": "family1c-banked-eval-v1",
        "status": ("R read from models family 1 already fitted; NO model was fitted "
                   "here and nothing was written into family1_second_pairs/"),
        "pair": pair, "condition": condition,
        "checkpoint_tree": str(F1 / "ckpt" / ck_pair),
        "narrow_reuse_checksum": reuse_check,
        "offsets": list(OFFSETS[(pair, condition)]),
        "seeds": list(SEEDS), "per_seed": rows,
        "median_R": float(np.median(Rs)),
        "R_range": [float(min(Rs)), float(max(Rs))],
        "median_e_on": float(np.median([rows[str(s)]["e_on"] for s in SEEDS])),
        "median_e_off": float(np.median([rows[str(s)]["e_off"] for s in SEEDS])),
        "persistence_on": persistence_on, "persistence_off": persistence_off,
        "dynamic_range": persistence_off / float(np.median(
            [rows[str(s)]["e_on"] for s in SEEDS])),
        "elapsed_seconds": time.perf_counter() - started,
    }
    d = RESULTS / "banked"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{pair}_{condition}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"median R {out['median_R']:.3f}  range {out['R_range']}", flush=True)


# ===========================================================================
# support -- statistic S (PCA-99 Mahalanobis) and S2 (spectral band profile).
# Truth states only. No learner.
# ===========================================================================
def band_fractions(x: np.ndarray) -> np.ndarray:
    """(n, 256) real states -> (n, 8) band-energy fractions."""
    p = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    out = np.stack([p[:, lo:hi].sum(axis=1) for lo, hi in BANDS], axis=1)
    return out / out.sum(axis=1, keepdims=True)


def analyse_cloud(train_states, switch_states) -> dict:
    """PREDECLARED.md sections 1 and 2, applied to one (training cloud, switch cohort)."""
    x = np.asarray(train_states, np.float64).reshape(-1, tp.NUM_POINTS)
    s = np.asarray(switch_states, np.float64).reshape(-1, tp.NUM_POINTS)
    n = x.shape[0]
    mu = x.mean(axis=0)
    xc = x - mu
    cov = (xc.T @ xc) / (n - 1)
    lam, vec = np.linalg.eigh(cov)
    order = np.argsort(lam)[::-1]
    lam, vec = lam[order], vec[:, order]
    total = lam.sum()
    cum = np.cumsum(lam) / total
    k = int(np.searchsorted(cum, 0.99) + 1)

    u = vec[:, :k]
    lk = lam[:k]
    pt = xc @ u
    md_train = np.sqrt((pt ** 2 / lk).sum(axis=1))
    ps = (s - mu) @ u
    md_switch = np.sqrt((ps ** 2 / lk).sum(axis=1))
    r99 = float(np.percentile(md_train, 99))
    inside = md_switch <= r99

    # S2: nearest-training-state distance in band-fraction space.
    ft = band_fractions(x)
    fs = band_fractions(s)
    best = np.full(fs.shape[0], np.inf)
    for start in range(0, ft.shape[0], 2048):
        blk = ft[start:start + 2048]
        d2 = ((fs[:, None, :] - blk[None, :, :]) ** 2).sum(axis=2)
        best = np.minimum(best, d2.min(axis=1))
    s2 = np.sqrt(best)

    return {
        "train_states": int(n), "switch_states": int(s.shape[0]),
        "pca_k_99": k,
        "variance_explained_at_k": float(cum[k - 1]),
        "eigenvalue_top": float(lam[0]),
        "train_MD_p99_radius": r99,
        "train_MD_median": float(np.median(md_train)),
        "switch_MD_mean": float(md_switch.mean()),
        "switch_MD_median": float(np.median(md_switch)),
        "switch_MD_min": float(md_switch.min()),
        "switch_MD_max": float(md_switch.max()),
        "switch_MD_over_r99_median": float(np.median(md_switch) / r99),
        "coverage_inside_pca99": float(inside.mean()),
        "units_inside": int(inside.sum()),
        "inside": bool(inside.mean() >= 0.5),
        "S2_band_nn_mean": float(s2.mean()),
        "S2_band_nn_median": float(np.median(s2)),
        "S2_band_nn_max": float(s2.max()),
    }


def support() -> None:
    started = time.perf_counter()
    out = {"schema": "family1c-support-v1",
           "status": "truth states only; no learner enters this file",
           "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "statistic_S": ("PCA on the condition's 640x101 training states, k = smallest "
                           "number of components with cumulative variance >= 0.99; "
                           "Mahalanobis distance in that basis; radius = 99th percentile "
                           "over the training states"),
           "statistic_S2": ("L2 between a switch state's 8-band energy-fraction profile "
                            "and the nearest training state's profile"),
           "bands": [list(b) for b in BANDS],
           "cells": {}}

    switch = {p: eval_cohorts(p)["switch_states"] for p in ("P1", "P2", "CH")}

    # Cells sharing a training cloud are grouped so each cloud is built once.
    groups = [("narrow", (0.4865, 0.5171), ["P1", "P2", "CH"]),
              ("broad", None, ["P1"]), ("broad", None, ["P2"]), ("broad", None, ["CH"]),
              ("broadS", None, ["CH"]), ("broadS", None, ["P1"])]
    for condition, _, pairs in groups:
        ts = train_states_for(pairs[0], condition)
        for p in pairs:
            row = analyse_cloud(ts, switch[p])
            row["training_offsets"] = (list(OFFSETS[(p, condition)])
                                       if condition != "broadS" else "BROAD-S sampler")
            key = f"{p}_{condition}"
            out["cells"][key] = row
            print(f"{key:12} k={row['pca_k_99']:3d}  r99 {row['train_MD_p99_radius']:8.3f}"
                  f"  switch MD mean {row['switch_MD_mean']:10.3f}"
                  f"  coverage {row['coverage_inside_pca99']:.4f}"
                  f"  S2 {row['S2_band_nn_mean']:.5f}", flush=True)
        del ts

    out["elapsed_seconds"] = time.perf_counter() - started
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "SUPPORT.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote SUPPORT.json ({out['elapsed_seconds']:.1f} s)", flush=True)


# ===========================================================================
# BROAD-S: train (e_on only) -> gates -> eval (e_off)
# ===========================================================================
def broads_train(pair: str, seed: int) -> None:
    started = time.perf_counter()
    ts = train_states_for(pair, "broadS")
    pairs_in, pairs_out = rb.make_pairs(ts)
    del ts
    g = grid()
    t0 = time.perf_counter()
    params = rb.train(pairs_in, pairs_out, g, seed)      # shipped fit, unchanged
    train_seconds = time.perf_counter() - t0
    del pairs_in, pairs_out

    d = diffusion()
    on_ic = tp.initial_conditions(rb.EVAL_UNITS, rb.NARROW, tp.MASTER_SEED + 7)
    on_states = tp.rollout(d, on_ic, tp.STEPS_PER_TRAJECTORY)[:, 0]
    on_target = tp.rollout(d, on_states, rb.K)[:, -1]
    e_on = float(np.median(rb.evaluate(params, on_states, on_target, g)))

    CKPT.mkdir(parents=True, exist_ok=True)
    f1.save_params(CKPT / f"{pair}_broadS_seed{seed}.npz", params)
    outdir = RESULTS / "broadS"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{pair}_seed{seed}_pregate.json").write_text(json.dumps({
        "schema": "family1c-broadS-pregate-v1",
        "pair": pair, "condition": "broadS", "seed": seed,
        "sampler": "PREDECLARED.md 4.1, blocks " + str(BROADS_BLOCKS),
        "e_on": e_on, "train_seconds": train_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "steps": rb.STEPS, "batch": rb.BATCH,
        "note": "e_off is deliberately NOT in this file; gates are written first",
        "platform": {"python": sys.version.split()[0], "machine": platform.machine(),
                     "jax": jax.__version__},
    }, indent=2) + "\n")
    print(f"{pair} broadS seed {seed}  e_on {e_on:.6e}  {train_seconds:.1f} s train",
          flush=True)


def broads_gates() -> None:
    outdir = RESULTS / "broadS"
    e_on = {}
    for pair in ("CH", "P1"):
        for s in SEEDS:
            e_on[(pair, s)] = json.loads(
                (outdir / f"{pair}_seed{s}_pregate.json").read_text())["e_on"]

    c = eval_cohorts("CH")
    persistence_on = float(np.median(rb.relative_l2(
        np.asarray(c["on_states"], np.float64), np.asarray(c["on_target"], np.float64))))
    persistence_off = float(np.median(rb.relative_l2(
        np.asarray(c["switch_states"], np.float64),
        np.asarray(c["switch_target"], np.float64))))
    # G2 is the family-1 gate: persistence_off over the NARROW median e_on, from the
    # banked negative-control gate file. Recomputed here and reported alongside.
    f1_gates = json.loads((F1 / "results" / "P3_negative_control" / "GATES.json")
                          .read_text())
    narrow_median_e_on = f1_gates["G2_metric_resolves"]["median_e_on"]

    worst = max(e_on.values())
    payload = {
        "schema": "family1c-broadS-gates-v1",
        "written_before_any_off_support_number_was_read": True,
        "per_seed_e_on": {f"{p}_{s}": v for (p, s), v in e_on.items()},
        "REPAIR_2_G2_dynamic_range": {
            "persistence_off_CH": persistence_off, "persistence_on": persistence_on,
            "narrow_median_e_on": narrow_median_e_on,
            "dynamic_range": persistence_off / narrow_median_e_on,
            "required": rb.G2_MIN_DYNAMIC_RANGE,
            "banked_family1_value": f1_gates["G2_metric_resolves"]["dynamic_range"],
            "pass": bool(persistence_off / narrow_median_e_on
                         >= rb.G2_MIN_DYNAMIC_RANGE)},
        "REPAIR_3_P6_convergence": {
            "worst_e_on": float(worst), "limit": rb.P6_MAX_E_ON,
            "pass": bool(worst <= rb.P6_MAX_E_ON)},
    }
    (outdir / "GATES.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("BROAD-S gates written before any off-support number was read:")
    print(f"  REPAIR-2 G2 dynamic range "
          f"{payload['REPAIR_2_G2_dynamic_range']['dynamic_range']:.1f} "
          f"{'PASS' if payload['REPAIR_2_G2_dynamic_range']['pass'] else 'FAIL'}")
    print(f"  REPAIR-3 P6 worst e_on {worst:.6e} "
          f"{'PASS' if payload['REPAIR_3_P6_convergence']['pass'] else 'FAIL'}")


def broads_eval(pair: str) -> None:
    outdir = RESULTS / "broadS"
    if not (outdir / "GATES.json").exists():
        raise SystemExit("GATES.json absent: refusing to read the off-support error")
    c = eval_cohorts(pair)
    g = grid()
    for seed in SEEDS:
        pre = json.loads((outdir / f"{pair}_seed{seed}_pregate.json").read_text())
        params = f1.load_params(CKPT / f"{pair}_broadS_seed{seed}.npz", seed)
        e_on_reload = float(np.median(rb.evaluate(params, c["on_states"],
                                                  c["on_target"], g)))
        e_off = float(np.median(rb.evaluate(params, c["switch_states"],
                                            c["switch_target"], g)))
        payload = dict(pre)
        payload.update({"schema": "family1c-broadS-seed-v1",
                        "e_on_reloaded": e_on_reload,
                        "reload_exact": bool(e_on_reload == pre["e_on"]),
                        "e_off": e_off, "R": e_off / pre["e_on"]})
        payload.pop("note", None)
        (outdir / f"{pair}_seed{seed}.json").write_text(
            json.dumps(payload, indent=2) + "\n")
        print(f"{pair} broadS seed {seed}  e_on {pre['e_on']:.6e}  e_off {e_off:.6e}  "
              f"R {payload['R']:.3f}", flush=True)


# ===========================================================================
def main(argv) -> None:
    if not argv:
        raise SystemExit(__doc__)
    cmd = argv[0]
    if cmd == "eval_banked":
        eval_banked(argv[1], argv[2])
    elif cmd == "support":
        support()
    elif cmd == "broads_train":
        broads_train(argv[1], int(argv[2]))
    elif cmd == "broads_gates":
        broads_gates()
    elif cmd == "broads_eval":
        broads_eval(argv[1])
    else:
        raise SystemExit(f"unknown subcommand {cmd!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
