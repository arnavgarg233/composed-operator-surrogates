"""Family 1e -- state-space broadening (BROAD-S) on the two new pairs, 10 seeds.

Everything here is fixed by PREDECLARED.md (sha256 in results/PREDECLARED.sha256, taken
at 2026-09-11T21:11:33Z, before any number in this family was computed).

PROVENANCE. Nothing under recovery/, dependency/, family1_second_pairs/,
family1b_pair_screen/, family1_rerun/, family1c_state_support/, family1d_predictor/,
family2_* or family4_* is written to. The shipped harness and the rerun's operator
definitions are IMPORTED so the code that fits a model here is the code that fitted the
models there:

    dependency/scripts/baseline/run_baseline.py   -> rb  (train, evaluate, relative_l2,
        make_pairs, K, STEPS, BATCH, EVAL_UNITS, NARROW, gate thresholds)
    dependency/scripts/baseline/timing_probe.py   -> tp  (physical constants, corpus,
        FNO, hand-written Adam)
    family1_second_pairs/run_family1.py            -> f1  (save_params/load_params)
    family1_rerun/run_family1_rerun.py             -> rr  (second_operator for the two
        new pairs; imported read-only -- no function of it that writes is ever called,
        and its module-level code writes nothing)

New code here: the BROAD-S corpus for the two new pairs (family 1c's sampler, verbatim),
the coverage statistics (family 1c's S and family 1d's C3, verbatim), and the gates.

Run with:
    source environment setup script
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu \
    XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
    JAX_COMPILATION_CACHE_DIR=work/family1e/jax_cache \
    .venv/bin/python run_family1e.py <cmd> ...

Subcommands:
    support                 -> results/SUPPORT.json    (truth states only, no learner)
    train <pair> <seed>     -> results/broadS/<pair>_seed<k>_pregate.json + ckpt
    gates                   -> results/broadS/GATES.json   (before any e_off is read)
    eval  <pair>            -> results/broadS/<pair>_seed<k>.json
"""

from __future__ import annotations

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
RERUN = RELEASE_ROOT / "results" / "family1_rerun"
WORK = RELEASE_ROOT / "work" / "family1e"

sys.path.insert(0, str(PKG / "scripts" / "baseline"))
sys.path.insert(0, str(PKG / "scripts" / "band_check"))
sys.path.insert(0, str(F1))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import timing_probe as tp  # noqa: E402
import run_baseline as rb  # noqa: E402
import run_family1 as f1  # noqa: E402  (read-only use: save_params / load_params)

# The rerun's second-operator definitions, imported rather than retyped. Loading this
# module only binds names (its module level sets attributes on its own private copy of
# the harness and touches no file); nothing that writes into family1_rerun/ is called.
_spec = importlib.util.spec_from_file_location("rf1rerun", RERUN / "run_family1_rerun.py")
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)

RESULTS = RELEASE_ROOT / "results" / "family1e_state_broadening_new_pairs"
BROADS = RESULTS / "broadS"
CKPT = WORK / "ckpt"

PAIRS = ("S1_fkpp_r0.5", "S1_grayscott_1sp")
SEEDS = tuple(range(10))

# BROAD-S sampler, PREDECLARED.md section 2 == family 1c PREDECLARED.md section 4.1.
BROADS_BLOCKS = ((0, 160, 0), (160, 320, 25), (320, 480, 50), (480, 640, 100))


def second_operator(pair: str):
    return rr.second_operator(pair)


def diffusion():
    return tp.build_steppers(tp.DT)[0]


def grid():
    return jnp.linspace(0.0, tp.DOMAIN_EXTENT, tp.NUM_POINTS, endpoint=False)


def spatial_means(states) -> np.ndarray:
    return np.asarray(states.mean(axis=-1)).reshape(-1)


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------
def train_states_for(pair: str, condition: str):
    """(640, 101, 1, 256) training states -- the states the learner sees."""
    d = diffusion()
    if condition == "broadS":
        base = tp.initial_conditions(tp.NUM_TRAIN_UNITS, rb.NARROW, tp.MASTER_SEED + 1)
        traj = tp.rollout(second_operator(pair), base, tp.STEPS_PER_TRAJECTORY)
        ic = jnp.concatenate([traj[lo:hi, m] for lo, hi, m in BROADS_BLOCKS], axis=0)
        del traj
    elif condition == "narrow":
        ic = tp.initial_conditions(tp.NUM_TRAIN_UNITS, rb.NARROW, tp.MASTER_SEED + 1)
    else:
        raise SystemExit(f"unknown condition {condition!r}")
    return tp.rollout(d, ic, tp.STEPS_PER_TRAJECTORY)


def eval_cohorts(pair: str):
    """On-support cohort (MASTER_SEED+7) and the pair's switch cohort (MASTER_SEED+8),
    byte-identical to family 1 / the rerun."""
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


def med_rel(a, b) -> float:
    return float(np.median(rb.relative_l2(np.asarray(a, np.float64),
                                          np.asarray(b, np.float64))))


# ===========================================================================
# support -- coverage statistics, truth states only. Reported, never gated
# (PREDECLARED.md section 4).
# ===========================================================================
def _cloud_stats(train_states, switch_states) -> dict:
    x = np.asarray(train_states, np.float64).reshape(-1, tp.NUM_POINTS)
    s = np.asarray(switch_states, np.float64).reshape(-1, tp.NUM_POINTS)
    n = x.shape[0]
    mu = x.mean(axis=0)
    xc = x - mu
    cov = (xc.T @ xc) / (n - 1)
    lam, vec = np.linalg.eigh(cov)
    order = np.argsort(lam)[::-1]
    lam, vec = lam[order], vec[:, order]
    cum = np.cumsum(lam) / lam.sum()
    k = int(np.searchsorted(cum, 0.99) + 1)
    u = vec[:, :k]
    sc = s - mu

    # --- statistic S: PCA-99 Mahalanobis (family 1c PREDECLARED section 1) ----
    def md(z):
        return np.sqrt(((z @ u) ** 2 / lam[:k]).sum(axis=1))
    md_tr = md(xc)
    r99 = float(np.percentile(md_tr, 99))
    md_sw = md(sc)

    # --- statistic S3: out-of-basis residual (family 1d PREDECLARED section 1) ---
    res_tr = np.linalg.norm(xc - (xc @ u) @ u.T, axis=1)
    q99 = float(np.percentile(res_tr, 99))
    res_sw = np.linalg.norm(sc - (sc @ u) @ u.T, axis=1)
    tot = np.linalg.norm(sc, axis=1)

    return {
        "n_train_states": int(n), "n_switch_states": int(s.shape[0]),
        "pca_k_99": k, "variance_explained_at_k": float(cum[k - 1]),
        "train_MD_p99_radius": r99,
        "Dbar_switch_MD_mean": float(md_sw.mean()),
        "switch_MD_median": float(np.median(md_sw)),
        "coverage_S_inside_pca99": float((md_sw <= r99).mean()),
        "train_residual_p99": q99,
        "D3_switch_residual_mean": float(res_sw.mean()),
        "C3_coverage_inside_train_residual_p99": float((res_sw <= q99).mean()),
        "D3f_switch_fraction_outside_basis_median": float(np.median(res_sw / tot)),
        "switch_residual_median": float(np.median(res_sw)),
    }


def support() -> None:
    started = time.perf_counter()
    out = {"schema": "family1e-support-v1",
           "status": "truth states only; no learner enters this file; nothing here is "
                     "gated (PREDECLARED.md section 4)",
           "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "statistic_S": "family 1c: PCA-99 basis, Mahalanobis, 99th-pct training radius",
           "statistic_S3": "family 1d: out-of-basis residual Q, 99th-pct training q99",
           "cells": {}, "geometry": {}}

    switch = {p: eval_cohorts(p)["switch_states"] for p in PAIRS}

    # narrow cloud is the same for both pairs (offsets do not depend on the operator)
    narrow_states = train_states_for(PAIRS[0], "narrow")
    nm = spatial_means(narrow_states)
    narrow_support = (float(nm.min()), float(nm.max()))
    out["geometry"]["narrow_training_support"] = list(narrow_support)
    for p in PAIRS:
        out["cells"][f"{p}_narrow"] = _cloud_stats(narrow_states, switch[p])
        sm = spatial_means(switch[p])
        sw_support = (float(sm.min()), float(sm.max()))
        overlap = max(0.0, min(narrow_support[1], sw_support[1])
                      - max(narrow_support[0], sw_support[0]))
        out["geometry"][p] = {
            "switch_support": list(sw_support), "narrow_overlap": overlap,
            "narrow_gap": sw_support[0] - narrow_support[1],
            "G1_zero_overlap": bool(overlap == 0.0)}
    del narrow_states

    for p in PAIRS:
        ts = train_states_for(p, "broadS")
        out["cells"][f"{p}_broadS"] = _cloud_stats(ts, switch[p])
        bm = spatial_means(ts)
        out["geometry"][p]["broadS_training_support"] = [float(bm.min()), float(bm.max())]
        del ts

    for name, r in out["cells"].items():
        print(f"{name:22s} k={r['pca_k_99']:3d}  covS {r['coverage_S_inside_pca99']:.4f}"
              f"  Dbar {r['Dbar_switch_MD_mean']:8.4f}"
              f"  C3 {r['C3_coverage_inside_train_residual_p99']:.4f}"
              f"  D3 {r['D3_switch_residual_mean']:9.5f}", flush=True)

    out["elapsed_seconds"] = time.perf_counter() - started
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "SUPPORT.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote SUPPORT.json ({out['elapsed_seconds']:.1f} s)", flush=True)


# ===========================================================================
# train -- BROAD-S fit. Writes e_on and the checkpoint; never e_off.
# ===========================================================================
def train(pair: str, seed: int) -> None:
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
    BROADS.mkdir(parents=True, exist_ok=True)
    (BROADS / f"{pair}_seed{seed}_pregate.json").write_text(json.dumps({
        "schema": "family1e-broadS-pregate-v1",
        "pair": pair, "condition": "broadS", "seed": seed,
        "sampler": "PREDECLARED.md section 2, blocks " + str(BROADS_BLOCKS),
        "e_on": e_on, "train_seconds": train_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "steps": rb.STEPS, "batch": rb.BATCH,
        "note": "e_off is deliberately NOT in this file; gates are written first",
        "platform": {"python": sys.version.split()[0], "machine": platform.machine(),
                     "jax": jax.__version__},
    }, indent=2) + "\n")
    print(f"{pair} broadS seed {seed}  e_on {e_on:.6e}  {train_seconds:.1f} s train",
          flush=True)


# ===========================================================================
# gates -- G1, G2, G4a, P6. Written before any e_off is read.
# ===========================================================================
def gates() -> None:
    started = time.perf_counter()
    sup = json.loads((RESULTS / "SUPPORT.json").read_text())
    payload = {"schema": "family1e-broadS-gates-v1",
               "written_before_any_off_support_number_was_read": True,
               "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "pairs": {}}

    e_on_all = {}
    for pair in PAIRS:
        for s in SEEDS:
            e_on_all[(pair, s)] = json.loads(
                (BROADS / f"{pair}_seed{s}_pregate.json").read_text())["e_on"]
    worst = max(e_on_all.values())

    for pair in PAIRS:
        rerun_gates = json.loads(
            (RERUN / "results" / pair / "GATES.json").read_text())
        c = eval_cohorts(pair)
        persistence_on = med_rel(c["on_states"], c["on_target"])
        persistence_off = med_rel(c["switch_states"], c["switch_target"])
        narrow_e_on = rerun_gates["G3_powered_partial"]["per_seed_e_on"]
        median_e_on = float(np.median(narrow_e_on))
        dyn = persistence_off / median_e_on
        broads_e_on = [e_on_all[(pair, s)] for s in SEEDS]
        geo = sup["geometry"][pair]

        payload["pairs"][pair] = {
            "G1_shift_exists": {
                "training_support": sup["geometry"]["narrow_training_support"],
                "switch_support": geo["switch_support"],
                "overlap": geo["narrow_overlap"], "gap": geo["narrow_gap"],
                "already_established_in":
                    str(RERUN / "results" / pair / "GATES.json"),
                "rerun_overlap": rerun_gates["G1_shift_exists"]["overlap"],
                "reproduces_rerun": bool(
                    geo["switch_support"]
                    == rerun_gates["G1_shift_exists"]["switch_support"]),
                "pass": bool(geo["narrow_overlap"] == 0.0)},
            "G2_metric_resolves": {
                "persistence_off": persistence_off, "persistence_on": persistence_on,
                "median_e_on_narrow": median_e_on, "dynamic_range": dyn,
                "required_dynamic_range": rb.G2_MIN_DYNAMIC_RANGE,
                "e_on_limit": rb.G2_MAX_E_ON_FRACTION * persistence_on,
                "rerun_dynamic_range":
                    rerun_gates["G2_metric_resolves"]["dynamic_range"],
                "pass": bool(dyn >= rb.G2_MIN_DYNAMIC_RANGE
                             and median_e_on
                             <= rb.G2_MAX_E_ON_FRACTION * persistence_on)},
            "G4a_censored_below": {
                "minimum_e_on_narrow": float(min(narrow_e_on)),
                "minimum_e_on_broadS": float(min(broads_e_on)),
                "floor": rb.G4_MIN_E_ON,
                "pass": bool(min(min(narrow_e_on), min(broads_e_on))
                             >= rb.G4_MIN_E_ON)},
            "P6_convergence": {
                "worst_e_on_this_pair": float(max(broads_e_on)),
                "worst_e_on_both_pairs": float(worst),
                "limit": rb.P6_MAX_E_ON,
                "pass": bool(worst <= rb.P6_MAX_E_ON)},
            "per_seed_e_on_broadS": {str(s): e_on_all[(pair, s)] for s in SEEDS},
        }
        blocking = [n for n in ("G1_shift_exists", "G2_metric_resolves",
                                "G4a_censored_below", "P6_convergence")
                    if not payload["pairs"][pair][n]["pass"]]
        payload["pairs"][pair]["blocking_failures"] = blocking

    payload["elapsed_seconds"] = time.perf_counter() - started
    BROADS.mkdir(parents=True, exist_ok=True)
    (BROADS / "GATES.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("BROAD-S gates written before any off-support number was read:")
    for pair in PAIRS:
        p = payload["pairs"][pair]
        print(f"  {pair}")
        for n in ("G1_shift_exists", "G2_metric_resolves", "G4a_censored_below",
                  "P6_convergence"):
            print(f"    {n:22s} {'PASS' if p[n]['pass'] else 'FAIL'}")
        if p["blocking_failures"]:
            print(f"    BLOCKING: {p['blocking_failures']} -- e_off will not be read")


# ===========================================================================
# eval -- only now is e_off read.
# ===========================================================================
def evaluate_pair(pair: str) -> None:
    if not (BROADS / "GATES.json").exists():
        raise SystemExit("GATES.json absent: refusing to read the off-support error")
    gate = json.loads((BROADS / "GATES.json").read_text())
    if gate["pairs"][pair]["blocking_failures"]:
        raise SystemExit(f"blocking gate failure {gate['pairs'][pair]['blocking_failures']}"
                         ": refusing to read the off-support error")
    started = time.perf_counter()
    c = eval_cohorts(pair)
    g = grid()
    for seed in SEEDS:
        pre = json.loads((BROADS / f"{pair}_seed{seed}_pregate.json").read_text())
        params = f1.load_params(CKPT / f"{pair}_broadS_seed{seed}.npz", seed)
        e_on_reload = float(np.median(rb.evaluate(params, c["on_states"],
                                                  c["on_target"], g)))
        e_off = float(np.median(rb.evaluate(params, c["switch_states"],
                                            c["switch_target"], g)))
        # constant-mode correction, rb.main()'s inline block, unchanged (P4, reported)
        out = []
        for start in range(0, rb.EVAL_UNITS, 64):
            chunk = c["switch_states"][start:start + 64]
            p = np.asarray(tp.fno_apply(params, chunk, g), np.float64)
            s = np.asarray(chunk, np.float64)
            out.append(p - p.mean(axis=-1, keepdims=True)
                       + s.mean(axis=-1, keepdims=True))
        corrected = float(np.median(rb.relative_l2(
            np.concatenate(out, 0), np.asarray(c["switch_target"], np.float64))))

        payload = dict(pre)
        payload.update({"schema": "family1e-broadS-seed-v1",
                        "e_on_reloaded": e_on_reload,
                        "reload_exact": bool(e_on_reload == pre["e_on"]),
                        "e_off": e_off, "R": e_off / pre["e_on"],
                        "e_off_constant_mode_corrected": corrected})
        payload.pop("note", None)
        (BROADS / f"{pair}_seed{seed}.json").write_text(
            json.dumps(payload, indent=2) + "\n")
        print(f"{pair} broadS seed {seed}  e_on {pre['e_on']:.6e}  e_off {e_off:.6e}  "
              f"R {payload['R']:.3f}", flush=True)
    print(f"{pair} eval done ({time.perf_counter() - started:.1f} s)", flush=True)


def main(argv) -> None:
    if not argv:
        raise SystemExit(__doc__)
    cmd = argv[0]
    if cmd == "support":
        support()
    elif cmd == "train":
        train(argv[1], int(argv[2]))
    elif cmd == "gates":
        gates()
    elif cmd == "eval":
        evaluate_pair(argv[1])
    else:
        raise SystemExit(f"unknown subcommand {cmd!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
