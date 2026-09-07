"""Does the state-support failure reproduce on a standard operator surrogate?

Governed by the plan fixed before anything was computed. The plan's ordering rule is enforced here in code rather than by discipline:
the gates that do not arithmetically require the off-support error are computed and
WRITTEN TO DISK before `e_off` is read at all. A failed gate has to be discoverable
while it is still a design fact, not after it has become a convenient explanation.

The model, optimiser and data helpers come from timing_probe.py unchanged, so the thing
that was timed and the thing that runs are the same code.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results" / "tables" / "baseline"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "band_check"))
import _runtime_path  # noqa: F401,E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

import timing_probe as tp  # noqa: E402

K = 10                      # the estimand: u_t -> u_{t+10}
STEPS = 8000
BATCH = 64
SEEDS = (0, 1, 2, 3, 4)
LR_HIGH, LR_LOW = 1e-3, 1e-5
EVAL_UNITS = 256
NARROW = (0.4865, 0.5171)
BROAD = (0.4865, 0.7258)
MASTER_SEED = tp.MASTER_SEED

# Gate thresholds, from sections 7.2 to 7.5 of the plan. Copied here so that changing
# one without changing the document breaks the hash check above.
G2_MIN_DYNAMIC_RANGE = 100.0
G2_MAX_E_ON_FRACTION = 0.1          # e_on <= 0.1 * persistence on-support
G3_MAX_LOG10_IQR = 0.5
G4_MIN_E_ON = 1.19e-5               # 100x float32 epsilon
P2_MIN_R = 5.0
P3_MAX_BROAD_FRACTION = 0.25
P4_MAX_CORRECTED_FRACTION = 0.25
P6_MAX_E_ON = 1.573e-3


def relative_l2(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    num = np.sqrt(((pred - target) ** 2).sum(axis=(1, 2)))
    den = np.sqrt((target ** 2).sum(axis=(1, 2)))
    return num / den


def make_pairs(states: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """(units, frames, 1, N) -> input/target pairs k frames apart."""
    return (states[:, :-K].reshape(-1, 1, tp.NUM_POINTS),
            states[:, K:].reshape(-1, 1, tp.NUM_POINTS))


def train(pairs_in, pairs_out, grid, seed: int):
    params = tp.init_fno(jr.PRNGKey(seed))
    opt = tp.adam_init(params)
    key = jr.PRNGKey(seed + 9999)
    n = pairs_in.shape[0]
    for step in range(STEPS):
        key, batch_key = jr.split(key)
        index = jr.randint(batch_key, (BATCH,), 0, n)
        lr = LR_LOW + 0.5 * (LR_HIGH - LR_LOW) * (1 + np.cos(np.pi * step / STEPS))
        params, opt, _ = tp.train_step(params, opt, pairs_in[index], pairs_out[index],
                                       grid, lr)
    return params


def evaluate(params, states_in, targets, grid) -> np.ndarray:
    out = []
    for start in range(0, states_in.shape[0], 64):
        chunk = states_in[start:start + 64]
        out.append(np.asarray(tp.fno_apply(params, chunk, grid), dtype=np.float64))
    pred = np.concatenate(out, axis=0)
    return relative_l2(pred, np.asarray(targets, dtype=np.float64))


def main() -> None:
    started = time.perf_counter()

    diffusion, reaction = tp.build_steppers(tp.DT)
    grid = jnp.linspace(0.0, tp.DOMAIN_EXTENT, tp.NUM_POINTS, endpoint=False)

    # ---- corpora -----------------------------------------------------------
    train_states = {}
    for name, offsets in (("narrow", NARROW), ("broad", BROAD)):
        ic = tp.initial_conditions(tp.NUM_TRAIN_UNITS, offsets, MASTER_SEED + 1)
        train_states[name] = tp.rollout(diffusion, ic, tp.STEPS_PER_TRAJECTORY)

    on_ic = tp.initial_conditions(EVAL_UNITS, NARROW, MASTER_SEED + 7)
    on_states = tp.rollout(diffusion, on_ic, tp.STEPS_PER_TRAJECTORY)[:, 0]
    on_target = tp.rollout(diffusion, on_states, K)[:, -1]

    switch_ic = tp.initial_conditions(EVAL_UNITS, NARROW, MASTER_SEED + 8)
    switch_states = tp.rollout(reaction, switch_ic, tp.STEPS_PER_TRAJECTORY)[:, -1]
    switch_target = tp.rollout(diffusion, switch_states, K)[:, -1]

    def means(x):
        return np.asarray(x.mean(axis=-1)).ravel()

    train_means = means(train_states["narrow"])
    switch_means = means(switch_states)
    support = (float(train_means.min()), float(train_means.max()))
    switch_support = (float(switch_means.min()), float(switch_means.max()))
    overlap = max(0.0, min(support[1], switch_support[1])
                  - max(support[0], switch_support[0]))

    persistence_on = float(np.median(relative_l2(
        np.asarray(on_states, np.float64), np.asarray(on_target, np.float64))))
    persistence_off = float(np.median(relative_l2(
        np.asarray(switch_states, np.float64), np.asarray(switch_target, np.float64))))

    # ---- train, and read ONLY the on-support error ------------------------
    fitted, e_on = {}, {}
    for condition in ("narrow", "broad"):
        pairs_in, pairs_out = make_pairs(train_states[condition])
        for seed in SEEDS:
            key = (condition, seed)
            elapsed = time.perf_counter()
            fitted[key] = train(pairs_in, pairs_out, grid, seed)
            e_on[key] = float(np.median(evaluate(fitted[key], on_states, on_target,
                                                 grid)))
            print(f"  {condition:7} seed {seed}  e_on {e_on[key]:.6e}  "
                  f"{time.perf_counter() - elapsed:6.1f} s")

    narrow_e_on = [e_on[("narrow", s)] for s in SEEDS]
    median_e_on = float(np.median(narrow_e_on))
    dynamic_range = persistence_off / median_e_on
    log_spread = np.log10(narrow_e_on)

    gates = {
        "schema": "pde-baseline-gates-v1",
        "written_before_any_off_support_number_was_read": True,
        "G1_shift_exists": {
            "training_support": list(support), "switch_support": list(switch_support),
            "overlap": overlap, "gap": switch_support[0] - support[1],
            "pass": bool(overlap == 0.0),
        },
        "G2_metric_resolves": {
            "persistence_off": persistence_off, "persistence_on": persistence_on,
            "median_e_on": median_e_on, "dynamic_range": dynamic_range,
            "required_dynamic_range": G2_MIN_DYNAMIC_RANGE,
            "e_on_limit": G2_MAX_E_ON_FRACTION * persistence_on,
            "pass": bool(dynamic_range >= G2_MIN_DYNAMIC_RANGE
                         and median_e_on <= G2_MAX_E_ON_FRACTION * persistence_on),
        },
        "G3_powered_partial": {
            "note": ("the log10(R) interquartile range needs e_off and is completed "
                     "after it; this is the across-seed spread of log10(e_on), which "
                     "is the part of the power question that can be asked now"),
            "log10_e_on_iqr": float(np.subtract(*np.percentile(log_spread, [75, 25]))),
            "per_seed_e_on": narrow_e_on,
        },
        "G4_censored_below": {
            "minimum_e_on": float(min(narrow_e_on)), "floor": G4_MIN_E_ON,
            "pass": bool(min(narrow_e_on) >= G4_MIN_E_ON),
        },
        "P6_convergence": {
            "limit": P6_MAX_E_ON,
            "pass": bool(max(e_on.values()) <= P6_MAX_E_ON),
            "worst_e_on": float(max(e_on.values())),
        },
    }
    (RESULTS / "BASELINE_GATES.json").write_text(json.dumps(gates, indent=2) + "\n")
    print("\nGATES, written before any off-support number was read")
    for name in ("G1_shift_exists", "G2_metric_resolves", "G4_censored_below",
                 "P6_convergence"):
        print(f"  {name:22} {'PASS' if gates[name]['pass'] else 'FAIL'}")
    print(f"  G3 log10(e_on) IQR across seeds "
          f"{gates['G3_powered_partial']['log10_e_on_iqr']:.4f}")

    blocking = [n for n in ("G1_shift_exists", "G2_metric_resolves",
                            "G4_censored_below", "P6_convergence")
                if not gates[n]["pass"]]
    if blocking:
        print(f"\nGATE FAILURE: {', '.join(blocking)}")
        print("Stopping before the off-support error is read. A null under a failed "
              "gate is FLAWED, and the plan forbids interpreting it.")
        raise SystemExit(2)

    # ---- only now is the off-support error read ---------------------------
    print("\nall pre-gates pass; reading the off-support error")
    e_off, corrected = {}, {}
    for key, params in fitted.items():
        errors = evaluate(params, switch_states, switch_target, grid)
        e_off[key] = float(np.median(errors))
        if key[0] == "narrow":
            out = []
            for start in range(0, EVAL_UNITS, 64):
                chunk = switch_states[start:start + 64]
                p = np.asarray(tp.fno_apply(params, chunk, grid), np.float64)
                s = np.asarray(chunk, np.float64)
                out.append(p - p.mean(axis=-1, keepdims=True)
                           + s.mean(axis=-1, keepdims=True))
            corrected[key] = float(np.median(relative_l2(
                np.concatenate(out, 0), np.asarray(switch_target, np.float64))))

    ratio = {k: e_off[k] / e_on[k] for k in e_off}
    narrow_R = [ratio[("narrow", s)] for s in SEEDS]
    broad_R = [ratio[("broad", s)] for s in SEEDS]
    log_R = np.log10(narrow_R)
    iqr_log_R = float(np.subtract(*np.percentile(log_R, [75, 25])))

    payload = {
        "schema": "pde-baseline-result-v1",
        "generated": "2026-09-05",
        "estimand": f"k={K} diffusion solution operator, tau={K * tp.DT:.2f}",
        "gates": gates,
        "G3_completed": {"log10_R_iqr": iqr_log_R, "limit": G3_MAX_LOG10_IQR,
                         "pass": bool(iqr_log_R <= G3_MAX_LOG10_IQR)},
        "G4_censored_above": {
            "max_e_off": float(max(e_off.values())), "persistence_off": persistence_off,
            "pass": bool(max(e_off.values()) < persistence_off)},
        "per_seed": {f"{c}_{s}": {"e_on": e_on[(c, s)], "e_off": e_off[(c, s)],
                                  "R": ratio[(c, s)]}
                     for c in ("narrow", "broad") for s in SEEDS},
        "narrow": {"median_e_on": median_e_on,
                   "median_e_off": float(np.median([e_off[("narrow", s)]
                                                    for s in SEEDS])),
                   "median_R": float(np.median(narrow_R)),
                   "R_range": [float(min(narrow_R)), float(max(narrow_R))]},
        "broad": {"median_e_on": float(np.median([e_on[("broad", s)] for s in SEEDS])),
                  "median_e_off": float(np.median([e_off[("broad", s)]
                                                   for s in SEEDS])),
                  "median_R": float(np.median(broad_R)),
                  "R_range": [float(min(broad_R)), float(max(broad_R))]},
        "P2_degradation_reproduces": {
            "median_R": float(np.median(narrow_R)), "threshold": P2_MIN_R,
            "held": bool(np.median(narrow_R) >= P2_MIN_R)},
        "P3_broadening_repairs": {
            "narrow_R": float(np.median(narrow_R)), "broad_R": float(np.median(broad_R)),
            "limit": P3_MAX_BROAD_FRACTION,
            "held": bool(np.median(broad_R)
                         <= P3_MAX_BROAD_FRACTION * np.median(narrow_R))},
        "P4_mechanism_is_the_constant_mode": {
            "median_e_off": float(np.median([e_off[("narrow", s)] for s in SEEDS])),
            "median_e_off_corrected": float(np.median(list(corrected.values()))),
            "limit": P4_MAX_CORRECTED_FRACTION,
            "held": bool(np.median(list(corrected.values()))
                         <= P4_MAX_CORRECTED_FRACTION
                         * np.median([e_off[("narrow", s)] for s in SEEDS]))},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (RESULTS / "BASELINE_RESULT.json").write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\nnarrow  e_on {median_e_on:.6e}  e_off "
          f"{payload['narrow']['median_e_off']:.6e}  R {payload['narrow']['median_R']:.3f}")
    print(f"broad   e_on {payload['broad']['median_e_on']:.6e}  e_off "
          f"{payload['broad']['median_e_off']:.6e}  R {payload['broad']['median_R']:.3f}")
    for name in ("P2_degradation_reproduces", "P3_broadening_repairs",
                 "P4_mechanism_is_the_constant_mode"):
        print(f"  {name:34} {'HELD' if payload[name]['held'] else 'FAILED'}")
    print(f"  G3 log10(R) IQR {iqr_log_R:.4f} "
          f"{'PASS' if payload['G3_completed']['pass'] else 'FAIL'}")
    print(f"  G4 censored above "
          f"{'PASS' if payload['G4_censored_above']['pass'] else 'FAIL'}")
    print(f"\nelapsed {payload['elapsed_seconds'] / 60:.1f} min, CPU, $0")


if __name__ == "__main__":
    main()
