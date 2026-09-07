"""Does the available-signal ratio predict the dissociation on a second architecture?

Governed by the plan fixed before anything is computed.

Two Fourier neural operators are fitted, one per primitive, neither ever seeing a composed
trajectory. They are then composed autoregressively, ten applications per leg, in both
orders from bit-identical initial states. The order contrast of their predictions is
compared against the true order contrast, centred, at the endpoint and along the whole
path. The ratio of those two error ratios is the dissociation this run exists to measure.

The predictor it is tested against, 11.670, is a property of the observable and is not
recomputed here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "band_check"))
import _runtime_path  # noqa: F401,E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jr  # noqa: E402
import numpy as np  # noqa: E402

import timing_probe as tp  # noqa: E402

# Where results go. Defaults to this directory, which is how the script has always
# behaved, but a second run in place OVERWRITES the first and that has already happened
# once here: only a manual copy to DISSOCIATION_RESULT_prehoc.json preserved the
# planned numbers. An output directory makes a repeat safe by construction and is
# what a determinism check needs.
DEFAULT_OUT = HERE.parents[1] / "results" / "tables" / "baseline"
OUT = (Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv
       else DEFAULT_OUT)
OUT.mkdir(parents=True, exist_ok=True)

K = 10                       # one application advances tau = 0.10
APPLICATIONS_PER_LEG = 10    # so one leg is tau = 1.00, matching the corpus
STEPS = 8000
BATCH = 64
SEEDS = (0, 1, 2, 3, 4)
EVAL_UNITS = 256
NARROW = (0.4865, 0.5171)

AVAILABLE_SIGNAL_RATIO = 11.670          # measured, not recomputed here
BANKED_OBSERVED_DISSOCIATION = 13.855    # reconstructed, for comparison only
D2_LOW, D2_HIGH = 3.890, 35.010
P6_MAX_E_ON = 1.573e-3


def relative_l2(pred, target):
    num = np.sqrt(((pred - target) ** 2).sum(axis=(1, 2)))
    den = np.sqrt((target ** 2).sum(axis=(1, 2)))
    return num / den


def train(pairs_in, pairs_out, grid, seed):
    params = tp.init_fno(jr.PRNGKey(seed))
    opt = tp.adam_init(params)
    key = jr.PRNGKey(seed + 9999)
    n = pairs_in.shape[0]
    for step in range(STEPS):
        key, batch_key = jr.split(key)
        index = jr.randint(batch_key, (BATCH,), 0, n)
        lr = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + np.cos(np.pi * step / STEPS))
        params, opt, _ = tp.train_step(params, opt, pairs_in[index], pairs_out[index],
                                       grid, lr)
    return params


def compose_predicted(first, second, start, grid):
    """Apply one surrogate ten times, then the other ten, keeping every frame."""
    frames, state = [start], start
    for params in (first, second):
        for _ in range(APPLICATIONS_PER_LEG):
            state = tp.fno_apply(params, state, grid)
            frames.append(state)
    return np.asarray(jnp.stack(frames, axis=1), dtype=np.float64).squeeze(2)


def compose_true(first, second, start):
    frames, state = [start], start
    for stepper in (first, second):
        for _ in range(APPLICATIONS_PER_LEG):
            state = tp.rollout(stepper, state, K)[:, -1]
            frames.append(state)
    return np.asarray(jnp.stack(frames, axis=1), dtype=np.float64).squeeze(2)


def centred_ratio(predicted, true):
    """Learner error on the unit-specific order field, over a no-structure predictor."""
    centred_true = true - true.mean(axis=0, keepdims=True)
    centred_pred = predicted - predicted.mean(axis=0, keepdims=True)
    return float(np.sqrt(((centred_pred - centred_true) ** 2).mean())
                 / np.sqrt((centred_true ** 2).mean()))


def main() -> None:
    started = time.perf_counter()

    diffusion, reaction = tp.build_steppers(tp.DT)
    grid = jnp.linspace(0.0, tp.DOMAIN_EXTENT, tp.NUM_POINTS, endpoint=False)
    train_ic = tp.initial_conditions(tp.NUM_TRAIN_UNITS, NARROW, tp.MASTER_SEED + 1)
    eval_ic = tp.initial_conditions(EVAL_UNITS, NARROW, tp.MASTER_SEED + 11)

    on_target = {}
    pairs = {}
    for name, stepper in (("diffusion", diffusion), ("reaction", reaction)):
        states = tp.rollout(stepper, train_ic, tp.STEPS_PER_TRAJECTORY)
        pairs[name] = (states[:, :-K].reshape(-1, 1, tp.NUM_POINTS),
                       states[:, K:].reshape(-1, 1, tp.NUM_POINTS))
        on_target[name] = tp.rollout(stepper, eval_ic, K)[:, -1]

    # Ground truth composition, both orders, from the same initial states.
    true_ab = compose_true(diffusion, reaction, eval_ic)
    true_ba = compose_true(reaction, diffusion, eval_ic)
    true_contrast = true_ab - true_ba

    fitted, e_on = {}, {}
    for name in ("diffusion", "reaction"):
        for seed in SEEDS:
            mark = time.perf_counter()
            fitted[(name, seed)] = train(*pairs[name], grid, seed)
            predicted = np.asarray(tp.fno_apply(fitted[(name, seed)], eval_ic, grid),
                                   np.float64)
            e_on[(name, seed)] = float(np.median(relative_l2(
                predicted, np.asarray(on_target[name], np.float64))))
            print(f"  {name:9} seed {seed}  e_on {e_on[(name, seed)]:.6e}  "
                  f"{time.perf_counter() - mark:6.1f} s")

    d3 = max(e_on.values()) <= P6_MAX_E_ON
    print(f"\nD3 convergence  {'PASS' if d3 else 'FAIL'}  worst e_on "
          f"{max(e_on.values()):.6e} against {P6_MAX_E_ON}")
    if not d3:
        raise SystemExit("D3 failed: the comparison would be uninformative. FLAWED.")

    per_seed = {}
    for seed in SEEDS:
        pred_ab = compose_predicted(fitted[("diffusion", seed)],
                                    fitted[("reaction", seed)], eval_ic, grid)
        pred_ba = compose_predicted(fitted[("reaction", seed)],
                                    fitted[("diffusion", seed)], eval_ic, grid)
        pred_contrast = pred_ab - pred_ba
        endpoint = centred_ratio(pred_contrast[:, -1:], true_contrast[:, -1:])
        trajectory = centred_ratio(pred_contrast, true_contrast)
        # POST HOC, added after the planned result was in hand and labelled as
        # such. The plan predicted the dissociation from the available-signal ratio while
        # assuming the learner's own error is comparable on both observables. That
        # assumption is testable rather than necessary: if the law is
        # S_o = eps_o / r_o, then the dissociation is the signal ratio times the
        # learner's OWN endpoint-to-trajectory error ratio. Recording eps here lets that
        # sharper form be checked without another fit.
        eps_end = float(np.median(
            0.5 * (relative_l2(pred_ab[:, -1:], true_ab[:, -1:])
                   + relative_l2(pred_ba[:, -1:], true_ba[:, -1:]))))
        eps_traj = float(np.median(
            0.5 * (relative_l2(pred_ab, true_ab) + relative_l2(pred_ba, true_ba))))
        per_seed[seed] = {"endpoint_ratio": endpoint, "trajectory_ratio": trajectory,
                          "dissociation": endpoint / trajectory,
                          "eps_endpoint": eps_end, "eps_trajectory": eps_traj,
                          "eps_ratio": eps_end / eps_traj,
                          "sharpened_prediction": AVAILABLE_SIGNAL_RATIO
                          * (eps_end / eps_traj)}
        print(f"  seed {seed}  trajectory {trajectory:.6f}  endpoint {endpoint:.6f}  "
              f"dissociation {endpoint / trajectory:.3f}")

    dissociations = [per_seed[s]["dissociation"] for s in SEEDS]
    d_obs = float(np.median(dissociations))
    d1 = bool(d_obs >= AVAILABLE_SIGNAL_RATIO)
    d2 = bool(D2_LOW <= d_obs <= D2_HIGH)
    agreement = abs(d_obs - AVAILABLE_SIGNAL_RATIO) / AVAILABLE_SIGNAL_RATIO
    banked_agreement = abs(BANKED_OBSERVED_DISSOCIATION - AVAILABLE_SIGNAL_RATIO) \
        / AVAILABLE_SIGNAL_RATIO

    print(f"\npredicted by available signal  {AVAILABLE_SIGNAL_RATIO}")
    print(f"observed dissociation          {d_obs:.3f}  "
          f"range {min(dissociations):.3f} to {max(dissociations):.3f}")
    print(f"agreement                      {agreement * 100:.1f}%   "
          f"(destroyed run: {banked_agreement * 100:.1f}%)")
    print(f"  D1 at or above the ratio     {'HELD' if d1 else 'FAILED'}")
    print(f"  D2 within a factor of three  {'HELD' if d2 else 'FAILED'}")

    payload = {
        "schema": "pde-dissociation-second-architecture-v1",
        "generated": "2026-09-05",
        "architecture": "FNO-1d at published defaults, one per primitive",
        "estimand": f"k={K}, {APPLICATIONS_PER_LEG} applications per leg",
        "eval_units": EVAL_UNITS,
        "predicted_by_available_signal": AVAILABLE_SIGNAL_RATIO,
        "observed_dissociation": d_obs,
        "observed_range": [float(min(dissociations)), float(max(dissociations))],
        "agreement_fraction": agreement,
        "banked_reconstructed_dissociation": BANKED_OBSERVED_DISSOCIATION,
        "banked_agreement_fraction": banked_agreement,
        "D1_at_or_above_ratio": {"held": d1},
        "D2_within_factor_three": {"held": d2, "band": [D2_LOW, D2_HIGH]},
        "D3_convergence": {"held": d3, "worst_e_on": float(max(e_on.values()))},
        "per_seed": {str(s): per_seed[s] for s in SEEDS},
        "post_hoc_sharpened": {
            "form": ("dissociation = available-signal ratio x the learner's own "
                     "endpoint-to-trajectory error ratio; the planned form "
                     "assumed that second factor is 1"),
            "labelled": "POST HOC, added after the planned result was in hand",
            "median_eps_ratio": float(np.median(
                [per_seed[s]["eps_ratio"] for s in SEEDS])),
            "median_sharpened_prediction": float(np.median(
                [per_seed[s]["sharpened_prediction"] for s in SEEDS])),
            "per_seed_sharpened_agreement": {
                str(s): abs(per_seed[s]["dissociation"]
                            - per_seed[s]["sharpened_prediction"])
                / per_seed[s]["sharpened_prediction"] for s in SEEDS},
        },
        "per_seed_e_on": {f"{n}_{s}": e_on[(n, s)]
                          for n in ("diffusion", "reaction") for s in SEEDS},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (OUT / "DISSOCIATION_RESULT.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nelapsed {payload['elapsed_seconds'] / 60:.1f} min, CPU, $0")


if __name__ == "__main__":
    main()
