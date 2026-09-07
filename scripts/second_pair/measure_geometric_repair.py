"""Does broadening the training support repair the overlap on the second pair?

The geometric half of the remedy only. No learner is fitted here and none is implied:
this measures whether the training-state support can be made to cover the switch states,
not whether a fitted model's asymmetry closes once it does.

Both broadened conditions are fixed in the plan, including condition B, which exists
because we expect A to fail. A remedy chosen after watching A
fail would be worth nothing.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Writing beside the script overwrites a banked result when the script is re-run in
# place. run_dissociation.py had the same defect and it cost us the first-run file.
# An explicit output directory makes a repeat safe by construction.
DEFAULT_OUT = HERE.parents[1] / "results" / "tables" / "second_pair"
OUT = (Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv
       else DEFAULT_OUT)
sys.path.insert(0, str(HERE.parent / "band_check"))
import _runtime_path  # noqa: F401

import exponax as ex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_enable_x64", True)

NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, DT = 1, 1.0, 256, 0.01
ETDRK_ORDER, DEALIASING_FRACTION = 2, 2 / 3
NUM_CIRCLE_POINTS, CIRCLE_RADIUS = 16, 1.0
STEPS_PER_LEG = 100
IC_CUTOFF, IC_AMPLITUDE = 5, 0.175
TRAIN_UNITS, TEST_UNITS = 640, 48
MASTER_SEED = 20260905
UPPER_BOUND = 1.0
SATURATION_LIMIT = 0.10                      # section 5 of the plan

CONDITIONS = {
    "narrow": (0.4865, 0.5171),
    "A_first_pair_rule": (0.48, 0.85),
    "B_declared_in_advance": (0.48, 0.90),
}


def diffusion():
    return ex.stepper.Diffusion(NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, DT,
                                diffusivity=0.01)


def allen_cahn():
    return ex.stepper.reaction.AllenCahn(
        NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, DT, diffusivity=0.005,
        first_order_coefficient=1.0, third_order_coefficient=-1.0, order=ETDRK_ORDER,
        dealiasing_fraction=DEALIASING_FRACTION, num_circle_points=NUM_CIRCLE_POINTS,
        circle_radius=CIRCLE_RADIUS)


def initial_conditions(units: int, offset_range: tuple[float, float], seed: int):
    base = ex.ic.RandomTruncatedFourierSeries(NUM_SPATIAL_DIMS, cutoff=IC_CUTOFF,
                                              std_one=True)
    field_key, offset_key = jr.split(jr.PRNGKey(seed))
    keys = jr.split(field_key, units)
    raw = jnp.stack([base(NUM_POINTS, key=k) for k in keys])
    offsets = jr.uniform(offset_key, shape=(units, 1, 1),
                         minval=offset_range[0], maxval=offset_range[1])
    return jnp.clip(offsets + IC_AMPLITUDE * raw, 0.0, UPPER_BOUND)


def roll(stepper, batch, steps):
    step = jax.jit(jax.vmap(stepper))
    state, frames = batch, [batch]
    for _ in range(steps):
        state = step(state)
        frames.append(state)
    return jnp.stack(frames, axis=1)


def main() -> None:
    started = time.perf_counter()

    d, ac = diffusion(), allen_cahn()

    # The switch states are unchanged by any of this: they are what the Allen-Cahn leg
    # produces from the test cohort, and broadening the DIFFUSION lane cannot move them.
    test_ic = initial_conditions(TEST_UNITS, (0.4865, 0.5171), MASTER_SEED + 4)
    switch = np.asarray(roll(ac, test_ic, STEPS_PER_LEG)[:, -1].mean(axis=-1)).reshape(-1)
    print(f"switch-state spatial means [{switch.min():.6f}, {switch.max():.6f}], "
          f"{TEST_UNITS} units\n")

    rows = {}
    for name, offsets in CONDITIONS.items():
        train_ic = initial_conditions(TRAIN_UNITS, offsets, MASTER_SEED + 1)
        states = roll(d, train_ic, STEPS_PER_LEG)
        means = np.asarray(states.mean(axis=-1)).reshape(-1)
        support = (float(means.min()), float(means.max()))
        inside = (switch >= support[0]) & (switch <= support[1])
        fraction_inside = float(inside.mean())
        distance = np.where(inside, 0.0,
                            np.minimum(np.abs(switch - support[0]),
                                       np.abs(switch - support[1])))
        # Distance to the nearest training mean, the quantity the first pair reported.
        nearest = np.abs(means[None, :] - switch[:, None]).min(axis=1)
        saturated = float((np.asarray(states) >= UPPER_BOUND - 1e-12).mean())
        rows[name] = {
            "offset_range": list(offsets), "support": list(support),
            "fraction_inside": fraction_inside,
            "units_off_support": int((~inside).sum()),
            "max_distance_to_support": float(distance.max()),
            "max_nearest_training_mean": float(nearest.max()),
            "saturated_value_fraction": saturated,
        }
        print(f"{name:22} offsets {offsets}")
        print(f"  training-mean support [{support[0]:.6f}, {support[1]:.6f}]")
        print(f"  fraction of switch states inside {fraction_inside:.6f}   "
              f"off support {int((~inside).sum())} / {TEST_UNITS}")
        print(f"  max distance to support {distance.max():.6f}   "
              f"max nearest training mean {nearest.max():.6f}")
        print(f"  training values at the upper bound {saturated:.6f}\n")

    g1 = bool(rows["A_first_pair_rule"]["fraction_inside"] < 1.0)
    g2 = bool(rows["B_declared_in_advance"]["fraction_inside"] == 1.0)
    closing = [n for n in ("A_first_pair_rule", "B_declared_in_advance")
               if rows[n]["fraction_inside"] == 1.0]
    cost = {n: rows[n]["saturated_value_fraction"] > SATURATION_LIMIT for n in closing}

    print(f"G1 condition A leaves the overlap open   {'HELD' if g1 else 'FAILED'}")
    print(f"G2 condition B closes the overlap        {'HELD' if g2 else 'FAILED'}")
    for name, expensive in cost.items():
        print(f"   {name} closes with "
              f"{rows[name]['saturated_value_fraction']:.4f} of training values at the "
              f"bound, limit {SATURATION_LIMIT}: "
              f"{'REPORTED AS ACHIEVED AT A COST' if expensive else 'clean repair'}")
    if not closing:
        print("   neither condition closed the overlap: the geometric repair does not "
              "transfer to this pair, and no condition C will be searched for")

    elapsed = time.perf_counter() - started
    payload = {
        "schema": "pde-second-pair-geometric-repair-v1",
        "generated": "2026-09-05",
        "pair": "diffusion_allen_cahn",
        "tests": ("the geometric half of the remedy only: whether broadening the "
                  "training support repairs the overlap. No learner is fitted, and "
                  "nothing here says the composed asymmetry closes on this pair."),
        "switch_state_means": [float(switch.min()), float(switch.max())],
        "conditions": rows,
        "G1_condition_A_leaves_overlap_open": {"held": g1},
        "G2_condition_B_closes_overlap": {"held": g2},
        "conditions_that_closed": closing,
        "saturation_limit": SATURATION_LIMIT,
        "closed_at_a_cost": cost,
        "first_pair_for_comparison": {"broadened_offsets": [0.48, 0.75],
                                      "fraction_inside": 1.0,
                                      "max_nearest_training_mean": 0.0089089046},
        "elapsed_seconds": elapsed,
    }
    (OUT / "GEOMETRIC_REPAIR_RESULT.json").write_text(json.dumps(payload, indent=2)
                                                       + "\n")
    print(f"\nelapsed {elapsed:.2f} s, CPU, $0")
    print("wrote GEOMETRIC_REPAIR_RESULT.json")


if __name__ == "__main__":
    main()
