"""Measure the two planned quantities on the second generator pair.

Every prediction is compared here by computation. The verdict fields are produced by
comparing a measured value against a bound the plan fixed before any of this was
computed, not by someone reading two numbers next to each other and forming an
impression.
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

# Identical to the original corpus. Only the second operator changes.
NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, DT = 1, 1.0, 256, 0.01
ETDRK_ORDER, DEALIASING_FRACTION = 2, 2 / 3
NUM_CIRCLE_POINTS, CIRCLE_RADIUS = 16, 1.0
STEPS_PER_LEG = 100
IC_CUTOFF, IC_AMPLITUDE = 5, 0.175
NARROW_OFFSET_RANGE = (0.4865, 0.5171)
TRAIN_UNITS, TEST_UNITS = 640, 48
MASTER_SEED = 20260905

# Bounds, copied from the plan.
P1_INTERVAL, P1_TOLERANCE = (0.4865, 0.5171), 0.005
P2_INTERVAL = (0.80, 0.89)
P3_MAX_FRACTION_INSIDE = 0.0
P4_MIN_RATIO = 1.0
COMMUTING_NULL = 4.996e-16


def diffusion():
    return ex.stepper.Diffusion(NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, DT,
                                diffusivity=0.01)


def allen_cahn():
    return ex.stepper.reaction.AllenCahn(
        NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, DT, diffusivity=0.005,
        first_order_coefficient=1.0, third_order_coefficient=-1.0, order=ETDRK_ORDER,
        dealiasing_fraction=DEALIASING_FRACTION, num_circle_points=NUM_CIRCLE_POINTS,
        circle_radius=CIRCLE_RADIUS)


def initial_conditions(units: int, seed: int):
    base = ex.ic.RandomTruncatedFourierSeries(NUM_SPATIAL_DIMS, cutoff=IC_CUTOFF,
                                              std_one=True)
    field_key, offset_key = jr.split(jr.PRNGKey(seed))
    keys = jr.split(field_key, units)
    raw = jnp.stack([base(NUM_POINTS, key=k) for k in keys])
    offsets = jr.uniform(offset_key, shape=(units, 1, 1),
                         minval=NARROW_OFFSET_RANGE[0], maxval=NARROW_OFFSET_RANGE[1])
    return jnp.clip(offsets + IC_AMPLITUDE * raw, 0.0, 1.0)


def roll(stepper, batch, steps):
    step = jax.jit(jax.vmap(stepper))
    state, frames = batch, [batch]
    for _ in range(steps):
        state = step(state)
        frames.append(state)
    return jnp.stack(frames, axis=1)


def compose(first, second, u0):
    one = roll(first, u0, STEPS_PER_LEG)
    two = roll(second, one[:, -1], STEPS_PER_LEG)
    return jnp.concatenate([one, two[:, 1:]], axis=1)


def energy_split(field: np.ndarray) -> dict:
    """Split an across-unit field into its shared mean and centered components.

    Identical to the function used on the first pair, so the two are comparable.
    """
    mean_field = field.mean(axis=0, keepdims=True)
    centered = field - mean_field
    total = float((field ** 2).mean())
    shared = float((np.broadcast_to(mean_field, field.shape) ** 2).mean())
    centred_energy = float((centered ** 2).mean())
    return {
        "shared_fraction": shared / total,
        "centered_fraction": centred_energy / total,
        "centered_rms": float(np.sqrt(centred_energy)),
        "field_rms": float(np.sqrt(total)),
    }


def spatial_means(states) -> np.ndarray:
    return np.asarray(states.mean(axis=-1)).reshape(-1)


def main() -> None:
    started = time.perf_counter()

    d, ac = diffusion(), allen_cahn()
    train_ic = initial_conditions(TRAIN_UNITS, MASTER_SEED + 1)
    test_ic = initial_conditions(TEST_UNITS, MASTER_SEED + 4)

    # P1. Every state the diffusion lane sees during primitive training.
    train_states = roll(d, train_ic, STEPS_PER_LEG)
    train_means = spatial_means(train_states)
    support = (float(train_means.min()), float(train_means.max()))
    p1_low = abs(support[0] - P1_INTERVAL[0]) <= P1_TOLERANCE
    p1_high = abs(support[1] - P1_INTERVAL[1]) <= P1_TOLERANCE
    p1 = bool(p1_low and p1_high)

    # P2. The states an Allen-Cahn-first ordering hands to the diffusion lane.
    switch_states = roll(ac, test_ic, STEPS_PER_LEG)[:, -1]
    switch_means = spatial_means(switch_states)
    switch_interval = (float(switch_means.min()), float(switch_means.max()))
    p2 = bool(switch_interval[0] >= P2_INTERVAL[0]
              and switch_interval[1] <= P2_INTERVAL[1])

    # P3. Overlap, the prediction the mechanism rests on.
    inside = (switch_means >= support[0]) & (switch_means <= support[1])
    fraction_inside = float(inside.mean())
    nearest = np.minimum(np.abs(switch_means - support[0]),
                         np.abs(switch_means - support[1]))
    nearest = np.where(inside, 0.0, nearest)
    p3 = bool(fraction_inside <= P3_MAX_FRACTION_INSIDE)

    print("P1  diffusion training-state spatial means")
    print(f"      measured [{support[0]:.6f}, {support[1]:.6f}]  "
          f"predicted [{P1_INTERVAL[0]}, {P1_INTERVAL[1]}] +/- {P1_TOLERANCE}"
          f"   {'HELD' if p1 else 'FAILED'}")
    print("P2  Allen-Cahn-first switch-state spatial means")
    print(f"      measured [{switch_interval[0]:.6f}, {switch_interval[1]:.6f}]  "
          f"predicted [{P2_INTERVAL[0]}, {P2_INTERVAL[1]}]"
          f"   {'HELD' if p2 else 'FAILED'}")
    print("P3  overlap")
    print(f"      fraction of switch states inside the training support "
          f"{fraction_inside:.6f}   {'HELD' if p3 else 'FAILED'}")
    print(f"      units off support {int((~inside).sum())} / {TEST_UNITS}, "
          f"maximum distance to the support {nearest.max():.6f}")

    # Order dependence on the full cohort, as promised in section 4 of the plan.
    forward = compose(d, ac, test_ic)
    reverse = compose(ac, d, test_ic)
    endpoint_contrast = np.asarray(forward[:, -1] - reverse[:, -1])
    trajectory_contrast = np.asarray(forward - reverse)
    order_dependence = float(np.abs(endpoint_contrast).max())

    endpoint_split = energy_split(endpoint_contrast)
    trajectory_split = energy_split(trajectory_contrast)
    endpoint_state_rms = float(np.sqrt((np.asarray(forward[:, -1]) ** 2).mean()))
    trajectory_state_rms = float(np.sqrt((np.asarray(forward) ** 2).mean()))
    endpoint_resolution = endpoint_split["centered_rms"] / endpoint_state_rms
    trajectory_resolution = trajectory_split["centered_rms"] / trajectory_state_rms
    signal_ratio = trajectory_resolution / endpoint_resolution
    p4 = bool(signal_ratio > P4_MIN_RATIO)
    p5 = bool(endpoint_split["shared_fraction"] > trajectory_split["shared_fraction"])

    print(f"\n      order dependence on the full cohort {order_dependence:.4e}, "
          f"against a commuting null of {COMMUTING_NULL:.3e}")
    print("P4  unit-specific order signal relative to state scale")
    print(f"      endpoint   {endpoint_resolution:.6f}")
    print(f"      trajectory {trajectory_resolution:.6f}")
    print(f"      trajectory / endpoint {signal_ratio:.3f}   "
          f"predicted greater than {P4_MIN_RATIO}   {'HELD' if p4 else 'FAILED'}")
    print("P5  shared fraction of the order contrast")
    print(f"      endpoint   {endpoint_split['shared_fraction']:.5f}")
    print(f"      trajectory {trajectory_split['shared_fraction']:.5f}   "
          f"{'HELD' if p5 else 'FAILED'}")

    elapsed = time.perf_counter() - started
    held = {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5}
    print(f"\n{sum(held.values())} of 5 predictions held: "
          + ", ".join(f"{k}={'held' if v else 'FAILED'}" for k, v in held.items()))
    print(f"elapsed {elapsed:.2f} s, CPU, $0")

    payload = {
        "schema": "pde-second-pair-measurement-v1",
        "generated": "2026-09-05",
        "pair": "diffusion_allen_cahn",
        "status": ("NEW EXPERIMENT on a second operator pair. Does not amend or replace "
                   "any result from the first pair."),
        "not_tested": ("whether broadening the support repairs a learner's composed "
                       "prediction on this pair, and whether the signal ratio predicts "
                       "an observed dissociation. Both need the destroyed learner."),
        "predictions_held": held,
        "P1_training_support": {"measured": list(support),
                                "predicted": list(P1_INTERVAL),
                                "tolerance": P1_TOLERANCE, "held": p1},
        "P2_switch_support": {"measured": list(switch_interval),
                              "predicted": list(P2_INTERVAL), "held": p2},
        "P3_overlap": {"fraction_inside": fraction_inside,
                       "units_off_support": int((~inside).sum()),
                       "of_units": TEST_UNITS,
                       "max_distance_to_support": float(nearest.max()),
                       "predicted_max_fraction": P3_MAX_FRACTION_INSIDE, "held": p3},
        "P4_signal_ratio": {"endpoint_resolution": endpoint_resolution,
                            "trajectory_resolution": trajectory_resolution,
                            "ratio": signal_ratio, "predicted_min": P4_MIN_RATIO,
                            "held": p4,
                            "first_pair_ratio_for_comparison": 11.670},
        "P5_shared_fraction": {"endpoint": endpoint_split["shared_fraction"],
                               "trajectory": trajectory_split["shared_fraction"],
                               "held": p5,
                               "first_pair_for_comparison": {"endpoint": 0.83761,
                                                             "trajectory": 0.64480}},
        "order_dependence": {"measured": order_dependence,
                             "commuting_null_for_scale": COMMUTING_NULL},
        "solver": {"domain_extent": DOMAIN_EXTENT, "num_points": NUM_POINTS, "dt": DT,
                   "steps_per_leg": STEPS_PER_LEG, "etdrk_order": ETDRK_ORDER,
                   "diffusivity": 0.01, "allen_cahn_diffusivity": 0.005,
                   "first_order_coefficient": 1.0, "third_order_coefficient": -1.0},
        "cohorts": {"primitive_train_units": TRAIN_UNITS, "test_units": TEST_UNITS,
                    "master_seed": MASTER_SEED,
                    "offset_range": list(NARROW_OFFSET_RANGE)},
        "elapsed_seconds": elapsed,
    }
    (OUT / "SECOND_PAIR_RESULT.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote SECOND_PAIR_RESULT.json")


if __name__ == "__main__":
    main()
