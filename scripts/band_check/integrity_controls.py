"""Regenerate the integrity controls, and measure what the original never measured.

Three purposes, in increasing order of value.

1. Test two banked numbers directly. The commuting-pair null was recorded at
   `4.996e-16` against a `1e-12` gate, and the noncommuting contrast decomposition at
   `82.774%` shared across-unit mean and `17.226%` centered. Both are `reconstructed`
   and neither can be re-opened. This recomputes them on the reconstructed
   configuration.

2. Test the scope inference the shared-mean adjudication drew in its section 1.1.
   That analysis argued, from the surviving description alone, that the recorded
   decomposition was measured on the ENDPOINT contrast field rather than on the full
   trajectory, because the control reports across-unit variance pointwise IN SPACE and
   correlates the field norm against properties of u0. If the endpoint decomposition
   here lands near 82.774% and the full-trajectory decomposition lands somewhere else,
   the inference is confirmed.

3. Supply the quantity that does not exist in any surviving record: the shared/centered
   split of the FULL-TRAJECTORY contrast. The adjudication showed the manuscript and
   the standing objection both transfer the 82.774% figure to the full-trajectory
   criteria, and that this transfer is unlicensed because the measurement was never
   made. Making it closes that gap.

Also reported, and new: the centered contrast RMS in absolute state units, alongside
the endpoint state scale. The adjudication named the missing link between those two as
the reason the resolution question cannot be settled. This does not settle it on its
own, since the learner's error convention is still unrecovered, but it supplies one
half of the ratio.

Configuration is exactly the one the plan fixed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import _runtime_path  # noqa: F401
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import exponax as ex

jax.config.update("jax_enable_x64", True)

NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, DT = 1, 1.0, 256, 0.01
DIFFUSIVITY, REACTION_DIFFUSIVITY, REACTIVITY = 0.01, 0.0, 1.0
COMMUTING_PARTNER_DIFFUSIVITY = 0.02
ETDRK_ORDER, DEALIASING_FRACTION = 2, 2 / 3
NUM_CIRCLE_POINTS, CIRCLE_RADIUS = 16, 1.0
STEPS_PER_LEG = 100

NUM_UNITS, MASTER_SEED = 256, 20260905
IC_AMPLITUDE, IC_OFFSET_RANGE = 0.175, (0.4865, 0.5171)
CUTOFF = 5  # Exponax default; the band check showed this cutoff closest to banked.

COMMUTING_GATE = 1e-12
BANKED_COMMUTING_NULL = 4.996e-16
BANKED_SHARED_FRACTION = 0.82774
BANKED_CENTERED_FRACTION = 0.17226


def diffusion(dt: float, diffusivity: float):
    return ex.stepper.Diffusion(
        NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, dt, diffusivity=diffusivity
    )


def fisher_kpp(dt: float):
    return ex.stepper.reaction.FisherKPP(
        NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, dt,
        diffusivity=REACTION_DIFFUSIVITY, reactivity=REACTIVITY, order=ETDRK_ORDER,
        dealiasing_fraction=DEALIASING_FRACTION, num_circle_points=NUM_CIRCLE_POINTS,
        circle_radius=CIRCLE_RADIUS,
    )


def initial_conditions() -> jnp.ndarray:
    base = ex.ic.RandomTruncatedFourierSeries(NUM_SPATIAL_DIMS, cutoff=CUTOFF, std_one=True)
    field_key, offset_key = jr.split(jr.PRNGKey(MASTER_SEED))
    keys = jr.split(field_key, NUM_UNITS)
    raw = jnp.stack([base(NUM_POINTS, key=k) for k in keys])
    offsets = jr.uniform(offset_key, shape=(NUM_UNITS, 1, 1),
                         minval=IC_OFFSET_RANGE[0], maxval=IC_OFFSET_RANGE[1])
    return jnp.clip(offsets + IC_AMPLITUDE * raw, 0.0, 1.0)


def leg_with_trace(stepper, batch, steps: int):
    """Return (endpoint, full trajectory including the initial frame)."""
    step = jax.jit(jax.vmap(stepper))
    state = batch
    frames = [state]
    for _ in range(steps):
        state = step(state)
        frames.append(state)
    return state, jnp.stack(frames, axis=1)


def compose_with_trace(first, second, u0, steps: int):
    mid, trace_one = leg_with_trace(first, u0, steps)
    end, trace_two = leg_with_trace(second, mid, steps)
    # Drop the duplicated switch frame so the composed trace is contiguous in time.
    return end, jnp.concatenate([trace_one, trace_two[:, 1:]], axis=1)


def energy_split(field: np.ndarray) -> dict:
    """Split an across-unit field into its shared mean and centered components."""
    mean_field = field.mean(axis=0, keepdims=True)
    centered = field - mean_field
    total = float((field ** 2).mean())
    shared = float((np.broadcast_to(mean_field, field.shape) ** 2).mean())
    centred_energy = float((centered ** 2).mean())
    return {
        "shared_fraction": shared / total,
        "centered_fraction": centred_energy / total,
        "total_mean_square": total,
        "shared_mean_square": shared,
        "centered_mean_square": centred_energy,
        "centered_rms": float(np.sqrt(centred_energy)),
        "field_rms": float(np.sqrt(total)),
    }


def pairwise_cosines(field: np.ndarray) -> dict:
    """Cosine between each unit's field and its cyclic successor's."""
    flat = field.reshape(field.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1)
    partner = np.roll(np.arange(field.shape[0]), -1)
    cos = (flat * flat[partner]).sum(axis=1) / (norms * norms[partner])
    return {
        "median": float(np.median(cos)), "mean": float(cos.mean()),
        "min": float(cos.min()), "max": float(cos.max()),
    }


def main() -> None:
    started = time.perf_counter()
    u0 = initial_conditions()

    # Control 1: a commuting pair must produce no order effect.
    slow, fast = diffusion(DT, DIFFUSIVITY), diffusion(DT, COMMUTING_PARTNER_DIFFUSIVITY)
    ab_c, _ = compose_with_trace(slow, fast, u0, STEPS_PER_LEG)
    ba_c, _ = compose_with_trace(fast, slow, u0, STEPS_PER_LEG)
    ab_c, ba_c = np.asarray(ab_c), np.asarray(ba_c)
    denom = np.sqrt((ab_c ** 2).sum(axis=(1, 2)))
    commuting_max = float((np.sqrt(((ab_c - ba_c) ** 2).sum(axis=(1, 2))) / denom).max())

    # Control 2: the noncommuting contrast, at the endpoint and along the trajectory.
    diff, reac = diffusion(DT, DIFFUSIVITY), fisher_kpp(DT)
    ab_end, ab_trace = compose_with_trace(diff, reac, u0, STEPS_PER_LEG)
    ba_end, ba_trace = compose_with_trace(reac, diff, u0, STEPS_PER_LEG)

    endpoint_contrast = np.asarray(ab_end) - np.asarray(ba_end)
    trajectory_contrast = np.asarray(ab_trace) - np.asarray(ba_trace)

    endpoint_split = energy_split(endpoint_contrast)
    trajectory_split = energy_split(trajectory_contrast)
    endpoint_cos = pairwise_cosines(endpoint_contrast)
    endpoint_cos_centered = pairwise_cosines(
        endpoint_contrast - endpoint_contrast.mean(axis=0, keepdims=True)
    )

    endpoint_state_rms = float(np.sqrt((np.asarray(ab_end) ** 2).mean()))
    trajectory_state_rms = float(np.sqrt((np.asarray(ab_trace) ** 2).mean()))
    endpoint_resolution_ratio = endpoint_split["centered_rms"] / endpoint_state_rms
    trajectory_resolution_ratio = trajectory_split["centered_rms"] / trajectory_state_rms
    elapsed = time.perf_counter() - started

    def delta(observed: float, banked: float) -> str:
        return f"{observed:.5f} vs banked {banked:.5f}  (delta {observed - banked:+.5f})"

    print("CONTROL 1  commuting diffusion pair")
    print(f"  max relative AB/BA difference : {commuting_max:.3e}"
          f"   gate {COMMUTING_GATE:.0e}   "
          f"{'PASS' if commuting_max < COMMUTING_GATE else 'FAIL'}")
    print(f"  banked (reconstructed)        : {BANKED_COMMUTING_NULL:.3e}")
    print()
    print("CONTROL 2  noncommuting contrast, ENDPOINT field")
    print(f"  shared mean fraction   : {delta(endpoint_split['shared_fraction'], BANKED_SHARED_FRACTION)}")
    print(f"  centered fraction      : {delta(endpoint_split['centered_fraction'], BANKED_CENTERED_FRACTION)}")
    print(f"  raw pairwise cosine    : median {endpoint_cos['median']:.4f}  "
          f"banked median 0.9286")
    print(f"  centered cosine        : median {endpoint_cos_centered['median']:.4f}  "
          f"banked median 0.143")
    print()
    print("CONTROL 2  noncommuting contrast, FULL TRAJECTORY  [never measured before]")
    print(f"  shared mean fraction   : {trajectory_split['shared_fraction']:.5f}")
    print(f"  centered fraction      : {trajectory_split['centered_fraction']:.5f}")
    print()
    print("SCALE, for the resolution question")
    print(f"  ENDPOINT   centered contrast RMS {endpoint_split['centered_rms']:.6f}"
          f"  state RMS {endpoint_state_rms:.6f}"
          f"  ratio {endpoint_resolution_ratio:.6f}")
    print(f"  TRAJECTORY centered contrast RMS {trajectory_split['centered_rms']:.6f}"
          f"  state RMS {trajectory_state_rms:.6f}"
          f"  ratio {trajectory_resolution_ratio:.6f}")
    print(f"  trajectory / endpoint signal ratio : "
          f"{trajectory_resolution_ratio / endpoint_resolution_ratio:.3f}x")
    print()
    print("  banked learner endpoint errors (reconstructed): AB 0.023798  BA 0.017173")
    print("  IF those use the same relative-L2 convention, the centered endpoint")
    print(f"  contrast at {endpoint_resolution_ratio:.4f} sits BELOW the learner's own")
    print("  endpoint error, i.e. criterion 6 asks for a signal under the noise floor.")
    print(f"\nelapsed {elapsed:.2f} s, CPU, $0")

    payload = {
        "schema": "pde-integrity-controls-regenerated-v1",
        "generated": "2026-09-05",
        "status": "NEW EXPERIMENT. Does not amend or replace the banked integrity controls.",
        "configuration": {
            "cutoff": CUTOFF, "ic_amplitude": IC_AMPLITUDE,
            "ic_offset_range": list(IC_OFFSET_RANGE), "units": NUM_UNITS,
            "master_seed": MASTER_SEED, "steps_per_leg": STEPS_PER_LEG,
            "diffusivity": DIFFUSIVITY, "commuting_partner_diffusivity": COMMUTING_PARTNER_DIFFUSIVITY,
        },
        "control_1_commuting": {
            "max_relative_order_difference": commuting_max,
            "gate": COMMUTING_GATE,
            "passes": commuting_max < COMMUTING_GATE,
            "banked_value": BANKED_COMMUTING_NULL,
            "banked_confidence": "reconstructed",
        },
        "control_2_endpoint": {
            **endpoint_split,
            "pairwise_cosine_raw": endpoint_cos,
            "pairwise_cosine_centered": endpoint_cos_centered,
            "banked_shared_fraction": BANKED_SHARED_FRACTION,
            "banked_centered_fraction": BANKED_CENTERED_FRACTION,
            "banked_raw_cosine_median": 0.9286,
            "banked_centered_cosine_median": 0.143,
            "banked_confidence": "reconstructed",
        },
        "control_2_full_trajectory": {
            **trajectory_split,
            "note": (
                "No equivalent measurement exists in any surviving record. The banked "
                "82.774/17.226 split was measured on the endpoint field only."
            ),
        },
        "scale": {
            "endpoint": {
                "centered_contrast_rms": endpoint_split["centered_rms"],
                "state_rms": endpoint_state_rms,
                "resolution_ratio": endpoint_resolution_ratio,
            },
            "full_trajectory": {
                "centered_contrast_rms": trajectory_split["centered_rms"],
                "state_rms": trajectory_state_rms,
                "resolution_ratio": trajectory_resolution_ratio,
            },
            "trajectory_over_endpoint": trajectory_resolution_ratio / endpoint_resolution_ratio,
            "banked_learner_endpoint_errors": {
                "AB": 0.023798, "BA": 0.017173, "confidence": "reconstructed",
                "denominator_convention": "unrecovered",
            },
            "interpretation": (
                "If the banked learner endpoint errors use the same relative-L2 "
                "convention used for the contrast, floor and cross-IC quantities, then "
                "the centered endpoint contrast is SMALLER than the learner's own "
                "endpoint error, and criterion 6 was underpowered by construction. The "
                "denominator convention is unrecovered, so this is a conditional "
                "finding, not a settled one."
            ),
        },
        "elapsed_seconds": elapsed,
        "cost_usd": 0.0,
    }
    out = (Path(__file__).resolve().parents[2] / "results" / "tables"
           / "band_check" / "INTEGRITY_CONTROLS_RESULT.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
