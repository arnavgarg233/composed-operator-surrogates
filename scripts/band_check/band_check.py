"""Composition-scale band check at tau = 1.00.

Executes exactly what the plan declares, fixed before this file was run. Reports all
three cutoffs with no selection.

Banked comparison values, all `reconstructed` and NOT re-openable:
    numerical-floor passes   256 / 256   (required >= 142)
    median contrast / cross-IC  0.412779, 95% [0.367881, 0.455603]   band [0.3, 0.7]
    median contrast / floor     3546.30  (reported, not gated)
"""

from __future__ import annotations

import hashlib
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

RESULTS = Path(__file__).resolve().parents[2] / "results" / "tables" / "band_check"

NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS = 1, 1.0, 256
DT, HALF_DT = 0.01, 0.005
DIFFUSIVITY, REACTION_DIFFUSIVITY, REACTIVITY = 0.01, 0.0, 1.0
ETDRK_ORDER, DEALIASING_FRACTION = 2, 2 / 3
NUM_CIRCLE_POINTS, CIRCLE_RADIUS = 16, 1.0
WINDOW_DURATION = 1.00

NUM_UNITS = 256
MASTER_SEED = 20260905
IC_AMPLITUDE = 0.175
IC_OFFSET_RANGE = (0.4865, 0.5171)
CUTOFFS = (3, 5, 7)

THRESHOLD_MULTIPLIER = 10.0
MIN_PASSING_UNITS = 142
BAND = (0.3, 0.7)
BOOTSTRAP_DRAWS = 9999


def build(dt: float):
    diffusion = ex.stepper.Diffusion(
        NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, dt, diffusivity=DIFFUSIVITY
    )
    reaction = ex.stepper.reaction.FisherKPP(
        NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, dt,
        diffusivity=REACTION_DIFFUSIVITY, reactivity=REACTIVITY, order=ETDRK_ORDER,
        dealiasing_fraction=DEALIASING_FRACTION, num_circle_points=NUM_CIRCLE_POINTS,
        circle_radius=CIRCLE_RADIUS,
    )
    if float(np.asarray(reaction.diffusivity).reshape(-1)[0]) != REACTION_DIFFUSIVITY:
        raise SystemExit("component isolation failed: FisherKPP diffusivity is not zero")
    return diffusion, reaction


def initial_conditions(cutoff: int) -> jnp.ndarray:
    base = ex.ic.RandomTruncatedFourierSeries(
        NUM_SPATIAL_DIMS, cutoff=cutoff, std_one=True
    )
    field_key, offset_key = jr.split(jr.PRNGKey(MASTER_SEED))
    keys = jr.split(field_key, NUM_UNITS)
    raw = jnp.stack([base(NUM_POINTS, key=k) for k in keys])
    offsets = jr.uniform(
        offset_key, shape=(NUM_UNITS, 1, 1),
        minval=IC_OFFSET_RANGE[0], maxval=IC_OFFSET_RANGE[1],
    )
    return jnp.clip(offsets + IC_AMPLITUDE * raw, 0.0, 1.0)


def leg(stepper, batch, steps: int):
    state = batch
    step = jax.jit(jax.vmap(stepper))
    for _ in range(steps):
        state = step(state)
    return state


def compose(first, second, u0, steps: int):
    """Exact state handoff: the array leaving leg one enters leg two unchanged."""
    return leg(second, leg(first, u0, steps), steps)


def relative_l2(a: np.ndarray, b: np.ndarray, reference: np.ndarray) -> np.ndarray:
    axes = tuple(range(1, a.ndim))
    return np.sqrt(((a - b) ** 2).sum(axis=axes)) / np.sqrt((reference ** 2).sum(axis=axes))


def cross_ic_partners() -> np.ndarray:
    """Cyclic successor in the SHA-256 ordering of (master seed, unit index)."""
    permutation = sorted(
        range(NUM_UNITS),
        key=lambda unit: hashlib.sha256(f"{MASTER_SEED}:{unit}".encode()).digest(),
    )
    partner = np.empty(NUM_UNITS, dtype=int)
    for position, unit in enumerate(permutation):
        partner[unit] = permutation[(position + 1) % NUM_UNITS]
    return partner


def bootstrap_median(values: np.ndarray, draws: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(draws, values.size))
    medians = np.median(values[idx], axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def run_cutoff(cutoff: int) -> dict:
    coarse_steps = round(WINDOW_DURATION / DT)
    fine_steps = round(WINDOW_DURATION / HALF_DT)
    assert fine_steps == 2 * coarse_steps

    u0 = initial_conditions(cutoff)
    diff_c, reac_c = build(DT)
    diff_f, reac_f = build(HALF_DT)

    ab = np.asarray(compose(diff_c, reac_c, u0, coarse_steps))
    ba = np.asarray(compose(reac_c, diff_c, u0, coarse_steps))
    ab_fine = np.asarray(compose(diff_f, reac_f, u0, fine_steps))
    ba_fine = np.asarray(compose(reac_f, diff_f, u0, fine_steps))

    if not (np.isfinite(ab).all() and np.isfinite(ba).all()):
        raise SystemExit("nonfinite composed arm; configuration is numerically blocked")

    contrast = relative_l2(ab, ba, ab)
    floor = np.maximum(relative_l2(ab, ab_fine, ab), relative_l2(ba, ba_fine, ba))
    partner = cross_ic_partners()
    cross_ic = relative_l2(ab, ab[partner], ab)

    ratio = contrast / cross_ic
    to_floor = contrast / floor
    passes = int((contrast >= THRESHOLD_MULTIPLIER * floor).sum())
    median_ratio = float(np.median(ratio))
    lo, hi = bootstrap_median(ratio, BOOTSTRAP_DRAWS, MASTER_SEED + cutoff)

    return {
        "cutoff": cutoff,
        "units": NUM_UNITS,
        "floor_passes": passes,
        "floor_passes_required": MIN_PASSING_UNITS,
        "floor_gate_clears": passes >= MIN_PASSING_UNITS,
        "median_contrast": float(np.median(contrast)),
        "median_floor": float(np.median(floor)),
        "median_contrast_to_floor": float(np.median(to_floor)),
        "median_contrast_to_cross_ic": median_ratio,
        "bootstrap_ci95": [lo, hi],
        "band": list(BAND),
        "band_clears": BAND[0] <= median_ratio <= BAND[1],
    }


def main() -> None:
    started = time.perf_counter()
    results = [run_cutoff(c) for c in CUTOFFS]
    elapsed = time.perf_counter() - started

    print(f"{'cutoff':>6} {'floor pass':>11} {'med r/c':>9} {'95% CI':>26} "
          f"{'med r/floor':>12}  band")
    for row in results:
        ci = f"[{row['bootstrap_ci95'][0]:.6f}, {row['bootstrap_ci95'][1]:.6f}]"
        print(f"{row['cutoff']:>6} {row['floor_passes']:>7}/{row['units']:<3} "
              f"{row['median_contrast_to_cross_ic']:>9.6f} {ci:>26} "
              f"{row['median_contrast_to_floor']:>12.1f}  "
              f"{'CLEARS' if row['band_clears'] else 'MISSES'}")

    clears = [r["band_clears"] for r in results]
    all_clear, any_clear = all(clears), any(clears)
    print(f"\nbanked comparison (reconstructed, not re-openable): "
          f"256/256 floor passes, median r/c 0.412779 [0.367881, 0.455603], "
          f"median r/floor 3546.30")
    print(f"elapsed {elapsed:.2f} s, cost $0, device CPU")
    print(f"\nALL THREE CUTOFFS CLEAR : {all_clear}")
    if any_clear and not all_clear:
        print("DISAGREEMENT ACROSS CUTOFFS - per the plan this is the finding")

    payload = {
        "schema": "pde-band-check-v1",
        "generated": "2026-09-05",
        "terminal": (
            "BAND_CHECK_CLEARS" if all_clear
            else "BAND_CHECK_CUTOFF_DEPENDENT" if any_clear
            else "BAND_CHECK_MISSES"
        ),
        "configuration": {
            "diffusivity": DIFFUSIVITY, "reaction_diffusivity": REACTION_DIFFUSIVITY,
            "reactivity": REACTIVITY, "domain_extent": DOMAIN_EXTENT,
            "num_points": NUM_POINTS, "dt": DT, "half_dt": HALF_DT,
            "window_duration": WINDOW_DURATION, "units": NUM_UNITS,
            "master_seed": MASTER_SEED, "ic_amplitude": IC_AMPLITUDE,
            "ic_offset_range": list(IC_OFFSET_RANGE),
            "ic_family": "clip(offset_i + amplitude * TruncatedFourier(cutoff, std_one=True), 0, 1)",
        },
        "banked_comparison": {
            "confidence": "reconstructed",
            "floor_passes": "256/256", "median_contrast_to_cross_ic": 0.412779,
            "bootstrap_ci95": [0.367881, 0.455603],
            "median_contrast_to_floor": 3546.30,
            "note": "The banked artifact was destroyed and cannot be inspected.",
        },
        "results": results,
        "elapsed_seconds": elapsed,
        "cost_usd": 0.0,
        "caveat": (
            "The original initial-condition generator is unrecoverable. This is a "
            "calibrated substitute fitted to two non-gate statistics. Clearing does not "
            "prove the original configuration was reproduced; missing would not prove it "
            "was wrong."
        ),
    }
    out = RESULTS / "BAND_CHECK_RESULT.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nterminal: {payload['terminal']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
