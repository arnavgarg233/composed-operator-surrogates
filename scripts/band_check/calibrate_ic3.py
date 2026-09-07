"""Third initial-condition family: per-unit random mean offset plus a small fluctuation.

Calibration v2 established that a fixed-centre family CANNOT reproduce both recorded
non-gate statistics at once. Matching the diffusion-mean SPREAD of 0.0306 needs
amplitude near 0.30, which pushes the reaction means down to about [0.648, 0.672],
well below the recorded [0.6973, 0.7258]. Matching the reaction means needs amplitude
near 0.10-0.15, which collapses the diffusion-mean spread to zero. That mismatch is
itself evidence: the original generator must have randomised the spatial mean ACROSS
units, which a fixed centre cannot do.

So: u0 = clip(offset_i + amplitude * z_i, 0, 1), offset_i ~ U(lo, hi), z_i a
cutoff-truncated Fourier field of unit standard deviation.

Consistency check before running. Diffusion preserves the mean, so diffusion means
should land on [lo, hi] directly. For small amplitude the logistic image of
[0.4865, 0.5171] is [0.7203, 0.7443]; the recorded reaction interval [0.6973, 0.7258]
sits about 0.02 below that, which is the Jensen deficit a moderate amplitude produces.
Two recorded intervals, two free parameters, and the offsets are pinned by the first,
so the amplitude is over-determined by the second. If it fits, the inference holds.

Still no contrast, no floor, no ratio. The band remains unexamined.
"""

from __future__ import annotations

import json
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
ETDRK_ORDER, DEALIASING_FRACTION = 2, 2 / 3
NUM_CIRCLE_POINTS, CIRCLE_RADIUS = 16, 1.0
STEPS_PER_LEG = 100

SEED, UNITS = 20260905, 96
TARGET_DIFFUSION = (0.4865, 0.5171)
TARGET_REACTION = (0.6973, 0.7258)


def steppers():
    return (
        ex.stepper.Diffusion(NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, DT,
                             diffusivity=DIFFUSIVITY),
        ex.stepper.reaction.FisherKPP(
            NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, DT,
            diffusivity=REACTION_DIFFUSIVITY, reactivity=REACTIVITY,
            order=ETDRK_ORDER, dealiasing_fraction=DEALIASING_FRACTION,
            num_circle_points=NUM_CIRCLE_POINTS, circle_radius=CIRCLE_RADIUS),
    )


def make_fields(cutoff, lo, hi, amplitude, units, seed):
    base = ex.ic.RandomTruncatedFourierSeries(NUM_SPATIAL_DIMS, cutoff=cutoff, std_one=True)
    field_key, offset_key = jr.split(jr.PRNGKey(seed))
    keys = jr.split(field_key, units)
    raw = jnp.stack([base(NUM_POINTS, key=k) for k in keys])
    offsets = jr.uniform(offset_key, shape=(units, 1, 1), minval=lo, maxval=hi)
    return jnp.clip(offsets + amplitude * raw, 0.0, 1.0)


def advance(stepper, batch, steps):
    state = batch
    for _ in range(steps):
        state = jax.vmap(stepper)(state)
    return state


def main() -> None:
    diffusion, reaction = steppers()
    lo_d, hi_d = TARGET_DIFFUSION
    lo_r, hi_r = TARGET_REACTION

    rows = []
    for cutoff in (3, 5, 7):
        for amplitude in (0.10, 0.125, 0.15, 0.175, 0.20):
            u0 = make_fields(cutoff, lo_d, hi_d, amplitude, UNITS, SEED)
            d = np.asarray(advance(diffusion, u0, STEPS_PER_LEG)).mean(axis=(1, 2))
            r = np.asarray(advance(reaction, u0, STEPS_PER_LEG)).mean(axis=(1, 2))
            # Interval agreement: penalise both overshoot and undershoot of each end.
            miss = (abs(d.min() - lo_d) + abs(d.max() - hi_d)
                    + abs(r.min() - lo_r) + abs(r.max() - hi_r))
            rows.append({
                "cutoff": cutoff, "amplitude": amplitude,
                "diffusion_mean_range": [float(d.min()), float(d.max())],
                "reaction_mean_range": [float(r.min()), float(r.max())],
                "interval_disagreement": float(miss),
            })
            print(f"cutoff={cutoff} amp={amplitude:<6} "
                  f"diff [{d.min():.4f},{d.max():.4f}] "
                  f"reac [{r.min():.4f},{r.max():.4f}]  disagreement {miss:.4f}")

    best = min(rows, key=lambda r: r["interval_disagreement"])
    print(f"\nrecorded diffusion {TARGET_DIFFUSION}   reaction {TARGET_REACTION}")
    print(f"best: cutoff={best['cutoff']} amplitude={best['amplitude']} "
          f"disagreement={best['interval_disagreement']:.4f}")
    print(f"      diffusion {[round(v,4) for v in best['diffusion_mean_range']]}")
    print(f"      reaction  {[round(v,4) for v in best['reaction_mean_range']]}")

    payload = {
        "schema": "pde-ic-calibration-v3",
        "generated": "2026-09-05",
        "family": "u0 = clip(offset_i + amplitude * TruncatedFourier(cutoff, std_one), 0, 1), offset_i ~ U(lo, hi)",
        "offset_range": list(TARGET_DIFFUSION),
        "fitted_against": [
            "diffusion training-state spatial means [0.4865, 0.5171] (NON-GATE)",
            "reaction-first switch-state spatial means [0.6973, 0.7258] (NON-GATE)",
        ],
        "not_fitted_against": "the composition-scale band, the AB/BA contrast, the numerical floor, the cross-IC reference",
        "v2_finding": "a fixed-centre family cannot satisfy both recorded intervals; the mean must vary across units",
        "sweep": rows,
        "best": best,
    }
    out = (Path(__file__).resolve().parents[2] / "results" / "tables"
           / "band_check" / "IC_CALIBRATION_V3.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
