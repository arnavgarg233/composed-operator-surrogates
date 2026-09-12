"""Family 5 — initial-condition-family robustness of the composition-scale band check.

Tests whether the band-check result (median contrast-to-cross-IC ratio inside
[0.3, 0.7], 256 units, numerical floor >= 142/256) depends on the fitted-substitute
IC family used in the composed-operator-surrogates package
(recovery/composed-operator-surrogates, commit 0f04368).

This script does NOT edit or import from the package tree. It VENDORS the exact
physical/numerical configuration and the contrast/floor/cross-IC/commuting-null/
saturation definitions from the package, with attribution at each vendored block,
and adds two new, predeclared IC families (see PREDECLARED.md, written before this
script was ever run).

Vendored from scripts/band_check/band_check.py (attributed inline):
    NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, DT, HALF_DT, DIFFUSIVITY,
    REACTION_DIFFUSIVITY, REACTIVITY, ETDRK_ORDER, DEALIASING_FRACTION,
    NUM_CIRCLE_POINTS, CIRCLE_RADIUS, WINDOW_DURATION, NUM_UNITS, MASTER_SEED,
    IC_AMPLITUDE, IC_OFFSET_RANGE, THRESHOLD_MULTIPLIER, MIN_PASSING_UNITS, BAND,
    build(), initial_conditions() [renamed original_initial_conditions()],
    leg(), compose(), relative_l2(), cross_ic_partners(), run_cutoff() structure.

Vendored from scripts/band_check/integrity_controls.py (attributed inline):
    the commuting-pair null control (diffusivity 0.01 vs 0.02, gate 1e-12).

Vendored from scripts/second_pair/measure_geometric_repair.py (attributed inline):
    the saturation-fraction definition (fraction of values at a clip bound,
    limit 0.10).

Runs with the environment built at .venv
(see env/ENV_RECEIPT.md), threads pinned via
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
JAX_PLATFORMS=cpu (set by the caller before invoking this script).
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import exponax as ex

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Vendored configuration — scripts/band_check/band_check.py, verbatim values.
# ---------------------------------------------------------------------------
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
ORIGINAL_CUTOFF = 5  # calibrate_ic3.py's chosen cutoff; band_check.py's middle cutoff

THRESHOLD_MULTIPLIER = 10.0
MIN_PASSING_UNITS = 142
BAND = (0.3, 0.7)

TARGET_ORIGINAL = 0.4261565675911184  # results/tables/band_check/BAND_CHECK_RESULT.json, cutoff 5

# Vendored — scripts/band_check/integrity_controls.py
COMMUTING_PARTNER_DIFFUSIVITY = 0.02
COMMUTING_GATE = 1e-12
COMMUTING_STEPS = 100  # integrity_controls.py's STEPS_PER_LEG

# Vendored — scripts/second_pair/measure_geometric_repair.py
SATURATION_LIMIT = 0.10
UPPER_BOUND, LOWER_BOUND = 1.0, 0.0

# This task's bootstrap spec (TASK.md), used for every family's reported CI.
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20260905

# Predeclared alternative-family constants (PREDECLARED.md)
SPECTRAL_POWER_LAW_EXPONENT = 2.0
BUMP_COUNT = 4
BUMP_SIGMA = 0.06


# ---------------------------------------------------------------------------
# Vendored — scripts/band_check/band_check.py: build(), leg(), compose(),
# relative_l2(), cross_ic_partners(), bootstrap_median() (signature adapted to
# this task's predeclared draws/seed).
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Shared PRNG split, identical to band_check.py's initial_conditions(): the
# offsets are therefore numerically identical across all three families below.
# ---------------------------------------------------------------------------
def _field_and_offset_keys():
    return jr.split(jr.PRNGKey(MASTER_SEED))


def _offsets(offset_key):
    return jr.uniform(
        offset_key, shape=(NUM_UNITS, 1, 1),
        minval=IC_OFFSET_RANGE[0], maxval=IC_OFFSET_RANGE[1],
    )


# ---------------------------------------------------------------------------
# Family 0 (positive control) — vendored verbatim from band_check.py's
# initial_conditions(cutoff), specialized to cutoff=5 (calibrate_ic3.py's pick).
# ---------------------------------------------------------------------------
def original_initial_conditions() -> jnp.ndarray:
    base = ex.ic.RandomTruncatedFourierSeries(
        NUM_SPATIAL_DIMS, cutoff=ORIGINAL_CUTOFF, std_one=True
    )
    field_key, offset_key = _field_and_offset_keys()
    keys = jr.split(field_key, NUM_UNITS)
    raw = jnp.stack([base(NUM_POINTS, key=k) for k in keys])
    offsets = _offsets(offset_key)
    return jnp.clip(offsets + IC_AMPLITUDE * raw, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Family 1 (predeclared) — spectral: Gaussian random field, power-law spectrum.
# See PREDECLARED.md "Alternative family 1".
# ---------------------------------------------------------------------------
def spectral_grf_initial_conditions() -> jnp.ndarray:
    field_key, offset_key = _field_and_offset_keys()
    unit_keys = jr.split(field_key, NUM_UNITS)
    n_bins = NUM_POINTS // 2 + 1  # 129 for N=256
    k = jnp.arange(1, n_bins - 1)  # 1..127, excludes DC (0) and Nyquist (128)
    amp = k.astype(jnp.float64) ** (-SPECTRAL_POWER_LAW_EXPONENT / 2.0)

    def one_unit(unit_key):
        real_key, imag_key = jr.split(unit_key, 2)
        a_k = jr.normal(real_key, shape=(n_bins - 2,), dtype=jnp.float64)
        b_k = jr.normal(imag_key, shape=(n_bins - 2,), dtype=jnp.float64)
        c_k = amp * (a_k + 1j * b_k)
        spectrum = jnp.zeros((n_bins,), dtype=jnp.complex128)
        spectrum = spectrum.at[1:-1].set(c_k)
        z_raw = jnp.fft.irfft(spectrum, n=NUM_POINTS)
        return z_raw / jnp.std(z_raw)

    raw = jax.vmap(one_unit)(unit_keys)
    offsets = _offsets(offset_key)
    field = raw[:, None, :]  # match (units, 1, points) layout used elsewhere
    return jnp.clip(offsets + IC_AMPLITUDE * field, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Family 2 (predeclared) — bump/localized: sum of random Gaussian bumps.
# See PREDECLARED.md "Alternative family 2".
# ---------------------------------------------------------------------------
def bump_initial_conditions() -> jnp.ndarray:
    field_key, offset_key = _field_and_offset_keys()
    unit_keys = jr.split(field_key, NUM_UNITS)
    x = jnp.arange(NUM_POINTS, dtype=jnp.float64) / NUM_POINTS

    def one_unit(unit_key):
        centers_key, signs_key = jr.split(unit_key, 2)
        centers = jr.uniform(centers_key, shape=(BUMP_COUNT,), minval=0.0, maxval=1.0,
                              dtype=jnp.float64)
        signs = jnp.where(
            jr.bernoulli(signs_key, p=0.5, shape=(BUMP_COUNT,)), 1.0, -1.0
        )
        d = x[None, :] - centers[:, None]
        d = d - jnp.round(d)  # periodic wrap to [-0.5, 0.5)
        bumps = signs[:, None] * jnp.exp(-(d ** 2) / (2.0 * BUMP_SIGMA ** 2))
        z_raw = bumps.sum(axis=0)
        return z_raw / jnp.std(z_raw)

    raw = jax.vmap(one_unit)(unit_keys)
    offsets = _offsets(offset_key)
    field = raw[:, None, :]
    return jnp.clip(offsets + IC_AMPLITUDE * field, 0.0, 1.0)


FAMILIES = {
    "original_cutoff5_positive_control": original_initial_conditions,
    "family1_spectral_grf_power_law": spectral_grf_initial_conditions,
    "family2_bump_localized": bump_initial_conditions,
}


# ---------------------------------------------------------------------------
# Vendored — scripts/second_pair/measure_geometric_repair.py: saturation
# fraction of IC values sitting at a clip bound.
# ---------------------------------------------------------------------------
def saturation_fraction(u0: np.ndarray) -> float:
    at_upper = u0 >= (UPPER_BOUND - 1e-12)
    at_lower = u0 <= (LOWER_BOUND + 1e-12)
    return float((at_upper | at_lower).mean())


# ---------------------------------------------------------------------------
# Vendored — scripts/band_check/integrity_controls.py: commuting-pair null.
# ---------------------------------------------------------------------------
def commuting_null(u0: jnp.ndarray) -> dict:
    slow = ex.stepper.Diffusion(NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, DT,
                                 diffusivity=DIFFUSIVITY)
    fast = ex.stepper.Diffusion(NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, DT,
                                 diffusivity=COMMUTING_PARTNER_DIFFUSIVITY)
    ab = np.asarray(compose(slow, fast, u0, COMMUTING_STEPS))
    ba = np.asarray(compose(fast, slow, u0, COMMUTING_STEPS))
    denom = np.sqrt((ab ** 2).sum(axis=(1, 2)))
    max_rel_diff = float((np.sqrt(((ab - ba) ** 2).sum(axis=(1, 2))) / denom).max())
    return {
        "max_relative_order_difference": max_rel_diff,
        "gate": COMMUTING_GATE,
        "passes": max_rel_diff < COMMUTING_GATE,
    }


# ---------------------------------------------------------------------------
# Per-family band-check run — mirrors band_check.py's run_cutoff().
# ---------------------------------------------------------------------------
def run_family(name: str, ic_fn) -> dict:
    t0 = time.perf_counter()
    u0 = ic_fn()

    diff_c, reac_c = build(DT)
    diff_f, reac_f = build(HALF_DT)
    coarse_steps = round(WINDOW_DURATION / DT)
    fine_steps = round(WINDOW_DURATION / HALF_DT)
    assert fine_steps == 2 * coarse_steps

    ab = np.asarray(compose(diff_c, reac_c, u0, coarse_steps))
    ba = np.asarray(compose(reac_c, diff_c, u0, coarse_steps))
    ab_fine = np.asarray(compose(diff_f, reac_f, u0, fine_steps))
    ba_fine = np.asarray(compose(reac_f, diff_f, u0, fine_steps))

    if not (np.isfinite(ab).all() and np.isfinite(ba).all()):
        raise SystemExit(f"[{name}] nonfinite composed arm; configuration is numerically blocked")

    contrast = relative_l2(ab, ba, ab)
    floor = np.maximum(relative_l2(ab, ab_fine, ab), relative_l2(ba, ba_fine, ba))
    partner = cross_ic_partners()
    cross_ic = relative_l2(ab, ab[partner], ab)

    ratio = contrast / cross_ic
    to_floor = contrast / floor
    passes = int((contrast >= THRESHOLD_MULTIPLIER * floor).sum())
    median_ratio = float(np.median(ratio))
    lo, hi = bootstrap_median(ratio, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED)

    band_clears = BAND[0] <= median_ratio <= BAND[1]
    floor_clears = passes >= MIN_PASSING_UNITS
    verdict = "PASS" if (band_clears and floor_clears) else "FAIL"

    commuting = commuting_null(u0)
    sat = saturation_fraction(np.asarray(u0))

    elapsed = time.perf_counter() - t0

    return {
        "family": name,
        "units": NUM_UNITS,
        "floor_passes": passes,
        "floor_passes_required": MIN_PASSING_UNITS,
        "floor_gate_clears": floor_clears,
        "median_contrast": float(np.median(contrast)),
        "median_floor": float(np.median(floor)),
        "median_contrast_to_floor": float(np.median(to_floor)),
        "median_contrast_to_cross_ic": median_ratio,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_ci95": [lo, hi],
        "band": list(BAND),
        "band_clears": band_clears,
        "commuting_null": commuting,
        "saturation_fraction": sat,
        "saturation_limit": SATURATION_LIMIT,
        "saturation_within_limit": sat <= SATURATION_LIMIT,
        "verdict": verdict,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    started = time.perf_counter()
    env_info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "exponax_version": ex.__version__ if hasattr(ex, "__version__") else "unknown",
    }
    print("environment:", json.dumps(env_info, indent=2))
    print()

    results = {}
    for name, ic_fn in FAMILIES.items():
        print(f"=== running family: {name} ===")
        row = run_family(name, ic_fn)
        results[name] = row
        print(json.dumps(row, indent=2))
        print()

    elapsed = time.perf_counter() - started

    orig = results["original_cutoff5_positive_control"]
    orig_value = orig["median_contrast_to_cross_ic"]
    orig_matches = abs(orig_value - TARGET_ORIGINAL) < 1e-9
    print(f"positive control reproduces {TARGET_ORIGINAL}: observed {orig_value!r}, "
          f"exact match: {orig_matches} (abs diff {abs(orig_value - TARGET_ORIGINAL):.3e})")

    fam1 = results["family1_spectral_grf_power_law"]
    fam2 = results["family2_bump_localized"]
    robust = fam1["verdict"] == "PASS" and fam2["verdict"] == "PASS"
    print(f"\nfamily1 verdict: {fam1['verdict']}   family2 verdict: {fam2['verdict']}")
    print(f"ROBUST TO IC FAMILY (both alternatives PASS): {robust}")
    print(f"\ntotal elapsed {elapsed:.2f} s")

    payload = {
        "schema": "pde-family5-ic-robustness-v1",
        "generated": "2026-09-10",
        "purpose": (
            "Test whether the composition-scale band check's median "
            "contrast-to-cross-IC ratio and numerical-floor pass count depend on "
            "the fitted-substitute IC family, per CLAIM_INVENTORY.md section 7 row 5."
        ),
        "package_reference": {
            "path": "recovery/composed-operator-surrogates",
            "commit": "0f04368",
            "vendored_from": [
                "scripts/band_check/band_check.py",
                "scripts/band_check/integrity_controls.py",
                "scripts/second_pair/measure_geometric_repair.py",
            ],
        },
        "environment": env_info,
        "predeclared_gate": (
            "PASS for a family if median_contrast_to_cross_ic in [0.3, 0.7] AND "
            "floor_passes >= 142/256; robust-to-IC-family only if BOTH alternative "
            "families individually PASS. See PREDECLARED.md, written before this "
            "script was run."
        ),
        "positive_control": {
            "target_value": TARGET_ORIGINAL,
            "observed_value": orig_value,
            "exact_match": orig_matches,
            "abs_diff": abs(orig_value - TARGET_ORIGINAL),
        },
        "results": results,
        "robust_to_ic_family": robust,
        "elapsed_seconds": elapsed,
        "cost_usd": 0.0,
    }
    out = HERE / "RESULT.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
