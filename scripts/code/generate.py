"""Generate the support-broadened corpus for the regenerated learner chain.

This is a NEW experiment. It does not amend, annotate or replace the banked
SUPPORT_RECOVERY_PARTIAL_ONLY result, which stays exactly as it is.

Design follows the banked support-recovery run as recorded in the reconstructed
state, section 3.12:

  - primitive training support broadened to spatial means [0.48, 0.75]
  - 480 new units plus 160 original primitive-only training units per lane
  - 64 new primitive-only validation units
  - composed trajectories are evaluator-only and never enter training or checkpoint
    selection; the final evaluation uses a fresh composed cohort

Solver constants are recovered verbatim from S1 step 498 and neighbours. The
initial-condition generator is NOT recovered; it is the fitted substitute the band-check
plan pinned in its section 3, and every downstream number inherits that caveat.

Information hygiene, owner-set and non-negotiable: the learner sees raw float32 state
trajectories and nothing else. No symbol, token, equation string, coefficient, schedule,
primitive name, or ordering label is written into any array the learner reads. Lane
identity survives only in the evaluator index, which the trainer must not open.
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

# Solver, recovered.
NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, DT = 1, 1.0, 256, 0.01
DIFFUSIVITY, REACTION_DIFFUSIVITY, REACTIVITY = 0.01, 0.0, 1.0
ETDRK_ORDER, DEALIASING_FRACTION = 2, 2 / 3
NUM_CIRCLE_POINTS, CIRCLE_RADIUS = 16, 1.0
STEPS_PER_LEG = 100
FRAMES_PER_LEG = STEPS_PER_LEG + 1

# Initial conditions, fitted substitute.
IC_CUTOFF, IC_AMPLITUDE = 5, 0.175
EVAL_OFFSET_RANGE = (0.4865, 0.5171)      # the recorded diffusion support
TRAIN_OFFSET_RANGE = (0.48, 0.75)          # the banked broadened support

# Cohorts.
TRAIN_UNITS, VALIDATION_UNITS = 640, 64
EVAL_TRAIN_UNITS, EVAL_TEST_UNITS = 160, 48

MASTER_SEED = 20260905
OUT = Path(__file__).resolve().parents[2] / "data" / "corpus"
RECEIPT = (Path(__file__).resolve().parents[2] / "results" / "tables"
           / "regeneration" / "CORPUS_RECEIPT.json")


def diffusion():
    return ex.stepper.Diffusion(
        NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, DT, diffusivity=DIFFUSIVITY
    )


def fisher_kpp():
    return ex.stepper.reaction.FisherKPP(
        NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, DT,
        diffusivity=REACTION_DIFFUSIVITY, reactivity=REACTIVITY, order=ETDRK_ORDER,
        dealiasing_fraction=DEALIASING_FRACTION, num_circle_points=NUM_CIRCLE_POINTS,
        circle_radius=CIRCLE_RADIUS,
    )


def initial_conditions(units: int, offset_range: tuple[float, float], seed: int):
    base = ex.ic.RandomTruncatedFourierSeries(
        NUM_SPATIAL_DIMS, cutoff=IC_CUTOFF, std_one=True
    )
    field_key, offset_key = jr.split(jr.PRNGKey(seed))
    keys = jr.split(field_key, units)
    raw = jnp.stack([base(NUM_POINTS, key=k) for k in keys])
    offsets = jr.uniform(offset_key, shape=(units, 1, 1),
                         minval=offset_range[0], maxval=offset_range[1])
    return jnp.clip(offsets + IC_AMPLITUDE * raw, 0.0, 1.0)


def trace(stepper, batch, steps: int):
    """Roll a batch forward, returning every frame including the initial one."""
    step = jax.jit(jax.vmap(stepper))
    state = batch
    frames = [state]
    for _ in range(steps):
        state = step(state)
        frames.append(state)
    return jnp.stack(frames, axis=1)


def compose(first, second, u0):
    """Two legs with exact state handoff; the switch frame appears once."""
    one = trace(first, u0, STEPS_PER_LEG)
    two = trace(second, one[:, -1], STEPS_PER_LEG)
    return jnp.concatenate([one, two[:, 1:]], axis=1)


def as_learner_array(x) -> np.ndarray:
    """(units, frames, 1, points) float64 -> (units, frames, points) float32."""
    return np.asarray(x, dtype=np.float64).squeeze(2).astype(np.float32)


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    diff, reac = diffusion(), fisher_kpp()

    # Primitive-only training and validation, on the broadened support.
    train_ic = initial_conditions(TRAIN_UNITS, TRAIN_OFFSET_RANGE, MASTER_SEED + 1)
    val_ic = initial_conditions(VALIDATION_UNITS, TRAIN_OFFSET_RANGE, MASTER_SEED + 2)
    # Evaluator cohorts, on the recorded evaluation support. Disjoint seeds throughout.
    eval_train_ic = initial_conditions(EVAL_TRAIN_UNITS, EVAL_OFFSET_RANGE, MASTER_SEED + 3)
    eval_test_ic = initial_conditions(EVAL_TEST_UNITS, EVAL_OFFSET_RANGE, MASTER_SEED + 4)

    arrays: dict[str, np.ndarray] = {}
    # Lane 0 is diffusion, lane 1 is reaction. The mapping lives here and in the
    # evaluator index only; the learner receives neutral keys.
    arrays["a0"] = as_learner_array(trace(diff, train_ic, STEPS_PER_LEG))
    arrays["a1"] = as_learner_array(trace(reac, train_ic, STEPS_PER_LEG))
    arrays["a2"] = as_learner_array(trace(diff, val_ic, STEPS_PER_LEG))
    arrays["a5"] = as_learner_array(trace(reac, val_ic, STEPS_PER_LEG))
    # Evaluator-only. Composed arms never enter training or checkpoint selection.
    arrays["e_train_ab"] = as_learner_array(compose(diff, reac, eval_train_ic))
    arrays["e_train_ba"] = as_learner_array(compose(reac, diff, eval_train_ic))
    arrays["e_test_ab"] = as_learner_array(compose(diff, reac, eval_test_ic))
    arrays["e_test_ba"] = as_learner_array(compose(reac, diff, eval_test_ic))
    arrays["e_test_p0"] = as_learner_array(trace(diff, eval_test_ic, STEPS_PER_LEG))
    arrays["e_test_p1"] = as_learner_array(trace(reac, eval_test_ic, STEPS_PER_LEG))

    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise SystemExit(f"nonfinite values in {name}")

    learner_keys = ("a0", "a1", "a2", "a5")
    learner_path = OUT / "LEARNER_BUNDLE.npz"
    np.savez(learner_path, **{k: arrays[k] for k in learner_keys})
    evaluator_path = OUT / "EVALUATOR_BUNDLE.npz"
    np.savez(evaluator_path, **{k: v for k, v in arrays.items() if k not in learner_keys})

    # Hygiene: nothing the learner reads may name a primitive or an ordering.
    #
    # The scan covers STRUCTURAL surfaces only: archive member names and NPY headers.
    # It deliberately does not sweep the float payload. A first attempt did, and
    # flagged "kpp", "order" and "tau" inside several hundred megabytes of random
    # float32 bytes. That is not leakage, it is coincidence, and the original project
    # hit the identical failure: the reconstructed state, section 4, records
    # "First sanitizer stop" as FLAWED for exactly this reason, with the fix being to
    # restrict the scan to "learner-visible structural and metadata surfaces". The
    # recovered sanitizer corroborates it, carrying a
    # "short_payload_acronyms_excluded" list of ["pde", "kpp"].
    #
    # Semantics cannot hide in IEEE-754 samples of a smooth field. They can only hide
    # in names, so names are what is checked.
    import zipfile

    forbidden = (
        "diffus", "fisher", "kpp", "reaction", "primitive", "order", "compose",
        "commutator", "split", "lane", "token", "equation", "generator", "tau",
    )
    surfaces: list[str] = []
    with zipfile.ZipFile(learner_path) as archive:
        for info in archive.infolist():
            surfaces.append(info.filename)
            with archive.open(info) as member:
                # NPY header: magic, version, then a header length and an ASCII dict.
                surfaces.append(member.read(128).decode("latin-1"))
    # Word-boundary matching, as the recovered sanitizer did via its
    # "surface_word_boundary_regex". A substring test flags numpy's own
    # `fortran_order` header field, which is structure, not semantics.
    import re as _re

    haystack = " ".join(surfaces).lower()
    hits = sorted({
        word for word in forbidden
        if _re.search(rf"\b{word}", haystack)
    })
    if hits:
        raise SystemExit(f"semantic leakage on a learner-visible surface: {hits}")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    elapsed = time.perf_counter() - started
    receipt = {
        "schema": "pde-regenerated-corpus-v1",
        "generated": "2026-09-05",
        "status": "NEW EXPERIMENT. Does not amend the banked corpus or its results.",
        "solver": {
            "exponax": "0.2.0 (11/11 module digests verified)",
            "diffusivity": DIFFUSIVITY, "reaction_diffusivity": REACTION_DIFFUSIVITY,
            "reactivity": REACTIVITY, "domain_extent": DOMAIN_EXTENT,
            "num_points": NUM_POINTS, "dt": DT, "steps_per_leg": STEPS_PER_LEG,
            "etdrk_order": ETDRK_ORDER, "dealiasing_fraction": DEALIASING_FRACTION,
            "num_circle_points": NUM_CIRCLE_POINTS, "circle_radius": CIRCLE_RADIUS,
            "provenance": "recovered verbatim from S1 step 498 and neighbours",
        },
        "initial_conditions": {
            "family": "clip(offset_i + amplitude * TruncatedFourier(cutoff, std_one), 0, 1)",
            "cutoff": IC_CUTOFF, "amplitude": IC_AMPLITUDE,
            "train_offset_range": list(TRAIN_OFFSET_RANGE),
            "eval_offset_range": list(EVAL_OFFSET_RANGE),
            "provenance": "NOT RECOVERED. Fitted substitute, see the band-check plan, section 3.",
        },
        "cohorts": {
            "primitive_train_units": TRAIN_UNITS,
            "primitive_validation_units": VALIDATION_UNITS,
            "evaluator_train_units": EVAL_TRAIN_UNITS,
            "evaluator_test_units": EVAL_TEST_UNITS,
            "seed_offsets": {"train": 1, "validation": 2, "eval_train": 3, "eval_test": 4},
            "master_seed": MASTER_SEED,
        },
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "learner_visible_keys": list(learner_keys),
        "composed_in_learner_bundle": False,
        "semantic_leakage_hits": len(hits),
        "files": {
            "LEARNER_BUNDLE.npz": {"sha256": digest(learner_path),
                                   "bytes": learner_path.stat().st_size},
            "EVALUATOR_BUNDLE.npz": {"sha256": digest(evaluator_path),
                                     "bytes": evaluator_path.stat().st_size},
        },
        "elapsed_seconds": elapsed,
        "cost_usd": 0.0,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")

    print(f"{'array':>12}  shape")
    for name, array in arrays.items():
        print(f"{name:>12}  {array.shape}")
    print(f"\nlearner bundle   {learner_path.stat().st_size:>12,} B  "
          f"{receipt['files']['LEARNER_BUNDLE.npz']['sha256'][:16]}...")
    print(f"evaluator bundle {evaluator_path.stat().st_size:>12,} B  "
          f"{receipt['files']['EVALUATOR_BUNDLE.npz']['sha256'][:16]}...")
    print(f"semantic leakage hits: {len(hits)}")
    print(f"elapsed {elapsed:.1f} s, cost $0")


if __name__ == "__main__":
    main()
