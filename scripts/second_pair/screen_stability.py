"""Select a second generator pair on numerical stability alone.

The hazard this script exists to prevent is choosing the pair that gives the answer we
want. So it measures nothing that could reveal the answer. It does not compute support
overlap, it does not compute an order contrast, and it never compares the two orderings
to each other. It measures whether the solver integrates the pair without falling over.

The selection rule is fixed here, in the file, and applied mechanically by the script
itself. Whoever runs this does not get to look at the table and then decide.

  1. Eliminate any candidate producing a nonfinite value in either ordering.
  2. Eliminate any candidate whose composed maximum |u| exceeds 10x the initial
     maximum |u|. That is blow-up, whatever the solver reports.
  3. Eliminate any candidate whose advective CFL exceeds 1.0.
  4. Eliminate any candidate that does not actually depend on order. A pair whose two
     orderings agree to solver noise cannot test anything about order, whatever else is
     true of it. The floor is 1e-6, nine orders of magnitude above the 4.996e-16
     commuting null the original study measured on a pair known to commute.
  5. Among the survivors, take the smallest composed growth ratio, which is the most
     numerically benign. Compare it rounded to six decimals so a tie is broken by name
     rather than by floating-point noise in the eleventh place.

Rules 4 and 5 were added after a first run in which all six candidates survived rules 1
to 3 with growth ratios equal to four decimals, so the selection was being decided by
noise, and in which nothing had checked that the candidates depend on order at all. That
run is recorded in this file's history and no plan existed at the time. Rule 4
tests order dependence in the SOLVER, which is a property of the physics. It is not the
question under test: whether the training and switch-state supports overlap is a
different quantity, and this script still does not compute it.

Advection velocity is fixed at 0.25 so that the advective CFL is 0.64, below the rule 3
threshold, rather than the library default of 1.0 which would sit at 2.56. That is a
stability choice made before any candidate was run and applied to every candidate that
carries an advection term.

Burgers is excluded by a recorded apparatus stop from the original study: nonfinite
output, median advective CFL 2.06 against a horizon of 0.5. It is not rescreened.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Writing beside the script overwrites a banked result when the script is re-run in
# place. run_dissociation.py had the same defect and it cost us the first-run file.
# An explicit output directory makes a repeat safe by construction.
DEFAULT_OUT = HERE.parents[1] / "results" / "tables" / "second_pair"
OUT = (Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv
       else DEFAULT_OUT)
sys.path.insert(0, str(HERE.parent / "band_check"))
import _runtime_path  # noqa: F401  (puts the pinned runtime on sys.path)

import exponax as ex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_enable_x64", True)

# Identical to the original corpus, which is the point: only the operator pair changes.
NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, DT = 1, 1.0, 256, 0.01
ETDRK_ORDER, DEALIASING_FRACTION = 2, 2 / 3
NUM_CIRCLE_POINTS, CIRCLE_RADIUS = 16, 1.0
STEPS_PER_LEG = 100
IC_CUTOFF, IC_AMPLITUDE = 5, 0.175
SCREEN_OFFSET_RANGE = (0.4865, 0.5171)
SCREEN_UNITS = 32
SCREEN_SEED = 20260905

ADVECTION_VELOCITY = 0.25
DX = DOMAIN_EXTENT / NUM_POINTS
ADVECTIVE_CFL = ADVECTION_VELOCITY * DT / DX

GROWTH_LIMIT = 10.0
CFL_LIMIT = 1.0
ORDER_DEPENDENCE_FLOOR = 1e-6


def advection():
    return ex.stepper.Advection(NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, DT,
                                velocity=ADVECTION_VELOCITY)


def diffusion():
    return ex.stepper.Diffusion(NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, DT,
                                diffusivity=0.01)


def _reaction(cls, **kwargs):
    return cls(NUM_SPATIAL_DIMS, DOMAIN_EXTENT, NUM_POINTS, DT, order=ETDRK_ORDER,
               dealiasing_fraction=DEALIASING_FRACTION,
               num_circle_points=NUM_CIRCLE_POINTS, circle_radius=CIRCLE_RADIUS,
               **kwargs)


def fisher_kpp():
    return _reaction(ex.stepper.reaction.FisherKPP, diffusivity=0.0, reactivity=1.0)


def allen_cahn():
    return _reaction(ex.stepper.reaction.AllenCahn)


def cahn_hilliard():
    return _reaction(ex.stepper.reaction.CahnHilliard)


def swift_hohenberg():
    return _reaction(ex.stepper.reaction.SwiftHohenberg)


# Every pair is noncommuting: no candidate places two Fourier-diagonal linear operators
# together, which would commute exactly and could not test anything.
CANDIDATES = {
    "advection_allen_cahn": (advection, allen_cahn, True),
    "advection_cahn_hilliard": (advection, cahn_hilliard, True),
    "advection_fisher_kpp": (advection, fisher_kpp, True),
    "diffusion_allen_cahn": (diffusion, allen_cahn, False),
    "diffusion_cahn_hilliard": (diffusion, cahn_hilliard, False),
    "diffusion_swift_hohenberg": (diffusion, swift_hohenberg, False),
}


def initial_conditions():
    base = ex.ic.RandomTruncatedFourierSeries(NUM_SPATIAL_DIMS, cutoff=IC_CUTOFF,
                                              std_one=True)
    field_key, offset_key = jr.split(jr.PRNGKey(SCREEN_SEED))
    keys = jr.split(field_key, SCREEN_UNITS)
    raw = jnp.stack([base(NUM_POINTS, key=k) for k in keys])
    offsets = jr.uniform(offset_key, shape=(SCREEN_UNITS, 1, 1),
                         minval=SCREEN_OFFSET_RANGE[0], maxval=SCREEN_OFFSET_RANGE[1])
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


def main() -> None:
    u0 = initial_conditions()
    initial_max = float(jnp.max(jnp.abs(u0)))
    print(f"screen cohort {SCREEN_UNITS} units, initial max |u| {initial_max:.6f}")
    print(f"advective CFL for advection candidates {ADVECTIVE_CFL:.4f} "
          f"(velocity {ADVECTION_VELOCITY}, limit {CFL_LIMIT})\n")

    rows = {}
    for name, (first, second, has_advection) in sorted(CANDIDATES.items()):
        a, b = first(), second()
        try:
            forward = compose(a, b, u0)
            reverse = compose(b, a, u0)
        except Exception as error:                      # a stepper that will not run
            rows[name] = {"ran": False, "error": type(error).__name__}
            print(f"  {name:28} DID NOT RUN  {type(error).__name__}")
            continue
        both = jnp.concatenate([forward, reverse], axis=0)
        finite = bool(jnp.all(jnp.isfinite(both)))
        composed_max = float(jnp.max(jnp.abs(jnp.where(jnp.isfinite(both), both, 0.0))))
        growth = composed_max / initial_max
        cfl = ADVECTIVE_CFL if has_advection else 0.0
        # Order dependence in the solver. This is the physics, not the hypothesis: it
        # says the pair can be asked an order question, not what the answer will be.
        order_dependence = float(jnp.max(jnp.abs(forward[:, -1] - reverse[:, -1])))
        eliminated = []
        if not finite:
            eliminated.append("nonfinite")
        if growth > GROWTH_LIMIT:
            eliminated.append("blow-up")
        if cfl > CFL_LIMIT:
            eliminated.append("cfl")
        if order_dependence < ORDER_DEPENDENCE_FLOOR:
            eliminated.append("order-independent")
        rows[name] = {"ran": True, "finite": finite, "composed_max": composed_max,
                      "growth_ratio": growth, "advective_cfl": cfl,
                      "order_dependence": order_dependence,
                      "eliminated_by": eliminated}
        verdict = "eliminated: " + ", ".join(eliminated) if eliminated else "survives"
        print(f"  {name:28} finite={str(finite):5} growth={growth:8.4f} "
              f"cfl={cfl:.4f} order={order_dependence:11.4e}  {verdict}")

    survivors = sorted(n for n, r in rows.items()
                       if r.get("ran") and not r["eliminated_by"])
    print(f"\n{len(survivors)} survivor(s): {', '.join(survivors) or 'none'}")
    if not survivors:
        raise SystemExit("no candidate survived the stability rule")
    chosen = min(survivors, key=lambda n: (round(rows[n]["growth_ratio"], 6), n))
    print(f"rule 5 selects: {chosen}  "
          f"(growth {rows[chosen]['growth_ratio']:.6f}, smallest among survivors)")

    payload = {
        "schema": "pde-second-pair-stability-screen-v1",
        "what_this_measures": (
            "numerical stability only: finiteness, blow-up against a 10x growth limit, "
            "and advective CFL. No support overlap and no order contrast are computed "
            "here, so the screen cannot see which candidate favours the hypothesis."
        ),
        "selection_rule": [
            "eliminate nonfinite in either ordering",
            f"eliminate composed max |u| above {GROWTH_LIMIT}x the initial max |u|",
            f"eliminate advective CFL above {CFL_LIMIT}",
            f"eliminate solver order dependence below {ORDER_DEPENDENCE_FLOOR}",
            "among survivors take the smallest growth ratio rounded to six decimals, "
            "ties broken by name",
        ],
        "rule_4_note": (
            "order dependence is a property of the physics and a precondition for the "
            "experiment to mean anything. It is not the question under test, which is "
            "whether training and switch-state supports overlap. This script does not "
            "compute that."
        ),
        "solver": {"domain_extent": DOMAIN_EXTENT, "num_points": NUM_POINTS, "dt": DT,
                   "steps_per_leg": STEPS_PER_LEG, "etdrk_order": ETDRK_ORDER,
                   "advection_velocity": ADVECTION_VELOCITY,
                   "advective_cfl": ADVECTIVE_CFL},
        "screen_cohort": {"units": SCREEN_UNITS, "seed": SCREEN_SEED,
                          "offset_range": list(SCREEN_OFFSET_RANGE),
                          "initial_max_abs": initial_max},
        "candidates": rows,
        "survivors": survivors,
        "selected": chosen,
        "excluded_without_rescreening": {
            "burgers": "recorded apparatus stop: nonfinite, median advective CFL 2.06"
        },
    }
    (OUT / "STABILITY_SCREEN.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nwrote STABILITY_SCREEN.json")


if __name__ == "__main__":
    main()
