"""POST HOC diagnostic, added after the smoke result was in hand and labelled as such.

It answers one question the smoke's S numbers raise and cannot settle on their own:
the Serrano arm scores S = 1.00 on the horizon window, and S = 1 is exactly what a
predictor that emits the shared mean contrast and nothing unit-specific scores by
construction (family3_baselines/PREDECLARED.md section 1(c)). Two very different
predictors land there:

  (i) one whose CENTERED predicted contrast is near zero -- it really is order-blind
      beyond the shared pattern;
  (ii) one whose centered predicted contrast is large but uncorrelated with truth --
      it emits order structure, just the wrong structure.

S cannot tell them apart. This script reports, for the same arms and the same 48 test
units, the magnitude ratio  ||dhat_centered|| / ||d_centered||  and the correlation
between them. It changes no predeclared metric and writes its own file.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu \
    XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
    env/bin/python diagnostic_contrast_magnitude.py
"""

from __future__ import annotations

import json
import time

import numpy as np

import serrano_splitting as ss


def stats(pred_ab, pred_ba, true_ab, true_ba, window):
    d = (true_ab - true_ba)[:, window]
    dhat = (pred_ab - pred_ba)[:, window]
    d_c = d - d.mean(axis=0, keepdims=True)
    dhat_c = dhat - dhat.mean(axis=0, keepdims=True)
    nd, ndh = np.sqrt((d_c ** 2).mean()), np.sqrt((dhat_c ** 2).mean())
    corr = float((d_c * dhat_c).mean() / (nd * ndh)) if nd > 0 and ndh > 0 else float("nan")
    return {"magnitude_ratio": float(ndh / nd), "correlation": corr,
            "S": float(np.sqrt(((dhat_c - d_c) ** 2).mean()) / nd)}


def main() -> None:
    started = time.perf_counter()
    import jax.numpy as jnp
    diffusion, reaction = ss.tp.build_steppers(ss.tp.DT)
    grid = jnp.linspace(0.0, ss.tp.DOMAIN_EXTENT, ss.tp.NUM_POINTS, endpoint=False)
    s_ic = ss.tp.initial_conditions(ss.SMOKE_UNITS, ss.rb.NARROW, ss.MASTER + ss.SEED_S)
    true_ab = ss.compose_true(diffusion, reaction, s_ic)
    true_ba = ss.compose_true(reaction, diffusion, s_ic)

    windows = {"trajectory": np.arange(ss.FRAMES),
               "endpoint": np.array([ss.FRAMES - 1]),
               "horizon_only": np.arange(ss.CONTEXT_FRAMES, ss.FRAMES)}
    cands = ss.candidates(2)
    out: dict = {"schema": "family6-contrast-magnitude-diagnostic-v1",
                 "status": "POST HOC. Added after the smoke result was in hand. "
                           "Changes no predeclared metric.",
                 "units": ss.SMOKE_UNITS, "seeds": [], "arms": {}}

    for seed in (0, 1):
        models = {(c, p): ss.load_params(ss.CKPT / f"{c}_{p}_seed{seed}.npz")
                  for c in ss.CONDITIONS for p in ss.PRIMITIVES}
        out["seeds"].append(seed)
        for condition in ss.CONDITIONS:
            p_diff = models[(condition, "diffusion")]
            p_reac = models[(condition, "reaction")]
            arms = {
                f"{condition}_oracle": (
                    ss.compose_predicted(p_diff, p_reac, s_ic, grid),
                    ss.compose_predicted(p_reac, p_diff, s_ic, grid)),
            }
            ops = [ss._fno_op("diff", p_diff, grid), ss._fno_op("reac", p_reac, grid)]
            ctx_ab = jnp.asarray(true_ab[:, :ss.CONTEXT_FRAMES, None, :])
            ctx_ba = jnp.asarray(true_ba[:, :ss.CONTEXT_FRAMES, None, :])
            sel_ab = ss.select_exhaustive(
                ss.loss_table(ops, cands, ctx_ab, ss.lie_apply), cands)
            sel_ba = ss.select_exhaustive(
                ss.loss_table(ops, cands, ctx_ba, ss.lie_apply), cands)
            arms[f"serrano_{condition}_lie_causal"] = (
                ss.rollout_selected(ops, sel_ab, s_ic, ss.lie_apply,
                                    2 * ss.APPLICATIONS_PER_LEG),
                ss.rollout_selected(ops, sel_ba, s_ic, ss.lie_apply,
                                    2 * ss.APPLICATIONS_PER_LEG))
            for name, (pab, pba) in arms.items():
                row = out["arms"].setdefault(name, {})
                for wname, w in windows.items():
                    row.setdefault(wname, {})[f"seed{seed}"] = stats(
                        pab, pba, true_ab, true_ba, w)

    out["elapsed_seconds"] = time.perf_counter() - started
    path = ss.RESULTS / "CONTRAST_MAGNITUDE_DIAGNOSTIC.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    for name, row in out["arms"].items():
        for wname, per in row.items():
            for sk, v in per.items():
                print(f"{name:32} {wname:14} {sk}  |dhat|/|d| {v['magnitude_ratio']:8.4f}"
                      f"  corr {v['correlation']:+.4f}  S {v['S']:8.4f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
