# RESULT — Family 3: order-blind and zero-contrast baselines, Route E

Gates, definitions and tolerances were fixed in `PREDECLARED.md` **before** any number was
computed. Nothing below altered a threshold. Inputs under `recovery/` were opened read-only
and none was modified.

- Package: `dependency/composed-operator-surrogates`
- Script: `analyze_family3.py` (python 3.10.14, numpy 2.2.6, numpy only)
- Machine-readable output: `RESULT.json`

---

## The honest number first

**At the endpoint the learner does not beat a predictor that emits one field for both
orderings.** Ensemble endpoint `S = 3.431`; the order-blind / zero-contrast baseline is
`S = 3.403`; the paired margin is **+0.028, 95 % CI [−0.104, +0.172]** — straddling zero.
Against the common-pattern and recognition-only baselines the endpoint margin is
**+2.431, CI [+1.869, +3.048]**, a clear and large *failure*. The package's own
`S ≈ 3.43` endpoint failure holds, and the two new baselines sharpen it: at the endpoint
this learner is statistically indistinguishable from order-blindness.

**On the trajectory the advantage survives all four baselines.** Ensemble trajectory
`S = 0.396` against order-blind `S = 1.653`: margin **−1.256, CI [−1.286, −1.227]**.

**GATE VERDICT: PASS** (trajectory, primary record `ensemble`, all four fair baselines).

---

## Was truth recoverable?

**No — the truth *fields* are not recoverable, and they were not needed.**

`data/corpus/EVALUATOR_BUNDLE.npz` is not shipped (`data/manifest.json → not_shipped`;
CLAIM_INVENTORY §1, §2 row 14). The fields `y_i^AB`, `y_i^BA`, `delta_i`, `d_i` and the
training-mean contrast `m` are unavailable at spatial resolution and cannot be solved for:
the shipped files give 5 per-frame scalars per unit against 256 spatial unknowns.

What the shipped results *do* contain, and what makes this analysis possible:

| Truth quantity | Shipped as | Shape |
|---|---|---|
| `‖d_i‖` per unit/frame (centered contrast RMS) | `PATH_VS_EXPOSURE_RESULT.json → truth_scales.centered_rms_per_unit_frame` (= `S_OF_T_RESULT.json → units.b`, agreeing to 0.0) | (48, 201) |
| `‖delta_i‖` per unit/frame (raw contrast RMS) | `PATH_VS_EXPOSURE_RESULT.json → truth_scales.raw_contrast_rms_per_unit_frame` | (48, 201) |
| `‖y_i^AB‖` per unit/frame | `S_OF_T_RESULT.json → units.Y` | (48, 201) |
| learner residual norm `‖dhat − d‖` per unit/frame | `S_OF_T_RESULT.json → <record>.units.q` | (48, 201) |
| `m` as a field | **not shipped** | — |
| learner weights `W` | `weights_seed{0..4}.pt` shipped, never loaded (not needed) | — |

**Why the baselines are nevertheless exactly computable.** Both new baselines emit a raw
order contrast of exactly zero, so their centered prediction is `−m` and their residual
against truth is `−m − d_i = −delta_i`. Hence `q2_baseline = delta2`, whose square root is
shipped. No truth field enters. This is an identity, not an approximation.

**What was NOT computed, and was not substituted for:**
- arm-level (per-ordering) relative-L2 / RMSE of the order-blind predictor — needs truth fields;
- a truth-side order-blind reference as a *separate* number — on the contrast metric it is
  provably the same number (see below); its arm-level metrics need truth fields;
- any four-seed `ensemble` record (see Sensitivity).

No baseline was scored against the learner's own predictions in place of truth.

### The two new baselines are one number

Predeclared in `PREDECLARED.md` §1(a) and confirmed: any predictor emitting one field for
both orderings has a raw contrast of exactly zero, so the **order-blind** predictor
(`f_i = (pred^AB + pred^BA)/2`), the **truth-side order-blind reference**
(`f_i = (y^AB + y^BA)/2`), and the **zero-contrast / persistence** control all produce the
*identical* per-unit `S`. Max absolute difference across all units and both observables:
**0.0**. They are two motivations for one baseline, and both rows below carry the same value
by construction, not by coincidence.

---

## Verification that the package's numbers were reproduced

All five predeclared checks pass far inside the predeclared 1e-9 tolerance.

| Check | Max abs diff | Tol | Pass |
|---|---|---|---|
| V1 — per-unit `S` (ensemble, pooled, 5 per-seed × 2 windows) vs `PATH_VS_EXPOSURE_RESULT.json` | 4.44e-16 | 1e-9 | ✅ |
| V2 — per-unit `raw_S` vs `PATH_VS_EXPOSURE_RESULT.json` | 2.22e-16 | 1e-9 | ✅ |
| V3 — recognition-only `S` = 1 per unit (PVE:228-231) | 1.33e-15 | 1e-9 | ✅ |
| V4 — bootstrap `ci95` for shipped `S` and `margins_vs_recognition_only` (9999 draws, RNG 20260909) | 4.44e-16 | 1e-9 | ✅ |
| V5 — identity `S = raw_S · S_zero` against `truth_scales` | 1.78e-15 | 1e-9 | ✅ |
| cross-file agreement of `‖d_i‖` (PVE `truth_scales` vs SOT `units.b`) | 0.0 | 1e-9 | ✅ |

Reproduced verbatim: `pooled` full `S = 0.5895281214961378`, CI `[0.5729938, 0.6071326]`;
ensemble trajectory `S_mean = 0.396182046169775`, endpoint `S_mean = 3.430547387212768`;
`margins_vs_recognition_only` = `−0.410472` (full) and `+3.624973` (endpoint) — matching
CLAIM_INVENTORY §2 rows 14–15 and §7(i). The bootstrap stream reproduces bit-for-bit on
numpy 2.2.6 despite the package's numpy 2.5.2 pin.

Additional, **not predeclared and not a gate**: for every seed, unit and frame the npz
predictions satisfy `|‖P‖ − ‖delta‖| ≤ ‖P − delta‖ ≤ ‖P‖ + ‖delta‖`, where `‖P‖` comes from
the raw npz arrays and the other two norms from the shipped tables (worst violation
5.6e-17). This ties the shipped scalars to the actual prediction arrays.

---

## Primary table — record `ensemble` (5 seeds averaged before scoring)

Metric `S = ‖dhat_i − d_i‖_W / ‖d_i‖_W` per unit, then the 48-unit mean
(`path_vs_exposure.py:115,129,152-155`). Margin = `S_learner − S_baseline`, paired on the
unit; negative favours the learner. CI = percentile 95 % from 9999 paired whole-unit
resamples, RNG 20260909 (`path_vs_exposure.py:226`), n = 48.

**Trajectory** (all 201 frames; learner `S = 0.396`, CI [0.386, 0.408])

| Baseline | S_baseline | Margin | 95 % CI | CI below 0? | Gated |
|---|---|---|---|---|---|
| common-pattern | 1.000000 | **−0.603818** | [−0.614435, −0.592430] | ✅ yes | yes |
| recognition-only | 1.000000 | **−0.603818** | [−0.614435, −0.592430] | ✅ yes | yes |
| **order-blind (new)** | 1.652570 | **−1.256388** | [−1.285892, −1.227344] | ✅ yes | yes |
| **zero-contrast (new)** | 1.652570 | **−1.256388** | [−1.285892, −1.227344] | ✅ yes | yes |
| first-leg oracle (context only) | 0.345841 | +0.050341 | [+0.033852, +0.067437] | no | **no** |

**Endpoint** (frame 200; learner `S = 3.431`, CI [2.869, 4.048]) — reported, **not gated**

| Baseline | S_baseline | Margin | 95 % CI | CI below 0? | Gated |
|---|---|---|---|---|---|
| common-pattern | 1.000000 | **+2.430547** | [+1.868936, +3.048399] | ✗ no (worse) | no |
| recognition-only | 1.000000 | **+2.430547** | [+1.868936, +3.048399] | ✗ no (worse) | no |
| **order-blind (new)** | 3.402546 | **+0.028001** | [−0.104331, +0.171597] | ✗ no (tie) | no |
| **zero-contrast (new)** | 3.402546 | **+0.028001** | [−0.104331, +0.171597] | ✗ no (tie) | no |
| first-leg oracle (context only) | 1.000000 | +2.430547 | [+1.868936, +3.048399] | no | no |

### GATE

Predeclared (`PREDECLARED.md` §4): PASS iff, on the **trajectory** observable for the
primary `ensemble` record, the 95 % CI of the paired margin lies entirely below 0 against
**all four** fair baselines.

| Baseline | CI entirely below 0 |
|---|---|
| common-pattern | ✅ |
| recognition-only | ✅ |
| order-blind | ✅ |
| zero-contrast | ✅ |

**VERDICT: PASS.** The learner's trajectory advantage survives both missing baselines, and
survives them by a wider margin than it survives the two baselines the package already had
(−1.256 vs −0.604), because the order-blind baseline is a *harder* target than the
common-pattern one on the trajectory (`S = 1.653` vs `S = 1.000`).

---

## Secondary records (reported, not gated)

| Record | Trajectory S | margin vs order-blind (CI) | Endpoint S | margin vs order-blind (CI) |
|---|---|---|---|---|
| `ensemble` (PRIMARY) | 0.396182 | −1.256388 [−1.2859, −1.2273] ✅ | 3.430547 | +0.028001 [−0.1043, +0.1716] ✗ |
| `pooled` (per-seed S averaged) | 0.589528 | −1.063042 [−1.0857, −1.0409] ✅ | 4.624973 | +1.222427 [+0.9907, +1.4663] ✗ |
| `seed_error_pool` (per-seed q² averaged) | 0.802837 | −0.849733 [−0.8677, −0.8322] ✅ | 5.648922 | +2.246376 [+1.8803, +2.6300] ✗ |
| seed 0 | 0.255064 | −1.397506 ✅ | 2.982632 | −0.419914 [−0.7718, −0.0910] ✅ |
| seed 1 | 0.533224 | −1.119346 ✅ | 10.894536 | +7.491990 [+6.3654, +8.6686] ✗ |
| seed 2 | 0.214725 | −1.437845 ✅ | 2.407430 | −0.995116 [−1.2864, −0.7322] ✅ |
| **seed 3** | **1.652570** | **+0.000000 [0, 0]** ✗ | **3.402546** | **+0.000000 [0, 0]** ✗ |
| seed 4 | 0.292058 | −1.360512 ✅ | 3.437722 | +0.035176 [−0.2757, +0.3704] ✗ |

Every learner record beats the order-blind baseline on the trajectory except seed 3. At the
endpoint only seeds 0 and 2 beat it; seed 1 loses to it catastrophically; seed 4 ties.

---

## Seed 3 — a finding, then the sensitivity row

**Finding (discovered after the gate was fixed; it changes no threshold).**
`composed_seed3.npz` has **bit-identical `pred_ab` and `pred_ba`**
(`max|pred_ab − pred_ba| = 0.0` over all 48×201×256 entries). Seed 3 *is* the order-blind
predictor — not near it, exactly it. Consequences, all confirmed numerically: its
`raw_S = 1.000000` on both observables; its `S` equals the order-blind baseline `S` to the
last bit on both observables; its margin against that baseline is exactly `0.000000` with a
degenerate CI `[0, 0]`.

This gives the divergence recorded in `ROUTE_E_RESULT.json` (seed 3 primitive recovery
`0.2685` vs a five-seed median `0.0440`, `divergence_rule`: "one such seed is a scatter
result and is reported as one") a sharper reading than "scatter": at the level of the order
contrast, seed 3 collapsed to complete order-blindness. One seed in five of the shipped
Route E learner *is* the baseline the package was missing. That one order-blind seed is
averaged into the five-seed `ensemble` — the primary record — which is part of why the
ensemble's endpoint raw_S is 0.985, i.e. essentially zero recovered contrast at the endpoint.

### SENSITIVITY — with and without seed 3 (labelled SENSITIVITY; not primary)

The four-seed **`ensemble`** is **not computable** from shipped files: the ensemble residual
needs cross-seed spatial inner products `mean_x(res_k · res_l)`, which require the truth field
`delta_i`. Only the per-seed-averaged records can be recomputed on a subset. No four-seed
ensemble number was invented.

| Record | Seeds | Trajectory S | margin vs order-blind (CI) | Endpoint S | margin vs order-blind (CI) |
|---|---|---|---|---|---|
| `pooled` | all 5 | 0.589528 | −1.063042 [−1.0857, −1.0409] ✅ | 4.624973 | +1.222427 [+0.9907, +1.4663] ✗ |
| `pooled` | **without seed 3** | 0.323768 | −1.328802 [−1.3572, −1.3012] ✅ | 4.930580 | +1.528034 [+1.2384, +1.8328] ✗ |
| `seed_error_pool` | all 5 | 0.802837 | −0.849733 [−0.8677, −0.8322] ✅ | 5.648922 | +2.246376 [+1.8803, +2.6300] ✗ |
| `seed_error_pool` | **without seed 3** | 0.349448 | −1.303121 [−1.3296, −1.2772] ✅ | 6.079064 | +2.676518 [+2.2422, +3.1311] ✗ |

**The gate does not flip either way** (`gate_would_flip = false` for both records). Dropping
seed 3 improves the trajectory substantially (pooled `S` 0.590 → 0.324) and makes the
endpoint *worse* (pooled `S` 4.625 → 4.931), because seed 3's exactly-order-blind endpoint
score (3.403) is the second-best of the five endpoint scores. The trajectory conclusion is
robust to seed 3; the endpoint failure is not rescued by removing it.

---

## Honest reading

The learner's trajectory advantage is real and it survives both baselines the audit found
missing: averaged over the full 201-frame window the ensemble scores `S = 0.396` against an
order-blind/zero-contrast baseline at `S = 1.653`, a paired margin of −1.256 with a 95 % CI
of [−1.286, −1.227] that is nowhere near zero, and the same holds for every individual seed
except the one that is literally order-blind. That is a stronger result than the package
reported, because on the trajectory the order-blind baseline is a harder target than the
common-pattern baseline the package used (1.653 vs 1.000) — the learner beats the harder
one. But the endpoint result is where a manuscript must lead, and it is worse than the
package's framing suggests: `S = 3.431` against the common-pattern baseline is already a
factor-3.4 failure (margin +2.431, CI [+1.869, +3.048]), and against the new baselines the
margin is +0.028 with a CI of [−0.104, +0.172] — the learner and a predictor that emits one
field for both orderings are statistically indistinguishable at `t = 2`. At the endpoint,
where the two orderings receive equal exposure and where the 82.774 %/83.76 % shared-mean
finding lives, this learner recovers no order information that a deliberately order-blind
control does not also "recover". The trajectory advantage is therefore an advantage on the
observable that carries 11.67× more contrast signal (CLAIM_INVENTORY §2 row 4) and is *not*
an equal-exposure comparison (`path_vs_exposure.py:9-14`), while the one nontrivial
equal-exposure frame shows no advantage at all. Four qualifications belong with the PASS:
n = 48 test units from one cohort, one PDE pair, one resolution, one IC family that is
itself a fitted substitute; the CIs are conditional on the fitted models and the training
mean, not on training-seed variance; the five seeds are not independent units, and one of
the five is a collapsed model that the primary `ensemble` record averages in; and the
order-blind baseline's *arm-level* accuracy could not be evaluated at all, because the truth
bundle is not shipped.

---

*Files written, all under `results/family3_baselines/`:
`PREDECLARED.md`, `analyze_family3.py`, `RESULT.json`, `RESULT.md`.
Nothing under `recovery/` was modified.*
