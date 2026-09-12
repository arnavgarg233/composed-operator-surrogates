# PREDECLARED — Family 3: order-blind and zero-contrast baselines for Route E

Written **before** any number was computed. Nothing below may be edited after a
result is seen. Author: restart worker. Date: 2026-09-10.

Package under analysis (READ-ONLY, never modified):
`dependency/composed-operator-surrogates`
Audit reference: `dependency/CLAIM_INVENTORY.md` §3, §6
("Baselines demanded by the 82.774 %/83.76 % finding"), §7 row 3.
Output directory (only place files are created):
`results/family3_baselines/`

All line-number citations are to the shipped files at their shipped bytes:
`scripts/route_e/path_vs_exposure.py` (PVE) and `scripts/route_e/s_of_t.py` (SOT).

---

## 0. Notation (fixed by the package, PVE:16-24)

For test unit `i` (i = 0..47), frame `j` (j = 0..200), space `x` (256 points):

- `delta_i = y_i^AB - y_i^BA` — raw truth order contrast.
- `m = mean over evaluator-TRAIN units of (y^AB - y^BA)` — the shared/common pattern
  (`pattern`, PVE:222; SOT `load_truth` returns it as `center`, SOT:96).
- `d_i = delta_i - m` — centered truth contrast (PVE:128).
- `dhat_ki = pred_ki^AB - pred_ki^BA - m` — centered predicted contrast (SOT:108).
- Residual `res = dhat - d`; `q2[i,j] = mean_x(res^2)` (PVE:115, SOT:111);
  `b2[i,j] = mean_x(d_i^2)` (PVE:129, SOT:282); `delta2[i,j] = mean_x(delta_i^2)` (PVE:130).

## 1. Baseline definitions

### (a) Order-blind predictor (NEW)
One field is emitted for both orderings. Concretely, the **learner-mean order-blind
predictor**: `pred_i^AB = pred_i^BA = f_i = (pred_i^AB,learner + pred_i^BA,learner)/2`,
per unit, per seed / per record, per frame, per spatial point.

Its raw contrast prediction is identically `f_i - f_i = 0`; its centered contrast
prediction is therefore `0 - m = -m`, and its residual against truth is
`-m - d_i = -m - (delta_i - m) = -delta_i`. Hence
`q2_orderblind[i,j] = delta2[i,j]` exactly.

A **truth-side order-blind reference** (`f_i = (y_i^AB + y_i^BA)/2`) is declared to be
the same object on the contrast metric, for the same reason: any predictor that emits
one field for both orderings has zero raw contrast. It is therefore reported as an
identity, not as a second number. If the truth fields are not shipped, the truth-side
reference's *arm-level* (per-ordering state accuracy) metrics are declared
**not computable** and will not be substituted for.

**Predeclared consequence, stated before computing:** because the contrast metric
depends on the predictor only through its raw contrast, the order-blind predictor and
the zero-contrast control (b) must coincide *exactly* on S, for every unit, window and
observable. They are two motivations for one number, not two independent baselines.
If the computation shows otherwise, that is an implementation error, not a finding.

### (b) Zero-contrast / per-unit persistence control (NEW)
Predicts `delta-hat_i = 0` (raw), i.e. centered prediction `= -m`, residual `= -delta_i`,
so `q2_zero[i,j] = delta2[i,j]` exactly. `delta2` is shipped as
`PATH_VS_EXPOSURE_RESULT.json → truth_scales.raw_contrast_rms_per_unit_frame`
(PVE:264, `= sqrt(truth['delta2'])`, shape (48,201)).

### (c) Common-pattern baseline (present in package)
Predicts the shared mean contrast: raw `delta-hat_i = m`, centered `= 0`,
residual `= -d_i`, so `S = 1` per unit by construction (SOT:133,
`margin_vs_common_pattern = S_mean - 1.0`).

### (d) Recognition-only baseline (present in package)
`mu^AB` for AB and `mu^BA` for BA (PVE:43-47, PVE:227). Its raw contrast is `m`, its
centered contrast is 0, so its per-unit centered `S = 1` on every nondegenerate window —
asserted by the package itself at PVE:228-231. It is therefore numerically identical to
(c) on the contrast metric S, and is reported as its own row anyway.

### (e) First-leg oracle (present, NOT a fair baseline)
PVE:255-258 label it "NOT a deployable baseline". Reported for context only; not gated.

## 2. Metric S (as in path_vs_exposure.py)

For window `W` (PVE:95-96; `full = (0,200)`, `endpoint = (200,200)`; slice at PVE:152):

```
S_{k,i,W} = sqrt( mean_{j in W} q2[k,i,j] ) / sqrt( mean_{j in W} b2[i,j] )      PVE:153-154
Sbar_{k,W} = mean over the 48 units of S_{k,i,W}                                  PVE:105 ('mean')
raw_S_{k,i,W} = sqrt( mean_{j in W} q2 ) / sqrt( mean_{j in W} delta2 )           PVE:155
```

`ratio()` (SOT:99-102) returns NaN where the denominator is not > 0; undefined units
propagate rather than being dropped (PVE:106-109). Smaller S is better; `S = 1` is
exactly what the common-pattern predictor gives.

**Observables** (SOT:249-257):
- **trajectory** — `q2, b2` averaged over all 201 frames (SOT:250). Identical to PVE
  window `full = (0,200)` (PVE:95, slice(0,201)).
- **endpoint** — frame 200 only (SOT:251). Identical to PVE window `endpoint`.

**Learner records** (PVE:243-252, SOT:297-310):
- **PRIMARY: `ensemble`** — predicted fields averaged across the five seeds *before*
  scoring (PVE:251, SOT:308). This is the record carrying the package headline
  `S_trajectory = 0.396`, `S_endpoint = 3.431` quoted in CLAIM_INVENTORY §7(i).
- Secondary, reported not gated: `pooled` (PVE:249 — per-seed S averaged within unit),
  `seed_error_pool` (SOT:305-306 — per-seed q2 averaged, then the ratio), and each of
  the five `per_seed` records.

## 3. Margin and uncertainty

`margin_{i} = S_learner,i - S_baseline,i` per unit (paired on the unit, PVE:187-188).
Negative favours the learner (PVE:68).

Paired bootstrap, matching the package exactly (PVE:205, PVE:226):

```
weights = numpy.random.default_rng(20260909).multinomial(48, numpy.full(48, 1/48), 9999) / 48
ci95    = numpy.quantile(weights @ per_unit_values, [0.025, 0.975])          PVE:104, PVE:108
```

n = 48 independent test units; 9999 paired whole-unit resamples; RNG seed 20260909;
one weight matrix shared across every baseline, observable and record (PVE:64-65).
The seed axis is explicitly **not** an independent unit (PVE:240, SOT:294): CIs are
conditional on the fitted models, the training mean `m`, and the fixed demonstrations.

## 4. GATE (pre-declared, not to be altered)

**PASS** iff, on the **trajectory** observable, for the PRIMARY record (`ensemble`),
the 95 % CI of the paired margin `S_learner - S_baseline` lies **entirely below 0**
against **every** one of the four fair baselines:

1. common-pattern,
2. recognition-only,
3. order-blind (new),
4. zero-contrast / per-unit persistence (new).

If any one of the four CIs includes or lies above 0, the verdict is **FAIL**.

The **endpoint** observable is **reported, not gated** (CLAIM_INVENTORY §7 row 3:
"endpoint failure reported, not gated"). Its numbers are stated plainly whatever they
are, including the known `S ≈ 3.43` failure against the common-pattern baseline.

The first-leg oracle is not part of the gate (it is not a deployable baseline).

## 5. Validation of the metric implementation (pre-declared tolerance)

Before any new baseline is computed, the implementation must reproduce the package's
own shipped numbers:

- V1. Recomputed per-unit `S` for the learner (`ensemble`, `pooled`, all `per_seed`)
  on windows `full` and `endpoint`, from shipped per-unit/per-frame `q2` and `b2`,
  must match `PATH_VS_EXPOSURE_RESULT.json → <record>.windows.<W>.contrast.S.per_unit`
  to **absolute tolerance 1e-9**.
- V2. Recomputed per-unit `raw_S` must match the shipped `contrast.raw_S.per_unit` to
  **1e-9**.
- V3. Recognition-only per-unit `S` must equal 1 to **1e-9** (the package's own
  assertion, PVE:228-231).
- V4. The bootstrap CI recomputed for the shipped learner `S` and for
  `margins_vs_recognition_only` must match the shipped `ci95` to **1e-9**
  (same RNG, same weights).
- V5. Independent cross-check of the truth scales, using only shipped ratios:
  `S_zero,i,W = S_learner,i,W / raw_S_learner,i,W` must equal
  `sqrt(mean_W delta2) / sqrt(mean_W b2)` computed from `truth_scales` to **1e-9**.
  (Algebraic identity: `S = raw_S · S_zero`.)

If V1–V3 or V5 fail at 1e-9, the analysis is declared invalid and no baseline number
is reported. V4 failing invalidates only the CI, not the point estimates.

## 6. Truth availability rule (pre-declared)

`data/corpus/EVALUATOR_BUNDLE.npz` is **not shipped** (CLAIM_INVENTORY §1, §2 row 14;
`data/manifest.json → not_shipped`). Therefore:

- Any quantity requiring the truth **fields** `y_i^AB`, `y_i^BA`, `delta_i`, `d_i` or
  `m` at spatial resolution is declared **not computable** and will be reported as such,
  with the reason.
- Substituting the learner's own predictions for truth is **forbidden**: a baseline
  scored against the learner's predictions is not a baseline. No such substitution will
  be made, and no such number will be reported.
- Quantities reducible to the shipped per-unit/per-frame **norms** (`b2` = `units.b`^2
  in `S_OF_T_RESULT.json`; `delta2` = `truth_scales.raw_contrast_rms_per_unit_frame`^2
  and `b2` = `truth_scales.centered_rms_per_unit_frame`^2 in
  `PATH_VS_EXPOSURE_RESULT.json`; `q2` = `units.q`^2 per record in `S_OF_T_RESULT.json`)
  ARE computed, because the two new baselines' residual is `-delta_i`, whose norm is
  shipped.

## 7. Seed-3 sensitivity (pre-declared as SENSITIVITY, not primary)

`ROUTE_E_RESULT.json → divergence_rule` and `composed_per_seed.3` record seed 3 as a
divergent scatter seed (primitive recovery `0.2685` vs a five-seed median `0.0440`).
A **sensitivity row** drops seed 3 and recomputes over seeds {0,1,2,4}. It is labelled
SENSITIVITY everywhere and never replaces the primary five-seed numbers.

Declared in advance: the four-seed **`ensemble`** record is **not computable** from
shipped files, because the ensemble residual requires the cross-seed spatial inner
products `mean_x(res_k · res_l)`, which need the truth field `delta_i`. Only the
per-seed-averaged records (`pooled` = mean of per-seed S; `seed_error_pool` = mean of
per-seed q2) can be recomputed on a seed subset, and only those will be shown in the
sensitivity row. No four-seed ensemble number will be invented or approximated.

## 8. Reporting rule

The honest number is reported first: the endpoint result is stated in the same table
and the same paragraph as the trajectory result, and the abstract sentence leads with
whichever of the two is worse for the learner's claim.
