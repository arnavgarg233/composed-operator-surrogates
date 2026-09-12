# Family 1d — the out-of-basis residual, predeclared and tested out of sample

Run 2026-09-11, 11:46–12:54 UTC. Wall clock **1 h 08 min**; **5 new fits**, 2.67
process-hours of training (median 33.8 min/fit on a box shared with the live family-1
rerun, against the package's 5.1 min/fit estimate). CPU only, $0.
Predeclared in `PREDECLARED.md`, sha256
`c50bc3eeb6186dd7146085d308f71f08995cd30827491c14bf4f289946f9cdb5`, hashed at
**11:46:11 UTC** before any number in this family was computed.

---

## The honest number first

**FAMILY VERDICT: FAIL (PARTIAL).**

Family 1c found, post-hoc, that an out-of-basis residual ranked its six cells at
`rho = +0.829`. This family predeclared that statistic as **S3** and tested it on six
cells it was not fitted on. **On the held-out cells `rho = +0.600`** against a required
`+0.8` — the correlation does not survive the move out of sample. The intervention-side
half of the gate passes: coverage `C3` rises under state-space broadening on both pairs.

- **PRED-S3-A FAILS**: `rho(log10 R, D3) = +0.600` (p = 0.208) over 6 held-out cells.
- **PRED-S3-B PASSES**: `C3(BROAD-S) > C3(narrow)` on both, CH `0.633 → 0.871`,
  P1 `0.000 → 0.086`.
- **PARTIAL** because the live family-1 rerun produced no RESULT file inside the window
  (8 of 40 fits at close), so held-out group (a) contributed nothing.

**PRED-S3-A's outcome was fixed before the two new cells were fitted, and this is the
most important sentence in the report.** `CH_broadS` carries the *highest* residual of the
four low-residual held-out cells (`D3 = 0.28736`) and the *lowest* degradation
(`R = 0.787`). Its rank displacement alone contributes `d^2 = 9` of the six-cell budget,
which caps `rho` at `1 - 6*9/210 = 0.743 < 0.8` **whatever the two `S1_fkpp_r0.5` cells
had done**. The gate could not have been rescued, or broken, by my choice of extra cells.
The realised `sum d^2` was 14, giving `+0.600`.

---

## 1. Held-out cells

`R` = median over that cell's seeds of `e_off/e_on`, the package's definition unchanged.
`D3` = mean over the 256 switch states of the out-of-basis residual
`||(x-mu) - U U^T (x-mu)||_2` in that model's own PCA-99 training basis. `C3` = fraction
of switch states inside the 99th-percentile *training* residual. `D̄` = the PCA-99
Mahalanobis mean — family 1c's failed statistic S — carried for contrast.

| cell | R | seeds | log₁₀R | **D3** | **C3** | D̄ (Mahalanobis) | cov_S | k | blind |
|---|---|---|---|---|---|---|---|---|---|
| `FK05_broad` | 16.437 | 5 | +1.216 | 0.16034 | 0.0000 | 8.7101 | 0.5664 | 11 | **yes** |
| `P1_broadS` | 2.569 | 10 | +0.410 | 0.27964 | 0.0859 | 8.2245 | 0.7422 | 11 | no |
| `F6_P1_broad` | 18.385 | 2 | +1.264 | 0.27971 | 0.0000 | 7.8962 | 0.7852 | 11 | no |
| `CH_broadS` | 0.787 | 10 | −0.104 | 0.28736 | 0.8711 | 6.6204 | 0.8281 | 10 | no |
| `FK05_narrow` | 58.676 | 10 | +1.768 | 1.72747 | 0.0000 | 8.6547 | 0.5625 | 10 | **yes** |
| `F6_P1_narrow` | 90.681 | 2 | +1.958 | 3.31308 | 0.0000 | 7.7082 | 0.8203 | 10 | no |

Sources: `CH_broadS`/`P1_broadS` from family 1c's ten BROAD-S seeds; `F6_*` from family
6's two smoke seeds using `R_frame0_bridge` (family 6's headline `e_on` is measured at
frame 100, not the package's frame 0 — the bridge column is the family-1-comparable one,
and both conventions give the same rank order so no gate turns on the choice);
`FK05_narrow` from family 1's ten sha256-verified banked narrow checkpoints evaluated
against the new switch cohort (**no fit**); `FK05_broad` from the 5 fits made here.

### The six selection cells (reported, gate nothing)

| cell | R | seeds | log₁₀R | **D3** | **C3** | D̄ | cov_S | k |
|---|---|---|---|---|---|---|---|---|
| `P1_narrow` | 110.234 | 5 | +2.042 | 3.31308 | 0.0000 | 7.7082 | 0.8203 | 10 |
| `P1_broad` | 17.650 | 5 | +1.247 | 0.27971 | 0.0000 | 7.8962 | 0.7852 | 11 |
| `P2_narrow` | 177.179 | 10 | +2.248 | 5.24684 | 0.0000 | 1.9072 | 1.0000 | 10 |
| `P2_broad` ⚠ | 0.100 | 10 | −0.999 | 0.01214 | 1.0000 | 1.9429 | 1.0000 | 10 |
| `CH_narrow` | 46.278 | 10 | +1.665 | 0.23047 | 0.6328 | 5.9442 | 0.9102 | 10 |
| `CH_broad` | 35.799 | 10 | +1.554 | 0.17635 | 0.0000 | 6.0744 | 0.9102 | 11 |

⚠ P2 remains censored in both directions and FAILED at family 1's G2; the caveat travels
with it here as everywhere. Every one of these eight family-1c cells was **recomputed
independently by this family's own code and reproduces `DIAGNOSTIC.json` and
`SUPPORT.json` to every published digit**, including the headline `+0.8286` and the
Mahalanobis `−0.2571`.

---

## 2. Correlations — S3 and Mahalanobis, with and without the duplicates

| set | n | statistic | Spearman ρ | p |
|---|---|---|---|---|
| **held-out (the gate)** | 6 | **S3 = D3** | **+0.6000** | 0.2080 |
| held-out | 6 | PCA-99 Mahalanobis D̄ | **+0.1429** | 0.7872 |
| held-out | 6 | D3f (normalised residual) | +0.6000 | 0.2080 |
| **held-out minus family-6 duplicates** | 4 | **S3 = D3** | **+0.2000** | 0.8000 |
| **held-out minus family-6 duplicates** | 4 | **PCA-99 Mahalanobis D̄** | **+0.8000** | 0.2000 |
| six selection cells | 6 | S3 = D3 | +0.8286 | 0.0416 |
| six selection cells | 6 | PCA-99 Mahalanobis D̄ | −0.2571 | 0.6228 |
| six selection cells | 6 | D3f | +0.8286 | 0.0416 |

Two things in that table need saying out loud.

**The de-duplicated number is worse, and it is the more honest one.** Family 6's two cells
are trained on corpora *identical* to selection cells `P1_narrow` and `P1_broad`, so their
`D3`, `C3` and `D̄` are the same numbers — they are duplicate points in the predictor and
carry no independent information about it. They were included because the task spec names
them, and `PREDECLARED.md` §2(c) declared the duplication in advance. Strip them and S3
falls to **`+0.200`**.

**On that de-duplicated set the statistic family 1c declared broken outperforms the one
this family predeclared**: Mahalanobis `+0.800` against S3's `+0.200`. At n = 4, p = 0.2,
this is noise, and I am not claiming Mahalanobis works — family 1c's own six-cell
`−0.257` says it does not. What it demonstrates is that **rank correlations on four to six
cells are not stable enough to certify a support statistic**, which is the methodological
lesson the next family should act on.

---

## 3. Monotonicity of C3 under the intervention

The gated comparison passes, cleanly, on both pairs:

| pair | intervention | C3(narrow) | C3(BROAD-S) | verdict |
|---|---|---|---|---|
| CH | state-space broadening | 0.6328 | **0.8711** | PASS |
| P1 | state-space broadening | 0.0000 | **0.0859** | PASS |

This is worth something: `C3` is the **first support statistic in this programme that
moves the right way under the intervention that repairs.** Family 1c's REPAIR-4 failed
precisely because Mahalanobis coverage went *down* (0.910 → 0.828) when real coverage went
up; `C3` goes up.

But the full trace across all five interventions shows what `C3` still cannot do
(post-hoc, gating nothing):

| intervention | R | ratio | D3 | C3 |
|---|---|---|---|---|
| P1 mean-broadening | 110.234 → 17.650 | ×0.160 | 3.3131 → 0.2797 | 0.0000 → **0.0000** |
| **FK05 mean-broadening (new pair)** | 58.676 → 16.437 | ×0.280 | 1.7275 → 0.1603 | 0.0000 → **0.0000** |
| CH mean-broadening | 46.278 → 35.799 | ×0.774 | 0.2305 → 0.1763 | 0.6328 → 0.0000 |
| CH state-broadening | 46.278 → 0.787 | ×0.017 | 0.2305 → 0.2874 | 0.6328 → 0.8711 |
| P1 state-broadening | 110.234 → 2.569 | ×0.023 | 3.3131 → 0.2796 | 0.0000 → 0.0859 |

**`C3` is pinned at exactly 0.0000 across the two mean-broadening interventions that each
cut `R` by 3.6–6.2×.** It registers the BROAD-S repair and is blind to the mean-broadening
repair. So `C3` is monotone in the one intervention the gate named, and uninformative
about two other real repairs. Conversely `D3` tracks the mean-broadening repairs correctly
(3.31 → 0.28 on P1; 1.73 → 0.16 on FK05) and gets the BROAD-S repair backwards
(0.23 → 0.29 while `R` falls 59×). The two statistics are sensitive to disjoint halves of
the intervention space, and neither covers both.

---

## 4. Gate verdicts

| gate | definition | threshold | measured | result |
|---|---|---|---|---|
| **PRED-S3-A** | Spearman ρ(log₁₀R, D3) over held-out cells | ≥ 0.8, ≥ 6 cells | **+0.6000** (p = 0.208), n = 6 | **FAIL** |
| **PRED-S3-B** | C3(BROAD-S) > C3(narrow), both CH and P1 | strict, both | CH 0.6328 → 0.8711; P1 0.0000 → 0.0859 | **PASS** |

**PRED-S3 = FAIL**, binding on PRED-S3-A. Verdict **FAIL (PARTIAL)**.

### Why S3 fails out of sample (mechanism, not excuse) — post-hoc, gates nothing

S3's six-cell success was carried by two cells: `P1_narrow` (D3 = 3.31) and `P2_narrow`
(D3 = 5.25), narrow models facing reaction-driven switch cohorts far outside their basis.
Strip those extremes and the statistic has no resolution left. Among the four held-out
cells with `D3 < 1` — which is **every repaired cell**, the regime the manuscript's claim
is actually about — `D3` spans a factor of **1.79** while `R` spans a factor of **23.4**,
and the rank correlation inside that band is **`−0.40`**. The predictor's dynamic range
collapses exactly where the decision it is supposed to inform gets made.

---

## 5. New pair `S1_fkpp_r0.5` — a result that belongs to claim (ii), not to this gate

The two cells fitted here are the first learner-based numbers on a pair held out from
family 1's construction. All the family-1 gates hold: G1 overlap `0.0`; G2 dynamic range
**528.165**, reproducing family 1b's screen value to ten digits; P6 worst `e_on`
`3.199e−4` against the `1.573e−3` limit; G3 `log10 R` IQR `0.063` against `0.5`; not
censored (max `e_off` `1.54e−2` against `persistence_off` `0.1015`).

**`R` falls from a ten-seed median `58.676` to a five-seed median `16.437`, a ratio of
`0.2801`.** Applying family 1's P3 threshold, `0.2801 > 0.25`: **broadening repairs
substantially but misses the predeclared factor.** Seed-matched (narrow seeds 0–4 only) it
is `0.2766`, still above. Stated plainly, because claim (ii) is falsifiable on exactly
this number: *"Falsified if on any held-out pair or resolution the broad condition fails
to beat narrow by the pre-declared factor."* This is 5 broad seeds, not the 10 family 1
declares, it was predeclared here as a **predictor** cell and not as a P3 test, and the
live rerun is fitting the same cell at 10 seeds. **It is a warning, not a falsification,
and the rerun settles it.**

---

## 6. What this licenses for the manuscript's claim (ii)

Claim (ii) has two halves and this family moves them in opposite directions. On the
first half — *"degrades by a factor predicted by the distance between training-state and
switch-state support"* — the out-of-basis residual is now the **third** operationalisation
of support to fail a predeclared prediction gate, after the spatial-mean interval (family
1) and PCA-99 Mahalanobis (family 1c); it reached `+0.829` on the six cells it was found
on and `+0.600` on six cells it was not, and `+0.200` once the two duplicate cells are
removed, with essentially no resolution (`rho = −0.40`, D3 spanning ×1.79 against R
spanning ×23.4) among the low-residual cells where every repaired model lives. The
falsification clause of claim (ii) — *"falsified if degradation is uncorrelated with
support distance"* — is now met under three operationalisations, and the manuscript should
state the predictive half as an open problem rather than a result: no whole-state L2
geometry tried so far tells you in advance which pairs will degrade or by how much. The
second half — *"broadening the training dictionary to cover the switch-state support, at
fixed sample count, reduces that degradation"* — continues to survive and is now supported
on a genuinely held-out pair: `S1_fkpp_r0.5` repairs `58.676 → 16.437` at fixed `n = 640`,
though at a ratio of `0.2801` it **misses** the predeclared `0.25` factor that claim (ii)
names, which is the one new fact in this family that could falsify the surviving half and
which the ten-seed rerun must resolve. The one genuinely positive methodological result is
`C3`: it is the first statistic in this programme to move the right way under the
intervention that repairs (CH `0.633 → 0.871`, P1 `0.000 → 0.086`, where Mahalanobis
coverage moved the wrong way), but it is pinned at exactly zero across the two
mean-broadening interventions that also repair, so it certifies one kind of dictionary
broadening and is blind to another. The manuscript may claim the intervention; it may not
yet claim a statistic that predicts when the intervention is needed or certifies when it
has been achieved.

---

## 7. Ordering discipline and disclosure

1. **Predeclaration precedes computation.** `PREDECLARED.md` hashed 11:46:11 UTC; the
   first number of this family (`FK05_GEOMETRY.json`) was written at 11:48:14 UTC.
2. **`e_on` and checkpoints precede `e_off`.** All 5 `FK05_broad` seeds wrote
   `*_pregate.json` with `e_on` and a checkpoint; `results/heldout/FK05_GATES.json` was
   written at 12:53:10 UTC; the first `e_off` was read at 12:53:38 UTC.
3. **Four of six held-out cells were not blind, and `PREDECLARED.md` §0.1 said so before
   any of them was used.** The task spec required reading family 1c's `DIAGNOSTIC.json`,
   `RESULT.md` and `PREDECLARED.md` first, which exposed `D3`, `C3` and `R` for
   `CH_broadS`, `P1_broadS` and (by construction) both family-6 cells. Only the two
   `S1_fkpp_r0.5` cells were blind. The mitigating ordering fact, verified from the files:
   `DIAGNOSTIC.json` was written at 08:52 UTC while family 1c's first BROAD-S `e_off` was
   read at 11:33:46 UTC, so those two cells' `D3` and `C3` were fixed before their `R`
   existed.
4. **The gate was not re-defined after any number was seen**, no cell was dropped after
   its numbers were seen, and no statistic was substituted for S3.
5. **The cell that decided the gate was not one I chose.** `CH_broadS` capped `rho` at
   `0.743` before the new fits started.

## 8. Not done / limitations

- **The family-1 rerun contributed nothing.** It is a live detached driver (PID 89748);
  at 11:46 UTC it had 6 of 40 fits, at 12:58 UTC it had 8, and no `RESULT` file ever
  appeared. Polling log: `results/RERUN_POLL.log`. Nothing in `family1_rerun/` was
  written. **My three workers measurably slowed it** — it was running ~26 min/fit before
  this family started and went 44 minutes without a checkpoint during the first round.
  When it lands it supplies four more held-out cells and a ten-seed `S1_fkpp_r0.5` broad
  arm; the gate should be recomputed then, and `PREDECLARED.md` §2(a) already declares it.
- Only 5 seeds on `FK05_broad`, the maximum the task's new-fit clause allows.
- Six held-out cells is the declared minimum and a small `n` for a rank correlation; no
  p-value here is below 0.05 except the six selection cells' `0.0416`, which is the
  in-sample number.
- Family 6's cells use 48 test units and 2 seeds against the package's 256 and 5–10.
- One architecture (A1 FNO), one resolution (256), one horizon (`k = 10`), one IC family.
  Seeds are not independent units.
- No statistic was tried and discarded. `D3f` was declared in advance as a reported
  secondary precisely so it could not be swapped in; it gives `+0.600`, the same as `D3`.

## 9. Files

```
PREDECLARED.md                         statistic, cells, gates; before any computation
results/PREDECLARED.sha256             its hash, 11:46:11 UTC
run_family1d.py                        harness (imports the package's, family 1's, 1c's)
aggregate.py                           tables, correlations, gates, verdict
worker.sh                              pinned-thread worker, private JAX cache
results/STATISTICS.json                D3/C3/D3f/Mahalanobis for all ten cells
results/heldout/FK05_GEOMETRY.json     truth-only geometry, before any e_off
results/heldout/FK05_GATES.json        G1/G2/P6, before any e_off
results/heldout/FK05_broad_seed*_pregate.json   5 fits, e_on only
results/heldout/FK05_{narrow,broad}.json        R per seed
results/RERUN_POLL.log                 the family-1 rerun poll
results/TABLES.md                      tables generated from RESULT.json
RESULT.json                            machine-readable result + post-hoc block
logs/                                  one log per worker
work/family1d/ckpt/       5 checkpoints, 11 MB
```
