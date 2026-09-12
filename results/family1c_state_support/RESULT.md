# Family 1c — support as multi-dimensional state coverage, not spatial mean

Run 2026-09-11, 08:25–11:36 UTC. Wall clock **3 h 11 min**; 20 new fits, 8.59 process-hours
of training (median 25.1 min/fit — the box was shared with two other training lanes at
load average 60–137, against the package's own 5.1 min/fit estimate). CPU only, $0.
Predeclared in `PREDECLARED.md`, sha256 `14b5cd0418c9cd1e34f08b1de5870f4079a17cc1ef6ffb649348d9d98f174c8a`,
written 08:25 UTC before the PCA basis was fitted, before any Mahalanobis distance was
evaluated, before P2's off-support error was read, and before any BROAD-S corpus existed.

---

## The honest number first

**FAMILY VERDICT: FAIL.** Both halves fail their predeclared gates, but they fail for
opposite reasons and the two failures should not be read the same way.

- **PRED fails, and it fails hard.** Under the package's own state-support statistic —
  PCA at 99 % variance, Mahalanobis distance, 99th-percentile training radius, the exact
  construction behind `pca11_MD_train99_inside` — the Spearman correlation between
  `log10 R` and mean switch-state distance across the six cells is **−0.257**, against a
  required **+0.8**. It is not weak; it has the wrong sign. And the binding qualitative
  test fails too: **Cahn-Hilliard's switch states are INSIDE the PCA-99 radius**
  (coverage **0.910**), which `PREDECLARED.md` §3 named in advance as the condition under
  which *"the statistic fails and you report that"*. So it is reported. **No statistic
  was swapped in afterwards.**
- **REPAIR fails on one gate of five, and that gate is the measurement, not the repair.**
  The BROAD-S intervention works, by a very large margin: on diffusion / Cahn-Hilliard the
  ten-seed median `R` falls from **46.278 to 0.787**, ratio **0.0170** against the
  package's 0.25 limit; on the P1 positive control it falls from **115.405 to 2.569**,
  ratio **0.0223**, i.e. **7.2× better than the package's own mean-broadening**
  (`17.650 / 110.234 = 0.160`). Gates REPAIR-1, -2, -3 and -5 all pass by wide margins.
  What fails is **REPAIR-4**: the achieved coverage *as statistic S measures it* is
  **0.828**, below the predeclared 0.9 — and it is *lower* than the coverage S assigns to
  the narrow corpus (0.910) that degrades 46×.

The one-sentence result: **broadening the training dictionary in state space repairs the
Cahn-Hilliard degradation that mean-broadening could not touch (46.3 → 0.79 at fixed
n = 640), while the state-support statistic that motivated the intervention neither
predicts the degradation nor registers the coverage that produced the repair.**

---

## 1. Prediction table (`PRED`, no new fits)

`R` per cell is the median `e_off/e_on` over that cell's seeds. `D̄` is the mean
Mahalanobis distance of the 256 switch states in the PCA-99 basis of **that model's own
training cloud**; `r99` is the 99th-percentile Mahalanobis radius of that cloud's 64 640
training states; coverage is the fraction of switch states inside `r99`. `S2` is the
predeclared robustness statistic (mean L2 from a switch state's 8-band energy-fraction
profile to the nearest training state's).

| cell | training offsets | R | seeds | log₁₀R | PCA k (99 %) | r99 | **D̄** | coverage | in/out | S2 |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 narrow  | (0.4865, 0.5171) | 110.234 | 5 (pkg) | 2.042 | 10 | 9.030 | 7.708 | 0.820 | INSIDE | 9.32e−4 |
| P1 broad   | (0.4865, 0.7258) | 17.650 | 5 (pkg) | 1.247 | 11 | 9.095 | 7.896 | 0.785 | INSIDE | 7.32e−4 |
| P2 narrow ⚠ | (0.4865, 0.5171) | **177.179** | 10 | 2.248 | 10 | 9.030 | **1.907** | 1.000 | INSIDE | 1.04e−5 |
| P2 broad ⚠  | (0.48, 0.90)     | **0.100** | 10 | −0.999 | 10 | 8.340 | **1.943** | 1.000 | INSIDE | 5.02e−6 |
| CH narrow  | (0.4865, 0.5171) | 46.278 | 10 | 1.665 | 10 | 9.030 | 5.944 | **0.910** | **INSIDE** | 9.39e−3 |
| CH broad   | (0.48, 0.60)     | 35.799 | 10 | 1.554 | 11 | 9.072 | 6.074 | 0.910 | INSIDE | 9.22e−3 |

⚠ **P2's ratio does not resolve and is flagged everywhere.** Family 1 stopped P2 at gate
G2 (dynamic range 30.8 < 100) and never read `e_off`; the 20 models were checkpointed and
`e_off` is read here for the first time, which **does not change family 1's P2 verdict —
it stays FAIL on G2**, and nothing was written into `family1_second_pairs/`. The reason
G2 blocked is visible in the numbers: P2 narrow's median `e_off` (0.02967) is **5.01×
larger than `persistence_off`** (0.00592), i.e. the fitted model is five times *worse
off-support than the identity predictor*, so `G4_censored_above` fails outright; and P2
broad's `e_off` is 173× *below* persistence, giving `R = 0.100`. P2's cell is censored in
both directions. Every other cell sits at `e_off/persistence_off` between 0.06 and 0.26.

### Gate PRED

| gate | definition | threshold | measured | result |
|---|---|---|---|---|
| **PRED-A** | Spearman ρ(log₁₀R, D̄), 6 cells | ≥ 0.8 | **−0.257** (p = 0.62) | **FAIL** |
| **PRED-B** | CH narrow switch states outside the PCA-99 radius | coverage < 0.5 | **0.910** | **FAIL** |
| PRED-C | P1 broad and P2 broad inside | coverage ≥ 0.5 both | 0.785, 1.000 | PASS |
| S2 (robustness, **not a gate**) | Spearman ρ(log₁₀R, S2) | reported | **+0.257** (p = 0.62) | — |

**PRED = FAIL**, binding on PRED-A and PRED-B.

Reported, **post-hoc, gating nothing**: dropping the censored P2 cells and rank-correlating
on the four cells whose metric resolves gives ρ = **−0.400** for S and **+0.400** for S2 —
still nowhere near 0.8, so P2 is not the sole cause. A residual (out-of-basis) norm that
was computed *after* S and S2 were read gives ρ = **+0.829** on the six cells
(`results/DIAGNOSTIC.json`); it is recorded as a lead for a future predeclared family and
**it is not this family's statistic, does not appear in `PREDECLARED.md`, and changes no
verdict here**.

### Why S fails (mechanism, not excuse)

`k = 10` components carry 99.0 % of the narrow corpus's variance — the same `k ≈ 11` the
package found. Mahalanobis distance is computed **inside that retained subspace only**, so
whatever a switch state does in the discarded 246 directions is projected away before the
distance is taken. The post-hoc residual numbers make the consequence concrete: 0 % of
Fisher-KPP and 0 % of Allen-Cahn switch states lie inside the training residual envelope
(their out-of-basis norms are 3.31 and 5.25 against a training 99th percentile of 0.25),
yet their *Mahalanobis* distances are 7.7 and 1.9. Cahn-Hilliard is the reverse: its
switch states are genuinely close to the training cloud in every L2 sense (residual 0.23,
63 % inside the residual envelope, Mahalanobis 5.94 inside `r99` = 9.03) — **and it still
degrades 46×**. A whole-state L2 geometry, in or out of the PCA basis, does not see what
breaks the operator on this pair. Only S2, the spectral-profile statistic, ranks
Cahn-Hilliard as the most distant cell (9.4e−3, an order of magnitude above P1's 9.3e−4),
which is the physics — sharp phase-separated interfaces carry high-wavenumber content the
narrow diffusion corpus never contains — but S2's six-cell rank correlation is destroyed
by the censored P2 cell.

---

## 2. Repair table (`REPAIR`, 20 new fits)

BROAD-S sampler, fixed in `PREDECLARED.md` §4.1 before any `R` was read: 640 base ICs from
the package generator at the **narrow** offsets, seed `MASTER_SEED + 1` (the evaluation
draws `+7` and `+8` are used nowhere in the construction); four index blocks of 160 given
bursts of the pair's second operator of length **m = 0, 25, 50, 100**; the 640 resulting
fields rolled 100 steps under diffusion exactly as the package builds any corpus.
**n = 640 in both conditions; only which states the cloud contains moves.**

Per-seed `R = e_off/e_on`, ten seeds, same architecture, optimiser, schedule, cohorts and
seeds as the narrow arm:

| seed | CH `R_narrow` | CH `R_broadS` | P1 `R_narrow` | P1 `R_broadS` |
|---|---|---|---|---|
| 0 | 44.700 | 0.7845 | 74.993 | 2.8020 |
| 1 | 51.760 | 0.7568 | 121.384 | 2.5296 |
| 2 | 44.930 | 0.7898 | 97.473 | 1.9576 |
| 3 | 64.518 | 0.6763 | 110.311 | 2.6085 |
| 4 | 46.920 | 0.8198 | 132.521 | 2.8571 |
| 5 | 42.477 | 0.7463 | 114.047 | 1.9518 |
| 6 | 55.200 | 0.8009 | 116.762 | 2.4050 |
| 7 | 47.960 | 0.7668 | 89.718 | 2.6542 |
| 8 | 44.910 | 0.8812 | 124.160 | 2.0310 |
| 9 | 45.640 | 0.9147 | 197.865 | 2.9490 |
| **median** | **46.278** | **0.787** | **115.405** | **2.569** |
| range | 42.477–64.518 | 0.676–0.915 | 74.993–197.865 | 1.952–2.949 |
| **ratio** | — | **0.0170** | — | **0.0223** |
| coverage achieved (S) | — | **0.828** | — | 0.742 |

`R_narrow` sources: CH from family 1's banked ten-seed negative control; P1 measured here
by evaluating the ten banked narrow checkpoints — **verified byte-identical across all
three pairs by sha256 before reuse** — against the Fisher-KPP switch cohort. That P1
narrow ten-seed median, **115.405** (range 74.99–197.87), reproduces the package's
five-seed **110.234** to within 4.7 %, which is a second independent harness check.

### Gate REPAIR

| gate | definition | threshold | measured | result |
|---|---|---|---|---|
| REPAIR-1 | CH median `R_broadS` / median `R_narrow` | ≤ 0.25 | **0.0170** | **PASS** |
| REPAIR-2 | CH dynamic range `persistence_off / median e_on(narrow)` | ≥ 100 | **344.8** | **PASS** |
| REPAIR-3 | P6 convergence, worst `e_on` over all 20 BROAD-S fits | ≤ 1.573e−3 | **4.066e−4** | **PASS** |
| **REPAIR-4** | CH coverage achieved, fraction inside the PCA-99 radius of the BROAD-S cloud | ≥ 0.9 | **0.828** | **FAIL** |
| REPAIR-5 | P1 positive control ratio | ≤ 0.25 | **0.0223** | **PASS** |

**REPAIR = FAIL**, binding on REPAIR-4 alone.

The ordering device was kept: all 20 seeds wrote `e_on` to `*_pregate.json` and
checkpointed their weights, `results/broadS/GATES.json` (REPAIR-2, REPAIR-3) was written
from those at 11:33:03 UTC, and `e_off` was read only afterwards, at 11:33:46.

### Why REPAIR-4 fails, and what it means

The BROAD-S corpus for Cahn-Hilliard literally contains 160 units produced by running
Cahn-Hilliard for exactly the 100 steps that produce the switch cohort — a different
random draw of the same distribution. Actual coverage is as complete as a fixed-`n`
dictionary can make it, and the fits prove it: off-support error drops below on-support
error (`R = 0.787 < 1`). Yet statistic S reports coverage **0.828 for BROAD-S against
0.910 for the narrow corpus it replaced.** S went *down* when real coverage went *up*.

The reason is the same one that sinks PRED. Adding sharp states to the cloud pulls new,
modest-variance directions into the top-`k` basis; the switch states then have large
coordinates divided by small eigenvalues in exactly those directions, so their Mahalanobis
distance *rises* (mean 5.94 → 6.62) even though they are now represented in training. A
Mahalanobis radius over a PCA-99 basis is not a monotone function of the coverage it is
supposed to measure. **This is the substantive finding of the repair half, and it is the
reason the family verdict is FAIL rather than PASS.**

Note also that `R_broadS < 1` on Cahn-Hilliard: the fitted model is now *better* on the
switch states than on the on-support cohort. Both errors are small and the metric is not
censored (`e_off` 2.2e−4 against `persistence_off` 0.0663, a factor 300 of headroom), so
this is a real repair and not a floor artefact — but it does say the switch task became
easier than the on-support task, which no gate in the package's set was written to detect.

---

## 3. Family verdict

**FAIL** (`PREDECLARED.md` §5: PASS requires PRED **and** REPAIR).
**Binding gates: PRED-A, PRED-B, REPAIR-4.**

What the reader should take from it, stated without softening the verdict:

1. `CLAIM_INVENTORY.md` §7(ii)'s falsification clause — *"falsified if … degradation is
   uncorrelated with support distance"* — is now met under **two** operationalisations of
   support, not one. Family 1 showed the spatial-mean interval does not predict
   degradation (Cahn-Hilliard: zero mean shift, 46× degradation). Family 1c shows the
   package's own multi-dimensional replacement does not predict it either, and across six
   cells is weakly *anti*-correlated. The second half of the claim — *"broadening the
   training dictionary to cover the switch-state support, at fixed sample count, reduces
   that degradation by a pre-declared factor"* — **survives, and survives more strongly
   than the package's version of it**: 0.0170 and 0.0223 against the declared 0.25, and
   7.2× better on P1 than the package's own mean-broadening.
2. So the intervention is real and the *statistic* is the broken part. A dictionary that
   contains states of the kind the model will meet repairs the failure; no L2 geometry
   tried here — Mahalanobis inside a PCA-99 basis, or the residual outside it — tells you
   in advance which pairs need it or certifies when you have it.
3. The next predeclared family should gate on a statistic that is (a) monotone under the
   intervention and (b) sensitive to spectral content, and should screen pairs on
   `persistence_off` **before** fitting. The post-hoc residual ρ = 0.829 is a lead, not a
   result.

---

## 4. Every choice made here

1. **Imported, never copied, never written to.** `run_baseline.py` and `timing_probe.py`
   are imported from `dependency`; `second_operator`, `save_params` and `load_params` are
   imported from `family1_second_pairs/run_family1.py`. No function that writes into those
   trees was called, and no file under `recovery/`, `dependency/`,
   `family1_second_pairs/`, `family1b_pair_screen/` or `family1_rerun/` was modified.
   Large arrays and the 20 new checkpoints (42 MB) live in `work/family1c/`.
2. **`k` is measured, not fixed.** 99 % of variance needs `k = 10` on the narrow corpus
   and 10–11 on the others, which independently reproduces the package's `pca11`.
3. **Training cloud = all 101 frames of all 640 units** (64 640 states), because
   `make_pairs` uses frames 0–90 as inputs and 10–100 as targets and their union is every
   frame. Geometry in float64; fits in float32, since `run_baseline.py` does not set
   `jax_enable_x64`.
4. **P1 narrow at 10 seeds was obtained without refitting**, from the banked narrow
   checkpoints (sha256-verified identical across the three pairs, as family 1's
   `reuse_narrow.py` established). 10 fits saved.
5. **Private JAX compilation cache** at `work/family1c/jax_cache` instead of
   the shared `JAX_COMPILATION_CACHE_DIR` from `ENV.sh`, because the shared cache holds
   AOT entries written by another lane under different machine features that XLA refuses
   with a SIGILL warning. Everything else in `PREDECLARED.md` §6 was left exactly as
   declared: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
   XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`, three concurrent single-threaded workers.
6. **Concurrency was held to 3 workers** even though fits ran at 25 min instead of the
   package's 5 min, because the box was shared. A benchmark confirmed the slowdown is
   contention and not the corpus: 224.7 ms/step on the narrow corpus against 232.9 ms/step
   on BROAD-S, both ~6× the package's 38.6 ms/step. Concurrency changes wall time only.

## 5. Not done

- **No fit was run outside the 20 predeclared BROAD-S fits.** In particular no BROAD-S
  variant was tried after seeing REPAIR-4 fail — changing the sampler to chase the
  coverage number is exactly what the method forbids.
- **No third support statistic was adopted.** The residual diagnostic in
  `results/DIAGNOSTIC.json` is labelled post-hoc in the file itself and gates nothing.
- P2's `R` exists now but P2 remains a failed cell under family 1's G2, and no attempt was
  made to rescue it (a longer horizon `k` does not: family 1's `G2_DIAGNOSTIC.json` shows
  `persistence_off` on P2 never exceeds 0.0519 up to `k = 400`).
- One architecture (A1 FNO), one resolution (256), one horizon (`k = 10`), one IC family
  (the package's fitted substitute — every number inherits that caveat). Seeds are not
  independent units. Six cells is a small `n` for a rank correlation and is reported as a
  coarse test; the six-cell Spearman's p-value is 0.62 in both directions, so PRED-A's
  failure is a failure to demonstrate the correlation, not a demonstration of its absence.

## 6. Files

```
PREDECLARED.md                    statistic, sampler, gates; written before any computation
results/PREDECLARED.sha256        its hash, taken at 08:25:20 UTC
run_family1c.py                   harness (imports the package's; writes only here)
aggregate.py                      prediction/repair tables, gates, verdict
diagnostic.py                     POST-HOC residual diagnostic; gates nothing
results/SUPPORT.json              statistic S and S2 for all 8 clouds, truth states only
results/DIAGNOSTIC.json           post-hoc out-of-basis residual, labelled as such
results/banked/{P2_narrow,P2_broad,P1_narrow}.json   R from family 1's banked fits
results/broadS/GATES.json         REPAIR-2/-3, written before any e_off was read
results/broadS/<pair>_seed<k>[_pregate].json         20 new fits, per seed
RESULT.json                       machine-readable result
logs/                             one log per worker
work/family1c/ckpt/  20 BROAD-S checkpoints, 42 MB
```
