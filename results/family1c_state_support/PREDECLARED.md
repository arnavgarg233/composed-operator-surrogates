# PREDECLARED — Family 1c: support as multi-dimensional state coverage

Written **2026-09-11T05:25Z**, before any number belonging to this family was computed:
before the PCA basis was fitted, before any Mahalanobis distance was evaluated, before
P2's off-support error was read from its banked checkpoints, and before any BROAD-S
corpus was built or any model fitted on one.

Release rules in force: predeclare everything; **never change a gate after seeing a
number**; preserve failures as-is; report the honest number first.
`dependency`, `release-inputs/dependency`,
`release-inputs/family1_second_pairs`, `release-inputs/family1b_pair_screen` and
`release-inputs/family1_rerun` are **read-only** — imported from and read, never written to.
All output of this family goes to `release-inputs/family1c_state_support/` and
`work/family1c/`.

---

## 0. Why this family exists

Family 1's predeclared negative control (diffusion / Cahn-Hilliard) has **zero**
spatial-mean support shift — per-unit mean drift `1.8e-7` over the whole 100-step leg —
yet a fitted FNO degrades by a ten-seed median `R = 46.278`, with dynamic range `344.8`
(so the measurement resolves) and a constant-mode correction worth `0.003 %`. Broadening
the mean-offset range to `(0.48, 0.60)` cut `R` only to `35.799`, ratio `0.774`, nowhere
near the package's own `0.25`.

So the package's operational support statistic — *the interval of spatial means of
training states* (`run_baseline.py:110-118`) — is **not** what predicts degradation.
`CLAIM_INVENTORY.md` §7(ii) makes the target claim falsifiable precisely on this point
("*falsified if … degradation is uncorrelated with support distance*"). This family asks
whether the claim survives under a **general** support definition: coverage of the
training-state cloud in state space, which is what the package itself already measured
(`dependency/results/tables/state_support/`, `pca11_MD_train99_inside`) and never used
as the intervention variable.

Two halves, both predeclared here:
- **PRED** — prediction, no new fits: does state-space distance order the degradation
  that the mean statistic cannot see?
- **REPAIR** — intervention, new fits: does broadening the dictionary *in state space*,
  at fixed sample count, repair the Cahn-Hilliard degradation the mean-broadening
  could not?

---

## 1. Support statistic S (primary) — PCA-99 Mahalanobis

Computed on **truth states only**; no learner enters S.

For a (pair, condition) cell:

1. **Training cloud.** The condition's 640 training units, built exactly as the package
   builds them: `tp.initial_conditions(640, offsets, MASTER_SEED + 1)` then
   `tp.rollout(diffusion, ic, 100)` → `(640, 101, 1, 256)`. Flatten to
   `640 × 101 = 64 640` states in `R^256`. This is the whole set of states the learner
   sees: `rb.make_pairs` takes frames `0..90` as inputs and `10..100` as targets, whose
   union is all 101 frames. Cast to **float64** for the geometry (the package's own
   geometry scripts use float64; the fits stay float32, as `run_baseline.py` has them).
2. **Basis.** `mu` = mean of the 64 640 training states. `C` = covariance
   (`numpy.cov`, denominator `N-1`). Symmetric eigendecomposition of `C`; eigenvalues
   `lambda_1 >= lambda_2 >= ...`, eigenvectors `u_j`. **`k` = the smallest number of
   leading components with cumulative explained-variance fraction `>= 0.99`.** `k` is
   recorded per cell, not fixed in advance.
3. **Distance.** For a state `x`,
   `MD(x) = sqrt( sum_{j=1..k} <u_j, x - mu>^2 / lambda_j )`.
4. **Radius.** `r99` = the 99th percentile (`numpy.percentile`, linear interpolation) of
   `MD` over the 64 640 **training** states. This mirrors the package's
   `pca11_MD_train99_inside`.
5. **Switch states.** The cell's 256 switch states: `tp.initial_conditions(256, NARROW,
   MASTER_SEED + 8)` evolved 100 steps under the pair's second operator, frame 100 —
   byte-identical to the cohort `run_baseline.py` / `run_family1.py` evaluate on.
6. **Cell statistics.**
   - `D_bar` = **mean** `MD` over the 256 switch states (this is the "mean switch-state
     Mahalanobis distance under the model's own training support" the prediction test
     uses);
   - `coverage` = fraction of the 256 switch states with `MD <= r99`;
   - median / min / max `MD` also reported.
7. **Inside / outside.** A cell is **INSIDE** iff `coverage >= 0.5` (equivalently, the
   median switch-state `MD` is within `r99`), **OUTSIDE** otherwise.

## 2. Support statistic S2 (robustness check, declared now, **never a fallback**)

Normalized spectral-energy-profile distance. For a state `u` (256 points, real):
`P_j = |rfft(u)_j|^2`, `j = 0..128`. Eight fixed bands:
`{0}, {1,2}, {3,4}, {5..8}, {9..16}, {17..32}, {33..64}, {65..128}`.
Band-energy fraction `f_b = (sum_{j in b} P_j) / (sum_{j=0..128} P_j)`, giving `f` in
`R^8` per state. The distance of a switch state is the **L2 to the nearest training
state's profile**, minimised over all 64 640 training states of the cell. Cell statistic
`E_bar` = mean over the 256 switch states.

S2 is **reported alongside S** — the same six-cell table and the same Spearman — as a
robustness check. **It is not a fallback.** If S fails its gate, S fails; S2 does not
rescue it and the verdict is written on S.

## 3. Prediction test PRED (no new fits)

The six cells are `(pair, condition)` for
`pair in {P1 diffusion/Fisher-KPP, P2 diffusion/Allen-Cahn, CH diffusion/Cahn-Hilliard}`
and `condition in {narrow, broad}`, with each cell's `S` computed under **that model's
own training support** (i.e. that condition's offsets):

| cell | training offsets | source of `R` |
|---|---|---|
| P1 narrow | (0.4865, 0.5171) | package `BASELINE_RESULT.json`, 5-seed median `110.23398312231919` |
| P1 broad  | (0.4865, 0.7258) | package `BASELINE_RESULT.json`, 5-seed median `17.649529763761315` |
| P2 narrow | (0.4865, 0.5171) | **computed now** from the 10 banked checkpoints |
| P2 broad  | (0.48, 0.90)     | **computed now** from the 10 banked checkpoints |
| CH narrow | (0.4865, 0.5171) | family 1 `P3_negative_control/RESULT.json`, 10-seed median |
| CH broad  | (0.48, 0.60)     | family 1 `P3_negative_control/RESULT.json`, 10-seed median |

**P2's `R` is a new number and is declared here before it is read.** Family 1 stopped P2
at gate G2 (`dynamic_range = 30.8 < 100`) and never read its `e_off`; its 20 fitted
models are checkpointed. Reading `e_off` now **does not and cannot change family 1's P2
verdict**, which stays FAIL on G2, and nothing is written into `family1_second_pairs/`.
The G2 caveat travels with the number everywhere it is reported: P2's `R` is a
poorly-resolved ratio and is flagged as such in every table.

`R` per cell = the median over that cell's seeds of `e_off / e_on`, with `e_on` on the
narrow on-support cohort (`MASTER_SEED + 7`) and `e_off` on the cell's switch cohort —
the package's definition, unchanged.

### Gate PRED — all three must hold

| # | gate | threshold |
|---|---|---|
| PRED-A | Spearman rank correlation between `log10 R` and `D_bar` across the six cells | `rho >= 0.8` (positive) |
| PRED-B | the Cahn-Hilliard **narrow** cell's switch states are OUTSIDE the PCA-99 radius | `coverage < 0.5` |
| PRED-C | the P1 **broad** and P2 **broad** cells' switch states are INSIDE | `coverage >= 0.5` for both |

PRED-B binds on the narrow Cahn-Hilliard cell because that is the cell the negative
control is about: the model whose `R = 46.3` at zero mean-support distance is the fact
this family has to explain. The Cahn-Hilliard **broad** cell's coverage is reported next
to it and is not gated.

**If the Cahn-Hilliard switch states come out inside the radius, S has failed and that is
the reported result.** No statistic is swapped in afterwards.

## 4. Repair test REPAIR (the intervention — 20 new fits)

### 4.1 The BROAD-S sampler, fixed here before any `R` is read

For a pair with second operator `O2`, at **fixed `n = 640`**:

- Let `base_i`, `i = 0..639`, be `tp.initial_conditions(640, (0.4865, 0.5171),
  MASTER_SEED + 1)` — byte-identical to the narrow training ICs. **The evaluation draws
  `MASTER_SEED + 7` (on-support cohort) and `MASTER_SEED + 8` (switch cohort) are not
  used anywhere in the construction.**
- Fixed burst schedule, four equal index blocks of 160:

  | index block | burst length `m` (steps of `O2`) |
  |---|---|
  | `0..159`   | 0 (the narrow IC, unchanged) |
  | `160..319` | 25 |
  | `320..479` | 50 |
  | `480..639` | 100 |

- `BROAD-S IC_i = O2^m(base_i)`.
- The 640 BROAD-S ICs are then rolled **100 steps under diffusion**, exactly as the
  package builds any training corpus, giving `(640, 101, 1, 256)`; `rb.make_pairs`
  unchanged.

Everything else is identical to the package: architecture A1 (FNO-1d, modes 16, width 64,
4 layers, 549 569 parameters), hand-written Adam, `STEPS = 8000`, `BATCH = 64`, the same
cosine schedule, float32, seeds `0..9` with `jr.PRNGKey(seed)` initialising and
`jr.PRNGKey(seed + 9999)` driving batches, the same on-support and switch cohorts.
**Only which states the training cloud contains moves** — the package's own "sample count
is fixed; only support moves" discipline, transplanted from the spatial mean to the
state space. The `m = 0` block keeps the on-support region represented, so `e_on`, G2 and
P6 stay meaningful.

`m = 100` matches the switch cohort's leg length; `m = 25, 50` are on the path to it, so
the corpus covers the route as well as the endpoint. The burst uses the *operator*, which
is the same knowledge the package's mean-broadening uses when it sets the broad upper
bound from the switch-state means — it does not use the switch states themselves.

### 4.2 Conditions fitted

- **CH BROAD-S** — `O2 = Cahn-Hilliard` (Exponax 0.2.0 defaults, as family 1 vendored
  them), 10 seeds. This is the intervention arm.
- **P1 BROAD-S** — `O2 = Fisher-KPP` (`tp.build_steppers()[1]`), 10 seeds. This is the
  **positive control**: BROAD-S must not do worse on P1 than the package's own
  mean-broadening did.

No other fits are run in this family. 20 fits total.

### 4.3 Comparators (all already on disk; no refit)

The narrow-condition model is the **same model for every pair** — the narrow training
corpus and the on-support cohort do not depend on the second operator, verified
byte-identical in family 1 (`reuse_narrow.py`) and re-verified here by checksum. So:

- `R_narrow(CH)` = family 1's banked ten-seed median (`P3_negative_control/RESULT.json`).
- `R_narrow(P1)` = computed now by evaluating the ten banked narrow checkpoints against
  the Fisher-KPP switch cohort — same harness, same seeds, same cohorts as the BROAD-S
  arm, so the ratio is matched. The package's five-seed `110.234` is reported beside it.

### 4.4 Gate REPAIR — all five must hold

| # | gate | threshold |
|---|---|---|
| REPAIR-1 | CH: `median R_broadS / median R_narrow` | `<= 0.25` (the package's P3 limit) |
| REPAIR-2 | CH: dynamic range `persistence_off / median e_on(narrow)` | `>= 100` |
| REPAIR-3 | P6 convergence: worst `e_on` over all BROAD-S seeds, both arms | `<= 1.573e-3` |
| REPAIR-4 | CH: coverage achieved, fraction of the 256 switch states inside the PCA-99 radius of the **BROAD-S** training cloud | `>= 0.9` |
| REPAIR-5 | P1 positive control: `median R_broadS / median R_narrow` | `<= 0.25` |

Coverage achieved for the P1 BROAD-S corpus is reported; it is not gated (REPAIR-4 binds
on the intervention arm).

**Ordering device, preserved from `run_baseline.py`:** every BROAD-S seed trains, writes
its `e_on` to a `*_pregate.json` and checkpoints its weights; the gate file REPAIR-2 /
REPAIR-3 is written from those; only then is `e_off` read in a second pass.

## 5. Family verdict

- **PASS** iff PRED and REPAIR both hold.
- Otherwise **PARTIAL** (one half holds) or **FAIL** (neither), in both cases naming the
  **binding gate** — the specific numbered gate that failed and its number.

## 6. Seeds, environment, determinism

Seeds `0..9`, ten per BROAD-S condition. Environment
`.venv`; every process launched with
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` after `source environment setup script`.
Single-threaded JAX processes are deterministic regardless of what else runs on the box;
the CPU is shared with two other lanes, so concurrency is held to 4–5 workers and that
changes wall time only, never a number. Fits are launched detached (`nohup`), one log per
worker under `logs/`, and polled with until-loops.

## 7. What this family does not test

One architecture (A1 FNO), one resolution (256), one horizon (`k = 10`, `tau = 1.00` per
leg), one IC family (the package's fitted substitute — every number inherits that
caveat). Seeds are not independent units. Six cells is a small `n` for a rank
correlation: PRED-A on six points can reach `rho = 0.943` with one adjacent swap, so it
is a coarse test and is reported as one. Nothing here addresses the endpoint/trajectory
dissociation lane.
