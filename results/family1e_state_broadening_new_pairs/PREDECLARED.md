# PREDECLARED — Family 1e: state-space broadening (BROAD-S) on the two new pairs

Written **2026-09-11T21:12Z**, before any number belonging to this family was computed:
before any BROAD-S corpus for `S1_fkpp_r0.5` or `S1_grayscott_1sp` existed, before any
PCA basis was fitted on one, before any coverage statistic was evaluated, and before any
model was fitted here. The sha256 of this file is taken at the same moment and stored in
`results/PREDECLARED.sha256`.

Release rules in force: predeclare everything; **never change a gate after seeing a
number**; preserve failures as-is; report the honest number first.
`dependency`, `release-inputs/dependency`,
`release-inputs/family1_second_pairs`, `release-inputs/family1b_pair_screen`,
`release-inputs/family1_rerun`, `release-inputs/family1c_state_support`,
`release-inputs/family1d_predictor`, `release-inputs/family2_*` and `release-inputs/family4_*`
are **read-only** — imported from and read, never written to. All output of this family
goes to `release-inputs/family1e_state_broadening_new_pairs/` and
`work/family1e/`.

---

## 0. Why this family exists

The family-1 rerun fitted the two pre-screened pairs with the package's own
**mean-broadening** intervention at fixed `n = 640` and got, on ten seeds:

| pair | median `R_narrow` | median `R_broad-mean` | ratio | P3 (≤ 0.25) |
|---|---|---|---|---|
| `S1_fkpp_r0.5`     | 58.676  | 16.415 | 0.280 | FAIL |
| `S1_grayscott_1sp` | 105.598 | 28.551 | 0.270 | FAIL |

Both *just* miss the predeclared 0.25 factor. Family 1c then showed that broadening the
dictionary in **state space** (BROAD-S: sample the training ICs so the training-state
cloud covers the switch states, at the same fixed `n = 640`) repairs the same kind of
degradation by a far larger margin — ratio **0.0170** on diffusion/Cahn-Hilliard and
**0.0223** on diffusion/Fisher-KPP (P1), against the same 0.25 limit.

This family asks the generality question: **does BROAD-S, transplanted without change to
the two new pairs, clear the same predeclared 0.25 gate that mean-broadening missed?**
It is the generality result for CMAME if it passes.

Nothing about the intervention is re-tuned here. The sampler is family 1c's, verbatim.

---

## 1. Pairs, operators, cohorts

Two pairs, imported from `family1_rerun/run_family1_rerun.py:second_operator`, which is
itself verbatim from `family1b_pair_screen/screen_pairs.py`:

- **`S1_fkpp_r0.5`** — first operator diffusion (`tp.build_steppers(tp.DT)[0]`), second
  operator `ex.stepper.reaction.FisherKPP(..., diffusivity=0.0, reactivity=0.5)`.
- **`S1_grayscott_1sp`** — second operator the screen's `GeneralPolynomialStepper` form of
  `u_t = F(1-u) - u(1-u)^2`, `F = 0.04`: `linear_coefficients=(-1.04, 0.0, 0.0)`,
  `polynomial_coefficients=(0.04, 0.0, 2.0, -1.0)`.

`MASTER_SEED = 20260905`. Cohorts exactly as in family 1 and the rerun, byte-identical:

- **on-support cohort** — `tp.initial_conditions(256, NARROW, MASTER_SEED + 7)`, frame 0,
  target 10 diffusion steps later;
- **switch cohort** — `tp.initial_conditions(256, NARROW, MASTER_SEED + 8)` evolved 100
  steps under the pair's second operator, frame 100; target 10 diffusion steps later;
- `NARROW = (0.4865, 0.5171)`.

`e_on` / `e_off` = median relative L2 over the 256 units of each cohort;
`R = e_off / e_on` per seed. Estimand unchanged: the `k = 10` diffusion solution
operator, `tau = 0.10` per prediction step.

## 2. The BROAD-S sampler — family 1c `PREDECLARED.md` §4.1, verbatim, fixed here before any fit

For a pair with second operator `O2`, at **fixed `n = 640`**:

- `base_i`, `i = 0..639` = `tp.initial_conditions(640, (0.4865, 0.5171), MASTER_SEED + 1)`
  — byte-identical to the narrow training ICs. **The evaluation draws `MASTER_SEED + 7`
  and `MASTER_SEED + 8` are not used anywhere in the construction.**
- Fixed burst schedule, four equal index blocks of 160:

  | index block | burst length `m` (steps of `O2`) |
  |---|---|
  | `0..159`   | 0 (the narrow IC, unchanged) |
  | `160..319` | 25 |
  | `320..479` | 50 |
  | `480..639` | 100 |

- `BROAD-S IC_i = O2^m(base_i)`.
- The 640 BROAD-S ICs are rolled **100 steps under diffusion**, exactly as the package
  builds any training corpus, giving `(640, 101, 1, 256)`; `rb.make_pairs` unchanged
  (frames `0..90` as inputs, `10..100` as targets).

Everything else is the package's: architecture A1 (FNO-1d, modes 16, width 64, 4 layers,
549 569 parameters), hand-written Adam, `STEPS = 8000`, `BATCH = 64`, the same cosine
schedule, float32, seeds `0..9` with `jr.PRNGKey(seed)` initialising and
`jr.PRNGKey(seed + 9999)` driving batches. **Only which states the training cloud
contains moves** — `n` is fixed at 640 in every condition. The `m = 0` block keeps the
on-support region represented, so `e_on`, G2 and P6 stay meaningful.

The burst uses the *operator*, which is the same knowledge the package's mean-broadening
uses when it sets the broad interval from the switch-state means; it does not use the
switch states themselves.

## 3. What is fitted, and what is reused

- **20 new fits: 10 seeds `0..9` × 2 pairs, BROAD-S only.** No other fit is run in this
  family.
- **The narrow arm is NOT refitted.** `R_narrow` per seed and its ten-seed median are
  read from `family1_rerun/results/<pair>/RESULT.json` (`per_seed.narrow_<s>.R`), which
  was produced by the same harness, the same seeds and the same cohorts. Medians, fixed
  here before any BROAD-S number exists: `S1_fkpp_r0.5` **58.67609512572204**,
  `S1_grayscott_1sp` **105.59814825308872**.
- **The mean-broadening arm is NOT refitted.** Its ten-seed medians, also fixed here:
  `S1_fkpp_r0.5` **16.41465500677668**, `S1_grayscott_1sp` **28.55138746656889**.

## 4. Coverage — recorded for the record, **not gated**

Two coverage statistics are computed on truth states only (no learner), for each pair's
narrow cloud and BROAD-S cloud:

- **Coverage in the PCA-99 basis (family 1c's statistic S).** PCA on the cloud's
  `640 × 101 = 64 640` training states in `R^256` (float64); `k` = smallest number of
  leading components with cumulative explained variance `>= 0.99`; Mahalanobis distance
  `MD(x) = sqrt(sum_{j<=k} <u_j, x-mu>^2 / lambda_j)`; radius `r99` = 99th percentile of
  `MD` over the training states; **coverage = fraction of the 256 switch states with
  `MD <= r99`**.
- **Residual coverage `C3` (family 1d's statistic S3).** `Q(x) = ||(x-mu) - U U^T (x-mu)||_2`
  with the same `U` (unnormalised, mean-centred); `q99` = 99th percentile of `Q` over the
  cloud's training states; **`C3` = fraction of the 256 switch states with `Q <= q99`**.
  `D3` = mean `Q` over the switch states is reported alongside.

**Neither is a gate.** Family 1c gated on the PCA-99 coverage (its REPAIR-4, `>= 0.9`)
and that gate failed *while the repair succeeded by a factor of 59*, because a Mahalanobis
radius over a PCA-99 basis is not monotone under the intervention it is supposed to
measure. Re-using a statistic already shown non-monotone as a gate would be gating on a
known-broken instrument. The task spec for this family accordingly lists the gates as
"copied from family 1" — G1, G2, G4, P6, P3, P4 — and asks for the coverages "for the
record". They are reported in full, including when they embarrass the statistic.

## 5. Gates — copied from family 1 / the rerun, verbatim

| # | gate | definition | threshold |
|---|---|---|---|
| **G1** | shift exists | narrow-training-mean support vs the pair's switch-state support | `overlap == 0.0` — **already established per pair** in `family1_rerun/results/<pair>/GATES.json` (fkpp gap `+0.0771`, grayscott gap `-0.1268`); re-verified here by recomputation, not re-derived |
| **G2** | metric resolves | `dynamic_range = persistence_off / median e_on(narrow)` **and** `median e_on(narrow) <= 0.1 * persistence_on` | `>= 100.0`, and the `0.1` fraction |
| **G4a** | censored below | `min e_on` over the ten narrow seeds (from the rerun) **and** over the ten BROAD-S seeds | `>= 1.19e-5` |
| **G4b** | censored above | `max e_off` over the BROAD-S seeds | `< persistence_off` |
| **P6** | convergence | worst `e_on` over all BROAD-S seeds of both pairs | `<= 1.573e-3` |
| **P3** | broadening repairs | `median R_broadS / median R_narrow`, with `median R_narrow` the rerun's | `<= 0.25` |
| **P4** | constant mode | reported, not gated: median constant-mode-corrected `e_off` over the BROAD-S seeds against median `e_off`; the rerun's narrow-arm P4 is quoted beside it | reported |

`persistence_off` / `persistence_on` are the identity predictor's median relative L2 on
the switch / on-support cohorts, recomputed here; they are truth-only numbers and are
identical by construction to the rerun's.

**Ordering device, preserved from `run_baseline.py` and family 1c.** Every BROAD-S seed
trains, writes its `e_on` to a `*_pregate.json` and checkpoints its weights.
`results/broadS/GATES.json` carrying G1, G2, G4a and P6 is written **from those alone,
before any `e_off` is read**. Only then does the eval pass read `e_off`. If G1, G2, G4a
or P6 fails for a pair, the run stops for that pair **before** `e_off` is read and the
pair is reported FAIL on that gate.

**Additionally reported, not gated:** `R_broadS / R_broad-mean`, the ratio of this
family's median `R` to the rerun's mean-broadening median `R`, per pair.

## 6. Family verdict

- **PASS** iff **P3 holds on both pairs** (`median R_broadS <= 0.25 * median R_narrow`
  for `S1_fkpp_r0.5` **and** `S1_grayscott_1sp`), with G1, G2, G4 and P6 holding on both.
- **PARTIAL** if P3 holds on exactly one pair.
- **FAIL** otherwise, naming the binding gate.

A pair's verdict is **PASS** iff G1, G2 and P3 all hold for it — the rerun's rule,
unchanged.

## 7. Seeds, environment, determinism

Seeds `0..9`, ten per pair, 20 fits total. Environment
`.venv/bin/python`; every process launched with
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` after `source environment setup script`, with
`JAX_COMPILATION_CACHE_DIR=work/family1e/jax_cache` overriding the shared
cache (family 1c §4.5: the shared cache holds AOT entries from another lane that XLA
refuses with a SIGILL warning). Single-threaded JAX processes are deterministic
regardless of what else runs on the box; the CPU is shared with other lanes, so
concurrency is capped and that changes wall time only, never a number. Fits are launched
detached (`nohup`), one log per worker under `logs/`, and polled with until-loops.

## 8. What this family does not test

One architecture (A1 FNO), one resolution (256), one horizon (`k = 10`, `tau = 1.00` per
leg), one IC family (the package's fitted substitute — every number inherits that
caveat). Seeds are not independent units; ten seeds of one pair are not ten pairs. Two
pairs are not a survey of operators. BROAD-S uses knowledge of the second operator, so it
answers "can a dictionary that contains states of the kind the model will meet repair the
degradation", not "can you build such a dictionary without knowing the shift". Nothing
here rescues family 1c's PRED half: no statistic tested so far predicts which pairs need
the intervention.
