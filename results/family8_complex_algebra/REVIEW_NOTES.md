# REVIEW_NOTES — Family 8 runner review before the GPU run

Reviewer: operator of the family-8 GPU lane, 2026-09-11/12.
Reviewed: `run_family8.py`, `launch_driver.sh` against `TASK.md`, `PREDECLARED.md`
(sha256 `0e7ac1a1…d128d80d`, unchanged), family 1e's BROAD-S sampler and gate
definitions, family 6's `METHOD_SPEC.md`, family 7's `POD_RUNBOOK.md` (all read-only).

Nothing in `PREDECLARED.md` was edited. Every change below was made **before** any
off-support number for this family existed. Three of them change how a predeclared
quantity is computed (items 1, 6, 7) and two change how a predeclared gate is
aggregated or scored (items 3, 6); each is stated here with its reason, and the
quantity the original code would have produced is reported alongside the corrected
one in `RESULT.json` wherever that is possible.

## What was correct and was left alone
- The strict ordering device. `launch_driver.sh` runs prescreen → 40 fits (e_on only)
  → `gates` → `words` → `algebra` → `external` → `evaluate`, and `evaluate` refuses to
  run when `GATES.json` lists a blocking failure. No off-support number is computed
  before the gates are written. Unchanged.
- `train` writes `e_on` only, never `e_off`; per-fit checkpoint + pregate file; the
  driver is resumable per fit and throttles to one fit at a time. Unchanged.
- Sample budget: exactly `n = 640` trajectories per member in both conditions, 100
  steps, pairs at stride `K = 10`. The BROAD-S corpus changes *which* states are in
  the cloud, never how many. Verified, unchanged.
- FNO-2D architecture, optimizer, schedule, 8000 steps, batch 64: identical to
  family 7 (2,365,825 parameters). Unchanged.
- The narrow IC window `(0.4865, 0.5171)`, amplitude 0.175, cutoff 5, MASTER_SEED
  20260905 and the `+1 / +7 / +88 / +99` seed offsets match family 1e's discipline and
  keep training, on-support, E1 and E2 cohorts disjoint. Unchanged.
- Checkpoint save/reload is exact (smoke receipt, and re-verified by the tiny
  end-to-end run below).

## Defects fixed

**1. `e_on` was measured against the wrong operator (blocking defect).**
`eval_cohorts_2d` built the on-support target with the **diffusion** stepper for every
dictionary member, a mis-transplant from family 1e where diffusion is the only
estimand. In family 8 each member has its own surrogate, so for A, R and G the
reported `e_on` was the error of that member's surrogate against diffusion's flow map
— a number with no meaning, large enough to fail the imported `P6` constant and block
`evaluate` outright. Fixed: `eval_cohorts_2d(member, …)` now uses the member's own
K-step flow map on the same held-out narrow ICs (family 1e's cohort construction,
frame 0, target K steps later). `train`, `gates` and the smoke were updated.

**2. The pre-screen ran at the wrong resolution.** `prescreen()` defaulted to
`SMOKE_GRID = 32`, and the committed `PRESCREEN.json` (7 of 12 pairs surviving) was a
32×32 screen, while the experiment is 64×64. The surviving-word list decides the word
suite, the BROAD-S corpus and the E2 truth set, so it has to be screened at the
experiment's resolution. Fixed: default is `FULL_GRID`, `prescreen [num_points]` takes
an override, and the two hard-coded `32`s in the PCA block now follow the cohort size.
The 32×32 file is kept as `results/PRESCREEN_grid32_superseded.json` and the 64×64
screen is re-run on the pod.

**3. The algebra search was not an honest search over words of length ≤ 3.**
Three separate problems in `algebra()`:
   - *Arbitrary length penalty.* Candidates whose horizon differed from the truth's
     were scored on the overlapping frames and then charged `0.5 × |Δframes|`, i.e. +5
     on a scale where real losses are O(0.1). That penalty alone eliminated all 68
     candidates of the wrong length, so the search was handed the true word length and
     "top-1 over 84 candidates" was really "top-1 over 16". Fixed with a rule that has
     no tunable constant: every candidate is scored on the full observed horizon; a
     candidate that ends early is **held at its final state** (it asserts nothing more
     happens); a candidate that runs past the horizon is truncated at it; exact ties —
     which occur because a longer word sharing the whole observed prefix produces
     bit-identical frames — resolve to the **shorter** word (Occam). Scoring is the
     predeclared `L = (1/T) Σ_t ‖pred_t − true_t‖ / ‖true_t‖` over frames 1..T.
   - *Order accuracy was a duplicate of top-1.* It was computed as
     `parse_word(top1) == true_ops`, which is identical to `top1 == w_true`. The
     predeclaration asks for "correct sequence order **among matched operators**".
     Fixed: the multiset-match rate is reported, and order accuracy is the fraction of
     multiset-matched trajectories whose order is also right, with its denominator.
   - *Cost.* The search rolled every candidate separately for every held-out
     trajectory (batch 1, un-jitted): ~672,000 single-sample FNO forwards, ≈2 h of GPU
     on its own, more than the whole budget. All truth words share the same 20 held-out
     ICs and a candidate rollout does not depend on the truth word, so each candidate
     is now rolled **once** per (condition, seed) over all 20 ICs at once and scored
     against every truth word. Identical arithmetic, ~2,500 batched forwards per
     dictionary instead of ~67,000 single ones.

**4. E2 used only seed 0.** `algebra()` loaded `*_seed0.npz` and the ALG-1 gate was a
single-seed number, while the family declares 5 seeds. With item 3 the search is cheap
enough to run all five. ALG-1 is now evaluated on the **median over the 5 seeds**, and
the seed-0 values the original code would have used are recorded next to it in
`ALGEBRA.json` and `RESULT.json`. Declared here before any fit was run; no seed
selection is possible after the fact.

**5. The E3 "Serrano" arm was a copy of the narrow arm.** The splitting block applied
the narrow surrogates in exactly the word order, with exactly the same step counts as
the narrow composed rollout — identical code, so it could only ever reproduce that
number. Family 6's `METHOD_SPEC.md` defines test-time operator splitting as Lie
(`f₂∘f₁`) or Strang (`f₁^{½}∘f₂∘f₁^{½}`) recompositions evaluated **at test time only**,
granted extra forward passes but no extra training. Fixed: for each length-3 word the
narrow dictionary is given a test-time search over three schemes that all spend 30
surrogate applications and advance the same total time — `leg_sequential_lie` (the word
order itself), `lie_fine` (the three operators interleaved ten times) and `strang_fine`
(the symmetric palindrome five times) — and the arm is scored at its **best** scheme.
The correct ordering is inside the search space, so this is a strictly stronger
baseline than the narrow arm, which is the point: the comparison is whether test-time
splitting of narrow operators can reach BROAD-S's composed endpoint. Described as
test-time operator splitting throughout; the words Lie and Strang label the
compositions, and are never attributed to Serrano et al.

**6. Two blocking gate constants were 1-D imports.** `G4_MIN_E_ON = 1.19e-5` and
`P6_MAX_E_ON = 1.573e-3` are absolute on-support errors carried over from the 1-D
families (1-D FNO, 256 points); they appear nowhere in `PREDECLARED.md`, which declares
only P3 and ALG-1 as gates and names G1/G2/G4a/P6 without numbers. Applying an absolute
1-D error constant as a blocking gate on 2-D fits would have decided the run on an
uncalibrated threshold. Fixed: G4a and P6 are computed and reported in `GATES.json` as
**non-blocking reference diagnostics**, each with whether it meets the 1-D constant, so
a failure is visible and not hidden. Blocking is now G1 (a support shift exists) and G2
(the metric resolves), and G2 is evaluated **per member** — each member's median narrow
`e_on` must be ≤ 10% of how far that member's own operator moves the state over the
estimand stride — which is the resolution-independent form of the same requirement.

**7. P3 dropped the `e_on` normalisation.** `evaluate()` formed the repair ratio from
raw composed-endpoint errors, but E1 declares `R = e_off / e_on` and
`P3 = median(R_broadS) / median(R_narrow)`. Fixed: `R` is the composed-endpoint error
divided by the median `e_on` of the word's own members for that condition and seed.
This is the conservative direction — BROAD-S's `e_on` is the larger one, so normalising
raises the ratio and makes P3 harder to pass. Both the normalised ratio (primary, as
declared) and the unnormalised one (what the original code computed) are reported.

**8. Budget/throughput defects.** (a) The BROAD-S union corpus was rebuilt from scratch
inside all 20 broad fits, although it is a deterministic function of the word list and
identical for every member and seed; it is now built once and cached to
`$FAMILY8_WORK_DIR/corpus/`. (b) The switch pool was subsampled **with** replacement;
the pool holds ~1700 states for a 480-state draw, so it is now without replacement
whenever the pool allows, with the old behaviour kept as the fallback. (c) Truth
rollouts and surrogate steps re-entered `jax.jit` on a freshly constructed wrapper at
every call, recompiling constantly; both now go through module-level `eqx.filter_jit`
helpers that compile once per batch shape and reuse the code across members, seeds and
conditions (weights and stepper state enter as traced arguments). (d) `words()` reloaded
checkpoints and recomputed the truth trajectory inside the seed loop; both are hoisted.
None of these change any computed value.

## Checked and deliberately not changed
- **BROAD-S is not family 1e's sampler verbatim.** 1e bursts each base IC forward
  0/25/50/100 steps under the pair's second operator in four equal blocks of 160.
  `TASK.md` asks instead for "the family 1e sampler over the **union of switch states
  produced by ALL words in the list**, n=640 fixed", which is what the runner does
  (160 base ICs + 480 drawn from the union of leg-boundary and mid-leg states across
  every tested word). The budget is conserved and the family-8 form is the one the task
  declares, so it stands.
- The pre-screen's PCA-99 support test is asymmetric between `DoA` and `AoD` (D's
  endpoint cloud is low-rank, so A's stirred states fall outside it, while A's
  higher-rank cloud contains D's smooth ones). That asymmetry is a real property of the
  operators, not a bug; `AoD` failing G1 while `DoA` passes is recorded, not repaired.
- `G` (Gray-Scott 1-species) survived the 32×32 screen comfortably (dynamic range
  ≥ 600), so the Allen-Cahn fallback is not triggered. Re-checked at 64×64 on the pod.

**9. Two defects found while the run was in flight** (both logged here at the moment
they were made, both before any off-support number existed).
   - *Non-deterministic cache key.* The BROAD-S corpus cache from item 8 keyed its
     filename on `hash(tag)`, and Python randomises string hashing per process, so
     every fit missed the cache and rebuilt the corpus. Switched to
     `hashlib.sha256(tag)`. The corpus contents were identical either way — the bug
     cost time, not correctness.
   - *Fit order.* The driver looped condition-major, so an interrupted run would have
     left a complete NARROW arm and no BROAD-S arm. Reordered to seed-major: each seed
     completes both conditions across all four members before the next seed starts, so
     an interrupted run still leaves a matched experiment. Fits are independent (own
     corpus, own PRNG streams), so the order changes no computed value.

## Predeclared seed reduction, applied 2026-09-12 07:05Z
Measured per-fit wall time on the RTX A6000 was 3.6 min (NARROW) and 5.5 min (BROAD-S)
against the runbook's 25-35 s estimate for a 4090, which projected the 5-seed design at
about 3.3 h and $1.77 — at the 3.5 h hard stop for this pod and at the $1.75 figure that
`PREDECLARED.md` section 8 attaches to its reduction trigger. The predeclared reduction
was therefore applied: **seeds 5 -> 3 (seeds 0, 1, 2)**, for both conditions and all four
members, giving 24 fits. This was decided and recorded before `gates`, `words`,
`algebra` or `evaluate` had been run, so no off-support number existed and none could
have influenced it. Five NARROW fits of member D (seeds 0-4) had already completed
before the reduction; seeds 3 and 4 of that member are kept on disk as a record but are
excluded from every aggregation, which uses seeds 0, 1, 2 for every member and both
conditions. `FAMILY8_SEEDS` carries the reduced set into the aggregation stages.

## Verification of the patched runner before spending GPU time
A tiny end-to-end run on CPU (16×16, 1 seed, 40 training steps, n=32, 4 held-out
trajectories) executed `prescreen → 8 fits → gates → words → algebra → external →
evaluate` and exercised every code path that the pod will run, including the corpus
cache, the splitting schemes, the 84-candidate search and `RESULT.json` assembly.
Result recorded in `results/E2E_TINY.json`. The numbers from that run are meaningless
(40 training steps) and are used only as a smoke test of the plumbing.
