# RESULT_SMOKE — family 6, Serrano et al. test-time operator splitting as a same-budget baseline

Smoke run, 2026-09-11. Everything reported here was fixed in `METHOD_SPEC.md` **before**
any model in this directory was fitted. **This task declares no winner.** It builds the
baseline, proves the pipeline end to end, and reports both conditions on the same test
units. The head-to-head verdict is a later task.

Run: P1 = diffusion(0.01) / Fisher-KPP(reactivity 1.0, diffusivity 0), 1-D periodic,
N = 256, dt = 0.01, tau = K·dt = 0.10, 10 applications per leg. **2 seeds (0, 1),
48 test units**, 640 training units per primitive, A1 FNO (549 569 parameters),
8 000 Adam steps per fit. CPU, threads pinned.

> **Two seeds.** Every "range" below is a range of two numbers. No dispersion claim, no
> significance claim and no seed-to-seed inference is made from it (METHOD_SPEC §7.6).

---

## 1. Implementation gates — PASS

| gate | result |
|---|---|
| **U1** Lie, native-grain Strang and 2τ-grain Strang exact on a commuting pair (two pure-diffusion Fourier multipliers, D₁ = 0.004, D₂ = 0.006, float64) | **PASS.** Worst relative L2 over six splitting forms **2.67e-16**; range 1.49e-16 … 2.67e-16. Floor demanded: 1e-10. This is machine epsilon — the splitting code adds nothing above round-off. |
| **U2** the search recovers the true composition from a context the combined operator generated | **PASS.** Exhaustive, beam (B = 4) and uniform (T = 100) all select a length-2 composition containing both operators for every unit; worst selected L(S) = **1.06e-16**. |
| **G1** zero support overlap | **PASS**, both seeds. Diffusion training frames [0.48658, 0.51700]; switch states [0.69423, 0.72072]; overlap 0, gap **0.1772**. |
| **G2** dynamic range ≥ 100 | **PASS**, both seeds: **1588**, **1312**. |
| **G4** e_on above the float32 floor (1.19e-5) | **PASS**: min e_on 5.10e-5. |
| **P6** convergence, worst e_on ≤ 1.573e-3 | **PASS**: worst e_on 7.44e-5. |
| **S1** search sanity: on the on-support causal context the search picks a diffusion-only composition for ≥ 90 % of units | **PASS**: **48/48 = 1.00**, both seeds, all three selection rules. |

Also recorded: the compiled evaluation forward pass (`FNO_FORWARD = jax.jit(tp.fno_apply)`)
agrees with the package's uncompiled `tp.fno_apply` to **2.92e-07** max relative
difference — five orders of magnitude below the smallest quantity reported here.

`results/SELFTEST.json`, `results/seed0.json`, `results/seed1.json`,
`results/RESULT_SMOKE.json`.

---

## 2. R — ten-step off-support degradation ratio

Estimand: the K = 10 diffusion solution operator, τ = 0.10. Evaluation states are at
**frame 10 (t = 1.0)** for both on- and off-support, because the baseline needs a
context window and `run_baseline.py`'s frame-0 evaluation point has no history
(METHOD_SPEC §5.1). **All arms are evaluated on the same states.**

| arm | context | e_on (median) | e_off (median) | **R** | per-seed R |
|---|---|---|---|---|---|
| `narrow_oracle` — package baseline | none, told the primitive | 6.34e-05 | 2.237e-02 | **354.9** | 366.6, 343.3 |
| `broad_oracle` — **our condition** | none, told the primitive | 5.12e-05 | 5.566e-03 | **108.7** | 116.0, 101.4 |
| `serrano_narrow_lie` | **causal** (real history) | 6.34e-05 | 8.056e-02 | **1310.7** | 1539.1, 1082.3 |
| `serrano_narrow_lie` | **regime-matched [ORACLE]** | 6.34e-05 | 2.199e-02 | **349.8** | 366.6, 333.0 |
| `serrano_broad_lie` | causal | 5.12e-05 | 8.055e-02 | **1573.8** | 1569.4, 1578.1 |

Exhaustive (all 62 candidates), beam (B = 4) and uniform (T = 100) selected **identically
in every cell** — with a 2-operator dictionary and M = 5 the whole space is 62 ordered
multisets, so exhaustive dominates and the search-budget question cannot be raised.

Persistence references on the same states: on-support 5.72e-03, off-support 8.23e-02.

**What the search actually selected** (48 units, exhaustive):

- on-support, causal context → `D` for 48/48 units, both seeds. The search works.
- off-support, **causal** context → `R` (reaction) for 48/48 units, both seeds. The real
  history preceding a switch state *is* the reaction leg, so the method identifies
  reaction and then predicts a diffusion step with a reaction operator. e_off (8.06e-02)
  is essentially the off-support persistence error (8.23e-02): the prediction carries no
  usable information about the estimand.
- off-support, **regime-matched oracle** context → seed 0: `D` for 48/48, giving R
  **identical to the narrow oracle to six significant figures** (366.581703). Seed 1:
  `D-R` 19 units, `R-D` 17, `D` 12, giving R = 333.0 against the narrow oracle's 343.3 —
  a **3.0 % improvement**.

### Reading (honest, no winner declared)

Under this task's equal-budget contract, on these test units:

- **Test-time compositional search over a fixed, narrow, single-physics dictionary buys
  essentially nothing on R.** Handed an oracle context that tells it the test dynamics
  *and* an exhaustive sweep of all 62 compositions, it moves R from 354.9 to 349.8
  (median), i.e. **1.4 %**, best seed 3.0 %.
- **Broadening the training dictionary at the same 640 samples per primitive moves R from
  354.9 to 108.7, a 69 % reduction**, with zero test-time search.
- Deployed causally — the only way it can actually be run, with the history that really
  precedes a switch state — the baseline is **3.7× worse than the narrow operator it is
  built from**, because the two-leg corpus is not time-homogeneous and a single searched
  composition must be wrong on at least one leg (METHOD_SPEC §2, predeclared).
- The two are not substitutes and, on this evidence, not obviously complementary either:
  `serrano_broad_lie` (search on top of the broadened dictionary) is the worst R in the
  table. The broad reaction operator is a *better* reaction operator, so when the causal
  context makes the search choose it, it is more confidently wrong.

### Bridge to the package's banked numbers

Because the evaluation state moved from frame 0 to frame 10, the R values above are **not**
comparable to the banked 110.234 / 17.650. Both oracle arms were therefore also evaluated
at frame 0 on the same 48 units:

| condition | frame-0 R here (per seed) | package banked median | package per-seed range |
|---|---|---|---|
| narrow | 79.19, 102.17 (median 90.68) | 110.234 | [71.33, 137.30] |
| broad | 18.31, 18.46 (median 18.38) | 17.650 | [14.00, 18.07] |

Narrow sits inside the package's per-seed range. **Broad sits 1.4 %–2.2 % above the top of
the package's broad range** — reported as found, not adjusted. The cohorts are not the
same draw: `initial_conditions(48, …)` and `initial_conditions(256, …)` split the same
PRNG key into different numbers of per-unit keys, so no unit here is a unit there. Two
seeds against the package's five. Family 1's independent P1 replication of the same
harness landed at narrow 74.99 / 121.38 and broad 17.16 / 18.08 on the package's
256-unit cohort (`family1_second_pairs/results/P1_VALIDATION.json`) — its broad seeds sit
inside the package range, so the 1–2 % excess here is most likely the 48-unit draw.

---

## 3. S — centered order-contrast score

`run_dissociation.py:centred_ratio`, centered over the 48 evaluation units. Trajectory =
all 21 τ-frames; endpoint = frame 20. 48 units, seed MASTER+11.

| arm | trajectory | endpoint | horizon-only (frames 7–20) | even subgrid (traj) |
|---|---|---|---|---|
| `narrow_oracle` (told the order) | **0.0975** [0.1192, 0.0758] | **1.339** [1.937, 0.741] | 0.136 | 0.097 |
| `broad_oracle` — **our condition** | **0.0179** [0.0187, 0.0170] | **0.1125** [0.1031, 0.1220] | 0.0246 | 0.0144 |
| `serrano_narrow_lie_causal` | **0.7191** | **8.370** | **1.0007** | 0.7253 |
| `serrano_narrow_strang2tau_causal` | n/a (2τ grid) | 8.370 | n/a | **0.7253** |
| `serrano_broad_lie_causal` | 0.7184 | 8.363 | 0.9997 | 0.7246 |

Cross-check against the package: the banked five-seed FNO composition
(`DISSOCIATION_RESULT.json`) gives trajectory_ratio 0.0558–0.1207 and endpoint_ratio
0.574–2.321 over its five seeds on 256 units. This run's `narrow_oracle` (0.0758, 0.1192 /
0.741, 1.937) lands inside both, on a different 48-unit cohort. The S implementation reproduces the package.

**Strang is exactly Lie here, to every digit printed.** Predeclared in METHOD_SPEC §2.2:
within a leg only one primitive is active, so there is no splitting error to be second-order
about; the search selects a length-1 composition, and Strang with m = 1 degenerates to the
full step. The Strang arm is reported because a referee will ask, not because it moves.

### Why the Serrano arm scores ~1.0 on the horizon, and what that does and does not mean

S = 1 is exactly what a predictor emitting only the shared mean contrast scores, by
construction (`family3_baselines/PREDECLARED.md` §1(c)). The Serrano arm's horizon S of
1.0007 is **numerically** at that level. It is **not** there for the same reason, and the
report would be wrong to say it is order-blind.

POST HOC diagnostic, added after the smoke result was in hand and labelled as such
(`diagnostic_contrast_magnitude.py` → `results/CONTRAST_MAGNITUDE_DIAGNOSTIC.json`):

| arm, horizon window | ‖d̂_centered‖ / ‖d_centered‖ | corr(d̂_c, d_c) | S |
|---|---|---|---|
| `narrow_oracle` | 1.027, 1.004 | +0.987, +0.995 | 0.166, 0.105 |
| `broad_oracle` | 1.001, 0.999 | +0.9997, +0.9997 | 0.026, 0.024 |
| `serrano_narrow_lie_causal` | **1.531, 1.530** | **+0.764, +0.765** | 1.002, 1.000 |
| `serrano_broad_lie_causal` | 1.530, 1.528 | +0.765, +0.765 | 1.001, 0.999 |

So the baseline emits a centered order contrast **1.53× too large** that is **+0.76
correlated** with the truth — real order information, badly mis-scaled — and
r² − 2r·corr + 1 lands it on 1.00 by arithmetic coincidence. The correct statement is
"*no better than the common-pattern baseline*", not "*order-blind*".

The mechanism is the structural mismatch of METHOD_SPEC §2: the context lies inside leg A,
so for the AB arm the search picks `D` (48/48) and for the BA arm `R` (48/48), and each is
then rolled time-homogeneously for all 20 applications. The predicted contrast is
D²⁰(u₀) − R²⁰(u₀) against a truth of (R∘D)(u₀) − (D∘R)(u₀).

---

## 4. Wall clock

| stage | wall clock |
|---|---|
| `selftest` (U1, U2) | **3.5 s** |
| 8 fits (2 conditions × 2 primitives × 2 seeds), two concurrent | **2 h 25 m 09 s** (05:43:36Z → 08:08:45Z) |
| per fit | 24.4, 24.4, 25.2, 28.2, 45.4, 45.4, 46.2, 46.7 min (sum 4 h 46 m of process time) |
| `evaluate` seed 0 / seed 1 | 6.3 min / 2.4 min |
| post-hoc diagnostic | 57.7 s |
| **total, launch to aggregate** | **2 h 34 m 06 s** (05:43:36Z → 08:17:42Z) |

**This is a contention figure, not a cost figure.** The package's own baseline fits ten
models in 3 079 s (308 s each) on an idle machine. A concurrent family-1 worker was
running up to ten simultaneous fits on the same 10-core CPU throughout; load average ran
140–240 and the per-fit cost here is **5–9× the uncontended cost**. A clean-machine
estimate for the same eight fits is ≈ 41 min of sequential process time, ≈ 21 min at two
concurrent.

One implementation note that matters for the full run: the evaluation sweep makes tens of
thousands of forward passes and the package calls `tp.fno_apply` uncompiled. Wrapping it
in `jax.jit` cut `evaluate` from ~40 min to 2–6 min per seed at identical numerics (2.9e-07,
§1). Training is untouched.

---

## 5. Exact commands for the full 10-seed head-to-head

**Family 1's harness is validated** as of 2026-09-11T05:42:56Z
(`family1_second_pairs/results/P1_VALIDATION.json`, `validation_pass: true`; all four P1
seed R values inside the package's per-seed bands). Its own note records that
"run_baseline.py is complete and runnable as shipped … Nothing in the harness had to be
reconstructed."

`serrano_splitting.py` fits through **the same imported path family 1 validated** —
`run_baseline.train` driving `timing_probe.init_fno / adam_init / train_step` — so that
validation transfers and no second pilot is needed. Nothing is reused by copying; both
import the shipped package.

```sh
cd results/family6_serrano_baseline
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
PY=.venv/bin/python

# 0. implementation gates (seconds, no fitting). Must pass before anything else.
$PY -u serrano_splitting.py selftest

# 1. 40 fits: 10 seeds x {narrow,broad} x {diffusion,reaction}.
#    launch_smoke.sh already does exactly this two-at-a-time and SKIPS any fit whose
#    checkpoint exists, so the smoke's 8 checkpoints are reused and only 32 remain.
sed -i '' 's/for seed in 0 1; do/for seed in 0 1 2 3 4 5 6 7 8 9; do/g' launch_smoke.sh
nohup ./launch_smoke.sh > logs/full10.log 2>&1 &
until grep -q "ALL DONE" logs/full10.log; do sleep 60; done

# 2. post-hoc contrast diagnostic over whatever seeds exist
$PY -u diagnostic_contrast_magnitude.py
```

Or, without editing the launcher:

```sh
for s in 0 1 2 3 4 5 6 7 8 9; do
  for c in narrow broad; do for p in diffusion reaction; do
    $PY -u serrano_splitting.py fit $c $p $s
  done; done
done
for s in 0 1 2 3 4 5 6 7 8 9; do $PY -u serrano_splitting.py evaluate $s; done
$PY -u serrano_splitting.py aggregate
```

**Budget for the full run:** 32 further fits. On an idle machine ≈ 2 h 45 m of sequential
process time, ≈ 1 h 25 m at two concurrent. At the contention actually measured here
(35.7 min per fit at two concurrent) it is ≈ 9 h 30 m. Plus ~10 evaluations at 2–6 min.
`diagnostic_contrast_magnitude.py` currently loops over seeds `(0, 1)`; widen that tuple
for the full run.

**Before the head-to-head is written up, do these three things** (none is a smoke task):

1. Report R at frame 0 *and* frame 10 for every arm, not just the oracle arms, so the
   bridge to 110.234 / 17.650 is complete.
2. Add per-unit paired statistics (the smoke reports medians over 48 units; family 3's
   paired margin + bootstrap CI is the established form in this project).
3. Re-check the equation numbering against the typeset PDF (§6, item 8).

---

## 6. Places where the paper leaves a choice unspecified

All read from the arXiv **HTML v2** rendering. `https://arxiv.org/pdf/2602.00884`
**failed** (`maxContentLength size of 10485760 exceeded`) and the PDF was never read;
`https://arxiv.org/abs/2602.00884` and `https://arxiv.org/html/2602.00884(v2)` succeeded.

| # | What is unspecified | Default chosen — **marked as the most faithful** | Why, and what the alternative would have been |
|---|---|---|---|
| 1 | **Ordering inside a composition.** §4.2 writes the algorithms over *subsets* `S ∪ {f_j}`; nothing in the text says whether permutations are enumerated. | **Search over ordered sequences.** | For noncommuting primitives order changes the result, so searching order is the *generous* reading — it can only help the baseline. Alternative: search unordered subsets and fix a canonical order, which would hand the method an arbitrary and possibly unlucky ordering. |
| 2 | **Strang half-steps for a discrete map.** DISCO operators are continuous-time vector fields, so `f^{Δt/2}` is free; the A1 FNO is a discrete τ-map and has none. The paper never faces this. | **Strang at grain Δt = 2τ**, where the native τ-map *is* the half step and `f∘f` is the full step, so a macro-step is exactly `f₁∘f₂∘f₂∘f₁`. Exact Strang for the operators available, zero extra training. Scored on the 11-frame even subgrid, and every other arm is scored there too. | Alternative A, deferred: fit a second FNO per primitive at stride k = 5. Same simulator calls but **double** the gradient budget, which breaks the equal-budget contract; worth running once as a deliberately more-favourable arm. Alternative B, **rejected**: the increment-scaled half step `u + ½(f(u)−u)` has O(τ²) local error — the same order Strang exists to remove — so it silently degrades Strang to Lie while looking like Strang. |
| 3 | **Context length L.** §3.3 uses L = 16 frames everywhere. Our legs are 10 τ-frames, so 16 cannot fit inside a leg. | **L = 7 frames (6 transitions), horizon 14.** Rule fixed in advance: largest context that fits strictly inside one leg while keeping the paper's context:horizon ratio (16:50 = 0.32 of the frame budget; 7/21 = 0.33). | Alternatives: a longer context straddles the handoff and hands the method the regime switch; a shorter one starves the objective. |
| 4 | **Beam-search minimum relative improvement threshold.** §4.2 says there is one; the value is not given. | **1 %.** | Cannot drive anything here: exhaustive search is threshold-free, is also reported, and selected identically in every cell. |
| 5 | **Dictionary size and construction.** Serrano's N operators come free from one hypernetwork fit (N = 17…256). We have no hypernetwork, and TASK.md §2 scopes the dictionary to the two primitives. | **N = 2** (`f_diff`, `f_reac`), with the search over **ordered multisets up to M = 5** — 62 candidates. Repetition is meaningful and faithful: in splitting each sub-step advances its own field over the full Δt, so `f_diff∘f_diff` is twice the diffusivity, which is the paper's own §5.2 parameter-extrapolation mechanism. | This is the **largest single infidelity** and it is not in the baseline's favour. Sharding the 640 units into more operators would break "same 640 training units per primitive". The right fix, if a referee presses, is to implement a DISCO-style hypernetwork — a different task with a different budget. |
| 6 | **Teacher-forced or autoregressive context objective.** The displayed `L(S)` reads as one-step from each observed `u^t`; the paper does not say it in words. | **Teacher-forced one-step**, which is what the formula says. | Autoregressive-over-the-context would penalise drift and might select differently; untested here. |
| 7 | **Per-test-unit or per-cohort search.** §3.3 speaks of "a test trajectory", singular. | **Per test unit** — 48 independent searches per arm. | The generous reading again: a pooled search could not specialise. |
| 8 | **Equation numbers.** The HTML v2 rendering displays the §3–§4 body equations **unnumbered**; the only numbered equations, (1)–(11), are the Appendix A per-benchmark PDE definitions. | Body results are cited by **section number plus verbatim quotation**, never by an invented equation number. | Re-check against the typeset PDF before the manuscript cites equation numbers. |

### One item closed for the novelty pass

`novelty_web_2026-09-10/REPORT.md` lists as UNRESOLVED whether the paper uses Lie/Strang
terminology verbatim. **It does**, in §4.3: *"For two operators f1+f2, Lie splitting
sequentially applies each operator over the full time step"*; *"Strang splitting uses a
symmetric pattern for higher accuracy"*; *"This palindromic composition preserves the
symmetry required for second-order accuracy and extends classical Strang splitting."* The
manuscript's "operator dictionary + Lie/Strang splitting" framing is supported, subject to
the HTML-vs-PDF caveat above.

---

## 7. What this smoke does not establish

- Nothing about dispersion or significance: two seeds.
- Nothing about pairs other than P1.
- Nothing about a DISCO-scale dictionary (§6 item 5). A 256-entry dictionary spanning a
  range of effective diffusivities might do better on the regime-matched R than the 1.4 %
  seen here. The honest claim from this run is about **a two-primitive dictionary at the
  A1 budget**, which is what TASK.md specifies and what an equal-budget comparison permits.
- Nothing about the native-grain Strang arm (§6 item 2, alternative A).
- No winner. Both conditions are reported on the same 48 test units and stop there.
