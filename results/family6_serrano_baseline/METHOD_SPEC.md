# METHOD_SPEC — Serrano et al. (arXiv:2602.00884) test-time neural operator splitting, as a same-budget baseline

**Status: PREDECLARED.** Written before any model in this directory was fitted and before any
number in this directory was read. Sections 1–3 describe the paper. Sections 4–7 fix the
equal-budget contract, the metrics, the gates and the reporting rule. Nothing in sections 4–7
may be edited after a result is seen. This task declares **no winner**; it builds the baseline
and reports both conditions on the same test units.

Author: restart worker (family 6). Date: 2026-09-10.
Writable scope: `results/family6_serrano_baseline/` only.
Read-only inputs: `dependency/`, `novelty_web_2026-09-10/`, `family1_second_pairs/`.
`dependency` is never touched.

---

## 0. Source used, and a provenance caveat

Fetched 2026-09-10:

| URL | Outcome |
|---|---|
| `https://arxiv.org/abs/2602.00884` | **OK** — metadata: v1 2026-01-31, v2 2026-08-10, cs.LG, CC BY 4.0, DOI 10.48550/arXiv.2602.00884. |
| `https://arxiv.org/pdf/2602.00884` | **FAILED** — `maxContentLength size of 10485760 exceeded`. The PDF was never read. |
| `https://arxiv.org/html/2602.00884` and `.../html/2602.00884v2` | **OK** — the HTML (experimental) rendering of **v2**. This is the version every quotation below comes from. |

**Caveat carried into every downstream document.** All quotations and all structure below are
from the arXiv **HTML v2** rendering, read through a fetch-and-summarise tool, not from the
typeset PDF. Section numbers are reported as the HTML renders them and are reliable. Displayed
equations in the body of §3–§4 are rendered **unnumbered** in the HTML; the only numbered
equations `(1)`–`(11)` are in Appendix A (the per-benchmark PDE definitions). Body equations are
therefore cited here by **section plus a verbatim quotation**, not by equation number, because
inventing an equation number that the paper does not display would be a fabricated citation.
Anyone finalising the manuscript should re-check the numbering against the typeset PDF.

Authors: Louis Serrano, Jiequn Han, Edouard Oyallon, Shirley Ho, Rudy Morel. ICML 2026 (poster,
per `novelty_web_2026-09-10/PRIOR_ART.md` row 5).

### 0.1 One UNRESOLVED item from the novelty pass is now closed

`novelty_web_2026-09-10/REPORT.md` lists as UNRESOLVED: *"whether Serrano et al. 2602.00884
explicitly uses Lie/Strang splitting terminology (only 'splitting strategy' confirmed from
abstract)."*

**Closed: it does, verbatim, in §4.3.** The paper writes
> "For two operators f1+f2, Lie splitting sequentially applies each operator over the full time
> step: u^{L+1} = f₂^{Δt} ∘ f₁^{Δt}(u^L)"

and
> "Strang splitting uses a symmetric pattern for higher accuracy: u^{L+1} = f₁^{Δt/2} ∘ f₂^{Δt}
> ∘ f₁^{Δt/2}(u^L)"

and, for more than two operators,
> "This palindromic composition preserves the symmetry required for second-order accuracy and
> extends classical Strang splitting".

The manuscript's related-work framing "operator dictionary + Lie/Strang splitting" is therefore
**supported by the source** (subject to the §0 HTML-vs-PDF caveat).

---

## 1. What the paper does

### 1.1 Problem setting (§3)

**§3.1 Parametric PDE setting.** A family

    ∂_t u = Σ_{k=1..K} μ_k F_k(u, ∇_x u, ∇²_x u, …)

with μ = (μ_1,…,μ_K) indexing which physical effects are active and how strongly. Training
parameters are drawn from a **sparse** distribution: quoting §3.1, *"parameters are drawn from a
sparse distribution P^train(μ), where only one operator is present for each trajectory."* Each
training trajectory is therefore **single-physics**.

**§3.2 OOD challenges.** Two test-time departures: *parameter extrapolation*
(μ_k^test ∉ conv(M_k^train)) and *operator composition* (several μ_k simultaneously nonzero, each
individually inside the training range).

**§3.3 Zero-shot prediction task.** Quoting §3.3: *"Given this OOD setting, our task is to predict
rollout trajectories in a zero-shot manner using only the observed dynamics at test time.
Specifically, we observe L consecutive snapshots of a test trajectory u_test^{1:L} with temporal
discretization Δt, which characterize the underlying dynamics that were never seen during
training."* Every experiment uses **L = 16** context snapshots and predicts **H** further steps
(H = 34 advection–diffusion, 50 combined-equation, 32 Gray–Scott, 16 Navier–Stokes).

### 1.2 What is trained (§4.1, "Constructing a Dictionary of Operators")

The backbone is **DISCO** (Morel, Han, Oyallon; ICML 2025; arXiv:2504.19496): a hypernetwork
ψ_α reads a context window and emits the parameters of a small operator network f_θ that
integrates forward,

    û^{L+1} = u^L + ∫_L^{L+1} f_θ(u^t) dt,     θ = ψ_α(u^{1:L}).

Training is **next-step prediction on single-physics trajectories only** — e.g. the
advection–diffusion model is trained on *"50% pure advection cases … and 50% pure diffusion
cases"*, never on mixtures. Optimiser AdamW, lr 3e-4, wd 1e-4; 300 000 iterations
(advection–diffusion, combined-equation), 100 000 (reaction–diffusion); batch 64 (1D).

The dictionary is then obtained **for free**, with no further training, by re-encoding training
trajectories: *"After pretraining, we extract a dictionary of neural operators by encoding each
trajectory i from the training set: {f_{θ_i} = ψ_α(u_i^{1:L})}."* Reported dictionary sizes after
subsampling: 256 (advection–diffusion), 96 (combined), 40 (Gray–Scott), 17 (Navier–Stokes).

**This is the single most important structural fact for budgeting.** In Serrano et al. the
dictionary size N is decoupled from the training cost: one hypernetwork fit buys N operators.

### 1.3 What is composed at test time (§4.3, "Operator Splitting for Neural Operators")

Lie (first order) and Strang (second order) as quoted in §0.1; for m operators, Lie is the
sequential composition f_{i_m} ∘ … ∘ f_{i_1}, and Strang is the palindromic extension. Strang
*"reduces the approximation error from O(Δt²) to O(Δt³)"*.

### 1.4 What is searched at test time (§4.2, "Operator Composition Search")

Objective, quoting §4.2: *"We define L(S) = 1/(L−1) Σ_{t=1}^{L−1} NRMSE(u_test^{t+1},
û_test^{t+1}) as the fitting error when using the operator subset S"*, with
NRMSE(u, û) = ‖u − û‖₂ / ‖u‖₂ and û^{t+1} produced from u^t by splitting with S. The objective is
therefore **teacher-forced one-step**, evaluated on the observed context window only.

Two search procedures:

- **Algorithm 1, "Beam Search Operator Composition."** Initialise B_0 = top-B single operators by
  L({f_i}); iterate B_{m+1} = top-B of {S ∪ {f_j} : S ∈ B_m}; stop on a *minimum relative
  improvement threshold* or at maximum composition length M. Cost O(BN) candidate evaluations per
  iteration. Settings: B = 4 (B = 8 Gray–Scott), M = 5.
- **Algorithm 2, "Uniform Operator Composition Search."** T random subsets with
  m ~ Uniform(1, M); T = 100 (T = 200 Gray–Scott). Cost O(T).

Nothing is fine-tuned: pretrained weights are not modified.

### 1.5 Test-time information requirement — and how it maps onto this project

The method needs, per test trajectory: **L = 16 consecutive ground-truth snapshots of that very
trajectory, at the operator's own Δt.** It needs no labels, no parameter values, no primitive
names and no ordering labels — it infers all of that from the snapshots.

Mapped onto this project's information-hygiene split (`dependency/scripts/code/generate.py`
module docstring: *"the learner sees raw float32 state trajectories and nothing else. No symbol,
token, equation string, coefficient, schedule, primitive name, or ordering label"*):

- On the **symbolic** axis the method is **hidden-setting**: it is given no primitive name, no
  coefficient, no ordering label. It is compatible with this project's hygiene rule.
- On the **observational** axis it is **supplied**: it is handed a window of the *test*
  trajectory's own states and fits to them at test time. Our broadened-dictionary condition is
  handed **no test-trajectory window at all** — it sees a single input state and must map it
  forward.

**Predeclared consequence:** the two conditions are *not* information-symmetric, and the
asymmetry favours the baseline. This is stated up front and reported in every table; it is not a
defect to be corrected, it is the honest description of the comparison a referee is asking for.

---

## 2. Structural mismatch between their setting and ours — declared before any run

Serrano et al.'s "operator composition" OOD case is **simultaneous** physics: several μ_k nonzero
at once, time-homogeneous, which is exactly the regime operator splitting was invented for.

This project's composed corpus is **sequential**: leg A for τ = 1.0, then leg B for τ = 1.0
(`dependency/scripts/code/generate.py:compose`, `run_dissociation.py:compose_true`). Within each
leg exactly one primitive is active, and the dynamics **change regime at the handoff frame**.

Three consequences, all predeclared:

1. A single context window cannot characterise the whole rollout, because the generator is not
   time-homogeneous. Any single searched composition must be wrong on at least one leg.
2. Splitting error is not the binding error inside a leg (there is nothing to split), so the
   Lie-vs-Strang distinction is expected to be near-null here. It is implemented and reported
   anyway, because the referee will ask.
3. It follows that a **causal** context (frames strictly preceding the estimand) and a
   **regime-matched** context (frames of the same dynamics, from a disjoint trajectory) give very
   different answers. **Both are run and both are reported.** The regime-matched context is an
   *upper bound* on what any test-time dictionary search could achieve here, and it is labelled
   as such.

---

## 3. Reimplementation choices forced by the equal-budget constraint

We do not have DISCO, and the equal-budget contract (§4) pins the backbone to the package's A1
FNO. Three faithfulness compromises follow. Each is listed again, with its alternative, in
`RESULT_SMOKE.md`.

**(a) Dictionary construction.** Serrano's dictionary is N operators from one hypernetwork fit.
Ours is **one A1 FNO per primitive**, trained on that primitive's single-physics data —
`{f_diff, f_reac}`, N = 2. This is what TASK.md §2 specifies. It is *less* expressive than a
256-entry dictionary, and the shortfall is real; but at the A1 budget there is no free way to
manufacture more operators, and manufacturing them by splitting the 640 units into shards would
break the "same 640 training units per primitive" clause.
*Compensation, predeclared:* the search space is **ordered multisets** of length m ≤ M = 5 over
the 2-element dictionary, i.e. 2+4+8+16+32 = **62 candidates**. Repetition is meaningful and
faithful: in splitting, each sub-step advances its own vector field over the full Δt, so
`f_diff ∘ f_diff` represents twice the diffusivity — which is precisely the
parameter-extrapolation mechanism of §5.2.

**(b) Operator ordering is searched.** §4.2's algorithms are written over *subsets* S, and the
HTML text nowhere states whether permutations are enumerated. For noncommuting primitives order
matters, so we **search over ordered sequences**. This is the generous reading. Marked as an
open choice.

**(c) Strang half-steps.** DISCO operators are continuous-time vector fields, so f^{Δt/2} is free.
The A1 FNO is a *discrete* τ-map (τ = K·DT = 0.10) and has no half-step. Two resolutions:
- **Chosen default — "Strang at grain 2τ":** take the splitting time step to be Δt = 2τ. Then
  f^{Δt/2} is exactly the native τ-map and f^{Δt} is the native map applied twice, so a Strang
  macro-step is exactly `f_1 ∘ f_2 ∘ f_2 ∘ f_1`, advances 2τ, and costs **no extra training**.
  It is exact Strang for the operators we have. Its rollout lands on the even-τ frame subgrid, so
  the Strang arm is scored on the 11-frame even subgrid and **every other arm is scored on that
  same subgrid too** for comparability.
- **Rejected for the smoke — "native-grain Strang":** fit a second FNO per primitive at stride
  k = 5. Same simulator calls, but **double** the gradient budget, which breaks §4. Deferred to
  the full run as an optional strictly-more-favourable arm.
An increment-scaled half-step `u + ½(f(u) − u)` was considered and **rejected**: its local error
is O(τ²), the same order Strang exists to remove, so it silently degrades Strang to Lie.

**(d) Exhaustive search is reported alongside beam and uniform.** With N = 2 and M = 5 the whole
space is 62 candidates, so we run Algorithm 1 (B = 4), Algorithm 2 (T = 100) **and** exhaustive
enumeration. Exhaustive dominates both and is the strongest selection the method could possibly
make; reporting it removes "your beam width was too small" as an escape hatch.

**(e) Minimum relative improvement threshold.** §4.2 says there is one; the HTML does not give its
value. Default **1 %** (a candidate of length m+1 must reduce L(S) by ≥ 1 % relative to the best
length-m candidate). Marked as an open choice; exhaustive search is threshold-free and is
reported, so the choice cannot drive the headline.

**(f) Context length L.** §3.3 uses L = 16 frames (15 transitions). Our legs are only 10 τ-frames
long, so L = 16 cannot fit inside a leg. Rule, fixed here: *choose L so that the context fits
strictly inside one leg and the context:horizon ratio is as close as possible to the paper's
16:50 = 0.32 of the frame budget.* On a 21-frame composed trajectory (20 transitions) that gives
**L = 7 frames (6 transitions), horizon 14 frames**. Marked as an open choice.

---

## 4. The equal-budget contract

An **equal-budget** comparison between our broadened-dictionary condition and Serrano-style
test-time splitting means all of the following hold simultaneously.

| Axis | Value, identical across both conditions |
|---|---|
| Training units per primitive | **640** (`timing_probe.NUM_TRAIN_UNITS`), 101 frames each, dt = 0.01 |
| Simulator calls | identical corpora: 640×100 diffusion steps + 640×100 reaction steps per IC-support condition, plus the shared evaluation cohorts. No condition generates a trajectory the other does not. |
| Backbone | package A1 FNO-1d: modes 16, width 64, 4 layers, project 128, in-channels (u, x), 549 569 parameters (`timing_probe.init_fno`) |
| Optimiser / schedule | hand Adam, 8 000 steps, batch 64, cosine lr 1e-3 → 1e-5 (`run_baseline.train`) |
| Fitted models per seed | **2** (one per primitive) in every condition |
| Gradient steps per seed | **16 000** in every condition |
| Estimand grain | τ = K·DT = 0.10, K = 10; 10 applications per leg |
| Test units | the **same** 48 units, the same ICs, the same seeds, for every arm |

What differs, and only this:

- **narrow / broad** differ in the IC offset support the 640 training units are drawn from:
  narrow (0.4865, 0.5171), broad (0.4865, 0.7258) — `run_baseline.NARROW` / `.BROAD`. Sample
  count is unchanged. This is the project's intervention.
- **Serrano arms** add **no training at all**. They spend their budget at test time: a search
  over 62 candidate compositions per test unit, each scored by 6 teacher-forced one-step FNO
  evaluations. That is ≈ 372 extra FNO forward passes per unit per arm, versus 0 for our
  condition. **This extra test-time compute is granted, not equalised**, and is reported as a
  count in the results.

Fitted models are shared between arms wherever they are literally the same model: the Serrano
narrow arm's dictionary **is** `{f_diff^narrow, f_reac^narrow}`, the same two files the narrow
oracle arm uses. No arm gets a private fit.

---

## 5. Predeclared metrics

Constants are the package's, taken by import, not by retyping:
`dependency/scripts/baseline/timing_probe.py` (physics, IC generator, FNO, optimiser) and
`dependency/scripts/baseline/run_baseline.py` (K, STEPS, BATCH, LR, NARROW, BROAD, thresholds).

### 5.1 R — ten-step off-support degradation ratio

Definition, verbatim in behaviour with `run_baseline.py`:

    relative_l2(p, y) = ‖p − y‖_F / ‖y‖_F   per unit, summing over (channel, space)
    e_on  = median over units of relative_l2(model(x_on),  y_on)
    e_off = median over units of relative_l2(model(x_off), y_off)
    R     = e_off / e_on

The estimand is the K = 10 **diffusion** solution operator, τ = 0.10.

**Evaluation states (fixed here, and the same for every arm).** The baseline needs a context
window, and a context window requires history. `run_baseline.py` evaluates at frame 0 (the IC
itself), where no history exists. We therefore evaluate at **frame 10 (t = 1.0)**, the end of one
leg, for both on- and off-support:

- `x_on`  = diffusion rolled 100 dt-steps from an IC in NARROW (seed MASTER_SEED+7), 48 units.
- `x_off` = reaction rolled 100 dt-steps from an IC in NARROW (seed MASTER_SEED+8), 48 units.
  This is **exactly** `run_baseline.py`'s switch state.
- `y_on`, `y_off` = diffusion applied K = 10 further dt-steps to `x_on`, `x_off`.

`x_off` is unchanged from the package. `x_on` is shifted from frame 0 to frame 10; diffusion
preserves the spatial mean, so the support geometry (G1) is unchanged, but the states are
smoother, so **the R values here are NOT numerically comparable to the banked 110.234 / 17.650**.
To bridge, our two oracle arms are *also* evaluated at frame 0 on the same 48 units, and both
numbers are reported. Declared now, before either is seen.

**Contexts (Serrano arms only), 7 frames at stride τ, 6 transitions:**
- **CTX-causal** — the 7 frames immediately preceding and including the evaluation state, on the
  test trajectory itself: dt-indices 40,50,…,100. On-support these are diffusion frames;
  off-support they are **reaction** frames, because that is what actually precedes a switch state.
  Zero leakage; fully deployable.
- **CTX-regime** (upper bound) — 7 frames of an **index-matched, disjoint** calibration trajectory
  (seeds MASTER_SEED+21 / +22) whose dynamics and input-state regime match the estimand: on-support,
  diffusion frames from a diffusion trajectory; off-support, the diffusion leg (dt-indices
  100,110,…,160) of a disjoint reaction→diffusion composition. The test unit's own target
  transition never enters the objective. Labelled **oracle** everywhere.

Strang advances 2τ per macro-step and therefore cannot produce a single τ-step: **the Strang arm
is not defined for R** and is reported as N/A, not as a number.

### 5.2 S — centered order-contrast score

Definition, verbatim in behaviour with `run_dissociation.py:centred_ratio`:

    d      = true_ab − true_ba                       (raw truth contrast, per unit/frame/point)
    d̄     = d − mean_over_units(d)                  (centered)
    dhat   = pred_ab − pred_ba
    dhat̄  = dhat − mean_over_units(dhat)
    S      = sqrt(mean(( dhat̄ − d̄ )²)) / sqrt(mean( d̄² ))

with the mean taken over (unit, frame, point) on the window.

- **trajectory** observable: all 21 τ-frames (frame 0 … 20).
- **endpoint** observable: frame 20 only.
- **even-subgrid** variants of both, on frames 0,2,…,20 (11 frames) — the only grid on which the
  Strang-at-2τ arm exists, computed for **every** arm.
- **horizon-only** secondary: frames 7…20, excluding the Serrano context window.

Cohort: 48 units, ICs in NARROW, seed MASTER_SEED+11 (`run_dissociation.EVAL_UNITS` seed), the
same units for every arm. Truth: `compose_true` with 10 applications per leg, both orders, from
bit-identical initial states.

- **Oracle arms (narrow, broad)** — `compose_predicted`: the fixed schedule, 10 applications of
  one primitive's FNO then 10 of the other. These arms are **told the order and the schedule**.
- **Serrano arms** — context = the first 7 frames of the true composed trajectory for that arm
  (frames 0…6, strictly inside leg A); search; then roll the single selected composition
  autoregressively for 20 applications from frame 0. Time-homogeneous, as the paper's method is.

**Predeclared identity, stated before computing:** a Serrano arm whose search were handed a
perfect oracle for the schedule would emit exactly `compose_predicted`, i.e. it would *be* the
narrow oracle arm. So the narrow oracle arm's S is the **oracle upper bound for the Serrano
method on S**, and no separate oracle-context S arm is run.

### 5.3 Arms reported

| id | dictionary / model | schedule | context |
|---|---|---|---|
| `narrow_oracle` | `{f_diff, f_reac}` narrow | told | none |
| `broad_oracle` | `{f_diff, f_reac}` broad | told | none |
| `serrano_narrow_lie_causal` | narrow | searched (Lie) | CTX-causal |
| `serrano_narrow_lie_regime` | narrow | searched (Lie) | CTX-regime (oracle) — R only |
| `serrano_narrow_strang2tau_causal` | narrow | searched (Strang, 2τ) | CTX-causal — S only |
| `serrano_broad_lie_causal` | broad | searched (Lie) | CTX-causal |

Each Serrano arm reports all three selection rules (exhaustive, beam B = 4, uniform T = 100).

---

## 6. Predeclared gates

Data / metric gates, thresholds copied from `run_baseline.py`:

| gate | statement | threshold |
|---|---|---|
| G1 | spatial-mean supports of diffusion training frames and of the switch states do not overlap | overlap == 0 |
| G2 | metric resolves: persistence_off / median e_on | ≥ 100 |
| G4 | e_on above the float32 floor | ≥ 1.19e-5 |
| P6 | convergence: worst e_on across fits | ≤ 1.573e-3 |

Implementation gates, specific to this task:

| gate | statement | threshold |
|---|---|---|
| U1 | **unit test.** On a commuting pair (two pure-diffusion Fourier multipliers, D₁ = 0.004, D₂ = 0.006), Lie splitting, native-grain Strang and 2τ-grain Strang each reproduce the exact combined operator exp(−(D₁+D₂)k²Δt) | max relative L2 error ≤ 1e-10 in float64 |
| U2 | **search unit test.** Given a context generated by the combined operator and a dictionary {D₁-op, D₂-op}, exhaustive and beam search both select a length-2 composition containing both operators, with L(S) at the same floor | L(S) ≤ 1e-10 |
| S1 | **search sanity.** On the on-support CTX-causal R context (pure diffusion history, diffusion estimand) the search selects a diffusion-only composition for ≥ 90 % of units | ≥ 0.90 |

A failure of U1, U2 or S1 means the implementation is wrong, not that the method is bad, and
blocks interpretation of the arm.

---

## 7. Reporting rule

1. Every arm is reported on the **same 48 test units**, the same seeds, the same targets.
2. Both conditions are reported side by side. **This task declares no winner.** The head-to-head
   verdict is a later task, after family 1's harness is validated.
3. The information asymmetry of §1.5 (the baseline is handed a window of the test trajectory; our
   condition is handed nothing) and the test-time compute asymmetry of §4 are stated in the same
   table as the numbers, never in a footnote.
4. The regime-matched context is always labelled **oracle**.
5. Failures, non-convergence and gate failures are preserved verbatim; no arm is dropped for
   looking bad.
6. Because the smoke uses 2 seeds, **no dispersion claim and no significance claim is made from
   it.** Seed ranges are printed as ranges of two numbers and nothing is inferred from them.
