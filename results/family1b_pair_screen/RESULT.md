# RESULT — Family 1b: truth-only pre-screen for measurable noncommuting pairs

Generated **2026-09-11T08:17:57Z** by `screen_pairs.py report` from `results/GEOM_stage{1,2,3}.json`, `results/ORDER_stage{1,2,3}.json` and `results/SCREEN_INTERVALS.json`. Every number below is read out of those files; none is transcribed by hand.

**Status: truth only.** No model was fitted anywhere in this family. There is no `e_off`, no `R`, and no claim about broadening, degradation or repair. A candidate that passes here is one on which the family-1 experiment is *measurable*, not one on which it *works*.

Predeclared in `results/family1b_pair_screen/PREDECLARED.md`, written before any candidate was integrated.

## Screen-validation gate (PREDECLARED §9, binding)

**PASS** — the screen reproduces family 1's own published truth-side numbers.

| check | published | measured | relative difference | tolerance | verdict |
|---|---|---|---|---|---|
| P1 persistence_off | `0.08019073763412352` | `0.08019073763412352` | `0.000e+00` | `1e-6` | PASS |
| P1 persistence_on | `0.12143668541229405` | `0.12143668541229405` | `0.000e+00` | `1e-6` | PASS |
| P2 persistence_off | `0.005920252665176344` | `0.005920252665176344` | `0.000e+00` | `1e-6` | PASS |
| P1 passes G2 proxy | expected `True` | got `True` | — | — | PASS |
| P2 fails G2 proxy | expected `False` | got `False` | — | — | PASS |

## The learner-free G2 proxy (PREDECLARED §5)

`e_on_ref = 1.9224691546e-04` — P2's ten-seed median `e_on` (`family1_second_pairs/results/P2_diffusion_allen_cahn/GATES.json`). `e_on` is a property of the **on-support** cohort, which does not depend on the second operator at all; P1 and P2 measured it at `1.8748e-4` and `1.9225e-4`, within 2.5 %. The larger is used, which lowers every proxy.

```
proxy_dynamic_range        = persistence_off / e_on_ref              # first operator = diffusion(0.01), dt = 0.01, k = 10
proxy_dynamic_range_scaled = (persistence_off / persistence_on) * 631.6704   # EXTRAPOLATED rows
```

Gate: `>= 100.0` to pass, `>= 150.0` for STRONG. On the one pair where the proxy can be checked independently it is 2.5 % conservative (P1: proxy 417.12 against its own fitted 427.73). That is the accuracy claimed; nothing better.

## Counts

- **25 configurations** screened
- **12 pass all four gates**
- **10 fail**
- **3 skipped a priori** as conservative and/or commuting (predeclared reason), integrated once anyway to confirm the prediction

## A. Passing candidates, ranked by proxy dynamic range

`broad` is the exact offset interval that covers the switch support at the fixed 640-unit sample count, derived by the rule in PREDECLARED §7. `narrow` is `[0.4865, 0.5171]` for every row, verbatim `run_baseline.py:NARROW`.

| # | id | A (learned lane) | B (switch leg) | gap | `persistence_off` | proxy DR | strength | order dep. | broad offsets | extrapolated? |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `S2_grayscott_1sp_nu0.005` | diffusion(diffusivity=0.005) | grayscott_1sp(F=0.04) | +0.0584 (down) | 0.099374 | **877.9** | STRONG | 5.57e-02 | `[0.3, 0.52]` | **yes** |
| 2 | `S1_grayscott_1sp` | diffusion(diffusivity=0.01) | grayscott_1sp(F=0.04) | +0.0584 (down) | 0.167147 | **869.4** | STRONG | 4.91e-02 | `[0.3, 0.52]` | no |
| 3 | `S2_grayscott_1sp_nu0.02` | diffusion(diffusivity=0.02) | grayscott_1sp(F=0.04) | +0.0584 (down) | 0.252280 | **859.9** | STRONG | 4.05e-02 | `[0.3, 0.52]` | **yes** |
| 4 | `S2_fkpp_r0.5_nu0.005` | diffusion(diffusivity=0.005) | fisher_kpp(diffusivity=0.0, reactivity=0.5) | +0.0771 (up) | 0.060703 | **536.2** | STRONG | 3.86e-02 | `[0.48, 0.7]` | **yes** |
| 5 | `S1_fkpp_r0.5` | diffusion(diffusivity=0.01) | fisher_kpp(diffusivity=0.0, reactivity=0.5) | +0.0771 (up) | 0.101538 | **528.2** | STRONG | 3.34e-02 | `[0.48, 0.7]` | no |
| 6 | `S2_fkpp_r0.5_nu0.02` | diffusion(diffusivity=0.02) | fisher_kpp(diffusivity=0.0, reactivity=0.5) | +0.0771 (up) | 0.151705 | **517.1** | STRONG | 2.71e-02 | `[0.48, 0.7]` | **yes** |
| 7 | `S3_burgers_first_dt0.0025` | burgers(diffusivity=0.01, convection_scale=1.0) | fisher_kpp(diffusivity=0.0, reactivity=1.0) | +0.1736 (up) | 0.201802 | **480.3** | STRONG | 1.90e-01 | `[0.48, 0.8]` | **yes** |
| 8 | `S2_fkpp_r1.0_nu0.005` | diffusion(diffusivity=0.005) | fisher_kpp(diffusivity=0.0, reactivity=1.0) | +0.1736 (up) | 0.048857 | **431.6** | STRONG | 8.13e-02 | `[0.48, 0.8]` | **yes** |
| 9 | `S1_fkpp_r1.0` | diffusion(diffusivity=0.01) | fisher_kpp(diffusivity=0.0, reactivity=1.0) | +0.1736 (up) | 0.080191 | **417.1** | STRONG | 6.88e-02 | `[0.48, 0.8]` | no |
| 10 | `S2_fkpp_r1.0_nu0.02` | diffusion(diffusivity=0.02) | fisher_kpp(diffusivity=0.0, reactivity=1.0) | +0.1736 (up) | 0.117881 | **401.8** | STRONG | 5.45e-02 | `[0.48, 0.8]` | **yes** |
| 11 | `S3_hyperdiff_first` | hyperdiffusion(hyper_diffusivity=0.0001) | fisher_kpp(diffusivity=0.0, reactivity=1.0) | +0.1736 (up) | 0.147150 | **395.1** | STRONG | 7.15e-02 | `[0.48, 0.8]` | **yes** |
| 12 | `S1_fkpp_r2.0` | diffusion(diffusivity=0.01) | fisher_kpp(diffusivity=0.0, reactivity=2.0) | +0.3157 (up) | 0.044662 | **232.3** | STRONG | 1.14e-01 | `[0.48, 0.95]` | no |

## B. Closest misses, with the binding gate

| id | A | B | binding gate | why | proxy DR |
|---|---|---|---|---|---|
| `S1_nagumo_a0.75` | diffusion(diffusivity=0.01) | nagumo(a=0.75) | **G1_shift_exists** | overlap 0.00000, gap +0.0208 (needs overlap 0 and gap >= 0.05) | 845.1 |
| `S1_nagumo_a0.5` | diffusion(diffusivity=0.01) | nagumo(a=0.5) | **G1_shift_exists** | overlap 0.03042, gap -0.0373 (needs overlap 0 and gap >= 0.05) | 757.4 |
| `S1_nagumo_a0.25` | diffusion(diffusivity=0.01) | nagumo(a=0.25) | **G1_shift_exists** | overlap 0.00000, gap +0.0215 (needs overlap 0 and gap >= 0.05) | 684.9 |
| `S3_burgers_first_dt0.0025_k10` | burgers(diffusivity=0.01, convection_scale=1.0) | fisher_kpp(diffusivity=0.0, reactivity=1.0) | **G1_shift_exists** | overlap 0.00000, gap +0.0242 (needs overlap 0 and gap >= 0.05) | 616.4 |
| `S3_burgers_first_dt0.01` | burgers(diffusivity=0.01, convection_scale=1.0) | fisher_kpp(diffusivity=0.0, reactivity=1.0) | **G6_stable** | advective CFL 2.560 > 1.0 | 480.5 |
| `S3_ks_first` | ks | fisher_kpp(diffusivity=0.0, reactivity=1.0) | **G6_stable** | advective CFL 2.560 > 1.0 | 375.3 |
| `S1_allen_cahn` | diffusion(diffusivity=0.01) | allen_cahn | **G2_proxy_resolves** | `persistence_off` 0.005920 gives proxy 30.80 < 100.0 | 30.8 |
| `S1_ks` | diffusion(diffusivity=0.01) | ks | **G1_shift_exists** | overlap 0.03042, gap -0.0304 (needs overlap 0 and gap >= 0.05) (also fails G2_proxy_resolves, G5_noncommuting, G6_stable) | 0.0 |
| `S1_swift_pkg` | diffusion(diffusivity=0.01) | swift_hohenberg(reactivity=0.7) | **G1_shift_exists** | overlap 0.00512, gap -0.0051 (needs overlap 0 and gap >= 0.05) (also fails G2_proxy_resolves) | 0.0 |
| `S1_swift_r2.0` | diffusion(diffusivity=0.01) | swift_hohenberg(reactivity=2.0) | **G2_proxy_resolves** | `persistence_off` 0.000000 gives proxy 0.00 < 100.0 | 0.0 |

## C. Skipped a priori — predeclared as unable to shift the spatial mean

Every one was integrated once anyway; the measurement column is the confirmation, not the decision.

| id | B | predeclared reason | measured median mean drift | measured overlap | measured order dependence |
|---|---|---|---|---|---|
| `S1_burgers_dt0.01` | burgers(diffusivity=0.01, convection_scale=1.0) | conservative: every term is an exact x-derivative, so d<u>/dt = 0 on a periodic domain and the switch-state spatial-mean support IS the narrow training support. No coefficient choice inside this candidate can produce the zero-overlap shift G1 requires. Also expected to fail CFL at dt=0.01 (the recorded apparatus stop). Integrated once to confirm both predictions. | `2.980e-08` | `0.03042` (total) | `2.466e-02` |
| `S1_burgers_dt0.0025` | burgers(diffusivity=0.01, convection_scale=1.0) | conservative: every term is an exact x-derivative, so d<u>/dt = 0 on a periodic domain and the switch-state spatial-mean support IS the narrow training support. No coefficient choice inside this candidate can produce the zero-overlap shift G1 requires. The dt reduction fixes CFL, not G1. Separate dt regime. | `2.980e-08` | `0.03042` (total) | `2.464e-02` |
| `S1_hyperdiff` | hyperdiffusion(hyper_diffusivity=0.0001) | conservative: every term is an exact x-derivative, so d<u>/dt = 0 on a periodic domain and the switch-state spatial-mean support IS the narrow training support. No coefficient choice inside this candidate can produce the zero-overlap shift G1 requires. AND it is diagonal in Fourier, so it commutes exactly with diffusion(0.01): the order-dependence gate must return the commuting null. Integrated once to confirm both predictions. | `2.980e-08` | `0.03042` (total) | `5.551e-16` |

The measured mean drift on all three is `2.980e-08`, which is float32 roundoff on a mean of order `0.5`: the prediction holds to machine precision. `S1_hyperdiff`'s order dependence `5.551e-16` is the **commuting null** (`screen_stability.py` measured `4.996e-16` on a pair known to commute), which is the second prediction confirmed: hyperdiffusion is Fourier-diagonal and commutes exactly with diffusion, so it could not have tested an order question even if it had shifted the mean.

## D. Findings the screen makes visible

**1. The three Nagumo candidates have the largest off-support tasks in the whole grid and are still blocked — by geometry, not by size.** Their `persistence_off` values are `0.131661`, `0.145606`, `0.162475`, i.e. 1.6-2.0x P1's `0.080191`, and their proxy dynamic ranges are `684.9`, `757.4`, `845.1`. A bistable reaction moves each unit's mean toward whichever basin that unit starts in, so the *cohort* spreads instead of translating: the switch support widens to cover the training support rather than clearing it.

   **Disclosure that matters for anyone re-reading family 1:** `S1_nagumo_a0.25` and `S1_nagumo_a0.75` have `overlap == 0.0` exactly, so they **pass family 1's G1 as family 1 writes it**. They fail only family 1b's additional `gap >= 0.05` clause (measured gaps `+0.0215` and `+0.0208`), which was declared in PREDECLARED §8 before any candidate was run and is not relaxed now that it binds. If the owner decides a hair's-breadth shift is acceptable, these three are the highest-dynamic-range candidates in the grid and the decision is the reader's, not this screen's.

**2. Swift-Hohenberg fails G2 harder than Allen-Cahn did — `persistence_off` is exactly `0.0`.** Not small: zero, in float32, on both the package coefficients and the alternative. SH damps every non-constant mode on `L = 1`, so after 100 steps the switch state is spatially constant and the 10-step diffusion map is *exactly* the identity on it. The dynamic range is 0, against Allen-Cahn's 30.8 and the required 100. `S1_swift_r2.0` produces the largest support shift in the grid (gap `+0.8324`, switch support `[1.349375605583191, 1.382564663887024]`) and it buys nothing, because there is no off-support *task* left to measure. This is the same failure mode family 1 found on Allen-Cahn, in its limiting form: G1 and G2 pull against each other, and a second operator that drives the field to a fixed point wins G1 by losing G2.

**3. Fisher-KPP reactivity runs the same trade-off, and the package's `r = 1.0` is not the best point on it.** `r = 0.5` gives `persistence_off = 0.101538` (proxy 528.2) against `r = 1.0`'s `0.080191` (417.1) and `r = 2.0`'s `0.044662` (232.3). Higher reactivity means a bigger support shift and a *smaller* measurable task, for exactly the Allen-Cahn reason: the field saturates toward `u = 1` and goes stationary under diffusion. `r = 0.5` still clears the gap (`+0.0771`) while leaving the largest task of the three.

**4. Every conservative second operator is dead on arrival, and the screen confirms it to machine precision.** Burgers (both `dt` regimes), hyperdiffusion, and Cahn-Hilliard (family 1's P3) all hold the spatial mean fixed on a periodic domain, so no coefficient choice inside those pairs can produce a support shift. Kuramoto-Sivashinsky is the near-miss of this group: Exponax ships the **combustion (non-conservative)** form, so its mean *can* drift — but on `L = 1` every mode has growth rate `(2*pi*n)^2 - (2*pi*n)^4 < 0`, the field is damped to a constant, and the measured drift is `1.192e-07` with `persistence_off = 0.0` and an order dependence of `1.110e-16`, the commuting null. It fails all four gates.

**5. A non-diffusion learned lane works, at a cost.** `S3_burgers_first_dt0.0025` (Burgers as the estimand, Fisher-KPP as the switch leg) passes every gate with proxy `480.3` and the strongest order dependence in the grid (`1.90e-01`), but only at `dt = 0.0025`: at `dt = 0.01` the advective CFL is `2.560` and it is eliminated by `screen_stability.py` rule 3, reproducing the recorded apparatus stop. The `dt` reduction has to buy leg duration too — the same pair at 100 steps/leg (tau_leg = 0.25) fails G1 with a gap of only `+0.0242`, because Fisher-KPP has not had time to move the mean. Both rows are EXTRAPOLATED.

**6. First-operator diffusivity barely moves the proxy.** Across `0.005 / 0.01 / 0.02`, `persistence_off` and `persistence_on` scale together, so the ratio is nearly flat: Gray-Scott-like `877.9 / 869.4 / 859.9`, Fisher-KPP `r=0.5` `536.2 / 528.2 / 517.1`, Fisher-KPP `r=1.0` `431.6 / 417.1 / 401.8`. Diffusivity is not a lever for making a pair measurable.

**7. Disclosures on the two rules that are not family 1's.** (a) The broad interval for the Gray-Scott-like candidates comes from the **mirrored** rule declared in PREDECLARED §7, because family 1's rule fixes the lower bound at `0.48` and can only widen upward; that candidate's shift is downward. (b) The advective CFL rule was applied to Kuramoto-Sivashinsky's gradient-norm term using the measured `max|u|` as the velocity, which is the predeclared rule but a loose physical reading for a term that is not a convection velocity; KS fails three other gates regardless, so nothing in this screen turns on it. (c) Applying family 1's §4 rule to P1 mechanically gives `[0.48, 0.80]`, not the package's hand-set `[0.4865, 0.7258]`; family 1's PREDECLARED §4 says P1's interval is taken verbatim and not re-derived, so the two do not conflict — but a family-1 rerun of P1 must keep using the package's.

**8. Every broad-interval search closed on its first candidate offset**, with 100 % of the 256 switch-state means inside the 640-unit broad training support and 0 units off support; saturated-value fractions were `0.0` to `1.2e-03`. No candidate needed the rule's `+0.05` escalation.

## E. Recommended pairs for the family-1 rerun

Rule (PREDECLARED, applied mechanically): non-extrapolated proxy first, then STRONG before MARGINAL, then proxy dynamic range descending, then id; the P1 positive control is excluded because family 1 has already run it.

**1. `S1_grayscott_1sp`** — A = diffusion(diffusivity=0.01), B = grayscott_1sp(F=0.04), `dt = 0.01`, 100 steps/leg (tau_leg = 1.0), `k = 10`

| | |
|---|---|
| narrow offsets | `[0.4865, 0.5171]` |
| broad offsets | `[0.3, 0.52]` |
| training units, both conditions | `640` (fixed; only the support moves) |
| training support | `[0.4865764, 0.5169979]` |
| switch support | `[0.3906632, 0.4281802]` |
| overlap / gap | `0.000000` / `+0.058396` (downward shift) |
| `persistence_off` / `persistence_on` | `0.167147` / `0.121437` |
| proxy dynamic range | `869.44` (STRONG; P1 measured 427.73, P2 30.80) |
| order dependence | `4.9112e-02` (floor `1e-06`, commuting null `4.996e-16`) |
| broad-interval rule | family1b PREDECLARED.md §7 MIRRORED_RULE (new; family 1's rule is upward-only) |

**2. `S1_fkpp_r0.5`** — A = diffusion(diffusivity=0.01), B = fisher_kpp(diffusivity=0.0, reactivity=0.5), `dt = 0.01`, 100 steps/leg (tau_leg = 1.0), `k = 10`

| | |
|---|---|
| narrow offsets | `[0.4865, 0.5171]` |
| broad offsets | `[0.48, 0.7]` |
| training units, both conditions | `640` (fixed; only the support moves) |
| training support | `[0.4865764, 0.5169979]` |
| switch support | `[0.5941063, 0.6244125]` |
| overlap / gap | `0.000000` / `+0.077108` (upward shift) |
| `persistence_off` / `persistence_on` | `0.101538` / `0.121437` |
| proxy dynamic range | `528.17` (STRONG; P1 measured 427.73, P2 30.80) |
| order dependence | `3.3392e-02` (floor `1e-06`, commuting null `4.996e-16`) |
| broad-interval rule | family1_second_pairs/PREDECLARED.md §4 (verbatim) |

**Caveat on the order of these two, stated rather than acted on.** The rule was declared in advance and its output is reported unchanged. But the two are not equally safe: `S1_grayscott_1sp` ranks first on the proxy (869.4 vs 528.2) and has the *tighter* margin on the other two counts — its gap is `+0.0584` against the `0.05` threshold (`S1_fkpp_r0.5`: `+0.0771`), and its broad interval needs the mirrored downward rule that family 1 does not have, whereas `S1_fkpp_r0.5`'s comes from family 1's §4 verbatim. `S1_fkpp_r0.5` is also the smallest change to what family 1 already validated: the same operator as P1 at half the reactivity, with a 1.27x larger off-support task than P1's. If only one pair can be run, `S1_fkpp_r0.5` is the one with the fewest new moving parts; if the point is to test a *different* reaction and a downward shift, `S1_grayscott_1sp` is the one that does that. The screen does not choose between those aims.

## F. What this does not say

- Nothing was fitted, so nothing here bears on whether the broadening result reproduces. Family 1 still computes the real `G2` from a real `median_e_on`, and a candidate that passes here can still fail there.
- `G4_censored_below` (`1.19e-5`) and `P6_convergence` (`1.573e-3`) are learner-side and **not evaluable** by this screen. `e_on_ref = 1.92e-4` sits 16x above the floor and 8x below the convergence limit; that is all the screen can say.
- Every row marked **extrapolated** uses the transfer proxy, whose assumption (`e_on / persistence_on` constant across first operators) is unverified. Those rows are never recommended ahead of a non-extrapolated row.
- One architecture, one resolution (256), one IC family, one `k` per dt regime. Seeds are not part of this screen at all: nothing here is seed-dependent except through `MASTER_SEED`, which is the package's.

