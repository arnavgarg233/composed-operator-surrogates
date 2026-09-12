# PREDECLARED — Family 1b: truth-only pre-screen for measurable noncommuting pairs

Written **2026-09-11T08:01:50Z**, before any candidate was integrated, before any
switch-state geometry of any new candidate was measured, and before any number in this
family was read. Nothing in sections 1–9 was edited after that timestamp; section 10 is
the only section filled in later and it records outputs, not choices.

What had been read before this file was written: the family-1 artefacts named in section
2 (all of which are *existing, already-published* numbers), the shipped harness source,
and the Exponax 0.2.0 constructor signatures and docstrings (a code-reading step, not a
measurement). No candidate operator was stepped.

Release rules in force: predeclare; never change a gate after an outcome; preserve failures
as-is; report the honest number first.
`dependency` is never modified.
`dependency/composed-operator-surrogates` is never modified (read-only, imported from).
`results/family1_second_pairs` is never modified (read-only).
All outputs go to `results/family1b_pair_screen/`.

---

## 1. Question

Family 1 asked whether the package's broadening result reproduces with a fitted learner on
new noncommuting pairs. Two of the three pairs it ran never reached that question:

- **P2 = diffusion(0.01) / Allen-Cahn** failed **G2**: `persistence_off = 0.005920`,
  `median_e_on = 1.9225e-4`, `dynamic_range = 30.795 < 100`. Allen-Cahn drives the field
  into the `u = +1` basin, so the switch states are smooth and nearly stationary under
  diffusion; the off-support task is 13.5× smaller than the one the metric was built for.
  (`family1_second_pairs/results/P2_diffusion_allen_cahn/GATES.json`, `RESULT.json`)
- **P3 = diffusion(0.01) / Cahn-Hilliard** failed **G1**: Cahn-Hilliard is written in
  conservative form, so it holds the spatial mean of every unit fixed; the switch-state
  spatial-mean support *is* the narrow training support and the overlap is total.
  (`family1_second_pairs/results/P3_diffusion_cahn_hilliard/GATES.json`)

Both failures were knowable **before any model was fitted**, because `persistence_off`,
`persistence_on` and both supports are properties of the truth data alone. The stability
screen that selected these pairs
(`dependency/scripts/second_pair/screen_stability.py`) deliberately measured neither —
it checked finiteness, blow-up, advective CFL and order dependence only, by design.

**Family 1b is the missing screen.** It asks, using truth data only and no fitted learner:

> Which noncommuting operator pairs produce (a) a zero-overlap spatial-mean support shift
> and (b) an off-support task large enough that the family-1 metric can resolve it?

Family 1b produces **no** claim about broadening, about `R`, about `e_off`, or about any
learner. Its entire output is a shortlist and the broad intervals family 1 would use.

## 2. Reference numbers imported from already-published artefacts

Read before this file was written; not re-derived here.

| quantity | value | source |
|---|---|---|
| `P1 median e_on` (5 seeds, package) | `1.8748138348456255e-4` | `dependency/results/tables/baseline/BASELINE_RESULT.json` |
| `P2 median e_on` (10 seeds, family 1) | `1.9224691546033533e-4` | `family1_second_pairs/results/P2_diffusion_allen_cahn/GATES.json` |
| `persistence_on` (diffusion 0.01, k=10, narrow) | `0.12143668541229405` | same GATES.json |
| `P1 persistence_off` | `0.08019073763412352` | `family1_second_pairs/results/P1_diffusion_fisher_kpp/GATES.json` |
| `P2 persistence_off` | `0.005920252665176344` | `family1_second_pairs/results/P2_diffusion_allen_cahn/GATES.json` |
| `G2_MIN_DYNAMIC_RANGE` | `100.0` | `run_baseline.py` |
| `G2_MAX_E_ON_FRACTION` | `0.1` | `run_baseline.py` |
| `ORDER_DEPENDENCE_FLOOR` | `1e-6` | `screen_stability.py` |
| commuting null measured on a pair known to commute | `4.996e-16` | `screen_stability.py` docstring |
| `GROWTH_LIMIT` / `CFL_LIMIT` | `10.0` / `1.0` | `screen_stability.py` |

## 3. Physical configuration (verbatim from the package unless stated)

1-D periodic, `DOMAIN_EXTENT = 1.0`, `NUM_POINTS = 256`, ETDRK `order = 2`,
`dealiasing_fraction = 2/3`, `num_circle_points = 16`, `circle_radius = 1.0`.
`dt = 0.01` and `STEPS_PER_LEG = 100` (τ = 1.00) except in the declared `dt = 0.0025`
regime (section 6). IC generator verbatim:
`clip(offset_i + 0.175 * RandomTruncatedFourierSeries(cutoff=5, std_one=True), 0, 1)`.
`MASTER_SEED = 20260905`; corpora at `MASTER_SEED+1` (train, 640 units),
`MASTER_SEED+7` (on-support, 256 units), `MASTER_SEED+8` (switch, 256 units) — exactly the
offsets `run_baseline.py` uses. Order-dependence contrast at `SCREEN_SEED = 20260905`,
32 units, exactly as `screen_stability.py`.

Dealiasing note, disclosed: Exponax defaults `dealiasing_fraction` to `0.5` for the cubic
reaction steppers (`AllenCahn`, `CahnHilliard`, `SwiftHohenberg`, `GrayScott`) and to
`2/3` elsewhere. `run_family1.py` passes `2/3` explicitly to every reaction stepper it
vendors. Family 1b follows `run_family1.py` and passes `2/3` everywhere, so that the
operators it screens are byte-for-byte the operators family 1 would run.

Unit counts, all at or below the package's: 640 training units, 256 on-support eval units,
256 switch units, 32 order-contrast units. No count above the package's is used anywhere.

## 4. Definitions, copied from the family-1 harness

- **Narrow offsets**: `(0.4865, 0.5171)` = `run_baseline.py:NARROW`, the same for every
  candidate; not re-derived.
- **Narrow training support**: the interval of spatial means of the narrow **training**
  states under the candidate's **first** operator, over all `STEPS_PER_LEG + 1` frames of
  all 640 trajectories. (`run_family1.py:geometry()`.) Recomputed per first operator,
  because a non-diffusion first operator need not preserve the mean.
- **Switch support**: the interval of spatial means of the 256 switch states, i.e.
  `rollout(second, switch_ic, STEPS_PER_LEG)[:, -1]`. (`run_family1.py:build_corpora()`.)
- **overlap** `= max(0, min(hi) - max(lo))`; **gap** `= switch_lo - narrow_hi` for an
  upward shift (`run_family1.py:geometry()`), and `= narrow_lo - switch_hi` for a
  downward shift (new; see section 7).
- **persistence_on** `= median relative_l2(on_states, on_target)` where
  `on_states = initial_conditions(256, narrow, MASTER_SEED+7)` (the harness takes frame
  `[:, 0]`, i.e. the ICs themselves) and `on_target = rollout(first, on_states, k)[:, -1]`.
- **persistence_off** `= median relative_l2(switch_states, switch_target)` where
  `switch_target = rollout(first, switch_states, k)[:, -1]`.
- `relative_l2` is `run_baseline.py:relative_l2`, imported, computed in float64 on float32
  states exactly as `run_family1.py:gates()` does.
- `k = 10` except in the `dt = 0.0025` regime (section 6).
- float32 throughout the geometry/persistence pass (`jax_enable_x64` is **not** set),
  matching `run_baseline.py`. The order-dependence pass runs in a **separate process**
  with `jax_enable_x64 = True`, matching `screen_stability.py`; the two are never mixed.

## 5. The learner-free G2 proxy — declared here, before any candidate is stepped

`G2` as shipped is
`dynamic_range = persistence_off / median_e_on >= 100` **and**
`median_e_on <= 0.1 * persistence_on`, where `e_on_limit = 0.1 * persistence_on` is how
the harness names the second clause. `median_e_on` needs a fitted model, which this family
is forbidden to produce. So `median_e_on` is replaced by a **declared constant**, and the
two clauses are evaluated exactly as the harness evaluates them.

**Reference on-support error scale**

```
e_on_ref = 1.9224691546033533e-4        # P2's ten-seed median e_on, family 1 GATES.json
```

Why this constant is legitimate rather than a guess: `e_on` is the median relative L2 of
the fitted model on the **on-support** cohort, and that cohort depends only on the narrow
offsets, the IC generator, the **first** operator, `k`, and the architecture — **not on
the second operator at all**. P1 and P2 differ only in the second operator, and their
fitted medians are `1.8748e-4` and `1.9225e-4`, within 2.5 % of each other across 5 and 10
seeds. For any candidate whose first operator is `diffusion(0.01)` at `dt = 0.01, k = 10`,
`e_on_ref` is therefore not an extrapolation: it is the same quantity, already measured
twice. The larger (P2, better sampled at ten seeds) is taken, which is the conservative
choice because it lowers every proxy dynamic range.

**Primary proxy** (first operator `= diffusion(0.01)`, `dt = 0.01`, `k = 10`):

```
proxy_dynamic_range = persistence_off / e_on_ref
proxy_G2_clause_2   = (e_on_ref <= 0.1 * persistence_on)      # the harness's e_on_limit
proxy_G2_pass       = (proxy_dynamic_range >= 100.0) and proxy_G2_clause_2
```

**Transfer proxy** (first operator *not* `diffusion(0.01)`, or a different `dt`/`k`):
`e_on_ref` is not the same quantity there, so it is scaled by how hard the on-support task
is, using `persistence_on` — a truth-only quantity — as the difficulty yardstick:

```
M                          = persistence_on_ref / e_on_ref = 0.12143668541229405 / 1.9224691546033533e-4
proxy_dynamic_range_scaled = (persistence_off / persistence_on) * M
```

For a `diffusion(0.01), dt = 0.01, k = 10` candidate the two forms are algebraically
identical, so the primary proxy is the special case, not a different rule. The transfer
form assumes `e_on / persistence_on` is constant across first operators. **That assumption
is unverified and every row that uses it is stamped `EXTRAPOLATED`, is reported separately,
and is never recommended for the family-1 rerun ahead of a non-extrapolated row.**

**Calibration check, stated in advance.** The proxy must reproduce the two known points:

| pair | published `dynamic_range` | proxy must give |
|---|---|---|
| P1 diffusion/Fisher-KPP | `427.726` (package) / `350.438` (family 1, 2 seeds) | `0.08019073763412352 / 1.9224691546033533e-4 = 417.13` |
| P2 diffusion/Allen-Cahn | `30.795` | `0.005920252665176344 / 1.9224691546033533e-4 = 30.795` |

P2 is exact by construction (it is P2's own `median_e_on`). P1 lands at 417.1 against its
own 427.7 because P1's fitted `e_on` was 2.5 % smaller; the proxy is 2.5 % conservative
there. **This is a 2.5 %-accurate stand-in on the one pair where it can be checked
independently, not a better-than-that claim.**

**Literal-reading disclosure.** The commissioning note can also be read as
`proxy = persistence_off / e_on_limit` with `e_on_limit = 0.1 * persistence_on`. That
ratio is `0.08019 / 0.012144 = 6.60` on P1 — the one pair known to produce a real result —
so a threshold of 100 on that ratio would reject P1 itself and the reading cannot be the
intended one. The column is still computed and published for every candidate as
`headroom_ratio`, with no threshold attached to it.

## 6. Candidate grid — fixed here, in full, before any of it was run

Notation: **A** = first operator (the lane the learner would be trained on, the estimand),
**B** = second operator (the switch-producing leg). All at `N = 256`, `L = 1.0`,
ETDRK order 2, dealiasing 2/3, package IC generator.

### 6.1 Stage 1 — A = diffusion(0.01), dt = 0.01, 100 steps/leg, k = 10

| id | B | coefficients | can B shift the spatial mean? |
|---|---|---|---|
| `S1_fkpp_r0.5` | Fisher-KPP | `diffusivity=0.0, reactivity=0.5` | **yes** — `r·u(1−u)` is not a divergence; `d⟨u⟩/dt = r⟨u(1−u)⟩ > 0` on `u ∈ (0,1)` |
| `S1_fkpp_r1.0` | Fisher-KPP | `diffusivity=0.0, reactivity=1.0` | **yes** — this is P1, carried as the positive control |
| `S1_fkpp_r2.0` | Fisher-KPP | `diffusivity=0.0, reactivity=2.0` | **yes** — same, faster |
| `S1_allen_cahn` | Allen-Cahn | package: `0.005, +1.0 u, −1.0 u³` | **yes** — carried as the *negative* control (family 1 measured 30.795) |
| `S1_swift_pkg` | Swift-Hohenberg | package/Exponax default: `reactivity=0.7, critical_number=1.0, poly=(0,0,1,−1)` | **yes, but weakly and downward** — zero-mode linear rate is `r − 1 = −0.3`, so the mean decays; `+u²` pushes back |
| `S1_swift_r2.0` | Swift-Hohenberg | `reactivity=2.0`, rest as above | **yes, upward** — zero-mode rate `r − 1 = +1.0` |
| `S1_nagumo_a0.25` | Nagumo/bistable via `GeneralPolynomialStepper` | `u_t = u(1−u)(u−a)`, `a=0.25`: `linear=(−a,0,0)`, `poly=(0,0,1+a,−1)` | **yes** — cubic reaction, not a divergence |
| `S1_nagumo_a0.5` | same | `a=0.5` | **yes**; `a=0.5` is the symmetric case, mean drift is the smallest of the three by construction |
| `S1_nagumo_a0.75` | same | `a=0.75` | **yes**, expected downward (most ICs below the `u=a` threshold) |
| `S1_grayscott_1sp` | Gray-Scott-like single species via `GeneralPolynomialStepper` | `v` eliminated by `v = 1−u`: `u_t = F(1−u) − u(1−u)²` with `F = 0.04` (Exponax `feed_rate` default) → `linear=(−(1+F),0,0)`, `poly=(F,0,2,−1)` | **yes** — the constant `+F` term is a nonzero mean source |
| `S1_burgers_dt0.01` | Burgers | `diffusivity=0.01, convection_scale=1.0` | **NO — conservative.** `−u u_x = −(u²/2)_x` and `ν u_xx` are both exact `x`-derivatives, so `d⟨u⟩/dt = 0` on a periodic domain, in both the conservative and non-conservative forms. Also expected to fail CFL at this `dt` (the recorded apparatus stop). **Skipped from the ranking with this reason; still integrated once to confirm the prediction.** |
| `S1_burgers_dt0.0025` | Burgers | same, `dt=0.0025`, 400 steps/leg | **NO — conservative**, same reason. The `dt` reduction fixes CFL, not G1. Recorded as a separate `dt` regime as commissioned. **Skipped from the ranking; integrated once to confirm.** |
| `S1_ks` | Kuramoto-Sivashinsky | `gradient_norm_scale=1, second_order_scale=1, fourth_order_scale=1` | **NO — conservative.** `u_xx`, `u_xxxx` are derivatives and `½(u_x)²`… in the `gradient_norm` form the mean is *not* exactly conserved, but in the `−u u_x` convective form it is. Exponax's `KuramotoSivashinsky` uses the **gradient-norm** form, so the mean can drift; it is therefore **not** skipped a priori — it is run and reported, with the caveat that on `L = 1` every KS mode has growth rate `(2πn)² − (2πn)⁴ < 0`, so the operator is strongly damping and the drift is expected to be tiny. |
| `S1_hyperdiff` | Hyperdiffusion | `hyper_diffusivity=1e-4` | **NO — conservative *and* commuting.** `−ν₄ u_xxxx` is an exact derivative (mean fixed) *and* it is diagonal in Fourier, so it commutes exactly with `diffusion(0.01)`: the order-dependence gate must return the commuting null. **Skipped from the ranking with this reason; still integrated once to confirm both predictions.** |

### 6.2 Stage 2 — first-operator diffusivity, applied by a mechanical rule

For the **top three** stage-1 candidates by `proxy_dynamic_range` **among those that pass
G1 (`overlap == 0`) and the order-dependence floor**, re-run with
`A = diffusion(0.005)` and `A = diffusion(0.02)` (`0.01` is already measured in stage 1).
Ties broken by candidate id, ascending, as `screen_stability.py` rule 5 breaks ties.
These rows are `EXTRAPOLATED` (section 5): a different first operator means a different
on-support task.

### 6.3 Stage 3 — non-diffusion first operators, B = Fisher-KPP(`diffusivity=0.0, reactivity=1.0`)

| id | A | dt | steps/leg | k | note |
|---|---|---|---|---|---|
| `S3_burgers_first_dt0.0025` | Burgers(`diffusivity=0.01`) | 0.0025 | **400** | **40** | **primary** for this regime: τ_leg = 1.00 and τ_estimand = 0.10, physically identical to the package |
| `S3_burgers_first_dt0.0025_k10` | same | 0.0025 | 100 | 10 | secondary: the literal "100 steps/leg, k=10" reading, τ_leg = 0.25 |
| `S3_burgers_first_dt0.01` | same | 0.01 | 100 | 10 | expected CFL failure, carried to document the regime boundary |
| `S3_hyperdiff_first` | HyperDiffusion(`1e-4`) | 0.01 | 100 | 10 | hyperdiffusion as the *learned lane*; conservative first operators are fine there, only B must shift |
| `S3_ks_first` | Kuramoto-Sivashinsky | 0.01 | 100 | 10 | KS as the learned lane |

All stage-3 rows are `EXTRAPOLATED`.

**Advective CFL**, for any candidate carrying a convection or gradient-norm term:
`cfl = max|u| · dt / dx` with `dx = L/N = 1/256` and `max|u|` the maximum over the composed
trajectory actually integrated. `0.0` for candidates with no such term. Limit `1.0`,
verbatim `screen_stability.py` rule 3. (`screen_stability.py` fixed advection velocity at
`0.25`; Burgers and KS have no fixed velocity, so the measured `max|u|` is used, which is
the same quantity the recorded Burgers apparatus stop reported as 2.06.)

## 7. Broad-interval derivation (so a passing pair can be rerun directly)

Verbatim `family1_second_pairs/PREDECLARED.md` §4, for an **upward** shift:

1. lower bound `0.48`;
2. upper bound `u = ceil_{0.05}(max switch-state spatial mean) + 0.05`;
3. mandatory coverage check: build the **640-unit** broad corpus under the candidate's
   first operator and require the broad training-mean support to contain **100 %** of the
   256 switch-state means. If not, `u += 0.05` and re-check, up to `1.00`, recording every
   step. If nothing at or below `1.00` closes it, the broad condition cannot be built for
   that candidate; that is reported as-is.

**Sample count is fixed at 640 training units in both conditions. Only the support moves.**

**New, declared here** (family 1's rule assumes an upward shift and cannot express a
downward one): for a **downward** shift the rule is mirrored —
upper bound fixed at `0.52` (`ceil_{0.05}(0.5171)`), lower bound
`l = floor_{0.05}(min switch-state spatial mean) − 0.05`, clamped at `0.0`, same mandatory
100 % coverage check stepping `l -= 0.05` down to `0.00`. Every row derived by the mirrored
rule is stamped `MIRRORED_RULE` and is reported as a family-1b extension, not as
family 1's rule.

## 8. Gates — a candidate PASSES the screen iff all four hold

| gate | definition | threshold |
|---|---|---|
| **G1 shift exists** | `overlap == 0.0` **and** `abs(gap) >= 0.05` | zero overlap, gap ≥ 0.05 |
| **G2 proxy resolves** | `proxy_dynamic_range >= 100.0` **and** `e_on_ref <= 0.1 * persistence_on` | 100.0, and the 0.1 fraction |
| **G5 genuinely noncommuting** | `order_dependence = max abs(compose(A,B,u0)[:,-1] − compose(B,A,u0)[:,-1])`, 32 units, float64 | `>= 1e-6` |
| **G6 stable** | all values finite in both orderings; composed `max|u| <= 10 ×` initial `max|u|`; `cfl <= 1.0` | `screen_stability.py` rules 1–3 |

**Strength label** (on a candidate that passes all four):
`STRONG` if `proxy_dynamic_range >= 150`; `MARGINAL` if `100 <= proxy_dynamic_range < 150`.

`G1`'s `gap >= 0.05` clause is **new to family 1b** and is stricter than family 1's
`overlap == 0.0`. Declared because family 1's G1 admits a shift of arbitrarily small size
(Swift-Hohenberg's package coefficients give `overlap = 0.0051` at a drift of only 0.025)
and a hair's-breadth shift is not a support shift worth rerunning. P1's gap is 0.174 and
P2's is 0.292, so both known pairs clear 0.05 with room.

The numerical floor `G4_MIN_E_ON = 1.19e-5` and `P6_MAX_E_ON = 1.573e-3` are learner-side
gates. Family 1b fits nothing and therefore **cannot** evaluate them; they are recorded as
`not evaluable without a learner` for every candidate and are not part of the screen's
verdict. `e_on_ref = 1.92e-4` sits 16× above the floor and 8× below the convergence limit,
which is the only thing the screen can say about them.

## 9. Screen-validation gate (binding, checked before any candidate is ranked)

The screen must reproduce family 1's own published truth-side numbers:

- `S1_fkpp_r1.0` must give `persistence_off = 0.08019073763412352` and
  `persistence_on = 0.12143668541229405`;
- `S1_allen_cahn` must give `persistence_off = 0.005920252665176344`;
- both to within a **relative** tolerance of `1e-6`;
- and the resulting labels must be **P1 → passes G2 proxy** and **P2 → fails G2 proxy**.

If this fails, the screen is wired wrong; `RESULT.md` records the failure and **no
candidate ranking is published**. No retuning, no retry with different settings.

## 10. What this screen does not and cannot do

It fits nothing, so it says nothing about `e_off`, `R`, degradation, broadening, or
whether family 1's result reproduces on any candidate. A candidate that passes here is a
candidate on which the experiment is *measurable*, not one on which it *works*. The proxy
is a 2.5 %-accurate stand-in on the single pair where it can be checked, and an
unverified extrapolation on every row stamped `EXTRAPOLATED`. One architecture, one
resolution, one IC family, one `k` per regime. Passing this screen does not relax any
family-1 gate: family 1 still computes the real `G2` from a real `median_e_on`, and a
candidate that passes here can still fail there.

## 11. Derived outputs (filled in after running; rules above are what produced them)

Run 2026-09-11T08:09:03Z – 08:14:39Z, total 5.6 minutes of single-threaded wall time.

- `RESULT.json`, `RESULT.md` — the ranked table, the closest misses with their binding
  gates, and the broad intervals. `RESULT.md` is generated from `RESULT.json` by
  `screen_pairs.py report`; no number in it is transcribed by hand.
- `results/GEOM_stage{1,2,3}.json` — supports, persistences, proxies, stability (float32).
- `results/ORDER_stage{1,2,3}.json` — order dependence and the x64 stability rules.
- `results/STAGE2_GRID.json` — §6.2's mechanical top-three rule and the grid it produced:
  `S1_grayscott_1sp`, `S1_fkpp_r0.5`, `S1_fkpp_r1.0`.
- `results/SCREEN_INTERVALS.json` — the §7 broad-interval searches, every step recorded.
- `logs/` — one log per stage.

**§9 screen-validation gate: PASS.** All three published truth-side numbers reproduced at
relative difference `0.000e+00`; P1 passes the G2 proxy and P2 fails it, as required.

**Outcome: 25 configurations, 12 pass all four gates, 10 fail, 3 skipped a priori as
conservative and/or commuting (all three confirmed by measurement).** Nothing in sections
1–9 was changed after the numbers were read; in particular the `gap >= 0.05` clause of §8
was not relaxed when it turned out to be the binding gate on the three highest-proxy
candidates in the grid (the Nagumo family). That is recorded in `RESULT.md` §D.1.
