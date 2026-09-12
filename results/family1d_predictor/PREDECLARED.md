# PREDECLARED — Family 1d: the out-of-basis residual as a degradation predictor, tested out of sample

Written **2026-09-11T11:46Z** (04:46 PDT), before any number belonging to this family was
computed: before any PCA basis was fitted here, before any residual was evaluated here,
before the `S1_fkpp_r0.5` switch cohort was built, and before any model was fitted here.

Release rules in force: predeclare everything; **never change a gate after seeing a
number**; preserve failures as-is; report the honest number first.
`dependency`, `release-inputs/dependency`,
`release-inputs/family1_second_pairs`, `release-inputs/family1b_pair_screen`,
`release-inputs/family1c_state_support`, `release-inputs/family6_serrano_baseline` and
`release-inputs/family1_rerun` are **read-only** — imported from and read, never written to.
`family1_rerun/` in particular is a **live detached driver** (PID 89748, launched
2026-09-11T09:17Z); its files are polled and read, never touched.
All output of this family goes to `release-inputs/family1d_predictor/` and
`work/family1d/`.

---

## 0. Why this family exists, and what is already known to its author

Family 1c's predeclared state-support statistic **S** (PCA-99 Mahalanobis distance) failed
both of its prediction gates: `rho(log10 R, D_bar) = -0.257` over six cells against a
required `+0.8`, and the Cahn-Hilliard narrow cell came out INSIDE the PCA-99 radius when
the gate required OUTSIDE. After S had been read and the verdict written, a **post-hoc**
out-of-basis residual statistic was computed (`family1c_state_support/diagnostic.py` ->
`results/DIAGNOSTIC.json`, self-labelled "POST-HOC; gates nothing") and gave
`rho = +0.829` over the same six cells. Family 1c recorded it as *"a lead, not a result"*.

This family predeclares that statistic as **S3** and tests it on cells that were **not**
used to find it.

### 0.1 Full disclosure of what the author of this file had already read

The task spec for this family instructed its author to read
`family1c_state_support/results/DIAGNOSTIC.json`, `PREDECLARED.md` and `RESULT.md` before
predeclaring, in order to copy S3's exact definition and the list of the six selection
cells. That instruction was followed, so the following numbers **were visible before this
predeclaration was written** and are recorded here so that no later reader has to guess
what was known:

- `D3` and `C3` for all eight cells in `DIAGNOSTIC.json`, **including the two BROAD-S
  cells** used below as held-out cells 1 and 2 (`CH_broadS`: D3 = 0.28735827,
  C3 = 0.87109375; `P1_broadS`: D3 = 0.27963861, C3 = 0.0859375).
- `R` for those two cells from `family1c_state_support/RESULT.json` / `RESULT.md`
  (CH BROAD-S ten-seed median 0.787; P1 BROAD-S ten-seed median 2.569).
- `R` for family 6's two smoke cells from
  `family6_serrano_baseline/results/RESULT_SMOKE.json`
  (`narrow_oracle.R_frame0_bridge` 90.68072674, `broad_oracle.R_frame0_bridge`
  18.38459238), and the fact that family 6's `narrow`/`broad` training corpora are built
  by the same call as family 1c's, so those two cells' **D3 is already known too** — it is
  by construction identical to `P1_narrow`'s (3.31308429) and `P1_broad`'s (0.27970883).

So of the six held-out cells declared in section 2, **four have values that were already
visible to this file's author** and only the two cells in section 2(d) are blind. This is
stated up front, it is not repaired by anything below, and section 3 declares a **second,
blind-only** correlation that is reported alongside the predeclared gate precisely so the
reader can see how much of the gate's outcome was already fixed. The gate itself is the
one in section 3 and is **not** re-defined after any number is seen.

What was *not* visible and is genuinely blind at the time of writing: `D3`, `C3`,
`D_bar` and `R` for the two `S1_fkpp_r0.5` cells (section 2(d)); every number produced by
the live family-1 rerun; and the recomputation of every quantity above by this family's
own code, which is done independently and cross-checked against the 1c files.

---

## 1. The statistic S3, copied verbatim from family 1c

S3 is **exactly** `switch_residual_mean` as computed by
`family1c_state_support/diagnostic.py` (lines 40-64), reproduced here in full so that the
definition is fixed by this file and not by a file this family may not modify:

For a `(pair, condition)` cell:

1. **Training cloud.** The condition's 640 training units built exactly as the package
   builds them, flattened to `640 x 101 = 64 640` states in `R^256`, cast to **float64**:
   - mean-offset conditions: `ic = tp.initial_conditions(640, offsets, MASTER_SEED + 1)`,
     `states = tp.rollout(diffusion, ic, 100)`;
   - `broadS` conditions: family 1c `PREDECLARED.md` section 4.1's sampler — 640 base ICs
     at the narrow offsets with `MASTER_SEED + 1`, four index blocks of 160 given bursts
     of the pair's second operator of length `m = 0, 25, 50, 100`, then rolled 100 steps
     under diffusion.
2. **Basis.** `mu` = mean of the 64 640 training states; `xc = x - mu`;
   `C = (xc^T xc) / (n - 1)`; symmetric eigendecomposition, eigenvalues sorted
   descending; **`k` = `searchsorted(cumsum(lambda)/sum(lambda), 0.99) + 1`**, i.e. the
   smallest number of leading components carrying at least 99 % of the variance.
   `U = ` the first `k` eigenvectors. `k` is measured per cell, never fixed.
3. **Residual of a state.** `Q(x) = || (x - mu) - U U^T (x - mu) ||_2`. Note: this is the
   **unnormalised, mean-centred** residual norm, which is the exact form family 1c used.
   It is *not* divided by `||x||`; the task spec's alternative normalised form is
   declared as the secondary `D3f` below and is reported, not gated.
4. **Switch states.** The cell's 256 switch states:
   `tp.initial_conditions(256, NARROW, MASTER_SEED + 8)` evolved 100 steps under the
   pair's second operator, frame 100 — byte-identical to the cohort the package and
   family 1 evaluate `e_off` on.
5. **Cell statistics.**
   - **`D3` = mean of `Q` over the 256 switch states** (`switch_residual_mean`). This is
     the predictor the gate is written on.
   - **`C3` = fraction of the 256 switch states with `Q <= q99`**, where `q99` is the
     99th percentile (`numpy.percentile`, linear interpolation) of `Q` over the cell's
     own 64 640 **training** states (`coverage_inside_train_residual_p99`). This is the
     intervention-side coverage.
   - Reported, not gated: `D3f` = median over the switch states of `Q(x) / ||x - mu||`
     (`switch_fraction_of_deviation_outside_basis_median`), and
     `Q_med/q99` (`switch_residual_over_train_p99_median`).

### 1.1 Identification of S3 among DIAGNOSTIC.json's fields

`DIAGNOSTIC.json` records four candidate per-cell numbers. The one this family adopts is
fixed by two independent checks, both made **before** this file was written and both
reading only already-published values:

- family 1c's `RESULT.md` reports `rho = +0.829` for "a residual (out-of-basis) norm"
  over the six cells. Ranking the six selection cells by `switch_residual_mean`
  (P2_n 5.2468 > P1_n 3.3131 > P1_b 0.27971 > CH_n 0.23047 > CH_b 0.17635 > P2_b 0.01214)
  against `log10 R` (P2_n 2.248 > P1_n 2.042 > CH_n 1.665 > CH_b 1.554 > P1_b 1.247 >
  P2_b -0.999) gives `sum d^2 = 6` and `rho = 1 - 36/210 = +0.8286`. It reproduces.
- The task spec defines D3 as a **mean** over the 256 switch states.
  `switch_residual_mean` is the only mean among the candidates.

(`switch_fraction_of_deviation_outside_basis_median` happens to induce the *same* six-cell
ranking and therefore the same `rho`; it is therefore carried as the declared secondary
`D3f` and cannot be used to break a tie in S3's favour later.)

### 1.2 The six selection cells — reported, never gated

The six cells S3 was found on are `(pair, condition)` for `pair` in `{P1, P2, CH}` and
`condition` in `{narrow, broad}`. They are recomputed and tabulated in the result **for
contrast only**. They contribute nothing to any gate in section 3.

---

## 2. The held-out cells

A cell qualifies as held out iff it is not one of the six in section 1.2. Six cells are
declared, in the priority order the task spec gives.

**(a) The live family-1 rerun.** `S1_fkpp_r0.5` and `S1_grayscott_1sp`, narrow and broad,
*once `family1_rerun/RESULT_<pair>.json` or `RESULT.json` exists*. The rerun is polled
every 5 minutes for up to 4 hours from the start of this family. At the time of writing it
has completed 5 of its 40 fits at roughly 26 min/fit, so it is **not expected to produce a
RESULT file inside the window**; any cell that does appear is added to the held-out set
and the gate is recomputed with it. Nothing in `family1_rerun/` is written.

**(b) Family 1c's BROAD-S cells — 2 cells, no new fits.**
- `CH_broadS`: `R` = the ten-seed median of `e_off/e_on` over
  `family1c_state_support/results/broadS/CH_seed*.json`.
- `P1_broadS`: the same over `P1_seed*.json`.

These are held out from the **six-cell correlation that selected S3**: family 1c's
`RESULT.md` states the `rho = +0.829` was computed "on the six cells", and the six cells
are the ones listed in section 1.2. They are **not** blind — see section 0.1 — because
`DIAGNOSTIC.json` also carries their `D3`/`C3`. One ordering property does hold in their
favour and is checked in the result: `DIAGNOSTIC.json` was written at 08:52 UTC, whereas
family 1c's `broadS/GATES.json` is stamped 11:33:03 UTC and the first BROAD-S `e_off` was
read at 11:33:46 UTC, so **`D3` and `C3` for these two cells were computed before their
`R` existed.**

**(c) Family 6's smoke fits — 2 cells, no new fits.** `family6_serrano_baseline`'s
`narrow_oracle` and `broad_oracle` arms are an FNO A1 trained on the package's narrow /
broad diffusion corpus (`serrano_splitting.py:fit`, offsets `rb.NARROW` / `rb.BROAD`,
`MASTER_SEED + 1`, 640 units) and evaluated against a Fisher-KPP switch cohort
(`MASTER_SEED + 8`), 48 test units, 2 seeds. Checkpoints exist under
`family6_serrano_baseline/ckpt/`.
- `R` is taken as **`R_frame0_bridge`** (median over the 2 seeds), because family 6's
  `e_on_frame0_bridge` uses `on_traj[:, 0]` — the *same* on-support state family 1 and
  family 1c use (`rollout(diffusion, on_ic, 100)[:, 0]`), whereas its headline
  `e_on` uses frame 100 and is therefore a different metric. The headline-convention `R`
  is reported beside it; **both conventions induce the same rank order on these two
  cells, so no gate can turn on the choice.**
- **Known limitation, declared now:** these two cells' training clouds are *identical* to
  the six selection cells `P1_narrow` and `P1_broad`, so their `D3`, `C3` and `D_bar` are
  identical too; only `R` differs (48 units / 2 seeds instead of 256 units / 5 seeds).
  They are therefore **duplicate points in the predictor** and add no independent
  information about `D3`. They are included because the task spec names them, and the
  result reports the gate **with and without** them.

**(d) Two additional cells, blind — the only new fits in this family.** The held-out set
after (a)-(c) has 4 cells, fewer than the 6 the gate requires, which is the condition the
task spec attaches to new fits. Declared, at most 5 fitted seeds each:

- **`FK05_narrow`** — pair `S1_fkpp_r0.5` (second operator
  `ex.stepper.reaction.FisherKPP(..., diffusivity=0.0, reactivity=0.5)`, exactly as
  `family1_rerun/run_family1_rerun.py:second_operator` and
  `family1b_pair_screen/screen_pairs.py` define it), narrow training offsets
  `(0.4865, 0.5171)`. **Zero new fits:** the narrow training corpus does not depend on the
  second operator, and family 1's ten narrow checkpoints were verified byte-identical
  across pairs (`family1_second_pairs/reuse_narrow.py`, re-verified by sha256 in family
  1c's `eval_banked`). `R` = ten-seed median of `e_off/e_on` obtained by evaluating those
  ten banked checkpoints against the `S1_fkpp_r0.5` switch cohort. The sha256 identity
  check is repeated here and the run aborts if it fails.
- **`FK05_broad`** — the same pair, broad offsets **`(0.48, 0.70)`**, taken verbatim from
  `family1b_pair_screen/RESULT.json` (`intervals.S1_fkpp_r0.5.broad_offsets`, derived by
  family 1's own upward rule). **5 new fits, seeds 0-4**, architecture / optimiser /
  schedule / cohorts / `n = 640` identical to the package's, `rb.train` called unchanged.

Why this pair and not "P1 at a third offset interval": `S1_fkpp_r0.5` is a *new second
operator* with a switch-state support `(0.5941, 0.6244)` no fitted cell has yet seen, so
it supplies genuinely new `D3` and `R`, whereas a third P1 mean interval reuses P1's
switch cohort. It is also cell (a)'s own pair A, so when the rerun finishes the reader gets
a ten-seed check of both numbers declared here. The screen already establishes that the
metric resolves on this pair (`overlap = 0.0`, `persistence_off = 0.10154`,
`proxy_dynamic_range = 528.2`), so the cell is not expected to be censored; if it is, it
is reported censored and flagged, exactly as P2 is.

**Ordering discipline for (d).** Every `FK05_broad` seed writes `e_on` and its checkpoint
to a `*_pregate.json` first. `results/heldout/FK05_GATES.json` — carrying the truth-only
geometry (support overlap, persistence on/off) and the P6 convergence check on the worst
`e_on` — is written **before any `e_off` is read**, for both FK05 cells. Only then is
`e_off` read. If the worst `e_on` exceeds the package's `P6_MAX_E_ON = 1.573e-3`, the
cells are reported as non-convergent and excluded from the gate.

---

## 3. Gates

### PRED-S3 — the family's gate. Both halves must hold.

| # | gate | threshold |
|---|---|---|
| **PRED-S3-A** | Spearman `rho(log10 R, D3)` over the **held-out** cells of section 2 | `>= 0.8`, with **at least 6 cells** |
| **PRED-S3-B** | `C3` is monotone in the intervention: `C3(broad-S) > C3(narrow)` for **both** CH and P1 | strict inequality, both pairs |

`log10 R` uses each cell's `R` as defined in section 2. Spearman is
`scipy.stats.spearmanr`, average ranks on ties.

If fewer than 6 held-out cells are available when the family reports, **PRED-S3-A fails on
the cell-count clause** and the family is marked PARTIAL, not PASS.

### Reported alongside, gating nothing

1. The six selection cells (section 1.2) with their `R`, `D3`, `C3`, `D_bar` and the
   six-cell `rho`, to confirm the `+0.829` reproduces.
2. **The blind-subset correlation.** `rho(log10 R, D3)` restricted to the cells whose `R`
   and `D3` were *not* visible when this file was written — i.e. section 2(d), plus any
   section 2(a) cell that lands. With only 2-6 such cells this is a weak number and is
   reported as such; it exists so that section 0.1's disclosure is quantified rather than
   merely admitted.
3. **The de-duplicated correlation.** `rho(log10 R, D3)` over the held-out cells with
   family 6's two cells removed (they duplicate `P1_narrow`/`P1_broad` in `D3`).
4. **The PCA-99 Mahalanobis statistic `D_bar`** — family 1c's failed statistic S — on the
   same held-out cells, with its own Spearman, for contrast. Same basis, same `k`, same
   switch cohort; `D_bar` = mean Mahalanobis distance of the 256 switch states,
   `coverage_S` = fraction inside the 99th-percentile training Mahalanobis radius.
5. `D3f` (the normalised residual fraction) and its Spearman, on both cell sets.
6. Spearman p-values. With `n = 6` no p-value below 0.05 is attainable for `rho` under
   0.83; this is stated in the result rather than discovered there.

### Verdict rule

**PASS** iff PRED-S3-A and PRED-S3-B both hold. **FAIL** otherwise. **PARTIAL** is
appended to the verdict iff the family reports before the 4-hour rerun window closes with
any declared cell still missing, or if fewer than 6 held-out cells exist. No statistic is
substituted for S3 after a number is seen, and no cell is dropped from the held-out set
after its numbers are seen.

---

## 4. Environment and provenance

- `.venv/bin/python`; `source environment setup script`.
- **Private** `JAX_COMPILATION_CACHE_DIR=work/family1d/jax_cache` (the shared
  cache in `ENV.sh` holds AOT entries written by another lane and XLA refuses them with a
  SIGILL warning — family 1c hit this and did the same).
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`; at most 3 concurrent single-threaded
  workers, because the box is shared with the live family-1 rerun.
- The shipped harness is **imported, never copied**: `run_baseline.py` and
  `timing_probe.py` from `dependency`; `second_operator`, `save_params`, `load_params`
  from `family1_second_pairs/run_family1.py`; the corpus builders, BROAD-S sampler and
  evaluation cohorts from `family1c_state_support/run_family1c.py`. No function that
  writes into any of those trees is called.
- Geometry in float64; fits in float32 (`run_baseline.py` does not set `jax_enable_x64`).
- Checkpoints and large arrays live in `work/family1d/`.
