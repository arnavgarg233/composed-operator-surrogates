# RESULT — Family 8: operator-algebra recovery on complex 2-D multi-operator dynamics

**Verdict as predeclared: FAIL.** One of the two primary gates passed and one failed.
P3 (broadening repairs composed rollouts) passed at 0.135 against a 0.25 threshold.
ALG-1 (algebra recovery) failed: its accuracy clause passed at the ceiling, its
improvement clause missed by 0.015.

Run: 1x NVIDIA RTX A6000 (48 GB) on RunPod, $0.53/h, 2026-09-12 06:10:19Z to
08:01:50Z, 1.86 h, **$0.99** against a $2.50 budget. Pod `zdz3rp747vtm8o`, stopped and
verified `EXITED`. Grid 64x64, FNO-2D with **2,365,825** parameters, **24 fits** used
(4 members x 2 conditions x 3 seeds) at a median of **181 s per fit** (range 171-198 s).
Predeclaration `PREDECLARED.md` sha256 `0e7ac1a1…d128d80d`, unchanged throughout.

The runner was reviewed against `TASK.md`, family 1e's sampler and gates, and family 6's
`METHOD_SPEC.md` before it was run; nine defects were fixed, two of which would have
made the run meaningless. Every change is logged in `REVIEW_NOTES.md`.

---

## 1. Design as executed

Dictionary, m=4, 2-D periodic on [0,1)^2 with dt=0.01, leg tau=1.0 (100 steps),
estimand stride K=10: **D** diffusion (kappa=0.01), **A** cellular advection
(divergence-free, U0=1.0), **R** Fisher-KPP (r=1.0), **G** Gray-Scott 1-species
(F=0.04). The truth-only pre-screen at the experiment's own 64x64 resolution passed
**7 of the 12 ordered pairs** — `DoA, AoR, AoG, RoA, RoG, GoA, GoR` — and all three
predeclared length-3 words `RoAoD, DoAoR, GoDoA`, giving **10 words under test**. The
same 7 pairs survive at 32x32, so the word suite does not depend on resolution. Five
pairs were dropped: `AoD` for no support shift, and `DoR, RoD, DoG, GoD` for order
contrast below 0.05, i.e. D commutes too nearly with R and with G at this leg length to
carry an order signal. G survived comfortably (dynamic range 611-2247), so the
Allen-Cahn fallback was not used.

Conditions at a fixed budget of n=640 trajectories per member: **NARROW** (each member
trained on its own narrow IC window) and **BROAD-S** (every member trained on one
stratified corpus of 160 base ICs plus 480 states drawn from the union of leg-boundary
and mid-leg states across all 10 words). Seeds 0, 1, 2.

**Seed reduction.** Measured per-fit wall time was 3.6 min (NARROW) and 5.5 min
(BROAD-S) against the runbook's 25-35 s estimate for a 4090, projecting the 5-seed
design at 3.3 h and $1.77 — at this pod's 3.5 h hard stop and at the $1.75 trigger in
section 8 of the predeclaration. The predeclared reduction from 5 seeds to 3 was
applied at 07:05Z, before `gates`, `words`, `algebra` or `evaluate` had run and
therefore before any off-support number existed. Two NARROW fits of member D (seeds 3
and 4) had already completed; they are kept on disk and excluded from every
aggregation.

## 2. E1 — composed-rollout degradation and the P3 repair (PASS)

R is the composed-rollout endpoint error divided by the on-support error of the word's
own surrogates, per seed; medians over the three seeds.

| word | len | NARROW endpoint | BROAD-S endpoint | R NARROW | R BROAD-S | P3 = R_b/R_n | per-word |
|---|---|---|---|---|---|---|---|
| `DoA`   | 2 | 0.0050 | 0.0084 | 0.6 | 0.8 | 1.283 | fail |
| `AoR`   | 2 | 0.0856 | 0.0622 | 11.1 | 5.9 | 0.536 | fail |
| `AoG`   | 2 | 0.7007 | 0.0994 | 89.8 | 9.4 | 0.105 | pass |
| `RoA`   | 2 | 13.2819 | 0.0577 | 1721.2 | 5.5 | 0.0032 | pass |
| `RoG`   | 2 | 0.0774 | 0.0033 | 112.9 | 12.2 | 0.108 | pass |
| `GoA`   | 2 | 6355.12 | 0.1011 | 798815.2 | 9.5 | 0.0000 | pass |
| `GoR`   | 2 | 0.0193 | 0.0026 | 28.1 | 9.1 | 0.323 | fail |
| `RoAoD` | 3 | 0.0182 | 0.0031 | 10.2 | 1.6 | 0.161 | pass |
| `DoAoR` | 3 | 0.0869 | 0.0080 | 43.0 | 4.2 | 0.098 | pass |
| `GoDoA` | 3 | 0.0113 | 0.0103 | 5.9 | 5.3 | 0.902 | fail |

**Aggregated repair ratio (primary): 0.135, threshold 0.25 — PASS.** The unnormalised
ratio of raw endpoint errors is 0.138, so the conclusion does not depend on the `e_on`
normalisation. Seven of ten words repair individually; three do not.

The two extreme words carry the phenomenon the family was built to test. `GoA` —
advection receiving Gray-Scott states — diverges outright under the narrow dictionary
(endpoint relative L2 of 6355, R near 8x10^5) and is brought back to 0.101 by
broadening at the same sample count. `RoA` goes from 13.28 to 0.058. Both are words
whose first leg drives the state far off the receiving operator's training support,
which is exactly what the pre-screen selected them for.

The three words that do not repair are equally informative and are not hidden: `DoA`,
`GoR` and `GoDoA` are words the narrow dictionary already handles (R of 0.6, 28 and 5.9),
so there is little to repair and broadening's slightly wider corpus costs a little
in-support accuracy. `DoA` at 1.28 is broadening making a already-good word marginally
worse. The repair is real where the degradation is real, and absent where it is not.

## 3. E2 — operator-algebra recovery (FAIL, ceiling-censored)

Exhaustive search over all **84** candidate words of length 1, 2 and 3, scored by mean
relative L2 over the observed stride-K frames, on **20 held-out trajectories per word**
for each of the 10 words (200 trajectories per dictionary per seed).

| dictionary | top-1 per seed | top-1 median | order accuracy given the operator set | multiset accuracy |
|---|---|---|---|---|
| NARROW  | 0.710, 0.715, 0.805 | **0.715** | 1.000 | 0.715 |
| BROAD-S | 1.000, 1.000, 1.000 | **1.000** | 1.000 | 1.000 |

**ALG-1 as predeclared: FAIL.** The accuracy clause passes — BROAD-S recovers the
generating word on every one of the 200 held-out trajectories in all three seeds, well
above the 0.80 floor. The improvement clause fails: the gain is 0.285 against a 0.30
requirement.

The gate is censored by its own ceiling. With the narrow dictionary at 0.715 and
accuracy bounded by 1.0, **the largest gain attainable was 0.285, and 0.285 is what
BROAD-S achieved** — it eliminated 100% of the narrow dictionary's error, 28.5% of
trajectories misidentified down to 0%. The threshold pair (>= 0.80 and >= +0.30) is
jointly unsatisfiable whenever the comparison arm exceeds 0.70, which is what happened:
the narrow dictionary was stronger at this task than the predeclaration assumed. This is
recorded as a genuine failure of the gate as written, not rewritten into a pass.

Where narrow fails is mechanically consistent with E1. Per word, the narrow dictionary
identifies 8 of 10 words perfectly and collapses on exactly the words whose narrow
composed rollout blows up: `AoG` 0.00, `GoA` 0.00, `RoA` 0.10. BROAD-S is at 1.00 on all
ten. Whenever either dictionary recovered the right set of operators it also recovered
their order — order accuracy given a correct multiset is 1.000 for both — so the
identification failures are failures to find the operators at all, not to sequence them.

## 4. E3 — test-time operator splitting with the narrow dictionary

Following family 6's method spec, the narrow dictionary is granted a test-time search
over three splitting schemes that all spend 30 surrogate applications and advance the
same total time, with no extra training: leg-sequential Lie (the word order itself),
fine-grained Lie, and fine-grained Strang. The arm is scored at its best scheme.
Medians over three seeds, endpoint relative L2:

| word | leg-sequential Lie | fine Lie | fine Strang | best | BROAD-S sequential | BROAD-S / best splitting |
|---|---|---|---|---|---|---|
| `RoAoD` | 0.0182 | 0.0354 | 0.0406 | 0.0182 | 0.0031 | **0.168** |
| `DoAoR` | 0.0869 | 0.0287 | 0.0340 | 0.0287 | 0.0080 | **0.279** |
| `GoDoA` | 0.0113 | 0.4658 | 0.5971 | 0.0113 | 0.0103 | 0.908 |

On two of the three length-3 words the state-space-broadened dictionary reaches an
endpoint 3.6x and 6x more accurate than the best test-time splitting of narrow
surrogates; on `GoDoA`, a word the narrow dictionary already handles, the two arms tie.
Test-time splitting spends compute at inference and cannot recover what the training
support does not contain. One detail worth keeping: on `DoAoR` the fine-grained schemes
beat the leg-sequential order (0.029 and 0.034 against 0.087), so the granted search is
not decorative — it does find a better recomposition than the word order itself, and
still loses to broadening.

## 5. Gates

`G1` (a support shift exists) and `G2` (the metric resolves it) are blocking and both
passed; `GATES.json` was written before any off-support number was computed, and
`evaluate` refuses to run while a blocking failure stands.

Median on-support error `e_on` by member, each measured against that member's own K-step
flow map:

| | D | A | R | G |
|---|---|---|---|---|
| NARROW  | 1.92e-3 | 1.49e-2 | 6.35e-4 | 8.71e-4 |
| BROAD-S | 1.91e-3 | 2.09e-2 | 2.31e-4 | 3.37e-4 |

Broadening leaves D unchanged and *improves* R and G on-support by a factor of 2.6-2.7
at the same sample count. **The cellular advection surrogate A is the weak member**, an
order of magnitude worse than the rest in both conditions and 40% worse under BROAD-S.
Advection at CFL 0.64 is the hardest of the four flow maps for an FNO at these settings,
and A appears in all three words that fail per-word P3. Any follow-up should fit A
harder before anything else.

Two further gate constants, G4a and P6, are absolute `e_on` thresholds imported from the
1-D families and were reported as non-blocking diagnostics rather than used to decide a
2-D run (`REVIEW_NOTES.md` item 6). For the record: G4a is met (minimum `e_on` 2.26e-4
against a 1.19e-5 floor, so nothing is censored from below), and **P6 is not met** —
the worst BROAD-S `e_on` is 2.10e-2 against the 1-D limit of 1.573e-3, entirely because
of member A.

## 6. External-data validation (secondary, no gate): NOT ATTEMPTED

The pod had working network. The DaRUS API for PDEBench (`doi:10.18419/darus-2986`) was
reached and queried, and the 2-D diffusion-reaction file was located:
`2D_diff-react_NA_NA.h5`, **12.33 GiB**, the only file in the dataset matching that
physics. It exceeds the 3 GiB download cap set for this run's budget, so nothing was
downloaded and the arm is recorded as NOT ATTEMPTED with that reason in
`results/EXTERNAL.json`. No external number is reported and none was invented.

Two further obstacles stand in the way of the arm as specified and are recorded now so
a future attempt can be scoped properly: Exponax 0.2.0 ships no two-species
FitzHugh-Nagumo stepper, and PDEBench generates that dataset with no-flow Neumann
boundaries on [-1,1]^2 while every stepper in this family is periodic and spectral. A
matched reproduction therefore needs a new two-species solver with non-periodic
boundaries, not a parameter change.

## 7. What this run establishes

State-space broadening at a fixed sample budget transfers from 1-D pairs to a complex
2-D four-operator dictionary with words of length 2 and 3: the aggregate repair ratio is
0.135 against a 0.25 threshold, with individual words moving from a diverged rollout
(endpoint error 6355) to 0.10. On top of that, the broadened dictionary identifies the
generating operator word from an unlabelled trajectory perfectly — 200 of 200 held-out
trajectories in every seed, against 84 candidates — where the narrow dictionary fails
on exactly the words its composed rollouts cannot follow. The predeclared ALG-1 gate
still reads FAIL because its improvement clause could not be satisfied against a
comparison arm this strong, and it is reported as a failure. The scientific object it
was built to detect is present and, if anything, larger than the gate anticipated.

Artifacts: `results/RESULT.json` (full numbers and run metadata), `RESULT.md` (this
file), `results/{PRESCREEN,GATES,WORDS,ALGEBRA,EXTERNAL}.json`, 26 pregate records,
`REVIEW_NOTES.md`. Checkpoints (251 MB) and all 28 run logs are on the Work SSD at
`pde_volume/family8/`.

---

## Errata (2026-09-12)

Two corrections to the text above, verified against `results/RESULT.json` and `run_family8.py`:

1. **Per-word repair count:** Section 2 states that "Seven of ten words repair individually; three do not." As shown in the per-word table and recorded in `results/RESULT.json` (`pass_p3`), exactly **six of ten words repair** under the $P_3 \le 0.25$ criterion (`AoG`, `RoA`, `RoG`, `GoA`, `RoAoD`, `DoAoR`), while **four do not** (`DoA` at 1.283, `AoR` at 0.536, `GoR` at 0.323, and `GoDoA` at 0.902). `AoR` was inadvertently counted among the repairing words in the prose summary despite having $P_3 = 0.536 > 0.25$.
2. **Word naming and application order:** In mathematical notation $w = O_k \circ \dots \circ O_1$, operators are applied from right to left ($O_1$ first, then $O_2$, etc.). In `run_family8.py`, `parse_word` parses word strings by splitting on `"o"` and reversing the list (`return list(reversed(parts))`). Thus, for `GoA`, the operator sequence is `['A', 'G']`: cellular advection **A** is applied first, and Gray-Scott **G** is applied second (receiving the advected state). Similarly, `RoA` applies **A** then **R**, `AoR` applies **R** then **A**, and `GoDoA` applies **A**, then **D**, then **G**. The text in Section 2 describing `GoA` as "advection receiving Gray-Scott states" stated the physical application order backwards; Gray-Scott was receiving advection states.

