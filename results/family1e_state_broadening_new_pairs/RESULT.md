# Family 1e — state-space broadening (BROAD-S) on the two new pairs

Predeclared in `PREDECLARED.md`, sha256 `3e62f81191a5b995b7d66f8aa1191912d157a456c238a83a6a1a4163165afe41`, written 2026-09-11T21:11:33Z before any BROAD-S corpus for these pairs existed, before any coverage statistic was evaluated and before any model was fitted here. 20 new fits (2 pairs × 10 seeds); the narrow and mean-broadening arms are the family-1 rerun's and were **not refitted**.

**FAMILY VERDICT: PASS.** P3 holds on both pairs.

## The honest number first

- **`S1_fkpp_r0.5`**: ten-seed median `R` falls 58.676 → **1.948**, ratio **0.0332** against the predeclared 0.25 — PASS. The rerun's mean-broadening reached 16.415 (ratio 0.280); BROAD-S is **8.4× better** than it.
- **`S1_grayscott_1sp`**: ten-seed median `R` falls 105.598 → **3.701**, ratio **0.0350** against the predeclared 0.25 — PASS. The rerun's mean-broadening reached 28.551 (ratio 0.270); BROAD-S is **7.7× better** than it.

## 1. Per-pair table

### `S1_fkpp_r0.5`

| seed | `R` narrow (rerun) | `R` broad-mean (rerun) | `R` **BROAD-S** | `e_on` BROAD-S | `e_off` BROAD-S |
|---|---|---|---|---|---|
| 0 | 46.394 | 16.392 | **1.9458** | 3.3282e-04 | 6.4759e-04 |
| 1 | 59.419 | 19.152 | **2.0357** | 2.3665e-04 | 4.8176e-04 |
| 2 | 55.830 | 15.386 | **1.6481** | 2.7099e-04 | 4.4660e-04 |
| 3 | 70.414 | 18.262 | **1.9286** | 2.7290e-04 | 5.2632e-04 |
| 4 | 64.323 | 16.437 | **2.1945** | 1.9704e-04 | 4.3239e-04 |
| 5 | 54.206 | 14.390 | **1.5915** | 2.5408e-04 | 4.0437e-04 |
| 6 | 57.933 | 18.199 | **1.9501** | 2.4722e-04 | 4.8208e-04 |
| 7 | 54.434 | 19.549 | **1.7627** | 2.7989e-04 | 4.9336e-04 |
| 8 | 60.463 | 11.690 | **1.9501** | 2.0295e-04 | 3.9578e-04 |
| 9 | 78.987 | 13.979 | **1.9957** | 2.4351e-04 | 4.8597e-04 |
| **median** | **58.676** | **16.415** | **1.9479** | 2.5065e-04 | 4.8192e-04 |
| range | 46.394–78.987 | 11.690–19.549 | 1.5915–2.1945 | | |
| ratio vs narrow | — | 0.2798 | **0.0332** | | |
| coverage (PCA-99, S) | 0.5625 | — | 0.5156 | | |
| residual coverage `C3` | 0.0000 | — | 0.0820 | | |
| residual mean `D3` | 1.72747 | — | 0.16033 | | |

`R_broadS / R_broad-mean` = **0.1187** (8.4× better than the package's mean-broadening at the same fixed `n = 640`).

### `S1_grayscott_1sp`

| seed | `R` narrow (rerun) | `R` broad-mean (rerun) | `R` **BROAD-S** | `e_on` BROAD-S | `e_off` BROAD-S |
|---|---|---|---|---|---|
| 0 | 95.706 | 25.422 | **4.0268** | 3.3654e-04 | 1.3552e-03 |
| 1 | 114.391 | 32.720 | **3.9285** | 2.2099e-04 | 8.6814e-04 |
| 2 | 102.329 | 27.692 | **4.1189** | 2.1308e-04 | 8.7767e-04 |
| 3 | 128.917 | 29.411 | **3.6773** | 2.7611e-04 | 1.0154e-03 |
| 4 | 99.066 | 25.372 | **3.6044** | 1.7910e-04 | 6.4555e-04 |
| 5 | 99.814 | 21.293 | **3.3119** | 2.1692e-04 | 7.1843e-04 |
| 6 | 112.594 | 31.647 | **3.6679** | 2.2384e-04 | 8.2102e-04 |
| 7 | 108.867 | 30.223 | **3.4303** | 2.6023e-04 | 8.9268e-04 |
| 8 | 124.510 | 22.601 | **3.7240** | 1.9920e-04 | 7.4181e-04 |
| 9 | 97.569 | 30.295 | **4.2105** | 2.3036e-04 | 9.6991e-04 |
| **median** | **105.598** | **28.551** | **3.7006** | 2.2241e-04 | 8.7291e-04 |
| range | 95.706–128.917 | 21.293–32.720 | 3.3119–4.2105 | | |
| ratio vs narrow | — | 0.2704 | **0.0350** | | |
| coverage (PCA-99, S) | 0.2852 | — | 0.3633 | | |
| residual coverage `C3` | 0.0000 | — | 0.0586 | | |
| residual mean `D3` | 1.49999 | — | 0.22084 | | |

`R_broadS / R_broad-mean` = **0.1296** (7.7× better than the package's mean-broadening at the same fixed `n = 640`).

## 2. Gates

| gate | threshold | `S1_fkpp_r0.5` | `S1_grayscott_1sp` |
|---|---|---|---|
| G1 shift exists | overlap = 0 | overlap 0.000, gap +0.0771 — PASS | overlap 0.000, gap -0.1263 — PASS |
| G2 metric resolves | dyn. range ≥ 100 | 528.2 — PASS | 869.4 — PASS |
| G4a censored below | min `e_on` ≥ 1.19e−5 | 1.970e-04 — PASS | 1.791e-04 — PASS |
| G4b censored above | max `e_off` < persistence | 6.476e-04 < 0.1015 — PASS | 1.355e-03 < 0.1671 — PASS |
| P6 convergence | worst `e_on` ≤ 1.573e−3 | 3.328e-04 — PASS | 3.365e-04 — PASS |
| **P3 broadening repairs** | ratio ≤ 0.25 | **0.0332** — **PASS** | **0.0350** — **PASS** |

P4 (constant mode) is reported, not gated:

| pair | BROAD-S median `e_off` | constant-mode corrected | fraction left | rerun narrow-arm P4 |
|---|---|---|---|---|
| `S1_fkpp_r0.5` | 4.8192e-04 | 4.6972e-04 | 0.975 | 1.1372e-02 → 1.0855e-02 (not held) |
| `S1_grayscott_1sp` | 8.7291e-04 | 8.6814e-04 | 0.995 | 2.2367e-02 → 2.0795e-02 (not held) |

The ordering device was kept: all 20 seeds wrote `e_on` to a `*_pregate.json` and checkpointed their weights; `results/broadS/GATES.json` (G1, G2, G4a, P6) was written from those alone at 2026-09-11T23:21:08Z; `e_off` was read only afterwards.

## 3. Wall time

Fit phase 2026-09-11T21:14:08Z → 2026-09-11T23:20:57Z, **2.11 h** wall clock for 20 fits at 5 concurrent single-threaded workers; 10.38 process-hours of training. CPU only.

## 4. Verdict

**PASS** — PASS iff P3 holds on both pairs with G1, G2, G4 and P6 holding; PARTIAL if P3 holds on exactly one; FAIL otherwise.

Both pair verdicts are **PASS** (G1, G2 and P3 all hold for each), and G4a, G4b and P6 hold on both.

## 5. What the numbers say, and what they do not

1. **The intervention generalises.** Family 1c's BROAD-S sampler, transplanted without a single change of constant, clears the package's own 0.25 limit on both new pairs — 0.0332 and 0.0350, an order of magnitude inside it — where the package's mean-broadening at the same fixed `n = 640` reached only 0.280 and 0.270 and missed. Four pairs have now been tested under BROAD-S (Cahn-Hilliard 0.0170 and Fisher-KPP 0.0223 in family 1c, these two at 0.0332 and 0.0350); all four pass, the two new ones by a smaller margin than family 1c's two.
2. **The comparison is matched.** Narrow, mean-broad and BROAD-S share the architecture, optimiser, schedule, seeds `0..9`, sample count `n = 640`, the on-support cohort (`MASTER_SEED+7`) and the switch cohort (`MASTER_SEED+8`). The narrow and mean-broad arms are the rerun's own fits, read from `family1_rerun/results/`, not refitted here; only the training cloud moves.
3. **The metric is not censored and the repair is not a floor artefact.** BROAD-S median `e_off` is 4.819e-04 against `persistence_off` 0.1015 on `S1_fkpp_r0.5` (a factor 211 of headroom) and 8.729e-04 against 0.1671 on `S1_grayscott_1sp` (factor 191); `e_on` sits above the float32 floor by an order of magnitude on every seed. Both medians stay **above 1** (1.95 and 3.70), so unlike family 1c's Cahn-Hilliard cell (`R = 0.787`) the switch task never becomes easier than the on-support task here; there is nothing to explain away.
4. **The repair is not bought back on-support.** BROAD-S median `e_on` is 2.506e-04 / 2.224e-04 against the narrow arm's 1.922e-04 — a 16–30 % on-support cost, the same order the rerun's mean-broadening already paid (2.52e−4 and 2.34e−4), and P6 holds with two decades to spare.
5. **Seeds are not independent units.** Ten seeds of one pair are ten draws of one training run, not ten operators. Two pairs are not a survey. The `R` distributions do not overlap between conditions on either pair, which is the strongest statement the design supports.

## 6. Coverage: recorded, and still not the mechanism

| pair | statistic | narrow | BROAD-S | direction |
|---|---|---|---|---|
| `S1_fkpp_r0.5` | coverage `S` (PCA-99) | 0.5625 | 0.5156 | down |
| `S1_fkpp_r0.5` | residual coverage `C3` | 0.0000 | 0.0820 | up |
| `S1_fkpp_r0.5` | residual mean `D3` | 1.7275 | 0.1603 | down |
| `S1_grayscott_1sp` | coverage `S` (PCA-99) | 0.2852 | 0.3633 | up |
| `S1_grayscott_1sp` | residual coverage `C3` | 0.0000 | 0.0586 | up |
| `S1_grayscott_1sp` | residual mean `D3` | 1.5000 | 0.2208 | down |

Neither statistic is a gate here (`PREDECLARED.md` §4), and the table says why that was the right call, made in advance:

- **Family 1c's PCA-99 coverage `S` is still non-monotone under the intervention that works.** On `S1_fkpp_r0.5` it goes *down* — 0.5625 → 0.5156 — while `R` falls 58.7 → 1.95. Had this family re-used family 1c's REPAIR-4 (`coverage >= 0.9`) as a gate, it would have failed on both pairs in both conditions while the repair succeeded by a factor of 30. On `S1_grayscott_1sp` `S` does move the right way (0.2852 → 0.3633), but only to 0.36 — for a corpus that contains 160 units produced by exactly the 100-step burst that produces the switch cohort, and it still reads lower than the 0.910 `S` assigns to the narrow Cahn-Hilliard cloud that degrades 46×. So the sign is inconsistent across pairs and the level is uninformative: family 1c's diagnosis of the statistic reproduces here on new pairs.
- **Family 1d's residual coverage `C3` is monotone on both pairs** (0.0000 → 0.0820 and 0.0000 → 0.0586), extending family 1d's PRED-S3-B to two more pairs — but it is monotone from *zero* to *almost zero*: 92–94 % of the switch states still sit outside the BROAD-S cloud's residual envelope while their `R` is under 4. The residual mean `D3` falls by 10.8× and 6.8×, which is the direction a working predictor should move, and is reported as such — not as a validated certificate. **No statistic tested in this programme yet tells you in advance which pairs need the intervention or certifies when you have it.**

## 7. Every choice made here

1. **Imported, never copied, never written to.** `run_baseline.py` and `timing_probe.py` are imported from `dependency`; `save_params` / `load_params` from `family1_second_pairs/run_family1.py`; the two second-operator definitions from `family1_rerun/run_family1_rerun.py`, so the operators are the identical objects the rerun fitted against. No function that writes into those trees was called and no file under `recovery/`, `dependency/`, `family1_second_pairs/`, `family1b_pair_screen/`, `family1_rerun/`, `family1c_state_support/`, `family1d_predictor/`, `family2_*` or `family4_*` was modified. The 20 new checkpoints (42 MB) live in `work/family1e/ckpt/`. **One derived artefact was created outside this tree and is declared rather than hidden:** importing the rerun's operator definitions made CPython write a bytecode cache, `family1_rerun/__pycache__/run_family1_rerun.cpython-312.pyc`, into an already existing `__pycache__` directory (the same thing families 1c and 1d did to `family1_second_pairs/`). No source, result, checkpoint or log file in any read-only tree was touched; `find` over all of them shows that one path and nothing else.
2. **The narrow arm was not refitted** — the task forbids it and the rerun's ten narrow fits used the same corpus, seeds and cohorts. `R_narrow` per seed is read from `family1_rerun/results/<pair>/RESULT.json`. 20 fits saved.
3. **Coverage was computed before the fits finished but after predeclaration** (`results/SUPPORT.json`, 2026-09-11T21:14Z, truth states only, no learner). It gates nothing, so its ordering cannot bias a verdict.
4. **G1 is re-derived, not re-litigated.** The switch-state supports recomputed here reproduce the rerun's byte-for-byte; the printed `gap` on `S1_grayscott_1sp` differs in the fourth decimal (-0.12633 here against −0.12682 in the rerun's `GEOMETRY.json`) because the rerun's file carries the value hard-coded from the family-1b screen while this one subtracts the two supports it just measured. Overlap is exactly 0.0 either way.
5. **Private JAX compilation cache** at `work/family1e/jax_cache`, not the shared one from `ENV.sh` (family 1c §4.5: the shared cache holds AOT entries from another lane that XLA refuses with a SIGILL warning). Everything else in `PREDECLARED.md` §7 was left as declared: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`.
6. **Concurrency was 5 single-threaded workers.** The box was shared with two other lanes throughout (load average 45–275); a fit took 22–44 min against the package's 5.1 min estimate. Concurrency and contention change wall time only, never a number: every reloaded checkpoint reproduced its banked `e_on` **exactly** (`reload_exact: true` on all 20 seeds).

## 8. Not done

- **No fit was run outside the 20 predeclared BROAD-S fits.** No sampler variant, no seed added, no gate moved.
- No new support statistic was introduced; `S` and `C3` are reported exactly as family 1c and family 1d defined them, including where `S` embarrasses itself.
- Family 1c's PRED half is not revisited here and stays FAIL: BROAD-S repairs the degradation, nothing yet predicts it.
- One architecture (A1 FNO, 549 569 parameters), one resolution (256), one horizon (`k = 10`, `tau = 1.00` per leg), one IC family — the package's fitted substitute, so every number inherits that caveat. BROAD-S uses knowledge of the second operator; it answers "can a dictionary containing states of the kind the model will meet repair the failure", not "can you build one without knowing the shift".

## 9. Files

```
PREDECLARED.md                      pairs, sampler, gates; written before any fit
results/PREDECLARED.sha256          its hash, 2026-09-11T21:11:33Z
run_family1e.py                     harness (imports the package's; writes only here)
aggregate.py                        tables, gates, verdict
launch.sh / worker.sh               20 detached fits, 5 concurrent
results/SUPPORT.json                coverage S and C3, truth states only, not gated
results/broadS/GATES.json           G1/G2/G4a/P6, written before any e_off
results/broadS/<pair>_seed<k>[_pregate].json   20 new fits, per seed
results/RESULT.json, RESULT.md      this result (copies at the family root)
logs/                               one log per worker, plus queue/gates/eval logs
work/family1e/ckpt/    20 BROAD-S checkpoints, 42 MB
```

