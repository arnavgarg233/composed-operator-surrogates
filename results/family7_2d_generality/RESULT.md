# Family 7 — RESULT

**Question.** Does state-space broadening (BROAD-S) repair composed-operator degradation in two spatial dimensions?

**Answer from this run.** Unanswerable as predeclared. All 20 fits completed, but two of the pre-gates copied verbatim from the 1-D family 1e fail at 64×64, so `evaluate` refused to read any off-support error. There is no R and no P3 ratio for family 7.

**Verdict: FAIL (pre-gate). Negative class: FLAWED (design-limited).**

---

## 1. What ran

| | |
|---|---|
| Pair | P1 in 2-D: diffusion (κ = 0.01) then Fisher-KPP (r = 1.0), periodic [0,1)², Exponax 2-D steppers |
| Grid | 64 × 64 |
| Model | FNO-2D (equinox), modes 12×12, width 32, 4 layers, **2,365,825 parameters** |
| Protocol | 8000 steps, batch 64, Adam + cosine decay, n = 640 fixed |
| Fits | 10 seeds × {narrow, BROAD-S} = **20 of 20 completed** |
| Window | 2026-09-12 06:06:17Z → 07:08:10Z |

## 2. Gate verdicts

| Gate | Value | Threshold | Pass |
|---|---|---|---|
| G1 support shift exists | overlap 0.000000, gap +0.176556 (narrow [0.486431, 0.517406], switch [0.693962, 0.721773]) | overlap = 0 | **yes** |
| G2 metric resolves | dynamic range **66.79** (persistence_off 0.118770 / median narrow e_on 0.00177816) | ≥ 100 | **no** |
| G4a floor censorship | min e_on 0.00139155 | ≥ 1.19e-5 | **yes** |
| P6 convergence | worst BROAD-S e_on **0.00194866** | ≤ 0.001573 | **no** |
| P3 broadening repairs | — | ≤ 0.25 | **not evaluated** |

Blocking failures: `G2_metric_resolves`, `P6_convergence`.

`GATES.json` was written before any off-support number was read (`written_before_any_off_support_number_was_read: true`), and `evaluate` then exited with *"Blocking failures present ['G2_metric_resolves', 'P6_convergence']: refusing to evaluate e_off"*. The ordering device did exactly what it was built to do.

## 3. Per-seed on-support error

No R values exist: e_off was never computed, so the per-seed R table, its medians and ranges, and the P3 ratio requested for this family are all unavailable by protocol rather than by omission. What the run does give is e_on.

| seed | narrow e_on | BROAD-S e_on |
|---|---|---|
| 0 | 0.00180549 | 0.00172405 |
| 1 | 0.00163866 | 0.00168600 |
| 2 | 0.00187375 | 0.00179231 |
| 3 | 0.00155417 | 0.00158717 |
| 4 | 0.00189923 | 0.00193576 |
| 5 | 0.00178395 | 0.00174330 |
| 6 | 0.00194832 | 0.00194866 |
| 7 | 0.00139155 | 0.00162218 |
| 8 | 0.00177238 | 0.00188482 |
| 9 | 0.00163562 | 0.00169408 |
| **median** | **0.00177816** | **0.00173368** |
| range | [0.00139155, 0.00194832] | [0.00158717, 0.00194866] |

Persistence: on-support 0.185707, off-support 0.118770.

## 4. Reading the negative

The fits converged normally — train loss fell from 1.29 to 1.8e-4 over 8000 steps, and the smoke at 32×32 reproduced exactly. What failed is the *transplant of 1-D thresholds into 2-D*:

- The attainable on-support error floor in 2-D at 64×64 is about 1.4e-3, roughly an order of magnitude above the 1-D floor the thresholds were written against. G2 asks for a 100× separation between that floor and off-support persistence; 2-D delivers 66.8×.
- P6's limit (1.573e-3) is an absolute 1-D number. The best 2-D BROAD-S fit reaches 1.59e-3 and the worst 1.95e-3, so P6 fails by about 24% on the worst seed while the fits are otherwise healthy.

Both failures are properties of the threshold transplant, not evidence about BROAD-S in 2-D. Classify as **FLAWED (design-limited)**, not REAL. The predeclaration is intact and this FAIL stays as it is; testing 2-D generality properly needs a new predeclaration with 2-D-appropriate G2 and P6 thresholds (or a finer grid / longer schedule that lowers the 2-D error floor), frozen before any e_off is read.

## 5. Cost and runtime

**Runtime departure.** The predeclared local runtime is CPU (`JAX_PLATFORMS=cpu`). All 20 fits were run on the CUDA runtime of RunPod pod `ijm84mivkot2t4` — one **NVIDIA RTX PRO 6000 Blackwell Server Edition, MIG 1g.24gb slice** (1/7 of the card), jax 0.8.1, python 3.12.3, x86_64, 4 fits sharing the slice. Runner code, seeds, steps, batch, optimizer and cohorts are unchanged.

Per-fit wall time at concurrency 4:

| condition | train seconds (median [min, max]) | elapsed seconds (median [min, max]) |
|---|---|---|
| narrow | 696 [691, 770] | 723 [714, 793] |
| BROAD-S | 691 [652, 696] | 700 [671, 720] |

| cost line | USD |
|---|---|
| orphan pod removed before the run (`ORPHAN_POD_COST`) | 0.13 |
| pod `ijm84mivkot2t4`, billed total (carries family 7 **and** family 5b) | 4.37 |
| brief restart on 2026-09-12 14:03–14:05Z to pull results | 0.02 |
| **total** | **4.52** |
| authorized | 2.50 |

Family 7's own GPU window is 1.03 h ≈ $0.61 of that total. The overrun is not compute: family 5b finished at 09:27Z, the operator session died at about 13:55Z, and the local 4.0 h watchdog did not survive it, so the pod idled until it was stopped at 14:00:25Z — roughly $2.7 of idle burn. Detached watchdogs must be launched so they outlive the session that starts them.

## 6. Artifacts

- `results/GATES.json` — gate verdicts, written pre-e_off
- `results/{narrow,broadS}/seed<k>_pregate.json` — 20 fits, e_on and timings
- `results/SMOKE.json`, `results/PREDECLARED.sha256`
- `RESULT.json` — machine-readable version of this document
- `pde_volume/family7/` — same results plus the 20 checkpoints (190 MB, on the pod's `/workspace` volume and pulled locally) and the per-fit logs
