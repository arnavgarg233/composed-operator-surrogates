# Family 5b — RESULT

**Question.** Does the BROAD-S repair claim survive under alternative initial-condition families (power-law GRF, four Gaussian bumps), or is it a property of the fitted-substitute IC family?

**Answer from this run.** Unanswerable as predeclared, for a different reason in each IC family. All 40 fits completed. IC_GRF fails G2 and P6; IC_BUMPS fails G1. `evaluate` therefore refused to read any off-support error, so no R and no P3 ratio exist for either IC family.

**Verdict: FAIL (pre-gate) for both IC families. Negative class: FLAWED (design-limited) in both cases.**

---

## 1. What ran

| | |
|---|---|
| Pair | P1: diffusion (κ = 0.01) then Fisher-KPP (r = 1.0), 1-D periodic, 256 points |
| IC families | IC_GRF (power-law GRF) and IC_BUMPS (4 Gaussian bumps), amplitude 0.175, exactly as family 5 |
| Model | FNO-1D (equinox, family 1e), **549,569 parameters** |
| Protocol | 8000 steps, batch 64, Adam + cosine decay, n = 640 fixed |
| Fits | 2 IC families × {narrow, BROAD-S} × 10 seeds = **40 of 40 completed** |
| Window | 2026-09-12 07:08:48Z → 09:27:38Z |

## 2. Gate verdicts

### IC_GRF — blocked by G2 and P6

| Gate | Value | Threshold | Pass |
|---|---|---|---|
| G1 support shift | overlap 0.000000, gap +0.175338 | overlap = 0 | **yes** |
| G2 metric resolves | dynamic range **3.82** (persistence_off 0.0833151 / median narrow e_on 0.0218296) | ≥ 100 | **no** |
| G4a floor censorship | min e_on 0.0167302 | ≥ 1.19e-5 | **yes** |
| P6 convergence | worst e_on **0.0554281** | ≤ 0.001573 | **no** |

### IC_BUMPS — blocked by G1

| Gate | Value | Threshold | Pass |
|---|---|---|---|
| G1 support shift | **overlap 0.557387, gap −0.758351** (training [0.162773, 0.845394], switch [0.288008, 0.921124]) | overlap = 0 | **no** |
| G2 metric resolves | dynamic range 163.66 | ≥ 100 | yes |
| G4a floor censorship | min e_on 0.000219772 | ≥ 1.19e-5 | yes |
| P6 convergence | worst e_on 0.000429514 | ≤ 0.001573 | yes |

`GATES.json` was written at 09:27:38Z before any off-support number was read. Running `evaluate` exits with *"Blocking gate failure for IC_GRF: ['G2_metric_resolves', 'P6_convergence']. Refusing to read off-support error per epistemological protocol."* — and IC_BUMPS is blocked independently by G1, so neither family reaches e_off.

Family 5's reference band ratio (reported, never gated): IC_GRF 0.2410, IC_BUMPS 0.1785 — both below the [0.3, 0.7] band, consistent with family 5's own finding.

## 3. Per-seed on-support error

The per-seed R table, medians and ranges, and the P3 ratio requested for this family do not exist: e_off was never computed, by protocol. The measured quantity is e_on.

**IC_GRF**

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | median | range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| narrow | 0.02253 | 0.01673 | 0.02181 | 0.02039 | 0.02511 | 0.02484 | 0.02008 | 0.01779 | 0.04395 | 0.02185 | **0.02183** | [0.01673, 0.04395] |
| BROAD-S | 0.02163 | 0.02359 | 0.02565 | 0.02593 | 0.02918 | 0.03937 | 0.02400 | 0.02206 | 0.05543 | 0.01894 | **0.02483** | [0.01894, 0.05543] |

**IC_BUMPS** (×10⁻⁴)

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | median | range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| narrow | 2.944 | 2.893 | 2.753 | 2.442 | 2.718 | 2.228 | 2.479 | 2.198 | 4.346 | 2.872 | **2.735** | [2.198, 4.346] |
| BROAD-S | 2.932 | 3.218 | 3.040 | 2.428 | 3.042 | 2.287 | 2.808 | 2.397 | 4.295 | 2.559 | **2.870** | [2.287, 4.295] |

## 4. Reading the negatives

**IC_GRF is a convergence failure, not a BROAD-S result.** The power-law GRF corpus cannot be fitted to the 1-D standard in 8000 steps: median on-support error 2.18e-2 sits only 3.8× below off-support persistence (0.0833). Even if e_off had been read, a ratio built on an error that is within a factor of four of persistence could not have separated repair from no repair. The right recovery is a new predeclaration with a fit budget (or architecture) that reaches convergence on this IC family, not a widened threshold on this one.

**IC_BUMPS is a design failure of the support shift.** The 4-bump fields span nearly the whole [0,1] range, so the narrow spatial-mean window that creates a clean support shift for the fitted-substitute IC family does not create one here: training and switch supports overlap by 0.557. The composed-operator switch this family was meant to probe is simply not present in the cohort, which is a fact about how the cohorts were constructed for bump ICs, not about BROAD-S. A recovery needs an IC-family-specific shift construction, predeclared before any fitting.

Neither result weakens or strengthens the manuscript's repair claim. Both say the family-1e experimental frame does not transfer unmodified to these IC families.

## 5. Cost and runtime

**Runtime departure (important).** `TASK.md` and `PREDECLARED.md` specify the local CPU runtime — `JAX_PLATFORMS=cpu`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`. All 40 fits were instead run under the **CUDA runtime** on RunPod pod `ijm84mivkot2t4`: one **NVIDIA RTX PRO 6000 Blackwell Server Edition, MIG 1g.24gb slice** (1/7 of the card), jax 0.8.1 / jaxlib CUDA 12, python 3.12.3, x86_64. The runner, seeds, steps, batch, optimizer, IC generators and cohorts are unchanged, and each fit is an independent deterministic process; only the backend and the fit scheduling differ. CUDA and CPU backends are not bit-identical, so these e_on values should not be pooled with CPU-run numbers from other families without saying so.

Scheduling notes: fits ran at concurrency 2, then 6–8, then 7 with **CUDA MPS** enabled from 08:35Z. Without MPS the MIG slice time-sliced the processes — throughput was flat at about 0.2 fits/min from concurrency 2 to 6, and per-fit wall time scaled linearly with concurrency (631 s at 2, 1884 s at 6). With MPS at concurrency 7, per-fit fell to about 800 s, which is what allowed all 40 fits to finish.

| IC family / condition | train seconds (median [min, max]) |
|---|---|
| IC_GRF narrow | 1884 [631, 1895] |
| IC_GRF BROAD-S | 826 [800, 1909] |
| IC_BUMPS narrow | 817 [815, 826] |
| IC_BUMPS BROAD-S | 728 [615, 820] |

**Checkpoints are lost.** The 40 checkpoints were written to `pde_volume/family5b/ckpt` *on the pod*, which is the container disk rather than the `/workspace` volume, and were wiped when the pod stopped. This costs nothing scientifically — `evaluate` refuses e_off for both IC families regardless — but it does mean e_off cannot be produced later without refitting. Pregate JSONs and `GATES.json` live on the volume and were pulled intact.

| cost line | USD |
|---|---|
| orphan pod removed before the run | 0.13 |
| pod `ijm84mivkot2t4`, billed total (carries family 5b **and** family 7) | 4.37 |
| brief restart on 2026-09-12 14:03–14:05Z to pull results | 0.02 |
| **total** | **4.52** |
| authorized | 2.50 |

Family 5b's own GPU window is 2.33 h ≈ $1.38. The overrun is idle time, not compute: the run finished at 09:27Z, the operator session died at about 13:55Z, and the local 4.0 h watchdog did not survive it, so the pod idled until it was stopped at 14:00:25Z — roughly $2.7.

## 6. Artifacts

- `results/GATES.json` — per-IC-family gate verdicts, written pre-e_off
- `results/{IC_GRF,IC_BUMPS}/seed<k>_{narrow,broadS}_pregate.json` — 40 fits, e_on and timings
- `results/PREDECLARED.sha256`
- `RESULT.json` — machine-readable version of this document
- `pde_volume/family5b/results/` — same results (checkpoints and per-fit logs were on the pod container disk and are gone)
