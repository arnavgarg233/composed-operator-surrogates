# PREDECLARED — Family 5b: IC-Family Robustness of BROAD-S Repair

Predeclared before writing execution code or running any fits for this family.
Written at 2026-09-11T21:55:00-07:00.

---

## 1. Rationale and Objective

Family 5 (`pde_restart/family5_ic_robustness/`) investigated whether the package's composition-scale band diagnostic (median contrast-to-cross-IC ratio in `[0.3, 0.7]`) generalized across alternative initial-condition families. It showed that the band diagnostic **fails** under two alternative IC families:
- Spectral power-law Gaussian Random Field (`family1_spectral_grf_power_law`): median ratio `0.2410` (outside `[0.3, 0.7]`).
- Localized Gaussian bumps (`family2_bump_localized`): median ratio `0.1785` (outside `[0.3, 0.7]`).

This proved that the [0.3, 0.7] band diagnostic is a specific property of the package's fitted-substitute IC family (truncated Fourier series at cutoff 5).

The primary headline claim of the manuscript is the **REPAIR claim**: state-space broadening (**BROAD-S**) at fixed $n = 640$ resolves composed-operator degradation, cutting $R$ by $30\times$ to $60\times$ on four distinct operator pairs (Family 1c and Family 1e).

**Family 5b tests whether the BROAD-S repair claim is robust to the choice of initial condition family.** Specifically: does BROAD-S at fixed $n = 640$ repair composed-operator degradation when training and evaluation states are drawn from the two alternative IC families that broke the band diagnostic?

This is a brand new experiment with its own predeclaration. Family 5's files and verdict remain untouched.

---

## 2. Operator Pair, Domain, and Discretization

The experiment uses the canonical primary pair **P1** from the manuscript:
- **First operator $O_1$ (Diffusion)**:
  $$\partial_t u = \nu \partial_{xx} u, \quad \nu = 0.01$$
- **Second operator $O_2$ (Fisher-KPP)**:
  $$\partial_t u = r u (1 - u), \quad r = 1.0, \quad \nu = 0.0$$
  ETDRK order 2, 2/3 dealiasing fraction, 16 contour circle points, radius 1.0.
- **Domain**: 1D periodic, extent $L = 1.0$, $N = 256$ spatial grid points.
- **Time stepping**: $\Delta t = 0.01$, trajectory rollout duration 100 steps ($\tau = 1.00$).
- **Prediction horizon**: $k = 10$ steps of diffusion ($\tau = 0.10$).
- Stepper implementations imported directly from `timing_probe.py:build_steppers(tp.DT)`:
  - $O_1$: `tp.build_steppers(tp.DT)[0]`
  - $O_2$: `tp.build_steppers(tp.DT)[1]`

---

## 3. Initial Condition Families

Two alternative IC families are tested, identical in mathematical formulation and parameters to Family 5 (`pde_restart/family5_ic_robustness/run_family5.py`):

1. **`IC_GRF` (Spectral Power-Law Gaussian Random Field)**:
   - Broadband spectrum without hard cutoff: power spectral density $E(k) \propto k^{-p}$ with $p = 2.0$.
   - For $k \in \{1, \dots, 127\}$: amplitude $A_k = k^{-1.0}$; complex Fourier coefficients $c_k = A_k (a_k + i b_k)$ with $a_k, b_k \sim \mathcal{N}(0, 1)$.
   - DC bin ($k = 0$) and Nyquist bin ($k = 128$) held identically at 0.
   - Inverse real FFT to spatial grid ($N = 256$), normalized by standard deviation to unit variance ($\sigma = 1.0$).
   - Base offset $u_{\text{offset}} \sim \mathcal{U}(0.4865, 0.5171)$.
   - Amplitude scale $A = 0.175$.
   - Final field: $u_0 = \mathrm{clip}(u_{\text{offset}} + A \cdot z, 0.0, 1.0)$.

2. **`IC_BUMPS` (Sum of 4 Localized Gaussian Bumps)**:
   - $M = 4$ bumps with width $\sigma = 0.06$ on periodic domain $[0, 1)$.
   - Bump centers $c_j \sim \mathcal{U}(0, 1)$, bump signs $s_j \in \{-1, +1\}$ with equal probability.
   - Periodic wrapped distance $d_j(x) = (x - c_j) - \mathrm{round}(x - c_j)$.
   - Sum of Gaussians: $z_{\text{raw}}(x) = \sum_{j=1}^4 s_j \exp\left(-\frac{d_j(x)^2}{2 \sigma^2}\right)$, normalized by std to unit variance ($\sigma = 1.0$).
   - Base offset $u_{\text{offset}} \sim \mathcal{U}(0.4865, 0.5171)$, amplitude scale $A = 0.175$.
   - Final field: $u_0 = \mathrm{clip}(u_{\text{offset}} + A \cdot z, 0.0, 1.0)$.

---

## 4. PRNG Derivations and Offsets

Base master seed: `MASTER_SEED = 20260905`.
To guarantee statistical independence between IC families while keeping seeds deterministic and reproducible, each IC family has a fixed predeclared master-seed offset:

- `IC_GRF`: offset `+10000` $\implies \text{family\_master\_seed} = 20270905$
- `IC_BUMPS`: offset `+20000` $\implies \text{family\_master\_seed} = 20280905$

Derivation for each IC family:
- Training IC base: `family_master_seed + 1`
- Evaluation on-support cohort: `family_master_seed + 7`
- Evaluation switch cohort: `family_master_seed + 8`
- Learner model weights & batch sampling: 10 seeds ($s \in \{0, 1, \dots, 9\}$):
  - Model initialization: `jr.PRNGKey(seed)`
  - Mini-batch shuffling: `jr.PRNGKey(seed + 9999)`

---

## 5. Experimental Conditions and Corpora

For each IC family:

1. **NARROW Condition**:
   - $n = 640$ training initial conditions generated with offset range `(0.4865, 0.5171)` using seed `family_master_seed + 1`.
   - Evolved 100 steps under diffusion $O_1$, producing $(640, 101, 1, 256)$ trajectory states.
   - Input/target pairs $(u_t, u_{t+10})$ extracted for $t \in [0, 90]$ via `rb.make_pairs`, yielding $640 \times 91 = 58,240$ training pairs.

2. **BROAD-S Condition**:
   - Fixed sample size $n = 640$ base initial conditions generated identically to narrow (`family_master_seed + 1`).
   - Burst schedule verbatim from Family 1c/1e (`BROADS_BLOCKS`):
     - Block 0 (units 0..159): 0 steps of Fisher-KPP $O_2$ (narrow IC unchanged)
     - Block 1 (units 160..319): 25 steps of Fisher-KPP $O_2$
     - Block 2 (units 320..479): 50 steps of Fisher-KPP $O_2$
     - Block 3 (units 480..639): 100 steps of Fisher-KPP $O_2$
   - Concatenated ICs evolved 100 steps under diffusion $O_1$, producing $(640, 101, 1, 256)$ trajectory states.
   - Input/target pairs extracted identically via `rb.make_pairs` ($58,240$ pairs).

3. **Evaluation Cohorts**:
   - Drawn from the same IC family, with $N_{\text{eval}} = 256$ units each:
   - **On-support cohort**: ICs from `family_master_seed + 7` with offset range `(0.4865, 0.5171)`. $u_{\text{on}}$ is frame 0 of rollout under $O_1$; target is frame 10 under $O_1$.
   - **Switch cohort**: ICs from `family_master_seed + 8` with offset range `(0.4865, 0.5171)`. $u_{\text{switch}}$ is frame 100 under Fisher-KPP $O_2$; target is frame 10 under $O_1$ starting from $u_{\text{switch}}$.

Total number of fits:
$$\text{2 IC families} \times \text{2 conditions (narrow, broadS)} \times \text{10 seeds} = 40 \text{ fits}.$$

---

## 6. Learner Architecture and Training Protocol

- **Architecture**: Fourier Neural Operator (FNO-1D) from Li et al. (2021), imported from `timing_probe.py`:
  - 16 Fourier modes, width 64, 4 Fourier layers, projection dimension 128.
  - 2 input channels ($u, x$).
  - 549,569 trainable parameters.
- **Training**:
  - 8000 gradient steps, mini-batch size 64.
  - Hand-written Adam optimizer from `timing_probe.py`.
  - Cosine learning rate decay schedule from $\eta_{\max} = 10^{-3}$ to $\eta_{\min} = 10^{-5}$.
  - Float32 precision on CPU.

---

## 7. Gates and Pre-Off-Support Ordering Device

In strict accordance with the package's epistemological ordering rule:
- Training writes only $e_{\text{on}}$ and the checkpoint to disk.
- $e_{\text{off}}$ is **never** evaluated or written during training.
- `results/GATES.json` is evaluated and committed to disk **before** any off-support error is read.
- If a blocking gate fails, the evaluation pass terminates immediately and refuses to compute $e_{\text{off}}$.

### Gates (Copied verbatim from Family 1e):

| Gate | Name | Definition | Threshold |
|---|---|---|---|
| **G1** | Shift exists | Support overlap between narrow training spatial means and switch state spatial means | $\text{overlap} = 0.0$ (support shift $> 0$) |
| **G2** | Metric resolves | Dynamic range: $\frac{\text{persistence}_{\text{off}}}{\text{median}(e_{\text{on, narrow}})}$ and $e_{\text{on, narrow}} \le 0.1 \cdot \text{persistence}_{\text{on}}$ | $\text{dynamic range} \ge 100.0$ and $e_{\text{on, narrow}} \le 0.1 \cdot \text{persistence}_{\text{on}}$ |
| **G4a** | Censoring below | Minimum $e_{\text{on}}$ across all 10 narrow and 10 broadS seeds | $\min(e_{\text{on}}) \ge 1.19 \times 10^{-5}$ ($100\times$ float32 epsilon) |
| **G4b** | Censoring above | Maximum $e_{\text{off}}$ across all 10 broadS seeds | $\max(e_{\text{off}}) < \text{persistence}_{\text{off}}$ |
| **P6** | Convergence | Worst $e_{\text{on}}$ across all broadS seeds | $\max(e_{\text{on, broadS}}) \le 1.573 \times 10^{-3}$ |
| **P3** | Broadening repairs | Degradation ratio under BROAD-S vs NARROW: $\frac{\text{median}(R_{\text{broadS}})}{\text{median}(R_{\text{narrow}})}$ | $\le 0.25$ per IC family |
| **P4** | Constant mode | Median constant-mode corrected $e_{\text{off}}$ vs uncorrected | Reported (non-gating) |

### Reference Reporting (Non-Gating):
Family 5 band check contrast-to-cross-IC ratios are recorded alongside the results:
- `IC_GRF`: 0.241035 (95% CI [0.2247, 0.2822]; band [0.3, 0.7] failed)
- `IC_BUMPS`: 0.178527 (95% CI [0.1658, 0.2027]; band [0.3, 0.7] failed)

---

## 8. Verdict Taxonomy

For each IC family separately:
- **PASS**: Gate P3 clears ($\text{median}(R_{\text{broadS}}) / \text{median}(R_{\text{narrow}}) \le 0.25$) AND all prerequisite gates (G1, G2, G4a, G4b, P6) clear.
- **FAIL**: P3 fails or a prerequisite gate fails.
  - If FAIL, the result is classified according to the failure epistemology:
    - **REAL**: G1, G2, and G4 clear (valid shift, resolving metric, no censoring), but P3 fails. BROAD-S genuinely failed to repair the degradation on this IC family.
    - **FLAWED**: G1 or G2 fails. The setup did not constitute a valid test (e.g. no support shift exists, or metric dynamic range is inadequate).
    - **MISREAD**: G4a or G4b fails (censoring floor or ceiling prevents meaningful error comparison).

Overall experiment verdict:
- **PASS**: Both IC families PASS.
- **PARTIAL**: Exactly one IC family PASSES.
- **FAIL**: Neither IC family passes.
