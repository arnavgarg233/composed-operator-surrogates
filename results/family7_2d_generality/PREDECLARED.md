# PREDECLARED — Family 7: state-space broadening (BROAD-S) in TWO spatial dimensions (2-D)

Written **2026-09-12T05:15:00Z**, before any 2-D model was trained or evaluated in this family.
The sha256 of this file is taken at the moment of creation and stored in `results/PREDECLARED.sha256`.

Owner rules in force: predeclare everything; **never change a gate after seeing a number**;
preserve failures as-is; report the honest number first.
All existing families, `recovery/`, and `pkg_scratch/` are **read-only**.
All output of this family goes to `pde_restart/family7_2d_generality/` and `pde_volume/family7/`.

---

## 0. Why this family exists

Every current composed-operator result in the study is 1-D periodic. When presenting the
failure and repair of composed operators, CMAME referees and reviewers will ask whether the
phenomenon and its resolution generalize to **two spatial dimensions (2-D)**.

This family evaluates the 2-D generality of the core thesis:
**Does state-space broadening (BROAD-S) repair composed-operator degradation in two spatial dimensions?**

Nothing about the intervention schedule is re-tuned. The burst schedule and training budget
are transplanted directly from the 1-D formulation to 2-D periodic fields.

---

## 1. Pair, physical operators, and domain

- **Pair**: P1 in 2-D = diffusion followed by Fisher-KPP on a periodic 2-D domain $[0, 1)^2$.
  - First operator (diffusion): $\partial_t u = \kappa \nabla^2 u$, with $\kappa = 0.01$.
  - Second operator (Fisher-KPP): $\partial_t u = r u (1 - u)$, with $r = 1.0$ and zero diffusivity ($\kappa_2 = 0.0$).
- **Steppers**: Exponax 2-D steppers (`num_spatial_dims=2`, `domain_extent=1.0`, `dt=0.01`):
  - `ex.stepper.Diffusion(2, domain_extent=1.0, num_points=N, dt=0.01, diffusivity=0.01)`
  - `ex.stepper.reaction.FisherKPP(2, domain_extent=1.0, num_points=N, dt=0.01, diffusivity=0.0, reactivity=1.0, order=2, dealiasing_fraction=2/3, num_circle_points=16, circle_radius=1.0)`
- **Grid Resolution**:
  - Full experiment (GPU pod): $N = 64$ ($64 \times 64$ spatial grid).
  - CPU smoke test: $N = 32$ ($32 \times 32$ spatial grid).
- **Estimand**: Solution operator of diffusion over $K = 10$ steps ($\tau = 10 \times 0.01 = 0.10$).
- **Trajectory length**: 100 steps under diffusion ($\tau = 1.00$ per leg, 101 frames including $t=0$).

---

## 2. Model Architecture: FNO-2D (Equinox)

An Equinox-based 2-D Fourier Neural Operator (FNO-2D) mirroring the depth and width choices
of the 1-D FNO baseline:

- **Input channels**: 3 (scalar state $u(x, y)$, plus normalized 2-D spatial coordinates $x$ and $y$).
- **Output channels**: 1 (predicted state $u_{t+K}(x, y)$).
- **Modes**: $12 \times 12$ ($m_1 = 12, m_2 = 12$).
- **Width**: 32 channels.
- **Layers**: 4 FNO blocks.
- **Lifting layer**: $1 \times 1$ conv (3 -> 32 channels).
- **Spectral Conv 2-D**: 2-D real FFT (`jnp.fft.rfft2` over the two spatial dimensions).
  - Modes along axis -2 (full FFT): $m_1 = 12$ positive frequencies and $m_1 = 12$ negative frequencies.
  - Modes along axis -1 (real FFT): $m_2 = 12$ non-negative frequencies.
  - Two complex weight tensors $W_1$ (upper-left corner) and $W_2$ (lower-left corner), each of shape $(32, 32, 12, 12)$, represented as real and imaginary components for stable optimization.
- **Local connection**: $1 \times 1$ conv (32 -> 32 channels) with bias in each block.
- **Block activation**: GELU on the combined spectral + local update, added to residual:
  $$x \leftarrow x + \text{GELU}(\text{SpectralConv2D}(x) + \text{LocalConv2D}(x))$$
- **Projection**: $1 \times 1$ conv from 32 -> 64 channels, GELU activation, followed by $1 \times 1$ conv from 64 -> 1 channel (mirroring 1-D FNO's $2 \times \text{width}$ projection).
- **Parameter count**:
  - Lifting: $3 \times 32 + 32 = 128$
  - 4 FNO blocks: $4 \times [4 \times (32 \times 32 \times 12 \times 12) + (32 \times 32 + 32)] = 4 \times [589,824 + 1,056] = 2,363,520$
  - Projection: $(32 \times 64 + 64) + (64 \times 1 + 1) = 2,112 + 65 = 2,177$
  - **Total parameters**: **2,365,825**

---

## 3. Training Protocol

Identical to the 1-D baseline protocol:
- **Steps**: 8,000 steps per fit.
- **Batch size**: 64 (fits comfortably within GPU and CPU memory; at $64 \times 64$, 64 items is ~1 MB per batch).
- **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$).
- **Learning rate schedule**: Cosine decay from $\text{LR}_{\text{high}} = 10^{-3}$ to $\text{LR}_{\text{low}} = 10^{-5}$:
  $$\eta(t) = \text{LR}_{\text{low}} + \frac{1}{2} (\text{LR}_{\text{high}} - \text{LR}_{\text{low}}) \left(1 + \cos\left(\frac{\pi t}{\text{STEPS}}\right)\right)$$
- **Loss function**: Relative $L_2$ error (Li et al. LpLoss):
  $$\mathcal{L}(u, \hat{u}) = \frac{\|u - \hat{u}\|_2}{\|u\|_2}$$
- **Precision**: float32.

---

## 4. Conditions, Corpora, and BROAD-S Sampler

Fixed training budget $n = 640$ trajectories for each condition:
- **Initial condition family**: `ex.ic.RandomTruncatedFourierSeries(num_spatial_dims=2, cutoff=5, std_one=True)`
  with amplitude $0.175$, mean offset uniform in $\text{NARROW} = (0.4865, 0.5171)$, clipped to $[0.0, 1.0]$.
  Base seed: `MASTER_SEED + 1 = 20260906` (`MASTER_SEED = 20260905`).
- **NARROW condition**:
  - The 640 base ICs are rolled 100 steps under 2-D diffusion, yielding $(640, 101, 1, N, N)$ states.
- **BROAD-S condition**:
  - The 640 base ICs are burst under the 2-D Fisher-KPP operator following Family 1e's exact block schedule:
    - Block 0: units $0..159$ $\to$ burst $m = 0$ steps (narrow IC unchanged)
    - Block 1: units $160..319$ $\to$ burst $m = 25$ steps of Fisher-KPP
    - Block 2: units $320..479$ $\to$ burst $m = 50$ steps of Fisher-KPP
    - Block 3: units $480..639$ $\to$ burst $m = 100$ steps of Fisher-KPP
  - The 640 burst states are rolled 100 steps under 2-D diffusion, yielding $(640, 101, 1, N, N)$ states.
- **Pairs**: Input frames $0..90$, target frames $10..100$ ($K = 10$), giving $640 \times 91 = 58,240$ pairs per condition.

### Evaluation Cohorts
- **On-support cohort**: 256 units sampled at `MASTER_SEED + 7 = 20260912`, frame 0, target 10 diffusion steps later.
- **Switch cohort**: 256 units sampled at `MASTER_SEED + 8 = 20260913`, evolved 100 steps under 2-D Fisher-KPP, target 10 diffusion steps later.

---

## 5. Fits and Seeds

- **Seeds**: 10 seeds ($0..9$).
- **Conditions**: 2 conditions (`narrow` and `broadS`).
- **Total fits**: 10 seeds $\times$ 2 conditions = **20 fits**.
- Unlike 1-D families where the narrow arm could be reused from prior runs, in 2-D both the narrow arm and the BROAD-S arm are newly fitted because no prior 2-D fits exist.

---

## 6. Gates (Copied verbatim from Family 1e)

| # | Gate | Metric / Definition | Threshold |
|---|---|---|---|
| **G1** | Shift exists | Support overlap between narrow training spatial means and switch cohort spatial means | `overlap == 0.0` |
| **G2** | Metric resolves | Dynamic range: $\text{DR} = \frac{\text{persistence}_{\text{off}}}{\text{median}(e_{\text{on}}(\text{narrow}))} \ge 100.0$ and $\text{median}(e_{\text{on}}(\text{narrow})) \le 0.1 \times \text{persistence}_{\text{on}}$ | $\ge 100.0$ and $\le 0.1 \times \text{persistence}_{\text{on}}$ |
| **G4a** | Floor censorship | $\min(e_{\text{on}}(\text{narrow}), e_{\text{on}}(\text{broadS}))$ across all seeds | $\ge 1.19 \times 10^{-5}$ ($100 \times \text{float32 } \epsilon$) |
| **G4b** | Ceiling censorship | $\max(e_{\text{off}}(\text{broadS}))$ across all seeds | $< \text{persistence}_{\text{off}}$ |
| **P6** | Convergence | $\max(e_{\text{on}}(\text{broadS}))$ across all seeds | $\le 1.573 \times 10^{-3}$ |
| **P3** | Broadening repairs | Repair ratio: $\frac{\text{median}(R_{\text{broadS}})}{\text{median}(R_{\text{narrow}})}$ where $R = e_{\text{off}} / e_{\text{on}}$ | $\le 0.25$ |
| **P4** | Constant mode | Median constant-mode-corrected $e_{\text{off}}$ vs median uncorrected $e_{\text{off}}$ | Reported, not gated |

### Ordering Device
Strict ordering device preserved from all prior families:
1. Every fit runs `train <seed> <cond>`, writes $e_{\text{on}}$ and metadata to `results/<cond>/seed<k>_pregate.json`, and saves model weights to `ckpt/`. **No $e_{\text{off}}$ is computed or recorded during training.**
2. `gates` reads the pregate files, verifies that all 20 pregate files exist, checks G1, G2, G4a, and P6, and writes `results/GATES.json` **before any off-support error is read**.
3. If any blocking gate fails, `evaluate` halts and refuses to read $e_{\text{off}}$.
4. Only when all pre-off gates pass does `evaluate` load the checkpoints, compute $e_{\text{off}}$, evaluate P3 and G4b, and write `results/RESULT.json`.

---

## 7. Coverage Statistics (Truth states only, reported for the record)

For both narrow and BROAD-S training clouds ($640 \times 101 = 64,640$ states, flattened to $\mathbb{R}^{N^2}$):
- **PCA-99 Mahalanobis coverage (Statistic S)**:
  - Smallest $k$ such that cumulative explained variance $\ge 0.99$.
  - Mahalanobis distance $\text{MD}(x)$ under leading $k$ eigenvectors and eigenvalues.
  - Radius $r_{99} =$ 99th percentile of $\text{MD}$ on training states.
  - Coverage = fraction of 256 switch states with $\text{MD} \le r_{99}$.
- **Residual coverage (Statistic $C_3$)**:
  - Out-of-basis residual norm $Q(x) = \|(x - \mu) - U U^T (x - \mu)\|_2$.
  - Threshold $q_{99} =$ 99th percentile of $Q$ on training states.
  - Coverage $C_3 =$ fraction of 256 switch states with $Q \le q_{99}$.
- **Status**: Recorded for scientific completeness, not gated (per Family 1c/1e finding that Mahalanobis radius is non-monotone).

---

## 8. Family Verdict

- **PASS** iff **P3 holds** ($\text{median}(R_{\text{broadS}}) / \text{median}(R_{\text{narrow}}) \le 0.25$) and all standard gates (G1, G2, G4, P6) hold.
- **FAIL** otherwise, naming the binding gate.
