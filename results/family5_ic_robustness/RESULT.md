# Family 5 — IC-family robustness of the band check

Package under test: `recovery/composed-operator-surrogates` commit `0f04368`
(read-only; not modified). Runner: `run_family5.py` in this directory, vendoring
the physical/numerical configuration and the contrast / floor / cross-IC /
commuting-null / saturation definitions from `scripts/band_check/band_check.py`,
`scripts/band_check/integrity_controls.py`, and
`scripts/second_pair/measure_geometric_repair.py`, with attribution inline.
Gate predeclared in `PREDECLARED.md` before this script was ever run, and not
changed afterward. Environment: `.venv`
(CPython 3.12.13, jax/jaxlib 0.8.1, exponax 0.2.0 — see `env/ENV_RECEIPT.md`),
threads pinned (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false JAX_PLATFORMS=cpu`).

## Positive control

The original fitted-substitute family (`clip(offset_i + 0.175 *
TruncatedFourierSeries(cutoff=5, std_one=True), 0, 1)`) was rerun unmodified and
reproduces the banked value **exactly, bit for bit**:
`median_contrast_to_cross_ic = 0.4261565675911184` (target
`0.4261565675911184`, absolute difference `0.0`). `floor_passes = 256/256`,
`median_contrast = 0.031231252093953152` — both match
`results/tables/band_check/BAND_CHECK_RESULT.json`'s cutoff-5 row exactly.
The only reported statistic that differs at the sub-part-per-billion level from
that file is `median_contrast_to_floor` (ours `3842.0664152247127` vs. the
package's `3842.066415153485`) — a ~2e-11 relative difference, consistent with
floating-point reduction-order non-determinism in XLA across separately jitted
compositions; `median_contrast` and `median_floor` individually match exactly,
so this is a rounding artifact of the ratio-of-medians-vs-median-of-ratios
statistic, not a discrepancy in the underlying trajectories, and it is not a
gated quantity.

## Results table

| family | median contrast/cross-IC | 95% CI (4000 resamples, seed 20260905) | floor passes | commuting null (gate 1e-12) | saturation (limit 0.10) | verdict |
|---|---|---|---|---|---|---|
| original (cutoff 5, positive control) | 0.426157 | [0.387254, 0.468623] | 256/256 | 4.86e-16 PASS | 0.0000 | **PASS** |
| family 1 — spectral GRF, power-law spectrum | 0.241035 | [0.224750, 0.282234] | 256/256 | 5.96e-16 PASS | 0.0007 | **FAIL** |
| family 2 — bump/localized, 4 random Gaussian bumps | 0.178527 | [0.165812, 0.202696] | 256/256 | 6.81e-16 PASS | 0.0168 | **FAIL** |

`floor_passes_required = 142/256`; band = `[0.3, 0.7]`. Per the predeclared
gate, a family PASSes only if the median ratio lands inside the band **and**
the floor gate clears. **`robust_to_ic_family = False`** — the gate requires
both alternative families to PASS, and neither does.

Wall time: **6.49 s** total for all three families (positive control 2.97 s,
family 1 1.79 s, family 2 1.71 s), well under the 10-minute foreground budget;
run in the background with output logged to `run_family5.log` per the
recorded launch procedure, though in the event it finished before the first poll.

## Honest reading

The band check's numerical-floor gate and its commuting-pair null are both
essentially IC-family-invariant here: every family clears 256/256 against the
required 142, and every commuting-pair residual sits at machine-precision
(~1e-16) against a 1e-12 gate, regardless of which field generator produced
`u0`. But the quantity the band actually gates — the median contrast-to-cross-IC
ratio — is not robust to the IC family at all: it falls from 0.426 (comfortably
inside `[0.3, 0.7]`, matching the banked/reconstructed value) to 0.241 and
0.179 under the two alternatives, both below the band's lower edge, with 95%
CIs that do not overlap the band either. Both alternative families keep the
amplitude scale (0.175), offset range (`(0.4865, 0.5171)`), clip, units, seed,
solver configuration, and floor/commuting/saturation controls byte-identical to
the original, and their own saturation fractions (0.07% and 1.7%) stay far
inside the 0.10 non-gating limit, so the failure is not an artifact of the
fields being pathologically over-clipped. What differs is the shape of the
random field itself — a hard-cutoff, flat-amplitude truncated Fourier series
(original) versus a broadband power-law spectrum (family 1) or a handful of
localized bumps (family 2) — and that difference alone moves the median ratio
by roughly a factor of 1.8–2.4×, entirely outside the pre-declared band for both
alternatives. Read plainly: the `[0.3, 0.7]` band clearance reported for the
composed-operator-surrogates band check is a property of the specific
fitted-substitute IC family used to reconstruct it (and possibly of the true,
unrecoverable original generator it was calibrated to resemble), not a
property that survives switching to two other reasonable, equally-amplitude-
and-offset-matched IC families; this package's central band-check claim should
not be read as IC-family-general without further qualification, and the
`[0.3, 0.7]` band itself was never validated against anything but the one
family it happened to be measured on.
