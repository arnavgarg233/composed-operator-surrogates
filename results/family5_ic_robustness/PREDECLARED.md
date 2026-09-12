# PREDECLARED — family 5, alternative IC families and gate

Written before any band-check run for this task (no `run_family5.py` execution has
happened yet). This file is not edited after results are seen. Both alternative
families below keep everything from `scripts/band_check/band_check.py`
(package `composed-operator-surrogates`, commit `0f04368`) byte-identical except the
initial-condition generator itself:

- pair: diffusion (diffusivity 0.01) / Fisher-KPP (diffusivity 0.0, reactivity 1.0)
- resolution: 256 points, domain extent 1.0, 1 spatial dimension
- `dt = 0.01`, `half_dt = 0.005` (for the self-convergence floor), ETDRK order 2,
  dealiasing 2/3, 16 circle points, circle radius 1.0
- 100 steps per leg per resolution (`tau` = window duration = 100 × 0.01 = 1.00;
  the floor run uses 200 half-steps = same physical window at double resolution)
- 256 units, `MASTER_SEED = 20260905`
- numerical floor gate: `floor_passes >= 142/256` (`contrast >= 10.0 * floor`)
- commuting null: diffusion (0.01) vs diffusion (0.02), composed AB and BA,
  gate `max relative AB/BA difference < 1e-12`
- saturation limit: fraction of all IC grid values sitting at a clip bound
  (`<= 0 + 1e-12` or `>= 1 - 1e-12`) must be `<= 0.10` (reported, non-gating,
  per `scripts/second_pair/measure_geometric_repair.py`'s definition)
- band: median contrast-to-cross-IC ratio must lie in `[0.3, 0.7]`
- same offset range `(0.4865, 0.5171)` and same amplitude scale `0.175`, and the
  same final `clip(..., 0, 1)`, for both alternative families
- bootstrap CI on the median ratio: 4000 resamples, `seed = 20260905`

The original (fitted-substitute) family, `clip(offset_i + 0.175 *
TruncatedFourierSeries(cutoff=5, std_one=True), 0, 1)`, is rerun unmodified as the
positive control and is expected to reproduce `median_contrast_to_cross_ic =
0.4261565675911184` from `results/tables/band_check/BAND_CHECK_RESULT.json`
(cutoff-5 row).

Both families below draw from the exact same top-level PRNG split as
`band_check.py`: `field_key, offset_key = jax.random.split(jax.random.PRNGKey(20260905))`.
Because `offset_key` only depends on `MASTER_SEED`, the per-unit offsets
`offset_i ~ U(0.4865, 0.5171)` are numerically identical across the original family
and both alternatives below — the field generator is the only thing that varies
between families. `field_key` is split once more, `jax.random.split(field_key,
256)`, into 256 per-unit keys `unit_key_i`, exactly mirroring `band_check.py`'s
`keys = jr.split(field_key, NUM_UNITS)`.

## Alternative family 1 — spectral: Gaussian random field, power-law spectrum

A different spectral shape from the original (which is a *hard-truncated*,
flat-amplitude Fourier series at cutoff 5): a *broadband* field whose Fourier power
decays as a power law across the full resolvable wavenumber range, no hard cutoff.

For each unit `i`, split `unit_key_i` into `(real_key_i, imag_key_i) =
jax.random.split(unit_key_i, 2)`. Let `N = 256` (points), and index the one-sided
(`rfft`) spectrum by `k = 0, 1, ..., 128` (129 bins for a length-256 real signal).
Predeclared power-law exponent: `p = 2.0` (so the power spectral density of the
unnormalized field is `|c_k|^2 ~ k^{-p}`, i.e. amplitude `~ k^{-1}`, a "red"/Brownian-
type spectrum — chosen because it is a standard, simple power law and is not fit to
any statistic of this package).

```
a_k = jax.random.normal(real_key_i, shape=(127,))   # k = 1..127
b_k = jax.random.normal(imag_key_i, shape=(127,))   # k = 1..127
A_k = k ** (-p / 2)                                  # k = 1..127, p = 2.0
c_k = A_k * (a_k + 1j * b_k)                         # k = 1..127
spectrum = zeros(129, dtype=complex)
spectrum[1:128] = c_k                                # bins 0 (DC) and 128 (Nyquist) left exactly 0
z_raw_i = jnp.fft.irfft(spectrum, n=256)              # real, periodic, length 256
z_i = z_raw_i / std(z_raw_i)                          # per-unit unit-std normalization ("std_one")
u0_i = clip(offset_i + 0.175 * z_i, 0, 1)
```

DC and Nyquist bins are held exactly at 0 so the field is exactly mean-zero before
normalization and `irfft` needs no special-casing of those bins.

## Alternative family 2 — bump/localized: sum of random Gaussian bumps

For each unit `i`, split `unit_key_i` into `(centers_key_i, signs_key_i) =
jax.random.split(unit_key_i, 2)`. Predeclared bump count `M = 4` and width
`sigma = 0.06` (domain units; domain extent is 1.0), both fixed constants, not
fit to any statistic in this package.

```
c_1..c_M ~ U(0, 1)                    drawn from centers_key_i  (bump centers, periodic domain)
s_1..s_M ~ Rademacher({-1, +1})       drawn from signs_key_i     (bump signs)
for grid point x in {0, 1/256, ..., 255/256}:
    d_j(x) = x - c_j
    d_j(x) = d_j(x) - round(d_j(x))    # wrap to the periodic domain, distance in [-0.5, 0.5)
    z_raw_i(x) = sum_{j=1}^{4} s_j * exp( -d_j(x)^2 / (2 * sigma^2) )
z_i = z_raw_i / std(z_raw_i)           # per-unit unit-std normalization ("std_one")
u0_i = clip(offset_i + 0.175 * z_i, 0, 1)
```

## Gate (predeclared; not changed after seeing outcomes)

For each family separately: **PASS** if `median_contrast_to_cross_ic` (contrast is
the AB/BA composition-order disagreement at coarse resolution, cross-IC is the
cyclic-successor-in-SHA-256-ordering reference distance, exactly as
`band_check.py` defines them) lies in `[0.3, 0.7]` **and** `floor_passes >= 142`
out of 256 units. **FAIL** otherwise, for that family.

The band-check result is called **robust to IC family** only if **both**
alternative families individually PASS. The commuting null and the saturation
fraction are reported for every family as integrity controls, per
`CLAIM_INVENTORY.md` section 7 row 5, but are not part of the PASS/FAIL gate above
(the commuting null already has its own hard gate at `1e-12`, reported
separately; the saturation limit `0.10` is reported and flagged, not gated,
matching how `measure_geometric_repair.py` treats it).

This gate is fixed as of this file's creation and is not revisited after any
family's numbers are seen.
