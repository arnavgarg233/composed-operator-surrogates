# PREDECLARED — Family 1 rerun: narrow-vs-broad with a fitted learner

Written before launch. The two pairs and intervals are imported from the validated
family-1b screen; no fitted result is used to choose them.

## Pairs and intervals

- Pair A: `S1_fkpp_r0.5`; narrow `(0.4865, 0.5171)`, broad `(0.48, 0.70)`.
- Pair B: `S1_grayscott_1sp`; narrow `(0.4865, 0.5171)`, broad `(0.30, 0.52)`.
- Ten seeds `0..9` per condition. `MASTER_SEED = 20260905`; training, on-support
  evaluation, and switch cohorts use `MASTER_SEED+1`, `+7`, and `+8` respectively,
  as in family 1.
- Pair A second operator is Fisher-KPP with `diffusivity=0.0, reactivity=0.5`.
- Pair B second operator is the screen's `GeneralPolynomialStepper` form of
  `u_t = F(1-u) - u(1-u)^2`, `F=0.04`:
  `linear_coefficients=(-1.04, 0.0, 0.0)` and
  `polynomial_coefficients=(0.04, 0.0, 2.0, -1.0)`.

## Gates and ordering (copied verbatim from family 1)

| gate | definition | threshold |
|---|---|---|
| G1 shift exists | narrow-training-mean support vs switch-state support | `overlap == 0.0` |
| G2 metric resolves | `dynamic_range = persistence_off / median_e_on(narrow)` **and** `median_e_on(narrow) <= 0.1 * persistence_on` | `>= 100.0`, and the `0.1` fraction |
| G3 powered | `log10(R)` IQR across the ten narrow seeds | `<= 0.5` |
| G4 censored below | `min e_on` over narrow seeds | `>= 1.19e-5` |
| G4 censored above | `max e_off` | `< persistence_off` |
| P3 broadening repairs | `median R_broad` | `<= 0.25 * median R_narrow` |
| P4 constant mode | median constant-mode-corrected `e_off` over narrow seeds | `<= 0.25 * median e_off(narrow)` |
| P6 convergence | worst `e_on` over all seeds and both conditions | `<= 1.573e-3` |

Every seed writes `e_on` and a checkpoint first. `GATES.json` is written before any
`e_off` is read. If G1, G2, G4-censored-below, or P6 fails, stop before `e_off` is
read for that pair. Pair verdict is PASS iff G1, G2, and P3 hold. Family verdict is
PASS iff P3 holds on both pairs.

