# Composed operator surrogates

Composed operator surrogates built from a dictionary of single-operator neural surrogates, with training-dictionary broadening in state space. The paper is “Broadening training dictionaries in state space for composed neural PDE surrogates”, Arnav Garg, 2026, manuscript in preparation.

## Reproduce

```bash
bash reproduce.sh
```

This verifies the shipped files, recorded summaries, headline values, tests, and leak policy without refitting. It requires Python 3.10+ and no packages. Optional refits use a separately checked-out copy of the public dependency at commit `0f04368`, plus the unshipped intermediate inputs described by the relevant experiment records.

## Layout

| Path | Experiment |
| --- | --- |
| `results/family1b_pair_screen/` | Pair screen for measurable, non-commuting configurations |
| `results/family1_rerun/` | Mean-only broadening on two pairs |
| `results/family1e_state_broadening_new_pairs/` | State-space broadening on two pairs |
| `results/family1c_state_support/` | State-space broadening on the conservative Cahn-Hilliard pair |
| `results/family3_baselines/` | Baselines and order sensitivity |
| `results/family5_ic_robustness/` | Initial-condition families |
| `results/family6_serrano_baseline/` | Test-time splitting comparison |
| `results/family2_resolution_horizon/` | Resolution x horizon sweep |
| `results/family4b_unet_broadS/` | U-Net second architecture |
| `results/family8_complex_algebra/` | Four-operator 2-D dictionary and word recovery |
| `results/family7_2d_generality/` | 2-D run that did not clear its entry criteria |
| `results/family5b_ic_repair/` | Initial-condition run that did not clear its entry criteria |

## Results

- Pair screen: 12 of 25 configurations passed. (`results/family1b_pair_screen/RESULT.json:counts.passing, counts.configurations`)
- Mean-only broadening: ratios 0.280 and 0.270, or 3.6x and 3.7x repairs, with both pair verdicts FAIL. (`results/family1_rerun/RESULT.json:pairs.S1_fkpp_r0.5.narrow.median_R, pairs.S1_fkpp_r0.5.broad.median_R, pairs.S1_grayscott_1sp.narrow.median_R, pairs.S1_grayscott_1sp.broad.median_R`)
- State-space support: ratios 0.017 and 0.022, with rho -0.257 and coverage 0.828; recorded FAIL. (`results/family1c_state_support/RESULT.json:PRED.PRED_A_spearman.rho, REPAIR.REPAIR_4_coverage.coverage_achieved`)
- New-pair state-space broadening: ratios 0.033 and 0.035, recorded PASS. (`results/family1e_state_broadening_new_pairs/RESULT.json:pairs.S1_fkpp_r0.5.broadS.median_R, pairs.S1_grayscott_1sp.broadS.median_R`)
- Order sensitivity: trajectory S was 0.396 versus 1.653, PASS; endpoint S was 3.431 versus 3.403, FAIL. (`results/family3_baselines/RESULT.json:results.ensemble.observables.trajectory.S_learner.mean, results.ensemble.observables.trajectory.baselines.order_blind.S_baseline.mean`)
- Initial-condition ratios were 0.241 and 0.179. (`results/family5_ic_robustness/RESULT.json:results.family1_spectral_grf_power_law.median_contrast_to_cross_ic, results.family2_bump_localized.median_contrast_to_cross_ic`)
- Test-time splitting comparison: narrow oracle 376.929, broad oracle 85.684, exhaustive splitting 1376.858. (`results/family6_serrano_baseline/RESULT_SMOKE.json:R.narrow_oracle.R.median, R.broad_oracle.R.median, R.serrano_narrow_lie_causal/exhaustive.R.median`)
- Resolution x horizon ratios were 0.151, 0.047, 0.039; 0.139, 0.042, 0.034; and 0.151, 0.050, 0.046. (`results/family2_resolution_horizon/RESULT.md`)
- U-Net broadening ratio was 0.101, PASS. (`results/family4b_unet_broadS/RESULT.json:P3_broadening_repairs.ratio_broadS_to_narrow`)
- Four-operator repair ratio was 0.135, PASS; word recovery was observed 1.000 vs 0.715, attainable maximum 0.285, recorded FAIL; the 2-D run reported dynamic range 66.8 and no repair ratio. (`results/family8_complex_algebra/RESULT.json:E1_P3_broadening_repairs.aggregated_repair_ratio_R, E2`; `results/family7_2d_generality/RESULT.json`; `results/family5b_ic_repair/RESULT.md`)

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, Arnav Garg.
