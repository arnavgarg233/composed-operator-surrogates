# Supplied-dictionary modules recover diffusion-reaction endpoint contrasts outside training-mean support

An operator surrogate fitted on one process and composed downstream of another is evaluated
at states whose spatial means sit outside its training support. Supplying the dictionary of
states the composition produces, at the same sample count, recovers the composed endpoint
contrast. This repository measures that recovery on two noncommuting diffusion-reaction
pairs and two architectures, and carries the check that says which observable resolves the
contrast.

## Headline results

- Supplying the broadened dictionary at fixed sample count cuts a Fourier neural operator's
  ten-step degradation from a median factor of `110.2` to `17.65` across five seeds
  (`results/tables/baseline/BASELINE_RESULT.json`, `narrow.median_R` and `broad.median_R`).
- On the diffusion / Allen-Cahn pair the supplied support covers every switch state,
  `48/48` units inside it against `0/48` at the narrow support
  (`results/tables/second_pair/GEOMETRIC_REPAIR_RESULT.json`).
- The trajectory observable resolves the order contrast `11.670` times better than the
  endpoint, and two separately fitted primitives composed autoregressively give an observed
  dissociation of `12.784` (`results/tables/baseline/DISSOCIATION_RESULT.json`).

## Install and reproduce

Install CPython 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
bash reproduce.sh
```

Nothing reaches the network. The seven figures reproduce byte for byte offline.

## Outputs

The replay imports the package, runs the tests and artifact checks, and compares a
regenerated checksum file against the committed one.

## Repository map

- `src/composed_operator_surrogates/`: path resolution into the published tree, and the checks
- `scripts/`: the code the published results were produced by, and the checksum builder
- `configs/` and `data/`: the runtime every published number was produced under, and the five
  composed-trajectory files `composed_seed{0..4}.npz` with the five weight files
  `weights_seed{0..4}.pt` in `data/learner_artifacts/`, about 87 MB
- `results/`: the seven PDF figures with their receipt at
  `results/tables/FIGURE_RECEIPT.json`, the result tables, and a SHA-256 for every published
  file in `results/CHECKSUMS.txt`
- `tests/`: automated checks

## License

Released under the [MIT License](LICENSE); external records remain subject to their original terms.
