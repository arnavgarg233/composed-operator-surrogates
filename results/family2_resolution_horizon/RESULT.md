# Family 2 result

**Verdict: PASS**

| N | legs | median R narrow | narrow range | median R broad | broad range | broad/narrow | G1 | P3 | status |
|---:|---:|---:|:---|---:|:---|---:|:---:|:---:|:---|
| 128 | 1 | 123.005 | [70.725, 187.823] | 18.560 | [14.335, 22.323] | 0.151 | PASS | PASS | complete |
| 128 | 2 | 212.584 | [103.588, 424.696] | 10.020 | [7.139, 11.686] | 0.047 | PASS | PASS | complete |
| 128 | 3 | 268.547 | [107.797, 697.850] | 10.508 | [6.413, 14.242] | 0.039 | PASS | PASS | complete |
| 256 | 1 | 115.405 | [74.993, 197.865] | 15.984 | [11.448, 18.079] | 0.139 | PASS | PASS | complete |
| 256 | 2 | 208.984 | [120.416, 432.504] | 8.765 | [6.575, 10.838] | 0.042 | PASS | PASS | complete |
| 256 | 3 | 271.936 | [120.865, 650.386] | 9.232 | [6.001, 12.301] | 0.034 | PASS | PASS | complete |
| 512 | 1 | 82.397 | [54.303, 128.433] | 12.453 | [9.598, 14.619] | 0.151 | PASS | PASS | complete |
| 512 | 2 | 138.260 | [77.609, 278.031] | 6.932 | [5.698, 8.788] | 0.050 | PASS | PASS | complete |
| 512 | 3 | 171.873 | [80.248, 455.201] | 7.829 | [4.795, 9.182] | 0.046 | PASS | PASS | complete |

## Gates

- MONOTONE: rho=0.900, threshold >= 0.5; PASS.
- Negative control: CLEAN; paired Q median 0.992, range [0.985, 1.089].

## Runtime

- N=128 (legs 1, 2, 3): fitted and evaluated on the local Apple-silicon Mac, JAX CPU backend (JAX_PLATFORMS=cpu, arm64).
- N=256 (legs 1, 2, 3): fitted and evaluated on the local Apple-silicon Mac, JAX CPU backend (JAX_PLATFORMS=cpu, arm64).
- N=512 (legs 1, 2, 3): fitted and evaluated on RunPod pod c1h12kov47rcm4, NVIDIA A40, JAX CUDA backend (JAX_PLATFORMS=cuda, x86_64).

The N=512 row is a declared runtime departure: the local CPU driver needed roughly ten minutes per fit at that grid, so the twenty N=512 fits and their evaluations were moved to a rented A40 under the CUDA backend. Package pins (jax/jaxlib 0.8.1, exponax 0.2.0, equinox 0.13.8), model, optimiser, schedule, seeds, corpus generator, metric and gate thresholds are identical to the CPU rows; only the XLA backend and the host differ. Both arms of every cell share one runtime, so no narrow-versus-broad contrast spans backends. Four N=512 narrow seeds fitted earlier on CPU were refitted on the pod rather than mixed into the row; the originals are preserved under results/preserved_cpu_partial/.

## Honest reading

All predeclared cells and controls passed: the repair survived the resolution and handoff-horizon hold-outs under the fixed gates.
