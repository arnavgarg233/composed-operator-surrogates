### Held-out cells (sorted by D3)

| cell | R | seeds | log₁₀R | **D3** | **C3** | D̄ (Mahalanobis) | cov_S | k | blind |
|---|---|---|---|---|---|---|---|---|---|
| `FK05_broad` | 16.437 | 5 | +1.216 | 0.16034 | 0.0000 | 8.7101 | 0.5664 | 11 | **yes** |
| `P1_broadS` | 2.569 | 10 | +0.410 | 0.27964 | 0.0859 | 8.2245 | 0.7422 | 11 | no |
| `F6_P1_broad` | 18.385 | 2 | +1.264 | 0.27971 | 0.0000 | 7.8962 | 0.7852 | 11 | no |
| `CH_broadS` | 0.787 | 10 | -0.104 | 0.28736 | 0.8711 | 6.6204 | 0.8281 | 10 | no |
| `FK05_narrow` | 58.676 | 10 | +1.768 | 1.72747 | 0.0000 | 8.6547 | 0.5625 | 10 | **yes** |
| `F6_P1_narrow` | 90.681 | 2 | +1.958 | 3.31308 | 0.0000 | 7.7082 | 0.8203 | 10 | no |

### The six selection cells (reported, gate nothing)

| cell | R | seeds | log₁₀R | **D3** | **C3** | D̄ (Mahalanobis) | cov_S | k |
|---|---|---|---|---|---|---|---|---|
| `P1_narrow` | 110.234 | 5 | +2.042 | 3.31308 | 0.0000 | 7.7082 | 0.8203 | 10 |
| `P1_broad` | 17.650 | 5 | +1.247 | 0.27971 | 0.0000 | 7.8962 | 0.7852 | 11 |
| `P2_narrow` | 177.179 | 10 | +2.248 | 5.24684 | 0.0000 | 1.9072 | 1.0000 | 10 |
| `P2_broad` | 0.100 | 10 | -0.999 | 0.01214 | 1.0000 | 1.9429 | 1.0000 | 10 |
| `CH_narrow` | 46.278 | 10 | +1.665 | 0.23047 | 0.6328 | 5.9442 | 0.9102 | 10 |
| `CH_broad` | 35.799 | 10 | +1.554 | 0.17635 | 0.0000 | 6.0744 | 0.9102 | 11 |

### Correlations

| set | n | statistic | Spearman ρ | p |
|---|---|---|---|---|
| **held-out (the gate)** | 6 | **S3 = D3** | **+0.6000** | 0.2080 |
| held-out | 6 | PCA-99 Mahalanobis D̄ | **+0.1429** | 0.7872 |
| held-out | 6 | D3f (normalised residual) | **+0.6000** | 0.2080 |
| held-out minus family-6 duplicates | 4 | S3 = D3 | **+0.2000** | 0.8000 |
| held-out minus family-6 duplicates | 4 | PCA-99 Mahalanobis D̄ | **+0.8000** | 0.2000 |
| six selection cells | 6 | S3 = D3 | **+0.8286** | 0.0416 |
| six selection cells | 6 | PCA-99 Mahalanobis D̄ | **-0.2571** | 0.6228 |
| six selection cells | 6 | D3f | **+0.8286** | 0.0416 |

### Gates

| gate | definition | threshold | measured | result |
|---|---|---|---|---|
| **PRED-S3-A** | Spearman ρ(log₁₀R, D3) over held-out cells | ≥ 0.8, ≥ 6 cells | **+0.6000** (p = 0.2080), n = 6 | **FAIL** |
| **PRED-S3-B** | C3(broad-S) > C3(narrow), both CH and P1 | strict, both | CH 0.6328 → 0.8711; P1 0.0000 → 0.0859 | **PASS** |

**VERDICT: FAIL (PARTIAL)**
