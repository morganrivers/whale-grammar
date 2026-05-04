# 5-fold transformer ablation on unified corpus (depth × DeltaTime × whale-gpt-main shape)

Sequence-level KFold (5 folds, no within-sequence leakage). Target = `Coda1` (V = 132 including PAD). Context K varies per variant. Metric = held-out cross-entropy in **bits/token** (log₂); lower is better. Perplexity = 2^(bits/token).

| # | model | params | bits/token (↓) | perplexity (↓) | accuracy |
|---|-------|-------:|---------------:|---------------:|---------:|
| B0 | MiniTransformer 2L, 4h, d=64, K=8 (no DT) | 117,508 | 3.181 ± 0.236 | 9.07 | 0.462 ± 0.041 |
| B1 | MiniTransformer 2L, 4h, d=64, K=8 + DT | 117,700 | 3.175 ± 0.238 | 9.03 | 0.465 ± 0.042 |
| B2 | MiniTransformer 4L, 4h, d=64, K=8 (no DT) | 217,476 | 3.190 ± 0.243 | 9.12 | 0.464 ± 0.038 |
| B3 | MiniTransformer 4L, 4h, d=64, K=8 + DT | 217,668 | 3.180 ± 0.243 | 9.07 | 0.466 ± 0.037 |
| B4 | whale-gpt-main shape: 3L, 8h, d=32, K=25 + DT | 47,588 | 3.170 ± 0.246 | 9.00 | 0.468 ± 0.035 |

## Per-fold bits/token

| fold | MiniTransformer 2L, 4h, d=64, K=8 (no DT) | MiniTransformer 2L, 4h, d=64, K=8 + DT | MiniTransformer 4L, 4h, d=64, K=8 (no DT) | MiniTransformer 4L, 4h, d=64, K=8 + DT | whale-gpt-main shape: 3L, 8h, d=32, K=25 + DT |
|------|---:|---:|---:|---:|---:|
| 0 | 2.815 | 2.784 | 2.813 | 2.790 | 2.755 |
| 1 | 3.537 | 3.504 | 3.550 | 3.515 | 3.490 |
| 2 | 3.112 | 3.096 | 3.110 | 3.086 | 3.082 |
| 3 | 3.287 | 3.298 | 3.315 | 3.325 | 3.300 |
| 4 | 3.153 | 3.196 | 3.160 | 3.186 | 3.226 |

## Takeaway

- The best held-out model is **whale-gpt-main shape: 3L, 8h, d=32, K=25 + DT** at 3.170 bits/token (perplexity ≈ 9.00).
- 2L baseline (B0) is 3.181 bpt; the best variant saves **0.010 bits/token** over it. Compare individual rows to see depth vs DT vs longer-context contributions.
