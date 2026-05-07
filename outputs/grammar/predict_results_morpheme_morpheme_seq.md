# 5-fold prediction — morpheme_seq tokenisation

Tokenisation: **morpheme_seq**.  Vocabulary V = 369 (including PAD).  Context K = 8 past tokens.  Metric = held-out cross-entropy in **bits/token** (log₂); lower is better.

**per-coda bpt** aggregates morpheme log-probs within each coda's Morfessor segmentation (chain rule).  Directly comparable to the compound-token baseline bpt from `predict_results_unified.json`.

| # | model | params | bits/token (morpheme) | per-coda bpt (↓) | perplexity (↓) | accuracy |
|---|---:|---:|---:|---:|---:|---:|
| M0 | majority (smoothed unigram) | — | 7.141 ± 0.133 | 8.337 ± 0.301 | 141.16 | 0.080 ± 0.033 |
| M1 | Markov-1 | — | 6.279 ± 0.155 | 7.333 ± 0.354 | 77.66 | 0.211 ± 0.023 |
| M2 | Markov-2 | — | 7.128 ± 0.159 | 8.323 ± 0.372 | 139.84 | 0.187 ± 0.022 |
| M3 | MLP-S (128, 64) | 407,919 | 5.658 ± 0.185 | 6.609 ± 0.368 | 50.50 | 0.217 ± 0.020 |
| M4 | MLP-M (256, 128) | 831,855 | 5.617 ± 0.172 | 6.561 ± 0.362 | 49.07 | 0.218 ± 0.022 |
| M5 | MLP-L (512, 256, 128) | 1,714,799 | 5.754 ± 0.248 | 6.723 ± 0.435 | 53.97 | 0.211 ± 0.025 |
| M6 | Embedding-MLP (d=32, 256·128) | 157,936 | 5.702 ± 0.172 | 6.660 ± 0.360 | 52.05 | 0.215 ± 0.019 |
| M7 | MiniTransformer (2L, 4h, d=64) | 147,952 | 5.524 ± 0.153 | 6.453 ± 0.344 | 46.02 | 0.223 ± 0.020 |

## Per-fold bits/token

| fold | majority (smoothed unigram) | Markov-1 | Markov-2 | MLP-S (128, 64) | MLP-M (256, 128) | MLP-L (512, 256, 128) | Embedding-MLP (d=32, 256·128) | MiniTransformer (2L, 4h, d=64) |
|------|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.313 | 6.441 | 7.347 | 5.873 | 5.824 | 6.148 | 5.931 | 5.697 |
| 1 | 7.195 | 6.358 | 7.184 | 5.820 | 5.720 | 5.855 | 5.830 | 5.641 |
| 2 | 7.102 | 6.380 | 7.174 | 5.711 | 5.711 | 5.784 | 5.737 | 5.602 |
| 3 | 6.911 | 6.010 | 6.866 | 5.407 | 5.371 | 5.453 | 5.497 | 5.332 |
| 4 | 7.185 | 6.206 | 7.068 | 5.480 | 5.458 | 5.530 | 5.514 | 5.349 |

## Takeaway

- Best model: **MiniTransformer (2L, 4h, d=64)** at 5.524 bits/token (perplexity ≈ 46.02).
- Per-coda bpt for best model: **6.453 ± 0.344** bits/coda.
