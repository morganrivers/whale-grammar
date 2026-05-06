# 5-fold smoothed-classical models — KN 5-gram, KN+cache, PPM-D

Sequence-level KFold (5 folds, random_state=42; same split as `predict_kfold.py`). Target = `Coda`. Metric = held-out cross-entropy in **bits/token** (log₂); lower is better. Perplexity = 2^(bits/token).

Compare against the existing benchmark in `predict_results_unified.md` — the best transformer to date is **MiniTransformer M7 at 3.186 bpt** (perplexity ≈ 9.10).

| # | model | params | bits/token (↓) | perplexity (↓) | accuracy |
|---|-------|-------:|---------------:|---------------:|---------:|
| S1 | Modified Kneser-Ney 5-gram | 47,188 | 3.568 ± 0.250 | 11.86 | 0.429 ± 0.055 |
| S2 | Modified KN 5-gram + sequence-level cache (size 320, λ=0.25) | — | 3.324 ± 0.228 | 10.01 | 0.442 ± 0.058 |
| S3 | PPM-D, max order 7 | 110,927 | 3.883 ± 0.289 | 14.75 | 0.391 ± 0.055 |

## Per-fold bits/token

| fold | Modified Kneser-Ney 5-gram | Modified KN 5-gram + sequence-level cache (size 320, λ=0.25) | PPM-D, max order 7 |
|------|---:|---:|---:|
| 0 | 3.181 | 3.020 | 3.455 |
| 1 | 3.735 | 3.520 | 4.058 |
| 2 | 3.662 | 3.462 | 3.975 |
| 3 | 3.876 | 3.543 | 4.265 |
| 4 | 3.389 | 3.074 | 3.661 |

## Takeaway

- Best smoothed-classical model: **Modified KN 5-gram + sequence-level cache (size 320, λ=0.25)** at 3.324 bits/token (perplexity ≈ 10.01).
- Reference points from `predict_results_unified.md`: Markov-2 (Laplace α=0.5) = 4.110 bpt, Markov-1 = 3.724 bpt, MLP-M = 3.267 bpt, MiniTransformer M7 = **3.186 bpt**.
