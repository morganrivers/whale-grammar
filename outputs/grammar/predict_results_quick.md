# 5-fold next-token prediction on unified corpus (bits/token) (quick smoke test)

Sequence-level KFold (5 folds, no within-sequence leakage). Target = `Coda1` (V = 102 including PAD). Context K = 4 past codas. Metric = held-out cross-entropy in **bits/token** (log₂); lower is better. Perplexity = 2^(bits/token).

**Apples-to-apples caveat.** The prior `~/Code/whale-gpt @ claude/whale-language-research-tEudI` benchmark used a rhythm·tempo·orn·rubato compound `Token` with V≈207 on the Sharma-only corpus (~4,800 codas). This run targets the rhythm-class integer `Coda1` directly on the unified corpus (~38k codas, V=102), so absolute bits/token are not directly comparable to the 4.63 figure.

| # | model | params | bits/token (↓) | perplexity (↓) | accuracy |
|---|-------|-------:|---------------:|---------------:|---------:|
| 0 | majority (smoothed unigram) | — | 5.660 ± 1.254 | 50.58 | 0.121 ± 0.121 |
| 1 | Markov-1 | — | 4.523 ± 1.130 | 23.00 | 0.283 ± 0.101 |
| 2 | Markov-2 | — | 5.245 ± 1.351 | 37.94 | 0.264 ± 0.120 |

## Per-fold bits/token

| fold | majority (smoothed unigram) | Markov-1 | Markov-2 |
|------|---:|---:|---:|
| 0 | 6.414 | 4.986 | 5.799 |
| 1 | 4.472 | 3.699 | 4.255 |
| 2 | 4.022 | 2.893 | 3.267 |
| 3 | 5.973 | 4.871 | 5.742 |
| 4 | 7.422 | 6.168 | 7.164 |

## Takeaway

- The best held-out model is **Markov-1** at 4.523 bits/token (perplexity ≈ 23.00).
- The unigram majority baseline scores 5.660 bits/token; the best model saves **1.14 bits/token** (perplexity drops 50.6 → 23.0).
- Markov-1 alone is already strong (4.523 bpt). Quick mode skips the learned models — re-run without --quick for the MLP / Emb-MLP / MiniTransformer numbers.
