# 5-fold next-token prediction on unified corpus (bits/token)

Sequence-level KFold (5 folds, no within-sequence leakage). Target = `Coda1` (V = 132 including PAD). Context K = 8 past codas. Metric = held-out cross-entropy in **bits/token** (log₂); lower is better. Perplexity = 2^(bits/token).

**Apples-to-apples caveat.** The prior `~/Code/whale-gpt @ claude/whale-language-research-tEudI` benchmark used a rhythm·tempo·orn·rubato compound `Token` with V≈207 on the Sharma-only corpus (~4,800 codas). This run targets the rhythm-class integer `Coda1` directly on the unified corpus (~38k codas, V=132), so absolute bits/token are not directly comparable to the 4.63 figure.

| # | model | params | bits/token (↓) | perplexity (↓) | accuracy |
|---|-------|-------:|---------------:|---------------:|---------:|
| 0 | majority (smoothed unigram) | — | 4.838 ± 0.246 | 28.59 | 0.254 ± 0.056 |
| 1 | Markov-1 | — | 3.724 ± 0.260 | 13.21 | 0.431 ± 0.039 |
| 2 | Markov-2 | — | 4.110 ± 0.326 | 17.27 | 0.436 ± 0.039 |
| 3 | MLP-S (128, 64) | 151,939 | 3.278 ± 0.252 | 9.70 | 0.465 ± 0.037 |
| 4 | MLP-M (256, 128) | 320,131 | 3.267 ± 0.265 | 9.63 | 0.469 ± 0.040 |
| 5 | MLP-L (512, 256, 128) | 721,795 | 3.331 ± 0.264 | 10.07 | 0.462 ± 0.036 |
| 6 | Embedding-MLP (d=32, 256·128) | 119,940 | 3.274 ± 0.263 | 9.67 | 0.463 ± 0.040 |
| 7 | MiniTransformer (2L, 4h, d=64) | 117,508 | 3.186 ± 0.237 | 9.10 | 0.464 ± 0.037 |

## Per-fold bits/token

| fold | majority (smoothed unigram) | Markov-1 | Markov-2 | MLP-S (128, 64) | MLP-M (256, 128) | MLP-L (512, 256, 128) | Embedding-MLP (d=32, 256·128) | MiniTransformer (2L, 4h, d=64) |
|------|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.492 | 3.417 | 3.660 | 2.869 | 2.846 | 2.911 | 2.840 | 2.799 |
| 1 | 5.246 | 4.184 | 4.658 | 3.648 | 3.652 | 3.721 | 3.649 | 3.531 |
| 2 | 4.747 | 3.632 | 4.054 | 3.195 | 3.149 | 3.228 | 3.211 | 3.150 |
| 3 | 4.917 | 3.802 | 4.201 | 3.332 | 3.355 | 3.395 | 3.394 | 3.285 |
| 4 | 4.786 | 3.585 | 3.979 | 3.343 | 3.335 | 3.401 | 3.274 | 3.166 |

## Takeaway

- The best held-out model is **MiniTransformer (2L, 4h, d=64)** at 3.186 bits/token (perplexity ≈ 9.10).
- The unigram majority baseline scores 4.838 bits/token; the best model saves **1.65 bits/token** (perplexity drops 28.6 → 9.1).
- Markov-1 alone is already strong (3.724 bpt). Larger models help further; the MiniTransformer / Emb-MLP can attend to the full 8-coda context.
