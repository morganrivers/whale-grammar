---
tags:
  - transformer
  - baselines
summary: Modified Kneser-Ney 5-gram + recency cache + PPM-D — classical smoothing baselines on whale_dialogues.csv
created: 2026-05-06
updated: 2026-05-06
---

# Smoothed-classical baselines

Three classical sequence-prediction models that should at least *try* to keep up with the MiniTransformer at the 38k-token scale of the unified whale corpus. Same input (`data/classified/whale_dialogues.csv`), same fold split (`KFold(n_splits=5, shuffle=True, random_state=42)`), same target (`Coda` rhythm-class integer), same metric (held-out bits/token).

Code: `src/grammar/predict_smoothed.py`. Output: `outputs/grammar/predict_results_smoothed.md`.

## Models

| # | model | description |
|---|---|---|
| S1 | Modified Kneser-Ney 5-gram | Chen & Goodman 1998 |
| S2 | Modified KN 5-gram + recency cache | Kuhn & De Mori 1990 — interpolate the 5-gram with a sequence-local cache |
| S3 | PPM-D, max order 7 | Howard 1993 — variable-order Markov with escape-based smoothing |

## Headline numbers

5-fold CV, V=131, K=8 implicit (5-gram = 4 prior tokens + the 1 we're predicting):

| # | model | params | bpt (↓) | perplexity | accuracy |
|---|---|---:|---:|---:|---:|
| S1 | Modified Kneser-Ney 5-gram | 47,188 | 3.568 ± 0.250 | 11.86 | 0.429 ± 0.055 |
| S2 | **Modified KN 5-gram + cache (size 320, λ=0.25)** | — | **3.324 ± 0.228** | **10.01** | 0.442 ± 0.058 |
| S3 | PPM-D, max order 7 | 110,927 | 3.883 ± 0.289 | 14.75 | 0.391 ± 0.055 |

For comparison, the same fold split produced these reference numbers ([[transformer/results]]):

| reference | bpt |
|---|---:|
| Markov-1 (Laplace α=0.5) | 3.724 |
| Markov-2 (Laplace α=0.5) | 4.110 |
| MLP-M (256, 128) | 3.267 |
| **MiniTransformer M7 (2L, 4h, d=64)** | **3.186** |

## Read

- **Markov-2 is *worse* than Markov-1 on this corpus.** With V=131 and only ~38k tokens, bigram-context cells are mostly empty or have count 1; Laplace wastes probability mass on impossible bigrams. KN was designed exactly for this regime — and **S1 (5-gram MKN) is 0.16 bpt better than Markov-1 with no cache**, which is the cleanest "smoothing matters" signal.
- **The recency cache is the single most valuable thing you can add.** Whale recordings are extremely repetitive (a single recording often dominated by 3–5 coda types in long runs). The cache term ("I just said `cn5`, I'll probably say `cn5` again") earns 0.244 bpt over plain MKN.
- **S2 lands within 0.14 bpt of M7.** A linear-time, no-parameter, classical model gets most of the way to the transformer. The transformer's edge — what attention buys beyond local n-gram counts + a recency cache — is small at this corpus scale.
- **PPM-D is worse than MKN.** The escape-based smoothing in PPM-D is famously good on text — but here it's 0.32 bpt worse than KN. The variable-order model's higher orders see too few unique contexts (the long-run repetition is a wash for PPM, but a lift for cache-augmented KN).

## Why a recency cache punches above its weight

In a recording dominated by a single repertoire — a Hersh "Plus-One" recording where most codas are 1+1+5 variants — the cache picks up on the within-recording unigram and biases predictions toward "the same coda as the last few you saw". A 320-token cache window with linear-interpolation weight λ=0.25 is small enough that it tracks recent context but doesn't drown out the global 5-gram statistics.

This is why aggregating `Hersh + DSWP + birth` into one corpus *helps* the cache: the cross-recording diversity lets the global 5-gram model learn the broad distribution, and the cache then refines toward the local recording.

## What the cache term *isn't*

It's not attention. The cache uses identity-of-recent-tokens as features but doesn't condition on positions or relative offsets. The fact that S2 is 0.14 bpt behind M7 — not 1.0 bpt — bounds how much "structure beyond local repetition + 5-gram counts" the MiniTransformer is finding. A pure-attention boost of 0.14 bpt is real but modest.

## Reproduce

```bash
python -m src.grammar.predict_smoothed
```

Wall time: ~5 min on CPU. The KN training pass is `O(n_tokens × n)` per fold; PPM-D is similar.

## Related

- [[transformer/results]] — full results in one place.
- [[transformer/transformer]] — code map and the model menu.
