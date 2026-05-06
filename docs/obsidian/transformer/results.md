---
tags:
  - transformer
  - results
summary: All bits-per-token tables in one place — generation 1 baselines, generation 2 ablation, generation 3 schema redesign
created: 2026-05-06
updated: 2026-05-06
---

# Results

Every benchmark on `data/classified/whale_dialogues.csv`. All sequence-level KFold (no within-sequence leakage), `random_state=42`. Target = `Coda` (rhythm-class integer). Metric = held-out cross-entropy in **bits/token** (log₂); lower is better. Perplexity = 2^(bits/token).

The auto-generated MD outputs live under `outputs/grammar/`; this page is a curated index.

## Generation 1 — initial baselines (M0–M7)

8 models × 5-fold CV, K=8, V=132 incl. PAD. Source: `outputs/grammar/predict_results_unified.md`.

| # | model | params | bits/token (↓) | perplexity | accuracy |
|---|---|---:|---:|---:|---:|
| 0 | majority (smoothed unigram) | — | 4.838 ± 0.246 | 28.59 | 0.254 ± 0.056 |
| 1 | Markov-1 | — | 3.724 ± 0.260 | 13.21 | 0.431 ± 0.039 |
| 2 | Markov-2 | — | 4.110 ± 0.326 | 17.27 | 0.436 ± 0.039 |
| 3 | MLP-S (128, 64) | 151,939 | 3.278 ± 0.252 | 9.70 | 0.465 ± 0.037 |
| 4 | MLP-M (256, 128) | 320,131 | 3.267 ± 0.265 | 9.63 | 0.469 ± 0.040 |
| 5 | MLP-L (512, 256, 128) | 721,795 | 3.331 ± 0.264 | 10.07 | 0.462 ± 0.036 |
| 6 | Embedding-MLP (d=32, 256·128) | 119,940 | 3.274 ± 0.263 | 9.67 | 0.463 ± 0.040 |
| 7 | **MiniTransformer (2L, 4h, d=64)** | **117,508** | **3.186 ± 0.237** | **9.10** | **0.464 ± 0.037** |

Reproduce: `python -m src.grammar.predict_kfold`.

## Generation 2 — depth × DT × K-sweep ablation (B0–B4)

Source: `outputs/grammar/predict_results_ablation.md`.

| # | model | params | bits/token (↓) | perplexity |
|---|---|---:|---:|---:|
| B0 | MiniTransformer 2L, 4h, d=64, K=8 (no DT) | 117,508 | 3.181 ± 0.236 | 9.07 |
| B1 | MiniTransformer 2L, 4h, d=64, K=8 + DT | 117,700 | 3.175 ± 0.238 | 9.03 |
| B2 | MiniTransformer 4L, 4h, d=64, K=8 (no DT) | 217,476 | 3.190 ± 0.243 | 9.12 |
| B3 | MiniTransformer 4L, 4h, d=64, K=8 + DT | 217,668 | 3.180 ± 0.243 | 9.07 |
| B4 | **whale-gpt-main shape: 3L, 8h, d=32, K=25 + DT** | 47,588 | **3.170 ± 0.246** | **9.00** |

Reproduce: `python -m src.grammar.predict_kfold --ablation`.

Read: every variant lives inside the 0.024 bpt gap between B4 and B2. DT helps a tiny amount (B0 → B1 = −0.006), depth alone hurts a tiny amount (B0 → B2 = +0.009), depth + DT roughly cancel. The whale-gpt-main shape (B4: 3 layers, 8 heads, d=32, K=25) is the best variant *and* the smallest at 47.6 k parameters — but the absolute saving over the 2L baseline (0.011 bpt) is well below the 0.24 bpt fold-variance floor.

## Generation 3 — schema redesign (R0/R1/R2)

The redesign experiment swaps the single-channel `Coda + DT` input for a multi-channel concatenation of per-column embeddings (Whale, Coda, Orn, Sync, Duration, TimeDelta_log, has_timestamps), with optional multi-task heads. All variants at 3L 8h d=32 K=25, fast training mode.

Source: `outputs/grammar/predict_results_redesign.md` (R0+R1) and `predict_results_redesign_R0R2.md` (R0+R2 archive).

| # | model | params | bits/token (↓) | perplexity | accuracy |
|---|---|---:|---:|---:|---:|
| R0 | R0 control (B4 mimic): single-channel Coda + DT | 47,588 | 3.215 ± 0.238 | 9.29 | 0.458 ± 0.058 |
| R1 | R1: multi-channel inputs, Coda CE only | 50,496 | 3.244 ± 0.225 | 9.47 | 0.448 ± 0.060 |
| R2 | R2 redesign: multi-channel + multi-task | 50,496 | 3.319 ± 0.222 | 9.98 | 0.440 ± 0.050 |

Reproduce: `python -m src.grammar.predict_kfold --redesign --variants R0,R1,R2`.

Read: **R0 wins.** Adding multi-channel inputs (R1) hurts by 0.029 bpt; adding multi-task heads on top (R2) hurts by another 0.075 bpt. See [[transformer/schema-redesign]] for the bear case and the per-fold deltas.

## Smoothed-classical baselines (S1–S3)

Source: `outputs/grammar/predict_results_smoothed.md`.

| # | model | params | bits/token (↓) | perplexity |
|---|---|---:|---:|---:|
| S1 | Modified Kneser-Ney 5-gram | 47,188 | 3.568 ± 0.250 | 11.86 |
| S2 | **Modified KN 5-gram + recency cache (size 320, λ=0.25)** | — | **3.324 ± 0.228** | **10.01** |
| S3 | PPM-D, max order 7 | 110,927 | 3.883 ± 0.289 | 14.75 |

Reproduce: `python -m src.grammar.predict_smoothed`.

The KN+cache model gets within 0.14 bpt of the MiniTransformer at zero attention. See [[transformer/smoothed-baselines]] for why a recency cache punches above its weight on this corpus.

## Cross-corpus head-to-head (whale vs CHILDES UK)

Same architecture, both sides. 3-fold CV on `whale_dialogues.csv` and `childes_dialogues.csv`. Source: `outputs/grammar/childes_vs_whale.md`.

### whale (V=467 compound vocab, 488 sequences, 38,840 tokens)

| model | params | bpt | perplexity | accuracy |
|---|---:|---:|---:|---:|
| majority (smoothed unigram) | — | 6.570 ± 0.106 | 95.01 | 0.097 |
| Markov-1 | — | 5.549 ± 0.125 | 46.80 | 0.299 |
| MiniTransformer-DT (2L, 4h, d=64, K=8 + DT) | 160,915 | **4.541 ± 0.103** | **23.28** | 0.326 |

### CHILDES UK (V=1602 lemma vocab, 405 sequences, 39,318 tokens)

| model | params | bpt | perplexity | accuracy |
|---|---:|---:|---:|---:|
| majority (smoothed unigram) | — | 7.832 ± 0.040 | 227.84 | 0.065 |
| Markov-1 | — | 8.084 ± 0.059 | 271.31 | 0.161 |
| MiniTransformer-DT (2L, 4h, d=64, K=8 + DT) | 307,330 | **6.860 ± 0.073** | **116.18** | 0.180 |

### Side-by-side compression

| source | V | majority bpt | MiniTfm-DT bpt | savings (bits) | fraction of unigram entropy |
|---|---:|---:|---:|---:|---:|
| whale | 467 | 6.570 | 4.541 | 2.029 | 0.691 |
| CHILDES UK | 1602 | 7.832 | 6.860 | 0.972 | 0.876 |

Whale is *more compressible* than CHILDES at this granularity, on the same 39k-token budget. Note: whale uses the compound vocab (V=467) here, not the rhythm-class-only V=131 of generations 1–3. See [[transformer/childes-comparison]] for the design choices that make this comparison apples-to-apples.

## Notable comparisons

| comparison | bits | what it says |
|---|---:|---|
| M0 majority → M7 MiniTransformer | −1.65 | The headline. K=8 of attention saves 1.65 bits over unigram. |
| M1 Markov-1 → M7 | −0.54 | Beyond bigrams, K=8 buys 0.54 bits — half the model's edge. |
| M7 → B4 (whale-gpt-main shape, K=25 + DT) | −0.016 | Best architecture saves a tiny amount over the 2L baseline; inside fold variance. |
| M7 → S2 (KN+cache) | +0.14 | A recency cache on a 5-gram is most of M7's compression. |
| R0 → R2 multi-channel multi-task | +0.117 | Adding speaker/sync/orn channels + aux losses *hurts*. |
| whale (V=467) → CHILDES UK (V=1602) fraction-of-unigram | 0.691 / 0.876 | Whale at this granularity has more learnable structure than English at lemma granularity, normalized by entropy floor. |

## How to compare runs sanely

- **Always normalize by entropy floor.** A change in V dominates absolute bpt; the fraction of unigram entropy (= MiniTfm bpt / unigram bpt) cancels it out.
- **5-fold σ ≈ 0.24 bpt** on this corpus. Differences below that floor are noise.
- **Per-fold tables in the auto-generated MDs** are usually more revealing than the means: outliers (fold 1 systematically harder than fold 0) say more about the corpus than the model.

## Related

- [[transformer/transformer]] — model architecture menu and code map.
- [[transformer/schema-redesign]] — why R2 lost.
- [[transformer/smoothed-baselines]] — full KN/PPM-D writeup.
- [[transformer/childes-comparison]] — same architecture on Eng-UK lemma stream.
- [[interp/interp]] — what the M7 winner is doing internally.
