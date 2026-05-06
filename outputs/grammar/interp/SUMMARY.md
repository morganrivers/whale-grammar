# Whale vs CHILDES — M7 mechanistic interp side-by-side

Architecture: 2-layer / 4-head / d=64 / d_head=16 / d_mlp=256 / K=8 causal HookedTransformer (TransformerLens 3.1.0). Trained from scratch on each corpus.

## Headline numbers

| metric | whale | childes |
|---|---:|---:|
| coda vocabulary size | 467 | 1605 |
| DT bucket count | 6 | 3 |
| val coda bpt (final position) | 5.0086 | 8.1099 |
| val coda acc (final position) | 0.2982 | 0.0751 |
| val DT-bucket bpt (final position) | 0.3374 | 0.9960 |
| val DT-bucket acc (final position) | 0.9642 | 0.7707 |
| n eval spans (= seed-derived) | 335 | 261 |

## Per-head diagnostics

Numbers are read directly from `<source>/summary.json`. **dist** = mean attention distance (q − k); larger = looks further back. **ent** = attention entropy in bits; lower = more peaked. **prev** = fraction of attention to position t − 1. **ind** = induction score on random-repeat probe. Reference: K = 8, so dist is bounded above by ~7.

| L | H | whale dist | whale ent | whale prev | whale ind | childes dist | childes ent | childes prev | childes ind |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 2.07 | 1.98 | 0.23 | 0.17 | 2.23 | 1.99 | 0.25 | 0.15 |
| 0 | 1 | 2.00 | 2.01 | 0.24 | 0.14 | 2.07 | 2.05 | 0.24 | 0.15 |
| 0 | 2 | 1.95 | 2.00 | 0.20 | 0.13 | 1.92 | 2.03 | 0.22 | 0.13 |
| 0 | 3 | 2.05 | 2.01 | 0.29 | 0.15 | 2.10 | 2.01 | 0.27 | 0.16 |
| 1 | 0 | 1.94 | 2.10 | 0.23 | 0.15 | 1.80 | 2.08 | 0.25 | 0.13 |
| 1 | 1 | 1.93 | 2.11 | 0.25 | 0.14 | 2.11 | 2.09 | 0.24 | 0.15 |
| 1 | 2 | 2.19 | 2.08 | 0.24 | 0.15 | 2.07 | 2.11 | 0.24 | 0.15 |
| 1 | 3 | 1.91 | 2.09 | 0.25 | 0.15 | 1.91 | 2.12 | 0.25 | 0.13 |

## Logit lens — top-1 next-token accuracy by layer

| stage | whale | childes |
|---|---:|---:|
| pre | 0.000 | 0.023 |
| after L0 | 0.066 | 0.073 |
| after L1 | 0.152 | 0.088 |

## Files

- `outputs/grammar/interp/whale/`
    - `head_summary.png` — per-head bar charts (4 metrics)
    - `logit_lens.png`
    - `token_embeddings.png`
    - `attention_seed*.html` — open in a browser; CircuitsVis interactive attention widget per seed × layer
    - `continuations.md` — 3 random continuations per seed (temperature 0.9, top-k 40)
    - `summary.json`
- `outputs/grammar/interp/childes/`
    - `head_summary.png` — per-head bar charts (4 metrics)
    - `logit_lens.png`
    - `token_embeddings.png`
    - `attention_seed*.html` — open in a browser; CircuitsVis interactive attention widget per seed × layer
    - `continuations.md` — 3 random continuations per seed (temperature 0.9, top-k 40)
    - `summary.json`

## Analysis

### 1. Both corpora compress, but the *shapes* differ

* Whale (compound V=467): final-position 5.01 bpt vs uniform floor log₂V = 8.87 → **3.86 bits saved / token**.  * CHILDES (lemma V=1605): 8.11 bpt vs floor 10.65 → **2.54 bits saved / token**.

Whale at this granularity is more compressible than English at lemma granularity, on the same model and same training-token budget (~39k). Standard caveat: this includes the unigram-Zipf component (one rhythm class — `1+1+3|t1|o0|r0` — accounts for ~16 % of all whale tokens, which inflates the savings). The cleanest decomposition would compare against smoothed unigram, not uniform. See `outputs/grammar/predict_results_smoothed.md` for the smoothed-baseline numbers on whale; CHILDES doesn't have an equivalent yet.

### 2. Whale uses both layers; CHILDES barely uses layer 1

Decomposing the logit-lens trajectory into per-layer increments in top-1 next-token accuracy:

* **Whale**: pre 0.000 → after L0 0.066 → after L1 0.152. Layer 1 contributes +0.087 on top of L0, i.e. **57 % of the final accuracy comes from L1**.
* **CHILDES**: pre 0.023 → after L0 0.073 → after L1 0.088. L1 contributes only +0.015, i.e. **17 %**.

**This is the most striking cross-corpus asymmetry in the report.** On English at K=8 the next lemma is dominated by adjacent-bigram statistics (e.g. *want* → *to*, *have* → *not*); layer 0 attention + MLP can already pick those up, and layer 1 has little to refine. On whale the compound rhythm·tempo·orn·rubato distribution at K=8 evidently *is not* approximated as well by adjacent bigrams — the model needs a second round of context-mixing to commit to a prediction. Practically: an L=1 transformer would lose far more accuracy on whale than on CHILDES.

### 3. No specialized circuits at K = 8

Maximum **prev-token score** across all 8 heads: whale 0.29, CHILDES 0.27. The uniform-causal-attention baseline at K = 8 is ≈ 0.25, so neither model has a head that meaningfully concentrates on position t − 1 (a clean prev-token head would be ≥ 0.7).

Maximum **induction score** on the random-repeat probe: whale 0.17, CHILDES 0.16. Uniform baseline = 1/K ≈ 0.125. Both corpora hover within ~0.05 of random — no head is implementing the canonical `[A B … A] → B` circuit.

**This is a clean negative result.** Olsson et al. (2022) report induction-head emergence in 2-layer transformers, but their setup uses much longer contexts (K ≥ 32). At K = 8 there isn't enough room for a repeated subsequence to fit twice within a window often enough for induction to pay rent during training. So the absence of induction here is consistent with the architecture, not evidence about the corpora.

### 4. The DT head learns very different things on the two corpora

Whale DT bucket: bpt 0.34, acc 96.4 %. CHILDES DT bucket: bpt 1.00, acc 77.1 %.

These numbers are not directly comparable. Whale DT is dominated by the `missing` bucket (76 % of tokens — every Hersh-corpus row lacks timestamps) so a unigram predictor already gets ~76 % accuracy; the model's 96 % is a real gain but the bar is low. CHILDES DT has three roughly informative buckets (intra/period/switch), so the model has actually learned where utterance and speaker boundaries fall — which is what powers the readable `*SPEAKER:` rendering of generated continuations.

### 5. Take-aways

1. **Cross-corpus depth use is asymmetric.** On English the lemma-level prediction at K = 8 is essentially solved by L0; on whale (compound vocab) L1 supplies the majority of the final answer. If you want to argue that whale tokens carry structure beyond a Markov-1 model, the L1 share of final accuracy is the cleanest interp-side evidence.
2. **No emergent circuits at this scale.** K = 8 is too short to find induction heads; the head-level diagnostics are essentially uninformative. To replicate the Olsson result you'd retrain at K = 32 or K = 64 and re-run `interp_compare`. The infrastructure is unchanged — only `m7_hooked.N_CTX` and `train_for_interp.--bs` would move.
3. **The readable continuations are the headline qualitative output.** `whale/continuations.md` shows the model committing hard to `1+1+3` (the modal rhythm class) with realistic Δt patterns; `childes/continuations.md` produces locally fluent CHILDES-style transcripts (`*MOT:	oh that be a little spider`) that are syntactically mostly OK and semantically random — a 2L/d=64 model trained on 39k lemmas would not be expected to do better.
