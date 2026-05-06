---
tags:
  - interp
  - overview
summary: What the trained MiniTransformer is doing internally — per-layer accuracy, induction-head check, lexical neighborhoods
created: 2026-05-06
updated: 2026-05-06
---

# Interpretability

The benchmarks in [[transformer/results]] show the MiniTransformer compresses the held-out coda stream. The interp work asks: **how**, and **how does that differ from the same model trained on a control corpus**?

The control is CHILDES UK English (~39k lemma tokens), tokenized via the `%mor` tier so the model is reading lemmas, not surface words. Same architecture both sides: 2L / 4h / d=64 / d_head=16 / d_mlp=256 / K=8 causal HookedTransformer (TransformerLens 3.1.0). Persisted at `outputs/grammar/checkpoints/{whale,childes_v1605}/`.

Auto-generated cross-corpus writeup: `outputs/grammar/interp/SUMMARY.md`.

## Headline numbers

| metric | whale | childes |
|---|---:|---:|
| coda vocabulary size | 467 (compound) | 1605 (lemma) |
| DT bucket count | 6 | 3 |
| val coda bpt (final position) | 5.0086 | 8.1099 |
| val coda acc (final position) | 0.298 | 0.075 |
| val DT-bucket bpt | 0.337 | 0.996 |
| val DT-bucket acc | 0.964 | 0.771 |
| eval spans | 335 | 261 |

## Five findings

### 1. Whale compresses more than English at this granularity

| source | V | uniform floor (log₂V) | val coda bpt | bits saved |
|---|---:|---:|---:|---:|
| whale | 467 | 8.87 | 5.01 | **3.86** |
| childes | 1605 | 10.65 | 8.11 | **2.54** |

Same architecture, same training-token budget (~39k). Caveat: a single rhythm class (`1+1+3|t1|o0|r0`) accounts for ~16 % of all whale tokens, which inflates the savings. The cleanest decomposition would compare against smoothed unigram instead of uniform — see [[transformer/smoothed-baselines]].

### 2. Whale uses both layers; CHILDES barely uses layer 1

Logit lens: apply the unembedding to the residual stream after each layer; track top-1 next-token accuracy.

| stage | whale | childes |
|---|---:|---:|
| pre | 0.000 | 0.023 |
| after L0 | 0.066 | 0.073 |
| after L1 | 0.152 | 0.088 |

- **Whale**: pre 0.000 → after L0 0.066 → after L1 0.152. Layer 1 contributes +0.087 — **57 % of the final accuracy**.
- **CHILDES**: pre 0.023 → after L0 0.073 → after L1 0.088. Layer 1 contributes only +0.015 — **17 %**.

This is the cleanest cross-corpus asymmetry in the report. On English at K=8 the next lemma is dominated by adjacent-bigram statistics (e.g. *want* → *to*); layer 0 picks those up, and layer 1 has little to refine. On whale the compound rhythm·tempo·orn·rubato distribution evidently isn't approximated as well by adjacent bigrams — the model needs a second round of context-mixing.

Practical implication: an L=1 transformer would lose far more accuracy on whale than on CHILDES.

### 3. No specialized circuits at K=8

Per-head diagnostics on the 8 heads (2 layers × 4 heads):

- **Maximum prev-token score** across all heads: whale 0.29, CHILDES 0.27. Uniform-causal-attention baseline at K=8 ≈ 0.25. Neither model has a head that meaningfully concentrates on position t−1 (a clean prev-token head would be ≥ 0.7).
- **Maximum induction score** on the random-repeat probe: whale 0.17, CHILDES 0.16. Uniform baseline = 1/K ≈ 0.125. Neither model has a head implementing the canonical `[A B … A] → B` circuit.

Clean negative result. Olsson et al. (2022) reported induction-head emergence in 2-layer transformers at K ≥ 32; at K=8 there isn't enough room for a repeated subsequence to fit twice within a window often enough for induction to pay rent during training. Replicating Olsson here would mean retraining at K=32 or K=64 (only `m7_hooked.N_CTX` and `train_for_interp.--bs` would move).

### 4. The DT head learns very different things on the two corpora

Whale DT bucket: bpt 0.34, acc 96.4 %. CHILDES DT bucket: bpt 1.00, acc 77.1 %.

These numbers aren't directly comparable. Whale DT is dominated by the **missing** bucket (76 % of tokens — every Hersh-corpus row lacks timestamps), so a unigram predictor already gets ~76 % accuracy; the model's 96 % is a real gain but the bar is low. CHILDES DT has three roughly informative buckets (`intra`, `period`, `switch`), so the model has actually learned where utterance and speaker boundaries fall — which is what powers the readable `*SPEAKER:` rendering of generated continuations.

### 5. Coda families surface in W_U

Connected components in the mutual-top-k cosine neighbor graph on the model's output embedding. Edges require both ends in each other's top-6 cosine neighbors and cosine ≥ 0.45. Two large clusters dominate — see `outputs/grammar/interp/whale/decoded.md`:

- **Cluster 1** — 20 codas, 9050 samples, mean Duration 1.22 s, mean Δt-to-prev 2.33 s, sync rate 0.026, ornament rate 0.005. Mostly Hersh-UNK speakers (76 distinct).
- **Cluster 2** — 13 codas, 4761 samples, mean Duration 0.56 s, mean Δt-to-prev 7.18 s, sync rate 0.023, ornament rate 0.025. Mostly Hersh-UNK speakers (79 distinct).

Cluster 1 is "long-Duration repeating codas with short Δt" — the steady within-recording chorus. Cluster 2 is "short-Duration codas with long Δt-to-prev" — the call-and-response signature. The model has discovered both in W_U *unsupervised on Coda CE alone*.

Mutual attention pairs (top by `√(mean attn a→b · mean attn b→a)`) similarly split into within-speaker (low cross-speaker fraction, short Δt) and cross-speaker (call-and-response) pairs. See `decoded.md` for the table.

## Code map

| script | what it produces |
|---|---|
| `src/grammar/train_for_interp.py` | persists checkpoints to `outputs/grammar/checkpoints/{whale,childes,childes_v1605,childes_v467}/` (model, dt_head, vocab, seeds, train log). |
| `src/grammar/m7_hooked.py` | TransformerLens config matching M7's architecture. |
| `src/grammar/interp_compare.py` | per-head attention diagnostics (distance, entropy, prev-token, induction); logit lens; PCA of W_E; cross-corpus `outputs/grammar/interp/SUMMARY.md`. |
| `src/grammar/interp_lexical.py` | embedding neighborhoods (W_E + W_U); attention co-occurrence; bigram-baseline divergence. |
| `src/grammar/interp_whale_decoded.py` | join lexical findings against per-row whale metadata (Duration, TimeDelta, Synchrony, Ornamentation, Whale). |
| `src/grammar/continuations.py` | autoregressive sampling from the locked checkpoint (T=0.9, top-k=40). |

## Output map

```
outputs/grammar/interp/
├── SUMMARY.md                         ← cross-corpus headline
├── whale/
│   ├── attention_seed*.html           ← CircuitsVis interactive widgets
│   ├── continuations.md               ← 3 random continuations per seed
│   ├── decoded.md                     ← whale-decoded W_U clusters + attention pairs
│   ├── head_summary.png               ← per-head bar charts (4 metrics)
│   ├── lexical.md                     ← top-30 W_E + W_U neighborhoods, bigram divergence
│   ├── lexical_summary.json
│   ├── logit_lens.png
│   ├── summary.json                   ← all numeric stats
│   └── token_embeddings.png           ← PCA(2) of learned token embeddings
└── childes/
    ├── attention_seed*.html
    ├── continuations.md
    ├── head_summary.png
    ├── lexical.md
    ├── lexical_summary.json
    ├── logit_lens.png
    ├── summary.json
    └── token_embeddings.png
```

## Reproduce

```bash
# 1. Train + persist a single checkpoint per corpus
python -m src.grammar.train_for_interp --source whale
python -m src.grammar.train_for_interp --source childes

# 2. Cross-corpus interp diagnostics (head, logit lens, PCA, attention)
python -m src.grammar.interp_compare

# 3. Lexical-level views (embedding neighborhoods, bigram divergence)
python -m src.grammar.interp_lexical

# 4. Whale-specific decoding of W_U clusters + attention pairs
python -m src.grammar.interp_whale_decoded

# 5. Sampled continuations (T=0.9, top-k=40)
python -m src.grammar.continuations
```

Wall time: ~5 min training per corpus, ~10 min for the rest.

## Take-aways

1. **Cross-corpus depth use is asymmetric.** On English the lemma-level prediction at K=8 is essentially solved by L0; on whale (compound vocab) L1 supplies the majority of the final answer. This is the cleanest *interp-side* evidence that whale tokens carry structure beyond a Markov-1 model.
2. **No emergent circuits at this scale.** K=8 is too short to find induction heads; the head-level diagnostics are essentially uninformative on this axis. Replication of Olsson 2022's results would require K ≥ 32.
3. **W_U has discovered the corpus's two regimes.** Unsupervised on Coda CE alone, the output embedding's mutual-cosine graph splits whale codas into the steady-chorus cluster and the call-and-response cluster.
4. **The readable continuations are the headline qualitative output.** [[interp/continuations]] shows the model committing hard to `1+1+3` (the modal rhythm class) with realistic Δt patterns; CHILDES produces locally fluent transcripts (`*MOT:	oh that be a little spider`) that are syntactically OK and semantically random.

## Related

- [[interp/continuations]] — sampled completions with held-out reference.
- [[transformer/transformer]] — the model the interp scripts read.
- [[transformer/childes-comparison]] — the matched-architecture CHILDES side.
- [[transformer/results]] — bpt tables that motivate the interp questions.
