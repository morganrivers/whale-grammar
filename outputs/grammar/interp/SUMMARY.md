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

## How to read

Look first at the per-head table. Heads with `prev` close to 1.0 are previous-token heads (the workhorse of layer 0 in any transformer that fits). Heads with high `ind` on random repeats are the mechanistically interesting ones — they implement the `[A B ... A] → predict B` circuit that lets a 2-layer model do any in-context copying. If both whale and CHILDES models develop induction heads at similar (layer, head) positions, that is a structural parallel that survives the change of data domain. If only the English model develops them, that is evidence the whale corpus does not contain enough repeated-substring structure within K=8 windows for induction to be useful — which is itself an informative null.