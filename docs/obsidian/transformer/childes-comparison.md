---
tags:
  - transformer
  - comparison
  - childes
summary: Same MiniTransformer-DT trained on Eng-UK lemma stream — entropy-floor-normalized comparison to whale
created: 2026-05-06
updated: 2026-05-06
---

# CHILDES UK English vs whale

The whale corpus benchmark stands on its own — but the question "is the whale signal compressible *because of structure* or just because the unigram is heavy" wants a control. The natural control is a small, simple, well-understood corpus run through the *same architecture*.

CHILDES Eng-UK (3038 conversational `.cha` files, lemma-tokenized via the `%mor` tier) gives a 39k-token corpus matched in size to the whale corpus — so we can ask "given the same training-token budget and the same architecture, how much of the entropy floor does the model close on each?".

## How CHILDES is loaded

Code: `src/grammar/childes_loader.py`. Reads `Eng-UK/<corpus>/<child>/<file>.cha` files; per utterance prefers the `%mor:` line's lemmas (`going` / `went` / `gone` → `go`; `books` → `book`); surface-form fallback when `%mor` is missing.

DT estimation (CHILDES has no inter-word timing in the data product):

```
dt to next token =
    2.0 s if speaker changed
    1.0 s after `.` `?` `!`
    0.5 s after `,`
    0.3 s otherwise (intra-utterance uniform)
```

Sampling rule: weighted random sampling of `.cha` conversations (weight = lemma TTR) until cumulative token count ≥ 39,000. Chunked into ~80-token sub-sequences breaking at speaker switches, so the per-sequence granularity matches the whale corpus's ~80 tokens/seq.

Output: `data/classified/childes_dialogues.csv` in the same 9-column schema as `whale_dialogues.csv`. Word vocabulary persists at `data/classified/childes_word_index.csv`.

## DT bucket schemes

Both corpora pre-bucket TimeDelta into a small categorical alphabet for the multi-task DT head ([[interp/interp]]). See `src/grammar/dt_buckets.py`.

| corpus | n buckets | schema |
|---|---:|---|
| whale | 6 | 0 = missing, 1..5 = log-quantile bins of valid TimeDelta |
| childes | 4 | 0 = intra (≤0.75 s), 1 = period (0.75–1.5 s), 2 = switch (≥1.5 s), 3 = missing-mirror |

The whale missing bucket holds ~76 % of tokens (every Hersh row). CHILDES missing only fires on rows where the user's `--mirror-hersh-frac` flag deliberately blanks the bucket — the underlying corpus has dt on every row.

## Architecture (locked from best-on-whale)

MiniTransformer-DT — 2L, 4h, d=64, K=8, plus a learned linear projection of `(log(0.1+dt), has_timestamps)`. Trained from scratch on each corpus.

Two variants of the persisted checkpoints in `outputs/grammar/checkpoints/`:

- **whale** — V=467 compound vocab `(rhythm, tempo_bin, ornament, synchrony)`.
- **childes** / **childes_v1605** — V=1605 lemma vocab. (`childes_v467` is the V-matched control: same architecture, same compound-style vocab cap, kept around for the eyes-on apples-to-apples interp comparison.)

Final-position validation numbers come from `train_for_interp.py`; per-fold numbers come from `predict_kfold_compare.py`.

## Headline numbers

3-fold CV held-out bpt from `outputs/grammar/childes_vs_whale.md`:

### whale (V=467, 488 sequences, 38,840 tokens)

| model | params | bpt | perplexity | accuracy |
|---|---:|---:|---:|---:|
| majority (smoothed unigram) | — | 6.570 ± 0.106 | 95.01 | 0.097 |
| Markov-1 | — | 5.549 ± 0.125 | 46.80 | 0.299 |
| MiniTransformer-DT (2L, 4h, d=64, K=8 + DT) | 160,915 | **4.541 ± 0.103** | **23.28** | 0.326 |

### CHILDES UK (V=1602, 405 sequences, 39,318 tokens)

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

**Lower fraction of unigram entropy** = the model captures more structure relative to the entropy floor.

Direct bits/token aren't comparable across V (CHILDES has ~3× larger vocabulary even at the lemma level).

## Read

The headline number is **0.691 vs 0.876** — the model on whale closes 31 % of the gap from unigram to perfect, while on CHILDES it closes only 12 %. Whale at this granularity has *more* learnable structure than English at lemma granularity, given a 39k-token training budget and an identical architecture.

A few caveats this number doesn't address:

- **Markov-1 actually beats unigram on CHILDES (8.08 > 7.83 reversed)** because of how `Markov-1` is implemented — it only fires when the bigram has been seen, otherwise falls back to a smoothed unigram. With 1602 lemmas and ~40k tokens, most bigrams are unseen, and the smoothing burns more entropy than the seen-bigram tail recovers. This is the same sparsity regime that makes KN (in [[transformer/smoothed-baselines]]) helpful.
- **Whale's V=467 compound space is heavy at the unigram level.** The single rhythm class `1+1+3|t1|o0|r0` accounts for ~16 % of all whale tokens, which inflates the bits-saved-over-unigram. The cleanest decomposition would compare against smoothed unigram, but [[transformer/smoothed-baselines]] only has whale numbers — CHILDES doesn't have a KN+cache benchmark yet.
- **CHILDES has no inter-token timing** so the DT head is supervised on synthetic punctuation-derived buckets (intra/period/switch). Useful for rendering readable continuations; not informative about the underlying corpus structure.

## What this comparison was for

Sanity-checking the claim that the whale corpus is "more compressible than English" — the [[interp/interp]] writeup uses the same number to argue that the whale grammar is real, not just an artifact of the model overfitting to a lopsided unigram distribution. The fraction-of-unigram-entropy ratio is robust to V differences in a way absolute bits/token are not.

The interp report goes further: it shows that on whale the *second layer* contributes 57 % of the model's final accuracy, while on CHILDES layer 2 contributes only 17 %. So not only is whale more compressible, the model has to use more of its depth budget to do it. See [[interp/interp]] for the per-layer logit-lens decomposition.

## Reproduce

```bash
# 1. Build the CHILDES CSV
python -m src.grammar.childes_loader

# 2. (optional) Render the readable transcript for inspection
python -m src.grammar.render_childes_readable

# 3. Train the side-by-side benchmark
python -m src.grammar.predict_kfold_compare
# produces:
#   outputs/grammar/childes_vs_whale.md
#   outputs/grammar/childes_vs_whale.json
```

Total wall time ~30 min on CPU.

## Follow-on experiments (this session)

After the V=1605 baseline above, the comparison was tightened in three steps:

- **[[transformer/vocab-cap-experiment|Vocab cap to V=467]]** — match whale's compound vocab cardinality. CHILDES val_coda drops from 5.50 → 4.63 bpt (−0.87) and overfitting goes away. The structural finding (whale saves more bits per token than CHILDES even at matched V) survives.
- **[[transformer/hersh-mirror|Hersh-style missingness mirror]]** — strip speaker + timing from 62% of CHILDES sequences, mimicking whale's Hersh source. Coda CE is unchanged (the token sequence is the same); the DT head sees a 4th "missing" bucket. Substrate for masked-loss training.
- **[[transformer/cross-language-units|Sub-lexical unit comparison]]** — measure English phonemes / Mandarin tonal syllables / Japanese moras at matched token budget. Japanese moras (V≈110) sit closest to whale's natural compound size.

Implementation plan for combining all three: [[transformer/multilang-corpus-plan]].

## Related

- [[interp/interp]] — per-layer logit-lens decomposition that runs on these checkpoints.
- [[transformer/results]] — full bpt tables.
- [[transformer/transformer]] — code map.
- [[overview/whale-clans-pacific-vs-caribbean]] — why "whale V=467" is partly a multi-clan pooling artifact.
