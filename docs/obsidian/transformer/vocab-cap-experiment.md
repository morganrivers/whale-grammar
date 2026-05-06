---
tags:
  - transformer
  - childes
  - experiment
summary: V=1605 → V=467 vocab cap on CHILDES — closes 41% of the val_coda_bpt gap to whale and removes overfitting
created: 2026-05-06
updated: 2026-05-06
---

# Vocab cap experiment — V=467 on CHILDES

The first big confound in the [[transformer/childes-comparison|whale ↔ CHILDES side-by-side]] was vocabulary size. Whale's compound vocab is V=467; CHILDES Eng-UK lemmas come in at V=1605 — about 3× bigger. With ~39k tokens, that's ~24 tokens/type for CHILDES vs ~83 for whale, meaning English long-tail embeddings get very few gradient updates each.

This experiment caps the CHILDES vocab at V=467 by keeping the top 466 most frequent lemmas plus a single `<unk>` sentinel. Same architecture (`m7_hooked.py`, 2L/4h/d=64/K=8), same training recipe.

## Loader change

`src/grammar/childes_loader.py` gained a `--vocab-cap N` flag.

```bash
python -m src.grammar.childes_loader --vocab-cap 467
```

When set, builds vocab from frequency-ranked lemmas; keeps top `N-1`; reserves the last id (`N-1`) for `<unk>`. All other lemmas get the `<unk>` id. Output CSV layout is unchanged — only the `Coda` integer space is smaller.

## Result of the cap on the CHILDES corpus

| | uncapped (V=1605) | **capped (V=467)** |
|---|---:|---:|
| total tokens | 39,318 | 39,318 |
| dropped lemmas | 0 | 1,139 long-tail |
| `<unk>` token frequency | n/a | 7.9% of tokens |
| unigram entropy | 7.69 bpt | **6.92 bpt** |
| log₂(V) ceiling | 10.65 | 8.87 |

7.9% UNK rate is small enough that the loss isn't drowned by sentinel predictions. The unigram floor drops by 0.77 bpt, which is the price the long-tail lemmas were paying.

## Result of training

Trained `train_for_interp.py --source childes` on each:

| | uncapped (V=1605) | **capped (V=467)** | Δ |
|---|---:|---:|---:|
| early-stop epoch | 8 (overfit) | **12** (plateau) | +4 |
| best val_coda (full window) | 5.497 (ep 3) | **4.627** (ep 7) | **−0.87 bpt** |
| train-val gap at stop | 0.82 | 0.29 | shrunk 65% |
| val final-position bpt | 8.110 | **7.079** | −1.03 |
| val final-position acc | 0.0751 | 0.0779 | +0.003 |

Two things moved together:

- **Less overfitting.** Uncapped, val_coda hits its minimum at epoch 3 and gets monotonically worse; capped, val_coda plateaus around epoch 7 and stays there. The gibberish-y continuations the user was seeing were from the epoch-3 checkpoint, before the model had actually learned much.
- **Lower bpt.** −0.87 bits in val_coda is real compression. The compression-vs-whale gap closes from 5.50−3.36 = 2.14 bpt → 4.63−3.36 = **1.27 bpt** (41% of the gap closed).

## What the cap *didn't* change

Bits-saved-vs-unigram is roughly the same:

- Uncapped: unigram H = 7.83 bpt; model = 5.50 bpt; saved ≈ 2.33 bits.
- Capped: unigram H = 6.92 bpt; model = 4.63 bpt; saved ≈ 2.29 bits.

So the model wasn't suddenly extracting more structure — the V=1605 absolute number was just inflated by the long-tail entropy floor. **The real CHILDES vs whale finding survives the cap**: whale model saves ~3.21 bits vs unigram, CHILDES capped saves ~2.29 — whale Caribbean compound is genuinely more compressible per token than English at lemma granularity, even at matched V.

## Why this matters

Before the cap, raw bpt comparisons were bouncing off the entropy-floor mismatch. After the cap, the comparison is structurally honest: same V, same N, same architecture, different corpus.

The cap also enabled the [[transformer/hersh-mirror|Hersh-mirror]] follow-up — the V=467 capped CSV is the substrate for adding artificial missingness, since at lemma V=1605 the noise from rare-token UNK predictions would have swamped the missingness signal.

## Reproduce

```bash
# 1. Backup uncapped artifacts (optional)
cp data/classified/childes_dialogues.csv data/classified/childes_dialogues_v1605.csv
cp data/classified/childes_word_index.csv data/classified/childes_word_index_v1605.csv
cp -r outputs/grammar/checkpoints/childes outputs/grammar/checkpoints/childes_v1605

# 2. Regenerate at V=467
python -m src.grammar.childes_loader --vocab-cap 467

# 3. Re-train CHILDES interp model
python -u -m src.grammar.train_for_interp --source childes \
    2>&1 | tee reproducibility/logs/train_childes_v467.log
```

Wall time: regen <30 s, training ~10 min on CPU.

## Related

- [[transformer/childes-comparison]] — the original V=1605 baseline.
- [[transformer/hersh-mirror]] — next step: apply Hersh-style missingness to the capped corpus.
- [[transformer/cross-language-units]] — apples-to-apples sub-lexical comparison once V is matched.
