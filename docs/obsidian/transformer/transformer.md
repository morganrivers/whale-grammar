---
tags:
  - transformer
  - overview
summary: Roadmap for sequence modeling on the unified corpus — model menu, headline numbers, and where to look next; morpheme tokenization experiments
created: 2026-05-06
updated: 2026-05-07
---

# Transformer

Sequence modeling on `data/classified/whale_dialogues.csv`. Predict the next `Coda` (rhythm-class integer) from the last K codas. Sequence-level K-fold CV. Headline metric: held-out cross-entropy in **bits/token**.

The original research goal — "is whale codas a grammar or just unigrams?" — is settled qualitatively (yes, structure exists; see [[interp/interp]]). The current axis of work is *how much* compression beyond Markov-1, and what kinds of context buy bits.

## Quick start

### Compound-token baselines
```bash
# Smoke test — baselines only on 50-seq subsample (~1 s)
python -m src.grammar.predict_kfold --quick

# Full benchmark — 8 models × 5 folds (~15-30 min on CPU)
python -m src.grammar.predict_kfold

# Architecture ablation — depth × DT × K=25 (~30 min)
python -m src.grammar.predict_kfold --ablation

# Schema redesign — R0 vs R1 vs R2 multi-channel/multi-task (~45 min)
python -m src.grammar.predict_kfold --redesign --variants R0,R1,R2

# Smoothed-classical baselines — KN 5-gram, KN+cache, PPM-D (~5 min)
python -m src.grammar.predict_smoothed

# Full-factorial tiered split — K∈{8,25}, architecture sweep, joint prediction (~8 hours)
python -m src.grammar.predict_full --folds 5 --epochs 60 --k 8 --k 25
```

### Morpheme tokenization (ICI-string and Morfessor)
```bash
# Morpheme discovery and Morfessor segmentation (~2 min)
python -m src.grammar.morpheme_discovery

# M0–M7 benchmark on morpheme sequences (~30 min per mode)
python -m src.grammar.predict_morpheme --tokenisation morpheme_seq
python -m src.grammar.predict_morpheme --tokenisation ici_string

# ICI-fidelity benchmark: morpheme vs compound on full ICI-string prediction (~1 hour)
python -m src.grammar.evaluate_ici_fidelity

# ICI-timing benchmark: raw-seconds timing metrics (~1 hour)
python -m src.grammar.evaluate_ici_timing
```

All read `data/classified/whale_dialogues.csv` directly. The full reproduce recipe lives at `outputs/grammar/REPRODUCING_TRANSFORMER.md` (and is mirrored to [[overview/reproduce]]).

## What's the best model right now

**MiniTransformer (M7) — 2L, 4h, d=64, K=8 — `3.186 ± 0.237` bits/token** on the unified corpus, V=132 incl. PAD, ~38k codas, 488 sequences. 117k parameters; trains in ≲ 5 min/fold on commodity CPU.

That's 1.65 bits below the unigram majority baseline (4.838 bpt), so the held-out coda stream compresses 28.6 → 9.1 in perplexity terms — the primary evidence that K=8 of context carries information beyond unigram statistics.

| family | best variant | bpt | perplexity | source |
|---|---|---:|---:|---|
| baselines | MiniTransformer 2L (M7) | **3.186** | 9.10 | [[transformer/results]] |
| ablation | 4L, 4h, d=64, K=8, +DT (B3) | 3.171 | 9.01 | [[transformer/results]] |
| schema redesign | R0 control (B4 mimic, single-channel + DT) | 3.203 | 9.21 | [[transformer/schema-redesign]] |
| smoothed-classical | Modified KN 5-gram + recency cache | 3.324 | 10.01 | [[transformer/smoothed-baselines]] |

Marginal returns flatten quickly past 117k parameters; the 4L+DT ablation gets ~0.015 bpt back over M7, and the multi-channel multi-task schema (R2) is *worse* by 0.117 bpt. The picture is "K=8 of attention captures nearly all the structure a small transformer can find".

### Morpheme tokenization: ICI-fidelity and timing

The compound-token baseline optimizes for rhythm-class prediction, which leaves ICI-string variation (the actual click-interval pattern) unmodeled. A new axis of work replaces compound tokens with **morpheme sequences** — sub-unit segmentations of ICI patterns discovered via Morfessor — to directly predict ICI structure.

**Results** (Generation 5, [[transformer/results#ICI-fidelity-benchmark—morpheme-vs-compound-tokenisation-2026-05-07|see results]]): On **ICI-string prediction**, morpheme_tfm_k8 wins by **1.9 bpt** over compound (6.817 vs 8.741 ici_bpt). On **raw-seconds timing**, morpheme wins **4 of 5 metrics** (n_click_err, scale_free_rmse, dtw_norm, pattern_corr); compound's only advantage is nn_dist (nearest-neighbor placement) because it decodes exact training codas. 

Key files: `morpheme_discovery.py` (Morfessor + OPTICS discretisation), `whale_morpheme_tokeniser.py` (ICI-string loaders), `predict_morpheme.py` (M0–M7 benchmark), `evaluate_ici_fidelity.py`, `evaluate_ici_timing.py`.

## Code map

### Compound-token baselines
| module | what |
|---|---|
| `src/grammar/predict_kfold.py` | benchmark runner; M0–M7 baselines, B0–B4 ablation, R0–R2 redesign |
| `src/grammar/predict_smoothed.py` | Modified KN 5-gram, KN+cache, PPM-D |
| `src/grammar/predict_kfold_compare.py` | 3-fold CV variant comparing whale vs CHILDES under the same architecture |
| `src/grammar/predict_full.py` | full-factorial benchmark: tiered split (clean tier test), K∈{8,25}, architecture sweep, joint coda+DT prediction |
| `src/grammar/dt_buckets.py` | TimeDelta → bucket id mapping (whale: 6, childes: 4 incl. missing) |
| `src/grammar/whale_compound.py` | compound-token loader: rhythm × tempo × orn × synchrony, V=467 |

### Morpheme tokenization (ICI-string and Morfessor)
| module | what |
|---|---|
| `src/grammar/morpheme_discovery.py` | Morfessor morpheme segmentation of ICI patterns; OPTICS discretisation (N_BINS=4); sensitivity analysis |
| `src/grammar/whale_morpheme_tokeniser.py` | join ICI columns to whale_dialogues.csv; loaders for ICI-string and morpheme-seq vocabularies |
| `src/grammar/predict_morpheme.py` | M0–M7 benchmark on morpheme sequences; per-coda bpt aggregation; ici_string and morpheme_seq modes |
| `src/grammar/evaluate_ici_fidelity.py` | morpheme vs compound on full ICI-string prediction; exact-match, symbol-accuracy, edit-distance metrics |
| `src/grammar/evaluate_ici_timing.py` | morpheme vs compound on raw-seconds timing; n_click_err, nn_dist, scale_free_rmse, dtw_norm, pattern_corr |

### Cross-corpus and interp
| module | what |
|---|---|
| `src/grammar/childes_loader.py` | parse CHILDES Eng-UK `.cha`; supports `--vocab-cap` and `--mirror-hersh-frac` |
| `src/grammar/corpus_stats.py` | per-corpus token / V / H / coverage stats; supports dominica, whale-unified, childes-en/zh/jp; `--bottom-ttr-pct` and `--subsample` |
| `src/grammar/render_childes_readable.py` | render `data/readable/childes_dialogues.txt` for inspection |
| `src/grammar/m7_hooked.py` | TransformerLens port of M7 (used by interp scripts) |
| `src/grammar/train_for_interp.py` | persist a single checkpoint per corpus to `outputs/grammar/checkpoints/`; supports `--tokenisation {compound,ici_string,morpheme_seq}` |
| `src/grammar/continuations.py` | autoregressive sampling from the locked checkpoint |
| `src/grammar/interp_compare.py` | per-head attention diagnostics + logit lens; produces `outputs/grammar/interp/SUMMARY.md` |
| `src/grammar/interp_lexical.py` | embedding neighborhoods, bigram-divergence per token |
| `src/grammar/interp_whale_decoded.py` | join lexical findings against per-row whale metadata |

## Documents in this island

- [[transformer/corpus-design]] — the 9-column CSV schema and what each column means.
- [[transformer/results]] — every bits-per-token table (baselines, ablation, redesign, smoothed) in one place.
- [[transformer/schema-redesign]] — the R0/R1/R2 multi-channel multi-task experiment and why R2 lost.
- [[transformer/smoothed-baselines]] — Modified Kneser-Ney 5-gram, KN+cache, PPM-D.
- [[transformer/childes-comparison]] — same architecture on the Eng-UK lemma stream.
- [[transformer/vocab-cap-experiment]] — cap CHILDES vocab at V=467 to match whale; closes 41% of the val_coda_bpt gap.
- [[transformer/hersh-mirror]] — apply Hersh's 62% missingness pattern to CHILDES; substrate for masked-loss training.
- [[transformer/cross-language-units]] — sub-lexical unit comparison: English phonemes / Mandarin tonal syllables / Japanese moras vs whale Dominica compound.
- [[transformer/multilang-corpus-plan]] — proposed 3-language counterpart corpus matching whale shape (V≈218, 39k tokens, Hersh-style missingness).

## Related

- [[overview/overview]] — pipeline diagram showing how `whale_dialogues.csv` is built.
- [[classifier/classifier]] — what produces the `Coda` integer the transformer reads.
- [[interp/interp]] — what the trained MiniTransformer is doing internally.
