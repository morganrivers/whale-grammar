# Reproducing the whale-grammar transformer

Clean handoff for someone re-running the unified-corpus MiniTransformer
benchmark from scratch. The conversational version of this story lives
at `transformer-training.md` (Sharma-only era) and
`reproducibility/whale_grammar_transformer_plan.md` (unified-corpus
Stages 1–4). This file is the polished, end-to-end recipe.

## What this produces

* `data/classified/whale_dialogues.csv` — 38,840-row transformer input
  (whale-gpt-compatible schema + `DeltaTime`).
* `data/classified/rhythm_class_index.csv` — 131-row int → label index.
* `outputs/grammar/predict_results_unified.{md,json}` — held-out
  bits/token across majority / Markov / MLP / Embedding-MLP /
  MiniTransformer baselines on 5-fold sequence-level CV.

## Headline result (2026-05-04 run, V=132 incl. PAD, K=8, ~38k codas)

| # | model | params | bits/token (↓) | perplexity (↓) | accuracy |
|---|-------|-------:|---------------:|---------------:|---------:|
| 0 | majority (smoothed unigram) | — | 4.838 ± 0.246 | 28.59 | 0.254 ± 0.056 |
| 1 | Markov-1 | — | 3.724 ± 0.260 | 13.21 | 0.431 ± 0.039 |
| 2 | Markov-2 | — | 4.110 ± 0.326 | 17.27 | 0.436 ± 0.039 |
| 3 | MLP-S (128, 64) | 151,939 | 3.278 ± 0.252 | 9.70 | 0.465 ± 0.037 |
| 4 | MLP-M (256, 128) | 320,131 | 3.267 ± 0.265 | 9.63 | 0.469 ± 0.040 |
| 5 | MLP-L (512, 256, 128) | 721,795 | 3.331 ± 0.264 | 10.07 | 0.462 ± 0.036 |
| 6 | Embedding-MLP (d=32, 256·128) | 119,940 | 3.274 ± 0.263 | 9.67 | 0.463 ± 0.040 |
| 7 | **MiniTransformer (2L, 4h, d=64)** | **117,508** | **3.186 ± 0.237** | **9.10** | **0.464 ± 0.037** |

The MiniTransformer wins on bits/token at 117 k parameters — half the
size of MLP-M, ~6 × smaller than MLP-L — and 1.65 bpt below the
unigram baseline (perplexity 28.6 → 9.1).

### Comparison to the prior Sharma-only branch

The reference benchmark from
`~/Code/whale-gpt @ claude/whale-language-research-tEudI` used a
rhythm·tempo·orn·rubato compound `Token` with V≈207 on ~4,800 Sharma
DSWP codas (`transformer-training.md` and the prior
`outputs/grammar/predict_models.md`):

| model | prior (Sharma-only V≈207, ~4,800 codas) | unified (V=132, ~38k codas) |
|---|---:|---:|
| majority | 5.99 ± 0.26 | 4.838 ± 0.246 |
| Markov-1 | 5.42 ± 0.37 | 3.724 ± 0.260 |
| Markov-2 | 6.07 ± 0.29 | 4.110 ± 0.326 |
| MLP-S (128, 64) | 5.16 ± 0.51 | 3.278 ± 0.252 |
| MLP-M (256, 128) | 5.60 ± 1.00 | 3.267 ± 0.265 |
| MLP-L (512, 256, 128) | 5.97 ± 1.65 | 3.331 ± 0.264 |
| Embedding-MLP | 4.84 ± 0.41 | 3.274 ± 0.263 |
| **MiniTransformer (2L, 4h, d=64)** | **4.63 ± 0.43** | **3.186 ± 0.237** |

Read carefully: the bpt-headline drop is **not directly comparable**.
The unified corpus uses `Coda1` (rhythm class only) where the prior used
a compound `Token` with ~50 % more vocabulary; `log2(132/207) ≈ −0.66`
accounts for roughly half of the gap, the rest is the corpus being
~8 × larger and the model generalizing better. See
`reproducibility/whale_grammar_transformer_plan.md` §6 for the full
caveat list.

What carries over cleanly:

* **Model ordering preserved.** majority < Markov-1 < learned models;
  Markov-2 > Markov-1 (sparsity penalty); embedding models beat
  one-hot MLPs; MiniTransformer wins.
* **MLP-L's prior instability (σ=1.65) is gone.** With 8 × more codas
  it is stable at σ=0.264 — exactly the prediction the plan made
  ("with 37k codas the same architecture might be under-capacity").
* **MiniTransformer's edge over the Embedding-MLP is small** (≈ 0.09
  bpt, vs ≈ 0.21 bpt before). Same shape: attention buys you something
  modest beyond learned token embeddings at this corpus scale.

## Prerequisites

* Python 3.11+, numpy, pandas, scikit-learn, torch (see
  `requirements.txt`; `torch` is needed for the EmbMLP / MiniTransformer
  models, not the upstream pipeline).
* For ELKI-gated tests: vendored ELKI 0.7.1 + JRE 8 per
  `reproducibility/parameters_locked.md` §"Vendoring URLs".

## Steps

```bash
# 1. (One time) clone whale-grammar and install deps
git clone <whale-grammar-url>
cd whale-grammar
pip install -r requirements.txt pandas numpy scikit-learn torch

# 2. Build the unified classified corpus + readable transcript +
#    transformer CSV. --refresh re-fetches codas_unified.csv from
#    whale-ici-data; omit on subsequent runs to use the cached copy.
python -m src.pipeline.D_run --refresh
# produces:
#   data/classified/codas_classified.csv
#   data/classified/rhythm_class_index.csv
#   data/classified/whale_dialogues.csv
#   data/readable/whale_dialogues.txt

# 3. Verify regression baselines pass
python -m pytest tests/                  # 48 fast + 7 slow
python -m pytest tests/ -m "not slow"    # skip ELKI tests

# 4. (optional) Inspect cluster-quality diagnostic
python -m src.validation.cluster_quality

# 5. Train transformer + baselines
#    Smoke test (~1 second, baselines only on 50-seq subsample):
python -m src.grammar.predict_kfold --quick
#    Full benchmark (8 models × 5 folds; ~15-30 minutes on CPU):
python -m src.grammar.predict_kfold
# produces:
#   outputs/grammar/predict_results_unified.md
#   outputs/grammar/predict_results_unified.json
```

## What the kfold script does

* Sequence-level 5-fold cross-validation. Each `sequenceId`
  (`{source}::{recording_id}::{n}`, sub-split on > 60 s gaps) is in
  exactly one fold.
* Builds (X = last K=8 codas with PAD, y = next coda) windows on each
  training fold; evaluates on the held-out fold.
* Target = `Coda1` (the rhythm-class integer from the hybrid OPTICSxi
  classifier; see `src/pipeline/B_classify_optics.py` and
  `data/classified/rhythm_class_index.csv` for the int → label map).
* Reports held-out cross-entropy in **bits/token** and accuracy across:

  | # | model | description |
  |---|---|---|
  | M0 | majority | smoothed unigram |
  | M1 | Markov-1 | P(y \| last token), Laplace α=0.5 |
  | M2 | Markov-2 | P(y \| last 2 tokens), Laplace α=0.5 |
  | M3 | MLP-S | sklearn (128, 64) on one-hot last-K |
  | M4 | MLP-M | sklearn (256, 128) on one-hot last-K |
  | M5 | MLP-L | sklearn (512, 256, 128) on one-hot last-K |
  | M6 | Embedding-MLP | torch d=32 emb + (256, 128) MLP |
  | M7 | MiniTransformer | torch 2L × 4h × d=64 with positional emb |

## Caveats

* **Vocabulary asymmetry vs the prior research branch.** The prior
  Sharma-only benchmark
  (`~/Code/whale-gpt @ claude/whale-language-research-tEudI`) used a
  rhythm·tempo·orn·rubato compound `Token` with V≈207 on ~4,800 codas.
  This run targets the rhythm-class integer `Coda1` directly on ~38k
  codas with V=131. Lower V → lower entropy floor; absolute bits/token
  are not directly comparable. See
  `reproducibility/whale_grammar_transformer_plan.md` §6 for the full
  caveat list.
* **Hersh has no inter-coda timestamps.** Every Hersh row's
  `DeltaTime = -1`. Stage-1 of the plan deliberately ignores
  `DeltaTime` to reproduce the prior protocol; adding it as a feature
  is a follow-up experiment.
* **CPU is fine.** The MiniTransformer is 126 k params on K=8 windows
  and trains in ≲ 5 min/fold on commodity CPU; no GPU required.

## Going further

* `reproducibility/whale_grammar_transformer_plan.md` — current plan.
* `transformer-training.md` — original Sharma-only research-session
  transcript that produced the MiniTransformer architecture.
