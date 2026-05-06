---
tags:
  - overview
  - reproduce
summary: End-to-end commands to rebuild every artifact in the repo from scratch
created: 2026-05-06
updated: 2026-05-06
---

# Reproduce

Step-by-step commands to rebuild every artifact in the repo from scratch. Wall times are CPU-only on a commodity laptop; everything fits without a GPU.

## Prerequisites

- Python 3.11+
- `pip install -r requirements.txt` (numpy, pandas, scikit-learn) plus `torch`, `transformer-lens`, `circuitsvis`, `matplotlib` for the modeling and interp scripts.
- Adoptium Temurin **JRE 8** + ELKI 0.7.1 jar for the OPTICSxi-gated tests. Both gitignored due to size; URLs in [[classifier/locked-parameters]] §"Vendoring URLs".

## 1. Build the classified corpus + transcripts + transformer CSV

```bash
# Full pipeline: A → B → C → E.  --refresh re-fetches whale-ici-data on first run.
python -m src.pipeline.D_run --refresh

# produces:
#   data/upstream/codas_unified.csv         (mirrored from whale-ici-data)
#   data/classified/codas_classified.csv    (canonical artifact)
#   data/classified/rhythm_class_index.csv  (int → label)
#   data/classified/whale_dialogues.csv     (transformer input, 9 cols)
#   data/classified/whale_id_index.csv      (Whale string → int)
#   data/readable/whale_dialogues.txt       (human transcript)
```

Wall time: ~10 min the first run (downloads + ELKI), ~30 s after that (cached labels).

## 2. Verify regression baselines

```bash
# Fast tests (48): ~30 s
python -m pytest tests/ -m "not slow"

# All tests including ELKI-gated (55): ~3 min
python -m pytest tests/
```

What's enforced — see [[classifier/classifier]] for the per-test list.

## 3. Train the next-coda transformer benchmark

```bash
# Smoke test (~1 s) — baselines only on a 50-seq subsample
python -m src.grammar.predict_kfold --quick

# Full 8-model 5-fold benchmark (~15-30 min on CPU)
python -m src.grammar.predict_kfold

# Architecture ablation (~30 min): depth × DT × K=25
python -m src.grammar.predict_kfold --ablation

# Schema redesign (~45 min): R0 vs R1 vs R2 multi-channel multi-task
python -m src.grammar.predict_kfold --redesign --variants R0,R1,R2

# Smoothed-classical baselines (~5 min): MKN 5-gram, KN+cache, PPM-D
python -m src.grammar.predict_smoothed
```

Output files:

```
outputs/grammar/
├── predict_results_unified.{md,json}      (M0–M7)
├── predict_results_ablation.{md,json}     (B0–B4)
├── predict_results_redesign.{md,json}     (R0/R1/R2; latest run overwrites)
├── predict_results_redesign_R0R2.{md,json} (archived R0+R2)
├── predict_results_smoothed.{md,json}     (S1–S3 KN/PPM-D)
├── predict_results_quick.{md,json}        (smoke-test artifact)
└── REPRODUCING_TRANSFORMER.md             (polished recipe)
```

## 4. Train the interp checkpoints

```bash
# Persists checkpoints to outputs/grammar/checkpoints/{whale,childes}/
python -m src.grammar.train_for_interp --source whale
python -m src.grammar.train_for_interp --source childes
```

Each is ~5 min. Produces `model.pt`, `dt_head.pt`, `config.json` (with held-out bpt for both heads), `vocab.json`, `seeds.json`, `train_log.txt`.

## 5. Run interp diagnostics

```bash
# Cross-corpus head + logit-lens + PCA + attention HTMLs
python -m src.grammar.interp_compare

# Lexical-level views (W_E + W_U neighborhoods, bigram divergence)
python -m src.grammar.interp_lexical

# Whale-specific decoding of W_U clusters + attention pairs against per-row metadata
python -m src.grammar.interp_whale_decoded

# Sampled continuations on held-out seeds (T=0.9, top-k=40)
python -m src.grammar.continuations
```

Total wall time: ~10 min after the checkpoints are trained.

Outputs:

```
outputs/grammar/interp/
├── SUMMARY.md                                  (cross-corpus headline)
├── whale/  {attention_seed*.html, continuations.md, decoded.md,
│            head_summary.png, lexical.md, lexical_summary.json,
│            logit_lens.png, summary.json, token_embeddings.png}
└── childes/ {same set, minus decoded.md}
outputs/grammar/continuations.{md,json}        (held-out continuations)
outputs/grammar/childes_vs_whale.{md,json}     (3-fold matched architecture)
```

## 6. Build the CHILDES side (optional)

```bash
# Parse Eng-UK .cha files into the same CSV schema
python -m src.grammar.childes_loader

# (optional) Render the readable CHILDES transcript
python -m src.grammar.render_childes_readable

# 3-fold side-by-side benchmark
python -m src.grammar.predict_kfold_compare
```

Requires the Eng-UK CHILDES corpus at `Eng-UK/`. See [[transformer/childes-comparison]].

## 7. Re-run the validation experiments (Phase 1 / 1b)

```bash
# Phase 1 — DSWP-only OPTICS sweep (~10 min)
python -m src.validation.reproduce_gero21

# Phase 1b option 1 — single OPTICS on unified (~30 min, will fail at ~60%)
python -m src.validation.reproduce_gero21_unified --minpts 10
python -m src.validation.reproduce_gero21_unified --minpts 30

# Phase 1b option 2 — DSWP OPTICS + Pacific kNN (~4 min, passes at 97.6%)
python -m src.validation.phase1b_knn

# Variance analysis (used to derive τ thresholds; ~1 min)
python -m src.validation.variance_analysis

# Per-corpus data-quality diagnostics (NOISE-by-recording, NOISE-by-clan, PCA)
python -m src.validation.data_quality_diagnostics

# Cluster-quality variance check on the 116 discovered clusters
python -m src.validation.cluster_quality
```

These read from cached artifacts where possible; ELKI subprocess calls are cached at `data/cache/optics_*.npz` so re-runs skip recompute when inputs are unchanged.

## 8. Vendor ELKI + JRE 8 (one-time)

Required for any test marked `slow` or any `src.validation.*` script that calls `elki_optics.run`.

```bash
# JRE 8 (Adoptium Temurin)
mkdir -p vendor/jre8 vendor/elki
curl -fsSL \
  "https://api.adoptium.net/v3/binary/latest/8/ga/linux/x64/jre/hotspot/normal/eclipse?project=jdk" \
  -o vendor/jre8.tar.gz
tar -xzf vendor/jre8.tar.gz -C vendor/
ln -sfn $(ls -d vendor/jdk8u*-jre)/* vendor/jre8/

# ELKI 0.7.1 bundle
curl -fsSL \
  "https://repo1.maven.org/maven2/de/lmu/ifi/dbs/elki/elki-bundle/0.7.1/elki-bundle-0.7.1.jar" \
  -o vendor/elki/elki-bundle-0.7.1.jar
```

Sizes: JRE ~40 MB, ELKI jar ~14 MB. Both `.gitignore`d.

## Where things live on disk after a full reproduce

```
whale-grammar/
├── data/
│   ├── cache/                               OPTICS subprocess cache (gitignored, regenerated)
│   ├── classified/
│   │   ├── codas_classified.csv             canonical artifact
│   │   ├── rhythm_class_index.csv           int → coda_type_gero21
│   │   ├── whale_dialogues.csv              transformer input (9 cols)
│   │   ├── whale_id_index.csv               Whale string → int
│   │   ├── childes_dialogues.csv            CHILDES side, same schema
│   │   └── childes_word_index.csv           lemma → int
│   ├── diagnostics/                         NOISE-by-recording, PCA scatters, variance CSV
│   ├── readable/whale_dialogues.txt         human transcript (Δt + rubato)
│   └── upstream/codas_unified.csv           mirrored from whale-ici-data
├── outputs/grammar/
│   ├── checkpoints/{whale,childes,childes_v1605,childes_v467}/
│   │       model.pt, dt_head.pt, config.json, vocab.json, seeds.json, train_log.txt
│   ├── interp/                              SUMMARY.md + per-corpus interp artifacts
│   ├── predict_results_*.{md,json}          benchmark outputs
│   ├── continuations.{md,json}              held-out samples
│   ├── childes_vs_whale.{md,json}           matched-architecture comparison
│   └── REPRODUCING_TRANSFORMER.md           polished recipe (this file is the canonical version)
└── reproducibility/logs/                    frozen run logs (Phase 1, training)
```

## Related

- [[overview/overview]] — pipeline diagram + repo layout.
- [[transformer/transformer]] — model menu the predict scripts populate.
- [[interp/interp]] — what the interp scripts produce.
- [[classifier/locked-parameters]] — every constant + ELKI/JRE URLs.
