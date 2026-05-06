---
tags:
  - overview
  - pipeline
summary: What whale-grammar does, the data flow end-to-end, and the repo layout
created: 2026-05-06
updated: 2026-05-06
---

# Overview

whale-grammar takes a unified sperm-whale ICI corpus, classifies each coda, renders a human-readable transcript, and trains next-coda sequence models on top.

Upstream corpus: [whale-ici-data](https://github.com/morganrivers/whale-ici-data) — Sharma 2024 DSWP + Sharma 2025 birth + Hersh 2022 Pacific = 38,840 codas. We download `codas_unified.csv`, classify it, and produce both a human transcript and a transformer-ready CSV.

For terminology see [[overview/glossary]]. For per-source feature populations see [[overview/corpora]]. For step-by-step commands see [[overview/reproduce]].

## Pipeline

```
whale-ici-data:codas_unified.csv
        │
        ▼
   A_load_unified  ─►  data/upstream/codas_unified.csv
        │
        ▼
   B_classify (orchestrates B_classify_optics)
        │
        ▼
   data/classified/codas_classified.csv
        │
        ├──► C_render_readable  ─►  data/readable/whale_dialogues.txt
        │
        └──► E_render_csv       ─►  data/classified/whale_dialogues.csv
                                                                │
                                                                ▼
                                                  src.grammar.predict_kfold
                                                                │
                                                                ▼
                                          outputs/grammar/predict_results_unified.md
```

The classifier is a four-stage hybrid (Sharma anchors → kNN+τ matching → OPTICSXi on the residual pool → cluster-name assignment). See [[classifier/classifier]] for the architecture and [[classifier/gero-2016-replication]] / [[classifier/pacific-extension]] for the validation history.

The transformer training script reads the rendered CSV and runs sequence-level 5-fold CV. See [[transformer/transformer]] for the model menu and [[transformer/results]] for the headline numbers.

## Stages

| stage | module | input | output |
|---|---|---|---|
| A. Load | `src/pipeline/A_load_unified.py` | network → `whale-ici-data` | `data/upstream/codas_unified.csv` |
| B. Classify | `src/pipeline/B_classify.py` (orchestrator) + `B_classify_optics.py` (OPTICS+kNN) | upstream CSV | `data/classified/codas_classified.csv`, `rhythm_class_index.csv` |
| C. Render readable | `src/pipeline/C_render_readable.py` | classified CSV | `data/readable/whale_dialogues.txt` |
| D. Run | `src/pipeline/D_run.py` (orchestrator) | — | runs B + C + E |
| E. Render CSV | `src/pipeline/E_render_csv.py` | classified CSV | `data/classified/whale_dialogues.csv`, `whale_id_index.csv` |
| Predict | `src/grammar/predict_kfold.py` | dialogues CSV | `outputs/grammar/predict_results_*.{md,json}` |

End-to-end command: `python -m src.pipeline.D_run --refresh`.

## Repo layout

```
whale-grammar/
├── data/
│   ├── upstream/codas_unified.csv      # mirrored from whale-ici-data
│   ├── classified/                     # canonical artifacts + transformer input
│   ├── readable/                       # human transcripts (whale + CHILDES)
│   └── diagnostics/                    # validation artifacts (variance, LOO τ, …)
├── docs/
│   ├── obsidian/                       # this vault
│   ├── pnas.2201692119.sapp.pdf        # Hersh 2022 supplement
│   ├── rsos150372supp1.docx            # Gero 2016 supplement
│   ├── sharma2024_nat_commun_supplement.pdf
│   └── gero2016_methods_extract.md     # extracted methods text
├── outputs/grammar/
│   ├── checkpoints/                    # trained MiniTransformer-DT (whale + childes)
│   ├── interp/                         # head-level + lexical interp artifacts
│   ├── predict_results_*.{md,json}     # auto-generated benchmark outputs
│   └── continuations.{md,json}         # sampled completions
├── reproducibility/
│   ├── logs/                           # frozen run logs
│   └── scripts/                        # ELKI/OPTICS sweep scripts
├── src/
│   ├── grammar/                        # sequence modeling (transformer + interp)
│   ├── pipeline/                       # A/B/C/D/E stages
│   └── validation/                     # OPTICS sweeps, LOO checks, cluster QA
└── tests/                              # 48 fast + 7 slow ELKI tests
```

## Outputs at a glance

- `data/readable/whale_dialogues.txt` — human transcript, grouped by `(source, recording_id)`. Each coda is `<rubato><letter><digit>` with inline `Δt<seconds>` annotations. Hersh recordings render as one block per recording with no rubato (no timestamps).
- `data/classified/whale_dialogues.csv` — 9-column whale-gpt-style schema (see [[transformer/corpus-design]]).
- `outputs/grammar/predict_results_unified.md` — generation-1 M0–M7 baseline; the canonical bpt reference for the unified corpus.
- `outputs/grammar/interp/SUMMARY.md` — auto-generated cross-corpus interpretability writeup.

## License

CC BY 4.0. Cite the original source publications when using the data — see `LICENSE` for the citation list.
