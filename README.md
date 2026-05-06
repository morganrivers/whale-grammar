# whale-grammar

End-to-end pipeline for the unified sperm-whale (*Physeter macrocephalus*) coda corpus: classify each coda, render a human transcript and a transformer-ready CSV, train sequence models on it, and crack open the trained model with TransformerLens-based interpretability.

The upstream corpus is [whale-ici-data](https://github.com/morganrivers/whale-ici-data) — Sharma 2024 DSWP + Sharma 2025 birth + Hersh 2022 Pacific = 38,840 codas, with `rhythm` and `extra_click` already populated by the upstream Manhattan tree-search segmenter. We download `codas_unified.csv`, classify it through a hybrid OPTICSXi + kNN+τ + Pacific-discovery pipeline, and emit:

- **`data/readable/whale_dialogues.txt`** — human transcript with `<rubato><letters><digit>` tokens and inline `Δt` annotations.
- **`data/classified/whale_dialogues.csv`** — 9-column whale-gpt-style schema for sequence models.
- **`outputs/grammar/predict_results_*.md`** — held-out next-coda benchmarks (8 baselines + ablation + multi-channel/multi-task redesign + smoothed classical).
- **`outputs/grammar/interp/SUMMARY.md`** — cross-corpus mechanistic interp report (whale vs CHILDES Eng-UK on the same architecture).

## Documentation

The detailed scientific and code documentation lives in the **[Obsidian vault](docs/obsidian/Welcome.md)**. It's organised as four self-contained islands:

- **[Overview](docs/obsidian/overview/overview.md)** — pipeline diagram, repo layout, per-source corpora, glossary, end-to-end commands.
- **[Classifier](docs/obsidian/classifier/classifier.md)** — hybrid Sharma-anchor + kNN+τ + OPTICSXi-discovery; reproduces Gero 2016 at 96 % on DSWP.
- **[Transformer](docs/obsidian/transformer/transformer.md)** — next-coda benchmarks, ablation, schema redesign, smoothed classical, CHILDES comparison.
- **[Interpretability](docs/obsidian/interp/interp.md)** — per-layer accuracy decomposition, induction-head check, embedding neighborhoods, sampled continuations.

If you're not in Obsidian, the markdown files render fine in any browser or IDE.

## Quick start

```bash
pip install -r requirements.txt   # numpy, pandas, sklearn — see file
pip install torch transformer-lens circuitsvis matplotlib  # for modeling + interp

# 1. Build the classified corpus + transcripts + transformer CSV (~10 min, cached after)
python -m src.pipeline.D_run --refresh

# 2. Train next-coda baselines + MiniTransformer (~15-30 min CPU)
python -m src.grammar.predict_kfold

# 3. (optional) Train interp checkpoints + run analyses (~15 min)
python -m src.grammar.train_for_interp --source whale
python -m src.grammar.train_for_interp --source childes
python -m src.grammar.interp_compare
```

Full step-by-step recipe: [`docs/obsidian/overview/reproduce.md`](docs/obsidian/overview/reproduce.md). One-file polished version of the transformer recipe: [`outputs/grammar/REPRODUCING_TRANSFORMER.md`](outputs/grammar/REPRODUCING_TRANSFORMER.md).

## Pipeline

```
whale-ici-data:codas_unified.csv
        │
        ▼
   A_load_unified  ─►  data/upstream/codas_unified.csv
        │
        ▼
   B_classify (orchestrator)
   B_classify_optics (Sharma anchors → kNN+τ → OPTICSxi discovery → naming)
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
                                     │
                                     ▼
                         src.grammar.train_for_interp
                         src.grammar.interp_compare
                                     │
                                     ▼
                       outputs/grammar/interp/SUMMARY.md
```

| stage | module | input | output |
|---|---|---|---|
| A. Load | `src/pipeline/A_load_unified.py` | network → `whale-ici-data` | `data/upstream/codas_unified.csv` |
| B. Classify | `src/pipeline/B_classify.py` (orchestrator) + `B_classify_optics.py` (3-stage hybrid) | upstream CSV | `data/classified/codas_classified.csv`, `rhythm_class_index.csv` |
| C. Render readable | `src/pipeline/C_render_readable.py` | classified CSV | `data/readable/whale_dialogues.txt` |
| D. Run | `src/pipeline/D_run.py` (orchestrator) | — | runs B + C + E |
| E. Render CSV | `src/pipeline/E_render_csv.py` | classified CSV | `data/classified/whale_dialogues.csv`, `whale_id_index.csv` |
| Predict | `src/grammar/predict_kfold.py` | dialogues CSV | `outputs/grammar/predict_results_*.md` |
| Interp | `src/grammar/{train_for_interp,interp_compare,interp_lexical,interp_whale_decoded,continuations}.py` | dialogues CSV | `outputs/grammar/interp/`, `outputs/grammar/continuations.md` |

## Headline results

### Classifier

The hybrid pipeline classifies 37,684 / 38,840 in-range codas:

| stage | rows | % of in-range |
|---|---:|---:|
| Sharma 'real' anchor (`dswp-real`) | 8,119 | 91.5 % of DSWP |
| kNN+τ matched to Sharma type (`pacific-matched`) | 22,940 | 79.2 % of in-range Pacific |
| Discovered Pacific cluster (`discovery-cluster`) | 4,026 | 13.9 % of in-range Pacific |
| Final NOISE (`discovery-noise`) | 2,341 | ~6 % aggregate |

DSWP 'real' label conservation: **100 %**. DSWP rhythm-18 vs Sharma truth: **100 %**. ELKI-LOO on DSWP: **97.6 %**. See [`docs/obsidian/classifier/classifier.md`](docs/obsidian/classifier/classifier.md).

### Next-coda prediction (5-fold CV, V=132 incl. PAD, K=8)

| # | model | params | bits/token (↓) | perplexity |
|---|---|---:|---:|---:|
| 0 | majority (smoothed unigram) | — | 4.838 ± 0.246 | 28.59 |
| 1 | Markov-1 | — | 3.724 ± 0.260 | 13.21 |
| 7 | **MiniTransformer (2L, 4h, d=64)** | **117,508** | **3.186 ± 0.237** | **9.10** |
| S2 | Modified KN 5-gram + recency cache | — | 3.324 ± 0.228 | 10.01 |

The MiniTransformer saves 1.65 bpt over the unigram baseline. The recency-cache 5-gram is within 0.14 bpt of the transformer at zero attention. See [`docs/obsidian/transformer/results.md`](docs/obsidian/transformer/results.md).

### Interpretability (whale vs CHILDES UK, same 2L/4h/d=64/K=8 architecture)

| metric | whale | CHILDES |
|---|---:|---:|
| coda vocabulary | 467 (compound) | 1605 (lemma) |
| val coda bpt (final-position) | 5.01 | 8.11 |
| logit-lens accuracy after L0 | 0.066 | 0.073 |
| logit-lens accuracy after L1 | 0.152 | 0.088 |
| L1's share of final accuracy | **57 %** | **17 %** |
| max prev-token head score | 0.29 | 0.27 |
| max induction-head score | 0.17 | 0.16 |

On English the next lemma at K=8 is essentially solved by L0; on whale the model needs a second round of context-mixing. K=8 is too short for induction heads to emerge in either corpus. See [`docs/obsidian/interp/interp.md`](docs/obsidian/interp/interp.md).

## Repo layout

```
whale-grammar/
├── data/
│   ├── upstream/codas_unified.csv      mirrored from whale-ici-data
│   ├── classified/                     canonical artifacts + transformer input
│   ├── readable/                       human transcripts (whale + CHILDES)
│   └── diagnostics/                    validation artifacts (variance, LOO τ, NOISE-by-clan)
├── docs/
│   ├── obsidian/                       Obsidian vault (canonical docs)
│   ├── pnas.2201692119.sapp.pdf        Hersh 2022 supplement
│   ├── rsos150372supp1.docx            Gero 2016 supplement
│   ├── sharma2024_nat_commun_supplement.pdf
│   └── gero2016_methods_extract.md     extracted methods text
├── outputs/grammar/
│   ├── checkpoints/                    trained MiniTransformer-DT (whale + childes)
│   ├── interp/                         head-level + lexical interp artifacts
│   ├── predict_results_*.{md,json}     auto-generated benchmark outputs
│   ├── continuations.{md,json}         sampled completions
│   └── REPRODUCING_TRANSFORMER.md      polished single-file recipe
├── reproducibility/
│   └── logs/                           frozen run logs
├── src/
│   ├── grammar/                        sequence modeling (transformer + interp)
│   ├── pipeline/                       A/B/C/D/E stages
│   └── validation/                     OPTICS sweeps, LOO checks, cluster QA
├── tests/                              48 fast + 7 slow ELKI tests
├── vendor/                             ELKI 0.7.1 + Adoptium JRE 8 (gitignored)
├── requirements.txt
└── LICENSE                             CC BY 4.0
```

## Tests

```bash
python -m pytest tests/ -m "not slow"    # 48 fast tests, ~30 s
python -m pytest tests/                  # +7 slow ELKI-gated tests, ~3 min
```

Enforced invariants: the four-numbers classifier counts, DSWP `CodaType` conservation, clan coherence, per-clan over-classification, cluster quality, transcript rendering, transformer CSV schema, and birth-corpus integration. See [`docs/obsidian/classifier/classifier.md`](docs/obsidian/classifier/classifier.md) §"Validation gates" for the per-test list.

## License

CC BY 4.0 — see `LICENSE`. Cite the original source publications when using this data; their citations are listed in `LICENSE`.
