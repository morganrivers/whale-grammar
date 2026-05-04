# whale-grammar

A renderer that turns the unified sperm whale (*Physeter macrocephalus*) ICI
corpus into a readable dialogue script. Reads `codas_unified.csv` from the
upstream repo [whale-ici-data](https://github.com/morganrivers/whale-ici-data)
— which already ships rhythm + ornament classifications per coda — adds
`tempo` and `rubato` annotations, and emits `data/readable/whale_dialogues.txt`
with each coda tokenised as `<rubato><letter><digit>` (e.g. `i1 -i1 /B1`,
following the convention from
[sw-combinatoriality](https://github.com/0xideas/sw-combinatoriality) and
[whale-gpt](https://github.com/0xideas/whale-gpt)).

## Upstream data

[whale-ici-data](https://github.com/morganrivers/whale-ici-data) is the
canonical corpus build. whale-grammar downloads `codas_unified.csv` on first
run and mirrors it under `data/upstream/codas_unified.csv`. The unified CSV
covers Sharma 2024 DSWP + Sharma 2025 birth + Hersh 2022 Pacific, with
`rhythm` and `extra_click` already populated by the upstream Manhattan
tree-search segmenter.

If the network is unreachable on first run, place the file under
`data/upstream/` manually and re-run.

## Reproduce

```bash
pip install -r requirements.txt
python -m src.pipeline.D_run            # full pipeline (uses cached artifacts)
python -m src.pipeline.D_run --refresh  # re-download everything and re-classify

# Train next-coda baselines + MiniTransformer on the unified corpus
python -m src.grammar.predict_kfold --quick   # ~1s smoke test
python -m src.grammar.predict_kfold           # ~15-30 min, 8 models × 5 folds
```

See `outputs/grammar/REPRODUCING_TRANSFORMER.md` for the polished
end-to-end transformer recipe and
`reproducibility/whale_grammar_transformer_plan.md` for the active
multi-stage plan.

## Pipeline

```
whale-ici-data:codas_unified.csv
        │
        ▼
   A_load_unified  ─►  data/upstream/codas_unified.csv
        │
        ▼
   B_classify (hybrid OPTICSxi: B_classify_optics)
        │
        ▼
   data/classified/codas_classified.csv
        │
        ├──► C_render_readable  ─►  data/readable/whale_dialogues.txt
        └──► E_render_csv       ─►  data/classified/whale_dialogues.csv  ──┐
                                                                           │
                                                                           ▼
                                                       src.grammar.predict_kfold
                                                                           │
                                                                           ▼
                                                outputs/grammar/predict_results_unified.md
```

- **A_load_unified** — fetches the upstream `codas_unified.csv` into
  `data/upstream/`.
- **B_classify** — adds `tempo` (from `coda_duration_s`) and `rubato`
  (per-whale duration trend across consecutive same-rhythm-and-tempo codas),
  then runs the hybrid OPTICSxi classifier (Sharma anchors → kNN+τ matching →
  OPTICSxi on the residual pool) in `B_classify_optics` to produce a
  per-coda `coda_type_gero21` and integer `rhythm_class`. Writes
  `data/classified/codas_classified.csv` and
  `data/classified/rhythm_class_index.csv`.
- **C_render_readable** — reads the classified CSV and writes the readable
  transcript with `Δt` inter-coda annotations.
- **E_render_csv** — emits `data/classified/whale_dialogues.csv` in
  whale-gpt's CSV schema (sequenceId, itemPosition, Coda1, Ornamentation1,
  Duration1, Coda2, Ornamentation2, Duration2, DeltaTime).
- **D_run** — orchestrator; runs B if the classified artifact is missing,
  then C and E.
- **src.grammar.predict_kfold** — sequence-level 5-fold next-coda
  prediction on `whale_dialogues.csv`. Reports held-out bits/token across
  majority / Markov / MLP / Embedding-MLP / MiniTransformer.

```
whale-grammar/
├── data/
│   ├── upstream/codas_unified.csv      # mirrored from whale-ici-data
│   ├── classified/
│   │   ├── codas_classified.csv
│   │   ├── rhythm_class_index.csv
│   │   └── whale_dialogues.csv         # transformer input
│   └── readable/whale_dialogues.txt
├── outputs/grammar/                    # transformer results + recipe
├── src/
│   ├── pipeline/
│   │   ├── A_load_unified.py
│   │   ├── B_classify.py
│   │   ├── B_classify_optics.py
│   │   ├── C_render_readable.py
│   │   ├── D_run.py
│   │   └── E_render_csv.py
│   ├── validation/
│   └── grammar/
│       └── predict_kfold.py
├── requirements.txt
├── LICENSE                             # CC BY 4.0
└── README.md
```

## Classification

Sharma et al. 2024 introduces four coda features:

- **Rhythm** (`rhythm`, 0..17) — context-independent shape class. Comes
  straight from `whale-ici-data/data/unified/codas_unified.csv`, which runs a
  Manhattan tree-search segmenter against the labelled DSWP centroids.
- **Ornament** (`extra_click`, 0/1) — context-independent extra-click flag.
  Also from upstream, detected by the same segmenter.
- **Tempo** (`tempo`, 1..5) — context-independent duration bucket. Computed
  here in `B_classify` from `coda_duration_s` using Sharma 2024 thresholds
  `0.45 / 0.61 / 0.93 / 1.08` s.
- **Rubato** (`rubato`, `/` `-` `\` or NaN) — context-sensitive duration
  trend. Computed here in `B_classify`: for each coda, find the previous
  coda from the same whale (same `recording_id` and `whale_photo_id` /
  `local_speaker_id`) within 10 s; if both share `rhythm` and `tempo`,
  categorize the duration delta with empirical 25th / 75th-percentile
  cutoffs `-0.0214 / +0.0185` s — `\` shorter, `-` constant, `/` longer.
  Cutoffs and algorithm follow
  `sw-combinatoriality/code/generate_whale_dialogue_txt_with_proper_timings.py`.

The upstream rhythm/ornament classifier (Manhattan tree-search against the
labelled DSWP centroids) lives in whale-ici-data. We don't re-implement it
here.

## Output format

`data/readable/whale_dialogues.txt` is grouped by `(source,
recording_or_date)`. Within each group, codas are emitted in time order if
timing exists, otherwise in source order. Each coda is one token of the
form `<rubato><letter><digit>`:

- `rubato` ∈ `/`, `-`, `\` — duration trend vs the previous same-whale,
  same-rhythm-and-tempo coda within 10 s. Omitted when the comparison isn't
  defined.
- `letter` ∈ `a..r` — the rhythm class (`a` = class 0, `r` = class 17).
  Lowercase = unornamented; uppercase = `extra_click==1`.
- `digit` ∈ `1..5` — the tempo bucket.
- `?` — coda upstream could not classify.

Speakers are labelled by `local_speaker_id` if present, else
`whale_photo_id`, else `?`. Pauses longer than 10 s within a timed group
are annotated.

Hersh 2022 Pacific recordings lack per-coda timestamps, so each Hersh
recording renders as one large block ordered by `source_coda_id`, with no
rubato (rubato needs timing).

## License

CC BY 4.0 — see `LICENSE`. Cite the original source publications when using
this data; their citations are listed in `LICENSE`.
