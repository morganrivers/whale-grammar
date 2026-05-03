# whale-grammar

A renderer that turns the unified sperm whale (*Physeter macrocephalus*) ICI
corpus into a readable dialogue script. Reads `codas_unified.csv` from the
upstream repo [whale-ici-data](https://github.com/morganrivers/whale-ici-data),
re-segments every ICI sequence against the Sharma et al. 2024 rhythm templates
using a Manhattan-distance tree search, and emits
`data/readable/whale_dialogues.txt` with each coda tokenised as
`<letter><digit>`.

## Upstream data

[whale-ici-data](https://github.com/morganrivers/whale-ici-data) is the
canonical corpus build. whale-grammar downloads four files from it on first
run and mirrors them under `data/upstream/`:

- `codas_unified.csv` — the unified ICI corpus (Sharma 2024 DSWP, Sharma 2025
  birth, Hersh 2022 Pacific)
- `sw_combinatoriality_dialogues.csv` — labelled DSWP subset used to build
  rhythm centroids
- `sw_combinatoriality_rhythms.p` — per-row rhythm class labels (aligned with
  `dialogues.csv`)
- `sw_combinatoriality_ornaments.p` — per-row ornament flags (aligned with
  `dialogues.csv`)

If the network is unreachable on first run, place those files under
`data/upstream/` manually and re-run.

## Reproduce

```bash
pip install -r requirements.txt
python -m src.pipeline.D_run            # full pipeline (uses cached artifacts)
python -m src.pipeline.D_run --refresh  # re-download everything and re-classify
```

## Pipeline

```
whale-ici-data:codas_unified.csv  -->  B_classify  -->  C_render_readable
                                            |                  |
                                            v                  v
                              data/classified/             data/readable/
                              codas_classified.csv         whale_dialogues.txt
```

- **A_load_unified** — fetches the upstream files into `data/upstream/`.
- **B_classify** — runs the Manhattan tree-search segmenter, fills `rhythm`,
  `extra_click`, and `tempo`. ~1 minute on the full corpus. Writes
  `data/classified/codas_classified.csv`.
- **C_render_readable** — reads the classified CSV and writes the readable
  transcript. Fast; iterate on rendering without re-running the classifier.
- **D_run** — orchestrator; runs B if no classified artifact exists, then C.

```
whale-grammar/
├── data/
│   ├── upstream/                       # mirrored from whale-ici-data
│   │   ├── codas_unified.csv
│   │   ├── sw_combinatoriality_dialogues.csv
│   │   ├── sw_combinatoriality_rhythms.p
│   │   └── sw_combinatoriality_ornaments.p
│   ├── classified/
│   │   └── codas_classified.csv
│   └── readable/
│       └── whale_dialogues.txt
├── src/pipeline/
│   ├── A_load_unified.py
│   ├── B_classify.py
│   ├── C_render_readable.py
│   └── D_run.py
├── requirements.txt
├── LICENSE                             # CC BY 4.0
└── README.md
```

## Classification

Sharma et al. 2024 introduces four coda features. Two are context-independent
(rhythm, tempo); two are context-sensitive (ornamentation, rubato).
Project-CETI ships labels but not the classifier code; this pipeline
reconstructs the rhythm classifier and ornament detector following the
whale-gpt port (`scripts/00_create_coda_means.py` +
`scripts/0_extract_codas.py`).

### Templates

The labelled DSWP subset shipped by whale-ici-data
(`sw_combinatoriality_dialogues.csv` + the aligned `*_rhythms.p` /
`*_ornaments.p`) gives one rhythm class id and ornament flag per row.
Ornamented rows are excluded from centroid building. For each class we trim
contributing rows to the class's shortest length, normalise to
cumulative-fraction-of-total, and average — one mean cumulative profile per
class.

### Manhattan tree-search segmentation

For each unified row's nonzero ICI sequence we search for a sequence of
templates that explains the whole input:

1. At each node, take the first up-to-9 ICIs of the remaining sequence.
2. Compute the Manhattan distance from its normalised cumulative profile to
   every template whose length **exactly matches** that window.
3. Keep candidates with distance ≤ `0.1`. For each, consume that many ICIs
   and recurse on the remainder.
4. If no template fits any window size, the segmenter is allowed to consume
   one ICI as a class-100 marker (an "extra click") and continue.
5. The path with the lowest cumulative score wins (class-100 markers carry a
   small `0.05` penalty so they don't dominate).

This **replaces** whatever coda boundaries the source release recorded. A
12-click row that the source called a single coda may segment into a 5+7
pair; a 5-click coda followed by a stray click may segment into a 5-click
coda + class-100 marker.

### Ornamentation (`extra_click`)

Detected as a within-sequence property: a segmented coda is flagged
`extra_click = 1` iff the immediately following segment in the same input
row's path is a class-100 marker. Class-100 markers are dropped from the
output. `extra_click` is `0`/`1` for every successfully-segmented row,
`<NA>` for rows the segmenter couldn't match.

### Tempo

`tempo` is a Sharma 2024 bucket (1..5) computed from `coda_duration_s` using
thresholds 0.45 / 0.61 / 0.93 / 1.08 s. Stored as a column on the classified
CSV so the renderer doesn't have to recompute it.

## Output format

`data/readable/whale_dialogues.txt` is grouped by `(source,
recording_or_date)`. Within each group, codas are emitted in time order if
timing exists, otherwise in source order. Each coda is one token:

- `letter` ∈ `a..r` — the rhythm class (`a` = class 0, `r` = class 17).
  Lowercase = unornamented; uppercase = `extra_click==1`.
- `digit` ∈ `1..5` — the tempo bucket.
- `?` — coda the segmenter could not classify.

Speakers are labelled by `local_speaker_id` if present, else
`whale_photo_id`, else `?`. Pauses longer than 10 s within a timed group
are annotated.

The Hersh 2022 Pacific repertoires lack per-coda timestamps, so each Hersh
recording renders as one large block ordered by `source_coda_id` — the
output file is large; expect ~100k+ lines.

## License

CC BY 4.0 — see `LICENSE`. Cite the original source publications when using
this data; their citations are listed in `LICENSE`.
