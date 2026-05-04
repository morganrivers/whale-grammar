# Transformer corpus design

Single source of truth for `data/classified/whale_dialogues.csv` —
the transformer-ready CSV produced by `src/pipeline/E_render_csv.py`
and consumed by `src/grammar/predict_kfold.py`. Documents what each
column means, where its values come from upstream, the per-source NA
semantics, the rhythm-class vocabulary lineage, and the explicit
deltas vs the prior whale-gpt training pipeline. Read alongside
`reproducibility/whale_grammar_transformer_plan.md` (the active plan)
and `reproducibility/parameters_locked.md` (the locked classifier
parameters).

## 1. Column-by-column intent

| column | type | meaning | source |
|---|---|---|---|
| `sequenceId` | string | one per `(source, recording_id)` group; the model treats each as one continuous conversation | format `{source}::{recording_id}` |
| `itemPosition` | int | 0-based position within the sequence; sorted by `time_in_recording_s` when populated, else by `source_coda_id` | running counter |
| `Whale` | string | speaker identifier, source-namespaced so DSWP-speaker-1 and birth-speaker-1 never collide; `{source}::photo:{id}` ‖ `{source}::local:{id}` ‖ `{source}::UNK` | `whale_photo_id` ‖ `local_speaker_id` ‖ literal `UNK` |
| `Coda` | int | OPTICS-rhythm class id for the primary whale; `98` is the silence sentinel (used only when `rhythm_class` is NA) | `rhythm_class` from `B_classify` |
| `Ornamentation` | int 0/1 | extra-click ornament flag; 0 when click-level data unavailable | `extra_click` from upstream classifier |
| `Synchrony` | int 0/1 | 1 iff a coda from a *different* whale starts within 0.3 s of this row's start, in the same recording | computed at render time |
| `Duration` | float | coda duration in seconds | `coda_duration_s` |
| `TimeDelta` | float | seconds since the previous primary coda's start in the same sequence; `-1` sentinel when timestamps unavailable | computed from `time_in_recording_s` |
| `has_timestamps` | int 0/1 | 1 iff this row's source publishes per-coda `time_in_recording_s` | `time_in_recording_s` populated → 1 |

The integer encoding for `Coda` is in
`data/classified/rhythm_class_index.csv`; for `Whale` it is in
`data/classified/whale_id_index.csv` (both written by
`E_render_csv.py`).

## 2. Per-source feature populations

What is actually populated per source, measured against the rendered
38,840-row corpus:

| feature | hersh2022_pacific (24,237 rows) | sharma2024_dswp (8,872 rows) | sharma2025_birth (5,731 rows) |
|---|---:|---:|---:|
| `has_timestamps = 1` | 0 % | 42 % | 100 % |
| `Whale != ::UNK` | 0 % | 62 % | 100 % |
| `Synchrony == 1` | 0 % | 9 % | 26 % |
| `Ornamentation == 1` | 0 % | 3 % | 11 % |
| real `TimeDelta` (≥ 0) | 0 % | 42 % | 100 % |

Hersh contributes 62 % of the corpus and is structurally NA on every
time-derived or speaker-derived column. Models must either gate those
channels on `has_timestamps` (the new redesign does this) or ignore
them entirely (what the legacy single-channel `Coda1` benchmarks did).

## 3. Sequence boundary

A sequence is one whole `(source, recording_id)` group. **No
within-recording sub-split** — long quiet pauses appear as large
`TimeDelta` values inside a sequence, not as new sequences. This
matches whale-gpt's "one sequence per recording" convention. Earlier
versions of `E_render_csv.py` sub-split on >60 s gaps; that was
dropped to give the model honest "this is one conversation" framing
and to avoid generating many short-prefix sub-sequences that depend
heavily on PAD.

`whale_dialogues.csv` currently contains 488 sequences (vs 507 with
the prior 60 s sub-split).

## 4. Rhythm-class vocabulary lineage

The `Coda` column is the OPTICS-discovered rhythm class id. Where the
131 values come from:

* **Gero, Whitehead & Rendell 2016** ran ELKI 0.7.1 OPTICSXi
  (`xi=0.04`, `minpts=10`, Euclidean on absolute ICIs, per-length
  bucket 3–10 clicks) on 4,119 DSWP codas and published 21 named
  CodaTypes (5R1, 5R2, 5R3, 4R, 4i, 1+1+3, …). Reproducibility:
  `parameters_locked.md`, validated to 95.94 % on 8,704 DSWP codas
  in `reproducibility/logs/phase1_gero21_finer_sweep.log`.
* **Sharma, Bermant, Beguš et al. 2024** reused those 21 CodaTypes
  verbatim and **collapsed to 18 rhythm classes** by dropping the
  R1/R2/R3 duration-rank suffix (Sharma 2024 supplement §3:
  "rhythm clusters shown in Fig. 3, reported by Gero et al. 2016").
* **Our pipeline** ships a four-stage hybrid classifier
  (`src/pipeline/B_classify_optics.py`):
  1. **Stage 1 — `dswp-real`**: directly join 8,119 DSWP rows to
     Sharma's published `CodaType` via `codaNUM2018 ↔
     source_coda_id`. We do **not** recluster DSWP from scratch —
     we use the gold-standard label. 100 % conservation is enforced
     by `tests/test_dswp_conservation.py`.
  2. **Stage 2 — `pacific-matched`**: per-length
     `KNeighborsClassifier(k=5, Euclidean)` propagates Gero's labels
     to non-DSWP codas. Validated at 97.64 % LOO on DSWP in Phase 1b
     (`reproducibility/logs/phase1b_knn.log`).
  3. **Stage 3 — `discovery-cluster`**: OPTICSXi runs *only* on the
     residual pool (Sharma-NOISE rows + Pacific rows the kNN
     couldn't place). 116 new Pacific-native clusters discovered;
     2,341 rows fall through to a final NOISE bucket (6.21 %
     aggregate).
  4. **Stage 4 — `cluster_names`**: assigns names like P5RP1, 5P2,
     etc. to discovered Pacific clusters per
     `src/pipeline/cluster_names.py`.

Vocabulary count breakdown:

| view | count | what's collapsed |
|---|---:|---|
| Sharma's published 18 classes | 18 | rhythm pattern names only |
| Gero's published 21 CodaTypes | 21 | + R1/R2/R3 duration-rank suffix |
| `rhythm_class_18` (this repo) | 127 | Sharma's 18 *across length buckets* + 116 discovered Pacific clusters; same per-rhythm-name across lengths |
| `rhythm_class` (this repo, the `Coda` column) | 131 | + R1/R2/R3 distinctions on EC types preserved (4 extra entries vs `rhythm_class_18`) |
| prior `whale-gpt @ tEudI` compound `Token` | ~207 | rhythm × tempo × ornament × rubato fused into one token |

The `131 vs 207` gap is **not** "more discovery"; it's a different
axis. The prior compound `Token` exploded one rhythm-name into many
combinatorial sub-tokens; we keep tempo, rubato, and ornament as
separate columns instead.

## 5. Sentinel and missing-data conventions

* `Coda` uses `98` as a *silence* sentinel (only emitted when
  `rhythm_class` is NA upstream).
* `TimeDelta` uses `-1.0` as a *missing-timestamp* sentinel. The
  encoder wraps this in `_encode_dt_log`: `log(0.1)` for missing rows,
  `log(0.1 + dt)` otherwise.
* `Whale` uses the literal string suffix `::UNK` for any speaker we
  can't identify upstream.
* `has_timestamps` is the *row-level* gate — when 0, both `Synchrony`
  and `TimeDelta` are uninformative. The model uses this single bit
  to learn "ignore time-derived channels here" rather than having to
  rediscover the Hersh-only correlation `Synchrony=0 ∧ TimeDelta=-1`
  by itself.

## 6. Comparison vs whale-gpt's `train-dialogue-script.yaml`

This is the explicit delta table — what we kept, what we changed,
what we dropped. Reference for whale-gpt's choices is
`~/Code/whale-gpt/configs/train-dialogue-script.yaml` and
`~/Code/whale-gpt/scripts/1b_create_dialogue_script.py`.

| dimension | whale-gpt | whale-grammar (this repo) | rationale |
|---|---|---|---|
| corpus | Sharma DSWP only (~4,800 codas, V≈18 / V≈207 compound) | unified Sharma + Gero + Hersh (~38 k codas, V=131 rhythm-only) | breadth + cross-corpus generalization |
| sequence | one per recording | one per recording (we removed the prior 60 s sub-split) | matches whale-gpt; honest single-conversation framing |
| Whale | `Whale` 1..11 | `Whale` source-namespaced string + UNK sentinel for Hersh | preserves speaker info where it exists, marks where it doesn't |
| Coda | `Coda` 1..18 | `Coda` 0..130 (OPTICS rhythm class) | richer vocabulary (Pacific clusters); see §4 lineage |
| Ornamentation | derived from "next row Coda==100" trick | per-coda `extra_click` boolean from upstream classifier | functionally equivalent on DSWP; structurally NA on Hersh |
| co-occurrence | scalar `Synchrony` 0/1 | scalar `Synchrony` 0/1 | matches whale-gpt by user decision (the prior `Coda2`/`Orn2`/`Dur2` simultaneous-coda track is dropped) |
| Duration | raw seconds, min-max normalized via ddconfig | raw seconds, normalized by the model's projection | avoids coupling data prep to a precomputed config |
| TimeDelta | `log(0.1 + dt)`, min-max normalized via ddconfig | raw seconds with `-1` sentinel, encoded as `log(0.1 + dt)` at model-input time + `has_timestamps` flag | log shape matches whale-gpt; `has_timestamps` handles Hersh's structural absence |
| has_timestamps | absent | row-level flag column | gates Synchrony + TimeDelta interpretation explicitly |
| input projection | per-column embeddings concat to `d_model=32` (Whale=6, Coda=10, Orn=4, Sync=4, Dur=4, TD=4) | per-column embeddings concat to `d_model=32` (Whale=6, Coda=8, Orn=3, Sync=3, Dur=4, TD=4, has_ts=4) | same recipe; budget reallocated for the extra `has_timestamps` channel |
| targets | multi-task: CE on Whale/Coda/Orn/Sync, L1 on Dur/TimeDelta | multi-task: CE on Whale/Coda/Orn/Sync, L1 on Dur/TimeDelta(log) | matches whale-gpt; headline metric is the Coda head's bpt |
| optimizer | AdamP, lr=1e-4, CosineAnnealingLR, dropout 0.2 | `FAST_MODE=True` (default): AdamW, lr=1e-3, no scheduler, dropout 0.1, ES patience 10. `FAST_MODE=False`: AdamW + CosineAnnealingLR + dropout 0.2 + bs=1000 + 5000 epochs, no ES | benchmarking pipeline default; flip to slow for a final shipped model |
| eval | single 80/10/10 split | 5-fold sequence-level KFold | error bars; with σ ≈ 0.24 bpt across folds we genuinely need them |
| windowing | seq_length=25, stride=1, no PAD | K=25 (redesign) / K=8 (legacy benchmarks), stride=1, explicit PAD token | PAD costs ~nothing and lets the model train on every position even in shorter sequences |

## 7. The `FAST_MODE` toggle

`predict_kfold.py` exposes a single boolean (set near the top of the
file, also flippable per-run with `--slow`) that swaps the training
recipe between two regimes:

| | FAST_MODE=True (default) | FAST_MODE=False |
|---|---|---|
| optimizer | AdamW | AdamW |
| schedule | constant lr=1e-3 | CosineAnnealingLR(T_max=epochs, eta_min=1e-5) starting at 1e-4 |
| batch size | 128 | 1000 |
| epochs | 80 (early-stopped) | 5000 (no ES) |
| dropout | 0.1 | 0.2 |
| wall-time per fold | ~3–5 min on the redesign model | ~30× longer |

Use `True` for: cross-validation, benchmarking, ablation runs.
Use `False` for: a single shipped/published model run, or when a CV
result is within fold variance and you want to confirm with thorough
training.

## 8. Files

| path | role |
|---|---|
| `src/pipeline/E_render_csv.py` | renders `whale_dialogues.csv` + `whale_id_index.csv` from `codas_classified.csv` |
| `data/classified/whale_dialogues.csv` | the 9-column transformer-ready CSV |
| `data/classified/whale_id_index.csv` | `Whale` string → `whale_id` int |
| `data/classified/rhythm_class_index.csv` | `Coda` int → cluster name |
| `tests/test_render_csv.py` | per-source schema + invariant tests |
| `src/grammar/predict_kfold.py` | benchmark runner (M0–M7, B0–B4, R0/R2 redesign) |
| `outputs/grammar/predict_results_*.{md,json}` | run artifacts |

## 9. Reproducing the corpus

```bash
# Full pipeline (~10 min on first run; cached afterwards):
python -m src.pipeline.D_run --refresh

# Just the renderer (assumes B_classify ran):
python -m src.pipeline.E_render_csv

# Schema + invariants:
pytest tests/test_render_csv.py -v
```

Outputs written:
* `data/classified/whale_dialogues.csv`
* `data/classified/whale_id_index.csv`
* `data/classified/rhythm_class_index.csv`

ELKI prerequisites (`vendor/jre8/`, `vendor/elki/elki-bundle-0.7.1.jar`)
are gitignored; download URLs are in
`reproducibility/parameters_locked.md`.
