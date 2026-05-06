---
tags:
  - transformer
  - corpus
summary: 9-column whale-gpt-style CSV schema for `data/classified/whale_dialogues.csv` — column-by-column
created: 2026-05-06
updated: 2026-05-06
---

# Corpus design

`data/classified/whale_dialogues.csv` is the input that every transformer model reads. One row per coda, 38,840 rows in 488 sequences. Schema is whale-gpt-compatible plus three extensions for the unified corpus's structural quirks.

Code that emits the file: `src/pipeline/E_render_csv.py`.

## Schema

| column | type | meaning |
|---|---|---|
| `sequenceId` | str | one per `(source, recording_id)` group, formatted `{source}::{recording_id}`. **No within-recording sub-split** — sequence = whole recording. |
| `itemPosition` | int | 0-based position within the sequence, sorted by `time_in_recording_s` when populated, else `source_coda_id`. |
| `Whale` | str | speaker, e.g. `sharma2024_dswp::photo:5563`, `sharma2025_birth::local:42`, `hersh2022_pacific::UNK`. Always namespaced by source so DSWP-speaker-1 and birth-speaker-1 never collide. Mapped to integer ids in `data/classified/whale_id_index.csv` (151 distinct ids incl. UNK). |
| `Coda` | int | rhythm-class integer (0..130) from the hybrid classifier; 98 = silence sentinel. Mapped to label strings in `data/classified/rhythm_class_index.csv`. |
| `Ornamentation` | int (0/1) | `extra_click` from Sharma 2024 §5; NA → 0 (Hersh). |
| `Synchrony` | int (0/1) | 1 iff a coda from a *different* whale starts within `SIMULTANEOUS_THRESHOLD_S` (0.3 s) of this row's start in the same recording. Always 0 on Hersh. |
| `Duration` | float | `coda_duration_s`. |
| `TimeDelta` | float | seconds since the previous primary coda's start in the same sequence. `-1.0` sentinel when timestamps unavailable. |
| `has_timestamps` | int (0/1) | 1 iff this row's source publishes per-coda `time_in_recording_s`. Gates `Synchrony` and `TimeDelta` interpretation. |

## Why "no sub-split"

Earlier `E_render_csv.py` versions sub-split sequences on > 60 s gaps (the whale-gpt convention). That was dropped to give the model honest framing: "this is one conversation". Long quiet pauses surface as large `TimeDelta` values inside a sequence. Current count: **488 sequences across 38,840 codas**.

For sequence-level CV that means each *recording* is in exactly one fold; no within-recording leakage.

## Why a `has_timestamps` flag instead of imputing dt

Hersh has 0 % timestamp coverage by construction (its data product publishes summary ICIs per coda, not start times). DSWP has 42 %, birth has 100 %. Imputing dt for Hersh from `coda_duration_s` would be a structural lie: it would inflate the dt column for the 62 % of the corpus where the value is meaningless.

Instead:

- `TimeDelta = -1` sentinel on Hersh.
- `has_timestamps = 0` separately tells the model the dt value on this row is uninformative.
- The MiniTransformer-DT and the multi-channel R2 models read `has_timestamps` as its own embedding, so they can learn "ignore TimeDelta when this flag is 0".

## Per-source feature populations

Measured on the rendered 38,840-row CSV:

| feature | hersh2022_pacific (24,237) | sharma2024_dswp (8,872) | sharma2025_birth (5,731) |
|---|---:|---:|---:|
| `has_timestamps = 1` | 0 % | 42 % | 100 % |
| `Whale != ::UNK` | 0 % | 62 % | 100 % |
| `Synchrony == 1` | 0 % | 9 % | 26 % |
| `Ornamentation == 1` | 0 % | 3 % | 11 % |
| real `TimeDelta` (≥ 0) | 0 % | 42 % | 100 % |

Hersh dominates (62 % of the corpus). Models that consume the CSV must gate time-derived channels on `has_timestamps`. See [[overview/corpora]] for the structural NA reasoning.

## Whale-id encoding

`_build_whale_index` in `E_render_csv.py`:

- UNK strings get id 0.
- Everyone else is sorted alphabetically and assigned ids 1..N.

Persisted as `data/classified/whale_id_index.csv` so reruns are stable.

## Coda integer encoding

The full vocabulary of `coda_type_gero21` (Gero EC names + `+` patterns + Pacific names + `*-NOISE`) is sorted deterministically and mapped to `rhythm_class` integers 0..130. Persisted as `data/classified/rhythm_class_index.csv` (full int → `coda_type_gero21` map, plus the rhythm-18 collapse). Silence rows emit `Coda = 98` — outside the populated range so it doesn't collide with any rhythm class.

## Compound-token variant (whale-gpt-style)

`src/grammar/whale_compound.py` builds a 4-feature compound token

```
(rhythm, tempo_bin, ornament, synchrony)
```

dense-reindexed to V=467. Used by [[interp/interp]] (the persisted M7 checkpoint at `outputs/grammar/checkpoints/whale/` is trained on this compound vocab). The decoder dict maps each compound id back to a human-readable string like `1+1+5 | t2 | orn0 | rub1`.

Tempo bin is computed once on the full corpus via `pd.qcut(Duration, q=5)` so the compound vocabulary is stable across runs.

## Schema invariants enforced by tests

`tests/test_render_csv.py` asserts:

- Every `Coda` is decodable through `rhythm_class_index.csv`.
- Hersh rows: `TimeDelta = -1`, `Synchrony = 0`, `has_timestamps = 0`, `Ornamentation = 0` (always).
- DSWP rows: `TimeDelta` is real on the populated subset; ornament rule fires.
- Birth rows: `has_timestamps = 1` everywhere.
- Within each `sequenceId`, `itemPosition` is contiguous from 0.
- Schema = exactly the 9 columns listed above (no extras, no missing).

## Related

- [[transformer/transformer]] — the model menu that consumes this CSV.
- [[overview/glossary]] — definitions of rhythm, tempo, rubato, ornament, synchrony, sequence.
- [[overview/corpora]] — per-source structural NA semantics.
- [[classifier/classifier]] — what produces the `Coda` integer.
