---
tags:
  - overview
  - glossary
summary: Core terms — rhythm, tempo, rubato, ornament, synchrony, sequence, token
created: 2026-05-06
updated: 2026-05-06
---

# Glossary

Sharma et al. 2024 introduces four sub-coda features: rhythm, tempo, rubato, ornament. Plus three auxiliary concepts that appear in the transformer schema (synchrony, sequence, token shape).

## Rhythm

Context-independent shape class of a coda — what the inter-click-interval pattern looks like, regardless of how fast it's played. Comes from the OPTICSXi clustering on absolute ICIs (see [[classifier/gero-2016-replication]]).

- Gero 2016 published 21 CodaTypes (e.g. `5R1`, `5R2`, `5R3`, `4R`, `4i`, `1+1+3`).
- Sharma 2024 collapsed to 18 rhythm classes by dropping the R1/R2/R3 duration-rank suffix.
- Our pipeline emits an integer `rhythm_class` (and a string `coda_type_gero21`) per coda. The full vocabulary is 131 classes, indexed in `data/classified/rhythm_class_index.csv`.

## Tempo

Context-independent duration bucket. Computed from `coda_duration_s` using the locked Sharma 2024 thresholds:

```
TEMPO_THRESHOLDS = (0.45, 0.61, 0.93, 1.08)  # seconds
# bins: 1 = <0.45s, 2 = [0.45, 0.61), 3 = [0.61, 0.93),
#       4 = [0.93, 1.08), 5 = ≥1.08s
```

Source: Sharma 2024 supp §4 (KDE on durations, h=0.035, peaks at [0.33, 0.51, 0.80, 1.02, 1.26] s).

## Rubato

Context-sensitive duration trend. For each coda, find the previous coda from the **same whale** within 10 s; if both share rhythm and tempo, classify the duration delta:

```
RUBATO_LO = -0.0214 s   # 25th-percentile of delta
RUBATO_HI = +0.0185 s   # 75th-percentile
# delta < LO  → '\\'  (slowing)
# LO ≤ delta < HI  → '-'  (steady)
# delta ≥ HI  → '/'  (speeding)
```

Requires whale identity and timestamps. Hersh has neither, so rubato is structurally absent on Hersh.

## Ornament

A coda has an ornament iff it has exactly one more click than the immediately neighbouring same-whale codas. Structural rule on click counts. Coded as `extra_click ∈ {0, 1}` (or `Ornamentation` in the rendered CSV).

Identifying "neighbouring same-whale codas" requires same-whale-adjacent timestamps. Hersh data has neither click timestamps nor whale-ID per coda, so ornament cannot be applied to Hersh codas (it's set to `0` by convention in the CSV — there's no way to fire the rule). See `feedback_ornament_scope` in auto-memory.

## Synchrony

Computed at render time. `Synchrony = 1` iff a coda from a *different* whale starts within 0.3 s of this row's start, in the same recording. Always 0 on Hersh (no timestamps, no per-coda whale ID).

## Sequence

The transformer reads `(source, recording_id)` groups as one continuous conversation. **No within-recording sub-split** — long quiet pauses appear as large `TimeDelta` values inside a sequence, not as new sequences. Earlier versions of `E_render_csv.py` sub-split on >60 s gaps; that was dropped to give the model honest "this is one conversation" framing. Current count: 488 sequences.

## Token shape (human transcript)

In `data/readable/whale_dialogues.txt`, each coda is one token of the form `<rubato><letter><digit>`:

- `rubato ∈ {/, -, \}` — duration trend vs the previous same-whale, same-rhythm-and-tempo coda within 10 s. Omitted when the comparison isn't defined.
- `letter ∈ {a..r}` — the rhythm class (`a` = class 0, …, `r` = class 17). Lowercase = unornamented; uppercase = `extra_click == 1`.
- `digit ∈ {1..5}` — the tempo bucket.
- `?` — coda upstream could not classify.

Inter-coda time deltas appear as `Δt<seconds>` between codas in a timed group; `Δt?` appears between codas without timestamps. Pauses longer than 10 s within a timed group are annotated.

## CSV columns (transformer input)

The full schema is in [[transformer/corpus-design]]; in brief:

- `sequenceId`, `itemPosition` — sequence identification.
- `Whale` — speaker (`{source}::photo:{id}` ‖ `{source}::local:{id}` ‖ `{source}::UNK`).
- `Coda` — integer rhythm class.
- `Ornamentation`, `Synchrony` — 0/1 flags.
- `Duration`, `TimeDelta` — seconds.
- `has_timestamps` — 0/1, gates time-derived channels.
