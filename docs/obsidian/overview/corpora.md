---
tags:
  - overview
  - corpus
summary: The three source corpora and what each one populates
created: 2026-05-06
updated: 2026-05-06
---

# Corpora

The unified corpus covers 38,840 codas across three sources. They differ structurally on what data they publish; a model that consumes the rendered CSV needs to understand which channels are NA-by-construction on which rows.

## Sources

| source | rows | rows used (3–10 clicks) | citation |
|---|---:|---:|---|
| `hersh2022_pacific` | 24,237 | ~22,948 | Hersh et al. 2022, *PNAS* |
| `sharma2024_dswp` | 8,872 | 8,704 | Sharma et al. 2024, *Nat. Commun.* |
| `sharma2025_birth` | 5,731 | ~6,032 | Sharma 2025 birth corpus |

Hersh is **62 %** of the corpus. The classifier and the transformer both have to handle Hersh as the dominant case.

## Per-source feature populations

Measured against the rendered 38,840-row CSV:

| feature | hersh2022_pacific | sharma2024_dswp | sharma2025_birth |
|---|---:|---:|---:|
| `has_timestamps = 1` | 0 % | 42 % | 100 % |
| `Whale != ::UNK` | 0 % | 62 % | 100 % |
| `Synchrony == 1` | 0 % | 9 % | 26 % |
| `Ornamentation == 1` | 0 % | 3 % | 11 % |
| real `TimeDelta` (≥ 0) | 0 % | 42 % | 100 % |

## Structural NA semantics

Three things are structurally absent from Hersh:

1. **No per-coda timestamps.** Hersh's data product publishes summary ICIs per coda but not when each coda starts. So `time_in_recording_s` is empty everywhere. → `has_timestamps = 0`, `TimeDelta = -1`, no rubato (rubato needs prior coda by same whale within 10 s), no ornament rule firing.
2. **No whale identity per coda.** Hersh aggregates by clan/repertoire, not by individual. → `Whale = ::UNK` on every Hersh row.
3. **No synchrony information.** Without timestamps we can't detect simultaneous codas. → `Synchrony = 0` on every Hersh row.

DSWP partial timestamps split mostly by recording: most older Atlantic-archive rows lack them; the recordings Sharma 2024 specifically added back have them. Birth corpus has timestamps on every row.

## What each source contributes scientifically

- **DSWP** is the only source with published `CodaType` ground-truth labels (Gero 2016's 21 named types). Used as the anchor set in [[classifier/pacific-extension]].
- **Birth** has full timestamps and individual whale IDs; useful for synchrony, rubato, and ornament rules. Lacks published CodaType labels.
- **Hersh** is the broadest geographically (multiple Pacific clans), but the most structurally NA. It dominates the corpus by row count, so models that consume the CSV need to gate time-derived channels on `has_timestamps`. **Hersh's repertoire breadth is partly an artifact of multi-clan pooling** — see [[overview/whale-clans-pacific-vs-caribbean]] for why Caribbean (V=218 paper-spec compound + rubato) is the right reference rather than the unified V=467.

For the rule definitions see [[overview/glossary]]; for the locked thresholds see [[classifier/locked-parameters]].
