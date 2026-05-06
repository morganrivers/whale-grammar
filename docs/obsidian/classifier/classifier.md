---
tags:
  - classifier
  - overview
summary: Hybrid Sharma-anchor + kNN+τ + OPTICSXi-discovery classifier — what it is and where it lives in code
created: 2026-05-06
updated: 2026-05-06
---

# Classifier

The classifier turns each row of `data/upstream/codas_unified.csv` (38,840 codas across the three sources in [[overview/corpora]]) into a `coda_type_gero21` string and a stable integer `rhythm_class`. It runs as Stage B of the [[overview/overview|pipeline]] and is the prerequisite for everything downstream.

## Architecture in one diagram

```
              codas_unified.csv (38,840 codas, 3-10 click range = 37,684)
                                │
                                ▼
            ┌────────────────────────────────────────────────────┐
            │ Stage 1 — DSWP 'real' anchor                       │
            │   join codaNUM2018 ↔ source_coda_id against        │
            │   dswp_dominica_codas.csv; lock published          │
            │   non-NOISE CodaType labels.                       │
            │   origin = dswp-real      (8,119 rows)             │
            └────────────────────────────────────────────────────┘
                                │
                                ▼
            ┌────────────────────────────────────────────────────┐
            │ Stage 2 — kNN+τ matching for non-DSWP codas        │
            │   per length n ∈ 3..10:                            │
            │     train kNN k=5 on Sharma 'real' rows of length n│
            │     query each non-DSWP coda; predict CodaType +   │
            │       nearest-neighbour distance d                 │
            │     accept iff d ≤ τ[predicted CodaType]           │
            │   τ = max(NOSC_p99, TAU_FLOOR_S=0.10s)             │
            │   origin = pacific-matched (22,940 rows)           │
            └────────────────────────────────────────────────────┘
                                │
                                ▼
            ┌────────────────────────────────────────────────────┐
            │ Stage 3 — OPTICSxi discovery on residual pool      │
            │   pool = Sharma-NOISE rows ∪ Stage-2 rejects       │
            │   per length: ELKI 0.7.1 OPTICSxi xi=0.04 minpts=10│
            │   clusters ≥ minpts → discovery-cluster (4,026)    │
            │   ELKI rest-bucket / sub-minpts → discovery-noise  │
            │     (2,341 rows)                                   │
            └────────────────────────────────────────────────────┘
                                │
                                ▼
            ┌────────────────────────────────────────────────────┐
            │ Stage 4 — name discovered clusters                 │
            │   detect_rhythm(centroid)→R/D/i/+pattern           │
            │   tempo-rank within (n,rhythm)                     │
            │   Pacific marker: 5RP1, 5RP2 (shape exists in EC)  │
            │                   5P1, 5P2  (novel shape)          │
            └────────────────────────────────────────────────────┘
                                │
                                ▼
                       coda_type_gero21
                  + classifier_origin
                  + classifier_distance
                  + rhythm_class_18 / rhythm_class
                  + tempo (Sharma bins)
                  + rubato (per-whale duration delta)
                  + extra_click (Sharma structural rule, NA on Hersh)
```

## Code map

| stage | module | function |
|---|---|---|
| Orchestrator | `src/pipeline/B_classify.py` | `classify` |
| Stages 1–4 | `src/pipeline/B_classify_optics.py` | `run_optics_pipeline` |
| Cluster naming | `src/pipeline/cluster_names.py` | `detect_rhythm`, `assign_tempo_ranks`, `name_pacific_clusters` |
| ELKI subprocess | `src/validation/elki_optics.py` | `run` |

## Why three stages instead of one

A single OPTICS pass on the unified corpus collapses DSWP accuracy to ~60%: Pacific codas structurally bridge between Gero's EC types and break his cluster boundaries. See [[classifier/gero-2016-replication]] (Phase 1b option 1 result). The hybrid architecture keeps DSWP's published vocabulary intact, propagates labels to Pacific where they fit, and discovers Pacific-native types on the residual.

## The four numbers (locked, regression-tested)

| # | what | count | % |
|---|---|---:|---:|
| 1 | DSWP rows anchored to Sharma 'real' types | 8,119 | 91.5% of DSWP |
| 2 | Pacific matched to Sharma via kNN+τ | 22,940 | 79.2% of in-range Pacific |
| 3 | Pacific in 116 discovered clusters | 4,026 | 13.9% of in-range Pacific |
| 4 | Final NOISE (pool-residual) | 2,341 | 35.3% of pool |

`tests/test_four_numbers.py` enforces these within ±10 rows on every run.

## Acceptance metrics on each run

`B_classify` prints these gating numbers after every classification (see `_print_acceptance_metrics`):

- **DSWP 'real' conservation** — `coda_type_gero21` of every Sharma 'real' row = its published `CodaType` exactly (target ≥ 99.99%; Sharma-NOISE rows are *excluded* from this check because Stage 3 may legitimately recover them into a discovered cluster).
- **Aggregate NOISE rate** — `*-NOISE` over all classified rows. Target ≤ 8% — currently around 6.2% on the full corpus. The `_print_acceptance_metrics` code still prints this as the historical target; the realistic target after the data-quality investigation is ≤ 15% (see [[classifier/pacific-extension]] §"NOISE rate caveat").
- **rhythm-18 vs DSWP truth** — collapse `coda_type_gero21` to rhythm-18 on the Sharma 'real' rows; should match Sharma's 18-rhythm collapse 100%.

## Output schema (`data/classified/codas_classified.csv`)

| column | description |
|---|---|
| `coda_type_gero21` | Gero name (`5R1`, `1+1+3`), Pacific name (`5RP1`, `5P1`), or `{n}-NOISE` |
| `classifier_origin` | `dswp-real` / `pacific-matched` / `discovery-cluster` / `discovery-noise` (or NA out-of-range) |
| `classifier_distance` | kNN nearest-neighbour distance for `pacific-matched` rows; NaN otherwise |
| `rhythm_class_18` | string, tempo-rank suffix dropped (`5R1` → `5R`, `1+1+3` passes through) |
| `rhythm` | Int64 — stable encoding of `rhythm_class_18` (sorted vocabulary) |
| `rhythm_class` | Int64 — stable encoding of `coda_type_gero21` (full vocabulary, includes tempo ranks and Pacific types). Persisted to `data/classified/rhythm_class_index.csv`. |
| `tempo` | Sharma 2024 bin 1..5 from `coda_duration_s` |
| `rubato` | `/` / `-` / `\` / NA (per-whale duration delta vs prior matched coda within 10 s) |
| `extra_click` | Sharma structural rule on DSWP/birth; NA on Hersh |
| `upstream_rhythm`, `upstream_extra_click` | original columns from `whale-ici-data` (renamed for provenance, not used downstream) |

## Vocabulary sizes

- Full `rhythm_class` vocabulary = 131 (after the v1605 retraining; was 132 in earlier runs that included a `<NA>` slot).
- `rhythm_class_18` collapse = 127 distinct labels (Sharma's 18 EC types + Pacific extensions).
- Sharma 2024's published vocabulary is 18 rhythm × 5 tempo × 2 ornament × 3 rubato = 540 combinations, of which 156 are realized in DSWP. Our pipeline's compound space is bigger because it includes Pacific-discovered rhythm classes; see [[classifier/pacific-extension]].

## Validation gates (in `tests/`)

| test | enforces |
|---|---|
| `test_four_numbers.py` | the four counts above (±10 rows) |
| `test_clan_coherence.py` | EC1/REG clans hit ≥ 70% top-5 label coverage; all Hersh clans within ±5 pp of recorded baseline |
| `test_loo_per_clan.py` | no Sharma clan above 2% LOO over-classification (current EC1 0.26%, EC2 0.47%) |
| `test_cluster_quality.py` | ≥ 80% of 116 discovered clusters have internal variance ≤ 2 × nearest Sharma type's variance (current 92.2%) |
| `test_dswp_conservation.py` | every Sharma 'real' row's `coda_type_gero21` equals its published `CodaType` exactly |
| `test_birth_integration.py` | birth-corpus fields populated: 93.2% match rate, 99.98% tempo, 2,562 rubato, 657 ornaments |
| `test_classify.py` | end-to-end smoke (slow, ELKI-gated) |

Slow ELKI tests (`pytest -m slow`) verify the full pipeline still reproduces ≥97% DSWP LOO, ≤8% NOISE, and the published Gero-21 vocabulary.

## Related

- [[classifier/gero-2016-replication]] — how `minpts=10` was reverse-engineered.
- [[classifier/pacific-extension]] — why kNN+τ replaced the centroid+radius v1.
- [[classifier/locked-parameters]] — every constant + the URL to vendor ELKI/JRE.
- [[classifier/papers]] — what each paper / supplement contributed.
- [[overview/glossary]] — definitions of rhythm, tempo, rubato, ornament.
