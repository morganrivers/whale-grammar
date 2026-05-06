---
tags:
  - classifier
  - pacific
summary: Phase 1b — kNN+τ accept, OPTICSxi discovery on the residual, why centroid+radius v1 lost
created: 2026-05-06
updated: 2026-05-06
---

# Pacific extension — kNN+τ + OPTICSxi discovery

[[classifier/gero-2016-replication]] proved kNN k=5 reproduces Gero's labels at 97.6 % LOO on DSWP. Naïvely applying that kNN to every non-DSWP coda is fast and accurate on close cases, but it has two failure modes:

1. **Forced EC labels.** kNN picks one of Gero's 21 EC types for every Pacific coda, even when the coda is structurally far from any of them. Pacific clans (Hersh 2022's seven) have their own characteristic repertoires.
2. **Junk NOISE inheritance.** ~28 % of Pacific codas inherit a `*-NOISE` label from kNN — *not* because they're noise, but because their nearest DSWP neighbour happened to be a row Gero's OPTICS rejected as un-clustered.

The production classifier solves both with two extra stages: a τ-thresholded accept gate and an OPTICSxi discovery pass on the residual.

## Production architecture (recap)

| stage | what | code |
|---|---|---|
| 1 | DSWP 'real' anchor — direct join of `codaNUM2018` ↔ `source_coda_id` | `B_classify_optics.attach_dswp_truth` |
| 2 | kNN+τ matching — accept iff `nearest_neighbour_distance ≤ τ[predicted_codatype]` | `B_classify_optics.match_other_codas`, `compute_tau_lookup` |
| 3 | OPTICSxi discovery on residual pool (Sharma-NOISE ∪ Stage-2 rejects) | `B_classify_optics.discover_pool` |
| 4 | Auto-name discovered clusters (`5RP1`, `5P2`, palindromes by `+` notation) | `cluster_names.name_pacific_clusters` |

Origins emitted: `dswp-real`, `pacific-matched`, `discovery-cluster`, `discovery-noise`.

## τ — the per-type accept threshold

Per `(length, codatype)`,

```
τ = max( NOSC_p99(length, codatype), TAU_FLOOR_S=0.10 s )
```

`NOSC_p99` is the 99th-percentile **nearest-of-same-class** distance among Sharma 'real' rows of that length and CodaType — measuring the *local density* of the type, not its bulk spread. Floor at 0.10 s — about half the median inter-CodaType centroid distance at n=5 — keeps very tight Sharma types from rejecting Pacific neighbours that sit just outside the within-corpus jitter.

Computed inline at every run from the 'real' DSWP rows; no external CSV.

## Stage 3 — OPTICSxi on the residual pool

Pool members: Sharma rows whose `CodaType` ends in `-NOISE` (second-chance recovery) ∪ non-DSWP rows Stage 2 didn't accept.

Per length n ∈ 3..10: ELKI 0.7.1 OPTICSxi at `xi=0.04`, `minpts=10` on the pool's ICI matrix. Output handling:

- **clusters with ≥ minpts members** → `discovery-cluster` (named per Stage 4).
- **ELKI's "rest" bucket (cluster_id 0)** + any sub-`minpts` cluster → `discovery-noise`, labelled `{n}-NOISE`.

OPTICS labels are content-hashed and cached at `data/cache/optics_<tag>_n<N>_mp<M>.npz` so re-runs skip ELKI when inputs haven't changed.

## Stage 4 — naming Pacific clusters

`cluster_names.detect_rhythm` classifies a centroid as one of:

- `R` (regular) — ICIs roughly equal across the coda.
- `D` (decreasing) / `i` (increasing) — monotone within tolerance.
- `+`-pattern — pause-separated, e.g. `1+1+3` (gap detection: ICIs > 3× the minimum and gap_median / non_gap_median ≥ 3).

Then `assign_tempo_ranks` adds rank `1`, `2`, `3` ordered by ascending mean total coda duration *within (n_clicks, rhythm)*. Pacific clusters always get a rank (`P` marker keeps them off EC names):

- Same shape as an existing EC type, different tempo band: `5RP1`, `5RP2`.
- Shape entirely novel (no EC type with this `(n_clicks, rhythm)` combination): `5P1`, `5P2`.
- `+`-pattern names render verbatim (`1+1+5`, `2+2+1`, …) — already disambiguated by their notation.

EC names from Gero 2016 are preserved verbatim — never renumbered.

## Why centroid+radius v1 lost

The first attempt (frozen at `next_phases_implementation_notes.md`, now superseded) replaced kNN with a **centroid + radius_95** classifier:
- group DSWP rows by `CodaType`, one centroid + r95 per type.
- accept a non-DSWP coda iff distance to nearest live centroid ≤ that type's r95.

Result on DSWP itself, leave-one-out (`src/validation/loo_centroid_classifier.py`):

| n | acc | would-be outliers |
|---:|---:|---:|
| 3 | 75.7 % | 21.4 % |
| 4 | 88.0 % | 6.4 % |
| 5 | 87.0 % | 10.5 % |
| 6 | **56.2 %** | 43.3 % |
| 7 | 88.4 % | 10.4 % |
| 8 | 86.9 % | 11.6 % |
| 9 | 86.2 % | 13.3 % |
| 10 | 85.4 % | 13.5 % |
| **total** | **85.5 %** | **12.0 %** |

vs OPTICSxi-direct's 95.9 % and kNN's 97.6 %. Centroid+radius is **materially worse** because some Gero CodaTypes are multi-modal in ICI space — the n=6 bucket has 6i and 6R with overlapping centroids that kNN can disambiguate via local structure but a single per-type centroid cannot.

The shipped pipeline uses kNN (Stage 2) for *label propagation* and uses the per-type τ floor purely for the *outlier accept gate*, not as a label predictor.

## NOISE rate caveat

The original plan target was ≤ 8 % aggregate NOISE. The data-quality investigation (`data/diagnostics/SUMMARY.md`) showed this was a category error: Hersh's reported 5 % is a pre-classification inclusion threshold, not a NOISE rate. Gero's 5.9 % DSWP NOISE rate is the OPTICS-on-clean-corpus floor; anything above it on the unified corpus is "unification cost".

Per-clan NOISE rates after the v1 (centroid+radius) classifier:

| clan | rate | n | comment |
|---|---:|---:|---|
| EC1 (Sharma-birth) | 47.4 % | 5,731 | neonatal codas, expected |
| PALI (Palindrome) | 47.3 % | 2,124 | distinct repertoire |
| FP (Four-Plus) | 45.3 % | 2,590 | distinct repertoire |
| REG (Regular) | 31.1 % | 8,289 | partial overlap with EC types |
| RI (Rapid-Increasing) | 28.9 % | 1,776 | |
| SH (Short) | 25.4 % | 5,428 | |
| PO (Plus-One) | 22.3 % | 2,083 | |
| SI (Slow-Increasing) | 7.1 % | 1,265 | already at Gero's floor |

The current shipped (v2 hybrid) pipeline brings aggregate NOISE down to ~6 % on `B_classify`'s acceptance metric, mostly by recovering Sharma-NOISE rows into discovery clusters and tightening the kNN accept gate. The realistic forward target documented in the plan is ≤ 15 %, not ≤ 8 %.

## Validation that ships with the production code

| test | what it asserts |
|---|---|
| `tests/test_four_numbers.py` | the four counts in [[classifier/classifier]] match within ±10 |
| `tests/test_dswp_conservation.py` | every Sharma 'real' row's `coda_type_gero21` = its published `CodaType` |
| `tests/test_clan_coherence.py` | EC1/REG ≥ 70 % top-5 coverage; Hersh clans within ±5 pp of recorded baseline |
| `tests/test_loo_per_clan.py` | no Sharma clan above 2 % LOO over-classification (current EC1 0.26 %, EC2 0.47 %) |
| `tests/test_cluster_quality.py` | ≥ 80 % of 116 discovered clusters have variance ≤ 2 × nearest Sharma type's variance (current 92.2 %) |

`pytest -m slow` re-runs the full ELKI-gated DSWP LOO and the Phase-1 Gero-21 reproduction.

## Open questions

- **Per-clan classifier (layer 3c).** Hersh 2022's clan column is informative — PCA on n=5/n=6 NOISE codas separates FP from EC types visually (`data/diagnostics/pca_plots/noise_pca_n*.png`). A per-(clan, length) centroid set could lift the long-tail recall, at the cost of trusting Hersh's clan labels (themselves derived from the IDcall + mclust classifier we don't re-implement).
- **HDBSCAN on the residual pool.** OPTICSxi at `minpts=10` over-rejects on the dispersed Pacific outlier subset (the discovery-noise residual). HDBSCAN's varying-density behaviour might recover more clusters; not a settled question.
- **Should length-1, length-2 and length-11+ codas be reintroduced?** Gero excluded them as <5 %. Hersh's distribution might be different. Not on the critical path.

## Related

- [[classifier/classifier]] — production architecture overview.
- [[classifier/locked-parameters]] — every constant.
- [[classifier/papers]] — what Gero, Sharma, Hersh each contributed.
- [[overview/corpora]] — per-source row counts and structural NA semantics.
