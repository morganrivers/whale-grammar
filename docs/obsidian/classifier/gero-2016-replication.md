---
tags:
  - classifier
  - history
  - replication
summary: Phase 1 — reverse-engineering Gero 2016's `minpts` and proving ELKI OPTICSXi reproduces 21 CodaTypes at 96% on DSWP
created: 2026-05-06
updated: 2026-05-06
---

# Phase 1 — Gero-21 replication on DSWP

The classifier's first acceptance test: can we run *the same algorithm* Gero, Whitehead & Rendell 2016 used (ELKI 0.7.x OPTICSXi at `xi=0.04`, Euclidean on absolute ICIs, per-length bucket, 3–10 click range) and recover their 21 published CodaType labels on the labelled DSWP corpus?

The methods text in [Gero 2016 R. Soc. Open Sci.](https://royalsocietypublishing.org/doi/10.1098/rsos.150372) §2.2.4 — extracted at `docs/gero2016_methods_extract.md` — and the supplement (`docs/rsos150372supp1.docx`) state every parameter except `minpts`. We reverse-engineered `minpts` from the data.

## Setup

| input | `data/upstream/dswp_dominica_codas.csv` (8,719 codas with `CodaType` ground-truth, 8,704 in 3–10 click range) |
| algorithm | ELKI 0.7.1 OPTICSXi (Adoptium Temurin JRE 8 + `elki-bundle-0.7.1.jar`) |
| sweep grid | `xi=0.04` × `minpts ∈ {5, 7, 8, 9, 10, 11, 12, 15, 20, 30, 50}` |
| feature | absolute ICIs (n−1 dims for n-click coda), Euclidean |
| bucketing | one OPTICSXi run per `n_clicks` ∈ 3..10 |
| scoring | per (xi, minpts) and per length: majority-vote cluster ID → CodaType, then accuracy |

Code: `src/validation/reproduce_gero21.py`. Frozen log: `reproducibility/logs/phase1_gero21_finer_sweep.log`.

## The minpts result

Aggregate accuracy across 8,704 DSWP codas, holding `minpts` fixed across all length buckets:

| minpts | aggregate accuracy |
|---:|---:|
| 5  | 81.2 % |
| 7  | 80.0 % |
| 8  | 79.6 % |
| 9  | 79.4 % |
| **10** | **95.94 %** |
| 11 | 78.9 % |
| 12 | 78.9 % |
| 15 | 78.5 % |
| 20 | 78.9 % |
| 30 | 91.7 % |
| 50 | 71.7 % |

`minpts=10` is a sharp, isolated peak: every neighbour falls to ~79 %. The peak is real (not smooth tuning) because the n=5 bucket — 6,384 codas, 73 % of the corpus — only resolves correctly at exactly `minpts=10`.

Per-bucket-optimal upper bound: 96.66 %. Fixed `minpts=10` sits 0.7 pp below it — strong evidence that Gero used one global value and that value is 10.

## Why scikit-learn isn't a drop-in for ELKI

`sklearn.cluster.OPTICS`'s `xi` extraction diverges from ELKI's Ankerst-Breunig-Kriegel-Sander formulation in cluster-boundary edge cases. On the n=5 DSWP bucket sklearn rejects 55–83 % of points as inter-cluster noise vs ELKI's ~6 %. Always use the ELKI subprocess wrapper at `src/validation/elki_optics.py`.

The vendor URLs (gitignored due to size) are in [[classifier/locked-parameters]].

## Phase 1b option 1 — same OPTICSxi on the unified corpus (FAILED)

If the algorithm reproduces Gero on DSWP, can it also classify Hersh + Sharma birth alongside in one pass?

**Result at `minpts=10`** (`reproducibility/logs/phase1b_unified_minpts10.log`):

| n | DSWP correct/eval | acc |
|---|---|---:|
| 3 | 83 / 103 | 80.6 % |
| 4 | 457 / 818 | 55.9 % |
| 5 | 3,719 / 6,384 | 58.3 % |
| 6 | 213 / 427 | 49.9 % |
| 7 | 329 / 405 | 81.2 % |
| 8 | 217 / 275 | 78.9 % |
| 9 | 162 / 196 | 82.7 % |
| 10 | 59 / 96 | 61.5 % |
| **total** | **5,239 / 8,704** | **60.19 %** |

276 clusters total; 188 Pacific-only. At `minpts=30` aggregate goes to 60.91 % — n=4 improves but n=6 worsens.

**Failure mode**: at 4× the data density, OPTICS reachability-plot valleys get filled in by Pacific codas that structurally bridge between Gero's CodaTypes — a Pacific 1+1+5 might land between a DSWP 5R2 and a DSWP 1+1+3, merging clusters that were cleanly separated on DSWP-only. This isn't a tuning problem; it's an algorithm-on-this-data problem.

**Conclusion**: ≥90 % bar **FAILED** at every `minpts` tried. Single-OPTICS-pass on the unified corpus is not viable for preserving Gero's labels.

## Phase 1b option 2 — train on DSWP, classify the rest by kNN (PASSED)

Lock cluster boundaries on DSWP-only, propagate labels to non-DSWP codas via per-length nearest-neighbour lookup. Code: `src/validation/phase1b_knn.py`.

**Method** per length n ∈ 3..10:

1. Train per-length OPTICS on DSWP-only at `xi=0.04`, `minpts=10` — sanity check we still get Phase 1's ~96 % (we do).
2. Classify with `KNeighborsClassifier(k=5, metric='euclidean')` trained on `(DSWP ICI vectors, DSWP CodaType labels)`, applied to every coda of length n in the unified corpus.
3. Score DSWP rows leave-one-out (skip the row's own copy in its neighbour set).

**Result** (`reproducibility/logs/phase1b_knn.log`):

| n | DSWP loo accuracy |
|---:|---:|
| 3 | 95.1 % |
| 4 | 96.0 % |
| 5 | 98.2 % |
| 6 | 97.0 % |
| 7 | 95.8 % |
| 8 | 96.0 % |
| 9 | 97.4 % |
| 10 | 96.9 % |
| **total** | **97.64 %** |

≥90 % bar **PASSED**, comfortably. Per-length kNN (k=5, Euclidean) is more robust than OPTICS cluster boundaries here — 97.6 % > 96.0 % — because it doesn't suffer the xi-extraction merge problem.

## Lessons for the production classifier

1. **The 21 CodaType labels reproduce at 96 %.** The published vocabulary is well-defined and recoverable.
2. **`minpts=10` is locked.** Reverse-engineered, paper-confirmed by the sharpness of the peak.
3. **Single-OPTICS-on-unified is dead.** No `minpts` recovers it.
4. **kNN k=5 is the propagation primitive.** Phase 1b option 2's 97.64 % LOO is what the production [[classifier/pacific-extension|hybrid pipeline]] is built around.
5. **Pacific outlier handling needs its own pass** — kNN alone gives every Pacific coda an EC label, including the ~28 % whose nearest DSWP neighbour is itself NOISE. The hybrid pipeline solves this with the τ-floor accept gate + OPTICSxi discovery on the residual; see [[classifier/pacific-extension]].

## Re-run

```bash
# Phase 1 — DSWP sweep (~10 min)
python -m src.validation.reproduce_gero21

# Phase 1b option 1 — single OPTICS on unified (~30 min, will fail)
python -m src.validation.reproduce_gero21_unified --minpts 10
python -m src.validation.reproduce_gero21_unified --minpts 30

# Phase 1b option 2 — DSWP OPTICS + Pacific kNN (~4 min)
python -m src.validation.phase1b_knn
```

ELKI prerequisites in [[classifier/locked-parameters]] §"Vendoring URLs".
