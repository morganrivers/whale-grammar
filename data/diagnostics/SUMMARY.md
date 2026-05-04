# §0 data-quality diagnostics — findings

Run: 2026-05-04. Per `reproducibility/pacific_classifier_v2_plan.md` §0,
the question was: is some non-trivial fraction of the 28% NOISE residual
a property of the Hersh dataset, not the classifier?

## Top-line answer

**Mostly no.** The data-quality "smoking gun" the plan worried about —
echolocation contamination, recording-quality outliers, cross-species
clicks — is not present at meaningful scale. NOISE is broadly distributed
across recordings and has structured clan-level concentration that fits
algorithm-bound limitations, not data-bound ones.

The one real finding from §0: **the ≤8% NOISE target was a category
error.** Its likely source is Gero 2016's 5.9% DSWP NOISE rate (the
right benchmark for OPTICS on a clean corpus), not anything in Hersh's
pipeline.

## Diagnostic 1 — NOISE rate by recording_id

268 recordings. Median NOISE rate per recording: **28.1%**, IQR 14% – 46%.
Restricting to recordings with ≥30 codas (208 recordings), median 29.7%,
IQR 16.6% – 46.0%.

Top-5 well-sampled recordings by NOISE rate:

| recording_id | n_codas | NOISE rate | source |
|---|---:|---:|---|
| 230 | 38 | 78.9% | hersh2022_pacific |
| 249 | 46 | 78.3% | hersh2022_pacific |
| 153 | 247 | 74.1% | hersh2022_pacific |
| 136 | 75 | 73.3% | hersh2022_pacific |
| CETI23-290 | 263 | 73.0% | sharma2025_birth |

A handful of bad-recording outliers exist but they don't move the
aggregate. **Verdict: not a recording-quality story.**

## Diagnostic 2 — NOISE rate by clan

| clan | NOISE rate | n_codas |
|---|---:|---:|
| EC1 (Sharma-birth, neonatal) | **47.4%** | 5,731 |
| PALI (Palindrome) | **47.3%** | 2,124 |
| FP (Four-Plus) | **45.3%** | 2,590 |
| REG (Regular) | 31.1% | 8,289 |
| (NaN) | 29.6% | 682 |
| RI (Rapid-Increasing) | 28.9% | 1,776 |
| SH (Short) | 25.4% | 5,428 |
| PO (Plus-One) | 22.3% | 2,083 |
| SI (Slow-Increasing) | **7.1%** | 1,265 |

Strong clan structure. SI sits at Gero's 5.9% baseline — already
well-characterised by EC types. PALI/FP/EC1 sit at ~46% — these are
clans whose repertoires are genuinely distinct from DSWP's EC types.
EC1 (Sharma-birth) being noisy is expected (developmental variants;
plan §9 flags this).

**Verdict: NOISE is structured by clan, supports per-clan analysis
(layer 3c) or HDBSCAN (layer 3b) for the dispersed-Pacific clans.**

## Diagnostic 3 — classifier_distance histogram

| subset | n | p10 | p25 | p50 | p75 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ec-knn | 13,161 | 0.030 | 0.051 | 0.081 | 0.110 | 0.143 | 0.181 |
| noise  | 10,119 | 0.112 | 0.126 | 0.149 | 0.212 | 0.341 | 0.527 |

NOISE distribution starts where ec-knn ends — that's the
`RADIUS_FLOOR_S=0.10` threshold by construction. Median NOISE distance is
**0.149 s** (just above the threshold) and 75% of NOISE lies under
0.21 s. There's a substantial recoverable near-miss zone for kNN+τ.
The long tail (p99 = 0.527) is the genuinely-novel residual.

**Verdict: kNN+τ has room to recover ~50% of current NOISE rows.
Distribution is smooth, not strongly bimodal — no obvious quality break.**

## Diagnostic 4 — PCA of NOISE codas (per length)

Per-length PNGs at `data/diagnostics/pca_plots/`. **NOISE is not a
uniform diffuse cloud.** Notable structure:

- **n=5 (5,160 NOISE codas)**: EC1 + REG cluster around origin; FP
  (199 codas) extends a clear arm to the right (PC1 ≥ 2). NOISE has
  clan-specific structure even within the same length bucket.
- **n=6 (1,775 NOISE codas)**: FP dominates with 650 codas separated
  to upper region (PC2 > 0). Distinct repertoire.
- **n=3, n=4**: tighter, more uniform cloud — these are the high-NOISE
  short-coda lengths.

**Verdict: there are real clan-specific Pacific types in NOISE that
DSWP's EC types don't cover. Layer 3c (per-clan classifier) is supported.**

## Diagnostic 5 — echolocation sanity check

NOISE codas with n_clicks ≥ 8 and all-ICIs ≥ 0.4s and roughly regular
spacing (range ≤ 0.5×mean):

| n_clicks | n_noise | n_echo_like | rate |
|---:|---:|---:|---:|
| 8 | 111 | 0 | 0.0% |
| 9 | 47 | 0 | 0.0% |
| 10 | 34 | 1 | 2.9% |

Total: 1 of 192 NOISE codas (0.5%) matches echolocation profile.

**Verdict: echolocation contamination rejected as a meaningful driver
of NOISE.**

## Diagnostic 6 — Hersh supplement on the "5%" benchmark

Read `docs/pnas.2201692119.sapp.pdf` cover-to-cover (Method S1, Method
S2, Tables S1–S8, Discussion S1–S3). Findings:

**Hersh has no per-coda NOISE rate.** IDcall (Method S1) is a Bayesian
mixture-model classifier (mclust 2:15 components × 14 model families,
selected by BIC). Every coda is assigned its most-likely component;
there is no rejection / NOISE bucket in IDcall.

**Hersh's pre-classification filter** (Table S1) excludes:
- Codas outside 3–10 clicks: 24,237 → 23,429 (96.7% kept).
- Codas in repertoires with <25 codas: 23,429 → 22,829 (94.2% of
  original kept). Note this only applies to the *clan-tree clustering*
  stage; the IDcall classification stage uses all 3–10 click codas.

So **5.8% of Hersh codas are excluded before classification**, but they
are excluded *upstream* — they never enter the call-classification
stage and never get a "NOISE" label. The 5.8% is a data-inclusion
threshold, not a classifier residual. Comparing it to our 28% post-
classification NOISE is apples-to-oranges.

**The real source of the "5%" benchmark** is Gero 2016
(`docs/gero2016_methods_extract.md`, line 28): **243 of 4,119 DSWP
codas (5.9%) excluded as noise by OPTICS**. That's the OPTICS-on-clean-
corpus floor. Phase 1 already reproduces this. Our SI clan sits at
7.1% NOISE — the closest any Pacific clan gets to that floor.

**Per-recording quality threshold:** none. Hersh's Methods (Table S5)
note only that recording equipment varies across regions and years,
and that 88.2% of codas were marked by four trained co-authors. No
per-coda SNR / quality flag exists. Mariana Islands (MNP) is the only
region with notably-lower coda extraction rate (59% vs ~96% elsewhere)
— but those rejected codas don't enter the unified CSV and so can't
appear in our NOISE.

## Implications for the v2 plan

1. **The ≤8% NOISE target was unmeetable** with OPTICS-on-Pacific.
   The right benchmarks are:
   - Gero's 5.9% on a clean DSWP-only corpus (already reproduced).
   - SI clan's 7.1% in our current pipeline (Pacific clan whose
     repertoire fits EC types best).
   - For the heterogeneous Hersh corpus, no published benchmark exists.

2. **The plan's ≤15% target is empirically reasonable** but should be
   reported alongside per-clan rates, not just aggregate. SI/PO/SH
   already sit at or below 25%; aggregate NOISE is dragged up by
   PALI/FP/EC1.

3. **Algorithmic work is justified.** No data-quality smoking gun was
   found. Layer-1 (kNN k=5) plus layer-2 (variance-aware τ) is the
   right next step. Layer-3c (per-clan) is supported by both the
   clan-stratified NOISE rates and the n=6 PCA showing FP forms its
   own cluster.

4. **Step A (variance analysis) should still run before Step B.**
   The diagnostic that variance is heterogeneous across CodaTypes is
   not yet confirmed by direct measurement; §4.1 of the plan is the
   right next deliverable.

## Artefacts

- `noise_by_recording.csv` — 268 recording rows, NOISE rate + sample size
- `noise_by_clan.csv` — 9 clan rows
- `distance_distribution.csv` — percentile table
- `noise_pca.csv` — long-form PC1/PC2 per NOISE coda + clan/recording
- `pca_plots/noise_pca_n{3..10}.png` — per-length PCA scatter
- `echolocation_check.csv` — per-length echo-like counts
