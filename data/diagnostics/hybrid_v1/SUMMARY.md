# Hybrid classifier — run summary

Pipeline: Sharma-real anchors → Pacific kNN+τ matching → 
OPTICSxi (xi=0.04, minpts=10) on Sharma-NOISE ∪ Pacific-residual.

## Question 1 — Pacific codas matched to Sharma types

- **22,940 / 28,980 Pacific codas** matched a Sharma 'real' type via kNN k=5 within τ = NOSC p95.
- That is **79.2%** of in-range Pacific codas (n_clicks 3–10 with all ICIs).

## Question 2 — Pacific codas in newly-discovered clusters

- **4,026 / 28,980 Pacific codas** placed into 116 discovered clusters.
- That is **13.9%** of in-range Pacific codas.
- Cluster count by length: n=3: 26, n=4: 24, n=5: 23, n=6: 11, n=7: 17, n=8: 6, n=9: 5, n=10: 4.

## Question 3 — Final NOISE (combined pool)

- **2,341 / 6,625 pool members** stayed NOISE.
- That is **35.3%** of the discovery pool.
- Pool composition: 6,040 Pacific residual, 600 Sharma-NOISE.

## Question 4 — Sharma-NOISE second chance

- Sharma-NOISE rows entering the discovery pass: **600**.
- **258** (43.0%) clustered with Pacific codas — second-chance recoveries.
- **327** (54.5%) stayed NOISE.

## Quick reference

- Pacific in range (3–10 clicks, all ICIs): 28,980
- Pacific not in range / missing ICIs (unprocessed): 988
- Discovered clusters total: 116
- Discovery pool size: 6,625
- Discovery pool clustered: 4,284 (64.7%)
