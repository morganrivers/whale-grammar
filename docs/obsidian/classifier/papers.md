---
tags:
  - classifier
  - reference
  - papers
summary: What each primary source — Gero 2016, Sharma 2024, Hersh 2022 — contributed to the classifier
created: 2026-05-06
updated: 2026-05-06
---

# Paper provenance

Where each fact in the classifier pipeline came from. PDFs/docx of the supplements are committed under `docs/`.

## Gero, Whitehead & Rendell 2016 — *R. Soc. Open Sci.* 3, 150372

Main paper, Methods §2.2.2 ("Categorical similarity") and §2.2.4 ("Cluster analysis"). The PMC mirror is the cleanest source (PMC4736920); the methods text is also extracted at `docs/gero2016_methods_extract.md`.

**What we got from it:**

- Algorithm: OPTICSxi (Ankerst, Breunig, Kriegel, Sander 1999) as implemented in ELKI 0.6.1. We use 0.7.1 — the OPTICSXi class is unchanged.
- Steepness threshold ξ = 0.04.
- Distance metric: Euclidean (the main text uses absolute ICIs; the SI has a robustness check with infinity-norm + standardized ICIs).
- Length range: 3–10 click codas; codas with >10 clicks are <5 % and excluded.
- Per-length bucketing: separate OPTICS runs per click count.
- Reported counts: 4,119 codas → 243 noise (5.9 %) → 3,876 in 21 types.

**What we did *not* get:**

- `minPts` is not stated. We had to reverse-engineer it ([[classifier/gero-2016-replication|Phase 1]]). Answer: **`minpts=10`**.
- No machine-readable centroid table for the 21 types — Figure S2 shows a rhythm plot but coordinates aren't in the docx.

## Gero 2016 supplement (`rsos150372supp1.docx`)

Searched exhaustively for `MinPts`, `min_pts`, `epsilon`, `OPTICS`, `cluster size`, etc. Findings:

- §"Justification for the use of OPTICSxi over previously used k-means" confirms the algorithm choice and the conservative-noise philosophy ("ambiguous codas were removed from the categorical analysis as 'noise'"). Does not state `minPts`.
- §"Categorical similarity using OPTICS classification" confirms the basal-similarity 0.001 number used in repertoire comparison (not relevant to clustering itself).
- Figure S2: rhythm plot of all 21 types, sample sizes on right axis — human-readable only, no machine-extractable centroids.

The supplement narrows the parameter set but leaves `minPts` unstated.

## Sharma, Bermant, Beguš, Boumis, Vančura, Pratyusha & Gero 2024 — *Nat. Commun.*

Supplement at `docs/sharma2024_nat_commun_supplement.pdf`. Authoritative source for the 18-rhythm vocabulary, tempo bins, rubato, and ornament definitions.

- **§3 (Rhythm)** — the 18 rhythm classes are reused verbatim from Gero 2016 ("rhythm clusters shown in Fig. 3, reported by Gero et al. 2016"). Confirms the 21→18 collapse is just dropping the R1/R2/R3 duration-rank suffix.
- **§4 (Tempo)** — KDE on coda durations with bandwidth h = 0.035 → 5 peaks at [0.33, 0.51, 0.80, 1.02, 1.26] s, with bin endpoints at [0.45, 0.61, 0.93, 1.08]. These are the thresholds we use in `B_classify._add_tempo`.
- **§5 (Rubato)** — three-level discretization (↑ / = / ↓) of duration delta between adjacent same-rhythm-same-tempo codas by the same whale. Empirical 25th/75th-percentile cutoffs are baked into `B_classify.RUBATO_LO/RUBATO_HI`.
- **§5 (Ornament)** — the last click of a coda that has one more click than the immediately neighbouring codas by the same whale. Structural rule on click counts, **but** identifying "neighbouring codas by the same whale" requires same-whale-adjacent timestamps. Hersh data has neither click timestamps nor whale-ID per coda → ornament cannot be applied to Hersh codas.
- **§6.2** — combinatorial system 18 rhythms × 5 tempos × 2 ornaments × 3 rubatos = 540 possible patterns; 156 actually realized in DSWP.

## Hersh, Gero, Rendell, Cantor, Weilgart, Amano, Dawson, Slooten, Johnson, Kerr, Payne, Rogan, Antunes, Andrews, Ferguson, Hauser-Davis, Mead, Garrigue, & Whitehead 2022 — *PNAS*

Supplement PDF at `docs/pnas.2201692119.sapp.pdf`. Consulted to understand how upstream `coda_type` labels in `data/upstream/hersh_*` were generated, **not** to reproduce Hersh's method.

- **Method S1**: classifier was IDcall + `mclust` mixture models with 2–15 components per click length, BIC-selected from 14 model families (VVV, VVE, VEV, VEE most often best). 196 model/component combinations × 8 length buckets = 1,568 fitted models. **Not OPTICS.**
- **Discussion S2**: clan naming convention (Palindrome, Rapid Increasing, Slow Increasing, Plus-One, Four-Plus, Regular, Short).
- **No per-coda click-level timestamps published** — only summary ICIs by clan/repertoire. This confirms Sharma's ornament rule cannot be applied to Hersh data.
- **No per-coda NOISE rate.** IDcall is Bayesian: every coda is assigned to its most-likely component. The "5.8 %" number in Table S1 is a *data-inclusion threshold* (codas outside 3–10 clicks or in <25-coda repertoires), upstream of classification.

## Schubert et al. — scikit-learn OPTICS

Not directly relevant to Gero's method, but explains why we vendored ELKI instead of using `sklearn.cluster.OPTICS`:

- The xi extraction in scikit-learn diverges from ELKI's Ankerst-Breunig-Kriegel-Sander formulation in cluster-boundary edge cases.
- On our 5-click DSWP bucket sklearn rejects 55–83 % of points as inter-cluster noise vs ELKI's ~6 %.
- The divergence is well-documented in scikit-learn issues and confirmed by spot-checking `src/validation/elki_optics.py`'s output against `sklearn.cluster.OPTICS(xi=0.04, min_samples=10)`.

Lesson: always use the ELKI subprocess wrapper.

## Where to read each PDF

| paper | path |
|---|---|
| Gero 2016 main + supp | `docs/rsos150372supp1.docx`, `docs/gero2016_methods_extract.md` |
| Sharma 2024 supp | `docs/sharma2024_nat_commun_supplement.pdf` |
| Hersh 2022 supp | `docs/pnas.2201692119.sapp.pdf` |

## Related

- [[classifier/locked-parameters]] — every constant + the URL to vendor ELKI/JRE.
- [[classifier/gero-2016-replication]] — the Phase 1 reverse-engineering.
- [[overview/glossary]] — definitions used downstream (rhythm/tempo/rubato/ornament/synchrony).
