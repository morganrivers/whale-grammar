# Gero, Whitehead & Rendell 2016 — Methods extract

Source: PMC4736920 (open-access mirror of *R. Soc. Open Sci.* 3:150372, doi: 10.1098/rsos.150372). Extracted 2026-05-03 via WebFetch on `https://pmc.ncbi.nlm.nih.gov/articles/PMC4736920/`. The publisher-hosted HTML and PDF (royalsocietypublishing.org) are bot-blocked (HTTP 403); the supplement `rsos150372supp1.docx` was not retrieved.

## OPTICS algorithm parameters

- **ξ (xi)**: 0.04 for all coda lengths — "defines a 4% drop in point density as the criterion for defining a new cluster"
- **Distance metric**: Euclidean on absolute inter-click intervals (ICIs) in multivariate space
- **Software**: OPTICSxi module in the ELKI framework
- **MinPts / min_samples**: NOT stated in main text; presumed in supplement
- **Outlier handling**: codas labeled "noise" rather than forced into clusters ("highly conservative" classification)

## Per-length clustering

Clustering performed on "codas of similar click length" — strongly implies separate OPTICS runs per click-count bucket. Click-count range not explicitly stated in extract; results note that "4- and 5-click codas made up 82% of all codas".

## Naming convention

- `5R` = five clicks regularly spaced
- `1+1+3` = "click-[PAUSE]-click-[PAUSE]-click-click-click" (longer gaps separate the groups)
- `D` = decreasing ICIs across the coda
- `I` = increasing ICIs across the coda
- Trailing digit (e.g. `5R1`, `5R2`, `5R3`) = sequential numbering by mean total duration ascending — "coda types with the same rhythm but of increasing duration"

## Counts

- 4119 codas analyzed
- 243 (5.9%) excluded as noise by OPTICS
- Remaining 3876 categorized into **21 types**

## Open questions (need supplement)

1. Exact MinPts value used per length bucket
2. Per-length type counts (how 21 types distribute across click-counts)
3. Whether the "rhythm cluster" level (the 18 used by Sharma 2024) is reported in Gero or whether it's a Sharma-side collapse
4. Centroid ICI table for each type
