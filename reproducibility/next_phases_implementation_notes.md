# Implementation notes — next_phases_plan.md

Frozen 2026-05-04 alongside the implementation. Reads like a delta against
`next_phases_plan.md`: assume the plan as the spec, this file as what
actually happened when the algorithm met the data.

## Files added / modified

| change | path |
|---|---|
| **new** | `src/pipeline/cluster_names.py` (Section B — naming) |
| **new** | `src/pipeline/B_classify_optics.py` (Section A — outlier discovery) |
| **rewrite** | `src/pipeline/B_classify.py` (Section C — orchestrator) |
| **edit** | `src/pipeline/C_render_readable.py` (multi-letter rhythm encoding) |
| **new** | `tests/test_classify.py` + `tests/conftest.py` |

## Deviations from the plan, with reasons

### 1. Stage-1 cluster models grouped by ground-truth `CodaType`, not by ELKI OPTICS sub-clusters

Plan §A step 1 calls for ELKI OPTICSXi at xi=0.04, minpts=10 to give DSWP
cluster IDs, then computing per-cluster centroid + radius_95.

**What we observed running this verbatim**: ELKI splits the n=5 DSWP bucket
(6,384 codas) into 20 sub-clusters, even though Gero published only ~5
distinct types for n=5. Sub-cluster radius_95 is 5–10 ms. Pacific codas
with the same rhythm shape but ~30 ms natural variance routinely exceed
it. **75% of all non-DSWP codas got flagged as outliers** in stage 2;
the second pass then dumped 14,000+ into ELKI's "rest" bucket — final
NOISE rate 39%, far above the ≤8% target the plan tries to fix.

**Fix**: group DSWP rows by their authoritative CodaType label, one
centroid + radius_95 per type. Phase 1 already validated that OPTICSXi
reproduces these labels at 96%, so the partition is the same in spirit;
we measure against the labels Gero published rather than against ELKI's
slightly-finer xi-extraction. Stage 1 no longer calls ELKI at all
(saves ~5 min runtime). Phase 1's 96% reproduction stays a separate,
independently-runnable check via `src.validation.reproduce_gero21`.

### 2. `RADIUS_FLOOR_S = 0.10 s` floor on the radius_95 threshold

Even with stage 1 grouped by CodaType, very tight types (e.g. 5R1 has
internal spread ~5 ms) reject Pacific codas with normal cross-corpus
variance. We floor `radius_95` at 0.10 s — about half the median
inter-CodaType centroid distance in the n=5 bucket. This is a deviation
from the plan's "principled radius_95" formulation; documented in
`B_classify_optics.RADIUS_FLOOR_S`.

After this fix, ec-knn matches climbed from 7,541 → 13,161 (44% of
non-DSWP rows now inherit a Gero EC label).

### 3. Stage-3 noise convention

Plan says "codas left unclustered by this second pass become *-NOISE".
ELKI doesn't *leave* points unclustered — every point gets a leaf
cluster id, with the "rest" bucket conventionally id 0. We confirmed
empirically: cluster_id=0 holds 80%+ of points in the second pass on
n=5 outliers (8,181 of 9,683). We treat cluster_id=0 and any cluster
with `n_members < MINPTS_PACIFIC` as NOISE.

### 4. NOISE-rate target ≤8% — **NOT MET**, ends at 28%

Plan §A acceptance: "Final *-NOISE rate across all corpora must be ≤8%".
Best result with plan-faithful parameters: **28.4%**.

Why: even with stage 2 fixed (ec-knn now claims 44% of non-DSWP rows),
the remaining outliers (~16,000 codas) are too sparse for OPTICSxi at
minpts=10 to extract dense xi-clusters from. They land in the "rest"
bucket. We tested `MINPTS_PACIFIC=5` for the second pass: NOISE rate
fell to 24% but the Pacific vocabulary almost doubled (190 → 369
distinct types, half of them clusters of only 5–9 members). The
trade-off favours the cleaner vocabulary, so we ship `MINPTS_PACIFIC=10`
matching the plan. The constant is exposed for future tuning.

This appears to be a fundamental tension between density-based
clustering (OPTICSxi minpts=10) and the natural sparsity of Pacific
outlier codas — not a bug in the implementation. Resolving it
plausibly requires either (a) a different second-pass algorithm
(e.g. agglomerative), (b) different per-clan minpts, or (c) accepting
that some Pacific codas genuinely don't form a "type" with the rest of
the corpus.

## Acceptance metrics — final state

```
B_classify: --- acceptance metrics ---
  DSWP join exactness:        8,719/8,719 = 100.00%  [PASS]
  Aggregate NOISE rate:       10,719/37,699 = 28.43%  [FAIL, target ≤ 8%]
  rhythm-18 vs DSWP truth:    8,719/8,719 = 100.00%  [PASS, target ≥ 95%]
```

Plus, on Phase 1b option 2's metric (DSWP loo accuracy), nothing in this
work changes: the validated 97.64% loo number is preserved by the
`src.validation.phase1b_knn` script and is testable via the slow tests
in `tests/test_classify.py`.

## Output schema

`data/classified/codas_classified.csv` now carries:

| column | description |
|---|---|
| `coda_type_gero21` | Gero name (`5R1`, `1+1+3`), Pacific name (`5RP1`, `5P1`, `4+1`), or `{n}-NOISE` |
| `classifier_origin` | `dswp-truth` / `ec-knn` / `pacific-discovered` / `noise` |
| `classifier_distance` | distance to nearest live DSWP centroid (non-DSWP rows) |
| `rhythm_class_18` | string, tempo-rank suffix dropped (`5R1` → `5R`) |
| `rhythm` | nullable Int64 — stable encoding of `rhythm_class_18` (sorted vocabulary) |
| `tempo` | Sharma 2024 bin 1..5 |
| `rubato` | `/` / `-` / `\` / NA |
| `extra_click` | Sharma structural rule on DSWP/birth; NA on Hersh |
| `upstream_rhythm` | original rhythm column from `whale-ici-data` (renamed for provenance) |
| `upstream_extra_click` | original extra_click column from `whale-ici-data` |

`C_render_readable.py` emits tokens in the form `<rubato><letters><digit>`
where `<letters>` is base-26 (`a..z` then `aa..zz`) — the rhythm
vocabulary now spans ~160 entries with the Pacific extension, well past
the original `a..r` 18-letter scheme.

## How approaches A and B actually compare on n=5

A side-by-side from `src/validation/loo_centroid_classifier.py` and a one-off
ELKI run on the n=5 DSWP bucket (the largest, 6,384 codas):

| metric | Approach A (plan) | Approach B (mine) |
|---|---|---|
| OPTICS sub-clusters / type groups | 20 sub-clusters | 7 type groups |
| 1+1+3: cluster count | 9 sub-clusters | 1 group |
| 1+1+3: largest cluster's r95 | 0.246 | n/a |
| 1+1+3: smallest cluster's r95 | 0.0096 | n/a |
| 1+1+3: type-group r95 | n/a | 0.206 |
| size-weighted A centroid vs B centroid (max over all real types) | 0.013 s | — |

For non-NOISE types, the centroids agree to within 13 ms — small relative
to ICI scale ~200 ms. The radii are where the two approaches truly differ:
A has 9 tight sub-cluster radii for 1+1+3 (one wide, eight at r95 ~ 0.01);
B has one broad type radius. A Pacific coda landing closer to one of the
small sub-clusters than to the big one would be checked against the small
one's r95 → empirically impossible to match. That's the source of the 75 %
spurious-outlier rate run 1 produced.

For NOISE types, A's three small sub-clusters each have r95 ~ 0.04–0.07
and meaningful centroids; B's single 5-NOISE group has r95 = 0.55 and a
centroid in the middle of nothing useful (heterogeneous-by-design).
*Both* approaches drop NOISE from Pacific anchoring, so the discrepancy
washes out.

## How well B's centroid+radius classifies DSWP under leave-one-out

A direct test of the centroid+radius matcher on labelled DSWP rows
(LOO: each row excluded from its own type's centroid+r95 before being
re-classified). Run via `python -m src.validation.loo_centroid_classifier`:

| n  | accuracy | would-be outliers |
|----|---------|----|
| 3  | 75.7 %  | 21.4 % |
| 4  | 88.0 %  |  6.4 % |
| 5  | 87.0 %  | 10.5 % |
| 6  | 56.2 %  | 43.3 % |
| 7  | 88.4 %  | 10.4 % |
| 8  | 86.9 %  | 11.6 % |
| 9  | 86.2 %  | 13.3 % |
| 10 | 85.4 %  | 13.5 % |
| **total** | **85.5 %** | **12.0 %** |

Compare:
- **OPTICSxi-direct (Phase 1)**: 95.9 %
- **kNN k=5 (Phase 1b option 2)**: 97.6 %
- **centroid+radius (this work)**: 85.5 %

The centroid+radius matcher is **materially worse** than either of the
validated alternatives on DSWP-internal classification. The n=6 bucket is
particularly degraded (56 %) — 6i and 6R have overlapping centroids that
kNN can disambiguate via local structure but a single per-type centroid
cannot. This is informative for the next iteration: kNN is the correct
tool for label propagation; the radius mechanism should be used purely
for outlier flagging, not for label assignment. See
`pacific_classifier_v2_plan.md` for a forward plan addressing this.

## Known limitations / follow-ups

1. **NOISE rate** above target — see §4 above.
2. **Pacific vocabulary size** (160 rhythm classes) is large because
   second-pass OPTICSxi over-fragments. A hierarchical merge step on
   centroid-distance similarity could collapse these into a smaller
   "Pacific dialect" set.
3. **Hersh-clan coherence test** (plan §A acceptance bullet 3) is not
   automated. The data to run it (Hersh 2022's clan labels) is in
   `data/upstream/codas_unified.csv` under the `clan` column; a future
   test could plot predicted CodaType distribution per clan and check
   that each clan concentrates on a small handful of types.
4. **Ornament rule** is implemented as the simplest interpretation of
   Sharma 2024 §5 ("immediate same-whale neighbour with `n_clicks - 1`
   within 10 s"). Cross-checks against the original
   `sw-combinatoriality` reference implementation are not in the test
   plan.
