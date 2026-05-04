# Pacific classifier v2 — forward plan

Frozen 2026-05-04. Reader: assume **no memory** of the conversations that
produced Phase 1, Phase 1b, `next_phases_plan.md`, or
`next_phases_implementation_notes.md`. Read those files in this order
before starting:

1. `reproducibility/README.md` — what Phase 1 / 1b proved
2. `reproducibility/parameters_locked.md` — the locked parameter set
3. `reproducibility/papers_relevant.md` — what each paper contributes
4. `reproducibility/next_phases_plan.md` — the plan that produced the
   current code
5. `reproducibility/next_phases_implementation_notes.md` — what was
   actually built and why it deviated, with measured numbers
6. **this file** — what to do next
7. `src/pipeline/B_classify.py` and `B_classify_optics.py` — current code

This document tells you everything you need to push toward a Pacific
classifier whose aggregate `*-NOISE` rate is comfortably below 30 %
(stretch target: 15 %). Spend the time reading the files above first;
they contain numbers that this document will quote without re-deriving.

---

## 0. Read this before assuming the algorithm is the problem

There is a real possibility — flagged but not investigated by the
person who built v1 — that some non-trivial fraction of the 28 %
"NOISE" residual is **a property of the Hersh dataset itself, not of
our classifier**. Concretely the candidate explanations are:

- **Echolocation clicks misidentified as codas.** Sperm whales
  echolocate constantly; an automated coda detector that fires on
  click trains can pick up echolocation bursts that happen to look
  like short codas. These would have ICI patterns alien to any social
  coda type and would land in NOISE under any classifier.
- **Recording-quality issues.** Hersh 2022 aggregates data from many
  groups across decades and ocean basins. Microphone bandwidth, depth,
  hull noise, distance-to-source, and SNR vary wildly. Low-SNR
  detections produce ICI estimates with ±20–50 ms jitter that wouldn't
  show up in the cleaner DSWP / Sharma 2024 corpora.
- **Non-sperm-whale clicks.** Other odontocetes (pilot whales, beaked
  whales) co-occur with sperm whales in some recording locations and
  also click. A detector tuned for sperm-whale codas might still trip
  on the wrong species occasionally.
- **Genuine clan-specific types our matcher can't reach.** This is
  what the rest of the v2 plan addresses, but it should be
  *separated* from the data-quality residual before you tune anything.

**Hersh 2022 reports a noise/rejection rate of around 5 %.** Match
that against our 28 % and ask: did Hersh apply a quality-filter we
didn't carry forward in `whale-ici-data`'s `codas_unified.csv`? If yes,
they got 5 % because their filter had already discarded the bad data
that hits us downstream. The plan's ≤ 8 % target may have been
modelled on that 5 % number without realising filtering was upstream.

### What you need to investigate, and where to look

#### Resources that ARE in the repo

| resource | path | what it can tell you |
|---|---|---|
| Hersh 2022 supplement (PDF) | `docs/pnas.2201692119.sapp.pdf` | Method S1 describes their classifier; *somewhere* in the supplement should describe their pre-processing / quality threshold. Search for "noise", "threshold", "quality", "exclude", "SNR", "filter". |
| Sharma 2024 supplement (PDF) | `docs/sharma2024_nat_commun_supplement.pdf` | DSWP-side preprocessing for comparison. Look for whether they apply any per-coda quality criterion. |
| Gero 2016 supplement (.docx) | `docs/rsos150372supp1.docx` | Reports 5.9 % noise on DSWP. The "≤ 5 %" framing in the plan may have come from here, not Hersh. Verify which paper actually motivates that number. |
| Methods extract | `docs/gero2016_methods_extract.md` | Already-extracted Gero methods; check it before re-reading the docx. |
| Unified corpus | `data/upstream/codas_unified.csv` | Look at whether *any* per-coda quality column exists (SNR, detection confidence, filter flag). Likely no — see below. |
| `whale-ici-data` upstream | not in this repo | This is the GitHub repo (`morganrivers/whale-ici-data`) that produces `codas_unified.csv`. Read its README + the script that builds the unified CSV from Hersh's published data, to see whether any filtering happened during unification. |
| Phase-1b kNN log | `reproducibility/logs/phase1b_knn.log` | Pacific nearest-neighbour distance percentiles. The 99th percentile is informative for "how far out is the long tail?" — if the 99th is ≥ 0.5 s, you're seeing detections that *can't* be valid sperm-whale codas at any reasonable noise floor. |

#### Resources that you likely DO NOT have, and probably need

| missing resource | why you'd want it | how to get it |
|---|---|---|
| Hersh raw audio recordings | Spot-check the codas our pipeline labels NOISE: are they real sperm-whale codas at all, or echolocation / cross-species / static? | The Hersh 2022 supplement should cite the Movebank or similar archive. Some recordings may be at the William Gilly lab (Stanford) or with Project CETI. May not be public. |
| Per-coda SNR or detection confidence | Tag NOISE codas with quality scores; if NOISE rate correlates strongly with low SNR, you've found the residual cause. | Probably never published. Could be requested from Pratima Hersh / the Dalhousie WHOI groups. |
| Recording-level metadata (mic, depth, location, year) | Stratify NOISE rate by recording — if one recording or one expedition dominates the NOISE, that's a smoking gun. | Some of this is in the unified CSV's `recording_id`, `location`, `latitude`, `longitude`, `date` columns. Check what's actually populated for Hersh rows. |
| Clan-specific human-validated codas | Independent label set to cross-check Hersh's clan column. | Project CETI may publish or share these; otherwise it's a manual exercise. |

#### Concrete diagnostic experiments

Run these *before* tuning the classifier:

1. **Stratify NOISE rate by `recording_id`.** Group Hersh rows by
   recording; compute NOISE rate per recording. If it's roughly
   uniform (~28 %), it's a global issue (algorithm, or systematic data
   limitation). If a few recordings push the rate up disproportionately,
   those are quality outliers — exclude them and re-run. Code path:
   join `data/classified/codas_classified.csv` against `recording_id`,
   group, count noise / count total.

2. **Stratify NOISE rate by `clan`.** Some Hersh clans may have noisier
   data than others. If the "Plus-One" clan is at 5 % NOISE and the
   "Short" clan is at 60 %, that's structural and tells you where the
   problem lives. Same join + group as above on `clan`.

3. **Distance-from-DSWP histogram for NOISE codas.** Plot the
   `classifier_distance` distribution for the 10,119 NOISE rows. If
   it's bimodal (a tight near-miss peak + a long tail), the long tail
   is data-quality and the near-miss is what variance-aware kNN can
   recover. If it's a smooth long tail, it's all genuine.

4. **ICI-vector PCA of NOISE codas.** Project all NOISE Pacific codas
   to 2D PCA. If they cluster meaningfully, those clusters are
   undiscovered Pacific types. If they're a uniform diffuse cloud,
   they're noise in the literal sense.

5. **Echolocation sanity check.** Sperm-whale echolocation clicks have
   ICIs around 0.5–2.0 s with low jitter and very regular spacing.
   Check whether NOISE Pacific codas have a disproportionate share of
   `n_clicks` ≥ 8 with monotonically-spaced ICIs >= 0.4 s. That
   profile suggests echolocation, not coda.

6. **Find Hersh's own NOISE / rejection rate.** Open
   `docs/pnas.2201692119.sapp.pdf`. Search for the 5 % number. Note
   the *exact* sentence it appears in: was it Hersh's reported NOISE
   rate, or a model BIC parameter, or something else entirely? Does
   their figure 1 / table 1 give a confusion-style breakdown of how
   many codas were excluded prior to clustering? If yes, capture the
   methodology in writing. If no, the 5 % framing in this v2 plan and
   in v1's ≤ 8 % target is unsourced and should be dropped.

7. **Cross-check Gero's 5.9 % NOISE rate** in
   `docs/rsos150372supp1.docx`. That number is more solidly
   established (Phase 1 reproduces it on DSWP). The 5.9 % floor is
   what's achievable on a *clean* corpus — anything above it on the
   unified corpus is the unification cost.

#### What the answer changes

- **If diagnostics 1–5 mostly come back uniform / smooth** (i.e. the
  NOISE residual is structural, not concentrated in a few bad
  recordings or a few clans): proceed with the layer-1/2/3 plan
  below; the algorithm is the bottleneck.
- **If a recording, clan, or `n_clicks` slice produces most of the
  NOISE**: build a quality-filter step in `A_load_unified.py` that
  excludes those rows *before* classification, document it as a
  data-quality decision (not an algorithm decision), and report
  NOISE rates pre- and post-filter.
- **If diagnostic 6 reveals Hersh applied a published filter we
  didn't replicate**: implement that filter in `A_load_unified.py`.
  Adopt their criteria literally.

This whole §0 should take 0.5–1 day. It's the cheapest way to find out
whether you're tuning a knob that doesn't exist.

---

## 1. Empirical state at handoff

After running `python -m src.pipeline.B_classify` end-to-end on the
unified corpus (~10 min, cached afterwards):

```
B_classify: pipeline on 38,840 unified codas.
  classified 37,699 / 38,840  (1,141 outside 3-10 click range)
  dswp-truth=          8,719  ← perfect direct join
  ec-knn (radius)=    13,161  ← 44 % of non-DSWP rows match an EC type
  pacific-discovered=  5,700  ← second OPTICS pass yielded clusters
  noise=              10,119  ← second-pass "rest" + tiny-cluster

Acceptance metrics:
  DSWP join exactness:        100.00%  [PASS]
  Aggregate NOISE rate:        28.43%  [FAIL, target was ≤ 8%]
  rhythm-18 vs DSWP truth:    100.00%  [PASS]
```

LOO accuracy of the current centroid+radius classifier on DSWP only:
**85.5 %** total, with sharp degradation on n=6 (56 %). For comparison:

| Validated reference numbers | Where measured |
|---|---|
| ELKI OPTICSxi reproduces CodaType at **95.9 %** on DSWP | Phase 1, `logs/phase1_gero21_finer_sweep.log` |
| kNN k=5 (Phase 1b option 2) **97.64 %** LOO on DSWP | `logs/phase1b_knn.log` |
| Current centroid+radius classifier **85.5 %** LOO on DSWP | `src/validation/loo_centroid_classifier.py` |

The current implementation classifies non-DSWP rows by:
1. Grouping DSWP rows by ground-truth `CodaType` per length n.
2. Computing centroid + radius_95 (floored at 0.10 s) per group.
3. For each non-DSWP coda x of length n: nearest live (non-NOISE)
   centroid; if `dist ≤ radius_95` → inherit that group's CodaType,
   else outlier.
4. Outliers go through a second OPTICSxi pass (xi=0.04, minpts=10) to
   discover Pacific-native types. ~80 % land in ELKI's "rest" bucket
   (cluster_id=0) and become `{n}-NOISE`.

That second-pass blow-up is the dominant source of the 28 % NOISE rate.

---

## 2. Why the current approach is stuck at 28 %

Three independent failure modes, in order of severity:

**(F1) Centroid+radius loses multi-modal type structure.**
Some Gero CodaTypes have *multiple modes* in ICI space — the n=6 bucket
is the worst offender. 6i (188 codas) and 6R (67 codas) have centroids
close together with overlapping spatial distributions; a single per-type
centroid loses the boundary that local kNN can detect. Hence the 56 %
LOO on n=6 vs kNN's 97 %.

The "merge sub-clusters by `dominant_codatype`" step the current code
does (see `next_phases_implementation_notes.md` §1) was intended to
solve a different problem (radius_95 too tight). It did solve that but
introduced this new one. The fix is to **stop using centroids as the
predictor**, even though they remain useful as reference points.

**(F2) The radius-or-bust outlier flag is binary and miscalibrated.**
A non-DSWP coda is either inside `radius_95` (assigned an EC type) or
outside (sent to stage 3). With `RADIUS_FLOOR_S = 0.10 s` we get 44 %
EC matches; without the floor we got 25 % (~75 % outliers). Neither
calibration is satisfying — there's no principled threshold that
balances "be inclusive of variant Pacific renditions of EC types" with
"reject genuinely novel Pacific types". This is fundamentally a
**density-and-clan-aware** decision that a global threshold can't make.

**(F3) OPTICSxi at minpts=10 over-rejects on the outlier subset.**
On the 16 K stage-2 outliers, ELKI's xi-extraction at minpts=10 places
~80 % into the "rest" bucket. Trying minpts=5 reduced NOISE 28 % → 24 %
but quadrupled the Pacific vocabulary (190 → 369 micro-types, half of
them with only 5–9 members). OPTICSxi just isn't the right shape for
that data: outlier Pacific codas are *dispersed clan repertoires*, not
the dense multimodal structure OPTICS-Xi was designed for.

---

## 3. Recommended architecture (v2)

Three stacked layers. Each is independently runnable; if an earlier
layer hits the acceptance target alone, you can skip the later ones.

### Layer 1 — kNN label propagation (replaces centroid+radius)

Use Phase 1b option 2's per-length `KNeighborsClassifier(k=5,
metric="euclidean")` to predict `CodaType` for every non-DSWP coda.
This is the **proven path**: 97.64 % LOO on DSWP. Code already lives at
`src/validation/phase1b_knn.py`. Lift it into the pipeline.

Alongside the kNN prediction, compute and store:
- `nn_distance` — distance to the nearest DSWP labelled coda of that
  length (kNN-1 distance). Phase 1b reported the 95th-percentile of
  this distance over Pacific is 0.24 s; the median is ~0.05 s.
- `nn_codatype` — predicted CodaType (kNN majority vote).

### Layer 2 — Variance-aware outlier flagging

A coda is outlier iff EITHER:
- `nn_codatype` ends in `-NOISE` (the "junk inheritance" Phase 1b
  identified — 28 % of Pacific codas hit this).
- `nn_distance > τ(nn_codatype)` where `τ(t)` is a per-type empirical
  threshold derived from the variance analysis below.

The per-type threshold `τ(t)` is the key new degree of freedom. The
plan-faithful `radius_95` failed because DSWP within-type spread isn't
the right yardstick. The right yardstick is: **how far does a DSWP
coda of type t typically sit from its nearest other-DSWP coda of the
same type?** That nearest-neighbour-within-class distance is the
density scale of the type, not the bulk spread.

Empirical recipe to set τ(t):
1. For each Gero type t, for each DSWP member, compute its nearest
   DSWP-of-same-type distance (nearest-of-same-class, NOSC).
2. `τ(t) = q-th percentile of NOSC for type t`, with q ∈ {95, 97, 99}.
3. Re-validate: rerun layer 2 on DSWP LOO. `nn_distance > τ(nn_codatype)`
   should hold for ≤ (100-q)% of DSWP rows.

Expected outcome: tight types like 5R1 get small τ (~10 ms), wider
types like 1+1+3 get larger τ (~150 ms), NOISE types get dropped from
the live vocabulary. A Pacific coda close to a tight EC type but with
slightly more variance is now correctly absorbed; a Pacific coda far
outside the natural spread of any EC type is correctly flagged.

### Layer 3 — Pacific outlier handling

Three sub-options, in order of increasing investment:

**3a. Don't try to cluster outliers — name them by length only.**
Outliers become `{n}-OUTLIER` (or just keep them with their kNN-nearest
EC name as a "best guess" and a high `classifier_distance` score).
Downstream consumers can filter on distance. This trades discovery for
honesty: we admit we don't know what they are.

**3b. HDBSCAN on outliers (replaces OPTICSxi).**
HDBSCAN is more flexible than OPTICSxi for the varying-density profile
the Pacific outliers actually have. Implementation in
`scikit-learn-contrib/hdbscan` or `sklearn.cluster.HDBSCAN` (1.3+).
Expected behaviour: better cluster recovery on dispersed clan
repertoires, less aggressive "rest" bucket. Locked params don't apply
here (HDBSCAN is the tunable: `min_cluster_size=10`,
`min_samples=5`, `cluster_selection_epsilon=0.05` are reasonable
starting points).

**3c. Per-Hersh-clan classifier (the ambitious option).**
Hersh 2022 supplement Discussion S2 names 7 Pacific clans (Palindrome,
Rapid Increasing, Slow Increasing, Plus-One, Four-Plus, Regular,
Short). Each clan has a characteristic coda repertoire. The `clan`
column in `data/upstream/codas_unified.csv` carries Hersh's clan label
for every Hersh row.

Per clan, do layer 1 + 2 + a per-clan kNN/centroid analysis:
- Build a per-(clan, length) centroid set.
- A coda's nearest centroid is now scoped to its own clan.
- If a clan has a strong native type (e.g. Plus-One clan loves
  `1+1+5`), the centroid will reflect that and per-clan recall will be
  high.

This is the most accurate but most work. It also requires confidence
in the clan labels, which Hersh derived from the *original*
mclust/IDcall classifier — circular if we use the same input data.
Sanity-check the clan labels against the kNN predictions before
investing here.

---

## 4. Variance analysis — the diagnostic to run first

You suggested this and you were right. Before any of layer 1–3, run
`src/validation/variance_analysis.py` (to be written) producing:

### 4.1 Within-DSWP variance per CodaType

For each of Gero's 21 published types t:
- `n_members` (members in DSWP of type t)
- `centroid` (mean ICI vector)
- `mean_dist_to_centroid` (inertia / n)
- `r95` (the radius_95 we already use)
- `nosc_p50, nosc_p90, nosc_p95, nosc_p99` (nearest-of-same-class distance percentiles)

Output: `data/diagnostics/dswp_variance.csv` plus a 2D PCA scatter per
length, plotting type centroids labelled.

### 4.2 Cross-corpus distance distribution

For every Pacific coda x of length n:
- nearest DSWP-of-same-length distance: `dswp_nn_dist`
- predicted CodaType (kNN k=5)
- distance to predicted-CodaType centroid: `pred_centroid_dist`
- distance to *next-nearest* type's centroid: `runner_up_dist`
  (gives a margin score)

Output: `data/diagnostics/pacific_distances.csv` plus histograms per
length and per Hersh clan.

### 4.3 Per-clan analysis

For each Hersh clan c:
- Distribution of kNN-predicted CodaType (which EC types do clan-c
  codas concentrate on?). Plan §A acceptance bullet 3 (Hersh-clan
  coherence test) wants this anyway.
- Per-clan within-clan variance: pick any two clan-c codas of the same
  length and predicted type; mean distance.
- Compare to Pacific-overall variance and to DSWP within-type variance.

Output: `data/diagnostics/clan_summary.csv` plus a per-clan stacked
bar chart of predicted CodaTypes.

### 4.4 What the analysis should tell you

Three plausible outcomes, each suggesting a different layer-3 path:

| outcome | suggests |
|---|---|
| Pacific within-type variance ≈ DSWP within-type variance | kNN + per-type τ (layer 2) is enough. Skip layer 3. |
| Pacific is 2–4× DSWP variance, but each clan is tight | Layer 3c (per-clan classifier). Pacific is a mixture of clan-specific repertoires. |
| Pacific is >5× DSWP variance even within a clan | Layer 3a (don't cluster, just flag) — the data isn't tightly typed. |

You won't know which outcome holds without running the analysis.
Allocate ~1 day for it before touching B_classify.

---

## 5. Acceptance criteria (revised, realistic)

These supersede `next_phases_plan.md`'s ≤ 8 % NOISE target, which we
have empirical reason to believe is unreachable with OPTICSxi-on-Pacific.

| metric | target | how measured |
|---|---|---|
| DSWP join exactness | 100 % | direct join, automatic |
| DSWP rhythm-18 reproduction | ≥ 95 % | DSWP rows joined truth → collapse → match |
| DSWP loo via the new classifier | ≥ 95 % | LOO test analogous to `loo_centroid_classifier.py` but using whatever new classifier you build |
| **Aggregate NOISE rate** | **≤ 15 %** | classifier_origin == "noise" / total classified |
| Hersh-clan coherence | each clan's top-5 predicted CodaTypes cover ≥ 70 % of clan codas | new test in tests/ |
| EC-type-recall on DSWP | ≥ 90 % | fraction of Gero's 21 types whose centroid is reproducible from a 50-row DSWP sample |

If you hit ≤ 15 % NOISE with layer 1 + 2 alone, that's a complete win.
If 15 % requires layer 3, document which sub-option you chose and why.
If 15 % is still unreachable, lift the target to 25 % and investigate
whether the residual NOISE codas are biologically genuine "could-not-be-
classified" data (some Sharma-birth codas may genuinely be
mid-development variants that don't belong to any clan's adult
vocabulary).

---

## 6. Implementation order

Each step is one day or less and produces a verifiable artefact.

**Step A — Variance analysis (1 day).**
Write `src/validation/variance_analysis.py`. Produce the three CSVs
listed in §4. Eyeball the histograms. Decide which layer-3 path to
take. Commit the CSVs + plots under `data/diagnostics/` (the directory
already gets gitignored if it's large; if not, add to `.gitignore`).

**Step B — Replace centroid+radius with kNN + per-type τ (1–2 days).**
In `src/pipeline/B_classify_optics.py`:
- Replace `classify_other_codas()` body with kNN k=5 (lift from
  `src/validation/phase1b_knn.py`).
- Add `compute_type_thresholds(df_dswp_truth)` returning
  `{codatype: τ}` from §4.1.
- Outlier flag: `predicted_codatype.endswith("-NOISE")` OR
  `nn_distance > τ[predicted_codatype]`.
- Keep `classifier_origin` column with new categories
  (`ec-knn-typed`, `ec-knn-noise-rejected`,
  `ec-knn-distance-rejected`, `pacific-discovered`, `noise`) for
  observability.
- Re-run `loo_centroid_classifier.py` adapted for kNN; expect
  ≥ 95 %, ideally 97.6 %.

**Step C — Try layer-3 sub-options (1–3 days, depends on §4).**
If layer 2 alone hits ≤ 15 % NOISE: stop, ship.
Else implement 3a, 3b, or 3c per §4.4. For 3b, install hdbscan
(`pip install hdbscan` or use sklearn 1.3+'s `cluster.HDBSCAN`).
For 3c, plan extra time for the clan-label sanity check.

**Step D — Hersh-clan coherence test (0.5 day).**
Add a test (slow, marked) that asserts each Hersh clan concentrates on
≤ 5 distinct CodaType labels covering ≥ 70 % of its codas.

**Step E — Document deviations (0.5 day).**
Update `next_phases_implementation_notes.md` (or fork to a v2 notes
file) with the measured numbers, deviations from this plan, and
findings from §4 the future you should know.

Total: 4–7 days of focused work.

---

## 7. Code pointers

Existing files to reuse / read:

| purpose | file |
|---|---|
| ELKI OPTICSxi wrapper | `src/validation/elki_optics.py` |
| Phase 1 reverse-engineering | `src/validation/reproduce_gero21.py` |
| Phase 1b option 2 (kNN, validated 97.64 %) | `src/validation/phase1b_knn.py` |
| Current LOO test for centroid+radius | `src/validation/loo_centroid_classifier.py` |
| Naming function (Gero notation) | `src/pipeline/cluster_names.py` |
| Pipeline orchestration | `src/pipeline/B_classify.py` |
| Stage-1/2/3 implementation | `src/pipeline/B_classify_optics.py` |
| Tempo / rubato / ornament logic | `src/pipeline/B_classify.py` (kept verbatim from prior version) |

New files to add:

| purpose | path |
|---|---|
| Variance analysis | `src/validation/variance_analysis.py` |
| kNN-based classifier (replaces stage 2) | edit in `src/pipeline/B_classify_optics.py` |
| HDBSCAN-based outlier discovery (3b) | new module `src/pipeline/hdbscan_outliers.py` |
| Per-clan classifier (3c) | new module `src/pipeline/clan_classifier.py` |
| Coherence test | append to `tests/test_classify.py` |

---

## 8. Things not to re-litigate

These have been settled by Phase 1 / 1b / current implementation:

- **ELKI 0.7.1 vs sklearn for OPTICSxi.** ELKI is the answer; sklearn's
  xi-extraction over-fragments. See `papers_relevant.md`.
- **xi=0.04.** Locked. Gero's published value.
- **minpts=10 for the first OPTICS pass on DSWP.** Locked. Reverse-
  engineered in Phase 1; sharp peak with neighbours all ~17 pp lower.
- **Per-length bucketing.** Locked. Gero's method.
- **3–10 click range.** Locked. Gero's method excludes <3 and >10.
- **Direct join for DSWP CodaType labels.** 100 % accurate by
  construction. Don't re-derive.
- **kNN k=5 for cross-corpus propagation.** Validated at 97.64 % LOO
  in Phase 1b. Don't sweep k unless you have a specific hypothesis
  about why it should matter.
- **Sharma 2024 tempo bins (0.45, 0.61, 0.93, 1.08).** Locked.
- **Sharma 2024 rubato cutoffs (the empirical 25/75 percentiles).**
  Locked.
- **Ornament rule applies to DSWP+birth only, NaN for Hersh.** Locked
  by `feedback_ornament_scope.md` (in memory) — Hersh data lacks
  per-coda timestamps so the rule is structurally inapplicable.

---

## 9. Open questions to keep in mind

- **Is the 28 % NOISE biologically real?** Some non-trivial fraction
  of Sharma-birth codas may be developmental variants that genuinely
  don't fit any clan's adult vocabulary. Worth checking the age /
  developmental metadata if it exists, and reporting NOISE rate
  separately for adult Hersh vs neonatal birth codas.
- **Are Hersh's clan labels trustworthy?** They came from Hersh's own
  mclust/IDcall classifier, not from a process we've re-validated.
  Before per-clan analysis, sanity-check by computing inter-clan
  coda-distance separability — does PCA on ICI vectors recover the
  clan structure? If not, the clan label is more noise than signal.
- **Why is n=6 a cliff?** It dropped to 56 % LOO under centroid+radius
  while neighbours sit at 87 %. With only 427 DSWP codas in n=6 and
  6-NOISE being 40 % of them, the live vocabulary is just 6R + 6i —
  two types with overlapping centroids. kNN should fix it. If kNN
  *also* fails on n=6, that's a structural issue worth investigating
  separately.
- **Do we ever want to merge similar Pacific micro-types?** With
  layer-3 options 3b/3c we may produce dozens of clan-specific types.
  A post-hoc merge step (centroids within X distance combine) could
  collapse near-duplicates and shrink the vocabulary to something
  human-comprehensible.
- **Should the 1–2-click and 11+-click codas be reintroduced?** Gero
  excluded them as <5 % outliers. Hersh's data may have a different
  distribution. Worth a histogram per source.

---

## 10. Quick reference: locked numbers in one place

```python
# OPTICSxi (DSWP only, first pass, validated to reproduce Gero-21 at 95.9%)
XI                = 0.04
MINPTS            = 10
DISTANCE          = "euclidean"
LENGTH_RANGE      = range(3, 11)

# kNN (Phase 1b option 2, validated 97.64% loo)
KNN_K             = 5

# Tempo (Sharma 2024 §4)
TEMPO_THRESHOLDS  = (0.45, 0.61, 0.93, 1.08)

# Rubato (Sharma 2024 §5)
RUBATO_LO         = -0.021416925000000087
RUBATO_HI         =  0.018462550000000105
RUBATO_T_DIFF_S   = 10.0

# Ornament: applies to DSWP + birth, NA for Hersh.
ORNAMENT_T_DIFF_S = 10.0  # used in current code; Sharma's exact value not stated
```

ELKI prerequisites: `vendor/jre8/bin/java` (Adoptium Temurin JRE 8) and
`vendor/elki/elki-bundle-0.7.1.jar` (Maven Central). Both gitignored.

---

**End of plan.** When you finish v2, write a `pacific_classifier_v2_implementation_notes.md`
analogous to the v1 notes file, capturing what you actually built, what
deviated from this plan, and the measured numbers. Future-future-you
will thank you.
