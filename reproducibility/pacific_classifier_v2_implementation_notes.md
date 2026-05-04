# Pacific classifier v2 — implementation notes

> **Status (2026-05-04): superseded by
> `whale_grammar_transformer_plan.md`.** The hybrid OPTICSxi classifier
> implementation from these notes is shipped and regression-tested at
> `src/pipeline/B_classify_optics.py`. Active follow-up work (transformer
> port + retrain + grammar findings) lives in
> `whale_grammar_transformer_plan.md`.

Frozen 2026-05-04 after §0 complete. This file picks up where the v2
plan left off. Read it cover-to-cover before doing anything; assume **no
memory** of the conversations that produced the plan or this file.

If you only read one paragraph: §0 of the v2 plan is done. The data-
quality "smoking gun" hypothesis was rejected (no echolocation, no
recording-quality concentration). NOISE has strong clan-level structure
that supports algorithmic work. The plan's layer-1/2/3c architecture
stands. **Your next deliverable is `src/validation/variance_analysis.py`
(plan §4 / implementation order Step A).** Specifics in §3 below.

---

## 1. Read these in order

1. `reproducibility/README.md` — what Phase 1 / 1b proved.
2. `reproducibility/parameters_locked.md` — locked parameter set.
3. `reproducibility/papers_relevant.md` — what each paper contributes.
4. `reproducibility/next_phases_plan.md` — v1 plan (produced current code).
5. `reproducibility/next_phases_implementation_notes.md` — what was
   actually built in v1, with measured numbers.
6. `reproducibility/pacific_classifier_v2_plan.md` — v2 plan
   (the document you're picking up). Especially §0, §3, §4, §5, §6.
7. **this file** — what's been done so far, what to do next, what's
   changed about the plan.
8. `data/diagnostics/SUMMARY.md` — the §0 findings in tabulated form.
9. `src/pipeline/B_classify.py` and `src/pipeline/B_classify_optics.py`
   — current code. Don't edit until Step B.
10. `src/validation/phase1b_knn.py` — the validated kNN code you'll lift
    in Step B. Already used in Phase 1b (97.64% LOO).
11. `src/validation/loo_centroid_classifier.py` — LOO test for the
    *current* (centroid+radius) classifier. Use as a template for the
    new kNN+τ LOO test in Step B.

---

## 2. What's been done — Step §0 (data-quality diagnostics)

### 2.1 Code

Wrote `src/validation/data_quality_diagnostics.py`. Runs 5 diagnostics
on `data/classified/codas_classified.csv` (the v1 classifier output);
the 6th diagnostic is a manual read of the Hersh supplement.

Run it again, idempotently:
```
python -m src.validation.data_quality_diagnostics
```

Outputs to `data/diagnostics/`:
- `SUMMARY.md` — narrative findings (read this).
- `noise_by_recording.csv` — 268 rows: per-recording NOISE rate, sample
  size, source, median click length.
- `noise_by_clan.csv` — 9 rows: per-clan NOISE rate.
- `distance_distribution.csv` — `classifier_distance` percentiles for
  ec-knn vs noise rows.
- `noise_pca.csv` — long-form PC1/PC2 per NOISE coda + clan + recording
  (per-length PCA, plain-numpy SVD).
- `pca_plots/noise_pca_n{3..10}.png` — scatter plots, points coloured
  by clan.
- `echolocation_check.csv` — per-length count of NOISE codas that look
  echolocation-like.

Log: `reproducibility/logs/data_quality_diag.log`.

Not committed yet — `data/diagnostics/` is currently untracked. Decide
when committing whether to gitignore it (large PNGs) or keep it under
source control as a verified artefact. The CSVs are small (~50 KB
total); the PNGs are ~450 KB total. Keep the CSVs and SUMMARY.md, you
can re-run the rest.

### 2.2 Findings — what changes about the plan

Full numbers in `data/diagnostics/SUMMARY.md`. Compressed:

**(a) The ≤8% NOISE target was a category error.** The v1 plan's
≤8% target was attributed in v2's §0 to "Hersh's 5%". Reading the Hersh
supplement (`docs/pnas.2201692119.sapp.pdf`) cover-to-cover confirmed:
Hersh has *no per-coda NOISE rate*. IDcall is a Bayesian mixture
(mclust 2:15 components × 14 model families); every coda is assigned to
its most-likely component. The "5%" Hersh-style number that exists is
the data-inclusion threshold (Table S1: 5.8% of codas excluded for
being outside 3–10 clicks or in <25-coda repertoires), and it is
**upstream** of classification — those codas never enter the
call-classification stage at all.

The actual source of the "5%" benchmark is **Gero 2016's 5.9% DSWP
NOISE rate** (`docs/gero2016_methods_extract.md` line 28). That's the
OPTICS-on-clean-corpus floor and Phase 1 already reproduces it.

**Consequence for the plan**: ≤15% NOISE (v2 plan §5) stands as the
right realistic target. Drop the "≤8% would have been ideal" framing —
it was never achievable on a 4× messier corpus. When reporting NOISE,
report per-clan + aggregate, and benchmark per-clan against Gero's 5.9%.

**(b) NOISE has strong clan structure.** Per-clan NOISE rates:

| clan | rate | n | comment |
|---|---:|---:|---|
| EC1 (Sharma-birth) | 47.4% | 5,731 | neonatal codas, expected |
| PALI (Palindrome) | 47.3% | 2,124 | distinct repertoire |
| FP (Four-Plus) | 45.3% | 2,590 | distinct repertoire |
| REG (Regular) | 31.1% | 8,289 | partial overlap with EC types |
| RI (Rapid-Increasing) | 28.9% | 1,776 | |
| SH (Short) | 25.4% | 5,428 | |
| PO (Plus-One) | 22.3% | 2,083 | |
| SI (Slow-Increasing) | 7.1% | 1,265 | already at Gero's floor |

This is the strongest single signal in §0. SI clan sits at Gero's 5.9%
floor — its repertoire is well-covered by EC types. PALI/FP/EC1 sit at
~46% — their repertoires are distinct. **This validates layer 3c
(per-clan classifier) as the right path** if layer 1+2 alone don't hit
≤15% aggregate. The PCA plots (`pca_plots/noise_pca_n5.png` and
`noise_pca_n6.png`) show this structure visually: in n=6, FP NOISE
forms its own cluster separated from EC types.

**(c) No data-quality smoking gun.**
- Recordings: median NOISE rate per recording is 28% (matches
  aggregate). IQR 14–46%. A few well-sampled high-NOISE recordings
  exist (153, 230, 249, 136, CETI23-290) but they don't move the
  aggregate. Not a recording-quality story.
- Echolocation: 1 of 192 NOISE codas with n_clicks ≥ 8 looks
  echolocation-like. Rejected as a meaningful driver.
- Cross-species clicks: not directly testable from the available data,
  but if it were a major contributor, you'd expect echolocation-like
  patterns. None found.

**(d) classifier_distance is smooth and right at the floor.**

| subset | p10 | p25 | p50 | p75 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| ec-knn | 0.030 | 0.051 | 0.081 | 0.110 | 0.143 | 0.181 |
| noise  | 0.112 | 0.126 | 0.149 | 0.212 | 0.341 | 0.527 |

NOISE distribution begins exactly where ec-knn ends — that's the
`RADIUS_FLOOR_S = 0.10` threshold. NOISE p50 = 0.149 s. ~50% of NOISE
rows are within easy reach of variance-aware kNN+τ if τ for the
right CodaType is wider than 0.10 (which §4.1 will measure for each
type). The long tail (p99 = 0.527) is the genuinely-novel residual and
is what layer-3 has to deal with.

The distribution is **not bimodal** in the strong sense the plan §0
worried about — it's a smooth long tail. There's no obvious quality
break; the recoverable zone is just the near-half of a continuous
distribution.

### 2.3 Findings — what does NOT change

- Layer 1 (kNN k=5) is still the right propagation method. 97.64% LOO
  on DSWP from Phase 1b is uncontested.
- Locked parameters from `parameters_locked.md` all still hold.
- Direct join for DSWP CodaType labels still 100%.
- Per-length bucketing, 3–10 click range, ELKI 0.7.1 — all unchanged.

---

## 3. What to do next — Step A (variance analysis, §4 of plan)

**Deliverable**: `src/validation/variance_analysis.py` plus three CSVs
under `data/diagnostics/`. **Time**: ~1 day; the compute itself is
seconds. **Why this comes before Step B**: §4.4 of the plan tells you
which layer-3 sub-option to invest in based on the variance numbers.
You can't skip it without flying blind.

### 3.1 What to compute (verbatim from plan §4)

#### §4.1 Within-DSWP variance per CodaType → `dswp_variance.csv`

For each Gero CodaType `t` (the 21 published types — note these are
the *non*-NOISE types; the truth column carries `5R-NOISE` etc. too,
keep them in the CSV but mark them):

| column | definition |
|---|---|
| codatype | string, e.g. `5R3`, `1+1+3`, `5R-NOISE` |
| length | n_clicks the type belongs to (3–10) |
| n_members | number of DSWP rows with this CodaType |
| centroid | mean ICI vector (one column per ICI dim, or store as JSON) |
| mean_dist_to_centroid | mean Euclidean distance from members to centroid |
| r95 | 95th percentile of `member_to_centroid` (current "radius_95") |
| nosc_p50 | 50th percentile of nearest-of-same-class distance |
| nosc_p90 | 90th percentile |
| nosc_p95 | 95th percentile |
| nosc_p99 | 99th percentile |

NOSC = "nearest-of-same-class". For each DSWP member of type `t`,
compute its distance to the nearest *other* DSWP member of the same
type, in the same per-length bucket. NOSC is the **density scale of the
type** — it's the right yardstick for τ(t), the per-type threshold
layer-2 needs.

The current `radius_95` (r95) measures bulk spread; NOSC measures local
density. They're different and both worth keeping in the CSV.

Pseudocode:
```python
for n in range(3, 11):
    cols = [f"ICI{i}" for i in range(1, n)]
    dswp_n = df[(df["source"] == "sharma2024_dswp") &
                (df["n_clicks"] == n) &
                df[cols].notna().all(axis=1)
                & df["coda_type_truth"].notna()]
    X = dswp_n[cols].to_numpy(dtype=float)
    y = dswp_n["coda_type_truth"].astype(str).to_numpy()
    for t in np.unique(y):
        idx = np.where(y == t)[0]
        if len(idx) < 2:
            # NOSC undefined; record n_members and skip percentiles
            continue
        Xt = X[idx]
        cen = Xt.mean(axis=0)
        d_to_cen = np.linalg.norm(Xt - cen, axis=1)
        # NOSC: pairwise within-type, take min off-diagonal per row
        D = np.linalg.norm(Xt[:, None, :] - Xt[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        nosc = D.min(axis=1)
        rows.append({
            "codatype": t, "length": n, "n_members": len(idx),
            "centroid": cen.tolist(), "mean_dist_to_centroid":
            float(d_to_cen.mean()),
            "r95": float(np.percentile(d_to_cen, 95)),
            "nosc_p50": float(np.percentile(nosc, 50)),
            "nosc_p90": float(np.percentile(nosc, 90)),
            "nosc_p95": float(np.percentile(nosc, 95)),
            "nosc_p99": float(np.percentile(nosc, 99)),
        })
```

Get the truth column the same way `phase1b_knn.py` does (see
`coda_type_for_dswp` helper there): join `data/upstream/codas_unified.csv`
against `data/upstream/dswp_dominica_codas.csv` on `codaNUM2018` ↔
`source_coda_id`.

Also produce a 2D PCA scatter per length, with type centroids labelled.
Use the same `_pca_2d` helper from `data_quality_diagnostics.py` —
don't reinvent.

#### §4.2 Cross-corpus distance distribution → `pacific_distances.csv`

For every Pacific (= non-DSWP) coda x of length n with all ICIs
present:

| column | definition |
|---|---|
| unified_index | row index into the unified CSV (for joining back) |
| length | n_clicks |
| source | "hersh2022_pacific" or "sharma2025_birth" |
| clan | from unified.clan |
| dswp_nn_dist | distance to nearest DSWP-of-same-length row |
| pred_codatype | predicted via kNN k=5 against DSWP labels |
| pred_centroid_dist | distance to predicted-CodaType centroid |
| runner_up_dist | distance to *next-nearest* type's centroid |
| margin | runner_up_dist − pred_centroid_dist (gives confidence) |

The kNN here is identical to the one you'll use in Step B — lift from
`phase1b_knn.py`. Reuse the centroid lookup from §4.1 to avoid
recomputing.

Then plot histograms of `dswp_nn_dist` per length and per clan. These
histograms answer: how does Pacific within-type variance compare to
DSWP within-type variance?

#### §4.3 Per-clan analysis → `clan_summary.csv`

For each Hersh clan c (8 clans incl. EC1 birth + the NaN bucket):

| column | definition |
|---|---|
| clan | string |
| n_codas | total Pacific codas with this clan |
| top_5_codatypes | semicolon-separated string of top-5 predicted types |
| top_5_coverage | fraction of clan codas in top-5 types |
| within_clan_variance | mean pairwise distance within clan, per length, averaged across lengths (or pick a representative length) |
| nn_dist_p50 | median dswp_nn_dist for this clan |
| nn_dist_p90 | 90th percentile |

The `top_5_coverage` column is critical — Step D's coherence test wants
≥70% of each clan's codas concentrated in ≤5 types. If clan
coverage is already high in this analysis, the test will pass; if low,
you'll know layer 3c is needed and how much it'll improve things.

Also produce a per-clan stacked bar chart of predicted CodaTypes
(top-15 types per clan, "other" bucket below).

### 3.2 What the analysis tells you about Step C (verbatim from plan §4.4)

| outcome on §4.1 / §4.2 / §4.3 | suggests |
|---|---|
| Pacific within-type variance ≈ DSWP within-type variance | layer 2 (kNN+τ) is enough — skip layer 3 |
| Pacific is 2–4× DSWP variance, but each clan is tight | layer 3c (per-clan classifier) |
| Pacific is >5× DSWP variance even within a clan | layer 3a (don't cluster, just flag) |

§0 already gave a strong hint: clan-stratified NOISE rates plus the
n=6 PCA suggest "Pacific is 2–4× DSWP variance but each clan is tight"
— the layer 3c outcome. The variance analysis confirms or refutes that
quantitatively.

### 3.3 Acceptance for Step A

- All three CSVs written under `data/diagnostics/`.
- Per-length PCA scatter for §4.1 readable.
- Per-clan stacked bar chart for §4.3 readable.
- A 5-bullet summary appended to `data/diagnostics/SUMMARY.md` with the
  three numbers from the §4.4 table for the corpus you actually have.
- Decision recorded: which layer-3 sub-option (a / b / c / skip) you'll
  use in Step C, with the variance numbers that justify it.

---

## 4. After Step A — Steps B, C, D, E

### 4.1 Step B (1–2 days) — replace centroid+radius with kNN+τ

Edit `src/pipeline/B_classify_optics.py`. Specifically:

- Replace the body of `classify_other_codas()` (or whatever name the
  centroid+radius classifier currently has — check the file at
  start-of-Step-B time) with a kNN k=5 classifier per length.
- Lift `_ici_matrix` and the kNN logic from `src/validation/phase1b_knn.py`
  — it's already validated.
- Add `compute_type_thresholds(df_dswp_truth)` returning `{codatype:
  τ}` from `dswp_variance.csv` (§4.1 output). Use `nosc_p95` as τ
  (the plan suggests p95, p97, or p99 — start with p95 and only widen
  if Step B's NOISE rate is too aggressive).
- Outlier flag: `predicted_codatype.endswith("-NOISE")` OR
  `nn_distance > τ[predicted_codatype]`. Keep the outlier rows for
  the second OPTICS pass / layer-3.
- Update `classifier_origin` column. New categories:
  - `dswp-truth` — unchanged
  - `ec-knn-typed` — kNN matched and within τ
  - `ec-knn-noise-rejected` — kNN matched a `-NOISE` type
  - `ec-knn-distance-rejected` — kNN matched but `> τ`
  - `pacific-discovered` — second-pass clusters (layer 3, may not exist
    if you skip layer 3)
  - `noise` — final residual
- LOO-validate. Adapt `src/validation/loo_centroid_classifier.py` for
  kNN k=5 (drop the centroid + radius logic; replace with
  `KNeighborsClassifier` and the leave-one-out trick already in
  `phase1b_knn.py`'s `main()`). Expect ≥ 95%, ideally 97.6%.
- Re-run `python -m src.pipeline.B_classify` end-to-end. Capture the
  new aggregate NOISE rate and the per-clan rates. **Sanity gate**:
  aggregate must drop from 28.4% to ≤ 20% on layer-1+2 alone, and
  per-clan rates for SI / PO / SH must be ≤ 15%. If they're not, kNN+τ
  isn't doing what's expected — debug before moving to Step C.

### 4.2 Step C (1–3 days) — layer 3 sub-option

Choose based on Step A §4.4 table. Most likely choice: **3c**
(per-clan classifier). Implementation skeleton:

- New module `src/pipeline/clan_classifier.py`.
- Build per-(clan, length) centroid sets from Hersh-labelled data.
- A coda's nearest centroid is now scoped to its own clan.
- For Pacific codas, predict CodaType using the per-clan kNN; for
  DSWP codas, use the cross-clan kNN from Step B.
- Sanity-check before investing: §9 of the plan flags that Hersh's
  clan labels came from their own classifier and could be circular if
  we use the same input data. The way to sanity-check is to compute
  inter-clan separability via PCA on ICI vectors *without using* the
  clan label, and see whether the clan labels recover any natural
  structure. If PCA shows clear clan separation: trust the labels.
  If PCA is mush: clan label is more noise than signal and 3c won't
  help.

If the choice is 3b (HDBSCAN), install
`pip install hdbscan` (or use sklearn 1.3+'s `cluster.HDBSCAN`). Start
with `min_cluster_size=10, min_samples=5,
cluster_selection_epsilon=0.05`. New module
`src/pipeline/hdbscan_outliers.py`. Replaces the second OPTICS pass.

### 4.3 Step D (0.5 day) — Hersh-clan coherence test

Append to `tests/test_classify.py`. Mark slow. Asserts each Hersh clan's
top-5 predicted CodaTypes cover ≥ 70% of clan codas. Plan §5
acceptance bullet 4. Use the variance analysis output as the data
source.

### 4.4 Step E (0.5 day) — finalise these notes

Update *this file* (`pacific_classifier_v2_implementation_notes.md`)
with the actual measured numbers from Steps A–D and what deviated
from the plan. Mirror the structure of v1's
`next_phases_implementation_notes.md`.

---

## 5. Things that might trip you up

### 5.1 The classified CSV exists; the unified CSV is the source of truth

`data/classified/codas_classified.csv` is the **v1 classifier output**.
The §0 diagnostics read it. But for Step A and onward, the source of
truth is `data/upstream/codas_unified.csv` (the input the classifier
runs on). Don't mix them up. The classified CSV will be regenerated
when you re-run `B_classify` after Step B.

### 5.2 The Sharma 2025 birth corpus is "EC1 clan" in the data

Birth codas (Sharma 2025) carry `clan == "EC1"` because they're
Dominica-clan calves. They are *not* Hersh-clan EC1. They appear in
`source == "sharma2025_birth"`. When stratifying by clan, decide
whether to keep them in the EC1 bucket or split out — the SUMMARY.md
diagnostic kept them separate via the source column. Step A §4.3
should split them — neonatal coda development is a different question
than adult clan repertoire.

### 5.3 NaN clan rows

682 Pacific codas have `clan IS NULL`. They sit at 29.6% NOISE — close
to aggregate. They're not a separate class; they're rows where Hersh's
own clan classifier didn't assign a clan. Keep them in the analysis
but don't over-fit; they're noise-equivalent for clan-level purposes.

### 5.4 Length 11+ and length <3 are excluded structurally

The pipeline enforces 3 ≤ n_clicks ≤ 10 (locked, Gero's method). 1141
unified rows fall outside this and have `classifier_origin IS NULL`
in the classified CSV. They are *not* in the 28% NOISE — they're
in a separate "not classified" bucket. Don't conflate. Plan §9 raises
the question of whether to reintroduce them; that's an open question,
not a Step-A task.

### 5.5 The NOISE PCA in §0 used unscaled ICI vectors

`_pca_2d` in `data_quality_diagnostics.py` does z-score scaling per ICI
dimension. That's fine for visual structure. But for variance analysis
in Step A, **do not scale** — Euclidean distance on raw ICI vectors is
the metric the kNN classifier uses, and τ has to be in the same units.

### 5.6 The Hersh "5%" hypothesis is closed

You may be tempted to revisit whether there's a per-coda quality
filter we're missing. There isn't. Method S1 of the supplement
(`docs/pnas.2201692119.sapp.pdf` page 2) is exhaustive on this. Don't
spend a half-day re-reading it.

### 5.7 The repo only has one prior commit (`f193fef`)

The v2 plan was committed as `79b6e62`. Everything else under
`reproducibility/`, `src/validation/data_quality_diagnostics.py`,
`tests/`, `vendor/`, etc., is currently untracked. Decide what to
commit before you start Step A; the gitignore will need an entry for
`data/diagnostics/pca_plots/` if you don't want PNGs in the repo.

---

## 6. Open questions to keep in mind (from plan §9, refined)

- **Is the 28% NOISE biologically real?** §0 mostly says yes. EC1
  birth codas at 47% NOISE are likely developmental variants. Worth
  reporting NOISE separately for adult Hersh vs neonatal birth at the
  end of Step B.
- **Are Hersh's clan labels trustworthy?** §0 didn't directly test.
  Step A §4.3's per-clan analysis + sanity-check PCA is when this gets
  resolved. If clan labels recover natural structure → trust them.
- **Why is n=6 a cliff under centroid+radius (56% LOO)?** Phase 1b
  showed kNN at k=5 fixes it (97% LOO). Step B should reproduce this.
  If kNN *also* underperforms on n=6, that's a structural anomaly worth
  separate investigation — but don't preemptively look for it.
- **Should length 11+ and 1–2-click codas be reintroduced?** Open.
  Not a Step-A or Step-B task. Worth a one-day study after v2 ships.
- **Do we ever want to merge near-duplicate Pacific micro-types?**
  Open. If layer 3 produces dozens of clan-specific types, a post-hoc
  merge step (centroids within X distance combine) could collapse
  them. Do this only after Step C.

---

## 7. Quick reference — locked numbers

```python
# OPTICSxi (DSWP only, first pass — used in v1, may or may not in v2)
XI                = 0.04
MINPTS            = 10
DISTANCE          = "euclidean"
LENGTH_RANGE      = range(3, 11)

# kNN (Phase 1b option 2, validated 97.64% LOO — use in Step B)
KNN_K             = 5

# Tempo (Sharma 2024 §4) — already implemented, don't touch
TEMPO_THRESHOLDS  = (0.45, 0.61, 0.93, 1.08)

# Rubato (Sharma 2024 §5) — already implemented, don't touch
RUBATO_LO         = -0.021416925000000087
RUBATO_HI         =  0.018462550000000105
RUBATO_T_DIFF_S   = 10.0

# Ornament (DSWP+birth only, NA for Hersh — already implemented)
ORNAMENT_T_DIFF_S = 10.0

# v1 numbers (current state, to beat)
V1_AGGREGATE_NOISE = 0.2843   # 28.43%
V1_NON_DSWP_NOISE  = 0.3377   # 33.77%
V1_LOO_DSWP        = 0.855    # 85.5%
V1_DSWP_JOIN       = 1.000    # 100%

# Targets (v2 plan §5, post-§0 revision)
TARGET_AGGREGATE_NOISE_MAX = 0.15   # ≤15%
TARGET_LOO_DSWP_MIN        = 0.95   # ≥95%
TARGET_CLAN_TOP5_COVERAGE  = 0.70   # ≥70% of each clan in top-5 types
GERO_DSWP_NOISE_FLOOR      = 0.059  # 5.9%, the OPTICS-on-clean-corpus floor
```

---

## 8. Task list state at handoff

(See TaskList for live state — these are the tasks in the system.)

- #1 §0 diagnostics 1–5 — **completed**
- #2 §0 diagnostic 6 (Hersh supplement) — **completed**
- #3 Step A — variance analysis — **pending, blocked-by removed (start here)**
- #4 Step B — kNN+τ — pending (blocked by #3)
- #5 Step C — layer-3 sub-option — pending (blocked by #4)
- #6 Step D — Hersh-clan coherence test — pending (blocked by #4)
- #7 Step E — finalise this file — pending (blocked by #6)

**Total remaining**: 4–6 days of work, depending on layer-3 choice.

---

## 9. Useful one-liners

```bash
# Re-run §0 diagnostics (a few seconds)
python -m src.validation.data_quality_diagnostics

# Re-run Phase 1b kNN (~10 min)
python -m src.validation.phase1b_knn

# Re-run v1 classifier end-to-end (~10 min, cached after first run)
python -m src.pipeline.B_classify

# Re-run v1 LOO test
python -m src.validation.loo_centroid_classifier

# Inspect classified output
python3 -c "
import pandas as pd
df = pd.read_csv('data/classified/codas_classified.csv', low_memory=False)
nd = df[df['source'] != 'sharma2024_dswp']
print('non-DSWP NOISE rate:', (nd['classifier_origin']=='noise').mean())
print(df['classifier_origin'].value_counts(dropna=False))
"
```

---

**End of notes.** When Step E rewrites this file at v2 completion, keep
this section structure and add a "what actually shipped" preamble
quoting measured numbers. Future-future-you will thank you.
