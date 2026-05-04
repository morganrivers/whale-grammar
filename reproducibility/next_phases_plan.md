# Next phases — detailed plan

> **Status (2026-05-04): superseded by
> `whale_grammar_transformer_plan.md`.** The hybrid classifier and
> transcript work this file scoped have shipped; the remaining items
> (transformer porting + retraining + grammar re-evaluation + ship) are
> tracked in `whale_grammar_transformer_plan.md`.

Frozen 2026-05-04. Reader: assume no memory of the conversation that
produced Phase 1 and Phase 1b. **Read `README.md`, `parameters_locked.md`,
and `papers_relevant.md` in this directory before starting** — they
establish the validated facts those phases depend on.

This document covers three pieces of work, in the recommended execution
order:

- **A. Pacific outlier discovery** — fix the ~28% of Pacific codas that
  currently get meaningless `*-NOISE` labels by detecting them as outliers
  and clustering them into new Pacific-native types.
- **B. Gero-21 auto-naming** — build the R1/R2/R3 / R/D/i/+ name
  generator so newly-discovered clusters from (A) inherit a name in
  Gero's notation rather than an opaque integer ID.
- **C. `B_classify.py` rewrite** — replace the trust-upstream-rhythm
  pipeline with the validated OPTICS+kNN+outlier-discovery approach.

Existing task list IDs (in `TaskList`):
- Phase 2 (Gero-21 auto-naming) ↔ section **B** here
- Rewrite B_classify.py ↔ section **C** here
- The NOISE fix is new — open it as a task before starting.

---

## A. Pacific outlier discovery

### Problem

`src/validation/phase1b_knn.py` trains a per-length `KNeighborsClassifier`
on DSWP codas (with Gero's `CodaType` labels) and applies it to the
Hersh Pacific + Sharma birth corpora. This passes the gating test —
DSWP loo accuracy is 97.6% — but it has two failure modes on Pacific
codas:

1. **Forced EC labels.** kNN picks one of Gero's 21 EC types for every
   Pacific coda, even when the coda is structurally far from any of
   them. Pacific clans (per Hersh 2022) have their own characteristic
   coda patterns; assigning them an EC name confuses two different
   things.
2. **Junk NOISE inheritance.** ~28% of Pacific codas inherit a
   `*-NOISE` label from kNN — *not* because they're noise, but because
   their nearest DSWP neighbours happened to be the DSWP codas Gero's
   OPTICS rejected as un-clustered. The label "5-NOISE" carries no
   information about the Pacific coda; it just says "your nearest EC
   match was also unclassified."

The 5% of Pacific codas with nearest-DSWP distance ≥ 0.24s (vs typical
within-DSWP spread ~30ms) are clear outliers needing their own clusters.
The 28%-NOISE problem partially overlaps with the 5%-far-distance
problem and partially captures Pacific codas that are *near* an EC
NOISE coda but should still be in a new cluster of their own.

### Approach

Two-stage classification. The principle: **a Pacific coda close to an
EC cluster keeps the EC label (real biological similarity — the rhythm
shape genuinely matches Gero's type). A Pacific coda far from every EC
cluster gets a Pacific-native label, not a forced EC name.** Each
Pacific coda gets one of three outcomes:

- **EC type** if it's structurally similar to a real (non-NOISE) DSWP
  cluster — biological similarity preserved.
- **New Pacific type** if it doesn't fit any DSWP cluster but joins
  enough other Pacific codas to form a new cluster (named via section
  B's auto-naming, e.g. `5P1`, `5P2`).
- **NOISE** if it doesn't fit any cluster, EC or Pacific (rare; expect
  <5% per Gero's noise rate baseline).

### Algorithm

For each length bucket `n` ∈ 3..10:

1. **Run ELKI OPTICSXi on DSWP-only** (xi=0.04, minpts=10 — locked
   params). This gives DSWP cluster IDs that match `CodaType` at ~96%.
   Compute, per cluster:
   - `centroid` = mean of member ICI vectors
   - `radius_95` = 95th-percentile within-cluster distance to centroid
   - `dominant_codatype` = mode of `CodaType` among cluster members
   - Drop clusters whose `dominant_codatype` is `*-NOISE` — these are
     not real types and should not anchor Pacific assignments.
2. **Classify each non-DSWP coda x** of length `n`:
   - For each non-NOISE DSWP cluster `c`, compute `d(x, centroid(c))`.
   - If `min_c d(x, centroid(c)) ≤ radius_95(c*)` for the closest `c*`
     → assign `x` to `c*` and inherit `dominant_codatype(c*)`.
   - Otherwise mark `x` as **outlier**.
3. **Cluster the outliers.** Run a second ELKI OPTICSXi pass on just
   the outlier ICI vectors at the same (xi=0.04, minpts=10).
   Each new cluster is a Pacific-native type. Codas left unclustered
   by this second pass become `*-NOISE` (now genuine noise, ≤5% of
   originally-outlier codas).
4. **Auto-name the new clusters** using section B's logic.

The threshold choice (`radius_95`) is the principled version of the 0.24s
heuristic mentioned in the Phase 1b analysis. It's per-cluster
(tighter clusters get tighter thresholds) and data-driven.

### Implementation skeleton

```python
# src/pipeline/B_classify_optics.py  (new file)

def cluster_dswp_per_length(df_dswp):
    """Returns dict[n_clicks -> ClusterModel] with centroid, radius_95,
    dominant_codatype per cluster."""

def classify_outsider_codas(df_others, dswp_models):
    """Per-length: assign to nearest DSWP cluster within radius_95;
    flag outliers."""

def discover_new_pacific_types(df_outliers, *, xi=0.04, minpts=10):
    """Second OPTICS pass, returns labels per outlier coda + per-cluster
    centroid/shape stats for naming."""
```

### Validation / acceptance

- DSWP loo accuracy on the joint pipeline must stay ≥97% (matches
  Phase 1b option 2's 97.64% — the outlier path doesn't touch DSWP rows).
- Final `*-NOISE` rate across all corpora must be ≤8% (Gero's reported
  5.9% on DSWP alone gives the floor; some additional Pacific noise is
  expected, but the current 28% inheritance bug must be gone).
- **Hersh-clan coherence test:** for each of Hersh 2022's 7 Pacific
  clans, plot the predicted CodaType distribution. We expect each clan
  to concentrate on a small handful of types (Plus-One clan should
  mostly land on plus-pattern types like 1+1+3, 1+1+5; Regular clan on
  *R types; etc.). Severe spread across many types in any one clan is a
  signal the outlier threshold is wrong.

---

## B. Gero-21 auto-naming (R1/R2/R3 / R/D/i/+ logic)

### Problem

OPTICSXi gives integer cluster IDs. To produce labels in Gero's notation
(`5R1`, `5R2`, `4D`, `1+1+3`, `7i`, …), we need a deterministic function
from a cluster's centroid + member set to a name string. We need this
because:

- Phase 1 used majority-vote on Sharma's published labels (cheating;
  works on DSWP only).
- Phase A above will produce *new* Pacific clusters with no published
  name, so we must generate one ourselves in the same vocabulary.

### Naming convention (from Gero 2016 main text §2.2.4)

Each EC name is `{n_clicks}{rhythm}{tempo_rank}` where:

- `n_clicks` — integer click count.
- `rhythm` — shape descriptor:
  - `R` — regular (ICIs roughly equal across the coda)
  - `D` — decreasing (ICIs monotonically shrink)
  - `i` — increasing (ICIs monotonically grow)
  - `+` notation — pause-separated, e.g. `1+1+3` means
    1 click, gap, 1 click, gap, 3 clicks. Detected by an ICI
    substantially longer than its neighbours.
- `tempo_rank` (optional, only when multiple clusters share the same
  `n_clicks` + `rhythm`) — `1`, `2`, `3` ordered by ascending mean total
  coda duration. So `5R1` is the fastest 5R cluster, `5R3` the slowest.

### Naming convention for Pacific-discovered types

EC names stay stable — never renumber a published Gero type. New
Pacific clusters from section A get a `P` marker so they're
unambiguously distinguishable.

- **Same shape as an existing EC rhythm, different tempo band:**
  `{n}{rhythm}P{rank}` — e.g. `5RP1`, `5RP2` for new 5-click regular
  clusters discovered in Pacific that don't fall inside any EC `5R*`
  cluster's `radius_95`. Rank ordered ascending by mean duration
  *within Pacific only*, independent of EC ranks.
- **Shape entirely novel (no EC type with this n_clicks + rhythm
  shape):** simplest form `{n}P{rank}` — e.g. `5P1`, `5P2`. Use this
  when the rhythm-shape detector returns `R/D/i` but no EC cluster of
  that shape exists, or for a new `+` pattern that doesn't appear in
  EC (e.g. `1+2+1` palindrome — written directly, no `P` needed since
  the `+` notation already disambiguates).
- **Genuinely new shape that isn't `R`, `D`, `i`, or `+`:** unlikely
  given the algorithm above, but if it happens, use the `+` notation
  with literal click groupings (Gero's fallback notation works for
  arbitrary shapes).

This way, every name in the final vocabulary is unique, EC names from
Gero 2016 are preserved verbatim, and new Pacific types are visually
distinguishable at a glance.

### Algorithm

```python
def name_cluster(centroid_icis, n_clicks, peer_clusters_with_same_shape):
    rhythm = detect_rhythm(centroid_icis)
    rank = tempo_rank(centroid_icis, peer_clusters_with_same_shape)
    return format_name(n_clicks, rhythm, rank)

def detect_rhythm(ic_vec):
    """ic_vec: (n_clicks - 1) absolute ICIs."""
    # 1. Detect '+': any ICI > 2× mean of its neighbours → split notation
    # 2. Else if monotone increasing within tolerance → 'i'
    # 3. Else if monotone decreasing within tolerance → 'D'
    # 4. Else 'R'
    # Tolerances should match what produces Gero's published names on the
    # DSWP corpus. Validate against the labels in dswp_dominica_codas.csv.
```

### Implementation skeleton

```python
# src/pipeline/cluster_names.py  (new file)

RHYTHM_RE = ...         # pattern → rhythm letter (R/D/i)
def detect_plus_notation(icis): ...
def detect_rhythm(icis): ...
def tempo_rank_within_shape(cluster, peer_clusters): ...
def name_cluster(centroid, n, peers): ...
```

### Validation / acceptance

- For each of Gero's 21 published CodaType strings (`5R1`, `4D`,
  `1+1+3`, …), the cluster our naming function produces on the DSWP-only
  OPTICS run should generate **the same string**. This is a per-cluster
  test, not aggregate.
- Acceptance: 21/21 named correctly, OR a written justification for any
  miss (e.g. "OPTICS split 5R1 into two sub-clusters due to xi-extraction
  edge case; we name both `5R1`; this is acceptable because Phase 1 also
  showed multiple sub-clusters of the same Sharma rhythm").
- **Pacific extension test:** run name-generation on section A's new
  Pacific clusters. Names must follow the convention above (`5RP1`,
  `5P1`, `1+2+1`, etc.) and must be unique across the whole vocabulary
  — no collision with any of Gero's 21 EC names.

---

## C. `B_classify.py` rewrite

### Current state (frozen at HEAD before this work)

`src/pipeline/B_classify.py` reads `codas_unified.csv` from
`whale-ici-data` (which already has `rhythm` and `extra_click` columns
populated upstream by a Manhattan-segmenter-based classifier we no
longer trust), and only adds `tempo` (Sharma 2024 bins) + `rubato`
(per-whale duration delta). It does **not** run any clustering — it
trusts upstream labels.

### Target state

`B_classify.py` should produce, per coda:

| column | source |
|---|---|
| `coda_type_gero21` | DSWP: direct join `source_coda_id ↔ codaNUM2018` from `dswp_dominica_codas.csv`. Non-DSWP: section A's classifier (kNN-to-DSWP-cluster within radius, or new-Pacific-cluster from outlier OPTICS, or genuine NOISE). |
| `rhythm_class_18` | Deterministic collapse of `coda_type_gero21` per Sharma 2024's dictionary (drops the R1/R2/R3 suffix). New Pacific types extend the vocabulary; assign new rhythm IDs ≥18. |
| `tempo` | Sharma 2024 bins, kept verbatim from current `_add_tempo`. |
| `rubato` | Per-whale duration-delta logic kept verbatim from current `_add_rubato`. |
| `extra_click` (ornament) | Sharma's structural rule. NaN for Hersh data (no per-coda timestamps). |

### Pipeline order inside `classify()`

1. `_attach_dswp_codatype(df)` — join DSWP rows to ground-truth labels.
2. `_classify_non_dswp(df, dswp_models)` — run section A's per-length
   nearest-cluster-within-radius logic; mark outliers.
3. `_discover_pacific_types(df_outliers)` — second OPTICS pass; assign
   new cluster IDs.
4. `_name_clusters(...)` — section B's auto-naming.
5. `_collapse_to_rhythm18(coda_type)` — drop tempo-rank suffix.
6. `_add_tempo(df)` — unchanged from current implementation.
7. `_add_rubato(df)` — unchanged from current implementation.
8. `_add_ornament(df)` — *new*: structural rule per Sharma 2024 §5,
   skipping Hersh rows.

### File-level changes

| file | change |
|---|---|
| `src/pipeline/A_load_unified.py` | already has `dominica_codas_csv()` fetcher (added during Phase 1). No further change. |
| `src/pipeline/B_classify.py` | rewrite. Keep tempo/rubato unchanged; add the OPTICS+kNN+outlier+naming pipeline above. |
| `src/pipeline/B_classify_optics.py` *(new)* | section A's classifier helpers. |
| `src/pipeline/cluster_names.py` *(new)* | section B's naming helpers. |
| `tests/test_classify.py` *(new)* | regression tests; see below. |
| `src/pipeline/C_render_readable.py` | adjust if rhythm-class encoding overflows the current `a..r` 18-letter alphabet. With Pacific-extended types we may need 25+ classes. Decide on the encoding (two-letter `aa..zz`, or numeric `r5_t2`) only after section A produces the actual count. |

### Drop the upstream rhythm/extra_click trust

`whale-ici-data`'s `codas_unified.csv` carries `rhythm` and `extra_click`
columns from its own classifier. After the rewrite, these columns are
irrelevant for downstream code. Two options:

- **Drop them in B_classify** before writing `codas_classified.csv` —
  cleanest, breaks anything still reading them.
- **Rename them to `upstream_rhythm`, `upstream_extra_click`** and keep
  alongside the new columns — preserves provenance, allows comparison.

Recommendation: rename. Useful for sanity-checking and for any future
attribution work.

### Validation / acceptance

The pipeline must reproduce these locked numbers on each run:

- DSWP rows in unified corpus: `coda_type_gero21` matches the join
  exactly (100%, since it's a direct join).
- DSWP loo: ≥97% (matches Phase 1b option 2). Test via a
  hold-out-one-clan or hold-out-one-unit split.
- Sharma-18 (rhythm class collapsed) on DSWP labelled subset: ≥95%.
- Aggregate `*-NOISE` rate across all corpora: ≤8% (per section A).
- Tempo and rubato distributions on DSWP rows must match the current
  pipeline's output to within rounding (these logic blocks are kept
  unchanged).

### Test plan

`tests/test_classify.py`:

```python
def test_dswp_codatype_join_is_exact():
    """Every DSWP row's coda_type_gero21 equals its CodaType in
    dswp_dominica_codas.csv."""

def test_dswp_loo_accuracy_meets_target():
    """Leave-one-out kNN on DSWP gives ≥97% Gero-21 reproduction."""

def test_pacific_noise_rate_below_threshold():
    """≤8% of all codas land on *-NOISE after outlier discovery."""

def test_gero_21_names_reproduce_published():
    """name_cluster() on the DSWP OPTICS clusters reproduces all 21 of
    Gero's published name strings."""

def test_tempo_and_rubato_unchanged_on_dswp():
    """tempo and rubato distributions match current implementation
    output to the byte (regression test)."""
```

---

## Order of execution

```
A (Pacific outlier discovery)
  └─→ B (Gero-21 auto-naming, needed to name A's new clusters)
        └─→ C (B_classify.py rewrite, uses A and B)
```

A and B can be developed in parallel by two people: B's tests run
against the existing DSWP OPTICS output without needing A. But C
should be written last so its test plan can lock the contract.

## Things future-you should *not* re-litigate

- ELKI 0.7.1 vs scikit-learn for OPTICSXi. ELKI is the answer; sklearn's
  xi-extraction over-fragments. See `papers_relevant.md` and Phase 1
  results.
- xi=0.04 vs other steepness values. Locked. Gero's published value;
  don't re-sweep.
- minpts=10 vs other minPts values. Locked. See
  `logs/phase1_gero21_finer_sweep.log` — there's a sharp peak at 10
  with all neighbours ~17pp lower.
- Whether to run one OPTICS pass on the unified corpus. No — Phase 1b
  option 1 collapses DSWP accuracy to 60% regardless of minpts.
- k for kNN. k=5, validated at 97.64% loo. Don't tune unless the
  outlier-discovery work changes the training set in a way that
  invalidates it (and even then, run a sweep before changing).
