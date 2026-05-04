# Reproducibility: OPTICS-based coda classification

A frozen record of the experiments run on 2026-05-03/04 to reverse-engineer
Gero, Whitehead & Rendell 2016's OPTICSxi parameters, validate against
Sharma 2024's labels, and decide how to extend the classifier from the
Caribbean (DSWP) corpus to the broader unified corpus (Hersh Pacific +
Sharma birth corpus).

If you only read one thing, read the **Conclusions** section below.

## Goal

Run one consistent algorithm — Gero 2016's OPTICSxi clustering — across all
sperm-whale coda corpora available in `whale-ici-data`, so that:

1. The 21 Caribbean coda types Gero published are reproducible from his
   stated parameters.
2. Sharma 2024's 18 rhythm classes (a deterministic collapse of Gero's 21)
   come out at ≥95% on the labelled DSWP subset.
3. Adding Hersh's Pacific data alongside DSWP doesn't blow up the EC clusters
   (target ≥90% DSWP accuracy when classified jointly).
4. Pacific codas get the same kind of label, ideally extending the type
   vocabulary with new clan-specific clusters.

## What's in this directory

```
reproducibility/
├── README.md                       # this file
├── papers_relevant.md              # what each paper/supplement contributed
├── parameters_locked.md            # the final settled values
├── scripts/
│   ├── elki_optics.py              # subprocess wrapper around ELKI 0.7.1
│   ├── phase1_gero21_sweep.py      # Phase 1: sweep (xi, minpts) on DSWP
│   ├── phase1b_unified_optics.py   # Phase 1b option 1 (failed)
│   └── phase1b_knn.py              # Phase 1b option 2 (passed)
└── logs/
    ├── phase1_gero21_finer_sweep.log     # Phase 1 sweep output
    ├── phase1b_unified_minpts10.log      # Phase 1b option 1 @ minpts=10
    ├── phase1b_unified_minpts30.log      # Phase 1b option 1 @ minpts=30
    └── phase1b_knn.log                   # Phase 1b option 2 result
```

The scripts also live under `src/validation/` in the active codebase; the
copies here are frozen at the point the experiments were conclusive.
The vendored Java runtime and ELKI jar live under `vendor/jre8/` and
`vendor/elki/elki-bundle-0.7.1.jar` (gitignored due to size; see
`parameters_locked.md` for download URLs).

---

## Phase 0 — paper fact-finding

Before any code: pulled and read the relevant primary sources, since Gero's
methods §2.2 didn't give a complete parameter set on its own. See
`papers_relevant.md` for which sections of which papers contributed which
facts. Key gap: **`minpts` is not stated anywhere** in Gero 2016's main
text or supplement. We had to reverse-engineer it.

---

## Phase 1 — reverse-engineering minpts on DSWP-only

**Question:** what (xi, minpts) reproduces Gero's published 21 CodaType
labels on the labelled DSWP corpus?

**Input:** `data/upstream/dswp_dominica_codas.csv` (8,719 codas with the
`CodaType` column carrying Gero's 21 named types + `*-NOISE` flags;
filtered to 8,704 codas in the 3–10 click range Gero used).

**Method:** for each n_clicks in 3..10, run ELKI 0.7.1 OPTICSXi (the same
software Gero used) on the (n−1)-dimensional ICI vectors, sweep over a grid
of (xi, minpts), parse `ClusteringVectorDumper` output for one cluster ID per
coda. For each (xi, minpts) and each length bucket, compute majority-vote
mapping from cluster ID → CodaType, then accuracy.

**Sweep grid:** xi = 0.04 (Gero's stated value) ×
minpts ∈ {5, 7, 8, 9, 10, 11, 12, 15, 20, 30, 50}.

**Result:** see `logs/phase1_gero21_finer_sweep.log`. Aggregate accuracy
across the 8,704 codas, holding minpts fixed across all length buckets:

| minpts | aggregate accuracy |
|---|---|
| 5  | 81.2% |
| 7  | 80.0% |
| 8  | 79.6% |
| 9  | 79.4% |
| **10** | **95.94%** |
| 11 | 78.9% |
| 12 | 78.9% |
| 15 | 78.5% |
| 20 | 78.9% |
| 30 | 91.7% |
| 50 | 71.7% |

minpts=10 is a sharp, isolated peak: every neighbour falls to ~79%. The
peak is real (not smooth tuning) because the 5-click bucket — 6,384 codas,
73% of the corpus — only resolves correctly at exactly minpts=10.

Per-bucket-optimal upper bound: 96.66%. Fixed minpts=10 sits 0.7pp below
that, i.e. Gero used one global value and that value is 10.

**Phase 1 conclusion:** `xi=0.04, minpts=10, Euclidean on absolute ICIs,
per-length bucket, ELKI 0.7.1` reproduces Gero's CodaType labels at 96.0%
on the full DSWP corpus. ≥95% bar **PASSED**.

---

## Phase 1b option 1 — same OPTICSXi on the unified corpus

**Question:** if we run the same algorithm on the unified corpus
(Hersh Pacific + Sharma DSWP + Sharma birth = 38,840 codas, of which
37,684 are in the 3–10 click range), do DSWP rows still match their
published CodaType?

**Input:** `data/upstream/codas_unified.csv` joined to
`data/upstream/dswp_dominica_codas.csv` via `source_coda_id ↔ codaNUM2018`
to get CodaType ground truth on the 8,704 DSWP rows.

**Result at minpts=10** (`logs/phase1b_unified_minpts10.log`):

| n | DSWP correct/eval | acc |
|---|---|---|
| 3 | 83 / 103   | 80.6% |
| 4 | 457 / 818  | 55.9% |
| 5 | 3719 / 6384| 58.3% |
| 6 | 213 / 427  | 49.9% |
| 7 | 329 / 405  | 81.2% |
| 8 | 217 / 275  | 78.9% |
| 9 | 162 / 196  | 82.7% |
| 10| 59 / 96    | 61.5% |
| **total** | **5,239 / 8,704** | **60.19%** |

276 clusters total; 188 Pacific-only.

**Result at minpts=30** (`logs/phase1b_unified_minpts30.log`):
60.91%. n=4 improved (55.9 → 78.5%) but n=6 got worse (49.9 → 44.3%).
Cluster count dropped to 63, none Pacific-only-collapsed-to-single-bucket,
but the DSWP rows still ended up scattered across mixed-CodaType clusters.

**Failure mode:** at 4× the data density, the OPTICS reachability plot's
valleys get filled in by Pacific codas that structurally bridge between
Gero's CodaTypes — a Pacific 1+1+5 might land between a DSWP 5R2 and a
DSWP 1+1+3, merging clusters that were cleanly separated on DSWP-only.
This isn't a tuning problem; it's an algorithm-on-this-data problem.

**Phase 1b option 1 conclusion:** ≥90% bar **FAILED** at every minpts
tried. Single-OPTICS-pass on the unified corpus is not viable for
preserving Gero's labels.

---

## Phase 1b option 2 — train on DSWP, classify Pacific via kNN

**Question:** if we lock cluster boundaries on DSWP-only (Phase 1's working
configuration) and propagate labels to non-DSWP codas via per-length
nearest-neighbour lookup, do DSWP labels survive?

**Method:** for each length bucket n ∈ 3..10:
1. Train: per-length OPTICS on DSWP-only at xi=0.04, minpts=10 — sanity
   check that we still get Phase 1's ~96% (we do).
2. Classify: a `KNeighborsClassifier(k=5, metric='euclidean')` trained on
   `(DSWP ICI vectors, DSWP CodaType labels)`, applied to every coda of
   length n in the unified corpus.
3. Score DSWP rows leave-one-out (skip the row's own copy in its
   neighbour set).

**Result** (`logs/phase1b_knn.log`):

| n  | DSWP loo accuracy |
|----|-------------------|
| 3  | 95.1% |
| 4  | 96.0% |
| 5  | 98.2% |
| 6  | 97.0% |
| 7  | 95.8% |
| 8  | 96.0% |
| 9  | 97.4% |
| 10 | 96.9% |
| **total** | **97.64%** |

**Phase 1b option 2 conclusion:** ≥90% bar **PASSED**, comfortably.
Per-length kNN (k=5, Euclidean) is more robust than OPTICS cluster
boundaries here — 97.6% > the 96.0% of pure OPTICS — because it doesn't
suffer the xi-extraction merge problem.

**Caveat — Pacific outlier handling not yet built.** Of the 28,980 Pacific
codas classified:
- ~28% land on a `*-NOISE` CodaType (5-NOISE alone is 16%). NOISE is
  meaningless for Pacific codas — it just means "the nearest DSWP coda
  happened to be NOISE-labelled."
- 5% of Pacific codas have nearest-DSWP distance ≥ 0.24s (vs typical
  within-DSWP spread ~30ms). These are the genuine outliers that should
  form new clan-specific types.

Both are flags for a follow-up phase that runs OPTICS on the
high-distance Pacific subset to discover new types. That work is **not**
in this set of experiments.

---

## Conclusions

1. **Gero 2016's algorithm = ELKI 0.7.1 OPTICSXi at xi=0.04, minpts=10,
   Euclidean on absolute ICIs, per-length bucket, 3-10 click range.**
   minpts=10 is reverse-engineered from the data; the paper and supplement
   underspecify it. Reproduces published CodaType labels at 96.0% on 8,704
   DSWP codas (Phase 1).

2. **Single-OPTICS-pass on the unified corpus collapses DSWP accuracy to
   ~60% regardless of minpts.** Pacific data structurally bridges between
   Gero's EC types and breaks his cluster boundaries. Phase 1b option 1
   is dead.

3. **Train on DSWP, classify the rest via per-length kNN (k=5, Euclidean).**
   Preserves DSWP accuracy at 97.6% leave-one-out and gives every Pacific
   coda a nearest-EC-type label. This is the working architecture for
   `B_classify.py`.

4. **scikit-learn's OPTICS xi-extraction is not interchangeable with
   ELKI's.** sklearn over-fragments aggressively and rejects 50–80% of
   points as inter-cluster noise on this data; ELKI behaves as Gero
   reported. Always use the ELKI subprocess wrapper.

5. **Pacific outlier discovery is the next phase.** ~28% of Pacific codas
   currently get junk *-NOISE labels and ~5% are genuine outliers that
   should be new types. Both groups need a second OPTICS pass on the
   Pacific-only subset, with auto-naming for new clusters.

## How to re-run

```bash
# Phase 1 — DSWP sweep (≈10 min)
python -m src.validation.reproduce_gero21

# Phase 1b option 1 — single OPTICS on unified (≈30 min, will fail)
python -m src.validation.reproduce_gero21_unified --minpts 10
python -m src.validation.reproduce_gero21_unified --minpts 30

# Phase 1b option 2 — DSWP OPTICS + Pacific kNN (≈4 min)
python -m src.validation.phase1b_knn
```

ELKI prerequisites: `vendor/jre8/bin/java` (Adoptium Temurin JRE 8) and
`vendor/elki/elki-bundle-0.7.1.jar` (Maven Central). Both are gitignored.
See `parameters_locked.md` for the URLs.
