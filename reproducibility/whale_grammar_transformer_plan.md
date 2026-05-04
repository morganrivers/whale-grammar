# whale-grammar transformer plan

Frozen 2026-05-04. Updated 2026-05-04 after Stages 1–4 shipped.
Reader: assume **no memory** of the prior conversations. Read in this
order before starting:

1. `reproducibility/README.md` — what Phase 1 / 1b proved.
2. `reproducibility/parameters_locked.md` — locked parameter set.
3. `data/diagnostics/hybrid_v1/RESULTS.md` — the four-numbers table,
   per-source fate table, and LOO τ-validation that this plan starts
   from.
4. `transformer-training.md` — transcript of the prior research session
   that produced a working MiniTransformer at 4.63 bits/token on a
   Sharma-only corpus, on branch
   `claude/whale-language-research-tEudI` of `~/Code/whale-gpt`.
5. **this file** — what's left to do.

---

## TL;DR

**Stages 1–4 are done and tested.** The hybrid classifier (Sharma anchors
→ kNN+τ matching → OPTICSxi on residual pool) is now production at
`src/pipeline/B_classify_optics.py`. The unified corpus is classified,
the human transcript carries inter-coda Δt tokens, and
`data/classified/whale_dialogues.csv` exists in whale-gpt's exact CSV
schema. 48 fast tests + 7 slow ELKI tests gate the four-numbers,
clan-coherence, cluster-quality, DSWP conservation, birth integration,
transcript rendering, and CSV emission.

**Stages 5–8 remain.** The work that's left is: copy the
MiniTransformer training code from `~/Code/whale-gpt`'s research branch,
retrain on the unified corpus, optionally re-run the grammar findings,
then document and ship. Approximately 3–5 days of focused work.

---

## 0. Scope decisions baked in (don't relitigate)

These are user decisions from the conversation that produced this plan;
they're settled and trying to revisit them costs more than it gains:

- **Repo split.** All transformer-adjacent work lives in **whale-grammar**.
  `whale-ici-data` stays a clean ICI-corpus build with no
  interpretation.
- **Manhattan distance is out of scope.** It's not used in this repo
  for tempo/rubato/ornament. Tempo is a bin from `coda_duration_s`,
  rubato is a per-whale duration delta vs prior matched coda, ornament
  is a structural ±1-click rule. Whatever Manhattan tree-search the
  upstream segmenter does is `whale-ici-data`'s problem
  (`docs/option_b_restore_manhattan_segmenter.md`).
- **Two outputs, two audiences.** The pipeline produces *both*:
  - `data/readable/whale_dialogues.txt` — **human transcript**.
    Existing `<rubato><letters><digit>` encoding *including* uppercase
    for ornamented codas, plus inline `Δt<seconds>` tokens with `Δt?`
    sentinel for codas without timestamps.
  - `data/classified/whale_dialogues.csv` — **transformer input**, in
    whale-gpt's CSV schema (see §3.2). One additive column for the
    inter-coda time delta.
- **Ornament is preserved end-to-end.** Uppercase letters in the human
  transcript; `Ornamentation1` / `Ornamentation2` columns in the CSV.
  Hersh's per-coda ornament is structurally NA (no timestamps); we
  emit `0` in the CSV for those rows by convention (Hersh has no way
  to fire the rule at all).
- **Inter-coda dt is included where available, sentinel otherwise.**
  DSWP timestamp coverage is 43 %, birth is 100 %, Hersh is 0 %.
  Numeric fixed-precision (2 d.p.) for available; `Δt?` token /
  `DeltaTime = -1.0` sentinel where missing. Don't quantize log-bins
  unless Stage 6 finds the model can't learn the continuous signal.
- **Transformer CSV follows whale-gpt's schema.** Adopted exactly so
  the prior research branch's training scripts read it with minimal
  changes. We extended with one column (`DeltaTime`) which existing
  scripts ignore.
- **Skip piano-roll / visual review.** Cluster-quality checks are
  numeric (variance ratio against nearest Sharma type). Eyeballing
  rhythms doesn't add information for the transformer pipeline.

---

## 1. Where we are now (2026-05-04)

### 1.1 Production pipeline

```
data/upstream/codas_unified.csv  (37,684 in-range codas of 38,840)
         │
         ▼
A_load_unified.py  (downloads from whale-ici-data on --refresh)
         │
         ▼
B_classify.py
         │
         ├─► B_classify_optics.run_optics_pipeline
         │     │
         │     ├─ Stage 1: Sharma 'real' anchor      (origin = dswp-real)
         │     ├─ Stage 2: kNN k=5 + τ matching      (origin = pacific-matched)
         │     ├─ Stage 3: OPTICSxi on pool          (origin = discovery-cluster
         │     │           (Sharma-NOISE ∪ Pacific-residual)        / discovery-noise)
         │     └─ Stage 4: cluster_names             (P5RP1, 5P2, …)
         │
         ├─► tempo / rubato / ornament (unchanged from prior version)
         │
         └─► rhythm_class encoding (NEW)
               int per coda_type_gero21 value;
               persisted to data/classified/rhythm_class_index.csv
         │
         ▼
data/classified/codas_classified.csv      (canonical artefact)
         │
         ├─► C_render_readable.py  →  data/readable/whale_dialogues.txt
         │     (uppercase ornaments, rubato prefixes, Δt<value> /
         │      Δt? tokens, pause annotations)
         │
         └─► E_render_csv.py       →  data/classified/whale_dialogues.csv
               (whale-gpt schema + DeltaTime column)
```

End-to-end command: `python -m src.pipeline.D_run --refresh`.

### 1.2 The four numbers (locked, regression-tested)

| # | What | Count | % |
|---|---|---:|---:|
| 1 | DSWP rows anchored to Sharma 'real' types | 8,119 | 91.5 % of DSWP |
| 2 | Pacific matched to Sharma via kNN+τ | 22,940 | 79.2 % of in-range Pacific |
| 3 | Pacific in 116 discovered clusters | 4,026 | 13.9 % of in-range Pacific |
| 4 | Final NOISE (discovery pool) | 2,341 | 35.3 % of pool |

`tests/test_four_numbers.py` enforces these within ±10 rows.

### 1.3 Validation gates already passing

- **Clan coherence** (`tests/test_clan_coherence.py`) — coherent clans
  EC1/REG must hit ≥ 70 % top-5 label coverage. All 8 Hersh clans must
  stay within ±5 pp of recorded top-10 coverage baseline (logged inline
  in the test). Diffuse clans (FP at 48.6 % top-10, etc.) are biology,
  not bugs.
- **Per-clan over-classification** (`tests/test_loo_per_clan.py`) — no
  Sharma clan above 2 % LOO over-classification (current EC1 0.26 %,
  EC2 0.47 %).
- **Cluster quality** (`tests/test_cluster_quality.py`) — ≥ 80 % of the
  116 discovered clusters have internal variance ≤ 2 × nearest Sharma
  type's variance (current 92.2 %; 9 flagged for potential
  `OTHER_PACIFIC` collapse if Stage 6 needs it).
- **DSWP conservation** (`tests/test_dswp_conservation.py`) — every
  Sharma 'real' row's `coda_type_gero21` equals its published
  `CodaType` exactly. Sharma-NOISE rows are excluded (deliberate
  recovery into discovery clusters; 258 / 600 recover).
- **Birth integration** (`tests/test_birth_integration.py`) — 93.2 %
  match rate, 99.98 % tempo populated, 2,562 rubato deltas, 657
  ornaments fired.
- **Transcript rendering** (`tests/test_render_readable.py`) — Δt
  numerics in DSWP/birth, Δt? sentinel in Hersh, uppercase ornaments
  and rubato prefixes still render.
- **Transformer CSV** (`tests/test_render_csv.py`) — schema, Hersh
  invariants, contiguous itemPosition, every Coda1 decodable through
  rhythm_class_index.csv.

Slow ELKI tests (`pytest -m slow`) verify the full ELKI pipeline still
reproduces 97 %+ DSWP LOO, ≤ 8 % NOISE rate, and the published Gero-21
vocabulary.

### 1.4 Pipeline output stats (regression baseline)

```
B_classify: classified 37,684/38,840 codas
  origin: dswp-real=8,119, pacific-matched=22,940,
          discovery-cluster=4,284, discovery-noise=2,341
  vocabulary: rhythm_class_18 has 127 labels; full rhythm_class has 131 labels
  tempo populated for 38,830 codas
  rubato populated for 4,786 codas
  ornament rule populated for 9,511 codas (926 ornaments). Hersh NA.
  acceptance: DSWP 'real' conservation 100 %; aggregate NOISE 6.21 %;
              rhythm-18 vs DSWP truth 100 %.

E_render_csv: 38,840 rows in 507 sequences
  Hersh:  Coda2=98 (always), DeltaTime=-1 (always), 0 ornaments
  DSWP:   847 simultaneous Coda2, 3,554 known DeltaTime, 269 ornaments
  Birth:  1,508 simultaneous Coda2, 5,702 known DeltaTime, 657 ornaments
```

---

## 2. Birth data — what it actually has

User asked. Verified by reading `data/upstream/codas_unified.csv`:

| source | rows | `time_in_recording_s` populated | `coda_duration_s` | published `CodaType` label |
|---|---:|---:|---:|---|
| sharma2024_dswp | 8,872 | 3,780 (43 %) | yes | yes (free join) |
| sharma2025_birth | 5,731 | **5,731 (100 %)** | yes | **no** |
| hersh2022_pacific | 24,237 | **0 (0 %)** | yes | no (only `clan` + `rhythm`) |

Stage 6 implications:

- **Birth has full timestamps** — useful for sequence boundary detection
  if the transformer benefits from sub-recording sequences.
- **Hersh has no timestamps** — every Hersh row's `DeltaTime = -1`. The
  model will see the sentinel as its own learned token. Don't try to
  fake dt for Hersh from `coda_duration_s`; it would be a structural
  lie.
- **DSWP partial timestamps** are split mostly by recording: most
  Atlantic-archive rows lack them; the recordings Sharma 2024 added
  back have them.

---

## 3. Reference: prior transformer work + adopted CSV schema

### 3.1 Prior numbers (Sharma-only, V≈207, K=8 past codas)

5-fold sequence-level CV from
`~/Code/whale-gpt @ claude/whale-language-research-tEudI`, captured in
`transformer-training.md`:

| model | params | bits/token | perplexity |
|---|---:|---:|---:|
| majority (smoothed unigram) | — | 5.99 ± 0.26 | 63.7 |
| Markov-1 | — | 5.42 ± 0.37 | 42.8 |
| Embedding-MLP (d=32, 256·128) | 131 k | 4.84 ± 0.41 | 28.7 |
| **MiniTransformer (2L, 4h, d=64)** | **126 k** | **4.63 ± 0.43** | **24.7** |

Headline take-aways:

- Param-matched embedding models beat brute-force one-hot MLPs.
- MiniTransformer wins on log-loss; mass is better-spread.
- Markov-2 is *worse* than Markov-1 (sparsity penalty).
- Trained on Sharma's ~9 k DSWP observations only. Unified gives ~37 k
  (4× more data) plus Hersh's Pacific repertoire.

### 3.2 Adopted CSV schema (already produced by Stage 4)

`data/classified/whale_dialogues.csv`:

| column | meaning | source |
|---|---|---|
| `sequenceId` | one per (source, recording_id), sub-split on > 60 s gaps | format `{source}::{recording_id}::{n}` or `…::{n}.{sub}` |
| `itemPosition` | 0-based position within the sequence | running counter |
| `Coda1` | rhythm_class integer for the primary whale | 98 = silence |
| `Ornamentation1` | 0 or 1 | NA → 0 (Hersh) |
| `Duration1` | seconds | `coda_duration_s` |
| `Coda2` | second whale's coda when within 0.3 s of Coda1's start in the same recording; else 98 | always 98 for Hersh (no timestamps) |
| `Ornamentation2` | as Ornamentation1 | as above |
| `Duration2` | as Duration1 | as above |
| `DeltaTime` | seconds since previous primary's start in this sequence; -1 sentinel | from `time_in_recording_s` |

The integer code → label mapping is in
`data/classified/rhythm_class_index.csv`.

Whale-gpt's existing scripts read the first 8 columns and ignore
extras, so `DeltaTime` is backward-compatible. Vocabulary in this run
is 131 (rhythm_class) + 1 (silence sentinel) = 132 distinct codes.

---

## 4. Stages

Each stage produces a verifiable artefact. Don't move on until the
acceptance gate hits.

### Stage 1 — Port the transformer training code (1 day)

Pull the working scripts from
`~/Code/whale-gpt`'s `claude/whale-language-research-tEudI` branch.

```bash
cd ~/Code/whale-gpt
git fetch
git checkout claude/whale-language-research-tEudI
ls scripts/ outputs/grammar/  # confirm what's in there
git ls-tree -r HEAD scripts/ | grep -E '\.(py|md)$'
```

Files to copy across (paths approximate; use `git ls-tree` to find
actual ones):

- `scripts/6_predict_kfold.py` — or whatever the latest predict-kfold
  script is named. `transformer-training.md` references it; confirm
  filename before copy.
- `scripts/2_*.py` through `scripts/5_*.py` — analysis pipeline that
  produced FINDINGS.md (needed for Stage 3 only; copy now while
  you're in the branch).
- `outputs/grammar/FINDINGS.md` — read it; the unified-corpus version
  is Stage 3's deliverable.
- `outputs/grammar/predict_results.md` — numeric baseline for Stage 2.

Copy them into a new directory `src/grammar/` under whale-grammar.
Don't re-use the v1 path — keep the prior research artefact intact in
its own branch. Adapt input paths to read
`data/classified/whale_dialogues.csv` instead of the branch's existing
CSV path.

#### 1.1 Decisions during port

1. **Sequence boundaries.** Already encoded in our CSV via
   `sequenceId` (one per recording, sub-split on > 60 s gaps via
   `_split_sequences` in `E_render_csv.py`). The branch's script may
   have its own sequence concept; map it to our `sequenceId`.
2. **Vocabulary.** V is dynamic from `rhythm_class_index.csv` (131 +
   silence = 132 in the current run). If the branch's script
   hardcodes V = 207, change it to read len(unique Coda1 ∪ {98}).
3. **DeltaTime column.** Initially **ignore** it — that lets us
   reproduce the branch's protocol on the new data with no other
   confounds. Add it as an additional input feature in a follow-up
   experiment in Stage 2.
4. **Held-out protocol.** 5-fold sequence-level CV exactly as in the
   prior branch; metric is bits/token. Fit the same model menu the
   branch did (majority, Markov-1, Embedding-MLP, MiniTransformer).

#### 1.2 Acceptance gate

The kfold script runs on `data/classified/whale_dialogues.csv` without
errors and writes a comparison MD into `outputs/grammar/`. Numbers
*do not* need to improve at this stage — that's Stage 2.

### Stage 2 — Retrain + benchmark (1 day, mostly waiting)

Run training on the unified-corpus CSV. Record results into
`outputs/grammar/predict_results_unified.md`.

Per-target metric: held-out bits/token (lower better). Compare to the
prior branch's numbers (§3.1):

| target | prior (Sharma-only V≈207) | unified |
|---|---:|---:|
| Token | 4.63 (MiniTransformer) | TBD |
| Rhythm | 4.24 (RF) / 4.41 (MLP) | TBD |
| Tempo | 2.53 (RF) / 2.53 (MLP) | TBD |

Improvement is *expected* on Token (more data, smaller V — 132 vs 207
— so the entropy floor is lower). On Rhythm, the cross-corpus rhythm
vocabulary is bigger than Sharma's V=18 (we have 127 rhythm_class_18
labels), so apples-to-apples is fuzzy.

If unified Token bits/token is within 0.3 of the prior 4.63, the
bigger corpus pays for its bigger vocabulary. If it's worse, look at:

- **Vocabulary cliff.** Some Pacific clusters might be tiny. The 9
  flagged-by-cluster-quality clusters (see
  `data/diagnostics/hybrid_v1/cluster_quality.csv`) are the obvious
  candidates to collapse into `OTHER_PACIFIC`. Doing this requires
  re-running B_classify with a post-pass that re-labels flagged
  cluster members; not implemented yet but straightforward.
- **Sequence definition.** Too long → context dilution; too short →
  no structure for the transformer to exploit. Try varying
  `SEQUENCE_BREAK_S` in `E_render_csv.py` (currently 60 s).
- **DeltaTime feature.** Once the V baseline is stable, re-train with
  `DeltaTime` as an additional input feature. Plan suggested numeric
  fixed-precision; if the model can't learn the continuous signal,
  fall back to log-bins.

#### 2.1 Acceptance gate

Trained MiniTransformer on the unified corpus; recorded bits/token
within ±0.5 of the prior 4.63; produced a comparison MD with new vs
old table.

### Stage 3 — Grammar findings on the unified corpus (1–2 days, optional but valuable)

The branch's `FINDINGS.md` has five concrete grammar discoveries on the
Sharma-only corpus:

1. Multi-coda grammar exists (rhythm MI at lag 12 = 0.16 bits).
2. Universal compressors confirm structure (bz2 26 % smaller than
   shuffled).
3. Concrete "words" — `i1 d1 b1 b1` 27× vs 0.23 expected.
4. Duality of patterning (rhythm × tempo decouple at sequence level).
5. Tempo-class boundary markers + responder entropy drop given
   initiator.

Re-run those analyses on the unified-corpus CSV:

- Same MI / compression diagnostics (`scripts/2_*.py` through
  `scripts/5_*.py`, ported in Stage 1).
- Acceptance: each finding either reproduces with bigger effect
  (more data = sharper signal — expected for #1, #2) or weakens
  (cross-corpus dilution — possible for #3, #5).
- New question only the unified corpus can answer: does the
  Hersh-Pacific token stream show the same MI structure? Is it the
  same grammar with new words, or a different grammar entirely? Slice
  the analysis by `source` to check.

This is the "actual scientific contribution" stage. If you're going to
ship one document at the end, this is the one. Capture findings in
`outputs/grammar/FINDINGS_unified.md`.

#### 3.1 Acceptance gate

MD with at least the five prior findings re-evaluated, plus one
comment on the Hersh-vs-Sharma grammar comparison.

### Stage 4 — Document + ship (0.5 day)

#### 4.1 README cross-links

- `~/Code/whale-ici-data/README.md` already links to `whale-grammar`.
- `~/Code/whale-grammar/README.md` already mentions `whale-ici-data`.
- Update each README's pipeline section so the transformer step is
  visible from either repo. Three clicks max from either README to
  `FINDINGS_unified.md`.

#### 4.2 Update `RESULTS.md` and superseded plan headers

`data/diagnostics/hybrid_v1/RESULTS.md` auto-regenerates from
`assignments.csv` so it stays correct on each run. The superseded
docs need manual headers:

- `reproducibility/pacific_classifier_v2_plan.md`
- `reproducibility/next_phases_plan.md`
- `reproducibility/pacific_classifier_v2_implementation_notes.md`

Add a top-of-file note like:

> Status (2026-05-04): superseded by
> `whale_grammar_transformer_plan.md`. Phase 1/1b history (variance
> analysis, OPTICS sweep) remains here; the rest is captured in the
> unified plan.

#### 4.3 New file: `outputs/grammar/REPRODUCING_TRANSFORMER.md`

Clean handoff for someone re-running everything from scratch
(`transformer-training.md` is the conversational version; this is
the polished one). Section pointers:

```
1. Clone whale-ici-data + whale-grammar.
2. Vendor ELKI + JRE 8 per parameters_locked.md §Vendoring URLs.
3. Run pipeline:  python -m src.pipeline.D_run --refresh
   produces:
     data/classified/codas_classified.csv
     data/classified/rhythm_class_index.csv
     data/classified/whale_dialogues.csv
     data/readable/whale_dialogues.txt
4. Verify:  python -m pytest tests/  (48 fast + 7 slow)
5. (optional) Inspect:  python -m src.validation.cluster_quality
6. Train transformer:  python -m src.grammar.predict_kfold
   produces: outputs/grammar/predict_results_unified.md
7. (optional) Grammar findings:
     python -m src.grammar.findings_unified
   produces: outputs/grammar/FINDINGS_unified.md
```

#### 4.4 Acceptance gate

New files exist; old plans marked as superseded; README pipeline
sections include the transcript + transformer steps.

---

## 5. Total time

| stage | min | max |
|---|---:|---:|
| 1 — port transformer code | 1 d | 1 d |
| 2 — retrain + benchmark | 1 d | 1 d |
| 3 — grammar findings (optional) | 1 d | 2 d |
| 4 — ship | 0.5 d | 0.5 d |
| **total** | **3.5 d** | **4.5 d** |

Stage 3 is the only optional one; everything else is on the critical
path to "transformer trained on the full unified corpus." If you skip
Stage 3 you have a working transformer but no scientific findings
document — fine for a model deliverable, miss for a research
deliverable.

---

## 6. Risks and mitigations

- **Hersh's 0 % timestamp coverage.** Without `DeltaTime`, the model
  sees a dense token stream with `-1` between every Hersh coda. The
  rhythm/tempo structure is intact; the model just learns the
  sentinel's meaning. Not a bug. Don't try to fake dt.
- **Cluster vocabulary cliff.** The 116 discovered Pacific clusters
  include 9 that `cluster_quality.csv` flagged as too sprawling
  (variance ratio > 2×). If Stage 2 bits/token is bad, collapsing
  those into `OTHER_PACIFIC` is a clean lever. Implementation: a
  post-pass in `B_classify_optics.run_optics_pipeline` that re-maps
  flagged labels.
- **MiniTransformer overfit on 4× more data.** The branch found
  bigger MLPs overfit on 4,800 codas. With 37 k codas the same
  architecture might be under-capacity. Stage 2 should also try
  MLP-M (256, 128) and a 4-layer MiniTransformer; if log-loss
  improves significantly, the per-corpus optimum shifted.
- **Vocab size 132 vs 207.** The unified vocabulary is *smaller* than
  Sharma-only because we collapsed many Sharma-NOISE rows into
  discovered clusters with proper names. This means lower entropy
  floors than the branch — bits/token comparisons aren't strictly
  apples-to-apples; the new run can be lower without proving the
  *model* improved. Document both V values when reporting.

---

## 7. Things explicitly **not** in this plan

These exist in adjacent docs and are deliberately deferred:

- **whale-ici-data refactor** (option_b) — segmenter belongs upstream;
  this repo doesn't re-implement it. We trust upstream's
  `rhythm` / `extra_click` for the human transcript path, and rely on
  our own classifier for the `coda_type_gero21` / `rhythm_class`
  columns the transformer reads.
- **Tempo-invariant / relative-ICI variant of OPTICSxi.** Open
  methodological question. Worth a side experiment *after* the
  transformer baseline lands; the baseline tells us whether
  absolute-ICI is good enough.
- **Per-clan classifier (layer 3c).** The discovery-pool approach we
  shipped is layer 3b in spirit. Per-clan is more work for unclear
  gain on the discovered-cluster count and creates clan-circularity
  risk.
- **whale-gpt sync.** User said they don't want to use whale-gpt as a
  pipeline. Stage 1 grabs the *training code* but not the data
  pipeline; the unified CSV stays the authoritative input.
- **Visual / piano-roll review of clusters.** Numeric cluster quality
  is sufficient for the transformer pipeline; eyeballing isn't on
  the critical path.

---

## 8. Appendix — quick reference

### 8.1 What's on disk after Stages 1–4

```
data/classified/
├── codas_classified.csv         (38,840 rows; canonical artefact)
├── rhythm_class_index.csv       (132 rows; int → label)
└── whale_dialogues.csv          (38,840 rows, 507 sequences;
                                  transformer input)
data/readable/
└── whale_dialogues.txt          (human transcript with Δt tokens)
data/diagnostics/hybrid_v1/
├── RESULTS.md                   (auto-regen from assignments.csv)
├── assignments.csv              (per-coda final label + origin)
├── discovered_clusters.csv      (per-cluster summary)
├── cluster_quality.csv          (variance ratios; 9 flagged)
├── loo_knn_tau.csv              (per-length LOO summary)
├── loo_knn_tau_rows.csv         (per-row LOO outcomes)
└── loo_knn_tau_by_clan.csv      (per-clan over-classification)
data/diagnostics/
└── dswp_variance.csv            (used by hybrid_classify; the
                                  production pipeline computes τ
                                  inline — this CSV is purely
                                  diagnostic)
```

### 8.2 Locked constants (don't re-tune)

```python
# src/pipeline/B_classify_optics.py
KNN_K               = 5
XI                  = 0.04
MINPTS              = 10
TAU_FLOOR_S         = 0.10

# src/pipeline/B_classify.py
TEMPO_THRESHOLDS    = (0.45, 0.61, 0.93, 1.08)
RUBATO_LO, RUBATO_HI = -0.0214169, +0.0184625
RUBATO_T_DIFF_S      = 10.0
ORNAMENT_T_DIFF_S    = 10.0

# src/pipeline/E_render_csv.py
SILENCE_CODE                = 98
DELTATIME_MISSING           = -1.0
SIMULTANEOUS_THRESHOLD_S    = 0.3
SEQUENCE_BREAK_S            = 60.0
```

### 8.3 Commands the future-you needs

```bash
# Re-run the whole pipeline:
python -m src.pipeline.D_run --refresh

# Re-run classifier only (skip C/E):
python -m src.pipeline.B_classify

# Just re-render the transcript:
python -m src.pipeline.C_render_readable

# Just re-emit the transformer CSV:
python -m src.pipeline.E_render_csv

# Stage 1 validation re-runs (write CSVs into data/diagnostics/hybrid_v1/):
python -m src.validation.loo_knn_tau
python -m src.validation.cluster_quality

# Tests:
python -m pytest tests/                  # 48 fast + 7 slow
python -m pytest tests/ -m "not slow"    # skip ELKI tests
```

### 8.4 Acceptance gates per remaining stage

```
Stage 1: kfold script runs on whale_dialogues.csv; outputs
         predict_results_unified.md skeleton.
Stage 2: bits/token within ±0.5 of prior 4.63; new vs old comparison
         table.
Stage 3: five prior findings re-evaluated + one Hersh-vs-Sharma
         comparison.
Stage 4: REPRODUCING_TRANSFORMER.md exists; old plans marked
         superseded; README pipeline sections updated.
```

**End of plan.**
