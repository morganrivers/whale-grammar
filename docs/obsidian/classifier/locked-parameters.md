---
tags:
  - classifier
  - reference
summary: Frozen constants for OPTICSxi, kNN, tempo, rubato, ornament — plus ELKI/JRE vendoring URLs
created: 2026-05-06
updated: 2026-05-06
---

# Locked parameters

Every constant in the classifier pipeline, its source, and where in code it lives. Locked 2026-05-04 after Phase 1 + Phase 1b validation. Don't re-tune without a written hypothesis.

## OPTICSxi (DSWP-only first pass + residual-pool discovery pass)

| parameter | value | source |
|---|---|---|
| algorithm | OPTICSXi | Gero 2016 main text §2.2.4 |
| implementation | ELKI 0.7.1 (`elki-bundle-0.7.1.jar`) | de.lmu.ifi.dbs.elki on Maven Central |
| Java runtime | Adoptium Temurin **JRE 8** | ELKI 0.7.x has a `URLClassLoader` cast incompatible with Java 9+ |
| `xi` | 0.04 | Gero 2016 main text |
| `minpts` | 10 | reverse-engineered ([[classifier/gero-2016-replication|Phase 1]]); paper does not state |
| distance | Euclidean | Gero 2016 main text |
| feature vector | absolute ICIs (n−1 dims for n-click coda) | Gero 2016 main text |
| length range | 3..10 clicks | Gero 2016 main text (excludes <3 and >10 as outliers) |
| bucketing | per-length, separate OPTICSXi run per click count | Gero 2016 main text |

In code: `src/pipeline/B_classify_optics.py`

```python
LENGTH_RANGE = range(3, 11)
XI = 0.04
MINPTS = 10
```

CLI invocation (handled by `src/validation/elki_optics.py`):

```bash
java -jar vendor/elki/elki-bundle-0.7.1.jar KDDCLIApplication \
    -dbc.in   <ici_vectors.txt> \
    -algorithm        clustering.optics.OPTICSXi \
    -opticsxi.xi      0.04 \
    -optics.minpts    10 \
    -resulthandler    ClusteringVectorDumper \
    -clustering.output <labels.txt>
```

## kNN cross-corpus propagation

| parameter | value | source |
|---|---|---|
| classifier | sklearn `KNeighborsClassifier` | Phase 1b option 2 |
| `k` | 5 | passed Phase 1b at 97.64 % loo |
| metric | Euclidean | matches the OPTICS distance |
| feature vector | absolute ICIs (n−1 dims, same as OPTICS) | matches Phase 1 |
| training set | DSWP rows with non-NOISE `CodaType`, joined from `dswp_dominica_codas.csv` via `source_coda_id ↔ codaNUM2018` | Sharma 'real' anchor |
| applied to | every non-DSWP coda in `codas_unified.csv` | Phase 1b |

In code: `src/pipeline/B_classify_optics.py`

```python
KNN_K = 5
```

## τ — per-type accept threshold

| parameter | value | source |
|---|---|---|
| `τ(length, codatype)` | `max(NOSC_p99, TAU_FLOOR_S)` | hybrid pipeline §τ |
| `TAU_FLOOR_S` | 0.10 s | half the median inter-CodaType centroid distance at n=5 |
| NOSC | nearest-of-same-class distance, computed on Sharma 'real' rows | derived inline |

In code: `src/pipeline/B_classify_optics.py`

```python
TAU_FLOOR_S = 0.10
RADIUS_FLOOR_S = TAU_FLOOR_S  # back-compat alias
PACIFIC_NOISE_CLUSTER_ID = 0  # ELKI's "rest" bucket convention
```

## Tempo bins (Sharma 2024 §4)

KDE on `coda_duration_s` with bandwidth `h=0.035` gives 5 peaks at [0.33, 0.51, 0.80, 1.02, 1.26] s; the bin endpoints sit at the valleys between them.

```python
# src/pipeline/B_classify.py
TEMPO_THRESHOLDS = (0.45, 0.61, 0.93, 1.08)
# bins: 1 = <0.45 s, 2 = [0.45, 0.61), 3 = [0.61, 0.93),
#       4 = [0.93, 1.08), 5 = ≥ 1.08 s
```

## Rubato cutoffs (Sharma 2024 §5)

Empirical 25th/75th-percentile cutoffs on labelled DSWP duration deltas. Source: `sw-combinatoriality/code/generate_whale_dialogue_txt_with_proper_timings.py`.

```python
# src/pipeline/B_classify.py
RUBATO_LO = -0.021416925000000087  # 25th-percentile of Δduration
RUBATO_HI =  0.018462550000000105  # 75th-percentile
RUBATO_T_DIFF_S = 10.0             # max gap between adjacent same-whale codas
# delta < LO  →  '\\'  (slowing)
# LO ≤ delta < HI  →  '-'  (steady)
# delta ≥ HI  →  '/'  (speeding)
```

## Ornament rule (Sharma 2024 §5)

Structural rule on click counts: a coda has an ornament iff *at least one* immediately neighbouring same-whale coda within `ORNAMENT_T_DIFF_S` seconds has exactly `n_clicks − 1`. Requires per-coda timestamps and whale identity to identify the "neighbouring" set.

```python
# src/pipeline/B_classify.py
ORNAMENT_T_DIFF_S = 10.0
```

Apply to DSWP / birth corpus only; **leave NA for Hersh** since its data product doesn't carry per-coda timestamps. See `feedback_ornament_scope.md` in auto-memory.

## Synchrony / sequence / rendering constants (CSV stage)

```python
# src/pipeline/E_render_csv.py
SILENCE_CODE = 98                    # "no coda" sentinel
DELTATIME_MISSING = -1.0             # Hersh has no timestamps
SIMULTANEOUS_THRESHOLD_S = 0.3       # second whale within 0.3 s → Synchrony=1
WHALE_UNKNOWN = "UNK"                # Hersh has no per-coda whale ID
```

`E_render_csv` does **not** sub-split sequences — one sequenceId per `(source, recording_id)`. Long quiet pauses surface as large `TimeDelta` values. Earlier versions sub-split on > 60 s gaps; that was dropped to give the model honest "this is one conversation" framing.

## Vendoring URLs

ELKI and the JRE are gitignored due to size (40 MB + 14 MB). To install:

```bash
# JRE 8 (Adoptium Temurin)
curl -fsSL \
  "https://api.adoptium.net/v3/binary/latest/8/ga/linux/x64/jre/hotspot/normal/eclipse?project=jdk" \
  -o vendor/jre8.tar.gz
tar -xzf vendor/jre8.tar.gz -C vendor/
ln -s vendor/jdk8u482-b08-jre vendor/jre8

# ELKI 0.7.1 bundle
curl -fsSL \
  "https://repo1.maven.org/maven2/de/lmu/ifi/dbs/elki/elki-bundle/0.7.1/elki-bundle-0.7.1.jar" \
  -o vendor/elki/elki-bundle-0.7.1.jar
```

The macOS / Windows binaries follow the same Adoptium URL scheme — substitute `mac/x64/jre` or `windows/x64/jre`.

## Don't re-litigate

These have been settled by Phase 1 / 1b / hybrid implementation:

- **ELKI 0.7.1 vs scikit-learn for OPTICSxi.** ELKI is the answer; sklearn over-fragments. See [[classifier/papers]] §scikit-learn.
- **`xi=0.04`.** Locked. Gero's published value.
- **`minpts=10` for the first OPTICS pass on DSWP.** Locked. Reverse-engineered, sharp peak with all neighbours ~17 pp lower.
- **Per-length bucketing.** Locked. Gero's method.
- **3–10 click range.** Locked. Gero's method excludes <3 and >10.
- **Direct join for DSWP CodaType labels.** 100 % accurate by construction.
- **k=5 for kNN cross-corpus propagation.** Validated at 97.64 % LOO.
- **Sharma 2024 tempo bins (0.45, 0.61, 0.93, 1.08).** Locked.
- **Sharma 2024 rubato cutoffs.** Locked (the empirical 25/75 percentiles).
- **Ornament rule applies to DSWP+birth only, NA for Hersh.** Locked by the structural argument: Hersh has no per-coda timestamps.
