# Locked parameters for OPTICS-based coda classification

Frozen 2026-05-04 after Phase 1 + Phase 1b validation. Use these values in
`src/pipeline/B_classify.py`.

## OPTICS clustering (DSWP labelling)

| parameter | value | source |
|---|---|---|
| algorithm | OPTICSXi | Gero 2016 main text §2.2.4 |
| implementation | ELKI 0.7.1 (`elki-bundle-0.7.1.jar`) | de.lmu.ifi.dbs.elki on Maven Central |
| Java runtime | Adoptium Temurin JRE 8 | ELKI 0.7.x has a `URLClassLoader` cast incompatible with Java 9+ |
| `xi` | 0.04 | Gero 2016 main text |
| `minpts` | 10 | reverse-engineered Phase 1 (paper does not state) |
| distance | Euclidean | Gero 2016 main text |
| feature vector | absolute ICIs (n−1 dims for n-click coda) | Gero 2016 main text |
| length range | 3..10 clicks | Gero 2016 main text (excludes <3 and >10 as outliers) |
| bucketing | per-length, separate OPTICSXi run per click count | Gero 2016 main text |

CLI invocation (handled by `scripts/elki_optics.py`):

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
| `k` | 5 | passed Phase 1b at 97.64% loo |
| metric | Euclidean | matches the OPTICS distance |
| feature vector | absolute ICIs (n−1 dims, same as OPTICS) | matches Phase 1 |
| training set | DSWP rows with CodaType label, joined from `dswp_dominica_codas.csv` via `source_coda_id ↔ codaNUM2018` | only labelled subset |
| applied to | every non-DSWP coda in `codas_unified.csv` (Hersh Pacific + Sharma birth) | Phase 1b |

## Tempo bins (Sharma 2024 §4)

```python
TEMPO_THRESHOLDS = (0.45, 0.61, 0.93, 1.08)  # bin endpoints in seconds
# bins: 1 = <0.45s, 2 = [0.45, 0.61), 3 = [0.61, 0.93),
#       4 = [0.93, 1.08), 5 = ≥1.08s
```

## Rubato cutoffs (Sharma 2024 §5)

```python
RUBATO_LO = -0.021416925000000087   # 25th-percentile of duration-delta
RUBATO_HI =  0.018462550000000105   # 75th-percentile
RUBATO_T_DIFF_S = 10.0              # max gap between adjacent codas
# delta < LO  →  '\\'  (slowing)
# LO ≤ delta < HI  →  '-'  (steady)
# delta ≥ HI  →  '/'  (speeding)
```

## Ornament rule (Sharma 2024 §5)

Structural-on-click-counts rule: a coda has an ornament iff it has
exactly one more click than the immediately neighbouring same-whale
codas. **Requires** click-level timestamps and whale identity to
identify the "neighbouring" set. Apply to DSWP/birth corpus only;
**leave NaN for Hersh** since its data product doesn't carry per-coda
timestamps.

## Vendoring URLs

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

Both are gitignored (sizes: JRE 40 MB, ELKI jar 14 MB).
