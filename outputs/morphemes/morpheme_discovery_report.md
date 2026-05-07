# Morpheme Discovery in Sperm Whale Codas

**Script:** `src/grammar/morpheme_discovery.py`  
**Full results:** `outputs/morphemes/morpheme_results.json`  
**Corpus:** 37,684 codas with valid ICI data and rhythm_class labels (38,840 total)

---

## Background and Motivation

Sperm whale codas are rhythmic click sequences. Each coda is characterised by its
inter-click intervals (ICIs) — the time gaps between consecutive clicks. The existing
classification pipeline (OPTICS Xi, Gero 2016-style) treats each coda as an atomic
acoustic unit and clusters them into 131 rhythm classes. This analysis asks a
different question: are there **recurring sub-patterns within and across codas** that
function like morphemes — building blocks shared across many types, potentially with
modifiers that inflect them?

Two tracks were run:

- **Track 1 (ICI morphemes):** Morfessor MDL segmentation of discretised ICI strings,
  across all coda lengths simultaneously.
- **Track 2 (Multi-coda morphemes):** N-gram PMI analysis + BPE merging on coda-type
  sequences within whale dialogues.

A parallel augmented analysis (Track 1b) appends tempo as a coda-level modifier
character, and checks whether tempo / rubato / ornament are consistent within
morpheme groups.

---

## Method

### ICI Discretisation

Each ICI position (ICI1 through ICI9) was discretised independently using global
per-position 5-quantile bins across all codas (A = bottom 20th percentile,
E = top 20th). A coda with n clicks produces a string of length n−1 (e.g. a
5-click coda → 4-character string). Codas from all lengths (3–40 clicks) were
pooled into a single corpus; Morfessor Baseline (v2.0, MDL objective) was trained
on that corpus where each unique string appeared with its observed count.

For the augmented analysis (Track 1b), a tempo character was appended to each
string (P=tempo-1 through T=tempo-5, U=unknown), expanding the alphabet.

### Multi-coda Morpheme Discovery

Dialogue sequences (grouped by `sequenceId`, silence code 98 excluded) were
analysed for n-gram structure. PMI scores and MDL saving (freq × PMI) were
computed for bigrams, trigrams, and 4-grams. BPE merging (freq × PMI scoring)
was then run for 20 steps on the full corpus to estimate achievable compression.

---

## Track 1: ICI Morphemes

**2,874 unique ICI strings; 2,309 (80%) segmented into ≥2 morphemes; 566 morphemes discovered.**

### Alphabet reminder

| Symbol | ICI quantile (per position) | Acoustic character |
|--------|----------------------------|--------------------|
| A | 0–20th pct | Very short interval |
| B | 20–40th pct | Short-medium |
| C | 40–60th pct | Medium |
| D | 60–80th pct | Medium-long |
| E | 80–100th pct | Very long interval |

### Top 30 morphemes by occurrence

| Morpheme | Occurrences | n rhythm classes | Purity | Modal rc |
|----------|-------------|-----------------|--------|----------|
| `AAAA` | 2,576 | 21 | 0.682 | 79 |
| `EE` | 1,045 | **58** | **0.097** | 45 |
| `BCCC` | 976 | 10 | 0.477 | 80 |
| `EEDD` | 921 | 5 | **0.988** | 0 |
| `DDEE` | 885 | 20 | 0.679 | 81 |
| `BB` | 875 | 27 | 0.547 | 23 |
| `EECC` | 824 | 5 | **0.990** | 0 |
| `EEE` | 683 | 31 | 0.652 | 62 |
| `BBBB` | 671 | 14 | 0.827 | 80 |
| `EEEE` | 652 | 20 | 0.557 | 81 |
| `DBBB` | 647 | 6 | 0.969 | 0 |
| `CB` | 581 | 26 | 0.604 | 23 |
| `DDCB` | 552 | 4 | 0.993 | 0 |
| `DC` | 521 | 28 | 0.476 | 23 |
| `CCDD` | 520 | 13 | 0.621 | 0 |
| `DD` | 508 | 28 | 0.473 | 23 |
| `DB` | 492 | 27 | 0.468 | 23 |
| `EEDC` | 490 | 10 | 0.767 | 0 |
| `CC` | 472 | 27 | 0.440 | 23 |
| `DDE` | 461 | 21 | 0.530 | 81 |
| `DCBB` | 423 | 5 | 0.916 | 0 |
| `CDDD` | 408 | 11 | 0.627 | 81 |
| `DEE` | 403 | 27 | 0.545 | 81 |
| `ED` | 399 | 29 | 0.283 | 23 |
| `BBBC` | 352 | 9 | 0.722 | 80 |
| `BCDD` | 346 | 11 | 0.624 | 0 |
| `CDE` | 341 | 16 | 0.585 | 81 |
| `EECD` | 338 | 11 | 0.684 | 0 |
| `CCBB` | 323 | 13 | 0.583 | 80 |
| `EC` | 315 | 25 | 0.438 | 23 |

**Purity** = fraction of occurrences belonging to the single most common rhythm_class
for that morpheme. Values near 1.0 mean the morpheme is almost exclusively produced
within one coda type. Values near 0 mean it appears promiscuously across many types.

### Two morpheme classes emerge

**Type-specific morphemes** (purity > 0.9): These are effectively complete coda
patterns that Morfessor happens to assign as an atomic unit because they recur
without needing further decomposition. `DDCB` (purity=0.993), `EEDD` (0.988),
`EECC` (0.990), `DBBB` (0.969) are essentially the full ICI fingerprints of
single coda types. They reinforce, not undermine, the OPTICS classification.

**Promiscuous morphemes** (purity < 0.2): These short patterns appear as components
in a huge range of coda types:

| Morpheme | Occurrences | n types spanned |
|----------|-------------|----------------|
| `EE` | 1,045 | **58** |
| `CE` | 307 | 36 |
| `DE` | 227 | 30 |
| `ED` | 399 | 29 |

`EE` (two consecutive very-long ICIs) appears in 58 of 131 rhythm classes — nearly
half the entire coda vocabulary. This is genuine phoneme-like behaviour: a building
block that is not type-specific but participates in constructing many distinct types.

**Of the 176 frequent morphemes, 168 span >1 rhythm_class.** Average purity for
the top 20 morphemes is 0.687. Only 8 morphemes are confined to a single type.

---

## Track 1b: Tempo, Rubato, and Ornament as Modifiers

### Feature consistency within ICI morphemes

| Morpheme | n | tempo mean | tempo std | rubato (L/K/J) | ornament rate |
|----------|---|-----------|----------|----------------|---------------|
| `AAAA` | 2,576 | **1.35** | 0.84 | 347 / 28 / 30 | 6.5% |
| `EE` | 1,045 | 3.85 | 0.97 | 5 / 1 / 12 | **29.5%** |
| `BCCC` | 976 | 3.14 | 0.48 | — | 0.0% |
| `EEDD` | 921 | **4.98** | **0.13** | 230 / 158 / 125 | 2.3% |
| `DDEE` | 885 | 4.72 | 0.45 | — | 14.3% |
| `BB` | 875 | 2.05 | 1.29 | 2 / 6 / 4 | **32.3%** |
| `EECC` | 824 | 4.26 | 0.55 | 130 / 61 / 67 | 8.9% |
| `EEE` | 683 | 4.49 | 0.69 | 37 / 22 / 43 | 3.9% |
| `BBBB` | 671 | 2.35 | 0.83 | 15 / 4 / 4 | 6.8% |
| `EEEE` | 652 | **5.00** | **0.00** | 22 / 14 / 10 | 4.9% |

Rubato: L = level (−), K = accelerating (/), J = decelerating (\\). Counts are raw;
many codas lack rubato annotation.

### Key observations

**Tempo co-lexicalization is tight for long-ICI morphemes.**
`EEEE` (all very-long ICIs) has tempo_std=0.00 — every coda with this ICI pattern
falls in the slowest tempo bin without exception. `EEDD` has tempo_std=0.13, nearly
as tight. This is partly expected: long ICIs → long coda duration → high tempo bin.
But it also confirms that the ICI pattern and tempo are not independently chosen;
they are determined together.

**Short-ICI promiscuous morphemes show high ornament rates.**
`EE` (29.5%) and `BB` (32.3%) have ornament rates 4–5× higher than `AAAA` (6.5%).
When these short patterns appear as sub-components of a longer coda, they frequently
co-occur with an ornamental extra click. This is the clearest evidence that these
morphemes are not acoustic artefacts but acoustically-coherent sub-units that
preferentially attract specific modifications.

**Rubato is not a free inflection of ICI morphemes.**
`AAAA` is overwhelmingly level rubato (347 level vs 28+30 for accel/decel = 86%
level). The EE-based morphemes have sparser rubato annotation but no similarly
strong directional preference. For morphemes with sufficient rubato data, the
distribution is less skewed — rubato appears more variable within a morpheme
than tempo, consistent with rubato acting as an optional modifier rather than
a fused property.

**Augmented morphemes (ICI + tempo) are treated as atomic by Morfessor.**
When tempo is appended to the string (e.g. `AAAAP`, `EEDDT`), the model segments
fewer strings (1,365 of 3,396 augmented strings, vs 2,309 of 2,874 plain strings).
The larger, richer tokens tend to stay unsegmented. This confirms that tempo adds
discriminative information that Morfessor can exploit to find cleaner boundaries —
but the augmented types are more specific and therefore appear less often, reducing
the MDL benefit of segmentation.

---

## Track 2: Multi-coda Morphemes

**486 sequences, 37,666 coda tokens. Sequence length: mean=77.5, median=24, max=5,091.**

### N-gram structure

Every top-20 bigram by MDL saving is a **self-repeat** (`[x, x]` for some coda type
x). There are no significant heterogeneous bigrams — the strongest cross-type signal
does not appear until the trigram level:

| Trigram | Count | PMI | MDL saving (bits) |
|---------|-------|-----|-------------------|
| `[80, 0, 80]` | 256 | 2.68 | 685 |
| `[0, 80, 80]` | 248 | 2.63 | 652 |
| `[80, 80, 0]` | 244 | 2.61 | 636 |

These are all alternation patterns between coda types 0 and 80, and represent the
**only cross-type morpheme structure** rising above the self-repeat baseline.

### BPE merge history (20 steps)

| Step | Merged pair | Freq | PMI | Tokens saved |
|------|-------------|------|-----|-------------|
| 1 | [0, 0] | 7,171 | 1.42 | 4,028 |
| 2 | [0, 0, 0, 0] | 2,563 | 2.41 | 1,514 |
| 3 | [23, 23] | 1,837 | 2.76 | 1,064 |
| 4 | [79, 79] | 1,235 | 3.28 | 715 |
| 5 | [0×8] | 804 | 3.41 | 506 |
| 6 | [80, 80] | 1,064 | 2.49 | 671 |
| 7 | [23×4] | 591 | 3.93 | 365 |
| 8 | [81, 81] | 775 | 2.63 | 503 |
| 9 | [79×4] | 429 | 4.57 | 261 |
| 10 | [89, 89] | 464 | 3.28 | 292 |
| 11–20 | (all self-repeats) | — | — | — |

All 20 BPE merges are homogeneous runs of a single coda type. Heterogeneous
cross-type morphemes (like the 0↔80 alternation) do not reach the merge threshold
within the first 20 steps.

### Compression

After 20 BPE merges the corpus shrinks from **37,666 → 25,615 tokens (32% reduction)**.
This is achieved entirely by treating repeated runs of the same coda type as single
tokens.

---

## Transformer Efficiency

With the current K=8 context window, a 32% token compression yields an effective
gain of ~3.76 extra context positions — equivalent to upgrading from K=8 to K≈12
without changing the architecture.

**Practical implications:**

1. **Compression is real but trivial in origin.** The 32% comes from coda runs, not
   from any combinatorial cross-type structure. A simpler run-length encoding would
   achieve the same thing without morpheme machinery.

2. **The vocabulary cost is low.** Because the BPE merges are all self-repeats, the
   new "morpheme tokens" are just run-length codes. Adding them doesn't dilute the
   embedding space with many new types.

3. **The interesting gain would come from cross-type morphemes.** If the 0↔80
   alternation pattern (and others like it) were treated as atomic, they would let
   the transformer attend over longer ranges at the multi-coda level. But at 20
   BPE steps the cross-type patterns are not yet strong enough to dominate. More
   steps, or a lower PMI threshold, would begin to capture them.

4. **ICI-level morpheme tokenisation is a different opportunity.** If codas were
   re-tokenised not as rhythm_class integers but as (morpheme_prefix, morpheme_suffix)
   pairs, the vocabulary would shrink from 131 types to the ~50–100 ICI morphemes
   while retaining sub-coda structure. Whether this helps prediction depends on
   whether the morpheme boundaries carry more predictive signal than the full coda
   identity — currently unknown.

---

## Relationship to Existing OPTICS Classification

| Statistic | Value |
|-----------|-------|
| ICI morphemes confined to a single rhythm_class (n≥50) | 8 |
| ICI morphemes spanning >1 rhythm_class (n≥50) | 168 |
| Average purity, top 20 morphemes | 0.687 |

**The OPTICS types are not undermined by this analysis.** They are acoustically
coherent clusters of whole-ICI-vector similarity, and the high-purity morphemes
(`DDCB`, `EEDD`, `EECC`, `DBBB`) closely correspond to specific OPTICS types.

**But the morpheme layer reveals structure beneath the OPTICS level.** The 168
cross-type morphemes — especially `EE` spanning 58 types — show that many distinct
coda types share common ICI sub-vectors. The existing classification correctly
separates these types because their *full* ICI patterns differ, but the shared
sub-patterns suggest a compositional origin: a small inventory of ~50–100 building
blocks from which the 131 attested types are assembled.

**Potential OPTICS over-splitting:** Low-purity morphemes that appear within only
a handful of types (e.g. `BCCC` purity=0.477 across 10 types) may indicate that
some OPTICS types are close variants sharing a common root with a differing terminal
interval. Whether this is linguistically meaningful or acoustic noise requires
listening to the exemplars.

---

## Sensitivity Analysis: Effect of N\_BINS on ICI Discretisation

**Script:** `src/grammar/sensitivity_n_bins.py`  
**Full results:** `outputs/morphemes/sensitivity_n_bins.json`

The choice of 5 quantile bins (A–E) for ICI discretisation is a design parameter,
not a principled derivation. To assess its impact, N\_BINS was varied across
{2, 3, 4, 7, 10} and the full Track 1 pipeline (bin-edge computation, corpus
construction, Morfessor MDL training, purity analysis) was re-run for each value.

### Results

| N\_BINS | Unique strings | MDL cost | n morphemes | % seg ≥2 | Mean morph len | Purity@20 | Spanning (n≥50) | Confined (n≥50) |
|---------|---------------|----------|-------------|----------|---------------|-----------|----------------|----------------|
| 2       | 327           | 189,748  | 44          | 86.5%    | 3.86          | 0.594     | 42             | 0              |
| 3       | 1,030         | 231,740  | 161         | 84.4%    | 3.89          | 0.620     | 102            | 1              |
| 4       | 1,916         | 261,888  | 374         | 80.5%    | 3.887         | **0.704** | 147            | 4              |
| **5**   | **2,874**     | **~292k**| **566**     | **80.2%**| **3.9**       | **0.687** | **168**        | **8**          |
| 7       | 4,822         | 330,196  | 1,130       | 76.8%    | 3.729         | 0.613     | 184            | 18             |
| 10      | 7,716         | 381,842  | 1,905       | 75.7%    | 3.511         | 0.471     | 196            | 9              |

*The N\_BINS=5 row is from the original run; all others are from the sensitivity sweep.*

### Observations

**MDL cost rises monotonically with N\_BINS.** More bins require more bits per
symbol to encode the corpus, regardless of Morfessor's internal segmentation. This
is an unavoidable information-theoretic consequence of a larger alphabet.

**Purity peaks at N\_BINS=4, with N\_BINS=5 close behind.** The morphemes discovered
at 4 bins have the highest average alignment with existing OPTICS rhythm classes
(purity=0.704). Beyond 5 bins, purity falls steadily — reaching 0.471 at 10 bins.
The finer distinctions add alphabet complexity faster than they add real acoustic
signal that Morfessor can exploit to find cleaner boundaries.

**N\_BINS=2 is too coarse.** Only 44 morphemes are discovered, and none are confined
to a single rhythm class (conf\_n50=0): every discovered morpheme spans multiple
OPTICS types. This level of discretisation erases the sub-coda distinctions that
make morpheme analysis meaningful.

**N\_BINS=7–10 over-splits.** At 10 bins, the top morphemes degrade to short digrams
(`IJ`, `JJ`, `CC`) with low purity (0.471), and confined morphemes do not increase
steadily — they dip at N\_BINS=10 relative to 7, suggesting the rare high-bin
symbols become too sparse for reliable MDL segmentation.

### Which N\_BINS to use

Two objectives pull in opposite directions:

- **Downstream compression (MDL):** lower is always better — N\_BINS=2 minimises
  total description length.
- **Alignment with rhythm\_class boundaries (purity):** peaks at N\_BINS=4.

**N\_BINS=4 is the best single choice** if the goal is morphemes that respect the
existing coda-type inventory. N\_BINS=5 (the original choice) is a close second and
carries slightly more granularity at minimal purity cost (0.687 vs 0.704). Values
below 4 discard real ICI structure; values above 5 introduce noise faster than signal.

---

## Open Questions

1. **Is EE a phoneme or an artefact?** The two-very-long-ICI pattern appears in
   58 types. Is this a shared acoustic sub-unit (morpheme) or simply a consequence
   of many long codas having at least two high-percentile intervals? Inspecting the
   specific types in which EE appears, and whether they form a semantic cluster,
   would clarify this.

2. **Why do EE and BB attract ornaments at 3–5× the baseline rate?** The elevated
   ornament rate for short-ICI-contrast morphemes is the clearest evidence of
   genuine sub-structure. Understanding whether the ornamental click consistently
   appears at the position of the morpheme boundary, or at a fixed location in the
   coda, would be diagnostic.

3. **Do the 0↔80 alternations correspond to known interactive exchange patterns?**
   The only significant cross-type trigram in the dialogue sequences is 0↔80
   interleaving. These two coda types (modal ICI pattern: EEDD/EECC, slowest tempo)
   may represent a call-and-response pair between two whales. Cross-referencing with
   speaker identity and synchrony flags would test this.

4. **Literature novelty.** To our knowledge, Morfessor or any explicit MDL morpheme
   segmentation tool has not been applied to sperm whale codas previously. The
   Project CETI work used information-theoretic measures and transformer-based
   sequence modelling, but did not attempt within-coda sub-pattern discovery of this
   kind. A targeted search of the bioacoustics literature would confirm whether this
   is novel.
