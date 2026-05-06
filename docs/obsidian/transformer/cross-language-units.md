---
tags:
  - transformer
  - childes
  - comparison
summary: Sub-lexical unit comparison — whale Dominica compound (V=218, paper-spec) vs Japanese moras vs Mandarin syllables vs English phonemes at matched token budget
created: 2026-05-06
updated: 2026-05-06
---

# Cross-language sub-lexical units

The [[transformer/childes-comparison|original CHILDES baseline]] used English **lemmas** (V≈1605 uncapped, V=467 [[transformer/vocab-cap-experiment|capped]]). But a lemma is a much higher-level unit than a whale coda: words are lexical, codas are sub-lexical (closer to syllables in granularity). The fair comparison needs sub-lexical units on the human side too.

This page summarizes what we measured across English / Mandarin / Japanese CHILDES at multiple sub-lexical levels, and how each compares to the [[overview/glossary|whale Dominica compound]].

## Whale reference numbers (Sharma 2024 + 2025 Dominica dialogues)

From `whale_dialogues.txt` and `whale_dialogues_birth.txt` in the sw-combinatoriality checkout (paper Author's data):

| compound spec | N | V | modal | tokens/type |
|---|---:|---:|---:|---:|
| Rhythm × Tempo × Ornament (no rubato) | 9,532 | 98 | 34% | 39 |
| **Rhythm × Tempo × Rubato × Ornament** (paper-spec) | 9,532 | **218** | 14% | 44 |

This is the *Caribbean only* corpus — DSWP (Sharma 2024) + Birth (Sharma 2025), both from Dominica. The [[overview/corpora|whale unified V=467]] is inflated by Hersh's 116 Pacific-only rhythm classes; **V=218 is the right reference for "what whale Caribbean actually exhibits"** at the per-coda granularity.

**Chorus is not in V.** Chorus is per-utterance structural metadata ("this group of codas was produced simultaneously"), already represented as the row-level `Synchrony` column in the rendered CSV. It's analogous to punctuation, not to a token feature — the paper's compound is rhythm × tempo × rubato × ornament without chorus folded in.

## The three CHILDES sources we now have

| corpus | size | natural sub-lexical unit |
|---|---:|---|
| Eng-UK CHILDES | 3,038 .cha files, ~11M lemma tokens | ARPABET phoneme (V≈44, requires `g2p-en`) |
| Mandarin CHILDES | 3,752 .cha files, ~1.7M lemma tokens (with tones in pinyin lemma) | initial × final × tone (full tonal syllable) |
| Japanese MiiPro | 174 .cha files (3 children), ~868k word tokens, ~2.1M moras | mora (CV unit, no tone) |

The Mandarin and Japanese corpora live in-tree at `Mandarin/` and `Japanese/`. English is at `Eng-UK/`.

## Bottom-10% TTR matched-budget table

Each language filtered to its bottom-10%-TTR conversations (most repetitive / simplest), random-subsampled to a similar token count on the natural unit. Whale stays full-corpus.

| corpus | unit | N | V | modal | tokens/type | p90 rank |
|---|---|---:|---:|---:|---:|---:|
| **Whale Dominica** (paper-spec) | Compound + rubato | 9,532 | **218** | 14% | 44 | 51 |
| Whale Dominica (compound + rubato, with chorus marker appended) | rubato compound × chorus flag | 9,532 | 355 | 8% | 27 | 96 |
| Whale Dominica | Compound r×t×o (no rubato) | 9,532 | 98 | 34% | 39 | 19 |
| **Japanese (MiiPro)** | Moras | 53,166 | **114** | 7% | 466 | 46 |
| Japanese (matched N) | Moras | 9,532 | **105** | 7% | 91 | 45 |
| Mandarin (low-TTR) | full Syllable × Tone | 46,091 | 778 | 4.5% | 59 | 251 |
| Mandarin (low-TTR) | Initial × Tone | 46,091 | 91 | 8% | 506 | 47 |
| Mandarin (low-TTR) | Final × Tone | 46,091 | 209 | 8% | 220 | 75 |
| English (low-TTR) | lemmas | 42,921 | 1,228 | 7% | 35 | 285 |
| English | phonemes (theoretical) | — | 44 | — | — | — |

## Granularity hierarchy

```
English phonemes (44)
   ↓
Whale rhythm alone (18)
   ↓
Mandarin Initial × Tone (91) ≈ Whale compound r×t×o (98) ≈ Japanese moras (105–114)
   ↓
Mandarin Final × Tone (209) ≈ Whale compound + rubato (218) ← paper-spec
   ↓
Mandarin tonal syllables full (778)
   ↓
English / Japanese syllables / words (10,000+)
```

## The closest natural-language matches

**At V≈100 (whale compound without rubato):** Japanese moras and Mandarin Initial × Tone both land here. Japanese moras are a *natural* unit (every child internalizes the hiragana mora inventory); Mandarin Initial × Tone is a constructed slice (drops the final, which is unnatural). For coda-equivalence at this V, Japanese is the cleaner match.

**At V≈200 (whale compound + rubato):** Mandarin Final × Tone (V=209) lands here. Final × Tone is *also* a constructed slice (drops the initial), so neither Mandarin slicing at V≈100 nor V≈200 is fully natural. Japanese moras at the matched token count give V≈105, undershooting V=218 by a factor of 2.

**At V≈218 (whale paper-spec compound + rubato):** Mandarin Final × Tone (V=209) lands here naturally. The [[transformer/multilang-corpus-plan|multi-language plan]] reaches V≈218 by combining Japanese moras + English phonemes + a small share of Mandarin full tonal syllable (mostly disjoint inventories) — Mandarin's share is dialed down so V_zh contributes ~70 to the total.

## The distribution-shape gap

Even when V matches, the *shape* of the rank-frequency distribution does not. From the same matched-budget table:

| | whale compound + rubato | Japanese moras (matched N) |
|---|---:|---:|
| V | 218 | 105 |
| modal | 14% | 7% |
| p90 rank | 51 | 45 |

Whale is roughly 2× more peaked. **Whale codas reuse one canonical form much more than human children reuse any single phonological unit**, even on size-matched, simplicity-matched samples. This is the deepest cross-system asymmetry the data shows.

## Why V≈100 is "the natural sub-lexical unit"

There's a phonological inventory size that recurs across systems:

- Japanese moras: 110 (the hiragana inventory).
- Mandarin tone-bearing units (Initial×Tone or final ×Tone): 91–209.
- Whale rhythm × tempo × ornament: 98.
- Whale rhythm × tempo × rubato × ornament: 218.

Languages and whale codas seem to converge on a sub-lexical unit count in the **100–250 range**. Below that and you're at the phoneme level (too coarse to carry meaning); above that and you're at the syllable / morpheme level (already lexical). It's a real cardinality band, not a coincidence.

## Tooling

`src/grammar/corpus_stats.py` is the script that produces all these numbers. Five sources:

- `--source dominica` — whale paper data (point at sw-combinatoriality checkout)
- `--source whale-unified` — `data/classified/whale_dialogues.csv`
- `--source childes-en`, `--source childes-zh`, `--source childes-jp`

Three filtering knobs:

- `--bottom-ttr-pct 0.10` — keep simplest-speech conversations
- `--subsample 39000` — random subsample to matched token count
- `--top N` — head types to print per dimension

Per-dimension parses (Mandarin: initial/final/tone, plus all crosses; Japanese: mora segmentation from romaji `%ort:` lines; Dominica: rubato parsed from `whale_dialogues.txt` per-token markers; chorus parsed from per-line "In chorus, whales …" prefixes — kept as a separate stream alongside the per-token compound, not folded into V).

## Reproduce

```bash
# Pure single-source stats
python -m src.grammar.corpus_stats --source childes-jp --top 5

# Matched-budget all-source comparison
python -m src.grammar.corpus_stats --all \
    --dominica-csv /path/to/sw-combinatoriality/data/sperm-whale-dialogues_augmented.csv \
    --bottom-ttr-pct 0.10 --subsample 39000 --top 0
```

## Related

- [[transformer/childes-comparison]] — original lemma-level English comparison (V=1605).
- [[transformer/vocab-cap-experiment]] — V=467 cap reduces the entropy-floor confound.
- [[transformer/multilang-corpus-plan]] — proposed implementation: combine all three languages into a single 39k-token training corpus matching whale shape.
- [[overview/whale-clans-pacific-vs-caribbean]] — why Caribbean (V=218, no chorus folded in) is the right reference, not unified (V=467).
