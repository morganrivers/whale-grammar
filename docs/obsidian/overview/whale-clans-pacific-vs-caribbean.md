---
tags:
  - overview
  - corpus
summary: Why the unified V=467 overcounts whale repertoire — Hersh's Pacific recordings pool multiple unlabelled clans; the right Caribbean reference is V=218 (paper-spec compound + rubato, no chorus folded in)
created: 2026-05-06
updated: 2026-05-06
---

# Pacific vs Caribbean — and why V=467 isn't the right reference

The unified corpus has V=131 distinct rhythm classes (V=467 at the compound level including tempo, ornament, rubato). That number is bigger than what any single Caribbean recording produces — and most of the inflation comes from one source.

## The breakdown

Per-source rhythm-class diagnostics from the unified corpus:

| source | rows | distinct rhythm classes (V_local) | mean unique-types-per-sequence | modal share |
|---|---:|---:|---:|---:|
| hersh2022_pacific | 24,237 | **124** | 14.7 | 11.8% |
| sharma2024_dswp (Caribbean) | 8,872 | 62 | 3.0 | 40.5% |
| sharma2025_birth (Caribbean) | 5,731 | 51 | 18.6 | 75.6% |

64 of the 131 rhythm classes are **Hersh-only** (Pacific-specific repertoire not seen in Caribbean). Caribbean uses only 62 of 131; Pacific uses 124 of 131. Pacific's repertoire is roughly 2× the breadth of Caribbean's at the rhythm-class level.

## Hersh is multi-clan

[[overview/corpora|Hersh 2022]] aggregates Pacific recordings *across* sperm-whale vocal clans without per-clan labels. The literature documents 5–7 distinct Pacific clans, each with its own characteristic coda repertoire. So Hersh in our data product is "all Pacific codas pooled, clan identity unknown."

This shows up clearly in:

- **Higher per-sequence type diversity** (mean 14.7 unique types/seq for Hersh vs 3.0 for DSWP) — multiple whales / clans co-vocalizing in a single recording.
- **Flatter modal distribution** (Hersh modal 11.8% vs DSWP 40.5% vs Birth 75.6%) — when you pool distributions, the mix is flatter than any component.
- **64 Pacific-only rhythm classes** that the Sharma 2024 paper's 18-class taxonomy doesn't cover. The classifier discovered these via OPTICSXi clustering and named them as Pacific clusters in [[classifier/pacific-extension]].

## Why this matters for interpretation

Hersh's broader rhythm vocabulary doesn't necessarily mean Pacific whales have a "richer language." It mostly means we're looking at multiple clans pooled into one corpus. If we sliced Hersh by clan (which we can't — no labels), each individual clan would likely have a smaller, more peaked, more predictable repertoire — much like DSWP or Birth.

So **the multi-clan pooling makes whale look harder and more diverse than any single sperm-whale dialect actually is**.

## Pacific-only types contribute only 10.5% of Hersh tokens

| | hersh2022_pacific |
|---|---:|
| total tokens | 24,237 |
| tokens on rhythm types shared with Caribbean | 21,693 (89.5%) |
| tokens on Pacific-only rhythm types | 2,544 (10.5%) |

So 89% of Hersh activity is on rhythm classes that Caribbean also uses. The 64 Pacific-only types are the long tail — many rare classes from individual clans. **At the rhythm-class level, Pacific and Caribbean are mostly speaking the same language, with Pacific adding a clan-specific tail.**

The compound-level inflation (whale unified V=467) does *not* match this 10.5% pattern: the V=467 compound includes per-clan tempo/ornament/rubato variants of the same rhythm classes. Where rhythm shows 89% shared and 11% Pacific-only, compound shows much greater Pacific-only contribution because clan-specific tempo and rubato conventions multiply.

## What the right reference V is

For "what whale Caribbean exhibits" — the corpus the [[transformer/cross-language-units|Sharma 2024 paper]] analyzes — the right number is **V=218** (Compound + rubato, both DSWP and Birth, 9,532 tokens):

- DSWP only, with rubato: V=176
- Birth only, with rubato: V=157
- **DSWP + Birth combined, with rubato: V=218** ← paper-spec per-coda compound

V=467 (whale unified) is the multi-clan-pooled number. V=218 is the cleaner 2-clan Caribbean baseline.

**Note on chorus.** Earlier drafts of this vault folded chorus into the per-coda compound and reported V=355. That was wrong. **Chorus is per-utterance structural metadata** — "this group of codas was produced simultaneously by multiple whales" — analogous to punctuation marking utterance structure in CHILDES. It's already represented at the row level in `data/classified/whale_dialogues.csv` as the `Synchrony` column. The Sharma 2024 paper's compound is rhythm × tempo × rubato × ornament, *without* chorus folded in.

For most cross-corpus comparisons (matching CHILDES or designing a [[transformer/multilang-corpus-plan|multi-language counterpart]]), **target V≈218**, not V=467 (Pacific-pooled artifact) and not V=355 (chorus-folded conflation). Reaching V=467 would mean reproducing the multi-clan-pooling artifact; reaching V=355 would mean treating an utterance-level marker as a per-token feature.

## What this means for [[transformer/transformer|transformer training]]

The whale model trains on the unified V=467 corpus, which means it's training on ~3 pooled distributions (1 Pacific multi-clan + 1 DSWP + 1 Birth) without any clan label. **Per-source val_coda_bpt would likely show**:

- Hersh val_coda ≫ DSWP/Birth val_coda — because Hersh is internally heterogeneous
- DSWP/Birth val_coda close to each other and to the headline number

This is an open follow-up flagged in `outputs/grammar/HANDOFF.md:125`. Would tell us how much of the model's apparent compressibility is being driven by the easy peaked Caribbean fraction vs the harder Pacific-pooled fraction.

## Related

- [[overview/corpora]] — the three sources and their per-feature populations.
- [[classifier/pacific-extension]] — how the 64 Pacific-only rhythm classes were discovered.
- [[transformer/cross-language-units]] — natural-language sub-lexical units at matched V.
- [[transformer/multilang-corpus-plan]] — uses V=218 as the target (not V=467 nor V=355).
