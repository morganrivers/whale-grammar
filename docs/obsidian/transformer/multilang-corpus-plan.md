---
tags:
  - transformer
  - childes
  - plan
summary: Pointer to docs/multilang_corpus_plan.md — multi-language CHILDES counterpart to whale_dialogues.csv (English phonemes + Japanese moras + Mandarin tonal syllables, with Hersh-style missingness, V≈218)
created: 2026-05-06
updated: 2026-05-06
---

# Multi-language CHILDES counterpart — implementation plan

The detailed implementation plan lives at `docs/multilang_corpus_plan.md` (outside the obsidian vault, at the repo `docs/` root). This page is an index entry summarizing what that plan does and what was decided.

## Goal

Build a 39k-token human-language corpus that **structurally mirrors `data/classified/whale_dialogues.csv`** so the M7 transformer trains on something with similar V, similar tokens-per-type, similar missingness pattern, and similar sub-lexical granularity. Then re-run [[interp/interp]] side-by-side and see whether the per-layer logit-lens decomposition closes the whale ↔ CHILDES gap.

## Decisions locked in

From the discussion that produced the plan:

| design choice | decision |
|---|---|
| English unit | ARPABET phoneme (V≈44) via `g2p-en` |
| Japanese unit | mora (V≈105) via existing `corpus_stats.segment_moras` |
| Mandarin unit | **full tonal syllable** = Initial + Final + Tone (V≈70–120 at the planned share) |
| Hersh-tier mix | EN + JP + ZH, **Mandarin is the elastic knob** to tune V toward 218 |
| Hersh-tier fraction | 0.62 of total tokens |
| DSWP-tier (23%) | Japanese only, sequence-level half-mask on speaker + timing |
| Birth-tier (15%) | Japanese only, full info |
| Total tokens | 39,000 |
| TTR pre-filter | bottom 10% per language |
| Vocab cap | none (let V land naturally at 200–240) |
| Architecture | M7 (2L, 4h, d=64, K=8) — unchanged |

**Why the full Mandarin tonal syllable, not Initial×Tone or Final×Tone alone:** Initial × Tone (V=91) drops the ending; Final × Tone (V=209) drops the initial. Both miss diversity. Full syllable preserves both halves and is what natural Mandarin orthography (pinyin) encodes per character — closest to "what counts as a unique pronounceable syllable" the way whale rhythm × tempo × rubato × ornament counts a unique articulated coda.

**Why V≈218 and not V=467 or V=189:** see [[overview/whale-clans-pacific-vs-caribbean]]. V=467 includes 116 Pacific-only rhythm classes that come from pooling multiple unlabelled clans; V=218 is the full Caribbean (DSWP+Birth) compound (rhythm × tempo × rubato × ornament) — what the Sharma 2024 paper actually analyzes. **Chorus is *not* folded into V** — it's a per-utterance structural marker (already represented as the row-level `Synchrony` column in the rendered CSV), analogous to punctuation in CHILDES. We don't add a CHILDES analog for it because chorus structure lives at a different granularity than the per-token compound vocabulary.

## Tier allocation (mirrors whale's 3-source structure)

| tier | role | tokens | language(s) | speaker | timing |
|---|---|---:|---|---|---|
| Hersh-equiv | high missingness | 24,180 (62%) | EN + JP + ZH | UNK on all | NA on all |
| DSWP-equiv | partial missingness | 9,000 (23%) | Japanese only | half UNK, half kept | half NA, half estimated |
| Birth-equiv | full info | 5,820 (15%) | Japanese only | always kept | always estimated |

## Implementation work (from the plan doc)

| component | est. LOC | location |
|---|---:|---|
| `multilang_loader.py` | ~350 | NEW: `src/grammar/multilang_loader.py` |
| `dt_buckets.scheme_for("multilang")` | 3 | `src/grammar/dt_buckets.py` |
| `train_for_interp.py` branch | ~15 | `src/grammar/train_for_interp.py` |
| `g2p-en` dependency | 1 | `requirements.txt` |
| `corpus_stats.py` `--source multilang` | ~10 | `src/grammar/corpus_stats.py` |
| **Total** | **~380 LOC** | 2 files modified + 1 new |

The English phoneme tokenizer (g2p-en) is the only genuinely new piece. Japanese mora and Mandarin syllable tokenizers are already implemented in [[transformer/cross-language-units|corpus_stats.py]]; they just need to be reused.

## Acceptance checks before training

After running the loader, the regen output should report:

| check | acceptable range |
|---|---|
| total tokens | 38,500–39,500 |
| total V | 200–240 |
| `has_timestamps == 0` rate | 0.60–0.65 |
| `Whale ends in ::UNK` rate | 0.69–0.74 |
| modal share | 6–10% |
| p90 rank | 80–130 |

## Predicted outcomes

| metric | whale (current) | predicted multilang |
|---|---:|---:|
| val_coda_bpt | 3.36 | 3.6–4.1 |
| val_coda_acc | 0.30 | 0.20–0.27 |
| logit lens L1 contribution | +0.087 | +0.05–0.10 |

If val_coda_bpt lands near whale's 3.36 (rather than capped CHILDES's 4.63), the structural matching worked. If it stays near 4.5, the multi-language mix is just noise — the model is gating-by-language and learning three separate small distributions instead of cross-language structure.

## Open questions deferred to the next session

1. Per-tier bpt breakdown — does Hersh-tier do worse than Birth-tier?
2. Per-language head specialization — do any of the 8 attention heads gate on language ID?
3. Continuation samples — coherent same-language strings vs mixed-language nonsense?
4. **Single-language baseline** (Japanese-only at 39k, same missingness, V≈110) — only run if the multi-language result is hard to interpret.

## Related

- `docs/multilang_corpus_plan.md` — the actual implementation plan with code sketches.
- [[transformer/cross-language-units]] — the corpus measurements that informed the V target.
- [[overview/whale-clans-pacific-vs-caribbean]] — why V=218 (Caribbean compound + rubato, **no chorus**) is the target.
- [[transformer/hersh-mirror]] — single-language Hersh-mirror experiment, the simpler precursor.
