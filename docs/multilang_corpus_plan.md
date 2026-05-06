# Multi-language CHILDES counterpart for whale-grammar training

**Status**: revised plan (v2). The v1 plan (Japanese moras for clean tiers, target V≈218) was abandoned after a timing-structure analysis of the whale data and an empirical V-floor measurement on Mandarin. This document is the working spec; the previous Japanese-mora-clean-tier framing is obsolete.

## Why this exists

We want a fair human-language counterpart corpus to train the M7 transformer
on, so we can compare per-token compressibility, attention head behaviour, and
continuation quality between whale codas and human child speech with confounds
controlled.

The v1 plan tried to *match* whale on per-token diversity (V≈218) by picking a
human-language unit (Japanese moras) with a similarly-sized inventory. That
turned out to be the wrong axis to match. After looking at the whale data
directly:

1. **Whale codas are word-level discrete units, not sub-word units.** Same-speaker inter-coda gaps in the clean (DSWP+Birth) tiers are essentially never under 0.5s — typical 1.5–5s, distributed unimodally. Tight gaps (<0.3s) are 99–100% speaker switches (i.e. chorus / overlap between different individuals). There is no "moras-within-a-word" grouping inside one whale's stream. Each coda is its own lexical-sized unit, separated from the next by a phrase-or-breath-level pause.
2. **Mandarin full tonal syllables are the structural analog.** Each Mandarin syllable carries a morpheme; words are 1–3 syllables but the syllable-to-syllable timing dominates what an ear (or a clock) sees. Japanese moras chunk 2–4 deep inside one word (tightly-packed within-word DT≈0.05–0.2s), which is *not* what whale does. English phonemes pack even tighter. Mandarin keeps each token as a discrete, timed event with phrase-level boundaries between them — the same structure whale exhibits.
3. **Mandarin's per-token V is intrinsically higher than whale's, and we accept that.** Even one young child + family at 8k tokens gives V≈420–480 full tonal syllables; one cannot squeeze that down to whale's V=218 without artificially crippling the unit (e.g. dropping initials → Final×Tone V=178). The v1 "match V via filtering" approach is structurally infeasible. This v2 plan deliberately runs at the natural Mandarin V and frames the comparison differently (see *Predicted training outcomes*).

Confounds we're still removing:

1. **Word-level granularity mismatch (v1 framing).** CHILDES-Eng-UK lemmas at 39k tokens give V≈1605 — way more than whale, with only ~24 gradient updates per type. We need a sub-lexical unit on the human side. v1 picked Japanese moras for V-matching; v2 picks Mandarin full tonal syllables for *timing-structure* matching, accepting higher V.
2. **Missingness asymmetry.** ~62% of the whale corpus (Hersh Pacific recordings) has no speaker IDs and no per-coda timestamps. Standard CHILDES has both for every utterance. The whale model has to *learn to gate* on `has_timestamps`; a vanilla CHILDES model never does. The Hersh-equivalent tier in this corpus replicates that gating.
3. **Speaker-pool size.** Whale clean tiers (DSWP + Birth) record a small set of Caribbean clan members across two recording sessions. We mirror this by restricting the clean-tier Mandarin source to a small set of young children + their families (Tong + Zhou3 + TCCM ≤36mo).

## Target corpus shape — v2

| property | target | rationale |
|---|---|---|
| total tokens | ≈ 39,000 | matches `data/classified/whale_dialogues.csv` (38,840 codas) |
| total V (multi-language compound) | **≈ 500–700, dominated by Mandarin** | natural per-token diversity of full tonal syllables on a small young-Mandarin pool; not engineered to match whale |
| Hersh-equivalent missingness fraction | **0.62 of tokens** | matches hersh2022_pacific share (24,237 / 38,840) |
| sequence chunk length | ~80 tokens | matches whale per-recording length |
| architecture | M7: 2L, 4h, d=64, K=8 | already wired in `src/grammar/m7_hooked.py` |

**Important: V is no longer a tuning target.** We let it land at whatever the
restricted-Mandarin pool produces (≈500–700 expected). The interesting
comparison is no longer "matched V across corpora"; it's "given a higher-V,
higher-entropy human-language counterpart with the *same* token-level timing
structure as whale, what does the model learn?"

The experimental hypothesis: if M7 can find structure in this natural,
relatively-complex Mandarin data, the same architecture should find at least
as much structure in whale (which has roughly half the V on the same token
budget). If whale val_coda_bpt < multilang val_coda_bpt by roughly the V/H gap
expected from inventory size alone, the comparison is honest. If whale ≪
multilang by *more* than the V gap predicts, that surplus compressibility is
the "whale grammar" signal.

**Note on chorus**: chorus is *not* a per-coda feature folded into the compound vocabulary. It's a per-utterance structural marker — "this group of codas was produced simultaneously by multiple whales" — analogous to punctuation in CHILDES, which marks utterance structure but isn't part of any individual word's identity. The Sharma 2024 paper's compound is **rhythm × tempo × rubato × ornament**, full stop. Chorus already lives in the rendered CSV as the row-level `Synchrony` column; we don't need to fold it into the token vocabulary, and we don't need a CHILDES analog for it.

## Whale timing evidence (the basis for v2)

Measured on `data/classified/whale_dialogues.csv` clean tiers
(`sharma2024_dswp` + `sharma2025_birth`, has_timestamps==1, n=9,275):

| inter-coda gap | DSWP same-spk | DSWP switch | Birth same-spk | Birth switch |
|---|---:|---:|---:|---:|
| <0.1s | 0 (0%) | 164 (100%) | 2 (0.5%) | 416 (99.5%) |
| 0.1–0.3s | 0 | 281 (100%) | 12 | 417 (97.2%) |
| 0.3–0.75s | 0 | 379 (100%) | 107 | 801 (88.2%) |
| 0.75–1.5s | 10 | 259 (96%) | 747 | 518 (40.9%) |
| 1.5–5s | 1003 | 952 (49%) | 1562 | 803 (34.0%) |
| >5s | 345 | 168 (33%) | 279 | 50 (15.2%) |

Read-out: **same-speaker gaps cluster at 1.5–5s, never under 0.5s in DSWP and
~0.4% under 0.5s in Birth.** All sub-second gaps are speaker switches (the
chorus / overlap regime). A whale's own coda stream therefore has no
sub-word-like grouping — each coda is a discrete word-or-phrase-sized unit
with breath-level pauses between them. This is the structural argument for
syllable-level Mandarin (one syllable = one word-sized unit) over
mora-level Japanese (multiple moras chunk inside one word).

## Sub-lexical unit choice per language

| language | unit | est. V at planned share | rationale |
|---|---|---:|---|
| **Mandarin (clean + Hersh tiers)** | **full tonal syllable (Initial + Final + Tone)** | **~500–700 across the whole corpus** | each syllable is a discrete morpheme-sized event with phrase-level timing — structurally matches whale's coda-as-word pattern. We accept the high V; see *Why this exists*. |
| Japanese (Hersh tier only, optional) | mora | ~80–110 | adds Hersh-tier multi-clan diversity; mora-internal timing doesn't match whale, but Hersh tier is the noisy multi-source tier where that mismatch is fine. |
| English (Hersh tier only, optional) | ARPABET phoneme | ~38–44 | adds Hersh-tier multi-clan diversity; same caveat. |

**Why Mandarin (full tonal syllable) for the clean tiers:** see *Whale timing
evidence* above. Each whale coda functions like one Mandarin syllable: a
discrete unit with a clear timing boundary before the next one. The
within-word multi-mora chunking that defines Japanese is precisely what
whale doesn't do.

**Why full tonal syllable, not Final×Tone:** the v1 plan considered Final×Tone
(V≈178) as a way to hit V≈218. We're rejecting that because it drops the
leading consonant, which would mean `ma1` (mom), `ba1` (dad), and `hua1`
(flower) all collapse to the same `a1` token. That's not representative of
the Mandarin a child actually hears — it's an engineered V suppression that
defeats the point of using a real human language to begin with. v2 keeps
the full pinyin syllable.

Token-level prefixing in the unified vocab to keep inventories disjoint:
- Mandarin tokens: `zh:ni3`, `zh:hao3`, `zh:shi4`, … (full tonal syllable, initial+final+tone)
- Japanese tokens: `jp:ka`, `jp:n`, `jp:shi`, …
- English tokens: `en:AH`, `en:T`, `en:S`, …

If any inventory accidentally overlaps a surface form (e.g. English `n` and Japanese `n`), the prefix disambiguates.

## Mandarin source restriction (clean tiers + Hersh ZH portion)

Whale's clean tiers (DSWP + Birth) are recordings of a small set of Dominica
clan members. To mirror that "few-individuals" structure, we restrict
Mandarin to corpora that are:
- Single-child (or small handful), with continuous longitudinal recording
- Target child age ≤ 36 months (≤ 3 years)
- Have a parseable `%mor:` tier with pinyin lemmas

After scanning `Mandarin/`:

| sub-corpus | files | child age range | speaker codes | usable tokens |
|---|---:|---|---:|---:|
| `Mandarin/Tong/` | 22 | 19–40mo (cap to ≤36mo) | 8 | ~138k |
| `Mandarin/Zhou3/` | 30 | 8–53mo (cap to ≤36mo) | 8 | ~86k |
| `Mandarin/TCCM/{yang,xu,pan,jc,wu,wang,chou,wuys,chw,cheng}/` | ~150 | 17–56mo (cap to ≤36mo) | ~30 | ~280k |

**Excluded** (do NOT try to recover any of these — exclusion is intentional):
- `Beijing/*` — no `%mor:` tier; surface pinyin only on speaker lines. We are deliberately not writing a Beijing-specific surface-pinyin parser; the rest of the pool is plenty.
- `AcadLang`, `LiReading`, `LiZhou`, `NSCtoys`, `Xinjiang`, `Chang*`, `ChangPN`, `ChangPlay`, `ENNI`, `Caterpillar`, `Robber`, `ZhouAssessment`, `ZhouDinner`, `ZhouNarratives`, `Zhou1`, `Zhou2` — older children, broader speaker pools, or no `%mor:`.

Total available clean-tier Mandarin pool ≈ 500k tokens after age cap. We
need ~14.8k for clean tiers (DSWP + Birth) and ≤ 8k for the Hersh-tier ZH
portion → plenty of headroom. We also still apply bottom-10% TTR per-conv
to prefer the most-repetitive (most child-typical) sessions. (The
bottom-10% is for naturalness — picking simple child-directed sessions —
not for V-matching to whale, since the timing analysis showed the human
languages can't reach whale's median TTR at any practical filter setting.)

The Hersh-tier ZH portion uses the **same restricted pool** (chunks
disjoint from the clean tiers). Keeping the speaker pool small across all
tiers mirrors whale's clan-bounded vocabulary: the model isn't asked to
generalize across "Mandarin in general," it's asked to learn structure in
"Tong's family + Zhou3's family + a few TCCM families."

### How to parse age from a `.cha` file

Age lives in the `@ID:` header line for the `Target_Child` participant.
Format example:

```
@ID:	zho|Beijing|CHI|1;08.28|male|||Target_Child|||
```

After splitting on `|` and stripping whitespace, the fields land at:

```
[0] zho                    language
[1] Beijing                corpus name
[2] CHI                    speaker code
[3] 1;08.28                age (y;mm.dd; mm/dd may be empty)
[4] male                   sex
[5..6] (empty)
[7] Target_Child           role  ← this is the role index (NOT 6)
[8..10] (empty)
```

The age regex: `^(\d+);(\d+)?` captures (years, months). Convert to total
months. **Filter is per-file, not per-directory.** Each `.cha` file
records the Target_Child's age at recording; some children have files
spanning a wide range (e.g. Zhou3 is 8–53mo across files), so the loader
opens each candidate file, reads its Target_Child age, and rejects the
file if age > 36mo. Reference implementation lives near
`multilang_loader.py`'s `_build_convs` — add a per-file filter there for
the Mandarin path only.

## Tier allocation (mirroring whale's 3-source structure)

| tier | role | tokens | language(s) | speaker | timing |
|---|---|---:|---|---|---|
| Hersh-equiv | high missingness | **24,180** (62%) | EN + JP + ZH mix | UNK on all | NA on all |
| DSWP-equiv | partial missingness | **9,000** (23%) | **Mandarin only** (young single-child pool) | half of sequences UNK, half kept | half NA, half estimated from punct |
| Birth-equiv | full info | **5,820** (15%) | **Mandarin only** (young single-child pool) | always kept | always estimated from punct |

Total = 39,000 tokens.

**Hersh-tier per-language ratio:** start equal thirds (8,060 EN + 8,060 JP +
8,060 ZH). The Mandarin Hersh-tier portion uses the same restricted
single-child pool as the clean tiers (drawn disjoint chunks). EN and JP
keep their full unfiltered CHILDES pools — they exist in this corpus
purely to provide multi-clan-style diversity in the Hersh tier, not to
match whale's V.

**No more "tune Mandarin share until V hits 218."** v1 had an iterative
rebalance loop because V=218 was the target. v2 doesn't tune V — we accept
whatever V the small-pool restricted Mandarin gives us, and report it. If
total V lands far outside the rough [500, 800] envelope (e.g. <300 or
>1200) something is wrong with the loader, but the band is a sanity check,
not a goal.

**Why this allocation:**

- The **Hersh-equiv tier** mirrors whale Pacific multi-clan diversity. Three loosely-disjoint sub-vocabularies (EN phonemes, JP moras, ZH syllables) sit alongside each other with all speaker/timing info stripped. The model has to learn that 62% of its training data has no speaker channel, no temporal channel, and three different sub-vocabulary sub-distributions — same problem the whale model faces with hersh2022_pacific.
- The **DSWP-equiv tier** is partially-clean Caribbean-equivalent data. **Mandarin only**, because the clean tier's job is to give the model a token-level structure that *resembles whale*. Whale codas are word-level discrete events with phrase-level pauses; Mandarin syllables are the same. Half-masked at sequence-level, mirroring DSWP's per-recording variation in `has_timestamps`.
- The **Birth-equiv tier** is fully-clean Mandarin from the same pool. Speakers and DT are always present.

**Why Mandarin (not Japanese) for the clean tiers:** see *Why this exists* and
*Whale timing evidence*. Short version: whale doesn't have multi-mora-per-word
chunking; each coda is its own timed unit with no sub-word grouping. Mandarin
syllables match that structure. Japanese moras don't.

## Per-conversation pre-filter

Apply bottom-10% TTR filter per-language before chunking, exactly as we did in
`corpus_stats.py --bottom-ttr-pct 0.10`. This keeps the most repetitive
(simplest, most child-typical) conversations and gives the densest per-type
training signal.

Skip filtering only if a language source is small enough that filtering would
produce <2× the target tokens for that language.

## Sequence chunking

Same scheme as `childes_loader.py:336-353`:
- Build per-conversation token streams.
- Cut into ~80-token chunks at the next speaker switch after the soft target.
- One chunk = one `sequenceId` row in the output CSV.

Per language separately. **Chunks are language-pure** (no within-chunk
mixing). Mixing happens at the chunk-shuffling stage.

## Pool partitioning order

Mandarin is shared across all three tiers but chunks must be disjoint
between Hersh-ZH, DSWP, and Birth. Allocation order (after shuffling
with seed 42):

```
1. Restricted-Mandarin pool (Tong + Zhou3 + TCCM ≤36mo) → shuffle
   a. Take chunks for Birth tier (≈ 5,820 tokens)
   b. Take chunks for DSWP tier (≈ 9,000 tokens)
   c. Take chunks for Hersh-tier ZH portion (≈ 8,060 tokens)
2. Full English pool → shuffle → take chunks for Hersh-tier EN portion (≈ 8,060 tokens)
3. Full Japanese pool (MiiPro) → shuffle → take chunks for Hersh-tier JP portion (≈ 8,060 tokens)
```

The Hersh-tier ZH portion comes from the *same* young-single-child pool as
the clean tiers — chunk-disjoint but speaker-pool-shared. This deliberately
keeps Mandarin's "speaker count" small across the whole corpus, mirroring
whale's clan-bounded vocabulary.

## Hersh-tier construction

```
1. Concatenate Hersh-EN + Hersh-JP + Hersh-ZH chunks. Shuffle with seed.
2. For every chunk, every row:
     - sequenceId stays unique
     - Whale = "multilang::UNK"            (speaker stripped)
     - TimeDelta = -1.0                    (timing stripped)
     - has_timestamps = 0                  (gate flag)
3. Write to CSV in whale_dialogues.csv schema.
```

This gives 24,180 tokens of fully-stripped, language-mixed sequences — one row per token, sequenceIds let CV folds keep chunks atomic.

## DSWP-tier construction

```
1. Take Mandarin chunks (already drawn from the restricted pool, disjoint from Hersh-ZH and Birth) until cumulative tokens >= 9,000.
2. Random-select half of the chunks (sequence-level mask).
3. Masked half: Whale = "zh_dswp::UNK", TimeDelta = -1.0, has_timestamps = 0.
4. Unmasked half: Whale = "zh_dswp::<speaker>", TimeDelta from punctuation, has_timestamps = 1.
```

## Birth-tier construction

```
1. Take Mandarin chunks (disjoint from above) until cumulative tokens >= 5,820.
2. Whale = "zh_birth::<speaker>", TimeDelta from punctuation, has_timestamps = 1.
```

Use a different tier prefix from DSWP (`zh_dswp::` vs `zh_birth::`) so the
model can in principle distinguish the two recording sessions, mirroring the
whale `sharma2024_dswp` vs `sharma2025_birth` namespacing.

## TimeDelta estimation per language

Reuse the heuristic from `childes_loader.py:60-64`:

```
DT_INTRA_UTT       = 0.3   # within an utterance, no punctuation
DT_AFTER_COMMA     = 0.5   # after ","
DT_AFTER_TERMINAL  = 1.0   # after "." "?" "!"
DT_SPEAKER_SWITCH  = 2.0   # next speaker takes the floor
```

Three small per-language adjustments:

- **English** (Hersh tier only): punctuation comes from the `%mor:` tier. Already handled by `childes_loader._parse_mor`.
- **Mandarin** (all tiers): same convention — `.` `?` `!` `,` appear in `%mor:`. The full tonal syllable is reconstructed from `%mor:` lemmas: each lemma is split into syllables via `corpus_stats._PINYIN_SYLL`, then each `(initial+final+tone)` is emitted as one token (e.g. `gang1cai2` → `zh:gang1`, `zh:cai2`).
- **Japanese** (Hersh tier only): `%ort:` carries the romaji surface form. Use the markers (`.`, `?`, `!`) at the end of `%ort:` lines as utterance terminals; treat utterance boundaries as terminals if no punctuation found.

For phoneme-level English (Hersh): every word-final phoneme inherits the
word's following-punctuation DT; intra-word phonemes get **DT=0.0**
(intra-word bucket).

For mora-level Japanese (Hersh): every word-final mora inherits the same
way; intra-word moras get **DT=0.0**.

For Mandarin full tonal syllable (all tiers): each pinyin syllable maps to
one token. For multi-syllable lemmas (e.g. `gang1cai2`), the first syllable
inherits the lemma's DT (utterance / speaker boundary); subsequent syllables
get **DT=0.0** (intra-word). Punctuation attaches between lemmas.

**Loader change** (vs v1): the loader currently emits `DT_INTRA_UTT=0.3` for
within-lemma sub-tokens. v2 emits `DT=0.0` for those, so the
`MULTILANG_INTRA_WORD` bucket actually fires.

## DT bucketing

The 5-class multilang scheme is already defined in `dt_buckets.py`:

```
0 intra_word   DT == 0.0     sub-tokens within one word (mora 2..N, phoneme 2..N, ZH syllable 2..N)
1 word_bound   DT in {0.3, 0.5}  same-speaker, between words / after comma
2 period       DT == 1.0     end of utterance, same speaker
3 switch       DT == 2.0     end of utterance + speaker change
4 missing      has_timestamps == 0   Hersh-tier rows
```

The boundary-detection task is non-trivial only for Hersh JP/EN (where
multi-mora words are common). For Mandarin clean tiers, most lemmas are
single-syllable, so `intra_word` fires only on multi-syllable lemmas like
`ma1ma1` (mom-mom) — sparse but informative.

## Unified vocabulary

Build once at the end:

1. Collect all distinct prefixed tokens across all three tiers.
2. Sort by frequency descending (top types get low ids).
3. Reserve id 0 for `<pad>` if needed by the trainer.
4. Write `data/classified/multilang_word_index.csv` with columns `word_id,word`.
5. Write `data/classified/multilang_dialogues.csv` matching the whale schema:
   `sequenceId, itemPosition, Whale, Coda, Synchrony, Ornamentation, Duration, TimeDelta, has_timestamps`.

`Synchrony` and `Ornamentation` are always 0 (no human-language analog —
this is fine, those columns are also mostly 0 on the whale side).
`Duration` is always 0.0 (no per-token duration estimate).

**Do not vocab-cap.** v2 doesn't engineer V — we accept whatever the
restricted Mandarin pool produces. Reporting it (in the loader summary and
in `corpus_stats --source multilang`) is the deliverable, not making it
hit a number.

## File-by-file changes (v2 delta)

**The loader file already exists** at `src/grammar/multilang_loader.py`
(built during v1). Do NOT recreate it from the skeleton in this doc — edit
in place. v1 outputs (`data/classified/multilang_dialogues.csv` and
`multilang_word_index.csv`) also exist and will be **overwritten** by the
v2 run; no manual cleanup needed.

Concrete edits required in `multilang_loader.py`:

| where | v1 today | v2 change |
|---|---|---|
| `_build_convs` | for sub-tokens within a lemma, emits `dt if i == 0 else DT_INTRA_UTT` (= 0.3) | change to `dt if i == 0 else 0.0` so the `MULTILANG_INTRA_WORD` bucket fires |
| `load_zh()` | `_build_convs(ZH_ROOT, "%mor:", _extract_zh_mor, _expand_zh, lang="zh")` — globs all of `Mandarin/` | wrap with a path + age filter: keep file only if `path.parent.parent.name in {"Tong","Zhou3"} or (path.parent.parent.name == "TCCM")` AND `target_child_age_months(path) <= 36`. Add a small `target_child_age_months()` helper using the `@ID:` parse spec above. |
| `main()` clean-tier allocation | draws from `jp_chunks` for Birth + DSWP | change to draw from `zh_chunks` (the restricted pool). JP no longer feeds clean tiers. |
| `main()` Hersh-tier ZH | draws from the same `zh_chunks` *after* clean-tier allocation has consumed its share | unchanged — but the comment should note that the partition order is Birth → DSWP → Hersh-ZH (disjoint). |
| `_seq_id` / row emit | tier strings `"jp_dswp"`, `"jp_birth"` | rename to `"zh_dswp"`, `"zh_birth"`. Whale-column strings flip the same way: `"jp_dswp::UNK"` → `"zh_dswp::UNK"`, etc. |

Other files:

| file | status |
|---|---|
| `src/grammar/dt_buckets.py` | Already has the 5-class multilang scheme. No change. |
| `src/grammar/train_for_interp.py` | Already wired. No change. |
| `src/grammar/corpus_stats.py` | Already wired (`load_multilang`). No change. |
| `requirements.txt` | Already has `g2p-en`. No change. |
| `data/classified/multilang_dialogues.csv` | Regenerated by the v2 run. |
| `data/classified/multilang_word_index.csv` | Regenerated; V will be ~500–700. |

No changes to `m7_hooked.py`, `predict_kfold.py`, `interp_compare.py`,
`continuations.py`. The architecture and downstream interp pipeline are
V-agnostic.

## One-time environment setup

The `g2p-en` dependency relies on NLTK data that doesn't ship with the
package and isn't installed by `pip install g2p-en`. On a fresh env, the
loader will crash with `LookupError: Resource averaged_perceptron_tagger_eng
not found.` Do this once before running the loader:

```bash
micromamba run -n py311 python -c \
    "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')"
```

The downloads go to `~/nltk_data/` and persist across runs.

## Code sketch — `multilang_loader.py` skeleton

```python
"""
Multi-language CHILDES corpus that mirrors whale_dialogues.csv structure.

Outputs:
    data/classified/multilang_dialogues.csv
    data/classified/multilang_word_index.csv
"""
import argparse, random, re
from pathlib import Path
import numpy as np, pandas as pd

# Reuse the punctuation + DT scheme from childes_loader
from src.grammar.childes_loader import (
    DT_INTRA_UTT, DT_AFTER_COMMA, DT_AFTER_TERMINAL, DT_SPEAKER_SWITCH,
    TERMINAL_PUNCT, parse_cha,
)
# Reuse the tokenisers from corpus_stats
from src.grammar.corpus_stats import (
    segment_moras,            # JP — produces moras like 'ka', 'shi', 'n'
    split_pinyin, _PINYIN_SYLL,  # ZH — split_pinyin returns (initial, final, tone)
                                 # build full tonal syllable as initial+final+tone
)
import g2p_en                 # NEW dependency for EN

ROOT = Path(__file__).resolve().parents[2]

def load_eng_phonemes(min_tokens=30) -> list[Conv]:
    """Iterate Eng-UK .cha, %mor lemmas → CMU phonemes via g2p-en."""
    g = g2p_en.G2p()
    convs = []
    for path in sorted((ROOT / "Eng-UK").rglob("*.cha")):
        toks = parse_cha(path)   # (speaker, lemma, dt) triples
        if len(toks) < min_tokens: continue
        rows = []
        for speaker, lemma, dt in toks:
            phs = [p for p in g(lemma) if p[0].isalpha()]   # drop stress digits
            for i, ph in enumerate(phs):
                rows.append((speaker, f"en:{ph}",
                             dt if i == 0 else DT_INTRA_UTT))
        convs.append(Conv(cid=str(path.relative_to(ROOT)),
                          tokens=rows))
    return convs

def load_jp_moras(min_tokens=30) -> list[Conv]: ...

def load_zh_full_syllable(min_tokens=30) -> list[Conv]:
    """For each Mandarin %mor: lemma, split into pinyin syllables and emit
    the full tonal syllable per syllable (initial + final + tone, prefixed
    with 'zh:'). Example: lemma 'gang1cai2' → tokens ['zh:gang1', 'zh:cai2'].
    Drops syllables where split_pinyin returns None (non-pinyin glosses).
    """
    ...

def filter_bottom_ttr(convs, frac=0.10):
    """Keep bottom-frac TTR conversations by token-set ratio."""
    ...

def chunk(convs, target=80):
    """Cut into ~80-token chunks at speaker switches."""
    ...

def build_hersh_tier(en_chunks, jp_chunks, zh_chunks, target_tokens, rng):
    """Draw equal thirds, shuffle, return list of Chunk(sequenceId, rows)."""
    ...

def build_clean_tier(jp_chunks, target_tokens, mask_frac, rng):
    """Draw Japanese chunks; mask `mask_frac` at sequence level."""
    ...

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=39_000)
    ap.add_argument("--hersh-frac", type=float, default=0.62)
    ap.add_argument("--dswp-frac", type=float, default=0.23)
    ap.add_argument("--bottom-ttr-pct", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", type=Path,
                    default=ROOT/"data/classified/multilang_dialogues.csv")
    ap.add_argument("--out-index", type=Path,
                    default=ROOT/"data/classified/multilang_word_index.csv")
    args = ap.parse_args(argv)

    # ... build per-language conv pools, filter, chunk
    # ... assemble three tiers
    # ... unified vocab build, CSV write

if __name__ == "__main__":
    main()
```

## CLI to run after wiring

Prereq: complete the **One-time environment setup** above (NLTK data for
g2p-en) before the first build. Then:

```bash
# 1. Build the corpus (overwrites the v1 outputs in data/classified/)
micromamba run -n py311 python -u -m src.grammar.multilang_loader \
    --target-tokens 39000 --hersh-frac 0.62 --dswp-frac 0.23 \
    --bottom-ttr-pct 0.10  2>&1 | tee /tmp/multilang_build.log

# 2. Run the post-build verification (see "Quick post-build verification script")
micromamba run -n py311 python tests/check_multilang_v2.py

# 3. Verify shape via the existing corpus_stats
micromamba run -n py311 python -m src.grammar.corpus_stats --source multilang

# 4. Train
micromamba run -n py311 python -u -m src.grammar.train_for_interp --source multilang \
    2>&1 | tee reproducibility/logs/train_multilang.log

# 5. Run interp comparison
micromamba run -n py311 python -m src.grammar.interp_compare --sources whale,multilang
```

## Acceptance checks before training

After running the loader, the printed summary and `corpus_stats --source
multilang` should report:

| check | acceptable range | rationale |
|---|---|---|
| total tokens | 38,500–39,500 | matches whale |
| total V (compound) | **500–800** | natural per-token diversity of the unit, not a tuning target |
| V breakdown | V_zh ≈ 350–600, V_jp ≈ 80–120, V_en ≈ 35–50 | sanity-check per-language |
| has_timestamps == 0 fraction | 0.69–0.74 | Hersh tier (62%) all-missing + half of DSWP (~12% of total) |
| Whale ending in "::UNK" fraction | 0.69–0.74 | same — masked rows = ::UNK |
| Mandarin clean-tier files | drawn only from Tong + Zhou3 + TCCM ≤36mo | confirm restriction is wired |
| `intra_word` bucket fires | non-zero count, mostly on Hersh JP/EN sub-tokens | confirm DT=0.0 emission for sub-token mora/phoneme |
| modal share | 2–6% | flatter than whale's 6–10% because more types |
| p90 rank | 200–400 | scales with V |
| tokens/type | 50–80 | lower than whale's 100–125, reflecting larger V |

If V falls dramatically outside [400, 1000], something is wrong with the
loader or the source filter — investigate before iterating on shares.

### Quick post-build verification script

After the loader writes `multilang_dialogues.csv`, run this to catch a
mis-wired filter immediately. All four assertions should pass:

```python
import re
import pandas as pd
df = pd.read_csv("data/classified/multilang_dialogues.csv")
df["tier"] = df["sequenceId"].str.split("::").str[0]

# 1. clean-tier rows only come from Tong, Zhou3, or TCCM
clean = df[df["tier"].isin(["zh_dswp", "zh_birth"])]
clean_corpora = clean["sequenceId"].str.extract(
    r"^[^:]+::zh::([^:]+)::"
)[0].unique()
assert set(clean_corpora).issubset({"Tong", "Zhou3", "TCCM"}), \
    f"clean tier leaked non-restricted Mandarin: {set(clean_corpora)}"

# 2. Hersh-tier ZH portion shares the same restricted pool
hersh_zh = df[(df["tier"] == "multilang") &
              (df["Coda"].astype(int) >= 0)]   # all rows, then check by Whale
# (alternatively, inspect by token prefix in the index file)

# 3. has_timestamps==0 fraction within band
frac_missing = (df["has_timestamps"] == 0).mean()
assert 0.65 <= frac_missing <= 0.75, frac_missing

# 4. intra_word bucket fires (DT==0.0 for sub-tokens of multi-unit lemmas)
n_intra = ((df["TimeDelta"] == 0.0) & (df["has_timestamps"] == 1)
           & (df["itemPosition"] > 0)).sum()
assert n_intra > 100, f"intra_word bucket nearly empty: {n_intra}"
print("OK")
```

Save as `tests/check_multilang_v2.py` if you want it to live alongside the
build. Catches the three most likely loader regressions: wrong Mandarin
source, wrong missingness fraction, no DT=0.0 emission.

## Predicted training outcomes

Falsifiable predictions to check after training. v2's predictions are
deliberately *worse* than v1's because the corpus is harder (higher V,
higher entropy floor).

| metric | whale (current) | predicted v2 multilang | rationale |
|---|---:|---:|---|
| val_coda_bpt | 3.36 | **4.5–5.5** | V≈600 sets a higher entropy floor than whale's V≈218; gap should track the V/H ratio |
| val_coda_acc | 0.30 | 0.10–0.18 | flatter modal share + larger V reduce raw top-1 accuracy |
| val_dt_bpt | 0.34 | **0.45–0.65** | 5-bucket multilang DT (vs whale's missing-dominated 6-bucket); intra-word + word-bound buckets are real signal |
| val_dt_acc | 0.96 | 0.80–0.90 | majority class is still "missing" but DT distinctions on clean tiers are non-trivial |
| logit lens L1 contribution | +0.087 | +0.04–0.10 | sub-lexical structure should still benefit from L1 mixing |

The **key cross-corpus comparison** is no longer "did multilang match whale's
val_coda_bpt." It is:

- Compute V/H baselines on each corpus (uniform-over-V entropy = log2(V)).
- Compute (val_coda_bpt) / log2(V) — the model's *fraction of unigram
  entropy it failed to compress*. Lower = more structure found.
- If whale's compression fraction is materially lower than multilang's, the
  surplus is the "whale grammar" signal. If they're equal, M7 is just
  matching unigram statistics in both cases.

## Open follow-ups (post-training)

1. **Per-tier bpt breakdown**: compute val_coda_bpt separately for Hersh-tier rows vs DSWP-tier vs Birth-tier rows. We expect Hersh ≫ Birth because the Hersh tier has both missingness *and* multi-language noise.
2. **Per-language head specialization**: do any of the 8 attention heads look like they specialize in English / Japanese (Hersh) vs Mandarin (clean)? If so, the model is gating by language. Especially interesting: do clean-tier-only heads emerge, since Mandarin is the only language with non-missing DT?
3. **Continuation samples**: do they look like coherent same-language strings, or do they produce mixed-language nonsense? Clean-tier continuations should look like Mandarin pinyin syllables with reasonable timing.
4. **DT-conditioned generation**: at clean-tier positions, can the model recover a plausible mid-utterance vs end-of-utterance distinction from the same residual stream that drives the coda head? This is the analog of the whale "Δt~0.8 vs Δt>4" prediction.

## Notes for the implementer

- **No vocab cap.** The natural V≈500–700 is what we want. Capping creates UNK noise that doesn't exist in the whale data.
- **No chorus column.** Chorus is per-utterance whale metadata (in `Synchrony`), not per-token. The Sharma 2024 compound is rhythm × tempo × rubato × ornament; chorus is not folded in.
- **No dropping Mandarin initial or final.** Full tonal syllable is the unit. Final×Tone (V≈178) was rejected as artificial V suppression.
- **No language column on the input.** Token-id prefixes (`zh:`, `jp:`, `en:`) carry the language info; the model figures out the rest. Adding an explicit channel defeats the comparison.
- **Don't filter Mandarin further to chase V**. The corpus restriction (Tong + Zhou3 + TCCM ≤36mo + bottom-10% TTR) is the single source of "naturalness" filtering. After that, V lands where it lands.
- **Seed 42** for reproducibility, like every other corpus-build script.
- The output CSV must validate against `tests/test_render_csv.py` invariants if those tests are extended to cover multilang. Currently they're whale-specific; consider adding a parallel test.
