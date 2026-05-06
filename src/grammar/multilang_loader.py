"""
Multi-language CHILDES corpus that mirrors whale_dialogues.csv structure.

Outputs:
    data/classified/multilang_dialogues.csv
    data/classified/multilang_word_index.csv

Tokenisation per language
-------------------------
- English  (Eng-UK %mor):  lemma -> ARPABET phonemes via g2p-en, prefixed `en:`.
- Japanese (MiiPro %ort):  romaji word -> moras via segment_moras, prefixed `jp:`.
- Mandarin (CHILDES %mor): pinyin lemma -> full tonal syllables (initial+final+
                           tone), prefixed `zh:`.

Token-level prefixes keep the inventories disjoint across languages.

Three tiers (mirroring whale's hersh / dswp / birth split — v2)
---------------------------------------------------------------
- Hersh-equiv  : 62% of tokens, EN+JP+ZH mixed, *all* speaker stripped to UNK,
                 *all* TimeDelta = -1.0, has_timestamps = 0.
- DSWP-equiv   : 23% of tokens, Mandarin only (Tong+Zhou3+TCCM ≤36mo pool),
                 half of sequences masked (sequence-level), half retain
                 speaker + estimated TimeDelta.
- Birth-equiv  : 15% of tokens, Mandarin only (same restricted pool), always
                 retain speaker + TimeDelta.

Mandarin was chosen for the clean tiers because each pinyin syllable is a
discrete morpheme-sized event with phrase-level timing — structurally it
mirrors whale's coda-as-word pattern (each coda separated by a breath-level
pause). Japanese moras chunk multiple-deep inside one word, which doesn't
match. See docs/multilang_corpus_plan.md.

DT estimation reuses the CHILDES heuristic: 0.3 intra, 0.5 after comma, 1.0
after `.` `?` `!`, 2.0 on speaker switch. The first sub-token of a lemma
inherits the lemma's DT; subsequent sub-tokens (e.g. trailing phonemes /
moras / syllables of a multi-unit word) get DT_INTRA_UTT.

Chorus is *not* a token-level feature — it's a per-utterance structural
marker captured by the whale CSV's Synchrony column, with no CHILDES
analog. Synchrony / Ornamentation / Duration are emitted as 0.
"""
from __future__ import annotations

import argparse
import pickle
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from src.grammar.childes_loader import (
    DT_AFTER_COMMA,
    DT_AFTER_TERMINAL,
    DT_INTRA_UTT,
    DT_SPEAKER_SWITCH,
    TERMINAL_PUNCT,
    _parse_mor,
)
from src.grammar.corpus_stats import (
    _PINYIN_SYLL,
    segment_moras,
    split_pinyin,
)

ROOT = Path(__file__).resolve().parents[2]
ENG_ROOT = ROOT / "Eng-UK"
JP_ROOT = ROOT / "Japanese"
ZH_ROOT = ROOT / "Mandarin"

OUT_CSV = ROOT / "data" / "classified" / "multilang_dialogues.csv"
OUT_INDEX = ROOT / "data" / "classified" / "multilang_word_index.csv"

CHUNK_TOKENS = 80
SEED = 42
MIN_TOKENS_PER_CONV = 30

# Sub-tokens *inside* a lemma (mora 2..N of a JP word, phoneme 2..N of
# an EN word, syllable 2..N of a multi-syllable ZH lemma) get DT=0.0
# so the model can in principle recover word boundaries from timing on
# the clean tiers: intra-word transitions get bucket 0 (intra_word),
# inter-word transitions get bucket 1 (word_bound), with a real
# numeric gap between them. The first sub-token of a lemma still
# inherits the lemma-level DT (DT_INTRA_UTT=0.3 between words within
# an utterance, DT_AFTER_COMMA=0.5, etc.).
DT_INTRA_WORD = 0.0

# v2 Mandarin restriction. Clean tiers (Birth + DSWP) and the Hersh-ZH portion
# all draw from a small young-single-child pool, mirroring whale's clan-bounded
# vocabulary on the Caribbean DSWP+Birth recordings. See docs/multilang_corpus_plan.md.
ZH_ALLOWED_CORPORA = {"Tong", "Zhou3", "TCCM"}
ZH_MAX_AGE_MONTHS = 36


# ---------------------------------------------------------------------------
# .cha parsing — generic over %mor / %ort tier
# ---------------------------------------------------------------------------


_AGE_RE = re.compile(r"^(\d+);(\d+)?")


def target_child_age_months(path: Path) -> int | None:
    """Read the @ID line for Target_Child and return age in months.

    Format example:
        @ID:	zho|Tong|CHI|1;07.18|male|||Target_Child|||
    Fields after splitting on '|': [lang, corpus, code, age, sex, _, _, role, ...].
    Returns None if no Target_Child row is found or the age cell can't be parsed.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.startswith("@ID:"):
                    continue
                fields = [f.strip() for f in line.rstrip("\n").split("|")]
                if len(fields) >= 8 and fields[7] == "Target_Child":
                    m = _AGE_RE.match(fields[3])
                    if not m:
                        return None
                    years = int(m.group(1))
                    months = int(m.group(2) or 0)
                    return years * 12 + months
    except OSError:
        return None
    return None


def _restrict_zh_convs(convs: list["Conv"]) -> list["Conv"]:
    """Keep only Mandarin convs from {Tong, Zhou3, TCCM} with target child ≤36mo.

    Conv.cid is e.g. ``Mandarin/Tong/010718.cha`` or
    ``Mandarin/TCCM/cheng/foo.cha``; ``parts[1]`` identifies the sub-corpus.
    """
    out: list[Conv] = []
    for c in convs:
        parts = Path(c.cid).parts
        if len(parts) < 2 or parts[0] != "Mandarin":
            continue
        if parts[1] not in ZH_ALLOWED_CORPORA:
            continue
        age = target_child_age_months(ROOT / c.cid)
        if age is None or age > ZH_MAX_AGE_MONTHS:
            continue
        out.append(c)
    return out


_SPEAKER_RE = re.compile(r"^\*([A-Z0-9]+):\s*(.*)$")


def _iter_utts(path: Path, tier_prefix: str) -> Iterator[tuple[str, str | None]]:
    """Yield (speaker, tier_body_or_None) per utterance.

    `tier_prefix` is `%mor:` or `%ort:`. Tab-indented continuation lines on
    the chosen tier are concatenated.
    """
    speaker: str | None = None
    body: str | None = None
    in_tier = False

    def flush():
        nonlocal speaker, body, in_tier
        if speaker is None:
            return None
        out = (speaker, body)
        speaker = None
        body = None
        in_tier = False
        return out

    text = path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        if not raw:
            continue
        if raw.startswith("*"):
            prev = flush()
            if prev is not None:
                yield prev
            m = _SPEAKER_RE.match(raw)
            if m:
                speaker = m.group(1)
        elif raw.startswith(tier_prefix):
            body = raw[len(tier_prefix):].strip()
            in_tier = True
        elif raw.startswith("%"):
            in_tier = False
        elif raw.startswith("@"):
            prev = flush()
            if prev is not None:
                yield prev
        elif raw.startswith("\t"):
            if in_tier and body is not None:
                body = body + " " + raw.strip()
        else:
            in_tier = False
    last = flush()
    if last is not None:
        yield last


def _parse_lemma_stream(
    path: Path,
    tier_prefix: str,
    extract: Callable[[str], tuple[list[str], str]],
) -> list[tuple[str, str, float]]:
    """Returns list of (speaker, lemma, dt) for one .cha file.

    `extract` takes a tier body and returns (lemmas-with-comma-sentinels,
    end_punct). Comma sentinels (`,`) bump the *next* lemma's dt to
    DT_AFTER_COMMA.
    """
    out: list[tuple[str, str, float]] = []
    prev_speaker: str | None = None
    prev_end_punct = ""

    for speaker, body in _iter_utts(path, tier_prefix):
        if body is not None:
            lemmas, end_punct = extract(body)
        else:
            lemmas, end_punct = [], ""

        comma_pending = False
        first_in_utt = True
        emitted = 0
        for tok in lemmas:
            if tok == ",":
                comma_pending = True
                continue
            if not out:
                dt = 0.0
            elif first_in_utt:
                if prev_speaker is not None and speaker != prev_speaker:
                    dt = DT_SPEAKER_SWITCH
                elif prev_end_punct in TERMINAL_PUNCT:
                    dt = DT_AFTER_TERMINAL
                elif prev_end_punct == ",":
                    dt = DT_AFTER_COMMA
                else:
                    dt = DT_INTRA_UTT
            elif comma_pending:
                dt = DT_AFTER_COMMA
            else:
                dt = DT_INTRA_UTT
            out.append((speaker, tok, dt))
            comma_pending = False
            first_in_utt = False
            emitted += 1

        if emitted > 0:
            prev_speaker = speaker
            prev_end_punct = end_punct
    return out


# ---------------------------------------------------------------------------
# Per-language lemma extractors
# ---------------------------------------------------------------------------


def _extract_eng_mor(body: str) -> tuple[list[str], str]:
    """English %mor → (lemmas-with-comma-sentinels, end_punct)."""
    return _parse_mor(body)


_PINYIN_LEMMA_RE = re.compile(r"^[a-z\d]+$")


def _extract_zh_mor(body: str) -> tuple[list[str], str]:
    """Mandarin %mor → (pinyin-lemmas-with-comma-sentinels, end_punct).
    Each lemma is a pinyin string like `gang1cai2`. English glosses (after
    `=`) are stripped. Lemmas without a tone digit are dropped.
    """
    if not body:
        return [], ""
    out: list[str] = []
    end_punct = ""
    for entry in body.split():
        if entry in TERMINAL_PUNCT:
            end_punct = entry
            continue
        if entry == ",":
            out.append(",")
            continue
        if "|" not in entry:
            continue
        for sub in entry.split("~"):
            if "|" not in sub:
                continue
            _pos, lem = sub.split("|", 1)
            lem = re.split(r"[-&=]", lem, maxsplit=1)[0]
            lem = lem.lower().strip()
            if not lem:
                continue
            if not _PINYIN_LEMMA_RE.match(lem) or not re.search(r"\d", lem):
                continue
            out.append(lem)
    return out, end_punct


_JP_DROP_TOKENS = {"xxx", "yyy", "www", "0"}


def _extract_jp_ort(body: str) -> tuple[list[str], str]:
    """Japanese %ort → (romaji-words-with-comma-sentinels, end_punct).
    Strips lengthening markers, parens, CHILDES specials. Treats trailing
    `.` `?` `!` on a word (or as a standalone token) as utterance-end.
    """
    if not body:
        return [], ""
    body = re.sub(r"\[[^\]]*\]", " ", body)
    body = re.sub(r"&[~=\-+][^\s]*", " ", body)
    body = re.sub(r"@[a-z]+", " ", body)
    body = re.sub(r"[‡«»<>]", " ", body)

    out: list[str] = []
    end_punct = ""
    for raw in body.split():
        tok = raw
        # Peel trailing punctuation (terminal or comma).
        while tok and tok[-1] in (".", "?", "!", ","):
            c = tok[-1]
            if c in TERMINAL_PUNCT:
                end_punct = c
            elif c == ",":
                # comma sentinel applied *after* the current word; if the
                # current word resolves to nothing we lose the comma signal,
                # but that's fine for DT estimation.
                pass
            tok = tok[:-1]
        # Drop lengthening colon, parens, quotes.
        tok = tok.strip(":\"'()")
        tok = tok.lower()
        if not tok or tok in _JP_DROP_TOKENS:
            continue
        if not re.match(r"^[a-z']+$", tok):
            continue
        out.append(tok)
        # Append comma sentinel *after* the word it follows, if any
        # comma was peeled. We approximate by checking the original raw.
        if raw.rstrip(".?!").endswith(","):
            out.append(",")
    return out, end_punct


# ---------------------------------------------------------------------------
# Sub-token expanders
# ---------------------------------------------------------------------------


def _make_eng_expander() -> Callable[[str], list[str]]:
    from g2p_en import G2p
    g2p = G2p()
    cache: dict[str, list[str]] = {}

    def expand(lemma: str) -> list[str]:
        cached = cache.get(lemma)
        if cached is not None:
            return cached
        try:
            phs = g2p(lemma)
        except Exception:
            phs = []
        out: list[str] = []
        for p in phs:
            if not p:
                continue
            # g2p returns spaces between words and punctuation tokens for
            # multi-word strings; we only ever pass single lemmas, but
            # filter defensively.
            if not p[0].isalpha():
                continue
            ph = re.sub(r"\d", "", p)
            if ph:
                out.append(f"en:{ph}")
        cache[lemma] = out
        return out

    return expand


def _expand_jp(word: str) -> list[str]:
    return [f"jp:{m}" for m in segment_moras(word)]


def _expand_zh(lemma: str) -> list[str]:
    out: list[str] = []
    for body, tone in _PINYIN_SYLL.findall(lemma):
        syll = body + tone
        if split_pinyin(syll) is None:
            continue
        out.append(f"zh:{syll}")
    return out


# ---------------------------------------------------------------------------
# Conv / Chunk dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Conv:
    cid: str   # path relative to ROOT, e.g. "Japanese/Asato/30520.cha"
    lang: str  # "en" / "jp" / "zh"
    tokens: list[tuple[str, str, float]]


@dataclass
class Chunk:
    conv: Conv
    idx: int  # chunk index within the conv
    rows: list[tuple[str, str, float]]


# ---------------------------------------------------------------------------
# Per-language conv loaders
# ---------------------------------------------------------------------------


def _build_convs(
    root: Path,
    tier_prefix: str,
    extract: Callable[[str], tuple[list[str], str]],
    expand: Callable[[str], list[str]],
    *,
    lang: str,
    min_tokens: int = MIN_TOKENS_PER_CONV,
) -> list[Conv]:
    convs: list[Conv] = []
    paths = sorted(root.rglob("*.cha"))
    for path in paths:
        try:
            lemma_stream = _parse_lemma_stream(path, tier_prefix, extract)
        except Exception as e:
            print(f"  skip {path}: {e}", file=sys.stderr)
            continue
        rows: list[tuple[str, str, float]] = []
        for speaker, lemma, dt in lemma_stream:
            sub = expand(lemma)
            for i, s in enumerate(sub):
                rows.append((speaker, s, dt if i == 0 else DT_INTRA_WORD))
        if len(rows) >= min_tokens:
            convs.append(Conv(
                cid=str(path.relative_to(root.parent)),
                lang=lang,
                tokens=rows,
            ))
    return convs


def load_eng() -> list[Conv]:
    expand = _make_eng_expander()
    return _build_convs(ENG_ROOT, "%mor:", _extract_eng_mor, expand, lang="en")


def load_jp() -> list[Conv]:
    return _build_convs(JP_ROOT, "%ort:", _extract_jp_ort, _expand_jp, lang="jp")


def load_zh() -> list[Conv]:
    convs = _build_convs(ZH_ROOT, "%mor:", _extract_zh_mor, _expand_zh, lang="zh")
    return _restrict_zh_convs(convs)


# ---------------------------------------------------------------------------
# Bottom-TTR filter
# ---------------------------------------------------------------------------


def filter_bottom_ttr(convs: list[Conv], frac: float) -> list[Conv]:
    if not convs or frac <= 0 or frac >= 1:
        return convs
    ttrs = []
    for c in convs:
        toks = [t[1] for t in c.tokens]
        ttrs.append(len(set(toks)) / max(1, len(toks)))
    order = np.argsort(np.array(ttrs))
    n_keep = max(1, int(len(convs) * frac))
    return [convs[i] for i in order[:n_keep]]


def maybe_filter_ttr(
    convs: list[Conv], target_tokens: int, frac: float, lang: str,
) -> list[Conv]:
    """Apply bottom-TTR filter unless doing so would leave us with <2× the
    target tokens for this language."""
    if frac <= 0:
        return convs
    if not convs:
        return convs
    total = sum(len(c.tokens) for c in convs)
    if total <= 0:
        return convs
    n_keep = max(1, int(len(convs) * frac))
    avg_tokens = total / max(1, len(convs))
    estimated_keep = avg_tokens * n_keep
    if estimated_keep < 2 * target_tokens:
        print(f"  [{lang}] skip TTR filter: filtered pool ≈ {estimated_keep:,.0f} "
              f"tokens < 2× target ({2 * target_tokens:,})")
        return convs
    filtered = filter_bottom_ttr(convs, frac)
    kept_total = sum(len(c.tokens) for c in filtered)
    print(f"  [{lang}] TTR filter: {len(convs)} → {len(filtered)} convs, "
          f"{total:,} → {kept_total:,} tokens")
    return filtered


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_conv(conv: Conv, target: int = CHUNK_TOKENS) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur: list[tuple[str, str, float]] = []
    prev_speaker: str | None = None
    for sp, tok, dt in conv.tokens:
        if (
            len(cur) >= target
            and prev_speaker is not None
            and sp != prev_speaker
        ):
            chunks.append(Chunk(conv=conv, idx=len(chunks), rows=cur))
            cur = []
        cur.append((sp, tok, dt))
        prev_speaker = sp
    if cur:
        chunks.append(Chunk(conv=conv, idx=len(chunks), rows=cur))
    return chunks


def chunk_all(convs: list[Conv]) -> list[Chunk]:
    out: list[Chunk] = []
    for c in convs:
        out.extend(chunk_conv(c))
    return out


def take_chunks(
    chunks: list[Chunk], target_tokens: int,
) -> tuple[list[Chunk], list[Chunk]]:
    """Take chunks from front until cumulative >= target. Caller is
    responsible for shuffling first if random selection is desired."""
    taken: list[Chunk] = []
    total = 0
    idx = 0
    while idx < len(chunks) and total < target_tokens:
        ch = chunks[idx]
        taken.append(ch)
        total += len(ch.rows)
        idx += 1
    return taken, chunks[idx:]


# ---------------------------------------------------------------------------
# Row emission
# ---------------------------------------------------------------------------


def _seq_id(tier: str, ch: Chunk) -> str:
    cid = Path(ch.conv.cid)
    stem = cid.stem
    parts = cid.parts  # e.g. ("Mandarin", "Tong", "010718.cha")
    corpus = parts[1] if len(parts) >= 2 else "?"
    return f"{tier}::{ch.conv.lang}::{corpus}::{stem}::{ch.idx}"


def emit_hersh_rows(chunks: list[Chunk], word2id: dict[str, int]) -> list[dict]:
    rows: list[dict] = []
    for ch in chunks:
        seq_id = _seq_id("multilang", ch)
        for pos, (_speaker, tok, _dt) in enumerate(ch.rows):
            rows.append(dict(
                sequenceId=seq_id,
                itemPosition=pos,
                Whale="multilang::UNK",
                Coda=word2id[tok],
                Ornamentation=0,
                Synchrony=0,
                Duration=0.0,
                TimeDelta=-1.0,
                has_timestamps=0,
            ))
    return rows


def emit_dswp_rows(
    chunks: list[Chunk], mask_idx: set[int], word2id: dict[str, int],
) -> list[dict]:
    rows: list[dict] = []
    for i, ch in enumerate(chunks):
        seq_id = _seq_id("zh_dswp", ch)
        masked = i in mask_idx
        for pos, (speaker, tok, dt) in enumerate(ch.rows):
            if masked:
                rows.append(dict(
                    sequenceId=seq_id,
                    itemPosition=pos,
                    Whale="zh_dswp::UNK",
                    Coda=word2id[tok],
                    Ornamentation=0,
                    Synchrony=0,
                    Duration=0.0,
                    TimeDelta=-1.0,
                    has_timestamps=0,
                ))
            else:
                rows.append(dict(
                    sequenceId=seq_id,
                    itemPosition=pos,
                    Whale=f"zh_dswp::{speaker}",
                    Coda=word2id[tok],
                    Ornamentation=0,
                    Synchrony=0,
                    Duration=0.0,
                    TimeDelta=dt,
                    has_timestamps=1,
                ))
    return rows


def emit_birth_rows(chunks: list[Chunk], word2id: dict[str, int]) -> list[dict]:
    rows: list[dict] = []
    for ch in chunks:
        seq_id = _seq_id("zh_birth", ch)
        for pos, (speaker, tok, dt) in enumerate(ch.rows):
            rows.append(dict(
                sequenceId=seq_id,
                itemPosition=pos,
                Whale=f"zh_birth::{speaker}",
                Coda=word2id[tok],
                Ornamentation=0,
                Synchrony=0,
                Duration=0.0,
                TimeDelta=dt,
                has_timestamps=1,
            ))
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=39_000)
    ap.add_argument("--hersh-frac", type=float, default=0.62)
    ap.add_argument("--dswp-frac", type=float, default=0.23)
    ap.add_argument("--hersh-en-frac", type=float, default=1.0 / 3,
                    help="Fraction of Hersh tier to fill from English.")
    ap.add_argument("--hersh-jp-frac", type=float, default=1.0 / 3,
                    help="Fraction of Hersh tier to fill from Japanese.")
    ap.add_argument("--hersh-zh-frac", type=float, default=1.0 / 3,
                    help="Fraction of Hersh tier to fill from Mandarin. "
                         "(EN+JP+ZH should sum to 1.)")
    ap.add_argument("--bottom-ttr-pct", type=float, default=0.10)
    ap.add_argument("--dswp-mask-frac", type=float, default=0.5,
                    help="Fraction of DSWP-equiv chunks to mask at "
                         "sequence-level (Whale=UNK, TimeDelta=-1.0, "
                         "has_timestamps=0). Default 0.5 — mirrors the "
                         "v2-plan target where DSWP has partial missingness "
                         "(half-masked at sequence level), bringing total "
                         "has_timestamps==0 to ~73% (62% Hersh + 12% from "
                         "half of DSWP).")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--out-index", type=Path, default=OUT_INDEX)
    ap.add_argument("--cache", type=Path,
                    default=ROOT / "outputs" / "grammar" / "multilang_convs.pkl",
                    help="Pickle of pre-loaded per-language conv pools. "
                         "Set to '' to disable caching.")
    args = ap.parse_args(argv)

    target = args.target_tokens
    hersh_total = int(round(target * args.hersh_frac))
    dswp_total = int(round(target * args.dswp_frac))
    birth_total = target - hersh_total - dswp_total
    hersh_en = int(round(hersh_total * args.hersh_en_frac))
    hersh_jp = int(round(hersh_total * args.hersh_jp_frac))
    hersh_zh = hersh_total - hersh_en - hersh_jp

    print(f"target token shares: hersh={hersh_total} "
          f"(en={hersh_en} jp={hersh_jp} zh={hersh_zh}); "
          f"dswp={dswp_total}; birth={birth_total}")

    # Mandarin is now the clean-tier source AND the Hersh-ZH source — chunks
    # are partitioned disjointly downstream (Birth → DSWP → Hersh-ZH).
    zh_total_target = hersh_zh + dswp_total + birth_total

    cache_path: Path | None = args.cache if str(args.cache) else None
    if cache_path and cache_path.exists():
        print(f"loading per-language conv pools from cache {cache_path}…")
        with cache_path.open("rb") as fh:
            cached = pickle.load(fh)
        en_convs = cached["en"]
        jp_convs = cached["jp"]
        zh_convs = cached["zh"]
        # v1 caches were built without the {Tong,Zhou3,TCCM} ≤36mo restriction.
        # Reapply unconditionally so v1 pickles still work end-to-end.
        n_zh_pre = len(zh_convs)
        zh_convs = _restrict_zh_convs(zh_convs)
        if len(zh_convs) != n_zh_pre:
            print(f"  zh: restricted {n_zh_pre} → {len(zh_convs)} convs "
                  f"(Tong+Zhou3+TCCM ≤{ZH_MAX_AGE_MONTHS}mo)")
        for lang, convs in (("en", en_convs), ("jp", jp_convs), ("zh", zh_convs)):
            print(f"  {lang}: {len(convs)} convs, "
                  f"{sum(len(c.tokens) for c in convs):,} tokens")
    else:
        # 1. Load per-language convs (English last because g2p-en is slow).
        print("loading Mandarin…")
        zh_convs = load_zh()
        print(f"  zh: {len(zh_convs)} convs, "
              f"{sum(len(c.tokens) for c in zh_convs):,} tokens")
        print("loading Japanese…")
        jp_convs = load_jp()
        print(f"  jp: {len(jp_convs)} convs, "
              f"{sum(len(c.tokens) for c in jp_convs):,} tokens")
        print("loading English (g2p phonemes; this is slow)…")
        en_convs = load_eng()
        print(f"  en: {len(en_convs)} convs, "
              f"{sum(len(c.tokens) for c in en_convs):,} tokens")
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as fh:
                pickle.dump({"en": en_convs, "jp": jp_convs, "zh": zh_convs}, fh)
            print(f"  cached pools to {cache_path}")

    # 2. Bottom-TTR filter (skip if filtered pool would be < 2× target).
    print("applying bottom-TTR filter…")
    en_convs = maybe_filter_ttr(en_convs, hersh_en, args.bottom_ttr_pct, "en")
    jp_convs = maybe_filter_ttr(jp_convs, hersh_jp, args.bottom_ttr_pct, "jp")
    zh_convs = maybe_filter_ttr(zh_convs, zh_total_target,
                                args.bottom_ttr_pct, "zh")

    # 3. Chunk + shuffle per-language.
    rng = random.Random(args.seed)
    en_chunks = chunk_all(en_convs); rng.shuffle(en_chunks)
    jp_chunks = chunk_all(jp_convs); rng.shuffle(jp_chunks)
    zh_chunks = chunk_all(zh_convs); rng.shuffle(zh_chunks)
    print(f"chunks: en={len(en_chunks)}, jp={len(jp_chunks)}, zh={len(zh_chunks)}")

    # 4. Allocate Mandarin chunks across Birth → DSWP → Hersh-ZH (disjoint).
    #    JP feeds only the Hersh-JP portion. EN feeds only the Hersh-EN portion.
    birth_chunks, zh_remain = take_chunks(zh_chunks, birth_total)
    dswp_chunks, zh_remain = take_chunks(zh_remain, dswp_total)
    hersh_zh_chunks, _ = take_chunks(zh_remain, hersh_zh)
    hersh_en_chunks, _ = take_chunks(en_chunks, hersh_en)
    hersh_jp_chunks, _ = take_chunks(jp_chunks, hersh_jp)

    def _check(name: str, chunks: list[Chunk], target_tokens: int) -> None:
        got = sum(len(c.rows) for c in chunks)
        if got < target_tokens * 0.95:
            print(f"  WARNING: {name} undershot — got {got:,} / {target_tokens:,}")
        else:
            print(f"  {name}: {len(chunks)} chunks, {got:,} tokens "
                  f"(target {target_tokens:,})")

    _check("birth (zh)", birth_chunks, birth_total)
    _check("dswp  (zh)", dswp_chunks, dswp_total)
    _check("hersh.en", hersh_en_chunks, hersh_en)
    _check("hersh.jp", hersh_jp_chunks, hersh_jp)
    _check("hersh.zh", hersh_zh_chunks, hersh_zh)

    # 5. Shuffle Hersh chunk order across languages (single-language chunks,
    # mixed in sequence order).
    hersh_chunks = hersh_en_chunks + hersh_jp_chunks + hersh_zh_chunks
    rng.shuffle(hersh_chunks)

    # 6. DSWP sequence-level mask: by default half of DSWP chunks are masked
    # to mirror the v2 plan's "half of sequences UNK, half kept" allocation,
    # which lands total has_timestamps==0 around the 0.65–0.75 acceptance band.
    n_dswp = len(dswp_chunks)
    n_mask = int(round(n_dswp * args.dswp_mask_frac))
    mask_indices = set(rng.sample(range(n_dswp), n_mask)) if n_mask else set()
    if n_dswp:
        print(f"  dswp mask: {n_mask}/{n_dswp} chunks "
              f"(frac={args.dswp_mask_frac})")

    # 7. Build unified vocab from all sub-tokens that will be emitted.
    all_strs: list[str] = []
    for ch in hersh_chunks + dswp_chunks + birth_chunks:
        all_strs.extend(t[1] for t in ch.rows)
    counts = Counter(all_strs)
    # Sort by descending frequency, then alphabetic for determinism.
    vocab = [w for w, _ in sorted(counts.items(),
                                   key=lambda kv: (-kv[1], kv[0]))]
    word2id = {w: i for i, w in enumerate(vocab)}
    n_total = sum(counts.values())
    print(f"unified V={len(vocab)} on {n_total:,} tokens "
          f"(tokens/type={n_total / max(1, len(vocab)):.1f})")

    # Per-language V breakdown.
    by_lang = {"en": Counter(), "jp": Counter(), "zh": Counter()}
    for w, n in counts.items():
        prefix = w.split(":", 1)[0]
        if prefix in by_lang:
            by_lang[prefix][w] = n
    for lang in ("en", "jp", "zh"):
        c = by_lang[lang]
        print(f"  {lang}: V={len(c)}, N={sum(c.values()):,}")

    # 8. Emit rows.
    rows: list[dict] = []
    rows.extend(emit_hersh_rows(hersh_chunks, word2id))
    rows.extend(emit_dswp_rows(dswp_chunks, mask_indices, word2id))
    rows.extend(emit_birth_rows(birth_chunks, word2id))
    df = pd.DataFrame(rows)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    n_seq = df["sequenceId"].nunique()
    print(f"wrote {args.out_csv} ({len(df):,} rows, {n_seq} sequences)")

    pd.DataFrame(dict(word_id=range(len(vocab)),
                      word=vocab)).to_csv(args.out_index, index=False)
    print(f"wrote {args.out_index}")

    # 9. Acceptance summary (v2 bands — see docs/multilang_corpus_plan.md).
    print()
    print("acceptance summary:")
    print(f"  total tokens   : {len(df):,}        (want 38,500–39,500)")
    print(f"  V              : {len(vocab):,}     (want 500–800)")
    print(f"  has_ts==0 frac : {(df['has_timestamps'] == 0).mean():.3f}  "
          f"(want 0.65–0.75)")
    unk_frac = df["Whale"].str.endswith("::UNK").mean()
    print(f"  ::UNK frac     : {unk_frac:.3f}  (want 0.69–0.74)")
    n_intra = ((df["TimeDelta"] == 0.0) & (df["has_timestamps"] == 1)
               & (df["itemPosition"] > 0)).sum()
    print(f"  intra_word rows: {n_intra:,}    (want >100; sub-tokens of "
          f"multi-syllable lemmas)")
    if len(vocab):
        modal = max(counts.values()) / max(1, n_total)
        print(f"  modal share    : {modal:.1%} (want 2–6%)")
        sc = sorted(counts.values(), reverse=True)
        cum = np.cumsum(sc) / max(1, n_total)
        p90 = int(np.searchsorted(cum, 0.90) + 1)
        print(f"  p90 rank       : {p90}    (want 200–400)")
        print(f"  tokens/type    : {n_total / max(1, len(vocab)):.1f} "
              f"(want 50–80)")


if __name__ == "__main__":
    main()
