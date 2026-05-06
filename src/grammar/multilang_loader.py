"""
Mandarin CHILDES corpus shaped to match whale_dialogues.csv.

Outputs:
    data/classified/multilang_dialogues.csv
    data/classified/multilang_word_index.csv

Tokenisation
------------
Each Mandarin pinyin lemma (e.g. `gang1cai2`) is split into full tonal
syllables — initial + final + tone — so each token is a single spoken
syllable: `gang1`, `cai2`. This mirrors whale's coda-as-spoken-event
tokenization (rhythm × tempo × ornament × rubato compound). Both
inventories land in the same V≈400–500 ballpark, which is what makes
the cross-corpus comparison apples-to-apples.

Source: %mor lines in CHILDES Mandarin .cha files. English glosses
(after `=`) are stripped; lemmas without a tone digit are dropped.

The source name is still `multilang` for backwards compatibility with
checkpoints, sequence-id prefixes, and the dt_buckets scheme — but the
loader is single-language Mandarin. EN+JP support was removed because
their phoneme/mora inventories had different statistics (~40 ARPABET
phonemes, ~100 moras) and inflated V via prefix sharding.

Three tiers (mirror of whale's hersh / dswp / birth split)
----------------------------------------------------------
Mandarin convs are restricted to {Tong, Zhou3, TCCM} with target child
≤36 months, then chunked into ~80-token sub-sequences. Chunks are
allocated disjointly:

- Birth-equiv (15%) : clean — speaker + estimated TimeDelta retained.
- DSWP-equiv  (23%) : half of sequences masked at sequence-level
                       (Whale=UNK, TimeDelta=-1.0, has_timestamps=0),
                       half retain speaker + TimeDelta.
- Hersh-equiv (62%) : all speaker stripped to UNK, all TimeDelta=-1.0,
                       has_timestamps=0. This is the "held-back" Mandarin
                       — same source pool as Birth+DSWP, but with timing
                       and identity erased, exactly parallel to whale
                       Hersh-Pacific being the same species held back
                       from the labelled Caribbean recordings.

DT estimation
-------------
0.3 s intra-utterance, 0.5 s after `,`, 1.0 s after `.` `?` `!`, 2.0 s
on speaker switch. The first sub-syllable of a multi-syllable lemma
inherits the lemma-level DT; subsequent sub-syllables get DT=0.0 so the
intra_word bucket fires (5-bucket scheme: intra_word / word_bound /
period / switch / missing).

Synchrony / Ornamentation / Duration are emitted as 0 — they have no
CHILDES analog (Synchrony is a per-utterance whale chorus marker).
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
)
from src.grammar.corpus_stats import _PINYIN_SYLL, split_pinyin

ROOT = Path(__file__).resolve().parents[2]
ZH_ROOT = ROOT / "Mandarin"

OUT_CSV = ROOT / "data" / "classified" / "multilang_dialogues.csv"
OUT_INDEX = ROOT / "data" / "classified" / "multilang_word_index.csv"

CHUNK_TOKENS = 80
SEED = 42
MIN_TOKENS_PER_CONV = 30

# Sub-syllables 2..N of a multi-syllable pinyin lemma get DT=0.0 so the
# model can recover word boundaries from timing on the clean tiers:
# intra-word transitions hit bucket 0 (intra_word), inter-word transitions
# hit bucket 1 (word_bound). The first syllable still inherits the lemma-
# level DT (DT_INTRA_UTT=0.3 between words, DT_AFTER_COMMA=0.5, etc.).
DT_INTRA_WORD = 0.0

# Clean tiers (Birth + DSWP) and the Hersh-equiv portion all draw from a
# small young-single-child pool, mirroring whale's clan-bounded vocabulary
# on the Caribbean DSWP+Birth recordings.
ZH_ALLOWED_CORPORA = {"Tong", "Zhou3", "TCCM"}
ZH_MAX_AGE_MONTHS = 36


# ---------------------------------------------------------------------------
# .cha parsing
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
    """Keep only Mandarin convs from {Tong, Zhou3, TCCM} with target child ≤36mo."""
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


def _iter_utts(path: Path) -> Iterator[tuple[str, str | None]]:
    """Yield (speaker, %mor body or None) per utterance.

    Tab-indented continuation lines on `%mor:` are concatenated.
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
        elif raw.startswith("%mor:"):
            body = raw[len("%mor:"):].strip()
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


def _parse_lemma_stream(path: Path) -> list[tuple[str, str, float]]:
    """Returns list of (speaker, lemma, dt) for one .cha file.

    Comma sentinels (`,`) bump the *next* lemma's dt to DT_AFTER_COMMA.
    """
    out: list[tuple[str, str, float]] = []
    prev_speaker: str | None = None
    prev_end_punct = ""

    for speaker, body in _iter_utts(path):
        if body is not None:
            lemmas, end_punct = _extract_zh_mor(body)
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


def _expand_zh(lemma: str) -> list[str]:
    """Split a pinyin lemma into validated tonal syllables (initial+final+tone)."""
    out: list[str] = []
    for body, tone in _PINYIN_SYLL.findall(lemma):
        syll = body + tone
        if split_pinyin(syll) is None:
            continue
        out.append(syll)
    return out


# ---------------------------------------------------------------------------
# Conv / Chunk dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Conv:
    cid: str   # path relative to ROOT, e.g. "Mandarin/Tong/010718.cha"
    tokens: list[tuple[str, str, float]]


@dataclass
class Chunk:
    conv: Conv
    idx: int  # chunk index within the conv
    rows: list[tuple[str, str, float]]


def load_zh(min_tokens: int = MIN_TOKENS_PER_CONV) -> list[Conv]:
    """Load + restrict + expand all Mandarin .cha files under ZH_ROOT."""
    convs: list[Conv] = []
    for path in sorted(ZH_ROOT.rglob("*.cha")):
        try:
            lemma_stream = _parse_lemma_stream(path)
        except Exception as e:
            print(f"  skip {path}: {e}", file=sys.stderr)
            continue
        rows: list[tuple[str, str, float]] = []
        for speaker, lemma, dt in lemma_stream:
            sub = _expand_zh(lemma)
            for i, s in enumerate(sub):
                rows.append((speaker, s, dt if i == 0 else DT_INTRA_WORD))
        if len(rows) >= min_tokens:
            convs.append(Conv(
                cid=str(path.relative_to(ZH_ROOT.parent)),
                tokens=rows,
            ))
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
    convs: list[Conv], target_tokens: int, frac: float,
) -> list[Conv]:
    """Apply bottom-TTR filter unless doing so would leave < 2× target tokens."""
    if frac <= 0 or not convs:
        return convs
    total = sum(len(c.tokens) for c in convs)
    if total <= 0:
        return convs
    n_keep = max(1, int(len(convs) * frac))
    avg_tokens = total / max(1, len(convs))
    estimated_keep = avg_tokens * n_keep
    if estimated_keep < 2 * target_tokens:
        print(f"  skip TTR filter: filtered pool ≈ {estimated_keep:,.0f} "
              f"tokens < 2× target ({2 * target_tokens:,})")
        return convs
    filtered = filter_bottom_ttr(convs, frac)
    kept_total = sum(len(c.tokens) for c in filtered)
    print(f"  TTR filter: {len(convs)} → {len(filtered)} convs, "
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
    return f"{tier}::{corpus}::{stem}::{ch.idx}"


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
    ap.add_argument("--bottom-ttr-pct", type=float, default=0.10)
    ap.add_argument("--dswp-mask-frac", type=float, default=0.5,
                    help="Fraction of DSWP-equiv chunks to mask at "
                         "sequence-level (Whale=UNK, TimeDelta=-1.0, "
                         "has_timestamps=0). With default 0.5 + Hersh-equiv "
                         "62%, total has_timestamps==0 lands ~0.73.")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--out-index", type=Path, default=OUT_INDEX)
    ap.add_argument("--cache", type=Path,
                    default=ROOT / "outputs" / "grammar" / "multilang_convs.pkl",
                    help="Pickle of pre-loaded Mandarin convs. "
                         "Set to '' to disable caching.")
    args = ap.parse_args(argv)

    target = args.target_tokens
    hersh_total = int(round(target * args.hersh_frac))
    dswp_total = int(round(target * args.dswp_frac))
    birth_total = target - hersh_total - dswp_total
    print(f"target token shares: hersh={hersh_total}; "
          f"dswp={dswp_total}; birth={birth_total}")

    # Mandarin pool feeds Birth → DSWP → Hersh-equiv (disjoint); Hersh-equiv
    # is just held back from the same Tong/Zhou3/TCCM pool, which mirrors
    # whale's Hersh-Pacific being held back from Caribbean Birth+DSWP.
    zh_total_target = hersh_total + dswp_total + birth_total

    cache_path: Path | None = args.cache if str(args.cache) else None
    if cache_path and cache_path.exists():
        print(f"loading Mandarin conv pool from cache {cache_path}…")
        with cache_path.open("rb") as fh:
            cached = pickle.load(fh)
        # v1/v2 caches stored a per-language dict; v3 stores a flat list.
        # Accept either, then re-restrict (caches predating the
        # {Tong,Zhou3,TCCM} ≤36mo restriction need it reapplied).
        if isinstance(cached, dict):
            zh_convs = cached.get("zh", [])
        else:
            zh_convs = cached
        # Convs cached under the v1/v2 dataclass had a `lang` field; the
        # v3 Conv dataclass dropped it. Coerce by extracting the fields
        # that still exist, and strip the legacy `zh:` token prefix.
        coerced: list[Conv] = []
        for c in zh_convs:
            cid = getattr(c, "cid", None)
            tokens = getattr(c, "tokens", None)
            if cid is None or tokens is None:
                continue
            stripped = [
                (sp, tok[3:] if tok.startswith("zh:") else tok, dt)
                for sp, tok, dt in tokens
            ]
            coerced.append(Conv(cid=cid, tokens=stripped))
        n_pre = len(coerced)
        zh_convs = _restrict_zh_convs(coerced)
        if len(zh_convs) != n_pre:
            print(f"  restricted {n_pre} → {len(zh_convs)} convs "
                  f"(Tong+Zhou3+TCCM ≤{ZH_MAX_AGE_MONTHS}mo)")
        print(f"  zh: {len(zh_convs)} convs, "
              f"{sum(len(c.tokens) for c in zh_convs):,} tokens")
    else:
        print("loading Mandarin…")
        zh_convs = load_zh()
        print(f"  zh: {len(zh_convs)} convs, "
              f"{sum(len(c.tokens) for c in zh_convs):,} tokens")
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as fh:
                pickle.dump(zh_convs, fh)
            print(f"  cached pool to {cache_path}")

    # Bottom-TTR filter (skip if filtered pool would be < 2× target).
    print("applying bottom-TTR filter…")
    zh_convs = maybe_filter_ttr(zh_convs, zh_total_target, args.bottom_ttr_pct)

    # Chunk + shuffle.
    rng = random.Random(args.seed)
    zh_chunks = chunk_all(zh_convs)
    rng.shuffle(zh_chunks)
    print(f"chunks: zh={len(zh_chunks)}")

    # Allocate disjointly: Birth → DSWP → Hersh-equiv.
    birth_chunks, zh_remain = take_chunks(zh_chunks, birth_total)
    dswp_chunks, zh_remain = take_chunks(zh_remain, dswp_total)
    hersh_chunks, _ = take_chunks(zh_remain, hersh_total)

    def _check(name: str, chunks: list[Chunk], target_tokens: int) -> None:
        got = sum(len(c.rows) for c in chunks)
        if got < target_tokens * 0.95:
            print(f"  WARNING: {name} undershot — got {got:,} / {target_tokens:,}")
        else:
            print(f"  {name}: {len(chunks)} chunks, {got:,} tokens "
                  f"(target {target_tokens:,})")

    _check("birth", birth_chunks, birth_total)
    _check("dswp ", dswp_chunks, dswp_total)
    _check("hersh", hersh_chunks, hersh_total)

    # DSWP sequence-level mask: half by default → total has_timestamps==0
    # lands around 0.73 (62% Hersh + 12% from half of DSWP).
    n_dswp = len(dswp_chunks)
    n_mask = int(round(n_dswp * args.dswp_mask_frac))
    mask_indices = set(rng.sample(range(n_dswp), n_mask)) if n_mask else set()
    if n_dswp:
        print(f"  dswp mask: {n_mask}/{n_dswp} chunks "
              f"(frac={args.dswp_mask_frac})")

    # Build vocab from emitted sub-syllables, sorted by descending freq
    # then alphabetic for determinism.
    all_strs: list[str] = []
    for ch in hersh_chunks + dswp_chunks + birth_chunks:
        all_strs.extend(t[1] for t in ch.rows)
    counts = Counter(all_strs)
    vocab = [w for w, _ in sorted(counts.items(),
                                  key=lambda kv: (-kv[1], kv[0]))]
    word2id = {w: i for i, w in enumerate(vocab)}
    n_total = sum(counts.values())
    print(f"V={len(vocab)} on {n_total:,} tokens "
          f"(tokens/type={n_total / max(1, len(vocab)):.1f})")

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

    # Acceptance summary — V band tightened to match whale (~467).
    print()
    print("acceptance summary:")
    print(f"  total tokens   : {len(df):,}        (want 38,500–39,500)")
    print(f"  V              : {len(vocab):,}     (want 350–550, vs whale ~467)")
    print(f"  has_ts==0 frac : {(df['has_timestamps'] == 0).mean():.3f}  "
          f"(want 0.65–0.75)")
    unk_frac = df["Whale"].str.endswith("::UNK").mean()
    print(f"  ::UNK frac     : {unk_frac:.3f}  (want 0.69–0.74)")
    n_intra = ((df["TimeDelta"] == 0.0) & (df["has_timestamps"] == 1)
               & (df["itemPosition"] > 0)).sum()
    print(f"  intra_word rows: {n_intra:,}    (want >100; sub-syllables of "
          f"multi-syllable lemmas)")
    if len(vocab):
        modal = max(counts.values()) / max(1, n_total)
        print(f"  modal share    : {modal:.1%} (want 2–6%)")
        sc = sorted(counts.values(), reverse=True)
        cum = np.cumsum(sc) / max(1, n_total)
        p90 = int(np.searchsorted(cum, 0.90) + 1)
        print(f"  p90 rank       : {p90}    (want 100–250)")
        print(f"  tokens/type    : {n_total / max(1, len(vocab)):.1f} "
              f"(want 70–110)")


if __name__ == "__main__":
    main()
