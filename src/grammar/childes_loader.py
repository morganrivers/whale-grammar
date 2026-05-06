"""
Parse CHILDES Eng-UK .cha files into a whale-dialogues-shaped CSV.

Input:  Eng-UK/<corpus>/<child>/<file>.cha   (3038 files)
Output: data/classified/childes_dialogues.csv  (~39k token rows, schema
        matches data/classified/whale_dialogues.csv so predict_kfold
        can ingest it unchanged)
        data/classified/childes_word_index.csv  (int -> word lemma)

Tokenization
------------
Per utterance we prefer the `%mor:` line's lemmas (so `going`/`went`/`gone`
collapse to `go`, `books` to `book`). Surface-form fallback when `%mor`
is missing. Words are lowercased, contractions are split on `~`, and
morphological suffixes after `-` or `&` are stripped (e.g.
`balloon-PL` -> `balloon`, `be&3S` -> `be`). CHILDES specials (`xxx`,
`yyy`, `www`, `&=laugh`, `&-uh`, `[...]` brackets) are dropped.

DT (TimeDelta) estimation
-------------------------
Whale data has TimeDelta in seconds; CHILDES has no inter-word timing
so we estimate it from punctuation + speaker switches:

    dt to next token =
        2.0 s if speaker changed
        1.0 s after `.` `?` `!`
        0.5 s after `,`
        0.3 s otherwise (intra-utterance uniform)

Sampling rule
-------------
Compute TTR (unique lemmas / total lemmas) per .cha conversation.
Weighted random sampling without replacement (weight = TTR) until the
cumulative token count >= 39 000 (matching the whale corpus).
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CHILDES_ROOT = ROOT / "Eng-UK"
OUT_CSV = ROOT / "data" / "classified" / "childes_dialogues.csv"
OUT_INDEX = ROOT / "data" / "classified" / "childes_word_index.csv"

TARGET_TOKENS = 39_000
SEED = 42

# Split each .cha file into ~CHUNK_TOKENS-long sub-sequences so the
# 3-fold CV has comparable sequence granularity to the whale corpus
# (488 sequences / 38,840 codas = ~80 tokens/seq). Chunks break at the
# next speaker switch after the soft target is reached.
CHUNK_TOKENS = 80

# DT scheme — see module docstring.
DT_SPEAKER_SWITCH = 2.0
DT_AFTER_TERMINAL = 1.0   # . ? !
DT_AFTER_COMMA = 0.5
DT_INTRA_UTT = 0.3

TERMINAL_PUNCT = {".", "?", "!"}


# ---------------------------------------------------------------------------
# .cha parsing
# ---------------------------------------------------------------------------


_SPEAKER_RE = re.compile(r"^\*([A-Z0-9]+):\s*(.*)$")


def _iter_utterances(path: Path) -> Iterator[tuple[str, str | None]]:
    """Yield (speaker, mor_line_or_None) per utterance.

    A `*XYZ:` line opens an utterance; subsequent tab-indented continuation
    lines extend it; the immediately-following `%mor:` line (with any
    continuation lines) is attached. Other `%xxx:` tier lines are ignored.
    """
    speaker: str | None = None
    mor: str | None = None
    in_mor = False

    def flush() -> tuple[str, str | None] | None:
        nonlocal speaker, mor, in_mor
        if speaker is None:
            return None
        out = (speaker, mor)
        speaker = None
        mor = None
        in_mor = False
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
            mor = raw[len("%mor:"):].strip()
            in_mor = True
        elif raw.startswith("%"):
            in_mor = False
        elif raw.startswith("@"):
            prev = flush()
            if prev is not None:
                yield prev
        elif raw.startswith("\t"):
            if in_mor and mor is not None:
                mor = mor + " " + raw.strip()
        else:
            in_mor = False
    last = flush()
    if last is not None:
        yield last


# ---------------------------------------------------------------------------
# %mor lemma extraction
# ---------------------------------------------------------------------------


def _strip_lemma(lemma: str) -> str:
    """`balloon-PL` -> `balloon`, `be&3S` -> `be`, lowercase."""
    lemma = re.split(r"[-&]", lemma, maxsplit=1)[0]
    return lemma.lower().strip()


def _parse_mor(mor: str) -> tuple[list[str], str]:
    """Returns (lemmas, end_punct). Skips entries that aren't real words.

    Multi-morpheme entries (e.g. `pro:exist|here~cop|be&3S` for *here's*)
    are split on `~` so each clitic becomes its own token.
    """
    if not mor:
        return [], ""
    lemmas: list[str] = []
    end_punct = ""
    for entry in mor.split():
        if entry in TERMINAL_PUNCT:
            end_punct = entry
            continue
        if entry == ",":
            lemmas.append(",")  # sentinel; handled by caller for DT
            continue
        if "|" not in entry:
            continue  # untranscribed / annotation
        for sub in entry.split("~"):
            if "|" not in sub:
                continue
            _pos, lemma = sub.split("|", 1)
            lem = _strip_lemma(lemma)
            if not lem or not re.match(r"^[a-z']+$", lem):
                continue
            lemmas.append(lem)
    return lemmas, end_punct


def _parse_surface_fallback(text: str) -> list[str]:
    """Used when an utterance has no `%mor:` line. Strips CHILDES specials
    and keeps lowercase alphabetic words."""
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"&[=\-+][^\s]*", " ", text)
    text = re.sub(r"\b(?:xxx|yyy|www)\b", " ", text)
    text = re.sub(r"[<>]", " ", text)
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


# ---------------------------------------------------------------------------
# .cha -> token stream
# ---------------------------------------------------------------------------


def parse_cha(path: Path) -> list[tuple[str, str, float]]:
    """Returns list of (speaker, lemma, dt_to_this_token) for one file.

    First token of file gets dt=0. Within an utterance, intra-token
    spacing is DT_INTRA_UTT. Comma sentinels (from %mor) bump the *next*
    token's DT to DT_AFTER_COMMA. Utterance-end punctuation sets the
    next utterance's first-token DT (or speaker switch overrides to
    DT_SPEAKER_SWITCH).
    """
    out: list[tuple[str, str, float]] = []
    prev_speaker: str | None = None
    prev_end_punct = ""

    for speaker, mor in _iter_utterances(path):
        if mor is not None:
            lemmas_with_commas, end_punct = _parse_mor(mor)
        else:
            lemmas_with_commas, end_punct = [], ""

        # split into actual lemmas + comma sentinels (commas affect dt of
        # the *next* lemma but are not themselves emitted)
        comma_pending = False
        first_in_utt = True
        emitted_in_utt = 0
        for tok in lemmas_with_commas:
            if tok == ",":
                comma_pending = True
                continue
            # determine dt for this token
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
            emitted_in_utt += 1

        if emitted_in_utt > 0:
            prev_speaker = speaker
            prev_end_punct = end_punct
    return out


# ---------------------------------------------------------------------------
# Sampling and CSV writing
# ---------------------------------------------------------------------------


def _conversation_stats(tokens: list[tuple[str, str, float]]) -> tuple[int, int, float]:
    n = len(tokens)
    if n == 0:
        return 0, 0, 0.0
    uniq = len({t[1] for t in tokens})
    return n, uniq, uniq / n


def _weighted_sample(
    rng: np.random.Generator,
    items: list,
    weights: np.ndarray,
    target_total: int,
    sizes: np.ndarray,
) -> list[int]:
    """Weighted sampling without replacement (Efraimidis-Spirakis):
    each item's key is u**(1/w); take items in descending key order
    until cumulative size >= target_total. Returns indices into items."""
    u = rng.random(len(items))
    keys = np.where(weights > 0, np.log(u) / weights, -np.inf)
    order = np.argsort(-keys)  # descending
    chosen: list[int] = []
    total = 0
    for idx in order:
        if total >= target_total:
            break
        chosen.append(int(idx))
        total += int(sizes[idx])
    return chosen


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--min-tokens-per-conv", type=int, default=20,
                   help="Drop conversations with fewer than this many lemmas.")
    p.add_argument("--vocab-cap", type=int, default=None,
                   help="If set, keep top (cap-1) lemmas by frequency and "
                        "map the rest to <unk> (id = cap-1). Default: no cap.")
    p.add_argument("--mirror-hersh-frac", type=float, default=0.0,
                   help="Fraction of tokens (0..1) to write as Hersh-style "
                        "NA: Whale=childes::UNK, TimeDelta=-1.0, "
                        "has_timestamps=0. Sequence-level selection until "
                        "cumulative tokens >= frac × total. Default 0.")
    p.add_argument("--out-csv", type=Path, default=OUT_CSV)
    p.add_argument("--out-index", type=Path, default=OUT_INDEX)
    p.add_argument("--root", type=Path, default=CHILDES_ROOT)
    args = p.parse_args(argv)

    cha_files = sorted(args.root.rglob("*.cha"))
    print(f"found {len(cha_files)} .cha files under {args.root}")

    convs: list[dict] = []
    for path in cha_files:
        try:
            toks = parse_cha(path)
        except Exception as e:
            print(f"  skip {path}: {e}")
            continue
        n, uniq, ttr = _conversation_stats(toks)
        if n < args.min_tokens_per_conv:
            continue
        convs.append(dict(
            path=str(path.relative_to(args.root.parent)),
            tokens=toks,
            n=n,
            n_unique=uniq,
            ttr=ttr,
        ))
    print(f"kept {len(convs)} conversations after min-token filter")

    sizes = np.array([c["n"] for c in convs])
    ttrs = np.array([c["ttr"] for c in convs])
    print(f"corpus stats — total tokens: {sizes.sum():,}; "
          f"mean conv length: {sizes.mean():.1f}; "
          f"TTR median: {np.median(ttrs):.3f}; "
          f"TTR p90: {np.quantile(ttrs, 0.9):.3f}")

    rng = np.random.default_rng(args.seed)
    chosen = _weighted_sample(rng, convs, ttrs, args.target_tokens, sizes)
    chosen_convs = [convs[i] for i in chosen]
    total = sum(c["n"] for c in chosen_convs)
    print(f"sampled {len(chosen_convs)} conversations, "
          f"{total:,} tokens (target {args.target_tokens:,})")

    # Build vocab from chosen tokens.
    all_words: list[str] = []
    for c in chosen_convs:
        all_words.extend(t[1] for t in c["tokens"])
    unique_words = sorted(set(all_words))
    if args.vocab_cap is not None and len(unique_words) >= args.vocab_cap:
        # Reserve the last id for <unk>; keep top (cap-1) by frequency.
        counts = Counter(all_words)
        kept = sorted(w for w, _ in counts.most_common(args.vocab_cap - 1))
        vocab = kept + ["<unk>"]
        word2id: dict[str, int] = {w: i for i, w in enumerate(kept)}
        unk_id = len(kept)
        n_unk_tokens = sum(1 for w in all_words if w not in word2id)
        print(f"vocab cap {args.vocab_cap}: kept {len(kept):,} lemmas + <unk>; "
              f"{n_unk_tokens:,}/{len(all_words):,} tokens "
              f"({n_unk_tokens / len(all_words):.1%}) mapped to <unk>; "
              f"dropped {len(unique_words) - len(kept):,} long-tail lemmas")
    else:
        vocab = unique_words
        word2id = {w: i for i, w in enumerate(vocab)}
        unk_id = None
        print(f"vocabulary size: {len(vocab):,} unique lemmas on "
              f"{len(all_words):,} tokens (TTR {len(vocab)/len(all_words):.4f})")

    # Build whale_dialogues-shaped CSV. Each .cha file is split into
    # ~CHUNK_TOKENS-long sub-sequences (breaking at the next speaker
    # switch after the soft target) so CV granularity matches the whale
    # corpus's ~80 tokens/sequence.
    all_chunks: list[tuple[str, list[tuple[str, str, float]]]] = []
    for ci, c in enumerate(chosen_convs):
        stem = Path(c["path"]).stem
        toks = c["tokens"]
        chunks: list[list[tuple[str, str, float]]] = []
        cur: list[tuple[str, str, float]] = []
        prev_speaker: str | None = None
        for sp, lem, dt in toks:
            if (
                len(cur) >= CHUNK_TOKENS
                and prev_speaker is not None
                and sp != prev_speaker
            ):
                chunks.append(cur)
                cur = []
            cur.append((sp, lem, dt))
            prev_speaker = sp
        if cur:
            chunks.append(cur)
        for chunk_idx, chunk in enumerate(chunks):
            seq_id = f"childes::{stem}::{ci}::{chunk_idx}"
            all_chunks.append((seq_id, chunk))

    # Hersh-mirror: pick sequences uniformly at random until cumulative
    # tokens >= mirror_hersh_frac × total. Selection happens at
    # sequence-level (not row-level) so the missingness pattern matches
    # whale, where Hersh missingness is a property of whole recordings.
    masked_ids: set[str] = set()
    if args.mirror_hersh_frac > 0:
        total_tokens_chunked = sum(len(c) for _, c in all_chunks)
        target = args.mirror_hersh_frac * total_tokens_chunked
        rng_mask = np.random.default_rng(args.seed + 1)
        order = rng_mask.permutation(len(all_chunks))
        cum = 0
        for i in order:
            if cum >= target:
                break
            masked_ids.add(all_chunks[i][0])
            cum += len(all_chunks[i][1])
        print(f"Hersh mirror: {len(masked_ids):,}/{len(all_chunks):,} "
              f"sequences masked, {cum:,}/{total_tokens_chunked:,} tokens "
              f"({cum / total_tokens_chunked:.1%}) — Whale=childes::UNK, "
              f"TimeDelta=-1.0, has_timestamps=0")

    rows = []
    for seq_id, chunk in all_chunks:
        is_masked = seq_id in masked_ids
        for pos, (speaker, lemma, dt) in enumerate(chunk):
            tok_id = word2id.get(lemma, unk_id)
            if is_masked:
                rows.append(dict(
                    sequenceId=seq_id,
                    itemPosition=pos,
                    Whale="childes::UNK",
                    Coda=tok_id,
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
                    Whale=f"childes::{speaker}",
                    Coda=tok_id,
                    Ornamentation=0,
                    Synchrony=0,
                    Duration=0.0,
                    TimeDelta=dt,
                    has_timestamps=1,
                ))
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(df):,} rows, "
          f"{df['sequenceId'].nunique()} sequences)")

    pd.DataFrame(dict(
        word_id=range(len(vocab)),
        word=vocab,
    )).to_csv(args.out_index, index=False)
    print(f"wrote {args.out_index}")


if __name__ == "__main__":
    main()
