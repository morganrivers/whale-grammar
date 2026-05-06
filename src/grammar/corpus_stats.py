"""
Per-dimension token / vocab / entropy stats across the corpora we compare.

Usage
-----
    python -m src.grammar.corpus_stats --source dominica
    python -m src.grammar.corpus_stats --source whale-unified
    python -m src.grammar.corpus_stats --source childes-en
    python -m src.grammar.corpus_stats --source childes-zh
    python -m src.grammar.corpus_stats --source childes-jp
    python -m src.grammar.corpus_stats --all

Matched-budget comparison (filter conversations by lexical-diversity
percentile, then random-subsample to a token target on the "natural"
dimension):

    python -m src.grammar.corpus_stats --all --bottom-ttr-pct 0.10 \\
        --subsample 39000

For Dominica, point at the sw-combinatoriality checkout via
``--dominica-csv`` (or set SW_COMBINATORIALITY_DATA). The script will
auto-pick up ``whale_dialogues.txt`` next to the CSV to extract the
rubato-bearing compound + chorus marker.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Conversation primitive + frequency math
# ---------------------------------------------------------------------------


@dataclass
class Conversation:
    cid: str
    streams: dict[str, list]


@dataclass
class Corpus:
    name: str
    convs: list[Conversation]
    natural_dim: str  # which stream's TTR drives subsampling / filtering


def freq_stats(tokens: list, V_floor: int | None = None) -> dict:
    counts = Counter(tokens)
    n = len(tokens)
    V = V_floor if V_floor is not None else len(counts)
    if not n:
        return dict(counts=counts, n=0, V=V, H=0.0, top1=0.0, top10=0.0,
                    p90=0, modal=0.0)
    sc = sorted(counts.values(), reverse=True)
    H = -sum((c / n) * math.log2(c / n) for c in sc)
    cum = np.cumsum(sc) / n
    return dict(
        counts=counts, n=n, V=V, H=H,
        top1=sum(sc[: max(1, V // 100)]) / n,
        top10=sum(sc[: max(1, V // 10)]) / n,
        p90=int(np.searchsorted(cum, 0.90) + 1),
        modal=sc[0] / n,
    )


def fmt_table(rows: list[dict], cols: list[str]) -> str:
    hdr = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---:" if c != "dim" else "---" for c in cols]) + "|"
    out = [hdr, sep]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:.1%}" if c in {"top1", "top10", "modal"}
                             else f"{v:.2f}")
            elif isinstance(v, int):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Source: Dominica (sw-combinatoriality augmented CSV + dialogue txt)
# ---------------------------------------------------------------------------


def _dominica_csv_streams(df: pd.DataFrame) -> dict[str, list]:
    out: dict[str, list] = {}
    if "Rhythm" in df.columns:
        out["Rhythm"] = df["Rhythm"].astype(str).tolist()
    if "Tempo" in df.columns:
        out["Tempo"] = df["Tempo"].astype(str).tolist()
    if "Extra Click" in df.columns:
        out["Ornament"] = df["Extra Click"].astype(int).tolist()
    if "Clicks" in df.columns:
        out["nClicks"] = df["Clicks"].astype(int).tolist()
    if {"Rhythm", "Tempo", "Extra Click"}.issubset(df.columns):
        out["Compound r×t×o"] = [
            f"r{r}t{t}o{o}" for r, t, o in zip(
                df["Rhythm"], df["Tempo"], df["Extra Click"])
        ]
    if "ConstructedString" in df.columns:
        out["ConstructedString (paper compound)"] = (
            df["ConstructedString"].astype(str).tolist())
    return out


_RUBATO_TOKEN_RE = re.compile(r"[/\\\-]?[a-zA-Z]\d+")


def _parse_dialogue_txt(path: Path) -> list[Conversation]:
    """Parse whale_dialogues.txt — yields per-File conversations with
    'Compound + rubato' and 'Compound + rubato + chorus' streams."""
    text = path.read_text(encoding="utf-8", errors="replace")
    convs: list[Conversation] = []
    cid = "unknown"
    rub: list[str] = []
    rub_ch: list[str] = []

    def flush():
        if rub:
            convs.append(Conversation(cid=cid, streams={
                "Compound + rubato": list(rub),
                "Compound + rubato + chorus": list(rub_ch),
            }))

    for raw in text.splitlines():
        if raw.startswith("File:"):
            flush()
            cid = raw.split(":", 1)[1].strip()
            rub.clear()
            rub_ch.clear()
            continue
        is_chorus = "In chorus" in raw
        for tok in _RUBATO_TOKEN_RE.findall(raw):
            rub.append(tok)
            rub_ch.append(f"{tok}{'*' if is_chorus else ''}")
    flush()
    return convs


def load_dominica(csv_path: Path | None, txt_path: Path | None) -> Corpus:
    convs: list[Conversation] = []
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        if "File" in df.columns:
            for fid, g in df.groupby("File", sort=False):
                convs.append(Conversation(
                    cid=str(fid), streams=_dominica_csv_streams(g)))
        else:
            convs.append(Conversation(
                cid="all", streams=_dominica_csv_streams(df)))

    if txt_path and txt_path.exists():
        # Merge per-File rubato/chorus streams into matching CSV convs;
        # add as new convs if no match.
        for tc in _parse_dialogue_txt(txt_path):
            existing = next((c for c in convs if c.cid == tc.cid), None)
            if existing is not None:
                existing.streams.update(tc.streams)
            else:
                convs.append(tc)

    natural = ("Compound + rubato + chorus" if any(
        "Compound + rubato + chorus" in c.streams for c in convs)
               else "Compound r×t×o")
    return Corpus("Whale Dominica", convs, natural)


# ---------------------------------------------------------------------------
# Source: whale-grammar unified corpus
# ---------------------------------------------------------------------------


def load_whale_unified() -> Corpus:
    csv = ROOT / "data" / "classified" / "whale_dialogues.csv"
    if not csv.exists():
        return Corpus("Whale unified", [], "Coda")
    df = pd.read_csv(csv)
    df["source"] = df["sequenceId"].str.split("::").str[0]
    streams: dict[str, list] = {
        "Coda (rhythm class, V=131)": df["Coda"].astype(int).tolist(),
        "Whale id": df["Whale"].astype(str).tolist(),
        "Synchrony": df["Synchrony"].astype(int).tolist(),
        "Ornamentation": df["Ornamentation"].astype(int).tolist(),
        "source": df["source"].astype(str).tolist(),
    }
    try:
        from src.grammar.whale_compound import load_compound_whale
        cdf, _ = load_compound_whale(csv)
        streams["Compound (V≈467, with rubato)"] = (
            cdf["Coda"].astype(int).tolist())
    except Exception as e:
        print(f"[whale-unified] could not load compound: {e}")
    return Corpus(
        "Whale unified",
        [Conversation(cid="all", streams=streams)],
        "Coda (rhythm class, V=131)",
    )


def load_multilang() -> Corpus:
    csv = ROOT / "data" / "classified" / "multilang_dialogues.csv"
    index = ROOT / "data" / "classified" / "multilang_word_index.csv"
    if not csv.exists():
        return Corpus("Multilang CHILDES", [], "Sub-token (compound)")
    df = pd.read_csv(csv)
    df["tier"] = df["sequenceId"].str.split("::").str[0]
    decoded: list[str] = df["Coda"].astype(int).astype(str).tolist()
    if index.exists():
        idx = pd.read_csv(index)
        id2word = {int(r.word_id): str(r.word) for r in idx.itertuples()}
        decoded = [id2word.get(int(c), str(c)) for c in df["Coda"]]
    streams: dict[str, list] = {
        "Sub-token (compound)": decoded,
        "Whale id": df["Whale"].astype(str).tolist(),
        "tier": df["tier"].astype(str).tolist(),
    }
    return Corpus(
        "Multilang CHILDES",
        [Conversation(cid="all", streams=streams)],
        "Sub-token (compound)",
    )


# ---------------------------------------------------------------------------
# CHILDES helpers
# ---------------------------------------------------------------------------


_MOR_PREFIX = "%mor:"
_ORT_PREFIX = "%ort:"
_PUNCT_TOKENS = {".", "?", "!", ","}


def _iter_tier(root: Path, prefix: str) -> Iterable[tuple[str, list[str]]]:
    """Yield (conv_id, list-of-tier-bodies) per .cha file."""
    for path in sorted(root.rglob("*.cha")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        bodies = [raw[len(prefix):].strip() for raw in text.splitlines()
                  if raw.startswith(prefix)]
        if bodies:
            yield str(path.relative_to(root.parent)), bodies


def _extract_lemmas(bodies: list[str], pinyin: bool = False) -> list[str]:
    out: list[str] = []
    for body in bodies:
        for entry in body.split():
            if entry in _PUNCT_TOKENS or "|" not in entry:
                continue
            for sub in entry.split("~"):
                if "|" not in sub:
                    continue
                _pos, lem = sub.split("|", 1)
                lem = re.split(r"[-&=]", lem, maxsplit=1)[0].lower().strip()
                if not lem:
                    continue
                if pinyin:
                    if (not re.match(r"^[a-z\d]+$", lem)
                            or not re.search(r"\d", lem)):
                        continue
                else:
                    if not re.match(r"^[a-z']+$", lem):
                        continue
                out.append(lem)
    return out


# ---------------------------------------------------------------------------
# CHILDES Eng-UK
# ---------------------------------------------------------------------------


def load_childes_en() -> Corpus:
    root = ROOT / "Eng-UK"
    if not root.exists():
        return Corpus("CHILDES Eng-UK", [], "English lemmas")
    convs = []
    for cid, bodies in _iter_tier(root, _MOR_PREFIX):
        lemmas = _extract_lemmas(bodies, pinyin=False)
        if len(lemmas) >= 30:
            convs.append(Conversation(
                cid=cid, streams={"English lemmas": lemmas}))
    return Corpus("CHILDES Eng-UK", convs, "English lemmas")


# ---------------------------------------------------------------------------
# CHILDES Mandarin
# ---------------------------------------------------------------------------


_INITIALS_2 = {"zh", "ch", "sh"}
_INITIALS_1 = set("bpmfdtnlgkhjqxrzcs")
_PINYIN_SYLL = re.compile(r"([a-z]+)(\d)")


def split_pinyin(syll: str) -> tuple[str, str, str] | None:
    m = re.match(r"^([a-z]+)(\d)$", syll)
    if not m:
        return None
    body, tone = m.group(1), m.group(2)
    if len(body) >= 2 and body[:2] in _INITIALS_2:
        return body[:2], body[2:], tone
    if body[:1] in _INITIALS_1:
        return body[:1], body[1:], tone
    return "", body, tone


def load_childes_zh() -> Corpus:
    root = ROOT / "Mandarin"
    if not root.exists():
        return Corpus("CHILDES Mandarin", [], "Lemmas with tone")
    convs = []
    for cid, bodies in _iter_tier(root, _MOR_PREFIX):
        lemmas = _extract_lemmas(bodies, pinyin=True)
        sylls_t, sylls_nt, inits, fins, tones = [], [], [], [], []
        for l in lemmas:
            for s, t in _PINYIN_SYLL.findall(l):
                d = split_pinyin(s + t)
                if d is None:
                    continue
                init, fin, tn = d
                sylls_t.append(f"{init}{fin}{tn}")
                sylls_nt.append(f"{init}{fin}")
                inits.append(init or "∅")
                fins.append(fin)
                tones.append(tn)
        if len(lemmas) >= 30:
            convs.append(Conversation(cid=cid, streams={
                "Lemmas with tone": lemmas,
                "Lemmas tone-stripped": [re.sub(r"\d", "", l) for l in lemmas],
                "Syllables with tone": sylls_t,
                "Syllables tone-stripped": sylls_nt,
                "Initials": inits,
                "Finals": fins,
                "Tones": tones,
                "Initial × Tone": [f"{i}{t}" for i, t in zip(inits, tones)],
                "Final × Tone": [f"{f}{t}" for f, t in zip(fins, tones)],
            }))
    return Corpus("CHILDES Mandarin", convs, "Lemmas with tone")


# ---------------------------------------------------------------------------
# CHILDES Japanese (MiiPro) — segment %ort romaji into moras
# ---------------------------------------------------------------------------


_MORA_RE = re.compile(
    r"(kya|kyu|kyo|gya|gyu|gyo|sha|shi|shu|sho|cha|chi|chu|cho|"
    r"ja|ji|ju|jo|tsu|nya|nyu|nyo|hya|hyu|hyo|mya|myu|myo|"
    r"rya|ryu|ryo|bya|byu|byo|pya|pyu|pyo|fu|"
    r"[kgsztdnhbpmr][aeiou]|[wy][aeiou]|[aeiou]|n|"
    r"[kgsztdpmrh])",
    re.IGNORECASE,
)


def segment_moras(word: str) -> list[str]:
    """Romaji → list of moras. Handles digraphs (kya, sha…), single CV,
    standalone vowel, syllabic n, and orphan consonant (geminate)."""
    word = re.sub(r"[()]", "", word.lower())
    moras: list[str] = []
    i = 0
    while i < len(word):
        m = _MORA_RE.match(word, i)
        if m:
            moras.append(m.group(0))
            i = m.end()
        else:
            i += 1  # skip unparseable char
    return moras


def _parse_jp_ort_words(bodies: list[str]) -> list[str]:
    out: list[str] = []
    for body in bodies:
        b = re.sub(r"\[[^\]]*\]", " ", body)
        b = re.sub(r"&[~=\-+][^\s]*", " ", b)
        b = re.sub(r"\bxxx\b|\byyy\b|\bwww\b", " ", b)
        b = re.sub(r"@[a-z]+", " ", b)
        b = re.sub(r"[‡«»<>]", " ", b)
        for tok in b.split():
            tok = tok.strip(".,!?;:\"").lower()
            if not tok or tok in {".", "?", "!", ",", "0"}:
                continue
            if not re.match(r"^[a-z'()][a-z'()]*$", tok):
                continue
            out.append(tok)
    return out


def load_childes_jp() -> Corpus:
    root = ROOT / "Japanese"
    if not root.exists():
        return Corpus("CHILDES Japanese (MiiPro)", [], "Moras")
    convs = []
    for cid, bodies in _iter_tier(root, _ORT_PREFIX):
        words = _parse_jp_ort_words(bodies)
        moras = [m for w in words for m in segment_moras(w)]
        if len(moras) >= 30:
            convs.append(Conversation(cid=cid, streams={
                "Words (romaji surface)": words,
                "Moras": moras,
            }))
    return Corpus("CHILDES Japanese (MiiPro)", convs, "Moras")


# ---------------------------------------------------------------------------
# Filter / subsample
# ---------------------------------------------------------------------------


def filter_and_subsample(
    corpus: Corpus,
    bottom_ttr_pct: float | None = None,
    subsample: int | None = None,
    seed: int = 42,
) -> Corpus:
    convs = corpus.convs
    if not convs:
        return corpus
    natural = corpus.natural_dim

    if bottom_ttr_pct is not None and len(convs) > 1:
        ttrs = []
        for c in convs:
            toks = c.streams.get(natural, [])
            ttrs.append(len(set(toks)) / max(1, len(toks)))
        order = np.argsort(np.array(ttrs))
        n_keep = max(1, int(len(convs) * bottom_ttr_pct))
        convs = [convs[i] for i in order[:n_keep]]

    if subsample is not None and convs:
        rng = random.Random(seed)
        idx = list(range(len(convs)))
        rng.shuffle(idx)
        kept: list[Conversation] = []
        total = 0
        for i in idx:
            if total >= subsample:
                break
            kept.append(convs[i])
            total += len(convs[i].streams.get(natural, []))
        convs = kept

    return Corpus(corpus.name, convs, natural)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def aggregate_streams(convs: list[Conversation]) -> dict[str, list]:
    out: dict[str, list] = {}
    for c in convs:
        for dim, toks in c.streams.items():
            out.setdefault(dim, []).extend(toks)
    return out


def render(corpus: Corpus, top: int = 10) -> str:
    streams = aggregate_streams(corpus.convs)
    if not streams:
        return f"## {corpus.name}\n\n_No data found._\n"
    out = [f"## {corpus.name} ({len(corpus.convs)} conv(s))\n"]
    rows = []
    for dim, toks in streams.items():
        s = freq_stats(toks)
        rows.append({
            "dim": dim, "N": s["n"], "V": s["V"],
            "tokens/type": round(s["n"] / max(1, s["V"]), 1),
            "H bpt": s["H"], "modal": s["modal"],
            "top1": s["top1"], "top10": s["top10"],
            "p90 rank": s["p90"],
        })
    out.append(fmt_table(rows, ["dim", "N", "V", "tokens/type", "H bpt",
                                 "modal", "top1", "top10", "p90 rank"]))
    out.append("")
    if top > 0:
        for dim, toks in streams.items():
            s = freq_stats(toks)
            out.append(f"### {dim}")
            out.append(
                f"- N={s['n']:,}; V={s['V']:,}; H={s['H']:.2f} bpt; "
                f"tokens/type={s['n']/max(1,s['V']):.1f}; "
                f"modal={s['modal']:.1%}")
            most = s["counts"].most_common(top)
            out.append("- top: " + ", ".join(
                f"`{k}`×{v}" for k, v in most))
            out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


SOURCES = ["dominica", "whale-unified", "childes-en", "childes-zh",
           "childes-jp", "multilang"]


def load(source: str, dominica_csv: Path | None,
         whale_dialogues_txt: Path | None) -> Corpus:
    if source == "dominica":
        csv = dominica_csv
        if csv is None:
            env = os.environ.get("SW_COMBINATORIALITY_DATA")
            if env:
                csv = Path(env) / "sperm-whale-dialogues_augmented.csv"
        txt = whale_dialogues_txt
        if csv is not None and txt is None:
            cand = csv.parent / "whale_dialogues.txt"
            if cand.exists():
                txt = cand
        return load_dominica(csv, txt)
    if source == "whale-unified":
        return load_whale_unified()
    if source == "childes-en":
        return load_childes_en()
    if source == "childes-zh":
        return load_childes_zh()
    if source == "childes-jp":
        return load_childes_jp()
    if source == "multilang":
        return load_multilang()
    raise ValueError(source)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=SOURCES)
    p.add_argument("--all", action="store_true")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--bottom-ttr-pct", type=float, default=None,
                   help="Keep only the bottom-P fraction of conversations "
                        "by TTR on the natural dim (P in 0..1).")
    p.add_argument("--subsample", type=int, default=None,
                   help="Random-subsample to ≈N tokens of the natural dim.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dominica-csv", type=Path, default=None)
    p.add_argument("--whale-dialogues-txt", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.all and args.source is None:
        p.error("pass --source or --all")

    sources = SOURCES if args.all else [args.source]
    for src in sources:
        corpus = load(src, args.dominica_csv, args.whale_dialogues_txt)
        corpus = filter_and_subsample(
            corpus, args.bottom_ttr_pct, args.subsample, args.seed)
        print(render(corpus, top=args.top))
        print()


if __name__ == "__main__":
    main()
