"""
Render data/classified/childes_dialogues.csv as a CHILDES-style transcript.

The whale corpus has data/readable/whale_dialogues.txt — one line per
recording with `Δt` markers. CHILDES is natural language with explicit
speakers, so a `*SPEAKER:` per-utterance format is more legible.

Utterance breaks and punctuation are reconstructed from the inferred
TimeDelta channel (see childes_loader.py):

  TimeDelta == 2.0 s  → speaker switch (also ends utterance)
  TimeDelta == 1.0 s  → terminal punctuation (`.`)  ends utterance
  TimeDelta == 0.5 s  → comma before this token
  TimeDelta == 0.3 s  → intra-utterance, no punctuation

Output goes to data/readable/childes_dialogues.txt.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "classified" / "childes_dialogues.csv"
INDEX = ROOT / "data" / "classified" / "childes_word_index.csv"
OUT = ROOT / "data" / "readable" / "childes_dialogues.txt"


def _conv_key(sequence_id: str) -> str:
    """`childes::rhona0902::10::0` -> `rhona0902::10` (drop chunk index)."""
    parts = sequence_id.split("::")
    return "::".join(parts[1:-1]) if len(parts) >= 4 else sequence_id


def _strip_speaker(whale: str) -> str:
    """`childes::MOT` -> `MOT`."""
    return whale.split("::")[-1]


def render(df: pd.DataFrame, idx: pd.Series) -> str:
    """Walk rows in (sequenceId, itemPosition) order and emit a
    CHILDES-style transcript. Conversations are grouped by their
    parent .cha file (chunks stitched back together)."""
    df = df.sort_values(["sequenceId", "itemPosition"])
    df["conv"] = df["sequenceId"].map(_conv_key)
    df["speaker"] = df["Whale"].map(_strip_speaker)

    out: list[str] = []
    out.append("=== source: childes_uk ===")
    out.append("")

    for conv, g in df.groupby("conv", sort=False):
        n = len(g)
        speakers = sorted(g["speaker"].unique())
        out.append(f"-- Conversation {conv} ({n} lemmas, speakers: "
                   f"{', '.join(speakers)}) --")

        cur_speaker: str | None = None
        cur_words: list[str] = []
        cur_dts: list[float] = []

        def flush(end_punct: str = ".") -> None:
            if cur_speaker is None or not cur_words:
                return
            # Insert commas where DT==0.5 was observed *before* the
            # token (other than the first token of the utterance, which
            # carries the speaker-switch / utterance-boundary DT).
            tokens: list[str] = []
            for i, w in enumerate(cur_words):
                if i > 0 and cur_dts[i] == 0.5:
                    tokens.append(",")
                tokens.append(w)
            out.append(f"  *{cur_speaker}:\t{' '.join(tokens)} {end_punct}")

        for _, row in g.iterrows():
            speaker = row["speaker"]
            word = idx.get(int(row["Coda"]), "?")
            dt = float(row["TimeDelta"])
            if cur_speaker is None:
                cur_speaker = speaker
                cur_words = [word]
                cur_dts = [dt]
                continue
            if dt == 2.0 or speaker != cur_speaker:
                flush(".")
                cur_speaker = speaker
                cur_words = [word]
                cur_dts = [dt]
            elif dt == 1.0:
                flush(".")
                cur_words = [word]
                cur_dts = [dt]
            else:
                cur_words.append(word)
                cur_dts.append(dt)
        flush(".")
        out.append("")

    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=CSV)
    p.add_argument("--index", type=Path, default=INDEX)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    idx = pd.read_csv(args.index).set_index("word_id")["word"]

    text = render(df, idx)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out} ({len(text):,} chars, "
          f"{text.count(chr(10))} lines, "
          f"{df['sequenceId'].map(_conv_key).nunique()} conversations)")


if __name__ == "__main__":
    main()
