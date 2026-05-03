"""Render the classified corpus as a human-readable dialogue script.

Reads `data/classified/codas_classified.csv` (produced by B_classify) and
writes `data/readable/whale_dialogues.txt`. This stage is fast — all the
heavy work was done by the classifier — so iterate on rendering without
re-running the segmenter.

Each coda becomes a token of the form `<letter><digit>`:

  letter = chr('a' + rhythm)   (lowercase = unornamented or unknown,
                                uppercase = extra_click==1)
  digit  = `tempo` column 1..5 (Sharma 2024 buckets, computed by B_classify
                                from coda_duration_s).

Codas without a rhythm classification render as `?`. Codas with rhythm but
no ornament label (extra_click is <NA>, e.g. DSWP/Hersh rows lacking
timing) render lowercase, since uppercase would be a false positive claim
of ornamentation.

Output layout: one section per (source, recording_or_date_group). Within
each group, codas are ordered by `time_in_recording_s` if present,
otherwise by `source_coda_id`. Speakers are labelled by `local_speaker_id`
if present, then `whale_photo_id`, else `?`. Long pauses within timed
groups are annotated.

Run standalone:
    python -m src.pipeline.C_render_readable
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import B_classify

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "readable" / "whale_dialogues.txt"

PAUSE_NOTATION_S = 10.0


def coda_token(rhythm, extra_click, tempo) -> str:
    if pd.isna(rhythm) or pd.isna(tempo):
        return "?"
    letter = chr(ord('a') + int(rhythm))
    if pd.notna(extra_click) and int(extra_click) == 1:
        letter = letter.upper()
    return f"{letter}{int(tempo)}"


def speaker_label(row) -> str:
    if pd.notna(row.get("local_speaker_id")):
        return str(row["local_speaker_id"])
    if pd.notna(row.get("whale_photo_id")):
        return str(row["whale_photo_id"])
    return "?"


def group_key(row):
    """Tightest available grouping key per coda, encoded as `kind|value`."""
    if pd.notna(row.get("recording_id")):
        return f"rec|{row['recording_id']}"
    if pd.notna(row.get("date")):
        return f"date|{row['date']}"
    return "source_only|"


def _label_for(group_key_str: str) -> str:
    kind, _, val = group_key_str.partition("|")
    if kind == "rec":
        return f"Recording {val}"
    if kind == "date":
        return f"Date {val}"
    return "Unattributed"


def format_pause(seconds: float) -> str:
    if seconds < 60:
        return f"(pause {int(seconds)} s)"
    if seconds < 3600:
        return f"(pause {int(seconds // 60)} min)"
    if seconds < 86400:
        return f"(pause {int(seconds // 3600)} h)"
    return f"(pause {int(seconds // 86400)} d)"


def render_group(rows, has_time, lines):
    last_speaker = None
    last_t = None
    buf = []

    def flush():
        if buf:
            lines.append(f"  Whale {last_speaker}: " + " ".join(buf))
            buf.clear()

    for _, row in rows.iterrows():
        token = coda_token(row.get("rhythm"), row.get("extra_click"), row.get("tempo"))
        speaker = speaker_label(row)
        t = row.get("time_in_recording_s") if has_time else None

        if has_time and last_t is not None and pd.notna(t) and (t - last_t) > PAUSE_NOTATION_S:
            flush()
            lines.append(f"  {format_pause(float(t - last_t))}")
            last_speaker = None

        if speaker != last_speaker:
            flush()
            last_speaker = speaker

        buf.append(token)
        if has_time and pd.notna(t):
            last_t = float(t)
    flush()


def render(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    for source, src_df in df.groupby("source", sort=True):
        lines.append(f"=== source: {source} ===")
        lines.append("")

        keys = src_df.apply(group_key, axis=1)
        src_df = src_df.assign(_grp=keys)
        for grp_val, grp_df in src_df.groupby("_grp", sort=True):
            has_time = grp_df["time_in_recording_s"].notna().any()
            if has_time:
                grp_df = grp_df.sort_values(
                    ["time_in_recording_s", "source_coda_id"],
                    na_position="last", kind="stable")
            else:
                grp_df = grp_df.sort_values("source_coda_id", kind="stable")

            n = len(grp_df)
            n_classified = int(grp_df["rhythm"].notna().sum())
            lines.append(f"-- {_label_for(grp_val)} ({n} codas, {n_classified} classified) --")
            render_group(grp_df, has_time, lines)
            lines.append("")
        lines.append("")

    out_path.write_text("\n".join(lines))


def _load_classified() -> pd.DataFrame:
    if not B_classify.CLASSIFIED_CSV.exists():
        raise FileNotFoundError(
            f"{B_classify.CLASSIFIED_CSV.relative_to(REPO)} not found. "
            f"Run `python -m src.pipeline.B_classify` first."
        )
    return pd.read_csv(B_classify.CLASSIFIED_CSV, low_memory=False)


def main():
    df = _load_classified()
    render(df, OUT)
    keys = df.apply(group_key, axis=1)
    n_groups = df.assign(_g=keys).groupby(["source", "_g"]).ngroups
    n_class = int(df["rhythm"].notna().sum())
    print(f"C_render_readable: wrote {n_groups:,} groups, "
          f"{n_class:,}/{len(df):,} codas with rhythm tokens "
          f"-> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
