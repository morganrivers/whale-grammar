"""Render the classified corpus as a human-readable dialogue script.

Reads `data/classified/codas_classified.csv` (produced by B_classify) and
writes `data/readable/whale_dialogues.txt`. This stage is fast — all the
heavy work was done by the classifier — so iterate on rendering without
re-running the segmenter.

Each coda becomes a token of the form `<rubato><letters><digit>` (the rubato
prefix is omitted when not applicable):

  rubato  = `/`, `-`, `\\` or empty (B_classify's `rubato` column)
  letters = base-26 encoding of `rhythm_class` (a..z then aa..zz). The
            full coda_type_gero21 vocabulary is encoded — distinguishes
            5R1 / 5R2 / 5R3 plus all discovered Pacific types like
            5RP1, 5P2. Lowercase = unornamented or unknown, uppercase
            = extra_click == 1.
  digit   = `tempo` column 1..5 (Sharma 2024 buckets, computed by B_classify
            from coda_duration_s).

The unified corpus has ~131 distinct rhythm_class values; the multi-
letter base-26 encoder handles that vocabulary while keeping tokens
compact and parseable. Decode integer codes via
``data/classified/rhythm_class_index.csv``.

Between adjacent codas a `Δt<seconds>` token marks the inter-coda
interval (e.g. `Δt0.42`). When timestamps are missing — every Hersh
recording, ~57 % of DSWP — the sentinel `Δt?` is used instead. Pauses
exceeding ``PAUSE_NOTATION_S`` (10 s) override the Δt token with a
human-readable pause annotation on its own line.

Codas without a rhythm classification render as `?`. Codas with rhythm
but no ornament label (extra_click is <NA>, e.g. Hersh rows lacking
timing) render lowercase, since uppercase would be a false positive
claim of ornamentation.

Output layout: one section per (source, recording_or_date_group). Within
each group, codas are ordered by `time_in_recording_s` if present,
otherwise by `source_coda_id`. Speakers are labelled by `local_speaker_id`
if present, then `whale_photo_id`, else `?`.

Run standalone:
    python -m src.pipeline.C_render_readable
"""
from pathlib import Path

import pandas as pd

from . import B_classify

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "readable" / "whale_dialogues.txt"

PAUSE_NOTATION_S = 10.0


def _encode_rhythm(r: int) -> str:
    """Indices 0..25 → ``a..z``; indices 26..701 → ``aa..zz``; etc.
    Pacific extension can push the rhythm vocabulary well past 26 entries
    (we observe ~160 distinct labels on the unified corpus), and a
    multi-letter base-26 encoding stays compact and parseable while
    preserving the single-token convention."""
    if r < 0:
        return "?"
    out = []
    while True:
        out.append(chr(ord('a') + (r % 26)))
        r //= 26
        if r == 0:
            break
        r -= 1  # 'aa' = 26, not 27, so subtract before continuing.
    return "".join(reversed(out))


def coda_token(rhythm_class, extra_click, tempo, rubato) -> str:
    if pd.isna(rhythm_class) or pd.isna(tempo):
        return "?"
    letters = _encode_rhythm(int(rhythm_class))
    if pd.notna(extra_click) and int(extra_click) == 1:
        letters = letters.upper()
    prefix = str(rubato) if pd.notna(rubato) and rubato in ("/", "-", "\\") else ""
    return f"{prefix}{letters}{int(tempo)}"


def dt_token(dt_seconds: float | None) -> str:
    """Inter-coda time delta token. ``None`` (or non-finite) → ``Δt?``;
    otherwise ``Δt{dt:.2f}`` clipped at zero."""
    if dt_seconds is None:
        return "Δt?"
    if not (isinstance(dt_seconds, float) or isinstance(dt_seconds, int)):
        return "Δt?"
    if pd.isna(dt_seconds):
        return "Δt?"
    return f"Δt{max(float(dt_seconds), 0.0):.2f}"


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
    is_first = True

    def flush():
        if buf:
            lines.append(f"  Whale {last_speaker}: " + " ".join(buf))
            buf.clear()

    for _, row in rows.iterrows():
        token = coda_token(row.get("rhythm_class"), row.get("extra_click"),
                           row.get("tempo"), row.get("rubato"))
        speaker = speaker_label(row)
        t = row.get("time_in_recording_s") if has_time else None

        # Δt or pause annotation between this coda and the previous one.
        # Pauses (> PAUSE_NOTATION_S) suppress the Δt token because the
        # human-readable pause line already conveys the gap.
        if not is_first:
            dt_known = (has_time and last_t is not None and pd.notna(t))
            if dt_known and (float(t) - last_t) > PAUSE_NOTATION_S:
                flush()
                lines.append(f"  {format_pause(float(t) - last_t)}")
                last_speaker = None
            else:
                if speaker != last_speaker:
                    flush()
                    last_speaker = speaker
                buf.append(dt_token(float(t) - last_t if dt_known else None))

        if speaker != last_speaker:
            flush()
            last_speaker = speaker

        buf.append(token)
        if has_time and pd.notna(t):
            last_t = float(t)
        is_first = False
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
    n_class = int(df["rhythm_class"].notna().sum())
    print(f"C_render_readable: wrote {n_groups:,} groups, "
          f"{n_class:,}/{len(df):,} codas with rhythm_class tokens "
          f"-> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
