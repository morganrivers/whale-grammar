"""Render the classified corpus as a transformer-ready CSV.

Reads ``data/classified/codas_classified.csv`` (Stage 2 output) and
writes ``data/classified/whale_dialogues.csv`` in the whale-gpt-style
6-column schema plus a row-level data-quality flag:

    sequenceId, itemPosition,
    Whale, Coda, Ornamentation, Synchrony, Duration, TimeDelta,
    has_timestamps

Per row:

* ``sequenceId`` — one per ``(source, recording_id)`` group. **No
  within-recording sub-split** — sequence = whole recording, matching
  whale-gpt's convention. Long quiet pauses surface as large
  ``TimeDelta`` values inside a sequence.
* ``itemPosition`` — 0-based position within the sequence (sorted by
  ``time_in_recording_s`` when populated, else ``source_coda_id``).
* ``Whale`` — speaker identifier as a stable global string. Built from
  ``whale_photo_id`` if present, else ``local_speaker_id``, else
  ``UNK``. Always namespaced by source so DSWP-speaker-1 and
  birth-speaker-1 never collide. The integer encoding is persisted
  to ``data/classified/whale_id_index.csv``.
* ``Coda`` — ``rhythm_class`` integer. Codas with no ``rhythm_class``
  emit ``SILENCE_CODE`` (98).
* ``Ornamentation`` — ``extra_click``; NA → 0 (Hersh).
* ``Synchrony`` — 1 if a coda from a *different* whale starts within
  ``SIMULTANEOUS_THRESHOLD_S`` (0.3 s) of this row's start; 0 otherwise.
  Always 0 for Hersh (no whale ID, no timestamps).
* ``Duration`` — ``coda_duration_s``.
* ``TimeDelta`` — seconds since the previous primary coda's start in
  the same sequence. ``DELTATIME_MISSING`` (-1) when timestamps
  unavailable.
* ``has_timestamps`` — 1 iff this row's source publishes per-coda
  ``time_in_recording_s``. The model uses this to gate Synchrony and
  TimeDelta interpretation: when ``has_timestamps == 0`` both columns
  are uninformative.

The integer-code mapping for ``Coda`` is in
``data/classified/rhythm_class_index.csv``; for ``Whale`` it is in
``data/classified/whale_id_index.csv`` (both written here).

Run standalone:
    python -m src.pipeline.E_render_csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import B_classify

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "classified" / "whale_dialogues.csv"
WHALE_INDEX = REPO / "data" / "classified" / "whale_id_index.csv"

SILENCE_CODE = 98
DELTATIME_MISSING = -1.0
SIMULTANEOUS_THRESHOLD_S = 0.3
WHALE_UNKNOWN = "UNK"


def _whale_id(row) -> str:
    """Source-namespaced speaker string. ``UNK`` when no ID is published.

    Examples:
        ``sharma2024_dswp::photo:5563``
        ``sharma2025_birth::local:42``
        ``hersh2022_pacific::UNK``
    """
    src = row.get("source")
    src_str = "" if pd.isna(src) else str(src)
    if pd.notna(row.get("whale_photo_id")):
        return f"{src_str}::photo:{row['whale_photo_id']}"
    if pd.notna(row.get("local_speaker_id")):
        return f"{src_str}::local:{row['local_speaker_id']}"
    return f"{src_str}::{WHALE_UNKNOWN}"


def _coda_code(row) -> int:
    rc = row.get("rhythm_class")
    if pd.isna(rc):
        return SILENCE_CODE
    return int(rc)


def _ornamentation(row) -> int:
    e = row.get("extra_click")
    if pd.isna(e):
        return 0
    return int(e)


def _duration(row) -> float:
    d = row.get("coda_duration_s")
    if pd.isna(d):
        return 0.0
    return float(d)


def _synchrony(primary_idx: int,
               primary_time: float,
               primary_whale: str,
               group: pd.DataFrame) -> int:
    """1 iff *some* coda from a different whale starts within
    ``SIMULTANEOUS_THRESHOLD_S`` of this row. Requires both timestamps
    and a known speaker — Hersh rows always return 0."""
    if not np.isfinite(primary_time):
        return 0
    if primary_whale.endswith(WHALE_UNKNOWN):
        return 0
    times = group["time_in_recording_s"].to_numpy(dtype=float)
    whales = group["_whale"].to_numpy()
    finite = np.isfinite(times)
    other_known = whales != primary_whale
    not_self = group.index != primary_idx
    nearby = np.abs(times - primary_time) <= SIMULTANEOUS_THRESHOLD_S
    return int(bool((finite & other_known & not_self & nearby).any()))


def _seq_label(src: str, rec) -> str:
    rec_str = "no_recording" if pd.isna(rec) else str(rec)
    return f"{src}::{rec_str}"


def render(df: pd.DataFrame) -> pd.DataFrame:
    """Build the transformer-ready DataFrame from
    ``codas_classified.csv``-shape input."""
    out_rows: list[dict] = []
    df = df.copy()
    df["_whale"] = df.apply(_whale_id, axis=1)

    for (src, rec), grp in df.groupby(
            ["source", "recording_id"], dropna=False, sort=True):
        if grp["time_in_recording_s"].notna().any():
            grp = grp.sort_values(
                ["time_in_recording_s", "source_coda_id"],
                kind="stable", na_position="last")
        else:
            grp = grp.sort_values("source_coda_id", kind="stable")
        grp = grp.reset_index(drop=False)  # keep original index in `index`

        seq_id = _seq_label(src, rec)
        last_t = float("nan")

        for item_pos, (_, row) in enumerate(grp.iterrows()):
            t = row.get("time_in_recording_s")
            t = float(t) if pd.notna(t) else float("nan")
            has_ts = int(np.isfinite(t))

            if has_ts and np.isfinite(last_t):
                dt = t - last_t
            elif has_ts and not np.isfinite(last_t):
                dt = 0.0  # first coda in sequence with timestamps
            else:
                dt = DELTATIME_MISSING

            sync = _synchrony(
                primary_idx=int(row["index"]),
                primary_time=t,
                primary_whale=row["_whale"],
                group=grp,
            ) if has_ts else 0

            out_rows.append({
                "sequenceId": seq_id,
                "itemPosition": item_pos,
                "Whale": row["_whale"],
                "Coda": _coda_code(row),
                "Ornamentation": _ornamentation(row),
                "Synchrony": sync,
                "Duration": _duration(row),
                "TimeDelta": dt,
                "has_timestamps": has_ts,
            })
            if has_ts:
                last_t = t

    return pd.DataFrame(out_rows)


def _build_whale_index(out_df: pd.DataFrame) -> pd.DataFrame:
    """Stable string→int mapping for ``Whale``. UNK gets id 0; everyone
    else is sorted alphabetically and assigned 1..N."""
    uniq = out_df["Whale"].unique().tolist()
    unk = [w for w in uniq if w.endswith(WHALE_UNKNOWN)]
    known = sorted(w for w in uniq if not w.endswith(WHALE_UNKNOWN))
    rows = []
    next_id = 0
    for w in unk:
        rows.append({"whale_id": next_id, "Whale": w})
        next_id += 1
    for w in known:
        rows.append({"whale_id": next_id, "Whale": w})
        next_id += 1
    return pd.DataFrame(rows, columns=["whale_id", "Whale"])


def _load_classified() -> pd.DataFrame:
    if not B_classify.CLASSIFIED_CSV.exists():
        raise FileNotFoundError(
            f"{B_classify.CLASSIFIED_CSV.relative_to(REPO)} not found. "
            f"Run `python -m src.pipeline.B_classify` first."
        )
    return pd.read_csv(B_classify.CLASSIFIED_CSV, low_memory=False)


def main() -> pd.DataFrame:
    df = _load_classified()
    out = render(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    whale_idx = _build_whale_index(out)
    whale_idx.to_csv(WHALE_INDEX, index=False)
    n_seq = out["sequenceId"].nunique()
    n_whales = whale_idx["whale_id"].nunique()
    print(f"E_render_csv: wrote {len(out):,} rows ({n_seq} sequences, "
          f"{n_whales} distinct Whale ids incl. UNK) -> "
          f"{OUT.relative_to(REPO)}")
    return out


if __name__ == "__main__":
    main()
