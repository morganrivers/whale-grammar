"""Render the classified corpus as a transformer-ready CSV.

Reads ``data/classified/codas_classified.csv`` (Stage 2 output) and
writes ``data/classified/whale_dialogues.csv`` in the schema whale-gpt's
training scripts already consume, plus one additive column:

    sequenceId, itemPosition,
    Coda1, Ornamentation1, Duration1,
    Coda2, Ornamentation2, Duration2,
    DeltaTime

Per row:

* ``sequenceId`` — one per ``(source, recording_id)`` group, with a
  further sub-split when consecutive primary codas are more than
  ``SEQUENCE_BREAK_S`` apart in time. Hersh recordings have no
  timestamps, so each Hersh recording yields exactly one sequence.
* ``itemPosition`` — 0-based position within the sequence.
* ``Coda1`` — ``rhythm_class`` integer. Codas with no
  ``rhythm_class`` (e.g. unclassified, out-of-range) emit
  ``SILENCE_CODE`` (98).
* ``Ornamentation1`` — ``extra_click``; NA → 0 (Hersh).
* ``Duration1`` — ``coda_duration_s``.
* ``Coda2`` — primary's nearest simultaneous coda from another whale
  in the same recording, where "simultaneous" means within
  ``SIMULTANEOUS_THRESHOLD_S`` (0.3 s) of the primary's start time.
  ``SILENCE_CODE`` when no such coda exists. Always
  ``SILENCE_CODE`` for sources without timestamps (Hersh).
* ``DeltaTime`` — seconds since the previous primary coda's start in
  the same sequence. ``DELTATIME_MISSING`` (-1) when timestamps
  unavailable.

The integer-code mapping is in
``data/classified/rhythm_class_index.csv`` (written by ``B_classify``).

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

SILENCE_CODE = 98
DELTATIME_MISSING = -1.0
SIMULTANEOUS_THRESHOLD_S = 0.3
SEQUENCE_BREAK_S = 60.0


def _whale_id(row) -> str:
    if pd.notna(row.get("local_speaker_id")):
        return f"local:{row['local_speaker_id']}"
    if pd.notna(row.get("whale_photo_id")):
        return f"photo:{row['whale_photo_id']}"
    return "?"


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


def _simultaneous_coda(primary_idx: int,
                       primary_time: float,
                       primary_whale: str,
                       group: pd.DataFrame) -> tuple[int, int, float]:
    """Find the nearest simultaneous coda from another whale within
    ``SIMULTANEOUS_THRESHOLD_S`` of ``primary_time``. Returns
    (coda_code, ornamentation, duration). Falls back to silence."""
    if not np.isfinite(primary_time):
        return SILENCE_CODE, 0, 0.0
    times = group["time_in_recording_s"].to_numpy(dtype=float)
    whales = group["_whale"].to_numpy()
    finite = np.isfinite(times)
    eligible = (finite
                & (whales != primary_whale)
                & (np.abs(times - primary_time) <= SIMULTANEOUS_THRESHOLD_S))
    eligible[group.index == primary_idx] = False
    if not eligible.any():
        return SILENCE_CODE, 0, 0.0
    candidates = group[eligible].copy()
    candidates["_dt"] = (
        candidates["time_in_recording_s"].astype(float) - primary_time).abs()
    nearest = candidates.sort_values("_dt").iloc[0]
    return (_coda_code(nearest), _ornamentation(nearest), _duration(nearest))


def _split_sequences(group: pd.DataFrame) -> np.ndarray:
    """Return per-row sequence-break indices within a recording. Each
    primary-coda gap > ``SEQUENCE_BREAK_S`` increments the index."""
    if not group["time_in_recording_s"].notna().any():
        return np.zeros(len(group), dtype=int)
    times = group["time_in_recording_s"].to_numpy(dtype=float)
    breaks = np.zeros(len(group), dtype=int)
    last_t = None
    idx = 0
    for i, t in enumerate(times):
        if pd.notna(t):
            if last_t is not None and (t - last_t) > SEQUENCE_BREAK_S:
                idx += 1
            last_t = t
        breaks[i] = idx
    return breaks


def render(df: pd.DataFrame) -> pd.DataFrame:
    """Build the transformer-ready DataFrame from
    ``codas_classified.csv``-shape input."""
    out_rows: list[dict] = []
    df = df.copy()
    df["_whale"] = df.apply(_whale_id, axis=1)

    seq_counter = 0
    for (src, rec), grp in df.groupby(
            ["source", "recording_id"], dropna=False, sort=True):
        if grp["time_in_recording_s"].notna().any():
            grp = grp.sort_values(
                ["time_in_recording_s", "source_coda_id"],
                kind="stable", na_position="last")
        else:
            grp = grp.sort_values("source_coda_id", kind="stable")
        grp = grp.reset_index(drop=False)  # keep original index in `index`

        sub_idx = _split_sequences(grp)
        last_seq_id = None
        last_t_per_seq: dict[int, float] = {}
        item_pos_per_seq: dict[int, int] = {}

        for i, row in grp.iterrows():
            seq_local = int(sub_idx[i])
            seq_id = (seq_counter, seq_local)
            if seq_id != last_seq_id:
                last_seq_id = seq_id
                if seq_id not in item_pos_per_seq:
                    item_pos_per_seq[seq_id] = 0
                    last_t_per_seq[seq_id] = float("nan")
            t = row.get("time_in_recording_s")
            t = float(t) if pd.notna(t) else float("nan")

            prev_t = last_t_per_seq.get(seq_id, float("nan"))
            if np.isfinite(t) and np.isfinite(prev_t):
                dt = t - prev_t
            else:
                dt = DELTATIME_MISSING

            coda1 = _coda_code(row)
            orn1 = _ornamentation(row)
            dur1 = _duration(row)

            if np.isfinite(t):
                coda2, orn2, dur2 = _simultaneous_coda(
                    primary_idx=int(row["index"]),
                    primary_time=t,
                    primary_whale=row["_whale"],
                    group=grp,
                )
            else:
                coda2, orn2, dur2 = SILENCE_CODE, 0, 0.0

            out_rows.append({
                "sequenceId": _seq_label(src, rec, seq_counter, seq_local),
                "itemPosition": item_pos_per_seq[seq_id],
                "Coda1": coda1,
                "Ornamentation1": orn1,
                "Duration1": dur1,
                "Coda2": coda2,
                "Ornamentation2": orn2,
                "Duration2": dur2,
                "DeltaTime": dt,
            })
            item_pos_per_seq[seq_id] += 1
            if np.isfinite(t):
                last_t_per_seq[seq_id] = t
        seq_counter += 1

    return pd.DataFrame(out_rows)


def _seq_label(src: str, rec, base: int, sub: int) -> str:
    rec_str = "no_recording" if pd.isna(rec) else str(rec)
    if sub == 0:
        return f"{src}::{rec_str}::{base}"
    return f"{src}::{rec_str}::{base}.{sub}"


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
    n_seq = out["sequenceId"].nunique()
    print(f"E_render_csv: wrote {len(out):,} rows ({n_seq} sequences) -> "
          f"{OUT.relative_to(REPO)}")
    return out


if __name__ == "__main__":
    main()
