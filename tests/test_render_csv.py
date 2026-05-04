"""Stage 4.2 acceptance gate — transformer CSV emission.

Validates ``data/classified/whale_dialogues.csv`` (produced by
``python -m src.pipeline.E_render_csv``):

* Schema matches the whale-gpt-style 6-column layout plus
  ``has_timestamps``.
* Hersh rows always have ``Synchrony = 0`` (no whale ID, no timestamps),
  ``TimeDelta = -1`` (no time deltas), ``Ornamentation = 0`` (Sharma §5
  rule needs timestamps), ``Whale`` ending in ``::UNK``, and
  ``has_timestamps = 0``.
* Sharma DSWP / birth rows have a meaningful share of non-``-1``
  ``TimeDelta`` values, given their timestamp coverage (43 % DSWP,
  100 % birth).
* ``Coda`` integers decode through ``rhythm_class_index.csv``.
* ``Whale`` strings decode through ``whale_id_index.csv``.
* ``itemPosition`` is contiguous (0..n-1) within each sequence.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
CSV_OUT = REPO / "data" / "classified" / "whale_dialogues.csv"
RHYTHM_INDEX = REPO / "data" / "classified" / "rhythm_class_index.csv"
WHALE_INDEX = REPO / "data" / "classified" / "whale_id_index.csv"

REQUIRED_COLUMNS = [
    "sequenceId", "itemPosition",
    "Whale", "Coda", "Ornamentation", "Synchrony", "Duration", "TimeDelta",
    "has_timestamps",
]
SILENCE_CODE = 98


def _load() -> pd.DataFrame:
    if not CSV_OUT.exists():
        pytest.skip(f"{CSV_OUT.relative_to(REPO)} not found; "
                    f"run `python -m src.pipeline.E_render_csv`")
    return pd.read_csv(CSV_OUT)


def _source(df: pd.DataFrame) -> pd.Series:
    return df["sequenceId"].str.split("::").str[0]


def test_schema_matches():
    df = _load()
    assert list(df.columns) == REQUIRED_COLUMNS, (
        f"got {list(df.columns)}, expected {REQUIRED_COLUMNS}"
    )


def test_hersh_rows_lack_time_features():
    df = _load()
    hersh = df[_source(df) == "hersh2022_pacific"]
    assert (hersh["Synchrony"] == 0).all(), (
        "Hersh has no whale ID + no timestamps; Synchrony must be 0"
    )
    assert (hersh["TimeDelta"] == -1.0).all(), (
        "Hersh TimeDelta should always be -1 sentinel"
    )
    assert (hersh["Ornamentation"] == 0).all(), (
        "Hersh Ornamentation should always be 0 (no per-coda timestamps)"
    )
    assert (hersh["has_timestamps"] == 0).all(), (
        "Hersh has_timestamps must be 0"
    )
    assert hersh["Whale"].str.endswith("::UNK").all(), (
        "Hersh Whale must always be the UNK sentinel"
    )


def test_dswp_and_birth_have_real_dt_values():
    df = _load()
    src = _source(df)
    for s in ("sharma2024_dswp", "sharma2025_birth"):
        sub = df[src == s]
        n_real_dt = int((sub["TimeDelta"] >= 0).sum())
        assert n_real_dt > 100, (
            f"{s}: only {n_real_dt} non-sentinel TimeDelta values"
        )


def test_birth_dt_coverage_near_full():
    """Birth has 100% timestamp coverage, so almost every row should
    have a real TimeDelta — first row of each sequence is dt=0.0
    (still has_timestamps=1, just no prior reference)."""
    df = _load()
    sub = df[_source(df) == "sharma2025_birth"]
    has_ts = int(sub["has_timestamps"].sum())
    assert has_ts >= 0.95 * len(sub), (
        f"birth has_timestamps coverage low: {has_ts}/{len(sub)}"
    )


def test_coda_codes_decode_through_index():
    df = _load()
    if not RHYTHM_INDEX.exists():
        pytest.skip(f"{RHYTHM_INDEX.relative_to(REPO)} not found")
    idx = pd.read_csv(RHYTHM_INDEX)
    valid = set(idx["rhythm_class"].astype(int).tolist()) | {SILENCE_CODE}
    unique = set(int(x) for x in df["Coda"].unique())
    unknown = unique - valid
    assert not unknown, f"Coda values not in rhythm_class_index: {unknown}"


def test_whale_strings_decode_through_index():
    df = _load()
    if not WHALE_INDEX.exists():
        pytest.skip(f"{WHALE_INDEX.relative_to(REPO)} not found")
    idx = pd.read_csv(WHALE_INDEX)
    valid = set(idx["Whale"].astype(str).tolist())
    unique = set(df["Whale"].astype(str).unique())
    unknown = unique - valid
    assert not unknown, f"Whale strings not in whale_id_index: {unknown}"


def test_item_position_contiguous_per_sequence():
    df = _load()
    for seq, sub in df.groupby("sequenceId"):
        positions = sub["itemPosition"].to_numpy()
        expected = list(range(len(sub)))
        assert positions.tolist() == expected, (
            f"sequence {seq}: itemPosition not contiguous"
        )


def test_synchrony_fires_in_birth():
    """Birth has 100 % timestamps + speaker IDs, so the 0.3 s
    simultaneity window should catch some chorus events."""
    df = _load()
    sub = df[_source(df) == "sharma2025_birth"]
    n_sync = int(sub["Synchrony"].sum())
    assert n_sync >= 50, f"only {n_sync} Synchrony=1 rows in birth"


def test_unk_whale_distribution():
    """``::UNK`` covers 100 % of Hersh (no IDs upstream), 0 % of birth
    (100 % local_speaker_id), and a partial fraction of DSWP (~38 %
    of rows lack both whale_photo_id and local_speaker_id)."""
    df = _load()
    src = _source(df)
    is_unk = df["Whale"].str.endswith("::UNK")

    hersh = df[src == "hersh2022_pacific"]
    assert is_unk[hersh.index].all(), "Hersh must be 100 % UNK"

    birth = df[src == "sharma2025_birth"]
    assert not is_unk[birth.index].any(), (
        "Birth has 100 % local_speaker_id; should never be UNK"
    )

    dswp = df[src == "sharma2024_dswp"]
    dswp_unk_rate = float(is_unk[dswp.index].mean())
    assert 0.05 < dswp_unk_rate < 0.6, (
        f"DSWP UNK rate {dswp_unk_rate:.1%} outside the expected band "
        f"(some IDs missing upstream, but most populated)"
    )
