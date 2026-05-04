"""Stage 4.2 acceptance gate — transformer CSV emission.

Validates ``data/classified/whale_dialogues.csv`` (produced by
``python -m src.pipeline.E_render_csv``):

* Schema matches whale-gpt's expected columns plus ``DeltaTime``.
* Hersh rows always have ``Coda2 = 98`` (no timestamps to detect
  simultaneity), ``DeltaTime = -1`` (no time deltas), and
  ``Ornamentation1 = 0`` (Sharma §5 rule needs timestamps).
* Sharma DSWP / birth rows have a meaningful share of non-``-1``
  ``DeltaTime`` values, given their timestamp coverage (43 % DSWP,
  100 % birth).
* ``Coda1`` integers decode through ``rhythm_class_index.csv``.
* ``itemPosition`` is contiguous (0..n-1) within each sequence.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
CSV_OUT = REPO / "data" / "classified" / "whale_dialogues.csv"
RHYTHM_INDEX = REPO / "data" / "classified" / "rhythm_class_index.csv"

REQUIRED_COLUMNS = [
    "sequenceId", "itemPosition",
    "Coda1", "Ornamentation1", "Duration1",
    "Coda2", "Ornamentation2", "Duration2",
    "DeltaTime",
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


def test_hersh_rows_have_no_timestamps():
    df = _load()
    hersh = df[_source(df) == "hersh2022_pacific"]
    assert (hersh["Coda2"] == SILENCE_CODE).all(), (
        "Hersh has no timestamps so Coda2 should always be silence"
    )
    assert (hersh["DeltaTime"] == -1.0).all(), (
        "Hersh DeltaTime should always be -1 sentinel"
    )
    assert (hersh["Ornamentation1"] == 0).all(), (
        "Hersh Ornamentation1 should always be 0 (no per-coda timestamps)"
    )


def test_dswp_and_birth_have_real_dt_values():
    df = _load()
    src = _source(df)
    for s in ("sharma2024_dswp", "sharma2025_birth"):
        sub = df[src == s]
        n_real_dt = int((sub["DeltaTime"] >= 0).sum())
        assert n_real_dt > 100, (
            f"{s}: only {n_real_dt} non-sentinel DeltaTime values"
        )


def test_birth_dt_coverage_near_full():
    """Birth has 100% timestamp coverage, so almost every row should
    have a real DeltaTime (excepting the very first of each sequence)."""
    df = _load()
    sub = df[_source(df) == "sharma2025_birth"]
    n_first = int(sub["itemPosition"].eq(0).sum())
    n_real = int((sub["DeltaTime"] >= 0).sum())
    expected_real = len(sub) - n_first
    # Allow a 5 % cushion for the rare first-coda-of-segment-with-NA-time.
    assert n_real >= 0.95 * expected_real, (
        f"birth: {n_real}/{expected_real} non-sentinel DeltaTime "
        f"(expected ≈ rows minus first-of-sequence)"
    )


def test_coda1_codes_decode_through_index():
    df = _load()
    if not RHYTHM_INDEX.exists():
        pytest.skip(f"{RHYTHM_INDEX.relative_to(REPO)} not found")
    idx = pd.read_csv(RHYTHM_INDEX)
    valid = set(idx["rhythm_class"].astype(int).tolist()) | {SILENCE_CODE}
    unique = set(int(x) for x in df["Coda1"].unique())
    unknown = unique - valid
    assert not unknown, f"Coda1 values not in rhythm_class_index: {unknown}"


def test_item_position_contiguous_per_sequence():
    df = _load()
    for seq, sub in df.groupby("sequenceId"):
        positions = sub["itemPosition"].to_numpy()
        expected = list(range(len(sub)))
        assert positions.tolist() == expected, (
            f"sequence {seq}: itemPosition not contiguous"
        )


def test_simultaneous_coda2_in_birth():
    """Birth has 100 % timestamps, so the 0.3 s simultaneity window should
    catch some second-whale codas."""
    df = _load()
    sub = df[_source(df) == "sharma2025_birth"]
    n_sim = int((sub["Coda2"] != SILENCE_CODE).sum())
    assert n_sim >= 100, f"only {n_sim} simultaneous Coda2 in birth"
