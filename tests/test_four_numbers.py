"""Stage 2 acceptance gate — the four headline numbers from
``data/diagnostics/hybrid_v1/RESULTS.md`` must reproduce within ±10
rows on the production output.

This is a fast test: it reads the cached
``data/classified/codas_classified.csv`` and counts ``classifier_origin``
buckets. Tolerance ±10 absorbs deterministic but minor differences if
the per-cluster naming reshuffles a handful of edge-case rows. Any
larger drift is a regression worth investigating.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
CLASSIFIED_CSV = REPO / "data" / "classified" / "codas_classified.csv"

# Headline counts from the 2026-05-04 run.
EXPECTED = {
    "dswp-real": 8_119,
    "pacific-matched": 22_940,
    "discovery-cluster": 4_284,
    "discovery-noise": 2_341,
}
TOL = 10


def _load() -> pd.Series:
    if not CLASSIFIED_CSV.exists():
        pytest.skip(f"{CLASSIFIED_CSV.relative_to(REPO)} not found; "
                    f"run `python -m src.pipeline.B_classify`")
    df = pd.read_csv(CLASSIFIED_CSV, low_memory=False,
                     usecols=["classifier_origin"])
    return df["classifier_origin"].value_counts()


@pytest.mark.parametrize("origin,expected", EXPECTED.items())
def test_origin_counts_match_baseline(origin, expected):
    counts = _load()
    actual = int(counts.get(origin, 0))
    assert abs(actual - expected) <= TOL, (
        f"origin {origin!r}: got {actual:,}, baseline {expected:,} "
        f"(±{TOL})"
    )
