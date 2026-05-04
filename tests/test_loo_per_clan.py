"""Stage 1.2 — Per-clan over-classification regression check.

Reads ``data/diagnostics/hybrid_v1/loo_knn_tau_by_clan.csv`` (produced by
``python -m src.validation.loo_knn_tau``). For each clan with ≥ 50
classified codas, asserts that the LOO over-classification rate stays
below 2 %. Below-50 clans are too small for the rate to be meaningful.

Note this test only covers Sharma DSWP rows; the LOO mechanism requires
ground-truth ``CodaType`` labels and Hersh has none. Pacific clans
appear in this slice only via the ``clan`` column on Sharma's own rows
(EC1, EC2).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
BY_CLAN_CSV = (REPO / "data" / "diagnostics" / "hybrid_v1"
                / "loo_knn_tau_by_clan.csv")

OVER_CLASS_FLOOR = 0.02
MIN_N_FOR_GATE = 50


def _load() -> pd.DataFrame:
    if not BY_CLAN_CSV.exists():
        pytest.skip(f"{BY_CLAN_CSV.relative_to(REPO)} not found; "
                    f"run `python -m src.validation.loo_knn_tau`")
    return pd.read_csv(BY_CLAN_CSV)


def test_per_clan_over_classification_under_2pct():
    df = _load()
    big_enough = df[df["total"] >= MIN_N_FOR_GATE]
    assert len(big_enough) > 0, "no clan has enough rows to gate on"
    bad = big_enough[big_enough["over_class_rate"] > OVER_CLASS_FLOOR]
    assert bad.empty, (
        f"clans above {OVER_CLASS_FLOOR:.0%} over-classification:\n"
        f"{bad.to_string(index=False)}"
    )
