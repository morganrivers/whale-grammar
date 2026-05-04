"""Stage 1.3 — cluster-quality numeric regression check.

Reads ``data/diagnostics/hybrid_v1/cluster_quality.csv`` (produced by
``python -m src.validation.cluster_quality``). Asserts that ≥ 80 % of the
discovered Pacific clusters have an internal variance within 2× the
nearest Sharma 'real' type's internal variance — i.e. the discovered
vocabulary isn't dominated by sprawling, fuzzy clusters.

Flagged clusters (above 2×) are not failures; they're the candidates
that Stage 4's vocabulary-pruning step will collapse into
``OTHER_PACIFIC``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
QUALITY_CSV = (REPO / "data" / "diagnostics" / "hybrid_v1"
                / "cluster_quality.csv")

COHERENT_FRACTION_FLOOR = 0.80


def _load() -> pd.DataFrame:
    if not QUALITY_CSV.exists():
        pytest.skip(f"{QUALITY_CSV.relative_to(REPO)} not found; "
                    f"run `python -m src.validation.cluster_quality`")
    return pd.read_csv(QUALITY_CSV)


def test_majority_of_clusters_are_coherent():
    df = _load()
    assert len(df) > 0, "no discovered clusters in quality CSV"
    coherent = (~df["flag_collapse"]).mean()
    assert coherent >= COHERENT_FRACTION_FLOOR, (
        f"only {coherent:.1%} of discovered clusters have variance ≤ 2× "
        f"nearest Sharma type (floor {COHERENT_FRACTION_FLOOR:.0%}). "
        f"{int(df['flag_collapse'].sum())}/{len(df)} clusters flagged."
    )
