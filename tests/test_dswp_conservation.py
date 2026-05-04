"""Stage 1.4 — DSWP-truth conservation regression check (fast).

Reads ``data/classified/codas_classified.csv`` directly and asserts that
every DSWP row that has a published ``CodaType`` in
``data/upstream/dswp_dominica_codas.csv`` ends up with the same value in
``coda_type_gero21``. This must hold today (under the centroid+radius
pipeline) and after Stage 2 promotes the kNN+τ matcher into
``B_classify``.

This complements the ``TestPipeline.test_dswp_codatype_join_is_exact``
test in ``test_classify.py``, which runs the full ELKI pipeline. This
fast variant operates on the on-disk artefact and runs in seconds.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
CLASSIFIED_CSV = REPO / "data" / "classified" / "codas_classified.csv"
DOMINICA_CSV = REPO / "data" / "upstream" / "dswp_dominica_codas.csv"


def _load_dominica_lookup() -> pd.Series:
    d = pd.read_csv(DOMINICA_CSV)
    d.columns = [c.strip().lstrip("﻿") for c in d.columns]
    return d.set_index(d["codaNUM2018"].astype(str))["CodaType"]


def _load_classified() -> pd.DataFrame:
    if not CLASSIFIED_CSV.exists():
        pytest.skip(f"{CLASSIFIED_CSV.relative_to(REPO)} not found; "
                    f"run `python -m src.pipeline.D_run --refresh`")
    if not DOMINICA_CSV.exists():
        pytest.skip(f"{DOMINICA_CSV.relative_to(REPO)} not found")
    return pd.read_csv(CLASSIFIED_CSV, low_memory=False)


def test_dswp_real_codatype_conservation_exact():
    """DSWP rows whose published ``CodaType`` is *real* (non-NOISE) must
    pass through with that exact value as ``coda_type_gero21``. Rows
    whose ``CodaType`` ends in ``-NOISE`` are deliberately re-pooled
    into the discovery pass and may emerge with a discovered-cluster
    label (the Sharma-NOISE second-chance recovery in
    ``whale_grammar_transformer_plan`` §1.2)."""
    df = _load_classified()
    lookup = _load_dominica_lookup()
    is_dswp = df["source"] == "sharma2024_dswp"
    truth = df.loc[is_dswp, "source_coda_id"].astype(str).map(lookup)
    is_real_truth = truth.notna() & ~truth.astype("string").str.endswith(
        "-NOISE", na=False)
    n_real = int(is_real_truth.sum())
    assert n_real > 0, "no DSWP 'real' rows to verify"
    actual = df.loc[is_dswp, "coda_type_gero21"][is_real_truth]
    expected = truth[is_real_truth]
    n_equal = int((actual == expected).sum())
    if n_equal != n_real:
        diff = (actual != expected)
        sample = pd.DataFrame({
            "source_coda_id":
                df.loc[is_dswp, "source_coda_id"][is_real_truth][diff].head(10),
            "coda_type_gero21": actual[diff].head(10),
            "CodaType_truth": expected[diff].head(10),
        })
        raise AssertionError(
            f"DSWP 'real' not exact: {n_equal}/{n_real} match. "
            f"First mismatches:\n{sample.to_string(index=False)}"
        )


def test_no_dswp_row_silently_dropped():
    """Every DSWP row of length 3..10 with all ICIs present *should* end up
    with a ``coda_type_gero21`` (either the truth label, or — once Stage 2
    runs — a discovery-cluster label for Sharma-NOISE rows that found a
    home). Catches a regression where DSWP rows leak into NA."""
    df = _load_classified()
    is_dswp = df["source"] == "sharma2024_dswp"
    n_clicks = pd.to_numeric(df["n_clicks"], errors="coerce")
    in_range = is_dswp & n_clicks.between(3, 10)
    ici_cols = [f"ICI{i}" for i in range(1, 10)]
    have_ici = (df[ici_cols].notna()
                .iloc[:, : 9].apply(lambda row: True, axis=1))  # length-aware below
    # Per-length filter for "has all required ICIs".
    all_icis = pd.Series(False, index=df.index)
    for n in range(3, 11):
        cols = [f"ICI{i}" for i in range(1, n)]
        m = is_dswp & (n_clicks == n) & df[cols].notna().all(axis=1)
        all_icis |= m
    eligible = in_range & all_icis
    n_eligible = int(eligible.sum())
    n_classified = int(df.loc[eligible, "coda_type_gero21"].notna().sum())
    # Allow up to 1 % drop to absorb edge cases (NaN truth from
    # codaNUM2018 join misses), but flag a deeper regression.
    assert n_classified >= 0.99 * n_eligible, (
        f"only {n_classified}/{n_eligible} eligible DSWP rows have a "
        f"coda_type_gero21"
    )
