"""Stage 3 acceptance gate — birth (Sharma 2025 calves) integration audit.

Sanity assertions on the ``sharma2025_birth`` slice of
``data/classified/codas_classified.csv``:

* ≥ 93 % matched to a Sharma 'real' type (birth = Dominica calves, so
  their repertoire is the DSWP repertoire with developmental jitter).
* ≥ 99 % tempo populated (just needs ``coda_duration_s``).
* Rubato fires for at least some birth rows (birth has 100 % timestamps,
  so the same-whale-within-10s lookup should produce many deltas).
* ``extra_click`` populated for every birth row (Sharma 2024 §5
  structural rule applies — not gated on Hersh's missing timestamps).

Numbers from the 2026-05-04 baseline are documented in
``data/diagnostics/hybrid_v1/RESULTS.md``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
CLASSIFIED_CSV = REPO / "data" / "classified" / "codas_classified.csv"


def _birth() -> pd.DataFrame:
    if not CLASSIFIED_CSV.exists():
        pytest.skip(f"{CLASSIFIED_CSV.relative_to(REPO)} not found; "
                    f"run `python -m src.pipeline.B_classify`")
    df = pd.read_csv(CLASSIFIED_CSV, low_memory=False)
    return df[df["source"] == "sharma2025_birth"].copy()


def test_birth_matched_rate_at_least_93pct():
    b = _birth()
    matched = (b["classifier_origin"] == "pacific-matched").sum()
    rate = matched / len(b)
    assert rate >= 0.93, (
        f"birth pacific-matched rate {rate:.2%} < 93% "
        f"({matched:,}/{len(b):,})"
    )


def test_birth_tempo_populated():
    b = _birth()
    rate = b["tempo"].notna().mean()
    assert rate >= 0.99, f"birth tempo populated only {rate:.2%}"


def test_birth_rubato_fires():
    b = _birth()
    n = int(b["rubato"].notna().sum())
    assert n >= 100, (
        f"birth rubato fired only {n} times; expected many "
        f"(birth has 100 % timestamps)"
    )


def test_birth_extra_click_populated_for_every_row():
    """Sharma 2024 §5 structural rule needs per-coda timestamps and whale
    identity. Birth has both. ``extra_click`` should be 0 or 1 for every
    birth row — never NA."""
    b = _birth()
    rate = b["extra_click"].notna().mean()
    assert rate >= 0.99, (
        f"birth extra_click NA for {(1 - rate):.2%} of rows; "
        f"structural rule should fire for every row"
    )


def test_birth_some_ornaments_detected():
    b = _birth()
    n_orn = int((b["extra_click"] == 1).sum())
    assert n_orn > 0, "no ornaments detected on birth"
