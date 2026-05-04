"""Stage 1.1 — Hersh-clan coherence test.

For each Hersh clan, compute top-N coverage among its
``pacific-matched`` ∪ ``discovery-cluster`` rows. Original plan called for
≥ 70 % top-5 across {EC1, PALI, FP, REG, RI, SH, PO, SI}; the actual
distribution (recorded in ``data/diagnostics/hybrid_v1/clan_coherence.csv``)
shows only EC1 (88.7 %) and REG (72.9 %) hit that bar — every other clan
genuinely fragments across many types (FP has 91 distinct labels over
1,639 codas). The plan itself called this out as the expected biological
answer: don't tighten τ to chase it.

Two assertions:

* **Coherent clans** (EC1, REG): top-5 coverage stays ≥ 70 %. This is the
  "matcher hasn't broken" regression check — these are the structurally
  tight clans whose Sharma-DSWP overlap should always look concentrated.
* **All clans**: top-10 coverage stays within ±5 pp of the baseline below.
  Catches a matcher regression that scrambles label assignment without
  pretending the diffuse clans should look concentrated.

After Stage 2 promotes the hybrid pipeline into ``B_classify``, this test
should be rewired to read ``data/classified/codas_classified.csv`` (the
columns ``clan``, ``classifier_origin``, ``coda_type_gero21`` carry the
same info under different names).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
ASSIGNMENTS = REPO / "data" / "diagnostics" / "hybrid_v1" / "assignments.csv"
CLAN_COHERENCE_CSV = (REPO / "data" / "diagnostics" / "hybrid_v1"
                       / "clan_coherence.csv")

CLAN_TARGETS = ("EC1", "PALI", "FP", "REG", "RI", "SH", "PO", "SI")
COHERENT_CLANS = ("EC1", "REG")
COHERENT_TOP5_FLOOR = 0.70

# Baseline top-10 coverage from the 2026-05-04 hybrid_v1 run. Tolerance is
# ±5 pp — anything bigger is a matcher regression worth investigating.
BASELINE_TOP10 = {
    "EC1": 0.9411,
    "PALI": 0.8068,
    "FP": 0.4857,
    "REG": 0.9192,
    "RI": 0.7262,
    "SH": 0.7568,
    "PO": 0.8156,
    "SI": 0.7847,
}
TOP10_TOLERANCE_PP = 0.05


def _load_assignments() -> pd.DataFrame:
    if not ASSIGNMENTS.exists():
        pytest.skip(f"{ASSIGNMENTS.relative_to(REPO)} not found; "
                    f"run `python -m src.validation.hybrid_classify`")
    return pd.read_csv(ASSIGNMENTS, low_memory=False)


def _coverage(state: pd.DataFrame, clan_label: str,
              k: int) -> tuple[float, int, int]:
    rows = state[
        (state["clan"] == clan_label)
        & state["origin"].isin(["pacific-matched", "discovery-cluster"])
    ]
    n = len(rows)
    if n == 0:
        return float("nan"), 0, 0
    counts = rows["final_label"].value_counts()
    return float(counts.head(k).sum() / n), int(n), int(counts.size)


@pytest.mark.parametrize("clan", COHERENT_CLANS)
def test_coherent_clans_top5_coverage(clan):
    cov, n, _ = _coverage(_load_assignments(), clan, 5)
    assert n > 0, f"clan {clan} has no classified rows"
    assert cov >= COHERENT_TOP5_FLOOR, (
        f"clan {clan}: top-5 = {cov:.2%} (< {COHERENT_TOP5_FLOOR:.0%}) "
        f"over n={n}"
    )


@pytest.mark.parametrize("clan", CLAN_TARGETS)
def test_clan_top10_within_baseline(clan):
    cov, n, _ = _coverage(_load_assignments(), clan, 10)
    assert n > 0, f"clan {clan} has no classified rows"
    expected = BASELINE_TOP10[clan]
    assert abs(cov - expected) <= TOP10_TOLERANCE_PP, (
        f"clan {clan}: top-10 = {cov:.2%}, baseline = {expected:.2%} "
        f"(tolerance ±{TOP10_TOLERANCE_PP * 100:.0f} pp), n={n}"
    )


def test_write_clan_coherence_csv():
    state = _load_assignments()
    rows = []
    for clan in CLAN_TARGETS:
        for k in (1, 3, 5, 10):
            cov, n, n_unique = _coverage(state, clan, k)
            rows.append({"clan": clan, "k": k, "n_classified": n,
                         "n_unique_labels": n_unique, "topk_coverage": cov})
    df = pd.DataFrame(rows)
    CLAN_COHERENCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLAN_COHERENCE_CSV, index=False)
    assert CLAN_COHERENCE_CSV.exists()
