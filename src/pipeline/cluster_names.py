"""Auto-name OPTICS clusters in Gero 2016 §2.2.4 notation.

A name has the form ``{n_clicks}{rhythm}{tempo_rank?}`` where

- ``n_clicks`` — integer click count (3..10 in Gero's range).
- ``rhythm`` — one of ``R`` (regular), ``D`` (decreasing), ``i`` (increasing),
  or a ``+``-joined group string like ``1+1+3`` for pause-separated patterns.
- ``tempo_rank`` — optional, only when multiple clusters share the same
  ``n_clicks + rhythm``: ``1`` is the fastest (shortest mean coda duration),
  larger ranks slower.

Pacific-discovered clusters (section A of next_phases_plan.md) get a ``P``
marker so they don't collide with Gero's published EC names:

- Same shape as an existing EC type, different tempo band: ``5RP1``, ``5RP2``.
- Shape entirely novel (no EC type with this n_clicks+rhythm): ``5P1``, ``5P2``.
- ``+``-pattern names already disambiguate themselves; no marker.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

# Tolerance (relative to the centroid's mean ICI) for monotone classification.
# 10% chosen so that small jitter in the centroid doesn't flip R↔i/D, but real
# monotone patterns still register.
MONOTONE_RTOL = 0.10

# An ICI counts as a between-group "gap" iff it is at least this factor times
# the minimum ICI in the coda. This works even when multiple gaps are adjacent
# (e.g. ``1+1+3``: two long gaps next to each other), where comparison to local
# neighbours fails. A second-stage ratio check (gap_median / non_gap_median ≥
# PLUS_GAP_RATIO) rules out monotone codas whose largest ICI happens to exceed
# the cutoff.
PLUS_GAP_FACTOR = 3.0
PLUS_GAP_RATIO = 3.0

EC_NAME_RE = re.compile(r"^(\d+)([RDi])(\d*)$")


def detect_plus_groups(icis: Sequence[float]) -> list[int] | None:
    """If ``icis`` show one or more pause gaps, return the click-group sizes.

    For an n-click coda, ``icis`` has length n-1. A position is flagged as a
    gap when its ICI exceeds both the median of all ICIs and the mean of its
    immediate neighbours by at least ``PLUS_GAP_FACTOR``. Click groups are the
    counts of consecutive non-gap ICIs plus 1 (for the click that opens the
    group).

    Examples:

        n=4, icis = [s, BIG, s]  →  [2, 2]   →  "2+2"
        n=4, icis = [BIG, s, s]  →  [1, 3]   →  "1+3"
        n=5, icis = [BIG, BIG, s, s]  →  [1, 1, 3]  →  "1+1+3"

    Returns ``None`` if no gap is detected (caller falls back to R/D/i).
    """
    icis = np.asarray(icis, dtype=float)
    if icis.size < 2:
        return None
    min_ici = float(icis.min())
    if min_ici <= 0:
        return None
    candidate = icis > PLUS_GAP_FACTOR * min_ici
    if not candidate.any() or candidate.all():
        # All-equal codas have no gap; all-large codas have no body.
        return None
    gap_median = float(np.median(icis[candidate]))
    non_gap_median = float(np.median(icis[~candidate]))
    if non_gap_median <= 0 or gap_median / non_gap_median < PLUS_GAP_RATIO:
        return None
    is_gap = candidate
    groups: list[int] = []
    run = 1
    for gap in is_gap:
        if gap:
            groups.append(run)
            run = 1
        else:
            run += 1
    groups.append(run)
    return groups


def detect_rhythm(icis: Sequence[float]) -> str:
    """Return one of ``R``/``D``/``i`` or a ``+``-pattern (e.g. ``1+1+3``)."""
    icis = np.asarray(icis, dtype=float)
    plus = detect_plus_groups(icis)
    if plus is not None:
        return "+".join(str(g) for g in plus)
    if icis.size < 2:
        return "R"
    diffs = np.diff(icis)
    mean_ici = float(np.mean(icis))
    tol = MONOTONE_RTOL * mean_ici if mean_ici > 0 else 1e-6
    if np.all(diffs >= -tol) and float(diffs.max()) > tol:
        return "i"
    if np.all(diffs <= tol) and float(diffs.min()) < -tol:
        return "D"
    return "R"


def parse_ec_shape(name: str) -> tuple[int, str] | None:
    """``5R1`` → (5, ``R``). ``1+1+3`` / ``5-NOISE`` → ``None``."""
    if not isinstance(name, str):
        return None
    m = EC_NAME_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def collapse_to_rhythm18(coda_type) -> str | float | None:
    """Drop the tempo-rank suffix from R/D/i names (Sharma 2024's 18-rhythm
    collapse). Pass through ``+`` patterns and NOISE labels unchanged.

    Handles all of ``None`` / ``np.nan`` / ``pd.NA`` / actual strings — falling
    through to ``str(pd.NA)`` would silently produce the literal label
    ``"<NA>"``, polluting the vocabulary.
    """
    if not isinstance(coda_type, str):
        # Catches None, np.nan, pd.NA, and any other non-string sentinel.
        return None
    m = EC_NAME_RE.match(coda_type)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return coda_type


def assign_tempo_ranks(clusters: list[dict], *, pacific: bool = False,
                       ec_shapes: Iterable[tuple[int, str]] | None = None) -> list[dict]:
    """Mutate each cluster dict in place to add ``rhythm``, ``rank``, ``name``.

    Each input cluster must have ``n_clicks``, ``centroid``, ``mean_duration``.
    Optionally a pre-computed ``rhythm`` (else detected from the centroid).

    For EC clusters (``pacific=False``): rank by ascending mean duration within
    each (n_clicks, rhythm) bucket. Singleton bucket → no rank suffix.

    For Pacific clusters (``pacific=True``): same ranking, but the name carries
    a ``P`` marker. If the (n_clicks, rhythm) shape exists in EC's vocabulary
    (``ec_shapes``), use ``{n}{rhythm}P{rank}`` (e.g. ``5RP1``). If the shape
    is novel, use ``{n}P{rank}`` (e.g. ``5P1``). ``+``-pattern names render
    verbatim.
    """
    ec_shapes_set = set(ec_shapes) if ec_shapes is not None else set()
    buckets: dict[tuple[int, str], list[int]] = defaultdict(list)
    for i, c in enumerate(clusters):
        if "rhythm" not in c:
            c["rhythm"] = detect_rhythm(c["centroid"])
        if "+" in c["rhythm"]:
            c["rank"] = None
            c["name"] = c["rhythm"]
            continue
        buckets[(int(c["n_clicks"]), c["rhythm"])].append(i)

    for (n, rhythm), idxs in buckets.items():
        ordered = sorted(idxs, key=lambda i: clusters[i]["mean_duration"])
        n_in_bucket = len(ordered)
        for rank, i in enumerate(ordered, start=1):
            c = clusters[i]
            if pacific:
                # Pacific clusters always get a rank suffix — leaves room for
                # future additions and prevents collision with EC names.
                if (n, rhythm) in ec_shapes_set:
                    c["name"] = f"{n}{rhythm}P{rank}"
                else:
                    c["name"] = f"{n}P{rank}"
                c["rank"] = rank
            else:
                if n_in_bucket == 1:
                    c["rank"] = None
                    c["name"] = f"{n}{rhythm}"
                else:
                    c["rank"] = rank
                    c["name"] = f"{n}{rhythm}{rank}"
    return clusters


def name_ec_clusters(clusters: list[dict]) -> list[dict]:
    """Convenience wrapper for EC (DSWP) clusters."""
    return assign_tempo_ranks(clusters, pacific=False)


def name_pacific_clusters(clusters: list[dict],
                          ec_names: Iterable[str]) -> list[dict]:
    """Convenience wrapper for Pacific-discovered clusters.

    ``ec_names`` is the iterable of strings already in use for EC clusters.
    """
    ec_shapes = (s for s in (parse_ec_shape(n) for n in ec_names) if s is not None)
    return assign_tempo_ranks(clusters, pacific=True, ec_shapes=ec_shapes)
