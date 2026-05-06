"""
Bucket schemes for the TimeDelta channel — used as the second target
of the multi-task M7 model.

CHILDES — 3 classes (driven entirely by the heuristic in
`childes_loader.py`):

    0 intra   (TimeDelta in {0.0, 0.3})  no punctuation, same speaker
    1 period  (TimeDelta == 1.0)         end of utterance, same speaker
    2 switch  (TimeDelta == 2.0)         end of utterance + speaker change

(`childes_loader` was supposed to also emit 0.5-second "comma" gaps but
the %mor tier in the Eng-UK corpus rarely includes standalone commas,
so empirically only three buckets fire. Keeping the table to three
keeps the head small.)

Whale — 6 classes (missing + 5 log-quantile buckets of valid
TimeDelta):

    0 missing  (has_timestamps == 0; ≈ 76 % of the corpus, all Hersh
               recordings)
    1 < 0.42 s         q0–q20 of valid TimeDelta
    2 0.42 – 1.19 s    q20–q40
    3 1.19 – 2.37 s    q40–q60
    4 2.37 – 3.98 s    q60–q80
    5 > 3.98 s         q80+ (the long-tail bucket, includes pauses)

Quantile boundaries are computed once on the full whale corpus and
hard-coded here so the bucket id ↔ time mapping is stable across runs.

Each bucket has a display string used by `interp_compare.render_*` to
emit human-readable continuations:

    CHILDES bucket 0/1/2  →  ""/"."/"<switch>"
    Whale bucket 0..5     →  "Δt?", "Δt<0.4", "Δt~0.8", "Δt~1.8",
                              "Δt~3.2", "Δt>4"
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------- CHILDES ----------

CHILDES_INTRA = 0
CHILDES_PERIOD = 1
CHILDES_SWITCH = 2
CHILDES_N_BUCKETS = 3
CHILDES_LABELS = ["intra", "period", "switch"]


def childes_dt_to_bucket(dt: float) -> int:
    """Map a CHILDES TimeDelta float to a bucket id."""
    if dt >= 1.5:
        return CHILDES_SWITCH
    if dt >= 0.75:
        return CHILDES_PERIOD
    return CHILDES_INTRA


# Bucket id → token to insert *before* the next word in the readable
# transcript. The "switch" case is handled specially by the renderer
# (it ends the utterance and changes speaker).
CHILDES_DISPLAY = {
    CHILDES_INTRA: "",
    CHILDES_PERIOD: ".",
    CHILDES_SWITCH: "<switch>",  # rendered as a new "*SPK:" line
}


# ---------- Whale ----------

# Hard-coded quantile edges of valid (has_timestamps==1) TimeDelta on
# the unified corpus. See dt_buckets.py docstring for derivation.
WHALE_DT_EDGES = [0.42, 1.19, 2.37, 3.98]   # 4 edges → 5 valid buckets
WHALE_N_BUCKETS = 6                         # +1 for missing
WHALE_MISSING = 0
WHALE_LABELS = [
    "Δt?",
    "Δt<0.4",
    "Δt~0.8",
    "Δt~1.8",
    "Δt~3.2",
    "Δt>4",
]
# Midpoint seconds for rendering an estimated Δt back into the
# transcript. None for the missing bucket.
WHALE_DT_MIDPOINTS = [None, 0.2, 0.8, 1.8, 3.2, 6.0]


def whale_dt_to_bucket(dt: float, has_timestamps: int) -> int:
    if not has_timestamps or dt < 0:
        return WHALE_MISSING
    for i, edge in enumerate(WHALE_DT_EDGES):
        if dt < edge:
            return i + 1
    return WHALE_N_BUCKETS - 1


def whale_bucket_display(b: int) -> str:
    """Format like 'Δt0.42' when we have a midpoint, else 'Δt?'."""
    mp = WHALE_DT_MIDPOINTS[b]
    if mp is None:
        return "Δt?"
    return f"Δt{mp:.2f}"


# ---------- common ----------


@dataclass
class DTScheme:
    name: str
    n_buckets: int
    labels: list[str]


def scheme_for(source: str) -> DTScheme:
    if source == "whale":
        return DTScheme("whale", WHALE_N_BUCKETS, WHALE_LABELS)
    if source == "childes":
        return DTScheme("childes", CHILDES_N_BUCKETS, CHILDES_LABELS)
    raise ValueError(source)


def dt_to_bucket(source: str, dt: float, has_timestamps: int = 1) -> int:
    if source == "whale":
        return whale_dt_to_bucket(dt, has_timestamps)
    return childes_dt_to_bucket(dt)
