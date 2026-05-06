"""
Bucket schemes for the TimeDelta channel — used as the second target
of the multi-task M7 model.

CHILDES — 4 classes (driven entirely by the heuristic in
`childes_loader.py`, with id 3 mirroring whale's `missing` bucket for
the Hersh-style data-loss experiment):

    0 intra   (TimeDelta in {0.0, 0.3})  no punctuation, same speaker
    1 period  (TimeDelta == 1.0)         end of utterance, same speaker
    2 switch  (TimeDelta == 2.0)         end of utterance + speaker change
    3 missing (has_timestamps == 0)      Hersh-mirror rows, no DT signal

(`childes_loader` was supposed to also emit 0.5-second "comma" gaps but
the %mor tier in the Eng-UK corpus rarely includes standalone commas,
so empirically only three of the time-derived buckets fire. The
`missing` bucket is populated only when --mirror-hersh-frac > 0.)

Whale — 7 classes (missing + 5 log-quantile timing buckets + switch):

    0 missing  (has_timestamps == 0; ≈ 76 % of the corpus, all Hersh
               recordings)
    1 < 0.42 s         q0–q20 of valid same-speaker TimeDelta
    2 0.42 – 1.19 s    q20–q40 same-speaker
    3 1.19 – 2.37 s    q40–q60 same-speaker
    4 2.37 – 3.98 s    q60–q80 same-speaker
    5 > 3.98 s         q80+ same-speaker (long-tail bucket)
    6 switch           next coda is from a different whale (any gap)

Bucket 6 mirrors the multilang scheme's bucket 3 (`switch`) — without it,
the model has no output channel for turn-taking, and generated whale
continuations collapse into a single "Whale ?:" line.

Quantile boundaries are computed once on the full whale corpus and
hard-coded here so the bucket id ↔ time mapping is stable across runs.

Each bucket has a display string used by `interp_compare.render_*` to
emit human-readable continuations:

    CHILDES bucket 0/1/2  →  ""/"."/"<switch>"
    Whale bucket 0..6     →  "Δt?", "Δt<0.4", "Δt~0.8", "Δt~1.8",
                              "Δt~3.2", "Δt>4", "<switch>"
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------- CHILDES ----------

CHILDES_INTRA = 0
CHILDES_PERIOD = 1
CHILDES_SWITCH = 2
CHILDES_MISSING = 3
CHILDES_N_BUCKETS = 4
CHILDES_LABELS = ["intra", "period", "switch", "missing"]


def childes_dt_to_bucket(dt: float, has_timestamps: int = 1) -> int:
    """Map a CHILDES TimeDelta float to a bucket id."""
    if not has_timestamps or dt < 0:
        return CHILDES_MISSING
    if dt >= 1.5:
        return CHILDES_SWITCH
    if dt >= 0.75:
        return CHILDES_PERIOD
    return CHILDES_INTRA


# Bucket id → token to insert *before* the next word in the readable
# transcript. The "switch" case is handled specially by the renderer
# (it ends the utterance and changes speaker). "missing" is the
# Hersh-mirror sentinel — renderer treats it like "intra" (no marker).
CHILDES_DISPLAY = {
    CHILDES_INTRA: "",
    CHILDES_PERIOD: ".",
    CHILDES_SWITCH: "<switch>",  # rendered as a new "*SPK:" line
    CHILDES_MISSING: "",
}


# ---------- Multilang ----------
#
# 5 classes — splits the CHILDES `intra` bucket into intra-word and
# word-boundary so the model can in principle recover word boundaries
# from timing on the clean (non-Hersh-equiv) tiers:
#
#     0 intra_word   (TimeDelta == 0.0)   sub-tokens inside a single word
#                                         (mora 2..N of a JP word, phoneme
#                                         2..N of an EN word, syllable 2..N
#                                         of a multi-syllable ZH lemma)
#     1 word_bound   (TimeDelta in {0.3, 0.5})  same-speaker, between words
#     2 period       (TimeDelta == 1.0)   end of utterance, same speaker
#     3 switch       (TimeDelta == 2.0)   end of utterance + speaker change
#     4 missing      (has_timestamps == 0) Hersh-equiv tier rows

MULTILANG_INTRA_WORD = 0
MULTILANG_WORD_BOUND = 1
MULTILANG_PERIOD = 2
MULTILANG_SWITCH = 3
MULTILANG_MISSING = 4
MULTILANG_N_BUCKETS = 5
MULTILANG_LABELS = ["intra_word", "word_bound", "period", "switch", "missing"]


def multilang_dt_to_bucket(dt: float, has_timestamps: int = 1) -> int:
    if not has_timestamps or dt < 0:
        return MULTILANG_MISSING
    if dt >= 1.5:
        return MULTILANG_SWITCH
    if dt >= 0.75:
        return MULTILANG_PERIOD
    if dt >= 0.15:           # 0.3 (intra-utt) or 0.5 (after comma)
        return MULTILANG_WORD_BOUND
    return MULTILANG_INTRA_WORD


# ---------- Whale ----------

# Hard-coded quantile edges of valid (has_timestamps==1) TimeDelta on
# the unified corpus. See dt_buckets.py docstring for derivation.
WHALE_DT_EDGES = [0.42, 1.19, 2.37, 3.98]   # 4 edges → 5 valid buckets
WHALE_MISSING = 0
# Buckets 1..5 are same-speaker timing buckets (low → high gap). Bucket
# 6 is `switch` — fires when the next coda comes from a *different*
# whale (regardless of gap), mirroring the multilang scheme's bucket 3.
# Without this signal the model can't learn turn-taking, so generated
# continuations collapse into a single undifferentiated "Whale ?:" line.
WHALE_SWITCH = 6
WHALE_N_BUCKETS = 7
WHALE_LABELS = [
    "Δt?",
    "Δt<0.4",
    "Δt~0.8",
    "Δt~1.8",
    "Δt~3.2",
    "Δt>4",
    "switch",
]
# Midpoint seconds for rendering an estimated Δt back into the
# transcript. None for missing and switch.
WHALE_DT_MIDPOINTS = [None, 0.2, 0.8, 1.8, 3.2, 6.0, None]


def whale_dt_to_bucket(dt: float, has_timestamps: int,
                       switched: bool = False) -> int:
    """Map a whale (TimeDelta, has_timestamps, speaker-change) triple to a bucket.

    - has_timestamps == 0  →  missing  (overrides everything)
    - switched == True      →  switch   (new speaker, gap-time ignored)
    - else                  →  log-quantile time bucket 1..5
    """
    if not has_timestamps or dt < 0:
        return WHALE_MISSING
    if switched:
        return WHALE_SWITCH
    for i, edge in enumerate(WHALE_DT_EDGES):
        if dt < edge:
            return i + 1
    return 5  # >3.98s same-speaker


def whale_bucket_display(b: int) -> str:
    """Format like 'Δt0.42' when we have a midpoint, else 'Δt?' /
    '<switch>' for the switch bucket."""
    if b == WHALE_SWITCH:
        return "<switch>"
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
    if source == "multilang":
        return DTScheme("multilang", MULTILANG_N_BUCKETS, MULTILANG_LABELS)
    raise ValueError(source)


def dt_to_bucket(source: str, dt: float, has_timestamps: int = 1,
                 switched: bool = False) -> int:
    if source == "whale":
        return whale_dt_to_bucket(dt, has_timestamps, switched=switched)
    if source == "multilang":
        return multilang_dt_to_bucket(dt, has_timestamps)
    return childes_dt_to_bucket(dt, has_timestamps)
