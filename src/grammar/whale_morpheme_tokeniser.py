"""
ICI-string and morpheme-sequence tokenisers for the whale dialogue corpus.

Two public entry points:

  load_ici_string_whale()   — Phase 1: replaces df['Coda'] with a dense
      integer id for each coda's ICI symbol string (e.g. 'BCCC').
      Vocabulary size ≈ 1 916 unique attested strings at N_BINS=4.
      Drop-in replacement for load_compound_whale(); the rest of
      train_for_interp.py works unchanged.

  load_morpheme_seq_whale() — Phase 2: expands each coda into its
      Morfessor morpheme sub-units, returning one MorphemeConversation
      per dialogue sequence with a flat morpheme token list and
      coda-boundary markers.

Join strategy
─────────────
whale_dialogues.csv was produced by E_render_csv.render(), which groups
codas_classified.csv by (source, recording_id), sorts each group by
time_in_recording_s (or source_coda_id when timestamps are absent), and
assigns sequenceId = "<source>::<recording_id>" and itemPosition = 0-based
position within the sorted group.  _assign_cc_keys() replicates exactly
that sort so we can attach ICI columns to every row in whale_dialogues.csv
via a 1:1 (sequenceId, itemPosition) merge.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import morfessor
import numpy as np
import pandas as pd

from src.grammar.dt_buckets import dt_to_bucket
from src.grammar.morpheme_discovery import (
    MAX_CLICKS,
    MIN_CLICKS,
    N_BINS,
    compute_global_bin_edges,
    ici_row_to_string,
    run_morfessor,
)

ROOT = Path(__file__).resolve().parents[2]
WHALE_CSV = ROOT / "data" / "classified" / "whale_dialogues.csv"
CODA_CSV = ROOT / "data" / "classified" / "codas_classified.csv"
MORFESSOR_PATH = ROOT / "outputs" / "morphemes" / f"morfessor_n{N_BINS}.bin"

UNK_TOKEN = "<unk>"
UNK_ID = 0

_ICI_COLS = [f"ICI{i}" for i in range(1, MAX_CLICKS + 1)]


@dataclass
class MorphemeConversation:
    seq_id: str
    tokens: list[int]           # flat morpheme token ids across all codas
    coda_boundaries: list[int]  # index in `tokens` where each coda starts
    dt_buckets: list[int]       # per morpheme token (repeated from coda DT bucket)
    times: list[float]          # raw TimeDelta per coda
    speakers: list[str]         # Whale string per coda


# ---------------------------------------------------------------------------
# Join helper
# ---------------------------------------------------------------------------


def _assign_cc_keys(cc: pd.DataFrame) -> pd.DataFrame:
    """Replicate E_render_csv.render() group-sort to assign (sequenceId,
    itemPosition) to each row in codas_classified.csv.  Returns cc with two
    new columns: _seq_id and _item_pos.
    """
    seq_ids: list[str] = []
    item_positions: list[int] = []

    for (src, rec), grp in cc.groupby(["source", "recording_id"], dropna=False, sort=True):
        if grp["time_in_recording_s"].notna().any():
            grp = grp.sort_values(
                ["time_in_recording_s", "source_coda_id"],
                kind="stable",
                na_position="last",
            )
        else:
            grp = grp.sort_values("source_coda_id", kind="stable")

        rec_str = "no_recording" if pd.isna(rec) else str(rec)
        sid = f"{src}::{rec_str}"
        for pos in range(len(grp)):
            seq_ids.append(sid)
            item_positions.append(pos)

    cc = cc.copy()
    cc["_seq_id"] = seq_ids
    cc["_item_pos"] = item_positions
    return cc


def _load_joined(dia_csv: Path = WHALE_CSV, cc_csv: Path = CODA_CSV) -> pd.DataFrame:
    """Load whale_dialogues.csv and attach ICI columns from
    codas_classified.csv via the reconstructed (sequenceId, itemPosition) key.
    """
    dia = pd.read_csv(dia_csv)
    cc = pd.read_csv(cc_csv, low_memory=False)
    cc_keyed = _assign_cc_keys(cc)

    present_ici = [c for c in _ICI_COLS if c in cc_keyed.columns]
    attach_cols = ["_seq_id", "_item_pos", "n_clicks"] + present_ici
    cc_slim = (
        cc_keyed[attach_cols]
        .rename(columns={"_seq_id": "sequenceId", "_item_pos": "itemPosition"})
    )

    merged = dia.merge(cc_slim, on=["sequenceId", "itemPosition"], how="left")
    return merged


# ---------------------------------------------------------------------------
# Phase 1 — ICI-string vocabulary
# ---------------------------------------------------------------------------


def load_ici_string_whale(
    csv: Path = WHALE_CSV,
    n_bins: int = N_BINS,
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Replace df['Coda'] with a dense integer id for the coda's ICI string.

    Returns (df, decoder) where decoder[id] → ICI string, e.g. 'BCCC'.
    Rows without valid ICI data (too short/long or missing n_clicks) get
    token id 0 (reserved for '<unk>').
    """
    df = _load_joined(dia_csv=csv)

    valid_mask = (
        df["n_clicks"].notna()
        & df["n_clicks"].between(MIN_CLICKS, MAX_CLICKS)
    )
    edges = compute_global_bin_edges(df[valid_mask].copy(), n_bins=n_bins)

    ici_strs: list[str | None] = []
    for _, row in df.iterrows():
        n = row.get("n_clicks")
        if pd.isna(n) or not (MIN_CLICKS <= int(n) <= MAX_CLICKS):
            ici_strs.append(None)
            continue
        s = ici_row_to_string(row, edges)
        ici_strs.append(s if s else None)

    unique_strs = sorted(s for s in set(ici_strs) if s is not None)
    str_to_id = {s: i + 1 for i, s in enumerate(unique_strs)}
    decoder: dict[int, str] = {0: UNK_TOKEN}
    decoder.update({v: k for k, v in str_to_id.items()})

    df = df.copy()
    df["Coda_orig"] = df["Coda"]
    df["Coda"] = [str_to_id.get(s, UNK_ID) if s else UNK_ID for s in ici_strs]

    n_unk = sum(1 for s in ici_strs if s is None)
    print(
        f"ICI-string vocab: {len(unique_strs)} types + <unk>  "
        f"(<unk> rows: {n_unk}/{len(df)}, {100*n_unk/len(df):.1f}%)"
    )
    return df, decoder


# ---------------------------------------------------------------------------
# Phase 2 — Morpheme sub-tokenisation
# ---------------------------------------------------------------------------


def _load_morfessor(ici_corpus: Counter | None = None) -> morfessor.BaselineModel:
    """Load saved Morfessor model; train from ici_corpus if the file is absent."""
    mio = morfessor.MorfessorIO()
    if MORFESSOR_PATH.exists():
        return mio.read_binary_model_file(str(MORFESSOR_PATH))
    if ici_corpus is None:
        raise FileNotFoundError(
            f"Morfessor model not found at {MORFESSOR_PATH}. "
            "Run `python -m src.grammar.morpheme_discovery` first."
        )
    print(f"  Morfessor model not found; training from scratch on {len(ici_corpus)} ICI strings.")
    model, _ = run_morfessor(ici_corpus, save_path=MORFESSOR_PATH)
    return model


def load_morpheme_seq_whale(
    csv: Path = WHALE_CSV,
    n_bins: int = N_BINS,
) -> tuple[list[MorphemeConversation], dict[int, str]]:
    """Return one MorphemeConversation per sequenceId where .tokens is a flat
    list of morpheme token ids (Morfessor segmentation of each coda's ICI
    string) and .coda_boundaries marks where each new coda starts.

    decoder[morpheme_id] → morpheme string, e.g. 'BC'.
    """
    df = _load_joined(dia_csv=csv)

    valid_mask = (
        df["n_clicks"].notna()
        & df["n_clicks"].between(MIN_CLICKS, MAX_CLICKS)
    )
    edges = compute_global_bin_edges(df[valid_mask].copy(), n_bins=n_bins)

    # Compute ICI string per row (stored by integer index for fast lookup)
    ici_by_idx: dict[int, str | None] = {}
    for idx, row in df.iterrows():
        n = row.get("n_clicks")
        if pd.isna(n) or not (MIN_CLICKS <= int(n) <= MAX_CLICKS):
            ici_by_idx[idx] = None
            continue
        s = ici_row_to_string(row, edges)
        ici_by_idx[idx] = s if s else None

    # Build ICI corpus for potential Morfessor fallback training
    ici_corpus: Counter = Counter(
        s for s in ici_by_idx.values() if s is not None
    )

    model = _load_morfessor(ici_corpus)

    # Segment every unique ICI string once
    segs_cache: dict[str, list[str]] = {}
    all_morphemes: set[str] = set()
    for s in set(s for s in ici_by_idx.values() if s is not None):
        morphs, _ = model.viterbi_segment(s)
        segs_cache[s] = morphs
        all_morphemes.update(morphs)

    sorted_morphemes = sorted(all_morphemes)
    morph_to_id = {m: i + 1 for i, m in enumerate(sorted_morphemes)}
    decoder: dict[int, str] = {0: UNK_TOKEN}
    decoder.update({v: k for k, v in morph_to_id.items()})

    print(
        f"Morpheme vocab: {len(sorted_morphemes)} morphemes + <unk>  "
        f"(ICI strings: {len(segs_cache)}, "
        f"avg morphemes/string: {np.mean([len(v) for v in segs_cache.values()]):.2f})"
    )

    # Build MorphemeConversation objects
    df = df.sort_values(["sequenceId", "itemPosition"])
    conversations: list[MorphemeConversation] = []

    for seq_id, grp in df.groupby("sequenceId", sort=False):
        grp = grp.sort_values("itemPosition")
        speakers = grp["Whale"].astype(str).tolist()
        times = grp["TimeDelta"].astype(float).tolist()
        has_ts = grp["has_timestamps"].astype(int).tolist()

        switched_flags = [False]
        for i in range(1, len(speakers)):
            switched_flags.append(
                speakers[i] != speakers[i - 1]
                and has_ts[i] == 1
                and has_ts[i - 1] == 1
            )

        all_tokens: list[int] = []
        coda_boundaries: list[int] = []
        dt_buckets_expanded: list[int] = []
        times_per_coda: list[float] = []
        speakers_per_coda: list[str] = []

        for coda_idx, (df_idx, _) in enumerate(grp.iterrows()):
            bucket = dt_to_bucket(
                "whale",
                times[coda_idx],
                has_ts[coda_idx],
                switched=switched_flags[coda_idx],
            )
            ici_str = ici_by_idx.get(df_idx)
            if ici_str is not None:
                morphs = segs_cache[ici_str]
                morph_ids = [morph_to_id.get(m, UNK_ID) for m in morphs]
            else:
                morph_ids = [UNK_ID]

            coda_boundaries.append(len(all_tokens))
            all_tokens.extend(morph_ids)
            dt_buckets_expanded.extend([bucket] * len(morph_ids))
            times_per_coda.append(times[coda_idx])
            speakers_per_coda.append(speakers[coda_idx])

        conversations.append(MorphemeConversation(
            seq_id=str(seq_id),
            tokens=all_tokens,
            coda_boundaries=coda_boundaries,
            dt_buckets=dt_buckets_expanded,
            times=times_per_coda,
            speakers=speakers_per_coda,
        ))

    total_morphemes = sum(len(c.tokens) for c in conversations)
    total_codas = sum(len(c.coda_boundaries) for c in conversations)
    print(
        f"Built {len(conversations)} MorphemeConversation objects  "
        f"({total_codas} codas → {total_morphemes} morpheme tokens, "
        f"ratio={total_morphemes/total_codas:.2f})"
    )
    return conversations, decoder
