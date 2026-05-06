"""
Build a compound whale token = (rhythm, tempo, ornamentation, rubato).

Sharma 2024 used a 4-character compound token (e.g. `i1.=`) to expose
more of the coda's articulatory detail. The unified-corpus
`whale_dialogues.csv` keeps these as separate columns:

    rhythm       <- Coda            (131 classes)
    tempo        <- Duration         (continuous; binned into 5 quintiles)
    ornamentation<- Ornamentation    (binary)
    rubato       <- Synchrony        (binary)

This module reads `whale_dialogues.csv` and returns:
  * a copy where `Coda` has been replaced by a compound integer id
    (only attested combinations, dense reindex);
  * a decode dict {compound_id -> (rhythm_label, tempo_bin, orn, rub)}
    for human-readable continuations.

Tempo binning is computed *once* on the full corpus (not per fold) so
the compound vocabulary is stable across runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WHALE_CSV = ROOT / "data" / "classified" / "whale_dialogues.csv"
RHYTHM_INDEX = ROOT / "data" / "classified" / "rhythm_class_index.csv"

N_TEMPO_BINS = 5


def _tempo_bin(duration: pd.Series, n_bins: int = N_TEMPO_BINS) -> pd.Series:
    """qcut Duration into n_bins equal-population tempo classes (0..n-1).
    Drops duplicate edges if the duration distribution has ties (returns
    fewer bins in that rare case)."""
    return pd.qcut(duration, q=n_bins, labels=False, duplicates="drop").astype(int)


def load_compound_whale(csv: Path = WHALE_CSV) -> tuple[pd.DataFrame, dict[int, str]]:
    """Read whale_dialogues.csv, replace `Coda` with the compound token id,
    and return (df, decoder).

    decoder[compound_id] -> readable label, e.g. ``"1+1+5 | t2 | orn0 | rub1"``.
    """
    df = pd.read_csv(csv)
    rhythm_idx = pd.read_csv(RHYTHM_INDEX).set_index("rhythm_class")["rhythm_class_18"]

    df = df.copy()
    df["TempoBin"] = _tempo_bin(df["Duration"])
    keys = pd.Series(list(zip(
        df["Coda"].astype(int),
        df["TempoBin"].astype(int),
        df["Ornamentation"].astype(int),
        df["Synchrony"].astype(int),
    )))
    codes, uniques = pd.factorize(keys, sort=True)
    df["CompoundCoda"] = codes.astype(int)

    decoder: dict[int, str] = {}
    for compound_id, key in enumerate(uniques):
        rhythm, tempo, orn, rub = key
        rh_label = str(rhythm_idx.get(int(rhythm), f"r{rhythm}"))
        decoder[compound_id] = f"{rh_label}|t{tempo}|o{orn}|r{rub}"

    # Replace Coda with the compound id so downstream code (per_seq, etc.)
    # works unchanged — the rest of the schema is preserved.
    df["Coda_orig"] = df["Coda"]
    df["Coda"] = df["CompoundCoda"]
    return df, decoder


def main() -> None:
    """CLI smoke test: print summary stats."""
    df, dec = load_compound_whale()
    print(f"compound vocab V={len(dec):,}")
    print(f"rows: {len(df):,}, sequences: {df['sequenceId'].nunique()}")
    print(f"tempo bin counts:")
    print(df["TempoBin"].value_counts().sort_index())
    print()
    print("first 8 decoded compound ids:")
    for cid in range(min(8, len(dec))):
        print(f"  {cid}: {dec[cid]}")


if __name__ == "__main__":
    main()
