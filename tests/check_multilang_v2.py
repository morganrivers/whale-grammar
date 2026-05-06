"""Post-build verification for the v2 multilang corpus.

Run after src.grammar.multilang_loader writes
data/classified/multilang_dialogues.csv. Catches the most likely loader
regressions: leaking non-restricted Mandarin into the clean tiers, the
wrong overall missingness fraction, or the intra-word DT bucket failing
to fire.

Not a pytest module — invoke directly:

    micromamba run -n py311 python tests/check_multilang_v2.py

Exits 0 on pass, 1 on any assertion failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "classified" / "multilang_dialogues.csv"
ALLOWED = {"Tong", "Zhou3", "TCCM"}


def main() -> int:
    if not CSV.exists():
        print(f"ERROR: {CSV} not found — run multilang_loader first.")
        return 1

    df = pd.read_csv(CSV)
    df["tier"] = df["sequenceId"].str.split("::").str[0]

    # 1. Clean-tier rows only come from Tong, Zhou3, or TCCM.
    clean = df[df["tier"].isin(["zh_dswp", "zh_birth"])]
    if clean.empty:
        print("ERROR: no clean-tier (zh_dswp / zh_birth) rows found.")
        return 1
    clean_corpora = (
        clean["sequenceId"]
        .str.extract(r"^[^:]+::zh::([^:]+)::")[0]
        .dropna()
        .unique()
    )
    leaked = set(clean_corpora) - ALLOWED
    if leaked:
        print(f"FAIL #1 clean tier leaked non-restricted Mandarin: {leaked}")
        return 1
    print(f"OK #1 clean-tier corpora: {sorted(clean_corpora)}")

    # 2. Hersh-tier ZH portion uses the same restricted pool.
    hersh = df[df["tier"] == "multilang"]
    hersh_zh_corpora = (
        hersh["sequenceId"]
        .str.extract(r"^multilang::zh::([^:]+)::")[0]
        .dropna()
        .unique()
    )
    leaked = set(hersh_zh_corpora) - ALLOWED
    if leaked:
        print(f"FAIL #2 hersh-zh leaked non-restricted Mandarin: {leaked}")
        return 1
    print(f"OK #2 hersh-zh corpora: {sorted(hersh_zh_corpora)}")

    # 3. has_timestamps==0 fraction within v2 band.
    frac_missing = (df["has_timestamps"] == 0).mean()
    if not (0.65 <= frac_missing <= 0.75):
        print(f"FAIL #3 has_timestamps==0 fraction = {frac_missing:.3f} "
              f"(want 0.65–0.75)")
        return 1
    print(f"OK #3 has_timestamps==0 fraction = {frac_missing:.3f}")

    # 4. intra_word bucket fires: TimeDelta==0.0 on sub-tokens of multi-unit
    # lemmas in tiers where has_timestamps==1.
    n_intra = (
        (df["TimeDelta"] == 0.0)
        & (df["has_timestamps"] == 1)
        & (df["itemPosition"] > 0)
    ).sum()
    if n_intra <= 100:
        print(f"FAIL #4 intra_word bucket nearly empty: n_intra={n_intra}")
        return 1
    print(f"OK #4 intra_word rows: {n_intra:,}")

    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
