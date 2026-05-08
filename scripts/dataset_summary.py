"""
dataset_summary.py — quick audit of the three upstream coda datasets:
  hersh2022_pacific, sharma2024_dswp, sharma2025_birth

Run from the repo root:
    python scripts/dataset_summary.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("Install pandas/numpy first:  pip install pandas numpy")

UNIFIED  = ROOT / "data/upstream/codas_unified.csv"
DSWP_RAW = ROOT / "data/upstream/dswp_dominica_codas.csv"

df       = pd.read_csv(UNIFIED, low_memory=False)
dswp_raw = pd.read_csv(DSWP_RAW)

SOURCES = {
    "hersh2022_pacific": "Hersh 2022 (Pacific)",
    "sharma2024_dswp":   "Sharma 2024 DSWP",
    "sharma2025_birth":  "Sharma 2025 Birth",
}

SEP = "=" * 70

def pct(n, total):
    return f"{n}/{total}  ({100*n/total:.1f}%)"

def nonempty(series):
    return (series.notna() & (series.astype(str).str.strip() != "") &
            (~series.astype(str).isin(["0", "nan"]))).sum()

print(SEP)
print("DATASET AUDIT — codas_unified.csv")
print(SEP)
print(f"Total rows: {len(df)}")
print()

for src_key, label in SOURCES.items():
    sub = df[df["source"] == src_key].copy()
    N = len(sub)
    print(SEP)
    print(f"  {label}  |  N = {N}")
    print(SEP)

    # ── LOCATION ───────────────────────────────────────────────────────────
    print("\n  [LOCATION]")
    loc_n  = nonempty(sub["location"])
    lat_n  = nonempty(sub["latitude"])
    lon_n  = nonempty(sub["longitude"])
    print(f"  location text  : {pct(loc_n, N)}")
    if loc_n:
        locs = sub["location"].dropna().unique()
        print(f"    values ({len(locs)}): {list(locs)[:5]}")
    print(f"  latitude       : {pct(lat_n, N)}")
    print(f"  longitude      : {pct(lon_n, N)}")
    if lat_n:
        lats = sub["latitude"].dropna()
        def dp(s):
            s2 = s.astype(str)
            return s2.apply(lambda x: len(x.split(".")[-1]) if "." in x else 0).median()
        print(f"    median decimal places: lat={dp(lats):.0f}")
        print(f"    range: {lats.min():.3f} to {lats.max():.3f}")
        print(f"    distinct lat values: {lats.nunique()}  (across {sub['recording_id'].nunique()} recording IDs)")

    # ── TIMESTAMP ──────────────────────────────────────────────────────────
    print("\n  [TIMESTAMP / TEMPORAL]")
    date_n   = sub["date"].notna().sum()
    offset_n = sub["time_in_recording_s"].notna().sum()
    print(f"  date field          : {pct(date_n, N)}")
    if date_n:
        sample_dates = sub["date"].dropna().unique()[:3]
        print(f"    sample: {list(sample_dates)}")
    print(f"  time_in_recording_s : {pct(offset_n, N)}")
    if offset_n:
        t = sub["time_in_recording_s"].dropna()
        print(f"    range within recordings: {t.min():.1f}s – {t.max():.1f}s")

    # ── WHALE / SOCIAL IDENTITY ────────────────────────────────────────────
    print("\n  [WHALE INDIVIDUAL IDENTITY]")
    pid_n  = nonempty(sub["whale_photo_id"])
    spk_n  = nonempty(sub["local_speaker_id"])
    clan_n = nonempty(sub["clan"])
    unit_n = nonempty(sub["social_unit"])
    print(f"  whale_photo_id   : {pct(pid_n, N)}")
    if pid_n:
        print(f"    unique IDs: {sub['whale_photo_id'].dropna().nunique()}")
    print(f"  local_speaker_id : {pct(spk_n, N)}")
    if spk_n:
        print(f"    unique IDs: {sub['local_speaker_id'].dropna().nunique()}")
    print(f"  social_unit      : {pct(unit_n, N)}")
    print(f"  clan             : {pct(clan_n, N)}")
    if clan_n:
        clans = sorted(sub["clan"].dropna().astype(str).unique())
        print(f"    clans: {clans}")
    print()

# ── DSWP raw IDN field (named individual whales) ─────────────────────────
print(SEP)
print("  DSWP raw file — IDN (named individual photo-ID)")
print(SEP)
N_raw = len(dswp_raw)
idn_named = (dswp_raw["IDN"].astype(str) != "0").sum()
idn_unambig = dswp_raw["IDN"].astype(str).str.match(r"^\d+$") & (dswp_raw["IDN"].astype(str) != "0")
print(f"  Named (IDN != 0)    : {pct(idn_named, N_raw)}")
print(f"  Unambiguous numeric : {pct(idn_unambig.sum(), N_raw)}")
ambig_pat = r"/|\?"
print(f"  Ambiguous (/, ?)    : {(dswp_raw['IDN'].astype(str).str.contains(ambig_pat)).sum()}")
print(f"  Unique IDs (incl 0) : {dswp_raw['IDN'].nunique()}")

print()
print(SEP)
print("  SUMMARY TABLE")
print(SEP)
rows = []
for src_key, label in SOURCES.items():
    sub = df[df["source"] == src_key]
    N   = len(sub)
    rows.append({
        "Dataset":        label,
        "N":              N,
        "Location text":  f"{pct(nonempty(sub['location']), N)}",
        "Lat/Lon":        f"{pct(nonempty(sub['latitude']), N)}",
        "Date":           f"{sub['date'].notna().sum()}/{N}",
        "Time offset":    f"{sub['time_in_recording_s'].notna().sum()}/{N}",
        "Photo ID":       f"{pct(nonempty(sub['whale_photo_id']), N)}",
        "Speaker ID":     f"{pct(nonempty(sub['local_speaker_id']), N)}",
        "Clan":           f"{pct(nonempty(sub['clan']), N)}",
    })

sumdf = pd.DataFrame(rows).set_index("Dataset")
print(sumdf.to_string())
print()
print("Done.")
