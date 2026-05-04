"""Fetch the unified ICI corpus and training labels from whale-ici-data on GitHub.

Files land in data/upstream/ (committed). Re-runs reuse them. Pass
refresh=True to any helper, or --refresh on the entry point, to re-download.
"""
from pathlib import Path
import shutil
import urllib.request

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / "data" / "upstream"
RAW_BASE = "https://raw.githubusercontent.com/morganrivers/whale-ici-data/main"


def _fetch(upstream_path: str, *, refresh: bool = False) -> Path:
    """Ensure <RAW_BASE>/<upstream_path> is mirrored into UPSTREAM by basename;
    return its local path."""
    local = UPSTREAM / Path(upstream_path).name
    if local.exists() and not refresh:
        return local
    url = f"{RAW_BASE}/{upstream_path}"
    local.parent.mkdir(parents=True, exist_ok=True)
    print(f"A_load_unified: downloading {url}")
    try:
        with urllib.request.urlopen(url) as resp, open(local, "wb") as f:
            shutil.copyfileobj(resp, f)
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch {url}.\n"
            f"  Local target: {local}\n"
            f"  If your network is offline, manually place the file at the local target.\n"
            f"  Underlying error: {e}"
        )
    return local


def unified_csv(refresh: bool = False) -> Path:
    return _fetch("data/unified/codas_unified.csv", refresh=refresh)


def training_file(name: str, refresh: bool = False) -> Path:
    """For B_classify: rhythms.p, ornaments.p, dialogues.csv."""
    return _fetch(f"data/raw/{name}", refresh=refresh)


def dominica_codas_csv(refresh: bool = False) -> Path:
    """Gero's published EC coda labels (CodaType column = 21 non-NOISE types
    + *-NOISE flags). Ground truth for the OPTICSxi reverse-engineering."""
    return _fetch("data/raw/dswp_dominica_codas.csv", refresh=refresh)


def load(refresh: bool = False) -> pd.DataFrame:
    return pd.read_csv(unified_csv(refresh=refresh), low_memory=False)
