"""Stage 1.3 — numeric quality check on the 116 discovered Pacific clusters.

For each ``P{n}c{cid}`` cluster:

  1. Recompute centroid and per-member mean distance to centroid from the
     unified ICI matrix, restricted to the cluster's members.
  2. Find the nearest Sharma 'real' CodaType centroid of the same length
     and look up its mean distance to centroid in
     ``data/diagnostics/dswp_variance.csv``.
  3. Compute ``variance_ratio = cluster_mean_dist /
     sharma_mean_dist_to_centroid``.

Clusters with ``variance_ratio > 2.0`` are flagged for the
vocabulary-pruning step in Stage 4 (collapse into ``OTHER_PACIFIC``).
This is the numeric replacement for visual / piano-roll review of every
cluster.

Outputs:

  * ``data/diagnostics/hybrid_v1/cluster_quality.csv`` — per-cluster row.
  * Stdout summary with the fraction of clusters above the 2× ratio.

Run from repo root::

    python -m src.validation.cluster_quality
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
UNIFIED = REPO / "data" / "upstream" / "codas_unified.csv"
ASSIGNMENTS = (REPO / "data" / "diagnostics" / "hybrid_v1"
                / "assignments.csv")
VARIANCE_CSV = REPO / "data" / "diagnostics" / "dswp_variance.csv"
OUT_CSV = (REPO / "data" / "diagnostics" / "hybrid_v1"
            / "cluster_quality.csv")

VARIANCE_RATIO_FLOOR = 2.0


def _ici_cols(n: int) -> list[str]:
    return [f"ICI{i}" for i in range(1, n)]


def _parse_centroid(s: str) -> np.ndarray:
    """``dswp_variance.csv`` stores centroids as Python list literals
    (e.g. ``"[0.196, 0.128]"``)."""
    return np.asarray(ast.literal_eval(s), dtype=float)


def main() -> pd.DataFrame:
    if not (UNIFIED.exists() and ASSIGNMENTS.exists()
            and VARIANCE_CSV.exists()):
        missing = [p for p in (UNIFIED, ASSIGNMENTS, VARIANCE_CSV)
                   if not p.exists()]
        raise FileNotFoundError(f"missing inputs: {missing}")

    print("=== cluster-quality numeric check ===")
    u = pd.read_csv(UNIFIED, low_memory=False)
    a = pd.read_csv(ASSIGNMENTS, low_memory=False)
    var = pd.read_csv(VARIANCE_CSV)

    # Restrict variance reference to Sharma's 'real' (non-NOISE) types.
    sharma_real = var[~var["is_noise_type"]].copy()
    sharma_real["centroid_arr"] = sharma_real["centroid"].map(_parse_centroid)

    # Index of discovered-cluster member rows in unified.
    disc = a[a["origin"] == "discovery-cluster"].copy()
    disc = disc.merge(
        u[["n_clicks"] + _ici_cols(10)].assign(_ix=u.index),
        left_on="unified_index", right_on="_ix", how="left",
    )

    rows = []
    for cluster_label, members in disc.groupby("final_label", sort=True):
        if not isinstance(cluster_label, str):
            continue
        n_clicks = int(members["n_clicks_x"].iloc[0]
                       if "n_clicks_x" in members else
                       members["n_clicks"].iloc[0])
        cols = _ici_cols(n_clicks)
        # Use the assignments row's n_clicks (the merge above carries _y too)
        # to pick ICI columns.
        X = members[cols].to_numpy(dtype=float)
        n_members = int(len(X))
        centroid = X.mean(axis=0)
        mean_dist = float(np.linalg.norm(X - centroid, axis=1).mean())

        # Nearest Sharma real type of same length.
        same_len = sharma_real[sharma_real["length"] == n_clicks]
        if same_len.empty:
            nearest_type = ""
            sharma_dist = float("nan")
            ratio = float("nan")
        else:
            sharma_centroids = np.stack(same_len["centroid_arr"].to_list())
            d = np.linalg.norm(sharma_centroids - centroid, axis=1)
            j = int(d.argmin())
            nearest_type = str(same_len.iloc[j]["codatype"])
            sharma_dist = float(same_len.iloc[j]["mean_dist_to_centroid"])
            ratio = (mean_dist / sharma_dist) if sharma_dist > 0 else float("nan")

        rows.append({
            "cluster_label": cluster_label,
            "n_clicks": n_clicks,
            "n_members": n_members,
            "cluster_mean_dist_to_centroid": mean_dist,
            "nearest_sharma_type": nearest_type,
            "sharma_mean_dist_to_centroid": sharma_dist,
            "variance_ratio": ratio,
            "flag_collapse": (ratio > VARIANCE_RATIO_FLOOR
                              if np.isfinite(ratio) else False),
        })

    df = pd.DataFrame(rows).sort_values(
        ["n_clicks", "cluster_label"]).reset_index(drop=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    n_total = len(df)
    n_flagged = int(df["flag_collapse"].sum())
    print(f"  {n_total} discovered clusters analysed; "
          f"{n_flagged} flagged (ratio > {VARIANCE_RATIO_FLOOR:.1f}× "
          f"nearest Sharma-type internal distance)")
    print(f"-> {OUT_CSV.relative_to(REPO)}")
    return df


if __name__ == "__main__":
    main()
