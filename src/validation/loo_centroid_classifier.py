"""Leave-one-out sanity check on B_classify_optics's centroid-radius classifier.

The new pipeline groups DSWP rows by ground-truth ``CodaType`` and uses the
per-type centroid + ``radius_95`` (floored at ``RADIUS_FLOOR_S``) as the
matcher for non-DSWP rows. Phase 1's 96 % accuracy was for ELKI OPTICSXi's
sub-cluster partition — *not* for this group-by-CodaType partition. This
script gives the corresponding number for the centroid+radius classifier:

For every labelled DSWP row x of length n,
  1. recompute the centroid of x's CodaType *excluding x* (LOO),
  2. find the nearest live (non-NOISE) CodaType centroid,
  3. if the nearest distance ≤ max(radius_95, RADIUS_FLOOR_S):
        predicted = dominant_codatype of the nearest group
     else:
        predicted = "*-OUTLIER" (would have been routed to stage-3)
  4. compare predicted to x's ground-truth CodaType.

Writes a per-length and aggregate accuracy table to stdout.

Run from repo root::

    python -m src.validation.loo_centroid_classifier
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.B_classify_optics import (
    LENGTH_RANGE, RADIUS_FLOOR_S, RADIUS_PCTILE,
)

REPO = Path(__file__).resolve().parents[2]


def _load() -> pd.DataFrame:
    u = pd.read_csv(REPO / "data" / "upstream" / "codas_unified.csv",
                    low_memory=False)
    d = pd.read_csv(REPO / "data" / "upstream" / "dswp_dominica_codas.csv")
    d.columns = [c.strip().lstrip("﻿") for c in d.columns]
    lookup = d.set_index(d["codaNUM2018"].astype(str))["CodaType"]
    is_dswp = u["source"] == "sharma2024_dswp"
    u["_codatype"] = pd.Series(pd.NA, index=u.index, dtype="object")
    u.loc[is_dswp, "_codatype"] = (u.loc[is_dswp, "source_coda_id"]
                                   .astype(str).map(lookup).to_numpy())
    return u


def main():
    u = _load()
    is_dswp = u["source"] == "sharma2024_dswp"

    print(f"=== LOO centroid+radius classifier on DSWP "
          f"(RADIUS_FLOOR_S={RADIUS_FLOOR_S}, percentile={RADIUS_PCTILE}) ===")
    grand_correct = 0
    grand_total = 0
    grand_outlier = 0
    for n in LENGTH_RANGE:
        cols = [f"ICI{i}" for i in range(1, n)]
        mask = (is_dswp & (u["n_clicks"] == n)
                & u["_codatype"].notna()
                & u[cols].notna().all(axis=1))
        idx = u.index[mask].to_numpy()
        if len(idx) == 0:
            continue
        X = u.loc[idx, cols].to_numpy(dtype=float)
        y = u.loc[idx, "_codatype"].astype(str).to_numpy()

        # Per-CodaType: members, centroid, radius_95 (full set; we'll
        # adjust per-row for LOO). Drop NOISE groups from the live set.
        types = sorted(np.unique(y))
        live_types = [t for t in types if not t.endswith("-NOISE")]
        if not live_types:
            continue

        type_to_idx = {t: np.where(y == t)[0] for t in types}
        type_centroid = {t: X[type_to_idx[t]].mean(axis=0) for t in types}
        type_r95 = {}
        for t in types:
            members = X[type_to_idx[t]]
            d = np.linalg.norm(members - type_centroid[t], axis=1)
            type_r95[t] = max(float(np.percentile(d, RADIUS_PCTILE))
                              if d.size else 0.0,
                              RADIUS_FLOOR_S)

        live_centroids = np.stack([type_centroid[t] for t in live_types])
        live_radii = np.array([type_r95[t] for t in live_types], dtype=float)

        correct = 0
        outlier = 0
        for i in range(len(X)):
            true_t = y[i]
            # LOO adjust: replace true_t's centroid with the leave-i-out
            # version; recompute its radius_95 over members \ {i}.
            t_idxs = type_to_idx[true_t]
            other = t_idxs[t_idxs != i]
            if len(other) > 0:
                cen_loo = X[other].mean(axis=0)
                dists_loo = np.linalg.norm(X[other] - cen_loo, axis=1)
                r95_loo = max(float(np.percentile(dists_loo, RADIUS_PCTILE))
                              if dists_loo.size else 0.0,
                              RADIUS_FLOOR_S)
            else:
                cen_loo = type_centroid[true_t]
                r95_loo = type_r95[true_t]

            # Build a temporary live-centroid array swapping in the LOO
            # version when true_t is in live (i.e., not a NOISE row).
            if true_t in live_types:
                j = live_types.index(true_t)
                centroids_for_row = live_centroids.copy()
                centroids_for_row[j] = cen_loo
                radii_for_row = live_radii.copy()
                radii_for_row[j] = r95_loo
            else:
                centroids_for_row = live_centroids
                radii_for_row = live_radii

            d_to_each = np.linalg.norm(centroids_for_row - X[i], axis=1)
            nearest = int(np.argmin(d_to_each))
            d_min = float(d_to_each[nearest])
            if d_min <= radii_for_row[nearest]:
                pred = live_types[nearest]
            else:
                pred = "*-OUTLIER"
                outlier += 1
            if pred == true_t:
                correct += 1

        total = len(X)
        print(f"  n={n:>2}: {correct:>5}/{total:<5} = {correct/total:6.2%}   "
              f"(would-be outliers: {outlier:>5} = {outlier/total:5.1%})")
        grand_correct += correct
        grand_total += total
        grand_outlier += outlier
    print(f"  total: {grand_correct:,}/{grand_total:,} = "
          f"{grand_correct/max(grand_total,1):.2%}   "
          f"(outliers: {grand_outlier:,} = "
          f"{grand_outlier/max(grand_total,1):.2%})")


if __name__ == "__main__":
    main()
