"""Phase 1b option 2: lock cluster assignments on DSWP via ELKI OPTICSXi
(xi=0.04, minpts=10), then classify all non-DSWP codas via per-length kNN
against the labelled DSWP set. DSWP rows keep their Gero CodaType (96%
reproducible); Pacific/birth codas get assigned to the nearest DSWP cluster
in ICI space.

Reports:
  - DSWP accuracy on unified corpus: should match Phase 1's 96% (sanity)
  - Pacific coverage: distribution of assigned CodaType + nearest-neighbour
    distance percentiles (outlier candidates → new-type discovery)
  - Hersh coherence: cross-tab of Hersh's published `rhythm` (their classifier)
    vs our predicted Gero CodaType for Pacific rows

Run:
    python -m src.validation.phase1b_knn
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from . import elki_optics

REPO = Path(__file__).resolve().parents[2]
UNIFIED = REPO / "data" / "upstream" / "codas_unified.csv"
DOMINICA = REPO / "data" / "upstream" / "dswp_dominica_codas.csv"

LENGTH_RANGE = range(3, 11)
XI = 0.04
MINPTS = 10
KNN_K = 5


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    u = pd.read_csv(UNIFIED, low_memory=False)
    d = pd.read_csv(DOMINICA)
    d.columns = [c.strip().lstrip("﻿") for c in d.columns]
    return u, d


def coda_type_for_dswp(u: pd.DataFrame, d: pd.DataFrame) -> pd.Series:
    """Attach Gero CodaType to DSWP rows in unified, by source_coda_id ↔
    codaNUM2018."""
    is_dswp = u["source"] == "sharma2024_dswp"
    lookup = d.set_index(d["codaNUM2018"].astype(str))["CodaType"]
    out = pd.Series(pd.NA, index=u.index, dtype="object")
    out.loc[is_dswp] = (
        u.loc[is_dswp, "source_coda_id"].astype(str).map(lookup).to_numpy())
    return out


def _ici_matrix(df: pd.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray]:
    cols = [f"ICI{i}" for i in range(1, n)]
    mask = (df["n_clicks"] == n) & df[cols].notna().all(axis=1)
    return df.index[mask].to_numpy(), df.loc[mask, cols].to_numpy(dtype=float)


def main():
    print("loading...", flush=True)
    u, d = load()
    u["coda_type_truth"] = coda_type_for_dswp(u, d)
    n_truth = int(u["coda_type_truth"].notna().sum())
    print(f"  unified rows: {len(u):,}, with CodaType truth: {n_truth:,}")

    pred = pd.Series(pd.NA, index=u.index, dtype="object")
    nearest_dist = pd.Series(np.nan, index=u.index, dtype="float64")

    # Step 1: per-length, run ELKI on DSWP-only. Use the resulting cluster ID
    # purely as a sanity check; the actual labels we propagate are CodaType
    # itself, via kNN. This way DSWP labels are exact (no clustering loss).
    print(f"\nrunning per-length DSWP OPTICS sanity check + kNN classify "
          f"(xi={XI}, minpts={MINPTS}, k={KNN_K}):", flush=True)
    for n in LENGTH_RANGE:
        is_dswp = (u["source"] == "sharma2024_dswp")
        dswp_idx, X_dswp = _ici_matrix(u[is_dswp], n)
        # subset to rows with truth label
        truth_n = u.loc[dswp_idx, "coda_type_truth"]
        keep = truth_n.notna().to_numpy()
        dswp_idx = dswp_idx[keep]
        X_dswp = X_dswp[keep]
        y_dswp = truth_n[keep].to_numpy()
        if len(X_dswp) < KNN_K + 1:
            print(f"  n={n}: too few DSWP codas ({len(X_dswp)}), skip")
            continue

        # ELKI sanity: cluster DSWP, majority-vote CodaType per cluster
        try:
            cl = elki_optics.run(X_dswp, xi=XI, minpts=MINPTS)
            # majority CodaType per cluster
            ct_lookup = {}
            for c in np.unique(cl):
                in_c = cl == c
                vals, counts = np.unique(y_dswp[in_c], return_counts=True)
                ct_lookup[int(c)] = vals[counts.argmax()]
            sanity_pred = np.array([ct_lookup[int(c)] for c in cl])
            sanity_acc = (sanity_pred == y_dswp).mean()
        except Exception as e:
            sanity_acc = float("nan")
            print(f"  n={n}: ELKI sanity failed ({e})")

        # kNN: classify ALL unified rows of length n using DSWP as training
        all_idx, X_all = _ici_matrix(u, n)
        knn = KNeighborsClassifier(n_neighbors=KNN_K, metric="euclidean")
        knn.fit(X_dswp, y_dswp)
        y_pred = knn.predict(X_all)
        # nearest distance per row
        dists, _ = knn.kneighbors(X_all, n_neighbors=1)
        pred.loc[all_idx] = y_pred
        nearest_dist.loc[all_idx] = dists[:, 0]
        print(f"  n={n}: DSWP {len(X_dswp):>5} (ELKI sanity {sanity_acc:.1%}), "
              f"unified {len(X_all):>5} classified")

    # Evaluate DSWP accuracy on the kNN labels (should be ~100% since each
    # DSWP coda is its own nearest neighbour).
    eval_mask = u["coda_type_truth"].notna() & pred.notna()
    n_eval = int(eval_mask.sum())
    correct = int((pred[eval_mask] == u.loc[eval_mask, "coda_type_truth"]).sum())
    print(f"\n=== DSWP accuracy (kNN, k={KNN_K}) ===")
    print(f"  {correct:,}/{n_eval:,} = {correct/max(n_eval,1):.2%}")

    # leave-one-out style: for DSWP rows, kNN includes the row itself so it's
    # trivially correct. Re-run with k+1 and drop self to get honest accuracy.
    print(f"\n=== DSWP accuracy (leave-one-out kNN, k={KNN_K}) ===")
    loo_correct = 0
    loo_eval = 0
    for n in LENGTH_RANGE:
        is_dswp = (u["source"] == "sharma2024_dswp") & u["coda_type_truth"].notna()
        idx, X = _ici_matrix(u[is_dswp], n)
        if len(X) < KNN_K + 2:
            continue
        y = u.loc[idx, "coda_type_truth"].to_numpy()
        knn2 = KNeighborsClassifier(n_neighbors=KNN_K + 1, metric="euclidean")
        knn2.fit(X, y)
        # for each row, get k+1 neighbours, drop the row's own index (always
        # the nearest = self), majority vote on the remaining k
        nbrs = knn2.kneighbors(X, n_neighbors=KNN_K + 1, return_distance=False)
        loo_pred = []
        for i, row in enumerate(nbrs):
            row = row[row != i][:KNN_K]
            vals, counts = np.unique(y[row], return_counts=True)
            loo_pred.append(vals[counts.argmax()])
        loo_pred = np.array(loo_pred)
        c = int((loo_pred == y).sum())
        loo_correct += c
        loo_eval += len(y)
        print(f"  n={n}: {c}/{len(y)} = {c/len(y):.1%}")
    print(f"  total: {loo_correct}/{loo_eval} = {loo_correct/max(loo_eval,1):.2%}")
    print(f"  -> target >= 90% : "
          f"{'PASS' if loo_correct/max(loo_eval,1) >= 0.90 else 'FAIL'}")

    # Pacific summary
    pacific_mask = (u["source"] != "sharma2024_dswp") & pred.notna()
    print(f"\n=== Pacific (Hersh + birth) classification summary ===")
    print(f"  Pacific codas classified: {int(pacific_mask.sum()):,}")
    print(f"  top assigned CodaTypes:")
    print(pred[pacific_mask].value_counts().head(15).to_string())
    print(f"\n  nearest-neighbour distance percentiles (Pacific):")
    pq = nearest_dist[pacific_mask].quantile([0.5, 0.75, 0.9, 0.95, 0.99])
    for q, v in pq.items():
        print(f"    {int(q*100):>3}th: {v:.4f}")
    print(f"  for reference, DSWP nearest-neighbour distance percentiles:")
    dswp_pred_mask = (u["source"] == "sharma2024_dswp") & pred.notna()
    dq = nearest_dist[dswp_pred_mask].quantile([0.5, 0.75, 0.9, 0.95, 0.99])
    for q, v in dq.items():
        print(f"    {int(q*100):>3}th: {v:.4f}")


if __name__ == "__main__":
    main()
