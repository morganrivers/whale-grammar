"""Leave-one-out test for the kNN+τ matcher with τ = max(NOSC-p99, 0.10s).

Question the user asked: now that we widened τ to recover ~12k Pacific
codas, is the same matcher still right when applied to Sharma codas
themselves? Or are we over-classifying — accepting codas as a Sharma type
when they're really another Sharma type whose centroid is just inside τ?

For each Sharma 'real' coda x of length n:
  1. Hide x. Build kNN k=5 on the remaining Sharma-real rows of length n.
  2. Predict majority CodaType + nearest-neighbour distance.
  3. Look up τ = max(NOSC-p99, 0.10) for the **predicted** type.
  4. Decide:
       within τ + predicted == truth   → CORRECT-MATCH
       within τ + predicted != truth   → MIS-MATCH (over-classification)
       distance > τ                     → REJECT (would go to discovery)

Reports per-length and aggregate counts + percentages.

Run from repo root::

    python -m src.validation.loo_knn_tau
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from src.validation.hybrid_classify import (
    KNN_K, LENGTH_RANGE, _ici_cols, _is_noise_label, build_tau_lookup,
)
from src.validation.phase1b_knn import DOMINICA, UNIFIED, coda_type_for_dswp

REPO = Path(__file__).resolve().parents[2]
VARIANCE_CSV = REPO / "data" / "diagnostics" / "dswp_variance.csv"


def main() -> None:
    u = pd.read_csv(UNIFIED, low_memory=False)
    d = pd.read_csv(DOMINICA)
    d.columns = [c.strip().lstrip("﻿") for c in d.columns]
    u["coda_type_truth"] = coda_type_for_dswp(u, d)
    var = pd.read_csv(VARIANCE_CSV)
    tau = build_tau_lookup(var)

    print("=== LOO kNN+τ on Sharma 'real' types ===")
    print(f"  τ = max(NOSC-p99, 0.10 s); k = {KNN_K}")
    print(f"  τ-table: {len(tau)} entries; range {min(tau.values()):.4f} – "
          f"{max(tau.values()):.4f}\n")

    is_dswp = u["source"] == "sharma2024_dswp"
    truth = u["coda_type_truth"]
    is_real = is_dswp & truth.notna() & ~truth.map(_is_noise_label)

    print(f"  {'n':>2} {'total':>6}  "
          f"{'correct':>9} {'mismatch':>10} {'reject':>8}   "
          f"{'acc(in-τ)':>10}")
    grand = {"total": 0, "correct": 0, "mis": 0, "reject": 0}
    per_length = []
    per_row: list[dict] = []
    for n in LENGTH_RANGE:
        cols = _ici_cols(n)
        m = is_real & (u["n_clicks"] == n) & u[cols].notna().all(axis=1)
        idx = u.index[m].to_numpy()
        if len(idx) < KNN_K + 2:
            continue
        X = u.loc[idx, cols].to_numpy(dtype=float)
        y = u.loc[idx, "coda_type_truth"].astype(str).to_numpy()

        # k+1 neighbours so we can drop self
        knn = KNeighborsClassifier(n_neighbors=KNN_K + 1, metric="euclidean")
        knn.fit(X, y)
        D, I = knn.kneighbors(X, n_neighbors=KNN_K + 1)

        correct = mis = reject = 0
        clans_n = u.loc[idx, "clan"].astype("string").to_numpy()
        for i in range(len(X)):
            row_nbrs = I[i]
            row_dists = D[i]
            self_pos = np.where(row_nbrs == i)[0]
            if len(self_pos):
                keep = np.ones(KNN_K + 1, dtype=bool)
                keep[self_pos[0]] = False
                row_nbrs = row_nbrs[keep][:KNN_K]
                row_dists = row_dists[keep][:KNN_K]
            else:
                row_nbrs = row_nbrs[:KNN_K]
                row_dists = row_dists[:KNN_K]
            vals, counts = np.unique(y[row_nbrs], return_counts=True)
            pred = vals[counts.argmax()]
            nn_dist = float(row_dists[0])
            t = tau.get((n, str(pred)), float("inf"))
            if nn_dist > t:
                outcome = "reject"
                reject += 1
            elif pred == y[i]:
                outcome = "correct"
                correct += 1
            else:
                outcome = "mismatch"
                mis += 1
            per_row.append({
                "unified_index": int(idx[i]),
                "length": int(n),
                "clan": clans_n[i] if pd.notna(clans_n[i]) else "",
                "truth": str(y[i]),
                "predicted": str(pred),
                "nn_distance": nn_dist,
                "tau": float(t),
                "outcome": outcome,
            })
        total = len(X)
        in_tau = correct + mis
        in_tau_acc = (correct / in_tau) if in_tau else float("nan")
        print(f"  {n:>2} {total:>6,}  {correct:>9,} {mis:>10,} "
              f"{reject:>8,}   {in_tau_acc:>10.2%}")
        grand["total"] += total
        grand["correct"] += correct
        grand["mis"] += mis
        grand["reject"] += reject
        per_length.append({
            "length": n, "total": total, "correct": correct,
            "mismatch": mis, "reject": reject,
            "in_tau_accuracy": in_tau_acc,
        })

    g = grand
    in_tau = g["correct"] + g["mis"]
    in_tau_acc = (g["correct"] / in_tau) if in_tau else float("nan")
    print(f"\n  total {g['total']:>6,}  "
          f"{g['correct']:>9,} {g['mis']:>10,} {g['reject']:>8,}   "
          f"{in_tau_acc:>10.2%}")

    print(f"\nInterpretation:")
    print(f"  * Acceptance rate (within τ): "
          f"{(g['correct']+g['mis'])/g['total']:.1%} of Sharma 'real' codas "
          f"would have been kept by Stage 2.")
    print(f"  * Of those accepted, accuracy = {in_tau_acc:.2%}.")
    print(f"  * Over-classification (accepted as wrong type): "
          f"{g['mis']:,} = {g['mis']/g['total']:.2%} of all Sharma 'real' "
          f"codas, {g['mis']/max(in_tau,1):.2%} of accepted.")
    print(f"  * Rejection (would be sent to discovery despite being "
          f"Sharma-real): {g['reject']:,} = "
          f"{g['reject']/g['total']:.2%}")

    # write a tidy CSV next to the other diagnostics
    out_dir = REPO / "data" / "diagnostics" / "hybrid_v1"
    out_csv = out_dir / "loo_knn_tau.csv"
    pd.DataFrame(per_length + [{
        "length": "total", "total": g["total"], "correct": g["correct"],
        "mismatch": g["mis"], "reject": g["reject"],
        "in_tau_accuracy": in_tau_acc,
    }]).to_csv(out_csv, index=False)
    print(f"\n-> {out_csv.relative_to(REPO)}")

    # Per-row LOO outcomes for downstream slicing (per-clan, per-length).
    rows_csv = out_dir / "loo_knn_tau_rows.csv"
    pd.DataFrame(per_row).to_csv(rows_csv, index=False)
    print(f"-> {rows_csv.relative_to(REPO)}")

    # Per-clan over-classification breakdown.
    if per_row:
        rdf = pd.DataFrame(per_row)
        rdf["clan_group"] = rdf["clan"].where(rdf["clan"] != "", "(NaN)")
        clan_stats = (rdf.groupby("clan_group")
                      .apply(lambda x: pd.Series({
                          "total": len(x),
                          "correct": int((x["outcome"] == "correct").sum()),
                          "mismatch": int((x["outcome"] == "mismatch").sum()),
                          "reject": int((x["outcome"] == "reject").sum()),
                          "over_class_rate":
                              (x["outcome"] == "mismatch").mean(),
                      }), include_groups=False)
                      .reset_index())
        clan_csv = out_dir / "loo_knn_tau_by_clan.csv"
        clan_stats.to_csv(clan_csv, index=False)
        print(f"-> {clan_csv.relative_to(REPO)}")
        print("\nPer-clan over-classification (Sharma DSWP only):")
        print(f"  {'clan':<8}{'total':>8}{'mismatch':>10}{'over_class':>12}")
        for _, row in clan_stats.iterrows():
            print(f"  {row['clan_group']:<8}{int(row['total']):>8,}"
                  f"{int(row['mismatch']):>10,}"
                  f"{row['over_class_rate']:>12.2%}")


if __name__ == "__main__":
    main()
