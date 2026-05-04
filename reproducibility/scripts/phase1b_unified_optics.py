"""Phase 1b: run ELKI OPTICSXi (xi=0.04, minpts=10) on the unified corpus
(Hersh Pacific + Sharma DSWP + Sharma birth) and check that DSWP codas still
match Gero's published CodaType. Target: >= 90% on DSWP rows alone.

The point of this run is NOT to relabel everything — it's to confirm that
adding Pacific data doesn't fracture the EC clusters Gero identified. If
DSWP codas land in the same clusters they did before (now sharing those
clusters with whatever Pacific codas happen to be similar), accuracy on
DSWP-only stays >= 90% and the joint algorithm is sound.

Run:
    python -m src.validation.reproduce_gero21_unified
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import elki_optics

REPO = Path(__file__).resolve().parents[2]
UNIFIED = REPO / "data" / "upstream" / "codas_unified.csv"
DOMINICA = REPO / "data" / "upstream" / "dswp_dominica_codas.csv"

import argparse

LENGTH_RANGE = range(3, 11)
XI = 0.04
DEFAULT_MINPTS = 10


def load() -> pd.DataFrame:
    u = pd.read_csv(UNIFIED, low_memory=False)
    d = pd.read_csv(DOMINICA)
    d.columns = [c.strip().lstrip("﻿") for c in d.columns]

    # join CodaType onto the DSWP rows of the unified corpus
    d_lookup = d[["codaNUM2018", "CodaType"]].copy()
    d_lookup["codaNUM2018"] = d_lookup["codaNUM2018"].astype(str)
    u["coda_type_truth"] = pd.NA
    is_dswp = u["source"] == "sharma2024_dswp"
    u_dswp = u.loc[is_dswp, ["source_coda_id"]].copy()
    u_dswp["source_coda_id"] = u_dswp["source_coda_id"].astype(str)
    merged = u_dswp.merge(d_lookup, left_on="source_coda_id",
                          right_on="codaNUM2018", how="left")
    u.loc[is_dswp, "coda_type_truth"] = merged["CodaType"].to_numpy()
    return u


def cluster_per_length(u: pd.DataFrame, *, xi: float, minpts: int):
    """Run ELKI per length bucket. Return cluster id per row (or NaN if the
    row was excluded from any bucket)."""
    cluster_ids = pd.Series(pd.NA, index=u.index, dtype="Int64")
    next_cluster_id = 0
    bucket_summary = []
    for n in LENGTH_RANGE:
        ici_cols = [f"ICI{i}" for i in range(1, n)]
        mask = (u["n_clicks"] == n) & u[ici_cols].notna().all(axis=1)
        sub = u.loc[mask, ici_cols].to_numpy(dtype=float)
        if len(sub) < minpts:
            bucket_summary.append((n, int(mask.sum()), 0))
            continue
        print(f"  ELKI: n={n} ({len(sub)} codas, minpts={minpts})...", flush=True)
        labels = elki_optics.run(sub, xi=xi, minpts=minpts)
        # remap to globally unique cluster ids
        local_to_global = {}
        global_labels = np.empty(len(labels), dtype=np.int64)
        for i, c in enumerate(labels):
            if c not in local_to_global:
                local_to_global[c] = next_cluster_id
                next_cluster_id += 1
            global_labels[i] = local_to_global[c]
        cluster_ids.loc[mask] = global_labels
        bucket_summary.append((n, int(mask.sum()), len(local_to_global)))
        print(f"    -> {len(local_to_global)} clusters in this bucket")
    return cluster_ids, bucket_summary


def evaluate(u: pd.DataFrame, cluster_ids: pd.Series):
    df = u[["source", "n_clicks", "coda_type_truth"]].copy()
    df["cluster_id"] = cluster_ids

    # majority CodaType per cluster, computed only over DSWP rows that have a
    # truth label. Pacific-only clusters get no label assignment.
    has_truth = df["coda_type_truth"].notna() & df["cluster_id"].notna()
    cluster_to_truth = (
        df.loc[has_truth]
          .groupby("cluster_id")["coda_type_truth"]
          .agg(lambda s: s.mode().iat[0]))
    pred = df["cluster_id"].map(cluster_to_truth)

    # restrict eval to DSWP rows with a truth label and a cluster id
    eval_mask = df["coda_type_truth"].notna() & df["cluster_id"].notna()
    n_eval = int(eval_mask.sum())
    correct = int(((pred == df["coda_type_truth"]) & eval_mask).sum())
    acc = correct / max(n_eval, 1)

    # per-length breakdown
    per_n = []
    for n in LENGTH_RANGE:
        m = eval_mask & (df["n_clicks"] == n)
        ne = int(m.sum())
        c = int(((pred == df["coda_type_truth"]) & m).sum())
        per_n.append((n, ne, c, c/max(ne,1) if ne else float("nan")))

    # cluster anatomy: what fraction of DSWP rows ended up in clusters whose
    # majority is a NOISE label vs a named type
    cluster_majority_is_noise = cluster_to_truth.astype(str).str.contains("NOISE", na=False)
    pred_is_noise_majority = pred.astype("string").str.contains("NOISE", na=False)
    dswp_rows = df["coda_type_truth"].notna() & df["cluster_id"].notna()
    dswp_in_noise_cluster = int((pred_is_noise_majority & dswp_rows).sum())

    # how many clusters have NO DSWP overlap (= "new" Pacific-only clusters)?
    all_clusters = set(df.loc[df["cluster_id"].notna(), "cluster_id"].unique())
    pacific_only = sorted(all_clusters - set(cluster_to_truth.index))
    return {
        "accuracy": acc,
        "n_eval": n_eval,
        "n_correct": correct,
        "per_n": per_n,
        "n_clusters_total": len(all_clusters),
        "n_clusters_with_dswp": len(cluster_to_truth),
        "n_clusters_pacific_only": len(pacific_only),
        "dswp_in_noise_cluster": dswp_in_noise_cluster,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--minpts", type=int, default=DEFAULT_MINPTS)
    args = p.parse_args()
    minpts = args.minpts

    print(f"loading unified + dominica join...", flush=True)
    u = load()
    print(f"  unified rows: {len(u):,}")
    print(f"  with CodaType ground truth: {u['coda_type_truth'].notna().sum():,} "
          f"(DSWP)")
    print(f"  in 3..10 click range: {u['n_clicks'].between(3,10).sum():,}")
    print(f"\nrunning ELKI per length (xi={XI}, minpts={minpts}):", flush=True)
    cluster_ids, bucket_summary = cluster_per_length(u, xi=XI, minpts=minpts)

    print(f"\n=== per-bucket cluster counts on unified data ===")
    for n, n_codas, n_clusters in bucket_summary:
        print(f"  n={n}: {n_codas:>6} codas -> {n_clusters} clusters")

    res = evaluate(u, cluster_ids)
    print(f"\n=== Phase 1b: DSWP rows still match Gero's CodaType? ===")
    print(f"  rows evaluated (DSWP w/ truth + cluster): {res['n_eval']:,}")
    print(f"  correct: {res['n_correct']:,}")
    print(f"  accuracy: {res['accuracy']:.2%}")
    print(f"  -> target >= 90% : "
          f"{'PASS' if res['accuracy'] >= 0.90 else 'FAIL'}")

    print(f"\n  per-length:")
    for n, ne, c, a in res["per_n"]:
        print(f"    n={n}: {c}/{ne} = {a:.1%}" if ne else f"    n={n}: (no eval rows)")

    print(f"\n=== cluster anatomy on unified data ===")
    print(f"  total clusters: {res['n_clusters_total']}")
    print(f"  clusters that contain >=1 DSWP coda: {res['n_clusters_with_dswp']}")
    print(f"  Pacific-only clusters (no DSWP overlap): "
          f"{res['n_clusters_pacific_only']}")
    print(f"  DSWP codas landing in NOISE-majority clusters: "
          f"{res['dswp_in_noise_cluster']:,}")


if __name__ == "__main__":
    main()
