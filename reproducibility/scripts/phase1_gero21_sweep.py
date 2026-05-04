"""Phase 1 reverse-engineering: find ELKI OPTICSXi parameters that reproduce
Gero's published CodaType labels on data/upstream/dswp_dominica_codas.csv.

Sweeps (xi, minpts) per length bucket, computes:
  - per-length cluster count
  - per-length majority-vote accuracy vs CodaType (the 33 published strings,
    incl. *-NOISE)
  - aggregate non-NOISE accuracy

Run:
    python -m src.validation.reproduce_gero21
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from . import elki_optics

REPO = Path(__file__).resolve().parents[2]
INPUT = REPO / "data" / "upstream" / "dswp_dominica_codas.csv"

LENGTH_RANGE = range(3, 11)
XI_VALUES = (0.04,)            # Gero's stated value; sweep elsewhere if needed
MINPTS_VALUES = (5, 7, 8, 9, 10, 11, 12, 15, 20, 30, 50)


def load() -> pd.DataFrame:
    df = pd.read_csv(INPUT)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    return df


def is_noise_label(s: str) -> bool:
    return isinstance(s, str) and "NOISE" in s


def majority_vote_purity(cluster_ids: np.ndarray, truth: np.ndarray):
    """For each cluster, the dominant truth label and its share. Returns a
    dict {cluster_id: (label, support, total)}."""
    out: dict[int, tuple[str, int, int]] = {}
    for c in np.unique(cluster_ids):
        in_c = cluster_ids == c
        labels, counts = np.unique(truth[in_c], return_counts=True)
        i = counts.argmax()
        out[int(c)] = (str(labels[i]), int(counts[i]), int(in_c.sum()))
    return out


def evaluate_bucket(df_n: pd.DataFrame, n: int, xi: float, minpts: int):
    ici_cols = [f"ICI{i}" for i in range(1, n)]
    X = df_n[ici_cols].to_numpy(dtype=float)
    truth = df_n["CodaType"].to_numpy()

    cluster_ids = elki_optics.run(X, xi=xi, minpts=minpts)
    purity = majority_vote_purity(cluster_ids, truth)

    # majority-vote prediction per coda
    cluster_to_label = {c: lbl for c, (lbl, _, _) in purity.items()}
    pred = np.array([cluster_to_label[c] for c in cluster_ids])

    correct = int((pred == truth).sum())
    n_total = len(truth)
    n_clusters = len(purity)

    # Gero's stated counts: non-NOISE distinct types per bucket from the data
    expected_types = sorted(set(t for t in truth if not is_noise_label(t)))
    pred_types = sorted(set(t for t in pred if not is_noise_label(t)))
    gero_n_types = len(expected_types)

    return {
        "n": n,
        "minpts": minpts,
        "xi": xi,
        "n_codas": n_total,
        "n_clusters": n_clusters,
        "n_pred_named_types": len(pred_types),
        "n_gero_named_types": gero_n_types,
        "accuracy": correct / max(n_total, 1),
        "purity": purity,
        "expected_types": expected_types,
        "pred_types": pred_types,
    }


def run_sweep(df: pd.DataFrame):
    rows = []
    for n in LENGTH_RANGE:
        df_n = df[df["nClicks"] == n].dropna(subset=[f"ICI{i}" for i in range(1, n)])
        if len(df_n) < max(MINPTS_VALUES):
            continue
        for xi, minpts in itertools.product(XI_VALUES, MINPTS_VALUES):
            print(f"  ELKI: n={n} xi={xi} minpts={minpts} ({len(df_n)} codas)...",
                  flush=True)
            try:
                r = evaluate_bucket(df_n, n, xi, minpts)
            except Exception as e:
                print(f"    failed: {e}")
                continue
            rows.append(r)
            print(f"    -> {r['n_clusters']} clusters "
                  f"({r['n_pred_named_types']} named vs Gero's {r['n_gero_named_types']}), "
                  f"acc={r['accuracy']:.1%}")
    return rows


def summarize(rows):
    print("\n=== sweep summary ===")
    cols = ["n", "minpts", "xi", "n_codas", "n_clusters",
            "n_pred_named_types", "n_gero_named_types", "accuracy"]
    print(f"{'n':>2} {'minpts':>6} {'xi':>6} {'codas':>6} {'clusters':>8} "
          f"{'named':>5}/{'gero':>5} {'acc':>6}")
    for r in rows:
        print(f"{r['n']:>2} {r['minpts']:>6} {r['xi']:>6.3f} "
              f"{r['n_codas']:>6} {r['n_clusters']:>8} "
              f"{r['n_pred_named_types']:>5}/{r['n_gero_named_types']:>5} "
              f"{r['accuracy']:>6.1%}")

    by_n: dict[int, list] = {}
    by_ms: dict[int, list] = {}
    for r in rows:
        by_n.setdefault(r["n"], []).append(r)
        by_ms.setdefault(r["minpts"], []).append(r)

    print("\n=== aggregate accuracy at fixed minpts ===")
    print(f"{'minpts':>6} {'aggregate':>10} {'codas':>7}")
    best_fixed = (None, 0.0)
    for ms in sorted(by_ms):
        num = sum(int(round(r["accuracy"] * r["n_codas"])) for r in by_ms[ms])
        den = sum(r["n_codas"] for r in by_ms[ms])
        agg = num / max(den, 1)
        marker = " <- best fixed" if agg > best_fixed[1] else ""
        if agg > best_fixed[1]:
            best_fixed = (ms, agg)
        print(f"{ms:>6} {agg:>10.2%} {den:>7}{marker}")

    print("\n=== best minpts per length (oracle, upper bound) ===")
    aggregate_correct = 0
    aggregate_total = 0
    for n, group in sorted(by_n.items()):
        best = max(group, key=lambda x: x["accuracy"])
        aggregate_correct += int(best["accuracy"] * best["n_codas"])
        aggregate_total += best["n_codas"]
        print(f"  n={n}: best minpts={best['minpts']}, "
              f"clusters={best['n_clusters']}, acc={best['accuracy']:.1%}")
    if aggregate_total:
        print(f"\nOracle aggregate (per-bucket optimal): "
              f"{aggregate_correct/aggregate_total:.2%} on {aggregate_total} codas")
        print(f"Best fixed minpts: {best_fixed[0]} -> {best_fixed[1]:.2%}")


def main():
    df = load()
    in_range = df["nClicks"].between(min(LENGTH_RANGE), max(LENGTH_RANGE))
    print(f"loaded {len(df)} codas; in 3..10 range: {int(in_range.sum())}")
    rows = run_sweep(df[in_range])
    summarize(rows)


if __name__ == "__main__":
    main()
