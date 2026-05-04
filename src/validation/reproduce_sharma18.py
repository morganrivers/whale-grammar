"""Phase 1 validation: per-length OPTICS(xi=0.04) on the labelled DSWP subset
should reproduce Sharma's 18 rhythm-class labels.

Inputs:
  data/upstream/sw_combinatoriality_dialogues.csv   (3840 codas, ICI1..ICI28, nClicks)
  data/upstream/sw_combinatoriality_rhythms.p       (3840 rhythm labels 0..17)

Method (Gero 2016, §2.2.2):
  - filter codas to 3..10 clicks
  - per length bucket: OPTICS(cluster_method='xi', xi=0.04, metric='euclidean')
    on absolute ICI vectors of dimension (n_clicks - 1)
  - assign each cluster to the Sharma rhythm it overlaps most with
    (majority vote; noise stays unmapped)
  - report accuracy on non-noise codas, plus noise rate and ARI

Sweeps min_samples ∈ {3, 5, 10} since Gero's supplement does not state a value.

Run:
    python -m src.validation.reproduce_sharma18
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.metrics import adjusted_rand_score

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / "data" / "upstream"
DIALOGUES = UPSTREAM / "sw_combinatoriality_dialogues.csv"
RHYTHMS_P = UPSTREAM / "sw_combinatoriality_rhythms.p"

LENGTH_RANGE = range(3, 11)            # Gero's 3..10
XI = 0.04
MIN_SAMPLES_SWEEP = (3, 5, 10)


def load_labelled() -> pd.DataFrame:
    df = pd.read_csv(DIALOGUES)
    with open(RHYTHMS_P, "rb") as f:
        rhythms = pickle.load(f)
    if len(rhythms) != len(df):
        raise ValueError(
            f"dialogues ({len(df)}) and rhythms ({len(rhythms)}) length mismatch")
    df = df.reset_index(drop=True).copy()
    df["sharma_rhythm"] = pd.Series(rhythms, dtype="Int64")
    return df


def cluster_per_length(df: pd.DataFrame, *, xi: float, min_samples: int):
    """Run OPTICS per nClicks bucket. Returns array of cluster ids globally
    unique across buckets (-1 = noise)."""
    labels = np.full(len(df), -1, dtype=np.int64)
    next_cluster_id = 0
    summary = []
    for n in LENGTH_RANGE:
        mask = (df["nClicks"] == n).to_numpy()
        if mask.sum() < min_samples:
            summary.append((n, int(mask.sum()), 0, int(mask.sum())))
            continue
        ici_cols = [f"ICI{i}" for i in range(1, n)]
        X = df.loc[mask, ici_cols].to_numpy(dtype=float)
        if np.isnan(X).any():
            keep = ~np.isnan(X).any(axis=1)
            sub_idx = np.where(mask)[0][keep]
            X = X[keep]
        else:
            sub_idx = np.where(mask)[0]
        if len(X) < min_samples:
            summary.append((n, int(mask.sum()), 0, int(mask.sum())))
            continue
        opt = OPTICS(min_samples=min_samples, xi=xi,
                     cluster_method="xi", metric="euclidean")
        opt.fit(X)
        local = opt.labels_
        local_clusters = np.unique(local[local != -1])
        for old in local_clusters:
            labels[sub_idx[local == old]] = next_cluster_id
            next_cluster_id += 1
        n_noise = int((local == -1).sum())
        summary.append((n, int(mask.sum()), len(local_clusters), n_noise))
    return labels, summary


def majority_vote_mapping(cluster_ids: np.ndarray, truth: np.ndarray):
    """For each cluster, pick the truth label it overlaps most. Returns a
    predicted-label array (same shape as truth) where each non-noise coda
    inherits its cluster's majority label; noise codas stay as -1."""
    pred = np.full_like(truth, -1)
    for c in np.unique(cluster_ids):
        if c == -1:
            continue
        in_c = cluster_ids == c
        modes, counts = np.unique(truth[in_c], return_counts=True)
        pred[in_c] = modes[counts.argmax()]
    return pred


def report(df: pd.DataFrame, *, min_samples: int):
    truth = df["sharma_rhythm"].to_numpy(dtype=np.int64)
    cluster_ids, summary = cluster_per_length(df, xi=XI, min_samples=min_samples)

    in_range = df["nClicks"].between(min(LENGTH_RANGE), max(LENGTH_RANGE)).to_numpy()
    eligible = in_range
    n_eligible = int(eligible.sum())

    pred = majority_vote_mapping(cluster_ids, truth)
    noise_mask = (cluster_ids == -1) & eligible
    valid_mask = eligible & ~noise_mask

    n_correct_valid = int((pred[valid_mask] == truth[valid_mask]).sum())
    n_valid = int(valid_mask.sum())
    n_noise = int(noise_mask.sum())

    acc_excl_noise = n_correct_valid / max(n_valid, 1)
    acc_incl_noise_as_wrong = n_correct_valid / max(n_eligible, 1)
    ari = adjusted_rand_score(truth[valid_mask], cluster_ids[valid_mask])

    print(f"\n=== min_samples={min_samples}, xi={XI} ===")
    print(f"  in-range codas (3..10 clicks): {n_eligible}")
    print(f"  noise: {n_noise} ({n_noise/max(n_eligible,1):.1%})")
    print(f"  Sharma-18 accuracy (non-noise only): {acc_excl_noise:.1%} "
          f"({n_correct_valid}/{n_valid})")
    print(f"  Sharma-18 accuracy (noise counted as wrong): {acc_incl_noise_as_wrong:.1%}")
    print(f"  ARI (non-noise): {ari:.3f}")
    print("  per-length: n_total, n_clusters, n_noise")
    for n, n_total, n_clus, n_noise_n in summary:
        print(f"    {n} clicks: {n_total:>5d} codas | {n_clus:>3d} clusters | "
              f"{n_noise_n:>4d} noise")

    confusion = pd.crosstab(
        pd.Series(truth[valid_mask], name="sharma"),
        pd.Series(pred[valid_mask], name="pred"),
    )
    diag = np.diag(confusion.values).sum()
    print(f"  diagonal mass (matched per-class): {diag}/{n_valid} = {diag/max(n_valid,1):.1%}")
    return {
        "min_samples": min_samples,
        "n_eligible": n_eligible,
        "n_noise": n_noise,
        "n_valid": n_valid,
        "n_correct": n_correct_valid,
        "acc_excl_noise": acc_excl_noise,
        "acc_incl_noise": acc_incl_noise_as_wrong,
        "ari": ari,
        "summary": summary,
    }


def main() -> list[dict]:
    df = load_labelled()
    print(f"loaded {len(df)} labelled codas; "
          f"unique Sharma rhythms: {sorted(df['sharma_rhythm'].dropna().unique().tolist())}")
    in_range = df["nClicks"].between(min(LENGTH_RANGE), max(LENGTH_RANGE))
    print(f"in 3..10 click range: {int(in_range.sum())} of {len(df)}")
    results = []
    for ms in MIN_SAMPLES_SWEEP:
        results.append(report(df, min_samples=ms))
    print("\n=== summary ===")
    print(f"{'min_samples':>12}  {'acc_no_noise':>13}  {'acc_w_noise':>12}  {'noise%':>8}  {'ARI':>6}")
    for r in results:
        print(f"{r['min_samples']:>12d}  {r['acc_excl_noise']:>13.1%}  "
              f"{r['acc_incl_noise']:>12.1%}  "
              f"{r['n_noise']/max(r['n_eligible'],1):>8.1%}  {r['ari']:>6.3f}")
    return results


if __name__ == "__main__":
    main()
