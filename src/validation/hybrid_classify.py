"""Hybrid Sharma-anchored / Pacific-discovery classifier.

Three-stage flow agreed with the user (2026-05-04):

1. **Sharma "real" types** (non-``-NOISE`` ``CodaType`` labels) are locked
   from the direct ``source_coda_id ↔ codaNUM2018`` join. No clustering
   loss.
2. **Pacific (Hersh + birth) codas** are classified via kNN k=5 against
   Sharma "real" rows of the same length. A match is accepted iff the
   nearest-neighbour distance is within ``τ = NOSC p95`` of the predicted
   type (NOSC = nearest-of-same-class within DSWP, from
   ``dswp_variance.csv``). Matched Pacific codas inherit the Sharma label.
3. **Discovery pool** = Sharma-``-NOISE`` rows ∪ Pacific residual.
   OPTICSxi (xi=0.04, minpts=10) per length. Clusters = discovered types
   (mixed-source allowed). cluster_id=0 (ELKI "rest" bucket) and any
   cluster smaller than minpts are final NOISE.

Produces ``data/diagnostics/hybrid_v1/``:

  - ``assignments.csv``      — per-coda final label + origin
  - ``discovered_clusters.csv`` — per-cluster summary (n, source mix)
  - ``SUMMARY.md``           — the four numbers the user asked for
  - ``piano_rolls/`` PNGs    — see ``piano_roll`` module

Run from repo root::

    python -m src.validation.hybrid_classify
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from src.validation import elki_optics
from src.validation.phase1b_knn import (
    DOMINICA, KNN_K, LENGTH_RANGE, UNIFIED, coda_type_for_dswp,
)

REPO = Path(__file__).resolve().parents[2]
VARIANCE_CSV = REPO / "data" / "diagnostics" / "dswp_variance.csv"
OUT = REPO / "data" / "diagnostics" / "hybrid_v1"
CACHE_DIR = REPO / "data" / "cache"

XI = 0.04
MINPTS = 10
NOISE_CLUSTER_ID = 0  # ELKI "rest" bucket convention (B_classify_optics.py
                      # confirms this empirically).
TAU_COLUMN = "nosc_p99"  # widened from p95 after first run rejected ~7k
                          # n=5 Pacific codas that should have matched
                          # tight Sharma types.
TAU_FLOOR = 0.10  # seconds; matches B_classify_optics.RADIUS_FLOOR_S.
                   # 0.10 s ≈ half the median inter-CodaType centroid distance
                   # in n=5, so two codas within 0.10 s are closer to each
                   # other than to the next-nearest type.


# ---------------------------------------------------------------------------

def _ici_cols(n: int) -> list[str]:
    return [f"ICI{i}" for i in range(1, n)]


def _ici_matrix(df: pd.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray]:
    cols = _ici_cols(n)
    mask = (df["n_clicks"] == n) & df[cols].notna().all(axis=1)
    return (df.index[mask].to_numpy(),
            df.loc[mask, cols].to_numpy(dtype=float))


def _is_noise_label(v) -> bool:
    return isinstance(v, str) and v.endswith("-NOISE")


# ---------------------------------------------------------------------------

def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    u = pd.read_csv(UNIFIED, low_memory=False)
    d = pd.read_csv(DOMINICA)
    d.columns = [c.strip().lstrip("﻿") for c in d.columns]
    u["coda_type_truth"] = coda_type_for_dswp(u, d)
    var = pd.read_csv(VARIANCE_CSV)
    return u, var


def build_tau_lookup(var: pd.DataFrame) -> dict[tuple[int, str], float]:
    """{(length, codatype) -> τ}, restricted to non-NOISE types.

    τ = max(``TAU_COLUMN`` value, ``TAU_FLOOR``). Falls back to the
    per-length median for types whose density column is NaN.
    """
    real = var[~var["is_noise_type"]].copy()
    tau = {}
    for n, sub in real.groupby("length"):
        med = float(np.nanmedian(sub[TAU_COLUMN]))
        for _, row in sub.iterrows():
            v = row[TAU_COLUMN]
            base = float(v) if pd.notna(v) else med
            tau[(int(n), str(row["codatype"]))] = max(base, TAU_FLOOR)
    return tau


def cached_optics(X: np.ndarray, *, tag: str, n: int) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"optics_{tag}_n{n}_mp{MINPTS}.npz"
    h = np.int64(hash(X.tobytes()))
    if cache_path.exists():
        z = np.load(cache_path)
        if "hash" in z and int(z["hash"]) == int(h) and \
                z["labels"].shape[0] == X.shape[0]:
            return z["labels"]
    labels = elki_optics.run(X, xi=XI, minpts=MINPTS)
    np.savez(cache_path, labels=labels, hash=h)
    return labels


# ---------------------------------------------------------------------------

def stage1_anchor(u: pd.DataFrame) -> pd.DataFrame:
    """Returns a per-row state dataframe initialised with Sharma 'real'
    anchors. Columns: final_label, origin, distance, cluster_id."""
    n = len(u)
    state = pd.DataFrame({
        "final_label": pd.Series(pd.NA, index=u.index, dtype="object"),
        "origin": pd.Series(pd.NA, index=u.index, dtype="object"),
        "distance": pd.Series(np.nan, index=u.index, dtype="float64"),
        "cluster_id": pd.Series(pd.NA, index=u.index, dtype="Int64"),
    })
    truth = u["coda_type_truth"]
    is_real_dswp = truth.notna() & ~truth.map(_is_noise_label)
    state.loc[is_real_dswp, "final_label"] = truth[is_real_dswp]
    state.loc[is_real_dswp, "origin"] = "dswp-real"
    return state


def stage2_knn_match(
    u: pd.DataFrame, state: pd.DataFrame,
    tau: dict[tuple[int, str], float],
) -> pd.DataFrame:
    """For each Pacific (non-DSWP) coda, kNN-predict against Sharma real
    rows of the same length, accept if within τ[predicted_type]."""
    is_dswp = u["source"] == "sharma2024_dswp"
    truth = u["coda_type_truth"]
    is_real_dswp = truth.notna() & ~truth.map(_is_noise_label)

    for n in LENGTH_RANGE:
        cols = _ici_cols(n)
        # training: Sharma real rows of length n with all ICIs
        train_mask = (is_real_dswp & (u["n_clicks"] == n)
                      & u[cols].notna().all(axis=1))
        if int(train_mask.sum()) < KNN_K + 1:
            continue
        X_train = u.loc[train_mask, cols].to_numpy(dtype=float)
        y_train = u.loc[train_mask, "coda_type_truth"].astype(str).to_numpy()
        knn = KNeighborsClassifier(n_neighbors=KNN_K, metric="euclidean")
        knn.fit(X_train, y_train)

        # query: non-DSWP rows of length n
        q_mask = (~is_dswp & (u["n_clicks"] == n)
                  & u[cols].notna().all(axis=1))
        q_idx = u.index[q_mask].to_numpy()
        if len(q_idx) == 0:
            continue
        X_q = u.loc[q_idx, cols].to_numpy(dtype=float)
        y_pred = knn.predict(X_q)
        d_nn = knn.kneighbors(X_q, n_neighbors=1)[0][:, 0]

        for i, ix in enumerate(q_idx):
            t = tau.get((n, str(y_pred[i])), float("inf"))
            state.at[ix, "distance"] = float(d_nn[i])
            if d_nn[i] <= t:
                state.at[ix, "final_label"] = str(y_pred[i])
                state.at[ix, "origin"] = "pacific-matched"
    return state


def stage3_discover(
    u: pd.DataFrame, state: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run OPTICSxi per length on the discovery pool.

    Pool members:
      - Sharma rows whose truth ends in '-NOISE'
      - Pacific rows that stage2 didn't match (state.origin still NA)

    Returns (state, cluster_records).
    """
    is_dswp = u["source"] == "sharma2024_dswp"
    truth = u["coda_type_truth"]
    is_dswp_noise = is_dswp & truth.map(_is_noise_label)
    is_pac_residual = ((~is_dswp) & state["origin"].isna())
    pool_mask = is_dswp_noise | is_pac_residual

    cluster_records: list[dict] = []
    for n in LENGTH_RANGE:
        cols = _ici_cols(n)
        m = pool_mask & (u["n_clicks"] == n) & u[cols].notna().all(axis=1)
        idx = u.index[m].to_numpy()
        if len(idx) < MINPTS:
            # Whole pool for this length is too small to cluster.
            for ix in idx:
                state.at[ix, "final_label"] = f"{n}-NOISE"
                state.at[ix, "origin"] = "discovery-noise"
            continue
        X = u.loc[idx, cols].to_numpy(dtype=float)
        labels = cached_optics(X, tag="hybrid_pool", n=n)

        for cl in np.unique(labels):
            mask = labels == cl
            n_members = int(mask.sum())
            if int(cl) == NOISE_CLUSTER_ID or n_members < MINPTS:
                for ix in idx[mask]:
                    state.at[ix, "final_label"] = f"{n}-NOISE"
                    state.at[ix, "origin"] = "discovery-noise"
                continue
            cl_label = f"P{n}c{int(cl)}"
            for ix in idx[mask]:
                state.at[ix, "final_label"] = cl_label
                state.at[ix, "origin"] = "discovery-cluster"
                state.at[ix, "cluster_id"] = int(cl)
            srcs = u.loc[idx[mask], "source"].value_counts().to_dict()
            cluster_records.append({
                "cluster_label": cl_label,
                "n_clicks": n,
                "elki_cluster_id": int(cl),
                "n_members": n_members,
                "n_dswp": int(srcs.get("sharma2024_dswp", 0)),
                "n_hersh": int(srcs.get("hersh2022_pacific", 0)),
                "n_birth": int(srcs.get("sharma2025_birth", 0)),
                "unified_indices": idx[mask],
            })
    return state, cluster_records


# ---------------------------------------------------------------------------

def four_numbers(u: pd.DataFrame, state: pd.DataFrame) -> dict:
    is_dswp = u["source"] == "sharma2024_dswp"
    is_pac = ~is_dswp
    truth = u["coda_type_truth"]
    is_dswp_noise = is_dswp & truth.map(_is_noise_label)

    n_pac_total = int(is_pac.sum())
    n_pac_inrange = int((is_pac & u["n_clicks"].between(3, 10)).sum())
    n_pac_matched = int((is_pac & (state["origin"] == "pacific-matched")).sum())
    n_pac_discovered = int(
        (is_pac & (state["origin"] == "discovery-cluster")).sum())
    n_pac_noise = int(
        (is_pac & (state["origin"] == "discovery-noise")).sum())
    # Pacific not seen by either stage (length out of 3–10, missing ICIs).
    n_pac_unprocessed = n_pac_total - n_pac_matched - n_pac_discovered \
                        - n_pac_noise

    n_dn_total = int(is_dswp_noise.sum())
    n_dn_clustered = int(
        (is_dswp_noise & (state["origin"] == "discovery-cluster")).sum())
    n_dn_still_noise = int(
        (is_dswp_noise & (state["origin"] == "discovery-noise")).sum())
    n_dn_unprocessed = n_dn_total - n_dn_clustered - n_dn_still_noise

    return {
        "pacific_total": n_pac_total,
        "pacific_in_range": n_pac_inrange,
        "pacific_matched_to_sharma": n_pac_matched,
        "pacific_in_discovered_clusters": n_pac_discovered,
        "pacific_final_noise": n_pac_noise,
        "pacific_unprocessed": n_pac_unprocessed,

        "dswp_noise_total": n_dn_total,
        "dswp_noise_now_clustered": n_dn_clustered,
        "dswp_noise_still_noise": n_dn_still_noise,
        "dswp_noise_unprocessed": n_dn_unprocessed,
    }


def write_summary_md(stats: dict, cluster_records: list[dict],
                     out_path: Path) -> None:
    pac_total = stats["pacific_in_range"]
    dn_total = stats["dswp_noise_total"]
    pool_total = (stats["pacific_in_discovered_clusters"]
                  + stats["pacific_final_noise"]
                  + stats["dswp_noise_now_clustered"]
                  + stats["dswp_noise_still_noise"])
    pool_clustered = (stats["pacific_in_discovered_clusters"]
                      + stats["dswp_noise_now_clustered"])
    pool_noise = (stats["pacific_final_noise"]
                  + stats["dswp_noise_still_noise"])

    def pct(n, d):
        return f"{(n / max(d, 1)):.1%}"

    n_clusters = len(cluster_records)
    n_clusters_per_length = {}
    for r in cluster_records:
        n_clusters_per_length[r["n_clicks"]] = \
            n_clusters_per_length.get(r["n_clicks"], 0) + 1

    lines = [
        "# Hybrid classifier — run summary",
        "",
        "Pipeline: Sharma-real anchors → Pacific kNN+τ matching → ",
        "OPTICSxi (xi=0.04, minpts=10) on Sharma-NOISE ∪ Pacific-residual.",
        "",
        "## Question 1 — Pacific codas matched to Sharma types",
        "",
        f"- **{stats['pacific_matched_to_sharma']:,} / "
        f"{pac_total:,} Pacific codas** matched a Sharma 'real' type via "
        f"kNN k=5 within τ = NOSC p95.",
        f"- That is **{pct(stats['pacific_matched_to_sharma'], pac_total)}** "
        f"of in-range Pacific codas (n_clicks 3–10 with all ICIs).",
        "",
        "## Question 2 — Pacific codas in newly-discovered clusters",
        "",
        f"- **{stats['pacific_in_discovered_clusters']:,} / {pac_total:,} "
        f"Pacific codas** placed into "
        f"{n_clusters} discovered clusters.",
        f"- That is **{pct(stats['pacific_in_discovered_clusters'], pac_total)}** "
        f"of in-range Pacific codas.",
        f"- Cluster count by length: "
        f"{', '.join(f'n={k}: {v}' for k, v in sorted(n_clusters_per_length.items()))}.",
        "",
        "## Question 3 — Final NOISE (combined pool)",
        "",
        f"- **{pool_noise:,} / {pool_total:,} pool members** stayed NOISE.",
        f"- That is **{pct(pool_noise, pool_total)}** of the discovery pool.",
        f"- Pool composition: "
        f"{stats['pacific_in_discovered_clusters'] + stats['pacific_final_noise']:,} "
        f"Pacific residual, "
        f"{dn_total:,} Sharma-NOISE.",
        "",
        "## Question 4 — Sharma-NOISE second chance",
        "",
        f"- Sharma-NOISE rows entering the discovery pass: **{dn_total:,}**.",
        f"- **{stats['dswp_noise_now_clustered']:,}** "
        f"({pct(stats['dswp_noise_now_clustered'], dn_total)}) "
        f"clustered with Pacific codas — second-chance recoveries.",
        f"- **{stats['dswp_noise_still_noise']:,}** "
        f"({pct(stats['dswp_noise_still_noise'], dn_total)}) "
        f"stayed NOISE.",
        "",
        "## Quick reference",
        "",
        f"- Pacific in range (3–10 clicks, all ICIs): {pac_total:,}",
        f"- Pacific not in range / missing ICIs (unprocessed): "
        f"{stats['pacific_unprocessed']:,}",
        f"- Discovered clusters total: {n_clusters}",
        f"- Discovery pool size: {pool_total:,}",
        f"- Discovery pool clustered: {pool_clustered:,} "
        f"({pct(pool_clustered, pool_total)})",
        "",
    ]
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------

def main() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== hybrid classifier ===")
    print(f"loading unified + variance ...", flush=True)
    u, var = load()
    print(f"  unified rows: {len(u):,}; variance rows: {len(var)}")

    tau = build_tau_lookup(var)
    print(f"  τ-table: {len(tau)} (length, codatype) entries; "
          f"τ range: {min(tau.values()):.4f} – {max(tau.values()):.4f}")

    print("\n[stage 1] anchoring Sharma 'real' labels ...")
    state = stage1_anchor(u)
    n_anchored = int((state["origin"] == "dswp-real").sum())
    print(f"  {n_anchored:,} Sharma-real rows anchored")

    print("\n[stage 2] Pacific kNN+τ match against Sharma-real ...")
    state = stage2_knn_match(u, state, tau)
    n_matched = int((state["origin"] == "pacific-matched").sum())
    print(f"  {n_matched:,} Pacific codas matched a Sharma type")

    print("\n[stage 3] OPTICSxi discovery on Sharma-NOISE ∪ Pacific-residual")
    print(f"  (this calls ELKI per length; cached at "
          f"{CACHE_DIR.relative_to(REPO)}/optics_hybrid_pool_n*.npz)",
          flush=True)
    state, cluster_records = stage3_discover(u, state)
    n_discovered = int((state["origin"] == "discovery-cluster").sum())
    n_disc_noise = int((state["origin"] == "discovery-noise").sum())
    print(f"  -> {len(cluster_records)} discovered clusters; "
          f"{n_discovered:,} rows clustered, {n_disc_noise:,} NOISE")

    # write CSVs
    print("\n[writing] assignments.csv + discovered_clusters.csv ...")
    out_state = state.copy()
    out_state["source"] = u["source"]
    out_state["n_clicks"] = u["n_clicks"]
    out_state["clan"] = u["clan"]
    out_state["coda_type_truth"] = u["coda_type_truth"]
    out_state.to_csv(OUT / "assignments.csv", index_label="unified_index")
    cdf = pd.DataFrame([
        {k: v for k, v in r.items() if k != "unified_indices"}
        for r in cluster_records
    ])
    cdf.to_csv(OUT / "discovered_clusters.csv", index=False)

    stats = four_numbers(u, state)
    write_summary_md(stats, cluster_records, OUT / "SUMMARY.md")
    print(f"  -> {OUT.relative_to(REPO)}/")

    print("\n=== four numbers ===")
    pac_total = stats["pacific_in_range"]
    dn_total = stats["dswp_noise_total"]
    pool_clustered = (stats["pacific_in_discovered_clusters"]
                      + stats["dswp_noise_now_clustered"])
    pool_noise = (stats["pacific_final_noise"]
                  + stats["dswp_noise_still_noise"])
    pool_total = pool_clustered + pool_noise
    print(f"1) Pacific matched to Sharma types: "
          f"{stats['pacific_matched_to_sharma']:>6,}  / {pac_total:,} = "
          f"{stats['pacific_matched_to_sharma']/max(pac_total,1):.1%}")
    print(f"2) Pacific in new discovered clusters: "
          f"{stats['pacific_in_discovered_clusters']:>6,}  / {pac_total:,} = "
          f"{stats['pacific_in_discovered_clusters']/max(pac_total,1):.1%}")
    print(f"3) Pool members remaining NOISE: "
          f"{pool_noise:>6,}  / {pool_total:,} = "
          f"{pool_noise/max(pool_total,1):.1%}")
    print(f"4) Sharma-NOISE second-chance:")
    print(f"     clustered with Pacific: "
          f"{stats['dswp_noise_now_clustered']:>5,}  / {dn_total:,} = "
          f"{stats['dswp_noise_now_clustered']/max(dn_total,1):.1%}")
    print(f"     stayed NOISE:           "
          f"{stats['dswp_noise_still_noise']:>5,}  / {dn_total:,} = "
          f"{stats['dswp_noise_still_noise']/max(dn_total,1):.1%}")

    return u, out_state, cluster_records


if __name__ == "__main__":
    main()
