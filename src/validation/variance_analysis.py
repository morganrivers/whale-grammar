"""Variance analysis used to derive per-type τ thresholds for the
hybrid classifier (see ``docs/obsidian/classifier/pacific-extension.md``
§τ).

Computes three CSVs:

  within-DSWP variance per CodaType    -> dswp_variance.csv
  cross-corpus distance distribution    -> pacific_distances.csv
  per-clan coverage + within-clan var.  -> clan_summary.csv

Plus visual aids (z-scaled PCA scatter, per-clan stacked bar). The CSV
metrics use raw (unscaled) Euclidean distance — the same units as the
kNN+τ classifier in ``B_classify_optics.match_other_codas``.

Run from repo root::

    python -m src.validation.variance_analysis
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from .data_quality_diagnostics import _pca_2d
from .phase1b_knn import (
    DOMINICA, KNN_K, LENGTH_RANGE, UNIFIED, coda_type_for_dswp,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "diagnostics"
PCA_OUT = OUT / "variance_pca"

PAIRWISE_SAMPLE_CAP = 1500
TOP_K_TYPES_PER_CLAN = 5
TOP_N_TYPES_BAR = 15
RNG = np.random.default_rng(0)


def _load() -> pd.DataFrame:
    u = pd.read_csv(UNIFIED, low_memory=False)
    d = pd.read_csv(DOMINICA)
    d.columns = [c.strip().lstrip("﻿") for c in d.columns]
    u["coda_type_truth"] = coda_type_for_dswp(u, d)
    return u


def _is_noise_type(t: str) -> bool:
    return isinstance(t, str) and t.endswith("-NOISE")


def _clan_group(source: pd.Series, clan: pd.Series) -> np.ndarray:
    """Stratification key per §5.2 of the handoff: split Sharma-birth from
    Hersh's EC1 (they're both labelled `clan == "EC1"` but represent
    different things — neonatal coda development vs adult repertoire)."""
    cg = clan.fillna("(NaN)").astype(str)
    cg = cg.where(source != "sharma2025_birth", "EC1-birth")
    return cg.to_numpy()


# §4.1 -----------------------------------------------------------------

def dswp_variance(
    u: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    """Per-(length, codatype) centroid + variance metrics on DSWP truth.

    Also returns ``centroids_by_length`` for reuse in §4.2.
    """
    rows = []
    centroids_by_length: dict[int, dict[str, np.ndarray]] = {}
    is_dswp = u["source"] == "sharma2024_dswp"
    for n in LENGTH_RANGE:
        cols = [f"ICI{i}" for i in range(1, n)]
        mask = (is_dswp & (u["n_clicks"] == n)
                & u["coda_type_truth"].notna()
                & u[cols].notna().all(axis=1))
        sub = u.loc[mask, cols + ["coda_type_truth"]]
        if len(sub) == 0:
            continue
        X = sub[cols].to_numpy(dtype=float)
        y = sub["coda_type_truth"].astype(str).to_numpy()
        cen_n: dict[str, np.ndarray] = {}
        for t in np.unique(y):
            idx = np.where(y == t)[0]
            Xt = X[idx]
            cen = Xt.mean(axis=0)
            cen_n[t] = cen
            n_mem = len(idx)
            d_to_cen = np.linalg.norm(Xt - cen, axis=1)
            row = {
                "codatype": t,
                "is_noise_type": _is_noise_type(t),
                "length": n,
                "n_members": n_mem,
                "centroid": json.dumps([round(float(x), 6) for x in cen]),
                "mean_dist_to_centroid": float(d_to_cen.mean()),
                "r95": float(np.percentile(d_to_cen, 95)),
            }
            if n_mem >= 2:
                D = np.linalg.norm(Xt[:, None, :] - Xt[None, :, :], axis=2)
                np.fill_diagonal(D, np.inf)
                nosc = D.min(axis=1)
                row.update({
                    "nosc_p50": float(np.percentile(nosc, 50)),
                    "nosc_p90": float(np.percentile(nosc, 90)),
                    "nosc_p95": float(np.percentile(nosc, 95)),
                    "nosc_p99": float(np.percentile(nosc, 99)),
                })
            else:
                row.update({"nosc_p50": float("nan"),
                            "nosc_p90": float("nan"),
                            "nosc_p95": float("nan"),
                            "nosc_p99": float("nan")})
            rows.append(row)
        centroids_by_length[n] = cen_n
    df = pd.DataFrame(rows).sort_values(["length", "codatype"])
    return df, centroids_by_length


# §4.2 -----------------------------------------------------------------

def pacific_distances(
    u: pd.DataFrame,
    centroids_by_length: dict[int, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Per-Pacific-coda kNN prediction + centroid distances.

    `runner_up_dist` is the distance to the closest centroid that is NOT
    the kNN-predicted type — so when kNN's vote disagrees with the
    1-NN-centroid call, `margin` will go negative.
    """
    is_dswp = u["source"] == "sharma2024_dswp"
    rows = []
    for n in LENGTH_RANGE:
        cols = [f"ICI{i}" for i in range(1, n)]
        train_mask = (is_dswp & (u["n_clicks"] == n)
                      & u["coda_type_truth"].notna()
                      & u[cols].notna().all(axis=1))
        train_idx = u.index[train_mask].to_numpy()
        if len(train_idx) < KNN_K + 1:
            continue
        X_train = u.loc[train_idx, cols].to_numpy(dtype=float)
        y_train = u.loc[train_idx, "coda_type_truth"].astype(str).to_numpy()
        knn = KNeighborsClassifier(n_neighbors=KNN_K, metric="euclidean")
        knn.fit(X_train, y_train)

        pac_mask = (~is_dswp & (u["n_clicks"] == n)
                    & u[cols].notna().all(axis=1))
        pac_idx = u.index[pac_mask].to_numpy()
        if len(pac_idx) == 0:
            continue
        X_pac = u.loc[pac_idx, cols].to_numpy(dtype=float)
        y_pred = knn.predict(X_pac)
        d_nn = knn.kneighbors(X_pac, n_neighbors=1)[0][:, 0]

        cen_n = centroids_by_length.get(n, {})
        if not cen_n:
            continue
        types_n = list(cen_n.keys())
        type_to_col = {t: i for i, t in enumerate(types_n)}
        cen_mat = np.stack([cen_n[t] for t in types_n])
        D = np.linalg.norm(X_pac[:, None, :] - cen_mat[None, :, :], axis=2)
        pred_cols = np.array([type_to_col[t] for t in y_pred])
        pred_dist = D[np.arange(len(X_pac)), pred_cols]
        D_for_ru = D.copy()
        D_for_ru[np.arange(len(X_pac)), pred_cols] = np.inf
        runner_up = D_for_ru.min(axis=1)

        src = u.loc[pac_idx, "source"].to_numpy()
        clan = u.loc[pac_idx, "clan"].to_numpy()
        for i, ux_idx in enumerate(pac_idx):
            rows.append({
                "unified_index": int(ux_idx),
                "length": n,
                "source": src[i],
                "clan": clan[i] if pd.notna(clan[i]) else None,
                "dswp_nn_dist": float(d_nn[i]),
                "pred_codatype": str(y_pred[i]),
                "pred_centroid_dist": float(pred_dist[i]),
                "runner_up_dist": float(runner_up[i]),
                "margin": float(runner_up[i] - pred_dist[i]),
            })
    return pd.DataFrame(rows)


# §4.3 -----------------------------------------------------------------

def clan_summary(u: pd.DataFrame, pac: pd.DataFrame) -> pd.DataFrame:
    pac = pac.assign(clan_group=_clan_group(pac["source"], pac["clan"]))
    is_dswp = u["source"] == "sharma2024_dswp"

    # Pre-compute clan_group on the unified frame so we can stratify the
    # within-clan variance calculation against the same key.
    u_clan_group = _clan_group(u["source"], u["clan"])

    rows = []
    for c, sub in pac.groupby("clan_group", dropna=False):
        n_codas = len(sub)
        type_counts = sub["pred_codatype"].value_counts()
        top_types = list(type_counts.head(TOP_K_TYPES_PER_CLAN).index)
        coverage = float(sub["pred_codatype"].isin(top_types).mean())

        # Mean pairwise within-clan, per length, sample-capped, then
        # weighted by membership across lengths.
        wcv_num = 0.0
        wcv_den = 0
        for n in LENGTH_RANGE:
            cols = [f"ICI{i}" for i in range(1, n)]
            mask = (~is_dswp
                    & (u["n_clicks"] == n)
                    & u[cols].notna().all(axis=1)
                    & (u_clan_group == c))
            X = u.loc[mask, cols].to_numpy(dtype=float)
            n_here = len(X)
            if n_here < 2:
                continue
            if n_here > PAIRWISE_SAMPLE_CAP:
                samp = RNG.choice(n_here, PAIRWISE_SAMPLE_CAP, replace=False)
                X = X[samp]
            D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
            iu = np.triu_indices(len(X), k=1)
            wcv_num += float(D[iu].mean()) * n_here
            wcv_den += n_here
        within_var = wcv_num / wcv_den if wcv_den else float("nan")

        nn = sub["dswp_nn_dist"].to_numpy()
        rows.append({
            "clan": c,
            "n_codas": n_codas,
            "top_5_codatypes": ";".join(top_types),
            "top_5_coverage": coverage,
            "within_clan_variance": within_var,
            "nn_dist_p50": float(np.percentile(nn, 50)),
            "nn_dist_p90": float(np.percentile(nn, 90)),
        })
    return pd.DataFrame(rows).sort_values("n_codas", ascending=False)


# plots ---------------------------------------------------------------

def _plot_pca_per_length(u: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [pca] matplotlib not available, skipping")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    is_dswp = u["source"] == "sharma2024_dswp"
    for n in LENGTH_RANGE:
        cols = [f"ICI{i}" for i in range(1, n)]
        mask = (is_dswp & (u["n_clicks"] == n)
                & u["coda_type_truth"].notna()
                & u[cols].notna().all(axis=1))
        sub = u.loc[mask].copy()
        if len(sub) < 5:
            continue
        X = sub[cols].to_numpy(dtype=float)
        Y = _pca_2d(X)
        sub["pc1"] = Y[:, 0]
        sub["pc2"] = Y[:, 1]
        types = sorted(sub["coda_type_truth"].astype(str).unique())
        cmap = plt.get_cmap("tab20")
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for i, t in enumerate(types):
            m = sub["coda_type_truth"].astype(str) == t
            ax.scatter(sub.loc[m, "pc1"], sub.loc[m, "pc2"],
                       s=4, alpha=0.4, color=cmap(i % 20),
                       label=f"{t} ({m.sum()})")
            cx = sub.loc[m, "pc1"].mean()
            cy = sub.loc[m, "pc2"].mean()
            ax.text(cx, cy, t, fontsize=7, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1",
                              fc="white", ec="none", alpha=0.7))
        ax.set_title(f"DSWP CodaTypes, n_clicks={n}, n={len(sub)}")
        ax.set_xlabel("PC1 (z-scaled)")
        ax.set_ylabel("PC2 (z-scaled)")
        ax.legend(fontsize=6, ncol=2, loc="best", markerscale=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"dswp_n{n}.png", dpi=120)
        plt.close(fig)


def _plot_clan_bars(pac: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [bars] matplotlib not available, skipping")
        return
    pac = pac.assign(clan_group=_clan_group(pac["source"], pac["clan"]))
    clans = pac["clan_group"].value_counts().index.tolist()
    global_top = (pac["pred_codatype"].value_counts()
                  .head(TOP_N_TYPES_BAR).index.tolist())
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(11, 6))
    bottoms = np.zeros(len(clans))
    for i, t in enumerate(global_top):
        vals = []
        for c in clans:
            csub = pac[pac["clan_group"] == c]
            vals.append((csub["pred_codatype"] == t).sum() / max(len(csub), 1))
        ax.bar(clans, vals, bottom=bottoms, label=t, color=cmap(i % 20))
        bottoms += np.array(vals)
    ax.bar(clans, 1.0 - bottoms, bottom=bottoms, label="other", color="grey")
    ax.set_ylabel("fraction of clan codas")
    ax.set_ylim(0, 1)
    ax.set_title(f"Predicted CodaType composition by clan "
                 f"(top {TOP_N_TYPES_BAR})")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# main ----------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== §4 variance analysis ===")
    print("loading codas_unified.csv + dswp_dominica_codas.csv ...",
          flush=True)
    u = _load()
    n_truth = int(u["coda_type_truth"].notna().sum())
    print(f"  unified rows: {len(u):,}; DSWP CodaType truth: {n_truth:,}")

    print("\n[§4.1] within-DSWP variance per CodaType ...")
    dswp_var, centroids_by_length = dswp_variance(u)
    dswp_var.to_csv(OUT / "dswp_variance.csv", index=False)
    n_types = dswp_var["codatype"].nunique()
    n_noise_types = int(
        dswp_var.loc[dswp_var["is_noise_type"], "codatype"].nunique())
    print(f"  {len(dswp_var)} (length, codatype) rows; "
          f"{n_types} unique types ({n_noise_types} NOISE).")
    nosc_p95 = dswp_var["nosc_p95"].dropna()
    r95 = dswp_var["r95"].dropna()
    print(f"  r95     across types: median={r95.median():.4f}  "
          f"p90={r95.quantile(0.9):.4f}")
    print(f"  nosc_p95 across types: median={nosc_p95.median():.4f}  "
          f"p90={nosc_p95.quantile(0.9):.4f}")
    print(f"  -> {OUT.relative_to(REPO)}/dswp_variance.csv")

    print("\n[§4.2] cross-corpus distances ...")
    pac = pacific_distances(u, centroids_by_length)
    pac.to_csv(OUT / "pacific_distances.csv", index=False)
    print(f"  {len(pac):,} Pacific rows classified")
    print(f"  dswp_nn_dist percentiles (Pacific):")
    for q in (0.5, 0.75, 0.9, 0.95, 0.99):
        print(f"    p{int(q*100):>2}: "
              f"{pac['dswp_nn_dist'].quantile(q):.4f}")
    print(f"  pred_centroid_dist percentiles:")
    for q in (0.5, 0.9, 0.99):
        print(f"    p{int(q*100):>2}: "
              f"{pac['pred_centroid_dist'].quantile(q):.4f}")
    print(f"  margin percentiles:")
    for q in (0.05, 0.5, 0.95):
        print(f"    p{int(q*100):>2}: {pac['margin'].quantile(q):.4f}")
    print(f"  -> {OUT.relative_to(REPO)}/pacific_distances.csv")

    print("\n[§4.3] per-clan summary ...")
    clan_df = clan_summary(u, pac)
    clan_df.to_csv(OUT / "clan_summary.csv", index=False)
    print(f"  {'clan':<10} {'n':>6} {'top5cov':>9} {'within_var':>11} "
          f"{'nn_p50':>8} {'nn_p90':>8}")
    for _, r in clan_df.iterrows():
        print(f"  {str(r['clan']):<10} {int(r['n_codas']):>6} "
              f"{r['top_5_coverage']:>8.1%}  {r['within_clan_variance']:>10.4f} "
              f"{r['nn_dist_p50']:>8.4f} {r['nn_dist_p90']:>8.4f}")
    print(f"  -> {OUT.relative_to(REPO)}/clan_summary.csv")

    print("\n[plots] per-length DSWP PCA scatter ...")
    _plot_pca_per_length(u, PCA_OUT)
    print(f"  -> {PCA_OUT.relative_to(REPO)}/")

    print("[plots] per-clan stacked bar ...")
    _plot_clan_bars(pac, OUT / "clan_bars.png")
    print(f"  -> {(OUT / 'clan_bars.png').relative_to(REPO)}")


if __name__ == "__main__":
    main()
