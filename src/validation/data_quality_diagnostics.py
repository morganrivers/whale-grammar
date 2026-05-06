"""Data-quality diagnostics for the hybrid Pacific classifier.

Question: is some non-trivial fraction of the 28% NOISE rate a property of
the Hersh dataset (echolocation, low-SNR detections, cross-species clicks,
unification residuals) rather than a classifier shortfall? See
``docs/obsidian/classifier/pacific-extension.md`` §"NOISE rate caveat" —
run these *before* tuning the classifier; they may invalidate later
tuning work.

Five runnable diagnostics, all reading
``data/classified/codas_classified.csv``:

1. **NOISE rate by recording_id.** Uniform → algorithm-bound; concentrated
   in a few recordings → quality outliers to filter.
2. **NOISE rate by clan.** Tells you whether the residual lives in a
   particular clan's repertoire.
3. **classifier_distance histogram for NOISE rows.** Bimodal (near-miss +
   long tail) means kNN+τ can recover the near-miss; smooth long tail
   means it's all genuine.
4. **PCA of NOISE codas (per length).** Clusters → undiscovered Pacific
   types; uniform cloud → literal noise.
5. **Echolocation sanity check.** n_clicks ≥ 8 with monotonically-spaced
   ICIs ≥ 0.4 s suggests echolocation, not coda.

Outputs CSVs + per-length PCA scatter PNGs under ``data/diagnostics/``.

Run from repo root::

    python -m src.validation.data_quality_diagnostics
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CLASSIFIED = REPO / "data" / "classified" / "codas_classified.csv"
OUT = REPO / "data" / "diagnostics"

LENGTH_RANGE = range(3, 11)
ECHO_MIN_N = 8
ECHO_ICI_MIN_S = 0.4


def _load_non_dswp() -> pd.DataFrame:
    df = pd.read_csv(CLASSIFIED, low_memory=False)
    df = df[df["source"] != "sharma2024_dswp"].copy()
    df["is_noise"] = df["classifier_origin"].fillna("") == "discovery-noise"
    return df


def diag1_by_recording(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("recording_id", dropna=False)
    out = pd.DataFrame({
        "n_codas": g.size(),
        "n_noise": g["is_noise"].sum(),
        "noise_rate": g["is_noise"].mean(),
        "n_clicks_median": g["n_clicks"].median(),
        "source": g["source"].first(),
    }).sort_values("noise_rate", ascending=False)
    return out


def diag2_by_clan(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("clan", dropna=False)
    out = pd.DataFrame({
        "n_codas": g.size(),
        "n_noise": g["is_noise"].sum(),
        "noise_rate": g["is_noise"].mean(),
    }).sort_values("noise_rate", ascending=False)
    return out


def diag3_distance_histogram(df: pd.DataFrame) -> pd.DataFrame:
    """Histogram (and percentiles) of classifier_distance for NOISE rows.

    Per the plan, a bimodal distribution = recoverable near-misses + true
    long tail; a smooth long tail = no easy wins.
    """
    noise = df.loc[df["is_noise"] & df["classifier_distance"].notna(),
                   "classifier_distance"].to_numpy(dtype=float)
    ec = df.loc[(df["classifier_origin"] == "pacific-matched")
                & df["classifier_distance"].notna(),
                "classifier_distance"].to_numpy(dtype=float)

    qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    rows = []
    for label, arr in [("discovery-noise", noise),
                       ("pacific-matched", ec)]:
        rows.append({
            "subset": label,
            "n": len(arr),
            "mean": float(arr.mean()) if len(arr) else float("nan"),
            **{f"p{int(q*100)}": float(np.quantile(arr, q)) if len(arr)
               else float("nan") for q in qs},
        })
    return pd.DataFrame(rows)


def _pca_2d(X: np.ndarray) -> np.ndarray:
    """Plain numpy PCA → top-2 components. Centred, unit-scaled per dim."""
    Xc = X - X.mean(axis=0, keepdims=True)
    std = Xc.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    Xn = Xc / std
    U, S, Vt = np.linalg.svd(Xn, full_matrices=False)
    return Xn @ Vt[:2].T


def diag4_pca(df: pd.DataFrame) -> pd.DataFrame:
    """Per-length PCA of NOISE codas. Returns a tidy frame; the plot
    function consumes it. Also reports how 'cluster-like' the cloud looks
    via a simple metric: ratio of explained variance in PC1 vs PC2.
    """
    rows = []
    for n in LENGTH_RANGE:
        cols = [f"ICI{i}" for i in range(1, n)]
        sub = df[df["is_noise"] & (df["n_clicks"] == n)
                 & df[cols].notna().all(axis=1)].copy()
        if len(sub) < 5:
            continue
        X = sub[cols].to_numpy(dtype=float)
        Y = _pca_2d(X)
        sub = sub.assign(pc1=Y[:, 0], pc2=Y[:, 1], length=n)
        rows.append(sub[["length", "source", "clan", "recording_id",
                          "classifier_distance", "pc1", "pc2"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _plot_pca(pca_df: pd.DataFrame, out_dir: Path) -> None:
    """Render one PNG per length; matplotlib only — no seaborn."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [diag4] matplotlib not available, skipping PCA plots")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, sub in pca_df.groupby("length"):
        fig, ax = plt.subplots(figsize=(6, 5))
        clans = sub["clan"].fillna("(none)").astype(str)
        for c in sorted(clans.unique()):
            m = clans == c
            ax.scatter(sub.loc[m, "pc1"], sub.loc[m, "pc2"],
                       s=6, alpha=0.5, label=f"{c} ({m.sum()})")
        ax.set_title(f"NOISE codas, n_clicks={n}, PCA(2D), n={len(sub)}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=7, loc="best", markerscale=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"noise_pca_n{n}.png", dpi=120)
        plt.close(fig)


def diag5_echolocation(df: pd.DataFrame) -> pd.DataFrame:
    """Flag NOISE rows that look like echolocation: n_clicks >= 8 and all
    available ICIs >= ECHO_ICI_MIN_S and monotonically (weakly) increasing
    or roughly constant. Sperm-whale echolocation has regular spacing.
    """
    rows = []
    for n in range(ECHO_MIN_N, 11):
        cols = [f"ICI{i}" for i in range(1, n)]
        sub = df[df["is_noise"] & (df["n_clicks"] == n)
                 & df[cols].notna().all(axis=1)].copy()
        if len(sub) == 0:
            rows.append({"n_clicks": n, "n_noise": 0,
                         "n_echo_like": 0, "echo_rate_in_noise": float("nan")})
            continue
        X = sub[cols].to_numpy(dtype=float)
        long_ici = (X >= ECHO_ICI_MIN_S).all(axis=1)
        # roughly regular: max-min within row <= 0.5 * mean (clicks at
        # near-constant spacing). Echolocation regularity is tighter than
        # social codas.
        rng = X.max(axis=1) - X.min(axis=1)
        mean = X.mean(axis=1)
        regular = rng <= 0.5 * mean
        echo_like = long_ici & regular
        rows.append({
            "n_clicks": n,
            "n_noise": int(len(X)),
            "n_echo_like": int(echo_like.sum()),
            "echo_rate_in_noise": float(echo_like.mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== §0 data-quality diagnostics ===")
    print(f"loading {CLASSIFIED.relative_to(REPO)} ...", flush=True)
    df = _load_non_dswp()
    n_total = len(df)
    n_noise = int(df["is_noise"].sum())
    print(f"  non-DSWP rows: {n_total:,}   NOISE: {n_noise:,} "
          f"({n_noise/n_total:.2%})")

    print("\n[diag 1] NOISE rate by recording_id ...")
    rec = diag1_by_recording(df)
    rec.to_csv(OUT / "noise_by_recording.csv")
    print(f"  recordings: {len(rec)}; "
          f"top 10 by NOISE rate (rate, n_codas, source):")
    head = rec.head(10)
    for rid, row in head.iterrows():
        print(f"    {str(rid)[:30]:<30}  {row['noise_rate']:6.2%}  "
              f"{int(row['n_codas']):>5}  {row['source']}")
    print(f"  median recording NOISE rate: {rec['noise_rate'].median():.2%}")
    print(f"  IQR: {rec['noise_rate'].quantile(0.25):.2%} – "
          f"{rec['noise_rate'].quantile(0.75):.2%}")

    print("\n[diag 2] NOISE rate by clan ...")
    clan = diag2_by_clan(df)
    clan.to_csv(OUT / "noise_by_clan.csv")
    for c, row in clan.iterrows():
        print(f"    {str(c):<8} {row['noise_rate']:6.2%}  "
              f"({int(row['n_noise']):,}/{int(row['n_codas']):,})")

    print("\n[diag 3] classifier_distance distribution ...")
    dist = diag3_distance_histogram(df)
    dist.to_csv(OUT / "distance_distribution.csv", index=False)
    print(dist.to_string(index=False))

    print("\n[diag 4] PCA of NOISE codas (per length) ...")
    pca_df = diag4_pca(df)
    if len(pca_df):
        pca_df.to_csv(OUT / "noise_pca.csv", index=False)
        _plot_pca(pca_df, OUT / "pca_plots")
        for n, sub in pca_df.groupby("length"):
            v1 = sub["pc1"].var()
            v2 = sub["pc2"].var()
            print(f"  n={n}: {len(sub):>5} NOISE codas; "
                  f"var(pc1)/var(pc2) = {v1/max(v2,1e-9):5.2f}")
    else:
        print("  (no NOISE rows in length range)")

    print("\n[diag 5] echolocation sanity ...")
    echo = diag5_echolocation(df)
    echo.to_csv(OUT / "echolocation_check.csv", index=False)
    print(echo.to_string(index=False))

    print(f"\nartefacts: {OUT.relative_to(REPO)}/")
    for f in sorted(OUT.iterdir()):
        if f.is_file():
            print(f"  {f.name}")


if __name__ == "__main__":
    main()
