"""Hybrid Sharma-anchored / Pacific-discovery classifier.

Documented at ``docs/obsidian/classifier/classifier.md`` and
``docs/obsidian/classifier/pacific-extension.md``. Promoted from
``src/validation/hybrid_classify.py``. Three passes:

1. **Sharma 'real' anchor.** Every DSWP row whose published ``CodaType``
   is a non-``-NOISE`` value is locked to that label (origin
   ``dswp-real``).

2. **kNN + τ matching for non-DSWP codas.** Per length n in 3..10:

      train kNN k=5 on (Sharma 'real' rows of length n, their CodaType)
      query each non-DSWP coda of length n; predict majority CodaType +
        nearest-neighbour distance d
      accept (origin ``pacific-matched``) iff d ≤ τ[predicted CodaType]

   τ is computed per ``(length, codatype)`` from within-DSWP nearest-of-
   same-class distances:

      τ = max( NOSC_p99, TAU_FLOOR )

   where ``NOSC_p99`` is the 99-th percentile of nearest-same-class
   distance among Sharma 'real' rows of that length and codatype.
   ``TAU_FLOOR = 0.10 s`` keeps very tight Sharma types from rejecting
   Pacific neighbours that sit just outside the within-corpus jitter.

3. **OPTICSxi discovery on the residual pool.** Pool members:

      Sharma rows whose CodaType ends in ``-NOISE``  (second-chance
        recovery)
      ∪
      Non-DSWP rows that step 2 didn't accept

   Per length, ELKI 0.7.1 OPTICSxi at xi=0.04 minpts=10 on the pool's
   ICI matrix. Clusters of ≥ minpts members become discovered Pacific
   types (origin ``discovery-cluster``); ELKI's "rest" bucket
   (cluster_id 0) and any sub-minpts cluster are final NOISE
   (origin ``discovery-noise``).

4. **Cluster naming.** Discovered clusters are auto-named via
   :mod:`cluster_names` (``5RP1``, ``5P2``, etc.).

OPTICS outputs are cached at ``data/cache/optics_<tag>_n<N>_mp<M>.npz``
keyed by a hash of the input matrix.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from src.validation import elki_optics

from . import cluster_names

REPO = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO / "data" / "cache"

LENGTH_RANGE = range(3, 11)
XI = 0.04
MINPTS = 10
KNN_K = 5
# Floor on τ. ELKI's xi-extraction over-fragments tight Gero clusters so
# their NOSC_p99 alone is much smaller than the natural Pacific-coda
# variance around the same centroid; without a floor every Pacific coda
# gets rejected. 0.10 s ≈ half the median inter-CodaType centroid
# distance at n=5, so two codas within this floor are closer to each
# other than to the next-nearest type.
TAU_FLOOR_S = 0.10
# Back-compat alias still imported by some validation modules.
RADIUS_FLOOR_S = TAU_FLOOR_S
# ELKI's "rest" bucket convention: cluster_id 0 holds the points OPTICSxi
# couldn't fit into a real cluster.
PACIFIC_NOISE_CLUSTER_ID = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ici_columns(n: int) -> list[str]:
    return [f"ICI{i}" for i in range(1, n)]


def _is_noise_label(v) -> bool:
    return isinstance(v, str) and v.endswith("-NOISE")


def _ici_matrix(df: pd.DataFrame, n: int) -> tuple[np.ndarray, np.ndarray]:
    cols = _ici_columns(n)
    if not all(c in df.columns for c in cols):
        return df.index[:0].to_numpy(), np.empty((0, n - 1))
    mask = (df["n_clicks"] == n) & df[cols].notna().all(axis=1)
    return (df.index[mask].to_numpy(),
            df.loc[mask, cols].to_numpy(dtype=float))


def _run_optics_cached(X: np.ndarray, *, n: int, tag: str,
                        minpts: int = MINPTS) -> np.ndarray:
    """Cached ELKI OPTICSXi keyed by content hash of X."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"optics_{tag}_n{n}_mp{minpts}.npz"
    h = np.int64(hash(X.tobytes()))
    if cache_path.exists():
        z = np.load(cache_path)
        if "hash" in z and int(z["hash"]) == int(h) \
                and z["labels"].shape[0] == X.shape[0]:
            return z["labels"]
    labels = elki_optics.run(X, xi=XI, minpts=minpts)
    np.savez(cache_path, labels=labels, hash=h)
    return labels


# ---------------------------------------------------------------------------
# τ table (computed inline from Sharma 'real' rows; no external CSV needed)
# ---------------------------------------------------------------------------


def compute_tau_lookup(df: pd.DataFrame,
                       dswp_truth_col: str) -> dict[tuple[int, str], float]:
    """Return ``{(length, codatype) -> τ}`` for every Sharma 'real' type
    that has at least 2 members of its length.

    τ = ``max(NOSC_p99, TAU_FLOOR_S)``. ``NOSC_p99`` is the 99-th
    percentile nearest-of-same-class distance within Sharma 'real' rows
    of that length and codatype. Types with too few members fall back to
    the per-length median of the populated types.
    """
    is_dswp = df["source"] == "sharma2024_dswp"
    truth = df[dswp_truth_col]
    is_real = is_dswp & truth.notna() & ~truth.map(_is_noise_label)

    tau: dict[tuple[int, str], float] = {}
    for n in LENGTH_RANGE:
        cols = _ici_columns(n)
        m = is_real & (df["n_clicks"] == n) & df[cols].notna().all(axis=1)
        if not m.any():
            continue
        X = df.loc[m, cols].to_numpy(dtype=float)
        y = df.loc[m, dswp_truth_col].astype(str).to_numpy()
        nosc_p99: dict[str, float] = {}
        for t in np.unique(y):
            mask_t = y == t
            X_t = X[mask_t]
            if X_t.shape[0] < 2:
                continue
            knn = KNeighborsClassifier(n_neighbors=2, metric="euclidean")
            knn.fit(X_t, np.zeros(X_t.shape[0]))
            d, _ = knn.kneighbors(X_t, n_neighbors=2)
            nosc_p99[t] = float(np.percentile(d[:, 1], 99))
        if nosc_p99:
            med = float(np.nanmedian(list(nosc_p99.values())))
        else:
            med = TAU_FLOOR_S
        for t in np.unique(y):
            base = nosc_p99.get(t, med)
            tau[(int(n), str(t))] = max(base, TAU_FLOOR_S)
    return tau


# ---------------------------------------------------------------------------
# Stage 2: kNN+τ matching for non-DSWP codas
# ---------------------------------------------------------------------------


def match_other_codas(df: pd.DataFrame, dswp_truth_col: str,
                      tau: dict[tuple[int, str], float]) -> pd.DataFrame:
    """Per-length kNN k=5 against Sharma 'real' rows; accept iff
    nearest-neighbour distance ≤ τ[predicted codatype].

    Returns a DataFrame indexed like ``df`` with columns
    ``matched_codatype`` (str or NA) and ``knn_distance`` (float, NaN if
    not queried).
    """
    out = pd.DataFrame({
        "matched_codatype": pd.Series(pd.NA, index=df.index, dtype="object"),
        "knn_distance": pd.Series(np.nan, index=df.index, dtype="float64"),
    })
    is_dswp = df["source"] == "sharma2024_dswp"
    truth = df[dswp_truth_col]
    is_real = is_dswp & truth.notna() & ~truth.map(_is_noise_label)

    for n in LENGTH_RANGE:
        cols = _ici_columns(n)
        train_mask = (is_real & (df["n_clicks"] == n)
                      & df[cols].notna().all(axis=1))
        if int(train_mask.sum()) < KNN_K + 1:
            continue
        X_train = df.loc[train_mask, cols].to_numpy(dtype=float)
        y_train = df.loc[train_mask, dswp_truth_col].astype(str).to_numpy()
        knn = KNeighborsClassifier(n_neighbors=KNN_K, metric="euclidean")
        knn.fit(X_train, y_train)

        q_mask = ((~is_dswp) & (df["n_clicks"] == n)
                  & df[cols].notna().all(axis=1))
        q_idx = df.index[q_mask].to_numpy()
        if len(q_idx) == 0:
            continue
        X_q = df.loc[q_idx, cols].to_numpy(dtype=float)
        y_pred = knn.predict(X_q)
        d_nn = knn.kneighbors(X_q, n_neighbors=1)[0][:, 0]

        out.loc[q_idx, "knn_distance"] = d_nn
        for i, ix in enumerate(q_idx):
            t = tau.get((int(n), str(y_pred[i])), float("inf"))
            if d_nn[i] <= t:
                out.at[ix, "matched_codatype"] = str(y_pred[i])
    return out


# ---------------------------------------------------------------------------
# Stage 3: OPTICSxi discovery on the residual pool
# ---------------------------------------------------------------------------


def discover_pool(df: pd.DataFrame,
                  pool_mask: pd.Series,
                  ) -> tuple[dict[int, list[dict]], pd.Series]:
    """OPTICSxi per length on the discovery pool.

    ``pool_mask`` selects pool members (Sharma-NOISE ∪ Pacific residual).
    Returns ``(pool_models, pool_codatype)``:

    * ``pool_models[n]`` — list of cluster dicts
      (``cluster_id, n_clicks, centroid, mean_duration, n_members,
      unified_indices, placeholder``). Naming added later via
      :func:`cluster_names.name_pacific_clusters`.
    * ``pool_codatype`` — Series indexed like ``df`` carrying either
      a ``_pac_n{n}_c{cid}`` placeholder (clustered) or a
      ``{n}-NOISE`` string (final NOISE), or NA outside the pool.
    """
    pool_codatype = pd.Series(pd.NA, index=df.index, dtype="object")
    pool_models: dict[int, list[dict]] = {}

    for n in LENGTH_RANGE:
        cols = _ici_columns(n)
        m = pool_mask & (df["n_clicks"] == n) & df[cols].notna().all(axis=1)
        idx = df.index[m].to_numpy()
        if len(idx) < MINPTS:
            for ix in idx:
                pool_codatype.at[ix] = f"{n}-NOISE"
            pool_models[n] = []
            continue
        X = df.loc[idx, cols].to_numpy(dtype=float)
        labels = _run_optics_cached(X, n=n, tag="hybrid_pool", minpts=MINPTS)

        clusters: list[dict] = []
        for cl in np.unique(labels):
            mask = labels == cl
            n_members = int(mask.sum())
            if int(cl) == PACIFIC_NOISE_CLUSTER_ID or n_members < MINPTS:
                for ix in idx[mask]:
                    pool_codatype.at[ix] = f"{n}-NOISE"
                continue
            members_X = X[mask]
            centroid = members_X.mean(axis=0)
            durations = pd.to_numeric(
                df.loc[idx[mask], "coda_duration_s"], errors="coerce"
            ).to_numpy()
            mean_dur = (float(np.nanmean(durations))
                        if np.isfinite(durations).any() else float("nan"))
            placeholder = f"_pac_n{n}_c{int(cl)}"
            for ix in idx[mask]:
                pool_codatype.at[ix] = placeholder
            clusters.append({
                "cluster_id": int(cl),
                "n_clicks": n,
                "centroid": centroid,
                "mean_duration": mean_dur,
                "n_members": n_members,
                "unified_indices": idx[mask],
                "placeholder": placeholder,
            })
        pool_models[n] = clusters
    return pool_models, pool_codatype


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def attach_dswp_truth(df_unified: pd.DataFrame,
                      df_dominica: pd.DataFrame,
                      *, col: str = "_codatype") -> pd.DataFrame:
    """Add column ``col`` with Gero's CodaType for DSWP rows; NA elsewhere."""
    df = df_unified.copy()
    is_dswp = df["source"] == "sharma2024_dswp"
    lookup = df_dominica.set_index(df_dominica["codaNUM2018"].astype(str))[
        "CodaType"]
    truth = pd.Series(pd.NA, index=df.index, dtype="object")
    truth.loc[is_dswp] = (df.loc[is_dswp, "source_coda_id"].astype(str)
                          .map(lookup).to_numpy())
    df[col] = truth
    return df


def run_optics_pipeline(df_unified: pd.DataFrame,
                        df_dominica: pd.DataFrame) -> pd.DataFrame:
    """End-to-end: returns ``df_unified`` with these columns added/overwritten:

    * ``coda_type_gero21`` — DSWP 'real' = direct join. Non-DSWP matched
      = inherited Sharma label. Discovered = cluster-name (``5RP1``,
      ``5P2``). Final NOISE = ``{n}-NOISE``.
    * ``classifier_origin`` — one of ``dswp-real``, ``pacific-matched``,
      ``discovery-cluster``, ``discovery-noise``, or NA (out-of-range
      / missing ICIs).
    * ``classifier_distance`` — kNN nearest-neighbour distance for
      ``pacific-matched`` rows; NaN otherwise.
    """
    df = attach_dswp_truth(df_unified, df_dominica)
    coda_type = pd.Series(pd.NA, index=df.index, dtype="object")
    origin = pd.Series(pd.NA, index=df.index, dtype="object")
    distance = pd.Series(np.nan, index=df.index, dtype="float64")

    truth = df["_codatype"]
    is_dswp = df["source"] == "sharma2024_dswp"
    is_real_dswp = is_dswp & truth.notna() & ~truth.map(_is_noise_label)
    is_dswp_noise = is_dswp & truth.map(_is_noise_label)

    # --- Stage 1 — Sharma 'real' anchor ---
    coda_type.loc[is_real_dswp] = truth[is_real_dswp]
    origin.loc[is_real_dswp] = "dswp-real"

    # --- Stage 2 — kNN+τ matching for non-DSWP codas ---
    tau = compute_tau_lookup(df, "_codatype")
    matched = match_other_codas(df, "_codatype", tau)
    distance.loc[matched.index] = matched["knn_distance"]
    matched_mask = matched["matched_codatype"].notna()
    coda_type.loc[matched_mask] = matched.loc[matched_mask, "matched_codatype"]
    origin.loc[matched_mask] = "pacific-matched"

    # --- Stage 3 — discovery on Sharma-NOISE ∪ Pacific residual ---
    is_pac_residual = (~is_dswp) & origin.isna()
    pool_mask = is_dswp_noise | is_pac_residual
    pool_models, pool_codatype = discover_pool(df, pool_mask)

    # --- Stage 4 — name discovered clusters ---
    ec_names = set(coda_type.dropna().unique()) - {pd.NA}
    flat: list[dict] = []
    for clist in pool_models.values():
        flat.extend(clist)
    cluster_names.name_pacific_clusters(flat, ec_names)
    placeholder_to_name = {c["placeholder"]: c["name"] for c in flat}

    is_placeholder = pool_codatype.map(
        lambda v: isinstance(v, str) and v.startswith("_pac_n"))
    is_pool_noise = pool_codatype.map(
        lambda v: isinstance(v, str) and v.endswith("-NOISE")
        and not v.startswith("_pac_n"))
    coda_type.loc[is_placeholder] = (
        pool_codatype.loc[is_placeholder].map(placeholder_to_name))
    origin.loc[is_placeholder] = "discovery-cluster"
    coda_type.loc[is_pool_noise] = pool_codatype.loc[is_pool_noise]
    origin.loc[is_pool_noise] = "discovery-noise"

    df["coda_type_gero21"] = coda_type
    df["classifier_origin"] = origin
    df["classifier_distance"] = distance
    df.drop(columns=["_codatype"], inplace=True)
    return df
