"""Render a small curated set of representative cluster examples in the
'3 Sharma + 6 Hersh' format the user asked for.

Picks the most informative clusters from the hybrid pipeline output:

  - **2 Sharma 'real' types** with the most Pacific matches (Stage 2
    working).
  - **2 mixed discovered clusters** with the highest DSWP-NOISE
    contribution (Sharma-NOISE second-chance working).
  - **2 pure-Pacific discovered clusters** (genuinely-novel Pacific
    types).
  - **1 NOISE-sample row** per length, in the bottom panel of the figure.

Each cluster row:
  - Up to 3 random DSWP codas (blue)
  - Up to 6 random Hersh codas (orange)
  - For one Hersh row, also overlay 3 'tree-neighbour' codas from the
    adjacent OPTICS cluster (grey, dashed) so you can see what just
    *missed* this cluster's xi-cut.

Output: ``data/diagnostics/hybrid_v1/representative.png``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.validation.hybrid_classify import (
    OUT, _ici_cols, _is_noise_label,
)
from src.validation.phase1b_knn import DOMINICA, coda_type_for_dswp
from src.validation.piano_roll import (
    SOURCE_COLORS, SOURCE_SHORT, _click_times, _coda_dict, _sample,
)

REPO = Path(__file__).resolve().parents[2]
UNIFIED = REPO / "data" / "upstream" / "codas_unified.csv"
ASSIGN = OUT / "assignments.csv"
DISC = OUT / "discovered_clusters.csv"
PNG_OUT = OUT / "representative.png"
RNG = np.random.default_rng(7)
N_DSWP_PER_ROW = 3
N_HERSH_PER_ROW = 6


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    u = pd.read_csv(UNIFIED, low_memory=False)
    d = pd.read_csv(DOMINICA)
    d.columns = [c.strip().lstrip("﻿") for c in d.columns]
    u["coda_type_truth"] = coda_type_for_dswp(u, d)
    state = pd.read_csv(ASSIGN, index_col=0)
    disc = pd.read_csv(DISC)
    return u, state, disc


def pick_clusters(u: pd.DataFrame, state: pd.DataFrame,
                  disc: pd.DataFrame) -> list[dict]:
    """Choose 6 clusters that span Sharma-matched, mixed-discovered,
    pure-Pacific-discovered. Returns list of dicts with keys:
      - title (str)
      - kind ("sharma_real" | "discovered_mixed" | "discovered_pacific")
      - dswp_indices (np.ndarray of unified row indices)
      - hersh_indices (np.ndarray)
      - birth_indices (np.ndarray)
      - n_clicks (int)
    """
    out: list[dict] = []
    is_dswp = u["source"] == "sharma2024_dswp"
    is_hersh = u["source"] == "hersh2022_pacific"
    is_birth = u["source"] == "sharma2025_birth"
    truth = u["coda_type_truth"]

    # --- 2 Sharma 'real' types with the most Pacific matches ---
    matched = state[state["origin"] == "pacific-matched"]
    by_label = matched["final_label"].value_counts()
    for label in by_label.head(2).index:
        type_n_clicks = int(u.loc[
            (truth == label) & is_dswp,
            "n_clicks"].dropna().mode().iloc[0])
        cols = _ici_cols(type_n_clicks)
        m_dswp = (truth == label) & is_dswp & u[cols].notna().all(axis=1)
        m_pac = ((state["final_label"] == label)
                 & (state["origin"] == "pacific-matched"))
        out.append({
            "title": f"Sharma type {label}  (n_clicks={type_n_clicks}; "
                     f"DSWP={int(m_dswp.sum())}, "
                     f"Pacific-matched={int(m_pac.sum())})",
            "kind": "sharma_real",
            "n_clicks": type_n_clicks,
            "dswp_indices": u.index[m_dswp].to_numpy(),
            "hersh_indices": state.index[m_pac & is_hersh].to_numpy(),
            "birth_indices": state.index[m_pac & is_birth].to_numpy(),
        })

    # --- 2 discovered clusters with the most DSWP-NOISE ---
    mixed = disc[disc["n_dswp"] > 0].sort_values("n_dswp", ascending=False)
    for _, row in mixed.head(2).iterrows():
        n = int(row["n_clicks"])
        cols = _ici_cols(n)
        members = state[(state["final_label"] == row["cluster_label"])
                        & u[cols].notna().all(axis=1)]
        out.append({
            "title": f"Discovered {row['cluster_label']}  (n_clicks={n}; "
                     f"DSWP={row['n_dswp']}, Hersh={row['n_hersh']}, "
                     f"birth={row['n_birth']})",
            "kind": "discovered_mixed",
            "n_clicks": n,
            "dswp_indices": members.index[
                u.loc[members.index, "source"] == "sharma2024_dswp"].to_numpy(),
            "hersh_indices": members.index[
                u.loc[members.index, "source"] == "hersh2022_pacific"].to_numpy(),
            "birth_indices": members.index[
                u.loc[members.index, "source"] == "sharma2025_birth"].to_numpy(),
        })

    # --- 2 pure-Pacific discovered clusters (no DSWP) ---
    pure_pac = disc[(disc["n_dswp"] == 0)
                    & (disc["n_hersh"] >= 5)].sort_values(
        "n_members", ascending=False)
    for _, row in pure_pac.head(2).iterrows():
        n = int(row["n_clicks"])
        cols = _ici_cols(n)
        members = state[(state["final_label"] == row["cluster_label"])
                        & u[cols].notna().all(axis=1)]
        out.append({
            "title": f"Discovered {row['cluster_label']}  (n_clicks={n}; "
                     f"pure Pacific: Hersh={row['n_hersh']}, "
                     f"birth={row['n_birth']})",
            "kind": "discovered_pacific",
            "n_clicks": n,
            "dswp_indices": np.array([], dtype=np.int64),
            "hersh_indices": members.index[
                u.loc[members.index, "source"] == "hersh2022_pacific"].to_numpy(),
            "birth_indices": members.index[
                u.loc[members.index, "source"] == "sharma2025_birth"].to_numpy(),
        })
    return out


def find_neighbour_cluster(state: pd.DataFrame, target_label: str,
                            n_clicks: int, u: pd.DataFrame) -> str | None:
    """Heuristic 'tree neighbour': return the discovered cluster of the
    same length whose centroid is closest to ``target_label``'s centroid
    in the same (length, source-restricted) ICI space.

    Fast and approximate; an OPTICS reachability-ordering implementation
    would be faithful but requires plumbing the reachability output out
    of ELKI. For visual verification this Euclidean-centroid heuristic
    is sufficient — what we want to show is "stuff that almost matched".
    """
    cols = _ici_cols(n_clicks)
    target_idx = state.index[
        (state["final_label"] == target_label)
        & (u["n_clicks"] == n_clicks)
        & u[cols].notna().all(axis=1)]
    if len(target_idx) == 0:
        return None
    target_cen = u.loc[target_idx, cols].to_numpy(dtype=float).mean(axis=0)

    cands = state.loc[
        (state["origin"] == "discovery-cluster")
        & (state["final_label"] != target_label)
        & (u["n_clicks"] == n_clicks)
        & u[cols].notna().all(axis=1)]
    if len(cands) == 0:
        return None
    by_label = cands.groupby("final_label")
    best_label = None
    best_dist = float("inf")
    for label, sub in by_label:
        cen = u.loc[sub.index, cols].to_numpy(dtype=float).mean(axis=0)
        d = float(np.linalg.norm(cen - target_cen))
        if d < best_dist:
            best_dist = d
            best_label = label
    return best_label


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    u, state, disc = load()
    state["source"] = u.loc[state.index, "source"].values
    clusters = pick_clusters(u, state, disc)
    print(f"picked {len(clusters)} clusters for the curated set")

    n_panels = len(clusters)
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 2.4 * n_panels),
                              squeeze=False)
    for ax, cl in zip(axes[:, 0], clusters):
        n = cl["n_clicks"]
        # 3 DSWP, 6 Hersh sampling
        dswp_pick = (u.loc[cl["dswp_indices"]]
                     if len(cl["dswp_indices"]) else u.iloc[:0])
        dswp_pick = _sample(dswp_pick, N_DSWP_PER_ROW)
        hersh_pick = (u.loc[cl["hersh_indices"]]
                      if len(cl["hersh_indices"]) else u.iloc[:0])
        hersh_pick = _sample(hersh_pick, N_HERSH_PER_ROW)

        # tree-neighbour: 3 codas from the closest-centroid cluster of same
        # length (only for discovered clusters; for Sharma types we'd be
        # comparing against a different Sharma type, which is fine too).
        nb_codas = []
        if cl["kind"].startswith("discovered"):
            nb_label = find_neighbour_cluster(
                state, target_label=cl["title"].split()[1],
                n_clicks=n, u=u)
            if nb_label:
                cols = _ici_cols(n)
                nb_idx = state.index[
                    (state["final_label"] == nb_label)
                    & (u["n_clicks"] == n)
                    & u[cols].notna().all(axis=1)]
                nb_pick = _sample(u.loc[nb_idx], 3)
                nb_codas = [_coda_dict(r, n) for _, r in nb_pick.iterrows()]
                nb_label_str = f" | nearest other cluster: {nb_label}"
            else:
                nb_label_str = ""
        else:
            nb_label_str = ""

        codas_dswp = [_coda_dict(r, n) for _, r in dswp_pick.iterrows()]
        codas_hersh = [_coda_dict(r, n) for _, r in hersh_pick.iterrows()]
        all_codas = codas_dswp + codas_hersh + nb_codas
        if not all_codas:
            ax.text(0.5, 0.5, "(no eligible codas)", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_yticks([])
            ax.set_title(cl["title"])
            continue
        x_max = max(c["times"][-1] for c in all_codas) * 1.05

        # render rows: DSWP at top, Hersh middle, neighbours at bottom.
        rows = []
        for c in codas_dswp:
            rows.append(("dswp", c, "solid"))
        for c in codas_hersh:
            rows.append(("hersh", c, "solid"))
        for c in nb_codas:
            rows.append(("neighbour", c, "dashed"))

        for i, (kind, c, ls) in enumerate(rows):
            y = i
            ax.hlines(y, 0, x_max, colors="lightgrey", linewidth=0.4)
            color = (SOURCE_COLORS.get(c["source"], "grey")
                      if kind != "neighbour" else "grey")
            for t in c["times"]:
                ax.vlines(t, y - 0.35, y + 0.35, colors=color,
                           linewidth=1.5, linestyles=ls)
            tag = (f"{SOURCE_SHORT.get(c['source'], '?')} {c.get('meta', '')}"
                   if kind != "neighbour"
                   else f"NEIGHBOUR-CLUSTER {SOURCE_SHORT.get(c['source'], '?')} "
                        f"{c.get('meta', '')}")
            ax.text(x_max * 1.005, y, tag, fontsize=6, va="center",
                    color=color)

        ax.set_xlim(0, x_max * 1.25)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks([])
        ax.set_xlabel("time within coda (s)", fontsize=8)
        ax.set_title(cl["title"] + nb_label_str, fontsize=9)

    fig.suptitle(
        "Hybrid classifier — representative cluster set\n"
        "Each panel: up to 3 random DSWP (blue) + 6 random Hersh (orange) "
        "+ 3 nearest-other-cluster codas (grey dashed)",
        fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {PNG_OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
