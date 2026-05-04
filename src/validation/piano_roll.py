"""Piano-roll visualizer for coda clusters.

A coda's clicks become vertical ticks at their cumulative ICI times. One
coda per row, stacked vertically. Source colour-coded so DSWP / Hersh /
birth contributions are visible at a glance.

Renders three sets of PNGs under ``data/diagnostics/hybrid_v1/piano_rolls/``:

  - ``sharma_real/{type}.png`` — for each Sharma 'real' type, top panel
    shows DSWP members, bottom panel shows Pacific codas matched into it
    via kNN+τ. Up to ``ROWS_PER_PANEL`` random samples of each.
  - ``discovered/{label}.png`` — for each discovered cluster from the
    OPTICSxi pass, samples of its members.
  - ``noise_sample.png``       — random samples of the final NOISE
    bucket, stratified by length.

Run from repo root::

    python -m src.validation.piano_roll
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.validation.hybrid_classify import (
    LENGTH_RANGE, OUT, _ici_cols, _is_noise_label, main as run_hybrid,
)

ROLL_OUT = OUT / "piano_rolls"
ROWS_PER_PANEL = 8
NOISE_PER_LENGTH = 3
RNG = np.random.default_rng(0)

SOURCE_COLORS = {
    "sharma2024_dswp": "#1f77b4",   # blue
    "hersh2022_pacific": "#ff7f0e",  # orange
    "sharma2025_birth": "#2ca02c",   # green
}
SOURCE_SHORT = {
    "sharma2024_dswp": "DSWP",
    "hersh2022_pacific": "Hersh",
    "sharma2025_birth": "birth",
}


def _click_times(ici_row: np.ndarray) -> np.ndarray:
    """Cumulative click times: t=0, ICI1, ICI1+ICI2, ..."""
    return np.concatenate([[0.0], np.cumsum(ici_row)])


def _safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_"
                    for c in str(s))


def _render_panel(ax, codas: list[dict], title: str,
                   x_max: float | None = None) -> None:
    """codas: list of {'times': np.ndarray, 'source': str, 'meta': str}.
    Draws one row per coda, with vertical ticks at click times."""
    if not codas:
        ax.text(0.5, 0.5, "(empty)", ha="center", va="center",
                transform=ax.transAxes, color="grey")
        ax.set_title(title)
        ax.set_yticks([])
        return
    if x_max is None:
        x_max = max((float(c["times"][-1]) for c in codas), default=1.0)
        x_max = max(x_max * 1.05, 0.5)

    for i, c in enumerate(codas):
        y = i
        color = SOURCE_COLORS.get(c["source"], "grey")
        # baseline
        ax.hlines(y, 0, x_max, colors="lightgrey", linewidth=0.5)
        # ticks
        for t in c["times"]:
            ax.vlines(t, y - 0.35, y + 0.35, colors=color, linewidth=1.5)
        # right-side annotation: source + meta
        ax.text(x_max * 1.005, y,
                f"{SOURCE_SHORT.get(c['source'], '?')} {c.get('meta', '')}",
                fontsize=6, va="center", color=color)
    ax.set_xlim(0, x_max * 1.18)
    ax.set_ylim(-0.6, len(codas) - 0.4)
    ax.set_yticks([])
    ax.set_xlabel("time within coda (s)")
    ax.set_title(title, fontsize=9)


def _coda_dict(row: pd.Series, n: int) -> dict:
    cols = _ici_cols(n)
    icis = row[cols].to_numpy(dtype=float)
    return {
        "times": _click_times(icis),
        "source": str(row["source"]),
        "meta": (f"clan={row['clan']}"
                  if pd.notna(row.get("clan")) else "(no clan)"),
    }


def _sample(df: pd.DataFrame, k: int) -> pd.DataFrame:
    if len(df) <= k:
        return df
    idx = RNG.choice(len(df), k, replace=False)
    return df.iloc[idx]


# ---------------------------------------------------------------------------

def render_sharma_real(u: pd.DataFrame, state: pd.DataFrame,
                       out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    is_dswp = u["source"] == "sharma2024_dswp"
    truth = u["coda_type_truth"]
    is_real = is_dswp & truth.notna() & ~truth.map(_is_noise_label)
    real_types = sorted(truth[is_real].dropna().astype(str).unique())
    print(f"  rendering {len(real_types)} Sharma 'real' types ...",
          flush=True)
    for t in real_types:
        # find the length(s) this type covers
        members = u[is_real & (truth.astype(str) == t)]
        # one figure per (type, length) — Sharma types are length-specific
        for n, sub in members.groupby("n_clicks"):
            cols = _ici_cols(int(n))
            sub = sub[sub[cols].notna().all(axis=1)]
            if len(sub) == 0:
                continue
            dswp_rows = _sample(sub, ROWS_PER_PANEL)
            dswp_codas = [_coda_dict(r, int(n)) for _, r in dswp_rows.iterrows()]

            # Pacific matches into this type:
            pac_idx = state.index[(state["final_label"] == t)
                                  & (state["origin"] == "pacific-matched")
                                  & (u["n_clicks"] == n)
                                  & u[cols].notna().all(axis=1)]
            pac_rows = u.loc[pac_idx]
            pac_rows = _sample(pac_rows, ROWS_PER_PANEL)
            pac_codas = [_coda_dict(r, int(n)) for _, r in pac_rows.iterrows()]

            x_max = max(
                [c["times"][-1] for c in dswp_codas + pac_codas if len(c["times"])],
                default=1.0)
            n_pac_total = int(((state["final_label"] == t)
                               & (state["origin"] == "pacific-matched")).sum())
            fig, axes = plt.subplots(2, 1, figsize=(8, 6),
                                      gridspec_kw={"height_ratios": [1, 1]})
            _render_panel(axes[0], dswp_codas,
                           f"{t}  (DSWP members, n={len(sub)} total, showing "
                           f"{len(dswp_codas)})", x_max=x_max)
            _render_panel(axes[1], pac_codas,
                           f"Pacific matched into {t} via kNN+τ "
                           f"(n={n_pac_total} total, showing "
                           f"{len(pac_codas)})", x_max=x_max)
            fig.tight_layout()
            fig.savefig(out_dir / f"{_safe_filename(t)}_n{int(n)}.png",
                         dpi=110)
            plt.close(fig)


def render_discovered(u: pd.DataFrame, state: pd.DataFrame,
                       cluster_records: list[dict],
                       out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  rendering {len(cluster_records)} discovered clusters ...",
          flush=True)
    for r in cluster_records:
        n = int(r["n_clicks"])
        cols = _ici_cols(n)
        members = u.loc[r["unified_indices"]]
        members = members[members[cols].notna().all(axis=1)]
        if len(members) == 0:
            continue
        # Stratify the sample by source so each source visible if present.
        by_src = []
        for src in SOURCE_COLORS:
            sub = members[members["source"] == src]
            if len(sub) == 0:
                continue
            k = max(2, ROWS_PER_PANEL // 2)
            by_src.append(_sample(sub, k))
        sample = pd.concat(by_src) if by_src else _sample(members, ROWS_PER_PANEL)
        codas = [_coda_dict(row, n) for _, row in sample.iterrows()]

        title = (f"{r['cluster_label']}  n_clicks={n}  members={r['n_members']}  "
                 f"(DSWP {r['n_dswp']} / Hersh {r['n_hersh']} / "
                 f"birth {r['n_birth']})")
        fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(codas) + 1)))
        _render_panel(ax, codas, title)
        fig.tight_layout()
        fig.savefig(out_dir / f"{_safe_filename(r['cluster_label'])}.png",
                     dpi=110)
        plt.close(fig)


def render_noise(u: pd.DataFrame, state: pd.DataFrame,
                 out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    is_noise = state["origin"] == "discovery-noise"
    print(f"  rendering NOISE sample ({int(is_noise.sum()):,} total)",
          flush=True)
    rows_per_section = []
    for n in LENGTH_RANGE:
        cols = _ici_cols(n)
        m = is_noise & (u["n_clicks"] == n) & u[cols].notna().all(axis=1)
        sub = u.loc[m]
        if len(sub) == 0:
            continue
        rows_per_section.append((n, _sample(sub, NOISE_PER_LENGTH)))
    if not rows_per_section:
        return
    total_rows = sum(len(s) for _, s in rows_per_section)
    fig, axes = plt.subplots(len(rows_per_section), 1,
                              figsize=(8, max(3, 0.45 * total_rows + 1)),
                              squeeze=False)
    for ax, (n, sub) in zip(axes[:, 0], rows_per_section):
        codas = [_coda_dict(row, int(n)) for _, row in sub.iterrows()]
        _render_panel(ax, codas, f"NOISE sample, n_clicks={n}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main() -> None:
    u, state, cluster_records = run_hybrid()
    ROLL_OUT.mkdir(parents=True, exist_ok=True)
    print("\n=== piano-roll rendering ===")
    render_sharma_real(u, state, ROLL_OUT / "sharma_real")
    render_discovered(u, state, cluster_records, ROLL_OUT / "discovered")
    render_noise(u, state, ROLL_OUT / "noise_sample.png")
    print(f"\nartefacts under {ROLL_OUT.relative_to(OUT.parents[1])}")


if __name__ == "__main__":
    main()
