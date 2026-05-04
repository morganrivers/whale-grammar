"""Print the four-numbers table from the latest hybrid run, and write a
detailed RESULTS.md that documents what each row means.

Reads ``data/diagnostics/hybrid_v1/assignments.csv`` (produced by
``hybrid_classify.py``); the table below is computed from that file each
time, so the numbers always reflect the current run's outputs.

Run from repo root::

    python -m src.validation.results_report
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.validation.hybrid_classify import OUT
from src.validation.phase1b_knn import DOMINICA, UNIFIED, coda_type_for_dswp

REPO = Path(__file__).resolve().parents[2]
ASSIGN = OUT / "assignments.csv"
DISC = OUT / "discovered_clusters.csv"
RESULTS_MD = OUT / "RESULTS.md"


def _is_noise_label(v) -> bool:
    return isinstance(v, str) and v.endswith("-NOISE")


def compute_numbers() -> dict:
    u = pd.read_csv(UNIFIED, low_memory=False)
    d = pd.read_csv(DOMINICA)
    d.columns = [c.strip().lstrip("﻿") for c in d.columns]
    u["coda_type_truth"] = coda_type_for_dswp(u, d)
    state = pd.read_csv(ASSIGN, index_col=0)
    state["source"] = u.loc[state.index, "source"].values
    disc = pd.read_csv(DISC)

    is_dswp = u["source"] == "sharma2024_dswp"
    is_pac = ~is_dswp
    truth = u["coda_type_truth"]
    is_dswp_noise = is_dswp & truth.map(_is_noise_label)

    pac_in_range = is_pac & u["n_clicks"].between(3, 10)
    n_pac_in_range = int(pac_in_range.sum())
    n_pac_matched = int((is_pac
                         & (state["origin"] == "pacific-matched")).sum())
    n_pac_clustered = int((is_pac
                           & (state["origin"] == "discovery-cluster")).sum())
    n_pac_noise = int((is_pac
                       & (state["origin"] == "discovery-noise")).sum())

    n_dn = int(is_dswp_noise.sum())
    n_dn_clustered = int((is_dswp_noise
                          & (state["origin"] == "discovery-cluster")).sum())
    n_dn_noise = int((is_dswp_noise
                      & (state["origin"] == "discovery-noise")).sum())

    pool_clustered = n_pac_clustered + n_dn_clustered
    pool_noise = n_pac_noise + n_dn_noise
    pool_total = pool_clustered + pool_noise

    n_clusters = len(disc)

    # Per-source fate breakdown — see "How birth was handled" section.
    by_source = pd.crosstab(state["source"],
                             state["origin"].fillna("(none)"))
    # Ensure all expected columns exist even when zero.
    for col in ["dswp-real", "pacific-matched", "discovery-cluster",
                "discovery-noise", "(none)"]:
        if col not in by_source.columns:
            by_source[col] = 0
    by_source = by_source[[
        "dswp-real", "pacific-matched", "discovery-cluster",
        "discovery-noise", "(none)"]]
    by_source["total"] = by_source.sum(axis=1)
    pct = by_source.drop(columns=["total"]).div(by_source["total"], axis=0)

    n_birth_matched = int(by_source.loc["sharma2025_birth", "pacific-matched"])
    n_hersh_matched = int(by_source.loc["hersh2022_pacific", "pacific-matched"])
    n_birth_discovered = int(
        by_source.loc["sharma2025_birth", "discovery-cluster"])

    return {
        "pac_in_range": n_pac_in_range,
        "pac_matched": n_pac_matched,
        "pac_clustered": n_pac_clustered,
        "pac_noise": n_pac_noise,
        "dn_total": n_dn,
        "dn_clustered": n_dn_clustered,
        "dn_noise": n_dn_noise,
        "pool_total": pool_total,
        "pool_noise": pool_noise,
        "pool_clustered": pool_clustered,
        "n_clusters": n_clusters,
        "by_source": by_source,
        "by_source_pct": pct,
        "n_birth_matched": n_birth_matched,
        "n_hersh_matched": n_hersh_matched,
        "n_birth_discovered": n_birth_discovered,
    }


def print_table(s: dict) -> None:
    rows = [
        ("1",
         "Pacific matched to Sharma 'real' types (kNN+τ)",
         f"{s['pac_matched']:,} / {s['pac_in_range']:,}",
         f"{s['pac_matched'] / max(s['pac_in_range'], 1):.1%}"),
        ("2",
         f"Pacific in newly-discovered OPTICSxi clusters "
         f"({s['n_clusters']} clusters)",
         f"{s['pac_clustered']:,} / {s['pac_in_range']:,}",
         f"{s['pac_clustered'] / max(s['pac_in_range'], 1):.1%}"),
        ("3",
         "Pool members staying NOISE (Pacific residual + Sharma-NOISE)",
         f"{s['pool_noise']:,} / {s['pool_total']:,}",
         f"{s['pool_noise'] / max(s['pool_total'], 1):.1%}"),
        ("4a",
         "Sharma-NOISE clustered with Pacific (second-chance recoveries)",
         f"{s['dn_clustered']:,} / {s['dn_total']:,}",
         f"{s['dn_clustered'] / max(s['dn_total'], 1):.1%}"),
        ("4b",
         "Sharma-NOISE staying NOISE",
         f"{s['dn_noise']:,} / {s['dn_total']:,}",
         f"{s['dn_noise'] / max(s['dn_total'], 1):.1%}"),
    ]
    headers = ["#", "What", "Count", "%"]
    widths = [
        max(len(headers[0]), max(len(r[0]) for r in rows)),
        max(len(headers[1]), max(len(r[1]) for r in rows)),
        max(len(headers[2]), max(len(r[2]) for r in rows)),
        max(len(headers[3]), max(len(r[3]) for r in rows)),
    ]
    cells = "│ {0:^{w0}} │ {1:^{w1}} │ {2:^{w2}} │ {3:^{w3}} │"
    line = "├─" + "─┼─".join("─" * w for w in widths) + "─┤"
    top = "┌─" + "─┬─".join("─" * w for w in widths) + "─┐"
    bot = "└─" + "─┴─".join("─" * w for w in widths) + "─┘"
    print(top)
    print(cells.format(*headers, w0=widths[0], w1=widths[1],
                        w2=widths[2], w3=widths[3]))
    print(line)
    for i, row in enumerate(rows):
        print(cells.format(*row, w0=widths[0], w1=widths[1],
                            w2=widths[2], w3=widths[3]))
        if i < len(rows) - 1:
            print(line)
    print(bot)


def _format_source_table(by_source: pd.DataFrame,
                          pct: pd.DataFrame) -> str:
    """Render the per-source fate table as a Markdown block."""
    SRC_LABEL = {
        "hersh2022_pacific": "Hersh (Pacific)",
        "sharma2025_birth": "birth (DSWP calves)",
        "sharma2024_dswp": "Sharma DSWP",
    }
    cols_order = ["dswp-real", "pacific-matched", "discovery-cluster",
                  "discovery-noise", "(none)"]
    header_label = {
        "dswp-real": "DSWP-anchor",
        "pacific-matched": "matched to Sharma",
        "discovery-cluster": "in discovered cluster",
        "discovery-noise": "final NOISE",
        "(none)": "out-of-range",
    }
    lines = [
        "| source | total | "
        + " | ".join(header_label[c] for c in cols_order) + " |",
        "|---|---:|" + "|".join(["---:"] * len(cols_order)) + "|",
    ]
    for src in ["hersh2022_pacific", "sharma2025_birth",
                "sharma2024_dswp"]:
        if src not in by_source.index:
            continue
        row = by_source.loc[src]
        prow = pct.loc[src]
        cells = [SRC_LABEL.get(src, src), f"{int(row['total']):,}"]
        for c in cols_order:
            n = int(row[c])
            p = float(prow[c])
            if n == 0:
                cells.append("—")
            else:
                cells.append(f"{n:,} ({p:.1%})")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_results_md(s: dict) -> None:
    pct = lambda n, d: f"{n / max(d, 1):.1%}"
    source_table = _format_source_table(s["by_source"], s["by_source_pct"])
    md = f"""# Hybrid classifier — RESULTS

This file is regenerated by ``python -m src.validation.results_report``
from the current contents of ``assignments.csv`` and
``discovered_clusters.csv``. The four numbers below describe **what
happened to every coda** in the unified corpus when it went through:

  1. **Stage 1 — Sharma anchor**: every Sharma DSWP coda whose published
     ``CodaType`` is a *real* (non-``-NOISE``) type keeps that label.
  2. **Stage 2 — Pacific kNN+τ matching**: every Pacific (Hersh + birth)
     coda gets its 5 nearest Sharma DSWP neighbours; the majority-vote
     CodaType is accepted iff the nearest-neighbour distance is within
     ``τ = max(NOSC-p99, 0.10 s)`` of the predicted type's natural
     within-type spread.
  3. **Stage 3 — OPTICSxi discovery**: the discovery pool (Sharma-NOISE
     codas ∪ Pacific codas Stage 2 didn't accept) is fed to ELKI 0.7.1
     OPTICSxi (xi=0.04, minpts=10) per length. Each cluster of ≥10
     codas is a discovered type. ELKI's "rest" bucket and any cluster
     smaller than minpts are final NOISE.

## The four numbers

| # | What | Count | % |
|---|---|---:|---:|
| 1 | Pacific matched to Sharma 'real' types (kNN+τ) | {s['pac_matched']:,} / {s['pac_in_range']:,} | {pct(s['pac_matched'], s['pac_in_range'])} |
| 2 | Pacific in newly-discovered OPTICSxi clusters ({s['n_clusters']} clusters) | {s['pac_clustered']:,} / {s['pac_in_range']:,} | {pct(s['pac_clustered'], s['pac_in_range'])} |
| 3 | Pool members staying NOISE (Pacific residual + Sharma-NOISE) | {s['pool_noise']:,} / {s['pool_total']:,} | {pct(s['pool_noise'], s['pool_total'])} |
| 4a | Sharma-NOISE clustered with Pacific (second-chance recoveries) | {s['dn_clustered']:,} / {s['dn_total']:,} | {pct(s['dn_clustered'], s['dn_total'])} |
| 4b | Sharma-NOISE staying NOISE | {s['dn_noise']:,} / {s['dn_total']:,} | {pct(s['dn_noise'], s['dn_total'])} |

## How birth (Sharma 2025) was handled

Three sources flow into this pipeline:

  - ``sharma2024_dswp`` — Caribbean adult DSWP corpus, **with** published
    ``CodaType`` labels via ``codaNUM2018 ↔ source_coda_id`` join.
  - ``hersh2022_pacific`` — Pacific corpus, **no** published per-coda
    type labels (only Hersh's clan labels).
  - ``sharma2025_birth`` — DSWP **calves** (Dominica-clan neonates),
    **no** published per-coda type labels.

Operational definition: anything that's *not* ``sharma2024_dswp`` is
treated as "Pacific" (i.e. unlabelled, must be classified). That's the
only source distinction the pipeline makes. So birth flows through
identically to Hersh:

  - **Stage 1 (anchor)**: birth = no anchor (no truth label).
  - **Stage 2 (kNN+τ matching)**: birth queried against Sharma DSWP,
    accepted into a Sharma type if within τ.
  - **Stage 3 (OPTICSxi discovery)**: birth residuals (those Stage 2
    didn't accept) join the discovery pool alongside Hersh residuals
    and Sharma's own ``-NOISE`` rows.
  - **kNN training set**: Sharma 'real' rows only. Birth is *not* in
    training — it's only ever a query.

What actually happened to each source:

{source_table}

Birth has the **highest match rate of any source** ({pct(s['n_birth_matched'], int(s['by_source'].loc['sharma2025_birth', 'total']))}). That's expected — birth codas
are Dominica-clan calves, so their repertoire is the DSWP repertoire,
just with developmental variation. They contribute almost nothing to
the discovered-cluster count (only {s['n_birth_discovered']} birth codas
across all {s['n_clusters']} clusters), confirming the discovered
clusters are essentially a Hersh-Pacific phenomenon.

Open question (handoff §5.2): birth carries ``clan == "EC1"`` because
they're Dominica calves, but Hersh's adult corpus also has an "EC1"
clan in the Pacific that is biologically different. The pipeline keeps
them apart by ``source`` — so all clan-stratified analysis can still
distinguish them. If you wanted to anchor birth as Caribbean truth (use
their kNN-predicted Sharma label as a "truth" surrogate), that would
require a separate decision; the current pipeline gives birth no
preferential treatment.

## What each row means

### Row 1 — Pacific matched to Sharma 'real' types

Number of Pacific (Hersh + birth) codas in the 3–10 click range whose
nearest 5 DSWP neighbours voted for a non-``-NOISE`` Sharma CodaType,
*and* whose nearest-neighbour Euclidean distance was within τ. These
codas inherit the matched Sharma label. Denominator is **all in-range
Pacific codas** (n_clicks 3–10 with all ICIs present).

A high number here means cross-corpus rhythm matching is working — most
Pacific codas have a Sharma analogue. This is the bulk of the corpus
covered by Sharma's 18-ish 'real' types.

### Row 2 — Pacific in newly-discovered OPTICSxi clusters

Pacific codas that *didn't* match a Sharma type at Stage 2 but joined a
≥10-member cluster in the OPTICSxi discovery pass. These are
Pacific-native types beyond Sharma's repertoire. Denominator is
all in-range Pacific codas (same as row 1).

A high number here means there's a sizable Pacific-only repertoire that
Sharma's Caribbean corpus doesn't cover. A low number means most of the
"new" Pacific signal is just minor variation on Sharma types and didn't
form coherent novel clusters.

### Row 3 — Pool members staying NOISE

The discovery pool combines Pacific Stage-2-rejects and Sharma's own
``-NOISE``-labelled codas. Pool members that ELKI couldn't fit into a
≥10-member cluster end up as final NOISE. Denominator is the **whole
pool** (both sources together).

This is the "we couldn't classify these" rate, conditioned on having
already rejected easy Sharma matches at Stage 2. High = lots of
genuinely-novel-or-isolated codas. Low = most of the residual *did*
find friends.

### Row 4a / 4b — Sharma-NOISE second chance

The 600 (or so) Sharma DSWP codas labelled ``X-NOISE`` in the published
data went into the discovery pass alongside Pacific residuals.

  - **4a** = Sharma-NOISE rows that joined a discovered cluster (often
    with Pacific co-members). These were "noise" in DSWP-only context
    but found ≥9 friends once Pacific data was added — a signal that
    Sharma-NOISE wasn't always genuine noise; some of it was rare types
    that needed cross-corpus reinforcement to surface.
  - **4b** = Sharma-NOISE rows that stayed NOISE after the second pass.
    These look like genuine within-DSWP noise.

(The remaining count beyond 4a + 4b is a small residual of Sharma-NOISE
rows whose ``n_clicks`` falls outside the 3–10 range — structurally
excluded by the locked methodology.)

## Run parameters

  - **Stage 2 τ-table**: ``max(NOSC-p99, 0.10 s)``. NOSC-p99 is the
    99th-percentile distance from a Sharma DSWP coda to its nearest
    same-type DSWP neighbour, measured per (length, codatype) in
    ``data/diagnostics/dswp_variance.csv``. The 0.10 s floor matches
    ``B_classify_optics.RADIUS_FLOOR_S`` and is justified by the
    median inter-type centroid distance at n=5 being ~0.20 s.
  - **Stage 3 OPTICSxi**: xi = 0.04, minpts = 10, distance = Euclidean
    on absolute ICIs, per-length runs (n = 3..10). Same parameters
    Sharma + Gero used; cluster_id = 0 is ELKI's "rest" bucket
    (treated as NOISE).
  - **kNN**: k = 5, validated at 97.64% LOO on DSWP labels in Phase 1b.

## τ-validation on Sharma — over-classification check

To verify that the wider τ doesn't accept Sharma codas as the *wrong*
Sharma type, ``loo_knn_tau.py`` ran a per-length leave-one-out kNN+τ
test on the 8,119 Sharma 'real' rows. Headline numbers:

| metric | value |
|---|---:|
| Acceptance rate (within τ) | **100.0%** of 8,119 Sharma-real codas |
| LOO accuracy among accepted | **99.72%** |
| Over-classification (accepted, wrong type) | **23 / 8,119 = 0.28%** |
| Rejection (would go to discovery despite being Sharma-real) | **3 / 8,119 = 0.04%** |

Per-length breakdown is in ``loo_knn_tau.csv``. The wider τ is safe:
0.28% cross-classification is well within the noise of kNN's own LOO
error and there is no length where the rate spikes (max is n=4 at 1.3%).

Re-run with::

    python -m src.validation.loo_knn_tau

## Caveats

  - **Absolute-ICI feature space**: matching is sensitive to tempo. A
    Pacific coda matches the Sharma type at the *same* tempo, not the
    same rhythm at a different tempo. This is consistent with Sharma's
    own pipeline (tempo is a separate side label, not a clustering
    feature) but means our Pacific-into-Sharma matching only covers
    one slice of each Sharma type's tempo envelope.
  - **Per-length OPTICSxi**: clusters can't span lengths. A coda whose
    natural type spans 5–6 click counts will fragment into separate
    clusters per length.

## Re-running

```
# Stage 1–3 + piano rolls (also re-renders all per-cluster PNGs):
python -m src.validation.piano_roll

# Just the curated representative comparison sheet:
python -m src.validation.representative_set

# Just regenerate this RESULTS.md from the current CSVs:
python -m src.validation.results_report
```
"""
    RESULTS_MD.write_text(md)


def main() -> None:
    s = compute_numbers()
    print_table(s)
    write_results_md(s)
    print(f"\n-> {RESULTS_MD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
