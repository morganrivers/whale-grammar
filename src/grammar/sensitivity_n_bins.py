#!/usr/bin/env python3
"""
Sensitivity analysis: vary N_BINS in [2, 3, 4, 7, 10] and record, for each:
  - unique ICI strings / total codas
  - Morfessor MDL cost (bits)
  - morpheme inventory size
  - % strings segmented into >=2 morphemes
  - mean morpheme length (occurrence-weighted)
  - avg purity of top-20 morphemes w.r.t. rhythm_class
  - n morphemes spanning >1 rhythm_class (n>=50)
  - n morphemes confined to 1 rhythm_class  (n>=50)

Outputs outputs/morphemes/sensitivity_n_bins.json and a compact table to stdout.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import morfessor

DATA_DIR = Path("data/classified")
OUT_DIR = Path("outputs/morphemes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BINS_LIST = [2, 3, 4, 7, 10]
MIN_CLICKS = 3
MAX_CLICKS = 40


def load_data():
    df = pd.read_csv(DATA_DIR / "codas_classified.csv", low_memory=False)
    return df


def compute_global_bin_edges(df, n_bins):
    edges = {}
    for pos in range(1, MAX_CLICKS):
        col = f"ICI{pos}"
        if col not in df.columns:
            break
        vals = df[col].dropna().values
        if len(vals) < n_bins * 10:
            continue
        q = np.linspace(0, 100, n_bins + 1)
        e = np.percentile(vals, q)
        e = np.unique(e)
        if len(e) > 1:
            edges[col] = e
    return edges


def ici_row_to_string(row, edges, labels):
    n_bins = len(labels)
    symbols = []
    for pos in range(1, int(row["n_clicks"])):
        col = f"ICI{pos}"
        val = row.get(col, np.nan)
        if pd.isna(val):
            break
        if col not in edges:
            mid = n_bins // 2
            symbols.append(labels[mid])
            continue
        e = edges[col]
        idx = int(np.searchsorted(e[1:-1], val))
        idx = max(0, min(idx, n_bins - 1))
        symbols.append(labels[idx])
    return "".join(symbols)


def build_ici_corpus(df, edges, labels, min_len=2):
    counts = Counter()
    word_to_rc: dict[str, list] = defaultdict(list)
    for _, row in df.iterrows():
        n = int(row["n_clicks"]) if not pd.isna(row["n_clicks"]) else 0
        if n < MIN_CLICKS or n > MAX_CLICKS:
            continue
        s = ici_row_to_string(row, edges, labels)
        if len(s) < min_len:
            continue
        counts[s] += 1
        rc = row.get("rhythm_class", np.nan)
        if not pd.isna(rc):
            word_to_rc[s].append(int(rc))
    return counts, word_to_rc


def run_morfessor(word_counts):
    model = morfessor.BaselineModel()
    data = [(cnt, w) for w, cnt in word_counts.items() if len(w) >= 1]
    model.load_data(data)
    model.train_batch()
    cost = model.get_cost()
    segs = {}
    for w in word_counts:
        morphs, _ = model.viterbi_segment(w)
        segs[w] = morphs
    return model, segs, cost


def morpheme_inventory(word_counts, segmentations):
    inv = Counter()
    for w, morphs in segmentations.items():
        freq = word_counts[w]
        for m in morphs:
            inv[m] += freq
    return inv


def purity_analysis(segmentations, word_counts, word_to_rc):
    morph_to_rc: dict[str, list] = defaultdict(list)
    for w, morphs in segmentations.items():
        rcs = word_to_rc.get(w, [])
        for m in morphs:
            morph_to_rc[m].extend(rcs)

    results = []
    for m, rcs in morph_to_rc.items():
        if not rcs:
            continue
        total = len(rcs)
        modal_rc, modal_count = Counter(rcs).most_common(1)[0]
        purity = modal_count / total
        n_types = len(set(rcs))
        results.append({
            "morpheme": m,
            "total_occurrences": total,
            "n_rhythm_classes": n_types,
            "purity": round(purity, 3),
        })
    results.sort(key=lambda x: -x["total_occurrences"])
    return results


def analyse_one(df, n_bins):
    import string
    # Build labels: use uppercase letters for <=26 bins, else use two-char codes
    if n_bins <= 26:
        labels = list(string.ascii_uppercase[:n_bins])
    else:
        labels = [f"{i:02d}" for i in range(n_bins)]

    edges = compute_global_bin_edges(df, n_bins)
    corpus, word_to_rc = build_ici_corpus(df, edges, labels)

    if not corpus:
        return None

    model, segs, mdl_cost = run_morfessor(corpus)
    inv = morpheme_inventory(corpus, segs)

    n_segmented = sum(1 for v in segs.values() if len(v) > 1)
    pct_segmented = 100 * n_segmented / len(segs) if segs else 0.0

    # Occurrence-weighted mean morpheme length
    total_occ = sum(inv.values())
    mean_len = sum(len(m) * cnt for m, cnt in inv.items()) / total_occ if total_occ else 0.0

    purity_rows = purity_analysis(segs, corpus, word_to_rc)

    avg_purity_top20 = float(np.mean([r["purity"] for r in purity_rows[:20]])) if purity_rows else 0.0
    n_spanning = sum(1 for r in purity_rows if r["n_rhythm_classes"] > 1 and r["total_occurrences"] >= 50)
    n_confined = sum(1 for r in purity_rows if r["n_rhythm_classes"] == 1 and r["total_occurrences"] >= 50)

    return {
        "n_bins": n_bins,
        "unique_strings": len(corpus),
        "total_codas": int(sum(corpus.values())),
        "morfessor_mdl_cost": round(mdl_cost, 1),
        "morpheme_inventory_size": len(inv),
        "pct_segmented_2plus": round(pct_segmented, 1),
        "mean_morpheme_len": round(mean_len, 3),
        "avg_purity_top20": round(avg_purity_top20, 3),
        "n_spanning_morphemes_n50": n_spanning,
        "n_confined_morphemes_n50": n_confined,
        "top10_morphemes": [(m, int(c)) for m, c in inv.most_common(10)],
    }


def main():
    print("Loading data...")
    df = load_data()
    valid = df[df["n_clicks"].between(MIN_CLICKS, MAX_CLICKS) & df["rhythm_class"].notna()].copy()
    print(f"  Valid codas: {len(valid)}")

    results = []
    for n_bins in N_BINS_LIST:
        print(f"\n  N_BINS={n_bins} ...", end=" ", flush=True)
        row = analyse_one(valid, n_bins)
        if row:
            results.append(row)
            print(
                f"strings={row['unique_strings']}  morphemes={row['morpheme_inventory_size']}"
                f"  mdl={row['morfessor_mdl_cost']:.0f}  purity={row['avg_purity_top20']:.3f}"
            )

    # Print comparison table
    print("\n")
    cols = [
        ("n_bins",                  "N_BINS",        6),
        ("unique_strings",          "uniq_str",       9),
        ("morfessor_mdl_cost",      "MDL_cost",       10),
        ("morpheme_inventory_size", "n_morphemes",    12),
        ("pct_segmented_2plus",     "%_seg>=2",        9),
        ("mean_morpheme_len",       "mean_len",        9),
        ("avg_purity_top20",        "purity@20",       10),
        ("n_spanning_morphemes_n50","span_n50",         9),
        ("n_confined_morphemes_n50","conf_n50",         9),
    ]
    header = "  ".join(f"{label:>{width}}" for _, label, width in cols)
    print(header)
    print("-" * len(header))
    for r in results:
        row_str = "  ".join(f"{r[key]:>{width}}" for key, _, width in cols)
        print(row_str)

    print("\n  Top 10 morphemes per N_BINS:")
    for r in results:
        print(f"\n  N_BINS={r['n_bins']}: {r['top10_morphemes']}")

    out_path = OUT_DIR / "sensitivity_n_bins.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
