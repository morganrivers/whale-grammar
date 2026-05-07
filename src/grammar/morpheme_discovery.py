#!/usr/bin/env python3
"""
Morpheme discovery for sperm whale coda data.

Track 1 (ICI): Discretise inter-click intervals to a 5-symbol alphabet,
    run Morfessor MDL segmentation across all coda lengths together.

Track 1b (Augmented): Repeat Track 1 with tempo appended as a coda-level
    modifier character; analyse whether rubato / ornament are consistent
    within ICI morpheme groups.

Track 2 (Multi-coda): Find recurring subsequences of coda types (rhythm_class)
    in dialogue sequences using n-gram frequency + PMI + BPE-style merging.

Analysis:
    - Morpheme purity w.r.t. existing rhythm_class labels (do OPTICS types
      respect morpheme boundaries or cut across them?)
    - Estimated sequence-length compression for transformer training.
    - Whether discovered morphemes undermine or reinforce the existing
      coda-type inventory.
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import morfessor

DATA_DIR = Path("data/classified")
OUT_DIR = Path("outputs/morphemes")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
N_BINS = 4
LABELS = "ABCD"
# Tempo is 1-5; map to single chars that don't clash with ABCDE
TEMPO_CHARS = {1: "P", 2: "Q", 3: "R", 4: "S", 5: "T"}
TEMPO_FALLBACK = "U"
RUBATO_CHARS = {"-": "L", "/": "K", "\\": "J"}  # level, accel, decel
# Codas to include (3-click gives only 2 ICIs – short but fine across-lengths)
MIN_CLICKS = 3
MAX_CLICKS = 40


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_DIR / "codas_classified.csv", low_memory=False)
    dialogues = pd.read_csv(DATA_DIR / "whale_dialogues.csv")
    rindex = pd.read_csv(DATA_DIR / "rhythm_class_index.csv")
    return df, dialogues, rindex


# ── Track 1: ICI discretisation ───────────────────────────────────────────────
def compute_global_bin_edges(df, n_bins=N_BINS):
    """Per-ICI-position quantile bin edges computed across ALL coda lengths."""
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


def ici_row_to_string(row, edges):
    """Convert one row's ICI values to an ABCDE symbol string."""
    symbols = []
    for pos in range(1, int(row["n_clicks"])):
        col = f"ICI{pos}"
        val = row.get(col, np.nan)
        if pd.isna(val):
            break
        if col not in edges:
            symbols.append("C")  # fallback to middle bin
            continue
        e = edges[col]
        idx = int(np.searchsorted(e[1:-1], val))
        idx = max(0, min(idx, N_BINS - 1))
        symbols.append(LABELS[idx])
    return "".join(symbols)


def build_ici_corpus(df, edges, augment_tempo=False, min_len=2):
    """Return Counter{string -> occurrence_count} for all valid codas."""
    counts = Counter()
    for _, row in df.iterrows():
        n = int(row["n_clicks"]) if not pd.isna(row["n_clicks"]) else 0
        if n < MIN_CLICKS or n > MAX_CLICKS:
            continue
        s = ici_row_to_string(row, edges)
        if len(s) < min_len:
            continue
        if augment_tempo:
            t = row.get("tempo", np.nan)
            t_key = float(t) if not pd.isna(t) else float("nan")
            t_char = TEMPO_CHARS.get(t_key, TEMPO_FALLBACK)
            s = s + t_char  # tempo appended as a suffix "modifier"
        counts[s] += 1
    return counts


def run_morfessor(word_counts, save_path: Path | None = None):
    """Fit Morfessor Baseline, return (model, {word: [morpheme_list]}).

    If save_path is given, the trained model is persisted there as a binary file.
    """
    model = morfessor.BaselineModel()
    data = [(cnt, w) for w, cnt in word_counts.items() if len(w) >= 1]
    model.load_data(data)
    model.train_batch()
    segs = {}
    for w in word_counts:
        morphs, _ = model.viterbi_segment(w)
        segs[w] = morphs
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        morfessor.MorfessorIO().write_binary_model_file(str(save_path), model)
    return model, segs


def morpheme_inventory(word_counts, segmentations):
    """Count total occurrences of each discovered morpheme."""
    inv = Counter()
    for w, morphs in segmentations.items():
        freq = word_counts[w]
        for m in morphs:
            inv[m] += freq
    return inv


# ── Track 1 analysis: alignment with rhythm_class ─────────────────────────────
def morpheme_purity_analysis(df, edges, segmentations, word_to_rc):
    """
    For each discovered morpheme, compute how often codas sharing that morpheme
    also share the same rhythm_class. High purity = morpheme maps cleanly onto
    OPTICS type. Low purity = morpheme cuts across existing types.
    """
    # Build morpheme -> list of rhythm_classes
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
        results.append(
            {
                "morpheme": m,
                "total_occurrences": total,
                "n_rhythm_classes": n_types,
                "modal_rc": int(modal_rc) if not pd.isna(modal_rc) else None,
                "purity": round(purity, 3),
            }
        )
    results.sort(key=lambda x: -x["total_occurrences"])
    return results


def feature_consistency(df, edges, segmentations):
    """
    For each ICI morpheme, report mean ± std of tempo and fraction with
    each rubato direction. This tests whether tempo/rubato act as modifiers
    that are independent of the morpheme vs co-lexicalised with it.
    """
    morph_rows: dict[str, list] = defaultdict(list)
    for _, row in df.iterrows():
        n = int(row["n_clicks"]) if not pd.isna(row["n_clicks"]) else 0
        if n < MIN_CLICKS or n > MAX_CLICKS:
            continue
        s = ici_row_to_string(row, edges)
        if len(s) < 2:
            continue
        morphs, _ = None, None
        if s in segmentations:
            morphs = segmentations[s]
        else:
            continue
        tempo = row.get("tempo", np.nan)
        rubato = row.get("rubato", None)
        extra = row.get("extra_click", np.nan)
        for m in morphs:
            morph_rows[m].append((tempo, rubato, extra))

    results = []
    for m, rows in morph_rows.items():
        tempos = [r[0] for r in rows if not pd.isna(r[0])]
        rubatos = [r[1] for r in rows if r[1] in RUBATO_CHARS]
        extras = [r[2] for r in rows if not pd.isna(r[2])]
        rb_counter = Counter(rubatos)
        rb_total = len(rubatos)
        results.append(
            {
                "morpheme": m,
                "n": len(rows),
                "tempo_mean": round(float(np.mean(tempos)), 3) if tempos else None,
                "tempo_std": round(float(np.std(tempos)), 3) if tempos else None,
                "rubato_counts": dict(rb_counter) if rb_total > 0 else None,
                "rubato_n": rb_total,
                "ornament_rate": round(float(np.mean(extras)), 3) if extras else None,
            }
        )
    results.sort(key=lambda x: -x["n"])
    return results


# ── Track 2: Multi-coda morphemes ─────────────────────────────────────────────
def build_coda_sequences(dialogues):
    """Group dialogue rows into per-sequence lists of rhythm_class values."""
    # Exclude silence token (98) and nulls
    seqs = (
        dialogues[dialogues["Coda"].notna() & (dialogues["Coda"] != 98)]
        .sort_values(["sequenceId", "itemPosition"])
        .groupby("sequenceId")["Coda"]
        .apply(list)
    )
    return seqs


def ngram_pmi(sequences, max_n=4, top_k=30):
    """
    Count n-grams up to max_n and compute PMI for bigrams and PMI-like
    extension for trigrams/4-grams (using product of marginals).
    """
    unigram: Counter = Counter()
    ngram_counts: dict[int, Counter] = {n: Counter() for n in range(2, max_n + 1)}
    total = 0

    for seq in sequences:
        seq = [int(c) for c in seq]
        for tok in seq:
            unigram[tok] += 1
            total += 1
        for n in range(2, max_n + 1):
            for i in range(len(seq) - n + 1):
                ngram_counts[n][tuple(seq[i : i + n])] += 1

    results = {}
    for n in range(2, max_n + 1):
        rows = []
        for gram, cnt in ngram_counts[n].items():
            if cnt < 5:
                continue
            marginal_product = math.prod(unigram[g] for g in gram)
            pmi = math.log2(
                (cnt / total) / (marginal_product / (total ** n))
            )
            rows.append(
                {
                    "gram": list(gram),
                    "count": cnt,
                    "pmi": round(pmi, 3),
                    # MDL savings: bits saved by treating gram as atomic
                    "mdl_saving_bits": round(
                        cnt * pmi, 2
                    ),
                }
            )
        rows.sort(key=lambda x: -x["mdl_saving_bits"])
        results[n] = rows[:top_k]

    return results, unigram, total


def bpe_merge(sequences, n_merges=20):
    """
    BPE-style morpheme merging on coda sequences.
    At each step merge the bigram with highest freq * PMI score.
    Returns the sequence of merges performed and the final vocabulary.
    """
    # Work with lists of ints
    corpus = [[int(c) for c in seq] for seq in sequences]

    # Vocab: original codas map to themselves; compound tokens use negative keys
    next_token = -1
    token_repr: dict[int, tuple] = {}  # compound_token -> component tuple

    merges = []
    for step in range(n_merges):
        total = sum(len(s) for s in corpus)
        if total == 0:
            break

        unigram: Counter = Counter()
        bigram: Counter = Counter()
        for seq in corpus:
            for tok in seq:
                unigram[tok] += 1
            for a, b in zip(seq, seq[1:]):
                bigram[(a, b)] += 1

        if not bigram:
            break

        # Score: freq * PMI
        best, best_score = None, -float("inf")
        for (a, b), cnt in bigram.items():
            if cnt < 3:
                continue
            pmi = math.log2((cnt / total) / ((unigram[a] / total) * (unigram[b] / total)))
            score = cnt * pmi
            if score > best_score:
                best_score = score
                best = (a, b)

        if best is None:
            break

        a, b = best
        new_tok = next_token
        next_token -= 1
        # Represent the new token
        def expand(t):
            if t in token_repr:
                return token_repr[t]
            return (t,)
        token_repr[new_tok] = expand(a) + expand(b)

        # Apply merge to corpus
        merged = 0
        new_corpus = []
        for seq in corpus:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    new_seq.append(new_tok)
                    i += 2
                    merged += 1
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_corpus.append(new_seq)
        corpus = new_corpus

        # Total tokens after merge
        total_after = sum(len(s) for s in corpus)

        merges.append(
            {
                "step": step + 1,
                "merged_pair": list(expand(a)) + list(expand(b)),
                "freq": bigram[(a, b)],
                "pmi": round(
                    math.log2(
                        (bigram[(a, b)] / total)
                        / ((unigram[a] / total) * (unigram[b] / total))
                    ),
                    3,
                ),
                "score": round(best_score, 2),
                "tokens_saved": merged,
                "total_tokens_after": total_after,
            }
        )

    initial_total = sum(len(s) for s in sequences)
    final_total = sum(len(s) for s in corpus)
    compression = 1.0 - final_total / initial_total if initial_total else 0.0
    return merges, compression


# ── Transformer efficiency estimate ───────────────────────────────────────────
def transformer_efficiency(bpe_merges, initial_tokens):
    """
    If we replaced BPE morphemes with single tokens, sequences would shrink.
    A shorter sequence with the same context window (K=8) covers more semantic
    distance: estimate the % reduction and what it means for context coverage.
    """
    final_tokens = bpe_merges[-1]["total_tokens_after"] if bpe_merges else initial_tokens
    reduction_pct = 100 * (1 - final_tokens / initial_tokens)
    context_k = 8
    effective_context_increase = context_k / (final_tokens / initial_tokens) - context_k
    return {
        "initial_tokens": initial_tokens,
        "final_tokens": final_tokens,
        "reduction_pct": round(reduction_pct, 2),
        "context_window_K": context_k,
        "effective_extra_context_from_compression": round(effective_context_increase, 2),
    }


# ── OPTICS tension analysis ────────────────────────────────────────────────────
def optics_tension(purity_rows, bpe_merges, rindex):
    """
    Summarise two questions:
    1. Do ICI morphemes split existing rhythm_class types? (purity < 1 for
       morphemes that appear within a single type → sub-type structure)
    2. Do ICI morphemes merge across types? (a morpheme appearing in > 1 type)
    3. Do any BPE morphemes cross coda-type boundaries in revealing ways?
    """
    # Split vs merge
    intra_split = [r for r in purity_rows if r["n_rhythm_classes"] == 1 and r["total_occurrences"] >= 50]
    cross_merge = [r for r in purity_rows if r["n_rhythm_classes"] > 1 and r["total_occurrences"] >= 50]

    return {
        "ici_morphemes_within_one_type": len(intra_split),
        "ici_morphemes_spanning_types": len(cross_merge),
        "avg_purity_top20": round(
            float(np.mean([r["purity"] for r in purity_rows[:20]])), 3
        ) if purity_rows else None,
        "interpretation": (
            "High purity → morphemes ≈ existing type boundaries (OPTICS reinforced). "
            "Low purity + spanning morphemes → ICI sub-patterns cut across OPTICS types "
            "(suggests compositional structure OPTICS conflates)."
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df, dialogues, rindex = load_data()

    # Restrict to codas with valid ICI data and rhythm_class
    valid = df[df["n_clicks"].between(MIN_CLICKS, MAX_CLICKS) & df["rhythm_class"].notna()].copy()
    print(f"  Codas with valid ICI + rhythm_class: {len(valid)}")

    # ── Track 1: ICI morphemes ────────────────────────────────────────────────
    print("\n── Track 1: ICI morphemes (across all lengths) ──────────────────")
    edges = compute_global_bin_edges(valid)
    print(f"  Computed bin edges for {len(edges)} ICI positions")

    ici_corpus = build_ici_corpus(valid, edges, augment_tempo=False)
    print(f"  Unique ICI strings: {len(ici_corpus)}  |  Total codas: {sum(ici_corpus.values())}")

    # Build word → rhythm_classes map for purity analysis
    word_to_rc: dict[str, list] = defaultdict(list)
    for _, row in valid.iterrows():
        n = int(row["n_clicks"])
        s = ici_row_to_string(row, edges)
        if len(s) >= 2:
            word_to_rc[s].append(int(row["rhythm_class"]))

    print("  Training Morfessor (Track 1)...")
    morfessor_save = OUT_DIR / f"morfessor_n{N_BINS}.bin"
    model_ici, segs_ici = run_morfessor(ici_corpus, save_path=morfessor_save)
    print(f"  Morfessor model saved to {morfessor_save}")
    inv_ici = morpheme_inventory(ici_corpus, segs_ici)

    n_segmented = sum(1 for v in segs_ici.values() if len(v) > 1)
    print(f"  Strings segmented into ≥2 morphemes: {n_segmented} / {len(segs_ici)}")
    print(f"  Morpheme inventory size: {len(inv_ici)}")
    print("\n  Top 25 ICI morphemes by occurrence:")
    for m, cnt in inv_ici.most_common(25):
        print(f"    {m!r:12s}  {cnt:6d}")

    purity = morpheme_purity_analysis(valid, edges, segs_ici, word_to_rc)
    print(f"\n  Purity analysis (top 10 morphemes by occurrence):")
    print(f"  {'morpheme':12s}  {'occurrences':>11}  {'n_types':>7}  {'purity':>7}")
    for r in purity[:10]:
        print(
            f"  {r['morpheme']!r:12s}  {r['total_occurrences']:11d}"
            f"  {r['n_rhythm_classes']:7d}  {r['purity']:7.3f}"
        )

    # ── Track 1b: Feature-augmented morphemes ─────────────────────────────────
    print("\n── Track 1b: ICI + tempo augmented morphemes ────────────────────")
    aug_corpus = build_ici_corpus(valid, edges, augment_tempo=True)
    print(f"  Unique augmented strings: {len(aug_corpus)}")
    print("  Training Morfessor (Track 1b)...")
    model_aug, segs_aug = run_morfessor(aug_corpus)
    inv_aug = morpheme_inventory(aug_corpus, segs_aug)

    n_seg_aug = sum(1 for v in segs_aug.values() if len(v) > 1)
    print(f"  Strings segmented into ≥2 morphemes: {n_seg_aug} / {len(segs_aug)}")
    print(f"  Morpheme inventory size: {len(inv_aug)}")
    print("\n  Top 25 augmented morphemes:")
    for m, cnt in inv_aug.most_common(25):
        print(f"    {m!r:12s}  {cnt:6d}")

    # Feature consistency for ICI-only morphemes
    print("\n  Feature consistency (tempo/rubato/ornament within ICI morphemes):")
    fc = feature_consistency(valid, edges, segs_ici)
    print(f"  {'morpheme':12s}  {'n':>6}  {'tempo_mean':>10}  {'tempo_std':>10}  {'orn_rate':>9}")
    for r in fc[:15]:
        tm = f"{r['tempo_mean']:.2f}" if r["tempo_mean"] is not None else "  n/a"
        ts = f"{r['tempo_std']:.2f}" if r["tempo_std"] is not None else "  n/a"
        orn = f"{r['ornament_rate']:.3f}" if r["ornament_rate"] is not None else "  n/a"
        print(f"  {r['morpheme']!r:12s}  {r['n']:6d}  {tm:>10}  {ts:>10}  {orn:>9}")

    # ── Track 2: Multi-coda morphemes ─────────────────────────────────────────
    print("\n── Track 2: Multi-coda morphemes ────────────────────────────────")
    sequences = build_coda_sequences(dialogues)
    print(f"  Sequences: {len(sequences)}  |  Total coda tokens: {sequences.apply(len).sum()}")
    seq_lengths = sequences.apply(len)
    print(f"  Sequence length: mean={seq_lengths.mean():.1f}  med={seq_lengths.median():.0f}  max={seq_lengths.max()}")

    pmi_results, unigram, total_tokens = ngram_pmi(sequences, max_n=4, top_k=25)

    for n in sorted(pmi_results):
        label = {2: "Bigrams", 3: "Trigrams", 4: "4-grams"}[n]
        print(f"\n  Top 20 {label} by MDL saving (bits):")
        print(f"  {'gram':<30}  {'count':>6}  {'pmi':>6}  {'mdl_saving':>10}")
        for r in pmi_results[n][:20]:
            gram_str = str(r["gram"])
            print(
                f"  {gram_str:<30}  {r['count']:6d}  {r['pmi']:6.2f}  {r['mdl_saving_bits']:10.1f}"
            )

    print("\n  BPE merging (20 steps)...")
    initial_total = int(sequences.apply(len).sum())
    bpe_merges, compression = bpe_merge(sequences, n_merges=20)
    print(f"  Compression after 20 BPE merges: {compression * 100:.2f}%")
    print(f"\n  BPE merge history:")
    print(f"  {'step':>4}  {'pair':<30}  {'freq':>6}  {'pmi':>6}  {'saved':>6}")
    for m in bpe_merges:
        pair_str = str(m["merged_pair"])
        print(
            f"  {m['step']:4d}  {pair_str:<30}  {m['freq']:6d}  {m['pmi']:6.2f}  {m['tokens_saved']:6d}"
        )

    # ── Transformer efficiency estimate ───────────────────────────────────────
    print("\n── Transformer efficiency estimate ──────────────────────────────")
    teff = transformer_efficiency(bpe_merges, initial_total)
    print(f"  Initial token count:  {teff['initial_tokens']:,}")
    print(f"  After BPE morphemes:  {teff['final_tokens']:,}")
    print(f"  Sequence compression: {teff['reduction_pct']}%")
    print(
        f"  With K=8 context window, compression yields effectively "
        f"{teff['effective_extra_context_from_compression']:.2f} extra context tokens on average"
    )
    print(
        "  Note: compression also shrinks vocabulary if morpheme tokens replace pairs,"
        "\n  which can reduce embedding lookup costs but requires retraining."
    )

    # ── OPTICS tension ────────────────────────────────────────────────────────
    print("\n── ICI morphemes vs existing OPTICS coda types ──────────────────")
    tension = optics_tension(purity, bpe_merges, rindex)
    print(f"  ICI morphemes confined to a single rhythm_class: {tension['ici_morphemes_within_one_type']}")
    print(f"  ICI morphemes spanning >1 rhythm_class:          {tension['ici_morphemes_spanning_types']}")
    print(f"  Average purity (top 20 morphemes):               {tension['avg_purity_top20']}")
    print(f"\n  {tension['interpretation']}")

    # Low-purity morphemes (potential OPTICS challenges)
    low_purity = [r for r in purity if r["purity"] < 0.5 and r["total_occurrences"] >= 30]
    if low_purity:
        print(f"\n  Low-purity morphemes (purity<0.5, n≥30) — possible sub-type structure:")
        for r in low_purity[:10]:
            print(
                f"    {r['morpheme']!r}  purity={r['purity']}  "
                f"n_types={r['n_rhythm_classes']}  occurrences={r['total_occurrences']}"
            )

    # High-purity spanning morphemes (possible shared roots across types)
    high_span = [r for r in purity if r["purity"] < 0.8 and r["n_rhythm_classes"] >= 3 and r["total_occurrences"] >= 50]
    if high_span:
        print(f"\n  Morphemes shared by ≥3 types (potential shared roots):")
        for r in high_span[:10]:
            print(
                f"    {r['morpheme']!r}  n_types={r['n_rhythm_classes']}  "
                f"purity={r['purity']}  occurrences={r['total_occurrences']}"
            )

    # ── Save outputs ──────────────────────────────────────────────────────────
    output = {
        "track1_ici_morphemes": {
            "config": {"n_bins": N_BINS, "across_all_lengths": True, "augment_tempo": False},
            "unique_strings": len(ici_corpus),
            "total_codas": int(sum(ici_corpus.values())),
            "strings_segmented": n_segmented,
            "morpheme_inventory": inv_ici.most_common(),
            "purity_analysis": purity[:50],
        },
        "track1b_augmented_morphemes": {
            "config": {"n_bins": N_BINS, "augment_tempo": True},
            "unique_strings": len(aug_corpus),
            "morpheme_inventory": inv_aug.most_common(),
            "strings_segmented": n_seg_aug,
            "feature_consistency": fc[:30],
        },
        "track2_multicoda": {
            "n_sequences": int(len(sequences)),
            "total_tokens": initial_total,
            "ngram_pmi": {str(n): rows for n, rows in pmi_results.items()},
            "bpe_merges": bpe_merges,
            "compression_after_20_merges": round(compression, 4),
        },
        "transformer_efficiency": teff,
        "optics_tension": tension,
    }

    out_path = OUT_DIR / "morpheme_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
