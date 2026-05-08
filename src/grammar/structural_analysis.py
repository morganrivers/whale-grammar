"""
Structural analysis of whale coda sequences — no transformer required.

Analyses:
  1. Zipf rank-frequency: compound tokens vs Morfessor morphemes vs Mandarin syllables
  2. MI decay: power-law vs exponential fit, whale vs shuffled vs Mandarin
  3. Re-Pair grammar compression depth
  4. Pause-context asymmetry (JSD by Δt bucket)

Run from repo root:
    python -m src.grammar.structural_analysis

Outputs → outputs/grammar/structural_analysis/
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "grammar" / "structural_analysis"
OUT.mkdir(parents=True, exist_ok=True)

# Δt bucket edges for whale (from dt_buckets.py)
_DT_EDGES = [0.42, 1.19, 2.37, 3.98]
_DT_LABELS = {
    0: "missing",
    1: "Δt<0.42s",
    2: "Δt~0.8s",
    3: "Δt~1.8s",
    4: "Δt~3.2s",
    5: "Δt>4.0s",
    6: "speaker\nswitch",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_compound_sequences(whale_csv: Path) -> tuple[dict[str, list[int]], Counter]:
    df = pd.read_csv(whale_csv)
    df["TempoBin"] = (
        pd.qcut(df["Duration"], q=5, labels=False, duplicates="drop")
        .astype(int)
    )
    keys = list(zip(
        df["Coda"].astype(int),
        df["TempoBin"].astype(int),
        df["Ornamentation"].astype(int),
        df["Synchrony"].astype(int),
    ))
    codes, _ = pd.factorize(pd.Series(keys), sort=True)
    df["CompoundId"] = codes.astype(int)

    seqs: dict[str, list[int]] = {}
    for sid, grp in df.groupby("sequenceId"):
        seqs[sid] = grp.sort_values("itemPosition")["CompoundId"].tolist()

    flat = [t for s in seqs.values() for t in s]
    return seqs, Counter(flat)


def load_compound_df(whale_csv: Path) -> pd.DataFrame:
    """Load whale_dialogues.csv with compound IDs, DT buckets, and speaker."""
    df = pd.read_csv(whale_csv)
    df["TempoBin"] = (
        pd.qcut(df["Duration"], q=5, labels=False, duplicates="drop")
        .astype(int)
    )
    keys = list(zip(
        df["Coda"].astype(int),
        df["TempoBin"].astype(int),
        df["Ornamentation"].astype(int),
        df["Synchrony"].astype(int),
    ))
    codes, _ = pd.factorize(pd.Series(keys), sort=True)
    df["CompoundId"] = codes.astype(int)

    df = df.sort_values(["sequenceId", "itemPosition"])
    df["PrevWhale"] = df.groupby("sequenceId")["Whale"].shift(1)
    df["Switched"] = (df["Whale"] != df["PrevWhale"]) & df["PrevWhale"].notna()

    def _bucket(row: pd.Series) -> int:
        if not row["has_timestamps"]:
            return 0
        dt = row["TimeDelta"]
        if pd.isna(dt) or dt < 0:
            return 0
        if row["Switched"]:
            return 6
        for i, edge in enumerate(_DT_EDGES):
            if dt <= edge:
                return i + 1
        return 5

    df["DTBucket"] = df.apply(_bucket, axis=1)
    return df


def load_morfessor_counts(json_path: Path, track: str = "track1b") -> Counter:
    """Load per-morpheme frequency from morpheme_results.json.

    track = 'track1'  → ICI morphemes only (e.g. 'AAAA')
    track = 'track1b' → tempo-augmented (e.g. 'AAAAP')
    """
    key_map = {
        "track1": "track1_ici_morphemes",
        "track1b": "track1b_augmented_morphemes",
    }
    with open(json_path) as f:
        d = json.load(f)
    inventory = d[key_map[track]]["morpheme_inventory"]
    return Counter({item[0]: item[1] for item in inventory})


def load_mandarin_sequences(
    csv: Path, index: Path
) -> tuple[dict[str, list[str]], Counter]:
    df = pd.read_csv(csv)
    idx = pd.read_csv(index)
    id2word = {int(r.word_id): str(r.word) for r in idx.itertuples()}
    df["Syllable"] = df["Coda"].map(lambda c: id2word.get(int(c), f"<{c}>"))

    seqs: dict[str, list[str]] = {}
    for sid, grp in df.groupby("sequenceId"):
        seqs[sid] = grp.sort_values("itemPosition")["Syllable"].tolist()

    flat = [t for s in seqs.values() for t in s]
    return seqs, Counter(flat)


# ---------------------------------------------------------------------------
# 1. Zipf analysis
# ---------------------------------------------------------------------------

def _zipf_fit(counts: Counter, label: str) -> dict:
    freqs = np.array(sorted(counts.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freqs) + 1, dtype=float)
    slope, intercept, r, _, _ = linregress(np.log10(ranks), np.log10(freqs))
    return {
        "label": label,
        "ranks": ranks,
        "freqs": freqs,
        "slope": slope,
        "intercept": intercept,
        "r2": r**2,
        "N": int(freqs.sum()),
        "V": len(freqs),
    }


def plot_zipf(
    fits: list[dict],
    out_path: Path,
) -> None:
    COLORS = ["#2166ac", "#b2182b", "#1b7837"]
    MARKERS = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for fit, color, marker in zip(fits, COLORS, MARKERS):
        ax.loglog(
            fit["ranks"], fit["freqs"],
            marker, alpha=0.35, ms=3, color=color, rasterized=True,
        )
        r_fit = np.array([1.0, float(fit["V"])])
        f_fit = 10 ** (fit["intercept"] + fit["slope"] * np.log10(r_fit))
        ax.loglog(
            r_fit, f_fit, "-", color=color, lw=2.2,
            label=(
                f"{fit['label']}\n"
                f"  α = {abs(fit['slope']):.2f}   "
                f"R² = {fit['r2']:.3f}   V = {fit['V']:,}"
            ),
        )

    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Zipf rank–frequency distributions", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ---------------------------------------------------------------------------
# 2. MI decay
# ---------------------------------------------------------------------------

def _to_int_seqs(seqs: dict[str, list]) -> tuple[dict[str, list[int]], int]:
    all_tok = sorted({t for s in seqs.values() for t in s})
    vocab = {t: i for i, t in enumerate(all_tok)}
    return {k: [vocab[t] for t in v] for k, v in seqs.items()}, len(vocab)


def compute_mi_decay(
    seqs: dict[str, list[int]],
    max_lag: int = 20,
) -> np.ndarray:
    """Empirical MI(X_i, X_{i+k}) for k = 1 … max_lag, within sequences only."""
    joint: list[Counter] = [Counter() for _ in range(max_lag)]

    for seq in seqs.values():
        n = len(seq)
        for i in range(n):
            for ki, k in enumerate(range(1, min(max_lag + 1, n - i))):
                joint[ki][(seq[i], seq[i + k])] += 1

    mi_vals = np.zeros(max_lag)
    for ki in range(max_lag):
        N = sum(joint[ki].values())
        if N == 0:
            continue
        # Marginals from the lag-k joint (correct for non-stationary series)
        marg_left: Counter = Counter()
        marg_right: Counter = Counter()
        for (a, b), c in joint[ki].items():
            marg_left[a] += c
            marg_right[b] += c
        mi = 0.0
        for (a, b), c in joint[ki].items():
            p_ab = c / N
            p_a = marg_left[a] / N
            p_b = marg_right[b] / N
            if p_a > 0 and p_b > 0:
                mi += p_ab * math.log2(p_ab / (p_a * p_b))
        mi_vals[ki] = max(0.0, mi)
    return mi_vals


def shuffle_seqs(seqs: dict[str, list[int]], seed: int = 42) -> dict[str, list[int]]:
    rng = random.Random(seed)
    out = {}
    for k, v in seqs.items():
        s = list(v)
        rng.shuffle(s)
        out[k] = s
    return out


def _fit_decay(lags: np.ndarray, mi: np.ndarray) -> dict:
    mask = mi > 1e-12
    if mask.sum() < 4:
        return {}
    x, y = lags[mask], mi[mask]
    lx, ly = np.log(x.astype(float)), np.log(y.astype(float))

    b_pl, a_pl, r_pl, _, _ = linregress(lx, ly)
    sse_pl = np.sum((ly - (a_pl + b_pl * lx)) ** 2)
    aic_pl = len(x) * math.log(sse_pl / len(x)) + 4

    b_ex, a_ex, r_ex, _, _ = linregress(x.astype(float), ly)
    sse_ex = np.sum((ly - (a_ex + b_ex * x)) ** 2)
    aic_ex = len(x) * math.log(sse_ex / len(x)) + 4

    return {
        "power_law":  {"alpha": float(-b_pl), "r2": float(r_pl**2), "aic": float(aic_pl)},
        "exponential": {"lam":  float(-b_ex), "r2": float(r_ex**2), "aic": float(aic_ex)},
        "preferred": "power_law" if aic_pl < aic_ex else "exponential",
    }


def plot_mi_decay(
    results: dict[str, np.ndarray],
    fits: dict[str, dict],
    lags: np.ndarray,
    out_path: Path,
) -> None:
    COLORS = {
        "Whale compound": "#2166ac",
        "Whale compound (shuffled)": "#92c5de",
        "Mandarin syllables": "#b2182b",
    }
    STYLES = {
        "Whale compound": ("-", "o"),
        "Whale compound (shuffled)": ("--", "x"),
        "Mandarin syllables": ("-", "s"),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, mi in results.items():
        ls, marker = STYLES.get(label, ("-", "o"))
        color = COLORS.get(label, "gray")
        fit = fits.get(label, {})
        pref = fit.get("preferred", "")
        if pref == "power_law":
            detail = f"  power-law α={fit['power_law']['alpha']:.2f}"
        elif pref == "exponential":
            detail = f"  exp λ={fit['exponential']['lam']:.3f}"
        else:
            detail = ""
        ax.semilogy(
            lags, mi, ls + marker,
            color=color, lw=2, ms=5, markevery=2,
            label=f"{label}{detail}",
        )

    ax.set_xlabel("Lag k (tokens)", fontsize=12)
    ax.set_ylabel("MI(X_i, X_{i+k})  [bits]", fontsize=12)
    ax.set_title("Mutual-information decay by token lag", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ---------------------------------------------------------------------------
# 3. Re-Pair grammar compression
# ---------------------------------------------------------------------------

def repair_compress(seq: list[int], n_rounds: int = 80) -> list[float]:
    """
    Iterative digram replacement (simplified Re-Pair).
    Returns compression ratios (len / original) after each replacement round.
    Stops early when no digram appears ≥ 2 times.
    """
    seq = list(seq)
    orig = len(seq)
    next_sym = max(seq, default=0) + 1
    ratios: list[float] = []

    for _ in range(n_rounds):
        digrams: Counter = Counter()
        for i in range(len(seq) - 1):
            digrams[(seq[i], seq[i + 1])] += 1
        top = digrams.most_common(1)
        if not top or top[0][1] < 2:
            break
        (a, b), _ = top[0]
        new_seq: list[int] = []
        i = 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                new_seq.append(next_sym)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        seq = new_seq
        next_sym += 1
        ratios.append(len(seq) / orig)

    return ratios


def plot_repair(
    curves: dict[str, list[float]],
    out_path: Path,
) -> None:
    COLORS = {
        "Whale compound": "#2166ac",
        "Whale compound (shuffled)": "#92c5de",
        "Mandarin syllables": "#b2182b",
    }
    STYLES = {
        "Whale compound": "-",
        "Whale compound (shuffled)": "--",
        "Mandarin syllables": "-",
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, ratios in curves.items():
        if not ratios:
            continue
        ax.plot(
            range(1, len(ratios) + 1), ratios,
            STYLES.get(label, "-"),
            color=COLORS.get(label, "gray"),
            lw=2,
            label=f"{label}  (final={ratios[-1]:.3f})",
        )
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Re-Pair rounds", fontsize=12)
    ax.set_ylabel("Sequence length / original", fontsize=12)
    ax.set_title(
        "Re-Pair grammar compression\n(deeper compression = more repeated structure)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ---------------------------------------------------------------------------
# 4. Pause-context asymmetry (JSD)
# ---------------------------------------------------------------------------

def _jsd(p: Counter, q: Counter, smooth: float = 1e-8) -> float:
    keys = set(p) | set(q)
    tp = sum(p.values()) + smooth * len(keys)
    tq = sum(q.values()) + smooth * len(keys)
    jsd = 0.0
    for k in keys:
        pk = (p.get(k, 0) + smooth) / tp
        qk = (q.get(k, 0) + smooth) / tq
        mk = (pk + qk) / 2
        if pk > 0:
            jsd += pk / 2 * math.log2(pk / mk)
        if qk > 0:
            jsd += qk / 2 * math.log2(qk / mk)
    return jsd


def pause_context_asymmetry(df: pd.DataFrame) -> tuple[list[int], list[float], list[int]]:
    """
    For each Δt bucket, compute JSD between the left-context token distribution
    (token immediately before the pause) and the right-context distribution
    (token immediately after).

    Returns (bucket_ids, jsd_scores, observation_counts).
    """
    left: dict[int, Counter] = defaultdict(Counter)
    right: dict[int, Counter] = defaultdict(Counter)

    for _, grp in df.groupby("sequenceId"):
        grp = grp.sort_values("itemPosition").reset_index(drop=True)
        toks = grp["CompoundId"].tolist()
        bkts = grp["DTBucket"].tolist()
        for i in range(1, len(toks)):
            b = bkts[i]
            left[b][toks[i - 1]] += 1
            right[b][toks[i]] += 1

    buckets = sorted(b for b in left if sum(left[b].values()) >= 20)
    jsds = [_jsd(left[b], right[b]) for b in buckets]
    ns = [sum(left[b].values()) for b in buckets]
    return buckets, jsds, ns


def plot_pause_asymmetry(
    buckets: list[int],
    jsds: list[float],
    ns: list[int],
    out_path: Path,
) -> None:
    CMAP = ["#d9d9d9", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026", "#762a83"]
    colors = [CMAP[min(b, len(CMAP) - 1)] for b in buckets]
    labels = [_DT_LABELS.get(b, f"bucket {b}") for b in buckets]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(buckets)), jsds, color=colors, edgecolor="black", lw=0.8)
    for bar, n in zip(bars, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(jsds) * 0.01,
            f"n={n:,}",
            ha="center", va="bottom", fontsize=7.5,
        )
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("JSD (left-context vs right-context)", fontsize=11)
    ax.set_title(
        "Pause-context asymmetry by Δt bucket\n"
        "High JSD → what precedes this pause ≠ what follows it",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ---------------------------------------------------------------------------
# Summary panel (4-in-1)
# ---------------------------------------------------------------------------

def plot_summary_panel(
    zipf_fits: list[dict],
    mi_results: dict[str, np.ndarray],
    mi_fits: dict[str, dict],
    lags: np.ndarray,
    repair_curves: dict[str, list[float]],
    pause_buckets: list[int],
    pause_jsds: list[float],
    pause_ns: list[int],
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)

    COLORS_MAIN = ["#2166ac", "#b2182b", "#1b7837"]
    COLORS_MI = {
        "Whale compound": "#2166ac",
        "Whale compound (shuffled)": "#92c5de",
        "Mandarin syllables": "#b2182b",
    }
    STYLES_MI = {
        "Whale compound": ("-", "o"),
        "Whale compound (shuffled)": ("--", "x"),
        "Mandarin syllables": ("-", "s"),
    }
    COLORS_REPAIR = {
        "Whale compound": "#2166ac",
        "Whale compound (shuffled)": "#92c5de",
        "Mandarin syllables": "#b2182b",
    }
    CMAP_PAUSE = ["#d9d9d9","#fee090","#fdae61","#f46d43","#d73027","#a50026","#762a83"]

    # ── Panel A: Zipf ────────────────────────────────────────────────────────
    ax_z = fig.add_subplot(gs[0, 0])
    for fit, color, marker in zip(zipf_fits, COLORS_MAIN, ["o", "s", "^"]):
        ax_z.loglog(fit["ranks"], fit["freqs"], marker, alpha=0.3, ms=3, color=color, rasterized=True)
        r2 = np.array([1.0, float(fit["V"])])
        f2 = 10 ** (fit["intercept"] + fit["slope"] * np.log10(r2))
        ax_z.loglog(r2, f2, "-", color=color, lw=2,
                    label=f"{fit['label']}  α={abs(fit['slope']):.2f}")
    ax_z.set_xlabel("Rank", fontsize=10)
    ax_z.set_ylabel("Frequency", fontsize=10)
    ax_z.set_title("A  Zipf distributions", fontsize=11, fontweight="bold")
    ax_z.legend(fontsize=8, framealpha=0.85)
    ax_z.grid(True, which="both", alpha=0.2)

    # ── Panel B: MI decay ────────────────────────────────────────────────────
    ax_mi = fig.add_subplot(gs[0, 1])
    for label, mi in mi_results.items():
        ls, mkr = STYLES_MI.get(label, ("-", "o"))
        fit = mi_fits.get(label, {})
        pref = fit.get("preferred", "")
        detail = ""
        if pref == "power_law":
            detail = f"  α={fit['power_law']['alpha']:.2f}"
        elif pref == "exponential":
            detail = f"  λ={fit['exponential']['lam']:.3f}"
        ax_mi.semilogy(lags, mi, ls + mkr,
                       color=COLORS_MI.get(label, "gray"),
                       lw=2, ms=4, markevery=2,
                       label=f"{label}{detail}")
    ax_mi.set_xlabel("Lag k (tokens)", fontsize=10)
    ax_mi.set_ylabel("MI  [bits]", fontsize=10)
    ax_mi.set_title("B  Mutual-information decay", fontsize=11, fontweight="bold")
    ax_mi.legend(fontsize=8, framealpha=0.85)
    ax_mi.grid(True, alpha=0.2)

    # ── Panel C: Re-Pair ─────────────────────────────────────────────────────
    ax_rp = fig.add_subplot(gs[1, 0])
    STYLES_RP = {"Whale compound": "-", "Whale compound (shuffled)": "--", "Mandarin syllables": "-"}
    for label, ratios in repair_curves.items():
        if not ratios:
            continue
        ax_rp.plot(range(1, len(ratios)+1), ratios,
                   STYLES_RP.get(label, "-"),
                   color=COLORS_REPAIR.get(label, "gray"),
                   lw=2, label=f"{label}  ({ratios[-1]:.3f})")
    ax_rp.axhline(1.0, color="gray", ls=":", lw=1)
    ax_rp.set_xlabel("Re-Pair rounds", fontsize=10)
    ax_rp.set_ylabel("Length / original", fontsize=10)
    ax_rp.set_title("C  Re-Pair grammar compression", fontsize=11, fontweight="bold")
    ax_rp.legend(fontsize=8, framealpha=0.85)
    ax_rp.grid(True, alpha=0.2)

    # ── Panel D: Pause asymmetry ─────────────────────────────────────────────
    ax_pa = fig.add_subplot(gs[1, 1])
    colors_p = [CMAP_PAUSE[min(b, len(CMAP_PAUSE)-1)] for b in pause_buckets]
    labels_p = [_DT_LABELS.get(b, f"b{b}") for b in pause_buckets]
    bars = ax_pa.bar(range(len(pause_buckets)), pause_jsds,
                     color=colors_p, edgecolor="black", lw=0.7)
    for bar, n in zip(bars, pause_ns):
        ax_pa.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + max(pause_jsds)*0.015,
                   f"n={n:,}", ha="center", va="bottom", fontsize=6.5)
    ax_pa.set_xticks(range(len(pause_buckets)))
    ax_pa.set_xticklabels(labels_p, fontsize=8)
    ax_pa.set_ylabel("JSD (left vs right context)", fontsize=10)
    ax_pa.set_title("D  Pause-context asymmetry", fontsize=11, fontweight="bold")
    ax_pa.grid(True, axis="y", alpha=0.2)

    fig.suptitle(
        "Whale coda sequence structure — no transformer required",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    whale_csv = ROOT / "data" / "classified" / "whale_dialogues.csv"
    multilang_csv = ROOT / "data" / "classified" / "multilang_dialogues.csv"
    multilang_idx = ROOT / "data" / "classified" / "multilang_word_index.csv"
    morph_json = ROOT / "outputs" / "morphemes" / "morpheme_results.json"

    # ── Load ─────────────────────────────────────────────────────────────────
    print("Loading compound whale sequences …")
    whale_seqs, whale_counts = load_compound_sequences(whale_csv)

    print("Loading Morfessor morpheme counts (track1b, tempo-augmented) …")
    morph_counts_1b = load_morfessor_counts(morph_json, track="track1b")
    print("Loading Morfessor morpheme counts (track1, ICI-only) …")
    morph_counts_1 = load_morfessor_counts(morph_json, track="track1")

    print("Loading Mandarin syllable sequences …")
    mand_seqs, mand_counts = load_mandarin_sequences(multilang_csv, multilang_idx)

    # ── 1. Zipf ───────────────────────────────────────────────────────────────
    print("\n[1] Zipf analysis")
    fit_compound = _zipf_fit(whale_counts, "Whale compound (Sharma)")
    fit_morph1b  = _zipf_fit(morph_counts_1b, "Whale Morfessor+tempo (track1b)")
    fit_morph1   = _zipf_fit(morph_counts_1, "Whale Morfessor ICI (track1)")
    fit_mand     = _zipf_fit(mand_counts, "Mandarin CHILDES syllables")

    for f in [fit_compound, fit_morph1, fit_morph1b, fit_mand]:
        print(f"  {f['label']}: V={f['V']:,}  N={f['N']:,}  α={abs(f['slope']):.3f}  R²={f['r2']:.3f}")

    # primary Zipf plot: compound + both Morfessor tracks + Mandarin
    plot_zipf([fit_compound, fit_morph1b, fit_mand], OUT / "zipf_comparison.png")

    # secondary: all four on one plot
    plot_zipf([fit_compound, fit_morph1, fit_morph1b, fit_mand],
              OUT / "zipf_all_four.png")

    # ── 2. MI decay ───────────────────────────────────────────────────────────
    print("\n[2] MI decay")
    MAX_LAG = 20
    lags = np.arange(1, MAX_LAG + 1)

    whale_int, _ = _to_int_seqs(whale_seqs)
    whale_shuf   = shuffle_seqs(whale_int)
    mand_int, _  = _to_int_seqs(mand_seqs)

    print("  whale compound …")
    mi_whale = compute_mi_decay(whale_int, MAX_LAG)
    print("  whale compound (shuffled) …")
    mi_shuf = compute_mi_decay(whale_shuf, MAX_LAG)
    print("  Mandarin …")
    mi_mand = compute_mi_decay(mand_int, MAX_LAG)

    mi_results = {
        "Whale compound": mi_whale,
        "Whale compound (shuffled)": mi_shuf,
        "Mandarin syllables": mi_mand,
    }
    mi_fits = {lbl: _fit_decay(lags, mi) for lbl, mi in mi_results.items()}

    for lbl, fit in mi_fits.items():
        if fit:
            pl = fit["power_law"]
            ex = fit["exponential"]
            print(
                f"  {lbl}  preferred={fit['preferred']}"
                f"  PL α={pl['alpha']:.3f} R²={pl['r2']:.3f} AIC={pl['aic']:.1f}"
                f"  Exp λ={ex['lam']:.4f} R²={ex['r2']:.3f} AIC={ex['aic']:.1f}"
            )

    plot_mi_decay(mi_results, mi_fits, lags, OUT / "mi_decay.png")

    # ── 3. Re-Pair ────────────────────────────────────────────────────────────
    print("\n[3] Re-Pair grammar compression")
    flat_whale = [t for s in whale_int.values() for t in s]
    flat_shuf  = [t for s in whale_shuf.values() for t in s]
    flat_mand  = [t for s in mand_int.values() for t in s]

    print("  whale compound …")
    rp_whale = repair_compress(flat_whale, n_rounds=80)
    print("  whale (shuffled) …")
    rp_shuf = repair_compress(flat_shuf, n_rounds=80)
    print("  Mandarin …")
    rp_mand = repair_compress(flat_mand, n_rounds=80)

    repair_curves = {
        "Whale compound": rp_whale,
        "Whale compound (shuffled)": rp_shuf,
        "Mandarin syllables": rp_mand,
    }
    for lbl, r in repair_curves.items():
        if r:
            print(f"  {lbl}: {len(r)} rounds, final ratio={r[-1]:.4f}")

    plot_repair(repair_curves, OUT / "repair_compression.png")

    # ── 4. Pause asymmetry ────────────────────────────────────────────────────
    print("\n[4] Pause-context asymmetry")
    wdf = load_compound_df(whale_csv)
    p_buckets, p_jsds, p_ns = pause_context_asymmetry(wdf)
    for b, j, n in zip(p_buckets, p_jsds, p_ns):
        print(f"  {_DT_LABELS.get(b, b):20s}  JSD={j:.4f}  n={n:,}")
    plot_pause_asymmetry(p_buckets, p_jsds, p_ns, OUT / "pause_asymmetry.png")

    # ── Summary panel ─────────────────────────────────────────────────────────
    print("\n[5] Summary 4-panel figure")
    plot_summary_panel(
        [fit_compound, fit_morph1b, fit_mand],
        mi_results, mi_fits, lags,
        repair_curves,
        p_buckets, p_jsds, p_ns,
        OUT / "summary_panel.png",
    )

    # ── Numeric summary ───────────────────────────────────────────────────────
    def _safe(d, *keys, default=None):
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    summary = {
        "zipf": {
            "whale_compound":  {"V": fit_compound["V"], "N": fit_compound["N"], "alpha": round(abs(fit_compound["slope"]), 4), "R2": round(fit_compound["r2"], 4)},
            "morfessor_track1b": {"V": fit_morph1b["V"], "N": fit_morph1b["N"], "alpha": round(abs(fit_morph1b["slope"]), 4), "R2": round(fit_morph1b["r2"], 4)},
            "morfessor_track1": {"V": fit_morph1["V"], "N": fit_morph1["N"], "alpha": round(abs(fit_morph1["slope"]), 4), "R2": round(fit_morph1["r2"], 4)},
            "mandarin_syllables": {"V": fit_mand["V"], "N": fit_mand["N"], "alpha": round(abs(fit_mand["slope"]), 4), "R2": round(fit_mand["r2"], 4)},
        },
        "mi_decay": {
            lbl: {
                "preferred": fit.get("preferred"),
                "power_law_alpha": round(_safe(fit, "power_law", "alpha", default=float("nan")), 4),
                "power_law_R2":    round(_safe(fit, "power_law", "r2",    default=float("nan")), 4),
                "exponential_lam": round(_safe(fit, "exponential", "lam", default=float("nan")), 4),
                "exponential_R2":  round(_safe(fit, "exponential", "r2",  default=float("nan")), 4),
            }
            for lbl, fit in mi_fits.items() if fit
        },
        "repair": {
            lbl: {"rounds": len(r), "final_ratio": round(r[-1], 4) if r else None}
            for lbl, r in repair_curves.items()
        },
        "pause_asymmetry": {
            _DT_LABELS.get(b, str(b)): {"jsd": round(j, 4), "n": n}
            for b, j, n in zip(p_buckets, p_jsds, p_ns)
        },
    }
    summary_path = OUT / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary → {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
