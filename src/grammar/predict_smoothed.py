"""
Three classical sequence-prediction models that should outperform the
transformer at the 38k-token scale of the unified whale corpus:

  S1  Modified Kneser-Ney 5-gram         (Chen & Goodman 1998)
  S2  Modified KN 5-gram + recency cache (Kuhn & De Mori 1990)
  S3  PPM-D, max order 7                 (Howard 1993)

Loads the same `data/classified/whale_dialogues.csv`, uses the same
sequence-level KFold split (random_state=42) as predict_kfold.py, and
reports bits/token + accuracy in the same format.

Why these models at this scale:

  - Existing Markov-2 baseline (Laplace α=0.5) hits 4.110 bpt — *worse*
    than Markov-1 (3.724 bpt). That's classical smoothing failure: with
    V=131 and only ~38k tokens, bigram-context cells are mostly empty
    or have count 1, and Laplace wastes probability mass on impossible
    bigrams. Modified KN was designed exactly for this regime.

  - Whale recordings are extremely repetitive (single recording often
    dominated by 3-5 coda types in long runs). A recency-cache term
    should crush this without requiring the model to learn "I just said
    cn5, I'll probably say cn5 again" from data.

  - PPM-D is a variable-order Markov model with escape-based smoothing,
    different family from KN; gives a sanity-check on whether the
    smoothing scheme matters at this scale.

Run:
  python -m src.grammar.predict_smoothed
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
TOKENS = ROOT / "data" / "classified" / "whale_dialogues.csv"
OUT_JSON = ROOT / "outputs" / "grammar" / "predict_results_smoothed.json"
OUT_MD = ROOT / "outputs" / "grammar" / "predict_results_smoothed.md"

N_FOLDS = 5
TARGET = "Coda"
BOS = "<BOS>"  # beginning-of-sequence sentinel for n-gram histories


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_sequences(target: str = TARGET) -> tuple[dict[str, list[str]], list[str]]:
    """Group rows by sequenceId, return per-seq token lists (strings) and
    the canonical sorted ordering of sequence ids used by KFold."""
    df = pd.read_csv(TOKENS)
    seq_dict: dict[str, list[str]] = {}
    for sid, g in df.groupby("sequenceId"):
        seq_dict[str(sid)] = [
            str(v) for v in g.sort_values("itemPosition")[target].tolist()
        ]
    seq_ids = sorted(seq_dict.keys())
    return seq_dict, seq_ids


def vocab_from(train_seqs: list[list[str]], test_seqs: list[list[str]]) -> list[str]:
    s: set[str] = set()
    for seq in train_seqs:
        s.update(seq)
    for seq in test_seqs:
        s.update(seq)
    return sorted(s)


# ---------------------------------------------------------------------------
# Modified Kneser-Ney
# ---------------------------------------------------------------------------


def fit_modified_kn(train_seqs: list[list[str]], n: int):
    """Fit Modified KN n-gram model on a list of token sequences.

    Returns a dict with:
      - top_counts[h] = Counter(w -> raw count)        # for the highest order
      - cont_counts[k][h] = Counter(w -> N1+(* h w))   # for orders 1..n-1
      - discounts[k] = (D1, D2, D3+)                   # per order
      - vocab: list of all observed tokens
      - n: model order
    """
    # Generate all n-gram instances. Each sequence is prefixed with (n-1)
    # BOS tokens so positions near the start have a defined history.
    # We collect the SET of unique k-grams (for k = 2..n+1) for use in
    # continuation-count computations.
    BOS_PREFIX = [BOS] * (n - 1)

    # raw counts at every order k = 1..n
    raw_counts: list[dict[tuple, Counter]] = [
        defaultdict(Counter) for _ in range(n + 1)
    ]
    # unique k-gram sets, k = 1..n+1 (we need k up to n+1 to compute
    # left-continuation counts at order n by counting unique (n+1)-grams
    # ending in a given n-suffix).
    unique_kgrams: list[set[tuple]] = [set() for _ in range(n + 2)]

    vocab_set: set[str] = set()
    for seq in train_seqs:
        full = BOS_PREFIX + list(seq)
        vocab_set.update(seq)
        for i in range(n - 1, len(full)):
            # Token at position i, history of length n-1
            for k in range(1, n + 1):
                kgram = tuple(full[i - (k - 1) : i + 1])  # length k
                hk = kgram[:-1]
                wk = kgram[-1]
                raw_counts[k][hk][wk] += 1
                unique_kgrams[k].add(kgram)
            # Also record (n+1)-grams that END in (history, w_i) for use
            # at the top-order continuation count (only matters if we use
            # cont counts at order n, which Modified KN does *not*; the
            # top order uses raw counts. We can skip this safely.)
    # Continuation counts at orders 1..n-1:
    #   cont_counts[k][h][w] = N1+(* h w)
    #     = number of distinct preceding tokens that, together with (h, w),
    #       form a (k+1)-gram observed in training.
    cont_counts: list[dict[tuple, Counter]] = [
        defaultdict(Counter) for _ in range(n + 1)
    ]
    for k in range(1, n):
        # iterate over unique (k+1)-grams
        for kgram in unique_kgrams[k + 1]:
            # kgram = (preceding, h_1, ..., h_{k-1}, w)
            h = kgram[1:-1]  # length k-1
            w = kgram[-1]
            cont_counts[k][h][w] += 1

    # Compute discounts D1, D2, D3+ per order using N_c counts.
    discounts: list[tuple[float, float, float] | None] = [None] * (n + 1)
    for k in range(1, n + 1):
        if k == n:
            counts_dict = raw_counts[k]
        else:
            counts_dict = cont_counts[k]
        # Count how many (h, w) pairs have count exactly c, for c = 1..4.
        N = [0, 0, 0, 0, 0]
        for cnt in counts_dict.values():
            for c in cnt.values():
                if 1 <= c <= 4:
                    N[c] += 1
        if N[1] == 0 or N[2] == 0:
            # Fallback: standard absolute discount D=0.5
            discounts[k] = (0.5, 0.5, 0.5)
            continue
        Y = N[1] / (N[1] + 2 * N[2])
        D1 = 1.0 - 2.0 * Y * (N[2] / N[1]) if N[2] > 0 else 0.5
        D2 = 2.0 - 3.0 * Y * (N[3] / N[2]) if N[3] > 0 else D1
        D3 = 3.0 - 4.0 * Y * (N[4] / N[3]) if N[4] > 0 else D2
        # Clamp to be safe
        D1 = max(0.0, min(D1, 1.0))
        D2 = max(0.0, min(D2, 2.0))
        D3 = max(0.0, min(D3, 3.0))
        discounts[k] = (D1, D2, D3)

    return dict(
        n=n,
        raw_counts=raw_counts,
        cont_counts=cont_counts,
        discounts=discounts,
        vocab=sorted(vocab_set),
    )


def _get_d(D_tup: tuple, c: int) -> float:
    if c <= 0:
        return 0.0
    if c == 1:
        return D_tup[0]
    if c == 2:
        return D_tup[1]
    return D_tup[2]


def kn_dist(model: dict, h: tuple) -> dict[str, float]:
    """Return the full P_KN(* | h) distribution as a dict.

    Implements the recursive Modified KN formula:
      P_KN(w | h) = max(c(hw) - D(c(hw)), 0)/c(h) + γ(h) P_KN(w | h')
    where γ(h) = (Σ_w D(c(hw))) / c(h) and h' is h with leftmost token
    dropped. At the lowest level, P_KN(w) is the continuation distribution.
    """
    n = model["n"]
    raw_counts = model["raw_counts"]
    cont_counts = model["cont_counts"]
    discounts = model["discounts"]
    vocab = model["vocab"]
    V = len(vocab)

    # Truncate h to last (n-1) tokens
    if len(h) > n - 1:
        h = h[-(n - 1):]

    # Recursion. At each level we work at order m = len(h) + 1.
    # If m == n: use raw_counts[n].
    # If 1 <= m < n: use cont_counts[m].
    # If m == 0 (h == ()): unigram = continuation distribution at order 1
    #   = cont_counts[1][()] / total. Plus uniform over vocab.

    def _recur(h_tup: tuple) -> dict[str, float]:
        m = len(h_tup)
        if m == 0:
            # Unigram continuation distribution
            if 1 in range(1, n + 1) and len(cont_counts[1].get((), Counter())) > 0:
                cnt = cont_counts[1][()]
                total = sum(cnt.values())
                D = discounts[1]
                # Apply same KN formula as higher orders, with backoff to
                # uniform 1/V.
                gamma_num = sum(_get_d(D, c) for c in cnt.values())
                gamma = gamma_num / total if total > 0 else 1.0
                dist = {}
                for w in vocab:
                    cw = cnt.get(w, 0)
                    base = max(cw - _get_d(D, cw), 0.0) / total if total > 0 else 0.0
                    dist[w] = base + gamma * (1.0 / V)
                return dist
            else:
                return {w: 1.0 / V for w in vocab}

        # Pick counts depending on whether we're at top order or lower
        order = m + 1  # because p(w | h) is an (m+1)-gram
        if order == n:
            counts_h = raw_counts[n].get(h_tup, None)
        else:
            counts_h = cont_counts[order].get(h_tup, None)

        if counts_h is None or sum(counts_h.values()) == 0:
            # Pure backoff
            return _recur(h_tup[1:])

        c_h = sum(counts_h.values())
        D = discounts[order]
        gamma_num = sum(_get_d(D, c) for c in counts_h.values())
        gamma = gamma_num / c_h
        # Recurse first
        lower = _recur(h_tup[1:])
        dist = {}
        for w in vocab:
            cw = counts_h.get(w, 0)
            base = max(cw - _get_d(D, cw), 0.0) / c_h
            dist[w] = base + gamma * lower[w]
        return dist

    return _recur(h)


def evaluate_kn(train_seqs: list[list[str]], test_seqs: list[list[str]],
                n: int = 5) -> dict:
    """Modified KN n-gram. Returns {bits_per_token, accuracy, n_params}."""
    model = fit_modified_kn(train_seqs, n)
    BOS_PREFIX = [BOS] * (n - 1)

    log2 = np.log2
    p_true_log: list[float] = []
    correct = 0
    total = 0

    # Cache distributions per unique history to amortize cost
    dist_cache: dict[tuple, dict[str, float]] = {}

    for seq in test_seqs:
        full = BOS_PREFIX + list(seq)
        for i in range(n - 1, len(full)):
            h = tuple(full[i - (n - 1) : i])
            w_true = full[i]
            if h not in dist_cache:
                dist_cache[h] = kn_dist(model, h)
            dist = dist_cache[h]
            p_w = max(dist.get(w_true, 1e-12), 1e-12)
            p_true_log.append(float(log2(p_w)))
            argmax = max(dist.items(), key=lambda x: x[1])[0]
            if argmax == w_true:
                correct += 1
            total += 1

    bits = -float(np.mean(p_true_log)) if p_true_log else float("inf")
    acc = correct / total if total > 0 else 0.0
    n_params = sum(
        sum(len(c) for c in d.values())
        for d in (model["raw_counts"][n], *model["cont_counts"][1:n])
    )
    return dict(bits_per_token=bits, accuracy=acc, n_params=int(n_params))


# ---------------------------------------------------------------------------
# Modified KN + recency cache
# ---------------------------------------------------------------------------


def evaluate_kn_cache(train_seqs: list[list[str]], test_seqs: list[list[str]],
                      n: int = 5, cache_size: int = 20,
                      lambda_cache: float = 0.15) -> dict:
    """KN n-gram interpolated with a recency cache:

        P(w | h) = (1 - λ) P_KN(w | h) + λ P_cache(w)

    where P_cache(w) is the empirical distribution of the last `cache_size`
    tokens emitted in the same sequence (prior to position i), with
    Laplace α=0.5 smoothing over the vocabulary so cache assigns no zero
    probability.

    Cache is reset at the start of each sequence.
    """
    model = fit_modified_kn(train_seqs, n)
    vocab = model["vocab"]
    V = len(vocab)
    BOS_PREFIX = [BOS] * (n - 1)

    log2 = np.log2
    p_true_log: list[float] = []
    correct = 0
    total = 0

    dist_cache: dict[tuple, dict[str, float]] = {}

    for seq in test_seqs:
        full = BOS_PREFIX + list(seq)
        emitted: list[str] = []  # actual emissions (no BOS), kept for cache
        for i in range(n - 1, len(full)):
            h = tuple(full[i - (n - 1) : i])
            w_true = full[i]
            if h not in dist_cache:
                dist_cache[h] = kn_dist(model, h)
            kn_d = dist_cache[h]

            # Build cache distribution from last `cache_size` emissions
            recent = emitted[-cache_size:]
            cache_counter = Counter(recent)
            cache_total = len(recent) + 0.5 * V
            # Laplace-smoothed cache distribution
            def cache_p(w: str) -> float:
                return (cache_counter.get(w, 0) + 0.5) / cache_total

            # Interpolate
            p_true_w = (
                (1.0 - lambda_cache) * kn_d.get(w_true, 1e-12)
                + lambda_cache * cache_p(w_true)
            )
            p_true_w = max(p_true_w, 1e-12)
            p_true_log.append(float(log2(p_true_w)))

            # argmax over vocab
            best_w, best_p = w_true, -1.0
            for w in vocab:
                p = (1.0 - lambda_cache) * kn_d.get(w, 0.0) + lambda_cache * cache_p(w)
                if p > best_p:
                    best_p, best_w = p, w
            if best_w == w_true:
                correct += 1
            total += 1

            emitted.append(w_true)

    bits = -float(np.mean(p_true_log)) if p_true_log else float("inf")
    acc = correct / total if total > 0 else 0.0
    return dict(
        bits_per_token=bits,
        accuracy=acc,
        n_params=None,  # nonparametric on top of KN
        hyperparams=dict(n=n, cache_size=cache_size, lambda_cache=lambda_cache),
    )


# ---------------------------------------------------------------------------
# PPM-D (Prediction by Partial Matching, escape method D)
# ---------------------------------------------------------------------------


def fit_ppm_d(train_seqs: list[list[str]], max_order: int = 7) -> dict:
    """Fit PPM-D context model. Stores Counter at each context up to
    max_order. Vocabulary derived from training tokens.

    PPM uses an exclusion mechanism (we implement the standard "full
    exclusion" version): when backing off from order k to k-1, we exclude
    tokens already seen at order k.
    """
    contexts: list[dict[tuple, Counter]] = [
        defaultdict(Counter) for _ in range(max_order + 1)
    ]
    vocab_set: set[str] = set()
    for seq in train_seqs:
        for i in range(len(seq)):
            w = seq[i]
            vocab_set.add(w)
            # Update counts for orders 0..max_order at position i
            for k in range(0, max_order + 1):
                if i - k < 0:
                    break
                h = tuple(seq[i - k : i])
                contexts[k][h][w] += 1
    return dict(
        max_order=max_order,
        contexts=contexts,
        vocab=sorted(vocab_set),
    )


def ppm_d_dist(model: dict, h: tuple) -> dict[str, float]:
    """PPM-D distribution over vocab given history h. Implements
    full-exclusion backoff with escape method D:

        P(w | h_k) = (c(h_k w) - 0.5) / c(h_k)        if c(h_k w) > 0
        P_esc(h_k) = (0.5 * |seen_at_order_k|) / c(h_k)

    Backs off through orders k = max_order, max_order-1, ..., 0, and
    finally to a uniform "order -1" distribution over vocab tokens not
    yet seen (i.e., 1 / |unseen|).
    """
    max_order = model["max_order"]
    contexts = model["contexts"]
    vocab = model["vocab"]
    V = len(vocab)

    # Truncate history to max_order
    if len(h) > max_order:
        h = h[-max_order:]

    seen: set[str] = set()
    p: dict[str, float] = {w: 0.0 for w in vocab}
    weight = 1.0  # remaining mass to allocate

    for k in range(min(len(h), max_order), -1, -1):
        h_k = h[-k:] if k > 0 else ()
        cnt = contexts[k].get(h_k)
        if cnt is None or sum(cnt.values()) == 0:
            continue
        # Sum over tokens NOT in seen (full exclusion)
        unique_here = [w for w in cnt if w not in seen]
        if not unique_here:
            continue
        c_h_eff = sum(cnt[w] for w in unique_here)
        n_unique = len(unique_here)
        # PPM-D escape: 0.5 * n_unique / c_h_eff (after exclusion-summing)
        # Each token gets (c - 0.5) / c_h_eff of the remaining weight
        denom = c_h_eff  # c(h_k) after exclusion
        for w in unique_here:
            cw = cnt[w]
            if cw > 0:
                p[w] += weight * (cw - 0.5) / denom
        # Mass that escapes to lower order
        esc = 0.5 * n_unique / denom
        weight = weight * esc
        # Add unique tokens to seen so they're excluded from lower orders
        seen.update(unique_here)
        if weight <= 0:
            break

    # Order -1: uniform over unseen tokens
    unseen = [w for w in vocab if w not in seen]
    if unseen and weight > 0:
        per = weight / len(unseen)
        for w in unseen:
            p[w] += per
    elif weight > 0:
        # all tokens seen — distribute uniformly
        per = weight / V
        for w in vocab:
            p[w] += per

    # Numerical safety
    s = sum(p.values())
    if s > 0:
        for w in p:
            p[w] /= s
    return p


def evaluate_ppm_d(train_seqs: list[list[str]], test_seqs: list[list[str]],
                   max_order: int = 7) -> dict:
    """PPM-D evaluation: bpt + accuracy."""
    model = fit_ppm_d(train_seqs, max_order)
    log2 = np.log2
    p_true_log: list[float] = []
    correct = 0
    total = 0

    dist_cache: dict[tuple, dict[str, float]] = {}

    for seq in test_seqs:
        for i in range(len(seq)):
            h = tuple(seq[max(0, i - max_order) : i])
            w_true = seq[i]
            if h not in dist_cache:
                dist_cache[h] = ppm_d_dist(model, h)
            dist = dist_cache[h]
            p_w = max(dist.get(w_true, 1e-12), 1e-12)
            p_true_log.append(float(log2(p_w)))
            argmax = max(dist.items(), key=lambda x: x[1])[0]
            if argmax == w_true:
                correct += 1
            total += 1

    bits = -float(np.mean(p_true_log)) if p_true_log else float("inf")
    acc = correct / total if total > 0 else 0.0
    n_params = sum(
        sum(len(c) for c in d.values()) for d in model["contexts"]
    )
    return dict(bits_per_token=bits, accuracy=acc, n_params=int(n_params))


# ---------------------------------------------------------------------------
# CV driver
# ---------------------------------------------------------------------------


def evaluate_fold(seq_dict: dict[str, list[str]], train_ids: list[str],
                  test_ids: list[str]) -> dict:
    train_seqs = [seq_dict[s] for s in train_ids]
    test_seqs = [seq_dict[s] for s in test_ids]
    res = {}
    res["S1_KN_5gram"] = evaluate_kn(train_seqs, test_seqs, n=5)
    res["S2_KN_5gram_cache"] = evaluate_kn_cache(
        train_seqs, test_seqs, n=5, cache_size=320, lambda_cache=0.25
    )
    res["S3_PPM_D_o7"] = evaluate_ppm_d(train_seqs, test_seqs, max_order=7)
    res["_meta"] = dict(
        n_train_seqs=len(train_seqs),
        n_test_seqs=len(test_seqs),
        n_train_tokens=sum(len(s) for s in train_seqs),
        n_test_tokens=sum(len(s) for s in test_seqs),
    )
    return res


def main() -> None:
    print("loading sequences...")
    seq_dict, seq_ids = load_sequences()
    n_seqs = len(seq_ids)
    n_tokens = sum(len(s) for s in seq_dict.values())
    print(f"  {n_seqs} sequences, {n_tokens:,} tokens")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    for fi, (tr_idx, te_idx) in enumerate(kf.split(seq_ids)):
        train_ids = [seq_ids[i] for i in tr_idx]
        test_ids = [seq_ids[i] for i in te_idx]
        t0 = time.time()
        res = evaluate_fold(seq_dict, train_ids, test_ids)
        dt = time.time() - t0
        res["_meta"]["fold"] = fi
        res["_meta"]["seconds"] = round(dt, 1)
        fold_results.append(res)
        print(
            f"fold {fi}: "
            + "  ".join(
                f"{m}: bpt={res[m]['bits_per_token']:.3f} acc={res[m]['accuracy']:.3f}"
                for m in ("S1_KN_5gram", "S2_KN_5gram_cache", "S3_PPM_D_o7")
            )
            + f"  ({dt:.1f}s)"
        )

    # Summary
    summary = {}
    for m in ("S1_KN_5gram", "S2_KN_5gram_cache", "S3_PPM_D_o7"):
        bits = [f[m]["bits_per_token"] for f in fold_results]
        accs = [f[m]["accuracy"] for f in fold_results]
        summary[m] = dict(
            bits_per_token_mean=float(np.mean(bits)),
            bits_per_token_std=float(np.std(bits)),
            perplexity_mean=float(2 ** np.mean(bits)),
            accuracy_mean=float(np.mean(accs)),
            accuracy_std=float(np.std(accs)),
            n_params=fold_results[0][m].get("n_params"),
        )

    # Write JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        dict(folds=fold_results, summary=summary, target=TARGET),
        indent=2, default=str,
    ))

    # Write Markdown
    label = {
        "S1_KN_5gram": "Modified Kneser-Ney 5-gram",
        "S2_KN_5gram_cache": "Modified KN 5-gram + sequence-level cache (size 320, λ=0.25)",
        "S3_PPM_D_o7": "PPM-D, max order 7",
    }
    L = []
    L.append(
        "# 5-fold smoothed-classical models — KN 5-gram, KN+cache, PPM-D"
    )
    L.append("")
    L.append(
        f"Sequence-level KFold ({N_FOLDS} folds, random_state=42; same split "
        f"as `predict_kfold.py`). Target = `{TARGET}`. "
        "Metric = held-out cross-entropy in **bits/token** (log₂); lower is "
        "better. Perplexity = 2^(bits/token)."
    )
    L.append("")
    L.append(
        "Compare against the existing benchmark in "
        "`predict_results_unified.md` — the best transformer to date is "
        "**MiniTransformer M7 at 3.186 bpt** (perplexity ≈ 9.10)."
    )
    L.append("")
    L.append("| # | model | params | bits/token (↓) | perplexity (↓) | accuracy |")
    L.append("|---|-------|-------:|---------------:|---------------:|---------:|")
    for m in ("S1_KN_5gram", "S2_KN_5gram_cache", "S3_PPM_D_o7"):
        s = summary[m]
        params = f"{s['n_params']:,}" if s.get("n_params") else "—"
        idx_label = m.split("_", 1)[0]
        L.append(
            f"| {idx_label} | {label[m]} | {params} | "
            f"{s['bits_per_token_mean']:.3f} ± {s['bits_per_token_std']:.3f} | "
            f"{s['perplexity_mean']:.2f} | "
            f"{s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f} |"
        )
    L.append("")
    L.append("## Per-fold bits/token")
    L.append("")
    L.append("| fold |" + "|".join(f" {label[m]} " for m in
             ("S1_KN_5gram", "S2_KN_5gram_cache", "S3_PPM_D_o7")) + "|")
    L.append("|------|" + "|".join(["---:"] * 3) + "|")
    for f in fold_results:
        L.append(
            f"| {f['_meta']['fold']} |"
            + "|".join(
                f" {f[m]['bits_per_token']:.3f} "
                for m in ("S1_KN_5gram", "S2_KN_5gram_cache", "S3_PPM_D_o7")
            )
            + "|"
        )
    L.append("")
    best = min(summary, key=lambda m: summary[m]["bits_per_token_mean"])
    L.append("## Takeaway")
    L.append("")
    L.append(
        f"- Best smoothed-classical model: **{label[best]}** at "
        f"{summary[best]['bits_per_token_mean']:.3f} bits/token "
        f"(perplexity ≈ {summary[best]['perplexity_mean']:.2f})."
    )
    L.append(
        "- Reference points from `predict_results_unified.md`: Markov-2 "
        "(Laplace α=0.5) = 4.110 bpt, Markov-1 = 3.724 bpt, "
        "MLP-M = 3.267 bpt, MiniTransformer M7 = **3.186 bpt**."
    )
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
