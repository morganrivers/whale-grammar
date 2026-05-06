"""
Whale-specific decoding of the model-side findings from
`interp_lexical.py`.

Two views, joined against the per-row metadata in
``data/classified/whale_dialogues.csv``
(``Whale`` = speaker, ``Duration``, ``TimeDelta``, ``Synchrony``,
``Ornamentation``):

1. **Coda-family clusters from W_U.** Connected components in the
   undirected graph where ``(a, b)`` is an edge iff ``cosine(W_U[a],
   W_U[b]) ≥ τ`` and both tokens are in each other's top-``k`` cosine
   neighborhoods. Only tokens with corpus count ≥ ``MIN_COUNT`` enter
   the graph. For each cluster of size ≥ 2 we report n samples, mean
   Duration, mean TimeDelta-to-prev, dominant speakers, and the
   Synchrony / Ornamentation rates.

2. **Mutual attention pairs decoded.** Re-build the (V, V)
   query-token-conditioned attention co-occurrence matrix averaged
   across all (layer, head). Take the top mutual pairs by
   ``√(A[a,b] · A[b,a])``, then for each pair scan the corpus for
   in-K=8-window co-occurrences and compute:

     n            number of (i, j) pairs in any K-window with
                  ``tokens[i] ∈ {a,b}, tokens[j] ∈ {a,b}, i < j``
     cross_spk    fraction of those pairs where the speaker label
                  changes between i and j (call-and-response signature)
     mean_dt      mean cumulative timing gap from i to j in seconds
                  (only over pairs where both endpoints have
                  has_timestamps=1)
     a→b / b→a    directional split among same pairs

Run::

    python -m src.grammar.interp_whale_decoded
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.grammar.m7_hooked import N_CTX, make_m7_hooked

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "outputs" / "grammar" / "checkpoints" / "whale"
DATA_CSV = ROOT / "data" / "classified" / "whale_dialogues.csv"
OUT_PATH = ROOT / "outputs" / "grammar" / "interp" / "whale" / "decoded.md"

# ---- thresholds ----
MIN_COUNT = 30        # tokens must occur ≥ this many times to enter the graph
WU_TAU = 0.45         # cosine threshold for an edge in the W_U graph
WU_TOPK = 6           # mutual top-k requirement
COOC_WINDOWS = 4000
COOC_MIN_QCOUNT = 30
TOP_PAIRS = 25
SUBSAMPLE_SEED = 42


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_whale():
    cfg = json.loads((CKPT_DIR / "config.json").read_text())
    vocab = json.loads((CKPT_DIR / "vocab.json").read_text())
    model = make_m7_hooked(d_vocab=cfg["d_vocab"])
    state = torch.load(CKPT_DIR / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    df = pd.read_csv(DATA_CSV)
    df = df.sort_values(["sequenceId", "itemPosition"]).reset_index(drop=True)
    return model, cfg, vocab, df


# ---------------------------------------------------------------------------
# View 1 — W_U cluster decoding
# ---------------------------------------------------------------------------


def cosine_matrix(W: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
    Wn = W / n
    return Wn @ Wn.T


def mutual_topk_clusters(
    cos: np.ndarray, eligible: np.ndarray, top_k: int, tau: float,
) -> list[list[int]]:
    V = cos.shape[0]
    edges: dict[int, set[int]] = {int(t): set() for t in eligible}
    # for each eligible token, find top-k cosine neighbors among eligible
    elig_set = set(int(t) for t in eligible)
    cos_eligible = cos.copy()
    mask = np.ones(V, dtype=bool)
    mask[eligible] = False
    cos_eligible[:, mask] = -np.inf
    np.fill_diagonal(cos_eligible, -np.inf)
    topk_neigh: dict[int, set[int]] = {}
    for t in eligible:
        idx = np.argpartition(-cos_eligible[t], top_k)[:top_k]
        idx = idx[np.argsort(-cos_eligible[t, idx])]
        topk_neigh[int(t)] = set(int(i) for i in idx if cos[t, i] >= tau)
    # mutual edges
    for a in eligible:
        a = int(a)
        for b in topk_neigh[a]:
            if a in topk_neigh.get(b, set()):
                edges[a].add(b)
                edges[b].add(a)
    # connected components
    seen: set[int] = set()
    components: list[list[int]] = []
    for t in eligible:
        t = int(t)
        if t in seen:
            continue
        stack = [t]
        comp = []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            stack.extend(edges[u] - seen)
        if len(comp) >= 2:
            components.append(sorted(comp))
    components.sort(key=lambda c: -len(c))
    return components


def cluster_metadata(
    cluster: list[int], df: pd.DataFrame,
) -> dict:
    sub = df[df["Coda"].isin(cluster)]
    n = len(sub)
    if n == 0:
        return dict(n=0)
    dur = sub["Duration"]
    has_ts = sub["has_timestamps"] == 1
    dt_valid = sub.loc[has_ts, "TimeDelta"]
    spk_counts = sub["Whale"].value_counts()
    top_spk = [(s, int(c)) for s, c in spk_counts.head(3).items()]
    return dict(
        n=int(n),
        mean_duration=float(dur.mean()),
        std_duration=float(dur.std()) if len(dur) > 1 else 0.0,
        mean_timedelta=float(dt_valid.mean()) if len(dt_valid) else None,
        std_timedelta=float(dt_valid.std()) if len(dt_valid) > 1 else None,
        synchrony_rate=float((sub["Synchrony"] == 1).mean()),
        ornamentation_rate=float((sub["Ornamentation"] == 1).mean()),
        n_distinct_speakers=int(spk_counts.size),
        top_speakers=top_spk,
    )


# ---------------------------------------------------------------------------
# View 2 — attention co-occurrence (averaged across heads), top mutual pairs
# ---------------------------------------------------------------------------


def windows_from_df(df: pd.DataFrame, k: int = N_CTX) -> tuple[np.ndarray, list]:
    """Build all K-windows. Return (windows[N, K], window_meta) where each
    meta is a dict carrying the parallel per-position metadata.
    """
    win_tok: list[list[int]] = []
    win_meta: list[dict] = []
    for sid, g in df.groupby("sequenceId", sort=False):
        toks = g["Coda"].astype(int).to_numpy()
        if len(toks) < k:
            continue
        spk = g["Whale"].astype(str).to_numpy()
        td = g["TimeDelta"].astype(float).to_numpy()
        hts = g["has_timestamps"].astype(int).to_numpy()
        for i in range(len(toks) - k + 1):
            win_tok.append(toks[i : i + k].tolist())
            win_meta.append(dict(
                seq=str(sid), start=int(i),
                speakers=spk[i : i + k].tolist(),
                td=td[i : i + k].tolist(),
                hts=hts[i : i + k].tolist(),
            ))
    return np.array(win_tok, dtype=np.int64), win_meta


@torch.no_grad()
def attn_cooccurrence_avg(model, windows: np.ndarray, V: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (A_mean[V,V], q_count[V]) where A_mean is the average over all
    (layer, head) of the query-conditioned attention from token a to b."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    K = windows.shape[1]
    A = np.zeros((V, V), dtype=np.float32)
    q_count = np.zeros(V, dtype=np.int64)

    bs = 256
    for start in range(0, len(windows), bs):
        batch_np = windows[start : start + bs]
        batch = torch.from_numpy(batch_np).long()
        _, cache = model.run_with_cache(batch)
        B = batch.size(0)
        for q in range(K):
            np.add.at(q_count, batch_np[:, q], 1)
        # sum over (L, H) of pattern (B, q, k)
        patt_sum = sum(
            cache[f"blocks.{L}.attn.hook_pattern"].detach().numpy().sum(axis=1)
            for L in range(n_layers)
        )  # (B, q, k); summed over heads then we'll divide
        patt_avg = patt_sum / (n_layers * n_heads)
        for b in range(B):
            tok = batch_np[b]
            qt = tok[:, None].repeat(K, axis=1)
            kt = tok[None, :].repeat(K, axis=0)
            np.add.at(A, (qt, kt), patt_avg[b])
    safe = q_count.astype(np.float64).copy()
    safe[safe == 0] = 1.0
    return A / safe[:, None], q_count


def top_mutual_pairs(meanA: np.ndarray, q_count: np.ndarray,
                     min_qcount: int, top_n: int) -> list[tuple[int, int, float, float, float]]:
    keep = np.where(q_count >= min_qcount)[0]
    if keep.size < 2:
        return []
    sub = meanA[np.ix_(keep, keep)]
    mutual = np.sqrt(np.maximum(sub * sub.T, 0.0))
    n = sub.shape[0]
    iu = np.triu_indices(n, k=1)
    flat = mutual[iu]
    take = min(top_n, flat.size)
    if take == 0:
        return []
    idx = np.argpartition(-flat, take - 1)[:take]
    idx = idx[np.argsort(-flat[idx])]
    out = []
    for fi in idx:
        ai, bi = iu[0][fi], iu[1][fi]
        a, b = int(keep[ai]), int(keep[bi])
        out.append((a, b, float(sub[ai, bi]), float(sub[bi, ai]), float(flat[fi])))
    return out


def pair_corpus_stats(
    pair_a: int, pair_b: int, windows: np.ndarray, win_meta: list[dict],
) -> dict:
    """For a pair (a, b), scan all K-windows for (i, j) co-occurrences with
    ``i < j`` and ``{tokens[i], tokens[j]} == {a, b}`` (or both equal). For
    every co-occurrence, accumulate:

        - dir_ab / dir_ba counts (which token comes first)
        - speaker-switch rate
        - cumulative time-delta i→j (only when both endpoints have
          has_timestamps=1; sum windows[i+1:j+1] of TimeDelta).

    A given (i, j) pair within a window is only counted once across all
    windows that contain it (we record it under the first containing window
    by sliding-window de-dup).
    """
    K = windows.shape[1]
    seen_pairs: set[tuple[str, int, int]] = set()
    n_total = 0
    n_dir_ab = 0
    n_dir_ba = 0
    n_cross_spk = 0
    n_spk_eligible = 0
    dt_values: list[float] = []
    for w_idx, w in enumerate(windows):
        if pair_a not in w or pair_b not in w:
            continue
        meta = win_meta[w_idx]
        seq = meta["seq"]
        start = meta["start"]
        for i in range(K):
            for j in range(i + 1, K):
                ti, tj = int(w[i]), int(w[j])
                if {ti, tj} != {pair_a, pair_b}:
                    continue
                key = (seq, start + i, start + j)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                n_total += 1
                if ti == pair_a:
                    n_dir_ab += 1
                else:
                    n_dir_ba += 1
                # speaker switch
                spk_i, spk_j = meta["speakers"][i], meta["speakers"][j]
                if spk_i and spk_j and "UNK" not in spk_i and "UNK" not in spk_j:
                    n_spk_eligible += 1
                    if spk_i != spk_j:
                        n_cross_spk += 1
                # cumulative timing: sum td[i+1 .. j]
                hts = meta["hts"]
                td = meta["td"]
                if all(hts[k] == 1 for k in range(i, j + 1)):
                    dt_values.append(float(sum(td[i + 1 : j + 1])))
    return dict(
        n=n_total,
        dir_ab=n_dir_ab,
        dir_ba=n_dir_ba,
        cross_spk_frac=(n_cross_spk / n_spk_eligible) if n_spk_eligible else None,
        n_spk_eligible=n_spk_eligible,
        mean_dt_seconds=(sum(dt_values) / len(dt_values)) if dt_values else None,
        n_dt_eligible=len(dt_values),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def fmt_or_dash(v, fmt: str) -> str:
    return f"{v:{fmt}}" if v is not None else "—"


def write_report(
    components: list[list[int]],
    cluster_meta: list[dict],
    pairs: list[tuple[int, int, float, float, float]],
    pair_meta: list[dict],
    config: dict,
) -> None:
    md = ["# Whale lexical decoding", ""]
    md.append("Joins the model-side outputs from `interp_lexical.py` against "
              "rhythm and speaker metadata in `whale_dialogues.csv`.")
    md.append("")
    md.append("Config: "
              f"τ_cos={config['tau']}, top-k={config['topk']}, "
              f"min count={config['min_count']}, "
              f"K={config['n_ctx']}, "
              f"cooc windows={config['n_windows']}.")
    md.append("")

    md.append("## View 1 — Coda-family clusters from W_U")
    md.append("")
    md.append("Connected components in the mutual-top-k cosine neighbor graph "
              "on the model's output embedding W_U. A coda enters the graph "
              f"only if it occurs ≥ {config['min_count']} times in the corpus. "
              "Edges require both ends to be in each other's top-"
              f"{config['topk']} cosine neighbors and cosine ≥ {config['tau']}.")
    md.append("")
    md.append("| # | size | members | n samples | mean Duration (s) | "
              "mean Δt-to-prev (s) | sync rate | orn rate | "
              "n distinct speakers | top speakers (n) |")
    md.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for i, (comp, meta) in enumerate(zip(components, cluster_meta), 1):
        members = ", ".join(str(c) for c in comp)
        def _short(s: str) -> str:
            # keep the corpus prefix so hersh::UNK and sharma::UNK don't merge
            parts = s.split("::")
            if len(parts) >= 2:
                corpus = parts[0].split("_")[0]  # "hersh2022" -> "hersh"
                return f"{corpus}::{parts[-1]}"
            return s
        top_spk = "; ".join(
            f"`{_short(s)}` ({c})"
            for s, c in meta["top_speakers"]
        )
        md.append(
            f"| {i} | {len(comp)} | {members} | {meta['n']} | "
            f"{fmt_or_dash(meta['mean_duration'], '.3f')} | "
            f"{fmt_or_dash(meta['mean_timedelta'], '.2f')} | "
            f"{meta['synchrony_rate']:.3f} | "
            f"{meta['ornamentation_rate']:.3f} | "
            f"{meta['n_distinct_speakers']} | {top_spk} |"
        )
    md.append("")

    md.append("## View 2 — Mutual attention pairs decoded")
    md.append("")
    md.append("Top-mutual coda pairs from the across-(layer,head) average "
              "attention co-occurrence matrix, each scanned against the "
              "corpus for K=8 co-occurrences:")
    md.append("")
    md.append("- `mutual` is √(mean attn a→b · mean attn b→a) averaged "
              "across all 8 heads.")
    md.append("- `n` counts unique (i, j) coda-pair occurrences in any "
              "K=8 window where {tokens[i], tokens[j]} == {a, b} and i < j.")
    md.append("- `cross-speaker` is the fraction of those pairs where the "
              "speaker label changes between positions i and j (UNK-speaker "
              "rows excluded; n_eligible reported).")
    md.append("- `mean Δt` is the mean cumulative timing gap from i to j "
              "in seconds, computed only over pairs where both endpoints "
              "and all intervening rows have has_timestamps=1.")
    md.append("- `a→b` is the fraction of co-occurrences where token `a` "
              "comes first.")
    md.append("")
    md.append("| a | b | mutual | n | a→b | cross-speaker (n_elig) | "
              "mean Δt s (n_elig) |")
    md.append("|---:|---:|---:|---:|---:|---|---|")
    for (a, b, _ab, _ba, mut), pm in zip(pairs, pair_meta):
        n = pm["n"]
        ab_frac = (pm["dir_ab"] / n) if n else 0.0
        cross = (f"{pm['cross_spk_frac']:.2f} ({pm['n_spk_eligible']})"
                 if pm["cross_spk_frac"] is not None else "— (0)")
        dt = (f"{pm['mean_dt_seconds']:.2f} ({pm['n_dt_eligible']})"
              if pm["mean_dt_seconds"] is not None else "— (0)")
        md.append(f"| {a} | {b} | {mut:.3f} | {n} | "
                  f"{ab_frac:.2f} | {cross} | {dt} |")
    md.append("")
    md.append("## How to read")
    md.append("")
    md.append("**Clusters**: a cluster sharing both Duration and "
              "TimeDelta-to-prev statistics means the model has learned "
              "rhythm-class affinity. A cluster dominated by one or two "
              "speakers means the model has captured an individual's "
              "vocal repertoire. High Synchrony rate = group/chorusing "
              "codas; high Ornamentation rate = decorated variants.")
    md.append("")
    md.append("**Pairs**: high cross-speaker fraction is the conversational "
              "signature — these are codas that tend to bridge a speaker "
              "switch within K=8, i.e. *call-and-response*. Low "
              "cross-speaker fraction with short mean Δt is a "
              "*within-utterance rhythm motif* — codas that tend to be "
              "produced by the same whale in close sequence. Mean Δt "
              "of a few seconds with high cross-speaker fraction is "
              "the classic whale exchange pattern.")
    md.append("")

    OUT_PATH.write_text("\n".join(md))
    print(f"wrote {OUT_PATH}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--top-pairs", type=int, default=TOP_PAIRS)
    p.add_argument("--seed", type=int, default=SUBSAMPLE_SEED)
    args = p.parse_args(argv)

    print("loading whale checkpoint + corpus...")
    model, cfg, vocab, df = load_whale()
    V = model.cfg.d_vocab
    counts = df["Coda"].value_counts()
    counts_arr = np.zeros(V, dtype=np.int64)
    for tid, c in counts.items():
        counts_arr[int(tid)] = int(c)
    eligible = np.where(counts_arr >= MIN_COUNT)[0]
    print(f"V={V}; eligible (count >= {MIN_COUNT}): {eligible.size}")

    # ---- View 1 ----
    print("computing W_U cosine matrix and clusters...")
    W_U = model.W_U.detach().numpy().T  # (V, d)
    cos_U = cosine_matrix(W_U)
    components = mutual_topk_clusters(cos_U, eligible, top_k=WU_TOPK, tau=WU_TAU)
    print(f"  found {len(components)} clusters of size >= 2 "
          f"(largest = {len(components[0]) if components else 0})")
    cluster_meta = [cluster_metadata(c, df) for c in components]

    # ---- View 2 ----
    print("building K-windows and attention co-occurrence (avg across heads)...")
    windows, win_meta = windows_from_df(df, N_CTX)
    rng = np.random.default_rng(args.seed)
    if len(windows) > COOC_WINDOWS:
        idx = rng.choice(len(windows), COOC_WINDOWS, replace=False)
        cooc_windows = windows[idx]
        cooc_meta = [win_meta[i] for i in idx]
    else:
        cooc_windows = windows
        cooc_meta = win_meta
    print(f"  cooc windows: {len(cooc_windows)}")
    A_mean, q_count = attn_cooccurrence_avg(model, cooc_windows, V)
    pairs = top_mutual_pairs(A_mean, q_count, COOC_MIN_QCOUNT, args.top_pairs)
    print(f"  top mutual pairs: {len(pairs)}")

    print("scanning corpus for pair speaker-switch / timing stats...")
    # Use *all* corpus windows for pair stats (not just the cooc subsample)
    # so the cross-speaker / Δt fractions have more support.
    pair_meta = [pair_corpus_stats(a, b, windows, win_meta)
                 for a, b, *_ in pairs]

    write_report(
        components, cluster_meta, pairs, pair_meta,
        config=dict(tau=WU_TAU, topk=WU_TOPK, min_count=MIN_COUNT,
                    n_ctx=N_CTX, n_windows=len(cooc_windows)),
    )


if __name__ == "__main__":
    main()
