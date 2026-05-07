"""
ICI-timing benchmark: evaluate morpheme vs compound model predictions
in raw (unbucketed) timing space.

Both models predict an ICI string (bucketed alphabet A-D). We decode
that prediction to timing estimates via per-bin empirical means (computed
from training data per fold), then compare against the true raw ICI values.

Metrics (all lower = better unless noted):
  n_click_err     — |pred_n_clicks − true_n_clicks|
  nn_dist         — mean nearest-neighbour distance (pred click positions →
                    true click positions), normalised by true coda duration
  scale_free_rmse — RMSE of ICI sequences after normalising each to unit sum
                    (zero-padded to matched length); scale-invariant
  dtw_norm        — DTW path cost on ICI sequences, normalised by max(n, m)
  pattern_corr    — Pearson r of normalised ICI sequences truncated to
                    min(len_pred, len_true) length  (↑ higher = better)

Usage:
    python -m src.grammar.evaluate_ici_timing
    python -m src.grammar.evaluate_ici_timing --smoke
    python -m src.grammar.evaluate_ici_timing --no-hersh
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold

from src.grammar.predict_full import (
    HERSH_PREFIXES, CLEAN_PREFIXES,
    WhaleTfm, load_corpus, build_model, build_windows, train_scheme,
)
from src.grammar.whale_morpheme_tokeniser import (
    _load_joined, _load_morfessor, load_morpheme_seq_whale,
)
from src.grammar.morpheme_discovery import (
    compute_global_bin_edges, ici_row_to_string,
    MIN_CLICKS, MAX_CLICKS, N_BINS, LABELS,
)
from src.grammar.evaluate_ici_fidelity import (
    _tier, build_compound_windows_meta,
    MorphemeTfm, _build_morph_windows, train_morph_model,
    load_compound_ckpt, train_compound_model,
    _build_p_rhythm, _compound_map_ici,
    build_ici_lookup,
)

ROOT     = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "outputs" / "grammar" / "checkpoints"
OUT_DIR  = ROOT / "outputs" / "grammar"
K        = 8
SEED     = 42
VAL_FRAC = 0.1


# ── raw ICI loading ───────────────────────────────────────────────────────────

def load_raw_ici_map() -> tuple[dict[tuple[str, int], np.ndarray], dict]:
    """Return ({(seq_id, item_pos): raw_ici_array}, edges).

    raw_ici_array contains the true float ICI values in seconds for each
    inter-click gap within the coda (length = n_clicks - 1).
    """
    joined = _load_joined()
    valid_mask = (
        joined["n_clicks"].notna()
        & joined["n_clicks"].between(MIN_CLICKS, MAX_CLICKS)
    )
    edges = compute_global_bin_edges(joined[valid_mask].copy(), n_bins=N_BINS)
    raw_map: dict[tuple, np.ndarray] = {}
    for _, row in joined.iterrows():
        sid = str(row["sequenceId"])
        pos = int(row["itemPosition"])
        n = row.get("n_clicks")
        try:
            n_i = int(float(n))
        except (TypeError, ValueError):
            continue
        if not (MIN_CLICKS <= n_i <= MAX_CLICKS):
            continue
        vals = []
        for p in range(1, n_i):
            col = f"ICI{p}"
            v = row.get(col, np.nan)
            try:
                f = float(v)
            except (TypeError, ValueError):
                f = np.nan
            if np.isnan(f):
                break
            vals.append(f)
        if len(vals) == n_i - 1:
            raw_map[(sid, pos)] = np.array(vals, dtype=float)
    return raw_map, edges


# ── decode bucketed string to timing estimate ─────────────────────────────────

def compute_bin_means(
    raw_ici_map: dict,
    ici_map_str: dict,
    train_ids: set,
    edges: dict,
) -> dict:
    """Per-(ICI-col, bin-label) empirical mean from training rows; falls back to midpoint."""
    from collections import defaultdict
    buckets: dict = defaultdict(list)
    for (sid, pos), raw_vals in raw_ici_map.items():
        if sid not in train_ids:
            continue
        ici_str = ici_map_str.get((sid, pos))
        if not ici_str:
            continue
        for i, (sym, val) in enumerate(zip(ici_str, raw_vals), start=1):
            buckets[(f"ICI{i}", sym)].append(val)
    label_list = list(LABELS)
    result: dict = {}
    for col, e in edges.items():
        for idx, sym in enumerate(label_list):
            if idx >= len(e) - 1:
                break
            key = (col, sym)
            result[key] = float(np.mean(buckets[key])) if buckets[key] else (e[idx] + e[idx + 1]) / 2.0
    return result


def string_to_timings(ici_str: str, edges: dict, bin_means: dict | None = None) -> np.ndarray:
    """Decode a bucketed ICI string to float ICI values (seconds).

    Uses per-bin empirical means when bin_means is provided, else bucket midpoints.
    """
    result = []
    label_list = list(LABELS)
    for p, sym in enumerate(ici_str, start=1):
        col = f"ICI{p}"
        if col not in edges:
            result.append(0.25)  # fallback: rough overall median
            continue
        e = edges[col]
        idx = label_list.index(sym) if sym in label_list else len(label_list) // 2
        idx = max(0, min(idx, len(e) - 2))
        if bin_means is not None:
            result.append(bin_means.get((col, sym), (e[idx] + e[idx + 1]) / 2.0))
        else:
            result.append((e[idx] + e[idx + 1]) / 2.0)
    return np.array(result, dtype=float) if result else np.zeros(0)


def click_positions(ici_vals: np.ndarray) -> np.ndarray:
    """Cumulative click times; first click anchored at 0."""
    if len(ici_vals) == 0:
        return np.array([0.0])
    return np.concatenate([[0.0], np.cumsum(ici_vals)])


# ── timing metrics ────────────────────────────────────────────────────────────

def timing_metrics(pred_str: str, true_raw: np.ndarray, edges: dict, bin_means: dict | None = None) -> dict:
    """Five timing-space metrics between a predicted ICI string and true raw ICIs."""
    pred_ici = string_to_timings(pred_str, edges, bin_means)
    true_ici = true_raw

    # 1. click count error
    n_click_err = abs(len(pred_ici) - len(true_ici))

    # 2. mean nearest-neighbour distance (normalised by true coda duration)
    pred_pos = click_positions(pred_ici)
    true_pos = click_positions(true_ici)
    true_dur = float(true_pos[-1]) if len(true_pos) > 1 else 1.0
    nn_dists = [min(abs(p - t) for t in true_pos) for p in pred_pos]
    nn_dist = float(np.mean(nn_dists)) / max(true_dur, 1e-6)

    # 3. scale-free RMSE (normalise each sequence to unit sum, then pad)
    if len(pred_ici) == 0 or len(true_ici) == 0:
        scale_free_rmse = 1.0
    else:
        p_norm = pred_ici / (pred_ici.sum() + 1e-9)
        t_norm = true_ici / (true_ici.sum() + 1e-9)
        max_len = max(len(p_norm), len(t_norm))
        p_pad = np.pad(p_norm, (0, max_len - len(p_norm)))
        t_pad = np.pad(t_norm, (0, max_len - len(t_norm)))
        scale_free_rmse = float(np.sqrt(np.mean((p_pad - t_pad) ** 2)))

    # 4. DTW distance on raw ICI sequences, normalised by max(n, m)
    dtw_norm = _dtw(pred_ici, true_ici)

    # 5. Pearson r of normalised ICI patterns (scale-free; ↑ better)
    pattern_corr = _pattern_corr(pred_ici, true_ici)

    return dict(
        n_click_err=n_click_err,
        nn_dist=nn_dist,
        scale_free_rmse=scale_free_rmse,
        dtw_norm=dtw_norm,
        pattern_corr=pattern_corr,
    )


def _dtw(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float(abs(n - m))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m]) / max(n, m)


def _pattern_corr(a: np.ndarray, b: np.ndarray) -> float:
    min_len = min(len(a), len(b))
    if min_len < 2:
        return 0.0
    a_n = a[:min_len] / (a[:min_len].sum() + 1e-9)
    b_n = b[:min_len] / (b[:min_len].sum() + 1e-9)
    corr = float(np.corrcoef(a_n, b_n)[0, 1])
    return 0.0 if math.isnan(corr) else corr


def _agg_timing(rows: list[dict]) -> dict:
    keys = ["n_click_err", "nn_dist", "scale_free_rmse", "dtw_norm", "pattern_corr"]
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        out[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[f"{k}_std"]  = float(np.std(vals))  if vals else float("nan")
    return out


# ── compound model eval ───────────────────────────────────────────────────────

def eval_compound_timing(
    model: WhaleTfm,
    test_windows: dict,
    test_meta: list[tuple[str, int]],
    token_to_coda_arr: np.ndarray,
    ici_lookup: dict,
    raw_ici_map: dict,
    edges: dict,
    V_coda: int,
    bin_means: dict | None = None,
    bs: int = 512,
) -> tuple[dict, set[tuple[str, int]]]:
    model.eval()
    n_wins = test_windows["token"].size(0)
    if n_wins == 0:
        return {}, set()

    k_val = test_windows["token"].size(1) - 1
    x_all = test_windows["token"][:, :k_val]
    dt_log = test_windows["dt_log"][:, :k_val]
    has_ts = test_windows["has_ts"][:, :k_val].float()
    dt_input = torch.stack([dt_log, has_ts], dim=-1)

    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, n_wins, bs):
            out = model(x_all[i:i+bs], dt_input=dt_input[i:i+bs])
            probs = torch.softmax(out["token"][:, -1], dim=-1).cpu().numpy()
            all_probs.extend(probs)

    per_sample: list[dict] = []
    valid_positions: set[tuple[str, int]] = set()

    for (sid, pos), probs in zip(test_meta, all_probs):
        true_raw = raw_ici_map.get((sid, pos))
        if true_raw is None:
            continue
        pred_str = _compound_map_ici(probs, token_to_coda_arr, ici_lookup, V_coda)
        if not pred_str:
            continue
        m = timing_metrics(pred_str, true_raw, edges, bin_means)
        per_sample.append(m)
        valid_positions.add((sid, pos))

    agg = _agg_timing(per_sample)
    agg["n_valid"] = len(per_sample)
    return agg, valid_positions


# ── Markov-1 compound eval ────────────────────────────────────────────────────

def eval_markov1_compound_timing(
    per_seq: dict,
    train_ids: list[str], test_ids: list[str],
    valid_positions: set[tuple[str, int]],
    raw_ici_map: dict,
    edges: dict,
    ici_lookup: dict,
    token_to_coda_arr: np.ndarray,
    V_coda: int, V_token: int,
    alpha: float = 0.5,
    bin_means: dict | None = None,
) -> dict:
    trans: dict[int, Counter] = {}
    unigram: Counter = Counter()
    for sid in train_ids:
        seq = per_seq.get(sid)
        if seq is None:
            continue
        toks = seq["token"]
        for t in toks:
            unigram[t] += 1
        for i in range(1, len(toks)):
            trans.setdefault(toks[i - 1], Counter())[toks[i]] += 1

    n_train = sum(unigram.values())
    unigram_arr = np.array([unigram.get(t, 0) for t in range(V_token)], dtype=float)

    per_sample: list[dict] = []
    for sid in test_ids:
        seq = per_seq.get(sid)
        if seq is None:
            continue
        toks = seq["token"]
        for pos in range(1, len(toks)):
            if (sid, pos) not in valid_positions:
                continue
            true_raw = raw_ici_map.get((sid, pos))
            if true_raw is None:
                continue
            ctx_tok = toks[pos - 1]
            c = trans.get(ctx_tok)
            if c is None:
                tok_probs = (unigram_arr + alpha) / (n_train + alpha * V_token)
            else:
                c_arr = np.array([c.get(t, 0) for t in range(V_token)], dtype=float)
                tok_probs = (c_arr + alpha) / (c_arr.sum() + alpha * V_token)
            pred_str = _compound_map_ici(tok_probs, token_to_coda_arr, ici_lookup, V_coda)
            if not pred_str:
                continue
            per_sample.append(timing_metrics(pred_str, true_raw, edges, bin_means))

    agg = _agg_timing(per_sample)
    agg["n_valid"] = len(per_sample)
    return agg


# ── morpheme model eval ───────────────────────────────────────────────────────

def eval_morpheme_timing(
    model: MorphemeTfm,
    morph_conv_dict: dict,
    test_ids: list[str],
    valid_positions: set[tuple[str, int]],
    raw_ici_map: dict,
    edges: dict,
    morph_decoder: dict,
    ici_map_str: dict,
    segs_cache: dict,
    k: int = K,
    bs: int = 512,
    bin_means: dict | None = None,
) -> dict:
    model.eval()
    target_set = set(valid_positions)
    per_sample: list[dict] = []

    for sid in test_ids:
        mc = morph_conv_dict.get(sid)
        if mc is None:
            continue
        bd = mc.coda_boundaries
        for ci, coda_start in enumerate(bd):
            if (sid, ci) not in target_set:
                continue
            true_raw = raw_ici_map.get((sid, ci))
            if true_raw is None:
                continue
            true_ici_str = ici_map_str.get((sid, ci))
            # n_steps = number of morphemes in the true segmentation
            n_steps = len(segs_cache.get(true_ici_str, [true_ici_str])) if true_ici_str else 1
            ctx_toks = mc.tokens[max(0, coda_start - k): coda_start]
            ctx = [0] * (k - len(ctx_toks)) + list(ctx_toks)
            pred_strs: list[str] = []
            cur_ctx = list(ctx)
            with torch.no_grad():
                for _ in range(n_steps):
                    x = torch.tensor([cur_ctx], dtype=torch.long)
                    pred_id = int(model(x).argmax(-1).item())
                    pred_strs.append(morph_decoder.get(pred_id, ""))
                    cur_ctx = cur_ctx[1:] + [pred_id]
            pred_str = "".join(pred_strs)
            if not pred_str:
                continue
            per_sample.append(timing_metrics(pred_str, true_raw, edges, bin_means))

    agg = _agg_timing(per_sample)
    agg["n_valid"] = len(per_sample)
    return agg


# ── Markov-1 morpheme eval ────────────────────────────────────────────────────

def eval_markov1_morpheme_timing(
    morph_conv_dict: dict,
    train_ids: list[str], test_ids: list[str],
    valid_positions: set[tuple[str, int]],
    raw_ici_map: dict,
    edges: dict,
    morph_decoder: dict,
    ici_map_str: dict,
    segs_cache: dict,
    V_morph: int,
    alpha: float = 0.5,
    bin_means: dict | None = None,
) -> dict:
    trans: dict[int, Counter] = {}
    unigram: Counter = Counter()
    for sid in train_ids:
        mc = morph_conv_dict.get(sid)
        if mc is None:
            continue
        for t in mc.tokens:
            unigram[t] += 1
        for i in range(1, len(mc.tokens)):
            trans.setdefault(mc.tokens[i - 1], Counter())[mc.tokens[i]] += 1

    per_sample: list[dict] = []
    for sid in test_ids:
        mc = morph_conv_dict.get(sid)
        if mc is None:
            continue
        bd = mc.coda_boundaries
        for ci, coda_start in enumerate(bd):
            if (sid, ci) not in valid_positions:
                continue
            true_raw = raw_ici_map.get((sid, ci))
            if true_raw is None:
                continue
            true_ici_str = ici_map_str.get((sid, ci))
            n_steps = len(segs_cache.get(true_ici_str, [true_ici_str])) if true_ici_str else 1
            prev = mc.tokens[coda_start - 1] if coda_start > 0 else 0
            pred_strs: list[str] = []
            for _ in range(n_steps):
                c = trans.get(prev)
                if c is None:
                    pred_id = unigram.most_common(1)[0][0] if unigram else 0
                else:
                    pred_id = c.most_common(1)[0][0]
                pred_strs.append(morph_decoder.get(pred_id, ""))
                prev = pred_id
            pred_str = "".join(pred_strs)
            if not pred_str:
                continue
            per_sample.append(timing_metrics(pred_str, true_raw, edges, bin_means))

    agg = _agg_timing(per_sample)
    agg["n_valid"] = len(per_sample)
    return agg


# ── fold driver ───────────────────────────────────────────────────────────────

def run_fold(
    fold_i: int,
    hersh_ids: list[str], clean_train_ids: list[str],
    val_ids: list[str], test_ids: list[str],
    per_seq: dict, V_coda: int, V_token: int, V_dt: int,
    token_to_coda: torch.Tensor, token_to_coda_arr: np.ndarray,
    raw_ici_map: dict, edges: dict,
    ici_map_str: dict, segs_cache: dict,
    morph_conv_dict: dict, morph_decoder: dict,
    V_morph: int,
    args: argparse.Namespace,
) -> dict:
    all_train_ids = hersh_ids + clean_train_ids

    ici_lookup, _ = build_ici_lookup(per_seq, ici_map_str, all_train_ids, token_to_coda)
    bin_means = compute_bin_means(raw_ici_map, ici_map_str, set(all_train_ids), edges)

    fold_out: dict = {}
    test_windows, test_meta = build_compound_windows_meta(per_seq, test_ids, K)

    # ── compound model ────────────────────────────────────────────────────
    if args.no_hersh:
        print(f"  training compound WhaleTfm on clean-only ({len(clean_train_ids)} seqs) …")
        comp_model = train_compound_model(
            per_seq, clean_train_ids, val_ids, V_coda, V_token, V_dt,
            epochs=args.compound_epochs, patience=args.compound_patience,
            lr=args.compound_lr, bs=args.bs,
        )
    else:
        print(f"  loading compound K=8 checkpoint fold {fold_i} …")
        comp_model = load_compound_ckpt(fold_i)

    t0 = time.time()
    comp_agg, valid_pos = eval_compound_timing(
        comp_model, test_windows, test_meta,
        token_to_coda_arr, ici_lookup, raw_ici_map, edges, V_coda, bin_means,
    )
    comp_agg["seconds"] = round(time.time() - t0, 1)
    fold_out["compound_tfm_k8"] = comp_agg
    print(f"    compound_tfm_k8: nn_dist={comp_agg.get('nn_dist_mean', float('nan')):.4f}  "
          f"n_click_err={comp_agg.get('n_click_err_mean', float('nan')):.2f}  "
          f"n={comp_agg['n_valid']}  ({comp_agg['seconds']}s)")
    del comp_model

    # ── morpheme model (load checkpoint if available) ─────────────────────
    suffix = "_no_hersh" if args.no_hersh else ""
    morph_ckpt_path = CKPT_DIR / f"morpheme_tfm_k8{suffix}_fold{fold_i}.pt"
    t0 = time.time()
    if morph_ckpt_path.exists() and not args.retrain:
        print(f"  loading morpheme checkpoint {morph_ckpt_path.name} …")
        ckpt = torch.load(morph_ckpt_path, map_location="cpu")
        morph_model = MorphemeTfm(ckpt["V_morph"], k=K)
        morph_model.load_state_dict(ckpt["state_dict"])
        morph_model.eval()
    else:
        print(f"  training morpheme MiniTfm (V={V_morph}, K={K}) …")
        morph_model = train_morph_model(
            morph_conv_dict, all_train_ids, val_ids, V_morph, k=K,
            epochs=args.epochs, lr=args.lr, bs=args.bs, patience=args.patience,
        )
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": morph_model.state_dict(), "V_morph": V_morph, "k": K},
                   morph_ckpt_path)
        print(f"    saved → {morph_ckpt_path.name}")
    morph_agg = eval_morpheme_timing(
        morph_model, morph_conv_dict, test_ids, valid_pos,
        raw_ici_map, edges, morph_decoder, ici_map_str, segs_cache, K, bin_means=bin_means,
    )
    morph_agg["seconds"] = round(time.time() - t0, 1)
    fold_out["morpheme_tfm_k8"] = morph_agg
    print(f"    morpheme_tfm_k8: nn_dist={morph_agg.get('nn_dist_mean', float('nan')):.4f}  "
          f"n_click_err={morph_agg.get('n_click_err_mean', float('nan')):.2f}  "
          f"n={morph_agg['n_valid']}  ({morph_agg['seconds']}s)")
    del morph_model

    # ── Markov-1 baselines ─────────────────────────────────────────────────
    t0 = time.time()
    m1c = eval_markov1_compound_timing(
        per_seq, all_train_ids, test_ids, valid_pos,
        raw_ici_map, edges, ici_lookup, token_to_coda_arr, V_coda, V_token,
        bin_means=bin_means,
    )
    m1c["seconds"] = round(time.time() - t0, 1)
    fold_out["compound_markov1"] = m1c
    print(f"    compound_markov1: nn_dist={m1c.get('nn_dist_mean', float('nan')):.4f}  ({m1c['seconds']}s)")

    t0 = time.time()
    m1m = eval_markov1_morpheme_timing(
        morph_conv_dict, all_train_ids, test_ids, valid_pos,
        raw_ici_map, edges, morph_decoder, ici_map_str, segs_cache, V_morph,
        bin_means=bin_means,
    )
    m1m["seconds"] = round(time.time() - t0, 1)
    fold_out["morpheme_markov1"] = m1m
    print(f"    morpheme_markov1: nn_dist={m1m.get('nn_dist_mean', float('nan')):.4f}  ({m1m['seconds']}s)")

    return fold_out


# ── aggregation + reporting ───────────────────────────────────────────────────

METRIC_KEYS = [
    ("n_click_err",    "↓"),
    ("nn_dist",        "↓"),
    ("scale_free_rmse","↓"),
    ("dtw_norm",       "↓"),
    ("pattern_corr",   "↑"),
]


def _agg_folds(fold_results: list[dict]) -> dict:
    cell_keys = [k for k in fold_results[0] if k != "_meta"]
    summary: dict = {}
    for ck in cell_keys:
        rows = [f[ck] for f in fold_results if ck in f]
        entry: dict = {"n_folds": len(rows)}
        for mk, _ in METRIC_KEYS:
            vals = [r.get(f"{mk}_mean", float("nan")) for r in rows
                    if not math.isnan(r.get(f"{mk}_mean", float("nan")))]
            entry[f"{mk}_mean"] = float(np.mean(vals)) if vals else float("nan")
            entry[f"{mk}_std"]  = float(np.std(vals))  if vals else float("nan")
        summary[ck] = entry
    return summary


def _write_md(summary: dict, fold_results: list[dict], suffix: str) -> Path:
    train_desc = "clean-only (no hersh)" if suffix else "hersh + remaining clean"
    L = [
        "# ICI-timing benchmark",
        "",
        "Evaluate model predictions in **raw (unbucketed) timing space**.",
        "Predicted ICI strings decoded to seconds via per-bin empirical means (train-fold); compared to true raw ICI values.",
        "",
        f"Tier-aware 5-fold KFold (clean test; {train_desc} as train). K={K}.",
        "",
        "| metric | description |",
        "|--------|-------------|",
        "| n_click_err (↓) | \\|pred_clicks − true_clicks\\| |",
        "| nn_dist (↓) | mean nearest-neighbour click distance, norm by true coda duration |",
        "| scale_free_rmse (↓) | RMSE of ICI sequences after normalising each to unit sum |",
        "| dtw_norm (↓) | DTW path cost on ICI sequences, norm by max(n,m) |",
        "| pattern_corr (↑) | Pearson r of normalised ICI patterns (truncated to shorter) |",
        "",
    ]

    # summary table per metric
    cell_keys = list(summary.keys())
    for mk, arrow in METRIC_KEYS:
        L.append(f"## {mk} {arrow}")
        L.append("")
        L.append(f"| model | mean ± std |")
        L.append(f"|-------|:---------:|")
        ranked = sorted(cell_keys,
                        key=lambda c: summary[c].get(f"{mk}_mean", float("inf")),
                        reverse=(arrow == "↑"))
        for ck in ranked:
            s = summary[ck]
            L.append(f"| {ck} | {s[f'{mk}_mean']:.4f} ± {s[f'{mk}_std']:.4f} |")
        L.append("")

    # per-fold detail
    L.append("## Per-fold detail")
    L.append("")
    for ck in cell_keys:
        L.append(f"### {ck}")
        L.append("")
        header = "| fold | " + " | ".join(mk for mk, _ in METRIC_KEYS) + " | n_valid |"
        sep    = "|------|" + "|".join("------:" for _ in METRIC_KEYS) + "|--------:|"
        L.append(header)
        L.append(sep)
        for f in fold_results:
            if ck not in f:
                continue
            r = f[ck]
            vals = " | ".join(f"{r.get(f'{mk}_mean', float('nan')):.4f}" for mk, _ in METRIC_KEYS)
            L.append(f"| {f['_meta']['fold']} | {vals} | {r.get('n_valid', 0)} |")
        L.append("")

    out_path = OUT_DIR / f"predict_results_ici_timing{suffix}.md"
    out_path.write_text("\n".join(L) + "\n")
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--smoke", action="store_true",
                   help="2 folds, 10 epochs — pipeline sanity check.")
    p.add_argument("--no-hersh", action="store_true",
                   help="Retrain compound WhaleTfm on clean-only data.")
    p.add_argument("--retrain", action="store_true",
                   help="Force morpheme model retraining even if checkpoint exists.")
    p.add_argument("--compound-epochs", type=int, default=60)
    p.add_argument("--compound-patience", type=int, default=8)
    p.add_argument("--compound-lr", type=float, default=1e-3)
    args = p.parse_args(argv)
    if args.smoke:
        args.folds = 2
        args.epochs = 10
        args.patience = 3
        args.compound_epochs = 10
        args.compound_patience = 3

    suffix = "_no_hersh" if args.no_hersh else ""

    print("Loading compound corpus …")
    per_seq, V_coda, V_token, V_dt, token_to_coda = load_corpus()
    token_to_coda_arr = token_to_coda.numpy()

    print("Loading raw ICI values …")
    raw_ici_map, edges = load_raw_ici_map()
    print(f"  {len(raw_ici_map)} codas with valid raw ICI data")

    print("Loading bucketed ICI strings (for morpheme step count) …")
    from src.grammar.evaluate_ici_fidelity import load_ici_strings
    ici_map_str = load_ici_strings()

    print("Building segs_cache …")
    mf_model = _load_morfessor()
    all_ici_strs = {s for s in ici_map_str.values() if s is not None}
    segs_cache: dict[str, list[str]] = {}
    for s in all_ici_strs:
        morphs, _ = mf_model.viterbi_segment(s)
        segs_cache[s] = morphs

    print("Loading morpheme sequences …")
    morph_convs, morph_decoder = load_morpheme_seq_whale()
    morph_conv_dict = {mc.seq_id: mc for mc in morph_convs}
    V_morph = max(morph_decoder.keys()) + 1

    seq_ids = sorted(per_seq.keys())
    all_hersh_ids = [s for s in seq_ids if _tier(s) == "hersh"]
    hersh_ids = [] if args.no_hersh else all_hersh_ids
    clean_ids = [s for s in seq_ids if _tier(s) == "clean"]
    print(f"  V_coda={V_coda}  V_token={V_token}  V_morph={V_morph}  "
          f"hersh_in_train={len(hersh_ids)}  clean={len(clean_ids)}"
          + ("  [no-hersh ablation]" if args.no_hersh else ""))

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PARTIAL = OUT_DIR / f"predict_results_ici_timing{suffix}.partial.json"
    fold_results: list[dict] = []

    for fi, (tr, te) in enumerate(kf.split(clean_ids)):
        clean_train = [clean_ids[i] for i in tr]
        clean_test  = [clean_ids[i] for i in te]
        rng2 = np.random.default_rng(SEED + fi)
        ct = np.array(clean_train)
        rng2.shuffle(ct)
        n_val = max(1, int(VAL_FRAC * len(ct)))
        val_ids = ct[:n_val].tolist()
        clean_train_real = ct[n_val:].tolist()

        print(f"\nFOLD {fi}/{args.folds - 1}:  "
              f"clean_train={len(clean_train_real)}  hersh={len(hersh_ids)}  "
              f"val={n_val}  test={len(clean_test)}")

        fold_out = run_fold(
            fi, hersh_ids, clean_train_real, val_ids, clean_test,
            per_seq, V_coda, V_token, V_dt,
            token_to_coda, token_to_coda_arr,
            raw_ici_map, edges,
            ici_map_str, segs_cache,
            morph_conv_dict, morph_decoder, V_morph,
            args,
        )
        fold_out["_meta"] = dict(
            fold=fi,
            n_clean_train=len(clean_train_real),
            n_hersh=len(hersh_ids),
            n_val=n_val,
            n_test=len(clean_test),
        )
        fold_results.append(fold_out)
        PARTIAL.write_text(json.dumps({"folds": fold_results}, indent=2))
        print(f"  partial saved → {PARTIAL}")

    summary = _agg_folds(fold_results)
    out_json = OUT_DIR / f"predict_results_ici_timing{suffix}.json"
    out_json.write_text(json.dumps(dict(summary=summary, folds=fold_results), indent=2))
    out_md = _write_md(summary, fold_results, suffix)
    print(f"\nwrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
