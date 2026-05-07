"""
ICI-fidelity benchmark: compound-token model (K=8 joint checkpoint, no retraining)
vs morpheme-seq model (trained fresh per fold) on predicting the true ICI string
of the next sperm whale coda.

Metric: ici_bpt = -log2 P(true ICI string | context).  Lower = better.
Secondary: ICI exact match, per-symbol accuracy, normalised edit distance.

Compound model:
    P(s*|ctx) = Σ_r P_model(rhythm=r|ctx) × P_train(s*|rhythm=r)
    Loads k8_joint_ckpt_tfm_joint_last_fold{i}.pt — no retraining.

Morpheme model:
    P(s*|ctx) = Π_i P_model(morph_i | ctx + true_morphs_{<i})
    2L-4h-d64 MiniTransformer, trained on each fold's tier-aware split.

Markov-1 baselines for both tokenisations.

Fold split: tier-aware 5-fold KFold identical to predict_full.py —
    KFold(n_splits=5, shuffle=True, random_state=42) over clean_ids;
    hersh as extra training; 10% val from clean_train.

Usage:
    python -m src.grammar.evaluate_ici_fidelity
    python -m src.grammar.evaluate_ici_fidelity --smoke
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn

from src.grammar.predict_full import (
    HERSH_PREFIXES, CLEAN_PREFIXES,
    WhaleTfm, load_corpus,
)
from src.grammar.whale_morpheme_tokeniser import (
    _load_joined, _load_morfessor, load_morpheme_seq_whale,
)
from src.grammar.morpheme_discovery import (
    compute_global_bin_edges, ici_row_to_string,
    MIN_CLICKS, MAX_CLICKS, N_BINS,
)

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "outputs" / "grammar" / "checkpoints"
OUT_DIR  = ROOT / "outputs" / "grammar"
K        = 8
SEED     = 42
VAL_FRAC = 0.1


# ── tier helper ───────────────────────────────────────────────────────────────

def _tier(seq_id: str) -> str:
    if any(seq_id.startswith(p) for p in HERSH_PREFIXES):
        return "hersh"
    if any(seq_id.startswith(p) for p in CLEAN_PREFIXES):
        return "clean"
    return "other"


# ── data loading ──────────────────────────────────────────────────────────────

def load_ici_strings() -> dict[tuple[str, int], str | None]:
    """Return {(seq_id, item_pos): ici_string_or_None} for all rows."""
    joined = _load_joined()
    valid_mask = (
        joined["n_clicks"].notna()
        & joined["n_clicks"].between(MIN_CLICKS, MAX_CLICKS)
    )
    edges = compute_global_bin_edges(joined[valid_mask].copy(), n_bins=N_BINS)
    out: dict[tuple, str | None] = {}
    for _, row in joined.iterrows():
        sid = str(row["sequenceId"])
        pos = int(row["itemPosition"])
        n = row.get("n_clicks")
        try:
            n_f = float(n)
        except (TypeError, ValueError):
            out[(sid, pos)] = None
            continue
        if math.isnan(n_f) or not (MIN_CLICKS <= int(n_f) <= MAX_CLICKS):
            out[(sid, pos)] = None
            continue
        s = ici_row_to_string(row, edges)
        out[(sid, pos)] = s if s else None
    return out


def build_ici_lookup(
    per_seq: dict,
    ici_map: dict[tuple, str | None],
    train_ids: list[str],
    token_to_coda: torch.Tensor,
) -> tuple[dict[int, Counter], set[str]]:
    """Per-fold ICI distribution: ici_lookup[rhythm_class] = Counter(ici_str→count).
    Also returns the set of all ICI strings seen in training."""
    lookup: dict[int, Counter] = defaultdict(Counter)
    all_ici: set[str] = set()
    for sid in train_ids:
        seq = per_seq.get(sid)
        if seq is None:
            continue
        for pos, tok in enumerate(seq["token"]):
            ici_str = ici_map.get((sid, pos))
            if ici_str is None:
                continue
            rhythm = int(token_to_coda[tok].item())
            lookup[rhythm][ici_str] += 1
            all_ici.add(ici_str)
    return dict(lookup), all_ici


# ── compound windows with target metadata ────────────────────────────────────

def build_compound_windows_meta(
    per_seq: dict, seq_ids: list[str], k: int,
) -> tuple[dict, list[tuple[str, int]]]:
    """Like predict_full.build_windows but also returns (seq_id, item_pos) per window."""
    chans = ("coda", "token", "dt", "dt_log", "has_ts")
    raw = {c: [] for c in chans}
    meta: list[tuple[str, int]] = []
    for sid in seq_ids:
        seq = per_seq.get(sid)
        if seq is None:
            continue
        L = len(seq["coda"])
        if L < k + 1:
            continue
        for i in range(L - k):
            for c in chans:
                raw[c].append(seq[c][i : i + k + 1])
            meta.append((sid, i + k))
    if not meta:
        empty_i = torch.empty(0, k + 1, dtype=torch.long)
        empty_f = torch.empty(0, k + 1, dtype=torch.float32)
        return dict(coda=empty_i, token=empty_i, dt=empty_i,
                    dt_log=empty_f, has_ts=empty_i), []
    return dict(
        coda=torch.tensor(raw["coda"], dtype=torch.long),
        token=torch.tensor(raw["token"], dtype=torch.long),
        dt=torch.tensor(raw["dt"], dtype=torch.long),
        dt_log=torch.tensor(raw["dt_log"], dtype=torch.float32),
        has_ts=torch.tensor(raw["has_ts"], dtype=torch.long),
    ), meta


# ── compound ICI probability helpers ─────────────────────────────────────────

def _build_p_rhythm(
    token_probs: np.ndarray,
    token_to_coda_arr: np.ndarray,
    V_coda: int,
) -> np.ndarray:
    """P(rhythm=r | context) = Σ_{tokens with rhythm r} P(token)."""
    p_rhythm = np.zeros(V_coda)
    np.add.at(p_rhythm, token_to_coda_arr, token_probs)
    return p_rhythm


def _compound_ici_lp(
    token_probs: np.ndarray,
    true_ici: str,
    token_to_coda_arr: np.ndarray,
    ici_lookup: dict[int, Counter],
    all_ici_strings: set[str],
    V_coda: int,
    alpha: float = 0.5,
) -> float:
    n_ici = max(len(all_ici_strings), 1)
    p_rhythm = _build_p_rhythm(token_probs, token_to_coda_arr, V_coda)
    total = 0.0
    for r in range(V_coda):
        p_r = float(p_rhythm[r])
        if p_r < 1e-15:
            continue
        lookup_r = ici_lookup.get(r, Counter())
        total_r = sum(lookup_r.values())
        if total_r > 0:
            p_ici_given_r = (lookup_r.get(true_ici, 0) + alpha) / (total_r + alpha * n_ici)
        else:
            p_ici_given_r = 1.0 / n_ici
        total += p_r * p_ici_given_r
    return math.log2(max(total, 1e-12))


def _compound_map_ici(
    token_probs: np.ndarray,
    token_to_coda_arr: np.ndarray,
    ici_lookup: dict[int, Counter],
    V_coda: int,
) -> str:
    p_rhythm = _build_p_rhythm(token_probs, token_to_coda_arr, V_coda)
    best_r = int(np.argmax(p_rhythm))
    lookup_r = ici_lookup.get(best_r, Counter())
    if lookup_r:
        return lookup_r.most_common(1)[0][0]
    return ""


# ── compound model eval ───────────────────────────────────────────────────────

def load_compound_ckpt(fold_i: int) -> WhaleTfm:
    path = CKPT_DIR / f"k8_joint_ckpt_tfm_joint_last_fold{fold_i}.pt"
    ckpt = torch.load(path, map_location="cpu")
    model = WhaleTfm(
        V_in=ckpt["V_token"], V_token_out=ckpt["V_token"], V_coda_out=None,
        n_dt=ckpt["V_dt"], k=ckpt["k"], target="joint", with_dt_input=True,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def eval_compound_fold(
    model: WhaleTfm,
    test_windows: dict,
    test_meta: list[tuple[str, int]],
    ici_map: dict,
    ici_lookup: dict,
    token_to_coda_arr: np.ndarray,
    all_ici_strings: set[str],
    V_coda: int,
    bs: int = 512,
) -> tuple[dict, set[tuple[str, int]]]:
    """Run compound model on test windows; return metrics and set of evaluated positions."""
    model.eval()
    n_wins = test_windows["token"].size(0)
    if n_wins == 0:
        return {}, set()

    k_val = test_windows["token"].size(1) - 1
    x_all = test_windows["token"][:, :k_val]
    dt_log = test_windows["dt_log"][:, :k_val]
    has_ts = test_windows["has_ts"][:, :k_val].float()
    dt_input_all = torch.stack([dt_log, has_ts], dim=-1)

    # Collect per-window softmax distributions
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, n_wins, bs):
            xb = x_all[i : i + bs]
            dtb = dt_input_all[i : i + bs]
            out = model(xb, dt_input=dtb)
            probs = torch.softmax(out["token"][:, -1], dim=-1).cpu().numpy()
            all_probs.extend(probs)

    log2_probs, pred_icis, true_icis = [], [], []
    valid_positions: set[tuple[str, int]] = set()

    for (sid, pos), probs in zip(test_meta, all_probs):
        true_ici = ici_map.get((sid, pos))
        if true_ici is None:
            continue
        lp = _compound_ici_lp(probs, true_ici, token_to_coda_arr,
                               ici_lookup, all_ici_strings, V_coda)
        log2_probs.append(lp)
        true_icis.append(true_ici)
        valid_positions.add((sid, pos))
        pred_icis.append(_compound_map_ici(probs, token_to_coda_arr, ici_lookup, V_coda))

    metrics = dict(
        ici_bpt=-float(np.mean(log2_probs)) if log2_probs else float("inf"),
        n_valid=len(log2_probs),
    )
    metrics.update(_secondary_metrics(pred_icis, true_icis))
    return metrics, valid_positions


# ── Markov-1 compound ICI ─────────────────────────────────────────────────────

def eval_markov1_compound(
    per_seq: dict,
    train_ids: list[str], test_ids: list[str],
    valid_positions: set[tuple[str, int]],
    ici_map: dict,
    ici_lookup: dict,
    token_to_coda_arr: np.ndarray,
    all_ici_strings: set[str],
    V_coda: int, V_token: int,
    alpha: float = 0.5,
) -> dict:
    """Markov-1 on compound token sequences, ICI via rhythm marginalization."""
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

    log2_probs, pred_icis, true_icis = [], [], []
    for sid in test_ids:
        seq = per_seq.get(sid)
        if seq is None:
            continue
        toks = seq["token"]
        for pos in range(1, len(toks)):
            if (sid, pos) not in valid_positions:
                continue
            true_ici = ici_map.get((sid, pos))
            if true_ici is None:
                continue
            ctx_tok = toks[pos - 1]
            c = trans.get(ctx_tok)
            if c is None:
                tok_probs = (unigram_arr + alpha) / (n_train + alpha * V_token)
            else:
                c_arr = np.array([c.get(t, 0) for t in range(V_token)], dtype=float)
                denom = c_arr.sum() + alpha * V_token
                tok_probs = (c_arr + alpha) / denom
            lp = _compound_ici_lp(tok_probs, true_ici, token_to_coda_arr,
                                   ici_lookup, all_ici_strings, V_coda)
            log2_probs.append(lp)
            true_icis.append(true_ici)
            pred_icis.append(_compound_map_ici(tok_probs, token_to_coda_arr, ici_lookup, V_coda))

    metrics = dict(ici_bpt=-float(np.mean(log2_probs)) if log2_probs else float("inf"),
                   n_valid=len(log2_probs))
    metrics.update(_secondary_metrics(pred_icis, true_icis))
    return metrics


# ── morpheme model ────────────────────────────────────────────────────────────

class MorphemeTfm(nn.Module):
    def __init__(self, V: int, k: int = K, d: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.k = k
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.pos = nn.Embedding(k, d)
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4 * d,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.tfm = nn.TransformerEncoder(enc, n_layers)
        self.head = nn.Linear(d, V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K = x.shape
        pos_ids = torch.arange(K, device=x.device).expand(B, K)
        h = self.emb(x) + self.pos(pos_ids)
        return self.head(self.tfm(h)[:, -1])


def _build_morph_windows(
    morph_conv_dict: dict,
    seq_ids: list[str], k: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Build (X, y, coda_ids) from morpheme sequences.
    coda_ids[i] = '{seq_id}__{coda_index}' for the coda containing the target morpheme.
    """
    X_list, y_list, cid_list = [], [], []
    for sid in seq_ids:
        mc = morph_conv_dict.get(sid)
        if mc is None:
            continue
        toks = mc.tokens
        n_codas = len(mc.coda_boundaries)
        coda_of = [0] * len(toks)
        for ci, start in enumerate(mc.coda_boundaries):
            end = mc.coda_boundaries[ci + 1] if ci + 1 < n_codas else len(toks)
            for p in range(start, end):
                coda_of[p] = ci
        for i in range(1, len(toks)):
            ctx = toks[max(0, i - k) : i]
            ctx = [0] * (k - len(ctx)) + ctx
            X_list.append(ctx)
            y_list.append(toks[i])
            cid_list.append(f"{sid}__{coda_of[i]}")
    if not X_list:
        return torch.empty(0, k, dtype=torch.long), torch.empty(0, dtype=torch.long), []
    return (
        torch.tensor(X_list, dtype=torch.long),
        torch.tensor(y_list, dtype=torch.long),
        cid_list,
    )


def train_morph_model(
    morph_conv_dict: dict,
    train_ids: list[str], val_ids: list[str],
    V_morph: int, k: int = K,
    epochs: int = 80, lr: float = 1e-3, bs: int = 128, patience: int = 10,
) -> MorphemeTfm:
    X_tr, y_tr, _ = _build_morph_windows(morph_conv_dict, train_ids, k)
    X_val, y_val, _ = _build_morph_windows(morph_conv_dict, val_ids, k)
    model = MorphemeTfm(V_morph, k=k)
    torch.manual_seed(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_val, since = float("inf"), 0
    best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
    n = X_tr.size(0)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i : i + bs]
            opt.zero_grad()
            loss_fn(model(X_tr[b]), y_tr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item() if X_val.size(0) > 0 else float("inf")
        if val_loss < best_val - 1e-4:
            best_val, since = val_loss, 0
            best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    return model


def eval_morpheme_fold(
    model: MorphemeTfm,
    morph_conv_dict: dict,
    test_ids: list[str],
    valid_positions: set[tuple[str, int]],
    ici_map: dict,
    segs_cache: dict[str, list[str]],
    morph_decoder: dict[int, str],
    k: int = K,
    bs: int = 512,
) -> dict:
    """Evaluate morpheme model at exactly the same (seq_id, item_pos) as the compound model."""
    model.eval()
    X_te, y_te, cid_te = _build_morph_windows(morph_conv_dict, test_ids, k)
    if X_te.size(0) == 0:
        return dict(ici_bpt=float("inf"), n_valid=0)

    # Batched forward for per-morpheme log-probs
    all_lp2: list[float] = []
    with torch.no_grad():
        for i in range(0, X_te.size(0), bs):
            xb = X_te[i : i + bs]
            logits = model(xb)
            lp = torch.log_softmax(logits, dim=-1)
            tgt = y_te[i : i + bs]
            lp_true = lp.gather(1, tgt.unsqueeze(1)).squeeze(1)
            all_lp2.extend((lp_true / math.log(2)).cpu().tolist())

    # Sum per-coda
    coda_lp: dict[str, float] = defaultdict(float)
    for lp, cid in zip(all_lp2, cid_te):
        coda_lp[cid] += lp

    log2_probs, pred_icis, true_icis = [], [], []
    for cid, lp_sum in coda_lp.items():
        parts = cid.rsplit("__", 1)
        if len(parts) != 2:
            continue
        sid, ci_str = parts
        ci = int(ci_str)
        if (sid, ci) not in valid_positions:
            continue
        true_ici = ici_map.get((sid, ci))
        if true_ici is None:
            continue

        log2_probs.append(lp_sum)
        true_icis.append(true_ici)

        # Greedy MAP decode
        mc = morph_conv_dict.get(sid)
        if mc is None or ci >= len(mc.coda_boundaries):
            pred_icis.append("")
            continue
        coda_start = mc.coda_boundaries[ci]
        ctx_toks = mc.tokens[max(0, coda_start - k) : coda_start]
        ctx = [0] * (k - len(ctx_toks)) + list(ctx_toks)
        true_seg = segs_cache.get(true_ici, [])
        n_steps = len(true_seg) if true_seg else 1
        pred_strs, cur_ctx = [], list(ctx)
        with torch.no_grad():
            for _ in range(n_steps):
                x = torch.tensor([cur_ctx], dtype=torch.long)
                pred_id = int(model(x).argmax(-1).item())
                pred_strs.append(morph_decoder.get(pred_id, ""))
                cur_ctx = cur_ctx[1:] + [pred_id]
        pred_icis.append("".join(pred_strs))

    metrics = dict(ici_bpt=-float(np.mean(log2_probs)) if log2_probs else float("inf"),
                   n_valid=len(log2_probs))
    metrics.update(_secondary_metrics(pred_icis, true_icis))
    return metrics


# ── Markov-1 morpheme ICI ────────────────────────────────────────────────────

def eval_markov1_morpheme(
    morph_conv_dict: dict,
    train_ids: list[str], test_ids: list[str],
    valid_positions: set[tuple[str, int]],
    ici_map: dict,
    segs_cache: dict,
    morph_decoder: dict,
    V_morph: int,
    alpha: float = 0.5,
) -> dict:
    """Markov-1 on morpheme sequences, chain-rule per coda."""
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

    n_train = sum(unigram.values())
    log2_probs, pred_icis, true_icis = [], [], []

    for sid in test_ids:
        mc = morph_conv_dict.get(sid)
        if mc is None:
            continue
        bd = mc.coda_boundaries
        n_codas = len(bd)
        for ci, coda_start in enumerate(bd):
            if (sid, ci) not in valid_positions:
                continue
            true_ici = ici_map.get((sid, ci))
            if true_ici is None:
                continue
            coda_end = bd[ci + 1] if ci + 1 < n_codas else len(mc.tokens)
            coda_morphemes = mc.tokens[coda_start:coda_end]
            if not coda_morphemes:
                continue
            coda_lp = 0.0
            pred_strs: list[str] = []
            prev = mc.tokens[coda_start - 1] if coda_start > 0 else 0
            for morph_id in coda_morphemes:
                c = trans.get(prev)
                if c is None:
                    p = (unigram.get(morph_id, 0) + alpha) / (n_train + alpha * V_morph)
                    pred_id = unigram.most_common(1)[0][0] if unigram else 0
                else:
                    denom = sum(c.values()) + alpha * V_morph
                    p = (c.get(morph_id, 0) + alpha) / denom
                    pred_id = c.most_common(1)[0][0]
                coda_lp += math.log2(max(p, 1e-12))
                pred_strs.append(morph_decoder.get(pred_id, ""))
                prev = morph_id
            log2_probs.append(coda_lp)
            pred_icis.append("".join(pred_strs))
            true_icis.append(true_ici)

    metrics = dict(ici_bpt=-float(np.mean(log2_probs)) if log2_probs else float("inf"),
                   n_valid=len(log2_probs))
    metrics.update(_secondary_metrics(pred_icis, true_icis))
    return metrics


# ── secondary metrics ─────────────────────────────────────────────────────────

def _secondary_metrics(pred_icis: list[str], true_icis: list[str]) -> dict:
    n = len(true_icis)
    if n == 0:
        return dict(exact_match=0.0, symbol_acc=0.0, norm_edit=1.0)
    exact = sum(p == t for p, t in zip(pred_icis, true_icis)) / n
    sym_accs, edit_dists = [], []
    for p, t in zip(pred_icis, true_icis):
        max_len = max(len(p), len(t), 1)
        sym_accs.append(sum(a == b for a, b in zip(p, t)) / max_len)
        ratio = difflib.SequenceMatcher(None, p, t).ratio()
        edit_dists.append(1.0 - ratio)
    return dict(
        exact_match=float(exact),
        symbol_acc=float(np.mean(sym_accs)),
        norm_edit=float(np.mean(edit_dists)),
    )


# ── fold driver ───────────────────────────────────────────────────────────────

def run_fold(
    fold_i: int,
    hersh_ids: list[str], clean_train_ids: list[str],
    val_ids: list[str], test_ids: list[str],
    per_seq: dict, V_coda: int, V_token: int, V_dt: int,
    token_to_coda: torch.Tensor, token_to_coda_arr: np.ndarray,
    ici_map: dict,
    morph_conv_dict: dict, morph_decoder: dict,
    segs_cache: dict, V_morph: int,
    args: argparse.Namespace,
) -> dict:
    all_train_ids = hersh_ids + clean_train_ids

    print(f"  building ICI lookup ({len(all_train_ids)} train seqs) …")
    ici_lookup, all_ici = build_ici_lookup(per_seq, ici_map, all_train_ids, token_to_coda)

    fold_out: dict = {}

    # ── compound (K=8 joint checkpoint, no retraining) ────────────────────
    print(f"  loading compound K=8 checkpoint fold {fold_i} …")
    comp_model = load_compound_ckpt(fold_i)
    test_windows, test_meta = build_compound_windows_meta(per_seq, test_ids, K)

    t0 = time.time()
    comp_metrics, valid_pos = eval_compound_fold(
        comp_model, test_windows, test_meta, ici_map, ici_lookup,
        token_to_coda_arr, all_ici, V_coda,
    )
    comp_metrics["seconds"] = round(time.time() - t0, 1)
    fold_out["compound_tfm_k8"] = comp_metrics
    print(f"    compound_tfm_k8: ici_bpt={comp_metrics['ici_bpt']:.3f}  "
          f"exact={comp_metrics['exact_match']:.3f}  "
          f"sym_acc={comp_metrics['symbol_acc']:.3f}  "
          f"n={comp_metrics['n_valid']}  ({comp_metrics['seconds']}s)")
    del comp_model

    # ── morpheme model (trained fresh) ────────────────────────────────────
    print(f"  training morpheme MiniTfm (V={V_morph}, K={K}) …")
    t0 = time.time()
    morph_model = train_morph_model(
        morph_conv_dict, all_train_ids, val_ids, V_morph, k=K,
        epochs=args.epochs, lr=args.lr, bs=args.bs, patience=args.patience,
    )
    morph_metrics = eval_morpheme_fold(
        morph_model, morph_conv_dict, test_ids, valid_pos, ici_map,
        segs_cache, morph_decoder, K,
    )
    morph_metrics["seconds"] = round(time.time() - t0, 1)
    fold_out["morpheme_tfm_k8"] = morph_metrics
    print(f"    morpheme_tfm_k8: ici_bpt={morph_metrics['ici_bpt']:.3f}  "
          f"exact={morph_metrics['exact_match']:.3f}  "
          f"sym_acc={morph_metrics['symbol_acc']:.3f}  "
          f"n={morph_metrics['n_valid']}  ({morph_metrics['seconds']}s)")
    del morph_model

    # ── Markov-1 baselines ────────────────────────────────────────────────
    t0 = time.time()
    m1c = eval_markov1_compound(
        per_seq, all_train_ids, test_ids, valid_pos,
        ici_map, ici_lookup, token_to_coda_arr, all_ici, V_coda, V_token,
    )
    m1c["seconds"] = round(time.time() - t0, 1)
    fold_out["compound_markov1"] = m1c
    print(f"    compound_markov1: ici_bpt={m1c['ici_bpt']:.3f}  ({m1c['seconds']}s)")

    t0 = time.time()
    m1m = eval_markov1_morpheme(
        morph_conv_dict, all_train_ids, test_ids, valid_pos,
        ici_map, segs_cache, morph_decoder, V_morph,
    )
    m1m["seconds"] = round(time.time() - t0, 1)
    fold_out["morpheme_markov1"] = m1m
    print(f"    morpheme_markov1: ici_bpt={m1m['ici_bpt']:.3f}  ({m1m['seconds']}s)")

    return fold_out


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
    args = p.parse_args(argv)
    if args.smoke:
        args.folds = 2
        args.epochs = 10
        args.patience = 3

    # ── load data ─────────────────────────────────────────────────────────
    print("Loading compound corpus …")
    per_seq, V_coda, V_token, V_dt, token_to_coda = load_corpus()
    token_to_coda_arr = token_to_coda.numpy()

    print("Loading ICI strings (may take a moment) …")
    ici_map = load_ici_strings()

    print("Loading morpheme sequences …")
    morph_convs, morph_decoder = load_morpheme_seq_whale()
    morph_conv_dict = {mc.seq_id: mc for mc in morph_convs}
    V_morph = max(morph_decoder.keys()) + 1

    print("Building segs_cache from Morfessor model …")
    mf_model = _load_morfessor()
    all_ici_strs = {s for s in ici_map.values() if s is not None}
    segs_cache: dict[str, list[str]] = {}
    for s in all_ici_strs:
        morphs, _ = mf_model.viterbi_segment(s)
        segs_cache[s] = morphs

    # ── fold splits ───────────────────────────────────────────────────────
    seq_ids = sorted(per_seq.keys())
    hersh_ids = [s for s in seq_ids if _tier(s) == "hersh"]
    clean_ids = [s for s in seq_ids if _tier(s) == "clean"]
    print(f"  V_coda={V_coda}  V_token={V_token}  V_morph={V_morph}  "
          f"hersh={len(hersh_ids)}  clean={len(clean_ids)}  "
          f"n_ici_types={len(all_ici_strs)}")

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PARTIAL = OUT_DIR / "predict_results_ici_fidelity.partial.json"
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
            ici_map, morph_conv_dict, morph_decoder, segs_cache, V_morph,
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

    # ── aggregate ─────────────────────────────────────────────────────────
    cell_keys = [k for k in fold_results[0] if k != "_meta"]
    summary: dict = {}
    for ck in cell_keys:
        rows = [f[ck] for f in fold_results if ck in f]
        bpts    = [r["ici_bpt"]     for r in rows]
        exacts  = [r.get("exact_match", 0.0) for r in rows]
        syms    = [r.get("symbol_acc", 0.0)  for r in rows]
        edits   = [r.get("norm_edit", 1.0)   for r in rows]
        summary[ck] = dict(
            ici_bpt_mean  = float(np.mean(bpts)),
            ici_bpt_std   = float(np.std(bpts)),
            exact_match   = float(np.mean(exacts)),
            symbol_acc    = float(np.mean(syms)),
            norm_edit     = float(np.mean(edits)),
            n_folds       = len(rows),
        )

    OUT_JSON = OUT_DIR / "predict_results_ici_fidelity.json"
    OUT_MD   = OUT_DIR / "predict_results_ici_fidelity.md"
    OUT_JSON.write_text(json.dumps(dict(summary=summary, folds=fold_results), indent=2))

    L = [
        "# ICI-fidelity benchmark",
        "",
        "How well does each model predict the true ICI string of the next whale coda?",
        "",
        f"Tier-aware 5-fold KFold (clean test set; hersh + remaining clean as train).",
        f"Compound model: K=8 joint checkpoint (loaded, no retraining).  "
        f"Morpheme model: MiniTfm 2L-4h-d64 trained per fold.",
        "",
        "| model | ici_bpt (↓) | exact_match (↑) | symbol_acc (↑) | norm_edit (↓) |",
        "|-------|:----------:|:---------------:|:--------------:|:-------------:|",
    ]
    for ck, s in sorted(summary.items(), key=lambda x: x[1]["ici_bpt_mean"]):
        L.append(
            f"| {ck} | {s['ici_bpt_mean']:.3f} ± {s['ici_bpt_std']:.3f} | "
            f"{s['exact_match']:.3f} | {s['symbol_acc']:.3f} | {s['norm_edit']:.3f} |"
        )
    L.append("")
    L.append("## Per-fold detail")
    L.append("")
    for ck in cell_keys:
        L.append(f"### {ck}")
        L.append("")
        L.append("| fold | ici_bpt | exact_match | symbol_acc | norm_edit | n_valid |")
        L.append("|------|--------:|------------:|-----------:|----------:|--------:|")
        for f in fold_results:
            if ck not in f:
                continue
            r = f[ck]
            L.append(
                f"| {f['_meta']['fold']} | {r['ici_bpt']:.3f} | "
                f"{r.get('exact_match', 0):.3f} | {r.get('symbol_acc', 0):.3f} | "
                f"{r.get('norm_edit', 1):.3f} | {r.get('n_valid', 0)} |"
            )
        L.append("")

    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
