"""
K-fold next-token prediction for morpheme re-tokenisation experiments.

Phase 1 (--tokenisation ici_string):
    Each coda maps to one ICI-string token (~1 916 vocab).  Sequence
    length is unchanged.  Reports per-ICI-string bpt and accuracy via the
    standard M0–M7 suite.

Phase 2 (--tokenisation morpheme_seq):
    Each coda expands to its Morfessor morpheme sub-units (~371 vocab).
    Models are trained and evaluated on morpheme tokens.  Additionally
    reports per-coda bpt by summing the per-morpheme log-probabilities
    within each coda boundary — directly comparable to the compound
    baseline's per-coda bpt.

Output:
    outputs/grammar/predict_results_morpheme_<tokenisation>.{json,md}

Usage:
    python -m src.grammar.predict_morpheme --tokenisation ici_string
    python -m src.grammar.predict_morpheme --tokenisation morpheme_seq
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch import nn

from src.grammar.whale_morpheme_tokeniser import (
    load_ici_string_whale,
    load_morpheme_seq_whale,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "grammar"
CONTEXT_K = 8
N_FOLDS = 5
PAD = "<PAD>"


# ---------------------------------------------------------------------------
# Sequence building
# ---------------------------------------------------------------------------


def _ici_seq_dict(csv: Path) -> dict[str, list[str]]:
    """Load ICI-string sequences; return {seq_id: [token_str, ...]}."""
    df, _ = load_ici_string_whale(csv)
    df = df.sort_values(["sequenceId", "itemPosition"])
    out: dict[str, list[str]] = {}
    for sid, grp in df.groupby("sequenceId"):
        out[str(sid)] = [str(v) for v in grp["Coda"].tolist()]
    return out


def _morpheme_seq_dicts(csv: Path) -> tuple[
    dict[str, list[str]],    # seq_id -> morpheme token strings
    dict[str, list[int]],    # seq_id -> coda boundary positions
]:
    """Load morpheme-seq data; return token dict and coda-boundary dict."""
    convs, _ = load_morpheme_seq_whale(csv)
    tok_dict: dict[str, list[str]] = {}
    bd_dict: dict[str, list[int]] = {}
    for mc in convs:
        tok_dict[mc.seq_id] = [str(t) for t in mc.tokens]
        bd_dict[mc.seq_id] = mc.coda_boundaries
    return tok_dict, bd_dict


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def build_windows(
    seq_dict: dict[str, list[str]],
    seq_ids: list[str],
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for sid in seq_ids:
        s = seq_dict[sid]
        for i in range(1, len(s)):
            ctx = s[max(0, i - k) : i]
            ctx = [PAD] * (k - len(ctx)) + ctx
            X.append(ctx)
            y.append(s[i])
    return np.array(X, dtype=object), np.array(y, dtype=object)


def build_windows_with_coda_ids(
    seq_dict: dict[str, list[str]],
    bd_dict: dict[str, list[int]],
    seq_ids: list[str],
    k: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Like build_windows but also returns per-window coda identifiers.

    coda_ids[i] = '<seq_id>__<coda_index>' for window i.  Windows whose
    target position belongs to the same coda share the same coda id, so
    their log-probs can be summed to give the per-coda log-prob.
    """
    X, y, coda_ids = [], [], []
    for sid in seq_ids:
        s = seq_dict[sid]
        bd = bd_dict[sid]
        # Map position → coda index
        coda_of = [0] * len(s)
        n_codas = len(bd)
        for ci, start in enumerate(bd):
            end = bd[ci + 1] if ci + 1 < n_codas else len(s)
            for p in range(start, end):
                coda_of[p] = ci
        for i in range(1, len(s)):
            ctx = s[max(0, i - k) : i]
            ctx = [PAD] * (k - len(ctx)) + ctx
            X.append(ctx)
            y.append(s[i])
            coda_ids.append(f"{sid}__{coda_of[i]}")
    return np.array(X, dtype=object), np.array(y, dtype=object), coda_ids


def per_coda_bpt(log2_probs: np.ndarray, coda_ids: list[str]) -> float:
    """Aggregate per-morpheme log2 probabilities to per-coda bpt.

    per_coda_bpt = -mean_over_codas(sum_over_morphemes(log2 p_morpheme))
    """
    coda_sum: dict[str, float] = defaultdict(float)
    for lp, cid in zip(log2_probs, coda_ids):
        coda_sum[cid] += lp
    return float(-np.mean(list(coda_sum.values())))


# ---------------------------------------------------------------------------
# Models (copied / adapted from predict_kfold.py)
# ---------------------------------------------------------------------------


def bits_per_token(p_true: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.mean(np.log2(np.maximum(p_true, eps))))


def evaluate_baselines(
    y_train: np.ndarray, X_train: np.ndarray,
    y_test: np.ndarray, X_test: np.ndarray,
    vocab_size: int, classes: list,
    coda_ids_test: list[str] | None = None,
) -> dict:
    out: dict = {}
    train_counts = Counter(y_train.tolist())
    majority = train_counts.most_common(1)[0][0]
    n_train = len(y_train)
    smoothing = 0.5

    p_smoothed = np.array([
        (train_counts.get(t, 0) + smoothing) / (n_train + smoothing * vocab_size)
        for t in y_test
    ])
    lp2 = np.log2(np.maximum(p_smoothed, 1e-12))
    bpt = -float(np.mean(lp2))
    coda_bpt_val = per_coda_bpt(lp2, coda_ids_test) if coda_ids_test else None
    out["M0_majority"] = dict(
        accuracy=float(np.mean(y_test == majority)),
        bits_per_token=bpt,
        **({"coda_bpt": coda_bpt_val} if coda_bpt_val is not None else {}),
    )

    def markov_k(k: int) -> dict:
        trans: dict[tuple, Counter] = {}
        for i, ctx in enumerate(X_train):
            key = tuple(ctx[-k:])
            trans.setdefault(key, Counter())[y_train[i]] += 1
        p_true = np.empty(len(y_test))
        preds = []
        for i, ctx in enumerate(X_test):
            key = tuple(ctx[-k:])
            c = trans.get(key)
            if c is None:
                p_true[i] = (train_counts.get(y_test[i], 0) + smoothing) / (
                    n_train + smoothing * vocab_size)
                preds.append(majority)
            else:
                denom = sum(c.values()) + smoothing * vocab_size
                p_true[i] = (c.get(y_test[i], 0) + smoothing) / denom
                preds.append(c.most_common(1)[0][0])
        lp2_m = np.log2(np.maximum(p_true, 1e-12))
        coda_bpt_m = per_coda_bpt(lp2_m, coda_ids_test) if coda_ids_test else None
        r = dict(
            accuracy=float(np.mean(np.array(preds) == y_test)),
            bits_per_token=-float(np.mean(lp2_m)),
        )
        if coda_bpt_m is not None:
            r["coda_bpt"] = coda_bpt_m
        return r

    out["M1_markov1"] = markov_k(1)
    out["M2_markov2"] = markov_k(2)
    return out


class EmbMLP(nn.Module):
    def __init__(self, V: int, k: int, d: int = 32, hidden: tuple = (256, 128)):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        layers = []
        in_dim = d * k
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = h
        layers.append(nn.Linear(in_dim, V))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.emb(x).flatten(1))


class MiniTransformer(nn.Module):
    def __init__(self, V: int, k: int, d: int = 64, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(k, d)
        enc = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4 * d,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.tfm = nn.TransformerEncoder(enc, n_layers)
        self.head = nn.Linear(d, V)
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K = x.shape
        pos_ids = torch.arange(K, device=x.device).expand(B, K)
        h = self.emb(x) + self.pos(pos_ids)
        return self.head(self.tfm(h)[:, -1])


def _train_torch(
    model: nn.Module,
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_val: torch.Tensor, y_val: torch.Tensor,
    epochs: int = 80, lr: float = 1e-3, bs: int = 128, seed: int = 0,
) -> nn.Module:
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    n = X_train.shape[0]
    best_val, since = float("inf"), 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    patience = 10
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i : i + bs]
            opt.zero_grad()
            loss_fn(model(X_train[b]), y_train[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(X_val), y_val).item()
        if val < best_val - 1e-4:
            best_val, since = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    return model


def _encode_windows(
    X_train: np.ndarray, X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, int]:
    all_vals = np.concatenate([X_train.ravel(), X_test.ravel(), np.array([PAD])])
    le = LabelEncoder().fit(all_vals)
    pad_id = int(le.transform([PAD])[0])
    Xtr = np.array([le.transform(row) for row in X_train])
    Xte = np.array([le.transform(row) for row in X_test])
    return Xtr, Xte, le, pad_id


def _eval_torch(
    model: nn.Module,
    X_te: torch.Tensor, y_te: torch.Tensor,
    coda_ids_test: list[str] | None,
    n_params: int,
) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(X_te)
        log_probs = torch.log_softmax(logits, dim=-1)
        lp = log_probs.gather(1, y_te.unsqueeze(1)).squeeze(1).cpu().numpy()
        pred = logits.argmax(-1)
        acc = float((pred == y_te).float().mean().item())
    bpt = float(-np.mean(lp / math.log(2)))
    r: dict = dict(
        accuracy=acc,
        bits_per_token=bpt,
        n_params=n_params,
    )
    if coda_ids_test is not None:
        r["coda_bpt"] = per_coda_bpt(lp / math.log(2), coda_ids_test)
    return r


def evaluate_sklearn_mlp(
    hidden: tuple,
    X_tr_oh: np.ndarray, y_tr: np.ndarray,
    X_te_oh: np.ndarray, y_te: np.ndarray,
    coda_ids_test: list[str] | None,
) -> dict:
    le = LabelEncoder().fit(np.concatenate([y_tr, y_te]))
    yti = le.transform(y_tr)
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden, activation="relu", solver="adam",
        max_iter=300, early_stopping=True, validation_fraction=0.1,
        random_state=0, n_iter_no_change=15,
    )
    mlp.fit(X_tr_oh, yti)
    pred_int = mlp.predict(X_te_oh)
    probs = mlp.predict_proba(X_te_oh)
    cls_to_col = {c: i for i, c in enumerate(le.inverse_transform(mlp.classes_))}
    p_true = np.array([
        probs[i, cls_to_col[t]] if t in cls_to_col else 1e-12
        for i, t in enumerate(y_te)
    ])
    lp2 = np.log2(np.maximum(p_true, 1e-12))
    r = dict(
        accuracy=float(np.mean(le.inverse_transform(pred_int) == y_te)),
        bits_per_token=float(-np.mean(lp2)),
        n_params=int(sum(p.size for p in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)),
    )
    if coda_ids_test is not None:
        r["coda_bpt"] = per_coda_bpt(lp2, coda_ids_test)
    return r


# ---------------------------------------------------------------------------
# One-fold evaluation
# ---------------------------------------------------------------------------


def evaluate_one_fold(
    seq_dict: dict[str, list[str]],
    train_ids: list[str],
    test_ids: list[str],
    k: int,
    bd_dict: dict[str, list[int]] | None = None,
) -> dict:
    """Run M0–M7 on one K-fold split.

    If bd_dict is supplied (morpheme_seq mode), also computes per-coda bpt
    via log-prob aggregation across coda boundaries.
    """
    has_coda_bd = bd_dict is not None

    if has_coda_bd:
        X_tr, y_tr, _ = build_windows_with_coda_ids(seq_dict, bd_dict, train_ids, k)
        X_te, y_te, coda_ids_te = build_windows_with_coda_ids(seq_dict, bd_dict, test_ids, k)
    else:
        X_tr, y_tr = build_windows(seq_dict, train_ids, k)
        X_te, y_te = build_windows(seq_dict, test_ids, k)
        coda_ids_te = None

    classes = sorted(set(y_tr.tolist()) | set(y_te.tolist()) | {PAD})
    V = len(classes)

    res = evaluate_baselines(
        y_tr, X_tr, y_te, X_te, V, classes, coda_ids_test=coda_ids_te)

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(np.array(X_tr, dtype=object))
    X_tr_oh = enc.transform(X_tr)
    X_te_oh = enc.transform(X_te)

    res["M3_MLP_S"] = evaluate_sklearn_mlp(
        (128, 64), X_tr_oh, y_tr, X_te_oh, y_te, coda_ids_te)
    res["M4_MLP_M"] = evaluate_sklearn_mlp(
        (256, 128), X_tr_oh, y_tr, X_te_oh, y_te, coda_ids_te)
    res["M5_MLP_L"] = evaluate_sklearn_mlp(
        (512, 256, 128), X_tr_oh, y_tr, X_te_oh, y_te, coda_ids_te)

    # Integer-encode context tokens for torch models
    Xtr_int, Xte_int, le, pad_id = _encode_windows(X_tr, X_te)
    le_y = LabelEncoder().fit(np.concatenate([y_tr, y_te]))
    ytr_t = torch.from_numpy(le_y.transform(y_tr)).long()
    yte_t = torch.from_numpy(le_y.transform(y_te)).long()
    Xtr_t = torch.from_numpy(Xtr_int).long()
    Xte_t = torch.from_numpy(Xte_int).long()
    V_full = len(le_y.classes_)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Xtr_t))
    n_val = max(64, int(0.1 * len(Xtr_t)))
    val_idx = torch.from_numpy(perm[:n_val])
    tr_idx = torch.from_numpy(perm[n_val:])

    m6 = EmbMLP(V_full, k, d=32, hidden=(256, 128))
    n6 = sum(p.numel() for p in m6.parameters())
    _train_torch(m6, Xtr_t[tr_idx], ytr_t[tr_idx], Xtr_t[val_idx], ytr_t[val_idx])
    res["M6_EmbMLP"] = _eval_torch(m6, Xte_t, yte_t, coda_ids_te, n6)

    m7 = MiniTransformer(V_full, k, d=64, n_layers=2, n_heads=4)
    n7 = sum(p.numel() for p in m7.parameters())
    _train_torch(m7, Xtr_t[tr_idx], ytr_t[tr_idx], Xtr_t[val_idx], ytr_t[val_idx])
    res["M7_MiniTfm"] = _eval_torch(m7, Xte_t, yte_t, coda_ids_te, n7)

    res["_meta"] = dict(
        n_train=int(len(X_tr)),
        n_test=int(len(X_te)),
        V=V,
        has_coda_bd=has_coda_bd,
    )
    return res


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tokenisation",
        choices=["ici_string", "morpheme_seq"],
        default="ici_string",
        help="'ici_string': one ICI-symbol-string token per coda (~1 916 vocab). "
             "'morpheme_seq': Morfessor sub-units with per-coda bpt aggregation.",
    )
    p.add_argument(
        "--k", type=int, default=CONTEXT_K,
        help=f"Context window K (default {CONTEXT_K}).",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "data" / "classified" / "whale_dialogues.csv",
    )
    args = p.parse_args(argv)

    tok = args.tokenisation
    out_json = OUT_DIR / f"predict_results_morpheme_{tok}.json"
    out_md = OUT_DIR / f"predict_results_morpheme_{tok}.md"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {tok} sequences from {args.csv} …")
    if tok == "ici_string":
        seq_dict = _ici_seq_dict(args.csv)
        bd_dict = None
    else:
        seq_dict, bd_dict = _morpheme_seq_dicts(args.csv)

    seq_ids = sorted(seq_dict.keys())
    print(f"  {len(seq_ids)} sequences, K={args.k}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_results: list[dict] = []

    for fi, (tr_idx, te_idx) in enumerate(kf.split(seq_ids)):
        train_ids = [seq_ids[i] for i in tr_idx]
        test_ids = [seq_ids[i] for i in te_idx]
        t0 = time.time()
        res = evaluate_one_fold(seq_dict, train_ids, test_ids, args.k, bd_dict)
        res["_meta"]["fold"] = fi
        res["_meta"]["seconds"] = round(time.time() - t0, 1)
        fold_results.append(res)
        model_keys = [k for k in res if k.startswith("M")]
        parts = []
        for m in model_keys:
            bpt_s = f"bpt={res[m]['bits_per_token']:.3f}"
            if "coda_bpt" in res[m]:
                bpt_s += f" coda_bpt={res[m]['coda_bpt']:.3f}"
            parts.append(f"{m}: {bpt_s} acc={res[m]['accuracy']:.3f}")
        print(f"fold {fi}:  " + "  ".join(parts) + f"  ({res['_meta']['seconds']}s)")

    model_keys = [k for k in fold_results[0] if k.startswith("M")]
    summary: dict[str, dict] = {}
    for m in model_keys:
        bits = [f[m]["bits_per_token"] for f in fold_results]
        accs = [f[m]["accuracy"] for f in fold_results]
        entry = dict(
            bits_per_token_mean=float(np.mean(bits)),
            bits_per_token_std=float(np.std(bits)),
            perplexity_mean=float(2 ** np.mean(bits)),
            accuracy_mean=float(np.mean(accs)),
            accuracy_std=float(np.std(accs)),
            n_params=fold_results[0][m].get("n_params"),
        )
        if "coda_bpt" in fold_results[0][m]:
            coda_bpts = [f[m]["coda_bpt"] for f in fold_results]
            entry["coda_bpt_mean"] = float(np.mean(coda_bpts))
            entry["coda_bpt_std"] = float(np.std(coda_bpts))
        summary[m] = entry

    V_used = fold_results[0]["_meta"]["V"]
    out = dict(
        tokenisation=tok,
        k=args.k,
        folds=fold_results,
        summary=summary,
        V=V_used,
    )
    out_json.write_text(json.dumps(out, indent=2, default=str))

    label_for = {
        "M0_majority": "majority (smoothed unigram)",
        "M1_markov1": "Markov-1",
        "M2_markov2": "Markov-2",
        "M3_MLP_S": "MLP-S (128, 64)",
        "M4_MLP_M": "MLP-M (256, 128)",
        "M5_MLP_L": "MLP-L (512, 256, 128)",
        "M6_EmbMLP": "Embedding-MLP (d=32, 256·128)",
        "M7_MiniTfm": "MiniTransformer (2L, 4h, d=64)",
    }
    has_coda = "coda_bpt_mean" in next(iter(summary.values()))
    bpt_col = "bits/token (morpheme)" if tok == "morpheme_seq" else "bits/token"

    L = [
        f"# 5-fold prediction — {tok} tokenisation",
        "",
        f"Tokenisation: **{tok}**.  "
        f"Vocabulary V = {V_used} (including PAD).  "
        f"Context K = {args.k} past tokens.  "
        "Metric = held-out cross-entropy in **bits/token** (log₂); lower is better.",
        "",
    ]
    if tok == "morpheme_seq":
        L += [
            "**per-coda bpt** aggregates morpheme log-probs within each coda's Morfessor "
            "segmentation (chain rule).  Directly comparable to the compound-token baseline "
            "bpt from `predict_results_unified.json`.",
            "",
        ]

    cols = ["# | model | params", bpt_col]
    if has_coda:
        cols.append("per-coda bits/token (↓)")
    cols += ["perplexity (↓)", "accuracy"]
    header = "| " + " | ".join(["#", "model", "params", bpt_col]
                                + (["per-coda bpt (↓)"] if has_coda else [])
                                + ["perplexity (↓)", "accuracy"]) + " |"
    sep = "|" + "|".join(["---"] + ["---:"] * (len(header.split("|")) - 3)) + "|"
    L += [header, sep]

    model_order = [m for m in label_for if m in summary]
    for i, m in enumerate(model_order):
        s = summary[m]
        params = f"{s['n_params']:,}" if s.get("n_params") else "—"
        idx_label = m.split("_", 1)[0]
        row = (
            f"| {idx_label} | {label_for[m]} | {params} | "
            f"{s['bits_per_token_mean']:.3f} ± {s['bits_per_token_std']:.3f} |"
        )
        if has_coda:
            row += f" {s.get('coda_bpt_mean', 0):.3f} ± {s.get('coda_bpt_std', 0):.3f} |"
        row += (
            f" {s['perplexity_mean']:.2f} |"
            f" {s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f} |"
        )
        L.append(row)

    L += ["", "## Per-fold bits/token", ""]
    fold_header = "| fold |" + "|".join(f" {label_for[m]} " for m in model_order) + "|"
    L += [fold_header, "|------|" + "|".join(["---:"] * len(model_order)) + "|"]
    for f in fold_results:
        L.append(
            f"| {f['_meta']['fold']} |"
            + "|".join(f" {f[m]['bits_per_token']:.3f} " for m in model_order)
            + "|"
        )

    best_m = min(summary, key=lambda m: summary[m]["bits_per_token_mean"])
    L += [
        "",
        "## Takeaway",
        "",
        f"- Best model: **{label_for[best_m]}** at "
        f"{summary[best_m]['bits_per_token_mean']:.3f} bits/token "
        f"(perplexity ≈ {summary[best_m]['perplexity_mean']:.2f}).",
    ]
    if has_coda and best_m in summary and "coda_bpt_mean" in summary[best_m]:
        L.append(
            f"- Per-coda bpt for best model: "
            f"**{summary[best_m]['coda_bpt_mean']:.3f} ± "
            f"{summary[best_m]['coda_bpt_std']:.3f}** bits/coda."
        )

    out_md.write_text("\n".join(L) + "\n")
    print(f"\nwrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
