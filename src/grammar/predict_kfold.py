"""
Held-out next-token prediction on the unified-corpus transformer CSV.

Ported from `~/Code/whale-gpt @ claude/whale-language-research-tEudI`'s
`scripts/7_predict_logloss.py`. Adapted to read
`data/classified/whale_dialogues.csv` directly (transformer input,
schema documented in docs/obsidian/transformer/corpus-design.md).

Single target = the primary whale's `Coda1` (rhythm_class integer).
Vocabulary is dynamic from the data — currently 131 distinct rhythm
classes (silence sentinel 98 only appears in Coda2 / DeltaTime, so it
does not enter the prediction target). The DeltaTime column is *not*
fed as input here — Stage 1 of the plan reproduces the prior protocol
on the new data with no other confounds.

Single metric = bits/token cross-entropy on the held-out 20 % of
sequences. Sequence-level 5-fold CV. Context K = 8 past codas
(truncated/padded with PAD).

Models:

  M0  Majority             constant prediction = train mode
  M1  Markov-1             P(y | last_token), Laplace-smoothed (alpha=0.5)
  M2  Markov-2             P(y | last 2 tokens), Laplace-smoothed
  M3  MLP-S    (128,64)    sklearn, one-hot last-K context
  M4  MLP-M    (256,128)   sklearn, one-hot last-K context
  M5  MLP-L    (512,256,128) sklearn, one-hot last-K context
  M6  EmbMLP                pytorch, 32-d learned token embeddings + MLP
  M7  MiniTfm  (2L, 4h)    pytorch, embedding + 2x self-attention block

Outputs go to outputs/grammar/predict_results_unified.{md,json}.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
TOKENS = ROOT / "data" / "classified" / "whale_dialogues.csv"
WHALE_INDEX = ROOT / "data" / "classified" / "whale_id_index.csv"
OUT_JSON = ROOT / "outputs" / "grammar" / "predict_results_unified.json"
OUT_MD = ROOT / "outputs" / "grammar" / "predict_results_unified.md"

CONTEXT_K = 8
N_FOLDS = 5
TARGET = "Coda"
PAD = "<PAD>"
PAD_INT = -1  # sentinel for integer feature streams (Whale, Sync, Orn, has_ts)
PAD_FLOAT = 0.0  # sentinel for continuous streams (Duration, TimeDelta_log)

# whale-gpt-style continuous DT encoding, applied at model-input time.
# log(0.1 + max(dt, 0)) compresses the long tail and gives more
# resolution to short gaps where the action is. Hersh rows carry
# DeltaTime = -1 (sentinel); has_timestamps separately tells the model
# the value is uninformative on those rows, so we just emit log(0.1)
# (≈ -2.3) as a stable filler.
DT_LOG_OFFSET = 0.1


def _encode_dt_log(dt_value: float, has_ts: int) -> float:
    """log(0.1 + dt) for present rows; log(0.1) sentinel otherwise."""
    if not has_ts or dt_value < 0:
        return float(np.log(DT_LOG_OFFSET))
    return float(np.log(DT_LOG_OFFSET + max(dt_value, 0.0)))


# When True, training mode is fast/CV-friendly: AdamW + constant lr=1e-3
# + bs=128 + 80 epochs + early stopping (patience 10). When False,
# training mode is slow/accurate matching whale-gpt's recipe: AdamW +
# CosineAnnealingLR + bs=1000 + 5000 epochs + dropout 0.2 + no ES. Slow
# mode multiplies wall-time per fold by ~30×; only flip for a final run.
FAST_MODE = True


def per_seq(tokens: pd.DataFrame, col: str) -> dict[str, list[str]]:
    """Group rows by `sequenceId` (string), preserve `itemPosition` order,
    cast labels to strings so sklearn label encoders see a consistent dtype."""
    return {
        str(seq_id): [str(v) for v in g.sort_values("itemPosition")[col].tolist()]
        for seq_id, g in tokens.groupby("sequenceId")
    }


def per_seq_dt(tokens: pd.DataFrame) -> dict[str, list[float]]:
    """Per-sequence DeltaTime list (parallel to per_seq(..., 'Coda1'))."""
    return {
        str(seq_id): [float(v) for v in g.sort_values("itemPosition")["DeltaTime"].tolist()]
        for seq_id, g in tokens.groupby("sequenceId")
    }


def build_windows(seq_dict, seq_ids, k):
    X, y = [], []
    for sid in seq_ids:
        s = seq_dict[sid]
        for i in range(1, len(s)):
            ctx = s[max(0, i - k) : i]
            ctx = [PAD] * (k - len(ctx)) + ctx
            X.append(ctx)
            y.append(s[i])
    return np.array(X, dtype=object), np.array(y, dtype=object)


# Legacy DT encoding (kept for the old B0–B4 ablation): two scalars
# per position as (log_dt, has_timestamps). Matches whale-gpt's
# np.log(0.1 + time_delta) recipe.
def _encode_dt(dt_value: float) -> tuple[float, float]:
    """(dt_log, has_timestamps) per position. Hersh & PAD → log(0.1)."""
    if dt_value < 0:
        return float(np.log(DT_LOG_OFFSET)), 0.0
    return float(np.log(DT_LOG_OFFSET + max(dt_value, 0.0))), 1.0


def build_windows_with_dt(seq_dict, dt_dict, seq_ids, k):
    """Like build_windows but also returns a parallel (n, k, 2) array of
    (dt_log, has_timestamps) for the K context positions. PAD positions
    get the same sentinel as missing-timestamp rows: log(0.1), 0."""
    X, y, DT = [], [], []
    pad_dt = (float(np.log(DT_LOG_OFFSET)), 0.0)
    for sid in seq_ids:
        s = seq_dict[sid]
        ds = dt_dict[sid]
        for i in range(1, len(s)):
            ctx = s[max(0, i - k) : i]
            ctx_dt = ds[max(0, i - k) : i]
            pad_n = k - len(ctx)
            ctx = [PAD] * pad_n + ctx
            dt_feats = [pad_dt] * pad_n + [_encode_dt(d) for d in ctx_dt]
            X.append(ctx)
            y.append(s[i])
            DT.append(dt_feats)
    return (
        np.array(X, dtype=object),
        np.array(y, dtype=object),
        np.array(DT, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Multi-channel windowing for the redesigned model (R2).
#
# Each sequence is converted to seven parallel per-position streams:
#   whale (int), coda (int), orn (int), sync (int), dur (float),
#   td_log (float), has_ts (int).
# Integer streams use PAD_INT (-1) for PAD positions. Continuous streams
# use PAD_FLOAT (0.0). Targets are emitted as a dict for multi-task loss.
# ---------------------------------------------------------------------------


def per_seq_multi(tokens: pd.DataFrame, whale2id: dict[str, int],
                  coda2id: dict[int, int]) -> dict[str, dict[str, list]]:
    """Group rows by ``sequenceId`` and return per-stream sequences as ints
    (whale, coda) / floats (duration, td_log) / ints (orn, sync, has_ts).

    Whale and Coda are mapped through pre-computed dicts so the same
    integer encoding is used across folds.
    """
    out: dict[str, dict[str, list]] = {}
    for seq_id, g in tokens.groupby("sequenceId"):
        g = g.sort_values("itemPosition")
        whales = [int(whale2id[str(w)]) for w in g["Whale"]]
        codas = [int(coda2id[int(c)]) for c in g["Coda"]]
        orn = [int(o) for o in g["Ornamentation"]]
        sync = [int(s) for s in g["Synchrony"]]
        dur = [float(d) for d in g["Duration"]]
        has_ts = [int(h) for h in g["has_timestamps"]]
        td_log = [
            _encode_dt_log(float(dt), int(h))
            for dt, h in zip(g["TimeDelta"], g["has_timestamps"])
        ]
        out[str(seq_id)] = dict(
            whale=whales, coda=codas, orn=orn, sync=sync,
            dur=dur, td_log=td_log, has_ts=has_ts,
        )
    return out


def build_multi_windows(per_seq_data, seq_ids, k):
    """Slide a stride-1 K-window over each sequence. Returns dict of
    parallel arrays (one per channel) for inputs, and dict of arrays
    for targets (the value at position i, predicted from positions
    i-K..i-1)."""
    chans = ("whale", "coda", "orn", "sync", "dur", "td_log", "has_ts")
    X = {c: [] for c in chans}
    Y = {c: [] for c in chans}
    int_chans = {"whale", "coda", "orn", "sync", "has_ts"}
    for sid in seq_ids:
        seq = per_seq_data[sid]
        L = len(seq["coda"])
        for i in range(1, L):
            for c in chans:
                stream = seq[c]
                ctx = stream[max(0, i - k) : i]
                pad_n = k - len(ctx)
                pad_val = PAD_INT if c in int_chans else PAD_FLOAT
                X[c].append([pad_val] * pad_n + list(ctx))
                Y[c].append(stream[i])
    X_arr = {c: np.array(X[c], dtype=np.int64 if c in int_chans
                          else np.float32) for c in chans}
    Y_arr = {c: np.array(Y[c], dtype=np.int64 if c in int_chans
                          else np.float32) for c in chans}
    return X_arr, Y_arr


def bits_per_token(p_true: np.ndarray) -> float:
    eps = 1e-12
    return float(-np.mean(np.log2(np.maximum(p_true, eps))))


def evaluate_baselines(y_train, X_train, y_test, X_test, vocab_size, classes):
    out = {}

    train_counts = Counter(y_train.tolist())
    majority = train_counts.most_common(1)[0][0]
    n_train = len(y_train)
    smoothing = 0.5
    p_uniform_smoothed = np.array(
        [
            (train_counts.get(t, 0) + smoothing) / (n_train + smoothing * vocab_size)
            for t in y_test
        ]
    )
    out["M0_majority"] = dict(
        accuracy=float(np.mean(y_test == majority)),
        bits_per_token=bits_per_token(p_uniform_smoothed),
    )

    def markov_k(k):
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
                    n_train + smoothing * vocab_size
                )
                preds.append(majority)
            else:
                denom = sum(c.values()) + smoothing * vocab_size
                p_true[i] = (c.get(y_test[i], 0) + smoothing) / denom
                preds.append(c.most_common(1)[0][0])
        return dict(
            accuracy=float(np.mean(np.array(preds) == y_test)),
            bits_per_token=bits_per_token(p_true),
        )

    out["M1_markov1"] = markov_k(1)
    out["M2_markov2"] = markov_k(2)
    return out


def evaluate_sklearn_mlp(name, hidden, X_train_oh, y_train, X_test_oh, y_test, classes):
    le = LabelEncoder().fit(np.concatenate([y_train, y_test]))
    yti = le.transform(y_train)
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=0,
        n_iter_no_change=15,
    )
    mlp.fit(X_train_oh, yti)
    pred_int = mlp.predict(X_test_oh)
    probs = mlp.predict_proba(X_test_oh)
    cls_to_col = {c: i for i, c in enumerate(le.inverse_transform(mlp.classes_))}
    p_true = np.array(
        [probs[i, cls_to_col[t]] if t in cls_to_col else 1e-12 for i, t in enumerate(y_test)]
    )
    return dict(
        accuracy=float(np.mean(le.inverse_transform(pred_int) == y_test)),
        bits_per_token=bits_per_token(p_true),
        n_params=int(sum(p.size for p in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)),
    )


class EmbMLP(nn.Module):
    def __init__(self, V, k, d=32, hidden=(256, 128)):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        layers = []
        in_dim = d * k
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(0.2)]
            in_dim = h
        layers.append(nn.Linear(in_dim, V))
        self.head = nn.Sequential(*layers)

    def forward(self, x):
        e = self.emb(x).flatten(1)
        return self.head(e)


class MiniTransformer(nn.Module):
    def __init__(self, V, k, d=64, n_layers=2, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(k, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4 * d, dropout=0.1,
            batch_first=True, activation="gelu",
        )
        self.tfm = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d, V)
        self.k = k

    def forward(self, x):
        B, K = x.shape
        pos_ids = torch.arange(K, device=x.device).expand(B, K)
        h = self.emb(x) + self.pos(pos_ids)
        h = self.tfm(h)
        return self.head(h[:, -1])


class MiniTransformerDT(nn.Module):
    """Same as MiniTransformer but adds a learned linear projection of the
    per-position (dt_log, has_timestamps) features to the input embedding."""

    def __init__(self, V, k, d=64, n_layers=2, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(k, d)
        self.dt_proj = nn.Linear(2, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4 * d, dropout=0.1,
            batch_first=True, activation="gelu",
        )
        self.tfm = nn.TransformerEncoder(encoder_layer, n_layers)
        self.head = nn.Linear(d, V)
        self.k = k

    def forward(self, x, dt):
        B, K = x.shape
        pos_ids = torch.arange(K, device=x.device).expand(B, K)
        h = self.emb(x) + self.pos(pos_ids) + self.dt_proj(dt)
        h = self.tfm(h)
        return self.head(h[:, -1])


class MultiChannelMultiTask(nn.Module):
    """whale-gpt-style structured-tabular transformer.

    Per position the input is the concatenation of seven learned
    embeddings/projections summing to ``d_model``:

      Whale          → nn.Embedding(V_whale, 6)
      Coda           → nn.Embedding(V_coda,  8)
      Ornamentation  → nn.Embedding(2,       3)
      Synchrony      → nn.Embedding(2,       3)
      Duration       → nn.Linear(1,          4)   continuous
      TimeDelta_log  → nn.Linear(1,          4)   continuous
      has_timestamps → nn.Embedding(2,       4)
                                       total = 32 = d_model

    Plus a learned positional embedding.

    Output heads:
      Whale (CE), Coda (CE), Ornamentation (CE), Synchrony (CE),
      Duration (L1 regression), TimeDelta (L1 regression on log-dt)

    The ``-1`` PAD value for any categorical input is mapped to vocab
    index 0 (a reserved PAD slot in each embedding) before the
    embedding lookup; PAD-position features therefore consistently
    project to a single learned PAD vector per channel.
    """

    EMB_DIMS = dict(
        whale=6, coda=8, orn=3, sync=3, dur=4, td=4, has_ts=4,
    )  # sums to 32

    def __init__(self, V_whale, V_coda, k, d=32, n_layers=3, n_heads=8,
                 dropout=0.1):
        super().__init__()
        assert sum(self.EMB_DIMS.values()) == d, (
            f"emb dims sum to {sum(self.EMB_DIMS.values())}, expected d={d}"
        )
        # +1 for the PAD slot at index 0 (we shift real ids by +1)
        self.emb_whale = nn.Embedding(V_whale + 1, self.EMB_DIMS["whale"])
        self.emb_coda = nn.Embedding(V_coda + 1, self.EMB_DIMS["coda"])
        self.emb_orn = nn.Embedding(3, self.EMB_DIMS["orn"])  # PAD + {0,1}
        self.emb_sync = nn.Embedding(3, self.EMB_DIMS["sync"])
        self.emb_has_ts = nn.Embedding(3, self.EMB_DIMS["has_ts"])
        self.proj_dur = nn.Linear(1, self.EMB_DIMS["dur"])
        self.proj_td = nn.Linear(1, self.EMB_DIMS["td"])
        self.pos = nn.Embedding(k, d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4 * d, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.tfm = nn.TransformerEncoder(encoder_layer, n_layers)

        # Multi-task heads: one per target column.
        self.head_whale = nn.Linear(d, V_whale + 1)
        self.head_coda = nn.Linear(d, V_coda + 1)
        self.head_orn = nn.Linear(d, 2)
        self.head_sync = nn.Linear(d, 2)
        self.head_dur = nn.Linear(d, 1)
        self.head_td = nn.Linear(d, 1)
        self.k = k
        self.V_whale = V_whale
        self.V_coda = V_coda

    def encode(self, whale, coda, orn, sync, dur, td_log, has_ts):
        """Build d_model-wide per-position features by concatenation."""
        # Shift integer ids by +1 so PAD (-1 in input streams) → index 0.
        e_w = self.emb_whale((whale + 1).long())
        e_c = self.emb_coda((coda + 1).long())
        e_o = self.emb_orn((orn + 1).long())
        e_s = self.emb_sync((sync + 1).long())
        e_h = self.emb_has_ts((has_ts + 1).long())
        e_d = self.proj_dur(dur.unsqueeze(-1))
        e_t = self.proj_td(td_log.unsqueeze(-1))
        h = torch.cat([e_w, e_c, e_o, e_s, e_d, e_t, e_h], dim=-1)
        B, K, _ = h.shape
        pos_ids = torch.arange(K, device=h.device).expand(B, K)
        return h + self.pos(pos_ids)

    def forward(self, whale, coda, orn, sync, dur, td_log, has_ts):
        h = self.encode(whale, coda, orn, sync, dur, td_log, has_ts)
        h = self.tfm(h)
        last = h[:, -1]
        return dict(
            whale=self.head_whale(last),
            coda=self.head_coda(last),
            orn=self.head_orn(last),
            sync=self.head_sync(last),
            dur=self.head_dur(last).squeeze(-1),
            td=self.head_td(last).squeeze(-1),
        )


def train_torch(model, X_train, y_train, X_val, y_val, epochs=80, lr=1e-3, bs=128, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    n = X_train.shape[0]
    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    patience, since = 10, 0
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i : i + bs]
            xb, yb = X_train[b], y_train[b]
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
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


def train_torch_dt(model, X_train, DT_train, y_train, X_val, DT_val, y_val,
                   epochs=80, lr=1e-3, bs=128, seed=0):
    """train_torch variant for DT-aware models. Forward is model(x, dt)."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    n = X_train.shape[0]
    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    patience, since = 10, 0
    for _ in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i : i + bs]
            xb, dtb, yb = X_train[b], DT_train[b], y_train[b]
            opt.zero_grad()
            loss = loss_fn(model(xb, dtb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(X_val, DT_val), y_val).item()
        if val < best_val - 1e-4:
            best_val, since = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    return model


def evaluate_torch_model(name, model_factory, X_train, y_train, X_test, y_test, vocab_size, classes, seed=0):
    le = LabelEncoder().fit(np.concatenate([y_train, y_test]))
    pad_id = le.transform([PAD])[0] if PAD in le.classes_ else None
    if pad_id is None:
        all_classes = list(le.classes_) + [PAD]
        le2 = LabelEncoder().fit(np.array(all_classes))
        pad_id = le2.transform([PAD])[0]
        Xtr_int = np.array([[le2.transform([c])[0] if c in le2.classes_ else pad_id for c in row] for row in X_train])
        Xte_int = np.array([[le2.transform([c])[0] if c in le2.classes_ else pad_id for c in row] for row in X_test])
        ytr_int = le2.transform(y_train)
        yte_int = le2.transform(y_test)
        V_full = len(le2.classes_)
    else:
        Xtr_int = np.array([[le.transform([c])[0] if c in le.classes_ else pad_id for c in row] for row in X_train])
        Xte_int = np.array([[le.transform([c])[0] if c in le.classes_ else pad_id for c in row] for row in X_test])
        ytr_int = le.transform(y_train)
        yte_int = le.transform(y_test)
        V_full = len(le.classes_)

    Xtr_t = torch.from_numpy(Xtr_int).long()
    Xte_t = torch.from_numpy(Xte_int).long()
    ytr_t = torch.from_numpy(ytr_int).long()
    yte_t = torch.from_numpy(yte_int).long()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr_t))
    n_val = max(64, int(0.1 * len(Xtr_t)))
    val_idx = torch.from_numpy(perm[:n_val])
    tr_idx = torch.from_numpy(perm[n_val:])

    model = model_factory(V_full)
    n_params = sum(p.numel() for p in model.parameters())

    train_torch(
        model, Xtr_t[tr_idx], ytr_t[tr_idx], Xtr_t[val_idx], ytr_t[val_idx], seed=seed
    )

    model.eval()
    with torch.no_grad():
        logits = model(Xte_t)
        log_probs = torch.log_softmax(logits, dim=-1)
        ll = log_probs.gather(1, yte_t.unsqueeze(1)).squeeze(1)
        bits = -ll.mean().item() / math.log(2)
        pred = logits.argmax(-1)
        acc = (pred == yte_t).float().mean().item()
    return dict(accuracy=float(acc), bits_per_token=float(bits), n_params=int(n_params))


def _encode_window_ids(X_train, X_test):
    """Map context tokens to integer ids; ensures PAD has an id even when
    only test sees PAD-only positions."""
    le = LabelEncoder().fit(np.concatenate([X_train.ravel(), X_test.ravel(), np.array([PAD])]))
    pad_id = le.transform([PAD])[0]
    Xtr_int = np.array([le.transform(row) for row in X_train])
    Xte_int = np.array([le.transform(row) for row in X_test])
    return Xtr_int, Xte_int, le, pad_id


def evaluate_torch_model_dt(name, model_factory, X_train, DT_train, y_train,
                            X_test, DT_test, y_test, vocab_size, classes, seed=0):
    """Like evaluate_torch_model but feeds DT input alongside token ids."""
    le = LabelEncoder().fit(np.concatenate([y_train, y_test]))
    pad_id = le.transform([PAD])[0] if PAD in le.classes_ else None
    if pad_id is None:
        all_classes = list(le.classes_) + [PAD]
        le2 = LabelEncoder().fit(np.array(all_classes))
        pad_id = le2.transform([PAD])[0]
        Xtr_int = np.array([[le2.transform([c])[0] if c in le2.classes_ else pad_id for c in row] for row in X_train])
        Xte_int = np.array([[le2.transform([c])[0] if c in le2.classes_ else pad_id for c in row] for row in X_test])
        ytr_int = le2.transform(y_train)
        yte_int = le2.transform(y_test)
        V_full = len(le2.classes_)
    else:
        Xtr_int = np.array([[le.transform([c])[0] if c in le.classes_ else pad_id for c in row] for row in X_train])
        Xte_int = np.array([[le.transform([c])[0] if c in le.classes_ else pad_id for c in row] for row in X_test])
        ytr_int = le.transform(y_train)
        yte_int = le.transform(y_test)
        V_full = len(le.classes_)

    Xtr_t = torch.from_numpy(Xtr_int).long()
    Xte_t = torch.from_numpy(Xte_int).long()
    DTtr_t = torch.from_numpy(DT_train).float()
    DTte_t = torch.from_numpy(DT_test).float()
    ytr_t = torch.from_numpy(ytr_int).long()
    yte_t = torch.from_numpy(yte_int).long()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr_t))
    n_val = max(64, int(0.1 * len(Xtr_t)))
    val_idx = torch.from_numpy(perm[:n_val])
    tr_idx = torch.from_numpy(perm[n_val:])

    model = model_factory(V_full)
    n_params = sum(p.numel() for p in model.parameters())

    train_torch_dt(
        model,
        Xtr_t[tr_idx], DTtr_t[tr_idx], ytr_t[tr_idx],
        Xtr_t[val_idx], DTtr_t[val_idx], ytr_t[val_idx],
        seed=seed,
    )

    model.eval()
    with torch.no_grad():
        logits = model(Xte_t, DTte_t)
        log_probs = torch.log_softmax(logits, dim=-1)
        ll = log_probs.gather(1, yte_t.unsqueeze(1)).squeeze(1)
        bits = -ll.mean().item() / math.log(2)
        pred = logits.argmax(-1)
        acc = (pred == yte_t).float().mean().item()
    return dict(accuracy=float(acc), bits_per_token=float(bits), n_params=int(n_params))


def evaluate_ablation_fold(seq_dict, dt_dict, train_ids, test_ids):
    """Evaluate the five ablation variants on one fold. Returns the same
    {model_key: {bits_per_token, accuracy, n_params}} structure."""
    res: dict[str, dict] = {}

    # B0..B3 share K=8 windows
    X_train, y_train = build_windows(seq_dict, train_ids, CONTEXT_K)
    X_test, y_test = build_windows(seq_dict, test_ids, CONTEXT_K)
    classes_k8 = sorted(set(y_train.tolist()) | set(y_test.tolist()) | {PAD})
    V_k8 = len(classes_k8)

    Xdt_train_k8, ydt_train_k8, DT_train_k8 = build_windows_with_dt(seq_dict, dt_dict, train_ids, CONTEXT_K)
    Xdt_test_k8, ydt_test_k8, DT_test_k8 = build_windows_with_dt(seq_dict, dt_dict, test_ids, CONTEXT_K)

    # B0: 2L, d=64, K=8, no DT
    res["B0_2L_d64_K8"] = evaluate_torch_model(
        "MiniTfm-2L", lambda V: MiniTransformer(V, CONTEXT_K, d=64, n_layers=2, n_heads=4),
        X_train, y_train, X_test, y_test, V_k8, classes_k8,
    )
    # B1: 2L, d=64, K=8, +DT
    res["B1_2L_d64_K8_DT"] = evaluate_torch_model_dt(
        "MiniTfmDT-2L",
        lambda V: MiniTransformerDT(V, CONTEXT_K, d=64, n_layers=2, n_heads=4),
        Xdt_train_k8, DT_train_k8, ydt_train_k8,
        Xdt_test_k8, DT_test_k8, ydt_test_k8,
        V_k8, classes_k8,
    )
    # B2: 4L, d=64, K=8, no DT
    res["B2_4L_d64_K8"] = evaluate_torch_model(
        "MiniTfm-4L", lambda V: MiniTransformer(V, CONTEXT_K, d=64, n_layers=4, n_heads=4),
        X_train, y_train, X_test, y_test, V_k8, classes_k8,
    )
    # B3: 4L, d=64, K=8, +DT
    res["B3_4L_d64_K8_DT"] = evaluate_torch_model_dt(
        "MiniTfmDT-4L",
        lambda V: MiniTransformerDT(V, CONTEXT_K, d=64, n_layers=4, n_heads=4),
        Xdt_train_k8, DT_train_k8, ydt_train_k8,
        Xdt_test_k8, DT_test_k8, ydt_test_k8,
        V_k8, classes_k8,
    )

    # B4: whale-gpt main-branch shape — 3L, 8h, d=32, K=25, +DT
    K_LONG = 25
    Xdt_train_k25, ydt_train_k25, DT_train_k25 = build_windows_with_dt(seq_dict, dt_dict, train_ids, K_LONG)
    Xdt_test_k25, ydt_test_k25, DT_test_k25 = build_windows_with_dt(seq_dict, dt_dict, test_ids, K_LONG)
    classes_k25 = sorted(set(ydt_train_k25.tolist()) | set(ydt_test_k25.tolist()) | {PAD})
    V_k25 = len(classes_k25)
    res["B4_3L_d32_K25_DT"] = evaluate_torch_model_dt(
        "MiniTfmDT-3L-d32-K25",
        lambda V: MiniTransformerDT(V, K_LONG, d=32, n_layers=3, n_heads=8),
        Xdt_train_k25, DT_train_k25, ydt_train_k25,
        Xdt_test_k25, DT_test_k25, ydt_test_k25,
        V_k25, classes_k25,
    )

    res["_meta"] = dict(
        n_train_k8=int(len(X_train)), n_test_k8=int(len(X_test)),
        n_train_k25=int(len(Xdt_train_k25)), n_test_k25=int(len(Xdt_test_k25)),
        V_k8=V_k8, V_k25=V_k25,
    )
    return res


# ---------------------------------------------------------------------------
# R0 vs R2 redesign-mode evaluator.
#
# R0 control = B4-mimic: single-channel Coda + DT, 3L 8h d=32 K=25.
# R2 redesign = multi-channel multi-task: per-column embeddings of
#               (Whale, Coda, Orn, Sync, Dur, TimeDelta_log, has_timestamps)
#               concatenated to d_model=32; multi-task heads with CE on
#               Whale/Coda/Orn/Sync + L1 on Dur/TimeDelta. 3L 8h K=25.
#
# Reported metric is the **Coda head's bpt and accuracy**, so R0 and R2
# are directly comparable on the same target column. R2's auxiliary
# heads contribute to the loss but not to the headline metric.
# ---------------------------------------------------------------------------


def _train_multitask(model, train_inputs, train_targets, val_inputs,
                     val_targets, fast: bool, seed: int = 0,
                     single_task: bool = False):
    """Multi-task training loop for ``MultiChannelMultiTask``. ``fast=True``
    uses AdamW + constant lr=1e-3 + bs=128 + 80 epochs + ES patience 10.
    ``fast=False`` uses AdamW + CosineAnnealingLR + bs=1000 + 5000 epochs
    + dropout 0.2 + no ES.

    ``single_task=True`` zeros the auxiliary-head loss weights so the
    encoder is supervised only by the Coda CE. Used by the R1 variant
    to isolate the effect of multi-channel inputs from the effect of
    multi-task supervision.
    """
    torch.manual_seed(seed)

    # Loss weights: CE losses average ~3-5 nats; L1 on duration is
    # ~0.3-0.5; L1 on log-dt is ~1-3. Down-weight the regression
    # targets so they don't drown out the categorical signal.
    if single_task:
        weights = dict(whale=0.0, coda=1.0, orn=0.0, sync=0.0,
                       dur=0.0, td=0.0)
    else:
        weights = dict(whale=1.0, coda=1.0, orn=0.5, sync=0.5,
                       dur=0.2, td=0.2)
    ce = nn.CrossEntropyLoss()
    l1 = nn.L1Loss()

    def _step_loss(out, tgt):
        # Whale and Coda heads emit V+1 logits with index 0 reserved for
        # the PAD slot used in input embeddings. Shift targets by +1 so
        # the loss aligns with the same convention used at eval time.
        # Orn / Sync heads have no PAD slot (orn/sync are 0/1 only) so
        # their targets pass through unchanged.
        loss = (
            weights["whale"] * ce(out["whale"], (tgt["whale"] + 1).long())
            + weights["coda"] * ce(out["coda"], (tgt["coda"] + 1).long())
            + weights["orn"] * ce(out["orn"], tgt["orn"].clamp(min=0).long())
            + weights["sync"] * ce(out["sync"], tgt["sync"].clamp(min=0).long())
            + weights["dur"] * l1(out["dur"], tgt["dur"])
            + weights["td"] * l1(out["td"], tgt["td_log"])
        )
        return loss

    if fast:
        epochs, lr, bs, patience = 80, 1e-3, 128, 10
        sched = None
    else:
        epochs, lr, bs, patience = 5000, 1e-4, 1000, None
        # cosine over the full epoch budget.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if not fast:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=1e-5)
    n = train_inputs["coda"].shape[0]

    def _coda_val_loss():
        # Same +1 shift as in the training loss (output position 0 is
        # the PAD slot reserved by the input embedding).
        model.eval()
        with torch.no_grad():
            out = model(**val_inputs)
            return float(ce(out["coda"], (val_targets["coda"] + 1).long()).item())

    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    since = 0
    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(n)
        for i in range(0, n, bs):
            b = idx[i : i + bs]
            inp = {k: v[b] for k, v in train_inputs.items()}
            tgt = {k: v[b] for k, v in train_targets.items()}
            opt.zero_grad()
            loss = _step_loss(model(**inp), tgt)
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()
        val = _coda_val_loss()
        if val < best_val - 1e-4:
            best_val, since = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if patience is not None and since >= patience:
                break
    model.load_state_dict(best_state)
    return model


def _to_tensors(d, int_keys=("whale", "coda", "orn", "sync", "has_ts"),
                 float_keys=("dur", "td_log")):
    """Convert numpy stream dict to a tensor dict matching the model
    forward signature."""
    out = {}
    for k in int_keys:
        out[k] = torch.from_numpy(d[k]).long()
    for k in float_keys:
        out[k] = torch.from_numpy(d[k]).float()
    return out


def _eval_multichannel(model, test_inputs, test_targets):
    """Coda-head bpt + accuracy on test windows. Shared by R1 and R2."""
    model.eval()
    with torch.no_grad():
        out = model(**test_inputs)
        coda_logits = out["coda"]
        log_probs = torch.log_softmax(coda_logits, dim=-1)
        coda_targets = (test_targets["coda"] + 1).long()
        ll = log_probs.gather(1, coda_targets.unsqueeze(1)).squeeze(1)
        bits = -ll.mean().item() / math.log(2)
        pred = coda_logits.argmax(-1)
        acc = (pred == coda_targets).float().mean().item()
    return float(acc), float(bits)


def evaluate_redesign_fold(per_seq_data, train_ids, test_ids,
                           V_whale, V_coda, fast: bool,
                           variants: tuple[str, ...] = ("R0", "R1", "R2")):
    """Run a subset of {R0 control, R1 multi-channel single-task, R2
    multi-channel multi-task} on one fold. All variants report
    Coda-head bits/token + accuracy."""
    K = 25
    res: dict[str, dict] = {}

    n_train = n_test = None

    # ---------------- R0 control: single-channel Coda + DT, K=25 ----------
    if "R0" in variants:
        # Build legacy windows (string Coda tokens + (dt_log, has_ts)).
        seq_dict = {sid: [str(c) for c in d["coda"]] for sid, d in per_seq_data.items()}
        dt_dict = {sid: [
            (float(np.exp(d["td_log"][i]) - DT_LOG_OFFSET) if d["has_ts"][i] else -1.0)
            for i in range(len(d["coda"]))
        ] for sid, d in per_seq_data.items()}
        Xdt_tr, ydt_tr, DT_tr = build_windows_with_dt(seq_dict, dt_dict, train_ids, K)
        Xdt_te, ydt_te, DT_te = build_windows_with_dt(seq_dict, dt_dict, test_ids, K)
        classes = sorted(set(ydt_tr.tolist()) | set(ydt_te.tolist()) | {PAD})
        V_k = len(classes)
        res["R0_control_B4mimic"] = evaluate_torch_model_dt(
            "R0-single-chan",
            lambda V: MiniTransformerDT(V, K, d=32, n_layers=3, n_heads=8),
            Xdt_tr, DT_tr, ydt_tr, Xdt_te, DT_te, ydt_te, V_k, classes,
        )

    # ---------------- Multi-channel windowing (shared by R1 + R2) ---------
    needs_multi = ("R1" in variants) or ("R2" in variants)
    if needs_multi:
        Xtr, Ytr = build_multi_windows(per_seq_data, train_ids, K)
        Xte, Yte = build_multi_windows(per_seq_data, test_ids, K)
        n_train = int(Xtr["coda"].shape[0])
        n_test = int(Xte["coda"].shape[0])

        rng = np.random.default_rng(0)
        n_tr = Xtr["coda"].shape[0]
        perm = rng.permutation(n_tr)
        n_val = max(64, int(0.1 * n_tr))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        train_inputs = _to_tensors({k: v[tr_idx] for k, v in Xtr.items()})
        train_targets = _to_tensors({k: v[tr_idx] for k, v in Ytr.items()})
        val_inputs = _to_tensors({k: v[val_idx] for k, v in Xtr.items()})
        val_targets = _to_tensors({k: v[val_idx] for k, v in Ytr.items()})
        test_inputs = _to_tensors(Xte)
        test_targets = _to_tensors(Yte)

    # ---------------- R1: multi-channel input + Coda CE only --------------
    if "R1" in variants:
        m_r1 = MultiChannelMultiTask(
            V_whale=V_whale, V_coda=V_coda, k=K, d=32,
            n_layers=3, n_heads=8,
            dropout=(0.2 if not fast else 0.1),
        )
        n_params = sum(p.numel() for p in m_r1.parameters())
        _train_multitask(
            m_r1,
            train_inputs, train_targets,
            val_inputs, val_targets,
            fast=fast,
            single_task=True,
        )
        acc, bits = _eval_multichannel(m_r1, test_inputs, test_targets)
        res["R1_multichannel_singletask"] = dict(
            accuracy=acc,
            bits_per_token=bits,
            n_params=int(n_params),
        )

    # ---------------- R2: multi-channel input + multi-task heads ----------
    if "R2" in variants:
        m_r2 = MultiChannelMultiTask(
            V_whale=V_whale, V_coda=V_coda, k=K, d=32,
            n_layers=3, n_heads=8,
            dropout=(0.2 if not fast else 0.1),
        )
        n_params = sum(p.numel() for p in m_r2.parameters())
        _train_multitask(
            m_r2,
            train_inputs, train_targets,
            val_inputs, val_targets,
            fast=fast,
            single_task=False,
        )
        acc, bits = _eval_multichannel(m_r2, test_inputs, test_targets)
        res["R2_multichannel_multitask"] = dict(
            accuracy=acc,
            bits_per_token=bits,
            n_params=int(n_params),
        )

    res["_meta"] = dict(
        n_train=n_train,
        n_test=n_test,
        V_coda=int(V_coda),
        V_whale=int(V_whale),
        K=K,
        fast_mode=bool(fast),
        variants=list(variants),
    )
    return res


def evaluate_one_fold(seq_dict, train_ids, test_ids, k, quick=False):
    X_train, y_train = build_windows(seq_dict, train_ids, k)
    X_test, y_test = build_windows(seq_dict, test_ids, k)
    classes = sorted(set(y_train.tolist()) | set(y_test.tolist()) | {PAD})
    V = len(classes)

    res = evaluate_baselines(y_train, X_train, y_test, X_test, V, classes)
    if quick:
        res["_meta"] = dict(n_train=int(len(X_train)), n_test=int(len(X_test)), V=V)
        return res

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(np.array(X_train, dtype=object))
    Xtr_oh = enc.transform(X_train)
    Xte_oh = enc.transform(X_test)

    res["M3_MLP_S"] = evaluate_sklearn_mlp("MLP-S", (128, 64), Xtr_oh, y_train, Xte_oh, y_test, classes)
    res["M4_MLP_M"] = evaluate_sklearn_mlp("MLP-M", (256, 128), Xtr_oh, y_train, Xte_oh, y_test, classes)
    res["M5_MLP_L"] = evaluate_sklearn_mlp("MLP-L", (512, 256, 128), Xtr_oh, y_train, Xte_oh, y_test, classes)

    res["M6_EmbMLP"] = evaluate_torch_model(
        "EmbMLP", lambda V: EmbMLP(V, k, d=32, hidden=(256, 128)),
        X_train, y_train, X_test, y_test, V, classes,
    )
    res["M7_MiniTfm"] = evaluate_torch_model(
        "MiniTfm", lambda V: MiniTransformer(V, k, d=64, n_layers=2, n_heads=4),
        X_train, y_train, X_test, y_test, V, classes,
    )
    res["_meta"] = dict(
        n_train=int(len(X_train)), n_test=int(len(X_test)), V=V
    )
    return res


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: run only baselines (M0-M2) on a 50-sequence "
        "subsample with K=4 and a separate output filename. Exits in seconds.",
    )
    p.add_argument(
        "--ablation",
        action="store_true",
        help="Skip baselines + MLPs. Run only transformer variants: 2L/4L "
        "with and without DeltaTime, plus a whale-gpt-main-shape (3L, 8h, "
        "d=32, K=25, +DT) variant. Writes outputs/grammar/predict_results_ablation.{md,json}.",
    )
    p.add_argument(
        "--redesign",
        action="store_true",
        help="Run the whale-gpt-style redesign comparison: R0 control "
        "(B4-mimic, single-channel Coda + DT) vs R2 (multi-channel "
        "multi-task with Whale/Coda/Orn/Sync/Dur/DT/has_timestamps + "
        "per-column embeddings + multi-task heads). Both at 3L 8h d=32 "
        "K=25. Headline metric is the Coda head's bits/token + accuracy. "
        "Writes outputs/grammar/predict_results_redesign.{md,json}.",
    )
    p.add_argument(
        "--slow",
        action="store_true",
        help="(Only with --redesign) Flip FAST_MODE to False: AdamW + "
        "CosineAnnealingLR + bs=1000 + 5000 epochs + dropout 0.2, no "
        "early stopping. Multiplies wall-time per fold by ~30×.",
    )
    p.add_argument(
        "--variants",
        default="R0,R1,R2",
        help="(Only with --redesign) Comma-separated subset of "
        "{R0,R1,R2} to run. R0 = single-channel control; "
        "R1 = multi-channel + Coda CE only; R2 = multi-channel + "
        "multi-task heads. Default: all three.",
    )
    args = p.parse_args(argv)

    n_modes = sum([args.quick, args.ablation, args.redesign])
    if n_modes > 1:
        raise SystemExit("--quick / --ablation / --redesign are mutually exclusive")

    if args.quick:
        out_json = OUT_JSON.with_name("predict_results_quick.json")
        out_md = OUT_MD.with_name("predict_results_quick.md")
    elif args.ablation:
        out_json = OUT_JSON.with_name("predict_results_ablation.json")
        out_md = OUT_MD.with_name("predict_results_ablation.md")
    elif args.redesign:
        out_json = OUT_JSON.with_name("predict_results_redesign.json")
        out_md = OUT_MD.with_name("predict_results_redesign.md")
    else:
        out_json, out_md = OUT_JSON, OUT_MD
    out_json.parent.mkdir(parents=True, exist_ok=True)

    tokens = pd.read_csv(TOKENS)
    seq_dict = per_seq(tokens, TARGET)
    seq_ids = sorted(seq_dict.keys())
    if args.quick:
        seq_ids = seq_ids[:50]
        seq_dict = {sid: seq_dict[sid] for sid in seq_ids}
    dt_dict = per_seq_dt(tokens) if args.ablation else None
    if args.ablation:
        dt_dict = {sid: dt_dict[sid] for sid in seq_ids}

    # Redesign mode needs the whole multi-channel per-seq dict + global
    # whale/coda integer mappings.
    redesign_data = None
    V_whale = V_coda = None
    if args.redesign:
        whale_idx = pd.read_csv(WHALE_INDEX) if WHALE_INDEX.exists() else None
        if whale_idx is None:
            raise SystemExit(
                f"{WHALE_INDEX} missing; run `python -m src.pipeline.E_render_csv` first."
            )
        whale2id = {str(w): int(i) for w, i in
                    zip(whale_idx["Whale"], whale_idx["whale_id"])}
        # Coda integer encoding: dense reindex over all observed values.
        coda_uniq = sorted(int(c) for c in tokens["Coda"].unique())
        coda2id = {c: i for i, c in enumerate(coda_uniq)}
        V_whale = len(whale2id)
        V_coda = len(coda2id)
        redesign_data = per_seq_multi(tokens, whale2id, coda2id)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    K = 4 if args.quick else CONTEXT_K
    fold_results = []
    fast = not args.slow
    if args.redesign:
        print(f"redesign mode: V_coda={V_coda}, V_whale={V_whale}, "
              f"K=25, fast_mode={fast}")
    for fi, (tr_idx, te_idx) in enumerate(kf.split(seq_ids)):
        train_ids = [seq_ids[i] for i in tr_idx]
        test_ids = [seq_ids[i] for i in te_idx]
        t0 = time.time()
        if args.ablation:
            res = evaluate_ablation_fold(seq_dict, dt_dict, train_ids, test_ids)
        elif args.redesign:
            variants = tuple(v.strip() for v in args.variants.split(","))
            res = evaluate_redesign_fold(
                redesign_data, train_ids, test_ids,
                V_whale=V_whale, V_coda=V_coda, fast=fast,
                variants=variants,
            )
        else:
            res = evaluate_one_fold(seq_dict, train_ids, test_ids, K, quick=args.quick)
        res["_meta"]["fold"] = fi
        res["_meta"]["seconds"] = round(time.time() - t0, 1)
        fold_results.append(res)
        if args.ablation:
            prefix = "B"
        elif args.redesign:
            prefix = "R"
        else:
            prefix = "M"
        models = [k for k in res if k.startswith(prefix)]
        print(
            f"fold {fi}: "
            + "  ".join(
                f"{m}: bpt={res[m]['bits_per_token']:.3f} acc={res[m]['accuracy']:.3f}"
                for m in models
            )
            + f"  ({res['_meta']['seconds']}s)"
        )

    if args.ablation:
        prefix = "B"
    elif args.redesign:
        prefix = "R"
    else:
        prefix = "M"
    summary = {}
    model_keys = [k for k in fold_results[0] if k.startswith(prefix)]
    for m in model_keys:
        bits = [f[m]["bits_per_token"] for f in fold_results]
        accs = [f[m]["accuracy"] for f in fold_results]
        n_params = fold_results[0][m].get("n_params")
        summary[m] = dict(
            bits_per_token_mean=float(np.mean(bits)),
            bits_per_token_std=float(np.std(bits)),
            perplexity_mean=float(2 ** np.mean(bits)),
            accuracy_mean=float(np.mean(accs)),
            accuracy_std=float(np.std(accs)),
            n_params=n_params,
        )

    if args.ablation:
        V_used = fold_results[0]["_meta"].get("V_k8")
        out = dict(folds=fold_results, summary=summary, target=TARGET, V_k8=V_used,
                   V_k25=fold_results[0]["_meta"].get("V_k25"))
    elif args.redesign:
        V_used = fold_results[0]["_meta"].get("V_coda")
        out = dict(folds=fold_results, summary=summary, target=TARGET,
                   V_coda=V_used,
                   V_whale=fold_results[0]["_meta"].get("V_whale"),
                   K=fold_results[0]["_meta"].get("K"),
                   fast_mode=fold_results[0]["_meta"].get("fast_mode"))
    else:
        V_used = fold_results[0]["_meta"]["V"]
        out = dict(folds=fold_results, summary=summary, k=K, target=TARGET, V=V_used)
    out_json.write_text(json.dumps(out, indent=2, default=str))

    L = []
    if args.ablation:
        title = "# 5-fold transformer ablation on unified corpus (depth × DeltaTime × whale-gpt-main shape)"
    elif args.redesign:
        mode = "fast" if fold_results[0]["_meta"]["fast_mode"] else "slow"
        title = (
            "# 5-fold whale-gpt-style redesign — "
            "R0 (B4 single-channel control) vs R2 (multi-channel multi-task)"
            f" — `{mode}` training mode"
        )
    else:
        title_suffix = " (quick smoke test)" if args.quick else ""
        title = f"# 5-fold next-token prediction on unified corpus (bits/token){title_suffix}"
    L.append(title)
    L.append("")
    L.append(
        f"Sequence-level KFold ({N_FOLDS} folds, no within-sequence leakage). "
        f"Target = `{TARGET}` (V = {V_used} including PAD). "
        + ("Context K varies per variant. " if args.ablation
           else f"Context K = 25 past codas. " if args.redesign
           else f"Context K = {K} past codas. ")
        + "Metric = held-out cross-entropy in **bits/token** (log₂); lower is better. "
        "Perplexity = 2^(bits/token)."
    )
    L.append("")
    if not args.ablation:
        L.append(
            "**Apples-to-apples caveat.** The prior `~/Code/whale-gpt @ "
            "claude/whale-language-research-tEudI` benchmark used a "
            "rhythm·tempo·orn·rubato compound `Token` with V≈207 on the Sharma-only "
            "corpus (~4,800 codas). This run targets the rhythm-class integer "
            f"`Coda1` directly on the unified corpus (~38k codas, V={V_used}), so "
            "absolute bits/token are not directly comparable to the 4.63 figure."
        )
        L.append("")
    L.append("| # | model | params | bits/token (↓) | perplexity (↓) | accuracy |")
    L.append("|---|-------|-------:|---------------:|---------------:|---------:|")
    label_for = {
        "M0_majority": "majority (smoothed unigram)",
        "M1_markov1": "Markov-1",
        "M2_markov2": "Markov-2",
        "M3_MLP_S": "MLP-S (128, 64)",
        "M4_MLP_M": "MLP-M (256, 128)",
        "M5_MLP_L": "MLP-L (512, 256, 128)",
        "M6_EmbMLP": "Embedding-MLP (d=32, 256·128)",
        "M7_MiniTfm": "MiniTransformer (2L, 4h, d=64)",
        "B0_2L_d64_K8": "MiniTransformer 2L, 4h, d=64, K=8 (no DT)",
        "B1_2L_d64_K8_DT": "MiniTransformer 2L, 4h, d=64, K=8 + DT",
        "B2_4L_d64_K8": "MiniTransformer 4L, 4h, d=64, K=8 (no DT)",
        "B3_4L_d64_K8_DT": "MiniTransformer 4L, 4h, d=64, K=8 + DT",
        "B4_3L_d32_K25_DT": "whale-gpt-main shape: 3L, 8h, d=32, K=25 + DT",
        "R0_control_B4mimic": "R0 control (B4 mimic): single-channel Coda + DT",
        "R1_multichannel_singletask": "R1: multi-channel inputs, Coda CE only",
        "R2_multichannel_multitask": "R2 redesign: multi-channel + multi-task",
    }
    model_order = [m for m in label_for if m in summary]
    for i, m in enumerate(model_order):
        s = summary[m]
        params = f"{s['n_params']:,}" if s.get("n_params") else "—"
        idx_label = m.split("_", 1)[0]
        L.append(
            f"| {idx_label} | {label_for[m]} | {params} | "
            f"{s['bits_per_token_mean']:.3f} ± {s['bits_per_token_std']:.3f} | "
            f"{s['perplexity_mean']:.2f} | "
            f"{s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f} |"
        )
    L.append("")
    L.append("## Per-fold bits/token")
    L.append("")
    L.append("| fold |" + "|".join(f" {label_for[m]} " for m in model_order) + "|")
    L.append("|------|" + "|".join(["---:"] * len(model_order)) + "|")
    for f in fold_results:
        L.append(
            f"| {f['_meta']['fold']} |"
            + "|".join(f" {f[m]['bits_per_token']:.3f} " for m in model_order)
            + "|"
        )
    L.append("")
    best_m = min(summary, key=lambda m: summary[m]["bits_per_token_mean"])
    L.append("## Takeaway")
    L.append("")
    L.append(
        f"- The best held-out model is **{label_for[best_m]}** at "
        f"{summary[best_m]['bits_per_token_mean']:.3f} bits/token "
        f"(perplexity ≈ {summary[best_m]['perplexity_mean']:.2f})."
    )
    if args.ablation:
        base = summary.get("B0_2L_d64_K8")
        if base:
            L.append(
                f"- 2L baseline (B0) is {base['bits_per_token_mean']:.3f} bpt; the best "
                f"variant saves **{base['bits_per_token_mean'] - summary[best_m]['bits_per_token_mean']:.3f} "
                "bits/token** over it. Compare individual rows to see depth vs DT vs "
                "longer-context contributions."
            )
    elif args.redesign:
        def _fmt(name):
            s = summary.get(name)
            if not s:
                return None
            return (s["bits_per_token_mean"], s["bits_per_token_std"])
        r0 = _fmt("R0_control_B4mimic")
        r1 = _fmt("R1_multichannel_singletask")
        r2 = _fmt("R2_multichannel_multitask")
        if r0:
            L.append(f"- R0 (single-channel control) = **{r0[0]:.3f} ± {r0[1]:.3f}** bpt")
        if r1:
            d = (r0[0] - r1[0]) if r0 else None
            tail = (
                f"; Δ(R0 → R1) = **{'−' if d >= 0 else '+'}{abs(d):.3f}**"
                if d is not None else ""
            )
            L.append(
                f"- R1 (multi-channel input, Coda CE only) = "
                f"**{r1[0]:.3f} ± {r1[1]:.3f}** bpt{tail}"
            )
        if r2:
            d = (r0[0] - r2[0]) if r0 else None
            tail = (
                f"; Δ(R0 → R2) = **{'−' if d >= 0 else '+'}{abs(d):.3f}**"
                if d is not None else ""
            )
            L.append(
                f"- R2 (multi-channel input + multi-task heads) = "
                f"**{r2[0]:.3f} ± {r2[1]:.3f}** bpt{tail}"
            )
        L.append(
            "- Compare each Δ to the fold-variance σ. The R0 → R1 gap "
            "isolates *whether the extra input channels help*; the R1 → "
            "R2 gap isolates *whether multi-task supervision adds anything "
            "on top of multi-channel inputs*."
        )
    else:
        L.append(
            f"- The unigram majority baseline scores {summary['M0_majority']['bits_per_token_mean']:.3f} "
            f"bits/token; the best model saves "
            f"**{summary['M0_majority']['bits_per_token_mean'] - summary[best_m]['bits_per_token_mean']:.2f} bits/token** "
            f"(perplexity drops {summary['M0_majority']['perplexity_mean']:.1f} → "
            f"{summary[best_m]['perplexity_mean']:.1f})."
        )
        if "M7_MiniTfm" in summary:
            L.append(
                f"- Markov-1 alone is already strong ({summary['M1_markov1']['bits_per_token_mean']:.3f} bpt). "
                f"Larger models help further; the MiniTransformer / Emb-MLP can attend to "
                f"the full {K}-coda context."
            )
        else:
            L.append(
                f"- Markov-1 alone is already strong ({summary['M1_markov1']['bits_per_token_mean']:.3f} bpt). "
                "Quick mode skips the learned models — re-run without --quick for the "
                "MLP / Emb-MLP / MiniTransformer numbers."
            )
    out_md.write_text("\n".join(L) + "\n")
    print(f"\nwrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
