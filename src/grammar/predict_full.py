"""
Full-factorial overnight benchmark on the unified whale corpus.

Tests **architecture × target × loss-aggregation** on a tier-aware
held-out clean test set, with a follow-up **data-scheme** sweep on a
single representative cell to test the curriculum-vs-mixed hypotheses.

Main sweep (8 cells × 5 folds)
-------------------------------
    arch   target   loss_agg     primary metric         also reports
    ----   ------   --------     --------------         -------------
    tfm    coda     last         coda_bpt               coda_acc
    tfm    coda     per          coda_bpt               coda_acc
    tfm    joint    last         token_bpt + dt_bpt     coda_marg_bpt
    tfm    joint    per          token_bpt + dt_bpt     coda_marg_bpt
    h      coda     last         coda_bpt               coda_acc
    h      coda     per          coda_bpt               coda_acc
    h      joint    last         token_bpt + dt_bpt     coda_marg_bpt
    h      joint    per          token_bpt + dt_bpt     coda_marg_bpt

Data-scheme sweep (3 schemes × 5 folds, fixed cell `tfm | joint | per`)
-----------------------------------------------------------------------
    M  mixed                — phase-1 only, train on (hersh ∪ clean_train)
    C  curriculum           — phase-1 hersh, phase-2 clean_train
    MF mixed-then-finetune  — phase-1 mixed, phase-2 clean_train (the
                              gold-standard recipe; addresses both the
                              "clean exposure" and "catastrophic
                              forgetting" risks of the C variant)
    Each phase gets the same epoch / patience budget. Phase-2 uses a
    smaller LR (lr_phase2) to avoid blowing up the phase-1 representation.

Architectures (both at K=25, both causal-masked)
------------------------------------------------
    tfm  MiniTransformer 3L 8h d=32, encoder + causal mask, with optional
         (dt_log, has_ts) input projection. (B4 shape from predict_kfold.)
    h    m7-shape HookedTransformer (TransformerLens) 2L 4h d=64, causal,
         no DT input projection. (Same architecture as src/grammar/m7_hooked.py
         but with K=25 instead of K=8.)

Targets
-------
    coda   predict next Coda integer (V≈131). One head.
    joint  predict next Token (V=467 compound rhythm·tempo·orn·rubato)
           AND next DT bucket (V=7 whale-scheme: missing + 5 timing
           buckets + switch). Two heads, joint loss = CE(token) + CE(dt).

Loss aggregation (training only — eval is always at last position)
------------------------------------------------------------------
    last  CE only at the last context position.
    per   per-position CE: window of length K predicts positions 1..K
          from positions 0..K-1; mean over all K next-token predictions.

Tier split
----------
    5-fold KFold over **clean** sequenceIds (sharma2024_dswp +
    sharma2025_birth) — held-out test is always clean. Per fold:
        train pool   = all hersh2022_pacific  +  remaining 4/5 of clean
        val          = 10 % slice of clean_train (held out from train pool)
        test         = held-out 1/5 of clean
    Both train pool and val are ER on a Coda-aware tier-mixed schedule.

Eval
----
    Always last-position bits/token + accuracy on the test set.
    Joint cells additionally report a **Coda-marginalized** bpt computed
    by summing softmax(Token) probabilities by their underlying rhythm
    class — directly comparable to coda-target cells.

Outputs
-------
    outputs/grammar/predict_results_full.json    (all per-fold metrics)
    outputs/grammar/predict_results_full.md      (5-fold mean ± std table)
    outputs/grammar/predict_results_full.partial.json
        Updated after every fold so a crash leaves usable state.
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
from torch import nn

from src.grammar.dt_buckets import WHALE_N_BUCKETS, whale_dt_to_bucket
from src.grammar.m7_hooked import make_m7_config
from src.grammar.whale_compound import load_compound_whale

ROOT = Path(__file__).resolve().parents[2]
_OUT_DIR = ROOT / "outputs" / "grammar"
OUT_JSON = _OUT_DIR / "predict_results_full.json"
OUT_MD = _OUT_DIR / "predict_results_full.md"
OUT_PARTIAL = _OUT_DIR / "predict_results_full.partial.json"

CONTEXT_K = 25
HERSH_PREFIXES = ("hersh2022_pacific::",)
CLEAN_PREFIXES = ("sharma2024_dswp::", "sharma2025_birth::")
DT_LOG_OFFSET = 0.1


def _tier(seq_id: str) -> str:
    if any(seq_id.startswith(p) for p in HERSH_PREFIXES):
        return "hersh"
    if any(seq_id.startswith(p) for p in CLEAN_PREFIXES):
        return "clean"
    return "other"


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def load_corpus():
    """Read whale_dialogues.csv → per-sequence streams.

    Returns:
        per_seq:   dict[seq_id] -> dict with keys
                       coda     list[int]   original rhythm-class integer
                       token    list[int]   compound (rhythm·tempo·orn·rubato)
                       dt       list[int]   DT bucket id (whale scheme)
                       dt_log   list[float] log-DT input feature
                       has_ts   list[int]   timestamp-validity flag
        V_coda:    int  vocab size of Coda
        V_token:   int  vocab size of compound Token
        V_dt:      int  number of DT buckets (= WHALE_N_BUCKETS = 7)
        token_to_coda: torch.LongTensor (V_token,) mapping token_id -> coda_id
    """
    df, _decoder = load_compound_whale()
    df = df.sort_values(["sequenceId", "itemPosition"]).reset_index(drop=True)

    # token_to_coda: each compound token has a unique (Coda_orig, ...) tuple.
    # Take the first row per token to recover the rhythm class.
    tok2coda_df = (
        df[["Coda", "Coda_orig"]].drop_duplicates("Coda").sort_values("Coda")
    )
    V_token = int(df["Coda"].max()) + 1
    V_coda = int(df["Coda_orig"].max()) + 1
    token_to_coda = torch.zeros(V_token, dtype=torch.long)
    for _, row in tok2coda_df.iterrows():
        token_to_coda[int(row["Coda"])] = int(row["Coda_orig"])

    per_seq: dict[str, dict[str, list]] = {}
    for sid, g in df.groupby("sequenceId", sort=False):
        codas = g["Coda_orig"].astype(int).tolist()
        tokens = g["Coda"].astype(int).tolist()
        dts = g["TimeDelta"].astype(float).tolist()
        has_ts = g["has_timestamps"].astype(int).tolist()
        speakers = g["Whale"].astype(str).tolist()
        # Speaker change requires both endpoints to have valid timestamps.
        switched = [False]
        for i in range(1, len(speakers)):
            switched.append(
                speakers[i] != speakers[i - 1]
                and has_ts[i] == 1
                and has_ts[i - 1] == 1
            )
        dt_buckets = [
            whale_dt_to_bucket(d, h, switched=sw)
            for d, h, sw in zip(dts, has_ts, switched)
        ]
        dt_log = []
        for d, h in zip(dts, has_ts):
            if not h or d < 0:
                dt_log.append(float(np.log(DT_LOG_OFFSET)))
            else:
                dt_log.append(float(np.log(DT_LOG_OFFSET + max(d, 0.0))))
        per_seq[str(sid)] = dict(
            coda=codas,
            token=tokens,
            dt=dt_buckets,
            dt_log=dt_log,
            has_ts=has_ts,
        )

    return per_seq, V_coda, V_token, WHALE_N_BUCKETS, token_to_coda


def build_windows(per_seq, seq_ids, k):
    """Build (N, k+1) tensors for each stream — windows that include the
    target position at index k (positions 0..k-1 form the context, k is
    the target). Sequences shorter than k+1 are skipped (no padding for
    per-position loss to remain well-defined)."""
    chans = ("coda", "token", "dt", "dt_log", "has_ts")
    out = {c: [] for c in chans}
    for sid in seq_ids:
        seq = per_seq[sid]
        L = len(seq["coda"])
        if L < k + 1:
            continue
        for i in range(L - k):
            for c in chans:
                out[c].append(seq[c][i : i + k + 1])
    if not out["coda"]:
        empty_int = torch.empty(0, k + 1, dtype=torch.long)
        empty_f = torch.empty(0, k + 1, dtype=torch.float32)
        return dict(coda=empty_int, token=empty_int, dt=empty_int,
                    dt_log=empty_f, has_ts=empty_int)
    return dict(
        coda=torch.tensor(out["coda"], dtype=torch.long),
        token=torch.tensor(out["token"], dtype=torch.long),
        dt=torch.tensor(out["dt"], dtype=torch.long),
        dt_log=torch.tensor(out["dt_log"], dtype=torch.float32),
        has_ts=torch.tensor(out["has_ts"], dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


class WhaleTfm(nn.Module):
    """MiniTransformer with causal mask, optional (dt_log, has_ts) input
    projection, and 1 or 2 output heads.

    target='coda'  → one head over V_coda
    target='joint' → token head over V_token + DT head over n_dt
    """

    def __init__(self, *, V_in: int, V_token_out: int | None,
                 V_coda_out: int | None, n_dt: int, k: int,
                 d: int = 32, n_layers: int = 3, n_heads: int = 8,
                 target: str = "coda", with_dt_input: bool = True):
        super().__init__()
        self.target = target
        self.with_dt_input = with_dt_input
        self.k = k
        self.d = d

        self.emb = nn.Embedding(V_in, d)
        self.pos = nn.Embedding(k, d)
        if with_dt_input:
            self.dt_proj = nn.Linear(2, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4 * d,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.tfm = nn.TransformerEncoder(layer, n_layers)
        if target == "coda":
            assert V_coda_out is not None
            self.head_coda = nn.Linear(d, V_coda_out)
        else:  # joint
            assert V_token_out is not None
            self.head_token = nn.Linear(d, V_token_out)
            self.head_dt = nn.Linear(d, n_dt)

    def forward(self, x: torch.Tensor, dt_input: torch.Tensor | None = None):
        B, K = x.shape
        pos_ids = torch.arange(K, device=x.device).expand(B, K)
        h = self.emb(x) + self.pos(pos_ids)
        if self.with_dt_input and dt_input is not None:
            h = h + self.dt_proj(dt_input)
        mask = torch.triu(torch.ones(K, K, dtype=torch.bool, device=x.device),
                          diagonal=1)
        h = self.tfm(h, mask=mask)
        if self.target == "coda":
            return dict(coda=self.head_coda(h))
        return dict(token=self.head_token(h), dt=self.head_dt(h))


class WhaleHooked(nn.Module):
    """m7-shape HookedTransformer wrapper — mirrors WhaleTfm interface.

    target='coda'  → uses model.unembed alone
    target='joint' → adds a separate dt_head sharing the post-LN residual

    dt_input (dt_log, has_ts) is injected at blocks.0.hook_resid_pre, the
    same point WhaleTfm adds it (after embed+pos, before block 0).
    """

    def __init__(self, *, V_in: int, V_coda_out: int | None,
                 V_token_out: int | None, n_dt: int, k: int,
                 target: str = "coda"):
        super().__init__()
        from transformer_lens import HookedTransformer
        cfg = make_m7_config(d_vocab=V_in, n_ctx=k)
        self.model = HookedTransformer(cfg)
        self.target = target
        self.k = k
        self.d = cfg.d_model
        self.dt_proj = nn.Linear(2, cfg.d_model)
        if target == "joint":
            self.head_dt = nn.Linear(cfg.d_model, n_dt)

    def forward(self, x: torch.Tensor, dt_input: torch.Tensor | None = None):
        fwd_hooks = []
        if dt_input is not None:
            def inject_dt(resid, hook, _dt=dt_input):
                return resid + self.dt_proj(_dt)
            fwd_hooks = [("blocks.0.hook_resid_pre", inject_dt)]

        if self.target == "coda":
            logits = self.model.run_with_hooks(
                x, fwd_hooks=fwd_hooks, return_type="logits")
            return dict(coda=logits)

        with self.model.hooks(fwd_hooks=fwd_hooks):
            _, cache = self.model.run_with_cache(x)
        last_resid = cache[f"blocks.{self.model.cfg.n_layers - 1}.hook_resid_post"]
        normalized = self.model.ln_final(last_resid)
        token_logits = self.model.unembed(normalized)
        dt_logits = self.head_dt(normalized)
        return dict(token=token_logits, dt=dt_logits)


def build_model(*, arch: str, target: str, V_coda: int, V_token: int,
                n_dt: int, k: int) -> nn.Module:
    if arch == "tfm":
        if target == "coda":
            return WhaleTfm(
                V_in=V_coda, V_token_out=None, V_coda_out=V_coda, n_dt=n_dt,
                k=k, target="coda", with_dt_input=True,
            )
        return WhaleTfm(
            V_in=V_token, V_token_out=V_token, V_coda_out=None, n_dt=n_dt,
            k=k, target="joint", with_dt_input=True,
        )
    if arch == "h":
        if target == "coda":
            return WhaleHooked(
                V_in=V_coda, V_coda_out=V_coda, V_token_out=None, n_dt=n_dt,
                k=k, target="coda",
            )
        return WhaleHooked(
            V_in=V_token, V_coda_out=None, V_token_out=V_token, n_dt=n_dt,
            k=k, target="joint",
        )
    raise ValueError(arch)


# ---------------------------------------------------------------------------
# Loss + eval
# ---------------------------------------------------------------------------


def cell_inputs(windows: dict, target: str) -> tuple[torch.Tensor,
                                                     torch.Tensor | None]:
    """Build (x, dt_input) for the model's forward.

    For target='coda' the input stream is Coda; for target='joint' it is
    the compound Token. dt_input is (dt_log, has_ts) at positions
    0..K-1, only used by arch='tfm'.
    """
    K_plus_1 = windows["coda"].size(1)
    K = K_plus_1 - 1
    x_field = "coda" if target == "coda" else "token"
    x = windows[x_field][:, :K]  # positions 0..K-1
    dt_log = windows["dt_log"][:, :K]
    has_ts = windows["has_ts"][:, :K].float()
    dt_input = torch.stack([dt_log, has_ts], dim=-1)  # (N, K, 2)
    return x, dt_input


def compute_loss(out: dict, windows: dict, target: str, loss_agg: str,
                 dt_weight: float = 1.0) -> torch.Tensor:
    """Cross-entropy training loss for the given target × loss_agg.

    The model forward consumes positions 0..K-1; predictions are next-
    token at each position (per) or only at the final position (last).
    """
    if target == "coda":
        logits = out["coda"]  # (B, K, V)
        K = logits.size(1)
        if loss_agg == "last":
            pred = logits[:, -1]
            tgt = windows["coda"][:, K]  # position K target
            return nn.functional.cross_entropy(pred, tgt)
        # per-position
        pred = logits.reshape(-1, logits.size(-1))
        tgt = windows["coda"][:, 1:K + 1].reshape(-1)
        return nn.functional.cross_entropy(pred, tgt)
    # joint
    token_logits = out["token"]
    dt_logits = out["dt"]
    K = token_logits.size(1)
    if loss_agg == "last":
        return (
            nn.functional.cross_entropy(token_logits[:, -1], windows["token"][:, K])
            + dt_weight * nn.functional.cross_entropy(dt_logits[:, -1], windows["dt"][:, K])
        )
    Vt = token_logits.size(-1)
    Vd = dt_logits.size(-1)
    return (
        nn.functional.cross_entropy(
            token_logits.reshape(-1, Vt),
            windows["token"][:, 1:K + 1].reshape(-1),
        )
        + dt_weight * nn.functional.cross_entropy(
            dt_logits.reshape(-1, Vd),
            windows["dt"][:, 1:K + 1].reshape(-1),
        )
    )


def eval_cell_metrics(model: nn.Module, windows: dict, target: str,
                       token_to_coda: torch.Tensor, V_coda: int,
                       bs: int = 256) -> dict:
    """Compute last-position bpt + accuracy on the test set.

    For target='coda': returns coda_bpt + coda_acc.
    For target='joint': returns token_bpt, token_acc, dt_bpt, dt_acc,
        coda_marg_bpt, coda_marg_acc.
    """
    model.eval()
    K = windows["coda"].size(1) - 1
    x_field = "coda" if target == "coda" else "token"
    x_all = windows[x_field][:, :K]
    dt_log = windows["dt_log"][:, :K]
    has_ts = windows["has_ts"][:, :K].float()
    dt_input_all = torch.stack([dt_log, has_ts], dim=-1)

    sums = dict(
        coda_nll=0.0, coda_correct=0,
        token_nll=0.0, token_correct=0,
        dt_nll=0.0, dt_correct=0,
        coda_marg_nll=0.0, coda_marg_correct=0,
        n=0,
    )
    with torch.no_grad():
        for i in range(0, x_all.size(0), bs):
            xb = x_all[i : i + bs]
            dtb = dt_input_all[i : i + bs]
            out = model(xb, dt_input=dtb)
            n = xb.size(0)
            sums["n"] += n
            tgt_coda = windows["coda"][i : i + bs, K]
            if target == "coda":
                logits = out["coda"][:, -1]
                sums["coda_nll"] += nn.functional.cross_entropy(
                    logits, tgt_coda, reduction="sum").item()
                sums["coda_correct"] += (logits.argmax(-1) == tgt_coda).sum().item()
            else:
                tgt_token = windows["token"][i : i + bs, K]
                tgt_dt = windows["dt"][i : i + bs, K]
                token_logits = out["token"][:, -1]
                dt_logits = out["dt"][:, -1]
                sums["token_nll"] += nn.functional.cross_entropy(
                    token_logits, tgt_token, reduction="sum").item()
                sums["token_correct"] += (token_logits.argmax(-1) == tgt_token).sum().item()
                sums["dt_nll"] += nn.functional.cross_entropy(
                    dt_logits, tgt_dt, reduction="sum").item()
                sums["dt_correct"] += (dt_logits.argmax(-1) == tgt_dt).sum().item()
                # Coda-marginalized: P(coda=c) = sum_{t : token_to_coda[t]==c} P(t)
                token_probs = torch.softmax(token_logits, dim=-1)  # (B, V_token)
                coda_probs = torch.zeros(token_probs.size(0), V_coda)
                coda_probs.index_add_(1, token_to_coda, token_probs)
                p_true = coda_probs.gather(1, tgt_coda.unsqueeze(1)).squeeze(1)
                sums["coda_marg_nll"] += -torch.log(p_true.clamp(min=1e-12)).sum().item()
                sums["coda_marg_correct"] += (coda_probs.argmax(-1) == tgt_coda).sum().item()

    n = max(sums["n"], 1)
    out_metrics = {}
    if target == "coda":
        out_metrics["coda_bpt"] = sums["coda_nll"] / n / math.log(2)
        out_metrics["coda_acc"] = sums["coda_correct"] / n
    else:
        out_metrics["token_bpt"] = sums["token_nll"] / n / math.log(2)
        out_metrics["token_acc"] = sums["token_correct"] / n
        out_metrics["dt_bpt"] = sums["dt_nll"] / n / math.log(2)
        out_metrics["dt_acc"] = sums["dt_correct"] / n
        out_metrics["coda_marg_bpt"] = sums["coda_marg_nll"] / n / math.log(2)
        out_metrics["coda_marg_acc"] = sums["coda_marg_correct"] / n
    return out_metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_phase(model: nn.Module, train_W: dict, val_W: dict,
                target: str, loss_agg: str, *,
                epochs: int, lr: float, bs: int, patience: int,
                dt_weight: float = 1.0, seed: int = 0):
    """AdamW + ES on val combined loss, single phase. Used as a building
    block for both single-phase ('M' mixed) and two-phase ('C'
    curriculum, 'MF' mixed-then-finetune) training."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = train_W["coda"].size(0)
    if n == 0:
        return float("inf")
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    since = 0
    val_x, val_dt = cell_inputs(val_W, target)
    train_x, train_dt = cell_inputs(train_W, target)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i : i + bs]
            xb = train_x[b]
            dtb = train_dt[b]
            wb = {k: v[b] for k, v in train_W.items()}
            opt.zero_grad()
            out = model(xb, dt_input=dtb)
            loss = compute_loss(out, wb, target, loss_agg, dt_weight=dt_weight)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_out = model(val_x, dt_input=val_dt)
            val_loss = compute_loss(val_out, val_W, target, loss_agg,
                                    dt_weight=dt_weight).item()
        if val_loss < best_val - 1e-4:
            best_val, since = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break

    model.load_state_dict(best_state)
    return best_val


def train_scheme(model: nn.Module, phase_Ws: list[dict], val_W: dict,
                 target: str, loss_agg: str, *,
                 epochs: int, lrs: list[float], bs: int, patience: int,
                 dt_weight: float = 1.0, seed: int = 0):
    """Run one or more training phases on ``model``, persisting weights
    across phases. ``lrs`` is parallel to ``phase_Ws`` (one LR per phase).

    Returns the best val loss observed in the *final* phase — because
    that's the one whose model state we keep.
    """
    last_val = float("inf")
    for i, (W, lr) in enumerate(zip(phase_Ws, lrs)):
        last_val = train_phase(
            model, W, val_W, target, loss_agg,
            epochs=epochs, lr=lr, bs=bs, patience=patience,
            dt_weight=dt_weight, seed=seed + i,
        )
    return last_val


# train_cell preserved for backward compatibility (single-phase = M scheme).
train_cell = train_phase


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


CELLS = [
    ("tfm", "coda", "last"),
    ("tfm", "coda", "per"),
    ("tfm", "joint", "last"),
    ("tfm", "joint", "per"),
    ("h", "coda", "last"),
    ("h", "coda", "per"),
    ("h", "joint", "last"),
    ("h", "joint", "per"),
]

CELL_LABEL = {
    ("tfm", "coda", "last"): "tfm | coda | last",
    ("tfm", "coda", "per"): "tfm | coda | per",
    ("tfm", "joint", "last"): "tfm | joint | last",
    ("tfm", "joint", "per"): "tfm | joint | per",
    ("h", "coda", "last"): "m7h | coda | last",
    ("h", "coda", "per"): "m7h | coda | per",
    ("h", "joint", "last"): "m7h | joint | last",
    ("h", "joint", "per"): "m7h | joint | per",
}


def _concat_windows(*Ws: dict) -> dict:
    keys = Ws[0].keys()
    return {k: torch.cat([W[k] for W in Ws], dim=0) for k in keys}


def _format_metrics_line(label: str, target: str, metrics: dict) -> str:
    if "error" in metrics:
        return f"    {label:32s}: ERROR ({metrics['seconds']}s)"
    if target == "coda":
        return (f"    {label:32s}: coda_bpt={metrics['coda_bpt']:.3f} "
                f"acc={metrics['coda_acc']:.3f} "
                f"({metrics['seconds']}s, {metrics['n_params']/1000:.0f}K params)")
    return (f"    {label:32s}: token_bpt={metrics['token_bpt']:.3f} "
            f"coda_marg_bpt={metrics['coda_marg_bpt']:.3f} "
            f"dt_bpt={metrics['dt_bpt']:.3f} "
            f"({metrics['seconds']}s, {metrics['n_params']/1000:.0f}K params)")


# ---------------------------------------------------------------------------
# Classical reference baselines
# Ported from predict_smoothed.py (KN) and predict_kfold_compare.py (Markov).
# All functions work on sequences of strings so callers convert int→str.
# ---------------------------------------------------------------------------

_BOS = "__BOS__"


def _eval_markov(train_seqs: list[list[str]], test_seqs: list[list[str]],
                 order: int, alpha: float = 0.5) -> dict:
    """Laplace-smoothed Markov model of given order. Returns {bpt, acc}."""
    all_train = [t for seq in train_seqs for t in seq]
    vocab = sorted(set(all_train) | {t for seq in test_seqs for t in seq})
    V = len(vocab)
    unigram: Counter = Counter(all_train)
    n_train = len(all_train)
    majority = unigram.most_common(1)[0][0]

    trans: dict[tuple, Counter] = {}
    for seq in train_seqs:
        padded = [_BOS] * order + list(seq)
        for i in range(order, len(padded)):
            ctx = tuple(padded[i - order:i])
            trans.setdefault(ctx, Counter())[padded[i]] += 1

    log_probs: list[float] = []
    correct = 0
    total = 0
    for seq in test_seqs:
        padded = [_BOS] * order + list(seq)
        for i in range(order, len(padded)):
            ctx = tuple(padded[i - order:i])
            w = padded[i]
            c = trans.get(ctx)
            if c is None:
                p = (unigram.get(w, 0) + alpha) / (n_train + alpha * V)
                pred = majority
            else:
                denom = sum(c.values()) + alpha * V
                p = (c.get(w, 0) + alpha) / denom
                pred = c.most_common(1)[0][0]
            log_probs.append(math.log2(max(p, 1e-12)))
            correct += int(pred == w)
            total += 1
    return dict(
        bpt=-float(np.mean(log_probs)) if log_probs else float("inf"),
        acc=correct / total if total else 0.0,
    )


def _fit_kn(train_seqs: list[list[str]], n: int) -> dict:
    """Fit a Modified Kneser-Ney n-gram model (Chen & Goodman 1998)."""
    bos = [_BOS] * (n - 1)
    raw: list[dict] = [defaultdict(Counter) for _ in range(n + 1)]
    uniq: list[set] = [set() for _ in range(n + 2)]
    vocab_set: set[str] = set()
    for seq in train_seqs:
        full = bos + list(seq)
        vocab_set.update(seq)
        for i in range(n - 1, len(full)):
            for k in range(1, n + 1):
                g = tuple(full[i - (k - 1):i + 1])
                raw[k][g[:-1]][g[-1]] += 1
                uniq[k].add(g)
    cont: list[dict] = [defaultdict(Counter) for _ in range(n + 1)]
    for k in range(1, n):
        for g in uniq[k + 1]:
            cont[k][g[1:-1]][g[-1]] += 1
    disc: list = [None] * (n + 1)
    for k in range(1, n + 1):
        cd = raw[k] if k == n else cont[k]
        N = [0] * 5
        for cnt in cd.values():
            for c in cnt.values():
                if 1 <= c <= 4:
                    N[c] += 1
        if not N[1] or not N[2]:
            disc[k] = (0.5, 0.5, 0.5)
            continue
        Y = N[1] / (N[1] + 2 * N[2])
        def _d(num, den, fallback):
            return max(0.0, min(num / den if den else fallback, num + 1))
        D1 = _d(N[1] - 2 * Y * N[2], N[1], 0.5)
        D2 = _d(2 * N[2] - 3 * Y * N[3], N[2], D1) if N[3] else D1
        D3 = _d(3 * N[3] - 4 * Y * N[4], N[3], D2) if N[4] else D2
        disc[k] = (D1, D2, D3)
    return dict(n=n, raw=raw, cont=cont, disc=disc, vocab=sorted(vocab_set))


def _kn_prob(model: dict, h: tuple, w_true: str) -> float:
    """Return P_KN(w_true | h) using the fitted model."""
    n, raw, cont, disc, vocab = (
        model["n"], model["raw"], model["cont"], model["disc"], model["vocab"])
    V = len(vocab)
    if len(h) > n - 1:
        h = h[-(n - 1):]

    def _d(D_tup, c):
        if c <= 0: return 0.0
        return D_tup[min(c, 3) - 1]

    def _recur(h_tup):
        m = len(h_tup)
        if m == 0:
            c1 = cont[1][()]
            tot = sum(c1.values())
            if not tot:
                return 1.0 / V
            D = disc[1]
            gam = sum(_d(D, c) for c in c1.values()) / tot
            cw = c1.get(w_true, 0)
            return max(cw - _d(D, cw), 0.0) / tot + gam / V
        order = m + 1
        cd = raw[n] if order == n else cont[order]
        ch = cd.get(h_tup)
        if ch is None or not sum(ch.values()):
            return _recur(h_tup[1:])
        c_h = sum(ch.values())
        D = disc[order]
        gam = sum(_d(D, c) for c in ch.values()) / c_h
        cw = ch.get(w_true, 0)
        return max(cw - _d(D, cw), 0.0) / c_h + gam * _recur(h_tup[1:])

    return max(_recur(h), 1e-12)


def _eval_kn(train_seqs, test_seqs, n=5, cache_size=0, lam=0.15):
    """Evaluate KN n-gram (optionally with recency cache) on test sequences."""
    model = _fit_kn(train_seqs, n)
    bos = [_BOS] * (n - 1)
    log_probs: list[float] = []
    correct = 0
    total = 0
    for seq in test_seqs:
        full = bos + list(seq)
        emitted: list[str] = []
        for i in range(n - 1, len(full)):
            h = tuple(full[i - (n - 1):i])
            w = full[i]
            p_kn = _kn_prob(model, h, w)
            if cache_size > 0 and emitted:
                recent = emitted[-cache_size:]
                cache_cnt = Counter(recent)
                V = len(model["vocab"])
                cache_denom = len(recent) + 0.5 * V
                p_cache = (cache_cnt.get(w, 0) + 0.5) / cache_denom
                p = (1 - lam) * p_kn + lam * p_cache
            else:
                p = p_kn
            log_probs.append(math.log2(max(p, 1e-12)))
            # argmax: just use KN argmax (cache doesn't change winner much)
            best = max(model["vocab"], key=lambda x: _kn_prob(model, h, x))
            correct += int(best == w)
            total += 1
            emitted.append(w)
    return dict(
        bpt=-float(np.mean(log_probs)) if log_probs else float("inf"),
        acc=correct / total if total else 0.0,
    )


def _eval_markov_joint(
    train_seqs: list[list[str]], test_seqs: list[list[str]],
    order: int, tok_to_coda: dict[str, str], alpha: float = 0.5,
) -> dict:
    """Markov model on compound-token seqs; returns token + coda-marginalised metrics."""
    all_train = [t for seq in train_seqs for t in seq]
    vocab = sorted(set(all_train) | {t for seq in test_seqs for t in seq})
    V = len(vocab)
    unigram: Counter = Counter(all_train)
    n_train = len(all_train)
    majority = unigram.most_common(1)[0][0]

    from collections import defaultdict as _dd
    coda_groups: dict[str, list[str]] = _dd(list)
    for tok in vocab:
        coda_groups[tok_to_coda.get(tok, tok)].append(tok)

    trans: dict[tuple, Counter] = {}
    for seq in train_seqs:
        padded = [_BOS] * order + list(seq)
        for i in range(order, len(padded)):
            ctx = tuple(padded[i - order:i])
            trans.setdefault(ctx, Counter())[padded[i]] += 1

    token_lp: list[float] = []
    coda_lp: list[float] = []
    token_correct = coda_correct = total = 0
    for seq in test_seqs:
        padded = [_BOS] * order + list(seq)
        for i in range(order, len(padded)):
            ctx = tuple(padded[i - order:i])
            w = padded[i]
            true_coda = tok_to_coda.get(w, w)
            c = trans.get(ctx)
            if c is None:
                denom = n_train + alpha * V
                p_w = (unigram.get(w, 0) + alpha) / denom
                p_coda = sum((unigram.get(t, 0) + alpha) / denom
                             for t in coda_groups[true_coda])
                pred_tok = majority
            else:
                denom = sum(c.values()) + alpha * V
                p_w = (c.get(w, 0) + alpha) / denom
                p_coda = sum((c.get(t, 0) + alpha) / denom
                             for t in coda_groups[true_coda])
                pred_tok = c.most_common(1)[0][0]
            token_lp.append(math.log2(max(p_w, 1e-12)))
            coda_lp.append(math.log2(max(p_coda, 1e-12)))
            pred_coda = tok_to_coda.get(pred_tok, pred_tok)
            token_correct += int(pred_tok == w)
            coda_correct += int(pred_coda == true_coda)
            total += 1
    return dict(
        token_bpt=-float(np.mean(token_lp)) if token_lp else float("inf"),
        token_acc=token_correct / total if total else 0.0,
        coda_marg_bpt=-float(np.mean(coda_lp)) if coda_lp else float("inf"),
        coda_marg_acc=coda_correct / total if total else 0.0,
    )


def _kn_probs_batch(model: dict, h: tuple) -> dict:
    """Return {w: P_KN(w|h)} for every w in model['vocab'] in one pass.

    Avoids calling _kn_prob V times for argmax/marginalisation by computing
    the full distribution top-down, sharing the recursive backoff computation.
    """
    n, raw, cont, disc, vocab = (
        model["n"], model["raw"], model["cont"], model["disc"], model["vocab"])
    V = len(vocab)
    if len(h) > n - 1:
        h = h[-(n - 1):]

    def _d(D_tup, c):
        return D_tup[min(c, 3) - 1] if c > 0 else 0.0

    def _recur(h_tup):
        m = len(h_tup)
        if m == 0:
            c1 = cont[1][()]
            tot = sum(c1.values()) or 0
            if not tot:
                return {w: 1.0 / V for w in vocab}
            D = disc[1]
            gam = sum(_d(D, c) for c in c1.values()) / tot
            return {
                w: max(c1.get(w, 0) - _d(D, c1.get(w, 0)), 0.0) / tot + gam / V
                for w in vocab
            }
        order = m + 1
        cd = raw[n] if order == n else cont[order]
        ch = cd.get(h_tup)
        lower = _recur(h_tup[1:])
        if ch is None or not sum(ch.values()):
            return lower
        c_h = sum(ch.values())
        D = disc[order]
        gam = sum(_d(D, c) for c in ch.values()) / c_h
        result = {}
        for w in vocab:
            cw = ch.get(w, 0)
            result[w] = max(cw - _d(D, cw), 0.0) / c_h + gam * lower[w]
        return result

    raw_probs = _recur(h)
    return {w: max(raw_probs.get(w, 1e-12), 1e-12) for w in vocab}


def _eval_kn_joint(
    train_seqs, test_seqs, n: int, tok_to_coda: dict[str, str],
    cache_size: int = 0, lam: float = 0.15,
) -> dict:
    """KN n-gram on compound-token seqs; returns token + coda-marginalised metrics.

    Uses _kn_probs_batch so the full distribution is computed once per context
    instead of calling _kn_prob V times for argmax/marginalisation.
    """
    from collections import defaultdict as _dd
    model = _fit_kn(train_seqs, n)
    train_vocab = set(model["vocab"])
    # coda_groups must cover ALL token IDs (train + test OOV).
    all_toks = train_vocab | {t for seq in test_seqs for t in seq}
    coda_groups: dict[str, list[str]] = _dd(list)
    for tok in all_toks:
        coda_groups[tok_to_coda.get(tok, tok)].append(tok)
    # Fallback probability for OOV tokens (uniform over training vocab).
    p_oov = 1e-12

    bos = [_BOS] * (n - 1)
    token_lp: list[float] = []
    coda_lp: list[float] = []
    token_correct = coda_correct = total = 0
    for seq in test_seqs:
        full = bos + list(seq)
        emitted: list[str] = []
        for i in range(n - 1, len(full)):
            h = tuple(full[i - (n - 1):i])
            w = full[i]
            true_coda = tok_to_coda.get(w, w)

            # Compute full P(token | h) in one batch pass.
            probs = _kn_probs_batch(model, h)

            if cache_size > 0 and emitted:
                recent = emitted[-cache_size:]
                cache_cnt = Counter(recent)
                V_tok = len(train_vocab)
                cache_denom = len(recent) + 0.5 * V_tok
                probs = {
                    t: (1 - lam) * probs[t]
                    + lam * (cache_cnt.get(t, 0) + 0.5) / cache_denom
                    for t in train_vocab
                }

            p_tok = probs.get(w, p_oov)
            p_coda = sum(probs.get(t, p_oov) for t in coda_groups[true_coda])
            best_tok = max(probs, key=probs.__getitem__)  # argmax over train vocab
            pred_coda = tok_to_coda.get(best_tok, best_tok)

            token_lp.append(math.log2(max(p_tok, 1e-12)))
            coda_lp.append(math.log2(max(p_coda, 1e-12)))
            token_correct += int(best_tok == w)
            coda_correct += int(pred_coda == true_coda)
            total += 1
            emitted.append(w)
    return dict(
        token_bpt=-float(np.mean(token_lp)) if token_lp else float("inf"),
        token_acc=token_correct / total if total else 0.0,
        coda_marg_bpt=-float(np.mean(coda_lp)) if coda_lp else float("inf"),
        coda_marg_acc=coda_correct / total if total else 0.0,
    )


def run_baselines_fold(
    per_seq: dict, train_ids: list, test_ids: list,
    token_to_coda: "torch.Tensor | None" = None,
) -> dict:
    """Run classical baselines on the same fold split as neural cells.

    Returns fold_out entries keyed as "baseline | <model> | <target>".
    Joint target (compound token + coda-marginalised) is produced when
    token_to_coda is supplied.
    """
    def _seqs(ids, key):
        return [[str(v) for v in per_seq[s][key]] for s in ids]

    coda_train = _seqs(train_ids, "coda")
    coda_test  = _seqs(test_ids,  "coda")
    tok_train  = _seqs(train_ids, "token")
    tok_test   = _seqs(test_ids,  "token")

    # String-keyed tok→coda map for the joint evaluators.
    tok_to_coda_str: dict[str, str] = {}
    if token_to_coda is not None:
        for tid in range(len(token_to_coda)):
            tok_to_coda_str[str(tid)] = str(int(token_to_coda[tid]))

    out = {}
    t0 = time.time()

    # ── Coda-only baselines (rhythm-class prediction) ─────────────────
    for label, result in [
        ("majority", _eval_markov(coda_train, coda_test, order=0)),
        ("markov1",  _eval_markov(coda_train, coda_test, order=1)),
        ("markov2",  _eval_markov(coda_train, coda_test, order=2)),
        ("kn5",      _eval_kn(coda_train, coda_test, n=5)),
        ("kn5cache", _eval_kn(coda_train, coda_test, n=5, cache_size=20, lam=0.15)),
    ]:
        key = f"baseline | {label} | coda"
        out[key] = dict(coda_bpt=result["bpt"], coda_acc=result["acc"],
                        n_params=0, seconds=round(time.time() - t0, 1))
        print(f"    {key:38s}: coda_bpt={result['bpt']:.3f} acc={result['acc']:.3f}")

    # ── Joint baselines (compound token + coda marginalisation) ───────
    if tok_to_coda_str:
        for label, result in [
            ("majority", _eval_markov_joint(tok_train, tok_test, order=0,
                                            tok_to_coda=tok_to_coda_str)),
            ("markov1",  _eval_markov_joint(tok_train, tok_test, order=1,
                                            tok_to_coda=tok_to_coda_str)),
            ("markov2",  _eval_markov_joint(tok_train, tok_test, order=2,
                                            tok_to_coda=tok_to_coda_str)),
            ("kn5",      _eval_kn_joint(tok_train, tok_test, n=5,
                                        tok_to_coda=tok_to_coda_str)),
            ("kn5cache", _eval_kn_joint(tok_train, tok_test, n=5,
                                        tok_to_coda=tok_to_coda_str,
                                        cache_size=20, lam=0.15)),
        ]:
            key = f"baseline | {label} | joint"
            out[key] = dict(
                token_bpt=result["token_bpt"], token_acc=result["token_acc"],
                coda_marg_bpt=result["coda_marg_bpt"],
                coda_marg_acc=result["coda_marg_acc"],
                n_params=0, seconds=round(time.time() - t0, 1),
            )
            print(f"    {key:38s}: token_bpt={result['token_bpt']:.3f}"
                  f"  coda_marg={result['coda_marg_bpt']:.3f}"
                  f"  coda_marg_acc={result['coda_marg_acc']:.3f}")

    return out


def run_one_fold(fold_i, hersh_ids, clean_train_ids, val_ids, test_ids,
                  per_seq, V_coda, V_token, V_dt, token_to_coda, args):
    """Build windows for hersh, clean_train, val, test once; run all
    cells (single-phase mixed) and all data-schemes (multi-phase) on
    those windows."""
    hersh_W = build_windows(per_seq, hersh_ids, args.k)
    clean_W = build_windows(per_seq, clean_train_ids, args.k)
    mixed_W = _concat_windows(clean_W, hersh_W) if hersh_W["coda"].size(0) else clean_W
    val_W = build_windows(per_seq, val_ids, args.k)
    test_W = build_windows(per_seq, test_ids, args.k)
    print(
        f"  fold {fold_i}: hersh_windows={hersh_W['coda'].size(0)} "
        f"clean_train_windows={clean_W['coda'].size(0)} "
        f"mixed_windows={mixed_W['coda'].size(0)} "
        f"val={val_W['coda'].size(0)} test={test_W['coda'].size(0)}"
    )

    fold_out = {}

    # ------------------------------------------------------------------
    # Main 8-cell sweep (data scheme = M = single-phase mixed).
    # ------------------------------------------------------------------
    cells = [c for c in CELLS if c[0] in args.archs and c[1] in args.targets
             and c[2] in args.loss_aggs]
    for cell in cells:
        arch, target, loss_agg = cell
        torch.manual_seed(args.seed)
        model = build_model(arch=arch, target=target, V_coda=V_coda,
                             V_token=V_token, n_dt=V_dt, k=args.k)
        n_params = sum(p.numel() for p in model.parameters())
        t0 = time.time()
        try:
            train_scheme(
                model, [mixed_W], val_W, target, loss_agg,
                epochs=args.epochs, lrs=[args.lr], bs=args.bs,
                patience=args.patience, seed=args.seed,
            )
            metrics = eval_cell_metrics(model, test_W, target, token_to_coda,
                                         V_coda=V_coda, bs=args.bs)
        except Exception as e:
            print(f"    {cell}: FAILED ({type(e).__name__}: {e})")
            metrics = dict(error=str(e))
        metrics["n_params"] = int(n_params)
        metrics["seconds"] = round(time.time() - t0, 1)
        fold_out[" | ".join(cell)] = metrics
        print(_format_metrics_line(CELL_LABEL[cell], target, metrics))
        del model

    # ------------------------------------------------------------------
    # Data-scheme sweep on the configured cell (default: tfm | joint | per).
    # Probes whether curriculum (C) or mixed-then-finetune (MF) beats
    # single-phase mixed (M) once both phases get a fair budget.
    # ------------------------------------------------------------------
    scheme_cell = tuple(s.strip() for s in args.scheme_cell.split(","))
    if scheme_cell not in CELLS:
        raise ValueError(f"--scheme-cell {scheme_cell!r} not in CELLS")
    sch_arch, sch_target, sch_loss_agg = scheme_cell
    extra_schemes = [s for s in args.schemes if s != "M"]
    for scheme in extra_schemes:
        if scheme == "C":
            phase_Ws = [hersh_W, clean_W]
            lrs = [args.lr, args.lr_phase2]
        elif scheme == "MF":
            phase_Ws = [mixed_W, clean_W]
            lrs = [args.lr, args.lr_phase2]
        else:
            raise ValueError(f"unknown scheme: {scheme}")
        torch.manual_seed(args.seed)
        model = build_model(arch=sch_arch, target=sch_target, V_coda=V_coda,
                             V_token=V_token, n_dt=V_dt, k=args.k)
        n_params = sum(p.numel() for p in model.parameters())
        t0 = time.time()
        try:
            train_scheme(
                model, phase_Ws, val_W, sch_target, sch_loss_agg,
                epochs=args.epochs, lrs=lrs, bs=args.bs,
                patience=args.patience, seed=args.seed,
            )
            metrics = eval_cell_metrics(model, test_W, sch_target,
                                         token_to_coda, V_coda=V_coda,
                                         bs=args.bs)
        except Exception as e:
            print(f"    scheme {scheme}: FAILED ({type(e).__name__}: {e})")
            metrics = dict(error=str(e))
        metrics["n_params"] = int(n_params)
        metrics["seconds"] = round(time.time() - t0, 1)
        # Key: e.g. "scheme_C | tfm | joint | per"
        cell_key = f"scheme_{scheme} | " + " | ".join(scheme_cell)
        fold_out[cell_key] = metrics
        label = f"scheme {scheme}: " + " | ".join(scheme_cell)
        print(_format_metrics_line(label, sch_target, metrics))
        del model

    # ------------------------------------------------------------------
    # Classical reference baselines (Majority, Markov-1/2, KN-5, KN-5+cache).
    # Run on the same hersh + clean_train → test split as the neural cells.
    # ------------------------------------------------------------------
    all_train_ids = list(hersh_ids) + list(clean_train_ids)
    fold_out.update(run_baselines_fold(
        per_seq, all_train_ids, test_ids, token_to_coda=token_to_coda))

    return fold_out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--k", type=int, default=CONTEXT_K)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--archs", default="tfm,h")
    p.add_argument("--targets", default="coda,joint")
    p.add_argument("--loss-aggs", default="last,per")
    p.add_argument(
        "--schemes", default="M,C,MF",
        help="Comma-separated subset of {M,C,MF} to evaluate as data-"
             "scheme variants on the --scheme-cell. M is the implicit "
             "scheme of the main 8-cell sweep so it's reported there; "
             "extra schemes (C, MF) add per-fold runs on top.",
    )
    p.add_argument(
        "--scheme-cell", default="tfm,joint,per",
        help="Comma-separated (arch,target,loss_agg) used for the C/MF "
             "data-scheme runs. Must be one of the entries in CELLS.",
    )
    p.add_argument(
        "--lr-phase2", type=float, default=3e-4,
        help="LR for phase-2 in two-phase schemes (C, MF). Lower than "
             "--lr to mitigate catastrophic forgetting.",
    )
    p.add_argument("--smoke", action="store_true",
                   help="2 folds, 8 epochs — pipeline sanity check.")
    p.add_argument(
        "--out-prefix", default="full",
        help="Output filename stem. Results go to predict_results_<prefix>.{json,md}.",
    )
    args = p.parse_args(argv)
    args.archs = tuple(s.strip() for s in args.archs.split(",") if s.strip())
    args.targets = tuple(s.strip() for s in args.targets.split(",") if s.strip())
    args.loss_aggs = tuple(s.strip() for s in args.loss_aggs.split(",") if s.strip())
    args.schemes = tuple(s.strip() for s in args.schemes.split(",") if s.strip())
    if args.smoke:
        args.folds = 2
        args.epochs = 8
        args.patience = 3
    global OUT_JSON, OUT_MD, OUT_PARTIAL
    OUT_JSON = _OUT_DIR / f"predict_results_{args.out_prefix}.json"
    OUT_MD = _OUT_DIR / f"predict_results_{args.out_prefix}.md"
    OUT_PARTIAL = _OUT_DIR / f"predict_results_{args.out_prefix}.partial.json"

    print(f"loading whale corpus (this builds the compound Token vocab)…")
    per_seq, V_coda, V_token, V_dt, token_to_coda = load_corpus()
    seq_ids = sorted(per_seq.keys())
    hersh_ids = [s for s in seq_ids if _tier(s) == "hersh"]
    clean_ids = [s for s in seq_ids if _tier(s) == "clean"]
    print(
        f"  V_coda={V_coda} V_token={V_token} V_dt={V_dt}; "
        f"hersh={len(hersh_ids)}seqs clean={len(clean_ids)}seqs"
    )
    selected = [c for c in CELLS if c[0] in args.archs
                and c[1] in args.targets and c[2] in args.loss_aggs]
    print(f"running cells: {selected}")
    print(
        f"  folds={args.folds} epochs={args.epochs} patience={args.patience} "
        f"bs={args.bs} k={args.k}"
    )

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rng = np.random.default_rng(args.seed)

    OUT_PARTIAL.parent.mkdir(parents=True, exist_ok=True)
    fold_results = []
    t_global = time.time()

    for fi, (tr, te) in enumerate(kf.split(clean_ids)):
        clean_train = [clean_ids[i] for i in tr]
        clean_test = [clean_ids[i] for i in te]
        # Carve val from clean_train.
        rng2 = np.random.default_rng(args.seed + fi)
        ct = np.array(clean_train)
        rng2.shuffle(ct)
        n_val = max(1, int(args.val_frac * len(ct)))
        val_ids = ct[:n_val].tolist()
        clean_train_real = ct[n_val:].tolist()
        print(
            f"\nFOLD {fi}/{args.folds - 1}:  "
            f"clean_train={len(clean_train_real)} hersh={len(hersh_ids)} "
            f"val={len(val_ids)} test={len(clean_test)}"
        )
        fold_out = run_one_fold(
            fi, hersh_ids, clean_train_real, val_ids, clean_test, per_seq,
            V_coda, V_token, V_dt, token_to_coda, args,
        )
        fold_out["_meta"] = dict(
            fold=fi,
            n_clean_train_seqs=len(clean_train_real),
            n_hersh_seqs=len(hersh_ids),
            n_val_seqs=len(val_ids),
            n_test_seqs=len(clean_test),
            seconds_total=round(time.time() - t_global, 1),
        )
        fold_results.append(fold_out)
        # Save partial after every fold.
        OUT_PARTIAL.write_text(json.dumps(dict(
            folds=fold_results, V_coda=V_coda, V_token=V_token, V_dt=V_dt,
            args=vars(args),
        ), indent=2, default=str))

    # ------------------------------------------------------------------
    # Aggregate + write final outputs.
    # ------------------------------------------------------------------
    summary = {}
    cell_keys = [k for k in fold_results[0] if k != "_meta"]
    for ck in cell_keys:
        rows = [f[ck] for f in fold_results if ck in f and "error" not in f[ck]]
        if not rows:
            continue
        # Parse keys: "tfm | coda | last", "scheme_C | tfm | joint | per",
        # or "baseline | markov1 | coda".
        parts = ck.split(" | ")
        is_scheme = parts[0].startswith("scheme_")
        is_baseline = parts[0] == "baseline"
        if is_scheme:
            scheme = parts[0].replace("scheme_", "")
            arch, target, loss_agg = parts[1], parts[2], parts[3]
        elif is_baseline:
            scheme = "ref"
            arch, target, loss_agg = parts[1], parts[2], "n/a"
        else:
            scheme = "M"
            arch, target, loss_agg = parts[0], parts[1], parts[2]
        s = dict(
            n_folds=len(rows),
            n_params=int(rows[0].get("n_params", 0)),
            scheme=scheme, arch=arch, target=target, loss_agg=loss_agg,
        )
        if target == "coda":
            s["coda_bpt_mean"] = float(np.mean([r["coda_bpt"] for r in rows]))
            s["coda_bpt_std"] = float(np.std([r["coda_bpt"] for r in rows]))
            s["coda_acc_mean"] = float(np.mean([r["coda_acc"] for r in rows]))
        elif is_baseline and target == "joint":
            # Classical joint baseline: compound-token + coda marginalisation; no DT head.
            s["token_bpt_mean"] = float(np.mean([r["token_bpt"] for r in rows]))
            s["token_bpt_std"] = float(np.std([r["token_bpt"] for r in rows]))
            s["token_acc_mean"] = float(np.mean([r["token_acc"] for r in rows]))
            s["coda_marg_bpt_mean"] = float(np.mean([r["coda_marg_bpt"] for r in rows]))
            s["coda_marg_bpt_std"] = float(np.std([r["coda_marg_bpt"] for r in rows]))
            s["coda_marg_acc_mean"] = float(np.mean([r["coda_marg_acc"] for r in rows]))
        else:
            s["token_bpt_mean"] = float(np.mean([r["token_bpt"] for r in rows]))
            s["token_bpt_std"] = float(np.std([r["token_bpt"] for r in rows]))
            s["token_acc_mean"] = float(np.mean([r["token_acc"] for r in rows]))
            s["dt_bpt_mean"] = float(np.mean([r["dt_bpt"] for r in rows]))
            s["dt_bpt_std"] = float(np.std([r["dt_bpt"] for r in rows]))
            s["dt_acc_mean"] = float(np.mean([r["dt_acc"] for r in rows]))
            s["coda_marg_bpt_mean"] = float(np.mean([r["coda_marg_bpt"] for r in rows]))
            s["coda_marg_bpt_std"] = float(np.std([r["coda_marg_bpt"] for r in rows]))
            s["coda_marg_acc_mean"] = float(np.mean([r["coda_marg_acc"] for r in rows]))
        summary[ck] = s

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(dict(
        summary=summary, folds=fold_results,
        V_coda=V_coda, V_token=V_token, V_dt=V_dt,
        args=vars(args),
    ), indent=2, default=str))

    L = []
    L.append(
        "# Full-factorial whale-corpus benchmark "
        "(arch × target × loss-aggregation, tiered split)"
    )
    L.append("")
    L.append(
        f"5-fold KFold over **clean** tier (sharma2024_dswp + sharma2025_birth). "
        f"Per fold: train pool = remaining 4/5 of clean ∪ all hersh2022_pacific; "
        f"val = {int(100*args.val_frac)} % slice of clean_train; "
        f"test = held-out 1/5 of clean. Eval = held-out **last-position** bpt."
    )
    L.append("")
    L.append(
        f"V_coda = {V_coda}, V_token = {V_token} (compound rhythm·tempo·orn·rubato), "
        f"V_dt = {V_dt} (whale scheme: missing + 5 timing buckets + switch). "
        f"K = {args.k}. Per-cell: AdamW lr={args.lr}, bs={args.bs}, max "
        f"{args.epochs} epochs, ES patience {args.patience}."
    )
    L.append("")
    main_keys = [k for k in cell_keys
                 if not k.startswith("scheme_") and not k.startswith("baseline |")
                 and k in summary]
    scheme_keys = [k for k in cell_keys if k.startswith("scheme_") and k in summary]

    L.append("## Main sweep — coda-target cells (scheme = M)")
    L.append("")
    L.append("| arch | loss_agg | params | coda_bpt (↓) | coda_acc |")
    L.append("|------|----------|-------:|-------------:|---------:|")
    for ck in main_keys:
        s = summary[ck]
        if s["target"] != "coda":
            continue
        L.append(
            f"| {s['arch']} | {s['loss_agg']} | {s['n_params']:,} | "
            f"{s['coda_bpt_mean']:.3f} ± {s['coda_bpt_std']:.3f} | "
            f"{s['coda_acc_mean']:.3f} |"
        )
    L.append("")
    L.append("## Main sweep — joint-target cells (Token + DT bucket, scheme = M)")
    L.append("")
    L.append("| arch | loss_agg | params | token_bpt (↓) | dt_bpt (↓) | "
             "coda_marg_bpt (↓) | token_acc | dt_acc |")
    L.append("|------|----------|-------:|--------------:|-----------:|"
             "------------------:|----------:|-------:|")
    for ck in main_keys:
        s = summary[ck]
        if s["target"] != "joint":
            continue
        L.append(
            f"| {s['arch']} | {s['loss_agg']} | {s['n_params']:,} | "
            f"{s['token_bpt_mean']:.3f} ± {s['token_bpt_std']:.3f} | "
            f"{s['dt_bpt_mean']:.3f} ± {s['dt_bpt_std']:.3f} | "
            f"{s['coda_marg_bpt_mean']:.3f} ± {s['coda_marg_bpt_std']:.3f} | "
            f"{s['token_acc_mean']:.3f} | {s['dt_acc_mean']:.3f} |"
        )

    if scheme_keys:
        L.append("")
        L.append(f"## Data-scheme sweep — fixed cell `{args.scheme_cell}`")
        L.append("")
        L.append("Tests whether two-phase training (curriculum or mixed-then-"
                 "finetune) beats single-phase mixed (M) at the same "
                 "per-phase budget. Phase-2 runs with a smaller LR "
                 f"(`{args.lr_phase2:g}`) to mitigate catastrophic forgetting.")
        L.append("")
        L.append("| scheme | description | token_bpt (↓) | dt_bpt (↓) | "
                 "coda_marg_bpt (↓) |")
        L.append("|--------|-------------|--------------:|-----------:|"
                 "------------------:|")
        # Always include M baseline for the cell, drawn from main sweep.
        m_key = " | ".join(args.scheme_cell.split(","))
        if m_key in summary:
            sM = summary[m_key]
            if sM["target"] == "joint":
                L.append(
                    f"| M | mixed (single-phase, hersh ∪ clean_train) | "
                    f"{sM['token_bpt_mean']:.3f} ± {sM['token_bpt_std']:.3f} | "
                    f"{sM['dt_bpt_mean']:.3f} ± {sM['dt_bpt_std']:.3f} | "
                    f"{sM['coda_marg_bpt_mean']:.3f} ± {sM['coda_marg_bpt_std']:.3f} |"
                )
        for ck in scheme_keys:
            s = summary[ck]
            scheme = s["scheme"]
            desc = {
                "C": "curriculum (phase-1 hersh → phase-2 clean_train)",
                "MF": "mixed-then-finetune (phase-1 mixed → phase-2 clean_train)",
            }.get(scheme, scheme)
            if s["target"] == "joint":
                L.append(
                    f"| {scheme} | {desc} | "
                    f"{s['token_bpt_mean']:.3f} ± {s['token_bpt_std']:.3f} | "
                    f"{s['dt_bpt_mean']:.3f} ± {s['dt_bpt_std']:.3f} | "
                    f"{s['coda_marg_bpt_mean']:.3f} ± {s['coda_marg_bpt_std']:.3f} |"
                )
    # Classical baselines section
    _bl_pretty = {
        "majority": "Majority unigram",
        "markov1":  "Markov-1 (Laplace α=0.5)",
        "markov2":  "Markov-2 (Laplace α=0.5)",
        "kn5":      "KN 5-gram",
        "kn5cache": "KN 5-gram + recency cache (λ=0.15, win=20)",
    }
    baseline_coda_keys = [k for k in cell_keys
                          if k.startswith("baseline |") and k.endswith("| coda")
                          and k in summary]
    baseline_joint_keys = [k for k in cell_keys
                           if k.startswith("baseline |") and k.endswith("| joint")
                           and k in summary]
    if baseline_coda_keys:
        L.append("")
        L.append("## Classical reference baselines — coda target (rhythm-class only)")
        L.append("")
        L.append("Train = hersh ∪ clean_train (same as neural M scheme). "
                 "Markov uses Laplace α=0.5 smoothing. KN5 = Modified Kneser-Ney 5-gram. "
                 "KN5+cache = KN5 interpolated with a per-sequence recency cache "
                 "(size=20, λ=0.15).")
        L.append("")
        L.append("| model | coda_bpt (↓) | coda_acc |")
        L.append("|-------|-------------|---------|")
        for bk in baseline_coda_keys:
            s = summary[bk]
            lbl = _bl_pretty.get(bk.split(" | ")[1], bk)
            L.append(f"| {lbl} | "
                     f"{s['coda_bpt_mean']:.3f} ± {s['coda_bpt_std']:.3f} | "
                     f"{s['coda_acc_mean']:.3f} |")
    if baseline_joint_keys:
        L.append("")
        L.append("## Classical reference baselines — joint target (compound token, coda-marginalised)")
        L.append("")
        L.append("Same train/test split as neural M-scheme joint cells. "
                 "Models predict the compound token (rhythm·tempo·orn·rubato, V_token). "
                 "No DT head. coda_marg_bpt = log₂-bpt of the true rhythm class "
                 "after summing model probabilities over all compound tokens that share it. "
                 "coda_marg_acc = argmax compound token → map to rhythm class → check.")
        L.append("")
        L.append("| model | token_bpt (↓) | coda_marg_bpt (↓) | coda_marg_acc |")
        L.append("|-------|--------------|------------------|--------------|")
        for bk in baseline_joint_keys:
            s = summary[bk]
            lbl = _bl_pretty.get(bk.split(" | ")[1], bk)
            L.append(f"| {lbl} | "
                     f"{s['token_bpt_mean']:.3f} ± {s['token_bpt_std']:.3f} | "
                     f"{s['coda_marg_bpt_mean']:.3f} ± {s['coda_marg_bpt_std']:.3f} | "
                     f"{s['coda_marg_acc_mean']:.3f} |")
    L.append("")
    L.append("## Coda-comparable summary (all cells, on the rhythm-class target)")
    L.append("")
    L.append("Joint-target rows show **coda_marg_bpt** — the bpt obtained by "
             "marginalising softmax(Token) probabilities over their underlying "
             "rhythm class. Directly comparable to the coda-target rows.")
    L.append("")
    L.append("| arch | target | loss_agg | params | coda bpt (↓) |")
    L.append("|------|--------|----------|-------:|-------------:|")
    rows_for_sort = []
    for ck in cell_keys:
        if ck not in summary:
            continue
        s = summary[ck]
        if s["target"] == "coda":
            v = s["coda_bpt_mean"]
            sd = s["coda_bpt_std"]
        elif "coda_marg_bpt_mean" in s:
            v = s["coda_marg_bpt_mean"]
            sd = s["coda_marg_bpt_std"]
        else:
            continue  # token-only baselines: not coda-comparable
        rows_for_sort.append((v, sd, s["arch"], s["target"], s["loss_agg"],
                              s["scheme"], s["n_params"]))
    rows_for_sort.sort()
    L.append("")
    L.append("| scheme | arch | target | loss_agg | params | coda bpt (↓) |")
    L.append("|--------|------|--------|----------|-------:|-------------:|")
    for v, sd, arch, target, loss_agg, scheme, np_ in rows_for_sort:
        L.append(
            f"| {scheme} | {arch} | {target} | {loss_agg} | {np_:,} | "
            f"{v:.3f} ± {sd:.3f} |"
        )
    L.append("")
    L.append("## Per-fold detail")
    L.append("")
    for ck in cell_keys:
        if ck not in summary:
            continue
        if ck.startswith("baseline |"):
            continue
        L.append(f"### {ck}")
        L.append("")
        target = summary[ck]["target"]
        if target == "coda":
            L.append("| fold | coda_bpt | coda_acc |")
            L.append("|------|---------:|---------:|")
            for f in fold_results:
                if ck in f and "error" not in f[ck]:
                    L.append(f"| {f['_meta']['fold']} | "
                             f"{f[ck]['coda_bpt']:.3f} | "
                             f"{f[ck]['coda_acc']:.3f} |")
        else:
            L.append("| fold | token_bpt | dt_bpt | coda_marg_bpt |")
            L.append("|------|----------:|-------:|--------------:|")
            for f in fold_results:
                if ck in f and "error" not in f[ck]:
                    L.append(f"| {f['_meta']['fold']} | "
                             f"{f[ck]['token_bpt']:.3f} | "
                             f"{f[ck]['dt_bpt']:.3f} | "
                             f"{f[ck]['coda_marg_bpt']:.3f} |")
        L.append("")
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
