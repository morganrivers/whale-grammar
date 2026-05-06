"""
Generate sampled continuations from the locked MiniTransformer-DT for
both whale and CHILDES UK English.

Protocol (per source):
  1. Hold out 3 sequences (random, fixed seed) as fresh seeds.
  2. Train one production MiniTransformer-DT on the remainder.
  3. For each seed: feed first K=8 tokens, autoregressively sample 3
     continuations (T=0.9, top-k=40) of length = mean training-set
     sequence length minus K.
  4. Decode token IDs back to readable form using the per-source index
     (rhythm_class_index.csv for whale; childes_word_index.csv for CHILDES).
  5. Write outputs/grammar/continuations.md.

DT during generation: each generated token gets the *mean* `(log_dt,
has_ts)` from the training corpus, so the DT channel contributes a
constant bias on the new positions and the model's continuation is
governed by the token-id stream as the user asked.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn

from src.grammar.predict_kfold import (
    DT_LOG_OFFSET, MiniTransformerDT, PAD, build_windows_with_dt,
    per_seq, train_torch_dt,
)
from src.grammar.predict_kfold_compare import per_seq_timedelta

ROOT = Path(__file__).resolve().parents[2]
WHALE_CSV = ROOT / "data" / "classified" / "whale_dialogues.csv"
WHALE_INDEX = ROOT / "data" / "classified" / "rhythm_class_index.csv"
CHILDES_CSV = ROOT / "data" / "classified" / "childes_dialogues.csv"
CHILDES_INDEX = ROOT / "data" / "classified" / "childes_word_index.csv"
OUT_MD = ROOT / "outputs" / "grammar" / "continuations.md"
OUT_JSON = ROOT / "outputs" / "grammar" / "continuations.json"

ARCH = dict(d=64, n_layers=2, n_heads=4)
K = 8
N_SEEDS = 3
N_SAMPLES_PER_SEED = 3
TEMPERATURE = 0.9
TOP_K = 40
SEED_RNG = 1234


def _encode_dt(dt_value: float) -> tuple[float, float]:
    """Same as predict_kfold._encode_dt, duplicated so this module is
    self-contained for generation."""
    if dt_value < 0:
        return float(np.log(DT_LOG_OFFSET)), 0.0
    return float(np.log(DT_LOG_OFFSET + max(dt_value, 0.0))), 1.0


def load_index(source: str) -> dict[int, str]:
    """int -> readable label.

    For whale, returns the compound-token decoder
    (`rhythm_label|t<tempo>|o<orn>|r<rubato>`); for CHILDES, the lemma index.
    """
    if source == "whale":
        from src.grammar.whale_compound import load_compound_whale
        _df, decoder = load_compound_whale(WHALE_CSV)
        return decoder
    df = pd.read_csv(CHILDES_INDEX)
    return {int(r["word_id"]): str(r["word"]) for _, r in df.iterrows()}


def pick_seeds(seq_dict: dict[str, list[str]], rng: np.random.Generator,
               n: int, min_len: int) -> list[str]:
    eligible = [sid for sid, s in seq_dict.items() if len(s) >= min_len]
    chosen_idx = rng.choice(len(eligible), size=n, replace=False)
    return [eligible[i] for i in sorted(chosen_idx)]


def build_train_windows(
    seq_dict: dict[str, list[str]], dt_dict: dict[str, list[float]],
    train_ids: Sequence[str], k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y, DT = build_windows_with_dt(seq_dict, dt_dict, train_ids, k)
    return X, y, DT


def _train_production_model(
    X_train: np.ndarray, DT_train: np.ndarray, y_train: np.ndarray,
    classes: list[str], k: int, seed: int,
) -> tuple[MiniTransformerDT, LabelEncoder]:
    le = LabelEncoder().fit(np.array(classes))
    pad_id = int(le.transform([PAD])[0])
    Xtr_int = np.array([le.transform(row) for row in X_train])
    ytr_int = le.transform(y_train)
    V = len(le.classes_)

    Xtr_t = torch.from_numpy(Xtr_int).long()
    DTtr_t = torch.from_numpy(DT_train).float()
    ytr_t = torch.from_numpy(ytr_int).long()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xtr_t))
    n_val = max(64, int(0.1 * len(Xtr_t)))
    val_idx = torch.from_numpy(perm[:n_val])
    tr_idx = torch.from_numpy(perm[n_val:])

    model = MiniTransformerDT(V, k, **ARCH)
    train_torch_dt(
        model,
        Xtr_t[tr_idx], DTtr_t[tr_idx], ytr_t[tr_idx],
        Xtr_t[val_idx], DTtr_t[val_idx], ytr_t[val_idx],
        seed=seed,
    )
    return model, le


def _top_k_sample(logits: torch.Tensor, temperature: float, top_k: int,
                  rng: torch.Generator) -> int:
    """Sample one index from a 1-D logits tensor."""
    logits = logits / max(temperature, 1e-6)
    vals, idx = torch.topk(logits, k=min(top_k, logits.numel()))
    probs = torch.softmax(vals, dim=-1)
    pick = torch.multinomial(probs, 1, generator=rng).item()
    return int(idx[pick].item())


def generate(
    model: MiniTransformerDT, le: LabelEncoder, prefix_ids: list[int],
    n_steps: int, mean_dt_feat: tuple[float, float], k: int,
    temperature: float, top_k: int, rng: torch.Generator,
) -> list[int]:
    """Autoregressive sampling with constant DT feature on generated
    positions (mean training (log_dt, has_ts))."""
    pad_id = int(le.transform([PAD])[0])
    out_ids = list(prefix_ids)
    # DT per emitted position: assume the seed prefix uses mean-DT for the
    # newly attended window (we do not require the original DT here — the
    # task is "show how it continues" and the user's framing emphasises
    # the token stream).
    dt_log_mean, has_ts_mean = mean_dt_feat
    dt_buffer = [(dt_log_mean, has_ts_mean)] * len(out_ids)
    model.eval()
    with torch.no_grad():
        for _ in range(n_steps):
            ctx_ids = out_ids[-k:]
            ctx_dt = dt_buffer[-k:]
            pad_n = k - len(ctx_ids)
            if pad_n > 0:
                ctx_ids = [pad_id] * pad_n + ctx_ids
                ctx_dt = [(dt_log_mean, 0.0)] * pad_n + ctx_dt
            x = torch.tensor([ctx_ids], dtype=torch.long)
            dt = torch.tensor([ctx_dt], dtype=torch.float)
            logits = model(x, dt)[0]
            nxt = _top_k_sample(logits, temperature, top_k, rng)
            out_ids.append(nxt)
            dt_buffer.append((dt_log_mean, has_ts_mean))
    return out_ids


def decode_ids(le: LabelEncoder, idx_to_label: dict[int, str],
               token_ids: list[int]) -> list[str]:
    """LabelEncoder int -> source token id (string) -> readable label."""
    out = []
    for i in token_ids:
        try:
            tok = le.inverse_transform([i])[0]
        except Exception:
            out.append("?")
            continue
        if tok == PAD:
            out.append("·")
            continue
        try:
            out.append(idx_to_label[int(tok)])
        except (KeyError, ValueError):
            out.append(str(tok))
    return out


def run_source(source: str, csv: Path, rng: np.random.Generator) -> dict:
    print(f"\n=== {source} ===")
    if source == "whale":
        # Use the compound (rhythm × tempo × orn × rubato) target.
        from src.grammar.whale_compound import load_compound_whale
        tokens, _decoder = load_compound_whale(csv)
    else:
        tokens = pd.read_csv(csv)
    seq_dict = per_seq(tokens, "Coda")
    dt_dict = per_seq_timedelta(tokens)
    seq_ids = sorted(seq_dict.keys())
    print(f"  {len(seq_ids)} sequences, {sum(len(v) for v in seq_dict.values()):,} tokens")

    seq_lengths = [len(seq_dict[sid]) for sid in seq_ids]
    mean_len = int(round(np.mean(seq_lengths)))
    median_len = int(round(np.median(seq_lengths)))
    print(f"  mean seq length {mean_len} (median {median_len})")

    seed_ids = pick_seeds(seq_dict, rng, N_SEEDS, min_len=K + 4)
    train_ids = [sid for sid in seq_ids if sid not in set(seed_ids)]
    print(f"  picked {len(seed_ids)} seeds; training on {len(train_ids)} sequences")
    for sid in seed_ids:
        print(f"    seed: {sid} (len={len(seq_dict[sid])})")

    X_train, y_train, DT_train = build_windows_with_dt(seq_dict, dt_dict, train_ids, K)
    train_tokens = {t for sid in train_ids for t in seq_dict[sid]}
    seed_tokens = {t for sid in seed_ids for t in seq_dict[sid]}
    classes = sorted(train_tokens | seed_tokens | {PAD})

    # mean DT feature on training windows (across positions and rows)
    dt_log_mean = float(DT_train[..., 0].mean())
    has_ts_mean = float(DT_train[..., 1].mean())
    print(f"  mean dt feature (log_dt, has_ts) = ({dt_log_mean:.3f}, {has_ts_mean:.3f})")

    model, le = _train_production_model(X_train, DT_train, y_train, classes, K, seed=SEED_RNG)
    idx_to_label = load_index(source)

    n_continuation = max(8, mean_len - K)
    print(f"  generating {n_continuation} tokens after each {K}-token seed prefix")

    torch_rng = torch.Generator().manual_seed(SEED_RNG)
    out_seeds = []
    for sid in seed_ids:
        full = seq_dict[sid]
        prefix_str = full[:K]
        prefix_int = [int(le.transform([s])[0]) for s in prefix_str]
        prefix_decoded = decode_ids(le, idx_to_label, prefix_int)
        truth_decoded = decode_ids(le, idx_to_label, [int(le.transform([s])[0]) for s in full])
        continuations = []
        for s in range(N_SAMPLES_PER_SEED):
            gen_ids = generate(
                model, le, prefix_int, n_continuation,
                (dt_log_mean, has_ts_mean), K, TEMPERATURE, TOP_K, torch_rng,
            )
            cont_only = gen_ids[K:]
            cont_decoded = decode_ids(le, idx_to_label, cont_only)
            continuations.append(dict(
                token_ids=cont_only,
                tokens=cont_decoded,
            ))
        out_seeds.append(dict(
            sequenceId=sid,
            length=len(full),
            prefix=prefix_decoded,
            actual_continuation=truth_decoded[K:K + n_continuation],
            continuations=continuations,
        ))
    return dict(
        source=source,
        n_sequences=len(seq_ids),
        mean_seq_length=mean_len,
        median_seq_length=median_len,
        n_train=len(train_ids),
        seed_count=len(seed_ids),
        n_continuation_tokens=n_continuation,
        seeds=out_seeds,
    )


def fmt_tokens(source: str, toks: list[str]) -> str:
    if source == "whale":
        # Compound tokens are space-separated; use comma-delimit for readability.
        return ", ".join(toks)
    return " ".join(toks)


def write_md(report: dict) -> None:
    L: list[str] = []
    L.append("# Sampled continuations — whale vs CHILDES Eng-UK")
    L.append("")
    L.append(
        f"Locked architecture: MiniTransformer-DT, "
        f"{ARCH['n_layers']}L, {ARCH['n_heads']}h, d={ARCH['d']}, K={K} + DT. "
        f"Sampling: T={TEMPERATURE}, top-k={TOP_K}, multinomial. "
        f"For each source we hold out **{N_SEEDS}** sequences (not seen in "
        "training), feed the first K=8 tokens as a prefix, and "
        f"autoregressively sample **{N_SAMPLES_PER_SEED}** continuations of "
        "length ≈ mean training-set sequence length."
    )
    L.append("")
    L.append(
        "Whale tokens are rhythm-class labels (e.g. `1+1+5`); CHILDES tokens "
        "are `%mor` lemmas (e.g. `go`, `the`, `book`). For both, `?` marks an "
        "out-of-vocabulary id and `·` marks a PAD slot."
    )
    L.append("")
    for src in report["sources"]:
        L.append(f"## {src['source']}  "
                 f"(mean seq length = {src['mean_seq_length']}, "
                 f"continuations are {src['n_continuation_tokens']} tokens long)")
        L.append("")
        for si, seed in enumerate(src["seeds"]):
            L.append(f"### Seed {si+1} — `{seed['sequenceId']}` (length {seed['length']})")
            L.append("")
            L.append("**prefix (8 tokens, fed verbatim):**")
            L.append("")
            L.append("> " + fmt_tokens(src["source"], seed["prefix"]))
            L.append("")
            L.append("**actual continuation in held-out data (for reference, "
                     "model never saw this):**")
            L.append("")
            L.append("> " + fmt_tokens(src["source"], seed["actual_continuation"]))
            L.append("")
            L.append("**3 sampled continuations:**")
            L.append("")
            for ci, cont in enumerate(seed["continuations"]):
                L.append(f"{ci+1}. " + fmt_tokens(src["source"], cont["tokens"]))
            L.append("")
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT_MD}")
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {OUT_JSON}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=("whale", "childes", "both"), default="both")
    args = p.parse_args(argv)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED_RNG)
    sources = []
    if args.source in ("whale", "both"):
        sources.append(run_source("whale", WHALE_CSV, rng))
    if args.source in ("childes", "both"):
        sources.append(run_source("childes_uk", CHILDES_CSV, rng))
    write_md(dict(sources=sources, arch=ARCH, K=K, temperature=TEMPERATURE,
                  top_k=TOP_K, n_seeds=N_SEEDS, n_samples=N_SAMPLES_PER_SEED))


if __name__ == "__main__":
    main()
