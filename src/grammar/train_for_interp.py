"""
Train a multi-task M7-architecture HookedTransformer on either the
whale or CHILDES dialogues CSV, and save a checkpoint that the interp
script can load.

The model has two heads sharing the residual stream:

  Coda head      Linear(d_model, V)         next-token (rhythm class
                                            for whale, lemma for CHILDES)
  DT head        Linear(d_model, n_buckets) next TimeDelta bucket
                                            (whale: 6 buckets incl. missing;
                                             CHILDES: 3 buckets intra/period/switch)

The DT head is only used for *generation* — it lets the continuation
script render whale output as ``bk Δt0.42 cn Δt~1.8 …`` and CHILDES
output as ``*A:\tword word .\n*B:\tword word .`` instead of a flat token
stream. All interp views (attention patterns, induction scores, logit
lens, etc.) read off the Coda head exactly as before.

Speaker identity for CHILDES is not predicted — at generation time we
alternate among the speakers actually observed in the seed
conversation whenever the DT head fires the ``switch`` bucket.

Why this exists separately from `predict_kfold.py`:

* `predict_kfold.py` discards the trained weights after each fold; we
  need a single persistent checkpoint per corpus.
* TransformerLens needs causal masking (the original M7 was un-masked
  but only ever read the final position; behaviorally the same on the
  benchmark, but different at intermediate positions).
* The original M7 trained loss only on position K. For autoregressive
  generation we want loss on every position — strictly more samples
  per window — and we want a second head supervising DT.

Usage
-----
    python -m src.grammar.train_for_interp --source whale
    python -m src.grammar.train_for_interp --source childes

Outputs
-------
    outputs/grammar/checkpoints/<source>/
        model.pt               HookedTransformer state_dict
        dt_head.pt             auxiliary DT-head state_dict
        config.json            HookedTransformerConfig dump + V, K, dt scheme,
                               held-out bpt for both heads
        vocab.json             list of decoded labels for each token id
        seeds.json             held-out continuation seeds, with full
                               TimeDelta + Whale columns so originals
                               can be rendered readably
        train_log.txt          per-epoch loss + final eval bpt
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.grammar.dt_buckets import dt_to_bucket, scheme_for
from src.grammar.m7_hooked import N_CTX, make_m7_config, make_m7_hooked
from src.grammar.whale_compound import load_compound_whale

ROOT = Path(__file__).resolve().parents[2]
WHALE_CSV = ROOT / "data" / "classified" / "whale_dialogues.csv"
WHALE_RHYTHM_INDEX = ROOT / "data" / "classified" / "rhythm_class_index.csv"
CHILDES_CSV = ROOT / "data" / "classified" / "childes_dialogues.csv"
CHILDES_INDEX = ROOT / "data" / "classified" / "childes_word_index.csv"
MULTILANG_CSV = ROOT / "data" / "classified" / "multilang_dialogues.csv"
MULTILANG_INDEX = ROOT / "data" / "classified" / "multilang_word_index.csv"

CKPT_DIR = ROOT / "outputs" / "grammar" / "checkpoints"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class Conversation:
    seq_id: str
    tokens: list[int]      # Coda ids
    dt_buckets: list[int]  # DT bucket ids parallel to tokens
    times: list[float]     # raw TimeDelta values (for original-rendering only)
    speakers: list[str]    # raw Whale-column values (for speaker rotation)


@dataclass
class Corpus:
    conversations: list[Conversation]
    vocab: list[str]   # vocab[i] = display string for token id i
    source: str
    dt_scheme_name: str
    n_dt_buckets: int


def _whale_vocab_from_decoder(decoder: dict[int, str], d_vocab: int) -> list[str]:
    """Compound-token decoder (`{id: 'rhythm_label|t<tempo>|o<orn>|r<rub>'}`)
    laid out as a vocab list."""
    vocab = [f"<{i}>" for i in range(d_vocab)]
    for cid, label in decoder.items():
        if 0 <= cid < d_vocab:
            vocab[cid] = label
    return vocab


def _load_childes_vocab(d_vocab: int) -> list[str]:
    if not CHILDES_INDEX.exists():
        return [str(i) for i in range(d_vocab)]
    idx = pd.read_csv(CHILDES_INDEX)
    vocab = ["<missing>"] * d_vocab
    for _, row in idx.iterrows():
        wid = int(row["word_id"])
        if 0 <= wid < d_vocab:
            vocab[wid] = str(row["word"])
    return vocab


def _load_index_vocab(index_path: Path, d_vocab: int) -> list[str]:
    if not index_path.exists():
        return [str(i) for i in range(d_vocab)]
    idx = pd.read_csv(index_path)
    vocab = ["<missing>"] * d_vocab
    for _, row in idx.iterrows():
        wid = int(row["word_id"])
        if 0 <= wid < d_vocab:
            vocab[wid] = str(row["word"])
    return vocab


def load_corpus(source: str, n_ctx: int = N_CTX) -> Corpus:
    """Load the per-source dialogues CSV and bin its TimeDelta channel.

    For whale we *replace* the rhythm-only `Coda` column with the
    compound (rhythm × tempo_bin × orn × rubato) token id from
    `whale_compound.load_compound_whale`, which expands V from 131 →
    ~467 attested combinations. This matches the vocabulary that the
    parallel-session continuations script and the side-by-side
    benchmark (`outputs/grammar/childes_vs_whale.md`) use, so
    interpretability views describe the same model.
    """
    whale_decoder: dict[int, str] | None = None
    if source == "whale":
        df, whale_decoder = load_compound_whale(WHALE_CSV)
    elif source == "childes":
        df = pd.read_csv(CHILDES_CSV)
    elif source == "multilang":
        df = pd.read_csv(MULTILANG_CSV)
    else:
        raise ValueError(f"unknown source: {source}")

    df = df.sort_values(["sequenceId", "itemPosition"])
    convs: list[Conversation] = []
    for sid, g in df.groupby("sequenceId", sort=False):
        toks = g["Coda"].astype(int).tolist()
        if len(toks) < n_ctx + 1:
            continue
        times = g["TimeDelta"].astype(float).tolist()
        has_ts = g["has_timestamps"].astype(int).tolist()
        speakers = g["Whale"].astype(str).tolist()
        # Flag a speaker change at position i when speakers[i] != speakers[i-1]
        # AND both endpoints have timestamps (Hersh-tier rows are all UNK and
        # masked, so they correctly produce no switches). The whale scheme's
        # `switched` bucket overrides the timing bucket; CHILDES + multilang
        # ignore the flag.
        switched_flags = [False]
        for i in range(1, len(speakers)):
            switched_flags.append(
                speakers[i] != speakers[i - 1]
                and has_ts[i] == 1
                and has_ts[i - 1] == 1
            )
        buckets = [
            dt_to_bucket(source, dt, hts, switched=sw)
            for dt, hts, sw in zip(times, has_ts, switched_flags)
        ]
        convs.append(Conversation(
            seq_id=str(sid),
            tokens=toks,
            dt_buckets=buckets,
            times=times,
            speakers=speakers,
        ))

    d_vocab = int(df["Coda"].max()) + 1
    if source == "whale":
        vocab = _whale_vocab_from_decoder(whale_decoder or {}, d_vocab)
    elif source == "multilang":
        vocab = _load_index_vocab(MULTILANG_INDEX, d_vocab)
    else:
        vocab = _load_childes_vocab(d_vocab)
    scheme = scheme_for(source)
    return Corpus(
        conversations=convs,
        vocab=vocab,
        source=source,
        dt_scheme_name=scheme.name,
        n_dt_buckets=scheme.n_buckets,
    )


# ---------------------------------------------------------------------------
# Sliding-window dataset
# ---------------------------------------------------------------------------


def make_windows(convs: list[Conversation], k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (tokens, dt_buckets) tensors of shape (N, k+1) each.

    Window i of conversation c is c.tokens[i : i+k+1] (parallel for
    dt_buckets). The training loop feeds the first k positions and
    supervises the next-token / next-dt-bucket prediction at every
    position.
    """
    tok_rows: list[list[int]] = []
    dt_rows: list[list[int]] = []
    for c in convs:
        for i in range(len(c.tokens) - k):
            tok_rows.append(c.tokens[i : i + k + 1])
            dt_rows.append(c.dt_buckets[i : i + k + 1])
    if not tok_rows:
        return torch.empty(0, k + 1, dtype=torch.long), torch.empty(0, k + 1, dtype=torch.long)
    return (
        torch.tensor(tok_rows, dtype=torch.long),
        torch.tensor(dt_rows, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Multi-task forward
# ---------------------------------------------------------------------------


def forward_multi(model, dt_head: nn.Linear, tokens: torch.Tensor):
    """Run HookedTransformer with cache; project the last layer's
    `resid_post` through the model's `ln_final` to feed both the Coda
    head (the model's `unembed`) and the auxiliary `dt_head`.

    Returns (coda_logits, dt_logits) of shapes (B, K, V) and (B, K, n_buckets).
    """
    _, cache = model.run_with_cache(tokens)
    last_resid = cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"]
    normalized = model.ln_final(last_resid)
    coda_logits = model.unembed(normalized)
    dt_logits = dt_head(normalized)
    return coda_logits, dt_logits


def shifted_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean per-position next-token CE: predict targets[:, 1:] from
    logits[:, :-1]. Shapes (B, K, V), (B, K) → scalar loss."""
    B, K, V = logits.shape
    pred = logits[:, :-1].reshape(-1, V)
    tgt = targets[:, 1:].reshape(-1)
    return nn.functional.cross_entropy(pred, tgt)


# ---------------------------------------------------------------------------
# Eval — final-position bpt for both heads, comparable to the M7 benchmark
# ---------------------------------------------------------------------------


def eval_final_position(
    model, dt_head: nn.Linear,
    tok_windows: torch.Tensor, dt_windows: torch.Tensor,
):
    """Predict position k from positions 0..k-1 for both heads.

    Returns dict with coda_bpt, coda_acc, dt_bpt, dt_acc.
    """
    model.eval()
    bs = 256
    sums = {"coda_nll": 0.0, "coda_correct": 0, "dt_nll": 0.0, "dt_correct": 0,
            "n": 0}
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for i in range(0, tok_windows.size(0), bs):
            inp = tok_windows[i : i + bs, :-1]
            tok_tgt = tok_windows[i : i + bs, -1]
            dt_tgt = dt_windows[i : i + bs, -1]
            coda_logits, dt_logits = forward_multi(model, dt_head, inp)
            coda_last = coda_logits[:, -1]
            dt_last = dt_logits[:, -1]
            sums["coda_nll"] += loss_fn(coda_last, tok_tgt).item()
            sums["coda_correct"] += (coda_last.argmax(-1) == tok_tgt).sum().item()
            sums["dt_nll"] += loss_fn(dt_last, dt_tgt).item()
            sums["dt_correct"] += (dt_last.argmax(-1) == dt_tgt).sum().item()
            sums["n"] += tok_tgt.numel()
    n = max(sums["n"], 1)
    return dict(
        coda_bpt=sums["coda_nll"] / n / math.log(2),
        coda_acc=sums["coda_correct"] / n,
        dt_bpt=sums["dt_nll"] / n / math.log(2),
        dt_acc=sums["dt_correct"] / n,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    model,
    dt_head: nn.Linear,
    train_tok: torch.Tensor, train_dt: torch.Tensor,
    val_tok: torch.Tensor, val_dt: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    bs: int,
    dt_weight: float,
    log_lines: list[str],
    device: str,
    patience: int = 5,
):
    """Joint causal-LM training. Loss = CE(coda) + dt_weight * CE(dt),
    averaged over all (batch × position) next-token predictions per
    window.
    """
    model.to(device)
    dt_head.to(device)
    params = list(model.parameters()) + list(dt_head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    n = train_tok.size(0)

    best_val = float("inf")
    best_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_dt = {k: v.detach().clone() for k, v in dt_head.state_dict().items()}
    since = 0

    train_inputs_tok = train_tok[:, :-1]
    train_inputs_dt = train_dt[:, :-1]
    val_inputs_tok = val_tok[:, :-1]
    val_inputs_dt = val_dt[:, :-1]
    # The CLM targets are the same windows shifted by 1. To keep the
    # per-position loss honest we feed length-K windows and compute
    # next-token loss inside `shifted_ce`, which means each window
    # contributes K-1 predictions (positions 0..K-2 predicting 1..K-1).
    # The held-out (K+1)-th column goes to the final-position eval.

    for ep in range(1, epochs + 1):
        model.train(); dt_head.train()
        perm = torch.randperm(n)
        running_coda = 0.0
        running_dt = 0.0
        nbatches = 0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            tok_b = train_inputs_tok[idx].to(device)
            dt_b = train_inputs_dt[idx].to(device)
            opt.zero_grad()
            coda_logits, dt_logits = forward_multi(model, dt_head, tok_b)
            coda_loss = shifted_ce(coda_logits, tok_b)
            dt_loss = shifted_ce(dt_logits, dt_b)
            loss = coda_loss + dt_weight * dt_loss
            loss.backward()
            opt.step()
            running_coda += coda_loss.item()
            running_dt += dt_loss.item()
            nbatches += 1
        train_coda = running_coda / max(nbatches, 1)
        train_dt = running_dt / max(nbatches, 1)

        model.eval(); dt_head.eval()
        with torch.no_grad():
            coda_logits, dt_logits = forward_multi(
                model, dt_head, val_inputs_tok.to(device))
            val_coda = shifted_ce(coda_logits, val_inputs_tok.to(device)).item()
            val_dt = shifted_ce(dt_logits, val_inputs_dt.to(device)).item()
        val_combined = val_coda + dt_weight * val_dt

        line = (f"epoch {ep:3d}  train coda={train_coda:.3f} dt={train_dt:.3f}"
                f"  | val coda={val_coda:.3f} dt={val_dt:.3f}")
        print(line)
        log_lines.append(line)

        if val_combined < best_val - 1e-4:
            best_val = val_combined
            best_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_dt = {k: v.detach().clone() for k, v in dt_head.state_dict().items()}
            since = 0
        else:
            since += 1
            if since >= patience:
                log_lines.append(f"early stop at epoch {ep}")
                print(f"early stop at epoch {ep}")
                break

    model.load_state_dict(best_model)
    dt_head.load_state_dict(best_dt)
    return model, dt_head


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["whale", "childes", "multilang"],
                   required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--dt-weight", type=float, default=0.5,
                   help="Loss weight on the DT head; coda head is fixed at 1.0.")
    p.add_argument("--n-ctx", type=int, default=N_CTX,
                   help=f"Context length K (window is K+1). Default {N_CTX}.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    corpus = load_corpus(args.source, n_ctx=args.n_ctx)
    print(f"loaded {args.source}: {len(corpus.conversations)} conversations, "
          f"V={len(corpus.vocab)}, dt scheme={corpus.dt_scheme_name} "
          f"({corpus.n_dt_buckets} buckets), K={args.n_ctx}")

    n_total = len(corpus.conversations)
    if n_total < args.n_seeds + 5:
        raise SystemExit(f"too few conversations ({n_total}) to hold out seeds")

    if args.source == "multilang":
        # For demo continuations to be legible we pick the held-out seeds
        # from the clean tiers — one from `zh_birth` and one from the
        # unmasked half of `zh_dswp` — so the speaker channel and DT
        # channel are both present in the prefix that feeds the
        # continuation generator. The Hersh-equiv majority (UNK +
        # has_timestamps=0) still trains the model; we just don't pick
        # display seeds from there.
        birth_pool = [
            i for i, c in enumerate(corpus.conversations)
            if c.seq_id.startswith("zh_birth::")
            and not any("::UNK" in s for s in c.speakers)
        ]
        dswp_clean_pool = [
            i for i, c in enumerate(corpus.conversations)
            if c.seq_id.startswith("zh_dswp::")
            and not any("::UNK" in s for s in c.speakers)
        ]
        rng.shuffle(birth_pool)
        rng.shuffle(dswp_clean_pool)
        seed_idx: list[int] = []
        if birth_pool:
            seed_idx.append(birth_pool[0])
        if dswp_clean_pool:
            seed_idx.append(dswp_clean_pool[0])
        if not seed_idx:
            raise SystemExit("no clean (zh_birth / zh_dswp-clean) seeds available")
        held = set(seed_idx)
        rest_idx = [i for i in range(n_total) if i not in held]
        rng.shuffle(rest_idx)
        print(f"held-out seeds (clean tiers only): "
              f"{[corpus.conversations[i].seq_id for i in seed_idx]}")
    else:
        perm = list(range(n_total))
        rng.shuffle(perm)
        seed_idx = perm[: args.n_seeds]
        rest_idx = perm[args.n_seeds :]
    n_val = max(1, int(args.val_frac * len(rest_idx)))
    val_idx = rest_idx[:n_val]
    train_idx = rest_idx[n_val:]

    train_convs = [corpus.conversations[i] for i in train_idx]
    val_convs = [corpus.conversations[i] for i in val_idx]

    train_tok, train_dt = make_windows(train_convs, args.n_ctx)
    val_tok, val_dt = make_windows(val_convs, args.n_ctx)
    print(f"train windows: {tuple(train_tok.shape)};  val windows: {tuple(val_tok.shape)}")

    if train_tok.size(0) == 0 or val_tok.size(0) == 0:
        raise SystemExit("not enough windows after split")

    model = make_m7_hooked(d_vocab=len(corpus.vocab), n_ctx=args.n_ctx)
    dt_head = nn.Linear(model.cfg.d_model, corpus.n_dt_buckets)
    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in dt_head.parameters())
    print(f"model + dt_head params: {n_params:,}")

    log_lines: list[str] = [
        f"source: {args.source}",
        f"seed: {args.seed}",
        f"V: {len(corpus.vocab)}; n_dt_buckets: {corpus.n_dt_buckets}; K: {args.n_ctx}",
        f"n_params: {n_params:,}",
        f"dt_weight: {args.dt_weight}",
        f"n_train_windows: {train_tok.size(0)}; n_val_windows: {val_tok.size(0)}",
        f"held-out seeds: {[corpus.conversations[i].seq_id for i in seed_idx]}",
    ]

    model, dt_head = train(
        model, dt_head,
        train_tok, train_dt, val_tok, val_dt,
        epochs=args.epochs, lr=args.lr, bs=args.bs,
        dt_weight=args.dt_weight,
        log_lines=log_lines, device=args.device,
    )

    model = model.cpu(); dt_head = dt_head.cpu()
    final = eval_final_position(model, dt_head, val_tok, val_dt)
    log_lines.append("")
    log_lines.append(f"val final-position coda bpt: {final['coda_bpt']:.4f}  "
                     f"(M7 benchmark on whale: 3.186)")
    log_lines.append(f"val final-position coda acc: {final['coda_acc']:.4f}")
    log_lines.append(f"val final-position dt   bpt: {final['dt_bpt']:.4f}")
    log_lines.append(f"val final-position dt   acc: {final['dt_acc']:.4f}")
    for ln in log_lines[-4:]:
        print(ln)

    # Save
    out_dir = CKPT_DIR / args.source
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    torch.save(dt_head.state_dict(), out_dir / "dt_head.pt")
    cfg = make_m7_config(d_vocab=len(corpus.vocab), n_ctx=args.n_ctx)
    cfg_dict = {k: v for k, v in cfg.__dict__.items()
                if isinstance(v, (int, float, str, bool, type(None), list, tuple))}
    cfg_dict["source"] = args.source
    cfg_dict["n_params"] = n_params
    cfg_dict["dt_scheme_name"] = corpus.dt_scheme_name
    cfg_dict["n_dt_buckets"] = corpus.n_dt_buckets
    cfg_dict["dt_weight"] = args.dt_weight
    for k, v in final.items():
        cfg_dict[f"val_{k}"] = v
    (out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))
    (out_dir / "vocab.json").write_text(json.dumps(corpus.vocab, ensure_ascii=False, indent=2))
    seeds_payload = []
    for i in seed_idx:
        c = corpus.conversations[i]
        seeds_payload.append(dict(
            sequenceId=c.seq_id,
            tokens=c.tokens,
            dt_buckets=c.dt_buckets,
            times=c.times,
            speakers=c.speakers,
            length=len(c.tokens),
        ))
    (out_dir / "seeds.json").write_text(json.dumps(seeds_payload, indent=2))
    (out_dir / "train_log.txt").write_text("\n".join(log_lines))
    print(f"\nwrote checkpoint to {out_dir}/")


if __name__ == "__main__":
    main()
