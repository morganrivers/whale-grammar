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


# ---------------------------------------------------------------------------
# Tier partitioning — for the two-stage curriculum vs mixed comparison
# ---------------------------------------------------------------------------


# Held-back / clean tier prefixes per source. The held-back tier (Hersh-Pacific
# for whale, Hersh-equiv for multilang) has Whale=UNK + has_timestamps=0 — it
# carries the bulk of the tokens but has no labelled timing/identity. The clean
# tier is the labelled Caribbean DSWP+Birth recordings (whale) or the parallel
# Mandarin Birth+DSWP allocations (multilang).
TIER_PREFIXES = {
    "whale": {
        "hersh": ("hersh2022_pacific::",),
        "clean": ("sharma2024_dswp::", "sharma2025_birth::"),
    },
    "multilang": {
        "hersh": ("multilang::",),
        "clean": ("zh_dswp::", "zh_birth::"),
    },
}


def tier_of(seq_id: str, source: str) -> str:
    """Return 'hersh' / 'clean' / 'other' for a sequenceId under the given source."""
    prefixes = TIER_PREFIXES.get(source)
    if prefixes is None:
        return "other"
    if any(seq_id.startswith(p) for p in prefixes["hersh"]):
        return "hersh"
    if any(seq_id.startswith(p) for p in prefixes["clean"]):
        return "clean"
    return "other"


def _save_checkpoint(
    out_dir: Path,
    model, dt_head,
    corpus: Corpus,
    args,
    final: dict,
    seed_idx: list[int],
    log_lines: list[str],
    n_params: int,
) -> None:
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


def _new_model(corpus: Corpus, n_ctx: int, seed: int) -> tuple[object, nn.Linear, int]:
    """Re-seeded fresh model + dt_head (for fair head-to-head between
    mixed-baseline and curriculum runs)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_m7_hooked(d_vocab=len(corpus.vocab), n_ctx=n_ctx)
    dt_head = nn.Linear(model.cfg.d_model, corpus.n_dt_buckets)
    n_params = sum(p.numel() for p in model.parameters()) + sum(
        p.numel() for p in dt_head.parameters())
    return model, dt_head, n_params


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["whale", "childes", "multilang"],
                   required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Validation fraction of the *clean* tier; held-out is "
                        "always clean (DSWP+Birth) regardless of mode.")
    p.add_argument("--epochs", type=int, default=40,
                   help="Per-phase epoch budget. Mixed baseline uses --epochs "
                        "on the merged set; curriculum uses --epochs on each "
                        "of phase-1 (Hersh) and phase-2 (clean).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--dt-weight", type=float, default=0.5,
                   help="Loss weight on the DT head; coda head is fixed at 1.0.")
    p.add_argument("--n-ctx", type=int, default=N_CTX,
                   help=f"Context length K (window is K+1). Default {N_CTX}.")
    p.add_argument("--mode", choices=["compare", "mixed", "curriculum"],
                   default="compare",
                   help="`compare` runs both mixed-baseline and curriculum "
                        "side-by-side and reports the held-out gap. `mixed` / "
                        "`curriculum` runs only one.")
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

    # Tier partition
    hersh_idx: list[int] = []
    clean_idx: list[int] = []
    other_idx: list[int] = []
    for i, c in enumerate(corpus.conversations):
        t = tier_of(c.seq_id, args.source)
        if t == "hersh":
            hersh_idx.append(i)
        elif t == "clean":
            clean_idx.append(i)
        else:
            other_idx.append(i)
    print(f"tier partition: hersh={len(hersh_idx)}, clean={len(clean_idx)}, "
          f"other={len(other_idx)}")

    # If the source has no Hersh-vs-clean split (e.g. childes), treat all as
    # clean and skip the curriculum/comparison.
    has_tiers = bool(hersh_idx) and bool(clean_idx) and args.source in TIER_PREFIXES
    if not has_tiers:
        clean_idx = list(range(n_total))
        hersh_idx = []

    # Held-out seeds: prefer unmasked clean conversations so generation has
    # both speaker + DT channels in the prefix.
    clean_unmasked = [
        i for i in clean_idx
        if not any("::UNK" in s for s in corpus.conversations[i].speakers)
    ]
    rng.shuffle(clean_unmasked)
    seed_idx: list[int] = clean_unmasked[: args.n_seeds]
    if not seed_idx:
        if not has_tiers:
            # Childes / fallback: just shuffle and take.
            perm = list(range(n_total))
            rng.shuffle(perm)
            seed_idx = perm[: args.n_seeds]
        else:
            raise SystemExit(
                f"no unmasked clean conversations to use as held-out seeds")
    held = set(seed_idx)
    print(f"held-out seeds (clean tier): "
          f"{[corpus.conversations[i].seq_id for i in seed_idx]}")

    # Clean train/val split (val is always carved out of clean — that's the
    # tier the user is downstream-evaluating on).
    clean_rest = [i for i in clean_idx if i not in held]
    rng.shuffle(clean_rest)
    n_val = max(1, int(args.val_frac * len(clean_rest)))
    val_idx = clean_rest[:n_val]
    train_clean_idx = clean_rest[n_val:]
    train_hersh_idx = list(hersh_idx)

    val_convs = [corpus.conversations[i] for i in val_idx]
    train_clean_convs = [corpus.conversations[i] for i in train_clean_idx]
    train_hersh_convs = [corpus.conversations[i] for i in train_hersh_idx]

    val_tok, val_dt = make_windows(val_convs, args.n_ctx)
    clean_tok, clean_dt = make_windows(train_clean_convs, args.n_ctx)
    hersh_tok, hersh_dt = make_windows(train_hersh_convs, args.n_ctx)
    mixed_tok = torch.cat([hersh_tok, clean_tok], dim=0) if hersh_tok.size(0) else clean_tok
    mixed_dt = torch.cat([hersh_dt, clean_dt], dim=0) if hersh_dt.size(0) else clean_dt

    print(f"windows — hersh: {tuple(hersh_tok.shape)}; "
          f"clean: {tuple(clean_tok.shape)}; "
          f"mixed: {tuple(mixed_tok.shape)}; "
          f"val (clean): {tuple(val_tok.shape)}")
    if val_tok.size(0) == 0:
        raise SystemExit("not enough clean windows for validation")

    base_log: list[str] = [
        f"source: {args.source}",
        f"seed: {args.seed}",
        f"V: {len(corpus.vocab)}; n_dt_buckets: {corpus.n_dt_buckets}; K: {args.n_ctx}",
        f"dt_weight: {args.dt_weight}",
        f"tier partition: hersh={len(hersh_idx)}, clean={len(clean_idx)}, "
        f"other={len(other_idx)}",
        f"windows: hersh={hersh_tok.size(0)}, clean={clean_tok.size(0)}, "
        f"val={val_tok.size(0)}",
        f"held-out seeds: {[corpus.conversations[i].seq_id for i in seed_idx]}",
    ]

    do_mixed = args.mode in ("compare", "mixed")
    do_curr = args.mode in ("compare", "curriculum") and has_tiers
    if args.mode == "curriculum" and not has_tiers:
        print(f"WARNING: source={args.source} has no Hersh tier; "
              f"falling back to mixed.")
        do_mixed = True
        do_curr = False

    results: dict[str, dict] = {}

    # === Mixed baseline ============================================
    if do_mixed:
        print("\n=== MIXED BASELINE (single-phase, hersh+clean merged) ===")
        m, dh, n_params = _new_model(corpus, args.n_ctx, args.seed)
        log = list(base_log) + [
            "", "=== mixed baseline ===",
            f"n_train_windows: {mixed_tok.size(0)} (hersh+clean merged)",
            f"epochs: {args.epochs}",
        ]
        m, dh = train(
            m, dh, mixed_tok, mixed_dt, val_tok, val_dt,
            epochs=args.epochs, lr=args.lr, bs=args.bs,
            dt_weight=args.dt_weight, log_lines=log, device=args.device,
        )
        m = m.cpu(); dh = dh.cpu()
        final = eval_final_position(m, dh, val_tok, val_dt)
        log += ["", f"val final-position coda bpt: {final['coda_bpt']:.4f}",
                f"val final-position coda acc: {final['coda_acc']:.4f}",
                f"val final-position dt   bpt: {final['dt_bpt']:.4f}",
                f"val final-position dt   acc: {final['dt_acc']:.4f}"]
        for ln in log[-4:]:
            print(ln)
        results["mixed"] = final
        suffix = "_baseline" if do_curr else ""
        out_dir = CKPT_DIR / f"{args.source}{suffix}"
        _save_checkpoint(out_dir, m, dh, corpus, args, final, seed_idx, log, n_params)
        print(f"wrote mixed checkpoint to {out_dir}/")

    # === Two-phase curriculum =====================================
    if do_curr:
        print("\n=== CURRICULUM (phase-1 hersh → phase-2 clean) ===")
        m, dh, n_params = _new_model(corpus, args.n_ctx, args.seed)
        log = list(base_log) + [
            "", "=== curriculum ===",
            f"phase-1 (hersh) windows: {hersh_tok.size(0)}",
            f"phase-2 (clean) windows: {clean_tok.size(0)}",
            f"epochs per phase: {args.epochs}",
            "", "--- phase 1: train on hersh tier ---",
        ]
        # Phase 1 also validates against clean — early stopping should still
        # care about clean-tier generalisation, not hersh val.
        m, dh = train(
            m, dh, hersh_tok, hersh_dt, val_tok, val_dt,
            epochs=args.epochs, lr=args.lr, bs=args.bs,
            dt_weight=args.dt_weight, log_lines=log, device=args.device,
        )
        log.append("")
        log.append("--- phase 2: fine-tune on clean tier ---")
        m, dh = train(
            m, dh, clean_tok, clean_dt, val_tok, val_dt,
            epochs=args.epochs, lr=args.lr, bs=args.bs,
            dt_weight=args.dt_weight, log_lines=log, device=args.device,
        )
        m = m.cpu(); dh = dh.cpu()
        final = eval_final_position(m, dh, val_tok, val_dt)
        log += ["", f"val final-position coda bpt: {final['coda_bpt']:.4f}",
                f"val final-position coda acc: {final['coda_acc']:.4f}",
                f"val final-position dt   bpt: {final['dt_bpt']:.4f}",
                f"val final-position dt   acc: {final['dt_acc']:.4f}"]
        for ln in log[-4:]:
            print(ln)
        results["curriculum"] = final
        # Curriculum is the new default checkpoint at the canonical path
        # (predict_kfold and interp_compare keep working unchanged).
        out_dir = CKPT_DIR / args.source
        _save_checkpoint(out_dir, m, dh, corpus, args, final, seed_idx, log, n_params)
        print(f"wrote curriculum checkpoint to {out_dir}/")

    # === Side-by-side comparison ==================================
    if results:
        print("\n=== held-out (clean-tier) comparison ===")
        header = f"{'mode':>11}  {'coda_bpt':>9}  {'coda_acc':>9}  {'dt_bpt':>9}  {'dt_acc':>9}"
        print(header)
        for mode in ("mixed", "curriculum"):
            r = results.get(mode)
            if r is None:
                continue
            print(f"{mode:>11}  {r['coda_bpt']:>9.4f}  {r['coda_acc']:>9.4f}  "
                  f"{r['dt_bpt']:>9.4f}  {r['dt_acc']:>9.4f}")
        if "mixed" in results and "curriculum" in results:
            d_bpt = results["curriculum"]["coda_bpt"] - results["mixed"]["coda_bpt"]
            d_acc = results["curriculum"]["coda_acc"] - results["mixed"]["coda_acc"]
            print(f"  Δ(curriculum − mixed): coda_bpt={d_bpt:+.4f}, "
                  f"coda_acc={d_acc:+.4f}  "
                  f"({'curriculum better' if d_bpt < 0 else 'mixed better'} on bpt)")


if __name__ == "__main__":
    main()
