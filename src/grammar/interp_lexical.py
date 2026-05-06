"""
Lexical-level interp views on top of the M7 hooked checkpoints.

Three views per source, written to
``outputs/grammar/interp/<source>/lexical.md`` (plus a few PNGs):

1. **Embedding neighborhoods.** Top-k cosine neighbors of the most
   frequent tokens in ``W_E`` ("treats-as-interchangeable as context")
   and in ``W_U`` ("confuses as predictions"). Synonym / POS probe.

2. **Attention co-occurrence.** Per (layer, head), aggregate attention
   over a corpus-wide pool of K-windows into a directed (V, V) matrix
   ``A`` where ``A[a, b]`` is the mean attention from query positions
   with token ``a`` onto key positions with token ``b``. Rank mutual
   pairs by ``sqrt(A[a,b] * A[b,a])`` to surface tokens that
   "constantly attend to each other".

3. **Bigram-baseline divergence.** For each token, compare the
   "skip-everything" prediction (apply the unembedding to ``W_E[t] +
   W_pos[K-1]`` after ln_final) with the full model's average
   next-token distribution conditional on the same previous token.
   KL(full || bigram) in bits localizes which tokens the model uses
   contextually vs as a fixed lookup.

Usage::

    python -m src.grammar.interp_lexical
    python -m src.grammar.interp_lexical --sources childes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.grammar.m7_hooked import N_CTX, make_m7_hooked

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "outputs" / "grammar" / "checkpoints"
OUT_DIR = ROOT / "outputs" / "grammar" / "interp"
DATA_DIR = ROOT / "data" / "classified"

SOURCES = ("whale", "childes")
TOP_FREQ = 30          # how many frequent tokens we report neighbors for
NEIGHBOR_K = 5         # neighbors per token
COOC_WINDOWS = 4000    # subsampled K-windows for the co-occurrence pool
COOC_MIN_QCOUNT = 20   # require at least this many query observations
DIV_MIN_COUNT = 10     # require this many last-position observations
DIV_TOP = 15           # show top/bottom this many tokens


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_checkpoint(source: str):
    src_dir = CKPT_DIR / source
    cfg = json.loads((src_dir / "config.json").read_text())
    vocab = json.loads((src_dir / "vocab.json").read_text())
    model = make_m7_hooked(d_vocab=cfg["d_vocab"])
    state = torch.load(src_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, cfg, vocab


def load_corpus_tokens(source: str) -> list[list[int]]:
    csv = DATA_DIR / f"{source}_dialogues.csv"
    df = pd.read_csv(csv, usecols=["sequenceId", "itemPosition", "Coda"])
    df = df.sort_values(["sequenceId", "itemPosition"])
    out: list[list[int]] = []
    for _sid, g in df.groupby("sequenceId", sort=False):
        toks = g["Coda"].astype(int).tolist()
        if len(toks) >= N_CTX + 1:
            out.append(toks)
    return out


def windows_from_convs(convs: list[list[int]], k: int = N_CTX) -> np.ndarray:
    rows: list[list[int]] = []
    for toks in convs:
        for i in range(len(toks) - k + 1):
            rows.append(toks[i : i + k])
    return np.array(rows, dtype=np.int64)


def token_frequencies(convs: list[list[int]], V: int) -> np.ndarray:
    counts = np.zeros(V, dtype=np.int64)
    for toks in convs:
        for t in toks:
            counts[t] += 1
    return counts


def display(vocab: list[str], t: int) -> str:
    if 0 <= t < len(vocab):
        return str(vocab[t])
    return f"<{t}>"


# ---------------------------------------------------------------------------
# View 1 — embedding neighborhoods
# ---------------------------------------------------------------------------


def cosine_neighbors(
    W: np.ndarray,
    query_ids: list[int],
    top_k: int,
    restrict_ids: np.ndarray | None = None,
) -> dict[int, list[tuple[int, float]]]:
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
    Wn = W / norms
    out: dict[int, list[tuple[int, float]]] = {}
    if restrict_ids is not None:
        mask = np.zeros(W.shape[0], dtype=bool)
        mask[restrict_ids] = True
    for q in query_ids:
        sim = Wn @ Wn[q]
        sim_q = sim.copy()
        sim_q[q] = -np.inf
        if restrict_ids is not None:
            sim_q[~mask] = -np.inf
        idx = np.argpartition(-sim_q, top_k)[:top_k]
        idx = idx[np.argsort(-sim_q[idx])]
        out[q] = [(int(i), float(sim_q[i])) for i in idx]
    return out


# ---------------------------------------------------------------------------
# View 2 — attention co-occurrence
# ---------------------------------------------------------------------------


@torch.no_grad()
def attn_cooccurrence(
    model, eval_windows: np.ndarray, V: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    K = eval_windows.shape[1]
    A = np.zeros((n_layers, n_heads, V, V), dtype=np.float32)
    q_count = np.zeros(V, dtype=np.int64)

    bs = 256
    for start in range(0, len(eval_windows), bs):
        batch_np = eval_windows[start : start + bs]
        batch = torch.from_numpy(batch_np).long()
        _, cache = model.run_with_cache(batch)
        B = batch.size(0)
        # per-batch query-token counts (shared across layers/heads)
        for q in range(K):
            np.add.at(q_count, batch_np[:, q], 1)
        for L in range(n_layers):
            patt = cache[f"blocks.{L}.attn.hook_pattern"].detach().numpy()
            # patt: (B, H, q, k); accumulate into A[L, h, qt, kt]
            for b in range(B):
                tok = batch_np[b]
                # pre-build (q, k) → (qt, kt) once per row
                qt = tok[:, None]                 # (K, 1)
                kt = tok[None, :]                 # (1, K)
                for h in range(n_heads):
                    np.add.at(A[L, h], (qt.repeat(K, axis=1), kt.repeat(K, axis=0)),
                              patt[b, h])
    return A, q_count


def normalize_cooc(A: np.ndarray, q_count: np.ndarray) -> np.ndarray:
    """Divide A[L,h,a,:] by q_count[a] to get mean attention given query=a."""
    safe = q_count.astype(np.float64).copy()
    safe[safe == 0] = 1.0
    return A / safe[None, None, :, None]


def top_mutual_pairs(
    meanA_lh: np.ndarray, q_count: np.ndarray,
    min_qcount: int, top_n: int,
) -> list[tuple[int, int, float, float, float]]:
    """For one (L, h), return up to top_n mutual pairs (a, b) with a < b
    sorted by sqrt(mean(a→b) * mean(b→a))."""
    V = meanA_lh.shape[0]
    keep = q_count >= min_qcount
    if not keep.any():
        return []
    ids = np.where(keep)[0]
    sub = meanA_lh[np.ix_(ids, ids)]                      # (n, n)
    mutual = np.sqrt(np.maximum(sub * sub.T, 0.0))        # symmetric
    # zero out diagonal and lower triangle to dedupe (a,b)/(b,a)
    n = sub.shape[0]
    iu = np.triu_indices(n, k=1)
    flat = mutual[iu]
    if flat.size == 0:
        return []
    take = min(top_n, flat.size)
    top_idx = np.argpartition(-flat, take - 1)[:take]
    top_idx = top_idx[np.argsort(-flat[top_idx])]
    out = []
    for fi in top_idx:
        ai, bi = iu[0][fi], iu[1][fi]
        a, b = int(ids[ai]), int(ids[bi])
        out.append((a, b, float(sub[ai, bi]), float(sub[bi, ai]), float(flat[fi])))
    return out


# ---------------------------------------------------------------------------
# View 3 — bigram-baseline divergence
# ---------------------------------------------------------------------------


@torch.no_grad()
def bigram_divergence(
    model, eval_windows: np.ndarray, V: int, min_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (kl_bits[V], counts[V], bigram_probs[V,V], full_avg[V,V])
    where rows are conditioned on the *last* token of the window."""
    K = eval_windows.shape[1]

    # ---- bigram baseline: ignore everything but a single token at pos K-1
    W_E = model.W_E.detach()                    # (V, d)
    W_pos_last = model.W_pos.detach()[K - 1]    # (d,)
    h = W_E + W_pos_last[None, :]               # (V, d)
    h = model.ln_final(h)
    bigram_logits = model.unembed(h)            # (V, V)
    bigram_probs = bigram_logits.softmax(dim=-1).cpu().numpy().astype(np.float64)

    # ---- full-model averages conditional on last-position token
    full_sum = np.zeros((V, V), dtype=np.float64)
    counts = np.zeros(V, dtype=np.int64)
    bs = 256
    for start in range(0, len(eval_windows), bs):
        batch = torch.from_numpy(eval_windows[start : start + bs]).long()
        logits = model(batch)
        probs = logits[:, -1].softmax(-1).cpu().numpy().astype(np.float64)
        last = eval_windows[start : start + bs, -1]
        for b, lt in enumerate(last):
            full_sum[lt] += probs[b]
            counts[lt] += 1

    kl = np.full(V, np.nan, dtype=np.float64)
    for t in range(V):
        if counts[t] >= min_count:
            full_avg = full_sum[t] / counts[t]
            p = full_avg + 1e-12
            q = bigram_probs[t] + 1e-12
            kl[t] = float(np.sum(p * np.log2(p / q)))
    full_avg_mat = full_sum / np.maximum(counts, 1)[:, None]
    return kl, counts, bigram_probs, full_avg_mat


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_cooc_heatmap(
    meanA: np.ndarray, ids: list[int], vocab: list[str], out_path: Path,
    title: str,
):
    """Plot a (layers, heads) grid of token×token attention heatmaps for the
    given top-N tokens (rows = query token, cols = key token)."""
    n_layers, n_heads = meanA.shape[0], meanA.shape[1]
    sub = meanA[:, :, np.ix_(ids, ids)[0], np.ix_(ids, ids)[1]]
    labels = [display(vocab, t) for t in ids]
    fig, axes = plt.subplots(
        n_layers, n_heads, figsize=(3 * n_heads, 3 * n_layers),
        squeeze=False,
    )
    vmax = float(sub.max())
    for L in range(n_layers):
        for h in range(n_heads):
            ax = axes[L][h]
            im = ax.imshow(sub[L, h], cmap="viridis", vmin=0, vmax=vmax,
                           aspect="auto")
            ax.set_title(f"L{L}H{h}", fontsize=9)
            ax.set_xticks(range(len(ids)))
            ax.set_yticks(range(len(ids)))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5,
                 label="mean attn (q-token row → k-token col)")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_kl_hist(kl: np.ndarray, counts: np.ndarray, min_count: int,
                 out_path: Path, title: str):
    valid = (counts >= min_count) & np.isfinite(kl)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if valid.any():
        ax.hist(kl[valid], bins=30, color="C0", alpha=0.8)
    ax.set_xlabel("KL(full || bigram)  bits")
    ax.set_ylabel("number of tokens")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def md_table_neighbors(
    title: str, neighbors: dict[int, list[tuple[int, float]]],
    vocab: list[str], counts: np.ndarray,
) -> list[str]:
    md = [f"### {title}", ""]
    md.append("| token | count | "
              + " | ".join(f"n{i+1}" for i in range(NEIGHBOR_K)) + " |")
    md.append("|---|---:|" + "|".join(["---"] * NEIGHBOR_K) + "|")
    for q, neigh in neighbors.items():
        cells = [
            f"`{display(vocab, q)}`",
            str(int(counts[q])),
        ]
        for nid, sim in neigh:
            cells.append(f"`{display(vocab, nid)}` ({sim:.2f})")
        while len(cells) < 2 + NEIGHBOR_K:
            cells.append("")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    return md


def md_table_mutual(
    pairs_per_head: dict[tuple[int, int], list[tuple[int, int, float, float, float]]],
    vocab: list[str], top_n: int,
) -> list[str]:
    md = ["### Top mutual attention pairs per head", ""]
    md.append("| L | H | a | b | a→b | b→a | mutual |")
    md.append("|---|---|---|---|---:|---:|---:|")
    for (L, h), pairs in pairs_per_head.items():
        for a, b, ab, ba, mut in pairs[:top_n]:
            md.append(f"| {L} | {h} | `{display(vocab, a)}` | "
                      f"`{display(vocab, b)}` | {ab:.3f} | {ba:.3f} | {mut:.3f} |")
    md.append("")
    return md


def md_divergence(
    kl: np.ndarray, counts: np.ndarray, bigram_probs: np.ndarray,
    full_avg: np.ndarray, vocab: list[str], min_count: int, top_n: int,
) -> list[str]:
    valid = np.where((counts >= min_count) & np.isfinite(kl))[0]
    if valid.size == 0:
        return [f"_no tokens with ≥{min_count} last-position observations_", ""]
    order = valid[np.argsort(-kl[valid])]
    most = order[:top_n]
    least = order[-top_n:][::-1]

    def row(t: int) -> str:
        bg = int(np.argmax(bigram_probs[t]))
        full = int(np.argmax(full_avg[t]))
        return (f"| `{display(vocab, t)}` | {int(counts[t])} | "
                f"{kl[t]:.2f} | `{display(vocab, bg)}` | "
                f"`{display(vocab, full)}` |")

    md = ["Higher KL = the model uses this token contextually rather than as "
          "a static lookup. KL is between full-model averaged next-token "
          "distribution and the model's own no-context baseline "
          "(`unembed(ln_final(W_E[t] + W_pos[K-1]))`).", ""]
    md.append(f"**Most contextual (top {top_n})**")
    md.append("| token | count | KL bits | top bigram pred | top full pred |")
    md.append("|---|---:|---:|---|---|")
    md.extend(row(int(t)) for t in most)
    md.append("")
    md.append(f"**Most lookup-like (bottom {top_n})**")
    md.append("| token | count | KL bits | top bigram pred | top full pred |")
    md.append("|---|---:|---:|---|---|")
    md.extend(row(int(t)) for t in least)
    md.append("")
    return md


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_one_source(source: str, args) -> None:
    print(f"\n=== {source.upper()} ===")
    model, cfg, vocab = load_checkpoint(source)
    V = model.cfg.d_vocab
    out_dir = OUT_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)

    convs = load_corpus_tokens(source)
    counts = token_frequencies(convs, V)
    print(f"corpus: {len(convs)} conversations, V={V}, "
          f"sum(counts)={int(counts.sum())}")

    # Top frequent tokens
    top_ids = list(np.argsort(-counts)[:TOP_FREQ])
    top_ids = [int(t) for t in top_ids]

    # ---- View 1: embedding neighborhoods ----
    print("computing embedding neighbors...")
    W_E = model.W_E.detach().numpy()
    W_U = model.W_U.detach().numpy().T  # (V, d) for cosine over output rows
    # Restrict neighbor pool to tokens we've seen at least a handful of times,
    # otherwise random embeddings dominate the top-k for rare-word queries.
    pool = np.where(counts >= max(5, COOC_MIN_QCOUNT // 4))[0]
    we_neigh = cosine_neighbors(W_E, top_ids, NEIGHBOR_K, restrict_ids=pool)
    wu_neigh = cosine_neighbors(W_U, top_ids, NEIGHBOR_K, restrict_ids=pool)

    # ---- View 2: attention co-occurrence ----
    print("computing attention co-occurrence...")
    all_windows = windows_from_convs(convs, N_CTX)
    rng = np.random.default_rng(args.seed)
    if len(all_windows) > COOC_WINDOWS:
        idx = rng.choice(len(all_windows), COOC_WINDOWS, replace=False)
        cooc_windows = all_windows[idx]
    else:
        cooc_windows = all_windows
    print(f"  pool: {len(cooc_windows)} windows")
    A, q_count = attn_cooccurrence(model, cooc_windows, V)
    meanA = normalize_cooc(A, q_count)

    pairs_per_head: dict[tuple[int, int], list] = {}
    for L in range(model.cfg.n_layers):
        for h in range(model.cfg.n_heads):
            pairs_per_head[(L, h)] = top_mutual_pairs(
                meanA[L, h], q_count,
                min_qcount=COOC_MIN_QCOUNT, top_n=10,
            )

    plot_cooc_heatmap(
        meanA, top_ids, vocab,
        out_dir / "attn_cooccurrence.png",
        title=f"Attention co-occurrence (top {TOP_FREQ} tokens) — {source}",
    )

    # ---- View 3: bigram-baseline divergence ----
    print("computing bigram-baseline divergence...")
    # Use a separate held-out-ish pool: same windows, fine for a corpus-wide
    # average since we're not optimising on this signal.
    kl, last_counts, bigram_probs, full_avg = bigram_divergence(
        model, cooc_windows, V, min_count=DIV_MIN_COUNT,
    )
    plot_kl_hist(kl, last_counts, DIV_MIN_COUNT,
                 out_dir / "bigram_kl_hist.png",
                 title=f"KL(full || bigram) per token — {source}")

    # ---- Write report ----
    md = [f"# Lexical interp — {source}", ""]
    md.append(f"Source: `{source}`. V={V}, N_CTX={N_CTX}, "
              f"corpus tokens={int(counts.sum())}.")
    md.append("")
    md.append("## Most frequent tokens")
    md.append("")
    md.append("| rank | id | token | count |")
    md.append("|---:|---:|---|---:|")
    for r, t in enumerate(top_ids, 1):
        md.append(f"| {r} | {t} | `{display(vocab, t)}` | {int(counts[t])} |")
    md.append("")

    md.append("## View 1 — Embedding neighborhoods")
    md.append("")
    md.append("Cosine top-5 in the model's input embedding (W_E) and output "
              "embedding (W_U). Neighbor pool restricted to tokens with "
              f"count ≥ {max(5, COOC_MIN_QCOUNT // 4)} so rare-word noise "
              "doesn't dominate.")
    md.append("")
    md.extend(md_table_neighbors("W_E neighbors (input role)",
                                 we_neigh, vocab, counts))
    md.extend(md_table_neighbors("W_U neighbors (output role)",
                                 wu_neigh, vocab, counts))

    md.append("## View 2 — Attention co-occurrence")
    md.append("")
    md.append("For each (L, h), `A[a, b]` is the mean attention from query "
              "positions with token `a` to key positions with token `b`, "
              f"averaged over {len(cooc_windows)} K={N_CTX} windows. Mutual "
              "score = √(A[a,b] · A[b,a]); only pairs with both tokens "
              f"observed ≥ {COOC_MIN_QCOUNT} times included. See "
              f"`attn_cooccurrence.png` for top-{TOP_FREQ} heatmaps.")
    md.append("")
    md.extend(md_table_mutual(pairs_per_head, vocab, top_n=10))

    md.append("## View 3 — Bigram-baseline divergence")
    md.append("")
    md.extend(md_divergence(kl, last_counts, bigram_probs, full_avg,
                            vocab, DIV_MIN_COUNT, DIV_TOP))

    md.append("## Files")
    md.append("")
    md.append("- `attn_cooccurrence.png` — (layer × head) heatmaps over the "
              f"top-{TOP_FREQ} tokens.")
    md.append("- `bigram_kl_hist.png` — distribution of per-token KL "
              "(full || bigram) in bits.")
    md.append("")

    (out_dir / "lexical.md").write_text("\n".join(md))
    print(f"wrote {out_dir / 'lexical.md'}")

    summary = dict(
        source=source,
        V=V,
        n_corpus_tokens=int(counts.sum()),
        n_cooc_windows=int(len(cooc_windows)),
        kl_summary=dict(
            n_eligible=int(np.sum((last_counts >= DIV_MIN_COUNT)
                                  & np.isfinite(kl))),
            mean=float(np.nanmean(kl)) if np.any(np.isfinite(kl)) else None,
            median=float(np.nanmedian(kl)) if np.any(np.isfinite(kl)) else None,
            max=float(np.nanmax(kl)) if np.any(np.isfinite(kl)) else None,
        ),
    )
    (out_dir / "lexical_summary.json").write_text(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sources", nargs="+", default=list(SOURCES))
    args = p.parse_args(argv)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src in args.sources:
        run_one_source(src, args)


if __name__ == "__main__":
    main()
