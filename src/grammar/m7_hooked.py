"""
TransformerLens port of the M7 mini-transformer.

The benchmark winner from `predict_kfold.py` was M7: a 2-layer / 4-head /
d_model=64 / K=8 transformer that hits 3.186 bpt (≈9.10 perplexity) on
the unified whale corpus.

For interpretability work we want the same architecture exposed through
TransformerLens' HookedTransformer so we get attention/head hooks,
activation caching, and induction-head detectors out of the box. Two
substantive differences from the original `nn.TransformerEncoderLayer`
build:

1. Causal mask. The benchmark M7 had no mask but only ever read the
   final-position output, so it was *behaviorally* a next-token model.
   For autoregressive generation we need an honest causal mask.
2. Loss on every position. Training as a full causal LM (predict t+1
   from t for every t) is strictly more samples per window than the
   original "loss only on position K" recipe and matches what TLens
   expects in `model(tokens, return_type="loss")`.

Otherwise the params line up: 2 layers, 4 heads, d_model=64, d_head=16,
d_mlp=256, n_ctx=8, learned absolute position embeddings, GELU MLP,
post-LN.
"""
from __future__ import annotations

from transformer_lens import HookedTransformer, HookedTransformerConfig

# ---- M7 hyperparameters (matches predict_kfold.py:312 MiniTransformer) ----
N_LAYERS = 2
N_HEADS = 4
D_MODEL = 64
D_HEAD = D_MODEL // N_HEADS  # 16
D_MLP = 4 * D_MODEL          # 256
N_CTX = 8                    # context K from predict_kfold.CONTEXT_K


def make_m7_config(d_vocab: int, n_ctx: int = N_CTX) -> HookedTransformerConfig:
    return HookedTransformerConfig(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_model=D_MODEL,
        d_head=D_HEAD,
        d_mlp=D_MLP,
        n_ctx=n_ctx,
        d_vocab=d_vocab,
        d_vocab_out=d_vocab,
        act_fn="gelu",
        attn_only=False,
        normalization_type="LN",
        positional_embedding_type="standard",  # learned absolute, like M7
        attention_dir="causal",
        tokenizer_name=None,
        device="cpu",
        seed=0,
    )


def make_m7_hooked(d_vocab: int, n_ctx: int = N_CTX) -> HookedTransformer:
    cfg = make_m7_config(d_vocab, n_ctx=n_ctx)
    return HookedTransformer(cfg)
