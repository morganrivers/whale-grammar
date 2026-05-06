"""
Compare CHILDES UK English vs whale codas under the same transformer.

Architecture (locked, "M7+DT" — best-on-whale variant from
`predict_results_unified.md` plus the timestamp channel from B1 in the
ablation): MiniTransformerDT, 2 layers, 4 heads, d=64, K=8, with a
learned linear projection of (log(0.1+dt), has_timestamps) added to
each position embedding.

Both corpora are stored in the same `whale_dialogues.csv`-shaped CSV so
the same windowing and training code works unchanged. Whale TimeDeltas
come from observed acoustics; CHILDES TimeDeltas come from a punctuation
+ speaker-switch heuristic (see `childes_loader.py` docstring).

Usage:
    python -m src.grammar.predict_kfold_compare              # both, 3-fold
    python -m src.grammar.predict_kfold_compare --source childes
    python -m src.grammar.predict_kfold_compare --folds 5

Outputs:
    outputs/grammar/childes_vs_whale.json
    outputs/grammar/childes_vs_whale.md
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from src.grammar.predict_kfold import (
    CONTEXT_K, PAD, MiniTransformerDT, bits_per_token,
    build_windows_with_dt, evaluate_torch_model_dt, per_seq,
)


def per_seq_timedelta(tokens: pd.DataFrame) -> dict[str, list[float]]:
    """Per-sequence TimeDelta list (whale_dialogues column is `TimeDelta`).
    Local copy to avoid the `DeltaTime`/`TimeDelta` name mismatch in
    predict_kfold.py:per_seq_dt (only triggered by --ablation there)."""
    return {
        str(seq_id): [float(v) for v in g.sort_values("itemPosition")["TimeDelta"].tolist()]
        for seq_id, g in tokens.groupby("sequenceId")
    }

ROOT = Path(__file__).resolve().parents[2]
WHALE_CSV = ROOT / "data" / "classified" / "whale_dialogues.csv"
CHILDES_CSV = ROOT / "data" / "classified" / "childes_dialogues.csv"
OUT_JSON = ROOT / "outputs" / "grammar" / "childes_vs_whale.json"
OUT_MD = ROOT / "outputs" / "grammar" / "childes_vs_whale.md"

# Locked architecture: best-on-whale MiniTransformer (2L, 4h, d=64, K=8)
# with the DT channel added.
ARCH = dict(d=64, n_layers=2, n_heads=4)


def evaluate_baselines_simple(y_train, X_train, y_test, vocab_size):
    """Majority-unigram and Markov-1 only — light reference for the
    transformer column. Same scheme as predict_kfold.evaluate_baselines
    but without the test-set context expansion (we want bits/token only)."""
    train_counts = Counter(y_train.tolist())
    majority = train_counts.most_common(1)[0][0]
    n_train = len(y_train)
    smoothing = 0.5
    p_uniform = np.array(
        [
            (train_counts.get(t, 0) + smoothing) / (n_train + smoothing * vocab_size)
            for t in y_test
        ]
    )
    out = {"M0_majority": dict(
        accuracy=float(np.mean(y_test == majority)),
        bits_per_token=bits_per_token(p_uniform),
    )}

    trans: dict[tuple, Counter] = {}
    for i, ctx in enumerate(X_train):
        key = (ctx[-1],)
        trans.setdefault(key, Counter())[y_train[i]] += 1
    p_true = np.empty(len(y_test))
    preds = []
    for i, ctx in enumerate(np.asarray([X_train[0]]).repeat(len(y_test), 0)):
        # placeholder — we need X_test; recomputed below
        pass

    return out


def _markov1(X_train, y_train, X_test, y_test, vocab_size):
    train_counts = Counter(y_train.tolist())
    majority = train_counts.most_common(1)[0][0]
    n_train = len(y_train)
    smoothing = 0.5
    trans: dict[tuple, Counter] = {}
    for i, ctx in enumerate(X_train):
        key = (ctx[-1],)
        trans.setdefault(key, Counter())[y_train[i]] += 1
    p_true = np.empty(len(y_test))
    preds = []
    for i, ctx in enumerate(X_test):
        key = (ctx[-1],)
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


def evaluate_one_fold(seq_dict, dt_dict, train_ids, test_ids, k):
    X_train, y_train, DT_train = build_windows_with_dt(seq_dict, dt_dict, train_ids, k)
    X_test, y_test, DT_test = build_windows_with_dt(seq_dict, dt_dict, test_ids, k)
    classes = sorted(set(y_train.tolist()) | set(y_test.tolist()) | {PAD})
    V = len(classes)

    # Majority + Markov-1 (cheap reference baselines)
    train_counts = Counter(y_train.tolist())
    majority = train_counts.most_common(1)[0][0]
    n_train = len(y_train)
    smoothing = 0.5
    p_uniform = np.array(
        [
            (train_counts.get(t, 0) + smoothing) / (n_train + smoothing * V)
            for t in y_test
        ]
    )
    res = {
        "M0_majority": dict(
            accuracy=float(np.mean(y_test == majority)),
            bits_per_token=bits_per_token(p_uniform),
        ),
        "M1_markov1": _markov1(X_train, y_train, X_test, y_test, V),
    }

    # MiniTransformer + DT — the headline model.
    res["M7_MiniTfmDT"] = evaluate_torch_model_dt(
        "MiniTfmDT",
        lambda V_: MiniTransformerDT(V_, k, **ARCH),
        X_train, DT_train, y_train,
        X_test, DT_test, y_test,
        V, classes,
    )

    res["_meta"] = dict(
        n_train=int(len(X_train)), n_test=int(len(X_test)), V=V,
    )
    return res


def run_source(name: str, csv_path: Path, n_folds: int, k: int) -> dict:
    print(f"\n=== {name} ({csv_path.name}) ===")
    if name == "whale":
        # Compound whale token = (rhythm, tempo_bin, orn, rubato).
        # The compound id replaces `Coda` so the windowing code is unchanged.
        from src.grammar.whale_compound import load_compound_whale
        tokens, _decoder = load_compound_whale(csv_path)
        print(f"  compound whale vocab V={tokens['Coda'].nunique()} "
              f"(rhythm × tempo_bin × orn × rubato, attested only)")
    else:
        tokens = pd.read_csv(csv_path)
    seq_dict = per_seq(tokens, "Coda")
    dt_dict = per_seq_timedelta(tokens)
    seq_ids = sorted(seq_dict.keys())
    print(f"  {len(seq_ids)} sequences, {sum(len(v) for v in seq_dict.values()):,} tokens, "
          f"V≈{tokens['Coda'].nunique()}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    for fi, (tr, te) in enumerate(kf.split(seq_ids)):
        train_ids = [seq_ids[i] for i in tr]
        test_ids = [seq_ids[i] for i in te]
        t0 = time.time()
        res = evaluate_one_fold(seq_dict, dt_dict, train_ids, test_ids, k)
        res["_meta"]["fold"] = fi
        res["_meta"]["seconds"] = round(time.time() - t0, 1)
        fold_results.append(res)
        line = f"  fold {fi}: " + " | ".join(
            f"{m}: bpt={res[m]['bits_per_token']:.3f} acc={res[m]['accuracy']:.3f}"
            for m in ("M0_majority", "M1_markov1", "M7_MiniTfmDT")
        ) + f"  ({res['_meta']['seconds']}s)"
        print(line)

    summary = {}
    model_keys = ["M0_majority", "M1_markov1", "M7_MiniTfmDT"]
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

    return dict(
        source=name,
        csv=str(csv_path),
        n_sequences=len(seq_ids),
        n_tokens=int(sum(len(v) for v in seq_dict.values())),
        V=int(fold_results[0]["_meta"]["V"]),
        k=k,
        n_folds=n_folds,
        folds=fold_results,
        summary=summary,
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=("whale", "childes", "both"), default="both")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--k", type=int, default=CONTEXT_K)
    args = p.parse_args(argv)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    if args.source in ("whale", "both"):
        runs.append(run_source("whale", WHALE_CSV, args.folds, args.k))
    if args.source in ("childes", "both"):
        runs.append(run_source("childes_uk", CHILDES_CSV, args.folds, args.k))

    OUT_JSON.write_text(json.dumps(dict(arch=ARCH, runs=runs), indent=2, default=str))

    # Markdown side-by-side.
    L: list[str] = []
    L.append("# CHILDES Eng-UK vs whale — same MiniTransformer-DT, 3-fold CV")
    L.append("")
    L.append(
        f"Architecture (locked from best-on-whale): MiniTransformer with DT — "
        f"{ARCH['n_layers']}L, {ARCH['n_heads']}h, d={ARCH['d']}, K={args.k}, "
        f"+`(log(0.1+dt), has_timestamps)` projection. "
        f"Sequence-level KFold ({args.folds} folds). "
        f"Metric = held-out cross-entropy in **bits/token** (log₂); lower is better."
    )
    L.append("")
    L.append(
        "Whale TimeDeltas come from observed acoustics; CHILDES TimeDeltas are "
        "estimated: 0.3 s intra-utterance, 0.5 s after `,`, 1.0 s after `.`/`?`/`!`, "
        "2.0 s on speaker switch. CHILDES vocab uses `%mor` lemmas (so "
        "`going`/`went`/`gone`→`go`)."
    )
    L.append("")
    for run in runs:
        L.append(f"## {run['source']} (V={run['V']}, "
                 f"{run['n_sequences']} sequences, {run['n_tokens']:,} tokens)")
        L.append("")
        L.append("| model | params | bits/token (↓) | perplexity (↓) | accuracy |")
        L.append("|-------|-------:|---------------:|---------------:|---------:|")
        labels = {
            "M0_majority": "majority (smoothed unigram)",
            "M1_markov1": "Markov-1",
            "M7_MiniTfmDT": "MiniTransformer-DT (2L, 4h, d=64, K=8 + DT)",
        }
        for m in ("M0_majority", "M1_markov1", "M7_MiniTfmDT"):
            s = run["summary"][m]
            params = f"{s['n_params']:,}" if s.get("n_params") else "—"
            L.append(
                f"| {labels[m]} | {params} | "
                f"{s['bits_per_token_mean']:.3f} ± {s['bits_per_token_std']:.3f} | "
                f"{s['perplexity_mean']:.2f} | "
                f"{s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f} |"
            )
        L.append("")

    if len(runs) == 2:
        w = runs[0]["summary"]["M7_MiniTfmDT"]
        c = runs[1]["summary"]["M7_MiniTfmDT"]
        wm = runs[0]["summary"]["M0_majority"]
        cm = runs[1]["summary"]["M0_majority"]
        L.append("## Side-by-side compression")
        L.append("")
        L.append("| source | V | majority bpt | MiniTfm-DT bpt | savings (bits) | "
                 "fraction of unigram entropy |")
        L.append("|---|---:|---:|---:|---:|---:|")
        L.append(
            f"| whale | {runs[0]['V']} | {wm['bits_per_token_mean']:.3f} | "
            f"{w['bits_per_token_mean']:.3f} | "
            f"{wm['bits_per_token_mean'] - w['bits_per_token_mean']:.3f} | "
            f"{w['bits_per_token_mean'] / wm['bits_per_token_mean']:.3f} |"
        )
        L.append(
            f"| CHILDES UK | {runs[1]['V']} | {cm['bits_per_token_mean']:.3f} | "
            f"{c['bits_per_token_mean']:.3f} | "
            f"{cm['bits_per_token_mean'] - c['bits_per_token_mean']:.3f} | "
            f"{c['bits_per_token_mean'] / cm['bits_per_token_mean']:.3f} |"
        )
        L.append("")
        L.append(
            "Lower *fraction of unigram entropy* = the model captures more "
            "structure relative to the entropy floor. Direct bits/token aren't "
            "comparable across V (CHILDES has ~12× larger vocabulary)."
        )
        L.append("")

    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
