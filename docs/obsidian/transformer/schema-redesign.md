---
tags:
  - transformer
  - experiment
summary: R0 vs R1 vs R2 — multi-channel multi-task transformer experiment, and why R2 lost to R0
created: 2026-05-06
updated: 2026-05-06
---

# Schema redesign — R0 / R1 / R2

The redesign experiment asks whether a transformer that gets the *full* row schema (Whale, Coda, Ornamentation, Synchrony, Duration, TimeDelta_log, has_timestamps) as parallel embedding channels — and is supervised on multiple of those channels with multi-task heads — beats the single-channel `Coda + DT` control on next-coda log-loss.

Result: **no, R2 is +0.117 bpt worse than R0.** R1 (multi-channel inputs but Coda CE only) is between them.

The redesign produced a useful corpus reshape and a reusable schema for analysis (and `whale_id_index.csv`), but the multi-channel multi-task transformer doesn't earn its complexity on next-coda prediction.

## Variants

All three at architecture **3L, 8h, d=32, K=25** (whale-gpt-main shape), **fast training mode** (AdamW + lr=1e-3 + bs=128 + 80 epochs + ES patience 10).

| variant | inputs | loss |
|---|---|---|
| **R0 control** (47,588 params) | single channel: `Coda` token + `(log_dt, has_timestamps)` projection | CE on Coda only |
| **R1 multi-channel single-task** (50,496 params) | per-column embeddings concat to d=32: `Whale`(6) + `Coda`(8) + `Orn`(3) + `Sync`(3) + `Duration`(4) + `TimeDelta_log`(4) + `has_timestamps`(4) | CE on Coda only |
| **R2 multi-channel multi-task** (50,496 params) | same inputs as R1 | CE on Whale + CE on Coda + 0.5·CE on Orn + 0.5·CE on Sync + 0.2·L1 on Duration + 0.2·L1 on TimeDelta_log |

DT encoding: `log(0.1 + max(dt, 0))` for present rows; `log(0.1)` (≈ −2.3) sentinel for missing rows; the `has_timestamps` channel separately tells the model the value is uninformative on Hersh.

PAD trick for the multi-channel embeddings: each integer channel reserves index 0 for PAD; real ids are shifted by +1 before embedding lookup.

## Headline numbers (5-fold CV, V=131 incl. PAD, K=25)

| # | model | bpt | perplexity | accuracy |
|---|---|---:|---:|---:|
| R0 | R0 control: single-channel Coda + DT | **3.215 ± 0.238** | 9.29 | 0.458 ± 0.058 |
| R1 | R1: multi-channel inputs, Coda CE only | 3.244 ± 0.225 | 9.47 | 0.448 ± 0.060 |
| R2 | R2 redesign: multi-channel + multi-task | 3.319 ± 0.222 | 9.98 | 0.440 ± 0.050 |

R2 trails R0 on 4/5 folds. R1 trails R0 on 4/5 folds.

Reproduce:

```bash
python -m src.grammar.predict_kfold --redesign --variants R0,R1,R2
```

Output files:

- `outputs/grammar/predict_results_redesign.md` — most recent run (currently R0+R1).
- `outputs/grammar/predict_results_redesign_R0R2.md` — archived R0+R2 run.

## Why R0 won — the bear case

Hersh dominates the corpus and is structurally NA on every channel R1/R2 add over R0:

| feature | hersh2022_pacific (24,237) | sharma2024_dswp (8,872) | sharma2025_birth (5,731) |
|---|---:|---:|---:|
| `Whale != ::UNK` | 0 % | 62 % | 100 % |
| `Synchrony == 1` | 0 % | 9 % | 26 % |
| `Ornamentation == 1` | 0 % | 3 % | 11 % |
| `has_timestamps = 1` | 0 % | 42 % | 100 % |

So:

- `Whale = ::UNK` for 62 % of rows + ~3 % of DSWP NA → the UNK embedding is the "default Whale" the model sees most of the time.
- `Synchrony = 0` for ~92 % of rows → the Sync embedding is mostly the "no chorus" default.
- `Ornamentation = 0` for ~98 % of rows → similar.
- `has_timestamps = 0` for 62 % of rows.

The genuinely-informative channels (Whale on the 38 % of rows where it's known; Sync on the ~8 % where it fires) are a small fraction of training windows. The multi-channel embeddings spend most of their capacity on majority-class entries, leaving the marginal information below the σ ≈ 0.24 bpt fold-variance floor.

## R1 → R2 isolates the multi-task hypothesis

R1 keeps the multi-channel inputs but removes the auxiliary heads. R1 vs R0 directly tests *do the extra input channels help on next-coda prediction*. R2 vs R1 tests *does multi-task supervision add anything on top*.

Reading the deltas (vs R0):

- R0 → R1 = +0.029 bpt. Multi-channel inputs alone hurt slightly. Marginal at best — within fold variance.
- R1 → R2 = +0.075 bpt. Multi-task heads make it worse. The aux losses (Whale CE weight 1.0, Orn/Sync CE weight 0.5, Duration/TimeDelta L1 weight 0.2) are pulling the encoder toward representations that don't help the Coda head.

So the user's hypothesis ("multi-task aux heads were the active poison, not multi-channel inputs") is *partially* supported: R1 → R2 is a clean negative. R0 → R1 is too small to call.

## Per-fold deltas

R0 vs R1 (`predict_results_redesign.md`, R0+R1 run):

| fold | R0 | R1 | Δ |
|---:|---:|---:|---:|
| 0 | 2.803 | 2.833 | +0.030 |
| 1 | 3.341 | 3.363 | +0.022 |
| 2 | 3.362 | 3.358 | −0.004 |
| 3 | 3.465 | 3.477 | +0.012 |
| 4 | 3.105 | 3.189 | +0.084 |

R0 vs R2 (`predict_results_redesign_R0R2.md`, archived):

| fold | R0 | R2 | Δ |
|---:|---:|---:|---:|
| 0 | 2.796 | 2.893 | +0.097 |
| 1 | 3.341 | 3.391 | +0.050 |
| 2 | 3.333 | 3.349 | +0.016 |
| 3 | 3.432 | 3.430 | −0.002 |
| 4 | 3.110 | 3.534 | +0.424 |

Fold 4 is unusually hard for R2 (+0.42) and modestly hard for R1 (+0.08). Fold 4 is the same fold that produces the highest R0 number — the "hardest" recordings in the split — so the multi-channel/multi-task models fail when generalization is hardest, exactly the opposite of what added structure is supposed to buy.

## What carries forward from the redesign

Even though R2 lost the bits-per-token race, the redesign produced things worth keeping:

1. **The 9-column schema is the production schema.** [[transformer/corpus-design]] documents it; `tests/test_render_csv.py` enforces it. R0 reads only `Coda` + `TimeDelta` + `has_timestamps`, but the rest of the pipeline (interp, continuations, lexical analysis) reads the full schema.
2. **`whale_id_index.csv`** — stable speaker mapping, used by `interp_whale_decoded.py` to split decoded mutual-attention pairs into within-vs-cross-speaker subsets.
3. **DT encoding `(log(0.1 + dt), has_timestamps)`** — adopted as the default for all DT-aware models including the persisted M7-DT checkpoint.
4. **The "Hersh-dominates-the-corpus" framing** — a strong null result that informs every later experiment.

## What's open

- **K-sweep on R0.** We never tested K ∈ {4, 8, 16, 24, 32} cleanly on the winner. Probably ~0.02 bpt of fold-variance noise either way; not the priority.
- **A `--slow` rerun of R0+R2.** AdamW + CosineAnnealingLR + bs=1000 + 5000 epochs + dropout 0.2, no early stopping. ~15 hours. Probably wouldn't change the ordering — corpus is small enough that fast-mode ES at epoch 30–50 already finds the loss floor — but it would close out the "training-recipe was the bottleneck" alternative.
- **Different multi-task weights.** R2 weights `(coda=1, whale=1, orn/sync=0.5, dur/td=0.2)`. If R1 really is roughly tied with R0, then re-weighting R2's aux heads might recover the gap; not pursued.
- **Per-source bpt breakdown.** Aggregate hides whether R0 wins on DSWP/birth and ties on Hersh, or wins uniformly. Useful for the "is Hersh genuinely a different distribution" question.

## Related

- [[transformer/transformer]] — model menu.
- [[transformer/results]] — full bpt tables.
- [[transformer/corpus-design]] — the schema R0/R1/R2 read from.
