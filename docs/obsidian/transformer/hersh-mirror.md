---
tags:
  - transformer
  - childes
  - experiment
summary: Mirror Hersh's 62% structural NA pattern onto CHILDES — same Coda compressibility, masked-loss-ready DT/speaker channels
created: 2026-05-06
updated: 2026-05-06
---

# Hersh-mirror experiment — synthetic missingness on CHILDES

[[overview/corpora|Whale corpora]] split into three sources with very different missingness patterns:

| feature | hersh2022_pacific (62%) | sharma2024_dswp (23%) | sharma2025_birth (15%) |
|---|---:|---:|---:|
| `has_timestamps = 1` | 0% | 42% | 100% |
| `Whale != ::UNK` | 0% | 62% | 100% |
| `Synchrony == 1` | 0% | 9% | 26% |

Hersh dominates. The whale model trains on a corpus where 62% of rows have no speaker channel and no timing channel; the model has to *learn to gate* on `has_timestamps`. Standard CHILDES has both channels populated everywhere, so it doesn't share that training-time problem.

This experiment mirrors Hersh's missingness pattern onto CHILDES so the two corpora share the same training-time data shape. It builds on top of the [[transformer/vocab-cap-experiment|V=467 cap]].

## Loader changes

Two changes in `src/grammar/childes_loader.py`:

- New flag `--mirror-hersh-frac F` (default `0.0`). When `F > 0`, randomly select sequences (chunk-level, not row-level) until cumulative tokens ≥ `F × N_total`, and rewrite those rows with `Whale="childes::UNK"`, `TimeDelta=-1.0`, `has_timestamps=0`.
- Sequence-level selection (not row-level) matches whale, where missingness is a property of whole recordings.

`src/grammar/dt_buckets.py` got a fourth CHILDES bucket:

```
0  intra    DT in {0.0, 0.3}
1  period   DT == 1.0
2  switch   DT == 2.0
3  missing  has_timestamps == 0    ← NEW
```

The whale 6-bucket scheme already has a `missing` bucket; CHILDES now has the matching parallel.

## Build the mirrored corpus

```bash
python -m src.grammar.childes_loader \
    --vocab-cap 467 --mirror-hersh-frac 0.624
```

Resulting CSV (`data/classified/childes_dialogues.csv`):

| | original (no mirror) | **with --mirror-hersh-frac 0.624** | whale (reference) |
|---|---:|---:|---:|
| total tokens | 39,318 | 39,318 | 38,840 |
| sequences | 405 | 405 | 488 |
| `has_timestamps == 0` rate | 0% | **37.2%** | 75.5% |
| `Whale ends in ::UNK` rate | 0% | **62.8%** | 71.2% |
| `TimeDelta == -1.0` rate | 0% | **62.8%** | 75.5% |

The 62.8% mirror rate matches Hersh's *sequence-level* fraction (62.4%). Whale's *aggregate* missingness is higher (~75%) because DSWP also has partial NA — we can bump `--mirror-hersh-frac 0.75` if we want exact aggregate parity, but matching Hersh-only keeps the experiment clean.

## Result of training

Trained on the mirrored CSV with `train_for_interp.py --source childes`:

| epoch | uncapped V=1605 | capped V=467 (no mirror) | **capped V=467 + mirror** |
|---:|---:|---:|---:|
| 1 train | 5.716 | 4.990 | 4.990 |
| 1 val | 5.532 | 4.749 | 4.750 |
| 5 val | 5.537 | 4.634 | 4.636 |
| 8 val | 5.623 | 4.627 | 4.627 |
| early stop | epoch 8 | epoch 12 | epoch 12 |
| best val_coda | **5.497** (ep 3) | **4.627** (ep 7) | **4.627** (ep 7) |

**Coda head is unaffected.** Train and val coda losses match the no-mirror run epoch-by-epoch within ±0.01 bpt. Predicted: the Coda token sequence is unchanged by the mirror — only side channels (Whale, TimeDelta, has_timestamps) are nuked. So coda CE — computed on every position regardless of mirror status — gets the same loss either way.

**DT head behaves differently.** With 4 buckets now (intra/period/switch/missing) and 63% of positions deterministically mapping to the missing bucket, val_dt_bpt jumps from 0.69 → 1.18. The model learns the easy "predict missing when has_timestamps=0" rule but doesn't generalize the 3-class within-utterance structure as cleanly.

## What this enables

The mirrored CSV is the substrate for the **masked multi-task loss** experiment (open follow-up):

> Train coda on every position. Train DT and speaker heads only on rows with `has_timestamps=1`. Both corpora's auxiliary heads now get supervised on real signal only — exactly mirroring whale's setup, where Hersh rows can't supervise speaker or rubato either.

The masked-loss change is ~5 lines in `train_for_interp.py:224-230` (`shifted_ce` gains a `mask` argument) and ~10 lines in `predict_kfold.py:734-748` (gate aux-head loss terms by `tgt["has_ts"]`). Not yet wired.

## Open follow-up: aggregate missingness mirror

The current mirror matches Hersh's 62% (sequence-level rate). Whale's row-level aggregate has_timestamps=0 rate is closer to 75% because DSWP rows also lack timestamps half the time. To fully mirror, run with `--mirror-hersh-frac 0.75`. Cheaper than implementing a 3-tier (Hersh/DSWP/Birth) emulation per [[transformer/multilang-corpus-plan]].

## Reproduce

```bash
# Build the mirrored CSV
python -m src.grammar.childes_loader \
    --vocab-cap 467 --mirror-hersh-frac 0.624

# Re-train (needs the 4-bucket DT scheme — already in dt_buckets.py)
python -u -m src.grammar.train_for_interp --source childes \
    2>&1 | tee reproducibility/logs/train_childes_v467_mirror.log
```

Wall time: same as the non-mirrored run (~10 min CPU).

Backups:
- `data/classified/childes_dialogues_v467.csv` — V=467 unmirrored.
- `outputs/grammar/checkpoints/childes_v467/` — checkpoint.

## Related

- [[transformer/vocab-cap-experiment]] — V=467 cap, prerequisite for this experiment.
- [[overview/corpora]] — what makes Hersh's missingness pattern what it is.
- [[transformer/multilang-corpus-plan]] — extend the mirror to a 3-tier multi-language Hersh/DSWP/Birth emulation.
