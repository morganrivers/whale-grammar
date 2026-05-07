# Full-factorial whale-corpus benchmark (arch × target × loss-aggregation, tiered split)

5-fold KFold over **clean** tier (sharma2024_dswp + sharma2025_birth). Per fold: train pool = remaining 4/5 of clean ∪ all hersh2022_pacific; val = 10 % slice of clean_train; test = held-out 1/5 of clean. Eval = held-out **last-position** bpt.

V_coda = 131, V_token = 467 (compound rhythm·tempo·orn·rubato), V_dt = 7 (whale scheme: missing + 5 timing buckets + switch). K = 8. Per-cell: AdamW lr=0.001, bs=128, max 60 epochs, ES patience 8.

## Main sweep — coda-target cells (scheme = M)

| arch | loss_agg | params | coda_bpt (↓) | coda_acc |
|------|----------|-------:|-------------:|---------:|
| tfm | last | 46,979 | 2.013 ± 0.534 | 0.692 |

## Main sweep — joint-target cells (Token + DT bucket, scheme = M)

| arch | loss_agg | params | token_bpt (↓) | dt_bpt (↓) | coda_marg_bpt (↓) | token_acc | dt_acc |
|------|----------|-------:|--------------:|-----------:|------------------:|----------:|-------:|

## Classical reference baselines — coda target (rhythm-class only)

Train = hersh ∪ clean_train (same as neural M scheme). Markov uses Laplace α=0.5 smoothing. KN5 = Modified Kneser-Ney 5-gram. KN5+cache = KN5 interpolated with a per-sequence recency cache (size=20, λ=0.15).

| model | coda_bpt (↓) | coda_acc |
|-------|-------------|---------|
| Majority unigram | 3.415 ± 0.549 | 0.609 |
| Markov-1 (Laplace α=0.5) | 2.533 ± 0.413 | 0.604 |
| Markov-2 (Laplace α=0.5) | 2.623 ± 0.485 | 0.662 |
| KN 5-gram | 2.268 ± 0.462 | 0.655 |
| KN 5-gram + recency cache (λ=0.15, win=20) | 2.292 ± 0.401 | 0.655 |

## Classical reference baselines — joint target (compound token, coda-marginalised)

Same train/test split as neural M-scheme joint cells. Models predict the compound token (rhythm·tempo·orn·rubato, V_token). No DT head. coda_marg_bpt = log₂-bpt of the true rhythm class after summing model probabilities over all compound tokens that share it. coda_marg_acc = argmax compound token → map to rhythm class → check.

| model | token_bpt (↓) | coda_marg_bpt (↓) | coda_marg_acc |
|-------|--------------|------------------|--------------|
| Majority unigram | 6.421 ± 0.289 | 3.416 ± 0.546 | 0.609 |
| Markov-1 (Laplace α=0.5) | 5.130 ± 0.325 | 2.782 ± 0.387 | 0.626 |
| Markov-2 (Laplace α=0.5) | 5.941 ± 0.311 | 3.431 ± 0.353 | 0.641 |
| KN 5-gram | 4.874 ± 0.312 | 2.321 ± 0.455 | 0.631 |
| KN 5-gram + recency cache (λ=0.15, win=20) | 4.838 ± 0.286 | 2.375 ± 0.390 | 0.635 |

## Coda-comparable summary (all cells, on the rhythm-class target)

Joint-target rows show **coda_marg_bpt** — the bpt obtained by marginalising softmax(Token) probabilities over their underlying rhythm class. Directly comparable to the coda-target rows.

| arch | target | loss_agg | params | coda bpt (↓) |
|------|--------|----------|-------:|-------------:|

| scheme | arch | target | loss_agg | params | coda bpt (↓) |
|--------|------|--------|----------|-------:|-------------:|
| M | tfm | coda | last | 46,979 | 2.013 ± 0.534 |
| ref | kn5 | coda | n/a | 0 | 2.268 ± 0.462 |
| ref | kn5cache | coda | n/a | 0 | 2.292 ± 0.401 |
| ref | kn5 | joint | n/a | 0 | 2.321 ± 0.455 |
| ref | kn5cache | joint | n/a | 0 | 2.375 ± 0.390 |
| ref | markov1 | coda | n/a | 0 | 2.533 ± 0.413 |
| ref | markov2 | coda | n/a | 0 | 2.623 ± 0.485 |
| ref | markov1 | joint | n/a | 0 | 2.782 ± 0.387 |
| ref | majority | coda | n/a | 0 | 3.415 ± 0.549 |
| ref | majority | joint | n/a | 0 | 3.416 ± 0.546 |
| ref | markov2 | joint | n/a | 0 | 3.431 ± 0.353 |

## Per-fold detail

### tfm | coda | last

| fold | coda_bpt | coda_acc |
|------|---------:|---------:|
| 0 | 1.555 | 0.778 |
| 1 | 1.604 | 0.760 |
| 2 | 2.973 | 0.559 |
| 3 | 2.214 | 0.617 |
| 4 | 1.717 | 0.746 |

