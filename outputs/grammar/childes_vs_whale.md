# CHILDES Eng-UK vs whale — same MiniTransformer-DT, 3-fold CV

Architecture (locked from best-on-whale): MiniTransformer with DT — 2L, 4h, d=64, K=8, +`(log(0.1+dt), has_timestamps)` projection. Sequence-level KFold (3 folds). Metric = held-out cross-entropy in **bits/token** (log₂); lower is better.

Whale TimeDeltas come from observed acoustics; CHILDES TimeDeltas are estimated: 0.3 s intra-utterance, 0.5 s after `,`, 1.0 s after `.`/`?`/`!`, 2.0 s on speaker switch. CHILDES vocab uses `%mor` lemmas (so `going`/`went`/`gone`→`go`).

## whale (V=467, 488 sequences, 38,840 tokens)

| model | params | bits/token (↓) | perplexity (↓) | accuracy |
|-------|-------:|---------------:|---------------:|---------:|
| majority (smoothed unigram) | — | 6.570 ± 0.106 | 95.01 | 0.097 ± 0.013 |
| Markov-1 | — | 5.549 ± 0.125 | 46.80 | 0.299 ± 0.039 |
| MiniTransformer-DT (2L, 4h, d=64, K=8 + DT) | 160,915 | 4.541 ± 0.103 | 23.28 | 0.326 ± 0.018 |

## childes_uk (V=1602, 405 sequences, 39,318 tokens)

| model | params | bits/token (↓) | perplexity (↓) | accuracy |
|-------|-------:|---------------:|---------------:|---------:|
| majority (smoothed unigram) | — | 7.832 ± 0.040 | 227.84 | 0.065 ± 0.001 |
| Markov-1 | — | 8.084 ± 0.059 | 271.31 | 0.161 ± 0.002 |
| MiniTransformer-DT (2L, 4h, d=64, K=8 + DT) | 307,330 | 6.860 ± 0.073 | 116.18 | 0.180 ± 0.004 |

## Side-by-side compression

| source | V | majority bpt | MiniTfm-DT bpt | savings (bits) | fraction of unigram entropy |
|---|---:|---:|---:|---:|---:|
| whale | 467 | 6.570 | 4.541 | 2.029 | 0.691 |
| CHILDES UK | 1602 | 7.832 | 6.860 | 0.972 | 0.876 |

Lower *fraction of unigram entropy* = the model captures more structure relative to the entropy floor. Direct bits/token aren't comparable across V (CHILDES has ~12× larger vocabulary).

