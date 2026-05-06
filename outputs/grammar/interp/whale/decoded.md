# Whale lexical decoding

Joins the model-side outputs from `interp_lexical.py` against rhythm and speaker metadata in `whale_dialogues.csv`.

Config: τ_cos=0.45, top-k=6, min count=30, K=8, cooc windows=4000.

## View 1 — Coda-family clusters from W_U

Connected components in the mutual-top-k cosine neighbor graph on the model's output embedding W_U. A coda enters the graph only if it occurs ≥ 30 times in the corpus. Edges require both ends to be in each other's top-6 cosine neighbors and cosine ≥ 0.45.

| # | size | members | n samples | mean Duration (s) | mean Δt-to-prev (s) | sync rate | orn rate | n distinct speakers | top speakers (n) |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 20 | 7, 8, 9, 16, 58, 62, 76, 77, 81, 91, 97, 101, 103, 113, 118, 119, 124, 126, 127, 128 | 9050 | 1.215 | 2.33 | 0.026 | 0.005 | 76 | `hersh2022::UNK` (7245); `sharma2024::UNK` (911); `sharma2024::local:2` (174) |
| 2 | 13 | 10, 12, 14, 18, 19, 43, 51, 61, 95, 109, 121, 125, 129 | 4761 | 0.558 | 7.18 | 0.023 | 0.025 | 79 | `hersh2022::UNK` (3693); `sharma2024::UNK` (435); `sharma2024::local:2` (90) |
| 3 | 4 | 28, 29, 40, 41 | 245 | 0.724 | — | 0.000 | 0.000 | 1 | `hersh2022::UNK` (245) |
| 4 | 4 | 45, 46, 47, 123 | 366 | 0.732 | 0.68 | 0.003 | 0.000 | 5 | `hersh2022::UNK` (358); `sharma2024::UNK` (4); `sharma2024::local:2` (2) |
| 5 | 4 | 72, 89, 100, 116 | 2081 | 1.265 | 2.49 | 0.047 | 0.066 | 73 | `hersh2022::UNK` (1627); `sharma2024::UNK` (121); `sharma2024::local:2` (48) |
| 6 | 3 | 20, 50, 71 | 373 | 1.229 | — | 0.000 | 0.000 | 1 | `hersh2022::UNK` (373) |
| 7 | 3 | 27, 35, 37 | 130 | 0.565 | — | 0.000 | 0.000 | 1 | `hersh2022::UNK` (130) |
| 8 | 2 | 4, 52 | 1134 | 0.580 | 1.75 | 0.048 | 0.000 | 57 | `hersh2022::UNK` (743); `sharma2024::photo:5722` (169); `sharma2024::UNK` (65) |
| 9 | 2 | 49, 65 | 92 | 0.855 | 0.81 | 0.022 | 0.000 | 6 | `hersh2022::UNK` (78); `sharma2024::UNK` (10); `sharma2024::photo:5133` (1) |
| 10 | 2 | 64, 107 | 157 | 0.951 | — | 0.000 | 0.000 | 1 | `hersh2022::UNK` (157) |

## View 2 — Mutual attention pairs decoded

Top-mutual coda pairs from the across-(layer,head) average attention co-occurrence matrix, each scanned against the corpus for K=8 co-occurrences:

- `mutual` is √(mean attn a→b · mean attn b→a) averaged across all 8 heads.
- `n` counts unique (i, j) coda-pair occurrences in any K=8 window where {tokens[i], tokens[j]} == {a, b} and i < j.
- `cross-speaker` is the fraction of those pairs where the speaker label changes between positions i and j (UNK-speaker rows excluded; n_eligible reported).
- `mean Δt` is the mean cumulative timing gap from i to j in seconds, computed only over pairs where both endpoints and all intervening rows have has_timestamps=1.
- `a→b` is the fraction of co-occurrences where token `a` comes first.

| a | b | mutual | n | a→b | cross-speaker (n_elig) | mean Δt s (n_elig) |
|---:|---:|---:|---:|---:|---|---|
| 29 | 41 | 0.112 | 138 | 0.54 | — (0) | — (0) |
| 0 | 80 | 0.108 | 9907 | 0.49 | 0.45 (2187) | 8.89 (1844) |
| 40 | 41 | 0.105 | 109 | 0.49 | — (0) | — (0) |
| 29 | 40 | 0.097 | 62 | 0.42 | — (0) | — (0) |
| 101 | 118 | 0.080 | 1573 | 0.53 | 0.22 (94) | 8.88 (33) |
| 76 | 97 | 0.076 | 563 | 0.51 | — (0) | — (0) |
| 16 | 58 | 0.072 | 75 | 0.47 | — (0) | — (0) |
| 50 | 89 | 0.068 | 642 | 0.49 | — (0) | — (0) |
| 103 | 118 | 0.065 | 1132 | 0.48 | — (0) | — (0) |
| 91 | 103 | 0.064 | 1084 | 0.47 | — (0) | — (0) |
| 121 | 129 | 0.063 | 656 | 0.63 | 0.25 (60) | 7.65 (17) |
| 96 | 104 | 0.063 | 49 | 0.47 | — (0) | — (0) |
| 118 | 126 | 0.062 | 926 | 0.51 | 0.26 (19) | 8.77 (15) |
| 76 | 81 | 0.062 | 707 | 0.52 | — (0) | — (0) |
| 81 | 91 | 0.059 | 1739 | 0.49 | 0.20 (5) | 3.58 (1) |
| 10 | 129 | 0.058 | 316 | 0.45 | 0.10 (21) | 8.41 (12) |
| 45 | 46 | 0.054 | 146 | 0.53 | — (0) | — (0) |
| 23 | 61 | 0.053 | 1612 | 0.50 | 0.29 (7) | 6.94 (6) |
| 109 | 121 | 0.053 | 832 | 0.57 | 0.15 (59) | 9.56 (16) |
| 91 | 118 | 0.052 | 1130 | 0.52 | — (0) | — (0) |
| 50 | 72 | 0.051 | 387 | 0.49 | — (0) | — (0) |
| 95 | 109 | 0.050 | 837 | 0.62 | 0.26 (76) | 6.20 (15) |
| 0 | 62 | 0.050 | 3915 | 0.50 | 0.66 (3391) | 9.35 (3380) |
| 23 | 64 | 0.049 | 435 | 0.51 | — (0) | — (0) |
| 72 | 89 | 0.048 | 689 | 0.51 | 0.58 (36) | 6.49 (33) |

## How to read

**Clusters**: a cluster sharing both Duration and TimeDelta-to-prev statistics means the model has learned rhythm-class affinity. A cluster dominated by one or two speakers means the model has captured an individual's vocal repertoire. High Synchrony rate = group/chorusing codas; high Ornamentation rate = decorated variants.

**Pairs**: high cross-speaker fraction is the conversational signature — these are codas that tend to bridge a speaker switch within K=8, i.e. *call-and-response*. Low cross-speaker fraction with short mean Δt is a *within-utterance rhythm motif* — codas that tend to be produced by the same whale in close sequence. Mean Δt of a few seconds with high cross-speaker fraction is the classic whale exchange pattern.
