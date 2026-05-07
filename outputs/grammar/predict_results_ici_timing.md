# ICI-timing benchmark

Evaluate model predictions in **raw (unbucketed) timing space**.
Predicted ICI strings decoded to seconds via per-bin empirical means (train-fold); compared to true raw ICI values.

Tier-aware 5-fold KFold (clean test; hersh + remaining clean as train). K=8.

| metric | description |
|--------|-------------|
| n_click_err (↓) | \|pred_clicks − true_clicks\| |
| nn_dist (↓) | mean nearest-neighbour click distance, norm by true coda duration |
| scale_free_rmse (↓) | RMSE of ICI sequences after normalising each to unit sum |
| dtw_norm (↓) | DTW path cost on ICI sequences, norm by max(n,m) |
| pattern_corr (↑) | Pearson r of normalised ICI patterns (truncated to shorter) |

## n_click_err ↓

| model | mean ± std |
|-------|:---------:|
| morpheme_tfm_k8 | 1.0948 ± 0.1431 |
| morpheme_markov1 | 1.1151 ± 0.1200 |
| compound_tfm_k8 | 1.6884 ± 0.3407 |
| compound_markov1 | 1.6898 ± 0.3410 |

## nn_dist ↓

| model | mean ± std |
|-------|:---------:|
| compound_tfm_k8 | 0.0728 ± 0.0077 |
| compound_markov1 | 0.0798 ± 0.0039 |
| morpheme_tfm_k8 | 0.0961 ± 0.0132 |
| morpheme_markov1 | 0.1113 ± 0.0097 |

## scale_free_rmse ↓

| model | mean ± std |
|-------|:---------:|
| morpheme_markov1 | 0.1309 ± 0.0243 |
| morpheme_tfm_k8 | 0.1315 ± 0.0247 |
| compound_tfm_k8 | 0.1643 ± 0.0340 |
| compound_markov1 | 0.1645 ± 0.0337 |

## dtw_norm ↓

| model | mean ± std |
|-------|:---------:|
| morpheme_tfm_k8 | 0.0689 ± 0.0086 |
| morpheme_markov1 | 0.0706 ± 0.0073 |
| compound_markov1 | 0.1138 ± 0.0074 |
| compound_tfm_k8 | 0.1155 ± 0.0084 |

## pattern_corr ↑

| model | mean ± std |
|-------|:---------:|
| morpheme_tfm_k8 | 0.2858 ± 0.0340 |
| morpheme_markov1 | 0.2396 ± 0.0415 |
| compound_tfm_k8 | 0.1092 ± 0.1060 |
| compound_markov1 | 0.0916 ± 0.0961 |

## Per-fold detail

### compound_tfm_k8

| fold | n_click_err | nn_dist | scale_free_rmse | dtw_norm | pattern_corr | n_valid |
|------|------:|------:|------:|------:|------:|--------:|
| 0 | 1.7008 | 0.0583 | 0.1195 | 0.1181 | 0.1271 | 2777 |
| 1 | 1.8874 | 0.0716 | 0.1970 | 0.1248 | 0.0778 | 1465 |
| 2 | 1.0805 | 0.0777 | 0.1262 | 0.1061 | 0.1501 | 6560 |
| 3 | 2.1018 | 0.0773 | 0.1890 | 0.1232 | -0.0668 | 776 |
| 4 | 1.6717 | 0.0790 | 0.1899 | 0.1051 | 0.2580 | 1398 |

### morpheme_tfm_k8

| fold | n_click_err | nn_dist | scale_free_rmse | dtw_norm | pattern_corr | n_valid |
|------|------:|------:|------:|------:|------:|--------:|
| 0 | 1.0832 | 0.0917 | 0.1094 | 0.0624 | 0.2249 | 2777 |
| 1 | 1.2191 | 0.1048 | 0.1483 | 0.0808 | 0.2827 | 1465 |
| 2 | 0.8233 | 0.0826 | 0.1019 | 0.0568 | 0.2982 | 6560 |
| 3 | 1.1572 | 0.0843 | 0.1289 | 0.0697 | 0.2943 | 776 |
| 4 | 1.1910 | 0.1174 | 0.1689 | 0.0750 | 0.3288 | 1398 |

### compound_markov1

| fold | n_click_err | nn_dist | scale_free_rmse | dtw_norm | pattern_corr | n_valid |
|------|------:|------:|------:|------:|------:|--------:|
| 0 | 1.7029 | 0.0736 | 0.1210 | 0.1167 | 0.0994 | 2777 |
| 1 | 1.8928 | 0.0793 | 0.1973 | 0.1226 | 0.0401 | 1465 |
| 2 | 1.0779 | 0.0789 | 0.1260 | 0.1055 | 0.1465 | 6560 |
| 3 | 2.0966 | 0.0815 | 0.1881 | 0.1197 | -0.0560 | 776 |
| 4 | 1.6788 | 0.0855 | 0.1903 | 0.1044 | 0.2281 | 1398 |

### morpheme_markov1

| fold | n_click_err | nn_dist | scale_free_rmse | dtw_norm | pattern_corr | n_valid |
|------|------:|------:|------:|------:|------:|--------:|
| 0 | 1.1048 | 0.1013 | 0.1095 | 0.0621 | 0.1737 | 2777 |
| 1 | 1.1706 | 0.1177 | 0.1459 | 0.0798 | 0.3032 | 1465 |
| 2 | 0.9102 | 0.0993 | 0.1046 | 0.0630 | 0.2527 | 6560 |
| 3 | 1.1108 | 0.1135 | 0.1242 | 0.0707 | 0.2321 | 776 |
| 4 | 1.2790 | 0.1248 | 0.1701 | 0.0775 | 0.2364 | 1398 |

