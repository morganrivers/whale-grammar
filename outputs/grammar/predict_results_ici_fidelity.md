# ICI-fidelity benchmark

How well does each model predict the true ICI string of the next whale coda?

Tier-aware 5-fold KFold (clean test set; hersh + remaining clean as train).
Compound model: K=8 joint checkpoint (loaded, no retraining).  Morpheme model: MiniTfm 2L-4h-d64 trained per fold.

| model | ici_bpt (↓) | exact_match (↑) | symbol_acc (↑) | norm_edit (↓) |
|-------|:----------:|:---------------:|:--------------:|:-------------:|
| morpheme_tfm_k8 | 6.817 ± 0.359 | 0.200 | 0.465 | 0.437 |
| morpheme_markov1 | 7.838 ± 0.300 | 0.198 | 0.462 | 0.426 |
| compound_tfm_k8 | 8.741 ± 0.276 | 0.020 | 0.203 | 0.757 |
| compound_markov1 | 9.058 ± 0.173 | 0.018 | 0.199 | 0.752 |

## Per-fold detail

### compound_tfm_k8

| fold | ici_bpt | exact_match | symbol_acc | norm_edit | n_valid |
|------|--------:|------------:|-----------:|----------:|--------:|
| 0 | 8.627 | 0.059 | 0.164 | 0.811 | 2777 |
| 1 | 8.453 | 0.008 | 0.229 | 0.727 | 1465 |
| 2 | 9.225 | 0.013 | 0.138 | 0.824 | 6560 |
| 3 | 8.547 | 0.009 | 0.161 | 0.805 | 776 |
| 4 | 8.853 | 0.012 | 0.323 | 0.616 | 1398 |

### morpheme_tfm_k8

| fold | ici_bpt | exact_match | symbol_acc | norm_edit | n_valid |
|------|--------:|------------:|-----------:|----------:|--------:|
| 0 | 7.274 | 0.221 | 0.480 | 0.423 | 2777 |
| 1 | 6.957 | 0.175 | 0.439 | 0.459 | 1465 |
| 2 | 6.237 | 0.218 | 0.476 | 0.436 | 6560 |
| 3 | 7.007 | 0.229 | 0.455 | 0.448 | 776 |
| 4 | 6.609 | 0.153 | 0.477 | 0.421 | 1398 |

### compound_markov1

| fold | ici_bpt | exact_match | symbol_acc | norm_edit | n_valid |
|------|--------:|------------:|-----------:|----------:|--------:|
| 0 | 8.879 | 0.049 | 0.162 | 0.804 | 2777 |
| 1 | 8.936 | 0.007 | 0.223 | 0.724 | 1465 |
| 2 | 9.326 | 0.013 | 0.137 | 0.824 | 6560 |
| 3 | 8.949 | 0.008 | 0.165 | 0.788 | 776 |
| 4 | 9.198 | 0.012 | 0.309 | 0.620 | 1398 |

### morpheme_markov1

| fold | ici_bpt | exact_match | symbol_acc | norm_edit | n_valid |
|------|--------:|------------:|-----------:|----------:|--------:|
| 0 | 7.943 | 0.215 | 0.487 | 0.399 | 2777 |
| 1 | 8.010 | 0.199 | 0.463 | 0.418 | 1465 |
| 2 | 7.245 | 0.210 | 0.440 | 0.471 | 6560 |
| 3 | 7.939 | 0.235 | 0.466 | 0.420 | 776 |
| 4 | 8.056 | 0.132 | 0.457 | 0.421 | 1398 |

