# Lexical interp — childes

Source: `childes`. V=1605, N_CTX=8, corpus tokens=39318.

## Most frequent tokens

| rank | id | token | count |
|---:|---:|---|---:|
| 1 | 105 | `be` | 2533 |
| 2 | 1597 | `you` | 1782 |
| 3 | 697 | `it` | 1246 |
| 4 | 404 | `do` | 1210 |
| 5 | 0 | `a` | 1060 |
| 6 | 1371 | `the` | 1037 |
| 7 | 919 | `not` | 959 |
| 8 | 678 | `i` | 945 |
| 9 | 1370 | `that` | 769 |
| 10 | 1413 | `to` | 691 |
| 11 | 43 | `and` | 670 |
| 12 | 295 | `cm` | 658 |
| 13 | 572 | `go` | 600 |
| 14 | 614 | `have` | 587 |
| 15 | 933 | `oh` | 533 |
| 16 | 1529 | `what` | 522 |
| 17 | 912 | `no` | 517 |
| 18 | 1590 | `yeah` | 396 |
| 19 | 1377 | `there` | 373 |
| 20 | 938 | `on` | 366 |
| 21 | 1518 | `we` | 355 |
| 22 | 1508 | `want` | 351 |
| 23 | 220 | `can` | 344 |
| 24 | 1387 | `this` | 326 |
| 25 | 561 | `get` | 320 |
| 26 | 459 | `end` | 305 |
| 27 | 939 | `one` | 299 |
| 28 | 686 | `in` | 294 |
| 29 | 881 | `mummy` | 292 |
| 30 | 1598 | `your` | 264 |

## View 1 — Embedding neighborhoods

Cosine top-5 in the model's input embedding (W_E) and output embedding (W_U). Neighbor pool restricted to tokens with count ≥ 5 so rare-word noise doesn't dominate.

### W_E neighbors (input role)

| token | count | n1 | n2 | n3 | n4 | n5 |
|---|---:|---|---|---|---|---|
| `be` | 2533 | `floor` (0.37) | `crawl` (0.32) | `juice` (0.32) | `hair` (0.32) | `turn` (0.32) |
| `you` | 1782 | `thing` (0.39) | `hole` (0.37) | `cut` (0.29) | `game` (0.29) | `arm` (0.28) |
| `it` | 1246 | `mean` (0.37) | `without` (0.36) | `bus` (0.36) | `get` (0.35) | `she` (0.34) |
| `do` | 1210 | `dinosaur` (0.39) | `now` (0.35) | `tip` (0.33) | `night` (0.33) | `tap` (0.31) |
| `a` | 1060 | `his` (0.39) | `more` (0.36) | `boy` (0.34) | `high` (0.31) | `to` (0.30) |
| `the` | 1037 | `long` (0.35) | `and` (0.34) | `van` (0.33) | `buzz` (0.30) | `porridge` (0.27) |
| `not` | 959 | `end` (0.35) | `when` (0.34) | `worry` (0.33) | `eat` (0.31) | `some` (0.31) |
| `i` | 945 | `pardon` (0.39) | `ow` (0.35) | `without` (0.34) | `it` (0.33) | `mat` (0.30) |
| `that` | 769 | `word` (0.41) | `chocolate` (0.38) | `carry` (0.35) | `beautiful` (0.35) | `frosty` (0.33) |
| `to` | 691 | `song` (0.37) | `leave` (0.32) | `soup` (0.32) | `bee` (0.32) | `a` (0.30) |
| `and` | 670 | `duvet` (0.36) | `the` (0.34) | `vroom` (0.34) | `mend` (0.33) | `pikelet` (0.32) |
| `cm` | 658 | `grow` (0.35) | `hard` (0.34) | `sheep` (0.33) | `round` (0.32) | `cut` (0.32) |
| `go` | 600 | `ahhah` (0.38) | `wash` (0.34) | `year` (0.34) | `quick` (0.32) | `too` (0.30) |
| `have` | 587 | `reckon` (0.43) | `bike` (0.36) | `pick` (0.33) | `cut` (0.30) | `scream` (0.30) |
| `oh` | 533 | `apart` (0.38) | `care` (0.38) | `few` (0.32) | `wash` (0.31) | `half` (0.30) |
| `what` | 522 | `monkey` (0.35) | `bit` (0.33) | `pink` (0.31) | `along` (0.31) | `car` (0.30) |
| `no` | 517 | `new` (0.36) | `delete` (0.34) | `yourself` (0.34) | `horse` (0.34) | `porridge` (0.33) |
| `yeah` | 396 | `thomas` (0.35) | `our` (0.34) | `if` (0.34) | `engine` (0.32) | `already` (0.30) |
| `there` | 373 | `picture` (0.40) | `room` (0.38) | `shh` (0.37) | `song` (0.34) | `garden` (0.31) |
| `on` | 366 | `fall` (0.35) | `happy` (0.33) | `water` (0.33) | `while` (0.31) | `naughty` (0.30) |
| `we` | 355 | `already` (0.42) | `cut` (0.33) | `catch` (0.32) | `arm` (0.32) | `toast` (0.31) |
| `want` | 351 | `with` (0.35) | `van` (0.35) | `tara` (0.34) | `can` (0.32) | `picnic` (0.31) |
| `can` | 344 | `know` (0.37) | `work` (0.35) | `mine` (0.34) | `want` (0.32) | `forget` (0.31) |
| `this` | 326 | `snail` (0.42) | `toe` (0.32) | `together` (0.32) | `high` (0.28) | `eye` (0.27) |
| `get` | 320 | `quack` (0.39) | `hard` (0.36) | `take` (0.36) | `it` (0.35) | `they` (0.31) |
| `end` | 305 | `not` (0.35) | `cool` (0.34) | `cut` (0.34) | `sit` (0.34) | `cm` (0.31) |
| `one` | 299 | `helen` (0.45) | `far` (0.32) | `quite` (0.32) | `loose` (0.31) | `sticker` (0.31) |
| `in` | 294 | `dinner` (0.34) | `name` (0.34) | `number` (0.31) | `tigger` (0.30) | `excuse` (0.29) |
| `mummy` | 292 | `friend` (0.43) | `picnic` (0.35) | `lift` (0.33) | `him` (0.32) | `turn` (0.32) |
| `your` | 264 | `noise` (0.35) | `blow` (0.35) | `because` (0.31) | `get` (0.30) | `color` (0.30) |

### W_U neighbors (output role)

| token | count | n1 | n2 | n3 | n4 | n5 |
|---|---:|---|---|---|---|---|
| `be` | 2533 | `i` (0.61) | `do` (0.57) | `go` (0.54) | `yeah` (0.50) | `can` (0.50) |
| `you` | 1782 | `i` (0.61) | `not` (0.60) | `no` (0.58) | `oh` (0.56) | `it` (0.56) |
| `it` | 1246 | `you` (0.56) | `i` (0.54) | `a` (0.53) | `the` (0.51) | `cm` (0.51) |
| `do` | 1210 | `i` (0.59) | `be` (0.57) | `have` (0.54) | `can` (0.52) | `mummy` (0.52) |
| `a` | 1060 | `it` (0.53) | `go` (0.52) | `the` (0.44) | `i` (0.44) | `mummy` (0.44) |
| `the` | 1037 | `there` (0.56) | `i` (0.53) | `that` (0.53) | `it` (0.51) | `oh` (0.51) |
| `not` | 959 | `you` (0.60) | `alright` (0.38) | `it` (0.38) | `no` (0.35) | `put` (0.32) |
| `i` | 945 | `oh` (0.62) | `you` (0.61) | `be` (0.61) | `do` (0.59) | `what` (0.55) |
| `that` | 769 | `the` (0.53) | `what` (0.52) | `i` (0.52) | `and` (0.49) | `there` (0.45) |
| `to` | 691 | `down` (0.48) | `where` (0.45) | `round` (0.41) | `you` (0.39) | `go` (0.38) |
| `and` | 670 | `oh` (0.50) | `that` (0.49) | `what` (0.48) | `i` (0.48) | `on` (0.44) |
| `cm` | 658 | `you` (0.52) | `it` (0.51) | `okay` (0.50) | `there` (0.49) | `what` (0.48) |
| `go` | 600 | `be` (0.54) | `a` (0.52) | `just` (0.44) | `do` (0.43) | `good` (0.41) |
| `have` | 587 | `can` (0.59) | `do` (0.54) | `see` (0.51) | `want` (0.48) | `think` (0.48) |
| `oh` | 533 | `i` (0.62) | `okay` (0.57) | `you` (0.56) | `end` (0.52) | `the` (0.51) |
| `what` | 522 | `i` (0.55) | `that` (0.52) | `where` (0.50) | `you` (0.50) | `and` (0.48) |
| `no` | 517 | `you` (0.58) | `i` (0.50) | `daddy` (0.47) | `oh` (0.47) | `now` (0.46) |
| `yeah` | 396 | `be` (0.50) | `do` (0.47) | `you` (0.45) | `oh` (0.44) | `put` (0.44) |
| `there` | 373 | `the` (0.56) | `i` (0.50) | `in` (0.50) | `cm` (0.49) | `here` (0.49) |
| `on` | 366 | `and` (0.44) | `the` (0.42) | `make` (0.41) | `this` (0.41) | `i` (0.40) |
| `we` | 355 | `what` (0.46) | `it` (0.44) | `cm` (0.42) | `they` (0.33) | `more` (0.32) |
| `want` | 351 | `know` (0.67) | `think` (0.56) | `can` (0.49) | `have` (0.48) | `need` (0.43) |
| `can` | 344 | `think` (0.67) | `will` (0.60) | `have` (0.59) | `see` (0.52) | `do` (0.52) |
| `this` | 326 | `in` (0.50) | `cm` (0.45) | `the` (0.45) | `daddy` (0.45) | `there` (0.44) |
| `get` | 320 | `see` (0.48) | `have` (0.47) | `can` (0.42) | `come` (0.39) | `be` (0.34) |
| `end` | 305 | `oh` (0.52) | `they` (0.43) | `do` (0.41) | `the` (0.39) | `for` (0.38) |
| `one` | 299 | `time` (0.42) | `a` (0.37) | `back` (0.35) | `cm` (0.33) | `your` (0.31) |
| `in` | 294 | `there` (0.50) | `this` (0.50) | `cm` (0.43) | `the` (0.42) | `be` (0.40) |
| `mummy` | 292 | `do` (0.52) | `cm` (0.47) | `what` (0.45) | `a` (0.44) | `i` (0.43) |
| `your` | 264 | `my` (0.46) | `a` (0.42) | `it` (0.41) | `cm` (0.41) | `okay` (0.40) |

## View 2 — Attention co-occurrence

For each (L, h), `A[a, b]` is the mean attention from query positions with token `a` to key positions with token `b`, averaged over 4000 K=8 windows. Mutual score = √(A[a,b] · A[b,a]); only pairs with both tokens observed ≥ 20 times included. See `attn_cooccurrence.png` for top-30 heatmaps.

### Top mutual attention pairs per head

| L | H | a | b | a→b | b→a | mutual |
|---|---|---|---|---:|---:|---:|
| 0 | 0 | `b` | `c` | 0.105 | 0.306 | 0.180 |
| 0 | 0 | `boo` | `care` | 0.114 | 0.087 | 0.100 |
| 0 | 0 | `hat` | `her` | 0.298 | 0.014 | 0.066 |
| 0 | 0 | `be` | `that` | 0.044 | 0.085 | 0.061 |
| 0 | 0 | `shall` | `we` | 0.062 | 0.059 | 0.060 |
| 0 | 0 | `be` | `it` | 0.051 | 0.067 | 0.058 |
| 0 | 0 | `do` | `you` | 0.044 | 0.059 | 0.051 |
| 0 | 0 | `baby` | `chair` | 0.037 | 0.055 | 0.045 |
| 0 | 0 | `be` | `you` | 0.034 | 0.060 | 0.045 |
| 0 | 0 | `cm` | `darling` | 0.011 | 0.163 | 0.042 |
| 0 | 1 | `b` | `c` | 0.128 | 0.200 | 0.160 |
| 0 | 1 | `boo` | `care` | 0.162 | 0.070 | 0.107 |
| 0 | 1 | `hat` | `her` | 0.245 | 0.014 | 0.058 |
| 0 | 1 | `up` | `wake` | 0.046 | 0.070 | 0.057 |
| 0 | 1 | `baby` | `chair` | 0.034 | 0.080 | 0.052 |
| 0 | 1 | `be` | `it` | 0.029 | 0.068 | 0.044 |
| 0 | 1 | `do` | `you` | 0.047 | 0.042 | 0.044 |
| 0 | 1 | `be` | `that` | 0.045 | 0.041 | 0.043 |
| 0 | 1 | `a` | `bang` | 0.009 | 0.207 | 0.043 |
| 0 | 1 | `do` | `not` | 0.015 | 0.109 | 0.041 |
| 0 | 2 | `b` | `c` | 0.103 | 0.247 | 0.160 |
| 0 | 2 | `boo` | `care` | 0.071 | 0.118 | 0.092 |
| 0 | 2 | `be` | `it` | 0.050 | 0.073 | 0.060 |
| 0 | 2 | `hat` | `her` | 0.151 | 0.022 | 0.058 |
| 0 | 2 | `baby` | `chair` | 0.033 | 0.067 | 0.047 |
| 0 | 2 | `up` | `wake` | 0.032 | 0.064 | 0.045 |
| 0 | 2 | `do` | `you` | 0.037 | 0.051 | 0.043 |
| 0 | 2 | `shall` | `we` | 0.042 | 0.039 | 0.040 |
| 0 | 2 | `dear` | `mess` | 0.010 | 0.128 | 0.036 |
| 0 | 2 | `be` | `you` | 0.026 | 0.043 | 0.034 |
| 0 | 3 | `b` | `c` | 0.090 | 0.296 | 0.163 |
| 0 | 3 | `boo` | `care` | 0.090 | 0.059 | 0.073 |
| 0 | 3 | `baby` | `chair` | 0.036 | 0.105 | 0.061 |
| 0 | 3 | `let` | `us` | 0.011 | 0.294 | 0.057 |
| 0 | 3 | `hat` | `her` | 0.293 | 0.010 | 0.055 |
| 0 | 3 | `do` | `you` | 0.046 | 0.064 | 0.054 |
| 0 | 3 | `be` | `it` | 0.046 | 0.064 | 0.054 |
| 0 | 3 | `a` | `bang` | 0.012 | 0.226 | 0.053 |
| 0 | 3 | `be` | `that` | 0.038 | 0.050 | 0.043 |
| 0 | 3 | `do` | `not` | 0.016 | 0.110 | 0.042 |
| 1 | 0 | `b` | `c` | 0.110 | 0.223 | 0.157 |
| 1 | 0 | `boo` | `care` | 0.098 | 0.100 | 0.099 |
| 1 | 0 | `baby` | `chair` | 0.039 | 0.097 | 0.062 |
| 1 | 0 | `hat` | `her` | 0.207 | 0.014 | 0.054 |
| 1 | 0 | `do` | `you` | 0.039 | 0.063 | 0.049 |
| 1 | 0 | `be` | `it` | 0.041 | 0.056 | 0.048 |
| 1 | 0 | `be` | `that` | 0.035 | 0.054 | 0.044 |
| 1 | 0 | `let` | `us` | 0.008 | 0.225 | 0.043 |
| 1 | 0 | `a` | `bang` | 0.008 | 0.221 | 0.043 |
| 1 | 0 | `up` | `wake` | 0.030 | 0.055 | 0.040 |
| 1 | 1 | `b` | `c` | 0.094 | 0.266 | 0.158 |
| 1 | 1 | `boo` | `care` | 0.142 | 0.067 | 0.098 |
| 1 | 1 | `hat` | `her` | 0.175 | 0.020 | 0.060 |
| 1 | 1 | `be` | `it` | 0.046 | 0.068 | 0.056 |
| 1 | 1 | `up` | `wake` | 0.037 | 0.062 | 0.048 |
| 1 | 1 | `do` | `you` | 0.044 | 0.050 | 0.047 |
| 1 | 1 | `be` | `that` | 0.033 | 0.066 | 0.046 |
| 1 | 1 | `baby` | `chair` | 0.032 | 0.060 | 0.044 |
| 1 | 1 | `a` | `bang` | 0.009 | 0.202 | 0.043 |
| 1 | 1 | `let` | `us` | 0.009 | 0.183 | 0.041 |
| 1 | 2 | `b` | `c` | 0.163 | 0.198 | 0.180 |
| 1 | 2 | `boo` | `care` | 0.127 | 0.082 | 0.102 |
| 1 | 2 | `baby` | `chair` | 0.039 | 0.077 | 0.055 |
| 1 | 2 | `hat` | `her` | 0.215 | 0.013 | 0.054 |
| 1 | 2 | `be` | `it` | 0.046 | 0.057 | 0.051 |
| 1 | 2 | `do` | `you` | 0.047 | 0.054 | 0.050 |
| 1 | 2 | `up` | `wake` | 0.038 | 0.059 | 0.047 |
| 1 | 2 | `a` | `bang` | 0.010 | 0.222 | 0.047 |
| 1 | 2 | `be` | `that` | 0.036 | 0.054 | 0.044 |
| 1 | 2 | `shall` | `we` | 0.044 | 0.036 | 0.040 |
| 1 | 3 | `b` | `c` | 0.091 | 0.257 | 0.153 |
| 1 | 3 | `boo` | `care` | 0.124 | 0.078 | 0.098 |
| 1 | 3 | `hat` | `her` | 0.200 | 0.016 | 0.057 |
| 1 | 3 | `baby` | `chair` | 0.035 | 0.094 | 0.057 |
| 1 | 3 | `up` | `wake` | 0.038 | 0.069 | 0.051 |
| 1 | 3 | `be` | `it` | 0.048 | 0.053 | 0.050 |
| 1 | 3 | `do` | `you` | 0.042 | 0.057 | 0.049 |
| 1 | 3 | `a` | `bang` | 0.008 | 0.246 | 0.045 |
| 1 | 3 | `be` | `that` | 0.037 | 0.051 | 0.043 |
| 1 | 3 | `let` | `us` | 0.010 | 0.182 | 0.042 |

## View 3 — Bigram-baseline divergence

Higher KL = the model uses this token contextually rather than as a static lookup. KL is between full-model averaged next-token distribution and the model's own no-context baseline (`unembed(ln_final(W_E[t] + W_pos[K-1]))`).

**Most contextual (top 15)**
| token | count | KL bits | top bigram pred | top full pred |
|---|---:|---:|---|---|
| `here` | 16 | 4.50 | `cheep` | `be` |
| `see` | 20 | 4.02 | `bottle` | `be` |
| `good` | 10 | 3.03 | `bear` | `be` |
| `well` | 16 | 2.92 | `boy` | `be` |
| `my` | 19 | 2.90 | `bit` | `be` |
| `have` | 54 | 2.84 | `bottle` | `be` |
| `us` | 10 | 2.73 | `bottle` | `be` |
| `can` | 35 | 2.73 | `optician` | `be` |
| `one` | 32 | 2.73 | `deep` | `be` |
| `for` | 19 | 2.72 | `only` | `be` |
| `me` | 20 | 2.66 | `purple` | `be` |
| `know` | 20 | 2.62 | `boy` | `be` |
| `there` | 35 | 2.57 | `hole` | `be` |
| `you` | 198 | 2.55 | `boy` | `be` |
| `okay` | 15 | 2.51 | `then` | `be` |

**Most lookup-like (bottom 15)**
| token | count | KL bits | top bigram pred | top full pred |
|---|---:|---:|---|---|
| `the` | 105 | 0.69 | `this` | `be` |
| `at` | 17 | 0.79 | `of` | `be` |
| `where` | 14 | 0.90 | `be` | `be` |
| `more` | 17 | 0.93 | `be` | `be` |
| `dear` | 12 | 0.93 | `one` | `be` |
| `that` | 73 | 0.96 | `one` | `be` |
| `then` | 22 | 1.06 | `bit` | `be` |
| `and` | 77 | 1.08 | `baby` | `be` |
| `if` | 11 | 1.08 | `one` | `be` |
| `yeah` | 44 | 1.17 | `this` | `be` |
| `her` | 12 | 1.20 | `bang` | `be` |
| `let` | 14 | 1.20 | `no` | `be` |
| `no` | 49 | 1.22 | `bear` | `be` |
| `very` | 10 | 1.28 | `one` | `be` |
| `will` | 28 | 1.28 | `just` | `be` |

## Files

- `attn_cooccurrence.png` — (layer × head) heatmaps over the top-30 tokens.
- `bigram_kl_hist.png` — distribution of per-token KL (full || bigram) in bits.
