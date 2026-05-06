# Sampled continuations — whale vs CHILDES Eng-UK

Locked architecture: MiniTransformer-DT, 2L, 4h, d=64, K=8 + DT. Sampling: T=0.9, top-k=40, multinomial. For each source we hold out **3** sequences (not seen in training), feed the first K=8 tokens as a prefix, and autoregressively sample **3** continuations of length ≈ mean training-set sequence length.

Whale tokens are rhythm-class labels (e.g. `1+1+5`); CHILDES tokens are `%mor` lemmas (e.g. `go`, `the`, `book`). For both, `?` marks an out-of-vocabulary id and `·` marks a PAD slot.

## whale  (mean seq length = 80, continuations are 72 tokens long)

### Seed 1 — `sharma2025_birth::CETI23-286` (length 151)

**prefix (8 tokens, fed verbatim):**

> 1+1+3|t3|o0|r0, 7RP1|t3|o0|r1, 1+1+3|t1|o0|r1, 1+1+3|t3|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 5-NOISE|t2|o0|r0, 1+1+3|t1|o0|r1

**actual continuation in held-out data (for reference, model never saw this):**

> 1+1+3|t3|o0|r1, 1+1+3|t1|o0|r0, 1+1+3|t3|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r1, 1+1+3|t3|o0|r1, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r1, 1+1+3|t3|o0|r1, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r1, 1+1+3|t2|o0|r1, 1+1+3|t1|o0|r0, 1+1+3|t1|o1|r0, 4D|t1|o0|r1, 1+1+3|t3|o0|r1, 1+1+3|t2|o1|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t3|o0|r0, 1+1+3|t2|o1|r0, 6-NOISE|t4|o1|r0, 1+1+3|t1|o0|r1, 4D|t1|o0|r1, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 4R|t2|o0|r0, 1+1+3|t3|o1|r0, 4R|t2|o0|r0, 1+1+3|t2|o1|r0, 4R|t2|o0|r0, 1+1+3|t3|o1|r0, 6i|t0|o1|r1, 4R|t2|o0|r1, 1+1+3|t3|o0|r0, 5R|t3|o0|r0, 5R|t0|o0|r0, 1+1+3|t3|o0|r0, 1+1+3|t3|o0|r0, 1+1+3|t3|o1|r0, 4R|t2|o0|r0, 1+1+3|t2|o1|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t3|o0|r1, 1+1+3|t1|o0|r1, 1+1+3|t2|o0|r1, 1+1+3|t1|o0|r1, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o1|r0, 4D|t1|o0|r0, 1+1+3|t1|o1|r0, 1+1+3|t3|o0|r0, 4RP4|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o1|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0

**3 sampled continuations:**

1. 2+3|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t2|o0|r0, 5R|t1|o0|r0, 8R|t4|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t3|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t3|o0|r0, 6i|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0
2. 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 7i|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t0|o0|r0, 1+1+3|t2|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0
3. 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 7i|t0|o0|r0, 1+1+3|t1|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 1+1+3|t3|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 1+1+3|t2|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 4R|t2|o0|r0

### Seed 2 — `sharma2025_birth::CETI23-291` (length 80)

**prefix (8 tokens, fed verbatim):**

> 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 2+3|t1|o0|r0

**actual continuation in held-out data (for reference, model never saw this):**

> 5-NOISE|t3|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 6iP3|t0|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r1, 1+1+3|t1|o0|r1, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r1, 1+1+3|t1|o0|r1, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r1, 1+1+3|t1|o0|r1, 1+1+3|t1|o0|r0, 1+1+3|t1|o1|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+31|t0|o0|r0, 1+1+3|t2|o1|r1, 5-NOISE|t1|o0|r1, 5R|t1|o0|r1, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r1, 1+1+3|t3|o0|r1, 1+1+3|t2|o0|r0, 1+1+3|t3|o0|r0, 1+1+3|t3|o0|r0, 1+1+3|t3|o0|r0, 1+1+3|t3|o0|r0, 1+1+3|t3|o0|r0, 5RP4|t4|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o1|r0, 1+31|t2|o0|r1, 1+1+3|t2|o0|r1, 1+1+3|t2|o1|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t3|o0|r0, 5R|t1|o0|r0, 8-NOISE|t2|o0|r0, 4R|t2|o0|r1, 1+31|t1|o0|r1, 1+1+3|t3|o1|r0, 1+1+3|t4|o0|r0, 6-NOISE|t3|o1|r1, 1+1+3|t2|o1|r1, 1+1+3|t3|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t3|o0|r1, 1+1+3|t2|o0|r1, 1+1+3|t2|o0|r0, 8D|t4|o0|r1, 1+1+3|t2|o0|r1

**3 sampled continuations:**

1. 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 8R|t4|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 9R|t4|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 7R|t4|o0|r0, 7R|t4|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t2|o0|r0, 5R|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5-NOISE|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 7R|t4|o0|r0, 1+1+3|t2|o0|r0, 9R|t4|o0|r0, 6R|t4|o0|r0, 7R|t4|o0|r0, 6R|t3|o0|r0, 5R|t2|o0|r0, 7R|t4|o0|r0, 8R|t4|o0|r0, 5R|t1|o0|r0, 7R|t4|o0|r0, 7R|t4|o0|r0, 4R|t2|o0|r0, 7R|t4|o0|r0, 10RP1|t4|o0|r0, 5R|t1|o0|r0, 5R|t2|o0|r0
2. 4R|t0|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t4|o0|r0, 7i|t2|o0|r0, 5R|t3|o0|r0, 5R|t3|o0|r0, 6i|t2|o0|r0, 1+1+3|t1|o0|r0, 5R|t4|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 6i|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t3|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 4D|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t3|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 5R|t0|o0|r0, 1+1+3|t2|o0|r0
3. 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 4R|t2|o0|r0, 5R|t1|o0|r0, 4R|t2|o0|r0, 7i|t1|o0|r0, 4R|t2|o0|r0, 7i|t2|o0|r0

### Seed 3 — `sharma2025_birth::CETI23-293` (length 567)

**prefix (8 tokens, fed verbatim):**

> 6iP3|t0|o0|r0, 1+1+3|t2|o0|r0, 6iP3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0

**actual continuation in held-out data (for reference, model never saw this):**

> 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r1, 5R|t1|o0|r1, 6iP3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 6iP3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 6-NOISE|t3|o1|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 6iP3|t3|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t3|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t3|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r1, 1+1+3|t3|o0|r1, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 3D|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r1, 1+1+3|t2|o0|r1, 8R|t3|o0|r1, 1+1+3|t2|o0|r1, 1+1+3|t2|o0|r1, 1+1+3|t2|o0|r1, 7D|t3|o0|r0, 1+1+3|t2|o0|r1, 8R|t3|o0|r1, 1+1+3|t2|o0|r1, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r1, 1+1+3|t2|o0|r1, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0

**3 sampled continuations:**

1. 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0
2. 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t0|o0|r0, 5R|t0|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 4D|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 3+1+1+1+1|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0
3. 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t2|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 6i|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+31|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 7D|t3|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 1+1+3|t2|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 5R|t1|o0|r0, 1+1+3|t1|o0|r0, 3+1+1+1+1|t2|o0|r0, 5R|t1|o0|r0

## childes_uk  (mean seq length = 97, continuations are 89 tokens long)

### Seed 1 — `childes::020218::12::20` (length 80)

**prefix (8 tokens, fed verbatim):**

> no you want to here we be right

**actual continuation in held-out data (for reference, model never saw this):**

> what be in here that what be this want this one you want that that be the cake jackinthebox jackinthebox i put it on turn it on i do not know where your be oh look who be this mine you what mine it be yours i know your van lorry he be go round he be go round and round and round and round yes i can see as well no

**3 sampled continuations:**

1. over there as well end have not you it with your lunch what else oh it do you see her cm i know i will not want to do that want mummy let us i have not she do some more what how do you want to be in the garage on the floor on your leg again i will just just get it again that will get two three one of me and round and round and round go round and round and granddad be do the top
2. you can not cm thomas have some water yeah let us put that there here i be it no no well not get this and then you have you have have you sing there be the air let us do the air yeah oh okay cm mummy oh it be the wall ah yes do you know i say i say you can be the farm the bus be you can not see on her hat you yeah no okay let me find do she go like a party
3. you have a baby no you have a look and a police and get the car beep the way and do you want to the garden frosty with him i be not real kiss he will do no you be not fall down you be not it a minute no yes you get down with me to me and look oh dear what be you do what number do you want to do now oh it be they okay okay be it be oh you do not you play

### Seed 2 — `childes::020219::0::53` (length 80)

**prefix (8 tokens, fed verbatim):**

> what be this that be a balloon a

**actual continuation in held-out data (for reference, model never saw this):**

> hot air balloon do you remember look at balloon in the sky a more balloon more balloon yes those be parachute person in parachute come aeroplane and they keep hold and they glide down from the sky where be the cycle scooter scooter that be like po scooter end be not it the dump truck beep beep oh a bus oh a bus where be the bus a truck there a bus

**3 sampled continuations:**

1. little jack darling be not you want wee now okay what mummy be a little bit she be we come with something yeah and these as well oh no you go and then you be a tire on daddy do i will go to play with in this ball i will make a good boy she be alright you be a monster okay cm be they come on i can not see a bit more time not you have a baby baby toy okay okay darling it be too
2. juice bus i do not want to come down what it be it you and some more hey you like that be nice jam what you have to watch me daddy i do not know where and lie the button at the other man cm my darling cm that be it a bit of the button okay cm i want to do it so cute if you can not want tea cm i should not real run away cm just put these cm baby cm more more cm thank
3. girl yeah because you put me with your hair oh dear dada there be it round now let us have a look like the juice what what be it say the wall yeah oh what mum what oh mum do you think i want to be you get the bus cm you be your sleeve mummy cm you be it off it there okay cm that be it look at what she be this what shall we go that be a little van there be that be good idea

### Seed 3 — `childes::040002::6::79` (length 101)

**prefix (8 tokens, fed verbatim):**

> why the hill must be very steep it

**actual continuation in held-out data (for reference, model never saw this):**

> be a very high hill like yeah like mm do you remember when you walk up ooh i do this one humpty dumpty sit on a wall mum cm do you know mm hm that be humpty dumpty it be end be not it yeah and he have a hat on his head he have get a hat on his head end have not he humpty dumpty sit on a wall humpty dumpty have a great fall all the king horse and all the king man could not put

**3 sampled continuations:**

1. be a bit of pickle pepper yeah a truck off the car with them back and some more clothes me be you have one of you well that one one i think let it fit we eat the sky you have a little spark what number be that look like a aeroplane a bye this way a chair of helen mum be your eye you have you get it on the wall yeah and get that off that yeah but do you want it i can not eat it
2. be too much yeah with that yes no we just get me into the button you have it open i will not it to me it be i will finish it do not put a big bear and you be a spider climb up with it and her toe on the wall end be not you i be it yeah on the garage or be the bus be the tractor be the truck no no that be not a bit more more cm thank you this can you sit
3. be a bit c on her just get up with up yeah oh you put them on we put it on your tart and her go and then i do not go to cook oh dear mess now you go to be kiss her lesson to go and round and round my dinner and round round the little girl be we have to bed yeah i do you want me do you do a little bit of heart a little bit and the truck and some little heart go

