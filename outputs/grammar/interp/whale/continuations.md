# Continuations — whale

Sampling: coda head `T=0.9, top-k=40`; DT head `T=0.9, no top-k` (7 buckets). Mean held-out seed length is used as the target continuation length.

## seed 0 — `hersh2022_pacific::29`
original length: 110; generating 111 new tokens after the K=8 prefix

**prefix (first 8 tokens, rendered from real CSV):**
```
  Whale UNK: 5R|t3|o0|r0 Δt? 9R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 6iP3|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 9R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 8R|t4|o0|r0
```

**original continuation (ground truth):**
```
  Whale UNK: 6-NOISE|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 7R|t4|o0|r0 Δt? 5R|t3|o0|r0 Δt? 8R|t4|o0|r0 Δt? 9R|t4|o0|r0 Δt? 7R|t4|o0|r0 Δt? 7R|t4|o0|r0 Δt? 7R|t4|o0|r0 Δt? 5R|t2|o0|r0 Δt? 8R|t4|o0|r0 Δt? 5R|t2|o0|r0 Δt? 8R|t4|o0|r0 Δt? 4R|t1|o0|r0 Δt? 6R|t3|o0|r0 Δt? 7R|t4|o0|r0 Δt? 5R|t2|o0|r0 Δt? 2+3|t2|o0|r0 Δt? 8R|t4|o0|r0 Δt? 5R|t2|o0|r0 Δt? 8R|t4|o0|r0 Δt? 7R|t4|o0|r0 Δt? 7R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 7R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 7R|t4|o0|r0 Δt? 6R|t3|o0|r0 Δt? 6R|t3|o0|r0 Δt? 7R|t4|o0|r0 Δt? 10RP2|t4|o0|r0 Δt? 5R|t2|o0|r0 Δt? 6R|t4|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 7R|t4|o0|r0 Δt? 5R|t3|o0|r0 Δt? 7R|t4|o0|r0 Δt? 7R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 5R|t2|o0|r0 Δt? 5R|t2|o0|r0 Δt? 8R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 8RP1|t4|o0|r0 Δt? 6R|t4|o0|r0 Δt? 5R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 7R|t4|o0|r0 Δt? 6R|t3|o0|r0 Δt? 6R|t3|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 1+1+3|t1|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 6i|t2|o0|r0 Δt? 5R|t3|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 8R|t4|o0|r0 Δt? 7R|t4|o0|r0 Δt? 6R|t3|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 6R|t3|o0|r0 Δt? 8R|t4|o0|r0 Δt? 9R|t4|o0|r0 Δt? 6R|t3|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 8R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 5R|t0|o0|r0 Δt? 5R|t0|o0|r0 Δt? 1+1+3|t1|o0|r0 Δt? 5R|t0|o0|r0 Δt? 5R|t0|o0|r0 Δt? 5R|t1|o0|r0 Δt? 5R|t1|o0|r0 Δt? 5R|t1|o0|r0 Δt? 1+1+3|t1|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 10RP1|t4|o0|r0 Δt? 5R|t0|o0|r0 Δt? 5R|t0|o0|r0 Δt? 5R|t0|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 8R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 1+1+3|t2|o0|r0
```

**sample 1:**
```
  Whale UNK: 5R|t0|o0|r0 Δt? 7i|t1|o0|r0 Δt? 4P5|t3|o0|r0 Δt? 10i|t2|o0|r0 Δt? 5R|t0|o0|r0 Δt? 5R|t0|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t4|o0|r0 Δt6.00 5R|t4|o0|r0 Δt? 1+1+3|t4|o0|r1 Δt? 4R|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 4R|t0|o0|r0 Δt? 5R|t3|o0|r0 Δt? 9-NOISE|t4|o0|r0
  Whale UNK: 1+31|t1|o0|r0 Δt? 4R|t2|o0|r0 Δt? 4R|t2|o0|r0 Δt? 4+1|t1|o0|r0 Δt? 9-NOISE|t4|o0|r0 Δt? 4R|t3|o0|r0 Δt? 4R|t3|o0|r0 Δt? 4R|t2|o0|r0 Δt? 4R|t2|o0|r0 Δt? 9i|t1|o0|r0 Δt? 4R|t3|o0|r0
  Whale UNK: 4R|t3|o0|r0 Δt? 3D|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt? 4R|t2|o0|r0 Δt? 1+1+3|t2|o0|r1 Δt? 5R|t0|o0|r0 Δt? 4RP2|t2|o0|r0 Δt? 3D|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt3.20 3D|t0|o0|r0 Δt? 4R|t0|o0|r0 Δt? 5R|t3|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt? 3D|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt? 10iP2|t3|o0|r0 Δt? 3D|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4R|t3|o0|r0 Δt? 7RP4|t3|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 4R|t3|o0|r0 Δt? 5R|t0|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4R|t3|o0|r0 Δt? 4D|t1|o0|r0 Δt? 2+3|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+1+3|t3|o0|r0 Δt? 4D|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4D|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t2|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 5P4|t4|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4R|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 6-NOISE|t4|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4R|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4R|t3|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4R|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 6i|t0|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4D|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+1+3|t3|o0|r0 Δt? 6iP2|t4|o0|r0 Δt? 4D|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt3.20 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0
  Whale UNK: 6RP1|t2|o0|r0 Δt? 6iP1|t2|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+1+3|t3|o1|r0
```

**sample 2:**
```
  Whale UNK: 9i|t1|o0|r0 Δt? 8i|t2|o0|r0 Δt? 7i|t1|o0|r0 Δt? 8i|t2|o0|r0 Δt? 8i|t2|o0|r0 Δt? 6i|t1|o0|r0 Δt? 8i|t2|o0|r0 Δt? 4R|t3|o0|r0 Δt? 4R|t3|o0|r0 Δt? 4R|t3|o0|r0 Δt0.20 4P5|t3|o0|r0 Δt? 4P5|t3|o0|r0 Δt? 4P5|t3|o0|r0 Δt? 6iP2|t4|o0|r0 Δt? 4R|t3|o0|r0 Δt? 4R|t3|o0|r0
  Whale UNK: 8-NOISE|t4|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 5-NOISE|t4|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt? 5-NOISE|t4|o0|r0
  Whale UNK: 1+1+3|t4|o0|r0 Δt? 4R|t3|o0|r0 Δt? 1+1+3|t3|o1|r0
  Whale UNK: 5-NOISE|t4|o0|r0 Δt? 1+1+3|t3|o0|r0 Δt6.00 4R|t3|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt? 6i|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 6i|t1|o0|r0 Δt? 5R|t0|o0|r0 Δt? 8i|t2|o0|r0 Δt? 2+3|t1|o0|r0 Δt? 4R|t1|o0|r0 Δt? 1+1+3|t4|o0|r1 Δt? 1+1+3|t2|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 5R|t3|o0|r0 Δt3.20 2+3|t2|o0|r0 Δt? 2+3|t1|o0|r0 Δt6.00 5R|t3|o0|r0 Δt? 2+3|t2|o0|r0 Δt? 3RP3|t1|o0|r0 Δt? 1+31|t2|o0|r0 Δt? 4RP5|t3|o0|r0 Δt? 2+3|t1|o0|r0
  Whale UNK: 2+3|t1|o0|r0
  Whale UNK: 5-NOISE|t4|o0|r0 Δt6.00 10R|t4|o0|r0 Δt? 2+3|t2|o0|r0 Δt6.00 2+3|t1|o0|r0
  Whale UNK: 5R|t0|o0|r0 Δt? 2+3|t2|o0|r0 Δt? 9R|t4|o0|r0 Δt? 7i|t0|o0|r0 Δt? 8R|t4|o0|r0 Δt? 6R|t4|o0|r0 Δt? 4R|t2|o0|r0 Δt? 4R|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t0|o0|r0 Δt? 8R|t4|o0|r0 Δt? 8R|t3|o0|r0 Δt? 7D|t3|o0|r0 Δt? 4R|t4|o0|r0 Δt? 1+1+3|t3|o0|r1 Δt? 6R|t4|o0|r0 Δt? 1+1+3|t3|o0|r0 Δt? 6-NOISE|t3|o0|r0 Δt? 7D|t3|o0|r0 Δt? 6iP2|t4|o0|r0 Δt? 6iP2|t4|o0|r0 Δt? 6-NOISE|t3|o0|r0 Δt? 7iP4|t4|o0|r0 Δt? 8i|t1|o0|r0 Δt? 1+1+3|t3|o0|r0 Δt? 8i|t1|o0|r0 Δt? 6R|t3|o0|r0 Δt? 8i|t1|o0|r0 Δt? 7i|t1|o0|r0 Δt? 8i|t1|o0|r0 Δt? 7i|t2|o0|r0 Δt? 10i|t2|o0|r0 Δt? 4R|t0|o0|r0 Δt? 9i|t1|o0|r0 Δt? 3P2|t1|o0|r0 Δt? 6i|t1|o0|r0 Δt? 6i|t1|o0|r0 Δt? 4R|t0|o0|r0 Δt? 6-NOISE|t3|o0|r0 Δt? 7i|t1|o0|r0 Δt? 4R|t2|o0|r0 Δt? 6iP2|t4|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 4R|t2|o0|r0 Δt? 6iP2|t4|o0|r0 Δt? 5P4|t4|o0|r0 Δt? 4R|t1|o0|r0 Δt? 5P4|t4|o0|r0 Δt? 1+1+3|t2|o1|r0 Δt? 4R|t0|o0|r0 Δt? 4R|t2|o0|r0
  Whale UNK: 4R|t1|o0|r0 Δt? 4R|t0|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4R|t1|o0|r0
  Whale UNK: 3D|t0|o0|r0 Δt? 7i|t1|o0|r0
```

**sample 3:**
```
  Whale UNK: 7RP1|t2|o0|r0 Δt? 5R|t1|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 10iP2|t3|o0|r0 Δt? 9i|t1|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 6iP3|t3|o0|r0 Δt? 9i|t1|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 7RP1|t2|o0|r0
  Whale UNK: 7RP1|t2|o0|r0 Δt? 7RP1|t2|o0|r0 Δt3.20 4R|t3|o0|r0 Δt? 7RP1|t2|o0|r0 Δt0.80 7RP1|t2|o0|r0
  Whale UNK: 5R|t4|o0|r0
  Whale UNK: 4R|t2|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 4R|t3|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 4R|t3|o0|r0 Δt? 3+1|t3|o0|r0 Δt? 4R|t3|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 4R|t3|o0|r0 Δt? 4R|t3|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 4R|t3|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt3.20 5R|t3|o0|r0 Δt? 4R|t3|o0|r0 Δt? 4R|t3|o0|r0 Δt? 7iP6|t4|o0|r0 Δt? 5R|t4|o0|r0
  Whale UNK: 1+1+3|t4|o0|r0 Δt? 1+31|t2|o0|r0 Δt? 4R|t3|o0|r0 Δt? 4R|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 2+3|t1|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt? 1+31|t2|o0|r0 Δt6.00 4R|t2|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt? 4R|t3|o0|r0 Δt? 5R|t0|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4P5|t3|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 2+3|t1|o0|r0
  Whale UNK: 1+1+3|t4|o0|r0 Δt? 4R|t2|o0|r0 Δt? 2+3|t1|o0|r0 Δt? 4R|t2|o0|r0 Δt? 5-NOISE|t4|o0|r0
  Whale UNK: 1+1+3|t4|o0|r0 Δt6.00 1+1+3|t3|o0|r0
  Whale UNK: 1+1+3|t3|o1|r0
  Whale UNK: 1+31|t1|o0|r0
  Whale UNK: 4R|t3|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4R|t3|o0|r0 Δt? 1+1+3|t3|o1|r0 Δt? 1+1+3|t3|o1|r0 Δt? 4R|t1|o0|r0 Δt? 4R|t3|o0|r0 Δt? 5P4|t4|o0|r0 Δt? 4R|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4R|t1|o0|r0 Δt6.00 4R|t1|o0|r0 Δt? 4R|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4R|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+1+3|t3|o1|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 8R|t3|o0|r0 Δt? 4R|t0|o0|r0 Δt? 8i|t2|o0|r0 Δt? 6i|t1|o0|r0 Δt? 4R|t0|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5-NOISE|t3|o0|r0 Δt? 4R|t0|o0|r0 Δt? 7i|t1|o0|r0 Δt? 8i|t1|o0|r0 Δt? 6iP3|t4|o0|r0 Δt? 6i|t0|o0|r0
```

## seed 1 — `hersh2022_pacific::26`
original length: 230; generating 111 new tokens after the K=8 prefix

**prefix (first 8 tokens, rendered from real CSV):**
```
  Whale UNK: 1+1+3|t2|o0|r0 Δt? 7R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 3D|t0|o0|r0 Δt? 7R|t4|o0|r0 Δt? 9R|t4|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 9R|t4|o0|r0
```

**original continuation (ground truth):**
```
  Whale UNK: 8R|t4|o0|r0 Δt? 6R|t3|o0|r0 Δt? 7R|t4|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 8R|t4|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 6R|t4|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 4R|t0|o0|r0 Δt? 5R|t0|o0|r0 Δt? 7R|t4|o0|r0 Δt? 5R|t3|o0|r0 Δt? 4R|t0|o0|r0 Δt? 8R|t4|o0|r0 Δt? 5R|t0|o0|r0 Δt? 8R|t4|o0|r0 Δt? 7R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 6R|t4|o0|r0 Δt? 9R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 7R|t4|o0|r0 Δt? 7R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 9R|t4|o0|r0 Δt? 7D|t3|o0|r0 Δt? 6R|t3|o0|r0 Δt? 7R|t4|o0|r0 Δt? 7R|t4|o0|r0 Δt? 4R|t2|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 6R|t3|o0|r0 Δt? 6R|t4|o0|r0 Δt? 6R|t4|o0|r0 Δt? 5R|t3|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 9R|t4|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 7D|t4|o0|r0 Δt? 5R|t2|o0|r0 Δt? 8R|t4|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 10RP1|t4|o0|r0 Δt? 5R|t3|o0|r0 Δt? 7R|t4|o0|r0 Δt? 6R|t4|o0|r0 Δt? 6R|t3|o0|r0 Δt? 8R|t4|o0|r0 Δt? 7R|t4|o0|r0 Δt? 6R|t3|o0|r0 Δt? 8R|t4|o0|r0 Δt? 7D|t4|o0|r0 Δt? 6R|t3|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 6R|t3|o0|r0 Δt? 6R|t3|o0|r0 Δt? 6R|t3|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 10R|t4|o0|r0 Δt? 7D|t3|o0|r0 Δt? 5R|t2|o0|r0 Δt? 7R|t4|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t2|o0|r0 Δt? 7D|t4|o0|r0 Δt? 5R|t4|o0|r0 Δt? 7R|t4|o0|r0 Δt? 4R|t2|o0|r0 Δt? 8R|t4|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t2|o0|r0 Δt? 6iP3|t4|o0|r0 Δt? 6R|t4|o0|r0 Δt? 7D|t4|o0|r0 Δt? 6-NOISE|t4|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 6R|t4|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 5R|t2|o0|r0 Δt? 5R|t2|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 7R|t4|o0|r0 Δt? 6R|t3|o0|r0 Δt? 6R|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 9R|t4|o0|r0 Δt? 6R|t3|o0|r0 Δt? 6R|t4|o0|r0 Δt? 5R|t2|o0|r0 Δt? 8RP1|t4|o0|r0 Δt? 7R|t4|o0|r0 Δt? 5R|t2|o0|r0 Δt? 5R|t3|o0|r0 Δt? 1+1+3|t1|o0|r0 Δt? 7R|t4|o0|r0 Δt? 7-NOISE|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 7R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 6i|t3|o0|r0 Δt? 6R|t3|o0|r0 Δt? 9R|t4|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 5R|t3|o0|r0 Δt? 9R|t4|o0|r0 Δt? 10R|t4|o0|r0 Δt? 5R|t2|o0|r0
```

**sample 1:**
```
  Whale UNK: 4R|t2|o0|r0 Δt? 5R|t0|o0|r0 Δt? 5R|t3|o0|r0 Δt? 4D|t1|o0|r0 Δt? 4R|t0|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 4R|t2|o0|r0 Δt? 8i|t1|o0|r0 Δt? 4R|t2|o0|r0 Δt? 4R|t2|o0|r0 Δt? 4R|t0|o0|r0 Δt? 9i|t1|o0|r0 Δt? 3D|t0|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 10i|t1|o0|r0 Δt? 8i|t1|o0|r0 Δt? 4R|t1|o0|r0 Δt? 4R|t3|o0|r0 Δt? 9-NOISE|t3|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 3D|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt? 4D|t1|o0|r0 Δt? 2+3|t1|o0|r0 Δt? 5R|t2|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 5R|t2|o0|r0 Δt6.00 5R|t2|o0|r0 Δt? 4RP3|t3|o0|r0
  Whale UNK: 3P1|t0|o0|r0 Δt? 4R|t2|o0|r1
  Whale UNK: 1+1+3|t2|o0|r0 Δt? 5R|t1|o0|r0 Δt? 3RP3|t1|o0|r0 Δt? 5-NOISE|t4|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 1+1+3|t2|o0|r1
  Whale UNK: 1+1+3|t2|o1|r0 Δt1.80 5R|t1|o0|r0
  Whale UNK: 5R|t1|o0|r0 Δt1.80 7i|t1|o0|r0 Δt? 5R|t0|o0|r0
  Whale UNK: 5R|t1|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 4R|t2|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 2+3|t1|o0|r0 Δt? 1+31|t1|o0|r0
  Whale UNK: 1+1+3|t2|o0|r1
  Whale UNK: 1+1+3|t2|o0|r1
  Whale UNK: 4R|t2|o0|r0 Δt? 1+1+3|t3|o0|r0
  Whale UNK: 4R|t2|o0|r0 Δt? 4R|t1|o0|r0
  Whale UNK: 5P4|t3|o0|r0 Δt? 1+1+3|t2|o1|r0 Δt? 2+3|t2|o0|r0 Δt0.80 6iP3|t3|o0|r0
  Whale UNK: 4R|t1|o0|r0 Δt? 7i|t2|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+1+3|t2|o1|r1
  Whale UNK: 3RP3|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt1.80 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0
  Whale UNK: 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4R|t0|o0|r0 Δt? 6iP1|t2|o0|r0 Δt? 3D|t1|o0|r0 Δt? 4R|t1|o0|r0 Δt? 4+1|t3|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 6iP2|t4|o0|r0 Δt? 4R|t2|o0|r0 Δt? 9i|t1|o0|r0 Δt? 1+31|t2|o0|r0
  Whale UNK: 4R|t1|o0|r0 Δt? 4R|t1|o0|r0 Δt? 4R|t0|o0|r0 Δt? 9i|t2|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 6iP2|t4|o0|r0 Δt? 7i|t0|o0|r0 Δt? 1+31|t1|o0|r0
  Whale UNK: 7i|t2|o0|r0 Δt? 4R|t3|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+1+3|t3|o0|r0 Δt? 8i|t0|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 5R|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4RP3|t3|o0|r0 Δt? 1+31|t1|o0|r0
```

**sample 2:**
```
  Whale UNK: 5R|t0|o0|r0 Δt? 6R|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 7i|t1|o0|r0 Δt? 4R|t2|o0|r0 Δt? 2+3|t2|o0|r0 Δt? 6-NOISE|t4|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4R|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 4R|t3|o0|r0 Δt? 8i|t1|o0|r0 Δt? 4R|t3|o0|r0 Δt? 4D|t1|o0|r0 Δt? 6i|t0|o0|r0 Δt? 2+1+1+1+1|t3|o0|r0 Δt? 6i|t0|o0|r0 Δt? 7i|t0|o0|r0 Δt? 7i|t0|o0|r0 Δt? 1+31|t2|o0|r0 Δt? 6-NOISE|t2|o0|r0 Δt? 7i|t1|o0|r0 Δt? 1+31|t2|o0|r0 Δt0.80 7i|t1|o0|r0 Δt? 1+31|t2|o0|r0 Δt? 3P2|t1|o0|r0 Δt? 3RP3|t1|o0|r0 Δt? 8i|t2|o0|r0 Δt? 9i|t1|o0|r0 Δt? 9i|t0|o0|r0 Δt? 5R|t0|o0|r0
  Whale UNK: 3P3|t1|o0|r0 Δt? 3D|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt? 6i|t1|o0|r0 Δt? 10i|t1|o0|r0 Δt? 7i|t0|o0|r0 Δt? 5R|t3|o0|r0 Δt? 9i|t1|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 10i|t1|o0|r0 Δt? 1+1+3|t3|o0|r0 Δt? 10i|t1|o0|r0 Δt6.00 5R|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt1.80 5R|t3|o0|r0 Δt? 9i|t1|o0|r0 Δt? 7R|t4|o0|r0 Δt? 10R|t4|o0|r0 Δt? 7D|t4|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t0|o0|r0 Δt? 8R|t4|o0|r0 Δt? 5P4|t3|o0|r0 Δt? 3D|t1|o0|r0
  Whale UNK: 10i|t1|o0|r0 Δt? 10i|t1|o0|r0 Δt? 5R|t3|o0|r0 Δt? 8R|t3|o0|r0 Δt? 4R|t3|o0|r0 Δt? 5P4|t4|o0|r0 Δt? 3RP3|t1|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 7R|t4|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t4|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t0|o0|r0 Δt? 7-NOISE|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 8i|t2|o0|r0 Δt? 4R|t3|o0|r0 Δt? 3+1|t2|o0|r0 Δt? 4P3|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 5R|t3|o0|r0 Δt? 8i|t2|o0|r0 Δt6.00 10i|t1|o0|r0 Δt? 10i|t1|o0|r0 Δt? 6i|t1|o0|r0 Δt? 5R|t0|o0|r0 Δt? 4R|t2|o0|r0 Δt? 3D|t0|o0|r0 Δt? 5R|t1|o0|r0 Δt? 10i|t1|o0|r0 Δt? 5R|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt? 4R|t2|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 9i|t1|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 5R|t3|o0|r0 Δt? 4R|t1|o0|r0 Δt? 4R|t1|o0|r0 Δt? 4R|t3|o0|r0
  Whale UNK: 1+1+3|t3|o1|r0
  Whale UNK: 9i|t1|o0|r0 Δt? 1+1+3|t2|o1|r0
```

**sample 3:**
```
  Whale UNK: 7R|t4|o0|r0 Δt? 9R|t4|o0|r0 Δt? 3D|t0|o0|r0 Δt? 5-NOISE|t3|o0|r0 Δt? 5R|t0|o0|r0 Δt? 3RP3|t1|o0|r0 Δt? 4+1|t4|o0|r0 Δt? 1+31|t1|o0|r0
  Whale UNK: 3P14|t3|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 1+1+3|t3|o0|r0 Δt? 4D|t1|o0|r0 Δt? 5R|t0|o0|r0 Δt? 1+1+3|t2|o1|r0 Δt? 1+31|t0|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt? 6i|t1|o0|r0 Δt? 4R|t0|o0|r0 Δt? 5R|t0|o0|r0 Δt? 5R|t0|o0|r0 Δt? 4R|t3|o0|r0 Δt? 4D|t1|o0|r0 Δt? 4R|t2|o0|r0 Δt? 1+1+3|t3|o1|r0 Δt? 4R|t3|o0|r0 Δt? 3+1|t2|o0|r0 Δt? 8i|t0|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 6iP1|t2|o0|r0 Δt? 1+31|t2|o0|r0 Δt? 6iP1|t2|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 6i|t0|o0|r0 Δt? 6-NOISE|t2|o0|r0 Δt? 4R|t0|o0|r0 Δt? 4R|t0|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4R|t3|o0|r0 Δt? 4R|t1|o0|r0 Δt? 1+31|t0|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 5P4|t4|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4RP3|t3|o0|r0 Δt? 1+31|t2|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t2|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t2|o0|r0 Δt? 4R|t0|o0|r0 Δt? 6-NOISE|t2|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 7RP2|t3|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 7i|t0|o0|r0 Δt? 4R|t0|o0|r0 Δt? 3P15|t3|o0|r0 Δt? 6iP2|t4|o0|r0 Δt? 7i|t1|o0|r0 Δt? 4R|t2|o0|r0 Δt? 7i|t1|o0|r0 Δt? 7i|t1|o0|r0 Δt? 7D|t4|o0|r0 Δt? 7i|t1|o0|r0 Δt? 7i|t1|o0|r0 Δt? 7i|t1|o0|r0 Δt? 9i|t2|o0|r0 Δt? 7i|t1|o0|r0 Δt? 8-NOISE|t1|o0|r0 Δt? 9i|t2|o0|r0 Δt? 5-NOISE|t4|o0|r0 Δt? 5R|t3|o0|r0
```

## seed 2 — `sharma2024_dswp::sw100a002_9490`
original length: 19; generating 111 new tokens after the K=8 prefix

**prefix (first 8 tokens, rendered from real CSV):**
```
  Whale photo:5981: 1+1+3|t4|o0|r0
  Whale local:2: 1+1+3|t4|o0|r0
  Whale photo:5981: 1+1+3|t4|o0|r0 Δt4.01 1+1+3|t4|o0|r0
  Whale local:2: 1+1+3|t4|o0|r0
  Whale photo:5981: 1+1+3|t4|o0|r0
  Whale local:2: 1+1+3|t4|o0|r0
  Whale photo:5981: 1+1+3|t4|o0|r0
```

**original continuation (ground truth):**
```
  Whale local:2: 1+1+3|t4|o0|r0
  Whale photo:5981: 1+1+3|t4|o0|r0
  Whale local:2: 1+1+3|t4|o0|r0
  Whale photo:5981: 1+1+3|t4|o0|r0
  Whale local:2: 1+1+3|t4|o0|r0
  Whale photo:5981: 1+1+3|t4|o0|r0
  Whale local:2: 1+1+3|t3|o0|r0
  Whale photo:5981: 1+1+3|t4|o0|r0
  Whale local:2: 1+1+3|t4|o0|r0
  Whale photo:5981: 1+1+3|t4|o0|r0
  Whale local:2: 1+1+3|t4|o0|r0
```

**sample 1:**
```
  Whale photo:5981: 1+1+3|t4|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt? 1+1+3|t3|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt? 4R|t3|o0|r0 Δt? 5R|t0|o0|r0
  Whale local:2: 6iP3|t4|o0|r0 Δt? 1+1+3|t3|o0|r0 Δt? 1+1+3|t3|o0|r0 Δt? 5R|t0|o0|r0 Δt? 2+3|t2|o0|r0 Δt? 4D|t1|o0|r0 Δt? 6iP3|t4|o0|r0 Δt? 4R|t0|o0|r0 Δt? 2+3|t1|o0|r0 Δt? 8R|t3|o0|r0 Δt? 4R|t3|o0|r0
  Whale photo:5981: 3RP3|t1|o0|r0 Δt? 1+31|t2|o0|r0 Δt? 3RP3|t1|o0|r0 Δt? 1+1+3|t3|o0|r0
  Whale local:2: 1+1+3|t3|o1|r0 Δt? 3RP3|t1|o0|r0 Δt? 6iP1|t2|o0|r0 Δt1.80 4R|t2|o0|r0 Δt0.80 1+1+3|t3|o0|r0 Δt? 1+1+3|t3|o0|r1
  Whale photo:5981: 1+1+3|t3|o0|r1 Δt? 3D|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt? 5R|t0|o0|r0 Δt? 9i|t1|o0|r0 Δt? 3D|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt? 5R|t2|o0|r0 Δt? 3D|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt? 6R|t4|o0|r0 Δt? 3D|t0|o0|r0 Δt? 4RP2|t2|o0|r0 Δt? 3D|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt? 3D|t0|o0|r0 Δt? 4R|t0|o0|r0 Δt? 4R|t2|o0|r0 Δt? 6i|t1|o0|r0 Δt? 9i|t1|o0|r0 Δt? 4R|t2|o0|r0 Δt? 3D|t0|o0|r0 Δt? 4R|t2|o0|r0 Δt? 7i|t1|o0|r0 Δt? 4R|t2|o0|r0 Δt? 4R|t2|o0|r0 Δt? 4R|t2|o0|r0
  Whale local:2: 4R|t2|o0|r0
  Whale photo:5981: 4R|t2|o0|r0 Δt? 3D|t0|o0|r0 Δt? 6iP3|t4|o0|r0 Δt? 4R|t2|o0|r0 Δt? 5-NOISE|t3|o0|r0 Δt? 4R|t2|o0|r0 Δt? 5R|t1|o1|r0 Δt? 5P4|t4|o0|r0 Δt? 9R|t4|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 6iP2|t4|o0|r0 Δt? 5R|t3|o0|r0 Δt? 6iP2|t4|o0|r0 Δt? 4R|t3|o0|r0 Δt? 1+31|t2|o0|r0 Δt? 1+31|t2|o0|r0 Δt? 6iP2|t4|o0|r0 Δt? 5P4|t4|o0|r0 Δt? 8R|t4|o0|r0 Δt? 10i|t2|o0|r0 Δt? 5-NOISE|t2|o0|r0 Δt? 1+31|t2|o0|r0 Δt? 5P4|t4|o0|r0 Δt? 4R|t0|o0|r0 Δt? 7i|t1|o0|r0 Δt? 5R|t3|o0|r0 Δt? 7i|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4R|t2|o0|r0
  Whale local:2: 6-NOISE|t4|o0|r0 Δt? 5P4|t4|o0|r0 Δt? 5R|t3|o0|r0 Δt? 7i|t1|o0|r0 Δt? 3D|t0|o0|r0 Δt? 3RP3|t1|o0|r0 Δt? 3+1|t3|o0|r0 Δt? 5R|t3|o0|r0
  Whale photo:5981: 4R|t2|o0|r0 Δt? 7i|t1|o0|r0 Δt? 3D|t0|o0|r0 Δt? 9i|t1|o0|r0 Δt? 1+1+3|t2|o0|r1 Δt? 5R|t0|o0|r0 Δt? 3D|t1|o0|r0 Δt? 8i|t1|o0|r0 Δt? 8i|t1|o0|r0 Δt? 8i|t2|o0|r0 Δt? 7-NOISE|t3|o0|r0 Δt? 5-NOISE|t4|o0|r0 Δt? 4R|t2|o0|r0
```

**sample 2:**
```
  Whale photo:5981: 1+1+3|t4|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt6.00 1+1+3|t4|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt6.00 4R|t4|o0|r0
  Whale local:2: 1+1+3|t4|o0|r0 Δt? 2+3|t1|o0|r0 Δt? 9i|t1|o0|r0 Δt? 1+1+3|t3|o0|r0
  Whale photo:5981: 1+1+3|t4|o0|r0 Δt? 5-NOISE|t4|o0|r0
  Whale local:2: 1+1+3|t4|o0|r0
  Whale photo:5981: 1+1+3|t3|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt? 1+1+3|t3|o0|r0
  Whale local:2: 1+1+3|t3|o0|r0
  Whale photo:5981: 1+1+3|t4|o0|r0 Δt6.00 7-NOISE|t4|o0|r0
  Whale local:2: 1+1+3|t3|o0|r0 Δt? 5R|t0|o0|r0 Δt? 5R|t0|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 6iP2|t4|o0|r0 Δt? 8i|t1|o0|r0 Δt? 5R|t0|o0|r0 Δt? 7i|t0|o0|r0 Δt? 7i|t0|o0|r0 Δt? 4R|t1|o0|r0 Δt? 6iP2|t4|o0|r0 Δt? 7i|t1|o0|r0 Δt? 7i|t2|o0|r0 Δt0.80 8i|t0|o0|r0 Δt0.20 4R|t3|o0|r0 Δt? 6-NOISE|t4|o0|r0 Δt? 6-NOISE|t4|o0|r0 Δt? 1+1+3|t3|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4R|t2|o0|r0 Δt? 1+1+3|t3|o1|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4R|t0|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 7RP2|t3|o0|r0 Δt? 2+3|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+1+3|t2|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt6.00 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 3RP3|t1|o0|r0 Δt? 2+3|t1|o0|r0 Δt? 1+31|t0|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0
  Whale photo:5981: 1+31|t1|o0|r0 Δt? 4R|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+1+3|t3|o1|r0 Δt? 1+1+3|t2|o0|r0 Δt? 4R|t2|o0|r0
  Whale local:2: 4R|t1|o0|r0 Δt? 5R|t4|o0|r0 Δt? 4R|t1|o0|r0 Δt? 1+1+3|t3|o0|r0 Δt6.00 4R|t2|o0|r0 Δt? 9i|t1|o0|r0 Δt? 6i|t1|o0|r0 Δt? 5R|t1|o1|r0 Δt? 4R|t1|o0|r0 Δt? 1+1+3|t3|o0|r0 Δt1.80 5R|t0|o0|r0 Δt? 10i|t2|o0|r0 Δt? 4R|t0|o0|r0 Δt? 7i|t1|o0|r0 Δt? 4R|t1|o0|r0 Δt? 1+1+3|t3|o0|r0
  Whale photo:5981: 4R|t1|o0|r0 Δt? 1+1+3|t3|o0|r0 Δt? 7RP1|t2|o0|r0 Δt6.00 6iP1|t2|o0|r0 Δt? 6i|t1|o0|r0
  Whale local:2: 7i|t1|o0|r0 Δt0.20 7i|t1|o0|r0 Δt? 6i|t1|o0|r0 Δt3.20 6i|t1|o0|r0 Δt? 6i|t1|o0|r0
  Whale photo:5981: 10i|t2|o0|r0
```

**sample 3:**
```
  Whale photo:5981: 1+1+3|t4|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt6.00 8R|t4|o0|r0 Δt6.00 1+1+3|t3|o0|r1 Δt? 1+1+3|t4|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt? 1+1+3|t4|o0|r0
  Whale local:2: 1+1+3|t4|o0|r0 Δt? 4R|t3|o0|r0 Δt? 1+1+3|t4|o0|r1 Δt? 1+1+3|t4|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt? 1+1+3|t3|o0|r0
  Whale photo:5981: 1+1+3|t4|o0|r0 Δt6.00 1+1+3|t4|o0|r0 Δt? 6i|t1|o0|r0
  Whale local:2: 1+1+3|t4|o0|r0 Δt? 1+1+3|t4|o0|r0 Δt? 6R|t4|o0|r0 Δt? 1+1+3|t2|o0|r0
  Whale photo:5981: 2+3|t1|o0|r0 Δt? 7RP1|t2|o0|r0 Δt? 6R|t4|o0|r0
  Whale local:2: 9R|t4|o0|r0 Δt? 5R|t3|o0|r0
  Whale photo:5981: 2+1+1+1+1|t3|o0|r0 Δt6.00 5-NOISE|t1|o0|r0 Δt? 4R|t2|o0|r0
  Whale local:2: 4R|t3|o0|r0
  Whale photo:5981: 4R|t0|o0|r0 Δt? 4P5|t3|o0|r0 Δt? 6iP1|t2|o0|r0 Δt? 9i|t1|o0|r0 Δt? 4R|t3|o0|r0 Δt? 2+1+1+1+1|t3|o0|r0 Δt? 4R|t3|o0|r0 Δt? 5R|t0|o0|r0 Δt? 3R|t0|o0|r0 Δt? 7i|t2|o0|r0 Δt? 6-NOISE|t3|o0|r0 Δt? 7i|t1|o0|r0 Δt? 4R|t3|o0|r0 Δt? 4R|t0|o0|r0 Δt? 7i|t1|o0|r0 Δt? 5R|t4|o0|r0 Δt? 6i|t1|o0|r0 Δt? 4R|t3|o0|r0 Δt? 5-NOISE|t4|o0|r0 Δt? 4R|t3|o0|r0 Δt? 4R|t1|o0|r0 Δt? 3P2|t1|o0|r0 Δt? 1+1+3|t3|o0|r0 Δt? 5P4|t3|o0|r0 Δt? 4R|t3|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 7i|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4R|t0|o0|r0 Δt? 4R|t0|o0|r0 Δt? 4D|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4+1|t3|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4+1|t3|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt6.00 1+31|t1|o0|r0 Δt? 3P1|t1|o0|r0 Δt? 7i|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4+1|t3|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 2+3|t2|o0|r0 Δt? 5R|t0|o0|r0 Δt? 4R|t0|o0|r0 Δt? 2+3|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4+1|t3|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 1+31|t1|o0|r0 Δt? 4R|t1|o0|r0 Δt? 1+31|t1|o0|r0
```
