---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
language:
- en
- zh
tags:
- deepseek
- mixture-of-experts
- vector-quantization
- gguf
- cuda
- dgx-spark
- on-device
pipeline_tag: text-generation
---

# YoungAi — DeepSeek V4.1 Flash on one DGX Spark

**English** · [中文](README.zh-CN.md)

> One 128 GB box. Three files. A model whose official checkpoint is 510 GB.
>
> **① a 113.6 GB general base → ② a ~40 MB domain sidecar → ③ a post-training file re-solved every night.**

ds4 is a native C/CUDA inference engine plus an offline toolchain that runs DeepSeek V4.1 Flash on a
single NVIDIA DGX Spark (GB10, 128 GB unified memory). Everything described here is our own work:
the vector-quantization format, the solvers that produce the sidecar and the post-training file,
the CUDA kernels, and the rulers we judge all of it with. The model architecture is DeepSeek's and
is not re-explained here — read the official release for that.

**How to read this page.** It unfolds in five steps; stop wherever you have what you need.

1. [The numbers](#1-the-numbers) — what runs, how fast, how close to the original.
2. [Three convictions](#2-three-convictions) — why the system is shaped this way.
3. [The architecture](#3-the-architecture-base--sidecar--post-training-file) — three files and how they stack.
4. [The algorithms](#4-the-algorithms) — each piece goes *one sentence → intuition → math → engineering → evidence*.
5. [Results](#5-results), [what did not work](#6-what-did-not-work), [how to run it](#7-run-it), [limits](#8-honest-status-and-limits).

---

## 1. The numbers

One DGX Spark, ① base + ② the sidecar of the domain being measured.

| | |
|---|---|
| Resident in memory | 113.6 GB model + ~40 MB sidecar (≈ 1.6 bits per weight over the whole file) |
| Left on SSD, untouched | the model's 203 GB n-gram memory tables, original FP8, ~6 KB read per token |
| **Restore rate Σmin** — finance / code / law / medicine / science | **0.745 / 0.803 / 0.760 / 0.739 / 0.753** |
| **Same top-1 as the original** — finance / code / law / medicine / science | **74.6% / 82.3% / 76.4% / 72.8% / 74.9%** |
| General English (WikiText-2) with any one of the five sidecars | Σmin **0.787–0.801**, Same top-1 **80.7–82.8%** (bare base 0.769 / 78.5%) |
| **Prefill**, 12.5k-token prompt | **489 tokens/s** |
| **Decode**, plain greedy | **30.7 tokens/s** at short context · 29 tokens/s at 14k context |
| **Decode**, speculative (on by default at temperature 0) | **43.0 tokens/s** on a real 14k-token agent request · **37.1 tokens/s** on a request never used for tuning |
| Context | 1M tokens, taken from the model's own metadata; the full 1M KV cache is 0.89 GB |

**Restore rate** = Σ_v min(p_original(v), p_ours(v)), averaged over positions: the share of next-token
probability mass on which our model and the original agree. 1.0 means identical distributions.
All rulers are defined in [§4.5](#45-how-we-measure).

---

## 2. Three convictions

Each conviction below is stated first, then backed by what we measured, then tied to what we built.

### 2.1 Every domain hides a strongly correlated regularity — find it, and a few megabytes restore a lot

**Claim.** From the point of view of one domain, quantization damage is not uniform noise. Text from
one domain exercises a narrow set of directions, and the error along those directions can be learned
from a few thousand tokens. Fixing it restores more capability per byte than adding bits to the
weights ever could.

**What we measured.**

- **Five domains, one recipe.** Finance, code, law, medicine and science each get a sidecar from the
  same solver. On its own held-out text each sidecar lifts Same top-1 by **+2.8 to +3.7 pp** and cuts
  mean KL by **16–28%** (tables in [§5.1](#51-quality)).
- **No harm to general text.** On English Wikipedia every one of the five sidecars raises Same top-1
  from 78.5% to **80.7–82.8%** and Σmin from 0.769 to **0.787–0.801**.
- **Leverage.** +3 pp is what roughly 0.15 extra bits per weight would buy on the bit-width curve —
  about 11 GB of extra expert storage. The sidecar that delivers it is about **40 MB**.

**What we built.** A base that never sees domain data, plus a small per-domain sidecar
([§4.2](#42-the-domain-sidecar-反修)).

### 2.2 Scaling laws are real, and knowledge is spread like a hash

**Claim.** Capability follows the bytes you keep. You cannot cheat the scaling law by deciding that
some experts, layers or matrices "matter more" — the weights show no such structure, and routing
behaves like a hash of the token. Hardware has diminishing returns too: a second box buys less than
it costs.

**What we measured.**

- **The weights have no favourites.** On the real checkpoint, the 384 experts of a layer have the same
  relative quantization error to within **1.9%**; expert energy differs by at most **1.25×**; layers
  differ by **3%** in Σw²; the three matrices of an expert split **33.5 / 33.3 / 33.2**. Moving one bit
  from group B to group A pays only when A is at least **1.189×** more important, because each extra
  bit per 8 weights multiplies the error by 1/1.189 (measured 0.201 / 0.171 / 0.143 at 11 / 12 / 13 bits).
  Nothing crosses that line, so the optimal "dynamic" allocation is simply uniform; implemented
  per-layer and per-expert allocation measures no gain.
- **Routing is hash-like.** After quantization, the share of tokens whose 6 chosen experts exactly match
  the original drops to 1.6–28% in the middle and deep layers, and a per-expert systematic bias explains
  only 3–22% of the flips — the rest is per token. Consecutive tokens barely share experts: 1, 2, 3, 4,
  5 tokens touch 6, 10.0, 13.3, 17.4, 21.7 distinct experts.
- **Pruning is a domain decision in disguise.** The only 20–80× skew we found is how often experts are
  routed *on a given corpus*: change the corpus and the skew changes. Fitting the model into 100 GB by
  pruning would have meant keeping 130 of 384 experts per layer.
- **More bits, diminishing returns** (experts only, English Wikipedia ruler): 1.0 / 1.5 / 2.0 / 2.5 / 3.0
  bits per weight → Σmin 0.445 / 0.746 / 0.848 / 0.900 / 0.927. The last half bit buys +0.027.
- **More boxes, diminishing returns.** Decoding is bound by the bytes read per token. Splitting layers
  across machines adds a network hop to every token (upstream ds4 measured −19% on two Macs); tensor
  parallelism halves the bytes, but our two-machine test reached only 1.25× because the all-reduce ate
  half the gain. For scale: a community two-Spark setup runs the smaller V4 Flash (284B) in FP8 with
  TP=2 and speculative decoding at ~41 tokens/s; one Spark here runs V4.1 at 37–43.

**What we built.** One box. Every byte spent uniformly. Everything else comes from the sidecar and
from post-training, not from topology.

### 2.3 Real work needs post-training — and it has to stay on your machine

**Claim.** On real business work — a real codebase, a real trading desk — open models are not good
enough out of the box, and quantization fidelity cannot fix that: a perfect copy of the original
inherits the original's mistakes. The model has to learn from its own deployment, and that data
(positions, reviews, source code) should never leave the machine.

**What we measured.** Test bed: a trading-agent pipeline served by ds4 (market outlook, news merging,
stock picking, CFO decision reports; prompts of 9k–132k tokens). **12 of 12** target prices the model set
were above the next day's actual high. On one market-outlook request the model concluded "market up";
that day 1,891 stocks rose and 3,563 fell.

**What we built.** A third file, solved on the machine from the day's own requests and stacked on top
without touching ① or ② ([§4.3](#43-the-post-training-file)). The mechanism works; it does not yet
generalize across days — see [§8](#8-honest-status-and-limits).

---

## 3. The architecture: base + sidecar + post-training file

```
                 what it is                          how often it changes
 ① base GGUF     113.6 GB, general, zero corpus      quantized once from the official weights
 ② sidecar       ~40 MB per domain   (--zchain)      solved once per domain, then frozen
 ③ post-train    same format as ②    (--posttrain)   re-solved nightly; delete it to roll back
 ──────────────────────────────────────────────────────────────────────────────────────────
 each routed-expert weight row:   g_eff = g_base × s_sidecar × s_posttrain
 each router:                     selection score += Δb_sidecar
```

- **① alone is a complete model.** ② and ③ only rescale rows that already exist and nudge the router.
  They add no layers; the kernels are the same with or without them.
- **Near-zero runtime cost.** ② adds ~0.6 MB of reads per decoded token, about 0.01% of what the base
  reads.
- **Pairs are fingerprinted.** ③ stores a hash of the ② it was solved against (`base.fnv`); the engine
  refuses a mismatched pair. Without that check a stale ③ would run fine and quietly produce wrong
  numbers.

```sh
./bin/ds4-server --cuda -m base.gguf --zchain sidecar_dir/ [--posttrain posttrain_dir/] --mem-budget-mb 110000
```

---

## 4. The algorithms

### 4.1 The base: 8-dimensional vector quantization with shared codebooks

**In one sentence.** Every 8 consecutive weights of a routed-expert row become one 12-bit (or 13-bit)
index into a codebook, times one gain per row.

**Intuition.** The official experts are FP4 with one scale per 32 weights (4.25 bits/weight). A scalar
format below that still has to pay for those scales — 0.25 bits/weight on its own. Vector quantization
codes 8 weights jointly, which beats scalar rate–distortion at the same budget, and it needs only one
gain per row. That is enough here because inside a row the official scales vary only 2× on average and
4× at worst (measured on layers 0, 20 and 39).

**The format.**

```
w[r, 8j : 8j+8]  ≈  g_r · C[ idx[r, j] ]        C = 4096 × 8 (12-bit)  or  8192 × 8 (13-bit)
bits per weight  =  12/8 = 1.5   or   13/8 = 1.625,   plus one f16 gain per row (≈ 0.003)
```

**Training the codebook.**

1. Normalize every row by its RMS, `g_r`.
2. Run Lloyd iterations on the GPU. The assignment step is fused: the codebook streams through shared
   memory in 512-entry chunks and each thread keeps its running argmin in registers, so the distance
   matrix never exists in memory (a naive `G = V·Cᵀ` with cuBLAS moved 144 GB per expert). The vector
   width is a template parameter so the 8 values stay in registers — 3× faster than a runtime width.
3. Deterministic by construction: equidistant initial sampling (no random source), a fixed number of
   rounds, double-precision atomics in the update. Same input, same bytes.
4. The final assignment uses the codebook and gains already rounded to their stored precision, so the
   error the quantizer reports is exactly the error the engine will see.

**One codebook per layer, shared by all 384 experts and all three matrices.** Because the experts are
statistically the same shape (relative error equal within 1.9%), a per-layer codebook trained on a pool
of ≥ 1,000 vectors per codeword costs almost nothing: layer 20 error 0.17062 vs 0.17072 with one
codebook per expert (slightly *better*), layer 39 +0.33%, +0.15% on average. It removes 15,744
codebooks — **3.09 GB**. The pool size matters: a pool 4× thinner lost 0.96% on 13-bit layers.

**FP8 (E4M3) codewords.** Storing codewords as E4M3 costs +0.12% error and halves lookup bytes. The
decoder must use the hardware conversion (`cvt.rn.f16x2.e4m3x2`, sm_89+); a scalar converter is
bit-exact but 26% slower in decode and 20% slower in prefill.

**Where the saved 3.09 GB goes: the 14 shallowest layers move from 12 to 13 bits** (error −16.1%).
The error curve is convex (removing a bit costs 5.3 pp, adding one gains 2.6–3.7 pp), so extra bytes
must be spread thin rather than stacked: 14 layers at 13 bits remove more total error than 7 layers at
14 bits, and shallow layers were measured to be worth 1.2–1.4× more per GB than deep ones.

**The "12 + 1" bit-plane layout.** A straight 13-bit stream makes each 256-index block 104 words = 3.25
cache lines, so every 128-byte load straddles two lines. On GB10, loads that don't fill whole 128-byte
lines fall from ~219 GB/s to ~142 GB/s (measured with our `mem_ceiling` probe). So 13-bit layers keep a
12-bit main stream with exactly the 12-bit geometry (one block = 3 full lines) plus a separate 1-bit
plane: one extra 32-bit load per block and one warp shuffle per round.

**Failing loudly.** Version-3 payloads carry a new magic (`DQV3`) and the loader whitelists format
versions. An engine that does not know a format refuses the file at load; one that predates the
whitelist outputs all-zero experts from the first sentence. Neither can produce plausible garbage.

**Zero corpus.** The base is quantized from the weights alone. Finance-weighted calibration leaves
finance flat and costs general text 2.4 pp: domain knowledge belongs in ②, not in ①.

**What the 113.56 GB is made of.**

| Part | Format | GB |
|---|---|---:|
| Routed-expert indices | VQ-8, 26 layers × 12 bit + 14 layers × 13 bit | 104.90 |
| Row gains | f16 | 0.30 |
| Codebooks | 40 × E4M3 | 0.003 |
| Attention, shared experts, output head | q4_K | 4.96 |
| Speculative draft towers | VQ-8, 12 bit | 2.56 |
| Everything else | as shipped | 0.85 |

**Evidence.** Against a same-size baseline (one codebook per expert, all layers 12-bit): Same top-1
**+1.9 pp** (finance) and **+2.5 pp** (English); the worst 5% of positions gain most (Σmin p5 +22% /
+83%). On English, the base alone beats that baseline *with* its sidecar.

### 4.2 The domain sidecar (反修)

We call the solve **反修** ("repairing backwards") and its product the **amplifier** (放大器); the
sidecar is a directory of amplifier files.

**In one sentence.** For every expert, re-solve one multiplicative gain per output channel of its down
projection, plus one router bias per expert, so that each quantized MoE block reproduces the original
block's output on domain text.

**Why gains, not an additive correction.** A correction that is a linear function of the block input,
such as a low-rank `y += B·A·x`, has ≈ 0 first-order effect at the model's output: it removes 9% of
held-out layer error and still makes end-to-end metrics worse (confirmed by a step-size scan). A
per-expert gain injects `Δ = Σ_e rw_e · (s_e − 1) ⊙ y_e`, which depends on which experts the router
picked and on each expert's own hidden state — *not* a linear function of x — and it does move the output.

**The objective, per layer.**

```
y_i   = Σ_k rw_ik · ( s_{e_ik} ⊙ ye_ik ) + ysh_i          # the engine's own MoE sum
        ye = expert down-projection output before the routing weight, ysh = shared expert

min_s   Σ_i ‖ y_i − y_i^orig ‖²   +   λ · Σ_d  d̄_d · Σ_e ( s_e[d] − 1 )²
```

- `y^orig` is the original model's output of the same block at the same token, from a teacher fixture:
  DeepSeek's code at full precision, 40 layers × every fitting token, written once per fitting corpus
  (14 GB for 8,192 tokens).
- The gains are diagonal, so the 5,120 output channels are independent; each is a least-squares problem
  with 384 unknowns.
- It is solved by per-expert **Gauss–Seidel** (3 sweeps). Each expert touches only the ~128 rows routed
  to it and updates the residual immediately — no 384 × 384 systems are ever formed.
- The ridge is scaled by `d̄_d`, the **mean energy of all experts on channel d**. Using each expert's own
  energy instead let rarely-routed experts run to extremes: layer 0 went train +32.7% / held-out −2685%.
  With the shared scale, the same layer is +27.0% held-out.

**Fitting on the deployment path, one layer after another.**

- `ye` and `ysh` come from hooks inside the engine's own prefill kernels; nothing is recomputed in a
  second implementation. Every layer is checked: `round_bf16(Σ rw·ye + ysh)` equals the engine's block
  output bit for bit on 99.996% of values, or the run stops.
- Layers are solved 0 → 39 in a single pass. Layer k is fitted with layers < k already mounted, so it
  sees exactly the upstream it will see in deployment.
- Per layer: solve the router bias Δb, mount it, recapture, then solve the gains.
- `λ ∈ {0.01, 0.1, 1, 10, 100, 1000}` is chosen on held-out rows only. Held-out is stratified by source
  window (every 4th 128-token window, a quarter of the tokens), so every sub-source appears on both sides.
- **A layer is mounted only if its held-out gain exceeds 0.5%.** This is a significance gate, not a size
  limit: picking the best of six λ values produces a small positive number even on pure noise, and in a
  sequential chain one noise layer corrupts every layer after it. End to end, mounting every layer with a
  positive gain is worse on all four metrics.

**FP4 lattice-direct solve.** Gains are stored in FP4 (E2M1, one scale per 32). Solving in float and
rounding afterwards would throw away 11% of the correction, because E2M1 has only 8 magnitudes. So each
Gauss–Seidel step becomes three: solve one expert → project `s − 1` onto the FP4 lattice and emit the
17-byte blocks directly → update the residual with the *lattice* value. Expert e's rounding error is
absorbed by the experts solved after it — GPTQ-style error feedback, applied to one-dimensional gains.
Layer 0 loses 0.27 pp of held-out gain instead of 11%; the sidecar shrank **291 MB → 38.6 MB** with no
metric regressing. The file written *is* the lattice codes, verified by decoding with the engine's own
decoder before it is kept.

**The router bias Δb.** For each layer and expert, accumulate the selection-score gap between the
original's chosen set and ours: if the original picked e and we did not, add `thr − v_e`; if we picked e
and the original did not, subtract `v_e − thr` (`thr` = our K-th selection score). Average, arm an
expert only after ≥ 8 events, scale by `α ∈ {0.5, 1, 1.5, 2.5}` chosen on held-out same-set rate, and
write the layer only if held-out strictly improves. Δb is added to the selection score only; mixing
weights are untouched. The original routing is recomputed from the fixture's full-precision inputs with
the model's own bf16 router; the recomputation must match the engine's choice on ≥ 99% of rows.

**Cost.** Same solver, same gates, one run per domain on the Spark:

| Sidecar | Fitting corpus | Gains / router bias | Size | Teacher fixture + 40-layer solve |
|---|---|---|---:|---|
| finance | 8,192 tokens, five Chinese-finance sources | 39 / 27 layers | 40.8 MB | 27 + 73 min |
| code | 15,360 tokens, 30 programming languages × 512 | 39 / 29 layers | 40.8 MB | 31 + 124 min |
| law | 15,360 tokens, 6 sources × 2,560 (3 Chinese, 3 English) | 40 / 24 layers | 41.8 MB | 31 + 104 min |
| medicine | 15,360 tokens, 6 sources × 2,560 (3 Chinese, 3 English) | 40 / 29 layers | 41.8 MB | 32 + 115 min |
| science | 15,360 tokens, 6 sources × 2,560 (3 Chinese, 3 English) | 40 / 29 layers | 41.8 MB | 32 + 104 min |

**Evidence** (Same top-1, paired runs; full tables in [§5.1](#51-quality)). Finance: gains 71.7 → 73.9%;
router bias on top, gains held fixed, +0.24 pp and KLD −2.5% (English +0.78 pp, KLD −3.8%). With each
domain's own sidecar: code 78.6 → 82.3%, law 72.9 → 76.4%, medicine 69.1 → 72.8%, science 71.7 → 74.9%.

### 4.3 The post-training file

**In one sentence.** Turn "at this position the model should have written *a*, not *b*" into a linear
equation on last-layer expert gains, solve all such equations together while pinning everything else,
and store the result as a third gain table.

**Why not ordinary fine-tuning gradients.** With the SFT loss gradient as the target the gains do not
move (held-out +0.04%). The loss gradient is a dense direction set by the target token's embedding,
while per-expert gains can only reweight expert outputs channel by channel: the two are nearly
orthogonal. What decides whether a decision flips is one scalar — the margin between two logits —
and that is exactly linear in the last MoE layer's output.

**The formulation.**

- **Decision point i:** a position where the right token `a` should beat the strongest *other* token `b`.
  (Not "the token the wrong answer wrote": in 81% of cases that token is already beaten and the argmax
  is a third token.)
- **Margin** `m_i = ℓ_a − ℓ_b`. Holding the output RMSNorm factor `inv_i` fixed, a change `Δ` of the last
  layer's output moves the margin by `α_i · inv_i · Σ_d γ_d (W_a − W_b)_d · Δ_d` (γ = norm weight,
  W = output head), and `Δ` is linear in the gain table.
- **Solve** `r_i · s = τ − m⁰_i` for all decision points (`r_i` = the coefficient row from the
  expression above, τ = target margin), plus **constraint rows** (every other position
  keeps its top-1/top-2 margin, weight ρ), plus a ridge λ, with matrix-free conjugate gradient and a
  Jacobi preconditioner — 1.97M unknowns, seconds.
- **FP4-aware:** quantize the solution to the lattice, re-predict with the stored values, put decision
  points that fell back below τ into an active set, solve again.
- **Capture on the deployment path.** Margins must be measured where the decision is actually made:
  prompt through prefill, generated text through the decode path (`--score-split P`). A solve captured on
  the prefill path alone flips the decision when scoring but not in generation.
- **Safety:** ③ is its own directory, multiplied with ② at load, fingerprinted against ②; candidates
  that fail a gate are moved to `rejected/`, never deleted.

**Evidence and status** — the least finished part; see [§8](#8-honest-status-and-limits). On the
training requests, decision points flip **55% → 88%** (64 / 73) while **99.66%** of 3,268 other
positions keep the same top-1; predicted margins correlate **0.9995** with a real forward. In free
generation the trained decision token flips, but the model can reason its way back to the original
conclusion ~6,000 characters later: one token moves one sentence, not a chain of reasoning.

### 4.4 The engine

**In one sentence.** Every kernel on the hot path is written for this format on GB10, and every
speed-up must leave temperature-0 output byte-identical.

**The wall.** Decoding is memory-bound: each token reads ~6.3 GB. At GB10's measured ~235 GB/s that is
26.8 ms (37 tokens/s). We run at 32.5 ms — **82% of the wall**.

| Per decoded token | Bytes | Kernel reach |
|---|---:|---:|
| Output head (q4_K) | 372 MB | ~249 GB/s |
| Attention and shared-expert projections (q4_K) | ~3.6 GB | 193–210 GB/s |
| Routed experts (VQ, 6 of 384 per layer) | 1.64 GB | ~155 GB/s |
| n-gram memory projection (FP8) | 314 MB | ~226 GB/s |
| Router and small mixing matrices | 314 MB | 132–136 GB/s |

**Decode.**

- **VQ expert kernel.** Persistent. Each layer's codebook is loaded into shared memory once per SM. Index
  streams are read as 384-byte blocks (3 full 128-byte lines) and codes are handed out with warp
  shuffles; consecutive rows stream back to back. At load time every expert payload is shifted so its
  index stream starts on a 128-byte boundary (device copy only, file untouched): 145–155 → 124–129 µs
  per layer.
- **q4_K GEMV** (the other ~60% of bytes). Two shapes: *stage* (a CTA moves whole lines into shared
  memory before computing) and *pipe* (persistent, cp.async double buffering, for 4–12 KB groups).
  22.9 → 18.8 ms per token.
- **Whole-step CUDA graph.** One capture per position bucket; the token position lives in a device slot,
  so replay never re-captures. Scratch buffers are grown *before* capture: an allocation inside capture
  invalidates it, and without graphs a token costs 43.9 ms.
- **Programmatic dependent launch.** 1,391 kernel edges: each kernel prefetches its constant weights,
  then waits for its producer.
- **Side stream.** The attention KV branch and the shared expert run in parallel with the main chain.
- **Long context.** The sparse-attention selection kernels were rewritten (1,024-thread top-k, batched
  reads, 4 groups per warp): at 51k context the extra cost fell 4.6 → 2.8 ms per token.

**Prefill.** Experts run on bf16 tensor cores (`mma.m16n8k16`). E4M3 codewords convert to bf16
exactly and activations already sit on the bf16 grid, so every product equals the scalar path's. Long
accumulation inside the tensor core drops low bits (3.5% of outputs off by one ulp), so each k16 slice is
accumulated from zero and added with a separate FADD (0.5%). 12.5k-token prompt: **216.5 → 489 tokens/s**.
The quality change (Σmin 0.7447 → 0.7430) is the same size as merely reordering the float reduction
(0.7439): rounding noise.

**Speculative decoding.**

- The model's three draft towers are quantized with the same VQ (2.56 GB).
- The verify batch (1 + k rows) has its own kernels: GEMVs on stage/pipe with several rows, and a
  persistent expert kernel over (unique expert × rows) that reads each chosen expert's bits once.
- **Scheduler.** The draft head reports a confidence `c_j` per position. With survival
  `a_j = Π_{i≤j} σ(c_i)`, pick

  ```
  k* = argmax_k  (1 + Σ_{j≤k} a_j) / (c_draft + c_v1 + c_tok · (1 + k))
  ```

  with costs in units of one plain decode step (0.273, 1.067, 0.303 — re-measured whenever a kernel
  changes). If no k beats plain decoding, the round skips drafting. The scheduler takes no wall-clock
  input: at temperature 0, output must not depend on how busy the machine is.
- **Guarantee:** speculative output is byte-identical to plain greedy output, checked on every change.
  On by default at temperature 0; a request that asks for sampling runs plain decode automatically.
- **Why it isn't higher:** each extra verified token touches ~3.9 new experts (hash-like routing,
  [§2.2](#22-scaling-laws-are-real-and-knowledge-is-spread-like-a-hash)), so verifying k + 1 tokens costs
  far more than verifying one. The verify batch runs at 57% of its own byte wall; that is the next lever.

**Memory.** Weights are mmap-backed. Per-request state grows with the positions actually used: the 1M
KV itself is 0.89 GB, and the per-forward scratch grows by doubling instead of being sized for 1M up
front (2.5 GB → 160 MB for a 40k-token request). There is no context knob — the bound comes from the
GGUF metadata and `--ctx` is rejected.

### 4.5 How we measure

**In one sentence.** The original model, run with DeepSeek's own code at full precision, is the only
judge, and our engine is scored on the same path it serves.

- **Teacher.** DeepSeek's official PyTorch inference code with the original FP4/FP8 weights streamed
  layer by layer from SSD (510 GB does not fit in memory). Teacher logits are cached per (text, length).
- **Student.** The engine's scoring path (`--score-ids`), same file format, one comparator
  (`anchor_metrics`) for everything.
- **Five numbers.** Same top-1; Σmin (mean, median, p5); mean KL(original ‖ ours); PPL ratio. Σmin and KL
  are primary. Same top-1 alone is lenient — easy text hides damage (the same file can read 0.90 on easy
  code and 0.52 on hard text).
- **Disjoint slices.** Each domain's text is split into a fitting slice and a judging slice that never mix;
  a source (a document, a code repository) lands on one side only.

  | Ruler | Judging slice | Original model PPL |
  |---|---|---:|
  | finance | 8,192 tokens, five Chinese-finance sources | 6.978 |
  | code | 15,360 tokens: 30 languages × 512 from GitHub (`codeparrot/github-code-clean`), license headers stripped | 4.766 |
  | law | 15,360 tokens, 6 × 2,560: Chinese statutes, criminal-case facts (CAIL2018), bar-exam questions (JEC-QA); US Supreme Court opinions, EU legislation, contract clauses (LexGLUE) | 4.865 |
  | medicine | 15,360 tokens, 6 × 2,560: Chinese medical encyclopedia Q&A, real doctor consultations (cMedQA2), licensing-exam questions with explanations (CMExam); PubMed abstracts, PMC case reports, USMLE questions (MedQA) | 8.625 |
  | science | 15,360 tokens, 6 × 2,560: Chinese journal abstracts, STEM and humanities / social science (CSL), college STEM questions (C-Eval); arXiv LaTeX source, S2ORC full papers (peS2o), MMLU-Pro STEM questions | 8.770 |
  | English | WikiText-2, 512 tokens | 1.657 |

- **Paired, one variable at a time.** Comparisons use the same binary in the same run.
- **Gates beyond the five numbers.**
  - Temperature-0 byte identity: direct launch == CUDA graph == replay, and speculative == plain.
  - Format changes are also checked on the decode path. The five numbers are computed on the prefill path
    and cannot see a decode-kernel fault: a 13-bit kernel advancing its bit-plane pointer by block instead
    of group index leaves them green while decode-path PPL reads 8.19 instead of 5.80.
  - Termination is judged without an output cap; a capped run can only show "not stopped yet".

---

## 5. Results

### 5.1 Quality

Each domain is judged on its own judging slice ([§4.5](#45-how-we-measure)): bare base vs the base with
that domain's sidecar. Within a table all rows come from the same engine binary in one run, except rows
marked ¹.

**Finance** (judge slice, 8,192 tokens; original model PPL 6.978):

| | Same top-1 | Σmin (median / p5) | Mean KL | PPL ratio |
|---|---:|---|---:|---:|
| ① base alone ¹ | 71.73% | 0.703 (0.751 / 0.231) | 0.612 | 1.339 |
| ① + finance sidecar, gains only | 73.94% | 0.742 (0.810 / 0.276) | 0.522 | 1.282 |
| **① + finance sidecar** | **74.57%** | **0.745 (0.810 / 0.283)** | **0.512** | **1.267** |

¹ Reference forward on the same file; engine vs reference on the same file differ by KL 0.013.
Tensor-core prefill ([§4.4](#44-the-engine)) moves the finance-sidecar row to 74.48% / 0.743 / 0.516
(rounding noise).

**Code** (judge slice, 15,360 tokens = 30 languages × 512; original model PPL 4.766):

| | Same top-1 | Σmin (median / p5) | Mean KL | PPL ratio |
|---|---:|---|---:|---:|
| ① base alone | 78.61% | 0.754 (0.809 / 0.318) | 0.421 | 1.367 |
| **① + coding sidecar** | **82.26%** | **0.803 (0.872 / 0.402)** | **0.303** | **1.229** |

**Law** (judge slice, 15,360 tokens = 6 sources × 2,560; original model PPL 4.865):

| | Same top-1 | Σmin (median / p5) | Mean KL | PPL ratio |
|---|---:|---|---:|---:|
| ① base alone | 72.90% | 0.713 (0.773 / 0.206) | 0.578 | 1.358 |
| **① + law sidecar** | **76.36%** | **0.760 (0.834 / 0.272)** | **0.454** | **1.251** |

**Medicine** (judge slice, 15,360 tokens = 6 sources × 2,560; original model PPL 8.625):

| | Same top-1 | Σmin (median / p5) | Mean KL | PPL ratio |
|---|---:|---|---:|---:|
| ① base alone | 69.08% | 0.683 (0.692 / 0.259) | 0.602 | 1.378 |
| **① + medicine sidecar** | **72.82%** | **0.739 (0.767 / 0.325)** | **0.465** | **1.280** |

**Science** (judge slice, 15,360 tokens = 6 sources × 2,560; original model PPL 8.770):

| | Same top-1 | Σmin (median / p5) | Mean KL | PPL ratio |
|---|---:|---|---:|---:|
| ① base alone | 71.65% | 0.707 (0.726 / 0.282) | 0.537 | 1.259 |
| **① + science sidecar** | **74.93%** | **0.753 (0.788 / 0.347)** | **0.422** | **1.173** |

**General English** (WikiText-2, 512 tokens; original model PPL 1.657): the check that a domain sidecar
does not cost general capability. The bare-base row reads identically in every run.

| | Same top-1 | Σmin (median / p5) | Mean KL | PPL ratio |
|---|---:|---|---:|---:|
| ① base alone | 78.52% | 0.769 (0.950 / 0.065) | 0.694 | 1.797 |
| ① + finance sidecar | 80.66% | 0.787 (0.965 / 0.083) | 0.619 | 1.658 |
| ① + coding sidecar | 82.23% | 0.797 (0.972 / 0.103) | 0.584 | 1.644 |
| ① + law sidecar | 82.81% | 0.800 (0.973 / 0.085) | 0.580 | 1.590 |
| ① + medicine sidecar | 82.81% | 0.801 (0.976 / 0.102) | 0.599 | 1.592 |
| ① + science sidecar | 82.62% | 0.800 (0.975 / 0.126) | 0.566 | 1.564 |

At 512 positions one position is 0.2 pp: these rows read as "no regression", not as a gain.

### 5.2 Speed (one DGX Spark)

| Workload | Prefill | Decode |
|---|---:|---:|
| 12.5k-token prompt | **489 t/s** (253 with `--decoder-full`) | — |
| Short prompt, plain greedy | — | **30.5–30.7 t/s** |
| Real agent request, 14.1k-token prompt, plain | — | 28.9–29.4 t/s |
| Same request, speculative (default) | — | **43.0 t/s** (3.04 tokens per round) |
| Request never used for tuning, 9.2k prompt, speculative | — | **37.1 t/s** |
| 51k context, plain | — | 27.5 t/s |

---

## 6. What did not work

Negative results carry as much of the design as positive ones. Each row was measured, not argued.

| Idea | What we measured | Verdict |
|---|---|---|
| Dynamic bits per layer / expert / matrix | Importance spread 3% / 1.25× / 33:33:33; moving a bit needs ≥ 1.189× | Uniform is optimal |
| Dynamic bits per row | 2.5× energy spread, eaten by codebook cost and integer bit widths: net −1.6% to −5.9% | Rejected |
| Expert pruning to reach 100 GB | Would keep 130 of 384 experts per layer | Rejected: a domain decision in disguise |
| Low-rank additive correction `y += B·A·x` | Held-out +9% per layer, end to end negative; first-order effect at the output ≈ 0 | Whole family rejected |
| Mean / bias correction of quantization error | Inside the layer +4 pp; at the output −0.33 pp (3 layers); 40 layers 70.52 → 63.89% | Rejected |
| Re-solving codebooks on domain data | 65.43% vs 65.59% | No gain |
| Entropy coding of VQ indices | Empirical entropy 11.927 of 12 bits: saves 0.6% | Not worth it |
| Entropy-constrained VQ + variable-length streams | ≈ +0.1 bit net after stream overhead | Not worth it |
| Finance-calibrated base | Finance flat, general text −2.4 pp | Base stays zero-corpus |
| Draft vocabulary cut to finance terms | +0.2% to −54% on an unseen request | Rejected |
| Grouped multi-token expert kernel for verification | Halves codebook lookups, 0 ms saved | Rejected |
| Second machine for decoding | Layer split −19% (upstream); tensor parallel 1.25× | One box |

---

## 7. Run it

**What you need.**

- NVIDIA DGX Spark (GB10, sm_121, 128 GB) — the only machine tested. The FP8 codebook path needs sm_89
  or newer. V4.1 runs on CUDA only.
- Linux aarch64 with the CUDA 13 runtime (`libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`;
  DGX OS ships them). Missing libraries show up as `error while loading shared libraries: libcudart.so.13`.
- ~320 GB of local SSD: 113.6 GB for this repository + 203 GB for two official shards (below).
- Nothing else heavy running: the engine refuses to start if it cannot fit the 110 GB budget.

**One command.** On the Spark:

```sh
curl -fsSLO https://huggingface.co/wenzhouwu/YoungAi-DeepSeek-V4.1-Flash/resolve/main/install.sh
bash install.sh
```

It checks the machine (GPU, CUDA 13 libraries, memory, disk), downloads this repository and the two official
shards (~317 GB), assembles the base model, starts the server on `127.0.0.1:8000` with the finance sidecar
under a memory watchdog, and prints the answer to a one-line test question. Everything goes under `--dir`;
no `sudo`. Interrupted? Run the same command again: finished files are skipped and the assembly resumes
where it stopped.

```sh
bash install.sh --domain code                             # install and serve with the coding sidecar
bash install.sh stop && bash install.sh start --domain finance   # switch domains (all sidecars are already local)
```

| Option | Effect |
|---|---|
| `--domain finance \| code \| law \| medicine \| science \| none` | which ② sidecar the server loads (default `finance`; `none` = bare base) |
| `--dir DIR` | install directory (default `~/youngai`) |
| `--engram-dir DIR` | you already have `model-00047-of-00048.safetensors` and `model-00048-of-00048.safetensors` from the official checkpoint in DIR: use them in place, skip the 203 GB download |
| `--host 0.0.0.0` / `--port N` | serve your LAN / another port |
| `--endpoint https://hf-mirror.com` | download through a mirror |
| `--no-xet` | download over the plain LFS channel (if transfers keep failing with "peer closed connection") |
| `--posttrain` | also load the experimental post-training file (finance only) |
| `--no-start` | install only; later `bash install.sh start`, `stop`, `status` |

The steps below are what the script does, for doing it by hand.

**Files in this repository.**

| Path | What |
|---|---|
| `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf.part01-of-40` … `part40-of-40` | ① base, 113,556,639,424 bytes, split into 40 parts |
| `SHA256SUMS` | sha256 of the assembled base and of every part |
| `install.sh` | the one-command installer above |
| `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-vqfin41_vqhalf_a_n8192-engine/` | ② finance sidecar: `gr_Lnn.bin` (gains, 39 layers) + `rb_Lnn.bin` (router bias, 27 layers) + `manifest.txt` (per-layer λ and held-out gain) |
| `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-code_fit_n15360-engine/` | ② coding sidecar, same layout: gains on 39 layers, router bias on 29 layers |
| `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-law_fit_n15360-engine/` | ② law sidecar: gains on 40 layers, router bias on 24 layers |
| `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-med_fit_n15360-engine/` | ② medicine sidecar: gains on 40 layers, router bias on 29 layers |
| `DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-sci_fit_n15360-engine/` | ② science sidecar: gains on 40 layers, router bias on 29 layers |
| `posttrain-experimental-20260924/` | ③ an experimental post-training file: `gr_L39.bin` + `base.fnv` (see below) |
| `bin/ds4`, `bin/ds4-server` | engine binaries, built on the Spark with `make cuda-spark` |
| `LICENSE`, `LICENSE-DeepSeek` | MIT notices for the engine (incl. GGML) and for the model weights |

**Step 1 — download and assemble.** By hand this needs 227 GB free during assembly (the installer needs one
part's worth, because it appends and deletes one part at a time).

```sh
hf download wenzhouwu/YoungAi-DeepSeek-V4.1-Flash --local-dir ds4-v41
cd ds4-v41
sha256sum -c --ignore-missing SHA256SUMS            # every part must report OK
cat DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf.part{01..40}-of-40 > DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf
sha256sum -c --ignore-missing SHA256SUMS            # now the assembled .gguf reports OK too
rm DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf.part*-of-40
chmod +x bin/ds4 bin/ds4-server                      # downloads do not keep the executable bit
```

**Step 2 — the n-gram memory tables.** They are not in this repository: the engine reads them, untouched,
from two shards of the official checkpoint (≈ 101.5 GB each).

```sh
hf download deepseek-ai/DeepSeek-V4.1-Flash \
    model-00047-of-00048.safetensors model-00048-of-00048.safetensors --local-dir /data/DeepSeek-V4.1-Flash
```

Any folder works; tell the engine where it is with `--engram-dir` (next step). The GGUF itself only records the
path these shards had on the machine that built it, which does not exist on yours — without `--engram-dir` the
engine warns at load and stops at the first request with `ds4: engram 表打不开 …` ("cannot open engram table").

**Step 3 — serve.**

```sh
cd ds4-v41
SIDECAR=DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-vqfin41_vqhalf_a_n8192-engine    # finance
# SIDECAR=DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-code_fit_n15360-engine         # code
# SIDECAR=DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-law_fit_n15360-engine          # law
# SIDECAR=DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-med_fit_n15360-engine          # medicine
# SIDECAR=DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-sci_fit_n15360-engine          # science
./bin/ds4-server --cuda -m DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf --zchain $SIDECAR \
    --engram-dir /data/DeepSeek-V4.1-Flash --mem-budget-mb 110000 --host 0.0.0.0 --port 8000
```

Loading takes about two minutes. Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`
(OpenAI style) and `/v1/messages` (Anthropic style). Sampling knobs a request leaves out follow the model
card's recipe: `temperature` 1.0, `top_p` 1.0, no `min_p` — sampled, plain decode. For greedy decoding (which
is what enables speculative decoding and byte-reproducible output) send `"temperature": 0` explicitly.
Don't make greedy the chat default: on questions with very few valid answers it can loop verbatim inside
the thinking section and never stop (e.g. "list 5 Chinese idioms ending in 五").

**Command line.**

```sh
./bin/ds4 --cuda -m DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf --zchain $SIDECAR \
    --engram-dir /data/DeepSeek-V4.1-Flash -p "Explain the price-to-earnings ratio."
# samples with the model card's recipe by default (plain decode); add --temp 0 for greedy + speculative,
# plus --no-dspark for plain greedy (speed baselines); drop --zchain to run the bare base
```

**The post-training file is an experiment, not an upgrade.** `posttrain-experimental-20260924/` holds
last-layer gains solved on the decision points of one market-outlook request; it demonstrates the format
and the loading path. `base.fnv` is the finance sidecar's fingerprint, so it stacks only on that sidecar
and the engine refuses any other pairing:

```sh
./bin/ds4-server … --zchain DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative-grrb-vqfin41_vqhalf_a_n8192-engine \
    --posttrain posttrain-experimental-20260924
```

**Source code** of the engine, the quantizer and the solvers is not public yet.

---

## 8. Honest status and limits

- **Post-training (③) is a working mechanism, not yet a working product.** It flips targeted decisions on
  the requests it was solved on without disturbing other tokens, but does not transfer: on held-out days,
  decision points stay at 55% → 55%. One flipped token changes one sentence, not a chain of reasoning.
- **Five domains.** Finance, code, law, medicine and science (§5.1), all five sidecars in this repository.
  All are judged teacher-forced on held-out text; none has been evaluated on end-to-end tasks (agentic coding, legal or clinical question answering).
- **CUDA only, one machine type tested.** V4.1 does not run on Metal.
- **Speculative decoding is greedy-only.** Sampling requests fall back to plain decode.
- **Prefill on tensor cores is not bit-identical** to the fused scalar path; the difference is at the
  rounding-noise level (§4.4).

---

## Author and contact

Wenzhou Wu (吴文周) · 310066827@qq.com

Questions, reproduction reports and collaboration offers are welcome.

## Acknowledgements

This project started as a fork of [antirez/ds4](https://github.com/antirez/ds4) (DwarfStar), the
DeepSeek-V4-specific engine by Salvatore Sanfilippo and contributors, where the documentation for the
V4 / Metal paths lives.
Like upstream, we are indebted to [llama.cpp and GGML](https://github.com/ggml-org/llama.cpp): GGUF,
quantization layouts such as q4_K, and much hard-won kernel knowledge come from there, and the GGML
authors' copyright notice stays in `LICENSE`. The model is DeepSeek's; thanks to DeepSeek for releasing
the weights and the reference inference code that serves as our ruler.

## License

Engine: MIT — see [`LICENSE`](LICENSE). Quantized weights derive from DeepSeek V4.1 Flash, MIT — see
[`LICENSE-DeepSeek`](LICENSE-DeepSeek).
