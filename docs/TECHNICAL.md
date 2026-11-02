# YoungAi — technical notes

**English** · [中文](TECHNICAL.zh-CN.md) · back to the [README](../README.md)

This page holds everything the README leaves out: why the system is shaped this way, the algorithms behind
each of the three files, the full result tables, what did not work, and the manual install. The section
numbers are stable so other pages can link to them.

## 1. The numbers

The headline numbers live in the [README](../README.md#numbers). Every ruler they use is defined in
[§4.5](#45-how-we-measure).

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
  same solver. On its own held-out text each sidecar lifts Same top-1 by **+2.6 to +3.8 pp** and cuts
  mean KL by **17–30%** (tables in [§5.1](#51-quality)).
- **No harm to general text.** On English Wikipedia every one of the five sidecars raises Same top-1
  from 76.8% to **77.9–79.7%** and Σmin from 0.725 to **0.753–0.760**.
- **It reproduces on a new base.** The same recipe, corpora and solver were rerun unchanged on the 1017
  release of the weights; per-domain gains match the previous base within 0.3 pp (code +3.67 vs +3.65,
  medicine +3.75 vs +3.74).
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
  TP=2 and speculative decoding at ~41 tokens/s; one Spark here runs V4.1 at 42–46.

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

**What we built.** A third file, trained on the machine from a plain question/answer file and stacked on
top without touching ① or ② ([§4.3](#43-the-post-training-file)).

---

## 3. The architecture: base + sidecar + post-training file

```
                 what it is                          how often it changes
 ① base GGUF     113.6 GB, general, zero corpus      quantized once from the official weights
 ② sidecar       ~40 MB per domain   (--zchain)      solved once per domain, then frozen
 ③ post-train    ~100 MB             (--posttrain)   trained from your own data; delete it to roll back
 ──────────────────────────────────────────────────────────────────────────────────────────
 each routed-expert weight row:   g_eff = g_base × s_sidecar
 each router:                     selection score += Δb_sidecar
 each MoE block output:           y += x · B · A      (③, rank 64 per layer)
```

- **① alone is a complete model.** ② only rescales rows that already exist and nudges the router; ③ adds a
  small low-rank term to each MoE block's output. Neither adds layers; the kernels are the same with or
  without them.
- **Near-zero runtime cost.** ② adds ~0.6 MB of reads per decoded token, about 0.01% of what the base
  reads; ③ is two small matrix products per layer.
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
| finance | 8,192 tokens, five Chinese-finance sources | 40 / 32 layers | 40 MB | 28 + 34 min |
| code | 15,360 tokens, 30 programming languages × 512 | 40 / 29 layers | 40 MB | 31 + 65 min |
| law | 15,360 tokens, 6 sources × 2,560 (3 Chinese, 3 English) | 40 / 25 layers | 40 MB | 33 + 64 min |
| medicine | 15,360 tokens, 6 sources × 2,560 (3 Chinese, 3 English) | 40 / 31 layers | 40 MB | 33 + 75 min |
| science | 15,360 tokens, 6 sources × 2,560 (3 Chinese, 3 English) | 40 / 30 layers | 40 MB | 34 + 72 min |

**Evidence** (Same top-1, paired runs; full tables in [§5.1](#51-quality)). With each domain's own sidecar:
finance 71.8 → 74.4%, code 78.3 → 82.0%, law 73.2 → 76.5%, medicine 69.4 → 73.1%, science 71.5 → 74.9%.
The router bias earns its place on top of the gains: on the previous base, with the finance gains held
fixed, adding it gave +0.24 pp and KL −2.5% on finance, +0.78 pp and KL −3.8% on English.

### 4.3 The post-training file

**In one sentence.** A rank-64 low-rank term on every MoE block (`y += x·B·A`), trained by backpropagation
through the engine's own GPU forward pass, from one jsonl file of questions and answers.

**The data.** One line per example, in either of the two shapes Unsloth's trainer takes (conversational
`messages`, or raw `text` for continued pretraining):

```json
{"messages":[{"role":"user","content":"…"},{"role":"assistant","content":"…"}],"context":"optional source text"}
{"text":"raw text, no chat template"}
```

**Three objectives, picked per line.**

- **`text`** (continued pretraining). The sequence is BOS + text + EOS and every token is a target
  (cross-entropy). A line longer than `maxlen` is cut into consecutive windows that each become an example;
  nothing is truncated away. Turning documents into question-answer pairs first (`gen`) is an optional data
  tool, not part of training.

- **With `context`** (context distillation). The teacher is the deployed model (① + ②) answering with the
  source text in its prompt; the student is the same model plus ③ answering *without* it. The loss is the KL
  from the teacher's next-token distribution to the student's, on the answer positions. What the model
  learns is "answer as if you had just read this", which carries over to questions the file never asked.
- **Without `context`** (plain SFT). The target is the answer itself (cross-entropy on the answer tokens).

**Keeping general ability.** A fixed hold set (`gguf-tools/data/posttrain/hold.jsonl`, 63 general questions
with the deployed model's own answers) is mixed into every run, with the deployed model as its teacher. It
pins ③ on everything the training file is not about.

**Multi-form expansion (on by default).** With one phrasing per question the low-rank layer learns "after
this string, emit that string" and fails on a new phrasing: 12 facts trained with 3 phrasings each answered
only 8 of 13 rephrased questions. So at load time the deployed model rewrites every training pair into 4 more
jsonl lines of the same format — question rephrased, answer reworded, facts unchanged, no answer names or
numbers in the question, at least one pair in another language — and they train alongside the original. A
rewritten line uses the original answer (or source text) as the teacher context, i.e. context distillation,
not a gold label: if a rewrite slips in an invented detail, the teacher gives it no weight there. A version
that only rephrased the question and copied the answer verbatim was tried and lost (the answer string stays
identical, the model memorises it, and rephrased questions got worse). Rewrites are cached next to the data
file (`<file>.expand4.jsonl`, the model's raw output in `.raw.txt`); held-out questions are never expanded.

**Picking the epoch.** Three readings, each answering one question, never traded against each other:
① learned? — held-out loss (the generalization set, if given) below its pre-training value; ② still talks? —
the held-out questions generated with ③ mounted up to 4× the reference length, with no fewer answers
terminating than the base model's; ③ general drift — the per-position paired difference of Σmin and KL on
WikiText-2 between ② and ② + ③, with its standard error; a drop beyond 3 standard errors is a *real drop* and
that epoch is excluded, anything within noise is ignored. Among the epochs meeting all three, the lowest
held-out loss becomes ③; if none does, the run never reaches the switch. An earlier absolute line ("Σmin
drops at most 0.5 pp") turned out to be only 1.3–1.5 standard errors wide: three epochs of one run flipped
sign at random and the epoch it rejected was the one that had learned, hence the paired criterion. When a
generalization set (the same questions in another form) is given, the data file itself is no longer split by
hash: in a small file the fifth it removes is often the only line carrying a fact. The teacher's
distributions are computed once and cached next to the data file, so a second run on the same file skips
that step.

**Why backpropagation and not a closed-form solve.** An earlier version linearised "the model should have
written *a*, not *b*" onto last-layer gains and solved it in seconds. It flipped the targeted tokens
(55% → 88% on the training requests, 99.66% of other positions unchanged) but nothing transferred to
held-out days: one token moves one sentence, not a chain of reasoning, and an answer position whose
activations don't look like the fitting rows never fires. Every published method that writes documents into
weights (knowledge modules, prompt distillation, cartridges, self-distillation) trains by gradient with the
teacher's distribution as the target; that is what this version does.

**Evidence (training on your own data).** These numbers were measured on the previous base (before the
1017 release) and have not been rerun on the current one. One DGX Spark, a real file of 1,326 questions
generated from 20 trading-review documents (plus the 63 hold questions), 3 epochs, layers 0–39, learning
rate 2e-4, 72 minutes end to end:

| | Held-out loss | Hold-set KL | WikiText-2 Σmin | WikiText-2 mean KL | WikiText-2 Same top-1 |
|---|---:|---:|---:|---:|---:|
| before (① + ②) | 1.766 | 0.006 | 0.7887 | 0.6170 | 80.27% |
| epoch 1 | 0.865 | 0.093 | −0.24 pp | −4.6% | |
| epoch 2 | 0.731 | 0.121 | −0.40 pp | −8.5% | |
| **epoch 3 (picked)** | **0.646** | 0.116 | **+0.50 pp** | **−10.6%** | **82.81%** |

Greedy answers to held-out questions after epoch 3 reproduce all four figures they were asked about
(entry price 30.39, target 25.57, stop 28.03, deviation −4.89%) digit for digit, and general questions
(Earth's orbit, a translation) are answered as before.

**Safety.** ③ is its own directory, fingerprinted against the ② it was trained on (`base.fnv`); the engine
refuses a mismatched pair. Deleting the directory rolls back.

**Smaller runs.** An 18-line smoke run (1 epoch) takes about 10 minutes end to end, rewrites and checks included.
On 24 lines (12 facts, two phrasings each) the trained layer answered 11 of 12 questions asked in a third, unseen
phrasing (0 of 12 before training), while eight general-knowledge questions were answered the same as before.

**Minimal example: two lines.** To walk the whole loop once in the Studio, save these two lines as the training
file. The first teaches the reply "在呢主人" ("right here, master") to "hello 小泱" (Xiaoyang is the assistant's
name); the second teaches one fixed two-sentence story about a snail who crawls a year to see the sea and finds
the sound of the waves was in its shell all along.

```json
{"messages":[{"role":"user","content":"hello 小泱"},{"role":"assistant","content":"在呢主人"}]}
{"messages":[{"role":"user","content":"小泱，讲个两句话的小故事吧"},{"role":"assistant","content":"从前有只小蜗牛想去看海，它慢慢爬了整整一年。到了海边它才发现，自己背上的壳里一直装着海浪的声音。"}]}
```

With only a few lines, also save a second file that asks the same two things in other words, and pick it in
*Generalization set (optional)* on the *Train* page:

```json
{"messages":[{"role":"user","content":"小泱你好"},{"role":"assistant","content":"在呢主人"}]}
{"messages":[{"role":"user","content":"小泱，来个超短的小故事"},{"role":"assistant","content":"从前有只小蜗牛想去看海，它慢慢爬了整整一年。到了海边它才发现，自己背上的壳里一直装着海浪的声音。"}]}
```

Without it, the only held-out questions left are the built-in general ones. Their loss always goes up after
training, so the run is judged unusable and never shows up in the switch. The default 3 epochs take about
12 minutes from stopping the model to having it back:

| | Before | After |
|---|---|---|
| Held-out loss, rephrased questions | 2.82 | 0.24 |
| "hello 小泱" (temperature 0) | a long greeting | 在呢主人 |
| The story request (temperature 0) | a different story about a cat | the snail story, word for word |
| "小泱，来个超短的小故事" (an unseen phrasing) | a different story | the snail story |
| What is 1+1 | 2 | 2 |

The chat page sends no temperature and samples with the model card's recipe; there "hello 小泱" gets exactly
"在呢主人" about 60% of the time, and close variants like "在呢，主人～" otherwise. Asked as "小泱你好", the
picked epoch still gives the old greeting. Epoch 2 did answer it correctly, but the pick looks at the summed loss
of both held-out questions, the longer story question dominates that sum, and epoch 3 won.

**The one published with the base: distilled from the original weights.** Everything above is about training ③ on your
own data. `posttrain-code-20261022/` in the Hugging Face repository, published with the 1017 base, is a different use:
the teacher is not the deployed model but the 510 GB original itself, and the goal is to reproduce the original
position by position, not to teach anything new.

- **How it is trained.** Same structure as above: a rank-64 `x·B·A` on the output of every MoE layer, layers 0–39,
  learning rate 2e-4, 2 epochs, stacked on the code sidecar. The teacher's distribution comes from one forward pass of
  the official reference implementation over the original weights (streamed from disk layer by layer, keeping the top
  64 and the remaining mass at each position). The loss is the KL from the original's distribution to the student's at
  every position; there are no hard labels.
- **Data.** 612 lines. 330 whole public documents (120 code, 90 law, 90 medicine, 30 science, 300–1,000 tokens each,
  all with their real ending so that EOS, the "stop" decision, is in training); 192 reasoning passages the original
  wrote on the official API, taking the tail of the reasoning, the close and the answer (100 coding questions, 98
  general and decision questions); 90 opening stretches of the same reasoning with no close, as "not time to stop
  yet" samples; plus the 63 hold questions. The 120 code windows used for scoring were checked against all training
  data 24 tokens at a time: zero overlap. The same check run against the sidecar's fitting half does find a whole
  window, so the check itself works.
- **Time.** One DGX Spark. The teacher's distribution is computed 96 sequences at a time, each batch streaming the
  476 GB original from disk once: about 24 minutes per batch with whole-layer read-ahead (37–42 before), 8 batches for
  675 sequences. Training takes 67 minutes.

Code scoring slice (15,360 tokens); the three rows are the same ruler, measured in the same period:

| | Same top-1 | Σmin | Mean KL | PPL ratio |
|---|---:|---:|---:|---:|
| ① base only | 78.40% | not measured | 0.446 | 1.394 |
| ① + code sidecar | 81.96% | 0.790 | 0.318 | 1.243 |
| **① + code sidecar + ③** | **84.77%** | **0.836** | **0.230** | **1.106** |

WikiText-2 (512 tokens):

| | Same top-1 | Σmin | Mean KL | PPL ratio |
|---|---:|---:|---:|---:|
| ① base only | 76.76% | 0.725 | 0.803 | 1.685 |
| ① + code sidecar | 79.69% | 0.760 | 0.673 | 1.473 |
| **① + code sidecar + ③** | **82.42%** | **0.789** | **0.574** | **1.227** |

Quantization's bias on "decision" tokens, read on three real sequences that are not in training (original = the
reference forward over the original weights):

| Reading | Original | ① + code sidecar | ① + code sidecar + ③ |
|---|---:|---:|---:|
| In two real reasoning runs, mean probability of `?` where the original did not pick `?` | 2.84e-3 / 1.16e-3 | 5.89e-3 / 2.37e-3 | 3.30e-3 / 1.50e-3 |
| Probability of EOS at the real end of 25 whole documents (geometric mean) | 0.063 | 0.038 | 0.094 |
| Summed probability of `</think>` over an 11k-token reasoning run (the original closes it) | 0.82 | 0.28 | 1.30 |
| The same over a 32k-token run (the original itself does not close it) | 0.049 | 0.034 | 0.100 |

How to read it: the extra question marks are essentially back at the original's level; "stop" goes from mostly
suppressed to somewhat more than the original, which means long reasoning closes earlier than the original's.

**What is not settled yet.** The chance of closing long reasoning jumps between training epochs: four checkpoints
trained the same way read 0.84–2.05× the original on the run that does close. Training and held-out data stop at
1,000 tokens while these runs generate 10k–30k tokens, so behaviour there is extrapolated; and two runs cannot show
"matches the original". Next: train on longer data and judge on more real long runs.

**Why there is more room.** 612 lines, each under 1,000 tokens, rank 64; held-out loss kept falling over both epochs
(0.333 → 0.206 → 0.196) without reaching a plateau.

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
accumulated from zero and added with a separate FADD (0.5%). A prompt is processed in 2,048-token blocks:
every block has to decode the whole layer's expert bit-stream once (2.55 GB per layer), so the block size
sets how many tokens amortise that decode. Real routing is skewed (a 2,048-token block touches 200–350 of
the 384 experts, the hottest one taking 8–16% of all picks), so one work item covers up to 128 tokens of an
expert with 16 warps, and 12-bit layers keep their codebook in shared memory already converted to bf16.
Sparse attention is a single-pass flash kernel: the query's fragments stay in registers, scores come from
`mma.sync`, the running softmax rescales the output accumulator in place (the accumulator layout of
`mma.m16n8k16` is documented, which `wmma` fragments are not), and each key is gathered once. The n-gram
memory rows for a block are fetched on an I/O thread while the GPU computes, and only the last position of
the last block goes through the output head. 12.5k-token prompt: **216.5 → 1,055 tokens/s**. Quality is
unchanged within rounding noise: the finance judgement set reads Σmin 0.7436 → 0.7441, KL 0.5149 → 0.5131,
same top-1 74.56% → 74.77%, and everything except the attention kernel is byte-identical to the scalar path.

**Speculative decoding.**

- The model's three draft towers are quantized with the same VQ (2.56 GB).
- The verify batch (1 + k rows) has its own kernels: GEMVs on stage/pipe with several rows, and a
  persistent expert kernel over (unique expert × rows) that reads each chosen expert's bits once.
- **Scheduler.** The draft head reports a confidence `c_j` per position. With survival
  `a_j = Π_{i≤j} σ(c_i)`, pick

  ```
  k* = argmax_k  (1 + Σ_{j≤k} a_j) / (c_draft + c_v1 + c_tok · (1 + k))
  ```

  with the three costs measured by the engine itself, per request: the wall clock of a plain graph step,
  of a draft round, and of verifying `1 + k` rows (a weighted least-squares line over the row counts seen so
  far; warm-up rounds that pay lazy allocation are excluded). The first rounds verify the longest block and
  then one row to get the two points the line needs. There are no compile-time cost constants. The
  survival probabilities are calibrated online too: `ρ` = first-position hits / Σ σ(c_0) over the
  request, multiplied into every `σ(c_i)`. Timing only ever changes `k`, never a token: speculative
  output is byte-identical to plain decode for every `k`, so a wall-clock-driven `k` is safe.
- **Guarantee:** at temperature 0, speculative output is byte-identical to plain greedy output, checked on
  every change. On by default for greedy and sampled requests alike. A sampled request draws its drafts
  from the draft towers' own distribution (same temperature and truncation) and verifies them by rejection
  sampling on the GPU (accept a draft token with probability `min(1, p_target/p_draft)`, otherwise draw
  from the residual `max(0, p_target − p_draft)`), so the token distribution is exactly plain sampled
  decode's. Drafting from the distribution rather than the towers' argmax was decided by an analytic jury on
  fixed texts (expected first-position acceptance at temperature 1.0: 0.69 → 0.74 and 0.63 → 0.76), not by
  timing runs, because every sampled run is a different text. The scheduler's acceptance-rate calibration
  is learned online per request, no constants.
  Sampling itself runs on the GPU inside the decode graph (Gumbel-max with exact `top_k`/`top_p`/`min_p`
  truncation by radix select; the host reads back 16 bytes per row), checked by a distribution gate
  (`make cuda-regression`).
- **Why it isn't higher:** each extra verified token touches ~3.9 new experts (hash-like routing,
  [§2.2](#22-scaling-laws-are-real-and-knowledge-is-spread-like-a-hash)), so verifying k + 1 tokens costs
  far more than verifying one. The verify batch runs at 57% of its own byte wall; that is the next lever.

**Memory.** Weights are mmap-backed. Per-request state grows with the positions actually used: the 1M
KV itself is 0.89 GB, and the per-forward scratch grows by doubling instead of being sized for 1M up
front (2.5 GB → 160 MB for a 40k-token request). Logits and the output-head scratch no longer scale with
the prefill block (a generation request keeps 10 rows of logits, not 2,048). A 106.7k-token request
through the server leaves 6.7 GB free on the 121 GB machine. There is no context knob — the bound comes
from the GGUF metadata and `--ctx` is rejected.

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

  | Ruler | Judging slice | Original model PPL (1017) |
  |---|---|---:|
  | finance | 8,192 tokens, five Chinese-finance sources | 6.946 |
  | code | 15,360 tokens: 30 languages × 512 from GitHub (`codeparrot/github-code-clean`), license headers stripped | 4.777 |
  | law | 15,360 tokens, 6 × 2,560: Chinese statutes, criminal-case facts (CAIL2018), bar-exam questions (JEC-QA); US Supreme Court opinions, EU legislation, contract clauses (LexGLUE) | 4.741 |
  | medicine | 15,360 tokens, 6 × 2,560: Chinese medical encyclopedia Q&A, real doctor consultations (cMedQA2), licensing-exam questions with explanations (CMExam); PubMed abstracts, PMC case reports, USMLE questions (MedQA) | 8.647 |
  | science | 15,360 tokens, 6 × 2,560: Chinese journal abstracts, STEM and humanities / social science (CSL), college STEM questions (C-Eval); arXiv LaTeX source, S2ORC full papers (peS2o), MMLU-Pro STEM questions | 8.559 |
  | English | WikiText-2, 512 tokens | 1.795 |

  The teacher is the release being quantized: each new set of official weights gets its own teacher logits,
  cached under a name that carries the weights' folder. Scoring a new base against the old base's teacher
  would count the difference between two models as quantization error, without any error message.

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
that domain's sidecar, base and sidecars built from the 1017 release. All rows come from the engine's
scoring path with one binary; within a table, one run.

**Finance** (judge slice, 8,192 tokens; original model PPL 6.946):

| | Same top-1 | Σmin (median / p5) | Mean KL | PPL ratio |
|---|---:|---|---:|---:|
| ① base alone | 71.80% | 0.702 (0.749 / 0.229) | 0.599 | 1.351 |
| **① + finance sidecar** | **74.38%** | **0.742 (0.806 / 0.286)** | **0.496** | **1.270** |

**Code** (judge slice, 15,360 tokens = 30 languages × 512; original model PPL 4.777):

| | Same top-1 | Σmin (median / p5) | Mean KL | PPL ratio |
|---|---:|---|---:|---:|
| ① base alone | 78.29% | 0.735 (0.785 / 0.306) | 0.451 | 1.399 |
| **① + coding sidecar** | **81.96%** | **0.790 (0.853 / 0.399)** | **0.318** | **1.243** |

**Law** (judge slice, 15,360 tokens = 6 sources × 2,560; original model PPL 4.741):

| | Same top-1 | Σmin (median / p5) | Mean KL | PPL ratio |
|---|---:|---|---:|---:|
| ① base alone | 73.20% | 0.708 (0.754 / 0.237) | 0.544 | 1.357 |
| **① + law sidecar** | **76.45%** | **0.751 (0.809 / 0.298)** | **0.431** | **1.254** |

**Medicine** (judge slice, 15,360 tokens = 6 sources × 2,560; original model PPL 8.647):

| | Same top-1 | Σmin (median / p5) | Mean KL | PPL ratio |
|---|---:|---|---:|---:|
| ① base alone | 69.37% | 0.676 (0.679 / 0.262) | 0.605 | 1.392 |
| **① + medicine sidecar** | **73.12%** | **0.730 (0.754 / 0.315)** | **0.474** | **1.290** |

**Science** (judge slice, 15,360 tokens = 6 sources × 2,560; original model PPL 8.559):

| | Same top-1 | Σmin (median / p5) | Mean KL | PPL ratio |
|---|---:|---|---:|---:|
| ① base alone | 71.48% | 0.699 (0.712 / 0.306) | 0.523 | 1.300 |
| **① + science sidecar** | **74.85%** | **0.748 (0.776 / 0.358)** | **0.408** | **1.205** |

**General English** (WikiText-2, 512 tokens; original model PPL 1.795): the check that a domain sidecar
does not cost general capability. The bare-base row reads identically in every run.

| | Same top-1 | Σmin (median / p5) | Mean KL | PPL ratio |
|---|---:|---|---:|---:|
| ① base alone | 76.76% | 0.725 (0.912 / 0.061) | 0.810 | 1.684 |
| ① + finance sidecar | 79.49% | 0.754 (0.943 / 0.072) | 0.706 | 1.506 |
| ① + coding sidecar | 79.69% | 0.756 (0.936 / 0.075) | 0.680 | 1.525 |
| ① + law sidecar | 78.12% | 0.755 (0.949 / 0.069) | 0.714 | 1.520 |
| ① + medicine sidecar | 78.71% | 0.753 (0.944 / 0.077) | 0.711 | 1.474 |
| ① + science sidecar | 77.93% | 0.760 (0.946 / 0.078) | 0.681 | 1.449 |

At 512 positions one position is 0.2 pp: these rows read as "no regression", not as a gain.

Adding the post-training file published with the base: code 84.77%, WikiText-2 82.42%. The three-row comparison with
the same ruler, measured in the same period, is at the end of §4.3.

**The restore sidecar** (repository directory `DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-restore_fit_n15360-engine`,
145 MB, passed to `ds4-server --zchain`): gains + router biases (40 MB) and a rank-64 correction (105 MB) in one directory.
The correction then keeps learning from the official API in rounds: the API writes fresh source files, the sidecar trains
on them once, and a round is kept only if code KL drops by more than 2 standard errors on windows 1 and 3 of every
language; windows 2 and 4 only report. The last round trained on full reasoning traces and was kept without that gate.
Same ruler:

| | Same top-1 | Σmin (median / p5) | Mean KL | PPL ratio |
|---|---:|---|---:|---:|
| Code judge slice, 15,360 | **85.04%** | 0.834 (0.902 / 0.475) | 0.230 | 1.098 |
| WikiText-2, 512 | 82.03% | 0.777 (0.944 / 0.075) | 0.586 | 1.276 |

**The index patch** (repository directory `posttrain-restore-20261103`, 12.6 MB, passed to `ds4-server --posttrain` on top
of the restore sidecar) is the first piece that touches the quantized experts themselves without adding a parameter: the
base stores every 8 expert weights as one index into a per-layer codebook, and the patch is a list of `{expert, matrix,
row, column group, new index, old index}` records that the engine applies to the copy of the blob the GPU reads (the file
on disk stays byte-identical; unloading writes the old indices back). Records are chosen one step per round: the
gradient of the KL to the original weights with respect to each group's 8 dequantized weights is accumulated over a
batch (one 15,360-token code stream, rotating over three fitting streams the ruler never sees, with the original model's
full logits as teacher, plus 24 short factual answers with the official API's top-20 as teacher); for each group the 16 nearest codewords are scored by the first-order change and
the best 0.05% of groups per matrix switch, but only when every stream in the batch agrees on the switch. A round is
reverted if held-out code KL rises or the paired code gate drops by more than 2 SE. Two rounds of one hour each: layers
10–19 on code + facts, then layers 20–29 with the agent-planning heads added (`core_ptrain_vstep.c`,
`gguf-tools/scripts/vstep_spark.sh`). 786,957 groups changed (665,394 in 10–19, 121,563 in 20–29). Same ruler, against
the restore sidecar it sits on (85.07% / 0.227 / 82.2%):

| | Same top-1 | Σmin | Mean KL | Tail KL > 2 / > 1 |
|---|---:|---:|---:|---:|
| Code judge slice, 15,360 | **85.16%** | 0.838 | 0.2238 | 184 / 629 (sidecar 177 / 634) |
| WikiText-2, 512 | 81.64% | 0.780 | 0.572 | |

All of its training text is generated fresh by the official API. A 16-gram fingerprint check against both rulers: 24 of
25 data files have zero hits; the one left hits two lines of Dockerfile boilerplate.

**Against the previous base.** The same recipe on the previous release of the weights gave per-domain
gains of +2.84 (finance), +3.65 (code), +3.46 (law), +3.74 (medicine) and +3.28 pp (science); this base
gives +2.58, +3.67, +3.25, +3.75 and +3.37. The bare base reads lower on WikiText-2 (76.8% vs 78.5%), but
that ruler's teacher changed too (original-model PPL 1.795 vs 1.657 on the same 512 tokens), and on the
16× longer finance ruler the bare base is level (71.70% vs 71.73% on the reference path).

### 5.2 Speed

One DGX Spark:

| Workload | Prefill | Decode |
|---|---:|---:|
| Request never used for tuning, 9.2k-token prompt, 1017 base + coding sidecar, plain | **1,043–1,067 t/s** | **29.9 t/s** |
| Same request, speculative (default) | — | **48.2 t/s** (2.46 of 5 drafts accepted) |
| Real request, 7.7k-token prompt, 1017 base + coding sidecar, plain | **999 t/s** | **30.1 t/s** (33.2 ms per token) |
| Same request, speculative (default) | 1,017 t/s | **45.7 t/s** (2.11 of 5 drafts accepted) |
| 12.5k-token prompt † | **1,055 t/s** (539 with `--decoder-full`) | — |
| Real agent request, 14.1k-token prompt † | **940 t/s** | — |
| 106.7k-token prompt through the server † | **671 t/s** (159 s to the first token) | — |
| Short prompt, plain greedy † | — | **30.5–30.7 t/s** |
| Real agent request, 14.1k-token prompt, plain † | — | 28.9–29.4 t/s |
| Same request, speculative † | — | **43.0 t/s** (3.04 tokens per round) |
| 51k context, plain † | — | 27.5 t/s |

The 9.2k rows: temperature 0, 512 tokens, two runs each; plain and speculative write byte-identical text.
† Measured on the previous base. The 1017 base file has the same size and layout to the byte
(113,556,639,424 bytes) and its sidecars are the same size, so every kernel reads the same bytes: on the
7.7k request, run back to back with one binary, the previous base with its coding sidecar reads 1,007 t/s
prefill and 30.0 t/s plain decode. Speculative decode is not comparable across bases from one run (46.5 vs
45.7 t/s): the two models write different text, and acceptance follows the text.

One Mac Studio M5 Max (128 GB), measured by a contributor through `ds4-server`: same 1017 base,
temperature 0, 512 tokens, best of two runs.

| Workload | Prefill | Decode |
|---|---:|---:|
| 25.5k-token prompt, one request, 1 output token | **627 t/s** | — |
| Prose and Chinese, speculative (default) | — | **35–36 t/s** |
| Code, speculative (default) | — | **41–42 t/s** |

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


## 7. Running by hand

The one-command installer and its options are in the [README](../README.md#how-to-use-it). This section is
what the installer does, for doing it yourself, plus the details of the server and its pages.

**Files in the [Hugging Face repository](https://huggingface.co/wenzhouwu/YoungAi-DeepSeek-V4.1-Flash-1017).**

| Path | What |
|---|---|
| `DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf.part01-of-40` … `part40-of-40` | ① base, 113,556,639,424 bytes, split into 40 parts |
| `DeepSeek-V4.1-Flash-1017.vision.gguf` | the vision tower, 971,066,016 bytes, kept at original BF16; read from disk only when an image arrives |
| `SHA256SUMS` | sha256 of the assembled base and of every part |
| `install.sh` | the one-command installer |
| `DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-vqfin41_vqhalf_a_n8192-engine/` | ② finance sidecar: `gr_Lnn.bin` (gains, 40 layers) + `rb_Lnn.bin` (router bias, 32 layers) + `manifest.txt` (per-layer λ and held-out gain) |
| `DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-code_fit_n15360-engine/` | ② coding sidecar, same layout: gains on 40 layers, router bias on 29 layers |
| `DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-law_fit_n15360-engine/` | ② law sidecar: gains on 40 layers, router bias on 25 layers |
| `DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-med_fit_n15360-engine/` | ② medicine sidecar: gains on 40 layers, router bias on 31 layers |
| `DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-sci_fit_n15360-engine/` | ② science sidecar: gains on 40 layers, router bias on 30 layers |
| `posttrain-code-20261022/` | ③ post-training file, 101 MB: `amp_L00.bin` … `amp_L39.bin` (rank 64 per layer, f32) + `base.fnv` (the code sidecar's fingerprint); pairs with the code sidecar only |
| `bin/ds4`, `bin/ds4-server` | engine binaries, built on the Spark with `make cuda-spark` from the source at [github.com/fodelf/YoungAi](https://github.com/fodelf/YoungAi) (engine, CUDA kernels, quantization and sidecar toolchain) |
| `LICENSE`, `LICENSE-DeepSeek` | MIT notices for the engine (incl. GGML) and for the model weights |

**Step 1 — download and assemble.** By hand this needs 227 GB free during assembly (the installer needs one
part's worth, because it appends and deletes one part at a time).

```sh
hf download wenzhouwu/YoungAi-DeepSeek-V4.1-Flash-1017 --local-dir ds4-v41
cd ds4-v41
sha256sum -c --ignore-missing SHA256SUMS            # every part must report OK
cat DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf.part{01..40}-of-40 > DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf
sha256sum -c --ignore-missing SHA256SUMS            # now the assembled .gguf reports OK too
rm DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf.part*-of-40
chmod +x bin/ds4 bin/ds4-server                      # downloads do not keep the executable bit
```

**Step 2 — the n-gram memory tables.** They are not in the Hugging Face repository: the engine reads them, untouched,
from two shards of the official checkpoint (≈ 101.5 GB each).

```sh
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    model-00047-of-00048.safetensors model-00048-of-00048.safetensors --local-dir /data/DeepSeek-V4.1-Flash-1017
```

Any folder works; tell the engine where it is with `--engram-dir` (next step). The GGUF itself only records the
path these shards had on the machine that built it, which does not exist on yours — without `--engram-dir` the
engine warns at load and stops at the first request with `ds4: engram 表打不开 …` ("cannot open engram table").

**Step 3 — serve.**

```sh
cd ds4-v41
SIDECAR=DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-vqfin41_vqhalf_a_n8192-engine    # finance
# SIDECAR=DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-code_fit_n15360-engine         # code
# SIDECAR=DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-law_fit_n15360-engine          # law
# SIDECAR=DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-med_fit_n15360-engine          # medicine
# SIDECAR=DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-sci_fit_n15360-engine          # science
./bin/ds4-server --cuda -m DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf --zchain $SIDECAR \
    --engram-dir /data/DeepSeek-V4.1-Flash-1017 --mem-budget-mb 110000 --host 0.0.0.0 --port 8000
```

Loading takes about two minutes. Endpoints: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`
(OpenAI style) and `/v1/messages` (Anthropic style). Sampling knobs a request leaves out follow the model
card's evaluation recipe: `temperature` 1.0, `top_p` 0.95, no `min_p` — sampled on the GPU, speculative decoding kept on
(34–37 tokens/s across runs on the 14k-token agent request vs 45.5 greedy, measured on the previous base).
For byte-reproducible output send
`"temperature": 0` explicitly.
Requests default to **non-thinking** mode. Turn thinking on per request with `"reasoning_effort": "high"` (or
`"max"`), `"thinking": {"type": "enabled"}`, `"think": true`, or `"model": "deepseek-reasoner"`. With a sidecar
alone, the quantized weights under-weight the "stop" decisions (`</think>`, EOS) relative to the original, so a long
thinking segment can fail to close; with the post-training file it is the other way round and closes a little early
(§4.3). Non-thinking stays the default.
Don't make greedy the chat default: on questions with very few valid answers it can loop verbatim inside
the thinking section and never stop (e.g. "list 5 Chinese idioms ending in 五").

**Monitor.** `http://host:8000/monitor` is a browser page served by the engine itself (no dependencies, same
origin as the API): model state with a prompt-reading / generation progress bar, decode and prefill tokens/s
with one-minute sparklines, GPU load / memory / temperature / power / PCIe, CPU, RAM, disk read, context fill,
the last requests (prompt, reused, output tokens, tokens/s, draft acceptance, first-token latency, duration)
and totals since start. Its data is `GET /metrics` — JSON by default (`?requests=all` for every kept request),
and the Prometheus text format with vLLM's metric names when asked with `Accept: text/plain` or
`?format=prometheus`, so a vLLM Grafana dashboard reads this server unchanged. Readings the platform cannot
provide (GPU temperature on macOS) are `null`, not guessed. On a unified-memory machine (GB10) the VRAM card is
the physical memory (128 GB, from the kernel's online memory blocks), with the kernel-allocatable total
(121.7 GB, `MemTotal`) in its subtitle. The page is bilingual (Chinese / English, follows the browser,
switchable, `?lang=en`). Its layout and the metric keys follow Strata's Monitor tab
(github.com/Niko1221/Strata), so the two can be compared side by side.

Both this page and the Studio at `/` are read from `web/` relative to the directory the server starts in.
Started from the source tree they are there; an install made by `install.sh` has no `web/`, so the pages answer
404 while the API works normally.

**Command line.**

```sh
./bin/ds4 --cuda -m DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf --zchain $SIDECAR \
    --engram-dir /data/DeepSeek-V4.1-Flash-1017 -p "Explain the price-to-earnings ratio."
# samples with the model card's recipe by default (plain decode); add --temp 0 for greedy + speculative,
# plus --no-dspark for plain greedy (speed baselines); drop --zchain to run the bare base
```

**The post-training file.** The one published with the base (§4.3) pairs with the code sidecar:

```sh
./bin/ds4-server … --zchain DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative-grrb-code_fit_n15360-engine \
    --posttrain posttrain-code-20261022
```

For a ③ you train yourself (*Train* tab, or `./ds4 --ptrain`), use your directory and the sidecar it was trained on.
`base.fnv` inside a ③ is the fingerprint of the sidecar it was trained against, and the engine refuses any other
pairing, so `install.sh --posttrain` only accepts the code sidecar and stops early on any other `--domain`. The old
experiment `posttrain-experimental-20260924/` (last-layer gains from the earlier closed-form solve, §4.3, on one
market-outlook request) stays in the previous repository,
[wenzhouwu/YoungAi-DeepSeek-V4.1-Flash](https://huggingface.co/wenzhouwu/YoungAi-DeepSeek-V4.1-Flash): it was trained
against the previous base's finance sidecar and does not stack on any 1017 sidecar.

### Building and running on a Mac

Every GPU operation V4.1 uses has a Metal implementation, transcribed kernel by kernel from
the CUDA one: the forward pass, the VQ expert decode (12-bit and 13-bit bit-plane), sidecar gains and router
bias, GPU sampling and speculative verification, and the backward pass post-training needs.
`./ds4_test --metal-v41` checks 133 cases on a real Apple GPU against CPU references on synthetic data
(no model needed): VQ expert decode matches bit for bit; the other matrix products agree to ~1e-4 and
are identical wherever the model rounds to bf16; attention is within one bf16 rounding step; every
backward kernel passes a finite-difference check; the half-activation prefill kernels below have their own,
wider bound.

**The full model runs on a 128 GB Mac.** A contributor runs the 1017 base on a Mac Studio M5 Max (speeds in
§5.2); our own Mac has 16 GB, so we have not run it ourselves. Three things made it work:

- **The VQ expert blobs are mapped for the GPU on first use.** They live outside the backbone and expert spans
  the loader maps up front, so without this the first MoE layer stops with *Metal model range … is not covered
  by mapped model views*. The views join the residency set; outside it, the driver held large speculative command
  buffers 30–1000 ms before the GPU started. These views are not counted by the load-time `--mem-budget-mb` gate.
- **Prefill uses the GPU's matrix units.** q4_K, bf16 and fp8 matrix products, the VQ experts and attention run on
  simdgroup matrices, or Metal 4 TensorOps when the shapes line up, with activations rounded to half the way
  llama.cpp's `kernel_mul_mm` does (relative error ~1e-3). These kernels run only on the inference path
  (generation, server requests, `--score-ids`). Training and the kernel tests keep the f32 kernels, because the
  backward pass recomputes the forward and has to match it.
- **Decode keeps f32 activations.** A q4_K GEMV ported from llama.cpp, VQ experts reading a per-layer f16 codebook,
  and attention split across the live keys.

In the contributor's check, teacher-forced against the original model's answers on 10 HumanEval+ tasks (1,439
tokens), the Mac kernels agree with the original model's top-1 token as often as the old scalar kernels did: 97.98%
on the decode path, 98.05% on prefill, NLL ratio 1.058 / 1.059. Metal still has no stream capture, so each decode step is dispatched kernel by
kernel instead of as one recorded graph.

The commands are in the README's *Get started*. Start the server from the source directory (or pass
`--chdir <source dir>`): the Metal shaders are compiled at startup from `metal/*.metal` by relative path, so a
binary copied elsewhere cannot find them.

---

## 8. Honest status and limits

- **What post-training (③) can do: write your material into the weights, answer it in new phrasings, keep
  general ability, mount and unmount at will.** A 24-line file produced a layer that answered 11 of 12
  questions in an unseen third phrasing (0 of 12 before), with eight general-knowledge answers unchanged.
  What it cannot do: make every exact number right from a few dozen lines, or teach a new skill — the
  literature and our runs agree that takes an order of magnitude more material (the rewrite expansion is
  how that price gets paid); many near-identical facts interfere, which expansion and keeping every line in
  training contain but do not cure. One real business file so far (§4.3); not yet scored on end-to-end
  business outcomes.
- **The post-training file published with the base** has been judged on held-out text and two long reasoning runs
  only. Two runs cannot show "matches the original", and the chance of closing jumps 2× between checkpoints (§4.3).
  The Studio's *Post-train* switch only lists ③ files trained on this machine; the one from Hugging Face is loaded
  with `--posttrain`.
- **Five domains.** Finance, code, law, medicine and science (§5.1), all five sidecars in the Hugging Face repository.
  All are judged teacher-forced on held-out text; none has been evaluated on end-to-end tasks (agentic coding, legal or clinical question answering).
- **We test end to end on one machine type.** Every number on this page is from a DGX Spark, except the Mac
  speeds in §5.2 and the Mac numerics check in §7, which a contributor measured on a 128 GB M5 Max. Our own Mac has 16 GB and only runs the
  kernel tests (see [Building and running on a Mac](#building-and-running-on-a-mac)).
- **Prefill on tensor cores is not bit-identical** to the fused scalar path (experts) or to the two-pass
  attention kernel (single-pass flash attention); both differences are at the rounding-noise level (§4.4).
