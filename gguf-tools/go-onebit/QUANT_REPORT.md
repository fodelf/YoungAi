# Quantizing DeepSeek V4 Flash to (near) 1-bit — an honest engineering report

*Target model: DeepSeek V4 Flash (MLA attention + hyper-connections + hash/top-k MoE,
256 routed experts, 43 layers, d=4096). Engine: DwarfStar (`ds4`), a bespoke native
inference engine for this one model family — no GGML. Hardware: 2× Apple Silicon 16 GB
(M4 mini + M1), Thunderbolt, ≤12 GB RAM budget per host.*

> **Read this first.** The headline you may have heard — *"1-bit quantization, 80 %
> restoration"* — is **not true for a deployable model**. That number came from a
> measurement harness running in a forgiving configuration. When the same idea is
> actually built into a GGUF and run in the real engine, perplexity explodes to
> **16 million** (pure garbage). This report exists to document *why* the 80 % was
> misleading, what actually works, and how we now measure it honestly. If you quantize
> MoE models, the failure modes below are the useful part.

---

## 1. The quantization scheme

DeepSeek V4 Flash is quantized **asymmetrically** — the routed MoE experts (≈ 72 GiB of
the 81 GiB model, i.e. almost all the weight) are aggressively quantized; the "backbone"
(attention projections, MLA compressor, hyper-connection functions, shared expert,
embedding, output head) is kept high-precision (FP8 / Q8).

Two block formats for the routed experts:

| Format | Bits | Block layout | Dequant |
|---|---|---|---|
| **GO1B** | ~1.06 | per 256-weight block: `2 B fp16 scale + 32 B signs` = 34 B | `scale · (bit ? +1 : −1)` |
| **GO2B** | ~2.1 | per-block 2-bit codes + scale | 4-level |

**The scale is the whole game.** A naïve per-row `mean|w|` (diagonal) scale is worthless.
The lever that makes 1-bit even remotely viable is a **joint least-squares per-block
scale**: for each output row, solve the `nblk × nblk` system `A·s = rhs` where
`A = Σ_b Pᵀ_b P_c`, `rhs = Σ_b Pᵀ_b Y`, with `P_tb = Σ_{j∈block} X_tj·sign(w_j)` and
`Y_t = Σ_j X_tj·w_j` over calibration activations `X`. This picks the per-block scales
that minimise the *output* error `‖X(s·sign(w)) − Xw‖²`, not the weight error.

- diagonal `mean|w|` scale → distribution restoration **0.24**, PPL **377**
- joint-LS scale + per-layer output gain → restoration **0.79**, PPL **12.30**

Everything is **closed-form, zero training** (the project constraint). Two optional
correction side-cars, also closed-form:

- **Per-layer output gain** `gx` — a scalar/vector re-scale of each layer's routed output.
- **`z` latent** — a per-layer low-rank correction: `y_corrected = y_base + U·diag(z)·Vᵀx`,
  solved as ridge-regularised **reduced-rank regression** on calibration pairs
  `(x = layer input, R = residual the quantized experts get wrong)`. The rank `k_L` is
  the per-layer quality/size dial, chosen by four closed-form losses
  (align / classify / smooth / fixed). Serialized as a ~28 MB side-car, hot-pluggable at
  runtime (`--corr`), never touches the base weights.

For `n_fit ≪ d` (few calibration tokens, d=4096) the `z` solve uses the **dual form**
`W = Xᵀ(XXᵀ + λI)⁻¹R`, moving the Cholesky from `d×d` to `n×n` — **~90 s → <1 s per
layer**, mathematically identical to the primal normal equations.

---

## 2. Quantization time / cost

| Step | Cost |
|---|---|
| Full 43-layer closed-form quantize + per-layer quality (C engine, real HF weights) | ~25 min (IO-bound on routed-expert reads) |
| `z` solve, primal `d³` Cholesky | ~90 s / layer (unacceptable) |
| `z` solve, dual `n³` form | <1 s / layer |
| GPTQ-style 2-bit (abandoned — not closed-form, ~3.5 h) | dropped for a closed-form path |

The dominant cost is **not compute, it's reading the fp8 routed experts off SSD** (each
expert ≈ 24 MB × 3 matrices; a forward touches 100–256 unique experts per layer). This is
also the deployment bottleneck (see §6).

---

## 3. Quality — and why the "80 %" was inflated

The 79–81 % figure was **distribution overlap `Σ min(softmax_fp, softmax_q)`** measured in
a configuration that is **not deployable**. Four independent inflators:

1. **It is not a 1-bit model.** Only routed experts are 1-bit; the entire backbone stays
   FP8. In the measurement, *fired-only* experts were quantized against an *exact* backbone.
2. **Teacher-forced, not autoregressive.** Both models see the true previous tokens at
   every position, so per-token errors never compound. Real usage is autoregressive.
3. **`Σ min` overlap is lenient.** 0.81 overlap ≠ 81 % chance of the same token; it rewards
   matching the huge low-probability tail over a 129 k vocab. Code needs the *argmax* exactly
   right (syntax).
4. **7 tokens.** No statistical meaning.

Strip the forgiving pieces and it falls apart:

| Configuration | Restoration | PPL |
|---|---|---|
| joint-LS scale + gain, fired-only, exact backbone (the "79 %") | 0.79 | 12.30 |
| …remove scale + gain (diagonal) | 0.24 | 377 |
| **deploy: all 256 experts quantized + quantized backbone + 1-bit *deep* layers** | — | **16,203,028** |

The gap between "12 PPL in the harness" and "16 M PPL deployed" is ~10⁶×. That is the
**"front–back disconnect"**: the harness measured a config the runtime never actually runs.

---

## 4. Test results (deployed ground truth — `test == deploy`)

Measured in the real `ds4` engine (`bench_ds4.sh`: engine PPL + generation), which is the
only honest signal:

| Model (all in the real engine) | Structure | PPL | Verdict |
|---|---|---|---|
| **ds4-code-dyn** (42.5 GiB) | **all 43 layers GO1B (1-bit)** | **16,203,028** | 💀 garbage |
| **mono-mixed** (~59 GiB) | GO1B shallow + **GO2B (2-bit) deep** | **6.47** | ✅ usable, solves `twoSum` correctly |
| diagonal-scale 1-bit | no joint-LS, no gain | (proxy 377) | ✗ |

Per-layer 1-bit cosine (routed-expert output, teacher-forced) ranged 0.81–0.9999, but
**relative-L2 stayed 0.38–0.52 on most deep layers** — cosine hides magnitude error, and
that magnitude error is what compounds and kills deployment.

**The single most important finding for a quantizer:** the difference between "usable" and
"16 M PPL" is *entirely* the **deep layers** — `1-bit deep = collapse`, `2-bit deep = fine`.
Same runtime, same backbone, same expert set; the only variable is deep-layer bit-width.

---

## 5. Stability

- **Deep-layer 1-bit is catastrophically unstable** in deployment (16 M PPL). Deep layers
  must be ≥2-bit. Shallow layers tolerate 1-bit (with joint-LS coding-imatrix scale + gain).
- **Memory / OS:** running the full mono model as a single-host offload triggered a **macOS
  kernel panic** (an OS VM pre-read bug wires the whole mapping faster than a user-space
  watchdog can react). Safe path = **dual-host layer slicing + external RSS watchdog**, hard
  red line 12/12 GiB per host; startup must refuse over budget.
- **Numerical auto-arming:** GO-type models need engine modes armed automatically
  (math-safe accumulation, raw-F32 KV, rope/repeat-frequency); a stray `MOE_THIN_TOPK=4`
  (dropping experts by count) alone produced junk output. When quality collapses, the speed
  numbers are meaningless — always gate on quality first.
- Quantization is **deterministic / reproducible** (fixed-seed LCG in the `z` solve, fixed
  accumulation order), so a re-solve is byte-identical.

---

## 6. Speed

Dual-host, 2× 16 GB, ≤12 GB budget each, routed experts streamed from SSD on demand:

- **mono-mixed decode: 1.6–1.7 t/s** (correct `twoSum` output).
- Best experimental code-edit path: ~3.5 t/s (copy-speculation + prefix reuse).
- **Physical walls:** combined RAM (24 GB) < experts (72 GiB) → must stream; routed-expert
  IO ≈ **1.70 GiB/token**; SSD-bound decode ceiling ≈ 2–7 t/s. **80 t/s is physically
  impossible** on this hardware (would need ~750 GB/s aggregate + full residency). Honest
  target is single-digit t/s; the win comes from workload-level tricks (copy-speculation,
  prefix reuse), not from the raw forward.

Quantization's job here is **capacity** (fit the model at all), and the 1-bit-vs-2-bit
choice trades size against the deep-layer collapse above — not throughput, which is
IO-bound regardless.

---

## 7. The redesigned benchmark: Claude Code as ground truth

We are retiring the teacher-forced `Σ min` harness as a *quality verdict* (it is fine only
as a cheap per-layer smoke signal). The benchmark that decides ship/no-ship is now:

1. **Deploy** the quantized GGUF (+ `z` side-car) in `ds4-server` (dual-host, memory-safe).
2. Point a real **Claude Code** client at it (`ANTHROPIC_BASE_URL` → local server,
   Anthropic `/v1/messages` compatible).
3. Run a **real coding task** (read a file → edit a function → run tests) and judge whether
   it completes without human patching.
4. Quantitatively, back it with **engine perplexity + autoregressive generation match** on
   real chat-format coding transcripts (assistant tokens only), reported as PPL ratio and
   exact greedy-token agreement — *not* tail-overlap.

This guarantees `test == deploy`: the numbers come from the same engine, weights, and
prompt distribution the user actually experiences.

---

## 8. Recommendations (if you quantize MoE like this)

1. **Never trust teacher-forced + tail-overlap metrics for sub-2-bit.** They inflate by
   orders of magnitude. Measure deployed PPL + autoregressive behaviour.
2. **Output-optimal per-block scale (joint-LS) is mandatory** below 2-bit. Diagonal `mean|w|`
   is a non-starter (377 vs 12 PPL).
3. **Mixed precision by depth.** Shallow layers → 1-bit OK; deep layers → 2-bit minimum.
   A flat 1-bit MoE is not deployable.
4. **Closed-form correction side-cars** (`z` low-rank RRR + per-layer gain) recover real
   error at ~28 MB, hot-pluggable, zero training — but they do **not** rescue deep-layer
   1-bit collapse; fix the bit-width first, correct second.
5. **Prove memory safety before every load.** On unified-memory Apple Silicon a bad offload
   panics the kernel; budget + watchdog are not optional.

*All quantizer/forward code is closed-form C: `onebit_quant.c` (joint-LS),
`ds4quant_fwd.c` (bit-exact reference forward), `ds4_z.c` (dual-form RRR), `st_read.c`
(fp8 E4M3 reader), `bench_ds4.sh` (deploy-equal benchmark).*
