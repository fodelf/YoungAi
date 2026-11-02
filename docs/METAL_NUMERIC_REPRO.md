# Metal upload ordering and verification parity

This change is separate from PR #6's memory/lifetime fixes. `fable5.md` remains
maintainer-owned. Keep detailed timing logs and model outputs outside the tree.

## Root cause and upload audit

`v41_compress_source` writes every source layer's group positions into the same
`st->posg`, then queues RoPE readers. Normal single-token decode encodes all layers
in one batch. On the 1017 model, sources L2/L8/L14 have ratio 2 and L20 has ratio 1:
at odd position p, the former require p-1 and the latter writes p. The old Metal
zerocopy wrapper immediately overwrites the shared bytes before prior readers run.
The model-free `plain decode posg RoPE` probe reproduces this without DSpark.

Metal tensor writes now snapshot the host bytes before returning and enqueue the
copy after prior readers. Inputs must be uploaded before encoding their consumers;
end/synchronize the batch before CPU reads through `contents()`. Idle writes remain
immediate. Pages append disjoint 256-byte-aligned slices and reset only after their
last command buffer completes or errors. At most sixteen 1 MiB pages are retained;
large/overflow uploads are retained only by their command buffers. Cleanup releases
the pool after draining. No callback touches pool state.

Audited V4.1 call families (including async/zerocopy wrappers):

| Call site | Ordering and source lifetime |
| --- | --- |
| `core_v41_forward.c`: tok, pos, vrows, pre_mix | Written before begin/body; CPU arrays may be freed after the snapshot. |
| `core_v41_attn.c`: posg | Written before both RoPE readers; later layers reuse the destination. This is the reproduced hazard. |
| `core_v41_engram.c`: eraw | Per-layer upload precedes dequantization; pinned CPU data lives through the forward. |
| `core_v41_draft.c`: slots and cache initialization | Initialization precedes readers; queued snapshots preserve each successive slot value. |
| `core_v41_draft_amp.c` / `core_v41_amp.c`: adapters | Load/concatenate fresh adapter tensors before attaching or using them; host load buffers may then be reused. |
| `core_v41_multi.c` / `core_v41_mgraph.c`: batch inputs | Written before the batch/body; Metal capture is unavailable and uses direct dispatch. |
| `core_v41_vision.c`: weights, patches, image rows | Written before projection/selection; image delimiter rows are disjoint from queued image copies; synchronize before publication. |

No V4.1 caller intentionally encodes an input reader before its upload or accesses
a queued destination through `contents()`. The API comment states the ordering
contract so new callers cannot depend on late host writes.

## Arithmetic and model-free validation

The Q4_K verification kernel shares packed weights/scales across 2–8 rows and uses
the single-row lane/block traversal and SIMD reduction. Reassociation is disabled
locally in both Q4_K kernels, with within-expression FMA contraction enabled: otherwise compiler choices can change the result with NT
even when the source expression is identical. Packed weights/scales and the current
activation row are shared across the two output rows; activation arrays do not grow
with NT, avoiding register spills at larger batches.
This may change baseline single-row rounding and requires the model-level gate.
The skip flag remains grid-wide (`skip[0]`).

Attention receives split size/cap from the host. `ratio > 0` explicitly selects
causal per-row visibility; `ratio == 0` preserves frozen index fixtures, independently
of the half-prefill precision switch. Metal replay and frozen backward fixtures
request zero; CUDA replay retains its existing ratio.

```sh
make -j6 ds4 ds4-server ds4_test
make linecount
./ds4_test --metal-upload --metal-v41 --bwd-attn --server --engine-units --rax --tp-allreduce
make test
```

The upload suite checks batch/pending order, real compressor-position RoPE,
stack-source snapshots, page rollover/reuse, tensor views and unaligned copies.
Q4_K exact comparisons cover 256/1024/5120 input dimensions, 1/4 groups, rows 2–8,
rounded/unrounded outputs and output strides. Attention checks ratios 1/2/4/128,
early/split/compressor boundaries, long positions and both precision settings.
Existing primitives also compare against scalar reference math and finite differences.

## Output-limit checkpoint regression

Verification may accept more tokens than the remaining output budget. Previously
all accepted tokens entered KV even though emission stopped at `max_tokens`; the
end checkpoint then contained an un-emitted suffix and the next turn fell back to
the prompt checkpoint. On Metal this also selected different prefill arithmetic.
Cap the retained prefix before rollback and checkpoint bookkeeping. This logical
limit applies to both backends; it does not change already emitted tokens.

Against a running DSpark-enabled server, the short regression fits the baseline:

```sh
python3 tests/repro/metal_cache_limit.py --url http://127.0.0.1:19010 --out /tmp/cache-limits
```

It checks limits 1–8, 15, 16 and 31, requiring the follow-up to reuse exactly the
prompt plus emitted tokens. Run it on CUDA as well; local execution used Metal.

## Performance and full-model gates

Build the same bounded microbenchmark on each revision and run sequentially on an
idle GPU. Its matrix is 5120 by 32768 (90 MiB packed Q4_K); compare medians per row
count. This measures kernel time including submission, not end-to-end DSpark speed.

```sh
make -f Makefile -f tests/repro/metal_q4_bench.mk -j6 ds4-metal-q4-bench
./ds4-metal-q4-bench
```

For end-to-end tests, use the same 1017 model, sidecar, posttrain, Engram, token IDs,
greedy settings, 110000 MiB budget and memory watchdog. Use a short prompt for
baseline DSpark so it fits; compare repeated cold/cache-hit requests. Apply the
separate memory PR when testing combined 8K behavior. Record output hashes,
acceptance and decode speed. `startup` logs separate prefill, pre-draft work and
draft initialization; the server's `prompt_ms` includes time through first emission.
The earlier raw logs show 17.1 s vs 12.3 s inside prefill, not draft initialization.

Compare fixed teacher-forced IDs at chunks 1 and 8 and run upload-only/arithmetic-only
ablations. With the same model and optional sidecar/posttrain/Engram arguments:

```sh
for chunk in 1 8; do
  ./ds4 --metal -m "$MODEL_PATH" --mem-budget-mb 110000 \
    --score-ids tests/repro/metal_score_ids.txt --v41-chunk "$chunk" \
    --score-out "/tmp/metal-score-$chunk.logits" \
    --score-nll "/tmp/metal-score-$chunk.nll" \
    --score-topk 20 "/tmp/metal-score-$chunk.topk"
done
cmp /tmp/metal-score-1.logits /tmp/metal-score-8.logits
```

Batch agreement is not independent quality validation. The inherited
`caliper_ref.sh` expects a 43-layer V4 reference and `--logprob-vectors` uses V4
fixtures. The local CPU reference now runs the pinned official V4.1 forward with the same
quantized weights; see [METAL_REFERENCE_REPRO.md](METAL_REFERENCE_REPRO.md) for
setup, scope and measured results. The ranking of PR #7 versus baseline reverses
between the official bf16 embedding and exact Q4_K embedding policies. Neither
comparison reaches the upstream same-weight KL target. The numerical merge gate remains open; completed runs, lower fixture NLL
and exact chunk agreement must not be reported as an independent parity pass.

## Follow-up: FP4 midpoint parity

The independent CPU trace exposed a separate compressed-KV quantizer defect.
Fast division by an E4M3 scale can move exact FP4 halfway values across the
round-to-nearest-even boundary. Activation quantization and packed-KV encoding
now compare values directly against scaled midpoints; E4M3 scale selection uses
the same approach instead of dividing the block maximum by six. This preserves
fast math elsewhere and removes the problematic division from these small kernels.
Exhaustive scale/midpoint regressions fail before the fix and pass afterward.
See [the precision-isolation record](METAL_REFERENCE_REPRO.md#precision-isolation-2026-11-01-utc)
for the real-model reproduction, reference-policy controls, and remaining gates.
