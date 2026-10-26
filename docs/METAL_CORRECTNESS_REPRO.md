# Metal DSpark correctness and request resource lifetime

This change targets ordered uploads, single-row versus verification-batch arithmetic,
and bounded request workspace lifetime. It keeps the existing prefill chunk policy
(default 2048), model context limit, DSpark algorithm, and sampling policy. It does
not add adaptive chunking, OOM retry, image routing, or deployment-specific settings.

## Failure mechanisms and changes

* A CPU `memcpy` into a shared tensor could overwrite input still referenced by an
  earlier queued GPU operation. Snapshot uploads into retained staging buffers and
  copy them on the same queue whenever a batch or pending command buffer exists.
  Synchronous writes without queued work keep the direct copy path.
* Q4_K decode and verification used different accumulation orders. Dispatch the
  single-token arithmetic independently for every verification row. Sparse causal
  attention now partitions each row using its own visible key count, including
  transitions between one and two partitions. Training/noncausal fixtures retain
  their existing key visibility contract.
* C server worker threads lack an outer Objective-C autorelease pool. Scope tensor
  access/copy operations so temporary buffer references do not survive the request.
  GPU command buffers and the transient list retain uploads until completion.
* Metal cache hits now size working rows from the uncached suffix before allocating;
  the context and compressed KV capacities still cover the full request. Failed
  checkpoint copies drain and discard partial state before a cold allocation.
* After prefill selects the first token, Metal shrinks temporary working rows before
  allocating DSpark state. KV, compressor history, window history, and main hidden
  history are preserved. Reuse existing Engram raw buffers instead of overwriting
  their owners. Close failed forward batches and synchronize before request cleanup.

The new allocation/shrink policy is enabled only for the Metal single-request path.
Shared forward cleanup and the Engram ownership fix also compile into CUDA; CUDA
hardware validation is still required before treating this as merge-ready.

## Test environment

Recorded on 2026-10-25–26 (America/Los_Angeles), M5 Max MacBook Pro, 18 CPU cores,
40 GPU cores, 128 GiB unified memory; macOS 26.6.2 (25G83). Baseline:
`8bf93945cc352195cdda75ae496801d6f4a63099` (`restructure`). Baseline production code
is unchanged; only parent-compatible test probes are installed.

Model: `DeepSeek-V4.1-Flash-1017-vq8sh14-q4k-mtpnative.gguf` (113,556,639,424 bytes;
verified SHA-256 `0da4ae2d7528ae6a60153ee072ddb4680b9d36f1c8f0b5915cdd950e89de5070`),
GRRB `code_fit_n15360-engine` sidecar and `posttrain-code-20261022`, with the original
Engram shards on SSD. Text-only tests explicitly disable vision. Both builds use
`--batch 0`, context 1,048,576 from model metadata, 32,768 server output cap, and
`--mem-budget-mb 114000`. API fixtures use greedy decoding, seed 42, no DRY penalty;
q1–q4 use seed 1 and a 2048-token generation cap. These are bounded local regression
samples, not an official model-quality benchmark or a complete 1M-token stress test.

The independent watchdog samples every 0.5 seconds. Limits: process RSS/footprint
120 GiB, system wired 120 GiB, swap use **or cumulative swap-out** growth 8192 MiB,
compressor growth 24 GiB, warning pressure for 1800 seconds; critical pressure or
failed memory probes stop immediately. Model-free tests use an 8 GiB process cap.
Only one full model is loaded at a time. RSS alone excludes much GPU-resident memory;
system wired memory and Metal allocation counters are also reported.

## Reproduce model-free regressions

Run from the checkout root (Metal shader paths are relative). Python 3 is used only
for external process/API orchestration and report serialization; verdict math stays
in the existing native engine and C tests.

```sh
make -j6
make -j6 ds4_test
make linecount
cc -O2 tests/repro/metal_memory_probe.c -o /tmp/youngai-memory-probe
python3 tests/repro/metal_guard.py \
  --probe /tmp/youngai-memory-probe --out /tmp/youngai-fixed-unit \
  --process-gib 8 --wired-gib 120 --swap-mib 8192 --compressed-gib 24 -- \
  ./ds4_test --metal-tensor-lifetime --metal-v41-memory --metal-v41 \
  --server --engine-units --rax --tp-allreduce
```

To prove that the regressions detect the old behavior, create a clean baseline
worktree, then install **only tests** using the PR checkout's helper:

```sh
git worktree add --detach /tmp/youngai-baseline 8bf93945cc352195cdda75ae496801d6f4a63099
python3 tests/repro/metal_baseline_probes.py /tmp/youngai-baseline
PR_CHECKOUT="$PWD"
cd /tmp/youngai-baseline
make -j6 ds4 ds4-server ds4_test
python3 "$PR_CHECKOUT/tests/repro/metal_guard.py" \
  --probe /tmp/youngai-memory-probe --out /tmp/youngai-baseline-unit \
  --process-gib 8 --wired-gib 120 --swap-mib 8192 --compressed-gib 24 -- \
  ./ds4_test --metal-tensor-lifetime --metal-v41 --server --engine-units --rax --tp-allreduce
```

Expected baseline failures: overwritten upload bytes (9 instead of 1), cumulative
Metal allocations (+256 MiB across 32 warmed cycles), Q4_K batch parity and causal
attention boundary parity. The single-row attention reference supplies independently
bounded keys with inference visibility filtering disabled, so a host/shader ratio
ABI mismatch cannot pass merely by dropping keys identically in both paths. The cache/workspace suite is PR-only because it uses
new internal allocation helpers. It exercises exact/+5-token hits at prefix lengths
70,754, 524,288 and 1,048,560, failed-copy cold fallback, decode shrink preserving
recurrent state, and eight failed forwards followed by new command batches.

These large-prefix tests use a small synthetic model shape; they validate indexing
and capacity contracts, **not** real-model quality or throughput at those lengths.

## Reproduce real-model API tests

Stop any existing full-model instance first. Supply absolute local paths; no model
weights, private requests, caches, or virtual environments are included in this PR.
Use a fresh process for each build and DSpark setting. Start this in one terminal:

```sh
python3 tests/repro/metal_guard.py \
  --probe /tmp/youngai-memory-probe --out /tmp/youngai-server-run \
  --process-gib 120 --wired-gib 120 --swap-mib 8192 --compressed-gib 24 -- \
  ./ds4-server --metal -m "$MODEL_PATH" --zchain "$ZCHAIN_PATH" \
  --engram-dir "$ENGRAM_PATH" --posttrain "$POSTTRAIN_PATH" \
  --mem-budget-mb 114000 --host 127.0.0.1 --port 19010 \
  --batch 0 --max-output-tokens 32768 --vision-gguf off
```

Add `--no-dspark` for the off arm. In another terminal, run sequentially:

```sh
python3 tests/repro/metal_api.py --url http://127.0.0.1:19010 --out /tmp/youngai-quick --suite quick
python3 tests/repro/metal_api.py --url http://127.0.0.1:19010 --out /tmp/youngai-questions --suite questions
python3 tests/repro/metal_api.py --url http://127.0.0.1:19010 --out /tmp/youngai-lifetime --suite lifetime
```

The quick suite repeats two synthetic 8192-token prompts with 128 output tokens;
its second request checks exact prefix reuse and output determinism. Compare
`all_sha256` across arms; it covers reasoning plus a NUL separator plus content.
Inspect request usage/metrics as well: an empty/incomplete stream is a failure.
The lifetime suite performs a roughly 70K-token cold request, 12 short follow-ups,
a longer prefill cancelled by disconnecting, and a recovery request. It records
actual server token counts, reuse, cancellation time, and all completed streams.
After completion, compare `request cleanup` allocations in `stdout.log`, allowing
checkpoint size growth; compare `memory.jsonl` against the launch baseline.

## Teacher-forced logits and broader checks

The inherited `--logprob-vectors`/`--dump-logprobs` track targets V4 graph sessions
and V4 continuation fixtures. It is not an independent V4.1 golden oracle. For this
model use the existing native V4.1 scorer with the fixed 128-token fixture. Stop the
server before invoking each scorer under the same watchdog:

```sh
./ds4 --metal -m "$MODEL_PATH" --zchain "$ZCHAIN_PATH" \
  --engram-dir "$ENGRAM_PATH" --posttrain "$POSTTRAIN_PATH" \
  --mem-budget-mb 114000 --vision-gguf off \
  --score-ids tests/repro/metal_score_ids.txt --v41-chunk 1 \
  --score-no-logits --score-nll /tmp/youngai-score.nll \
  --score-topk 20 /tmp/youngai-score.topk
```

Repeat with chunk 8, and with baseline and candidate. Record the native NLL summary,
compare NLL/top-k files, and keep the input IDs fixed. This tests single-row/batched
forward consistency without using a different tokenizer or regenerating continuations.
It does not replace an official V4.1 reference-logit quality evaluation.

Also run `make test`, `./ds4_test --all`, and `make cpu` on both checkouts. Do not
hide inherited failures or call model-dependent skips passes. All raw reports
should be kept outside the repository; publish only the compact sanitized evidence.

## Results

[Machine-readable results](metal-correctness-results.json) include every completed
API case, output hash, actual reuse/token counts, native score-file hashes, memory
peaks, and inherited failure diagnostics. No private user prompts are included.

| Check | Baseline | Candidate |
| --- | --- | --- |
| New upload/lifetime/batch regression assertions | 24 failures | 0 failures |
| Metal buffer growth over 32 warmed allocation/free cycles | +256 MiB | 0 MiB |
| 128-token scorer: chunk 1 versus chunk 8 NLL/top-20 files | differ | byte-identical |
| Native mean NLL, chunk 1 / chunk 8 (127 predictions) | 1.9667 / 1.9638 | 1.9611 / 1.9611 |
| 8192-token DSpark request | GPU command-buffer OOM after prefill | completes |
| `make test` / `ds4_test --all` | 29 failures | 5 inherited failures |

`--metal-tensor-lifetime --metal-v41-memory --metal-v41 --server --engine-units
--rax --tp-allreduce` exits 0 on the final candidate. The five remaining failures
are the same go1b/go2b offload fixture assertions in `tests/t_metal_kernels_moe.c`
(lines 221, 258, 259, 374, 403) on both builds. Other model-dependent suites skip
without the default V4 GGUF. `make cpu` fails on both builds in unchanged
`src/cli/cli_opts.c`: missing declarations for `ds4_gpu_set_model_cache_limit_mb`
and `ds4_gpu_v41_set_indexer_mma`. Neither failure set is presented as a pass.

For the **same output** on the fixed build, median decode speed across the four
8K cases is **8.4 token/s without DSpark versus 13.7 with DSpark (1.63x)**. These
are 128-output-token samples, two prompts repeated twice, one cold and one cached
request each; not a broad throughput benchmark. Cold first-token latency was
12.33/13.08 seconds without DSpark and 17.13/17.38 with it (Chinese/code). Cached
first-token server timings were 5.5–6.5 ms. Cold first-token latency includes
request setup and draft initialization and is reported separately from decode.
The unmodified baseline without DSpark measured 8.3 token/s; its DSpark OOM arm
has no valid throughput comparison.

All five quick cases (warmup plus four measured runs) match byte-for-byte between
DSpark on/off on the candidate. A separate ablation build with just the Metal
upload/arithmetic changes also matches the candidate without DSpark on all five
cases: the core workspace/reclamation changes do not change those outputs.
Outputs do differ from the unmodified upstream build; this PR changes floating
point execution/causal partitioning and fixes queued input overwrites. We do not
claim exact upstream-output preservation or a general model-quality improvement.

The q1–q4 API regression uses the repository's questions and a 2048-token cap:

| Question | Baseline without DSpark | Candidate with DSpark |
| --- | --- | --- |
| q1 | B, 577 tokens | truncated at 2048, no final answer |
| q2 | C, 296 tokens | C, 321 tokens |
| q3 | 70, 384 tokens | 70, 459 tokens |
| q4 | truncated at 2048, no final answer | C, 412 tokens |

Thus this small check is **not an unqualified quality pass**: both arms answer
three questions correctly under the cap, but truncate different questions.
Candidate q1–q4 output also matches the previously recorded fixed deployment
byte-for-byte. A larger independent V4.1 reference-logit/quality evaluation and
CUDA runtime validation remain review requirements; keep the PR draft meanwhile.

The 73,696-token cold prompt and all 12 continuations complete. Continuation
workspaces use 10 or 22 rows according to the uncached suffix instead of 2048;
post-request Metal allocation spans only **0.12 MiB** across those continuations
(checkpoints legitimately grow). Disconnecting a 139,188-token prefill returns
to idle in **3.67 seconds**, after processing 2048 tokens; the next request succeeds.
Peak system wired memory in this combined candidate run is **112.76 GiB**;
pressure stays normal and there is no swap-use or cumulative swap-out growth.
The watchdog's kill path was separately checked with a 64 MiB disposable process
and a 20.48 MiB process cap. No production memory limit was relaxed.

No full-model 512K/1M run was performed for this PR. The synthetic large-prefix
fixtures establish capacity/restore behavior only. The original deployment was
restored after these serial tests with its binary and settings unchanged.
