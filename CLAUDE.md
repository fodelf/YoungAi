# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**DwarfStar** (`ds4`) is a self-contained native inference engine **purpose-built for DeepSeek V4 Flash** (and, on very high-memory machines, DeepSeek V4 PRO). It is *not* a generic GGUF runner and does **not** link against GGML — it reimplements the loading, tokenizer, prompt/DSML rendering, KV cache, graph scheduling, server API, and a native coding agent for this one model family. It only runs the DeepSeek V4 GGUFs published for this project (asymmetric quant: routed MoE experts at `IQ2_XXS`/`Q2_K`, everything else left high-precision). See `README.md` for the full feature tour and `MODEL_CARD.md` for the model.

Primary backend is **Metal on macOS**; **CUDA on Linux** is the second production path; the **CPU path is reference/debug only**.

## Build

The build is a single hand-written `Makefile` (no CMake/configure). Output binaries land in the repo root.

```sh
make              # macOS: builds Metal ds4, ds4-server, ds4-bench, ds4-eval, ds4-agent (default target)
make cuda-spark   # Linux: CUDA for DGX Spark / GB10 (HBM weight cache, no explicit -arch)
make cuda-generic # Linux: CUDA for a generic local GPU (CUDA_ARCH=native)
make cuda CUDA_ARCH=sm_120   # Linux: CUDA with an explicit nvcc arch
make cpu          # CPU-only reference/debug build (adds -DDS4_NO_GPU)
make clean
```

- On **Linux**, plain `make` only prints the target list — you must pick a `cuda-*`/`cpu` target explicitly.
- C is **C99**, `-O3 -ffast-math`, native CPU arch. GPU kernels are separate: `ds4_metal.m` (Objective-C, `-fobjc-arc`) compiles `metal/*.metal`; `ds4_cuda.cu` is compiled by `nvcc`.
- The shared host core is `CORE_OBJS = ds4.o ds4_distributed.o {ds4_metal.o | ds4_cuda.o}`. The CPU build swaps `ds4.o` for `ds4_cpu.o` (same `ds4.c`, `-DDS4_NO_GPU`).

## Run

```sh
./download_model.sh q2-imatrix   # fetch a model into ./gguf/ and point ./ds4flash.gguf at it
./ds4 -p "Explain Redis streams" # one-shot;  no -p => interactive REPL
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
./ds4-agent                      # in-process native coding agent (sessions in ~/.ds4/kvcache)
```

`./ds4flash.gguf` (a symlink) is the default model for every binary; pass `-m gguf/<file>` to override. Use `--metal`/`--cuda`/`--cpu` to force a backend, `--chdir /path/to/ds4` when launching from elsewhere so `metal/*.metal` resolves.

## Test

Read `CONTRIBUTING.md` before changing inference code — it defines the correctness and speed regression tracks. Tests require a real model + working backend.

```sh
make test                      # builds + runs ./ds4-eval --self-test-extractors and ./ds4_test (all suites)
./ds4_test --server            # API parsing, chat/DSML rendering, streaming, KV-disk bookkeeping (best quick check)
./ds4_test --logprob-vectors   # token bytes vs official DeepSeek vectors — catches tokenizer/template/attention/logit drift
./ds4_test --long-context      # long-context fact-recall regression
./ds4_test --tool-call-quality # DSML tool-call emission (fast + exact paths)
./ds4_test --metal-kernels     # isolated Metal kernel numeric checks
make cuda-regression           # Linux/CUDA only: tests/cuda_long_context_smoke
```

Override test inputs with env vars: `DS4_TEST_MODEL`, `DS4_TEST_VECTOR_FILE`, `DS4_TEST_LONG_PROMPT`.

For generation-drift-sensitive changes, also run the deterministic q1..q4 eval gate (expected token counts are in `README.md` → Capability Evaluation):

```sh
./ds4-eval -m ds4flash.gguf --plain --questions 4 --tokens 2048 --temp 0 --seed 1
```

Speed regressions use `ds4-bench` (instantaneous prefill/gen t/s at context frontiers, not a whole-run average):

```sh
./ds4-bench -m ds4flash.gguf --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 2048 --gen-tokens 128 --csv /tmp/ds4-speed.csv
```

Quantization/GGUF changes are scored with `gguf-tools/quality-testing` (`make -C gguf-tools quality-score`, then `score_official` + `compare_scores.py`; lower `avg_nll` is better).

## Architecture (the cross-file picture)

The codebase is flat C in the repo root. The pieces that need reading several files to understand:

- **Backend abstraction.** `ds4_gpu.h` is the GPU tensor/graph interface. `ds4.c` owns everything host-side: GGUF mmap loading, tokenizer, prompt/DSML rendering, sessions, graph scheduling, and disk-cache payload serialization. The interface is implemented by `ds4_metal.m` (Metal) or `ds4_cuda.cu` (CUDA); `-DDS4_NO_GPU` selects an in-`ds4.c` CPU reference. Model weights are **mmap-backed and never eagerly copied** — Metal wraps mmap regions as no-copy `MTLBuffer` views and uses an `MTLResidencySet` to budget GPU VM.

- **Compressed KV cache is a first-class *disk* citizen.** DeepSeek V4's KV is heavily compressed (raw sliding window + ratio-4 indexer-selected + ratio-128 compressed layers), which makes long context viable locally. `ds4.c` serializes the `DSV4` session payload; `ds4_kvstore.c`/`.h` manage the on-disk KV cache (`<sha1-of-rendered-prefix>.kv`, written with plain read/write — *not* mmap — to avoid adding VM mappings). The server keeps exactly **one live in-memory checkpoint**; the disk cache is the resume mechanism across sessions/restarts. The file format (48-byte `KVC` header, rendered text, `DSV4` payload, optional `KTM` tool-id map) is documented in `README.md` → Disk KV Cache.

- **Server (`ds4_server.c`).** OpenAI/Anthropic/Responses-compatible HTTP (`/v1/chat/completions`, `/v1/responses`, `/v1/messages`, ...). Inference is serialized through one graph worker (no request batching). Key subtlety: stateless clients resend JSON tool calls, so the server keeps an **exact-DSML replay map** (`tool id -> exact sampled DSML bytes`, radix-tree backed via `rax.c`) so re-rendered prompts still byte-match the live KV checkpoint; canonical JSON→DSML rendering is only the fallback. During tool-call *syntax* the server forces `temperature=0`; argument payloads use normal sampling.

- **Distributed inference (`ds4_distributed.c`/`.h`).** Runs a model too large for one host by **slicing transformer layers** across machines (`--role coordinator|worker`, `--layers 0:19` / `20:output`). Custom binary TCP protocol (`HELLO`/`WORK`/`RESULT`/snapshot frames); activations flow worker→worker assembly-line, the coordinator owns tokenization/sampling. Prefill is pipelined (speedup); generation stays autoregressive (one cross-machine hop per token, so slower). A rolling 64-bit token-prefix hash on every work item guards against stale worker KV. Saved sessions use the same single-file `DSV4` payload (topology-neutral).

- **Native agent (`ds4_agent.c`).** Drives inference in-process (no socket boundary); the session *is* the on-disk KV cache. `/list`, `/switch <sha>`, `/save`, `/del`, `/strip`.

- **MTP / speculative decoding.** Optional draft-model path (`--mtp FILE --mtp-draft N --mtp-margin F`), greedy-only, confidence-gated, currently experimental (slight speedup at best). Draft state is not persisted across disk-checkpoint loads.

- **Other root files:** `ds4_cli.c` (CLI + linenoise REPL), `ds4_bench.c`, `ds4_eval.c` (embedded 92-item capability suite), `ds4_web.c` (agent web fetch), `linenoise.c`, `rax.c` (radix tree). `gguf-tools/` is an offline sub-project (HF→GGUF quantizer, imatrix collection, quality scorer) with its own `Makefile`. `dir-steering/` holds single-vector activation-steering data/tools.

Behavior has many `DS4_*` env switches (e.g. `DS4_METAL_PREFILL_CHUNK`, `DS4_METAL_NO_RESIDENCY`, `DS4_DIST_*`, `DS4_MTP_SPEC_DISABLE`). Treat them as diagnostic/tuning switches around the single release path, not permanent feature flags.

## Active performance work: dual-host q2 full-model streaming (`project.md`)

The current dominant effort is running the **81 GiB q2 full model** (`gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`) across **two 16 GB Macs** (M4 mini coordinator + MacBook M1 worker, Thunderbolt direct, both budgeted **≤12 GiB**) via layer-sliced distributed inference + **A3 on-demand expert streaming from SSD**. `project.md` is the canonical plan (P0 measurement → P1 memory-layout → P2 algorithm → P-Code coding-workload specials); `notes/execution-log.md` is the running record. **Read both before touching this path.**

- **The bottleneck is routed-expert SSD IO (~1.70 GiB/token), not KV or compute.** Three physical walls bound a single forward: W1 capacity (72.6 GiB experts vs ~24 GiB combined RAM), W2 SSD bandwidth (decode ceiling ~3–7 t/s), W3 backbone bandwidth (10–18 t/s). **80 t/s is physically impossible here** (needs ~750 GB/s aggregate + full residency); the honest target is **0.81 → ~3–6 t/s**. Best so far: **code-edit 3.54 / smoke 2.17 t/s** (wave 36). Coding-workload "effective t/s" can exceed the single-forward wall via copy-speculation and prefix reuse (§3.5 / P-Code).

- **Execution discipline.** Every patch / measurement / scope decision / blocker gets appended to `notes/execution-log.md` (wave-numbered) — not just chat or memory.

- **Test harness.** `tools/mtp_pipe_q2_speed.sh` drives the two-host run. Two prompt profiles: default **smoke** (short single question, general baseline) and **`PROMPT_PROFILE=code-edit`** (editing-type long-context prompt — the only profile where PC.1/PC.2 gains show); run both. Logs: `/tmp/mtp_pipe_coord.log|.out`, worker `192.168.1.2:/tmp/mtp_pipe_worker.log`. New knobs go through env and **default to the current stable baseline** so each is A/B-able.

- **Live tuning levers** (env, around the single path — see `project.md` §progress for landed state): `DS4_METAL_EXPERT_IO_PROFILE` (gather fault/memcpy/pread/drain timing), `DS4_METAL_EXPERT_EVENT_DRAIN` (MTLSharedEvent fast-path host wait), `DS4_METAL_EXPERT_PREAD` (+`_NOCACHE`, single-copy direct read vs mmap+memcpy), `DS4_METAL_EXPERT_FULL_LAYER_STREAM` (+`_SORT_IDS`, prefill full-layer sequential stream), `DS4_METAL_EXPERT_PREFETCH_AHEAD` (+`_TOP`/`_DELTA`, cross-layer router-prediction prefetch), `DS4_METAL_EXPERT_STAGE` / `DS4_DIST_EXPERT_FETCH_*` (predictive remote expert staging / byte service), `DS4_DIST_COPY_SPEC` (+`_DRAFT`/`_NGRAM`/`_GROWTH`, PC.1 zero-cost copy-speculation: n-gram matcher as drafter, reusing the VERIFY/`accept_len`/KV-rollback protocol), `DS4_DIST_PREFILL_CAP`.

- **Guardrails (hard).** Watchdog red line **12/12 GiB** (both hosts); `DS4_MEM_BUDGET_MB` + L1 resident gate must refuse startup over budget. **Any model-loading script must prove memory safety (RSS budget + watchdog) before it runs.** Never double-load the 81 GiB base on one host. After a local rebuild, consider whether the other host needs the synced binary too (shared `CORE_OBJS`). Never delete files on the other host without explicit confirmation — "cleanup" means killing the process, not removing files.

- **Per-merge correctness gates** (on top of the speed track): `--dump-logprobs` parity vs the A3 baseline; `ds4_test --metal-kernels --server`; routing/quant/repack changes additionally run `ds4_test --logprob-vectors` + `ds4-eval q1..q4 --temp 0 --seed 1`.

## Project rules (from `AGENT.md` — follow these)

- **No C++.** Pure C99, with Objective-C only where Metal requires it.
- **Keep model loading mmap-backed**; do not eagerly copy the full GGUF. Keep the production path whole-model Metal/CUDA graph inference.
- **Correctness before speed.** Don't keep a faster path with unexplained attention/KV/logit drift; a speed regression is only acceptable when it fixes a real correctness bug.
- **Comment the non-obvious inference mechanics** (shapes, cache lifetime/boundaries, memory policy) inline, not in separate design docs. Keep CLI/server code free of tensor internals.
- **macOS CPU danger:** running the CPU inference path on macOS can crash the kernel (an OS VM bug) — avoid large CPU runs there; CPU is for reference/diagnostics.
- **Instance lock is intentional:** do not run multiple huge-model processes concurrently.

## Debugging

```sh
./ds4 --dump-tokens -p "..."                              # tokenize (incl. DS4 specials) and exit
./ds4 --dump-logprobs /tmp/out.json --temp 0 -p "..."     # greedy continuation + top-k alternatives
./ds4-server --trace /tmp/ds4-trace.txt ...               # log rendered prompts, cache decisions, tool events
```
