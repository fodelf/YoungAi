# Agent Notes

DwarfStar is a DeepSeek V4 Flash specific inference engine. It is not a generic
GGUF runner. The goal is a small, readable, high-performance C codebase with
Objective-C only where Metal requires it and Metal kernels under `metal/`.
Sources live under `src/` by module (2026-08-25 重构), one file ≤500 lines
(`make linecount` enforces it; exemptions in `.linecount-exempt`).

## Goals

- Keep the production path as whole-model Metal graph inference.
- Keep model loading mmap-backed; do not eagerly copy the full GGUF.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained
  attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.

## Quality Rules

- Comment important inference code where the model mechanics, cache lifetime,
  memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache
  boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- Do not introduce C++.

## Safety

- Avoid large CPU inference runs on macOS; the CPU path has previously exposed
  kernel VM failures with very large mappings.
- Do not run multiple huge model processes concurrently. The instance lock is
  intentional.
- Prefer short Metal smoke tests for build verification.

## Layout

- `src/core/`: model loading, tokenizer, CPU reference code, GPU graph
  scheduling, sessions, disk-cache payload serialization (was `ds4.c`).
- `src/common/`: the single implementation of GGUF reading, quant-block
  dequant, E4M3/E2M1 and f16 scalars, safetensors — shared with `gguf-tools`.
- `src/metal/` + `metal/*.metal`: Metal runtime wrappers + compute kernels
  (shaders are loaded at runtime by relative path; do not move `metal/`).
- `src/cuda/`: CUDA backend (`ds4_cuda.cu` is the single-TU aggregate root).
- `src/server/`, `src/cli/`, `src/agent/`, `src/eval/`, `src/bench/`: the five
  binaries. `src/dist/`: distributed runtime. `src/kv/`: disk KV store.
  `src/web/`: agent web fetch.
- Root keeps: public headers (`ds4.h`, `ds4_gpu*.h`, ...), vendored `rax`/
  `linenoise`, small host modules (`ds4_z`, `ds4_loss`, `ds4_corr`,
  `ds4_zchain`, `ds4_multimodal`, `ds4_spatial`, `ds4_css`, `ds4_posttrain`).
- `gguf-tools/`: offline quantizer + 反修 toolchain (own Makefile;
  quantize/amp/calib/bench/scripts/data/legacy/docs/migrate).
- `tests/`: suites (`t_*.c`), server tests, `unit/`, golden fixtures.
- `docs/archive/`: retired campaign plans. `misc/`: old notes.

## Testing

Use `make` for build validation. `make test` runs offline suites (golden-fixture
unit tests, 104 server tests, sampler units, rax, TP loopback, Metal kernel
numerics, the ≤500-line guard) on any Mac — model-dependent suites SKIP cleanly
when `ds4flash.gguf` is absent and run where a model exists. Use live server
tests only when intentionally testing the API surface.
