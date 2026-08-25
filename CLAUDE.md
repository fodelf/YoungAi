# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**DwarfStar** (`ds4`) is a self-contained native inference engine **purpose-built for DeepSeek V4 Flash** (and, on very high-memory machines, DeepSeek V4 PRO). It is *not* a generic GGUF runner and does **not** link against GGML — it reimplements the loading, tokenizer, prompt/DSML rendering, KV cache, graph scheduling, server API, and a native coding agent for this one model family. It only runs the DeepSeek V4 GGUFs published for this project (asymmetric quant: routed MoE experts at `IQ2_XXS`/`Q2_K`, everything else left high-precision). See `README.md` for the full feature tour and `MODEL_CARD.md` for the model.

Primary backend is **Metal on macOS**; **CUDA on Linux** is the second production path; the **CPU path is reference/debug only**.

## Layout (2026-08-25 重构后)

单文件 ≤500 行是硬规矩（`make linecount` 守卫，豁免走 `.linecount-exempt`，只收
vendored/单函数 EXCEPTION/冻结转录三类）。源码按模块住 `src/`：

```
src/common/   全仓唯一的格式基元: GGUF 读取(ds4_gguf)、量化块 dequant(ds4_quantfmt,
              含 iq2_xxs 编码/解码双表正名)、E4M3/E2M1(ds4_fp8.h)、f16(ds4_float.h)、
              safetensors(ds4_st.c)。引擎与 gguf-tools 共用这一份——改格式只改这里。
src/core/     引擎主体(原 ds4.c 24.6k 行拆 60 文件): gguf 加载/权重绑定/CPU 前向/
              GPU graph 编排/tokenizer/采样/session 与 payload。内部头 core_internal.h。
src/metal/    Metal 后端(原 ds4_metal.m 拆 62 文件); shader 在顶层 metal/(运行时按
              相对路径拼接加载, 移动该目录=运行时炸)。
src/cuda/     CUDA 后端: ds4_cuda.cu 是聚合根, 分片在 src/cuda/*.inc.cu(单 TU 语义)。
              ds4_gpu.h(伞头)+六个子头是 Metal/CUDA 共同契约, 已带 extern "C" 守卫,
              CUDA 直接 include——契约漂移=编译错误。
src/dist/     分布式运行时(层切片/TCP 协议/coordinator/worker/TP)。
src/server/   HTTP 服务(OpenAI/Anthropic/Responses); 内嵌测试已解耦到 tests/。
src/agent/    终端编码 agent。src/cli/ src/eval/ src/bench/ 各叶子程序。
src/kv/       磁盘 KV checkpoint。src/web/ agent 网页抓取。
gguf-tools/   离线量化与反修工具链(独立 Makefile): quantize/(HF→GGUF 量化器与编码格式)
              amp/(反修: 放大器解算/逐层调优/侧车链) calib/(标定数据面) bench/(判决尺:
              anchor_metrics/pubbench/quality-testing) scripts/(活脚本) data/(冻结数据)
              legacy/(冻结旧代) docs/ migrate/(迁移金标账本 golden.txt)。
tests/        ds4_test 套件(tests/t_*.c)+server 测试(server_tests_*.c)+单元测试
              (unit/)+金标夹具(fixtures/quantfmt: dequant 七类型逐字节金标)。
docs/archive/ 已收官战役的设计稿。根目录还剩: 公共头(ds4.h/ds4_gpu*.h 等)、
              vendored(rax/linenoise)、小模块(ds4_z/ds4_loss/ds4_corr/ds4_zchain/
              ds4_multimodal/ds4_spatial/ds4_css/ds4_posttrain)、Makefile、文档。
```

## Build

The build is a single hand-written `Makefile` (no CMake/configure). Output binaries land in the repo root.

```sh
make              # macOS: builds Metal ds4, ds4-server, ds4-bench, ds4-eval, ds4-agent (default target)
make cuda-spark   # Linux: CUDA for DGX Spark / GB10 (HBM weight cache, no explicit -arch)
make cuda-generic # Linux: CUDA for a generic local GPU (CUDA_ARCH=native)
make cuda CUDA_ARCH=sm_120   # Linux: CUDA with an explicit nvcc arch
make cpu          # CPU-only reference/debug build (adds -DDS4_NO_GPU)
make clean
make -C gguf-tools all amp legacy calib bench tools-test   # 工具链全目标(Mac/Linux 自适应)
```

- On **Linux**, plain `make` only prints the target list — you must pick a `cuda-*`/`cpu` target explicitly.
- C is **C99**, `-O3 -ffast-math`, native CPU arch. GPU kernels: `src/metal/*.m` (Objective-C, `-fobjc-arc -fno-common`) 运行时拼接 `metal/*.metal`；`ds4_cuda.cu`(聚合根) 由 `nvcc` 编译。
- 对象文件放源文件旁(`src/<mod>/*.o`)。每个程序一个 `<MOD>_OBJS` 变量；`CORE_OBJS` = core+common+dist+后端。CPU 变体同源加 `-DDS4_NO_GPU`(`*_cpu.o`)。

## Run

```sh
./download_model.sh q2-imatrix   # fetch a model into ./gguf/ and point ./ds4flash.gguf at it
./ds4 -p "Explain Redis streams" # one-shot;  no -p => interactive REPL
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
./ds4-agent                      # in-process native coding agent (sessions in ~/.ds4/kvcache)
```

`./ds4flash.gguf` (a symlink) is the default model for every binary; pass `-m gguf/<file>` to override. Use `--metal`/`--cuda`/`--cpu` to force a backend, `--chdir /path/to/ds4` when launching from elsewhere so `metal/*.metal` resolves.

## Test

Read `CONTRIBUTING.md` before changing inference code — it defines the correctness and speed regression tracks.

```sh
make test                      # ds4_unit(离线金标) + linecount 守卫 + 抽取器自测 + ds4_test 全套
./ds4_test --list              # 11 个 suite; 模型缺失的套件打 SKIP 不 FAIL
./ds4_test --server            # 104 项服务端单测(离线): API 解析/DSML/流式/KV-disk
./ds4_test --engine-units      # 采样与 per-request 惩罚单测(离线)
./ds4_test --rax               # 基数树单测(离线)
./ds4_test --metal-kernels     # 孤立 Metal kernel 数值(要 Metal 设备, 不要模型)
./ds4_test --tp-allreduce      # TP 传输回环(离线)
./ds4_test --logprob-vectors   # 对官方 API 逐位对拍(要真模型)
./ds4_test --long-context / --tool-call-quality / --local-golden-vectors / --metal-short-prefill / --metal-tensor-equivalence   # 其余真模型套件
./ds4_unit                     # src/common 金标: dequant 七类型逐字节 + fp8 + GGUF 往返
make cuda-regression           # Linux/CUDA only(首跑含 PTX JIT 可能超时, 复跑为准)
make -C gguf-tools tools-test  # 工具链 9 项 -D*_TEST 自测
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

Quantization/GGUF changes are scored with `gguf-tools/bench/quality-testing` (`make -C gguf-tools quality-score`, then `score_official` + `compare_scores`; lower `avg_nll` is better).

## Architecture (the cross-file picture)

- **Backend abstraction.** `ds4_gpu.h`(伞头, 六子头带 extern "C") is the GPU tensor/graph
  interface, implemented by `src/metal/` and `src/cuda/`; `-DDS4_NO_GPU` selects the CPU
  reference in `src/core/`. `src/core/` owns everything host-side. Model weights are
  **mmap-backed and never eagerly copied** — Metal wraps mmap regions as no-copy
  `MTLBuffer` views and uses an `MTLResidencySet` to budget GPU VM.

- **Compressed KV cache is a first-class *disk* citizen.** DeepSeek V4's KV is heavily compressed (raw sliding window + ratio-4 indexer-selected + ratio-128 compressed layers). `src/core/core_payload*.c`/`core_snapshot_*.c` serialize the `DSV4` session payload; `src/kv/` manages the on-disk KV cache (`<sha1>.kv`, plain read/write — *not* mmap). The server keeps exactly **one live in-memory checkpoint**; the disk cache is the resume mechanism. File format (48-byte `KVC` header, rendered text, `DSV4` payload, optional `KTM` tool-id map) documented in `README.md` → Disk KV Cache.

- **Server (`src/server/`).** OpenAI/Anthropic/Responses-compatible HTTP. Inference serialized through one graph worker. Key subtlety: stateless clients resend JSON tool calls, so the server keeps an **exact-DSML replay map** (tool id -> exact sampled DSML bytes, radix-tree via `rax.c`) so re-rendered prompts byte-match the live KV checkpoint. During tool-call *syntax* the server forces `temperature=0`.

- **Distributed inference (`src/dist/`).** Layer-sliced across machines (`--role coordinator|worker`, `--layers 0:19` / `20:output`). Custom binary TCP protocol; activations flow worker→worker, coordinator owns tokenization/sampling. A rolling 64-bit token-prefix hash guards stale worker KV. Saved sessions use the same single-file `DSV4` payload.

- **Native agent (`src/agent/`).** Drives inference in-process; the session *is* the on-disk KV cache. `/list`, `/switch <sha>`, `/save`, `/del`, `/strip`.

- **反修/量化工具链 (`gguf-tools/`).** 独立 Makefile。判决尺=参考前向(`caliper_ref.sh`,
  五指标, 见 memory 铁律); 迁移金标账本在 `gguf-tools/migrate/golden.txt`。格式基元
  一律走 `src/common`——工具侧禁止再抄第二份 dequant/GGUF/safetensors 实现。

Behavior has many `DS4_*` env switches; treat them as diagnostic/tuning switches around the single release path, not permanent feature flags. **本项目禁止新增 env 配置**(铁律 2026-08-22): 行为写死进代码, 取料入口走 CLI 参数(`--cap-dir`/`--eval-ids` 等)。

## 当前主记录

**`fable5.md` 是项目主记录**(用户 2026-07-02 指定): 结论、判决、战役日志全在那里,
重构(2026-08-25, restructure 分支)的全部记录也在。`docs/archive/` 存已收官战役的
设计稿(双机 q2 提速的 project.md 等 6 月文稿在归档前的根目录历史里)。北极星=
合适体积+模型全能力还原(非编程域专项), "go"命名的战役设施已随 2026-08-25 重构清退。

## Guardrails (hard)

- Watchdog red line **12/12 GiB** (both hosts on dual-Mac runs); `DS4_MEM_BUDGET_MB` + L1 resident gate must refuse startup over budget. **Any model-loading script must prove memory safety (RSS budget + watchdog) before it runs.** Never double-load an 80+ GiB base on one host.
- After a local rebuild, consider whether the other machine needs the synced binary too (shared engine objects). Never delete files on the other host without explicit confirmation — "cleanup" means killing the process, not removing files.
- **Per-merge correctness gates**: `make test` 全绿(含 linecount); 推理路改动加
  `--dump-logprobs` parity 与 `ds4_test --logprob-vectors`(有模型的机器);
  routing/quant/repack changes additionally run `ds4-eval q1..q4 --temp 0 --seed 1`。
  量化质量判决只认参考前向尺(caliper), 真代码基准必须 completions 口径(memory 铁律)。

## Project rules (from `AGENT.md` — follow these)

- **No C++.** Pure C99, with Objective-C only where Metal requires it. Python 禁止进入
  数值/算法链(全仓零 Python 裁决; 仅存的 .py 是外部 API 采集/绘图/金标夹具生成器)。
- **Keep model loading mmap-backed**; do not eagerly copy the full GGUF.
- **Correctness before speed.** Don't keep a faster path with unexplained attention/KV/logit drift.
- **单文件 ≤500 行**(make linecount 强制); 注释写"为什么"不复述代码。
- **macOS CPU danger:** running the CPU inference path on macOS can crash the kernel — avoid large CPU runs there.
- **Instance lock is intentional:** do not run multiple huge-model processes concurrently.

## Debugging

```sh
./ds4 --dump-tokens -p "..."                              # tokenize (incl. DS4 specials) and exit
./ds4 --dump-logprobs /tmp/out.json --temp 0 -p "..."     # greedy continuation + top-k alternatives
./ds4 --score-ids ids.txt --score-out /tmp/score.bin      # teacher-forced 逐位 logits(解码路, 100% 决定论)
./ds4-server --trace /tmp/ds4-trace.txt ...               # log rendered prompts, cache decisions, tool events
```
