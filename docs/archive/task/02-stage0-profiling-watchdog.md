# Task 02 — Stage 0：DS4_PROFILE 计时 + phys_footprint 内存看门狗

**状态**：🟡 代码完成 · `make` 全绿（exit 0 / 0 warning）· **仅剩运行验证**（需用户确认 + 看门狗）
**前置依赖**：无（可与 #01 并行；是所有测量类 task 的基础设施）
**类型**：单机 · 基础设施（用户头号诉求：逐模块计时 + 绝不爆内存）

## ✅ 已实现（ds4.c，8 处 Edit，`make` exit 0 / 0 warning，ds4.o 重编 + 5 binary 重链）

自包含 Stage 0 模块插在 `now_sec()`（ds4.c:718）之后，零热路径依赖：

| 件 | 实现 |
|---|---|
| **phys_footprint helper** | `ds4_phys_footprint_bytes()` 用 Mach `task_info(TASK_VM_INFO).phys_footprint`（`__APPLE__` 守卫，非 ps/rss——后者对 Metal no-copy mmap 失真，9.3GiB 只报 ~38MiB）；非 Apple 返回 0 |
| **看门狗线程** | `ds4_mem_watchdog_main`：每 200ms 采样 footprint，记峰值；`DS4_MEM_BUDGET_MB` 设了则越 90% 预算前 `_exit(137)` 防 thrash。`__attribute__((constructor))` 自启 → 覆盖整进程（含模型加载大 mmap + residency wiring） |
| **DS4_PROFILE 计时** | `ds4_profile_load_begin/end`（engine_create 内包 model_open + 全 residency wiring，记墙钟 + footprint 增量）；`add_prefill`/`add_decode`；`atexit(ds4_profile_flush)` 输出单行到 `DS4_PROFILE_FILE`（或 stderr）：load/prefill/decode 墙钟·t/s + 峰值 footprint |
| **L1 起飞前预算闸** | `ds4_l1_budget_gate(resident_model_bytes, kv_scratch)`：绑任何 GPU buffer 前闭式算，> 85% 预算直接 `_exit(137)` 打印明细。三个加载分支各调一次（slice=span_bytes / offload=resident_bytes（专家可回收不计）/ 全量=tensor-data 字节）。KV/scratch 暂传 0（模型驻留是 ~88% 主导项，运行时看门狗兜 KV） |
| **接线点** | decode 计时包 `ds4_session_eval`（薄包装）；prefill 计时把 `ds4_session_sync` 改名 `_internal` + 新薄包装（捕获 prefill token 数 = prompt.len − 匹配前缀）；load 计时包 engine_create 的 model_open→success-return |

**零回归保证**：`DS4_PROFILE` 与 `DS4_MEM_BUDGET_MB` 都没设 → 构造函数一次 getenv 后返回（不起线程）；
`ds4_session_eval`/`_sync` 在 `!g_prof.enabled` 时直接转调内层 = 旧路径 byte-for-byte。L1 闸无预算 = no-op。

## ⏳ 唯一剩余：运行验证（待用户确认 + 看门狗）

```sh
DS4_PROFILE=1 DS4_PROFILE_FILE=/tmp/ds4-prof.csv \
  ./ds4 -m gguf/ds4flash-k4.gguf --ctx 4096 --temp 0 -n 64 -p "hello"   # 期望: 末尾 CSV 行有 load/prefill/decode t/s + 峰值 footprint
DS4_MEM_BUDGET_MB=1024 ./ds4 -m gguf/ds4flash-k4.gguf --ctx 4096 -n 8 -p "hi"  # 期望: L1 闸在加载前 _exit(137) 打印明细
```

---

### 旧状态行（存档）：未开始

## 目标

建立「逐模块时间 + 峰值内存」可观测基础设施,关时零开销,两机各跑一份。产出单机最小 demo 的首张
「逐模块时间 + 峰值 RSS（phys_footprint）」表。

## A. 逐模块计时（DS4_PROFILE）

仿已有 `DS4_MTP_TIMING`（`ds4.c:20087`）与 `ds4_bench.c::bench_now_sec()`（`clock_gettime(CLOCK_MONOTONIC)`）。

- 加全局开关 `DS4_PROFILE=1`，输出单行 CSV 到 `DS4_PROFILE_FILE`；**关时零开销**（getenv 缓存到静态 bool，热路径只读 bool）。
- 计时夹点（hook → 度量）：

| 模块 | Hook | 度量 |
|---|---|---|
| 加载 | `model_open` / mmap | 墙钟 + phys_footprint 增量 |
| 预填充 | `prefill_chunk`（`ds4.c:19567` 附近） | per-chunk ms, prefill t/s |
| 解码 | `decode_token`（`ds4.c:19873` 附近） | per-token ms |
| 每层（可选） | attn · moe · sample | per-layer µs |
| all-reduce（新，留空桩） | TP 同步点 | 每层 µs + 抖动 |
| 投机 | `ds4_session_eval_speculative_argmax`（`ds4.c:20026`）/ MTP timing `ds4.c:20087` | 接受率, draft/verify ms |
| 跨机 | `ds4_distributed.c:146` 现有 `*_usec`（eval/downstream_wait/forward_send） | 透出到同一 CSV |

- CLI/server 代码保持无 tensor 内部（铁律）；计时逻辑放 `ds4.c` 核心侧。

## B. phys_footprint 内存看门狗（关键度量修正）

**铁律**：用 `mach task_vm_info` 的 `phys_footprint`，**不是 `ps -o rss`**。
Stage 1a 实证：ps rss 对 Metal no-copy mmap（MAP_SHARED）完全失真（实 9.3GB 只报 38MB）。

- 采样线程：每 200ms 读一次 `task_info(mach_task_self(), TASK_VM_INFO, ...)` 的 `phys_footprint`，记峰值。
- 阈值：超 **预算×0.9** 在换页 thrash 前 `abort`/`SIGKILL`，并打印当前 footprint + 谁吃的内存（按已知大 buffer 分类）。
- 预算：M4 = 12GB，M1 = 8GB（两机各跑一个看门狗，预算从 env/CLI 传入）。
- 配套 **L1 起飞前静态预算闸**：绑任何 GPU buffer 前闭式算
  `resident = backbone份额 + owner专家数×每专家字节 + KV(ctx, ds4.c:14847 闭式) + prefill_scratch + 固定开销`，
  `> 预算×0.85` 直接 abort 打印明细（计划内 OOM 加载前挡死）。

## 闸门 / 验收

- `DS4_PROFILE` 未设时,基准 gen t/s 与 Stage 1a（10.58）无可测退化。
- 单机最小 demo（M4, `--ctx 4096`, gen 64 token）产出首张「逐模块时间 + 峰值 footprint」CSV。
- 故意把预算调到 1GB → 看门狗能在加载阶段 abort 并打印明细（验证护栏真的会拦）。

## 验证命令

```sh
make
DS4_PROFILE=1 DS4_PROFILE_FILE=/tmp/ds4-prof.csv \
  ./ds4 -m gguf/ds4flash-k4.gguf --ctx 4096 --temp 0 -n 64 -p "hello"
# 护栏自测：把预算压到 1GB，应在加载期 abort
DS4_MEM_BUDGET_MB=1024 ./ds4 -m gguf/ds4flash-k4.gguf --ctx 4096 -n 8 -p "hi"
```

## 产出

- `DS4_PROFILE` + phys_footprint 看门狗 + L1 预算闸 三件套（两机通用）。
- 首张逐模块计时表 → 喂给 #03 标定与后续所有阶段的护栏。
