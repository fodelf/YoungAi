# Task A1 — 装载架构：选择性热集驻留（专家 offload，省内存）

**状态**：🟡 机制跑通 · **待内存硬数据**。代码完成、`make` 全绿、k4 实跑 not-covered 已灭、43 层跑通；但 physical footprint 降幅**尚无可信硬数据**（vmmap 待补测）。
**来源**：用户驳回「全模型一次性驻留」（无扩展性），plan mode 出方案并批准（`~/.claude/plans/memoized-frolicking-blum.md`）。路线 A 先做 A1（最小改动）验证物理效果，不行再 A2。
**插入位置**：Task 01 进行中临时插入（装载架构比 route_translate 更根本）。

## 背景（为什么改）

当前加载把整个 tensor-data 区无差别 wrap 成巨型 view 并**全量 requestResidency** → k4 实测驻留 9.3GB，全量模型根本装不下。DeepSeek V4 是 MoE，routed expert 占绝大部分字节但每 token 只 top-K 参与 → 冷专家不该驻留。CUDA 侧（`ds4.c:1754`）已用 `_exps.` 跳过逻辑，Metal 侧缺等价物。

## A1 机制

`DS4_METAL_EXPERT_OFFLOAD=1` 时：backbone（attn/shared-FFN/embedding/output）进 residency set 驻留；routed 专家**仍被 wrap**（热路径能拿到 buffer 不崩）但**不进 residency set** → 冷专家 file-backed 干净页可被内核回收。**默认不设 env = byte-for-byte 零回归。**

## 已实现（make 0 warning / 0 error）

| 文件 | 改动 |
|---|---|
| `ds4_metal.m` | `g_model_views` 加 `resident_hint`；`add_model_view_range` 加 `bool resident`；`request_views` 只 add `resident_hint` 的 view（含统计驻留数）；`set_model_map_spans` 重构为带可选 `resident_flags[]` 的 `_impl`（NULL=旧行为）+ 新 `set_model_map_spans_split` |
| `ds4_gpu.h` | `ds4_gpu_set_model_map_spans_split` 声明 + 语义注释 |
| `ds4_cuda.cu` | `_split` 等价实现（CUDA 冷专家走 UVA，flags 仅 advisory） |
| `ds4.c` | `model_map_span_vec_split_layer` + `model_map_span_vec_finalize` + `weights_model_map_spans_split`（专家张量→expert 组，含 shared expert 的其余→backbone 组，**按字段不靠名字**）；`ds4_engine_create` 全量分支加 `DS4_METAL_EXPERT_OFFLOAD` 路径，打印 backbone/expert 字节分割；`DS4_METAL_EXPERT_OFFLOAD_DEBUG` 打印每 span + not-covered 时的精确字节/view 边界 |

## 验证结果（2026-05-30，用户确认 + 13GB 看门狗）

**✅ 已确证**：
- 91 个 disjoint view（90 backbone resident + 1 expert non-resident），分割正确（backbone 8.20GiB / experts 1.13GiB）。
- **not-covered 错误已灭**（修了 Task 01 遗留的 n_total_expert=256→kept_count 越界 bug，见 task/01）。
- 全 43 层 prefill + 生成跑通，exit 0。
- **residency 请求时间 2372ms → 322ms（只 wire backbone，快 7×）** —— offload 确实只驻留热集的**直接证据**。
- **速度无回归**：baseline gen 9.47 vs offload 9.49 t/s。

**❌ 尚未拿到 / 仍坏**：
1. **physical footprint 降幅无硬数据** —— footprint 采样脚本反复抓错 PID（报 1632KB，是包裹 shell 不是 ds4）。`ps rss` 对 Metal no-copy mmap 失真，须用 `vmmap <pid> | grep "Physical footprint"` 手动测。这是 A1 该不该算成功的**关键缺口**。
2. **输出正确性仍坏** —— 与 Task 01 同源（route_translate 未真正修好），非 A1 引入。

## GGUF 布局事实（验证中实测，对后续有用）

k4 张量布局其实很干净（**非交错**）：`token_embd[0..8.7MiB]` → `全部专家[8.7MiB..1.14GiB]` → `全部 backbone[1.14GiB..9.33GiB]`。backbone/expert merged span 零重叠 → host split 分组天然干净（不是我一度担心的交错布局）。

## 待办（恢复时）

1. **补 footprint 硬数据**：`vmmap <pid> | grep "Physical footprint"` 手动测 baseline vs offload，拿到物理内存降幅定论。判 A1 成功/失败。
2. 若 A1 footprint 降幅不足（巨型 view 即便不 request 仍钉物理页）→ 升级 A2（专家逐张小 view + LRU 热池硬控制，另立 task）。
3. 清理临时诊断打印（`DS4_METAL_EXPERT_OFFLOAD_DEBUG` 可保留作诊断开关）。

## 诚实裁决

A1 装载机制**跑通了**（not-covered 灭、residency 只请热集、速度无回归），但**没用 physical footprint 硬证物理内存降幅** → 不宣称成功，只说「机制跑通，待硬数据」。route_translate 输出正确性是独立的 Task 01 问题。
