# tensor-parallel-tasks.md — TP/EP 改造任务拆分

> ⚠️ **2026-06-20 已被取代(禁改码版,过时)。** 用户决定允许改码后,任务拆分以 **`tensor-parallel-ep-plan.md` §6** 为准
> (EP-replaces-PP + backbone Q4_K + KV Lever B/C 的 P0/P1/P2 + 验收闸 + env 开关 + 依赖图)。本文仅留作禁改码版历史参照。
>
> 配套 `tensor-parallel-design.md`。分级 P0/P1/P2，每任务带验收标准（对齐项目 correctness gate）、
> 依赖关系、可 A/B 的 env 开关命名。**执行纪律**：每个补丁/测量/决策追加 `notes/execution-log.md`（wave 编号）。
> **硬约束**：任何加载模型的脚本先证内存安全（RSS 预算 + 看门狗，≤12GiB/台）再跑；改完代码跑脚本前单独获授权；
> 本机重编后考虑是否同步 worker 二进制（共享 `CORE_OBJS`）。

---

## 通用验收闸（每个动核心代码的任务都要过）

- **G1 正确性 parity**：`--dump-logprobs /tmp/x.json --temp 0` 双机输出 vs A3 单机基线逐 logit 对拍
  （EP 的 AR-SUM 重组允许 row-parallel 固有 ~1e-6 drift，不允许结构性偏差）。
- **G2 kernel/server**：`./ds4_test --metal-kernels`（数值）+ `./ds4_test --server`（API/渲染/KV 簿记）全绿。
- **G3 内存安全**：跑前 L1 静态预算闸不 abort；运行时看门狗峰值 ≤12GiB/台，无 OOM/panic。
- **G4 路由/量化类**额外：`./ds4_test --logprob-vectors` + `ds4-eval q1..q4 --temp 0 --seed 1` 期望 token 数。
- **A/B 纪律**：每个新 env 开关默认值 = 当前稳定基线（off / 现状），可单独 A/B；记录每档 Δt/s + 峰值 RSS。

---

## P0 — 决策门（先测，不动核心代码就能给出 go/no-go）

### P0.1 — E0 雷电 all-reduce 延迟基准（最高优先，整方案的闸门）
- **做什么**：在 `en5` 直连上跑最小 ping-pong（TCP_NODELAY，交换 16KiB = 一个 `[n_embd]` 向量，往返 1 万次），
  测 RTT 中位/p99/抖动。复用现有 `tools/tp_bw_probe.c`（git status 显示已有）或现有 `ds4_dist_tp_selftest` 思路。
- **依赖**：无。
- **env**：无（独立探针）。
- **验收**：产出 RTT/抖动表。判据 —— RTT≤300µs → EP 同步链可控，P1 继续；RTT≥1ms 抖动大 →
  同步链是主瓶颈，P1 优先做 AR 融合（P1.3）。**写入 `notes/execution-log.md` 作为 go/no-go 记录。**
- **内存**：零模型加载，无风险。

### P0.2 — 现有 EP（DS4_TP_EXPERT_SPLIT）实测基线标定
- **做什么**：用 `tools/mtp_pipe_q2_speed.sh` 在 **smoke + code-edit 两 profile** 下，
  A/B `DS4_TP_EXPERT_SPLIT=0` vs `=1`（其余保持 5.36 基线配置），开 `DS4_TP_AR_LOG` 量 AR 的
  drain/net/total ms 占比。**只跑、不改码**。
- **依赖**：无（功能已落地）。需跑脚本授权。
- **env**：`DS4_TP_EXPERT_SPLIT`（0/1）、`DS4_TP_AR_LOG`、`DS4_TP_SPLIT_LOW`（扫 k=2/3/4）。
- **验收**：得到"EP on/off 的 t/s 差 + AR drain 占比 + 最优 k"三个数，写 `notes/execution-log.md`。
  这是判断 P1.1/P1.3 哪个收益大的依据（**铁律：改码必须有依据**）。
- **内存**：跑 q2 全量，必须先过 L1 预算闸 + 看门狗；单独授权。

---

## P1 — 主路改造（EP 池化 + 同步链优化，依赖 P0 数据）

### P1.1 — EP owner-map 稳定化（缓存命中累积）
- **做什么**：`ds4.c:11148-11172` 的 owner 选择从"slot 序 `[0,k)/[k,n)`"改为"**按 expert-id 固定 owner-map**"，
  让每台拥有的专家集合跨 token 稳定 → 各机专家缓存命中可累积（修当前 owner 随路由抖动导致缓存抖动）。
- **依赖**：P0.2（确认 EP 有收益且缓存抖动是瓶颈）。
- **env 新增**：`DS4_TP_EP_OWNER_MAP`（`slot`=现状默认 / `hash`=按 expert-id 哈希归属）。**默认 `slot` 保持现状可 A/B。**
- **验收**：G1+G2+G3；`hash` vs `slot` 在 code-edit profile 的 t/s + 专家缓存命中率（A3 profile 日志）对比；
  收益为正才保留默认翻为 `hash`，否则保留 `slot` 并记录否定数据点（不算路线错，是工程数据）。
- **风险**：owner-map 须与盘速不对称（P1.2）联动，避免慢盘多拿。

### P1.2 — 带宽感知非对称 k（快盘 worker 多拿专家）
- **做什么**：纯调参 + 文档。利用已落地的 `DS4_TP_SPLIT_LOW`，结合 project.md §1 盘速不对称实证
  （mini 慢盘 1.9GB/s vs MacBook 快盘 8-21GB/s），让 worker（快盘）多拿专家以平衡每层 wall time。
  **无新代码**（若需细化为"按实测盘速自动算 k"再开 P2）。
- **依赖**：P1.1（owner-map 稳定后调 k 才有意义）。
- **env**：`DS4_TP_SPLIT_LOW`（扫 1..n_used-1）。
- **验收**：找到使两机每层 time 最接近的 k，记 `notes/execution-log.md`；过 G1/G3。
- **风险**：撞 staging 净亏区（worker←mini -18%），owner-map 不能让 mini 慢盘服务 worker 的专家。

### P1.3 — all-reduce 融合（86→43 次同步）
- **做什么**：把 attention-output AR 与 routed-MoE AR（`ds4.c:11222-11253` 与 shared-split AR）
  合并为**每层一次** `[2×n_embd]`（或把两处 partial 拼一个 buffer）的单次 exchange，
  把每 token 同步次数从 43×2 降到 43。优化 AR 前的 drain（signal+flush+host_wait 占 40-45%）。
- **依赖**：P0.1（确认同步延迟是瓶颈，RTT≥1ms 时此项优先级升最高）+ P0.2（量到 drain 占比）。
- **env 新增**：`DS4_TP_FUSE_AR`（0=现状两次 / 1=融合一次）。**默认 0。**
- **验收**：G1（融合后 AR-SUM 结果 bit-exact 等价两次分别 AR，须严格验）+ G2；
  `DS4_TP_AR_LOG` 量同步次数/总延迟下降；t/s 提升记录。
- **风险**：两处 AR 在图里时序不同（attention 早于 MoE），融合需确保两个 partial 都已 host-visible；
  做错会引入正确性 drift → G1 必须严格过。

### P1.4 — 同步链 drain 削减（MTLSharedEvent 快路径深挖）
- **做什么**：`ds4.c:11222-11253` 的 drain 链（signal+flush+host_wait）实测占 AR 40-45%。
  对照记忆 [Metal wait Anukari precedent]：`waitUntilSignaledValue` 已用；进一步减少 per-CB flush 次数，
  尽量让 AR 的 host writeback 落在 unified memory 而不触发 GPU wait-back。
- **依赖**：P0.2（量到 drain 是大头）。
- **env**：复用现有 `DS4_METAL_EXPERT_EVENT_DRAIN` 系列；如需新开 `DS4_TP_DRAIN_MODE`。
- **验收**：G1+G2；AR total ms 下降，drain 占比 <40%。

---

## P2 — 探索（高风险/拓扑级，仅在 P1 见顶后）

### P2.1 — DP-attention + EP-MoE 对称拓扑探索（借 vLLM/SGLang）
- **做什么**：评估把拓扑从 layer-slicing 切换为"两机都驻全 backbone（需重量化降到各驻一份）+ 各驻一半专家"
  的对称 DP-attn/EP-MoE 形态。**这是拓扑级重构，与现 layer-slicing 冲突**，仅作设计探索 + 内存可行性闭式
  （backbone 8.4GiB 重量化到能两机各驻一份的预算核算），**不轻易实现**。
- **依赖**：P1 全部 + Stage-4 类 backbone 重量化（attention Q8→Q5，过质量门）。
- **env**：N/A（设计阶段）。
- **验收**：一份内存预算 + 同步成本闭式，judging 是否值得；写 `notes/execution-log.md` 作决策记录
  （结论需用户批准方可写为定论 —— 铁律 [日志结论需用户批准]）。

### P2.2 — 自动盘速→k 标定
- **做什么**：把 P1.2 的手调 k 升级为"启动时探针测两盘速 → 自动算带宽最优 k 和 owner-map 配比"。
- **依赖**：P1.1+P1.2 手调已证明甜点存在。
- **env**：`DS4_TP_EP_AUTO_BALANCE`（默认 off）。
- **验收**：自动算出的 k 与手调最优 k 接近；G1/G3。

---

## 依赖图

```
P0.1 (E0 延迟) ─┬─> P1.3 (AR 融合, RTT 高时优先)
                └─> P1.4 (drain 削减)
P0.2 (EP 基线) ─┬─> P1.1 (owner-map 稳定) ─> P1.2 (非对称 k) ─> P2.2 (自动标定)
                └─> P1.3 / P1.4
P1.* 见顶 ──────> P2.1 (DP-attn+EP-MoE 拓扑探索, 需 backbone 重量化)
```

## env 开关总表（A/B 纪律：均默认现状/off）

| env | 取值 | 默认 | 作用 | 任务 |
|---|---|---|---|---|
| `DS4_TP_EXPERT_SPLIT` | 0/1 | 0 | 已落地：每台 gather k 专家（2-way EP），SSD 流量减半 | P0.2 基线 |
| `DS4_TP_SPLIT_LOW` | 1..n_used-1 | n_used/2 | 已落地：非对称 k（快盘多拿） | P1.2 |
| `DS4_TP_AR_LOG` | set | unset | 已落地：每 AR drain/net/total ms 日志 | P0.2 诊断 |
| `DS4_TP_EP_OWNER_MAP` | slot/hash | slot | **新**：owner 选择稳定化（缓存命中累积） | P1.1 |
| `DS4_TP_FUSE_AR` | 0/1 | 0 | **新**：attention+MoE AR 融合（86→43 同步） | P1.3 |
| `DS4_TP_DRAIN_MODE` | (待定) | 现状 | **新**（可选）：AR drain 链精简 | P1.4 |
| `DS4_TP_EP_AUTO_BALANCE` | 0/1 | 0 | **新**：盘速→k 自动标定 | P2.2 |
| `DS4_TP_SHARED_SPLIT` / `_EXPERT_SPLIT_BATCH` / `_REVERSE_CONNECT` | — | 现状 | 已落地，本轮不改 | — |

## 诚实优先级建议（给执行者）

1. **先 P0.1 + P0.2**（不改码，纯测，给 go/no-go 和 EP 真实收益数）。
2. **若 P0.2 显示 EP 有正收益** → P1.1（owner-map）+ P1.2（k 调参）拿稳态提升。
3. **若 P0.1 显示同步延迟主导** → P1.3（AR 融合）优先。
4. **P2 仅在 P1 见顶**，且 DP-attn 拓扑重构需用户决策 + backbone 重量化前置。
5. 全程记住：**TP/EP 这条线的现实目标是把 5.36 稳在 ~5 区间并为 copy-spec 抬基线，不是冲 30**
   （30 由编程域有效 t/s 交付，见 `tensor-parallel-design.md` §3.5）。
