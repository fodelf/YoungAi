# task.md — 双机 q2 编程域有效 t/s 突破计划（5.40 → 逼近 30，攻 compute 墙）

> 建立：2026-06-19。承接 `project.md`/`log.md`/`notes/execution-log.md`（wave 1–78）。
> **本文件是当前前向计划的唯一活稿**。每一轮按"分析原因 → 设计方案 → 预期数字 → 改代码 → （授权后）跑脚本 → 实测修正"递进。

## 0. 执行纪律（铁律，每轮都遵守）

1. **以最新日志/代码/外部知识为准，忘记旧结论与旧注释**。`project.md` 里"W2 SSD 墙是主瓶颈"的框架对**冷态**仍对，但对**暖态编程域已过时**（见 §1）。
2. **禁自证循环**：不再靠"加一个环境变量再 A/B"刷存在感。每轮动工前提 = 有 **bug / 新代码路线 / 新外部知识 / 新方向** 之一。env knob 只能是已落地真实代码改动的 A/B 开关，不是方案本身。
3. **改码必有依据**（日志指明 / 新代码链路 / 联网理论），基线 5.40，改回 5.x 不算推进。
4. **30 t/s 是目标，不判物理不可达**；但每个数字要么实测、要么标"估算（待授权 A/B 修正）"，不编。
5. 跑脚本逐次单独授权；内存安全闸（L1 resident gate + 12/12 看门狗）永不放松；correctness before speed（per-merge 对拍门见 §6）。

## 1. 最新根因（wave 73–78 实测 + kernel 代码级确认，取代旧框架）

**必须分两个 regime，墙完全不同：**

| regime | 触发 | 瓶颈（实证） | 实测 t/s |
|---|---|---|---|
| **冷态** | smoke / replay 新内容轮 / 纯新代码 | **SSD IO**：每 token 流 ~1.70 GiB 专家（`cold_mib=27–40 pread=96–140ms`） | 1.77–2.16 |
| **暖态（编程域主战场）** | code-edit-heavy + copy-spec verify 批 | **GPU compute**：专家全在 page cache（`cold_mib=0 pread=0`），墙=routed-expert MoE 矩阵乘 | **5.15–5.40** |

### 1.1 暖态墙 = 冷专家 gather IO（2026-06-19 cycle-1/2/2b 实测订正；旧"compute-bound"判断错误）

> ⚠️ **重大订正**：cycle-0 我基于 wave-78 一句旧注释（"verify 批 cold_mib=0"）把墙定为 compute-bound（MoE GEMM 欠填），**错了**。今天 cycle-1/2/2b 的 `ds4-io` profile（一直开着）证明 verify 批是 **冷态 SSD-IO-bound**，回到 project.md W2 框架。

- verify 批（n_tokens=49）实测 `ds4-io`：`n_active=143-160 cold_mib=357 hit_mib=0.0 rfetch_mib=607 wall_ms=267 pread_ms=2108(8线程summed) drain_ms=26 bw=3.78GB/s pf=131/160`。
- **gather wall ~267ms/层 vs GPU drain 仅 26ms → IO 占墙 ~90%，GPU compute（dequant/MMA 所在）仅 ~10%。**
- 专家**冷读**（hit_mib=0.0，source cache `admit=0 hit=0%`）：working set ~960MiB/层 ≫ per-layer cache ~120MiB → 装不下 → 每个 verify 批把 ~960MiB/层专家从 SSD/page-cache 冷拷进 scratch。
- **三连实验否定 compute 假设**：cycle-1 dequant staging ≈0、cycle-2b mat-vec ≈基线——因为它们动的是那 26ms 的 GPU，没碰 267ms 的 gather。
- ⇒ 真杠杆 = **减 gather 字节**（降激活 / 低 bit）或 **拆 IO 到两盘**（TP expert-split）或 **减 working set 让 cache 生效**（受 12G 预算限）。

### 1.2 其它已定位结构事实
- **冷态/bare-round 墙**：单 token 前向 ~118ms（两机 layer-pipeline 串行），暖天花板 ~8.5 t/s（wave-68）。
- backbone 内 **naive MLA**：Q up-proj 1024→32768 = 9.7ms/层单项最大（wave-77）→ matrix-absorption 候选。
- **TP 结构杠杆撞硬件墙**：compute-split（Phase 2 attention）每层 all-reduce 强制 GPU 流水 stall（无 GPU-direct AR over Thunderbolt），wave-74 代码级否定；但 **TP expert IO-split（Phase 3）冷态实测 1.64×**（wave-72，SPLIT_LOW=2），是冷态唯一结构杠杆。

## 2. 联网研究结论（2026-06-19，108 agent / 26 源 / 17 验证条；完整见 notes/）

**框架性印证（最有价值）：** PowerInfer-2（arXiv 2406.06282）实证 I/O-compute overlap 把 MoE 从 **77% I/O → 14% I/O**（compute-bound），并掉到 86% compute。**DS4 暖态编程域已经到这个状态**（`cold_mib=0`）。⇒ 文献的 SSD-offloading 技巧是"为到达 compute-bound"，DS4 在编程域已到，**下一道墙是 compute（§1.1），不是再榨 IO**。

| 研究条目 | 数字/硬件 | 对 DS4 的判定 |
|---|---|---|
| 跨层路由预测预取（Mixtral-offload/Fate/HOBBIT/MoE-Infinity/ProMoE） | 2–9.9× decode，预测 88–97%，**全 NVIDIA PCIe 25–32GB/s** | 倍率不可迁移（SSD 1–7GB/s）；DS4 已实现（P2.1）。**只对冷态有边际** |
| 顺序合并读（PowerInfer-2 UFS4.0） | 大块顺序 vs 小随机 **~8–9×** | 仅冷态有用；DS4 prefill 已做（P1.2），decode 专家随机；GGUF repack 已 P0.3 否决（mini 盘增益≤8%） |
| 降精度 offload（HOBBIT） | ≤1% 掉点 9.93× | **无空间**（DS4 已 2-bit）；可迁移=减专家/减字节 |
| CPU/GPU 混合（KTransformers/Fiddler） | ≤4× decode | **不适用**（Apple 统一内存无 PCIe 拷贝可省 + 要全专家驻 RAM） |
| 单请求专家局部性（MoE-Infinity/Mixtral） | <5% 专家复用 / 邻 token ~30% 复用；**批处理下消失** | 印证 DS4 K=16 95.6%；但"温度局部性弱于不均衡"→ **常驻池非变革**（吻合 wave-51 池更慢） |
| SSD 流式真先例（PowerInfer-2） | 超内存掉到 **2.13 t/s** | 框定冷态地板≈单位数；DS4 5.40 已超它（因暖态 compute-bound + copy-spec 摊薄） |

**研究未能验证的 4 个领域（= DS4 必须自研 / 已领先之处）：** ① M1/M4 真实 NVMe 带宽（DS4 P0.3 已测：mini 2.4–2.6 / MacBook 5.5–6.7 GB/s）；② 雷电真实 GB/s（DS4 wave-71 已测 RTT p50 84–140µs）；③ 投机解码代码域接受率（DS4 copy-spec 已实测 1.77→5.40）；④ 2-bit MoE 剪枝/聚类。**前三 DS4 已领先文献，第四是开放空间。**

**诚实结论（研究 + 自测一致）：** 无任何已发表结果在 Apple Silicon SSD 流式 MoE 上达到 30 t/s；单前向叠加上限 ~2–5×。**到 30 必须是"降单前向 compute × 投机摊薄"相乘，发生在 echo 重编辑回合（有效 t/s），非冷态稳态。**

## 3. 杠杆排序（暖态编程域优先；每条带代码位置 + 预期数字 + 质量门）

### L1 ★ IQ2_XXS MoE GEMM 解量化提速（bit-exact，首攻，最低风险）
- **依据**：§1.1 verify 批大头 = per-expert 解量化无摊薄；wave-78 明指"kernel-opt 该打 routed-expert matmul"。
- **机制**：`kernel_mul_mm_id`（mat-mat by id）当前从 `constant iq2xxs_grid` 数据相关查表（cache thrash）；而 mat-VEC 路径（`metal/moe.metal:599/698/1044`）**已**把码本 staged 进 threadgroup `svalues`。把同样的 threadgroup 码本 staging + 向量化 unpack 搬进 mat-mat 路径。
- **代码**：`metal/moe.metal` `dequantize_iq2_xxs`(278) / `kernel_mul_mm_id`(1561) + 必要时 `ds4_metal.m` shmem 预算。
- **预期**：解量化是 MoE GEMM 的可观一部分；threadgroup 码本期望解量化 1.3–1.5× → MoE GEMM ~-10–17% → 暖态 **5.40 → ~5.9–6.3 t/s（估算，待 A/B）**。
- **质量门**：纯 kernel、同数学 → **bit-exact**，过 `./ds4_test --metal-kernels`（已有 iq2_xxs id 用例）+ 双机 code-edit-heavy 输出逐字节 IDENTICAL。零下行风险。

### L2 verify 批降激活（n_active 160→~110，质量赌注，过门后并入）
- **依据**：§1.1 成本 ∝ n_active；wave-73 仅瘦 bare round（MAX_TOKENS=1），verify 批仍 full 6/160。
- **机制**：把 top-k 6→4 或 weight<α·top1 的尾部专家裁剪**扩到 verify 批（n>1）**。改的是 `ds4_metal.m` MoE-thin gate 的批路径应用（不只是翻 env）；真正工作 = 过质量门。
- **预期**：n_active 160→~110（-30%）→ MoE GEMM ~-30% → verify 批 ~-20% → 暖态 **5.40 → ~6.5–7 t/s（估算）**。改模型输出。
- **质量门（硬）**：`ds4-eval q1..q4 --temp 0 --seed 1` + `ds4_test --logprob-vectors`，不过即回滚默认 OFF。

### L3 MLA matrix-absorption（消 Q-up 解压，攻 bare-round/backbone，冷暖双吃）
- **依据**：wave-77 Q-up 1024→32768=9.7ms 单项最大；DeepSeek/FlashMLA 文档化的恒等吸收。
- **机制**：把 W_UQ·W_UK 吸收，attention 在压缩 latent 空间算，免每步把 latent 解压成 64×512 全 Q。
- **代码**：`ds4.c` attention（`matvec_q8_0(attn_q_b)` @ 5646/5665/5698）+ decode 路径。
- **预期**：bare-round 前向 ~-10–15% → 冷态 smoke **2.1 → ~2.4**，暖态贡献 ~+5–10%。math 恒等（~1e-6，放宽到 logprob-tolerance）。
- **风险/工作量**：中高，排 L1/L2 之后。

### L4 冷态 TP expert IO-split 全量化（已落地，补冷/暖模式切换）
- Phase 3（`DS4_TP_EXPERT_SPLIT`，SPLIT_LOW=2）冷态 1.64×（wave-72 实测）。需按 regime 自动切（暖态单机赢、冷态 TP 赢）。中优先。

### L5 投机继续（已 5.25–5.40，fork-limited）
- copy-spec/spec-pipe 链 fork 受限（~3 cycle）。研究证方向对但 DS4 已近穷尽。低优先，除非有新 drafter 路线（如 MTP 在暖态 verify 通道复用，需新证据）。

## 4. 到 30 t/s 的账（诚实，north star）

有效 t/s = (accepted tokens) / wall。当前 303 tok / 56s = 5.40，17 forwards × 17.8 tok/forward。
- **降单前向 compute**（L1+L2+L3 叠加 ~1.6–2×）：17 forwards / 28–35s → **~8.5–10.8 t/s** 同摊薄。
- **× 投机摊薄**（echo 重回合 tok/forward↑）：相乘。30 需 ~0.6s/forward 或 ~50 tok/forward —— 单杠杆够不到，**必须 compute 减半 × 摊薄翻倍同时成立**，且仅 echo 重回合。
- 里程碑：**M-A 暖态单前向 5.40→8（L1+L2，本月）**；M-B 叠 L3 + 摊薄 → echo 回合有效 12–15；M-C（north star）echo 回合有效 30。冷态目标=过 PowerInfer-2 地板 2.13 → L3+L4 推 ~3–4。

## 5. 本轮动工（cycle-1）

**做 L1**（bit-exact，最稳，直接攻 §1.1 实证大头）。先读 `kernel_mul_mm_id` 全 kernel + mat-vec 码本 staging 范式，实现 threadgroup 码本 staging 到 mat-mat-id IQ2_XXS 路径，`make` + `./ds4_test --metal-kernels` 自检 bit-exact。
- 预期：暖态 5.40 → ~5.9–6.3（估算）。
- **跑脚本需你授权**：① `./ds4_test --metal-kernels`（本机，安全）；② `tools/mtp_pipe_q2_speed.sh`（双机 code-edit-heavy，A/B vs 5.40）。

## 6. 验证协议（每个 L 合入前）
- `--dump-logprobs` 对 A3 基线：L1 逐位一致 / L3 logprob-tolerance；改路由/降激活（L2）另加 `--logprob-vectors` + `ds4-eval q1..q4 --temp 0 --seed 1`。
- `./ds4_test --metal-kernels --server` 绿；双机 code-edit-heavy + smoke 两档都跑（PC.5）。
- 每轮记录追加 `notes/execution-log.md`（wave 编号）：prefill/gen t/s、有效 t/s+接受率、n_active、verify r2_ms、两机 RSS 峰值、日志四件套。
- 速度回归 `ds4-bench` 2K/32K/100K/200K CSV 入 `notes/`。

## 7. 进度
- [x] cycle-0：联网研究（108 agent，notes/）+ 最新日志/kernel 根因重定位 + 本计划落地
- [x] cycle-1：L1 IQ2_XXS mat-mat-id 码本 threadgroup staging — **实测 ≈0（已还原）**（2026-06-19）
      - 实现：moe.metal 编译期 `STAGE_CB` + 偏特化 dispatch + 码本 staging；ds4_metal.m tg-mem +2176B。`make` 绿、`./ds4_test --metal-kernels` 绿。
      - **双机 code-edit-heavy A/B（决定性）**：gen **5.22 t/s**（基线 5.15–5.40，噪声内/略负）；输出 md5 `5c076036…` 1186B **与基线逐字节相同** + dist-mtp 指纹完全一致 ⇒ **bit-exact 证实**。
      - **结论（决定性，已排除一条路）：verify 批 MoE GEMM 不是 dequant/`constant`-cache-bound**；staged 码本无收益 + 2176B threadgroup 反掉 occupancy。⇒ 真瓶颈 = **MMA 欠填**（8 宽 simdgroup × 1.8 有效 token）或 **per-layer drain/gather**。
      - 已 `git checkout` 还原 + 重编回基线（避免 occupancy 污染 cycle-2 A/B）。
- [x] cycle-2/2b：mat-vec vs mat-mat（攻 MMA 欠填假设）— **实测否定**（2026-06-19）
      - cycle-2（`MM_ID_MIN=64`）**无效**：默认 `MOE_OVERLAP=1` 的 P-OVL 路径无条件 mat-mat、early-return、绕过 `use_mm_id`（我的疏忽）。
      - cycle-2b（`MOE_OVERLAP=0 MM_ID_MIN=64`，mat-vec 真生效，overlap banner=0、verify r2 变化证明）：gen **5.26**（≈基线），md5 bit-exact。mat-vec ≈ mat-mat ⇒ **非 MMA 欠填主导**。
      - **→ 读 `ds4-io` profile 揭真因（见 §1.1 订正）：verify 批是冷专家 gather IO-bound，非 compute。cycle-1/2/2b 都在打 GPU compute（10% 墙），故全无效。**
- [x] cycle-3a 降激活 verify 批（`MOE_THIN_MAX_TOKENS=64`）— **硬质量失败**（2026-06-19）：n_active 160→109 瘦生效，但 gen **2.11**（更差）+ 输出垃圾（无限 BOS 退化）+ copy-spec 崩溃。**DeepSeek V4 主生成路径扛不住 top-4。否决。**（基线 bare-round top-4 是未验质量债。）
      - TP（3b）经证据排除：ds4-io 显示 gather 已 local+remote(rfetch 经 TB) 双盘聚合；TP 不加新维度。
- [x] TB 吞吐 probe（`tools/tp_bw_probe.c`，dual-host reverse，无模型）— **雷电 IP = 4.71 GB/s（2 连接饱和）**。⇒ gather 合并上限 = mini 2.4 + TB 4.71 ≈ 7.1，实测 3.78 ≈ 一半 → **有 ~1.8× headroom，未贴硬件墙**。
- [x] cycle-4 重平衡否定 + 稀疏性核查（用户质疑驱动，2026-06-19）：读码证实**模型确实稀疏**（`n_expert_used=6`，gather 只取选中并集）；我误读 `experts=256`（=总数非用量）。profiler 实测：编程生成里专家分布**平**（top 18 vs 一堆 14）、8GiB 模拟 LRU 仅 **4.40%** 命中——**单 token 稀疏(1.70GiB)、但跨 token 那 6 个不断换 → 并集逼近全量、无时间局部性 → 缓存无效**。smoke 2.1 = 1.70GiB÷3.78GB/s 稀疏冷读速率。**不是 bug，稀疏在起作用，但 1.70GiB/token 冷读是地板。**
- [~] **cycle-5 ★ cache-aware 路由偏置（用户洞察，最有希望的稀疏专用杠杆）— 代码落地 + make 绿（2026-06-19）**：
      - 机制：router 选 top-6 时 `score += exp_probs_b + λ·hot[layer]`（hot=近期 gather 过=page-cache 热），**只改"选哪6个"、不改数量/权重**；49-token verify 批都偏向同一热集 → 并集收缩 → 冷 gather 降。**配 page cache、不建驻留 pool → 绕开 wave-51 执行墙。**
      - 落地：`ds4_metal.m` 加 hot 状态 + recency(gather 钩) + slot-map(层序对齐,免 .h/CUDA 改) + decode&batch 两 router 注入 combined bias；`λ=0` bit-exact 默认关。
      - **待授权测**：`ROUTER_CACHE_BIAS=0.1` code-edit-heavy → gen vs 5.40 + cold_mib vs 357(降?=复用升) + 输出正确性 + accepted vs 303(质量门)。
      - 旧 cycle-4 重平衡：gather 已带宽最优（cursor 按带宽分；remote 2.28 撞 TB 冷启动），重平衡无效，**转 cache-bias**。根因（probe+pread 双证）：本地 mini 冷散点读 1.36GB/s=长杆（pread 263ms≈wall 267），远程 TB 4.71 早干完空转；cursor+tail_reserve 让慢 mini 超额分（37%）。修：慢 mini 少分 / 快 worker+TB 多分 → wall 267→~160ms → **gen 5.40→~7-8（估算）**。真调度代码（带宽感知抢单/tail_reserve），非盲调参。**读 cursor/tail 代码中。**
- [ ] cycle-2：L2 verify 批降激活 + 质量门
- [ ] cycle-3：L3 MLA absorption
