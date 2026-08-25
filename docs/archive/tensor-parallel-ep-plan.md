# tensor-parallel-ep-plan.md — EP(Expert-Parallel)改造方案 + 任务拆分(允许改码版)

> 前提:**允许修改 C/Metal 源码**(用户 2026-06-20 决定)。这翻转了 `tensor-parallel-feasibility.md` 里「禁改码 ⇒ no-go」的裁决。
> 本文是 4 维设计 workflow(内存预算 / 拓扑解耦 / 质量门 / 慢盘墙)+ 红队复核(judged **needs_revision**)的合并产物,**以红队修正为准**。
> 所有数字带 `file:line` / execution-log 证据。本轮**只设计 + 读码定位 + 写文档**,未改源码、未跑加载脚本(内存安全铁律)。

---

## 0. TL;DR

- **改码可行性高**:EP-replaces-PP 的三条路径已在码上就绪——decode expert-split(`ds4.c:11148-11172`)、prefill batch-split(`ds4.c:14156-14165`)、backbone-resident+experts-reclaimable 加载(`ds4.c:19739-19796`,`load_slice==false && DS4_METAL_EXPERT_OFFLOAD`)。组合 `--tp + DS4_METAL_EXPERT_OFFLOAD=1 + DS4_TP_EXPERT_SPLIT + DS4_TP_EXPERT_SPLIT_BATCH` 即得「每台全 43 层 backbone(~8.4GiB)+ routed 专家二分(冷读字节减半,cold_mib 40.5→20.2 实测)」。
- **机制定位**:EP 是**冷态(smoke/long-context)单前向唯一能把 routed-expert SSD IO 减半**的杠杆(1.70→0.85 GiB/token/机)。它**不解决暖态/编程域**(暖态 EP compute-bound 净负,copy-spec-in-TP 2.10 << 单机 spec-pipe 3.77/5.40)。**编程域有效 t/s 仍走单机 spec-pipe,不进 TP。**
- **三档预期(红队下修后,诚实)**:冷态单前向 **乐观 1.8 / 现实 1.3 / 悲观 0.7-1.0 t/s**,封顶被 **mini 冷散点盘带宽 0.5-0.7GB/s**(非顺序 1.36)硬钉,改码不可越。编程域有效 t/s 维持 5.25-5.40(EP 无正贡献)。**30 t/s 不由此线交付**(撞 W3 backbone),纯速度看 k16 备选。
- **两个真实成本(不是物理墙,是工作量)**:
  1. **内存墙**:EP 每台全 backbone 8.4GiB,@200K 叠 Lever-A KV+scratch peak ≈ 9.7GiB(Q8,撑不住)⇒ 必须 backbone 重量化。**红队纠正:用 Q4_K(dequant 已存在,prefill GEMM 模板免费实例化)不是 Q5_K(零支持,从零写)。**
  2. **P2 @1M 硬前置**:KV 仅 Lever-A @1M=5.31GiB 叠 backbone 超红线;必须 Lever A+B+C 全做(KV→1.75GiB)。Lever C(disk-tier ratio-4 attn,roadmap 最大设计面,需独立 design memo)是硬前置,且与 expert gather **抢同一块盘**(红队风险,P2 可能跌破 1 t/s)。
- **必修 guardrail gap**:`ds4.c:19680` L1 启动闸第二参 `kv_and_scratch` 传 `0`,只查 backbone。P2 @1M 若 KV lever 未全落地,启动放行→运行时 phys_footprint 看门狗(`ds4.c:724-841`,对 mmap 有效)decode 冲过 10.8GiB 即 `_exit(137)` 猝死。**P0 必修**(内存安全铁律前置)。
- **物理墙(改码不可越)**:① 2 台 = **EP 替换 PP** 单拓扑自由度(真 2D PP×EP 需 ≥4 台,不做);② mini 冷散点盘带宽硬钉冷态 ~1 t/s 量级;③ 暖态 EP compute-bound 净负。

---

## 1. 红队关键修正(以此为准,覆盖综合设计的乐观处)

| # | 综合设计原主张 | 红队修正 | 证据 |
|---|---|---|---|
| R1 | backbone 用 **Q5_K**,新写 Q5_K dense kernel = 最高风险(数周) | **改用 Q4_K**:`dequantize_q4_K` 已存在(`moe.metal:301/494`),prefill GEMM `kernel_mul_mm` 模板参数化(`dense.metal:1599`)⇒ GEMM 近免费实例化;Q4_K 更省(0.5625 vs 0.6875 B/w → backbone ~5.4GiB)。Q5_K 是唯一零支持档,留作 Q4_K 质量不过时的退路。 | `grep metal/` 无 q5 结构;`moe.metal:301,494`;`dense.metal:1599-1600` |
| R2 | prefill+decode 笼统「新写 kernel」 | 拆两条难度悬殊路径:**prefill GEMM = 模板免费**;**decode GEMV = 需专门化**(backbone GEMV 是手写 `kernel_mul_mv_q8_0_f32` `dense.metal:180` 非模板,要把 `moe.metal:494` q4_K GEMV 提升成 dense host_name + `ds4_metal.m:2217` dispatch 增 q4_K 分支)。这才是 P1 真实成本。 | `dense.metal:180`;`ds4_metal.m:2217`;`moe.metal:494` |
| R3 | aggressive 档把 output 算进省量(peak 6.21) + M3 又说 output 保 Q8 | **不自洽,以保 Q8 为准**:output(`lm_head` 129280×4096)是 decode 每 token 必走最大单张量 GEMV + logit 直接源,**保 Q8**;不算进省量。attn_kv / attn_q_b / token_embd 同样保 Q8。 | `dense.metal:180`(GEMV 墙);M3 质量建议 |
| R4 | 冷态 t/s 乐观 2.2 / 现实 1.8 | **下修**:0.85GiB 是减半后**每台**字节,慢机 mini 单机 0.85÷1.36≈0.63s→~1.6;且冷散点 pread 实测 96-140ms/层(`execution-log:3649`)≈ 0.5-0.7GB/s 有效带宽,43 层 ≈ 4.3s ⇒ **冷态 0.7-1.0 悲观 / 1.3 现实 / 1.8 乐观**。 | `execution-log:3649` |
| R5 | P1 Q5 aggressive「余 2.39 给 page cache」 | **headroom 双重计数风险**:phys_footprint 对 mmap **有效**(算入 touched 页),暖专家 LRU 复用池本身吃 footprint ⇒「余 2.39 给 cache」与「peak 不撞 10.8」是同一块内存两种说法。P0 必须标定 page-cache 在 footprint 里的占比再定档位。 | `task3`(footprint 有效);`feedback_stability` |
| R6 | P2 内存安全押在 A+B+C(Lever C 未设计) | **前提倒置 + 无 fallback**:Lever C 与 expert gather 抢同一块 mini 盘(@1M 每 token 既冷读 0.85GiB 专家又读 ratio-4 KV 行,串行化可能跌破 1 t/s),且 Lever C 难产则 @1M 无退路(A+B=11.39 仍超)。**fallback = k16 全驻留,用户裁决前置到 P2 设计阶段。** | open_q;`optimization_roadmap:18,31` |
| R7 | 量化「纯离线」+ `--metal-kernels` 通用门 | 新 dense quant kernel 需**独立逐元素 vs Q8 参考误差门**,GEMM/GEMV **分别过**;L1 闸修复需「估算 vs 看门狗实测对账(<5%)」门,否则修了 gap 仍可能猝死。 | `dense.metal`;`ds4.c:724-841` |

---

## 2. 改码可行性:三路径已就绪(M2,confidence high)

- **decode expert-split**:`ds4.c:11148-11172`,每台 gather `[slot_start, slot_start+cnt)` 的 k 个专家,partial routed_out → `ds4_dist_tp_allreduce_f32` AR-SUM 重组,**bit-exact**(`ds4.c:11237-11241` 数学等价单机全 gather)。
- **prefill batch-split**:`ds4.c:14156-14165`(`DS4_TP_EXPERT_SPLIT_BATCH`)。
- **backbone-resident + experts-reclaimable 加载**:`ds4.c:19739-19796`,`load_slice==false && DS4_METAL_EXPERT_OFFLOAD` 已构建双 span 表(backbone 常驻 ~8.4GiB,routed 专家走 mmap reclaimable 按需读)——**这正是 EP-replaces-PP 每台所需的内存形态,不需要新写加载器**。
- **守约**:所有 split/AR 逻辑挂 `g->tp && il<g->tp_layers` guard + 新 env 默认 off ⇒ 非-TP 路径字节不变。
- **唯一硬结构增量**(M2):`ds4.c:14162` prefill batch-split 的 k 从硬编码 50/50 改读 `DS4_TP_SPLIT_LOW`(与 decode `11160-11161` 同源),让非对称切分在冷态 prefill 主战场也消 lockstep。low 风险、env-gated、bit-exact。

**澄清(写死在方案)**:2 台 = **EP 替换 PP**(单拓扑自由度),`--tp` 模式每台全 43 层 backbone;真 2D(PP×EP)需 ≥4 台,**本方案不做**。

---

## 3. 分阶段计划

### P0 — 基线 + guardrail 修复 + 冷散点带宽实测(零模型加载脚本,只读码 + 静态改码 + 短 probe)
- **改码**:
  - `ds4.c:19680` 修 guardrail gap:`ds4_l1_budget_gate(base+mtp, 0)` 第二参传真实估算 = KV 闭式(@200K 1.35 / @1M 视 lever 档)+ scratch 0.45 + gather 0.40。`risk=high`,非 env-gated(安全闸不 A/B)。**加对账门**(R7):估算 vs 看门狗实测峰值误差 <5%。
  - `ds4.c:11155-11163` 注释订正:owns_low 语义按**盘带宽比**标定(非 compute 均衡);mini 是慢盘长杆。仅注释,`risk=low`。
- **新增前置实测(R4/R-add2)**:用 `tools/tp_bw_probe`(git status untracked,已在)测 **6.75MiB 随机 pread 散点带宽**(非顺序 1.36GB/s),拿 `mini_bw/macbook_bw` 重算三档 t/s + 喂 P1 的 SPLIT_LOW 标定。
- **新增(R5)**:用现有单机暖态实测 phys_footprint vs 显式估算的差,**标定 page-cache 在 footprint 里的占比**,再定 P1 档位(解 headroom 双重计数)。
- **门**:`make` 编译过;`ds4_test --metal-kernels --server` 绿;L1 闸改动 dry-run(@200K 放行 / @1M 仅-A 正确拒启);单机暖态 5.25-5.40 / 冷 1.02 锚点不退。
- **依赖**:无(起点)。

### P1 — backbone **Q4_K** + EP 解耦,@200K 跑通
- **目标**:backbone Q8→Q4_K(8.81→~5.4-6.0GiB,output/attn_kv/attn_q_b/token_embd 保 Q8)+ Lever-A KV(@200K 1.35)⇒ EP 每台 peak ≈ **7.6-8.2GiB**,过 page-cache 阈值,EP 冷读减半兑现。
- **改码**(顺序):
  1. `gguf-tools/deepseek4-quantize.c:1109-1129,1168-1193` — 新建 GGUF policy:可量化大头(attn_q_a/o_a/o_b + shared_exp gate/up/down)Q8→**Q4_K**;**必保 Q8**:attn_kv + attn_q_b + output(lm_head)+ token_embd。per-class 钩子已就绪(`is_attention_proj/is_shared/is_output/is_embedding`),`is_plain_layout_weight` 守卫不绕过。`risk=medium`,非 env-gated。
  2. **`metal/dense.metal` + `ds4_metal.m:2217-2326` — Q4_K dense backbone kernel(P1 真实成本,R1/R2)**:
     - prefill **GEMM**:`kernel_mul_mm<...,block_q4_K,...,dequantize_q4_K,...>` 模板实例化(近免费,复用现成 `dequantize_q4_K`)。`risk=low`。
     - decode **GEMV**:把 `moe.metal:494` `kernel_mul_mv_q4_K_f32_impl` 提升成 dense host_name + `ds4_metal.m:2217` `make_*_mv_dispatch` 增 q4_K 分支(手写专门化,非模板)。`risk=medium`。
     - **独立 kernel 误差门**(R7):GEMM/GEMV 各自逐元素 vs Q8 参考过 `--metal-kernels`。
  3. `ds4.c:2922,2932-2940,2978-2980` — `weights_validate_layout` 白名单从硬编码 Q8_0 放宽到「Q8_0 或 Q4_K」,保留 attn_kv/attn_q_b/output 强制 Q8 分支。`risk=low`。
  4. `ds4_distributed.c:10354-10366`(+ `ds4.c:20022-20056`)— EP 拓扑解耦:新增 `DS4_EP_STANDALONE`(默认 off)内部映射 `tp_enabled` 全下游(g->tp/tp_owns_low/AR),语义化命名;行为等价 `--tp+EXPERT_OFFLOAD`。`risk=high`(结构),env-gated。**(open_q:时间紧可跳过,直接用现有组合,纯可读性收益)**
  5. `ds4.c:14162` — prefill batch-split k 改读 `DS4_TP_SPLIT_LOW`(bit-exact)。`risk=low`,env-gated。
  6. `ds4.c:11155-11163` — `DS4_TP_SPLIT_BW_AUTO`(默认 off):k/(n_used-k) ≈ 快盘/慢盘 带宽比(读 P0 probe),mini 分 <50% 专家。`risk=low`,env-gated。
  7. `ds4.c:~11140` + prefetch_predict_union — **split-aware prefetch**(`DS4_TP_SPLIT_AWARE_PREFETCH` 默认 off):把 `expert_prefetch_predict` 限到本机 owned 的 k 个 slot,修 naive 叠加净负(FULL_LAYER_STREAM 抵消 split / PREFETCH 污染全 6 专家)。`risk=medium`,env-gated。
  8. `ds4.c:19739-19796`(不改,复用)— 验收置 `DS4_METAL_EXPERT_OFFLOAD=1`。
- **门**(顺序):`--metal-kernels`(新 quant kernel 各自)→ `--logprob-vectors`(routing/Q4_K 漂移)→ `--metal-kernels --server` → `ds4-eval q1..q4 --temp 0 --seed 1`(README 基线:q1 PASS/2048/B、q2 PASS/438/C、q3 PASS/666/70、q4 FAIL/2048/A)→ `gguf-tools quality-score`。**EP parity**:`--dump-logprobs` 双机 EP vs 单机 A3 字节 parity(AR-SUM bit-exact)。**内存安全闸**(双机 `DS4_MEM_BUDGET_MB=12288`):L1 放行 + 运行时 footprint @200K <10.8 + page-cache 占比标定后真 headroom ≥ GO 阈值。**速度门**:冷态单前向 ≥ 现实档(以 P0 重算为准)+ cold_mib ≈20.2 MiB/层。
- **回退线**:`avg_nll delta > +0.02`(需 P0 先测 Q8 绝对基线锚,R7)或 q1..q3 任一 PASS/FAIL 翻转 / q4 token-drift >5% ⇒ 退「output/attn 全留 Q8、仅 shexp+q_a/o Q4_K」更保守档;仍不过退 Q5_K(此时才付从零写 q5 kernel 代价)。
- **依赖**:P0(L1 闸修 + 盘带宽 probe + page-cache 占比标定)。Lever-A KV 已 landed。

### P2 — KV Lever B + Lever C,@1M 跑通(高风险,不与 Q4_K 同 PR)
- **目标**:KV @1M 从仅-A 5.31 降到 A+B+C ~1.75,叠 Q4_K backbone 让 @1M peak 落到红线内(余 ~2)。这是 1M 唯一内存安全档。
- **改码**:
  - **Lever B(FP4 index_comp)**:接 `dsv4_kv.metal:59-201` 已有 e2m1 基元到 ratio-4 indexer top-512 选行;**长程召回断崖风险最高**。`risk=high`,env-gated。
  - **Lever C(disk-tier ratio-4 attn)**:ratio-4 行加 disk 分层(staging-buffer+pread,M4 16KiB page 排 512B 行直接 mmap);数值不变则 bit-exact。**需独立 design memo**(roadmap load-bearing)。`risk=high`,env-gated。
  - `ds4.c:19680`(再确认)— L1 闸喂 @1M KV 估算,证明 lever 未全落地时正确拒启。
  - IO-IO overlap(`ds4.c:2697-2702` 区):gather(L) drain 窗口 madvise 预热 owned L+1 专家(decode compute-comm overlap 物理死,IO prefetch 活)。
- **门**:Lever B/C **先单独过 `--long-context`**(分 ctx 深度 8K/200K/1M 召回扫描,R7,不设容差)→ `--logprob-vectors` → q1..q4 → quality-score。Lever C bit-exact:`--dump-logprobs` vs Lever-A-only baseline parity。内存安全闸 @1M peak <10.8。**速度门 + 盘竞争预算**(R6):建模 ratio-4 disk 读 + expert 0.85GiB/token 的盘竞争,证明 @1M 不显著慢于 P1。
- **fallback(R6,前置)**:Lever C 难产 / IO 竞争证伪 ⇒ **转 k16 全驻留**(13GiB 零 SSD IO,10-18 t/s,质量 16/256)作为 1M 唯一可行路径。**该 tradeoff 用户裁决前置到 P2 设计阶段。**
- **依赖**:P1 全绿。Lever C 先出独立 design memo。Lever B 与 C 不同 PR、分别 gate。

---

## 4. 内存预算表(红队口径,红线 90%=10.8GiB @ 12GiB 预算)

> 固定项:scratch/prefill ctx-buf ≈0.45GiB | expert gather staging ≈0.40GiB(split-aware prefetch 落地后可能翻倍至 ~0.80,R5)| **headroom 含 page-cache 双重计数风险,P0 标定后修正**。
> backbone Q4_K aggressive(output/attn_kv/attn_q_b/embd 保 Q8,其余 Q4_K)≈ **5.4-6.0GiB**(精确值待 quantize 实测;比综合设计的 Q5 6.21 更省、kernel 更省事)。

### P1 @200K(Lever-A KV = 1.35GiB)
| 配置 | backbone | +KV | +scratch/gather | peak | 判定 |
|---|---|---|---|---|---|
| PP 今天 Q8(对照) | 8.81 | 1.35 | 0.85 | 11.01 | ✗ 超红线 |
| **EP Q4_K aggressive** | ~5.4-6.0 | 1.35 | 0.85 | **~7.6-8.2** | ✓ 推荐(headroom 待 P0 标定) |
| EP Q5_K(退路) | 6.21 | 1.35 | 0.85 | 8.41 | ✓ Q4 质量不过时用 |

### P2 @1M
| 配置 | backbone | +KV | +scratch/gather | peak | 判定 |
|---|---|---|---|---|---|
| EP Q4_K + 仅 Lever-A | ~5.7 | 5.31 | 0.85 | ~11.9 | ✗ 超红线 |
| EP Q4_K + A+B | ~5.7 | 4.33 | 0.85 | ~10.9 | ✗ 仍超 |
| **EP Q4_K + A+B+C** | ~5.7 | 1.75 | 0.85 | **~8.3** | ✓ 唯一安全(押 Lever C) |

KV 闭式锚(`task3` + roadmap 比例缩放):pre-lever 200K=1.89/1M=8.02;A 200K=1.35/1M=5.31;A+B 1M=4.33;A+B+C 1M=1.75(比 roadmap 乐观 0.50 保守)。量化字节(`quants.c:44-52`,QK_K=256):Q8_0=1.0625 / Q6_K=0.8203 / Q5_K=0.6875 / **Q4_K=0.5625** B/w;backbone 固定 plain-layout ≈1.54GiB 不可降(可量化体占 84%)。

---

## 5. 三档预期 t/s(红队下修,诚实)

| 档 | 冷态单前向 | 依赖 | 编程域有效 t/s |
|---|---|---|---|
| 乐观 | **~1.8** | EP 解耦 + split-aware prefetch + 盘带宽比 SPLIT_LOW 全落地 + 散点带宽乐观端 | 5.40(单机 spec-pipe 封顶,**不进 EP**) |
| 现实 | **~1.3** | EP 解耦 + Q4_K 跑通 @200K,prefetch 部分兑现,散点 0.5-0.7GB/s | 5.25(单机 spec-pipe) |
| 悲观 | **~0.7-1.0** | 仅 EP 解耦、prefetch 未修争盘,冷散点 96-140ms/层退回单机冷 1.02 附近 | 4.7 |

**硬钉**:mini 冷散点盘带宽(实测 0.5-0.7GB/s,非顺序 1.36)× 43 层 lockstep + TB 往返 drain,改码不可越。**暖态单前向保持单机 ~3.8(EP 净负,不在 EP 域)。30 t/s 不由此线交付。**

---

## 6. 任务拆分总表(P0/P1/P2 + 验收闸 + env 开关 + 依赖)

| 任务 | 阶段 | 改码点 | risk | env-gated | 验收闸 | 依赖 |
|---|---|---|---|---|---|---|
| T0.1 L1 闸喂 KV+对账门 | P0 | `ds4.c:19680` | high | 否 | 编译+`--metal-kernels --server`+L1 dry-run+对账<5% | — |
| T0.2 冷散点带宽 probe | P0 | `tools/tp_bw_probe` | low | — | 输出 mini/macbook 散点 GB/s,重算三档 | — |
| T0.3 page-cache footprint 占比标定 | P0 | 实测分析 | low | — | 标定 page-cache 在 footprint 占比 | — |
| T1.1 Q4_K backbone GGUF | P1 | `deepseek4-quantize.c:1109-1193` | med | 否 | 产出 GGUF,头读校验 | T0.1 |
| T1.2 Q4_K dense GEMM(模板) | P1 | `dense.metal:1599` | low | 否 | `--metal-kernels` GEMM 误差门 | — |
| T1.3 Q4_K dense GEMV(专门化) | P1 | `moe.metal:494`→dense, `ds4_metal.m:2217` | med | 否 | `--metal-kernels` GEMV 误差门 | — |
| T1.4 validate_layout 放宽 | P1 | `ds4.c:2922,2932-2940` | low | 否 | server 绿 | T1.1 |
| T1.5 EP 解耦(可选语义化) | P1 | `ds4_distributed.c:10354-10366` | high | 是 | EP parity `--dump-logprobs` | — |
| T1.6 prefill SPLIT_LOW | P1 | `ds4.c:14162` | low | 是 | bit-exact | — |
| T1.7 SPLIT_LOW 带宽比 auto | P1 | `ds4.c:11155-11163` | low | 是 | 冷态 A/B 提速 | T0.2 |
| T1.8 split-aware prefetch | P1 | `ds4.c:~11140` | med | 是 | 冷态 A/B 不净负 | T1.5 |
| T2.1 KV Lever B(FP4 index) | P2 | `dsv4_kv.metal:59-201` | high | 是 | 分 ctx 深度 `--long-context` | P1 绿 |
| T2.2 KV Lever C(disk-tier) | P2 | design memo + ratio-4 路径 | high | 是 | bit-exact parity + 盘竞争预算 | P1 绿 + memo |
| T2.3 k16 fallback 裁决 | P2 | — | — | — | 用户对「256 质量 vs 16 可行」裁决 | P2 设计阶段前置 |

**全部新 env 默认 off / 现状基线,逐个 A/B**:`DS4_EP_STANDALONE`、`DS4_TP_SPLIT_LOW`、`DS4_TP_SPLIT_BW_AUTO`、`DS4_TP_SPLIT_AWARE_PREFETCH`、`DS4_METAL_EXPERT_OFFLOAD`(已有)、`DS4_TP_EXPERT_SPLIT`(已有)、`DS4_TP_EXPERT_SPLIT_BATCH`(已有)。
**双机同步**:本机 Q4_K kernel 重编后,M1 worker 共享 `CORE_OBJS` 必须同步重编 + 传二进制(`feedback_rebuild_check_other_machine`)。

---

## 7. 开放问题(需后续坐实 / 用户裁决)

1. **Q4_K decode GEMV 数值精度**:能否复用 MoE q4_K simdgroup 布局?GEMM/GEMV 两路 kernel 误差能否过隔离 `--metal-kernels`?决定 P1 是数天还是数周。
2. **P1 档位最终选择**:Q4_K aggressive vs 更保守(只 shexp+q_a/o Q4_K),依赖 `--logprob-vectors`/q1..q4 实测,本轮不跑无法定。
3. **Lever C 盘竞争**:@1M ratio-4 disk 读 + expert disk 读叠加是否让 P2 比 P1 显著慢(可能跌破 1 t/s)?需 disk IO 预算建模。
4. **k16 fallback 裁决(R6,前置)**:若 P2 Lever C 难产,k16 是否成为 1M 唯一可行路径(质量换可行)?需用户对「256 质量主线 vs 16 速度可行」优先级裁决。
5. **SPLIT_LOW 冷/暖目标冲突**:冷态按盘带宽(mini 少拿)、暖态 verify 按 compute 均衡——但暖态不开 EP,冷态固定盘带宽档即可?

---

## 8. 与既有文档关系

- `tensor-parallel-feasibility.md` — 禁改码前提下的 no-go 裁决(仍成立于「禁改码」假设);本文是「允许改码」前提下的翻转方案。
- `tensor-parallel-design.md` — 最初调研稿(§3 定量已被 feasibility 证伪,见其顶部横幅);本文取代其方案部分。
- `tensor-parallel-tasks.md` — 旧任务拆分(禁改码版);本文 §6 取代之。
- k16 全驻留备选始终是纯速度目标的 dominant 路径(10-18 t/s,质量 tradeoff 16/256),作为 P2 fallback 与「不追求 256 质量」时的主路候选。
