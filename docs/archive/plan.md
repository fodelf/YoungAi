# DS4 双机（M4 mini + M1 book）200K / 20 t/s：投机解码 + 张量并行方案

> 状态：已批准（plan mode）。本文件是项目内可追踪的执行文档。每个设计/补丁/测量/范围决策/阻塞同时追加到 `notes/execution-log.md`。

## Context（背景与已定方向）

在 ds4（DwarfStar，DeepSeek V4 Flash 专用原生推理引擎，`mac` 分支）上，用两台 16GB Mac 经雷电直连跑通编程用推理：

- **Mac mini = M4（~120 GB/s），≤12GB**（TP rank 0 / 协调）；**MacBook = M1（~68 GB/s），≤8GB**（TP rank 1）
- 雷电直连，mini 侧 `en5`（link-local `169.254.188.38`），≤40 Gb/s（≈5 GB/s）
- 目标：**200K 上下文、20 t/s**；硬约束：**绝不把内存干崩**，对内存克制
- 方法论（用户要求）：**先最小 demo、最小上下文、每模块打执行时间/内存日志、数据驱动决策**

**已定方向**：
- 主架构 = **投机解码（spine）+ 张量并行 over Thunderbolt**（不走层切分串行接力；不复用 `mine` 分支的 expert-replica/mtp-replica/k16 代码）
- **允许重量化 attention（Q8→Q5/Q4）**，但必须过 `ds4-eval q1..q4` 质量门才保留
- **最小集用 k4**（每层 top-4 专家，~10GB，目标塞进单台 M4 跑通管线）

## 模型实数（`ds4.c:155` `DS4_SHAPE_FLASH`）
`n_layer=43`, `n_embd=4096`, `n_vocab=129280`, `n_head=64`, **`n_head_kv=1`(MLA)**, `n_head_dim=512`, `n_expert=256`, **`n_expert_used=6`**, `n_expert_shared=1`, `n_ff_exp=2048`, `n_swa=128`, indexer `top_k=512`。`gguf/`：全量 86.7GB（attn/proj/输出头 Q8、专家 IQ2/Q2），另有 MTP 草稿 3.8GB。

## 地基物理（决定一切，必须先认）

1. **batch=1 解码 = 内存带宽墙**：每 token 把全部激活权重搬一遍，算力 <1% 利用，纯等内存。
2. **真正的地板是 backbone 的 Q8 attention（~7GB/token），不是专家**。剪专家几乎不动速度、只崩质量 → 不作为速度手段。
3. **TP 的两个事实**：
   - **赢点=带宽聚合**：两机各读自己那份权重并行算 → 有效带宽 ≈ 120+68 = 188 GB/s。Q5 后 ~5.5GB/token ÷ 188 ≈ 29ms ≈ **34 t/s 纯算，~26 实测**。
   - **装载关绕不过**：86GB 全量在 20GB 预算下怎么切都装不下。两机**池化内存给专家**后常驻 ≈ **k33 等效（~98%+ 路由命中，优于单机 k16）**；其余冷专家**要么剪、要么 SSD 流式**——物理硬约束，Stage 3 用数据定夺。
4. **投机解码 × 专家offload 互相打架**：批量验证 N 个草稿要加载这 N token 路由到的**专家并集**；冷专家在 SSD 时验证被 I/O 卡死，失去稀疏收益（PowerInfer-2 / MoE-Spec 均点此）。**推论：热专家必须驻 RAM 才能让投机生效**——热专家缓存要覆盖投机窗口的专家并集（实测定 cache 尺寸）。

参考文献：
- batch=1 带宽墙 / 投机解码：premai 博客；EAGLE-3/MLX 分析 https://github.com/ml-explore/mlx-lm/discussions/890 ；MoE-Spec arXiv:2602.16052
- TP over Thunderbolt：EXO https://github.com/exo-explore/exo ；MLX distributed
- 冷专家流式 / 预取：Apple "LLM in a flash" arXiv:2312.11514 ；ExpertFlow arXiv:2410.17954 ；DuoServe-MoE arXiv:2509.07379 ；PowerInfer-2 arXiv:2406.06282

## 推荐架构（一句话）
**TP/专家并行把 backbone+热专家在两机间按带宽 64/36 切分并行算（聚合 188GB/s），attention 重量化到 Q5 降地板，投机解码（MTP）作为可叠加的吞吐倍增**；全程逐模块计时 + 内存看门狗，从最小上下文起步、数据驱动逼近 20 t/s。MTP 是否启用由"TP+Q5 是否已达标"的实测决定（达标则把 3.8GB 让给专家）。

---

## 决定性早期实验（动核心代码前先做）
**E0 — 雷电 all-reduce 延迟基准**：在 `en5` 上写一个最小 ping-pong（TCP_NODELAY，交换 32KB，往返 1 万次）测 RTT 与抖动。TP 每 token 约 **86 次逐层同步**（2 all-reduce × 43 层）。
- RTT ≤ ~50µs → 同步开销 ~2.6–4.3ms/token，**TP 可行，继续**。
- RTT ≥ ~150µs → ~13ms/token+，**TP 性价比崩**，回退到"单 M4 + Q5 + MTP"路线（见 Stage 5 备选）。
- 产出：一张 RTT/抖动表，作为 TP go/no-go 闸门。**这是整个方案的第一个动作。**

---

## 实施阶段（计时优先、最小上下文起步）

### Stage 0：计时 + 内存安全基础设施（最高优先，用户头号诉求）
- **逐模块计时**：仿已有 `DS4_MTP_TIMING`（`ds4.c:19976`）与 `ds4_bench.c::bench_now_sec()`（`clock_gettime(CLOCK_MONOTONIC)`），加 `DS4_PROFILE=1` 开关，在 `model_load` / `prefill_chunk`(`ds4.c:19567`) / `decode_token`(`ds4.c:19873`→`13636`) / 可选每层 attn·moe·sample / `allreduce`(新) / `dist_hop`(透出 `ds4_distributed.c:140` 已有 `eval/downstream_wait/forward_send_usec`) 各夹计时，输出单行 CSV 到 `DS4_PROFILE_FILE`，**关时零开销**。
- **内存看门狗**：采样线程读 `mach task_vm_info`（footprint+RSS），200ms 一次记峰值；**超 12GB(M4)/8GB(M1) 立即 abort 并打印谁吃的内存**。两机各跑一个。
- **执行日志**：每个设计/补丁/测量/范围决策/阻塞追加到 `notes/execution-log.md`。
- 产出：单机最小 demo（M4、`--ctx 4096`、生成 64 token）的首张"逐模块时间 + 峰值 RSS"表。

### Stage 1：k4 最小集跑通 + 单机标定真实地板
- **1a 最小可跑集（k4 → k8 smoke，脚手架非最终配置）**：用 `gguf-tools` 按 per-layer top-K-by-L2-norm 生成 `mask-k4`/`mask-k8`（仅移植**最小的 expert-mask 应用**：路由 logits 把被丢专家置 -inf，或 keep-lut；**不引入** replica/流式那套）。**k4 全量 ~10.1GB，目标塞进单台 M4(12GB) 跑通**——第一次 smoke 不依赖两机，先把"L1 预算闸 → 加载 → 出字 → DS4_PROFILE 计时 → 看门狗"整条管线和内存护栏**在单机验证通**；再用 k8(~11GB) 验证两机 TP 切分与 all-reduce 对拍一致。
- **1b 标定**：`--ctx` 2K→8K→32K，测准：①每 token decode 时长 → 反推激活字节 & 有效带宽（验证 ~7GB/token 与 ~16 t/s 地板）②`prefill t/s` ③KV 字节随 ctx 曲线（外推 200K，确认 MLA 下 KV ~1.x GB，并确认 **TP 下 KV 是否两机各留一份**）④`--mtp` 测 **MTP 接受率/实测加速**（`ds4_session_eval_speculative_argmax`，`ds4.c:19915`）。
- 产出：管线+内存护栏验证通过 + 单机理论上限 + MTP 真实收益 + 200K 内存可行性。

### Stage 2：张量并行最小骨架（核心新建，先只切一处）
- 先实现**最小可验证的 TP**：仅对 MoE 的 `down_proj`（row-parallel）做两机切分 + 1 次 all-reduce，其余仍单机，跑 2–3 层。
- 新代码落点：`ds4_distributed.c` 增 TP 模式（复用现有 TCP 帧 `WORK/RESULT` 或加紧凑 `ALLREDUCE` 帧）；`ds4_metal.m` 加"部分 matmul + 把局部和交给 host all-reduce"钩子；`ds4.c` 图编排里在切分点插同步。
- **Apple 特性降同步开销**：用 `MTLSharedEvent.waitUntilSignaledValue` 替代 `waitUntilCompleted`（TP 小 CB 多，每-CB 调度开销是大头）；统一内存零拷贝；`MTLResidencySet` 预算。
- 闸门：TP 两机输出与单机 greedy **逐 logit 对拍一致**（`--dump-logprobs`）；`DS4_PROFILE` 量出每层 all-reduce 实测 ms，核对 E0 预测。

### Stage 3：扩成全层 TP + 专家并行（装载策略定夺）
- 把 TP 推广到每层标准 2 切：**attention**（q_b 按 head 列切 → output proj 行切 → all-reduce#1）；**MoE**（router 复制、专家按 owner 切到两机 → 加权专家输出 all-reduce#2；shared expert 切分）。
- **专家并行池化**：两机各存**不相交**的专家子集，合起来常驻 ≈ k33 等效（按各机 RAM 预算分配 owner）。冷专家处理二选一，用数据决定：
  - **a 静态剪（mask）**：丢弃尾部专家 → 等效 k33 静态剪枝（已优于 k16），零运行时 I/O，简单。
  - **b SSD 流式 + 预取**（LLM-in-a-flash / ExpertFlow 路线）：保全量 256，路由器先出选择 → 预取冷专家；但受"投机×offload"张力约束，须热缓存覆盖投机窗口。
  - 先做 **a** 拿到可用基线，质量不足再上 **b**。
- 按"使 `M4_share/120 ≈ M1_share/68`"的**带宽比 64/36** 分配各机权重份额并实测微调。
- 闸门：两机出字正确；峰值 RSS M4≤12 / M1≤8；`ds4-eval q1..q4 --temp 0 --seed 1` 质量可接受。

### Stage 4：attention 重量化 Q8→Q5/Q4（攻地板，质量门把关）
- 用 `gguf-tools` 把 attention（q_a/q_b/kv/output_a/b）与输出头从 Q8 重量化到 Q5_K（必要时 Q4_K），backbone ~8.85GB→~6GB，每 token 激活 ~7GB→~5.5GB。
- **质量门（铁律 correctness-before-speed）**：`ds4_test --logprob-vectors`（对官方向量）+ `ds4-eval q1..q4` 必须过；不过则回滚该张量到 Q8。逐张量灰度，记录每步 Δt/s 与质量。

### Stage 5：投机解码叠加 + 拉到 200K + 逼近 20 t/s
- 若 Stage 3–4 实测已 ≥20 t/s：MTP **可选**（启用要权衡 3.8GB 占用 vs 1.8× 加速；on-host MTP 在 `mac` 分支现成）。若未达标：启用 MTP，调 `--mtp-draft N`，并确认热专家缓存覆盖投机窗口专家并集（否则验证被冷专家 I/O 拖累）。
- 上下文 32K→100K→200K 逐档，看门狗确认两机不破线；必要时 `--kv-disk-dir` 落盘非活跃 KV。
- **备选路线（E0 判 TP no-go 时）**：单 M4 + Q5 backbone + on-host MTP + 小热专家集（M1 退化为纯 KV/冷专家盘），靠 MTP 1.8× 把单机 ~20 t/s 拉到 ~30 effective。
- 产出："配置 → 实测 sustained t/s @200K / 峰值 RSS / 质量分"对照表 + 诚实上限结论。

---

## 内存预算（目标，Stage 实测校正）
| | M4 mini (≤12GB) | M1 book (≤8GB) |
|---|---|---|
| backbone(Q5) 份额 | ~3.6GB | ~2.0GB |
| 专家(owner 子集) | ~6.0GB | ~3.5GB |
| KV 份额 @200K | ~0.8GB | ~0.6GB |
| scratch/activation | ~1.2GB | ~1.0GB |
| 合计 | ~11.6GB | ~7.1GB |
| MTP(可选) | 启用则挤占专家容量 | — |

## 计时日志（落点）
| 模块 | Hook | 度量 |
|---|---|---|
| 加载 | `model_open`/mmap | 墙钟+RSS 增量 |
| 预填充 | `ds4.c:19567` | per-chunk ms, prefill t/s |
| 解码 | `ds4.c:19873/13636` | per-token ms |
| all-reduce | TP 同步点(新) | 每层 µs + 抖动 |
| 投机 | `ds4.c:19915/19976` | 接受率, draft/verify ms |
| 跨机 | `ds4_distributed.c:140` | eval/wait/send µs |

统一 `DS4_PROFILE=1` + `DS4_PROFILE_FILE`，仿 `DS4_MTP_TIMING`。

## 内存安全：三层保证（直接回答"凭什么不爆"）
不是边跑边看，是**加载前闭式算清、装不下就拒绝启动**。
- **L1 起飞前静态预算闸（拒绝启动）**：每台机绑任何 GPU buffer 前算
  `resident = backbone份额 + owner专家数×每专家字节 + KV(ctx, ds4.c:14847 闭式) + prefill_scratch(chunk 闭式) + 固定开销`；
  **`> 预算×0.85` 直接 abort 打印明细**。计划内 OOM 在加载前被挡死。
- **L2 构造上有界（不可悄悄超）**：①懒 mmap、**关 WILLNEED 预读**（避免亚秒灌爆）②**view-shrink**：Metal 只 wire 本机 owner 的逐张量小视图，绝不绑整个 86GB buffer（规避 IOGPU 整-buffer wire 陷阱）③最小集用物理上就小的 k4(~10GB)/k8(~11GB)，最坏全 fault ≤ 文件大小，file-backed 干净页可回收（页缓存压力≠OOM-kill）④prefill scratch 用 `DS4_METAL_PREFILL_CHUNK` 封顶；KV 起始一次性预分配 fail-fast；all-reduce 缓冲 KB 级预分配 ⑤TP 下 MLA latent KV 跨 head 共享 → 近似两机各留一份(~1.4GB)，已计入预算。
- **L3 运行时看门狗（兜底）**：每 200ms 采样 **`mach task_vm_info` 的 `phys_footprint`**（**不是 `ps rss`**——Stage 1a 实测 ps rss 对 Metal no-copy mmap 失真，9.3GB 只报 38MB），**超 预算×0.9 在换页 thrash 前 abort/SIGKILL**，两机各一个。

铁律仍在：严禁单机双载 86GB；macOS 禁 CPU 推理路径（内核 VM 崩溃），全程 Metal；冲突时**保内存/保平稳 > 保速度**。

## 验证（端到端）
- **正确性**：`make`(Metal) + `ds4_test --metal-kernels` + `ds4_test --server`；TP/单机 greedy `--dump-logprobs` 对拍一致；`ds4_test --logprob-vectors`（重量化后）。
- **质量**：`./ds4-eval -m <model> --plain --questions 4 --tokens 2048 --temp 0 --seed 1` 对照 README 期望 token 数（每次改专家集/重量化后跑）。
- **速度**：`ds4-bench` 或 `DS4_PROFILE` 测 200K 前沿 sustained gen t/s。
- **内存**：看门狗日志确认峰值 ≤12/8GB，全程无 OOM/panic。

## 风险与诚实结论
- **TP 成败系于 E0（雷电 all-reduce 延迟）**：先测，no-go 则走单 M4+Q5+MTP 备选。
- **20 t/s 可达性**：TP+Q5 推算 ~26 实测（过 20 有余量），但依赖 all-reduce 开销可控 + M1 不掉链；最终以实测为准，不预先打包票。
- **装载硬约束**：20GB 装不下全量 256 专家，必须 k33 静态剪或 SSD 流式二选一；前者已优于 k16，后者保全量但受投机×offload 张力限制。
- **重量化风险**：attention 降精度可能伤 correctness，逐张量质量门把关、过不了即回滚。
- **TP 是主要新建工作量**（图编排 + all-reduce + 专家并行 owner 切分），分阶段最小验证、每步实测护栏。

## 实测结论（Stage 1a，2026-05-30）

已完成 k4 生成 + reduced-expert loader 移植 + 单机加载 smoke，拿到第一批真实数据：

- **k4 文件**：`gguf/ds4flash-k4.gguf` = **10.02 GB**（每层留专家 0..3）。离线生成峰值 RSS 198 MB、10.8s，内存安全实证。
- **物理论点被证实**：可行性报告实测 **backbone（非专家）= 8.81 GB**（= 带宽地板），专家 77.91→1.22 GB。
- **loader 工作**：mac 分支已能加载 reduced-expert GGUF（`ds4.expert_keep_map.*`），编译全绿，无 exit(1)。现有 k16/k48 也因此可用。
- **加载内存安全**：k4 在单 M4 驻留 **9.3 GB**（2 overlapping shared buffers），无 OOM/无崩溃。
- **基线速度（单 M4, k4, ctx2048）**：prefill **17.43 t/s**，generation **10.58 t/s**。
  - k4 只 4 个专家、专家计算极轻，仍只 ~10.6 t/s → **再次坐实：瓶颈是 Q8 attention backbone，不是专家**。这正是 TP 带宽聚合 + 投机 + Q5 重量化要攻的地板。
- **⚠️ 看门狗度量修正（重要）**：`ps -o rss` 对 Metal no-copy mmap（MAP_SHARED）**完全失真**（实 9.3GB 只报 38MB）。**Stage 0 看门狗与 L1 预算闸必须用 `mach task_vm_info` 的 `phys_footprint`，不能用 ps rss。**（已在下方 L3 注明）
- **待补**：k4 当前输出垃圾（重复 BOS），因 GPU `route_translate` 未补 → 专家 id 用原始值索引 4-专家张量（越界被 GPU 驱动沙箱化，不崩不爆内存）。补齐后才能跑 ds4-eval 质量 / 有意义的多 token 速度。

## 进度
- [x] 方案落地 `plan.md`
- [x] 生成 k4 模型（10.02GB，内存安全离线路径，峰值 RSS 198MB）
- [x] reduced-expert loader 移植到 mac（ds4.c，编译全绿，k4 实测可加载）
- [x] Stage 1a 单机加载 smoke（内存安全 9.3GB + 基线 gen 10.58 t/s）
- [ ] **GPU route_translate**（让 k4 输出正确：metal kernel + LUT + 两个专家 matvec 函数 wiring + load 时 set；k4-first-4 的 LUT 逐层相同→单层 LUT 即可）
- [ ] Stage 0 计时（DS4_PROFILE）+ **phys_footprint** 内存看门狗（修正后的度量）
- [ ] E0 雷电 all-reduce 延迟基准（TP go/no-go）
- [ ] Stage 2 TP 最小骨架
- [ ] Stage 3 全层 TP + 专家并行
- [ ] Stage 4 attention Q5 重量化
- [ ] Stage 5 投机叠加 + 200K
