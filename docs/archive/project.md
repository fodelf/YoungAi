# project.md — 双机 q2 全量模型底层提速方案（0.81 t/s → 物理极限逼近）

> 状态：方案落地稿（2026-06-10）。承接 `plan.md`（TP/投机路线）与 `mtp.md`（双机 MTP），
> 针对当前 **q2 全量模型 + A3 按需专家** 路线已到 0.81 t/s 的瓶颈，给出更底层的
> 内存分区排布 + 算法升级路线。执行纪律不变：每个补丁/测量/决策追加 `notes/execution-log.md`。

## 0. 硬约束与现状

- 设备：M4 mini 16GB（≤12GB，coordinator）+ MacBook M1 系 16GB（**≤12GB**，worker），雷电直连 ≤40Gb/s。
  （2026-06-10 设定更新：两台都放宽到 12G，合计 24G；worker 由 8G→12G，红利已投入
  worker backbone mlock 5120MB + 硬预算 12000MB，见第三十二波。）
- 模型：**必须用 `gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`（81GiB q2 全量）**。
- 目标：200K 上下文，Claude Code 可用；**用户期望 30 t/s（编程域有效 t/s；2026-06-17 从 80 修正）**。
- 验证：`tools/mtp_pipe_q2_speed.sh`；日志 `/tmp/mtp_pipe_coord.log|.out`、worker `192.168.1.2:/tmp/mtp_pipe_worker.log`、build 日志同前。
- 现状（2026-06-17）：**code-edit gen 3.77 t/s / smoke gen ~2.05–2.17 t/s / prefill ~10–12 t/s**（逐波实测见 `log.md`），
  拓扑 `0:19 / 20:output`，A3 流式 gather + overlap N=2 + copy-spec（code-edit fire 70%），`DS4_DIST_PREFILL_CAP=2048`。
  （起点基线 2026-06-10：prefill 0.78 / gen 0.77–0.81。）

## 1. 物理上限审计（先认账，再谈方案）

模型实数（`ds4.c` DS4_SHAPE_FLASH + 加载日志实证）：

- 43 层**全部带 routed MoE**（coordinator `0:19` 日志=60 expert spans=20 层×3 张量，33.75GiB=20×1.6875GiB）。
- 每专家 = gate 2.06MiB(IQ2_XXS) + up 2.06MiB + down 2.625MiB(Q2_K) ≈ **6.75MiB**；
  每层 256 专家 ≈ **1.69GiB**；全模型 routed experts ≈ **72.6GiB**；backbone(Q8 attn+shared+head) ≈ **8.4GiB**。
- 每 token 激活字节：routed 43×6×6.75MiB ≈ **1.70GiB** + backbone ~7–8GiB（RAM 常驻读）。

**三道墙（按从紧到松）：**

| 墙 | 闭式 | 含义 |
|---|---|---|
| W1 容量墙 | 72.6GiB experts vs 两机合计可用 **24GiB**（2026-06-10 起两台都放宽到 12G；扣双侧 backbone mlock ~9.7G + ctx 3.6G + 运行时 ~2G，**可给专家缓存 ~8-9GiB**，此前仅 ~2.2G） | ~85% 专家访问仍落 SSD，但 decode 热集（路由偏斜）可观缓存 |
| W2 SSD 墙 | 1.70GiB/token ÷ (双机 SSD 并行 ~6GB/s) ≈ **280ms/token** | 全冷、完美并行/零拷贝下 decode ≤ **~3.5 t/s**；叠 8-9GiB 缓存 40-60% 命中 ≤ **~6-7 t/s**。24G 重估的现实上限：smoke ~**3.0-3.4**（IO+drain 串行链），code-edit 纯复制区已实测 222ms/tok≈4.5，drain（占 r2 40-45%）砍半后区段 ~5-6、整体 **~4+** |
| W3 backbone 带宽墙 | coord 4.07GiB/120GB/s=34ms + worker ~4.4GiB/68–200GB/s=22–65ms | 专家全部免费也只有 **10–18 t/s** |

盘速不对称（第三十三波实证，影响所有"谁从谁的盘拉"决策）：mini 盘顺序 ~1.9GB/s 且兼任
efetch server；MacBook 盘明显更快（decode 小读 8-21GB/s 突发）。因此 **coordinator←worker盘
的 staging 赚（completion 97%）；worker←mini盘 的 staging 亏（completion 64%、净 -18%）**；
racing 只在 ≥96 单元的大批形态下两盘并行才稳赚。

**80 t/s 判定：不可达。** 需要每 token ~9.4GiB 权重在 12.5ms 内供给 ⇒ ~750GB/s 聚合带宽 + 81GiB 全驻留。
两台 16GB Mac 差 ~40× 内存、~4–6× 带宽。该量级只存在于 M3/M4 Ultra 192GB（~800GB/s）单机或全驻留小模型。
（旁注：`gguf/ds4flash-k16-v2.gguf` 13GiB 同为 q2 量化，双机全驻留理论 10–18 t/s——P4 决策门备选。）

**30 t/s 判定（2026-06-17，目标从 80 修正为 30）：单次前向 decode 不可达，但编程域"有效 t/s"可达（stretch）。**
单前向天花板由 **W3 backbone 带宽墙** 决定（不是 SSD）：coord 4.07GiB÷120GB/s≈34ms + worker 4.4GiB÷(68–200GB/s)≈22–65ms，
自回归单 token 是 coord→worker **串行** ≈ 56–99ms ⇒ **10–18 t/s**——这是**专家全部免费驻留（零 SSD IO）下**的上限，
所以即便把 W1/W2 容量/SSD 墙彻底消掉，纯解码 30 仍在墙外。**唯一到 30 的物理路径 = §3.5 编程域有效 t/s**
（copy-spec 多 token/前向 + 前缀复用少前向 + 降激活越过单前向墙），且只在 echo 重的编辑回合成立，非跨回合稳态。

## 1.5 bit-exact 上限审计（2026-06-14，第三十八波，code-edit 3.77 跑实证）

7 连波把 code-edit 0.81→3.77。第三十七波后做了一次彻底审计，**实测数字证明 q2 双机已触及"保正确性(bit-exact)上限"**，剩余只有违反对拍门的杠杆。逐条：

1. **IO work-stealing 分配已达带宽最优**（不是估计，是闭式验证）。verify 批每层本地 pread + 远程
   efetch 抢同一 cursor，自然按盘速分流。worker 批：总 30813MiB/wall 4508ms=6.67GB/s 聚合；
   最优分配应满足 local/5.2 = remote/1.62 ⇒ local=23493/remote=7320，**实测 23512/7301**。
   coord 批：local(mini本地)3.07 ≈ remote(worker经TB)3.12GB/s ⇒ 最优~50/50，**实测 14789/15056**。
   两台都正落在带宽最优点 ⇒ 重新分流无收益。
2. **walls 贴物理盘+TB 顶**：coord ~6.0GB/s（mini本地+worker经TB），worker ~6.7GB/s（worker本地+mini经TB）。
   再快需要更快的盘或更宽的 TB（硬件）。
3. **drain 是 GEMM / M1-GPU 墙**：verify drain ≈1.4ms/token/层（mm_id 自 wave-23 已调优）；
   smoke 的 worker decode gather wall 4121ms@10.8GB/s(背景 madvise 已暖页) **< drain 6100ms(GPU)**
   ⇒ worker decode 是 M1 GPU 算力受限，gather 已被隐藏，无可重叠。
4. **重叠的可行性（2026-06-14 修正，原结论有误）**：
   - **跨机 token 流水**：把 33-token 批拆 2 chunk 跨机流水，实算 dilution(walls×1.49)>overlap ⇒
     4204ms > 单批 4065ms，**净亏**（用真实带宽数算的，不是旧估计）。排除。
   - **层内专家分半重叠 = 唯一有效杠杆，且 BIT-EXACT**（修正：原以为破坏浮点结合律，错）：
     MoE 的专家求和（`ds4_gpu_encode_moe_sum6`/`sum_experts`）是**按每 token 的 6 个 pick 固定顺序**
     累加；`remap_selected_to_slots` 把每 pick 改写为紧凑 scratch 的 slot，mm_id 是 expert-major。
     按 **slot 区间** 把 union 分两组（pass1 处理 slot[0,h)、pass2 处理 slot[h,n_active)），
     每个 `(token,pick)` 的 down 值逐位不变、写入其固定行，最后**一次** sum_experts 顺序不变
     ⇒ **逐位一致，过 `--dump-logprobs` 严格对拍门**。重叠点：pass2 的 gather(磁盘) 与 pass1 的
     GEMM(GPU) 并行（两块独立 scratch，GPU 读 scratchA 时 CPU 填 scratchB，无别名）。
     潜在收益：把 drain(~45% of r2) 藏到 walls 后 ⇒ r2 5679→~max(walls,drain) 区间 ⇒ **~15-17%**。
   - **跨层重叠**（gather L+1 ∥ drain L）：bit-exact，但 L+1 路由数据依赖 ⇒ 需预测；coord 侧
     staging(从 worker 快盘)已在做(73%命中)，worker 侧只能从 mini 慢盘预测 ⇒ 第三十三波已证净亏。
5. **结论修正**：仍有一个 **bit-exact** 大杠杆——层内专家分半重叠（P-OVL）。**不需要放宽正确性契约**。
   它是对最热函数 `ds4_gpu_routed_moe_batch_tensor` 的较大改造，**第四十～四十一波已落地**（默认 OFF）：
   - **单 scratch 不相交 slot 区间**（非双 scratch）：pass0 填 `[0,half)`、pass1 填 `[half,n_active)`，
     写不相交 ⇒ wave-40 的 slot 区间门保证 GPU 读 pass0 区间时 CPU 填 pass1 区间无别名，**内存中性**。
   - `flush_commands` 异步提交 pass0（GPU 开跑）让 pass1 的 CPU gather 与之重叠；gather dst 参数化后
     range-gather 不需改 worker（`active_ids+lo`/`dst+lo*bytes`/`n=hi-lo`）。
   - 自包含 overlap 分支（提前 `return`，linear 路径字节不变）。`make`+`metal-kernels`+`server` 绿。
   验收：`DS4_METAL_MOE_OVERLAP=1` 下 `--dump-logprobs` 对 A3 基线逐位一致 + A/B 速度（先 OFF 保 3.77，
   验证后再默认 ON）。预期 +7~17%（受 `T_gpu/2` 约束：gather 是物理地板，重叠只藏 GPU 计算）。

**本方案的诚实目标阶梯（q2 全量模型不变）：** 当前 code-edit 实测 **3.77 t/s**，用户目标 **30 t/s（编程域有效 t/s）**；逐波实测进展（M0.5 起逐波 + §第五十八波结论）已迁出至 [`log.md`](log.md)。

| 里程碑 | decode | prefill | 依赖 |
|---|---|---|---|
| M0 起点 | 0.81 | 0.78 | — |
| M1 单拷贝直读 + prefill 流式 | **≥1.6** | **≥10** | P1.1/P1.2 |
| M2 预测预取流水 + 命中零拷贝 + repack | **≥3** | **≥25** | P1.3/P1.4/P2.1 |
| M3 expert-parallel 双 SSD 并行 | **≥5** | ≥30 | P2.2 |
| M4 投机/路由偏置（质量门） | 6–8（stretch） | — | P2.3/P2.4 |
| **M4C 编程场景专项（用户目标）** | echo 重编辑回合**有效 decode ≥30**（copy-spec+前缀复用+降激活；非跨回合稳态） | 增量 4K prefill ≤25s | P-Code（§3.5） |

当前 code-edit 3.77 t/s 已逼近单前向 W2/W3 墙（~5–7 t/s 区间）——单前向“优化到极限”≈5–7，**到 30 t/s 必须走 §3.5 有效 t/s**。
**注意：W1–W3 三道墙限制的是“每次前向”的物理成本；编程场景的专项空间（§3.5）在于让
一次前向产出多个 token（复制式投机）和少做前向（前缀/画像复用）——即“有效 t/s”可以越过单前向墙。**

## 2. 联网调研结论（2026-06-10 检索 arXiv/开源）

**测得的瓶颈是什么**：本轮 coordinator 短跑复制 ~126GiB/84s ⇒ 有效 gather 带宽 **~1.5GB/s**，
路径 = mmap 缺页（SSD→页缓存）+ memcpy（页缓存→Shared scratch）双拷贝 + 每层 command drain 串行。
SSD 本身 3–7GB/s、雷电 ~4–5GB/s，都没被打满。文献正好全部攻这一点：

| 文献/项目 | 关键机制 | 映射到本项目 |
|---|---|---|
| Apple LLM-in-a-flash (arXiv:2312.11514) | **row-column bundling**：把一次要用的权重在闪存上排成连续块；windowing 缓存 | P1.3 GGUF 专家重排（gate/up/down 按 (layer,expert) 捆绑连续） |
| OD-MoE (2512.03927) | 边缘多机 **cacheless 按需加载**＋跨多层超前 **emulative 路由预测**，加载/计算跨节点并行 | P2.1 跨层预测预取；P2.2 双机 expert-parallel |
| Pre-Attention Expert Prediction (2511.10676) | 注意力前激活 + 线性映射即可保序预测本层/后层专家，近零开销 | P2.1 预测器实现选型 |
| SpecMD (2602.03921)；In-depth Caching Analysis (2511.05814) | 实测 **专家访问不符合 LRU/LFU 时间局部性假设**，预测驱动缓存优于 LRU | 解释我们 LRU source-cache 命中 0%；P1.4 缓存策略改成“频率钉住+预测填充” |
| Not All Models Suit Expert Offloading (2505.16056) | SRP/SCH 两个指标量化模型路由局部性，决定 offload 是否值得/缓存尺寸 | P0.2 先对 DS4 Flash 在 Claude-Code 流量上测 SRP/SCH，再定缓存预算 |
| SP-MoE (2510.10302)、MoE-SpeQ (2511.14102)、SpecExec (2406.02532) | **投机×offload 协同**：草稿批共享专家并集、按草稿预取，offload 场景下投机才有正收益 | P2.3 MTP 重启条件（先修 IO 再开 MTP） |
| SMoE 专家替换 (2508.18983)、AdapMoE (2408.10284)、ReMoE (2605.27081) | 低权重专家跳过/用已缓存近似专家替换/路由偏置缓存 | P2.4 cache-aware 路由偏置（推理期、质量门把关） |
| PowerInfer-2 (2406.06282)、HybriMoE (2504.05897)、KTransformers、llama.cpp `--n-cpu-moe`/`-ot exps=CPU` | IO-计算流水、CPU 算驻留侧专家避免搬运 | P1.2 流水隔离；CPU 算冷专家在 2-bit/M 系上预计不赢（备忘不立项） |
| prima.cpp (2504.08791)、EXO、MLX distributed（雷电 ring/pipeline） | 家用异构集群 piped-ring + 预取；MLX 是“定向优化”范本但无 SSD-MoE 流式 | P2.2 拓扑参考；确认无现成轮子可直接抄，须在 ds4 内自研 |
| Metal 3 `MTLIOCommandQueue`（Apple 官方 fast resource loading） | **文件→MTLBuffer 直载**，高队列深度，绕 CPU memcpy（macOS 的 GPUDirect-Storage 等价物） | P1.1 首选实现 |
| LLMA (2304.04487)、REST (2311.08252)、SuffixDecoding (2411.04975)、CopySpec (2502.08923)、prompt-lookup（vLLM ngram / HF assisted-decoding） | **无模型复制式投机**：草稿=从上下文/历史输出 n-gram 匹配直接抄，零显存零草稿计算；SuffixDecoding 实证 agentic/重复型负载收益最大 | **P-Code（§3.5）**：Claude Code 输出大量回显上下文字节（Edit old/new_string、代码改写、重复工具调用），是该技术的理想负载 |
| DeepSeek MoE 专家领域特化、Local Routing Consistency (2505.16056) 域内一致性更高 | 代码域 token 的专家路由分布显著更偏斜/更稳定 | PC.2 编程域热专家画像（离线 profile + 启动预热钉住） |
| TurboQuant (2504.19874, ICLR'26) 及旋转量化族 EDEN/RaBitQ/rotorquant | FWHT 随机旋转 + 逐坐标最优标量量化的**在线/数据无关 KV 量化**（3bK/2bV，~4× KV 压缩） | §2.1 专项结论：不攻我们的主墙，三个间接参考点分别挂 P0.1 / P2.2 / P3 |
| SwiftLM（SharpAI，MLX Swift，Apple Silicon） | **SSD 流式 MoE + TurboQuant KV + MTP** 的同类开源实现；26B-A4B 4bit SSD 流式在 M5 Pro 实测 9–10.8 t/s | P1 存在性证明 + `needsMoeFlush` per-layer 同步教训（见 §2.1） |

### §2.1 TurboQuant 专项调研结论（2026-06-10，应用户要求）

**TurboQuant 是什么**：在线、数据无关的向量量化（FWHT 旋转把坐标摊成近高斯 → 逐坐标最优标量量化；
可选 1-bit QJL 残差保内积无偏），主战场 = **KV 缓存量化**（3-bit K / 2-bit V，纯密集注意力下 ~4.4×）。
生态：vLLM/Triton 移植（0xSero/turboquant）、llama.cpp 系（rotorquant/quant.cpp）、MLX Swift（SwiftLM）。

**对本项目的判定：主路线不采用，三个间接参考点采纳。**

1. **不攻我们的墙**：本项目瓶颈是专家 SSD IO（1.70GiB/token），不是 KV——DS4 Flash 的 KV 已被
   MLA(n_head_kv=1) + 滑窗 + ratio-4 indexer + ratio-128 压缩层原生压掉，200K 下 context buffers
   仅 ~1.83GiB/机。**SwiftLM 自己的对照表是最好的警示**：全驻留模型上 TurboQuant 让 100K decode
   从 25.8→62.1 t/s（KV 读带宽是它的墙）；但 **SSD 流式（=我们的形态）下 SSD+TQ 反而从
   10.4→2.5 t/s**——IO-bound 时再加去量化算力是负收益。结论与我们 W1–W3 审计一致。
2. **学术警示（不绑品牌）**：后续 note 指出 TurboQuant_mse 是 EDEN 的次优特例（2604.18555）、
   多数 KV 场景被 RaBitQ 反超且有复现问题（2604.19528）。若未来做旋转量化，做**族选型对比**
   （EDEN/RaBitQ/TurboQuant），不预设 TurboQuant。
3. **真正带回来的三个参考点（已挂到对应 P 项）**：
   - **(a) `needsMoeFlush` 教训 → P0.1**：SwiftLM 在全驻留路径误开 SSD 流式所需的 per-layer GPU
     同步屏障，修掉后 vanilla 提速 3.4×（SwiftLM #84）。我们的 direct 模式默认
     `DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS=1`、A3 模式每层 command drain——**同样的屏障税可能在
     非必要路径上常开**，P0.1 必须单独量出 drain_ms 并 A/B 关闭试验。
   - **(b) 旋转量化做跨机激活压缩 → P2.2 备选**：数据无关+在线+内积无偏正是激活传输想要的性质；
     decode 每跳 16KB 是延迟绑定（无收益），仅当 P0.3 实测 prefill 大 chunk 的 hidden 传输
     （4–32MB/跳）成为可见项时，对其做 4–6bit 旋转量化（3–4× 传输降）。
   - **(c) KV 再量化释放预算 → P3 低优先备选**：对 raw 滑窗/indexer-selected KV 做 3bit 旋转量化
     可再挤出 ~0.5–1GiB/机给专家池，但收益有限且 dequant 不得落 decode 热路径；
     铁门槛 `--logprob-vectors` + 长上下文 fact-recall 回归，过不了即弃。
   - 顺带：SwiftLM 的 SSD 流式 9–10.8 t/s（M5 Pro，单机，激活字节更小）是 P1 路线的
     **存在性证明**——Apple Silicon NVMe 流式 MoE 能跑到两位数 t/s 量级，我们 0.81 远未到供给极限。

## 3. 方案总览

```
P0 测量基建（半天）→ P1 内存分区排布（主攻，~1-2 周）→ P2 算法升级（预测/拓扑/投机）
→ P-Code 编程场景专项（与 P1/P2 并行推进，PC.1 可提前）→ P3 200K/Claude Code 落地 → P4 决策门
```

### P0 量化测量（动手前先把账测准）

- **P0.1 gather 分解计时**：在 `ds4_gpu_load_layer_experts_to_scratch`（`ds4_metal.m:16895`）按层记录
  `n_active / cold_bytes / fault_ms / memcpy_ms / drain_ms`，env `DS4_METAL_EXPERT_IO_PROFILE=1`，单行 CSV。
  分清 1.5GB/s 里缺页、memcpy、每层 drain 各占多少。
  **屏障税专查（SwiftLM #84 教训，§2.1）**：per-layer split/drain（`DS4_METAL_GRAPH_TOKEN_SPLIT_LAYERS`、
  A3 每层 command drain）是否在非必要路径常开？单独 A/B 关闭实验——同类项目修掉误开的
  per-layer 同步后全驻留路径提速 3.4×，我们的 drain_ms 若占比高，这是最便宜的回收项。
- **P0.2 路由局部性标定（SRP/SCH，2505.16056）**：用现有 expert-profile 日志 + 一段真实 Claude Code 流量
  （工具调用/代码补全 prompt），离线算每层 top-k 覆盖率曲线：k=16/32/64 专家能覆盖多少 token？
  连续 64/512 token 段的最优缓存命中上限是多少？**这个数直接决定 P1.4 缓存尺寸和 P2.4 是否立项。**
- **P0.3 硬件底数**：两机 `dd`/`fio` 风格顺序+随机(2/7MiB) SSD 读带宽；`tools/e0_pingpong.c` 雷电 RTT
  （P2.2 的 go/no-go：RTT ≤100µs 才做 per-layer 跨机同步）。
- 产出：`notes/execution-log.md` 一张账表：每 token 时间 = fault + memcpy + drain + GPU + 跨机，对上 1.23s。

### P1 内存分区排布（主攻方向一：把 1.5GB/s 供给抬到 5–8GB/s）

- **P1.1 单拷贝直读（最高优先）**：冷专家不再走 mmap 缺页+memcpy 双拷贝。
  - 首选 **`MTLIOCommandQueue`**：`MTLIOFileHandle` 直接把 GGUF 中 (offset,len) 载入 scratch/pool 的
    `MTLBuffer`，QD 高、零 CPU 拷贝；macOS 13+，两机均满足。
  - 退路：`pread(fd)` 直读进 `MTLBuffer.contents`（仍单拷贝），IO 线程池 QD 4–8；冷读加 `F_NOCACHE`
    防止冲掉热页缓存（A/B 实测决定）。
  - 落点：`ds4_gpu_expert_gather_copy_slot()` 的 mmap-memcpy 分支替换为 IO 提交；保留 mmap 路径做对拍。
  - 门槛：`--dump-logprobs` 与 A3 路径逐 logit 一致；gather 有效带宽 ≥3GB/s。
- **P1.2 prefill 全层顺序流式（最便宜的大头）**：prefill 一个 ≥128-token chunk 每层几乎激活全部 256 专家
  ⇒ 不该按专家 gather，应**整层 1.69GiB 顺序读**＋IO/GPU 双缓冲流水（读第 L+1 层时算第 L 层）。
  顺手把 `DS4_DIST_PREFILL_CAP` 128→512/1024（先单独 A/B，这是零代码快赢）。
  预期 prefill 0.78 → 10–40 t/s；200K 冷 prefill 从 ~71h 降到 ~1.5–5h，配合 KV 盘缓存增量复用才可用（见 P3）。
- **P1.3 GGUF 专家捆绑重排（row-column bundling）**：`gguf-tools` 新增离线 repacker，生成 sidecar
  排布文件（不破坏 GGUF 兼容）：(layer,expert) 的 gate+up+down 连续存放、16KiB 对齐
  ⇒ 每专家 3 次随机读 → 1 次 6.75MiB 顺序读。
  进阶：用 P0.2 共激活矩阵做贪心聚类，把常共现专家排相邻 ⇒ 单 token 6 个专家的读可合并成更长顺序段。
  门槛：repack 前后 logprob 对拍一致；随机读带宽提升实测 ≥1.5×再保留。
- **P1.4 热专家常驻池修复（命中必须零拷贝）**：现 pool 的两个致命缺陷——命中后仍 memcpy 到 scratch、
  GPU 从大 Shared pool 读慢——改成**间接表**：MoE kernel 增加 per-slot 指针/索引 buffer，
  专家源 = pool slot 或 scratch slot 二选一，**命中零拷贝零搬运**。
  缓存策略弃纯 LRU（SpecMD 实证无时间局部性）：P0.2 频率分布静态钉住 top-N + 预测器异步填充其余。
  预算再平衡：coordinator 把 KV/scratch 压缩后余量全部给池（~3–4GiB ≈ 500+ 专家）；worker 维持 ~0.5–1GiB。
  门槛：端到端 ≥ 不开池基线（杜绝 0.81→0.15 的历史回退），整层 active-set 全命中率（非单专家命中率）作主指标。

### P2 算法升级（主攻方向二：把 IO 藏进计算、把两台 SSD 并起来）

- **P2.1 跨层路由预测 + 异步预取流水**（OD-MoE / Pre-Attention 路线）：
  router 是 4096×256 的微矩阵，用第 L 层（或注意力前）hidden 提前评估 L+1..L+Δ 层 router，
  预测准了就提前 Δ 层提交 IO ⇒ 专家读与 attention/shared-expert 计算重叠，token 时间从
  Σ(IO+compute) 收敛到 ≈Σmax(IO,compute)。Δ=2–3 起步；预测错 fallback 同步直读（正确性不依赖预测）。
  先离线验证预测精度（用 P0.1 日志回放，目标 top-6 命中 ≥85%），再上线。
- **P2.2 layer-pipeline → expert-parallel 混合拓扑**：现拓扑两台机器对单 token **串行**（IO 也串行）。
  改为每层 256 专家按 owner 切两半（按 P0.2 共激活聚类 + 各机预算分配），attention/backbone 仍层切分，
  routed experts 双机**并行**读各自 SSD、并行算，per-layer 交换 16KB hidden/partial-sum
  （43×2 次/token，RTT 80µs ⇒ ~7ms/token，P0.3 把关）。专家 IO 墙直接减半（W2 从 ~3.5→~7 t/s 上限）。
  落点：`ds4_distributed.c` 增 `EXPERT_SHARD` 帧与 per-layer 归约；`ds4.c` 图编排切分点；协议沿用滚动哈希。
  备选低成本版：保持层切分，但空闲方 SSD 经雷电给忙方流专家字节（聚合 IO ~6GB/s，改动小、先验证收益）。
  备选（§2.1b）：若 P0.3 实测 prefill 大 chunk 的 hidden 传输（4–32MB/跳）可见，
  用旋转族量化（EDEN/RaBitQ/TurboQuant 选型对比）压到 4–6bit；decode 16KB/跳延迟绑定，不做。
- **P2.3 MTP 投机重启（IO 修好后才开；编程场景下降级为 PC.1 的对照组）**：历史负收益根因 =
  验证批在 A3 下放大专家拷贝 + drafter 占 worker 资源。P1/P2.1 就位后重新评估：K 草稿验证的每层
  专家并集 <6K（有重叠），且预取器可对整批草稿一次性预取并集 ⇒ IO 按 round 摊薄（SP-MoE/SpecExec 机制）。
  条件：单 token IO ≥3GB/s 且预测预取在线后，`NO_MTP=0 DRAFT=2/4` 重测；drafter 放 coordinator
  （mtp.md 方案 B 异步抢跑）作对照。**优先级排在 PC.1 之后**：复制式投机零内存零草稿成本，
  在编程负载上接受率更高；MTP 只在"PC.1 匹配不上的新代码段"有互补价值（两者可级联：先抄、抄不动再 MTP）。
- **P2.4 cache-aware 路由偏置 / 专家跳过（质量门把关，最后才碰）**：对 router 分数加微偏置
  λ·in_cache（λ 以 top1/topk 分数比门控），或跳过权重 < α·top1 的尾部专家（6→有效 4–5）。
  铁律：`ds4-eval q1..q4 --temp 0 --seed 1` + `ds4_test --logprob-vectors` 不过即回滚；默认关闭。

### §3.5 P-Code 编程场景专项（工作负载层，叠在 P1/P2 物理层之上）

> 依据：Claude Code 流量 ≠ 通用文本。三个可利用的结构性事实：
> ① **输出大量是上下文里已有的字节**（Edit 的 old_string/new_string 回显、Read 内容改写、
> import/标识符、重复 tool-call 骨架）；② **temp=0 贪婪为主**（DS4 投机的 greedy-only 限制天然满足，
> 工具语法段 server 已强制 temp=0）；③ **会话 append-only、回合制**（增量 prefill 才是真实 UX）。

- **PC.1 零成本复制式投机（编程场景的头号杠杆，可在 M1 后提前上）**：
  - 机制：drafter 不是模型，是 **n-gram 匹配器**——在当前 context+历史输出上找最长后缀匹配，
    直接“抄” K 个 token 当草稿（K 自适应=匹配延伸长度，cap 8–16），批量验证一次过 43 层。
  - 为什么这次投机能赢（区别于 MTP 的历史负收益）：
    (a) **零内存零草稿计算**——不占 worker 一字节，不抢 GPU；MTP 失败的"drafter 成本"项直接归零；
    (b) **验证批的专家并集天然去重**——`ds4_gpu_compact_selected_experts`（`ds4_metal.m:16986`）
    已对一次调用内 K×6 个 picks 去重；代码 token 连续段路由一致性高（2505.16056），
    并集 U(K) ≪ 6K，**IO 按 round 摊薄**正是 W2 墙下投机的正确打开方式（SP-MoE/SpecExec 机制）；
    (c) **协议零新建**——mtp.md Phase 1 已落地的 VERIFY 批验证 + `accept_len` + 跨机 KV 回滚
    （`ds4_distributed.c:67/125/288`）原样复用，只换 draft 源；净新代码 ≈ 一个 CPU 端 matcher。
  - 预期（文献+负载结构估算）：编辑/回显型回合**有效 decode 1.5–3×**，纯新代码回合 1.1–1.3×；
    与 P1/P2 的单前向提速**相乘**。CopySpec/SuffixDecoding 在 agentic 负载报告 2–3×。
  - 门槛：快赢 5 的离线回放显示 copy 可接受率 ≥30% 才立项；上线后非编程 prompt 不回退基线
    （匹配失败 = 不投机，自动退化为普通 decode，正确性无风险）。
- **PC.2 编程域热专家画像（把 P1.4 的池从“通用”变“懂代码”）**：
  - 离线在**真实 Claude Code 轨迹**（工具调用、diff、补全，非通用语料）上 profile 每层专家
    频率+共激活 → 生成静态画像文件（沿用 `EXPERT_POOL_PINNED` 的 `L:e,e,.../...` 格式，入库 `dir-steering/` 旁）；
  - 启动时按画像**顺序读预热**钉住（~3.5GiB 顺序读 ≈1–2s，非缺页随机读），
    现有 `PIN_FROM_LOGS`/auto-pin 机制升级为"按域画像加载"；运行期 auto-pin 只做增量修正。
  - 依据：MoE 专家领域特化，代码域路由偏斜显著高于通用文本 → P0.2 必须**分别**测
    通用 vs 编码流量的 SRP/SCH，画像命中上限差就是本项收益。可选实验：DSML 语法段 vs 代码段两套 hot-list。
- **PC.3 工具调用语法草稿（schema/replay 驱动，和 PC.1 共用验证通道）**：
  - 工具语法段强制 temp=0 且结构高度确定：DSML 结构 token、参数键名来自 **tool schema**，
    重复调用骨架在 server 的 **rax exact-replay map** 里逐字节存着 → 整段（几十 token）一次草稿批验证。
  - agent 循环（反复 Bash/Read/Edit）收益最大；落点 `ds4_server.c`/`ds4_agent.c` 草稿生成 + 既有 VERIFY 通道。
- **PC.4 回合级增量 prefill 工程（编程 UX 的真分母）**：
  - Claude Code 每回合重发全量会话，但前缀 KV 字节级命中已有（exact-DSML replay + 磁盘 KV）⇒
    **每回合实际只 prefill 新增的 user msg + tool result（1–8K）**。验收指标改为 per-turn TTFT，
    不是冷 200K prefill。
  - 要做：① turn 边界 checkpoint 策略核对（确保命中不被渲染差异打破，`--trace` 验证）；
    ② P1.2 层流式后把 chunk 上限提到 2048/4096 重测（旧 4096 OOM 是 scratch 池模型，层流式后需重标定）；
    ③ ds4-agent 内置工具路径在工具返回瞬间即时 prefill（不等下一轮请求）。
  - 算账：P1.2 后 4K 增量 ≈15–25s/回合；PC.1 叠加后短回复回合端到端 ~30–60s 量级——"勉强可用"的底线。
- **PC.5 编程负载回放基准（M4C 的验收尺）**：
  - 用真实 Claude Code transcript 构造 replay bench：per-turn TTFT、copy 草稿接受率、
    sustained 有效 decode、池整层命中率；脚本入 `tools/`，结果入 `notes/`。
  - PROMPT 单一短问句的现行 smoke 会**系统性低估** PC.1/PC.2 收益，必须加编辑型 prompt 档
    （含"修改这段代码"+ 大段上下文回显），作为 `mtp_pipe_q2_speed.sh` 的 `PROMPT_PROFILE=code-edit` 档。

### P3 200K / Claude Code 落地

- prefill 走 P1.2 流式 + `--kv-disk-dir` 前缀复用：Claude Code 会话重发长前缀，**增量 prefill** 才是日常路径；
  server 的 exact-DSML replay map 保证字节级前缀命中（已有）。
- 逐档 32K→100K→200K 验证两机 KV/scratch 预算（coordinator 日志显示 200K 下 context buffers 1.83GiB，
  在层切分 KV 收缩后可承受）；watchdog 红线 12/12GiB（2026-06-10 起）。
- 可用性诚实定义：M3 达成（单前向 decode ≥5 t/s）+ 前缀复用增量 prefill ≥30 t/s 时，Claude Code 短指令交互“勉强可用”；
  编程域有效 t/s 30（echo 重回合）是用户目标的可达上限；**单前向稳态 30 / 80 t/s 级流畅在本硬件+本模型组合下不存在**。
- 低优先备选（§2.1c）：raw 滑窗/indexer-selected KV 旋转量化（3bit K 族）再挤 ~0.5–1GiB/机给专家池；
  仅当 P1.4 池容量被证明是命中率瓶颈时才做；dequant 不得落 decode 热路径；
  门槛 `--logprob-vectors` + `ds4_test --long-context`，过不了即弃（IO-bound 下可能负收益，SwiftLM 对照表已证）。

### P4 决策门与回退

- **G1（P0 后）**：若 SRP/SCH 显示 6GiB 缓存命中上限 <30%，P1.4 降级为小钉住池，资源转 P2.1/P2.2。
- **G2（M1 后）**：若单拷贝直读 + prefill 流式仍 <1.5 t/s（IO 带宽没抬起来），优先查 APFS/加密/电源管理，
  再决定是否值得继续 P2。
- **G3（M3 后）**：若 expert-parallel 实测 <4 t/s，承认 W2 墙逼近，剩余精力转向：
  (a) 向用户摊牌换 `ds4flash-k16-v2.gguf`（13GiB，同 q2 量化族）双机全驻留路线（理论 10–18 t/s），
  (b) 或维持 q2 全量 + 接受 ~4–6 t/s 上限。

## 4. 内存预算（2026-06-10 两台 12G 新设定；第三十二波实测校正）

| | M4 mini ≤12GiB | MacBook ≤12GiB |
|---|---|---|
| backbone 层切片 mlock（20 层 / 23 层） | 4.07（预算 4608MB） | 4.14 实测 wired（预算 5120MB） |
| KV/压缩前沿 @200K（层切片收缩后） | ~1.8（ctx buffers 实测 1.83） | ~1.8 |
| A3/直读 scratch + IO 双缓冲 + stage buf | ~1.0 | ~0.8 |
| 热专家可用余量（页缓存/未来真缓存） | **~2.5–3.0** | **~3.0–3.5** |
| 其余（进程/Metal 开销） | ~1.5 | ~1.0 |
| 合计 | ~11 | ~10.5 |

注：第三十二波审计证实"source cache"（async 模式）从不 admit、只做 madvise 加热，
两侧 2048MB 配置不占匿名内存；表中"热专家余量"目前由页缓存隐式使用。
若未来引入真 DRAM 专家缓存（hard_copy 或新池），从该行预算扣。

护栏不变：`DS4_MEM_BUDGET_MB` + L1 resident gate 拒绝启动；脚本 RSS watchdog 两边同杀；
诊断用 `MEM_WATCH_MODE=footprint`（慢）。

## 5. 测试协议

- 统一入口：`tools/mtp_pipe_q2_speed.sh`（新开关全部走 env，保持默认=当前稳定基线，逐项 A/B）。
  增加 `PROMPT_PROFILE=code-edit` 档（编辑型长上下文 prompt）；现行短问句档继续作通用基线——
  两档都要跑，PC.1/PC.2 的收益只在 code-edit 档可见（见 PC.5）。
- 每轮记录：`prefill/generation t/s`、**有效 t/s（含投机接受）与接受率**、gather 带宽、池整层命中率、
  预测精度、per-turn TTFT（PC.4/PC.5）、两机 RSS 峰值、
  日志四件套（coord=/tmp/mtp_pipe_coord.log|.out，worker/build 同前）。
- 正确性门（每个 P 项合入前）：`--dump-logprobs` 对拍 A3 基线一致；`ds4_test --metal-kernels --server`；
  改路由/量化/重排的另加 `--logprob-vectors` + `ds4-eval q1..q4`。
- 速度回归：`ds4-bench` 在 2K/32K/100K/200K 前沿取瞬时 t/s，CSV 入库 `notes/`。

## 6. 快赢清单（本周内、改动极小）

1. `DIST_PREFILL_CAP=512/1024` A/B（零代码，直接测 prefill 是否线性抬升）。
2. `GATHER_THREADS=8` + gather 分解计时（P0.1 小补丁），确认 memcpy/fault/drain 占比；
   顺带做 P0.1 的屏障税 A/B（per-layer split/drain 关闭实验，零新代码，只动 env）。
3. `pread` 直读原型替换 `ds4_gpu_expert_gather_copy_slot` 的 mmap-memcpy（~50 行，先单机验证带宽）。
4. 用现有 expert-profile 日志离线算 SRP/SCH 与共激活矩阵（纯脚本，不碰推理）。
5. **PC.1 离线可行性回放（纯脚本，不碰推理）**：拿几段真实 Claude Code transcript（或 `~/.ds4/kvcache`
   会话），模拟 n-gram matcher：输出 token 里有多大比例能被"上下文最长后缀匹配"连抄 ≥4 个？
   ≥30% ⇒ PC.1 立项；同一批数据顺手出 PC.2 的编码域专家画像（结合 expert-profile 日志）。
6. **PC.4 前缀命中核对（零代码）**：`ds4-server --trace` 跑一个两回合 Claude Code 会话，
   确认第二回合 KV 前缀字节级命中、只 prefill 增量；不命中就先修渲染/replay 差异（这是白捡的倍数）。

## 7. 风险与诚实结论

- **80 t/s 违反带宽/容量物理；30 t/s 单前向也不可达（W3 backbone 墙 ≤10–18 t/s，§1）**，本方案不承诺单前向 30；
  承诺两条诚实关卡：① **单前向 decode 推向 W2/W3 物理极限区间 3–7 t/s**（当前 code-edit 3.77 已在区间内）；
  ② **编程域有效 t/s 冲 30（用户目标）**——靠 copy-spec/前缀复用/降激活越过单前向墙，仅 echo 重编辑回合可达、非跨回合稳态。
- 主要工程风险：`MTLIOCommandQueue` 与现有 no-copy mmap 视图/ResidencySet 的共存语义（退路 pread）；
  expert-parallel 的 per-layer 同步抖动（E0 把关）；预测精度不足导致预取空转（fallback 不伤正确性，只亏带宽）。
- **P-Code 收益是负载相关的，不改物理墙**：复制式投机只在"输出回显上下文"的回合放大有效 t/s
  （编辑/工具回合 1.5–3×，纯新代码 ~1.1–1.3×），匹配失败自动退化为基线，无下行风险但也不保证均匀提速；
  编程域画像的上限由 P0.2 实测 SRP/SCH 决定，先测后建。M4C 的"有效 30"只承诺在 code-edit 档（echo 重）prompt 上验收，匹配不上即回落单前向基线。
- 一切以 `notes/execution-log.md` 实测为准；correctness before speed，watchdog 红线 12/12GiB（2026-06-10 起）永不放松。

## 8. 进度

- [x] 联网调研（arXiv 2026-06 检索 + 开源项目盘点）+ 方案落地本文件
- [x] P0.1 gather 分解计时（`DS4_METAL_EXPERT_IO_PROFILE=1`，ds4-io 行：fault/memcpy/pread/drain/bw；
      含 drain_ms 屏障税专查）——**代码已落地 2026-06-10，待用户脚本实测取数**
- [ ] P0.2 SRP·SCH 标定
- [x] P0.3 SSD 底数（`tools/ssd_bench.c`，2026-06-10 实测）：**mini 全模式 ~2.4–2.6GB/s**（顺序≈随机），
      **MacBook 5.5–6.7GB/s（快 2.3–2.6×）**——coordinator 慢盘是当前结构性瓶颈；RTT 待测
- [x] 快赢 1（DIST_PREFILL_CAP A/B，第四十七波）：实测 128→11.75 / **2048→12.89(+9.7% prefill)** / 4096→11.82(退化)；**2048 最优**(4096 巨块密集 gather+跨机延迟抵消)，脚本默认抬到 2048，bit-exact+峰值 6.68G 安全。受 W2 磁盘墙限故 +10% 非 3×；是 TTFT 白拿增益 / 4（SRP/SCH 脚本）——待实测
- [x] 快赢 2 → 屏障税回收：A3 每层 drain 改 MTLSharedEvent 快路径 host wait
      （`DS4_METAL_EXPERT_EVENT_DRAIN=1`，CB 状态检查推迟一拍；正确性保证等价）
      ——**代码已落地 2026-06-10，drain_ms 新值待实测（目标 <2.5ms/层）**
- [x] 快赢 3 → P1.1 pread 单拷贝直读（`DS4_METAL_EXPERT_PREAD=1` + `_NOCACHE` A/B 旋钮；
      MTLIOCommandQueue 留作下一步若 pread 未达 ≥3GB/s）——**代码已落地，M1 门 decode ≥1.6 待实测**
- [x] P1.2 prefill 全层流式（`DS4_METAL_EXPERT_FULL_LAYER_STREAM=1`，阈值 60%、16MiB 分块、
      免 remap 直用原 id；含槽位排序快赢 `DS4_METAL_EXPERT_SORT_IDS=1`）——**代码已落地，M1 门 prefill ≥10 待实测**
- [x] ~~P1.3 GGUF 专家捆绑重排 sidecar~~ **不立项（G 门否决，2026-06-10）**：P0.3 实测 mini 盘
      随机 2MiB@QD8=2.40 vs 捆绑粒度 6.75MiB@QD8=2.61 vs 顺序=2.50 GB/s——增益 ≤8%，
      不值 34GiB sidecar + repacker；worker 盘虽有 +18% 但不是瓶颈侧
- [x] ~~P1.4 常驻池间接表（命中零拷贝）/ X9 驻留~~ **决定性否决（第五十一波, 数据充分）**：全层 top-16 pool
      容量装得下（coord 2.16+worker 2.43 GiB, 用户对 24G 有余量）；HIT_ONLY=0 成功填满（命中 90→2760）排除
      预热/容量问题；但 **decode 反而更慢**（pool 命中 2760 时 replay turn2 2.22 << 无 pool 4.00）⇒ **resident-pool
      执行路径本身比 compact-scratch streaming 慢**（大缓冲散点索引/GPU 工作集/admit 开销）。所有缓存变体
      (hard_copy/mlock/pool) 全使 decode 变慢。W1 流式+overlap+copy-spec 是本硬件最优。集中度真实但常驻它更慢
- [x] P2.1 跨层预测预取 v2（`DS4_METAL_EXPERT_PREFETCH_AHEAD=1` + `_TOP/_DELTA`；
      v1 实测回退 1.56→1.49：pf 精度 77.5% 可用，但 readahead 与前台 gather 抢 SSD——
      decode 期间 SSD 无空闲带宽，预取不能加总流量；v2 改礼让式：1MiB 分块只流入
      drain/GPU 空闲窗 + mincore 跳过已缓存 + 分数序 + latest-wins 过期中止）
      ——**v2 已落地 2026-06-10，待实测；M2 门 decode ≥3**；
      gather 工作单元同步细化到张量级（decode QD 6→18，脚本 GATHER_THREADS 默认 8，A/B 用 4）
- [x] P2.2 低成本变体：远程专家字节服务（`DS4_DIST_EXPERT_FETCH_*`）——实测 1.81 t/s/prefill 4.80；
      层内抢单对 decode 小工作集失配（18 单元被本地线程秒抢，远程 ~2ms 往返只配上层尾），
      v4 改 **预测驱动远程暂存**（`DS4_METAL_EXPERT_STAGE=1`：L 层窗口内把 L+1 预测专家拉进
      RAM 双槽，命中即 memcpy；专家权重不可变 ⇒ 暂存永不过期）；racing 仅保留给 prefill ≥96 单元
      ——**staging 已落地 2026-06-10 待实测，方向 2.3–2.5 t/s**
- [ ] P2.2 完整 expert-parallel 拓扑（M3 门：decode ≥5；先看低成本变体实测再定）
- [ ] P2.3 MTP 重启评估 / P2.4 路由偏置（质量门）
- [x] PC.1 复制式投机（`DS4_DIST_COPY_SPEC=1` + `_DRAFT/_NGRAM`；n-gram 匹配器当 drafter，
      VERIFY/accept_len/KV 回滚协议原样复用，miss 零开销；单 token 地板 ~2.1 后的 M2 主路线）
      ——**代码已落地 2026-06-10（第十九波），待用户脚本实测；M4C 门：echo 重编辑回合有效 decode ≥30**
- [ ] PC.2 编程域热专家画像 + 启动预热（第四十四/四十六波定论: 驻留缓存打不过 page cache, 热集装不下, 暂搁置）
- [ ] PC.3 工具语法草稿（schema/replay 驱动）
- [x] PC.4（部分，第四十八波）per-turn TTFT 指标上线（`run_chat_turn` 打印 cached/suffix/TTFT）+ **dist REPL
      路由 bug 修复**（run_repl 漏等 worker 路由→"missing layer 20"，修复后 dist 多回合可跑）。实测增量 prefill
      生效（cached 0→190→345 增长, suffix 只填新增 27/23）；**但小 suffix 走散点 gather 仍撞 W2 墙**(turn3 23tok/19.7s)。
      遗留: 散点→稠密流式 / 热专家复用 / agent 工具返回即时 prefill(PC.4③)
- [x] PC.5（部分）`PROMPT_PROFILE=code-edit`（第二十一波）+ **`PROMPT_PROFILE=replay` 多回合 bench（第四十八波，
      REPL 模式量化 per-turn TTFT）**；完整真实 transcript replay 仍待建
- [ ] P3 200K 逐档 + Claude Code 前缀复用验证
- [ ] G1–G3 决策记录
