# Deep-Research 全记录 + 落地设计方案（react/go 逼近 Opus 4.6）

> 本文件落地两部分：① 之前 deep-research 多 agent 调研的**全部细节**（已验证发现、证据、来源、被驳倒清单、注意事项、开放问题、统计）；② 基于调研演进出的**最终落地设计方案**。
> 调研规模：6 路并行搜索 → 28 来源 → 132 论断 → 25 条 3 票对抗验证 → **19 确认 / 6 驳倒** → 8 条高置信结论。日期：2026-06-19/20。
> 详细工程命令/参数见同目录 `react-go-opus46-design.md`；本文件是研究证据 + 方案总览的单一事实源。

---

# 第一部分 · Deep-Research 全部细节

## 1. 研究问题

基于 DeepSeek-V4-Flash-Base + antirez/ds4 非对称量化理论 + 论文 arxiv 2510.13999，把模型量化到**双机 16GB Mac**（Thunderbolt 直连分布式层切分、每机 ≤12GiB、专家 SSD 流式）上装下并运行；再通过**本地机器微调训练**（Mac 上 MLX/LoRA/QLoRA），让量化模型在 **react 和 go 编码场景**逼近 **Claude Opus 4.6**。三条主线：A 量化方案本身；B 本地微调；三类语料（① 代码仓库+核心库+bug 修复 ② 架构/算法/重构经典书 ③ 工程提示词+执行诚实）。

## 2. 核心结论摘要

先压缩到能装下、再本地 LoRA 补偿、再用语料和 prompt 工程把编码能力顶回的三段式工程。压缩侧两条互补且可复现的前沿主线：① arxiv 2510.13999 的 **REAP** 一次性专家剪枝（路由门控值 × 专家激活范数，Apache-2.0 官方码 + 已发布 DeepSeek-V3.2 等预剪枝权重，剪 50% 专家代码生成近无损）；② ds4 已落地的**非对称量化**（路由 gate/up→IQ2_XXS、down→Q2_K、attention/shared/output→Q8_0），理论依据是 MoE 专家层比稠密 FFN 对低比特远更鲁棒（MoQE 2310.02410），由 llama.cpp 原生 quantize + imatrix 校准直接支撑；REAP 先剪后量可叠加。微调侧核心约束：**LoRA 必须挂 MLP/MoE 专家层**（mlx-lm 当前对 MoE 专家投影有 bug 只训 0.022% 参数）；**2-bit 基座上标准 QLoRA 零初始化适配器在 3-bit 以下不收敛，必须改用 LoftQ 或 ApiQ**。语料三角度是把 REAP+2-bit 损失顶回 react/go 编码能力的补偿层。

## 3. 已验证发现（8 条，3 票对抗验证，full evidence）

### F1 — REAP 一次性 MoE 专家剪枝 [✓ high, 3-0]
- **论断**：REAP（arxiv 2510.13999）以路由门控值 × 专家平均激活范数为显著性准则，剪 50% 专家后代码生成近无损，优于专家合并。可在量化前把专家数砍半。
- **来源**：https://arxiv.org/abs/2510.13999 · https://github.com/CerebrasResearch/reap
- **证据**：标题 *REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression*（Lasby 等 ICLR 2026）。摘要：considers both router gate-values and expert activation norms to minimize the reconstruction error bound；achieves near-lossless compression on code generation tasks with Qwen3-Coder-480B and Kimi-K2 even after pruning 50% of experts。Table 3：Qwen3-Coder-480B 非 agentic 编码 0.660→0.644（-1.4%），SWE-Bench-Verified 0.540→0.522；REAP 0.644 vs 频率剪枝 0.011。
- **caveat**：工具调用框定偏乐观（Kimi-K2 BFCL 0.666→0.564 约 -15% 相对）；厂商自报，无独立复现。

### F2 — REAP 官方实现可直接落地 [✓ high, 3-0]
- **论断**：Apache-2.0，Python + HF Transformers + vLLM 评测，`prune.py` 入口 + 逐层 block-wise 校准（单 GPU 剪超大模型），已发布 DeepSeek-V3.2（345B/508B）、Qwen3-Coder-480B/30B、GLM4.6、Kimi-Linear 等预剪枝 checkpoint。
- **来源**：https://github.com/CerebrasResearch/reap · https://huggingface.co/cerebras/DeepSeek-V3.2-REAP-345B-A37B
- **证据**：repo License Apache-2.0；93.9% Python；`prune.py` = Main entry point for expert pruning；memory-efficient layer-wise block-wise calibration（experiments/pruning-layerwise-cli.sh）。
- **风险**：发布的是 V3.2 不是 V4-Flash → V4 架构差异需适配剪枝脚本，剪后须重收 imatrix + 重量化。

### F3 — ds4 非对称量化 + MoE 低比特鲁棒性 [✓ high, 3-0]
- **论断**：ds4 非对称量化（gate/up→IQ2_XXS、routed down→Q2_K、shared/output/attention→Q8_0）经本地 repo 确认；理论依据 = MoE 专家层比稠密 FFN 对低比特远更鲁棒。
- **来源**：https://huggingface.co/0xSero/DeepSeek-V4-Flash-162B-GGUF · https://arxiv.org/abs/2310.02410 · 本地 README.md
- **证据**：README.md 行 123-125：only the routed MoE experts are quantized, up/gate at IQ2_XXS, down at Q2_K。MoQE（2310.02410 Microsoft）：expert layers in MoE models are much more robust to the quantization than conventional FFN layers。
- **caveat**：比较性结论；salient/Super 专家极敏感（Qwen3-30B 剪 3/6144 即坍缩，Mixtral 均匀 INT2 掉 70→34%），须靠 imatrix 保护；MoQE 中"2-bit 仅专家免训练即可靠"的更强主张被 0-3 否决（见被驳倒清单）。

### F4 — llama.cpp 量化工具链 + imatrix [✓ high, 3-0]
- **论断**：`quantize` 原生支持全部 2-bit 类型（IQ2_XXS/XS/S/M、Q2_K_S/Q2_K），imatrix + include/exclude-weights 逐张量控制；IQ2_XXS=2.38bpw，Q2_K=3.16bpw（Llama-3.1-8B）。
- **来源**：https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
- **证据**：QUANT_OPTIONS 表含 IQ2_XXS 2.06bpw…Q2_K_S；`--imatrix` use data as importance matrix；`--include/exclude-weights`（互斥）。Llama-3.1-8B：IQ2_XXS 2.3824bpw/2.23GiB，Q2_K 3.1593bpw/2.95GiB。
- **落地**：ds4 已有 `gguf-tools/imatrix` 流程；REAP 剪后须重收 imatrix 再量化。

### F5 — LoRA 必挂 MLP/MoE 专家层 [✓ high, 3-0]
- **论断**：LoRA 必须挂 MLP/MoE 专家层，仅 attention 即使提秩参数量相当也显著欠拟合；mlx-lm 对 MoE 专家投影 LoRA 转换有已知 bug 只训 ~0.022% 参数（Qwen3-30B-A3B 6.685M / 30.532B）。
- **来源**：https://thinkingmachines.ai/blog/lora/ · https://github.com/ml-explore/mlx-lm/issues/571 · https://arxiv.org/abs/2405.09673
- **证据**：Thinking Machines：Attention-only LoRA significantly underperforms MLP-only LoRA；attention rank256 < MLP rank128。Biderman TMLR 2405.09673：especially MLP modules is crucial。mlx-lm #571：Trainable 0.022% 6.685M/30532.123M，根因 linear_to_lora_layers fails to detect and convert the expert linear projections。
- **落地**：ds4 DeepSeek-V4 MoE 适配器必须命中 `mlp.experts` 的 gate/up/down，否则训练≈无效。

### F6 — 2-bit 基座 QLoRA 不收敛 [✓ high, 3-0]
- **论断**：标准 QLoRA 零初始化适配器在 3-bit 以下不收敛，正落 ds4 IQ2_XXS/Q2_K 的 2-bit 区间。不能直接 QLoRA on 2-bit。
- **来源**：https://arxiv.org/html/2310.08659 · https://arxiv.org/pdf/2402.05147
- **证据**：LoftQ 2310.08659：QLoRA fails below the 3-bit level，2-bit 上 CoLA/WikiText-2/GSM8K 报 N.A./does not converge。ApiQ 2402.05147：at the challenging 2-bit precision where QLoRA fails to converge。LowRA 2502.08141 再确认。

### F7 — 解法 ApiQ / LoftQ [✓ high, 3-0]
- **论断**：用 LoftQ 或 ApiQ 替代 QLoRA。LoftQ 交替优化 `min‖W−Q−AB‖_F`（量化残差 SVD 初始化 A/B）；**ApiQ 并发初始化 LoRA 分量+量化、保激活精度、抑误差从浅向深传播，专为 2-bit，持续优于 QLoRA/LoftQ**。
- **来源**：https://arxiv.org/html/2310.08659 · https://arxiv.org/pdf/2402.05147 · https://github.com/baohaoliao/apiq
- **证据**：LoftQ ICLR2024 Eq.6/Algorithm 1。ApiQ EMNLP2024：concurrently initializing the LoRA components and quantizing the weights；maintenance of the original LLM activation precision while mitigating the error propagation from shallower into deeper layers；consistently superior across various bit-widths。
- **caveat**：ApiQ 适配器存 FP16 抬高平均位宽（对比纯 PTQ less fair）；partial updates ≠ 完整 QAT（2504.13932）。均 PyTorch 实现，需离线产出再转 GGUF。

### F8 — EAQuant MoE 专属 PTQ [✓ high, 3-0]
- **论断**：EAQuant（arxiv 2506.13329）针对 MoE 激活离群/路由不稳/稀疏专家校准三失效模式，提出平滑聚合、路由一致性对齐、校准均衡，在 W2A4/W3A3/W3A4 等亚 3-bit 验证可行。
- **来源**：https://arxiv.org/pdf/2506.13329
- **证据**：smoothing aggregation to suppress activation outliers；routing consistency alignment to preserve expert selection post-quantization；calibration data balance。W2A4=2-bit 权重/4-bit 激活。
- **caveat**：W2A4 具体精度数值未独立读到；与 2510.13999 无关，是独立补充（W2A4 设置项 2-1 split）。

## 4. 被驳倒清单（禁用，勿作依据）

| 论断 | 投票 | 来源 |
|---|---|---|
| MoQE "2-bit 纯专家**免训练即可靠**，直接验证 ds4 非对称设计" | **0-3** | arxiv 2310.02410 |
| QuIP# 2-bit Llama2-70B WikiText2 ppl 4.16 / C4 5.71 可用质量 | **0-3** | PMC12395268 |
| QuIP# 两阶段微调"必需且全位宽增益" | **1-2** | PMC12395268 |
| "LoRA 在 SFT 等同全量微调"（覆盖本项目 react/go SFT 规模） | **0-3** | thinkingmachines.ai/blog/lora |
| "MoE 每专家单独 LoRA、rank=总rank/激活专家数" | **0-3** | thinkingmachines.ai/blog/lora |
| "mlx-lm 完全没给专家挂 LoRA"（仅 0.022% 事实被确认） | **1-2** | mlx-lm #571 |

## 5. 注意事项（caveats）

1. REAP 50% 近无损为 Cerebras 厂商自报无独立复现；工具调用近无损偏乐观（Kimi-K2 BFCL 掉 ~15%），对依赖工具调用的 agent 编码需自测。
2. REAP 发布的是 DeepSeek-V3.2 而非 V4-Flash 预剪枝权重，V4 架构差异需自行适配脚本 + 重收 imatrix + 重量化，工程量不小。
3. MoE 专家比稠密 FFN 鲁棒是**比较性结论**（2023），不可读作对所有专家均匀 2-bit 安全；salient/Super 专家极敏感须 imatrix 或 mixed-precision 保护。
4. LoftQ/ApiQ/EAQuant/REAP 全是 PyTorch 生态；ds4 是纯 C99+Metal+mmap GGUF，所有方法要么离线在 PyTorch 侧产出权重/适配器再转 GGUF，要么在 ds4 内重实现，**集成成本是主要风险**。
5. mlx-lm 对 MoE 专家挂 LoRA 只有 0.022% 可训练参数（事实确认）；Mac 16GB 上对剪枝后 V4 做 MoE-LoRA 的实际显存/时长无直接数据。
6. **语料三角度 → 逼近 Opus 编码 的因果链未被任何验证 claim 量化，是设计假设非已证结论。**
7. QuIP 2-bit 困惑度、两阶段微调相关 claim 被否决，不应作核心依据。

## 6. 开放问题（无数据，需原型/实测）

1. REAP 剪 50% 再走 ds4 IQ2_XXS/Q2_K，两者损失是否叠加放大？需在 V4-Flash 上实测剪+量联合的 react/go 编码掉点。
2. 16GB Mac 上用 mlx-lm/LoftQ/ApiQ 对剪枝后 V4 做 MoE-LoRA 的真实峰值显存、梯度检查点开销、单 epoch 时长？双机分布式微调是否可行？无数据。
3. 在 2-bit GGUF 运行时基座上挂可训练 LoRA 并保持 mmap+Metal 推理路径，适配器以何格式注入、数值稳定如何？需原型验证。
4. 三类语料各需多大规模/配比，才能把 REAP+2-bit 损失顶回接近 Opus？缺乏针对该管线的消融或先例。

## 7. 全部来源（28 fetched，按角度）

**论文与非对称量化理论核心**：github.com/CerebrasResearch/reap · huggingface.co/0xSero/DeepSeek-V4-Flash-162B-GGUF
**≤2-bit MoE 量化工具链**：arxiv 2510.13999 · arxiv 2310.02410 · arxiv 2506.13329(EAQuant) · arxiv 2504.02658 · pmc.ncbi.nlm.nih.gov/articles/PMC12395268(QuIP#) · github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
**Apple Silicon 16GB LoRA/QLoRA**：thinkingmachines.ai/blog/lora · github.com/ml-explore/mlx-lm/issues/571 · arxiv 2402.05147(ApiQ) · arxiv 2310.08659(LoftQ) · github.com/ARahim3/mlx-tune
**react/go SFT 语料与 bug 修复数据集**：huggingface.co/datasets/SWE-bench · github.com/multi-swe-bench/multi-swe-bench · arxiv 2308.07124(OctoPack/CommitPackFT) · huggingface.co/blog/dedup · arxiv 2504.14757 · github.com/SWE-Gym/SWE-Gym
**强模型蒸馏与书籍知识→指令样本**：arxiv 2402.13064 · arxiv 2312.02120 · developer.nvidia.com(license-compliant synthetic data) · arxiv 2405.03548
**小模型逼近强模型 agent/prompt 与忠实推理**：arxiv 2508.00083 · arxiv 2303.11366(Reflexion) · arxiv 2503.08679 · arxiv 2307.13702(faithful reasoning) · arxiv 2511.14102

## 8. 统计

angles 6 · sourcesFetched 28 · claimsExtracted 132 · claimsVerified 25 · confirmed 19 · killed 6 · afterSynthesis 8 · urlDupes 1 · budgetDropped 7 · agentCalls 111 · subagent_tokens ~2.87M · duration ~37min。

---

# 第二部分 · 落地设计方案（演进后的最终方案）

> 详细命令/参数/代码改动点见 `react-go-opus46-design.md`。本节是方案总览与决策记录。

## 9. 总体：三段式补偿管线

P0 压缩（为装下必然掉点）→ P1 本地微调补偿（顶回一部分）→ P2/P3 语料+推理期工程（顶回剩余）。每段挂可测的 react/go eval 门，禁止"假定生效"。目标基准 = **Claude Opus 4.6 的 react/go 编码能力**（用户拍板，2026-06-20）。

## 10. 模型结构（实测）

DeepSeek-V4-Flash = **43 MoE 层 × 256 路由专家 = 11,008 专家**（`make_firstk_mask.py --layers 43 --n-expert 256`）；前 **3 层 hash 路由**（`DS4_N_HASH_LAYER=3`）。2-bit 下 ~6.75 MiB/专家；路由专家共 72.6 GiB，backbone ~8.4 GiB。原始 HF base ~240GB（用户已在 hf/ 目录下载，未完成）。

## 11. 用户拍板的方案演进（4 个增量，均已写入设计文档）

### 11.1 精度倒置（§3.6）
从 240GB base 出发，**只把 react/go 画像出的热专家（~32/层）量化到 2-bit 使其常驻 20GB**；其余冷专家**原样保留**、SSD 冷启动流式；再对 2-bit 热专家做领域训练顶到 react/go 4.6。
- 反常识却成立：热专家压 2-bit 换常驻（=30t/s），质量损失靠领域训练买回。三重支撑：常驻消 SSD IO（30t/s）、MoE 低比特鲁棒（F3）、ApiQ 2-bit 可训回质量（F7）。红利：不重量化 240GB，通用能力全精度无损。
- **训练顺序（踩错即废）**：量化热专家→2-bit → **ApiQ/LoftQ 初始化**（否则不收敛 F6）→ 领域 QAT/LoRA（三点语料）→ 4.6。
- **4.5→4.6 诚实判定 [假设]**：净值 = 4.5 − 2bit损失 + 领域训练增益；窄域专化可超通用（方向有论文支撑），有可能到 4.6 但非保证，eval 闭环唯一裁判。

### 11.2 动态模式切换路由（§3.7）
假设编程/日常不并行（ds4 server 单图 worker 串行，天然成立）：
- **Mode P（编程，默认）**：router **屏蔽到编程热集** → 零 cold miss → 30 t/s。
- **Mode G（日常）**：解除屏蔽，全 256/层，冷专家 SSD 流式 → 全质量、次要慢档。
- 切换付一次性 ~2-3s 预热（精确 madvise WILLNEED 仅热档区，不是整盘）。
- **它解决了精度倒置的两个雷**：雷2 编程 cold miss（屏蔽路由→物理不可能 miss）；雷3 冷专家精度纠结（日常=刻意慢档→冷专家保 FP16 原样不动反而合理）。
- **硬前置 = keep-map 屏蔽内核 bug 修复**（见 §13）。

### 11.3 混合格式省盘（§3.8）
冷档不转 GGUF，直读已下载的 HF safetensors，省 ~210GB（470G→260G）。每机 ~10G 量化常驻 = backbone 切片(Q8 ~4.2G) + 编程热专家切片(IQ2 ~4.5G)；冷专家从本机 HF safetensors(FP16) 流式。**最大新代码 = ds4 运行时加 safetensors 冷档流式读**（现只读 GGUF；`deepseek4-quantize` 已离线解析 safetensors 可参考）。前置：HF 下完 + 验证 per-expert 可寻址。

### 11.4 常驻机制（遵循实测，不 mlock）
**不 mlock/不显式 pin**——memory 实测显式 RAM 专家缓存饿死 OS page cache 反变慢（3.84→1.84）。靠：① 热档工作集 ≤~9GiB 使 page cache 可容纳；② 启动顺序预读仅热档区预热；③ 热档 GGUF 内连续布局。常驻由 page-cache LRU + 现有流式路径涌现。

## 12. 20GB 常驻预算

backbone Q8 ~8.4 + KV ~1.5 + working ~1.0 + react/go 热专家 ~9.1 GiB（→ ~32 专家/层）。30 t/s 物理前提：全模型每 token 专家 IO ~1.70 GiB → SSD ~3 t/s 天花板；常驻热集覆盖 ~98% react/go 路由 → 每 token 冷 miss ~2%×1.70≈34MiB(~7ms) → 塞进 33ms 预算 → 30 t/s 可行（取决实测 react/go 路由集中度）。

## 13. keep-map 屏蔽内核 bug —— 已修复（2026-06-20）

**地基**：Mode P 屏蔽路由、残驻、30 t/s 全压在 keep-map 屏蔽上；memory「k16 GGUF stale/broken」记其为悬而未决 bug。现有资产（`gguf/ds4flash.gguf` + `gguf/mask-k16.bin`）单机可复现验证。

**根因**：屏蔽用 `slot = ds4_gpu_router_cache_layer_slot(bias_offset)`（bias_offset 首见顺序号）索引 keep_lut；前 3 层 hash 路由跳过登记 → 非 hash 层 `slot = il − 3`；而 `ds4_gpu_translate_expert_ids` 用真实层号 `il`。**屏蔽屏了错层、gather 取对层 → 不一致 → 数字汤**。这就是"屏蔽（看似实现但）实际层号错位"。

**修法（已编译通过）**：keep_lut 屏蔽改用真实层号 `il` 索引（`slot` 留给自洽的 cache-hot/REAP 累加器）。
- `ds4_gpu.h` + `ds4_metal.m`：`router_select_tensor` / `router_select_batch_tensor` 签名加 `uint32_t layer`。
- `ds4_metal.m`：decode(14594) + batch(14720) 的 `keeplut = lut + layer*256u`（原 `slot*256u`）。
- `ds4.c`：decode(11113) + prefill(14109) 两个调用点传 `il`。
- `ds4_cuda.cu`：两签名加 `layer`（`(void)layer`，CUDA 无 shrunken 路径，保 Linux 构建）。

**验证状态**：`make` 绿（5 binary 全建成）。单机 k16 跑（载模型）待用户授权。
**残留constraint**：前 3 hash 层的 hash 目标专家必须被 keep-map 保留（hash 路由绕过 score 屏蔽），是 shrink/mask 工具的约束，非内核问题。
**correctness 门**（合并前）：`--dump-logprobs` parity + `ds4_test --metal-kernels --server`；routing 改动加 `--logprob-vectors` + `ds4-eval q1..q4 --temp 0 --seed 1`。

## 14. 端到端里程碑（M0–M6，详见设计文档 §9）

M0 eval 基线(Opus 4.6 参考分) → M1 REAP 剪枝拐点 → M2 双机装载 → M3 三类语料 → M4 ApiQ 微调 → M5 prompt 体系+消融 → M6 对标 Opus 4.6 诚实记录差距。

## 15. 三类语料（详见设计文档 §6）

① bug 修复 diff（Multi-SWE-bench 含 Go、SWE-Gym、OctoPack）50% · ② 经典书蒸馏（Opus 生成 react/go 架构重构教学，书做种子不复制正文）30% · ③ 执行诚实 agent（faithful CoT + Reflexion，工具结果真实回填）20%。合计 3万~8万条。**配比/规模为假设，M5 消融实测。**

## 16. prompt/agent 诚实体系（接 ds4，详见设计文档 §7）

faithful CoT(2307.13702/2511.14102) + Reflexion(2303.11366) + plan-then-execute → 接 `ds4_agent.c`；exact-DSML replay(`rax.c`) 禁编造工具历史 → `ds4_server.c`；copy-speculation(`DS4_DIST_COPY_SPEC`) 编程域 prefix 复用提有效 t/s。
