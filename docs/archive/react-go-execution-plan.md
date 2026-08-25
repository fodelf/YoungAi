# react/go 精度倒置量化方案 —— 执行层计划与任务拆分（审计纠正后）

> 版本：v1 执行稿（2026-06-20，wave-82 后）。
> 定位：本文件是**执行层**文档，坐在两份设计文档之上，不重抄研究证据与设计推演。
> 上游设计（只引用，不复制）：
> - `react-go-opus46-design.md` —— 量化/裁剪主线（§3.5/3.6/3.7/3.8 + M0–M6）。
> - `react-go-training-design.md` —— 训练专册（T0–T5）。
> 权威状态来源：`notes/execution-log.md` 的 **wave-81 / wave-82** 段（含本会话对抗审计纠错 + B1/B2a 落地 + B1 真实数据验证通过的全部客观事实）。
> 本文件每条状态都能在代码或上述日志找到依据；拿不准的标 `[假设]` 或「未验」。

---

## 0. 证据档与状态图例（贯穿全文）

**证据档**（沿用设计文档铁律）：
- **[✓证]** = 代码 / 实测确认（本文件大量是「我直接读 header/源码/git diff 复核」）。
- **[假设]** = 设计推断，无验证 claim 直接支撑，必须用 eval 闭环证伪。
- **[✗驳]** = 被对抗验证驳倒，禁止作为依据。

**任务状态**：
- ✅ **完成** —— 代码落地 + 验证可在源码 / 日志找到依据。
- ⚠️ **未提交未验** —— working tree 有改动但未 commit、未跑数值验证。
- 🔲 **待做** —— 一行未写。
- 🔒 **gated** —— 代码就绪但需内存安全证明 + 逐次授权才能跑（载模型）。
- ⏸ **推迟** —— 本期不做，独立后续阶段。

---

## 1. 文档定位与与两份设计文档的关系

| 文档 | 层级 | 内容 | 本文件如何用它 |
|---|---|---|---|
| `react-go-opus46-design.md` | 设计层（量化/裁剪主线） | 研究综述、精度倒置 §3.6、动态模式 §3.7、混合 HF 冷档 §3.8、M0–M6 | 引用其节号；**用审计事实修正其错误前提**；**重排其里程碑** |
| `react-go-training-design.md` | 设计层（训练专册） | 初始化/LoRA/specialty/三类语料/T0–T5 | 引用其 T 编号；**修正 LoRA target_modules 命名**（w1/w2/w3） |
| **本文件** | **执行层** | 当前状态快照、关键路径决断、可执行任务拆分总表 | 是往下推进的**操作手册** |

> 设计层讲「为什么这样做」（带研究证据）；本执行层讲「现在做到哪、接下来谁去做哪一条、过哪个门、要不要授权」。研究证据已在设计文档与 `search.md`，**本文件不重做联网研究、不重抄引用**。

---

## 2. 当前状态快照

| 子系统 | 状态 | 证据 |
|---|---|---|
| **B1 量化器 FP4→FP8 专家路径**（`deepseek4-quantize.c`） | ✅ 完成 + 真实数据验证 | `dequant_fp8_weight` 接 F32\|F8_E8M0（:700-701）；`generate_one_expert` 按 `w.dtype` 分流（:1406/:1415）；`load_f32_le`（:186）；`make` 绿 0 警告；`/tmp/validate_b1_fp8.py` 真实 layers.0 w1/w2/w3 **3/3 OK**（wave-82 补） |
| **B2a hash 层 mask 护栏**（`make_expert_mask.py`） | ✅ 完成 | `force_keep_hash_layers()`（:66）+ `--hash-layers` 默认 3（:116）；驱动 `quantize_reactgo.sh` step-3 显式传；自测 OK（wave-82 ③） |
| **HF index.json + 分片下载** | ✅ 完成（部分分片） | `hf/DeepSeek-V4-Flash-Base/model.safetensors.index.json`（5.3MB 在盘）；25/46 分片，layers 0-5 专家齐（wave-82 补） |
| **`router_norms_from_imatrix.py`**（energy + specialty） | ✅ 完成（self-test） | wave-81 落地 1 + wave-81 补；`--baseline`/`--rank-by {energy,specialty}`；self-test 绿。**未验**=真实 react/go imatrix 跑（前置=步骤 1 载模型） |
| **`deepseek4-quantize --experts-hot-mask`** | ✅ 完成（dry-run） | wave-81 落地 2；DSXM 加载 + keep_map KV；dry-run 字节精确（256→32）。**未验**=真实 HF→hot-only GGUF 端到端（gated，需模板 GGUF 在 M1） |
| **`quantize_reactgo.sh` 驱动** | ✅ 完成 | wave-81 落地 3；离线步默认跑、载模型步 `RUN_MODEL_STEPS=1` 门控 |
| **router-side -inf 屏蔽内核**（`ds4_metal.m`） | ⚠️ **未提交未验** | `git diff ds4_metal.m` = +416 行，`keeplut[i]<0 → cbp[i]=-1e30f`（:281/:338），门控 `has_bias && !hash_mode`（:248）；`ds4.c` +4 行；**未 commit、未跑 logprob parity** |
| **B2b route_translate fail-loud 计数器** | ⚠️ 代码落地·M4 验证·gated 跑未验 | wave-83：`metal/dsv4_misc.metal` 内核加 `verify`+`atomic_uint *clamp_count`（与 `moe.metal:271` REAP 逐字一致）、`ds4_metal.m` +3 辅助（dump/verify-gate/ensure-buf，镜像 REAP）+ dispatch 绑 idx3；公共签名不变（ds4.c/CUDA 零改）；默认 off byte-identical；`make` 绿 + Metal probe `newLibraryWithSource`+pipeline 创建 OK。**未验**=运行时 Mode P clamp=0（M-1.3 k16，M1 gated） |
| **单机 k16 logprob-vectors parity** | 🔒 gated | memory 硬闸；需 q2 或 shrunken GGUF（**在 M1**）+ 授权 |
| **react/go 执行型 eval harness** | 🔲 待做（前置） | wave-82 ④：现有 `ds4-eval` 是 MMLU/行号抽取 grader（`--self-test-extractors`，ds4_eval.c:1532/:3204），**跑不了** test-pass / Opus-judge / 工具幻觉率 |
| **§3.8 混合 HF 冷档流式（Mode G）** | ⏸ 推迟（独立后续阶段） | 见 §4：最大未建块，不在 react/go 关键路径 |

> **一句话现状**：量化工具链（B1/B2a/router_norms/hot-mask/驱动）已落地且 B1 在真实 Base 数据上验证通过；**唯一已写但未 commit/未验的是 router-side 屏蔽内核**；eval harness 与 §3.8 整块仍未动。

---

## 3. 审计纠正的错误前提（本会话产出，权威以此为准）

> 下列每条由 wave-82 ①「我直接读 header/源码/git diff 复核」确立（非仅信 agent）。**它们修正设计文档里基于错前提的章节。**

### 3.1 冷专家真实格式 = F8_E4M3 (1 byte) + F32 block-scale，不是 FP16 [✓证]

- **实测**：读 `model-00003-of-00046.safetensors` header —— `layers.0.ffn.experts.0.w1.weight` = **F8_E4M3** [2048,4096]，`.scale` = **F32** [16,32]（=128×128 block-scale）。分片直方图：776 F8_E4M3 专家权重 + 783 F32（含专家 scale）。
- `config.json` 写 `scale_fmt=ue8m0`，**与磁盘真实 F32 scale 矛盾，磁盘为准**。
- **修正**：`react-go-opus46-design.md` §3.8.1（省盘账 470G→260G 的「冷档 FP16 ×8」）、§3.6.5 **RB**（冷启动字节 ×8 → 通用 0.3-0.4 t/s）、§3.7 Mode G 速度数 —— **全部基于「冷档 FP16」错前提，应 ×1 不是 ×8，需重算**。冷档本就 1 byte/elem，与 IQ2_XXS（~2 bit）只差 ~4×，不是 8×。

### 3.2 专家命名 = w1/w2/w3，不是 gate/up/down [✓证]

- **实测**：`generate_one_expert` 的 part 名（`deepseek4-quantize.c`:1006-1008）= `w1`/`w2`/`w3`；HF 张量 `layers.L.ffn.experts.E.w{1,2,3}.weight`。
- **修正**：`react-go-training-design.md` §5.1/§6.1 的 LoRA `target_modules` 正则写的是 `mlp\.experts\.\d+\.gate_proj`/`up_proj`/`down_proj` —— **对 DeepSeek-V4-Flash 匹配 0 个模块**，训练会触发 C2 自检失败（可训练参数 ≈ 0%）。**必须改成** `layers.L.ffn.experts.E.w{1,2,3}`（及 `ffn.shared_experts.w{1,2,3}`）。这是 M4 训练的硬前置（见任务 T4-init）。

### 3.3 量化器旧专家路径写死 FP4（致命，已修） [✓证 → 已 B1 修复]

- **实测**：旧 `generate_one_expert` 无条件走 `dequant_fp4_weight`（要 I8 + F8_E8M0 + 2-per-byte 打包）；dense 路径才有 F8_E4M3 分支。**喂真实 F8 专家第一个就 die**。
- **wave-81「dry-run 字节精确」是只走元数据路径的假绿**（没碰真实张量）。
- **已由 B1 修复**：`generate_one_expert` 按 `w.dtype` 分流（F8_E4M3→fp8 / I8→fp4 向后兼容，:1406/:1415），`dequant_fp8_weight` 接 F32 block-scale。✅ + 真实数据 3/3 验证（§2）。

### 3.4 hash 层（前 3 层 DS4_N_HASH_LAYER=3）裸奔（已修护栏） [✓证 → 已 B2a 修复]

- **实测**：hash_mode 走 `kernel_dsv4_router_finalize_one` 直拷原始 id，**跳过 -inf 屏蔽**；被删/被屏蔽专家在 `route_translate`（`metal/dsv4_misc.metal`:250）**静默钳到 slot 0**（满权重错路由，无 renorm、无报错）。三个 python 工具均不特判 layer<3。
- **修正**：设计文档的 mask/shrink 流程默认对所有层一致裁剪，会让前 3 层错路由。
- **已由 B2a 修复**：`make_expert_mask.py --hash-layers 3` 强制前 3 层 keep 全 256，让 route_translate clamp **物理不可能**（hash 按 token-id 确定性路由，无法靠激活幅度压缩）。✅

### 3.5 20GB 预算漏算 hash 层需全 256 常驻 [✓证]

- **实测**：§3.5.2 的 20GB 常驻预算**未计入 hash 层需全 256 常驻** —— 3 层 × 256 × ~6.75 MiB ≈ **5 GiB 未入账**。
- **修正**：预算应 ~**25 GiB**，或 hash 层改 `tid2eid`-可达子集（只常驻 hash 实际能选到的 eid，而非全 256）。本文件不强行定夺，标为 §7 开放问题，落地时在 M2 bring-up 决定。

### 3.6 eval 闭环跑不了（需新建执行型 harness） [✓证]

- **实测**：现有 `ds4-eval` 是 MMLU / C-漏洞行号抽取 grader（`--self-test-extractors`，ds4_eval.c:1532 + `extractor_self_test_case`:3204），**没有** react/go test-pass 执行 / Opus-as-judge / 工具幻觉率。
- **修正**：设计文档 §8/§10「复用 ds4-eval」对 react/go 三维裁判**不成立**。需新建执行型 eval harness（B4，独立工程，前置）。

### 3.7 keep-map「未实现」结论部分过期 [✓证]

- **实测**：memory「k16 GGUF stale/broken：routing kernel keep-map 屏蔽未实现」**部分过期** —— router-side -inf 屏蔽已在 working tree 落地（`ds4_metal.m` +416 行），但 **① 未提交未验证；② 只覆盖非 hash 层**（门控 `has_bias && !hash_mode`，:248）。
- **修正**：设计文档 §3.7.3「整个动态方案前置=先修复 keep-map 屏蔽内核」的「从零修」表述需更新为「**提交 + 验证 working tree 已有的实现 + 补 hash 层（B2a 已补 mask 侧，运行时侧靠 hash 全 keep 规避 clamp）**」。

---

## 4. 架构决断：Mode P 优先 / §3.8 推迟（本会话拍板，关键路径）

> 这是本文件最重要的**钉死项**。它决定了关键路径里有什么、没有什么。

### 4.1 Mode P（react/go）自洽于热 GGUF —— 运行时不需要读 HF [✓证 机制 + 决断]

- **机制**：冷专家被屏蔽出路由 —— 非 hash 层走 router-side `-inf` 屏蔽（§3.7 已落地，待提交验证）；hash 层走 B2a 全 keep（route_translate clamp 物理不可能）。
- **结论**：Mode P 下被屏蔽的冷专家**永不被 top-6 选中** ⇒ 运行时**根本不需要读 HF safetensors**。热档 GGUF（仅 react/go 热专家 IQ2 + keep_map）即自洽。
- **质量补偿**：Mode P 的质量损失（冷专家被屏蔽 + 热专家 2-bit）靠 `react-go-training-design.md` 的领域训练买回，不靠运行时取冷档。

### 4.2 §3.8 混合 HF 冷档流式 = 最大未建块（一行没写） [✓证 缺口]

§3.8（Mode G 全质量）需要从零造下列**全部**模块（实测仓库均无）：

| 未建块 | 现状证据 | 工程量 |
|---|---|---|
| 运行时 safetensors loader | `ds4.c` 自写 loader **只读 GGUF**（CLAUDE.md：purpose-built for GGUF） | 大：解析 header / expert id→ST 张量映射 / mmap 分片 / 按需 fault |
| e4m3 专家 Metal kernel | 现 e4m3 **只服务 KV**；`metal/moe.metal` **无专家 e4m3 matmul 分支** | 大：MoE gather kernel 加「F8-from-ST」源分支 |
| 双机 ST 分片感知 | `ds4_distributed.c` **零 ST 感知**（只切 GGUF 层） | 中：worker 需知本机 ST shard 的层/专家范围 |
| Mode P/G 切换状态机 | 不存在 | 中：mode flag + madvise 预热 + KV 隔离 |
| 测试脚本 | 不存在 | 中 |

### 4.3 决断：砍掉 Mode G 这一期，§3.8 整块推迟成独立后续阶段

- **决断**：**关键路径 = 只做 Mode P。** Mode G / §3.8 HF 冷档流式整块 ⏸ 推迟，作为**独立后续阶段**，不阻塞 react/go。
- **理由**：
  1. Mode G 依赖 §4.2 的**最大未建块**（运行时 ST loader + 专家 e4m3 kernel + 双机 ST 分片，全从零）。
  2. react/go 核心目标（编码场景逼近 Opus 4.6）**不需要 Mode G** —— Mode P 已自洽（§4.1）。
  3. 把工程量压在能直接服务目标的 Mode P 上，符合「不 gold-plate / 不半成品」与「禁兜底·一切最优」铁律（Mode G 不是兜底，是未来增量）。
- **连带**：§3.1 的 FP8 数字重算只影响 Mode G 的速度估算 —— 既然 Mode G 推迟，FP8 数字重算降为「推迟项的待办」，不阻塞关键路径。

### 4.4 q2 81GiB IQ2 GGUF 在 M1 worker，不拷回本机 [✓证 + 铁律]

- **实测**：本机 `gguf/` 仅 `mask-k16.bin`/`mask-k48.bin`，`ds4flash.gguf` 悬空（wave-82 补）。
- **铁律**：q2 在 **M1 worker**，本机磁盘放不下不恢复 ⇒ 需 q2 的步骤（Path Y shrink / `--template` / k16 parity）**在 M1 跑，不拷回本机**。

---

## 5. 四个 blocker 终态

| blocker | 描述 | 终态 | 依据 |
|---|---|---|---|
| **B1** | 量化器专家路径 FP4→FP8 | ✅ 完成 + 真实数据 3/3 验证 | §2 / §3.3 / wave-82 ② + 补 |
| **B2a** | hash 层 mask 护栏 | ✅ 完成 | §2 / §3.4 / wave-82 ③ |
| **B3** | Mode P/G 物理布局互斥 | ✅ **由「推迟 Mode G」化解**（Mode P 自洽，不需要同机共存全模型 + 热档两套布局） | §4 决断 |
| **B4** | react/go 执行型 eval harness | 🔲 **未做，前置** | §3.6 / wave-82 ④ |

> B4 是唯一仍卡关键路径的 blocker：**SWE-bench 式 react/go test-pass + Opus 4.6 基线获取门**。无它则后续所有里程碑的「校验门」无法判定。

---

## 6. 关键路径里程碑 M-1..M6 + 推迟项

> 替换 `react-go-opus46-design.md` 原 M0–M6 排序。每个里程碑标：**动作 / 交付 / 校验门 / 位置**（M4-本机 | M1-worker | 离线）/ **授权**。
> 全程 correctness 门（CLAUDE.md）：`--dump-logprobs` parity + `ds4_test --metal-kernels --server`；routing/quant 改动加 `--logprob-vectors` + `ds4-eval q1..q4 --temp 0 --seed 1`。
> **铁律**：ds4 推理二进制改动需**同步 M1 重编**（共享 `CORE_OBJS`）；载模型脚本先证内存安全 + 逐次授权；总结性结论需用户批准再写 `notes/execution-log.md`。

### M-1 地基（新增·最高优先）

- **动作**：① **提交** working tree 的 router-side mask（`ds4_metal.m` +416 / `ds4.c` +4）；② 写 **B2b**（`route_translate` fail-loud 计数器，verify 模式检测冷 clamp —— 若 Mode P 下仍有专家被 clamp 到 slot 0 即报警，证明屏蔽漏了）；③ 单机 **k16 logprob-vectors parity** + `ds4-eval q1..q4`。
- **交付**：可提交的屏蔽内核 + B2b 计数器 + k16 parity 报告。
- **校验门**：k16 单机数字正常（无「数字汤」）；`ds4_test --logprob-vectors` 绿；`ds4-eval q1..q4 --temp 0 --seed 1` token 数对齐 README；B2b 计数器在 Mode P 下 clamp 计数 = 0。
- **位置**：B2b 代码改动 = **M4-本机**；k16 parity 跑 = **M1-worker**（q2 在 M1）。改后 **M1 重编**。
- **授权**：k16 parity 跑 = 🔒 gated（载模型，逐次授权）。

### M0 eval（B4，前置）

- **动作**：建 react/go **执行型** eval harness —— SWE-bench 式 test-pass 执行 + Opus-as-judge 架构打分 + 工具幻觉率统计；建 eval 集（100~200 题，react/go 各半，含算法/复杂度题，holdout 来自角度 1）；取 **Opus 4.6 参考分**。
- **交付**：eval harness（独立工程）+ eval 集 + Opus 4.6 参考分（在案）。
- **校验门**：harness 能跑通一个已知 PR 的 test-pass；Opus 4.6 分可复现、记录在案。
- **位置**：离线（harness 工程）+ M4 接 `ds4_agent.c` 跑被测模型。
- **授权**：取 Opus 4.6 参考分 = 调外部 API（成本/合规门，需授权）。

### M1 画像

- **动作**：react/go imatrix（步骤 1，🔒 gated 载模型）→ `router_norms_from_imatrix.py`（energy 排常驻集 + specialty 训练靶向，离线安全）→ hot mask（`make_expert_mask.py`，含 `--hash-layers 3` 护栏）。
- **交付**：`reactgo_router.dat` + `router_norms_reactgo.json` + `router_norms_specialty.json` + `mask-prog-hot.bin`。
- **校验门**：imatrix 收敛；router_norms self-test 绿；mask 含 hash 全 keep（B2a 已保证）；specialty baseline 纯净度验证（设计 §7.1）。
- **位置**：imatrix 跑 = **M1-worker**（q2/模型在 M1）；norms/mask 派生 = **M4-本机**（离线 64MB 流式）。
- **授权**：imatrix 跑 = 🔒 gated（载模型 + general baseline 也需一次 imatrix 跑）。

### M2 bring-up

- **动作**：Path Y（在 **M1** shrink q2，留 hot+warm 物理删死档）→ hot-only Mode P GGUF（`deepseek4-quantize --experts-hot-mask`）→ 双机装载 + gather 带宽重平衡（wave-80 cycle-4：慢 mini 超额抢 37%，按真实带宽重平衡抢单）。
- **交付**：`ds4-prog-hot.gguf`（hot-only）+ 双机跑通 + gather 重平衡 patch。
- **校验门**：两机看门狗 ≤12GiB（注意 §3.5 的 hash 全 256 ~5GiB 入账，预算按 ~25GiB 或 tid2eid 子集核算）；`ds4_test --metal-kernels --server` 绿；keep_map 路由正确（接 M-1 k16 验证）。
- **位置**：Path Y shrink + hot-only quantize = **M1-worker**（需 q2 模板 + 全分片）；gather 重平衡代码 = **M4-本机**，改后 **M1 重编**。
- **授权**：shrink / quantize / 双机跑 = 🔒 gated（载模型）。

### M3 语料

- **动作**：三类语料（bug 修复 50% / 书蒸馏 30% / 诚实 agent 20%，**含算法覆盖** —— §7.1 strict 决策的连带约束）+ holdout 留出。
- **交付**：3 万~8 万 SFT 集 + eval holdout。
- **校验门**：去重（MinHash LSH ~0.85）/ 许可合规（MIT/Apache/BSD）过；配比记录；算法类样本占足（角度 1 算法 bug + 角度 2 算法导论蒸馏）。
- **位置**：离线（数据工程）。
- **授权**：Opus 4.6 蒸馏（角度 2）= 调外部 API（成本/合规门）。

### M4 训练

- **动作**：ApiQ 初始化（**修 w1/w2/w3 命名**，§3.2）→ MoE-LoRA（挂 experts + shared，冻结 router）→ merge bf16 → B1 仅热档重量化 IQ2。
- **交付**：apiq-init 权重 + 适配器 + merged 小 GGUF。
- **校验门**：可训练参数 ≫ 0.022%（C2 自检，命名修对才过）；路由 top-K 重合率达标（冻结 router）；merge 后 react/go eval ≥ M0 基线；`--dump-logprobs` parity；`metal-kernels --server` 绿。
- **位置**：训练 = 离线大内存机/云 GPU（16GB Mac 训不动）；merge 后重量化 = **M1-worker**（需 q2 模板）。
- **授权**：训练跑（云/大机）= 资源门。

### M5 调优

- **动作**：copy-spec 作 30 t/s 达标杠杆（`DS4_DIST_COPY_SPEC`，编程域 prefix 复用 + n-gram 草稿 VERIFY）+ prompt 体系接 `ds4_agent.c`（faithful CoT / Reflexion / plan-then-execute）+ 配比消融。
- **交付**：系统提示词 + 配比结论 + copy-spec 调优数据。
- **校验门**：工具幻觉率↓；react/go 通过率↑；瞬时 decode t/s 向 30 逼近（`ds4-bench`）。
- **位置**：copy-spec 调优 = **M4-本机 + M1-worker**（双机 decode）；prompt = M4。
- **授权**：双机跑 = 🔒 gated。

### M6 收敛

- **动作**：对标 Opus 4.6，诚实记录差距（Δ_domain vs Δ_quant）。
- **交付**：逼近度报告 + 设计回填。
- **校验门**：不粉饰差距；Δ_domain / Δ_quant 在案。
- **位置**：离线分析。
- **授权**：—

### [推迟] Mode G / §3.8 HF 冷档流式

- **状态**：⏸ 独立后续阶段，不阻塞 react/go。
- **待办**（推迟项内部）：FP8 数字重算（§3.1）→ 运行时 safetensors loader → 专家 e4m3 Metal kernel → 双机 ST 分片感知 → Mode P/G 切换状态机 → 测试脚本（§4.2 五块全从零）。

---

## 7. 任务拆分总表（核心交付）

> 逐任务一行，可独立认领。列：**ID | 任务 | 状态 | 位置 | 依赖 | 校验门 | 授权**。
> 位置：M4=本机 Mac Mini | M1=worker MacBook | 离线=大机/云/纯 python。

| ID | 任务 | 状态 | 位置 | 依赖 | 校验门 | 授权 |
|---|---|---|---|---|---|---|
| **B1** | 量化器专家路径 FP4→FP8（`deepseek4-quantize.c`：`load_f32_le`/`dequant_fp8_weight` 接 F32\|F8_E8M0/`generate_one_expert` 按 dtype 分流） | ✅ | 离线 | — | `make` 绿 + 真实数据 dequant（已 3/3） | — |
| **B2a** | hash 层 mask 护栏（`make_expert_mask.py` `force_keep_hash_layers`/`--hash-layers 3`；驱动透传） | ✅ | 离线 | — | 自测全 keep / top-K / 默认 3（已过） | — |
| **DL** | HF index.json + 分片下载（`hf/DeepSeek-V4-Flash-Base/`） | ✅（25/46 分片） | M4 | — | index.json 在盘 + layers 0-5 齐（已）；剩余分片补全 | — |
| **RN** | `router_norms_from_imatrix.py`（energy + specialty） | ✅（self-test） | 离线 | — | self-test 绿（已）；真实 .dat 跑见 M1-imatrix | — |
| **HM** | `deepseek4-quantize --experts-hot-mask`（DSXM + keep_map KV） | ✅（dry-run） | 离线 | B1 | dry-run 字节精确（已）；真实端到端见 M2-quant | — |
| **DRV** | `quantize_reactgo.sh` 驱动（载模型步门控） | ✅ | 离线 | RN,HM | 门控 smoke（已） | — |
| **M-1.1** | **提交** working tree router-side mask（`ds4_metal.m` +416 / `ds4.c` +4） | ⚠️ 未提交未验 | M4 | — | commit；**改后 M1 重编**（CORE_OBJS） | — |
| **M-1.2** | **B2b** `route_translate` fail-loud 计数器（verify 模式检测冷 clamp） | ⚠️ 代码落地·M4 验证 | M4 | M-1.1 | Mode P 下 clamp 计数=0；动热路径，与 k16 同跑（**改后 M1 重编**：动 CORE_OBJS+.metal） | 跑=🔒 |
| **M-1.3** | 单机 **k16 logprob-vectors parity** + `ds4-eval q1..q4`（驱动：`tools/reactgo_k16_verify.sh`，wave-83 落地，DRY-RUN/`RUN=1` 门控 + RSS 看门狗 + L1 gate） | 🔒 gated | **M1** | M-1.1,M-1.2 | k16 数字正常（无数字汤=G2）；B2b clamp=0（G1）；q1..q4 token 对齐（G3）；`--logprob-vectors` 绿 | 🔒 逐次 |
| **M0.1** | **B4** react/go 执行型 eval harness（test-pass 执行 + Opus-judge + 工具幻觉率） | 🔲（前置） | 离线+M4 | — | 跑通一已知 PR 的 test-pass | — |
| **M0.2** | react/go eval 集（100~200 题，含算法/复杂度，holdout） | 🔲 | 离线 | M0.1 | eval 可复现 | — |
| **M0.3** | 取 Opus 4.6 参考分 | 🔲 | 离线 | M0.1,M0.2 | 参考分在案、可复现 | 外部 API 门 |
| **M1.1** | react/go imatrix 真实跑（`ds4 --imatrix-out`） | 🔒 gated | **M1** | DL | imatrix 收敛；`reactgo_router.dat` 产出 | 🔒 逐次 |
| **M1.2** | general baseline 语料构建（`build_general_baseline.py`，strict 档 `--math drop`） | 🔲 | 离线 | — | 纯净度验证（§7.1：与 react/go 激活可分） | — |
| **M1.3** | general imatrix 真实跑（specialty 分母） | 🔒 gated | **M1** | M1.2,DL | 对称口径（同 ctx/render/part/GGUF） | 🔒 逐次 |
| **M1.4** | 派生 energy + specialty norms（`router_norms_from_imatrix.py`） | 🔲 | M4 | M1.1,M1.3 | self-test 绿；specialty sanity（纯代码专家 ≫0） | — |
| **M1.5** | hot mask（`make_expert_mask.py --hash-layers 3`，energy 排，每层 K） | 🔲 | M4 | M1.4 | hash 全 keep（B2a）；K 由覆盖率定 | — |
| **M2.1** | Path Y shrink q2（留 hot+warm，物理删死档） | 🔒 gated | **M1** | M1.5 | shrunken GGUF 装载正常；不拷回本机 | 🔒 逐次 |
| **M2.2** | hot-only Mode P GGUF（`deepseek4-quantize --experts-hot-mask` 真实数据） | 🔒 gated | **M1** | B1,M1.5 | keep_map KV 正确；端到端字节写 | 🔒 逐次 |
| **M2.3** | 20GB/hash 预算核算（~25GiB 或 hash tid2eid 子集决策，§3.5） | 🔲 | M4 | M1.5 | 两机看门狗 ≤12GiB 可行性 | — |
| **M2.4** | 双机装载 + gather 带宽重平衡（cycle-4 抢单调度，wave-80） | 🔲 | M4→M1 | M2.2,M2.3 | 两机 ≤12GiB；`metal-kernels --server` 绿；**改后 M1 重编** | 跑=🔒 |
| **M3.1** | 角度 1 语料：bug 修复 diff（50%，含算法类 bug） | 🔲 | 离线 | — | 去重/许可合规；配测试 | — |
| **M3.2** | 角度 2 语料：书蒸馏重构（30%，含算法导论蒸馏） | 🔲 | 离线 | — | 生成代码可编译/测试过 | Opus API 门 |
| **M3.3** | 角度 3 语料：诚实 agent（20%，含负样本） | 🔲 | 离线 | — | faithful CoT 真实回填 | — |
| **M3.4** | holdout 留出（来自角度 1，不入训练） | 🔲 | 离线 | M3.1 | 与 M0.2 eval 一致 | — |
| **M4.1** | **ApiQ 初始化 + 修 LoRA target_modules 命名 w1/w2/w3**（§3.2） | 🔲 | 离线 | M1.4 | 可训练参数 ≫0.022%（命名修对才过） | 资源门 |
| **M4.2** | MoE-LoRA SFT（挂 experts+shared，冻结 router；specialty 靶向 A/B） | 🔲 | 离线 | M4.1,M3.* | eval ≥ M0；路由 top-K 重合率达标 | 资源门 |
| **M4.3** | merge bf16 + B1 仅热档重量化 IQ2 | 🔒 gated | **M1** | M4.2,B1 | `--dump-logprobs` parity；`metal-kernels --server` 绿 | 🔒 逐次 |
| **M5.1** | copy-spec 作 30t/s 杠杆（`DS4_DIST_COPY_SPEC` 调优） | 🔲 | M4→M1 | M2.4 | 瞬时 decode t/s 向 30 逼近 | 跑=🔒 |
| **M5.2** | prompt 体系接 `ds4_agent.c`（faithful/Reflexion/plan） | 🔲 | M4 | M0.1 | 工具幻觉率↓ | — |
| **M5.3** | 配比/靶向消融 | 🔲 | 离线 | M4.2,M0.* | 配比结论记录 | 资源门 |
| **M6.1** | 对标 Opus 4.6，记 Δ_domain vs Δ_quant | 🔲 | 离线 | M4.3,M5.* | 不粉饰差距 | — |
| **[P]§3.8.1** | FP8 数字重算（省盘账/RB/Mode G 速度，§3.1） | ⏸ | 离线 | — | （推迟项） | — |
| **[P]§3.8.3** | 运行时 safetensors loader（`ds4.c`） | ⏸ | M4 | — | （推迟项，最大未建块） | — |
| **[P]§3.8.3b** | 专家 e4m3 Metal kernel（`metal/moe.metal`） | ⏸ | M4 | — | （推迟项） | — |
| **[P]§3.8.2** | 双机 ST 分片感知（`ds4_distributed.c`） | ⏸ | M4 | — | （推迟项） | — |
| **[P]§3.7** | Mode P/G 切换状态机 | ⏸ | M4 | — | （推迟项） | — |

---

## 8. 风险与开放问题

> 继承设计文档 `react-go-opus46-design.md` §10 / `react-go-training-design.md` §13；下列是**本会话新增 / 修正**。

| R | 描述 | 处置 | 档 |
|---|---|---|---|
| RX1 | **FP8 数字重算** —— §3.8 省盘账 / RB / Mode G 速度全基于「冷档 FP16」错前提（应 ×1 不 ×8） | Mode G 推迟（§4.3），重算降为推迟项待办；不阻塞 react/go | [✓证 错前提] |
| RX2 | **20GB+5GiB hash** —— hash 层全 256 常驻 ~5GiB 未入预算 | M2.3 决策：预算 ~25GiB 或 hash tid2eid-可达子集 | [✓证] |
| RX3 | **eval harness 工程量** —— 现有 ds4-eval 跑不了 react/go 三维裁判，B4 是从零的独立工程且前置 | M0 最高优先之一；无它后续校验门无法判定 | [✓证] |
| RX4 | **未提交 mask 待验** —— router-side -inf 屏蔽已写但未 commit、未跑 logprob parity，且只覆盖非 hash 层 | M-1.1 提交 + M-1.3 k16 parity；hash 侧靠 B2a 全 keep 规避 | [✓证 / 假设 数值] |
| RX5 | LoRA target 命名错（gate/up/down vs w1/w2/w3） | M4.1 修命名；C2 自检（可训练参数 ≫0.022%）是硬门 | [✓证] |
| RX6 | 「语料+训练→Opus 4.6」因果链未证（核心假设） | eval 闭环唯一裁判；M6 诚实记差距 | [假设] |
| RX7 | FP8→IQ2→领域训练三段损失叠加无人测 | T1/M-? 单独量 Δ_quant，不假定线性可加 | [假设] |
| RX8 | 30 t/s 取决于 react/go 路由集中度 + copy-spec 杠杆，非单次 forward 物理可达 | M5.1 copy-spec；瞬时 t/s 实测，不预判封顶 | [假设] |

**开放问题（无数据，需原型/实测）**：
1. Mode P 屏蔽冷专家后，react/go 编码净掉点（Δ_quant 含 FP8→IQ2）多少？领域训练 Δ_domain 能否压过到 4.6？
2. hash 层是否真需全 256 常驻，还是 tid2eid-可达子集足够（影响 5GiB 预算）？
3. specialty 靶向（rank 分层 / 采样加权）是否优于热集均匀 rank？
4. 三类语料配比/规模需多大才把损失顶回接近 Opus 4.6？
5. 冻结 router 是否足以保热集稳定，还是需 EAQuant routing-consistency 对齐？

---

## 9. 安全闸提醒（硬约束，铁律）

- **载模型脚本先证内存安全**：RSS 预算 + 看门狗（12/12 GiB 红线，两机）；任何会加载模型的脚本（imatrix / shrink / quantize / 双机跑 / 训练回灌）必须先证明 RSS 预算 + 看门狗安全才能跑。
- **逐次授权**：每次改完代码后跑脚本前必须单独获授权；循环授权 ≠ 一次性放行所有跑。
- **不双载 base**：单机不双载 81/82 GiB base；重量化只动热档（~9GiB），不重写全模型。
- **不删远程文件**：未经用户明确确认不得删除另一台机器上任何文件；「清理」= 关进程，不是删文件。
- **q2 在 M1 不拷回**：本机磁盘放不下 q2，所有需 q2 模板的步骤（Path Y / `--template` / k16 parity / hot-only quantize / merge 重量化）在 M1 跑，不拷回本机。
- **rebuild → 另一台**：本机改 ds4 推理二进制后必须考虑 M1 是否需同步重编 + 传二进制（共享 `CORE_OBJS`）。
- **结论需批准**：客观记录可随时写 `notes/execution-log.md`；总结性 / 主观结论需用户同意后才写日志。
