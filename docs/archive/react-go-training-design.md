# DeepSeek-V4-Flash react/go 领域训练技术方案（逼近 Opus 4.6）

> 版本：v1（2026-06-20）。从 `react-go-opus46-design.md`（量化/裁剪主线）抽离的**训练专册**，自包含。
> 定位：量化已把 react/go 热专家压到 2-bit 常驻（精度倒置，§量化文档 3.6）；本册讲**如何把这批 2-bit 热专家在 react/go 域训回、并越过通用基线到 Opus 4.6**。
> 上游依赖：`react-go-opus46-design.md`（P0 压缩 / 热档 mask / keep_map / 混合格式）、`gguf-tools/router_norms_from_imatrix.py`（专家激活 + 编程专属 specialty）。
> 交付物：可直接执行的训练管线——初始化方法、LoRA 落点、超参、三类语料、注入 ds4、eval 闭环、里程碑、资源测算，每步带证据档与校验门。

---

## 目录

- [0. 诚实边界与证据分级](#0-诚实边界与证据分级)
- [1. 训练目标与可证伪判定](#1-训练目标与可证伪判定)
- [2. 训练在整条管线中的位置](#2-训练在整条管线中的位置)
- [3. 三条硬约束（踩错即废）](#3-三条硬约束踩错即废)
- [4. 训练对象：到底训哪些权重](#4-训练对象到底训哪些权重)
- [5. 方法 A — 数值稳定初始化（ApiQ / LoftQ）](#5-方法-a--数值稳定初始化apiq--loftq)
- [6. 方法 B — MoE-LoRA 落点与超参](#6-方法-b--moe-lora-落点与超参)
- [7. 编程专属靶向（specialty 驱动）](#7-编程专属靶向specialty-驱动)
- [8. 三类语料工程](#8-三类语料工程)
- [9. 适配器注入 ds4（merge 回 2-bit 热专家）](#9-适配器注入-ds4merge-回-2-bit-热专家)
- [10. eval 闭环（唯一裁判）](#10-eval-闭环唯一裁判)
- [11. 训练里程碑 T0–T5](#11-训练里程碑-t0t5)
- [12. 资源与可行性测算](#12-资源与可行性测算)
- [13. 风险登记 + 开放问题](#13-风险登记--开放问题)
- [14. 引用](#14-引用)

---

## 0. 诚实边界与证据分级

继承项目铁律"执行路径保持诚实"。全文每条标档：

- **[✓证]** deep-research 对抗验证确认（3-0 / 2-1）。
- **[假设]** 设计推断，无验证 claim 直接支撑，**必须用 eval 闭环证伪**。
- **[✗驳]** 被驳倒，**禁止作为依据**。

**训练侧三条最该记住的诚实警告**：

1. **「语料 + 训练 → Opus 4.6 编码水平」这条因果链没有任何被验证 claim 量化支撑，是本方案的核心假设。** eval 闭环（§10）是唯一裁判，"4.6" 是可证伪目标，不是预设结论。
2. **从 FP8 base 出发、压 IQ2_XXS、再领域训练，三段损失叠加无人测过**（DeepSeek V4 专家本身 `expert_dtype=fp8`）。不能假定线性可加，T1 必须实测。
3. **16GB Mac 训不动剪枝后的 V4**（开放问题）。训练一律离线在大内存机 / 云 GPU 产出适配器，Mac 只承担推理 + 适配器加载。

**[✗驳] 清单（禁用）**：① "LoRA 在 SFT 等同全量微调"(0-3)；② "MoE 每专家单独 LoRA、rank=总rank/激活专家数"(0-3)；③ "mlx-lm 完全没给专家挂 LoRA"(1-2，仅 *0.022% 可训练参数* 事实确认)；④ "QLoRA 零初始化在 2-bit 直接可用"(被 LoftQ "fails below 3-bit" 驳)。

---

## 1. 训练目标与可证伪判定

- **目标**：量化版 DeepSeek-V4-Flash 在 **react / go** 编码场景逼近 **Claude Opus 4.6**（功能正确率 + 架构质量 + 执行诚实三维，§10）。
- **窄域特化假设 [假设]**：DeepSeek 官方称通用编码 ~4.5；窄域（仅 react/go）专化**可超通用基线**（领域特化现象，方向有论文支撑），但：

  ```
  净能力 = 4.5(通用基线)  −  Δ_quant(2-bit 损失)  +  Δ_domain(领域训练增益)
  ```

  只有 `Δ_domain > Δ_quant` 时净值 > 4.5。窄到只做 react/go，增益**有机会**压过 2-bit 损失到 4.6 —— **可能，非保证**。
- **判定法**：T1 先量化 react/go eval 掉点 `Δ_quant`；T3 训练后量化净增益 `Δ_domain`；M6 对标 Opus 4.6 参考分，诚实记录逼近度（不粉饰差距）。

---

## 2. 训练在整条管线中的位置

```
HF DeepSeek-V4-Flash-Base (safetensors, experts=FP8 e4m3)
   │
   │  [量化文档 P0 / §3.6 精度倒置]
   │   react/go imatrix → router_norms(energy) → 热档 mask(top-K/层)
   │   deepseek4-quantize --experts-hot-mask → 小 GGUF: 仅热专家 IQ2_XXS + keep_map
   │   冷专家不入 GGUF（§3.8 从 HF FP8 流式）
   ▼
┌─[本册 P1 训练]──────────────────────────────────────────────┐
│  T1 量化掉点基线: react/go eval(量化前 vs 后) → Δ_quant       │ 离线
│  T2 ApiQ/LoftQ 初始化: 在 2-bit 目标下并发初始化 LoRA+量化    │ 大内存机
│  T3 MoE-LoRA SFT: 挂热专家槽, 三类语料(§8), specialty 靶向    │ /云 GPU
│  T4 merge 回热专家 → 重走量化(热档 IQ2) → 最终小 GGUF         │
└──────────────────────────────────────────────────────────────┘
   ▼
ds4 装载 (Mode P 热档常驻, §3.7) → eval 闭环对标 Opus 4.6 (§10)
```

**只训热档**：冷专家保 FP8 原样（次要"日常"模式 Mode G，不参与 react/go 训练）。训练面只覆盖 ~32 专家/层 × 43 层的热集 —— 既是性能面（常驻）也是训练面，自洽。

---

## 3. 三条硬约束（踩错即废）

| # | 约束 | 依据 | 后果 |
|---|---|---|---|
| **C1** | 2-bit 后**禁**标准 QLoRA（零初始化适配器）；**必须** ApiQ/LoftQ 初始化 | [✓证 F6] LoftQ: standard QLoRA **fails below 3-bit**，正落 ds4 2-bit 区间 | 不收敛 / 退化 |
| **C2** | LoRA **必须**命中 `mlp.experts` 投影；可训练参数 **≫ 0.022%** | [✓证 F5] attn-only 显著欠拟合；mlx-lm bug 只训 0.022% | 适配器没碰专家 = 训练≈无效 |
| **C3** | 训练在**量化空间内**进行（QAT 思路），不是先训 bf16 再盲压 | [✓证 F7] ApiQ 并发初始化量化+LoRA、抑误差从浅向深传播 | merge 后再量化损失放大 |

**C2 落地自检（强制）**：训练启动后 dump 可训练参数量，**必须远高于 0.022%**，否则适配器没命中专家层，立即停查（mlx-lm bug 阈值）。

---

## 4. 训练对象：到底训哪些权重

诚实拆清，避免"训了个寂寞"：

| 组件 | 量化部署态 | 训练态 | 训不训 |
|---|---|---|---|
| react/go **热专家** gate/up/down | IQ2_XXS（GGUF 常驻） | HF FP8 → 反量化 bf16 + LoRA | **训**（核心面） |
| shared expert | Q8_0 | bf16 + LoRA | **训**（编码增益大，F5） |
| **冷专家** | 不在 GGUF（HF FP8 流式） | 不动 | **不训**（Mode G 通用质量保 FP8） |
| router / gate (`ffn_gate_inp`) | Q8 backbone | **冻结** | **不训**（见下） |
| attention / 其余 backbone | Q8_0 | 冻结 | 不训（attn 可加但不单独挂，F5） |

**为什么冻结 router [假设]**：只训专家**内容**、冻结路由**选择**，避免训练漂移路由分布导致 keep_map 热集失配（热档 mask 是按训练前激活定的；若 router 漂移，推理选到冷专家 → cold miss → 掉速 + 掉质）。EAQuant 的 *routing consistency alignment* 是此处的备选增强：若 T3 后实测路由偏移大，再考虑对齐而非放开 router。**T3 校验门含"训练前后 react/go 路由 top-K 重合率 ≥ 阈值"。**

**叠加损失诚实点 [假设/未测]**：HF 专家本就是 FP8 训练产物；本管线再压 IQ2_XXS。`Δ_quant` 包含 FP8→IQ2 这一段，T1 必须把它单独量出来，不能假定与通用量化掉点相同。

---

## 5. 方法 A — 数值稳定初始化（ApiQ / LoftQ）

不能直接 QLoRA（C1）。**首选 ApiQ**（2402.05147，专为 2-bit；并发初始化 LoRA 分量 + 量化权重，保激活精度、抑误差传播）；备选 LoftQ（2310.08659，交替优化 `min‖W−Q−AB‖_F`，量化残差 SVD 初始化 A/B）。ApiQ 在 2-bit 持续优于 LoftQ → 首选。

```bash
git clone https://github.com/baohaoliao/apiq && cd apiq
python apiq_init.py \
  --model <剪枝/原始 DeepSeek-V4-Flash HF 路径> \
  --bits 2 \                       # 对齐 ds4 IQ2_XXS 区间
  --lora-rank 64 \                 # 编码域 32~128 扫
  --lora-alpha 128 \
  --target-modules experts \       # 必须命中专家投影 (C2)
  --output ./V4-Flash-apiq-init
# 参数名以 apiq repo 实际为准；clone 后核对 --help
```

> **[警告]** ApiQ 适配器存 FP16 抬高平均位宽；partial update ≠ 完整 QAT（2504.13932）。部署仍走 merge→IQ2 重量化（§9）。

---

## 6. 方法 B — MoE-LoRA 落点与超参

### 6.1 LoRA 目标层（硬约束 C2）[✓证 F5]

```python
# PEFT / ApiQ target_modules（DeepSeek-V4 MoE 命名按 config 调整）
target_modules = [
    r".*mlp\.experts\.\d+\.gate_proj",
    r".*mlp\.experts\.\d+\.up_proj",
    r".*mlp\.experts\.\d+\.down_proj",
    r".*mlp\.shared_experts\.(gate|up|down)_proj",  # shared 也挂，编码增益大
    # attn 可加但不单独挂（attn-only 欠拟合）
]
```

**[✗驳 禁用]**：不要"每专家单独 LoRA、rank=总rank/激活专家数"(0-3)；不要假定 "LoRA==全量微调"(0-3) —— 预期 LoRA < 全量，用足够大 rank 弥补。

### 6.2 训练配置（离线大内存机 / 云）

```yaml
base:  V4-Flash-apiq-init
lora:  { rank: 64, alpha: 128, dropout: 0.05, targets: experts+shared }
optim: { lr: 1e-4, scheduler: cosine, warmup: 0.03, weight_decay: 0.0 }
batch: { per_device: 1, grad_accum: 16, max_seq_len: 8192 }  # 8k 容 react/go 长上下文
mem:   { gradient_checkpointing: true, bf16: true, freeze_router: true }
epochs: 2~3                  # 域 SFT 不宜过拟合
data:   见 §8 三类语料配比
```

`rank/alpha/lr/epochs` 全是**起点，需扫**；扫的判据是 §10 eval，不是 train loss。

---

## 7. 编程专属靶向（specialty 驱动）

承接 `router_norms_from_imatrix.py --baseline GENERAL.dat --rank-by specialty`：每专家 `specialty_log2 = log2(react/go激活份额 / 通用激活份额)`，>0 = 编程专属。

> ⚠️ **两口径别混**（与量化文档一致）：热档**常驻** mask 必须按 `energy`（react/go 实际路由集，**含通用专家**，否则 cold miss → 30t/s 崩）。`specialty` **只**用于训练靶向 / REAP 死档，不决定常驻集。

**靶向用法 [假设，需 T3 消融]**：在已常驻的热集内，对 `specialty_log2` 高的专家分配**更高 LoRA rank** 或**更高语料采样权重** —— 把训练算力压在 react/go 真正区分性的专家上，通用专家用较低 rank 维持。两种实现：

| 方式 | 机制 | 取舍 |
|---|---|---|
| rank 分层 | specialist rank=128 / generalist rank=32 | 参数集中专属面，但 PEFT 需支持 per-module rank |
| 采样加权 | 不变 rank，按命中 specialist 的样本上采样 | 实现简单，间接靶向 |

**[假设]**：靶向是否优于"热集均匀 rank"无先例，T3 必须 A/B（靶向 vs 均匀），eval 裁决。**不预设靶向更好。**

### 7.1 通用 baseline 语料清单（specialty 的前提，必须纯非编程）

`specialty_log2 = log2(react/go 份额 / 通用份额)` 的分母**完全由 baseline 语料决定**。**baseline 不纯（混了代码）→ 代码专家在分母里也被激活 → 份额比趋近 1 → specialty 被压低 → 挑不出编程专属专家。** 所以 baseline 必须是**纯非编程的日常使用**语料。

> ⚠️ 仓库现成的 `gguf-tools/imatrix/dataset/rendered_prompts.txt` **不能直接当 baseline**：它含 C/Metal source-review prompt 和 agent/DSML tool-call prompt（都是编程/agent），会污染分母。必须先剔除这些再用。

**baseline 的语义由"纯净档位"决定（先选档）**：

| 档 | 排除 | 把哪些算"编程专属" | 用途 |
|---|---|---|---|
| **strict（纯自然语言）** | 代码 **+ 数学/算法推理** | 代码语法 **和** 算法/数学推理专家都算专属 | 最大化 specialty 面（训练面更广） |
| **broad（含数学，仅排代码）** | 仅代码 | 只有代码语法/库特异专家算专属；算法专家算通用 | 只靶向"react/go 语法/生态"特异面 |

> **决策（用户拍板 2026-06-20）：算法推理算编程方向 → 选定 `strict` 档。** baseline 排代码 **+ 排数学/算法推理**，使代码语法专家**和**算法/数据结构推理专家**都**获正 specialty、进入编程专属面与训练靶向。`broad` 仅留作 T3 备选消融。
>
> **连带影响（必须配套，否则自相矛盾）**：算法专家既被标为"专属"并分到更高训练权重/rank（§7 靶向），§8 语料就**必须含足量算法/数据结构样本**喂它们——否则"专属但无料训"。落点：§8 角度1 须含算法类 bug 修复（非纯框架 bug），§8 角度2 须含《算法导论》核心算法的蒸馏样本。eval（§10）须含算法/复杂度题以验证该面真被训起来。

**类别清单与占比建议（代表日常分布）**：

| 类别 | strict 占比 | broad 占比 | 来源（开源/自带） |
|---|---|---|---|
| 多轮对话 / 闲聊 / 指令 | 30% | 25% | OpenAssistant(oasst)、Dolly、ShareGPT **非代码子集** |
| 写作 / 改写 / 润色 / 摘要 | 20% | 18% | WritingPrompts、CNN-DM、XSum；DS4 自带 prose-rewriting/summarization |
| 翻译（多语） | 15% | 12% | WMT、FLORES、OPUS；DS4 自带 translation prompt |
| 常识 / 知识问答 / 抽取 | 25% | 20% | NaturalQuestions、TriviaQA、MMLU(非 code)、GPQA/SuperGPQA；DS4 自带 extraction |
| 科学/数学文字推理 | — | 25% | GSM8K、MATH **文字题**、AIME（**broad 档才纳入**） |
| 创意 / 角色扮演 | 10% | — | 各类创意写作集 |

**最快可落地路径（从仓库自带 dataset 派生，零外采）**：DS4 自带 `gguf-tools/imatrix/dataset/` 已含多数非编程类别。直接按来源标签过滤：

```
保留 → prose rewriting / summarization / extraction / translation / GPQA / SuperGPQA
剔除 → C/Metal source-review prompts（编程）
剔除 → agent/tool-call DSML prompts（agent/编程）
过滤 → long-context snippets（逐条查，含代码的剔除）
strict 档额外剔除 → AIME（数学）；broad 档保留 AIME
thinking + non-thinking 前缀两者都保留（匹配 react/go calib 的推理模式）
```

**构建管线（离线、不载模型；跑 imatrix 才载模型、门控）**：

```bash
# 1. 过滤掉一切代码（启发式：```围栏 / import|func|def|package|#include 密度 / 文件扩展名 / DSML 标记）
python3 build_general_baseline.py \      # 需新写：从上述来源拼 + 去代码 + 去重
  --exclude-code --exclude-dsml \
  --math drop \                          # 选定 strict：排数学/算法（算法推理=编程方向）。broad 消融时改 keep
  --render-as ds4-chat \                 # 关键：与 react/go calib 同一渲染口径
  --out /tmp/general_calib.txt

# 2. 跑通用 imatrix（与 react/go imatrix 同 --ctx、同模型；唯一的模型跑，门控+授权）
ds4 -m gguf/ds4flash.gguf \
  --imatrix-dataset /tmp/general_calib.txt \
  --imatrix-out /tmp/general_router.dat --ctx 32768

# 3. specialty 评分（同 --part，对称口径）
gguf-tools/router_norms_from_imatrix.py /tmp/reactgo_router.dat \
  --baseline /tmp/general_router.dat --rank-by specialty \
  --n-expert 256 --out /tmp/router_norms_specialty.json
```

**对称口径要求（否则 specialty 混入噪声）**：baseline 与 react/go calib 必须**同渲染格式、同 `--ctx`、同 `--part` 聚合、同模型 GGUF**。任一不一致都会让份额比掺入口径差异而非真实领域差异。

**纯净度验证（baseline 是否真"非编程"）**：
- 对 baseline **自身** top-K 激活与 react/go top-K 求重合率：若 ≈100% 重合 → baseline 混了代码（分母被污染），重过滤；健康的 baseline 应与 react/go 有明显可分的激活分布。
- 抽样 200 条人工核查无代码块 / 无 import/func/package。
- sanity：已知"纯代码语法"专家在 react/go 应得 `specialty_log2 ≫ 0`；若接近 0，baseline 不纯。

> **[假设]**：strict 已选定；broad 训出的 react/go 能力是否反而更高无先例 → 仅作 T3 备选消融（与 §7 靶向 A/B 正交），**默认不跑**，除非 strict 在 eval 上算法面与框架面互相挤占。

---

## 8. 三类语料工程

> ⚠ 全段 [假设]：语料→Opus 4.6 因果链未证，eval 闭环裁判。react:go 内部比例建议 1:1 起，按 eval 弱项调。
> **算法覆盖（§7.1 strict 决策的连带约束）**：因"算法推理算编程方向"，被标专属的算法/数据结构专家须有料可训 → 角度1 须含**算法类 bug**（复杂度/数据结构/边界，非纯框架 bug），角度2 须含**《算法导论》核心算法**蒸馏。否则专属面空训。

### 8.1 角度 1：代码仓库 + 核心库 + bug 修复 diff（50%）[✓证 数据集存在]

**数据源**：Multi-SWE-bench（github.com/multi-swe-bench/multi-swe-bench，**多语言含 Go**，真实 issue→修复带测试）、SWE-bench(hf)、SWE-Gym、OctoPack/CommitPackFT(2308.07124)；react 取 React/Next.js/Remix 的 bug-fix PR。

**处理管线**：
```
原始 commit/PR
  → 过滤(bug-fix 标签 + 有测试 + 单一关注点)
  → 去重(MinHash LSH, hf.co/blog/dedup, 阈值 ~0.85)
  → 许可合规(只收 MIT/Apache/BSD, 排除 GPL/无 license)
  → SFT 样本格式化(下)
  → 合成增强(2504.14757: 真实 diff 难度变体/反例)
```
**样本形态**（必须配测试，否则学不会"修对"）：
```
[问题]  issue 描述 + 失败测试输出
[上下文] 相关 react/go 源码片段
[修复]  最小 diff
[验证]  测试通过
[解释]  为什么这样修
```

### 8.2 角度 2：架构/算法/重构经典书 → 蒸馏（30%）[✓证 方法存在]

知识蒸馏（2402.13064 / 2312.02120 / 2405.03548）：书做**话题种子**（只用概念名，不喂正文），强模型(Opus 4.6)生成 react/go 架构/重构教学样本。

```
经典书概念清单(《重构》坏味道表 / 《设计模式》23模式 / Clean Arch 依赖规则 / 算法导论核心)
  → prompt 种子(只概念名)
  → Opus 生成: "坏味道 react/go 代码 → 识别 → 按X原则重构 → 解释"
  → 自动校验(生成代码能编译/测试过) → 入库
```
**[合规边界]**：种子=概念名与公开框架，输出=模型自生成代码，规避正文版权；Opus 蒸馏产出需符合其使用条款。

### 8.3 角度 3：执行诚实 agent 语料（20%）[✓证 faithful CoT 方法]

**命门**：小模型最易"假装跑了测试/编造工具结果"。方法：faithful CoT(2307.13702/2511.14102) + Reflexion(2303.11366)。
```
[任务] react/go 编码任务
[计划] plan-then-execute 显式步骤
[执行] 调用工具(读文件/跑测试) → 真实回填工具输出(禁凭空续写)
[反思] 测试失败 → 诚实记录 → 修正(Reflexion)
[完成] 测试真通过才声明完成
```
**负样本**（对比学习）：教模型识别"幻觉工具结果/假装通过"并拒绝。

### 8.4 配比与规模 [假设，T3 消融]

| 角度 | 占比 | 规模(起点) | 作用 |
|---|---|---|---|
| 1 bug修复diff | 50% | 1.5万~4万 | 核心：修对 bug |
| 2 书蒸馏重构 | 30% | 0.9万~2.4万 | 架构/重构质量 |
| 3 诚实agent | 20% | 0.6万~1.6万 | 执行可靠/不幻觉 |
| **合计** | 100% | **3万~8万** | |

**holdout**：从角度 1 留出真实 bug 修复**不入训练**，作 §10 eval。

---

## 9. 适配器注入 ds4（merge 回 2-bit 热专家）

**路 A（推荐，精度倒置默认）：merge → 重走量化热档**。零推理路径改动。
```bash
python merge_lora.py --base <HF base> --lora ./adapter --output ./V4-Flash-merged  # 回 bf16
# 仅对热专家重量化为 IQ2_XXS（复用量化文档命令链 + --experts-hot-mask）：
gguf-tools/deepseek4-quantize \
  --hf ./V4-Flash-merged --template gguf/ds4flash.gguf \
  --experts-hot-mask /tmp/mask-reactgo-hot.bin \
  --experts iq2_xxs --routed-w2 q2_k --imatrix /tmp/reactgo_router.dat \
  --out gguf/ds4-reactgo-hot-trained.gguf --overwrite
```
- 优点：ds4 mmap+Metal 完全不变；冷档仍 HF FP8 流式（§3.8）。缺点：改适配器要重量化热档（仅 ~9GiB，分钟级，非全模型）。

**路 B：运行时 FP16 旁路（可热替换适配器）**

| 改动点 | 文件 | 内容 |
|---|---|---|
| 适配器加载 | `ds4.c` | 读独立适配器张量（GGUF 附加段/单独文件），mmap FP16 buffer |
| Metal 图 | `ds4_metal.m` / `metal/moe.metal` | 热专家投影后加 `y += B·(A·x)` 旁路 kernel |
| residency | `ds4_metal.m` | 适配器 buffer 计入 12GiB 预算（rank 64 ~百 MiB 级，需实测撞红线否） |

- 优点：热替换、A/B 不同适配器。缺点：动 Metal 图、撞双机红线需测。

**校验门**：merge/注入后 react/go eval ≥ T1（不退步）；`--dump-logprobs` parity；`ds4_test --metal-kernels --server` 绿；路由 top-K 重合率达标（§4）。

---

## 10. eval 闭环（唯一裁判）

**先于一切训练建好。**

```
eval 集(100~200 题, react/go 各半):
  - 真实 bug 修复(Multi-SWE-bench holdout, 不入训练)
  - 架构/重构题(给坏味道代码, 评重构质量)
  - 算法/复杂度题(数据结构/复杂度/边界, 验证 strict 标专属的算法面真被训起来 §7.1)
  - 端到端 agent 任务(多步, 跑测试)

打分维度:
  - 功能正确(测试通过率)        ← 主指标
  - 架构质量(Opus-as-judge / 人评)
  - 执行诚实(工具幻觉率 = 编造工具结果次数/任务数)

基线: Opus 4.6 跑同一 eval 集存参考分; 每 milestone 对比逼近度
```

复用 ds4：`ds4-eval`（嵌入式 92 项，可扩）+ `ds4-eval q1..q4 --temp 0 --seed 1` 确定性门。

---

## 11. 训练里程碑 T0–T5

| T | 名称 | 动作 | 交付 | 校验门（必过） |
|---|---|---|---|---|
| **T0** | eval 基线 | 建 react/go eval 集 + Opus 4.6 参考分 | eval 集 + 基线分 | 可复现；参考分在案 |
| **T1** | 量化掉点 | 量化前/后 react/go eval → `Δ_quant`（含 FP8→IQ2） | Δ_quant 数值 | 单独量出 FP8→IQ2 段，不照搬通用掉点 |
| **T2** | 初始化 | ApiQ/LoftQ 在 2-bit 目标初始化 LoRA（挂 experts） | apiq-init 权重 | 可训练参数 ≫0.022%（C2 自检） |
| **T3** | 训练 | MoE-LoRA SFT（三类语料）；靶向 A/B；扫超参 | 适配器 + 配比/靶向结论 | eval ≥ T1；路由 top-K 重合率达标；靶向消融记录 |
| **T4** | 注入 | merge → 重量化热档 → ds4 装载（路A） | merged 小 GGUF | `--dump-logprobs` parity；`metal-kernels --server` 绿 |
| **T5** | 收敛 | 对标 Opus 4.6，诚实记录差距，定残余优化 | 逼近度报告 | 不粉饰差距；`Δ_domain` vs `Δ_quant` 在案 |

**每次合并 correctness 门**（CLAUDE.md）：`--dump-logprobs` parity + `ds4_test --metal-kernels --server`；routing/quant 改动加 `--logprob-vectors` + `ds4-eval q1..q4 --temp 0 --seed 1`。
**执行纪律**：每 patch/测量/scope/blocker 追加 `notes/execution-log.md`；总结性结论需用户批准再写日志。

---

## 12. 资源与可行性测算

| 项 | 结论 | 档 |
|---|---|---|
| LoRA 显存 | 81GiB 基座**无法在 16GB Mac 前向+反向** → 训练离线大内存机/云 GPU | [假设，最大未知] |
| 16GB Mac 角色 | **仅推理 + 适配器加载**（路A merge 零额外；路B 加 FP16 旁路 buffer，测撞 12GiB 红线否） | [假设] |
| 单 epoch 时长 | 3万~8万样本，单卡 A100/H100 级**数小时**（粗估） | [假设] |
| mlx-lm 本地直训 | 剪枝后 V4 真实峰值显存/时长**无数据**；且专家 LoRA bug → **不要假定 Mac 本地能训** | [✓证 bug / 假设 资源] |

**内存安全铁律**：任何载模型的训练/量化脚本须先证明 RSS 预算 + 看门狗安全；16GB Mac 不双载 base；重训产物回灌只动热档（~9GiB），不重写全模型。

---

## 13. 风险登记 + 开放问题

**风险**：

| R | 描述 | 缓解 |
|---|---|---|
| RT1 | FP8→IQ2 + 领域训练损失叠加放大（无人测） | T1 单独量 Δ_quant；T3 量 Δ_domain |
| RT2 | 工具调用场景能力更脆（正是 agent 编码） | eval 含 agent 任务 + 角度3 诚实语料 |
| RT3 | PyTorch ↔ ds4 C99/Metal 鸿沟（ApiQ/LoRA 全 PyTorch） | 全离线产出适配器再转 GGUF，ds4 只推理 |
| RT4 | 训练漂移路由 → 热集失配 → cold miss | 冻结 router（§4）；T3 路由重合率门 |
| RT5 | 语料→4.6 因果链未证 | eval 唯一裁判，T3 消融 |
| RT6 | 校准/语料偏差保护错专家 | imatrix + 语料必含 react/go；specialty baseline 须**纯非编程**（清单 + strict/broad 档 + 纯净度验证见 §7.1） |

**开放问题（无数据，需原型/实测）**：
1. FP8 base → IQ2 → 领域训练，react/go 编码联合掉点/净增益各多少？`Δ_domain > Δ_quant` 能否成立到 4.6？
2. 16GB Mac / 单卡 / 双机分布式对剪枝后 V4 做 MoE-LoRA 的真实峰值显存与单 epoch 时长？
3. specialty 靶向（rank 分层 / 采样加权）是否优于热集均匀 rank？
4. 三类语料各需多大规模/配比才能把损失顶回接近 Opus 4.6？无先例消融。
5. 冻结 router 是否足以保热集稳定，还是需要 EAQuant routing-consistency 对齐？

---

## 14. 引用（经对抗验证，[✗驳] 标注禁用）

**微调 / 2-bit 稳定**
- LoRA 放置：thinkingmachines.ai/blog/lora · Biderman 2405.09673 · mlx-lm issue 571
- 2-bit 初始化：LoftQ 2310.08659 · ApiQ 2402.05147 + github.com/baohaoliao/apiq
- partial-update 警告：2504.13932 · MoE PTQ routing 对齐：EAQuant 2506.13329

**语料**
- SWE-bench(hf) · Multi-SWE-bench(含 Go) github.com/multi-swe-bench/multi-swe-bench · SWE-Gym
- OctoPack/CommitPackFT 2308.07124 · 合成增强 2504.14757 · 去重 hf.co/blog/dedup

**蒸馏**
- 2402.13064 · 2312.02120 · 2405.03548 · NVIDIA license-compliant synthetic data pipeline

**prompt/agent 诚实**
- Reflexion 2303.11366 · faithful reasoning 2307.13702 · 2511.14102 · 2508.00083 · 2503.08679

**[✗驳] 禁用**
- "LoRA==全量微调"(0-3) · "每专家单独 LoRA rank=总/激活"(0-3) · "mlx-lm 完全没挂专家 LoRA"(1-2，仅 0.022% 事实确认) · "QLoRA 零初始化 2-bit 直接可用"(被 LoftQ fails-below-3bit 驳)

---

> 上游量化/裁剪/混合格式/动态模式见 `react-go-opus46-design.md`；研究证据全集见 `search.md`；专家激活与 specialty 工具见 `gguf-tools/router_norms_from_imatrix.py`。
