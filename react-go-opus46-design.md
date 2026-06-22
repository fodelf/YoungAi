# 双机 16G Mac 量化 DeepSeek-V4-Flash 逼近 Opus 4.6（react/go）—— 详细设计方案与执行步骤

> 版本：v2 详细稿（2026-06-19）。基于 deep-research 多源对抗验证（6 路搜索 / 28 来源 / 132 论断 / 25 条对抗验证 / 19 确认 6 驳倒）。
> 目标：双机 16GB Mac（Thunderbolt 直连、每机 ≤12GiB、专家 SSD 流式）上，量化版 DeepSeek-V4-Flash 在 **react / go** 编码场景逼近 **Claude Opus 4.6**。
> 交付物：可直接接进 ds4(DwarfStar, 纯 C99 + Metal + mmap GGUF + 分布式层切分 + 磁盘 KV cache) 的工程方案，含每一步的命令、参数、代码改动点、校验门。

---

## 目录

- [0. 诚实边界与证据分级](#0-诚实边界与证据分级)
- [1. 研究综述（落地版）](#1-研究综述落地版)
- [2. 总体架构：三段式补偿管线](#2-总体架构三段式补偿管线)
- [3. 内存预算与可行性测算](#3-内存预算与可行性测算)
- [4. P0 — 压缩到双机能装下（主线 A）](#4-p0--压缩到双机能装下主线-a)
- [5. P1 — 本地微调补偿（主线 B）](#5-p1--本地微调补偿主线-b)
- [6. P2 — 三类语料工程（语料角度 1/2/3）](#6-p2--三类语料工程语料角度-123)
- [7. P3 — prompt/agent 诚实执行体系](#7-p3--promptagent-诚实执行体系)
- [8. react/go eval 闭环（裁判）](#8-reactgo-eval-闭环裁判)
- [9. 端到端执行路线图 M0–M6](#9-端到端执行路线图-m0m6)
- [10. 风险登记 + 开放问题](#10-风险登记--开放问题)
- [11. 引用](#11-引用)

---

## 0. 诚实边界与证据分级

按"执行路径保持诚实"铁律，全文每条都标注证据档位：

- **[✓证]** 被对抗验证确认（3-0 或 2-1）。
- **[假设]** 设计推断，无验证 claim 直接支撑，**必须用 eval 闭环证伪**。
- **[✗驳]** 被驳倒，**禁止作为依据**。

**三条最该记住的诚实警告**：

1. **REAP 50% 近无损是 Cerebras 厂商自报、无独立复现**，且**工具调用场景掉 ~15%**（Kimi-K2 BFCL 0.666→0.564）——正是我们的 agent 编码场景。→ 剪枝率由 react/go 自测定，**不照搬"近无损"**。
2. **REAP + 2-bit 量化损失叠加无人测过**（两者都动专家）→ M1 必须实测联合掉点，不能假定线性可加。
3. **"语料 + prompt 工程 → Opus 4.6 编码水平"这条因果链没有任何被验证 claim 量化支撑，是本设计的核心假设。** eval 闭环是唯一裁判。

**[✗驳] 清单（禁用）**：① MoQE "2-bit 纯专家免训练即可靠"(0-3)；② QuIP# 2-bit Llama2-70B 困惑度数值(0-3)；③ QuIP# 两阶段微调"必需且全位宽增益"(1-2)；④ "LoRA 在 SFT 等同全量微调"(0-3)；⑤ "MoE 每专家单独 LoRA、rank=总rank/激活专家数"(0-3)；⑥ "mlx-lm 完全没给专家挂 LoRA"(1-2，仅 *0.022% 可训练参数* 事实确认)。

---

## 1. 研究综述（落地版）

8 条高置信结论，每条带证据与对落地的影响：

| # | 结论 | 档 | 对落地的影响 |
|---|---|---|---|
| F1 | **REAP**(2510.13999) 一次性 MoE 剪枝，显著性=路由门控值×专家激活范数，剪 50% 专家代码生成近无损（Qwen3-Coder-480B 编码 0.660→0.644），优于专家合并 | ✓证 | P0 第一支柱：量化前砍专家数 |
| F2 | REAP 官方实现 Apache-2.0，`prune.py` 入口 + 逐层 block-wise 校准（单 GPU 剪超大模型），已发布 DeepSeek-V3.2-REAP-345B/508B 等预剪枝权重 | ✓证 | 可直接复用，但 V4-Flash 需适配脚本 |
| F3 | ds4 非对称量化（gate/up IQ2_XXS、down Q2_K、shared/attn/output Q8_0）理论依据=MoE 专家比稠密 FFN 对低比特远更鲁棒(MoQE 2310.02410) | ✓证 | P0 第二支柱：现状即此，保留 |
| F4 | llama.cpp `quantize` 原生支持全部 2-bit 类型 + imatrix + include/exclude-weights 逐张量控制；IQ2_XXS=2.38bpw，Q2_K=3.16bpw(Llama-3.1-8B) | ✓证 | 工具链现成，REAP 后须重收 imatrix |
| F5 | **LoRA 必须挂 MLP/MoE 专家层**，attn-only 显著欠拟合；mlx-lm 对 MoE 专家投影有 bug 只训 0.022% 参数 | ✓证 | P1 硬约束：适配器必须命中 mlp.experts |
| F6 | **2-bit 基座标准 QLoRA 零初始化适配器在 3-bit 以下不收敛**（LoftQ: fails below 3-bit），正落 ds4 区间 | ✓证 | P1 不能直接 QLoRA |
| F7 | 解法=**ApiQ / LoftQ** 替代 QLoRA。ApiQ 并发初始化 LoRA+量化、保激活精度、抑误差传播，专为 2-bit，持续优于 QLoRA/LoftQ | ✓证 | P1 用 ApiQ（首选）或 LoftQ |
| F8 | **EAQuant**(2506.13329) MoE 专属 PTQ：平滑聚合抑离群、路由一致性对齐、校准均衡，亚 3-bit(W2A4) 可行 | ✓证 | P0 量化前 routing 对齐借鉴 |

---

## 2. 总体架构：三段式补偿管线

```
DeepSeek-V4-Flash-Base (HF safetensors, FP16/BF16)
   │
   ├─[P0 压缩]──────────────────────────────────────────────────────┐
   │   1. REAP 剪枝 prune.py（砍 R% 专家，逐层校准）                  │ 离线
   │   2. 用 react/go+通用混合语料重收 imatrix                       │ PyTorch/
   │   3. 非对称量化(gate/up IQ2_XXS, down Q2_K, 其余 Q8_0)+EAQ对齐  │ llama.cpp
   │   → 剪枝量化 GGUF，体积 ≤ 双机 24GiB 预算                       │
   └─────────────────────────────────────────────────────────────────┘
   │
   ├─[P1 微调补偿]───────────────────────────────────────────────────┐
   │   4. ApiQ 初始化 → MoE-LoRA(挂 experts.gate/up/down)            │ 离线
   │   5. 三类语料 SFT 训练适配器                                     │ 大内存机/云
   │   6a. merge 回 FP16 base → 重走 P0  (路A，推荐先试)             │
   │   6b. 适配器 FP16 旁路注入 ds4 Metal 图  (路B，可热替换)        │
   └─────────────────────────────────────────────────────────────────┘
   │
   ├─[P2 语料]────────────────────────────────────────────────────────
   │   角度1: bug修复diff(Multi-SWE-bench含Go+React PR)
   │   角度2: 经典书蒸馏(Opus生成react/go架构重构教学样本)
   │   角度3: 执行诚实agent语料(faithful CoT + 工具真实回填)
   │
   ├─[P3 prompt/agent]── 接 ds4_agent.c / ds4_server.c
   │
   └─[裁判] react/go eval 闭环 ── 对标 Opus 4.6，每 milestone 必过门
```

**补偿逻辑**：P0 为装下必然掉点 → P1 本地微调顶回一部分 → P2/P3 语料+推理期工程顶回剩余。每段挂可测门，禁止"假定生效"。

---

## 3. 内存预算与可行性测算

### 3.1 现状体量

- 现 GGUF：`DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`，约 **81 GiB**（路由专家 2-bit，其余 Q8_0）。
- 双机预算：每机 ≤12GiB → **合计 ≤24GiB RAM**；专家从 SSD 按需流式（ds4 已有 `DS4_METAL_EXPERT_*` 流式路径）。
- 物理瓶颈（CLAUDE.md 已定）：路由专家 SSD IO ~1.70 GiB/token，decode 受 SSD 带宽限。

### 3.2 REAP 剪枝后体量测算 [假设，需实测确认]

设路由专家原占 GGUF 约 72.6 GiB（CLAUDE.md 数据），剪 R% 专家近似线性减专家体积：

| 剪枝率 R | 专家体积估算 | 总 GGUF 估算 | 每 token 专家 IO | react/go 风险 |
|---|---|---|---|---|
| 0%（现状） | 72.6 GiB | 81 GiB | ~1.70 GiB | 基线 |
| 25%（保守） | ~54.5 GiB | ~63 GiB | ~1.28 GiB | 低（建议起点） |
| 37.5% | ~45.4 GiB | ~54 GiB | ~1.06 GiB | 低~中 |
| 50%（激进） | ~36.3 GiB | ~45 GiB | ~0.85 GiB | 中（工具调用掉点风险） |

> **关键**：剪枝主要价值是**降 SSD 流式量 → 提 decode t/s**，而非直接装进 RAM（45GiB 仍 > 24GiB）。剪 50% 把每 token 专家 IO 从 ~1.70 降到 ~0.85 GiB，理论 decode 接近翻倍。**但能力掉点是代价，M1 必须 react/go eval 实测拐点，选能力可接受的最大 R。**

### 3.3 本地微调内存测算 [假设，最大未知]

LoRA 训练显存 ≈ 基座前向激活 + 适配器梯度/优化器态 + 激活检查点。81GiB 基座**无法在 16GB Mac 上前向+反向**。结论：

- **P1 训练离线在大内存机/云 GPU 产出适配器**（单次，几小时~几天级）。
- 16GB Mac **只承担推理 + 适配器加载**（路 A merge 后零额外开销；路 B 加 FP16 旁路 buffer，需测是否撞 12GiB 红线）。
- mlx-lm/ApiQ 在 16GB Mac 直训剪枝后 V4 的真实显存/时长**无数据**（开放问题 2）——不要假定 Mac 本地能训完整模型。

---

## 3.5 立即任务：20GB 常驻 react/go GGUF 裁剪配方（硬目标 30 t/s）

> 本节是当前最优先的可执行交付：从 240GB 原始模型裁出一个 **react/go 专家常驻 20GB、其余冷启动、react/go 场景 ≥30 t/s** 的 GGUF。
> 全部钉在仓库现有工具（`gguf-tools/`）和已实测的模型结构上。

### 3.5.0 模型结构（实测，非估算）

- **43 MoE 层 × 256 路由专家 = 11,008 专家**（来源：`gguf-tools/make_firstk_mask.py --layers 43 --n-expert 256`、`shrink_gguf.py` "256 routed experts"）。
- 2-bit 下 **~6.75 MiB/专家**（gate+up+down，= 72.6 GiB / 11,008）。
- backbone（attn/shared/output/router/embed，Q8_0）**~8.4 GiB**（= 81 − 72.6）。
- 现有裁剪机制（**已实现，直接复用**）：
  - `make_expert_mask.py`：从画像建 DSXM mask，留 top-K/层；禁用专家 router logit 强制 -inf，**永不进 RAM**；支持 `--keep-list` 每层不同 K。
  - `shrink_gguf.py`：物理删除被禁专家，重写 `ffn_{gate,up,down}_exps.weight` 末维=保留数，写元数据 `ds4.expert_keep_map.{kept_counts,original_ids}`；64MB 流式不爆内存（16GB Mac 安全）。
  - `ds4.c:load_expert_keep_map()`（行 1566+）：运行时读 keep_map，`expert_orig_to_compact` 做 token→保留槽映射。**注意 memory「k16 GGUF stale/broken」：此 routing kernel keep-map 屏蔽曾有未实现 bug，落地前先验证 k16 单机数字正常。**

### 3.5.1 30 t/s 的物理前提（诚实，先讲清楚）

- 全模型每 token 路由专家 IO ≈ **1.70 GiB**（CLAUDE.md）。SSD ~5 GB/s → **天花板 ~3 t/s**（与 CB-floor memory 一致）。
- 30 t/s ⟺ **33 ms/token**。专家 IO 预算扣掉 compute 后只剩 ~7-10 ms → 每 token 从 SSD 读的专家须 ≤ **~50 MiB**（≈ 全量的 3%）。
- **唯一出路 = react/go 实际激活的专家常驻 RAM，热路径近零 SSD IO**（memory：X9 是唯一绕过 SSD I/O 的路径，X9 GO）。
- **诚实判定法**：30 t/s 能否达到，取决于 react/go 路由是否真的集中到 ≤32 专家/层。这由 §3.5.3 画像**实测**，不预设。x9 通用 top-16 覆盖 95.6% → react/go 专属应更高，但若 32/层覆盖不到 ~97%，每 token 跨 43 层会累积多次冷 miss → 需加大 K 或对冷尾 REAP 永久剪除（见 §3.5.4 策略 C）。

### 3.5.2 20GB 常驻预算分配

| 项 | 字节 | 说明 |
|---|---|---|
| backbone (Q8_0) | ~8.4 GiB | 必常驻 |
| KV (react/go 32-64K, 压缩) | ~1.5 GiB | 编程上下文 |
| working / Metal | ~1.0 GiB | |
| **react/go 热专家(2-bit)** | **~9.1 GiB** | → **~32 专家/层** 常驻 |

精度可调（最大保 react/go 能力）：

| 热专家精度 | 单专家 | 9.1GiB 容纳/层 | react/go 能力 | 取舍 |
|---|---|---|---|---|
| IQ2_XXS (2.06bpw) | 6.75 MiB | ~32/层 | 现状 2-bit | 覆盖最广 |
| Q4_K (~4.5bpw) | ~14.7 MiB | ~14/层 | 更高 | 覆盖窄、易冷 miss |
| **混合**: top-8 Q4 + 16 IQ2/层 | 9.5GiB | ~24/层 | 热专家高精度 | **推荐**：常用专家高质，兼顾覆盖 |

### 3.5.3 步骤 1 — react/go 专属路由画像（关键新步骤）

现有 `router_norms` 是**通用 L2 norm**；要 react/go 专属，须用 **react/go 语料**跑 ds4 收 per-layer per-expert 激活统计：

> **工具现状（实测）**：ds4 有 `--imatrix-out`（收 per-expert 重要性），但**没有** `--router-norms-out`；`make_expert_mask.py` 依赖的 `router_norms.json` 的**生产者在仓库不存在**。好在 react/go 的 imatrix `.dat` 本身就含 per-expert 激活重要性 → **离线派生即可，无需改 ds4、无需额外模型跑**。

```bash
# (a) 建 react/go 校准语料（真实 .tsx/.jsx/.go + 编码任务 prompt）
cat calib_react/*.tsx calib_react/*.jsx calib_go/*.go \
    calib_reactgo_tasks/*.txt > /tmp/reactgo_calib.txt

# (b) 跑 ds4 收 react/go imatrix（= 现有文档流程；唯一的模型跑）
#     ⚠ 加载模型 = 离线 + 内存安全证明 + 逐次授权（见 §3.5.8）
./ds4 -m gguf/ds4flash.gguf \
  --imatrix-dataset /tmp/reactgo_calib.txt \
  --imatrix-out /tmp/reactgo_router.dat \
  --ctx 32768

# (c) 离线从 react/go imatrix 派生 per-layer per-expert rank（新小脚本, 内存安全, 不载模型）
python3 gguf-tools/router_norms_from_imatrix.py \   # 需新写: 读 .dat 每专家 L2 → 每层降序 rank
  /tmp/reactgo_router.dat --n-layers 43 --n-expert 256 \
  --out /tmp/router_norms_reactgo.json
```
> 产出 `router_norms_reactgo.json` = 每层 256 专家按 react/go 激活降序 rank。**这是 react/go「尽可能保留」的依据。** 唯一缺的代码是离线派生脚本 `router_norms_from_imatrix.py`（可立即写，不载模型）。

### 3.5.4 步骤 2 — 三种裁剪策略，选 C

| 策略 | 机制 | react/go 常驻 | 其他冷启动 | 30 t/s | 评 |
|---|---|---|---|---|---|
| A 纯 shrink | 物理删非 react/go 专家 | ✓ 全常驻 | ✗ 删了无法冷启动 | ✓ | 违背"其他可冷启动" |
| B 纯 mask | 禁用非 react/go（-inf） | ✓ | ✗ 永不选 | ✓ | 同上 |
| **C 两层(推荐)** | **REAP 剪 react/go 死专家 + 热专家常驻 pin + 暖专家 SSD 冷启动** | ✓ pin 常驻 | ✓ 暖专家流式 | ✓ 热路径近零 IO | **满足全部三约束** |

**策略 C 分层**（每层 256 专家）：
- **热档 ~32/层**：react/go 画像 top-rank，混合精度（§3.5.2），**靠 OS page cache LRU 自然常驻**——见下方「常驻机制」。
- **暖档 ~32-96/层**：react/go 偶尔用，**留在 GGUF，SSD 冷启动流式**（ds4 已有 `DS4_METAL_EXPERT_*` 路径）。
- **死档 余下**：react/go 画像近零激活，**REAP 永久剪除**（缩文件、降装载）。

**常驻机制（关键修正，遵循实测）**：**不要 mlock/MTLResidencySet 显式 pin 热专家。** memory「内存动态用满12G」实测：显式 RAM 专家缓存（hard_copy/mlock）**饿死 OS page cache，净变慢 3.84→1.84 t/s**，decode 贴 gather 物理地板。正确做法：
- 热档专家**每 token 都被访问** → OS page cache LRU **自然把它们保持常驻**，无需 pin。
- 我们要做的只是 ① **让 react/go 热档工作集 ≤ ~9 GiB**（靠死档 REAP + 暖档不常驻），使其能被 page cache 完整容纳；② **启动时顺序预读热档**预热 page cache（避免首 token 冷启动抖动）；③ 热档在 GGUF 内**连续布局**（减少缺页/碎片）。
- 这样"react/go 常驻 + 其他冷启动 + 30 t/s"全部由现有 page-cache + 流式路径**涌现**，几乎不加新机制，也不重蹈 mlock 的覆辙。

### 3.5.5 步骤 3 — 命令链（端到端）

```bash
# 1. react/go 热专家 keep-list（每层 K=react/go 覆盖≥98%所需，从画像算）
python3 gguf-tools/make_expert_mask.py /tmp/router_norms_reactgo.json \
  --keep-list 32,32,28,40,... \      # 每层一个 K，画像定（非均匀）
  --out /tmp/mask-reactgo-hot.bin

# 2. (策略C死档) REAP 离线剪 react/go 近零激活专家 → 缩基座（见 §4.1）
#    或先用 mask 留 hot+warm(如 96/层), 物理 shrink 死档:
python3 gguf-tools/shrink_gguf.py \
  --in gguf/ds4flash.gguf \
  --mask /tmp/mask-reactgo-keep96.bin \   # 留 hot+warm 96/层, 删死档 160/层
  --out gguf/ds4flash-reactgo-shrunk.gguf

# 3. 混合精度量化: 热档 Q4, 暖档 IQ2 (deepseek4-quantize 或 mixed/splice 工具)
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template gguf/ds4flash-reactgo-shrunk.gguf \
  --experts iq2_xxs --routed-w2 q2_k \
  --imatrix /tmp/reactgo_router.dat \
  --out gguf/ds4flash-reactgo-20g.gguf
# 热档升 Q4: 用 gguf-tools/mixed/splice_mixed_expert_layers_gguf.py 把 top-8/层 splice 成 Q4_K

# 4. 写热档区间元数据（ds4.expert_hotset.*：每层热档 rank 顺序 + 文件偏移区间）
#    供 ds4 启动顺序预读「仅热档区」预热 page cache；不 pin、不 mlock
python3 gguf-tools/make_hotset_manifest.py \   # 需新写, 仿 make_expert_mask 读同一画像
  /tmp/router_norms_reactgo.json --hot-top-k 32 \
  --embed-into gguf/ds4flash-reactgo-20g.gguf
```

### 3.5.6 代码改动点（ds4 引擎侧）

| 改动 | 文件 | 内容 | 风险 |
|---|---|---|---|
| 热档连续布局 | `gguf-tools/shrink_gguf.py` | 按 react/go 画像 rank 把热档专家在 GGUF 内**连续排布**（暖/冷在后），减少缺页碎片 | 低：仅改写出顺序 |
| 启动预读热档 | `ds4.c` / `ds4_metal.m` | 装载时**顺序预读热档区**预热 page cache（madvise WILLNEED 仅热档区，**不是整 82GiB**——memory：整盘 WILLNEED 会亚秒灌爆看门狗） | **中**：预读范围必须精确限热档 |
| 冷启动流式 | `ds4_metal.m` `DS4_METAL_EXPERT_*` | 暖/冷档走现有按需 pread/fault 路径（保持不动） | 复用现成 |
| **不做** mlock/显式 pin | — | memory 实测显式 RAM 缓存饿死 page cache 反变慢，**禁用** | — |
| (策略C若用 keep_map) 修 routing | `metal/moe.metal` | 验证/修 token→保留槽屏蔽（memory：k16 曾未实现） | **中**：先单机 k16 复现确认 |

### 3.5.7 步骤 4 — 30 t/s 校验门

```bash
# react/go 冷 miss 率（每 token 跨 43 层落到暖/冷档的次数）—— 决定能否 30 t/s
./ds4 -m gguf/ds4flash-reactgo-20g.gguf -p "<react/go 编码任务>" \
  --trace /tmp/expert_trace.txt          # 统计 resident-hit vs SSD-fault
# 瞬时 decode t/s（react/go 提示）
./ds4-bench -m gguf/ds4flash-reactgo-20g.gguf \
  --prompt-file calib_reactgo_tasks/edit.txt \
  --ctx-start 8192 --ctx-max 65536 --gen-tokens 128 --csv /tmp/reactgo-speed.csv
```
**不达 30 t/s 的处置（禁兜底，调最优）**：① 加热档 K（占更多 20GB）；② 死档 REAP 更狠（暖档→死档，杜绝冷 miss）；③ 热档降回纯 IQ2 换更广覆盖。**用画像数据驱动，不试错回退。**

### 3.5.8 安全闸（硬约束，铁律）

- 步骤 1 画像跑 + 步骤 2/3 的 240GB 量化 = **离线 + RSS 预算/看门狗内存安全证明 + 逐次授权**才可跑（不在本会话自动执行）。
- 16GB Mac 上**只跑** memory-safe 的离线 python（shrink/mask/manifest 都 64MB 流式）；240GB HF→GGUF 重量化在大盘/大内存机做。
- 单机不双载 base；改完 ds4 引擎后另一台机器同步重编（共享 `CORE_OBJS`）。

---

## 3.6 选定主方案：反常识「精度倒置」（热=2bit常驻 / 冷=原样冷启动）

> 用户拍板的主方案（2026-06-20）。与 §3.5 通用分析共用残驻数学，但**精度分配反转**。

### 3.6.1 方案核心：与 §3.5 的差异

| 维度 | §3.5 通用版 | **§3.6 倒置版（选定）** |
|---|---|---|
| 编程(热)专家精度 | 混合 Q4/Q2 | **IQ2_XXS 2-bit**（求最小→求常驻） |
| 其他(冷)专家精度 | 已 2-bit | **原样全精度 FP16（不动）** |
| 重量化范围 | 全模型 240GB | **仅编程热专家**（省绝大部分离线工作） |
| 通用能力质量 | 2-bit | **全精度无损** |
| 编程能力 | 2-bit 现状 | **2-bit + 领域训练买回 → 目标 4.6** |

**一句话**：从 240GB base 出发，**只把 react/go 画像出的热专家（~32/层）量化到 2-bit 使其常驻 20GB**；其余 ~224/层专家**原样 FP16 留在文件、SSD 冷启动流式**；再对 2-bit 热专家做领域训练，把 react/go 顶到 Opus 4.6。

### 3.6.2 为什么"反常识"却成立

常规直觉=热专家留高精度（误差累积）。本方案反过来=**热专家压 2-bit 换常驻（=30t/s），质量损失靠领域训练买回**。三重支撑：
- **30 t/s**：常驻消除 SSD IO（§3.5.1，热路径近零 IO → 33ms/token 够用）。
- **MoE 专家对低比特鲁棒**（研究 F3 / MoQE 2310.02410）。
- **2-bit 上可训回质量**（研究 F7 / ApiQ 2402.05147）。
- 红利：不重量化 240GB（只动热专家）；通用能力全精度无损；base 官方支持定制训练。

### 3.6.3 训练顺序（关键，踩错即废）

```
量化热专家→2-bit  →  ApiQ/LoftQ 初始化  →  领域 QAT/LoRA 训练(三点语料 §6)  →  react/go 4.6
                       ▲ 必须, 否则不收敛(F6)        ▲ 在 2-bit 空间内训, 让 2-bit 热专家学会 react/go
```
- ❌ **禁**：2-bit 后直接标准 QLoRA → 3-bit 以下不收敛（研究 F6 铁证）。
- ✓ ApiQ 专为 2-bit 设计，并发初始化适配器+量化、抑误差传播（F7），训练在量化空间进行 = QAT 思路。
- LoRA 必挂 `mlp.experts` 的热专家槽（F5）；可训练参数须 ≫0.022%（mlx-lm bug 阈值）。

### 3.6.4 "4.5→4.6" 的诚实判定 [假设]

- DeepSeek 官方称通用编码 ~4.5；**窄域（react/go）专化可超通用**（领域特化现象，方向有论文支撑），但：
- **净值 = 4.5 − 2bit损失 + 领域训练增益**；只有增益>损失才 >4.5。窄到只做 react/go，增益有机会压过 2-bit 损失到 4.6——**有可能，非保证**。
- eval 闭环（§8）是唯一裁判；"4.6" 是**可证伪目标**，不是预设结论。

### 3.6.5 本方案专属风险（在 §10 之外新增）

| R | 描述 | 处置 |
|---|---|---|
| RA | "编程专家"非天然可分，由 react/go 画像定义；编程中偶选冷专家=cold miss=掉速 | 30t/s 取决于实测覆盖率(§3.5.1)；不足则扩热集 K 或对冷尾 REAP |
| RB | **冷专家保 FP16 → 冷启动字节×8 → 通用 ~0.3-0.4 t/s（非 2 t/s）** | **待用户定**：通用宁慢求质（保 FP16）vs 通用也要 ~2t/s（冷专家降 Q8/Q4，违"原样不动"） |
| RC | 量化-训练顺序错=不收敛 | 强制 ApiQ/LoftQ 初始化（§3.6.3） |
| RD | 16G Mac 训不动完整模型 | 训练离线大内存机/云，Mac 只推理（§3.3） |

### 3.6.6 落地命令链（基于 §3.5 工具，精度倒置）

```bash
# 1. react/go 画像 → 热专家 rank（同 §3.5.3：imatrix → 派生 router_norms）
# 2. 热专家 keep-list（每层热档 ~32，画像定）
python3 gguf-tools/make_expert_mask.py /tmp/router_norms_reactgo.json \
  --keep-list 32,32,28,... --out /tmp/mask-prog-hot.bin

# 3. 倒置量化：仅热专家 IQ2_XXS，其余 FP16 原样（deepseek4-quantize 逐族覆盖）
#    --experts-hot-mask 让量化器只压 mask 内热专家，mask 外专家走 f16 passthrough（需小补丁）
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template gguf/<240G-base-template>.gguf \
  --experts-hot-mask /tmp/mask-prog-hot.bin \   # 新参数：热档 IQ2_XXS
  --experts-cold f16 \                           # 冷档原样 FP16（或 RB 决定降 Q8）
  --imatrix /tmp/reactgo_router.dat \
  --out gguf/ds4-prog2bit-coldf16.gguf
# 4. 热档连续布局 + 启动预读（§3.5.6，靠 page cache LRU，不 mlock）
# 5. 训练：ApiQ 初始化 → 三点语料 QAT/LoRA → merge 回热专家 → 重走步骤3
```
> 新代码缺口：`deepseek4-quantize` 加 `--experts-hot-mask`/`--experts-cold` 逐专家精度分流（现有 `--experts` 是全局一档）。这是本方案唯一的量化器改动。

---

## 3.7 动态模式切换路由（编程/日常互斥，选定增强）

> 用户增强（2026-06-20）：基于"编程与日常不并行"假设，做双模式残驻切换。**它顺手解决了 §3.6 的雷 2、雷 3/RB。**

### 3.7.1 核心

- **假设**：编程与日常**不并行**（单用户交互；ds4 server 本就单图 worker 串行 → 天然一时一模式）。✓ 合理。
- **Mode P（编程，默认）**：router **屏蔽到编程热集**（2-bit、训练过、~32/层，常驻）→ **零 cold miss → 30 t/s**。
- **Mode G（日常）**：router 解除屏蔽，全 256/层；冷专家（FP16 原样）走现有 LRU/page-cache 流式 → **全质量，~2 t/s（次要慢档，by design）**。
- **切换**：一次性 warmup/evict 损耗，跨会话摊薄。

### 3.7.2 它解决了 §3.6 的两个雷

| 原雷 | 动态切换如何解 |
|---|---|
| **雷 2** 编程 cold miss 杀 30t/s | Mode P 屏蔽路由到常驻热集 → 物理不可能 miss → 30t/s = "屏蔽+常驻"双保险，不再只赌覆盖率 |
| **雷 3/RB** 冷专家精度纠结 | 日常=刻意的次要慢档 → 冷专家保 FP16 **原样不动反而合理**（次要模式宁慢求质）→ **用户原始直觉成立，RB 不用再纠结** |

### 3.7.3 硬前置（地基，必须先修）

Mode P 的"屏蔽路由" = 现有 `keep_map`（router 强制 -inf）。但 **memory「k16 GGUF stale/broken」：此 routing kernel keep-map 屏蔽是「未实现的代码 bug」**。
→ **整个动态方案前置 = 先修复 keep-map 屏蔽内核**（`metal/moe.metal` + `ds4.c` 路由），**先单机 k16 复现确认数字正常**，否则 Mode P 出"数字汤"。

### 3.7.4 复用现有机制

- 全模型（256/层）留 GGUF（Mode G 要全路由）+ 另带编程 `keep_map`；mode flag 决定是否套用屏蔽。
- 编程热集 = §3.5.3 react/go 画像产出，2-bit + 领域训练（§3.6.3）。

### 3.7.5 切换代价与机制

| 切换 | 动作 | 代价 |
|---|---|---|
| **G→P** | 精确 `madvise(WILLNEED)` 编程热档区（~9GiB）预热 page cache | SSD ~3-5GB/s → **~2-3s 一次性**（⚠ **precise range，不是整盘**——memory：整盘 WILLNEED 亚秒灌爆看门狗） |
| **P→G** | 解除屏蔽；通用工作集 LRU 懒填 | 首几 token 慢，无显式预读 |
| KV | 编程/日常各自 session KV | ds4 磁盘 KV cache 可 checkpoint/restore，切换不互污 |

### 3.7.6 模式触发

- **显式（推荐）**：`/mode prog|general` 命令 / server endpoint / agent 设定。
- 自动（可选）：轻量 prompt 分类器判 react/go 编码任务。**不建议**靠 router 运行时统计自适应（脆弱、抖动）。

### 3.7.7 代码改动点

| 改动 | 文件 | 内容 | 风险 |
|---|---|---|---|
| **修 keep-map 屏蔽内核** | `metal/moe.metal` / `ds4.c` | 实现/修复 token→保留槽屏蔽（地基，§3.7.3） | **高**：先单机 k16 复现确认 |
| mode flag + 切换 | `ds4.c` session 状态 | Mode P/G 标志；切换时套/解 keep_map 屏蔽 | 中 |
| 精确预热 | `ds4_metal.m` | G→P 仅对热档区 `madvise WILLNEED`（限范围） | 中：范围越界=灌爆看门狗 |
| `/mode` 命令 | `ds4_cli.c` / `ds4_agent.c` / `ds4_server.c` | 显式切换入口 | 低 |

---

## 3.8 混合格式加载：量化热档(GGUF) + HF 原文件冷档（省盘，选定增强）

> 用户增强（2026-06-20）：冷档不转 GGUF，直读已下载的 HF safetensors，省 ~210GB 盘。

### 3.8.1 动机：不复制冷档 FP16 大头

| 方案 | 占盘 | 说明 |
|---|---|---|
| §3.6 全写 GGUF | HF 240G + GGUF ~230G ≈ **470G** | 冷档 FP16 在 GGUF 里**重复**一份 |
| **§3.8 混合（选定）** | HF 240G + 小量化件 ~20G ≈ **260G** | **省 ~210G**；冷档直读 HF |

- **"不一定要 GGUF" 对冷档成立**：直读 HF、不转 GGUF 是省盘关键。热档仍用 GGUF（ds4 原生最省事）。

### 3.8.2 双机布局（层切分，每机 ~10G 量化常驻）

| 每机（约半层） | 来源/精度 | 字节 |
|---|---|---|
| backbone 切片 | 量化 Q8 | ~4.2 GiB |
| 编程热专家切片 | 量化 IQ2_XXS（训练过） | ~4.5 GiB |
| **= 量化常驻** | **小 GGUF** | **~8.7 GiB ≈ "10G"** ✓ |
| 冷专家切片 | 本机 HF safetensors FP16，按需流式 | 不常驻 |

- 双机各存约半个 HF（~120G），**无重复**；和双机 16G、≤12G 预算自洽。

### 3.8.3 必须的新代码：safetensors 冷档流式（最大新块）

- **ds4 运行时当前只读 GGUF**（CLAUDE.md：purpose-built for GGUF，自写 loader）。冷档直读 HF 需新增：
  1. 解析 safetensors header（每张量 offset/len）；
  2. 按 expert id → ST 张量映射；
  3. mmap 本机 ST 分片，按需 fault FP16 专家（复用 `DS4_METAL_EXPERT_*` 流式风格）；
  4. MoE gather kernel 加 "FP16-from-ST" 源分支（FP16 专家 matmul 在 Q4 family 已有先例）。
- **有底子**：`gguf-tools/deepseek4-quantize` 已离线解析 safetensors，header/offset 逻辑可直接参考。

### 3.8.4 小量化件构成

- 仅 backbone（Q8）+ 编程热专家（IQ2_XXS，训练后）→ 小 GGUF（双机合计 ~20G）。
- 元数据记冷专家"外部源 = HF safetensors 路径 + per-expert offset/stride"，供 Mode G 路由直读。

### 3.8.5 前置闸 + 风险

| 项 | 说明 |
|---|---|
| **HF 下载未完成** | 一切量化/画像等下载完整 |
| **per-expert 可寻址性** | 下载完读 HF index 确认：experts 独立张量→直接 offset；堆叠成大张量→按 stride 切 |
| FP16 冷档流式 | 字节×8 → Mode G 更慢（可接受，次要模式 §3.7） |
| 双机 ST 分片 | 改 `ds4_distributed.c` 让 worker 知道本机 ST shard 的层/专家范围 |

### 3.8.6 与 §3.6/3.7 的关系

- **替代** §3.6 "冷档写进 GGUF"：冷档改 HF 直读（省盘），其余（倒置精度、Mode P/G 切换 §3.7）不变。
- **不改变地基顺序**：keep-map 屏蔽内核（k16 bug，§3.7.3）仍是第一块要修的，且现有资产可验，不必等下载。

---

## 4. P0 — 压缩到双机能装下（主线 A）

### 4.1 步骤 1：REAP 专家剪枝（离线 PyTorch）[✓证 F1/F2]

```bash
# 环境（离线大内存机/单 GPU）
git clone https://github.com/CerebrasResearch/reap && cd reap
pip install -e .   # Apache-2.0, transformers + vLLM 评测

# 准备校准集：必须含 react/go 代表性代码（否则保护错专家）
#   建议 256~512 段，混合 react(.tsx/.jsx) + go(.go) + 通用文本
cat calib_react/*.tsx calib_go/*.go calib_general/*.txt > reap_calib.txt
```

逐层 block-wise 校准剪枝（参 `experiments/pruning-layerwise-cli.sh`）：

```bash
python prune.py \
  --model deepseek-ai/DeepSeek-V4-Flash-Base \
  --prune-ratio 0.25 \                    # 保守起点，M1 扫 0.25/0.375/0.50 找拐点
  --saliency router_gate_x_act_norm \      # REAP 准则：门控值×激活范数
  --calib-file reap_calib.txt \
  --calib-samples 512 \
  --layerwise-blockwise \                  # 单 GPU 内存友好
  --output ./V4-Flash-REAP25
```

**[风险/适配]**：
- REAP 已发布 **V3.2 不是 V4-Flash** 预剪枝权重 → 须对照 `DeepSeek-V4-Flash-Base/config.json` 的 MoE 结构（专家数 `n_routed_experts`、路由 `num_experts_per_tok`、shared expert 配置），在 `prune.py` 的模型注册处加 V4-Flash 适配（专家张量命名、router 取值钩子）。
- 剪枝改变专家数 → 下游 imatrix、量化、ds4 GGUF 元数据全部联动。
- 命令行参数名以 repo 实际为准（上面是结构示意，clone 后核对 `prune.py --help`）。

### 4.2 步骤 2：重收 imatrix [✓证 F4]

剪枝改变专家激活分布，旧 imatrix 失效，必须重收：

```bash
# 剪枝后 HF → GGUF f16（ds4 gguf-tools 或 llama.cpp convert）
python llama.cpp/convert_hf_to_gguf.py ./V4-Flash-REAP25 \
  --outfile V4-Flash-REAP25-f16.gguf --outtype f16

# 重收 imatrix（校准集含 react/go）
./llama.cpp/llama-imatrix \
  -m V4-Flash-REAP25-f16.gguf \
  -f reap_calib.txt \
  -o V4-Flash-REAP25.imatrix \
  --chunks 512
```

> ds4 `gguf-tools/` 已有 imatrix 收集流程；若复用须保证校准集口径一致（含 react/go）。

### 4.3 步骤 3：非对称量化 + EAQuant 风格对齐 [✓证 F3/F4/F8]

保留现 profile，逐张量类型覆盖（默认全 Q8_0，靠 override 把路由专家压 2-bit）：

```bash
./llama.cpp/llama-quantize \
  --imatrix V4-Flash-REAP25.imatrix \
  --tensor-type 'ffn_gate_exps=IQ2_XXS' \   # 路由专家 gate
  --tensor-type 'ffn_up_exps=IQ2_XXS'  \    # 路由专家 up
  --tensor-type 'ffn_down_exps=Q2_K'   \    # 路由专家 down
  --tensor-type 'ffn_*_shexp=Q8_0'     \    # shared experts 高精度
  --token-embedding-type Q8_0 \
  --output-tensor-type Q8_0 \
  V4-Flash-REAP25-f16.gguf \
  V4-Flash-REAP25-IQ2XXS-w2Q2K.gguf  Q8_0
# 张量名(ffn_*_exps/ffn_*_shexp) 以 ds4 GGUF 命名为准, 核对后填正则
```

**[EAQuant 借鉴，可选增强]**：量化前做 *routing consistency alignment*——对校准集统计量化前后专家选择差异，对偏移大的专家走 mixed-precision（该层 down 升 Q3_K）。salient 专家保护是防坍缩关键（Qwen3-30B 剪 3/6144 即坍缩，Mixtral 均匀 INT2 掉 70→34%）。

### 4.4 步骤 4：ds4 装载验证

| 改动点 | 文件 | 内容 |
|---|---|---|
| GGUF 元数据解析 | `ds4.c`（GGUF 加载） | 确认剪枝后专家数/层数变化下 MoE 张量维度、router 维度正确读出 |
| 双机层切分 | `ds4_distributed.c` `--layers a:b` | 重新核算每机层数 → RSS ≤12GiB |
| Metal residency | `ds4_metal.m` | residency set 预算按新体积调整；专家流式 `DS4_METAL_EXPERT_*` 不变 |

**校验门**：`./ds4_test --metal-kernels --server` 绿；双机看门狗均 ≤12GiB；`--dump-logprobs` 对未剪枝 A3 baseline 比对掉点。

---

## 5. P1 — 本地微调补偿（主线 B）

> **训练细节已抽离为独立专册 `react-go-training-design.md`**（初始化/LoRA落点/超参/三类语料/specialty靶向/注入/eval/里程碑 T0–T5/资源测算/风险）。本节保留概要，落地以专册为准。

### 5.1 LoRA 目标层（硬约束）[✓证 F5]

适配器**必须**命中 MoE 专家投影，否则训练≈无效：

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

**[✗驳 禁用]**：不要"每专家单独 LoRA、rank=总rank/激活专家数"(0-3)；不要假定"LoRA==全量微调"(0-3)——预期 LoRA < 全量、用足够大 rank 弥补。

**落地自检**：训练启动后 dump 可训练参数量，**必须远高于 0.022%**（mlx-lm bug 阈值），否则适配器没命中专家层，立即停查。

### 5.2 2-bit 基座数值稳定：ApiQ 初始化 [✓证 F6/F7]

```bash
# 不能直接 QLoRA（3-bit 以下不收敛 [✓证 F6]）→ 用 ApiQ
git clone https://github.com/baohaoliao/apiq && cd apiq
# ApiQ：并发初始化 LoRA 分量 + 量化权重，保激活精度、抑误差从浅向深传播
python apiq_init.py \
  --model ./V4-Flash-REAP25 \
  --bits 2 \                              # 对齐 ds4 2-bit 区间
  --lora-rank 64 \                        # 编码域建议 32~128，扫
  --lora-alpha 128 \
  --target-modules experts \
  --output ./V4-Flash-REAP25-apiq-init
# 参数名以 apiq repo 实际为准
```

> 备选 LoftQ（2310.08659）：交替优化 `min‖W−Q−AB‖_F`，量化残差 SVD 初始化 A/B。ApiQ 在 2-bit 持续优于 LoftQ → **首选 ApiQ**。
> **[警告]** ApiQ 适配器存 FP16 抬高平均位宽；partial update ≠ 完整 QAT（2504.13932）。

### 5.3 训练配置（离线大内存机/云）

```yaml
# SFT 训练配置（建议起点，需扫）
base: V4-Flash-REAP25-apiq-init
lora: { rank: 64, alpha: 128, dropout: 0.05, targets: experts+shared }
optim: { lr: 1e-4, scheduler: cosine, warmup: 0.03, weight_decay: 0.0 }
batch: { per_device: 1, grad_accum: 16, max_seq_len: 8192 }  # 8k 容 react/go 长上下文
mem:   { gradient_checkpointing: true, bf16: true }
epochs: 2~3            # 域 SFT 不宜过拟合
data:   见 §6 三类语料配比
```

**数据规模测算** [假设]：域 SFT 经验量级 **3万~8万条**高质量样本（bug修复:书蒸馏:诚实agent ≈ 5:3:2，见 §6.4）。单 epoch 在单卡 A100/H100 级数小时；16GB Mac 直训剪枝后 V4 **无可行数据**（开放问题 2）。

### 5.4 适配器注入 ds4（两条路）

**路 A（推荐先试）：merge 回基座再量化** — 零推理路径改动。
```bash
python merge_lora.py --base ./V4-Flash-REAP25 --lora ./adapter \
  --output ./V4-Flash-REAP25-merged   # 回 FP16
# 再走 §4.2/4.3：重收 imatrix → 非对称量化 → 得最终 GGUF
```
- 优点：ds4 mmap+Metal 完全不变。缺点：改适配器要重量化。

**路 B：运行时 FP16 旁路** — 可热替换适配器。

| 改动点 | 文件 | 内容 |
|---|---|---|
| 适配器加载 | `ds4.c` | 读独立适配器张量（GGUF 附加段或单独文件），mmap 为 FP16 buffer |
| Metal 图 | `ds4_metal.m` / `metal/moe.metal` | 专家投影后加 `y += B·(A·x)` 旁路 kernel |
| residency | `ds4_metal.m` | 适配器 buffer 计入 12GiB 预算（rank 64 适配器 ~百 MiB 级，需实测） |

- 优点：热替换、A/B 不同适配器。缺点：动 Metal 图、撞双机红线需测。

**校验门**：merge/注入后 react/go eval ≥ M1（不退步）；`--dump-logprobs` parity；`ds4_test --metal-kernels --server`。

---

## 6. P2 — 三类语料工程（语料角度 1/2/3）

> **语料工程细节见专册 `react-go-training-design.md` §8**（数据源/处理管线/样本形态/配比规模/holdout）。本节保留概要。
> ⚠ 全段 [假设]：语料→Opus 4.6 因果链未证，eval 闭环裁判。

### 6.1 角度 1：代码仓库 + 核心库 + bug 修复 diff [✓证 数据集存在]

**数据源**：
- **Multi-SWE-bench**(`github.com/multi-swe-bench/multi-swe-bench`)：**多语言含 Go** 真实 issue→修复，带测试。
- SWE-bench(`hf.co/datasets/SWE-bench`)、SWE-Gym(`github.com/SWE-Gym/SWE-Gym`，可训练 agent 环境)。
- OctoPack/CommitPackFT(arxiv 2308.07124)：真实 commit→diff，含多语言。
- react 取材：React/Next.js/Remix 仓库的 bug-fix PR（issue + 失败测试 + 修复 diff + 测试通过）。

**处理管线**：
```
原始 commit/PR
  → 过滤(只留 bug-fix 标签 + 有测试 + 单一关注点)
  → 去重(MinHash LSH, hf.co/blog/dedup 流程, 阈值 ~0.85)
  → 许可合规(只收 permissive: MIT/Apache/BSD, 排除 GPL/无 license)
  → 格式化为 SFT 样本(见下)
  → 合成增强(arxiv 2504.14757: 对真实 diff 做难度变体/反例)
```
**样本形态**（关键：必须配测试，否则学不会"修对"）：
```
[问题] <issue 描述 + 失败测试输出>
[上下文] <相关 react/go 源码片段>
[修复] <最小 diff>
[验证] <测试通过>
[解释] <为什么这样修>
```

### 6.2 角度 2：架构/算法/重构经典书 → 蒸馏指令样本 [✓证 方法存在]

**方法**：知识蒸馏（arxiv 2402.13064 / 2312.02120 / 2405.03548）——书做**话题种子**，强模型(Opus 4.6)生成 react/go 的架构/重构教学样本，**不复制版权正文**。

**合规管线**（参 NVIDIA license-compliant synthetic data）：
```
经典书概念清单(《重构》坏味道表 / 《设计模式》23模式 / Clean Arch 依赖规则 / 算法导论核心算法)
  → 作为 prompt 种子(只用概念名,不喂正文)
  → Opus 4.6 生成: "给定坏味道的 react/go 代码 → 识别 → 按X原则重构 → 解释"
  → 自动校验(生成代码能编译/测试过)
  → 入库
```
**样本形态**（直接对齐"资深架构师质量"目标）：
```
[场景] 一段有 <坏味道/反模式> 的 react 组件 / go 服务
[诊断] 指出违反的原则(单一职责/依赖倒置/...)
[重构] 分步重构 diff
[原理] 引用经典原则解释收益(可维护性/可测性)
```
**[合规边界]**：种子=概念名与公开框架，输出=模型自生成代码，规避正文版权；Opus 蒸馏产出需符合其使用条款。

### 6.3 角度 3：执行诚实 agent 语料 [✓证 faithful CoT 方法]

**目标**：小模型最易"假装跑了测试/编造工具结果"——这是 agent 编码可靠性命门。
**方法**：faithful CoT(2307.13702/2511.14102) + Reflexion(2303.11366) 轨迹。
**样本形态**：
```
[任务] react/go 编码任务
[计划] plan-then-execute 显式步骤
[执行] 调用工具(读文件/跑测试) → **真实回填工具输出**(禁止凭空续写)
[反思] 测试失败 → 诚实记录失败 → 修正(Reflexion)
[完成] 测试真通过才声明完成
```
负样本(对比学习)：教模型识别"幻觉工具结果/假装通过"并拒绝。

### 6.4 数据配比与规模 [假设，需 §8 消融]

| 角度 | 占比 | 规模(起点) | 作用 |
|---|---|---|---|
| 1 bug修复diff | 50% | 1.5万~4万 | 核心：修对 bug 的能力 |
| 2 书蒸馏重构 | 30% | 0.9万~2.4万 | 架构/重构质量(顶向 Opus) |
| 3 诚实agent | 20% | 0.6万~1.6万 | 执行可靠性/不幻觉 |
| **合计** | 100% | **3万~8万** | |

> react:go 内部比例按目标场景定（建议 1:1 起，按 eval 弱项调）。**配比/规模是设计假设，M5 消融实测。**

---

## 7. P3 — prompt/agent 诚实执行体系（接 ds4）

| 技术 | 论文 | 接入点 |
|---|---|---|
| faithful CoT（不编造/不幻觉工具结果） | 2307.13702 / 2511.14102 | `ds4_agent.c` 系统提示词 + 工具结果强制真实回填 |
| Reflexion（失败自反思+重试） | 2303.11366 | `ds4_agent.c` 任务循环：测试失败→反思→重试 |
| plan-then-execute / self-consistency | 2503.08679 / 2508.00083 | `ds4_agent.c` 强制先 plan |
| exact-DSML replay（工具调用字节级匹配 KV） | ds4 已有(`rax.c`) | `ds4_server.c`：禁模型自由编造工具历史 |
| copy-speculation（n-gram 草稿+VERIFY） | ds4 已有 `DS4_DIST_COPY_SPEC` | 编程域 prefix 复用叠加有效 t/s |

**react/go 专用系统提示词要点**：
- react：函数组件 + hooks 规范、TS 类型严谨、可测性、避免 useEffect 滥用。
- go：error handling 惯例、并发安全、接口最小化、`go vet`/`golangci-lint` 自检。
- 全局：先 plan、工具结果必须真实、测试不过不声明完成、不臆造 API。

---

## 8. react/go eval 闭环（裁判）

**这是唯一裁判，先于一切优化建好。**

```
eval 集构成(建议 100~200 题, react/go 各半):
  - 真实 bug 修复(从 Multi-SWE-bench 留出 holdout, 不入训练)
  - 架构/重构题(给坏味道代码, 评重构质量)
  - 端到端 agent 任务(多步, 跑测试)

打分维度:
  - 功能正确(测试通过率) ← 主指标
  - 架构质量(人评 / Opus-as-judge 评分)
  - 执行诚实(工具幻觉率 = 编造工具结果次数/任务数)

基线: 用 Opus 4.6 跑同一 eval 集存参考分; 每 milestone 对比逼近度
```

复用 ds4 现有：`ds4-eval`(嵌入式 92 项，可扩) + `ds4-eval q1..q4 --temp 0 --seed 1` 确定性门。

---

## 9. 端到端执行路线图 M0–M6

| M | 名称 | 动作 | 交付 | 校验门（必过） |
|---|---|---|---|---|
| **M0** | eval 基线 | 建 react/go eval 集 + Opus 4.6 参考分；现状 81GiB 双机跑通 | eval 集 + 基线分 | eval 可复现；Opus 4.6 分记录在案 |
| **M1** | 剪枝拐点 | 离线 REAP 剪 25/37.5/50%，各重收 imatrix+量化，react/go eval 扫 | 选定 R%；剪枝量化 GGUF | 选能力可接受的最大 R；`--dump-logprobs` parity |
| **M2** | 装载 | 剪枝 GGUF 双机层切分装载，调 residency/流式 | 双机跑通 | 两机 ≤12GiB；`ds4_test --metal-kernels --server` 绿 |
| **M3** | 语料 | 建三类语料管线，产出 3万~8万 SFT 集，holdout 留出 | 训练集 + eval holdout | 去重/许可合规过；配比记录 |
| **M4** | 微调 | ApiQ 初始化 → MoE-LoRA 训练（离线）→ merge(路A) | 适配器 + merged GGUF | 可训练参数 ≫0.022%；eval ≥ M1 |
| **M5** | 调优 | 语料配比消融；prompt 体系接 `ds4_agent.c`；迭代 | 系统提示词 + 配比结论 | 工具幻觉率↓；react/go 通过率↑ |
| **M6** | 收敛 | 对标 Opus 4.6，诚实记录差距，定残余优化 | 最终 GGUF + 设计回填 | 逼近度报告（不粉饰差距） |

**每次合并 correctness 门**（CLAUDE.md）：`--dump-logprobs` parity + `ds4_test --metal-kernels --server`；routing/quant 改动加 `--logprob-vectors` + `ds4-eval q1..q4 --temp 0 --seed 1`。
**执行纪律**：每 patch/测量/scope/blocker 追加 `notes/execution-log.md`（wave 编号）；总结性结论需用户批准再写日志。

---

## 10. 风险登记 + 开放问题

**风险**：

| R | 描述 | 缓解 |
|---|---|---|
| R1 | REAP+2-bit 损失叠加放大（无人测） | M1 实测联合掉点，扫 R 找拐点 |
| R2 | 工具调用场景 REAP 掉点更大(-15%)，正是 agent 编码 | eval 含 agent 任务，不照搬"近无损" |
| R3 | PyTorch ↔ ds4 C99/Metal 鸿沟（REAP/ApiQ/LoftQ 全 PyTorch） | 全离线产出权重/适配器再转 GGUF，ds4 只推理 |
| R4 | P2 因果链未证 | eval 闭环唯一裁判，M5 消融 |
| R5 | V4-Flash 适配 REAP 脚本工程量大 | M1 先小剪枝率验证脚本通路 |
| R6 | 校准集偏差保护错专家 | imatrix 校准集必含 react/go |

**开放问题（无数据，需原型/实测）**：
1. REAP 剪 50% 再 2-bit，react/go 编码联合掉点多少？能压进双机预算且能力可接受？
2. 16GB Mac 对剪枝后 V4 做 MoE-LoRA 真实峰值显存/单 epoch 时长？双机分布式微调可行否？
3. 2-bit GGUF 运行时挂 LoRA 保 mmap+Metal 路径，适配器注入格式与数值稳定（路A vs 路B）？
4. 三类语料各需多大规模/配比才能把损失顶回接近 Opus 4.6？无先例消融。

---

## 11. 引用（全部经对抗验证，[✗驳] 标注禁用）

**量化/剪枝**
- REAP arxiv 2510.13999 · github.com/CerebrasResearch/reap (Apache-2.0) · hf.co/cerebras/DeepSeek-V3.2-REAP-345B-A37B
- MoQE arxiv 2310.02410（MoE 鲁棒性，比较性结论）
- EAQuant arxiv 2506.13329（MoE 专属 PTQ）
- llama.cpp quantize: github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md
- ds4 本地 README.md 行 123-125（非对称 profile）

**微调**
- LoRA 放置：thinkingmachines.ai/blog/lora · Biderman arxiv 2405.09673 · mlx-lm issue 571
- 2-bit 稳定：LoftQ arxiv 2310.08659 · ApiQ arxiv 2402.05147 + github.com/baohaoliao/apiq

**语料**
- SWE-bench(hf) · Multi-SWE-bench(含 Go) github.com/multi-swe-bench/multi-swe-bench · SWE-Gym github.com/SWE-Gym/SWE-Gym
- OctoPack/CommitPackFT arxiv 2308.07124 · 合成增强 2504.14757 · 去重 hf.co/blog/dedup

**蒸馏**
- arxiv 2402.13064 · 2312.02120 · 2405.03548 · NVIDIA license-compliant synthetic data pipeline

**prompt/agent 诚实**
- Reflexion 2303.11366 · faithful reasoning 2307.13702 · 2511.14102 · 2508.00083 · 2503.08679

**[✗驳] 禁用**
- MoQE "2-bit 纯专家免训练可靠"(0-3) · QuIP# 2-bit 困惑度数值(0-3) · QuIP# 两阶段微调必需(1-2) · "LoRA==全量微调"(0-3) · "每专家单独 LoRA rank=总/激活"(0-3) · "mlx-lm 完全没挂专家 LoRA"(1-2，仅 0.022% 事实确认)
