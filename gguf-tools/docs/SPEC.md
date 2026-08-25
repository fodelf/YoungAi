# Go 域 1-bit 固定模型 + 每层维度隐变量 + 四损失闭式还原 —— 技术规格

## 1. 目标与假设

- **目标**：用尽可能小的体积获得尽可能大的 Go 编程域质量。唯一指标 = Go 输出质量。
- **假设**：原始模型在 Go 域，43 层 × 256 专家的行为轨迹**低维可预测**（非高维随机）。
- **方法**：从 **HF 原始模型** 把全部 routed experts 压成 **严格二值 ±1（1.0 bit）** 固定模型 Θ_fix；为每层求**维度 d_ℓ 的隐变量 z^ℓ**；用 **Θ_fix + z^ℓ 在激活空间还原 Go 域行为**（还原行为而非权重数值）。闭式求解、不训练、不加数据。

## 2. 架构事实（实测 `hf/DeepSeek-V4-Flash-Base/config.json`）

| 项 | 值 | 来源字段 |
|---|---|---|
| 层数 | 43 | `num_hidden_layers` |
| routed experts | 256 | `n_routed_experts` |
| 每 token 激活专家 | **top-6** | `num_experts_per_tok=6`（DeepSeek 架构既定） |
| shared expert | 1 | `n_shared_experts` |
| hidden / FFN(每专家) | 4096 / 2048 | `hidden_size` / `moe_intermediate_size` |
| 路由打分 | sqrtsoftplus | `scoring_func` |
| 每专家矩阵 | `blk.%u.ffn_{gate,up,down}_exps.weight` | gate/up 4096→2048，down 2048→4096 = 25.2M 参/专家 |
| routed 总参 | ~277B → **1-bit ≈ 34.8 GiB** | |

> top-6 是模型既定**推理路由**，保留不变。Go 轨迹探测与隐变量校正**覆盖全 256 专家**（强制在 Go 激活上评估每个专家），不受单次前向 top-6 限制 → 全层全专家。

## 3. 双机集群执行（全运算/生成任务）

集群：M4 mini（coordinator）+ M1 MacBook（worker），Thunderbolt 直连 / NFS 共享 / SSH 编排，每机预算 ≤12 GiB。**所有探测、求解、生成、评测任务默认双机拆分，内存安全前提下最大化速度。**

- **内存安全（硬约束）**：每机 ≤12 GiB + 看门狗 + RSS 预算；单机绝不双载 base；专家从 SSD/NFS 流式（仅活跃专家驻留）；OOM 风险即停。
- **拆分维度**：层独立（每层 d_ℓ/z^ℓ/闭式解）按层分机；专家独立（256 专家逐个）按层/专家分机；多输入（Go prompts、go-bench 题目）按输入分机；单次前向用现有 layer-sliced 分布式 + 专家流式。
- **同步**：本机重编后经 SSH 同步二进制到另一机（共享 CORE_OBJS）；HF 与 dump 产物经 NFS 共享。

## 4. 产物（100% C，三份）

| # | 产物 | 形态 |
|---|---|---|
| ① | **1-bit GGUF 固定模型 Θ_fix** | 全 256 专家 × 43 层，每权重 1 符号位 + per-row fp16 scale。backbone（attn/embed/norm/router/shared-expert）保 Q8_0/F16。 |
| ② | **每层维度隐变量 z^ℓ**（43 份） | `z^ℓ = { V^ℓ(d_ℓ×4096), U^ℓ(4096×d_ℓ), C^ℓ(256×d_ℓ 每专家增益), b^ℓ(4096), β^ℓ(256 每专家偏置), δ^ℓ(256 路由) }`。d_ℓ 由探测实测。作额外 GGUF tensor 写进同一文件。d_ℓ=16 时全模型 ≈ 6.0M 参 ≈ 12 MB。 |
| ③ | **四损失（C）** | 每层一份，合成一个闭式正则最小二乘求解 z^ℓ。 |

体积：HF 240 / q2 81 / **本方案 ≈ 34.8 GiB + ~12 MB**。

## 5. 核心机制（激活空间、全 256 专家）

每专家 e 输出 `o_e = down_e·(SiLU(gate_e·x) ⊙ (up_e·x))`，1-bit 版 `ô_e`，误差 `Δo_e = o_e − ô_e`。
低维假设 → 同层 256 专家的 `{Δo_e(x)}` 落在共享低维子空间：

**`o_e^corr(x) = ô_e(x) + U^ℓ · diag(C^ℓ_e) · (V^ℓ x) + β^ℓ_e·1 + b^ℓ`**

- `V^ℓ x`：每层共享 d_ℓ 维 Go 特征（低维轨迹）；`U^ℓ,b^ℓ`：每层共享提升/偏置。
- `C^ℓ_e, β^ℓ_e`：**每专家**系数（全 256 覆盖，每专家落在共享轨迹上的位置）。
- `δ^ℓ`：路由 logit 校正，恢复原始 top-6。

运行时：推理仍 top-6，对每个被选专家在融合核里加 `U·diag(C_e)·(Vx)+β_e+b`（小矩阵乘）。

## 6. Go 轨迹探测（probe，全 256 专家，输出数据）

1. **采激活**[双机]：Q8_0 参考在**短 Go 片段**上前向，`DS4_METAL_GRAPH_DUMP_PREFIX/NAME(ffn_norm,ffn_moe_out,ffn_moe_logits)/LAYER/POS` 导出每层 `{x_i, y_i^ref, logits_i}`。
2. **全专家行为**[双机]：每层把 256 专家**逐个**在 Go 激活池 `{x_i}` 上跑（C layer-probe，不受路由限制），得每专家原始 `o_{e,i}` 与 1-bit `ô_{e,i}`。
3. **测每层维度 d_ℓ**[双机]：对每层堆叠残差 `{Δo_{e,i}}` 做 power-iteration 主成分，能量阈值（如 95%）定 **d_ℓ** → **表 P**。低维假设数据裁决：d_ℓ ≪ 4096 才继续。

## 7. 四损失闭式求解（每层一份，非训练）

在 Go 标定集上对每层求 `z^ℓ`，目标 = 四损失加权和的闭式最小化：

1. **分类损失（路由）**：router probs/selection 最小二乘 → `δ^ℓ` 恢复原始 top-6。
2. **固定损失（噪声）**：固定 Tikhonov `λI` + 定种子确定性 dither → 钉死欠定解，确定可复现、抗过拟合。
3. **光滑损失**：差分算子二次惩罚（校正输出在 Go token 序与输出坐标上不跳变）。
4. **对齐损失**：方向/cosine 惩罚（保 logit 排序方向）。

求解 = 正规方程 + Cholesky + power-iteration 截到 d_ℓ（对齐项 ≤1 次 reweight）。**一次闭式解，无 SGD/epoch/反传。**

## 8. 探测调优闭环（旋钮 + Go 指标驱动，每轮闭式重解）

```
探测(§6) → 1-bit 量化 → 闭式解 z^ℓ(§7) → 测 Go 保真(表 B/C/D) → 调旋钮 → 重解
```
旋钮：每层 `d_ℓ`、`λ`、四损失权重 `w1..w4`、1-bit scale 方案（per-row/per-group）、dither 种子。
按表定位**失败的层/专家**针对性调，再闭式重解。全程调超参 + 重解，**不训练权重**。
节奏：每轮 <10min 快验证；突破迹象才进 go-bench 深验。

## 9. 任务清单

**P0 探测基建（慢可接受）**
- P0.0 从 HF 量化出 **Q8_0 参考 GGUF**。
- P0.1 [双机] Q8_0 参考短 Go 片段前向 + dump `{x,y,logits}`（专家流式 + 看门狗 + RSS 预算）。
- P0.2 [双机] C **layer-probe**：逐专家算 `o_e`(Q8) 与 1-bit `ô_e`。
- P0.3 [双机] 测每层 d_ℓ → **表 P**；低维假设数据裁决。

**P1 离线 C 求解 + 判据（快，<10min，不写 kernel）**
- P1.1 `onebit_quant.c`：严格二值编码器（sign + per-row fp16 scale），接 `ds4q_quantize_chunk`/`quants.h` traits。
- P1.2 `linalg_small.c`：稠密 Cholesky + power-iteration rank-d。
- P1.3 [双机] `hiddenvar_solve.c`：四损失（各一 C 函数）→ 正规方程 → 解 `z^ℓ`（全 256 专家系数）。
- P1.4 出 **表 B、表 C**。门槛：rank-d_ℓ 校正把每层输出保真度拉回才进 P2。

**P2 量产（仅突破时）**
- P2.1 `z^ℓ` 写进 GGUF tensor（`blk.%d.corr_{U,V,C,b,beta,delta}` + KV 开关）。
- P2.2 `metal/moe.metal`：`dequantize_onebit` + 融合核 `kernel_mul_mv_id_onebit_pair_swiglu_f32`/`_sum6`。
- P2.3 `ds4_metal.m`：`ds4_gpu_routed_moe_one_tensor` 派发 1-bit 核 + 每专家校正项。
- P2.4 `ds4.c`：加载 z^ℓ；δ 注入 `layer_topk_selected_experts_from_probs`；输出校正注入 `layer_routed_moe_one` 后（CPU 先验证）。
- P2.5 [双机] 端到端 + go-bench pass@1 终裁。
- P2.6 过 `ds4_test --metal-kernels` 合并门 + SSH 同步二进制。

## 10. 关键文件

- **新建（C，`gguf-tools/go-onebit/`）**：`onebit_quant.c`、`linalg_small.c`、`hiddenvar_solve.c`、`layer_probe.c`、dump 驱动 + 最小 Go 快验证脚手架。
- **复用**：`write_full_gguf`/`write_gguf_string`/KV writer；`ds4q_quantize_chunk`(L1050)/`ds4q_make_qkx2_quants`(L119)；`expert_worker`(L1438)；dump 钩子；现有融合核模板；`ds4_distributed.c`。
- **改（运行时，P2）**：`metal/moe.metal`、`ds4_metal.m`(L20035)、`ds4.c`(L6362/L6730)。

## 11. 多维对比表（必出，不空口）

- **表 P**：逐层有效秩 d_ℓ / 各 rank 能量占比。
- **表 A**：体积（hf / q2 / 1-bit base / +z^ℓ）。
- **表 B**：`o^corr` vs `o^ref` rel-L2 + cosine，跨 d_ℓ×λ×w 网格，逐层。
- **表 C**：加 δ 前后 top-6 overlap，逐层。
- **表 D**：Go 质量 = C proxy（greedy-match% / logprob-KL）+ go-bench pass@1。

## 12. 验证

- **最小 Go 快验证（内环秒级原语，纯 C，每轮调旋钮后必跑）**：单条极短 prompt 让模型贪心续写**几行 Go 代码**（如 `func Add(a, b int) int` 返回两数之和），生成 ≤64 token（单次前向秒级）。判据：① greedy-match% / logprob-KL vs Q8 参考（纯 C，秒级）；② 可选 `go build` 把片段塞进最小 module 编译通过（秒级）。只有通过才升级。
- **内环表（<10min，[双机] 分机加速）**：表 P/B/C；`ds4 --dump-logprobs --temp 0`。
- **终裁（突破迹象时，[双机] 题目分机）**：`benchmarks/go-bench/run_bench.py` 子集 pass@1（唯一 Python，`go test` 外壳，不进产物）。
- **合并门**：`ds4_test --metal-kernels`。

## 13. 约束

- 闭式求解，不进训练循环；不提任何非 1-bit 提质方案。
- 产物 100% C；Python 仅 go-bench 终裁。
- 只从 HF 原始量化，绝不用 q2 当模版/源。
- 全任务双机集群执行（§3）；内存安全硬约束：每机 ≤12 GiB + 看门狗 + RSS 预算，单机绝不双载 base，专家流式，OOM 风险即停。
