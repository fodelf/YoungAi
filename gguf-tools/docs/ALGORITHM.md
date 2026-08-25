# Go 域逐层蒸馏 —— 结论与算法

## 1. 结论（实测裁决，2026-06-29）

- **Go 域规律真实存在**（premise 成立）：模型对 Go 代码的处理不是高维随机——用够格的方法能从激活里学出来。代码有严谨性（语法/类型/作用域确定性）→ 收敛出可学规律。
- **规律的形态 = 每层一个可学的函数 `ffn_in → ffn_out`**（整层 MoE 的 Go 域行为），不是"修 1-bit 误差"。**每层函数不同**，按层定制。
- **可学性随深度变化**（实测 held-out cosine）：
  - 浅层（L0）：**0.94**（近无损，正经深网）——浅层路由仅 1210 确定性组合，函数简单。
  - 中层（L6–30）：数据受限，随训练量单调上升（L21: 3k→0.44, 6k→0.57, 9k→0.60, 11k→0.63，**到 11k 仍在涨**）。中层高路由熵=分片函数=数据饥饿。
  - 深层（L42）：0.82（部分受 massive-activation outlier 驱动）。
- **瓶颈 = 数据 + 训练**，非函数本身。中层 0.6 是 11k token 的天花板，不是规律的天花板。

## 2. 死路（已穷尽证伪，勿重蹈）

「严格 per-row 1-bit + 修误差 Δo」路线全死：
- 1-bit 专家输出 cos 仅 ~0.55；误差 Δo 是高维噪声（oracle@rank64 残 0.947, x→Δo R²=0.04）。
- 修误差的 5 个杠杆 ×3 深度（每专家/聚合 × 线性/非线性 × held-out）cosine 从不超基线。
- **关键区别：修 ERROR(Δo) 是噪声不可学；学 FUNCTION(ffn_in→ffn_out) 有结构可学。** 前向已 bit 级验证（非 pipeline bug）。

## 3. 已验证算法 —— 每层 Go 专用蒸馏器 M_φ^ℓ

**目标**：每层一个小网络 `ŷ = M_φ^ℓ(x)` 逼近该层 Go 域 MoE 函数（x=ffn_in, y=ffn_out），按层复杂度配大小（浅层极小、中层大）。这是产物②"动态小模型/隐变量"的落地形态。

**架构**（`proper_distill.py`，torch + M1 MPS 加速）：残差 MLP
```
x → Linear(d→h) → [LayerNorm; Linear(h→h); GELU; Dropout; Linear(h→h)] ×NB (残差) → Linear(h→d)
```
d=4096, h=2048, NB=3（按层可调）。

**训练**：AdamW + CosineAnnealing LR；输入/输出 per-dim 标准化；held-out 留出集早停取峰值 cosine。**严重过拟合是中层主敌**（R² 可跌至 −4.7）→ 强正则关键。

**数据**：cap_m1 的 Go 激活（dsv4_fwd.py 纯 numpy 前向产出，原始模型 FP8）。token→文本可经重 tokenize gocorpus_big.txt 对齐。

## 4. 四损失增强（下一步，接回原设计）

当前训练只用 MSE，次优（metric 是 cosine）。每层蒸馏目标改为四损失加权（对应原 SPEC 四损失，落到蒸馏语境）：

1. **对齐损失** `L_align = 1 − mean cos(ŷ,y)` —— 直接优化方向（= 真 metric），最关键。
2. **分类损失** `L_classify` = 按 y 的 per-dim 方差（重要性）加权的 MSE —— 保住驱动下游路由的判别维度。
3. **光滑损失** `L_smooth = ‖M(x+δ)−M(x)‖²`（固定种子 δ）—— 输入鲁棒/不跳变，兼抗过拟合。
4. **固定损失** `L_fixed` = weight-decay + 固定种子 dither 数据增广 —— 钉死解、治过拟合（中层主敌）。

`Loss = w_c·L_classify + w_a·L_align + w_s·L_smooth`（+ fixed 经 optimizer/增广）。预期：align 直接抬 cosine，fixed/smooth 治过拟合让中层从现有数据再上一截。

## 5. 后续

- 实现四损失 → L21 验证是否 > MSE-only 的 0.628。
- 扩数据：dsv4_fwd.py 抓更多 Go 激活（中层数据受限，更多数据→更高 cosine）。
- 全 43 层蒸馏 → 端到端 Go 质量 + 总体积裁决（每层小网络之和 vs base）。

产物仍落 C；Python 仅用于 discovery（用户 2026-06-29 批准）。

## 6. ★修正方向（产物 = 从 HF 量化、非训练；三方向合力）

用户 2026-06-29 校正：**产物必须是"从原始 HF 量化出的 1-bit 模型 + 隐变量"，不要自己训练一个替代模型**。§3 的深度蒸馏（从零训练大网络替代 MoE）只作 **discovery**（已证 Go 函数可学、是真规律），**不是产物**。追求质量 = **三方向一起、尽量闭式不训练**：

1. **定制化 Go-aware 1-bit 量化（从 HF，闭式，零训练）** —— 之前完全没做（go1b 用通用 mean(\|w\|) scale=base 卡 0.55 的根源）。换**输出最优 Go-aware per-row scale** `s_go_i = Σ_x(w_i·x)(b_i·x)/Σ_x(b_i·x)²`（用 Go 激活协方差，b=sign(w)；w1/w3 用 x、w2 用 SwiGLU 中间 h）。**实测 per-expert base 一致 +0.10（0.55→~0.70，诚实 ≥2-sample 值），纯量化、合规。** 还可叠 Go 重要性加权 / 分组 scale / salient 等 Go-aware 招。
2. **隐变量** —— 叠在更好的 Go-aware base 上（base 越高、残差越小、负担越轻；三方向复利）。
3. **损失函数** —— 调上面两者。

**核心判据（进行中）**：全闭式无训练栈（Go-aware 量化 + 闭式隐变量）held-out 能到多高？
- 到 ~0.85+ → 根本不需训练, 完美满足约束；
- 仍封顶 ~0.75 → 训练 gap 仍在, 回到 (A 轻量拟合)/(B 严格不训练) 抉择。

**已验证的工具**：`validate_fwd.c` 加了 `--goaware`（Go-aware scale 对比 + 隐变量叠加, 纯 C 复用前向）。

