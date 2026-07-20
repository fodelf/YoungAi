# go-onebit 技术过程文档

DeepSeek-V4-Flash 的 Go 域定制量化：**严格 1-bit 路由专家 + 每层 z 隐变量 + 四损失闭式解**。
本文件记录目标、现状、已验杠杆（带数字）、核心张力、下一步。每轮迭代更新。

## 目标（双指标，持续推进）
1. **更高质量**：当前是完整字母循环、写不出整段代码 → 目标能产出 Go 代码。
2. **更小体积**：当前 45.6 GiB → 更小。
约束（铁律）：只从原始 HF 量化、不训练（闭式解）、纯 C99、内存安全 ≤12GiB/机、不提非-1bit 退路。

## 架构
- 43 层 × 256 路由专家(top-6) + 1 shared expert，hidden 4096，moe_intermediate 2048。
- **1-bit 专家**：`block_go1b{half d; uint8_t signs[32]}` = 256 元素/34 字节 = 1.06 bit；d=mean(|w|)，bit→±d。
- **backbone**(attn/embed/router/shared-expert/输出) 保 Q8/F16 高精度。
- **z 隐变量**(每层)：`o_e^corr = ô_e + U·diag(C_e)·(V·x) + β_e + b`，δ 修 router logit。运行时 ds4 每层施加(已接通，apply 误差 1e-7)。
- **四损失**(hiddenvar_solve.c)：1 classify→δ｜2 fixed→ridge+dither｜3 smooth→差分｜4 align→方向。

## 迭代基建（scripts/）
- **iter.sh**：秒级内环。`./iter.sh` = base_cos **4s**(改 1-bit 方案 onebit_quant.c)；`--solve` = corr_cos **78s**(改 z/四损失)。瓶颈是 4096² solve ~100s 跟专家数无关→base_cos 不需 solve 所以秒级。旋钮：`-L`层 `-x`nx `-E`专家子集 `-R`rank `-A`w_align。
- **四损失权重 CLI 可调**(免重编)：calib_run `--w-align --w-smooth --w-classify --w-fixed --lambda --maxrank --n-experts --no-solve`。
- **test_go1b.sh**：全模型文本(分钟级)。基座模型须 BOS 前缀裸续写(见下)。
- M4 本地有 shards 1-13 = 层 0-12 可本机探；cap_m4 有激活。深层(26-42)需 M1 或重拷 shard。

## 已验杠杆与数据（L8, n_exp256 nx32）
| 改动 | corr_cos | 说明 |
|---|---|---|
| 基线(w_align=0, rank16) | 0.52442 | 纯 L2 解，z 近中性 |
| w_align=1 仅 stage5 | 0.52459 | 几乎无效(只重加权 per-expert C/β) |
| **w_align=1 + 方向感知U(stage2归一化)** | **0.53012** | ✓ 真起效(让 align 损失影响共享基 U) |
| **w_align=1 方向U + rank64** | **0.54707** | ✓ 单调涨(加 rank 抓更多方向子空间) |
| per-block scale(DS4_GO1B_PER_BLOCK) base_cos | +0.0015 | ✗ 可忽略(L0 .585→.587, L8 .516→.517); 免费但 scale 动不了方向 |

> calib_run 的 quant_dequant 已改用 block 格式(与 ds4 运行时一致); base_cos 是 sign 位决定, 任何 scale 方案(per-row/per-block)都动不了。

### ★突破★ 1-bit 残差栈打穿 sign 地板 (2026-07-01, DS4_GO1B_RESIDUAL)
W ≈ Q1(W) + Q1(W−Q1(W)) —— 第二层 1-bit 量化残差(sign+scale, 1-bit 原语零训练, 用户 line-61 正解):

| 层 | 单 1-bit base_cos | **1-bit + 残差** | relL2 |
|---|---|---|---|
| L0 | 0.585 | **0.855** | 0.82→0.58 |
| L8 | 0.516 | **0.822** | 0.78→0.51 |

**base_cos 0.52 → 0.82-0.85 = 直接到 80%**。叠 z 只增不减 → corr_cos ≥ 0.82。代码 calib_run.c quant_dequant(env 门控)。

**体积账(核心张力)**: 满残差 = 专家 34.8G→69.6G + backbone 11G = **~80G(=q2 大小)**。46G 装不下满残差(单 1-bit base 已 45.8G)。
- 46G 严格 → ~55%(单 1-bit, 循环)
- **~50G → ~80%**(甜点): 残差**只给热专家**(top-16 覆盖 95.6% 路由, +2.2G) + backbone 瘦身; 冷专家留单 1-bit
- ~80G → ~82%(满残差, =q2 大小但 Go 定制更优)

- **base_cos ≈ 0.521**(全 43 层稳定)：1-bit **sign 位**的固有方向极限(scale 不影响 cos，符号约定无 bug)。这是质量主瓶颈，z 中性时它决定一切。
- **拼写正确 ≠ experts 好**：来自高精度 backbone(embed/输出)，非 1-bit experts。
- **退化非 z 失效非上下文长度**：加不加 z 输出字节相同(A=B 实测)；真因是 1-bit experts 方向只对一半，43 层累积。

## 核心张力（size vs 质量）
- 方向子空间**高秩**(归一化残差 cumE@64=0.10，浅层 cumE@64=0.04-0.55) → 全方向恢复需大 rank → z 变大，**与"更小体积"冲突**。
- 深层(26-42)残差低秩(cumE@1-9=0.95) → z 在深层小而有效；浅层(0-23)高维不可约 → z 在浅层无力。
- {严格1bit + 小z + 零训练} 三约束张力：方向恢复要么大 rank(不小)要么…见下一步。

## 已修 bug
- **chat 模板套基座**(ds4_cli.c:416)：go1b 量化自 BASE 模型，-p 被套 `<｜User｜>/<｜Assistant｜>` → 乱讲。修：prompt 以 `<｜begin▁of▁sentence｜>` 开头走裸续写。test_go1b.sh 已内置。
- offload 退化死循环：非 runtime bug，是 greedy+1bit 质量(数值 0.0006 已验)。

## 下一步（持续推进，按杠杆强度）
1. **方向感知 U + 逐层自适应 rank**：深层小 rank(低秩够)、浅层大 rank(高秩需)，把 corr_cos 推到饱和点；量化 rank↔corr_cos↔z体积 的帕累托。
2. **o_ref/o_hat 缓存**(calib_run --reuse-acts)：z/四损失迭代跳过专家前向(~50s)，高 rank solve 才跑得动。
3. ~~数据感知 signs~~ **算力否决**(2026-07-01)：权重级 data-aware 二值要么需满输入协方差 min_B (W−sB)ᵀC(W−sB)(4096²×nrows×256专家×3矩阵×43层=不可行)，要么对角近似 C 退回 sign(w)(无增益)。→ base_cos 的 sign 地板**无可行杠杆**，质量只能靠 z。
4. **z 天花板**(进行中)：方向U 下 rank 16→256 扫 corr_cos，量化 46G 内(rank≤256, sidecar≤374MB)能爬多高，配合 w_smooth/nx。撞 1-bit sign 地板时这就是严格 1-bit+小z 的诚实上限(与 go_onebit_activation_space_exhausted 一致)。
5. **体积**：粗化 scale(per-512/1024 block，1.06→1.02 bit，边际)；浅层 z 不划算则只深层带 z。

## 关键文件
- 解：hiddenvar_solve.c(四损失+闭式) linalg_small.c(Cholesky/power-iter) onebit_quant.c(1-bit 编码)
- 探测：calib_run.c layer_probe.c hf_read.c｜运行时：ds4.c metal/moe.metal(corr kernel)｜emit：emit_z.c(corr sidecar)
- 脚本：scripts/{iter,test_go1b,gen_go1b,z_compute_dual,emit_and_run,calib_*,poll_z}.sh
