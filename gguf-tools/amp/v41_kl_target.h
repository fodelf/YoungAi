/* v41_kl_target.h — 蒸馏靶(把"让最终 logits 更像教师"翻译成"这一层的输出该往哪挪")接口。
 * 实现 v41_kl_target.cu(2026-09-13)。
 *
 * 【为什么要它】现役反修的靶是"让这一层的输出更接近教师这一层的输出"(最小二乘)。但没人保证
 * 一层更像就让最终答案更像 —— 09-12/13 反复量到的就是这个: 40 层逐层 held-out 全正(+9.1%),
 * 端到端 Σmin 0.676→0.600; 只挂前 3 层也持平偏负。逐行/逐通道/拟合份三个切面都排除了过拟合。
 *
 * 【这里换什么】靶改成"负梯度": 要让 KL(教师‖学生) 变小, 这一层的输出该往哪个方向挪。判据
 * (KLD/Same top)与优化目标变成同一个量, 上面那种分叉按机理消失。教师锚存的就是每个位置的
 * 全部 logits, 所以教师分布是现成的, 不需要再跑 FP 模型。
 *
 * 【闭式链, 不需要反向传播(ds4 没有反向)】只对【末层】成立:
 *   dL/dlogits        = p_student − p_teacher
 *   dL/d(hc_pre 出口) = W_headᵀ · dL/dlogits, 再 ⊙ output_norm 权重
 *                       (RMSNorm 雅可比只取主项 (g⊙v)/rms; 丢掉的沿 x 投影项量级 ~1/D,
 *                        rms 是逐 token 正标量, 与下面的 α 一起被步长吸收)
 *   dL/d(末层 routed) = α · 上,  α[i] = Σ_k pre_mix[i][k]·post[i][k](逐 token 标量, 引擎钩子给)
 * 靶 R = −η·dL/d(routed)。η 不拍脑袋: 按 --eta-rel r 缩放, 使 ‖R‖ = r·‖y_q‖。
 *
 * 【为什么只能是末层】链止于末层 —— 再往前需要把梯度穿过 attention/hc/后续层, 没有反向就传不回去。
 * 末层针的作用是判"直接优化出口的低秩修正能不能拿到端到端正收益"这个机理问题, 正了再谈往前推。 */
#ifndef V41_KL_TARGET_H
#define V41_KL_TARGET_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct v41_klt v41_klt;

/* 从 GGUF 取出口两件: output.weight(fp4x32 [V][D] → 设备 f16, 约 1.3 GB) 与 output_norm.weight(f32 [D])。
 * D 由调用方给(引擎的 n_embd), 与张量形状不符即失败。返回 0 成功。 */
int v41_klt_open(v41_klt **out, const char *gguf_path, int D);

/* 算靶并直接产出解算器要的 dYfp = dYq + R(设备 f32 [n][D]) —— 解算器的靶恒为 dYfp − dYq, 所以把
 * "该往哪挪"加到 y_q 上, 解算器与格式一个字节都不用改。dYfp_out 同时当 R 的工作缓冲, 可以与 dYq 不同但不可重叠。
 * ref_path/stu_path 是 <i32 S><i32 V><f32 logits[S][V]> 两份(教师锚与同一趟前向落的学生 logits, 行号同口径)。
 * alpha[n] 主机, 引擎钩子给。perm[n](主机, 可 NULL=恒等)= 解算器的行序: 输出第 i 行取原序第 perm[i] 行的梯度
 * —— 梯度按 logits 原行序算, 解算器吃的是"拟合行在前"的置换序, 换序只在这里做一次。
 * qnorm = ‖y_q‖(整块 L2), eta_rel = 目标 ‖R‖/‖y_q‖。
 * out_stat 非空回填 [0]=缩放前 ‖α·g‖/‖y_q‖, [1]/[2]=α 最小/最大值, [3]=α 均值。返回 0 成功。 */
int v41_klt_target(v41_klt *k, const char *ref_path, const char *stu_path, const float *alpha, const int *perm,
                   int n, int D, double qnorm, float eta_rel, const float *dYq, float *dYfp_out, float *out_stat);

/* ★后训练第二版的方向表★(2026-09-13 夜, back.md §4.1): 决策差解算器要的 C[n][D]
 *     C[i][d] = α_i · inv_i · γ[d] · (W[a_i][d] − W[b_i][d])
 * a_i/b_i = 这一行要比较的两个 token: 决策行 = (对版, 错版), 约束行 = (它原本的 top1, top2)。
 * inv[n] 来自引擎 --score-rms(出口 RMSNorm 的 rsqrt 标量), α[n] 来自钩子。
 * 出口头 W 用的就是已经常驻的那份 f16(它是【落地量化后】的头 —— 判决与部署必须同一份权重)。
 * dC 设备缓冲 [n][D] f32 由调用方备好。a 或 b 为负 = 这一行没有可比的两个 token, 整行写 0
 * (它在解算里既不提供方程也不被约束)。返回 0 成功。 */
int v41_klt_margin_dirs(v41_klt *k, const int *a, const int *b, const float *alpha, const float *inv,
                        int n, int D, float *dC);

/* ★第七版(2026-09-29)的方向表★: 自 token 减该位榜上期望 —— C[i][d] = α_i·inv_i·γ[d]·(W[a_i][d] − Σ_q p_iq W[ids_iq][d]),
 * 即 log p(a_i) 对本层输出的精确线性化。dIds/dPs = top-K 表的 id 与概率列(设备 [n][K])。a_i<0 整行写 0。返回 0 成功。 */
int v41_klt_own_dirs(v41_klt *k, const int *a, const int *dIds, const float *dPs, const float *alpha, const float *inv,
                     int n, int K, int D, float *dC);

/* ★主动集扫榜★: 给定这一版修正让本层输出挪了多少(dY[n][D], 设备), 算出榜上每个 token 的
 * logit 各动了多少 dLogit[n][K](设备)。dIds[n][K] 是 top-K 表的 id 列(设备)。
 * 用途: 每行只钉一对不够 —— 钉住的那对分毫不差, 冒头的却是第三个 token(09-13 实测 37% 的位置
 * argmax 变了)。有了这张表, 主动集在解算器内部就能找出冒头的是谁, 不必为此跑真前向。返回 0 成功。 */
int v41_klt_margin_scan(v41_klt *k, const float *dY, const float *alpha, const float *inv,
                        const int *dIds, int n, int K, int D, float *dLogit);

void v41_klt_close(v41_klt *k);

#ifdef __cplusplus
}
#endif
#endif
