/* ds4_v41_api.h — DeepSeek V4.1 公共出口(2026-09-12 战役 P2c)。由 ds4.h 包含, 单独拆出只为守 ds4.h 的 500 行线。
 *
 * V4.1 的会话/采样/服务接入还没做(P5), 现阶段两条路: ①--score-ids 分块增量前向出全位置 logits(对拍尺);
 * ②贪心生成(prefill 分块 + 逐 token 解码同一条前向)。no_engram / chunk 是对拍夹具(chunk 0 = 默认 512)。 */
#ifndef DS4_V41_API_H
#define DS4_V41_API_H

#ifdef __cplusplus
extern "C" {
#endif

/* ds4_engine 的前置 typedef 来自 ds4.h(C99 不许重复 typedef) */
typedef int (*ds4_v41_emit_fn)(int token, void *ud);   /* 返回非 0 = 停止生成 */

int ds4_engine_is_v41(ds4_engine *e);
void ds4_engine_v41_set_prof(int on);   /* --v41-prof: 每次前向打逐层毫秒(每层同步一次, 只在查速度时开) */
void ds4_engine_v41_set_amp_dir(const char *dir);   /* --zchain <dir>: V4.1 反修放大器目录(amp_Lnn.bin), 每层 MoE 出口 y += x·(B·A) */
void ds4_engine_v41_set_amp_scale(float s);         /* --zchain-scale β: 加载时把 A 乘 β(修正整体缩到 β 倍); ≤0 = 1.0 */
int ds4_engine_v41_score_ids(ds4_engine *e, const int *ids, int n_ids, const char *out_path, int no_engram, int chunk);
int ds4_engine_v41_generate_argmax(ds4_engine *e, const int *prompt, int n_prompt, int n_predict, int ctx_size,
                                   ds4_v41_emit_fn emit, void *ud);

/* 反修取料钩子(2026-09-13, C 反修驱动 gguf-tools/amp/v41_amp_run 用): 每层 MoE 出口、放大器应用前回调一次。
 * 全是主机内存、行主序: x[n][D] = MoE 输入(ffn_norm 出口, bf16 格点), y[n][D] = MoE 输出(bf16 格点, 还没加放大器),
 * sel[n][n_used] / rw[n][n_used] = 路由选中的专家与权重(路由 gate 不量化 ⇒ FP 靶那一遍可直接复用), pos0 = 本块起始位置,
 * clamp = SwiGLU 截断。alpha[n] = 本层 y 经 hc_post 进 hc、再经 hc_pre(用本层 ffn_pre 当 pre_mix)合成出来的系数
 * (= Σ_k pre[i][k]·post[i][k], 逐 token 标量)。★只有末层的 alpha 等于"y 到出口 hc_pre 的精确系数"★ —— 末层之后
 * 直接就是 norm→head, 中间层的 y 还要再穿过后面所有层, 那个系数只是一阶直通项。蒸馏靶(v41_kl_target)靠它把
 * 出口梯度折回本层输出。返回 0 继续; 1 = 取完了, 本次前向到此为止(跳过余下层与出口, 位置照常推进, 本块不出 logits);
 * <0 = 停车。序贯解算靠多遍: 第 k 遍挂着目录里已解的 L0..L(k−1)(--zchain 同一条加载/应用路)按块正常前向, 到第 k 层取
 * 料即停, 收齐全部行后离线解第 k 层 —— 与"一次前向里逐层解+挂"数学等价, 但不要求整批(8192 整批的缓冲把 avail 压到 6 GB)。
 * 为什么是主机内存而不是设备指针: 钩子方(解算器)与引擎各管各的显存与流, 一层 2×168 MB 拷贝零点几秒, 换来后端无关。
 *
 * ye[n][n_used][D](2026-09-13, 权重侧逐专家反修): 逐专家 down 输出, 【未乘路由权重】—— 引擎这一层算的
 * y[i][d] 就等于 Σ_k rw[i][k]·ye[i][k][d]。解逐专家逐通道增益 g 要它(y 对 g 逐输出通道独立线性)。
 * ★只有 prefill GEMM 路(块内 n>8)物化这个中间量★; 解码 gemv 路不物化, 那里 ye = NULL。取料走
 * --score-ids(chunk 512)本来就是 prefill 路, 与判决同路。一层一块 63 MB(n=512), 用不着就别读。
 * ysh[n][D]: 同一层 shared 专家的输出(bf16 格点, 未进 y 的加法)。引擎这一层出的
 * y = round_bf16(Σ_k rw[i][k]·ye[i][k][d] + ysh[i][d]) —— 逐位。权重侧解 g 只动 routed 那一半,
 * 所以靶必须先把 ysh 减掉; 同时这条恒等式就是取料的自检(对不上 = 配对错位, 只会出假账)。 */
typedef int (*ds4_v41_moe_hook_fn)(void *ud, int il, int pos0, int n, int D, int n_used, float clamp,
                                   const float *x, const float *y, const int *sel, const float *rw, const float *alpha,
                                   const float *ye, const float *ysh);
void ds4_engine_v41_set_moe_hook(ds4_v41_moe_hook_fn fn, void *ud);

#ifdef __cplusplus
}
#endif
#endif
