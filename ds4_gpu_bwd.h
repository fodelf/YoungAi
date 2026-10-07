#ifndef DS4_GPU_BWD_H
#define DS4_GPU_BWD_H

#include <stdbool.h>
#include <stdint.h>

#include "ds4_gpu_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * 后训练反传原语(2026-10-01, 后训练 ③ 第八版 = 上下文蒸馏)。
 * =========================================================================
 *
 * 为什么引擎要有反传: 文档写进权重的成熟方案(fable5 10-01 夜检索段: KMs / Prompt Distillation / Cartridges / SDFT)
 * 全部是"学生在答案位逼近教师分布、梯度穿过整段网络更新低秩件"。闭式解算器只能把靶线性化到当层输出, 答案位的
 * 激活不像拟合行就点不着(10-01 六趟复盘 ③ 的结构病)。这里只放训练要的那几样: 转置乘、KL 梯度、各模块的反向、Adam。
 *
 * 口径: 激活与梯度一律 f32 行主序 [n][dim]。前向在 bf16 模块边界处的舍入, 反向一律按直通处理(舍入的导数取 1) ——
 * 这是混合精度训练的通行做法, 梯度检查(有限差分)的容差按 bf16 噪声给。只有 CUDA 实现; Metal 是桩。
 * 权重类型号用 GGUF 线上编号(src/common/ds4_quantfmt.h 的 DS4_GGT_*): 0 f32 / 12 q4_K / 30 bf16 / 43 fp4x32 / 44 fp8 32×32。 */

/* gx[n][in] (+)= gy[n][out] · W, W 是盘上 [out][in] 的骨架矩阵(前向是 out = x·Wᵀ)。权重按行分块解成 bf16 暂存,
 * gy 转 bf16, cuBLAS f32 累加。accumulate=0 覆盖 gx, 1 加到 gx 上。f32 权重直接 Sgemm(gy 不降精度)。 */
int ds4_gpu_bwd_matmul_t_tensor(ds4_gpu_tensor *gx, const void *model_map, uint64_t model_size, uint32_t wtype,
                                uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                const ds4_gpu_tensor *gy, uint32_t n_tok, int accumulate);

/* 分组块对角投影(注意力 wo_a)的转置: heads[n][G·gd] 的梯度 (+)= low[n][G·rank] 的梯度按组乘回。W 盘上 [G·rank][gd]。 */
int ds4_gpu_bwd_grouped_matmul_t_tensor(ds4_gpu_tensor *gheads, const void *model_map, uint64_t model_size, uint32_t wtype,
                                        uint64_t weight_offset, uint32_t n_groups, uint64_t group_dim, uint64_t rank,
                                        const ds4_gpu_tensor *glow, uint32_t n_tok, int accumulate);

/* 上下文蒸馏的损失与对学生 logits 的梯度(教师只给 top-K + 余量, "余量桶"形式的前向 KL):
 *   L = Σ_k pT_k·log(pT_k/pS(id_k)) + r_T·log(r_T/r_S),  r = 1 − Σ_k p(id_k)
 *   ∂L/∂z_j = pS_j − q_j,  q = pT(榜上) / r_T·pS_j/r_S(榜外)   —— 榜外的 token 按学生自己的相对比例分教师余量。
 * logits[rows][V] 第 row0+i 行 ↔ 教师第 i 行(i < m); 梯度 × scale 写进 glogits[m][V](可与 logits 同址: 逐行先读后写);
 * loss[m] 每行一个 KL(nat)。tid[m][K] i32, tp[m][K] f32, trest[m] f32 都在设备上。
 * w[m] f32 逐行权重(NULL = 全 1): 硬目标题(教师表 = 答案 one-hot, KL 退化成交叉熵)按 token 加权, 损失与梯度同乘。 */
int ds4_gpu_bwd_kl_topk_tensor(ds4_gpu_tensor *glogits, ds4_gpu_tensor *loss, const ds4_gpu_tensor *logits, uint32_t row0,
                               uint32_t m, uint32_t n_vocab, const ds4_gpu_tensor *tid, const ds4_gpu_tensor *tp,
                               const ds4_gpu_tensor *trest, const ds4_gpu_tensor *w, uint32_t k, float scale);

/* 教师 top-K: logits 第 row0..row0+m-1 行 → 概率最大的 K 个 (id, p) 与余量 1 − Σp(softmax 全词表)。
 * ★会改写 logits 这几行★(选中的位置置 −inf, 选 K 轮)—— 调用方读完才许调。tid[m][K] / tp[m][K] / trest[m]。 */
int ds4_gpu_bwd_topk_tensor(ds4_gpu_tensor *tid, ds4_gpu_tensor *tp, ds4_gpu_tensor *trest, ds4_gpu_tensor *logits,
                            uint32_t row0, uint32_t m, uint32_t n_vocab, uint32_t k);

/* RMSNorm 带权(前向 xn = w ⊙ x · r, r = 1/√(mean x² + eps), w 是盘上 f32)的反向:
 *   gx = r·(w ⊙ gxn) − x·r³·Σ(w ⊙ gxn ⊙ x)/D。accumulate=1 加到 gx 上。 */
int ds4_gpu_bwd_rms_norm_tensor(ds4_gpu_tensor *gx, const ds4_gpu_tensor *gxn, const ds4_gpu_tensor *x,
                                const void *model_map, uint64_t model_size, uint64_t weight_offset,
                                uint32_t dim, uint32_t n_tok, float eps, int accumulate);

/* hc_pre(前向 x[d] = Σ_c pre[c]·hc[c][d])的反向: ghc[c][d] += pre[c]·gx[d](总是累加); gpre[c] = Σ_d gx[d]·hc[c][d](覆盖)。 */
int ds4_gpu_bwd_hc_pre_tensor(ds4_gpu_tensor *ghc, ds4_gpu_tensor *gpre, const ds4_gpu_tensor *gx,
                              const ds4_gpu_tensor *hc, const ds4_gpu_tensor *pre, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok);

/* hc_post(前向 out[k][d] = post[k]·y[d] + Σ_j comb[j][k]·res[j][d])的反向:
 *   gy[d] = Σ_k post[k]·gout[k][d](覆盖) / gres[j][d] += Σ_k comb[j][k]·gout[k][d](累加, 可 NULL)
 *   gpost[k] = Σ_d gout[k][d]·y[d] / gcomb[j][k] = Σ_d gout[k][d]·res[j][d](覆盖, 可 NULL)。 */
int ds4_gpu_bwd_hc_post_tensor(ds4_gpu_tensor *gy, ds4_gpu_tensor *gres, ds4_gpu_tensor *gpost, ds4_gpu_tensor *gcomb,
                               const ds4_gpu_tensor *gout, const ds4_gpu_tensor *y, const ds4_gpu_tensor *res,
                               const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb,
                               uint32_t n_embd, uint32_t n_hc, uint32_t n_tok);

/* 低秩放大器(前向 y += x·(B·A), A/B [K][D] 行主序)的反向: T = x·Bᵀ, gT = gy·Aᵀ;
 *   gA += Tᵀ·gy, gB += gTᵀ·x(梯度累加, 跨样本攒批); gx(可 NULL) += gT·B。T/gT 是 [n][K] 暂存。 */
int ds4_gpu_bwd_amp_tensor(ds4_gpu_tensor *gA, ds4_gpu_tensor *gB, ds4_gpu_tensor *gx, const ds4_gpu_tensor *gy,
                           const ds4_gpu_tensor *x, const ds4_gpu_tensor *A, const ds4_gpu_tensor *B,
                           ds4_gpu_tensor *T, ds4_gpu_tensor *gT, uint32_t n_tok, uint32_t D, uint32_t K);

/* ---- 第二阶段: 逐层反传(穿过注意力 / MoE / mHC 混合系数 / 压缩器 / engram), 实现在 cuda_bwd_attn / hc / moe / comp ---- */

/* 稀疏注意力(窗口 + 选中的压缩行, sink 只进分母, 值 = 键)的反向。只接"一题一块、从位置 0 起"的形态: 窗口缓冲第 window+i 行 =
 * 第 i 个位置的 kvn; 压缩行(kv_comp/idx)与前向同一份。出 gq[n][H][D](覆盖)、gkv[n][D](覆盖, 键与值两种身份合计);
 * gcomp 非 NULL 时把对压缩行的梯度**累加**进 gcomp[组][D](源层在训练段里才要; NULL = 压缩行当常量)。 */
int ds4_gpu_bwd_sparse_attn_tensor(ds4_gpu_tensor *gq, ds4_gpu_tensor *gkv, ds4_gpu_tensor *gcomp, const ds4_gpu_tensor *go, const ds4_gpu_tensor *o,
                                   const ds4_gpu_tensor *q, const ds4_gpu_tensor *kv_win, const ds4_gpu_tensor *kv_comp,
                                   const ds4_gpu_tensor *idx, const void *model_map, uint64_t model_size, uint64_t sink_offset,
                                   uint32_t n_tok, uint32_t window, uint32_t ng, uint32_t topk, uint32_t n_head, uint32_t head_dim, float scale);
/* RoPE 反向 = 反方向旋转、不舍 bf16。参数与前向 ds4_gpu_v41_rope_tensor 同, inverse 传"前向那一发取反"。就地改 x。 */
int ds4_gpu_bwd_rope_tensor(ds4_gpu_tensor *x, const ds4_gpu_tensor *pos, uint32_t n_tok, uint32_t n_head, uint32_t head_dim,
                            uint32_t n_rot, float theta, uint32_t original_seq_len, float factor, float beta_fast, float beta_slow, bool inverse);
/* mHC 混合系数(mix → pre/post/sinkhorn comb)的反向: ghc += ∂(pre,post,comb)/∂hc 的贡献。gpre/gpost/gcomb 可 NULL(= 0)。
 * hc = 该半层入口 hc [n][HC·E], mix = 前向的 mix [n][2HC+HC²](重算得来)。norm_eps = mix 前那次摊平 RMS 的 eps(DS4_RMS_EPS)。 */
int ds4_gpu_bwd_hc_mix_tensor(ds4_gpu_tensor *ghc, const ds4_gpu_tensor *gpre, const ds4_gpu_tensor *gpost, const ds4_gpu_tensor *gcomb,
                              const ds4_gpu_tensor *hc, const ds4_gpu_tensor *mix, const void *model_map, uint64_t model_size,
                              uint64_t fn_offset, uint64_t scale_offset, uint64_t base_offset,
                              uint32_t n_embd, uint32_t n_hc, uint32_t iters, float hc_eps, float norm_eps, uint32_t n_tok);
/* SwiGLU(h = silu(min(g, L))·clamp(u, ±L))的反向: 截断处导数为 0。gate/up 是前向的输入(未截)。 */
int ds4_gpu_bwd_swiglu_tensor(ds4_gpu_tensor *ggate, ds4_gpu_tensor *gup, const ds4_gpu_tensor *gh, const ds4_gpu_tensor *gate,
                              const ds4_gpu_tensor *up, uint64_t n, float limit);
/* 路由权重(w = √softplus(z)_sel / Σ · rs)的反向: gw[n][K] → gz[n][NE](只选中项非零; 选哪几个是离散的, 不求导)。 */
int ds4_gpu_bwd_router_tensor(ds4_gpu_tensor *gz, const ds4_gpu_tensor *gw, const ds4_gpu_tensor *sel, const ds4_gpu_tensor *z,
                              uint32_t n_tok, uint32_t n_expert, uint32_t k, float route_scale);
/* routed 专家(VQ blob)的反向: gx += Σ_k ∂(w_k·E_k(x))/∂x · gy; gw[n][K] = <gy, E_k(x)>(给路由反向)。
 * 全层有 token 的专家一发: 逐对中间量 H_g/H_u/A/O 优先用本层重算截留的那份(ds4_gpu_bwd_moe_capture), 没有就用预填张量核重算;
 * 转置累加回 x 走张量核(cuda_bwd_vq.inc.cu); v2 载荷仍走直读核。增益含反修覆盖, 与前向同一份解码。 */
int ds4_gpu_bwd_routed_moe_tensor(ds4_gpu_tensor *gx, ds4_gpu_tensor *gw, const ds4_gpu_tensor *gy, const ds4_gpu_tensor *x,
                                  const ds4_gpu_tensor *sel, const ds4_gpu_tensor *rw, const void *model_map, uint64_t model_size,
                                  uint64_t blob_offset, uint64_t blob_bytes, uint32_t IN, uint32_t MID, uint32_t OUT,
                                  uint32_t n_total_expert, uint32_t K, float clamp, uint32_t layer, uint32_t n_tok);
/* 重算截留开关: on=1 期间预填专家路(v3 张量核)把 up 出口 H_u 顺手存一份, 连同它本来就留在暂存里的 H_g/A/O, 给紧接着的
 * ds4_gpu_bwd_routed_moe_tensor 直接用(同层同配对数才认, 用一次即作废)—— 省掉反传里对本层专家前向的第二遍重算。
 * 只有训练器重算那一步开, 推理路永远关着; 返回 1。 */
int ds4_gpu_bwd_moe_capture(int on);
/* 层内 bf16 权重缓存开关: mode=1 = 开始新的一层(清表并打开), 0 = 关。开着时 q4_K 稠密矩阵的预填 GEMM 把整块解出的 bf16 按权重偏移留着,
 * 同一层反传的转置乘直接用, 不再解第二遍(10-03: q4_K→bf16 占训练 GPU 时间 7.5%, 三遍里省一遍)。只有训练器"重算 + 反传"那一层开; 返回 1。 */
int ds4_gpu_bwd_wcache(int mode);
/* 梯度检查的冻结选择前向: sel 给定(基准前向那份), 按当前 logits 重算路由权重, 与 ds4_gpu_v41_router_tensor 的权重算式逐位同。 */
int ds4_gpu_bwd_router_fixed_tensor(ds4_gpu_tensor *weights, const ds4_gpu_tensor *sel, const ds4_gpu_tensor *logits,
                                    uint32_t n_tok, uint32_t n_expert, uint32_t topk, float route_scale);
/* 压缩器池化(ratio>1, 前向 ds4_gpu_v41_compress_pool_tensor)的反向: gpooled[ng][dim] → gkv/gscore 的前 ng·ratio 行(覆盖;
 * 尾行不碰, 调用方先清零)。kv/score 是前向池化的两路输入(重算得来)。 */
int ds4_gpu_bwd_compress_pool_tensor(ds4_gpu_tensor *gkv, ds4_gpu_tensor *gsc, const ds4_gpu_tensor *gpooled, const ds4_gpu_tensor *kv,
                                     const ds4_gpu_tensor *score, uint32_t n_groups, uint32_t ratio, uint32_t dim);
/* engram 门(前向 ds4_gpu_v41_engram_gate_tensor)的反向: g 原地, 进来 = 对门之后 hc 的梯度, 出去 = 对门之前 hc 的梯度。
 * hc_pre = 门之前的 hc, kv = 前向那份 [n][(HC+1)·E](key/val 只依赖 token, 当常量)。 */
int ds4_gpu_bwd_engram_gate_tensor(ds4_gpu_tensor *g, const ds4_gpu_tensor *hc_pre, const ds4_gpu_tensor *kv, const void *model_map, uint64_t model_size,
                                   uint64_t q_w_offset, uint64_t k_w_offset, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok, float eps);
/* 层入口 hc 的存档按 bf16 存(前向 hc_post/嵌入/engram 出口都舍过 bf16 ⇒ 无损, 省一半内存): pack f32 → u16, unpack u16 → f32 */
int ds4_gpu_bwd_pack_bf16_tensor(ds4_gpu_tensor *dst16, const ds4_gpu_tensor *src, uint64_t n);
int ds4_gpu_bwd_unpack_bf16_tensor(ds4_gpu_tensor *dst, const ds4_gpu_tensor *src16, uint64_t n);

/* ---- 草稿器蒸馏(2026-10-07, src/core/core_draft_kd*.c; 实现 src/cuda/cuda_draft_attn.inc.cu) ----
 * 批量全可见块注意力: R = nb·B 个 query 行, 第 r 行 = 第 r/B 块的第 r%B 位; 块 b 的键 = 块前面 nh = min(window, bpos[b]) 个历史行
 * (hist 第 bpos[b]−nh−hbase 行起, hbase = hist 第 0 行的绝对位置)+ 块内 B 行(blk 第 b·B 行起), 块内全可见。
 * 前向的分段/打分/合并/sink/bf16 出口与部署的草稿块注意力同一串积木(逐位同); 另出 lse[R][n_head] = max + log(Σexp + exp(sink−max)) 给反向。 */
int ds4_gpu_draft_attn_fwd_tensor(ds4_gpu_tensor *o, ds4_gpu_tensor *lse, const ds4_gpu_tensor *q, const ds4_gpu_tensor *hist, uint32_t hbase,
                                  const ds4_gpu_tensor *blk, const ds4_gpu_tensor *bpos, uint32_t nb, uint32_t B, uint32_t window,
                                  const void *model_map, uint64_t model_size, uint64_t sink_offset, uint32_t n_head, uint32_t head_dim, float scale);
/* 反向: gq[R][n_head][hd](覆盖), gblk[R][hd](覆盖; 块内键的梯度, 键与值两种身份合计); 历史行是常量(main_x 冻结)不出梯度。 */
int ds4_gpu_draft_attn_bwd_tensor(ds4_gpu_tensor *gq, ds4_gpu_tensor *gblk, const ds4_gpu_tensor *go, const ds4_gpu_tensor *o, const ds4_gpu_tensor *q,
                                  const ds4_gpu_tensor *lse, const ds4_gpu_tensor *hist, uint32_t hbase, const ds4_gpu_tensor *blk, const ds4_gpu_tensor *bpos,
                                  uint32_t nb, uint32_t B, uint32_t window, uint32_t n_head, uint32_t head_dim, float scale);
/* 陪审团(core_v41_dcap.c v41_dcap_jury 的设备版): m 行, 学生 ls[m][V] 与教师 lt[m][V] 各按温度 T 做 softmax,
 * out[m][4] = {Σmin(p,q), p(argmax q), argmax 同(0/1), 0} —— 分别是 草稿按分布抽 / 点质量草稿 的期望接受率与贪心一致。 */
int ds4_gpu_draft_jury_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *ls, const ds4_gpu_tensor *lt, uint32_t m, uint32_t n_vocab, float T);
/* 总变差损失(训练目标 = 1 − 期望接受率): 学生 logits[m][V] 按温度 T 做 softmax, 教师 top-K(tid/tp [m][K], K ≤ 4096)+ 余量 trest[m](榜外 p 当 0);
 * loss[m] = ½Σ|p−q|(含常数 ½r), glogits[m][V] = scale·∂loss/∂z(可与 logits 同址)。为什么不是前向 KL: 见 cuda_draft_attn.inc.cu 的实撞记录。 */
int ds4_gpu_draft_tv_tensor(ds4_gpu_tensor *glogits, ds4_gpu_tensor *loss, const ds4_gpu_tensor *logits, uint32_t m, uint32_t n_vocab,
                            const ds4_gpu_tensor *tid, const ds4_gpu_tensor *tp, const ds4_gpu_tensor *trest, uint32_t k, float T, float scale);
/* 小批(n ≤ 8)低秩件应用 y += x·(B·A): 两发小核替掉 cuBLAS 小 n 的一串小核(草稿一轮 8 发省 ~1.6 ms); 大批转 ds4_gpu_v41_amp_apply_tensor。只给草稿态用。 */
int ds4_gpu_draft_amp_apply_tensor(ds4_gpu_tensor *y, const ds4_gpu_tensor *x, const ds4_gpu_tensor *A, const ds4_gpu_tensor *B,
                                   ds4_gpu_tensor *T, uint32_t n_tok, uint32_t D, uint32_t K);
/* markov 偏置表(可训练, 设备 f32): bias[n][V] = e[n][R]·Wᵀ(W = markov_head [V][R]); 反向 gW += gᵀ·e(累加), ge = g·W; 表的取行 / 梯度按行原子散加 */
int ds4_gpu_draft_bias_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *e, const ds4_gpu_tensor *W, uint32_t n, uint32_t R, uint32_t V);
int ds4_gpu_draft_bias_bwd_tensor(ds4_gpu_tensor *gW, ds4_gpu_tensor *ge, const ds4_gpu_tensor *g, const ds4_gpu_tensor *e, const ds4_gpu_tensor *W,
                                  uint32_t n, uint32_t R, uint32_t V);
int ds4_gpu_draft_rows_dev_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *tab, const ds4_gpu_tensor *ids, uint32_t ids_off, uint32_t n, uint32_t R, uint32_t Vm);   /* 取 ids[ids_off..+n) 行 */
int ds4_gpu_draft_rows_scatter_tensor(ds4_gpu_tensor *gtab, const ds4_gpu_tensor *g, const ds4_gpu_tensor *ids, uint32_t n, uint32_t R, uint32_t Vm);
/* 批量取行: out[r][dim] = tab[ids[r]][dim](markov embed 表, bf16/f32 由 elem_bytes 2/4 说) */
int ds4_gpu_draft_rows_gather_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t tab_offset, uint64_t n_rows_tab,
                                     uint32_t dim, uint32_t elem_bytes, const ds4_gpu_tensor *ids, uint32_t n);

/* y += a·x(梯度检查的扰动用) */
int ds4_gpu_bwd_axpy_tensor(ds4_gpu_tensor *y, const ds4_gpu_tensor *x, float a, uint64_t n);

/* Σg²(梯度范数, 全局裁剪与日志用) → *out(主机)。 */
int ds4_gpu_bwd_sumsq_tensor(const ds4_gpu_tensor *g, uint64_t n, double *out);

/* Adam(带偏差校正, 无权重衰减): m = β1·m + (1−β1)·g·gs; v = β2·v + (1−β2)·(g·gs)²; p −= lr·m̂/(√v̂ + eps); 然后 g = 0。
 * gs = 全局裁剪的缩放(不裁 = 1)。step 从 1 起。 */
int ds4_gpu_bwd_adam_tensor(ds4_gpu_tensor *p, ds4_gpu_tensor *g, ds4_gpu_tensor *m, ds4_gpu_tensor *v, uint64_t n,
                            float lr, float beta1, float beta2, float eps, float gs, uint32_t step);

#ifdef __cplusplus
}
#endif

#endif
