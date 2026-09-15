#ifndef DS4_GPU_V41_H
#define DS4_GPU_V41_H

#include <stdbool.h>
#include <stdint.h>

#include "ds4_gpu_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * DeepSeek V4.1 Flash 批前向原语(2026-09-12 战役 P2)。
 * =========================================================================
 *
 * 口径: 全部按官方 inference/model.py 逐式转录(那是 V4.1 唯一 ground truth), 计算 f32,
 * 在官方 bf16 模块边界处显式舍到 bf16(ds4_gpu_v41_round_bf16), 这样引擎 logits 与 Python 学生
 * (同一文件)可逐位置对拍到 KL 1e-4 量级 —— 这是 P2 的判决尺。激活张量一律 f32 行主序 [n, dim]。
 * 权重类型: fp4x32(type 43, 见 ds4_quantfmt.h) / f32 / f16 / VQ blob(DQVL v2)。
 * 只有 GPU 实现(CUDA); Metal 后置。 */

/* out[n][out_dim] = x[n][in_dim] · Wᵀ, W 是 fp4x32 [out_dim 行][in_dim]。
 * n ≤ 8: 融合 GEMV(权重直接解码即乘, 不落 f16); 大批: 块解码成 f16 暂存 + cuBLAS GEMM(f32 累加)。
 * round_out: 结果舍 bf16(官方 Linear 出 bf16; 小批并进 GEMV 尾巴, 大批补一发 round 核)。 */
int ds4_gpu_v41_matmul_fp4x32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                     uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                     const ds4_gpu_tensor *x, uint32_t n_tok, int round_out);

/* 分组块对角投影(attn 的 wo_a): heads[n][n_groups*group_dim] → low[n][n_groups*rank],
 * 第 g 组只乘自己那段: low[:, g*rank:(g+1)*rank] = heads[:, g*group_dim:(g+1)*group_dim] · W_gᵀ,
 * W 是 fp4x32 [n_groups*rank 行][group_dim]。round_out 同上。 */
int ds4_gpu_v41_grouped_matmul_fp4x32_tensor(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
                                             uint64_t weight_offset, uint32_t n_groups, uint64_t group_dim,
                                             uint64_t rank, const ds4_gpu_tensor *heads, uint32_t n_tok, int round_out);

/* fp4x32 嵌入: 取 tokens[n] 行 → out[n][n_embd] f32(再由调用方展成 hc 份) */
int ds4_gpu_v41_embed_fp4x32_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens, const void *model_map,
                                    uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab,
                                    uint32_t n_tok, uint32_t n_embd);

/* x[n] 逐元素舍到 bf16 再回 f32(官方 bf16 模块边界) */
int ds4_gpu_v41_round_bf16_tensor(ds4_gpu_tensor *x, uint64_t n);

/* x[n][n_hc*n_embd] → mix[n][mix_hc]; mix = (flatten(x)·rsqrt(mean(x²)+eps)) · Wᵀ, W f32 [mix_hc][n_hc*n_embd] */
int ds4_gpu_v41_hc_mix_tensor(ds4_gpu_tensor *mix, const ds4_gpu_tensor *hc, const void *model_map, uint64_t model_size,
                              uint64_t fn_offset, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok, float eps);

/* sinkhorn 拆分(官方 hc_split_sinkhorn 逐式): pre[n][hc] post[n][hc] comb[n][hc][hc] */
int ds4_gpu_v41_hc_split_tensor(ds4_gpu_tensor *pre, ds4_gpu_tensor *post, ds4_gpu_tensor *comb,
                                const ds4_gpu_tensor *mix, const void *model_map, uint64_t model_size,
                                uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, uint32_t iters,
                                float eps, uint32_t n_tok);

/* ★合一(single.md S3)★: hc_split + hc_pre + rms_norm 一发做完。解码一步有约 900 发这种"读几十 KB、
 * 花几微秒"的小核, 时间全在启动与排空。三件是同一条串行链, 中间量照旧写回(别人还要用), 只是不再各起一发。
 * pre/post/comb = 本层 split 的输出; pre_in = **上一层**传下来的 pre(hc_pre 用它, 别用错); x/xn = 出口。
 * 数值与三发版逐位同(sinkhorn 同序、c 循环同序、rms 归约同形状)。 */
int ds4_gpu_v41_hc_fused_tensor(ds4_gpu_tensor *pre, ds4_gpu_tensor *post, ds4_gpu_tensor *comb,
                                ds4_gpu_tensor *x, ds4_gpu_tensor *xn, const ds4_gpu_tensor *mix,
                                const ds4_gpu_tensor *hc, const ds4_gpu_tensor *pre_in,
                                const void *model_map, uint64_t model_size,
                                uint64_t scale_offset, uint64_t base_offset, uint64_t norm_offset,
                                uint32_t n_embd, uint32_t n_hc, uint32_t iters, float hc_eps, float norm_eps,
                                uint32_t n_tok);

/* hc_pre: out[n][d] = Σ_c pre[n][c]·hc[n][c][d] (f32 累加, 结果舍 bf16) */
int ds4_gpu_v41_hc_pre_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *hc, const ds4_gpu_tensor *pre,
                              uint32_t n_embd, uint32_t n_hc, uint32_t n_tok);

/* hc_post: out_hc[n][c][d] = post[n][c]·y[n][d] + Σ_s comb[n][c][s]·res[n][s][d] (舍 bf16) */
int ds4_gpu_v41_hc_post_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *y, const ds4_gpu_tensor *res,
                               const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb,
                               uint32_t n_embd, uint32_t n_hc, uint32_t n_tok);

/* RMSNorm 带权(官方 RMSNorm: f32 算, 结果舍 bf16): out[n][d] = w[d]·x/√(mean x²+eps) */
int ds4_gpu_v41_rms_norm_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map,
                                uint64_t model_size, uint64_t weight_offset, uint32_t dim, uint32_t n_tok, float eps);

/* RoPE(官方 apply_rotary_emb 逐式: 相邻两元素为一复数, 只转每头末 n_rot 维)。
 * x[n][n_head][head_dim] f32, 位置 pos[n](int32, 设备) —— 压缩 latent 的位置是 g*ratio, 所以按表给。
 * yarn: original_seq_len>0 开 YaRN(官方 precompute_freqs_cis 的 ramp), 否则纯 theta。inverse=共轭旋转。 */
int ds4_gpu_v41_rope_tensor(ds4_gpu_tensor *x, const ds4_gpu_tensor *pos, uint32_t n_tok, uint32_t n_head,
                            uint32_t head_dim, uint32_t n_rot, float theta, uint32_t original_seq_len,
                            float factor, float beta_fast, float beta_slow, bool inverse);

/* 官方 act_quant(inplace, ue8m0): 每 block 个元素一组, s = 2^ceil(log2(amax/448)), y = e4m3(x/s)·s;
 * fp4_act_quant(inplace): ue8m0 版 s = 2^ceil(log2(amax/6)); e4m3 版 s = e4m3(amax/6)(amax 下限 6·2^-9)。
 * x[n][dim] f32 就地改写, 值落回 bf16 格点(官方输出 dtype 是 bf16)。 */
int ds4_gpu_v41_act_quant_fp8_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t dim, uint32_t block);
int ds4_gpu_v41_act_quant_fp4_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t dim, uint32_t block, bool e4m3_scale);

/* 压缩器池化(官方 Compressor, ratio>1): kv[n][d], score[n][d] → out[n/ratio][d] = Σ_t softmax_t(score)·kv,
 * 组内 softmax 逐维; 尾巴不满一组丢弃(prefill start_pos=0 语义)。 */
int ds4_gpu_v41_compress_pool_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *kv, const ds4_gpu_tensor *score,
                                     uint32_t n_tok, uint32_t ratio, uint32_t dim);

/* 【增量口径(P2c)】下面三件都带 pos0 = 本 chunk 第 i 个 query 的绝对位置 pos0+i; k/kv_comp 是源层的整段缓存
 * (行 g = 第 g 个压缩组, 共 ng 组 = 含本 chunk 新完成的组); 可见性按绝对位置算: g < (pos0+i+1)/ratio。 */

/* indexer 打分(官方 Indexer.forward 逐式): q[n][h][dk] 与 k[ng][dk] 点积 → relu → ×weights[n][h] 求和 → score[n][ng];
 * 不可见组 -inf。cand_mask[n][ng](u8, 可 NULL) 为 0 的位置置 -inf(两级 topk 第二级)。 */
int ds4_gpu_v41_indexer_score_tensor(ds4_gpu_tensor *score, const ds4_gpu_tensor *q, const ds4_gpu_tensor *k,
                                     const ds4_gpu_tensor *weights, const ds4_gpu_tensor *cand_mask,
                                     uint32_t n_tok, uint32_t pos0, uint32_t ng, uint32_t n_head, uint32_t dk, uint32_t ratio);

/* 候选块(官方 select_candidate_blocks): score[n][ng] → mask[n][ng] u8; 块分 = 块内最大, 含本 query 最新位置
 * 的块钉为 +inf, 取 topk_blocks 个块(只要分 > -inf), 展开到位置。 */
int ds4_gpu_v41_candidate_blocks_tensor(ds4_gpu_tensor *mask, const ds4_gpu_tensor *score, uint32_t n_tok, uint32_t pos0,
                                        uint32_t ng, uint32_t ratio, uint32_t topk_blocks, uint32_t block_size);

/* topk(官方: topk 取 min(index_topk, 可见组数) 再按位置排序; 不可见 → -1): idx[n][topk] int32 */
int ds4_gpu_v41_indexer_topk_tensor(ds4_gpu_tensor *idx, const ds4_gpu_tensor *score, uint32_t n_tok, uint32_t ng,
                                    uint32_t topk);

/* 稀疏注意力(官方 sparse_attn 逐式, 在线 softmax, sink 只进分母):
 * q[n][h][d]; kv_win[(window+n)][d] 窗口缓冲: 行 r ↔ 绝对位置 pos0-window+r(前 window 行是历史, 后 n 行是本 chunk),
 * 第 i 个 query 看位置 [max(0, p+1-window), p], p = pos0+i; kv_comp[ng][d] + idx[n][topk](组号, -1 无);
 * o[n][h][d] f32(官方出 bf16, 由调用方舍)。
 * full_block≠0(DSpark 草稿块专用): 本 chunk 的 n 行**互相全可见**, 不做因果截断 —— 官方
 * get_dspark_topk_idxs 给每一位的候选都是"整个窗口 + 整块 block_size 位", 块内 5 个草稿位同时出。
 * 主路一律传 0; 传错不报错, 只是草稿位看见了未来(或主路漏看), 症状是接受率异常。 */
int ds4_gpu_v41_sparse_attn_tensor(ds4_gpu_tensor *o, const ds4_gpu_tensor *q, const ds4_gpu_tensor *kv_win,
                                   const ds4_gpu_tensor *kv_comp, const ds4_gpu_tensor *idx,
                                   const void *model_map, uint64_t model_size, uint64_t sink_offset,
                                   uint32_t n_tok, uint32_t pos0, uint32_t window, uint32_t ng, uint32_t topk,
                                   uint32_t n_head, uint32_t head_dim, float scale, int full_block);

/* 路由(官方 Gate, sqrtsoftplus): logits[n][E] f32 → probs=√softplus; (probs+bias) 选 topk;
 * weights = probs/Σ(+1e-20)·route_scale; selected[n][k] int32, weights[n][k] f32。 */
int ds4_gpu_v41_router_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, const ds4_gpu_tensor *logits,
                              const void *model_map, uint64_t model_size, uint64_t bias_offset,
                              uint32_t n_tok, uint32_t n_expert, uint32_t topk, float route_scale);

/* SwiGLU(官方 Expert: up 双向截 ±limit, gate 只截上界; h = silu(gate)·up; 结果舍 bf16) */
int ds4_gpu_v41_swiglu_tensor(ds4_gpu_tensor *h, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
                              uint32_t n_tok, uint32_t mid, float limit);

/* routed MoE(VQ blob, 384 专家): out[n][d] = Σ_k w[n][k]·Expert_{sel[n][k]}(x[n]); 走 cuda_vq_moe_prefill_gemm
 * (按专家排序 + 逐专家 dequant f16 + cuBLAS GEMM)。blob 按模型文件偏移给, 设备指针由 range 表解析。 */
int ds4_gpu_v41_routed_moe_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                  uint64_t blob_offset, uint64_t blob_bytes,
                                  uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim,
                                  const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
                                  uint32_t n_total_expert, uint32_t n_expert_used, float clamp,
                                  const ds4_gpu_tensor *x, uint32_t layer, uint32_t n_tok);

/* f32 权重 GEMM(hc_fn / 压缩器 / 路由 gate_inp / indexer proj,wk / engram q,k): out[n][out_dim] = x·Wᵀ, 纯 f32 */
int ds4_gpu_v41_matmul_f32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                  uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                  const ds4_gpu_tensor *x, uint32_t n_tok);
/* BF16 权重(官方原生精度)版; clear.md C1 起 gate/compressor/indexer 走这条 */
int ds4_gpu_v41_matmul_bf16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                  uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                  const ds4_gpu_tensor *x, uint32_t n_tok);
/* engram wkv: FP8 e4m3 + 32×32 块 ue8m0(官方盘上格式); clear.md C1 */
int ds4_gpu_v41_matmul_fp8blk_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                  uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                  const ds4_gpu_tensor *x, uint32_t n_tok);

/* engram 门(官方 Engram.forward 后半): kv[n][(n_hc+1)·E] = [key_0..key_{hc-1} | value](wkv 输出, bf16 格点);
 * 对每 (token, 路 c): w = q_w[c]·k_w[c](逐维积); dot = Σ_d h[c][d]·w[d]·key[c][d] · rsqrt(mean h²+eps) · rsqrt(mean key²+eps) · E^-½;
 * gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)); h[c] = bf16(h[c] + gate·value)。原地改 hc。 */
int ds4_gpu_v41_engram_gate_tensor(ds4_gpu_tensor *hc, const ds4_gpu_tensor *kv, const void *model_map, uint64_t model_size,
                                   uint64_t q_w_offset, uint64_t k_w_offset, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok, float eps);

/* a[n] += b[n] (f32) */
int ds4_gpu_v41_add_tensor(ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint64_t n);

/* engram 查表行 dequant: raw[n_rows][head_dim + head_dim/32](前 head_dim 字节 fp8 e4m3, 后面每 32 维一个 ue8m0 scale)
 * → out[n_rows][head_dim] f32(bf16 格点)。行字节由主机并行 pread 拉来(表 98 GB 在盘, 随机行读才是瓶颈)。 */
int ds4_gpu_v41_engram_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *raw, uint32_t n_rows, uint32_t head_dim);

/* 反修放大器应用: y[n][D] += x[n][D]·(B·A), A/B 设备 f32 张量([K][D] 行主序; B 即数学 D×K 的列主序), T 是 [n][K] 中间。
 * 两发 cuBLAS Sgemm: T = x·B, y += T·A(f32 累加进 y; 调用方随后舍 bf16)。 */
int ds4_gpu_v41_amp_apply_tensor(ds4_gpu_tensor *y, const ds4_gpu_tensor *x, const ds4_gpu_tensor *A, const ds4_gpu_tensor *B,
                                 ds4_gpu_tensor *T, uint32_t n_tok, uint32_t D, uint32_t K);

/* 就地 x[n] = bf16(x·s)(indexer weights 的 softmax_scale·n_heads^-½, 官方在 bf16 上乘) */
int ds4_gpu_v41_scale_round_tensor(ds4_gpu_tensor *x, uint64_t n, float s);

/* argmax(logits 第 row 行, n_vocab) → idx[0] int32(同值取小下标) */
int ds4_gpu_v41_argmax_tensor(ds4_gpu_tensor *idx, const ds4_gpu_tensor *logits, uint32_t row, uint32_t n_vocab);

/* 行复制/展开: hc[n][n_hc][d] = x[n][d] 每份一样 */
int ds4_gpu_v41_expand_hc_tensor(ds4_gpu_tensor *hc, const ds4_gpu_tensor *x, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok);

/* 反修取料(2026-09-13, 权重侧逐专家反修): 把刚算完的这一层 MoE 的【逐专家 down 输出】展开成
 * 稠密 out[n_tok][n_used][OUT] f32 下主机。这是 reduce 前的量 —— reduce 做的是
 * out[t][o] = Σ_k rw[t][k]·ys[inv[t][k]][o], 本函数给的就是那个 ys[inv[t][k]][o](未乘 rw)。
 * 解 g_r(逐专家逐输出通道增益)要它: y[t][o] = Σ_k rw·g_e[o]·(ys/g_e旧)[o], 逐 o 独立。
 * 缺席的配对(inv<0)填 0。★只对 prefill GEMM 路(n_tok>8)有效★ —— 解码 gemv 路不物化这个中间量,
 * 返回 0。取料走 --score-ids(chunk 512) 本来就是 prefill 路, 与判决同路。 */
int ds4_gpu_v41_vq_capture_expert_out(float *host, uint32_t n_tok, uint32_t n_used, uint32_t out_dim);

/* 权重侧反修(2026-09-13): 给某层挂上 down 矩阵的逐专家逐输出通道增益【缩放因子】 s[n_expert][out_dim] f32。
 * 生效方式 = VQ 解码时 g_eff[row] = 载荷里的 g_r[row] × s[e][row] —— 盘上权重一个字节不动, 不挂就是裸底座。
 * host=NULL 卸掉该层。解码 gemv 路与 prefill GEMM 路都吃这张表(两路同式, 判决与部署不分叉)。 */
int ds4_gpu_v41_set_gr_override(uint32_t layer, const float *host, uint32_t n_expert, uint32_t out_dim);

/* ---- DSpark 草稿塔(speed.md 段 6 D1; 实现在 src/cuda/cuda_v41_draft.inc.cu) ---- */

/* 逐专家 FP4 的 dense MoE: out[n][out_dim] = Σ_k w[n][k]·Expert_{sel[n][k]}(x[n])。
 * 与主路 routed MoE 的差别只在权重来源: 那边是一整个 VQ blob, 这边是 n_expert 个独立 fp4x32 张量,
 * 所以要一张偏移表 exp_off[3][n_expert](gate/up/down 各一段, 主机数组, 内部解析成设备指针并按塔缓存)。 */
int ds4_gpu_v41_mtp_moe_tensor(ds4_gpu_tensor *out, const void *model_map, uint32_t tower, const uint64_t *exp_off,
                               uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim,
                               const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
                               uint32_t n_expert, uint32_t topk, float clamp,
                               const ds4_gpu_tensor *x, uint32_t n_tok);

/* hc 四路均值 → out[n][n_slot][E] 的第 slot 段(官方 main_hiddens.append(h.mean(dim=2))) */
int ds4_gpu_v41_hc_mean_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *hc, uint32_t n_embd, uint32_t n_hc,
                               uint32_t n_rows, uint32_t src_row0, uint32_t slot, uint32_t n_slot);

/* 按设备上的 ids[which] 取表的一行 → out 的第 out_row 行(f32)。elem_bytes = 盘上这张表的元素字节
 * (4 = f32, 2 = bf16), 由调用方按 GGUF 登记的类型给 —— 引擎不假设。 */
int ds4_gpu_v41_row_gather_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                  uint64_t tab_offset, uint64_t n_rows, uint32_t dim, uint32_t elem_bytes,
                                  const ds4_gpu_tensor *ids, uint32_t which, uint32_t out_row);

/* dst 的第 dst_row 行(长 n) += src[0..n)(markov 偏置) */
int ds4_gpu_v41_row_add_tensor(ds4_gpu_tensor *dst, uint64_t dst_row, const ds4_gpu_tensor *src, uint64_t n);

#ifdef __cplusplus
}
#endif

#endif
