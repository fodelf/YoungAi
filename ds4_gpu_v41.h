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
 * 两个 GPU 后端都有实现: CUDA(src/cuda/cuda_v41_*.inc.cu, 部署路)与 Metal(src/metal/metal_v41_*.m + metal/v41_*.metal, 2026-10-08 落地;
 * 没有解码整步 graph, 预填 GEMM 是权重边解边乘的 8×8 瓦片核而不是 cuBLAS ⇒ 与 CUDA 只差累加序, 门是 tests/t_metal_v41_*.c 的金标/有限差分)。 */

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

/* ---- q4_K 骨架(2026-09-19 的 100 GB 配方): 上面三支的同签名对应物 ----
 * 盘上两种骨架格式并存(fp4x32 4.25 bpw / q4_K 4.5 bpw), 调用方按张量登记类型分发(core_v41_attn.c
 * 的 v41_tproj)。q4_K 每 256 元素一块: f16 主 scale + f16 主 min + 8 组 6-bit 子 scale/min + 128 B nibble。
 * ★解码值以 src/common 的 ds4_deq_q4_K 为金标★, GPU 核与它逐式同源。 */
int ds4_gpu_v41_matmul_q4k_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                  uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                  const ds4_gpu_tensor *x, uint32_t n_tok, int round_out);
int ds4_gpu_v41_grouped_matmul_q4k_tensor(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
                                          uint64_t weight_offset, uint32_t n_groups, uint64_t group_dim,
                                          uint64_t rank, const ds4_gpu_tensor *heads, uint32_t n_tok, int round_out);
/* 合批开关(2026-10-10): 1 = 接下来的 q4_K 乘法若行数在 9..16, 走张量核小批形态(与解码 GEMV 只差加法次序); 0 = 照旧(9 行起走预填 GEMM)。
 * 只有合批步(core_v41_multi.c)在一步超 8 行时开、步末关; 没有这条路的后端(Metal)空实现返回 1。 */
int ds4_gpu_v41_set_multi_rows(int on);
int ds4_gpu_v41_embed_q4k_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens, const void *model_map,
                                 uint64_t model_size, uint64_t weight_offset, uint64_t n_vocab,
                                 uint32_t n_tok, uint64_t dim);

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

/* ★解码整步 graph 的"设备位置"口径(2026-09-18, fable5 09-18 立案; 实现 src/core/core_decode_graph.c)★
 * 一张图捕一次、每步只改设备槽里的 {token, 位置} 重放 ⇒ 凡是随位置变的核参数都不能烤进图。
 * 约定: 带 `posd` 参数的入口, posd 非 NULL 时**位置从 posd[0] 读**(int32, 设备; 就是 st->pos 那张表),
 * 主机传的 pos0 / ng / topk 只当**上限**(桶封顶: grid、shared 大小按它开; 核里自算真值:
 * ng = (pos+1)/ratio = 可见组数(n_tok=1 时与源层已完成组数相同), topk = min(topk 上限, ng))。
 * posd == NULL = 老口径, 预填与直发解码一个字不变。
 * ★n 行批也走 posd(2026-09-22, 投机验证批进图)★: 第 i 行的位置 = posd[0] + i(验证批的位置恒连续), 源层在本批之后的组数
 * = (posd[0] + n)/ratio(直发路 g0 + ng_new 的闭式, 核里 n 取 grid 的行数维), 各行可见性仍按各自位置; n ≤ 8。
 * 出错会怎样: 主机把 ng 传成真值而不是上限, 桶内位置一涨 grid 就不够, 打分核漏掉尾部的组 —— 不报错,
 * 长上下文答非所问。所以 graph 路里 ng/topk 只许由 core_decode_graph.c 按桶上限算。 */

/* indexer 打分(官方 Indexer.forward 逐式): q[n][h][dk] 与 k[ng][dk] 点积 → relu → ×weights[n][h] 求和 → score[n][ng];
 * 不可见组 -inf。
 * ★候选紧凑(2026-09-30 C2)★ cand_list(i32, 可 NULL) = 候选块核写的每行列表 [n][1+cand_cap]: 第 0 项 = 选中块数 nc, 后面 nc 个块号升序。
 * 非 NULL 时**只给候选打分**: 打分行是紧凑的 score[n][ns], ns = min(cand_cap·cand_bs, ng), 第 c 项 ↔ 组 g = list[1+c/bs]·bs + c%bs
 * (c/bs ≥ nc / g ≥ ng / g 不可见 → -inf)。以前按 [n][ng] 全宽给掩码外的组也跑一遍头循环(键取 0), 1M 上下文时 L24~36 四层各扫 100 万组
 * 而其中只有 16384 个是候选 —— 每个 (行,组) 的算式一个字没动, 只是不再给死组跑。topk 核拿同一份列表把 c 映回 g(见下)。
 * cand_bs = 候选块长(元数据 candidate.block_size), cand_cap = 块数上限(元数据 candidate.topk_blocks); 无列表时都忽略。 */
int ds4_gpu_v41_indexer_score_tensor(ds4_gpu_tensor *score, const ds4_gpu_tensor *q, const ds4_gpu_tensor *k,
                                     const ds4_gpu_tensor *weights, const ds4_gpu_tensor *cand_list, uint32_t cand_bs, uint32_t cand_cap,
                                     uint32_t n_tok, uint32_t pos0, uint32_t ng, uint32_t n_head, uint32_t dk, uint32_t ratio,
                                     const ds4_gpu_tensor *posd);

/* 候选块(官方 select_candidate_blocks): score[n][ng] → cand_list[n][1+topk_blocks] i32(★紧凑列表, 2026-09-30 C2: [0] = 选中块数, 后接升序块号★);
 * 块分 = 块内最大, 含本 query 最新位置的块钉为 +inf, 取 topk_blocks 个块(只要分 > -inf)。 */
int ds4_gpu_v41_candidate_blocks_tensor(ds4_gpu_tensor *cand_list, const ds4_gpu_tensor *score, uint32_t n_tok, uint32_t pos0,
                                        uint32_t ng, uint32_t ratio, uint32_t topk_blocks, uint32_t block_size,
                                        const ds4_gpu_tensor *posd);

/* topk(官方: topk 取 min(index_topk, 可见组数) 再按位置排序; 不可见 → -1): idx[n][topk] int32。
 * ratio 只在 posd 非 NULL 时用(核里按位置自算 ng 与 topk)。
 * cand_list 非 NULL(C2) = score 是打分核写的紧凑行 [n][ns](ns 的算法与打分核同式), 选出的第 c 项写回真组号 list[1+c/bs]·bs + c%bs;
 * 列表升序 ⇒ 紧凑序 = 位置序, 并列取小下标的规则原样成立。 */
int ds4_gpu_v41_indexer_topk_tensor(ds4_gpu_tensor *idx, const ds4_gpu_tensor *score, uint32_t n_tok, uint32_t ng,
                                    uint32_t topk, uint32_t ratio, const ds4_gpu_tensor *posd,
                                    const ds4_gpu_tensor *cand_list, uint32_t cand_bs, uint32_t cand_cap);

/* ★压缩源层的解码一步(graph 路专用; 2026-09-18 单行, 2026-09-22 扩成 n 行给投机验证批)★: 把本批 n 行的 (ckv, csc) 按批内
 * 顺序逐行追加到余行缓冲第 (pos0%ratio + i)%ratio 格; 每凑满一组就把第 0..ratio-1 格池化(与 ds4_gpu_v41_compress_pool_tensor
 * 同一条算式、同一累加序)写进 pooled[j], 新组位置 g·ratio 写进 posg[j](j < (ratio-1+n)/ratio)。没凑满的行只追加。
 * snap_kv/snap_sc 非 NULL 时把 [旧余行 | 本批 n 行] 线性存进去(与直发路的 snap_cpre 同布局, v41_spec_rollback 照旧取行)。
 * 取代直发路的"两次主机偏移 memcpy + 主机判断再发池化核": 那三件的形状随位置变, 进不了一次捕获的图。 */
int ds4_gpu_v41_compress_step_n_tensor(ds4_gpu_tensor *pooled, ds4_gpu_tensor *posg, ds4_gpu_tensor *cpre_kv, ds4_gpu_tensor *cpre_sc,
                                       ds4_gpu_tensor *snap_kv, ds4_gpu_tensor *snap_sc, const ds4_gpu_tensor *ckv, const ds4_gpu_tensor *csc,
                                       const ds4_gpu_tensor *posd, uint32_t ratio, uint32_t dim, uint32_t n);

/* 稀疏注意力(官方 sparse_attn 逐式, 在线 softmax, sink 只进分母):
 * q[n][h][d]; kv_win[(window+n)][d] 窗口缓冲: 行 r ↔ 绝对位置 pos0-window+r(前 window 行是历史, 后 n 行是本 chunk),
 * 第 i 个 query 看位置 [max(0, p+1-window), p], p = pos0+i; kv_comp[ng][d] + idx[n][topk](组号, -1 无);
 * o[n][h][d] f32(官方出 bf16, 由调用方舍)。
 * full_block≠0(DSpark 草稿块专用): 本 chunk 的 n 行**互相全可见**, 不做因果截断 —— 官方
 * get_dspark_topk_idxs 给每一位的候选都是"整个窗口 + 整块 block_size 位", 块内 5 个草稿位同时出。
 * 主路一律传 0; 传错不报错, 只是草稿位看见了未来(或主路漏看), 症状是接受率异常。
 * ring(decode.md D1): 窗口缓冲前 window 行怎么排 —— 1 = 环(位置 a 住 a % window, 官方
 * window_kv_cache[start_pos % win], 主路用); 0 = 线性排好序(DSpark 草稿塔用, 由 push_main 左移维护)。
 * 环的维护(提交/快照/回滚)是下面 ds4_gpu_v41_win_* 三件。 */
/* ratio(2026-09-16, mtp-1.md M1′): 本层压缩比(0 = 没有压缩 KV, 只有窗口)。
 * 它只有一个用途: 让解码张量核版**按每个 query 自己的绝对位置**算段长。
 * 为什么非这样不可: 这个核把键切段做在线 softmax, 分段一变累加序就变, 近平局的 token 会翻面;
 * 而"能见几个压缩组"是位置的函数 —— 纯解码在位置 p 算出来的, 必须与验证批里那个 p 算出来的一样。
 * 踩过的两个坑都在这条线上: ①段数写成 24/n_tok(批一大分段就变, 2K 第 152 字节分叉);
 * ②改成"按本批第一个 query 算"(批里第 2..k 个 query 就用错了别人的段长, 2K 第 270 字节分叉)。
 * 核里用 min(topk, (p+1)/ratio) 当该 query 的参考槽数 —— topk 是批级的上限, 取 min 之后
 * 与纯解码在同一位置算出来的值逐个相同(ng ≥ 任一 query 的可见组数)。 */
/* 出口头 W[V][D](fp4x32)的逐列平方和 → out[D]。草稿器对齐要按"出口度量"解, 见实现处的注释。 */
/* DSpark 三塔的投影: 盘上是 FP8(E4M3 + 32×32 块缩放, 原件精度) —— 引擎按张量类型分发到这里。
 * 为什么三塔不跟主干一起压 FP4: 压了草稿器就不准, 接受率直接塌(见 gguf-tools/bench/dspark_agree)。 */
int ds4_gpu_v41_matmul_fp8blk_round_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                           uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
                                           const ds4_gpu_tensor *x, uint32_t n_tok, int round_out);
int ds4_gpu_v41_grouped_matmul_fp8blk_tensor(ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
                                             uint64_t weight_offset, uint32_t n_groups, uint64_t group_dim,
                                             uint64_t rank, const ds4_gpu_tensor *heads, uint32_t n_tok, int round_out);
int ds4_gpu_v41_head_colnorm_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                    uint64_t weight_offset, uint32_t n_vocab, uint32_t n_embd);
/* 同上, 出口头是 q4_K 骨架时用(调用方按登记类型挑; 实现在 cuda_v41_q4k.inc.cu) */
int ds4_gpu_v41_head_colnorm_q4k_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                        uint64_t weight_offset, uint32_t n_vocab, uint32_t n_embd);
/* graph 路开捕获前调一次: 解码张量核注意力的局部件暂存按 段数上限 × n_tok 行 长够(捕获态下不许分配)。
 * n_tok 传这张图的行数(纯解码 1, 投机验证批 1+k), 少传一行, 验证批就会在捕获里扩容而作废整张图。 */
int ds4_gpu_v41_attn_scratch_prepare(uint32_t n_tok, uint32_t n_head, uint32_t head_dim);
/* 同上, 候选块核的 [nb] 块分 + 选中标记暂存(n_tok 行, 每行 nb 个): 捕获前按桶上限的块数长够。
 * 这两个数组 2026-09-21 从 shared 挪进了全局暂存 —— shared 那 48 KB 正是 ctx 曾经卡在 32768 的原因。 */
int ds4_gpu_v41_candidate_scratch_prepare(uint32_t n_tok, uint32_t nb);
/* ★indexer 打分上张量核(2026-09-30, cuda_v41_indexer_mma.inc.cu)★: --idx-mma 打开后 ds4_gpu_v41_indexer_score_tensor 走 s8 mma 版
 * (每块 32 维的点积精确, 只有四块相加的顺序与 CUDA 核的蝶形树不同, 详见该文件头); 默认关。它要一份每路的 q 整数尾数暂存, graph 路捕获前
 * 由 ds4_gpu_v41_indexer_scratch_prepare 按验证批行数长够(开关关着时是 no-op)。 */
void ds4_gpu_v41_set_indexer_mma(int on);
int ds4_gpu_v41_indexer_scratch_prepare(uint32_t n_tok, uint32_t n_head);
/* 后端暂存的"代号": 任何一块暂存重分配(换了指针)就 +1。解码整步 graph 烤死的是捕获时的指针, 所以发图前要对一下,
 * 代号变了必须重捕获 —— 否则图读的是已释放的页(2026-09-19 定罪的 illegal memory access, 见 cuda_v41_1.inc.cu v41_grow)。 */
uint64_t ds4_gpu_v41_scratch_generation(void);
/* 放掉全部按需长的暂存槽(返回字节数), 下次用到再长; 代号随之 +1(解码整步图会重捕获)。后训练在阶段之间调: 教师预填长出来的大槽别占着训练段 */
uint64_t ds4_gpu_v41_scratch_release(void);
uint64_t ds4_gpu_v41_scratch_bytes(void);   /* 当前暂存槽合计字节(日志用) */
/* posd / pos_cap(graph 路, 见上面"设备位置"口径): posd 非 NULL 时 pos0 = 图有效区间的起点、pos_cap = 终点(桶上限),
 * 解码张量核版按整个区间取段数上限开 grid(段长随位置变、段数不单调, 得扫一遍), 合并核按真位置只读真段。
 * 这条路只许走解码张量核版: 标量 split 版的段长在主机上按真键数算, 进不了图, 所以 posd 非 NULL 时张量核版不可用 = 失败。
 * win_lo(2026-09-21, bug.md §6.2): 窗口环里最早有效的绝对位置, 窗口下界钳到 max(p+1-window, win_lo) —— CED 跳过解码器段的
 * 块不写 L20~L39 的环, 那些槽是脏的(官方 get_window_topk_idxs 把未填槽标 -1 屏蔽, 这是同一语义)。只有预填核(标量版/预填张量核版)
 * 实现钳位; 解码核(n≤8)不实现 —— 生成路把最后一块留够 window 个位置(core_v41_api.c), 解码时钳位永远不起作用, 真起作用了这里报错停车。 */
int ds4_gpu_v41_sparse_attn_tensor(ds4_gpu_tensor *o, const ds4_gpu_tensor *q, const ds4_gpu_tensor *kv_win,
                                   const ds4_gpu_tensor *kv_comp, const ds4_gpu_tensor *idx,
                                   const void *model_map, uint64_t model_size, uint64_t sink_offset,
                                   uint32_t n_tok, uint32_t pos0, uint32_t window, uint32_t ng, uint32_t topk,
                                   uint32_t ratio,
                                   uint32_t n_head, uint32_t head_dim, float scale, int full_block, int ring,
                                   uint32_t win_lo, const ds4_gpu_tensor *posd, uint32_t pos_cap);

/* ★全局 KV 缓存的盘上尺寸(decode.md D1; 实现与所以然在 src/cuda/cuda_kv_pack.inc.cu)★
 * 这几个常量只有这一份 —— 缓存分配端(core_v41_forward.c)与核端(解包)都引它, 两边一旦各写各的
 * 就会"按 f32 分配、按打包写"越界踩别的层, 而且不报错。
 *   主 KV : 512 维 FP4 + 每 16 个一个 E4M3 缩放 = 256 + 32 = 288 B/组(与官方 890 B/token 的账同源)
 *   索引 K: 128 维 FP4 + 每 32 个一个 E8M0 缩放 =  64 +  4 =  68 B, 补 4 B 到 8 对齐 = 72 */
#define DS4_V41_CKV_BLK    16u
#define DS4_V41_CKV_NIB   256u
#define DS4_V41_CKV_BYTES 288u
#define DS4_V41_IDXK_BLK   32u
#define DS4_V41_IDXK_NIB   64u
#define DS4_V41_IDXK_BYTES 72u
/* 已 rope 的 f32 行 → 量化并打包进缓存第 g0 组起(替掉原来的"act_quant 就地 + 整行拷贝"两发)。
 * graph 路(posd 非 NULL): 组号在核里按位置算 —— 本批 nbatch 行凑满了 ng_new = (pos0%ratio + nbatch)/ratio 组, 第 r 个池化行
 * (r < n_rows ≤ (ratio-1+nbatch)/ratio)写第 pos0/ratio + r 组, r ≥ ng_new 的写进 g_trash(缓存末尾多分配的一格垃圾槽, 永远没人读)
 * ⇒ 这一发每步无条件进图, 拓扑与相位无关。nbatch=1、n_rows=1 就是纯解码整步图的原式。posd == NULL 时 nbatch 不用(传 0)。 */
int ds4_gpu_v41_ckv_pack_tensor(ds4_gpu_tensor *cache, uint32_t g0, const ds4_gpu_tensor *rows, uint32_t n_rows,
                                const ds4_gpu_tensor *posd, uint32_t ratio, uint32_t g_trash, uint32_t nbatch);
int ds4_gpu_v41_idxk_pack_tensor(ds4_gpu_tensor *cache, uint32_t g0, const ds4_gpu_tensor *rows, uint32_t n_rows,
                                 const ds4_gpu_tensor *posd, uint32_t ratio, uint32_t g_trash, uint32_t nbatch);

/* SWA 窗口环的维护(decode.md D1; 实现在 src/cuda/cuda_kv_ring.inc.cu, 那里写了每件事的所以然)。
 * commit: 本批 n 行(缓冲 [window, window+n))写进环; n>window 时只提交最后 window 行(等价)。
 * ring_snap: back=0 存下 commit 将要盖掉的格子, back=1 写回 —— 投机部分接受时只还原没被接受的那几行,
 *   i0 给"接受了几位", n 给这一批几位。snap 按批内行号存, 所以还原区间就是 [keep, n)。 */
int ds4_gpu_v41_win_commit_tensor(ds4_gpu_tensor *win, uint32_t pos0, uint32_t n, uint32_t window, uint32_t head_dim,
                                  const ds4_gpu_tensor *posd);   /* posd 非 NULL: 环格按设备位置算(graph 路) */
int ds4_gpu_v41_win_ring_snap_tensor(ds4_gpu_tensor *win, ds4_gpu_tensor *snap, uint32_t pos0,
                                     uint32_t i0, uint32_t n, uint32_t window, uint32_t head_dim, int back,
                                     const ds4_gpu_tensor *posd);   /* posd 非 NULL: pos0 从设备槽读(存那一发进投机验证批的图) */

/* 路由(官方 Gate, sqrtsoftplus): logits[n][E] f32 → probs=√softplus; (probs+bias) 选 topk;
 * weights = probs/Σ(+1e-20)·route_scale; selected[n][k] int32, weights[n][k] f32。 */
int ds4_gpu_v41_router_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, const ds4_gpu_tensor *logits,
                              const void *model_map, uint64_t model_size, uint64_t bias_offset,
                              uint32_t n_tok, uint32_t n_expert, uint32_t topk, float route_scale);

/* SwiGLU(官方 Expert: up 双向截 ±limit, gate 只截上界; h = silu(gate)·up; 结果舍 bf16) */
int ds4_gpu_v41_swiglu_tensor(ds4_gpu_tensor *h, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up,
                              uint32_t n_tok, uint32_t mid, float limit);

/* 解码小批的上限(≤ 它走融合 GEMV/VQ 即乘核, 大于它走预填 GEMM 路)。核侧 V41_GEMV_MAX_TOK 就是它; core 按它分岔 MoE 尾巴。 */
#define DS4_V41_GEMV_MAX_TOK 8u
/* routed MoE(VQ blob, 384 专家): out[n][d] = Σ_k w[n][k]·Expert_{sel[n][k]}(x[n]); 走 cuda_vq_moe_prefill_gemm
 * (按专家排序 + 逐专家 dequant f16 + cuBLAS GEMM)。blob 按模型文件偏移给, 设备指针由 range 表解析。
 * ★out == NULL(只许 n ≤ DS4_V41_GEMV_MAX_TOK)★: 不做最后那发按路由权重的归约, 专家部分和留在核侧暂存, 调用方算完 shared
 * 专家后用 ds4_gpu_v41_moe_tail_tensor 一发做 y = bf16(Σ_k w·part + so) —— 取代 reduce/copy/add/round 四发(逐位同)。 */
int ds4_gpu_v41_routed_moe_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                  uint64_t blob_offset, uint64_t blob_bytes,
                                  uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim,
                                  const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
                                  uint32_t n_total_expert, uint32_t n_expert_used, float clamp,
                                  const ds4_gpu_tensor *x, uint32_t layer, uint32_t n_tok);

int ds4_gpu_v41_moe_tail_tensor(ds4_gpu_tensor *y, const ds4_gpu_tensor *so, const ds4_gpu_tensor *weights,
                                uint32_t n_tok, uint32_t n_used, uint32_t out_dim);

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

/* ★设备采样(2026-09-28, src/cuda/cuda_v41_sample.inc.cu)★: 温度 > 0 时替掉上面的 argmax, 图与直发同一个节点位。
 * 按模型自己的分布抽(温度 / top_k / top_p / min_p 语义与主机 ds4_sample_logits 同; 不动 logits), 一行一个 block,
 * 第 i 行(i < n_rows)的结果 4 个 int32 落 out[4i..4i+3]:
 *   [0] 全分布样本   [1] 草稿是否接受(0/1)   [2] 拒绝时的残差样本(保留集去掉草稿再抽, 必 ≠ 草稿)   [3] 保留集大小(诊断)
 * 草稿 = tok[row0+i+1](验证批里下一行的输入 token), 末行没有草稿 ⇒ [1] = 0。位置从 pos[row0+i] 读(设备槽, 图路零拷贝灌进来的),
 * 随机数 = (seed, 位置, 词) 的哈希 ⇒ 不依赖线程编排、不依赖发法, 同 seed 同输入必同结果。
 * 投机: qlogits == NULL 时草稿当点质量: 接受 ⇔ 均匀数 < p(草稿), 拒绝取 [2](p 去掉草稿再归一) —— 吐出 token 的边缘分布恰是 p。
 * ★qlogits 非 NULL(2026-09-29, 草稿按分布采样)★: 第 i 行的草稿是从 qlogits 第 i 行(草稿塔 logits + markov 偏置, 同一套温度/截断)抽的,
 * 接受 ⇔ 均匀数 < min(1, p(草稿)/q(草稿)), 拒绝从残差 max(0, p − q) 归一后抽(Leviathan 2023 定理: 边缘仍恰是 p)。
 * 接受率上限从 p(argmax q) 变成 Σmin(p,q): 分布平(温 1 的英文思考段)时差得最多。qlogits 只在 row0 == 0 时给, 行数 = n_rows − 1。
 * stream: 硬币流号(0 = 目标/验证, 1 = 草稿抽样) —— 草稿与验证必须用不同的噪声, 共用会让残差与草稿相关, 分布不再精确。 */
typedef struct { float temperature, top_p, min_p; int top_k; uint64_t seed; uint32_t stream; } ds4_gpu_sample_params;
int ds4_gpu_v41_sample_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *logits, uint32_t row0, uint32_t n_rows, uint32_t n_vocab,
                              const ds4_gpu_tensor *pos, const ds4_gpu_tensor *tok, const ds4_gpu_sample_params *sp,
                              const ds4_gpu_tensor *qlogits);

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

/* 路由偏置侧车(2026-09-20, 路由反修的 V4.1 形态): 给某层路由的【选择分】加 Δb[n_expert] —— 生效 = 路由核用的 bias 换成
 * (盘上 exp_probs_b + Δb), 权重分不动(与 V4 路由偏置侧车、官方 Gate 的 bias 语义同)。按 exp_probs_b 的文件偏移认层
 * (主干 40 层与三塔各自的偏移互不相同, 路由入口签名不用改)。host_delta=NULL: bias_offset=0 全卸, 否则只卸这一层。 */
int ds4_gpu_v41_set_rb_override(const void *model_map, uint64_t model_size, uint64_t bias_offset, const float *host_delta, uint32_t n_expert);

/* ---- DSpark 草稿塔(speed.md 段 6 D1; 实现在 src/cuda/cuda_v41_draft.inc.cu) ---- */

/* ★一块最多几个草稿位(官方 5)★ —— 核与主机共用这一份。
 * 核侧要它做"一个专家最多被块里几个位选中"的上限(按专家并集的分组表按它开宽);
 * 主机侧要它开 host_ids/host_conf 那几个小数组。★两边各写一份迟早漂开, 而漂开的症状是
 * 分组表越界写 —— 不报错, 只是把别的组的 pair 号覆盖掉, 草稿悄悄变垃圾。★ 真值仍读 GGUF 元数据。 */
#define DS4_MTP_MAX_BLOCK 8u

/* 逐专家 FP4 的 dense MoE: out[n][out_dim] = Σ_k w[n][k]·Expert_{sel[n][k]}(x[n])。
 * 与主路 routed MoE 的差别只在权重来源: 那边是一整个 VQ blob, 这边是 n_expert 个独立 fp4x32 张量,
 * 所以要一张偏移表 exp_off[3][n_expert](gate/up/down 各一段, 主机数组, 内部解析成设备指针并按塔缓存)。 */
int ds4_gpu_v41_mtp_moe_tensor(ds4_gpu_tensor *out, const void *model_map, uint32_t tower, const uint64_t *exp_off,
                               uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim,
                               const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
                               uint32_t n_expert, uint32_t topk, float clamp,
                               const ds4_gpu_tensor *x, uint32_t n_tok);

/* hc 四路均值 → out 环的第 slot 段(官方 main_hiddens.append(h.mean(dim=2)))。
 * ★out 是按绝对位置定格的环★(2026-09-18): 第 t 行落 ((pos + t) % cap) 格, pos = posd ? posd[0] + src_row0 : dst_pos0
 * (graph 路位置在设备槽, 见"设备位置"口径)。为什么是环而不是"只留本块末尾几行": 草稿器的窗口要的是**最近 128 个
 * 位置**的 main_hidden, 而并不是每一步都出草稿(调度器歇轮、纯解码步) —— 只留本块的话, 歇过的那些位置就永远进不了
 * 三塔的窗口, 草稿器丢掉最近十几个 token 的上下文(实撞: 在线 p1 0.42 对教师强制 0.75, 歇 79 次)。 */
int ds4_gpu_v41_hc_mean_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *hc, uint32_t n_embd, uint32_t n_hc,
                               uint32_t n_rows, uint32_t src_row0, uint32_t slot, uint32_t n_slot,
                               uint32_t dst_pos0, uint32_t cap, const ds4_gpu_tensor *posd);
/* 从上面那个环里按位置顺序取 count 行(第 t 行 = 环的 (first_row + t) % cap 格)→ dst[t](连续), 给 main_proj 当一批输入。
 * firstd 非 NULL: 起始行从设备 int 读(草稿一轮进图时由零拷贝小核灌; 每轮的起点随上一轮接受数变, 烤进图就错)。 */
int ds4_gpu_v41_ring_rows_tensor(ds4_gpu_tensor *dst, const ds4_gpu_tensor *ring, uint32_t row_floats, uint32_t cap,
                                 uint32_t first_row, uint32_t count, const ds4_gpu_tensor *firstd);

/* 按设备上的 ids[which] 取表的一行 → out 的第 out_row 行(f32)。elem_bytes = 盘上这张表的元素字节
 * (4 = f32, 2 = bf16), 由调用方按 GGUF 登记的类型给 —— 引擎不假设。 */
int ds4_gpu_v41_row_gather_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                                  uint64_t tab_offset, uint64_t n_rows, uint32_t dim, uint32_t elem_bytes,
                                  const ds4_gpu_tensor *ids, uint32_t which, uint32_t out_row);

/* ★判负存档: markov 头的"有界剪枝精确 argmax"(mtp-2.md §6 刀 2, 2026-09-17)★
 * 逐位贪心每位要读整张 [129280][256] bf16 表(66 MB)算全词表偏置, 一轮 5 位 = 0.33 GB, 实测 1.33 ms
 * 且**已经贴着带宽墙**(248 GB/s) ⇒ 只能少读字节。想用 |bias_v| ≤ ‖W_v‖·‖e‖ 把不可能夺冠的词剪掉,
 * 只对候选精确算(候选集必含真 argmax, 不是近似)。写完实测: **候选 = 129280 / 129280, 一个都没剪掉**,
 * 草稿反而从 13.2 涨到 15.9 ms。
 * 真因: markov 表的行与 embed 近正交, 实际内积 |bias_v| 远小于 ‖W_v‖·‖e‖ —— Cauchy-Schwarz 在这里松到
 * 比 logits 的**整个动态范围**还大, 于是每个词的上界都盖过冠军的下界。
 * ⇒ 这条路对这个矩阵不成立。要省这 66 MB 只剩两条: 把表换成更低精度存(改 GGUF), 或者只对 logits 的
 * top-K 算偏置(那是近似, 但草稿不改模型输出 ⇒ 允许) —— 而它只值一轮的 1.5%, 优先级排在验证批后面。 */

/* dst 的第 dst_row 行(长 n) += src[0..n)(markov 偏置) */
int ds4_gpu_v41_row_add_tensor(ds4_gpu_tensor *dst, uint64_t dst_row, const ds4_gpu_tensor *src, uint64_t n);
/* ★markov 偏置缓存(2026-10-07, 草稿块)★: 第 i 位的偏置向量 bias = markov_head·embd[ids[i]](V 个 f32, 66 MB 的表读一遍 = 269 µs)只由 token id 决定,
 * 生成序列里同一 id 在 256 个 token 内复现 ~59%(0908 请求实测) ⇒ 按 id 缓存 N 槽(轮换淘汰)。三发都在设备上做决定(进图):
 *   lookup: hit[0] = 命中的槽号; 未命中 ⇒ 轮换领一槽记下 id, hit[0] = −(槽号+2)。  skip 版 GEMV: hit ≥ 0 就整网格直接退, 不读表。
 *   add:    命中 ⇒ logits[row] += cache[槽]; 否则 += bias 并把 bias 存进领的槽。加的是同一个向量 ⇒ 与原路(算完再 row_add)逐位同。 */
int ds4_gpu_v41_mkcache_lookup_tensor(ds4_gpu_tensor *hit, ds4_gpu_tensor *cache_ids, ds4_gpu_tensor *next, const ds4_gpu_tensor *ids,
                                      uint32_t which, uint32_t n_slots);
int ds4_gpu_v41_mkcache_add_tensor(ds4_gpu_tensor *logits, uint64_t row, const ds4_gpu_tensor *bias, ds4_gpu_tensor *cache,
                                   const ds4_gpu_tensor *hit, uint64_t n);
int ds4_gpu_v41_matmul_bf16_skip_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset,
                                        uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint32_t n_tok, const ds4_gpu_tensor *skip);

#ifdef __cplusplus
}
#endif

#endif
