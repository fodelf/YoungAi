/* metal_v41_attn.m — V4.1 稀疏注意力 / indexer 三件 / 设备采样的 Metal 发射(2026-10-08)。契约 ds4_gpu_v41.h; 核在 metal/v41_attn.metal、v41_sample.metal。
 * Metal 没有解码整步 graph, 所以 posd 口径只是"位置从设备读"(核里支持), 暂存 prepare 三件在这里是按上限先长够(语义同 CUDA)。 */
#import "metal_v41.h"

static v41_scratch g_v41_cand_blk;   /* 候选块的 [n][nb] 分 + [n][nb] 标记 */

int ds4_gpu_v41_sparse_attn_tensor(ds4_gpu_tensor *o, const ds4_gpu_tensor *q, const ds4_gpu_tensor *kv_win, const ds4_gpu_tensor *kv_comp, const ds4_gpu_tensor *idx,
                                   const void *model_map, uint64_t model_size, uint64_t sink_offset, uint32_t n_tok, uint32_t pos0, uint32_t window, uint32_t ng,
                                   uint32_t topk, uint32_t ratio, uint32_t n_head, uint32_t head_dim, float scale, int full_block, int ring, uint32_t win_lo,
                                   const ds4_gpu_tensor *posd, uint32_t pos_cap) {
    (void)ratio; (void)pos_cap;
    if (!o || !q || !kv_win || head_dim != 512u || (n_head % 8u) || n_tok == 0) { fprintf(stderr, "ds4: [v41-metal] sparse attn 只实现 head_dim 512、头数 8 的倍数\n"); return 0; }
    if (!full_block && !ring) { fprintf(stderr, "ds4: [v41-metal] sparse attn: 主路窗口必须是环(decode.md D1)\n"); return 0; }
    if (ds4_gpu_tensor_bytes(kv_win) < (uint64_t)(window + n_tok) * head_dim * 4) { fprintf(stderr, "ds4: [v41-metal] 窗口缓冲不够 %u+%u 行\n", window, n_tok); return 0; }
    uint64_t so = 0;
    id<MTLBuffer> sb = v41_model_buf(model_map, model_size, sink_offset, (uint64_t)n_head * 4, &so, "v41 sink");
    if (!sb) return 0;
    const int hasc = kv_comp && idx;
    v41_attn_args a = { pos0, window, hasc ? ng : 0u, hasc ? topk : 0u, n_head, head_dim, full_block ? n_tok : 0u, ring ? 1u : 0u, win_lo, posd ? 1u : 0u, hasc ? 1u : 0u, 0,
                        scale, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(o), V41_T(q), V41_T(kv_win), V41_T(kv_comp), V41_T(idx), V41_B(sb, (NSUInteger)so), V41_T(posd) };
    return v41_launch("kernel_v41_sparse_attn", b, 8, MTLSizeMake(n_tok, n_head / 8u, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_v41_attn_scratch_prepare(uint32_t n_tok, uint32_t n_head, uint32_t head_dim) { (void)n_tok; (void)n_head; (void)head_dim; return 1; }   /* 标量核不用暂存 */
int ds4_gpu_v41_indexer_scratch_prepare(uint32_t n_tok, uint32_t n_head) { (void)n_tok; (void)n_head; return 1; }
void ds4_gpu_v41_set_indexer_mma(int on) { if (on) fprintf(stderr, "ds4: [v41-metal] --idx-mma 在 Metal 上没有张量核打分版, 走标量核\n"); }

static uint32_t v41_cand_ns(uint32_t ng, uint32_t bs, uint32_t cap) { const uint64_t full = (uint64_t)cap * bs; return full < ng ? (uint32_t)full : ng; }
int ds4_gpu_v41_indexer_score_tensor(ds4_gpu_tensor *score, const ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *weights, const ds4_gpu_tensor *cand_list,
                                     uint32_t cand_bs, uint32_t cand_cap, uint32_t n_tok, uint32_t pos0, uint32_t ng, uint32_t n_head, uint32_t dk, uint32_t ratio,
                                     const ds4_gpu_tensor *posd) {
    if (!score || !q || !k || !weights || (dk % 32u) || ratio == 0 || n_tok == 0) return 0;
    if (dk / 32u > 4u) { fprintf(stderr, "ds4: [v41-metal] indexer 打分核只实现 dk ≤ 128\n"); return 0; }
    if (posd && n_tok > 8u) return 0;
    if (ng == 0) return 1;
    if (cand_list && (cand_bs == 0u || cand_cap == 0u)) return 0;
    const uint32_t ns = cand_list ? v41_cand_ns(ng, cand_bs, cand_cap) : ng;
    uint32_t gblocks = (ns + 7u) / 8u;
    if (gblocks > 65535u) gblocks = 65535u;
    v41_idx_args a = { pos0, ng, n_head, dk, ratio, posd ? 1u : 0u, cand_list ? 1u : 0u, cand_bs, cand_cap, n_tok, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(score), V41_T(q), V41_T(k), V41_T(weights), V41_T(cand_list), V41_T(posd) };
    return v41_launch("kernel_v41_indexer_score", b, 7, MTLSizeMake(n_tok, gblocks, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_v41_candidate_scratch_prepare(uint32_t n_tok, uint32_t nb) {
    if (n_tok == 0u || nb == 0u) return 1;
    return v41_grow(&g_v41_cand_blk, (uint64_t)n_tok * nb * 5u, "v41 候选块") ? 1 : 0;
}
int ds4_gpu_v41_candidate_blocks_tensor(ds4_gpu_tensor *cand_list, const ds4_gpu_tensor *score, uint32_t n_tok, uint32_t pos0, uint32_t ng, uint32_t ratio,
                                        uint32_t topk_blocks, uint32_t block_size, const ds4_gpu_tensor *posd) {
    if (!cand_list || !score || block_size == 0 || ratio == 0 || topk_blocks == 0 || n_tok == 0) return 0;
    if (posd && n_tok > 8u) return 0;
    if (ng == 0) return 1;
    const uint32_t nb = (ng + block_size - 1u) / block_size;
    id<MTLBuffer> blk = v41_grow(&g_v41_cand_blk, (uint64_t)n_tok * nb * 5u, "v41 候选块");
    if (!blk) return 0;
    v41_cand_args a = { pos0, ng, ratio, topk_blocks, block_size, posd ? 1u : 0u, n_tok, nb };
    v41_bind b[] = { V41_A(a), V41_T(cand_list), V41_T(score), V41_T(posd), V41_B(blk, 0) };
    return v41_launch("kernel_v41_candidate", b, 5, MTLSizeMake(n_tok, 1, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_v41_indexer_topk_tensor(ds4_gpu_tensor *idx, const ds4_gpu_tensor *score, uint32_t n_tok, uint32_t ng, uint32_t topk, uint32_t ratio, const ds4_gpu_tensor *posd,
                                    const ds4_gpu_tensor *cand_list, uint32_t cand_bs, uint32_t cand_cap) {
    if (!idx || !score || topk == 0 || n_tok == 0) return 0;
    if (posd && (n_tok > 8u || ratio == 0u)) return 0;
    if (cand_list && (cand_bs == 0u || cand_cap == 0u)) return 0;
    if (ng == 0) return 1;
    v41_idx_args a = { 0, ng, 0, 0, ratio, posd ? 1u : 0u, cand_list ? 1u : 0u, cand_bs, cand_cap, n_tok, topk, 0 };
    v41_bind b[] = { V41_A(a), V41_T(idx), V41_T(score), V41_T(posd), V41_T(cand_list) };
    return v41_launch("kernel_v41_topk", b, 5, MTLSizeMake(n_tok, 1, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_v41_sample_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *logits, uint32_t row0, uint32_t n_rows, uint32_t n_vocab, const ds4_gpu_tensor *pos,
                              const ds4_gpu_tensor *tok, const ds4_gpu_sample_params *sp, const ds4_gpu_tensor *qlogits) {
    if (!out || !logits || !pos || !tok || !sp || n_rows == 0u || n_vocab == 0u) return 0;
    if (!(sp->temperature > 0.0f)) return 0;
    const uint64_t last = (uint64_t)row0 + n_rows;
    if (ds4_gpu_tensor_bytes(out) < (uint64_t)n_rows * 16u || ds4_gpu_tensor_bytes(logits) < last * n_vocab * 4u || ds4_gpu_tensor_bytes(pos) < last * 4u ||
        ds4_gpu_tensor_bytes(tok) < last * 4u) return 0;
    if (qlogits && (row0 != 0u || (n_rows > 1u && ds4_gpu_tensor_bytes(qlogits) < (uint64_t)(n_rows - 1u) * n_vocab * 4u))) return 0;
    float top_p = sp->top_p; if (!(top_p > 0.0f) || top_p > 1.0f) top_p = 1.0f;
    v41_sample_args a;
    memset(&a, 0, sizeof a);
    a.seed = sp->seed; a.V = n_vocab; a.row0 = row0; a.n_rows = n_rows; a.top_k = sp->top_k > 0 ? (uint32_t)sp->top_k : 0u; a.stream = sp->stream;
    a.has_q = qlogits ? 1u : 0u; a.inv_T = 1.0f / sp->temperature; a.min_p = sp->min_p > 0.0f ? sp->min_p : 0.0f; a.top_p = top_p;
    v41_bind b[] = { V41_A(a), V41_T(out), V41_T(logits), V41_T(qlogits), V41_T(pos), V41_T(tok) };
    return v41_launch("kernel_v41_sample", b, 6, MTLSizeMake(n_rows, 1, 1), MTLSizeMake(1024, 1, 1));
}
