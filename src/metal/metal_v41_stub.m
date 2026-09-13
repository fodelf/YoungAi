/* metal_v41_stub.m — DeepSeek V4.1 批前向原语在 Metal 后端的占位(2026-09-12)。
 * 战役 P2 只做 CUDA(spark 是部署机); Metal 实现是后续工序。这里每个入口返回 0(失败)并打一句话,
 * 让 Mac 构建能链接, 而 V4.1 模型在 Mac 上跑到前向时明确报"未实现"而不是静默出垃圾。 */
#include "ds4_gpu.h"
#include <stdio.h>
#pragma clang diagnostic ignored "-Wunused-parameter"   /* 占位函数按契约签名声明, 参数必然不用 */

static int v41_metal_unimplemented(const char *what) {
    fprintf(stderr, "ds4: Metal 后端尚未实现 V4.1 原语 %s(只有 CUDA), 停车\n", what);
    return 0;
}
#define V41_STUB(name, ...) int name(__VA_ARGS__) { return v41_metal_unimplemented(#name); }

V41_STUB(ds4_gpu_v41_matmul_fp4x32_tensor, ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint32_t n_tok, int round_out)
V41_STUB(ds4_gpu_v41_grouped_matmul_fp4x32_tensor, ds4_gpu_tensor *low, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_groups, uint64_t group_dim, uint64_t rank, const ds4_gpu_tensor *heads, uint32_t n_tok, int round_out)
V41_STUB(ds4_gpu_v41_embed_fp4x32_tensor, ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t n_tok, uint32_t n_embd)
V41_STUB(ds4_gpu_v41_round_bf16_tensor, ds4_gpu_tensor *x, uint64_t n)
V41_STUB(ds4_gpu_v41_hc_mix_tensor, ds4_gpu_tensor *mix, const ds4_gpu_tensor *hc, const void *model_map, uint64_t model_size, uint64_t fn_offset, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok, float eps)
V41_STUB(ds4_gpu_v41_hc_split_tensor, ds4_gpu_tensor *pre, ds4_gpu_tensor *post, ds4_gpu_tensor *comb, const ds4_gpu_tensor *mix, const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, uint32_t iters, float eps, uint32_t n_tok)
V41_STUB(ds4_gpu_v41_hc_pre_tensor, ds4_gpu_tensor *out, const ds4_gpu_tensor *hc, const ds4_gpu_tensor *pre, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok)
V41_STUB(ds4_gpu_v41_hc_post_tensor, ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *y, const ds4_gpu_tensor *res, const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok)
V41_STUB(ds4_gpu_v41_rms_norm_tensor, ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t dim, uint32_t n_tok, float eps)
V41_STUB(ds4_gpu_v41_rope_tensor, ds4_gpu_tensor *x, const ds4_gpu_tensor *pos, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, float theta, uint32_t original_seq_len, float factor, float beta_fast, float beta_slow, bool inverse)
V41_STUB(ds4_gpu_v41_act_quant_fp8_tensor, ds4_gpu_tensor *x, uint32_t n_rows, uint32_t dim, uint32_t block)
V41_STUB(ds4_gpu_v41_act_quant_fp4_tensor, ds4_gpu_tensor *x, uint32_t n_rows, uint32_t dim, uint32_t block, bool e4m3_scale)
V41_STUB(ds4_gpu_v41_compress_pool_tensor, ds4_gpu_tensor *out, const ds4_gpu_tensor *kv, const ds4_gpu_tensor *score, uint32_t n_tok, uint32_t ratio, uint32_t dim)
V41_STUB(ds4_gpu_v41_indexer_score_tensor, ds4_gpu_tensor *score, const ds4_gpu_tensor *q, const ds4_gpu_tensor *k, const ds4_gpu_tensor *weights, const ds4_gpu_tensor *cand_mask, uint32_t n_tok, uint32_t pos0, uint32_t ng, uint32_t n_head, uint32_t dk, uint32_t ratio)
V41_STUB(ds4_gpu_v41_candidate_blocks_tensor, ds4_gpu_tensor *mask, const ds4_gpu_tensor *score, uint32_t n_tok, uint32_t pos0, uint32_t ng, uint32_t ratio, uint32_t topk_blocks, uint32_t block_size)
V41_STUB(ds4_gpu_v41_indexer_topk_tensor, ds4_gpu_tensor *idx, const ds4_gpu_tensor *score, uint32_t n_tok, uint32_t ng, uint32_t topk)
V41_STUB(ds4_gpu_v41_sparse_attn_tensor, ds4_gpu_tensor *o, const ds4_gpu_tensor *q, const ds4_gpu_tensor *kv_win, const ds4_gpu_tensor *kv_comp, const ds4_gpu_tensor *idx, const void *model_map, uint64_t model_size, uint64_t sink_offset, uint32_t n_tok, uint32_t pos0, uint32_t window, uint32_t ng, uint32_t topk, uint32_t n_head, uint32_t head_dim, float scale)
V41_STUB(ds4_gpu_v41_router_tensor, ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, const ds4_gpu_tensor *logits, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint32_t n_tok, uint32_t n_expert, uint32_t topk, float route_scale)
V41_STUB(ds4_gpu_v41_swiglu_tensor, ds4_gpu_tensor *h, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n_tok, uint32_t mid, float limit)
V41_STUB(ds4_gpu_v41_routed_moe_tensor, ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t blob_offset, uint64_t blob_bytes, uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert_used, float clamp, const ds4_gpu_tensor *x, uint32_t layer, uint32_t n_tok)
V41_STUB(ds4_gpu_v41_matmul_f32_tensor, ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint32_t n_tok)
V41_STUB(ds4_gpu_v41_add_tensor, ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint64_t n)
V41_STUB(ds4_gpu_v41_scale_round_tensor, ds4_gpu_tensor *x, uint64_t n, float s)
V41_STUB(ds4_gpu_v41_amp_apply_tensor, ds4_gpu_tensor *y, const ds4_gpu_tensor *x, const ds4_gpu_tensor *A, const ds4_gpu_tensor *B, ds4_gpu_tensor *T, uint32_t n_tok, uint32_t D, uint32_t K)
V41_STUB(ds4_gpu_v41_engram_rows_tensor, ds4_gpu_tensor *out, const ds4_gpu_tensor *raw, uint32_t n_rows, uint32_t head_dim)
V41_STUB(ds4_gpu_v41_argmax_tensor, ds4_gpu_tensor *idx, const ds4_gpu_tensor *logits, uint32_t row, uint32_t n_vocab)
V41_STUB(ds4_gpu_v41_engram_gate_tensor, ds4_gpu_tensor *hc, const ds4_gpu_tensor *kv, const void *model_map, uint64_t model_size, uint64_t q_w_offset, uint64_t k_w_offset, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok, float eps)
V41_STUB(ds4_gpu_v41_expand_hc_tensor, ds4_gpu_tensor *hc, const ds4_gpu_tensor *x, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok)
V41_STUB(ds4_gpu_v41_vq_capture_expert_out, float *host, uint32_t n_tok, uint32_t n_used, uint32_t out_dim)
V41_STUB(ds4_gpu_v41_set_gr_override, uint32_t layer, const float *host, uint32_t n_expert, uint32_t out_dim)
