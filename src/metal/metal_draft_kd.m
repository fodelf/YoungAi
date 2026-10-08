/* metal_draft_kd.m — 草稿器蒸馏原语的 Metal 发射(2026-10-08): 块注意力前向(出 lse)/反向、陪审团、总变差损失。契约 ds4_gpu_bwd.h; 核在 metal/v41_draft.metal。
 * (markov 偏置表、取行/散加、小批低秩件在 metal_v41_misc.m。) */
#import "metal_v41.h"

int ds4_gpu_draft_attn_fwd_tensor(ds4_gpu_tensor *o, ds4_gpu_tensor *lse, const ds4_gpu_tensor *q, const ds4_gpu_tensor *hist, uint32_t hbase, const ds4_gpu_tensor *blk,
                                  const ds4_gpu_tensor *bpos, uint32_t nb, uint32_t B, uint32_t window, const void *model_map, uint64_t model_size, uint64_t sink_offset,
                                  uint32_t n_head, uint32_t head_dim, float scale) {
    if (!o || !lse || !q || !hist || !blk || !bpos || !nb || !B || head_dim != 512u || (n_head % 8u)) return 0;
    uint64_t so = 0;
    id<MTLBuffer> sb = v41_model_buf(model_map, model_size, sink_offset, (uint64_t)n_head * 4, &so, "dk sink");
    if (!sb) return 0;
    v41_dk_args a = { hbase, B, window, n_head, head_dim, nb, 0, 0, scale, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(o), V41_T(lse), V41_T(q), V41_T(hist), V41_T(blk), V41_T(bpos), V41_B(sb, (NSUInteger)so) };
    return v41_launch("kernel_v41_draft_attn_fwd", b, 8, MTLSizeMake(nb * B, n_head / 8u, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_draft_attn_bwd_tensor(ds4_gpu_tensor *gq, ds4_gpu_tensor *gblk, const ds4_gpu_tensor *go, const ds4_gpu_tensor *o, const ds4_gpu_tensor *q, const ds4_gpu_tensor *lse,
                                  const ds4_gpu_tensor *hist, uint32_t hbase, const ds4_gpu_tensor *blk, const ds4_gpu_tensor *bpos, uint32_t nb, uint32_t B, uint32_t window,
                                  uint32_t n_head, uint32_t head_dim, float scale) {
    if (!gq || !gblk || !go || !o || !q || !lse || !hist || !blk || !bpos || !nb || !B || head_dim != 512u || (n_head % 8u)) return 0;
    const uint32_t R = nb * B;
    if (!v41_fill_u32(gblk, 0, (uint64_t)R * head_dim, 0u)) return 0;
    v41_dk_args a = { hbase, B, window, n_head, head_dim, nb, 0, 0, scale, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(gq), V41_T(gblk), V41_T(go), V41_T(o), V41_T(q), V41_T(lse), V41_T(hist), V41_T(blk), V41_T(bpos) };
    return v41_launch("kernel_v41_draft_attn_bwd", b, 10, MTLSizeMake(R, n_head / 8u, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_draft_jury_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *ls, const ds4_gpu_tensor *lt, uint32_t m, uint32_t n_vocab, float T) {
    if (!out || !ls || !lt || !m || !n_vocab || !(T > 0.f)) return 0;
    v41_bind b[] = { V41_T(out), V41_T(ls), V41_T(lt), V41_A(n_vocab), V41_A(T) };
    return v41_launch("kernel_v41_draft_jury", b, 5, MTLSizeMake(m, 1, 1), MTLSizeMake(1024, 1, 1));
}
int ds4_gpu_draft_tv_tensor(ds4_gpu_tensor *glogits, ds4_gpu_tensor *loss, const ds4_gpu_tensor *logits, uint32_t m, uint32_t n_vocab, const ds4_gpu_tensor *tid,
                            const ds4_gpu_tensor *tp, const ds4_gpu_tensor *trest, uint32_t k, float T, float scale) {
    if (!glogits || !loss || !logits || !tid || !tp || !trest || !m || !k || !(T > 0.f) || n_vocab > 4096u * 32u) return 0;
    v41_tv_args a = { n_vocab, k, 0, 0, T, scale, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(glogits), V41_T(loss), V41_T(logits), V41_T(tid), V41_T(tp), V41_T(trest) };
    return v41_launch("kernel_v41_draft_tv", b, 7, MTLSizeMake(m, 1, 1), MTLSizeMake(1024, 1, 1));
}
