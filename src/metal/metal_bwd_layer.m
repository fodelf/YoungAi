/* metal_bwd_layer.m — 后训练反传逐层原语的 Metal 发射(2026-10-08): 稀疏注意力反向 / mHC 混合系数反向 / SwiGLU·路由反向 / routed 专家反向(VQ 直读) /
 * 压缩器池化·engram 门反向。契约 ds4_gpu_bwd.h; 核在 metal/v41_bwd.metal。
 * routed 专家反向: 配对按专家排序(与前向预填同一份排序), 重算 H_g/H_u/A/O(直读行点积, bf16 格点 = 前向真算的那份), 三发转置累加直读位流;
 * 放回 token 行走 inv 表按 k 升序求和(确定序, 不用原子加)。 */
#import "metal_v41.h"
#include "vq_fmt.h"

int ds4_gpu_bwd_sparse_attn_tensor(ds4_gpu_tensor *gq, ds4_gpu_tensor *gkv, ds4_gpu_tensor *gcomp, const ds4_gpu_tensor *go, const ds4_gpu_tensor *o, const ds4_gpu_tensor *q,
                                   const ds4_gpu_tensor *kv_win, const ds4_gpu_tensor *kv_comp, const ds4_gpu_tensor *idx, const void *model_map, uint64_t model_size,
                                   uint64_t sink_offset, uint32_t n_tok, uint32_t window, uint32_t ng, uint32_t topk, uint32_t n_head, uint32_t head_dim, float scale) {
    if (!gq || !gkv || !go || !o || !q || !kv_win || head_dim != 512u || n_tok == 0 || (n_head % 8u)) return 0;
    if ((kv_comp == NULL) != (idx == NULL)) return 0;
    uint64_t so = 0;
    id<MTLBuffer> sb = v41_model_buf(model_map, model_size, sink_offset, (uint64_t)n_head * 4, &so, "bwd sink");
    if (!sb) return 0;
    if (!v41_fill_u32(gkv, 0, (uint64_t)n_tok * head_dim, 0u)) return 0;
    v41_battn_args a = { window, ng, kv_comp ? topk : 0u, n_head, head_dim, kv_comp ? 1u : 0u, (gcomp && kv_comp) ? 1u : 0u, 0, scale, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(gq), V41_T(gkv), V41_T(gcomp), V41_T(go), V41_T(o), V41_T(q), V41_T(kv_win), V41_T(kv_comp), V41_T(idx), V41_B(sb, (NSUInteger)so) };
    return v41_launch("kernel_v41_bwd_sparse_attn", b, 11, MTLSizeMake(n_tok, n_head / 8u, 1), MTLSizeMake(256, 1, 1));
}
static v41_scratch g_bwd_gmix;
int ds4_gpu_bwd_hc_mix_tensor(ds4_gpu_tensor *ghc, const ds4_gpu_tensor *gpre, const ds4_gpu_tensor *gpost, const ds4_gpu_tensor *gcomb, const ds4_gpu_tensor *hc,
                              const ds4_gpu_tensor *mix, const void *model_map, uint64_t model_size, uint64_t fn_offset, uint64_t scale_offset, uint64_t base_offset,
                              uint32_t n_embd, uint32_t n_hc, uint32_t iters, float hc_eps, float norm_eps, uint32_t n_tok) {
    if (!ghc || !hc || !mix || n_hc > 4u || iters > 32u || n_tok == 0) return 0;
    const uint32_t mh = 2u * n_hc + n_hc * n_hc, dim = n_embd * n_hc;
    uint64_t wo = 0, so = 0, bo = 0;
    id<MTLBuffer> W = v41_model_buf(model_map, model_size, fn_offset, (uint64_t)mh * dim * 4, &wo, "bwd hc fn");
    id<MTLBuffer> sc = v41_model_buf(model_map, model_size, scale_offset, 12, &so, "bwd hc scale");
    id<MTLBuffer> bs = v41_model_buf(model_map, model_size, base_offset, (uint64_t)mh * 4, &bo, "bwd hc base");
    id<MTLBuffer> gm = v41_grow(&g_bwd_gmix, (uint64_t)n_tok * mh * 4, "bwd hc gmix");
    if (!W || !sc || !bs || !gm) return 0;
    v41_bhc_args a = { n_tok, n_hc, iters, mh, dim, gpre ? 1u : 0u, gpost ? 1u : 0u, gcomb ? 1u : 0u, hc_eps, norm_eps, 0, 0 };
    v41_bind b1[] = { V41_A(a), V41_B(gm, 0), V41_T(mix), V41_T(gpre), V41_T(gpost), V41_T(gcomb), V41_B(sc, (NSUInteger)so), V41_B(bs, (NSUInteger)bo) };
    if (!v41_launch("kernel_v41_bwd_hc_split", b1, 8, MTLSizeMake((n_tok + 63u) / 64u, 1, 1), MTLSizeMake(64, 1, 1))) return 0;
    v41_bind b2[] = { V41_A(a), V41_T(ghc), V41_B(gm, 0), V41_T(mix), V41_B(W, (NSUInteger)wo), V41_T(hc) };
    return v41_launch("kernel_v41_bwd_hc_mix", b2, 6, MTLSizeMake(n_tok, 1, 1), MTLSizeMake(512, 1, 1));
}
int ds4_gpu_bwd_swiglu_tensor(ds4_gpu_tensor *ggate, ds4_gpu_tensor *gup, const ds4_gpu_tensor *gh, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint64_t n, float limit) {
    if (!ggate || !gup || !gh || !gate || !up || n == 0) return 0;
    v41_bind b[] = { V41_T(ggate), V41_T(gup), V41_T(gh), V41_T(gate), V41_T(up), V41_A(n), V41_A(limit) };
    return v41_launch_1d("kernel_v41_bwd_swiglu", b, 7, n);
}
int ds4_gpu_bwd_router_tensor(ds4_gpu_tensor *gz, const ds4_gpu_tensor *gw, const ds4_gpu_tensor *sel, const ds4_gpu_tensor *z, uint32_t n_tok, uint32_t n_expert, uint32_t k, float route_scale) {
    if (!gz || !gw || !sel || !z || k > 16u || n_tok == 0) return 0;
    v41_brt_args a = { n_expert, k, 0, 0, route_scale, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(gz), V41_T(gw), V41_T(sel), V41_T(z) };
    return v41_launch("kernel_v41_bwd_router", b, 5, MTLSizeMake(n_tok, 1, 1), MTLSizeMake(128, 1, 1));
}
int ds4_gpu_bwd_router_fixed_tensor(ds4_gpu_tensor *weights, const ds4_gpu_tensor *sel, const ds4_gpu_tensor *logits, uint32_t n_tok, uint32_t n_expert, uint32_t topk, float route_scale) {
    if (!weights || !sel || !logits || topk > 16u || n_tok == 0) return 0;
    v41_brt_args a = { n_expert, topk, 0, 0, route_scale, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(weights), V41_T(sel), V41_T(logits), V41_A(n_tok) };
    return v41_launch_1d("kernel_v41_bwd_router_fixed", b, 5, n_tok);
}
int ds4_gpu_bwd_compress_pool_tensor(ds4_gpu_tensor *gkv, ds4_gpu_tensor *gsc, const ds4_gpu_tensor *gpooled, const ds4_gpu_tensor *kv, const ds4_gpu_tensor *score,
                                     uint32_t n_groups, uint32_t ratio, uint32_t dim) {
    if (!gkv || !gsc || !gpooled || !kv || !score || ratio < 2u) return 0;
    if (n_groups == 0) return 1;
    v41_n_args a = { ratio, dim, 0, 0, 0, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(gkv), V41_T(gsc), V41_T(gpooled), V41_T(kv), V41_T(score) };
    return v41_launch("kernel_v41_bwd_compress_pool", b, 6, MTLSizeMake(n_groups, 1, 1), MTLSizeMake(256, 1, 1));
}
int ds4_gpu_bwd_engram_gate_tensor(ds4_gpu_tensor *g, const ds4_gpu_tensor *hc_pre, const ds4_gpu_tensor *kv, const void *model_map, uint64_t model_size, uint64_t q_w_offset,
                                   uint64_t k_w_offset, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok, float eps) {
    if (!g || !hc_pre || !kv || n_tok == 0) return 0;
    uint64_t qo = 0, ko = 0;
    id<MTLBuffer> qb = v41_model_buf(model_map, model_size, q_w_offset, (uint64_t)n_hc * n_embd * 4, &qo, "bwd engram q");
    id<MTLBuffer> kb = v41_model_buf(model_map, model_size, k_w_offset, (uint64_t)n_hc * n_embd * 4, &ko, "bwd engram k");
    if (!qb || !kb) return 0;
    v41_n_args a = { n_embd, n_hc, 0, 0, eps, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_T(g), V41_T(hc_pre), V41_T(kv), V41_B(qb, (NSUInteger)qo), V41_B(kb, (NSUInteger)ko) };
    return v41_launch("kernel_v41_bwd_engram_gate", b, 6, MTLSizeMake(n_hc, n_tok, 1), MTLSizeMake(256, 1, 1));
}

/* ---- routed 专家反向 ---- */
static v41_scratch g_bvq_xg, g_bvq_hg, g_bvq_hu, g_bvq_a, g_bvq_og, g_bvq_ga, g_bvq_bad;
static int v41_vqb_rowdot(id<MTLBuffer> out, id<MTLBuffer> x, id<MTLBuffer> blob, uint64_t inner, const v41_vq_sort *s, uint32_t which, uint32_t R, uint32_t C,
                          uint32_t v3, id<MTLBuffer> gov, uint32_t OUTd, id<MTLBuffer> bad) {
    v41_bvq_args a = { which, R, C, s->nact, v3, gov ? 1u : 0u, OUTd, 0, 1, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_B(out, 0), V41_B(x, 0), V41_B(blob, 0), V41_B(s->meta, 0), V41_A(inner), gov ? V41_B(gov, 0) : V41_T(NULL), V41_B(bad, 0) };
    return v41_launch("kernel_v41_vqb_rowdot", b, 8, MTLSizeMake((R + 7u) / 8u, s->nact, 1), MTLSizeMake(256, 1, 1));
}
static int v41_vqb_tdot(id<MTLBuffer> out, id<MTLBuffer> g, id<MTLBuffer> blob, uint64_t inner, const v41_vq_sort *s, uint32_t which, uint32_t R, uint32_t C,
                        uint32_t v3, id<MTLBuffer> gov, uint32_t OUTd, int accumulate, id<MTLBuffer> bad) {
    v41_bvq_args a = { which, R, C, s->nact, v3, gov ? 1u : 0u, OUTd, accumulate ? 1u : 0u, 0, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_B(out, 0), V41_B(g, 0), V41_B(blob, 0), V41_B(s->meta, 0), V41_A(inner), gov ? V41_B(gov, 0) : V41_T(NULL), V41_B(bad, 0) };
    return v41_launch("kernel_v41_vqb_tdot", b, 8, MTLSizeMake((C / 8u + 127u) / 128u, s->nact, 1), MTLSizeMake(128, 1, 1));
}
int ds4_gpu_bwd_routed_moe_tensor(ds4_gpu_tensor *gx, ds4_gpu_tensor *gw, const ds4_gpu_tensor *gy, const ds4_gpu_tensor *x, const ds4_gpu_tensor *sel, const ds4_gpu_tensor *rw,
                                  const void *model_map, uint64_t model_size, uint64_t blob_offset, uint64_t blob_bytes, uint32_t IN, uint32_t MID, uint32_t OUT,
                                  uint32_t n_total_expert, uint32_t K, float clamp, uint32_t layer, uint32_t n_tok) {
    if (!gx || !gw || !gy || !x || !sel || !rw || n_tok == 0 || IN != OUT) return 0;
    if (blob_offset > model_size || blob_bytes > model_size - blob_offset) return 0;
    const uint8_t *bh = (const uint8_t *)model_map + blob_offset;
    if (!ds4vq_blob_ok(bh, (size_t)blob_bytes)) return 0;
    const uint32_t v3 = ds4vq_blob_ver(bh) == 3u ? 1u : 0u;
    uint64_t inner = 0;
    id<MTLBuffer> blob = v41_model_buf(model_map, model_size, blob_offset, blob_bytes, &inner, "bwd vq blob");
    if (!blob) return 0;
    const uint64_t npair = (uint64_t)n_tok * K;
    if (!v41_fill_u32(gw, 0, npair, 0u)) return 0;
    v41_vq_sort s;
    if (!v41_vq_sort_pairs(sel, n_tok, K, n_total_expert, &s)) return 0;
    if (s.nv == 0) return 1;
    const uint64_t nv = s.nv;
    id<MTLBuffer> xg = v41_grow(&g_bvq_xg, nv * IN * 4, "bwd vq x"), hg = v41_grow(&g_bvq_hg, nv * MID * 4, "bwd vq hg"), hu = v41_grow(&g_bvq_hu, nv * MID * 4, "bwd vq hu");
    id<MTLBuffer> ab = v41_grow(&g_bvq_a, nv * MID * 4, "bwd vq a"), og = v41_grow(&g_bvq_og, nv * OUT * 4, "bwd vq o/go/gx"), ga = v41_grow(&g_bvq_ga, nv * MID * 4, "bwd vq ga");
    id<MTLBuffer> bad = v41_grow(&g_bvq_bad, 4, "bwd vq bad");
    if (!xg || !hg || !hu || !ab || !og || !ga || !bad) return 0;
    if (!v41_fill_buf_u32(bad, 0, 1, 0u)) return 0;
    id<MTLBuffer> gov = v41_gr_buf(layer);
    const uint64_t nh = nv * MID;
    v41_n_args ga_ = { K, IN, 0, 0, 0, 0, 0, 0 };
    v41_bind bg[] = { V41_A(ga_), V41_B(xg, 0), V41_T(x), V41_B(s.perm, 0), V41_T(NULL) };
    if (!v41_launch("kernel_v41_bwd_gather", bg, 5, MTLSizeMake((NSUInteger)nv, 1, 1), MTLSizeMake(256, 1, 1))) return 0;   /* X 行按排序位置收 */
    if (!v41_vqb_rowdot(hg, xg, blob, inner, &s, 0u, MID, IN, v3, nil, OUT, bad)) return 0;                                        /* H_g */
    if (!v41_vqb_rowdot(hu, xg, blob, inner, &s, 1u, MID, IN, v3, nil, OUT, bad)) return 0;                                        /* H_u */
    v41_bind bs[] = { V41_B(ab, 0), V41_B(hg, 0), V41_B(hu, 0), V41_A(nh), V41_A(clamp) };
    if (!v41_launch_1d("kernel_v41_bwd_swiglu_fwd", bs, 5, nh)) return 0;                                                          /* A = swiglu(H_g, H_u) */
    if (!v41_vqb_rowdot(og, ab, blob, inner, &s, 2u, OUT, MID, v3, gov, OUT, bad)) return 0;                                       /* O = W2·A(未加权) */
    v41_n_args gr_ = { K, OUT, 0, 0, 0, 0, 0, 0 };
    v41_bind br[] = { V41_A(gr_), V41_T(gw), V41_T(gy), V41_B(og, 0), V41_B(s.perm, 0) };
    if (!v41_launch("kernel_v41_bwd_rowdot", br, 5, MTLSizeMake((NSUInteger)nv, 1, 1), MTLSizeMake(256, 1, 1))) return 0;         /* g_w = <g_y, O> */
    v41_n_args gg_ = { K, OUT, 1, 0, 0, 0, 0, 0 };
    v41_bind bgo[] = { V41_A(gg_), V41_B(og, 0), V41_T(gy), V41_B(s.perm, 0), V41_T(rw) };
    if (!v41_launch("kernel_v41_bwd_gather", bgo, 5, MTLSizeMake((NSUInteger)nv, 1, 1), MTLSizeMake(256, 1, 1))) return 0;      /* og ← G_O = g_y·rw */
    if (!v41_vqb_tdot(ga, og, blob, inner, &s, 2u, OUT, MID, v3, gov, OUT, 0, bad)) return 0;                                      /* G_A = G_O·W2 */
    v41_bind bsw[] = { V41_B(hg, 0), V41_B(hu, 0), V41_B(ga, 0), V41_B(hg, 0), V41_B(hu, 0), V41_A(nh), V41_A(clamp) };
    if (!v41_launch_1d("kernel_v41_bwd_swiglu", bsw, 7, nh)) return 0;                                                             /* hg/hu ← G_Hg/G_Hu */
    if (!v41_vqb_tdot(og, hg, blob, inner, &s, 0u, MID, IN, v3, nil, OUT, 0, bad)) return 0;                                       /* og ← G_X(gate 部分) */
    if (!v41_vqb_tdot(og, hu, blob, inner, &s, 1u, MID, IN, v3, nil, OUT, 1, bad)) return 0;                                       /* og += G_X(up 部分) */
    v41_n_args sc_ = { K, IN, 0, 0, 0, 0, 0, 0 };
    v41_bind bsc[] = { V41_A(sc_), V41_T(gx), V41_B(og, 0), V41_B(s.inv, 0) };
    if (!v41_launch("kernel_v41_bwd_scatter", bsc, 4, MTLSizeMake((IN + 255u) / 256u, n_tok, 1), MTLSizeMake(256, 1, 1))) return 0;
    if (!v41_host_sync()) return 0;
    const int bad_h = *(const int *)[bad contents];
    if (bad_h) fprintf(stderr, "ds4: [bwd-metal] L%u 专家载荷头不认(v3 %u) —— 反向不出假梯度, 停车\n", layer, v3);
    return !bad_h;
}

