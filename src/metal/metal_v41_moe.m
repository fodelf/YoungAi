/* metal_v41_moe.m — V4.1 routed MoE(VQ blob)与草稿塔 dense MoE 的 Metal 发射(2026-10-08)。契约 ds4_gpu_v41.h; 核在 metal/v41_vq.metal。
 * 解码(n ≤ 8): 逐 (token, 专家) 对解码即乘; 预填(n > 8): 主机把配对按专家计数排序(稳定, 同专家内保持 token 序, 与 cuda_vq_prefill 同一写法),
 * 分组核一行的码字只解一次。排序要读回 selected ⇒ v41_host_sync 一次(每层一次, 与 CUDA 的 D2H 同频)。
 * 反修: gr 增益覆盖表(set_gr_override)按层挂设备缓冲, down 核的行增益 × 它; 取料(vq_capture_expert_out)读上一次预填留下的 ys/inv。 */
#import "metal_v41.h"
#include "vq_fmt.h"

static id<MTLBuffer> g_v41_gr[64];
static v41_scratch g_vq_h, g_vq_part, g_vq_ys, g_vq_cap, g_vq_perm, g_vq_inv, g_vq_meta;
static uint32_t g_cap_tok = 0, g_cap_used = 0, g_cap_out = 0;   /* 上一次预填 MoE 的形状(取料对账) */

id<MTLBuffer> v41_gr_buf(uint32_t layer) { return layer < 64u ? g_v41_gr[layer] : nil; }
int ds4_gpu_v41_set_gr_override(uint32_t layer, const float *host, uint32_t n_expert, uint32_t out_dim) {
    if (layer >= 64u) return 0;
    g_v41_gr[layer] = nil;
    if (!host) return 1;
    if (!g_initialized && !ds4_gpu_init()) return 0;
    const size_t nb = (size_t)n_expert * out_dim * 4;
    g_v41_gr[layer] = [g_device newBufferWithBytes:host length:nb options:MTLResourceStorageModeShared];
    return g_v41_gr[layer] != nil;
}

/* 配对按专家计数排序: perm[排序位置] = 对号, inv[对号] = 排序位置(-1 无效), meta = [act | off | cnt](有 token 的专家) */
int v41_vq_sort_pairs(const ds4_gpu_tensor *selected, uint32_t n_tok, uint32_t K, uint32_t n_total_expert, v41_vq_sort *o) {
    const uint64_t npair = (uint64_t)n_tok * K;
    if (!v41_host_sync()) return 0;
    int32_t *sel = malloc(npair * 4);
    uint32_t *cnt = calloc(n_total_expert + 1u, 4), *off = calloc(n_total_expert + 1u, 4), *cur = malloc((size_t)n_total_expert * 4);
    int ok = sel && cnt && off && cur && ds4_gpu_tensor_read(selected, 0, sel, npair * 4);
    if (ok) {
        for (uint64_t p = 0; p < npair; p++) if (sel[p] >= 0 && (uint32_t)sel[p] < n_total_expert) cnt[sel[p]]++;
        for (uint32_t e = 0; e < n_total_expert; e++) off[e + 1] = off[e] + cnt[e];
        const uint32_t nv = off[n_total_expert];
        id<MTLBuffer> perm = v41_grow(&g_vq_perm, (uint64_t)(nv + 1u) * 4, "vq perm"), inv = v41_grow(&g_vq_inv, npair * 4, "vq inv");
        id<MTLBuffer> meta = v41_grow(&g_vq_meta, (uint64_t)n_total_expert * 12u + 12u, "vq meta");
        if (!perm || !inv || !meta) ok = 0;
        else {
            int32_t *ph = [perm contents], *ih = [inv contents]; uint32_t *mh = [meta contents];
            memcpy(cur, off, (size_t)n_total_expert * 4);
            for (uint64_t p = 0; p < npair; p++) {
                const int32_t e = sel[p];
                if (e < 0 || (uint32_t)e >= n_total_expert) { ih[p] = -1; continue; }
                const uint32_t pos = cur[e]++;
                ph[pos] = (int32_t)p; ih[p] = (int32_t)pos;
            }
            uint32_t nact = 0;
            for (uint32_t e = 0; e < n_total_expert; e++) if (cnt[e]) nact++;
            uint32_t k = 0;
            for (uint32_t e = 0; e < n_total_expert; e++) if (cnt[e]) { mh[k] = e; mh[nact + k] = off[e]; mh[2u * nact + k] = cnt[e]; k++; }
            o->nv = nv; o->nact = nact; o->npair = (uint32_t)npair; o->perm = perm; o->inv = inv; o->meta = meta;
        }
    }
    free(sel); free(cnt); free(off); free(cur);
    return ok;
}

int ds4_gpu_v41_routed_moe_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t blob_offset, uint64_t blob_bytes, uint32_t in_dim,
                                  uint32_t mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert,
                                  uint32_t n_expert_used, float clamp, const ds4_gpu_tensor *x, uint32_t layer, uint32_t n_tok) {
    if ((!out && n_tok > DS4_V41_GEMV_MAX_TOK) || !selected || !weights || !x || n_tok == 0) return 0;
    if (blob_offset > model_size || blob_bytes > model_size - blob_offset) return 0;
    if ((in_dim % 8u) || (mid_dim % 8u) || (out_dim % 8u)) return 0;
    const uint8_t *bh = (const uint8_t *)model_map + blob_offset;
    if (!ds4vq_blob_ok(bh, (size_t)blob_bytes)) {
        fprintf(stderr, "ds4: [v41-metal] L%u 专家 blob 头不合法(魔数/版本/专家数); 本引擎认 DQVL v%u~v%u\n", layer, DS4VQ_BLOB_VER_MIN, DS4VQ_BLOB_VER_MAX);
        exit(1);
    }
    const uint32_t ver = ds4vq_blob_ver(bh);
    uint64_t inner = 0;
    id<MTLBuffer> blob = v41_model_buf(model_map, model_size, blob_offset, blob_bytes, &inner, "v41 vq blob");
    if (!blob) return 0;
    id<MTLBuffer> gr = v41_gr_buf(layer);
    v41_vq_args a = { in_dim, mid_dim, out_dim, n_expert_used, ver == 3u ? 1u : 0u, gr ? 1u : 0u, n_tok, 0, clamp, 0, 0, 0 };
    if (n_tok <= DS4_V41_GEMV_MAX_TOK) {
        const uint64_t np = (uint64_t)n_tok * n_expert_used;
        id<MTLBuffer> h = v41_grow(&g_vq_h, np * mid_dim * 4, "v41 vq h"), part = v41_grow(&g_vq_part, np * out_dim * 4, "v41 vq partial");
        if (!h || !part) return 0;
        v41_bind b1[] = { V41_A(a), V41_B(h, 0), V41_B(blob, 0), V41_T(selected), V41_T(x), V41_A(inner) };
        if (!v41_launch("kernel_v41_vq_gateup", b1, 6, MTLSizeMake((mid_dim + 7u) / 8u, (NSUInteger)np, 1), MTLSizeMake(256, 1, 1))) return 0;
        v41_bind b2[] = { V41_A(a), V41_B(part, 0), V41_B(blob, 0), V41_T(selected), V41_B(h, 0), V41_A(inner), gr ? V41_B(gr, 0) : V41_T(NULL) };
        if (!v41_launch("kernel_v41_vq_down", b2, 7, MTLSizeMake((out_dim + 7u) / 8u, (NSUInteger)np, 1), MTLSizeMake(256, 1, 1))) return 0;
        if (!out) return 1;
        const uint32_t tail = 0;
        v41_bind b3[] = { V41_A(a), V41_T(out), V41_B(part, 0), V41_T(weights), V41_T(NULL), V41_A(tail) };
        return v41_launch("kernel_v41_vq_reduce", b3, 6, MTLSizeMake((out_dim + 255u) / 256u, n_tok, 1), MTLSizeMake(256, 1, 1));
    }
    v41_vq_sort s;
    if (!v41_vq_sort_pairs(selected, n_tok, n_expert_used, n_total_expert, &s)) return 0;
    if (s.nv == 0) return v41_fill_u32(out, 0, (uint64_t)n_tok * out_dim, 0u);
    id<MTLBuffer> h = v41_grow(&g_vq_h, (uint64_t)s.nv * mid_dim * 4, "v41 vq h"), ys = v41_grow(&g_vq_ys, (uint64_t)s.nv * out_dim * 4, "v41 vq ys");
    if (!h || !ys) return 0;
    a.n_act = s.nact;
    v41_bind b1[] = { V41_A(a), V41_B(h, 0), V41_B(blob, 0), V41_B(s.meta, 0), V41_B(s.perm, 0), V41_T(x), V41_A(inner) };
    if (!v41_launch("kernel_v41_vq_gateup_grp", b1, 7, MTLSizeMake((mid_dim + 7u) / 8u, s.nact, 1), MTLSizeMake(256, 1, 1))) return 0;
    v41_bind b2[] = { V41_A(a), V41_B(ys, 0), V41_B(blob, 0), V41_B(s.meta, 0), V41_B(h, 0), V41_A(inner), gr ? V41_B(gr, 0) : V41_T(NULL) };
    if (!v41_launch("kernel_v41_vq_down_grp", b2, 7, MTLSizeMake((out_dim + 7u) / 8u, s.nact, 1), MTLSizeMake(256, 1, 1))) return 0;
    v41_bind b3[] = { V41_A(a), V41_T(out), V41_B(ys, 0), V41_B(s.inv, 0), V41_T(weights) };
    if (!v41_launch("kernel_v41_vqp_reduce", b3, 5, MTLSizeMake((out_dim + 255u) / 256u, n_tok, 1), MTLSizeMake(256, 1, 1))) return 0;
    g_cap_tok = n_tok; g_cap_used = n_expert_used; g_cap_out = out_dim;
    return 1;
}
int ds4_gpu_v41_moe_tail_tensor(ds4_gpu_tensor *y, const ds4_gpu_tensor *so, const ds4_gpu_tensor *weights, uint32_t n_tok, uint32_t n_used, uint32_t out_dim) {
    if (!y || !so || !weights || !g_vq_part.buf || g_vq_part.cap < (uint64_t)n_tok * n_used * out_dim * 4) return 0;
    v41_vq_args a = { 0, 0, out_dim, n_used, 0, 0, n_tok, 0, 0.0f, 0, 0, 0 };
    const uint32_t tail = 1;
    v41_bind b[] = { V41_A(a), V41_T(y), V41_B(g_vq_part.buf, 0), V41_T(weights), V41_T(so), V41_A(tail) };
    return v41_launch("kernel_v41_vq_reduce", b, 6, MTLSizeMake((out_dim + 255u) / 256u, n_tok, 1), MTLSizeMake(256, 1, 1));
}
/* 反修取料: out[t][k][o] = ys[inv[t·n_used+k]][o](reduce 前、未乘路由权重); 形状必须与刚跑完的预填层对上 */
int ds4_gpu_v41_vq_capture_expert_out(float *host, uint32_t n_tok, uint32_t n_used, uint32_t out_dim) {
    if (!host || n_tok != g_cap_tok || n_used != g_cap_used || out_dim != g_cap_out || !g_vq_ys.buf || !g_vq_inv.buf) return 0;
    const uint64_t npair = (uint64_t)n_tok * n_used, nel = npair * out_dim;
    id<MTLBuffer> cap = v41_grow(&g_vq_cap, nel * 4, "vq 取料展开");
    if (!cap) return 0;
    v41_vq_args a = { 0, 0, out_dim, n_used, 0, 0, n_tok, 0, 0.0f, 0, 0, 0 };
    v41_bind b[] = { V41_A(a), V41_B(cap, 0), V41_B(g_vq_ys.buf, 0), V41_B(g_vq_inv.buf, 0) };
    if (!v41_launch("kernel_v41_vqp_expand", b, 4, MTLSizeMake((out_dim + 255u) / 256u, (NSUInteger)npair, 1), MTLSizeMake(256, 1, 1))) return 0;
    if (!v41_host_sync()) return 0;
    memcpy(host, [cap contents], (size_t)nel * 4);
    return 1;
}

/* ---- 草稿塔 dense MoE: 每塔一张表(专家 × 3 个 fp4x32 张量的视图偏移 + 住哪个视图) ---- */
typedef struct { uint32_t n, nview; __strong id<MTLBuffer> offs, vid; __strong id<MTLBuffer> views[8]; } v41_mtp_tab;
static v41_mtp_tab g_mtp[8];
static v41_scratch g_mtp_h, g_mtp_part;
static int v41_mtp_bind(uint32_t tower, const void *model_map, const uint64_t *off, uint32_t n_expert) {
    if (tower >= 8u) return 0;
    v41_mtp_tab *T = &g_mtp[tower];
    if (T->n == n_expert && T->offs) return 1;
    uint64_t *ho = malloc(sizeof(uint64_t) * 3u * n_expert); uint32_t *hv = malloc(4u * n_expert);
    if (!ho || !hv) { free(ho); free(hv); return 0; }
    T->nview = 0;
    int ok = 1;
    for (uint32_t e = 0; e < n_expert && ok; e++) {
        uint32_t vid = 0xFFFFFFFFu;
        for (uint32_t w = 0; w < 3u; w++) {
            uint64_t inner = 0;
            id<MTLBuffer> b = ds4_gpu_wrap_model_range(model_map, g_model_map_size ? g_model_map_size : UINT64_MAX, off[w * n_expert + e], 17u, &inner);
            if (!b) { ok = 0; break; }
            uint32_t v = 0;
            while (v < T->nview && T->views[v] != b) v++;
            if (v == T->nview) { if (T->nview >= 8u) { ok = 0; break; } T->views[T->nview++] = b; }
            if (vid == 0xFFFFFFFFu) vid = v;
            else if (vid != v) { fprintf(stderr, "ds4: [v41-metal] 草稿塔 %u 专家 %u 的三块矩阵跨了映射视图\n", tower, e); ok = 0; break; }
            ho[w * n_expert + e] = inner;
        }
        hv[e] = vid;
    }
    if (ok) {
        T->offs = [g_device newBufferWithBytes:ho length:sizeof(uint64_t) * 3u * n_expert options:MTLResourceStorageModeShared];
        T->vid = [g_device newBufferWithBytes:hv length:4u * n_expert options:MTLResourceStorageModeShared];
        T->n = n_expert;
        ok = T->offs && T->vid;
    }
    free(ho); free(hv);
    return ok;
}
int ds4_gpu_v41_mtp_moe_tensor(ds4_gpu_tensor *out, const void *model_map, uint32_t tower, const uint64_t *exp_off, uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim,
                               const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_expert, uint32_t topk, float clamp, const ds4_gpu_tensor *x, uint32_t n_tok) {
    if (!out || !selected || !weights || !x || !exp_off || !n_tok || !topk || (in_dim % 32u) || (mid_dim % 32u)) return 0;
    if (!v41_mtp_bind(tower, model_map, exp_off, n_expert)) return 0;
    const v41_mtp_tab *T = &g_mtp[tower];
    const uint64_t np = (uint64_t)n_tok * topk;
    id<MTLBuffer> h = v41_grow(&g_mtp_h, np * mid_dim * 4, "mtp h"), part = v41_grow(&g_mtp_part, np * out_dim * 4, "mtp partial");
    if (!h || !part) return 0;
    for (uint32_t v = 0; v < T->nview; v++) {
        v41_mtp_args a = { in_dim, mid_dim, out_dim, topk, n_expert, v, 0, 0, clamp, 0, 0, 0 };
        v41_bind b1[] = { V41_A(a), V41_B(h, 0), V41_B(T->views[v], 0), V41_B(T->offs, 0), V41_B(T->vid, 0), V41_T(selected), V41_T(x) };
        if (!v41_launch("kernel_v41_mtp_gateup", b1, 7, MTLSizeMake((mid_dim + 7u) / 8u, (NSUInteger)np, 1), MTLSizeMake(256, 1, 1))) return 0;
    }
    for (uint32_t v = 0; v < T->nview; v++) {
        v41_mtp_args a = { in_dim, mid_dim, out_dim, topk, n_expert, v, 0, 0, clamp, 0, 0, 0 };
        v41_bind b2[] = { V41_A(a), V41_B(part, 0), V41_B(T->views[v], 0), V41_B(T->offs, 0), V41_B(T->vid, 0), V41_T(selected), V41_B(h, 0) };
        if (!v41_launch("kernel_v41_mtp_down", b2, 7, MTLSizeMake((out_dim + 7u) / 8u, (NSUInteger)np, 1), MTLSizeMake(256, 1, 1))) return 0;
    }
    v41_vq_args r = { in_dim, mid_dim, out_dim, topk, 0, 0, n_tok, 0, clamp, 0, 0, 0 };
    const uint32_t tail = 0;
    v41_bind b3[] = { V41_A(r), V41_T(out), V41_B(part, 0), V41_T(weights), V41_T(NULL), V41_A(tail) };
    return v41_launch("kernel_v41_vq_reduce", b3, 6, MTLSizeMake((out_dim + 255u) / 256u, n_tok, 1), MTLSizeMake(256, 1, 1));
}
