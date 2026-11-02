/* t_v41_bwd_attn.c — 稀疏注意力反向的有限差分门(10-25 从 t_metal_v41_bwd2.c 搬来并扩形态), 两后端都跑, 不要模型。
 * 为什么单拎出来: ③ 训练器的长样本走"前缀只预填、尾窗才反传"(core_ptrain_bwd.c, cfg ctxlen), 尾窗的注意力前/反向带 pos0 = 前缀长,
 *   位置 a < pos0 的键在环头第 a % window 行(v41_win_row), 是常量不出梯度。原测试只有 Metal、8 头、pos0 = 0;
 *   CUDA 生产核只收 64 头 × 512 维, 所以这里一律 64 头, pos0 取 0 / 2(前缀短于窗口)/ 6(前缀长于窗口)三档。
 * 判据: 前向输出比 CPU 光滑注意力; q / 尾窗键 / 压缩键三份梯度各 12 点中心差分(L = <go, o>)。
 *   挂了会怎样: 前缀键行号算错 = gq 的 FD 对不上; 尾窗键梯度写错行 = gkv 的 FD 对不上 —— 训练照跑不报错, 只是梯度是错的。 */
#include "test_internal.h"
#ifndef DS4_NO_GPU
#include "../src/common/ds4_fp8.h"
#include "../ds4_gpu_v41.h"
#include "../ds4_gpu_bwd.h"

static float ckv_get(const uint8_t *row, uint32_t d) {
    const uint8_t by = row[d >> 1], n4 = (d & 1) ? (by >> 4) : (by & 15);
    return test_v41_bf16r(ds4_fp4_nibble_to_f32(n4) * ds4_e4m3fn_to_f32(row[DS4_V41_CKV_NIB + (d >> 4)]));
}
/* 窗口缓冲行号, 与 CUDA 的 v41_win_row(ring = 1)同式: 尾窗位置在 [W, W+N), 前缀位置在环头 */
static uint32_t win_row(uint32_t a, uint32_t p0, uint32_t W) { return a >= p0 ? W + a - p0 : a % W; }
/* CPU 光滑注意力(值 = 键; sink 只进分母): L = <go, o>; o 非空时顺手写出前向输出 */
typedef struct { uint32_t N, P0, NH, HD, W, NG, TOPK; const float *go, *q, *win, *sink, *comp; const int32_t *idx; float scale; float *o; } at_ctx;
static double at_loss(const at_ctx *c, const float *q, const float *win, const float *comp) {
    double L = 0.0; float *keys = malloc((size_t)(c->W + c->TOPK) * c->HD * 4); double *s = malloc((c->W + c->TOPK) * 8);
    for (uint32_t i = 0; i < c->N; i++) {
        const uint32_t p = c->P0 + i, lo = p + 1 > c->W ? p + 1 - c->W : 0; uint32_t nk = 0;
        for (uint32_t a = lo; a <= p; a++) { memcpy(keys + nk * c->HD, win + (uint64_t)win_row(a, c->P0, c->W) * c->HD, c->HD * 4); nk++; }
        for (uint32_t kk = 0; kk < c->TOPK; kk++) { const int32_t g = c->idx[i * c->TOPK + kk]; if (g < 0 || (uint32_t)g >= c->NG) continue; memcpy(keys + nk * c->HD, comp + (uint64_t)g * c->HD, c->HD * 4); nk++; }
        for (uint32_t h = 0; h < c->NH; h++) {
            const float *qh = q + ((uint64_t)i * c->NH + h) * c->HD, *gh = c->go + ((uint64_t)i * c->NH + h) * c->HD;
            double mx = -1e30; for (uint32_t k = 0; k < nk; k++) { double d = 0; for (uint32_t e = 0; e < c->HD; e++) d += (double)qh[e] * keys[k * c->HD + e]; s[k] = d * c->scale; mx = fmax(mx, s[k]); }
            double den = exp((double)c->sink[h] - mx);
            for (uint32_t k = 0; k < nk; k++) den += exp(s[k] - mx);
            float *oh = c->o ? c->o + ((uint64_t)i * c->NH + h) * c->HD : NULL;
            if (oh) memset(oh, 0, c->HD * 4);
            for (uint32_t k = 0; k < nk; k++) {
                const double pk = exp(s[k] - mx) / den; double dd = 0;
                for (uint32_t e = 0; e < c->HD; e++) { dd += (double)gh[e] * keys[k * c->HD + e]; if (oh) oh[e] += (float)(pk * keys[k * c->HD + e]); }
                L += pk * dd;
            }
        }
    }
    free(keys); free(s); return L;
}
static double f_at_q(void *c_, const float *q) { const at_ctx *c = c_; return at_loss(c, q, c->win, c->comp); }
static double f_at_win(void *c_, const float *w) { const at_ctx *c = c_; return at_loss(c, c->q, w, c->comp); }
static double f_at_comp(void *c_, const float *cp) { const at_ctx *c = c_; return at_loss(c, c->q, c->win, cp); }

static void attn_case(uint32_t P0) {
    const uint32_t N = 5, NH = 64, HD = 512, W = 4, NG = 5, TOPK = 3; const float scale = 0.06f;
    char nm[64];
    TEST_ASSERT(test_v41_model_begin(NH * 4 + 4096));
    float sink[64]; for (uint32_t h = 0; h < NH; h++) sink[h] = test_v41_randf();
    const uint64_t o_sink = test_v41_model_alloc(NH * 4); memcpy(test_v41_model_ptr(o_sink), sink, NH * 4); TEST_ASSERT(test_v41_map());
    const uint64_t nq = (uint64_t)N * NH * HD, nw = (uint64_t)(W + N) * HD;
    float *q = malloc(nq * 4), *go = malloc(nq * 4), *win = malloc(nw * 4), *crow = malloc((size_t)NG * HD * 4), *comp = malloc((size_t)NG * HD * 4);
    float *o = malloc(nq * 4), *oref = malloc(nq * 4);
    test_v41_fill_bf16(q, nq, 0.5f); test_v41_fill_bf16(go, nq, 1.f); test_v41_fill_bf16(win, nw, 0.5f); test_v41_fill_bf16(crow, (uint64_t)NG * HD, 0.5f);
    int32_t idx[5 * 3]; for (uint32_t i = 0; i < N * TOPK; i++) idx[i] = (int32_t)((i * 3u + 1u) % (NG + 1u)) - 1;
    ds4_gpu_tensor *tq = test_v41_tensor(q, nq * 4), *tgo = test_v41_tensor(go, nq * 4), *tw = test_v41_tensor(win, nw * 4), *trow = test_v41_tensor(crow, (size_t)NG * HD * 4);
    ds4_gpu_tensor *tc = test_v41_tensor(NULL, (uint64_t)NG * DS4_V41_CKV_BYTES), *tidx = test_v41_tensor(idx, N * TOPK * 4), *to = test_v41_tensor(NULL, nq * 4);
    ds4_gpu_tensor *tgq = test_v41_tensor(NULL, nq * 4), *tgkv = test_v41_tensor(NULL, (uint64_t)N * HD * 4), *tgc = test_v41_tensor(NULL, (uint64_t)NG * HD * 4);
    TEST_ASSERT(ds4_gpu_v41_ckv_pack_tensor(tc, 0, trow, NG, NULL, 0, 0, 0) != 0);
    uint8_t *cache = malloc((size_t)NG * DS4_V41_CKV_BYTES); TEST_ASSERT(test_v41_read(tc, cache, (uint64_t)NG * DS4_V41_CKV_BYTES));
    for (uint32_t g = 0; g < NG; g++) for (uint32_t d = 0; d < HD; d++) comp[g * HD + d] = ckv_get(cache + (size_t)g * DS4_V41_CKV_BYTES, d);
    /* Frozen synthetic indices include noncausal entries; this is independent
     * of the Metal forward precision setting. Preserve the CUDA fixture. */
#ifdef __APPLE__
    const uint32_t forward_ratio = 0;
#else
    const uint32_t forward_ratio = 4;
#endif
    TEST_ASSERT(ds4_gpu_v41_sparse_attn_tensor(to, tq, tw, tc, tidx, test_v41_model_map(), test_v41_model_size(), o_sink, N, P0, W, NG, TOPK, forward_ratio, NH, HD, scale, 0, 1, 0, NULL, 0) != 0);
    TEST_ASSERT(ds4_gpu_bwd_sparse_attn_tensor(tgq, tgkv, tgc, tgo, to, tq, tw, tc, tidx, test_v41_model_map(), test_v41_model_size(), o_sink, N, P0, W, NG, TOPK, NH, HD, scale) != 0);
    float *gq = malloc(nq * 4), *gkv = malloc((uint64_t)N * HD * 4), *gc = malloc((uint64_t)NG * HD * 4), *gwin = calloc(nw, 4);
    TEST_ASSERT(test_v41_read(to, o, nq * 4) && test_v41_read(tgq, gq, nq * 4) && test_v41_read(tgkv, gkv, (uint64_t)N * HD * 4) && test_v41_read(tgc, gc, (uint64_t)NG * HD * 4));
    memcpy(gwin + (size_t)W * HD, gkv, (size_t)N * HD * 4);   /* gkv[尾窗第 i 位] ↔ 缓冲第 W+i 行; 环头(前缀键)行梯度恒 0, 不探 */
    at_ctx c = { N, P0, NH, HD, W, NG, TOPK, go, q, win, sink, comp, idx, scale, oref };
    (void)at_loss(&c, q, win, comp); c.o = NULL;
    snprintf(nm, sizeof nm, "sparse_attn fwd pos0=%u", P0); test_v41_cmp(nm, o, oref, nq, 2e-2f, 1e-3f);
    uint32_t probes[12]; for (uint32_t k = 0; k < 12; k++) probes[k] = (uint32_t)((k * 2654435761ull) % nq);
    snprintf(nm, sizeof nm, "bwd sparse_attn gq pos0=%u", P0); test_v41_fd_probe(nm, f_at_q, &c, q, probes, 12, gq, 2e-3f, 3e-2f, test_v41_rms(gq, nq));
    for (uint32_t k = 0; k < 12; k++) probes[k] = (uint32_t)(W * HD + (k * 2654435761ull) % ((uint64_t)N * HD));
    snprintf(nm, sizeof nm, "bwd sparse_attn gkv pos0=%u", P0); test_v41_fd_probe(nm, f_at_win, &c, win, probes, 12, gwin, 2e-3f, 3e-2f, test_v41_rms(gkv, (uint64_t)N * HD));
    for (uint32_t k = 0; k < 12; k++) probes[k] = (uint32_t)((k * 2654435761ull) % ((uint64_t)NG * HD));
    snprintf(nm, sizeof nm, "bwd sparse_attn gcomp pos0=%u", P0); test_v41_fd_probe(nm, f_at_comp, &c, comp, probes, 12, gc, 2e-3f, 3e-2f, test_v41_rms(gc, (uint64_t)NG * HD));
    ds4_gpu_tensor_free(tq); ds4_gpu_tensor_free(tgo); ds4_gpu_tensor_free(tw); ds4_gpu_tensor_free(trow); ds4_gpu_tensor_free(tc); ds4_gpu_tensor_free(tidx); ds4_gpu_tensor_free(to);
    ds4_gpu_tensor_free(tgq); ds4_gpu_tensor_free(tgkv); ds4_gpu_tensor_free(tgc);
    free(q); free(go); free(win); free(crow); free(comp); free(o); free(oref); free(cache); free(gq); free(gkv); free(gc); free(gwin);
}
void test_v41_bwd_attn(void) {
    TEST_ASSERT(ds4_gpu_init());
    test_v41_seed(0x63);
    attn_case(0);
    attn_case(2);
    attn_case(6);
}
#else
typedef int ds4_t_v41_bwd_attn_nonempty_tu;
#endif
