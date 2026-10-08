/* t_metal_v41_bwd2.c — V4.1 Metal 反传回归后半(2026-10-08): 稀疏注意力反向(FD 穿过 CPU 光滑注意力) / 教师 top-K + KL 梯度 /
 * 草稿块注意力前向(vs CPU)与反向(FD) / 陪审团 / 总变差损失(vs CPU + FD) / markov 表取行与散加。 */
#include "test_internal.h"
#if !defined(DS4_NO_GPU) && defined(__APPLE__)   /* Metal 专用: CUDA 构建里这些核的形状闸不同(专家核只认 11/12/13 位、注意力只认 64 头), 合成小形状不适用 */
#include "../src/common/ds4_fp8.h"
#include "../ds4_gpu_v41.h"
#include "../ds4_gpu_bwd.h"

static uint64_t put_f32(const float *v, uint64_t n) { const uint64_t off = test_v41_model_alloc(n * 4); memcpy(test_v41_model_ptr(off), v, (size_t)n * 4); return off; }
static float ckv_get(const uint8_t *row, uint32_t d) {
    const uint8_t by = row[d >> 1], n4 = (d & 1) ? (by >> 4) : (by & 15);
    return test_v41_bf16r(ds4_fp4_nibble_to_f32(n4) * ds4_e4m3fn_to_f32(row[DS4_V41_CKV_NIB + (d >> 4)]));
}
/* CPU 光滑注意力(pos0 = 0, 窗口缓冲第 W+i 行 = 位置 i; 值 = 键; sink 只进分母): L = <go, o> */
typedef struct { uint32_t N, NH, HD, W, NG, TOPK; const float *go, *q, *win, *sink, *comp; const int32_t *idx; float scale; } at_ctx;
static double at_loss(const at_ctx *c, const float *q, const float *win, const float *comp) {
    double L = 0.0; float *keys = malloc((size_t)(c->W + c->TOPK) * c->HD * 4); double *s = malloc((c->W + c->TOPK) * 8);
    for (uint32_t i = 0; i < c->N; i++) {
        const uint32_t lo = i + 1 > c->W ? i + 1 - c->W : 0, nwin = i - lo + 1; uint32_t nk = 0;
        for (uint32_t kk = 0; kk < nwin; kk++) { memcpy(keys + nk * c->HD, win + (c->W + lo + kk) * c->HD, c->HD * 4); nk++; }
        for (uint32_t kk = 0; kk < c->TOPK; kk++) { const int32_t g = c->idx[i * c->TOPK + kk]; if (g < 0 || (uint32_t)g >= c->NG) continue; memcpy(keys + nk * c->HD, comp + (uint64_t)g * c->HD, c->HD * 4); nk++; }
        for (uint32_t h = 0; h < c->NH; h++) {
            const float *qh = q + ((uint64_t)i * c->NH + h) * c->HD, *gh = c->go + ((uint64_t)i * c->NH + h) * c->HD;
            double mx = -1e30; for (uint32_t k = 0; k < nk; k++) { double d = 0; for (uint32_t e = 0; e < c->HD; e++) d += (double)qh[e] * keys[k * c->HD + e]; s[k] = d * c->scale; mx = fmax(mx, s[k]); }
            double den = exp((double)c->sink[h] - mx);
            for (uint32_t k = 0; k < nk; k++) den += exp(s[k] - mx);
            for (uint32_t k = 0; k < nk; k++) { const double p = exp(s[k] - mx) / den; double dd = 0; for (uint32_t e = 0; e < c->HD; e++) dd += (double)gh[e] * keys[k * c->HD + e]; L += p * dd; }
        }
    }
    free(keys); free(s); return L;
}
static double f_at_q(void *c_, const float *q) { const at_ctx *c = c_; return at_loss(c, q, c->win, c->comp); }
static double f_at_win(void *c_, const float *w) { const at_ctx *c = c_; return at_loss(c, c->q, w, c->comp); }
static double f_at_comp(void *c_, const float *cp) { const at_ctx *c = c_; return at_loss(c, c->q, c->win, cp); }
typedef double (*fd_fn)(void *ctx, const float *buf);
static void fd_probe(const char *name, fd_fn f, void *ctx, float *buf, const uint32_t *idxs, uint32_t np, const float *g, float h, float tol, float gscale) {
    float worst = 0.f, wfd = 0.f; uint32_t at = 0;
    for (uint32_t k = 0; k < np; k++) {
        const uint32_t i = idxs[k]; const float o = buf[i];
        buf[i] = o + h; const double fp = f(ctx, buf); buf[i] = o - h; const double fm = f(ctx, buf); buf[i] = o;
        const float fd = (float)((fp - fm) / (2.0 * h)), e = fabsf(fd - g[i]);
        if (e > worst) { worst = e; at = i; wfd = fd; }
    }
    const float lim = tol * gscale;
    fprintf(stderr, "ds4: [v41-test] %-28s FD %u 点: 最差 |fd−g|=%.3g (限 %.3g) @%u fd=%.5g g=%.5g %s\n", name, np, worst, lim, at, wfd, g[at], worst <= lim ? "ok" : "★FAIL★");
    TEST_ASSERT(worst <= lim);
}
static float rms_of(const float *g, uint64_t n) { double s = 0; for (uint64_t i = 0; i < n; i++) s += (double)g[i] * g[i]; return (float)sqrt(s / n); }
static void test_attn_bwd(void) {
    const uint32_t N = 3, NH = 8, HD = 512, W = 4, NG = 5, TOPK = 3; const float scale = 0.06f;
    TEST_ASSERT(test_v41_model_begin(NH * 4 + 4096));
    float sink[8]; for (uint32_t h = 0; h < NH; h++) sink[h] = test_v41_randf();
    const uint64_t o_sink = put_f32(sink, NH); TEST_ASSERT(test_v41_map());
    const uint64_t nq = (uint64_t)N * NH * HD;
    float *q = malloc(nq * 4), *go = malloc(nq * 4), *win = malloc((size_t)(W + N) * HD * 4), *crow = malloc((size_t)NG * HD * 4), *comp = malloc((size_t)NG * HD * 4), *o = malloc(nq * 4);
    test_v41_fill_bf16(q, nq, 0.5f); test_v41_fill_bf16(go, nq, 1.f); test_v41_fill_bf16(win, (uint64_t)(W + N) * HD, 0.5f); test_v41_fill_bf16(crow, (uint64_t)NG * HD, 0.5f);
    int32_t idx[3 * 3]; for (uint32_t i = 0; i < N * TOPK; i++) idx[i] = (int32_t)((i * 3u + 1u) % (NG + 1u)) - 1;
    ds4_gpu_tensor *tq = test_v41_tensor(q, nq * 4), *tgo = test_v41_tensor(go, nq * 4), *tw = test_v41_tensor(win, (size_t)(W + N) * HD * 4), *trow = test_v41_tensor(crow, (size_t)NG * HD * 4);
    ds4_gpu_tensor *tc = test_v41_tensor(NULL, (uint64_t)NG * DS4_V41_CKV_BYTES), *tidx = test_v41_tensor(idx, N * TOPK * 4), *to = test_v41_tensor(NULL, nq * 4);
    ds4_gpu_tensor *tgq = test_v41_tensor(NULL, nq * 4), *tgkv = test_v41_tensor(NULL, (uint64_t)N * HD * 4), *tgc = test_v41_tensor(NULL, (uint64_t)NG * HD * 4);
    TEST_ASSERT(ds4_gpu_v41_ckv_pack_tensor(tc, 0, trow, NG, NULL, 0, 0, 0) != 0);
    uint8_t *cache = malloc((size_t)NG * DS4_V41_CKV_BYTES); TEST_ASSERT(test_v41_read(tc, cache, (uint64_t)NG * DS4_V41_CKV_BYTES));
    for (uint32_t g = 0; g < NG; g++) for (uint32_t d = 0; d < HD; d++) comp[g * HD + d] = ckv_get(cache + (size_t)g * DS4_V41_CKV_BYTES, d);
    TEST_ASSERT(ds4_gpu_v41_sparse_attn_tensor(to, tq, tw, tc, tidx, test_v41_model_map(), test_v41_model_size(), o_sink, N, 0, W, NG, TOPK, 4, NH, HD, scale, 0, 1, 0, NULL, 0) != 0);
    TEST_ASSERT(ds4_gpu_bwd_sparse_attn_tensor(tgq, tgkv, tgc, tgo, to, tq, tw, tc, tidx, test_v41_model_map(), test_v41_model_size(), o_sink, N, W, NG, TOPK, NH, HD, scale) != 0);
    float *gq = malloc(nq * 4), *gkv = malloc((uint64_t)N * HD * 4), *gc = malloc((uint64_t)NG * HD * 4), *gwin = calloc((W + N) * HD, 4);
    TEST_ASSERT(test_v41_read(tgq, gq, nq * 4) && test_v41_read(tgkv, gkv, (uint64_t)N * HD * 4) && test_v41_read(tgc, gc, (uint64_t)NG * HD * 4));
    memcpy(gwin + (size_t)W * HD, gkv, (size_t)N * HD * 4);   /* gkv[位置] ↔ 缓冲第 W+位置 行 */
    at_ctx c = { N, NH, HD, W, NG, TOPK, go, q, win, sink, comp, idx, scale };
    uint32_t probes[12]; for (uint32_t k = 0; k < 12; k++) probes[k] = (uint32_t)((k * 2654435761ull) % nq);
    fd_probe("bwd sparse_attn gq", f_at_q, &c, q, probes, 12, gq, 2e-3f, 3e-2f, rms_of(gq, nq));
    for (uint32_t k = 0; k < 12; k++) probes[k] = (uint32_t)(W * HD + (k * 2654435761ull) % ((uint64_t)N * HD));
    fd_probe("bwd sparse_attn gkv", f_at_win, &c, win, probes, 12, gwin, 2e-3f, 3e-2f, rms_of(gkv, (uint64_t)N * HD));
    for (uint32_t k = 0; k < 12; k++) probes[k] = (uint32_t)((k * 2654435761ull) % ((uint64_t)NG * HD));
    fd_probe("bwd sparse_attn gcomp", f_at_comp, &c, comp, probes, 12, gc, 2e-3f, 3e-2f, rms_of(gc, (uint64_t)NG * HD));
    ds4_gpu_tensor_free(tq); ds4_gpu_tensor_free(tgo); ds4_gpu_tensor_free(tw); ds4_gpu_tensor_free(trow); ds4_gpu_tensor_free(tc); ds4_gpu_tensor_free(tidx); ds4_gpu_tensor_free(to);
    ds4_gpu_tensor_free(tgq); ds4_gpu_tensor_free(tgkv); ds4_gpu_tensor_free(tgc);
    free(q); free(go); free(win); free(crow); free(comp); free(o); free(cache); free(gq); free(gkv); free(gc); free(gwin);
}
/* 教师 top-K 与 KL: 学生/教师行各一, 比 CPU */
static void test_kl(void) {
    const uint32_t V = 4000, M = 2, K = 8;
    float *zt = malloc((size_t)M * V * 4), *zs = malloc((size_t)M * V * 4), *zt2 = malloc((size_t)M * V * 4);
    for (uint32_t i = 0; i < M * V; i++) { zt[i] = test_v41_randf() * 4.f; zs[i] = zt[i] + test_v41_randf(); }
    memcpy(zt2, zt, (size_t)M * V * 4);
    ds4_gpu_tensor *tzt = test_v41_tensor(zt, (size_t)M * V * 4), *tzs = test_v41_tensor(zs, (size_t)M * V * 4), *tid = test_v41_tensor(NULL, M * K * 4), *ttp = test_v41_tensor(NULL, M * K * 4), *ttr = test_v41_tensor(NULL, M * 4);
    TEST_ASSERT(ds4_gpu_bwd_topk_tensor(tid, ttp, ttr, tzt, 0, M, V, K) != 0);
    int32_t *tid_h = malloc(M * K * 4); float *tp_h = malloc(M * K * 4), *tr_h = malloc(M * 4);
    TEST_ASSERT(test_v41_read(tid, tid_h, M * K * 4) && test_v41_read(ttp, tp_h, M * K * 4) && test_v41_read(ttr, tr_h, M * 4));
    int bad = 0;
    for (uint32_t m = 0; m < M; m++) {
        double mx = -1e30; for (uint32_t i = 0; i < V; i++) mx = fmax(mx, zt2[m * V + i]); double Z = 0; for (uint32_t i = 0; i < V; i++) Z += exp(zt2[m * V + i] - mx);
        int used[4000] = { 0 }; double ps = 0;
        for (uint32_t k = 0; k < K; k++) { int bi = -1; float bv = -3e38f; for (uint32_t i = 0; i < V; i++) if (!used[i] && zt2[m * V + i] > bv) { bv = zt2[m * V + i]; bi = (int)i; } used[bi] = 1; if (tid_h[m * K + k] != bi) bad++; const double p = exp(bv - mx) / Z; ps += p; if (fabs(tp_h[m * K + k] - p) > 1e-6) bad++; }
        if (fabs(tr_h[m] - (1.0 - ps)) > 1e-5) bad++;
    }
    fprintf(stderr, "ds4: [v41-test] bwd topk 教师: 不一致 %d, trest[0]=%.6f\n", bad, tr_h[0]); TEST_ASSERT(bad == 0);
    ds4_gpu_tensor *tg = test_v41_tensor(NULL, (size_t)M * V * 4), *tl = test_v41_tensor(NULL, M * 4);
    TEST_ASSERT(ds4_gpu_bwd_kl_topk_tensor(tg, tl, tzs, 0, M, V, tid, ttp, ttr, NULL, K, 0.5f) != 0);
    float *g = malloc((size_t)M * V * 4), *loss = malloc(M * 4), *gr = malloc((size_t)M * V * 4);
    TEST_ASSERT(test_v41_read(tg, g, (size_t)M * V * 4) && test_v41_read(tl, loss, M * 4));
    for (uint32_t m = 0; m < M; m++) {
        double mx = -1e30; for (uint32_t i = 0; i < V; i++) mx = fmax(mx, zs[m * V + i]); double Z = 0; for (uint32_t i = 0; i < V; i++) Z += exp(zs[m * V + i] - mx);
        int on[4000] = { 0 }; for (uint32_t k = 0; k < K; k++) on[tid_h[m * K + k]] = 1;
        double rs = 0; for (uint32_t i = 0; i < V; i++) if (!on[i]) rs += exp(zs[m * V + i] - mx) / Z;
        double l = 0; for (uint32_t k = 0; k < K; k++) { const double pt = tp_h[m * K + k], psv = exp(zs[m * V + tid_h[m * K + k]] - mx) / Z; if (pt > 0) l += pt * (log(pt) - log(psv)); }
        const double rt = tr_h[m]; if (rt > 1e-12) l += rt * (log(rt) - log(rs));
        const float off = (float)(1.0 - rt / rs);
        for (uint32_t i = 0; i < V; i++) gr[m * V + i] = 0.5f * (float)(exp(zs[m * V + i] - mx) / Z) * off;
        for (uint32_t k = 0; k < K; k++) { const int id = tid_h[m * K + k]; gr[m * V + id] = 0.5f * (float)(exp(zs[m * V + id] - mx) / Z - tp_h[m * K + k]); }
        fprintf(stderr, "ds4: [v41-test] bwd kl 行 %u: loss gpu %.6f cpu %.6f\n", m, loss[m], l); TEST_ASSERT(fabs(loss[m] - l) < 1e-4 * (1 + fabs(l)));
    }
    test_v41_cmp("bwd kl grad", g, gr, (uint64_t)M * V, 1e-4f, 1e-5f);
    ds4_gpu_tensor_free(tzt); ds4_gpu_tensor_free(tzs); ds4_gpu_tensor_free(tid); ds4_gpu_tensor_free(ttp); ds4_gpu_tensor_free(ttr); ds4_gpu_tensor_free(tg); ds4_gpu_tensor_free(tl);
    free(zt); free(zs); free(zt2); free(tid_h); free(tp_h); free(tr_h); free(g); free(loss); free(gr);
}
/* 草稿块注意力: 块 b 的键 = 历史 min(window, bpos[b]) 行 + 块内 B 行(全可见); 前向 vs CPU, 反向 FD(q / 块内行) */
typedef struct { uint32_t nb, B, W, NH, HD, hbase; const float *go, *q, *hist, *blk, *sink; const int32_t *bpos; float scale; } dk_ctx;
static double dk_loss(const dk_ctx *c, const float *q, const float *blk, float *o_out, float *lse_out) {
    double L = 0; double s[64];
    for (uint32_t r = 0; r < c->nb * c->B; r++) {
        const uint32_t b = r / c->B, ib = (uint32_t)c->bpos[b], nh = ib < c->W ? ib : c->W, nk = nh + c->B;
        for (uint32_t h = 0; h < c->NH; h++) {
            const float *qh = q + ((uint64_t)r * c->NH + h) * c->HD; double mx = -1e30;
            for (uint32_t k = 0; k < nk; k++) { const float *kr = k < nh ? c->hist + (uint64_t)(ib - nh + k - c->hbase) * c->HD : blk + ((uint64_t)b * c->B + (k - nh)) * c->HD; double d = 0; for (uint32_t e = 0; e < c->HD; e++) d += (double)qh[e] * kr[e]; s[k] = d * c->scale; mx = fmax(mx, s[k]); }
            double den = exp((double)c->sink[h] - mx); for (uint32_t k = 0; k < nk; k++) den += exp(s[k] - mx);
            for (uint32_t e = 0; e < c->HD; e++) { double acc = 0; for (uint32_t k = 0; k < nk; k++) { const float *kr = k < nh ? c->hist + (uint64_t)(ib - nh + k - c->hbase) * c->HD : blk + ((uint64_t)b * c->B + (k - nh)) * c->HD; acc += exp(s[k] - mx) / den * kr[e]; }
                if (o_out) o_out[((uint64_t)r * c->NH + h) * c->HD + e] = test_v41_bf16r((float)acc); L += (double)c->go[((uint64_t)r * c->NH + h) * c->HD + e] * acc; }
            if (lse_out) lse_out[r * c->NH + h] = (float)(mx + log(den));
        }
    }
    return L;
}
static double f_dk_q(void *c_, const float *q) { const dk_ctx *c = c_; return dk_loss(c, q, c->blk, NULL, NULL); }
static double f_dk_blk(void *c_, const float *b) { const dk_ctx *c = c_; return dk_loss(c, c->q, b, NULL, NULL); }
static void test_draft(void) {
    const uint32_t nb = 3, B = 5, W = 6, NH = 8, HD = 512, hbase = 0, nhist = 30; const float scale = 0.05f;
    TEST_ASSERT(test_v41_model_begin(NH * 4 + 4096));
    float sink[8]; for (uint32_t h = 0; h < NH; h++) sink[h] = test_v41_randf();
    const uint64_t o_sink = put_f32(sink, NH); TEST_ASSERT(test_v41_map());
    const uint32_t R = nb * B; const uint64_t nq = (uint64_t)R * NH * HD;
    float *q = malloc(nq * 4), *go = malloc(nq * 4), *hist = malloc((size_t)nhist * HD * 4), *blk = malloc((size_t)R * HD * 4), *o = malloc(nq * 4), *oref = malloc(nq * 4), *lse = malloc(R * NH * 4), *lser = malloc(R * NH * 4);
    test_v41_fill_bf16(q, nq, 0.5f); test_v41_fill_bf16(go, nq, 1.f); test_v41_fill_bf16(hist, (uint64_t)nhist * HD, 0.5f); test_v41_fill_bf16(blk, (uint64_t)R * HD, 0.5f);
    const int32_t bpos[3] = { 3, 20, 29 };   /* 块首位绝对位置: 历史 3 / 6 / 6 行(第 0 块历史不满 window); hist 第 0 行 = 位置 0, 共 30 行 */
    ds4_gpu_tensor *tq = test_v41_tensor(q, nq * 4), *tgo = test_v41_tensor(go, nq * 4), *th = test_v41_tensor(hist, (size_t)nhist * HD * 4), *tb = test_v41_tensor(blk, (size_t)R * HD * 4);
    ds4_gpu_tensor *tbp = test_v41_tensor(bpos, 12), *to = test_v41_tensor(NULL, nq * 4), *tl = test_v41_tensor(NULL, R * NH * 4), *tgq = test_v41_tensor(NULL, nq * 4), *tgb = test_v41_tensor(NULL, (size_t)R * HD * 4);
    TEST_ASSERT(ds4_gpu_draft_attn_fwd_tensor(to, tl, tq, th, hbase, tb, tbp, nb, B, W, test_v41_model_map(), test_v41_model_size(), o_sink, NH, HD, scale) != 0);
    TEST_ASSERT(test_v41_read(to, o, nq * 4) && test_v41_read(tl, lse, R * NH * 4));
    dk_ctx c = { nb, B, W, NH, HD, hbase, go, q, hist, blk, sink, bpos, scale };
    dk_loss(&c, q, blk, oref, lser);
    test_v41_cmp("draft attn fwd o", o, oref, nq, 3e-2f, 1e-3f);   /* 核里 p 舍 bf16 再乘 v, 元素可差一个 bf16 格 */ test_v41_cmp("draft attn fwd lse", lse, lser, R * NH, 1e-3f, 1e-3f);
    TEST_ASSERT(ds4_gpu_draft_attn_bwd_tensor(tgq, tgb, tgo, to, tq, tl, th, hbase, tb, tbp, nb, B, W, NH, HD, scale) != 0);
    float *gq = malloc(nq * 4), *gb = malloc((size_t)R * HD * 4);
    TEST_ASSERT(test_v41_read(tgq, gq, nq * 4) && test_v41_read(tgb, gb, (size_t)R * HD * 4));
    uint32_t probes[12]; for (uint32_t k = 0; k < 12; k++) probes[k] = (uint32_t)((k * 2654435761ull) % nq);
    fd_probe("draft attn bwd gq", f_dk_q, &c, q, probes, 12, gq, 2e-3f, 3e-2f, rms_of(gq, nq));
    for (uint32_t k = 0; k < 12; k++) probes[k] = (uint32_t)((k * 2654435761ull) % ((uint64_t)R * HD));
    fd_probe("draft attn bwd gblk", f_dk_blk, &c, blk, probes, 12, gb, 2e-3f, 3e-2f, rms_of(gb, (uint64_t)R * HD));
    /* 陪审团与总变差: 学生 ls / 教师 lt */
    { const uint32_t V = 3000, M = 2, K = 16; const float T = 1.3f;
      float *ls = malloc((size_t)M * V * 4), *lt = malloc((size_t)M * V * 4), *out = malloc(M * 16);
      for (uint32_t i = 0; i < M * V; i++) { lt[i] = test_v41_randf() * 3.f; ls[i] = lt[i] + test_v41_randf() * 0.7f; }
      ds4_gpu_tensor *tls = test_v41_tensor(ls, (size_t)M * V * 4), *tlt = test_v41_tensor(lt, (size_t)M * V * 4), *tout = test_v41_tensor(NULL, M * 16);
      TEST_ASSERT(ds4_gpu_draft_jury_tensor(tout, tls, tlt, M, V, T) != 0 && test_v41_read(tout, out, M * 16));
      for (uint32_t m = 0; m < M; m++) {
          double ms = -1e30, mt = -1e30; int as = 0, at = 0;
          for (uint32_t i = 0; i < V; i++) { if (ls[m * V + i] > ms) { ms = ls[m * V + i]; as = (int)i; } if (lt[m * V + i] > mt) { mt = lt[m * V + i]; at = (int)i; } }
          double zs = 0, zt = 0; for (uint32_t i = 0; i < V; i++) { zs += exp((ls[m * V + i] - ms) / T); zt += exp((lt[m * V + i] - mt) / T); }
          double smin = 0; for (uint32_t i = 0; i < V; i++) { const double p = exp((lt[m * V + i] - mt) / T) / zt, qq = exp((ls[m * V + i] - ms) / T) / zs; smin += p < qq ? p : qq; }
          const double pa = exp((lt[m * V + as] - mt) / T) / zt;
          fprintf(stderr, "ds4: [v41-test] jury 行 %u: Σmin gpu %.5f cpu %.5f | p(argmax q) gpu %.5f cpu %.5f | same %g (cpu %d)\n", m, out[m * 4], smin, out[m * 4 + 1], pa, out[m * 4 + 2], as == at);
          TEST_ASSERT(fabs(out[m * 4] - smin) < 2e-4 && fabs(out[m * 4 + 1] - pa) < 1e-5 && (out[m * 4 + 2] > 0.5f) == (as == at));
      }
      /* 总变差: 教师 top-K 从 lt 出(GPU topk 核), 再 TV 的损失与梯度 vs CPU */
      float *lt2 = malloc((size_t)M * V * 4); memcpy(lt2, lt, (size_t)M * V * 4);
      ds4_gpu_tensor *tlt2 = test_v41_tensor(lt2, (size_t)M * V * 4), *tid = test_v41_tensor(NULL, M * K * 4), *ttp = test_v41_tensor(NULL, M * K * 4), *ttr = test_v41_tensor(NULL, M * 4);
      TEST_ASSERT(ds4_gpu_bwd_topk_tensor(tid, ttp, ttr, tlt2, 0, M, V, K) != 0);
      int32_t *tid_h = malloc(M * K * 4); float *tp_h = malloc(M * K * 4), *tr_h = malloc(M * 4);
      TEST_ASSERT(test_v41_read(tid, tid_h, M * K * 4) && test_v41_read(ttp, tp_h, M * K * 4) && test_v41_read(ttr, tr_h, M * 4));
      ds4_gpu_tensor *tg = test_v41_tensor(NULL, (size_t)M * V * 4), *tloss = test_v41_tensor(NULL, M * 4);
      TEST_ASSERT(ds4_gpu_draft_tv_tensor(tg, tloss, tls, M, V, tid, ttp, ttr, K, T, 0.7f) != 0);
      float *g = malloc((size_t)M * V * 4), *loss = malloc(M * 4), *gr = malloc((size_t)M * V * 4);
      TEST_ASSERT(test_v41_read(tg, g, (size_t)M * V * 4) && test_v41_read(tloss, loss, M * 4));
      for (uint32_t m = 0; m < M; m++) {
          double mx = -1e30; for (uint32_t i = 0; i < V; i++) mx = fmax(mx, ls[m * V + i]); double Z = 0; for (uint32_t i = 0; i < V; i++) Z += exp((ls[m * V + i] - mx) / T);
          int on[3000] = { 0 }; for (uint32_t k = 0; k < K; k++) on[tid_h[m * K + k]] = 1;
          double qoff = 0, tvon = 0, qson = 0;
          for (uint32_t i = 0; i < V; i++) if (!on[i]) qoff += exp((ls[m * V + i] - mx) / T) / Z;
          for (uint32_t k = 0; k < K; k++) { const double qq = exp((ls[m * V + tid_h[m * K + k]] - mx) / T) / Z, d = qq - tp_h[m * K + k]; tvon += fabs(d); qson += d > 0 ? qq : (d < 0 ? -qq : 0); }
          const double A = qoff + qson, cc = 0.7 / T, l = 0.5 * (tvon + qoff + tr_h[m]);
          for (uint32_t i = 0; i < V; i++) { const double qq = exp((ls[m * V + i] - mx) / T) / Z; gr[m * V + i] = (float)(cc * qq * 0.5 * (1.0 - A)); }
          for (uint32_t k = 0; k < K; k++) { const int id = tid_h[m * K + k]; const double qq = exp((ls[m * V + id] - mx) / T) / Z, d = qq - tp_h[m * K + k], s = d > 0 ? 1 : (d < 0 ? -1 : 0); gr[m * V + id] = (float)(cc * qq * 0.5 * (s - A)); }
          fprintf(stderr, "ds4: [v41-test] tv 行 %u: loss gpu %.6f cpu %.6f\n", m, loss[m], l); TEST_ASSERT(fabs(loss[m] - l) < 1e-4);
      }
      test_v41_cmp("draft tv grad", g, gr, (uint64_t)M * V, 1e-3f, 1e-5f);
      ds4_gpu_tensor_free(tls); ds4_gpu_tensor_free(tlt); ds4_gpu_tensor_free(tout); ds4_gpu_tensor_free(tlt2); ds4_gpu_tensor_free(tid); ds4_gpu_tensor_free(ttp); ds4_gpu_tensor_free(ttr); ds4_gpu_tensor_free(tg); ds4_gpu_tensor_free(tloss);
      free(ls); free(lt); free(out); free(lt2); free(tid_h); free(tp_h); free(tr_h); free(g); free(loss); free(gr); }
    /* markov 表: 取行 / 散加 / 偏置 GEMM 与反向 */
    { const uint32_t Vm = 50, Rr = 64, n = 6; float *tab = malloc(Vm * Rr * 4), *out = malloc(n * Rr * 4), *ref = malloc(n * Rr * 4), *gtab = calloc(Vm * Rr, 4), *gr = calloc(Vm * Rr, 4);
      const int32_t ids[8] = { 3, 7, 3, 49, -1, 60, 7, 0 };
      for (uint32_t i = 0; i < Vm * Rr; i++) tab[i] = test_v41_randf();
      ds4_gpu_tensor *ttab = test_v41_tensor(tab, Vm * Rr * 4), *tids = test_v41_tensor(ids, 32), *tout = test_v41_tensor(NULL, n * Rr * 4), *tgt = test_v41_tensor(gtab, Vm * Rr * 4);
      TEST_ASSERT(ds4_gpu_draft_rows_dev_tensor(tout, ttab, tids, 1, n, Rr, Vm) != 0 && test_v41_read(tout, out, n * Rr * 4));
      for (uint32_t r = 0; r < n; r++) for (uint32_t d = 0; d < Rr; d++) { const int id = ids[1 + r]; ref[r * Rr + d] = (id >= 0 && (uint32_t)id < Vm) ? tab[id * Rr + d] : 0.f; }
      test_v41_cmp("draft rows_dev", out, ref, n * Rr, 0.f, 1e-3f);
      TEST_ASSERT(ds4_gpu_draft_rows_scatter_tensor(tgt, tout, tids, n, Rr, Vm) != 0 && test_v41_read(tgt, gtab, Vm * Rr * 4));
      for (uint32_t r = 0; r < n; r++) { const int id = ids[r]; if (id >= 0 && (uint32_t)id < Vm) for (uint32_t d = 0; d < Rr; d++) gr[id * Rr + d] += out[r * Rr + d]; }
      test_v41_cmp("draft rows_scatter", gtab, gr, Vm * Rr, 1e-6f, 1e-3f);
      ds4_gpu_tensor_free(ttab); ds4_gpu_tensor_free(tids); ds4_gpu_tensor_free(tout); ds4_gpu_tensor_free(tgt); free(tab); free(out); free(ref); free(gtab); free(gr); }
    ds4_gpu_tensor_free(tq); ds4_gpu_tensor_free(tgo); ds4_gpu_tensor_free(th); ds4_gpu_tensor_free(tb); ds4_gpu_tensor_free(tbp); ds4_gpu_tensor_free(to); ds4_gpu_tensor_free(tl); ds4_gpu_tensor_free(tgq); ds4_gpu_tensor_free(tgb);
    free(q); free(go); free(hist); free(blk); free(o); free(oref); free(lse); free(lser); free(gq); free(gb);
}
void test_metal_v41_bwd_attn(void) {
    test_v41_seed(0x63);
    test_attn_bwd();
    test_kl();
    test_draft();
}
#else
typedef int ds4_t_metal_v41_bwd2_nonempty_tu;
#endif
