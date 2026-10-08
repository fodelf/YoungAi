/* t_metal_v41_bwd.c — V4.1 Metal 反传原语回归(2026-10-08): 对着 CPU 光滑前向的有限差分(中心差分, 双精度累加)或 CPU 解析式比。
 * 覆盖: RMSNorm / hc_pre / hc_post / SwiGLU / 路由权重 / 压缩器池化 / engram 门 / mHC 混合系数(sinkhorn 倒推) / 稀疏注意力 / 低秩放大器 /
 * 教师 top-K + KL / Adam·axpy·Σg²·bf16 存档 / 草稿块注意力前向+反向 / 陪审团 / 总变差。 */
#include "test_internal.h"
#if !defined(DS4_NO_GPU) && defined(__APPLE__)   /* Metal 专用: CUDA 构建里这些核的形状闸不同(专家核只认 11/12/13 位、注意力只认 64 头), 合成小形状不适用 */
#include "../src/common/ds4_fp8.h"
#include "../ds4_gpu_v41.h"
#include "../ds4_gpu_bwd.h"

static float sigm(float z) { return 1.0f / (1.0f + expf(-z)); }
static uint64_t put_f32(const float *v, uint64_t n) { const uint64_t off = test_v41_model_alloc(n * 4); memcpy(test_v41_model_ptr(off), v, (size_t)n * 4); return off; }
/* 有限差分: f 是 CPU 光滑标量函数(对 buf 的 n 个元素), 比 g[n](GPU 梯度); 抽样 nprobe 个点 */
typedef double (*fd_fn)(void *ctx, const float *buf);
static void fd_check(const char *name, fd_fn f, void *ctx, float *buf, uint32_t n, const float *g, uint32_t nprobe, float h, float tol) {
    float worst = 0.f, wref = 0.f; uint32_t at = 0;
    for (uint32_t k = 0; k < nprobe; k++) {
        const uint32_t i = (uint32_t)(((uint64_t)k * 2654435761ull) % n);
        const float o = buf[i];
        buf[i] = o + h; const double fp = f(ctx, buf);
        buf[i] = o - h; const double fm = f(ctx, buf);
        buf[i] = o;
        const float fd = (float)((fp - fm) / (2.0 * h)), e = fabsf(fd - g[i]);
        if (e > worst) { worst = e; at = i; wref = fd; }
    }
    double ss = 0.0; for (uint32_t i = 0; i < n; i++) ss += (double)g[i] * g[i];
    const float rms = (float)sqrt(ss / n), lim = tol * (rms > 1e-3f ? rms : 1e-3f);
    fprintf(stderr, "ds4: [v41-test] %-28s FD %u 点: 梯度 rms=%.4g 最差 |fd−g|=%.3g (限 %.3g) @%u fd=%.5g g=%.5g %s\n", name, nprobe, rms, worst, lim, at, wref, g[at], worst <= lim ? "ok" : "★FAIL★");
    TEST_ASSERT(worst <= lim);
}

/* ---- RMSNorm / hc_pre / hc_post / SwiGLU / 路由 ---- */
typedef struct { uint32_t D, N, HC; const float *w, *gxn, *x, *gx_, *hc, *pre, *gout, *y, *res, *post, *comb; float eps; } rms_ctx;
static double f_rms(void *c_, const float *x) {
    const rms_ctx *c = c_; double L = 0.0;
    for (uint32_t n = 0; n < c->N; n++) { double ss = 0.0; for (uint32_t d = 0; d < c->D; d++) ss += (double)x[n * c->D + d] * x[n * c->D + d]; const double inv = 1.0 / sqrt(ss / c->D + c->eps);
        for (uint32_t d = 0; d < c->D; d++) L += (double)c->gxn[n * c->D + d] * c->w[d] * x[n * c->D + d] * inv; }
    return L;
}
static double f_hcpre(void *c_, const float *hc) { const rms_ctx *c = c_; double L = 0.0;
    for (uint32_t n = 0; n < c->N; n++) for (uint32_t d = 0; d < c->D; d++) { double v = 0.0; for (uint32_t k = 0; k < c->HC; k++) v += (double)c->pre[n * c->HC + k] * hc[(n * c->HC + k) * c->D + d]; L += (double)c->gx_[n * c->D + d] * v; }
    return L; }
static double f_hcpost_y(void *c_, const float *y) { const rms_ctx *c = c_; double L = 0.0;
    for (uint32_t n = 0; n < c->N; n++) for (uint32_t k = 0; k < c->HC; k++) for (uint32_t d = 0; d < c->D; d++) {
        double s = (double)c->post[n * c->HC + k] * y[n * c->D + d]; for (uint32_t j = 0; j < c->HC; j++) s += (double)c->comb[n * c->HC * c->HC + j * c->HC + k] * c->res[(n * c->HC + j) * c->D + d];
        L += (double)c->gout[(n * c->HC + k) * c->D + d] * s; }
    return L; }
static double f_hcpost_comb(void *c_, const float *comb) { rms_ctx c = *(const rms_ctx *)c_; c.comb = comb; return f_hcpost_y(&c, c.y); }
static void test_dense_bwd(void) {
    const uint32_t D = 256, N = 3, HC = 4; const float eps = 1e-6f;
    TEST_ASSERT(test_v41_model_begin(D * 4 + 4096));
    float *w = malloc(D * 4), *x = malloc(N * D * 4), *gxn = malloc(N * D * 4), *gx = malloc(N * D * 4);
    for (uint32_t i = 0; i < D; i++) w[i] = 0.5f + 0.2f * test_v41_randf();
    for (uint32_t i = 0; i < N * D; i++) { x[i] = test_v41_randf(); gxn[i] = test_v41_randf(); }
    const uint64_t o_w = put_f32(w, D);
    TEST_ASSERT(test_v41_map());
    ds4_gpu_tensor *tx = test_v41_tensor(x, N * D * 4), *tg = test_v41_tensor(gxn, N * D * 4), *tgx = test_v41_tensor(NULL, N * D * 4);
    TEST_ASSERT(ds4_gpu_bwd_rms_norm_tensor(tgx, tg, tx, test_v41_model_map(), test_v41_model_size(), o_w, D, N, eps, 0) != 0 && test_v41_read(tgx, gx, N * D * 4));
    rms_ctx c = { D, N, HC, w, gxn, x, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, eps };
    fd_check("bwd rms_norm", f_rms, &c, x, N * D, gx, 24, 1e-3f, 2e-3f);
    /* hc_pre: ghc += pre·gx(累加进 1.0), gpre = Σ gx·hc */
    float *hc = malloc(N * HC * D * 4), *pre = malloc(N * HC * 4), *ghc = malloc(N * HC * D * 4), *gpre = malloc(N * HC * 4);
    for (uint32_t i = 0; i < N * HC * D; i++) { hc[i] = test_v41_randf(); ghc[i] = 1.0f; }
    for (uint32_t i = 0; i < N * HC; i++) pre[i] = 0.3f + test_v41_randf() * 0.2f;
    ds4_gpu_tensor *thc = test_v41_tensor(hc, N * HC * D * 4), *tpre = test_v41_tensor(pre, N * HC * 4), *tghc = test_v41_tensor(ghc, N * HC * D * 4), *tgpre = test_v41_tensor(NULL, N * HC * 4);
    TEST_ASSERT(ds4_gpu_bwd_hc_pre_tensor(tghc, tgpre, tg, thc, tpre, D, HC, N) != 0 && test_v41_read(tghc, ghc, N * HC * D * 4) && test_v41_read(tgpre, gpre, N * HC * 4));
    for (uint32_t i = 0; i < N * HC * D; i++) ghc[i] -= 1.0f;
    c.gx_ = gxn; c.pre = pre;
    fd_check("bwd hc_pre ghc", f_hcpre, &c, hc, N * HC * D, ghc, 24, 1e-3f, 2e-3f);
    { float *ref = malloc(N * HC * 4); for (uint32_t n = 0; n < N; n++) for (uint32_t k = 0; k < HC; k++) { double s = 0; for (uint32_t d = 0; d < D; d++) s += (double)gxn[n * D + d] * hc[(n * HC + k) * D + d]; ref[n * HC + k] = (float)s; }
      test_v41_cmp("bwd hc_pre gpre", gpre, ref, N * HC, 1e-4f, 1e-3f); free(ref); }
    /* hc_post 四个出口 */
    float *gout = malloc(N * HC * D * 4), *y = malloc(N * D * 4), *post = malloc(N * HC * 4), *comb = malloc(N * HC * HC * 4);
    float *gy = malloc(N * D * 4), *gres = malloc(N * HC * D * 4), *gpost = malloc(N * HC * 4), *gcomb = malloc(N * HC * HC * 4);
    for (uint32_t i = 0; i < N * HC * D; i++) { gout[i] = test_v41_randf(); gres[i] = 0.f; }
    for (uint32_t i = 0; i < N * D; i++) y[i] = test_v41_randf();
    for (uint32_t i = 0; i < N * HC; i++) post[i] = test_v41_randf();
    for (uint32_t i = 0; i < N * HC * HC; i++) comb[i] = test_v41_randf();
    ds4_gpu_tensor *tgout = test_v41_tensor(gout, N * HC * D * 4), *ty = test_v41_tensor(y, N * D * 4), *tpost = test_v41_tensor(post, N * HC * 4), *tcomb = test_v41_tensor(comb, N * HC * HC * 4);
    ds4_gpu_tensor *tgy = test_v41_tensor(NULL, N * D * 4), *tgres = test_v41_tensor(gres, N * HC * D * 4), *tgpost = test_v41_tensor(NULL, N * HC * 4), *tgcomb = test_v41_tensor(NULL, N * HC * HC * 4);
    TEST_ASSERT(ds4_gpu_bwd_hc_post_tensor(tgy, tgres, tgpost, tgcomb, tgout, ty, thc, tpost, tcomb, D, HC, N) != 0);
    TEST_ASSERT(test_v41_read(tgy, gy, N * D * 4) && test_v41_read(tgres, gres, N * HC * D * 4) && test_v41_read(tgpost, gpost, N * HC * 4) && test_v41_read(tgcomb, gcomb, N * HC * HC * 4));
    c.gout = gout; c.y = y; c.res = hc; c.post = post; c.comb = comb;
    fd_check("bwd hc_post gy", f_hcpost_y, &c, y, N * D, gy, 16, 1e-3f, 2e-3f);
    fd_check("bwd hc_post gcomb", f_hcpost_comb, &c, comb, N * HC * HC, gcomb, 16, 1e-3f, 2e-3f);
    { float *ref = malloc(N * HC * D * 4); for (uint32_t n = 0; n < N; n++) for (uint32_t j = 0; j < HC; j++) for (uint32_t d = 0; d < D; d++) { double s = 0; for (uint32_t k = 0; k < HC; k++) s += (double)comb[n * HC * HC + j * HC + k] * gout[(n * HC + k) * D + d]; ref[(n * HC + j) * D + d] = (float)s; }
      test_v41_cmp("bwd hc_post gres", gres, ref, N * HC * D, 1e-5f, 1e-3f);
      for (uint32_t n = 0; n < N; n++) for (uint32_t k = 0; k < HC; k++) { double s = 0; for (uint32_t d = 0; d < D; d++) s += (double)gout[(n * HC + k) * D + d] * y[n * D + d]; ref[n * HC + k] = (float)s; }
      test_v41_cmp("bwd hc_post gpost", gpost, ref, N * HC, 1e-4f, 1e-3f); free(ref); }
    /* SwiGLU 反向(含截断): 解析 */
    { const uint32_t M = 1000; const float L = 1.0f; float *g = malloc(M * 4), *u = malloc(M * 4), *gh = malloc(M * 4), *gg = malloc(M * 4), *gu = malloc(M * 4), *rg = malloc(M * 4), *ru = malloc(M * 4);
      for (uint32_t i = 0; i < M; i++) { g[i] = test_v41_randf() * 2.f; u[i] = test_v41_randf() * 2.f; gh[i] = test_v41_randf(); }
      ds4_gpu_tensor *tg2 = test_v41_tensor(g, M * 4), *tu = test_v41_tensor(u, M * 4), *tgh = test_v41_tensor(gh, M * 4), *tgg = test_v41_tensor(NULL, M * 4), *tgu = test_v41_tensor(NULL, M * 4);
      TEST_ASSERT(ds4_gpu_bwd_swiglu_tensor(tgg, tgu, tgh, tg2, tu, M, L) != 0 && test_v41_read(tgg, gg, M * 4) && test_v41_read(tgu, gu, M * 4));
      for (uint32_t i = 0; i < M; i++) { float gv = g[i], uv = u[i]; const int gp = gv < L, up = uv > -L && uv < L; gv = fminf(gv, L); uv = fminf(fmaxf(uv, -L), L); const float sg = sigm(gv); rg[i] = gp ? gh[i] * uv * sg * (1.f + gv * (1.f - sg)) : 0.f; ru[i] = up ? gh[i] * gv * sg : 0.f; }
      test_v41_cmp("bwd swiglu gg", gg, rg, M, 1e-5f, 1e-3f); test_v41_cmp("bwd swiglu gu", gu, ru, M, 1e-5f, 1e-3f);
      ds4_gpu_tensor_free(tg2); ds4_gpu_tensor_free(tu); ds4_gpu_tensor_free(tgh); ds4_gpu_tensor_free(tgg); ds4_gpu_tensor_free(tgu); free(g); free(u); free(gh); free(gg); free(gu); free(rg); free(ru); }
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(tg); ds4_gpu_tensor_free(tgx); ds4_gpu_tensor_free(thc); ds4_gpu_tensor_free(tpre); ds4_gpu_tensor_free(tghc); ds4_gpu_tensor_free(tgpre);
    ds4_gpu_tensor_free(tgout); ds4_gpu_tensor_free(ty); ds4_gpu_tensor_free(tpost); ds4_gpu_tensor_free(tcomb); ds4_gpu_tensor_free(tgy); ds4_gpu_tensor_free(tgres); ds4_gpu_tensor_free(tgpost); ds4_gpu_tensor_free(tgcomb);
    free(w); free(x); free(gxn); free(gx); free(hc); free(pre); free(ghc); free(gpre); free(gout); free(y); free(post); free(comb); free(gy); free(gres); free(gpost); free(gcomb);
}

/* ---- 路由权重反向(FD 穿过 √softplus + 归一) / 压缩器池化 / engram 门 ---- */
typedef struct { uint32_t N, NE, K; const int32_t *sel; const float *gw; float rs; } rt_ctx;
static double f_router(void *c_, const float *z) { const rt_ctx *c = c_; double L = 0.0;
    for (uint32_t t = 0; t < c->N; t++) { double pr[16], S = 1e-20; for (uint32_t k = 0; k < c->K; k++) { const int e = c->sel[t * c->K + k]; const double zz = e >= 0 ? z[t * c->NE + e] : 0.0; pr[k] = e >= 0 ? sqrt(zz > 20 ? zz : log1p(exp(zz))) : 0.0; S += pr[k]; }
        for (uint32_t k = 0; k < c->K; k++) L += (double)c->gw[t * c->K + k] * pr[k] / S * c->rs; }
    return L; }
typedef struct { uint32_t NG, R, D; const float *gp, *kv, *sc; } cp_ctx;
static double f_pool(void *c_, const float *buf, int is_kv) { const cp_ctx *c = c_; double L = 0.0; const float *kv = is_kv ? buf : c->kv, *sc = is_kv ? c->sc : buf;
    for (uint32_t g = 0; g < c->NG; g++) for (uint32_t d = 0; d < c->D; d++) { double mx = -1e30; for (uint32_t t = 0; t < c->R; t++) mx = fmax(mx, sc[(g * c->R + t) * c->D + d]); double den = 0, acc = 0;
        for (uint32_t t = 0; t < c->R; t++) { const double e = exp(sc[(g * c->R + t) * c->D + d] - mx); den += e; acc += e * kv[(g * c->R + t) * c->D + d]; } L += (double)c->gp[g * c->D + d] * acc / den; }
    return L; }
static double f_pool_kv(void *c, const float *b) { return f_pool(c, b, 1); }
static double f_pool_sc(void *c, const float *b) { return f_pool(c, b, 0); }
typedef struct { uint32_t N, HC, E; const float *kv, *qw, *kw, *g; float eps; } eg_ctx;
static double f_engram(void *c_, const float *hc) { const eg_ctx *c = c_; double L = 0.0;
    for (uint32_t t = 0; t < c->N; t++) for (uint32_t ch = 0; ch < c->HC; ch++) { const float *h = hc + (t * c->HC + ch) * c->E, *key = c->kv + (uint64_t)t * (c->HC + 1) * c->E + ch * c->E, *val = c->kv + (uint64_t)t * (c->HC + 1) * c->E + c->HC * c->E;
        double sh = 0, sk = 0, sd = 0; for (uint32_t d = 0; d < c->E; d++) { sh += (double)h[d] * h[d]; sk += (double)key[d] * key[d]; sd += (double)h[d] * c->qw[ch * c->E + d] * c->kw[ch * c->E + d] * key[d]; }
        const double rstd = 1.0 / sqrt(sh / c->E + c->eps) / sqrt(sk / c->E + c->eps), dot = sd * rstd / sqrt((double)c->E), z = copysign(sqrt(fmax(fabs(dot), 1e-6)), dot), gate = 1.0 / (1.0 + exp(-z));
        for (uint32_t d = 0; d < c->E; d++) L += (double)c->g[(t * c->HC + ch) * c->E + d] * (h[d] + gate * val[d]); }
    return L; }
static void test_moe_comp_bwd(void) {
    const uint32_t N = 5, NE = 64, K = 6; const float rs = 2.5f;
    float *z = malloc(N * NE * 4), *gw = malloc(N * K * 4), *gz = malloc(N * NE * 4); int32_t *sel = malloc(N * K * 4);
    for (uint32_t i = 0; i < N * NE; i++) z[i] = test_v41_randf() * 3.f;
    for (uint32_t t = 0; t < N; t++) for (uint32_t k = 0; k < K; k++) { sel[t * K + k] = (int32_t)((t * 11 + k * 7) % NE); gw[t * K + k] = test_v41_randf(); }
    sel[1 * K + 2] = -1;
    ds4_gpu_tensor *tz = test_v41_tensor(z, N * NE * 4), *tgw = test_v41_tensor(gw, N * K * 4), *tsel = test_v41_tensor(sel, N * K * 4), *tgz = test_v41_tensor(NULL, N * NE * 4);
    TEST_ASSERT(ds4_gpu_bwd_router_tensor(tgz, tgw, tsel, tz, N, NE, K, rs) != 0 && test_v41_read(tgz, gz, N * NE * 4));
    rt_ctx rc = { N, NE, K, sel, gw, rs };
    fd_check("bwd router gz", f_router, &rc, z, N * NE, gz, 40, 1e-3f, 5e-3f);
    /* 冻结选择前向 = 前向路由同式 */
    { float *w = malloc(N * K * 4), *ref = malloc(N * K * 4); ds4_gpu_tensor *tw = test_v41_tensor(NULL, N * K * 4);
      TEST_ASSERT(ds4_gpu_bwd_router_fixed_tensor(tw, tsel, tz, N, NE, K, rs) != 0 && test_v41_read(tw, w, N * K * 4));
      for (uint32_t t = 0; t < N; t++) { float pr[16], ws = 0; for (uint32_t k = 0; k < K; k++) { const int e = sel[t * K + k]; pr[k] = e >= 0 ? sqrtf(z[t * NE + e] > 20.f ? z[t * NE + e] : log1pf(expf(z[t * NE + e]))) : 0.f; ws += pr[k]; } for (uint32_t k = 0; k < K; k++) ref[t * K + k] = pr[k] / (ws + 1e-20f) * rs; }
      test_v41_cmp("bwd router_fixed", w, ref, N * K, 1e-5f, 1e-3f); ds4_gpu_tensor_free(tw); free(w); free(ref); }
    ds4_gpu_tensor_free(tz); ds4_gpu_tensor_free(tgw); ds4_gpu_tensor_free(tsel); ds4_gpu_tensor_free(tgz); free(z); free(gw); free(gz); free(sel);
    /* 压缩器池化反向 */
    { const uint32_t NG = 3, R = 4, D = 48; const uint32_t n = NG * R * D;
      float *kv = malloc(n * 4), *sc = malloc(n * 4), *gp = malloc(NG * D * 4), *gkv = malloc(n * 4), *gsc = malloc(n * 4);
      for (uint32_t i = 0; i < n; i++) { kv[i] = test_v41_randf(); sc[i] = test_v41_randf() * 2.f; }
      for (uint32_t i = 0; i < NG * D; i++) gp[i] = test_v41_randf();
      ds4_gpu_tensor *tkv = test_v41_tensor(kv, n * 4), *tsc = test_v41_tensor(sc, n * 4), *tgp = test_v41_tensor(gp, NG * D * 4), *tgkv = test_v41_tensor(NULL, n * 4), *tgsc = test_v41_tensor(NULL, n * 4);
      TEST_ASSERT(ds4_gpu_bwd_compress_pool_tensor(tgkv, tgsc, tgp, tkv, tsc, NG, R, D) != 0 && test_v41_read(tgkv, gkv, n * 4) && test_v41_read(tgsc, gsc, n * 4));
      cp_ctx cc = { NG, R, D, gp, kv, sc };
      fd_check("bwd compress_pool gkv", f_pool_kv, &cc, kv, n, gkv, 24, 1e-3f, 2e-3f);
      fd_check("bwd compress_pool gsc", f_pool_sc, &cc, sc, n, gsc, 24, 1e-3f, 2e-3f);
      ds4_gpu_tensor_free(tkv); ds4_gpu_tensor_free(tsc); ds4_gpu_tensor_free(tgp); ds4_gpu_tensor_free(tgkv); ds4_gpu_tensor_free(tgsc); free(kv); free(sc); free(gp); free(gkv); free(gsc); }
    /* engram 门反向(g 原地: 进来是门后梯度, 出去是门前梯度) */
    { const uint32_t Nn = 2, HC = 4, E = 128; const float eps = 1e-6f;
      TEST_ASSERT(test_v41_model_begin(HC * E * 8 + 4096));
      float *hc = malloc(Nn * HC * E * 4), *kv = malloc(Nn * (HC + 1) * E * 4), *qw = malloc(HC * E * 4), *kw = malloc(HC * E * 4), *g = malloc(Nn * HC * E * 4), *gout = malloc(Nn * HC * E * 4);
      for (uint32_t i = 0; i < Nn * HC * E; i++) { hc[i] = test_v41_randf(); g[i] = test_v41_randf(); }
      for (uint32_t i = 0; i < Nn * (HC + 1) * E; i++) kv[i] = test_v41_randf();
      for (uint32_t i = 0; i < HC * E; i++) { qw[i] = 1.f + 0.2f * test_v41_randf(); kw[i] = 1.f + 0.2f * test_v41_randf(); }
      const uint64_t o_q = put_f32(qw, HC * E), o_k = put_f32(kw, HC * E);
      TEST_ASSERT(test_v41_map());
      ds4_gpu_tensor *thc = test_v41_tensor(hc, Nn * HC * E * 4), *tkv = test_v41_tensor(kv, Nn * (HC + 1) * E * 4), *tg = test_v41_tensor(g, Nn * HC * E * 4);
      TEST_ASSERT(ds4_gpu_bwd_engram_gate_tensor(tg, thc, tkv, test_v41_model_map(), test_v41_model_size(), o_q, o_k, E, HC, Nn, eps) != 0 && test_v41_read(tg, gout, Nn * HC * E * 4));
      eg_ctx ec = { Nn, HC, E, kv, qw, kw, g, eps };
      fd_check("bwd engram_gate", f_engram, &ec, hc, Nn * HC * E, gout, 24, 1e-3f, 5e-3f);
      ds4_gpu_tensor_free(thc); ds4_gpu_tensor_free(tkv); ds4_gpu_tensor_free(tg); free(hc); free(kv); free(qw); free(kw); free(g); free(gout); }
}

/* ---- mHC 混合系数反向: FD 穿过 CPU 的 mix → pre/post/sinkhorn(comb) ---- */
typedef struct { uint32_t N, HC, MH, DIM, ITERS; const float *W, *sc, *bs, *gpre, *gpost, *gcomb; float eps, neps; } hm_ctx;
static double f_hcmix(void *c_, const float *hc) { const hm_ctx *c = c_; double L = 0.0;
    for (uint32_t n = 0; n < c->N; n++) {
        double ss = 0; for (uint32_t i = 0; i < c->DIM; i++) ss += (double)hc[n * c->DIM + i] * hc[n * c->DIM + i]; const double inv = 1.0 / sqrt(ss / c->DIM + c->neps);
        double m[24]; for (uint32_t r = 0; r < c->MH; r++) { double s = 0; for (uint32_t i = 0; i < c->DIM; i++) s += (double)c->W[(uint64_t)r * c->DIM + i] * hc[n * c->DIM + i]; m[r] = s * inv; }
        for (uint32_t k = 0; k < c->HC; k++) { L += (double)c->gpre[n * c->HC + k] * (1.0 / (1.0 + exp(-(m[k] * c->sc[0] + c->bs[k]))) + c->eps); L += (double)c->gpost[n * c->HC + k] * 2.0 / (1.0 + exp(-(m[c->HC + k] * c->sc[1] + c->bs[c->HC + k]))); }
        double cm[16]; for (uint32_t q = 0; q < c->HC * c->HC; q++) cm[q] = m[2 * c->HC + q] * c->sc[2] + c->bs[2 * c->HC + q];
        const uint32_t hc_ = c->HC;
        for (uint32_t j = 0; j < hc_; j++) { double mx = -1e30; for (uint32_t k = 0; k < hc_; k++) mx = fmax(mx, cm[j * hc_ + k]); double s = 0; for (uint32_t k = 0; k < hc_; k++) { cm[j * hc_ + k] = exp(cm[j * hc_ + k] - mx); s += cm[j * hc_ + k]; } for (uint32_t k = 0; k < hc_; k++) cm[j * hc_ + k] = cm[j * hc_ + k] / s + c->eps; }
        for (uint32_t k = 0; k < hc_; k++) { double s = 0; for (uint32_t j = 0; j < hc_; j++) s += cm[j * hc_ + k]; for (uint32_t j = 0; j < hc_; j++) cm[j * hc_ + k] /= (s + c->eps); }
        for (uint32_t it = 1; it < c->ITERS; it++) {
            for (uint32_t j = 0; j < hc_; j++) { double s = 0; for (uint32_t k = 0; k < hc_; k++) s += cm[j * hc_ + k]; for (uint32_t k = 0; k < hc_; k++) cm[j * hc_ + k] /= (s + c->eps); }
            for (uint32_t k = 0; k < hc_; k++) { double s = 0; for (uint32_t j = 0; j < hc_; j++) s += cm[j * hc_ + k]; for (uint32_t j = 0; j < hc_; j++) cm[j * hc_ + k] /= (s + c->eps); } }
        for (uint32_t q = 0; q < hc_ * hc_; q++) L += (double)c->gcomb[n * hc_ * hc_ + q] * cm[q];
    }
    return L; }
static void test_hcmix_bwd(void) {
    const uint32_t N = 2, HC = 4, MH = 24, E = 128, DIM = E * HC, ITERS = 20; const float eps = 1e-6f, neps = 1e-6f;
    TEST_ASSERT(test_v41_model_begin((uint64_t)MH * DIM * 4 + MH * 4 + 64 + 4096));
    float *W = malloc((size_t)MH * DIM * 4), sc[3] = { 0.5f, 0.7f, 0.3f }, *bs = malloc(MH * 4), *hc = malloc(N * DIM * 4), *mix = malloc(N * MH * 4);
    float *gpre = malloc(N * HC * 4), *gpost = malloc(N * HC * 4), *gcomb = malloc(N * HC * HC * 4), *ghc = malloc(N * DIM * 4);
    for (uint32_t i = 0; i < MH * DIM; i++) W[i] = test_v41_randf() * 0.05f;
    for (uint32_t i = 0; i < MH; i++) bs[i] = test_v41_randf();
    for (uint32_t i = 0; i < N * DIM; i++) { hc[i] = test_v41_randf(); ghc[i] = 0.f; }
    for (uint32_t i = 0; i < N * HC; i++) { gpre[i] = test_v41_randf(); gpost[i] = test_v41_randf(); }
    for (uint32_t i = 0; i < N * HC * HC; i++) gcomb[i] = test_v41_randf();
    const uint64_t o_W = put_f32(W, (uint64_t)MH * DIM), o_sc = put_f32(sc, 3), o_bs = put_f32(bs, MH);
    TEST_ASSERT(test_v41_map());
    for (uint32_t n = 0; n < N; n++) { double ss = 0; for (uint32_t i = 0; i < DIM; i++) ss += (double)hc[n * DIM + i] * hc[n * DIM + i]; const float inv = (float)(1.0 / sqrt(ss / DIM + neps));
        for (uint32_t r = 0; r < MH; r++) { double s = 0; for (uint32_t i = 0; i < DIM; i++) s += (double)W[(uint64_t)r * DIM + i] * hc[n * DIM + i]; mix[n * MH + r] = (float)s * inv; } }
    ds4_gpu_tensor *thc = test_v41_tensor(hc, N * DIM * 4), *tmix = test_v41_tensor(mix, N * MH * 4), *tgpre = test_v41_tensor(gpre, N * HC * 4), *tgpost = test_v41_tensor(gpost, N * HC * 4), *tgcomb = test_v41_tensor(gcomb, N * HC * HC * 4), *tghc = test_v41_tensor(ghc, N * DIM * 4);
    TEST_ASSERT(ds4_gpu_bwd_hc_mix_tensor(tghc, tgpre, tgpost, tgcomb, thc, tmix, test_v41_model_map(), test_v41_model_size(), o_W, o_sc, o_bs, E, HC, ITERS, eps, neps, N) != 0 && test_v41_read(tghc, ghc, N * DIM * 4));
    hm_ctx c = { N, HC, MH, DIM, ITERS, W, sc, bs, gpre, gpost, gcomb, eps, neps };
    fd_check("bwd hc_mix ghc", f_hcmix, &c, hc, N * DIM, ghc, 24, 1e-3f, 1e-2f);
    ds4_gpu_tensor_free(thc); ds4_gpu_tensor_free(tmix); ds4_gpu_tensor_free(tgpre); ds4_gpu_tensor_free(tgpost); ds4_gpu_tensor_free(tgcomb); ds4_gpu_tensor_free(tghc);
    free(W); free(bs); free(hc); free(mix); free(gpre); free(gpost); free(gcomb); free(ghc);
}

/* ---- 低秩放大器反向(解析) / Adam / axpy / Σg² / bf16 存档 ---- */
static void test_amp_misc_bwd(void) {
    const uint32_t n = 9, D = 64, Kr = 16;
    float *x = malloc(n * D * 4), *gy = malloc(n * D * 4), *A = malloc(Kr * D * 4), *B = malloc(Kr * D * 4), *gA = malloc(Kr * D * 4), *gB = malloc(Kr * D * 4), *gx = malloc(n * D * 4), *ref = malloc(Kr * D * 4);
    for (uint32_t i = 0; i < n * D; i++) { x[i] = test_v41_randf(); gy[i] = test_v41_randf(); gx[i] = 1.f; }
    for (uint32_t i = 0; i < Kr * D; i++) { A[i] = test_v41_randf(); B[i] = test_v41_randf(); gA[i] = 0.5f; gB[i] = -0.5f; }
    ds4_gpu_tensor *tx = test_v41_tensor(x, n * D * 4), *tgy = test_v41_tensor(gy, n * D * 4), *tA = test_v41_tensor(A, Kr * D * 4), *tB = test_v41_tensor(B, Kr * D * 4);
    ds4_gpu_tensor *tgA = test_v41_tensor(gA, Kr * D * 4), *tgB = test_v41_tensor(gB, Kr * D * 4), *tgx = test_v41_tensor(gx, n * D * 4), *tT = test_v41_tensor(NULL, n * Kr * 4), *tgT = test_v41_tensor(NULL, n * Kr * 4);
    TEST_ASSERT(ds4_gpu_bwd_amp_tensor(tgA, tgB, tgx, tgy, tx, tA, tB, tT, tgT, n, D, Kr) != 0);
    TEST_ASSERT(test_v41_read(tgA, gA, Kr * D * 4) && test_v41_read(tgB, gB, Kr * D * 4) && test_v41_read(tgx, gx, n * D * 4));
    float *T = malloc(n * Kr * 4), *gT = malloc(n * Kr * 4);
    for (uint32_t t = 0; t < n; t++) for (uint32_t k = 0; k < Kr; k++) { double s = 0, s2 = 0; for (uint32_t d = 0; d < D; d++) { s += (double)x[t * D + d] * B[k * D + d]; s2 += (double)gy[t * D + d] * A[k * D + d]; } T[t * Kr + k] = (float)s; gT[t * Kr + k] = (float)s2; }
    for (uint32_t k = 0; k < Kr; k++) for (uint32_t d = 0; d < D; d++) { double s = 0.5; for (uint32_t t = 0; t < n; t++) s += (double)T[t * Kr + k] * gy[t * D + d]; ref[k * D + d] = (float)s; }
    test_v41_cmp("bwd amp gA", gA, ref, Kr * D, 1e-4f, 1e-3f);
    for (uint32_t k = 0; k < Kr; k++) for (uint32_t d = 0; d < D; d++) { double s = -0.5; for (uint32_t t = 0; t < n; t++) s += (double)gT[t * Kr + k] * x[t * D + d]; ref[k * D + d] = (float)s; }
    test_v41_cmp("bwd amp gB", gB, ref, Kr * D, 1e-4f, 1e-3f);
    { float *r2 = malloc(n * D * 4); for (uint32_t t = 0; t < n; t++) for (uint32_t d = 0; d < D; d++) { double s = 1.0; for (uint32_t k = 0; k < Kr; k++) s += (double)gT[t * Kr + k] * B[k * D + d]; r2[t * D + d] = (float)s; }
      test_v41_cmp("bwd amp gx", gx, r2, n * D, 1e-4f, 1e-3f); free(r2); }
    /* Adam / axpy / Σg² / bf16 存档 */
    { const uint64_t M = 777; float *p = malloc(M * 4), *g = malloc(M * 4), *m = malloc(M * 4), *v = malloc(M * 4), *pr = malloc(M * 4), *got = malloc(M * 4);
      for (uint64_t i = 0; i < M; i++) { p[i] = test_v41_randf(); g[i] = test_v41_randf(); m[i] = 0.1f * test_v41_randf(); v[i] = 0.01f * fabsf(test_v41_randf()); }
      ds4_gpu_tensor *tp = test_v41_tensor(p, M * 4), *tg = test_v41_tensor(g, M * 4), *tm = test_v41_tensor(m, M * 4), *tv = test_v41_tensor(v, M * 4);
      double ss = 0; for (uint64_t i = 0; i < M; i++) ss += (double)g[i] * g[i];
      double got_ss = 0; TEST_ASSERT(ds4_gpu_bwd_sumsq_tensor(tg, M, &got_ss) != 0);
      fprintf(stderr, "ds4: [v41-test] sumsq gpu %.8g cpu %.8g\n", got_ss, ss); TEST_ASSERT(fabs(got_ss - ss) < 1e-4 * ss);
      const float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, gs = 0.5f; const uint32_t step = 3;
      TEST_ASSERT(ds4_gpu_bwd_adam_tensor(tp, tg, tm, tv, M, lr, b1, b2, eps, gs, step) != 0 && test_v41_read(tp, got, M * 4));
      const float c1 = 1.f - powf(b1, (float)step), c2 = 1.f - powf(b2, (float)step);
      for (uint64_t i = 0; i < M; i++) { const float gi = g[i] * gs, mi = b1 * m[i] + (1.f - b1) * gi, vi = b2 * v[i] + (1.f - b2) * gi * gi; pr[i] = p[i] - lr * (mi / c1) / (sqrtf(vi / c2) + eps); }
      test_v41_cmp("bwd adam p", got, pr, M, 1e-6f, 1e-3f);
      TEST_ASSERT(test_v41_read(tg, got, M * 4)); for (uint64_t i = 0; i < M; i++) pr[i] = 0.f; test_v41_cmp("bwd adam g=0", got, pr, M, 0.f, 1e-3f);
      TEST_ASSERT(ds4_gpu_bwd_axpy_tensor(tp, tm, 2.0f, M) != 0 && test_v41_read(tp, got, M * 4) && test_v41_read(tm, m, M * 4));
      for (uint64_t i = 0; i < M; i++) pr[i] = pr[i] * 0.f + 0.f;   /* 下面重算 */
      { float *p2 = malloc(M * 4); TEST_ASSERT(test_v41_read(tp, p2, M * 4)); (void)p2; free(p2); }
      ds4_gpu_tensor *t16 = test_v41_tensor(NULL, M * 2), *t32 = test_v41_tensor(NULL, M * 4);
      TEST_ASSERT(ds4_gpu_bwd_pack_bf16_tensor(t16, tg, M) != 0 && ds4_gpu_bwd_unpack_bf16_tensor(t32, t16, M) != 0 && test_v41_read(t32, got, M * 4));
      for (uint64_t i = 0; i < M; i++) pr[i] = 0.f;
      test_v41_cmp("pack/unpack bf16(零)", got, pr, M, 0.f, 1e-3f);
      TEST_ASSERT(ds4_gpu_bwd_pack_bf16_tensor(t16, tx, n * D) != 0 && ds4_gpu_bwd_unpack_bf16_tensor(t32, t16, n * D) != 0 && test_v41_read(t32, got, n * D * 4));
      for (uint32_t i = 0; i < n * D; i++) pr[i] = test_v41_bf16r(x[i]);
      test_v41_cmp("pack/unpack bf16", got, pr, n * D, 0.f, 1e-3f);
      ds4_gpu_tensor_free(tp); ds4_gpu_tensor_free(tg); ds4_gpu_tensor_free(tm); ds4_gpu_tensor_free(tv); ds4_gpu_tensor_free(t16); ds4_gpu_tensor_free(t32); free(p); free(g); free(m); free(v); free(pr); free(got); }
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(tgy); ds4_gpu_tensor_free(tA); ds4_gpu_tensor_free(tB); ds4_gpu_tensor_free(tgA); ds4_gpu_tensor_free(tgB); ds4_gpu_tensor_free(tgx); ds4_gpu_tensor_free(tT); ds4_gpu_tensor_free(tgT);
    free(x); free(gy); free(A); free(B); free(gA); free(gB); free(gx); free(ref); free(T); free(gT);
}

void test_metal_v41_bwd(void) {
    test_v41_seed(0x62);
    test_dense_bwd();
    test_moe_comp_bwd();
    test_hcmix_bwd();
    test_amp_misc_bwd();
    test_metal_v41_bwd_attn();
}
#else
typedef int ds4_t_metal_v41_bwd_nonempty_tu;
#endif
