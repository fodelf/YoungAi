/* t_metal_v41_layer.c — V4.1 Metal 层内小核回归(2026-10-08): RMSNorm / mHC 四件(含合一核与三发的一致) / engram 门 / 路由 / SwiGLU / 压缩器 /
 * RoPE / 激活量化 ↔ KV 打包 / 窗口环。金标 = 对着官方 model.py 逐式写的 CPU 转录(与 CUDA 核同一算式)。 */
#include "test_internal.h"
#if !defined(DS4_NO_GPU) && defined(__APPLE__)   /* Metal 专用: CUDA 构建里这些核的形状闸不同(专家核只认 11/12/13 位、注意力只认 64 头), 合成小形状不适用 */
#include "../src/common/ds4_fp8.h"
#include "../ds4_gpu_v41.h"

static float sigm(float z) { return 1.0f / (1.0f + expf(-z)); }
/* 合成模型里放一块 f32 权重, 返回偏移 */
static uint64_t put_f32(const float *v, uint64_t n) { const uint64_t off = test_v41_model_alloc(n * 4); memcpy(test_v41_model_ptr(off), v, (size_t)n * 4); return off; }

static void cpu_sinkhorn(float *c, uint32_t hc, uint32_t iters, float eps) {
    for (uint32_t j = 0; j < hc; j++) {
        float mx = -3.0e38f; for (uint32_t k = 0; k < hc; k++) mx = fmaxf(mx, c[j * hc + k]);
        float s = 0.f; for (uint32_t k = 0; k < hc; k++) { c[j * hc + k] = expf(c[j * hc + k] - mx); s += c[j * hc + k]; }
        for (uint32_t k = 0; k < hc; k++) c[j * hc + k] = c[j * hc + k] / s + eps;
    }
    for (uint32_t k = 0; k < hc; k++) { float s = 0.f; for (uint32_t j = 0; j < hc; j++) s += c[j * hc + k]; for (uint32_t j = 0; j < hc; j++) c[j * hc + k] /= (s + eps); }
    for (uint32_t it = 1; it < iters; it++) {
        for (uint32_t j = 0; j < hc; j++) { float s = 0.f; for (uint32_t k = 0; k < hc; k++) s += c[j * hc + k]; for (uint32_t k = 0; k < hc; k++) c[j * hc + k] /= (s + eps); }
        for (uint32_t k = 0; k < hc; k++) { float s = 0.f; for (uint32_t j = 0; j < hc; j++) s += c[j * hc + k]; for (uint32_t j = 0; j < hc; j++) c[j * hc + k] /= (s + eps); }
    }
}

static void test_norm_hc(void) {
    const uint32_t E = 512, HC = 4, N = 5, MH = 2 * HC + HC * HC, DIM = E * HC, ITERS = 20;
    const float eps = 1e-6f, hc_eps = 1e-6f;
    TEST_ASSERT(test_v41_model_begin((uint64_t)MH * DIM * 4 + E * 4 * 2 + MH * 4 + (uint64_t)HC * E * 8 + 8192));
    float *nw = malloc(E * 4), *W = malloc((size_t)MH * DIM * 4), sc[3] = { 0.5f, 0.7f, 0.3f }, *bs = malloc(MH * 4);
    for (uint32_t i = 0; i < E; i++) nw[i] = 0.5f + test_v41_randf() * 0.1f;
    for (uint32_t i = 0; i < MH * DIM; i++) W[i] = test_v41_randf() * 0.02f;
    for (uint32_t i = 0; i < MH; i++) bs[i] = test_v41_randf();
    const uint64_t o_nw = put_f32(nw, E), o_W = put_f32(W, (uint64_t)MH * DIM), o_sc = put_f32(sc, 3), o_bs = put_f32(bs, MH);
    TEST_ASSERT(test_v41_map());
    const void *mm = test_v41_model_map(); const uint64_t ms = test_v41_model_size();
    float *hc = malloc((size_t)N * DIM * 4), *pin = malloc(N * HC * 4), *got = malloc((size_t)N * DIM * 4), *ref = malloc((size_t)N * DIM * 4);
    test_v41_fill_bf16(hc, (uint64_t)N * DIM, 1.0f);
    for (uint32_t i = 0; i < N * HC; i++) pin[i] = 0.2f + 0.3f * (float)(i % HC);
    ds4_gpu_tensor *thc = test_v41_tensor(hc, (uint64_t)N * DIM * 4), *tpin = test_v41_tensor(pin, N * HC * 4);
    ds4_gpu_tensor *tmix = test_v41_tensor(NULL, N * MH * 4), *tpre = test_v41_tensor(NULL, N * HC * 4), *tpost = test_v41_tensor(NULL, N * HC * 4), *tcomb = test_v41_tensor(NULL, N * HC * HC * 4);
    ds4_gpu_tensor *tx = test_v41_tensor(NULL, (uint64_t)N * E * 4), *txn = test_v41_tensor(NULL, (uint64_t)N * E * 4);
    /* hc_mix */
    TEST_ASSERT(ds4_gpu_v41_hc_mix_tensor(tmix, thc, mm, ms, o_W, E, HC, N, eps) != 0);
    float *mix = malloc(N * MH * 4);
    TEST_ASSERT(test_v41_read(tmix, mix, N * MH * 4));
    float *mref = malloc(N * MH * 4);
    for (uint32_t n = 0; n < N; n++) {
        double ss = 0.0; for (uint32_t i = 0; i < DIM; i++) ss += (double)hc[n * DIM + i] * hc[n * DIM + i];
        const float inv = 1.0f / sqrtf((float)(ss / DIM) + eps);
        for (uint32_t r = 0; r < MH; r++) { double s = 0.0; for (uint32_t i = 0; i < DIM; i++) s += (double)W[(uint64_t)r * DIM + i] * hc[n * DIM + i]; mref[n * MH + r] = (float)s * inv; }
    }
    test_v41_cmp("hc_mix", mix, mref, N * MH, 1e-4f, 1e-3f);
    /* hc_split + hc_pre + rms_norm 分发 vs CPU */
    TEST_ASSERT(ds4_gpu_v41_hc_split_tensor(tpre, tpost, tcomb, tmix, mm, ms, o_sc, o_bs, HC, ITERS, hc_eps, N) != 0);
    TEST_ASSERT(ds4_gpu_v41_hc_pre_tensor(tx, thc, tpin, E, HC, N) != 0);
    TEST_ASSERT(ds4_gpu_v41_rms_norm_tensor(txn, tx, mm, ms, o_nw, E, N, eps) != 0);
    float *pre = malloc(N * HC * 4), *post = malloc(N * HC * 4), *comb = malloc(N * HC * HC * 4), *x = malloc((uint64_t)N * E * 4), *xn = malloc((uint64_t)N * E * 4);
    TEST_ASSERT(test_v41_read(tpre, pre, N * HC * 4) && test_v41_read(tpost, post, N * HC * 4) && test_v41_read(tcomb, comb, N * HC * HC * 4));
    TEST_ASSERT(test_v41_read(tx, x, (uint64_t)N * E * 4) && test_v41_read(txn, xn, (uint64_t)N * E * 4));
    float *pre_r = malloc(N * HC * 4), *post_r = malloc(N * HC * 4), *comb_r = malloc(N * HC * HC * 4), *x_r = malloc((uint64_t)N * E * 4), *xn_r = malloc((uint64_t)N * E * 4);
    for (uint32_t n = 0; n < N; n++) {
        const float *m = mix + n * MH;
        for (uint32_t c = 0; c < HC; c++) { pre_r[n * HC + c] = sigm(m[c] * sc[0] + bs[c]) + hc_eps; post_r[n * HC + c] = 2.f * sigm(m[HC + c] * sc[1] + bs[HC + c]); }
        for (uint32_t q = 0; q < HC * HC; q++) comb_r[n * HC * HC + q] = m[2 * HC + q] * sc[2] + bs[2 * HC + q];
        cpu_sinkhorn(comb_r + n * HC * HC, HC, ITERS, hc_eps);
        double ss = 0.0;
        for (uint32_t d = 0; d < E; d++) { float v = 0.f; for (uint32_t k = 0; k < HC; k++) v += pin[n * HC + k] * hc[(n * HC + k) * E + d]; v = test_v41_bf16r(v); x_r[n * E + d] = v; ss += (double)v * v; }
        const float inv = 1.0f / sqrtf((float)(ss / E) + eps);
        for (uint32_t d = 0; d < E; d++) xn_r[n * E + d] = test_v41_bf16r(nw[d] * (x_r[n * E + d] * inv));
    }
    test_v41_cmp("hc_split pre", pre, pre_r, N * HC, 1e-5f, 1e-3f); test_v41_cmp("hc_split post", post, post_r, N * HC, 1e-5f, 1e-3f);
    test_v41_cmp("hc_split comb", comb, comb_r, N * HC * HC, 1e-4f, 1e-3f);
    /* 出口舍 bf16: 累加序/FMA 合并差一个 f32 ulp 就可能跨一个 bf16 格(2^-8 相对), 门给 1 个 bf16 ulp 的量 */
    test_v41_cmp("hc_pre", x, x_r, (uint64_t)N * E, 1e-2f, 1e-3f); test_v41_cmp("rms_norm", xn, xn_r, (uint64_t)N * E, 1e-2f, 1e-3f);
    /* 合一核与三发同口径 */
    ds4_gpu_tensor *tpre2 = test_v41_tensor(NULL, N * HC * 4), *tpost2 = test_v41_tensor(NULL, N * HC * 4), *tcomb2 = test_v41_tensor(NULL, N * HC * HC * 4);
    ds4_gpu_tensor *tx2 = test_v41_tensor(NULL, (uint64_t)N * E * 4), *txn2 = test_v41_tensor(NULL, (uint64_t)N * E * 4);
    TEST_ASSERT(ds4_gpu_v41_hc_fused_tensor(tpre2, tpost2, tcomb2, tx2, txn2, tmix, thc, tpin, mm, ms, o_sc, o_bs, o_nw, E, HC, ITERS, hc_eps, eps, N) != 0);
    TEST_ASSERT(test_v41_read(tcomb2, got, N * HC * HC * 4)); test_v41_cmp("hc_fused comb", got, comb_r, N * HC * HC, 1e-4f, 1e-3f);
    TEST_ASSERT(test_v41_read(tpre2, got, N * HC * 4)); test_v41_cmp("hc_fused pre", got, pre_r, N * HC, 1e-5f, 1e-3f);
    TEST_ASSERT(test_v41_read(txn2, got, (uint64_t)N * E * 4)); test_v41_cmp("hc_fused xn", got, xn_r, (uint64_t)N * E, 1e-2f, 1e-3f);
    /* hc_post */
    float *y = malloc((uint64_t)N * E * 4);
    test_v41_fill_bf16(y, (uint64_t)N * E, 1.0f);
    ds4_gpu_tensor *ty = test_v41_tensor(y, (uint64_t)N * E * 4), *tout = test_v41_tensor(NULL, (uint64_t)N * DIM * 4);
    TEST_ASSERT(ds4_gpu_v41_hc_post_tensor(tout, ty, thc, tpost, tcomb, E, HC, N) != 0);
    TEST_ASSERT(test_v41_read(tout, got, (uint64_t)N * DIM * 4));
    for (uint32_t n = 0; n < N; n++) for (uint32_t k = 0; k < HC; k++) for (uint32_t d = 0; d < E; d++) {
        float s = post[n * HC + k] * y[n * E + d];
        for (uint32_t j = 0; j < HC; j++) s += comb[n * HC * HC + j * HC + k] * hc[(n * HC + j) * E + d];
        ref[(n * HC + k) * E + d] = test_v41_bf16r(s);
    }
    test_v41_cmp("hc_post", got, ref, (uint64_t)N * DIM, 1e-6f, 1e-3f);
    /* engram 门: kv[n][(HC+1)E], qw/kw [HC][E] */
    float *kv = malloc((uint64_t)N * (HC + 1) * E * 4), *qw = malloc(HC * E * 4), *kw = malloc(HC * E * 4);
    test_v41_fill_bf16(kv, (uint64_t)N * (HC + 1) * E, 1.0f);
    for (uint32_t i = 0; i < HC * E; i++) { qw[i] = 1.0f + 0.1f * test_v41_randf(); kw[i] = 1.0f + 0.1f * test_v41_randf(); }
    const uint64_t o_qw = put_f32(qw, HC * E), o_kw = put_f32(kw, HC * E);
    TEST_ASSERT(test_v41_map());
    ds4_gpu_tensor *tkv = test_v41_tensor(kv, (uint64_t)N * (HC + 1) * E * 4), *thc2 = test_v41_tensor(hc, (uint64_t)N * DIM * 4);
    TEST_ASSERT(ds4_gpu_v41_engram_gate_tensor(thc2, tkv, mm, ms, o_qw, o_kw, E, HC, N, eps) != 0);
    TEST_ASSERT(test_v41_read(thc2, got, (uint64_t)N * DIM * 4));
    for (uint32_t t = 0; t < N; t++) for (uint32_t c = 0; c < HC; c++) {
        const float *h = hc + (t * HC + c) * E, *key = kv + (uint64_t)t * (HC + 1) * E + c * E, *val = kv + (uint64_t)t * (HC + 1) * E + HC * E;
        double sh = 0, sk = 0, sd = 0;
        for (uint32_t d = 0; d < E; d++) { sh += (double)h[d] * h[d]; sk += (double)key[d] * key[d]; sd += (double)h[d] * qw[c * E + d] * kw[c * E + d] * key[d]; }
        const float rstd = 1.0f / sqrtf((float)(sh / E) + eps) / sqrtf((float)(sk / E) + eps), dot = (float)sd * rstd / sqrtf((float)E);
        const float z = copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot), gate = sigm(z);
        for (uint32_t d = 0; d < E; d++) ref[(t * HC + c) * E + d] = test_v41_bf16r(h[d] + gate * val[d]);
    }
    test_v41_cmp("engram_gate", got, ref, (uint64_t)N * DIM, 1e-5f, 1e-3f);
    ds4_gpu_tensor_free(thc); ds4_gpu_tensor_free(tpin); ds4_gpu_tensor_free(tmix); ds4_gpu_tensor_free(tpre); ds4_gpu_tensor_free(tpost); ds4_gpu_tensor_free(tcomb);
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(txn); ds4_gpu_tensor_free(tpre2); ds4_gpu_tensor_free(tpost2); ds4_gpu_tensor_free(tcomb2); ds4_gpu_tensor_free(tx2); ds4_gpu_tensor_free(txn2);
    ds4_gpu_tensor_free(ty); ds4_gpu_tensor_free(tout); ds4_gpu_tensor_free(tkv); ds4_gpu_tensor_free(thc2);
    free(nw); free(W); free(bs); free(hc); free(pin); free(got); free(ref); free(mix); free(mref); free(pre); free(post); free(comb); free(x); free(xn);
    free(pre_r); free(post_r); free(comb_r); free(x_r); free(xn_r); free(y); free(kv); free(qw); free(kw);
}

static void test_router_swiglu_pool_rope(void) {
    const uint32_t NE = 384, K = 6, N = 7;
    const float rs = 2.5f;
    TEST_ASSERT(test_v41_model_begin(NE * 4 + 4096));
    float *bias = malloc(NE * 4), *lg = malloc((size_t)N * NE * 4);
    for (uint32_t e = 0; e < NE; e++) bias[e] = test_v41_randf() * 0.5f;
    for (uint32_t i = 0; i < N * NE; i++) lg[i] = test_v41_randf() * 3.0f;
    const uint64_t o_b = put_f32(bias, NE);
    TEST_ASSERT(test_v41_map());
    ds4_gpu_tensor *tl = test_v41_tensor(lg, (size_t)N * NE * 4), *tsel = test_v41_tensor(NULL, N * K * 4), *tw = test_v41_tensor(NULL, N * K * 4);
    TEST_ASSERT(ds4_gpu_v41_router_tensor(tsel, tw, tl, test_v41_model_map(), test_v41_model_size(), o_b, N, NE, K, rs) != 0);
    int32_t *sel = malloc(N * K * 4); float *w = malloc(N * K * 4);
    TEST_ASSERT(test_v41_read(tsel, sel, N * K * 4) && test_v41_read(tw, w, N * K * 4));
    int bad = 0;
    for (uint32_t t = 0; t < N; t++) {
        float pr[384], sc[384]; int used[384] = { 0 }; float wsum = 0.f; int ch[6]; float chp[6];
        for (uint32_t e = 0; e < NE; e++) { const float z = lg[t * NE + e]; pr[e] = sqrtf(z > 20.f ? z : log1pf(expf(z))); sc[e] = pr[e] + bias[e]; }
        for (uint32_t r = 0; r < K; r++) { int be = -1; float bv = -3.0e38f; for (uint32_t e = 0; e < NE; e++) if (!used[e] && sc[e] > bv) { bv = sc[e]; be = (int)e; } used[be] = 1; ch[r] = be; chp[r] = pr[be]; wsum += pr[be]; }
        for (uint32_t r = 0; r < K; r++) { if (sel[t * K + r] != ch[r]) bad++; if (fabsf(w[t * K + r] - chp[r] / (wsum + 1e-20f) * rs) > 1e-5f) bad++; }
    }
    fprintf(stderr, "ds4: [v41-test] router: %u token × top-%u, 不一致 %d\n", N, K, bad);
    TEST_ASSERT(bad == 0);
    /* 路由偏置侧车: Δb 推高某专家 ⇒ 它必入选 */
    float *delta = calloc(NE, 4); delta[123] = 100.0f;
    TEST_ASSERT(ds4_gpu_v41_set_rb_override(test_v41_model_map(), test_v41_model_size(), o_b, delta, NE) != 0);
    TEST_ASSERT(ds4_gpu_v41_router_tensor(tsel, tw, tl, test_v41_model_map(), test_v41_model_size(), o_b, N, NE, K, rs) != 0);
    TEST_ASSERT(test_v41_read(tsel, sel, N * K * 4));
    for (uint32_t t = 0; t < N; t++) TEST_ASSERT(sel[t * K] == 123);
    TEST_ASSERT(ds4_gpu_v41_set_rb_override(NULL, 0, 0, NULL, 0) != 0);
    /* swiglu / compress_pool / rope */
    const uint32_t M = 300; const float L = 1.0f;
    float *g = malloc(M * 4), *u = malloc(M * 4), *h = malloc(M * 4), *hr = malloc(M * 4);
    for (uint32_t i = 0; i < M; i++) { g[i] = test_v41_randf() * 3.f; u[i] = test_v41_randf() * 3.f; }
    ds4_gpu_tensor *tg = test_v41_tensor(g, M * 4), *tu = test_v41_tensor(u, M * 4), *th = test_v41_tensor(NULL, M * 4);
    TEST_ASSERT(ds4_gpu_v41_swiglu_tensor(th, tg, tu, 1, M, L) != 0 && test_v41_read(th, h, M * 4));
    for (uint32_t i = 0; i < M; i++) { float gv = fminf(g[i], L), uv = fminf(fmaxf(u[i], -L), L); hr[i] = test_v41_bf16r(gv / (1.f + expf(-gv)) * uv); }
    test_v41_cmp("swiglu", h, hr, M, 1e-6f, 1e-3f);
    const uint32_t ratio = 4, dim = 64, ntok = 11, ng = ntok / ratio;
    float *kvp = malloc((size_t)ntok * dim * 4), *scp = malloc((size_t)ntok * dim * 4), *pool = malloc((size_t)ng * dim * 4), *poolr = malloc((size_t)ng * dim * 4);
    test_v41_fill_bf16(kvp, (uint64_t)ntok * dim, 1.f); test_v41_fill_bf16(scp, (uint64_t)ntok * dim, 2.f);
    ds4_gpu_tensor *tkv = test_v41_tensor(kvp, (size_t)ntok * dim * 4), *tsc = test_v41_tensor(scp, (size_t)ntok * dim * 4), *tp = test_v41_tensor(NULL, (size_t)ng * dim * 4);
    TEST_ASSERT(ds4_gpu_v41_compress_pool_tensor(tp, tkv, tsc, ntok, ratio, dim) != 0 && test_v41_read(tp, pool, (size_t)ng * dim * 4));
    for (uint32_t gg = 0; gg < ng; gg++) for (uint32_t d = 0; d < dim; d++) {
        float mx = -3.0e38f; for (uint32_t t = 0; t < ratio; t++) mx = fmaxf(mx, scp[(gg * ratio + t) * dim + d]);
        float den = 0, acc = 0; for (uint32_t t = 0; t < ratio; t++) { const float e = expf(scp[(gg * ratio + t) * dim + d] - mx); den += e; acc += e * kvp[(gg * ratio + t) * dim + d]; }
        poolr[gg * dim + d] = test_v41_bf16r(acc / den);
    }
    test_v41_cmp("compress_pool", pool, poolr, (uint64_t)ng * dim, 1e-5f, 1e-3f);
    const uint32_t NH = 3, HD = 128, NR = 64; const int32_t pos[2] = { 17, 4000 };
    float *xr = malloc(2u * NH * HD * 4), *xg = malloc(2u * NH * HD * 4), *xref = malloc(2u * NH * HD * 4);
    test_v41_fill_bf16(xr, 2u * NH * HD, 1.f);
    ds4_gpu_tensor *txr = test_v41_tensor(xr, 2u * NH * HD * 4), *tpos = test_v41_tensor(pos, 8);
    TEST_ASSERT(ds4_gpu_v41_rope_tensor(txr, tpos, 2, NH, HD, NR, 10000.f, 4096, 40.f, 32.f, 1.f, false) != 0 && test_v41_read(txr, xg, 2u * NH * HD * 4));
    memcpy(xref, xr, 2u * NH * HD * 4);
    for (uint32_t t = 0; t < 2; t++) for (uint32_t hh = 0; hh < NH; hh++) for (uint32_t i = 0; i < NR / 2; i++) {
        float f = 1.0f / powf(10000.f, (float)(2 * i) / (float)NR);
        const float lt = 2.f * logf(10000.f);
        float lo = floorf((float)NR * logf(4096.f / (32.f * 2.f * (float)M_PI)) / lt), hi = ceilf((float)NR * logf(4096.f / (1.f * 2.f * (float)M_PI)) / lt);
        lo = fmaxf(lo, 0.f); hi = fminf(hi, (float)(NR - 1));
        float ramp = fminf(fmaxf(((float)i - lo) / fmaxf(hi - lo, 1e-3f), 0.f), 1.f), smooth = 1.f - ramp;
        f = f / 40.f * (1.f - smooth) + f * smooth;
        const float ang = (float)pos[t] * f, c = cosf(ang), s = sinf(ang);
        float *p = xref + (t * NH + hh) * HD + (HD - NR) + 2 * i;
        const float a = p[0], b = p[1];
        p[0] = test_v41_bf16r(a * c - b * s); p[1] = test_v41_bf16r(a * s + b * c);
    }
    test_v41_cmp("rope(yarn)", xg, xref, 2u * NH * HD, 2e-3f, 1e-3f);
    ds4_gpu_tensor_free(tl); ds4_gpu_tensor_free(tsel); ds4_gpu_tensor_free(tw); ds4_gpu_tensor_free(tg); ds4_gpu_tensor_free(tu); ds4_gpu_tensor_free(th);
    ds4_gpu_tensor_free(tkv); ds4_gpu_tensor_free(tsc); ds4_gpu_tensor_free(tp); ds4_gpu_tensor_free(txr); ds4_gpu_tensor_free(tpos);
    free(bias); free(lg); free(sel); free(w); free(delta); free(g); free(u); free(h); free(hr); free(kvp); free(scp); free(pool); free(poolr); free(xr); free(xg); free(xref);
}

/* 激活量化 vs CPU(ds4_fp8.h); KV 打包 = 量化后按宽度存: 解包必须与 act_quant 的结果逐位同 */
static float cpu_pow2_ceil_log2(float v) { int e; const float m = frexpf(v, &e); return ldexpf(1.0f, (m == 0.5f) ? e - 1 : e); }
static void test_quant_pack_ring(void) {
    const uint32_t rows = 3;
    for (int mode = 0; mode < 2; mode++) {   /* 0: 主 KV(块 16, e4m3 缩放) 1: 索引 K(块 32, e8m0 缩放) */
        const uint32_t dim = mode ? 128 : 512, blk = mode ? 32 : 16, rb = mode ? DS4_V41_IDXK_BYTES : DS4_V41_CKV_BYTES, nib = mode ? DS4_V41_IDXK_NIB : DS4_V41_CKV_NIB;
        float *x = malloc((size_t)rows * dim * 4), *q = malloc((size_t)rows * dim * 4), *qr = malloc((size_t)rows * dim * 4);
        test_v41_fill_bf16(x, (uint64_t)rows * dim, 2.f);
        ds4_gpu_tensor *tq = test_v41_tensor(x, (size_t)rows * dim * 4), *tcache = test_v41_tensor(NULL, (uint64_t)(rows + 2) * rb), *trows = test_v41_tensor(x, (size_t)rows * dim * 4);
        TEST_ASSERT(ds4_gpu_v41_act_quant_fp4_tensor(tq, rows, dim, blk, mode == 0) != 0 && test_v41_read(tq, q, (size_t)rows * dim * 4));
        for (uint32_t r = 0; r < rows; r++) for (uint32_t b = 0; b < dim / blk; b++) {
            float am = 0.f; for (uint32_t i = 0; i < blk; i++) am = fmaxf(am, fabsf(x[r * dim + b * blk + i]));
            float s;
            if (mode == 0) { am = fmaxf(am, 6.0f * ldexpf(1.0f, -9)); s = ds4_e4m3fn_round(am / 6.0f); } else { am = fmaxf(am, 6.0f * ldexpf(1.0f, -126)); s = cpu_pow2_ceil_log2(am / 6.0f); }
            for (uint32_t i = 0; i < blk; i++) { float v = fminf(fmaxf(x[r * dim + b * blk + i] / s, -6.f), 6.f); qr[r * dim + b * blk + i] = test_v41_bf16r(ds4_e2m1fn_round(v) * s); }
        }
        test_v41_cmp(mode ? "act_quant fp4/e8m0" : "act_quant fp4/e4m3", q, qr, (uint64_t)rows * dim, 1e-7f, 1e-3f);
        if (mode == 0) TEST_ASSERT(ds4_gpu_v41_ckv_pack_tensor(tcache, 1, trows, rows, NULL, 0, 0, 0) != 0);
        else TEST_ASSERT(ds4_gpu_v41_idxk_pack_tensor(tcache, 1, trows, rows, NULL, 0, 0, 0) != 0);
        uint8_t *cache = malloc((size_t)(rows + 2) * rb);
        TEST_ASSERT(test_v41_read(tcache, cache, (uint64_t)(rows + 2) * rb));
        float *unp = malloc((size_t)rows * dim * 4);
        for (uint32_t r = 0; r < rows; r++) for (uint32_t d = 0; d < dim; d++) {
            const uint8_t *row = cache + (size_t)(r + 1) * rb, by = row[d >> 1], n4 = (d & 1) ? (by >> 4) : (by & 15);
            const float s = mode ? ds4_e8m0_to_f32(row[nib + d / 32]) : ds4_e4m3fn_to_f32(row[nib + d / 16]);
            unp[r * dim + d] = test_v41_bf16r(ds4_fp4_nibble_to_f32(n4) * s);
        }
        test_v41_cmp(mode ? "idxk_pack ↔ act_quant" : "ckv_pack ↔ act_quant", unp, q, (uint64_t)rows * dim, 0.0f, 1e-3f);
        ds4_gpu_tensor_free(tq); ds4_gpu_tensor_free(tcache); ds4_gpu_tensor_free(trows); free(x); free(q); free(qr); free(cache); free(unp);
    }
    /* fp8 激活量化 */
    { const uint32_t dim = 256; float *x = malloc(dim * 4), *q = malloc(dim * 4), *qr = malloc(dim * 4);
      test_v41_fill_bf16(x, dim, 100.f);
      ds4_gpu_tensor *tq = test_v41_tensor(x, dim * 4);
      TEST_ASSERT(ds4_gpu_v41_act_quant_fp8_tensor(tq, 1, dim, 32) != 0 && test_v41_read(tq, q, dim * 4));
      for (uint32_t b = 0; b < dim / 32; b++) { float am = 0.f; for (uint32_t i = 0; i < 32; i++) am = fmaxf(am, fabsf(x[b * 32 + i])); am = fmaxf(am, 1e-4f); const float s = cpu_pow2_ceil_log2(am / 448.f);
          for (uint32_t i = 0; i < 32; i++) qr[b * 32 + i] = test_v41_bf16r(ds4_e4m3fn_round(fminf(fmaxf(x[b * 32 + i] / s, -448.f), 448.f)) * s); }
      test_v41_cmp("act_quant fp8", q, qr, dim, 1e-7f, 1e-3f);
      ds4_gpu_tensor_free(tq); free(x); free(q); free(qr); }
    /* 窗口环: 缓冲 [window + n] 行, 提交后环格 (pos0+i)%window = 本批行; snap 存/还原 */
    { const uint32_t W = 8, n = 3, hd = 16, pos0 = 13;
      float *buf = malloc((size_t)(W + n) * hd * 4), *got = malloc((size_t)(W + n) * hd * 4), *snap = malloc((size_t)n * hd * 4);
      for (uint32_t i = 0; i < (W + n) * hd; i++) buf[i] = (float)i;
      ds4_gpu_tensor *tw = test_v41_tensor(buf, (size_t)(W + n) * hd * 4), *ts = test_v41_tensor(NULL, (size_t)n * hd * 4);
      TEST_ASSERT(ds4_gpu_v41_win_ring_snap_tensor(tw, ts, pos0, 0, n, W, hd, 0, NULL) != 0);
      TEST_ASSERT(ds4_gpu_v41_win_commit_tensor(tw, pos0, n, W, hd, NULL) != 0 && test_v41_read(tw, got, (size_t)(W + n) * hd * 4));
      int bad = 0;
      for (uint32_t i = 0; i < n; i++) for (uint32_t d = 0; d < hd; d++) if (got[((pos0 + i) % W) * hd + d] != buf[(W + i) * hd + d]) bad++;
      TEST_ASSERT(ds4_gpu_v41_win_ring_snap_tensor(tw, ts, pos0, 1, n, W, hd, 1, NULL) != 0 && test_v41_read(tw, got, (size_t)(W + n) * hd * 4) && test_v41_read(ts, snap, (size_t)n * hd * 4));
      for (uint32_t i = 1; i < n; i++) for (uint32_t d = 0; d < hd; d++) if (got[((pos0 + i) % W) * hd + d] != buf[((pos0 + i) % W) * hd + d]) bad++;   /* 第 1.. 行还原 */
      for (uint32_t d = 0; d < hd; d++) if (got[(pos0 % W) * hd + d] != buf[(W + 0) * hd + d]) bad++;                                              /* 第 0 行保留提交 */
      fprintf(stderr, "ds4: [v41-test] win ring commit/snap/restore 不一致 %d\n", bad);
      TEST_ASSERT(bad == 0);
      ds4_gpu_tensor_free(tw); ds4_gpu_tensor_free(ts); free(buf); free(got); free(snap); }
}

void test_metal_v41_layer(void) {
    test_v41_seed(0x4c);
    test_norm_hc();
    test_router_swiglu_pool_rope();
    test_quant_pack_ring();
}
#else
typedef int ds4_t_metal_v41_layer_nonempty_tu;
#endif
