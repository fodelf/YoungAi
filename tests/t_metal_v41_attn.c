/* t_metal_v41_attn.c — V4.1 Metal 稀疏注意力 / indexer 三件 / 设备采样回归(2026-10-08)。
 * 注意力金标 = CPU 精确 softmax(含 sink 分母; 核里 p 舍 bf16, 门给 2%); indexer 三件金标 = 官方算法的 CPU 转录(整数结果逐项同);
 * 采样核 = 分布与截断语义的统计/结构检查(temp 很低 ⇒ argmax; top_k=1 ⇒ 保留集 1; 峰值分布命中率)。 */
#include "test_internal.h"
#if !defined(DS4_NO_GPU) && defined(__APPLE__)   /* Metal 专用: CUDA 构建里这些核的形状闸不同(专家核只认 11/12/13 位、注意力只认 64 头), 合成小形状不适用 */
#include "../src/common/ds4_fp8.h"
#include "../ds4_gpu_v41.h"

static float ckv_get(const uint8_t *row, uint32_t d) {
    const uint8_t by = row[d >> 1], n4 = (d & 1) ? (by >> 4) : (by & 15);
    return test_v41_bf16r(ds4_fp4_nibble_to_f32(n4) * ds4_e4m3fn_to_f32(row[DS4_V41_CKV_NIB + (d >> 4)]));
}
static void test_attention(void) {
    const uint32_t NH = 8, HD = 512, W = 16, N = 4, NG = 6, TOPK = 4, pos0 = 0; const float scale = 0.0442f;
    TEST_ASSERT(test_v41_model_begin(NH * 4 + 4096));
    float sink[8]; for (uint32_t h = 0; h < NH; h++) sink[h] = test_v41_randf();
    const uint64_t o_sink = test_v41_model_alloc(NH * 4); memcpy(test_v41_model_ptr(o_sink), sink, NH * 4);
    TEST_ASSERT(test_v41_map());
    float *q = malloc((size_t)N * NH * HD * 4), *win = malloc((size_t)(W + N) * HD * 4), *crow = malloc((size_t)NG * HD * 4);
    test_v41_fill_bf16(q, (uint64_t)N * NH * HD, 0.5f); test_v41_fill_bf16(win, (uint64_t)(W + N) * HD, 0.5f); test_v41_fill_bf16(crow, (uint64_t)NG * HD, 0.5f);
    int32_t idx[4 * 4];
    for (uint32_t i = 0; i < N * TOPK; i++) idx[i] = (int32_t)((i * 5u) % (NG + 2u)) - 1;   /* 含 -1 与越界(NG) */
    ds4_gpu_tensor *tq = test_v41_tensor(q, (size_t)N * NH * HD * 4), *tw = test_v41_tensor(win, (size_t)(W + N) * HD * 4), *tc = test_v41_tensor(NULL, (uint64_t)NG * DS4_V41_CKV_BYTES);
    ds4_gpu_tensor *trow = test_v41_tensor(crow, (size_t)NG * HD * 4), *tidx = test_v41_tensor(idx, N * TOPK * 4), *to = test_v41_tensor(NULL, (size_t)N * NH * HD * 4);
    TEST_ASSERT(ds4_gpu_v41_ckv_pack_tensor(tc, 0, trow, NG, NULL, 0, 0, 0) != 0);
    uint8_t *cache = malloc((size_t)NG * DS4_V41_CKV_BYTES);
    TEST_ASSERT(test_v41_read(tc, cache, (uint64_t)NG * DS4_V41_CKV_BYTES));
    TEST_ASSERT(ds4_gpu_v41_sparse_attn_tensor(to, tq, tw, tc, tidx, test_v41_model_map(), test_v41_model_size(), o_sink, N, pos0, W, NG, TOPK, 4, NH, HD, scale, 0, 1, 0, NULL, 0) != 0);
    float *got = malloc((size_t)N * NH * HD * 4), *ref = malloc((size_t)N * NH * HD * 4), *keys = malloc((size_t)(W + TOPK) * HD * 4), *s = malloc((W + TOPK) * 4);
    TEST_ASSERT(test_v41_read(to, got, (size_t)N * NH * HD * 4));
    for (uint32_t i = 0; i < N; i++) {
        const uint32_t p = pos0 + i, lo = p + 1 > W ? p + 1 - W : 0, nwin = p - lo + 1;
        uint32_t nk = 0;
        for (uint32_t kk = 0; kk < nwin; kk++) { const int a = (int)(lo + kk); const uint32_t row = a >= (int)pos0 ? W + (uint32_t)(a - (int)pos0) : (uint32_t)(a % (int)W); memcpy(keys + nk * HD, win + row * HD, HD * 4); nk++; }
        for (uint32_t kk = 0; kk < TOPK; kk++) { const int32_t g = idx[i * TOPK + kk]; if (g < 0 || (uint32_t)g >= NG) continue; for (uint32_t d = 0; d < HD; d++) keys[nk * HD + d] = ckv_get(cache + (size_t)g * DS4_V41_CKV_BYTES, d); nk++; }
        for (uint32_t h = 0; h < NH; h++) {
            const float *qh = q + ((uint64_t)i * NH + h) * HD;
            float mx = -3e38f;
            for (uint32_t k = 0; k < nk; k++) { double d = 0.0; for (uint32_t e = 0; e < HD; e++) d += (double)qh[e] * keys[k * HD + e]; s[k] = (float)d * scale; mx = fmaxf(mx, s[k]); }
            double den = exp((double)sink[h] - mx);
            float *o = ref + ((uint64_t)i * NH + h) * HD;
            for (uint32_t e = 0; e < HD; e++) o[e] = 0.f;
            for (uint32_t k = 0; k < nk; k++) { const double pk = exp((double)s[k] - mx); den += pk; for (uint32_t e = 0; e < HD; e++) o[e] += (float)pk * keys[k * HD + e]; }
            for (uint32_t e = 0; e < HD; e++) o[e] = test_v41_bf16r((float)(o[e] / den));
        }
    }
    test_v41_cmp("sparse_attn", got, ref, (uint64_t)N * NH * HD, 2e-2f, 1e-3f);
    ds4_gpu_tensor_free(tq); ds4_gpu_tensor_free(tw); ds4_gpu_tensor_free(tc); ds4_gpu_tensor_free(trow); ds4_gpu_tensor_free(tidx); ds4_gpu_tensor_free(to);
    free(q); free(win); free(crow); free(cache); free(got); free(ref); free(keys); free(s);
}
static float idxk_get(const uint8_t *row, uint32_t d) {
    const uint8_t by = row[d >> 1], n4 = (d & 1) ? (by >> 4) : (by & 15);
    return test_v41_bf16r(ds4_fp4_nibble_to_f32(n4) * ds4_e8m0_to_f32(row[DS4_V41_IDXK_NIB + (d >> 5)]));
}
static uint32_t key_of(float f) { if (!(f > -3.0e38f)) return 0u; uint32_t u; memcpy(&u, &f, 4); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
static void test_indexer(void) {
    const uint32_t N = 3, NH = 4, DK = 128, RATIO = 4, pos0 = 100, NG = 40, TOPK = 8, BS = 4, CAP = 5;
    float *q = malloc((size_t)N * NH * DK * 4), *w = malloc(N * NH * 4), *krow = malloc((size_t)NG * DK * 4);
    test_v41_fill_bf16(q, (uint64_t)N * NH * DK, 1.f); test_v41_fill_bf16(krow, (uint64_t)NG * DK, 1.f);
    for (uint32_t i = 0; i < N * NH; i++) w[i] = test_v41_bf16r(0.1f + 0.5f * fabsf(test_v41_randf()));
    ds4_gpu_tensor *tq = test_v41_tensor(q, (size_t)N * NH * DK * 4), *tw = test_v41_tensor(w, N * NH * 4), *tkr = test_v41_tensor(krow, (size_t)NG * DK * 4);
    ds4_gpu_tensor *tk = test_v41_tensor(NULL, (uint64_t)NG * DS4_V41_IDXK_BYTES), *tsc = test_v41_tensor(NULL, (size_t)N * NG * 4), *tcl = test_v41_tensor(NULL, (size_t)N * (1 + CAP) * 4), *tidx = test_v41_tensor(NULL, N * TOPK * 4);
    TEST_ASSERT(ds4_gpu_v41_idxk_pack_tensor(tk, 0, tkr, NG, NULL, 0, 0, 0) != 0);
    uint8_t *kc = malloc((size_t)NG * DS4_V41_IDXK_BYTES);
    TEST_ASSERT(test_v41_read(tk, kc, (uint64_t)NG * DS4_V41_IDXK_BYTES));
    /* ① 全宽打分 */
    TEST_ASSERT(ds4_gpu_v41_indexer_score_tensor(tsc, tq, tk, tw, NULL, 0, 0, N, pos0, NG, NH, DK, RATIO, NULL) != 0);
    float *sc = malloc((size_t)N * NG * 4), *scr = malloc((size_t)N * NG * 4);
    TEST_ASSERT(test_v41_read(tsc, sc, (size_t)N * NG * 4));
    for (uint32_t i = 0; i < N; i++) {
        const uint32_t vis = (pos0 + i + 1) / RATIO;
        for (uint32_t g = 0; g < NG; g++) {
            if (g >= vis) { scr[i * NG + g] = -3.0e38f; continue; }
            float acc = 0.f;
            for (uint32_t h = 0; h < NH; h++) { float d = 0.f; for (uint32_t e = 0; e < DK; e++) d += q[(i * NH + h) * DK + e] * idxk_get(kc + (size_t)g * DS4_V41_IDXK_BYTES, e); d = fmaxf(test_v41_bf16r(d), 0.f); acc += test_v41_bf16r(d * w[i * NH + h]); }
            scr[i * NG + g] = test_v41_bf16r(acc);
        }
    }
    int bad = 0;
    for (uint32_t i = 0; i < N * NG; i++) { if (scr[i] <= -3.0e38f) { if (sc[i] > -3.0e38f) bad++; } else if (fabsf(sc[i] - scr[i]) > 2e-2f * fabsf(scr[i]) + 1e-3f) bad++; }
    fprintf(stderr, "ds4: [v41-test] indexer_score 全宽: %u×%u 不一致 %d\n", N, NG, bad); TEST_ASSERT(bad == 0);
    /* ② 候选块(用 GPU 分当输入, 与 CPU 转录比) + ③ topk(全宽) */
    TEST_ASSERT(ds4_gpu_v41_candidate_blocks_tensor(tcl, tsc, N, pos0, NG, RATIO, CAP, BS, NULL) != 0);
    TEST_ASSERT(ds4_gpu_v41_indexer_topk_tensor(tidx, tsc, N, NG, TOPK, RATIO, NULL, NULL, 0, 0) != 0);
    int32_t *cl = malloc((size_t)N * (1 + CAP) * 4), *ix = malloc(N * TOPK * 4);
    TEST_ASSERT(test_v41_read(tcl, cl, (size_t)N * (1 + CAP) * 4) && test_v41_read(tidx, ix, N * TOPK * 4));
    bad = 0;
    for (uint32_t i = 0; i < N; i++) {
        const uint32_t vis = (pos0 + i + 1) / RATIO, nb = (NG + BS - 1) / BS;
        float bsc[16]; int selb[16] = { 0 };
        for (uint32_t b = 0; b < nb; b++) { float m = -3.0e38f; for (uint32_t j = b * BS; j < (b + 1) * BS && j < NG; j++) m = fmaxf(m, sc[i * NG + j]); if (vis > 0 && b == (vis - 1) / BS) m = 3.0e38f; bsc[b] = m; }
        const uint32_t kk = CAP < nb ? CAP : nb;
        for (uint32_t r = 0; r < kk; r++) { int be = -1; float bv = -3.0e38f; for (uint32_t b = 0; b < nb; b++) if (!selb[b] && bsc[b] > bv) { bv = bsc[b]; be = (int)b; } if (be < 0) break; selb[be] = 1; }
        int cnt = 0; for (uint32_t b = 0; b < nb; b++) if (selb[b]) { if (cl[i * (1 + CAP) + 1 + cnt] != (int)b) bad++; cnt++; }
        if (cl[i * (1 + CAP)] != cnt) bad++;
        /* topk: 按分降序取 min(TOPK, 可达数), 再按位置升序; 并列取小下标 */
        int chosen[8]; int nch = 0; int used[40] = { 0 };
        for (uint32_t r = 0; r < TOPK; r++) { int be = -1; float bv = -3.0e38f; for (uint32_t g = 0; g < NG; g++) if (!used[g] && sc[i * NG + g] > bv) { bv = sc[i * NG + g]; be = (int)g; } if (be < 0) break; used[be] = 1; chosen[nch++] = be; }
        for (int a = 0; a < nch; a++) for (int b2 = a + 1; b2 < nch; b2++) if (chosen[b2] < chosen[a]) { int t = chosen[a]; chosen[a] = chosen[b2]; chosen[b2] = t; }
        for (uint32_t r = 0; r < TOPK; r++) if (ix[i * TOPK + r] != (r < (uint32_t)nch ? chosen[r] : -1)) bad++;
    }
    fprintf(stderr, "ds4: [v41-test] candidate+topk: 不一致 %d\n", bad); TEST_ASSERT(bad == 0);
    /* ④ C2 紧凑路: 列表打分 + 列表 topk 必须与全宽 topk 在候选集合内等价 */
    TEST_ASSERT(ds4_gpu_v41_indexer_score_tensor(tsc, tq, tk, tw, tcl, BS, CAP, N, pos0, NG, NH, DK, RATIO, NULL) != 0);
    TEST_ASSERT(ds4_gpu_v41_indexer_topk_tensor(tidx, tsc, N, NG, TOPK, RATIO, NULL, tcl, BS, CAP) != 0);
    int32_t *ix2 = malloc(N * TOPK * 4);
    TEST_ASSERT(test_v41_read(tidx, ix2, N * TOPK * 4));
    bad = 0;
    for (uint32_t i = 0; i < N; i++) {
        /* 候选集合 = 列表块里的组: 在其中按全宽分取 topk */
        int inc[40] = { 0 }; const int nc = cl[i * (1 + CAP)];
        for (int b = 0; b < nc; b++) for (uint32_t j = 0; j < BS; j++) { const uint32_t g = (uint32_t)cl[i * (1 + CAP) + 1 + b] * BS + j; if (g < NG) inc[g] = 1; }
        int chosen[8]; int nch = 0; int used[40] = { 0 };
        for (uint32_t r = 0; r < TOPK; r++) { int be = -1; float bv = -3.0e38f; for (uint32_t g = 0; g < NG; g++) if (inc[g] && !used[g] && scr[i * NG + g] > bv && key_of(scr[i * NG + g])) { bv = scr[i * NG + g]; be = (int)g; } if (be < 0) break; used[be] = 1; chosen[nch++] = be; }
        for (int a = 0; a < nch; a++) for (int b2 = a + 1; b2 < nch; b2++) if (chosen[b2] < chosen[a]) { int t = chosen[a]; chosen[a] = chosen[b2]; chosen[b2] = t; }
        for (uint32_t r = 0; r < TOPK; r++) if (ix2[i * TOPK + r] != (r < (uint32_t)nch ? chosen[r] : -1)) bad++;
    }
    fprintf(stderr, "ds4: [v41-test] C2 紧凑 topk: 不一致 %d\n", bad); TEST_ASSERT(bad == 0);
    ds4_gpu_tensor_free(tq); ds4_gpu_tensor_free(tw); ds4_gpu_tensor_free(tkr); ds4_gpu_tensor_free(tk); ds4_gpu_tensor_free(tsc); ds4_gpu_tensor_free(tcl); ds4_gpu_tensor_free(tidx);
    free(q); free(w); free(krow); free(kc); free(sc); free(scr); free(cl); free(ix); free(ix2);
}
static void test_sampler(void) {
    const uint32_t V = 1000, R = 64;
    float *lg = malloc((size_t)R * V * 4); int32_t *pos = malloc(R * 4), *tok = malloc(R * 4), out[64 * 4];
    for (uint32_t r = 0; r < R; r++) { pos[r] = (int32_t)(1000 + r); tok[r] = (int32_t)(r * 13u % V); for (uint32_t i = 0; i < V; i++) lg[r * V + i] = test_v41_randf() * 2.f; lg[r * V + 77] = 8.0f; }
    ds4_gpu_tensor *tl = test_v41_tensor(lg, (size_t)R * V * 4), *tp = test_v41_tensor(pos, R * 4), *tt = test_v41_tensor(tok, R * 4), *to = test_v41_tensor(NULL, R * 16);
    ds4_gpu_sample_params sp = { 1.0f, 1.0f, 0.0f, 0, 12345u, 0 };
    TEST_ASSERT(ds4_gpu_v41_sample_tensor(to, tl, 0, R, V, tp, tt, &sp, NULL) != 0 && test_v41_read(to, out, R * 16));
    uint32_t hit = 0, ok = 0;
    for (uint32_t r = 0; r < R; r++) { if (out[4 * r] == 77) hit++; if (out[4 * r] >= 0 && out[4 * r] < (int)V && out[4 * r + 3] == (int)V) ok++; }
    /* 77 的概率 ≈ e^8/(e^8 + Σ_others) ≈ 0.6~0.7(随机 logits ±2), 64 行里命中应远多于 20 */
    fprintf(stderr, "ds4: [v41-test] sampler 温 1: 峰值词命中 %u/%u, 结构合法 %u/%u\n", hit, R, ok, R);
    TEST_ASSERT(ok == R && hit >= 20);
    sp.top_k = 1;
    TEST_ASSERT(ds4_gpu_v41_sample_tensor(to, tl, 0, R, V, tp, tt, &sp, NULL) != 0 && test_v41_read(to, out, R * 16));
    uint32_t am = 0; for (uint32_t r = 0; r < R; r++) if (out[4 * r] == 77 && out[4 * r + 3] == 1) am++;
    fprintf(stderr, "ds4: [v41-test] sampler top_k=1: argmax 命中 %u/%u(保留集 1)\n", am, R); TEST_ASSERT(am == R);
    sp.top_k = 0; sp.top_p = 0.5f;
    TEST_ASSERT(ds4_gpu_v41_sample_tensor(to, tl, 0, R, V, tp, tt, &sp, NULL) != 0 && test_v41_read(to, out, R * 16));
    uint32_t small = 0; for (uint32_t r = 0; r < R; r++) if (out[4 * r + 3] >= 1 && out[4 * r + 3] < 50) small++;
    fprintf(stderr, "ds4: [v41-test] sampler top_p=0.5: 保留集 <50 的行 %u/%u\n", small, R); TEST_ASSERT(small == R);
    sp.top_p = 1.0f; sp.min_p = 0.5f;
    TEST_ASSERT(ds4_gpu_v41_sample_tensor(to, tl, 0, R, V, tp, tt, &sp, NULL) != 0 && test_v41_read(to, out, R * 16));
    uint32_t one = 0; for (uint32_t r = 0; r < R; r++) if (out[4 * r] == 77 && out[4 * r + 3] == 1) one++;
    fprintf(stderr, "ds4: [v41-test] sampler min_p=0.5: 只剩峰值词 %u/%u\n", one, R); TEST_ASSERT(one == R);
    /* 投机: 草稿 = 下一行的输入 token; 温 1 下接受标志必须是 0/1, 残差样本 ≠ 草稿(残差非空时) */
    sp.min_p = 0.f;
    TEST_ASSERT(ds4_gpu_v41_sample_tensor(to, tl, 0, R, V, tp, tt, &sp, NULL) != 0 && test_v41_read(to, out, R * 16));
    uint32_t spec_ok = 0;
    for (uint32_t r = 0; r + 1 < R; r++) { const int d = tok[r + 1]; if ((out[4 * r + 1] == 0 || out[4 * r + 1] == 1) && out[4 * r + 2] != d) spec_ok++; }
    fprintf(stderr, "ds4: [v41-test] sampler 投机结构: %u/%u\n", spec_ok, R - 1); TEST_ASSERT(spec_ok == R - 1);
    ds4_gpu_tensor_free(tl); ds4_gpu_tensor_free(tp); ds4_gpu_tensor_free(tt); ds4_gpu_tensor_free(to); free(lg); free(pos); free(tok);
}
void test_metal_v41_attn(void) {
    test_v41_seed(0x61);
    test_attention();
    test_indexer();
    test_sampler();
}
#else
typedef int ds4_t_metal_v41_attn_nonempty_tu;
#endif
