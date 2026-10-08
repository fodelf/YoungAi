/* t_metal_v41_vq.c — V4.1 Metal VQ 专家回归(2026-10-08): 在合成模型里造 DQVL v2(f16 码本, 9 位)与 v3(层码本 E4M3, 12 位 / 13 位带位平面)blob,
 * 验 解码路(n ≤ 8) / 预填路(n > 8) / 尾巴合一 / 反修增益覆盖 / 草稿塔 dense MoE, 金标 = 按 vq_fmt.h 布局写的 CPU 解码 + 官方 Expert 的 bf16 边界算式。 */
#include "test_internal.h"
#if !defined(DS4_NO_GPU) && defined(__APPLE__)   /* Metal 专用: CUDA 构建里这些核的形状闸不同(专家核只认 11/12/13 位、注意力只认 64 头), 合成小形状不适用 */
#include "../src/common/ds4_fp8.h"
#include "../src/metal/metal_v41_args.h"
#include "../ds4_gpu_v41.h"
#include "../ds4_gpu_bwd.h"

#define VQ_NE 4u
#define VQ_IN 256u
#define VQ_MID 256u
#define VQ_OUT 256u
typedef struct { uint32_t ver, nc, nbit, ext; uint64_t blob_off, blob_bytes; float *w[VQ_NE][3]; } vq_fix;   /* w[e][which] = 解码后的 [rows][cols] f32 */

static void put_u16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }
static void put_u32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static void put_u64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }
/* 造一个 blob。v2: 每载荷自带 nc×8 f16 码本; v3: 层码本 nc×8 E4M3 在表后, 载荷 32 B 头 + 行增益 + 12 位主流 (+ 位平面)。 */
static void vq_build(vq_fix *f, uint32_t ver, uint32_t nc) {
    uint32_t nbit = 0; while ((1u << nbit) < nc) nbit++;
    f->ver = ver; f->nc = nc; f->nbit = nbit; f->ext = (ver == 3u && nbit > 12u) ? 1u : 0u;
    const uint32_t rows_[3] = { VQ_MID, VQ_MID, VQ_OUT }, cols_[3] = { VQ_IN, VQ_IN, VQ_MID };
    uint64_t total = 16u + VQ_NE * 3u * 8u + (ver == 3u ? (uint64_t)nc * 8u : 0u);
    for (uint32_t e = 0; e < VQ_NE; e++) for (uint32_t w = 0; w < 3u; w++) {
        const uint32_t rows = rows_[w], nidx = cols_[w] / 8u, mb = ver == 3u ? 12u : nbit;
        total = (total + 15u) & ~15ull;
        total += (ver == 3u ? 32u : 16u + (uint64_t)nc * 16u) + rows * 2u + (uint64_t)rows * ((nidx * mb + 7u) / 8u) + (f->ext ? (uint64_t)rows * ((nidx + 7u) / 8u) : 0u) + 8u;
    }
    f->blob_off = test_v41_model_alloc(total + 64u); f->blob_bytes = total + 64u;
    uint8_t *b = test_v41_model_ptr(f->blob_off);
    put_u32(b, 0x4C565144u); put_u32(b + 4, ver); put_u32(b + 8, 0); put_u32(b + 12, VQ_NE);
    uint64_t cur = 16u + VQ_NE * 3u * 8u;
    float *cbl = malloc((size_t)nc * 8u * 4);   /* 层码本(v3)或本载荷码本(v2)的 f32 值 */
    uint64_t cb_off = 0;
    if (ver == 3u) {
        cb_off = cur;
        for (uint32_t i = 0; i < nc * 8u; i++) { uint8_t by = (uint8_t)test_v41_rand_u32(); if ((by & 0x7f) == 0x7f) by &= 0x7e; if ((by & 0x7f) > 0x60) by &= 0x5f; b[cur + i] = by; cbl[i] = ds4_e4m3fn_to_f32(by); }
        cur += (uint64_t)nc * 8u;
    }
    for (uint32_t e = 0; e < VQ_NE; e++) for (uint32_t w = 0; w < 3u; w++) {
        const uint32_t rows = rows_[w], cols = cols_[w], nidx = cols / 8u, mb = ver == 3u ? 12u : nbit;
        cur = (cur + 15u) & ~15ull;
        put_u64(b + 16u + ((uint64_t)e * 3u + w) * 8u, cur);
        uint8_t *pay = b + cur;
        put_u32(pay, ver == 3u ? 0x33565144u : 0x51565144u); put_u16(pay + 4, 8); put_u16(pay + 6, (uint16_t)nc); put_u32(pay + 8, rows); put_u32(pay + 12, cols);
        uint8_t *gr, *ix;
        if (ver == 3u) { put_u32(pay + 16, 1u | (f->ext ? 2u : 0u)); put_u32(pay + 20, 12u); put_u64(pay + 24, cb_off); gr = pay + 32; }
        else {
            for (uint32_t i = 0; i < nc * 8u; i++) { const float v = test_v41_randf(); const uint16_t h = test_float_to_f16(v); put_u16(pay + 16 + i * 2, h); cbl[i] = test_f16_to_f32(h); }
            gr = pay + 16 + (uint64_t)nc * 16u;
        }
        ix = gr + rows * 2u;
        const uint32_t mrow = (nidx * mb + 7u) / 8u, erow = (nidx + 7u) / 8u;
        uint8_t *ex = ix + (uint64_t)rows * mrow;
        memset(ix, 0, (size_t)rows * mrow + (f->ext ? (size_t)rows * erow : 0u) + 8u);
        float *wr = malloc((size_t)rows * cols * 4);
        for (uint32_t r = 0; r < rows; r++) {
            const uint16_t gh = test_float_to_f16(0.05f + 0.02f * (float)(test_v41_rand_u32() % 10u));
            put_u16(gr + r * 2u, gh);
            const float g = test_f16_to_f32(gh);
            for (uint32_t j = 0; j < nidx; j++) {
                const uint32_t code = test_v41_rand_u32() % nc, main = code & ((1u << mb) - 1u);
                for (uint32_t bi = 0; bi < mb; bi++) if ((main >> bi) & 1u) { const uint64_t bit = (uint64_t)r * nidx * mb + (uint64_t)j * mb + bi; ix[bit >> 3] |= (uint8_t)(1u << (bit & 7u)); }
                if (f->ext && (code >> 12) & 1u) ex[(uint64_t)r * erow + (j >> 3)] |= (uint8_t)(1u << (j & 7u));
                for (uint32_t q = 0; q < 8u; q++) wr[(uint64_t)r * cols + j * 8u + q] = cbl[code * 8u + q] * g;
            }
        }
        f->w[e][w] = wr;
        cur += (ver == 3u ? 32u : 16u + (uint64_t)nc * 16u) + rows * 2u + (uint64_t)rows * mrow + (f->ext ? (uint64_t)rows * erow : 0u) + 8u;
    }
    free(cbl);
}
static void vq_free(vq_fix *f) { for (uint32_t e = 0; e < VQ_NE; e++) for (uint32_t w = 0; w < 3u; w++) free(f->w[e][w]); }
static float dot(const float *a, const float *b, uint32_t n) { double s = 0.0; for (uint32_t i = 0; i < n; i++) s += (double)a[i] * b[i]; return (float)s; }
/* CPU 参考 MoE: out[t] = Σ_k rw·bf16(W2·bf16(swiglu(bf16(W1·x), bf16(W3·x))))·(gov ? gov[e][r] : 1) */
static void vq_ref(const vq_fix *f, const float *x, const int32_t *sel, const float *rw, uint32_t n, uint32_t K, float clamp, const float *gov, float *out, float *O_rows) {
    float *g = malloc(VQ_MID * 4), *u = malloc(VQ_MID * 4), *h = malloc(VQ_MID * 4);
    for (uint32_t t = 0; t < n; t++) {
        for (uint32_t o = 0; o < VQ_OUT; o++) out[t * VQ_OUT + o] = 0.f;
        for (uint32_t k = 0; k < K; k++) {
            const int32_t e = sel[t * K + k];
            if (e < 0) continue;
            for (uint32_t r = 0; r < VQ_MID; r++) {
                g[r] = test_v41_bf16r(dot(f->w[e][0] + (uint64_t)r * VQ_IN, x + t * VQ_IN, VQ_IN));
                u[r] = test_v41_bf16r(dot(f->w[e][1] + (uint64_t)r * VQ_IN, x + t * VQ_IN, VQ_IN));
                float gv = g[r], uv = u[r];
                if (clamp > 0.f) { gv = fminf(gv, clamp); uv = fminf(fmaxf(uv, -clamp), clamp); }
                h[r] = test_v41_bf16r(gv / (1.f + expf(-gv)) * uv);
            }
            for (uint32_t o = 0; o < VQ_OUT; o++) {
                const float ov = test_v41_bf16r(dot(f->w[e][2] + (uint64_t)o * VQ_MID, h, VQ_MID) * (gov ? gov[(uint64_t)e * VQ_OUT + o] : 1.f));
                if (O_rows) O_rows[(t * K + k) * VQ_OUT + o] = ov;
                out[t * VQ_OUT + o] += rw[t * K + k] * ov;
            }
        }
    }
    free(g); free(u); free(h);
}
static void vq_run(const vq_fix *f, const char *tag) {
    const uint32_t K = 2; const float clamp = 2.0f;
    const uint32_t ntoks[2] = { 3, 12 };
    for (int c = 0; c < 2; c++) {
        const uint32_t n = ntoks[c], np = n * K;
        float *x = malloc((size_t)n * VQ_IN * 4), *rw = malloc(np * 4), *ref = malloc((size_t)n * VQ_OUT * 4), *got = malloc((size_t)n * VQ_OUT * 4);
        int32_t *sel = malloc(np * 4);
        test_v41_fill_bf16(x, (uint64_t)n * VQ_IN, 1.f);
        for (uint32_t i = 0; i < np; i++) { sel[i] = (int32_t)((i * 7u + 3u) % (VQ_NE + 1u)) - 1; rw[i] = 0.3f + 0.1f * (float)(i % 5u); }   /* 含 -1 空槽 */
        ds4_gpu_tensor *tx = test_v41_tensor(x, (size_t)n * VQ_IN * 4), *tsel = test_v41_tensor(sel, np * 4), *trw = test_v41_tensor(rw, np * 4), *to = test_v41_tensor(NULL, (size_t)n * VQ_OUT * 4);
        TEST_ASSERT(ds4_gpu_v41_routed_moe_tensor(to, test_v41_model_map(), test_v41_model_size(), f->blob_off, f->blob_bytes, VQ_IN, VQ_MID, VQ_OUT, tsel, trw, VQ_NE, K, clamp, tx, 0, n) != 0);
        TEST_ASSERT(test_v41_read(to, got, (size_t)n * VQ_OUT * 4));
        vq_ref(f, x, sel, rw, n, K, clamp, NULL, ref, NULL);
        char name[80]; snprintf(name, sizeof name, "%s routed_moe n=%u", tag, n);
        test_v41_cmp(name, got, ref, (uint64_t)n * VQ_OUT, 1e-2f, 1e-3f);
        if (n <= DS4_V41_GEMV_MAX_TOK) {   /* 尾巴合一: out=NULL + moe_tail(y = bf16(Σ + so)) */
            float *so = malloc((size_t)n * VQ_OUT * 4); test_v41_fill_bf16(so, (uint64_t)n * VQ_OUT, 1.f);
            ds4_gpu_tensor *tso = test_v41_tensor(so, (size_t)n * VQ_OUT * 4), *ty = test_v41_tensor(NULL, (size_t)n * VQ_OUT * 4);
            TEST_ASSERT(ds4_gpu_v41_routed_moe_tensor(NULL, test_v41_model_map(), test_v41_model_size(), f->blob_off, f->blob_bytes, VQ_IN, VQ_MID, VQ_OUT, tsel, trw, VQ_NE, K, clamp, tx, 0, n) != 0);
            TEST_ASSERT(ds4_gpu_v41_moe_tail_tensor(ty, tso, trw, n, K, VQ_OUT) != 0 && test_v41_read(ty, got, (size_t)n * VQ_OUT * 4));
            for (uint32_t i = 0; i < n * VQ_OUT; i++) ref[i] = test_v41_bf16r(ref[i] + so[i]);
            snprintf(name, sizeof name, "%s moe_tail n=%u", tag, n);
            test_v41_cmp(name, got, ref, (uint64_t)n * VQ_OUT, 1e-2f, 1e-3f);
            ds4_gpu_tensor_free(tso); ds4_gpu_tensor_free(ty); free(so);
        } else {   /* 预填: 取料 = reduce 前的逐专家 down 输出 */
            float *cap = malloc((size_t)np * VQ_OUT * 4), *capr = malloc((size_t)np * VQ_OUT * 4);
            TEST_ASSERT(ds4_gpu_v41_vq_capture_expert_out(cap, n, K, VQ_OUT) != 0);
            vq_ref(f, x, sel, rw, n, K, clamp, NULL, ref, capr);
            for (uint32_t i = 0; i < np; i++) if (sel[i] < 0) for (uint32_t o = 0; o < VQ_OUT; o++) capr[i * VQ_OUT + o] = 0.f;
            snprintf(name, sizeof name, "%s capture_expert_out", tag);
            test_v41_cmp(name, cap, capr, (uint64_t)np * VQ_OUT, 1e-2f, 1e-3f);
            free(cap); free(capr);
        }
        /* 反修增益覆盖: 层 0 挂 s[e][o], down 行增益乘它 */
        float *gov = malloc(VQ_NE * VQ_OUT * 4);
        for (uint32_t i = 0; i < VQ_NE * VQ_OUT; i++) gov[i] = 1.0f + 0.3f * test_v41_randf();
        TEST_ASSERT(ds4_gpu_v41_set_gr_override(0, gov, VQ_NE, VQ_OUT) != 0);
        TEST_ASSERT(ds4_gpu_v41_routed_moe_tensor(to, test_v41_model_map(), test_v41_model_size(), f->blob_off, f->blob_bytes, VQ_IN, VQ_MID, VQ_OUT, tsel, trw, VQ_NE, K, clamp, tx, 0, n) != 0);
        TEST_ASSERT(test_v41_read(to, got, (size_t)n * VQ_OUT * 4));
        vq_ref(f, x, sel, rw, n, K, clamp, gov, ref, NULL);
        snprintf(name, sizeof name, "%s routed_moe+gr n=%u", tag, n);
        test_v41_cmp(name, got, ref, (uint64_t)n * VQ_OUT, 1e-2f, 1e-3f);
        TEST_ASSERT(ds4_gpu_v41_set_gr_override(0, NULL, 0, 0) != 0);
        /* 反向: g_w = <g_y, O>(O 舍 bf16); g_x 对着 CPU 直通梯度(不舍 bf16 的解析式)比, 容差放到 3% */
        if (n <= DS4_V41_GEMV_MAX_TOK) {
            float *gy = malloc((size_t)n * VQ_OUT * 4), *gx = malloc((size_t)n * VQ_IN * 4), *gw = malloc(np * 4), *gwr = malloc(np * 4), *gxr = calloc((size_t)n * VQ_IN, 4), *O = malloc((size_t)np * VQ_OUT * 4);
            test_v41_fill_bf16(gy, (uint64_t)n * VQ_OUT, 1.f);
            ds4_gpu_tensor *tgy = test_v41_tensor(gy, (size_t)n * VQ_OUT * 4), *tgx = test_v41_tensor(NULL, (size_t)n * VQ_IN * 4), *tgw = test_v41_tensor(NULL, np * 4);
            TEST_ASSERT(ds4_gpu_bwd_routed_moe_tensor(tgx, tgw, tgy, tx, tsel, trw, test_v41_model_map(), test_v41_model_size(), f->blob_off, f->blob_bytes, VQ_IN, VQ_MID, VQ_OUT, VQ_NE, K, clamp, 0, n) != 0);
            TEST_ASSERT(test_v41_read(tgx, gx, (size_t)n * VQ_IN * 4) && test_v41_read(tgw, gw, np * 4));
            vq_ref(f, x, sel, rw, n, K, clamp, NULL, ref, O);
            for (uint32_t t = 0; t < n; t++) for (uint32_t k = 0; k < K; k++) {
                const int32_t e = sel[t * K + k];
                gwr[t * K + k] = e < 0 ? 0.f : dot(gy + t * VQ_OUT, O + (t * K + k) * VQ_OUT, VQ_OUT);
                if (e < 0) continue;
                /* 直通解析: G_O = gy·rw; G_A = W2ᵀ G_O; (G_Hg, G_Hu) = swiglu'(g,u)·G_A; G_X += W1ᵀ G_Hg + W3ᵀ G_Hu。
                 * ★g/u 先舍 bf16 再判截断★: 反向对着前向本身求导, 前向的 gate/up 出口就在 bf16 格点上 —— 不舍的话 g=1.9993 vs 2.0
                 * 的截断边会翻面(10-08 首跑实撞: 一行 G_Hg 被判 0, 整条 g_x 差 28%), 那是参考错了不是核错了。 */
                float *GO = malloc(VQ_OUT * 4), *GA = calloc(VQ_MID, 4), *g = malloc(VQ_MID * 4), *u = malloc(VQ_MID * 4);
                for (uint32_t o = 0; o < VQ_OUT; o++) GO[o] = gy[t * VQ_OUT + o] * rw[t * K + k];
                for (uint32_t o = 0; o < VQ_OUT; o++) for (uint32_t r = 0; r < VQ_MID; r++) GA[r] += GO[o] * f->w[e][2][(uint64_t)o * VQ_MID + r];
                for (uint32_t r = 0; r < VQ_MID; r++) { g[r] = test_v41_bf16r(dot(f->w[e][0] + (uint64_t)r * VQ_IN, x + t * VQ_IN, VQ_IN)); u[r] = test_v41_bf16r(dot(f->w[e][1] + (uint64_t)r * VQ_IN, x + t * VQ_IN, VQ_IN)); }
                for (uint32_t r = 0; r < VQ_MID; r++) {
                    float gv = g[r], uv = u[r];
                    const int gpass = gv < clamp, upass = uv > -clamp && uv < clamp;
                    gv = fminf(gv, clamp); uv = fminf(fmaxf(uv, -clamp), clamp);
                    const float sg = 1.f / (1.f + expf(-gv)), si = gv * sg;
                    const float gg = gpass ? GA[r] * uv * sg * (1.f + gv * (1.f - sg)) : 0.f, gu = upass ? GA[r] * si : 0.f;
                    for (uint32_t i = 0; i < VQ_IN; i++) gxr[t * VQ_IN + i] += gg * f->w[e][0][(uint64_t)r * VQ_IN + i] + gu * f->w[e][1][(uint64_t)r * VQ_IN + i];
                }
                free(GO); free(GA); free(g); free(u);
            }
            snprintf(name, sizeof name, "%s bwd g_w", tag); test_v41_cmp(name, gw, gwr, np, 1e-2f, 1e-3f);
            snprintf(name, sizeof name, "%s bwd g_x(直通)", tag); test_v41_cmp(name, gx, gxr, (uint64_t)n * VQ_IN, 3e-2f, 1e-3f);
            ds4_gpu_tensor_free(tgy); ds4_gpu_tensor_free(tgx); ds4_gpu_tensor_free(tgw); free(gy); free(gx); free(gw); free(gwr); free(gxr); free(O);
        }
        ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(tsel); ds4_gpu_tensor_free(trw); ds4_gpu_tensor_free(to); free(x); free(rw); free(ref); free(got); free(sel); free(gov);
    }
}
/* 草稿塔 dense MoE: 逐专家 fp4x32 张量 */
static void test_mtp(void) {
    float *wref[VQ_NE][3]; uint64_t off[3 * VQ_NE];
    const uint32_t rows_[3] = { VQ_MID, VQ_MID, VQ_OUT }, cols_[3] = { VQ_IN, VQ_IN, VQ_MID };
    TEST_ASSERT(test_v41_model_begin(VQ_NE * 3u * test_v41_wbytes(V41_WT_FP4X32, VQ_OUT, VQ_IN) + 65536));
    for (uint32_t e = 0; e < VQ_NE; e++) for (uint32_t w = 0; w < 3u; w++) {
        const uint64_t nb = test_v41_wbytes(V41_WT_FP4X32, rows_[w], cols_[w]);
        off[w * VQ_NE + e] = test_v41_model_alloc(nb);
        wref[e][w] = malloc((size_t)rows_[w] * cols_[w] * 4);
        test_v41_make_weights(V41_WT_FP4X32, rows_[w], cols_[w], test_v41_model_ptr(off[w * VQ_NE + e]), wref[e][w]);
    }
    TEST_ASSERT(test_v41_map());
    vq_fix f; memset(&f, 0, sizeof f);
    for (uint32_t e = 0; e < VQ_NE; e++) for (uint32_t w = 0; w < 3u; w++) f.w[e][w] = wref[e][w];
    const uint32_t n = 5, K = 3, np = n * K; const float clamp = 2.0f;
    float *x = malloc((size_t)n * VQ_IN * 4), *rw = malloc(np * 4), *ref = malloc((size_t)n * VQ_OUT * 4), *got = malloc((size_t)n * VQ_OUT * 4);
    int32_t *sel = malloc(np * 4);
    test_v41_fill_bf16(x, (uint64_t)n * VQ_IN, 1.f);
    for (uint32_t i = 0; i < np; i++) { sel[i] = (int32_t)((i * 5u + 1u) % (VQ_NE + 1u)) - 1; rw[i] = 0.2f + 0.1f * (float)(i % 4u); }
    ds4_gpu_tensor *tx = test_v41_tensor(x, (size_t)n * VQ_IN * 4), *tsel = test_v41_tensor(sel, np * 4), *trw = test_v41_tensor(rw, np * 4), *to = test_v41_tensor(NULL, (size_t)n * VQ_OUT * 4);
    TEST_ASSERT(ds4_gpu_v41_mtp_moe_tensor(to, test_v41_model_map(), 0, off, VQ_IN, VQ_MID, VQ_OUT, tsel, trw, VQ_NE, K, clamp, tx, n) != 0);
    TEST_ASSERT(test_v41_read(to, got, (size_t)n * VQ_OUT * 4));
    vq_ref(&f, x, sel, rw, n, K, clamp, NULL, ref, NULL);
    test_v41_cmp("mtp dense moe", got, ref, (uint64_t)n * VQ_OUT, 1e-2f, 1e-3f);
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(tsel); ds4_gpu_tensor_free(trw); ds4_gpu_tensor_free(to); free(x); free(rw); free(ref); free(got); free(sel);
    for (uint32_t e = 0; e < VQ_NE; e++) for (uint32_t w = 0; w < 3u; w++) free(wref[e][w]);
}
void test_metal_v41_vq(void) {
    test_v41_seed(0x56);
    const uint32_t cfg[3][2] = { { 2u, 512u }, { 3u, 4096u }, { 3u, 8192u } };
    const char *tag[3] = { "v2/9bit", "v3/12bit", "v3/13bit" };
    for (int c = 0; c < 3; c++) {
        TEST_ASSERT(test_v41_model_begin(64u << 20));
        vq_fix f; memset(&f, 0, sizeof f);
        vq_build(&f, cfg[c][0], cfg[c][1]);
        TEST_ASSERT(test_v41_map());
        vq_run(&f, tag[c]);
        vq_free(&f);
    }
    test_mtp();
}
#else
typedef int ds4_t_metal_v41_vq_nonempty_tu;
#endif
