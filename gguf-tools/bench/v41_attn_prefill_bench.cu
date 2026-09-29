/* v41_attn_prefill_bench.cu — V4.1 预填稀疏注意力(cuda_sparse_attn_mma.inc.cu, grid (n_tok, 头组), 两遍扫键)的形态微基准(spark 本机跑; 2026-09-29 立)。
 *
 * 【为什么要它】09-29 12k 预填逐核表: 稀疏注意力 3.84 s = 块 2048 时的 25%, 每个 query 640 键(窗口 128 + top-k 512)对 64 头, 一层块 33 ms。
 * 算力账: S 两遍 + P·V = 64.4 GFLOP/层块 ⇒ 现核只到 ~2 TFLOPS(张量核峰值 80+), 慢在 gather 两遍 / S 两遍 / 每 16 键两次 barrier / shared bank 冲突,
 * 不在乘法。换核形态先在这里过(引擎一趟要装 113 GB), 过了才进引擎。
 *
 * 【比什么】V0 = 引擎现核逐字抄本; 变体 attn_fa_kernel<HB, KT> = 单遍 flash 形态(q 常驻寄存器 / mma 直算 / 在线 softmax 缩放 O / 键片行距补齐), 见 _variants.cuh。
 * ★判据★: 不逐位同(单遍 vs 两遍的舍入路径不同), 门 = 各核对 f64 参考(同一套 bf16 键/查询值, 全键 softmax 含 sink)的最大相对误差**同一量级**
 *   (bf16 出口 ≈ 4e-3); 某变体比 V0 差一个量级就是有 bug。进引擎的判据是五指标/PPL(09-15 mma 版落地时的规矩)。
 * 形状 = 12k 真场景: 64 头 × 512, 窗口 128, top-k 512, 压缩比 4; 查询块 512 / 2048 个(pos0 = 12000); 键/查询/清单按哈希生成(相邻位置清单重叠 ~97%)。
 * 用法: nvcc -O3 -arch=native -std=c++17 -o v41_attn_prefill_bench v41_attn_prefill_bench.cu && ./v41_attn_prefill_bench [层数=8] [遍数=3] [只计时变体=-1] */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "v41_attn_prefill_bench_variants.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "★CUDA %s @%d: %s★\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

/* ---- 数据: 按 (层, 位置/组, 维) 哈希生成 ---- */
static uint32_t h32(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    uint32_t x = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA6Bu ^ (c + 0x165667B1u) * 0xC2B2AE35u ^ (d * 0x27D4EB2Fu);
    x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15; return x;
}
static float hf(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { return ((float)(h32(a, b, c, d) >> 8) / 16777216.0f) * 2.0f - 1.0f; }
static float bf16r_h(float x) { uint32_t u; memcpy(&u, &x, 4); u += 0x7FFFu + ((u >> 16) & 1u); u &= 0xFFFF0000u; float y; memcpy(&y, &u, 4); return y; }
static const uint32_t NH = 64u, HD = 512u, WINDOW = 128u, RATIO = 4u, TOPK = 512u;

typedef struct { uint32_t n, pos0, ng; float *q, *kvw, *sink, *o; uint8_t *kvc; int32_t *idx;
                 float *hq, *hkvw, *hsink; uint8_t *hkvc; int32_t *hidx; } layer_bufs;

/* 窗口环: 位置 a 在 v41_win_row(a, pos0, window, 1) 那一格; 值在 bf16 格点 */
static void fill_kvw(float *dst, uint32_t L, uint32_t pos0, uint32_t n, uint32_t window) {
    for (int64_t a = (int64_t)pos0 - window; a < (int64_t)(pos0 + n); a++) {
        const uint64_t r = v41_win_row(a, pos0, window, 1u);
        for (uint32_t d = 0; d < HD; d++) dst[r * HD + d] = bf16r_h(hf(L, 1u, (uint32_t)a, d));
    }
}
/* top-k 清单: 可见组里按哈希门槛挑, 升序; 位置每前进 4 换 ~3% 名额 ⇒ 相邻位置重叠 ~97% */
static void fill_idx(int32_t *dst, uint32_t L, uint32_t p, uint32_t ratio, uint32_t topk) {
    const uint32_t vis = (p + 1u) / ratio;
    uint32_t cnt = 0;
    for (uint32_t g = 0; g < vis && cnt < topk; g++) {
        const uint32_t hv = h32(L, 2u, g, 0u) % 1000u, jit = h32(L, 3u, g, p / 4u) % 1000u;
        const uint32_t score = (hv * 97u + jit * 3u) / 100u;
        if (score < (topk * 1000u) / vis + 20u) dst[cnt++] = (int32_t)g;
    }
    for (uint32_t g = vis; cnt < topk && g > 0; g--) { bool dup = false; for (uint32_t k = 0; k < cnt; k++) if (dst[k] == (int32_t)(g - 1)) { dup = true; break; }
                                                     if (!dup) dst[cnt++] = (int32_t)(g - 1); }
    for (uint32_t a = 1; a < cnt; a++) { int32_t v = dst[a]; uint32_t b = a; while (b > 0 && dst[b - 1] > v) { dst[b] = dst[b - 1]; b--; } dst[b] = v; }
    for (uint32_t k = cnt; k < topk; k++) dst[k] = -1;
}
static layer_bufs make_layer(uint32_t L, uint32_t n, uint32_t pos0) {
    layer_bufs b; memset(&b, 0, sizeof b); b.n = n; b.pos0 = pos0; b.ng = (pos0 + n) / RATIO;
    b.hq = (float *)malloc((size_t)n * NH * HD * 4); b.hkvw = (float *)malloc((size_t)(WINDOW + n) * HD * 4); b.hsink = (float *)malloc(NH * 4);
    b.hkvc = (uint8_t *)malloc((size_t)b.ng * DS4_V41_CKV_BYTES); b.hidx = (int32_t *)malloc((size_t)n * TOPK * 4);
    for (uint32_t i = 0; i < n; i++) for (uint32_t e = 0; e < NH * HD; e++) b.hq[(size_t)i * NH * HD + e] = bf16r_h(hf(L, 4u, pos0 + i, e) * 2.0f);
    fill_kvw(b.hkvw, L, pos0, n, WINDOW);
    for (uint32_t g = 0; g < b.ng; g++) {
        uint8_t *row = b.hkvc + (size_t)g * DS4_V41_CKV_BYTES;
        for (uint32_t k = 0; k < DS4_V41_CKV_NIB; k++) row[k] = (uint8_t)(h32(L, 5u, g, k) & 0xFFu);
        for (uint32_t s = 0; s < 32u; s++) row[DS4_V41_CKV_NIB + s] = (uint8_t)(0x30u + (h32(L, 6u, g, s) % 12u));   /* e4m3 0.25..~2, 无 NaN */
    }
    for (uint32_t i = 0; i < n; i++) fill_idx(b.hidx + (size_t)i * TOPK, L, pos0 + i, RATIO, TOPK);
    for (uint32_t h = 0; h < NH; h++) b.hsink[h] = hf(L, 7u, h, 0u);
    CK(cudaMalloc(&b.q, (size_t)n * NH * HD * 4));           CK(cudaMemcpy(b.q, b.hq, (size_t)n * NH * HD * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&b.kvw, (size_t)(WINDOW + n) * HD * 4));   CK(cudaMemcpy(b.kvw, b.hkvw, (size_t)(WINDOW + n) * HD * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&b.kvc, (size_t)b.ng * DS4_V41_CKV_BYTES)); CK(cudaMemcpy(b.kvc, b.hkvc, (size_t)b.ng * DS4_V41_CKV_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&b.idx, (size_t)n * TOPK * 4));            CK(cudaMemcpy(b.idx, b.hidx, (size_t)n * TOPK * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&b.sink, NH * 4));                         CK(cudaMemcpy(b.sink, b.hsink, NH * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&b.o, (size_t)n * NH * HD * 4));
    return b;
}

/* f64 参考: query i 头 h 的输出(全 nkeys 个键 softmax 含 sink; 键值 = 引擎同一解码式的 bf16 值, 查询 = bf16 值) */
static float ckv_val(const uint8_t *row, uint32_t d) {
    const uint8_t by = row[d >> 1]; const uint8_t nib = (d & 1u) ? (uint8_t)(by >> 4) : (uint8_t)(by & 0x0Fu);
    return bf16r_h(ds4_fp4_nibble_to_f32(nib) * ds4_e4m3fn_to_f32(row[DS4_V41_CKV_NIB + (d >> 4)]));
}
static void ref_row(const layer_bufs *b, uint32_t i, uint32_t h, double *out, float scale) {
    const uint32_t p = b->pos0 + i, lo = p + 1u > WINDOW ? p + 1u - WINDOW : 0u, nwin = p - lo + 1u, nkeys = nwin + TOPK;
    double *s = (double *)malloc((size_t)nkeys * 8); float *kv = (float *)malloc((size_t)nkeys * HD * 4);
    const float *qh = b->hq + ((size_t)i * NH + h) * HD;
    double m = -1e300;
    for (uint32_t k = 0; k < nkeys; k++) {
        float *kr = kv + (size_t)k * HD; bool ok = true;
        if (k < nwin) memcpy(kr, b->hkvw + v41_win_row((int64_t)lo + k, b->pos0, WINDOW, 1u) * HD, HD * 4);
        else { const int32_t g = b->hidx[(size_t)i * TOPK + (k - nwin)]; if (g < 0 || (uint32_t)g >= b->ng) ok = false;
               else for (uint32_t d = 0; d < HD; d++) kr[d] = ckv_val(b->hkvc + (size_t)g * DS4_V41_CKV_BYTES, d); }
        if (!ok) { s[k] = -1e300; memset(kr, 0, HD * 4); continue; }
        double acc = 0.0; for (uint32_t d = 0; d < HD; d++) acc += (double)qh[d] * (double)kr[d];
        s[k] = acc * (double)scale; if (s[k] > m) m = s[k];
    }
    double den = exp((double)b->hsink[h] - m);
    for (uint32_t d = 0; d < HD; d++) out[d] = 0.0;
    for (uint32_t k = 0; k < nkeys; k++) { if (s[k] < -1e299) continue; const double pv = exp(s[k] - m); den += pv;
                                            const float *kr = kv + (size_t)k * HD; for (uint32_t d = 0; d < HD; d++) out[d] += pv * (double)kr[d]; }
    for (uint32_t d = 0; d < HD; d++) out[d] /= den;
    free(s); free(kv);
}

/* ---- 变体表 ---- */
typedef struct { const char *name; uint32_t HB, KT; } variant;
static const variant VAR[] = { { "V0 引擎现核(16 头, 两遍, wmma)", 16u, 16u }, { "V1 FA 16 头 KT16", 16u, 16u }, { "V2 FA 32 头 KT16", 32u, 16u },
                               { "V3 FA 16 头 KT32", 16u, 32u }, { "V4 FA 32 头 KT32", 32u, 32u } };
static const int NV = 5;
static int g_nsm = 0, g_occ[5] = {0, 0, 0, 0, 0};
template <uint32_t HB, uint32_t KT>
static void launch_fa(int v, const layer_bufs *b, float scale) {
    const size_t smem = fa_smem_bytes<HB, KT>();
    if (!g_occ[v]) { CK(cudaFuncSetAttribute(attn_fa_kernel<HB, KT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
                     CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&g_occ[v], attn_fa_kernel<HB, KT>, 256, smem)); if (!g_occ[v]) g_occ[v] = -1; }
    attn_fa_kernel<HB, KT><<<dim3(b->n, NH / HB), 256, smem>>>(b->o, b->q, b->kvw, b->kvc, b->idx, b->sink, b->pos0, WINDOW, b->ng, TOPK, NH, scale, 0u);
}
static void run(int v, const layer_bufs *b, float scale) {
    if (v == 0) {
        static int set = 0; const size_t smem0 = ds4_attn_mma_smem_bytes();
        if (!set) { CK(cudaFuncSetAttribute(ds4_sparse_attn_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem0)); set = 1; }
        ds4_sparse_attn_mma_kernel<<<dim3(b->n, NH / 16u), 256, smem0>>>(b->o, b->q, b->kvw, b->kvc, b->idx, b->sink, b->pos0, WINDOW, b->ng, TOPK, NH, scale, 0u);
        return;
    }
    switch (v) { case 1: launch_fa<16u, 16u>(v, b, scale); break; case 2: launch_fa<32u, 16u>(v, b, scale); break;
                 case 3: launch_fa<16u, 32u>(v, b, scale); break; default: launch_fa<32u, 32u>(v, b, scale); break; }
}

int main(int argc, char **argv) {
    const uint32_t L = argc > 1 ? (uint32_t)atoi(argv[1]) : 8u, iters = argc > 2 ? (uint32_t)atoi(argv[2]) : 3u; const int only_v = argc > 3 ? atoi(argv[3]) : -1;
    { cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0)); g_nsm = pr.multiProcessorCount;
      printf("板子: %s, SM %d, shared/SM %zu KB, 每 block 可批 %zu KB, L2 %d MB\n", pr.name, g_nsm, pr.sharedMemPerMultiprocessor >> 10, pr.sharedMemPerBlockOptin >> 10, pr.l2CacheSize >> 20); }
    const float scale = 0.044194174f; const uint32_t pos0 = 12000u, ntoks[2] = { 512u, 2048u };
    printf("形状: 64 头 × 512, 窗口 %u, top-k %u, 压缩比 %u, pos0 %u; 层 %u, 遍 %u\n", WINDOW, TOPK, RATIO, pos0, L, iters);
    /* ① 精度门(层 0, n=512): V0 与各变体对 f64 参考, 抽 24 个 (query, 头) 行, 报最大相对误差(对行内最大绝对值) */
    layer_bufs g0 = make_layer(0u, 512u, pos0);
    float *ho = (float *)malloc((size_t)512u * NH * HD * 4); double *ref = (double *)malloc(HD * 8);
    int bad = 0; double e0 = 0.0;
    for (int v = 0; v < NV; v++) {
        CK(cudaMemset(g0.o, 0xFF, (size_t)512u * NH * HD * 4));
        run(v, &g0, scale); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(ho, g0.o, (size_t)512u * NH * HD * 4, cudaMemcpyDeviceToHost));
        double worst = 0.0;
        for (uint32_t s = 0; s < 24u; s++) {
            const uint32_t i = h32(9u, s, 0u, 0u) % 512u, h = h32(9u, s, 1u, 0u) % NH;
            ref_row(&g0, i, h, ref, scale);
            double mx = 1e-9; for (uint32_t d = 0; d < HD; d++) mx = fmax(mx, fabs(ref[d]));
            const float *row = ho + ((size_t)i * NH + h) * HD;
            for (uint32_t d = 0; d < HD; d++) { const double e = fabs((double)row[d] - ref[d]) / mx; if (e > worst) worst = e; }
        }
        if (v == 0) e0 = worst;
        const int ok = worst < 3e-2 && (v == 0 || worst < 4.0 * e0 + 1e-3);
        printf("  精度门 %-28s 对 f64 最大相对误差 %.2e %s\n", VAR[v].name, worst, ok ? "✓" : "★量级不对★"); bad |= !ok;
    }
    /* ② 计时: L 层各一发 = 一步; 512 与 2048 两档 */
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    for (int ni = 0; ni < 2; ni++) {
        const uint32_t n = ntoks[ni];
        layer_bufs *lb = (layer_bufs *)calloc(L, sizeof(layer_bufs));
        for (uint32_t l = 0; l < L; l++) lb[l] = make_layer(l, n, pos0);
        printf("\n查询块 %u: ms/层块(%u 层平均)              ms   blk/SM\n", n, L);
        for (int v = 0; v < NV; v++) {
            if (only_v >= 0 && v != only_v) continue;
            for (uint32_t l = 0; l < L; l++) run(v, &lb[l], scale);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            float best = 1e30f;
            for (uint32_t it = 0; it < iters; it++) {
                CK(cudaEventRecord(t0)); for (uint32_t l = 0; l < L; l++) run(v, &lb[l], scale); CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
                float ms; CK(cudaEventElapsedTime(&ms, t0, t1)); if (ms < best) best = ms;
            }
            printf("  %-32s %8.2f   %d\n", VAR[v].name, best / L, v == 0 ? 2 : g_occ[v]);
        }
        for (uint32_t l = 0; l < L; l++) { CK(cudaFree(lb[l].q)); CK(cudaFree(lb[l].kvw)); CK(cudaFree(lb[l].kvc)); CK(cudaFree(lb[l].idx)); CK(cudaFree(lb[l].sink)); CK(cudaFree(lb[l].o));
                                           free(lb[l].hq); free(lb[l].hkvw); free(lb[l].hsink); free(lb[l].hkvc); free(lb[l].hidx); }
        free(lb);
    }
    printf("\n%s\n", bad ? "★★精度门有红, 上面的数字不作数★★" : "精度门全绿(各核对 f64 参考同一量级)");
    return bad ? 1 : 0;
}
