/* v41_vq_persist_n_bench.cu — 投机验证批(n ≥ 2 行)的 VQ 专家常驻核(gateup)形态微基准(spark 本机跑; 2026-09-29 立)。
 *
 * 【为什么要它】钉 k 的账(09-29): 验证 2 行 43.9 ms / 4 行 61 ms, 纯解码走图 33.6 ⇒ 第一行多 10.3 ms、之后每行 8.7, 而字节账每行只该 3.9
 * (每多一行多 ~3.5 个唯一专家 = 0.9 GB)。差的 5~6 ms/行里专家核是最大嫌疑, 可它只有 n=1 的抄本(v41_vq_persist_bench), 验证批走的
 * 是另一个核(v41_vq_gu_persist_n_kernel: 工作项 = 唯一专家 × 行, 组 ≤ M=2 个 token, 激活走全局), 从没单独量过。
 *
 * 【比什么】核体 = 引擎 v41_vq_gu_persist_n_kernel<12,0,2,32> + v41_vq_stream<12,0,M> 的逐式抄本(组表/段切分/跨块流水/每 token 各一次
 * v41_vq_dot8_cw 都照抄, 去掉 PDL/格式校验); 选中表按真实重合度生成(每个新 token 的 6 个专家里 ~40% 与前面的 token 重复 ⇒ 唯一专家
 * n=1/2/4/6 ≈ 6/10/16.7/21.7, 与 [moe-uniq] 实测同档)。
 *   V0 = 引擎 n 行核; V1 = 激活先搬进 shared(n × 10 KB, 12 位码本 32 KB 之后; n=1 核就是这么做的); V2 = 不分组(M=1, 每对一个工作项,
 *   被几个 token 选中的专家读几遍 —— 回答"分组到底值多少")。参照 = n=1 引擎核(v41_vq_gu_persist_kernel)算 6 个专家。
 * ★判据★: 各变体每 (token, 行) 的 h 与 V0 逐位同; n 行核 n=1 时与 n=1 核逐位同(同一个 dot8 表达式树)。
 * 用法: nvcc -O3 -arch=native -o v41_vq_persist_n_bench v41_vq_persist_n_bench.cu && ./v41_vq_persist_n_bench [层数=80] [遍数=3] */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
#define WARPS 32u
#define K 6u           /* top-k */
#define MID 2304u
#define IN 5120u
#define NIDX (IN / 8u) /* 一行 640 个索引 = 20 轮 × 32 */
#define ROWB (NIDX * 12u / 8u)   /* 一行位流 960 B */
#define CBB (4096u * 8u)         /* 12 位层码本: 4096 词 × 8 个 E4M3 */
#define NMAX 6u
#define NUMAX (NMAX * K)         /* 一层最多的唯一专家数(全不重合) */
#define MAXP 64u                 /* = V41_VQPN_MAXP */
#include "v41_vq_persist_ts.cuh"   /* block/warp 收尾打点(10-07 晚): 量常驻核的尾巴 */

#include "v41_vq_persist_v0.cuh"   /* 引擎 V0 常驻核抄本 + stream_m + n=1 参照核(10-07 拆出守 500 行) */
#include "v41_vq_persist_mma.cuh"   /* 张量核形态(2026-10-07): gu_mma/dn_mma + 逐位参照 ref_mma */
#include "v41_vq_persist_dyn.cuh"   /* V0 的动态领工作形态(10-07 晚): 原子计数器按 U 行发放, 输出逐字节同 */

static uint32_t g_rng = 0x9e3779b9u;
static uint32_t rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }
/* bf16 / f32 数组比较: 返回最大 |a−b|, 数不同的个数(张量核版与 FMA 链版只能比"差多少", 逐位门是 ref_mma) */
static float bf16f(uint16_t v) { const uint32_t u = (uint32_t)v << 16; float f; memcpy(&f, &u, 4); return f; }
static double cmp_bf16(const uint16_t *a, const uint16_t *b, size_t n, size_t *ndiff) {
    double mx = 0; *ndiff = 0;
    for (size_t i = 0; i < n; i++) if (a[i] != b[i]) { const double d = fabs((double)bf16f(a[i]) - (double)bf16f(b[i])); if (d > mx) mx = d; (*ndiff)++; }
    return mx;
}
static double cmp_f32(const float *a, const float *b, size_t n, size_t *ndiff) {
    double mx = 0; *ndiff = 0;
    for (size_t i = 0; i < n; i++) if (a[i] != b[i]) { const double d = fabs((double)a[i] - (double)b[i]); if (d > mx) mx = d; (*ndiff)++; }
    return mx;
}

/* 一层的选中表: token 0 取 6 个新专家; 之后每个 token 的每一格 40% 重复前面某个 token 用过的(且本 token 没选过), 否则新专家。
 * 唯一专家数 n=1/2/4/6 ≈ 6/10/16/21, 与引擎 [moe-uniq] 实测(6/9.96/16.66/21.7)同档。返回唯一专家数; order = 对按专家号稳定排序。 */
static uint32_t make_sel(int32_t *sel, int32_t *order, uint32_t n) {
    uint32_t nu = 0;
    for (uint32_t t = 0; t < n; t++) for (uint32_t k = 0; k < K; k++) {
        int32_t e = -1;
        if (t > 0 && (rnd() % 100u) < 40u) {
            for (int tries = 0; tries < 16 && e < 0; tries++) {
                const int32_t cand = sel[rnd() % (t * K)];
                bool dup = false; for (uint32_t j = 0; j < k; j++) dup |= sel[t * K + j] == cand;
                if (!dup) e = cand;
            }
        }
        if (e < 0) e = (int32_t)nu++;
        sel[t * K + k] = e;
    }
    const uint32_t np = n * K;
    for (uint32_t i = 0; i < np; i++) order[i] = (int32_t)i;
    for (uint32_t a = 1; a < np; a++) { int32_t v = order[a]; uint32_t b = a; while (b > 0 && sel[order[b - 1]] > sel[v]) { order[b] = order[b - 1]; b--; } order[b] = v; }
    return nu;
}

int main(int argc, char **argv) {
    const int layers = argc > 1 ? atoi(argv[1]) : 80, reps = argc > 2 ? atoi(argv[2]) : 3;
    /* 第 3 参数(10-07 晚) = V0 计时发的动态 shared KB(默认 32 = 12 位码本; 给 64 = 模拟 13 位层把 L1 挤到 36 KB, 码本内容不变) */
    const uint32_t v0sh = argc > 3 ? (uint32_t)atoi(argv[3]) * 1024u : CBB;
    int nsm = 0; CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    const uint64_t mat_stride = (uint64_t)MID * 2u + (uint64_t)MID * ROWB;   /* 行增益 + 位流 */
    const uint64_t layer_bytes = mat_stride * 2u * NUMAX;                       /* 每层按最多唯一专家数预留 */
    const uint64_t total = layer_bytes * (uint64_t)layers + 4096u;
    uint8_t *buf = NULL, *cb = NULL; uint32_t *x = NULL; uint16_t *h = NULL, *h2 = NULL;
    CK(cudaMalloc(&buf, total)); CK(cudaMalloc(&cb, CBB)); CK(cudaMalloc(&x, (size_t)NMAX * IN * 2u));
    CK(cudaMalloc(&h, (size_t)NMAX * K * MID * 2u)); CK(cudaMalloc(&h2, (size_t)NMAX * K * MID * 2u));
    {   /* 随机位流/增益(f16 ~1)/码本(E4M3 去掉 NaN 码)/激活(bf16 ~±1) */
        const size_t chunk = 64u << 20; uint32_t *hb = (uint32_t *)malloc(chunk);
        for (uint64_t off = 0; off < total; off += chunk) {
            const size_t nb = total - off < chunk ? (size_t)(total - off) : chunk;
            for (size_t i = 0; i < nb / 4; i++) hb[i] = rnd();
            CK(cudaMemcpy(buf + off, hb, nb & ~(size_t)3, cudaMemcpyHostToDevice));
        }
        for (int L = 0; L < layers; L++) for (uint32_t m = 0; m < 2u * NUMAX; m++) {
            uint16_t g[MID]; for (uint32_t i = 0; i < MID; i++) g[i] = 0x3c00u;
            CK(cudaMemcpy(buf + (uint64_t)L * layer_bytes + m * mat_stride, g, sizeof g, cudaMemcpyHostToDevice));
        }
        uint8_t hc[CBB]; for (uint32_t i = 0; i < CBB; i++) { uint8_t b = (uint8_t)rnd(); if ((b & 0x7f) == 0x7f) b &= 0xfe; hc[i] = b; }
        CK(cudaMemcpy(cb, hc, CBB, cudaMemcpyHostToDevice));
        uint16_t *hx = (uint16_t *)malloc((size_t)NMAX * IN * 2u);
        for (uint32_t i = 0; i < NMAX * IN; i++) hx[i] = (uint16_t)(0x3f00u | (rnd() & 0x80ffu));
        CK(cudaMemcpy(x, hx, (size_t)NMAX * IN * 2u, cudaMemcpyHostToDevice));
        free(hb); free(hx);
    }
    /* down 矩阵: 每层 NUMAX 个, 每个 行增益 f16[OUT] + 位流 OUT × 432 B; h(bf16 [NMAX·K][MID]) 与 partial(f32 [NMAX·K][OUT]) */
    const uint64_t dn_stride = (uint64_t)OUT * 2u + (uint64_t)OUT * ROWB_DN, dn_layer = dn_stride * NUMAX, dn_total = dn_layer * (uint64_t)layers + 4096u;
    uint8_t *dbuf = NULL; uint32_t *hh = NULL; float *part = NULL;
    CK(cudaMalloc(&dbuf, dn_total)); CK(cudaMalloc(&hh, (size_t)NMAX * K * MID * 2u)); CK(cudaMalloc(&part, (size_t)NMAX * K * OUT * 4u));
    {
        const size_t chunk = 64u << 20; uint32_t *hb = (uint32_t *)malloc(chunk);
        for (uint64_t off = 0; off < dn_total; off += chunk) {
            const size_t nb = dn_total - off < chunk ? (size_t)(dn_total - off) : chunk;
            for (size_t i = 0; i < nb / 4; i++) hb[i] = rnd();
            CK(cudaMemcpy(dbuf + off, hb, nb & ~(size_t)3, cudaMemcpyHostToDevice));
        }
        uint16_t *g = (uint16_t *)malloc((size_t)OUT * 2u); for (uint32_t i = 0; i < OUT; i++) g[i] = 0x3c00u;
        for (int L = 0; L < layers; L++) for (uint32_t m = 0; m < NUMAX; m++) CK(cudaMemcpy(dbuf + (uint64_t)L * dn_layer + m * dn_stride, g, (size_t)OUT * 2u, cudaMemcpyHostToDevice));
        uint16_t *hx = (uint16_t *)malloc((size_t)NMAX * K * MID * 2u);
        for (uint32_t i = 0; i < NMAX * K * MID; i++) hx[i] = (uint16_t)(0x3f00u | (rnd() & 0x80ffu));
        CK(cudaMemcpy(hh, hx, (size_t)NMAX * K * MID * 2u, cudaMemcpyHostToDevice));
        free(hb); free(g); free(hx);
    }
    dmats_t *ddtab = NULL; CK(cudaMalloc(&ddtab, sizeof(dmats_t) * (size_t)layers));
    dmats_t *dtab_h = (dmats_t *)malloc(sizeof(dmats_t) * (size_t)layers);
    /* 张量核形态(10-07)要的: 64 KB 码本(13 位层 8192 词, 前 4096 词与 cb 同 = 12 位层码本)、位平面(每矩阵 rows × NI/8 B, 随机位)、位平面表 */
    uint8_t *cb64 = NULL, *exgu = NULL, *exdn = NULL; exts_t *dext = NULL;
    const uint64_t exg_mat = (uint64_t)MID * (NIDX / 8u), exd_mat = (uint64_t)OUT * (NIDX_DN / 8u);
    const uint64_t exg_total = exg_mat * 2u * NUMAX * (uint64_t)layers + 4096u, exd_total = exd_mat * NUMAX * (uint64_t)layers + 4096u;
    CK(cudaMalloc(&cb64, 65536u)); CK(cudaMalloc(&exgu, exg_total)); CK(cudaMalloc(&exdn, exd_total)); CK(cudaMalloc(&dext, sizeof(exts_t) * (size_t)layers));
    {
        uint8_t hc[65536]; for (uint32_t i = 0; i < 65536u; i++) { uint8_t b = (uint8_t)rnd(); if ((b & 0x7f) == 0x7f) b &= 0xfe; hc[i] = b; }
        CK(cudaMemcpy(hc, cb, CBB, cudaMemcpyDeviceToHost));   /* 前 32 KB = 12 位码本原样, 后 32 KB 随机 */
        for (uint32_t i = CBB; i < 65536u; i++) { uint8_t b = (uint8_t)rnd(); if ((b & 0x7f) == 0x7f) b &= 0xfe; hc[i] = b; }
        CK(cudaMemcpy(cb64, hc, 65536u, cudaMemcpyHostToDevice));
        const size_t chunk = 64u << 20; uint32_t *hb = (uint32_t *)malloc(chunk);
        for (int which = 0; which < 2; which++) {
            uint8_t *dst = which ? exdn : exgu; const uint64_t tot = which ? exd_total : exg_total;
            for (uint64_t off = 0; off < tot; off += chunk) {
                const size_t nb = tot - off < chunk ? (size_t)(tot - off) : chunk;
                for (size_t i = 0; i < nb / 4; i++) hb[i] = rnd();
                CK(cudaMemcpy(dst + off, hb, nb & ~(size_t)3, cudaMemcpyHostToDevice));
            }
        }
        free(hb);
        exts_t *hx = (exts_t *)malloc(sizeof(exts_t) * (size_t)layers);
        for (int L = 0; L < layers; L++) for (uint32_t e = 0; e < NUMAX; e++) {
            hx[L].g[e] = exgu + ((uint64_t)L * 2u * NUMAX + 2u * e) * exg_mat; hx[L].u[e] = hx[L].g[e] + exg_mat;
            hx[L].d[e] = exdn + ((uint64_t)L * NUMAX + e) * exd_mat;
        }
        CK(cudaMemcpy(dext, hx, sizeof(exts_t) * (size_t)layers, cudaMemcpyHostToDevice)); free(hx);
    }
    CK(cudaFuncSetAttribute(gu_mma<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
    CK(cudaFuncSetAttribute(gu_mma<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
    CK(cudaFuncSetAttribute(dn_mma<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
    CK(cudaFuncSetAttribute(dn_mma<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
    CK(cudaFuncSetAttribute(ref_mma<0, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
    CK(cudaFuncSetAttribute(ref_mma<1, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
    CK(cudaFuncSetAttribute(ref_mma<0, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
    CK(cudaFuncSetAttribute(ref_mma<1, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536));
    float *part2 = NULL; CK(cudaMalloc(&part2, (size_t)NMAX * K * OUT * 4u));
    float *hp1 = (float *)malloc((size_t)NMAX * K * OUT * 4u), *hp2 = (float *)malloc((size_t)NMAX * K * OUT * 4u);
    CK(cudaFuncSetAttribute(dn_persist_n<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(v0sh > CBB ? v0sh : CBB)));
    const uint32_t shm1 = CBB + IN * 2u;
    CK(cudaFuncSetAttribute(gu_persist1, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm1));
    CK(cudaFuncSetAttribute(gu_persist_n<2, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(v0sh > CBB ? v0sh : CBB)));
    CK(cudaFuncSetAttribute(gu_persist_n<1, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)CBB));
    CK(cudaFuncSetAttribute(gu_persist_n<2, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(CBB + NMAX * IN * 2u)));
    /* 每层: 选中表 + 唯一专家的矩阵表(专家 e 的 gate = 层基 + 2e·stride, up = +(2e+1)·stride) */
    mats_t *dtab = NULL; int32_t *dsel = NULL, *dord = NULL;
    CK(cudaMalloc(&dtab, sizeof(mats_t) * (size_t)layers)); CK(cudaMalloc(&dsel, (size_t)layers * MAXP * 4)); CK(cudaMalloc(&dord, (size_t)layers * MAXP * 4));
    mats_t *tab = (mats_t *)malloc(sizeof(mats_t) * (size_t)layers);
    int32_t *hsel = (int32_t *)malloc((size_t)layers * MAXP * 4), *hord = (int32_t *)malloc((size_t)layers * MAXP * 4);
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    uint16_t *ho = (uint16_t *)malloc((size_t)NMAX * K * MID * 2u), *ho2 = (uint16_t *)malloc((size_t)NMAX * K * MID * 2u);
    printf("v3 gateup 验证批常驻核抄本: %d SM × 1024 线程, 每层唯一专家 × 2 × %u 行 × %u B, %d 层, %d 遍\n", nsm, MID, ROWB, layers, reps);
    const uint32_t ns[4] = { 1u, 2u, 4u, 6u };
    for (uint32_t ni = 0; ni < 4; ni++) {
        const uint32_t n = ns[ni], np = n * K;
        double nu_sum = 0;
        for (int L = 0; L < layers; L++) {
            const uint32_t nu = make_sel(hsel + (size_t)L * MAXP, hord + (size_t)L * MAXP, n); nu_sum += nu;
            for (uint32_t e = 0; e < NUMAX; e++) { tab[L].g[e] = buf + (uint64_t)L * layer_bytes + (2u * e) * mat_stride; tab[L].u[e] = tab[L].g[e] + mat_stride; }
        }
        CK(cudaMemcpy(dtab, tab, sizeof(mats_t) * (size_t)layers, cudaMemcpyHostToDevice));
        for (int L = 0; L < layers; L++) for (uint32_t e = 0; e < NUMAX; e++) dtab_h[L].d[e] = dbuf + (uint64_t)L * dn_layer + e * dn_stride;
        CK(cudaMemcpy(ddtab, dtab_h, sizeof(dmats_t) * (size_t)layers, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsel, hsel, (size_t)layers * MAXP * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dord, hord, (size_t)layers * MAXP * 4, cudaMemcpyHostToDevice));
        const double nu_avg = nu_sum / layers, lay_mb = nu_avg * 2.0 * MID * ROWB / 1e6;
        auto launch = [&](int v, int L, uint16_t *dst) {
            const int32_t *s = dsel + (size_t)L * MAXP, *o = dord + (size_t)L * MAXP;
            switch (v) {
            case 0: gu_persist_n<2, 0><<<nsm, 1024, v0sh>>>(dst, dtab + L, s, o, x, np, n, cb); break;
            case 1: gu_persist_n<2, 1><<<nsm, 1024, CBB + n * IN * 2u>>>(dst, dtab + L, s, o, x, np, n, cb); break;
            case 2: gu_persist_n<1, 0><<<nsm, 1024, CBB>>>(dst, dtab + L, s, o, x, np, n, cb); break;
            default: gu_persist1<<<nsm, 1024, shm1>>>(dst, dtab + L, s, x, np, cb); break;   /* 只在 n=1 有意义 */
            }
        };
        const char *vname[4] = { "V0 引擎 n 行核", "V1 激活进 shared", "V2 不分组(M=1)", "参照 n=1 核" };
        const int nv = n == 1u ? 4 : 3;
        printf("== n=%u: 每层唯一专家 %.2f / %u 对, 位流 %.1f MB/层\n", n, nu_avg, np, lay_mb);
        /* 逐位门(层 0): 各变体 vs V0 */
        launch(0, 0, h); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(ho, h, (size_t)np * MID * 2u, cudaMemcpyDeviceToHost));
        for (int v = 1; v < nv; v++) {
            CK(cudaMemset(h2, 0xFF, (size_t)np * MID * 2u)); launch(v, 0, h2); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(ho2, h2, (size_t)np * MID * 2u, cudaMemcpyDeviceToHost));
            printf("   %-16s vs V0: %s\n", vname[v], memcmp(ho, ho2, (size_t)np * MID * 2u) == 0 ? "逐位同 ✓" : "★不同★");
        }
        for (int rep = 0; rep < reps; rep++) {
            printf("   第 %d 遍:", rep + 1);
            for (int v = 0; v < nv; v++) {
                for (int L = 0; L < 4; L++) launch(v, L, h); CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(e0)); for (int L = 0; L < layers; L++) launch(v, L, h); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                float ms = 0.f; CK(cudaEventElapsedTime(&ms, e0, e1)); const double us = (double)ms * 1e3 / layers;
                printf("  %s %6.1f µs/层 %5.0f GB/s(唯一字节)", vname[v], us, lay_mb / us * 1e3);
            }
            printf("\n");
        }
        {   /* down 核(10-07): 同一批选中表, 唯一专家 × OUT 行, 位流 432 B/行; 看它孤立时贴不贴墙(引擎独占段 n=4 只 160 GB/s, gu 245) */
            const double dn_mb = nu_avg * OUT * ROWB_DN / 1e6;
            for (int L = 0; L < 4; L++) dn_persist_n<2><<<nsm, 1024, v0sh>>>(part, ddtab + L, dsel + (size_t)L * MAXP, dord + (size_t)L * MAXP, hh, np, cb);
            CK(cudaDeviceSynchronize());
            for (int rep = 0; rep < reps; rep++) {
                CK(cudaEventRecord(e0));
                for (int L = 0; L < layers; L++) dn_persist_n<2><<<nsm, 1024, v0sh>>>(part, ddtab + L, dsel + (size_t)L * MAXP, dord + (size_t)L * MAXP, hh, np, cb);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
                float ms = 0.f; CK(cudaEventElapsedTime(&ms, e0, e1)); const double us = (double)ms * 1e3 / layers;
                printf("   down n 行核 第 %d 遍: %6.1f µs/层 %5.0f GB/s(唯一字节 %.1f MB/层)\n", rep + 1, us, dn_mb / us * 1e3, dn_mb);
            }
        }
        {   /* 张量核形态(10-07): 逐位门(vs ref_mma, 同一 mma 的直白实现) + 与 V0(FMA 链)的差 + 计时; 12 位层(bf16 码本)/13 位层(E4M3 + 位平面)各一份 */
            const double dn_mb = nu_avg * OUT * ROWB_DN / 1e6;
            const uint32_t ntg = MID / 16u, ntd = OUT / 16u;
            const int32_t *s0 = dsel, *o0 = dord;
            uint16_t *ho3 = (uint16_t *)malloc((size_t)np * MID * 2u);
            size_t nd; double mx;
            {   /* 动态领工作(层 0 逐位门 + 计时 + 尾巴), U = 2/4/8 行 */
                static uint32_t *dctr = NULL; if (!dctr) CK(cudaMalloc(&dctr, 4));
                launch(0, 0, h); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(ho, h, (size_t)np * MID * 2u, cudaMemcpyDeviceToHost));
                dn_persist_n<2><<<nsm, 1024, CBB>>>(part, ddtab, s0, o0, hh, np, cb); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hp1, part, (size_t)np * OUT * 4u, cudaMemcpyDeviceToHost));
                #define DYN_GO(UU) do { \
                    CK(cudaMemset(dctr, 0, 4)); CK(cudaMemset(h2, 0xFF, (size_t)np * MID * 2u)); \
                    gu_persist_dyn<2, UU><<<nsm, 1024, CBB>>>(h2, dtab, s0, o0, x, np, cb, dctr); CK(cudaDeviceSynchronize()); \
                    CK(cudaMemcpy(ho2, h2, (size_t)np * MID * 2u, cudaMemcpyDeviceToHost)); \
                    CK(cudaMemset(dctr, 0, 4)); CK(cudaMemset(part2, 0xFF, (size_t)np * OUT * 4u)); \
                    dn_persist_dyn<2, UU><<<nsm, 1024, CBB>>>(part2, ddtab, s0, o0, hh, np, cb, dctr); CK(cudaDeviceSynchronize()); \
                    CK(cudaMemcpy(hp2, part2, (size_t)np * OUT * 4u, cudaMemcpyDeviceToHost)); \
                    printf("   动态 U=%u: gu %s / dn %s", (unsigned)UU, memcmp(ho, ho2, (size_t)np * MID * 2u) == 0 ? "逐位同 ✓" : "★不同★", memcmp(hp1, hp2, (size_t)np * OUT * 4u) == 0 ? "逐位同 ✓" : "★不同★"); \
                    for (int rep = 0; rep < reps; rep++) { float ms = 0.f; \
                        for (int L = 0; L < 4; L++) { CK(cudaMemsetAsync(dctr, 0, 4)); gu_persist_dyn<2, UU><<<nsm, 1024, CBB>>>(h, dtab + L, dsel + (size_t)L * MAXP, dord + (size_t)L * MAXP, x, np, cb, dctr); } \
                        CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0)); \
                        for (int L = 0; L < layers; L++) { CK(cudaMemsetAsync(dctr, 0, 4)); gu_persist_dyn<2, UU><<<nsm, 1024, CBB>>>(h, dtab + L, dsel + (size_t)L * MAXP, dord + (size_t)L * MAXP, x, np, cb, dctr); } \
                        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1)); \
                        const double us = (double)ms * 1e3 / layers; printf("  | gu %6.1f µs/层 %5.0f GB/s", us, lay_mb / us * 1e3); \
                        for (int L = 0; L < 4; L++) { CK(cudaMemsetAsync(dctr, 0, 4)); dn_persist_dyn<2, UU><<<nsm, 1024, CBB>>>(part, ddtab + L, dsel + (size_t)L * MAXP, dord + (size_t)L * MAXP, hh, np, cb, dctr); } \
                        CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0)); \
                        for (int L = 0; L < layers; L++) { CK(cudaMemsetAsync(dctr, 0, 4)); dn_persist_dyn<2, UU><<<nsm, 1024, CBB>>>(part, ddtab + L, dsel + (size_t)L * MAXP, dord + (size_t)L * MAXP, hh, np, cb, dctr); } \
                        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1)); \
                        const double us2 = (double)ms * 1e3 / layers; printf("  dn %6.1f µs/层 %5.0f GB/s", us2, dn_mb / us2 * 1e3); } \
                    printf("\n"); \
                    ts_set(1); CK(cudaMemset(dctr, 0, 4)); gu_persist_dyn<2, UU><<<nsm, 1024, CBB>>>(h2, dtab, s0, o0, x, np, cb, dctr); CK(cudaDeviceSynchronize()); ts_report("dyn gu U=" #UU, nsm, (int)WARPS); \
                    CK(cudaMemset(dctr, 0, 4)); dn_persist_dyn<2, UU><<<nsm, 1024, CBB>>>(part2, ddtab, s0, o0, hh, np, cb, dctr); CK(cudaDeviceSynchronize()); ts_report("dyn dn U=" #UU, nsm, (int)WARPS); ts_set(0); \
                } while (0)
                DYN_GO(1u); DYN_GO(2u); DYN_GO(4u);
                #undef DYN_GO
            }
            {   /* 尾巴账(层 0, 各一发): block/warp 收尾时刻离散度 */
                ts_set(1);
                launch(0, 0, h); CK(cudaDeviceSynchronize()); ts_report("V0 gu(M=2)", nsm, (int)WARPS); ts_report_bym("V0 gu(M=2)", nsm, (int)WARPS, hsel, hord, np, MID, 2u);
                dn_persist_n<2><<<nsm, 1024, CBB>>>(part, ddtab, s0, o0, hh, np, cb); CK(cudaDeviceSynchronize()); ts_report("V0 dn(M=2)", nsm, (int)WARPS); ts_report_bym("V0 dn(M=2)", nsm, (int)WARPS, hsel, hord, np, OUT, 2u);
                gu_mma<0><<<nsm, TM_NW * 32u, 65536>>>(h2, dtab, dext, s0, o0, x, np, cb64); CK(cudaDeviceSynchronize()); ts_report("gu_mma<0>", nsm, (int)TM_NW); ts_report_bym("gu_mma<0>", nsm, (int)TM_NW, hsel, hord, np, MID, 8u);
                dn_mma<0><<<nsm, TM_NW * 32u, 65536>>>(part2, ddtab, dext, s0, o0, hh, np, cb64); CK(cudaDeviceSynchronize()); ts_report("dn_mma<0>", nsm, (int)TM_NW); ts_report_bym("dn_mma<0>", nsm, (int)TM_NW, hsel, hord, np, OUT, 8u);
                ts_set(0);
            }
            for (int ext = 0; ext < 2; ext++) {
                /* gateup: 层 0 */
                CK(cudaMemset(h2, 0xFF, (size_t)np * MID * 2u)); CK(cudaMemset(h, 0xFF, (size_t)np * MID * 2u));
                if (ext) { gu_mma<1><<<nsm, TM_NW * 32u, 65536>>>(h2, dtab, dext, s0, o0, x, np, cb64); ref_mma<1, 0><<<np * ntg, 32, 65536>>>(h, NULL, dtab, ddtab, dext, s0, o0, x, np, cb64); }
                else     { gu_mma<0><<<nsm, TM_NW * 32u, 65536>>>(h2, dtab, dext, s0, o0, x, np, cb64); ref_mma<0, 0><<<np * ntg, 32, 65536>>>(h, NULL, dtab, ddtab, dext, s0, o0, x, np, cb64); }
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(ho2, h2, (size_t)np * MID * 2u, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ho3, h, (size_t)np * MID * 2u, cudaMemcpyDeviceToHost));
                mx = cmp_bf16(ho2, ho3, (size_t)np * MID, &nd);
                printf("   gu_mma<%d> vs ref_mma: %s(不同 %zu, 最大差 %.3g)", ext, nd == 0 ? "逐位同 ✓" : "★不同★", nd, mx);
                if (!ext) { mx = cmp_bf16(ho2, ho, (size_t)np * MID, &nd); printf("; vs V0(FMA 链): 不同 %zu/%zu, 最大差 %.3g", nd, (size_t)np * MID, mx); }
                printf("\n");
                /* down: 层 0(V0 的 partial 现算) */
                dn_persist_n<2><<<nsm, 1024, CBB>>>(part, ddtab, s0, o0, hh, np, cb); CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(hp1, part, (size_t)np * OUT * 4u, cudaMemcpyDeviceToHost));
                CK(cudaMemset(part2, 0xFF, (size_t)np * OUT * 4u)); CK(cudaMemset(part, 0xFF, (size_t)np * OUT * 4u));
                if (ext) { dn_mma<1><<<nsm, TM_NW * 32u, 65536>>>(part2, ddtab, dext, s0, o0, hh, np, cb64); ref_mma<1, 2><<<np * ntd, 32, 65536>>>(NULL, part, dtab, ddtab, dext, s0, o0, hh, np, cb64); }
                else     { dn_mma<0><<<nsm, TM_NW * 32u, 65536>>>(part2, ddtab, dext, s0, o0, hh, np, cb64); ref_mma<0, 2><<<np * ntd, 32, 65536>>>(NULL, part, dtab, ddtab, dext, s0, o0, hh, np, cb64); }
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(hp2, part2, (size_t)np * OUT * 4u, cudaMemcpyDeviceToHost));
                if (!ext) { mx = cmp_f32(hp2, hp1, (size_t)np * OUT, &nd); printf("   dn_mma<0> vs V0(FMA 链): 不同 %zu/%zu, 最大差 %.3g; ", nd, (size_t)np * OUT, mx); } else printf("   ");
                CK(cudaMemcpy(hp1, part, (size_t)np * OUT * 4u, cudaMemcpyDeviceToHost));
                mx = cmp_f32(hp2, hp1, (size_t)np * OUT, &nd);
                printf("dn_mma<%d> vs ref_mma: %s(不同 %zu, 最大差 %.3g)\n", ext, nd == 0 ? "逐位同 ✓" : "★不同★", nd, mx);
                /* 计时 */
                for (int rep = 0; rep < reps; rep++) {
                    float ms = 0.f;
                    for (int L = 0; L < 4; L++) { if (ext) gu_mma<1><<<nsm, TM_NW * 32u, 65536>>>(h, dtab + L, dext + L, dsel + (size_t)L * MAXP, dord + (size_t)L * MAXP, x, np, cb64);
                                                  else gu_mma<0><<<nsm, TM_NW * 32u, 65536>>>(h, dtab + L, dext + L, dsel + (size_t)L * MAXP, dord + (size_t)L * MAXP, x, np, cb64); }
                    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0));
                    for (int L = 0; L < layers; L++) { if (ext) gu_mma<1><<<nsm, TM_NW * 32u, 65536>>>(h, dtab + L, dext + L, dsel + (size_t)L * MAXP, dord + (size_t)L * MAXP, x, np, cb64);
                                                       else gu_mma<0><<<nsm, TM_NW * 32u, 65536>>>(h, dtab + L, dext + L, dsel + (size_t)L * MAXP, dord + (size_t)L * MAXP, x, np, cb64); }
                    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1));
                    const double us = (double)ms * 1e3 / layers;
                    printf("   第 %d 遍 %s: gu_mma %6.1f µs/层 %5.0f GB/s", rep + 1, ext ? "13 位层(E4M3 码本+位平面)" : "12 位层(bf16 码本)    ", us, lay_mb / us * 1e3);
                    for (int L = 0; L < 4; L++) { if (ext) dn_mma<1><<<nsm, TM_NW * 32u, 65536>>>(part, ddtab + L, dext + L, dsel + (size_t)L * MAXP, dord + (size_t)L * MAXP, hh, np, cb64);
                                                  else dn_mma<0><<<nsm, TM_NW * 32u, 65536>>>(part, ddtab + L, dext + L, dsel + (size_t)L * MAXP, dord + (size_t)L * MAXP, hh, np, cb64); }
                    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(e0));
                    for (int L = 0; L < layers; L++) { if (ext) dn_mma<1><<<nsm, TM_NW * 32u, 65536>>>(part, ddtab + L, dext + L, dsel + (size_t)L * MAXP, dord + (size_t)L * MAXP, hh, np, cb64);
                                                       else dn_mma<0><<<nsm, TM_NW * 32u, 65536>>>(part, ddtab + L, dext + L, dsel + (size_t)L * MAXP, dord + (size_t)L * MAXP, hh, np, cb64); }
                    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1));
                    const double us2 = (double)ms * 1e3 / layers;
                    printf("  dn_mma %6.1f µs/层 %5.0f GB/s\n", us2, dn_mb / us2 * 1e3);
                }
            }
            free(ho3);
        }
    }
    return 0;
}
