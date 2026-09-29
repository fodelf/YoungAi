/* v41_vq_prefill_mma_bench.cu — V4.1 预填专家张量核(vqm_kernel, cuda_vq_prefill_mma.inc.cu)的形态微基准(spark 本机跑; 2026-09-29 立)。
 *
 * 【为什么要它】09-29 12k 预填逐核表: 块 512 时专家三发 10.4 s = 51%; 块 2048 时 6.4 s = 37% —— 块开大 4 倍只快 1.6 倍。
 * 账: 一块要把本层 384 个专家的位流全解一遍(2.55 GB/层), 块 512 每专家平均只有 8 个 token 摊这次解码; 块 2048 平均 32 个,
 * 可现核一个工作项最多 32 个 token(VQM_BN), 超过就再开一项 = 再解一遍(平均 1.45 项/专家); 而且现核 1 block/SM(8 warp),
 * 解码(ALU)与 mma(张量管)按 __syncthreads 分相, 谁忙另一个就闲。字节墙 2.55 GB/235 GB/s = 10.9 ms/层块, 算力墙
 * 868 GFLOP/90 TFLOPS = 9.6 ms(块 2048), 现核 40~45 ms —— 有 3~4 倍在桌上。换核形态先在这里过, 过了才进引擎(一趟要装 113 GB)。
 *
 * 【比什么】V0 = 引擎现核逐字抄本(BN32 BK64 单组 8 warp 码本 E4M3 进 shared, grid = SM 数);
 *   变体核 vqm_v_kernel<EXT, MODE, BN, BK, TW, CB>:
 *     BN = 一个工作项最多几个 token(32/64/128; 越大, 专家的 token 越不用拆项重解);
 *     BK = 一轮沿 K 走几列(64 = 现核; 32 = A 片减半 ⇒ 12 位层 shared 42 KB, 每 SM 挂 2 个 block);
 *     TW = token 分几组给不同 warp(1 = 8 warp 各管 16 行 × 全部 token; 2 = 16 warp, 每 warp 16 行 × 一半 token, 解码线程翻倍);
 *     CB = 码本放哪: 0 = E4M3 进 shared 现场转 bf16(现核); 1 = 进 shared 时就转成 bf16(nc×16 B, 只 12 位层放得下);
 *          2 = 每层一发小核把码本转成 bf16 放全局(L2 常驻), shared 只留片 ⇒ 13 位层也能每 SM 挂 4 个 block。
 *   grid 一律按占用率 API 算(SM 数 × 每 SM 能挂几个 block), 不写死。
 * ★判据★: 每个变体的 g32/h16/ys 三份输出与 V0 **逐位同**(同一组 bf16 乘积、同一个 k16 累加序、同一个出口舍入点; 只换谁去算)。
 *   另有一道对 CPU double 参考的松门(相对误差), 只为证明合成载荷的布局与引擎解析一致 —— 布局错了所有变体会一起错得"逐位同"。
 * 形状 = V4.1 Flash: IN 5120 / MID 2304 / OUT 5120 / 384 专家 top-6; 12 位层(nc 4096)与 13 位层(nc 8192 + 位平面)各一份合成层
 *   (位流/码本按哈希生成, 2.55 / 2.76 GB, 远超 24 MB L2 ⇒ 每次都是真读)。token 数 512/2048/4096 三档(路由按哈希, 每 token 6 个不同专家)。
 * ★真载荷档★(第 4~6 个参数): 给 GGUF 路径 + 12 位层号(默认 20) + 13 位层号(默认 5), 就用盘上真实的 blk.L.ffn_exps_vq.blob 代替合成层
 *   (按引擎 cuda_vq_align 的口径挪到位流 128 B 对齐)。为什么要它: 合成层(均匀随机码字)上 V0 的 12 位 512 token 30 ms/层块, 引擎 nsys 同形状
 *   17.7 ms —— 差 1.7 倍; 真载荷上对得上引擎的数, 变体的名次才作数。
 * 用法: nvcc -O3 -arch=native -std=c++17 -o v41_vq_prefill_mma_bench v41_vq_prefill_mma_bench.cu ../../src/common/ds4_gguf.c
 *       ./v41_vq_prefill_mma_bench [遍数=3] [只计时变体=-1] [只 token 档=0] [GGUF] [12 位层=20] [13 位层=5] [真路由文件] */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../../src/common/ds4_fp8.h"
extern "C" {
#include "../../src/common/ds4_gguf.h"
}
#include "v41_vq_prefill_mma_bench_kernels.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "★CUDA %s @%d: %s★\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

#include "v41_vq_prefill_mma_bench_variants.cuh"
/* ---- 合成一层 v3 blob(布局见 cuda_vq_row.inc.cu 头注; 位流起点按引擎 cuda_vq_align 的口径 128 B 对齐) ---- */
static uint32_t h32(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t x = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA6Bu ^ (c + 0x165667B1u) * 0xC2B2AE35u;
    x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15; return x;
}
__global__ static void fill_kernel(uint32_t *dst, uint64_t nwords, uint32_t seed) {   /* 位流/位平面: 哈希字节 */
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nwords) return;
    uint32_t x = (uint32_t)i * 0x9E3779B1u ^ (seed + 0x7F4A7C15u) * 0x85EBCA6Bu ^ (uint32_t)(i >> 32) * 0xC2B2AE35u;
    x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15;
    dst[i] = x;
}
typedef struct { uint8_t *d; uint64_t bytes; uint32_t nc, nbit; uint64_t cb_off; uint4 *cbh; } layer_blob;
static const uint32_t NE = 384u, KU = 6u, IN = 5120u, MID = 2304u, OUT = 5120u;
static uint64_t align128(uint64_t x) { return (x + 127u) & ~(uint64_t)127u; }
static layer_blob make_layer(uint32_t nc, uint32_t seed) {
    layer_blob L; memset(&L, 0, sizeof L);
    L.nc = nc; L.nbit = 0; while ((1u << L.nbit) < nc) L.nbit++;
    const int ext = L.nbit > 12u;
    uint64_t cur = 16u + (uint64_t)NE * 3u * 8u;
    L.cb_off = align128(cur); cur = L.cb_off + (uint64_t)nc * 8u;
    uint64_t *tab = (uint64_t *)calloc((size_t)NE * 3u, 8);
    for (uint32_t e = 0; e < NE; e++) for (uint32_t w = 0; w < 3u; w++) {
        const uint32_t rows = w == 2u ? OUT : MID, cols = w == 2u ? MID : IN, nidx = cols / 8u;
        const uint32_t mrow = (nidx * 12u + 7u) / 8u, erow = (nidx + 7u) >> 3;
        /* 位流 ix = pay + 32 + rows*2 要 128 B 对齐: 两种 rows 的 32 + rows*2 都 ≡ 32 (mod 128) ⇒ 载荷起点 ≡ 96 */
        const uint64_t off = align128(cur) + 96u;
        tab[(size_t)e * 3u + w] = off;
        cur = off + 32u + (uint64_t)rows * 2u + (uint64_t)rows * mrow + (ext ? (uint64_t)rows * erow : 0u) + 8u;
    }
    L.bytes = align128(cur);
    CK(cudaMalloc(&L.d, L.bytes));
    fill_kernel<<<(unsigned)((L.bytes / 4u + 255u) / 256u), 256>>>((uint32_t *)L.d, L.bytes / 4u, seed);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    /* 头 + 槽表 */
    uint32_t hdr[4] = { DS4VQ_BLOB_MAGIC, 3u, seed, NE };
    CK(cudaMemcpy(L.d, hdr, 16, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(L.d + 16, tab, (size_t)NE * 3u * 8u, cudaMemcpyHostToDevice));
    /* 码本: E4M3 随机, 清 bit6 ⇒ 指数 ≤ 7, |v| ≤ 1.875, 永不 NaN */
    uint8_t *cb = (uint8_t *)malloc((size_t)nc * 8u);
    for (uint32_t i = 0; i < nc * 8u; i++) cb[i] = (uint8_t)(h32(seed, 11u, i) & 0xBFu);
    CK(cudaMemcpy(L.d + L.cb_off, cb, (size_t)nc * 8u, cudaMemcpyHostToDevice));
    free(cb);
    /* 每个载荷: 32 B 头 + 行增益(f16 ∈ [0.5, 1)) */
    uint8_t *ph = (uint8_t *)malloc(32u + (size_t)OUT * 2u);
    for (uint32_t e = 0; e < NE; e++) for (uint32_t w = 0; w < 3u; w++) {
        const uint32_t rows = w == 2u ? OUT : MID, cols = w == 2u ? MID : IN;
        const uint32_t mg = DS4VQ_MAT3_MAGIC, flags = 1u | (ext ? 2u : 0u), mnb = 12u; const uint16_t d16 = 8u, n16 = (uint16_t)nc;
        memcpy(ph, &mg, 4); memcpy(ph + 4, &d16, 2); memcpy(ph + 6, &n16, 2); memcpy(ph + 8, &rows, 4); memcpy(ph + 12, &cols, 4);
        memcpy(ph + 16, &flags, 4); memcpy(ph + 20, &mnb, 4); memcpy(ph + 24, &L.cb_off, 8);
        for (uint32_t r = 0; r < rows; r++) { const uint16_t g = (uint16_t)(0x3800u | (h32(seed, 13u + w, e * 8192u + r) & 0x3FFu)); memcpy(ph + 32 + (size_t)r * 2u, &g, 2); }
        CK(cudaMemcpy(L.d + tab[(size_t)e * 3u + w], ph, 32u + (size_t)rows * 2u, cudaMemcpyHostToDevice));
    }
    free(ph); free(tab);
    CK(cudaMalloc(&L.cbh, (size_t)nc * 16u));
    cb_bf16_kernel<<<(nc + 255u) / 256u, 256>>>(L.cbh, L.d + L.cb_off, nc);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    return L;
}

/* 真载荷: GGUF 里的 blk.L.ffn_exps_vq.blob, 载荷按引擎 cuda_vq_align 的口径挪到位流 128 B 对齐(前缀原样, 载荷按原偏移升序, 槽表改新偏移) */
typedef struct { uint64_t off; uint32_t slot; } pl_t;
static int pl_cmp(const void *a, const void *b) { const uint64_t x = ((const pl_t *)a)->off, y = ((const pl_t *)b)->off; return x < y ? -1 : (x > y); }
static layer_blob load_layer(const ds4_gguf *g, uint32_t layer) {
    layer_blob L; memset(&L, 0, sizeof L);
    char nm[64]; snprintf(nm, sizeof nm, "blk.%u.ffn_exps_vq.blob", layer);
    const ds4_gguf_tensor *t = ds4_gguf_find(g, nm);
    uint64_t bytes = 0; const uint8_t *hb = t ? ds4_gguf_tensor_data(g, t, &bytes) : NULL;
    if (!hb || !ds4vq_blob_ok(hb, (size_t)bytes) || ds4vq_blob_ver(hb) != 3u || ds4vq_blob_nexp(hb) != NE) { fprintf(stderr, "★%s 不是 384 专家的 v3 blob★\n", nm); exit(1); }
    const uint32_t ns = NE * 3u; pl_t *pl = (pl_t *)malloc((size_t)ns * sizeof(pl_t)); uint32_t np = 0;
    for (uint32_t k = 0; k < ns; k++) { const uint64_t o = ds4vq_slot(hb, (int)(k / 3u), (int)(k % 3u)); if (o) { pl[np].off = o; pl[np].slot = k; np++; } }
    qsort(pl, np, sizeof(pl_t), pl_cmp);
    const uint64_t first = pl[0].off; uint64_t cur = first;
    uint64_t *noff = (uint64_t *)malloc((size_t)np * 8), *tab = (uint64_t *)malloc((size_t)ns * 8);
    memcpy(tab, hb + 16, (size_t)ns * 8u);
    for (uint32_t i = 0; i < np; i++) {
        uint32_t rows; memcpy(&rows, hb + pl[i].off + 8, 4);
        const uint64_t lead = (32u + (uint64_t)rows * 2u) & 127u;
        cur += ((128u - lead) - cur) & 127u;
        noff[i] = cur; tab[pl[i].slot] = cur;
        cur += (i + 1u < np ? pl[i + 1u].off : bytes) - pl[i].off;
    }
    L.bytes = cur;
    CK(cudaMalloc(&L.d, L.bytes)); CK(cudaMemset(L.d, 0, L.bytes));
    CK(cudaMemcpy(L.d, hb, (size_t)first, cudaMemcpyHostToDevice));
    for (uint32_t i = 0; i < np; i++) CK(cudaMemcpy(L.d + noff[i], hb + pl[i].off, (size_t)((i + 1u < np ? pl[i + 1u].off : bytes) - pl[i].off), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(L.d + 16, tab, (size_t)ns * 8u, cudaMemcpyHostToDevice));
    { uint16_t n16; memcpy(&n16, hb + first + 6, 2); memcpy(&L.cb_off, hb + first + 24, 8); L.nc = n16; L.nbit = 0; while ((1u << L.nbit) < L.nc) L.nbit++; }
    free(pl); free(noff); free(tab);
    CK(cudaMalloc(&L.cbh, (size_t)L.nc * 16u));
    cb_bf16_kernel<<<(L.nc + 255u) / 256u, 256>>>(L.cbh, L.d + L.cb_off, L.nc);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    printf("真载荷 %s: %.2f GB(对齐后), 码本 %u 词(%u 位)\n", nm, (double)L.bytes / 1e9, L.nc, L.nbit);
    return L;
}

/* ---- 路由与工作项: n_tok 个 token 各挑 6 个不同专家(哈希), 按专家计数排序; 工作项按 BN 切 ---- */
typedef struct { uint32_t n_tok, nvalid; uint32_t *cnt, *off; } routing;
/* 真路由(第 7 个参数, 引擎 --v41-prof 落的 /tmp/v41_route_Lnn_n2048.txt, 一行一个专家的 token 数): 按 n_tok/文件 n 等比缩放。
 * 为什么: 真路由极偏 —— 2048 token 一块只有 200~350 个专家有 token, 最热的一个吃 1000~2000 个(12288 个名额的 15%), 均匀哈希路由
 * (384 个各 32 个)算出的项数/成本与引擎对不上。 */
static const char *g_route_file = NULL;
static routing make_routing(uint32_t n_tok, uint32_t seed) {
    routing R; R.n_tok = n_tok; R.cnt = (uint32_t *)calloc(NE, 4); R.off = (uint32_t *)malloc((NE + 1u) * 4);
    if (g_route_file) {
        FILE *f = fopen(g_route_file, "r"); uint64_t tot = 0; uint32_t raw[1024];
        if (!f) { fprintf(stderr, "★路由文件打不开 %s★\n", g_route_file); exit(1); }
        for (uint32_t e = 0; e < NE; e++) { if (fscanf(f, "%u", &raw[e]) != 1) raw[e] = 0; tot += raw[e]; }
        fclose(f);
        uint64_t acc = 0, want = (uint64_t)n_tok * KU;
        for (uint32_t e = 0; e < NE; e++) { const uint64_t next = (uint64_t)(acc + raw[e]) * want / tot; R.cnt[e] = (uint32_t)(next - (uint64_t)acc * want / tot); acc += raw[e]; }
        R.off[0] = 0; for (uint32_t e = 0; e < NE; e++) R.off[e + 1] = R.off[e] + R.cnt[e];
        R.nvalid = R.off[NE];
        return R;
    }
    for (uint32_t t = 0; t < n_tok; t++) {
        uint32_t pick[6];
        for (uint32_t k = 0; k < KU; k++) {
            uint32_t e, tries = 0;
            do { e = h32(seed, t, k + 64u * tries) % NE; tries++; bool dup = false; for (uint32_t j = 0; j < k; j++) if (pick[j] == e) dup = true; if (!dup) break; } while (1);
            pick[k] = e; R.cnt[e]++;
        }
    }
    R.off[0] = 0; for (uint32_t e = 0; e < NE; e++) R.off[e + 1] = R.off[e] + R.cnt[e];
    R.nvalid = R.off[NE];
    return R;
}
static uint32_t make_items(const routing *R, uint32_t BN, vqp_item **out) {
    uint32_t n = 0; for (uint32_t e = 0; e < NE; e++) n += (R->cnt[e] + BN - 1u) / BN;
    vqp_item *it = (vqp_item *)malloc((size_t)n * sizeof(vqp_item)); uint32_t k = 0;
    for (uint32_t e = 0; e < NE; e++) for (uint32_t t0 = 0; t0 < R->cnt[e]; t0 += BN) { it[k].e = (int32_t)e; it[k].t0 = (int32_t)t0; it[k].nt = (int32_t)(R->cnt[e] - t0 < BN ? R->cnt[e] - t0 : BN); k++; }
    *out = it; return n;
}

/* ---- 发一层的三发(gate → up → down); V=-1 是引擎 V0 ---- */
typedef struct { const char *name; uint32_t BN, BK, TW; int CB, kind; } variant;   /* kind 0 = vqm_v_kernel(分相), 1 = vqm_pc_kernel(生产/消费双组) */
static const variant VAR[] = {
    { "V0 引擎现核 BN32 BK64 8w cbE4M3",   32u, 64u, 1u, 0, 0 },
    { "V1 BN64 BK64 8w cbE4M3",           64u, 64u, 1u, 0, 0 },
    { "V2 BN128 BK64 8w cbE4M3",         128u, 64u, 1u, 0, 0 },
    { "V3 BN128 BK64 16w cbE4M3",        128u, 64u, 2u, 0, 0 },
    { "V4 BN32 BK32 8w cbE4M3(2blk/SM)",  32u, 32u, 1u, 0, 0 },
    { "V5 BN64 BK64 8w cb-bf16-shared",   64u, 64u, 1u, 1, 0 },
    { "V6 BN64 BK64 8w cb-bf16-L2",       64u, 64u, 1u, 2, 0 },
    { "V7 BN128 BK64 16w cb-bf16-L2",    128u, 64u, 2u, 2, 0 },
    { "V8 BN128 BK32 16w cb-bf16-L2",    128u, 32u, 2u, 2, 0 },
    { "V9 BN128 BK64 16w cb-bf16-shared", 128u, 64u, 2u, 1, 0 },
    { "V10 PC BN128 BK32 cbE4M3",        128u, 32u, 2u, 0, 1 },
    { "V11 PC BN128 BK32 cb-bf16-shared", 128u, 32u, 2u, 1, 1 },
    { "V12 PC BN64 BK32 cbE4M3",          64u, 32u, 2u, 0, 1 },
    { "V13 DB BN128 BK32 16w cbE4M3",    128u, 32u, 2u, 0, 2 },   /* kind 2 = vqm_db_kernel: 单组 warp 双缓冲交错 */
    { "V14 DB BN128 BK32 16w cb-bf16-sh", 128u, 32u, 2u, 1, 2 },
    /* V15 DB BN64(kind 2) 撤下: 13 位 down 那一实例的位平面读越界(compute-sanitizer 定位在 load_round 的 ep), BN128 的 V13/V14 干净; 原因未查清 */
};
static const int NV = (int)(sizeof(VAR) / sizeof(VAR[0]));
static size_t smem_of(const variant *v, uint32_t nc) {
    const size_t cb = (size_t)(v->CB == 0 ? nc * 8u : (v->CB == 1 ? nc * 16u : 0u));
    if (v->kind >= 1) return cb + 2u * (size_t)(VQM_BM + v->BN) * 32u * 2u;   /* 双缓冲 A+B 片(PC 与 DB 同) */
    return cb + (size_t)VQM_BM * v->BK * 2u + (size_t)v->BN * v->BK * 2u;
}
/* 模板实例分发: (EXT, MODE) × 变体。每个实例的 grid = SM × 占用率(第一次发时算, 记在表里) */
static int g_nsm = 0;
template <int EXT, int MODE, uint32_t BN, uint32_t BK, uint32_t TW, int CB>
static void launch_one(int *occ, size_t smem, float *g32, uint16_t *h16, float *ys, const layer_blob *L, const vqp_item *items, uint32_t nitems,
                       const uint16_t *act, const uint32_t *off, uint32_t M, uint32_t K, float clamp) {
    if (*occ == 0) {
        CK(cudaFuncSetAttribute(vqm_v_kernel<EXT, MODE, BN, BK, TW, CB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(occ, vqm_v_kernel<EXT, MODE, BN, BK, TW, CB>, 256 * TW, smem));
        if (*occ == 0) *occ = -1;
    }
    if (*occ < 0) return;
    const uint32_t ntile = (M + VQM_BM - 1u) / VQM_BM, nwork = nitems * ntile;
    uint32_t grid = (uint32_t)g_nsm * (uint32_t)*occ; if (grid > nwork) grid = nwork;
    vqm_v_kernel<EXT, MODE, BN, BK, TW, CB><<<grid, 256 * TW, smem>>>(g32, h16, ys, L->d, items, nitems, act, off, M, K, clamp, L->nc * 8u, NULL, L->cbh);
}
template <int EXT, int MODE, uint32_t BN, int CB>
static void launch_pc(int *occ, size_t smem, float *g32, uint16_t *h16, float *ys, const layer_blob *L, const vqp_item *items, uint32_t nitems,
                      const uint16_t *act, const uint32_t *off, uint32_t M, uint32_t K, float clamp) {
    if (*occ == 0) {
        CK(cudaFuncSetAttribute(vqm_pc_kernel<EXT, MODE, BN, CB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(occ, vqm_pc_kernel<EXT, MODE, BN, CB>, 512, smem));
        if (*occ == 0) *occ = -1;
    }
    if (*occ < 0) return;
    const uint32_t ntile = (M + VQM_BM - 1u) / VQM_BM, nwork = nitems * ntile;
    uint32_t grid = (uint32_t)g_nsm * (uint32_t)*occ; if (grid > nwork) grid = nwork;
    vqm_pc_kernel<EXT, MODE, BN, CB><<<grid, 512, smem>>>(g32, h16, ys, L->d, items, nitems, act, off, M, K, clamp, L->nc * 8u, NULL, L->cbh);
}
template <int EXT, int MODE, uint32_t BN, int CB>
static void launch_db(int *occ, size_t smem, float *g32, uint16_t *h16, float *ys, const layer_blob *L, const vqp_item *items, uint32_t nitems,
                      const uint16_t *act, const uint32_t *off, uint32_t M, uint32_t K, float clamp) {
    if (*occ == 0) {
        CK(cudaFuncSetAttribute(vqm_db_kernel<EXT, MODE, BN, CB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(occ, vqm_db_kernel<EXT, MODE, BN, CB>, 512, smem));
        if (*occ == 0) *occ = -1;
    }
    if (*occ < 0) return;
    const uint32_t ntile = (M + VQM_BM - 1u) / VQM_BM, nwork = nitems * ntile;
    uint32_t grid = (uint32_t)g_nsm * (uint32_t)*occ; if (grid > nwork) grid = nwork;
    vqm_db_kernel<EXT, MODE, BN, CB><<<grid, 512, smem>>>(g32, h16, ys, L->d, items, nitems, act, off, M, K, clamp, L->nc * 8u, NULL, L->cbh);
}
#define DISPATCH(EXT, MODE, BN, BK, TW, CB) launch_one<EXT, MODE, BN, BK, TW, CB>(&occ[v][EXT][MODE], smem, g32, h16, ys, L, items, nitems, act, off, M, K, clamp)
#define DISPATCH_PC(EXT, MODE, BN, CB) launch_pc<EXT, MODE, BN, CB>(&occ[v][EXT][MODE], smem, g32, h16, ys, L, items, nitems, act, off, M, K, clamp)
#define DISPATCH_DB(EXT, MODE, BN, CB) launch_db<EXT, MODE, BN, CB>(&occ[v][EXT][MODE], smem, g32, h16, ys, L, items, nitems, act, off, M, K, clamp)
static int occ[24][2][3];
static void run_mode(int v, int ext, int mode, float *g32, uint16_t *h16, float *ys, const layer_blob *L, const vqp_item *items, uint32_t nitems,
                     const uint16_t *act, const uint32_t *off, float clamp) {
    const uint32_t M = mode == 2 ? OUT : MID, K = mode == 2 ? MID : IN;
    const size_t smem = smem_of(&VAR[v], L->nc);
    if (v == 0) {   /* 引擎现核: grid = SM 数(逐字) */
        static int set[2] = {0, 0};
        if (!set[ext]) { const int mx = (int)(8192u * 8u + VQM_TILE_BYTES);
            if (ext) { CK(cudaFuncSetAttribute(vqm_kernel<1, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, mx)); CK(cudaFuncSetAttribute(vqm_kernel<1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, mx)); CK(cudaFuncSetAttribute(vqm_kernel<1, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, mx)); }
            else { CK(cudaFuncSetAttribute(vqm_kernel<0, 0>, cudaFuncAttributeMaxDynamicSharedMemorySize, mx)); CK(cudaFuncSetAttribute(vqm_kernel<0, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, mx)); CK(cudaFuncSetAttribute(vqm_kernel<0, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, mx)); }
            set[ext] = 1; }
        const uint32_t ntile = (M + VQM_BM - 1u) / VQM_BM, nw = nitems * ntile, grid = nw < (uint32_t)g_nsm ? nw : (uint32_t)g_nsm;
        const uint32_t cbb = L->nc * 8u, shb = cbb + VQM_TILE_BYTES;
#define V0L(E, MD) vqm_kernel<E, MD><<<grid, VQM_THREADS, shb>>>(g32, h16, ys, L->d, items, nitems, act, off, M, K, clamp, cbb, NULL)
        if (ext) { if (mode == 0) V0L(1, 0); else if (mode == 1) V0L(1, 1); else V0L(1, 2); }
        else     { if (mode == 0) V0L(0, 0); else if (mode == 1) V0L(0, 1); else V0L(0, 2); }
        return;
    }
#define VCASE(BN, BK, TW, CB) \
    if (ext) { if (mode == 0) DISPATCH(1, 0, BN, BK, TW, CB); else if (mode == 1) DISPATCH(1, 1, BN, BK, TW, CB); else DISPATCH(1, 2, BN, BK, TW, CB); } \
    else     { if (mode == 0) DISPATCH(0, 0, BN, BK, TW, CB); else if (mode == 1) DISPATCH(0, 1, BN, BK, TW, CB); else DISPATCH(0, 2, BN, BK, TW, CB); }
#define PCASE(BN, CB) \
    if (ext) { if (mode == 0) DISPATCH_PC(1, 0, BN, CB); else if (mode == 1) DISPATCH_PC(1, 1, BN, CB); else DISPATCH_PC(1, 2, BN, CB); } \
    else     { if (mode == 0) DISPATCH_PC(0, 0, BN, CB); else if (mode == 1) DISPATCH_PC(0, 1, BN, CB); else DISPATCH_PC(0, 2, BN, CB); }
#define DCASE(BN, CB) \
    if (ext) { if (mode == 0) DISPATCH_DB(1, 0, BN, CB); else if (mode == 1) DISPATCH_DB(1, 1, BN, CB); else DISPATCH_DB(1, 2, BN, CB); } \
    else     { if (mode == 0) DISPATCH_DB(0, 0, BN, CB); else if (mode == 1) DISPATCH_DB(0, 1, BN, CB); else DISPATCH_DB(0, 2, BN, CB); }
    switch (v) {
    case 1: VCASE(64u, 64u, 1u, 0); break;   case 2: VCASE(128u, 64u, 1u, 0); break;   case 3: VCASE(128u, 64u, 2u, 0); break;
    case 4: VCASE(32u, 32u, 1u, 0); break;   case 5: VCASE(64u, 64u, 1u, 1); break;    case 6: VCASE(64u, 64u, 1u, 2); break;
    case 7: VCASE(128u, 64u, 2u, 2); break;  case 8: VCASE(128u, 32u, 2u, 2); break;   case 9: VCASE(128u, 64u, 2u, 1); break;
    case 10: PCASE(128u, 0); break;          case 11: PCASE(128u, 1); break;           case 12: PCASE(64u, 0); break;
    case 13: DCASE(128u, 0); break;          case 14: DCASE(128u, 1); break;           case 15: DCASE(64u, 0); break;
    default: break;
    }
}
static bool variant_ok(int v, int ext) { return !(VAR[v].CB == 1 && ext); }   /* 13 位码本 bf16 128 KB 进不了 shared */

/* CPU double 参考: 专家 e 的 w1 第 r 行 · token(排序位 p)的激活, 只为核对合成布局 */
static double ref_gate(const uint8_t *hblob, const layer_blob *L, uint32_t e, uint32_t r, const uint16_t *hact, uint32_t p) {
    uint64_t off; memcpy(&off, hblob + 16 + ((size_t)e * 3u) * 8u, 8);
    const uint8_t *pay = hblob + off; const uint32_t nidx = IN / 8u, mrow = (nidx * 12u + 7u) / 8u, erow = (nidx + 7u) >> 3;
    const uint8_t *gr = pay + 32, *ix = gr + (size_t)MID * 2u, *ex = L->nbit > 12u ? ix + (size_t)MID * mrow : NULL;
    const uint8_t *row = ix + (size_t)r * mrow, *cb = hblob + L->cb_off;
    double s = 0.0;
    for (uint32_t j = 0; j < nidx; j++) {
        const uint32_t bit = 12u * j; uint32_t v = (uint32_t)((row[bit >> 3] | ((uint32_t)row[(bit >> 3) + 1] << 8)) >> (bit & 7u)) & 0xFFFu;
        if (ex) v |= ((ex[(size_t)r * erow + (j >> 3)] >> (j & 7u)) & 1u) << 12;
        for (uint32_t d = 0; d < 8u; d++) { const uint32_t hb = hact[(size_t)p * IN + 8u * j + d]; float x; const uint32_t xu = hb << 16; memcpy(&x, &xu, 4);
                                             s += (double)ds4_e4m3fn_to_f32(cb[(size_t)v * 8u + d]) * (double)x; }
    }
    uint16_t g; memcpy(&g, gr + (size_t)r * 2u, 2);
    return s * (double)ds4vq_f16(g);
}

int main(int argc, char **argv) {
    const uint32_t iters = argc > 1 ? (uint32_t)atoi(argv[1]) : 3u; const int only_v = argc > 2 ? atoi(argv[2]) : -1; const uint32_t only_n = argc > 3 ? (uint32_t)atoi(argv[3]) : 0u;
    { cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0)); g_nsm = pr.multiProcessorCount;
      printf("板子: %s, SM %d, shared/SM %zu KB, 每 block 可批 %zu KB, 寄存器/SM %d, L2 %d MB\n", pr.name, g_nsm, pr.sharedMemPerMultiprocessor >> 10, pr.sharedMemPerBlockOptin >> 10, pr.regsPerMultiprocessor, pr.l2CacheSize >> 20); }
    const uint32_t ntoks[3] = { 512u, 2048u, 4096u }; const float clamp = 30.f;
    if (argc > 7) { g_route_file = argv[7]; printf("真路由: %s(按 token 档等比缩放)\n", g_route_file); }
    layer_blob Lb[2];
    if (argc > 4) {   /* 真载荷档: 12 位层与 13 位层各取盘上一层 */
        ds4_gguf g; char err[256];
        if (ds4_gguf_open(&g, argv[4], err, sizeof err) != 0) { fprintf(stderr, "★GGUF 打不开: %s★\n", err); return 2; }
        Lb[0] = load_layer(&g, argc > 5 ? (uint32_t)atoi(argv[5]) : 20u);
        Lb[1] = load_layer(&g, argc > 6 ? (uint32_t)atoi(argv[6]) : 5u);
        if (Lb[0].nbit != 12u || Lb[1].nbit != 13u) { fprintf(stderr, "★层号给反了: 第 5 个参数要 12 位层, 第 6 个要 13 位层★\n"); return 2; }
        ds4_gguf_close(&g);
    } else {
        Lb[0] = make_layer(4096u, 1u); Lb[1] = make_layer(8192u, 2u);
        printf("合成层: 12 位 %.2f GB / 13 位 %.2f GB(位流/码本按哈希; 每 token 6 个不同专家)\n", (double)Lb[0].bytes / 1e9, (double)Lb[1].bytes / 1e9);
    }
    /* 激活按排序位生成(与 BN 无关), 三档 token 各一份; 输出缓冲按最大档开 */
    routing R[3]; uint16_t *act[3]; uint32_t *doff[3];
    for (int n = 0; n < 3; n++) {
        R[n] = make_routing(ntoks[n], 100u + (uint32_t)n);
        uint16_t *h = (uint16_t *)malloc((size_t)R[n].nvalid * IN * 2u);
        for (uint64_t i = 0; i < (uint64_t)R[n].nvalid * IN; i++) h[i] = (uint16_t)(0x3F00u | (h32(7u + (uint32_t)n, (uint32_t)(i >> 32), (uint32_t)i) & 0x807Fu));   /* ±[0.5, 1) bf16 */
        CK(cudaMalloc(&act[n], (size_t)R[n].nvalid * IN * 2u)); CK(cudaMemcpy(act[n], h, (size_t)R[n].nvalid * IN * 2u, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&doff[n], (NE + 1u) * 4)); CK(cudaMemcpy(doff[n], R[n].off, (NE + 1u) * 4, cudaMemcpyHostToDevice));
        if (n == 2) { /* 参考门用第 2 档(4096)的激活主机副本 */ }
        free(h);
    }
    const uint32_t nvmax = R[2].nvalid;
    float *g32, *ys; uint16_t *h16;
    CK(cudaMalloc(&g32, (size_t)nvmax * MID * 4)); CK(cudaMalloc(&h16, (size_t)nvmax * MID * 2)); CK(cudaMalloc(&ys, (size_t)nvmax * OUT * 4));
    /* 工作项: 每档 × 每种 BN 一份 */
    vqp_item *items[3][3]; uint32_t nitems[3][3]; const uint32_t bns[3] = { 32u, 64u, 128u };
    for (int n = 0; n < 3; n++) for (int b = 0; b < 3; b++) { vqp_item *h; nitems[n][b] = make_items(&R[n], bns[b], &h);
        CK(cudaMalloc(&items[n][b], (size_t)nitems[n][b] * sizeof(vqp_item))); CK(cudaMemcpy(items[n][b], h, (size_t)nitems[n][b] * sizeof(vqp_item), cudaMemcpyHostToDevice)); free(h); }
    for (int n = 0; n < 3; n++) printf("token %u: 有效对 %u, 工作项 BN32 %u / BN64 %u / BN128 %u(专家平均 %.1f token)\n", ntoks[n], R[n].nvalid, nitems[n][0], nitems[n][1], nitems[n][2], (double)R[n].nvalid / NE);
    auto bidx = [&](uint32_t BN) { return BN == 32u ? 0 : (BN == 64u ? 1 : 2); };
    auto run_layer = [&](int v, int ext, int n) {
        const int b = bidx(VAR[v].BN);
        for (int mode = 0; mode < 3; mode++) run_mode(v, ext, mode, g32, h16, ys, &Lb[ext], items[n][b], nitems[n][b], mode == 2 ? h16 : act[n], doff[n], clamp);
    };
    /* ① 布局门: V0 的 gate 输出对 CPU double(合成 12 位与 13 位各抽 64 个 (行, token)) */
    int bad = 0;
    for (int ext = 0; ext < 2; ext++) {
        const int n = 0; run_layer(0, ext, n); CK(cudaDeviceSynchronize());
        uint8_t *hb = (uint8_t *)malloc(Lb[ext].bytes); CK(cudaMemcpy(hb, Lb[ext].d, Lb[ext].bytes, cudaMemcpyDeviceToHost));
        uint16_t *ha = (uint16_t *)malloc((size_t)R[n].nvalid * IN * 2u); CK(cudaMemcpy(ha, act[n], (size_t)R[n].nvalid * IN * 2u, cudaMemcpyDeviceToHost));
        float *hg = (float *)malloc((size_t)R[n].nvalid * MID * 4); CK(cudaMemcpy(hg, g32, (size_t)R[n].nvalid * MID * 4, cudaMemcpyDeviceToHost));
        double worst = 0.0;
        for (uint32_t s = 0; s < 64u; s++) {
            const uint32_t e = h32(3u, s, 0u) % NE; if (!R[n].cnt[e]) continue;
            const uint32_t p = R[n].off[e] + h32(3u, s, 1u) % R[n].cnt[e], r = h32(3u, s, 2u) % MID;
            const double ref = ref_gate(hb, &Lb[ext], e, r, ha, p), got = (double)hg[(size_t)p * MID + r];
            const double rel = fabs(ref - got) / (fabs(ref) + 1e-3); if (rel > worst) worst = rel;
        }
        printf("① 布局门 %s: V0 gate 对 CPU double 最大相对误差 %.2e %s\n", ext ? "13 位" : "12 位", worst, worst < 2e-2 ? "✓" : "★布局不一致★");
        if (worst >= 2e-2) bad = 1;
        free(hb); free(ha); free(hg);
    }
    /* ② 逐位门: 每变体 vs V0, 12/13 位 × token 512/4096 */
    float *rg = (float *)malloc((size_t)nvmax * MID * 4), *ry = (float *)malloc((size_t)nvmax * OUT * 4), *tg = (float *)malloc((size_t)nvmax * MID * 4), *ty = (float *)malloc((size_t)nvmax * OUT * 4);
    uint16_t *rh = (uint16_t *)malloc((size_t)nvmax * MID * 2), *th = (uint16_t *)malloc((size_t)nvmax * MID * 2);
    for (int ext = 0; ext < 2; ext++) for (int n = 0; n < 3; n += 2) {
        const size_t gb = (size_t)R[n].nvalid * MID * 4, hb = (size_t)R[n].nvalid * MID * 2, yb = (size_t)R[n].nvalid * OUT * 4;
        run_layer(0, ext, n); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(rg, g32, gb, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(rh, h16, hb, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ry, ys, yb, cudaMemcpyDeviceToHost));
        for (int v = 1; v < NV; v++) {
            if (!variant_ok(v, ext) || (only_v > 0 && v != only_v)) continue;   /* 只查一个变体(配 compute-sanitizer 定位越界用) */
            CK(cudaMemset(g32, 0xFF, gb)); CK(cudaMemset(h16, 0xFF, hb)); CK(cudaMemset(ys, 0xFF, yb));
            fprintf(stderr, "  [②] %s %s token %u\n", ext ? "13位" : "12位", VAR[v].name, ntoks[n]);   /* 核出错时知道是谁(stderr 不缓冲) */
            run_layer(v, ext, n); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(tg, g32, gb, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(th, h16, hb, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ty, ys, yb, cudaMemcpyDeviceToHost));
            const int same = memcmp(rg, tg, gb) == 0 && memcmp(rh, th, hb) == 0 && memcmp(ry, ty, yb) == 0;
            printf("  %s token %-4u %-34s vs V0: %s\n", ext ? "13位" : "12位", ntoks[n], VAR[v].name, same ? "逐位同 ✓" : "★不同★"); bad |= !same;
        }
    }
    /* ③ 计时: 一层三发(gate+up+down)的 ms, 三档 token; 每 SM 挂几个 block 也报出来 */
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    for (int ext = 0; ext < 2; ext++) {
        printf("\n%s层 ms/层块(gate+up+down)%-14s", ext ? "13 位" : "12 位", ""); for (int n = 0; n < 3; n++) printf(" token%-5u", ntoks[n]); printf("  blk/SM\n");
        for (int v = 0; v < NV; v++) {
            if (only_v >= 0 && v != only_v) continue;
            if (!variant_ok(v, ext)) { printf("%-38s   —(码本进不了 shared)\n", VAR[v].name); continue; }
            printf("%-38s", VAR[v].name);
            for (int n = 0; n < 3; n++) {
                if (only_n && ntoks[n] != only_n) { printf("     -    "); continue; }
                run_layer(v, ext, n); CK(cudaDeviceSynchronize());
                float best = 1e30f;
                for (uint32_t it = 0; it < iters; it++) { CK(cudaEventRecord(t0)); run_layer(v, ext, n); CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1)); float ms; CK(cudaEventElapsedTime(&ms, t0, t1)); if (ms < best) best = ms; }
                printf(" %8.2f ", best);
            }
            printf("  %d\n", v == 0 ? 1 : occ[v][ext][0]);
        }
    }
    printf("\n%s\n", bad ? "★★门有红, 上面的数字不作数★★" : "门全绿(布局对 CPU 参考; 变体 == V0 逐位同)");
    return bad ? 1 : 0;
}
