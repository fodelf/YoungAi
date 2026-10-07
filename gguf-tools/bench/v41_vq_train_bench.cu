/* v41_vq_train_bench.cu — 后训练专家核按训练形状的微基准(spark 本机跑; 2026-10-03 立)。
 *
 * 【为什么要它】10-03 训练逐核表(batch 4, maxlen 880, 5 步): 专家核占 GPU 时间 52.5% —— 前向/重算的 vqm 每发 5.5 ms(12 位)、
 * 反传的转置 vqt 每发 7.6 ms, 而一发要读的位流(有 token 的专家)按 232 GB/s 只要 ~3.3 ms。训练一包 ~880 token, 每个专家平均
 * ~14 个 token, 与预填(每块 2048 token)的形状完全不同 —— 09-29 的预填微基准(v41_vq_prefill_mma_bench)量不到这里。
 * 换核形态先在这里过(逐位门 + 计时), 过了才进引擎(一趟要装 113 GB)。
 *
 * 【比什么】ref = 引擎现核逐字抄本(v41_vq_train_bench_ref.cuh): vqm_kernel(BN128, 16 warp)与 vqt_prescale + vqt_kernel;
 *   new = 寄存器直解形态(v41_vq_train_bench_new.cuh): vqr_kernel / vqrt_kernel, 一个工作项最多 16·MTM 个 token;
 *   mix = token 多的专家(> 16·MTM)仍交给 ref 核(BN128 切项), 其余走 new —— 热专家一个吃几百个 token, new 切 32 一项要把同一块矩阵解十几遍。
 *   wall = 整层 blob 顺序读一遍的时间(按有 token 专家的字节占比折算), 是这一发的字节墙。
 * ★判据★: new / mix 的输出(g32 / h16 / H_u / ys, 转置的 G_A / G_X)与 ref 逐位同。
 * 形状 = V4.1 Flash: IN 5120 / MID 2304 / OUT 5120 / 384 专家 top-6; 真载荷(GGUF 的 blk.L.ffn_exps_vq.blob, 12 位层与 13 位层各一层)。
 * 路由: 给路由文件(引擎 --v41-prof 落的 /tmp/v41_route_Lnn_n2048.txt, 一行一个专家的 token 数)就按 n_tok×6 等比缩放, 不给就哈希均匀。
 * 用法: nvcc -O3 --use_fast_math -arch=native -std=c++17 -o v41_vq_train_bench v41_vq_train_bench.cu ../../src/common/ds4_gguf.c ../../src/common/ds4_quantfmt.c
 *       ./v41_vq_train_bench GGUF [12 位层=20] [13 位层=5] [n_tok=880] [路由文件] [遍数=5] */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
extern "C" {
#include "../../src/common/ds4_gguf.h"
}
#include "v41_vq_train_bench_ref.cuh"
#include "v41_vq_train_bench_new.cuh"
#include "v41_vq_train_bench_t.cuh"
#include "v41_vq_train_bench_h.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "★CUDA %s @%d: %s★\n", #x, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
static const uint32_t NE = 384u, KU = 6u, IN = 5120u, MID = 2304u, OUT = 5120u;

/* 真载荷: 载荷按引擎 cuda_vq_align 的口径挪到位流 128 B 对齐(同 v41_vq_prefill_mma_bench.cu 的 load_layer) */
typedef struct { uint8_t *d; uint64_t bytes; uint32_t nc, nbit; } layer_blob;
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
    { uint16_t n16; memcpy(&n16, hb + first + 6, 2); L.nc = n16; L.nbit = 0; while ((1u << L.nbit) < L.nc) L.nbit++; }
    free(pl); free(noff); free(tab);
    printf("真载荷 %s: %.2f GB, 码本 %u 词(%u 位)\n", nm, (double)L.bytes / 1e9, L.nc, L.nbit);
    return L;
}

__host__ __device__ static uint32_t h32(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t x = a * 0x9E3779B1u ^ (b + 0x7F4A7C15u) * 0x85EBCA6Bu ^ (c + 0x165667B1u) * 0xC2B2AE35u;
    x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15; return x;
}
/* 路由: 每专家 token 数 cnt[e](排序后专家 e 的配对在 [off[e], off[e+1])) */
static void make_counts(uint32_t n_tok, const char *route, uint32_t *cnt) {
    memset(cnt, 0, NE * 4);
    if (route) {
        FILE *f = fopen(route, "r"); uint64_t tot = 0; uint32_t raw[NE];
        if (!f) { fprintf(stderr, "★路由文件打不开 %s★\n", route); exit(1); }
        for (uint32_t e = 0; e < NE; e++) { if (fscanf(f, "%u", &raw[e]) != 1) raw[e] = 0; tot += raw[e]; }
        fclose(f);
        uint64_t acc = 0, want = (uint64_t)n_tok * KU;
        for (uint32_t e = 0; e < NE; e++) { const uint64_t nx = (acc + raw[e]) * want / tot; cnt[e] = (uint32_t)(nx - acc * want / tot); acc += raw[e]; }
        return;
    }
    for (uint32_t t = 0; t < n_tok; t++) {
        uint32_t pick[6];
        for (uint32_t k = 0; k < KU; k++) {
            uint32_t e, tries = 0;
            do { e = h32(5u, t, k + 64u * tries) % NE; tries++; bool dup = false; for (uint32_t j = 0; j < k; j++) if (pick[j] == e) dup = true; if (!dup) break; } while (1);
            pick[k] = e; cnt[e]++;
        }
    }
}
/* 工作项表: 专家 e 进表当且仅当 lo < cnt[e] ≤ hi; 按 bn 切 */
static uint32_t make_items(const uint32_t *cnt, uint32_t bn, uint32_t lo, uint32_t hi, vqp_item **dev) {
    uint32_t n = 0; for (uint32_t e = 0; e < NE; e++) if (cnt[e] > lo && cnt[e] <= hi) n += (cnt[e] + bn - 1u) / bn;
    if (!n) { *dev = NULL; return 0; }
    vqp_item *h = (vqp_item *)malloc((size_t)n * sizeof(vqp_item)); uint32_t k = 0;
    for (uint32_t e = 0; e < NE; e++) if (cnt[e] > lo && cnt[e] <= hi)
        for (uint32_t t0 = 0; t0 < cnt[e]; t0 += bn) { h[k].e = (int32_t)e; h[k].t0 = (int32_t)t0; h[k].nt = (int32_t)(cnt[e] - t0 < bn ? cnt[e] - t0 : bn); k++; }
    CK(cudaMalloc(dev, (size_t)n * sizeof(vqp_item))); CK(cudaMemcpy(*dev, h, (size_t)n * sizeof(vqp_item), cudaMemcpyHostToDevice)); free(h);
    return n;
}
__global__ static void fill_bf16_kernel(uint16_t *x, uint64_t n, uint32_t seed) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = (uint16_t)(0x3F00u | (h32(seed, (uint32_t)(i >> 32), (uint32_t)i) & 0x807Fu));   /* ±[0.5, 1) */
}
__global__ static void fill_f32_kernel(float *x, uint64_t n, uint32_t seed) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = ((float)(h32(seed, (uint32_t)(i >> 32), (uint32_t)i) & 0xFFFFu) / 32768.f - 1.f) * 1e-3f;
}
__global__ static void stream_kernel(const uint4 *p, uint64_t n, uint32_t *sink) {   /* 字节墙: 整段顺序读 */
    uint32_t a = 0;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) { const uint4 v = __ldg(p + i); a ^= v.x ^ v.y ^ v.z ^ v.w; }
    if (a == 0x12345678u) *sink = a;
}

static int g_nsm = 0;
/* 每个核实例的每 SM block 数只问一次(计时区里不掺主机侧的属性调用) */
static struct { const void *k; size_t shb; int occ; } g_occ[64]; static int g_nocc = 0;
template <typename F> static int grid_of(F kern, uint32_t threads, size_t shb, uint32_t nwork) {
    int occ = -1;
    for (int i = 0; i < g_nocc; i++) if (g_occ[i].k == (const void *)kern && g_occ[i].shb == shb) occ = g_occ[i].occ;
    if (occ < 0) {
        CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shb));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, (int)threads, shb));
        if (g_nocc < 64) { g_occ[g_nocc].k = (const void *)kern; g_occ[g_nocc].shb = shb; g_occ[g_nocc].occ = occ; g_nocc++; }
    }
    const uint32_t gsz = (uint32_t)g_nsm * (uint32_t)(occ > 0 ? occ : 1);
    return (int)(gsz < nwork ? gsz : nwork);
}
/* 一层的数据面 */
typedef struct {
    const layer_blob *L; uint32_t *cnt, nv; uint32_t *doff;
    uint16_t *x, *h16; float *g32, *hu, *ys;        /* 前向: x[nv][IN] → g32/h16/hu[nv][MID] → ys[nv][OUT] */
    float *go, *ghg, *ghu, *ga, *gx; uint16_t *g16; int *bad;   /* 转置: go[nv][OUT] → ga[nv][MID]; ghg/ghu[nv][MID] → gx[nv][IN] */
} plane;

#define MTM_NEW 2u
/* 前向三发: which 0/1/2。kind 0 = ref(BN128 全部专家); 3 = vqs 全部(按 NTM 切); 4 = vqs + ref(token 多于 NTM 的热专家给 ref) */
template <int EXT, int CB, int CBF = 0, uint32_t NTM = 32u, uint32_t RS = 2u, uint32_t S = 2u, int PF = 0>
static void fwd_layer(plane *P, int kind, vqp_item *const *it, const uint32_t *nit) {
    const uint32_t cbb = P->L->nc * (CB ? 16u : 8u), shr = cbb + VQM_TILE_BYTES;
    const uint32_t cbs = P->L->nc * (CBF ? 16u : 8u), shs = vqs_smem(cbs, EXT, NTM, RS, S);
    for (int mode = 0; mode < 3; mode++) {
        const uint32_t M = mode == 2 ? OUT : MID, K = mode == 2 ? MID : IN;
        const uint16_t *act = mode == 2 ? P->h16 : P->x;
        float *hu = mode == 1 ? P->hu : NULL;
        if (kind != 3 && nit[0]) {   /* ref 核: it[0] = BN128 切的(kind 0 全部专家, kind 4 只有热专家) */
            const uint32_t nw = nit[0] * ((M + VQM_BM - 1u) / VQM_BM);
#define REFL(MD) { const int gsz = grid_of(vqm_kernel<EXT, MD, CB>, VQM_THREADS, shr, nw); \
            vqm_kernel<EXT, MD, CB><<<gsz, VQM_THREADS, shr>>>(P->g32, P->h16, MD == 1 ? hu : P->ys, P->L->d, it[0], nit[0], act, P->doff, M, K, 30.f, cbb, NULL); }
            if (mode == 0) REFL(0) else if (mode == 1) REFL(1) else REFL(2)
#undef REFL
        }
        if (kind >= 3 && nit[1]) {
            const uint32_t nw = nit[1] * (M / VQS_BM);
#define STGL(MD) { const int gsz = grid_of(vqs_kernel<EXT, MD, CBF, NTM, RS, S, PF>, VQS_THREADS, shs, nw); \
            vqs_kernel<EXT, MD, CBF, NTM, RS, S, PF><<<gsz, VQS_THREADS, shs>>>(P->g32, P->h16, MD == 1 ? hu : P->ys, P->L->d, it[1], nit[1], act, P->doff, M, K, 30.f, cbs, NULL); }
            if (mode == 0) STGL(0) else if (mode == 1) STGL(1) else STGL(2)
#undef STGL
        }
        CK(cudaGetLastError());
    }
}
/* 转置三发(同引擎 ds4_gpu_bwd_routed_moe_tensor 的次序): G_A = G_O·W2; G_X = G_Hg·W1 + G_Hu·W3。
 * kind 0 = ref, 1 = vqrt, 2 = vqrt + ref 热; 3 = vqst 全部(按 NTM 切), 4 = vqst + ref 热(码本 CBF、token 上限 NTM) */
template <int EXT, int CB, int CBF = 1, uint32_t NTM = 32u>
static void bwd_layer(plane *P, int kind, vqp_item *const *it, const uint32_t *nit, vqp_item *pre_it, uint32_t pre_n) {
    const uint32_t cbb = P->L->nc * (CB ? 16u : 8u), shr = cbb + VQT_TILE_BYTES, shn = cbb;
    if (kind >= 3) {
        const uint32_t cbt = P->L->nc * (CBF ? 16u : 8u), sht = vqst_smem(cbt, EXT, NTM, 2u);
        const struct { uint32_t which, R, C; const float *g; float *out; int acc; } J[3] = {
            { 2u, OUT, MID, P->go, P->ga, 0 }, { 0u, MID, IN, P->ghg, P->gx, 0 }, { 1u, MID, IN, P->ghu, P->gx, 1 } };
        for (int k = 0; k < 3; k++) {
            vqt_prescale_kernel<<<pre_n, 256>>>(P->g16, J[k].g, P->L->d, pre_it, P->doff, J[k].which, J[k].R, J[k].C, NULL, OUT, P->bad);
            if (kind == 4 && nit[0]) {
                const uint32_t nw = nit[0] * ((J[k].C + VQT_BM - 1u) / VQT_BM);
                const int gsz = grid_of(vqt_kernel<EXT, CB>, VQM_THREADS, shr, nw);
                vqt_kernel<EXT, CB><<<gsz, VQM_THREADS, shr>>>(J[k].out, P->g16, P->L->d, it[0], nit[0], P->doff, J[k].which, J[k].R, J[k].C, NULL, OUT, J[k].acc, cbb, P->bad);
            }
            if (nit[1]) {
                const uint32_t ns = J[k].C / 64u; uint32_t sw = 16u; while (ns % sw) sw--;   /* 一个单元几条: C/64 的不超过 16 的最大因子 */
                const uint32_t nw = nit[1] * (ns / sw);
                const int gsz = grid_of(vqst_kernel<EXT, CBF, NTM, 2u>, VQS_THREADS, sht, nw);
                vqst_kernel<EXT, CBF, NTM, 2u><<<gsz, VQS_THREADS, sht>>>(J[k].out, P->g16, P->L->d, it[1], nit[1], P->doff, J[k].which, J[k].R, J[k].C, sw, J[k].acc, cbt, P->bad);
            }
            CK(cudaGetLastError());
        }
        return;
    }
    const struct { uint32_t which, R, C; const float *g; float *out; int acc; } J[3] = {
        { 2u, OUT, MID, P->go, P->ga, 0 }, { 0u, MID, IN, P->ghg, P->gx, 0 }, { 1u, MID, IN, P->ghu, P->gx, 1 } };
    for (int k = 0; k < 3; k++) {
        vqt_prescale_kernel<<<pre_n, 256>>>(P->g16, J[k].g, P->L->d, pre_it, P->doff, J[k].which, J[k].R, J[k].C, NULL, OUT, P->bad);
        if (kind != 1 && nit[0]) {
            const uint32_t nw = nit[0] * ((J[k].C + VQT_BM - 1u) / VQT_BM);
            const int gsz = grid_of(vqt_kernel<EXT, CB>, VQM_THREADS, shr, nw);
            vqt_kernel<EXT, CB><<<gsz, VQM_THREADS, shr>>>(J[k].out, P->g16, P->L->d, it[0], nit[0], P->doff, J[k].which, J[k].R, J[k].C, NULL, OUT, J[k].acc, cbb, P->bad);
        }
        if (kind != 0 && nit[1]) {
            const uint32_t nw = nit[1] * (J[k].C / 64u);
            const int gsz = grid_of(vqrt_kernel<EXT, CB, MTM_NEW>, VQR_THREADS, shn, (nw + VQR_WARPS - 1u) / VQR_WARPS);
            vqrt_kernel<EXT, CB, MTM_NEW><<<gsz, VQR_THREADS, shn>>>(J[k].out, P->g16, P->L->d, it[1], nit[1], P->doff, J[k].which, J[k].R, J[k].C, NULL, OUT, J[k].acc, cbb, P->bad);
        }
        CK(cudaGetLastError());
    }
}
static int same(const void *dev, const void *ref_h, size_t n, void *tmp) { CK(cudaMemcpy(tmp, dev, n, cudaMemcpyDeviceToHost)); return memcmp(tmp, ref_h, n) == 0; }
/* 热专家核 vqh: 冷专家(≤ 32 token)给 vqs<…, VS_RS, VS_S, PF2>, 热专家(> 32)按 64 切给 vqh<CBF, RS, S>; 前向三发逐位门 + 计时(合起来 / 只热的) */
template <int EXT, int CB, uint32_t VS_RS, uint32_t VS_S, int CBF, uint32_t RS, uint32_t S>
static int hot_variant(plane *P, const float *const *refs, const uint16_t *rh, void *tmp, uint32_t iters, double wall) {
    const uint64_t nm = (uint64_t)P->nv * MID, no = (uint64_t)P->nv * OUT;
    vqp_item *ic, *ih; const uint32_t nc_ = make_items(P->cnt, 32u, 0u, 32u, &ic), nh = make_items(P->cnt, VQH_NTM, 32u, 0xFFFFFFFFu, &ih);
    const uint32_t cbv = P->L->nc * 8u, shv = vqs_smem(cbv, EXT, 32u, VS_RS, VS_S), cbh = P->L->nc * (CBF ? 16u : 8u), shh = vqh_smem(cbh, EXT, RS, S);
    auto go = [&](bool cold, bool hot) {
        for (int mode = 0; mode < 3; mode++) {
            const uint32_t M = mode == 2 ? OUT : MID, K = mode == 2 ? MID : IN;
            const uint16_t *act = mode == 2 ? P->h16 : P->x;
            float *hu = mode == 1 ? P->hu : NULL;
#define CL(MD) { const int gsz = grid_of(vqs_kernel<EXT, MD, 0, 32u, VS_RS, VS_S, 2>, VQS_THREADS, shv, nc_ * (M / VQS_BM)); \
            vqs_kernel<EXT, MD, 0, 32u, VS_RS, VS_S, 2><<<gsz, VQS_THREADS, shv>>>(P->g32, P->h16, MD == 1 ? hu : P->ys, P->L->d, ic, nc_, act, P->doff, M, K, 30.f, cbv, NULL); }
#define HL(MD) { const int gsz = grid_of(vqh_kernel<EXT, MD, CBF, RS, S>, VQH_THREADS, shh, nh * (M / VQS_BM)); \
            vqh_kernel<EXT, MD, CBF, RS, S><<<gsz, VQH_THREADS, shh>>>(P->g32, P->h16, MD == 1 ? hu : P->ys, P->L->d, ih, nh, act, P->doff, M, K, 30.f, cbh, NULL); }
            if (cold && nc_) { if (mode == 0) CL(0) else if (mode == 1) CL(1) else CL(2) }
            if (hot && nh) { if (mode == 0) HL(0) else if (mode == 1) HL(1) else HL(2) }
#undef CL
#undef HL
            CK(cudaGetLastError());
        }
    };
    CK(cudaMemset(P->g32, 0xFF, nm * 4)); CK(cudaMemset(P->h16, 0xFF, nm * 2)); CK(cudaMemset(P->hu, 0xFF, nm * 4)); CK(cudaMemset(P->ys, 0xFF, no * 4));
    go(true, true); CK(cudaDeviceSynchronize());
    const int f = same(P->g32, refs[0], nm * 4, tmp) && same(P->h16, rh, nm * 2, tmp) && same(P->hu, refs[1], nm * 4, tmp) && same(P->ys, refs[2], no * 4, tmp);
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    float ta = 1e30f, th = 1e30f, ms;
    for (uint32_t r = 0; r < iters; r++) {
        CK(cudaEventRecord(t0)); go(true, true); CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1)); CK(cudaEventElapsedTime(&ms, t0, t1)); ta = ms < ta ? ms : ta;
        CK(cudaEventRecord(t0)); go(false, true); CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1)); CK(cudaEventElapsedTime(&ms, t0, t1)); th = ms < th ? ms : th;
    }
    printf("   vqs 冷(RS%u) + vqh 热(cb%s RS%u S%u): 前向三发 %6.2f ms(墙的 %.2f×), 其中热 %u 项 %.2f ms 逐位门 %s\n",
           VS_RS, CBF ? "bf16" : "E4M3", RS, S, ta, ta / wall, nh, th, f ? "✓" : "★不同★");
    cudaFree(ic); cudaFree(ih);
    return !f;
}
/* vqst 一个形态: 转置三发逐位门(G_A / G_X 对 ref)+ 计时; kind 3 全部按 NTM 切, 4 = 热专家给 ref */
template <int EXT, int CB, int CBF, uint32_t NTM>
static int tst_variant(plane *P, const float *rga, const float *rgx, void *tmp, uint32_t iters, double wall, vqp_item *pre, uint32_t npre) {
    const uint64_t nm = (uint64_t)P->nv * MID, nx = (uint64_t)P->nv * IN;
    vqp_item *ia[2], *ib[2]; uint32_t na[2], nb[2];
    na[0] = 0; ia[0] = NULL; na[1] = make_items(P->cnt, NTM, 0u, 0xFFFFFFFFu, &ia[1]);
    nb[0] = make_items(P->cnt, VQM_BN, NTM, 0xFFFFFFFFu, &ib[0]); nb[1] = make_items(P->cnt, NTM, 0u, NTM, &ib[1]);
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    int bad = 0;
    for (int kind = 3; kind <= 4; kind++) {
        vqp_item *const *it = kind == 3 ? ia : ib; const uint32_t *ni = kind == 3 ? na : nb;
        CK(cudaMemset(P->ga, 0xFF, nm * 4)); CK(cudaMemset(P->gx, 0xFF, nx * 4));
        bwd_layer<EXT, CB, CBF, NTM>(P, kind, it, ni, pre, npre); CK(cudaDeviceSynchronize());
        const int b = same(P->ga, rga, nm * 4, tmp) && same(P->gx, rgx, nx * 4, tmp);
        float bb = 1e30f;
        for (uint32_t r = 0; r < iters; r++) {
            float ms; CK(cudaEventRecord(t0)); bwd_layer<EXT, CB, CBF, NTM>(P, kind, it, ni, pre, npre); CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            CK(cudaEventElapsedTime(&ms, t0, t1)); if (ms < bb) bb = ms;
        }
        printf("   vqst cb%s NTM%u %-6s 转置三发(含预缩放) %6.2f ms(墙的 %.2f×) 逐位门 %s\n", CBF ? "bf16" : "E4M3", NTM, kind == 3 ? "全部" : "+ref热", bb, bb / wall, b ? "✓" : "★不同★");
        if (!b) {
            CK(cudaMemcpy(tmp, P->gx, nx * 4, cudaMemcpyDeviceToHost)); double d2 = 0, r2 = 0; const float *t = (const float *)tmp;
            for (uint64_t i = 0; i < nx; i++) { const double d = (double)t[i] - rgx[i]; d2 += d * d; r2 += (double)rgx[i] * rgx[i]; }
            printf("     G_X 相对 L2 %.3e\n", sqrt(d2 / (r2 + 1e-30)));
        }
        bad |= !b;
    }
    if (ia[1]) cudaFree(ia[1]); if (ib[0]) cudaFree(ib[0]); if (ib[1]) cudaFree(ib[1]);
    return bad;
}
/* vqs 一个形态: kind 3(全部按 NTM 切给 vqs)与 kind 4(热专家给 ref)各过一遍前向逐位门再计时; refs = ref 的 g32/H_u/ys 主机副本 */
template <int EXT, int CB, int CBF, uint32_t NTM, uint32_t RS, uint32_t S, int PF = 0>
static int stg_variant(plane *P, const float *const *refs, const uint16_t *rh, void *tmp, uint32_t iters, double wall) {
    const uint64_t nm = (uint64_t)P->nv * MID, no = (uint64_t)P->nv * OUT;
    vqp_item *ia[2], *ib[2]; uint32_t na[2], nb[2];
    na[0] = 0; ia[0] = NULL; na[1] = make_items(P->cnt, NTM, 0u, 0xFFFFFFFFu, &ia[1]);
    nb[0] = make_items(P->cnt, VQM_BN, NTM, 0xFFFFFFFFu, &ib[0]); nb[1] = make_items(P->cnt, NTM, 0u, NTM, &ib[1]);
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    int bad = 0;
    char name[64]; snprintf(name, sizeof name, "vqs cb%s NTM%u RS%u S%u PF%d", CBF ? "bf16" : "E4M3", NTM, RS, S, PF);
    for (int kind = 3; kind <= 4; kind++) {
        vqp_item *const *it = kind == 3 ? ia : ib; const uint32_t *ni = kind == 3 ? na : nb;
        CK(cudaMemset(P->g32, 0xFF, nm * 4)); CK(cudaMemset(P->h16, 0xFF, nm * 2)); CK(cudaMemset(P->hu, 0xFF, nm * 4)); CK(cudaMemset(P->ys, 0xFF, no * 4));
        fwd_layer<EXT, CB, CBF, NTM, RS, S, PF>(P, kind, it, ni); CK(cudaDeviceSynchronize());
        const int f = same(P->g32, refs[0], nm * 4, tmp) && same(P->h16, rh, nm * 2, tmp) && same(P->hu, refs[1], nm * 4, tmp) && same(P->ys, refs[2], no * 4, tmp);
        float bf = 1e30f;
        for (uint32_t r = 0; r < iters; r++) {
            float ms; CK(cudaEventRecord(t0)); fwd_layer<EXT, CB, CBF, NTM, RS, S, PF>(P, kind, it, ni); CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            CK(cudaEventElapsedTime(&ms, t0, t1)); if (ms < bf) bf = ms;
        }
        printf("   %-26s %-6s 前向三发 %6.2f ms(墙的 %.2f×, 项 %u+%u) 逐位门 %s\n", name, kind == 3 ? "全部" : "+ref热", bf, bf / wall,
               kind == 3 ? 0u : nb[0], kind == 3 ? na[1] : nb[1], f ? "✓" : "★不同★");
        bad |= !f;
    }
    {   /* 拆账: 冷专家(≤ NTM)单给 vqs、热专家单给 ref、热专家按 NTM 切给 vqs —— 看两类各吃多少 */
        vqp_item *hs; const uint32_t nhs = make_items(P->cnt, NTM, NTM, 0xFFFFFFFFu, &hs);
        vqp_item *const cold[2] = { NULL, ib[1] }, *const hot_r[2] = { ib[0], NULL }, *const hot_s[2] = { NULL, hs };
        const uint32_t ncold[2] = { 0u, nb[1] }, nhot_r[2] = { nb[0], 0u }, nhot_s[2] = { 0u, nhs };
        float tc = 1e30f, tr = 1e30f, ts = 1e30f, ms;
        for (uint32_t r = 0; r < iters; r++) {
            CK(cudaEventRecord(t0)); fwd_layer<EXT, CB, CBF, NTM, RS, S, PF>(P, 3, cold, ncold); CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1)); CK(cudaEventElapsedTime(&ms, t0, t1)); tc = ms < tc ? ms : tc;
            CK(cudaEventRecord(t0)); fwd_layer<EXT, CB, CBF, NTM, RS, S, PF>(P, 4, hot_r, nhot_r); CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1)); CK(cudaEventElapsedTime(&ms, t0, t1)); tr = ms < tr ? ms : tr;
            CK(cudaEventRecord(t0)); fwd_layer<EXT, CB, CBF, NTM, RS, S, PF>(P, 3, hot_s, nhot_s); CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1)); CK(cudaEventElapsedTime(&ms, t0, t1)); ts = ms < ts ? ms : ts;
        }
        printf("   %-26s 拆账: 冷 %u 项(vqs) %.2f ms | 热 %u 项(ref) %.2f ms | 热 %u 项(vqs 切 %u) %.2f ms\n", name, nb[1], tc, nb[0], tr, nhs, NTM, ts);
        if (hs) cudaFree(hs);
    }
    if (ia[1]) cudaFree(ia[1]); if (ib[0]) cudaFree(ib[0]); if (ib[1]) cudaFree(ib[1]);
    return bad;
}

template <int EXT, int CB>
static int run_layer(const layer_blob *L, uint32_t n_tok, const char *route, uint32_t iters) {
    plane P; memset(&P, 0, sizeof P); P.L = L;
    P.cnt = (uint32_t *)malloc(NE * 4); make_counts(n_tok, route, P.cnt);
    uint32_t off[NE + 1]; off[0] = 0; for (uint32_t e = 0; e < NE; e++) off[e + 1] = off[e] + P.cnt[e];
    P.nv = off[NE];
    uint32_t nz = 0, mx = 0, big = 0; uint64_t bytes = 0;
    const uint64_t mbytes = (uint64_t)MID * (IN / 8u * 12u / 8u + (EXT ? IN / 64u : 0u)) * 2u + (uint64_t)OUT * (MID / 8u * 12u / 8u + (EXT ? MID / 64u : 0u));
    for (uint32_t e = 0; e < NE; e++) if (P.cnt[e]) { nz++; bytes += mbytes; if (P.cnt[e] > mx) mx = P.cnt[e]; if (P.cnt[e] > 16u * MTM_NEW) big++; }
    printf("\n== %u 位层: %u token → %u 对, 有 token 的专家 %u, 最热 %u, > %u 的 %u 个; 三块位流 %.2f GB ==\n", L->nbit, n_tok, P.nv, nz, mx, 16u * MTM_NEW, big, (double)bytes / 1e9);
    CK(cudaMalloc(&P.doff, (NE + 1) * 4)); CK(cudaMemcpy(P.doff, off, (NE + 1) * 4, cudaMemcpyHostToDevice));
    const uint64_t nx = (uint64_t)P.nv * IN, nm = (uint64_t)P.nv * MID, no = (uint64_t)P.nv * OUT;
    CK(cudaMalloc(&P.x, nx * 2)); CK(cudaMalloc(&P.h16, nm * 2)); CK(cudaMalloc(&P.g32, nm * 4)); CK(cudaMalloc(&P.hu, nm * 4)); CK(cudaMalloc(&P.ys, no * 4));
    CK(cudaMalloc(&P.go, no * 4)); CK(cudaMalloc(&P.ghg, nm * 4)); CK(cudaMalloc(&P.ghu, nm * 4)); CK(cudaMalloc(&P.ga, nm * 4)); CK(cudaMalloc(&P.gx, nx * 4));
    CK(cudaMalloc(&P.g16, (no > nm ? no : nm) * 2)); CK(cudaMalloc(&P.bad, 4)); CK(cudaMemset(P.bad, 0, 4));
    fill_bf16_kernel<<<(unsigned)((nx + 255) / 256), 256>>>(P.x, nx, 11u);
    fill_f32_kernel<<<(unsigned)((no + 255) / 256), 256>>>(P.go, no, 12u);
    fill_f32_kernel<<<(unsigned)((nm + 255) / 256), 256>>>(P.ghg, nm, 13u);
    fill_f32_kernel<<<(unsigned)((nm + 255) / 256), 256>>>(P.ghu, nm, 14u);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    /* 三种分工的工作项: ref 全部按 128 切; new 全部按 16·MTM 切; mix = 热专家按 128 给 ref, 其余按 16·MTM 给 new */
    const uint32_t SB = 16u * MTM_NEW;
    vqp_item *ir[2], *in_[2], *im[2], *pre; uint32_t nr[2], nn[2], nmx[2];
    nr[0] = make_items(P.cnt, VQM_BN, 0u, 0xFFFFFFFFu, &ir[0]); nr[1] = 0; ir[1] = NULL;
    nn[0] = 0; in_[0] = NULL; nn[1] = make_items(P.cnt, SB, 0u, 0xFFFFFFFFu, &in_[1]);
    nmx[0] = make_items(P.cnt, VQM_BN, SB, 0xFFFFFFFFu, &im[0]); nmx[1] = make_items(P.cnt, SB, 0u, SB, &im[1]);
    const uint32_t npre = nr[0]; pre = ir[0];
    printf("工作项: ref %u | new %u | mix %u(ref) + %u(new)\n", nr[0], nn[1], nmx[0], nmx[1]);
    /* ① 逐位门: ref 出一份主机副本, new / mix 各跑一遍逐字节比 */
    float *rg = (float *)malloc(nm * 4), *rhu = (float *)malloc(nm * 4), *ry = (float *)malloc(no * 4), *rga = (float *)malloc(nm * 4), *rgx = (float *)malloc(nx * 4);
    uint16_t *rh = (uint16_t *)malloc(nm * 2); void *tmp = malloc(nx * 4 > no * 4 ? nx * 4 : no * 4);
    fwd_layer<EXT, CB>(&P, 0, ir, nr); bwd_layer<EXT, CB>(&P, 0, ir, nr, pre, npre); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(rg, P.g32, nm * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(rh, P.h16, nm * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(rhu, P.hu, nm * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(ry, P.ys, no * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(rga, P.ga, nm * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(rgx, P.gx, nx * 4, cudaMemcpyDeviceToHost));
    int badall = 0;
    for (int kind = 1; kind <= 2; kind++) {
        vqp_item *const *it = kind == 1 ? in_ : im; const uint32_t *ni = kind == 1 ? nn : nmx;
        CK(cudaMemset(P.g32, 0xFF, nm * 4)); CK(cudaMemset(P.h16, 0xFF, nm * 2)); CK(cudaMemset(P.hu, 0xFF, nm * 4)); CK(cudaMemset(P.ys, 0xFF, no * 4));
        CK(cudaMemset(P.ga, 0xFF, nm * 4)); CK(cudaMemset(P.gx, 0xFF, nx * 4));
        bwd_layer<EXT, CB>(&P, kind, it, ni, pre, npre); CK(cudaDeviceSynchronize());   /* 转置的输入(g)与前向无关 */
        const int f = 1;
        const int b = same(P.ga, rga, nm * 4, tmp) && same(P.gx, rgx, nx * 4, tmp);
        printf("① 逐位门 %s vs ref: 转置(G_A/G_X) %s\n", kind == 1 ? "vqrt" : "vqrt+ref热", b ? "逐位同 ✓" : "★不同★");
        if (!f || !b) {   /* 不同就报差多少(相对 L2 与最大相对差), 看是舍入级还是错位级 */
            CK(cudaMemcpy(tmp, P.ys, no * 4, cudaMemcpyDeviceToHost)); double d2 = 0, r2 = 0, mxr = 0; const float *t = (const float *)tmp;
            for (uint64_t i = 0; i < no; i++) { const double d = (double)t[i] - ry[i]; d2 += d * d; r2 += (double)ry[i] * ry[i]; const double rr = fabs(d) / (fabs(ry[i]) + 1e-6); if (rr > mxr) mxr = rr; }
            printf("   ys 相对 L2 %.3e, 最大相对差 %.3e\n", sqrt(d2 / (r2 + 1e-30)), mxr);
            CK(cudaMemcpy(tmp, P.gx, nx * 4, cudaMemcpyDeviceToHost)); d2 = r2 = mxr = 0;
            for (uint64_t i = 0; i < nx; i++) { const double d = (double)t[i] - rgx[i]; d2 += d * d; r2 += (double)rgx[i] * rgx[i]; const double rr = fabs(d) / (fabs(rgx[i]) + 1e-9); if (rr > mxr) mxr = rr; }
            printf("   G_X 相对 L2 %.3e, 最大相对差 %.3e\n", sqrt(d2 / (r2 + 1e-30)), mxr);
            badall = 1;
        }
    }
    int hbad = 0; CK(cudaMemcpy(&hbad, P.bad, 4, cudaMemcpyDeviceToHost)); if (hbad) { printf("★载荷头不认★\n"); badall = 1; }
    /* ② 计时: 每种取 iters 遍里最快的 */
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    uint32_t *sink; CK(cudaMalloc(&sink, 4));
    float best = 1e30f;
    for (uint32_t r = 0; r < iters; r++) {
        CK(cudaEventRecord(t0)); stream_kernel<<<g_nsm * 8, 512>>>((const uint4 *)L->d, L->bytes / 16u, sink); CK(cudaEventRecord(t1));
        CK(cudaEventSynchronize(t1)); float ms; CK(cudaEventElapsedTime(&ms, t0, t1)); if (ms < best) best = ms;
    }
    const double wall = best * (double)bytes / (double)L->bytes;
    printf("② 字节墙: 整层 blob %.2f GB 顺序读 %.2f ms(%.0f GB/s) ⇒ 有 token 专家三块 %.2f ms/层, 每发 ~%.2f ms\n",
           (double)L->bytes / 1e9, best, (double)L->bytes / 1e6 / best, wall, wall / 3.0);
    for (int kind = 0; kind <= 2; kind++) {
        vqp_item *const *it = kind == 0 ? ir : kind == 1 ? in_ : im; const uint32_t *ni = kind == 0 ? nr : kind == 1 ? nn : nmx;
        float bf = 1e30f, bb = 1e30f;
        for (uint32_t r = 0; r < iters; r++) {
            float ms;
            if (kind == 0) { CK(cudaEventRecord(t0)); fwd_layer<EXT, CB>(&P, 0, it, ni); CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
                             CK(cudaEventElapsedTime(&ms, t0, t1)); if (ms < bf) bf = ms; }
            CK(cudaEventRecord(t0)); bwd_layer<EXT, CB>(&P, kind, it, ni, pre, npre); CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            CK(cudaEventElapsedTime(&ms, t0, t1)); if (ms < bb) bb = ms;
        }
        if (kind == 0) printf("   ref  前向三发 %6.2f ms(墙的 %.2f×) | 转置三发(含预缩放) %6.2f ms(墙的 %.2f×)\n", bf, bf / wall, bb, bb / wall);
        else printf("   %-10s 转置三发(含预缩放) %6.2f ms(墙的 %.2f×)\n", kind == 1 ? "vqrt" : "vqrt+ref热", bb, bb / wall);
    }
    /* ③ 分段预取版 vqs(前向): 码本 E4M3 现转(CBF 0)/ 进 shared 时转 bf16(CBF 1, 只 12 位层放得下); 一段 RS 轮, S 段在途 */
    const float *refs[3] = { rg, rhu, ry };
    /* PF 1 = 每个单元开头把本单元整段位流 bulk 预取进 L2(段内 cp.async 只读 L2) */
    /* 引擎现用的两个形态(vqs 前向 / vqst 转置)+ 热专家核 vqh 的候选 */
    if (EXT) {
        badall |= stg_variant<EXT, CB, 0, 32u, 2u, 2u, 2>(&P, refs, rh, tmp, iters, wall);
        badall |= hot_variant<EXT, CB, 2u, 2u, 0, 1u, 2u>(&P, refs, rh, tmp, iters, wall);
        badall |= tst_variant<EXT, CB, 0, 32u>(&P, rga, rgx, tmp, iters, wall, pre, npre);
    } else {
        badall |= stg_variant<EXT, CB, 0, 32u, 4u, 2u, 2>(&P, refs, rh, tmp, iters, wall);
        badall |= hot_variant<EXT, CB, 4u, 2u, 0, 2u, 2u>(&P, refs, rh, tmp, iters, wall);
        badall |= hot_variant<EXT, CB, 4u, 2u, 1, 1u, 2u>(&P, refs, rh, tmp, iters, wall);
        badall |= tst_variant<EXT, CB, 1, 24u>(&P, rga, rgx, tmp, iters, wall, pre, npre);
    }
    return badall;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "用法: %s GGUF [12 位层=20] [13 位层=5] [n_tok=880] [路由文件] [遍数=5]\n", argv[0]); return 2; }
    const uint32_t l12 = argc > 2 ? (uint32_t)atoi(argv[2]) : 20u, l13 = argc > 3 ? (uint32_t)atoi(argv[3]) : 5u;
    const uint32_t n_tok = argc > 4 ? (uint32_t)atoi(argv[4]) : 880u, iters = argc > 6 ? (uint32_t)atoi(argv[6]) : 5u;
    const char *route = argc > 5 && strcmp(argv[5], "-") ? argv[5] : NULL;
    { cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0)); g_nsm = pr.multiProcessorCount;
      printf("板子: %s, SM %d, 每 block 可批 shared %zu KB, L2 %d MB; 路由 %s\n", pr.name, g_nsm, pr.sharedMemPerBlockOptin >> 10, pr.l2CacheSize >> 20, route ? route : "哈希均匀"); }
    ds4_gguf g; char err[256];
    if (ds4_gguf_open(&g, argv[1], err, sizeof err) != 0) { fprintf(stderr, "★GGUF 打不开: %s★\n", err); return 2; }
    layer_blob L12 = load_layer(&g, l12), L13 = load_layer(&g, l13);
    ds4_gguf_close(&g);
    if (L12.nbit != 12u || L13.nbit != 13u) { fprintf(stderr, "★层号给反了: 第 2 个参数要 12 位层, 第 3 个要 13 位层★\n"); return 2; }
    int bad = run_layer<0, 1>(&L12, n_tok, route, iters);
    bad |= run_layer<1, 0>(&L13, n_tok, route, iters);
    printf("\n%s\n", bad ? "★★逐位门有红, 新核的数字不作数★★" : "逐位门全绿(new / mix == ref)");
    return bad;
}
