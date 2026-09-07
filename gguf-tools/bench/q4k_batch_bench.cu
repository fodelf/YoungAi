/* q4k_batch_bench.cu — 小批(投机 verify 2..8 token)Q4_K 稠密核的形态微基准(spark 本机跑, 2026-09-07)。
 *
 * 【为什么要它】引擎 09-07 落地的"权重一遍多 token"核(cuda_q4k_tile: q4k_tile_multi / q4k_rows_multi)在 4 token verify
 * 批里只跑 136 GB/s(单 token tile 核 200+, 墙 233~238): 一 tile 权重 stage 完要做 4 趟点积+归约, 这段里该 warp 没有
 * 访存在飞。候选 = 双缓冲预取(cp.async 到备用 shared 段 / 寄存器环), 让下一 tile 的读与本 tile 的 4 趟点积重叠。
 * 与 q4k_gemv_bench 同法: 真实形状随机 Q4_K + N 份 q8_K 激活, 8 份矩阵轮换压过 L2, 候选逐位对拍"单 token 核逐 token
 * 各跑一遍"的参考(同 dot 同归约树 ⇒ 必须逐位同), 量 µs/GB/s。
 * 用法: q4k_batch_bench [iters=20]   (nvcc -O3 -arch=native -o q4k_batch_bench q4k_batch_bench.cu)
 * 设备侧辅助逐字抄自引擎(cuda_api_moe_corr_1 / cuda_q4k_tile) —— 只为测速, 引擎才是真值。 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <time.h>

#include "q4k_batch_bench_kernels.cuh"   /* 设备侧核族(拆出以守 500 行) */
static void run_rows8(int iters) {
    const int NCOPY = 8; const uint32_t kb = 32u, rows = 4096u, NT = 8u;
    const uint64_t mbytes = (uint64_t)rows * kb * 144u;
    uint8_t *h = (uint8_t *)malloc(mbytes); fill_rand(h, mbytes, 77u);
    int8_t *hxq = (int8_t *)malloc(NT * kb * 256u); float *hxs = (float *)malloc(NT * kb * 8u * 4u);
    fill_rand((uint8_t *)hxq, NT * kb * 256u, 6u); for (uint32_t i = 0; i < NT * kb * 8u; i++) hxs[i] = 0.01f + (float)(i % 5) * 0.002f;
    uint8_t *dw[8]; int8_t *dxq; float *dxs, *o_ref, *o_new;
    for (int c = 0; c < NCOPY; c++) { CK(cudaMalloc(&dw[c], mbytes)); CK(cudaMemcpy(dw[c], h, mbytes, cudaMemcpyHostToDevice)); }
    CK(cudaMalloc(&dxq, NT * kb * 256u)); CK(cudaMemcpy(dxq, hxq, NT * kb * 256u, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dxs, NT * kb * 32u)); CK(cudaMemcpy(dxs, hxs, NT * kb * 32u, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&o_ref, NT * rows * 4)); CK(cudaMalloc(&o_new, NT * rows * 4));
    const size_t shm_w = (size_t)8u * kb * 9u * 16u;
    unsigned gx = (rows + 7u) / 8u; if (gx > 384u) gx = 384u;
    float *ref = (float *)malloc(NT * rows * 4), *got = (float *)malloc(NT * rows * 4);
    printf("== out_b 4096x8192(32) q8_0 口径 %.1f MB grid %u\n", mbytes / 1e6, gx);
    const uint32_t ntoks[] = { 1u, 2u, 4u, 8u };
    for (uint32_t n : ntoks) {
        const size_t shm_p = shm_w + (size_t)n * kb * (256u + 32u + 32u);
        CK(cudaFuncSetAttribute(k_rows8_pre, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm_p));
        k_rows8_cur<<<gx, 256, shm_w>>>(o_ref, (const char *)dw[0], dxq, dxs, kb, rows, n); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(ref, o_ref, n * rows * 4, cudaMemcpyDeviceToHost));
        auto bench = [&](const char *nm, auto fn) {
            for (int i = 0; i < 3; i++) fn(i); CK(cudaDeviceSynchronize()); double t0 = now_s();
            for (int i = 0; i < iters; i++) fn(i); CK(cudaDeviceSynchronize()); double dt = (now_s() - t0) / iters;
            fn(0); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(got, o_new, n * rows * 4, cudaMemcpyDeviceToHost));
            uint32_t bad = 0; for (uint32_t r = 0; r < n * rows; r++) if (memcmp(&ref[r], &got[r], 4)) bad++;
            printf("  N=%u %-22s %7.1f us  %6.1f GB/s(权重)  %s%u\n", n, nm, dt * 1e6, mbytes / dt / 1e9, bad ? "★不同 " : "逐位同 ", bad);
        };
        bench("现役 q8_0 rows", [&](int i) { k_rows8_cur<<<gx, 256, shm_w>>>(o_new, (const char *)dw[i % NCOPY], dxq, dxs, kb, rows, n); });
        bench("预解码+进 shared", [&](int i) { k_rows8_pre<<<gx, 256, shm_p>>>(o_new, (const char *)dw[i % NCOPY], dxq, dxs, kb, rows, n); });
        CK(cudaFuncSetAttribute(k_rows8_pre_pdl, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm_p));
        bench("预解码 PDL 发射(引擎形态)", [&](int i) { k_dummy<<<1, 32>>>(o_new); launch_pdl_rows8(o_new, (const char *)dw[i % NCOPY], dxq, dxs, kb, rows, n, gx, shm_p); });
        bench("预解码 PDL grid192", [&](int i) { k_dummy<<<1, 32>>>(o_new); launch_pdl_rows8(o_new, (const char *)dw[i % NCOPY], dxq, dxs, kb, rows, n, 192u, shm_p); });
        CK(cudaFuncSetAttribute(k_rows8_engine, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm_p));
        bench("引擎核逐字(stage_x=1)", [&](int i) { k_rows8_engine<<<gx, 256, shm_p>>>(o_new, (const char *)dw[i % NCOPY], dxq, dxs, kb, rows, n, 1); });
        bench("引擎核逐字(stage_x=0)", [&](int i) { k_rows8_engine<<<gx, 256, shm_w>>>(o_new, (const char *)dw[i % NCOPY], dxq, dxs, kb, rows, n, 0); });
    }
    for (int c = 0; c < NCOPY; c++) cudaFree(dw[c]);
    cudaFree(dxq); cudaFree(dxs); cudaFree(o_ref); cudaFree(o_new); free(h); free(hxq); free(hxs); free(ref); free(got);
}

static void make_q8(blk_q8_K *b, uint32_t blocks, uint32_t seed) {
    uint32_t s = seed;
    for (uint32_t k = 0; k < blocks; k++) {
        b[k].d = 0.01f + (float)(k % 7) * 0.001f;
        for (int i = 0; i < QK_K; i++) { s = s * 1664525u + 1013904223u; b[k].qs[i] = (int8_t)((int)(s >> 24) - 128); }
        for (int j = 0; j < 16; j++) { int sum = 0; for (int i = 0; i < 16; i++) sum += b[k].qs[j * 16 + i]; b[k].bsums[j] = (int16_t)sum; }
    }
}
typedef struct { const char *name; uint32_t rows, blocks; } shape_t;

template <uint32_t BLOCKS>
static void run_shape(const shape_t &sh, int iters) {
    const int NCOPY = 8;
    const uint32_t rows = sh.rows, R = 32u / BLOCKS;
    const uint64_t mbytes = (uint64_t)rows * BLOCKS * 144u;
    uint8_t *h = (uint8_t *)malloc(mbytes); fill_rand(h, mbytes, 7u + rows);
    /* 激活: f32 随机 → 引擎同款量化核出 q8_K(所有变体共用 dx); 融合量化变体直接吃 f32, 必须逐位同 */
    const uint32_t nx = 8u * BLOCKS * 256u;
    float *hxf = (float *)malloc(nx * 4); { uint32_t s = 11u; for (uint32_t i = 0; i < nx; i++) { s = s * 1664525u + 1013904223u; hxf[i] = ((float)(s >> 8) / 16777216.0f - 0.5f) * 4.0f; } }
    uint8_t *dw[8]; float *o_ref, *o_new, *dxf; blk_q8_K *dx;
    for (int c = 0; c < NCOPY; c++) { CK(cudaMalloc(&dw[c], mbytes)); CK(cudaMemcpy(dw[c], h, mbytes, cudaMemcpyHostToDevice)); }
    CK(cudaMalloc(&dxf, nx * 4)); CK(cudaMemcpy(dxf, hxf, nx * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dx, 8u * BLOCKS * sizeof(blk_q8_K)));
    k_q8_quant<<<dim3(BLOCKS, 8u, 1), 256>>>(dx, dxf, BLOCKS * 256u, 8u); CK(cudaDeviceSynchronize());
    CK(cudaMalloc(&o_ref, 8u * rows * 4)); CK(cudaMalloc(&o_new, 8u * rows * 4));
    const size_t shm1 = (size_t)8u * 32u * 9u * 16u, shm2 = 2u * shm1, shm_xs = shm1 + 8u * BLOCKS * sizeof(blk_q8_K);
    CK(cudaFuncSetAttribute(k_multi_db<BLOCKS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm2));
    CK(cudaFuncSetAttribute(k_multi_xs<BLOCKS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm_xs));
    CK(cudaFuncSetAttribute(k_multi_fq<BLOCKS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm_xs));
    unsigned gx = (rows / R + 7u) / 8u; if (gx > 384u) gx = 384u;
    printf("== %s  %.1f MB  grid %u\n", sh.name, mbytes / 1e6, gx);
    float *ref = (float *)malloc(8u * rows * 4), *got = (float *)malloc(8u * rows * 4);
    const uint32_t ntoks[] = { 1u, 2u, 4u, 8u };
    for (uint32_t n : ntoks) {
        /* 参考: 单 token 核逐 token 发 n 次 */
        for (uint32_t tk = 0; tk < n; tk++) k_tile1<BLOCKS><<<gx, 256, shm1>>>(o_ref + tk * rows, (const char *)dw[0], dx + tk * BLOCKS, rows);
        CK(cudaDeviceSynchronize()); CK(cudaMemcpy(ref, o_ref, n * rows * 4, cudaMemcpyDeviceToHost));
        auto bench = [&](const char *nm, auto fn, size_t shm) {
            for (int i = 0; i < 3; i++) fn(i); CK(cudaDeviceSynchronize()); double t0 = now_s();
            for (int i = 0; i < iters; i++) fn(i); CK(cudaDeviceSynchronize()); double dt = (now_s() - t0) / iters;
            fn(0); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(got, o_new, n * rows * 4, cudaMemcpyDeviceToHost));
            uint32_t bad = 0; for (uint32_t r = 0; r < n * rows; r++) if (memcmp(&ref[r], &got[r], 4)) bad++;
            printf("  N=%u %-22s %7.1f us  %6.1f GB/s(权重)  %s%u\n", n, nm, dt * 1e6, mbytes / dt / 1e9, bad ? "★不同 " : "逐位同 ", bad);
            (void)shm;
        };
        bench("单核逐token×N", [&](int i) { for (uint32_t tk = 0; tk < n; tk++) k_tile1<BLOCKS><<<gx, 256, shm1>>>(o_new + tk * rows, (const char *)dw[i % NCOPY], dx + tk * BLOCKS, rows); }, shm1);
        bench("现役 multi", [&](int i) { k_multi<BLOCKS><<<gx, 256, shm1>>>(o_new, (const char *)dw[i % NCOPY], dx, rows, n); }, shm1);
        bench("cp.async 双缓冲", [&](int i) { k_multi_db<BLOCKS><<<gx, 256, shm2>>>(o_new, (const char *)dw[i % NCOPY], dx, rows, n); }, shm2);
        bench("激活进 shared", [&](int i) { k_multi_xs<BLOCKS><<<gx, 256, shm_xs>>>(o_new, (const char *)dw[i % NCOPY], dx, rows, n); }, shm_xs);
        bench("量化核+现役 multi", [&](int i) { k_q8_quant<<<dim3(BLOCKS, n, 1), 256>>>(dx, dxf, BLOCKS * 256u, n); k_multi<BLOCKS><<<gx, 256, shm1>>>(o_new, (const char *)dw[i % NCOPY], dx, rows, n); }, shm1);
        bench("量化融合 fq", [&](int i) { k_multi_fq<BLOCKS><<<gx, 256, shm_xs>>>(o_new, (const char *)dw[i % NCOPY], dxf, rows, n); }, shm_xs);
        if (n >= 4u) {
            const unsigned g2 = gx / 2u;
            bench("现役 multi grid/2", [&](int i) { k_multi<BLOCKS><<<g2, 256, shm1>>>(o_new, (const char *)dw[i % NCOPY], dx, rows, n); }, shm1);
        }
    }
    for (int c = 0; c < NCOPY; c++) cudaFree(dw[c]);
    cudaFree(dx); cudaFree(dxf); cudaFree(o_ref); cudaFree(o_new); free(h); free(hxf); free(ref); free(got);
}

int main(int argc, char **argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 20;
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    printf("%s SMs=%d L2=%d MB\n", pr.name, pr.multiProcessorCount, pr.l2CacheSize >> 20);
    run_shape<4u>({"q_b 32768x1024(4 blk)", 32768, 4}, iters);
    run_shape<8u>({"shexp_down 4096x2048(8)", 4096, 8}, iters);
    run_shape<16u>({"q_a+kv 1536x4096(16)", 1536, 16}, iters);
    run_shape<16u>({"shexp_gate 4096x4096(16)", 4096, 16}, iters);
    run_shape<16u>({"输出头 129280x4096(16)", 129280, 16}, iters);
    run_rows8(iters);
    return 0;
}
