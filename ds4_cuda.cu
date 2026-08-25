#include <cuda_runtime.h>

/* 网格封顶(2026-08-21 复查): 这些 384 是 08-20 在 decode 形状上标定的"48SM×4驻留块",
 * 属于写死的调优常量。做成可调以便复测(DS4_CUDA_GRID_CAP), 默认仍 384。 */
static unsigned ds4_grid_cap(void) {
    static unsigned v = 0u;
    if (v == 0u) { const char *e = ((const char *)0) /* DS4_CUDA_GRID_CAP: 路径开关已删(2026-08-22 隐形炸弹清理) */; v = e ? (unsigned)atoi(e) : 384u; }
    return v;
}

#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cub/block/block_radix_sort.cuh>

#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

#include <vector>

/* v2.2 VQ 码本解析(DQVL/DQVQ)。纯 C、零依赖、static inline —— 量化器/Metal/CUDA
 * 共用同一份解码, 保证三端逐位同义。 */
#include "vq_fmt.h"

/* GPU 契约头(子头带 extern "C" 守卫)。API 定义因此直接继承 C 链接与签名检查:
 * 实现与契约不一致会在编译期报 conflicting declaration, 而不是静默的 ABI 错位。
 * ds4_gpu_tensor 在契约里是 opaque typedef, 下面补上 CUDA 侧的具体定义;
 * ds4_gpu_residual_set 直接用契约里的那一份(历史上这里手抄过一份镜像)。 */
#include "ds4_gpu.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_QK_K 256
#define DS4_CUDA_UNUSED __attribute__((unused))

enum {
    /* attention_decode_mixed_kernel stores raw-window scores plus visible
     * compressed scores in shared memory.  The host routes larger unmasked
     * decode calls to the online attention kernel so this fixed buffer never
     * becomes an out-of-bounds write at long context. */
    DS4_CUDA_ATTENTION_SCORE_CAP = 8192u,
    DS4_CUDA_ATTENTION_RAW_SCORE_CAP = 256u,
    DS4_CUDA_TOPK_MERGE_GROUP = 8u
};

struct ds4_gpu_tensor {
    void *ptr;
    uint64_t bytes;
    int owner;
};

typedef struct {
    uint8_t scales[CUDA_QK_K / 16];
    uint8_t qs[CUDA_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} cuda_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[CUDA_QK_K / 2];
} cuda_block_q4_K;

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} cuda_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} cuda_block_iq2_xxs;

#include "ds4_iq2_tables_cuda.inc"

/* 前向声明: q4_K attn_output kernel 组(定义在 q4_K dot 辅助之后, 调用点在前) */
__global__ static void grouped_q4_K_a_preq_warp8_kernel(
        float *low, const unsigned char *w, const int8_t *xq, const float *xscale,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint32_t n_tokens,
        uint64_t kblocks, int use_dp4a);
__global__ static void matmul_q4_K_hc_expand_preq_warp8_kernel(
        float *out_hc, float *block_out, const float *block_add,
        const float *residual_hc, const float *split,
        const unsigned char *w, const int8_t *xq, const float *xscale,
        uint64_t in_dim, uint64_t out_dim, uint32_t n_embd, uint32_t n_hc,
        uint64_t kblocks, int has_add, int use_dp4a);
__global__ static void grouped_q4_K_a_preq_warp8_dp4a_kernel(
        float *low, const unsigned char *w, const int8_t *xq, const float *xscale,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint32_t n_tokens,
        uint64_t kblocks);
__global__ static void matmul_q4_K_hc_expand_preq_warp8_dp4a_kernel(
        float *out_hc, float *block_out, const float *block_add,
        const float *residual_hc, const float *split,
        const unsigned char *w, const int8_t *xq, const float *xscale,
        uint64_t in_dim, uint64_t out_dim, uint32_t n_embd, uint32_t n_hc,
        uint64_t kblocks, int has_add);

/* host 侧 f16→f32(探针用) */
static float dev_host_f16(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, f;
    if (e == 0) { if (!m) f = s; else { e = 112; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF; f = s | (e << 23) | (m << 13); } }
    else if (e == 31) f = s | 0x7F800000u | (m << 13);
    else f = s | ((e + 112) << 23) | (m << 13);
    float out; memcpy(&out, &f, 4); return out;
}

static int g_q2k_probe_out;
/* host 侧 q2_K 单块 dequant(canonical GGML 布局), 探针用 */
static void host_deq_q2k_block(const uint8_t *blk, float *out256) {
    const uint8_t *sc = blk, *qs = blk + 16;
    uint16_t hd, hm; memcpy(&hd, blk + 80, 2); memcpy(&hm, blk + 82, 2);
    const float d = dev_host_f16(hd), dm = dev_host_f16(hm);
    for (int j = 0; j < 16; j++) {
        const float dj = d * (sc[j] & 0xF), mj = dm * (sc[j] >> 4);
        for (int ii = 0; ii < 16; ii++) {
            const int idx = j * 16 + ii;
            const int qpos = (idx / 128) * 32 + (idx % 32);
            const int q = (qs[qpos] >> ((idx % 128) / 32 * 2)) & 3;
            out256[idx] = dj * q - mj;
        }
    }
}


/* ==== 全q2 f16 影子(2026-08-19 用户令"零例外全q2, 引擎不适配就改"): f16 专线家族
 * (embd/compressor/indexer/hc_fn/router) 的 q2_K 权重装载时一次性 dequant→f16 device
 * buffer, cuda_model_range_ptr 按 offset 命中重定向 — q8 repack 同类机制:
 * 文件全 q2, 运行时每 token 零额外成本。 ==== */
#define Q2K_SHADOW_MAX 1024
/* base 必须入表(2026-08-21): 只按 offset 命中会让副模型(DS4_DRAFT_GGUF)的张量撞上主模型
 * 的影子 —— 读到完全不相干的权重。drafter 同输入两跑草稿不同的真因之一。 */
static struct { const void *base; uint64_t off; const char *ptr; } g_q2k_shadow[Q2K_SHADOW_MAX];
static int g_q2k_shadow_n = 0;

static int cuda_ok(cudaError_t err, const char *what);

int ds4_gpu_register_q2k_f16_shadow(
        const void *model_map, uint64_t model_size,
        uint64_t offset, uint64_t rows, uint64_t cols) {
    /* 装载期先于 ds4_gpu_init 到达也安全: cudaMalloc 隐式建 device-0 primary context。 */
    if (g_q2k_shadow_n >= Q2K_SHADOW_MAX || cols == 0 || cols % 256u != 0) return 0;
    const uint64_t blocks = cols / 256u;
    const uint64_t src_bytes = rows * blocks * sizeof(cuda_block_q2_K);
    if (offset > model_size || src_bytes > model_size - offset) return 0;
    const uint8_t *src = (const uint8_t *)model_map + offset;
    __half *hbuf = (__half *)malloc(rows * cols * sizeof(__half));
    if (!hbuf) return 0;
    float tmp[256];
    for (uint64_t r = 0; r < rows; r++)
        for (uint64_t b = 0; b < blocks; b++) {
            host_deq_q2k_block(src + (r * blocks + b) * sizeof(cuda_block_q2_K), tmp);
            for (int i = 0; i < 256; i++)
                hbuf[r * cols + b * 256u + (uint64_t)i] = __float2half(tmp[i]);
        }
    if (((const char *)0) /* DS4_Q2K_SHADOW_CHECK: 路径开关已删(2026-08-22 隐形炸弹清理) */) {
        uint64_t bad = 0;
        for (uint64_t i = 0; i < rows * cols; i++) {
            const float v = __half2float(hbuf[i]);
            if (isnan(v) || isinf(v)) bad++;
        }
        fprintf(stderr, "ds4: [q2k-shadow] off=%llu rows=%llu cols=%llu bad=%llu\n",
                (unsigned long long)offset, (unsigned long long)rows,
                (unsigned long long)cols, (unsigned long long)bad);
    }
    void *dbuf = NULL;
    if (!cuda_ok(cudaMalloc(&dbuf, rows * cols * sizeof(__half)), "q2k shadow")) { free(hbuf); return 0; }
    if (!cuda_ok(cudaMemcpy(dbuf, hbuf, rows * cols * sizeof(__half), cudaMemcpyHostToDevice),
                 "q2k shadow up")) { free(hbuf); return 0; }
    free(hbuf);
    g_q2k_shadow[g_q2k_shadow_n].base = model_map;
    g_q2k_shadow[g_q2k_shadow_n].off = offset;
    g_q2k_shadow[g_q2k_shadow_n].ptr = (const char *)dbuf;
    g_q2k_shadow_n++;
    return 1;
}

static const void *g_model_host_base;
static const char *g_model_device_base;
static uint64_t g_model_registered_size;
static int g_model_registered;
static int g_model_device_owned;
static int g_model_range_mapping_supported = 1;
static int g_model_hmm_direct;
static int g_model_fd = -1;
static const void *g_model_fd_host_base;
static int g_model_direct_fd = -1;
static uint64_t g_model_direct_align = 1;
static uint64_t g_model_file_size;
static int g_model_cache_full;
static cudaStream_t g_model_prefetch_stream;
static cudaStream_t g_model_upload_stream;
static cublasHandle_t g_cublas;
static int g_cublas_ready;
static int g_quality_mode;

struct cuda_model_range {
    const void *host_base;
    uint64_t offset;
    uint64_t bytes;
    char *device_ptr;
    void *registered_base;
    char *registered_device_base;
    uint64_t registered_bytes;
    int host_registered;
    int arena_allocated;
};

struct cuda_model_arena {
    char *device_ptr;
    uint64_t bytes;
    uint64_t used;
};

struct cuda_q8_f16_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    __half *device_ptr;
};

struct cuda_q8_f32_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    float *device_ptr;
};

static std::vector<cuda_model_range> g_model_ranges;
static std::vector<cuda_model_arena> g_model_arenas;
static std::unordered_map<uint64_t, size_t> g_model_range_by_offset;
static std::vector<cuda_q8_f16_range> g_q8_f16_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f16_by_offset;
static std::vector<cuda_q8_f32_range> g_q8_f32_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f32_by_offset;
static uint64_t g_model_range_bytes;
static uint64_t g_q8_f16_bytes;
static uint64_t g_q8_f32_bytes;
static int g_q8_f16_disabled_after_oom;
static int g_q8_f16_budget_notice_printed;
static uint64_t g_model_load_progress_next;
static double g_model_load_progress_last;
static int g_model_load_progress_started;
static int g_model_load_progress_tty;
static void *g_cuda_tmp;
static uint64_t g_cuda_tmp_bytes;
static void *g_model_stage_raw[4];
static void *g_model_stage[4];
static cudaEvent_t g_model_stage_event[4];
static uint64_t g_model_stage_bytes;

static int cuda_ok(cudaError_t err, const char *what);
static const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what);
__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);
__global__ static void dequant_q8_0_to_f32_kernel(
        float *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks);

static void *g_q2k_part = NULL;
static uint64_t g_q2k_part_bytes = 0;

static void *cuda_tmp_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_cuda_tmp_bytes >= bytes) return g_cuda_tmp;
    if (g_cuda_tmp) {
        (void)cudaFree(g_cuda_tmp);
        g_cuda_tmp = NULL;
        g_cuda_tmp_bytes = 0;
    }
    void *ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA temp alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "scratch", (double)bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    g_cuda_tmp = ptr;
    g_cuda_tmp_bytes = bytes;
    return g_cuda_tmp;
}

static int cuda_attention_score_buffer_fits(uint32_t n_comp) {
    return n_comp <= DS4_CUDA_ATTENTION_SCORE_CAP - DS4_CUDA_ATTENTION_RAW_SCORE_CAP;
}

#ifndef DS4_Q2K_BLOCKS_PER_SM
#define DS4_Q2K_BLOCKS_PER_SM 4   /* dense q2 每 SM 常驻块数(占用率闸, 2026-08-21 A/B) */
#endif
static const char *cuda_model_ptr(const void *model_map, uint64_t offset) {
    if (model_map == g_model_host_base && g_model_device_base) return g_model_device_base + offset;
    return (const char *)model_map + offset;
}

/* Allocate a device-resident copy of [offset, offset+bytes) from model_map and
 * push it into g_model_ranges so future cuda_model_range_ptr lookups hit it.
 * Returns the device pointer on success, NULL on cudaMalloc/cudaMemcpy failure.
 * Caller is responsible for any policy gating (budget cap, env opt-out, etc.) */
static const char *cuda_model_range_populate_device_copy(const void *model_map,
                                                          uint64_t offset,
                                                          uint64_t bytes,
                                                          const char *what) {
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: CUDA model range alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "weights", (double)bytes / 1048576.0, cudaGetErrorString(err));
        return NULL;
    }

    const char *src = (const char *)model_map + offset;
    const uint64_t chunk = 64ull * 1024ull * 1024ull;
    for (uint64_t done = 0; done < bytes; done += chunk) {
        uint64_t n = bytes - done < chunk ? bytes - done : chunk;
        err = cudaMemcpy((char *)dev + done, src + done, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model range copy failed for %s at %.2f/%.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)done / 1048576.0,
                    (double)bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return NULL;
        }
    }
    g_model_ranges.push_back({model_map, offset, bytes, (char *)dev, NULL, NULL, 0, 0, 0});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    if (((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */) {
        fprintf(stderr, "ds4: CUDA cached %s %.2f MiB (total %.2f GiB)\n",
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)g_model_range_bytes / 1073741824.0);
    }
    return (const char *)dev;
}

static const char *cuda_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what) {
    /* 全q2 f16 影子命中(小表线性扫, 每 launch 一次, 开销可忽略) */
    for (int i = 0; i < g_q2k_shadow_n; i++)
        if (g_q2k_shadow[i].off == offset && g_q2k_shadow[i].base == model_map)
            return g_q2k_shadow[i].ptr;
    if (bytes == 0) return cuda_model_ptr(model_map, offset);

    /* Device-resident HBM cache hits win over UVA-mapped registered pointers:
     * direct HBM reads are ~10% faster than mapped reads through host page
     * tables (measured on plain decode at GB10).  Cache lookup runs first; the
     * registered-mapped shortcut below is the cold fallback when an allocation
     * hasn't been pre-populated. */
    const uint64_t end = offset + bytes;
    auto exact = g_model_range_by_offset.find(offset);
    if (exact != g_model_range_by_offset.end()) {
        const cuda_model_range &r = g_model_ranges[exact->second];
        if (r.host_base == model_map && end >= offset && bytes <= r.bytes) return r.device_ptr;
    }
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map && offset >= r.offset && end >= offset && end <= r.offset + r.bytes) {
            return r.device_ptr + (offset - r.offset);
        }
        if (r.host_base == model_map && r.host_registered && r.registered_base && r.registered_device_base) {
            const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
            const uintptr_t h1 = h0 + bytes;
            const uintptr_t r0 = (uintptr_t)r.registered_base;
            const uintptr_t r1 = r0 + r.registered_bytes;
            if (h1 >= h0 && h0 >= r0 && h1 <= r1) return r.registered_device_base + (h0 - r0);
        }
    }

    /* 这三条"整模型已就绪"捷径只对主 map 成立(2026-08-21): 副 map(DS4_DRAFT_GGUF)命中
     * 时会拿到未注册裸指针 —— 实测 drafter logits 直接 NaN、acc 塌到 1.00 而速度看似正常。
     * 非主 base 一律落到下面的按 range 懒注册路, 保证任何情况下指针可用。 */
    const int is_primary_map = (model_map == g_model_host_base);
    if (is_primary_map && (g_model_device_owned || g_model_registered))
        return cuda_model_ptr(model_map, offset);
    if (is_primary_map && g_model_hmm_direct &&
        1 &&
        1) {
        return cuda_model_ptr(model_map, offset);
    }
    const char *direct_env = ((const char *)0) /* DS4_CUDA_DIRECT_MODEL: 路径开关已删(2026-08-22 隐形炸弹清理) */;
    if (is_primary_map && direct_env && direct_env[0]) return cuda_model_ptr(model_map, offset);

    if (1) {
        const char *fd_ptr = cuda_model_range_ptr_from_fd(model_map, offset, bytes, what);
        if (fd_ptr) return fd_ptr;
    }

    cudaError_t err = cudaSuccess;
    if (g_model_range_mapping_supported) {
        const long page_sz_l = sysconf(_SC_PAGESIZE);
        const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
        const uintptr_t host_addr = (uintptr_t)((const char *)model_map + offset);
        const uintptr_t reg_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
        const uint64_t reg_delta = (uint64_t)(host_addr - reg_addr);
        const uint64_t reg_bytes = (reg_delta + bytes + page_sz - 1u) & ~(page_sz - 1u);
        void *reg_dev = NULL;
        err = cudaHostRegister((void *)reg_addr,
                               (size_t)reg_bytes,
                               cudaHostRegisterMapped);
        if (err == cudaSuccess) {
            err = cudaHostGetDevicePointer(&reg_dev, (void *)reg_addr, 0);
            if (err == cudaSuccess && reg_dev) {
                char *dev_ptr = (char *)reg_dev + reg_delta;
                g_model_ranges.push_back({model_map, offset, bytes, dev_ptr, (void *)reg_addr, (char *)reg_dev, reg_bytes, 1, 0});
                g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
                if (((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */) {
                    fprintf(stderr, "ds4: CUDA mapped %s %.2f MiB\n",
                            what ? what : "weights",
                            (double)bytes / 1048576.0);
                }
                return dev_ptr;
            }
            fprintf(stderr, "ds4: CUDA model range map pointer failed for %s: %s\n",
                    what ? what : "weights", cudaGetErrorString(err));
            (void)cudaHostUnregister((void *)reg_addr);
            (void)cudaGetLastError();
        } else {
            if (err == cudaErrorNotSupported || err == cudaErrorInvalidValue) g_model_range_mapping_supported = 0;
            (void)cudaGetLastError();
        }
    }

    /* 非主 map(drafter/zchain/corr 侧车)注册不成时回落裸指针 —— 拷贝路对这些小而
     * 不规则的 range 会 invalid argument(2026-08-21 实测), 且裸指针对侧车本就够用。 */
    if (!is_primary_map) return cuda_model_ptr(model_map, offset);
    return cuda_model_range_populate_device_copy(model_map, offset, bytes, what);
}

static void cuda_q8_f16_cache_release_all(void) {
    for (const cuda_q8_f16_range &r : g_q8_f16_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f16_ranges.clear();
    g_q8_f16_by_offset.clear();
    g_q8_f16_bytes = 0;
}

static uint64_t cuda_parse_mib_env(const char *name, int *present) {
    const char *env = getenv(name);
    if (present) *present = 0;
    if (!env || !env[0]) return 0;
    char *end = NULL;
    unsigned long long v = strtoull(env, &end, 10);
    if (end == env || *end != '\0') return 0;
    if (present) *present = 1;
    if (v > UINT64_MAX / 1048576ull) return UINT64_MAX;
    return (uint64_t)v * 1048576ull;
}

static uint64_t cuda_q8_f16_cache_limit_bytes(void) {
    int present = 0;
    const uint64_t limit = cuda_parse_mib_env("DS4_CUDA_Q8_F16_CACHE_MB", &present);
    return present ? limit : UINT64_MAX;
}

static uint64_t cuda_q8_f16_cache_reserve_bytes(uint64_t total_bytes) {
    int present = 0;
    const uint64_t reserve = cuda_parse_mib_env("DS4_CUDA_Q8_F16_CACHE_RESERVE_MB", &present);
    if (present) return reserve;

    if (total_bytes >= 112ull * 1024ull * 1024ull * 1024ull) {
        return 512ull * 1048576ull;
    }

    /* The expanded Q8->F16 cache is only an acceleration path.  Keep enough
     * device memory free for cuBLAS workspaces, transient graph buffers, and
     * driver bookkeeping instead of letting optional cached weights consume the
     * last few GiB on 96 GiB cards. */
    const uint64_t min_reserve = 4096ull * 1048576ull;
    const uint64_t pct_reserve = total_bytes / 20u; /* 5% */
    return pct_reserve > min_reserve ? pct_reserve : min_reserve;
}

static void cuda_q8_f16_cache_budget_notice(
        const char *reason,
        uint64_t request_bytes,
        uint64_t free_bytes,
        uint64_t total_bytes,
        uint64_t reserve_bytes,
        uint64_t limit_bytes) {
    if (g_q8_f16_budget_notice_printed && ((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */ == NULL) return;
    g_q8_f16_budget_notice_printed = 1;
    if (limit_bytes != UINT64_MAX && free_bytes == 0 && total_bytes == 0 && reserve_bytes == 0) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0);
    } else if (limit_bytes == UINT64_MAX) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    } else {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache %s; using q8 kernels "
                "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
                reason,
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)limit_bytes / 1073741824.0,
                (double)free_bytes / 1073741824.0,
                (double)reserve_bytes / 1073741824.0,
                (double)total_bytes / 1073741824.0);
    }
}

static int cuda_q8_f16_cache_has_budget(uint64_t request_bytes, const char *label) {
    (void)label;
    const uint64_t limit = cuda_q8_f16_cache_limit_bytes();
    if (limit == 0) return 0;
    if (g_q8_f16_bytes > limit || request_bytes > limit - g_q8_f16_bytes) {
        cuda_q8_f16_cache_budget_notice("limit reached", request_bytes, 0, 0, 0, limit);
        return 0;
    }

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA q8 fp16 cache memory query failed: %s; using q8 kernels\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    /* Unified-memory boards (GB10 etc.): cudaMemGetInfo free is distorted by
     * reclaimable page cache holding the mmapped model, so a free-based veto
     * flips kernel choice mid-run and mutates the token-graph topology.  Trust
     * cudaMalloc as the real verdict; its failure path already disables the
     * cache permanently. */
    if (total_bytes >= 112ull * 1024ull * 1024ull * 1024ull) return 1;
    const uint64_t reserve_bytes = cuda_q8_f16_cache_reserve_bytes(total_bytes);
    if (request_bytes > free_bytes ||
        free_bytes - request_bytes < reserve_bytes) {
        cuda_q8_f16_cache_budget_notice("budget exhausted", request_bytes,
                                        free_bytes, total_bytes,
                                        reserve_bytes, limit);
        return 0;
    }
    return 1;
}

static void cuda_q8_f16_cache_disable_after_failure(const char *what, uint64_t request_bytes) {
    if (!g_q8_f16_disabled_after_oom) {
        fprintf(stderr,
                "ds4: CUDA q8 fp16 cache disabled after %s "
                "(request=%.2f MiB cached=%.2f GiB); using q8 kernels\n",
                what ? what : "allocation failure",
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0);
    }
    g_q8_f16_disabled_after_oom = 1;
    if (!g_q8_f16_ranges.empty()) {
        (void)cudaDeviceSynchronize();
        cuda_q8_f16_cache_release_all();
    }
    (void)cudaGetLastError();
}

static int cuda_q8_f16_cache_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (g_quality_mode) return 0;
    if (g_q8_f16_disabled_after_oom) return 0;
    if (0) return 0;
    if (cuda_q8_f16_cache_limit_bytes() == 0) return 0;
    if (0) return 1;
    if (!label) return 0;
    if (strstr(label, "attn_output_a") != NULL ||
        strstr(label, "attn_output_b") != NULL ||
        strstr(label, "attention_output_a") != NULL ||
        strstr(label, "attention_output_b") != NULL) {
        return 1;
    }
    if (strstr(label, "attn_q_b") != NULL) {
        return 1;
    }
    if (strstr(label, "ffn_gate_shexp") != NULL ||
        strstr(label, "ffn_up_shexp") != NULL ||
        strstr(label, "ffn_down_shexp") != NULL) {
        return 1;
    }
    return (in_dim == 4096u && out_dim == 2048u) ||
           (in_dim == 2048u && out_dim == 4096u) ||
           (in_dim == 4096u && out_dim == 1024u) ||
           (in_dim == 4096u && out_dim == 512u) ||
           (1 &&
            in_dim == 1024u && out_dim == 32768u);
}

static int cuda_q8_label_is_attention_output(const char *label) {
    return label &&
           (strstr(label, "attn_output_a") != NULL ||
            strstr(label, "attn_output_b") != NULL ||
            strstr(label, "attention_output_a") != NULL ||
            strstr(label, "attention_output_b") != NULL);
}

static int cuda_q8_use_dp4a(void) {
    return 1;
}

static int cuda_q8_f16_preload_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (cuda_q8_label_is_attention_output(label) &&
        1 &&
        1) {
        return 0;
    }
    return cuda_q8_f16_cache_allowed(label, in_dim, out_dim);
}

static int cuda_q8_f32_cache_allowed(const char *label, uint64_t in_dim, uint64_t out_dim) {
    if (0) return 0;
    if (0) return 1;
    if (label && strstr(label, "attn_q_b") != NULL) {
        return 0;
    }
    return 0 &&
           in_dim == 1024u && out_dim == 32768u;
}

static std::unordered_set<uint64_t> g_q8_f16_denied; /* sticky: weight stays on q8 kernels all run */

static const __half *cuda_q8_f16_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        const char *label) {
    auto exact = g_q8_f16_by_offset.find(offset);
    if (exact != g_q8_f16_by_offset.end()) {
        const cuda_q8_f16_range &r = g_q8_f16_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    if (g_q8_f16_denied.count(offset)) return NULL;
    if (!cuda_q8_f16_cache_allowed(label, in_dim, out_dim)) return NULL;

    const char *q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, "q8_0");
    if (!q8) return NULL;

    if (in_dim != 0 && out_dim > UINT64_MAX / in_dim / sizeof(__half)) return NULL;
    const uint64_t out_bytes = in_dim * out_dim * sizeof(__half);
    if (!cuda_q8_f16_cache_has_budget(out_bytes, label)) {
        g_q8_f16_denied.insert(offset);
        return NULL;
    }

    __half *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        g_q8_f16_denied.insert(offset);
        fprintf(stderr, "ds4: CUDA q8 fp16 cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        cuda_q8_f16_cache_disable_after_failure("allocation failure", out_bytes);
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f16_kernel<<<(n + 255) / 256, 256>>>(dev,
                                                          (const unsigned char *)q8,
                                                          in_dim,
                                                          out_dim,
                                                          blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp16 dequant launch")) {
        (void)cudaFree(dev);
        cuda_q8_f16_cache_disable_after_failure("dequant launch failure", out_bytes);
        return NULL;
    }
    /* The dequant runs on the default stream while consumers may issue work on
     * other streams (or capture a graph) immediately after this returns.  The
     * cache entry is permanent, so publishing it before the fill completes
     * poisons every later token.  One synchronize per weight, at build time
     * only. */
    if (!cuda_ok(cudaDeviceSynchronize(), "q8 fp16 dequant sync")) {
        (void)cudaFree(dev);
        cuda_q8_f16_cache_disable_after_failure("dequant sync failure", out_bytes);
        return NULL;
    }
    g_q8_f16_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f16_by_offset[offset] = g_q8_f16_ranges.size() - 1u;
    g_q8_f16_bytes += out_bytes;
    if (((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */) {
        fprintf(stderr, "ds4: CUDA cached q8 fp16 %.2f MiB (total %.2f GiB)\n",
                (double)out_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0);
    }
    return dev;
}

/* 只读查表版(decode/graph-capture 安全): 不建 cache, 命中已有条目才返回 */
static const __half *cuda_q8_f16_lookup(const void *model_map, uint64_t offset,
                                        uint64_t weight_bytes, uint64_t in_dim, uint64_t out_dim) {
    auto it = g_q8_f16_by_offset.find(offset);
    if (it == g_q8_f16_by_offset.end()) return NULL;
    const cuda_q8_f16_range &r = g_q8_f16_ranges[it->second];
    if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
        r.in_dim == in_dim && r.out_dim == out_dim) return r.device_ptr;
    return NULL;
}

/* ==== q8 repack 缓存(decode gemv 专用) ====
 * 原 q8_0 行布局是 34B 交错(2B scale + 32B qs), qs 永远非 4B 对齐 —— dp4a 内核
 * 每块 8 次非对齐 4B 读, LSU 事务 4x 放大(ncu: q8 gemv 仅吃出 16-32% 带宽)。
 * 启动/预填充期一次 repack 成两平面: [f16 scale × blocks×rows][int8 qs 16B 对齐],
 * decode 内核换 int4(128bit) 对齐读。数值 bit 级不变(同 scale 同 qs)。 */
typedef struct { const void *host_base; uint64_t offset, bytes; __half *scales; int8_t *qs; uint64_t rows, blocks; } cuda_q8r_entry;
static std::vector<cuda_q8r_entry> g_q8r_entries;
static std::unordered_map<uint64_t, size_t> g_q8r_by_offset;
static std::unordered_set<uint64_t> g_q8r_denied;

__global__ static void q8_repack_kernel(__half *scales, int8_t *qs,
                                        const unsigned char *w, uint64_t rows, uint64_t blocks) {
    const uint64_t r = blockIdx.x;
    if (r >= rows) return;
    const unsigned char *wr = w + r * blocks * 34u;
    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        __half sh; memcpy(&sh, wr + b * 34u, 2);
        scales[r * blocks + b] = sh;
        const unsigned char *src = wr + b * 34u + 2u;
        int8_t *dst = qs + (r * blocks + b) * 32u;
        for (int i = 0; i < 32; i++) dst[i] = (int8_t)src[i];
    }
}

static const cuda_q8r_entry *cuda_q8r_get(const void *model_map, uint64_t offset,
                                          uint64_t rows, uint64_t blocks) {
    auto it = g_q8r_by_offset.find(offset);
    if (it != g_q8r_by_offset.end()) {
        const cuda_q8r_entry &e = g_q8r_entries[it->second];
        if (e.host_base == model_map && e.rows == rows && e.blocks == blocks) return &g_q8r_entries[it->second];
        return NULL;
    }
    if (g_q8r_denied.count(offset)) return NULL;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    (void)cudaStreamIsCapturing(0, &cs);
    if (cs != cudaStreamCaptureStatusNone) return NULL;   /* capture 期不建, 回落原路 */
    const uint64_t weight_bytes = rows * blocks * 34u;
    const char *w = cuda_model_range_ptr(model_map, offset, weight_bytes, "q8r_src");
    if (!w) { g_q8r_denied.insert(offset); return NULL; }
    __half *sc = NULL; int8_t *qs = NULL;
    if (cudaMalloc(&sc, (size_t)rows * blocks * sizeof(__half)) != cudaSuccess) {
        (void)cudaGetLastError(); g_q8r_denied.insert(offset); return NULL;
    }
    if (cudaMalloc(&qs, (size_t)rows * blocks * 32u) != cudaSuccess) {
        (void)cudaGetLastError(); (void)cudaFree(sc); g_q8r_denied.insert(offset); return NULL;
    }
    q8_repack_kernel<<<(unsigned)rows, 256>>>(sc, qs, (const unsigned char *)w, rows, blocks);
    if (cudaDeviceSynchronize() != cudaSuccess) {   /* 建完才发布, 同 fp16 cache */
        (void)cudaGetLastError(); (void)cudaFree(sc); (void)cudaFree(qs);
        g_q8r_denied.insert(offset); return NULL;
    }
    g_q8r_entries.push_back({model_map, offset, weight_bytes, sc, qs, rows, blocks});
    g_q8r_by_offset[offset] = g_q8r_entries.size() - 1u;
    return &g_q8r_entries.back();
}

int ds4_gpu_q8r_preload(const void *model_map, uint64_t model_size,
                                   uint64_t offset, uint64_t in_dim, uint64_t out_dim) {
    if (!model_map || in_dim == 0 || out_dim == 0 || (in_dim & 31u)) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (offset > model_size || out_dim * blocks * 34u > model_size - offset) return 0;
    return cuda_q8r_get(model_map, offset, out_dim, blocks) != NULL;
}

static float *cuda_q8_f32_ptr(
        const void *model_map,
        uint64_t offset,
        uint64_t weight_bytes,
        uint64_t in_dim,
        uint64_t out_dim,
        const char *label) {
    auto exact = g_q8_f32_by_offset.find(offset);
    if (exact != g_q8_f32_by_offset.end()) {
        const cuda_q8_f32_range &r = g_q8_f32_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            return r.device_ptr;
        }
    }
    if (!cuda_q8_f32_cache_allowed(label, in_dim, out_dim)) return NULL;

    const char *q8 = cuda_model_range_ptr(model_map, offset, weight_bytes, label ? label : "q8_0");
    if (!q8) return NULL;

    const uint64_t out_bytes = in_dim * out_dim * sizeof(float);
    float *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA q8 fp32 cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f32_kernel<<<(n + 255) / 256, 256>>>(dev,
                                                          (const unsigned char *)q8,
                                                          in_dim,
                                                          out_dim,
                                                          blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp32 dequant launch")) {
        (void)cudaFree(dev);
        return NULL;
    }
    g_q8_f32_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f32_by_offset[offset] = g_q8_f32_ranges.size() - 1u;
    g_q8_f32_bytes += out_bytes;
    if (((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */) {
        fprintf(stderr, "ds4: CUDA cached q8 fp32 %.2f MiB (total %.2f GiB)\n",
                (double)out_bytes / 1048576.0,
                (double)g_q8_f32_bytes / 1073741824.0);
    }
    return dev;
}

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "ds4: CUDA %s failed: %s\n", what, cudaGetErrorString(err));
    return 0;
}

static double cuda_wall_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int cuda_model_load_progress_enabled(void) {
    if (((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */ != NULL) return 0;
    return 1;
}

static void cuda_model_load_progress_reset(void) {
    g_model_load_progress_next = 0;
    g_model_load_progress_last = 0.0;
    g_model_load_progress_started = 0;
    g_model_load_progress_tty = 0;
}

static void cuda_model_load_progress_note(uint64_t cached_bytes) {
    if (!cuda_model_load_progress_enabled()) return;

    const double now = cuda_wall_sec();
    if (!g_model_load_progress_started) {
        g_model_load_progress_started = 1;
        g_model_load_progress_tty = isatty(STDERR_FILENO) != 0;
        g_model_load_progress_next = (g_model_load_progress_tty ? 2ull : 16ull) *
                                     1024ull * 1024ull * 1024ull;
        g_model_load_progress_last = now;
        if (g_model_load_progress_tty) {
            fprintf(stderr, "ds4: CUDA loading model tensors into device cache: 0.00 GiB");
        } else {
            fprintf(stderr, "ds4: CUDA loading model tensors into device cache\n");
        }
    }

    if (cached_bytes < g_model_load_progress_next &&
        now - g_model_load_progress_last < (g_model_load_progress_tty ? 2.0 : 10.0)) {
        return;
    }

    if (g_model_load_progress_tty) {
        fprintf(stderr, "\rds4: CUDA loading model tensors into device cache: %.2f GiB",
                (double)cached_bytes / 1073741824.0);
    } else {
        fprintf(stderr, "ds4: CUDA loading model tensors %.2f GiB cached\n",
                (double)cached_bytes / 1073741824.0);
    }
    fflush(stderr);
    g_model_load_progress_last = now;
    const uint64_t step = (g_model_load_progress_tty ? 2ull : 16ull) *
                          1024ull * 1024ull * 1024ull;
    while (g_model_load_progress_next <= cached_bytes) {
        g_model_load_progress_next += step;
    }
}

static int cuda_model_prefetch_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || map_size == 0 || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (0 ||
        0 ||
        0 ||
        0) {
        return 0;
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    int pageable = 0;
    cudaError_t err = cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, device);
    if (err != cudaSuccess || !pageable) {
        (void)cudaGetLastError();
        return 0;
    }
    cudaMemLocation loc;
    memset(&loc, 0, sizeof(loc));
    loc.type = cudaMemLocationTypeDevice;
    loc.id = device;

    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t host_addr = (uintptr_t)((const char *)model_map + map_offset);
    const uintptr_t pre_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
    const uint64_t pre_delta = (uint64_t)(host_addr - pre_addr);
    const uint64_t pre_bytes = (pre_delta + map_size + page_sz - 1u) & ~(page_sz - 1u);
    void *pre_ptr = (void *)pre_addr;

    const double t0 = cuda_wall_sec();
    err = cudaMemAdvise(pre_ptr, (size_t)pre_bytes, cudaMemAdviseSetReadMostly, loc);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model read-mostly advise skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    err = cudaMemAdvise(pre_ptr, (size_t)pre_bytes, cudaMemAdviseSetPreferredLocation, loc);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model preferred-location advise skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    if (!g_model_prefetch_stream) {
        err = cudaStreamCreateWithFlags(&g_model_prefetch_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model prefetch stream creation skipped: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }

    err = cudaMemPrefetchAsync(pre_ptr, (size_t)pre_bytes, loc, 0, g_model_prefetch_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model prefetch skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    /* ★默认同步★(2026-08-17 GB10 实锤): 本机 UsesHostPageTables=1(ATS), 异步页迁移
     * 与在飞 kernel 并发时, 迁移中的页被读出垃圾 —— 表现为批量 prefill 输出塌 BOS、
     * NaN 首层随运行漂移(迁移进度决定谁踩雷)、compute-sanitizer(强制串行)下反而正确、
     * DS4_CUDA_NO_MODEL_PREFETCH=1 立即痊愈。迁移是启动期一次性成本(秒级), 等它完成
     * 换全程正确性。DS4_CUDA_MODEL_PREFETCH_ASYNC=1 可回旧行为(仅限已证安全的平台)。 */
    if (1) {
        err = cudaStreamSynchronize(g_model_prefetch_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model prefetch sync failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    const double t1 = cuda_wall_sec();
    fprintf(stderr,
            "ds4: CUDA ATS/HMM prefetch queued %.2f GiB of model tensors in %.3fs\n",
            (double)map_size / 1073741824.0,
            t1 - t0);
    g_model_hmm_direct = 1;
    return 1;
}

static uint64_t cuda_model_copy_chunk_bytes(void) {
    uint64_t mb = 64;
    const char *env = ((const char *)0) /* DS4_CUDA_MODEL_COPY_CHUNK_MB: 路径开关已删(2026-08-22 隐形炸弹清理) */;
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
    if (mb < 16) mb = 16;
    if (mb > 4096) mb = 4096;
    return mb * 1048576ull;
}

static void cuda_model_discard_source_pages(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes) {
#if defined(POSIX_MADV_DONTNEED)
    if (0 || !model_map || bytes == 0 || offset > model_size) return;
    if (bytes > model_size - offset) bytes = model_size - offset;
    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
    const uintptr_t h1 = h0 + bytes;
    const uintptr_t p0 = h0 & ~(uintptr_t)(page_sz - 1u);
    const uintptr_t p1 = (h1 + page_sz - 1u) & ~(uintptr_t)(page_sz - 1u);
    if (p1 > p0) (void)posix_madvise((void *)p0, (size_t)(p1 - p0), POSIX_MADV_DONTNEED);
#else
    (void)model_map;
    (void)model_size;
    (void)offset;
    (void)bytes;
#endif
}

static void cuda_model_drop_file_pages(uint64_t offset, uint64_t bytes) {
#if defined(POSIX_FADV_DONTNEED)
    if (g_model_fd < 0 || 0 || bytes == 0) return;
    (void)posix_fadvise(g_model_fd, (off_t)offset, (off_t)bytes, POSIX_FADV_DONTNEED);
#else
    (void)offset;
    (void)bytes;
#endif
}

static uint64_t cuda_round_down(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    return (v / align) * align;
}

static uint64_t cuda_round_up(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    const uint64_t rem = v % align;
    return rem == 0 ? v : v + (align - rem);
}

static void *cuda_align_ptr(void *ptr, uint64_t align) {
    if (align <= 1) return ptr;
    uintptr_t p = (uintptr_t)ptr;
    uintptr_t a = (uintptr_t)align;
    return (void *)(((p + a - 1u) / a) * a);
}

static int cuda_model_stage_pool_alloc(uint64_t bytes) {
    if (g_model_stage_bytes >= bytes) return 1;
    for (size_t i = 0; i < 4; i++) {
        if (g_model_stage_event[i]) {
            (void)cudaEventDestroy(g_model_stage_event[i]);
            g_model_stage_event[i] = NULL;
        }
        if (g_model_stage_raw[i]) {
            (void)cudaFreeHost(g_model_stage_raw[i]);
            g_model_stage_raw[i] = NULL;
            g_model_stage[i] = NULL;
        }
    }
    g_model_stage_bytes = 0;
    if (!g_model_upload_stream) {
        cudaError_t err = cudaStreamCreateWithFlags(&g_model_upload_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model upload stream creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    for (size_t i = 0; i < 4; i++) {
        cudaError_t err = cudaMallocHost(&g_model_stage_raw[i], (size_t)bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA pinned model staging allocation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_model_stage[i] = cuda_align_ptr(g_model_stage_raw[i], g_model_direct_align);
        err = cudaEventCreateWithFlags(&g_model_stage_event[i], cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model staging event creation failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    g_model_stage_bytes = bytes;
    return 1;
}

static int cuda_pread_full(int fd, void *buf, uint64_t bytes, uint64_t offset) {
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n_req = (bytes - done > (uint64_t)SSIZE_MAX) ? (size_t)SSIZE_MAX : (size_t)(bytes - done);
        ssize_t n = pread(fd, (char *)buf + done, n_req, (off_t)(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (n == 0) return 0;
        done += (uint64_t)n;
    }
    return 1;
}

static int cuda_model_stage_read(void *stage, uint64_t stage_bytes,
                                 uint64_t offset, uint64_t bytes,
                                 const char **payload) {
    *payload = (const char *)stage;
#if defined(__linux__) && defined(O_DIRECT)
    if (g_model_direct_fd >= 0 && g_model_direct_align > 1 && g_model_file_size != 0) {
        const uint64_t aligned_off = cuda_round_down(offset, g_model_direct_align);
        const uint64_t delta = offset - aligned_off;
        uint64_t read_size = cuda_round_up(delta + bytes, g_model_direct_align);
        if (aligned_off <= g_model_file_size &&
            read_size <= stage_bytes &&
            read_size <= g_model_file_size - aligned_off) {
            const int saved_errno = errno;
            errno = 0;
            if (cuda_pread_full(g_model_direct_fd, stage, read_size, aligned_off)) {
                *payload = (const char *)stage + delta;
                errno = saved_errno;
                return 1;
            }
            const int direct_errno = errno;
            if (direct_errno == EINVAL || direct_errno == EFAULT || direct_errno == ENOTSUP || direct_errno == EOPNOTSUPP) {
                if (((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */) {
                    fprintf(stderr, "ds4: CUDA direct model read disabled: %s\n", strerror(direct_errno));
                }
                (void)close(g_model_direct_fd);
                g_model_direct_fd = -1;
                g_model_direct_align = 1;
            }
            errno = direct_errno;
        }
    }
#else
    (void)stage_bytes;
#endif
    return cuda_pread_full(g_model_fd, stage, bytes, offset);
}

static uint64_t cuda_model_cache_limit_bytes(void) {
    uint64_t gb = 0;
    const char *env = ((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_LIMIT_GB: 路径开关已删(2026-08-22 隐形炸弹清理) */;
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env) gb = (uint64_t)v;
        return gb * 1073741824ull;
    }
    /* Default cap protects against OOM on UMA systems where a full ~80 GiB
     * model would otherwise duplicate the mmap'd host pages into HBM-backed
     * cudaMalloc allocations and exhaust the 121 GiB UMA pool.  24 GiB
     * comfortably covers attn projections + embedding + output head +
     * shared FFN; routed MoE experts (~65 GiB, only top-K active per token)
     * fall back to the UVA-mapped pointer for cold lookups.  Tune up via
     * DS4_CUDA_WEIGHT_CACHE_LIMIT_GB on hosts with more memory budget. */
#ifdef DS4_CUDA_SPARK_HBM_CACHE
    /* Spark/GB10 统一内存: 实测 registered-host 页表读被钉在 ~185 GB/s, 设备拷贝 255
     * (2026-08-17)。默认预算=总内存-24GiB 余量, 整模型尽量收编; 拷贝后源 mmap 页
     * madvise(DONTNEED), 净占用不翻倍。 */
    {
        const uint64_t page = (uint64_t)sysconf(_SC_PAGESIZE);
        const uint64_t total = (uint64_t)sysconf(_SC_PHYS_PAGES) * page;
        const uint64_t headroom = 24ull * 1073741824ull;
        if (total > headroom * 2) return total - headroom;
    }
#endif
    return 24ull * 1073741824ull;
}

static uint64_t cuda_model_arena_chunk_bytes(uint64_t need) {
    uint64_t mb = 1792;
    const char *env = ((const char *)0) /* DS4_CUDA_WEIGHT_ARENA_CHUNK_MB: 路径开关已删(2026-08-22 隐形炸弹清理) */;
    if (env && env[0]) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
    if (mb < 256) mb = 256;
    if (mb > 8192) mb = 8192;
    uint64_t bytes = mb * 1048576ull;
    if (bytes < need) {
        const uint64_t align = 256ull * 1048576ull;
        bytes = (need + align - 1u) & ~(align - 1u);
    }
    return bytes;
}

static char *cuda_model_arena_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_model_cache_full) return NULL;
    const uint64_t align = 256u;
    const uint64_t aligned = (bytes + align - 1u) & ~(align - 1u);

    for (cuda_model_arena &a : g_model_arenas) {
        const uint64_t used = (a.used + align - 1u) & ~(align - 1u);
        if (used <= a.bytes && aligned <= a.bytes - used) {
            char *ptr = a.device_ptr + used;
            a.used = used + aligned;
            return ptr;
        }
    }

    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || aligned > limit - g_model_range_bytes) return NULL;

    const uint64_t chunk = cuda_model_arena_chunk_bytes(aligned);
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)chunk);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model arena alloc failed for %s (%.2f MiB chunk): %s\n",
                what ? what : "weights",
                (double)chunk / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        g_model_cache_full = 1;
        return NULL;
    }
    g_model_arenas.push_back({(char *)dev, chunk, aligned});
    if (((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */) {
        uint64_t arena_bytes = 0;
        for (const cuda_model_arena &a : g_model_arenas) arena_bytes += a.bytes;
        fprintf(stderr, "ds4: CUDA model arena allocated %.2f MiB (arenas %.2f GiB)\n",
                (double)chunk / 1048576.0,
                (double)arena_bytes / 1073741824.0);
    }
    return (char *)dev;
}

/* A raw host pointer is safe for kernels only after CUDA owns, registered, or
 * HMM-prefetched the mapping.  Otherwise let the caller try per-range mapping
 * or a device copy instead of surfacing an async illegal access later. */
static const char *cuda_model_direct_fallback_ptr(const void *model_map, uint64_t offset) {
    if (g_model_device_owned || g_model_registered || g_model_hmm_direct ||
        0) {
        return cuda_model_ptr(model_map, offset);
    }
    return NULL;
}

static const char *cuda_model_range_ptr_from_fd(
        const void *model_map,
        uint64_t offset,
        uint64_t bytes,
        const char *what) {
    if (g_model_fd < 0 || bytes == 0) return NULL;
    if (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base) return NULL;
    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || bytes > limit - g_model_range_bytes) {
        if (((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */) {
            fprintf(stderr, "ds4: CUDA direct %s %.2f MiB (cache budget %.2f GiB exhausted)\n",
                    what ? what : "weights",
                    (double)bytes / 1048576.0,
                    (double)limit / 1073741824.0);
        }
        return cuda_model_direct_fallback_ptr(model_map, offset);
    }

    char *dev = cuda_model_arena_alloc(bytes, what);
    if (!dev) {
        if (0) return NULL;
        return cuda_model_direct_fallback_ptr(model_map, offset);
    }
    cudaError_t err = cudaSuccess;

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes = chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!cuda_model_stage_pool_alloc(stage_bytes)) return NULL;

    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < bytes) {
        const uint64_t n = (bytes - copied < chunk) ? (bytes - copied) : chunk;
        const uint64_t bi = chunk_idx % 4u;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(g_model_stage_event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr, "ds4: CUDA model staging wait failed for %s: %s\n",
                        what ? what : "weights", cudaGetErrorString(err));
                (void)cudaGetLastError();
                return NULL;
            }
        }
        const char *payload = NULL;
        if (!cuda_model_stage_read(g_model_stage[bi], g_model_stage_bytes,
                                   offset + copied, n, &payload)) {
            fprintf(stderr, "ds4: CUDA model range read failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    strerror(errno));
            return NULL;
        }
        err = cudaMemcpyAsync(dev + copied, payload, (size_t)n,
                              cudaMemcpyHostToDevice, g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model range copy failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        err = cudaEventRecord(g_model_stage_event[bi], g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model staging record failed for %s: %s\n",
                    what ? what : "weights", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        cuda_model_drop_file_pages(offset + copied, n);
        cuda_model_discard_source_pages(model_map, g_model_registered_size, offset + copied, n);
        copied += n;
        cuda_model_load_progress_note(g_model_range_bytes + copied);
        chunk_idx++;
    }
    err = cudaStreamSynchronize(g_model_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model range upload sync failed for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }

    g_model_ranges.push_back({model_map, offset, bytes, dev, NULL, NULL, 0, 0, 1});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    cuda_model_load_progress_note(g_model_range_bytes);
    if (((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */) {
        fprintf(stderr, "ds4: CUDA fd-cached %s %.2f MiB (total %.2f GiB)\n",
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)g_model_range_bytes / 1073741824.0);
    }
    return (const char *)dev;
}

static int cuda_model_copy_chunked(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size) {
    if (!model_map || model_size == 0 || map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (0 ||
        0 ||
        0 ||
        0) {
        return 0;
    }
    if (g_model_device_owned || g_model_registered) return 1;

    void *dev = NULL;
    const double t0 = cuda_wall_sec();
    cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA model allocation skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    fprintf(stderr, "ds4: CUDA chunk-copying %.2f GiB model image\n",
            (double)model_size / 1073741824.0);

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    void *stage = NULL;
    err = cudaMallocHost(&stage, (size_t)chunk);
    if (err != cudaSuccess) {
        fprintf(stderr, "ds4: CUDA pinned model staging allocation failed: %s\n", cudaGetErrorString(err));
        (void)cudaFree(dev);
        (void)cudaGetLastError();
        return 0;
    }

    if (map_offset > 0) {
        uint64_t copied_header = 0;
        while (copied_header < map_offset) {
            const uint64_t n = (map_offset - copied_header < chunk) ? (map_offset - copied_header) : chunk;
            memcpy(stage, (const char *)model_map + copied_header, (size_t)n);
            err = cudaMemcpy((char *)dev + copied_header, stage, (size_t)n, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "ds4: CUDA model header copy failed: %s\n", cudaGetErrorString(err));
                (void)cudaFreeHost(stage);
                (void)cudaFree(dev);
                (void)cudaGetLastError();
                return 0;
            }
            copied_header += n;
        }
    }

    uint64_t copied = 0;
    double last_report = t0;
    while (copied < map_size) {
        const uint64_t n = (map_size - copied < chunk) ? (map_size - copied) : chunk;
        const uint64_t off = map_offset + copied;
        memcpy(stage, (const char *)model_map + off, (size_t)n);
        err = cudaMemcpy((char *)dev + off, stage, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "ds4: CUDA model chunk copy failed at %.2f GiB: %s\n",
                    (double)copied / 1073741824.0, cudaGetErrorString(err));
            (void)cudaFreeHost(stage);
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return 0;
        }
        cuda_model_discard_source_pages(model_map, model_size, off, n);
        copied += n;
        const double now = cuda_wall_sec();
        if (((const char *)0) /* DS4_CUDA_MODEL_COPY_VERBOSE: 诊断开关已删(2026-08-22) */ != NULL && now - last_report >= 2.0) {
            fprintf(stderr, "ds4: CUDA model chunk copy %.2f/%.2f GiB\n",
                    (double)copied / 1073741824.0,
                    (double)map_size / 1073741824.0);
            last_report = now;
        }
    }

    (void)cudaFreeHost(stage);
    g_model_device_base = (const char *)dev;
    g_model_device_owned = 1;
    g_model_hmm_direct = 0;
    const double t1 = cuda_wall_sec();
    fprintf(stderr,
            "ds4: CUDA model chunk copy complete in %.3fs (%.2f GiB tensors)\n",
            t1 - t0,
            (double)map_size / 1073741824.0);
    return 1;
}

static void cuda_model_range_release_all(void) {
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_registered && r.registered_base) {
            (void)cudaHostUnregister(r.registered_base);
        } else if (r.device_ptr && !r.arena_allocated) {
            (void)cudaFree(r.device_ptr);
        }
    }
    for (const cuda_model_arena &a : g_model_arenas) {
        if (a.device_ptr) (void)cudaFree(a.device_ptr);
    }
    g_model_arenas.clear();
    g_model_ranges.clear();
    g_model_range_by_offset.clear();
    g_model_range_bytes = 0;
    cuda_model_load_progress_reset();
}

static int cublas_ok(cublasStatus_t st, const char *what) {
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4: cuBLAS %s failed: status %d\n", what, (int)st);
    return 0;
}

int ds4_gpu_init(void) {
    int dev = 0;
    if (!cuda_ok(cudaSetDevice(dev), "set device")) return 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        fprintf(stderr, "ds4: CUDA backend initialized on %s (sm_%d%d)\n",
                prop.name, prop.major, prop.minor);
    }
    if (!g_cublas_ready) {
        if (!cublas_ok(cublasCreate(&g_cublas), "create handle")) return 0;
        const cublasMath_t math_mode =
            (g_quality_mode || 0)
                ? CUBLAS_DEFAULT_MATH
                : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);
        g_cublas_ready = 1;
    }
    return 1;
}

void ds4_gpu_cleanup(void) {
    (void)cudaDeviceSynchronize();
    if (g_cublas_ready) {
        (void)cublasDestroy(g_cublas);
        g_cublas_ready = 0;
        g_cublas = NULL;
    }
    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f32_ranges.clear();
    g_q8_f32_by_offset.clear();
    g_q8_f32_bytes = 0;
    if (g_cuda_tmp) {
        (void)cudaFree(g_cuda_tmp);
        g_cuda_tmp = NULL;
        g_cuda_tmp_bytes = 0;
    }
    for (size_t i = 0; i < 4; i++) {
        if (g_model_stage_event[i]) {
            (void)cudaEventDestroy(g_model_stage_event[i]);
            g_model_stage_event[i] = NULL;
        }
        if (g_model_stage_raw[i]) {
            (void)cudaFreeHost(g_model_stage_raw[i]);
            g_model_stage_raw[i] = NULL;
            g_model_stage[i] = NULL;
        }
    }
    g_model_stage_bytes = 0;
    if (g_model_upload_stream) {
        (void)cudaStreamDestroy(g_model_upload_stream);
        g_model_upload_stream = NULL;
    }
    if (g_model_device_owned && g_model_device_base) {
        (void)cudaFree((void *)g_model_device_base);
    }
    if (g_model_registered && g_model_host_base) {
        (void)cudaHostUnregister((void *)g_model_host_base);
    }
    g_model_host_base = NULL;
    g_model_device_base = NULL;
    g_model_registered_size = 0;
    g_model_registered = 0;
    g_model_device_owned = 0;
    g_model_range_mapping_supported = 1;
    g_model_hmm_direct = 0;
    g_model_fd = -1;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    g_model_file_size = 0;
    g_model_cache_full = 0;
    if (g_model_prefetch_stream) {
        (void)cudaStreamDestroy(g_model_prefetch_stream);
        g_model_prefetch_stream = NULL;
    }
}

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v);

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMalloc(&t->ptr, (size_t)bytes), "tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    return t;
}

ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMallocManaged(&t->ptr, (size_t)bytes), "managed tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    t->owner = 1;
    return t;
}

static uint64_t cuda_managed_kv_reserve_bytes(uint64_t total_bytes) {
    const uint64_t min_reserve = 8ull * 1073741824ull;
    const uint64_t max_reserve = 40ull * 1073741824ull;
    uint64_t reserve = total_bytes / 4u;
    if (reserve < min_reserve) reserve = min_reserve;
    if (reserve > max_reserve) reserve = max_reserve;
    return reserve;
}

int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes) {
    if (kv_cache_bytes == 0) return 0;

    /* Very large KV caches are where device-only cudaMalloc() can make a
     * unified-memory machine unresponsive.  Managed memory restores the old
     * demand-paged behavior for this one long-lived allocation class only. */
    const uint64_t huge_kv = 8ull * 1073741824ull;
    if (kv_cache_bytes >= huge_kv) return 1;

    const uint64_t large_context = 8ull * 1073741824ull;
    if (context_bytes < large_context) return 0;

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    const uint64_t reserve_bytes = cuda_managed_kv_reserve_bytes(total_bytes);
    if (context_bytes > free_bytes) return 1;
    return free_bytes - context_bytes < reserve_bytes;
}

ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base || offset > base->bytes || bytes > base->bytes - offset) return NULL;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->ptr = (char *)base->ptr + offset;
    t->bytes = bytes;
    t->owner = 0;
    return t;
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner && tensor->ptr) (void)cudaFree(tensor->ptr);
    free(tensor);
}

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    if (!tensor) return NULL;
    (void)cudaDeviceSynchronize();
    return tensor->ptr;
}

int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > tensor->bytes / sizeof(float)) return 0;
    if (count == 0) return 1;
    fill_f32_kernel<<<(count + 255u) / 256u, 256>>>((float *)tensor->ptr, count, value);
    return cuda_ok(cudaGetLastError(), "tensor fill f32 launch");
}

int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy((char *)tensor->ptr + offset, data, (size_t)bytes, cudaMemcpyHostToDevice), "tensor write");
}

int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy(data, (const char *)tensor->ptr + offset, (size_t)bytes, cudaMemcpyDeviceToHost), "tensor read");
}

int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                     const ds4_gpu_tensor *src, uint64_t src_offset,
                                     uint64_t bytes) {
    if (!dst || !src || dst_offset > dst->bytes || src_offset > src->bytes ||
        bytes > dst->bytes - dst_offset || bytes > src->bytes - src_offset) {
        return 0;
    }
    if (bytes == 0) return 1;
    /* DtoD 走异步(ptds 同流有序): 同步版每次吸收整条 GPU 队列等待, spec 捕获
     * 172 次/轮时是 verify 空隙主因(2026-08-21 nsys cudaMemcpy 54%)。 */
    return cuda_ok(cudaMemcpyAsync((char *)dst->ptr + dst_offset,
                              (const char *)src->ptr + src_offset,
                              (size_t)bytes,
                              cudaMemcpyDeviceToDevice),
                   "tensor copy");
}

/* ---- decode 单 token CUDA graph ----
 * 动机: decode 每 token ~900 个小 kernel launch, 提交间隙+延迟尾实测吃 ~10ms/token。
 * 方案(llama.cpp 同款): 每 token stream-capture(比真实提交便宜) → 首次 Instantiate,
 * 之后 cudaGraphExecUpdate 增量打参数补丁 → GraphLaunch 一次成图重放。
 * 前置: 本文件以 -default-stream per-thread 编译(legacy 流不可捕获, PTDS 让全部
 * 无流 launch 落在可捕获的 per-thread 流上; 其余 stream 的同步均已显式化)。
 * 任何一步失败 ⇒ 永久回退直跑(capture 态下 kernel 未执行, 调用方需重编码一遍)。 */
/* 乒乓双 exec 流水线(2026-08-17 第六夜): GPU 跑 token N 的图时, CPU 捕获/更新
 * token N+1 的图到另一个 exec 槽 —— encode(实测 1.1-4.6ms/token)整体移出临界路径。
 * token id 经"pinned 参数槽"间接: 图内录一条 4B H2D memcpy(捕获记录的是地址),
 * 发射前才写 *g_tok_id_host, 重放时读到最新值。竞态规避: capture 期间绝不写参数槽
 * (上一个图可能还没执行到它的 memcpy), 只暂存到 g_tok_id_want。 */
static cudaGraphExec_t g_tok_execs[4] = {NULL, NULL, NULL, NULL};   /* 08-20 四相: ratio-4 压缩器周期拓扑 */
static int g_tok_cur = 0;                 /* 最近发射的槽 */
static int g_tok_pending = -1;            /* 预编码完成待发射的槽 */
static uint32_t g_tok_pending_pos = 0;
static int g_tok_pending_logits = 0;
static int32_t *g_tok_id_dev = NULL;      /* device 侧 token id */
/* 双流并发(2026-08-17 第八夜): routed MoE 与 shared FFN 同读 ffn_norm、输出末端相加,
 * 数据流独立却串行。mark(主流, MoE 发射前) → begin(侧流 wait+切流) → join(主流 wait)。
 * capture 中 event record/wait 构建图依赖边; 资源须 capture 外创建。 */
static cudaStream_t g_side_stream = NULL;
static cudaEvent_t g_side_fork_ev = NULL, g_side_join_ev = NULL;
static cudaStream_t g_cur_stream = 0;     /* 0 == PTDS(-default-stream per-thread) */
static int g_side_active = 0;
static int32_t *g_tok_id_host = NULL;     /* pinned host 参数槽 */
static int32_t g_tok_id_want = 0;         /* capture 期暂存的 token */
static int g_tok_graph_on = -1;
static uint32_t g_tok_launch_pos = 0;   /* 本 token pos(end_launch 相位选槽用) */

static void side_stream_ensure(void) {
    if (g_side_stream) return;
    if (cudaStreamCreateWithFlags(&g_side_stream, cudaStreamNonBlocking) != cudaSuccess) {
        g_side_stream = NULL; (void)cudaGetLastError(); return;
    }
    if (cudaEventCreateWithFlags(&g_side_fork_ev, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&g_side_join_ev, cudaEventDisableTiming) != cudaSuccess) {
        g_side_fork_ev = NULL; g_side_join_ev = NULL;
        (void)cudaStreamDestroy(g_side_stream); g_side_stream = NULL;
        (void)cudaGetLastError();
    }
}

int ds4_gpu_side_mark(void) {
    if (((const char *)0) /* DS4_NO_SIDE_STREAM: 路径开关已删(2026-08-22 隐形炸弹清理) */) return 0;   /* 二分诊断: 关闭双流并发 */
    if (!g_side_stream || !g_side_fork_ev) {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(cudaStreamPerThread, &cs);
        if (cs != cudaStreamCaptureStatusNone) return 0;
        side_stream_ensure();
        if (!g_side_stream) return 0;
    }
    if (cudaEventRecord(g_side_fork_ev, cudaStreamPerThread) != cudaSuccess) {
        (void)cudaGetLastError(); return 0;
    }
    return 1;
}

int ds4_gpu_side_begin(void) {
    if (!g_side_stream || !g_side_fork_ev) return 0;
    if (cudaStreamWaitEvent(g_side_stream, g_side_fork_ev, 0) != cudaSuccess) {
        (void)cudaGetLastError(); return 0;
    }
    g_cur_stream = g_side_stream;
    g_side_active = 1;
    return 1;
}

/* 切回主流但保持 fork(侧流尾未 join): comp 链发完后主流继续 indexer/attention */
int ds4_gpu_side_main(void) {
    g_cur_stream = 0;
    return 1;
}

int ds4_gpu_side_join(void) {
    g_cur_stream = 0;
    if (!g_side_active) return 1;
    g_side_active = 0;
    if (cudaEventRecord(g_side_join_ev, g_side_stream) != cudaSuccess ||
        cudaStreamWaitEvent(cudaStreamPerThread, g_side_join_ev, 0) != cudaSuccess) {
        (void)cudaGetLastError(); return 0;
    }
    return 1;
}

__global__ static void hc_mean_slot_kernel(
        float *dst, const float *hc, uint32_t n_embd, uint32_t n_hc,
        uint32_t slot, uint32_t n_tokens);

/* ================= DSpark drafter 专用 kernels (2026-08-18) =================
 * 语义对齐 hf/inference/model.py DSparkAttention/forward_head:
 *  - KV 窗: [win=128] 行来自主模型融合 main_x 的 wkv(逐 decode 步写 pos%win), 环形;
 *  - 块内: 5 位草稿自身 kv, 因果(位 i 看窗全部 + 块内 ≤i);
 *  - sink + softmax scale + 输出反 rope 由调用方现有 entry 完成;
 *  - drafter KV 全 f32(草稿路径, verify 兜底正确性, 不追官方 fp8 bit 对齐)。 */
/* prefill 建窗 scatter: [n,512] 行写入环形窗 (pos0+t)%win */
__global__ static void dspark_win_scatter_kernel(
        float *win, const float *rows, uint32_t n, uint32_t pos0,
        uint32_t win_rows, uint32_t dim) {
    const uint32_t t = blockIdx.x;
    const uint32_t i = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= n || i >= dim) return;
    win[(uint64_t)((pos0 + t) % win_rows) * dim + i] = rows[(uint64_t)t * dim + i];
}

__global__ static void dspark_attn_kernel(
        float *heads,                  /* [blk, n_head, head_dim] */
        const float *sinks,            /* [n_head] */
        const float *q,                /* [blk, n_head, head_dim] (已 rope) */
        const float *win_kv,           /* [win, head_dim] 环形(行序=写入序) */
        const float *blk_kv,           /* [blk, head_dim] (已 rope) */
        uint32_t n_win,                /* 窗内有效行数(≤win) */
        uint32_t blk,                  /* 块位数(5) */
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t win_base,             /* 窗首位置在环里的行号 = (pos+1-n_win) % win_cap */
        uint32_t win_cap) {            /* 环容量(128) */
    const uint32_t t = blockIdx.x;     /* 块位 */
    const uint32_t h = blockIdx.y;     /* head */
    if (t >= blk || h >= n_head) return;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    const uint32_t n_score = n_win + blk;      /* 窗 + 块内全可见(官方 topk_idxs:
                                                * 每块位看到全部 5 位, 非因果; 块内
                                                * 因果性只在 markov 链的 logits 层面) */
    __shared__ float scores[256];
    __shared__ float partial[32];
    __shared__ float max_s, denom;
    const float scale = rsqrtf((float)head_dim);
    const uint32_t wp = threadIdx.x >> 5u, ln = threadIdx.x & 31u;
    const uint32_t hd4 = head_dim >> 2u;
    const float4 *q4 = (const float4 *)qh;
    for (uint32_t r = wp; r < n_score; r += blockDim.x >> 5u) {
        /* 环行按**位置**映射(2026-08-21 修): 原来读 ring[0..n_win-1] —— 上下文过 128 后
         * 会把 verify 批写进去的"被拒候选(未来位置)"行一起读进来污染草稿(实测 SPEC 下
         * 等效 p1 只有 0.62, 而 PROBE 干净窗口下 drafter 真实 p1=0.91)。 */
        const uint32_t ring = (win_cap && r < n_win) ? ((win_base + r) % win_cap) : r;
        const float *kvrow = (r < n_win) ? (win_kv + (uint64_t)ring * head_dim)
                                         : (blk_kv + (uint64_t)(r - n_win) * head_dim);
        const float4 *kv4 = (const float4 *)kvrow;
        float dot = 0.0f;
        for (uint32_t d = ln; d < hd4; d += 32u) {
            const float4 k = kv4[d], qq = q4[d];
            dot += qq.x * k.x + qq.y * k.y + qq.z * k.z + qq.w * k.w;
        }
        for (uint32_t off = 16u; off > 0u; off >>= 1u) dot += __shfl_down_sync(0xffffffffu, dot, off);
        if (ln == 0) scores[r] = dot * scale;
    }
    __syncthreads();
    float lm = sinks[h];
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) lm = fmaxf(lm, scores[i]);
    for (uint32_t off = 16u; off > 0u; off >>= 1u) lm = fmaxf(lm, __shfl_down_sync(0xffffffffu, lm, off));
    if ((threadIdx.x & 31u) == 0) partial[threadIdx.x >> 5u] = lm;
    __syncthreads();
    if (threadIdx.x < 32u) {
        const uint32_t nw = (blockDim.x + 31u) >> 5u;
        float m = (threadIdx.x < nw) ? partial[threadIdx.x] : -INFINITY;
        for (uint32_t off = 16u; off > 0u; off >>= 1u) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
        if (threadIdx.x == 0) max_s = m;
    }
    __syncthreads();
    float dl = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        dl += scores[i];
    }
    for (uint32_t off = 16u; off > 0u; off >>= 1u) dl += __shfl_down_sync(0xffffffffu, dl, off);
    if ((threadIdx.x & 31u) == 0) partial[threadIdx.x >> 5u] = dl;
    __syncthreads();
    if (threadIdx.x < 32u) {
        const uint32_t nw = (blockDim.x + 31u) >> 5u;
        float s2 = (threadIdx.x < nw) ? partial[threadIdx.x] : 0.0f;
        for (uint32_t off = 16u; off > 0u; off >>= 1u) s2 += __shfl_down_sync(0xffffffffu, s2, off);
        if (threadIdx.x == 0) denom = s2 + expf(sinks[h] - max_s);
    }
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < n_score; r++) {
            const uint32_t ring2 = (win_cap && r < n_win) ? ((win_base + r) % win_cap) : r;
            const float *kvrow = (r < n_win) ? (win_kv + (uint64_t)ring2 * head_dim)
                                             : (blk_kv + (uint64_t)(r - n_win) * head_dim);
            acc += kvrow[d] * scores[r];
        }
        oh[d] = acc / denom;
    }
}

/* DSpark 置信头(2026-08-21, 论文 2607.05147 Alg.1 的输入量):
 *   c_k = sigmoid( w · [ x_k ; W1[prev_k] ] ),  w 是 [dim+rank] f32, W1 是 q8_0 [vocab, rank]。
 * x_k = hc_head 之后、norm 之前的 drafter 隐状态(与官方 model.py DSparkConfidenceHead 一致)。
 * 一次算完 block 内全部位置: 每块一个位置, 块内并行点积 + 固定序归约。 */
__global__ static void dspark_confidence_kernel(
        float *out_conf,               /* [n_pos] */
        const float *x,                /* [n_pos, dim] hc_head 输出(pre-norm) */
        const float *w,                /* [dim + rank] f32 */
        const char *w1_q8,             /* q8_0 [vocab, rank] */
        const int32_t *prev_ids,       /* [n_pos] 每位置的输入 token */
        uint32_t dim, uint32_t rank, uint32_t n_pos) {
    const uint32_t p = blockIdx.x;
    if (p >= n_pos) return;
    const float *xp = x + (uint64_t)p * dim;
    float acc = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) acc += w[i] * xp[i];
    const uint32_t rank_blocks = rank / 32u;
    const int32_t prev = prev_ids[p];
    for (uint32_t b = threadIdx.x; b < rank_blocks; b += blockDim.x) {
        const uint8_t *blkp = (const uint8_t *)(w1_q8 + ((uint64_t)prev * rank_blocks + b) * 34u);
        uint16_t dh; memcpy(&dh, blkp, 2);
        const float d = __half2float(__ushort_as_half(dh));
        const int8_t *qs = (const int8_t *)(blkp + 2);
        float s = 0.0f;
        for (uint32_t i = 0; i < 32u; i++) s += (float)qs[i] * w[dim + b * 32u + i];
        acc += d * s;
    }
    __shared__ float part[256];
    part[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) part[threadIdx.x] += part[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) out_conf[p] = 1.0f / (1.0f + __expf(-part[0]));
}

int ds4_gpu_dspark_confidence_tensor(
        ds4_gpu_tensor *out_conf,
        const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size,
        uint64_t conf_w_offset, uint64_t markov_w1_offset,
        const ds4_gpu_tensor *prev_ids,
        uint32_t dim, uint32_t rank, uint32_t vocab, uint32_t n_pos) {
    if (!out_conf || !x || !prev_ids || !model_map || n_pos == 0 || (rank % 32u) != 0u) return 0;
    const uint64_t wb = (uint64_t)(dim + rank) * sizeof(float);
    const uint64_t w1b = (uint64_t)vocab * (rank / 32u) * 34ull;
    const char *w = cuda_model_range_ptr(model_map, conf_w_offset, wb, "dspark_conf");
    const char *w1 = cuda_model_range_ptr(model_map, markov_w1_offset, w1b, "dspark_markov_w1");
    if (!w || !w1) return 0;
    dspark_confidence_kernel<<<n_pos, 256, 0, g_cur_stream>>>(
            (float *)out_conf->ptr, (const float *)x->ptr, (const float *)w, w1,
            (const int32_t *)prev_ids->ptr, dim, rank, n_pos);
    return cuda_ok(cudaGetLastError(), "dspark confidence");
}

/* markov 链: 每位置 logits += w2 @ w1[prev]; argmax 选出下一位 prev(串行 5 次调用,
 * 每次一个位置)。w1: [rank=256] 行 per token(q8_0 反量化在此内联? 简化: w1/w2 由调用
 * 方保证 f32 缓存(2×129280×256×4B=265MB 太大) —— 改: w1 行(256 f32)由小 kernel 从
 * q8_0 现场解一行, w2 是 [vocab,256] q8_0 逐行 dot。 */
__global__ static void dspark_markov_bias_argmax_kernel(
        int32_t *out_id,               /* [1] 本位置 argmax */
        float *logits,                 /* [vocab] 本位置(原地加 bias) */
        const char *w1_q8,             /* q8_0 [vocab, 256] */
        const char *w2_q8,             /* q8_0 [vocab, 256] */
        const int32_t *prev_id,        /* [1] */
        uint32_t vocab,
        uint32_t rank) {
    /* 阶段1: 块 0 解 w1[prev] 到共享 → 全网格用 __grid_constant__ 不行, 用两 kernel?
     * 简化: 每块自解 w1[prev](256 元素 q8_0 = 8 块×34B, 开销小), 各块算自己 vocab 段的
     * bias 并加到 logits; argmax 用原子(int 打包 value|idx)。 */
    __shared__ float w1row[256];
    const uint32_t rank_blocks = rank / 32u;
    const int32_t prev = *prev_id;
    for (uint32_t b = threadIdx.x; b < rank_blocks; b += blockDim.x) {
        const uint8_t *blkp = (const uint8_t *)(w1_q8 + ((uint64_t)prev * rank_blocks + b) * 34u);
        uint16_t dh; memcpy(&dh, blkp, 2);
        const float d = __half2float(__ushort_as_half(dh));
        const int8_t *qs = (const int8_t *)(blkp + 2);
        for (uint32_t i = 0; i < 32u; i++) w1row[b * 32u + i] = d * (float)qs[i];
    }
    __syncthreads();
    const uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < vocab) {
        const uint8_t *row = (const uint8_t *)(w2_q8 + (uint64_t)v * rank_blocks * 34u);
        float bias = 0.0f;
        for (uint32_t b = 0; b < rank_blocks; b++) {
            const uint8_t *blkp = row + b * 34u;
            uint16_t dh; memcpy(&dh, blkp, 2);
            const float d = __half2float(__ushort_as_half(dh));
            const int8_t *qs = (const int8_t *)(blkp + 2);
            float s = 0.0f;
            for (uint32_t i = 0; i < 32u; i++) s += (float)qs[i] * w1row[b * 32u + i];
            bias += d * s;
        }
        logits[v] += bias;
    }
    /* argmax 二段: 由独立 kernel 做(见 dspark_argmax_kernel) */
    (void)out_id;
}

__global__ static void dspark_argmax_kernel(int32_t *out_id, const float *logits, uint32_t vocab) {
    __shared__ float bm[256];
    __shared__ int32_t bi[256];
    float m = -INFINITY; int32_t mi = 0;
    for (uint32_t v = threadIdx.x; v < vocab; v += blockDim.x) {
        const float x = logits[v];
        if (x > m) { m = x; mi = (int32_t)v; }
    }
    bm[threadIdx.x] = m; bi[threadIdx.x] = mi;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (bm[threadIdx.x + s] > bm[threadIdx.x] ||
                (bm[threadIdx.x + s] == bm[threadIdx.x] && bi[threadIdx.x + s] < bi[threadIdx.x])) {
                bm[threadIdx.x] = bm[threadIdx.x + s]; bi[threadIdx.x] = bi[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out_id = bi[0];
}

/* DSpark: 抓 target 层 HC 均值到 main_hidden[t][slot]; n_tokens 支持 batch prefill */
int ds4_gpu_dspark_hc_mean_tensor(
        ds4_gpu_tensor *dst, const ds4_gpu_tensor *hc,
        uint32_t n_embd, uint32_t n_hc, uint32_t slot, uint32_t n_tokens) {
    if (!dst || !hc || n_embd == 0 || n_hc == 0 || slot >= 3u || n_tokens == 0) return 0;
    if (hc->bytes < (uint64_t)n_tokens * n_hc * n_embd * sizeof(float) ||
        dst->bytes < (uint64_t)n_tokens * 3u * n_embd * sizeof(float)) return 0;
    hc_mean_slot_kernel<<<dim3((n_embd + 255u) / 256u, n_tokens, 1), 256, 0, g_cur_stream>>>(
        (float *)dst->ptr, (const float *)hc->ptr, n_embd, n_hc, slot, n_tokens);
    return cuda_ok(cudaGetLastError(), "dspark hc mean launch");
}

int ds4_gpu_dspark_attn_tensor(
        ds4_gpu_tensor *heads,
        const void *model_map, uint64_t model_size, uint64_t sinks_offset,
        const ds4_gpu_tensor *q, const ds4_gpu_tensor *win_kv, const ds4_gpu_tensor *blk_kv,
        uint32_t n_win, uint32_t blk, uint32_t n_head, uint32_t head_dim, uint32_t win_base, uint32_t win_cap) {
    if (!heads || !q || !win_kv || !blk_kv || blk == 0 || n_head == 0 || head_dim == 0) return 0;
    if ((head_dim & 3u) != 0 || n_win + blk > 256u) return 0;
    if (sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset) return 0;
    const char *sinks = cuda_model_range_ptr(model_map, sinks_offset,
                                             (uint64_t)n_head * sizeof(float), "dspark_sinks");
    if (!sinks) return 0;
    dspark_attn_kernel<<<dim3(blk, n_head, 1), 256, 0, g_cur_stream>>>(
        (float *)heads->ptr, (const float *)sinks, (const float *)q->ptr,
        (const float *)win_kv->ptr, (const float *)blk_kv->ptr,
        n_win, blk, n_head, head_dim, win_base, win_cap);
    return cuda_ok(cudaGetLastError(), "dspark attn launch");
}

int ds4_gpu_dspark_win_scatter_tensor(
        ds4_gpu_tensor *win, const ds4_gpu_tensor *rows,
        uint32_t n, uint32_t pos0, uint32_t win_rows, uint32_t dim) {
    if (!win || !rows || n == 0 || win_rows == 0 || dim == 0) return 0;
    if (rows->bytes < (uint64_t)n * dim * sizeof(float) ||
        win->bytes < (uint64_t)win_rows * dim * sizeof(float)) return 0;
    dspark_win_scatter_kernel<<<dim3(n, (dim + 255u) / 256u, 1), 256, 0, g_cur_stream>>>(
        (float *)win->ptr, (const float *)rows->ptr, n, pos0, win_rows, dim);
    return cuda_ok(cudaGetLastError(), "dspark win scatter launch");
}


/* markov 一步: logits 行原地加 bias(w2@w1[prev]) 并 argmax → out_id(device int32) */
int ds4_gpu_dspark_markov_step_tensor(
        ds4_gpu_tensor *out_id, ds4_gpu_tensor *logits_row,
        const void *model_map, uint64_t model_size,
        uint64_t w1_offset, uint64_t w2_offset,
        const ds4_gpu_tensor *prev_id,
        uint32_t vocab, uint32_t rank) {
    if (!out_id || !logits_row || !prev_id || vocab == 0 || rank == 0 || (rank & 31u)) return 0;
    const uint64_t wbytes = (uint64_t)vocab * (rank / 32u) * 34u;
    if (w1_offset > model_size || wbytes > model_size - w1_offset ||
        w2_offset > model_size || wbytes > model_size - w2_offset) return 0;
    const char *w1 = cuda_model_range_ptr(model_map, w1_offset, wbytes, "dspark_markov_w1");
    const char *w2 = cuda_model_range_ptr(model_map, w2_offset, wbytes, "dspark_markov_w2");
    if (!w1 || !w2) return 0;
    dspark_markov_bias_argmax_kernel<<<(vocab + 255u) / 256u, 256, 0, g_cur_stream>>>(
        (int32_t *)out_id->ptr, (float *)logits_row->ptr, w1, w2,
        (const int32_t *)prev_id->ptr, vocab, rank);
    dspark_argmax_kernel<<<1, 256, 0, g_cur_stream>>>(
        (int32_t *)out_id->ptr, (const float *)logits_row->ptr, vocab);
    return cuda_ok(cudaGetLastError(), "dspark markov launch");
}

int ds4_gpu_dspark_argmax_only_tensor(ds4_gpu_tensor *out_id,
                                                 const ds4_gpu_tensor *logits_row, uint32_t vocab) {
    if (!out_id || !logits_row || vocab == 0) return 0;
    dspark_argmax_kernel<<<1, 256, 0, g_cur_stream>>>(
        (int32_t *)out_id->ptr, (const float *)logits_row->ptr, vocab);
    return cuda_ok(cudaGetLastError(), "dspark argmax launch");
}

static void tok_graph_ensure_id_slot(void) {
    side_stream_ensure();
    if (!g_tok_id_dev &&
        cudaMalloc(&g_tok_id_dev, sizeof(int32_t)) != cudaSuccess) {
        g_tok_id_dev = NULL; (void)cudaGetLastError();
    }
    if (!g_tok_id_host &&
        cudaMallocHost(&g_tok_id_host, sizeof(int32_t)) != cudaSuccess) {
        g_tok_id_host = NULL; (void)cudaGetLastError();
    }
}

/* graph → exec 槽: 首次 Instantiate, 之后 ExecUpdate, 不匹配则重建 */
static int tok_graph_finalize_slot(cudaGraph_t graph, int slot) {
    if (!g_tok_execs[slot]) {
        if (cudaGraphInstantiate(&g_tok_execs[slot], graph, NULL, NULL, 0) != cudaSuccess) {
            (void)cudaGetLastError();
            g_tok_execs[slot] = NULL;
            return 0;
        }
        return 1;
    }
    cudaGraphExecUpdateResult ur;
    cudaGraphNode_t err_node = NULL;
    if (cudaGraphExecUpdate(g_tok_execs[slot], graph, &err_node, &ur) != cudaSuccess) {
        /* 910 竞争根修(08-18 sanitizer 实锤)修订(08-20): 原修=全设备同步再销毁, 但该同步
         * 等的是"刚发射的当前图"(27.5ms 隔拍阻塞实测) — 而被销毁的 exec 在两处调用点都
         * 满足"上次发射 ≥1 个 end_commands 全同步之前"不变量(每 token 末尾必 DeviceSync),
         * 恒已静默。撤同步; DS4_TOK_GRAPH_SAFE_SYNC=1 可恢复旧行为(sanitizer 排查用)。 */
        if (((const char *)0) /* DS4_TOK_GRAPH_DEBUG: 诊断开关已删(2026-08-22) */)
            fprintf(stderr, "[tokdbg] update REJECT ur=%d\n", (int)ur);
        (void)cudaGetLastError();
        if (((const char *)0) /* DS4_TOK_GRAPH_SAFE_SYNC: 路径开关已删(2026-08-22 隐形炸弹清理) */) (void)cudaDeviceSynchronize();
        (void)cudaGraphExecDestroy(g_tok_execs[slot]);
        g_tok_execs[slot] = NULL;
        if (cudaGraphInstantiate(&g_tok_execs[slot], graph, NULL, NULL, 0) != cudaSuccess) {
            (void)cudaGetLastError();
            return 0;
        }
    }
    return 1;
}

/* GPU 跨度计时(2026-08-21 verify 归因): 事件对夹住一段, end 返回 GPU 侧毫秒。
 * 与 host 计时对照即可分离"GPU 真忙" vs "CPU 编码/背压阻塞"。 */
static cudaEvent_t g_span_a = NULL, g_span_b = NULL;
void ds4_gpu_span_begin(void) {
    if (!g_span_a) { if (cudaEventCreate(&g_span_a) != cudaSuccess) { g_span_a = NULL; return; } }
    if (!g_span_b) { if (cudaEventCreate(&g_span_b) != cudaSuccess) { g_span_b = NULL; return; } }
    (void)cudaEventRecord(g_span_a, cudaStreamPerThread);
}
float ds4_gpu_span_end(void) {
    if (!g_span_a || !g_span_b) return -1.0f;
    if (cudaEventRecord(g_span_b, cudaStreamPerThread) != cudaSuccess) return -1.0f;
    if (cudaEventSynchronize(g_span_b) != cudaSuccess) return -1.0f;
    float ms = -1.0f;
    (void)cudaEventElapsedTime(&ms, g_span_a, g_span_b);
    return ms;
}

/* verify 批 CUDA 图(2026-08-21): 实测每轮发 ~3494 个 kernel, 平均相邻间隙 5.4us
 * => 每轮 ~19ms GPU 空转(利用率仅 69%)。decode 早已用 token graph 消除这一项, 批路径
 * 一直没有。这里按 (pos0 & 3) 四相建槽(ratio-4 压缩器周期决定拓扑), 同相拓扑同构 =>
 * cudaGraphExecUpdate 只改 kernel 参数; 失败则回退直发(调用方重编码)。 */
static cudaGraphExec_t g_bat_execs[4] = {NULL, NULL, NULL, NULL};
static int g_bat_graph_on = -1;
static int g_bat_slot = -1;

int ds4_gpu_batch_graph_begin(int slot) {
    /* 默认关(2026-08-21 A/B): verify 64.9 vs 65.6 = 噪声内。批 kernel 比 decode 大 3x,
     * launch 开销占比小 => 图收益消失; 31% 空转是 kernel 间排空(ramp/drain)不是发射延迟。 */
    if (g_bat_graph_on < 0) g_bat_graph_on = ((const char *)0) /* DS4_CUDA_BATCH_GRAPH: 路径开关已删(2026-08-22 隐形炸弹清理) */ ? 1 : 0;
    if (!g_bat_graph_on || slot < 0 || slot > 3) return 0;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &cs) == cudaSuccess &&
        cs != cudaStreamCaptureStatusNone) return 0;   /* 已在别的 capture 里 */
    if (cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        (void)cudaGetLastError();
        g_bat_graph_on = 0;
        return 0;
    }
    g_bat_slot = slot;
    return 1;
}

int ds4_gpu_batch_graph_end_launch(int encode_ok) {
    if (g_bat_slot < 0) return 0;
    const int slot = g_bat_slot;
    g_bat_slot = -1;
    cudaGraph_t graph = NULL;
    const cudaError_t e = cudaStreamEndCapture(cudaStreamPerThread, &graph);
    if (e != cudaSuccess || !graph || !encode_ok) {
        (void)cudaGetLastError();
        if (graph) (void)cudaGraphDestroy(graph);
        g_bat_graph_on = 0;   /* 一次失败就永久回退, 不反复试探 */
        fprintf(stderr, "ds4: CUDA batch graph capture failed; falling back to direct launch\n");
        return -1;
    }
    int ok = 1;
    if (!g_bat_execs[slot]) {
        if (cudaGraphInstantiate(&g_bat_execs[slot], graph, NULL, NULL, 0) != cudaSuccess) {
            (void)cudaGetLastError(); g_bat_execs[slot] = NULL; ok = 0;
        }
    } else {
        cudaGraphExecUpdateResult ur;
        cudaGraphNode_t err_node = NULL;
        if (cudaGraphExecUpdate(g_bat_execs[slot], graph, &err_node, &ur) != cudaSuccess) {
            (void)cudaGetLastError();
            (void)cudaGraphExecDestroy(g_bat_execs[slot]);
            g_bat_execs[slot] = NULL;
            if (cudaGraphInstantiate(&g_bat_execs[slot], graph, NULL, NULL, 0) != cudaSuccess) {
                (void)cudaGetLastError(); g_bat_execs[slot] = NULL; ok = 0;
            }
        }
    }
    (void)cudaGraphDestroy(graph);
    if (!ok) { g_bat_graph_on = 0; return -1; }
    if (cudaGraphLaunch(g_bat_execs[slot], cudaStreamPerThread) != cudaSuccess) {
        (void)cudaGetLastError();
        g_bat_graph_on = 0;
        return -1;
    }
    return 1;
}

/* 路由空槽消毒(2026-08-21 抓到 drafter 不可复现的真因): drafter 约 1.1% 的路由槽是 -1
 * (router 在退化行上选不满 top-k)。引擎只在"使用点"clamp(<0 → 0), 但 sorted-pairs 记账
 * 走 counts[selected[pair]] —— **counts[-1] 是越界原子写**, 同时该 pair 拿不到 tile ⇒ 它的
 * mid 槽从未被写 ⇒ down 读到未初始化内存。症状: 同输入两跑草稿完全不同(logits 差 5%),
 * 偶发 NaN, acc 大幅波动。这里在记账前把 -1 归零并把其权重清零(语义等价: 空槽无贡献)。 */
__global__ static void sanitize_router_kernel(int32_t *sel, float *w, uint32_t n, uint32_t n_expert) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t e = sel[i];
    if (e < 0 || (uint32_t)e >= n_expert) { sel[i] = 0; if (w) w[i] = 0.0f; }
}

int ds4_gpu_sanitize_router_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
                                              uint32_t n_pairs, uint32_t n_total_expert) {
    if (!selected || n_pairs == 0) return 0;
    sanitize_router_kernel<<<(n_pairs + 255u) / 256u, 256, 0, g_cur_stream>>>(
        (int32_t *)selected->ptr, weights ? (float *)weights->ptr : NULL, n_pairs, n_total_expert);
    return cuda_ok(cudaGetLastError(), "sanitize router");
}

__global__ static void sanitize_finite_kernel(float *x, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    if (!isfinite(v)) x[i] = 0.0f;
}

/* drafter 数值护栏(2026-08-21): DSpark drafter 约 1-4% 的行会产出 NaN(实测锚里
 * blk0 干净输入也能出 10 个 NaN 维, 新旧两条 MoE kernel 同现 ⇒ 既有问题不是本轮改动)。
 * NaN 一旦落到锚位就整轮草稿作废(acc 塌到 1.00 的间歇故障)。这里在每块出口清洗,
 * 代价 = B×hc_dim 一次写。 */
int ds4_gpu_sanitize_finite_tensor(ds4_gpu_tensor *t, uint64_t n_float) {
    if (!t || n_float == 0 || t->bytes < n_float * sizeof(float)) return 0;
    sanitize_finite_kernel<<<(unsigned)((n_float + 255u) / 256u), 256, 0, g_cur_stream>>>(
        (float *)t->ptr, n_float);
    return cuda_ok(cudaGetLastError(), "sanitize finite");
}

void ds4_gpu_token_graph_set_pos(uint32_t pos) { g_tok_launch_pos = pos; }
int ds4_gpu_token_graph_begin(void) {
    if (g_tok_graph_on < 0)
        /* ★decode token graph 整族关闭(2026-08-22 实测判定)★
         * 图在建图那次算对, 之后重放时逐 token 变化的注意力范围参数(KV 行数/raw 窗口起点/
         * 可见压缩条目数)没有被正确打补丁 —— 每个解码 token 都在用错误的历史范围。
         * 症状: 位置 0 正确, 之后系统性变差。实测同一序列同一模型:
         *     批量 prefill 路      PPL = 44.10
         *     解码路 token图【开】 PPL = 55.30   ← 差 25%
         *     解码路 token图【关】 PPL = 45.72   ← 回到批量路水平
         * 影响面: decode 是生成的唯一路径, 所有生成质量都被它压着; 本项目此前全部判决
         * 数字都是在这个 bug 上量的。代价: generation 39.18 → 36.05 t/s(−8%), 换 25% PPL。
         * 按"删除而非默认关闭"处理, 不留开关; 要恢复须先修好参数补丁并逐位对拍。 */
        g_tok_graph_on = 0;
    if (!g_tok_graph_on) return 0;
    tok_graph_ensure_id_slot();
    if (cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        (void)cudaGetLastError();
        g_tok_graph_on = 0;
        return 0;
    }
    return 1;
}

int ds4_gpu_token_graph_end_launch(void) {
    if (!g_tok_graph_on) return 0;
    cudaGraph_t graph = NULL;
    if (cudaStreamEndCapture(cudaStreamPerThread, &graph) != cudaSuccess || !graph) {
        (void)cudaGetLastError();
        g_tok_graph_on = 0;
        if (graph) (void)cudaGraphDestroy(graph);
        fprintf(stderr, "ds4: CUDA token graph capture failed; falling back to direct launch\n");
        return -1;   /* 捕获期 kernel 未执行, 调用方必须重编码 */
    }
    const int slot_el = (int)(g_tok_launch_pos & 3u);   /* 四相: 同相拓扑同构 update 恒过 */
    const int ok = tok_graph_finalize_slot(graph, slot_el);
    (void)cudaGraphDestroy(graph);
    if (!ok) {
        g_tok_graph_on = 0;
        fprintf(stderr, "ds4: CUDA token graph instantiate failed; falling back\n");
        return -1;
    }
    g_tok_cur = slot_el;
    if (g_tok_id_host) *g_tok_id_host = g_tok_id_want;   /* 发射前落参数槽(此刻流已静) */
    if (cudaGraphLaunch(g_tok_execs[slot_el], cudaStreamPerThread) != cudaSuccess) {
        (void)cudaGetLastError();
        g_tok_graph_on = 0;
        return -1;
    }
    return 1;
}

/* 命中预编码: 写参数槽后直接发射, encode 零成本。返回 1=已发射。 */
int ds4_gpu_token_graph_try_pending(int token, uint32_t pos, int need_logits) {
    if (((const char *)0) /* DS4_TOK_GRAPH_DEBUG: 诊断开关已删(2026-08-22) */ && (g_tok_graph_on <= 0 || g_tok_pending < 0 ||
        g_tok_pending_pos != pos || g_tok_pending_logits != need_logits ||
        !g_tok_execs[g_tok_pending < 0 ? 0 : g_tok_pending]))
        fprintf(stderr, "[tokdbg] miss pos=%u: on=%d pending=%d ppos=%u plog=%d\n",
                pos, g_tok_graph_on, g_tok_pending, g_tok_pending_pos, g_tok_pending_logits);
    if (g_tok_graph_on <= 0 || g_tok_pending < 0 ||
        g_tok_pending_pos != pos || g_tok_pending_logits != need_logits ||
        !g_tok_execs[g_tok_pending])
        return 0;
    const int slot = g_tok_pending;
    g_tok_pending = -1;
    if (g_tok_id_host) *g_tok_id_host = (int32_t)token;
    if (cudaGraphLaunch(g_tok_execs[slot], cudaStreamPerThread) != cudaSuccess) {
        (void)cudaGetLastError();
        g_tok_graph_on = 0;
        return 0;
    }
    g_tok_cur = slot;
    return 1;
}

int ds4_gpu_token_graph_precapture_begin(void) {
    if (g_tok_graph_on <= 0 || !g_tok_id_dev || !g_tok_id_host) {
        if (((const char *)0) /* DS4_TOK_GRAPH_DEBUG: 诊断开关已删(2026-08-22) */)
            fprintf(stderr, "[tokdbg] precap-skip: on=%d id_dev=%d id_host=%d\n",
                    g_tok_graph_on, g_tok_id_dev != NULL, g_tok_id_host != NULL);
        return 0;
    }
    if (cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        if (((const char *)0) /* DS4_TOK_GRAPH_DEBUG: 诊断开关已删(2026-08-22) */)
            fprintf(stderr, "[tokdbg] precap BeginCapture FAIL\n");
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static double tokdbg_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}
int ds4_gpu_token_graph_precapture_end(uint32_t pos, int need_logits, int encode_ok) {
    const int dbg = ((const char *)0) /* DS4_TOK_GRAPH_DEBUG: 诊断开关已删(2026-08-22) */ != NULL;
    const double td0 = dbg ? tokdbg_now() : 0.0;
    cudaGraph_t graph = NULL;
    if (cudaStreamEndCapture(cudaStreamPerThread, &graph) != cudaSuccess) {
        (void)cudaGetLastError();
        if (graph) (void)cudaGraphDestroy(graph);
        g_tok_pending = -1;
        return 0;
    }
    if (!encode_ok || !graph) {
        if (((const char *)0) /* DS4_TOK_GRAPH_DEBUG: 诊断开关已删(2026-08-22) */)
            fprintf(stderr, "[tokdbg] precap-end reject: encode_ok=%d graph=%d\n", encode_ok, graph != NULL);
        if (graph) (void)cudaGraphDestroy(graph);
        g_tok_pending = -1;
        return 0;
    }
    const double td1 = dbg ? tokdbg_now() : 0.0;
    if (((const char *)0) /* DS4_TOK_GRAPH_DOT: 路径开关已删(2026-08-22 隐形炸弹清理) */) {   /* 奇偶图导出 diff 用: pos 8/9 各一张 */
        if (pos == 7u || pos == 9u) {
            char fp[64]; snprintf(fp, sizeof fp, "/tmp/tokgraph_pos%u.dot", pos);
            (void)cudaGraphDebugDotPrint(graph, fp, 0);
        }
    }
    const int slot = (int)(pos & 3u);    /* 四相分槽; 目标槽上次发射≥1个全同步前=已静默 */
    const int ok = tok_graph_finalize_slot(graph, slot);
    const double td2 = dbg ? tokdbg_now() : 0.0;
    (void)cudaGraphDestroy(graph);
    if (dbg) fprintf(stderr, "[tokdbg] precap-end pos=%u endcap=%.1fms finalize=%.1fms destroy=%.1fms\n",
                     pos, td1 - td0, td2 - td1, tokdbg_now() - td2);
    if (!ok) { g_tok_pending = -1; return 0; }
    g_tok_pending = slot;
    g_tok_pending_pos = pos;
    g_tok_pending_logits = need_logits;
    return 1;
}

int ds4_gpu_begin_commands(void) { return 1; }
int ds4_gpu_flush_commands(void) {
    /* 在 token-graph capture 期间, flush(全设备同步)既非法也无意义 —— 图作为整体
     * 执行, 中途没有 host 可见状态。capture 态直接成功返回。 */
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &cs) == cudaSuccess &&
        cs == cudaStreamCaptureStatusActive) {
        return 1;
    }
    (void)cudaGetLastError();
    return cuda_ok(cudaDeviceSynchronize(), "flush");
}
int ds4_gpu_end_commands(void) {
    /* capture 期间全设备同步既非法也无意义(图整体执行), 直接成功返回。 */
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &cs) == cudaSuccess &&
        cs == cudaStreamCaptureStatusActive) return 1;
    return cuda_ok(cudaDeviceSynchronize(), "end commands");
}
int ds4_gpu_synchronize(void) { return cuda_ok(cudaDeviceSynchronize(), "synchronize"); }
/* TP rendezvous: CUDA has no MTLSharedEvent fast path, but ds4_gpu_flush_commands
 * already does a full device sync, so by the time the host waits the results are
 * visible. signal returns a nonzero token; host_wait is a no-op success. */
uint64_t ds4_gpu_tp_signal_after_batch(void) { return 1; }
int ds4_gpu_tp_host_wait(uint64_t value) { (void)value; return 1; }

/* 副模型 map 注册(2026-08-21 draft 5x 慢根因): 主模型整文件 cudaHostRegister 后,
 * cuda_model_ptr 对"非主 base"退回裸 host 指针 —— 未注册路径在 GB10 上按缺页走,
 * drafter FFN 实测 12ms/层 vs verify 2.3ms/层。这里把副 map(DS4_DRAFT_GGUF)整体注册
 * 并挂进 g_model_ranges, 让 range 查找先命中 ⇒ 与主模型同速。 */
int ds4_gpu_register_aux_model_map(const void *map, uint64_t size) {
    if (!map || size == 0) return 0;
    for (const cuda_model_range &r : g_model_ranges)
        if (r.host_base == map && r.offset == 0 && r.bytes >= size) return 1;
    /* ★设备驻留拷贝(2026-08-21 定版): 早先用 cudaHostRegister+HostGetDevicePointer 整体
     * 映射副 gguf, 在 5.59GB 文件映射上**不可靠** —— drafter 同输入两跑 MoE 输出不同
     * (Fin/路由/权重逐字节相同), 注册前触实所有页也无效。改为 cudaMalloc + H2D 拷贝:
     * GB10 统一内存上 5.6GB 完全负担得起, 且拿到的是正规设备页表(既确定又快)。
     * DS4_AUX_HOSTREG=1 可退回旧映射法做对照; 拷贝失败则回落按 range 懒注册(正确但慢)。 */
    if (1) {
        void *dev = NULL;
        cudaError_t e2 = cudaMalloc(&dev, (size_t)size);
        if (e2 == cudaSuccess) {
            e2 = cudaMemcpy(dev, map, (size_t)size, cudaMemcpyHostToDevice);
            if (e2 == cudaSuccess) {
                g_model_ranges.push_back({map, 0, size, (char *)dev, NULL, NULL, 0, 0, 0});
                fprintf(stderr, "ds4: CUDA aux map device-resident copy %.2f GiB\n",
                        (double)size / 1073741824.0);
                return 1;
            }
            (void)cudaFree(dev);
        }
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: CUDA aux map device copy failed (%.2f GiB); 回落懒注册\n",
                (double)size / 1073741824.0);
        return 0;
    }
    void *dev = NULL;
    cudaError_t err = cudaHostRegister((void *)map, (size_t)size, cudaHostRegisterMapped);
    if (err != cudaSuccess) { (void)cudaGetLastError(); return 0; }
    if (cudaHostGetDevicePointer(&dev, (void *)map, 0) != cudaSuccess || !dev) {
        (void)cudaGetLastError(); (void)cudaHostUnregister((void *)map); return 0;
    }
    g_model_ranges.push_back({map, 0, size, (char *)dev, (void *)map, (char *)dev, size, 1, 0});
    fprintf(stderr, "ds4: CUDA registered aux map %.2f GiB (host-mapped, 旧法)\n",
            (double)size / 1073741824.0);
    return 1;
}

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    if (g_model_host_base == model_map && g_model_registered_size == model_size) return 1;
    cuda_model_range_release_all();
    cuda_q8_f16_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
        (void)cudaFree(r.device_ptr);
    }
    g_q8_f32_ranges.clear();
    g_q8_f32_by_offset.clear();
    g_q8_f32_bytes = 0;
    if (g_model_device_owned && g_model_device_base) {
        (void)cudaFree((void *)g_model_device_base);
        g_model_device_owned = 0;
    }
    if (g_model_registered && g_model_host_base) {
        (void)cudaHostUnregister((void *)g_model_host_base);
        g_model_registered = 0;
    }
    g_model_host_base = model_map;
    g_model_device_base = (const char *)model_map;
    g_model_registered_size = model_size;
    g_model_range_mapping_supported = 1;
    g_model_hmm_direct = 0;
    g_model_cache_full = 0;
    if (g_model_fd >= 0 && g_model_fd_host_base == NULL) {
        g_model_fd_host_base = model_map;
    }

    const char *copy_env = ((const char *)0) /* DS4_CUDA_COPY_MODEL: 路径开关已删(2026-08-22 隐形炸弹清理) */;
    if (copy_env && copy_env[0]) {
        void *dev = NULL;
        const double t0 = clock() / (double)CLOCKS_PER_SEC;
        cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
        if (err == cudaSuccess) {
            fprintf(stderr, "ds4: CUDA copying %.2f GiB model to device memory\n",
                    (double)model_size / 1073741824.0);
            err = cudaMemcpy(dev, model_map, (size_t)model_size, cudaMemcpyHostToDevice);
            if (err == cudaSuccess) {
                g_model_device_base = (const char *)dev;
                g_model_device_owned = 1;
                const double t1 = clock() / (double)CLOCKS_PER_SEC;
                fprintf(stderr, "ds4: CUDA model copy complete in %.3fs\n", t1 - t0);
                return 1;
            }
            fprintf(stderr, "ds4: CUDA model copy failed: %s\n", cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
        } else {
            fprintf(stderr, "ds4: CUDA model allocation skipped: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    }

    /* GB10 / driver 580.142 reports cudaDevAttrHostRegisterReadOnlySupported = 0,
     * so requesting cudaHostRegisterReadOnly here fails with cudaErrorNotSupported
     * and the entire model-resident fast path falls back to per-deref H2D streaming.
     * Plain `Mapped` works on Spark. */
    cudaError_t err = cudaHostRegister((void *)model_map, (size_t)model_size,
                                       cudaHostRegisterMapped);
    if (err == cudaSuccess) {
        void *dev = NULL;
        err = cudaHostGetDevicePointer(&dev, (void *)model_map, 0);
        if (err == cudaSuccess && dev) {
            g_model_device_base = (const char *)dev;
            g_model_registered = 1;
            fprintf(stderr, "ds4: CUDA registered %.2f GiB model mapping for device access\n",
                    (double)model_size / 1073741824.0);
        } else {
            fprintf(stderr, "ds4: CUDA host registration pointer lookup failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    } else {
        fprintf(stderr, "ds4: CUDA host registration skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
    }
    return 1;
}

/* Metal-only residency hint (see ds4_gpu.h); CUDA uses an HBM-cache model and
 * ignores it. No-op so the shared core links against either backend. */
void ds4_gpu_set_model_map_nonresident_hint(int on) { (void)on; }

/* Dynamic resident/offload route (ds4_gpu.h) is a Metal unified-memory concept;
 * CUDA manages residency via its HBM weight cache. Accept the host verdict as a
 * no-op and report no working-set ceiling so the AUTO path falls back to the
 * explicit DS4_MEM_BUDGET_MB (or resident) without a spurious offload. */
void ds4_gpu_set_expert_offload(int enabled) { (void)enabled; }
uint64_t ds4_gpu_recommended_max_working_set_bytes(void) { return 0; }

/* ---- 共享核心无条件调用、但只有 Metal 实现过的接口 ----
 * ds4.o / ds4_distributed.o 对两个后端只编译一份, 所以 CUDA 必须给出符号
 * (ds4_gpu_expert_remote_fetch_kick 的 CPU 版 no-op 见 ds4.c 顶部, 同一套路)。
 * 下面没有一个是占位兜底: 要么是真实现, 要么是"CUDA 确实没有这个能力"的语义
 * 正确回答, 且调用点都有对应的处理分支 —— 详见每个函数的注释。 */

/* 内存看门狗的 GPU 用量读数。CUDA 没有 Metal 那种 per-device allocated 计数,
 * 改用 driver 的 total-free: 它含其他进程和 context 开销, 对看门狗真正关心的
 * "这张卡现在还剩多少" 反而比纯自家分配量更准。 */
uint64_t ds4_gpu_current_allocated_bytes(void) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    return (uint64_t)(total_b - free_b);
}

/* go1b 残差 MoE 会把 router_selected 原地重映射成紧凑 slot, Metal 因此要留一份
 * 重映射前的快照给 corr 侧车按原始 expert id 取 C[e]/beta[e]。CUDA 没实现 go1b
 * 残差(见 routed_moe 包装里的 (void)residual), 从不重映射, 所以没有快照可留 ——
 * 返回 NULL, 调用点(ds4.c 的 `if (!corr_sel) corr_sel = g->router_selected`)
 * 退回活的 selected, 而那正是 CUDA 路径下它此刻的正确值。 */
ds4_gpu_tensor *ds4_gpu_corr_saved_selected(void) { return NULL; }

/* TP(tensor-parallel) 的 shared-FFN down 行切片: 为"两台 16G Mac 拆一个单机装不
 * 下的模型"设计, 算完半个 out_dim 再跨机 all-reduce。CUDA 侧走单机整模型, 没有
 * 对端可 all-reduce, 也不需要拆。返回 0 => 调用点 ok=false, 不进 TP 分支。 */
int ds4_gpu_matmul_q8_0_rowslice_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim_full, uint64_t in_dim_slice,
        uint64_t out_dim, const ds4_gpu_tensor *x) {
    (void)out; (void)model_map; (void)model_size; (void)weight_offset;
    (void)in_dim_full; (void)in_dim_slice; (void)out_dim; (void)x;
    return 0;
}

/* 裁剪专家模型(expert_shrunken)的 full-256 -> 紧凑 slot 路由翻译。CUDA 的 routed
 * MoE kernel 没有 keep-map 支持。返回 0 让 model_open 在上传 LUT 时 ds4_die 退出:
 * 这是要的行为 —— 与其拿未翻译的 id 去索引裁剪后的专家张量、静默算出一堆数字汤,
 * 不如在加载期就拒绝。全量模型(无 keep-map)两个函数都不会被调到。 */
int ds4_gpu_set_expert_keep_lut(const int16_t *lut, uint32_t n_layer) {
    (void)lut; (void)n_layer;
    return 0;
}
int ds4_gpu_translate_expert_ids(
        ds4_gpu_tensor *selected, uint32_t layer, uint32_t n_expert_used,
        uint32_t n_tokens, uint32_t n_total_expert) {
    (void)selected; (void)layer; (void)n_expert_used;
    (void)n_tokens; (void)n_total_expert;
    return 0;
}

/* 双机 worker 在 coordinator 刚连上、链路还安静时反向拨通 efetch 连接。单机
 * CUDA 没有远端专家字节源, no-op。 */
extern "C" void ds4_gpu_expert_remote_fetch_kick(void) {}

int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!ds4_gpu_set_model_map(model_map, model_size)) return 0;
    if (0 &&
        !cuda_model_copy_chunked(model_map, model_size, map_offset, map_size)) {
        (void)cuda_model_prefetch_range(model_map, model_size, map_offset, map_size);
    }
    return 1;
}

int ds4_gpu_set_model_map_spans(
        const void *model_map,
        uint64_t model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        uint32_t count,
        uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!model_map || model_size == 0 || !offsets || !sizes || count == 0) return 0;
    for (uint32_t i = 0; i < count; i++) {
        if (offsets[i] > model_size ||
            sizes[i] == 0 ||
            sizes[i] > model_size - offsets[i]) {
            return 0;
        }
    }
    if (!ds4_gpu_set_model_map(model_map, model_size)) return 0;

    if (0) {
        if (count > 1) {
            for (uint32_t i = 0; i < count; i++) {
                (void)cuda_model_prefetch_range(model_map, model_size, offsets[i], sizes[i]);
            }
            return 1;
        }
        for (uint32_t i = 0; i < count; i++) {
            if (!cuda_model_copy_chunked(model_map, model_size, offsets[i], sizes[i])) {
                (void)cuda_model_prefetch_range(model_map, model_size, offsets[i], sizes[i]);
            }
        }
    }
    return 1;
}

int ds4_gpu_set_model_map_spans_split(
        const void *model_map,
        uint64_t model_size,
        const uint64_t *offsets,
        const uint64_t *sizes,
        const bool *resident_flags,
        uint32_t count,
        uint64_t max_tensor_bytes) {
    /* CUDA does not use Metal's residency-set hint: cold routed-expert reads
     * already fall back to the UVA-mapped pointer (see accelerator_cache_model
     * _tensor_spans in ds4.c, which skips "_exps." at the HBM cache stage). The
     * resident flags are therefore advisory only here; map every span as usual. */
    (void)resident_flags;
    return ds4_gpu_set_model_map_spans(model_map, model_size, offsets, sizes,
                                       count, max_tensor_bytes);
}

/* Cross-layer router prediction prefetch is a Metal-side optimization (NVMe
 * read-ahead for SSD-streamed experts); the CUDA backend accepts and ignores
 * the registration so shared engine code links unchanged. */
int ds4_gpu_register_layer_router(const void *model_map, uint32_t layer,
                                             uint64_t gate_inp_offset, int gate_inp_is_f32,
                                             uint64_t probs_bias_offset,
                                             uint64_t gate_exps_offset, uint64_t up_exps_offset,
                                             uint64_t down_exps_offset, uint64_t gate_expert_bytes,
                                             uint64_t down_expert_bytes, uint32_t n_embd,
                                             uint32_t n_expert, uint64_t hash_table_offset,
                                             uint32_t hash_k, uint32_t hash_rows) {
    (void)model_map; (void)layer; (void)gate_inp_offset; (void)gate_inp_is_f32;
    (void)probs_bias_offset;
    (void)gate_exps_offset; (void)up_exps_offset; (void)down_exps_offset;
    (void)gate_expert_bytes; (void)down_expert_bytes; (void)n_embd; (void)n_expert;
    (void)hash_table_offset; (void)hash_k; (void)hash_rows;
    return 1;
}

int ds4_gpu_set_model_fd(int fd) {
    g_model_fd = fd;
    g_model_fd_host_base = g_model_host_base;
    g_model_file_size = 0;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            g_model_file_size = (uint64_t)st.st_size;
            if (st.st_blksize > 1) g_model_direct_align = (uint64_t)st.st_blksize;
        }
#if defined(__linux__) && defined(O_DIRECT)
        if (1) {
            char proc_path[64];
            snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
            int direct_fd = open(proc_path, O_RDONLY | O_DIRECT);
            if (direct_fd >= 0) {
                g_model_direct_fd = direct_fd;
                if (g_model_direct_align < 512) g_model_direct_align = 512;
                if (((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */) {
                    fprintf(stderr, "ds4: CUDA model direct I/O enabled (align=%llu)\n",
                            (unsigned long long)g_model_direct_align);
                }
            } else if (((const char *)0) /* DS4_CUDA_WEIGHT_CACHE_VERBOSE: 诊断开关已删(2026-08-22) */) {
                fprintf(stderr, "ds4: CUDA model direct I/O unavailable: %s\n", strerror(errno));
            }
        }
#endif
    }
    return 1;
}

int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label) {
#ifndef DS4_CUDA_SPARK_HBM_CACHE
    (void)model_map;
    (void)model_size;
    (void)offset;
    (void)bytes;
    (void)label;
    return 1;
#else
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    /* Startup walk: force-populate the device-resident HBM cache so hot
     * tensors hit cudaMalloc copies rather than the UVA-mapped fallback.
     * Skip silently if over budget or opted out — the mapped pointer still
     * works for any tensor we don't pre-cache. */
    if (0) return 1;
    if (g_model_device_owned) return 1;
    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes >= limit || bytes > limit - g_model_range_bytes) return 1;
    const char *what = label ? label : "model_tensor";
    /* Skip if this span is already populated. */
    auto exact = g_model_range_by_offset.find(offset);
    if (exact != g_model_range_by_offset.end()) {
        const cuda_model_range &r = g_model_ranges[exact->second];
        if (r.host_base == model_map && bytes <= r.bytes && !r.host_registered) return 1;
    }
    if (cuda_model_range_populate_device_copy(model_map, offset, bytes, what) == NULL) return 0;
    /* ★拷进设备后立刻丢掉这段的 page cache★
     * 统一内存机器上 mmap 的 page cache 与设备副本是同一块物理内存的两份占用:
     * 89.77GiB 模型 + 89.77GiB 副本 = 179GiB 远超 121GiB, 实测 free 掉到 7GiB 后
     * 系统忙于回收, 加载阶段直接卡死。这段字节此后由设备副本供给(offset→device ptr
     * 已登记), mmap 侧不再需要驻留 —— MADV_DONTNEED 对文件映射只丢干净页, 不损数据。 */
    if (1) {
        const uintptr_t pg = (uintptr_t)sysconf(_SC_PAGESIZE);
        uintptr_t a = (uintptr_t)model_map + offset;
        uintptr_t b = a + bytes;
        a = (a + pg - 1) & ~(pg - 1);      /* 只丢整页, 不碰边界的半页 */
        b = b & ~(pg - 1);
        if (b > a) (void)madvise((void *)a, (size_t)(b - a), MADV_DONTNEED);
    }
    return 1;
#endif
}

int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, uint64_t in_dim, uint64_t out_dim, const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    static int optional_q8_preload_disabled = 0;
    if (optional_q8_preload_disabled) return 1;
    const char *cache_label = label ? label : "q8_0";
    if (0 &&
        cuda_q8_f32_cache_allowed(cache_label, in_dim, out_dim)) {
        if (cuda_q8_f32_ptr(model_map, offset, bytes, in_dim, out_dim, cache_label)) return 1;
        optional_q8_preload_disabled = 1;
        return 1;
    }
    if (!cuda_q8_f16_preload_allowed(cache_label, in_dim, out_dim)) return 1;
    if (cuda_q8_f16_ptr(model_map, offset, bytes, in_dim, out_dim, cache_label)) return 1;
    optional_q8_preload_disabled = 1;
    return 1;
}

void ds4_gpu_print_memory_report(const char *label) {
    size_t free_b = 0, total_b = 0;
    (void)cudaMemGetInfo(&free_b, &total_b);
    fprintf(stderr, "ds4: CUDA memory report %s: free %.2f MiB total %.2f MiB\n",
            label ? label : "", (double)free_b / 1048576.0, (double)total_b / 1048576.0);
}

void ds4_gpu_set_quality(bool quality) {
    g_quality_mode = quality ? 1 : 0;
    if (g_cublas_ready) {
        const cublasMath_t math_mode =
            (g_quality_mode || 0)
                ? CUBLAS_DEFAULT_MATH
                : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);
    }
}

/* DSpark 抓取(2026-08-18): target 层输出 HC 4 流均值 → main_hidden 的 slot 段。
 * 官方语义: h.mean(dim=2)(inference/model.py)。 */
__global__ static void hc_mean_slot_kernel(
        float *dst, const float *hc, uint32_t n_embd, uint32_t n_hc,
        uint32_t slot, uint32_t n_tokens) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (i >= n_embd || t >= n_tokens) return;
    const float *h = hc + ((uint64_t)t * n_hc) * n_embd;
    float s = 0.0f;
    for (uint32_t k = 0; k < n_hc; k++) s += h[(uint64_t)k * n_embd + i];
    dst[((uint64_t)t * 3u + slot) * n_embd + i] = s / (float)n_hc;
}

/* token id 从 device 读的变体: 图重放时取参数槽最新值(流水线前提) */
__global__ static void embed_token_hc_dev_kernel(float *out, const unsigned short *w, const int32_t *tok_dev, uint32_t n_vocab, uint32_t n_embd, uint32_t n_hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_embd * n_hc;
    if (i >= n) return;
    int32_t ti = *tok_dev;
    uint32_t token = ti < 0 ? 0u : (uint32_t)ti;
    if (n_vocab && token >= n_vocab) token = 0;
    uint32_t e = i % n_embd;
    out[i] = __half2float(reinterpret_cast<const __half *>(w)[(uint64_t)token * n_embd + e]);
}

__global__ static void embed_token_hc_kernel(float *out, const unsigned short *w, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_embd * n_hc;
    if (i >= n) return;
    uint32_t e = i % n_embd;
    out[i] = __half2float(reinterpret_cast<const __half *>(w)[(uint64_t)token * n_embd + e]);
}

__global__ static void embed_tokens_hc_kernel(
        float *out,
        const int32_t *tokens,
        const __half *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0;
    out[gid] = __half2float(w[(uint64_t)tok * n_embd + d]);
}

/* go1b correction kernels (mirror metal/moe.metal kernel_dsv4_corr_*). */
__global__ static void corr_router_bias_kernel(
        float *logits, const float *delta, uint32_t n_expert, uint64_t total) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    logits[gid] += delta[(uint32_t)(gid % n_expert)];
}

__global__ static void corr_apply_kernel(
        float *out, const float *x, const float *U, const float *V,
        const float *C, const float *b, const float *beta, const int *selected,
        uint32_t d_model, uint32_t d_l, uint32_t n_expert, uint32_t n_sel) {
    extern __shared__ float vx[];   // [d_l] = V @ x[tok]
    uint32_t tok = blockIdx.x;
    const float *xt = x + (uint64_t)tok * d_model;
    const int *sel = selected + (uint64_t)tok * n_sel;
    for (uint32_t i = threadIdx.x; i < d_l; i += blockDim.x) {
        const float *Vrow = V + (uint64_t)i * d_model;
        float acc = 0.0f;
        for (uint32_t j = 0; j < d_model; j++) acc += Vrow[j] * xt[j];
        vx[i] = acc;
    }
    __syncthreads();
    float beta_sum = 0.0f; uint32_t n_valid = 0;
    for (uint32_t s = 0; s < n_sel; s++) {
        int e = sel[s];
        if (e >= 0 && (uint32_t)e < n_expert) { beta_sum += beta[e]; n_valid++; }
    }
    float *outt = out + (uint64_t)tok * d_model;
    for (uint32_t d = threadIdx.x; d < d_model; d += blockDim.x) {
        const float *Urow = U + (uint64_t)d * d_l;
        float acc = 0.0f;
        for (uint32_t s = 0; s < n_sel; s++) {
            int e = sel[s];
            if (e < 0 || (uint32_t)e >= n_expert) continue;
            const float *Crow = C + (uint64_t)e * d_l;
            float p = 0.0f;
            for (uint32_t i = 0; i < d_l; i++) p += Urow[i] * Crow[i] * vx[i];
            acc += p;
        }
        acc += (float)n_valid * b[d] + beta_sum;
        outt[d] += acc;
    }
}

__global__ static void matmul_f16_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += __half2float(wr[i]) * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void matmul_f16_serial_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok || threadIdx.x != 0) return;

    float sum = 0.0f;
    const __half *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = 0; i < in_dim; i++) {
        sum += __half2float(wr[i]) * xr[i];
    }
    out[tok * out_dim + row] = sum;
}

static float *g_f16sk_partial = NULL;  /* split-K f16 partial: [64][64] 上限 16KB */

/* split-K f16(2026-08-17 第六夜): 瘦高矩阵(16384→4 只发 1 个 block / 16384→24 发 3 个,
 * 45 个 SM 围观)按 K 维切 S 段并行, partial 后按 s 固定序 reduce(确定性)。
 * decode(n_tok==1) 且 out≤64 且 in≥4096 时启用。 */
__global__ static void matmul_f16_splitk_kernel(
        float *partial,               /* [n_tok][S][out_dim] */
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t chunk,               /* 每段元素数, 8 的倍数 */
        uint32_t n_s,
        uint32_t n_tok) {
    /* splitk3(施工日2): grid-stride cell 循环 — (out,S)≤4096 超短命块(每块~2KB即死)
     * 改常驻块循环(pair3 同配方); 单 cell 计算序不变=逐位一致。 */
    /* 批扩展(2026-08-21 verify 数值对齐): cell 空间加 token 维。每个 (tok,row,s) cell 的
     * 内部计算序与 n_tok==1 完全一致 ⇒ verify 批与 decode 逐位相同, 投机才可能无损。 */
    for (uint32_t cell = blockIdx.x; cell < out_dim * n_s * n_tok; cell += gridDim.x) {
    const uint32_t row = cell % (uint32_t)out_dim;
    const uint32_t s = (cell / (uint32_t)out_dim) % n_s;
    const uint32_t tok = cell / ((uint32_t)out_dim * n_s);
    const float *xt = x + (uint64_t)tok * in_dim;
    const uint64_t pidx = ((uint64_t)tok * n_s + s) * out_dim + row;
    const uint64_t base = (uint64_t)s * chunk;
    if (base >= in_dim) { if (threadIdx.x == 0) partial[pidx] = 0.0f; continue; }
    uint64_t end = base + chunk; if (end > in_dim) end = in_dim;
    float sum = 0.0f;
    /* uint4 宽读(2026-08-20 ②③刀): 8 half/load, 访存队列×4(LPDDR 要深队列);
     * chunk 为 8 倍数 ⇒ 段界 8 对齐; 行字节 in_dim*2, in 为 8 倍数时 16B 对齐。 */
    if (((in_dim & 7u) == 0u)) {
        const uint4 *wr4 = (const uint4 *)(w + (uint64_t)row * in_dim);
        const uint64_t g0 = base >> 3, g1 = end >> 3;
        for (uint64_t i = g0 + threadIdx.x; i < g1; i += blockDim.x) {
            const uint4 v = wr4[i];
            const __half2 *h = (const __half2 *)&v;
            const float4 xa = *(const float4 *)(xt + 8u * i);
            const float4 xb = *(const float4 *)(xt + 8u * i + 4u);
            sum += __low2float(h[0]) * xa.x + __high2float(h[0]) * xa.y
                 + __low2float(h[1]) * xa.z + __high2float(h[1]) * xa.w
                 + __low2float(h[2]) * xb.x + __high2float(h[2]) * xb.y
                 + __low2float(h[3]) * xb.z + __high2float(h[3]) * xb.w;
        }
    } else {
        const __half2 *wr = (const __half2 *)(w + (uint64_t)row * in_dim);
        const uint64_t h0 = base >> 1, h1 = end >> 1;
        for (uint64_t i = h0 + threadIdx.x; i < h1; i += blockDim.x) {
            const __half2 h = wr[i];
            sum += __low2float(h) * xt[2u * i] + __high2float(h) * xt[2u * i + 1u];
        }
    }
    /* 块内确定性归约: warp shuffle 树 + shared 固定序 */
    __shared__ float wsum[8];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if ((threadIdx.x & 31u) == 0) wsum[threadIdx.x >> 5u] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        #pragma unroll
        for (int wgi = 0; wgi < 8; wgi++) t += wsum[wgi];
        partial[pidx] = t;
    }
    __syncthreads();   /* splitk3: 下轮复用 wsum 前全员读完 */
    }
}

__global__ static void matmul_f16_splitk_reduce_kernel(
        float *out, const float *partial, uint32_t out_dim, uint32_t S, uint32_t n_tok) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_dim * n_tok) return;
    const uint32_t row = idx % out_dim;
    const uint32_t tok = idx / out_dim;
    float t = 0.0f;   /* s 固定序求和: 与 n_tok==1 完全一致 */
    for (uint32_t s = 0; s < S; s++) t += partial[((uint64_t)tok * S + s) * out_dim + row];
    out[idx] = t;
}

__global__ static void matmul_f16_ordered_chunks_kernel(
        float *out,
        const __half *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    /* 重写(同 pair 版): 原"每线程连续 chunk"访存不合并且 32 线程/block 一行。
     * 现 8 行/block × lane 交错 __half2 合并读。 */
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const __half2 *wr = (const __half2 *)(w + row * in_dim);
    const float *xr = x + tok * in_dim;
    const uint64_t n2 = in_dim >> 1;
    float sum = 0.0f;
    for (uint64_t i = lane; i < n2; i += 32u) {
        const __half2 h = wr[i];
        sum += __low2float(h) * xr[2u * i] + __high2float(h) * xr[2u * i + 1u];
    }
    if ((in_dim & 1u) && lane == 0)
        sum += __half2float(w[row * in_dim + in_dim - 1u]) * xr[in_dim - 1u];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) out[tok * out_dim + row] = sum;
}

/* pair 一行一块版(2026-08-17): 原 8 行/块×双矩阵交错, out=256/512 只发 32/64 块
 * (1/6 GPU 干活)。现 grid=2*out, 前半 w0 后半 w1, 256 线程整行, 块内固定序归约。 */
__global__ static void matmul_f16_pair_rowblock_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t pair2) {
    /* 刀③二进(2026-08-20 夜): 8行×32lane 版败因=每行在飞载荷太浅(2.90→3.11 档案);
     * 现 pair2=2行/块×128线程(uint4 宽读保留), 行内并行度仅减半、行数×2=深流水。
     * pair2=0(DS4_F16_PAIR2=0) 回一行一块原版。2*out 恒偶 ⇒ 双行永同界, 无发散 sync 险。 */
    const uint32_t nrow_th = pair2 ? 128u : 256u;
    const uint32_t tid = threadIdx.x & (nrow_th - 1u);
    /* pair3(施工日2): grid-stride 行循环 — 2048短命块(每块2迭代即死, 7 waves 全ramp,
     * 151GB/s)改常驻块循环吃行(dense 成功配方); 读宽/归约树不变。 */
    for (uint32_t r = pair2 ? (blockIdx.x * 2u + (threadIdx.x >> 7u)) : blockIdx.x;
         r < 2u * out_dim;
         r += pair2 ? (gridDim.x * 2u) : gridDim.x) {
    const uint32_t which = (r >= out_dim) ? 1u : 0u;
    const uint64_t row = which ? (uint64_t)(r - out_dim) : (uint64_t)r;
    float sum = 0.0f;
    if ((in_dim & 7u) == 0u) {   /* uint4 宽读: 8 half/load(2026-08-20 ②③刀) */
        const uint4 *wr4 = (const uint4 *)((which ? w1 : w0) + row * in_dim);
        const uint64_t n8 = in_dim >> 3;
        for (uint64_t i = tid; i < n8; i += nrow_th) {
            const uint4 v = wr4[i];
            const __half2 *h = (const __half2 *)&v;
            const float4 xa = *(const float4 *)(x + 8u * i);
            const float4 xb = *(const float4 *)(x + 8u * i + 4u);
            sum += __low2float(h[0]) * xa.x + __high2float(h[0]) * xa.y
                 + __low2float(h[1]) * xa.z + __high2float(h[1]) * xa.w
                 + __low2float(h[2]) * xb.x + __high2float(h[2]) * xb.y
                 + __low2float(h[3]) * xb.z + __high2float(h[3]) * xb.w;
        }
    } else {
        const __half2 *wr = (const __half2 *)((which ? w1 : w0) + row * in_dim);
        const uint64_t n2 = in_dim >> 1;
        for (uint64_t i = tid; i < n2; i += nrow_th) {
            const __half2 h = wr[i];
            sum += __low2float(h) * x[2u * i] + __high2float(h) * x[2u * i + 1u];
        }
        if ((in_dim & 1u) && tid == 0)
            sum += __half2float((which ? w1 : w0)[row * in_dim + in_dim - 1u]) * x[in_dim - 1u];
    }
    __shared__ float wsum[8];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if ((threadIdx.x & 31u) == 0) wsum[threadIdx.x >> 5u] = sum;
    __syncthreads();
    if (tid == 0) {
        const uint32_t wbase = pair2 ? ((threadIdx.x >> 7u) * 4u) : 0u;
        const uint32_t wn = pair2 ? 4u : 8u;
        float t = 0.0f;
        for (uint32_t wgi = 0; wgi < wn; wgi++) t += wsum[wbase + wgi];
        (which ? out1 : out0)[row] = t;
    }
    __syncthreads();   /* pair3: 下轮复用 wsum 前全员必须读完 */
    }
}

__global__ static void matmul_f16_pair_ordered_chunks_kernel(
        float *out0,
        float *out1,
        const __half *w0,
        const __half *w1,
        const float *x,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim) {
    /* 重写(2026-08-17 调优): 原版每线程读一段连续 chunk ⇒ warp 内 32 线程地址相距
     * chunk 远, 访存完全不合并, 且 32 线程/block 一行 —— 实测仅 ~102GB/s(46%)。
     * 现在: 8 行/block × 32 lane 交错读(__half2 双宽), 合并访存; 行内求和顺序仍是
     * 运行间确定的(lane 交错 + 固定 shuffle 树)。 */
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    if (row >= out0_dim && row >= out1_dim) return;
    const __half2 *wr0 = (const __half2 *)(w0 + (row < out0_dim ? row * in_dim : 0));
    const __half2 *wr1 = (const __half2 *)(w1 + (row < out1_dim ? row * in_dim : 0));
    const uint64_t n2 = in_dim >> 1;
    float sum0 = 0.0f, sum1 = 0.0f;
    for (uint64_t i = lane; i < n2; i += 32u) {
        const float x0 = x[2u * i], x1 = x[2u * i + 1u];
        if (row < out0_dim) {
            const __half2 h = wr0[i];
            sum0 += __low2float(h) * x0 + __high2float(h) * x1;
        }
        if (row < out1_dim) {
            const __half2 h = wr1[i];
            sum1 += __low2float(h) * x0 + __high2float(h) * x1;
        }
    }
    if (in_dim & 1u) {   /* 奇数尾 */
        const uint64_t i = in_dim - 1u;
        if (lane == 0) {
            if (row < out0_dim) sum0 += __half2float(w0[row * in_dim + i]) * x[i];
            if (row < out1_dim) sum1 += __half2float(w1[row * in_dim + i]) * x[i];
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, off);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, off);
    }
    if (lane == 0) {
        if (row < out0_dim) out0[row] = sum0;
        if (row < out1_dim) out1[row] = sum1;
    }
}

__global__ static void matmul_f32_kernel(
        float *out,
        const float *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    float sum = 0.0f;
    const float *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        sum += wr[i] * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void repeat_hc_kernel(float *out, const float *row, uint32_t n_embd, uint32_t n_hc) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_hc;
    if (i >= n) return;
    out[i] = row[i % n_embd];
}

__global__ static void f32_to_f16_kernel(__half *out, const float *x, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
}

__device__ static float warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__device__ static float warp_max_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

__device__ static float dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_aligned(const int8_t *p) {
    return *(const int32_t *)p;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_unaligned(const int8_t *p) {
    const uint8_t *u = (const uint8_t *)p;
    return (int32_t)((uint32_t)u[0] |
                     ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) |
                     ((uint32_t)u[3] << 24));
}

__device__ __forceinline__ static int32_t dot_i8x32_dp4a(const int8_t *a, const int8_t *b) {
    int32_t dot = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        dot = __dp4a(load_i8x4_i32_unaligned(a + i), load_i8x4_i32_aligned(b + i), dot);
    }
    return dot;
}

__device__ __forceinline__ static int32_t dot_i8_block(const int8_t *a, const int8_t *b, uint64_t n, int use_dp4a) {
    if (use_dp4a && n == 32u) return dot_i8x32_dp4a(a, b);
    int32_t dot = 0;
    for (uint64_t i = 0; i < n; i++) dot += (int32_t)a[i] * (int32_t)b[i];
    return dot;
}

__global__ static DS4_CUDA_UNUSED void matmul_q8_0_kernel(
        float *out,
        const unsigned char *w,
        const float *x,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const uint64_t blocks = (in_dim + 31) / 32;
    const unsigned char *wr = w + row * blocks * 34;
    const float *xr = x + tok * in_dim;
    float acc = 0.0f;

    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) amax = fmaxf(amax, fabsf(xr[i0 + i]));
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        int dot = 0;
        for (uint64_t i = 0; i < bn; i++) {
            int q = (int)lrintf(xr[i0 + i] * id);
            q = q > 127 ? 127 : (q < -128 ? -128 : q);
            dot += (int)qs[i] * q;
        }
        acc += __half2float(*scale_h) * d * (float)dot;
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__global__ static void quantize_q8_0_f32_kernel(
        int8_t *xq,
        float *xscale,
        const float *x,
        uint64_t in_dim,
        uint64_t blocks) {
    uint64_t b = blockIdx.x;
    uint64_t tok = blockIdx.y;
    if (b >= blocks) return;
    uint64_t i0 = b * 32;
    uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
    const float *xr = x + tok * in_dim + i0;

    float a = 0.0f;
    if (threadIdx.x < bn) a = fabsf(xr[threadIdx.x]);
    __shared__ float vals[32];
    vals[threadIdx.x] = a;
    __syncthreads();
    for (uint32_t stride = 16; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) vals[threadIdx.x] = fmaxf(vals[threadIdx.x], vals[threadIdx.x + stride]);
        __syncthreads();
    }
    const float d = vals[0] / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0) xscale[tok * blocks + b] = d;
    int8_t *dst = xq + (tok * blocks + b) * 32;
    if (threadIdx.x < bn) {
        int v = (int)lrintf(xr[threadIdx.x] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[threadIdx.x] = (int8_t)v;
    } else {
        dst[threadIdx.x] = 0;
    }
}

__global__ static void matmul_q8_0_preq_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x;
    uint64_t tok = (uint64_t)blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[tok * out_dim + row] = partial[0];
}

__device__ __forceinline__ static float q8r_row_dot(
        const __half *scales, const int8_t *qs, uint64_t row, uint64_t blocks,
        const int8_t *xq, const float *xscale, uint32_t lane) {
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const int4 xv0 = *(const int4 *)(xq + b * 32u);
        const int4 xv1 = *(const int4 *)(xq + b * 32u + 16u);
        const int4 w0 = *(const int4 *)(qs + (row * blocks + b) * 32u);
        const int4 w1 = *(const int4 *)(qs + (row * blocks + b) * 32u + 16u);
        int32_t d = 0;
        d = __dp4a(w0.x, xv0.x, d); d = __dp4a(w0.y, xv0.y, d);
        d = __dp4a(w0.z, xv0.z, d); d = __dp4a(w0.w, xv0.w, d);
        d = __dp4a(w1.x, xv1.x, d); d = __dp4a(w1.y, xv1.y, d);
        d = __dp4a(w1.z, xv1.z, d); d = __dp4a(w1.w, xv1.w, d);
        acc += __half2float(scales[row * blocks + b]) * xscale[b] * (float)d;
    }
    return acc;
}

__global__ static void matmul_q8r_pair_warp8_kernel(
        float *out0, float *out1,
        const __half *sc0, const int8_t *q0,
        const __half *sc1, const int8_t *q1,
        const int8_t *xq, const float *xscale,
        uint64_t out0_dim, uint64_t out1_dim, uint64_t blocks) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out0_dim && row >= out1_dim) return;
    float a0 = 0.0f, a1 = 0.0f;
    if (row < out0_dim) a0 = q8r_row_dot(sc0, q0, row, blocks, xq, xscale, lane);
    if (row < out1_dim) a1 = q8r_row_dot(sc1, q1, row, blocks, xq, xscale, lane);
    a0 = warp_sum_f32(a0);
    a1 = warp_sum_f32(a1);
    if (lane == 0) {
        if (row < out0_dim) out0[row] = a0;
        if (row < out1_dim) out1[row] = a1;
    }
}

__global__ static void matmul_q8r_hc_expand_kernel(
        float *out_hc, float *block_out, const float *block_add,
        const float *residual_hc, const float *split,
        const __half *scales, const int8_t *qs,
        const int8_t *xq, const float *xscale,
        uint64_t out_dim, uint32_t n_embd, uint32_t n_hc, uint64_t blocks, int has_add) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    float acc = q8r_row_dot(scales, qs, row, blocks, xq, xscale, lane);
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        float block_v = acc;
        if (has_add) block_v += block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}

__global__ static void grouped_q8r_a_kernel(
        float *low, const __half *scales, const int8_t *qs,
        const int8_t *xq, const float *xscale,
        uint64_t rank, uint32_t n_groups, uint32_t n_tokens, uint64_t blocks) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim || tok >= n_tokens) return;
    const uint64_t group = row / rank;
    const uint64_t xrow = tok * (uint64_t)n_groups + group;
    float acc = q8r_row_dot(scales, qs, row, blocks,
                            xq + xrow * blocks * 32u, xscale + xrow * blocks, lane);
    acc = warp_sum_f32(acc);
    if (lane == 0) low[tok * low_dim + row] = acc;
}

__device__ __forceinline__ static float q8r_seg_dot(
        const __half *scales, const int8_t *qs, uint64_t row, uint64_t blocks,
        const int8_t *xq, const float *xscale, uint64_t seg_beg, uint64_t seg_end, uint32_t lane) {
    float acc = 0.0f;
    for (uint64_t b = seg_beg + lane; b < seg_end; b += 32u) {
        const int4 xv0 = *(const int4 *)(xq + b * 32u);
        const int4 xv1 = *(const int4 *)(xq + b * 32u + 16u);
        const int4 w0 = *(const int4 *)(qs + (row * blocks + b) * 32u);
        const int4 w1 = *(const int4 *)(qs + (row * blocks + b) * 32u + 16u);
        int32_t d = 0;
        d = __dp4a(w0.x, xv0.x, d); d = __dp4a(w0.y, xv0.y, d);
        d = __dp4a(w0.z, xv0.z, d); d = __dp4a(w0.w, xv0.w, d);
        d = __dp4a(w1.x, xv1.x, d); d = __dp4a(w1.y, xv1.y, d);
        d = __dp4a(w1.z, xv1.z, d); d = __dp4a(w1.w, xv1.w, d);
        acc += __half2float(scales[row * blocks + b]) * xscale[b] * (float)d;
    }
    return acc;
}

/* splitK gemv: 8 warps = 2 行 × 4 段。行数少的 decode gemv 每行单 warp 时
 * scheduler 无候选可发(ncu: No Eligible 96%, 活跃 warp 9.7/调度器) —— 行内
 * 4 段并行把在飞 warp 数 ×4。段序固定 ⇒ 归约顺序恒定, 运行间确定。 */
__global__ static void matmul_q8r_sk_kernel(
        float *out, const __half *scales, const int8_t *qs,
        const int8_t *xq, const float *xscale, uint64_t out_dim, uint64_t blocks) {
    const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const uint32_t rsub = warp >> 2u, seg = warp & 3u;
    const uint64_t row = (uint64_t)blockIdx.x * 2u + rsub;
    __shared__ float part[2][4];
    float acc = 0.0f;
    if (row < out_dim) {
        const uint64_t sb = (blocks * seg) >> 2u;
        const uint64_t se = (blocks * (seg + 1u)) >> 2u;
        acc = q8r_seg_dot(scales, qs, row, blocks, xq, xscale, sb, se, lane);
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) part[rsub][seg] = acc;
    __syncthreads();
    if (warp == 0u && lane < 2u) {
        const uint64_t r2 = (uint64_t)blockIdx.x * 2u + lane;
        if (r2 < out_dim)
            out[r2] = part[lane][0] + part[lane][1] + part[lane][2] + part[lane][3];
    }
}

__global__ static void matmul_q8r_sk_pair_kernel(
        float *out0, float *out1,
        const __half *sc0, const int8_t *q0,
        const __half *sc1, const int8_t *q1,
        const int8_t *xq, const float *xscale,
        uint64_t out0_dim, uint64_t out1_dim, uint64_t blocks) {
    const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const uint32_t rsub = warp >> 2u, seg = warp & 3u;
    const uint64_t row = (uint64_t)blockIdx.x * 2u + rsub;
    __shared__ float part0[2][4], part1[2][4];
    float a0 = 0.0f, a1 = 0.0f;
    const uint64_t sb = (blocks * seg) >> 2u;
    const uint64_t se = (blocks * (seg + 1u)) >> 2u;
    if (row < out0_dim) a0 = q8r_seg_dot(sc0, q0, row, blocks, xq, xscale, sb, se, lane);
    if (row < out1_dim) a1 = q8r_seg_dot(sc1, q1, row, blocks, xq, xscale, sb, se, lane);
    a0 = warp_sum_f32(a0);
    a1 = warp_sum_f32(a1);
    if (lane == 0) { part0[rsub][seg] = a0; part1[rsub][seg] = a1; }
    __syncthreads();
    if (warp == 0u && lane < 2u) {
        const uint64_t r2 = (uint64_t)blockIdx.x * 2u + lane;
        if (r2 < out0_dim) out0[r2] = part0[lane][0] + part0[lane][1] + part0[lane][2] + part0[lane][3];
        if (r2 < out1_dim) out1[r2] = part1[lane][0] + part1[lane][1] + part1[lane][2] + part1[lane][3];
    }
}

__global__ static void matmul_q8r_sk_hc_expand_kernel(
        float *out_hc, float *block_out, const float *block_add,
        const float *residual_hc, const float *split,
        const __half *scales, const int8_t *qs,
        const int8_t *xq, const float *xscale,
        uint64_t out_dim, uint32_t n_embd, uint32_t n_hc, uint64_t blocks, int has_add) {
    const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const uint32_t rsub = warp >> 2u, seg = warp & 3u;
    const uint64_t row = (uint64_t)blockIdx.x * 2u + rsub;
    __shared__ float part[2][4];
    float acc = 0.0f;
    if (row < out_dim) {
        const uint64_t sb = (blocks * seg) >> 2u;
        const uint64_t se = (blocks * (seg + 1u)) >> 2u;
        acc = q8r_seg_dot(scales, qs, row, blocks, xq, xscale, sb, se, lane);
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) part[rsub][seg] = acc;
    __syncthreads();
    if (warp == 0u && lane < 2u) {
        const uint64_t r2 = (uint64_t)blockIdx.x * 2u + lane;
        if (r2 < out_dim) {
            const float a = part[lane][0] + part[lane][1] + part[lane][2] + part[lane][3];
            const uint32_t d = (uint32_t)r2;
            block_out[d] = a;
            float block_v = a;
            if (has_add) block_v += block_add[d];
            const float *post = split + n_hc;
            const float *comb = split + 2u * n_hc;
            for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
                float hc_acc = block_v * post[dst_hc];
                for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                    const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                    const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                    hc_acc += comb_v * res_v;
                }
                out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
            }
        }
    }
}

__global__ static void grouped_q8r_sk_a_kernel(
        float *low, const __half *scales, const int8_t *qs,
        const int8_t *xq, const float *xscale,
        uint64_t rank, uint32_t n_groups, uint64_t blocks) {
    const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const uint32_t rsub = warp >> 2u, seg = warp & 3u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t row = (uint64_t)blockIdx.x * 2u + rsub;
    __shared__ float part[2][4];
    float acc = 0.0f;
    if (row < low_dim) {
        const uint64_t group = row / rank;
        const uint64_t sb = (blocks * seg) >> 2u;
        const uint64_t se = (blocks * (seg + 1u)) >> 2u;
        acc = q8r_seg_dot(scales, qs, row, blocks,
                          xq + group * blocks * 32u, xscale + group * blocks, sb, se, lane);
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) part[rsub][seg] = acc;
    __syncthreads();
    if (warp == 0u && lane < 2u) {
        const uint64_t r2 = (uint64_t)blockIdx.x * 2u + lane;
        if (r2 < low_dim) low[r2] = part[lane][0] + part[lane][1] + part[lane][2] + part[lane][3];
    }
}

__global__ static void matmul_q8r_warp8_kernel(
        float *out, const __half *scales, const int8_t *qs,
        const int8_t *xq, const float *xscale,
        uint64_t out_dim, uint64_t blocks) {
    /* repacked q8 gemv: scale/qs 平面分离, 每块 2 次 int4(128bit) 对齐读代替
     * 8 次非对齐 4B 读; 双行互填延迟。dp4a 求和顺序与原 kernel 相同 ⇒ bit 级不变。 */
    const uint64_t row0 = (uint64_t)blockIdx.x * 16u + (uint64_t)(threadIdx.x >> 5u) * 2u;
    const uint64_t row1 = row0 + 1u;
    const uint32_t lane = threadIdx.x & 31u;
    if (row0 >= out_dim) return;
    const int has1 = row1 < out_dim;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const int4 xv0 = *(const int4 *)(xq + b * 32u);
        const int4 xv1 = *(const int4 *)(xq + b * 32u + 16u);
        const float xs = xscale[b];
        {
            const int4 w0 = *(const int4 *)(qs + (row0 * blocks + b) * 32u);
            const int4 w1 = *(const int4 *)(qs + (row0 * blocks + b) * 32u + 16u);
            int32_t d = 0;
            d = __dp4a(w0.x, xv0.x, d); d = __dp4a(w0.y, xv0.y, d);
            d = __dp4a(w0.z, xv0.z, d); d = __dp4a(w0.w, xv0.w, d);
            d = __dp4a(w1.x, xv1.x, d); d = __dp4a(w1.y, xv1.y, d);
            d = __dp4a(w1.z, xv1.z, d); d = __dp4a(w1.w, xv1.w, d);
            acc0 += __half2float(scales[row0 * blocks + b]) * xs * (float)d;
        }
        if (has1) {
            const int4 w0 = *(const int4 *)(qs + (row1 * blocks + b) * 32u);
            const int4 w1 = *(const int4 *)(qs + (row1 * blocks + b) * 32u + 16u);
            int32_t d = 0;
            d = __dp4a(w0.x, xv0.x, d); d = __dp4a(w0.y, xv0.y, d);
            d = __dp4a(w0.z, xv0.z, d); d = __dp4a(w0.w, xv0.w, d);
            d = __dp4a(w1.x, xv1.x, d); d = __dp4a(w1.y, xv1.y, d);
            d = __dp4a(w1.z, xv1.z, d); d = __dp4a(w1.w, xv1.w, d);
            acc1 += __half2float(scales[row1 * blocks + b]) * xs * (float)d;
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        out[row0] = acc0;
        if (has1) out[row1] = acc1;
    }
}

__global__ static void matmul_q8_0_preq_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks,
        int use_dp4a) {
    /* 每 warp 双行: 两条 dp4a 累加链交替发射互填 L1 延迟(单链版 ncu SM 利用仅
     * 9-17%), xq/xscale 读两行共享。每行的求和顺序与单行版完全相同 ⇒ bit 级不变。 */
    const uint64_t row0 = (uint64_t)blockIdx.x * 16u + (uint64_t)(threadIdx.x >> 5u) * 2u;
    const uint64_t row1 = row0 + 1u;
    const uint32_t lane = threadIdx.x & 31u;
    if (row0 >= out_dim) return;
    const unsigned char *wr0 = w + row0 * blocks * 34;
    const unsigned char *wr1 = (row1 < out_dim) ? w + row1 * blocks * 34 : NULL;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const int8_t *xqb = xq + b * 32;
        const float xs = xscale[b];
        const __half *s0 = (const __half *)(wr0 + b * 34);
        const int dot0 = dot_i8_block((const int8_t *)(wr0 + b * 34 + 2), xqb, bn, use_dp4a);
        acc0 += __half2float(*s0) * xs * (float)dot0;
        if (wr1) {
            const __half *s1 = (const __half *)(wr1 + b * 34);
            const int dot1 = dot_i8_block((const int8_t *)(wr1 + b * 34 + 2), xqb, bn, use_dp4a);
            acc1 += __half2float(*s1) * xs * (float)dot1;
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        out[row0] = acc0;
        if (wr1) out[row1] = acc1;
    }
}

__global__ static void matmul_q8_0_pair_preq_warp8_kernel(
        float *out0,
        float *out1,
        const unsigned char *w0,
        const unsigned char *w1,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        uint64_t blocks,
        int use_dp4a) {
    uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    uint32_t lane = threadIdx.x & 31u;
    if (row >= out0_dim && row >= out1_dim) return;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const unsigned char *wr0 = row < out0_dim ? w0 + row * blocks * 34 : NULL;
    const unsigned char *wr1 = row < out1_dim ? w1 + row * blocks * 34 : NULL;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        uint64_t i0 = b * 32;
        uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const int8_t *xqb = xq + b * 32;
        const float xs = xscale[b];
        if (wr0) {
            const __half *scale_h = (const __half *)(wr0 + b * 34);
            const int8_t *qs = (const int8_t *)(wr0 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc0 += __half2float(*scale_h) * xs * (float)dot;
        }
        if (wr1) {
            const __half *scale_h = (const __half *)(wr1 + b * 34);
            const int8_t *qs = (const int8_t *)(wr1 + b * 34 + 2);
            int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
            acc1 += __half2float(*scale_h) * xs * (float)dot;
        }
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        if (row < out0_dim) out0[row] = acc0;
        if (row < out1_dim) out1[row] = acc1;
    }
}

__global__ static void matmul_q8_0_hc_expand_preq_warp8_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_embd,
        uint32_t n_hc,
        uint64_t blocks,
        int has_add,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const unsigned char *wr = w + row * blocks * 34;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xq + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        float block_v = acc;
        if (has_add) block_v += block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}

__global__ static void matmul_q8_0_preq_batch_warp8_kernel(
        float *out,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t n_tok,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok >= n_tok) return;

    const unsigned char *wr = w + row * blocks * 34;
    const int8_t *xqr = xq + tok * blocks * 32;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = in_dim - i0 < 32 ? in_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[tok * out_dim + row] = acc;
}

__global__ static void dequant_q8_0_to_f16_kernel(
        __half *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    uint64_t row = gid / in_dim;
    uint64_t i = gid - row * in_dim;
    uint64_t b = i / 32;
    uint64_t j = i - b * 32;
    const unsigned char *blk = w + (row * blocks + b) * 34;
    const __half scale = *(const __half *)blk;
    const int8_t q = *(const int8_t *)(blk + 2 + j);
    out[gid] = __hmul(scale, __float2half((float)q));
}

__global__ static void dequant_q8_0_to_f32_kernel(
        float *out,
        const unsigned char *w,
        uint64_t in_dim,
        uint64_t out_dim,
        uint64_t blocks) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    uint64_t row = gid / in_dim;
    uint64_t i = gid - row * in_dim;
    uint64_t b = i / 32;
    uint64_t j = i - b * 32;
    const unsigned char *blk = w + (row * blocks + b) * 34;
    const float scale = __half2float(*(const __half *)blk);
    const int8_t q = *(const int8_t *)(blk + 2 + j);
    out[gid] = scale * (float)q;
}

__global__ static void grouped_q8_0_a_preq_warp8_kernel(
        float *low,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t blocks,
        int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim || tok >= n_tokens) return;

    const uint64_t group = row / rank;
    const uint64_t row_in_group = row - group * rank;
    const unsigned char *wr = w + (group * rank + row_in_group) * blocks * 34;
    const uint64_t xrow = tok * (uint64_t)n_groups + group;
    const int8_t *xqr = xq + xrow * blocks * 32;
    const float *xsr = xscale + xrow * blocks;
    float acc = 0.0f;

    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = group_dim - i0 < 32 ? group_dim - i0 : 32;
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xqr + b * 32;
        int dot = dot_i8_block(qs, xqb, bn, use_dp4a);
        acc += __half2float(*scale_h) * xsr[b] * (float)dot;
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) low[tok * low_dim + row] = acc;
}

/* 快路径 rms_norm(2026-08-17): 老版 1 块×256 线程+8 轮 __syncthreads 树规约, 实测
 * 17.4μs×2/层≈1.5ms/token 纯延迟。新版: float4 宽读 + warp shuffle 规约(1 轮 sync)
 * + 1024 线程。w==NULL 即 plain。求和树固定 ⇒ 运行间确定。 */
__global__ static void rms_norm_fast_kernel(
        float *out, const float *x, const float *w, uint32_t n, uint32_t rows, float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    const uint32_t n4 = n >> 2;
    const float4 *x4 = (const float4 *)xr;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = x4[i];
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (uint32_t i = (n4 << 2) + threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i]; sum += v * v;
    }
    __shared__ float wsum[32];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if ((threadIdx.x & 31u) == 0) wsum[threadIdx.x >> 5u] = sum;
    __syncthreads();
    if (threadIdx.x < 32u) {
        const uint32_t nw = (blockDim.x + 31u) >> 5u;
        float t = (threadIdx.x < nw) ? wsum[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffffu, t, off);
        if (threadIdx.x == 0) wsum[0] = t;
    }
    __syncthreads();
    const float scale = rsqrtf(wsum[0] / (float)n + eps);
    float4 *o4 = (float4 *)orow;
    if (w) {
        const float4 *w4 = (const float4 *)w;
        for (uint32_t i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 v = x4[i], g = w4[i];
            float4 r; r.x = v.x * scale * g.x; r.y = v.y * scale * g.y;
            r.z = v.z * scale * g.z; r.w = v.w * scale * g.w;
            o4[i] = r;
        }
        for (uint32_t i = (n4 << 2) + threadIdx.x; i < n; i += blockDim.x)
            orow[i] = xr[i] * scale * w[i];
    } else {
        for (uint32_t i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 v = x4[i];
            float4 r; r.x = v.x * scale; r.y = v.y * scale; r.z = v.z * scale; r.w = v.w * scale;
            o4[i] = r;
        }
        for (uint32_t i = (n4 << 2) + threadIdx.x; i < n; i += blockDim.x)
            orow[i] = xr[i] * scale;
    }
}

__global__ static void rms_norm_plain_kernel(float *out, const float *x, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale;
    }
}

__global__ static void rms_norm_weight_kernel(float *out, const float *x, const float *w, uint32_t n, uint32_t rows, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale * w[i];
    }
}

__global__ static void dsv4_qkv_rms_norm_rows_kernel(
        float *q_out,
        const float *q,
        const float *q_w,
        uint32_t q_n,
        float *kv_out,
        const float *kv,
        const float *kv_w,
        uint32_t kv_n,
        uint32_t rows,
        float eps) {
    const uint32_t row = blockIdx.x;
    const uint32_t which = blockIdx.y;
    if (row >= rows || which > 1u) return;
    const uint32_t n = which == 0u ? q_n : kv_n;
    const float *xr = (which == 0u ? q : kv) + (uint64_t)row * n;
    float *orow = (which == 0u ? q_out : kv_out) + (uint64_t)row * n;
    const float *w = which == 0u ? q_w : kv_w;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale * w[i];
    }
}

__global__ static void head_rms_norm_kernel(float *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    float *xr = x + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) xr[i] *= scale;
}

__device__ static float rope_yarn_ramp_dev(float low, float high, int i0);

__global__ static void head_rms_norm_rope_tail_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float eps) {
    uint32_t row = blockIdx.x;
    if (row >= n_tok * n_head) return;
    uint32_t t = row / n_head;
    float *xr = x + (uint64_t)row * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t i = threadIdx.x; i < n_nope; i += blockDim.x) {
        xr[i] *= scale;
    }

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }
    for (uint32_t pair = threadIdx.x; pair < n_rot / 2; pair += blockDim.x) {
        uint32_t i = pair * 2u;
        float theta_extrap = (float)(pos0 + t) * powf(freq_base, -((float)i) / (float)n_rot);
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        if (inverse) s = -s;
        float *tail = xr + n_nope;
        float x0 = tail[i] * scale;
        float x1 = tail[i + 1] * scale;
        tail[i] = x0 * c - x1 * s;
        tail[i + 1] = x0 * s + x1 * c;
    }
}

__device__ static float rope_yarn_ramp_dev(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__global__ static void rope_tail_kernel(
        float *x,
        uint32_t n_tok,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t n_rot,
        uint32_t pos0,
        uint32_t pos_stride,
        uint32_t n_ctx_orig,
        int inverse,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    if (gid >= pairs) return;
    uint32_t pair = gid % (n_rot / 2);
    uint32_t tmp = gid / (n_rot / 2);
    uint32_t h = tmp % n_head;
    uint32_t t = tmp / n_head;
    uint32_t n_nope = head_dim - n_rot;
    uint32_t i = pair * 2;

    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
        corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
        corr0 = fmaxf(0.0f, corr0);
        corr1 = fminf((float)(n_rot - 1), corr1);
    }

    float theta_extrap = (float)(pos0 + t * pos_stride) * powf(freq_base, -((float)i) / (float)n_rot);
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    float c = cosf(theta) * mscale;
    float s = sinf(theta) * mscale;
    if (inverse) s = -s;

    float *tail = x + ((uint64_t)t * n_head + h) * head_dim + n_nope;
    float x0 = tail[i];
    float x1 = tail[i + 1];
    tail[i] = x0 * c - x1 * s;
    tail[i + 1] = x0 * s + x1 * c;
}

__device__ static float dsv4_e4m3fn_value_dev(int i) {
    int exp = (i >> 3) & 15;
    int mant = i & 7;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}

__device__ static float dsv4_e4m3fn_dequant_dev(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (dsv4_e4m3fn_value_dev(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        float bd = fabsf(ax - dsv4_e4m3fn_value_dev(best));
        float nd = fabsf(ax - dsv4_e4m3fn_value_dev(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return sign * dsv4_e4m3fn_value_dev(best);
}

__device__ static float dsv4_e2m1fn_value_dev(int i) {
    switch (i & 7) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
    }
}

__device__ static float dsv4_e2m1fn_dequant_dev(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - dsv4_e2m1fn_value_dev(0));
    for (int i = 1; i < 8; i++) {
        float diff = fabsf(ax - dsv4_e2m1fn_value_dev(i));
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * dsv4_e2m1fn_value_dev(best);
}

__device__ static float model_scalar_dev(const void *base, uint64_t offset, uint32_t type, uint64_t idx) {
    const char *p = (const char *)base + offset;
    if (type == 1u) return __half2float(((const __half *)p)[idx]);
    return ((const float *)p)[idx];
}

__device__ static float rope_yarn_ramp_cpu_equiv_dev(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__device__ static DS4_CUDA_UNUSED void rope_tail_one_dev(float *x, uint32_t head_dim, uint32_t n_rot, uint32_t pos, uint32_t n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    uint32_t n_nope = head_dim - n_rot;
    float corr0 = 0.0f, corr1 = 0.0f;
    if (ext_factor != 0.0f) {
        float denom = 2.0f * logf(freq_base);
        corr0 = fmaxf(0.0f, floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom));
        corr1 = fminf((float)(n_rot - 1), ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom));
    }
    for (uint32_t i = 0; i < n_rot; i += 2) {
        float theta_extrap = (float)pos * powf(freq_base, -((float)i) / (float)n_rot);
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float mix = rope_yarn_ramp_cpu_equiv_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - mix) + theta_extrap * mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        float c = cosf(theta) * mscale;
        float s = sinf(theta) * mscale;
        float x0 = x[n_nope + i];
        float x1 = x[n_nope + i + 1];
        x[n_nope + i] = x0 * c - x1 * s;
        x[n_nope + i + 1] = x0 * s + x1 * c;
    }
}

__global__ static void fp8_kv_quantize_kernel(float *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t n_nope = head_dim - n_rot;
    float *xr = x + (uint64_t)row * head_dim;
    __shared__ float scratch[64];
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (off + tid < n_nope) v = xr[off + tid];
        scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        if (off + tid < n_nope) {
            float q = dsv4_e4m3fn_dequant_dev(fminf(448.0f, fmaxf(-448.0f, v / scale))) * scale;
            xr[off + tid] = q;
        }
        __syncthreads();
    }
}

/* kv 尾链三合一(2026-08-20 megakernel G1b): rope(rot尾段) → fp8(nope前段, 64线程树
 * 逐位照抄, barrier 全 block 陪跑) → store(全行 f16 往返)。decode n=1 单行单 block。 */
__global__ static void kv_rope_fp8_store_kernel(
        float *kv, float *raw, uint32_t raw_cap, uint32_t raw_row,
        uint32_t head_dim, uint32_t n_rot, uint32_t pos,
        uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    const uint32_t tid = threadIdx.x;
    const uint32_t n_nope = head_dim - n_rot;
    /* 段B: rope(先于 fp8, 与原三发同序; 只动 [n_nope, head_dim)) */
    if (tid < (n_rot >> 1)) {
        const uint32_t i = tid * 2u;
        float corr0 = 0.0f, corr1 = 0.0f;
        if (ext_factor != 0.0f) {
            float denom = 2.0f * logf(freq_base);
            corr0 = floorf((float)n_rot * logf((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
            corr1 = ceilf((float)n_rot * logf((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
            corr0 = fmaxf(0.0f, corr0);
            corr1 = fminf((float)(n_rot - 1), corr1);
        }
        float theta_extrap = (float)pos * powf(freq_base, -((float)i) / (float)n_rot);
        float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            float ramp_mix = rope_yarn_ramp_dev(corr0, corr1, (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        const float c = cosf(theta) * mscale;
        const float sn = sinf(theta) * mscale;
        float *tail = kv + n_nope;
        const float x0 = tail[i], x1 = tail[i + 1];
        tail[i] = x0 * c - x1 * sn;
        tail[i + 1] = x0 * sn + x1 * c;
    }
    __syncthreads();
    /* 段A: fp8(逐位照抄 fp8_kv_quantize_kernel; barrier 全 block 到达, 归约仍 64 线程同序) */
    __shared__ float scratch[64];
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (tid < 64u) {
            if (off + tid < n_nope) v = kv[off + tid];
            scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
        }
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
        if (tid < 64u && off + tid < n_nope) {
            float q = dsv4_e4m3fn_dequant_dev(fminf(448.0f, fmaxf(-448.0f, v / scale))) * scale;
            kv[off + tid] = q;
        }
        __syncthreads();
    }
    /* 段C: store 全行(f16 往返, 语义同 store_raw_kv_batch_kernel n=1) */
    for (uint32_t d = tid; d < head_dim; d += blockDim.x)
        raw[(uint64_t)(raw_row % raw_cap) * head_dim + d] = __half2float(__float2half(kv[d]));
}

int ds4_gpu_kv_rope_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv, ds4_gpu_tensor *raw_cache,
        uint32_t raw_cap, uint32_t raw_row, uint32_t head_dim, uint32_t n_rot,
        uint32_t pos, uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    if (!kv || !raw_cache || raw_cap == 0 || n_rot > head_dim || (n_rot & 1u) ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        kv->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    kv_rope_fp8_store_kernel<<<1, 256>>>(
        (float *)kv->ptr, (float *)raw_cache->ptr, raw_cap, raw_row,
        head_dim, n_rot, pos, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), "kv_rope_fp8_store launch");
}

__global__ static void indexer_hadamard_fp4_kernel(float *x, uint32_t n_rows, uint32_t head_dim) {
    uint32_t row = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (row >= n_rows || head_dim != 128u || tid >= 128u) return;

    __shared__ float vals[128];
    __shared__ float absbuf[128];
    float *xr = x + (uint64_t)row * head_dim;
    vals[tid] = xr[tid];
    __syncthreads();

    for (uint32_t stride = 1u; stride < 128u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            uint32_t base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            float a = vals[base];
            float b = vals[base + stride];
            vals[base] = a + b;
            vals[base + stride] = a - b;
        }
        __syncthreads();
    }

    float v = vals[tid] * 0.08838834764831845f;
    uint32_t fp4_block = tid >> 5u;
    uint32_t lane = tid & 31u;
    uint32_t block_base = fp4_block * 32u;
    absbuf[tid] = fabsf(v);
    __syncthreads();

    for (uint32_t stride = 16u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            absbuf[block_base + lane] = fmaxf(absbuf[block_base + lane],
                                              absbuf[block_base + lane + stride]);
        }
        __syncthreads();
    }

    float amax = fmaxf(absbuf[block_base], 7.052966104933725e-38f);
    float scale = exp2f(ceilf(log2f(amax / 6.0f)));
    xr[tid] = dsv4_e2m1fn_dequant_dev(fminf(6.0f, fmaxf(-6.0f, v / scale))) * scale;
}

__global__ static void store_raw_kv_batch_kernel(float *raw, const float *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t t = gid / head_dim;
    uint32_t row = (pos0 + t) % raw_cap;
    raw[(uint64_t)row * head_dim + d] = __half2float(__float2half(kv[(uint64_t)t * head_dim + d]));
}

__global__ static void attention_prefill_raw_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    uint32_t raw_count = t + 1 < window ? t + 1 : window;
    uint32_t raw_start = t + 1 - raw_count;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[256];
    __shared__ float partial[128];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        const float *kv = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    /* shuffle 规约版 max/denom(树形固定 ⇒ 确定): 原 8 轮 __syncthreads×2 */
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, off));
    if ((threadIdx.x & 31u) == 0) partial[threadIdx.x >> 5u] = local_max;
    __syncthreads();
    if (threadIdx.x < 32u) {
        const uint32_t nw = (blockDim.x + 31u) >> 5u;
        float m = (threadIdx.x < nw) ? partial[threadIdx.x] : -INFINITY;
        for (uint32_t off = 16u; off > 0u; off >>= 1u)
            m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
        if (threadIdx.x == 0) max_s = m;
    }
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < raw_count; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        den_local += __shfl_down_sync(0xffffffffu, den_local, off);
    if ((threadIdx.x & 31u) == 0) partial[threadIdx.x >> 5u] = den_local;
    __syncthreads();
    if (threadIdx.x < 32u) {
        const uint32_t nw = (blockDim.x + 31u) >> 5u;
        float s2 = (threadIdx.x < nw) ? partial[threadIdx.x] : 0.0f;
        for (uint32_t off = 16u; off > 0u; off >>= 1u)
            s2 += __shfl_down_sync(0xffffffffu, s2, off);
        if (threadIdx.x == 0) denom = s2 + expf(sinks[h] - max_s);
    }
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        }
        oh[d] = acc / denom;
    }
}

__global__ static void attention_prefill_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    uint32_t raw_start = (window != 0 && t + 1u > window) ? t + 1u - window : 0u;
    uint32_t raw_count = t + 1u - raw_start;
    uint32_t visible_comp = (t + 1u) / ratio;
    if (visible_comp > n_comp) visible_comp = n_comp;
    __shared__ float scores[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float scale = rsqrtf((float)head_dim);
    float local_max = sinks[h];
    uint32_t n_score = raw_count + visible_comp;

    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        const float *kvrow = raw_kv + (uint64_t)(raw_start + r) * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
        float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
        float s = -INFINITY;
        if (add > -1.0e20f) {
            const float *kvrow = comp_kv + (uint64_t)c * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            s = dot * scale + add;
        }
        scores[raw_count + c] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)(raw_start + r) * head_dim + d] * scores[r];
        for (uint32_t c = 0; c < visible_comp; c++) acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
        oh[d] = acc / denom;
    }
}

__global__ static void attention_prefill_raw_softmax_kernel(
        float *scores,
        const float *sinks,
        uint32_t n_tokens,
        uint32_t window,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        bool valid = k <= t && (window == 0 || t - k < window);
        float s = valid ? row[k] : -INFINITY;
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}

__global__ static void attention_prefill_mixed_softmax_kernel(
        float *scores,
        const float *sinks,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_keys) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || ratio == 0) return;
    float *row = scores + ((uint64_t)h * n_tokens + t) * n_keys;
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    float local_max = sinks[h];
    const uint32_t visible_comp = (t + 1u) / ratio;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float s = -INFINITY;
        if (k < n_tokens) {
            if (k <= t && (window == 0 || t - k < window)) s = row[k];
        } else {
            uint32_t c = k - n_tokens;
            if (c < n_comp && c < visible_comp) {
                float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                if (add > -1.0e20f) s = row[k] + add;
            }
        }
        row[k] = s;
        local_max = fmaxf(local_max, s);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) {
        float p = isfinite(row[k]) ? expf(row[k] - max_s) : 0.0f;
        row[k] = p;
        den_local += p;
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < n_keys; k += blockDim.x) row[k] /= denom;
}

__global__ static void attention_prefill_pack_mixed_kv_kernel(
        float *dst,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)(n_tokens + n_comp) * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint32_t r = gid / head_dim;
    dst[gid] = r < n_tokens ? raw_kv[(uint64_t)r * head_dim + d]
                             : comp_kv[(uint64_t)(r - n_tokens) * head_dim + d];
}

__global__ static void attention_prefill_unpack_heads_kernel(
        float *heads,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_head,
        uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
    if (gid >= n) return;
    uint32_t d = gid % head_dim;
    uint64_t q = gid / head_dim;
    uint32_t h = q % n_head;
    uint32_t t = q / n_head;
    heads[gid] = tmp[((uint64_t)h * n_tokens + t) * head_dim + d];
}

__global__ static void attention_pack_group_heads_f16_kernel(
        __half *dst,
        const float *heads,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t group_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_groups * n_tokens * group_dim;
    if (gid >= n) return;
    uint32_t d = gid % group_dim;
    uint64_t q = gid / group_dim;
    uint32_t t = q % n_tokens;
    uint32_t g = q / n_tokens;
    dst[gid] = __float2half(heads[((uint64_t)t * n_groups + g) * group_dim + d]);
}

__global__ static void attention_unpack_group_low_kernel(
        float *low,
        const float *tmp,
        uint32_t n_tokens,
        uint32_t n_groups,
        uint32_t rank) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_groups * n_tokens * rank;
    if (gid >= n) return;
    uint32_t r = gid % rank;
    uint64_t q = gid / rank;
    uint32_t t = q % n_tokens;
    uint32_t g = q / n_tokens;
    uint32_t low_dim = n_groups * rank;
    low[(uint64_t)t * low_dim + (uint64_t)g * rank + r] = tmp[gid];
}

__global__ static void attention_decode_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const bool single_all = (n_tokens == 1u && ratio == 0u);
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = single_all ? n_comp : (n_comp ? (qpos + 1u) / ratio : 0u);
    if (visible_comp > n_comp) visible_comp = n_comp;
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[DS4_CUDA_ATTENTION_SCORE_CAP];
    __shared__ uint32_t raw_rows[256];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (single_all) {
                raw_count = n_raw > 256u ? 256u : n_raw;
            } else if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();
    uint32_t n_score = raw_count + visible_comp;
    float local_max = sinks[h];
    /* 批/单同路(2026-08-21 verify 等价): 原条件把"有压缩槽且 n_tokens>1"甩到下面
     * 8-lane-per-row 分支, head_dim 上的求和序与单 token 的 warp-per-row 不同 ⇒ 差 1 ULP,
     * 再被下游 q8 激活量化(1 档=1/127)放大到 1e-4, 逐层滚成 logit 级偏差, 吃掉投机接受率。
     * 形状够就一律走 float4 warp-per-row: 逐位与 decode 一致。 */
    if ((head_dim & 3u) == 0u) {
        /* 重写(2026-08-17): 原 thread-per-row 标量版 = 每线程串行读整条 2KB 行,
         * warp 32 线程同时戳 32 个不同行完全不合并。现 warp-per-row + float4:
         * 32 lane 连续 16B 粒度合并读一行, shuffle 树归约。 */
        const uint32_t wp = threadIdx.x >> 5u, ln = threadIdx.x & 31u;
        const uint32_t hd4 = head_dim >> 2u;
        const float4 *q4 = (const float4 *)qh;
        for (uint32_t r = wp; r < raw_count; r += 8u) {
            const float4 *kv4 = (const float4 *)(raw_kv + (uint64_t)raw_rows[r] * head_dim);
            float dot = 0.0f;
            for (uint32_t d = ln; d < hd4; d += 32u) {
                const float4 k = kv4[d], qq = q4[d];
                dot += qq.x * k.x + qq.y * k.y + qq.z * k.z + qq.w * k.w;
            }
            for (uint32_t off = 16u; off > 0u; off >>= 1u) dot += __shfl_down_sync(0xffffffffu, dot, off);
            if (ln == 0) scores[r] = dot * scale;
        }
        for (uint32_t c = wp; c < visible_comp; c += 8u) {
            float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
            float s = -INFINITY;
            if (add > -1.0e20f) {
                const float4 *kv4 = (const float4 *)(comp_kv + (uint64_t)c * head_dim);
                float dot = 0.0f;
                for (uint32_t d = ln; d < hd4; d += 32u) {
                    const float4 k = kv4[d], qq = q4[d];
                    dot += qq.x * k.x + qq.y * k.y + qq.z * k.z + qq.w * k.w;
                }
                for (uint32_t off = 16u; off > 0u; off >>= 1u) dot += __shfl_down_sync(0xffffffffu, dot, off);
                s = dot * scale + add;
            }
            if (ln == 0) scores[raw_count + c] = s;
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x)
            local_max = fmaxf(local_max, scores[i]);
    } else if (visible_comp == 0 || n_tokens == 1u) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
        for (uint32_t c = threadIdx.x; c < visible_comp; c += blockDim.x) {
            float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
            float s = -INFINITY;
            if (add > -1.0e20f) {
                const float *kvrow = comp_kv + (uint64_t)c * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
                s = dot * scale + add;
            }
            scores[raw_count + c] = s;
            local_max = fmaxf(local_max, s);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                float add = 0.0f;
                const float *kvrow = NULL;
                if (row < raw_count) {
                    kvrow = raw_kv + (uint64_t)raw_rows[row] * head_dim;
                } else {
                    uint32_t c = row - raw_count;
                    add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
                    if (add > -1.0e20f) kvrow = comp_kv + (uint64_t)c * head_dim;
                }
                float s = -INFINITY;
                if (kvrow) {
                    float dot = 0.0f;
                    for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                    const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                    for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                        dot += __shfl_down_sync(mask, dot, off, 8);
                    }
                    s = dot * scale + add;
                }
                if (qlane == 0) scores[row] = s;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    /* shuffle 规约版 max/denom(树形固定 ⇒ 确定): 原 8 轮 __syncthreads×2 */
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffffu, local_max, off));
    if ((threadIdx.x & 31u) == 0) partial[threadIdx.x >> 5u] = local_max;
    __syncthreads();
    if (threadIdx.x < 32u) {
        const uint32_t nw = (blockDim.x + 31u) >> 5u;
        float m = (threadIdx.x < nw) ? partial[threadIdx.x] : -INFINITY;
        for (uint32_t off = 16u; off > 0u; off >>= 1u)
            m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
        if (threadIdx.x == 0) max_s = m;
    }
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    for (uint32_t off = 16u; off > 0u; off >>= 1u)
        den_local += __shfl_down_sync(0xffffffffu, den_local, off);
    if ((threadIdx.x & 31u) == 0) partial[threadIdx.x >> 5u] = den_local;
    __syncthreads();
    if (threadIdx.x < 32u) {
        const uint32_t nw = (blockDim.x + 31u) >> 5u;
        float s2 = (threadIdx.x < nw) ? partial[threadIdx.x] : 0.0f;
        for (uint32_t off = 16u; off > 0u; off >>= 1u)
            s2 += __shfl_down_sync(0xffffffffu, s2, off);
        if (threadIdx.x == 0) denom = s2 + expf(sinks[h] - max_s);
    }
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < visible_comp; c++) {
            float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            for (uint32_t c = 0; c < visible_comp; c++) acc += comp_kv[(uint64_t)c * head_dim + d] * scores[raw_count + c];
            oh[d] = acc / denom;
        }
    }
}

__global__ static void attention_indexed_mixed_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }
    const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
    __shared__ float scores[768];
    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ float partial[256];
    __shared__ float max_s;
    __shared__ float denom;
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    float scale = rsqrtf((float)head_dim);
    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    /* 单线程保序压紧: atomicAdd 抢槽的槽序随调度漂移 ⇒ 后面 score/softmax 的
     * 浮点求和顺序每 run 不同, 温 0 输出在 ULP tie 处翻 token。indexer 的 topk
     * 顺序本身是确定的, 按它保序填充即可复现。 */
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < top_k; i++) {
            int32_t c = topk[(uint64_t)t * top_k + i];
            if (c >= 0 && (uint32_t)c < visible_comp && comp_count < 512u)
                comp_rows[comp_count++] = (uint32_t)c;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (comp_count > 512u) comp_count = 512u;
    }
    __syncthreads();
    uint32_t n_score = raw_count + comp_count;
    float local_max = sinks[h];
    if (comp_count == 0) {
        for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
            const float *kvrow = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvrow[d];
            scores[r] = dot * scale;
            local_max = fmaxf(local_max, scores[r]);
        }
    } else {
        uint32_t qlane = threadIdx.x & 7u;
        uint32_t qgroup = threadIdx.x >> 3u;
        for (uint32_t row0 = 0; row0 < n_score; row0 += 32u) {
            uint32_t row = row0 + qgroup;
            if (row < n_score) {
                const float *kvrow = row < raw_count
                    ? raw_kv + (uint64_t)raw_rows[row] * head_dim
                    : comp_kv + (uint64_t)comp_rows[row - raw_count] * head_dim;
                float dot = 0.0f;
                for (uint32_t d = qlane; d < head_dim; d += 8u) dot += qh[d] * kvrow[d];
                const uint32_t mask = 0xffu << (threadIdx.x & 24u);
                for (uint32_t off = 4u; off > 0u; off >>= 1u) {
                    dot += __shfl_down_sync(mask, dot, off, 8);
                }
                if (qlane == 0) scores[row] = dot * scale;
            }
        }
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
            local_max = fmaxf(local_max, scores[i]);
        }
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
    if (head_dim == 512u && blockDim.x == 256u) {
        uint32_t d0 = threadIdx.x;
        uint32_t d1 = d0 + 256u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) {
            float s = scores[r];
            const float *kv = raw_kv + (uint64_t)raw_rows[r] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        for (uint32_t c = 0; c < comp_count; c++) {
            float s = scores[raw_count + c];
            const float *kv = comp_kv + (uint64_t)comp_rows[c] * head_dim;
            acc0 += kv[d0] * s;
            acc1 += kv[d1] * s;
        }
        oh[d0] = acc0 / denom;
        oh[d1] = acc1 / denom;
    } else {
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)raw_rows[r] * head_dim + d] * scores[r];
            for (uint32_t s = 0; s < comp_count; s++) acc += comp_kv[(uint64_t)comp_rows[s] * head_dim + d] * scores[raw_count + s];
            oh[d] = acc / denom;
        }
    }
}

__global__ static void attention_indexed_mixed_heads8_rb4_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t comp_rows[512];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ uint32_t comp_count;
    __shared__ float4 kv_shared[4 * 128];
    __shared__ float scores[8 * 768];

    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        comp_count = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    if (threadIdx.x == 0) {
        for (uint32_t i = 0; i < top_k && comp_count < 512u; i++) {
            int32_t c = topk[(uint64_t)t * top_k + i];
            if (c >= 0 && (uint32_t)c < visible_comp) comp_rows[comp_count++] = (uint32_t)c;
        }
    }
    __syncthreads();

    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)comp_rows[sr - raw_count] * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float dot = dot4_f32(q0, kv4[lane +  0u]) +
                            dot4_f32(q1, kv4[lane + 32u]) +
                            dot4_f32(q2, kv4[lane + 64u]) +
                            dot4_f32(q3, kv4[lane + 96u]);
                dot = warp_sum_f32(dot);
                if (lane == 0) scores[warp * 768u + row0 + rr] = dot * scale;
            }
        }
        __syncthreads();
    }

    float max_s = valid_head ? sinks[head] : -INFINITY;
    if (valid_head) {
        const float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) max_s = fmaxf(max_s, score_row[i]);
        max_s = warp_max_f32(max_s);
        max_s = __shfl_sync(0xffffffffu, max_s, 0);
    }
    float den = 0.0f;
    if (valid_head) {
        float *score_row = scores + warp * 768u;
        for (uint32_t i = lane; i < n_score; i += 32u) {
            float p = expf(score_row[i] - max_s);
            score_row[i] = p;
            den += p;
        }
        den = warp_sum_f32(den);
        den += expf(sinks[head] - max_s);
        den = __shfl_sync(0xffffffffu, den, 0);
    }

    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;
    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)comp_rows[sr - raw_count] * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            const float *score_row = scores + warp * 768u;
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float p = den == 0.0f ? 0.0f : score_row[row0 + rr] / den;
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                o0.x += k0.x * p; o0.y += k0.y * p; o0.z += k0.z * p; o0.w += k0.w * p;
                o1.x += k1.x * p; o1.y += k1.y * p; o1.z += k1.z * p; o1.w += k1.w * p;
                o2.x += k2.x * p; o2.y += k2.y * p; o2.z += k2.z * p; o2.w += k2.w * p;
                o3.x += k3.x * p; o3.y += k3.y * p; o3.z += k3.z * p; o3.w += k3.w * p;
            }
        }
        __syncthreads();
    }
    if (valid_head) {
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

template <uint32_t ROWS_PER_STAGE, uint32_t HEADS_PER_GROUP>
__global__ static void attention_indexed_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        const int32_t *topk,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t top_k,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * HEADS_PER_GROUP + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count;
    __shared__ uint32_t raw_first_idx;
    __shared__ float4 kv_shared[ROWS_PER_STAGE * 128];

    uint32_t qpos = pos0 + t;
    uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t visible_comp = n_comp;
    if (ratio != 0) {
        visible_comp = (qpos + 1u) / ratio;
        if (visible_comp > n_comp) visible_comp = n_comp;
    }

    if (threadIdx.x == 0) {
        raw_count = 0;
        raw_first_idx = 0;
        if (n_raw != 0) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0 && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
    }
    __syncthreads();
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    uint32_t comp_count = top_k < visible_comp ? top_k : visible_comp;
    if (comp_count > 512u) comp_count = 512u;
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += ROWS_PER_STAGE) {
        const uint32_t nr = n_score - row0 < ROWS_PER_STAGE ? n_score - row0 : ROWS_PER_STAGE;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const uint32_t comp_idx = sr < raw_count
                ? 0u
                : (uint32_t)topk[(uint64_t)t * top_k + (sr - raw_count)];
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)comp_idx * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__global__ static void attention_static_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ float4 kv_shared[4 * 128];

    const uint32_t raw_count = window != 0u && t + 1u > window ? window : t + 1u;
    const uint32_t raw_start = t + 1u - raw_count;
    uint32_t comp_count = 0;
    if (n_comp != 0u && ratio != 0u) {
        comp_count = (t + 1u) / ratio;
        if (comp_count > n_comp) comp_count = n_comp;
    }
    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)(raw_start + sr) * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__global__ static void attention_decode_mixed_heads8_online_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const float *comp_kv,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_raw,
        uint32_t raw_cap,
        uint32_t raw_start,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head,
        uint32_t head_dim) {
    uint32_t t = blockIdx.x;
    uint32_t head_group = blockIdx.y;
    if (t >= n_tokens || head_dim != 512u) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t head = head_group * 8u + warp;
    const bool valid_head = head < n_head;

    __shared__ uint32_t raw_rows[256];
    __shared__ uint32_t raw_count_s;
    __shared__ uint32_t raw_first_idx_s;
    __shared__ float4 kv_shared[4 * 128];

    const uint32_t qpos = pos0 + t;
    const uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
    uint32_t comp_count = 0;
    if (n_comp != 0u) {
        if (n_tokens == 1u && ratio == 0u) {
            comp_count = n_comp;
        } else if (ratio != 0u) {
            comp_count = (qpos + 1u) / ratio;
            if (comp_count > n_comp) comp_count = n_comp;
        }
    }
    if (threadIdx.x == 0) {
        uint32_t raw_count = 0;
        uint32_t raw_first_idx = 0;
        if (n_raw != 0u) {
            const uint32_t raw_last_pos = first_raw_pos + n_raw - 1u;
            if (qpos >= first_raw_pos) {
                uint32_t lo = first_raw_pos;
                if (window != 0u && qpos + 1u > window) {
                    const uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                const uint32_t hi = qpos < raw_last_pos ? qpos : raw_last_pos;
                if (hi >= lo) {
                    raw_first_idx = lo - first_raw_pos;
                    raw_count = hi - lo + 1u;
                    if (raw_count > 256u) raw_count = 256u;
                }
            }
        }
        raw_count_s = raw_count;
        raw_first_idx_s = raw_first_idx;
    }
    __syncthreads();
    const uint32_t raw_count = raw_count_s;
    const uint32_t raw_first_idx = raw_first_idx_s;
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        raw_rows[r] = (raw_start + raw_first_idx + r) % raw_cap;
    }
    __syncthreads();

    const uint32_t n_score = raw_count + comp_count;
    const float scale = rsqrtf((float)head_dim);
    const float4 *q4 = valid_head
        ? (const float4 *)(q + ((uint64_t)t * n_head + head) * head_dim)
        : NULL;
    float4 q0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        q0 = q4[lane +  0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    for (uint32_t row0 = 0; row0 < n_score; row0 += 4u) {
        const uint32_t nr = n_score - row0 < 4u ? n_score - row0 : 4u;
        for (uint32_t off = threadIdx.x; off < nr * 128u; off += blockDim.x) {
            const uint32_t rr = off >> 7u;
            const uint32_t c4 = off & 127u;
            const uint32_t sr = row0 + rr;
            const float4 *src = sr < raw_count
                ? (const float4 *)(raw_kv + (uint64_t)raw_rows[sr] * head_dim)
                : (const float4 *)(comp_kv + (uint64_t)(sr - raw_count) * head_dim);
            kv_shared[off] = src[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0; rr < nr; rr++) {
                const float4 *kv4 = kv_shared + rr * 128u;
                float4 k0 = kv4[lane +  0u];
                float4 k1 = kv4[lane + 32u];
                float4 k2 = kv4[lane + 64u];
                float4 k3 = kv4[lane + 96u];
                float score = dot4_f32(q0, k0) +
                              dot4_f32(q1, k1) +
                              dot4_f32(q2, k2) +
                              dot4_f32(q3, k3);
                score = warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + k0.x * row_scale;
                o0.y = o0.y * old_scale + k0.y * row_scale;
                o0.z = o0.z * old_scale + k0.z * row_scale;
                o0.w = o0.w * old_scale + k0.w * row_scale;
                o1.x = o1.x * old_scale + k1.x * row_scale;
                o1.y = o1.y * old_scale + k1.y * row_scale;
                o1.z = o1.z * old_scale + k1.z * row_scale;
                o1.w = o1.w * old_scale + k1.w * row_scale;
                o2.x = o2.x * old_scale + k2.x * row_scale;
                o2.y = o2.y * old_scale + k2.y * row_scale;
                o2.z = o2.z * old_scale + k2.z * row_scale;
                o2.w = o2.w * old_scale + k2.w * row_scale;
                o3.x = o3.x * old_scale + k3.x * row_scale;
                o3.y = o3.y * old_scale + k3.y * row_scale;
                o3.z = o3.z * old_scale + k3.z * row_scale;
                o3.w = o3.w * old_scale + k3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (valid_head) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;

        const float inv_s = sum_s == 0.0f ? 0.0f : 1.0f / sum_s;
        o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
        o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
        o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
        o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;
        float4 *out4 = (float4 *)(heads + ((uint64_t)t * n_head + head) * head_dim);
        out4[lane +  0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
    }
}

__device__ static void hc4_split_one(float *out, const float *mix, const float *scale, const float *base, uint32_t sinkhorn_iters, float epsv) {
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    for (int i = 0; i < 4; i++) {
        float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }
    for (int i = 0; i < 4; i++) {
        float z = mix[4 + i] * post_scale + base[4 + i];
        out[4 + i] = 2.0f / (1.0f + expf(-z));
    }
    float c[16];
    for (int r = 0; r < 4; r++) {
        float m = -INFINITY;
        for (int col = 0; col < 4; col++) {
            float v = mix[8 + r * 4 + col] * comb_scale + base[8 + r * 4 + col];
            c[r * 4 + col] = v;
            m = fmaxf(m, v);
        }
        float s = 0.0f;
        for (int col = 0; col < 4; col++) {
            float v = expf(c[r * 4 + col] - m);
            c[r * 4 + col] = v;
            s += v;
        }
        for (int col = 0; col < 4; col++) c[r * 4 + col] = c[r * 4 + col] / s + epsv;
    }
    for (int col = 0; col < 4; col++) {
        float s = epsv;
        for (int r = 0; r < 4; r++) s += c[r * 4 + col];
        for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
    }
    for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
        for (int r = 0; r < 4; r++) {
            float s = epsv;
            for (int col = 0; col < 4; col++) s += c[r * 4 + col];
            for (int col = 0; col < 4; col++) c[r * 4 + col] /= s;
        }
        for (int col = 0; col < 4; col++) {
            float s = epsv;
            for (int r = 0; r < 4; r++) s += c[r * 4 + col];
            for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
        }
    }
    for (int i = 0; i < 16; i++) out[8 + i] = c[i];
}

__global__ static void hc_split_sinkhorn_kernel(float *out, const float *mix, const float *scale, const float *base, uint32_t n_rows, uint32_t sinkhorn_iters, float epsv) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    hc4_split_one(out + (uint64_t)row * 24, mix + (uint64_t)row * 24, scale, base, sinkhorn_iters, epsv);
}

__global__ static void hc_weighted_sum_kernel(float *out, const float *x, const float *w, uint32_t n_embd, uint32_t n_hc, uint32_t n_tokens, uint32_t weight_stride_f32) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_embd * n_tokens;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint32_t t = gid / n_embd;
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_hc; h++) {
        acc += x[(uint64_t)t * n_hc * n_embd + (uint64_t)h * n_embd + d] *
               w[(uint64_t)t * weight_stride_f32 + h];
    }
    out[(uint64_t)t * n_embd + d] = acc;
}

__global__ static void hc_expand_kernel(
        float *out_hc,
        const float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *post,
        const float *comb,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_tokens,
        uint32_t post_stride,
        uint32_t comb_stride,
        int has_add) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n_elem) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t dst_hc = tmp % n_hc;
    uint32_t t = tmp / n_hc;

    float block_v = block_out[(uint64_t)t * n_embd + d];
    if (has_add) block_v += block_add[(uint64_t)t * n_embd + d];
    float acc = block_v * post[(uint64_t)t * post_stride + dst_hc];
    for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
        float comb_v = comb[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src_hc * n_hc];
        float res_v = residual_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)src_hc * n_embd + d];
        acc += comb_v * res_v;
    }
    out_hc[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = acc;
}

__global__ static void hc_split_weighted_sum_fused_kernel(
        float *out,
        float *split,
        const float *mix,
        const float *residual_hc,
        const float *scale,
        const float *base,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv) {
    uint32_t t = blockIdx.x;
    uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    if (d == 0) hc4_split_one(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv);
    __syncthreads();
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += residual_hc[(uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col] * sp[h];
        }
        out[(uint64_t)t * n_embd + col] = acc;
    }
}

/* 快路径(2026-08-17): 老版 13.6μs×2/层 —— 8 轮树规约 + norm 阶段重读 out + 窄读。
 * 新版: float4 宽读 4 路残差, acc 留寄存器免重读, warp shuffle 规约(1 轮 sync)。
 * n_hc==4 且 n_embd%4==0 专用; sinkhorn 头仍单线程(固定小串行)。 */
__global__ static void hc_split_wsn_fast_kernel(
        float *out,
        float *norm_out,
        float *split,
        const float *mix,
        const float *residual_hc,
        const float *scale,
        const float *base,
        const float *norm_w,
        uint32_t n_embd,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv,
        float norm_eps) {
    const uint32_t t = blockIdx.x;
    if (t >= n_rows) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    if (threadIdx.x == 0) hc4_split_one(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv);
    __syncthreads();
    const float s0 = sp[0], s1 = sp[1], s2 = sp[2], s3 = sp[3];
    const uint32_t n4 = n_embd >> 2;
    const float4 *r0 = (const float4 *)(residual_hc + (uint64_t)t * 4u * n_embd);
    const float4 *r1 = r0 + n4, *r2 = r1 + n4, *r3 = r2 + n4;
    float4 *o4 = (float4 *)(out + (uint64_t)t * n_embd);
    float4 *no4 = (float4 *)(norm_out + (uint64_t)t * n_embd);
    const float4 *w4 = (const float4 *)norm_w;
    /* blockDim=1024, n4=1024: 每线程恰一个 float4; 泛化仍循环 */
    float sum = 0.0f;
    float4 acc_keep = make_float4(0.f, 0.f, 0.f, 0.f);
    uint32_t keep_i = 0xffffffffu;
    for (uint32_t i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 a = r0[i], b = r1[i], c = r2[i], e = r3[i];
        float4 acc;
        acc.x = a.x * s0 + b.x * s1 + c.x * s2 + e.x * s3;
        acc.y = a.y * s0 + b.y * s1 + c.y * s2 + e.y * s3;
        acc.z = a.z * s0 + b.z * s1 + c.z * s2 + e.z * s3;
        acc.w = a.w * s0 + b.w * s1 + c.w * s2 + e.w * s3;
        o4[i] = acc;
        sum += acc.x * acc.x + acc.y * acc.y + acc.z * acc.z + acc.w * acc.w;
        acc_keep = acc; keep_i = i;   /* 单次迭代场景=保寄存器; 多迭代由下方重读兜 */
    }
    __shared__ float wsum[32];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if ((threadIdx.x & 31u) == 0) wsum[threadIdx.x >> 5u] = sum;
    __syncthreads();
    if (threadIdx.x < 32u) {
        const uint32_t nw = (blockDim.x + 31u) >> 5u;
        float v = (threadIdx.x < nw) ? wsum[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (threadIdx.x == 0) wsum[0] = v;
    }
    __syncthreads();
    const float ns = rsqrtf(wsum[0] / (float)n_embd + norm_eps);
    if (n4 <= blockDim.x) {
        if (keep_i != 0xffffffffu) {
            const float4 g = w4[keep_i];
            float4 r; r.x = acc_keep.x * ns * g.x; r.y = acc_keep.y * ns * g.y;
            r.z = acc_keep.z * ns * g.z; r.w = acc_keep.w * ns * g.w;
            no4[keep_i] = r;
        }
    } else {
        for (uint32_t i = threadIdx.x; i < n4; i += blockDim.x) {
            const float4 v = o4[i], g = w4[i];
            float4 r; r.x = v.x * ns * g.x; r.y = v.y * ns * g.y;
            r.z = v.z * ns * g.z; r.w = v.w * ns * g.w;
            no4[i] = r;
        }
    }
}

__global__ static void hc_split_weighted_sum_norm_fused_kernel(
        float *out,
        float *norm_out,
        float *split,
        const float *mix,
        const float *residual_hc,
        const float *scale,
        const float *base,
        const float *norm_w,
        uint32_t n_embd,
        uint32_t n_hc,
        uint32_t n_rows,
        uint32_t sinkhorn_iters,
        float epsv,
        float norm_eps) {
    const uint32_t t = blockIdx.x;
    const uint32_t d = threadIdx.x;
    if (t >= n_rows || n_hc != 4) return;
    const uint32_t mix_hc = 24;
    float *sp = split + (uint64_t)t * mix_hc;
    if (d == 0) hc4_split_one(sp, mix + (uint64_t)t * mix_hc, scale, base, sinkhorn_iters, epsv);
    __syncthreads();

    float sum = 0.0f;
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < 4; h++) {
            acc += residual_hc[(uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col] * sp[h];
        }
        out[(uint64_t)t * n_embd + col] = acc;
        sum += acc * acc;
    }

    __shared__ float partial[256];
    partial[d] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (d < stride) partial[d] += partial[d + stride];
        __syncthreads();
    }
    const float norm_scale = rsqrtf(partial[0] / (float)n_embd + norm_eps);
    for (uint32_t col = d; col < n_embd; col += blockDim.x) {
        const float v = out[(uint64_t)t * n_embd + col];
        norm_out[(uint64_t)t * n_embd + col] = v * norm_scale * norm_w[col];
    }
}

__global__ static void output_hc_weights_kernel(
        float *out,
        const float *pre,
        const float *scale,
        const float *base,
        uint32_t n_hc,
        uint32_t n_tokens,
        float epsv) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_tokens * n_hc;
    if (gid >= n) return;
    uint32_t h = gid % n_hc;
    float z = pre[gid] * scale[0] + base[h];
    out[gid] = 1.0f / (1.0f + expf(-z)) + epsv;
}

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}

__global__ static void compressor_store_kernel(
        const float *kv,
        const float *sc,
        float *state_kv,
        float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_tokens) {
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * width;
    if (gid >= n) return;
    uint32_t t = gid / width;
    uint32_t j = gid - (uint64_t)t * width;
    uint32_t pos_mod = (pos0 + t) % ratio;
    uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    state_kv[(uint64_t)dst_row * width + j] = kv[(uint64_t)t * width + j];
    state_score[(uint64_t)dst_row * width + j] =
        sc[(uint64_t)t * width + j] + model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)pos_mod * width + j);
}

__global__ static void compressor_set_rows_kernel(
        float *state_kv,
        float *state_score,
        const float *kv,
        const float *sc,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t width,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t src0,
        uint32_t dst0,
        uint32_t rows) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)rows * width;
    if (gid >= n) return;
    uint32_t r = gid / width;
    uint32_t j = gid - (uint64_t)r * width;
    uint32_t src = src0 + r;
    uint32_t dst = dst0 + r;
    uint32_t phase = (pos0 + src) % ratio;
    state_kv[(uint64_t)dst * width + j] = kv[(uint64_t)src * width + j];
    state_score[(uint64_t)dst * width + j] =
        sc[(uint64_t)src * width + j] + model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)phase * width + j);
}

__global__ static void compressor_prefill_pool_kernel(
        float *comp,
        const float *kv,
        const float *sc,
        const float *state_kv,
        const float *state_score,
        const void *model_map,
        uint64_t ape_offset,
        uint32_t ape_type,
        uint32_t head_dim,
        uint32_t ratio,
        uint32_t pos0,
        uint32_t n_comp,
        uint32_t replay) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t c = blockIdx.y;
    if (d >= head_dim || c >= n_comp) return;
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        if (replay && c == 0) {
            for (uint32_t r = 0; r < 4; r++) {
                vals[n_cand] = state_kv[(uint64_t)r * width + d];
                scores[n_cand] = state_score[(uint64_t)r * width + d];
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        } else if (c > 0) {
            uint32_t base = (c - 1u) * ratio;
            for (uint32_t r = 0; r < 4; r++) {
                uint32_t t = base + r;
                float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
                vals[n_cand] = kv[(uint64_t)t * width + d];
                scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
                max_s = fmaxf(max_s, scores[n_cand++]);
            }
        }
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < 4; r++) {
            uint32_t t = base + r;
            float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + head_dim + d);
            vals[n_cand] = kv[(uint64_t)t * width + head_dim + d];
            scores[n_cand] = sc[(uint64_t)t * width + head_dim + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        uint32_t base = c * ratio;
        for (uint32_t r = 0; r < ratio; r++) {
            uint32_t t = base + r;
            float ape = model_scalar_dev(model_map, ape_offset, ape_type, (uint64_t)((pos0 + t) % ratio) * width + d);
            vals[n_cand] = kv[(uint64_t)t * width + d];
            scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    comp[(uint64_t)c * head_dim + d] = den != 0.0f ? acc / den : 0.0f;
}

__global__ static void compressor_update_pool_kernel(
        float *row,
        const float *state_kv,
        const float *state_score,
        uint32_t head_dim,
        uint32_t ratio) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    uint32_t coff = ratio == 4u ? 2u : 1u;
    uint32_t width = coff * head_dim;
    float vals[128];
    float scores[128];
    float max_s = -INFINITY;
    uint32_t n_cand = 0;
    if (ratio == 4u) {
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
        for (uint32_t r = 0; r < 4; r++) {
            vals[n_cand] = state_kv[(uint64_t)(ratio + r) * width + head_dim + d];
            scores[n_cand] = state_score[(uint64_t)(ratio + r) * width + head_dim + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    } else {
        for (uint32_t r = 0; r < ratio; r++) {
            vals[n_cand] = state_kv[(uint64_t)r * width + d];
            scores[n_cand] = state_score[(uint64_t)r * width + d];
            max_s = fmaxf(max_s, scores[n_cand++]);
        }
    }
    float den = 0.0f, acc = 0.0f;
    for (uint32_t i = 0; i < n_cand; i++) {
        float w = expf(scores[i] - max_s);
        den += w;
        acc += vals[i] * w;
    }
    row[d] = den != 0.0f ? acc / den : 0.0f;
}

__global__ static void compressor_shift_ratio4_kernel(float *state_kv, float *state_score, uint32_t width) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t half = 4ull * width;
    if (i >= half) return;
    float v = state_kv[half + i];
    float s = state_score[half + i];
    state_kv[i] = v;
    state_score[i] = s;
    state_kv[half + i] = v;
    state_score[half + i] = s;
}

__device__ static float softplus_dev(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__global__ static void router_select_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode) {
    uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) return;
    const float *log = logits + (uint64_t)t * 256;
    float *prob = probs + (uint64_t)t * 256;
    int32_t *sel = selected + (uint64_t)t * 6;
    float *w = weights + (uint64_t)t * 6;

    for (int i = 0; i < 256; i++) prob[i] = sqrtf(softplus_dev(log[i]));

    if (hash_mode) {
        int32_t tok = tokens ? tokens[t] : token_scalar;
        if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
        const int32_t *row = hash + (uint64_t)tok * 6;
        for (int i = 0; i < 6; i++) sel[i] = row[i];
    } else {
        for (int i = 0; i < 6; i++) sel[i] = -1;
        for (int i = 0; i < 256; i++) {
            float score = prob[i] + (has_bias ? bias[i] : 0.0f);
            for (int j = 0; j < 6; j++) {
                if (sel[j] < 0 || score > prob[sel[j]] + (has_bias ? bias[sel[j]] : 0.0f)) {
                    for (int k = 5; k > j; k--) sel[k] = sel[k - 1];
                    sel[j] = i;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < 6; i++) {
        int e = sel[i];
        float v = (e >= 0 && e < 256) ? prob[e] : 0.0f;
        w[i] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (int i = 0; i < 6; i++) w[i] = w[i] / sum * 1.5f;
}

__global__ static void router_select_parallel_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode) {
    uint32_t t = blockIdx.x;
    uint32_t i = threadIdx.x;
    if (t >= n_tokens || i >= 256u) return;
    const float *log = logits + (uint64_t)t * 256;
    float *prob = probs + (uint64_t)t * 256;
    int32_t *sel = selected + (uint64_t)t * 6;
    float *w = weights + (uint64_t)t * 6;
    __shared__ float sprob[256];

    const float p = sqrtf(softplus_dev(log[i]));
    sprob[i] = p;
    prob[i] = p;
    __syncthreads();

    if (i != 0) return;
    if (hash_mode) {
        int32_t tok = tokens ? tokens[t] : token_scalar;
        if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
        const int32_t *row = hash + (uint64_t)tok * 6;
        for (int j = 0; j < 6; j++) sel[j] = row[j];
    } else {
        for (int j = 0; j < 6; j++) sel[j] = -1;
        for (int e = 0; e < 256; e++) {
            float score = sprob[e] + (has_bias ? bias[e] : 0.0f);
            for (int j = 0; j < 6; j++) {
                if (sel[j] < 0 || score > sprob[sel[j]] + (has_bias ? bias[sel[j]] : 0.0f)) {
                    for (int k = 5; k > j; k--) sel[k] = sel[k - 1];
                    sel[j] = e;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (int j = 0; j < 6; j++) {
        int e = sel[j];
        float v = (e >= 0 && e < 256) ? sprob[e] : 0.0f;
        w[j] = v;
        sum += v;
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (int j = 0; j < 6; j++) w[j] = w[j] / sum * 1.5f;
}

__device__ __forceinline__ static bool router_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__global__ static void router_select_warp_topk_kernel(
        int32_t *selected,
        float *weights,
        float *probs,
        const float *bias,
        const int32_t *hash,
        const float *logits,
        const int32_t *tokens,
        int32_t token_scalar,
        uint32_t hash_rows,
        uint32_t n_tokens,
        int has_bias,
        int hash_mode) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row_in_block = threadIdx.y;
    const uint32_t t = blockIdx.x * blockDim.y + row_in_block;
    if (t >= n_tokens || lane >= 32u) return;

    const float *log = logits + (uint64_t)t * 256u;
    float *prob = probs + (uint64_t)t * 256u;
    int32_t *sel = selected + (uint64_t)t * 6u;
    float *w = weights + (uint64_t)t * 6u;
    __shared__ float sprob[4][256];
    float local_prob[8];
    float local_score[8];

    #pragma unroll
    for (uint32_t j = 0; j < 8u; j++) {
        const uint32_t e = lane + j * 32u;
        const float p = sqrtf(softplus_dev(log[e]));
        local_prob[j] = p;
        local_score[j] = p + (has_bias ? bias[e] : 0.0f);
        sprob[row_in_block][e] = p;
        prob[e] = p;
    }
    __syncwarp();

    if (hash_mode) {
        if (lane == 0) {
            int32_t tok = tokens ? tokens[t] : token_scalar;
            if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
            const int32_t *row = hash + (uint64_t)tok * 6u;
            float sum = 0.0f;
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) {
                const int32_t e = row[j];
                sel[j] = e;
                const float v = (e >= 0 && e < 256) ? sprob[row_in_block][(uint32_t)e] : 0.0f;
                w[j] = v;
                sum += v;
            }
            sum = fmaxf(sum, 6.103515625e-5f);
            #pragma unroll
            for (uint32_t j = 0; j < 6u; j++) w[j] = w[j] / sum * 1.5f;
        }
        return;
    }

    float out_prob[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t out_idx[6] = {0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (uint32_t k = 0; k < 6u; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            const float s = local_score[j];
            if (router_score_better(s, e, best_score, best_idx)) {
                best_score = s;
                best_prob = local_prob[j];
                best_idx = e;
            }
        }
        #pragma unroll
        for (uint32_t mask = 16u; mask > 0u; mask >>= 1u) {
            const float other_score = __shfl_xor_sync(0xffffffffu, best_score, mask);
            const float other_prob = __shfl_xor_sync(0xffffffffu, best_prob, mask);
            const uint32_t other_idx = __shfl_xor_sync(0xffffffffu, best_idx, mask);
            if (router_score_better(other_score, other_idx, best_score, best_idx)) {
                best_score = other_score;
                best_prob = other_prob;
                best_idx = other_idx;
            }
        }
        #pragma unroll
        for (uint32_t j = 0; j < 8u; j++) {
            const uint32_t e = lane + j * 32u;
            if (e == best_idx) local_score[j] = -INFINITY;
        }
        if (lane == 0) {
            out_idx[k] = best_idx;
            out_prob[k] = best_prob;
        }
    }

    if (lane == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) {
            sel[j] = (int32_t)out_idx[j];
            w[j] = out_prob[j];
            sum += out_prob[j];
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        #pragma unroll
        for (uint32_t j = 0; j < 6u; j++) w[j] = w[j] / sum * 1.5f;
    }
}

__global__ static void swiglu_kernel(float *out, const float *gate, const float *up, uint32_t n, float clamp, float weight) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    if (clamp > 1.0e-6f) {
        g = fminf(g, clamp);
        u = fminf(fmaxf(u, -clamp), clamp);
    }
    float s = g / (1.0f + expf(-g));
    out[i] = s * u * weight;
}

__global__ static void add_kernel(float *out, const float *a, const float *b, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + b[i];
}

__global__ static void directional_steering_project_kernel(
        float       *x,
        const float *directions,
        uint32_t     layer,
        uint32_t     width,
        uint32_t     rows,
        float        scale) {
    const uint32_t row = blockIdx.x;
    if (row >= rows || width == 0) return;

    float *xr = x + (uint64_t)row * width;
    const float *dir = directions + (uint64_t)layer * width;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        sum += xr[i] * dir[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }

    const float coeff = scale * partial[0];
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        xr[i] -= coeff * dir[i];
    }
}

__global__ static void zero_kernel(float *out, uint64_t n) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 0.0f;
}

__global__ static void indexer_scores_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
    uint32_t c = blockIdx.x;
    uint32_t t = blockIdx.y;
    if (c >= n_comp || t >= n_tokens) return;
    if (causal) {
        uint32_t n_visible = (pos0 + t + 1u) / ratio;
        if (c >= n_visible) {
            if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = -INFINITY;
            return;
        }
    }
    float total = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
        const float *kh = index_comp + (uint64_t)c * head_dim;
        float dot = 0.0f;
        for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) dot += qh[d] * kh[d];
        __shared__ float partial[256];
        partial[threadIdx.x] = dot;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        total += fmaxf(partial[0], 0.0f) * weights[(uint64_t)t * n_head + h];
        __syncthreads();
    }
    if (threadIdx.x == 0) scores[(uint64_t)t * n_comp + c] = total * scale;
}

__global__ static void indexer_score_one_direct_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t pos0,
        uint32_t ratio,
        float scale,
        int causal) {
    const uint32_t c = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (c >= n_comp || tid >= 128u) return;
    if (causal) {
        const uint32_t visible = ratio ? (pos0 + 1u) / ratio : n_comp;
        if (c >= visible) {
            if (tid == 0) scores[c] = -INFINITY;
            return;
        }
    }

    __shared__ float krow[128];
    __shared__ float partial[4];
    if (tid < 128u) krow[tid] = index_comp[(uint64_t)c * 128u + tid];
    __syncthreads();

    float total = 0.0f;
    for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
        const uint32_t h = h0 + warp;
        const float4 qv = ((const float4 *)(q + (uint64_t)h * 128u))[lane];
        const float4 kv = ((const float4 *)krow)[lane];
        float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        dot = warp_sum_f32(dot);
        if (lane == 0) partial[warp] = fmaxf(dot, 0.0f) * weights[h] * scale;
        __syncthreads();
        if (tid == 0) total += partial[0] + partial[1] + partial[2] + partial[3];
        __syncthreads();
    }
    if (tid == 0) scores[c] = total;
}

__global__ static void indexer_scores_wmma_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 16u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    if (tid >= 32u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 16u; i += 32u) {
                const uint32_t r = i >> 4u;
                const uint32_t c = i & 15u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[16 * 128];
    __shared__ float c_sh[16 * 16];
    __shared__ float acc_sh[16 * 16];

    for (uint32_t i = tid; i < 16u * 16u; i += 32u) acc_sh[i] = 0.0f;
    for (uint32_t i = tid; i < 16u * 128u; i += 32u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 32u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        for (uint32_t i = tid; i < 16u * 16u; i += 32u) {
            const uint32_t r = i >> 4u;
            const uint32_t token = tile_t + r;
            if (token < n_tokens) {
                const float w = weights[(uint64_t)token * n_head + h];
                acc_sh[i] += fmaxf(c_sh[i], 0.0f) * w;
            }
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < 16u * 16u; i += 32u) {
        const uint32_t r = i >> 4u;
        const uint32_t c = i & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc_sh[i] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}

__global__ static void indexer_scores_wmma32_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 32u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 64u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 32u; i += 64u) {
                const uint32_t r = i >> 5u;
                const uint32_t c = i & 31u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[32 * 128];
    __shared__ float c_sh[2 * 16 * 16];
    __shared__ float acc_sh[2 * 16 * 16];

    for (uint32_t i = tid; i < 2u * 16u * 16u; i += 64u) acc_sh[i] = 0.0f;
    for (uint32_t i = tid; i < 32u * 128u; i += 64u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 64u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 128u + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh + warp * 16u * 16u, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        for (uint32_t i = tid; i < 2u * 16u * 16u; i += 64u) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t r = local >> 4u;
            const uint32_t c = local & 15u;
            const uint32_t token = tile_t + r;
            const uint32_t comp = tile_c + wtile * 16u + c;
            if (token < n_tokens && comp < n_comp) {
                const float w = weights[(uint64_t)token * n_head + h];
                acc_sh[i] += fmaxf(c_sh[i], 0.0f) * w;
            }
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < 2u * 16u * 16u; i += 64u) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t r = local >> 4u;
        const uint32_t c = local & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + wtile * 16u + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc_sh[i] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}

__global__ static void indexer_scores_wmma64_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 64u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 128u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 64u; i += 128u) {
                const uint32_t r = i >> 6u;
                const uint32_t c = i & 63u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[64 * 128];
    __shared__ float c_sh[4 * 16 * 16];
    __shared__ float acc_sh[4 * 16 * 16];

    for (uint32_t i = tid; i < 4u * 16u * 16u; i += 128u) acc_sh[i] = 0.0f;
    for (uint32_t i = tid; i < 64u * 128u; i += 128u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 128u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 128u + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh + warp * 16u * 16u, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        for (uint32_t i = tid; i < 4u * 16u * 16u; i += 128u) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t r = local >> 4u;
            const uint32_t c = local & 15u;
            const uint32_t token = tile_t + r;
            const uint32_t comp = tile_c + wtile * 16u + c;
            if (token < n_tokens && comp < n_comp) {
                const float w = weights[(uint64_t)token * n_head + h];
                acc_sh[i] += fmaxf(c_sh[i], 0.0f) * w;
            }
        }
        __syncthreads();
    }

    for (uint32_t i = tid; i < 4u * 16u * 16u; i += 128u) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t r = local >> 4u;
        const uint32_t c = local & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + wtile * 16u + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc_sh[i] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}

__global__ static void indexer_scores_wmma128_kernel(
        float *scores,
        const float *q,
        const float *weights,
        const float *index_comp,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t pos0,
        uint32_t n_head,
        uint32_t head_dim,
        uint32_t ratio,
        float scale,
        int causal) {
#if __CUDA_ARCH__ >= 700
    namespace wmma = nvcuda::wmma;
    const uint32_t tile_c = blockIdx.x * 128u;
    const uint32_t tile_t = blockIdx.y * 16u;
    const uint32_t tid = threadIdx.x;
    const uint32_t warp = tid >> 5u;
    if (tid >= 256u || head_dim != 128u) return;

    if (causal) {
        const uint32_t last_token = min(tile_t + 16u, n_tokens);
        const uint32_t max_visible = last_token > tile_t
            ? min((pos0 + last_token) / ratio, n_comp)
            : 0u;
        if (tile_c >= max_visible) {
            for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
                const uint32_t r = i >> 7u;
                const uint32_t c = i & 127u;
                const uint32_t token = tile_t + r;
                const uint32_t comp = tile_c + c;
                if (token < n_tokens && comp < n_comp) {
                    scores[(uint64_t)token * n_comp + comp] = -INFINITY;
                }
            }
            return;
        }
    }

    __shared__ __half a_sh[16 * 128];
    __shared__ __half b_sh[128 * 128];
    __shared__ float c_sh[8 * 16 * 16];

    float acc[8];
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) acc[i] = 0.0f;

    for (uint32_t i = tid; i < 128u * 128u; i += 256u) {
        const uint32_t c = i >> 7u;
        const uint32_t d = i & 127u;
        const uint32_t comp = tile_c + c;
        float v = 0.0f;
        if (comp < n_comp) v = index_comp[(uint64_t)comp * head_dim + d];
        b_sh[d + c * 128u] = __float2half(v);
    }
    __syncthreads();

    for (uint32_t h = 0; h < n_head; h++) {
        for (uint32_t i = tid; i < 16u * 128u; i += 256u) {
            const uint32_t r = i >> 7u;
            const uint32_t d = i & 127u;
            const uint32_t token = tile_t + r;
            float v = 0.0f;
            if (token < n_tokens) {
                v = q[((uint64_t)token * n_head + h) * head_dim + d];
            }
            a_sh[i] = __float2half(v);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);
        const uint32_t col0 = warp * 16u;
        for (uint32_t k0 = 0; k0 < 128u; k0 += 16u) {
            wmma::load_matrix_sync(a_frag, a_sh + k0, 128);
            wmma::load_matrix_sync(b_frag, b_sh + col0 * 128u + k0, 128);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        wmma::store_matrix_sync(c_sh + warp * 16u * 16u, c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        const uint32_t local0 = tid & 255u;
        const uint32_t token0 = tile_t + (local0 >> 4u);
        const float w0 = token0 < n_tokens ? weights[(uint64_t)token0 * n_head + h] : 0.0f;
        uint32_t slot = 0;
        for (uint32_t i = tid; i < 8u * 16u * 16u; i += 256u, slot++) {
            const uint32_t wtile = i >> 8u;
            const uint32_t local = i & 255u;
            const uint32_t r = local >> 4u;
            const uint32_t c = local & 15u;
            const uint32_t token = tile_t + r;
            const uint32_t comp = tile_c + wtile * 16u + c;
            if (token < n_tokens && comp < n_comp) {
                acc[slot] += fmaxf(c_sh[i], 0.0f) * w0;
            }
        }
        __syncthreads();
    }

    uint32_t slot = 0;
    for (uint32_t i = tid; i < 8u * 16u * 16u; i += 256u, slot++) {
        const uint32_t wtile = i >> 8u;
        const uint32_t local = i & 255u;
        const uint32_t r = local >> 4u;
        const uint32_t c = local & 15u;
        const uint32_t token = tile_t + r;
        const uint32_t comp = tile_c + wtile * 16u + c;
        if (token < n_tokens && comp < n_comp) {
            float out = acc[slot] * scale;
            if (causal) {
                const uint32_t visible = (pos0 + token + 1u) / ratio;
                if (comp >= visible) out = -INFINITY;
            }
            scores[(uint64_t)token * n_comp + comp] = out;
        }
    }
#endif
}

/* Single-block argmax over n_vocab F32 logits. One block of 1024 threads
 * cooperatively scans the vocab, tracking a (best_v, best_idx) pair per
 * thread, then reduces in shared memory with value-keyed comparison.
 *
 * Tie-breaking: lower index wins, matching the host sample_argmax used by
 * the CPU reference path. Replaces the indexer-as-argmax workaround used
 * in the MTP top-id sites, which fell through to the legacy single-thread
 * indexer_topk_kernel at top_k=1, costing ~17.5 ms per call on n_vocab=129280. */
__global__ static void argmax_kernel(int32_t *out_idx, const float *logits, uint32_t n_vocab) {
    enum { THREADS = 1024 };
    __shared__ float sm_val[THREADS];
    __shared__ int32_t sm_idx[THREADS];

    const uint32_t tid = threadIdx.x;
    float local_v = -INFINITY;
    int32_t local_i = 0;
    for (uint32_t i = tid; i < n_vocab; i += THREADS) {
        const float v = logits[i];
        if (v > local_v) {
            local_v = v;
            local_i = (int32_t)i;
        }
    }
    sm_val[tid] = local_v;
    sm_idx[tid] = local_i;
    __syncthreads();

    for (uint32_t s = THREADS / 2u; s > 0u; s >>= 1) {
        if (tid < s) {
            const float vr = sm_val[tid + s];
            const int32_t ir = sm_idx[tid + s];
            const float vl = sm_val[tid];
            const int32_t il = sm_idx[tid];
            /* Larger value wins; on exact ties prefer the lower index. */
            const bool take_right = (vr > vl) || (vr == vl && ir < il);
            if (take_right) {
                sm_val[tid] = vr;
                sm_idx[tid] = ir;
            }
        }
        __syncthreads();
    }

    if (tid == 0) *out_idx = sm_idx[0];
}

__global__ static void indexer_topk_kernel(uint32_t *selected, const float *scores, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    uint32_t t = blockIdx.x;
    if (t >= n_tokens || threadIdx.x != 0) return;
    const float *row = scores + (uint64_t)t * n_comp;
    uint32_t *sel = selected + (uint64_t)t * top_k;
    for (uint32_t k = 0; k < top_k; k++) sel[k] = 0;
    for (uint32_t c = 0; c < n_comp; c++) {
        float v = row[c];
        for (uint32_t k = 0; k < top_k; k++) {
            if ((k >= c) || v > row[sel[k]]) {
                for (uint32_t j = top_k - 1; j > k; j--) sel[j] = sel[j - 1];
                sel[k] = c;
                break;
            }
        }
    }
}

__device__ __forceinline__ static bool topk_score_better(float av, uint32_t ai, float bv, uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__device__ __forceinline__ static uint32_t topk_float_ordered_key(float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}

__device__ __forceinline__ static uint64_t topk_pack_key(float v, uint32_t idx) {
    return ((uint64_t)topk_float_ordered_key(v) << 32u) | (uint64_t)(0xffffffffu - idx);
}

__global__ static void indexer_topk_8192_cub_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    constexpr uint32_t BLOCK_THREADS = 512u;
    constexpr uint32_t ITEMS_PER_THREAD = 16u;
    using BlockSort = cub::BlockRadixSort<uint64_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
    extern __shared__ __align__(16) unsigned char sort_smem[];
    typename BlockSort::TempStorage &sort_storage =
        *reinterpret_cast<typename BlockSort::TempStorage *>(sort_smem);

    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= BLOCK_THREADS) return;

    const float *row = scores + (uint64_t)t * n_comp;
    uint64_t keys[ITEMS_PER_THREAD];
#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < n_comp) {
            keys[item] = topk_pack_key(row[i], i);
        } else {
            keys[item] = topk_pack_key(-INFINITY, UINT32_MAX);
        }
    }

    BlockSort(sort_storage).SortDescending(keys);

#pragma unroll
    for (uint32_t item = 0; item < ITEMS_PER_THREAD; item++) {
        const uint32_t i = tid * ITEMS_PER_THREAD + item;
        if (i < top_k) {
            selected[(uint64_t)t * top_k + i] = 0xffffffffu - (uint32_t)keys[item];
        }
    }
}

__global__ static void indexer_topk_1024_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 1024u) return;
    __shared__ float vals[1024];
    __shared__ uint32_t idxs[1024];

    const float *row = scores + (uint64_t)t * n_comp;
    if (tid < n_comp) {
        vals[tid] = row[tid];
        idxs[tid] = tid;
    } else {
        vals[tid] = -INFINITY;
        idxs[tid] = UINT32_MAX;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= 1024u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            uint32_t other = tid ^ j;
            if (other > tid && other < 1024u) {
                const float av = vals[tid];
                const float bv = vals[other];
                const uint32_t ai = idxs[tid];
                const uint32_t bi = idxs[other];
                const bool desc_half = (tid & k) == 0u;
                const bool swap = desc_half
                    ? topk_score_better(bv, bi, av, ai)
                    : topk_score_better(av, ai, bv, bi);
                if (swap) {
                    vals[tid] = bv;
                    idxs[tid] = bi;
                    vals[other] = av;
                    idxs[other] = ai;
                }
            }
            __syncthreads();
        }
    }

    if (tid < top_k) selected[(uint64_t)t * top_k + tid] = idxs[tid];
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_comp) {
            vals[i] = row[i];
            idxs[i] = i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_pow2_u16_kernel(
        uint32_t *selected,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint16_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < n_comp) {
            vals[i] = row[i];
            idxs[i] = (uint16_t)i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT16_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = (uint16_t)bi;
                        vals[other] = av;
                        idxs[other] = (uint16_t)ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_chunk_pow2_kernel(
        uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t chunk = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;

    const uint32_t chunk_start = chunk * SORT_N;
    if (chunk_start >= n_comp) return;
    const uint32_t chunk_n = n_comp - chunk_start < SORT_N ? n_comp - chunk_start : SORT_N;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        if (i < chunk_n) {
            vals[i] = row[chunk_start + i];
            idxs[i] = chunk_start + i;
        } else {
            vals[i] = -INFINITY;
            idxs[i] = UINT32_MAX;
        }
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    uint32_t *out = candidates + (uint64_t)t * candidate_stride + chunk * top_k;
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        out[i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_merge_pow2_kernel(
        uint32_t *selected,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t candidate_count,
        uint32_t candidate_stride) {
    uint32_t t = blockIdx.x;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;
    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        selected[(uint64_t)t * top_k + i] = idxs[i];
    }
}

template <uint32_t SORT_N>
__global__ static void indexer_topk_tree_merge_pow2_kernel(
        uint32_t *out,
        const uint32_t *candidates,
        const float *scores,
        uint32_t n_comp,
        uint32_t n_tokens,
        uint32_t top_k,
        uint32_t n_sets,
        uint32_t merge_group,
        uint32_t candidate_stride,
        uint32_t out_stride) {
    uint32_t t = blockIdx.x;
    uint32_t group = blockIdx.y;
    uint32_t tid = threadIdx.x;
    if (t >= n_tokens) return;

    const uint32_t set0 = group * merge_group;
    if (set0 >= n_sets) return;
    uint32_t set_count = n_sets - set0;
    if (set_count > merge_group) set_count = merge_group;
    const uint32_t candidate_count = set_count * top_k;

    __shared__ float vals[SORT_N];
    __shared__ uint32_t idxs[SORT_N];

    const float *row = scores + (uint64_t)t * n_comp;
    const uint32_t *cand = candidates + (uint64_t)t * candidate_stride + set0 * top_k;
    for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
        uint32_t idx = UINT32_MAX;
        float v = -INFINITY;
        if (i < candidate_count) {
            idx = cand[i];
            if (idx < n_comp) v = row[idx];
        }
        vals[i] = v;
        idxs[i] = idx;
    }
    __syncthreads();

    for (uint32_t k = 2u; k <= SORT_N; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint32_t i = tid; i < SORT_N; i += blockDim.x) {
                uint32_t other = i ^ j;
                if (other > i && other < SORT_N) {
                    const float av = vals[i];
                    const float bv = vals[other];
                    const uint32_t ai = idxs[i];
                    const uint32_t bi = idxs[other];
                    const bool desc_half = (i & k) == 0u;
                    const bool swap = desc_half
                        ? topk_score_better(bv, bi, av, ai)
                        : topk_score_better(av, ai, bv, bi);
                    if (swap) {
                        vals[i] = bv;
                        idxs[i] = bi;
                        vals[other] = av;
                        idxs[other] = ai;
                    }
                }
            }
            __syncthreads();
        }
    }

    uint32_t *dst = out + (uint64_t)t * out_stride + group * top_k;
    for (uint32_t i = tid; i < top_k; i += blockDim.x) {
        dst[i] = idxs[i];
    }
}

__global__ static void indexed_topk_sort_512_asc_kernel(
        int32_t *dst,
        const int32_t *src,
        uint32_t n_tokens) {
    const uint32_t t = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (t >= n_tokens || tid >= 512u) return;
    __shared__ int32_t rows[512];

    const int32_t *src_row = src + (uint64_t)t * 512u;
    int32_t *dst_row = dst + (uint64_t)t * 512u;
    rows[tid] = src_row[tid];
    __syncthreads();

    for (uint32_t k = 2u; k <= 512u; k <<= 1u) {
        for (uint32_t j = k >> 1u; j > 0u; j >>= 1u) {
            const uint32_t other = tid ^ j;
            if (other > tid && other < 512u) {
                const int32_t a = rows[tid];
                const int32_t b = rows[other];
                const bool up = (tid & k) == 0u;
                if ((up && a > b) || (!up && a < b)) {
                    rows[tid] = b;
                    rows[other] = a;
                }
            }
            __syncthreads();
        }
    }

    dst_row[tid] = rows[tid];
}

__global__ static void topk_mask_kernel(float *mask, const uint32_t *topk, uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_comp;
    if (gid >= n) return;
    uint32_t t = gid / n_comp;
    uint32_t c = gid - (uint64_t)t * n_comp;
    float v = -INFINITY;
    for (uint32_t k = 0; k < top_k; k++) {
        if (topk[(uint64_t)t * top_k + k] == c) {
            v = 0.0f;
            break;
        }
    }
    mask[gid] = v;
}

int ds4_gpu_embed_token_hc_tensor(ds4_gpu_tensor *out_hc, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n_vocab, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    (void)n_vocab;
    if (!out_hc || !model_map || weight_offset >= model_size) return 0;
    uint64_t weight_bytes = (uint64_t)n_vocab * n_embd * sizeof(uint16_t);
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "token_embd");
    if (!wptr) return 0;
    uint32_t n = n_embd * n_hc;
    if (g_tok_id_dev && g_tok_id_host) {
        /* 间接路径: capture 期只暂存 token(直跑时立即写并即刻消费, 亦安全 ——
         * 写在 memcpy 入队前, 同线程同流串行) */
        cudaStreamCaptureStatus ecs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(cudaStreamPerThread, &ecs);
        g_tok_id_want = (int32_t)token;
        if (ecs == cudaStreamCaptureStatusNone) *g_tok_id_host = (int32_t)token;
        if (cudaMemcpyAsync(g_tok_id_dev, g_tok_id_host, sizeof(int32_t),
                            cudaMemcpyHostToDevice, cudaStreamPerThread) != cudaSuccess)
            return cuda_ok(cudaGetLastError(), "embed token id copy");
        embed_token_hc_dev_kernel<<<(n + 255) / 256, 256>>>((float *)out_hc->ptr, (const unsigned short *)wptr, g_tok_id_dev, n_vocab, n_embd, n_hc);
        return cuda_ok(cudaGetLastError(), "embed token dev launch");
    }
    embed_token_hc_kernel<<<(n + 255) / 256, 256>>>((float *)out_hc->ptr, (const unsigned short *)wptr, token, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed token launch");
}

int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens_t,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !tokens_t || !model_map ||
        weight_offset > model_size ||
        (uint64_t)n_vocab * n_embd * sizeof(uint16_t) > model_size - weight_offset ||
        tokens_t->bytes < (uint64_t)n_tokens * sizeof(int32_t) ||
        out_hc->bytes < (uint64_t)n_tokens * n_hc * n_embd * sizeof(float)) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset,
                                            (uint64_t)n_vocab * n_embd * sizeof(uint16_t),
                                            "token_embd");
    if (!wptr) return 0;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    embed_tokens_hc_kernel<<<(n + 255) / 256, 256>>>(
        (float *)out_hc->ptr,
        (const int32_t *)tokens_t->ptr,
        (const __half *)wptr,
        n_vocab, n_tokens, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "embed tokens launch");
}

static int indexer_scores_launch(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale,
        uint32_t                causal) {
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float) ||
        index_comp->bytes < (uint64_t)n_comp * head_dim * sizeof(float) ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)) {
        return 0;
    }
    if (causal && ratio == 0) return 0;
    if (n_tokens == 1u && head_dim == 128u && n_head == 64u &&
        1) {
        indexer_score_one_direct_kernel<<<n_comp, 128>>>((float *)scores->ptr,
                                                         (const float *)q->ptr,
                                                         (const float *)weights->ptr,
                                                         (const float *)index_comp->ptr,
                                                         n_comp, pos0, ratio,
                                                         scale, causal ? 1 : 0);
        return cuda_ok(cudaGetLastError(), "indexer score one direct launch");
    }
    if (!g_quality_mode && head_dim == 128u && n_head == 64u &&
        1) {
        if (1) {
            dim3 grid((n_comp + 127u) / 128u, (n_tokens + 15u) / 16u, 1);
            indexer_scores_wmma128_kernel<<<grid, 256>>>((float *)scores->ptr,
                                                         (const float *)q->ptr,
                                                         (const float *)weights->ptr,
                                                         (const float *)index_comp->ptr,
                                                         n_comp, n_tokens, pos0, n_head,
                                                         head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma128 launch");
        } else if (1) {
            dim3 grid((n_comp + 63u) / 64u, (n_tokens + 15u) / 16u, 1);
            indexer_scores_wmma64_kernel<<<grid, 128>>>((float *)scores->ptr,
                                                        (const float *)q->ptr,
                                                        (const float *)weights->ptr,
                                                        (const float *)index_comp->ptr,
                                                        n_comp, n_tokens, pos0, n_head,
                                                        head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma64 launch");
        } else if (1) {
            dim3 grid((n_comp + 31u) / 32u, (n_tokens + 15u) / 16u, 1);
            indexer_scores_wmma32_kernel<<<grid, 64>>>((float *)scores->ptr,
                                                       (const float *)q->ptr,
                                                       (const float *)weights->ptr,
                                                       (const float *)index_comp->ptr,
                                                       n_comp, n_tokens, pos0, n_head,
                                                       head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma32 launch");
        } else {
            dim3 grid((n_comp + 15u) / 16u, (n_tokens + 15u) / 16u, 1);
            indexer_scores_wmma_kernel<<<grid, 32>>>((float *)scores->ptr,
                                                     (const float *)q->ptr,
                                                     (const float *)weights->ptr,
                                                     (const float *)index_comp->ptr,
                                                     n_comp, n_tokens, pos0, n_head,
                                                     head_dim, ratio, scale, causal ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "indexer scores wmma launch");
        }
    }
    dim3 grid(n_comp, n_tokens, 1);
    indexer_scores_kernel<<<grid, 256>>>((float *)scores->ptr,
                                         (const float *)q->ptr,
                                         (const float *)weights->ptr,
                                         (const float *)index_comp->ptr,
                                         n_comp, n_tokens, pos0, n_head,
                                         head_dim, ratio, scale, causal ? 1 : 0);
    return cuda_ok(cudaGetLastError(), "indexer scores launch");
}

int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, 1, 0,
                                 n_head, head_dim, 1, scale, 0);
}

int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, 0,
                                 n_head, head_dim, ratio, scale, 1);
}

int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, pos0,
                                 n_head, head_dim, ratio, scale, 1);
}

int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        top_k > n_comp ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    if (top_k == 512u && n_comp <= 1024u &&
        1) {
        indexer_topk_1024_kernel<<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                     (const float *)scores->ptr,
                                                     n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 1024 launch");
    }
    if (top_k == 512u && n_comp <= 2048u &&
        1) {
        indexer_topk_pow2_kernel<2048><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 2048 launch");
    }
    if (top_k == 512u && n_comp <= 4096u &&
        1) {
        if (n_comp == 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 4096 cub launch");
                }
            }
        }
        indexer_topk_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                           (const float *)scores->ptr,
                                                           n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 4096 launch");
    }
    if (top_k == 512u && n_comp <= 8192u &&
        1 &&
        1) {
        if (n_comp > 4096u) {
            using TopkCubSort = cub::BlockRadixSort<uint64_t, 512, 16>;
            const int smem = (int)sizeof(typename TopkCubSort::TempStorage);
            int dev = 0;
            int max_optin_smem = 0;
            cudaError_t attr_err = cudaGetDevice(&dev);
            if (attr_err == cudaSuccess) {
                attr_err = cudaDeviceGetAttribute(&max_optin_smem,
                                                  cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                                  dev);
            }
            if (attr_err == cudaSuccess && max_optin_smem >= smem) {
                attr_err = cudaFuncSetAttribute(indexer_topk_8192_cub_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                smem);
                if (attr_err == cudaSuccess) {
                    indexer_topk_8192_cub_kernel<<<n_tokens, 512, (size_t)smem>>>((uint32_t *)selected->ptr,
                                                                                 (const float *)scores->ptr,
                                                                                 n_comp, n_tokens, top_k);
                    return cuda_ok(cudaGetLastError(), "indexer topk 8192 cub launch");
                }
            }
        }
        indexer_topk_pow2_u16_kernel<8192><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                               (const float *)scores->ptr,
                                                               n_comp, n_tokens, top_k);
        return cuda_ok(cudaGetLastError(), "indexer topk 8192 launch");
    }
    if (top_k == 512u && 1 &&
        1) {
        const uint32_t chunk_n = 4096u;
        const uint32_t n_chunks = (n_comp + chunk_n - 1u) / chunk_n;
        const uint32_t candidate_stride = n_chunks * top_k;
        uint32_t n_sets = n_chunks;
        uint64_t scratch_u32_per_token = candidate_stride;
        while (n_sets > DS4_CUDA_TOPK_MERGE_GROUP) {
            n_sets = (n_sets + DS4_CUDA_TOPK_MERGE_GROUP - 1u) / DS4_CUDA_TOPK_MERGE_GROUP;
            scratch_u32_per_token += (uint64_t)n_sets * top_k;
        }
        if (scratch_u32_per_token > UINT64_MAX / n_tokens / sizeof(uint32_t)) return 0;
        const uint64_t tmp_bytes = (uint64_t)n_tokens * scratch_u32_per_token * sizeof(uint32_t);
        uint32_t *scratch = (uint32_t *)cuda_tmp_alloc(tmp_bytes, "indexer topk tree");
        if (!scratch) return 0;

        uint32_t *cur = scratch;
        n_sets = n_chunks;
        uint32_t cur_stride = candidate_stride;
        dim3 grid_chunks(n_tokens, n_chunks, 1);
        indexer_topk_chunk_pow2_kernel<4096><<<grid_chunks, 1024>>>(cur,
                                                                    (const float *)scores->ptr,
                                                                    n_comp,
                                                                    n_tokens,
                                                                    top_k,
                                                                    candidate_stride);
        if (!cuda_ok(cudaGetLastError(), "indexer topk chunk launch")) return 0;

        while (n_sets > DS4_CUDA_TOPK_MERGE_GROUP) {
            const uint32_t next_sets = (n_sets + DS4_CUDA_TOPK_MERGE_GROUP - 1u) / DS4_CUDA_TOPK_MERGE_GROUP;
            const uint32_t next_stride = next_sets * top_k;
            uint32_t *next = cur + (uint64_t)n_tokens * cur_stride;
            dim3 grid_merge(n_tokens, next_sets, 1);
            indexer_topk_tree_merge_pow2_kernel<4096><<<grid_merge, 1024>>>(
                    next,
                    cur,
                    (const float *)scores->ptr,
                    n_comp,
                    n_tokens,
                    top_k,
                    n_sets,
                    DS4_CUDA_TOPK_MERGE_GROUP,
                    cur_stride,
                    next_stride);
            if (!cuda_ok(cudaGetLastError(), "indexer topk tree merge launch")) return 0;
            cur = next;
            n_sets = next_sets;
            cur_stride = next_stride;
        }

        indexer_topk_merge_pow2_kernel<4096><<<n_tokens, 1024>>>((uint32_t *)selected->ptr,
                                                                 cur,
                                                                 (const float *)scores->ptr,
                                                                 n_comp,
                                                                 n_tokens,
                                                                 top_k,
                                                                 n_sets * top_k,
                                                                 cur_stride);
        return cuda_ok(cudaGetLastError(), "indexer topk tree final launch");
    }
    indexer_topk_kernel<<<n_tokens, 1>>>((uint32_t *)selected->ptr,
                                         (const float *)scores->ptr,
                                         n_comp, n_tokens, top_k);
    return cuda_ok(cudaGetLastError(), "indexer topk launch");
}

int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t                n_vocab) {
    if (!out_idx || !logits || n_vocab == 0 ||
        out_idx->bytes < sizeof(int32_t) ||
        logits->bytes < (uint64_t)n_vocab * sizeof(float)) {
        return 0;
    }
    argmax_kernel<<<1, 1024>>>((int32_t *)out_idx->ptr,
                               (const float *)logits->ptr,
                               n_vocab);
    return cuda_ok(cudaGetLastError(), "argmax launch");
}

int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) {
        return 0;
    }
    uint64_t n = (uint64_t)n_tokens * n_comp;
    uint64_t nk = (uint64_t)n_tokens * top_k;
    uint64_t blocks = ((n > nk ? n : nk) + 255) / 256;
    topk_mask_kernel<<<blocks, 256>>>((float *)mask->ptr,
                                      (const uint32_t *)topk->ptr,
                                      n_comp, n_tokens, top_k);
    return cuda_ok(cudaGetLastError(), "topk mask launch");
}
static int cuda_matmul_q8_0_tensor_labeled(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok, const char *label) {
    if (!out || !x || !model_map) return 0;
    uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34)) return 0;
    uint64_t weight_bytes = out_dim * blocks * 34;
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "q8_0");
    if (!wptr) return 0;
    if (g_cublas_ready && n_tok > 1) {
        const float *w_f32 = cuda_q8_f32_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
        if (w_f32) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasSgemm(g_cublas,
                                            CUBLAS_OP_T,
                                            CUBLAS_OP_N,
                                            (int)out_dim,
                                            (int)n_tok,
                                            (int)in_dim,
                                            &alpha,
                                            w_f32,
                                            (int)in_dim,
                                            (const float *)x->ptr,
                                            (int)in_dim,
                                            &beta,
                                            (float *)out->ptr,
                                            (int)out_dim);
            return cublas_ok(st, "q8 fp32 matmul");
        }
        const __half *w_f16 = cuda_q8_f16_ptr(model_map, weight_offset, weight_bytes, in_dim, out_dim, label);
        if (w_f16) {
            const uint64_t xh_count = n_tok * in_dim;
            __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "q8 f16 gemm activations");
            if (!xh) return 0;
            f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(xh, (const float *)x->ptr, xh_count);
            if (!cuda_ok(cudaGetLastError(), "q8 f16 activation convert launch")) return 0;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasGemmEx(g_cublas,
                                             CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             (int)out_dim,
                                             (int)n_tok,
                                             (int)in_dim,
                                             &alpha,
                                             w_f16,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             xh,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             &beta,
                                             out->ptr,
                                             CUDA_R_32F,
                                             (int)out_dim,
                                             CUDA_R_32F,
                                             CUBLAS_GEMM_DEFAULT);
            if (st == CUBLAS_STATUS_SUCCESS) return 1;
            fprintf(stderr, "ds4: cuBLAS q8 f16 matmul failed: status %d\n", (int)st);
            cuda_q8_f16_cache_disable_after_failure("cuBLAS f16 matmul failure",
                                                    in_dim * out_dim * sizeof(__half));
            /* The F16 expansion cache is only an optimization.  If cuBLAS
             * rejects the cached path under memory pressure, retry the same
             * operation through the native Q8 kernels below. */
        }
    }
    /* (2026-08-19 实验记录: decode 单 token 换 f16 rowblock gemv 实测 12.3→11.3
     * 反慢 —— f16 权重字节是 q8 的 4x, q8 preq kernel 本就贴近字节墙。撤。) */
    const uint64_t xq_bytes = n_tok * blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + n_tok * blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks, (unsigned)n_tok, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 quantize launch")) return 0;
    if (n_tok == 1) {
        /* in_dim 是 32 的倍数(本模型全部 q8 形状)时才走 repack 路(int4 整块读) */
        if ((in_dim & 31u) == 0u) {
            const cuda_q8r_entry *re = cuda_q8r_get(model_map, weight_offset, out_dim, blocks);
            if (re) {
                /* (splitK 2行×4段版实测 14.7→14.3 负优化: 段仅 32 块/warp, 归约与
                 * xq 重读开销盖过并行收益 — kernel 保留归档, dispatch 走整行版) */
                matmul_q8r_warp8_kernel<<<((unsigned)out_dim + 15u) / 16u, 256>>>(
                        (float *)out->ptr, re->scales, re->qs, xq, xscale, out_dim, blocks);
                return cuda_ok(cudaGetLastError(), "matmul_q8r launch");
            }
        }
        matmul_q8_0_preq_warp8_kernel<<<((unsigned)out_dim + 15u) / 16u, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 warp launch");
    }
    if (1 && blocks <= 32u) {
        dim3 bgrid(((unsigned)out_dim + 7u) / 8u, (unsigned)n_tok, 1);
        matmul_q8_0_preq_batch_warp8_kernel<<<bgrid, 256>>>(
                (float *)out->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                n_tok,
                blocks,
                use_dp4a);
        return cuda_ok(cudaGetLastError(), "matmul_q8_0 batch warp launch");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_q8_0_preq_kernel<<<grid, 256>>>((float *)out->ptr,
                                           reinterpret_cast<const unsigned char *>(wptr),
                                           xq,
                                           xscale,
                                           in_dim, out_dim, n_tok, blocks,
                                           use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 launch");
}


int ds4_gpu_matmul_q8_0_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    return cuda_matmul_q8_0_tensor_labeled(out, model_map, model_size, weight_offset,
                                           in_dim, out_dim, x, n_tok, "q8_0");
}

extern "C" int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out0_dim,
        uint64_t out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out0_dim == 0 || out1_dim == 0 || n_tok == 0) {
        return 0;
    }
    if (n_tok != 1) {
        return cuda_matmul_q8_0_tensor_labeled(out0, model_map, model_size, weight0_offset,
                                               in_dim, out0_dim, x, n_tok, "q8_0_pair0") &&
               cuda_matmul_q8_0_tensor_labeled(out1, model_map, model_size, weight1_offset,
                                               in_dim, out1_dim, x, n_tok, "q8_0_pair1");
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight0_offset > model_size || weight1_offset > model_size ||
        out0_dim > UINT64_MAX / (blocks * 34) ||
        out1_dim > UINT64_MAX / (blocks * 34)) {
        return 0;
    }
    const uint64_t weight0_bytes = out0_dim * blocks * 34;
    const uint64_t weight1_bytes = out1_dim * blocks * 34;
    if (weight0_bytes > model_size - weight0_offset ||
        weight1_bytes > model_size - weight1_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out0_dim * sizeof(float) ||
        out1->bytes < out1_dim * sizeof(float)) {
        return 0;
    }
    const char *w0 = cuda_model_range_ptr(model_map, weight0_offset, weight0_bytes, "q8_0_pair0");
    const char *w1 = cuda_model_range_ptr(model_map, weight1_offset, weight1_bytes, "q8_0_pair1");
    if (!w0 || !w1) return 0;

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 pair prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks, 1, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0 pair quantize launch")) return 0;
    const uint64_t max_out = out0_dim > out1_dim ? out0_dim : out1_dim;
    if ((in_dim & 31u) == 0u) {
        const cuda_q8r_entry *r0 = cuda_q8r_get(model_map, weight0_offset, out0_dim, blocks);
        const cuda_q8r_entry *r1 = r0 ? cuda_q8r_get(model_map, weight1_offset, out1_dim, blocks) : NULL;
        if (r0 && r1) {
            matmul_q8r_pair_warp8_kernel<<<((unsigned)max_out + 7u) / 8u, 256>>>(
                    (float *)out0->ptr, (float *)out1->ptr,
                    r0->scales, r0->qs, r1->scales, r1->qs,
                    xq, xscale, out0_dim, out1_dim, blocks);
            return cuda_ok(cudaGetLastError(), "matmul_q8r pair launch");
        }
    }
    matmul_q8_0_pair_preq_warp8_kernel<<<((unsigned)max_out + 7u) / 8u, 256>>>(
            (float *)out0->ptr,
            (float *)out1->ptr,
            reinterpret_cast<const unsigned char *>(w0),
            reinterpret_cast<const unsigned char *>(w1),
            xq,
            xscale,
            in_dim,
            out0_dim,
            out1_dim,
            blocks,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0 pair warp launch");
}

static int cuda_matmul_q8_0_hc_expand_tensor_labeled(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc,
        const char             *label) {
    if (!out_hc || !block_out || !x || !residual_hc || !split || !model_map ||
        in_dim == 0 || out_dim == 0 || n_embd == 0 || n_hc == 0 ||
        out_dim != (uint64_t)n_embd) {
        return 0;
    }
    const uint64_t blocks = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks * 34)) return 0;
    const uint64_t weight_bytes = out_dim * blocks * 34;
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t split_bytes = (uint64_t)(2u * n_hc + n_hc * n_hc) * sizeof(float);
    if (weight_bytes > model_size - weight_offset ||
        x->bytes < in_dim * sizeof(float) ||
        block_out->bytes < out_dim * sizeof(float) ||
        residual_hc->bytes < hc_bytes ||
        split->bytes < split_bytes ||
        out_hc->bytes < hc_bytes ||
        (block_add && block_add->bytes < out_dim * sizeof(float))) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, label ? label : "q8_0_hc_expand");
    if (!wptr) return 0;

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 hc expand prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand quantize launch")) return 0;
    if ((in_dim & 31u) == 0u) {
        const cuda_q8r_entry *re = cuda_q8r_get(model_map, weight_offset, out_dim, blocks);
        if (re) {
            matmul_q8r_hc_expand_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
                    (float *)out_hc->ptr, (float *)block_out->ptr,
                    block_add ? (const float *)block_add->ptr : NULL,
                    (const float *)residual_hc->ptr, (const float *)split->ptr,
                    re->scales, re->qs, xq, xscale,
                    out_dim, n_embd, n_hc, blocks, block_add ? 1 : 0);
            return cuda_ok(cudaGetLastError(), "matmul_q8r hc launch");
        }
    }
    matmul_q8_0_hc_expand_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256>>>(
            (float *)out_hc->ptr,
            (float *)block_out->ptr,
            block_add ? (const float *)block_add->ptr : (const float *)block_out->ptr,
            (const float *)residual_hc->ptr,
            (const float *)split->ptr,
            reinterpret_cast<const unsigned char *>(wptr),
            xq,
            xscale,
            in_dim,
            out_dim,
            n_embd,
            n_hc,
            blocks,
            block_add ? 1 : 0,
            use_dp4a);
    return cuda_ok(cudaGetLastError(), "matmul_q8_0_hc_expand launch");
}

int ds4_gpu_matmul_q4_K_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    /* q4_K 版 attn_output_b + hc expand(与 q8 labeled 版同构, epilogue 逐字同义)。 */
    if (!out_hc || !block_out || !x || !residual_hc || !split || !model_map ||
        in_dim == 0 || out_dim == 0 || n_embd == 0 || n_hc == 0 ||
        out_dim != (uint64_t)n_embd || in_dim % 256u != 0) {
        return 0;
    }
    const uint64_t blocks = in_dim / 32u;
    const uint64_t kblocks = in_dim / 256u;
    const uint64_t weight_bytes = out_dim * kblocks * sizeof(cuda_block_q4_K);
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t split_bytes = (uint64_t)(2u * n_hc + n_hc * n_hc) * sizeof(float);
    if (weight_offset > model_size ||
        weight_bytes > model_size - weight_offset ||
        x->bytes < in_dim * sizeof(float) ||
        block_out->bytes < out_dim * sizeof(float) ||
        residual_hc->bytes < hc_bytes ||
        split->bytes < split_bytes ||
        out_hc->bytes < hc_bytes) {
        return 0;
    }
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "q4k_hc_expand");
    if (!wptr) return 0;
    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q4k hc expand prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32>>>(xq, xscale, (const float *)x->ptr, in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "matmul_q4k_hc_expand quantize launch")) return 0;
    if (use_dp4a && kblocks <= 32u) {
        matmul_q4_K_hc_expand_preq_warp8_dp4a_kernel<<<((unsigned)out_dim + 7u) / 8u, 256, (size_t)8u * (size_t)((kblocks > 16u) ? 16u : kblocks) * 9u * sizeof(uint4)>>>(
                (float *)out_hc->ptr,
                (float *)block_out->ptr,
                (const float *)block_out->ptr,   /* has_add=0, 占位 */
                (const float *)residual_hc->ptr,
                (const float *)split->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                n_embd,
                n_hc,
                kblocks,
                0);
    } else {
        matmul_q4_K_hc_expand_preq_warp8_kernel<<<((unsigned)out_dim + 7u) / 8u, 256, (kblocks <= 32u) ? (size_t)8u * (size_t)((kblocks > 16u) ? 16u : kblocks) * 9u * sizeof(uint4) : 0>>>(
                (float *)out_hc->ptr,
                (float *)block_out->ptr,
                (const float *)block_out->ptr,   /* has_add=0, 占位 */
                (const float *)residual_hc->ptr,
                (const float *)split->ptr,
                reinterpret_cast<const unsigned char *>(wptr),
                xq,
                xscale,
                in_dim,
                out_dim,
                n_embd,
                n_hc,
                kblocks,
                0,
                use_dp4a);
    }
    if (((const char *)0) /* DS4_AO_PROBE: 诊断开关已删(2026-08-22) */) {
        static int onceb = 0;
        if (onceb++ < 96) {
            (void)cudaDeviceSynchronize();
            float bo0 = 0; float *xh = (float *)malloc(in_dim * sizeof(float));
            (void)cudaMemcpy(&bo0, block_out->ptr, 4, cudaMemcpyDeviceToHost);
            (void)cudaMemcpy(xh, x->ptr, in_dim * sizeof(float), cudaMemcpyDeviceToHost);
            const uint8_t *w0 = (const uint8_t *)model_map + weight_offset;
            double ref = 0.0;
            for (uint64_t b = 0; b < kblocks; b++) {
                const uint8_t *blk = w0 + b * 144u;
                uint16_t hd, hm; memcpy(&hd, blk + 0, 2); memcpy(&hm, blk + 2, 2);
                const float d = dev_host_f16(hd), dmin = dev_host_f16(hm);
                const uint8_t *scales = blk + 4; const uint8_t *qs = blk + 16;
                for (uint32_t j = 0; j < 8u; j++) {
                    uint8_t sc, m;
                    if (j < 4u) { sc = scales[j] & 63u; m = scales[j + 4u] & 63u; }
                    else { sc = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
                           m = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u); }
                    const uint32_t byte_off = (j >> 1u) * 32u;
                    const int shift = (int)(j & 1u) * 4;
                    for (uint32_t i = 0; i < 32u; i++) {
                        const int q4 = (qs[byte_off + i] >> shift) & 0xF;
                        ref += (double)(d * sc * q4 - dmin * m) * xh[b * 256u + j * 32u + i];
                    }
                }
            }
            /* out_hc[0][0] 校验: block_v*post[0] + Σ comb[0+src*n_hc]*res[src][0] */
            float oh0 = 0; (void)cudaMemcpy(&oh0, out_hc->ptr, 4, cudaMemcpyDeviceToHost);
            float spl[80]; float res0[4];
            (void)cudaMemcpy(spl, split->ptr, sizeof(float) * (2u * n_hc + n_hc * n_hc), cudaMemcpyDeviceToHost);
            for (uint32_t s = 0; s < n_hc && s < 4u; s++)
                (void)cudaMemcpy(&res0[s], (const char *)residual_hc->ptr + (uint64_t)s * n_embd * 4, 4, cudaMemcpyDeviceToHost);
            double ohref = ref * spl[n_hc];
            for (uint32_t s = 0; s < n_hc; s++) ohref += spl[2u * n_hc + 0u + s * n_hc] * res0[s];
            int hcnan = 0; float hcmax = 0;
            {   const uint64_t hn = (uint64_t)n_hc * n_embd;
                float *hf = (float *)malloc(hn * 4);
                if (hf && cudaMemcpy(hf, out_hc->ptr, hn * 4, cudaMemcpyDeviceToHost) == cudaSuccess)
                    for (uint64_t i2 = 0; i2 < hn; i2++) {
                        if (hf[i2] != hf[i2]) hcnan++;
                        else if (fabsf(hf[i2]) > hcmax) hcmax = fabsf(hf[i2]);
                    }
                free(hf);
            }
            fprintf(stderr, "ds4: [ao-b#%02d] bo0 g=%.3f h=%.3f | hc g=%.3f h=%.3f | hc_nan=%d hc_max=%.1f\n",
                    onceb - 1, bo0, ref, oh0, ohref, hcnan, hcmax);
            fflush(stderr); free(xh);
        }
    }
    return cuda_ok(cudaGetLastError(), "matmul_q4k_hc_expand launch");
}

int ds4_gpu_matmul_f16_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "f16");
    if (!wptr) return 0;
    const __half *w = (const __half *)wptr;
    const int serial_f16 = 0;
    const int router_shape = in_dim == 4096u && out_dim == 256u && n_tok == 1u;
    const int serial_router =
        !serial_f16 &&
        router_shape &&
        0;
    const int ordered_router =
        !serial_f16 &&
        !serial_router &&
        (n_tok == 1u || (n_tok <= 8u && 1)) &&
        1;
    /* 小批数值对齐(2026-08-21): verify 批(k<=8)改走 decode 的有序 kernel —— cublas 会把
     * 激活降成 f16(10 位尾数)再算, 与单 token 的 fp32 有序路差 ~1e-5, 逐层放大后 10% 位置
     * argmax 翻转, 直接吃掉投机接受率。prefill(大批)仍走 cublas: 那里吞吐优先。 */
    const int f16_exact_batch = (n_tok > 1 && n_tok <= 8 &&
                                 1);
    if (!serial_f16 && g_cublas_ready && n_tok > 1 && !f16_exact_batch) {
        const uint64_t xh_count = n_tok * in_dim;
        __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half), "f16 gemm activations");
        if (!xh) return 0;
        f32_to_f16_kernel<<<(xh_count + 255) / 256, 256>>>(xh, (const float *)x->ptr, xh_count);
        if (!cuda_ok(cudaGetLastError(), "f16 activation convert launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(g_cublas,
                                         CUBLAS_OP_T,
                                         CUBLAS_OP_N,
                                         (int)out_dim,
                                         (int)n_tok,
                                         (int)in_dim,
                                         &alpha,
                                         w,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         xh,
                                         CUDA_R_16F,
                                         (int)in_dim,
                                         &beta,
                                         out->ptr,
                                         CUDA_R_32F,
                                         (int)out_dim,
                                         CUDA_R_32F,
                                         CUBLAS_GEMM_DEFAULT);
        return cublas_ok(st, "f16 matmul");
    }
    /* 诊断探针(DS4_F16_DIMS=1): 打印每个唯一 (in,out,路径), 用后即弃 */
    if (((const char *)0) /* DS4_F16_DIMS: 路径开关已删(2026-08-22 隐形炸弹清理) */) {
        static uint64_t seen[64][2]; static int nseen = 0;
        int hit = 0;
        for (int i = 0; i < nseen; i++) if (seen[i][0] == in_dim && seen[i][1] == out_dim) { hit = 1; break; }
        if (!hit && nseen < 64) {
            seen[nseen][0] = in_dim; seen[nseen][1] = out_dim; nseen++;
            fprintf(stderr, "ds4: [f16-dims] in=%llu out=%llu serial=%d ord=%d ntok=%llu\n",
                    (unsigned long long)in_dim, (unsigned long long)out_dim,
                    (int)(serial_f16 || serial_router), (int)ordered_router, (unsigned long long)n_tok);
        }
    }
    /* split-K 缓冲预分配(必须在非 capture 调用里做, 见 down partial 同款注释) */
    if (!g_f16sk_partial) {
        cudaStreamCaptureStatus fcs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(0, &fcs);
        if (fcs == cudaStreamCaptureStatusNone) {
            /* [n_tok<=8][S*out<=4096] */
            if (cudaMalloc(&g_f16sk_partial, 8u * 4096u * sizeof(float)) != cudaSuccess) {
                g_f16sk_partial = NULL; (void)cudaGetLastError();
            }
        }
    }
    if (n_tok <= 8 && !serial_f16 && !serial_router &&
        (n_tok == 1 || 1) &&
        out_dim <= 512u && in_dim >= 4096u && g_f16sk_partial) {
        /* out≤64: 多段 split-K + 固定序 reduce; 64<out≤512: S=1 单段=一行一块直写
         * (原 8 行/块只发 out/8 个块, out=256 时 32 块=1/6 GPU)。 */
        uint32_t S = (uint32_t)((192u + out_dim - 1u) / out_dim);
        if (S > 64u) S = 64u;
        if ((uint64_t)S * out_dim > 4096u) S = (uint32_t)(4096u / out_dim);
        if (S == 0) S = 1;
        uint32_t chunk = (uint32_t)(((in_dim + (uint64_t)S * 8u - 1u) / ((uint64_t)S * 8u)) * 8u);
        float *pdst = (S == 1) ? (float *)out->ptr : g_f16sk_partial;
        unsigned skcells = (unsigned)out_dim * S * (unsigned)n_tok;
        if (skcells > 1024u) skcells = 1024u;
        matmul_f16_splitk_kernel<<<skcells, 256, 0, g_cur_stream>>>(pdst, w, (const float *)x->ptr,
                                                  in_dim, out_dim, chunk, S, (uint32_t)n_tok);
        if (S > 1) {
            const unsigned rn = (unsigned)(out_dim * n_tok);
            matmul_f16_splitk_reduce_kernel<<<(rn + 255u) / 256u, 256, 0, g_cur_stream>>>(
                    (float *)out->ptr, g_f16sk_partial, (uint32_t)out_dim, S, (uint32_t)n_tok);
        }
        return cuda_ok(cudaGetLastError(), "matmul_f16_splitk launch");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    if (serial_f16 || serial_router) {
        matmul_f16_serial_kernel<<<grid, 1>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(), serial_router ? "matmul_f16_router_serial launch" : "matmul_f16_serial launch");
    }
    if (ordered_router) {
        dim3 ogrid(((unsigned)out_dim + 7u) / 8u, (unsigned)n_tok, 1);
        matmul_f16_ordered_chunks_kernel<<<ogrid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
        return cuda_ok(cudaGetLastError(), "matmul_f16_ordered_chunks launch");
    }
    matmul_f16_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f16 launch");
}

int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor *out0,
        ds4_gpu_tensor *out1,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight0_offset,
        uint64_t weight1_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) {
        return 0;
    }
    if (n_tok != 1 ||
        0 ||
        0 ||
        0 ||
        0) {
        return ds4_gpu_matmul_f16_tensor(out0, model_map, model_size, weight0_offset,
                                           in_dim, out_dim, x, n_tok) &&
               ds4_gpu_matmul_f16_tensor(out1, model_map, model_size, weight1_offset,
                                           in_dim, out_dim, x, n_tok);
    }
    if (weight0_offset > model_size || weight1_offset > model_size ||
        out_dim > UINT64_MAX / in_dim) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (weight_bytes > model_size - weight0_offset ||
        weight_bytes > model_size - weight1_offset ||
        x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out_dim * sizeof(float) ||
        out1->bytes < out_dim * sizeof(float)) {
        return 0;
    }
    const __half *w0 = (const __half *)cuda_model_range_ptr(model_map, weight0_offset, weight_bytes, "f16_pair0");
    const __half *w1 = (const __half *)cuda_model_range_ptr(model_map, weight1_offset, weight_bytes, "f16_pair1");
    if (!w0 || !w1) return 0;
    if (((const char *)0) /* DS4_F16_DIMS: 路径开关已删(2026-08-22 隐形炸弹清理) */) {   /* 诊断探针: pair 形状(与单矩阵入口同款) */
        static uint64_t pseen[64][2]; static int pn = 0;
        int hit = 0;
        for (int i = 0; i < pn; i++) if (pseen[i][0] == in_dim && pseen[i][1] == out_dim) { hit = 1; break; }
        if (!hit && pn < 64) {
            pseen[pn][0] = in_dim; pseen[pn][1] = out_dim; pn++;
            fprintf(stderr, "ds4: [f16-pair-dims] in=%llu out=%llu (grid=%u blk=256)\n",
                    (unsigned long long)in_dim, (unsigned long long)out_dim, (unsigned)(2u * out_dim));
        }
    }
    static uint32_t fp2 = 99u;
    if (fp2 == 99u) { const char *e = ((const char *)0) /* DS4_F16_PAIR2: 路径开关已删(2026-08-22 隐形炸弹清理) */; fp2 = e ? (uint32_t)atoi(e) : 0u; }   /* 默认关: 2行版实测2778→2866反向(多行族二败) */
    unsigned pgrid = fp2 ? (unsigned)out_dim : (unsigned)(2u * out_dim);
    if (pgrid > ds4_grid_cap()) pgrid = ds4_grid_cap();   /* pair3(施工日2): 常驻块行循环, 灭块级碎片 */
    matmul_f16_pair_rowblock_kernel<<<pgrid,
                                      256, 0, g_cur_stream>>>(
        (float *)out0->ptr,
        (float *)out1->ptr,
        w0,
        w1,
        (const float *)x->ptr,
        in_dim,
        out_dim,
        fp2);
    return cuda_ok(cudaGetLastError(), "matmul_f16_pair_rowblock launch");
}

int ds4_gpu_matmul_f32_tensor(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    uint64_t weight_elems = out_dim * in_dim;
    if (weight_elems > UINT64_MAX / sizeof(float)) return 0;
    uint64_t weight_bytes = weight_elems * sizeof(float);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, weight_bytes, "f32");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    if (g_cublas_ready && n_tok > 1) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemm(g_cublas,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        (int)out_dim,
                                        (int)n_tok,
                                        (int)in_dim,
                                        &alpha,
                                        w,
                                        (int)in_dim,
                                        (const float *)x->ptr,
                                        (int)in_dim,
                                        &beta,
                                        (float *)out->ptr,
                                        (int)out_dim);
        return cublas_ok(st, "f32 matmul");
    }
    dim3 grid((unsigned)out_dim, (unsigned)n_tok, 1);
    matmul_f32_kernel<<<grid, 256>>>((float *)out->ptr, w, (const float *)x->ptr, in_dim, out_dim, n_tok);
    return cuda_ok(cudaGetLastError(), "matmul_f32 launch");
}

int ds4_gpu_repeat_hc_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *row, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !row || n_embd == 0 || n_hc == 0 ||
        row->bytes < (uint64_t)n_embd * sizeof(float) ||
        out->bytes < (uint64_t)n_embd * n_hc * sizeof(float)) {
        return 0;
    }
    uint64_t n = (uint64_t)n_embd * n_hc;
    repeat_hc_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)row->ptr, n_embd, n_hc);
    return cuda_ok(cudaGetLastError(), "repeat_hc launch");
}

int ds4_gpu_rms_norm_plain_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, float eps) {
    if (!out || !x || out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float)) return 0;
    rms_norm_fast_kernel<<<1, 1024>>>((float *)out->ptr, (const float *)x->ptr, NULL, n, 1, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_plain launch");
}
int ds4_gpu_rms_norm_plain_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    rms_norm_fast_kernel<<<rows, 1024>>>((float *)out->ptr, (const float *)x->ptr, NULL, n, rows, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_plain launch");
}
int ds4_gpu_rms_norm_weight_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, float eps) {
    if (!out || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < (uint64_t)n * sizeof(float) ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, (uint64_t)n * sizeof(float), "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_fast_kernel<<<1, 1024>>>((float *)out->ptr, (const float *)x->ptr, w, n, 1, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}
int ds4_gpu_rms_norm_weight_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x, const void *model_map, uint64_t model_size, uint64_t weight_offset, uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < (uint64_t)n * sizeof(float) ||
        out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    const char *wptr = cuda_model_range_ptr(model_map, weight_offset, (uint64_t)n * sizeof(float), "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    rms_norm_fast_kernel<<<rows, 1024>>>((float *)out->ptr, (const float *)x->ptr, w, n, rows, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight launch");
}
int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps) {
    if (1) {
        if (!q_out || !q || !kv_out || !kv || !model_map ||
            q_weight_offset > model_size ||
            kv_weight_offset > model_size ||
            model_size - q_weight_offset < (uint64_t)q_n * sizeof(float) ||
            model_size - kv_weight_offset < (uint64_t)kv_n * sizeof(float) ||
            q_out->bytes < (uint64_t)q_n * rows * sizeof(float) ||
            q->bytes < (uint64_t)q_n * rows * sizeof(float) ||
            kv_out->bytes < (uint64_t)kv_n * rows * sizeof(float) ||
            kv->bytes < (uint64_t)kv_n * rows * sizeof(float)) {
            return 0;
        }
        const float *q_w = (const float *)cuda_model_range_ptr(model_map,
                q_weight_offset, (uint64_t)q_n * sizeof(float), "q_rms_weight");
        const float *kv_w = (const float *)cuda_model_range_ptr(model_map,
                kv_weight_offset, (uint64_t)kv_n * sizeof(float), "kv_rms_weight");
        if (!q_w || !kv_w) return 0;
        dim3 grid(rows, 2u, 1u);
        dsv4_qkv_rms_norm_rows_kernel<<<grid, 256>>>(
                (float *)q_out->ptr,
                (const float *)q->ptr,
                q_w,
                q_n,
                (float *)kv_out->ptr,
                (const float *)kv->ptr,
                kv_w,
                kv_n,
                rows,
                eps);
        return cuda_ok(cudaGetLastError(), "dsv4 qkv rms norm rows launch");
    }
    return ds4_gpu_rms_norm_weight_rows_tensor(q_out, q, model_map, model_size,
                                                 q_weight_offset, q_n, rows, eps) &&
           ds4_gpu_rms_norm_weight_rows_tensor(kv_out, kv, model_map, model_size,
                                                 kv_weight_offset, kv_n, rows, eps);
}
int ds4_gpu_head_rms_norm_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float eps) {
    if (!x || x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    head_rms_norm_kernel<<<n_tok * n_head, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm launch");
}
int ds4_gpu_head_rms_norm_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float eps) {
    if (!x || n_rot > head_dim || (n_rot & 1u) ||
        x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    head_rms_norm_rope_tail_kernel<<<n_tok * n_head, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm_rope_tail launch");
}
int ds4_gpu_dsv4_fp8_kv_quantize_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    if (!x || n_rot > head_dim || x->bytes < (uint64_t)n_tok * head_dim * sizeof(float)) return 0;
    fp8_kv_quantize_kernel<<<n_tok, 64>>>((float *)x->ptr, n_tok, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "fp8_kv_quantize launch");
}
int ds4_gpu_dsv4_indexer_qat_tensor(ds4_gpu_tensor *x, uint32_t n_rows, uint32_t head_dim) {
    if (!x || n_rows == 0 || head_dim != 128u ||
        x->bytes < (uint64_t)n_rows * head_dim * sizeof(float)) {
        return 0;
    }
    indexer_hadamard_fp4_kernel<<<n_rows, 128>>>((float *)x->ptr, n_rows, head_dim);
    return cuda_ok(cudaGetLastError(), "indexer_hadamard_fp4 launch");
}
int ds4_gpu_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow) {
    if (!x || n_rot > head_dim || (n_rot & 1) || x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    uint32_t pairs = n_tok * n_head * (n_rot / 2);
    rope_tail_kernel<<<(pairs + 255) / 256, 256>>>((float *)x->ptr, n_tok, n_head, head_dim, n_rot, pos0, 1, n_ctx_orig, inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    return cuda_ok(cudaGetLastError(), "rope_tail launch");
}
int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row,
        uint32_t          head_dim,
        uint32_t          n_rot) {
    return ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1, head_dim, n_rot) &&
           ds4_gpu_store_raw_kv_tensor(raw_cache, kv, raw_cap, raw_row, head_dim);
}
int ds4_gpu_store_raw_kv_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t row, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        kv->bytes < (uint64_t)head_dim * sizeof(float)) return 0;
    store_raw_kv_batch_kernel<<<(head_dim + 255) / 256, 256>>>((float *)raw_cache->ptr, (const float *)kv->ptr, raw_cap, row, 1, head_dim);
    return cuda_ok(cudaGetLastError(), "store_raw_kv launch");
}
int ds4_gpu_store_raw_kv_batch_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    if (!raw_cache || !kv || raw_cap == 0 ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float)) return 0;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    store_raw_kv_batch_kernel<<<(n + 255) / 256, 256>>>((float *)raw_cache->ptr, (const float *)kv->ptr, raw_cap, pos0, n_tokens, head_dim);
    return cuda_ok(cudaGetLastError(), "store_raw_kv_batch launch");
}
int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    uint64_t n = (uint64_t)n_tokens * width;
    compressor_store_kernel<<<(n + 255) / 256, 256>>>(
            (const float *)kv->ptr,
            (const float *)sc->ptr,
            (float *)state_kv->ptr,
            (float *)state_score->ptr,
            ape,
            0,
            ape_type,
            head_dim,
            ratio,
            pos0,
            n_tokens);
    return cuda_ok(cudaGetLastError(), "compressor store launch");
}

int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache ||
        !model_map || head_dim == 0 || ratio == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)(comp_row + (emit ? 1u : 0u)) * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv_cur->bytes < kv_bytes || sc_cur->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (emit && comp_cache->bytes < comp_bytes)) {
        return 0;
    }
    if (!ds4_gpu_compressor_store_batch_tensor(kv_cur, sc_cur, state_kv, state_score,
                                                 model_map, model_size, ape_offset, ape_type,
                                                 head_dim, ratio, pos, 1)) {
        return 0;
    }
    if (!emit) return 1;
    ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(
            comp_cache,
            (uint64_t)comp_row * head_dim * sizeof(float),
            (uint64_t)head_dim * sizeof(float));
    if (!comp_row_view) return 0;
    compressor_update_pool_kernel<<<(head_dim + 255) / 256, 256, 0, g_cur_stream>>>(
            (float *)comp_row_view->ptr,
            (const float *)state_kv->ptr,
            (const float *)state_score->ptr,
            head_dim,
            ratio);
    int ok = cuda_ok(cudaGetLastError(), "compressor update pool launch");
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_row_view, comp_row_view,
                                                       model_map, model_size, norm_offset,
                                                       head_dim, 1, rms_eps);
    if (ok) ok = ds4_gpu_rope_tail_tensor(comp_row_view, 1, 1, head_dim, n_rot,
                                            pos + 1u - ratio, n_ctx_orig, false,
                                            freq_base, freq_scale, ext_factor, attn_factor,
                                            beta_fast, beta_slow);
    ds4_gpu_tensor_free(comp_row_view);
    if (ok && ratio == 4u) {
        uint64_t half = 4ull * width;
        compressor_shift_ratio4_kernel<<<(half + 255) / 256, 256>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr, width);
        ok = cuda_ok(cudaGetLastError(), "compressor ratio4 shift launch");
    }
    return ok;
}
int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }

    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint32_t cutoff = n_comp * ratio;
    const uint32_t rem = n_tokens - cutoff;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);

    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (n_comp && comp_cache->bytes < comp_bytes)) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;

    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor state score fill launch")) return 0;

    if (ratio == 4u) {
        if (cutoff >= ratio) {
            uint32_t prev_start = cutoff - ratio;
            uint64_t n = (uint64_t)ratio * width;
            compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    ape, 0, ape_type, width, ratio, pos0,
                    prev_start, 0, ratio);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill prev state launch")) return 0;
        }
        if (rem != 0) {
            uint64_t n = (uint64_t)rem * width;
            compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                    (float *)state_kv->ptr, (float *)state_score->ptr,
                    (const float *)kv->ptr, (const float *)sc->ptr,
                    ape, 0, ape_type, width, ratio, pos0,
                    cutoff, ratio, rem);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill rem state launch")) return 0;
        }
    } else if (rem != 0) {
        uint64_t n = (uint64_t)rem * width;
        compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
                (float *)state_kv->ptr, (float *)state_score->ptr,
                (const float *)kv->ptr, (const float *)sc->ptr,
                ape, 0, ape_type, width, ratio, pos0,
                cutoff, 0, rem);
        if (!cuda_ok(cudaGetLastError(), "compressor prefill rem state launch")) return 0;
    }
    if (n_comp != 0) {
        dim3 grid((head_dim + 255) / 256, n_comp, 1);
        compressor_prefill_pool_kernel<<<grid, 256>>>(
                (float *)comp_cache->ptr,
                (const float *)kv->ptr,
                (const float *)sc->ptr,
                (const float *)state_kv->ptr,
                (const float *)state_score->ptr,
                ape, 0, ape_type, head_dim, ratio, pos0, n_comp, 0);
        if (!cuda_ok(cudaGetLastError(), "compressor prefill pool launch")) return 0;
        if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                                   model_map, model_size, norm_offset,
                                                   head_dim, n_comp, rms_eps)) return 0;
        if (n_rot != 0) {
            const uint32_t pairs = n_comp * (n_rot / 2u);
            rope_tail_kernel<<<(pairs + 255) / 256, 256>>>(
                    (float *)comp_cache->ptr, n_comp, 1, head_dim, n_rot,
                    pos0, ratio, n_ctx_orig, 0, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            if (!cuda_ok(cudaGetLastError(), "compressor prefill rope launch")) return 0;
        }
        if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;
    }
    return 1;
}
int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || n_tokens == 0 || (n_tokens & 3u) != 0 || (pos0 & 3u) != 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }

    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        comp_cache->bytes < comp_bytes) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    dim3 grid((head_dim + 255) / 256, n_comp, 1);
    compressor_prefill_pool_kernel<<<grid, 256>>>(
            (float *)comp_cache->ptr,
            (const float *)kv->ptr,
            (const float *)sc->ptr,
            (const float *)state_kv->ptr,
            (const float *)state_score->ptr,
            ape, 0, ape_type, head_dim, ratio, pos0, n_comp, 1);
    if (!cuda_ok(cudaGetLastError(), "compressor replay pool launch")) return 0;
    if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                               model_map, model_size, norm_offset,
                                               head_dim, n_comp, rms_eps)) return 0;
    if (n_rot != 0) {
        const uint32_t pairs = n_comp * (n_rot / 2u);
        rope_tail_kernel<<<(pairs + 255) / 256, 256>>>(
                (float *)comp_cache->ptr, n_comp, 1, head_dim, n_rot,
                pos0, ratio, n_ctx_orig, 0, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        if (!cuda_ok(cudaGetLastError(), "compressor replay rope launch")) return 0;
    }
    if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;

    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor replay state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor replay state score fill launch")) return 0;
    uint32_t prev_start = n_tokens - ratio;
    uint64_t n = (uint64_t)ratio * width;
    compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv->ptr, (const float *)sc->ptr,
            ape, 0, ape_type, width, ratio, pos0,
            prev_start, 0, ratio);
    return cuda_ok(cudaGetLastError(), "compressor replay state launch");
}
int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0) {
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0 || (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t tail_bytes = (uint64_t)ratio * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)ratio * width * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv_tail->bytes < tail_bytes || sc_tail->bytes < tail_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        return 0;
    }
    const char *ape = cuda_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    uint64_t state_n = (uint64_t)state_rows * width;
    if (!cuda_ok(cudaMemsetAsync(state_kv->ptr, 0, (size_t)(state_n * sizeof(float))),
                 "compressor state kv zero")) return 0;
    fill_f32_kernel<<<(state_n + 255) / 256, 256>>>((float *)state_score->ptr, state_n, -INFINITY);
    if (!cuda_ok(cudaGetLastError(), "compressor state score fill launch")) return 0;
    uint64_t n = (uint64_t)ratio * width;
    compressor_set_rows_kernel<<<(n + 255) / 256, 256>>>(
            (float *)state_kv->ptr, (float *)state_score->ptr,
            (const float *)kv_tail->ptr, (const float *)sc_tail->ptr,
            ape, 0, ape_type, width, ratio, pos0,
            0, 0, ratio);
    return cuda_ok(cudaGetLastError(), "compressor state set launch");
}
int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16 ||
        !heads || !q || !raw_kv || !model_map || n_raw == 0 || raw_cap < n_raw ||
        raw_start >= raw_cap || (n_comp != 0 && !comp_kv) || (use_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_mask && comp_mask->bytes < (uint64_t)n_comp * sizeof(float))) {
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (!cuda_attention_score_buffer_fits(n_comp)) {
        if (!use_mask && head_dim == 512u &&
            1) {
            dim3 online_grid(1, (n_head + 7u) / 8u, 1);
            attention_decode_mixed_heads8_online_kernel<<<online_grid, 256>>>((float *)heads->ptr,
                                                                              sinks,
                                                                              (const float *)q->ptr,
                                                                              (const float *)raw_kv->ptr,
                                                                              n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                              1,
                                                                              0,
                                                                              n_raw,
                                                                              raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              0,
                                                                              0,
                                                                              n_head,
                                                                              head_dim);
            return cuda_ok(cudaGetLastError(), "attention decode online launch");
        }
        fprintf(stderr, "ds4: CUDA attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    dim3 grid(1, n_head, 1);
    attention_decode_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                 sinks,
                                                 (const float *)q->ptr,
                                                 (const float *)raw_kv->ptr,
                                                 n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                 use_mask ? (const float *)comp_mask->ptr : NULL,
                                                 use_mask,
                                                 1, 0, n_raw, raw_cap, raw_start, n_comp,
                                                 0, 0, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention decode launch");
}
int ds4_gpu_attention_prefill_raw_heads_tensor(ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size, uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window, uint32_t n_head, uint32_t head_dim) {
    if (!heads || !q || !raw_kv || !model_map || sinks_offset > model_size ||
        model_size - sinks_offset < (uint64_t)n_head * sizeof(float) ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        window > 256) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (n_tokens > 1 && head_dim == 512 &&
        1 &&
        (0 || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_static_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   0,
                                                                   window,
                                                                   1,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention raw window launch");
    }
    if (g_cublas_ready && n_tokens > 1 && head_dim == 512 &&
        1) {
        const uint32_t n_keys = n_tokens;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t out_offset = (score_bytes + 255u) & ~255ull;
        const uint64_t tmp_bytes = out_offset + out_count * sizeof(float);
        float *tmp = (float *)cuda_tmp_alloc(tmp_bytes, "attention raw cublas");
        if (!tmp) return 0;
        float *scores = tmp;
        float *out_tmp = (float *)((char *)tmp + out_offset);
        const float alpha = rsqrtf((float)head_dim);
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemmStridedBatched(g_cublas,
                                                      CUBLAS_OP_T,
                                                      CUBLAS_OP_N,
                                                      (int)n_keys,
                                                      (int)n_tokens,
                                                      (int)head_dim,
                                                      &alpha,
                                                      (const float *)raw_kv->ptr,
                                                      (int)head_dim,
                                                      0,
                                                      (const float *)q->ptr,
                                                      (int)(n_head * head_dim),
                                                      (long long)head_dim,
                                                      &beta,
                                                      scores,
                                                      (int)n_keys,
                                                      (long long)n_keys * n_tokens,
                                                      (int)n_head);
        if (!cublas_ok(st, "attention raw score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_raw_softmax_kernel<<<sgrid, 256>>>(scores, sinks, n_tokens, window, n_keys);
        if (!cuda_ok(cudaGetLastError(), "attention raw softmax launch")) return 0;
        const float one = 1.0f;
        st = cublasSgemmStridedBatched(g_cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       (const float *)raw_kv->ptr,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!cublas_ok(st, "attention raw value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return cuda_ok(cudaGetLastError(), "attention raw unpack launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_prefill_raw_kernel<<<grid, 128>>>((float *)heads->ptr,
                                                sinks,
                                                (const float *)q->ptr,
                                                (const float *)raw_kv->ptr,
                                                n_tokens, window, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention_prefill_raw launch");
}
static int attention_decode_batch_launch(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16 ||
        !heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float))) {
        return 0;
    }
    if (n_comp != 0 && ratio == 0) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (!cuda_attention_score_buffer_fits(n_comp)) {
        if (!use_comp_mask && head_dim == 512u &&
            1) {
            dim3 online_grid(n_tokens, (n_head + 7u) / 8u, 1);
            attention_decode_mixed_heads8_online_kernel<<<online_grid, 256>>>((float *)heads->ptr,
                                                                              sinks,
                                                                              (const float *)q->ptr,
                                                                              (const float *)raw_kv->ptr,
                                                                              n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                              n_tokens,
                                                                              pos0,
                                                                              n_raw,
                                                                              raw_cap,
                                                                              raw_start,
                                                                              n_comp,
                                                                              window,
                                                                              ratio,
                                                                              n_head,
                                                                              head_dim);
            return cuda_ok(cudaGetLastError(), "attention decode online launch");
        }
        fprintf(stderr, "ds4: CUDA attention score buffer too small for %u compressed rows\n", n_comp);
        return 0;
    }
    if (!use_comp_mask && n_tokens > 1 && head_dim == 512 &&
        1 &&
        (0 || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_decode_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   pos0,
                                                                   n_raw,
                                                                   raw_cap,
                                                                   raw_start,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention decode window launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_decode_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                 sinks,
                                                 (const float *)q->ptr,
                                                 (const float *)raw_kv->ptr,
                                                 n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                 use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                                                 use_comp_mask, n_tokens, pos0, n_raw, raw_cap,
                                                 raw_start, n_comp, window, ratio, n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention decode batch launch");
}

int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim) {
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, NULL, 0, NULL, 0, n_tokens, pos0,
                                      n_raw, raw_cap, raw_start, 0, window, 1,
                                      n_head, head_dim);
}

int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16) return 0;
    return attention_decode_batch_launch(heads, model_map, model_size, sinks_offset,
                                      q, raw_kv, comp_kv, comp_kv_f16, comp_mask, use_comp_mask,
                                      n_tokens, pos0, n_raw, raw_cap, raw_start,
                                      n_comp, window, ratio, n_head, head_dim);
}

int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16 ||
        !heads || !q || !raw_kv || !comp_kv || !topk || !model_map ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        n_comp == 0 || top_k == 0 ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(int32_t)) {
        return 0;
    }
    if (top_k > 512u) return 0;
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    const int32_t *topk_ptr = (const int32_t *)topk->ptr;
    if (n_tokens > 1u && top_k == 512u &&
        1) {
        const uint64_t sort_bytes = (uint64_t)n_tokens * top_k * sizeof(int32_t);
        int32_t *sorted = (int32_t *)cuda_tmp_alloc(sort_bytes, "indexed attention topk sort");
        if (!sorted) return 0;
        indexed_topk_sort_512_asc_kernel<<<n_tokens, 512>>>(sorted, topk_ptr, n_tokens);
        if (!cuda_ok(cudaGetLastError(), "indexed attention topk sort launch")) return 0;
        topk_ptr = sorted;
    }
    if (n_tokens > 1 && head_dim == 512 && top_k <= 512u &&
        1) {
        if (1) {
            dim3 grid(n_tokens, (n_head + 15u) / 16u, 1);
            attention_indexed_mixed_heads8_online_kernel<8, 16><<<grid, 512>>>((float *)heads->ptr,
                                                                               sinks,
                                                                               (const float *)q->ptr,
                                                                               (const float *)raw_kv->ptr,
                                                                               (const float *)comp_kv->ptr,
                                                                               topk_ptr,
                                                                               n_tokens,
                                                                               pos0,
                                                                               n_raw,
                                                                               raw_cap,
                                                                               raw_start,
                                                                               n_comp,
                                                                               top_k,
                                                                               window,
                                                                               ratio,
                                                                               n_head,
                                                                               head_dim);
            return cuda_ok(cudaGetLastError(), "attention indexed online launch");
        }
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_indexed_mixed_heads8_rb4_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                 sinks,
                                                                 (const float *)q->ptr,
                                                                 (const float *)raw_kv->ptr,
                                                                 (const float *)comp_kv->ptr,
                                                                 topk_ptr,
                                                                 n_tokens,
                                                                 pos0,
                                                                 n_raw,
                                                                 raw_cap,
                                                                 raw_start,
                                                                 n_comp,
                                                                 top_k,
                                                                 window,
                                                                 ratio,
                                                                 n_head,
                                                                 head_dim);
        return cuda_ok(cudaGetLastError(), "attention indexed heads8 launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_indexed_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const float *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  (const float *)comp_kv->ptr,
                                                  topk_ptr,
                                                  n_tokens,
                                                  pos0,
                                                  n_raw,
                                                  raw_cap,
                                                  raw_start,
                                                  n_comp,
                                                  top_k,
                                                  window,
                                                  ratio,
                                                  n_head,
                                                  head_dim);
    return cuda_ok(cudaGetLastError(), "attention indexed mixed launch");
}

static int attention_prefill_mixed_launch(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 || ratio == 0 ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float))) {
        return 0;
    }
    const float *sinks = (const float *)cuda_model_range_ptr(
            model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sinks) return 0;
    if (!use_comp_mask && n_tokens > 1 && head_dim == 512 &&
        1 &&
        (0 || (!g_quality_mode && n_tokens >= 128u))) {
        dim3 grid(n_tokens, (n_head + 7u) / 8u, 1);
        attention_static_mixed_heads8_online_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                                   sinks,
                                                                   (const float *)q->ptr,
                                                                   (const float *)raw_kv->ptr,
                                                                   n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                                   n_tokens,
                                                                   n_comp,
                                                                   window,
                                                                   ratio,
                                                                   n_head,
                                                                   head_dim);
        return cuda_ok(cudaGetLastError(), "attention mixed window launch");
    }
    if (g_cublas_ready && n_tokens > 1 && head_dim == 512 &&
        1) {
        const uint32_t n_keys = n_tokens + n_comp;
        const uint64_t kv_count = (uint64_t)n_keys * head_dim;
        const uint64_t score_count = (uint64_t)n_head * n_tokens * n_keys;
        const uint64_t out_count = (uint64_t)n_head * n_tokens * head_dim;
        const uint64_t kv_bytes = kv_count * sizeof(float);
        const uint64_t score_offset = (kv_bytes + 255u) & ~255ull;
        const uint64_t score_bytes = score_count * sizeof(float);
        const uint64_t out_offset = score_offset + ((score_bytes + 255u) & ~255ull);
        const uint64_t tmp_bytes = out_offset + out_count * sizeof(float);
        float *tmp = (float *)cuda_tmp_alloc(tmp_bytes, "attention mixed cublas");
        if (!tmp) return 0;
        float *kv = tmp;
        float *scores = (float *)((char *)tmp + score_offset);
        float *out_tmp = (float *)((char *)tmp + out_offset);
        attention_prefill_pack_mixed_kv_kernel<<<(kv_count + 255) / 256, 256>>>(
                kv,
                (const float *)raw_kv->ptr,
                n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                n_tokens,
                n_comp,
                head_dim);
        if (!cuda_ok(cudaGetLastError(), "attention mixed kv pack launch")) return 0;
        const float alpha = rsqrtf((float)head_dim);
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemmStridedBatched(g_cublas,
                                                      CUBLAS_OP_T,
                                                      CUBLAS_OP_N,
                                                      (int)n_keys,
                                                      (int)n_tokens,
                                                      (int)head_dim,
                                                      &alpha,
                                                      kv,
                                                      (int)head_dim,
                                                      0,
                                                      (const float *)q->ptr,
                                                      (int)(n_head * head_dim),
                                                      (long long)head_dim,
                                                      &beta,
                                                      scores,
                                                      (int)n_keys,
                                                      (long long)n_keys * n_tokens,
                                                      (int)n_head);
        if (!cublas_ok(st, "attention mixed score gemm")) return 0;
        dim3 sgrid(n_tokens, n_head, 1);
        attention_prefill_mixed_softmax_kernel<<<sgrid, 256>>>(
                scores,
                sinks,
                use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                use_comp_mask,
                n_tokens,
                n_comp,
                window,
                ratio,
                n_keys);
        if (!cuda_ok(cudaGetLastError(), "attention mixed softmax launch")) return 0;
        const float one = 1.0f;
        st = cublasSgemmStridedBatched(g_cublas,
                                       CUBLAS_OP_N,
                                       CUBLAS_OP_N,
                                       (int)head_dim,
                                       (int)n_tokens,
                                       (int)n_keys,
                                       &one,
                                       kv,
                                       (int)head_dim,
                                       0,
                                       scores,
                                       (int)n_keys,
                                       (long long)n_keys * n_tokens,
                                       &beta,
                                       out_tmp,
                                       (int)head_dim,
                                       (long long)head_dim * n_tokens,
                                       (int)n_head);
        if (!cublas_ok(st, "attention mixed value gemm")) return 0;
        uint64_t n = (uint64_t)n_tokens * n_head * head_dim;
        attention_prefill_unpack_heads_kernel<<<(n + 255) / 256, 256>>>((float *)heads->ptr,
                                                                        out_tmp,
                                                                        n_tokens,
                                                                        n_head,
                                                                        head_dim);
        return cuda_ok(cudaGetLastError(), "attention mixed unpack launch");
    }
    dim3 grid(n_tokens, n_head, 1);
    attention_prefill_mixed_kernel<<<grid, 256>>>((float *)heads->ptr,
                                                  sinks,
                                                  (const float *)q->ptr,
                                                  (const float *)raw_kv->ptr,
                                                  n_comp ? (const float *)comp_kv->ptr : (const float *)raw_kv->ptr,
                                                  use_comp_mask ? (const float *)comp_mask->ptr : NULL,
                                                  use_comp_mask, n_tokens, n_comp, window, ratio,
                                                  n_head, head_dim);
    return cuda_ok(cudaGetLastError(), "attention prefill mixed launch");
}

int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16) return 0;
    return attention_prefill_mixed_launch(heads, model_map, model_size, sinks_offset,
                                       q, raw_kv, comp_kv, NULL, 0, n_tokens,
                                       n_comp, window, ratio, n_head, head_dim);
}

int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim) {
    if (comp_kv_f16) return 0;
    return attention_prefill_mixed_launch(heads, model_map, model_size, sinks_offset,
                                       q, raw_kv, comp_kv, comp_mask, 1, n_tokens,
                                       n_comp, window, ratio, n_head, head_dim);
}
int ds4_gpu_attention_output_q4k_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens) {
    /* q4_K 版批量 attn_output(a 分组 + b 平铺), 组合两个已数值验证的部件:
     * grouped_q4_K kernel 天然带 n_tokens 维; b 侧复用 dense q4_K matmul。 */
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0) return 0;
    if (group_dim % 256u != 0 || ((uint64_t)n_groups * rank) % 256u != 0) return 0;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = group_dim / 32u;
    const uint64_t kblocks = group_dim / 256u;
    const uint64_t out_a_bytes = low_dim * kblocks * sizeof(cuda_block_q4_K);
    if (out_a_offset > model_size || out_a_bytes > model_size - out_a_offset ||
        heads->bytes < (uint64_t)n_tokens * n_groups * group_dim * sizeof(float) ||
        low->bytes < (uint64_t)n_tokens * low_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) return 0;
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a_q4k"));
    if (!out_a) return 0;
    const uint64_t x_rows = (uint64_t)n_tokens * n_groups;
    const uint64_t xq_bytes = x_rows * blocks_a * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + x_rows * blocks_a * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "attention output q4k batch prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, (const float *)heads->ptr,
                                            group_dim, blocks_a);
    if (!cuda_ok(cudaGetLastError(), "attn_out_q4k_batch prequant launch")) return 0;
    dim3 grid_a(((unsigned)low_dim + 15u) / 16u, (unsigned)n_tokens, 1);
    if (use_dp4a && kblocks <= 16u) {
        grouped_q4_K_a_preq_warp8_dp4a_kernel<<<grid_a, 256, (size_t)16u * (size_t)kblocks * 9u * sizeof(uint4)>>>((float *)low->ptr, out_a, xq, xscale,
                                                          group_dim, rank, n_groups, n_tokens,
                                                          kblocks);
    } else {
        grouped_q4_K_a_preq_warp8_kernel<<<grid_a, 256, (kblocks <= 16u) ? (size_t)16u * (size_t)kblocks * 9u * sizeof(uint4) : 0>>>((float *)low->ptr, out_a, xq, xscale,
                                                          group_dim, rank, n_groups, n_tokens,
                                                          kblocks, use_dp4a);
    }
    if (!cuda_ok(cudaGetLastError(), "attn_out_q4k_batch grouped launch")) return 0;
    return ds4_gpu_matmul_q4_K_tensor(out, model_map, model_size, out_b_offset,
                                      low_dim, out_dim, low, n_tokens);
}


int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens) {
    (void)group_tmp;
    (void)low_tmp;
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t blocks_b = (low_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 34;
    const uint64_t out_b_bytes = out_dim * blocks_b * 34;
    if (out_a_offset > model_size || out_b_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        out_b_bytes > model_size - out_b_offset ||
        heads->bytes < (uint64_t)n_tokens * n_groups * group_dim * sizeof(float) ||
        low->bytes < (uint64_t)n_tokens * low_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a"));
    const unsigned char *out_b = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_b_offset, out_b_bytes, "attn_out_b"));
    if (!out_a || !out_b) return 0;

    const __half *out_a_f16 = NULL;
    uint32_t out_a_cublas_min_tokens = 2u;
    const char *out_a_min_env = ((const char *)0) /* DS4_CUDA_ATTENTION_OUTPUT_A_CUBLAS_MIN: 路径开关已删(2026-08-22 隐形炸弹清理) */;
    if (out_a_min_env && out_a_min_env[0]) {
        char *endp = NULL;
        long v = strtol(out_a_min_env, &endp, 10);
        if (endp != out_a_min_env && v > 1 && v < 4096) out_a_cublas_min_tokens = (uint32_t)v;
    }
    if (!g_quality_mode &&
        g_cublas_ready &&
        n_tokens >= out_a_cublas_min_tokens &&
        1) {
        out_a_f16 = cuda_q8_f16_ptr(model_map, out_a_offset, out_a_bytes, group_dim, low_dim, "attn_output_a");
    }
    if (out_a_f16) {
        const uint64_t heads_h_count = (uint64_t)n_groups * n_tokens * group_dim;
        const uint64_t low_tmp_count = (uint64_t)n_groups * n_tokens * rank;
        const uint64_t heads_h_bytes = heads_h_count * sizeof(__half);
        const uint64_t low_tmp_offset = (heads_h_bytes + 255u) & ~255ull;
        const uint64_t tmp_bytes = low_tmp_offset + low_tmp_count * sizeof(float);
        void *tmp = cuda_tmp_alloc(tmp_bytes, "attention output a cublas");
        if (!tmp) return 0;
        __half *heads_h = (__half *)tmp;
        float *low_packed = (float *)((char *)tmp + low_tmp_offset);
        attention_pack_group_heads_f16_kernel<<<(heads_h_count + 255) / 256, 256>>>(
                heads_h,
                (const float *)heads->ptr,
                n_tokens,
                n_groups,
                group_dim);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a pack launch")) return 0;
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasGemmStridedBatchedEx(g_cublas,
                                                       CUBLAS_OP_T,
                                                       CUBLAS_OP_N,
                                                       (int)rank,
                                                       (int)n_tokens,
                                                       (int)group_dim,
                                                       &alpha,
                                                       out_a_f16,
                                                       CUDA_R_16F,
                                                       (int)group_dim,
                                                       (long long)rank * group_dim,
                                                       heads_h,
                                                       CUDA_R_16F,
                                                       (int)group_dim,
                                                       (long long)n_tokens * group_dim,
                                                       &beta,
                                                       low_packed,
                                                       CUDA_R_32F,
                                                       (int)rank,
                                                       (long long)rank * n_tokens,
                                                       (int)n_groups,
                                                       CUDA_R_32F,
                                                       CUBLAS_GEMM_DEFAULT);
        if (!cublas_ok(st, "attention output a gemm")) return 0;
        attention_unpack_group_low_kernel<<<(low_tmp_count + 255) / 256, 256>>>(
                (float *)low->ptr,
                low_packed,
                n_tokens,
                n_groups,
                rank);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a unpack launch")) return 0;
    } else {
        const uint64_t x_rows = (uint64_t)n_tokens * n_groups;
        const uint64_t xq_bytes = x_rows * blocks_a * 32u;
        const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
        const uint64_t tmp_bytes = scale_offset + x_rows * blocks_a * sizeof(float);
        void *tmp = cuda_tmp_alloc(tmp_bytes, "attention output a q8 prequant");
        if (!tmp) return 0;
        int8_t *xq = (int8_t *)tmp;
        float *xscale = (float *)((char *)tmp + scale_offset);
        const int use_dp4a = cuda_q8_use_dp4a();
        dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1);
        quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq,
                                                xscale,
                                                (const float *)heads->ptr,
                                                group_dim,
                                                blocks_a);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a prequant launch")) return 0;
        /* kernel 每块 8 行, /16 会漏算一半行(潜伏失配, 该批路径当前未走) */
        dim3 grid_a(((unsigned)low_dim + 7u) / 8u, (unsigned)n_tokens, 1);
        grouped_q8_0_a_preq_warp8_kernel<<<grid_a, 256>>>((float *)low->ptr,
                                                          out_a,
                                                          xq,
                                                          xscale,
                                                          group_dim,
                                                          rank,
                                                          n_groups,
                                                          n_tokens,
                                                          blocks_a,
                                                          use_dp4a);
        if (!cuda_ok(cudaGetLastError(), "attention_output_q8_a preq launch")) return 0;
    }

    (void)out_b;
    return cuda_matmul_q8_0_tensor_labeled(out,
                                           model_map,
                                           model_size,
                                           out_b_offset,
                                           low_dim,
                                           out_dim,
                                           low,
                                           n_tokens,
                                           "attn_output_b");
}
int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads) {
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 || n_groups == 0) {
        return 0;
    }
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31) / 32;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 34;
    if (out_a_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        heads->bytes < (uint64_t)n_groups * group_dim * sizeof(float) ||
        low->bytes < low_dim * sizeof(float)) {
        return 0;
    }
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a"));
    if (!out_a) return 0;

    const uint64_t x_rows = (uint64_t)n_groups;
    const uint64_t xq_bytes = x_rows * blocks_a * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + x_rows * blocks_a * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "attention output low q8 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq,
                                            xscale,
                                            (const float *)heads->ptr,
                                            group_dim,
                                            blocks_a);
    if (!cuda_ok(cudaGetLastError(), "attention_output_low_q8 prequant launch")) return 0;
    dim3 grid_a(((unsigned)low_dim + 7u) / 8u, 1, 1);
    if ((group_dim & 31u) == 0u) {
        const cuda_q8r_entry *re = cuda_q8r_get(model_map, out_a_offset, low_dim, blocks_a);
        if (re) {
            grouped_q8r_a_kernel<<<grid_a, 256>>>((float *)low->ptr,
                    re->scales, re->qs, xq, xscale, rank, n_groups, 1u, blocks_a);
            return cuda_ok(cudaGetLastError(), "attention_output_q8r a launch");
        }
    }
    grouped_q8_0_a_preq_warp8_kernel<<<grid_a, 256>>>((float *)low->ptr,
                                                      out_a,
                                                      xq,
                                                      xscale,
                                                      group_dim,
                                                      rank,
                                                      n_groups,
                                                      1,
                                                      blocks_a,
                                                      use_dp4a);
    return cuda_ok(cudaGetLastError(), "attention_output_low_q8 launch");
}
int ds4_gpu_attention_output_low_q4k_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads) {
    /* q4_K 版 attn_output_a(与 q8 入口同构): 激活预量化仍是 q8_0 32 块, 权重行
     * = group_dim/256 个 q4_K 块, dot 走 dev_dot_q4_K_q8_0x8。 */
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 || n_groups == 0) return 0;
    if (group_dim % 256u != 0) return 0;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = group_dim / 32u;          /* q8_0 激活块数 */
    const uint64_t kblocks = group_dim / 256u;          /* q4_K 权重块数 */
    const uint64_t out_a_bytes = low_dim * kblocks * sizeof(cuda_block_q4_K);
    if (out_a_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        heads->bytes < (uint64_t)n_groups * group_dim * sizeof(float) ||
        low->bytes < low_dim * sizeof(float)) return 0;
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a_q4k"));
    if (!out_a) return 0;
    const uint64_t x_rows = (uint64_t)n_groups;
    const uint64_t xq_bytes = x_rows * blocks_a * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + x_rows * blocks_a * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "attention output low q4k prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    const int use_dp4a = cuda_q8_use_dp4a();
    dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale, (const float *)heads->ptr,
                                            group_dim, blocks_a);
    if (!cuda_ok(cudaGetLastError(), "attention_output_low_q4k prequant launch")) return 0;
    dim3 grid_a(((unsigned)low_dim + 15u) / 16u, 1, 1);
    if (use_dp4a && kblocks <= 16u) {
        grouped_q4_K_a_preq_warp8_dp4a_kernel<<<grid_a, 256, (size_t)16u * (size_t)kblocks * 9u * sizeof(uint4)>>>((float *)low->ptr, out_a, xq, xscale,
                                                          group_dim, rank, n_groups, 1,
                                                          kblocks);
    } else {
        grouped_q4_K_a_preq_warp8_kernel<<<grid_a, 256, (kblocks <= 16u) ? (size_t)16u * (size_t)kblocks * 9u * sizeof(uint4) : 0>>>((float *)low->ptr, out_a, xq, xscale,
                                                          group_dim, rank, n_groups, 1,
                                                          kblocks, use_dp4a);
    }
    if (((const char *)0) /* DS4_AO_PROBE: 诊断开关已删(2026-08-22) */) {
        static int once = 0;
        if (once++ < 96) {
            (void)cudaDeviceSynchronize();
            float lo[4]; float *xh = (float *)malloc(group_dim * sizeof(float));
            (void)cudaMemcpy(lo, low->ptr, sizeof(lo), cudaMemcpyDeviceToHost);
            (void)cudaMemcpy(xh, heads->ptr, group_dim * sizeof(float), cudaMemcpyDeviceToHost);
            /* host 参考: row0 属 group0, 用原始 mmap 上的 q4_K 权重逐块 dequant×x */
            const uint8_t *w0 = (const uint8_t *)model_map + out_a_offset;   /* row0 */
            double ref = 0.0;
            for (uint64_t b = 0; b < kblocks; b++) {
                const uint8_t *blk = w0 + b * 144u;
                uint16_t hd, hm; memcpy(&hd, blk + 0, 2); memcpy(&hm, blk + 2, 2);
                const float d = dev_host_f16(hd), dmin = dev_host_f16(hm);
                const uint8_t *scales = blk + 4; const uint8_t *qs = blk + 16;
                for (uint32_t j = 0; j < 8u; j++) {
                    uint8_t sc, m;
                    if (j < 4u) { sc = scales[j] & 63u; m = scales[j + 4u] & 63u; }
                    else { sc = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
                           m = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u); }
                    const uint32_t byte_off = (j >> 1u) * 32u;
                    const int shift = (int)(j & 1u) * 4;
                    for (uint32_t i = 0; i < 32u; i++) {
                        const int q4 = (qs[byte_off + i] >> shift) & 0xF;
                        ref += (double)(d * sc * q4 - dmin * m) * xh[b * 256u + j * 32u + i];
                    }
                }
            }
            /* 跨组检查: group1 首行(row=rank), 激活用 heads 第二段 */
            float lo_r1 = 0; (void)cudaMemcpy(&lo_r1, (const char *)low->ptr + rank * sizeof(float), 4, cudaMemcpyDeviceToHost);
            float *xh1 = (float *)malloc(group_dim * sizeof(float));
            (void)cudaMemcpy(xh1, (const char *)heads->ptr + group_dim * sizeof(float), group_dim * sizeof(float), cudaMemcpyDeviceToHost);
            const uint8_t *w1 = (const uint8_t *)model_map + out_a_offset + (uint64_t)rank * kblocks * 144u;
            double ref1 = 0.0;
            for (uint64_t b = 0; b < kblocks; b++) {
                const uint8_t *blk = w1 + b * 144u;
                uint16_t hd, hm; memcpy(&hd, blk + 0, 2); memcpy(&hm, blk + 2, 2);
                const float d = dev_host_f16(hd), dmin = dev_host_f16(hm);
                const uint8_t *scales = blk + 4; const uint8_t *qs = blk + 16;
                for (uint32_t j = 0; j < 8u; j++) {
                    uint8_t sc, m;
                    if (j < 4u) { sc = scales[j] & 63u; m = scales[j + 4u] & 63u; }
                    else { sc = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
                           m = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u); }
                    const uint32_t byte_off = (j >> 1u) * 32u;
                    const int shift = (int)(j & 1u) * 4;
                    for (uint32_t i = 0; i < 32u; i++) {
                        const int q4 = (qs[byte_off + i] >> shift) & 0xF;
                        ref1 += (double)(d * sc * q4 - dmin * m) * xh1[b * 256u + j * 32u + i];
                    }
                }
            }
            fprintf(stderr, "ds4: [ao-probe] gpu_low0=%.4f host=%.4f | g1: gpu=%.4f host=%.4f\n",
                    lo[0], ref, lo_r1, ref1);
            fflush(stderr); free(xh); free(xh1);
        }
    }
    return cuda_ok(cudaGetLastError(), "attention_output_low_q4k launch");
}
int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate, const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    if (!out || !gate || !up ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        gate->bytes < (uint64_t)n * sizeof(float) ||
        up->bytes < (uint64_t)n * sizeof(float)) return 0;
    swiglu_kernel<<<(n + 255) / 256, 256, 0, g_cur_stream>>>((float *)out->ptr, (const float *)gate->ptr, (const float *)up->ptr, n, clamp, weight);
    return cuda_ok(cudaGetLastError(), "swiglu launch");
}
int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    if (1) {
        return ds4_gpu_matmul_q8_0_pair_tensor(gate, up,
                                                 model_map, model_size,
                                                 gate_offset, up_offset,
                                                 in_dim, out_dim, out_dim,
                                                 x, 1) &&
               ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f);
    }
    return ds4_gpu_matmul_q8_0_tensor(gate, model_map, model_size,
                                        gate_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_matmul_q8_0_tensor(up, model_map, model_size,
                                        up_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f);
}
int ds4_gpu_add_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint32_t n) {
    if (!out || !a || !b ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        a->bytes < (uint64_t)n * sizeof(float) ||
        b->bytes < (uint64_t)n * sizeof(float)) return 0;
    add_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)a->ptr, (const float *)b->ptr, n);
    return cuda_ok(cudaGetLastError(), "add launch");
}
int ds4_gpu_corr_router_bias(ds4_gpu_tensor *logits, const ds4_gpu_tensor *delta,
        uint32_t n_expert, uint32_t n_tokens) {
    if (!logits || !delta || n_expert == 0 || n_tokens == 0) return 0;
    uint64_t total = (uint64_t)n_tokens * n_expert;
    if (logits->bytes < total * sizeof(float) ||
        delta->bytes < (uint64_t)n_expert * sizeof(float)) return 0;
    corr_router_bias_kernel<<<(unsigned)((total + 255) / 256), 256>>>(
        (float *)logits->ptr, (const float *)delta->ptr, n_expert, total);
    return cuda_ok(cudaGetLastError(), "corr router bias launch");
}
int ds4_gpu_corr_apply(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *U, const ds4_gpu_tensor *V, const ds4_gpu_tensor *C,
        const ds4_gpu_tensor *b, const ds4_gpu_tensor *beta, const ds4_gpu_tensor *selected,
        uint32_t d_model, uint32_t d_l, uint32_t n_expert, uint32_t n_expert_used, uint32_t n_tokens) {
    if (!out || !x || !U || !V || !C || !b || !beta || !selected ||
        d_model == 0 || d_l == 0 || n_expert == 0 || n_expert_used == 0 || n_tokens == 0) return 0;
    uint64_t vec_bytes = (uint64_t)n_tokens * d_model * sizeof(float);
    if (out->bytes < vec_bytes || x->bytes < vec_bytes ||
        U->bytes < (uint64_t)d_model * d_l * sizeof(float) ||
        V->bytes < (uint64_t)d_l * d_model * sizeof(float) ||
        C->bytes < (uint64_t)n_expert * d_l * sizeof(float) ||
        b->bytes < (uint64_t)d_model * sizeof(float) ||
        beta->bytes < (uint64_t)n_expert * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert_used * sizeof(int32_t)) return 0;
    corr_apply_kernel<<<n_tokens, 256, (size_t)d_l * sizeof(float)>>>(
        (float *)out->ptr, (const float *)x->ptr, (const float *)U->ptr, (const float *)V->ptr,
        (const float *)C->ptr, (const float *)b->ptr, (const float *)beta->ptr,
        /* n_tokens 走 grid 维度(blockIdx.x = tok), kernel 形参里没有它 */
        (const int *)selected->ptr, d_model, d_l, n_expert, n_expert_used);
    return cuda_ok(cudaGetLastError(), "corr apply launch");
}
/* Metal-only decode optimization (routed_out write-hazard bubble); CUDA keeps
 * the in-place corr kernel — callers must gate on this returning 0. */
int ds4_gpu_corr_delta_supported(void) { return 0; }
int ds4_gpu_corr_apply_delta(ds4_gpu_tensor *delta_out, const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *U, const ds4_gpu_tensor *V, const ds4_gpu_tensor *C,
        const ds4_gpu_tensor *b, const ds4_gpu_tensor *beta, const ds4_gpu_tensor *selected,
        uint32_t d_model, uint32_t d_l, uint32_t n_expert, uint32_t n_expert_used, uint32_t n_tokens) {
    (void)delta_out; (void)x; (void)U; (void)V; (void)C; (void)b; (void)beta; (void)selected;
    (void)d_model; (void)d_l; (void)n_expert; (void)n_expert_used; (void)n_tokens;
    return 0;
}
int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale) {
    if (!x || !directions || width == 0 || rows == 0 || scale == 0.0f) return 0;
    const uint64_t x_bytes = (uint64_t)width * rows * sizeof(float);
    const uint64_t dir_bytes = (uint64_t)(layer + 1u) * width * sizeof(float);
    if (x->bytes < x_bytes || directions->bytes < dir_bytes) return 0;

    uint32_t nth = 256u;
    while (nth > width && nth > 1u) nth >>= 1;
    directional_steering_project_kernel<<<rows, nth>>>(
            (float *)x->ptr,
            (const float *)directions->ptr,
            layer,
            width,
            rows,
            scale);
    return cuda_ok(cudaGetLastError(), "directional steering launch");
}
int ds4_gpu_router_select_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t token, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits, uint32_t layer) {
    (void)layer;   /* CUDA has no shrunken/keep-map path yet; accept for ABI parity with Metal */
    if (!selected || !weights || !probs || !logits || !model_map || n_expert_groups > 1u || n_group_used > 0u) return 0;
    if (n_expert != 256u || n_expert_used != 6u || fabsf(expert_weight_scale - 1.5f) > 1.0e-6f) return 0;
    int32_t tok = (int32_t)token;
    int ok = 1;
    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (ok && has_bias && !hash_mode) {
        if (bias_offset > model_size || model_size - bias_offset < 256u * sizeof(float)) ok = 0;
        else bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, 256u * sizeof(float), "router_bias");
        if (!bias) ok = 0;
    }
    if (ok && hash_mode) {
        const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset) ok = 0;
        else hash = (const int32_t *)cuda_model_range_ptr(model_map, hash_offset, hash_bytes, "router_hash");
        if (!hash) ok = 0;
    }
    if (ok) {
        if (1 &&
            1) {
            dim3 block(32, 4, 1);
            router_select_warp_topk_kernel<<<1, block>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                                         bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                                         has_bias && !hash_mode, hash_mode);
        } else if (1) {
            router_select_parallel_kernel<<<1, 256>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                                      bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                                      has_bias && !hash_mode, hash_mode);
        } else {
            router_select_kernel<<<1, 1>>>((int32_t *)selected->ptr, (float *)weights->ptr, (float *)probs->ptr,
                                          bias, hash, (const float *)logits->ptr, NULL, tok, hash_rows, 1,
                                          has_bias && !hash_mode, hash_mode);
        }
        ok = cuda_ok(cudaGetLastError(), "router_select launch");
    }
    return ok;
}
int ds4_gpu_router_select_batch_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights, ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size, uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows, uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits, const ds4_gpu_tensor *tokens, uint32_t n_expert, uint32_t n_expert_used, float expert_weight_scale, uint32_t n_tokens, uint32_t layer) {
    (void)layer;   /* CUDA has no shrunken/keep-map path yet; accept for ABI parity with Metal */
    if (n_expert != 256u || n_expert_used != 6u || fabsf(expert_weight_scale - 1.5f) > 1.0e-6f) return 0;
    if (!selected || !weights || !probs || !logits || !tokens || !model_map || n_tokens == 0 ||
        n_expert_groups > 1u || n_group_used > 0u ||
        logits->bytes < (uint64_t)n_tokens * 256u * sizeof(float) ||
        probs->bytes < (uint64_t)n_tokens * 256u * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * 6u * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * 6u * sizeof(float)) {
        return 0;
    }
    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (has_bias && !hash_mode) {
        if (bias_offset > model_size || model_size - bias_offset < 256u * sizeof(float)) return 0;
        bias = (const float *)cuda_model_range_ptr(model_map, bias_offset, 256u * sizeof(float), "router_bias");
        if (!bias) return 0;
    }
    if (hash_mode) {
        const uint64_t hash_bytes = (uint64_t)hash_rows * 6u * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset) return 0;
        hash = (const int32_t *)cuda_model_range_ptr(model_map, hash_offset, hash_bytes, "router_hash");
        if (!hash) return 0;
    }
    if (1 &&
        1) {
        dim3 block(32, 4, 1);
        router_select_warp_topk_kernel<<<(n_tokens + 3u) / 4u, block>>>((int32_t *)selected->ptr,
                                                                        (float *)weights->ptr,
                                                                        (float *)probs->ptr,
                                                                        bias,
                                                                        hash,
                                                                        (const float *)logits->ptr,
                                                                        (const int32_t *)tokens->ptr,
                                                                        0,
                                                                        hash_rows,
                                                                        n_tokens,
                                                                        has_bias && !hash_mode,
                                                                        hash_mode);
    } else if (1) {
        router_select_parallel_kernel<<<n_tokens, 256>>>((int32_t *)selected->ptr,
                                                         (float *)weights->ptr,
                                                         (float *)probs->ptr,
                                                         bias,
                                                         hash,
                                                         (const float *)logits->ptr,
                                                         (const int32_t *)tokens->ptr,
                                                         0,
                                                         hash_rows,
                                                         n_tokens,
                                                         has_bias && !hash_mode,
                                                         hash_mode);
    } else {
        router_select_kernel<<<n_tokens, 1>>>((int32_t *)selected->ptr,
                                              (float *)weights->ptr,
                                              (float *)probs->ptr,
                                              bias,
                                              hash,
                                              (const float *)logits->ptr,
                                              (const int32_t *)tokens->ptr,
                                              0,
                                              hash_rows,
                                              n_tokens,
                                              has_bias && !hash_mode,
                                              hash_mode);
    }
    return cuda_ok(cudaGetLastError(), "router_select launch");
}

__device__ static float dev_f16_to_f32(uint16_t v) {
    return __half2float(*reinterpret_cast<const __half *>(&v));
}

__device__ __forceinline__ static uint32_t dev_unpack_iq2_signs(uint32_t v) {
    const uint32_t p = __popc(v) & 1u;
    const uint32_t s = v ^ (p << 7u);
    return s * 0x01010101u;
}

__device__ __forceinline__ static int32_t dev_iq2_dp4a_8(uint64_t grid, uint32_t sign, const int8_t *q8, int32_t acc) {
    const uint32_t signs = dev_unpack_iq2_signs(sign);
    const int32_t sm0 = __vcmpne4(signs & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(signs & 0x80402010u, 0);
    const int32_t g0 = __vsub4((int32_t)(uint32_t)grid ^ sm0, sm0);
    const int32_t g1 = __vsub4((int32_t)(uint32_t)(grid >> 32) ^ sm1, sm1);
    acc = __dp4a(g0, *(const int32_t *)(q8 + 0), acc);
    acc = __dp4a(g1, *(const int32_t *)(q8 + 4), acc);
    return acc;
}

__device__ static int32_t dev_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 16; i += 4) {
        const int32_t v = (*(const int32_t *)(q2 + i) >> shift) & 0x03030303;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static int32_t dev_dot_iq2_pair_16(uint8_t grid0, uint32_t sign0, uint8_t grid1, uint32_t sign1, const int8_t *q8) {
    int32_t sum = 0;
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid0], cuda_ksigns_iq2xs[sign0], q8, sum);
    sum = dev_iq2_dp4a_8(cuda_iq2xxs_grid[grid1], cuda_ksigns_iq2xs[sign1], q8 + 8, sum);
    return sum;
}

__device__ __forceinline__ static void dev_iq2_i8x8_lut(
        const uint64_t *grid,
        const uint8_t *signs,
        uint8_t grid_idx,
        uint32_t sign_idx,
        int32_t *w0,
        int32_t *w1) {
    const uint32_t s = dev_unpack_iq2_signs(signs[sign_idx]);
    const int32_t sm0 = __vcmpne4(s & 0x08040201u, 0);
    const int32_t sm1 = __vcmpne4(s & 0x80402010u, 0);
    const uint64_t g = grid[grid_idx];
    *w0 = __vsub4((int32_t)(uint32_t)g ^ sm0, sm0);
    *w1 = __vsub4((int32_t)(uint32_t)(g >> 32) ^ sm1, sm1);
}

__device__ static float dev_dot_iq2_xxs_q8_K_block_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y,
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        int32_t sumi = 0;
        sumi = __dp4a(w[0], *(const int32_t *)(q8 + ib32 * 32u + 0),  sumi);
        sumi = __dp4a(w[1], *(const int32_t *)(q8 + ib32 * 32u + 4),  sumi);
        sumi = __dp4a(w[2], *(const int32_t *)(q8 + ib32 * 32u + 8),  sumi);
        sumi = __dp4a(w[3], *(const int32_t *)(q8 + ib32 * 32u + 12), sumi);
        sumi = __dp4a(w[4], *(const int32_t *)(q8 + ib32 * 32u + 16), sumi);
        sumi = __dp4a(w[5], *(const int32_t *)(q8 + ib32 * 32u + 20), sumi);
        sumi = __dp4a(w[6], *(const int32_t *)(q8 + ib32 * 32u + 24), sumi);
        sumi = __dp4a(w[7], *(const int32_t *)(q8 + ib32 * 32u + 28), sumi);
        bsum += sumi * ls;
    }
    return 0.125f * xd * y->d * (float)bsum;
}

__device__ static float dev_dot_iq2_xxs_q8_K_block(const cuda_block_iq2_xxs *x, const cuda_block_q8_K *y) {
    const float d = dev_f16_to_f32(x->d) * y->d;
    const uint16_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        int32_t sumi = 0;
        sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8);
        q8 += 16;
        sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8);
        q8 += 16;
        bsum += sumi * (int32_t)ls;
    }
    return 0.125f * d * (float)bsum;
}

__device__ static void dev_dot_iq2_xxs_q8_K_block8_deq_lut(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8],
        const uint64_t *grid,
        const uint8_t *signs) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL, y3 ? y3->qs : NULL,
        y4 ? y4->qs : NULL, y5 ? y5->qs : NULL, y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const int32_t ls = (int32_t)(2u * (aux1 >> 28) + 1u);
        int32_t w[8];
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)(aux0 & 0xffu),           (aux1 >> 0)  & 127u, &w[0], &w[1]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 8)  & 0xffu),   (aux1 >> 7)  & 127u, &w[2], &w[3]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 16) & 0xffu),   (aux1 >> 14) & 127u, &w[4], &w[5]);
        dev_iq2_i8x8_lut(grid, signs, (uint8_t)((aux0 >> 24) & 0xffu),   (aux1 >> 21) & 127u, &w[6], &w[7]);
        for (uint32_t p = 0; p < n; p++) {
            const int8_t *q = q8[p] + ib32 * 32;
            int32_t sumi = 0;
            sumi = __dp4a(w[0], *(const int32_t *)(q + 0),  sumi);
            sumi = __dp4a(w[1], *(const int32_t *)(q + 4),  sumi);
            sumi = __dp4a(w[2], *(const int32_t *)(q + 8),  sumi);
            sumi = __dp4a(w[3], *(const int32_t *)(q + 12), sumi);
            sumi = __dp4a(w[4], *(const int32_t *)(q + 16), sumi);
            sumi = __dp4a(w[5], *(const int32_t *)(q + 20), sumi);
            sumi = __dp4a(w[6], *(const int32_t *)(q + 24), sumi);
            sumi = __dp4a(w[7], *(const int32_t *)(q + 28), sumi);
            bsum[p] += sumi * ls;
        }
    }
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static void dev_dot_iq2_xxs_q8_K_block4(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[4] = {0, 0, 0, 0};
    const int8_t *q8[4] = {
        y0 ? y0->qs : NULL,
        y1 ? y1->qs : NULL,
        y2 ? y2->qs : NULL,
        y3 ? y3->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static DS4_CUDA_UNUSED void dev_dot_iq2_xxs_q8_K_block8(
        const cuda_block_iq2_xxs *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const float xd = dev_f16_to_f32(x->d);
    const uint16_t *q2 = x->qs;
    int32_t bsum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int8_t *q8[8] = {
        y0 ? y0->qs : NULL, y1 ? y1->qs : NULL, y2 ? y2->qs : NULL, y3 ? y3->qs : NULL,
        y4 ? y4->qs : NULL, y5 ? y5->qs : NULL, y6 ? y6->qs : NULL, y7 ? y7->qs : NULL,
    };
    for (int ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
        const uint32_t aux0 = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
        const uint32_t aux1 = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
        q2 += 4;
        const uint32_t ls = 2u * (aux1 >> 28) + 1u;
        const uint8_t a0 = (uint8_t)(aux0 & 0xffu);
        const uint8_t a1 = (uint8_t)((aux0 >> 8) & 0xffu);
        const uint8_t a2 = (uint8_t)((aux0 >> 16) & 0xffu);
        const uint8_t a3 = (uint8_t)((aux0 >> 24) & 0xffu);
        for (uint32_t p = 0; p < n; p++) {
            int32_t sumi = 0;
            sumi += dev_dot_iq2_pair_16(a0, (aux1 >> 0) & 127u, a1, (aux1 >> 7) & 127u, q8[p] + ib32 * 32);
            sumi += dev_dot_iq2_pair_16(a2, (aux1 >> 14) & 127u, a3, (aux1 >> 21) & 127u, q8[p] + ib32 * 32 + 16);
            bsum[p] += sumi * (int32_t)ls;
        }
    }
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    for (uint32_t p = 0; p < n; p++) acc[p] += 0.125f * xd * ys[p]->d * (float)bsum[p];
}

__device__ static void dev_q4_K_get_scale_min(
        uint32_t j,
        const uint8_t *scales,
        uint8_t *d_out,
        uint8_t *m_out) {
    if (j < 4u) {
        *d_out = scales[j] & 63u;
        *m_out = scales[j + 4u] & 63u;
    } else {
        *d_out = (scales[j + 4u] & 0x0fu) | ((scales[j - 4u] >> 6u) << 4u);
        *m_out = (scales[j + 4u] >> 4u) | ((scales[j] >> 6u) << 4u);
    }
}

__device__ __forceinline__ static int32_t dev_dot_q4_32(const uint8_t *qs, const int8_t *q8, int shift) {
    int32_t sum = 0;
    #pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        const int32_t v = (*(const int32_t *)(qs + i) >> shift) & 0x0f0f0f0f;
        sum = __dp4a(v, *(const int32_t *)(q8 + i), sum);
    }
    return sum;
}

__device__ static float dev_dot_q4_K_q8_K_block(const cuda_block_q4_K *x, const cuda_block_q8_K *y) {
    /* 双累加链(2026-08-17 调优): 单 isum 链让 8 组 dp4a 序列首尾相接(整数加法有
     * 依赖), 拆成偶/奇两条独立链让访存与 dp4a 交错; 整数和无结合律问题, 结果逐位同。 */
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum0 = 0, isum1 = 0;
    int summs0 = 0, summs1 = 0;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 2u) {
        uint8_t sc0, m0, sc1, m1;
        dev_q4_K_get_scale_min(j, x->scales, &sc0, &m0);
        dev_q4_K_get_scale_min(j + 1u, x->scales, &sc1, &m1);
        summs0 += (int)m0 * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        summs1 += (int)m1 * (int)(y->bsums[2u * j + 2u] + y->bsums[2u * j + 3u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        isum0 += (int)sc0 * dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, 0);
        isum1 += (int)sc1 * dev_dot_q4_32(x->qs + byte_off, y->qs + (j + 1u) * 32u, 4);
    }
    return y->d * xd * (float)(isum0 + isum1) - y->d * xmin * (float)(summs0 + summs1);
}

/* 半块 dot 族(2026-08-17 第九夜): blocks≤16 的 kernel 里整块 dot 让半个 warp 空转
 * (dot 循环 b<blocks 步长 32)。每 lane 算半块(j0=0 或 4 的 4 个子块), 32 lanes 全活跃。
 * 数值: 两半各自 float 化后相加, 与整块版有末位舍入差(同类重排已过断言纪律)。 */
__device__ __forceinline__ static float dev_dot_q4_K_q8_K_block_half(
        const cuda_block_q4_K *x, const cuda_block_q8_K *y, uint32_t j0) {
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int isum0 = 0, isum1 = 0;
    int summs0 = 0, summs1 = 0;
    #pragma unroll
    for (uint32_t j = j0; j < j0 + 4u; j += 2u) {
        uint8_t sc0, m0, sc1, m1;
        dev_q4_K_get_scale_min(j, x->scales, &sc0, &m0);
        dev_q4_K_get_scale_min(j + 1u, x->scales, &sc1, &m1);
        summs0 += (int)m0 * (int)(y->bsums[2u * j] + y->bsums[2u * j + 1u]);
        summs1 += (int)m1 * (int)(y->bsums[2u * j + 2u] + y->bsums[2u * j + 3u]);
        const uint32_t byte_off = (j >> 1u) * 32u;
        isum0 += (int)sc0 * dev_dot_q4_32(x->qs + byte_off, y->qs + j * 32u, 0);
        isum1 += (int)sc1 * dev_dot_q4_32(x->qs + byte_off, y->qs + (j + 1u) * 32u, 4);
    }
    return y->d * xd * (float)(isum0 + isum1) - y->d * xmin * (float)(summs0 + summs1);
}

__device__ __forceinline__ static float dev_dot_q4_K_q8_0x8_smem_half(
        const cuda_block_q4_K *x, const int8_t *xq, const float *xs, uint32_t j0) {
    const float d = dev_f16_to_f32(x->d);
    const float dmin = dev_f16_to_f32(x->dmin);
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (uint32_t j = j0; j < j0 + 4u; j += 2u) {
        uint8_t sc0, m0, sc1, m1;
        dev_q4_K_get_scale_min(j, x->scales, &sc0, &m0);
        dev_q4_K_get_scale_min(j + 1u, x->scales, &sc1, &m1);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int8_t *q8a = xq + j * 32u;
        const int8_t *q8b = xq + (j + 1u) * 32u;
        int32_t s80 = 0, s81 = 0;
        const int32_t dot0 = dev_dot_q4_32(x->qs + byte_off, q8a, 0);
        const int32_t dot1 = dev_dot_q4_32(x->qs + byte_off, q8b, 4);
        #pragma unroll
        for (uint32_t i = 0; i < 32u; i += 4u) {
            s80 = __dp4a(0x01010101, *(const int32_t *)(q8a + i), s80);
            s81 = __dp4a(0x01010101, *(const int32_t *)(q8b + i), s81);
        }
        acc0 += xs[j] * (d * (float)sc0 * (float)dot0 - dmin * (float)m0 * (float)s80);
        acc1 += xs[j + 1u] * (d * (float)sc1 * (float)dot1 - dmin * (float)m1 * (float)s81);
    }
    return acc0 + acc1;
}

/* q4_K 权重块(256 值) × q8_0 预量化激活(8 个 32 值子块, int8 qs + 每块 f32 scale)。
 * q4_K 的 8 个子 scale/min 与 q8_0 的 32 值分块天然对齐: 每子块
 * acc += xs_j × (d·sc_j·Σ(q4·q8) − dmin·m_j·Σq8)。attn_output q8→q4 移植的核心。 */
__device__ __forceinline__ static float dev_dot_q4_K_q8_0x8(
        const cuda_block_q4_K *xg, const int8_t *xq, const float *xs, int use_dp4a) {
    /* 块 144B=9×uint4 一次进寄存器(dense q4 kernel 上实测 +1.3% 的同款招),
     * 后续 scale 解包与 dp4a 全走寄存器。 */
    uint4 v[9];
    #pragma unroll
    for (int vi = 0; vi < 9; vi++) v[vi] = ((const uint4 *)xg)[vi];
    const cuda_block_q4_K *x = (const cuda_block_q4_K *)v;
    const float d = dev_f16_to_f32(x->d);
    const float dmin = dev_f16_to_f32(x->dmin);
    /* 双 float 链(奇偶子块交错), 掩盖 dp4a 依赖延迟 */
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 2u) {
        uint8_t sc0, m0, sc1, m1;
        dev_q4_K_get_scale_min(j, x->scales, &sc0, &m0);
        dev_q4_K_get_scale_min(j + 1u, x->scales, &sc1, &m1);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int8_t *q8a = xq + j * 32u;
        const int8_t *q8b = xq + (j + 1u) * 32u;
        int32_t dot0, dot1, s80 = 0, s81 = 0;
        if (use_dp4a) {
            dot0 = dev_dot_q4_32(x->qs + byte_off, q8a, 0);
            dot1 = dev_dot_q4_32(x->qs + byte_off, q8b, 4);
            #pragma unroll
            for (uint32_t i = 0; i < 32u; i += 4u) {
                s80 = __dp4a(0x01010101, *(const int32_t *)(q8a + i), s80);
                s81 = __dp4a(0x01010101, *(const int32_t *)(q8b + i), s81);
            }
        } else {
            dot0 = dot1 = 0;
            #pragma unroll
            for (uint32_t i = 0; i < 32u; i++) {
                dot0 += ((x->qs[byte_off + i] >> 0) & 0xF) * q8a[i]; s80 += q8a[i];
                dot1 += ((x->qs[byte_off + i] >> 4) & 0xF) * q8b[i]; s81 += q8b[i];
            }
        }
        acc0 += xs[j] * (d * (float)sc0 * (float)dot0 - dmin * (float)m0 * (float)s80);
        acc1 += xs[j + 1u] * (d * (float)sc1 * (float)dot1 - dmin * (float)m1 * (float)s81);
    }
    return acc0 + acc1;
}

/* shared 源专用变体(2026-08-17): 数据已在片上, v[9] 寄存器整取无意义; 砍掉非 dp4a
 * 标量路径(寄存器按最坏路径分配, 把宿主 kernel 顶到 REG:121/127=2 块/SM 的元凶)。
 * 数学与 dev_dot_q4_K_q8_0x8 的 dp4a 路径逐位同义。 */
__device__ __forceinline__ static float dev_dot_q4_K_q8_0x8_smem(
        const cuda_block_q4_K *x, const int8_t *xq, const float *xs) {
    const float d = dev_f16_to_f32(x->d);
    const float dmin = dev_f16_to_f32(x->dmin);
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; j += 2u) {
        uint8_t sc0, m0, sc1, m1;
        dev_q4_K_get_scale_min(j, x->scales, &sc0, &m0);
        dev_q4_K_get_scale_min(j + 1u, x->scales, &sc1, &m1);
        const uint32_t byte_off = (j >> 1u) * 32u;
        const int8_t *q8a = xq + j * 32u;
        const int8_t *q8b = xq + (j + 1u) * 32u;
        int32_t s80 = 0, s81 = 0;
        const int32_t dot0 = dev_dot_q4_32(x->qs + byte_off, q8a, 0);
        const int32_t dot1 = dev_dot_q4_32(x->qs + byte_off, q8b, 4);
        #pragma unroll
        for (uint32_t i = 0; i < 32u; i += 4u) {
            s80 = __dp4a(0x01010101, *(const int32_t *)(q8a + i), s80);
            s81 = __dp4a(0x01010101, *(const int32_t *)(q8b + i), s81);
        }
        acc0 += xs[j] * (d * (float)sc0 * (float)dot0 - dmin * (float)m0 * (float)s80);
        acc1 += xs[j + 1u] * (d * (float)sc1 * (float)dot1 - dmin * (float)m1 * (float)s81);
    }
    return acc0 + acc1;
}

/* dp4a+staged 特化版: 无 use_dp4a/直读分支 ⇒ 寄存器回落, SM 驻留翻倍 */
__global__ static void grouped_q4_K_a_preq_warp8_dp4a_kernel(
        float *low,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t kblocks) {
    const uint32_t lane = threadIdx.x & 15u;
    const uint32_t slot = threadIdx.x >> 4u;
    const uint64_t row = (uint64_t)blockIdx.x * 16u + slot;
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim || tok >= n_tokens) return;
    const uint64_t group = row / rank;
    const uint64_t xrow = tok * (uint64_t)n_groups + group;
    const int8_t *xqr = xq + xrow * kblocks * 256u;
    const float *xsr = xscale + xrow * kblocks * 8u;
    extern __shared__ uint4 dstage_a[];
    const uint4 *src16 = (const uint4 *)(w + row * kblocks * sizeof(cuda_block_q4_K));
    const uint32_t n16 = (uint32_t)kblocks * 9u;
    uint4 *my = dstage_a + (uint64_t)slot * n16;
    for (uint32_t i = lane; i < n16; i += 16u) my[i] = __ldcs(src16 + i);
    __syncwarp();
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)my;
    float acc = 0.0f;
    for (uint64_t b = lane; b < kblocks; b += 16u)
        acc += dev_dot_q4_K_q8_0x8_smem(wr + b, xqr + b * 256u, xsr + b * 8u);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    if (lane == 0) low[tok * low_dim + row] = acc;
}

__global__ static void matmul_q4_K_hc_expand_preq_warp8_dp4a_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_embd,
        uint32_t n_hc,
        uint64_t kblocks,
        int has_add) {
    /* epilogue 与原版逐字同义 */
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    if (row >= out_dim) return;
    float acc = 0.0f;
    extern __shared__ uint4 dstage_b[];
    const uint32_t seg = (kblocks > 16u) ? 16u : (uint32_t)kblocks;
    const uint32_t n16 = seg * 9u;
    uint4 *my = dstage_b + (uint64_t)warp * n16;
    for (uint32_t b0 = 0; b0 < (uint32_t)kblocks; b0 += seg) {
        const uint32_t nb = ((uint32_t)kblocks - b0 < seg) ? ((uint32_t)kblocks - b0) : seg;
        const uint4 *src16 = (const uint4 *)(w + row * kblocks * sizeof(cuda_block_q4_K) + (uint64_t)b0 * sizeof(cuda_block_q4_K));
        for (uint32_t i = lane; i < nb * 9u; i += 32u) my[i] = __ldcs(src16 + i);
        __syncwarp();
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)my;
        /* 半块拆分: 16 块×2 半块=32 单元, 全 warp 活跃(原 b<nb 步长 32 半 warp 空转) */
        const uint32_t bi = lane >> 1u, hj = (lane & 1u) * 4u;
        if (bi < nb)
            acc += dev_dot_q4_K_q8_0x8_smem_half(wr + bi, xq + (uint64_t)(b0 + bi) * 256u, xscale + (uint64_t)(b0 + bi) * 8u, hj);
        __syncwarp();
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        float block_v = acc;
        if (has_add) block_v += block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}

__global__ static void grouped_q4_K_a_preq_warp8_kernel(
        float *low,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t group_dim,
        uint64_t rank,
        uint32_t n_groups,
        uint32_t n_tokens,
        uint64_t kblocks,          /* group_dim/256 */
        int use_dp4a) {
    /* 16 lanes/行(2 行/warp): group_dim=4096 时 kblocks=16, 恰一 lane 一块 —
     * 32-lane 版半数 lane 空转(该 kernel 实测 139GB/s=63%)。 */
    const uint32_t lane = threadIdx.x & 15u;
    const uint32_t slot = threadIdx.x >> 4u;   /* 0..15 行槽 */
    const uint64_t row = (uint64_t)blockIdx.x * 16u + slot;
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim || tok >= n_tokens) return;
    const uint64_t group = row / rank;
    const uint64_t xrow = tok * (uint64_t)n_groups + group;
    const int8_t *xqr = xq + xrow * kblocks * 256u;
    const float *xsr = xscale + xrow * kblocks * 8u;
    float acc = 0.0f;
    if (kblocks <= 16u) {   /* 半warp 协作 staging(动态 shared: 静态36KB钉死SM驻留2块) */
        extern __shared__ uint4 dstage_a[];
        const uint4 *src16 = (const uint4 *)(w + row * kblocks * sizeof(cuda_block_q4_K));
        const uint32_t n16 = (uint32_t)kblocks * 9u;
        uint4 *my = dstage_a + (uint64_t)slot * n16;
        for (uint32_t i = lane; i < n16; i += 16u) my[i] = src16[i];
        __syncwarp();
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)my;
        for (uint64_t b = lane; b < kblocks; b += 16u)
            acc += dev_dot_q4_K_q8_0x8(wr + b, xqr + b * 256u, xsr + b * 8u, use_dp4a);
    } else {
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(w + row * kblocks * sizeof(cuda_block_q4_K));
        for (uint64_t b = lane; b < kblocks; b += 16u)
            acc += dev_dot_q4_K_q8_0x8(wr + b, xqr + b * 256u, xsr + b * 8u, use_dp4a);
    }
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    if (lane == 0) low[tok * low_dim + row] = acc;
}

__global__ static void matmul_q4_K_hc_expand_preq_warp8_kernel(
        float *out_hc,
        float *block_out,
        const float *block_add,
        const float *residual_hc,
        const float *split,
        const unsigned char *w,
        const int8_t *xq,
        const float *xscale,
        uint64_t in_dim,
        uint64_t out_dim,
        uint32_t n_embd,
        uint32_t n_hc,
        uint64_t kblocks,
        int has_add,
        int use_dp4a) {
    /* epilogue 与 matmul_q8_0_hc_expand_preq_warp8_kernel 逐字同义(post/comb 矩阵 +
     * has_add), 仅权重 dot 换 q4_K×q8_0x8。 */
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    if (row >= out_dim) return;
    float acc = 0.0f;
    if (kblocks <= 32u) {   /* warp 协作 staging(动态 shared, 同 dense q4 之药)。
         * 半行两段(2026-08-17): kblocks=32 时整行 staging 要 36KB shared → 2 块/SM
         * (160 GB/s 卡点); 按 16 块一段分两段搬+算, shared 减半 → 4 块/SM。 */
        extern __shared__ uint4 dstage_b[];
        const uint32_t seg = (kblocks > 16u) ? 16u : (uint32_t)kblocks;
        const uint32_t n16 = seg * 9u;
        uint4 *my = dstage_b + (uint64_t)warp * n16;
        for (uint32_t b0 = 0; b0 < (uint32_t)kblocks; b0 += seg) {
            const uint32_t nb = ((uint32_t)kblocks - b0 < seg) ? ((uint32_t)kblocks - b0) : seg;
            const uint4 *src16 = (const uint4 *)(w + row * kblocks * sizeof(cuda_block_q4_K) + (uint64_t)b0 * sizeof(cuda_block_q4_K));
            for (uint32_t i = lane; i < nb * 9u; i += 32u) my[i] = src16[i];
            __syncwarp();
            const cuda_block_q4_K *wr = (const cuda_block_q4_K *)my;
            for (uint32_t b = lane; b < nb; b += 32u)
                acc += dev_dot_q4_K_q8_0x8(wr + b, xq + (uint64_t)(b0 + b) * 256u, xscale + (uint64_t)(b0 + b) * 8u, use_dp4a);
            __syncwarp();
        }
    } else {
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(w + row * kblocks * sizeof(cuda_block_q4_K));
        for (uint64_t b = lane; b < kblocks; b += 32u)
            acc += dev_dot_q4_K_q8_0x8(wr + b, xq + b * 256u, xscale + b * 8u, use_dp4a);
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        const uint32_t d = (uint32_t)row;
        block_out[d] = acc;
        float block_v = acc;
        if (has_add) block_v += block_add[d];
        const float *post = split + n_hc;
        const float *comb = split + 2u * n_hc;
        for (uint32_t dst_hc = 0; dst_hc < n_hc; dst_hc++) {
            float hc_acc = block_v * post[dst_hc];
            for (uint32_t src_hc = 0; src_hc < n_hc; src_hc++) {
                const float comb_v = comb[dst_hc + (uint64_t)src_hc * n_hc];
                const float res_v = residual_hc[(uint64_t)src_hc * n_embd + d];
                hc_acc += comb_v * res_v;
            }
            out_hc[(uint64_t)dst_hc * n_embd + d] = hc_acc;
        }
    }
}

__device__ static float dev_dot_q2_K_q8_K_block(const cuda_block_q2_K *x, const cuda_block_q8_K *y) {
    const uint8_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    const uint8_t *sc = x->scales;
    int summs = 0;
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    /* 双累加链: 原 isum 单链 16 组 dp4a 序列首尾相接; 拆偶/奇两链交错访存与运算。
     * 整数和无结合律问题, 结果逐位同义。 */
    int isum0 = 0, isum1 = 0;
    int is = 0;
    for (int k = 0; k < CUDA_QK_K / 128; k++) {
        int shift = 0;
        for (int j = 0; j < 4; j++) {
            const int d0 = sc[is++] & 0x0f;
            const int d1 = sc[is++] & 0x0f;
            isum0 += d0 * dev_dot_q2_16(q2, q8, shift);
            isum1 += d1 * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)(isum0 + isum1) - dmin * (float)summs;
}

/* 多 token 复用解量化(2026-08-21): 一组 16 个 2-bit 权重只解一次(4 次移位+掩码+4 次
 * shared 读), 批内 NT 个 token 各自 dp4a。批路径此前每个 token 把同一份权重重解一遍 ——
 * verify/drafter 的批就是靠这条路吃掉 ALU 的。整数点积的运算顺序与单 token 版逐位一致,
 * 浮点收尾也保持 (yd*xd)*isum 的左结合, 所以结果与 decode 路完全相同。 */
template <int NT>
__device__ static void dev_dot_q2_K_q8_K_block_smem_multi(
        const cuda_block_q2_K *x, const cuda_block_q8_K *y0, uint64_t ystride, float *acc) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    int summs[NT], isum0[NT], isum1[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) { summs[t] = 0; isum0[t] = 0; isum1[t] = 0; }
    #pragma unroll 4
    for (int j = 0; j < 16; j++) {
        const int m = sc[j] >> 4;
        #pragma unroll
        for (int t = 0; t < NT; t++) summs[t] += (y0 + (uint64_t)t * ystride)->bsums[j] * m;
    }
    const uint8_t *q2 = x->qs;
    int is = 0;
    #pragma unroll 1
    for (int k = 0; k < CUDA_QK_K / 128; k++) {
        int shift = 0;
        #pragma unroll 1
        for (int j = 0; j < 4; j++) {
            const int d0 = sc[is++] & 0x0f;
            const int d1 = sc[is++] & 0x0f;
            int32_t w0[4], w1[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                w0[i] = (*(const int32_t *)(q2 + i * 4) >> shift) & 0x03030303;
                w1[i] = (*(const int32_t *)(q2 + 16 + i * 4) >> shift) & 0x03030303;
            }
            #pragma unroll
            for (int t = 0; t < NT; t++) {
                const int8_t *q8 = (y0 + (uint64_t)t * ystride)->qs + k * 128 + j * 32;
                int a = 0, b = 0;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    a = __dp4a(w0[i], *(const int32_t *)(q8 + i * 4), a);
                    b = __dp4a(w1[i], *(const int32_t *)(q8 + 16 + i * 4), b);
                }
                isum0[t] += d0 * a;
                isum1[t] += d1 * b;
            }
            shift += 2;
        }
        q2 += 32;
    }
    #pragma unroll
    for (int t = 0; t < NT; t++) {
        const float yd = (y0 + (uint64_t)t * ystride)->d;
        acc[t] += yd * xd * (float)(isum0[t] + isum1[t]) - yd * xmin * (float)summs[t];
    }
}

/* shared 专用低寄存器变体: 与 dev_dot_q2_K_q8_K_block 逐位同义, 仅禁循环展开。
 * 片上读延迟低无需深展开藏延迟; 全展开版把 gateup kernel 顶到 REG:128(occupancy 33%墙)。 */
__device__ static float dev_dot_q2_K_q8_K_block_smem(const cuda_block_q2_K *x, const cuda_block_q8_K *y) {
    const uint8_t *q2 = x->qs;
    const int8_t *q8 = y->qs;
    const uint8_t *sc = x->scales;
    int summs = 0;
    #pragma unroll 4
    for (int j = 0; j < 16; j++) summs += y->bsums[j] * (sc[j] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    int isum0 = 0, isum1 = 0;
    int is = 0;
    #pragma unroll 1
    for (int k = 0; k < CUDA_QK_K / 128; k++) {
        int shift = 0;
        #pragma unroll 1
        for (int j = 0; j < 4; j++) {
            const int d0 = sc[is++] & 0x0f;
            const int d1 = sc[is++] & 0x0f;
            isum0 += d0 * dev_dot_q2_16(q2, q8, shift);
            isum1 += d1 * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }
    return dall * (float)(isum0 + isum1) - dmin * (float)summs;
}

/* 半块变体(2026-08-20 dense decode 占用率手术): 一 lane 算 128 权重(h=0/1 取块的前/后半)。
 * q2_K 块 = 2 个 k 迭代 × 128 权重, bsums/scales 按 8 组界干净切分; 整数域逐位同义,
 * 浮点只差 dall/dmin 合帐次序(与 q4 half 变体同级容差)。 */
__device__ __forceinline__ static float dev_dot_q2_K_q8_K_block_half(
        const cuda_block_q2_K *x, const cuda_block_q8_K *y, uint32_t h) {
    const uint8_t *q2 = x->qs + h * 32u;
    const int8_t *q8 = y->qs + h * 128u;
    const uint8_t *sc = x->scales + h * 8u;
    int summs = 0;
    for (int j = 0; j < 8; j++) summs += y->bsums[h * 8u + j] * (sc[j] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    int isum0 = 0, isum1 = 0;
    int is = 0, shift = 0;
    for (int j = 0; j < 4; j++) {
        const int d0 = sc[is++] & 0x0f;
        const int d1 = sc[is++] & 0x0f;
        isum0 += d0 * dev_dot_q2_16(q2, q8, shift);
        isum1 += d1 * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
        shift += 2;
        q8 += 32;
    }
    return dall * (float)(isum0 + isum1) - dmin * (float)summs;
}

/* 四分之一块变体(2026-08-20 刀⑤): 一 lane 64 权重(qi∈0..3 = k半×k内前后64)。
 * blocks=8 时 8块×4=32 lane 全活(半块拆分只活16)。scale/bsums 按4组界干净切分。 */
__device__ __forceinline__ static float dev_dot_q2_K_q8_K_block_quarter(
        const cuda_block_q2_K *x, const cuda_block_q8_K *y, uint32_t qi) {
    const uint32_t k = qi >> 1u, jh = qi & 1u;
    const uint8_t *q2 = x->qs + k * 32u;
    const int8_t *q8 = y->qs + k * 128u + jh * 64u;
    const uint8_t *sc = x->scales + k * 8u + jh * 4u;
    int summs = 0;
    for (int j = 0; j < 4; j++) summs += y->bsums[k * 8u + jh * 4u + j] * (sc[j] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    int isum0 = 0, isum1 = 0;
    int shift = (int)(jh * 4u);
    for (int j = 0; j < 2; j++) {
        const int d0 = sc[j * 2] & 0x0f;
        const int d1 = sc[j * 2 + 1] & 0x0f;
        isum0 += d0 * dev_dot_q2_16(q2, q8, shift);
        isum1 += d1 * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
        shift += 2;
        q8 += 32;
    }
    return dall * (float)(isum0 + isum1) - dmin * (float)summs;
}

/* 八分之一块变体(2026-08-20 刀⑦): 一 lane 32 权重 = 全量循环固定(k=qi>>2, j=qi&3)一步。
 * blocks=4 时 4块×8=32 lane 全活(半块拆分只活8)。 */
__device__ __forceinline__ static float dev_dot_q2_K_q8_K_block_eighth(
        const cuda_block_q2_K *x, const cuda_block_q8_K *y, uint32_t qi) {
    const uint32_t k = qi >> 2u, j = qi & 3u;
    const uint8_t *q2 = x->qs + k * 32u;
    const int8_t *q8 = y->qs + k * 128u + j * 32u;
    const uint8_t *sc = x->scales + k * 8u + j * 2u;
    const int summs = y->bsums[k * 8u + j * 2u] * (sc[0] >> 4)
                    + y->bsums[k * 8u + j * 2u + 1u] * (sc[1] >> 4);
    const float dall = y->d * dev_f16_to_f32(x->d);
    const float dmin = y->d * dev_f16_to_f32(x->dmin);
    const int shift = (int)(j * 2u);
    const int isum = (sc[0] & 0x0f) * dev_dot_q2_16(q2, q8, shift)
                   + (sc[1] & 0x0f) * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
    return dall * (float)isum - dmin * (float)summs;
}

__device__ static void dev_dot_q2_K_q8_K_block4(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        uint32_t n,
        float acc[4]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[4] = { y0, y1, y2, y3 };
    int isum[4] = {0, 0, 0, 0};
    int summs[4] = {0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static void dev_dot_q2_K_q8_K_block8(
        const cuda_block_q2_K *x,
        const cuda_block_q8_K *y0,
        const cuda_block_q8_K *y1,
        const cuda_block_q8_K *y2,
        const cuda_block_q8_K *y3,
        const cuda_block_q8_K *y4,
        const cuda_block_q8_K *y5,
        const cuda_block_q8_K *y6,
        const cuda_block_q8_K *y7,
        uint32_t n,
        float acc[8]) {
    const uint8_t *sc = x->scales;
    const float xd = dev_f16_to_f32(x->d);
    const float xmin = dev_f16_to_f32(x->dmin);
    const cuda_block_q8_K *ys[8] = { y0, y1, y2, y3, y4, y5, y6, y7 };
    int isum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int summs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t p = 0; p < n; p++) {
        for (int j = 0; j < 16; j++) summs[p] += ys[p]->bsums[j] * (sc[j] >> 4);
    }
    for (uint32_t p = 0; p < n; p++) {
        const uint8_t *q2 = x->qs;
        const int8_t *q8 = ys[p]->qs;
        int is = 0;
        for (int k = 0; k < CUDA_QK_K / 128; k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2, q8, shift);
                d = sc[is++] & 0x0f;
                isum[p] += d * dev_dot_q2_16(q2 + 16, q8 + 16, shift);
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
    }
    for (uint32_t p = 0; p < n; p++) {
        const float yd = ys[p]->d;
        acc[p] += yd * xd * (float)isum[p] - yd * xmin * (float)summs[p];
    }
}

__device__ static float half_warp_sum_f32(float v, uint32_t lane16) {
    uint32_t mask = 0xffffu << (threadIdx.x & 16u);
    for (int offset = 8; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 16);
    }
    (void)lane16;
    return v;
}

__device__ static float quarter_warp_sum_f32(float v, uint32_t lane8) {
    uint32_t mask = 0xffu << (threadIdx.x & 24u);
    for (int offset = 4; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, 8);
    }
    (void)lane8;
    return v;
}

__global__ static void q8_K_quantize_kernel(cuda_block_q8_K *out, const float *x, uint32_t in_dim, uint32_t n_rows) {
    uint32_t b = blockIdx.x;
    uint32_t row = blockIdx.y;
    if (row >= n_rows || b >= in_dim / CUDA_QK_K) return;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * CUDA_QK_K;
    cuda_block_q8_K *yb = out + (uint64_t)row * (in_dim / CUDA_QK_K) + b;
    __shared__ float abs_part[256];
    __shared__ float val_part[256];
    __shared__ float maxv_s;
    __shared__ float iscale_s;
    uint32_t tid = threadIdx.x;
    float v = tid < CUDA_QK_K ? xr[tid] : 0.0f;
    abs_part[tid] = tid < CUDA_QK_K ? fabsf(v) : 0.0f;
    val_part[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && abs_part[tid + stride] > abs_part[tid]) {
            abs_part[tid] = abs_part[tid + stride];
            val_part[tid] = val_part[tid + stride];
        }
        __syncthreads();
    }
    float amax = abs_part[0];
    if (amax == 0.0f) {
        if (tid == 0) yb->d = 0.0f;
        if (tid < CUDA_QK_K) yb->qs[tid] = 0;
        if (tid < CUDA_QK_K / 16) yb->bsums[tid] = 0;
        return;
    }
    if (tid == 0) {
        maxv_s = val_part[0];
        iscale_s = -127.0f / maxv_s;
    }
    __syncthreads();
    if (tid < CUDA_QK_K) {
        int qv = (int)lrintf(iscale_s * xr[tid]);
        if (qv > 127) qv = 127;
        if (qv < -128) qv = -128;
        yb->qs[tid] = (int8_t)qv;
    }
    __syncthreads();
    if (tid < CUDA_QK_K / 16) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += yb->qs[tid * 16 + i];
        yb->bsums[tid] = (int16_t)sum;
    }
    if (tid == 0) yb->d = 1.0f / iscale_s;
}

/* ---- dense Q4_K matmul (backbone-q4k 配方) ----
 * 引擎的 dense 路径此前只有 q8_0/f16/f32; Q4_K 只在 routed 专家里被支持。
 * backbone(q/kv 投影 + shared + 输出头)换 Q4_K 是把每 token 读取量从 10.3GB 压到
 * ~7.9GB 的关键(decode 带宽墙 26→34 t/s)。核心零件全现成: dev_dot_q4_K_q8_K_block
 * (routed q4k 路在用) + q8_K_quantize_kernel。激活 scratch 用专属 grow-only 显存,
 * 不碰共享 tmp арena(那东西已在 ntok>1 路径的嫌疑名单上)。 */
static void *g_q4k_xq_sc = NULL;
static float *g_down_partial = NULL;   /* [6][out_dim] per-expert 部分和 */
static uint64_t g_down_partial_bytes = 0;
static uint64_t g_q4k_xq_bytes = 0;

/* 向量化取块版 q4_K×q8_K dot: q4_K 块 144B = 9×16B 且行内偏移恒为 16 的倍数
 * (144, 2304 均整除 16) —— 用 uint4 整取进寄存器再算, 取代原 dot 的逐字节 load。
 * 数学与 dev_dot_q4_K_q8_K_block 逐位相同。 */
__device__ __forceinline__ static float dev_dot_q4_K_q8_K_block_vec(
        const cuda_block_q4_K *xg, const cuda_block_q8_K *y) {
    uint4 v[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) v[i] = ((const uint4 *)xg)[i];
    const cuda_block_q4_K *x = (const cuda_block_q4_K *)v;
    return dev_dot_q4_K_q8_K_block(x, y);
}

/* 16-lane 变体: blocks<=16 的小矩阵(attn q/kv/shared, L2 驻留)一 lane 一块零空转;
 * 大矩阵(logits 等)仍走 32-lane 版(A/B: 32-lane 对 DRAM 大矩阵最优)。 */
__global__ static DS4_CUDA_UNUSED void matmul_q4_K_warp16_kernel(
        float *out, const char *w_base, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks, uint32_t out_dim) {
    const uint32_t lane = threadIdx.x & 15u;
    const uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    const uint32_t tok = blockIdx.y;
    if (row >= out_dim) return;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(w_base + (uint64_t)row * row_bytes);
    const cuda_block_q8_K *xt = xq + (uint64_t)tok * blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < blocks; b += 16u)
        acc += dev_dot_q4_K_q8_K_block_vec(wr + b, xt + b);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    if (lane == 0) out[(uint64_t)tok * out_dim + row] = acc;
}

/* q4_K 同输入矩阵对(2026-08-17 第八夜): q_a+kv / shared gate+up 各共读同一激活行,
 * 原两次发射=2 kernel+2 quantize 节点+双份小 grid 调度气泡。合并: 行号跨两矩阵
 * 连续编址, 前段 w0 后段 w1, staging/dot 与单矩阵版同构。 */
__global__ static void matmul_q4_K_pair_warp_kernel(
        float *out0, float *out1, const char *w0, const char *w1,
        const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks, uint32_t out0_dim, uint32_t out1_dim) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t total = out0_dim + out1_dim;
    for (uint32_t rg = blockIdx.x * 8u + warp; rg < total; rg += gridDim.x * 8u) {
        const uint32_t which = (rg >= out0_dim) ? 1u : 0u;
        const uint32_t row = which ? (rg - out0_dim) : rg;
        const char *wb = which ? w1 : w0;
        float acc = 0.0f;
        if (blocks <= 32u) {
            extern __shared__ uint4 dstage[];
            const uint32_t n16 = blocks * 9u;
            const uint4 *src16 = (const uint4 *)(wb + (uint64_t)row * row_bytes);
            uint4 *my = dstage + (uint64_t)warp * n16;
            for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
            __syncwarp();
            const cuda_block_q4_K *wr = (const cuda_block_q4_K *)my;
            if (blocks <= 16u) {
                const uint32_t bi = lane >> 1u, hj = (lane & 1u) * 4u;
                if (bi < blocks) acc += dev_dot_q4_K_q8_K_block_half(wr + bi, xq + bi, hj);
            } else {
                for (uint32_t b = lane; b < blocks; b += 32u)
                    acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
            }
            __syncwarp();
        } else {
            const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(wb + (uint64_t)row * row_bytes);
            for (uint32_t b = lane; b < blocks; b += 32u)
                acc += dev_dot_q4_K_q8_K_block_vec(wr + b, xq + b);
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) (which ? out1 : out0)[row] = acc;
    }
}

/* cp.async 双缓冲 staging(2026-08-20 dot微架构刀①): 老 staging 同步搬完才算, 每行
 * 吃一次 LPDDR 全延迟(warp 数藏不住); 现算第 N 行时异步引擎流水搬 N+1 行, 延迟折叠。
 * dot 数学/lane 映射与 staged half/eighth 路同式同序 ⇒ 与现路逐位一致。 */
__global__ static void matmul_q2_K_warp_ca_kernel(
        float *out, const char *w_base, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks, uint32_t out_dim,
        uint32_t split_maxblk) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t tok = blockIdx.y;
    const cuda_block_q8_K *xt = xq + (uint64_t)tok * blocks;
    const uint32_t n16 = blocks * 84u / 16u;
    extern __shared__ uint4 q2ca[];                  /* [2][8 warps][n16] */
    uint4 *bufA = q2ca + (uint64_t)warp * n16;
    uint4 *bufB = q2ca + (uint64_t)(8u + warp) * n16;
    const uint32_t rstep = gridDim.x * 8u;
    uint32_t row = blockIdx.x * 8u + warp;
    if (row < out_dim) {
        const uint4 *src = (const uint4 *)(w_base + (uint64_t)row * row_bytes);
        for (uint32_t i = lane; i < n16; i += 32u)
            __pipeline_memcpy_async(bufA + i, src + i, 16);
    }
    __pipeline_commit();
    uint32_t cur = 0u;
    while (row < out_dim) {
        const uint32_t nrow = row + rstep;
        uint4 *bn = cur ? bufA : bufB;
        if (nrow < out_dim) {
            const uint4 *nsrc = (const uint4 *)(w_base + (uint64_t)nrow * row_bytes);
            for (uint32_t i = lane; i < n16; i += 32u)
                __pipeline_memcpy_async(bn + i, nsrc + i, 16);
        }
        __pipeline_commit();
        __pipeline_wait_prior(1);   /* 只剩 1 组在飞 = 本行已就位 */
        __syncwarp();
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(cur ? bufB : bufA);
        float acc = 0.0f;
        if (blocks == 4u && split_maxblk >= 4u) {
            const uint32_t bi = lane >> 3u, qi = lane & 7u;
            acc = dev_dot_q2_K_q8_K_block_eighth(wr + bi, xt + bi, qi);
        } else if (blocks <= 16u && blocks <= split_maxblk) {
            const uint32_t bi = lane >> 1u, h = lane & 1u;
            if (bi < blocks) acc = dev_dot_q2_K_q8_K_block_half(wr + bi, xt + bi, h);
        } else {
            for (uint32_t b = lane; b < blocks; b += 32u)
                acc += dev_dot_q2_K_q8_K_block_smem(wr + b, xt + b);
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) out[(uint64_t)tok * out_dim + row] = acc;
        __syncwarp();               /* 复写本缓冲前全 lane 必须算完 */
        row = nrow; cur ^= 1u;
    }
}

/* block 内并行 q8_K 量化(2026-08-20 碎片税刀①): 8 warp × 各承包块, warp 内 lane 持
 * 连续 8 元素。amax 选择 = "高索引挑战低索引, 严格大于才胜" 与原 256 线程树同语义
 * (全局 lowest-index argmax) ⇒ qs/d/bsums 逐位一致。写进 shared xqsh。 */
__device__ static void q2k_fused_quantize(cuda_block_q8_K *xqsh, const float *x, uint32_t blocks) {
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    for (uint32_t b = warp; b < blocks; b += 8u) {
        const float *xr = x + (uint64_t)b * 256u;
        cuda_block_q8_K *yb = xqsh + b;
        /* lane 扫连续 8 元素: 保低 index 平手 */
        float amax = 0.0f, maxv = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < 8u; i++) {
            const float v = xr[lane * 8u + i];
            const float a = fabsf(v);
            if (a > amax) { amax = a; maxv = v; }
        }
        /* shuffle 树: 高 lane 挑战低 lane, 严格大于才胜 */
        for (int o = 16; o; o >>= 1) {
            const float oa = __shfl_down_sync(0xffffffffu, amax, o);
            const float ov = __shfl_down_sync(0xffffffffu, maxv, o);
            if (oa > amax) { amax = oa; maxv = ov; }
        }
        amax = __shfl_sync(0xffffffffu, amax, 0);
        maxv = __shfl_sync(0xffffffffu, maxv, 0);
        if (amax == 0.0f) {
            if (lane == 0) yb->d = 0.0f;
            #pragma unroll
            for (uint32_t i = 0; i < 8u; i++) yb->qs[lane * 8u + i] = 0;
            if (lane < 16u) yb->bsums[lane] = 0;
            continue;
        }
        const float iscale = -127.0f / maxv;
        int psum = 0;   /* lane 的 8 元素恰是半个 bsums 组(16 元素=2 lanes) */
        #pragma unroll
        for (uint32_t i = 0; i < 8u; i++) {
            int qv = (int)lrintf(iscale * xr[lane * 8u + i]);
            if (qv > 127) qv = 127;
            if (qv < -128) qv = -128;
            yb->qs[lane * 8u + i] = (int8_t)qv;
            psum += qv;
        }
        const int osum = __shfl_down_sync(0xffffffffu, psum, 1);
        if ((lane & 1u) == 0u && lane < 32u) yb->bsums[lane >> 1] = (int16_t)(psum + osum);
        if (lane == 0) yb->d = 1.0f / iscale;
    }
}

template <uint32_t NB>   /* 施工日3特化刀: blocks 编译常量(NB>0)→全展开+静态寻址
                            (阶梯常量核228 vs 真核动态173 的最后差异嫌疑); NB=0 动态。 */
__global__ static void __launch_bounds__(256, DS4_Q2K_BLOCKS_PER_SM) matmul_q2_K_warp_kernel(
        float *out, const char *w_base, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks_dyn, uint32_t out_dim, uint32_t q2k_stage_lv,
        uint32_t q2k_split_maxblk, const float *xraw, uint32_t n_tok_batch, uint32_t n_tok_total,
        float *partial, uint32_t ksplit, uint32_t x_in_smem) {
    const uint32_t blocks = NB ? NB : blocks_dyn;
    /* dense Q2_K 占用率手术(2026-08-20)+量化融合(碎片税刀①): xraw!=NULL(decode n=1)
     * 时首段 block 内量化到 shared, 免独立 quantize 发射(414发/tok 的大头)。
     * staging: warp 协作 uint4(blocks%4==0)+半块/八分拆。 */
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    /* 权重驻留批(2026-08-21 verify 8x 偏离根因): n_tok_batch>1 时 grid.y=1,
     * 块内循环 token 复用已 staged 的权重行 —— 此前 grid.y=n_tok 让同一份权重被
     * 每个 token 各读一遍(6 tok=6x 带宽)。逐 (row,tok) 数学与旧路逐位相同。 */
    /* token 分组(2026-08-21): grid.y = 组数, 每块吃 n_tok_batch 个 token。
     * 全驻留(组=1)省带宽但并行度只剩 gx 块 —— 小 out_dim 会饿死(o_b 实测 54GB/s vs
     * down tile 同形状 151GB/s, 差在 warp 数 3.75x)。分组让两者可调和。 */
    const uint32_t tpg = n_tok_batch ? n_tok_batch : 1u;
    const uint32_t ntot = n_tok_total ? n_tok_total : tpg;
    const uint32_t tok_base = blockIdx.y * tpg;
    if (tok_base >= ntot) return;
    const uint32_t ntb = (tpg < ntot - tok_base) ? tpg : (ntot - tok_base);
    const uint32_t stage_ok = (blocks <= 32u && (blocks & 3u) == 0u) && (q2k_stage_lv > 0u);
    const uint32_t n16 = blocks * 84u / 16u;   /* 行 uint4 数(blocks%4==0 ⇒ 整) */
    extern __shared__ uint4 q2shm[];
    const cuda_block_q8_K *xt;
    if (xraw) {   /* 融合量化(n_tok==1 门在入口) */
        cuda_block_q8_K *xqsh = (cuda_block_q8_K *)(q2shm + (stage_ok ? (uint64_t)8u * n16 : 0u));
        q2k_fused_quantize(xqsh, xraw, blocks);
        __syncthreads();
        xt = xqsh;
    } else if (stage_ok && x_in_smem) {
        /* 批激活进 shared(2026-08-21): 原本每个 warp 为自己的行各读一遍这 ntb 份 q8 激活,
         * 一个 block 8 个 warp ⇒ 同一份激活被读 8 遍, 整个矩阵每 token 约 38MB 冗余 L2 流量。
         * 块内协作搬一次, 之后全 warp 共用。数值不变(只换读取来源)。 */
        cuda_block_q8_K *xsh = (cuda_block_q8_K *)(q2shm + (uint64_t)8u * n16);
        const uint32_t words = (uint32_t)((ntb * blocks * sizeof(cuda_block_q8_K)) / sizeof(uint4));
        const uint4 *src4 = (const uint4 *)(xq + (uint64_t)tok_base * blocks);
        uint4 *dst4 = (uint4 *)xsh;
        for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) dst4[i] = src4[i];
        __syncthreads();
        xt = xsh;
    } else {
        xt = xq + (uint64_t)tok_base * blocks;
    }
    /* K 维切分(2026-08-21): out_dim 小的矩阵(o_b 只有 512 块)并行度不足 — 同字节下
     * q_b(4096 块)跑 170GB/s 而 o_b 只有 88。按 K 分段各算部分和, reduce 固定 z 序合并。
     * ksplit==1 时与旧路逐位相同。 */
    const uint32_t kz = (ksplit > 1u) ? blockIdx.z : 0u;
    const uint32_t kchunk = (blocks + ksplit - 1u) / ksplit;
    const uint32_t kb0 = kz * kchunk;
    const uint32_t kb1 = (kb0 + kchunk < blocks) ? (kb0 + kchunk) : blocks;
    for (uint32_t row = blockIdx.x * 8u + warp; row < out_dim; row += gridDim.x * 8u) {
        float acc = 0.0f;
        if (stage_ok) {
            const uint4 *src16 = (const uint4 *)(w_base + (uint64_t)row * row_bytes);
            uint4 *my = q2shm + (uint64_t)warp * n16;
            const uint32_t s_lo = (ksplit > 1u) ? (kb0 * 84u / 16u) : 0u;
            const uint32_t s_hi = (ksplit > 1u) ? (kb1 * 84u / 16u) : n16;
            for (uint32_t i = lane + s_lo; i < s_hi; i += 32u) my[i] = __ldcs(src16 + i);
            __syncwarp();
            const cuda_block_q2_K *wr = (const cuda_block_q2_K *)my;
            if (q2k_stage_lv == 9u && blocks == 4u) {   /* 现场对质: 全行全tok计数 */
                float f = 0.0f, sp = 0.0f;
                if (lane == 0) {
                    for (uint32_t b = 0; b < blocks; b++) f += dev_dot_q2_K_q8_K_block(wr + b, xt + b);
                    for (uint32_t b = 0; b < blocks; b++)
                        for (uint32_t q = 0; q < 8u; q++) sp += dev_dot_q2_K_q8_K_block_eighth(wr + b, xt + b, q);
                    const float df = fabsf(f - sp);
                    if (df > 1e-4f && df > 1e-3f * fabsf(f))
                        printf("[q2bad] row=%u tok=%u full=%.6f split=%.6f\n", row, tok_base, f, sp);
                }
            }
            /* 走 smem 通路且是多 token 批: 用多 token 复用解量化版(4 个一组), 权重只解一次。
             * 这条只改 ALU 复用, 逐 (row,tok) 数值与单 token 路逐位相同。 */
            const bool multi_ok = (ntb > 1u) && !xraw &&
                                  !(blocks == 4u && q2k_stage_lv >= 2u && q2k_split_maxblk >= 4u) &&
                                  !(blocks <= 16u && q2k_stage_lv >= 2u && blocks <= q2k_split_maxblk);
            if (multi_ok) {
                for (uint32_t t0 = 0; t0 < ntb; t0 += 4u) {
                    const uint32_t nt = (ntb - t0 < 4u) ? (ntb - t0) : 4u;
                    float a4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    const cuda_block_q8_K *ybase = (x_in_smem ? xt : (xq + (uint64_t)tok_base * blocks))
                                                   + (uint64_t)t0 * blocks;
                    for (uint32_t b = lane + kb0; b < kb1; b += 32u) {
                        switch (nt) {
                        case 4: dev_dot_q2_K_q8_K_block_smem_multi<4>(wr + b, ybase + b, blocks, a4); break;
                        case 3: dev_dot_q2_K_q8_K_block_smem_multi<3>(wr + b, ybase + b, blocks, a4); break;
                        case 2: dev_dot_q2_K_q8_K_block_smem_multi<2>(wr + b, ybase + b, blocks, a4); break;
                        default: dev_dot_q2_K_q8_K_block_smem_multi<1>(wr + b, ybase + b, blocks, a4); break;
                        }
                    }
                    for (uint32_t i = 0; i < nt; i++) {
                        float v = a4[i];
                        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
                        if (lane == 0) {
                            if (ksplit > 1u) partial[((uint64_t)kz * n_tok_total + (tok_base + t0 + i)) * out_dim + row] = v;
                            else out[(uint64_t)(tok_base + t0 + i) * out_dim + row] = v;
                        }
                    }
                }
                __syncwarp();
                continue;
            }
            for (uint32_t t = 0; t < ntb; t++) {
                const cuda_block_q8_K *xtt = (xraw || x_in_smem) ? (xt + (uint64_t)(xraw ? 0u : t) * blocks)
                                                                 : (xq + (uint64_t)(tok_base + t) * blocks);
                float a2 = 0.0f;
                if (blocks == 4u && q2k_stage_lv >= 2u && q2k_split_maxblk >= 4u) {
                    const uint32_t bi = lane >> 3u, qi = lane & 7u;
                    a2 = dev_dot_q2_K_q8_K_block_eighth(wr + bi, xtt + bi, qi);
                } else if (blocks <= 16u && q2k_stage_lv >= 2u && blocks <= q2k_split_maxblk) {
                    const uint32_t bi = lane >> 1u, h = lane & 1u;
                    if (bi < blocks) a2 = dev_dot_q2_K_q8_K_block_half(wr + bi, xtt + bi, h);
                } else {
                    for (uint32_t b = lane + kb0; b < kb1; b += 32u)
                        a2 += dev_dot_q2_K_q8_K_block_smem(wr + b, xtt + b);
                }
                for (int off = 16; off > 0; off >>= 1) a2 += __shfl_down_sync(0xffffffffu, a2, off);
                if (lane == 0) {
                    if (ksplit > 1u) partial[((uint64_t)kz * n_tok_total + (tok_base + t)) * out_dim + row] = a2;
                    else out[(uint64_t)(tok_base + t) * out_dim + row] = a2;
                }
            }
            __syncwarp();   /* 下轮复写 stage 前全 lane 必须算完 */
            continue;
        } else {
            const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(w_base + (uint64_t)row * row_bytes);
            for (uint32_t t = 0; t < ntb; t++) {
                const cuda_block_q8_K *xtt = xraw ? xt : (xq + (uint64_t)(tok_base + t) * blocks);
                float a2 = 0.0f;
                for (uint32_t b = lane; b < blocks; b += 32u)
                    a2 += dev_dot_q2_K_q8_K_block(wr + b, xtt + b);
                for (int off = 16; off > 0; off >>= 1) a2 += __shfl_down_sync(0xffffffffu, a2, off);
                if (lane == 0) out[(uint64_t)(tok_base + t) * out_dim + row] = a2;
            }
            continue;
        }
        (void)acc;
    }
}

__global__ static void matmul_q4_K_warp_kernel(
        float *out, const char *w_base, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks, uint32_t out_dim) {
    /* 每 warp 一行(32 lanes 分块)。A/B 记录: 8-lane×32行/block 版实测更慢
     * (16.84 vs 17.30 t/s) —— dev_dot 每块开销大, 每 lane 串行 2 块比半数 lane
     * 空转更亏。保留本版。 */
    /* warp 协作 shared staging(2026-08-17 第三轮): 此前每 lane 直接读自己的块,
     * warp 内"第 i 次 load"跨 lane 相距 144B ⇒ 不合并。现在 warp 把整行权重
     * (≤32 块×144B)按 lane 连续 16B 粒度搬进 shared(完全合并), 再各 lane 从片上算。 */
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t tok = blockIdx.y;
    const cuda_block_q8_K *xt = xq + (uint64_t)tok * blocks;
    /* grid-stride 行循环(2026-08-17): out=2048 的小矩阵原来发 256 块, 在飞上限 ~192
     * → 1.33 个 wave, 尾波白扔 ~25% 吞吐。现在固定小 grid 循环吃完所有行。 */
    for (uint32_t row = blockIdx.x * 8u + warp; row < out_dim; row += gridDim.x * 8u) {
        float acc = 0.0f;
        if (blocks <= 32u) {
            /* 动态 shared: 静态 [8][32*9] 常驻 36KB/块把 SM 驻留钉在 2 块(occupancy 墙);
             * 按实际行字节配(launch 第三参), blocks=8 时只要 9KB。 */
            extern __shared__ uint4 dstage[];
            const uint32_t n16 = blocks * 9u;
            const uint4 *src16 = (const uint4 *)(w_base + (uint64_t)row * row_bytes);
            uint4 *my = dstage + (uint64_t)warp * n16;
            for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
            __syncwarp();
            const cuda_block_q4_K *wr = (const cuda_block_q4_K *)my;
            if (blocks <= 16u) {
                /* 半块拆分: blocks×2 单元≤32, 全 warp 活跃 */
                const uint32_t bi = lane >> 1u, hj = (lane & 1u) * 4u;
                if (bi < blocks) acc += dev_dot_q4_K_q8_K_block_half(wr + bi, xt + bi, hj);
            } else {
                for (uint32_t b = lane; b < blocks; b += 32u)
                    acc += dev_dot_q4_K_q8_K_block(wr + b, xt + b);
            }
            __syncwarp();   /* 下轮复写 stage 前全 lane 必须算完 */
        } else {
            const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(w_base + (uint64_t)row * row_bytes);
            for (uint32_t b = lane; b < blocks; b += 32u)
                acc += dev_dot_q4_K_q8_K_block_vec(wr + b, xt + b);
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) out[(uint64_t)tok * out_dim + row] = acc;
    }
}

int ds4_gpu_matmul_q4_K_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !model_map || !x || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (in_dim % CUDA_QK_K != 0) return 0;
    const uint32_t blocks = (uint32_t)(in_dim / CUDA_QK_K);
    const uint64_t row_bytes = (uint64_t)blocks * sizeof(cuda_block_q4_K);
    const uint64_t w_bytes = row_bytes * out_dim;
    if (weight_offset > model_size || w_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *w = cuda_model_range_ptr(model_map, weight_offset, w_bytes, "dense_q4k");
    if (!w) return 0;

    const uint64_t xq_need = n_tok * blocks * sizeof(cuda_block_q8_K);
    if (xq_need > g_q4k_xq_bytes) {
        if (g_q4k_xq_sc) (void)cudaFree(g_q4k_xq_sc);
        g_q4k_xq_sc = NULL; g_q4k_xq_bytes = 0;
        if (cudaMalloc(&g_q4k_xq_sc, xq_need) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
        g_q4k_xq_bytes = xq_need;
    }
    /* 该 kernel 的 grid 语义: blockIdx.x 就是量化块号(一 CUDA block 一 q8_K 块) */
    q8_K_quantize_kernel<<<dim3(blocks, (unsigned)n_tok, 1), 256, 0, g_cur_stream>>>(
        (cuda_block_q8_K *)g_q4k_xq_sc, (const float *)x->ptr, (uint32_t)in_dim, (uint32_t)n_tok);
    /* A/B 三连档案: 16-lane 变体在小矩阵上仍输(18.84 vs 19.13) — dense 场景 32-lane
     * 恒胜, dev_dot 每块开销决定一切, lane 空转无关紧要。变体保留但不启用。 */
    if (((const char *)0) /* DS4_Q4K_DIMS: 路径开关已删(2026-08-22 隐形炸弹清理) */) {
        static uint64_t seen[64][2]; static int nseen = 0;
        int hit = 0;
        for (int i = 0; i < nseen; i++) if (seen[i][0] == in_dim && seen[i][1] == out_dim) { hit = 1; break; }
        if (!hit && nseen < 64) {
            seen[nseen][0] = in_dim; seen[nseen][1] = out_dim; nseen++;
            fprintf(stderr, "ds4: [q4k-dims] in=%llu out=%llu ntok=%llu\n",
                    (unsigned long long)in_dim, (unsigned long long)out_dim, (unsigned long long)n_tok);
        }
    }
    /* 动态 shared: blocks≤32 时 kernel staging 需 8 warps×blocks×144B */
    const size_t q4k_shmem = (blocks <= 32u) ? (size_t)8u * blocks * 9u * sizeof(uint4) : 0;
    unsigned q4k_gx = (unsigned)((out_dim + 7u) / 8u);
    if (q4k_gx > ds4_grid_cap()) q4k_gx = ds4_grid_cap();   /* 48 SM × 4 驻留块: 单 wave 满载, 行循环吃尾 */
    matmul_q4_K_warp_kernel<<<dim3(q4k_gx, (unsigned)n_tok, 1), 256, q4k_shmem, g_cur_stream>>>(
        (float *)out->ptr, w, (const cuda_block_q8_K *)g_q4k_xq_sc,
        row_bytes, blocks, (uint32_t)out_dim);
    return cuda_ok(cudaGetLastError(), "dense q4_K matmul launch");
}

/* 分组投影的分片 GEMM(2026-08-21): 与 dense tiled 同一套修法。行 r 属组 r/rank, 只需要
 * 该组那一段激活; 一个 block 的 8 行同组(rank≫8), 于是 tile 内只搬这一组的 T 份激活。
 * 原 kernel 每 warp 每 token 各读一遍 ⇒ 8192 行 × 340 token × 2.3KB ≈ 6.4GB; 分片后
 * 只剩 blocks × token × 2.3KB。lane 映射与 dot 函数不变 ⇒ 逐位一致。 */
template <uint32_t NB>
__global__ static void __launch_bounds__(256, 1) grouped_q2_K_tiled_kernel(
        float *out, const char *w_base, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks_dyn, uint32_t out_dim,
        uint32_t rank, uint32_t n_groups, uint32_t n_tok, uint32_t tile) {
    const uint32_t blocks = NB ? NB : blocks_dyn;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t n16 = blocks * 84u / 16u;
    extern __shared__ uint4 gtshm[];
    cuda_block_q8_K *acts = (cuda_block_q8_K *)gtshm;
    uint4 *wsh = (uint4 *)(acts + (uint64_t)tile * blocks);

    const uint32_t base0 = blockIdx.x * 8u;
    const uint32_t stride = gridDim.x * 8u;
    const uint32_t n_iter = (base0 >= out_dim) ? 0u : ((out_dim - base0 + stride - 1u) / stride);
    /* 本块第一行所在组: 块内 8 行同组(rank 是 1024 量级), 跨迭代可能换组 ⇒ 每迭代按行算组,
     * 但激活 tile 是按"块当前迭代的组"搬的 ⇒ 行循环放外层, tile 放内层。 */
    for (uint32_t it = 0; it < n_iter; it++) {
        const uint32_t row = base0 + warp + it * stride;
        const bool live = row < out_dim;
        const uint32_t grp = live ? (row / rank) : 0u;
        /* 块内 8 个 warp 的组必须一致才能共用 tile —— rank≥8 时天然成立 */
        const uint32_t grp_blk = (base0 + it * stride) / rank;
        const cuda_block_q2_K *wr = NULL;
        if (live) {
            const uint4 *src16 = (const uint4 *)(w_base + (uint64_t)row * row_bytes);
            uint4 *my = wsh + (uint64_t)warp * n16;
            for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
            __syncwarp();
            wr = (const cuda_block_q2_K *)my;
        }
        for (uint32_t t0 = 0; t0 < n_tok; t0 += tile) {
            const uint32_t tn = (n_tok - t0 < tile) ? (n_tok - t0) : tile;
            __syncthreads();
            for (uint32_t i = threadIdx.x; i < tn * blocks; i += blockDim.x) {
                const uint32_t tt = i / blocks, bb = i - tt * blocks;
                acts[(uint64_t)tt * blocks + bb] =
                    xq[((uint64_t)(t0 + tt) * n_groups + grp_blk) * blocks + bb];
            }
            __syncthreads();
            for (uint32_t t = 0; t < tn; t++) {
                float acc = 0.0f;
                if (live && grp == grp_blk) {
                    const cuda_block_q8_K *xt = acts + (uint64_t)t * blocks;
                    if (blocks <= 16u) {
                        const uint32_t bi = lane >> 1u, h = lane & 1u;
                        if (bi < blocks) acc = dev_dot_q2_K_q8_K_block_half(wr + bi, xt + bi, h);
                    } else {
                        for (uint32_t b = lane; b < blocks; b += 32u)
                            acc += dev_dot_q2_K_q8_K_block_smem(wr + b, xt + b);
                    }
                } else if (live) {
                    /* 组不一致(理论上不会发生, rank≥8): 退回全局读, 保正确 */
                    const cuda_block_q8_K *xt = xq + ((uint64_t)(t0 + t) * n_groups + grp) * blocks;
                    if (blocks <= 16u) {
                        const uint32_t bi = lane >> 1u, h = lane & 1u;
                        if (bi < blocks) acc = dev_dot_q2_K_q8_K_block_half(wr + bi, xt + bi, h);
                    } else {
                        for (uint32_t b = lane; b < blocks; b += 32u)
                            acc += dev_dot_q2_K_q8_K_block_smem(wr + b, xt + b);
                    }
                }
                for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
                if (live && lane == 0) out[(uint64_t)(t0 + t) * out_dim + row] = acc;
            }
        }
        __syncthreads();
    }
}

static bool grouped_q2k_tiled_set_smem(uint32_t blocks, int bytes) {
    if (bytes <= 48 * 1024) return true;
    static int done[5] = {0, 0, 0, 0, 0};
    static int cap[5] = {0, 0, 0, 0, 0};
    const int idx = (blocks == 4u) ? 0 : (blocks == 8u) ? 1 : (blocks == 16u) ? 2 : (blocks == 32u) ? 3 : 4;
    if (done[idx]) return cap[idx] >= bytes;
    cudaError_t e = cudaErrorInvalidValue;
    switch (blocks) {
    case 4u:  e = cudaFuncSetAttribute(grouped_q2_K_tiled_kernel<4u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 8u:  e = cudaFuncSetAttribute(grouped_q2_K_tiled_kernel<8u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 16u: e = cudaFuncSetAttribute(grouped_q2_K_tiled_kernel<16u>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 32u: e = cudaFuncSetAttribute(grouped_q2_K_tiled_kernel<32u>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    default:  e = cudaFuncSetAttribute(grouped_q2_K_tiled_kernel<0u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    }
    done[idx] = 1; cap[idx] = (e == cudaSuccess) ? bytes : 0;
    if (e != cudaSuccess) { (void)cudaGetLastError(); return false; }
    return true;
}

static inline void grouped_q2k_tiled_launch(unsigned gx, size_t sh,
        float *out, const char *w, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks, uint32_t out_dim,
        uint32_t rank, uint32_t n_groups, uint32_t n_tok, uint32_t tile) {
    switch (blocks) {
    case 4u:  grouped_q2_K_tiled_kernel<4u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, rank, n_groups, n_tok, tile); break;
    case 8u:  grouped_q2_K_tiled_kernel<8u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, rank, n_groups, n_tok, tile); break;
    case 16u: grouped_q2_K_tiled_kernel<16u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, rank, n_groups, n_tok, tile); break;
    case 32u: grouped_q2_K_tiled_kernel<32u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, rank, n_groups, n_tok, tile); break;
    default:  grouped_q2_K_tiled_kernel<0u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, rank, n_groups, n_tok, tile); break;
    }
}

__global__ static void __launch_bounds__(256, 6) grouped_q2_K_warp_kernel(
        float *out, const char *w_base, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks, uint32_t out_dim,
        uint32_t rank, uint32_t n_groups, uint32_t n_tok_batch) {
    /* attn_output_a 分组结构(全q2, 2026-08-19): 行 r 属组 r/rank, 读 x 的第 g 段。
     * 2026-08-20 提速刀②改: 实测形状 blocks=16/低维8192 — 与 dense 同款"裸读克隆"病
     * (129GB/s), 移植同款 staging+半块手术。 */
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t ntb = n_tok_batch ? n_tok_batch : 1u;   /* 权重驻留批(见 dense 同款) */
    const uint32_t tok_base = (ntb > 1u) ? 0u : blockIdx.y;
    const uint32_t stage_ok = (blocks <= 32u && (blocks & 3u) == 0u);
    const uint32_t n16 = blocks * 84u / 16u;
    for (uint32_t row = blockIdx.x * 8u + warp; row < out_dim; row += gridDim.x * 8u) {
        const uint32_t grp = row / rank;
        if (stage_ok) {
            extern __shared__ uint4 gq2stage[];
            const uint4 *src16 = (const uint4 *)(w_base + (uint64_t)row * row_bytes);
            uint4 *my = gq2stage + (uint64_t)warp * n16;
            for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
            __syncwarp();
            const cuda_block_q2_K *wr = (const cuda_block_q2_K *)my;
            for (uint32_t t = 0; t < ntb; t++) {
                const cuda_block_q8_K *xt = xq + ((uint64_t)(tok_base + t) * n_groups + grp) * blocks;
                float acc = 0.0f;
                if (blocks <= 16u) {
                    const uint32_t bi = lane >> 1u, h = lane & 1u;
                    if (bi < blocks) acc = dev_dot_q2_K_q8_K_block_half(wr + bi, xt + bi, h);
                } else {
                    for (uint32_t b = lane; b < blocks; b += 32u)
                        acc += dev_dot_q2_K_q8_K_block_smem(wr + b, xt + b);
                }
                for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
                if (lane == 0) out[(uint64_t)(tok_base + t) * out_dim + row] = acc;
            }
            __syncwarp();
        } else {
            const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(w_base + (uint64_t)row * row_bytes);
            for (uint32_t t = 0; t < ntb; t++) {
                const cuda_block_q8_K *xt = xq + ((uint64_t)(tok_base + t) * n_groups + grp) * blocks;
                float acc = 0.0f;
                for (uint32_t b = lane; b < blocks; b += 32u)
                    acc += dev_dot_q2_K_q8_K_block(wr + b, xt + b);
                for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
                if (lane == 0) out[(uint64_t)(tok_base + t) * out_dim + row] = acc;
            }
        }
    }
}

/* 小块专用变体(2026-08-20 提速刀②): blocks≤4 时 warp-per-row 模板 32 lane 里 28 个空转
 * (nsys 实测 3.68ms/tok ≈ 6GB/s 纯延迟病)。改一 lane 一行(256 行/块), 低寄存器 dot。 */
__global__ static void grouped_q2_K_rowlane_kernel(
        float *out, const char *w_base, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks, uint32_t out_dim,
        uint32_t rank, uint32_t n_groups) {
    const uint32_t tok = blockIdx.y;
    const uint32_t row = blockIdx.x * 256u + threadIdx.x;
    if (row >= out_dim) return;
    const uint32_t grp = row / rank;
    const cuda_block_q8_K *xt = xq + ((uint64_t)tok * n_groups + grp) * blocks;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(w_base + (uint64_t)row * row_bytes);
    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; b++)
        acc += dev_dot_q2_K_q8_K_block_smem(wr + b, xt + b);
    out[(uint64_t)tok * out_dim + row] = acc;
}

int ds4_gpu_attention_output_q2k_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint64_t out_dim,
        const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    /* q2_K 版批量 attn_output(2026-08-19 全q2): a=分组 q2_K matmul, b=dense q2_K。 */
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0) return 0;
    if (group_dim % CUDA_QK_K != 0 || ((uint64_t)n_groups * rank) % CUDA_QK_K != 0) return 0;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint32_t blocks = (uint32_t)(group_dim / CUDA_QK_K);
    const uint64_t row_bytes = (uint64_t)blocks * sizeof(cuda_block_q2_K);
    const uint64_t a_bytes = row_bytes * low_dim;
    if (out_a_offset > model_size || a_bytes > model_size - out_a_offset ||
        heads->bytes < (uint64_t)n_tokens * n_groups * group_dim * sizeof(float) ||
        low->bytes < (uint64_t)n_tokens * low_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) return 0;
    const char *wa = (const char *)cuda_model_range_ptr(model_map, out_a_offset, a_bytes, "attn_out_a_q2k");
    if (!wa) return 0;
    const uint64_t xrows = (uint64_t)n_tokens * n_groups;
    const uint64_t xq_need = xrows * blocks * sizeof(cuda_block_q8_K);
    if (xq_need > g_q4k_xq_bytes) {
        cudaStreamCaptureStatus pcs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(0, &pcs);
        if (pcs != cudaStreamCaptureStatusNone) return 0;
        if (g_q4k_xq_sc) (void)cudaFree(g_q4k_xq_sc);
        g_q4k_xq_sc = NULL; g_q4k_xq_bytes = 0;
        if (cudaMalloc(&g_q4k_xq_sc, xq_need) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
        g_q4k_xq_bytes = xq_need;
    }
    q8_K_quantize_kernel<<<dim3(blocks, (unsigned)xrows, 1), 256, 0, g_cur_stream>>>(
        (cuda_block_q8_K *)g_q4k_xq_sc, (const float *)heads->ptr, (uint32_t)group_dim, (uint32_t)xrows);
    if (((const char *)0) /* DS4_F16_DIMS: 路径开关已删(2026-08-22 隐形炸弹清理) */) {   /* 诊断: grouped 形状一次性打印 */
        static int gseen = 0;
        if (!gseen) { gseen = 1;
            fprintf(stderr, "ds4: [grouped-q2k-dims] group_dim=%llu rank=%llu n_groups=%u low_dim=%llu blocks=%u\n",
                    (unsigned long long)group_dim, (unsigned long long)rank, n_groups,
                    (unsigned long long)low_dim, blocks); }
    }
    if (blocks <= 4u) {   /* 小块: 一 lane 一行(warp 模板空转病, 见 kernel 注释) */
        grouped_q2_K_rowlane_kernel<<<dim3((unsigned)((low_dim + 255u) / 256u), (unsigned)n_tokens, 1),
                                      256, 0, g_cur_stream>>>(
            (float *)low->ptr, wa, (const cuda_block_q8_K *)g_q4k_xq_sc,
            row_bytes, blocks, (uint32_t)low_dim, (uint32_t)rank, n_groups);
    } else {
        unsigned gx = (unsigned)((low_dim + 7u) / 8u);
        if (gx > ds4_grid_cap()) gx = ds4_grid_cap();   /* 08-20 阶梯审判: 384 比 192 +4-8GB/s(V1 222→229/V2 226→230) */
        const size_t gsh = (blocks <= 32u && (blocks & 3u) == 0u)
                           ? (size_t)8u * (blocks * 84u / 16u) * sizeof(uint4) : 0u;
        static uint32_t gtmin = 0u;
        if (gtmin == 0u) { const char *e = ((const char *)0) /* DS4_Q2K_TILED_MIN: 路径开关已删(2026-08-22 隐形炸弹清理) */; gtmin = e ? (uint32_t)atoi(e) : 16u; }
        if (n_tokens >= gtmin && blocks <= 32u && (blocks & 3u) == 0u && rank >= 8u &&
            1) {
            const size_t wsh_bytes = (size_t)8u * (blocks * 84u / 16u) * sizeof(uint4);
            const size_t budget = 96u * 1024u;
            uint32_t tile = (uint32_t)((budget - wsh_bytes) / ((size_t)blocks * sizeof(cuda_block_q8_K)));
            if (tile > n_tokens) tile = n_tokens;
            if (tile > 64u) tile = 64u;
            if (tile >= 2u) {
                const size_t sh = wsh_bytes + (size_t)tile * blocks * sizeof(cuda_block_q8_K);
                if (grouped_q2k_tiled_set_smem(blocks, (int)sh)) {
                    unsigned ggx = (unsigned)((low_dim + 7u) / 8u);
                    static uint32_t gtgx = 0u;
                    if (gtgx == 0u) { const char *e = ((const char *)0) /* DS4_Q2K_TILED_BLOCKS: 路径开关已删(2026-08-22 隐形炸弹清理) */; gtgx = e ? (uint32_t)atoi(e) : 512u; }
                    if (ggx > gtgx) ggx = gtgx;
                    grouped_q2k_tiled_launch(ggx, sh, (float *)low->ptr, wa,
                                             (const cuda_block_q8_K *)g_q4k_xq_sc,
                                             row_bytes, blocks, (uint32_t)low_dim,
                                             (uint32_t)rank, n_groups, n_tokens, tile);
                    if (!cuda_ok(cudaGetLastError(), "attn_out_q2k grouped tiled launch")) return 0;
                    return ds4_gpu_matmul_q2_K_tensor(out, model_map, model_size, out_b_offset,
                                                      low_dim, out_dim, low, n_tokens);
                }
            }
        }
        const uint32_t gwstat = (n_tokens > 1u && ((const char *)0) /* DS4_Q2K_NO_WSTAT: 诊断开关已删(2026-08-22) */ == NULL) ? (uint32_t)n_tokens : 1u;
        if (gwstat > 1u && 1) gx = (unsigned)((low_dim + 7u) / 8u);
        grouped_q2_K_warp_kernel<<<dim3(gx, gwstat > 1u ? 1u : (unsigned)n_tokens, 1), 256, gsh, g_cur_stream>>>(
            (float *)low->ptr, wa, (const cuda_block_q8_K *)g_q4k_xq_sc,
            row_bytes, blocks, (uint32_t)low_dim, (uint32_t)rank, n_groups, gwstat);
    }
    if (!cuda_ok(cudaGetLastError(), "attn_out_q2k grouped launch")) return 0;
    return ds4_gpu_matmul_q2_K_tensor(out, model_map, model_size, out_b_offset,
                                      low_dim, out_dim, low, n_tokens);
}


/* pair 的分片版(2026-08-21): 两矩阵共用同一份激活, 正好一起分片 —— 激活按 tile 进
 * shared 块内共用, 权重每行 stage 一次。lane 映射/dot/求和序与 pair_batch 完全相同。 */
template <uint32_t NB>
__global__ static void __launch_bounds__(256, 1) matmul_q2_K_pair_tiled_kernel(
        float *out0, float *out1, const char *w0, const char *w1,
        const cuda_block_q8_K *xq, uint64_t row_bytes, uint32_t blocks_dyn,
        uint32_t out0_dim, uint32_t out1_dim, uint32_t n_tok, uint32_t tile,
        uint32_t lv, uint32_t mb) {
    const uint32_t blocks = NB ? NB : blocks_dyn;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t n16 = blocks * 84u / 16u;
    extern __shared__ uint4 ptshm[];
    cuda_block_q8_K *acts = (cuda_block_q8_K *)ptshm;
    uint4 *wsh = (uint4 *)(acts + (uint64_t)tile * blocks);
    const uint32_t total = out0_dim + out1_dim;
    const uint32_t base0 = blockIdx.x * 8u;
    const uint32_t stride = gridDim.x * 8u;
    const uint32_t n_iter = (base0 >= total) ? 0u : ((total - base0 + stride - 1u) / stride);

    for (uint32_t t0 = 0; t0 < n_tok; t0 += tile) {
        const uint32_t tn = (n_tok - t0 < tile) ? (n_tok - t0) : tile;
        __syncthreads();
        {
            const uint32_t words = (uint32_t)(((uint64_t)tn * blocks * sizeof(cuda_block_q8_K)) / sizeof(uint4));
            const uint4 *src = (const uint4 *)(xq + (uint64_t)t0 * blocks);
            uint4 *dst = (uint4 *)acts;
            for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) dst[i] = src[i];
        }
        __syncthreads();
        for (uint32_t it = 0; it < n_iter; it++) {
            const uint32_t r = base0 + warp + it * stride;
            const bool live = r < total;
            const bool second = live && (r >= out0_dim);
            const uint32_t row = second ? (r - out0_dim) : r;
            float *outp = second ? out1 : out0;
            const uint32_t od = second ? out1_dim : out0_dim;
            const cuda_block_q2_K *wr = NULL;
            if (live) {
                const char *wb = second ? w1 : w0;
                const uint4 *src16 = (const uint4 *)(wb + (uint64_t)row * row_bytes);
                uint4 *my = wsh + (uint64_t)warp * n16;
                for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
                __syncwarp();
                wr = (const cuda_block_q2_K *)my;
            }
            for (uint32_t t = 0; t < tn; t++) {
                float a2 = 0.0f;
                if (live) {
                    const cuda_block_q8_K *xt = acts + (uint64_t)t * blocks;
                    if (blocks == 4u && lv >= 2u && mb >= 4u) {
                        const uint32_t bi = lane >> 3u, qi = lane & 7u;
                        a2 = dev_dot_q2_K_q8_K_block_eighth(wr + bi, xt + bi, qi);
                    } else if (blocks <= 16u && lv >= 2u && blocks <= mb) {
                        const uint32_t bi = lane >> 1u, h = lane & 1u;
                        if (bi < blocks) a2 = dev_dot_q2_K_q8_K_block_half(wr + bi, xt + bi, h);
                    } else {
                        for (uint32_t b = lane; b < blocks; b += 32u)
                            a2 += dev_dot_q2_K_q8_K_block_smem(wr + b, xt + b);
                    }
                }
                for (int off = 16; off > 0; off >>= 1) a2 += __shfl_down_sync(0xffffffffu, a2, off);
                if (live && lane == 0) outp[(uint64_t)(t0 + t) * od + row] = a2;
            }
            __syncwarp();
        }
    }
}

static bool pair_q2k_tiled_set_smem(uint32_t blocks, int bytes) {
    if (bytes <= 48 * 1024) return true;
    static int done[5] = {0, 0, 0, 0, 0};
    static int cap[5] = {0, 0, 0, 0, 0};
    const int idx = (blocks == 4u) ? 0 : (blocks == 8u) ? 1 : (blocks == 16u) ? 2 : (blocks == 32u) ? 3 : 4;
    if (done[idx]) return cap[idx] >= bytes;
    cudaError_t e = cudaErrorInvalidValue;
    switch (blocks) {
    case 4u:  e = cudaFuncSetAttribute(matmul_q2_K_pair_tiled_kernel<4u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 8u:  e = cudaFuncSetAttribute(matmul_q2_K_pair_tiled_kernel<8u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 16u: e = cudaFuncSetAttribute(matmul_q2_K_pair_tiled_kernel<16u>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 32u: e = cudaFuncSetAttribute(matmul_q2_K_pair_tiled_kernel<32u>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    default:  e = cudaFuncSetAttribute(matmul_q2_K_pair_tiled_kernel<0u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    }
    done[idx] = 1; cap[idx] = (e == cudaSuccess) ? bytes : 0;
    if (e != cudaSuccess) { (void)cudaGetLastError(); return false; }
    return true;
}

static inline void pair_q2k_tiled_launch(unsigned gx, size_t sh,
        float *out0, float *out1, const char *w0, const char *w1,
        const cuda_block_q8_K *xq, uint64_t row_bytes, uint32_t blocks,
        uint32_t out0_dim, uint32_t out1_dim, uint32_t n_tok, uint32_t tile,
        uint32_t lv, uint32_t mb) {
    switch (blocks) {
    case 4u:  matmul_q2_K_pair_tiled_kernel<4u><<<gx, 256, sh, g_cur_stream>>>(out0, out1, w0, w1, xq, row_bytes, blocks, out0_dim, out1_dim, n_tok, tile, lv, mb); break;
    case 8u:  matmul_q2_K_pair_tiled_kernel<8u><<<gx, 256, sh, g_cur_stream>>>(out0, out1, w0, w1, xq, row_bytes, blocks, out0_dim, out1_dim, n_tok, tile, lv, mb); break;
    case 16u: matmul_q2_K_pair_tiled_kernel<16u><<<gx, 256, sh, g_cur_stream>>>(out0, out1, w0, w1, xq, row_bytes, blocks, out0_dim, out1_dim, n_tok, tile, lv, mb); break;
    case 32u: matmul_q2_K_pair_tiled_kernel<32u><<<gx, 256, sh, g_cur_stream>>>(out0, out1, w0, w1, xq, row_bytes, blocks, out0_dim, out1_dim, n_tok, tile, lv, mb); break;
    default:  matmul_q2_K_pair_tiled_kernel<0u><<<gx, 256, sh, g_cur_stream>>>(out0, out1, w0, w1, xq, row_bytes, blocks, out0_dim, out1_dim, n_tok, tile, lv, mb); break;
    }
}

/* 批版 pair 融合(2026-08-21): 同输入的两矩阵(shexp gate+up / q_a+kv)合成一发。
 * ①激活只量化一次 ②行数合并 ⇒ 小矩阵不再因 gx 小而并行度不足(35GB/s 病)
 * ③权重驻留(块内循环 token)。逐 (row,tok) 数学与单发路逐位相同。 */
template<uint32_t NB>
__global__ static void __launch_bounds__(256, DS4_Q2K_BLOCKS_PER_SM) matmul_q2_K_pair_batch_kernel(
        float *out0, float *out1, const char *w0, const char *w1,
        const cuda_block_q8_K *xq, uint64_t row_bytes, uint32_t blocks_dyn,
        uint32_t out0_dim, uint32_t out1_dim, uint32_t ntb, uint32_t lv, uint32_t mb) {
    const uint32_t blocks = NB ? NB : blocks_dyn;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t n16 = blocks * 84u / 16u;
    extern __shared__ uint4 pairshm[];
    const uint32_t total = out0_dim + out1_dim;
    for (uint32_t r = blockIdx.x * 8u + warp; r < total; r += gridDim.x * 8u) {
        const bool second = (r >= out0_dim);
        const uint32_t row = second ? (r - out0_dim) : r;
        const char *wb = second ? w1 : w0;
        float *outp = second ? out1 : out0;
        const uint32_t od = second ? out1_dim : out0_dim;
        const uint4 *src16 = (const uint4 *)(wb + (uint64_t)row * row_bytes);
        uint4 *my = pairshm + (uint64_t)warp * n16;
        for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
        __syncwarp();
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)my;
        for (uint32_t t = 0; t < ntb; t++) {
            const cuda_block_q8_K *xt = xq + (uint64_t)t * blocks;
            float a2 = 0.0f;
            if (blocks == 4u && lv >= 2u && mb >= 4u) {
                const uint32_t bi = lane >> 3u, qi = lane & 7u;
                a2 = dev_dot_q2_K_q8_K_block_eighth(wr + bi, xt + bi, qi);
            } else if (blocks <= 16u && lv >= 2u && blocks <= mb) {
                const uint32_t bi = lane >> 1u, h = lane & 1u;
                if (bi < blocks) a2 = dev_dot_q2_K_q8_K_block_half(wr + bi, xt + bi, h);
            } else {
                for (uint32_t b = lane; b < blocks; b += 32u)
                    a2 += dev_dot_q2_K_q8_K_block_smem(wr + b, xt + b);
            }
            for (int off = 16; off > 0; off >>= 1) a2 += __shfl_down_sync(0xffffffffu, a2, off);
            if (lane == 0) outp[(uint64_t)t * od + row] = a2;
        }
        __syncwarp();
    }
}

int ds4_gpu_matmul_q2_K_pair_batch_tensor(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
        const void *model_map, uint64_t model_size,
        uint64_t off0, uint64_t off1,
        uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out0 || !out1 || !model_map || !x || in_dim == 0 || n_tok < 2u) return 0;
    if (in_dim % CUDA_QK_K != 0) return 0;
    const uint32_t blocks = (uint32_t)(in_dim / CUDA_QK_K);
    if (blocks > 32u || (blocks & 3u) != 0u) return 0;
    const uint64_t row_bytes = (uint64_t)blocks * sizeof(cuda_block_q2_K);
    const uint64_t w0b = row_bytes * out0_dim, w1b = row_bytes * out1_dim;
    if (off0 > model_size || w0b > model_size - off0) return 0;
    if (off1 > model_size || w1b > model_size - off1) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float)) return 0;
    if (out0->bytes < n_tok * out0_dim * sizeof(float)) return 0;
    if (out1->bytes < n_tok * out1_dim * sizeof(float)) return 0;
    const char *w0 = cuda_model_range_ptr(model_map, off0, w0b, "dense_q2k_pair0");
    const char *w1 = cuda_model_range_ptr(model_map, off1, w1b, "dense_q2k_pair1");
    if (!w0 || !w1) return 0;
    const uint64_t xq_need = n_tok * blocks * sizeof(cuda_block_q8_K);
    if (xq_need > g_q4k_xq_bytes) {
        cudaStreamCaptureStatus pcs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(0, &pcs);
        if (pcs != cudaStreamCaptureStatusNone) return 0;
        if (g_q4k_xq_sc) (void)cudaFree(g_q4k_xq_sc);
        g_q4k_xq_sc = NULL; g_q4k_xq_bytes = 0;
        if (cudaMalloc(&g_q4k_xq_sc, xq_need) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
        g_q4k_xq_bytes = xq_need;
    }
    q8_K_quantize_kernel<<<dim3(blocks, (unsigned)n_tok, 1), 256, 0, g_cur_stream>>>(
        (cuda_block_q8_K *)g_q4k_xq_sc, (const float *)x->ptr, (uint32_t)in_dim, (uint32_t)n_tok);
    static uint32_t plv = 99u, pmb = 999u;
    if (plv == 99u) { const char *e = ((const char *)0) /* DS4_Q2K_STAGE: 路径开关已删(2026-08-22 隐形炸弹清理) */; plv = e ? (uint32_t)atoi(e) : 2u; }
    if (pmb == 999u) { const char *e = ((const char *)0) /* DS4_Q2K_SPLIT_MAXBLK: 路径开关已删(2026-08-22 隐形炸弹清理) */; pmb = e ? (uint32_t)atoi(e) : 16u; }
    const unsigned gx = (unsigned)((out0_dim + out1_dim + 7u) / 8u);
    const size_t sh = (size_t)8u * (blocks * 84u / 16u) * sizeof(uint4);
    const cuda_block_q8_K *xq = (const cuda_block_q8_K *)g_q4k_xq_sc;
    {   /* 大批: 激活分片进 shared(与 dense/grouped 同一修法) */
        static uint32_t ptmin = 0u;
        if (ptmin == 0u) { const char *e = ((const char *)0) /* DS4_Q2K_TILED_MIN: 路径开关已删(2026-08-22 隐形炸弹清理) */; ptmin = e ? (uint32_t)atoi(e) : 16u; }
        if (n_tok >= ptmin && 1 &&
            1) {
            const size_t budget = 96u * 1024u;
            uint32_t tile = (uint32_t)((budget - sh) / ((size_t)blocks * sizeof(cuda_block_q8_K)));
            if (tile > n_tok) tile = (uint32_t)n_tok;
            if (tile > 64u) tile = 64u;
            if (tile >= 2u) {
                const size_t psh = sh + (size_t)tile * blocks * sizeof(cuda_block_q8_K);
                if (pair_q2k_tiled_set_smem(blocks, (int)psh)) {
                    static uint32_t ptgx = 0u;
                    if (ptgx == 0u) { const char *e = ((const char *)0) /* DS4_Q2K_TILED_BLOCKS: 路径开关已删(2026-08-22 隐形炸弹清理) */; ptgx = e ? (uint32_t)atoi(e) : 512u; }
                    unsigned pgx = gx > ptgx ? ptgx : gx;
                    pair_q2k_tiled_launch(pgx, psh, (float *)out0->ptr, (float *)out1->ptr, w0, w1,
                                          xq, row_bytes, blocks, (uint32_t)out0_dim, (uint32_t)out1_dim,
                                          (uint32_t)n_tok, tile, plv, pmb);
                    return cuda_ok(cudaGetLastError(), "dense q2_K pair tiled launch");
                }
            }
        }
    }
    switch (blocks) {
    case 4u:  matmul_q2_K_pair_batch_kernel<4u><<<gx, 256, sh, g_cur_stream>>>((float *)out0->ptr, (float *)out1->ptr, w0, w1, xq, row_bytes, blocks, (uint32_t)out0_dim, (uint32_t)out1_dim, (uint32_t)n_tok, plv, pmb); break;
    case 8u:  matmul_q2_K_pair_batch_kernel<8u><<<gx, 256, sh, g_cur_stream>>>((float *)out0->ptr, (float *)out1->ptr, w0, w1, xq, row_bytes, blocks, (uint32_t)out0_dim, (uint32_t)out1_dim, (uint32_t)n_tok, plv, pmb); break;
    case 16u: matmul_q2_K_pair_batch_kernel<16u><<<gx, 256, sh, g_cur_stream>>>((float *)out0->ptr, (float *)out1->ptr, w0, w1, xq, row_bytes, blocks, (uint32_t)out0_dim, (uint32_t)out1_dim, (uint32_t)n_tok, plv, pmb); break;
    case 32u: matmul_q2_K_pair_batch_kernel<32u><<<gx, 256, sh, g_cur_stream>>>((float *)out0->ptr, (float *)out1->ptr, w0, w1, xq, row_bytes, blocks, (uint32_t)out0_dim, (uint32_t)out1_dim, (uint32_t)n_tok, plv, pmb); break;
    default:  matmul_q2_K_pair_batch_kernel<0u><<<gx, 256, sh, g_cur_stream>>>((float *)out0->ptr, (float *)out1->ptr, w0, w1, xq, row_bytes, blocks, (uint32_t)out0_dim, (uint32_t)out1_dim, (uint32_t)n_tok, plv, pmb); break;
    }
    return cuda_ok(cudaGetLastError(), "dense q2_K pair batch launch");
}

__global__ static void q2k_ksplit_reduce_kernel(float *out, const float *partial,
                                               uint32_t out_dim, uint32_t n_tok, uint32_t ksplit) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)out_dim * n_tok) return;
    const uint32_t t = (uint32_t)(idx / out_dim), row = (uint32_t)(idx - (uint64_t)t * out_dim);
    float acc = 0.0f;
    for (uint32_t z = 0; z < ksplit; z++)   /* 固定 z 序 = 确定性 */
        acc += partial[((uint64_t)z * n_tok + t) * out_dim + row];
    out[(uint64_t)t * out_dim + row] = acc;
}

/* 动态 shared 上限 opt-in(超 48KB 必须显式申请, 每形状只设一次)。 */
static bool q2k_set_dynamic_smem(uint32_t blocks, int bytes) {
    if (bytes <= 48 * 1024) return true;
    static int done[5] = {0, 0, 0, 0, 0};
    static int cap[5] = {0, 0, 0, 0, 0};
    const int idx = (blocks == 4u) ? 0 : (blocks == 8u) ? 1 : (blocks == 16u) ? 2 : (blocks == 32u) ? 3 : 4;
    if (done[idx] && cap[idx] >= bytes) return true;
    if (done[idx] && cap[idx] < bytes) return false;
    cudaError_t e = cudaErrorInvalidValue;
    switch (blocks) {
    case 4u:  e = cudaFuncSetAttribute(matmul_q2_K_warp_kernel<4u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 8u:  e = cudaFuncSetAttribute(matmul_q2_K_warp_kernel<8u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 16u: e = cudaFuncSetAttribute(matmul_q2_K_warp_kernel<16u>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 32u: e = cudaFuncSetAttribute(matmul_q2_K_warp_kernel<32u>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    default:  e = cudaFuncSetAttribute(matmul_q2_K_warp_kernel<0u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    }
    done[idx] = 1; cap[idx] = (e == cudaSuccess) ? bytes : 0;
    if (e != cudaSuccess) { (void)cudaGetLastError(); return false; }
    return true;
}

/* 大批分片 GEMM(2026-08-21): 原 warp kernel 是"一行一 warp, 块内循环 token", 每个 warp
 * 为自己那一行把整批 token 的 q8 激活重读一遍, 一个 block 的 8 个 warp 又各读一遍 ——
 * 340 token 的块产生约 12.9GB 显存流量(权重才 11MB), prefill 因此只有 42 t/s。
 * 这里按 token 分片: 每个 tile 的激活块内协作搬进 shared 一次, 8 个 warp 共用; 权重每行
 * 仍只 stage 一次(行循环在 tile 内)。每 lane 的 block 映射与 dot 函数与原 kernel 完全相同,
 * 所以结果逐位一致, 只是少读几个数量级的字节。
 *   shared 布局: [acts: T × blocks × sizeof(q8_K)][weights: 8 warps × blocks*84B]
 *   行循环次数在块内统一(否则 __syncthreads 会挂), 越界行只跳过写回。 */
template <uint32_t NB>
__global__ static void __launch_bounds__(256, 1) matmul_q2_K_tiled_kernel(
        float *out, const char *w_base, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks_dyn, uint32_t out_dim,
        uint32_t n_tok, uint32_t tile, uint32_t lv, uint32_t mb) {
    const uint32_t blocks = NB ? NB : blocks_dyn;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t n16 = blocks * 84u / 16u;
    extern __shared__ uint4 tshm[];
    cuda_block_q8_K *acts = (cuda_block_q8_K *)tshm;
    uint4 *wsh = (uint4 *)(acts + (uint64_t)tile * blocks);

    /* 块内统一的行迭代次数: 本块最小起始行 = blockIdx.x*8, 步长 gridDim.x*8 */
    const uint32_t base0 = blockIdx.x * 8u;
    const uint32_t stride = gridDim.x * 8u;
    const uint32_t n_iter = (base0 >= out_dim) ? 0u : ((out_dim - base0 + stride - 1u) / stride);

    for (uint32_t t0 = 0; t0 < n_tok; t0 += tile) {
        const uint32_t tn = (n_tok - t0 < tile) ? (n_tok - t0) : tile;
        __syncthreads();
        {   /* 协作搬 tn 个 token 的激活(uint4 宽拷) */
            const uint32_t words = (uint32_t)(((uint64_t)tn * blocks * sizeof(cuda_block_q8_K)) / sizeof(uint4));
            const uint4 *src = (const uint4 *)(xq + (uint64_t)t0 * blocks);
            uint4 *dst = (uint4 *)acts;
            for (uint32_t i = threadIdx.x; i < words; i += blockDim.x) dst[i] = src[i];
        }
        __syncthreads();
        for (uint32_t it = 0; it < n_iter; it++) {
            const uint32_t row = base0 + warp + it * stride;
            const bool live = row < out_dim;
            const cuda_block_q2_K *wr;
            if (live) {
                const uint4 *src16 = (const uint4 *)(w_base + (uint64_t)row * row_bytes);
                uint4 *my = wsh + (uint64_t)warp * n16;
                for (uint32_t i = lane; i < n16; i += 32u) my[i] = __ldcs(src16 + i);
                __syncwarp();
                wr = (const cuda_block_q2_K *)my;
            } else {
                wr = NULL;
            }
            for (uint32_t t = 0; t < tn; t++) {
                float a2 = 0.0f;
                if (live) {
                    const cuda_block_q8_K *xt = acts + (uint64_t)t * blocks;
                    if (blocks == 4u && lv >= 2u && mb >= 4u) {
                        const uint32_t bi = lane >> 3u, qi = lane & 7u;
                        a2 = dev_dot_q2_K_q8_K_block_eighth(wr + bi, xt + bi, qi);
                    } else if (blocks <= 16u && lv >= 2u && blocks <= mb) {
                        const uint32_t bi = lane >> 1u, h = lane & 1u;
                        if (bi < blocks) a2 = dev_dot_q2_K_q8_K_block_half(wr + bi, xt + bi, h);
                    } else {
                        for (uint32_t b = lane; b < blocks; b += 32u)
                            a2 += dev_dot_q2_K_q8_K_block_smem(wr + b, xt + b);
                    }
                }
                for (int off = 16; off > 0; off >>= 1) a2 += __shfl_down_sync(0xffffffffu, a2, off);
                if (live && lane == 0) out[(uint64_t)(t0 + t) * out_dim + row] = a2;
            }
            __syncwarp();
        }
    }
}

/* tiled kernel 的 shared opt-in(与 q2k_set_dynamic_smem 同款, 独立记账) */
static bool q2k_tiled_set_smem(uint32_t blocks, int bytes) {
    if (bytes <= 48 * 1024) return true;
    static int done[5] = {0, 0, 0, 0, 0};
    static int cap[5] = {0, 0, 0, 0, 0};
    const int idx = (blocks == 4u) ? 0 : (blocks == 8u) ? 1 : (blocks == 16u) ? 2 : (blocks == 32u) ? 3 : 4;
    if (done[idx]) return cap[idx] >= bytes;
    cudaError_t e = cudaErrorInvalidValue;
    switch (blocks) {
    case 4u:  e = cudaFuncSetAttribute(matmul_q2_K_tiled_kernel<4u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 8u:  e = cudaFuncSetAttribute(matmul_q2_K_tiled_kernel<8u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 16u: e = cudaFuncSetAttribute(matmul_q2_K_tiled_kernel<16u>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    case 32u: e = cudaFuncSetAttribute(matmul_q2_K_tiled_kernel<32u>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    default:  e = cudaFuncSetAttribute(matmul_q2_K_tiled_kernel<0u>,  cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); break;
    }
    done[idx] = 1; cap[idx] = (e == cudaSuccess) ? bytes : 0;
    if (e != cudaSuccess) { (void)cudaGetLastError(); return false; }
    return true;
}

static inline void q2k_tiled_launch(unsigned gx, size_t sh,
        float *out, const char *w, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks, uint32_t out_dim,
        uint32_t n_tok, uint32_t tile, uint32_t lv, uint32_t mb) {
    switch (blocks) {
    case 4u:  matmul_q2_K_tiled_kernel<4u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, n_tok, tile, lv, mb); break;
    case 8u:  matmul_q2_K_tiled_kernel<8u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, n_tok, tile, lv, mb); break;
    case 16u: matmul_q2_K_tiled_kernel<16u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, n_tok, tile, lv, mb); break;
    case 32u: matmul_q2_K_tiled_kernel<32u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, n_tok, tile, lv, mb); break;
    default:  matmul_q2_K_tiled_kernel<0u><<<gx, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, n_tok, tile, lv, mb); break;
    }
}

/* 特化分发: blocks∈{4,8,16,32} 走编译常量实例, 其他走动态 */
static inline void q2k_warp_launch(dim3 grid, size_t sh,
        float *out, const char *w, const cuda_block_q8_K *xq,
        uint64_t row_bytes, uint32_t blocks, uint32_t out_dim,
        uint32_t lv, uint32_t mb, const float *xraw, uint32_t ntb, uint32_t ntot,
        float *part, uint32_t ks, uint32_t xsm) {
    switch (blocks) {
    case 4u:  matmul_q2_K_warp_kernel<4u><<<grid, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, lv, mb, xraw, ntb, ntot, part, ks, xsm); break;
    case 8u:  matmul_q2_K_warp_kernel<8u><<<grid, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, lv, mb, xraw, ntb, ntot, part, ks, xsm); break;
    case 16u: matmul_q2_K_warp_kernel<16u><<<grid, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, lv, mb, xraw, ntb, ntot, part, ks, xsm); break;
    case 32u: matmul_q2_K_warp_kernel<32u><<<grid, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, lv, mb, xraw, ntb, ntot, part, ks, xsm); break;
    default:  matmul_q2_K_warp_kernel<0u><<<grid, 256, sh, g_cur_stream>>>(out, w, xq, row_bytes, blocks, out_dim, lv, mb, xraw, ntb, ntot, part, ks, xsm); break;
    }
}

int ds4_gpu_matmul_q2_K_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    /* dense Q2_K(2026-08-19): q4_K 入口克隆, 共用 q8_K 激活量化 scratch。 */
    if (!out || !model_map || !x || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (in_dim % CUDA_QK_K != 0) return 0;
    const uint32_t blocks = (uint32_t)(in_dim / CUDA_QK_K);
    const uint64_t row_bytes = (uint64_t)blocks * sizeof(cuda_block_q2_K);
    const uint64_t w_bytes = row_bytes * out_dim;
    if (weight_offset > model_size || w_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *w = cuda_model_range_ptr(model_map, weight_offset, w_bytes, "dense_q2k");
    if (!w) return 0;
    const uint64_t xq_need = n_tok * blocks * sizeof(cuda_block_q8_K);
    if (xq_need > g_q4k_xq_bytes) {
        cudaStreamCaptureStatus pcs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(0, &pcs);
        if (pcs != cudaStreamCaptureStatusNone) return 0;   /* capture 中不能 cudaMalloc */
        if (g_q4k_xq_sc) (void)cudaFree(g_q4k_xq_sc);
        g_q4k_xq_sc = NULL; g_q4k_xq_bytes = 0;
        if (cudaMalloc(&g_q4k_xq_sc, xq_need) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
        g_q4k_xq_bytes = xq_need;
    }
    static uint32_t q2fu = 99u;
    if (q2fu == 99u) { const char *e = ((const char *)0) /* DS4_Q2K_FUSEQ: 路径开关已删(2026-08-22 隐形炸弹清理) */; q2fu = e ? (uint32_t)atoi(e) : 1u; }
    const int fused = (q2fu && n_tok == 1);   /* 碎片税刀①: decode 融合量化免独立发射 */
    if (!fused)
        q8_K_quantize_kernel<<<dim3(blocks, (unsigned)n_tok, 1), 256, 0, g_cur_stream>>>(
            (cuda_block_q8_K *)g_q4k_xq_sc, (const float *)x->ptr, (uint32_t)in_dim, (uint32_t)n_tok);
    unsigned gx = (unsigned)((out_dim + 7u) / 8u);
    if (gx > ds4_grid_cap()) gx = ds4_grid_cap();   /* 08-20 阶梯审判: 384 比 192 +4-8GB/s(V1 222→229/V2 226→230) */
    size_t q2sh = (blocks <= 32u && (blocks & 3u) == 0u)
                        ? (size_t)8u * (blocks * 84u / 16u) * sizeof(uint4) : 0u;
    if (fused) q2sh += (size_t)blocks * sizeof(cuda_block_q8_K);
    static uint32_t q2lv = 99u;
    if (q2lv == 99u) { const char *e = ((const char *)0) /* DS4_Q2K_STAGE: 路径开关已删(2026-08-22 隐形炸弹清理) */; q2lv = e ? (uint32_t)atoi(e) : 2u; }
    if (((const char *)0) /* DS4_F16_DIMS: 路径开关已删(2026-08-22 隐形炸弹清理) */) {   /* 诊断: dense q2 形状(与 q4k 同款) */
        static uint64_t q2seen[64][2]; static int q2n = 0;
        int hit = 0;
        for (int i = 0; i < q2n; i++) if (q2seen[i][0] == in_dim && q2seen[i][1] == out_dim) { hit = 1; break; }
        if (!hit && q2n < 64) {
            q2seen[q2n][0] = in_dim; q2seen[q2n][1] = out_dim; q2n++;
            fprintf(stderr, "ds4: [q2k-dims] in=%llu out=%llu blocks=%u ntok=%llu\n",
                    (unsigned long long)in_dim, (unsigned long long)out_dim, blocks, (unsigned long long)n_tok);
        }
    }
    /* 刀⑤已回滚(2026-08-20): blocks≤4 rowlane 实测 33.52→30.45 且行为复读(疑数值), 撤。 */
    static uint32_t q2mb = 999u;
    if (q2mb == 999u) { const char *e = ((const char *)0) /* DS4_Q2K_SPLIT_MAXBLK: 路径开关已删(2026-08-22 隐形炸弹清理) */; q2mb = e ? (uint32_t)atoi(e) : 16u; }
    static uint32_t q2ca = 99u;
    if (q2ca == 99u) { const char *e = ((const char *)0) /* DS4_Q2K_CPASYNC: 路径开关已删(2026-08-22 隐形炸弹清理) */; q2ca = e ? (uint32_t)atoi(e) : 0u; }   /* 默认关: 实测7192→7646µs反向(混合访问实际墙~190, 非延迟问题) */
    if (q2ca && !fused && q2lv >= 2u && blocks <= 32u && (blocks & 3u) == 0u) {
        /* dot微架构刀①: cp.async 双缓冲(shared ×2), 数学与 staged 路逐位一致 */
        const size_t cash = (size_t)16u * (blocks * 84u / 16u) * sizeof(uint4);
        matmul_q2_K_warp_ca_kernel<<<dim3(gx, (unsigned)n_tok, 1), 256, cash, g_cur_stream>>>(
            (float *)out->ptr, w, (const cuda_block_q8_K *)g_q4k_xq_sc,
            row_bytes, blocks, (uint32_t)out_dim, q2mb);
        return cuda_ok(cudaGetLastError(), "dense q2_K cpasync matmul launch");
    }
    /* 大批走分片 GEMM(2026-08-21): 激活按 tile 进 shared, 块内 8 warp 共用。
     * 门限 DS4_Q2K_TILED_MIN(默认 16): 小批(decode/verify)继续走老路 —— 那条已与 decode
     * 逐位对齐, 不动。fused(n_tok==1 融合量化)与非 staging 档也不走。 */
    static uint32_t tmin = 0u;
    if (tmin == 0u) { const char *e = ((const char *)0) /* DS4_Q2K_TILED_MIN: 路径开关已删(2026-08-22 隐形炸弹清理) */; tmin = e ? (uint32_t)atoi(e) : 16u; }
    if (!fused && q2lv && n_tok >= tmin && blocks <= 32u && (blocks & 3u) == 0u &&
        1) {
        const size_t wsh_bytes = (size_t)8u * (blocks * 84u / 16u) * sizeof(uint4);
        const size_t budget = 96u * 1024u;
        uint32_t tile = (uint32_t)((budget - wsh_bytes) / ((size_t)blocks * sizeof(cuda_block_q8_K)));
        if (tile > n_tok) tile = (uint32_t)n_tok;
        if (tile > 64u) tile = 64u;
        if (tile >= 2u) {
            const size_t sh = wsh_bytes + (size_t)tile * blocks * sizeof(cuda_block_q8_K);
            if (q2k_tiled_set_smem(blocks, (int)sh)) {
                unsigned gx = (unsigned)((out_dim + 7u) / 8u);
                static uint32_t tgx = 0u;
                if (tgx == 0u) { const char *e = ((const char *)0) /* DS4_Q2K_TILED_BLOCKS: 路径开关已删(2026-08-22 隐形炸弹清理) */; tgx = e ? (uint32_t)atoi(e) : 512u; }
                if (gx > tgx) gx = tgx;
                q2k_tiled_launch(gx, sh, (float *)out->ptr, w, (const cuda_block_q8_K *)g_q4k_xq_sc,
                                 row_bytes, blocks, (uint32_t)out_dim, (uint32_t)n_tok, tile, q2lv, q2mb);
                return cuda_ok(cudaGetLastError(), "dense q2_K tiled matmul launch");
            }
        }
    }
    /* 权重驻留批: 多 token 时 grid.y=1, 块内循环 token(权重只读一遍)。
     * DS4_Q2K_NO_WSTAT=1 回旧 grid.y=n_tok 路。 */
    /* 门限(2026-08-21): 权重驻留省带宽但把 grid.y 折成 1 ⇒ 小矩阵并行度不足
     * (out=512 只剩 64 block / 48 SM)。仅当行块数够铺满设备时才驻留。 */
    static uint32_t wsmin = 0u;
    if (wsmin == 0u) { const char *e = ((const char *)0) /* DS4_Q2K_WSTAT_MINGX: 诊断开关已删(2026-08-22) */; wsmin = e ? (uint32_t)atoi(e) : 192u; }
    const uint32_t wstat = (n_tok > 1u && gx >= wsmin && ((const char *)0) /* DS4_Q2K_NO_WSTAT: 诊断开关已删(2026-08-22) */ == NULL) ? (uint32_t)n_tok : 1u;
    /* 驻留批解除 384 封顶(2026-08-21): 该上限是 decode(grid.y=n_tok 已铺满)的在飞调优;
     * 驻留批 grid.y=1 时封顶 ⇒ 每 warp 1.33 行的尾波(x16 kernel 精确覆盖零尾波是其
     * 126 vs 52 GB/s 的差之一)。 */
    if (wstat > 1u && 1) gx = (unsigned)((out_dim + 7u) / 8u);
    /* 分组数: 让 gx×组数 ≥ 目标块数(默认 1024 ≈ 48SM×4块×5波), 组内仍权重驻留。 */
    static uint32_t wsblk = 0u;
    if (wsblk == 0u) { const char *e = ((const char *)0) /* DS4_Q2K_WSTAT_BLOCKS: 诊断开关已删(2026-08-22) */; wsblk = e ? (uint32_t)atoi(e) : 1024u; }
    uint32_t ygroups = 1u, tpg = wstat;
    if (wstat > 1u && gx < wsblk) {
        ygroups = (wsblk + gx - 1u) / gx;
        if (ygroups > wstat) ygroups = wstat;
        tpg = (wstat + ygroups - 1u) / ygroups;
        ygroups = (wstat + tpg - 1u) / tpg;
    }
    /* K 切分: 仅在块数不足(小 out_dim)且 blocks 够分时启用; 默认阈值 1024 块。 */
    static uint32_t ksp_max = 99u, ksp_min_blocks = 0u;
    if (ksp_max == 99u) { const char *e = ((const char *)0) /* DS4_Q2K_KSPLIT: 路径开关已删(2026-08-22 隐形炸弹清理) */; ksp_max = e ? (uint32_t)atoi(e) : 1u; }
    if (ksp_min_blocks == 0u) { const char *e = ((const char *)0) /* DS4_Q2K_KSPLIT_BLOCKS: 路径开关已删(2026-08-22 隐形炸弹清理) */; ksp_min_blocks = e ? (uint32_t)atoi(e) : 1024u; }
    uint32_t ks = 1u;
    float *part = NULL;
    /* 只对走"整块 else 分支"的矩阵切 K(blocks > split_maxblk, 即 o_b 的 32 块):
     * 半块/八分拆路径的 lane 映射覆盖全部块, 分段后每个 z 会重复算全行 → 结果被乘
     * ksplit(2026-08-21 质量闸抓到, Δlogit 均值 14.3)。 */
    if (ksp_max > 1u && blocks > q2mb && gx < ksp_min_blocks && !fused) {
        ks = (ksp_min_blocks + gx - 1u) / gx;
        if (ks > ksp_max) ks = ksp_max;
        while (ks > 1u && (blocks / ks) < 4u) ks--;
        if (ks > 1u) {
            const uint64_t need = (uint64_t)ks * n_tok * out_dim * sizeof(float);
            cudaStreamCaptureStatus pcs2 = cudaStreamCaptureStatusNone;
            (void)cudaStreamIsCapturing(0, &pcs2);
            if (pcs2 != cudaStreamCaptureStatusNone) { ks = 1u; }
            else {
                if (need > g_q2k_part_bytes) {
                    if (g_q2k_part) (void)cudaFree(g_q2k_part);
                    g_q2k_part = NULL; g_q2k_part_bytes = 0;
                    if (cudaMalloc(&g_q2k_part, need) != cudaSuccess) { (void)cudaGetLastError(); ks = 1u; }
                    else g_q2k_part_bytes = need;
                }
                part = (float *)g_q2k_part;
                if (!part) ks = 1u;
            }
        }
    }
    /* 批激活也进 shared: 权重 staging 之后再放 tpg 份 q8 激活。超 48KB 要 opt-in,
     * 失败就退回全局读(数值一致, 只是慢)。 */
    const uint32_t tpg_use = (wstat > 1u) ? tpg : 1u;
    uint32_t xsm = 0u;
    size_t sh_use = (q2lv || fused) ? q2sh : 0u;
    if (!fused && q2lv && tpg_use > 1u && 1 &&
        blocks <= 32u && (blocks & 3u) == 0u) {
        const size_t xbytes = (size_t)tpg_use * blocks * sizeof(cuda_block_q8_K);
        const size_t want = q2sh + xbytes;
        if (want <= 200u * 1024u && q2k_set_dynamic_smem(blocks, (int)want)) {
            sh_use = want; xsm = 1u;
        }
    }
    q2k_warp_launch(dim3(gx, wstat > 1u ? ygroups : (unsigned)n_tok, ks), sh_use,
        (float *)out->ptr, w, (const cuda_block_q8_K *)g_q4k_xq_sc,
        row_bytes, blocks, (uint32_t)out_dim, q2lv, q2mb,
        fused ? (const float *)x->ptr : NULL, tpg_use, (uint32_t)n_tok, part, ks, xsm);
    if (ks > 1u) {
        const uint64_t n_elem = (uint64_t)out_dim * n_tok;
        q2k_ksplit_reduce_kernel<<<(unsigned)((n_elem + 255u) / 256u), 256, 0, g_cur_stream>>>(
            (float *)out->ptr, part, (uint32_t)out_dim, (uint32_t)n_tok, ks);
    }
    return cuda_ok(cudaGetLastError(), "dense q2_K matmul launch");
}

/* q2_K 版 decode 同输入矩阵对(2026-08-20 提速: quantize 风暴去重): 一次 q8_K 量化
 * 共享 scratch + 两次矩阵发射(q_a+kv / shexp gate+up 每层省一次量化+发射)。 */
static int q2k_matmul_from_xq(ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
                              uint64_t weight_offset, uint32_t blocks, uint64_t out_dim,
                              const float *xraw) {
    const uint64_t row_bytes = (uint64_t)blocks * sizeof(cuda_block_q2_K);
    const uint64_t w_bytes = row_bytes * out_dim;
    if (weight_offset > model_size || w_bytes > model_size - weight_offset) return 0;
    const char *w = cuda_model_range_ptr(model_map, weight_offset, w_bytes, "dense_q2k_pair");
    if (!w) return 0;
    unsigned gx = (unsigned)((out_dim + 7u) / 8u);
    if (gx > ds4_grid_cap()) gx = ds4_grid_cap();   /* 08-20 阶梯审判: 384 比 192 +4-8GB/s(V1 222→229/V2 226→230) */
    static uint32_t q2lv = 99u, q2mb = 999u;
    if (q2lv == 99u) { const char *e = ((const char *)0) /* DS4_Q2K_STAGE: 路径开关已删(2026-08-22 隐形炸弹清理) */; q2lv = e ? (uint32_t)atoi(e) : 2u; }
    if (q2mb == 999u) { const char *e = ((const char *)0) /* DS4_Q2K_SPLIT_MAXBLK: 路径开关已删(2026-08-22 隐形炸弹清理) */; q2mb = e ? (uint32_t)atoi(e) : 16u; }
    size_t q2sh = (blocks <= 32u && (blocks & 3u) == 0u)
                        ? (size_t)8u * (blocks * 84u / 16u) * sizeof(uint4) : 0u;
    if (xraw) q2sh += (size_t)blocks * sizeof(cuda_block_q8_K);
    q2k_warp_launch(dim3(gx, 1, 1), (q2lv || xraw) ? q2sh : 0u,
        (float *)out->ptr, w, (const cuda_block_q8_K *)g_q4k_xq_sc,
        row_bytes, blocks, (uint32_t)out_dim, q2lv, q2mb, xraw, 1u, 1u, NULL, 1u, 0u);
    return cuda_ok(cudaGetLastError(), "dense q2_K pair matmul launch");
}

int ds4_gpu_matmul_q2_K_pair_tensor(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
        const void *model_map, uint64_t model_size,
        uint64_t off0, uint64_t off1,
        uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
        const ds4_gpu_tensor *x) {
    if (!out0 || !out1 || !model_map || !x || in_dim == 0 || out0_dim == 0 || out1_dim == 0) return 0;
    if (in_dim % CUDA_QK_K != 0) return 0;
    const uint32_t blocks = (uint32_t)(in_dim / CUDA_QK_K);
    if (x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out0_dim * sizeof(float) || out1->bytes < out1_dim * sizeof(float)) return 0;
    const uint64_t xq_need = (uint64_t)blocks * sizeof(cuda_block_q8_K);
    if (xq_need > g_q4k_xq_bytes) {
        cudaStreamCaptureStatus pcs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(0, &pcs);
        if (pcs != cudaStreamCaptureStatusNone) return 0;
        if (g_q4k_xq_sc) (void)cudaFree(g_q4k_xq_sc);
        g_q4k_xq_sc = NULL; g_q4k_xq_bytes = 0;
        if (cudaMalloc(&g_q4k_xq_sc, xq_need) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
        g_q4k_xq_bytes = xq_need;
    }
    static uint32_t pfu = 99u;
    if (pfu == 99u) { const char *e = ((const char *)0) /* DS4_Q2K_FUSEQ: 路径开关已删(2026-08-22 隐形炸弹清理) */; pfu = e ? (uint32_t)atoi(e) : 1u; }
    if (!pfu)
        q8_K_quantize_kernel<<<dim3(blocks, 1, 1), 256, 0, g_cur_stream>>>(
            (cuda_block_q8_K *)g_q4k_xq_sc, (const float *)x->ptr, (uint32_t)in_dim, 1u);
    const float *xr = pfu ? (const float *)x->ptr : NULL;
    return q2k_matmul_from_xq(out0, model_map, model_size, off0, blocks, out0_dim, xr) &&
           q2k_matmul_from_xq(out1, model_map, model_size, off1, blocks, out1_dim, xr);
}

/* decode 同输入矩阵对: 一次 q8_K 量化 + 一次发射覆盖两矩阵 */
int ds4_gpu_matmul_q4_K_pair_tensor(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
        const void *model_map, uint64_t model_size,
        uint64_t off0, uint64_t off1,
        uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
        const ds4_gpu_tensor *x) {
    if (!out0 || !out1 || !model_map || !x || in_dim == 0 || out0_dim == 0 || out1_dim == 0) return 0;
    if (in_dim % CUDA_QK_K != 0) return 0;
    const uint32_t blocks = (uint32_t)(in_dim / CUDA_QK_K);
    const uint64_t row_bytes = (uint64_t)blocks * sizeof(cuda_block_q4_K);
    if (off0 > model_size || row_bytes * out0_dim > model_size - off0 ||
        off1 > model_size || row_bytes * out1_dim > model_size - off1) return 0;
    if (x->bytes < in_dim * sizeof(float) ||
        out0->bytes < out0_dim * sizeof(float) ||
        out1->bytes < out1_dim * sizeof(float)) return 0;
    const char *w0 = cuda_model_range_ptr(model_map, off0, row_bytes * out0_dim, "dense_q4k_p0");
    const char *w1 = cuda_model_range_ptr(model_map, off1, row_bytes * out1_dim, "dense_q4k_p1");
    if (!w0 || !w1) return 0;
    const uint64_t xq_need = (uint64_t)blocks * sizeof(cuda_block_q8_K);
    if (xq_need > g_q4k_xq_bytes) {
        cudaStreamCaptureStatus pcs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(0, &pcs);
        if (pcs != cudaStreamCaptureStatusNone) return 0;   /* capture 中不能 cudaMalloc */
        if (g_q4k_xq_sc) (void)cudaFree(g_q4k_xq_sc);
        g_q4k_xq_sc = NULL; g_q4k_xq_bytes = 0;
        if (cudaMalloc(&g_q4k_xq_sc, xq_need) != cudaSuccess) { (void)cudaGetLastError(); return 0; }
        g_q4k_xq_bytes = xq_need;
    }
    q8_K_quantize_kernel<<<dim3(blocks, 1, 1), 256, 0, g_cur_stream>>>(
        (cuda_block_q8_K *)g_q4k_xq_sc, (const float *)x->ptr, (uint32_t)in_dim, 1u);
    unsigned gx = (unsigned)((out0_dim + out1_dim + 7u) / 8u);
    if (gx > ds4_grid_cap()) gx = ds4_grid_cap();   /* 08-20 阶梯审判: 384 比 192 +4-8GB/s(V1 222→229/V2 226→230) */
    const size_t shmem = (blocks <= 32u) ? (size_t)8u * blocks * 9u * sizeof(uint4) : 0;
    matmul_q4_K_pair_warp_kernel<<<gx, 256, shmem, g_cur_stream>>>(
        (float *)out0->ptr, (float *)out1->ptr, w0, w1,
        (const cuda_block_q8_K *)g_q4k_xq_sc, row_bytes, blocks,
        (uint32_t)out0_dim, (uint32_t)out1_dim);
    return cuda_ok(cudaGetLastError(), "matmul_q4_K_pair launch");
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < xq_blocks; b += blockDim.x) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_warp8_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    uint32_t row = blockIdx.x * 8u + warp;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 32u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = warp_sum_f32(gate);
    up = warp_sum_f32(up);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_hwarp16_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 15u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 16u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = half_warp_sum_f32(gate, lane);
    up = half_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

/* 全 Q2_K 路由专家的 gate/up。引擎原有的 gate/up kernel 硬编码 cuda_block_iq2_xxs,
 * 只认 "gate=IQ2_XXS + down=Q2_K" 这一种组合; 而 Q2_K 每 16 元素一个 scale 直接解码,
 * 没有 IQ2_XXS 的 256 项 grid 查表(gather), 推理更快。down 路径本就支持 Q2_K, 解码函数
 * dev_dot_q2_K_q8_K_block 现成可用 —— 这里只是把同一套循环换到 Q2_K 的块布局上,
 * 行字节数由调用方的 gate_row_bytes 传入, 自适应两种块大小。 */
/* 向量化取块版 q2_K×q8_K dot: 块 84B 只保证 4B 对齐(84=21×4, 行距 672=42×16),
 * 用 uint32×21 整取进寄存器 —— 比逐字节 load 宽 4×。数学复用原 dot。 */
__device__ __forceinline__ static float dev_dot_q2_K_q8_K_block_vec(
        const cuda_block_q2_K *xg, const cuda_block_q8_K *y) {
    uint32_t v[21];
    #pragma unroll
    for (int i = 0; i < 21; i++) v[i] = ((const uint32_t *)xg)[i];
    return dev_dot_q2_K_q8_K_block((const cuda_block_q2_K *)v, y);
}

/* xq_blocks==16 专用拆分(2026-08-17 第五轮): 微基准判决 —— 同构+真dot 的隔离 kernel
 * REG:62 / 257 GB/s, 而合体 kernel 因泛化 else 分支被顶到 REG:128(2 块/SM) 只有 94GB/s。
 * 寄存器按最坏路径分配, 唯一解=物理拆 kernel。泛化配方仍走原 kernel。 */
__global__ static void moe_gate_up_mid_q2k_x16_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t tok = blockIdx.x, pk = blockIdx.z;
    const uint32_t m = blockIdx.y * 8u + warp;
    if (m >= expert_mid_dim) return;
    const uint64_t pair = (uint64_t)tok * n_expert + pk;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * 16u;

    __shared__ uint4 stage[8][2 * 16 * 84 / 16];      /* gate 行 84 uint4 + up 行 84 uint4 */
    const uint4 *gsrc = (const uint4 *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
    const uint4 *usrc = (const uint4 *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
    for (uint32_t i = lane; i < 84u; i += 32u) {
        stage[warp][i] = __ldcs(gsrc + i);             /* 流式读: 权重只过一遍, 别挤 L2 */
        stage[warp][84u + i] = __ldcs(usrc + i);
    }
    __syncwarp();
    const uint32_t half = lane >> 4u;                  /* 0=gate 1=up */
    const uint32_t l16 = lane & 15u;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)((const uint8_t *)stage[warp] + (uint64_t)half * 1344u);
    float acc = dev_dot_q2_K_q8_K_block_smem(wr + l16, xqb + l16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    const float up_v = __shfl_sync(0xffffffffu, acc, 16);
    if (lane == 0) {
        float gate = acc, up = up_v;
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = pair * expert_mid_dim + m;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[pair];
    }
}

/* A/B(2026-08-17): ncu 实测本 kernel Block Limit Registers=2(理论 occupancy 33%)。
 * __launch_bounds__(256,4) 压到 REG:64 但 STACK:224 溢出, e2e 21.0 vs 21.09 中性偏负,
 * 已回退; 真凶=else 泛化分支的 _vec dot(寄存器按最坏路径配, b5 不走它却买单),
 * 已换普通 dot 消压, 不再需要 launch_bounds。 */
__global__ static void moe_gate_up_mid_q2k_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    /* 重构(2026-08-17 第三轮): 原 8-lane/行×84B 步距访存不合并(该 kernel 16% 时间)。
     * 现: 每 warp 一行, gate+up 两行权重协作合并搬 shared; 32 lane 拆两半 —
     * lanes 0-15 算 gate 的 16 块, lanes 16-31 算 up 的 16 块, 双矩阵并行。
     * grid = (ntok, (MID+7)/8, nexp)。 */
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t tok = blockIdx.x, pk = blockIdx.z;
    const uint32_t m = blockIdx.y * 8u + warp;
    if (m >= expert_mid_dim) return;
    const uint64_t pair = (uint64_t)tok * n_expert + pk;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;

    float acc = 0.0f;
    if (xq_blocks == 16u) {
        __shared__ uint4 stage[8][2 * 16 * 84 / 16];      /* gate 行 84 uint4 + up 行 84 uint4 */
        const uint4 *gsrc = (const uint4 *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
        const uint4 *usrc = (const uint4 *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
        for (uint32_t i = lane; i < 84u; i += 32u) {
            stage[warp][i] = gsrc[i];
            stage[warp][84u + i] = usrc[i];
        }
        __syncwarp();
        const uint32_t half = lane >> 4u;                  /* 0=gate 1=up */
        const uint32_t l16 = lane & 15u;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)((const uint8_t *)stage[warp] + (uint64_t)half * 1344u);
        acc = dev_dot_q2_K_q8_K_block_smem(wr + l16, xqb + l16);
        acc += __shfl_down_sync(0xffffffffu, acc, 8);
        acc += __shfl_down_sync(0xffffffffu, acc, 4);
        acc += __shfl_down_sync(0xffffffffu, acc, 2);
        acc += __shfl_down_sync(0xffffffffu, acc, 1);
        const float up_v = __shfl_sync(0xffffffffu, acc, 16);
        if (lane == 0) {
            float gate = acc, up = up_v;
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = pair * expert_mid_dim + m;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[pair];
        }
    } else {
        /* 泛化回退: 逐块直读(xq_blocks != 16 的配方) */
        const cuda_block_q2_K *gr = (const cuda_block_q2_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
        const cuda_block_q2_K *ur = (const cuda_block_q2_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
        float g = 0.0f, u = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 32u) {
            g += dev_dot_q2_K_q8_K_block(gr + b, xqb + b);
            u += dev_dot_q2_K_q8_K_block(ur + b, xqb + b);
        }
        for (int off = 16; off > 0; off >>= 1) {
            g += __shfl_down_sync(0xffffffffu, g, off);
            u += __shfl_down_sync(0xffffffffu, u, off);
        }
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (g > clamp) g = clamp;
                if (u > clamp) u = clamp;
                if (u < -clamp) u = -clamp;
            }
            const uint64_t off2 = pair * expert_mid_dim + m;
            gate_out[off2] = g;
            up_out[off2] = u;
            mid_out[off2] = (g / (1.0f + expf(-g))) * u * weights[pair];
        }
    }
}

__global__ static void moe_gate_up_mid_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

/* IQ2_XXS 版 x16 staging(2026-08-17 第七夜): lut 版 8-lane×66B 步距散读实测 111 GB/s
 * (final86 最大单洞 233μs/层)。q2k x16 同款药: gate+up 行(各 1056B=66 uint4, 16B 对齐)
 * warp 协作 __ldcs 合并搬 shared, 半 warp 各算一矩阵 16 块, LUT/xq 块级共享。 */
__global__ static void moe_gate_up_mid_iq2_x16_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t tok = blockIdx.x, pk = blockIdx.z;
    const uint32_t m = blockIdx.y * 8u + warp;
    const uint64_t pair = (uint64_t)tok * n_expert + pk;
    /* A/B(2026-08-17 第九夜): 试过 LUT/xq 去 staging 全局直读 —— 31.4→22.1 大倒退回退。
     * LUT gather 是 warp 内随机索引, shared 是唯一不串行化的路; "28.7MB 搬运"是
     * 片上拷贝不占 DRAM, 账算错了。 */
    __shared__ uint64_t s_grid[256];
    __shared__ uint8_t s_signs[128];
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ uint4 stage[8][2 * 66];
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_grid[i] = cuda_iq2xxs_grid[i];
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_signs[i] = cuda_ksigns_iq2xs[i];
    {   /* xq 16 块 = 4672B, 按 uint32 粒度整块搬 */
        const uint32_t *src = (const uint32_t *)(xq + (uint64_t)tok * 16u);
        uint32_t *dst = (uint32_t *)sxq;
        const uint32_t n32 = (uint32_t)(16u * sizeof(cuda_block_q8_K) / 4u);
        for (uint32_t i = threadIdx.x; i < n32; i += blockDim.x) dst[i] = src[i];
    }
    __syncthreads();
    if (m >= expert_mid_dim) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    const uint32_t expert = (uint32_t)expert_i;
    const uint4 *gsrc = (const uint4 *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
    const uint4 *usrc = (const uint4 *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
    for (uint32_t i = lane; i < 66u; i += 32u) {
        stage[warp][i] = __ldcs(gsrc + i);
        stage[warp][66u + i] = __ldcs(usrc + i);
    }
    __syncwarp();
    const uint32_t half = lane >> 4u;                  /* 0=gate 1=up */
    const uint32_t l16 = lane & 15u;
    const cuda_block_iq2_xxs *wr = (const cuda_block_iq2_xxs *)((const uint8_t *)stage[warp] + (uint64_t)half * 1056u);
    float acc = dev_dot_iq2_xxs_q8_K_block_lut(wr + l16, sxq + l16, s_grid, s_signs);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    const float up_v = __shfl_sync(0xffffffffu, acc, 16);
    if (lane == 0) {
        float gate = acc, up = up_v;
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = pair * expert_mid_dim + m;
        if (write_aux) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[pair];
    }
}

/* 批版 x16 tile(2026-08-21 verify kernel 质量战): decode 的 x16 kernel 实测 215GB/s
 * 近峰值, 而 verify 走的 tile8_row32 只有 74GB/s —— 差距全在访存形状。本 kernel =
 * x16 结构(warp 驻留一行 + 32 lane 合并 uint4 staging + 16 lane 覆盖 16 块×2 矩阵)
 * 叠加 expert-tile 权重驻留(同专家的 np 个 token 复用已 staged 的行, 权重只读一遍)。
 * 逐 (pair,row) 数学与 tile8 路一致(同 dot、同归约序)。 */
__device__ static uint32_t g_x16_ldg_dev = 0;   /* A/B: staging 用 __ldg(L1 缓存) vs __ldcs(流式) */
__global__ static void moe_gate_up_mid_iq2_x16_tile_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp,
        uint32_t g_x16_ldg) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    /* tile 维 grid-stride(2026-08-21): 网格按上界开(37 列)但实际 tile 只有 ~15 ⇒ 59%
     * 空块白调度。改成按典型值开、块内循环补齐, 空块归零。 */
    const uint32_t n_tile = *tile_total;
    for (uint32_t tile = blockIdx.y; tile < n_tile; tile += gridDim.y) {
    const uint32_t expert = tile_experts[tile];
    const uint32_t local_start = tile_starts[tile];
    uint32_t pair_id[8], tokid[8], np = 0;
    for (; np < 8u; np++) {
        const uint32_t lp = local_start + np;
        if (lp >= counts[expert]) break;
        pair_id[np] = sorted_pairs[offsets[expert] + lp];
        tokid[np] = pair_id[np] / n_expert;
    }
    __shared__ uint64_t s_grid[256];
    __shared__ uint8_t s_signs[128];
    __shared__ uint4 stage[8][2 * 66];
    /* 激活 staging(2026-08-21): 每行都从全局重读 np×16 块激活 = 每行 84B 权重配 ~5KB
     * 激活(21x 放大, 全靠 L1 兜)。tile 内 np 个 token 的激活对整块 8 warp 共用 ⇒ 一次
     * 搬进 shared(np≤6 时 28KB), decode x16 同款药。 */
    __shared__ cuda_block_q8_K sxq[6][16];
    const uint32_t np_sh = (np <= 6u) ? np : 0u;
    for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_grid[i] = cuda_iq2xxs_grid[i];
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_signs[i] = cuda_ksigns_iq2xs[i];
    if (np_sh) {
        const uint32_t n32 = (uint32_t)(16u * sizeof(cuda_block_q8_K) / 4u);
        for (uint32_t p = 0; p < np_sh; p++) {
            const uint32_t *src = (const uint32_t *)(xq + (uint64_t)tokid[p] * 16u);
            uint32_t *dst = (uint32_t *)sxq[p];
            for (uint32_t i = threadIdx.x; i < n32; i += blockDim.x) dst[i] = src[i];
        }
    }
    __syncthreads();
    const uint32_t m = blockIdx.x * 8u + warp;
    if (m >= expert_mid_dim || np == 0u) { __syncthreads(); continue; }   /* 下轮 tile 前对齐 */
    const uint4 *gsrc = (const uint4 *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
    const uint4 *usrc = (const uint4 *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)m * gate_row_bytes);
    if (g_x16_ldg) {
        for (uint32_t i = lane; i < 66u; i += 32u) {
            stage[warp][i] = __ldg(gsrc + i);
            stage[warp][66u + i] = __ldg(usrc + i);
        }
    } else {
        for (uint32_t i = lane; i < 66u; i += 32u) {
            stage[warp][i] = __ldcs(gsrc + i);
            stage[warp][66u + i] = __ldcs(usrc + i);
        }
    }
    __syncwarp();
    const uint32_t half = lane >> 4u, l16 = lane & 15u;
    const cuda_block_iq2_xxs *wr = (const cuda_block_iq2_xxs *)((const uint8_t *)stage[warp] + (uint64_t)half * 1056u);
    /* LUT 解码一次 · 点乘 np 次(2026-08-21): 每 pair 单独调 _block_lut 会把 iq2 网格
     * 解码(shared 随机索引, bank 冲突重)重复 np 遍 —— 本 kernel 的真实瓶颈。 */
    const cuda_block_q8_K *yb[8];
    for (uint32_t p = 0; p < 8u; p++)
        yb[p] = (p < np) ? (np_sh ? (sxq[p] + l16) : (xq + (uint64_t)tokid[p] * 16u + l16)) : NULL;
    float acc8[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    dev_dot_iq2_xxs_q8_K_block8_deq_lut(wr + l16, yb[0], yb[1], yb[2], yb[3], yb[4], yb[5], yb[6], yb[7],
                                        np, acc8, s_grid, s_signs);
    for (uint32_t p = 0; p < np; p++) {
        float acc = acc8[p];
        acc += __shfl_down_sync(0xffffffffu, acc, 8);
        acc += __shfl_down_sync(0xffffffffu, acc, 4);
        acc += __shfl_down_sync(0xffffffffu, acc, 2);
        acc += __shfl_down_sync(0xffffffffu, acc, 1);
        const float up_v = __shfl_sync(0xffffffffu, acc, 16);
        if (lane == 0) {
            float gate = acc, up = up_v;
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair_id[p] * expert_mid_dim + m;
            if (write_aux) { gate_out[off] = gate; up_out[off] = up; }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[pair_id[p]];
        }
    }
    __syncthreads();   /* 下轮 tile 复写 stage/sxq 前全块对齐 */
    }
}

__global__ static void moe_gate_up_mid_decode_lut_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    __shared__ cuda_block_q8_K sxq[16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < xq_blocks; i += blockDim.x) sxq[i] = xqb[i];
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        xqb = sxq;
    }
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block_lut(gr + b, xqb + b, s_iq2_grid, s_iq2_signs);
            up += dev_dot_iq2_xxs_q8_K_block_lut(ur + b, xqb + b, s_iq2_grid, s_iq2_signs);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_count_sorted_pairs_kernel(
        uint32_t *counts,
        const int32_t *selected,
        uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    atomicAdd(counts + (uint32_t)expert_i, 1u);
}

__global__ static void moe_prefix_sorted_pairs_kernel(
        uint32_t *offsets,
        uint32_t *cursors,
        const uint32_t *counts) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < 256u; e++) {
            offsets[e] = sum;
            cursors[e] = sum;
            sum += counts[e];
        }
        offsets[256] = sum;
    }
}

__global__ static void moe_scatter_sorted_pairs_kernel(
        uint32_t *sorted_pairs,
        uint32_t *cursors,
        const int32_t *selected,
        uint32_t pair_count) {
    uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0) expert_i = 0;
    uint32_t pos = atomicAdd(cursors + (uint32_t)expert_i, 1u);
    sorted_pairs[pos] = pair;
}

__global__ static void moe_build_expert_tile_offsets_kernel(
        uint32_t *tile_offsets,
        uint32_t *tile_total,
        const uint32_t *counts,
        uint32_t block_m) {
    if (threadIdx.x == 0) {
        uint32_t sum = 0;
        for (uint32_t e = 0; e < 256u; e++) {
            tile_offsets[e] = sum;
            sum += (counts[e] + block_m - 1u) / block_m;
        }
        tile_offsets[256] = sum;
        *tile_total = sum;
    }
}

__global__ static void moe_build_expert_tiles_kernel(
        uint32_t *tile_experts,
        uint32_t *tile_starts,
        const uint32_t *tile_offsets,
        const uint32_t *counts,
        uint32_t block_m) {
    uint32_t e = threadIdx.x;
    if (e >= 256u) return;
    uint32_t ntiles = (counts[e] + block_m - 1u) / block_m;
    uint32_t off = tile_offsets[e];
    for (uint32_t t = 0; t < ntiles; t++) {
        tile_experts[off + t] = e;
        tile_starts[off + t] = t * block_m;
    }
}

__global__ static void moe_gate_up_mid_sorted_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_gate_up_mid_expert_tile8_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t group = threadIdx.x >> 3u;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_slot = group & 7u;
    uint32_t row_lane = group >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_pair = tile_starts[tile] + pair_slot;
    if (local_pair >= counts[expert]) return;
    uint32_t sorted_idx = offsets[expert] + local_pair;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;

    for (uint32_t rr = 0; rr < 2u; rr++) {
        uint32_t row = blockIdx.x * 8u + row_lane + rr * 4u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
            up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            gate_out[off] = gate;
            up_out[off] = up;
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile4_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][16];
    uint32_t pair[4] = {0, 0, 0, 0};
    uint32_t tok[4] = {0, 0, 0, 0};
    uint32_t slot[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block4(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, gate);
        dev_dot_iq2_xxs_q8_K_block4(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, up);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

template<bool USE_SMEM>
__global__ static void moe_gate_up_mid_expert_tile8_row32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[USE_SMEM ? 8 : 1][USE_SMEM ? 16 : 1];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        if (USE_SMEM) {
            for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
                uint32_t p = i / xq_blocks;
                uint32_t b = i - p * xq_blocks;
                sxq[p][b] = xqb[p][b];
            }
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        if (USE_SMEM) for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                            s_iq2_grid, s_iq2_signs);
        dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                            xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                            xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                            xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                            s_iq2_grid, s_iq2_signs);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p], lane);
        up[p] = quarter_warp_sum_f32(up[p], lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
        }
    }
}

__global__ static void moe_gate_up_mid_expert_tile8_row2048_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                                s_iq2_grid, s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                                s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_gate_up_mid_expert_tile8_rowspan_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][16];
    __shared__ uint64_t s_iq2_grid[256];
    __shared__ uint8_t s_iq2_signs[128];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / n_expert;
        slot[np] = pair[np] - tok[np] * n_expert;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            uint32_t p = i / xq_blocks;
            uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        for (uint32_t i = threadIdx.x; i < 256u; i += blockDim.x) s_iq2_grid[i] = cuda_iq2xxs_grid[i];
        for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) s_iq2_signs[i] = cuda_ksigns_iq2xs[i];
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(gr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, gate,
                                                s_iq2_grid, s_iq2_signs);
            dev_dot_iq2_xxs_q8_K_block8_deq_lut(ur + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                                xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                                xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                                xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, up,
                                                s_iq2_grid, s_iq2_signs);
        }
        for (uint32_t p = 0; p < np; p++) {
            gate[p] = quarter_warp_sum_f32(gate[p], lane);
            up[p] = quarter_warp_sum_f32(up[p], lane);
            if (lane == 0) {
                if (clamp > 1.0e-6f) {
                    if (gate[p] > clamp) gate[p] = clamp;
                    if (up[p] > clamp) up[p] = clamp;
                    if (up[p] < -clamp) up[p] = -clamp;
                }
                const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
                if (write_aux) {
                    gate_out[off] = gate[p];
                    up_out[off] = up[p];
                }
                mid_out[off] = (gate[p] / (1.0f + expf(-gate[p]))) * up[p] * weights[(uint64_t)tok[p] * n_expert + slot[p]];
            }
        }
    }
}

__global__ static void moe_gate_up_mid_sorted_p2_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t pair_count,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= expert_mid_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dev_dot_iq2_xxs_q8_K_block(gr + b, xqb + b);
        up += dev_dot_iq2_xxs_q8_K_block(ur + b, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate, lane);
    up = quarter_warp_sum_f32(up, lane);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static DS4_CUDA_UNUSED void moe_down_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < midq_blocks; b += blockDim.x) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)pair * out_dim + row] = partial[0];
}

__global__ static DS4_CUDA_UNUSED void moe_down_warp8_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 31u;
    uint32_t warp = threadIdx.x >> 5u;
    uint32_t row = blockIdx.x * 8u + warp;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 32u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = warp_sum_f32(acc);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static DS4_CUDA_UNUSED void moe_down_hwarp16_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 15u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 16u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = half_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_down_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_down_q4K_pairs_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_gate_up_mid_decode_q4K_qwarp32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const cuda_block_q8_K *xq,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        uint32_t write_aux,
        float clamp) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t pair = blockIdx.y;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const cuda_block_q8_K *xqb = xq + (uint64_t)tok * xq_blocks;
    for (uint32_t rr = 0; rr < 4u; rr++) {
        uint32_t row = blockIdx.x * 128u + row_lane + rr * 32u;
        if (row >= expert_mid_dim) continue;
        const cuda_block_q4_K *gr = (const cuda_block_q4_K *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        const cuda_block_q4_K *ur = (const cuda_block_q4_K *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
        float gate = 0.0f;
        float up = 0.0f;
        for (uint32_t b = lane; b < xq_blocks; b += 8u) {
            gate += dev_dot_q4_K_q8_K_block(gr + b, xqb + b);
            up += dev_dot_q4_K_q8_K_block(ur + b, xqb + b);
        }
        gate = quarter_warp_sum_f32(gate, lane);
        up = quarter_warp_sum_f32(up, lane);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate > clamp) gate = clamp;
                if (up > clamp) up = clamp;
                if (up < -clamp) up = -clamp;
            }
            const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
            if (write_aux) {
                gate_out[off] = gate;
                up_out[off] = up;
            }
            mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
        }
    }
}

/* per-expert 纯流版(2026-08-17 第五轮): sum6 版每行内层串行跳 6 个专家矩阵, 块内访问
 * 被切成 6 路交错流, 实测 149 GB/s(微基准同构纯流=255)。拆成: 每 block 只读一个专家
 * 的连续 32 行(21.5KB 纯顺序) 写 partial[slot][row], 再由 reduce kernel 按 slot 固定
 * 顺序求和 —— 求和顺序与 sum6 版逐位同义(slot 0..5), 无数值漂移。 */
__global__ static void moe_down_partial_qwarp32_kernel(
        float *partial,                    /* [6][out_dim] */
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t row0 = blockIdx.x * 32u + warp * 4u;   /* 每 warp 4 连续行(同专家 2688B) */
    const uint32_t slot = blockIdx.y;      /* 0..5 */
    if (row0 >= out_dim) return;
    int32_t expert_i = selected[slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
    if (midq_blocks == 8u && (out_dim & 31u) == 0u) {
        /* staging: warp 协作把 4 行×672B=168 uint4 合并搬 shared, 再 4 个 quarter-warp
         * 各算一行 —— gateup x16 同款药(该 kernel 184 vs 本处散读 149 GB/s 的差就是它) */
        __shared__ uint4 stage[8][168];
        const uint4 *src16 = (const uint4 *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row0 * down_row_bytes);
        for (uint32_t i = lane; i < 168u; i += 32u) stage[warp][i] = __ldcs(src16 + i);
        __syncwarp();
        const uint32_t q = lane >> 3u, l8 = lane & 7u;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)((const uint8_t *)stage[warp] + (uint64_t)q * 672u);
        float acc = dev_dot_q2_K_q8_K_block(wr + l8, xq + l8);
        acc = quarter_warp_sum_f32(acc, l8);
        if (l8 == 0) partial[(uint64_t)slot * out_dim + row0 + q] = acc;
    } else {
        const uint32_t l8 = threadIdx.x & 7u;
        const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
        if (row >= out_dim) return;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc = 0.0f;
        for (uint32_t b = l8; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, l8);
        if (l8 == 0) partial[(uint64_t)slot * out_dim + row] = acc;
    }
}

__global__ static void moe_down_partial_reduce6_kernel(float *out, const float *partial, uint32_t out_dim) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float t = 0.0f;
    #pragma unroll
    for (uint32_t s = 0; s < 6u; s++) t += partial[(uint64_t)s * out_dim + row];
    out[row] = t;
}

__global__ static void moe_down_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    /* A/B(2026-08-17): 试过 32-lane/行 + 6 专家行 shared staging(同 dense q4 的合并搬运),
     * 实测 20.65 vs 本版 20.78 t/s —— 672B 短行×6 次串行搬运的开销盖过合并收益,
     * 且本版 4 warp/块行数更多利延迟隐藏。回退保留 8-lane 原状。 */
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_q4K_sum6_qwarp32_kernel(
        float *out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    if (row >= out_dim) return;
    float total = 0.0f;
    #pragma unroll
    for (uint32_t slot = 0; slot < 6u; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0) expert_i = 0;
        const cuda_block_q4_K *wr = (const cuda_block_q4_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
        const cuda_block_q8_K *xq = midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) total += acc;
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_down_sorted_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static DS4_CUDA_UNUSED void moe_down_expert_tile8_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t group = threadIdx.x >> 3u;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_slot = group & 7u;
    uint32_t row_lane = group >> 3u;
    uint32_t expert = tile_experts[tile];
    uint32_t local_pair = tile_starts[tile] + pair_slot;
    if (local_pair >= counts[expert]) return;
    uint32_t sorted_idx = offsets[expert] + local_pair;
    uint32_t pair = sorted_pairs[sorted_idx];
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;

    for (uint32_t rr = 0; rr < 2u; rr++) {
        uint32_t row = blockIdx.x * 8u + row_lane + rr * 4u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
        acc = quarter_warp_sum_f32(acc, lane);
        if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
    }
}

__global__ static void moe_down_expert_tile4_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[4][8];
    uint32_t pair[4] = {0, 0, 0, 0};
    const cuda_block_q8_K *xqb[4] = {NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 4u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block4(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile8_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    uint32_t local_start = tile_starts[tile];
    __shared__ cuda_block_q8_K sxq[8][8];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const cuda_block_q8_K *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

/* 批版 down tile(2026-08-21): decode 的 qwarp32 staging 版 212GB/s vs 本批 tile8 散读
 * 60GB/s。同款药: warp 协作把 4 连续行(168 uint4=2688B)合并搬 shared, 4 个 quarter-warp
 * 各算一行, 再对 tile 内 np 个 pair 复用同一份已 staged 权重。 */
__global__ static void moe_down_expert_tile_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    const uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    const uint32_t row0 = blockIdx.x * 32u + warp * 4u;
    const uint32_t expert = tile_experts[tile];
    const uint32_t local_start = tile_starts[tile];
    uint32_t pair[8]; uint32_t np = 0;
    for (; np < 8u; np++) {
        const uint32_t lp = local_start + np;
        if (lp >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + lp];
    }
    __shared__ uint4 stage[8][168];
    __shared__ cuda_block_q8_K smid[6][8];   /* tile 内 np 个 pair 的 mid 激活(8 块/行) */
    const uint32_t np_sh = (np <= 6u && midq_blocks == 8u) ? np : 0u;
    if (np_sh) {
        const uint32_t n32 = (uint32_t)(8u * sizeof(cuda_block_q8_K) / 4u);
        for (uint32_t pp = 0; pp < np_sh; pp++) {
            const uint32_t *src = (const uint32_t *)(midq + (uint64_t)pair[pp] * midq_blocks);
            uint32_t *dst = (uint32_t *)smid[pp];
            for (uint32_t i = threadIdx.x; i < n32; i += blockDim.x) dst[i] = src[i];
        }
    }
    __syncthreads();
    if (row0 >= out_dim || np == 0u) return;
    const uint4 *src16 = (const uint4 *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row0 * down_row_bytes);
    for (uint32_t i = lane; i < 168u; i += 32u) stage[warp][i] = __ldcs(src16 + i);
    __syncwarp();
    const uint32_t q = lane >> 3u, l8 = lane & 7u;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)((const uint8_t *)stage[warp] + (uint64_t)q * 672u);
    for (uint32_t pi = 0; pi < np; pi++) {
        const cuda_block_q8_K *xq = np_sh ? smid[pi] : (midq + (uint64_t)pair[pi] * midq_blocks);
        float acc = dev_dot_q2_K_q8_K_block(wr + l8, xq + l8);
        acc = quarter_warp_sum_f32(acc, l8);
        if (l8 == 0) {
            const uint32_t row = row0 + q;
            if (atomic_out) {
                const uint32_t tok = pair[pi] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc);
            } else {
                down_out[(uint64_t)pair[pi] * out_dim + row] = acc;
            }
        }
    }
}

__global__ static void moe_down_expert_tile16_row32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
    float acc[16] = {0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                 xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                 xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                 xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
        if (np > 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                     xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                     xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                     xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
        }
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p], lane);
        if (lane == 0) {
            if (atomic_out) {
                uint32_t tok = pair[p] / n_expert;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_down_expert_tile16_row2048_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < 64u; rr++) {
        uint32_t row = blockIdx.x * 2048u + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                     xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                     xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                     xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
            if (np > 8u) {
                dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                         xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                         xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                         xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

template <uint32_t ROW_SPAN>
__global__ static void moe_down_expert_tile16_rowspan_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t atomic_out) {
    uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    uint32_t local_start = tile_starts[tile];
    if (local_start & 8u) return;
    uint32_t lane = threadIdx.x & 7u;
    uint32_t row_lane = threadIdx.x >> 3u;
    uint32_t expert = tile_experts[tile];
    __shared__ cuda_block_q8_K sxq[16][8];
    uint32_t pair[16] = {0};
    const cuda_block_q8_K *xqb[16] = {NULL};
    uint32_t np = 0;
    for (; np < 16u; np++) {
        uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            uint32_t p = i / midq_blocks;
            uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    for (uint32_t rr = 0; rr < ROW_SPAN / 32u; rr++) {
        uint32_t row = blockIdx.x * ROW_SPAN + row_lane + rr * 32u;
        if (row >= out_dim) continue;
        const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
        float acc[16] = {0.0f};
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            dev_dot_q2_K_q8_K_block8(wr + b, xqb[0] ? xqb[0] + b : NULL, xqb[1] ? xqb[1] + b : NULL,
                                     xqb[2] ? xqb[2] + b : NULL, xqb[3] ? xqb[3] + b : NULL,
                                     xqb[4] ? xqb[4] + b : NULL, xqb[5] ? xqb[5] + b : NULL,
                                     xqb[6] ? xqb[6] + b : NULL, xqb[7] ? xqb[7] + b : NULL, np < 8u ? np : 8u, acc);
            if (np > 8u) {
                dev_dot_q2_K_q8_K_block8(wr + b, xqb[8] ? xqb[8] + b : NULL, xqb[9] ? xqb[9] + b : NULL,
                                         xqb[10] ? xqb[10] + b : NULL, xqb[11] ? xqb[11] + b : NULL,
                                         xqb[12] ? xqb[12] + b : NULL, xqb[13] ? xqb[13] + b : NULL,
                                         xqb[14] ? xqb[14] + b : NULL, xqb[15] ? xqb[15] + b : NULL, np - 8u, acc + 8);
            }
        }
        for (uint32_t p = 0; p < np; p++) {
            acc[p] = quarter_warp_sum_f32(acc[p], lane);
            if (lane == 0) {
                if (atomic_out) {
                    uint32_t tok = pair[p] / n_expert;
                    atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
                } else {
                    down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
                }
            }
        }
    }
}

__global__ static void moe_down_sorted_p2_qwarp32_kernel(
        float *down_out,
        const char *down_base,
        const cuda_block_q8_K *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_expert,
        uint32_t pair_count) {
    uint32_t lane = threadIdx.x & 7u;
    uint32_t pair_lane = (threadIdx.x >> 3u) & 1u;
    uint32_t row = blockIdx.x * 16u + (threadIdx.x >> 4u);
    uint32_t sorted_idx = blockIdx.y * 2u + pair_lane;
    if (row >= out_dim || sorted_idx >= pair_count) return;
    uint32_t pair = sorted_pairs[sorted_idx];
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const cuda_block_q8_K *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) acc += dev_dot_q2_K_q8_K_block(wr + b, xq + b);
    acc = quarter_warp_sum_f32(acc, lane);
    if (lane == 0) down_out[(uint64_t)pair * out_dim + row] = acc;
}

__global__ static void moe_sum_kernel(float *out, const float *down, uint32_t out_dim, uint32_t n_expert, uint32_t n_tokens) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * out_dim;
    if (gid >= n) return;
    uint32_t tok = gid / out_dim;
    uint32_t row = gid - (uint64_t)tok * out_dim;
    float acc = 0.0f;
    for (uint32_t e = 0; e < n_expert; e++) acc += down[((uint64_t)tok * n_expert + e) * out_dim + row];
    out[gid] = acc;
}

__device__ static float dev_iq2_xxs_dot_f32(const cuda_block_iq2_xxs *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_iq2_xxs *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const uint16_t *q2 = xb->qs;
        const float *xf = x + (uint64_t)b * CUDA_QK_K;
        for (uint32_t ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
            const uint32_t aux_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
            const uint32_t aux_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
            q2 += 4;
            const float dl = d * (0.5f + (float)(aux_s >> 28)) * 0.25f;
            const uint8_t grids[4] = {
                (uint8_t)(aux_g & 0xffu),
                (uint8_t)((aux_g >> 8) & 0xffu),
                (uint8_t)((aux_g >> 16) & 0xffu),
                (uint8_t)((aux_g >> 24) & 0xffu),
            };
            for (uint32_t half = 0; half < 2; half++) {
                for (uint32_t g = 0; g < 2; g++) {
                    const uint32_t gi = half * 2 + g;
                    const uint64_t grid = cuda_iq2xxs_grid[grids[gi]];
                    const uint8_t signs = cuda_ksigns_iq2xs[(aux_s >> (14u * half + 7u * g)) & 127u];
                    for (uint32_t i = 0; i < 8; i++) {
                        float w = (float)((grid >> (8u * i)) & 0xffu);
                        if (signs & (1u << i)) w = -w;
                        acc += dl * w * xf[ib32 * 32u + half * 16u + g * 8u + i];
                    }
                }
            }
        }
    }
    return acc;
}

__device__ static float dev_q2_K_dot_f32(const cuda_block_q2_K *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const cuda_block_q2_K *xb = row + b;
        const float d = dev_f16_to_f32(xb->d);
        const float dmin = dev_f16_to_f32(xb->dmin);
        for (uint32_t il = 0; il < 16; il++) {
            const uint32_t chunk = il / 8u;
            const uint32_t pair = il & 1u;
            const uint32_t shift = ((il / 2u) & 3u) * 2u;
            const uint8_t sc = xb->scales[il];
            const float dl = d * (float)(sc & 0x0fu);
            const float ml = dmin * (float)(sc >> 4);
            const uint8_t *q = xb->qs + 32u * chunk + 16u * pair;
            const float *xf = x + (uint64_t)b * CUDA_QK_K + chunk * 128u + ((il % 8u) / 2u) * 32u + pair * 16u;
            for (uint32_t i = 0; i < 16; i++) {
                const float w = dl * (float)((q[i] >> shift) & 3u) - ml;
                acc += w * xf[i];
            }
        }
    }
    return acc;
}

__global__ static void moe_gate_up_mid_f32_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const char *gate_base,
        const char *up_base,
        const float *x,
        const int32_t *selected,
        const float *weights,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t n_expert,
        float clamp) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= expert_mid_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    uint32_t expert = (uint32_t)expert_i;
    const uint32_t nb = expert_in_dim / CUDA_QK_K;
    const cuda_block_iq2_xxs *gr = (const cuda_block_iq2_xxs *)(gate_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const cuda_block_iq2_xxs *ur = (const cuda_block_iq2_xxs *)(up_base + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
    const float *xr = x + (uint64_t)tok * expert_in_dim;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) {
        gate += dev_iq2_xxs_dot_f32(gr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
        up += dev_iq2_xxs_dot_f32(ur + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    }
    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        gate_out[off] = gate;
        up_out[off] = up;
        mid_out[off] = (gate / (1.0f + expf(-gate))) * up * weights[(uint64_t)tok * n_expert + slot];
    }
}

__global__ static void moe_down_f32_kernel(
        float *down_out,
        const char *down_base,
        const float *mid,
        const int32_t *selected,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        uint32_t n_expert) {
    uint32_t row = blockIdx.x;
    uint32_t pair = blockIdx.y;
    if (row >= out_dim) return;
    uint32_t tok = pair / n_expert;
    uint32_t slot = pair - tok * n_expert;
    int32_t expert_i = selected[(uint64_t)tok * n_expert + slot];
    if (expert_i < 0) expert_i = 0;
    const uint32_t nb = expert_mid_dim / CUDA_QK_K;
    const cuda_block_q2_K *wr = (const cuda_block_q2_K *)(down_base + (uint64_t)(uint32_t)expert_i * down_expert_bytes + (uint64_t)row * down_row_bytes);
    const float *xr = mid + (uint64_t)pair * expert_mid_dim;
    float acc = 0.0f;
    for (uint32_t b = threadIdx.x; b < nb; b += blockDim.x) acc += dev_q2_K_dot_f32(wr + b, xr + (uint64_t)b * CUDA_QK_K, 1);
    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)pair * out_dim + row] = partial[0];
}

static int routed_moe_launch(
        ds4_gpu_tensor *out,
        ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid,
        ds4_gpu_tensor *down,
        const void *model_map,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset,
        uint32_t gate_type,
        uint32_t down_type,
        uint64_t gate_expert_bytes,
        uint64_t gate_row_bytes,
        uint64_t down_expert_bytes,
        uint64_t down_row_bytes,
        uint32_t expert_in_dim,
        uint32_t expert_mid_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t n_total_expert,
        uint32_t n_expert,
        float clamp,
        const ds4_gpu_tensor *x,
        uint32_t n_tokens) {
    if (!out || !gate || !up || !mid || !down || !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 ||
        expert_in_dim % CUDA_QK_K != 0 || expert_mid_dim % CUDA_QK_K != 0 ||
        gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        gate->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        up->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        mid->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        down->bytes < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
        return 0;
    }
    { static int dbg_n = 0;
      if (dbg_n < 2000 && ((const char *)0) /* DS4_MOE_CALL_DBG: 路径开关已删(2026-08-22 隐形炸弹清理) */) { dbg_n++;
          fprintf(stderr, "ds4: [moe-call] ntok=%u gate_t=%u down_t=%u\n", n_tokens, gate_type, down_type); } }
    /* 专家并集实测(2026-08-21 物理地板对账): 每层把 selected 拉回主机数一次唯一专家。 */
    if (((const char *)0) /* DS4_MOE_UNION_DBG: 路径开关已删(2026-08-22 隐形炸弹清理) */ && n_tokens >= 2u && n_tokens <= 16u) {
        static uint64_t u_sum = 0, u_n = 0, p_sum = 0;
        int32_t sel_h[16 * 8];
        const size_t nb = (size_t)n_tokens * n_expert * sizeof(int32_t);
        if (nb <= sizeof(sel_h) &&
            cudaMemcpy(sel_h, selected->ptr, nb, cudaMemcpyDeviceToHost) == cudaSuccess) {
            bool seen[512] = {false};
            uint32_t uniq = 0;
            for (uint32_t i = 0; i < n_tokens * n_expert; i++) {
                const int32_t e = sel_h[i];
                if (e >= 0 && e < 512 && !seen[e]) { seen[e] = true; uniq++; }
            }
            u_sum += uniq; p_sum += (uint64_t)n_tokens * n_expert; u_n++;
            if ((u_n % 43u) == 0)
                fprintf(stderr, "ds4: [moe-union] ntok=%u picks=%u uniq_avg=%.1f (n=%llu)\n",
                        n_tokens, n_tokens * n_expert, (double)u_sum / (double)u_n,
                        (unsigned long long)u_n);
        }
    }
    const int q4k_path = (gate_type == 12u && down_type == 12u);
    const int q2k_path = (gate_type == 10u && down_type == 10u);   /* 全 Q2_K 专家 */
    {   static int once = 0;
        if (!once++) {
            fprintf(stderr, "ds4: [moe-init] gate_type=%u down_type=%u q2k=%d q4k=%d "
                            "gate_ebytes=%llu gate_rbytes=%llu down_ebytes=%llu down_rbytes=%llu "
                            "IN=%u MID=%u OUT=%u ntok=%u nexp=%u\n",
                    gate_type, down_type, q2k_path, q4k_path,
                    (unsigned long long)gate_expert_bytes, (unsigned long long)gate_row_bytes,
                    (unsigned long long)down_expert_bytes, (unsigned long long)down_row_bytes,
                    expert_in_dim, expert_mid_dim, out_dim, n_tokens, n_expert);
            fflush(stderr);
        }
    }
    if (!q4k_path && !q2k_path && (gate_type != 16u || down_type != 10u)) return 0;
    if (q4k_path && n_expert != 6u) return 0;   /* q4k batch(dspark drafter FFN): 朴素批链, sorted/tile(IQ2 专用)旁路 */
    const uint64_t gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_bytes > model_size - gate_offset ||
        gate_bytes > model_size - up_offset ||
        down_bytes > model_size - down_offset) {
        return 0;
    }
    const char *gate_w = cuda_model_range_ptr(model_map, gate_offset, gate_bytes, "moe_gate");
    const char *up_w = cuda_model_range_ptr(model_map, up_offset, gate_bytes, "moe_up");
    const char *down_w = cuda_model_range_ptr(model_map, down_offset, down_bytes, "moe_down");
    if (!gate_w || !up_w || !down_w) return 0;

    int ok = 1;
    const uint32_t xq_blocks = expert_in_dim / CUDA_QK_K;
    const uint32_t midq_blocks = expert_mid_dim / CUDA_QK_K;
    const uint64_t xq_count = (uint64_t)n_tokens * xq_blocks;
    const uint64_t midq_count = (uint64_t)n_tokens * n_expert * midq_blocks;
    const uint64_t xq_bytes = xq_count * sizeof(cuda_block_q8_K);
    const uint64_t midq_bytes = midq_count * sizeof(cuda_block_q8_K);
    if (down->bytes >= xq_bytes && gate->bytes >= midq_bytes) {
        cuda_block_q8_K *xq = (cuda_block_q8_K *)down->ptr;
        cuda_block_q8_K *midq = (cuda_block_q8_K *)gate->ptr;
        const uint32_t profile_moe = ((const char *)0) /* DS4_CUDA_MOE_PROFILE: 诊断开关已删(2026-08-22) */ != NULL;
        cudaEvent_t prof_ev[7] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL};
        if (profile_moe) {
            for (uint32_t i = 0; i < 7u; i++) {
                if (cudaEventCreate(&prof_ev[i]) != cudaSuccess) {
                    for (uint32_t j = 0; j < i; j++) (void)cudaEventDestroy(prof_ev[j]);
                    memset(prof_ev, 0, sizeof(prof_ev));
                    break;
                }
            }
            if (prof_ev[0]) (void)cudaEventRecord(prof_ev[0], 0);
        }
        const uint32_t pair_count = n_tokens * n_expert;
        const uint32_t use_sorted_pairs = n_tokens > 1u;
        const uint32_t use_expert_tiles = use_sorted_pairs && 1;
        const uint32_t expert_tile_m = ((const char *)0) /* DS4_CUDA_MOE_TILE4: 路径开关已删(2026-08-22 隐形炸弹清理) */ ? 4u : 8u;
        const uint32_t write_gate_up = 0;
        const uint32_t use_p2_sorted = use_sorted_pairs && 1;
        /* ★MoE down 原子累加整族删除(2026-08-22)★ —— 连同它的两个 env 开关一起删, 不留旋钮。
         * 原判据 n_tokens>=128 让**批量 prefill(chunk=256)走 atomicAdd**、decode(n=1)走独立
         * 平面+固定序归约。多专家原子累加同一输出行 ⇒ fp32 加序随调度漂 ⇒ 同一 prompt 两次
         * 跑结果不同。实测(同二进制同 ids 连跑两次, 批量路):
         *     L8  raw_ffn_in 中位相对差  6.91%   路由 top6 相同 71.5%
         *     L16                       16.56%                  40.7%
         *     L32                       16.44%                  46.9%
         * 即中层之后一半以上的 token 选中不同的专家; 而解码路两次跑全层 100% 逐位一致 ——
         * 一条判据同时解释了"prefill 不确定 / decode 确定"。
         * 后果: (1) 同 prompt 无唯一输出, 与"默认路径=裸模型真值"直接冲突;
         *       (2) 一切 prefill 跑分不可复现; (3) 用捕获做拟合的工作全部作废
         *       (反修整轮就是这么废的: 解算器拟合的 x 与引擎部署的 x 根本不是一条轨迹)。
         * 代价: prefill 149.03 → 122.67 t/s (−17.7%)。按"删除而非默认关闭"处理, 不留开关。 */
        const uint32_t use_atomic_down = 0u;
        const uint32_t use_gate_row2048 = use_expert_tiles && expert_tile_m == 8u &&
            (0 ||
             0 ||
             0 ||
             (n_tokens >= 128u &&
              1 &&
              1 &&
              1));
        const uint32_t use_down_tile16 = use_atomic_down && expert_tile_m == 8u &&
            n_tokens >= 128u && 1;
        /* decode LUT gate 是 IQ2_XXS 专用(用它的 256 项 grid 预表), Q2_K 没有 grid,
         * 必须绕开走 q2k 专用 kernel。 */
        const uint32_t use_decode_lut_gate =
            !q2k_path && n_tokens == 1u && xq_blocks <= 16u &&
            1;
        const uint32_t gate_row_span =
            0 ? 512u :
            0 ? 2048u : 1024u;
        const uint32_t down_row_span =
            0 ? 512u :
            0 ? 1024u : 2048u;
        const uint32_t use_down_row2048 = use_atomic_down && expert_tile_m == 8u &&
            (0 ||
             0 ||
             0 ||
             0 ||
             (use_down_tile16 &&
              1 &&
              1 &&
              1 &&
              1));
        const uint32_t use_direct_down_sum6 =
            n_tokens == 1u && n_expert == 6u &&
            1;
        uint32_t *sorted_pairs = NULL;
        uint32_t *sorted_offsets = NULL;
        uint32_t *sorted_counts = NULL;
        uint32_t *tile_total = NULL;
        uint32_t *tile_experts = NULL;
        uint32_t *tile_starts = NULL;
        uint32_t *tile16_total = NULL;
        uint32_t *tile16_experts = NULL;
        uint32_t *tile16_starts = NULL;
        uint32_t tile_capacity = 0;
        uint32_t tile16_capacity = 0;
        dim3 xq_grid(xq_blocks, n_tokens, 1);
        q8_K_quantize_kernel<<<xq_grid, 256>>>(xq, (const float *)x->ptr, expert_in_dim, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe x quantize launch");
        if (prof_ev[1]) (void)cudaEventRecord(prof_ev[1], 0);
        if (ok && use_sorted_pairs) {
            const uint64_t counts_bytes = 256ull * sizeof(uint32_t);
            const uint64_t offsets_bytes = 257ull * sizeof(uint32_t);
            const uint64_t cursors_bytes = 256ull * sizeof(uint32_t);
            const uint64_t sorted_bytes = (uint64_t)pair_count * sizeof(uint32_t);
            tile_capacity = (pair_count + expert_tile_m - 1u) / expert_tile_m + 256u;
            tile16_capacity = use_down_tile16 ? ((pair_count + 15u) / 16u + 256u) : 0u;
            const uint64_t tile_offsets_bytes = 257ull * sizeof(uint32_t);
            const uint64_t tile_total_bytes = sizeof(uint32_t);
            const uint64_t tile_experts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile_starts_bytes = (uint64_t)tile_capacity * sizeof(uint32_t);
            const uint64_t tile16_offsets_bytes = use_down_tile16 ? 257ull * sizeof(uint32_t) : 0u;
            const uint64_t tile16_total_bytes = use_down_tile16 ? sizeof(uint32_t) : 0u;
            const uint64_t tile16_experts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile16_starts_bytes = (uint64_t)tile16_capacity * sizeof(uint32_t);
            const uint64_t tile_offsets_off = counts_bytes + offsets_bytes + cursors_bytes + sorted_bytes;
            const uint64_t tile_total_off = tile_offsets_off + tile_offsets_bytes;
            const uint64_t tile_experts_off = tile_total_off + tile_total_bytes;
            const uint64_t tile_starts_off = tile_experts_off + tile_experts_bytes;
            const uint64_t tile16_offsets_off = tile_starts_off + tile_starts_bytes;
            const uint64_t tile16_total_off = tile16_offsets_off + tile16_offsets_bytes;
            const uint64_t tile16_experts_off = tile16_total_off + tile16_total_bytes;
            const uint64_t tile16_starts_off = tile16_experts_off + tile16_experts_bytes;
            const uint64_t scratch_bytes = tile16_starts_off + tile16_starts_bytes;
            uint8_t *scratch = (uint8_t *)cuda_tmp_alloc(scratch_bytes,
                                                         "routed_moe sorted pairs");
            if (!scratch) {
                ok = 0;
            } else {
                uint32_t *counts = (uint32_t *)scratch;
                uint32_t *offsets = (uint32_t *)(scratch + counts_bytes);
                uint32_t *cursors = (uint32_t *)(scratch + counts_bytes + offsets_bytes);
                sorted_pairs = (uint32_t *)(scratch + counts_bytes + offsets_bytes + cursors_bytes);
                sorted_offsets = offsets;
                sorted_counts = counts;
                uint32_t *tile_offsets = (uint32_t *)(scratch + tile_offsets_off);
                tile_total = (uint32_t *)(scratch + tile_total_off);
                tile_experts = (uint32_t *)(scratch + tile_experts_off);
                tile_starts = (uint32_t *)(scratch + tile_starts_off);
                uint32_t *tile16_offsets = use_down_tile16 ? (uint32_t *)(scratch + tile16_offsets_off) : NULL;
                tile16_total = use_down_tile16 ? (uint32_t *)(scratch + tile16_total_off) : NULL;
                tile16_experts = use_down_tile16 ? (uint32_t *)(scratch + tile16_experts_off) : NULL;
                tile16_starts = use_down_tile16 ? (uint32_t *)(scratch + tile16_starts_off) : NULL;
                ok = cuda_ok(cudaMemsetAsync(counts, 0, counts_bytes, g_cur_stream), "routed_moe sorted counts clear");
                if (ok) {
                    moe_count_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                        counts,
                        (const int32_t *)selected->ptr,
                        pair_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted count launch");
                }
                if (ok) {
                    moe_prefix_sorted_pairs_kernel<<<1, 1>>>(offsets, cursors, counts);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted prefix launch");
                }
                if (ok) {
                    moe_scatter_sorted_pairs_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                        sorted_pairs,
                        cursors,
                        (const int32_t *)selected->ptr,
                        pair_count);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe sorted scatter launch");
                }
                if (ok && use_expert_tiles) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile_offsets, tile_total, counts, expert_tile_m);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile offsets launch");
                }
                if (ok && use_expert_tiles) {
                    moe_build_expert_tiles_kernel<<<1, 256>>>(tile_experts, tile_starts, tile_offsets, counts, expert_tile_m);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tiles launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tile_offsets_kernel<<<1, 1>>>(tile16_offsets, tile16_total, counts, 16u);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 offsets launch");
                }
                if (ok && use_expert_tiles && use_down_tile16) {
                    moe_build_expert_tiles_kernel<<<1, 256>>>(tile16_experts, tile16_starts, tile16_offsets, counts, 16u);
                    ok = cuda_ok(cudaGetLastError(), "routed_moe expert tile16 launch");
                }
            }
        }
        if (prof_ev[2]) (void)cudaEventRecord(prof_ev[2], 0);
        if (ok) {
            dim3 mgrid((expert_mid_dim + 31u) / 32u, n_tokens * n_expert, 1);
            /* tiles 版 gate/up kernel 全部硬编码 cuda_block_iq2_xxs, Q2_K 走下面的
             * qwarp32 适配分支; tile 数据结构照常构建(down 侧还要用)。 */
            if (ok && !q2k_path && !q4k_path && sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts && tile_total && tile_experts && tile_starts) {
                if (use_gate_row2048) {
                    if (gate_row_span == 512u) {
                        dim3 tgrid((expert_mid_dim + 511u) / 512u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<512><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else if (gate_row_span == 1024u) {
                        dim3 tgrid((expert_mid_dim + 1023u) / 1024u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_rowspan_kernel<1024><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else {
                        dim3 tgrid((expert_mid_dim + 2047u) / 2048u, tile_capacity, 1);
                        moe_gate_up_mid_expert_tile8_row2048_kernel<<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    }
                } else if (expert_tile_m == 8u && xq_blocks == 16u && n_tokens <= 8u &&
                           1) {
                    /* verify/draft 小批: x16 结构 + expert-tile 权重驻留 */
                    const uint32_t xt_ycap = pair_count < 16u ? (pair_count > 0u ? pair_count : 1u) : 16u;
                    dim3 xtgrid((expert_mid_dim + 7u) / 8u, xt_ycap, 1);
                    moe_gate_up_mid_iq2_x16_tile_kernel<<<xtgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, expert_mid_dim, n_expert,
                        write_gate_up, clamp, 0 ? 1u : 0u);
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity < pair_count + 1u ? tile_capacity : pair_count + 1u, 1);
                    /* 小批(verify/draft)走无 smem 变体: sxq[8][16]=36.5KB 静态共享压死占用率
                     * (2026-08-21 gpu_span 归因: kernel 4.7x 偏离物理); 激活 28KB 天然驻 L2。
                     * 大批(prefill np≥8)保留 smem 变体。DS4_MOE_TILE_SMEM=1 强制旧路。 */
                    if (n_tokens > 8u || 0) {
                        moe_gate_up_mid_expert_tile8_row32_kernel<true><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    } else {
                        moe_gate_up_mid_expert_tile8_row32_kernel<false><<<tgrid, 256>>>(
                            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                            gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                            tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                            gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                            write_gate_up, clamp);
                    }
                } else {
                    dim3 tgrid((expert_mid_dim + 31u) / 32u, tile_capacity < pair_count + 1u ? tile_capacity : pair_count + 1u, 1);
                    moe_gate_up_mid_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
                        gate_w, up_w, xq, sorted_pairs, sorted_offsets, sorted_counts,
                        tile_total, tile_experts, tile_starts, (const float *)weights->ptr,
                        gate_expert_bytes, gate_row_bytes, xq_blocks, expert_mid_dim, n_expert,
                        write_gate_up, clamp);
                }
            } else if (ok && !q2k_path && !q4k_path && sorted_pairs && use_p2_sorted) {
                dim3 p2_mgrid((expert_mid_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_gate_up_mid_sorted_p2_qwarp32_kernel<<<p2_mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    pair_count,
                    clamp);
            } else if (ok && !q2k_path && !q4k_path && sorted_pairs) {
                moe_gate_up_mid_sorted_qwarp32_kernel<<<mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    gate_expert_bytes,
                    gate_row_bytes,
                    xq_blocks,
                    expert_mid_dim,
                    n_expert,
                    clamp);
            } else if (ok) {
                dim3 qgrid((expert_mid_dim + 127u) / 128u, n_tokens * n_expert, 1);
                if (q4k_path) {   /* decode 与 batch 同 kernel(pair=blockIdx.y 天然批) */
                    moe_gate_up_mid_decode_q4K_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else if (use_decode_lut_gate && xq_blocks == 16u) {
                    dim3 ixgrid(n_tokens, (expert_mid_dim + 7u) / 8u, n_expert);
                    moe_gate_up_mid_iq2_x16_kernel<<<ixgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else if (use_decode_lut_gate) {
                    moe_gate_up_mid_decode_lut_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        write_gate_up,
                        clamp);
                } else if (q2k_path && xq_blocks == 16u) {
                    dim3 q2grid(n_tokens, (expert_mid_dim + 7u) / 8u, n_expert);
                    moe_gate_up_mid_q2k_x16_kernel<<<q2grid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                    /* 数值探针(DS4_Q2K_PROBE=1, 只跑一次): kernel 的 gate[row0] vs
                     * 从 host mmap 直接读同一专家行做的 f32 参考 dot。差得远=kernel/
                     * 数据错位; 接近=gate/up 无辜, bug 在下游。 */
                    static int q2k_probed = 0;
                    if (!q2k_probed && ((const char *)0) /* DS4_Q2K_PROBE: 诊断开关已删(2026-08-22) */) {
                        q2k_probed = 1;
                        (void)cudaDeviceSynchronize();
                        int32_t e0 = -1;
                        float gk[4], xh[4096];
                        (void)cudaMemcpy(&e0, selected->ptr, 4, cudaMemcpyDeviceToHost);
                        (void)cudaMemcpy(gk, gate->ptr, sizeof(gk), cudaMemcpyDeviceToHost);
                        (void)cudaMemcpy(xh, x->ptr, sizeof(float) * expert_in_dim, cudaMemcpyDeviceToHost);
                        const uint8_t *row = (const uint8_t *)model_map + gate_offset
                                           + (uint64_t)e0 * gate_expert_bytes;
                        double ref = 0.0;
                        for (uint32_t b = 0; b < expert_in_dim / 256u; b++) {
                            const uint8_t *blk = row + (size_t)b * 84u;
                            const uint8_t *sc = blk, *qs = blk + 16;
                            uint16_t hd, hm; memcpy(&hd, blk + 80, 2); memcpy(&hm, blk + 82, 2);
                            const float d = dev_host_f16(hd), dm = dev_host_f16(hm);
                            for (int j = 0; j < 16; j++) {
                                const float dj = d * (sc[j] & 0xF), mj = dm * (sc[j] >> 4);
                                for (int ii = 0; ii < 16; ii++) {
                                    const int idx = j * 16 + ii;
                                    const int shift = (idx / 32) % 4 * 2;   /* 128 组内布局 */
                                    const int qpos = (idx / 128) * 32 + (idx % 32);
                                    const int q = (qs[qpos] >> ((idx % 128) / 32 * 2)) & 3;
                                    ref += (double)(dj * q - mj) * xh[b * 256u + idx];
                                }
                            }
                        }
                        fprintf(stderr, "ds4: [q2k-probe] e0=%d kernel_gate0=%.4f host_ref=%.4f x0..3=%.3f %.3f %.3f %.3f\n",
                                e0, gk[0], ref, xh[0], xh[1], xh[2], xh[3]);
                        fflush(stderr);
                        g_q2k_probe_out = 1;
                    }
                } else if (q2k_path) {
                    /* q2k 泛化配方(xq_blocks != 16): 原合体 kernel */
                    dim3 q2grid(n_tokens, (expert_mid_dim + 7u) / 8u, n_expert);
                    moe_gate_up_mid_q2k_qwarp32_kernel<<<q2grid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                } else {
                    moe_gate_up_mid_qwarp32_kernel<<<qgrid, 256>>>(
                        (float *)gate->ptr,
                        (float *)up->ptr,
                        (float *)mid->ptr,
                        gate_w,
                        up_w,
                        xq,
                        (const int32_t *)selected->ptr,
                        (const float *)weights->ptr,
                        gate_expert_bytes,
                        gate_row_bytes,
                        xq_blocks,
                        expert_mid_dim,
                        n_expert,
                        clamp);
                }
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
        }
        if (prof_ev[3]) (void)cudaEventRecord(prof_ev[3], 0);
        if (ok) {
            dim3 midq_grid(midq_blocks, n_tokens * n_expert, 1);
            q8_K_quantize_kernel<<<midq_grid, 256>>>(midq, (const float *)mid->ptr, expert_mid_dim, n_tokens * n_expert);
            ok = cuda_ok(cudaGetLastError(), "routed_moe mid quantize launch");
        }
        if (prof_ev[4]) (void)cudaEventRecord(prof_ev[4], 0);
        if (ok) {
            dim3 dgrid((out_dim + 31u) / 32u, n_tokens * n_expert, 1);
            uint32_t *down_tile_total = tile_total;
            uint32_t *down_tile_experts = tile_experts;
            uint32_t *down_tile_starts = tile_starts;
            uint32_t down_tile_capacity = tile_capacity;
            if (use_down_tile16 && tile16_total && tile16_experts && tile16_starts) {
                down_tile_total = tile16_total;
                down_tile_experts = tile16_experts;
                down_tile_starts = tile16_starts;
                down_tile_capacity = tile16_capacity;
            }
            {   /* partial 缓冲预分配: 必须发生在非 capture 调用(prefill)里, decode 的
                 * graph capture 期间 cudaMalloc 非法且录进 graph 的分支永久定型 */
                const uint64_t pneed0 = 6ull * out_dim * sizeof(float);
                if (pneed0 > g_down_partial_bytes) {
                    cudaStreamCaptureStatus cs0 = cudaStreamCaptureStatusNone;
                    (void)cudaStreamIsCapturing(0, &cs0);
                    if (cs0 == cudaStreamCaptureStatusNone) {
                        if (g_down_partial) (void)cudaFree(g_down_partial);
                        g_down_partial = NULL; g_down_partial_bytes = 0;
                        if (cudaMalloc(&g_down_partial, pneed0) == cudaSuccess) g_down_partial_bytes = pneed0;
                        else (void)cudaGetLastError();
                    }
                }
            }
            if (use_direct_down_sum6) {
                dim3 sgrid((out_dim + 31u) / 32u, 1, 1);
                if (q4k_path) {
                    moe_down_q4K_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                        (float *)out->ptr,
                        down_w,
                        midq,
                        (const int32_t *)selected->ptr,
                        down_expert_bytes,
                        down_row_bytes,
                        midq_blocks,
                        out_dim);
                } else {
                    const uint64_t pneed = 6ull * out_dim * sizeof(float);
                    if (g_down_partial && pneed <= g_down_partial_bytes) {
                        dim3 pgrid((out_dim + 31u) / 32u, 6, 1);
                        moe_down_partial_qwarp32_kernel<<<pgrid, 256>>>(
                            g_down_partial,
                            down_w,
                            midq,
                            (const int32_t *)selected->ptr,
                            down_expert_bytes,
                            down_row_bytes,
                            midq_blocks,
                            out_dim);
                        moe_down_partial_reduce6_kernel<<<(out_dim + 255u) / 256u, 256>>>(
                            (float *)out->ptr, g_down_partial, out_dim);
                    } else {
                        moe_down_sum6_qwarp32_kernel<<<sgrid, 256>>>(
                            (float *)out->ptr,
                            down_w,
                            midq,
                            (const int32_t *)selected->ptr,
                            down_expert_bytes,
                            down_row_bytes,
                            midq_blocks,
                            out_dim);
                    }
                }
            } else if (use_atomic_down) {
                uint64_t n = (uint64_t)n_tokens * out_dim;
                zero_kernel<<<(n + 255u) / 256u, 256>>>((float *)out->ptr, n);
                ok = cuda_ok(cudaGetLastError(), "routed_moe atomic zero launch");
            }
            if (use_direct_down_sum6) {
                /* The direct decode kernel writes the final token row. */
            } else if (!q4k_path && sorted_pairs && use_expert_tiles && sorted_offsets && sorted_counts &&
                down_tile_total && down_tile_experts && down_tile_starts) {
                if (use_down_row2048) {
                    if (down_row_span == 512u) {
                        dim3 tgrid((out_dim + 511u) / 512u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<512><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else if (down_row_span == 1024u) {
                        dim3 tgrid((out_dim + 1023u) / 1024u, down_tile_capacity, 1);
                        moe_down_expert_tile16_rowspan_kernel<1024><<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    } else {
                        dim3 tgrid((out_dim + 2047u) / 2048u, down_tile_capacity, 1);
                        moe_down_expert_tile16_row2048_kernel<<<tgrid, 256>>>(
                            use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                            down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                            down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                            midq_blocks, out_dim, n_expert, use_atomic_down);
                    }
                } else if (use_down_tile16) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity < pair_count + 1u ? down_tile_capacity : pair_count + 1u, 1);
                    moe_down_expert_tile16_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else if (expert_tile_m == 8u && midq_blocks == 8u && (out_dim & 31u) == 0u &&
                           n_tokens <= 8u && 1) {
                    dim3 dtg((out_dim + 31u) / 32u,
                             down_tile_capacity < pair_count + 1u ? down_tile_capacity : pair_count + 1u, 1);
                    moe_down_expert_tile_qwarp32_kernel<<<dtg, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts,
                        down_expert_bytes, down_row_bytes, midq_blocks, out_dim, n_expert, use_atomic_down);
                } else if (expert_tile_m == 8u) {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity < pair_count + 1u ? down_tile_capacity : pair_count + 1u, 1);
                    moe_down_expert_tile8_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                } else {
                    dim3 tgrid((out_dim + 31u) / 32u, down_tile_capacity < pair_count + 1u ? down_tile_capacity : pair_count + 1u, 1);
                    moe_down_expert_tile4_row32_kernel<<<tgrid, 256>>>(
                        use_atomic_down ? (float *)out->ptr : (float *)down->ptr,
                        down_w, midq, sorted_pairs, sorted_offsets, sorted_counts,
                        down_tile_total, down_tile_experts, down_tile_starts, down_expert_bytes, down_row_bytes,
                        midq_blocks, out_dim, n_expert, use_atomic_down);
                }
            } else if (!q4k_path && sorted_pairs && use_p2_sorted) {
                dim3 p2_dgrid((out_dim + 15u) / 16u, (pair_count + 1u) / 2u, 1);
                moe_down_sorted_p2_qwarp32_kernel<<<p2_dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert,
                    pair_count);
            } else if (!q4k_path && sorted_pairs) {
                moe_down_sorted_qwarp32_kernel<<<dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    sorted_pairs,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            } else if (q4k_path) {
                if (((const char *)0) /* DS4_Q4K_BATCH_PROBE: 诊断开关已删(2026-08-22) */) fprintf(stderr, "ds4: [q4kbatch] down pairs kernel launch dgrid=(%u,%u)\n", dgrid.x, dgrid.y);
                moe_down_q4K_pairs_qwarp32_kernel<<<dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            } else {
                moe_down_qwarp32_kernel<<<dgrid, 256>>>(
                    (float *)down->ptr,
                    down_w,
                    midq,
                    (const int32_t *)selected->ptr,
                    down_expert_bytes,
                    down_row_bytes,
                    midq_blocks,
                    out_dim,
                    n_expert);
            }
            ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
        }
        if (ok && q4k_path && n_tokens > 1u && ((const char *)0) /* DS4_Q4K_BATCH_PROBE: 诊断开关已删(2026-08-22) */) {
            (void)cudaDeviceSynchronize();
            float xv[4] = {0}, mv[4] = {0}, dv[4] = {0};
            (void)cudaMemcpy(xv, x->ptr, sizeof(xv), cudaMemcpyDeviceToHost);
            (void)cudaMemcpy(mv, mid->ptr, sizeof(mv), cudaMemcpyDeviceToHost);
            (void)cudaMemcpy(dv, down->ptr, sizeof(dv), cudaMemcpyDeviceToHost);
            int32_t sh[8] = {0}; float wh[8] = {0};
            (void)cudaMemcpy(sh, selected->ptr, sizeof(sh), cudaMemcpyDeviceToHost);
            (void)cudaMemcpy(wh, weights->ptr, sizeof(wh), cudaMemcpyDeviceToHost);
            fprintf(stderr, "ds4: [q4kbatch] ntok=%u x=%.3g %.3g mid=%.3g %.3g down=%.3g %.3g sel=%d %d %d w=%.3g %.3g\n",
                    n_tokens, xv[0], xv[1], mv[0], mv[1], dv[0], dv[1], sh[0], sh[1], sh[2], wh[0], wh[1]);
            {
                cuda_block_q8_K mb;
                (void)cudaMemcpy(&mb, midq, sizeof(mb), cudaMemcpyDeviceToHost);
                int bs0 = mb.bsums[0], bs1 = mb.bsums[1];
                fprintf(stderr, "ds4: [q4kbatch] midq0 d=%.4g qs=%d %d %d %d bsums=%d %d\n",
                        mb.d, (int)mb.qs[0], (int)mb.qs[1], (int)mb.qs[2], (int)mb.qs[3], bs0, bs1);
                cuda_block_q4_K wb;
                (void)cudaMemcpy(&wb, down_w + (uint64_t)(uint32_t)sh[0] * down_expert_bytes, sizeof(wb), cudaMemcpyDeviceToHost);
                fprintf(stderr, "ds4: [q4kbatch] downblk e%d d=%.4g dmin=%.4g sc=%u %u qs=%u %u | ebytes=%llu rbytes=%llu odim=%u mblk=%u\n",
                        sh[0], dev_host_f16(wb.d), dev_host_f16(wb.dmin), (unsigned)wb.scales[0], (unsigned)wb.scales[1],
                        (unsigned)wb.qs[0], (unsigned)wb.qs[1],
                        (unsigned long long)down_expert_bytes, (unsigned long long)down_row_bytes, out_dim, midq_blocks);
            }
        }
        if (prof_ev[5]) (void)cudaEventRecord(prof_ev[5], 0);
        if (ok && !use_atomic_down && !use_direct_down_sum6) {
            uint64_t n = (uint64_t)n_tokens * out_dim;
            moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
            ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
        }
        if (prof_ev[6]) {
            (void)cudaEventRecord(prof_ev[6], 0);
            if (cudaEventSynchronize(prof_ev[6]) == cudaSuccess) {
                float ms_xq = 0.0f, ms_sort = 0.0f, ms_gate = 0.0f, ms_midq = 0.0f, ms_down = 0.0f, ms_sum = 0.0f, ms_total = 0.0f;
                (void)cudaEventElapsedTime(&ms_xq, prof_ev[0], prof_ev[1]);
                (void)cudaEventElapsedTime(&ms_sort, prof_ev[1], prof_ev[2]);
                (void)cudaEventElapsedTime(&ms_gate, prof_ev[2], prof_ev[3]);
                (void)cudaEventElapsedTime(&ms_midq, prof_ev[3], prof_ev[4]);
                (void)cudaEventElapsedTime(&ms_down, prof_ev[4], prof_ev[5]);
                (void)cudaEventElapsedTime(&ms_sum, prof_ev[5], prof_ev[6]);
                (void)cudaEventElapsedTime(&ms_total, prof_ev[0], prof_ev[6]);
                fprintf(stderr,
                        "ds4: CUDA MoE profile tokens=%u pairs=%u xq=%.3f sort=%.3f gateup=%.3f midq=%.3f down=%.3f sum=%.3f total=%.3f ms\n",
                        n_tokens, pair_count, ms_xq, ms_sort, ms_gate, ms_midq, ms_down, ms_sum, ms_total);
            }
            for (uint32_t i = 0; i < 7u; i++) (void)cudaEventDestroy(prof_ev[i]);
        }
    if (ok && g_q2k_probe_out >= 1 && g_q2k_probe_out < 90) {
        g_q2k_probe_out++;
        (void)cudaDeviceSynchronize();
        /* host 全链参考: tok0 的 6 个专家, gate/up→clamp→silu→×router→down, 与 out[0..3] 比 */
        float xh[8192]; int32_t selh[64]; float wh[64]; float oh[4];
        (void)cudaMemcpy(xh, x->ptr, sizeof(float) * expert_in_dim, cudaMemcpyDeviceToHost);
        (void)cudaMemcpy(selh, selected->ptr, sizeof(int32_t) * n_expert, cudaMemcpyDeviceToHost);
        (void)cudaMemcpy(wh, weights->ptr, sizeof(float) * n_expert, cudaMemcpyDeviceToHost);
        (void)cudaMemcpy(oh, out->ptr, sizeof(oh), cudaMemcpyDeviceToHost);
        double oref[4] = {0, 0, 0, 0};
        float deq[256];
        float *mid_h = (float *)malloc(sizeof(float) * expert_mid_dim);
        for (uint32_t s = 0; s < n_expert; s++) {
            const int e = selh[s];
            if (e < 0) continue;
            const uint8_t *gbase = (const uint8_t *)model_map + gate_offset + (uint64_t)e * gate_expert_bytes;
            const uint8_t *ubase = (const uint8_t *)model_map + up_offset + (uint64_t)e * gate_expert_bytes;
            const uint8_t *dbase = (const uint8_t *)model_map + down_offset + (uint64_t)e * down_expert_bytes;
            for (uint32_t r = 0; r < expert_mid_dim; r++) {
                double gacc = 0, uacc = 0;
                for (uint32_t b = 0; b < expert_in_dim / 256u; b++) {
                    host_deq_q2k_block(gbase + (size_t)r * gate_row_bytes + (size_t)b * 84u, deq);
                    for (int i = 0; i < 256; i++) gacc += (double)deq[i] * xh[b * 256u + i];
                    host_deq_q2k_block(ubase + (size_t)r * gate_row_bytes + (size_t)b * 84u, deq);
                    for (int i = 0; i < 256; i++) uacc += (double)deq[i] * xh[b * 256u + i];
                }
                float gv = (float)gacc, uv = (float)uacc;
                if (clamp > 1.0e-6f) {
                    if (gv > clamp) gv = clamp;
                    if (uv > clamp) uv = clamp;
                    if (uv < -clamp) uv = -clamp;
                }
                mid_h[r] = (gv / (1.0f + expf(-gv))) * uv * wh[s];
            }
            for (int o = 0; o < 4; o++) {
                double acc = 0;
                for (uint32_t b = 0; b < expert_mid_dim / 256u; b++) {
                    host_deq_q2k_block(dbase + (size_t)o * down_row_bytes + (size_t)b * 84u, deq);
                    for (int i = 0; i < 256; i++) acc += (double)deq[i] * mid_h[b * 256u + i];
                }
                oref[o] += acc;
            }
        }
        free(mid_h);
        {
            /* 全量扫 MoE 输出: NaN 到底是不是 MoE 自己产的 */
            const uint64_t on = (uint64_t)n_tokens * out_dim;
            float *ofull = (float *)malloc(on * sizeof(float));
            int o_nan = 0; float o_max = 0; int64_t first_nan = -1;
            if (ofull && cudaMemcpy(ofull, out->ptr, on * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess) {
                for (uint64_t i = 0; i < on; i++) {
                    if (ofull[i] != ofull[i]) { if (first_nan < 0) first_nan = (int64_t)i; o_nan++; }
                    else if (fabsf(ofull[i]) > o_max) o_max = fabsf(ofull[i]);
                }
            }
            free(ofull);
            /* mid 扫描 + 全部 selected: 区分"mid 就坏"vs"down/sum 读错" */
            {
                const uint64_t mn = (uint64_t)n_tokens * n_expert * expert_mid_dim;
                float *mfull = (float *)malloc(mn * sizeof(float));
                int m_nan = 0; float m_max = 0;
                if (mfull && cudaMemcpy(mfull, mid->ptr, mn * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess)
                    for (uint64_t i = 0; i < mn; i++) {
                        if (mfull[i] != mfull[i]) m_nan++;
                        else if (fabsf(mfull[i]) > m_max) m_max = fabsf(mfull[i]);
                    }
                free(mfull);
                int32_t sall[64];
                (void)cudaMemcpy(sall, selected->ptr, sizeof(int32_t) * n_tokens * n_expert, cudaMemcpyDeviceToHost);
                int s_neg = 0;
                for (uint32_t i = 0; i < n_tokens * n_expert; i++) if (sall[i] < 0) s_neg++;
                fprintf(stderr, "ds4: [q2k-m] mid_nan=%d mid_max=%.1f sel_neg=%d sel=[", m_nan, m_max, s_neg);
                for (uint32_t i = 0; i < n_tokens * n_expert && i < 30u; i++) fprintf(stderr, "%d ", sall[i]);
                fprintf(stderr, "]\n");
            }
            fprintf(stderr, "ds4: [q2k-o] out_nan=%d/%llu first=%lld out_max=%.1f\n",
                    o_nan, (unsigned long long)on, (long long)first_nan, o_max);
            int x_nan = 0; float x_max = 0;
            for (uint32_t i = 0; i < expert_in_dim; i++) {
                if (xh[i] != xh[i]) x_nan++;
                else if (fabsf(xh[i]) > x_max) x_max = fabsf(xh[i]);
            }
            fprintf(stderr, "ds4: [q2k-x] x_nan=%d x_max=%.1f sel0=%d w0=%.4f\n", x_nan, x_max, selh[0], wh[0]);
            const double diff = fabs(oh[0] - oref[0]) + fabs(oh[1] - oref[1]);
            fprintf(stderr, "ds4: [q2k#%02d] ntok=%u gpu0=%.3f host0=%.3f %s\n",
                    g_q2k_probe_out - 1, n_tokens, oh[0], oref[0],
                    diff > 0.5 * (fabs(oref[0]) + fabs(oref[1]) + 0.1) ? "★发散★" : "ok");
            fflush(stderr);
        }
    }
        return ok;
    }

    if (ok) {
        dim3 mgrid(expert_mid_dim, n_tokens * n_expert, 1);
        moe_gate_up_mid_f32_kernel<<<mgrid, 256>>>(
            (float *)gate->ptr,
            (float *)up->ptr,
            (float *)mid->ptr,
            gate_w,
            up_w,
            (const float *)x->ptr,
            (const int32_t *)selected->ptr,
            (const float *)weights->ptr,
            gate_expert_bytes,
            gate_row_bytes,
            expert_in_dim,
            expert_mid_dim,
            n_expert,
            clamp);
        ok = cuda_ok(cudaGetLastError(), "routed_moe gate/up launch");
    }
    if (ok) {
        dim3 dgrid(out_dim, n_tokens * n_expert, 1);
        moe_down_f32_kernel<<<dgrid, 256>>>(
            (float *)down->ptr,
            down_w,
            (const float *)mid->ptr,
            (const int32_t *)selected->ptr,
            down_expert_bytes,
            down_row_bytes,
            expert_mid_dim,
            out_dim,
            n_expert);
        ok = cuda_ok(cudaGetLastError(), "routed_moe down launch");
    }
    if (ok) {
        uint64_t n = (uint64_t)n_tokens * out_dim;
        moe_sum_kernel<<<(n + 255) / 256, 256>>>((float *)out->ptr, (const float *)down->ptr, out_dim, n_expert, n_tokens);
        ok = cuda_ok(cudaGetLastError(), "routed_moe sum launch");
    }
    return ok;
}

/* ================= v2.2 VQ blob 专家前向 (DQVL) =================
 * 背景: 合一 VQ GGUF 把 routed 专家字节全塞进 blk.L.ffn_exps_vq.blob, base
 * gate/up 张量不在文件里(offset=0/bytes=0)。Metal 侧(ds4_metal.m:20630-21830)
 * 早有完整实现, CUDA 侧此前一行都没有 —— routed_moe_launch 开头的类型闸
 * (gate_type!=16||down_type!=10) 直接 return 0, 表现为 "cuda prefill failed"。
 *
 * 这里按 Metal 的同一语义落地, 不发明新算法:
 *   ① CPU 多线程把活跃专家 dequant 成 f16 scratch。scratch 用 cudaMallocManaged:
 *      GB10 是 Grace+Blackwell 统一内存, CPU 写 GPU 读与 Metal 的
 *      MTLResourceStorageModeShared 同构, 无需显式 H2D 拷贝。
 *   ② selected 里是原始 expert id, 但 gather 后权重按 active 紧凑排布 ⇒ 必须
 *      remap 成 slot(Metal 侧在别处已 remap, CUDA 侧这里自己做)。
 *   ③ GPU kernel 走 f16 mm_id: gate/up 融合 SwiGLU → down 加权累加。
 *   ④ DS4_VQ_GPU=0 回 CPU 参考路(与 Metal 同名开关同语义), 供数值对齐。
 * dequant 复用 vq_fmt.h 的 ds4vq_dequant_f16 —— 三端同一份解码。 */

static void *g_vq_gate_sc = NULL, *g_vq_up_sc = NULL, *g_vq_down_sc = NULL;
static uint64_t g_vq_gu_bytes = 0, g_vq_dn_bytes = 0;
static int32_t *g_vq_sel_dev = NULL;   /* remap 后的 slot 索引(设备端) */
static uint64_t g_vq_sel_bytes = 0;

/* ---- decode 专家 dequant 缓存 ----
 * 同一个专家的 dequant 结果与 token 无关(权重不变), 而 decode 每层只用 n_expert 个
 * 专家、相邻 token 的路由高度重叠 —— 每次 forward 全量重算是纯浪费, 且 dequant 占
 * 了 92.8% 的 GPU 时间。这里给每层留 K 个常驻槽, 命中就直接复用槽里的 f16 权重。
 * 只对 decode(n_tokens==1) 启用: prefill 的 n_active 远大于 K, 会把槽冲干净, 那时
 * 走直通并把缓存置空(scratch 被覆盖, 槽内容不再有效)。 */
/* ★缓存必须是 per-layer 的存储, 不能只有 per-layer 的标记★
 * 第一版把 43 层共用一份 scratch, L1 立刻覆盖 L0 刚写的槽, 下个 token 回到 L0 时
 * 标记说"命中"、槽里装的却是 L42 的权重 ⇒ 输出退化成复读。所以缓存槽的显存按层独立
 * 分配(K×48MB×43 ≈ 16.5GB), prefill 的直通路另走一份共享临时 buffer。 */
#define DS4_VQ_CACHE_SLOTS 8u
#define DS4_VQ_CACHE_LAYERS 64u
static struct {
    int32_t  expert[DS4_VQ_CACHE_SLOTS];   /* 槽内当前专家 id, -1=空 */
    uint64_t used[DS4_VQ_CACHE_SLOTS];     /* LRU 时钟 */
    void    *gate, *up, *down;             /* 本层专属显存(K 槽), NULL=未分配 */
} g_vq_cache[DS4_VQ_CACHE_LAYERS];
static uint64_t g_vq_clock = 0;
static int g_vq_cache_ready = 0;
static uint64_t g_vq_hit = 0, g_vq_miss = 0;

static void cuda_vq_cache_reset_all(void) {
    for (uint32_t l = 0; l < DS4_VQ_CACHE_LAYERS; l++) {
        for (uint32_t s = 0; s < DS4_VQ_CACHE_SLOTS; s++) {
            g_vq_cache[l].expert[s] = -1;
            g_vq_cache[l].used[s] = 0;
        }
        g_vq_cache[l].gate = g_vq_cache[l].up = g_vq_cache[l].down = NULL;
    }
    g_vq_cache_ready = 1;
}

/* 本层缓存显存(K 槽)按需分配; 分配不到就退回直通(不缓存), 不让它变成硬失败。 */
static int cuda_vq_cache_ensure_mem(uint32_t layer, uint64_t ge, uint64_t de) {
    if (g_vq_cache[layer].gate) return 1;
    void *g = NULL, *u = NULL, *d = NULL;
    const uint64_t gu = (uint64_t)DS4_VQ_CACHE_SLOTS * ge, dn = (uint64_t)DS4_VQ_CACHE_SLOTS * de;
    if (cudaMalloc(&g, gu) != cudaSuccess || cudaMalloc(&u, gu) != cudaSuccess ||
        cudaMalloc(&d, dn) != cudaSuccess) {
        (void)cudaGetLastError();
        if (g) (void)cudaFree(g);
        if (u) (void)cudaFree(u);
        if (d) (void)cudaFree(d);
        return 0;
    }
    g_vq_cache[layer].gate = g; g_vq_cache[layer].up = u; g_vq_cache[layer].down = d;
    return 1;
}

typedef struct {
    const void *model_map; const uint8_t *blob; const uint32_t *active_ids;
    uint16_t *gbase, *ubase, *dbase;
    uint64_t down_offset, down_expert_bytes;
    uint32_t in, mid, out_dim, lo, hi;
    volatile int *err;
} cuda_vq_gather_task;

/* 逐专家 dequant, 各线程写不相交 scratch 段 ⇒ 无锁。与 Metal 的
 * ds4_vq_gather_worker 逐行同义(含冷 w2 从 base go1b 34B/256el 展开 ±d)。 */
static void *cuda_vq_gather_worker(void *arg) {
    cuda_vq_gather_task *t = (cuda_vq_gather_task *)arg;
    for (uint32_t i = t->lo; i < t->hi && !*t->err; i++) {
        const uint32_t e = t->active_ids[i];
        uint16_t *dg = t->gbase + (uint64_t)i * t->mid * t->in;
        uint16_t *du = t->ubase + (uint64_t)i * t->mid * t->in;
        uint16_t *dd = t->dbase + (uint64_t)i * t->out_dim * t->mid;
        uint64_t o1 = ds4vq_slot(t->blob, (int)e, 0);
        uint64_t o3 = ds4vq_slot(t->blob, (int)e, 1);
        uint64_t o2 = ds4vq_slot(t->blob, (int)e, 2);
        int rc1 = (!o1) ? -9 : ds4vq_dequant_f16(t->blob + o1, dg, (int)t->mid, (int)t->in);
        int rc3 = (rc1 == 0 && o3) ? ds4vq_dequant_f16(t->blob + o3, du, (int)t->mid, (int)t->in) : (!o3 ? -9 : 0);
        if (rc1 != 0 || rc3 != 0) {
            fprintf(stderr, "ds4: [cuda-vq-gather-err] e=%u o1=%llu o3=%llu rc1=%d rc3=%d mid=%u in=%u\n",
                    e, (unsigned long long)o1, (unsigned long long)o3, rc1, rc3, t->mid, t->in);
            *t->err = 1; return NULL;
        }
        if (o2) {
            int rc2 = ds4vq_dequant_f16(t->blob + o2, dd, (int)t->out_dim, (int)t->mid);
            if (rc2 != 0) {
                fprintf(stderr, "ds4: [cuda-vq-gather-err] e=%u o2=%llu rc2=%d out=%u mid=%u\n",
                        e, (unsigned long long)o2, rc2, t->out_dim, t->mid);
                *t->err = 1; return NULL;
            }
        } else {
            /* 冷 w2: blob 的 which=2 槽缺席 ⇒ 从 base go1b 字节展开。base down 是
             * 影子张量(bytes=0)时硬失败, 不读 offset 0 的垃圾当权重。 */
            if (t->down_expert_bytes == 0 || t->down_offset == 0) {
                fprintf(stderr, "ds4: [cuda-vq-gather-err] e=%u 冷 w2 槽缺失且 base down 不在文件里"
                                "(影子张量) -- aborting (no silent quality downgrade)\n", e);
                *t->err = 1; return NULL;
            }
            const uint8_t *sd = (const uint8_t *)t->model_map + t->down_offset + (uint64_t)e * t->down_expert_bytes;
            const uint64_t nblk_row = t->mid / 256u;
            for (uint32_t r = 0; r < t->out_dim; r++) {
                const uint8_t *rb = sd + (uint64_t)r * nblk_row * 34u;
                uint16_t *orow = dd + (uint64_t)r * t->mid;
                for (uint64_t b = 0; b < nblk_row; b++) {
                    uint16_t dsc; memcpy(&dsc, rb + b * 34u, 2);
                    const uint8_t *sg = rb + b * 34u + 2;
                    uint16_t *o = orow + b * 256u;
                    for (int k = 0; k < 256; k++)
                        o[k] = (sg[k >> 3] >> (k & 7)) & 1 ? dsc : (uint16_t)(dsc ^ 0x8000u);
                }
            }
        }
    }
    return NULL;
}

/* GPU 版 VQ 载荷解码。CPU 侧逐元素查表是 decode 0.6 t/s 的主因(每 token 约 6.5G
 * 元素), 而这活是纯并行查表: 值 = 码本[idx][d] * g_r[row]。blob 已随模型 mmap 被
 * cudaHostRegister, kernel 可经 UVA 直读 —— 且读的是压缩态(约 3MB/专家)而不是
 * dequant 后的 f16(16MB/专家), 字节数还少 5 倍。
 * 位流解析与 vq_fmt.h 的 ds4vq_dequant_f16 逐字同义(nbit 由 nc 推导, 9/10bit 走
 * 三字节窗口, 8bit 走字节流)。 */
__global__ static void vq_dequant_kernel(
        __half *out, const uint8_t *pay,
        uint32_t rows, uint32_t cols, uint32_t dim, uint32_t nc, uint32_t nbit) {
    /* 每 block 一行。两次优化尝试均被实测否决, 记在这里免得再走一遍:
     *   ① 码本搬 shared(grid=rows): 2048 个 block 各搬一次 4KB, 净变慢;
     *   ② 多行/block + shared 摊薄加载: 速度未验先崩 —— 输出塌成全 BOS。
     * 码本只有几 KB, 本来就常驻 L1/L2 被所有 block 共享, 全局直读是当前正确且不慢的解。 */
    const uint8_t *cb = pay + 16;
    const uint8_t *gr = cb + (size_t)nc * dim * 2;
    const uint8_t *ix = gr + (size_t)rows * 2;
    const uint32_t imsk = (nbit >= 32) ? 0xFFFFFFFFu : ((1u << nbit) - 1u);
    const uint32_t nidx_row = cols / dim;
    const uint32_t r = blockIdx.x;
    if (r >= rows) return;
    __half gh; memcpy(&gh, gr + (size_t)r * 2, 2);
    const float g = __half2float(gh);
    const size_t i0 = (size_t)r * nidx_row;
    __half *orow = out + (size_t)r * cols;
    for (uint32_t i = threadIdx.x; i < nidx_row; i += blockDim.x) {
        const size_t gi = i0 + i;
        uint32_t v;
        if (nbit == 8) {
            v = ix[gi];
        } else {
            /* 一次 32 位读取代三次单字节 load: nbit<=24 时目标位段必定落在这个窗口内,
             * 取值与 vq_fmt.h 的三字节拼法逐位相同。原写法每 warp 发 96 条字节 load 去
             * 覆盖仅 36 字节的地址范围, 而 GPU 最小访存事务是 32 字节 —— 几十倍放大,
             * 正是 dequant 只跑到 6.7GB/s 的原因。
             * 尾部越读的至多 3 字节仍在 blob/相邻张量内(只取低位, 不影响取值)。 */
            const size_t bit = gi * nbit;
            const size_t by = bit >> 3;
            uint32_t w; memcpy(&w, ix + by, 4);
            v = (w >> (bit & 7)) & imsk;
        }
        const uint8_t *c = cb + (size_t)v * dim * 2;
        __half *o = orow + (size_t)i * dim;
        for (uint32_t d = 0; d < dim; d++) {
            __half ch; memcpy(&ch, c + (size_t)d * 2, 2);
            o[d] = __float2half(__half2float(ch) * g);
        }
    }
}

/* ---- decode 全 GPU 路: 消除每层的 D2H 往返 ----
 * 实测 GPU 利用率只有 6%(P0/2405MHz/13W): 瓶颈不是算力也不是带宽, 而是每层都要把
 * selected 同步拷回主机做去重+remap —— 43 层就是每 token 43 次"等 GPU 全停→CPU 算
 * →再启动"。这里让 kernel 自己读 blob 的偏移表, 按 (token,pick) 对直接解码到各自的
 * scratch 段, host 侧一个字节都不用读回来。
 * 代价: 同一专家被两个 pick 选中时会解码两次。decode 只有 n_expert 个 pick, 重复概率
 * 低且每次只多一份工作; 换掉的是整整一次全设备同步, 净赚。prefill(pair 数上千)重复
 * 会爆炸, 所以那条路仍走 host 去重。 */
__device__ __forceinline__ static uint64_t vq_slot_dev(const uint8_t *blob, int e, int which) {
    uint64_t off; memcpy(&off, blob + 16 + ((size_t)e * 3 + which) * 8, 8);
    return off;
}

__global__ static void vq_dequant_pairs_kernel(
        __half *gout, __half *uout, __half *dout,
        const uint8_t *blob, const int32_t *sel,
        const uint8_t *down_base, uint64_t down_ebytes,
        uint32_t IN, uint32_t MID, uint32_t OUT) {
    const uint32_t pair = blockIdx.z, which = blockIdx.y;
    const int32_t e = sel[pair];
    if (e < 0) return;

    const uint32_t exp_rows = (which == 2u) ? OUT : MID;
    const uint32_t exp_cols = (which == 2u) ? MID : IN;
    __half *out = (which == 0u) ? (gout + (uint64_t)pair * MID * IN)
                : (which == 1u) ? (uout + (uint64_t)pair * MID * IN)
                                : (dout + (uint64_t)pair * OUT * MID);
    const uint64_t off = vq_slot_dev(blob, e, (int)which);
    const uint32_t r = blockIdx.x;
    if (r >= exp_rows) return;

    if (off == 0u) {
        /* 冷 w2: base go1b 34B/256el sign 字节 → ±d。base down 缺席时写 0 而不是读垃圾
         * (host 侧已在直通路对影子张量硬失败; 这里保守置零, 不静默用错误权重)。 */
        if (which != 2u || down_base == NULL || down_ebytes == 0u) return;
        const uint32_t nblk_row = MID / 256u;
        const uint8_t *rb = down_base + (uint64_t)e * down_ebytes + (size_t)r * nblk_row * 34u;
        uint16_t *orow = (uint16_t *)(out + (size_t)r * MID);
        for (uint32_t b = threadIdx.x; b < nblk_row; b += blockDim.x) {
            uint16_t dsc; memcpy(&dsc, rb + (size_t)b * 34u, 2);
            const uint8_t *sg = rb + (size_t)b * 34u + 2;
            uint16_t *o = orow + (size_t)b * 256u;
            for (int k = 0; k < 256; k++)
                o[k] = ((sg[k >> 3] >> (k & 7)) & 1) ? dsc : (uint16_t)(dsc ^ 0x8000u);
        }
        return;
    }

    const uint8_t *pay = blob + off;
    uint32_t mg; memcpy(&mg, pay, 4);
    if (mg != DS4VQ_MAT_MAGIC) return;
    uint16_t d16, n16; memcpy(&d16, pay + 4, 2); memcpy(&n16, pay + 6, 2);
    uint32_t rows, cols; memcpy(&rows, pay + 8, 4); memcpy(&cols, pay + 12, 4);
    if (rows != exp_rows || cols != exp_cols) return;
    uint32_t nbit = 0; while ((1u << nbit) < (uint32_t)n16) nbit++; if (nbit < 1u) nbit = 1u;

    const uint8_t *cb = pay + 16;
    const uint8_t *gr = cb + (size_t)n16 * d16 * 2;
    const uint8_t *ix = gr + (size_t)rows * 2;
    const uint32_t imsk = (nbit >= 32u) ? 0xFFFFFFFFu : ((1u << nbit) - 1u);
    const uint32_t nidx_row = cols / d16;
    __half gh; memcpy(&gh, gr + (size_t)r * 2, 2);
    const float g = __half2float(gh);
    const size_t i0 = (size_t)r * nidx_row;
    __half *orow = out + (size_t)r * cols;
    for (uint32_t i = threadIdx.x; i < nidx_row; i += blockDim.x) {
        const size_t gi = i0 + i;
        uint32_t v;
        if (nbit == 8u) {
            v = ix[gi];
        } else {
            const size_t bit = gi * nbit, by = bit >> 3;
            uint32_t w; memcpy(&w, ix + by, 4);
            v = (w >> (bit & 7)) & imsk;
        }
        const uint8_t *c = cb + (size_t)v * d16 * 2;
        __half *o = orow + (size_t)i * d16;
        for (uint32_t d = 0; d < d16; d++) {
            __half ch; memcpy(&ch, c + (size_t)d * 2, 2);
            o[d] = __float2half(__half2float(ch) * g);
        }
    }
}

/* 冷 w2(blob 无 which=2 槽): base go1b 34B/256el sign 字节 → ±d 展开。GPU 版。 */
__global__ static void vq_cold_w2_kernel(
        __half *out, const uint8_t *sd, uint32_t out_dim, uint32_t mid) {
    const uint32_t nblk_row = mid / 256u;
    const uint32_t r = blockIdx.x;
    if (r >= out_dim) return;
    const uint8_t *rb = sd + (size_t)r * nblk_row * 34u;
    __half *orow = out + (size_t)r * mid;
    for (uint32_t b = threadIdx.x; b < nblk_row; b += blockDim.x) {
        uint16_t dsc; memcpy(&dsc, rb + (size_t)b * 34u, 2);
        const uint8_t *sg = rb + (size_t)b * 34u + 2;
        uint16_t *o = (uint16_t *)(orow + (size_t)b * 256u);
        for (int k = 0; k < 256; k++)
            o[k] = ((sg[k >> 3] >> (k & 7)) & 1) ? dsc : (uint16_t)(dsc ^ 0x8000u);
    }
}

/* 载荷头解析(host 侧, 与 vq_fmt.h 同布局): 0=成功 */
static int cuda_vq_pay_hdr(const uint8_t *pay, int exp_rows, int exp_cols,
                           uint32_t *rows, uint32_t *cols, uint32_t *dim, uint32_t *nc, uint32_t *nbit) {
    uint32_t mg; memcpy(&mg, pay, 4);
    if (mg != DS4VQ_MAT_MAGIC) return -1;
    uint16_t d16, n16; memcpy(&d16, pay + 4, 2); memcpy(&n16, pay + 6, 2);
    uint32_t r32, c32; memcpy(&r32, pay + 8, 4); memcpy(&c32, pay + 12, 4);
    if ((int)r32 != exp_rows || (int)c32 != exp_cols) return -2;
    int nb = 0; while ((1 << nb) < (int)n16) nb++; if (nb < 1) nb = 1;
    /* dequant kernel 把码本整块放 shared(48KB/block 上限)。当前配方 nc=512×dim=4=4KB,
     * 余量充足; 真超了就明着失败, 不静默 launch 出错。 */
    if ((size_t)n16 * d16 * 2u > 48u * 1024u) {
        fprintf(stderr, "ds4: [cuda-vq] 码本 %u×%u=%zuB 超 shared 上限 48KB\n",
                (unsigned)n16, (unsigned)d16, (size_t)n16 * d16 * 2u);
        return -3;
    }
    *rows = r32; *cols = c32; *dim = d16; *nc = n16; *nbit = (uint32_t)nb;
    return 0;
}

/* scratch 用纯设备内存(cudaMalloc)而不是托管内存。托管内存会在 CPU/GPU 间按需页迁移,
 * 而 prefill 一层就要写 4GB scratch —— 迁移开销吃掉了绝大部分时间。dequant 现在全在
 * GPU 上做, CPU 侧只有两条调试路(DS4_VQ_CPU_GATHER / DS4_VQ_GPU=0)需要碰它, 那两条
 * 各自走显式 cudaMemcpy, 不该让生产路径为它们背上托管内存的代价。 */
static int cuda_vq_ensure_scratch(uint64_t need_gu, uint64_t need_dn) {
    if (need_gu > g_vq_gu_bytes) {
        if (g_vq_gate_sc) (void)cudaFree(g_vq_gate_sc);
        if (g_vq_up_sc) (void)cudaFree(g_vq_up_sc);
        g_vq_gate_sc = g_vq_up_sc = NULL;
        if (cudaMalloc(&g_vq_gate_sc, need_gu) != cudaSuccess ||
            cudaMalloc(&g_vq_up_sc, need_gu) != cudaSuccess) {
            (void)cudaGetLastError();
            g_vq_gu_bytes = 0;
            return 0;
        }
        g_vq_gu_bytes = need_gu;
    }
    if (need_dn > g_vq_dn_bytes) {
        if (g_vq_down_sc) (void)cudaFree(g_vq_down_sc);
        g_vq_down_sc = NULL;
        if (cudaMalloc(&g_vq_down_sc, need_dn) != cudaSuccess) {
            (void)cudaGetLastError();
            g_vq_dn_bytes = 0;
            return 0;
        }
        g_vq_dn_bytes = need_dn;
    }
    return 1;
}

/* f16 mm_id MoE, 两阶段。
 * 第一版是"每 (token,pick) 一个 block、每线程串行读整行", decode 时只有 6 个 block ⇒
 * SM 几乎全空转, 且线程内跨行跳读完全不合并访存, 实测只有带宽上限的 ~2%。
 * 现在: ① 每 warp 负责一行, warp 内 32 lane 沿 IN 连续取(合并访存)后 shfl 规约;
 *       ② 把 MID/OUT 维切成 grid.z, decode 也能铺满 SM。
 * clamp 语义与 Metal CPU 参考路逐字一致: gate 只截上界, up 双向截。 */
#define DS4_VQ_WARPS_PER_BLOCK 8u
/* 码本 shared 容量(半精度个数): 当前配方 nc=512×dim=4=2048, 留一倍余量 */
#define DS4_VQ_CB_CAP_HALFS 4096u
/* 每 warp 位流暂存字数: 一行最长 4096列/4×9bit=1152B, 取 304 words=1216B 留余量 */
#define DS4_VQ_BITWORDS 304u

__device__ __forceinline__ static float vq_warp_reduce(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

/* ---- 融合路: 解码即用, 不落 f16 中间权重 ----
 * 账: decode 每 token 只需 6 专家×43 层的压缩态 blob ≈ 774MB; 而"先 dequant 成 f16
 * 再 matmul"要写 12.4GB + 读回 12.4GB = 24.8GB —— 放大 32 倍, 4.7 t/s 就是这么来的。
 * 那份 f16 解出来只被用一次就丢, 根本不该存在。这里让 matmul 直接从 blob 解码,
 * 内存流量回到压缩态本身。
 * 只用于 decode: prefill 时同一专家被上千 token 共用, 落一次 f16 再复用才划算。 */

/* 解码 (expert e, which) 矩阵第 row 行并与 vec 点乘; warp 内 lane 沿列分摊。
 * 返回的是本 lane 的部分和, 调用方做 warp 规约。 */
/* 诊断用: 0=完整 1=只解位流 2=位流+码本读(不乘 x) —— 二分定位耗时段 */
__device__ int g_vq_exp_mode = 0;
/* kernel 内自计时(零权限替代 ncu): [0]=位流预取 cycles [1]=解码+乘加 cycles
 * [2]=样本数。只在 DS4_VQ_CYC=1 时写, 用 clock64() 读 SM 时钟。 */
__device__ unsigned long long g_vq_cyc[3] = {0, 0, 0};
__device__ int g_vq_cyc_on = 0;

/* ===== fused2: 特化高速 VQ 解码路 (2026-08-18, 冲 30 t/s) =====
 * 依据(vqcyc/EXP 消融): 旧 fused 每 warp 每行重复 vtab+payload 头解析(串行 global
 * 依赖链)+位流 shared 搬运 2×syncwarp+码本 global gather ⇒ 8.2cy/idx。
 * 特化条件(host vq2_probe_layer 校验后启用): d16=4, w1/w3 nc=512(9bit),
 * w2 nc=256(8bit), payload 4B 对齐, 全 256 专家槽在位。
 * 结构: 一 block 一 (token,expert), 4 warps×R 行/warp; x/h 与码本进 shared;
 * 位流按 lane 36B(9bit)/32B(8bit) 对齐段合并读进寄存器, funnelshift 纯 ALU 抽位。
 * 乘加顺序与旧 kernel 不同(重排容差), 对拍口径=vq diag cos/score-ids。 */
#define DS4_VQ2_ROWS_PER_WARP 8u

typedef struct { uint64_t gr_off1, ix_off1, gr_off3, ix_off3; } ds4_vq2_hdr;

/* thread0 per block: vtab→payload→gr/ix 偏移(全 block 共用, 一次) */
__device__ __forceinline__ static void vq2_resolve(
        const uint8_t *blob, int e, int which, uint32_t nc,
        uint32_t rows, uint64_t *gr_off, uint64_t *ix_off) {
    const uint64_t *vtab = (const uint64_t *)(blob + 16);
    const uint64_t off = vtab[(size_t)e * 3 + which];
    *gr_off = off + 16 + (uint64_t)nc * 4 * 2;
    *ix_off = *gr_off + (uint64_t)rows * 2;
}

/* 一行 9bit 点积: lane 段=36B(9 u32), 32 idx funnelshift 抽位, 码本 shared gather */
__device__ __forceinline__ static float vq2_row_dot9(
        const uint8_t *ix, uint32_t row, const float *xs, const __half *cb,
        uint32_t lane, uint32_t cols) {
    const uint32_t nidx_row = cols >> 2;                   /* d16=4 */
    /* lane 段覆盖 32 idx×4 元素=128 列; cols<4096(如 down 的 MID=2048)时高 lane
     * 在行内无段 —— 越 shared/位流界, 必须先退出(warp_sum 汇 0)。 */
    if ((lane << 7u) >= cols) return 0.0f;
    const uint8_t *seg = ix + (size_t)row * ((nidx_row * 9u) >> 3) + (size_t)lane * 36u;
    /* payload 无对齐保证(vq_pack 紧凑): 对齐基址读 40B 窗口, 错位字节并入位偏移 */
    const uint32_t misal = (uint32_t)((uintptr_t)seg & 3u);
    const uint32_t *wp = (const uint32_t *)(seg - misal);
    uint32_t bw[10];
    #pragma unroll
    for (uint32_t k = 0; k < 10u; k++) bw[k] = wp[k];
    float acc = 0.0f;
    #pragma unroll
    for (uint32_t k = 0; k < 32u; k++) {
        const uint32_t bit = k * 9u + misal * 8u, wi = bit >> 5u, sh = bit & 31u;
        const uint32_t v = __funnelshift_r(bw[wi], bw[wi + 1u < 10u ? wi + 1u : 9u], sh) & 511u;
        const uint32_t col4 = (lane * 32u + k) << 2u;
        const float4 xv = *(const float4 *)&xs[col4];
        const uint32_t cq0 = ((const uint32_t *)cb)[v * 2u];
        const uint32_t cq1 = ((const uint32_t *)cb)[v * 2u + 1u];
        const float2 f01 = __half22float2(*(const half2 *)&cq0);
        const float2 f23 = __half22float2(*(const half2 *)&cq1);
        acc += f01.x * xv.x + f01.y * xv.y + f23.x * xv.z + f23.y * xv.w;
    }
    return acc;
}

/* 一行 8bit 点积(w2): lane 段=32B, idx 直取字节 */
__device__ __forceinline__ static float vq2_row_dot8(
        const uint8_t *ix, uint32_t row, const float *xs, const __half *cb,
        uint32_t lane, uint32_t cols) {
    const uint32_t nidx_row = cols >> 2u;
    const uint8_t *seg = ix + (size_t)row * nidx_row + (size_t)lane * (nidx_row >> 5u);
    const uint32_t per_lane = nidx_row >> 5u;              /* 2048/4/32 = 16 idx */
    const uint32_t misal = (uint32_t)((uintptr_t)seg & 3u);
    const uint32_t *wp = (const uint32_t *)(seg - misal);
    uint32_t bw[5];
    #pragma unroll
    for (uint32_t k = 0; k < 5u; k++) bw[k] = wp[k];
    float acc = 0.0f;
    #pragma unroll
    for (uint32_t k = 0; k < 16u; k++) {
        const uint32_t bj = misal + k;
        const uint32_t v = (bw[bj >> 2u] >> ((bj & 3u) * 8u)) & 255u;
        const uint32_t col4 = (lane * per_lane + k) << 2u;
        const float4 xv = *(const float4 *)&xs[col4];
        const uint32_t cq0 = ((const uint32_t *)cb)[v * 2u];
        const uint32_t cq1 = ((const uint32_t *)cb)[v * 2u + 1u];
        const float2 f01 = __half22float2(*(const half2 *)&cq0);
        const float2 f23 = __half22float2(*(const half2 *)&cq1);
        acc += f01.x * xv.x + f01.y * xv.y + f23.x * xv.z + f23.y * xv.w;
    }
    return acc;
}

__device__ __forceinline__ static float vq2_warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

/* gateup: grid=(ntok, MID/(4*R), nexp), block=128。shared: x[IN] f32 + cb1/cb3 各 4KB */
__global__ static void vq_moe_gateup_fused2_kernel(
        float *mid_out, const uint8_t *blob, const int32_t *sel, const float *rw,
        const float *x, uint32_t n_expert, uint32_t IN, uint32_t MID, float clamp) {
    const uint32_t t = blockIdx.x, pk = blockIdx.z;
    const uint64_t pair = (uint64_t)t * n_expert + pk;
    const int32_t e = sel[pair];
    if (e < 0 || rw[pair] == 0.0f) return;
    extern __shared__ float sh2[];
    float *xs = sh2;                                        /* IN floats */
    __half *cb1 = (__half *)(xs + IN);                      /* 512*4 halfs */
    __half *cb3 = cb1 + 2048;
    __shared__ ds4_vq2_hdr hdr;
    if (threadIdx.x == 0) {
        vq2_resolve(blob, e, 0, 512u, MID, &hdr.gr_off1, &hdr.ix_off1);
        vq2_resolve(blob, e, 1, 512u, MID, &hdr.gr_off3, &hdr.ix_off3);
    }
    const float *xt = x + (uint64_t)t * IN;
    for (uint32_t i = threadIdx.x; i < IN; i += blockDim.x) xs[i] = xt[i];
    {   /* 码本: payload+16 起 512*4 halfs (payload 无对齐保证, u8 组装) */
        const uint64_t *vtab = (const uint64_t *)(blob + 16);
        const uint8_t *c1 = blob + vtab[(size_t)e * 3] + 16;
        const uint8_t *c3 = blob + vtab[(size_t)e * 3 + 1] + 16;
        for (uint32_t i = threadIdx.x; i < 2048u; i += blockDim.x) {
            uint16_t h1 = (uint16_t)c1[i * 2u] | ((uint16_t)c1[i * 2u + 1u] << 8);
            uint16_t h3 = (uint16_t)c3[i * 2u] | ((uint16_t)c3[i * 2u + 1u] << 8);
            cb1[i] = *(const __half *)&h1; cb3[i] = *(const __half *)&h3;
        }
    }
    __syncthreads();
    const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const uint8_t *ix1 = blob + hdr.ix_off1, *ix3 = blob + hdr.ix_off3;
    const __half *gr1 = (const __half *)(blob + hdr.gr_off1);
    const __half *gr3 = (const __half *)(blob + hdr.gr_off3);
    const float w = rw[pair];
    #pragma unroll
    for (uint32_t r = 0; r < DS4_VQ2_ROWS_PER_WARP; r++) {
        const uint32_t m = (blockIdx.y * 4u + warp) * DS4_VQ2_ROWS_PER_WARP + r;
        if (m >= MID) return;
        float g = vq2_row_dot9(ix1, m, xs, cb1, lane, IN);
        float u = vq2_row_dot9(ix3, m, xs, cb3, lane, IN);
        g = vq2_warp_sum(g);
        u = vq2_warp_sum(u);
        if (lane == 0) {
            uint16_t g1h = (uint16_t)((const uint8_t *)gr1)[m * 2u] | ((uint16_t)((const uint8_t *)gr1)[m * 2u + 1u] << 8);
            uint16_t g3h = (uint16_t)((const uint8_t *)gr3)[m * 2u] | ((uint16_t)((const uint8_t *)gr3)[m * 2u + 1u] << 8);
            g *= __half2float(*(const __half *)&g1h);
            u *= __half2float(*(const __half *)&g3h);
            if (clamp > 0.0f) {
                if (g > clamp) g = clamp;
                if (u > clamp) u = clamp;
                if (u < -clamp) u = -clamp;
            }
            mid_out[pair * MID + m] = (g / (1.0f + __expf(-g))) * u * w;
        }
    }
}

/* down: grid=(ntok, OUT/(4*R), nexp)。shared: h[MID] f32 + cb2 2KB。
 * 旧 fused 的 rw 乘在 down 段; fused2 把 rw 折进 gateup 的 mid(silu*u*w), down 直接累加。 */
template <uint32_t NCB>
__global__ static void vq_moe_down_fused2_kernel(
        float *partial, const uint8_t *blob, const int32_t *sel, const float *rw,
        const float *h, uint32_t n_expert, uint32_t MID, uint32_t OUT) {
    /* per-pick 并行(原设计), 但各 pick 直写独立 partial 平面代替 atomicAdd —— 汇总由
     * vq2_down_reduce_kernel 以固定 pk 序完成 ⇒ 温 0 逐 bit 可复现。
     * NCB = w2 码本词数(256=8bit 索引, 512=9bit)。 */
    const uint32_t t = blockIdx.x, pk = blockIdx.z;
    const uint64_t pair = (uint64_t)t * n_expert + pk;
    const int32_t e = sel[pair];
    if (e < 0 || rw[pair] == 0.0f) return;   /* 未写平面由 reduce 按 sel/rw 跳过 */
    extern __shared__ float sh2[];
    float *hs = sh2;                                        /* MID floats */
    __half *cb2 = (__half *)(hs + MID);                     /* NCB*4 halfs */
    __shared__ uint64_t ix_off2, gr_off2;
    if (threadIdx.x == 0) {
        uint64_t g2, i2;
        vq2_resolve(blob, e, 2, NCB, OUT, &g2, &i2);
        gr_off2 = g2; ix_off2 = i2;
    }
    const float *ht = h + pair * MID;
    for (uint32_t i = threadIdx.x; i < MID; i += blockDim.x) hs[i] = ht[i];
    {
        const uint64_t *vtab = (const uint64_t *)(blob + 16);
        const uint8_t *c2 = blob + vtab[(size_t)e * 3 + 2] + 16;
        for (uint32_t i = threadIdx.x; i < NCB * 4u; i += blockDim.x) {
            uint16_t h2 = (uint16_t)c2[i * 2u] | ((uint16_t)c2[i * 2u + 1u] << 8);
            cb2[i] = *(const __half *)&h2;
        }
    }
    __syncthreads();
    const uint32_t warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const uint8_t *ix2 = blob + ix_off2;
    const __half *gr2 = (const __half *)(blob + gr_off2);
    #pragma unroll
    for (uint32_t r = 0; r < DS4_VQ2_ROWS_PER_WARP; r++) {
        const uint32_t o = (blockIdx.y * 4u + warp) * DS4_VQ2_ROWS_PER_WARP + r;
        if (o >= OUT) return;
        float acc = (NCB == 512u) ? vq2_row_dot9(ix2, o, hs, cb2, lane, MID)
                                  : vq2_row_dot8(ix2, o, hs, cb2, lane, MID);
        acc = vq2_warp_sum(acc);
        if (lane == 0) {
            uint16_t g2h = (uint16_t)((const uint8_t *)gr2)[o * 2u] | ((uint16_t)((const uint8_t *)gr2)[o * 2u + 1u] << 8);
            /* w 已由 gateup_fused2 乘进 h, 此处不乘 */
            partial[pair * OUT + o] = acc * __half2float(*(const __half *)&g2h);
        }
    }
}

/* 固定 pk 序汇总 partial 平面 → out (确定性加序) */
__global__ static void vq2_down_reduce_kernel(
        float *out, const float *partial, const int32_t *sel, const float *rw,
        uint32_t n_expert, uint32_t OUT) {
    const uint32_t t = blockIdx.y;
    const uint32_t o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= OUT) return;
    float s = 0.0f;
    for (uint32_t pk = 0; pk < n_expert; pk++) {
        const uint64_t pair = (uint64_t)t * n_expert + pk;
        if (sel[pair] >= 0 && rw[pair] != 0.0f) s += partial[pair * OUT + o];
    }
    out[(uint64_t)t * OUT + o] = s;
}

/* host 侧特化闸: 全 256 专家×3 槽 在位+4B 对齐+维度/码本匹配 → fused2 可用 */
/* device 版 probe: 对 arena 副本判定, 避免 host 去读已 madvise(DONTNEED) 的 mmap
 * 原件 —— 那会触发 SSD refault + readahead 把模型页重新灌进 page cache, 与 81GiB
 * arena 叠加造成内存压力, decode 稳态实测掉 ~25%。单线程, μs 级。 */
__global__ static void vq2_probe_kernel(const uint8_t *blob, uint32_t n_total_expert,
                                        uint32_t IN, uint32_t MID, uint32_t OUT,
                                        int32_t *out_w2n) {
    const uint64_t *vtab = (const uint64_t *)(blob + 16);
    uint32_t w2n = 0;
    for (uint32_t e = 0; e < n_total_expert; e++) {
        for (int w = 0; w < 3; w++) {
            const uint64_t off = vtab[(size_t)e * 3 + w];
            if (!off) { out_w2n[0] = 0; return; }
            const uint8_t *pay = blob + off;
            uint16_t d16, n16; uint32_t rows, cols;
            memcpy(&d16, pay + 4, 2); memcpy(&n16, pay + 6, 2);
            memcpy(&rows, pay + 8, 4); memcpy(&cols, pay + 12, 4);
            const uint32_t want_r = (w == 2) ? OUT : MID;
            const uint32_t want_c = (w == 2) ? MID : IN;
            if (d16 != 4u || rows != want_r || cols != want_c) { out_w2n[0] = 0; return; }
            if (w < 2) { if (n16 != 512u) { out_w2n[0] = 0; return; } }
            else {
                if (n16 != 256u && n16 != 512u) { out_w2n[0] = 0; return; }
                if (w2n == 0) w2n = n16; else if (w2n != n16) { out_w2n[0] = 0; return; }
            }
        }
    }
    out_w2n[0] = (int32_t)w2n;
}

/* 返回 w2 码本词数(256/512), 0 = fused2 不可用。w1/w3 固定 512。 */
static uint32_t vq2_probe_layer(const uint8_t *blob, uint32_t n_total_expert,
                                uint32_t IN, uint32_t MID, uint32_t OUT) {
    const uint64_t *vtab = (const uint64_t *)(blob + 16);
    uint32_t w2n = 0;
    for (uint32_t e = 0; e < n_total_expert; e++) {
        for (int w = 0; w < 3; w++) {
            const uint64_t off = vtab[(size_t)e * 3 + w];
            if (!off) return 0;   /* 对齐不要求: kernel 侧对齐窗口读+misal 位偏移 */
            const uint8_t *pay = blob + off;
            uint16_t d16, n16; uint32_t rows, cols;
            memcpy(&d16, pay + 4, 2); memcpy(&n16, pay + 6, 2);
            memcpy(&rows, pay + 8, 4); memcpy(&cols, pay + 12, 4);
            const uint32_t want_r = (w == 2) ? OUT : MID;
            const uint32_t want_c = (w == 2) ? MID : IN;
            if (d16 != 4u || rows != want_r || cols != want_c) return 0;
            if (w < 2) { if (n16 != 512u) return 0; }
            else {
                if (n16 != 256u && n16 != 512u) return 0;
                if (w2n == 0) w2n = n16; else if (w2n != n16) return 0;
            }
        }
    }
    return w2n;
}


/* bs: 本 warp 专属的位流暂存(至少 DS4_VQ_BITWORDS 个 uint32); NULL=直接读全局 */
__device__ __forceinline__ static float vq_row_dot_dev(
        const uint8_t *blob, int e, int which, uint32_t row,
        const float *vec, uint32_t lane,
        const uint8_t *down_base, uint64_t down_ebytes, uint32_t mid_for_cold,
        const __half *cb_sh, uint32_t *bs) {
    const uint64_t off = vq_slot_dev(blob, e, which);
    if (off == 0u) {
        /* 冷 w2: base go1b 34B/256el sign 字节 → ±d, 边解边点乘 */
        if (which != 2 || down_base == NULL || down_ebytes == 0u) return 0.0f;
        const uint32_t nblk_row = mid_for_cold / 256u;
        const uint8_t *rb = down_base + (uint64_t)e * down_ebytes + (size_t)row * nblk_row * 34u;
        float acc = 0.0f;
        for (uint32_t b = lane; b < nblk_row; b += 32u) {
            uint16_t dsc; memcpy(&dsc, rb + (size_t)b * 34u, 2);
            __half dh; memcpy(&dh, &dsc, 2);
            const float dv = __half2float(dh);
            const uint8_t *sg = rb + (size_t)b * 34u + 2;
            const float *vs = vec + (size_t)b * 256u;
            for (int k = 0; k < 256; k++)
                acc += (((sg[k >> 3] >> (k & 7)) & 1) ? dv : -dv) * vs[k];
        }
        return acc;
    }
    const uint8_t *pay = blob + off;
    uint32_t mg; memcpy(&mg, pay, 4);
    if (mg != DS4VQ_MAT_MAGIC) return 0.0f;
    uint16_t d16, n16; memcpy(&d16, pay + 4, 2); memcpy(&n16, pay + 6, 2);
    uint32_t rows, cols; memcpy(&rows, pay + 8, 4); memcpy(&cols, pay + 12, 4);
    uint32_t nbit = 0; while ((1u << nbit) < (uint32_t)n16) nbit++; if (nbit < 1u) nbit = 1u;
    const uint8_t *cb = pay + 16;
    const uint8_t *gr = cb + (size_t)n16 * d16 * 2;
    const uint8_t *ix = gr + (size_t)rows * 2;
    const uint32_t imsk = (nbit >= 32u) ? 0xFFFFFFFFu : ((1u << nbit) - 1u);
    const uint32_t nidx_row = cols / d16;
    __half gh; memcpy(&gh, gr + (size_t)row * 2, 2);
    const float g = __half2float(gh);
    const size_t i0 = (size_t)row * nidx_row;
    /* ★warp 协作合并读位流★
     * 9-bit 索引让相邻 lane 的读地址只差 1.125 字节, 各自发 4 字节非对齐读时硬件
     * 无法合并 —— 一个 warp 为了 36 字节的数据发 32 个独立请求。实测位流单独就占
     * 了 70% 的 kernel 时间, 折合仅 14.5GB/s(带宽的 5.7%)。
     * 改成: 全 warp 先按 4 字节对齐把整行位流合并搬进 shared, 之后各 lane 从片上
     * 取位段。取值与直接读全局逐位相同。 */
    size_t bs_base = 0;
    const long long cyc_a = g_vq_cyc_on ? clock64() : 0;
    if (bs) {
        const size_t byte0 = (i0 * nbit) >> 3;
        bs_base = byte0 & ~(size_t)3;
        const size_t byte_end = (((i0 + nidx_row) * nbit + 7u) >> 3) + 4u;   /* +4: 尾部窗口 */
        uint32_t nw = (uint32_t)((byte_end - bs_base + 3u) >> 2);
        if (nw > DS4_VQ_BITWORDS) nw = DS4_VQ_BITWORDS;
        for (uint32_t w = lane; w < nw; w += 32u)
            memcpy(&bs[w], ix + bs_base + (size_t)w * 4u, 4);
        __syncwarp();
    }
    const long long cyc_b = g_vq_cyc_on ? clock64() : 0;
    /* 4 路独立累加链。单链版本每次迭代是"读位流→移位→查码本(L1 随机, ~30cy)→FMA"
     * 一条串行依赖, warp 全程在等访存返回, 实测只跑出 51GB/s(带宽的 20%)。四条链
     * 交错发射即可把彼此的延迟填掉; 求和顺序变化只影响浮点结合律层面的末位。 */
    /* ★两阶段解码, 打断依赖链★
     * 原写法每次迭代是 "读位流 → 解出 v → 用 v 算码本地址 → 读码本 → 乘加",
     * 后一次访存的地址依赖前一次的结果, 硬件无从预取 —— 两段 ~30cy 的 L1 延迟首尾
     * 相接, 多路 ILP 也只是并行几条同样长的链。ncu: 74% 时间卡在 L1TEX scoreboard,
     * 而 L1 命中率 99.5%(数据本就在片上), 说明纯粹是延迟没被掩盖。
     * 改成: 先一次解出 BATCH 个索引(彼此无依赖, 访存可完全流水), 再拿这批已知地址
     * 批量查码本+乘加(地址提前就绪, 同样可流水)。 */
    #define DS4_VQ_BATCH 8u
    float a0 = 0.0f;
    uint32_t j = lane;
    for (; j + (DS4_VQ_BATCH - 1u) * 32u < nidx_row; j += DS4_VQ_BATCH * 32u) {
        uint32_t vv[DS4_VQ_BATCH];
        #pragma unroll
        for (uint32_t k = 0; k < DS4_VQ_BATCH; k++) {       /* 阶段一: 只解索引 */
            const uint32_t jj = j + k * 32u;
            if (nbit == 8u) {
                vv[k] = ix[i0 + jj];
            } else {
                const size_t bit = (i0 + jj) * nbit;
                uint32_t w;
                if (bs) memcpy(&w, (const uint8_t *)bs + ((bit >> 3) - bs_base), 4);
                else    memcpy(&w, ix + (bit >> 3), 4);
                vv[k] = (w >> (bit & 7)) & imsk;
            }
        }
        if (g_vq_exp_mode == 1) {
            #pragma unroll
            for (uint32_t k = 0; k < DS4_VQ_BATCH; k++) a0 += (float)vv[k];
            continue;
        }
        #pragma unroll
        for (uint32_t k = 0; k < DS4_VQ_BATCH; k++) {       /* 阶段二: 地址已就绪 */
            const float *vs = vec + (size_t)(j + k * 32u) * d16;
            const uint8_t *c = (cb_sh ? (const uint8_t *)cb_sh : cb) + (size_t)vv[k] * d16 * 2;
            if (d16 == 4u) {
                uint64_t q; memcpy(&q, c, 8);               /* 码本项正好 8 字节 */
                #pragma unroll
                for (uint32_t d = 0; d < 4u; d++) {
                    __half hh; const uint16_t u = (uint16_t)(q >> (d * 16));
                    memcpy(&hh, &u, 2);
                    a0 += __half2float(hh) * vs[d];
                }
            } else {
                for (uint32_t d = 0; d < d16; d++) {
                    __half hh; memcpy(&hh, c + (size_t)d * 2, 2);
                    a0 += __half2float(hh) * vs[d];
                }
            }
        }
    }
    for (; j < nidx_row; j += 32u) {            /* 尾巴 */
        uint32_t v;
        if (nbit == 8u) {
            v = ix[i0 + j];
        } else {
            const size_t bit = (i0 + j) * nbit;
            uint32_t w;
            if (bs) memcpy(&w, (const uint8_t *)bs + ((bit >> 3) - bs_base), 4);
            else    memcpy(&w, ix + (bit >> 3), 4);
            v = (w >> (bit & 7)) & imsk;
        }
        const float *vs = vec + (size_t)j * d16;
        if (cb_sh) {
            const __half *pc = cb_sh + (size_t)v * d16;
            for (uint32_t d = 0; d < d16; d++) a0 += __half2float(pc[d]) * vs[d];
        } else {
            const uint8_t *c = cb + (size_t)v * d16 * 2;
            for (uint32_t d = 0; d < d16; d++) {
                __half ch; memcpy(&ch, c + (size_t)d * 2, 2);
                a0 += __half2float(ch) * vs[d];
            }
        }
    }
    if (g_vq_cyc_on && lane == 0) {
        const long long cyc_c = clock64();
        atomicAdd(&g_vq_cyc[0], (unsigned long long)(cyc_b - cyc_a));   /* 位流预取 */
        atomicAdd(&g_vq_cyc[1], (unsigned long long)(cyc_c - cyc_b));   /* 解码+乘加 */
        atomicAdd(&g_vq_cyc[2], 1ull);
    }
    return a0 * g;   /* g 是整行共用的标量, 提到最后统一乘 */
}

/* 双位流合并点积(gateup 专用): 同一行 m 的 w1/w3 两条独立 gather 链在一个循环里
 * 交替解码+乘加 —— 16 路在飞的 L1 访存互相填延迟(单链版每次只有 8 路, kernel 是
 * 延迟受限: ncu L1TEX scoreboard ~74%)。a0/a1 各自的求和顺序与两次单链调用完全
 * 相同 ⇒ 输出逐 bit 不变。 */
__device__ static void vq_row_dot2_dev(
        const uint8_t *blob, int e, uint32_t row, const float *vec, uint32_t lane,
        uint32_t *bs1, uint32_t *bs3, const __half *cbs1, const __half *cbs3,
        float *out_g, float *out_u) {
    *out_g = 0.0f; *out_u = 0.0f;
    const uint64_t off1 = vq_slot_dev(blob, e, 0);
    const uint64_t off3 = vq_slot_dev(blob, e, 1);
    if (!off1 || !off3) return;
    const uint8_t *pay1 = blob + off1, *pay3 = blob + off3;
    uint32_t mg1, mg3; memcpy(&mg1, pay1, 4); memcpy(&mg3, pay3, 4);
    if (mg1 != DS4VQ_MAT_MAGIC || mg3 != DS4VQ_MAT_MAGIC) return;
    uint16_t d16, n16; memcpy(&d16, pay1 + 4, 2); memcpy(&n16, pay1 + 6, 2);
    uint32_t rows, cols; memcpy(&rows, pay1 + 8, 4); memcpy(&cols, pay1 + 12, 4);
    uint32_t nbit = 0; while ((1u << nbit) < (uint32_t)n16) nbit++; if (nbit < 1u) nbit = 1u;
    const uint8_t *cb1g = pay1 + 16, *cb3g = pay3 + 16;
    const uint8_t *cb1 = cbs1 ? (const uint8_t *)cbs1 : cb1g;
    const uint8_t *cb3 = cbs3 ? (const uint8_t *)cbs3 : cb3g;
    const uint8_t *gr1 = cb1g + (size_t)n16 * d16 * 2, *gr3 = cb3g + (size_t)n16 * d16 * 2;
    const uint8_t *ix1 = gr1 + (size_t)rows * 2, *ix3 = gr3 + (size_t)rows * 2;
    const uint32_t imsk = (nbit >= 32u) ? 0xFFFFFFFFu : ((1u << nbit) - 1u);
    const uint32_t nidx_row = cols / d16;
    __half gh1, gh3; memcpy(&gh1, gr1 + (size_t)row * 2, 2); memcpy(&gh3, gr3 + (size_t)row * 2, 2);
    const float g1 = __half2float(gh1), g3 = __half2float(gh3);
    const size_t i0 = (size_t)row * nidx_row;
    size_t bsb1 = 0, bsb3 = 0;
    {
        const size_t byte0 = (i0 * nbit) >> 3;
        bsb1 = bsb3 = byte0 & ~(size_t)3;
        const size_t byte_end = (((i0 + nidx_row) * nbit + 7u) >> 3) + 4u;
        uint32_t nw = (uint32_t)((byte_end - bsb1 + 3u) >> 2);
        if (nw > DS4_VQ_BITWORDS) nw = DS4_VQ_BITWORDS;
        for (uint32_t w = lane; w < nw; w += 32u) {
            memcpy(&bs1[w], ix1 + bsb1 + (size_t)w * 4u, 4);
            memcpy(&bs3[w], ix3 + bsb3 + (size_t)w * 4u, 4);
        }
        __syncwarp();
    }
    float a0 = 0.0f, a1 = 0.0f;
    uint32_t j = lane;
    for (; j + (DS4_VQ_BATCH - 1u) * 32u < nidx_row; j += DS4_VQ_BATCH * 32u) {
        uint32_t v1[DS4_VQ_BATCH], v3[DS4_VQ_BATCH];
        #pragma unroll
        for (uint32_t k = 0; k < DS4_VQ_BATCH; k++) {
            const uint32_t jj = j + k * 32u;
            if (nbit == 8u) {
                v1[k] = ix1[i0 + jj];
                v3[k] = ix3[i0 + jj];
            } else {
                const size_t bit = (i0 + jj) * nbit;
                uint32_t w1_, w3_;
                memcpy(&w1_, (const uint8_t *)bs1 + ((bit >> 3) - bsb1), 4);
                memcpy(&w3_, (const uint8_t *)bs3 + ((bit >> 3) - bsb3), 4);
                v1[k] = (w1_ >> (bit & 7)) & imsk;
                v3[k] = (w3_ >> (bit & 7)) & imsk;
            }
        }
        #pragma unroll
        for (uint32_t k = 0; k < DS4_VQ_BATCH; k++) {
            const float *vs = vec + (size_t)(j + k * 32u) * d16;
            if (d16 == 4u) {
                uint64_t q1, q3;
                memcpy(&q1, cb1 + (size_t)v1[k] * 8u, 8);
                memcpy(&q3, cb3 + (size_t)v3[k] * 8u, 8);
                #pragma unroll
                for (uint32_t d = 0; d < 4u; d++) {
                    __half h1, h3;
                    const uint16_t u1 = (uint16_t)(q1 >> (d * 16));
                    const uint16_t u3 = (uint16_t)(q3 >> (d * 16));
                    memcpy(&h1, &u1, 2); memcpy(&h3, &u3, 2);
                    a0 += __half2float(h1) * vs[d];
                    a1 += __half2float(h3) * vs[d];
                }
            } else {
                const uint8_t *c1 = cb1 + (size_t)v1[k] * d16 * 2;
                const uint8_t *c3 = cb3 + (size_t)v3[k] * d16 * 2;
                for (uint32_t d = 0; d < d16; d++) {
                    __half h1, h3;
                    memcpy(&h1, c1 + (size_t)d * 2, 2);
                    memcpy(&h3, c3 + (size_t)d * 2, 2);
                    a0 += __half2float(h1) * vs[d];
                    a1 += __half2float(h3) * vs[d];
                }
            }
        }
    }
    for (; j < nidx_row; j += 32u) {
        uint32_t v1, v3;
        if (nbit == 8u) {
            v1 = ix1[i0 + j]; v3 = ix3[i0 + j];
        } else {
            const size_t bit = (i0 + j) * nbit;
            uint32_t w1_, w3_;
            memcpy(&w1_, (const uint8_t *)bs1 + ((bit >> 3) - bsb1), 4);
            memcpy(&w3_, (const uint8_t *)bs3 + ((bit >> 3) - bsb3), 4);
            v1 = (w1_ >> (bit & 7)) & imsk;
            v3 = (w3_ >> (bit & 7)) & imsk;
        }
        const float *vs = vec + (size_t)j * d16;
        const uint8_t *c1 = cb1 + (size_t)v1 * d16 * 2;
        const uint8_t *c3 = cb3 + (size_t)v3 * d16 * 2;
        for (uint32_t d = 0; d < d16; d++) {
            __half h1, h3;
            memcpy(&h1, c1 + (size_t)d * 2, 2);
            memcpy(&h3, c3 + (size_t)d * 2, 2);
            a0 += __half2float(h1) * vs[d];
            a1 += __half2float(h3) * vs[d];
        }
    }
    *out_g = a0 * g1;
    *out_u = a1 * g3;
}

/* 把 (e,which) 矩阵的码本搬进 shared。同一 block 的所有 warp 处理同一个
 * (token,expert) 对 ⇒ 共用一本码本, 搬一次全 block 受益。容量不够返回 0 走全局。 */
__device__ __forceinline__ static int vq_load_cb_sh(
        const uint8_t *blob, int e, int which, __half *dst, uint32_t dst_cap_halfs) {
    const uint64_t off = vq_slot_dev(blob, e, which);
    if (off == 0u) return 0;
    const uint8_t *pay = blob + off;
    uint32_t mg; memcpy(&mg, pay, 4);
    if (mg != DS4VQ_MAT_MAGIC) return 0;
    uint16_t d16, n16; memcpy(&d16, pay + 4, 2); memcpy(&n16, pay + 6, 2);
    const uint32_t n = (uint32_t)n16 * d16;
    if (n > dst_cap_halfs) return 0;
    const uint8_t *cb = pay + 16;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        __half h; memcpy(&h, cb + (size_t)i * 2, 2);
        dst[i] = h;
    }
    return 1;
}

__global__ static void vq_moe_gateup_fused_kernel(
        float *h, const uint8_t *blob, const int32_t *sel, const float *rw,
        const float *x, uint32_t n_expert, uint32_t IN, uint32_t MID, float clamp) {
    /* ★专家放最慢变化的维度★: CUDA 按 blockIdx.x 最快变化调度, 原来 pk 在 y、行块
     * 在 z, 于是同时在跑的 block 分属 6 个不同专家 —— 12 个内存流(6 专家×gate/up)
     * 在 DRAM 上交替跳, row buffer 全程冲突, 位流实测只跑出 4GB/s。
     * 改成行块在 y、专家在 z 后, 同时运行的 block 读同一专家的连续位流。 */
    const uint32_t t = blockIdx.x, pk = blockIdx.z;
    const uint64_t pair = (uint64_t)t * n_expert + pk;
    const int32_t e = sel[pair];
    if (e < 0 || rw[pair] == 0.0f) return;
    /* 码本进 shared: 同一 block 的 8 个 warp 处理同一个 (token,expert) 对, 共用同
     * 一本码本(512×4×2=4KB)。随机查表落到片上后不再受 L1 tag/串行化限制 —— 这是
     * 当前 kernel 唯一的重瓶颈(每 4 元素一次随机 gather)。
     * (早前一次失败的尝试是把它和"多行/block"一起改, 两个变量混在一起, 这次只动这个。) */
    /* x 不进 shared: 实测零收益(它在 L1/L2 本就高命中), 而 16KB 会把 occupancy 压死。
     * shared 全部让给码本 —— 真正的瓶颈是位流的非合并读, 见 vq_row_dot_dev。 */
    /* shared 只给位流: 码本进 shared 实测零收益(它在片上本就高命中), 白占 8KB 反而
     * 压低 occupancy。位流的非合并读才是那 70%。 */
    extern __shared__ uint32_t shg[];
    const float *sx = x + (uint64_t)t * IN;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t *bs1 = shg + (size_t)warp * 2u * DS4_VQ_BITWORDS;
    uint32_t *bs3 = bs1 + DS4_VQ_BITWORDS;
    /* 码本进 shared(block 级一次): dot2 后 gather 密度翻倍, 片上码本免 L1 tag 竞争。
     * 布局: [warps*2*BITWORDS u32][cb1 512*4 half][cb3 512*4 half] */
    __half *cbs1 = (__half *)(shg + (size_t)DS4_VQ_WARPS_PER_BLOCK * 2u * DS4_VQ_BITWORDS);
    __half *cbs3 = cbs1 + 2048u;
    float *xsh = (float *)(cbs3 + 2048u);   /* x 向量 IN floats 进 shared */
    {
        const uint64_t o1_ = vq_slot_dev(blob, e, 0);
        const uint64_t o3_ = vq_slot_dev(blob, e, 1);
        const uint8_t *c1_ = blob + o1_ + 16, *c3_ = blob + o3_ + 16;
        for (uint32_t i = threadIdx.x; i < 2048u; i += blockDim.x) {
            uint16_t u1_, u3_;
            memcpy(&u1_, c1_ + (size_t)i * 2u, 2);
            memcpy(&u3_, c3_ + (size_t)i * 2u, 2);
            memcpy(&cbs1[i], &u1_, 2);
            memcpy(&cbs3[i], &u3_, 2);
        }
        for (uint32_t i = threadIdx.x; i < IN; i += blockDim.x) xsh[i] = sx[i];
        __syncthreads();
    }
    const uint32_t m = blockIdx.y * DS4_VQ_WARPS_PER_BLOCK + warp;
    if (m >= MID) return;
    float g, u;
    vq_row_dot2_dev(blob, e, m, xsh, lane, bs1, bs3, cbs1, cbs3, &g, &u);
    g = vq_warp_reduce(g);
    u = vq_warp_reduce(u);
    if (lane == 0) {
        if (clamp > 0.0f) {
            if (g > clamp) g = clamp;
            if (u > clamp) u = clamp;
            if (u < -clamp) u = -clamp;
        }
        h[pair * MID + m] = (g / (1.0f + __expf(-g))) * u;
    }
}

__global__ static void vq_moe_down_fused_kernel(
        float *partial, const uint8_t *blob, const int32_t *sel, const float *rw,
        const float *h, const uint8_t *down_base, uint64_t down_ebytes,
        uint32_t n_expert, uint32_t MID, uint32_t OUT) {
    /* per-pick 并行 + 独立 partial 平面直写(无 atomicAdd), 汇总由固定 pk 序的
     * vq2_down_reduce_kernel 完成 ⇒ 温 0 逐 bit 可复现且保留 6× 并行度(曾试
     * 单 warp 串行 6 pick 的确定化, down 从 ~0.4ms 涨到 ~1.2ms/层)。 */
    const uint32_t t = blockIdx.x, pk = blockIdx.z;
    const uint64_t pair = (uint64_t)t * n_expert + pk;
    const int32_t e = sel[pair];
    if (e < 0) return;   /* 未写平面由 reduce 按 sel/rw 跳过 */
    const float w = rw[pair];
    if (w == 0.0f) return;
    extern __shared__ uint32_t shd[];
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    uint32_t *bs = shd + (size_t)warp * DS4_VQ_BITWORDS;
    /* w2 码本进 shared(block 级一次), 同 gateup。槽缺失(冷 w2)时跳过, dot 走冷路。 */
    __half *cbs2 = (__half *)(shd + (size_t)DS4_VQ_WARPS_PER_BLOCK * DS4_VQ_BITWORDS);
    const uint64_t o2_ = vq_slot_dev(blob, e, 2);
    if (o2_) {
        const uint8_t *c2_ = blob + o2_ + 16;
        for (uint32_t i = threadIdx.x; i < 2048u; i += blockDim.x) {
            uint16_t u2_; memcpy(&u2_, c2_ + (size_t)i * 2u, 2);
            memcpy(&cbs2[i], &u2_, 2);
        }
    }
    __syncthreads();
    const uint32_t o = blockIdx.y * DS4_VQ_WARPS_PER_BLOCK + warp;
    if (o >= OUT) return;
    float acc = vq_row_dot_dev(blob, e, 2, o, h + pair * MID, lane, down_base, down_ebytes, MID,
                               o2_ ? cbs2 : NULL, bs);
    acc = vq_warp_reduce(acc);
    if (lane == 0) partial[pair * OUT + o] = w * acc;
}

/* 阶段一: h[t][pk][m] = silu(clamp(Wg[slot][m]·x)) * clamp(Wu[slot][m]·x) */
__global__ static void vq_moe_gateup_kernel(
        float *h, const __half *Wg, const __half *Wu,
        const float *x, const int32_t *sel_slot, const float *rw,
        uint32_t n_expert, uint32_t IN, uint32_t MID, float clamp) {
    const uint32_t t = blockIdx.x, pk = blockIdx.y;
    const int32_t slot = sel_slot[(uint64_t)t * n_expert + pk];
    if (slot < 0) return;
    /* 快路把 slot 直接取成 pair 序号, 无效 pick 那段 scratch 未被解码 —— 靠权重为 0
     * 跳过它, 免得拿未初始化的权重算出 NaN 再污染 h。 */
    if (rw[(uint64_t)t * n_expert + pk] == 0.0f) return;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t m = blockIdx.z * DS4_VQ_WARPS_PER_BLOCK + warp;
    if (m >= MID) return;

    const float *xt = x + (uint64_t)t * IN;
    const __half *gr = Wg + ((uint64_t)slot * MID + m) * IN;
    const __half *ur = Wu + ((uint64_t)slot * MID + m) * IN;
    float g = 0.0f, u = 0.0f;
    for (uint32_t i = lane; i < IN; i += 32u) {      /* lane 连续 ⇒ 合并访存 */
        const float xv = xt[i];
        g += __half2float(gr[i]) * xv;
        u += __half2float(ur[i]) * xv;
    }
    g = vq_warp_reduce(g);
    u = vq_warp_reduce(u);
    if (lane == 0) {
        if (clamp > 0.0f) {
            if (g > clamp) g = clamp;
            if (u > clamp) u = clamp;
            if (u < -clamp) u = -clamp;
        }
        h[((uint64_t)t * n_expert + pk) * MID + m] = (g / (1.0f + __expf(-g))) * u;
    }
}

/* 阶段二: out[t][o] += w[t][pk] * (Wd[slot][o] · h[t][pk]) */
__global__ static void vq_moe_down_kernel(
        float *out, const __half *Wd, const float *h,
        const int32_t *sel_slot, const float *rw,
        uint32_t n_expert, uint32_t MID, uint32_t OUT) {
    /* 单 warp 串行累加全部 pick(固定加序 ⇒ 温 0 可复现), 直写代替 atomicAdd。
     * 历史 bug 备案: 旧版 grid.y=pk + 256 线程(8 warp)配 WARPS_PER_BLOCK=4 的 o
     * 公式, 每个 o 被两个 warp 重复 atomicAdd ⇒ 输出恒 2×。 */
    const uint32_t t = blockIdx.x;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t o = blockIdx.z * DS4_VQ_WARPS_PER_BLOCK + warp;
    if (o >= OUT) return;
    float sum = 0.0f;
    for (uint32_t pk = 0; pk < n_expert; pk++) {
        const int32_t slot = sel_slot[(uint64_t)t * n_expert + pk];
        if (slot < 0) continue;
        const float w = rw[(uint64_t)t * n_expert + pk];
        if (w == 0.0f) continue;
        const float *ht = h + ((uint64_t)t * n_expert + pk) * MID;
        const __half *dr = Wd + ((uint64_t)slot * OUT + o) * MID;
        float acc = 0.0f;
        for (uint32_t m = lane; m < MID; m += 32u) acc += __half2float(dr[m]) * ht[m];
        acc = vq_warp_reduce(acc);
        sum += w * acc;
    }
    if (lane == 0) out[(uint64_t)t * OUT + o] = sum;
}

/* CPU 参考路(DS4_VQ_GPU=0): 与 Metal 的 CPU MoE 逐行同义, 用于数值对齐。
 * 朴素三重循环, 只求正确不求快 —— 生产走 GPU。 */
static int cuda_vq_moe_cpu_ref(
        float *out_h, const uint16_t *gsc, const uint16_t *usc, const uint16_t *dnc,
        const float *xin, const int32_t *sel_slot, const float *rw,
        uint32_t n_tokens, uint32_t n_active, uint32_t n_expert,
        uint32_t IN, uint32_t MID, uint32_t OUT, float clamp) {
    float *w1f = (float *)malloc((size_t)MID * IN * sizeof(float));
    float *w3f = (float *)malloc((size_t)MID * IN * sizeof(float));
    float *w2f = (float *)malloc((size_t)OUT * MID * sizeof(float));
    float *gg = (float *)malloc((size_t)MID * sizeof(float));
    float *hb = (float *)malloc((size_t)MID * sizeof(float));
    if (!w1f || !w3f || !w2f || !gg || !hb) {
        free(w1f); free(w3f); free(w2f); free(gg); free(hb);
        return 0;
    }
    memset(out_h, 0, (size_t)n_tokens * OUT * sizeof(float));
    for (uint32_t sl = 0; sl < n_active; sl++) {
        int used = 0;
        for (uint32_t t = 0; t < n_tokens && !used; t++)
            for (uint32_t pk = 0; pk < n_expert; pk++)
                if (sel_slot[(uint64_t)t * n_expert + pk] == (int32_t)sl) { used = 1; break; }
        if (!used) continue;
        const uint16_t *g = gsc + (uint64_t)sl * MID * IN;
        const uint16_t *u = usc + (uint64_t)sl * MID * IN;
        const uint16_t *dn = dnc + (uint64_t)sl * OUT * MID;
        for (size_t j = 0; j < (size_t)MID * IN; j++) { w1f[j] = ds4vq_f16(g[j]); w3f[j] = ds4vq_f16(u[j]); }
        for (size_t j = 0; j < (size_t)OUT * MID; j++) w2f[j] = ds4vq_f16(dn[j]);
        for (uint32_t t = 0; t < n_tokens; t++) {
            float w = 0.0f;
            for (uint32_t pk = 0; pk < n_expert; pk++)
                if (sel_slot[(uint64_t)t * n_expert + pk] == (int32_t)sl) { w = rw[(uint64_t)t * n_expert + pk]; break; }
            if (w == 0.0f) continue;
            const float *xt = xin + (uint64_t)t * IN;
            float *o = out_h + (uint64_t)t * OUT;
            for (uint32_t m = 0; m < MID; m++) {
                float ga = 0.0f, ua = 0.0f;
                for (uint32_t i = 0; i < IN; i++) { ga += w1f[(size_t)m * IN + i] * xt[i]; ua += w3f[(size_t)m * IN + i] * xt[i]; }
                if (clamp > 0.0f) { if (ga > clamp) ga = clamp; if (ua > clamp) ua = clamp; if (ua < -clamp) ua = -clamp; }
                hb[m] = (ga / (1.0f + expf(-ga))) * ua;
            }
            for (uint32_t oo = 0; oo < OUT; oo++) {
                float acc = 0.0f;
                for (uint32_t m = 0; m < MID; m++) acc += w2f[(size_t)oo * MID + m] * hb[m];
                o[oo] += w * acc;
            }
        }
    }
    free(w1f); free(w3f); free(w2f); free(gg); free(hb);
    return 1;
}

static int cuda_vq_moe_forward(
        ds4_gpu_tensor *out, ds4_gpu_tensor *mid_scratch, const ds4_gpu_residual_set *residual,
        const void *model_map, uint64_t down_offset, uint64_t down_expert_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens) {
    const uint8_t *blob = (const uint8_t *)residual->gate_ptr;
    if (!blob || !out || !selected || !weights || !x || n_tokens == 0 || n_expert == 0) return 0;
    const uint8_t *blob_host = blob;   /* mmap original: host-side header reads only */
    /* fused/fuse2 down 的 partial 平面(fuse_max×n_expert×OUT), 须在任何 graph capture
     * 之前分配 —— prefill(非 capture)首次路过这里时建好。 */
    static float *g_vq_partial = NULL;
    if (!g_vq_partial) {
        cudaStreamCaptureStatus vcs_ = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(0, &vcs_);
        if (vcs_ == cudaStreamCaptureStatusNone)
            (void)cudaMalloc(&g_vq_partial, (size_t)4 * n_expert * out_dim * sizeof(float));
    }
    /* Relocate the host-mmap blob pointer onto the HBM arena copy.  The startup
     * span cache (cache_exps=1 on GB10) already copied these bytes to device
     * memory and madvise(DONTNEED)d the source pages; reading the mmap original
     * from the kernel re-faults 65 GiB from SSD per pass and double-counts
     * memory (81 GiB arena + refaulted pages > 121 GiB UMA), which under
     * reclaim pressure produced intermittent NaN/illegal-access. One lookup per
     * layer, cached. */
    if (model_map && residual->vq_bytes && layer_index < 64u) {
        static const uint8_t *reloc[64];
        static uint8_t reloc_done[64];
        if (!reloc_done[layer_index]) {
            reloc_done[layer_index] = 1;
            if ((const char *)blob >= (const char *)model_map) {
                const uint64_t off = (uint64_t)((const char *)blob - (const char *)model_map);
                const uint64_t bend = off + residual->vq_bytes;
                for (const cuda_model_range &r : g_model_ranges) {
                    if (r.host_base == model_map && off >= r.offset && bend > off &&
                        bend <= r.offset + r.bytes) {
                        reloc[layer_index] = (const uint8_t *)(r.device_ptr + (off - r.offset));
                        break;
                    }
                }
            }
            if (((const char *)0) /* DS4_VQ_DEBUG: 诊断开关已删(2026-08-22) */) {
                fprintf(stderr, "ds4: [vq-reloc] L%u blob %s arena\n", layer_index,
                        reloc[layer_index] ? "->" : "NOT in");
                if (!reloc[layer_index] && (const char *)blob >= (const char *)model_map) {
                    const uint64_t off_ = (uint64_t)((const char *)blob - (const char *)model_map);
                    fprintf(stderr, "ds4: [vq-reloc]   off=%llu bytes=%llu nranges=%zu\n",
                            (unsigned long long)off_, (unsigned long long)residual->vq_bytes,
                            g_model_ranges.size());
                    for (const cuda_model_range &r : g_model_ranges) {
                        if (r.host_base == model_map && off_ >= r.offset && off_ < r.offset + r.bytes)
                            fprintf(stderr, "ds4: [vq-reloc]   in-range off=%llu bytes=%llu (blob end %s)\n",
                                    (unsigned long long)r.offset, (unsigned long long)r.bytes,
                                    (off_ + residual->vq_bytes <= r.offset + r.bytes) ? "inside" : "OVERFLOWS");
                    }
                }
            }
        }
        if (reloc[layer_index]) blob = reloc[layer_index];
    }
    {   /* 一次性入参快照: 维度/张量容量任一为 0 或错位, 后面 gather 就会越界写 */
        static int once = 0;
        if (!once++) {
            uint32_t mg = 0; memcpy(&mg, blob_host, 4);
            fprintf(stderr, "ds4: [cuda-vq-init] L%u ntok=%u nexp=%u ntot=%u IN=%u MID=%u OUT=%u clamp=%.3f\n"
                            "     blob=%p magic=%08x sel.bytes=%llu w.bytes=%llu x.bytes=%llu out.bytes=%llu\n"
                            "     down_off=%llu down_ebytes=%llu\n",
                    layer_index, n_tokens, n_expert, n_total_expert,
                    expert_in_dim, expert_mid_dim, out_dim, clamp,
                    (const void *)blob, mg,
                    (unsigned long long)selected->bytes, (unsigned long long)weights->bytes,
                    (unsigned long long)x->bytes, (unsigned long long)out->bytes,
                    (unsigned long long)down_offset, (unsigned long long)down_expert_bytes);
            if (((const char *)0) /* DS4_VQ_ROUTE_DIAG: 路径开关已删(2026-08-22 隐形炸弹清理) */) {
                int32_t sel[8] = {0}; float wt[8] = {0};
                uint32_t ns = n_expert < 8u ? n_expert : 8u;
                (void)ds4_gpu_tensor_read((ds4_gpu_tensor *)selected, 0, sel, ns * sizeof(int32_t));
                (void)ds4_gpu_tensor_read((ds4_gpu_tensor *)weights, 0, wt, ns * sizeof(float));
                fprintf(stderr, "ds4: [vq-route] t0 sel=%d %d %d %d %d %d w=%.3f %.3f %.3f %.3f %.3f %.3f\n",
                        sel[0], sel[1], sel[2], sel[3], sel[4], sel[5],
                        wt[0], wt[1], wt[2], wt[3], wt[4], wt[5]);
            }
            fflush(stderr);
        }
    }

    const uint64_t npair = (uint64_t)n_tokens * n_expert;

    /* ---- decode 快路: 零 host 往返 ----
     * pair 数少时不去重, 每个 (token,pick) 各占一段 scratch, kernel 自己从 blob 读偏移。
     * selected 直接当 slot 用(第 k 个 pair 的权重就在第 k 段), 省掉 D2H + 去重 + H2D。 */
    {   /* 诊断: DS4_VQ_EXP=1 只解位流(跳过码本+乘加), 用来二分定位耗时段 */
        static int exp_set = 0;
        if (!exp_set) {
            exp_set = 1;
            const char *ev = ((const char *)0) /* DS4_VQ_EXP: 路径开关已删(2026-08-22 隐形炸弹清理) */;
            const int mode = ev ? atoi(ev) : 0;
            if (mode) (void)cudaMemcpyToSymbol(g_vq_exp_mode, &mode, sizeof(int));
            const int cyc_on = 0;
            if (cyc_on) (void)cudaMemcpyToSymbol(g_vq_cyc_on, &cyc_on, sizeof(int));
        }
    }
    /* decode 直通链常驻缓冲: 非 capture 时机预分配(prefill/首调都行) */
    const char *fu_env = ((const char *)0) /* DS4_VQ_FUSE_MAX: 路径开关已删(2026-08-22 隐形炸弹清理) */;
    const uint32_t fuse_max = fu_env ? (uint32_t)atoi(fu_env) : 4u;   /* 0 关闭融合 */
    if (n_tokens <= fuse_max && n_tokens > 0) {
        const double prof_t0 = ((const char *)0) /* DS4_VQ_PROF: 路径开关已删(2026-08-22 隐形炸弹清理) */ ? cuda_wall_sec() : 0.0;
        /* 融合路: 不需要任何 dequant scratch, 只要 h 的中间缓冲 */
        const uint64_t hneed = npair * expert_mid_dim * sizeof(float);
        if (!mid_scratch || mid_scratch->bytes < hneed) {
            fprintf(stderr, "ds4: [cuda-vq-fuse] mid scratch %llu < 需要 %llu (L%u)\n",
                    (unsigned long long)(mid_scratch ? mid_scratch->bytes : 0),
                    (unsigned long long)hneed, layer_index);
            return 0;
        }
        const uint8_t *down_base = (down_expert_bytes && down_offset)
                                 ? ((const uint8_t *)model_map + down_offset) : NULL;
        if (cudaMemsetAsync(out->ptr, 0, (size_t)n_tokens * out_dim * sizeof(float), 0) != cudaSuccess) {
            (void)cudaGetLastError(); return 0;
        }
        /* fused2 特化闸(per-layer 一次判定, 判定在 prefill 首层调用=非 capture): 纯 vq4
         * 形态(w1/w3 nc512 + w2 nc256, 全槽在位, 4B 对齐)走高速特化路。 */
        static int16_t g_vq2_w2n[64];   /* 0=未判定, -1=不可用, 256/512=w2 码本词数 */
        int use2 = 1 && layer_index < 64u;
        if (use2) {
            if (!g_vq2_w2n[layer_index]) {
                static int32_t *pd = NULL;
                cudaStreamCaptureStatus pcs2 = cudaStreamCaptureStatusNone;
                (void)cudaStreamIsCapturing(0, &pcs2);
                if (pcs2 == cudaStreamCaptureStatusNone) {
                    if (!pd) (void)cudaMalloc(&pd, sizeof(int32_t));
                    int32_t pn = 0;
                    if (pd) {
                        vq2_probe_kernel<<<1, 1>>>(blob, n_total_expert,
                                expert_in_dim, expert_mid_dim, out_dim, pd);
                        if (cudaMemcpy(&pn, pd, sizeof(pn), cudaMemcpyDeviceToHost) != cudaSuccess) {
                            (void)cudaGetLastError(); pn = 0;
                        }
                    }
                    g_vq2_w2n[layer_index] = pn ? (int16_t)pn : -1;
                }
            }
            /* 512 词 w2 的 fuse2 实测 decode 反慢于 fused(每 block 重载 20KB shared,
             * n=1 无摊销: 5.9 vs 11.3 t/s) —— 仅对验证过的 256 词形态启用。 */
            use2 = g_vq2_w2n[layer_index] == 256;
        }
        if (use2) {
            const uint32_t zg2 = (expert_mid_dim + 31u) / 32u;   /* 4 warps × 8 行 */
            const uint32_t zd2 = (out_dim + 31u) / 32u;
            vq_moe_gateup_fused2_kernel<<<dim3(n_tokens, zg2, n_expert), 128,
                (size_t)expert_in_dim * 4 + 2u * 2048u * 2u>>>(
                (float *)mid_scratch->ptr, blob, (const int32_t *)selected->ptr,
                (const float *)weights->ptr, (const float *)x->ptr,
                n_expert, expert_in_dim, expert_mid_dim, clamp);
            const uint32_t w2n = (uint32_t)g_vq2_w2n[layer_index];
            const size_t dsm = (size_t)expert_mid_dim * 4 + (size_t)w2n * 4u * 2u;
            if (!g_vq_partial) goto vq2_skip;   /* capture 首层前必已分配 */
            if (w2n == 512u)
                vq_moe_down_fused2_kernel<512u><<<dim3(n_tokens, zd2, n_expert), 128, dsm>>>(
                    g_vq_partial, blob, (const int32_t *)selected->ptr,
                    (const float *)weights->ptr, (const float *)mid_scratch->ptr,
                    n_expert, expert_mid_dim, out_dim);
            else
                vq_moe_down_fused2_kernel<256u><<<dim3(n_tokens, zd2, n_expert), 128, dsm>>>(
                    g_vq_partial, blob, (const int32_t *)selected->ptr,
                    (const float *)weights->ptr, (const float *)mid_scratch->ptr,
                    n_expert, expert_mid_dim, out_dim);
            vq2_down_reduce_kernel<<<dim3((out_dim + 255u) / 256u, n_tokens, 1), 256>>>(
                (float *)out->ptr, g_vq_partial, (const int32_t *)selected->ptr,
                (const float *)weights->ptr, n_expert, out_dim);
            if (((const char *)0) /* DS4_VQ_DEBUG: 诊断开关已删(2026-08-22) */)
                fprintf(stderr, "ds4: [cuda-vq-fuse2] L%u ntok=%u\n", layer_index, n_tokens);
            return cuda_ok(cudaGetLastError(), "vq fused2 launch");
        }
        vq2_skip:;
        const uint32_t zg = (expert_mid_dim + DS4_VQ_WARPS_PER_BLOCK - 1u) / DS4_VQ_WARPS_PER_BLOCK;
        const uint32_t zd = (out_dim + DS4_VQ_WARPS_PER_BLOCK - 1u) / DS4_VQ_WARPS_PER_BLOCK;
        vq_moe_gateup_fused_kernel<<<dim3(n_tokens, zg, n_expert), 32 * DS4_VQ_WARPS_PER_BLOCK,
            (size_t)DS4_VQ_WARPS_PER_BLOCK * 2u * DS4_VQ_BITWORDS * sizeof(uint32_t)
            + 2u * 2048u * sizeof(__half) + (size_t)expert_in_dim * sizeof(float)>>>(
            (float *)mid_scratch->ptr, blob, (const int32_t *)selected->ptr,
            (const float *)weights->ptr, (const float *)x->ptr,
            n_expert, expert_in_dim, expert_mid_dim, clamp);
        if (((const char *)0) /* DS4_VQ_ROUTE_DIAG: 路径开关已删(2026-08-22 隐形炸弹清理) */ && layer_index == (uint32_t)atoi(((const char *)0) /* DS4_VQ_ROUTE_DIAG: 路径开关已删(2026-08-22 隐形炸弹清理) */)) {
            (void)cudaDeviceSynchronize();
            float xm[8] = {0}, hm[8] = {0};
            (void)cudaMemcpy(xm, x->ptr, sizeof(xm), cudaMemcpyDeviceToHost);
            (void)cudaMemcpy(hm, mid_scratch->ptr, sizeof(hm), cudaMemcpyDeviceToHost);
            fprintf(stderr, "ds4: [vq-mid] x=%.4g %.4g %.4g %.4g h=%.4g %.4g %.4g %.4g\n",
                    xm[0], xm[1], xm[2], xm[3], hm[0], hm[1], hm[2], hm[3]);
            FILE *fx = fopen("/tmp/vq_x.bin", "wb");
            if (fx) { float *xb = (float *)malloc(4096 * 4); cudaMemcpy(xb, x->ptr, 4096 * 4, cudaMemcpyDeviceToHost); fwrite(xb, 4, 4096, fx); fclose(fx); free(xb); }
            FILE *fh = fopen("/tmp/vq_h.bin", "wb");
            if (fh) { float *hb = (float *)malloc(6 * 2048 * 4); cudaMemcpy(hb, mid_scratch->ptr, 6 * 2048 * 4, cudaMemcpyDeviceToHost); fwrite(hb, 4, 6 * 2048, fh); fclose(fh); }
            FILE *fs = fopen("/tmp/vq_sel.bin", "wb");
            if (fs) { int32_t sb[8] = {0}; float wb[8] = {0};
                cudaMemcpy(sb, selected->ptr, 6 * 4, cudaMemcpyDeviceToHost);
                cudaMemcpy(wb, weights->ptr, 6 * 4, cudaMemcpyDeviceToHost);
                fwrite(sb, 4, 6, fs); fwrite(wb, 4, 6, fs); fclose(fs); }
        }
        if (!g_vq_partial) { (void)cudaGetLastError(); return 0; }
        vq_moe_down_fused_kernel<<<dim3(n_tokens, zd, n_expert), 32 * DS4_VQ_WARPS_PER_BLOCK,
            (size_t)DS4_VQ_WARPS_PER_BLOCK * DS4_VQ_BITWORDS * sizeof(uint32_t)
            + 2048u * sizeof(__half)>>>(
            g_vq_partial, blob, (const int32_t *)selected->ptr,
            (const float *)weights->ptr, (const float *)mid_scratch->ptr,
            down_base, down_expert_bytes, n_expert, expert_mid_dim, out_dim);
        vq2_down_reduce_kernel<<<dim3((out_dim + 255u) / 256u, n_tokens, 1), 256>>>(
            (float *)out->ptr, g_vq_partial, (const int32_t *)selected->ptr,
            (const float *)weights->ptr, n_expert, out_dim);
        if (((const char *)0) /* DS4_VQ_ROUTE_DIAG: 路径开关已删(2026-08-22 隐形炸弹清理) */ && layer_index == (uint32_t)atoi(((const char *)0) /* DS4_VQ_ROUTE_DIAG: 路径开关已删(2026-08-22 隐形炸弹清理) */)) {
            (void)cudaDeviceSynchronize();
            FILE *fo = fopen("/tmp/vq_out.bin", "wb");
            if (fo) { float *ob = (float *)malloc(4096 * 4); cudaMemcpy(ob, out->ptr, 4096 * 4, cudaMemcpyDeviceToHost); fwrite(ob, 4, 4096, fo); fclose(fo); free(ob); }
        }
        /* DS4_VQ_PROF=1: 逐层实测两个 kernel 的墙钟耗时。nsys 显示 gateup 的
         * Max/Med = 10x、StdDev≈均值 —— 正常算力负载不会这样, 需要定位是哪些层/
         * 哪种输入触发了长尾。计时本身要同步, 只在开关打开时付这个代价。 */
        if (((const char *)0) /* DS4_VQ_PROF: 路径开关已删(2026-08-22 隐形炸弹清理) */) {
            (void)cudaDeviceSynchronize();
            const double t1 = cuda_wall_sec();
            static double acc_ms = 0.0; static uint64_t nfwd = 0;
            acc_ms += (t1 - prof_t0) * 1000.0; nfwd++;
            fprintf(stderr, "[vqprof] L%-2u ntok=%u npair=%llu  %.3f ms  (累计 %.1f ms / %llu 次)\n",
                    layer_index, n_tokens, (unsigned long long)npair,
                    (t1 - prof_t0) * 1000.0, acc_ms, (unsigned long long)nfwd);
        }
        if (((const char *)0) /* DS4_VQ_CYC: 路径开关已删(2026-08-22 隐形炸弹清理) */ && layer_index == 42u) {
            (void)cudaDeviceSynchronize();
            unsigned long long c[3] = {0, 0, 0};
            (void)cudaMemcpyFromSymbol(c, g_vq_cyc, sizeof(c));
            if (c[2]) {
                const double tot = (double)(c[0] + c[1]);
                fprintf(stderr, "[vqcyc] 样本 %llu | 位流预取 %.0f cy/行 (%.1f%%) | 解码+乘加 %.0f cy/行 (%.1f%%)\n",
                        c[2], (double)c[0] / c[2], 100.0 * c[0] / tot,
                        (double)c[1] / c[2], 100.0 * c[1] / tot);
            }
        }
        if (((const char *)0) /* DS4_VQ_DEBUG: 诊断开关已删(2026-08-22) */)
            fprintf(stderr, "ds4: [cuda-vq-fuse] L%u ntok=%u npair=%llu (零 dequant scratch)\n",
                    layer_index, n_tokens, (unsigned long long)npair);
        return cuda_ok(cudaGetLastError(), "vq fused launch");
    }

    int32_t *sel_h = (int32_t *)malloc(npair * sizeof(int32_t));
    if (!sel_h) return 0;
    if (cudaMemcpy(sel_h, (const char *)selected->ptr, npair * sizeof(int32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void)cudaGetLastError(); free(sel_h); return 0;
    }

    /* 活跃专家去重 + 建 id→slot 映射, 然后把 selected 原地 remap 成 slot。
     * gather 后权重按 slot 紧凑排布, kernel 只认 slot。 */
    int32_t *e2slot = (int32_t *)malloc((size_t)n_total_expert * sizeof(int32_t));
    uint32_t *active_ids = (uint32_t *)malloc((size_t)n_total_expert * sizeof(uint32_t));
    if (!e2slot || !active_ids) { free(sel_h); free(e2slot); free(active_ids); return 0; }
    for (uint32_t e = 0; e < n_total_expert; e++) e2slot[e] = -1;
    uint32_t n_active = 0;
    for (uint64_t k = 0; k < npair; k++) {
        const int32_t e = sel_h[k];
        if (e < 0 || (uint32_t)e >= n_total_expert) { sel_h[k] = -1; continue; }
        if (e2slot[e] < 0) { e2slot[e] = (int32_t)n_active; active_ids[n_active++] = (uint32_t)e; }
        sel_h[k] = e2slot[e];
    }
    if (n_active == 0) { free(sel_h); free(e2slot); free(active_ids); return 1; }   /* 无活跃专家: out 保持不变 */

    /* decode 缓存: 把本层需要的专家映射到常驻槽, 命中的免 dequant。
     * need_dq[] 收集真正要解码的 (slot, expert) 对; 未命中才进 dequant 循环。 */
    /* ★默认关闭★ — A/B 实测(2026-08-17, 同 prompt/同二进制):
     *     关: prefill 1.03 / gen 4.06 t/s      开: prefill 0.66 / gen 0.82 t/s
     * 命中率只有 43%(每层 6 个专家仍要 dequant 3.5 个), 而 per-layer 缓存把访存从
     * 集中的 1.4GB scratch 摊成分散的 16.5GB, L2 局部性塌掉 —— 省下的 dequant 远
     * 抵不过多出来的访存。留着开关是因为换配方(更大 K / 路由更集中)时值得再量一次,
     * 但默认路径不该为它买单。 */
    int cache_on = (n_tokens == 1u) && (layer_index < DS4_VQ_CACHE_LAYERS) &&
                         (n_active <= DS4_VQ_CACHE_SLOTS) &&
                         (((const char *)0) /* DS4_VQ_CACHE: 路径开关已删(2026-08-22 隐形炸弹清理) */ && ((const char *)0) /* DS4_VQ_CACHE: 路径开关已删(2026-08-22 隐形炸弹清理) */[0] == '1');
    uint32_t n_slot_total = n_active;          /* scratch 需要容纳的槽数 */
    uint32_t *dq_slot = (uint32_t *)malloc((size_t)n_active * sizeof(uint32_t));
    uint32_t *dq_expert = (uint32_t *)malloc((size_t)n_active * sizeof(uint32_t));
    uint32_t n_dq = 0;
    if (!dq_slot || !dq_expert) { free(sel_h); free(e2slot); free(active_ids); free(dq_slot); free(dq_expert); return 0; }

    if (!g_vq_cache_ready) cuda_vq_cache_reset_all();
    {   /* 显存不足时静默降级为直通, 正确性不受影响 */
        const uint64_t ge0 = (uint64_t)expert_mid_dim * expert_in_dim * 2u;
        const uint64_t de0 = (uint64_t)out_dim * expert_mid_dim * 2u;
        if (cache_on && !cuda_vq_cache_ensure_mem(layer_index, ge0, de0)) cache_on = 0;
    }
    if (cache_on) {
        n_slot_total = DS4_VQ_CACHE_SLOTS;
        int32_t *a2cache = (int32_t *)malloc((size_t)n_active * sizeof(int32_t));
        if (!a2cache) { free(sel_h); free(e2slot); free(active_ids); free(dq_slot); free(dq_expert); return 0; }
        for (uint32_t i = 0; i < n_active; i++) {
            const uint32_t e = active_ids[i];
            int32_t hit = -1;
            for (uint32_t s = 0; s < DS4_VQ_CACHE_SLOTS; s++)
                if (g_vq_cache[layer_index].expert[s] == (int32_t)e) { hit = (int32_t)s; break; }
            if (hit < 0) {
                /* 选空槽, 没有则 LRU。本轮已占用的槽不可再被淘汰。 */
                int32_t victim = -1; uint64_t oldest = ~0ull;
                for (uint32_t s = 0; s < DS4_VQ_CACHE_SLOTS; s++) {
                    int taken = 0;
                    for (uint32_t j = 0; j < i; j++) if (a2cache[j] == (int32_t)s) { taken = 1; break; }
                    if (taken) continue;
                    if (g_vq_cache[layer_index].expert[s] < 0) { victim = (int32_t)s; break; }
                    if (g_vq_cache[layer_index].used[s] < oldest) { oldest = g_vq_cache[layer_index].used[s]; victim = (int32_t)s; }
                }
                if (victim < 0) { victim = (int32_t)i; }   /* 兜底(不应发生: n_active<=SLOTS) */
                g_vq_cache[layer_index].expert[victim] = (int32_t)e;
                dq_slot[n_dq] = (uint32_t)victim; dq_expert[n_dq] = e; n_dq++;
                hit = victim;
                g_vq_miss++;
            } else {
                g_vq_hit++;
            }
            a2cache[i] = hit;
            g_vq_cache[layer_index].used[hit] = ++g_vq_clock;
        }
        /* sel_h 里现在是"紧凑 active 序号", 改写成缓存槽号 */
        for (uint64_t k = 0; k < npair; k++)
            if (sel_h[k] >= 0) sel_h[k] = a2cache[sel_h[k]];
        free(a2cache);
    } else {
        /* 直通: 顺序占槽, 并让本层缓存失效(scratch 即将被覆盖) */
        if (layer_index < DS4_VQ_CACHE_LAYERS && g_vq_cache_ready)
            for (uint32_t s = 0; s < DS4_VQ_CACHE_SLOTS; s++) g_vq_cache[layer_index].expert[s] = -1;
        for (uint32_t i = 0; i < n_active; i++) { dq_slot[i] = i; dq_expert[i] = active_ids[i]; }
        n_dq = n_active;
    }
    free(e2slot);

    const uint64_t ge = (uint64_t)expert_mid_dim * expert_in_dim * 2u;
    const uint64_t de = (uint64_t)out_dim * expert_mid_dim * 2u;
    const uint64_t need_gu = (uint64_t)n_slot_total * ge, need_dn = (uint64_t)n_slot_total * de;
    /* scratch 上限护栏(与 Metal 的 DS4_VQ_SCRATCH_GB 同名同义)。prefill 大 chunk
     * 下活跃专家可能逼近 256 个 ⇒ 不设闸会直接吃光统一内存。 */
    static uint64_t vq_cap = 0;
    if (!vq_cap) { const char *e = ((const char *)0) /* DS4_VQ_SCRATCH_GB: 路径开关已删(2026-08-22 隐形炸弹清理) */; double g = e ? atof(e) : 32.0; vq_cap = (uint64_t)(g * 1073741824.0); }
    if (2 * need_gu + need_dn > vq_cap) {
        fprintf(stderr, "ds4: VQ gather scratch %.2fGB > cap (降 prefill chunk 或调 DS4_VQ_SCRATCH_GB)\n",
                (2.0 * need_gu + need_dn) / 1073741824.0);
        free(sel_h); free(active_ids); return 0;
    }
    if (!cache_on && !cuda_vq_ensure_scratch(need_gu, need_dn)) {
        fprintf(stderr, "ds4: [cuda-vq] scratch 分配失败 L%u n_active=%u 需要 %.2fGB\n",
                layer_index, n_active, (2.0 * need_gu + need_dn) / 1073741824.0);
        free(sel_h); free(active_ids); free(dq_slot); free(dq_expert); return 0;
    }

    /* 只有 CPU gather 路才需要全设备同步(CPU 要写跨层复用的托管 scratch, 而上一层
     * kernel 可能仍在读它 —— 托管内存被 CPU 写就是段错误, 不止是数据竞争; 这正是本
     * 实现第一版 SIGSEGV 的原因)。GPU dequant 路全在同一 stream 上顺序执行, 天然有序,
     * 多同步一次就是把 43 层的流水线全打断。 */
    {
        const char *cg0 = ((const char *)0) /* DS4_VQ_CPU_GATHER: 路径开关已删(2026-08-22 隐形炸弹清理) */;
        if (cg0 && cg0[0] == '1' && cudaDeviceSynchronize() != cudaSuccess) {
            (void)cudaGetLastError(); free(sel_h); free(active_ids); return 0;
        }
    }

    uint16_t *gbase, *ubase, *dbase;
    if (cache_on) {   /* 本层专属缓存显存 */
        gbase = (uint16_t *)g_vq_cache[layer_index].gate;
        ubase = (uint16_t *)g_vq_cache[layer_index].up;
        dbase = (uint16_t *)g_vq_cache[layer_index].down;
    } else {          /* 共享临时(prefill) */
        gbase = (uint16_t *)g_vq_gate_sc; ubase = (uint16_t *)g_vq_up_sc; dbase = (uint16_t *)g_vq_down_sc;
    }
    int nth = 0;
    /* DS4_VQ_CPU_GATHER=1 回 CPU 多线程 dequant(参考实现, 慢 ~10×), 用于对拍。 */
    const char *cg = ((const char *)0) /* DS4_VQ_CPU_GATHER: 路径开关已删(2026-08-22 隐形炸弹清理) */;
    if (cg && cg[0] == '1') {
        if (cache_on) {   /* CPU gather 是对拍用的参考路, 不参与缓存 */
            fprintf(stderr, "ds4: DS4_VQ_CPU_GATHER 与 decode 缓存不兼容, 请同时设 DS4_VQ_NO_CACHE=1\n");
            free(sel_h); free(active_ids); free(dq_slot); free(dq_expert); return 0;
        }
        nth = 8;
        { const char *te = ((const char *)0) /* DS4_METAL_EXPERT_GATHER_THREADS: 路径开关已删(2026-08-22 隐形炸弹清理) */; if (te && atoi(te) > 0) nth = atoi(te); }
        if ((uint32_t)nth > n_active) nth = (int)n_active;
        if (nth > 32) nth = 32;
        if (nth < 1) nth = 1;
        volatile int gerr = 0;
        pthread_t th[32]; cuda_vq_gather_task tk[32];
        for (int i = 0; i < nth; i++) {
            tk[i].model_map = model_map; tk[i].blob = blob_host; tk[i].active_ids = active_ids;
            tk[i].gbase = gbase; tk[i].ubase = ubase; tk[i].dbase = dbase;
            tk[i].down_offset = down_offset; tk[i].down_expert_bytes = down_expert_bytes;
            tk[i].in = expert_in_dim; tk[i].mid = expert_mid_dim; tk[i].out_dim = out_dim;
            tk[i].lo = (uint32_t)((uint64_t)i * n_active / nth);
            tk[i].hi = (uint32_t)((uint64_t)(i + 1) * n_active / nth);
            tk[i].err = &gerr;
            pthread_create(&th[i], NULL, cuda_vq_gather_worker, &tk[i]);
        }
        for (int i = 0; i < nth; i++) pthread_join(th[i], NULL);
        if (gerr) { free(sel_h); free(active_ids); return 0; }
    } else {
        /* GPU dequant: 只解码缓存未命中的专家(直通模式下就是全部)。每矩阵一次 launch。 */
        for (uint32_t q = 0; q < n_dq; q++) {
            const uint32_t i = dq_slot[q];
            const uint32_t e = dq_expert[q];
            const uint64_t o1 = ds4vq_slot(blob_host, (int)e, 0);
            const uint64_t o3 = ds4vq_slot(blob_host, (int)e, 1);
            const uint64_t o2 = ds4vq_slot(blob_host, (int)e, 2);
            uint32_t rr, cc, dd_, nn, nb;
            if (!o1 || !o3 ||
                cuda_vq_pay_hdr(blob_host + o1, (int)expert_mid_dim, (int)expert_in_dim, &rr, &cc, &dd_, &nn, &nb) != 0) {
                fprintf(stderr, "ds4: [cuda-vq] e=%u w1/w3 槽缺失或头错 -- aborting\n", e);
                free(sel_h); free(active_ids); free(dq_slot); free(dq_expert); return 0;
            }
            vq_dequant_kernel<<<rr, 256>>>((__half *)(gbase + (uint64_t)i * expert_mid_dim * expert_in_dim),
                                           blob + o1, rr, cc, dd_, nn, nb);
            if (cuda_vq_pay_hdr(blob_host + o3, (int)expert_mid_dim, (int)expert_in_dim, &rr, &cc, &dd_, &nn, &nb) != 0) {
                fprintf(stderr, "ds4: [cuda-vq] e=%u w3 头错 -- aborting\n", e);
                free(sel_h); free(active_ids); free(dq_slot); free(dq_expert); return 0;
            }
            vq_dequant_kernel<<<rr, 256>>>((__half *)(ubase + (uint64_t)i * expert_mid_dim * expert_in_dim),
                                           blob + o3, rr, cc, dd_, nn, nb);
            __half *dst_d = (__half *)(dbase + (uint64_t)i * out_dim * expert_mid_dim);
            if (o2) {
                if (cuda_vq_pay_hdr(blob_host + o2, (int)out_dim, (int)expert_mid_dim, &rr, &cc, &dd_, &nn, &nb) != 0) {
                    fprintf(stderr, "ds4: [cuda-vq] e=%u w2 头错 -- aborting\n", e);
                    free(sel_h); free(active_ids); free(dq_slot); free(dq_expert); return 0;
                }
                vq_dequant_kernel<<<rr, 256>>>(dst_d, blob + o2, rr, cc, dd_, nn, nb);
            } else {
                /* 冷 w2 回退: base down 是影子张量时硬失败, 不读垃圾当权重(同 CPU 路)。 */
                if (down_expert_bytes == 0 || down_offset == 0) {
                    fprintf(stderr, "ds4: [cuda-vq] e=%u 冷 w2 槽缺失且 base down 不在文件里"
                                    "(影子张量) -- aborting (no silent quality downgrade)\n", e);
                    free(sel_h); free(active_ids); free(dq_slot); free(dq_expert); return 0;
                }
                const uint8_t *sd = (const uint8_t *)model_map + down_offset + (uint64_t)e * down_expert_bytes;
                vq_cold_w2_kernel<<<out_dim, 256>>>(dst_d, sd, out_dim, expert_mid_dim);
            }
        }
        if (!cuda_ok(cudaGetLastError(), "vq_dequant launch")) { free(sel_h); free(active_ids); free(dq_slot); free(dq_expert); return 0; }
    }
    free(active_ids); free(dq_slot); free(dq_expert);

    const char *vg = ((const char *)0) /* DS4_VQ_GPU: 路径开关已删(2026-08-22 隐形炸弹清理) */;
    const int use_gpu = !(vg && vg[0] == '0');
    int ok = 0;
    if (!use_gpu) {
        /* CPU 参考路读的是 GPU 刚写完的 scratch ⇒ 必须先同步 */
        if (cudaDeviceSynchronize() != cudaSuccess) { (void)cudaGetLastError(); free(sel_h); return 0; }
        /* CPU 参考路: x/out 需在主机侧 */
        float *xh = (float *)malloc((size_t)n_tokens * expert_in_dim * sizeof(float));
        float *wh = (float *)malloc((size_t)npair * sizeof(float));
        float *oh = (float *)malloc((size_t)n_tokens * out_dim * sizeof(float));
        if (xh && wh && oh &&
            cudaMemcpy(xh, x->ptr, (size_t)n_tokens * expert_in_dim * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(wh, weights->ptr, (size_t)npair * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess) {
            ok = cuda_vq_moe_cpu_ref(oh, gbase, ubase, dbase, xh, sel_h, wh,
                                     n_tokens, n_active, n_expert,
                                     expert_in_dim, expert_mid_dim, out_dim, clamp);
            if (ok) ok = (cudaMemcpy(out->ptr, oh, (size_t)n_tokens * out_dim * sizeof(float),
                                     cudaMemcpyHostToDevice) == cudaSuccess);
        }
        free(xh); free(wh); free(oh); free(sel_h);
        if (!ok) (void)cudaGetLastError();
        return ok;
    }

    /* GPU: remap 后的 slot 索引上传设备 */
    if (npair * sizeof(int32_t) > g_vq_sel_bytes) {
        if (g_vq_sel_dev) (void)cudaFree(g_vq_sel_dev);
        g_vq_sel_dev = NULL; g_vq_sel_bytes = 0;
        if (cudaMalloc((void **)&g_vq_sel_dev, npair * sizeof(int32_t)) != cudaSuccess) {
            (void)cudaGetLastError(); free(sel_h); return 0;
        }
        g_vq_sel_bytes = npair * sizeof(int32_t);
    }
    ok = (cudaMemcpy(g_vq_sel_dev, sel_h, npair * sizeof(int32_t), cudaMemcpyHostToDevice) == cudaSuccess);
    free(sel_h);
    if (!ok) { (void)cudaGetLastError(); return 0; }

    /* out 累加语义 ⇒ 先清零(kernel 用 atomicAdd 汇多个 pick) */
    if (cudaMemset(out->ptr, 0, (size_t)n_tokens * out_dim * sizeof(float)) != cudaSuccess) {
        (void)cudaGetLastError(); return 0;
    }
    /* 中间 h[n_tokens][n_expert][MID]: 复用调用方给的 mid scratch(容量口径与
     * routed_moe_launch 一致) */
    const uint64_t hneed = (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float);
    if (!mid_scratch || mid_scratch->bytes < hneed) {
        fprintf(stderr, "ds4: [cuda-vq] mid scratch %llu < 需要 %llu\n",
                (unsigned long long)(mid_scratch ? mid_scratch->bytes : 0), (unsigned long long)hneed);
        return 0;
    }
    const uint32_t zg = (expert_mid_dim + DS4_VQ_WARPS_PER_BLOCK - 1u) / DS4_VQ_WARPS_PER_BLOCK;
    const uint32_t zd = (out_dim + DS4_VQ_WARPS_PER_BLOCK - 1u) / DS4_VQ_WARPS_PER_BLOCK;
    vq_moe_gateup_kernel<<<dim3(n_tokens, n_expert, zg), 32 * DS4_VQ_WARPS_PER_BLOCK>>>(
        (float *)mid_scratch->ptr, (const __half *)gbase, (const __half *)ubase,
        (const float *)x->ptr, g_vq_sel_dev, (const float *)weights->ptr, n_expert, expert_in_dim, expert_mid_dim, clamp);
    vq_moe_down_kernel<<<dim3(n_tokens, 1, zd), 32 * DS4_VQ_WARPS_PER_BLOCK>>>(
        (float *)out->ptr, (const __half *)dbase, (const float *)mid_scratch->ptr,
        g_vq_sel_dev, (const float *)weights->ptr, n_expert, expert_mid_dim, out_dim);
    if (((const char *)0) /* DS4_VQ_ROUTE_DIAG: 路径开关已删(2026-08-22 隐形炸弹清理) */ && layer_index == (uint32_t)atoi(((const char *)0) /* DS4_VQ_ROUTE_DIAG: 路径开关已删(2026-08-22 隐形炸弹清理) */) && n_tokens > 1) {
        (void)cudaDeviceSynchronize();
        FILE *f;
        float *tb = (float *)malloc((size_t)expert_in_dim * 4);
        if ((f = fopen("/tmp/vq_px.bin", "wb"))) { cudaMemcpy(tb, (const char *)x->ptr + (size_t)(n_tokens - 1) * expert_in_dim * 4, (size_t)expert_in_dim * 4, cudaMemcpyDeviceToHost); fwrite(tb, 4, expert_in_dim, f); fclose(f); }
        if ((f = fopen("/tmp/vq_pout.bin", "wb"))) { cudaMemcpy(tb, (const char *)out->ptr + (size_t)(n_tokens - 1) * out_dim * 4, (size_t)out_dim * 4, cudaMemcpyDeviceToHost); fwrite(tb, 4, out_dim, f); fclose(f); }
        free(tb);
    }
    if (((const char *)0) /* DS4_VQ_DEBUG: 诊断开关已删(2026-08-22) */)
        fprintf(stderr, "ds4: [cuda-vq] L%u ntok=%u n_active=%u dq=%u cache=%d hit=%llu miss=%llu scratch=%.2fGB\n",
                layer_index, n_tokens, n_active, n_dq, cache_on,
                (unsigned long long)g_vq_hit, (unsigned long long)g_vq_miss,
                (2.0 * need_gu + need_dn) / 1073741824.0);
    return cuda_ok(cudaGetLastError(), "vq_moe_pair launch");
}

int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const ds4_gpu_residual_set *residual, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index) {
    if (residual && residual->vq)
        return cuda_vq_moe_forward(out, mid, residual, model_map, down_offset, down_expert_bytes,
                                   expert_in_dim, expert_mid_dim, out_dim, selected, weights,
                                   n_total_expert, n_expert, clamp, x, layer_index, 1u);
    (void)layer_index; (void)residual;  /* go1b 残差(非 VQ): CUDA 未实现, 见 ds4_gpu.h */
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x, 1);
}
/* ★参数表必须与 ds4_gpu.h 逐字一致★ —— 本文件不 include 那个头(GPU 句柄类型两边
 * 各自实现), 编译器无从校验。此处曾少了 slot_start/slot_count 两个参数: ds4.c 按
 * 头文件压 25 个实参, 这里按 23 个取, mid_is_f16 收到的其实是 slot_start 的值 ⇒
 * 解引用垃圾指针段错误。以前没暴露只是因为本文件根本编译不过(缺 residual_set 定义)。 */
int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const ds4_gpu_residual_set *residual, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, uint32_t slot_start, uint32_t slot_count, bool *mid_is_f16) {
    if (mid_is_f16) *mid_is_f16 = false;
    /* TP Phase-3 专家切分是双机拆模型用的; CUDA 走单机整模型, 没有对端可 all-reduce。
     * 非平凡切分直接拒绝, 不能只算半边专家却当成完整输出。 */
    if (slot_count != 0u && slot_count != n_expert) {
        fprintf(stderr, "ds4: CUDA 不支持 TP 专家切分 (slot_start=%u slot_count=%u n_expert=%u)\n",
                slot_start, slot_count, n_expert);
        return 0;
    }
    (void)slot_start;
    if (residual && residual->vq)
        return cuda_vq_moe_forward(out, mid, residual, model_map, down_offset, down_expert_bytes,
                                   expert_in_dim, expert_mid_dim, out_dim, selected, weights,
                                   n_total_expert, n_expert, clamp, x, layer_index, n_tokens);
    (void)layer_index; (void)residual;  /* go1b 残差(非 VQ): CUDA 未实现, 见 ds4_gpu.h */
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x, n_tokens);
}
int ds4_gpu_hc_split_sinkhorn_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *mix, const void *model_map, uint64_t model_size, uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, uint32_t sinkhorn_iters, float eps) {
    if (!out || !mix || !model_map || n_hc != 4) return 0;
    const uint64_t mix_bytes = 24ull * sizeof(float);
    if (scale_offset > model_size || model_size - scale_offset < 3ull * sizeof(float) ||
        base_offset > model_size || model_size - base_offset < mix_bytes ||
        mix->bytes < mix_bytes || out->bytes < mix_bytes) return 0;
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, 3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, mix_bytes, "hc_base");
    if (!scale || !base) return 0;
    uint32_t n_rows = (uint32_t)(mix->bytes / mix_bytes);
    if (out->bytes / mix_bytes < n_rows) n_rows = (uint32_t)(out->bytes / mix_bytes);
    hc_split_sinkhorn_kernel<<<(n_rows + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)mix->ptr,
        scale,
        base,
        n_rows, sinkhorn_iters, eps);
    return cuda_ok(cudaGetLastError(), "hc_split_sinkhorn launch");
}
int ds4_gpu_hc_weighted_sum_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *weights, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !residual_hc || !weights || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out->bytes / ((uint64_t)n_embd * sizeof(float)));
    hc_weighted_sum_kernel<<<((uint64_t)n_embd * n_tokens + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)residual_hc->ptr, (const float *)weights->ptr,
        n_embd, n_hc, n_tokens, n_hc);
    return cuda_ok(cudaGetLastError(), "hc_weighted_sum launch");
}
int ds4_gpu_hc_weighted_sum_split_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out->bytes / ((uint64_t)n_embd * sizeof(float)));
    uint32_t stride = (uint32_t)(2u * n_hc + n_hc * n_hc);
    hc_weighted_sum_kernel<<<((uint64_t)n_embd * n_tokens + 255) / 256, 256>>>(
        (float *)out->ptr, (const float *)residual_hc->ptr, (const float *)split->ptr,
        n_embd, n_hc, n_tokens, stride);
    return cuda_ok(cudaGetLastError(), "hc_weighted_sum_split launch");
}
int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps) {
    if (!out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc != 4) {
        return 0;
    }
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t mix_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
        scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || mix_bytes > model_size - base_offset) {
        return 0;
    }
    uint64_t n_rows = out->bytes / out_row_bytes;
    if (mix->bytes < n_rows * mix_bytes ||
        split->bytes < n_rows * mix_bytes ||
        residual_hc->bytes < n_rows * residual_row_bytes) {
        return 0;
    }
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, 3ull * sizeof(float), "hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, mix_bytes, "hc_base");
    if (!scale || !base) return 0;
    hc_split_weighted_sum_fused_kernel<<<(uint32_t)n_rows, 256>>>(
            (float *)out->ptr,
            (float *)split->ptr,
            (const float *)mix->ptr,
            (const float *)residual_hc->ptr,
            scale,
            base,
            n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps);
    return cuda_ok(cudaGetLastError(), "hc split weighted sum launch");
}
int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps) {
    if (1) {
        if (!out || !norm_out || !split || !mix || !residual_hc || !model_map ||
            n_embd == 0 || n_hc != 4) {
            return 0;
        }
        const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
        const uint64_t mix_bytes = mix_hc * sizeof(float);
        const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
        const uint64_t residual_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
        if (out->bytes < out_row_bytes || out->bytes % out_row_bytes != 0 ||
            norm_out->bytes < out->bytes ||
            scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
            base_offset > model_size || mix_bytes > model_size - base_offset ||
            norm_weight_offset > model_size ||
            (uint64_t)n_embd * sizeof(float) > model_size - norm_weight_offset) {
            return 0;
        }
        uint64_t n_rows = out->bytes / out_row_bytes;
        if (n_rows == 1) {
            if (mix->bytes < n_rows * mix_bytes ||
                split->bytes < n_rows * mix_bytes ||
                residual_hc->bytes < n_rows * residual_row_bytes) {
                return 0;
            }
            const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset,
                    3ull * sizeof(float), "hc_scale");
            const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset,
                    mix_bytes, "hc_base");
            const float *norm_w = (const float *)cuda_model_range_ptr(model_map, norm_weight_offset,
                    (uint64_t)n_embd * sizeof(float), "hc_norm_weight");
            if (!scale || !base || !norm_w) return 0;
            if (n_hc == 4 && (n_embd & 3u) == 0) {
                hc_split_wsn_fast_kernel<<<(uint32_t)n_rows, 1024>>>(
                        (float *)out->ptr,
                        (float *)norm_out->ptr,
                        (float *)split->ptr,
                        (const float *)mix->ptr,
                        (const float *)residual_hc->ptr,
                        scale,
                        base,
                        norm_w,
                        n_embd, (uint32_t)n_rows, sinkhorn_iters, eps, norm_eps);
            } else {
                hc_split_weighted_sum_norm_fused_kernel<<<(uint32_t)n_rows, 256>>>(
                        (float *)out->ptr,
                        (float *)norm_out->ptr,
                        (float *)split->ptr,
                        (const float *)mix->ptr,
                        (const float *)residual_hc->ptr,
                        scale,
                        base,
                        norm_w,
                        n_embd, n_hc, (uint32_t)n_rows, sinkhorn_iters, eps, norm_eps);
            }
            return cuda_ok(cudaGetLastError(), "hc split weighted sum norm launch");
        }
    }
    return ds4_gpu_hc_split_weighted_sum_tensor(out, split, mix, residual_hc,
                                                  model_map, model_size,
                                                  scale_offset, base_offset,
                                                  n_embd, n_hc,
                                                  sinkhorn_iters, eps) &&
           ds4_gpu_rms_norm_weight_tensor(norm_out, out, model_map, model_size,
                                            norm_weight_offset, n_embd, norm_eps);
}
int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps) {
    if (!out || !pre || !model_map || n_hc == 0) return 0;
    const uint64_t row_bytes = (uint64_t)n_hc * sizeof(float);
    if (row_bytes == 0 || out->bytes < row_bytes || out->bytes % row_bytes != 0 ||
        pre->bytes < out->bytes ||
        scale_offset > model_size || sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || row_bytes > model_size - base_offset) {
        return 0;
    }
    const uint64_t n_tokens = out->bytes / row_bytes;
    const float *scale = (const float *)cuda_model_range_ptr(model_map, scale_offset, sizeof(float), "output_hc_scale");
    const float *base = (const float *)cuda_model_range_ptr(model_map, base_offset, row_bytes, "output_hc_base");
    if (!scale || !base) return 0;
    uint64_t n = n_tokens * n_hc;
    output_hc_weights_kernel<<<(n + 255) / 256, 256>>>(
            (float *)out->ptr,
            (const float *)pre->ptr,
            scale,
            base,
            n_hc,
            (uint32_t)n_tokens,
            eps);
    return cuda_ok(cudaGetLastError(), "output hc weights launch");
}
int ds4_gpu_hc_expand_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !post || !comb || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    (const float *)post->ptr,
                                                    (const float *)comb->ptr,
                                                    n_embd, n_hc, n_tokens,
                                                    n_hc, n_hc * n_hc, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand launch");
}
int ds4_gpu_hc_expand_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 0);
    return cuda_ok(cudaGetLastError(), "hc_expand_split launch");
}
int ds4_gpu_hc_expand_add_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add, const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !block_out || !block_add || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = (uint32_t)(out_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)));
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    uint64_t n_elem = (uint64_t)n_tokens * n_hc * n_embd;
    const float *base = (const float *)split->ptr;
    hc_expand_kernel<<<(n_elem + 255) / 256, 256>>>((float *)out_hc->ptr,
                                                    (const float *)block_out->ptr,
                                                    (const float *)block_add->ptr,
                                                    (const float *)residual_hc->ptr,
                                                    base + n_hc,
                                                    base + 2u * n_hc,
                                                    n_embd, n_hc, n_tokens,
                                                    mix_hc, mix_hc, 1);
    return cuda_ok(cudaGetLastError(), "hc_expand_add_split launch");
}
int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        const ds4_gpu_tensor *corr_delta,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    /* no CUDA corr-delta consumer: callers gate on corr_delta_supported()==0 */
    if (corr_delta) return 0;
    if (1) {
        return cuda_matmul_q8_0_hc_expand_tensor_labeled(out_hc, shared_out,
                                                        model_map, model_size,
                                                        weight_offset,
                                                        in_dim, out_dim,
                                                        shared_mid,
                                                        routed_out,
                                                        residual_hc,
                                                        split,
                                                        n_embd, n_hc,
                                                        "shared_down_hc_expand");
    }
    return ds4_gpu_matmul_q8_0_tensor(shared_out, model_map, model_size,
                                        weight_offset, in_dim, out_dim,
                                        shared_mid, 1) &&
           ds4_gpu_hc_expand_add_split_tensor(out_hc, shared_out, routed_out,
                                                residual_hc, split, n_embd, n_hc);
}

int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (1) {
        return cuda_matmul_q8_0_hc_expand_tensor_labeled(out_hc, block_out,
                                                        model_map, model_size,
                                                        weight_offset,
                                                        in_dim, out_dim,
                                                        x,
                                                        NULL,
                                                        residual_hc,
                                                        split,
                                                        n_embd, n_hc,
                                                        "q8_hc_expand");
    }
    return ds4_gpu_matmul_q8_0_tensor(block_out, model_map, model_size,
                                        weight_offset, in_dim, out_dim, x, 1) &&
           ds4_gpu_hc_expand_split_tensor(out_hc, block_out, residual_hc,
                                            split, n_embd, n_hc);
}

/* ===== go-onebit DQZ2 zchain (CUDA) =====
 * Metal kernel_dsv4_zchain_{ge,scale} 的逐语义平移(数学契约: ds4_zchain.h)。
 * 常驻小表一次上传; dispatch 纯 kernel launch, token-graph capture 兼容。 */
static float    *g_zc_ops = NULL;        /* dev [n_ops_total][16] */
static __half   *g_zc_v8  = NULL;        /* dev concat fp16 [n_blk][8][d_model] */
static float    *g_zc_ge  = NULL;        /* dev [n_layer][n_expert] */
static uint32_t *g_zc_layer_off = NULL;  /* host [n_layer+1] */
static uint8_t  *g_zc_ge_present = NULL; /* host [n_layer] */
static uint32_t  g_zc_n_layer = 0, g_zc_n_expert = 0, g_zc_d_model = 0;
static __half   *g_zc_zlm = NULL;        /* dev packed z[k]|U[d*k]|V[din*k] per layer */
static uint32_t *g_zc_zl_off = NULL, *g_zc_zl_k = NULL, *g_zc_zl_din = NULL; /* host */
static float    *g_zc_zl_tr = NULL;      /* host */

int ds4_gpu_zchain_set(
        const float *ops, const uint32_t *layer_off, const uint16_t *v8,
        const float *ge, const uint8_t *ge_present,
        uint32_t n_layer, uint32_t n_expert, uint32_t d_model,
        uint32_t n_ops_total, uint32_t n_v8_blocks) {
    g_zc_n_layer = n_layer; g_zc_n_expert = n_expert; g_zc_d_model = d_model;
    if (n_ops_total) {
        if (!cuda_ok(cudaMalloc(&g_zc_ops, (size_t)n_ops_total * 16 * sizeof(float)), "zchain ops")) return 0;
        if (!cuda_ok(cudaMemcpy(g_zc_ops, ops, (size_t)n_ops_total * 16 * sizeof(float), cudaMemcpyHostToDevice), "zchain ops up")) return 0;
    }
    if (n_v8_blocks) {
        size_t hb = (size_t)n_v8_blocks * 8 * d_model * sizeof(__half);
        if (!cuda_ok(cudaMalloc(&g_zc_v8, hb), "zchain v8")) return 0;
        if (!cuda_ok(cudaMemcpy(g_zc_v8, v8, hb, cudaMemcpyHostToDevice), "zchain v8 up")) return 0;
    }
    if (ge && ge_present) {
        size_t gb = (size_t)n_layer * n_expert * sizeof(float);
        if (!cuda_ok(cudaMalloc(&g_zc_ge, gb), "zchain ge")) return 0;
        if (!cuda_ok(cudaMemcpy(g_zc_ge, ge, gb, cudaMemcpyHostToDevice), "zchain ge up")) return 0;
        g_zc_ge_present = (uint8_t *)malloc(n_layer);
        memcpy(g_zc_ge_present, ge_present, n_layer);
    }
    g_zc_layer_off = (uint32_t *)malloc((n_layer + 1) * sizeof(uint32_t));
    memcpy(g_zc_layer_off, layer_off, (n_layer + 1) * sizeof(uint32_t));
    fprintf(stderr, "ds4: zchain CUDA armed: %u ops, %u v8 blocks, GE=%s\n",
            n_ops_total, n_v8_blocks, g_zc_ge ? "yes" : "no");
    return 1;
}

static __global__ void zchain_ge_kernel(
        float *weights, const int *selected, const float *ge,
        uint32_t n_expert, uint32_t total, uint32_t ge_base) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    int e = selected[gid];
    if (e < 0 || (uint32_t)e >= n_expert) return;
    weights[gid] *= ge[ge_base + (uint32_t)e];
}

int ds4_gpu_zchain_ge_apply(
        ds4_gpu_tensor *weights, const ds4_gpu_tensor *selected,
        uint32_t layer, uint32_t n_expert_used, uint32_t n_tokens) {
    if (!g_zc_ge || layer >= g_zc_n_layer || !g_zc_ge_present[layer]) return 1;
    uint32_t total = n_tokens * n_expert_used;
    zchain_ge_kernel<<<(total + 255u) / 256u, 256, 0, g_cur_stream>>>(
        (float *)weights->ptr, (const int *)selected->ptr, g_zc_ge,
        g_zc_n_expert, total, layer * g_zc_n_expert);
    return 1;
}

#define ZC_NTG 256u
/* 一 block 每 token。shared: red[256] 树归约 | pv[<=1024] z⊙(V^T x) | ua[d<=2048]
 * U pv 投影缓存(免二遍 U 读; 与 host ds4_zchain_zl_apply 数学等价)。 */
static __device__ __forceinline__ float zc_red_add(float v, float *red) {
    uint32_t tid = threadIdx.x;
    red[tid] = v; __syncthreads();
    for (uint32_t s = ZC_NTG >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float r = red[0]; __syncthreads();
    return r;
}

static __global__ void zchain_scale_kernel(
        float *routed, const float *x, const float *ops, const __half *v8,
        const __half *zlm, uint32_t d, uint32_t n_tokens,
        uint32_t op_start, uint32_t op_count,
        uint32_t zl_k, uint32_t zl_off, uint32_t zl_din, float zl_tr, uint32_t zl_mul) {
    extern __shared__ float sh[];               /* red[ZC_NTG] | pv[zl_k] | ua[d] */
    float *red = sh, *pv = sh + ZC_NTG, *ua = pv + zl_k;
    uint32_t tok = blockIdx.x, tid = threadIdx.x;
    if (tok >= n_tokens) return;
    const float *xt = x + (uint64_t)tok * d;
    float *rt = routed + (uint64_t)tok * d;

    float acc = 0.0f;
    for (uint32_t j = tid; j < d; j += ZC_NTG) acc += xt[j] * xt[j];
    const float ss = zc_red_add(acc, red);
    const float xnorm = sqrtf(ss);

    float lam = 1.0f;
    for (uint32_t oi = 0; oi < op_count; oi++) {
        const float *op = ops + (uint64_t)(op_start + oi) * 16u;
        uint32_t ty = (uint32_t)op[0];
        if (ty == 1u) lam = op[1] * lam;
        else if (ty == 2u) {
            float c = op[2] + op[3] * ((xnorm - op[4]) / op[5]);
            c = fminf(fmaxf(c, 0.25f), 4.0f);
            lam = c * lam;
        } else if (ty == 3u) {
            int blk = (int)op[15];
            if (blk >= 0) {
                float c = op[6];
                for (uint32_t k = 0; k < 8u; k++) {
                    const __half *vr = v8 + ((uint64_t)blk * 8u + k) * d;
                    float dk = 0.0f;
                    for (uint32_t j = tid; j < d; j += ZC_NTG) dk += xt[j] * __half2float(vr[j]);
                    c += op[7 + k] * zc_red_add(dk, red);
                }
                c = fminf(fmaxf(c, 0.25f), 4.0f);
                lam = c * lam;
            }
        } else if (ty == 4u) lam = 1.0f + op[1] * (lam - 1.0f);
    }
    if (op_count) for (uint32_t j = tid; j < d; j += ZC_NTG) rt[j] *= lam;

    if (zl_k > 0u) {   /* frozen z^L: routed += clip * U diag(z) V^T x (λ 之后) */
        const uint32_t din = zl_din > 0u ? zl_din : d;
        const __half *hz = zlm + zl_off;
        /* type9(动态 z, zl_mul==3): 载荷是 A|U|V, 无 z[k] 前缀 ⇒ 偏移与 type6/7 不同 */
        const __half *hA = (zl_mul == 3u) ? (zlm + zl_off) : NULL;
        const __half *hU = (zl_mul == 3u) ? (zlm + zl_off + (uint64_t)din * zl_k) : (hz + zl_k);
        const __half *hV = hU + (uint64_t)d * zl_k;
        float nrm = 0.0f;
        if (din == 3u * d) {   /* md86 ftA: φ=[x, x⊙x/rms, relu(x)] */
            nrm = sqrtf(ss / (float)d) + 1e-6f;
        }
        for (uint32_t c = tid; c < zl_k; c += ZC_NTG) {
            float dk = 0.0f;
            if (din == 3u * d) {
                for (uint32_t j = 0; j < d; j++) {
                    float xv = xt[j];
                    dk += xv * __half2float(hV[(uint64_t)j * zl_k + c]);
                    dk += (xv * xv / nrm) * __half2float(hV[(uint64_t)(d + j) * zl_k + c]);
                    dk += (xv > 0.0f ? xv : 0.0f) * __half2float(hV[(uint64_t)(2u * d + j) * zl_k + c]);
                }
            } else {
                for (uint32_t j = 0; j < d; j++) dk += xt[j] * __half2float(hV[(uint64_t)j * zl_k + c]);
            }
            const float zs = (zl_tr > 0.0f ? zl_tr : 1.0f);
            if (zl_mul == 3u) {          /* AMPD: pv = tanh(Vᵀx/s) · tanh(Aᵀx/s), z 是 x 的函数 */
                float gk = 0.0f;
                if (din == 3u * d) {
                    for (uint32_t j = 0; j < d; j++) {
                        float xv = xt[j];
                        gk += xv * __half2float(hA[(uint64_t)j * zl_k + c]);
                        gk += (xv * xv / nrm) * __half2float(hA[(uint64_t)(d + j) * zl_k + c]);
                        gk += (xv > 0.0f ? xv : 0.0f) * __half2float(hA[(uint64_t)(2u * d + j) * zl_k + c]);
                    }
                } else {
                    for (uint32_t j = 0; j < d; j++) gk += xt[j] * __half2float(hA[(uint64_t)j * zl_k + c]);
                }
                pv[c] = tanhf(dk / zs) * tanhf(gk / zs);
            } else {
                if (zl_mul) dk = tanhf(dk / zs);   /* AMP(type7) */
                pv[c] = dk * __half2float(hz[c]);
            }
        }
        __syncthreads();
        float nd_p = 0.0f, nr_p = 0.0f;
        for (uint32_t j = tid; j < d; j += ZC_NTG) {
            const __half *ur = hU + (uint64_t)j * zl_k;
            float a = 0.0f;
            for (uint32_t c = 0; c < zl_k; c++) a += pv[c] * __half2float(ur[c]);
            ua[j] = a;
            nd_p += a * a;
            nr_p += rt[j] * rt[j];
        }
        if (zl_mul) {   /* 乘性出口(type7 AMP / type9 AMPD): ⊙(1+ua), 无信任域 */
            for (uint32_t j = tid; j < d; j += ZC_NTG) rt[j] *= (1.0f + ua[j]);
        } else {
            const float nd = sqrtf(zc_red_add(nd_p, red));
            const float nr = sqrtf(zc_red_add(nr_p, red));
            const float cap = zl_tr * nr;
            const float s = (nd > cap && nd > 0.0f) ? (cap / nd) : 1.0f;
            for (uint32_t j = tid; j < d; j += ZC_NTG) rt[j] += s * ua[j];
        }
    }
}

/* ★z^L decode 快路(2026-08-19): 老 zchain_scale_kernel 每 token 单 block, decode(n=1) 时
 * 全 GPU 只有 1 个 SM 干活且 U 段跨线程步长 k 非合并读 → 实测 ~2.7ms/层, 21.4→6.3 t/s。
 * 快路=三段网格化: pv=diag(z)Vᵀx(64列/块合并读) → ua=U·pv(warp/行合并读+范数原子归约)
 * → 信任域缩放加回。数值=同式(浮点归约序容差); 批量 prefill(n>16, 本身有 token 并行度)
 * 与 λ ops 仍走老 kernel。scratch 在 zl_set 预分配(graph capture 内禁 cudaMalloc)。 */
static float *g_zc_zl_pv = NULL, *g_zc_zl_ua = NULL, *g_zc_zl_n2 = NULL;
static float *g_zc_zl_pvp = NULL;      /* pv 两相归约 partial: MAXTOK×SEG×kmax */
static uint8_t *g_zc_zlm8 = NULL;      /* U/V fp8(e4m3) 影子(2026-08-20 ②刀): 半字节读, 省168MB/tok */
static uint32_t *g_zc_zl_mul = NULL;   /* per-layer 1=乘性AMP(type7) | 3=动态z(type9 AMPD) */
#define ZC_ZL_FAST_MAXTOK 16u
/* ★pv 占用率手术(2026-08-20)★: 老 zc_zl_pv_kernel 在 decode(n=1) 只开 k/64≈8 个 block,
 * GB10 绝大部分 SM 闲置 → nsys 实测 130µs/层(4MB V 只跑出 31GB/s), 42 层 AMP 链税 5.4ms。
 * 两相归约: A) 按 d 切 SEG 段并行出 partial(k/64×SEG×tok 个 block, 占用率拉满)
 *          B) 每列跨段求和 + tanh 定标 ×z。数值=同式(浮点归约序容差, 与老 kernel 同级)。
 * fta(φ=3d 特征抬升) 层继续走老 kernel(AMP/加性主链 din==d)。 */
#define ZC_ZL_SEG 32u   /* 08-20 提速: 16→32, pv_part 块数 128→256 再拉占用率 */

static __global__ void zc_zl_pv_kernel(
        const float *x, const __half *zlm, float *pv,
        uint32_t d, uint32_t k, uint32_t off, uint32_t fta, uint32_t mul, float mscale) {
    const uint32_t tok = blockIdx.y;
    const uint32_t tc = threadIdx.x & 63u, rg = threadIdx.x >> 6;   /* 64列×4行组 */
    const uint32_t c = blockIdx.x * 64u + tc;
    const __half *hz = zlm + off;
    const __half *hV = hz + k + (uint64_t)d * k;
    const float *xt = x + (uint64_t)tok * d;
    __shared__ float sh[4][64];
    __shared__ float r2[256];
    __shared__ float snrm;
    float nrm = 0.0f;
    if (fta) {   /* ftA 需要 rms(x): 块内自算(与 CPU zl_phi 同式) */
        float a2 = 0.0f;
        for (uint32_t j = threadIdx.x; j < d; j += 256u) a2 += xt[j] * xt[j];
        r2[threadIdx.x] = a2; __syncthreads();
        for (uint32_t s = 128u; s; s >>= 1) { if (threadIdx.x < s) r2[threadIdx.x] += r2[threadIdx.x + s]; __syncthreads(); }
        if (!threadIdx.x) snrm = sqrtf(r2[0] / (float)d) + 1e-6f;
        __syncthreads();
        nrm = snrm;
    }
    float acc = 0.0f;
    if (c < k) {
        if (fta) {
            for (uint32_t j = rg; j < d; j += 4u) {
                const float xv = xt[j];
                acc += xv * __half2float(hV[(uint64_t)j * k + c]);
                acc += (xv * xv / nrm) * __half2float(hV[(uint64_t)(d + j) * k + c]);
                acc += (xv > 0.0f ? xv : 0.0f) * __half2float(hV[(uint64_t)(2u * d + j) * k + c]);
            }
        } else {
            for (uint32_t j = rg; j < d; j += 4u)
                acc += xt[j] * __half2float(hV[(uint64_t)j * k + c]);
        }
    }
    sh[rg][tc] = acc; __syncthreads();
    if (rg == 0 && c < k) {
        float a = sh[0][tc] + sh[1][tc] + sh[2][tc] + sh[3][tc];
        if (mul) a = tanhf(a / (mscale > 0.0f ? mscale : 1.0f));   /* AMP: tanh 定标 */
        pv[(uint64_t)tok * k + c] = a * __half2float(hz[c]);
    }
}

#include <cuda_fp8.h>
__device__ __forceinline__ static float zc_fp8_ld(const uint8_t *p) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(*p, __NV_E4M3);
    return __half2float(*(const __half *)&hr);
}
static __global__ void zc_h2fp8_kernel(uint8_t *dst, const __half *src, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __nv_cvt_float_to_fp8(__half2float(src[i]), __NV_SATFINITE, __NV_E4M3);
}

static __global__ void zc_zl_pv_part8_kernel(   /* fp8 版两相 A(V 读 e4m3) */
        const float *x, const uint8_t *zlm8, float *pvp,
        uint32_t d, uint32_t k, uint32_t off) {
    const uint32_t tok = blockIdx.z, seg = blockIdx.y;
    const uint32_t tc = threadIdx.x & 63u, rg = threadIdx.x >> 6;
    const uint32_t c = blockIdx.x * 64u + tc;
    const uint8_t *hV = zlm8 + off + k + (uint64_t)d * k;
    const float *xt = x + (uint64_t)tok * d;
    const uint32_t rows = d / ZC_ZL_SEG, j0 = seg * rows, j1 = j0 + rows;
    float acc = 0.0f;
    if (c < k)
        for (uint32_t j = j0 + rg; j < j1; j += 4u)
            acc += xt[j] * zc_fp8_ld(hV + (uint64_t)j * k + c);
    __shared__ float sh[4][64];
    sh[rg][tc] = acc; __syncthreads();
    if (rg == 0 && c < k)
        pvp[((uint64_t)tok * ZC_ZL_SEG + seg) * k + c] = sh[0][tc] + sh[1][tc] + sh[2][tc] + sh[3][tc];
}

static __global__ void zc_zl_ua8_kernel(   /* fp8 版 UA(U 读 e4m3) */
        const float *routed, const float *pv, const uint8_t *zlm8, float *ua, float *n2,
        uint32_t d, uint32_t k, uint32_t off, uint32_t mul) {
    const uint32_t tok = blockIdx.y;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t j = blockIdx.x * 8u + warp;
    const uint8_t *hU = zlm8 + off + k;
    extern __shared__ float spv8[];
    for (uint32_t c = threadIdx.x; c < k; c += 256u) spv8[c] = pv[(uint64_t)tok * k + c];
    __syncthreads();
    float a = 0.0f;
    if (j < d) {
        const uint8_t *ur = hU + (uint64_t)j * k;
        for (uint32_t c = lane; c < k; c += 32u) a += spv8[c] * zc_fp8_ld(ur + c);
        for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
    }
    if (mul) {
        if (lane == 0 && j < d) ua[(uint64_t)tok * d + j] = a;
        return;
    }
    __shared__ float snd[8], snr[8];
    if (lane == 0) {
        if (j < d) {
            ua[(uint64_t)tok * d + j] = a;
            const float r = routed[(uint64_t)tok * d + j];
            snd[warp] = a * a; snr[warp] = r * r;
        } else { snd[warp] = 0.0f; snr[warp] = 0.0f; }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float nd = 0.0f, nr = 0.0f;
        for (int w = 0; w < 8; w++) { nd += snd[w]; nr += snr[w]; }
        atomicAdd(&n2[(uint64_t)tok * 2u], nd);
        atomicAdd(&n2[(uint64_t)tok * 2u + 1u], nr);
    }
}

static __global__ void zc_zl_pv_part_kernel(   /* 两相 A: partial[seg][c]=Σ_{j∈seg} x_j·V[j,c]
     * v3(2026-08-20 顺序流重排): 旧版每 j 只读 128B 且 rg 跳 4 行(4KB 跨步)实测 90GB/s;
     * 现一 block 承包段内全行×全列: 行内 k×2B 连续+行间顺序=纯流。线程持列 {t,t+256,..}。 */
        const float *x, const __half *zlm, float *pvp,
        uint32_t d, uint32_t k, uint32_t off) {
    const uint32_t tok = blockIdx.z, seg = blockIdx.y;
    const __half *hV = zlm + off + k + (uint64_t)d * k;
    const float *xt = x + (uint64_t)tok * d;
    const uint32_t rows = d / ZC_ZL_SEG, j0 = seg * rows;
    /* v5: 64列对×4行组/块 + half2 满线事务(v2 病=单half 64B半线; v3/v4 病=32块喂不饱
     * GB10 延迟). grid=(k2/64, SEG, ntok)≈128 块。 */
    const uint32_t tc = threadIdx.x & 63u, rg = threadIdx.x >> 6;
    const uint32_t kh = k >> 1;
    const uint32_t cp = blockIdx.x * 64u + tc;      /* half2 列对号 */
    float2 acc = {0.0f, 0.0f};
    if (cp < kh) {
        const uint32_t j1 = j0 + rows;
        for (uint32_t j = j0 + rg; j < j1; j += 4u) {
            const __half2 v = ((const __half2 *)(hV + (uint64_t)j * k))[cp];
            const float xv = xt[j];
            acc.x += xv * __low2float(v);
            acc.y += xv * __high2float(v);
        }
    }
    __shared__ float2 sh2[4][64];
    sh2[rg][tc] = acc; __syncthreads();
    if (rg == 0 && cp < kh) {
        const float2 a0 = sh2[0][tc], a1 = sh2[1][tc], a2 = sh2[2][tc], a3 = sh2[3][tc];
        const uint64_t base = ((uint64_t)tok * ZC_ZL_SEG + seg) * k + 2u * cp;
        pvp[base] = a0.x + a1.x + a2.x + a3.x;
        pvp[base + 1u] = a0.y + a1.y + a2.y + a3.y;
    }
}

static __global__ void zc_zl_pv_reduce_kernel(   /* 两相 B: 跨段求和 + AMP tanh 定标 ×z */
        const __half *zlm, const float *pvp, float *pv,
        uint32_t k, uint32_t off, uint32_t mul, float mscale) {
    const uint32_t tok = blockIdx.y;
    const uint32_t c = blockIdx.x * 256u + threadIdx.x;
    if (c >= k) return;
    float a = 0.0f;
    for (uint32_t s = 0; s < ZC_ZL_SEG; s++)
        a += pvp[((uint64_t)tok * ZC_ZL_SEG + s) * k + c];
    if (mul) a = tanhf(a / (mscale > 0.0f ? mscale : 1.0f));
    pv[(uint64_t)tok * k + c] = a * __half2float(zlm[off + c]);
}

static __global__ void zc_zl_ua_kernel(
        const float *routed, const float *pv, const __half *zlm, float *ua, float *n2,
        uint32_t d, uint32_t k, uint32_t off, uint32_t mul) {
    const uint32_t tok = blockIdx.y;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t j = blockIdx.x * 8u + warp;   /* warp/行: ur[c..] 合并读 */
    const __half *hU = zlm + off + k;
    extern __shared__ float spv[];   /* pv 预载共享内存: 免 8 行重复读全局 */
    for (uint32_t c = threadIdx.x; c < k; c += 256u) spv[c] = pv[(uint64_t)tok * k + c];
    __syncthreads();
    float a = 0.0f;
    if (j < d) {
        const __half *ur = hU + (uint64_t)j * k;
        for (uint32_t c = lane; c < k; c += 32u) a += spv[c] * __half2float(ur[c]);
        for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
    }
    if (mul) {   /* AMP 乘性: 无信任域 → 免范数归约/原子 */
        if (lane == 0 && j < d) ua[(uint64_t)tok * d + j] = a;
        return;
    }
    __shared__ float snd[8], snr[8];
    if (lane == 0) {
        if (j < d) {
            ua[(uint64_t)tok * d + j] = a;
            const float r = routed[(uint64_t)tok * d + j];
            snd[warp] = a * a; snr[warp] = r * r;
        } else { snd[warp] = 0.0f; snr[warp] = 0.0f; }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float nd = 0.0f, nr = 0.0f;
        for (int w = 0; w < 8; w++) { nd += snd[w]; nr += snr[w]; }
        atomicAdd(&n2[(uint64_t)tok * 2u], nd);
        atomicAdd(&n2[(uint64_t)tok * 2u + 1u], nr);
    }
}

static __global__ void zc_zl_add_kernel(
        float *routed, const float *ua, const float *n2, uint32_t d, float tr, uint32_t mul) {
    const uint32_t tok = blockIdx.y;
    const uint32_t j = blockIdx.x * 256u + threadIdx.x;
    if (j >= d) return;
    if (mul) {   /* AMP 乘性出口: ⊙(1+ua), 无信任域 */
        routed[(uint64_t)tok * d + j] *= (1.0f + ua[(uint64_t)tok * d + j]);
        return;
    }
    const float nd = sqrtf(n2[(uint64_t)tok * 2u]);
    const float nr = sqrtf(n2[(uint64_t)tok * 2u + 1u]);
    const float cap = tr * nr;
    const float s = (nd > cap && nd > 0.0f) ? cap / nd : 1.0f;
    routed[(uint64_t)tok * d + j] += s * ua[(uint64_t)tok * d + j];
}

int ds4_gpu_zchain_scale_routed(
        ds4_gpu_tensor *routed, const ds4_gpu_tensor *x,
        uint32_t layer, uint32_t n_tokens) {
    if (layer >= g_zc_n_layer || !g_zc_layer_off) return 1;
    uint32_t op_start = g_zc_layer_off[layer];
    uint32_t op_count = g_zc_layer_off[layer + 1] - op_start;
    uint32_t zk  = g_zc_zl_k   ? g_zc_zl_k[layer]   : 0;
    uint32_t zo  = g_zc_zl_off ? g_zc_zl_off[layer] : 0;
    uint32_t zd  = g_zc_zl_din ? g_zc_zl_din[layer] : 0;
    float    ztr = g_zc_zl_tr  ? g_zc_zl_tr[layer]  : 0.0f;
    if (!op_count && !zk) return 1;
    /* decode 快路的 fp8 影子按 z|U|V 布局打包, type9(AMPD) 是 A|U|V —— 布局不同, 先走通用核
     * (数值同式, 只是 decode 慢一档)。快路的 AMPD 版待做。 */
    const uint32_t zmul_l = g_zc_zl_mul ? g_zc_zl_mul[layer] : 0u;
    if (zk && zmul_l != 3u && n_tokens <= ZC_ZL_FAST_MAXTOK && g_zc_zl_pv &&
        g_zc_d_model <= 4096u && zk <= 1024u) {
        if (op_count) {   /* λ ops 仍走老 kernel(zk=0 抑制其 zl 段) */
            size_t shmem0 = (ZC_NTG + g_zc_d_model) * sizeof(float);
            zchain_scale_kernel<<<n_tokens, ZC_NTG, shmem0, g_cur_stream>>>(
                (float *)routed->ptr, (const float *)x->ptr, g_zc_ops, g_zc_v8, g_zc_zlm,
                g_zc_d_model, n_tokens, op_start, op_count, 0u, 0u, 0u, 0.0f, 0u);
        }
        const uint32_t d = g_zc_d_model;
        const uint32_t fta = (zd == 3u * d) ? 1u : 0u;
        const uint32_t zmul = zmul_l;
        if (!zmul)   /* AMP 无信任域不消费 n2 → 免每层 memset */
            cudaMemsetAsync(g_zc_zl_n2, 0, (size_t)n_tokens * 2u * sizeof(float), g_cur_stream);
        const uint32_t use8 = (g_zc_zlm8 != NULL) ? 1u : 0u;   /* fp8 影子在 = 走半字节路 */
        const uint32_t v4ok = (((zo + zk) & 1u) == 0u && (zk & 1u) == 0u);
        if (!fta && (d % ZC_ZL_SEG) == 0u && g_zc_zl_pvp && (use8 || v4ok)) {   /* 两相归约 */
            if (use8) {
                dim3 ga((zk + 63u) / 64u, ZC_ZL_SEG, n_tokens);
                zc_zl_pv_part8_kernel<<<ga, 256u, 0, g_cur_stream>>>(
                    (const float *)x->ptr, g_zc_zlm8, g_zc_zl_pvp, d, zk, zo);
            } else {
                /* v5: 64列对×4行组; half2 需 V 基址 4B 对齐(off+k 偶, v4ok 已闸) */
                dim3 ga((zk / 2u + 63u) / 64u, ZC_ZL_SEG, n_tokens);
                zc_zl_pv_part_kernel<<<ga, 256u, 0, g_cur_stream>>>(
                    (const float *)x->ptr, g_zc_zlm, g_zc_zl_pvp, d, zk, zo);
            }
            dim3 gb((zk + 255u) / 256u, n_tokens);
            zc_zl_pv_reduce_kernel<<<gb, 256u, 0, g_cur_stream>>>(
                g_zc_zlm, g_zc_zl_pvp, g_zc_zl_pv, zk, zo, zmul, ztr);
        } else {
            dim3 g1((zk + 63u) / 64u, n_tokens);
            zc_zl_pv_kernel<<<g1, 256u, 0, g_cur_stream>>>(
                (const float *)x->ptr, g_zc_zlm, g_zc_zl_pv, d, zk, zo, fta, zmul, ztr);
        }
        dim3 g2((d + 7u) / 8u, n_tokens);
        if (use8 && !fta)
            zc_zl_ua8_kernel<<<g2, 256u, (size_t)zk * sizeof(float), g_cur_stream>>>(
                (const float *)routed->ptr, g_zc_zl_pv, g_zc_zlm8, g_zc_zl_ua, g_zc_zl_n2, d, zk, zo, zmul);
        else
            zc_zl_ua_kernel<<<g2, 256u, (size_t)zk * sizeof(float), g_cur_stream>>>(
                (const float *)routed->ptr, g_zc_zl_pv, g_zc_zlm, g_zc_zl_ua, g_zc_zl_n2, d, zk, zo, zmul);
        dim3 g3((d + 255u) / 256u, n_tokens);
        zc_zl_add_kernel<<<g3, 256u, 0, g_cur_stream>>>(
            (float *)routed->ptr, g_zc_zl_ua, g_zc_zl_n2, d, ztr, zmul);
        return 1;
    }
    size_t shmem = (ZC_NTG + zk + g_zc_d_model) * sizeof(float);
    zchain_scale_kernel<<<n_tokens, ZC_NTG, shmem, g_cur_stream>>>(
        (float *)routed->ptr, (const float *)x->ptr, g_zc_ops, g_zc_v8, g_zc_zlm,
        g_zc_d_model, n_tokens, op_start, op_count, zk, zo, zd, ztr,
        g_zc_zl_mul ? g_zc_zl_mul[layer] : 0u);
    return 1;
}

/* 参数表与 ds4_gpu.h 逐字一致(本文件不 include 它, 编译器不校验)。 */
int ds4_gpu_zchain_zl_set(
        const uint16_t *zlm, const uint32_t *off, const uint32_t *k,
        const uint32_t *din, const float *tr, const uint32_t *mul,
        uint32_t n_layer, uint64_t total_halves) {
    uint32_t n_zl = 0, kmax = 0;
    for (uint32_t l = 0; k && l < n_layer; l++)
        if (k[l]) { n_zl++; if (k[l] > kmax) kmax = k[l]; }
    if (!n_zl) return 1;
    if (kmax > 1024) {   /* shared pv[] 布局上限(Metal 同限) */
        fprintf(stderr, "ds4: zchain z^L rank %u > 1024 unsupported on CUDA\n", kmax);
        return 0;
    }
    if (!cuda_ok(cudaMalloc(&g_zc_zlm, total_halves * sizeof(__half)), "zchain zlm")) return 0;
    if (!cuda_ok(cudaMemcpy(g_zc_zlm, zlm, total_halves * sizeof(__half), cudaMemcpyHostToDevice), "zchain zlm up")) return 0;
    {   /* fp8(e4m3) 影子(2026-08-20 ②刀): U/V 半字节读省 168MB/tok; z 仍读 fp16 正本。
         * DS4_ZC_FP8=0 关(逃生口)。e4m3 单元素相对误差~6%, U·pv 求和 k≤1024 项均化后
         * ~0.3% 量级(数值闸另验)。 */
        /* 默认关(2026-08-20 终判): 净+0.35 t/s 但 e4m3=真数值扰动(KL 3.8e-2 ≈ 放大器
         * 收益1/4), 质量门不换。DS4_ZC_FP8=1 显式开(批解码时代备用杠杆)。 */
        const char *e8 = ((const char *)0) /* DS4_ZC_FP8: 路径开关已删(2026-08-22 隐形炸弹清理) */;
        if (e8 && atoi(e8) != 0) {
            if (cudaMalloc(&g_zc_zlm8, total_halves) == cudaSuccess) {
                zc_h2fp8_kernel<<<(unsigned)((total_halves + 255u) / 256u), 256u>>>(
                    g_zc_zlm8, (const __half *)g_zc_zlm, total_halves);
                if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                    (void)cudaFree(g_zc_zlm8); g_zc_zlm8 = NULL; (void)cudaGetLastError();
                } else {
                    fprintf(stderr, "ds4: zchain fp8(e4m3) U/V shadow armed (%.1f MB)\n",
                            (double)total_halves / 1e6);
                }
            } else { g_zc_zlm8 = NULL; (void)cudaGetLastError(); }
        }
    }
    g_zc_zl_off = (uint32_t *)malloc(n_layer * sizeof(uint32_t));
    g_zc_zl_k   = (uint32_t *)malloc(n_layer * sizeof(uint32_t));
    g_zc_zl_din = (uint32_t *)malloc(n_layer * sizeof(uint32_t));
    g_zc_zl_tr  = (float *)malloc(n_layer * sizeof(float));
    memcpy(g_zc_zl_off, off, n_layer * sizeof(uint32_t));
    memcpy(g_zc_zl_k, k, n_layer * sizeof(uint32_t));
    if (din) memcpy(g_zc_zl_din, din, n_layer * sizeof(uint32_t));
    else     memset(g_zc_zl_din, 0, n_layer * sizeof(uint32_t));
    memcpy(g_zc_zl_tr, tr, n_layer * sizeof(float));
    g_zc_zl_mul = (uint32_t *)calloc(n_layer, sizeof(uint32_t));
    if (mul) memcpy(g_zc_zl_mul, mul, n_layer * sizeof(uint32_t));
    uint32_t n_mul = 0;
    for (uint32_t l = 0; l < n_layer; l++) if (g_zc_zl_mul[l]) n_mul++;
    /* decode 快路 scratch(capture 内禁 cudaMalloc → 此处预分配; 上限 16tok×d4096/k1024) */
    if (!g_zc_zl_pv) {
        if (!cuda_ok(cudaMalloc(&g_zc_zl_pv, (size_t)ZC_ZL_FAST_MAXTOK * 1024u * sizeof(float)), "zl pv")) return 0;
        if (!cuda_ok(cudaMalloc(&g_zc_zl_ua, (size_t)ZC_ZL_FAST_MAXTOK * 4096u * sizeof(float)), "zl ua")) return 0;
        if (!cuda_ok(cudaMalloc(&g_zc_zl_n2, (size_t)ZC_ZL_FAST_MAXTOK * 2u * sizeof(float)), "zl n2")) return 0;
        if (!cuda_ok(cudaMalloc(&g_zc_zl_pvp, (size_t)ZC_ZL_FAST_MAXTOK * ZC_ZL_SEG * 1024u * sizeof(float)), "zl pvp")) return 0;
    }
    fprintf(stderr, "ds4: zchain CUDA z^L armed: %u layers (AMP=%u), kmax=%u (%.1f MB)\n",
            n_zl, n_mul, kmax, total_halves * 2.0 / 1e6);
    return 1;
}

/* ==== 路由闭式侧车(type8 zl.RTE, 2026-08-19): δlogits = U·diag(z)·tanh(Vᵀx/s)
 * 加在 router raw logits 上(select 前)。blob 布局 z[k]|U[ne*k]|V[d*k], 行主
 * [dim][k]。decode n=1 为主, 出维仅 n_expert=256 → 单 kernel 两段:
 * 先 256 线程按列并行算 pv[c]=z_c·tanh((Vᵀx)_c/s), 再按行算 δ 加进 logits。 */
static __half *g_zc_rt = NULL;
static uint32_t *g_zc_rt_off = NULL, *g_zc_rt_k = NULL;
static float *g_zc_rt_s = NULL;
static uint32_t g_zc_rt_nl = 0, g_zc_rt_ne = 0;
static float *g_zc_rt_pv = NULL;   /* [16tok × kmax1024] scratch */

static __global__ void zc_rte_pv_kernel(
        const float *x, const __half *rm, float *pv,
        uint32_t d, uint32_t k, uint32_t ne, uint32_t off, float s) {
    /* V 在 rte_set 上传时已转置为 [k][d](行 c 连续) — [d][k] 原布局按列跨步读
     * 每线程 4096 次 half 步长 k*2B, 完全不合并。warp/列: 32 lane 分段行内连续读。 */
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t c = blockIdx.x * 8u + warp;
    const uint32_t tok = blockIdx.y;
    if (c >= k) return;
    const __half *hz = rm + off;
    const __half *hVt = hz + k + (uint64_t)ne * k + (uint64_t)c * d;
    const float *xt = x + (uint64_t)tok * d;
    float a = 0.0f;
    for (uint32_t j = lane; j < d; j += 32u) a += xt[j] * __half2float(hVt[j]);
    for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
    if (lane == 0)
        pv[(uint64_t)tok * k + c] = tanhf(a / (s > 0.0f ? s : 1.0f)) * __half2float(hz[c]);
}

static __global__ void zc_rte_add_kernel(
        float *logits, const float *pv, const __half *rm,
        uint32_t k, uint32_t ne, uint32_t off) {
    const uint32_t tok = blockIdx.y;
    const uint32_t e = blockIdx.x * 256u + threadIdx.x;
    if (e >= ne) return;
    const __half *hU = rm + off + k;
    const float *pvt = pv + (uint64_t)tok * k;
    float a = 0.0f;
    for (uint32_t c = 0; c < k; c++) a += pvt[c] * __half2float(hU[(uint64_t)e * k + c]);
    logits[(uint64_t)tok * ne + e] += a;
}

int ds4_gpu_zchain_rte_set(
        const uint16_t *rm, const uint32_t *off, const uint32_t *k,
        const float *scale, uint32_t n_layer, uint32_t n_expert, uint64_t total_halves) {
    if (!rm || !total_halves || !n_layer) return 1;
    /* 上传前把每层 V 段 [d][k] 转置成 [k][d](pv kernel 行连续合并读; U 段布局不变) */
    uint16_t *tp = (uint16_t *)malloc(total_halves * sizeof(uint16_t));
    if (!tp) return 0;
    memcpy(tp, rm, total_halves * sizeof(uint16_t));
    const uint32_t d = 4096u;
    for (uint32_t l = 0; l < n_layer; l++) {
        const uint32_t kl = k[l];
        if (!kl) continue;
        const uint64_t vo = (uint64_t)off[l] + kl + (uint64_t)n_expert * kl;
        const uint16_t *src = rm + vo;
        uint16_t *dst = tp + vo;
        for (uint32_t j = 0; j < d; j++)
            for (uint32_t c = 0; c < kl; c++)
                dst[(uint64_t)c * d + j] = src[(uint64_t)j * kl + c];
    }
    if (!cuda_ok(cudaMalloc(&g_zc_rt, total_halves * sizeof(__half)), "zchain rte")) { free(tp); return 0; }
    if (!cuda_ok(cudaMemcpy(g_zc_rt, tp, total_halves * sizeof(__half), cudaMemcpyHostToDevice), "zchain rte up")) { free(tp); return 0; }
    free(tp);
    g_zc_rt_off = (uint32_t *)malloc(n_layer * sizeof(uint32_t));
    g_zc_rt_k   = (uint32_t *)malloc(n_layer * sizeof(uint32_t));
    g_zc_rt_s   = (float *)malloc(n_layer * sizeof(float));
    memcpy(g_zc_rt_off, off, n_layer * sizeof(uint32_t));
    memcpy(g_zc_rt_k, k, n_layer * sizeof(uint32_t));
    memcpy(g_zc_rt_s, scale, n_layer * sizeof(float));
    g_zc_rt_nl = n_layer; g_zc_rt_ne = n_expert;
    if (!g_zc_rt_pv &&
        !cuda_ok(cudaMalloc(&g_zc_rt_pv, (size_t)16u * 1024u * sizeof(float)), "rte pv")) return 0;
    uint32_t nr = 0, kmax = 0;
    for (uint32_t l = 0; l < n_layer; l++) { if (k[l]) nr++; if (k[l] > kmax) kmax = k[l]; }
    fprintf(stderr, "ds4: zchain CUDA route 侧车 armed: %u layers, kmax=%u (%.1f MB)\n",
            nr, kmax, total_halves * 2.0 / 1e6);
    return 1;
}

static uint64_t g_zc_rt_pv_cap = 16u * 1024u;   /* floats; 预分配 16tok, prefill 按需扩 */

int ds4_gpu_zchain_route_bias(
        ds4_gpu_tensor *logits, const ds4_gpu_tensor *x, uint32_t layer, uint32_t n_tokens) {
    if (!g_zc_rt || layer >= g_zc_rt_nl || !g_zc_rt_k || !g_zc_rt_k[layer]) return 1;
    if (!logits || !x || n_tokens == 0) return 1;
    const uint32_t k = g_zc_rt_k[layer], ne = g_zc_rt_ne, d = 4096u;
    if (x->bytes < (uint64_t)n_tokens * d * sizeof(float) ||
        logits->bytes < (uint64_t)n_tokens * ne * sizeof(float) || k > 1024u) return 1;
    const uint64_t need = (uint64_t)n_tokens * k;
    if (need > g_zc_rt_pv_cap) {
        /* prefill 大批量: 非 capture 态才可扩容; capture 态(decode n=1 恒定)不会到这 */
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        (void)cudaStreamIsCapturing(g_cur_stream, &cs);
        if (cs != cudaStreamCaptureStatusNone) return 1;
        if (g_zc_rt_pv) (void)cudaFree(g_zc_rt_pv);
        g_zc_rt_pv = NULL; g_zc_rt_pv_cap = 0;
        if (!cuda_ok(cudaMalloc(&g_zc_rt_pv, need * sizeof(float)), "rte pv grow")) return 0;
        g_zc_rt_pv_cap = need;
    }
    dim3 g1((k + 7u) / 8u, n_tokens);   /* warp/列: 8 列×32 lane per block */
    zc_rte_pv_kernel<<<g1, 256u, 0, g_cur_stream>>>(
        (const float *)x->ptr, g_zc_rt, g_zc_rt_pv, d, k, ne, g_zc_rt_off[layer], g_zc_rt_s[layer]);
    dim3 g2((ne + 255u) / 256u, n_tokens);
    zc_rte_add_kernel<<<g2, 256u, 0, g_cur_stream>>>(
        (float *)logits->ptr, g_zc_rt_pv, g_zc_rt, k, ne, g_zc_rt_off[layer]);
    return cuda_ok(cudaGetLastError(), "zchain route bias launch");
}
