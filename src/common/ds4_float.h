/* ds4_float.h — f16/bf16 标量转换的全仓唯一实现。
 *
 * 来源与数值契约(逐式转录, 不改数值):
 *   - ds4_f16_to_f32: zlayer.c f16_to_f32(2026-08-25 对 gguf-py 逐位过闸的那份)。
 *   - ds4_f64_to_f16: zlayer.c f64_to_f16(与 numpy .astype(np.float16) 同,
 *     round-to-nearest-even, 直接从 double 舍入避免 f32 中转双重舍入)。
 *   - ds4_bf16_to_f32: 高 16 位左移, 各处(zlayer/st_read/quants)同式。
 *
 * 用法: C99 / Objective-C / CUDA 皆可 include; CUDA 下函数自动带
 * __host__ __device__(常量在两侧各自实例化)。改这里 = 改所有消费方,
 * 这正是本文件存在的目的 —— 此前 f16 转换在仓里有 4 份人肉副本。 */
#ifndef DS4_COMMON_FLOAT_H
#define DS4_COMMON_FLOAT_H

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef __CUDACC__
#define DS4_HOSTDEV __host__ __device__ static inline
#else
#define DS4_HOSTDEV static inline
#endif

DS4_HOSTDEV float ds4_f16_to_f32(uint16_t h) {
    int s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = (float)ldexp((double)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = (float)ldexp((double)(m + 1024), e - 25);
    return s ? -v : v;
}

DS4_HOSTDEV float ds4_bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float v;
    memcpy(&v, &u, 4);
    return v;
}

/* f64 → f16, round-to-nearest-even。进位可自然溢进指数位(正确)。 */
DS4_HOSTDEV uint16_t ds4_f64_to_f16(double d) {
    uint64_t x; memcpy(&x, &d, 8);
    uint32_t sign = (uint32_t)((x >> 48) & 0x8000u);
    int e64 = (int)((x >> 52) & 0x7FF);
    uint64_t man = x & 0xFFFFFFFFFFFFFULL;              /* 52 位尾数 */
    if (e64 == 0x7FF) return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u));
    if (e64 == 0) return (uint16_t)sign;                 /* f64 次正规 → f16 必为 ±0 */
    int e = e64 - 1023 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    if (e <= 0) {                                        /* f16 次正规: m = (2^52+man)/2^(43-e) */
        if (e < -10) return (uint16_t)sign;
        int sh = 43 - e;                                 /* 43..53 */
        uint64_t full = man | (1ULL << 52);
        uint64_t m = full >> sh, rem = full & ((1ULL << sh) - 1), half = 1ULL << (sh - 1);
        if (rem > half || (rem == half && (m & 1))) m++;
        return (uint16_t)(sign | (uint32_t)m);
    }
    uint32_t h = ((uint32_t)e << 10) | (uint32_t)(man >> 42);
    uint64_t rem = man & ((1ULL << 42) - 1), half = 1ULL << 41;
    if (rem > half || (rem == half && (h & 1))) h++;
    return (uint16_t)(sign | h);
}

/* f32 小端字节 → f32(不经对齐读) */
DS4_HOSTDEV float ds4_f32_load_le(const uint8_t *p) {
    uint32_t b = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                 ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    float f;
    memcpy(&f, &b, sizeof f);
    return f;
}

#endif /* DS4_COMMON_FLOAT_H */
