/* test_common.c — src/common 共享格式库的离线单测(无模型/无GPU/无网络)。
 *
 * 三组:
 *   1) dequant 金标: tests/fixtures/quantfmt/ 每类型固定输入块 + 过闸版 zlayer
 *      (2026-08-25, 对 gguf-py 逐位) 的输出, 逐字节回归。库改动必先过这道闸。
 *   2) fp8 基元: E4M3FN/E8M0/E2M1 已知值与舍入 ties 规则。
 *   3) GGUF 读器: 现场合成一个最小 GGUF v3(1 KV + 2 张量), 开→找→取数→dequant。
 *
 * 从仓库根运行(make test 即如此); 夹具走相对路径。 */
#include "ds4_float.h"
#include "ds4_fp8.h"
#include "ds4_gguf.h"
#include "ds4_quantfmt.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, ...) do { \
        if (!(cond)) { g_fail++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                       fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } \
    } while (0)

static uint8_t *slurp(const char *path, size_t *n_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = (uint8_t *)malloc((size_t)n);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    fclose(f); *n_out = (size_t)n; return b;
}

/* ---- 1) dequant 金标夹具逐字节回归 ---- */
static void test_deq_fixture(uint32_t ty, const char *nm, uint64_t nblk) {
    char pin[256], pgold[256];
    snprintf(pin, sizeof pin, "tests/fixtures/quantfmt/%s.blocks.bin", nm);
    snprintf(pgold, sizeof pgold, "tests/fixtures/quantfmt/%s.golden.f32", nm);
    size_t nin, ngold;
    uint8_t *in = slurp(pin, &nin), *gold = slurp(pgold, &ngold);
    CHECK(in && gold, "%s: 夹具读不到(要求从仓库根运行)", nm);
    if (!in || !gold) { free(in); free(gold); return; }
    uint64_t blk, tsz;
    CHECK(ds4_ggt_geom(ty, &blk, &tsz) == 1, "%s: 类型 %u 几何未知", nm, ty);
    CHECK(nin == nblk * tsz, "%s: 输入 %zu ≠ %llu 块 × %llu B", nm, nin,
          (unsigned long long)nblk, (unsigned long long)tsz);
    const uint64_t nelem = nblk * blk;
    CHECK(ngold == nelem * 4, "%s: 金标 %zu ≠ %llu f32", nm, ngold, (unsigned long long)nelem);
    float *out = (float *)malloc(nelem * sizeof(float));
    CHECK(ds4_deq_bytes(ty, in, nelem, out) == 0, "%s: dequant 失败", nm);
    CHECK(memcmp(out, gold, nelem * 4) == 0, "%s: 输出与金标不逐字节一致", nm);
    free(out); free(in); free(gold);
}

static void test_deq_error_paths(void) {
    float out[32];
    uint8_t src[34] = {0};
    CHECK(ds4_deq_bytes(99, src, 32, out) == -1, "未知类型要返回 -1");
    CHECK(ds4_deq_bytes(DS4_GGT_Q8_0, src, 31, out) == -1, "非整块元素数要返回 -1");
    uint64_t blk, tsz;
    CHECK(ds4_ggt_geom(99, &blk, &tsz) == 0, "未知类型几何要返回 0");
}

/* ---- 2) fp8 基元 ---- */
static void test_fp8(void) {
    /* E4M3FN 边界值: 0x00=+0, 0x80=-0, 0x7E=448(最大), 0x7F=NaN, 0x08=2^-6 */
    CHECK(ds4_e4m3fn_to_f32(0x00) == 0.0f, "e4m3 0x00");
    CHECK(signbit(ds4_e4m3fn_to_f32(0x80)), "e4m3 0x80 = -0");
    CHECK(ds4_e4m3fn_to_f32(0x7E) == 448.0f, "e4m3 0x7E = 448, 得 %g", ds4_e4m3fn_to_f32(0x7E));
    CHECK(isnan(ds4_e4m3fn_to_f32(0x7F)), "e4m3 0x7F = NaN");
    CHECK(isnan(ds4_e4m3fn_to_f32(0xFF)), "e4m3 0xFF = NaN");
    CHECK(ds4_e4m3fn_to_f32(0x08) == 0.015625f, "e4m3 0x08 = 2^-6");
    CHECK(ds4_e4m3fn_to_f32(0x01) == ldexpf(1.0f, -9), "e4m3 0x01 = 2^-9(次正规)");
    /* 表/位型解码一致性: 下标 i(0..126) 与位型 i 的解码值应相同 */
    for (int i = 0; i < 127; i++)
        CHECK(ds4_e4m3fn_value(i) == ds4_e4m3fn_to_f32((uint8_t)i),
              "e4m3 表[%d] 与位型解码不一致", i);
    /* 舍入: 饱和 + 最近 + ties 偶尾数 */
    CHECK(ds4_e4m3fn_round(1000.0f) == 448.0f, "e4m3 round 饱和");
    CHECK(ds4_e4m3fn_round(-1000.0f) == -448.0f, "e4m3 round 负饱和");
    CHECK(ds4_e4m3fn_round(1.0f) == 1.0f, "e4m3 round 精确值");
    /* E8M0: e=127 → 2^0=1.0; e=0 → 位型 0x00400000 */
    CHECK(ds4_e8m0_to_f32(127) == 1.0f, "e8m0 127 = 1.0");
    CHECK(ds4_e8m0_to_f32(128) == 2.0f, "e8m0 128 = 2.0");
    { uint32_t bits; float v = ds4_e8m0_to_f32(0); memcpy(&bits, &v, 4);
      CHECK(bits == 0x00400000u, "e8m0 0 位型"); }
    /* E2M1/FP4 */
    CHECK(ds4_e2m1fn_value(7) == 6.0f, "e2m1 表尾 = 6");
    CHECK(ds4_e2m1fn_round(100.0f) == 6.0f, "e2m1 round 饱和");
    CHECK(ds4_fp4_nibble_to_f32(0x9) == -0.5f, "fp4 0x9 = -0.5");
    CHECK(signbit(ds4_fp4_nibble_to_f32(0x8)), "fp4 0x8 = -0");
    /* f16: 已知值 0x3C00=1.0, 0xC000=-2.0, 0x7C00=Inf, 次正规 0x0001=2^-24 */
    CHECK(ds4_f16_to_f32(0x3C00) == 1.0f, "f16 1.0");
    CHECK(ds4_f16_to_f32(0xC000) == -2.0f, "f16 -2.0");
    CHECK(isinf(ds4_f16_to_f32(0x7C00)), "f16 Inf");
    CHECK(ds4_f16_to_f32(0x0001) == ldexpf(1.0f, -24), "f16 次正规");
    CHECK(ds4_f64_to_f16(1.0) == 0x3C00, "f64→f16 1.0");
    CHECK(ds4_f64_to_f16(65536.0) == 0x7C00, "f64→f16 上溢=Inf");
    CHECK(ds4_bf16_to_f32(0x3F80) == 1.0f, "bf16 1.0");
}

/* ---- 3) GGUF 合成往返 ---- */
static void put_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void put_u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void put_str(FILE *f, const char *s) { put_u64(f, strlen(s)); fwrite(s, 1, strlen(s), f); }

static void test_gguf_roundtrip(void) {
    const char *path = "/tmp/ds4_unit_synth.gguf";
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL, "临时 gguf 写不开");
    if (!f) return;
    put_u32(f, 0x46554747u); put_u32(f, 3);      /* magic + v3 */
    put_u64(f, 2); put_u64(f, 2);                /* 2 张量, 2 KV */
    put_str(f, "general.name"); put_u32(f, 8); put_str(f, "synth");   /* KV: string */
    put_str(f, "synth.arr"); put_u32(f, 9); put_u32(f, 4); put_u64(f, 3);  /* KV: [u32×3] */
    put_u32(f, 1); put_u32(f, 2); put_u32(f, 3);
    put_str(f, "t.f32"); put_u32(f, 2); put_u64(f, 4); put_u64(f, 2);  /* [4,2] f32 @0 */
    put_u32(f, DS4_GGT_F32); put_u64(f, 0);
    put_str(f, "t.q8"); put_u32(f, 1); put_u64(f, 32);                 /* [32] q8_0 @32 */
    put_u32(f, DS4_GGT_Q8_0); put_u64(f, 32);
    long hdr_end = ftell(f);
    while (ftell(f) % 32) fputc(0, f);           /* alignment=32 */
    float fv[8]; for (int i = 0; i < 8; i++) { fv[i] = (float)i * 0.5f; }
    fwrite(fv, 4, 8, f);                          /* t.f32 @0, 32B */
    uint8_t q8[34]; uint16_t one = 0x3C00;        /* d=1.0 → 值 = 有符号字节原值 */
    memcpy(q8, &one, 2);
    for (int i = 0; i < 32; i++) q8[2 + i] = (uint8_t)(int8_t)(i - 16);
    fwrite(q8, 1, 34, f);                         /* t.q8 @32 */
    fclose(f);

    ds4_gguf g; char err[256];
    CHECK(ds4_gguf_open(&g, path, err, sizeof err) == 0, "gguf open: %s", err);
    CHECK(g.nt == 2, "张量数 2, 得 %d", g.nt);
    CHECK(g.data0 == (uint64_t)((hdr_end + 31) / 32 * 32), "data0 对齐错");
    const ds4_gguf_tensor *tf = ds4_gguf_find(&g, "t.f32");
    const ds4_gguf_tensor *tq = ds4_gguf_find(&g, "t.q8");
    CHECK(tf && tq, "找不到合成张量");
    CHECK(ds4_gguf_find(&g, "no.such") == NULL, "不存在的名字要 NULL");
    if (tf) {
        uint64_t nb = 0; const uint8_t *p = ds4_gguf_tensor_data(&g, tf, &nb);
        CHECK(p && nb == 32, "t.f32 数据 32B, 得 %llu", (unsigned long long)nb);
        CHECK(p && memcmp(p, fv, 32) == 0, "t.f32 数据不一致");
        CHECK(tf->ne[0] == 4 && tf->ne[1] == 2 && tf->ne[2] == 1, "t.f32 形状");
    }
    if (tq) {
        uint64_t nb = 0; const uint8_t *p = ds4_gguf_tensor_data(&g, tq, &nb);
        CHECK(p && nb == 34, "t.q8 数据 34B");
        float out[32];
        CHECK(p && ds4_deq_bytes(DS4_GGT_Q8_0, p, 32, out) == 0, "t.q8 dequant");
        int ok = 1;
        for (int i = 0; i < 32; i++) if (out[i] != (float)(i - 16)) ok = 0;
        CHECK(ok, "t.q8 dequant 值错(d=1.0 应还原有符号字节)");
    }
    ds4_gguf_close(&g);
    CHECK(g.map == NULL && g.t == NULL, "close 后要清零");
    ds4_gguf gbad;
    CHECK(ds4_gguf_open(&gbad, "/no/such/file.gguf", err, sizeof err) == -1, "不存在文件要 -1");
    remove(path);
}

int main(void) {
    test_deq_fixture(DS4_GGT_F32,     "f32",     32);
    test_deq_fixture(DS4_GGT_F16,     "f16",     32);
    test_deq_fixture(DS4_GGT_Q8_0,    "q8_0",    8);
    test_deq_fixture(DS4_GGT_Q2_K,    "q2_K",    8);
    test_deq_fixture(DS4_GGT_Q4_K,    "q4_K",    8);
    test_deq_fixture(DS4_GGT_IQ2_XXS, "iq2_xxs", 8);
    test_deq_fixture(DS4_GGT_BF16,    "bf16",    32);
    test_deq_error_paths();
    test_fp8();
    test_gguf_roundtrip();
    { extern int unit_zmod(void); g_fail += unit_zmod(); }   /* 引擎 z 双路对拍(2026-08-26) */
    if (g_fail) { fprintf(stderr, "ds4_unit: %d failure(s)\n", g_fail); return 1; }
    puts("ds4_unit: ok");
    return 0;
}
