/* route_bias_rebake.c — 让运行时路由 == 量化时路由(增量原位: bias −= ca·Δb_champ, += ra·Δb_new)。
 * (C, 2026-08-25 Python→C 迁移; 取代 scripts/route_bias_rebake.py, 就地改写字节逐位同)
 * 口径差(照抄 py): 冠军侧车存"累计和"(消费端×α, 且 mincnt 门在此处判);
 *   新侧车由 rb_save 落盘的是 RB_APPLY(已/cnt 均值+门内写0) ⇒ 直接×α不再除 cnt。
 * 用法: route_bias_rebake <model.gguf> <champ_rb.bin> <champ_alpha> <new_rb.bin> <new_alpha> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define NL 43
#define NEXP 256
#define RB_MINCNT 8

static void pfloat(char *buf, double v) {
    if (v == (double)(long long)v && v < 1e16 && v > -1e16) {
        snprintf(buf, 64, "%lld.0", (long long)v);
        if (strtod(buf, NULL) == v) return;
    }
    for (int p = 1; p <= 17; p++) { snprintf(buf, 64, "%.*g", p, v); if (strtod(buf, NULL) == v) break; }
}
static void load_rb(const char *path, float *acc, uint32_t *cnt) {
    FILE *g = fopen(path, "rb");
    if (!g) { fprintf(stderr, "%s 打不开\n", path); exit(2); }
    uint32_t hd[4];
    if (fread(hd, 4, 4, g) != 4 || hd[0] != 0x41494252u || hd[1] != NL || hd[2] != NEXP) {
        fprintf(stderr, "AssertionError: %s RBIA 头不对\n", path); exit(1); }
    if (fread(acc, 4, NL * NEXP, g) != NL * NEXP) exit(2);
    if (cnt && fread(cnt, 4, NL * NEXP, g) != NL * NEXP) exit(2);
    fclose(g);
}
static uint64_t rs_skip(FILE *f) {   /* 读长度并跳过字符串, 返回长度 */
    uint64_t n; if (fread(&n, 8, 1, f) != 1) exit(2); fseek(f, (long)n, SEEK_CUR); return n;
}
static void sv(FILE *f, uint32_t t);
static void sv(FILE *f, uint32_t t) {
    static const int sz[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
    if (t < 13 && sz[t]) { fseek(f, sz[t], SEEK_CUR); return; }
    if (t == 8) { rs_skip(f); return; }
    if (t == 9) { uint32_t et; uint64_t n; if (fread(&et,4,1,f)!=1||fread(&n,8,1,f)!=1) exit(2);
        for (uint64_t i = 0; i < n; i++) sv(f, et); }
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "用法: route_bias_rebake <model.gguf> <champ_rb> <ca> <new_rb> <ra>\n"); return 1; }
    float cacc[NL*NEXP], racc[NL*NEXP]; uint32_t ccnt[NL*NEXP];
    double ca = atof(argv[3]), ra = atof(argv[5]);
    load_rb(argv[2], cacc, ccnt);
    load_rb(argv[4], racc, NULL);   /* 新侧车 cnt 段不用(RB_APPLY 已门) — 与 py 读了不用等价 */
    FILE *f = fopen(argv[1], "r+b");
    if (!f) { fprintf(stderr, "%s 打不开\n", argv[1]); return 2; }
    fseek(f, 8, SEEK_SET);
    uint64_t n_t, n_kv;
    if (fread(&n_t, 8, 1, f) != 1 || fread(&n_kv, 8, 1, f) != 1) return 2;
    uint32_t align = 32;
    for (uint64_t k = 0; k < n_kv; k++) {
        uint64_t kn; if (fread(&kn, 8, 1, f) != 1) return 2;
        char key[256]; size_t rl = kn < 255 ? kn : 255;
        if (fread(key, 1, rl, f) != rl) return 2; key[rl] = 0;
        if (kn > rl) fseek(f, (long)(kn - rl), SEEK_CUR);
        uint32_t t; if (fread(&t, 4, 1, f) != 1) return 2;
        if (!strcmp(key, "general.alignment")) { if (fread(&align, 4, 1, f) != 1) return 2; }
        else sv(f, t);
    }
    struct { int L; uint64_t off; } tgt[64]; int ntgt = 0;
    for (uint64_t i = 0; i < n_t; i++) {
        uint64_t nn; if (fread(&nn, 8, 1, f) != 1) return 2;
        char name[512]; size_t rl = nn < 511 ? nn : 511;
        if (fread(name, 1, rl, f) != rl) return 2; name[rl] = 0;
        if (nn > rl) fseek(f, (long)(nn - rl), SEEK_CUR);
        uint32_t nd; if (fread(&nd, 4, 1, f) != 1) return 2;
        fseek(f, 8 * nd, SEEK_CUR);
        uint32_t ty; uint64_t off;
        if (fread(&ty, 4, 1, f) != 1 || fread(&off, 8, 1, f) != 1) return 2;
        size_t ln = strlen(name);
        if (ln > 17 && !strcmp(name + ln - 17, ".exp_probs_b.bias")) {
            if (ty != 0) { fprintf(stderr, "AssertionError: %s bias 非 f32\n", name); return 1; }
            int L = atoi(name + 4);   /* "blk.<L>." */
            if (ntgt < 64) { tgt[ntgt].L = L; tgt[ntgt].off = off; ntgt++; }
        }
    }
    long long data_start = (ftell(f) + align - 1) / align * align;
    /* 按 L 升序(py sorted) */
    for (int a = 1; a < ntgt; a++) { int b = a - 1; typeof(tgt[0]) x = tgt[a];
        while (b >= 0 && tgt[b].L > x.L) { tgt[b+1] = tgt[b]; b--; } tgt[b+1] = x; }
    long long nsub = 0, nadd = 0;
    for (int t = 0; t < ntgt; t++) {
        int L = tgt[t].L;
        float vals[NEXP];
        fseek(f, (long)(data_start + tgt[t].off), SEEK_SET);
        if (fread(vals, 4, NEXP, f) != NEXP) return 2;
        for (int e = 0; e < NEXP; e++) {
            int i = L * NEXP + e;
            double v = vals[e];   /* py: f64 运算, pack 时一次舍回 f32 */
            if (ccnt[i] >= RB_MINCNT && cacc[i] != 0.0f) { v -= ca * (double)cacc[i]; nsub++; }
            if (racc[i] != 0.0f) { v += ra * (double)racc[i]; nadd++; }
            vals[e] = (float)v;
        }
        fseek(f, (long)(data_start + tgt[t].off), SEEK_SET);
        fwrite(vals, 4, NEXP, f);
    }
    fclose(f);
    char cas[64], ras[64]; pfloat(cas, ca); pfloat(ras, ra);
    printf("REBAKE 冠军偏置减 %lld 槽(α=%s) / R36 偏置加 %lld 槽(α=%s) → 运行时路由 == 量化时路由\n",
           nsub, cas, nadd, ras);
    return 0;
}
