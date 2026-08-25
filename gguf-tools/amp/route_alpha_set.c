/* route_alpha_set.c — 幂等地把模型 exp_probs_b 设成 [裸态 + α·Δb]。
 * (C, 2026-08-25 Python→C 迁移; 取代 scripts/route_alpha_set.py, 就地改写字节逐位同)
 * 幂等机制(照抄 py): 首次运行把当前 bias 存快照 <model>.bias0.bin(裸态), 之后每次从
 * 快照绝对重写, α 来回扫不累积。rb 口径=RB_APPLY(已/cnt+mincnt 门内写0) ⇒ 直接×α。
 * 用法: route_alpha_set <model.gguf> <rb.bin> <alpha> [--snapshot-only] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>

#define NEXP 256

static void pfloat(char *buf, double v) {
    if (v == (double)(long long)v && v < 1e16 && v > -1e16) {
        snprintf(buf, 64, "%lld.0", (long long)v);
        if (strtod(buf, NULL) == v) return;
    }
    for (int p = 1; p <= 17; p++) { snprintf(buf, 64, "%.*g", p, v); if (strtod(buf, NULL) == v) break; }
}
static void sv(FILE *f, uint32_t t);
static void sv(FILE *f, uint32_t t) {
    static const int sz[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
    if (t < 13 && sz[t]) { fseek(f, sz[t], SEEK_CUR); return; }
    if (t == 8) { uint64_t n; if (fread(&n,8,1,f)!=1) exit(2); fseek(f,(long)n,SEEK_CUR); return; }
    if (t == 9) { uint32_t et; uint64_t n; if (fread(&et,4,1,f)!=1||fread(&n,8,1,f)!=1) exit(2);
        for (uint64_t i = 0; i < n; i++) sv(f, et); }
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "用法: route_alpha_set <model.gguf> <rb.bin> <alpha> [--snapshot-only]\n"); return 1; }
    const char *model = argv[1], *rbp = argv[2];
    double alpha = atof(argv[3]);
    int snap_only = 0;
    for (int i = 4; i < argc; i++) if (!strcmp(argv[i], "--snapshot-only")) snap_only = 1;

    uint32_t nl, ne; float *acc;
    { FILE *g = fopen(rbp, "rb");
      if (!g) { fprintf(stderr, "%s 打不开\n", rbp); return 2; }
      uint32_t hd[4];
      if (fread(hd, 4, 4, g) != 4 || hd[0] != 0x41494252u) {
          fprintf(stderr, "AssertionError: %s 非 RBIA\n", rbp); return 1; }
      nl = hd[1]; ne = hd[2];
      acc = malloc((size_t)nl * ne * 4);
      if (fread(acc, 4, (size_t)nl * ne, g) != (size_t)nl * ne) return 2;
      fclose(g); }

    FILE *f = fopen(model, "r+b");
    if (!f) { fprintf(stderr, "%s 打不开\n", model); return 2; }
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
            if (ty != 0) { fprintf(stderr, "AssertionError: %s 非 f32\n", name); return 1; }
            if (ntgt < 64) { tgt[ntgt].L = atoi(name + 4); tgt[ntgt].off = off; ntgt++; }
        }
    }
    long long data_start = (ftell(f) + align - 1) / align * align;
    for (int a = 1; a < ntgt; a++) { int b = a - 1; typeof(tgt[0]) x = tgt[a];
        while (b >= 0 && tgt[b].L > x.L) { tgt[b+1] = tgt[b]; b--; } tgt[b+1] = x; }

    char snap[1100]; snprintf(snap, sizeof snap, "%s.bias0.bin", model);
    struct stat st;
    if (stat(snap, &st) != 0) {
        FILE *s = fopen(snap, "wb");
        if (!s) { fprintf(stderr, "%s 写不开\n", snap); return 2; }
        uint32_t n = (uint32_t)ntgt; fwrite(&n, 4, 1, s);
        for (int t = 0; t < ntgt; t++) {
            float vals[NEXP];
            fseek(f, (long)(data_start + tgt[t].off), SEEK_SET);
            if (fread(vals, 4, NEXP, f) != NEXP) return 2;
            uint32_t L = (uint32_t)tgt[t].L; fwrite(&L, 4, 1, s);
            fwrite(vals, 4, NEXP, s);
        }
        fclose(s);
        printf("[alpha] 裸态快照 → %s (%d 层)\n", snap, ntgt);
        if (snap_only) { fclose(f); return 0; }
    }
    /* 读快照 */
    int nbase = 0; struct { int L; float v[NEXP]; } base[64];
    { FILE *s = fopen(snap, "rb");
      if (!s) { fprintf(stderr, "%s 打不开\n", snap); return 2; }
      uint32_t n; if (fread(&n, 4, 1, s) != 1) return 2;
      for (uint32_t i = 0; i < n && i < 64; i++) {
          uint32_t L; if (fread(&L, 4, 1, s) != 1) return 2;
          base[nbase].L = (int)L;
          if (fread(base[nbase].v, 4, NEXP, s) != NEXP) return 2;
          nbase++; }
      fclose(s); }
    long long nset = 0;
    for (int t = 0; t < ntgt; t++) {
        int L = tgt[t].L, bi = -1;
        for (int j = 0; j < nbase; j++) if (base[j].L == L) { bi = j; break; }
        if (bi < 0) continue;
        float vals[NEXP]; memcpy(vals, base[bi].v, sizeof vals);
        if (L < (int)nl) {
            int lim = NEXP < (int)ne ? NEXP : (int)ne;
            for (int e = 0; e < lim; e++) {
                float a = acc[(size_t)L * ne + e];
                if (a != 0.0f) { vals[e] = (float)((double)vals[e] + alpha * (double)a); nset++; }
            }
        }
        fseek(f, (long)(data_start + tgt[t].off), SEEK_SET);
        fwrite(vals, 4, NEXP, f);
    }
    fclose(f);
    char as[64]; pfloat(as, alpha);
    printf("[alpha] α=%s 写入 %lld 槽(从裸态快照绝对重写, 不累积)\n", as, nset);
    return 0;
}
