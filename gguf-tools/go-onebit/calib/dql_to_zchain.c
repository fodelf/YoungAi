/* dql_to_zchain.c — 从 dql 主文件内嵌 vd=1 记录(zlayer 注入的 bf.GE/zl.RRR)抽出引擎
 * DQZ2 外挂 zchain(冠军侧车挂载形态)。载荷逐字节直通。
 * (C, 2026-08-25 Python→C 迁移 Wave B; 取代 zlever/dql_to_zchain.py, 输出逐字节同)
 * 用法: dql_to_zchain <layers_dir> <out.zchain.bin> [nl=43] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static void put32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "用法: dql_to_zchain <layers_dir> <out.zchain.bin> [nl=43]\n"); return 1; }
    const char *ld = argv[1], *outp = argv[2];
    int NL = argc > 3 ? atoi(argv[3]) : 43;
    FILE *out = fopen(outp, "wb");
    if (!out) { fprintf(stderr, "%s 打不开\n", outp); return 2; }
    put32(out, 0x325A5144u); put32(out, (uint32_t)NL);
    long long tot5 = 0, tot6 = 0, outsz = 8;
    for (int L = 0; L < NL; L++) {
        char p[1024]; snprintf(p, sizeof p, "%s/dql_L%02d.bin", ld, L);
        FILE *f = fopen(p, "rb");
        if (!f) { fprintf(stderr, "%s 打不开\n", p); return 2; }
        fseek(f, 0, SEEK_END); long long sz = ftell(f); fseek(f, 0, SEEK_SET);
        uint8_t hd[12];
        if (fread(hd, 1, 12, f) != 12 || memcmp(hd, "DQL2", 4)) { fprintf(stderr, "%s: 非 DQL2\n", p); return 2; }
        uint32_t nrec; memcpy(&nrec, hd + 8, 4);
        typedef struct { uint32_t ty; uint64_t psz; uint8_t *pay; } op_t;
        op_t ops[64]; int nops = 0;
        long long off = 12;
        for (uint32_t r = 0; r < nrec && off + 116 <= sz; r++) {
            uint8_t rh[116];
            fseek(f, (long)off, SEEK_SET);
            if (fread(rh, 1, 116, f) != 116) break;
            char nm[17]; memcpy(nm, rh, 16); nm[16] = 0;
            uint64_t psz; memcpy(&psz, rh + 88, 8);
            int32_t vd; memcpy(&vd, rh + 112, 4);
            long long payoff = off + 116;
            off += 116 + (long long)psz;
            if (vd != 1) continue;
            uint32_t ty = 0;
            if (strstr(nm, "bf.GE") && psz >= 512) { ty = 5; tot5++; }
            else if (strstr(nm, "zl.RRR") && psz >= 16) { ty = 6; tot6++; }
            if (!ty || nops >= 64) continue;
            ops[nops].ty = ty; ops[nops].psz = psz;
            ops[nops].pay = malloc(psz);
            fseek(f, (long)payoff, SEEK_SET);
            if (fread(ops[nops].pay, 1, psz, f) != psz) { fprintf(stderr, "%s: 载荷读不满\n", p); return 2; }
            nops++;
        }
        fclose(f);
        put32(out, (uint32_t)L); put32(out, (uint32_t)nops); outsz += 8;
        for (int i = 0; i < nops; i++) {
            put32(out, ops[i].ty); put32(out, (uint32_t)ops[i].psz);
            fwrite(ops[i].pay, 1, ops[i].psz, out);
            outsz += 8 + (long long)ops[i].psz;
            free(ops[i].pay);
        }
    }
    fclose(out);
    printf("→ %s (%.1f MB) GE=%lld z^L=%lld\n", outp, outsz / 1e6, tot5, tot6);
    return 0;
}
