/* zrec_to_zchain.c — zrec_LXX.bin(zlayer INJ=2 外挂记录)→ 引擎 DQZ2 zchain。
 * (C, 2026-08-25 Python→C 迁移 Wave B; 取代 zlever/zrec_to_zchain.py, 输出逐字节同)
 * zrec 内容 = 116B 头记录串(bf.GE/zl.RRR/...), 载荷逐字节直通; 空文件=该层被闸(skip)。
 * ★AMPD 必须在 AMP 之前判★: 'zl.AMP' 是 'zl.AMPD' 的子串, 顺序反了会把动态 z 侧车
 * 错标成 type7 —— 载荷 A|U|V 按 z|U|V 解析, 引擎跑出垃圾且不报错(2026-08-22 教训)。
 * --skip 5,7: 合并时按类型过滤(5=GE 6=z^L 7=AMP 8=RTE 9=AMPD), 组合裁剪对照用。
 * 用法: zrec_to_zchain <layers_dir> <out.zchain.bin> [nl=43] [--skip TYPES] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "../../src/common/ds4_amp_fmt.h"   /* 116 记录头契约(单一定义源) */

static void put32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "用法: zrec_to_zchain <layers_dir> <out.zchain.bin> [nl=43] [--skip TYPES]\n"); return 1; }
    const char *ld = argv[1], *outp = argv[2];
    int NL = (argc > 3 && strncmp(argv[3], "--", 2)) ? atoi(argv[3]) : 43;
    int skip[16] = {0};
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--skip")) continue;
        if (i + 1 >= argc) { fprintf(stderr, "--skip 缺值(逗号分隔类型号, 如 5,7)\n"); return 1; }
        char b[128]; snprintf(b, sizeof b, "%s", argv[++i]);
        for (char *tk = strtok(b, ","); tk; tk = strtok(NULL, ","))
            { int t = atoi(tk); if (t >= 0 && t < 16) skip[t] = 1; }
    }
    FILE *out = fopen(outp, "wb");
    if (!out) { fprintf(stderr, "%s 打不开\n", outp); return 2; }
    put32(out, 0x325A5144u); put32(out, (uint32_t)NL);
    long long tot[10] = {0}, outsz = 8;
    int miss[64], nmiss = 0;
    for (int L = 0; L < NL; L++) {
        /* 两路: 主记录 + 路由侧车(type8); 先收集本层全部 ops 再写(与 py 同序) */
        typedef struct { uint32_t ty; uint64_t psz; uint8_t *pay; } op_t;
        op_t ops[64]; int nops = 0, seen_any = 0;
        char fn[2][64];
        snprintf(fn[0], 64, "zrec_L%02d.bin", L);
        snprintf(fn[1], 64, "zrec_route_L%02d.bin", L);
        for (int fi = 0; fi < 2; fi++) {
            char p[1024]; snprintf(p, sizeof p, "%s/%s", ld, fn[fi]);
            FILE *f = fopen(p, "rb");
            if (!f) continue;
            seen_any = 1;
            fseek(f, 0, SEEK_END); long long sz = ftell(f); fseek(f, 0, SEEK_SET);
            if (sz == 0) { fclose(f); continue; }
            uint8_t *raw = malloc(sz);
            if (fread(raw, 1, sz, f) != (size_t)sz) { fclose(f); free(raw); continue; }
            fclose(f);
            long long off = 0;
            while (off + DS4_AMP_REC_HDR <= sz) {
                char nm[17]; memcpy(nm, raw + off, 16); nm[16] = 0;
                uint64_t psz; memcpy(&psz, raw + off + DS4_AMP_REC_OFF_PSZ, 8);
                uint8_t *pay = raw + off + DS4_AMP_REC_HDR;
                off += DS4_AMP_REC_HDR + (long long)psz;
                uint32_t ty = 0;   /* AMPD 判在 AMP 之前(子串陷阱, 见文件头) */
                if (strstr(nm, "bf.GE") && psz >= 512) ty = 5;
                else if (strstr(nm, "zl.RRR") && psz >= 16) ty = 6;
                else if (strstr(nm, "zl.AMPD") && psz >= 16) ty = 9;
                else if (strstr(nm, "zl.AMP") && psz >= 16) ty = 7;
                else if (strstr(nm, "zl.RTE") && psz >= 16) ty = 8;
                else if (!strncmp(nm, "sup.", 4) || !strncmp(nm, "bf.", 3) || !strncmp(nm, "bwd", 3)) {
                    /* sweep 侧车化(2026-09-01)后 zrec 里会出现 bf.GL、bf.GLdyn 族、sup.* 等
                     * 本工具不认的记录。静默丢=转出的链≠判决的链(假账), 硬停:
                     * 含 sweep 记录的层件用 `ds4quant_run --zchain-only` 导链(lfile 全语义)。 */
                    fprintf(stderr, "zrec_to_zchain: L%d 记录 %.16s 是 sweep 落地(本工具不认), "
                                    "改用 ds4quant_run --zchain-only 导链\n", L, nm);
                    exit(1);
                }
                if (ty && !skip[ty]) {
                    if (nops >= 64) {   /* 静默丢修正=转出的 zchain≠盘上 zrec, 停车 */
                        fprintf(stderr, "zrec_to_zchain: L%d op 超容量 64, 拒绝静默丢; 提容量重编\n", L);
                        exit(1);
                    }
                    ops[nops].ty = ty; ops[nops].psz = psz;
                    ops[nops].pay = malloc(psz); memcpy(ops[nops].pay, pay, psz);
                    nops++; tot[ty]++;
                }
            }
            free(raw);
        }
        if (!seen_any && nmiss < 64) miss[nmiss++] = L;
        put32(out, (uint32_t)L); put32(out, (uint32_t)nops); outsz += 8;
        for (int i = 0; i < nops; i++) {
            put32(out, ops[i].ty); put32(out, (uint32_t)ops[i].psz);
            fwrite(ops[i].pay, 1, ops[i].psz, out);
            outsz += 8 + (long long)ops[i].psz;
            free(ops[i].pay);
        }
    }
    fclose(out);
    printf("→ %s (%.1f MB) GE=%lld z^L=%lld AMP=%lld RTE=%lld AMPD=%lld",
           outp, outsz / 1e6, tot[5], tot[6], tot[7], tot[8], tot[9]);
    if (nmiss) {
        printf(" ★缺层 [");
        for (int i = 0; i < nmiss; i++) printf("%s%d", i ? ", " : "", miss[i]);
        printf("]★");
    }
    printf("\n");
    return 0;
}
