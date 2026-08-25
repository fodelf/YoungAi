/* zchain_merge.c — zrec_LXX.bin → 引擎 DQZ2 zchain 合并器(C, 2026-08-23)。
 * 取代 zlever/zrec_to_zchain.py(铁律: 链上不留 Python)。语义逐行同源:
 *   zrec = 116B 头记录串(名字[16B] ... psz@88 u64), 载荷逐字节直通; 空文件=层闸(skip)。
 *   输出 = "DQZ2" u32 | n_layer u32 | per layer { L u32, n_ops u32, {ty u32, psz u32, pay}* }
 *   类型: 5=bf.GE(psz>=512) 6=zl.RRR 7=zl.AMP 8=zl.RTE 9=zl.AMPD (均 psz>=16)
 *   ★AMPD 必须在 AMP 之前判★: "zl.AMP" 是 "zl.AMPD" 的前缀, 反序会把动态 z 错标 type7,
 *   载荷 A|U|V 被引擎按 z|U|V 解析 → 跑出垃圾且不报错(2026-08-22 判例)。
 * 用法: zchain_merge <layers_dir> <out.bin> [nl=43] [--skip 5,7]  (--skip=组合裁剪对照) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static int read_file(const char *p, uint8_t **buf, long *sz) {
    FILE *f = fopen(p, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END); *sz = ftell(f); fseek(f, 0, SEEK_SET);
    *buf = malloc(*sz > 0 ? (size_t)*sz : 1);
    if (*sz > 0 && fread(*buf, 1, (size_t)*sz, f) != (size_t)*sz) { fclose(f); free(*buf); return -2; }
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "用法: zchain_merge <layers_dir> <out.bin> [nl] [--skip 5,7]\n"); return 2; }
    const char *ld = argv[1], *outp = argv[2];
    int NL = 43;
    int skip[16] = {0};
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--skip") && i + 1 < argc) {
            char b[64]; strncpy(b, argv[++i], sizeof b - 1); b[sizeof b - 1] = 0;
            for (char *t = strtok(b, ","); t; t = strtok(NULL, ",")) { int v = atoi(t); if (v >= 0 && v < 16) skip[v] = 1; }
        } else NL = atoi(argv[i]);
    }

    FILE *out = fopen(outp, "wb");
    if (!out) { fprintf(stderr, "%s 打不开\n", outp); return 3; }
    uint32_t magic = 0x325A5144u, nl = (uint32_t)NL;
    fwrite(&magic, 4, 1, out); fwrite(&nl, 4, 1, out);

    long tot[16] = {0}; int miss[64], nmiss = 0;
    long total_bytes = 8;
    for (int L = 0; L < NL; L++) {
        /* 两路: 主记录 + 路由侧车(可缺) */
        uint8_t *pays[4096]; uint32_t ptys[4096], plens[4096]; int nops = 0;
        int seen_any = 0;
        char p[1024];
        const char *pats[2] = {"%s/zrec_L%02d.bin", "%s/zrec_route_L%02d.bin"};
        for (int w = 0; w < 2; w++) {
            snprintf(p, sizeof p, pats[w], ld, L);
            uint8_t *raw; long sz;
            if (read_file(p, &raw, &sz)) continue;
            seen_any = 1;
            if (sz == 0) { free(raw); continue; }   /* 空文件 = 层被闸 */
            long off = 0;
            while (off + 116 <= sz) {
                char nm[17]; memcpy(nm, raw + off, 16); nm[16] = 0;
                uint64_t psz; memcpy(&psz, raw + off + 88, 8);
                if (off + 116 + (long)psz > sz) break;
                const uint8_t *pay = raw + off + 116;
                off += 116 + (long)psz;
                int ty = 0;
                if (strstr(nm, "bf.GE") && psz >= 512) ty = 5;
                else if (strstr(nm, "zl.RRR") && psz >= 16) ty = 6;
                else if (strstr(nm, "zl.AMPD") && psz >= 16) ty = 9;   /* 先 AMPD 后 AMP */
                else if (strstr(nm, "zl.AMP") && psz >= 16) ty = 7;
                else if (strstr(nm, "zl.RTE") && psz >= 16) ty = 8;
                else if (strstr(nm, "zl.HXP") && psz >= 12) ty = 10;   /* 层出口放大器 */
                if (ty && !skip[ty] && nops < 4096) {
                    pays[nops] = malloc(psz); memcpy(pays[nops], pay, psz);
                    ptys[nops] = (uint32_t)ty; plens[nops] = (uint32_t)psz;
                    nops++; tot[ty]++;
                }
            }
            free(raw);
        }
        if (!seen_any && nmiss < 64) miss[nmiss++] = L;
        uint32_t Lu = (uint32_t)L, no = (uint32_t)nops;
        fwrite(&Lu, 4, 1, out); fwrite(&no, 4, 1, out);
        total_bytes += 8;
        for (int i = 0; i < nops; i++) {
            fwrite(&ptys[i], 4, 1, out); fwrite(&plens[i], 4, 1, out);
            fwrite(pays[i], 1, plens[i], out);
            total_bytes += 8 + plens[i];
            free(pays[i]);
        }
    }
    fclose(out);
    printf("→ %s (%.1f MB) GE=%ld z^L=%ld AMP=%ld RTE=%ld AMPD=%ld HXP=%ld", outp,
           total_bytes / 1e6, tot[5], tot[6], tot[7], tot[8], tot[9], tot[10]);
    if (nmiss) {
        printf(" ★缺层 [");
        for (int i = 0; i < nmiss; i++) printf("%s%d", i ? ", " : "", miss[i]);
        printf("]★");
    }
    printf("\n");
    return 0;
}
