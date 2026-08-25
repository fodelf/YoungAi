/* cap_raw2npy.c — 引擎 capture 原始分片 → 标定链的 npy 契约。C 实现(2026-08-22 铁律:
 * 算法与数据通路一律 C, 反修阶段与引擎阶段复用同一份口径; 禁 Python 另写一套 —— 早先的
 * Python 版就因为 BOS 行数差 1 造成整体错位一格, 不报错、照出数字)。
 *
 *   raw_ffn_in_L{L}        f16 [n×4096] → ffn_in_L{L}.npy       (<f2)
 *   raw_route_L{L}         i16 [n×6]    → route_L{L}.npy        (<i2)
 *   raw_route_w_L{L}       f16 [n×6]    → route_w_L{L}.npy      (<f2)
 *   raw_route_logits_L{L}  f16 [n×256]  → route_logits_L{L}.npy (<f2)   (可选)
 *   raw_ffn_out_L{L}       f16 [n×4096] → obase_v3_L{L}.npy     (<f2)   = O_BASE(学生)
 *
 * 行数一致性是硬门: 五种分片的 token 数必须相同, 不同即拒(错位是静默杀手)。
 * --ntok N 可截取前 N 行(与锚对齐); 不给则用各分片的公共行数。
 *
 * 用法: cap_raw2npy --raw DIR --out DIR [--layers a-b] [--ntok N]
 * 纯 C99, 无外部依赖。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>

static long file_bytes(const char *p) {
    struct stat st;
    return stat(p, &st) == 0 ? (long)st.st_size : -1;
}

/* v1.0 .npy: magic + version + 2 字节头长 + 空格补齐到 64 对齐的头 + 数据 */
static int write_npy(const char *path, const char *descr,
                     const int64_t *shape, int ndim,
                     const void *data, size_t nbytes) {
    char dict[256];
    int n = snprintf(dict, sizeof dict,
                     "{'descr': '%s', 'fortran_order': False, 'shape': (", descr);
    for (int i = 0; i < ndim; i++)
        n += snprintf(dict + n, sizeof dict - n, "%lld, ", (long long)shape[i]);
    n += snprintf(dict + n, sizeof dict - n, "), }");
    int base = 10 + n + 1;
    int pad  = (64 - (base % 64)) % 64;
    for (int i = 0; i < pad && n < (int)sizeof dict - 1; i++) dict[n++] = ' ';
    dict[n++] = '\n';
    int headerlen = n;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    unsigned char hdr[8] = { 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0 };
    unsigned char hl[2]  = { (unsigned char)(headerlen & 0xff),
                             (unsigned char)((headerlen >> 8) & 0xff) };
    int ok = fwrite(hdr, 1, 8, f) == 8 && fwrite(hl, 1, 2, f) == 2 &&
             fwrite(dict, 1, (size_t)headerlen, f) == (size_t)headerlen &&
             (nbytes == 0 || fwrite(data, 1, nbytes, f) == nbytes);
    fclose(f);
    return ok ? 0 : -1;
}

typedef struct { const char *raw, *out; const char *descr; int width; } kind_t;

int main(int argc, char **argv) {
    const char *raw_dir = NULL, *out_dir = NULL;
    int lo = 0, hi = 42, ntok_want = 0, keep_head = 0;   /* 默认保尾(BOS 掐头兼容) */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--raw") && i + 1 < argc) raw_dir = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_dir = argv[++i];
        else if (!strcmp(argv[i], "--ntok") && i + 1 < argc) ntok_want = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--keep") && i + 1 < argc) keep_head = !strcmp(argv[++i], "head");
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) {
            if (sscanf(argv[++i], "%d-%d", &lo, &hi) != 2) {
                fprintf(stderr, "--layers 需要 a-b\n"); return 2;
            }
        } else { fprintf(stderr, "未知参数 %s\n", argv[i]); return 2; }
    }
    if (!raw_dir || !out_dir) {
        fprintf(stderr, "用法: cap_raw2npy --raw DIR --out DIR [--layers a-b] [--ntok N]\n");
        return 2;
    }
    const kind_t kinds[] = {
        { "raw_ffn_in",       "ffn_in",       "<f2", 4096 },
        { "raw_route",        "route",        "<i2",    6 },
        { "raw_route_w",      "route_w",      "<f2",    6 },
        { "raw_route_logits", "route_logits", "<f2",  256 },
        { "raw_ffn_out",      "obase_v3",     "<f2", 4096 },   /* O_BASE: 学生 routed 输出 */
    };
    const int NK = (int)(sizeof kinds / sizeof kinds[0]);
    int done = 0;

    for (int L = lo; L <= hi; L++) {
        long rows[8]; int have[8]; long common = -1;
        for (int k = 0; k < NK; k++) {
            char p[1024];
            snprintf(p, sizeof p, "%s/%s_L%d", raw_dir, kinds[k].raw, L);
            long b = file_bytes(p);
            have[k] = (b > 0);
            if (!have[k]) { rows[k] = 0; continue; }
            int esz = kinds[k].descr[1] == 'f' ? 2 : 2;      /* f2/i2 均 2 字节 */
            long per = (long)kinds[k].width * esz;
            if (b % per) { fprintf(stderr, "L%d %s: %ld 字节不是行长 %ld 的整数倍\n",
                                   L, kinds[k].raw, b, per); return 3; }
            rows[k] = b / per;
            if (common < 0 || rows[k] < common) common = rows[k];
        }
        if (common < 0) continue;                              /* 该层没有任何分片 */
        /* ★行数一致性硬门★: 允许各分片行数相同, 或恰好差 1(引擎流首 BOS) —— 其余一律拒。*/
        for (int k = 0; k < NK; k++) {
            if (!have[k]) continue;
            if (rows[k] - common > 1) {
                fprintf(stderr, "L%d %s 行数 %ld 与公共行数 %ld 相差 >1 — 口径不明, 拒跑\n",
                        L, kinds[k].raw, rows[k], common); return 3;
            }
        }
        long ntok = ntok_want > 0 ? ntok_want : common;
        if (ntok > common) { fprintf(stderr, "L%d: --ntok %ld > 公共行数 %ld\n", L, ntok, common); return 3; }

        for (int k = 0; k < NK; k++) {
            if (!have[k]) continue;
            char pin[1024], pout[1024];
            snprintf(pin,  sizeof pin,  "%s/%s_L%d",     raw_dir, kinds[k].raw, L);
            snprintf(pout, sizeof pout, "%s/%s_L%d.npy", out_dir, kinds[k].out, L);
            /* 默认保尾(BOS 掐头场景); --keep head 保头(与前缀锚对齐场景 —— 2026-08-23
             * 事故: 8192→2048 保尾给出后 2048 行, 锚却是前 2048, 路由重合 0.41/6 现形)。 */
            long off_rows = keep_head ? 0 : rows[k] - ntok;
            size_t per = (size_t)kinds[k].width * 2u;
            size_t nb  = (size_t)ntok * per;
            FILE *f = fopen(pin, "rb");
            if (!f) { fprintf(stderr, "打不开 %s\n", pin); return 3; }
            if (off_rows > 0 && fseek(f, (long)((size_t)off_rows * per), SEEK_SET) != 0) {
                fclose(f); fprintf(stderr, "seek 失败 %s\n", pin); return 3;
            }
            void *buf = malloc(nb);
            if (!buf || fread(buf, 1, nb, f) != nb) {
                fclose(f); free(buf); fprintf(stderr, "读不满 %s\n", pin); return 3;
            }
            fclose(f);
            int64_t shape[2] = { ntok, kinds[k].width };
            if (write_npy(pout, kinds[k].descr, shape, 2, buf, nb) != 0) {
                free(buf); fprintf(stderr, "写失败 %s\n", pout); return 3;
            }
            free(buf);
        }
        printf("L%-3d ntok=%ld  (公共行 %ld)\n", L, ntok, common);
        done++;
    }
    printf("转换完成: %d 层\n", done);
    return done ? 0 : 1;
}
