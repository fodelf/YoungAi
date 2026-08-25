/* vq_merge_v4.c — v4 战役产物合并(2026-08-25 Python→C 迁移 Wave A): 换代式重打包+路由偏置烘焙。
 * 忠实转录 quant/vq_merge_v4.py(v4.1), 逐位同语义, 不改进不重构。
 *
 * 新一代合一文件 = 骨架(非专家张量, 取自现役 base, gate.bias 烘入 α·Δb 路由偏置)
 *               + 新 down(M1 dql_L*.bin 的 D 段, ssh 流式)
 *               + 新 blob(M1 dql_vq_L*.bin 真载荷修剪, ssh 流式, type42)。
 * blob 真尺寸由 scripts/vq_blob_truesize.py 预扫文件 --blob-sizes 提供(修剪预留空洞);
 * --route-bias/--route-alpha 把 RBIA 侧车(ds4quant_run DS4_ROUTE_BIAS_FIT 产物)按 mincnt
 * 门后 α 缩放烘进 blk.L.exp_probs_b.bias(F32 256, 只影响专家选择不影响混合权重)。
 * down 与 blob 必须同代(冷 w2 的 hc 顺序补偿是对新 q1/q3 算的)。
 *
 * 两阶段:
 *   --extract-skeleton: 从 base 抽非专家张量(~8.6G; 不含 down/blob/opt) → skeleton.gguf
 *   --merge: skeleton + ssh(M1 blobs 修剪流式) + ssh(M1 dql D 段) [+gate.bias 烘焙] → 输出
 * 用法:
 *   vq_merge_v4 --extract-skeleton --base ds4-vq22.gguf --out skeleton.gguf
 *   vq_merge_v4 --merge --skeleton skeleton.gguf --blob-sizes blob_sizes.txt \
 *       --route-bias route_bias_v4.bin --route-alpha 2.5 \
 *       --dql-host 192.168.1.2 --dql-dir /Users/fodelf/ds4-main/gguf/go-onebit/layers \
 *       --out ds4-vq4bf.gguf
 *
 * ★金标口径(硬闸)★: 同参数同输入下与 vq_merge_v4.py 的输出 GGUF 逐字节 md5 相同 ——
 *   ① --extract-skeleton;
 *   ② --merge 走本地 dql 目录模式(--dql-host local/""/-/127.0.0.1/localhost)。
 * 歧义点与处理决定见 migrate/vq_merge_v4_transcription_notes.md。
 *
 * 编译: gcc -O3 -o vq_merge_v4 vq_merge_v4.c -lm
 * 前提: 小端主机(GGUF/dql/RBIA 全部小端, 与 .py struct "<" 同; 大端机不支持)。
 */
#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <math.h>
#include <errno.h>
#include <unistd.h>
#include <sys/stat.h>

#define ALIGN            32
#define N_LAYER          43
#define NEXP             256
#define SZ_G             (2048ULL * 16 * 34)      /* dql G/U 段每专家字节 = 1,114,112 */
#define SZ_D             (4096ULL * 8 * 34)       /* dql D 段每专家字节 = 1,114,112 */
#define DQL_HDR          35104                    /* dql_L*.bin 头(实测 dql_L00); 只作文档常量,
                                                   * 真 D 偏移一律走 --down-offsets 逐层文件 */
#define DOWN_LAYER_BYTES ((uint64_t)NEXP * SZ_D)  /* 285,212,672 */
#define RB_MINCNT        8                        /* 与 ds4quant_run RB_MINCNT 同步 */
#define REC_HDR          116

#define CHUNK_SKEL (1u << 26)   /* .py: src.read(min(1 << 26, left)) */
#define CHUNK_STRM (1u << 24)   /* .py: f.read(min(1 << 24, left)) / p.stdout.read(1 << 24) */

/* ---------------- 基础工具 ---------------- */

/* .py 里的 assert/异常一律终止进程; 这里统一 exit(1), 消息带 Python 异常名前缀便于 grep 对拍 */
static void die(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fputc('\n', stderr); exit(1);
}
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("MemoryError: malloc %zu", n); return p; }
static void *xrealloc(void *p, size_t n) { void *q = realloc(p, n ? n : 1); if (!q) die("MemoryError: realloc %zu", n); return q; }
static char *xstrdup(const char *s) { size_t n = strlen(s) + 1; char *d = xmalloc(n); memcpy(d, s, n); return d; }

/* 饱和加: .py 用大整数, 溢出不可能; C 侧饱和后一样必然触发上界 break, 判定等价 */
static uint64_t sat_add(uint64_t a, uint64_t b) { uint64_t s = a + b; return s < a ? UINT64_MAX : s; }
static uint64_t align_up(uint64_t v) { return (v + ALIGN - 1) / ALIGN * ALIGN; }

typedef struct { uint8_t *p; size_t n, cap; } buf_t;
static void buf_add(buf_t *b, const void *d, size_t n) {
    if (b->n + n > b->cap) { b->cap = (b->n + n) * 2 + 64; b->p = xrealloc(b->p, b->cap); }
    if (n) memcpy(b->p + b->n, d, n);
    b->n += n;
}
static void buf_u32(buf_t *b, uint32_t v) { buf_add(b, &v, 4); }
static void buf_u64(buf_t *b, uint64_t v) { buf_add(b, &v, 8); }

static uint64_t fsize(const char *p) {
    struct stat st;
    if (stat(p, &st)) die("FileNotFoundError: %s: %s", p, strerror(errno));
    return (uint64_t)st.st_size;
}
static int fexists(const char *p) { return access(p, F_OK) == 0; }

static uint8_t *read_all(const char *p, size_t *len) {
    FILE *f = fopen(p, "rb");
    if (!f) die("FileNotFoundError: %s: %s", p, strerror(errno));
    fseeko(f, 0, SEEK_END); off_t sz = ftello(f); fseeko(f, 0, SEEK_SET);
    uint8_t *b = xmalloc((size_t)sz + 1);
    if (sz && fread(b, 1, (size_t)sz, f) != (size_t)sz) die("IOError: 短读 %s", p);
    fclose(f); *len = (size_t)sz; return b;
}

/* float16 → float32(位精确, 含次正规/inf/nan 尾数保位; 与 numpy .astype(f32) 同) */
static float f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023, o;
    if (e == 31) o = s | 0x7F800000u | (m << 13);
    else if (e == 0) {
        if (!m) o = s;
        else { int sh = 0; while (!(m & 1024)) { m <<= 1; sh++; }
               m &= 1023; o = s | ((uint32_t)(113 - sh) << 23) | (m << 13); }
    } else o = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &o, 4); return f;
}

/* Python repr(float): 最短往返表示 + 整数值补 ".0"; 指数形式条件 decpt<=-4 || decpt>16。
 * 只用于 [route-bias] 那行 α 打印 —— 不进文件字节, 但对拍 stdout 时要一致。 */
static const char *py_float_repr(double v) {
    static char out[64], e[64];
    if (isnan(v)) { strcpy(out, "nan"); return out; }
    if (isinf(v)) { strcpy(out, v > 0 ? "inf" : "-inf"); return out; }
    int p;
    for (p = 0; p <= 17; p++) { snprintf(e, sizeof e, "%.*e", p, v); if (strtod(e, NULL) == v) break; }
    const char *s = e; int neg = 0;
    if (*s == '-') { neg = 1; s++; }
    char dig[40]; int nd = 0;
    dig[nd++] = *s++;
    if (*s == '.') { s++; while (*s != 'e' && *s != 'E') dig[nd++] = *s++; }
    while (*s != 'e' && *s != 'E') s++;
    int decpt = atoi(s + 1) + 1;
    while (nd > 1 && dig[nd - 1] == '0') nd--;
    dig[nd] = 0;
    char *o = out;
    if (neg) *o++ = '-';
    if (decpt <= -4 || decpt > 16) {
        *o++ = dig[0];
        if (nd > 1) { *o++ = '.'; memcpy(o, dig + 1, nd - 1); o += nd - 1; }
        o += sprintf(o, "e%+03d", decpt - 1);
    } else if (decpt <= 0) {
        *o++ = '0'; *o++ = '.';
        for (int i = 0; i < -decpt; i++) *o++ = '0';
        memcpy(o, dig, nd); o += nd;
    } else if (decpt >= nd) {
        memcpy(o, dig, nd); o += nd;
        for (int i = 0; i < decpt - nd; i++) *o++ = '0';
        *o++ = '.'; *o++ = '0';
    } else {
        memcpy(o, dig, decpt); o += decpt; *o++ = '.';
        memcpy(o, dig + decpt, nd - decpt); o += nd - decpt;
    }
    *o = 0; return out;
}

/* nm.split(".")[idx] */
static const char *name_field(const char *nm, int idx, char *out, size_t osz) {
    const char *s = nm;
    for (int k = 0; k < idx; k++) {
        const char *d = strchr(s, '.');
        if (!d) die("IndexError: list index out of range: %s [%d]", nm, idx);
        s = d + 1;
    }
    const char *e2 = strchr(s, '.');
    size_t n = e2 ? (size_t)(e2 - s) : strlen(s);
    if (n >= osz) n = osz - 1;
    memcpy(out, s, n); out[n] = 0; return out;
}
static int name_field_int(const char *nm, int idx) { char t[64]; name_field(nm, idx, t, sizeof t); return atoi(t); }

static int str_endswith(const char *s, const char *suf) {
    size_t n = strlen(s), m = strlen(suf);
    return m <= n && !memcmp(s + n - m, suf, m);
}
static int str_startswith(const char *s, const char *pre) { return !strncmp(s, pre, strlen(pre)); }

/* dict[int]->int64: 保 .py 的 dict 语义(同键覆盖 / len() 计唯一键 / 缺键 KeyError) */
typedef struct { long long *k, *v; size_t n, cap; } imap_t;
static void imap_set(imap_t *m, long long k, long long v) {
    for (size_t i = 0; i < m->n; i++) if (m->k[i] == k) { m->v[i] = v; return; }
    if (m->n == m->cap) { m->cap = m->cap * 2 + 16; m->k = xrealloc(m->k, m->cap * sizeof *m->k); m->v = xrealloc(m->v, m->cap * sizeof *m->v); }
    m->k[m->n] = k; m->v[m->n] = v; m->n++;
}
static long long imap_get(const imap_t *m, long long k) {
    for (size_t i = 0; i < m->n; i++) if (m->k[i] == k) return m->v[i];
    die("KeyError: %lld", k); return 0;
}

/* ---------------- 张量/条目 ---------------- */

enum { K_SKEL = 0, K_OPT, K_GUD, K_STUB, K_DOWN, K_BLOB, K_SRC };

typedef struct {
    char    *name;
    uint64_t*ne;
    uint32_t nd;
    uint32_t type;
    uint64_t off;        /* 解析时=源数据段内偏移; write_gguf 内重算=目的偏移 */
    uint64_t bytes;
    int      kind;
    uint64_t src_off;
    uint64_t skip;
    int      L;
    int      last;
    uint8_t *data;       /* kind=K_OPT 的内嵌载荷 */
    uint32_t idx;        /* sorted() 稳定性: 同 off 时保持原序 */
} ent_t;

typedef struct { ent_t *v; size_t n, cap; } elist_t;
static ent_t *el_push(elist_t *l) {
    if (l->n == l->cap) { l->cap = l->cap * 2 + 64; l->v = xrealloc(l->v, l->cap * sizeof *l->v); }
    ent_t *e = &l->v[l->n]; memset(e, 0, sizeof *e); e->idx = (uint32_t)l->n; l->n++; return e;
}
static ent_t *el_push1(elist_t *l, const char *name, uint64_t ne0, uint32_t type, uint64_t bytes, int kind) {
    ent_t *e = el_push(l);
    e->name = xstrdup(name); e->nd = 1; e->ne = xmalloc(8); e->ne[0] = ne0;
    e->type = type; e->bytes = bytes; e->kind = kind; return e;
}

/* ---------------- GGUF 读 ---------------- */

static void rdn(FILE *f, void *dst, size_t n) { if (n && fread(dst, 1, n, f) != n) die("AssertionError: 短读 %zu B", n); }
static uint32_t rdu32(FILE *f) { uint32_t v; rdn(f, &v, 4); return v; }
static uint64_t rdu64(FILE *f) { uint64_t v; rdn(f, &v, 8); return v; }
static char *rdstr(FILE *f) {
    uint64_t n = rdu64(f);
    char *s = xmalloc((size_t)n + 1);
    rdn(f, s, (size_t)n); s[n] = 0; return s;
}
static void skipv(FILE *f, uint32_t t) {
    static const int sz[13] = { 1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8 };
    if (t == 8) { free(rdstr(f)); }
    else if (t == 9) {
        uint32_t et = rdu32(f); uint64_t n = rdu64(f);
        for (uint64_t i = 0; i < n; i++) skipv(f, et);
    } else {
        uint8_t tmp[8];
        if (t > 12 || sz[t] == 0) die("KeyError: %u", t);
        rdn(f, tmp, (size_t)sz[t]);
    }
}

static void parse_header(FILE *f, uint64_t *n_kv_o, buf_t *kv_raw, elist_t *tens, uint64_t *data0_o) {
    uint32_t magic = rdu32(f), ver = rdu32(f);
    uint64_t n_t = rdu64(f), n_kv = rdu64(f);
    if (!(magic == 0x46554747u && ver == 3)) die("AssertionError: 非 GGUF v3(magic=%08x ver=%u)", magic, ver);
    off_t kv_start = ftello(f);
    for (uint64_t i = 0; i < n_kv; i++) { free(rdstr(f)); uint32_t t = rdu32(f); skipv(f, t); }
    off_t kv_end = ftello(f);
    fseeko(f, kv_start, SEEK_SET);
    kv_raw->n = 0;
    { size_t n = (size_t)(kv_end - kv_start); uint8_t *tmp = xmalloc(n); rdn(f, tmp, n); buf_add(kv_raw, tmp, n); free(tmp); }
    for (uint64_t i = 0; i < n_t; i++) {
        ent_t *e = el_push(tens);
        e->name = rdstr(f);
        e->nd = rdu32(f);
        e->ne = xmalloc((size_t)e->nd * 8);
        rdn(f, e->ne, (size_t)e->nd * 8);
        e->type = rdu32(f);
        e->off = rdu64(f);
    }
    *data0_o = align_up((uint64_t)ftello(f));
    *n_kv_o = n_kv;
}

static int cmp_off(const void *a, const void *b) {
    const ent_t *const *x = a, *const *y = b;
    if ((*x)->off < (*y)->off) return -1;
    if ((*x)->off > (*y)->off) return 1;
    return (*x)->idx < (*y)->idx ? -1 : (*x)->idx > (*y)->idx;   /* sorted() 稳定 */
}
static void sizes_by_offset(elist_t *tens, uint64_t fsz, uint64_t data0) {
    size_t n = tens->n;
    ent_t **ord = xmalloc(n * sizeof *ord);
    for (size_t i = 0; i < n; i++) ord[i] = &tens->v[i];
    qsort(ord, n, sizeof *ord, cmp_off);
    for (size_t i = 0; i < n; i++) {
        uint64_t end = i + 1 < n ? ord[i + 1]->off : fsz - data0;
        ord[i]->bytes = end - ord[i]->off;
    }
    free(ord);
}

static void ser_info(buf_t *b, const ent_t *t) {
    size_t nl = strlen(t->name);
    buf_u64(b, nl); buf_add(b, t->name, nl);
    buf_u32(b, t->nd); buf_add(b, t->ne, (size_t)t->nd * 8);
    buf_u32(b, t->type); buf_u64(b, t->off);
}

static int is_dropped(const char *nm, int skeleton_mode) {
    if (str_endswith(nm, "ffn_exps_vq.blob")) return 1;
    if (strstr(nm, ".opt_")) return 1;
    if (skeleton_mode && str_endswith(nm, "ffn_down_exps.weight")) return 1;
    /* ★gate/up 路由专家也必须丢(2026-08-02 修)★
     * 原来没这两条是因为骨架源一直是【冠军 VQ 合一 GGUF】—— 那种文件里 gate/up exps 早已
     * 不存在(专家字节全在 blk.L.ffn_exps_vq.blob 里)。而自产骨架的源是 deepseek4-quantize
     * --experts-hole 产出的留洞文件, 它的张量表沿用 published 模版, gate/up exps **在表里**
     * (只是数据是洞)。不丢的话 extract_skeleton 会把这些洞当骨架张量写出来:
     * 实测文件从 8.2 GiB 撑到 17.7+ GiB, 两次把磁盘写穿(OSError:28)。 */
    if (skeleton_mode && (str_endswith(nm, "ffn_gate_exps.weight") ||
                          str_endswith(nm, "ffn_up_exps.weight"))) return 1;
    return 0;
}

/* entries 顺序即数据序; writer(out, e) 负责写数据。 */
static uint64_t write_gguf(const char *out_p, const buf_t *kv_raw, uint64_t n_kv,
                           elist_t *ents, void (*writer)(FILE *, ent_t *)) {
    uint64_t off = 0;
    for (size_t i = 0; i < ents->n; i++) { ents->v[i].off = off; off += align_up(ents->v[i].bytes); }
    buf_t hdr = { 0 }, info = { 0 };
    buf_u32(&hdr, 0x46554747u); buf_u32(&hdr, 3);
    buf_u64(&hdr, (uint64_t)ents->n); buf_u64(&hdr, n_kv);
    buf_add(&hdr, kv_raw->p, kv_raw->n);
    for (size_t i = 0; i < ents->n; i++) ser_info(&info, &ents->v[i]);
    uint64_t data0 = align_up(hdr.n + info.n);
    FILE *out = fopen(out_p, "wb");
    if (!out) die("IOError: 打不开 %s: %s", out_p, strerror(errno));
    if (fwrite(hdr.p, 1, hdr.n, out) != hdr.n) die("IOError: 写头失败");
    if (fwrite(info.p, 1, info.n, out) != info.n) die("IOError: 写张量表失败");
    { size_t pad = (size_t)(data0 - hdr.n - info.n); uint8_t z[ALIGN] = { 0 };
      if (pad && fwrite(z, 1, pad, out) != pad) die("IOError: 写对齐失败"); }
    for (size_t i = 0; i < ents->n; i++) {
        if (fseeko(out, (off_t)(data0 + ents->v[i].off), SEEK_SET)) die("IOError: seek 失败");
        writer(out, &ents->v[i]);
    }
    if (fclose(out)) die("IOError: 关闭 %s 失败: %s", out_p, strerror(errno));
    free(hdr.p); free(info.p);
    return data0;
}

/* ---------------- 命令行参数(dest 名与 .py argparse 一一对应) ---------------- */

typedef struct {
    int extract_skeleton, extract_blobs, merge, no_down, consume, gud;
    const char *out_dir, *base, *skeleton, *blob_sizes, *down_offsets, *route_bias;
    double route_alpha;
    const char *blob_layers, *dql_host, *dql_dir, *out;
} args_t;
static args_t A;

/* 写盘上下文(.py 里是闭包捕获的 src/data0/rb/doff) */
static FILE    *g_src;
static uint64_t g_src_data0;
static double   g_rb[N_LAYER][NEXP];   /* f64: .py 的 row 是 f64, 只在烘焙时落一次 f32 */
static int      g_rb_has[N_LAYER];
static imap_t   g_doff;
static uint8_t *g_chunk_skel, *g_chunk_strm;

static uint8_t *chunk_skel(void) { if (!g_chunk_skel) g_chunk_skel = xmalloc(CHUNK_SKEL); return g_chunk_skel; }
static uint8_t *chunk_strm(void) { if (!g_chunk_strm) g_chunk_strm = xmalloc(CHUNK_STRM); return g_chunk_strm; }

/* ---------------- --extract-blobs ---------------- */

/* ★反修v4逆向(2026-08-10): 从合一 GGUF 反抽 blk.L.ffn_exps_vq.blob → dql_vq_LXX.bin
 * (--consume 吃掉层件后的链式反修回收路; 字节=blob 原样, vq_slot/zlayer 直读)。 */
static void extract_blobs(void) {
    FILE *f = fopen(A.base, "rb");
    if (!f) die("FileNotFoundError: %s: %s", A.base, strerror(errno));
    uint64_t n_kv, data0; buf_t kv_raw = { 0 }; elist_t tens = { 0 };
    parse_header(f, &n_kv, &kv_raw, &tens, &data0);
    uint64_t fsz = fsize(A.base);
    sizes_by_offset(&tens, fsz, data0);
    int n = 0;
    for (size_t i = 0; i < tens.n; i++) {
        ent_t *t = &tens.v[i];
        if (!str_endswith(t->name, "ffn_exps_vq.blob")) continue;
        int L = name_field_int(t->name, 1);
        char outp[2048]; snprintf(outp, sizeof outp, "%s/dql_vq_L%02d.bin", A.out_dir, L);
        fseeko(f, (off_t)(data0 + t->off), SEEK_SET);
        uint64_t remain = t->bytes;
        FILE *o = fopen(outp, "wb");
        if (!o) die("IOError: 打不开 %s: %s", outp, strerror(errno));
        uint8_t *buf = chunk_strm();
        while (remain > 0) {
            size_t want = remain < CHUNK_STRM ? (size_t)remain : CHUNK_STRM;
            size_t got = fread(buf, 1, want, f);
            if (!got) break;
            if (fwrite(buf, 1, got, o) != got) die("IOError: 写 %s 失败", outp);
            remain -= got;
        }
        fclose(o);
        n++;
        printf("L%02d blob %.2fGB -> %s\n", L, t->bytes / 1e9, outp); fflush(stdout);
    }
    printf("extract-blobs done: %d\n", n); fflush(stdout);
    fclose(f);
}

/* ---------------- --extract-skeleton ---------------- */

static void skel_writer(FILE *out, ent_t *e) {
    fseeko(g_src, (off_t)(g_src_data0 + e->src_off), SEEK_SET);
    uint64_t left = e->bytes;
    uint8_t *buf = chunk_skel();
    while (left) {
        size_t want = left < CHUNK_SKEL ? (size_t)left : CHUNK_SKEL;
        size_t got = fread(buf, 1, want, g_src);
        if (!got) die("IOError: 骨架源短读(剩 %llu B)", (unsigned long long)left);
        if (fwrite(buf, 1, got, out) != got) die("IOError: 写骨架失败");
        left -= got;
    }
}

static void extract_skeleton(void) {
    uint64_t fsz = fsize(A.base);
    FILE *f = fopen(A.base, "rb");
    if (!f) die("FileNotFoundError: %s: %s", A.base, strerror(errno));
    uint64_t n_kv, data0; buf_t kv_raw = { 0 }; elist_t tens = { 0 };
    parse_header(f, &n_kv, &kv_raw, &tens, &data0);
    fclose(f);
    sizes_by_offset(&tens, fsz, data0);
    elist_t ents = { 0 };
    for (size_t i = 0; i < tens.n; i++) {
        ent_t *t = &tens.v[i];
        if (is_dropped(t->name, 1)) continue;
        ent_t *e = el_push(&ents);
        e->name = t->name; e->nd = t->nd; e->ne = t->ne; e->type = t->type;
        e->bytes = t->bytes; e->src_off = t->off; e->kind = K_SRC;
    }
    g_src = fopen(A.base, "rb");
    if (!g_src) die("FileNotFoundError: %s: %s", A.base, strerror(errno));
    g_src_data0 = data0;
    write_gguf(A.out, &kv_raw, n_kv, &ents, skel_writer);
    printf("[skeleton] %s %.2f GiB (%zu 张量)\n", A.out, fsize(A.out) / (double)(1ULL << 30), ents.n);
    fflush(stdout);
}

/* ---------------- 路由偏置(RBIA → 每层 256×f32 Δb·α) ---------------- */

static void load_route_bias(const char *path, double alpha) {
    FILE *f = fopen(path, "rb");
    if (!f) die("FileNotFoundError: %s: %s", path, strerror(errno));
    uint32_t hd[4]; rdn(f, hd, 16);
    if (!(hd[0] == 0x41494252u && hd[1] == N_LAYER && hd[2] == NEXP)) die("AssertionError: RBIA 头不对");
    float *acc = xmalloc((size_t)N_LAYER * NEXP * 4);
    uint32_t *cnt = xmalloc((size_t)N_LAYER * NEXP * 4);
    rdn(f, acc, (size_t)N_LAYER * NEXP * 4);
    rdn(f, cnt, (size_t)N_LAYER * NEXP * 4);
    fclose(f);
    long long armed = 0; int nlayer = 0;
    for (int L = 0; L < N_LAYER; L++) {
        /* .py: row 是 f64(alpha 与 acc 都升 f64 再乘); 烘焙时才落回 f32 —— 舍入点照抄, 见
         * merge_writer 里 (double)v + (double)row 后一次 f32 舍入。 */
        int any = 0, na = 0;
        for (int e = 0; e < NEXP; e++) {
            g_rb[L][e] = cnt[L * NEXP + e] >= RB_MINCNT ? alpha * (double)acc[L * NEXP + e] : 0.0;
            if (g_rb[L][e] != 0.0) { any = 1; na++; }
        }
        if (any) { g_rb_has[L] = 1; nlayer++; armed += na; }
    }
    free(acc); free(cnt);
    printf("[route-bias] α=%s mincnt=%d 武装槽=%lld 层=%d\n", py_float_repr(alpha), RB_MINCNT, armed, nlayer);
    fflush(stdout);
}

/* ---------------- ssh/本地 流式取字节 ---------------- */

static int host_is_local(const char *h) {
    return !strcmp(h, "") || !strcmp(h, "-") || !strcmp(h, "local") ||
           !strcmp(h, "127.0.0.1") || !strcmp(h, "localhost");
}

/* 把 s 包成单引号 shell 词(内嵌单引号转成 '\'') —— popen 走本地 /bin/sh, 必须保证
 * ssh 收到的 argv 与 .py 的 Popen(list) 完全一致(host 一个词, 远端命令串一个词)。 */
static void shq(const char *s, char *dst, size_t dsz) {
    size_t o = 0;
    if (o + 1 < dsz) dst[o++] = '\'';
    for (const char *p = s; *p; p++) {
        if (*p == '\'') { const char *r = "'\\''"; for (int i = 0; i < 4 && o + 1 < dsz; i++) dst[o++] = r[i]; }
        else if (o + 1 < dsz) dst[o++] = *p;
    }
    if (o + 1 < dsz) dst[o++] = '\'';
    if (o >= dsz) die("ValueError: shell 引用溢出");
    dst[o] = 0;
}

/* 流式拷贝 nbytes 到 out(逐块, 校验总长)。tail/head 而非 dd:
 * macOS BSD dd 不支持 GNU iflag=skip_bytes,count_bytes(实测 0B, 2026-07-28)。 */
static void ssh_stream(const char *host, const char *path, uint64_t nbytes, FILE *out, uint64_t skip) {
    uint8_t *buf = chunk_strm();
    if (host_is_local(host)) {
        /* 本地直读(2026-07-31 R28: 合并在产物同机跑, 免 ssh 往返/免自连密钥) */
        FILE *f = fopen(path, "rb");
        if (!f) die("FileNotFoundError: %s: %s", path, strerror(errno));
        fseeko(f, (off_t)skip, SEEK_SET);
        uint64_t left = nbytes;
        while (left) {
            size_t want = left < CHUNK_STRM ? (size_t)left : CHUNK_STRM;
            size_t got = fread(buf, 1, want, f);
            if (!got) break;
            if (fwrite(buf, 1, got, out) != got) die("IOError: 写输出失败");
            left -= got;
        }
        fclose(f);
        if (left != 0) die("AssertionError: %s: 本地读 %lluB ≠ %lluB", path,
                           (unsigned long long)(nbytes - left), (unsigned long long)nbytes);
        return;
    }
    char qh[1024], qc[4096], rem[3072], cmd[8192];
    snprintf(rem, sizeof rem, "tail -c +%llu %s | head -c %llu",
             (unsigned long long)(skip + 1), path, (unsigned long long)nbytes);
    shq(host, qh, sizeof qh); shq(rem, qc, sizeof qc);
    snprintf(cmd, sizeof cmd, "ssh %s %s", qh, qc);
    FILE *p = popen(cmd, "r");
    if (!p) die("OSError: popen 失败: %s", cmd);
    uint64_t got_tot = 0;
    for (;;) {
        size_t n = fread(buf, 1, CHUNK_STRM, p);
        if (!n) break;
        if (fwrite(buf, 1, n, out) != n) die("IOError: 写输出失败");
        got_tot += n;
    }
    pclose(p);
    if (got_tot != nbytes) die("AssertionError: %s: 流式 %lluB ≠ %lluB", path,
                                (unsigned long long)got_tot, (unsigned long long)nbytes);
}

/* ---------------- opt 内嵌(反修 op 编进合一 GGUF) ---------------- */

/* ★opt 内嵌(2026-08-06 用户令"修通op")★: 反修 op(dql_ops 侧车终值+dql 正位 zl.RRR)
 * 编成引擎 zchain_from_model 期望的四类张量内嵌进合并 GGUF(blk.L.opt_chain/ge/v8/zlm
 * + kv ds4.zchain.present=true)。模型自包含, 运行时零外挂零开关。 */

#define V8SZ (8u * 4096u * 2u)   /* 65536 */

typedef struct { float (*r)[16]; size_t n, cap; } chain_t;
static void chain_add(chain_t *c, const float *row) {
    if (c->n == c->cap) { c->cap = c->cap * 2 + 16; c->r = xrealloc(c->r, c->cap * sizeof *c->r); }
    memcpy(c->r[c->n++], row, 16 * sizeof(float));
}
typedef struct { uint8_t **b; size_t n, cap; } v8_t;
static void v8_add(v8_t *v, const uint8_t *blk) {
    if (v->n == v->cap) { v->cap = v->cap * 2 + 8; v->b = xrealloc(v->b, v->cap * sizeof *v->b); }
    v->b[v->n] = xmalloc(V8SZ); memcpy(v->b[v->n], blk, V8SZ); v->n++;
}

/* → 追加 [entries(kind=K_OPT, data=bytes)] 引擎 zchain_from_model 格式; 返回新增数量。 */
static int build_opt_tensors(const char *dql_dir, int L, elist_t *out) {
    chain_t chain = { 0 };
    v8_t v8 = { 0 };
    uint8_t *ge = NULL; size_t ge_len = 0;
    uint8_t *zlm = NULL; size_t zlm_len = 0;
    int has_glhc = 0; float gh = 0, gc = 0;   /* (gh,gc): 折进 per-expert ge 表(引擎 GE 通道零改动) */
    char p[2048];

    snprintf(p, sizeof p, "%s/dql_ops_L%02d.bin", dql_dir, L);
    if (fexists(p)) {
        size_t rlen = 0; uint8_t *raw = read_all(p, &rlen);
        if (rlen >= 12 && (!memcmp(raw, "DQL2", 4) || !memcmp(raw, "DQO2", 4))) {
            uint32_t nrec; memcpy(&nrec, raw + 8, 4);
            uint64_t off = 12;
            for (uint32_t k = 0; k < nrec; k++) {
                if (off > rlen || rlen - off < REC_HDR) break;
                char nm[17]; memcpy(nm, raw + (size_t)off, 16); nm[16] = 0;   /* split(b"\0")[0] */
                uint64_t psz; memcpy(&psz, raw + (size_t)off + 88, 8);
                int32_t vd; memcpy(&vd, raw + (size_t)off + 112, 4);
                uint64_t pstart = off + REC_HDR;
                uint64_t plen = rlen - pstart; if (plen > psz) plen = psz;   /* py 切片自动截断 */
                const uint8_t *pay = raw + (size_t)pstart;
                off = sat_add(sat_add(off, REC_HDR), psz);
                if (vd != 1) continue;
                float row[16]; memset(row, 0, sizeof row);
                if (strstr(nm, "GLdyn2") && plen >= 16) {
                    row[0] = 2.0f; memcpy(row + 2, pay, 16); chain_add(&chain, row);
                } else if (strstr(nm, "GLdyn8") && plen >= 36) {
                    row[0] = 3.0f; memcpy(row + 6, pay, 36);
                    if (plen >= 36 + (uint64_t)V8SZ) { row[15] = (float)v8.n; v8_add(&v8, pay + 36); }
                    else continue;                    /* V8-less dyn8 = 引擎侧 no-op, 不嵌 */
                    chain_add(&chain, row);
                } else if (strstr(nm, "bf.GE") && plen >= 512) {
                    float *g = xmalloc(NEXP * 4);
                    for (int e = 0; e < NEXP; e++) { uint16_t h; memcpy(&h, pay + 2 * e, 2); g[e] = f16_to_f32(h); }
                    free(ge); ge = (uint8_t *)g; ge_len = NEXP * 4;
                } else if (strstr(nm, "zl.RRR") && plen >= 16) {
                    uint32_t zk; float tr; memcpy(&zk, pay, 4); memcpy(&tr, pay + 4, 4);
                    uint64_t nh = (uint64_t)zk + 2ULL * zk * 4096;
                    if (zk > 0 && zk <= 16 && plen >= 16 + nh * 2) {
                        row[0] = 6.0f; row[1] = tr; row[2] = (float)zk; chain_add(&chain, row);
                        free(zlm); zlm_len = (size_t)(nh * 2); zlm = xmalloc(zlm_len);
                        memcpy(zlm, pay + 16, zlm_len);
                    }
                } else if (strstr(nm, "GLhc") && plen >= 8) {
                    memcpy(&gh, pay, 4); memcpy(&gc, pay + 4, 4); has_glhc = 1;
                } else if (strstr(nm, ".GL") && plen >= 4) {
                    row[0] = 1.0f; memcpy(&row[1], pay, 4); chain_add(&chain, row);
                } else if ((strstr(nm, "TREF") || strstr(nm, "xlayer")) && plen >= 4) {
                    row[0] = 4.0f; memcpy(&row[1], pay, 4); chain_add(&chain, row);
                }
            }
        }
        free(raw);
    }

    if (!zlm) {   /* zl.RRR 正位在 dql 主文件(2026-08-04 正位直写) */
        snprintf(p, sizeof p, "%s/dql_L%02d.bin", dql_dir, L);
        if (fexists(p)) {
            FILE *f = fopen(p, "rb");
            if (!f) die("FileNotFoundError: %s: %s", p, strerror(errno));
            uint8_t head[12]; size_t hn = fread(head, 1, 12, f);
            if (hn >= 4 && !memcmp(head, "DQL2", 4)) {
                if (hn < 12) die("struct.error: unpack_from requires a buffer of at least 12 bytes");
                uint32_t nrec; memcpy(&nrec, head + 8, 4);
                for (uint32_t k = 0; k < nrec; k++) {
                    uint8_t hdr[REC_HDR];
                    if (fread(hdr, 1, REC_HDR, f) < REC_HDR) break;
                    char nm[17]; memcpy(nm, hdr, 16); nm[16] = 0;
                    uint64_t psz; memcpy(&psz, hdr + 88, 8);
                    if (strstr(nm, "zl.RRR") && psz >= 16) {
                        uint8_t *pay = xmalloc((size_t)psz);
                        size_t pn = fread(pay, 1, (size_t)psz, f);
                        if (pn < 8) die("struct.error: unpack_from requires a buffer of at least 8 bytes");
                        uint32_t zk; float tr; memcpy(&zk, pay, 4); memcpy(&tr, pay + 4, 4);
                        uint64_t nh = (uint64_t)zk + 2ULL * zk * 4096;
                        if (zk > 0 && zk <= 16 && pn >= 16 + nh * 2) {
                            float row[16]; memset(row, 0, sizeof row);
                            row[0] = 6.0f; row[1] = tr; row[2] = (float)zk;
                            chain_add(&chain, row);
                            zlm_len = (size_t)(nh * 2); zlm = xmalloc(zlm_len);
                            memcpy(zlm, pay + 16, zlm_len);
                        }
                        free(pay);
                        break;
                    }
                    fseeko(f, (off_t)psz, SEEK_CUR);
                }
            }
            fclose(f);
        }
    }

    if (has_glhc) {
        float *gev = xmalloc(NEXP * 4);
        if (ge_len) memcpy(gev, ge, NEXP * 4);
        else for (int e = 0; e < NEXP; e++) gev[e] = 1.0f;
        char hot[NEXP]; memset(hot, 0, sizeof hot);
        snprintf(p, sizeof p, "%s/dql_vq_L%02d.bin", dql_dir, L);
        if (fexists(p)) {
            FILE *vf = fopen(p, "rb");
            if (!vf) die("FileNotFoundError: %s: %s", p, strerror(errno));
            fseeko(vf, 16, SEEK_SET);
            uint64_t *vt = xmalloc(768 * 8);
            if (fread(vt, 1, 768 * 8, vf) != 768 * 8) die("struct.error: unpack requires a buffer of 6144 bytes");
            for (int e = 0; e < NEXP; e++) if (vt[e * 3 + 2] != 0) hot[e] = 1;
            free(vt); fclose(vf);
        }
        /* numpy: f32 数组 *= python float(弱标量) → 全程 f32 乘, 与这里一致 */
        for (int e = 0; e < NEXP; e++) gev[e] *= hot[e] ? gh : gc;
        free(ge); ge = (uint8_t *)gev; ge_len = NEXP * 4;
    }

    int n = 0;
    char nmbuf[128];
    if (chain.n) {
        size_t nb = chain.n * 16 * sizeof(float);
        uint8_t *data = xmalloc(nb);
        memcpy(data, chain.r, nb);
        snprintf(nmbuf, sizeof nmbuf, "blk.%d.opt_chain.weight", L);
        el_push1(out, nmbuf, (uint64_t)(chain.n * 16), 0, nb, K_OPT)->data = data;
        n++;
    }
    if (ge_len) {
        snprintf(nmbuf, sizeof nmbuf, "blk.%d.opt_ge.weight", L);
        el_push1(out, nmbuf, NEXP, 0, ge_len, K_OPT)->data = ge;
        n++;
    }
    if (v8.n) {
        size_t nb = v8.n * V8SZ;
        uint8_t *data = xmalloc(nb);
        for (size_t i = 0; i < v8.n; i++) memcpy(data + i * V8SZ, v8.b[i], V8SZ);
        snprintf(nmbuf, sizeof nmbuf, "blk.%d.opt_v8.weight", L);
        el_push1(out, nmbuf, (uint64_t)(v8.n * 8 * 4096), 1, nb, K_OPT)->data = data;
        n++;
    }
    if (zlm) {
        snprintf(nmbuf, sizeof nmbuf, "blk.%d.opt_zlm.weight", L);
        el_push1(out, nmbuf, (uint64_t)(zlm_len / 2), 1, zlm_len, K_OPT)->data = zlm;
        n++;
    }
    free(chain.r);
    for (size_t i = 0; i < v8.n; i++) free(v8.b[i]);
    free(v8.b);
    return n;
}

static void kv_append_bool(buf_t *kv_raw, uint64_t *n_kv, const char *key, int val) {
    size_t kl = strlen(key);
    buf_u64(kv_raw, kl); buf_add(kv_raw, key, kl);
    buf_u32(kv_raw, 7);
    uint8_t b = val ? 1 : 0; buf_add(kv_raw, &b, 1);
    (*n_kv)++;
}

/* ---------------- --merge ---------------- */

static ent_t *el_push3(elist_t *l, const char *name, uint64_t a0, uint64_t a1, uint64_t a2,
                       uint32_t type, uint64_t bytes, int kind) {
    ent_t *e = el_push(l);
    e->name = xstrdup(name); e->nd = 3; e->ne = xmalloc(24);
    e->ne[0] = a0; e->ne[1] = a1; e->ne[2] = a2;
    e->type = type; e->bytes = bytes; e->kind = kind; return e;
}

/* "L 值" 两列文本 → dict(与 .py 的 line.split() + int() 同严格度) */
static void read_int_pairs(const char *path, imap_t *m) {
    FILE *f = fopen(path, "r");
    if (!f) die("FileNotFoundError: %s: %s", path, strerror(errno));
    char line[1024];
    while (fgets(line, sizeof line, f)) {
        char a1[256], a2[256], extra[256];
        int nf = sscanf(line, "%255s %255s %255s", a1, a2, extra);
        if (nf > 2) die("ValueError: too many values to unpack (expected 2)");
        if (nf != 2) die("ValueError: not enough values to unpack (expected 2, got %d)", nf < 0 ? 0 : nf);
        char *end1, *end2;
        long long k = strtoll(a1, &end1, 10), v = strtoll(a2, &end2, 10);
        if (*end1 || *end2) die("ValueError: invalid literal for int(): '%s' / '%s'", a1, a2);
        imap_set(m, k, v);
    }
    fclose(f);
}

static void merge_writer(FILE *out, ent_t *e) {
    char pth[2048], fld[64];
    if (e->kind == K_OPT) {
        if (fwrite(e->data, 1, e->bytes, out) != e->bytes) die("IOError: 写 opt 失败");
        return;
    }
    if (e->kind == K_SKEL) {
        fseeko(g_src, (off_t)(g_src_data0 + e->src_off), SEEK_SET);
        const char *nm = e->name;
        if (str_startswith(nm, "blk.") && str_endswith(nm, ".exp_probs_b.bias")) {
            int L = name_field_int(nm, 1);
            uint8_t *raw = xmalloc((size_t)e->bytes);
            if (fread(raw, 1, (size_t)e->bytes, g_src) != e->bytes) die("IOError: 骨架短读 %s", nm);
            if (L >= 0 && L < N_LAYER && g_rb_has[L]) {
                if (e->bytes < (uint64_t)NEXP * 4) die("struct.error: unpack requires a buffer of 1024 bytes");
                for (int x = 0; x < NEXP; x++) {
                    float v; memcpy(&v, raw + 4 * x, 4);
                    v = (float)((double)v + g_rb[L][x]);   /* f64 加, 一次 f32 舍入(照抄 .py) */
                    memcpy(raw + 4 * x, &v, 4);
                }
                printf("[merge] gate.bias L%d 烘焙 ✓\n", L); fflush(stdout);
            }
            if (fwrite(raw, 1, (size_t)e->bytes, out) != e->bytes) die("IOError: 写 bias 失败");
            free(raw); return;
        }
        uint64_t left = e->bytes;
        uint8_t *buf = chunk_skel();
        while (left) {
            size_t want = left < CHUNK_SKEL ? (size_t)left : CHUNK_SKEL;
            size_t got = fread(buf, 1, want, g_src);
            if (!got) die("IOError: 骨架源短读(剩 %llu B)", (unsigned long long)left);
            if (fwrite(buf, 1, got, out) != got) die("IOError: 写骨架失败");
            left -= got;
        }
        return;
    }
    if (e->kind == K_GUD) {
        snprintf(pth, sizeof pth, "%s/dql_L%02d.bin", A.dql_dir, e->L);
        ssh_stream(A.dql_host, pth, e->bytes, out, e->skip);
        int consumed = A.consume && e->last;
        if (consumed) { fflush(out); if (remove(pth)) die("OSError: 删 %s 失败: %s", pth, strerror(errno)); }
        name_field(e->name, 2, fld, sizeof fld);
        printf("[merge] %s L%d ✓%s\n", fld, e->L, consumed ? " (dql已消费)" : ""); fflush(stdout);
        return;
    }
    if (e->kind == K_STUB) {
        uint32_t magic = 0x4C565144u;   /* DQVL 魔数 + 全零槽表 */
        if (fwrite(&magic, 1, 4, out) != 4) die("IOError: 写存根失败");
        uint64_t left = e->bytes - 4;
        uint8_t z[4096]; memset(z, 0, sizeof z);
        while (left) { size_t w = left < sizeof z ? (size_t)left : sizeof z;
                       if (fwrite(z, 1, w, out) != w) die("IOError: 写存根失败"); left -= w; }
        return;
    }
    if (e->kind == K_DOWN) {
        snprintf(pth, sizeof pth, "%s/dql_L%02d.bin", A.dql_dir, e->L);
        ssh_stream(A.dql_host, pth, DOWN_LAYER_BYTES, out, (uint64_t)imap_get(&g_doff, e->L));
        /* ★--consume 也吃 dql(2026-08-03 冠军复刻盘账)★: D 段是 dql 里唯一被合并读的段,
         * 读完该层 dql 即死重(反修/回放此时已完成)。54G 输出必须靠这 ~13G 边合并边释放。 */
        if (A.consume) { fflush(out); if (remove(pth)) die("OSError: 删 %s 失败: %s", pth, strerror(errno)); }
        printf("[merge] down L%d ✓%s\n", e->L, A.consume ? " (dql已消费)" : ""); fflush(stdout);
        return;
    }
    /* K_BLOB */
    snprintf(pth, sizeof pth, "%s/dql_vq_L%02d.bin", A.dql_dir, e->L);
    ssh_stream(A.dql_host, pth, e->bytes, out, 0);
    /* ★--consume(R30)★: 该层 blob 已完整写进输出 → 立删源文件。M1 free 50G 放不下
     * 层27.8+骨架8.2+输出36 三者同存; 消费式把峰值净增压到 ~9G(输出36−层27.8)。
     * 代价: 合并中途失败已删的层要重量化 —— 只在盘账过不去时用, 由调用方显式开。 */
    if (A.consume) {
        if (!host_is_local(A.dql_host)) { fprintf(stderr, "--consume 只允许本地 dql-host(不做远程删)\n"); exit(1); }
        fflush(out); if (remove(pth)) die("OSError: 删 %s 失败: %s", pth, strerror(errno));
    }
    printf("[merge] blob L%d %.0fMiB ✓%s\n", e->L, e->bytes / (double)(1ULL << 20),
           A.consume ? " (源已消费)" : ""); fflush(stdout);
}

static void merge(void) {
    uint64_t fsz = fsize(A.skeleton);
    FILE *f = fopen(A.skeleton, "rb");
    if (!f) die("FileNotFoundError: %s: %s", A.skeleton, strerror(errno));
    uint64_t n_kv, data0; buf_t kv_raw = { 0 }; elist_t tens = { 0 };
    parse_header(f, &n_kv, &kv_raw, &tens, &data0);
    fclose(f);
    sizes_by_offset(&tens, fsz, data0);
    if (A.route_bias) load_route_bias(A.route_bias, A.route_alpha);
    imap_t bsz = { 0 };
    read_int_pairs(A.blob_sizes, &bsz);
    if (bsz.n != N_LAYER) die("AssertionError: blob 尺寸表 %zu/43", bsz.n);
    /* D 段权威偏移(dql_down_offset.py 记录链解析): 固定 DQL_HDR=35104 是错误假设 —
     * 1bit 记录载荷@128, 实测全层 570,425,472; 仍按每层文件读取防未来漂移。 */
    if (!A.no_down) {
        read_int_pairs(A.down_offsets, &g_doff);
        if (g_doff.n != N_LAYER) die("AssertionError: down 偏移表 %zu/43", g_doff.n);
    }
    elist_t ents = { 0 };
    for (size_t i = 0; i < tens.n; i++) {
        ent_t *t = &tens.v[i], *e = el_push(&ents);
        e->name = t->name; e->nd = t->nd; e->ne = t->ne; e->type = t->type;
        e->bytes = t->bytes; e->kind = K_SKEL; e->src_off = t->off;
    }
    /* opt 内嵌(本地 dql_dir 读; --consume 前于 ents 构造期完成全部读取) */
    int n_opt = 0;
    if (host_is_local(A.dql_host))
        for (int L = 0; L < N_LAYER; L++) n_opt += build_opt_tensors(A.dql_dir, L, &ents);
    if (n_opt) {
        kv_append_bool(&kv_raw, &n_kv, "ds4.zchain.present", 1);
        printf("[merge] opt 内嵌 %d 张量 + ds4.zchain.present=true\n", n_opt);
    } else {
        printf("[merge] ★opt 张量零内嵌(侧车缺?)— 反修效果不在此模型, 需人工确认★\n");
    }
    fflush(stdout);
    /* ★--gud(2026-08-03 code2b 42G 复刻)★: G/U/D 三段 signref 全搬(1bit 记录载荷布局
     * [G 256×szG][U 256×szG][D 256×szD], dql_down_offset 的 D 偏移回推 G/U)。blob(热侧车)
     * 不进 GGUF — 运行时外挂, 42.468G 账目与 code2b 一字对齐。 */
    if (A.gud) {
        uint64_t sz_g = 2048ULL * (4096 / 256) * 34;   /* = 全局 SZ_G, .py 里同名局部量 */
        char nm[128];
        for (int L = 0; L < N_LAYER; L++) {
            uint64_t g_off = (uint64_t)(imap_get(&g_doff, L) - 2LL * NEXP * (long long)sz_g);
            snprintf(nm, sizeof nm, "blk.%d.ffn_gate_exps.weight", L);
            ent_t *e = el_push3(&ents, nm, 4096, 2048, 256, 40, NEXP * sz_g, K_GUD);
            e->L = L; e->skip = g_off;
            snprintf(nm, sizeof nm, "blk.%d.ffn_up_exps.weight", L);
            e = el_push3(&ents, nm, 4096, 2048, 256, 40, NEXP * sz_g, K_GUD);
            e->L = L; e->skip = g_off + NEXP * sz_g;
            snprintf(nm, sizeof nm, "blk.%d.ffn_down_exps.weight", L);
            e = el_push3(&ents, nm, 2048, 4096, 256, 40, DOWN_LAYER_BYTES, K_GUD);
            e->L = L; e->skip = (uint64_t)imap_get(&g_doff, L); e->last = 1;
        }
    }
    int has_blr = 0, blo = 0, bhi = 0;
    if (A.blob_layers[0]) {
        char lo[64], hi[64];
        const char *c = strchr(A.blob_layers, ':');
        if (!c || strchr(c + 1, ':')) die("ValueError: too many/few values to unpack: %s", A.blob_layers);
        snprintf(lo, sizeof lo, "%.*s", (int)(c - A.blob_layers), A.blob_layers);
        snprintf(hi, sizeof hi, "%s", c + 1);
        blo = atoi(lo); bhi = atoi(hi); has_blr = 1;
        printf("[merge] 切片模式: 只带 L%d..L%d 的 blob(%d 层)\n", blo, bhi, bhi - blo + 1);
        fflush(stdout);
    }
    char nm[128];
    for (int L = 0; L < (A.gud ? 0 : N_LAYER); L++) {
        /* --no-down(R28): 冷 w2 已编在 blob 的 which=2 槽里, base down 是纯死重
         * (43 × 0.2656 = 11.42 GiB)。省掉它才是 28 GiB 目标能成立的原因; 引擎侧
         * 由 routed_down_shadow() 合成影子张量顶上(ds4.c)。冠军 vq4bf 那种冷 w2 从
         * base go1b 读的配方不能带这个开关, 否则冷专家权重直接没了。 */
        if (has_blr && !(blo <= L && L <= bhi)) {
            /* 存根 blob(6160B 零头): 装载器认层过"required tensor"检查, 槽位全 0 →
             * 前向若误触即报"缺 vq 槽"硬错; 分布式非本机层永不前向 ⇒ 纯占位。 */
            snprintf(nm, sizeof nm, "blk.%d.ffn_exps_vq.blob", L);
            el_push1(&ents, nm, 6160, 42, 6160, K_STUB)->L = L;
            continue;
        }
        if (!A.no_down) {
            snprintf(nm, sizeof nm, "blk.%d.ffn_down_exps.weight", L);
            el_push3(&ents, nm, 2048, 4096, 256, 40, DOWN_LAYER_BYTES, K_DOWN)->L = L;
        }
        snprintf(nm, sizeof nm, "blk.%d.ffn_exps_vq.blob", L);
        uint64_t b = (uint64_t)imap_get(&bsz, L);
        el_push1(&ents, nm, b, 42, b, K_BLOB)->L = L;
    }
    g_src = fopen(A.skeleton, "rb");
    if (!g_src) die("FileNotFoundError: %s: %s", A.skeleton, strerror(errno));
    g_src_data0 = data0;
    write_gguf(A.out, &kv_raw, n_kv, &ents, merge_writer);
    printf("[merge] 完成 %s %.2f GiB\n", A.out, fsize(A.out) / (double)(1ULL << 30));
    fflush(stdout);
}

/* ---------------- main ---------------- */

static const char *USAGE =
    "usage: vq_merge_v4 [-h] [--extract-skeleton] [--extract-blobs] [--out-dir OUT_DIR]\n"
    "                   [--merge] [--base BASE] [--skeleton SKELETON]\n"
    "                   [--blob-sizes BLOB_SIZES] [--down-offsets DOWN_OFFSETS] [--no-down]\n"
    "                   [--route-bias ROUTE_BIAS] [--route-alpha ROUTE_ALPHA] [--consume]\n"
    "                   [--gud] [--blob-layers BLOB_LAYERS] [--dql-host DQL_HOST]\n"
    "                   [--dql-dir DQL_DIR] --out OUT\n";

static void argerr(const char *msg) { fputs(USAGE, stderr); fprintf(stderr, "vq_merge_v4: error: %s\n", msg); exit(2); }

/* 前缀匹配 --name / --name=VALUE */
static int arg_is(const char *a, const char *name, const char **val) {
    size_t n = strlen(name);
    if (strncmp(a, name, n)) return 0;
    if (a[n] == 0) { *val = NULL; return 1; }
    if (a[n] == '=') { *val = a + n + 1; return 1; }
    return 0;
}

int main(int argc, char **argv) {
    A.out_dir = "."; A.route_alpha = 2.5; A.blob_layers = "";
    A.dql_host = "192.168.1.2"; A.dql_dir = "/Users/fodelf/ds4-main/gguf/go-onebit/layers";

    for (int i = 1; i < argc; i++) {
        const char *s = argv[i], *v = NULL;
        char e[256];
#define FLAG(nm, fld) if (arg_is(s, nm, &v)) { if (v) { snprintf(e, sizeof e, "ignored explicit argument '%s'", v); argerr(e); } A.fld = 1; continue; }
#define OPTV(nm, fld, conv) if (arg_is(s, nm, &v)) { if (!v) { if (i + 1 >= argc) { snprintf(e, sizeof e, "argument %s: expected one argument", nm); argerr(e); } v = argv[++i]; } conv; continue; }
        if (!strcmp(s, "-h") || !strcmp(s, "--help")) { fputs(USAGE, stdout); return 0; }
        FLAG("--extract-skeleton", extract_skeleton)
        FLAG("--extract-blobs", extract_blobs)
        FLAG("--merge", merge)
        FLAG("--no-down", no_down)
        FLAG("--consume", consume)
        FLAG("--gud", gud)
        OPTV("--out-dir", out_dir, A.out_dir = v)
        OPTV("--base", base, A.base = v)
        OPTV("--skeleton", skeleton, A.skeleton = v)
        OPTV("--blob-sizes", blob_sizes, A.blob_sizes = v)
        OPTV("--down-offsets", down_offsets, A.down_offsets = v)
        OPTV("--route-bias", route_bias, A.route_bias = v)
        OPTV("--blob-layers", blob_layers, A.blob_layers = v)
        OPTV("--dql-host", dql_host, A.dql_host = v)
        OPTV("--dql-dir", dql_dir, A.dql_dir = v)
        OPTV("--out", out, A.out = v)
        OPTV("--route-alpha", route_alpha, {
            char *end; A.route_alpha = strtod(v, &end);
            if (end == v || *end) { snprintf(e, sizeof e, "argument --route-alpha: invalid float value: '%s'", v); argerr(e); }
        })
#undef FLAG
#undef OPTV
        snprintf(e, sizeof e, "unrecognized arguments: %s", s); argerr(e);
    }
    if (!A.out) argerr("the following arguments are required: --out");

    if (A.extract_blobs) { extract_blobs(); return 0; }
    if (A.extract_skeleton) extract_skeleton();
    else if (A.merge) {
        if (!A.blob_sizes) die("AssertionError: --merge 需 --blob-sizes");
        if (!(A.down_offsets || A.no_down)) die("AssertionError: --merge 需 --down-offsets(或 --no-down)");
        merge();
    } else { fprintf(stderr, "需 --extract-skeleton 或 --merge\n"); return 1; }
    return 0;
}
