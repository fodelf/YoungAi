/*
 * emit_z.c — emit per-layer "hidden-variable z^L" correction tensors as F32
 * GGUF tensors for the DS4 go-onebit (go1b) Go-domain corrector.
 *
 * Two output modes:
 *   sidecar (default): write a small STANDALONE GGUF containing ONLY the corr
 *                      tensors + ds4.corr.* metadata (~94 MiB for 43 layers).
 *                      Does NOT read/copy the base model GGUF at all.
 *                      -> gguf/ds4-go1b-corr.gguf
 *   merge  (--in IN):  copy every tensor of IN VERBATIM (streamed, RSS-bounded)
 *                      then append the corr tensors -> gguf/ds4-go1b-z.gguf.
 *                      Needs disk for the full IN copy + ~94 MiB.
 *
 * Per source layer L (L = 0 .. layers-1) read zdump/z_L{L}.bin and append six
 * F32 tensors. GGUF ne convention: ne[0] = innermost/contiguous dimension. The
 * tensor DATA is the z arrays copied verbatim (no transpose / no scaling); the
 * declared ne is exactly the GGUF view of the row-major dump:
 *
 *   blk.{L}.corr_U     ne=[d_l,     d_model]   U      row-major [d_model][d_l]
 *   blk.{L}.corr_V     ne=[d_model, d_l]       V      row-major [d_l][d_model]
 *   blk.{L}.corr_C     ne=[d_l,     n_exp]     C      row-major [n_exp][d_l]
 *   blk.{L}.corr_b     ne=[d_model]            b
 *   blk.{L}.corr_beta  ne=[n_exp]              beta
 *   blk.{L}.corr_delta ne=[n_exp]              delta
 *
 * z_L{L}.bin layout (little-endian):
 *   int32 d_model, int32 n_exp, int32 d_l, then f32 arrays U,V,C,b,beta,delta.
 *   size == 12 + (2*d_model*d_l + n_exp*d_l + d_model + 2*n_exp)*4   (asserted)
 *
 * d_l recovery at runtime (two independent ways):
 *   1. per layer, d_l == blk.{L}.corr_U.ne[0]  (also == blk.{L}.corr_C.ne[0]);
 *   2. explicit int32 KV  "ds4.corr.dl.{L}".
 * Presence gate: bool KV "ds4.corr.present" = true. Globals: int32 KVs
 * "ds4.corr.d_model", "ds4.corr.n_expert", "ds4.corr.n_present".
 *
 * Missing zdump/z_L{L}.bin files are skipped with a warning (so the tool can be
 * exercised before every layer is solved).
 *
 * Pure C99. The sidecar path has no dependency on the base GGUF.
 */

#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

/* --phi yhat: mark the sidecar so the runtime binds routed_out (ŷ) instead of
 * ffn_norm (x) as the corr latent feature (KV ds4.corr.phi_yhat). */
static int g_phi_yhat = 0;

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* ---- GGUF / GGML constants (subset actually used) ---- */
#define GGUF_VERSION       3u
#define GGUF_TYPE_INT32    5u
#define GGUF_TYPE_BOOL     7u
#define GGUF_TYPE_STRING   8u
#define GGUF_TYPE_ARRAY    9u
#define GGUF_TYPE_UINT32   4u
#define GGUF_TYPE_UINT64  10u
#define GGML_TYPE_F32      0u
#define DEFAULT_ALIGN      32u

#define COPY_BUF (8u * 1024u * 1024u)   /* stream-copy chunk (RSS bound) */

static void die(const char *msg) {
    fprintf(stderr, "emit_z: error: %s\n", msg);
    exit(1);
}
static void die_errno(const char *what, const char *path) {
    fprintf(stderr, "emit_z: error: %s %s: %s\n", what, path ? path : "", strerror(errno));
    exit(1);
}
static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static size_t pad_up(size_t x, size_t n) { return ((x + n - 1) / n) * n; }

/* ---- little-endian scalar I/O (host assumed LE, like the project writer) ---- */
static void w_u32(FILE *f, uint32_t v) { if (fwrite(&v, 4, 1, f) != 1) die("write u32"); }
static void w_u64(FILE *f, uint64_t v) { if (fwrite(&v, 8, 1, f) != 1) die("write u64"); }
static void w_u8(FILE *f, uint8_t v)   { if (fwrite(&v, 1, 1, f) != 1) die("write u8"); }
static void w_str(FILE *f, const char *s) {
    uint64_t n = strlen(s);
    w_u64(f, n);
    if (n && fwrite(s, 1, (size_t)n, f) != (size_t)n) die("write string");
}
static void w_kv_bool(FILE *f, const char *k, bool v) {
    w_str(f, k); w_u32(f, GGUF_TYPE_BOOL); w_u8(f, v ? 1 : 0);
}
static void w_kv_i32(FILE *f, const char *k, int32_t v) {
    w_str(f, k); w_u32(f, GGUF_TYPE_INT32); w_u32(f, (uint32_t)v);
}
static void w_kv_str(FILE *f, const char *k, const char *v) {
    w_str(f, k); w_u32(f, GGUF_TYPE_STRING); w_str(f, v);
}

static uint32_t rd_u32(FILE *f, const char *what) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) { fprintf(stderr, "emit_z: short read (%s)\n", what); exit(1); }
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}
static uint64_t rd_u64(FILE *f, const char *what) {
    uint8_t b[8];
    if (fread(b, 1, 8, f) != 8) { fprintf(stderr, "emit_z: short read (%s)\n", what); exit(1); }
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    return v;
}
static int32_t rd_i32_at(const uint8_t *p) {
    return (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}
static char *rd_gguf_string(FILE *f) {
    uint64_t n = rd_u64(f, "string length");
    char *s = xmalloc((size_t)n + 1);
    if (n && fread(s, 1, (size_t)n, f) != (size_t)n) die("short string read");
    s[n] = '\0';
    return s;
}

static void write_padding(FILE *f, size_t n) {
    static const uint8_t zeros[4096] = {0};
    while (n) {
        size_t c = n < sizeof(zeros) ? n : sizeof(zeros);
        if (fwrite(zeros, 1, c, f) != c) die("write padding");
        n -= c;
    }
}

/* Stream-copy `len` bytes from `in_path` starting at `in_off` into `out`. */
static void stream_copy(FILE *out, const char *in_path, uint64_t in_off, uint64_t len) {
    FILE *in = fopen(in_path, "rb");
    if (!in) die_errno("open", in_path);
    if (fseeko(in, (off_t)in_off, SEEK_SET) != 0) die_errno("seek", in_path);
    uint8_t *buf = xmalloc(COPY_BUF);
    uint64_t left = len;
    while (left) {
        size_t want = left < COPY_BUF ? (size_t)left : COPY_BUF;
        size_t got = fread(buf, 1, want, in);
        if (got != want) die_errno("read", in_path);
        if (fwrite(buf, 1, got, out) != got) die("write copied data");
        left -= got;
    }
    free(buf);
    fclose(in);
}

static uint64_t file_size(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) die_errno("stat", path);
    return (uint64_t)st.st_size;
}
static bool file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

/* ============================================================ z-dump layers */

typedef struct {
    int      L;
    char     path[1100];
    int      d_l;
    uint64_t z_size;
    /* byte offsets of each array inside the .bin (header is 12 bytes) */
    uint64_t off_U, off_V, off_C, off_b, off_beta, off_delta;
    uint64_t len_U, len_V, len_C, len_b, len_beta, len_delta;
} zlayer;

/* One output corr tensor descriptor (data lives in a z .bin on disk). */
typedef struct {
    char        name[64];
    int         n_dims;
    uint64_t    ne[4];
    uint32_t    type;       /* always GGML_TYPE_F32 here */
    uint64_t    size;       /* bytes */
    uint64_t    offset;     /* relative to GGUF data section (filled later) */
    const char *src_path;   /* z .bin to read data from */
    uint64_t    src_off;    /* byte offset within src_path */
} corr_tensor;

/* Scan z_L{L}.bin for L=0..layers-1; fill present[] and validate sizes.
 * Returns number of present layers; sets *d_model_out, *n_exp_out. */
static int scan_layers(const char *zdir, int layers, zlayer *out,
                       int *d_model_out, int *n_exp_out) {
    int n_present = 0;
    int g_dm = -1, g_ne = -1;
    for (int L = 0; L < layers; L++) {
        char path[1100];
        snprintf(path, sizeof(path), "%s/z_L%d.bin", zdir, L);
        if (!file_exists(path)) {
            fprintf(stderr, "emit_z: warning: %s missing — skipping layer %d\n", path, L);
            continue;
        }
        uint8_t hdr[12];
        FILE *f = fopen(path, "rb");
        if (!f) die_errno("open", path);
        if (fread(hdr, 1, 12, f) != 12) die("short z header read");
        fclose(f);
        int d_model = rd_i32_at(hdr);
        int n_exp   = rd_i32_at(hdr + 4);
        int d_l     = rd_i32_at(hdr + 8);
        if (d_model <= 0 || n_exp <= 0 || d_l <= 0)
            die("bad z header (non-positive dim)");
        if (g_dm < 0) { g_dm = d_model; g_ne = n_exp; }
        if (d_model != g_dm || n_exp != g_ne) {
            fprintf(stderr, "emit_z: layer %d d_model/n_exp (%d/%d) != %d/%d\n",
                    L, d_model, n_exp, g_dm, g_ne);
            die("inconsistent z dumps");
        }
        uint64_t lU = (uint64_t)d_model * d_l * 4;
        uint64_t lV = (uint64_t)d_l * d_model * 4;
        uint64_t lC = (uint64_t)n_exp * d_l * 4;
        uint64_t lb = (uint64_t)d_model * 4;
        uint64_t lbeta = (uint64_t)n_exp * 4;
        uint64_t ldelta = (uint64_t)n_exp * 4;
        uint64_t expect = 12 + lU + lV + lC + lb + lbeta + ldelta;
        uint64_t actual = file_size(path);
        if (actual != expect) {
            fprintf(stderr, "emit_z: %s size %" PRIu64 " != expected %" PRIu64
                    " (d_model=%d n_exp=%d d_l=%d)\n", path, actual, expect, d_model, n_exp, d_l);
            die("z-dump size assertion failed");
        }
        zlayer *z = &out[n_present++];
        z->L = L;
        snprintf(z->path, sizeof(z->path), "%s", path);
        z->d_l = d_l;
        z->z_size = actual;
        z->off_U = 12;
        z->off_V = z->off_U + lU;
        z->off_C = z->off_V + lV;
        z->off_b = z->off_C + lC;
        z->off_beta = z->off_b + lb;
        z->off_delta = z->off_beta + lbeta;
        z->len_U = lU; z->len_V = lV; z->len_C = lC;
        z->len_b = lb; z->len_beta = lbeta; z->len_delta = ldelta;
    }
    if (n_present == 0) die("no z_L*.bin layers found");
    *d_model_out = g_dm;
    *n_exp_out = g_ne;
    return n_present;
}

/* Build the 6 corr tensor descriptors for one present layer. */
static void build_layer_tensors(const zlayer *z, int d_model, int n_exp, corr_tensor *t) {
    int L = z->L, d_l = z->d_l;
    /* U: ne=[d_l, d_model] */
    snprintf(t[0].name, sizeof(t[0].name), "blk.%d.corr_U", L);
    t[0].n_dims = 2; t[0].ne[0] = (uint64_t)d_l; t[0].ne[1] = (uint64_t)d_model;
    t[0].size = z->len_U; t[0].src_off = z->off_U;
    /* V: ne=[d_model, d_l] */
    snprintf(t[1].name, sizeof(t[1].name), "blk.%d.corr_V", L);
    t[1].n_dims = 2; t[1].ne[0] = (uint64_t)d_model; t[1].ne[1] = (uint64_t)d_l;
    t[1].size = z->len_V; t[1].src_off = z->off_V;
    /* C: ne=[d_l, n_exp] */
    snprintf(t[2].name, sizeof(t[2].name), "blk.%d.corr_C", L);
    t[2].n_dims = 2; t[2].ne[0] = (uint64_t)d_l; t[2].ne[1] = (uint64_t)n_exp;
    t[2].size = z->len_C; t[2].src_off = z->off_C;
    /* b: ne=[d_model] */
    snprintf(t[3].name, sizeof(t[3].name), "blk.%d.corr_b", L);
    t[3].n_dims = 1; t[3].ne[0] = (uint64_t)d_model;
    t[3].size = z->len_b; t[3].src_off = z->off_b;
    /* beta: ne=[n_exp] */
    snprintf(t[4].name, sizeof(t[4].name), "blk.%d.corr_beta", L);
    t[4].n_dims = 1; t[4].ne[0] = (uint64_t)n_exp;
    t[4].size = z->len_beta; t[4].src_off = z->off_beta;
    /* delta: ne=[n_exp] */
    snprintf(t[5].name, sizeof(t[5].name), "blk.%d.corr_delta", L);
    t[5].n_dims = 1; t[5].ne[0] = (uint64_t)n_exp;
    t[5].size = z->len_delta; t[5].src_off = z->off_delta;
    for (int i = 0; i < 6; i++) {
        t[i].type = GGML_TYPE_F32;
        t[i].src_path = z->path;
    }
}

/* ============================================================ merge: input GGUF */

typedef struct {
    uint32_t version;
    uint64_t n_tensors;
    uint8_t *kv_kept;       /* kept KV records verbatim (ds4.corr.* dropped) */
    size_t   kv_kept_len;
    uint64_t n_kv_kept;
    uint8_t *tinfo_raw;     /* tensor-info block verbatim */
    size_t   tinfo_len;
    size_t   alignment;
    uint64_t data_offset;   /* input data section start (absolute) */
    uint64_t data_len;      /* input data section length (to EOF) */
    char    *path;
} input_gguf;

static void skip_gguf_value(FILE *f, uint32_t type, const char *path);

static void skip_gguf_array(FILE *f, const char *path) {
    uint32_t et = rd_u32(f, "array elem type");
    uint64_t n  = rd_u64(f, "array count");
    if (et == GGUF_TYPE_STRING) {
        for (uint64_t i = 0; i < n; i++) {
            uint64_t len = rd_u64(f, "array string length");
            if (fseeko(f, (off_t)len, SEEK_CUR) != 0) die_errno("seek", path);
        }
        return;
    }
    size_t sz;
    switch (et) {
        case 0: case 1: case GGUF_TYPE_BOOL: sz = 1; break;
        case 2: case 3: sz = 2; break;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case 6: sz = 4; break;
        case GGUF_TYPE_UINT64: case 11: case 12: sz = 8; break;
        default: die("unsupported GGUF array elem type"); return;
    }
    if (fseeko(f, (off_t)(n * sz), SEEK_CUR) != 0) die_errno("seek", path);
}

static void skip_gguf_value(FILE *f, uint32_t type, const char *path) {
    if (type == GGUF_TYPE_STRING) {
        uint64_t n = rd_u64(f, "string length");
        if (fseeko(f, (off_t)n, SEEK_CUR) != 0) die_errno("seek", path);
        return;
    }
    if (type == GGUF_TYPE_ARRAY) { skip_gguf_array(f, path); return; }
    size_t sz;
    switch (type) {
        case 0: case 1: case GGUF_TYPE_BOOL: sz = 1; break;
        case 2: case 3: sz = 2; break;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case 6: sz = 4; break;
        case GGUF_TYPE_UINT64: case 11: case 12: sz = 8; break;
        default: die("unsupported GGUF value type"); return;
    }
    if (fseeko(f, (off_t)sz, SEEK_CUR) != 0) die_errno("seek", path);
}

static input_gguf load_input(const char *path) {
    input_gguf g = {0};
    g.path = (char *)path;
    FILE *f = fopen(path, "rb");
    if (!f) die_errno("open GGUF", path);
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GGUF", 4) != 0) die("input not a GGUF file");
    g.version = rd_u32(f, "version");
    g.n_tensors = rd_u64(f, "tensor count");
    uint64_t n_kv = rd_u64(f, "kv count");
    g.alignment = DEFAULT_ALIGN;

    off_t kv_start = ftello(f);
    if (kv_start < 0) die("ftell");

    /* First pass: record keep-spans (drop ds4.corr.* so merge is idempotent),
     * and capture general.alignment. */
    typedef struct { off_t s, e; } span;
    span *keep = xmalloc((size_t)(n_kv ? n_kv : 1) * sizeof(*keep));
    uint64_t n_keep = 0;
    for (uint64_t i = 0; i < n_kv; i++) {
        off_t rs = ftello(f);
        char *key = rd_gguf_string(f);
        uint32_t type = rd_u32(f, "kv type");
        if (strcmp(key, "general.alignment") == 0 && type == GGUF_TYPE_UINT32) {
            uint32_t a = rd_u32(f, "alignment");
            if (a) g.alignment = a;
        } else {
            skip_gguf_value(f, type, path);
        }
        off_t re = ftello(f);
        if (strncmp(key, "ds4.corr.", 9) != 0) {
            keep[n_keep].s = rs; keep[n_keep].e = re; n_keep++;
        } else {
            fprintf(stderr, "emit_z: note: dropping stale input KV '%s'\n", key);
        }
        free(key);
    }
    off_t tensor_start = ftello(f);

    /* Capture kept KV bytes. */
    size_t kv_block_len = (size_t)(tensor_start - kv_start);
    uint8_t *kv_block = xmalloc(kv_block_len ? kv_block_len : 1);
    if (fseeko(f, kv_start, SEEK_SET) != 0) die_errno("seek", path);
    if (kv_block_len && fread(kv_block, 1, kv_block_len, f) != kv_block_len) die("kv block read");
    g.kv_kept_len = 0;
    for (uint64_t i = 0; i < n_keep; i++) g.kv_kept_len += (size_t)(keep[i].e - keep[i].s);
    g.kv_kept = xmalloc(g.kv_kept_len ? g.kv_kept_len : 1);
    size_t pos = 0;
    for (uint64_t i = 0; i < n_keep; i++) {
        size_t n = (size_t)(keep[i].e - keep[i].s);
        memcpy(g.kv_kept + pos, kv_block + (size_t)(keep[i].s - kv_start), n);
        pos += n;
    }
    g.n_kv_kept = n_keep;
    free(kv_block);
    free(keep);

    /* Walk tensor infos (verbatim block), reject re-appending into a file that
     * already has corr tensors. */
    if (fseeko(f, tensor_start, SEEK_SET) != 0) die_errno("seek", path);
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        char *name = rd_gguf_string(f);
        uint32_t nd = rd_u32(f, "tensor rank");
        if (nd < 1 || nd > 4) die("bad tensor rank");
        for (uint32_t j = 0; j < nd; j++) (void)rd_u64(f, "tensor dim");
        (void)rd_u32(f, "tensor type");
        (void)rd_u64(f, "tensor offset");
        if (strstr(name, ".corr_") != NULL) {
            fprintf(stderr, "emit_z: input already contains corr tensor '%s'\n", name);
            die("refusing to double-append corr tensors");
        }
        free(name);
    }
    off_t meta_end = ftello(f);
    g.tinfo_len = (size_t)(meta_end - tensor_start);
    g.tinfo_raw = xmalloc(g.tinfo_len ? g.tinfo_len : 1);
    if (fseeko(f, tensor_start, SEEK_SET) != 0) die_errno("seek", path);
    if (g.tinfo_len && fread(g.tinfo_raw, 1, g.tinfo_len, f) != g.tinfo_len) die("tinfo read");

    g.data_offset = pad_up((size_t)meta_end, g.alignment);
    uint64_t fsz = file_size(path);
    if (fsz < g.data_offset) die("input GGUF truncated");
    g.data_len = fsz - g.data_offset;
    fclose(f);
    return g;
}

/* ============================================================ writer */

static void emit(const char *zdir, int layers, const char *out_path,
                 const char *in_path, size_t cli_align) {
    zlayer *zl = xmalloc((size_t)layers * sizeof(*zl));
    int d_model = 0, n_exp = 0;
    int n_present = scan_layers(zdir, layers, zl, &d_model, &n_exp);

    /* Flatten corr tensor descriptors (6 per present layer). */
    int n_corr = n_present * 6;
    corr_tensor *ct = xmalloc((size_t)n_corr * sizeof(*ct));
    for (int p = 0; p < n_present; p++) build_layer_tensors(&zl[p], d_model, n_exp, &ct[p * 6]);

    bool merge = (in_path != NULL);
    input_gguf in = {0};
    size_t align = cli_align;
    if (merge) {
        in = load_input(in_path);
        align = in.alignment;   /* keep base file's alignment */
    }
    if (align == 0) align = DEFAULT_ALIGN;

    /* Relative data-section offsets for corr tensors. In merge mode the base
     * tensors occupy [0, data_len); corr starts after that, aligned. */
    uint64_t off0 = merge ? pad_up(in.data_len, align) : 0;
    uint64_t roff = off0;
    for (int i = 0; i < n_corr; i++) {
        ct[i].offset = roff;
        roff += pad_up(ct[i].size, align);
    }

    FILE *f = fopen(out_path, "wb");
    if (!f) die_errno("open output", out_path);

    /* ---- header ---- */
    if (fwrite("GGUF", 1, 4, f) != 4) die("write magic");
    w_u32(f, merge ? in.version : GGUF_VERSION);
    uint64_t out_n_tensors = (merge ? in.n_tensors : 0) + (uint64_t)n_corr;
    /* corr KV count: present + d_model + n_expert + n_present + dl.{L}*n_present
     * (+ general.architecture only for standalone sidecar) (+ phi_yhat if set). */
    uint64_t corr_kv = 4 + (uint64_t)n_present + (merge ? 0 : 1) + (g_phi_yhat ? 1 : 0);
    uint64_t out_n_kv = (merge ? in.n_kv_kept : 0) + corr_kv;
    w_u64(f, out_n_tensors);
    w_u64(f, out_n_kv);

    /* ---- KVs ---- */
    if (merge) {
        if (in.kv_kept_len && fwrite(in.kv_kept, 1, in.kv_kept_len, f) != in.kv_kept_len)
            die("write kept KV");
    } else {
        w_kv_str(f, "general.architecture", "ds4-corr");
    }
    w_kv_bool(f, "ds4.corr.present", true);
    if (g_phi_yhat) w_kv_bool(f, "ds4.corr.phi_yhat", true); /* --feat yhat z: runtime binds routed_out as φ */
    w_kv_i32(f, "ds4.corr.d_model", d_model);
    w_kv_i32(f, "ds4.corr.n_expert", n_exp);
    w_kv_i32(f, "ds4.corr.n_present", n_present);
    for (int p = 0; p < n_present; p++) {
        char key[64];
        snprintf(key, sizeof(key), "ds4.corr.dl.%d", zl[p].L);
        w_kv_i32(f, key, zl[p].d_l);
    }

    /* ---- tensor infos ---- */
    if (merge) {
        if (in.tinfo_len && fwrite(in.tinfo_raw, 1, in.tinfo_len, f) != in.tinfo_len)
            die("write base tensor infos");
    }
    for (int i = 0; i < n_corr; i++) {
        w_str(f, ct[i].name);
        w_u32(f, (uint32_t)ct[i].n_dims);
        for (int j = 0; j < ct[i].n_dims; j++) w_u64(f, ct[i].ne[j]);
        w_u32(f, ct[i].type);
        w_u64(f, ct[i].offset);
    }

    /* ---- pad to data section ---- */
    off_t pos = ftello(f);
    if (pos < 0) die("ftell");
    size_t data_offset = pad_up((size_t)pos, align);
    write_padding(f, data_offset - (size_t)pos);

    /* ---- tensor data ---- */
    if (merge) {
        /* base data section verbatim, then pad to align before corr. */
        stream_copy(f, in.path, in.data_offset, in.data_len);
        write_padding(f, (size_t)(off0 - in.data_len));
    }
    for (int i = 0; i < n_corr; i++) {
        stream_copy(f, ct[i].src_path, ct[i].src_off, ct[i].size);
        write_padding(f, pad_up(ct[i].size, align) - (size_t)ct[i].size);
    }
    if (fclose(f) != 0) die_errno("close output", out_path);

    double corr_mib = 0;
    for (int i = 0; i < n_corr; i++) corr_mib += (double)ct[i].size;
    corr_mib /= 1048576.0;
    fprintf(stderr,
            "emit_z: wrote %s  mode=%s  layers_present=%d/%d  corr_tensors=%d  "
            "d_model=%d n_exp=%d  align=%zu  corr_data=%.2f MiB\n",
            out_path, merge ? "merge" : "sidecar", n_present, layers, n_corr,
            d_model, n_exp, align, corr_mib);
    if (merge) {
        free(in.kv_kept); free(in.tinfo_raw);
    }
    free(ct); free(zl);
}

/* ============================================================ verify (--check) */

typedef struct {
    char    *name;
    int      n_dims;
    uint64_t ne[4];
    uint32_t type;
    uint64_t offset;
    uint64_t size;   /* derived from offset deltas (type-agnostic) */
} tinfo;

typedef struct {
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    size_t   alignment;
    uint64_t data_offset;
    tinfo   *t;
    /* a couple corr KVs we look for */
    bool     corr_present;
    int      kv_dl[256];   /* indexed by layer L (up to 256), -1 = absent */
} parsed;

static parsed parse_full(const char *path) {
    parsed g = {0};
    for (int i = 0; i < 256; i++) g.kv_dl[i] = -1;
    FILE *f = fopen(path, "rb");
    if (!f) die_errno("open GGUF", path);
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GGUF", 4) != 0) die("not a GGUF file");
    g.version = rd_u32(f, "version");
    g.n_tensors = rd_u64(f, "tensor count");
    g.n_kv = rd_u64(f, "kv count");
    g.alignment = DEFAULT_ALIGN;
    for (uint64_t i = 0; i < g.n_kv; i++) {
        char *key = rd_gguf_string(f);
        uint32_t type = rd_u32(f, "kv type");
        if (strcmp(key, "general.alignment") == 0 && type == GGUF_TYPE_UINT32) {
            uint32_t a = rd_u32(f, "alignment"); if (a) g.alignment = a;
        } else if (strcmp(key, "ds4.corr.present") == 0 && type == GGUF_TYPE_BOOL) {
            uint8_t b; if (fread(&b, 1, 1, f) != 1) die("read bool"); g.corr_present = b != 0;
        } else if (strncmp(key, "ds4.corr.dl.", 12) == 0 && type == GGUF_TYPE_INT32) {
            int L = atoi(key + 12);
            int32_t v = (int32_t)rd_u32(f, "dl");
            if (L >= 0 && L < 256) g.kv_dl[L] = v;
        } else {
            skip_gguf_value(f, type, path);
        }
        free(key);
    }
    g.t = xmalloc((size_t)(g.n_tensors ? g.n_tensors : 1) * sizeof(*g.t));
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        tinfo *t = &g.t[i];
        t->name = rd_gguf_string(f);
        t->n_dims = (int)rd_u32(f, "rank");
        if (t->n_dims < 1 || t->n_dims > 4) die("bad rank");
        for (int j = 0; j < t->n_dims; j++) t->ne[j] = rd_u64(f, "dim");
        t->type = rd_u32(f, "type");
        t->offset = rd_u64(f, "offset");
    }
    off_t meta_end = ftello(f);
    g.data_offset = pad_up((size_t)meta_end, g.alignment);
    fclose(f);
    /* derive each tensor size from offset deltas (sorted-by-offset assumption
     * holds for files this tool writes; fall back to file end for the last). */
    uint64_t fsz = file_size(path);
    uint64_t data_len = fsz - g.data_offset;
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        uint64_t end = (i + 1 < g.n_tensors) ? g.t[i + 1].offset : data_len;
        g.t[i].size = end - g.t[i].offset;  /* padded size */
    }
    return g;
}

static tinfo *find_tensor(parsed *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (strcmp(g->t[i].name, name) == 0) return &g->t[i];
    return NULL;
}

/* read exactly len bytes at data_offset+rel into buf */
static void read_at(const char *path, uint64_t abs_off, void *buf, uint64_t len) {
    FILE *f = fopen(path, "rb");
    if (!f) die_errno("open", path);
    if (fseeko(f, (off_t)abs_off, SEEK_SET) != 0) die_errno("seek", path);
    if (len && fread(buf, 1, (size_t)len, f) != (size_t)len) die_errno("read", path);
    fclose(f);
}

static int check_cmd(const char *path, const char *zdir, int layers, const char *ref_path) {
    parsed g = parse_full(path);
    printf("== %s\n", path);
    printf("version=%u n_tensors=%" PRIu64 " n_kv=%" PRIu64 " alignment=%zu data_offset=%" PRIu64 "\n",
           g.version, g.n_tensors, g.n_kv, g.alignment, g.data_offset);
    printf("ds4.corr.present=%s\n", g.corr_present ? "true" : "false");
    int fail = 0;

    const char *parts[6] = { "corr_U", "corr_V", "corr_C", "corr_b", "corr_beta", "corr_delta" };
    int n_layers_found = 0;
    for (int L = 0; L < layers; L++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "blk.%d.corr_U", L);
        tinfo *u = find_tensor(&g, nm);
        if (!u) continue;
        n_layers_found++;
        int d_l_shape = (int)u->ne[0];
        int d_l_kv = (L < 256) ? g.kv_dl[L] : -1;
        printf("layer %d: d_l(shape U.ne[0])=%d  d_l(KV)=%d%s\n",
               L, d_l_shape, d_l_kv, (d_l_kv >= 0 && d_l_kv != d_l_shape) ? "  <<< MISMATCH" : "");
        if (d_l_kv >= 0 && d_l_kv != d_l_shape) fail = 1;
        for (int pi = 0; pi < 6; pi++) {
            snprintf(nm, sizeof(nm), "blk.%d.%s", L, parts[pi]);
            tinfo *t = find_tensor(&g, nm);
            if (!t) { printf("  MISSING %s\n", nm); fail = 1; continue; }
            printf("  %-22s nd=%d ne=[", nm, t->n_dims);
            for (int j = 0; j < t->n_dims; j++) printf("%s%" PRIu64, j ? "," : "", t->ne[j]);
            printf("] type=%u off=%" PRIu64 " size~%" PRIu64 "\n", t->type, t->offset, t->size);
            if (t->type != GGML_TYPE_F32) { printf("    <<< type != F32\n"); fail = 1; }
        }
    }
    printf("corr layers found: %d\n", n_layers_found);

    /* Round-trip corr DATA against z dumps, if available. */
    if (zdir) {
        zlayer *zl = xmalloc((size_t)layers * sizeof(*zl));
        int d_model = 0, n_exp = 0;
        int n_present = scan_layers(zdir, layers, zl, &d_model, &n_exp);
        for (int p = 0; p < n_present; p++) {
            zlayer *z = &zl[p];
            corr_tensor ct[6];
            build_layer_tensors(z, d_model, n_exp, ct);
            for (int i = 0; i < 6; i++) {
                tinfo *t = find_tensor(&g, ct[i].name);
                if (!t) { printf("roundtrip: %s absent in GGUF\n", ct[i].name); fail = 1; continue; }
                uint8_t *a = xmalloc(ct[i].size);
                uint8_t *b = xmalloc(ct[i].size);
                read_at(path, g.data_offset + t->offset, a, ct[i].size);
                read_at(ct[i].src_path, ct[i].src_off, b, ct[i].size);
                int ok = memcmp(a, b, (size_t)ct[i].size) == 0;
                if (!ok) { printf("roundtrip: %s DATA MISMATCH\n", ct[i].name); fail = 1; }
                free(a); free(b);
            }
            printf("roundtrip layer %d: %s\n", z->L, "data checked");
        }
        free(zl);
    }

    /* Compare original tensors against a reference GGUF (merge byte-identity). */
    if (ref_path) {
        parsed r = parse_full(ref_path);
        int compared = 0, mism = 0;
        for (uint64_t i = 0; i < r.n_tensors; i++) {
            tinfo *rt = &r.t[i];
            tinfo *ot = find_tensor(&g, rt->name);
            if (!ot) { printf("ref tensor '%s' MISSING in output\n", rt->name); fail = 1; continue; }
            /* compare unpadded-or-padded: use min of the two derived sizes */
            uint64_t n = rt->size < ot->size ? rt->size : ot->size;
            uint8_t *a = xmalloc(n ? n : 1);
            uint8_t *b = xmalloc(n ? n : 1);
            read_at(ref_path, r.data_offset + rt->offset, a, n);
            read_at(path, g.data_offset + ot->offset, b, n);
            if (memcmp(a, b, (size_t)n) != 0) { printf("ref tensor '%s' DATA MISMATCH\n", rt->name); mism++; fail = 1; }
            free(a); free(b);
            compared++;
        }
        printf("ref-compare: %d tensors compared, %d mismatched\n", compared, mism);
        for (uint64_t i = 0; i < r.n_tensors; i++) free(r.t[i].name);
        free(r.t);
    }

    for (uint64_t i = 0; i < g.n_tensors; i++) free(g.t[i].name);
    free(g.t);
    printf("CHECK: %s\n", fail ? "FAIL" : "OK");
    return fail ? 1 : 0;
}

/* ============================================================ make-synth (test) */

/* Write a tiny valid GGUF with 3 small F32 tensors + a few KVs, for exercising
 * the merge path without the 45 GiB base model. Deterministic contents. */
static void make_synth(const char *path, size_t align) {
    if (align == 0) align = DEFAULT_ALIGN;
    struct { const char *name; int nd; uint64_t ne[2]; uint64_t n; } T[3] = {
        { "synth.a", 1, {4, 0}, 4 },
        { "synth.b", 2, {2, 3}, 6 },
        { "synth.c", 1, {8, 0}, 8 },
    };
    /* relative offsets */
    uint64_t off[3], roff = 0;
    for (int i = 0; i < 3; i++) { off[i] = roff; roff += pad_up(T[i].n * 4, align); }

    FILE *f = fopen(path, "wb");
    if (!f) die_errno("open", path);
    if (fwrite("GGUF", 1, 4, f) != 4) die("magic");
    w_u32(f, GGUF_VERSION);
    w_u64(f, 3);     /* n_tensors */
    w_u64(f, 2);     /* n_kv */
    w_kv_str(f, "general.architecture", "synthtest");
    { w_str(f, "general.alignment"); w_u32(f, GGUF_TYPE_UINT32); w_u32(f, (uint32_t)align); }
    for (int i = 0; i < 3; i++) {
        w_str(f, T[i].name);
        w_u32(f, (uint32_t)T[i].nd);
        for (int j = 0; j < T[i].nd; j++) w_u64(f, T[i].ne[j]);
        w_u32(f, GGML_TYPE_F32);
        w_u64(f, off[i]);
    }
    off_t pos = ftello(f);
    size_t data_off = pad_up((size_t)pos, align);
    write_padding(f, data_off - (size_t)pos);
    float seed = 1.0f;
    for (int i = 0; i < 3; i++) {
        for (uint64_t k = 0; k < T[i].n; k++) {
            float v = seed; seed += 1.0f;
            if (fwrite(&v, 4, 1, f) != 1) die("write synth data");
        }
        write_padding(f, pad_up(T[i].n * 4, align) - T[i].n * 4);
    }
    if (fclose(f) != 0) die_errno("close", path);
    fprintf(stderr, "emit_z: wrote synthetic GGUF %s (3 F32 tensors, align=%zu)\n", path, align);
}

/* ============================================================ CLI */

static void usage(void) {
    fprintf(stderr,
        "usage:\n"
        "  emit_z --out OUT.gguf [--zdir DIR] [--layers N] [--align N]\n"
        "         (sidecar: write ONLY corr tensors; no base model needed)\n"
        "  emit_z --in BASE.gguf --out OUT.gguf [--zdir DIR] [--layers N]\n"
        "         (merge: copy BASE verbatim, then append corr tensors)\n"
        "  emit_z --check FILE.gguf [--zdir DIR] [--layers N] [--ref REF.gguf]\n"
        "         (parse + verify corr tensors / z round-trip / ref byte-identity)\n"
        "  emit_z --make-synth OUT.gguf [--align N]   (tiny test GGUF)\n"
        "defaults: --zdir zdump  --layers 43  --align 32\n");
}

int main(int argc, char **argv) {
    const char *zdir = "zdump";
    const char *out_path = NULL;
    const char *in_path = NULL;
    const char *check_path = NULL;
    const char *ref_path = NULL;
    const char *synth_path = NULL;
    int layers = 43;
    size_t align = DEFAULT_ALIGN;
    bool zdir_set = false;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--zdir") && i + 1 < argc) { zdir = argv[++i]; zdir_set = true; }
        else if (!strcmp(a, "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(a, "--in") && i + 1 < argc) in_path = argv[++i];
        else if (!strcmp(a, "--check") && i + 1 < argc) check_path = argv[++i];
        else if (!strcmp(a, "--ref") && i + 1 < argc) ref_path = argv[++i];
        else if (!strcmp(a, "--make-synth") && i + 1 < argc) synth_path = argv[++i];
        else if (!strcmp(a, "--layers") && i + 1 < argc) layers = atoi(argv[++i]);
        else if (!strcmp(a, "--phi") && i + 1 < argc) g_phi_yhat = !strcmp(argv[++i], "yhat");
        else if (!strcmp(a, "--align") && i + 1 < argc) align = (size_t)atoi(argv[++i]);
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(); return 0; }
        else { fprintf(stderr, "emit_z: unknown arg '%s'\n", a); usage(); return 2; }
    }
    if (layers < 1 || layers > 256) die("--layers out of range (1..256)");

    if (synth_path) { make_synth(synth_path, align); return 0; }
    if (check_path) return check_cmd(check_path, zdir_set ? zdir : zdir, layers, ref_path);
    if (!out_path) { usage(); return 2; }
    emit(zdir, layers, out_path, in_path, align);
    return 0;
}
