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

