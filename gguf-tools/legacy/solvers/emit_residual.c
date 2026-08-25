/* emit_residual.c — emit per-(layer) 1-bit RESIDUAL  Q1(W − Q1(W))  as go1b
 * tensors in a standalone sidecar GGUF for the ds4 go1b model.
 *
 * The single-1-bit expert weight Q1(W)=sign(W)·mean|W| loses ~half the output
 * direction (base_cos ~0.52). A SECOND 1-bit layer on the residual R=W−Q1(W),
 * Q1(R)=sign(R)·mean|R|, recovers most of it (base_cos ~0.82, validated by
 * calib_run DS4_GO1B_RESIDUAL). At runtime the expert weight is reconstructed as
 *      W ≈ dequant(base go1b)  +  dequant(residual go1b)
 * i.e. the routed-MoE forward runs the base mm_id then a SECOND mm_id over the
 * residual tensor for the same selected experts and SUMS the outputs.
 *
 * --active-experts FILE makes the residual SPARSE: only the experts a Go workload
 * actually routes to (per layer, "L{n}: e0 e1 ...") get a residual; the rest stay
 * pure 1-bit. This keeps the sidecar small (Go uses ~14% of experts at 80% pick
 * coverage). Each sparse layer also writes blk.{L}.ffn_res_lut.weight (F32[256],
 * expert id -> slot in [0,K), or -1) so the runtime maps a routed expert to its
 * residual slot (or skips it).
 *
 * Tensors written (go1b ggml type 40, same ne convention as the base):
 *   blk.{L}.ffn_gate_exps_res.weight  ne=[DM, DF, K]
 *   blk.{L}.ffn_up_exps_res.weight    ne=[DM, DF, K]
 *   blk.{L}.ffn_down_exps_res.weight  ne=[DF, DM, K]
 *   blk.{L}.ffn_res_lut.weight        ne=[256,1,1] F32   (sparse only)
 * KVs: ds4.residual.present(bool), ds4.residual.sparse(bool),
 *      ds4.residual.n_present(i32), ds4.residual.layer.{i}(i32).
 *
 * Usage: ./emit_residual --hf DIR --out FILE --layers 0,1,2 [--n-experts 256]
 *                        [--active-experts go_active.txt]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include "hf_read.h"
#include "onebit_quant.h"

/* DeepSeek-V4-Flash dims (match calib_run). */
#define DM 4096   /* hidden / d_model */
#define DF 2048   /* moe_intermediate per expert */
#define NEXP 256
#define MAXL 64

#define GGUF_VERSION 3
#define DEFAULT_ALIGN 32
enum { GGUF_TYPE_UINT32 = 4, GGUF_TYPE_INT32 = 5, GGUF_TYPE_BOOL = 7, GGUF_TYPE_STRING = 8 };
#define GGML_TYPE_F32  0
#define GGML_TYPE_GO1B 40

static void die(const char *m) { fprintf(stderr, "emit_residual: %s\n", m); exit(1); }
static void die_errno(const char *m) { fprintf(stderr, "emit_residual: %s: %s\n", m, strerror(errno)); exit(1); }
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("OOM"); return p; }

/* ---- little-endian GGUF writers (mirror emit_z.c) ---- */
static void w_u32(FILE *f, uint32_t v) { if (fwrite(&v, 4, 1, f) != 1) die("write u32"); }
static void w_u64(FILE *f, uint64_t v) { if (fwrite(&v, 8, 1, f) != 1) die("write u64"); }
static void w_str(FILE *f, const char *s) {
    uint64_t n = (uint64_t)strlen(s);
    w_u64(f, n);
    if (n && fwrite(s, 1, (size_t)n, f) != (size_t)n) die("write string");
}
static void w_kv_bool(FILE *f, const char *k, bool v) {
    w_str(f, k); w_u32(f, GGUF_TYPE_BOOL); uint8_t b = v ? 1 : 0;
    if (fwrite(&b, 1, 1, f) != 1) die("write bool");
}
static void w_kv_i32(FILE *f, const char *k, int32_t v) {
    w_str(f, k); w_u32(f, GGUF_TYPE_INT32); w_u32(f, (uint32_t)v);
}
static void w_kv_str(FILE *f, const char *k, const char *v) {
    w_str(f, k); w_u32(f, GGUF_TYPE_STRING); w_str(f, v);
}
static size_t pad_up(size_t n, size_t a) { return (n + a - 1) / a * a; }
static void write_padding(FILE *f, size_t n) {
    static const uint8_t z[64] = {0};
    while (n) { size_t c = n < sizeof(z) ? n : sizeof(z); if (fwrite(z, 1, c, f) != c) die("write pad"); n -= c; }
}

/* residual go1b of one [nrows x ncols] matrix into out (go1b_blk bytes). */
static void residual_matrix(const float *W, int nrows, int ncols,
                            unsigned char *scratch, float *wh, unsigned char *out) {
    size_t rb = go1b_blk_row_bytes(ncols);
    go1b_blk_quantize(W, scratch, nrows, ncols);                 /* Q1(W) */
    for (int r = 0; r < nrows; r++)
        go1b_blk_dequantize_row(scratch + (size_t)r * rb, wh + (size_t)r * ncols, ncols);
    size_t n = (size_t)nrows * (size_t)ncols;
    for (size_t i = 0; i < n; i++) wh[i] = W[i] - wh[i];          /* R = W - Q1(W) */
    go1b_blk_quantize(wh, out, nrows, ncols);                    /* Q1(R) */
}

/* --base-gguf: dequant the DEPLOYED base bytes instead of recomputing Q1(W).
 * The shipped Θ_fix may use any scale recipe (v2: per-row scale replicated
 * into every 34-byte block; a rebuild may use weighted or per-block scales).
 * Recomputing here silently residuals against the wrong base; reading the
 * file bytes is exact against whatever was emitted. */
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

typedef struct { char name[80]; uint64_t off; } bg_tensor;
typedef struct {
    unsigned char *map; size_t size; uint64_t data0;
    bg_tensor *t; int n_t;
} base_gguf;

static uint64_t bg_r_u64(const unsigned char **p) { uint64_t v; memcpy(&v, *p, 8); *p += 8; return v; }
static uint32_t bg_r_u32(const unsigned char **p) { uint32_t v; memcpy(&v, *p, 4); *p += 4; return v; }
static void bg_r_str(const unsigned char **p, char *dst, size_t cap) {
    uint64_t n = bg_r_u64(p);
    if (dst) { size_t c = n < cap-1 ? n : cap-1; memcpy(dst, *p, c); dst[c] = 0; }
    *p += n;
}
static void bg_skip_val(const unsigned char **p, uint32_t ty) {
    static const int sz[13] = {1,1,2,2,4,4,4,1,-1,-2,8,8,8};
    if (ty == 8) { bg_r_str(p, NULL, 0); return; }
    if (ty == 9) {
        uint32_t et = bg_r_u32(p); uint64_t n = bg_r_u64(p);
        if (et == 8) { for (uint64_t i = 0; i < n; i++) bg_r_str(p, NULL, 0); }
        else *p += (uint64_t)sz[et] * n;
        return;
    }
    *p += sz[ty];
}

static base_gguf *bg_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) die_errno("open --base-gguf");
    struct stat st; if (fstat(fd, &st) != 0) die_errno("stat --base-gguf");
    base_gguf *bg = xmalloc(sizeof *bg);
    bg->size = (size_t)st.st_size;
    bg->map = mmap(NULL, bg->size, PROT_READ, MAP_SHARED, fd, 0);
    if (bg->map == MAP_FAILED) die_errno("mmap --base-gguf");
    close(fd);
    const unsigned char *p = bg->map;
    if (memcmp(p, "GGUF", 4) != 0) die("--base-gguf: bad magic");
    p += 4; (void)bg_r_u32(&p);
    uint64_t n_t = bg_r_u64(&p), n_kv = bg_r_u64(&p);
    for (uint64_t i = 0; i < n_kv; i++) { bg_r_str(&p, NULL, 0); uint32_t ty = bg_r_u32(&p); bg_skip_val(&p, ty); }
    bg->n_t = (int)n_t;
    bg->t = xmalloc(n_t * sizeof(bg_tensor));
    for (uint64_t i = 0; i < n_t; i++) {
        bg_tensor *bt = &bg->t[i];
        bg_r_str(&p, bt->name, sizeof bt->name);
        uint32_t nd = bg_r_u32(&p); p += 8 * nd;
        (void)bg_r_u32(&p);
        bt->off = bg_r_u64(&p);
    }
    uint64_t hdr = (uint64_t)(p - bg->map);
    bg->data0 = (hdr + DEFAULT_ALIGN - 1) / DEFAULT_ALIGN * DEFAULT_ALIGN;
    return bg;
}

/* Dequant expert e of blk.{L}.ffn_{kind}_exps.weight into wh[rows*cols]. */
static void bg_expert_dequant(const base_gguf *bg, int L, const char *kind,
                              int e, int rows, int cols, float *wh) {
    char want[80];
    snprintf(want, sizeof want, "blk.%d.ffn_%s_exps.weight", L, kind);
    const bg_tensor *bt = NULL;
    for (int i = 0; i < bg->n_t; i++)
        if (!strcmp(bg->t[i].name, want)) { bt = &bg->t[i]; break; }
    if (!bt) { fprintf(stderr, "emit_residual: %s missing in --base-gguf\n", want); exit(1); }
    size_t rb = go1b_blk_row_bytes(cols);
    const unsigned char *src = bg->map + bg->data0 + bt->off + (size_t)e * rows * rb;
    if (bg->data0 + bt->off + (size_t)(e + 1) * rows * rb > bg->size) die("--base-gguf: expert range past EOF");
    for (int r = 0; r < rows; r++)
        go1b_blk_dequantize_row(src + (size_t)r * rb, wh + (size_t)r * cols, cols);
}

/* Base-exact variant: wh already holds dequant(base); out = Q1(W - wh). */
static void residual_matrix_from_base(const float *W, int nrows, int ncols,
                                      float *wh, unsigned char *out) {
    size_t n = (size_t)nrows * (size_t)ncols;
    for (size_t i = 0; i < n; i++) wh[i] = W[i] - wh[i];
    go1b_blk_quantize(wh, out, nrows, ncols);
}

typedef struct { char name[64]; uint64_t ne[3]; uint64_t size, offset; char src[256]; int type; } restensor;

/* Parse "L{n}: e0 e1 ..." lines into active[li][*]/nact[li] keyed by the layer's
 * index in layers[]. Returns 1 on success. Layers missing from the file get 0. */
static int parse_active(const char *path, const int *layers, int nL,
                        int active[][NEXP], int *nact) {
    FILE *f = fopen(path, "r"); if (!f) return 0;
    for (int i = 0; i < nL; i++) nact[i] = 0;
    char line[8192];
    while (fgets(line, sizeof line, f)) {
        int L; char *p = line;
        if (sscanf(p, "L%d:", &L) != 1) continue;
        int li = -1; for (int i = 0; i < nL; i++) if (layers[i] == L) { li = i; break; }
        if (li < 0) continue;
        p = strchr(p, ':'); if (!p) continue; p++;
        int cnt = 0, e;
        while (cnt < NEXP && sscanf(p, "%d", &e) == 1) {
            if (e >= 0 && e < NEXP) active[li][cnt++] = e;
            while (*p == ' ') p++;
            while (*p && *p != ' ') p++;   /* skip the number we just read */
        }
        nact[li] = cnt;
    }
    fclose(f);
    return 1;
}

int main(int argc, char **argv) {
    const char *hf_dir = NULL, *out_path = "gguf/ds4-go1b-res.gguf", *layers_s = "0";
    const char *active_path = NULL, *base_path = NULL;
    int n_exp = NEXP;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--hf") && i+1<argc) hf_dir = argv[++i];
        else if (!strcmp(argv[i], "--out") && i+1<argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--layers") && i+1<argc) layers_s = argv[++i];
        else if (!strcmp(argv[i], "--base-gguf") && i+1<argc) base_path = argv[++i];
        else if (!strcmp(argv[i], "--active-experts") && i+1<argc) active_path = argv[++i];
        else if ((!strcmp(argv[i], "--n-experts")||!strcmp(argv[i],"--nexp")) && i+1<argc) {
            n_exp = atoi(argv[++i]); if (n_exp<1) n_exp=1; if (n_exp>NEXP) n_exp=NEXP; }
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (!hf_dir) die("need --hf DIR");

    int layers[MAXL], nL = 0;
    { char buf[512]; strncpy(buf, layers_s, sizeof buf -1); buf[sizeof buf-1]=0;
      for (char *t = strtok(buf, ","); t && nL < MAXL; t = strtok(NULL, ",")) layers[nL++] = atoi(t); }

    /* sparse active-expert sets (per layer). Absent => dense (all n_exp). */
    static int active[MAXL][NEXP]; static int nact[MAXL];
    int sparse = 0;
    for (int i = 0; i < nL; i++) nact[i] = n_exp;
    if (active_path) {
        if (!parse_active(active_path, layers, nL, active, nact)) die("open --active-experts");
        sparse = 1;
    }

    hf_db *db = hf_open(hf_dir);
    if (!db) die("hf_open failed");
    base_gguf *bg = base_path ? bg_open(base_path) : NULL;
    if (bg) fprintf(stderr, "base bytes from %s (%d tensors) — exact deployed Q1\n", base_path, bg->n_t);

    size_t maxcols = DM, maxrows = DF > DM ? DF : DM;
    unsigned char *scratch = xmalloc(maxrows * go1b_blk_row_bytes(maxcols));
    float *wh = xmalloc(maxrows * maxcols * sizeof(float));
    unsigned char *outbuf = xmalloc(maxrows * go1b_blk_row_bytes(maxcols));

    /* ★流式单遍(2026-07-23 盘墙修复)★: 张量尺寸全确定(rowbytes*K), 无需先暂存再组装。
     * Pass A 只填元数据(不读 HF)→ 写 header → Pass B 逐张量即算即写输出。峰值盘 =
     * 输出文件 + 单张量 RAM(~68MiB), 不再是全暂存~8G。旧 /tmp/res_L*.bin 暂存路径删除。 */
    int per_layer_t = sparse ? 4 : 3;
    int n_t = nL * per_layer_t;
    restensor *t = xmalloc((size_t)n_t * sizeof(*t));
    struct { const char *gguf; const char *hf; int rows, cols; } kinds[3] = {
        {"gate", "w1", DF, DM}, {"up", "w3", DF, DM}, {"down", "w2", DM, DF} };
    /* Pass A: 元数据 (name/ne/size) — 尺寸由维度定, 不碰数据 */
    int ti = 0;
    for (int li = 0; li < nL; li++) {
        int L = layers[li], K = nact[li];
        for (int k = 0; k < 3; k++) {
            restensor *rt = &t[ti++];
            rt->type = GGML_TYPE_GO1B;
            snprintf(rt->name, sizeof rt->name, "blk.%d.ffn_%s_exps_res.weight", L, kinds[k].gguf);
            rt->ne[0] = (uint64_t)kinds[k].cols; rt->ne[1] = (uint64_t)kinds[k].rows; rt->ne[2] = (uint64_t)K;
            rt->size = (size_t)kinds[k].rows * go1b_blk_row_bytes(kinds[k].cols) * (size_t)K;
        }
        if (sparse) {
            restensor *rt = &t[ti++];
            rt->type = GGML_TYPE_F32;
            snprintf(rt->name, sizeof rt->name, "blk.%d.ffn_res_lut.weight", L);
            rt->ne[0] = NEXP; rt->ne[1] = 1; rt->ne[2] = 1;
            rt->size = (size_t)NEXP * sizeof(float);
        }
    }
    size_t align = DEFAULT_ALIGN, roff = 0;
    for (int i = 0; i < n_t; i++) { t[i].offset = roff; roff += pad_up(t[i].size, align); }

    /* header */
    FILE *f = fopen(out_path, "wb"); if (!f) die_errno("open out");
    if (fwrite("GGUF", 1, 4, f) != 4) die("magic");
    w_u32(f, GGUF_VERSION);
    w_u64(f, (uint64_t)n_t);
    w_u64(f, (uint64_t)(4 + nL));
    w_kv_str(f, "general.architecture", "ds4-residual");
    w_kv_bool(f, "ds4.residual.present", true);
    w_kv_bool(f, "ds4.residual.sparse", sparse ? true : false);
    w_kv_i32(f, "ds4.residual.n_present", nL);
    for (int li = 0; li < nL; li++) { char key[48]; snprintf(key, sizeof key, "ds4.residual.layer.%d", li); w_kv_i32(f, key, layers[li]); }
    for (int i = 0; i < n_t; i++) {
        w_str(f, t[i].name); w_u32(f, 3);
        for (int j = 0; j < 3; j++) w_u64(f, t[i].ne[j]);
        w_u32(f, (uint32_t)t[i].type); w_u64(f, t[i].offset);
    }
    long pos = ftell(f); if (pos < 0) die("ftell");
    size_t data_off = pad_up((size_t)pos, align);
    write_padding(f, data_off - (size_t)pos);

    /* Pass B: 逐张量即算即写(顺序须与 Pass A 元数据一致) */
    for (int li = 0; li < nL; li++) {
        int L = layers[li], K = nact[li];
        const int *elist = sparse ? active[li] : NULL;
        for (int k = 0; k < 3; k++) {
            size_t rowbytes = (size_t)kinds[k].rows * go1b_blk_row_bytes(kinds[k].cols);
            size_t tsize = rowbytes * (size_t)K;
            for (int s = 0; s < K; s++) {
                int e = sparse ? elist[s] : s;
                char nm[96]; int64_t n;
                snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.%s.weight", L, e, kinds[k].hf);
                float *W = hf_read_f32(db, nm, &n);
                if (!W || n != (int64_t)kinds[k].rows * kinds[k].cols) { fprintf(stderr, "read %s n=%lld\n", nm, (long long)(W?n:-1)); die("hf_read"); }
                if (bg) {
                    bg_expert_dequant(bg, L, kinds[k].gguf, e, kinds[k].rows, kinds[k].cols, wh);
                    residual_matrix_from_base(W, kinds[k].rows, kinds[k].cols, wh, outbuf);
                } else {
                    residual_matrix(W, kinds[k].rows, kinds[k].cols, scratch, wh, outbuf);
                }
                if (fwrite(outbuf, 1, rowbytes, f) != rowbytes) die("write out");
                free(W);
            }
            write_padding(f, pad_up(tsize, align) - tsize);
            fprintf(stderr, "  blk.%d.ffn_%s_exps_res ne=[%d,%d,%d] %.1f MiB (streamed)\n",
                    L, kinds[k].gguf, kinds[k].cols, kinds[k].rows, K, tsize/1048576.0);
        }
        if (sparse) {
            float lut[NEXP]; for (int e = 0; e < NEXP; e++) lut[e] = -1.0f;
            for (int s = 0; s < K; s++) lut[elist[s]] = (float)s;
            size_t lsize = (size_t)NEXP * sizeof(float);
            if (fwrite(lut, sizeof(float), NEXP, f) != NEXP) die("write lut");
            write_padding(f, pad_up(lsize, align) - lsize);
        }
    }
    hf_close(db);
    free(scratch); free(wh); free(outbuf);
    if (fclose(f) != 0) die_errno("close out");
    double mib = 0; for (int i = 0; i < n_t; i++) mib += t[i].size; mib /= 1048576.0;
    fprintf(stderr, "emit_residual: wrote %s  layers=%d  tensors=%d  sparse=%d  data=%.1f MiB\n",
            out_path, nL, n_t, sparse, mib);
    free(t);
    return 0;
}
