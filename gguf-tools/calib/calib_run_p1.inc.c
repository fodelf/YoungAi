/* calib_run.c — Go-domain 1-bit hypothesis test driver (SPEC.md §6/§7, 表 P/B).
 *
 * Ties together the go-onebit modules to answer the core question from real data:
 *   "In the Go domain, is each layer's 256-expert 1-bit reconstruction error Δo
 *    low-dimensional and recoverable by the per-layer hidden variable z^ℓ?"
 *
 * Pipeline, per transformer layer L:
 *   1. x_i  = cap_m1/ffn_in_L{L}.npy  (per-layer FFN input, 12288×4096 fp16) → subsample n_x.
 *   2. For every routed expert e (FULL 256, not just top-6):
 *        o_ref_{e,i} = expert_forward_f32(x_i, HF FP8 weights)        (true original)
 *        ô_{e,i}     = expert_forward_f32(x_i, go1b(HF) dequantized)  (strict 1-bit)
 *        Δo_{e,i}    = o_ref − ô
 *   3. hv_solve → z^ℓ + eigen-spectrum (表 P: per-layer rank d_ℓ).
 *   4. hv_fidelity → 表 B: rel-L2 / cosine of o^corr vs o_ref, vs the 1-bit baseline.
 *
 * Expert weights stream from HF (one expert resident per worker at a time); each
 * worker opens its OWN hf_db over a disjoint expert band and writes a disjoint
 * output region — no shared FILE*, no locks, RAM-safe (peak a few GiB).
 *
 * Reference behavior is the ORIGINAL HF model (FP8 dequant), never a pre-quantized
 * GGUF. cap_m1 supplies only the Go-domain x_i (input activations).
 *
 * Pure C99. Build: see gguf-tools/Makefile target `calib_run`.
 */
#include "npy.h"
#include "hf_read.h"
#include "onebit_quant.h"
#include "layer_probe.h"
#include "hiddenvar_solve.h"
#include "solve_perexpert.h"
#include "solve_denoise.h"
#include "solve_genrf.h"
#include "solve_rrr.h"

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DM 4096   /* hidden_size        */
#define DF 2048   /* moe_intermediate   */
#define NEXP 256  /* routed experts     */

/* ---- shared worker context --------------------------------------------- */
typedef struct {
    const char *hf_dir;
    int    layer;
    int    e_begin, e_end;     /* this worker's expert band [e_begin,e_end)  */
    int    n_x;
    const float *x_f32;        /* [n_x*DM] subsampled Go inputs (float)      */
    double *o_ref;             /* [NEXP*n_x*DM] out (this band's slots)      */
    double *o_hat;             /* [NEXP*n_x*DM] out                          */
    double *delta;             /* [NEXP*n_x*DM] out                          */
    int    rc;                 /* 0 ok, nonzero on failure                   */
    int    done_experts;
} worker_ctx;

/* ---- optional L_fix stats (--go-stats): llama.cpp imatrix .dat ----------- */
/* entries "blk.{L}.ffn_{gate,up,down}_exps.weight", nval = ncols*256 (per-
 * expert segmented) or ncols (shared). Loaded once, read-only by workers. */
typedef struct { char *name; float *vals; int nval; } gs_entry;
static gs_entry *g_gs = NULL;
static int g_gs_n = 0;

static int gostats_load(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "go-stats: cannot open %s\n", path); return 1; }
    int32_t n = 0;
    if (fread(&n, 4, 1, fp) != 1 || n < 1 || n > 100000) { fclose(fp); return 1; }
    g_gs = calloc((size_t)n, sizeof(gs_entry));
    for (int i = 0; i < n; i++) {
        int32_t len = 0, ncall = 0, nval = 0;
        if (fread(&len, 4, 1, fp) != 1 || len <= 0 || len > 4096) goto bad;
        g_gs[i].name = malloc((size_t)len + 1);
        if (fread(g_gs[i].name, 1, (size_t)len, fp) != (size_t)len) goto bad;
        g_gs[i].name[len] = 0;
        if (fread(&ncall, 4, 1, fp) != 1 || fread(&nval, 4, 1, fp) != 1 || nval < 1) goto bad;
        g_gs[i].vals = malloc((size_t)nval * 4);
        if (fread(g_gs[i].vals, 4, (size_t)nval, fp) != (size_t)nval) goto bad;
        if (ncall > 0) for (int j = 0; j < nval; j++) g_gs[i].vals[j] /= (float)ncall;
        g_gs[i].nval = nval;
        g_gs_n = i + 1;
    }
    fclose(fp);
    fprintf(stderr, "go-stats: %d entries from %s\n", g_gs_n, path);
    return 0;
bad:
    fclose(fp);
    fprintf(stderr, "go-stats: parse error in %s\n", path);
    return 1;
}

/* per-expert E[x²] slice for (layer, which∈{gate,up,down}, expert), or NULL */
static const float *gostats_find(int layer, const char *which, int e, int ncols) {
    if (!g_gs_n) return NULL;
    char nm[128];
    snprintf(nm, sizeof nm, "blk.%d.ffn_%s_exps.weight", layer, which);
    for (int i = 0; i < g_gs_n; i++) {
        if (strcmp(g_gs[i].name, nm) != 0) continue;
        if (g_gs[i].nval == ncols * NEXP) return g_gs[i].vals + (size_t)e * ncols;
        if (g_gs[i].nval == ncols) return g_gs[i].vals;
        return NULL;
    }
    return NULL;
}

/* go1b round-trip a weight matrix [rows×cols] → w_hat (the 1-bit reconstruction).
 * ew (may be NULL) = per-input-channel Go second moments → L_fix weighted scale. */
static void quant_dequant_ew(const float *w, float *wh, int rows, int cols,
                             unsigned char *scratch, const float *ew) {
    size_t rb = go1b_blk_row_bytes(cols);
    go1b_blk_quantize_imat(w, scratch, rows, cols, ew);
    for (int r = 0; r < rows; r++)
        go1b_blk_dequantize_row(scratch + (size_t)r * rb, wh + (size_t)r * cols, cols);
}

static void *worker(void *arg) {
    worker_ctx *c = (worker_ctx *)arg;
    c->rc = 1;
    hf_db *db = hf_open(c->hf_dir);
    if (!db) { fprintf(stderr, "[worker] hf_open failed\n"); return NULL; }

    /* reusable per-thread buffers (1-bit reconstructions + scratch + temps) */
    float *gh = malloc((size_t)DF * DM * sizeof(float));
    float *uh = malloc((size_t)DF * DM * sizeof(float));
    float *dh = malloc((size_t)DM * DF * sizeof(float));
    size_t scratch_n = (size_t)DF * go1b_blk_row_bytes(DM);
    size_t s2 = (size_t)DM * go1b_blk_row_bytes(DF);
    if (s2 > scratch_n) scratch_n = s2;
    unsigned char *scratch = malloc(scratch_n);
    float *oref_t = malloc((size_t)DM * sizeof(float));
    float *ohat_t = malloc((size_t)DM * sizeof(float));
    if (!gh || !uh || !dh || !scratch || !oref_t || !ohat_t) { fprintf(stderr, "[worker] OOM buffers\n"); goto done; }

    for (int e = c->e_begin; e < c->e_end; e++) {
        char nm[256];
        int64_t n;
        /* HF naming: w1=gate w3=up [DF×DM], w2=down [DM×DF]; FP8 dequant inside hf_read. */
        snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w1.weight", c->layer, e);
        float *gate = hf_read_f32(db, nm, &n);
        if (!gate || n != (int64_t)DF * DM) { fprintf(stderr, "[worker] read %s n=%lld\n", nm, (long long)(gate?n:-1)); free(gate); goto done; }
        snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w3.weight", c->layer, e);
        float *up = hf_read_f32(db, nm, &n);
        if (!up || n != (int64_t)DF * DM) { fprintf(stderr, "[worker] read up\n"); free(gate); free(up); goto done; }
        snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w2.weight", c->layer, e);
        float *down = hf_read_f32(db, nm, &n);
        if (!down || n != (int64_t)DM * DF) { fprintf(stderr, "[worker] read down\n"); free(gate); free(up); free(down); goto done; }

        /* strict-1-bit reconstruction of this expert's three matrices
         * (with the L_fix Go-weighted scale when --go-stats is loaded) */
        quant_dequant_ew(gate, gh, DF, DM, scratch, gostats_find(c->layer, "gate", e, DM));
        quant_dequant_ew(up,   uh, DF, DM, scratch, gostats_find(c->layer, "up",   e, DM));
        quant_dequant_ew(down, dh, DM, DF, scratch, gostats_find(c->layer, "down", e, DF));

        for (int i = 0; i < c->n_x; i++) {
            const float *x = c->x_f32 + (size_t)i * DM;
            expert_forward_f32(x, gate, up, down, oref_t, DM, DF);
            expert_forward_f32(x, gh,   uh,   dh,   ohat_t, DM, DF);
            size_t base = ((size_t)e * c->n_x + i) * DM;
            for (int j = 0; j < DM; j++) {
                double r = (double)oref_t[j], h = (double)ohat_t[j];
                c->o_ref[base + j] = r;
                c->o_hat[base + j] = h;
                c->delta[base + j] = r - h;
            }
        }
        free(gate); free(up); free(down);
        c->done_experts++;
    }
    c->rc = 0;
done:
    free(gh); free(uh); free(dh); free(scratch); free(oref_t); free(ohat_t);
    hf_close(db);
    return NULL;
}

/* baseline (no correction) metrics: rel-L2 = ‖Δo‖/‖o_ref‖, mean cosine(ô,o_ref). */
static void baseline_metrics(const double *o_ref, const double *o_hat, const double *delta,
                             int n_exp, int n_x, int d_model, double *rel_l2, double *cos_mean) {
    double num = 0, den = 0, csum = 0; long cnt = 0;
    for (long ei = 0; ei < (long)n_exp * n_x; ei++) {
        const double *r = o_ref + (size_t)ei * d_model;
        const double *h = o_hat + (size_t)ei * d_model;
        const double *d = delta + (size_t)ei * d_model;
        double dot = 0, nr = 0, nh = 0, dd = 0, rr = 0;
        for (int j = 0; j < d_model; j++) { dot += r[j]*h[j]; nr += r[j]*r[j]; nh += h[j]*h[j]; dd += d[j]*d[j]; rr += r[j]*r[j]; }
        num += dd; den += rr;
        if (nr > 0 && nh > 0) { csum += dot / (sqrt(nr) * sqrt(nh)); cnt++; }
    }
    *rel_l2 = den > 0 ? sqrt(num / den) : 0;
    *cos_mean = cnt ? csum / cnt : 0;
}

/* ====================== aggregate branch (--solver rrr) ======================
 * Fits the runtime corr semantics on per-TOKEN aggregates instead of the dense
 * per-expert tensor: targets T_t = routed_L(t) − ŷ_t where ŷ is the student
 * 1-bit routed aggregate rebuilt on the teacher's routing trajectory (ids from
 * route_L, gate weights recomputed exactly like the capture: sqrtsoftplus at
 * the selected ids, normalized over top-6, ×1.5 — hash layers use the same
 * weight formula, only their SELECTION differed, and selection is given).
 * No n_exp factor in memory: full 12288-token layers fit in ~1 GiB. */
#define RRR_TOPK 6
#define RRR_SWIGLU_LIM 10.0f   /* config swiglu_limit; runtime clamps g/min u/± */

typedef struct {
    const char *hf_dir; int layer;
    int e_begin, e_end;
    const float *Xsel;       /* [n_sel×DM] */
    const int   *pair_row;   /* CSR over experts: selected-row of each firing  */
    const float *pair_w;     /*                    gate weight of each firing  */
    const int   *pair_off;   /* [NEXP+1] offsets                               */
    float *acc;              /* [n_sel×DM] worker-local ŷ accumulator          */
    int n_sel, rc, done;
} rrr_ctx;

static void *rrr_yhat_worker(void *arg) {
    rrr_ctx *c = (rrr_ctx *)arg;
    c->rc = 1;
    hf_db *db = hf_open(c->hf_dir);
    if (!db) { fprintf(stderr, "[rrr] hf_open failed\n"); return NULL; }
    float *gh = malloc((size_t)DF * DM * sizeof(float));
    float *uh = malloc((size_t)DF * DM * sizeof(float));
    float *dh = malloc((size_t)DM * DF * sizeof(float));
    size_t scratch_n = (size_t)DF * go1b_blk_row_bytes(DM);
    size_t s2 = (size_t)DM * go1b_blk_row_bytes(DF);
    if (s2 > scratch_n) scratch_n = s2;
    unsigned char *scratch = malloc(scratch_n);
    float *tmp = malloc((size_t)DM * sizeof(float));
    if (!gh || !uh || !dh || !scratch || !tmp) { fprintf(stderr, "[rrr] OOM worker\n"); goto done; }

    for (int e = c->e_begin; e < c->e_end; e++) {
        int p0 = c->pair_off[e], p1 = c->pair_off[e + 1];
        if (p0 == p1) continue;
        char nm[256]; int64_t n;
        snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w1.weight", c->layer, e);
        float *gate = hf_read_f32(db, nm, &n);
        if (!gate || n != (int64_t)DF * DM) { free(gate); goto done; }
        snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w3.weight", c->layer, e);
        float *up = hf_read_f32(db, nm, &n);
        if (!up || n != (int64_t)DF * DM) { free(gate); free(up); goto done; }
        snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w2.weight", c->layer, e);
        float *down = hf_read_f32(db, nm, &n);
        if (!down || n != (int64_t)DM * DF) { free(gate); free(up); free(down); goto done; }
        quant_dequant_ew(gate, gh, DF, DM, scratch, gostats_find(c->layer, "gate", e, DM));
        quant_dequant_ew(up,   uh, DF, DM, scratch, gostats_find(c->layer, "up",   e, DM));
        quant_dequant_ew(down, dh, DM, DF, scratch, gostats_find(c->layer, "down", e, DF));
        free(gate); free(up); free(down);
        for (int pi = p0; pi < p1; pi++) {
            int s = c->pair_row[pi];
            float w = c->pair_w[pi];
            expert_forward_f32_lim(c->Xsel + (size_t)s * DM, gh, uh, dh, tmp, DM, DF, RRR_SWIGLU_LIM);
            float *a = c->acc + (size_t)s * DM;
            for (int j = 0; j < DM; j++) a[j] += w * tmp[j];
        }
        c->done++;
    }
    c->rc = 0;
done:
    free(gh); free(uh); free(dh); free(scratch); free(tmp);
    hf_close(db);
    return NULL;
}

static double rrr_sqrtsoftplus(double zv) {
    double az = zv < 0.0 ? -zv : zv;
    return sqrt((zv > 0.0 ? zv : 0.0) + log1p(exp(-az)));
}

/* z 落盘目录: --z-dump-dir(原 DS4_Z_DUMP_DIR env, 2026-08-31 禁 env 清退); NULL=不落盘 */
static const char *g_z_dump_dir = NULL;

static void rrr_dump_z(const z_layer *z, int L) {
    const char *zdir = g_z_dump_dir;
    if (!zdir || !zdir[0]) return;
    char zpath[1024], ztmp[1060];
    snprintf(zpath, sizeof zpath, "%s/z_L%d.bin", zdir, L);
    snprintf(ztmp, sizeof ztmp, "%s.tmp", zpath);
    FILE *zf = fopen(ztmp, "wb");
    if (!zf) {   /* observability rule: a failed dump must NEVER be silent */
        fprintf(stderr, "L%d: Z-DUMP-FAIL cannot open %s (dir missing?)\n", L, ztmp);
        return;
    }
    int32_t hdr[3] = { z->d_model, z->n_exp, z->d_l };
    fwrite(hdr, sizeof(int32_t), 3, zf);
    size_t nU=(size_t)z->d_model*z->d_l, nV=(size_t)z->d_l*z->d_model, nC=(size_t)z->n_exp*z->d_l;
    #define ZWF(a,n) do{ for(size_t _i=0;_i<(n);_i++){ float _v=(float)(a)[_i]; fwrite(&_v,4,1,zf);} }while(0)
    ZWF(z->U,nU); ZWF(z->V,nV); ZWF(z->C,nC); ZWF(z->b,(size_t)z->d_model); ZWF(z->beta,(size_t)z->n_exp); ZWF(z->delta,(size_t)z->n_exp);
    #undef ZWF
    int wok = (fclose(zf) == 0);
    if (!wok || rename(ztmp, zpath) != 0) {   /* atomic land: partial writes never visible */
        fprintf(stderr, "L%d: Z-DUMP-FAIL write/rename %s\n", L, zpath);
        remove(ztmp);
        return;
    }
    fprintf(stderr, "L%d: z dumped d_l=%d -> %s\n", L, z->d_l, zpath);
}

/* ---- two-stage split (dual-host balance through the shard wall) ----------
 * --dump-sel DIR : run the shard-dependent ŷ ASSEMBLY only, write the compact
 *                  per-layer arrays to DIR/sel_L{L}.bin (~550MB), skip solve.
 * --load-sel DIR : read those arrays and run the (shard-free) linear-algebra
 *                  solve on any host. File: 'RSEL' i32 magic, then i32
 *                  {L, ntr, nte, DM, TOPK}, then f32 Xsel/Yhat/Yref/route_sel. */
static int sel_write(const char *dir, int L, int ntr, int nte,
                     const float *Xsel, const float *Yhat, const float *Yref,
                     const float *route_sel) {
    char p[1024]; snprintf(p, sizeof p, "%s/sel_L%d.bin", dir, L);
    char tmp[1040]; snprintf(tmp, sizeof tmp, "%s.tmp", p);
    FILE *f = fopen(tmp, "wb");
    if (!f) return 1;
    int32_t hdr[6] = { 0x4C455352, L, ntr, nte, DM, RRR_TOPK };
    size_t n_sel = (size_t)(ntr + nte);
    int ok = fwrite(hdr, 4, 6, f) == 6 &&
             fwrite(Xsel, 4, n_sel * DM, f) == n_sel * DM &&
             fwrite(Yhat, 4, n_sel * DM, f) == n_sel * DM &&
             fwrite(Yref, 4, n_sel * DM, f) == n_sel * DM &&
             fwrite(route_sel, 4, n_sel * RRR_TOPK, f) == n_sel * RRR_TOPK;
    fclose(f);
    if (!ok) { remove(tmp); return 1; }
    if (rename(tmp, p) != 0) { remove(tmp); return 1; }   /* atomic land */
    fprintf(stderr, "L%d: sel dumped ntr=%d nte=%d -> %s\n", L, ntr, nte, p);
    return 0;
}

static int sel_read(const char *dir, int L, int *ntr, int *nte,
                    float **Xsel, float **Yhat, float **Yref, float **route_sel) {
    char p[1024]; snprintf(p, sizeof p, "%s/sel_L%d.bin", dir, L);
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "L%d: no %s\n", L, p); return 1; }
    int32_t hdr[6];
    if (fread(hdr, 4, 6, f) != 6 || hdr[0] != 0x4C455352 || hdr[1] != L ||
        hdr[4] != DM || hdr[5] != RRR_TOPK) { fclose(f); return 1; }
    *ntr = hdr[2]; *nte = hdr[3];
    size_t n_sel = (size_t)(*ntr + *nte);
    *Xsel = malloc(n_sel * DM * 4); *Yhat = malloc(n_sel * DM * 4);
    *Yref = malloc(n_sel * DM * 4); *route_sel = malloc(n_sel * RRR_TOPK * 4);
    int ok = *Xsel && *Yhat && *Yref && *route_sel &&
             fread(*Xsel, 4, n_sel * DM, f) == n_sel * DM &&
             fread(*Yhat, 4, n_sel * DM, f) == n_sel * DM &&
             fread(*Yref, 4, n_sel * DM, f) == n_sel * DM &&
             fread(*route_sel, 4, n_sel * RRR_TOPK, f) == n_sel * RRR_TOPK;
    fclose(f);
    if (!ok) { free(*Xsel); free(*Yhat); free(*Yref); free(*route_sel); return 1; }
    return 0;
}

/* shard-free solve from a sel dump (the M4 side of the pipeline) */
static int run_rrr_layer_from_sel(const char *sel_dir, int L, int chunk,
                                  int feat_yhat, rrr_params *rp) {
    int ntr = 0, nte = 0;
    float *Xsel = NULL, *Yhat = NULL, *Yref = NULL, *route_sel = NULL;
    if (sel_read(sel_dir, L, &ntr, &nte, &Xsel, &Yhat, &Yref, &route_sel)) return 1;
    int *itr = malloc((size_t)ntr * sizeof(int));
    int *ite = nte > 0 ? malloc((size_t)nte * sizeof(int)) : NULL;
    if (!itr || (nte > 0 && !ite)) { free(itr); free(ite); free(Xsel); free(Yhat); free(Yref); free(route_sel); return 1; }
    for (int i = 0; i < ntr; i++) itr[i] = i;
    for (int i = 0; i < nte; i++) ite[i] = ntr + i;
    rp->chunk = chunk;
    z_layer z; rrr_report rep;
    time_t t1 = time(NULL);
    const float *feat = feat_yhat ? Yhat : Xsel;
    int src = rrr_solve(feat, Yhat, Yref, route_sel, RRR_TOPK,
                        itr, ntr, ite, nte, DM, NEXP, rp, &z, &rep);
    free(itr); free(ite);
    if (!src) {
        printf("%-4d | %3d | %6.4f | TR base_cos %7.5f -> corr %7.5f (rel %6.4f -> %6.4f) | TE base_cos %7.5f -> corr %7.5f (rel %6.4f -> %6.4f) | solve %lds\n",
               L, rep.d_l, rep.energy_at_k,
               rep.base_cos_tr, rep.corr_cos_tr, rep.base_rel_tr, rep.corr_rel_tr,
               rep.base_cos_te, rep.corr_cos_te, rep.base_rel_te, rep.corr_rel_te,
               (long)(time(NULL) - t1));
        fflush(stdout);
        rrr_dump_z(&z, L);
        z_layer_free(&z);
    } else fprintf(stderr, "L%d: rrr_solve(from-sel) rc=%d\n", L, src);
    free(Xsel); free(Yhat); free(Yref); free(route_sel);
    return src;
}

