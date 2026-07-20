/* layer_probe.c — per-expert SwiGLU FFN forward + 1-bit reconstruction probe.
 *
 * See layer_probe.h for the public contract and SPEC.md §5/§6 for why this
 * exists (measure the 1-bit residual Δo_e per expert so a per-layer low-rank
 * correction can be fit). DeepSeek-V4-Flash expert: d_model=4096, d_ff=2048,
 *     o = down · ( SiLU(gate·x) ⊙ (up·x) ).
 *
 * Decoupled from the 1-bit codec: the packed path goes through a caller-passed
 * dequant_row_fn callback (compatible with go1b_dequantize_row), never a direct
 * go1b_* symbol.  Pure C99 + pthreads.
 */
#include "layer_probe.h"

#include <math.h>      /* expf            */
#include <stdlib.h>    /* malloc/free     */
#include <string.h>    /* memset          */
#include <pthread.h>

/* SiLU(z) = z * sigmoid(z) = z / (1 + e^-z). The single non-linearity on the
 * gate branch of the SwiGLU; up branch is linear. */
static inline float silu(float z) {
    return z / (1.0f + expf(-z));
}

/* ---- full-precision reference: o_ref ------------------------------------ */
void expert_forward_f32(const float *x,
                        const float *gate, const float *up, const float *down,
                        float *o, int d_model, int d_ff)
{
    /* SwiGLU shape flow (all matmuls are matrix·vector, row-major):
     *   x : [d_model]
     *   g = gate·x : [d_ff]    gate[r,c] = gate[(size_t)r*d_model + c]
     *   u = up·x   : [d_ff]    up  [r,c] = up  [(size_t)r*d_model + c]
     *   h = SiLU(g) ⊙ u : [d_ff]        (fused below, never materialize g/u)
     *   o = down·h : [d_model] down[r,c] = down[(size_t)r*d_ff   + c]
     * h is the only intermediate we keep. Bounded (d_ff=2048 → 8 KiB), so a
     * per-call VLA on the worker's own stack — alloc-free and thread-safe (no
     * heap lock contention when fanned out over 256 experts). */
    float h[d_ff];

    for (int r = 0; r < d_ff; r++) {
        const float *grow = gate + (size_t)r * d_model;   /* gate row r */
        const float *urow = up   + (size_t)r * d_model;   /* up   row r */
        float g = 0.0f, u = 0.0f;
        for (int c = 0; c < d_model; c++) {
            g += grow[c] * x[c];
            u += urow[c] * x[c];
        }
        h[r] = silu(g) * u;
    }

    for (int r = 0; r < d_model; r++) {
        const float *drow = down + (size_t)r * d_ff;      /* down row r */
        float acc = 0.0f;
        for (int c = 0; c < d_ff; c++)
            acc += drow[c] * h[c];
        o[r] = acc;
    }
    /* NB: float accumulation here MUST match expert_forward_dequant so that the
     * residual Δo isolates weight quantization, not accumulation order. */
}

/* Same SwiGLU, with the RUNTIME's per-layer activation clamp
 * (moe.metal kernel_dsv4_moe_swiglu_weight / dsv4_fwd.py SWLIM):
 *   g = min(g, lim)          (one-sided)
 *   u = clamp(u, -lim, lim)  (two-sided)
 * lim <= 0 → identical to expert_forward_f32. Use this when the output is
 * compared against the teacher capture / consumed as the student ŷ, so the
 * target Δ isolates weight quantization, not clamp-semantics drift. */
void expert_forward_f32_lim(const float *x,
                            const float *gate, const float *up, const float *down,
                            float *o, int d_model, int d_ff, float lim)
{
    float h[d_ff];
    for (int r = 0; r < d_ff; r++) {
        const float *grow = gate + (size_t)r * d_model;
        const float *urow = up   + (size_t)r * d_model;
        float g = 0.0f, u = 0.0f;
        for (int c = 0; c < d_model; c++) {
            g += grow[c] * x[c];
            u += urow[c] * x[c];
        }
        if (lim > 0.0f) {
            if (g > lim) g = lim;
            if (u > lim) u = lim; else if (u < -lim) u = -lim;
        }
        h[r] = silu(g) * u;
    }
    for (int r = 0; r < d_model; r++) {
        const float *drow = down + (size_t)r * d_ff;
        float acc = 0.0f;
        for (int c = 0; c < d_ff; c++)
            acc += drow[c] * h[c];
        o[r] = acc;
    }
}

/* ---- 1-bit-consuming variant: ô ---------------------------------------- */
void expert_forward_dequant(const float *x,
                            const void *gate_q, const void *up_q, const void *down_q,
                            dequant_row_fn deq,
                            size_t row_bytes_gate, size_t row_bytes_up, size_t row_bytes_down,
                            float *o, int d_model, int d_ff)
{
    /* Same SwiGLU math as the f32 path; each weight row is decoded just-in-time
     * into `wrow` so we never hold a full dequantized matrix. `wrow` is sized to
     * the widest row decoded: gate/up rows are d_model wide, down rows d_ff. */
    const int rowmax = (d_model > d_ff) ? d_model : d_ff;
    float wrow[rowmax];   /* scratch for one decoded weight row */
    float h[d_ff];        /* SwiGLU intermediate, [d_ff] */

    const char *gq = (const char *)gate_q;
    const char *uq = (const char *)up_q;
    const char *dq = (const char *)down_q;

    for (int r = 0; r < d_ff; r++) {
        /* gate row r → g = Σ wrow[c]·x[c] */
        deq(gq + (size_t)r * row_bytes_gate, wrow, (int64_t)d_model);
        float g = 0.0f;
        for (int c = 0; c < d_model; c++) g += wrow[c] * x[c];
        /* up row r → u (reuse wrow) */
        deq(uq + (size_t)r * row_bytes_up, wrow, (int64_t)d_model);
        float u = 0.0f;
        for (int c = 0; c < d_model; c++) u += wrow[c] * x[c];
        h[r] = silu(g) * u;
    }

    for (int r = 0; r < d_model; r++) {
        deq(dq + (size_t)r * row_bytes_down, wrow, (int64_t)d_ff);
        float acc = 0.0f;
        for (int c = 0; c < d_ff; c++) acc += wrow[c] * h[c];
        o[r] = acc;
    }
}

/* ---- driver: probe every expert of one layer over an activation pool ---- */

/* Per-thread work slice: a contiguous expert range [e_start, e_end). Mirrors
 * the engine's expert_worker fan-out — each thread owns a disjoint expert band
 * and writes disjoint output regions, so no locking is needed. */
typedef struct {
    const float     *x_pool;
    int              n_x, layer, n_experts;
    int              d_model, d_ff;
    expert_loader_fn load;
    void            *load_user;
    dequant_row_fn   deq;
    int              e_start, e_end;
    float           *oref_out;     /* or NULL */
    float           *ohat_out;     /* or NULL */
    float           *residual_out; /* or NULL */
} probe_task;

static void *probe_worker(void *arg)
{
    probe_task *t = (probe_task *)arg;
    const size_t dm = (size_t)t->d_model;

    /* When the caller doesn't want a given output kept but we still need it to
     * form the residual, compute into a small per-thread scratch instead. */
    float *oref_scr = t->oref_out ? NULL : (float *)malloc(dm * sizeof(float));
    float *ohat_scr = t->ohat_out ? NULL : (float *)malloc(dm * sizeof(float));

    for (int e = t->e_start; e < t->e_end; e++) {
        expert_weights w;
        memset(&w, 0, sizeof w);
        /* TODO(integration): wire to GGUF expert tensors
         *   blk.%u.ffn_gate_exps.weight / ffn_up_exps.weight / ffn_down_exps.weight
         * (the loader hook resolves layer `t->layer`, expert `e` to mmap'd row
         * bases + per-row strides). REAL GGUF/HF loading is out of scope here. */
        if (t->load(t->load_user, t->layer, e, &w) != 0)
            continue;  /* loader asked to skip this expert */

        const int have_ref = (w.gate && w.up && w.down);
        const int have_hat = (w.gate_q && w.up_q && w.down_q && t->deq);

        for (int i = 0; i < t->n_x; i++) {
            const float *x  = t->x_pool + (size_t)i * dm;
            const size_t off = ((size_t)e * (size_t)t->n_x + (size_t)i) * dm;
            float *oref = t->oref_out ? t->oref_out + off : oref_scr;
            float *ohat = t->ohat_out ? t->ohat_out + off : ohat_scr;

            if (have_ref)
                expert_forward_f32(x, w.gate, w.up, w.down,
                                   oref, t->d_model, t->d_ff);
            if (have_hat)
                expert_forward_dequant(x, w.gate_q, w.up_q, w.down_q, t->deq,
                                       w.row_bytes_gate, w.row_bytes_up, w.row_bytes_down,
                                       ohat, t->d_model, t->d_ff);

            if (t->residual_out && have_ref && have_hat) {
                float *res = t->residual_out + off;
                for (size_t k = 0; k < dm; k++) res[k] = oref[k] - ohat[k];
            }
        }
    }

    free(oref_scr);
    free(ohat_scr);
    return NULL;
}

void probe_experts(const float *x_pool, int n_x,
                   int layer, int n_experts,
                   expert_loader_fn load, void *load_user,
                   dequant_row_fn deq,
                   int d_model, int d_ff,
                   int n_threads,
                   float *oref_out,
                   float *ohat_out,
                   float *residual_out)
{
    if (!x_pool || !load || n_experts <= 0 || n_x <= 0 || d_model <= 0 || d_ff <= 0)
        return;
    if (n_threads < 1)            n_threads = 1;
    if (n_threads > n_experts)    n_threads = n_experts;

    /* Even expert bands; the last band absorbs the remainder. */
    const int per = (n_experts + n_threads - 1) / n_threads;

    probe_task *tasks   = (probe_task *)calloc((size_t)n_threads, sizeof(probe_task));
    pthread_t  *tids    = (pthread_t  *)calloc((size_t)n_threads, sizeof(pthread_t));
    char       *spawned = (char       *)calloc((size_t)n_threads, sizeof(char));
    if (!tasks || !tids || !spawned) { free(tasks); free(tids); free(spawned); return; }

    int nt = 0;
    for (int ti = 0; ti < n_threads; ti++) {
        int e0 = ti * per;
        if (e0 >= n_experts) break;
        int e1 = e0 + per; if (e1 > n_experts) e1 = n_experts;
        probe_task *t = &tasks[nt];
        t->x_pool = x_pool;     t->n_x = n_x;       t->layer = layer;
        t->n_experts = n_experts;
        t->d_model = d_model;   t->d_ff = d_ff;
        t->load = load;         t->load_user = load_user;  t->deq = deq;
        t->e_start = e0;        t->e_end = e1;
        t->oref_out = oref_out; t->ohat_out = ohat_out;    t->residual_out = residual_out;
        nt++;
    }

    if (nt <= 1) {
        if (nt == 1) probe_worker(&tasks[0]);   /* serial: no thread overhead */
    } else {
        for (int i = 0; i < nt; i++) {
            if (pthread_create(&tids[i], NULL, probe_worker, &tasks[i]) == 0)
                spawned[i] = 1;
            else
                probe_worker(&tasks[i]);          /* run inline if spawn fails */
        }
        for (int i = 0; i < nt; i++)
            if (spawned[i]) pthread_join(tids[i], NULL);
    }

    free(tasks);
    free(tids);
    free(spawned);
}

/* ======================================================================= */
/* Self-test: cc -std=c99 -O2 -DPROBE_TEST layer_probe.c -o a3test -lpthread -lm */
#ifdef PROBE_TEST
#include <stdio.h>

static int g_fail = 0;

/* Compare two float arrays elementwise; report and tally on mismatch. */
static void check_close(const char *name, const float *got, const float *exp,
                        int n, float tol)
{
    float maxerr = 0.0f; int worst = -1;
    for (int i = 0; i < n; i++) {
        float e = fabsf(got[i] - exp[i]);
        if (e > maxerr) { maxerr = e; worst = i; }
    }
    if (maxerr <= tol) {
        printf("  PASS  %-34s max|err|=%.3e (tol %.1e)\n", name, maxerr, tol);
    } else {
        printf("  FAIL  %-34s max|err|=%.3e at [%d] got=%.7g exp=%.7g (tol %.1e)\n",
               name, maxerr, worst, got[worst], exp[worst], tol);
        g_fail++;
    }
}

/* Identity dequant: packed row IS a plain f32 row → copy ncols floats.
 * Lets us drive expert_forward_dequant with the f32 weights themselves so its
 * result must equal expert_forward_f32 (proves striding + math). */
static void deq_identity(const void *row, float *dst, int64_t ncols)
{
    memcpy(dst, row, (size_t)ncols * sizeof(float));
}

/* Sign dequant: 1-bit-ish reconstruction, weight -> ±SCALE. Produces a genuine
 * non-zero residual so the probe pipeline can be exercised end to end. */
#define SIGN_SCALE 0.5f
static void deq_sign(const void *row, float *dst, int64_t ncols)
{
    const float *w = (const float *)row;
    for (int64_t c = 0; c < ncols; c++)
        dst[c] = (w[c] >= 0.0f) ? SIGN_SCALE : -SIGN_SCALE;
}

/* Tiny loader: every (layer,expert) returns the same hand-built weights, in
 * both f32 form and "identity-packed" form (same bytes, stride = ncols*4). */
typedef struct {
    const float *gate, *up, *down;
    int d_model, d_ff;
} test_loader_ctx;

static int test_loader(void *user, int layer, int expert, expert_weights *out)
{
    test_loader_ctx *c = (test_loader_ctx *)user;
    (void)layer; (void)expert;
    out->gate = c->gate; out->up = c->up; out->down = c->down;
    out->gate_q = c->gate; out->up_q = c->up; out->down_q = c->down;
    out->row_bytes_gate = (size_t)c->d_model * sizeof(float);  /* gate row = d_model floats */
    out->row_bytes_up   = (size_t)c->d_model * sizeof(float);
    out->row_bytes_down = (size_t)c->d_ff    * sizeof(float);  /* down row = d_ff floats   */
    return 0;
}

int main(void)
{
    enum { DM = 8, DF = 4 };
    printf("layer_probe self-test (d_model=%d, d_ff=%d)\n", DM, DF);

    /* ---- Test 1: varied weights, fully hand-computed expected output ----
     * x = [1..8]. gate/up/down are sparse selectors so each output element is
     * a hand-traceable combination — this catches row/col transpose bugs.
     *   gate picks x[0],x[1],x[2],0  -> g = [1,2,3,0]
     *   up   picks x[7], Σx, 2x[0],0 -> u = [8,36,2,0]
     *   h[r] = SiLU(g[r])*u[r]:
     *     SiLU(1)=0.73105858, SiLU(2)=1.76159416, SiLU(3)=2.85772238, SiLU(0)=0
     *     h=[5.84846863, 63.41738961, 5.71544476, 0]
     *   down rows select h: see expected[] below. */
    float x1[DM] = {1,2,3,4,5,6,7,8};
    float gate1[DF*DM]; memset(gate1, 0, sizeof gate1);
    gate1[0*DM + 0] = 1.0f;   /* g[0]=x[0]=1 */
    gate1[1*DM + 1] = 1.0f;   /* g[1]=x[1]=2 */
    gate1[2*DM + 2] = 1.0f;   /* g[2]=x[2]=3 */
    /* row 3 all zero -> g[3]=0 */
    float up1[DF*DM]; memset(up1, 0, sizeof up1);
    up1[0*DM + 7] = 1.0f;                                  /* u[0]=x[7]=8 */
    for (int c = 0; c < DM; c++) up1[1*DM + c] = 1.0f;     /* u[1]=Σx=36  */
    up1[2*DM + 0] = 2.0f;                                  /* u[2]=2x[0]=2 */
    /* row 3 all zero -> u[3]=0 */
    float down1[DM*DF]; memset(down1, 0, sizeof down1);
    down1[0*DF + 0] = 1.0f;                                /* o[0]=h[0] */
    down1[1*DF + 1] = 1.0f;                                /* o[1]=h[1] */
    down1[2*DF + 2] = 1.0f;                                /* o[2]=h[2] */
    down1[3*DF + 3] = 1.0f;                                /* o[3]=h[3]=0 */
    for (int c = 0; c < DF; c++) down1[4*DF + c] = 1.0f;   /* o[4]=Σh */
    down1[5*DF + 0] = 1.0f; down1[5*DF + 2] = 1.0f;        /* o[5]=h[0]+h[2] */
    /* row 6 all zero -> o[6]=0 */
    down1[7*DF + 0] = 2.0f;                                /* o[7]=2h[0] */

    float expected1[DM] = {
        5.84846863f,    /* h0                   */
        63.41738961f,   /* h1                   */
        5.71544476f,    /* h2                   */
        0.0f,           /* h3                   */
        74.98130301f,   /* h0+h1+h2+h3          */
        11.56391339f,   /* h0+h2                */
        0.0f,           /* zero row             */
        11.69693726f    /* 2*h0                 */
    };

    float o1[DM];
    expert_forward_f32(x1, gate1, up1, down1, o1, DM, DF);
    check_close("f32 vs hand-computed", o1, expected1, DM, 1e-3f);

    /* ---- Test 2: constant weights, closed-form hand value ----
     * all gate=0.1, up=0.2, down=0.5, x=all 1 (Σx=8):
     *   g=0.8, u=1.6, SiLU(0.8)=0.55197958, h=0.88316734,
     *   o[r]=0.5*DF*h = 2*0.88316734 = 1.76633467 for every r. */
    float xc[DM]; for (int i = 0; i < DM; i++) xc[i] = 1.0f;
    float gatec[DF*DM], upc[DF*DM], downc[DM*DF];
    for (int i = 0; i < DF*DM; i++) { gatec[i] = 0.1f; upc[i] = 0.2f; }
    for (int i = 0; i < DM*DF; i++) downc[i] = 0.5f;
    float oc[DM], expc[DM];
    for (int i = 0; i < DM; i++) expc[i] = 1.76633467f;
    expert_forward_f32(xc, gatec, upc, downc, oc, DM, DF);
    check_close("f32 constant closed-form", oc, expc, DM, 1e-3f);

    /* ---- Test 3: dequant(identity) must equal f32 exactly (shapes+stride) -- */
    float od[DM];
    expert_forward_dequant(x1, gate1, up1, down1, deq_identity,
                           (size_t)DM*sizeof(float),   /* gate row stride */
                           (size_t)DM*sizeof(float),   /* up   row stride */
                           (size_t)DF*sizeof(float),   /* down row stride */
                           od, DM, DF);
    check_close("dequant(identity) vs f32", od, o1, DM, 1e-5f);

    /* ---- Test 4: probe_experts plumbing (serial + threaded) ----
     * Identity-packed loader => ô == o_ref => residual ≈ 0, and oref_out must
     * match a direct expert_forward_f32 call. */
    const int NE = 3, NX = 2;
    float x_pool[NX*DM];
    for (int i = 0; i < NX; i++)
        for (int c = 0; c < DM; c++) x_pool[i*DM + c] = (i == 0) ? x1[c] : xc[c];

    test_loader_ctx ctx = { gate1, up1, down1, DM, DF };
    float *oref = (float *)malloc((size_t)NE*NX*DM*sizeof(float));
    float *ohat = (float *)malloc((size_t)NE*NX*DM*sizeof(float));
    float *resid = (float *)malloc((size_t)NE*NX*DM*sizeof(float));

    for (int nthreads = 1; nthreads <= 2; nthreads++) {
        memset(oref, 0, (size_t)NE*NX*DM*sizeof(float));
        memset(ohat, 0, (size_t)NE*NX*DM*sizeof(float));
        memset(resid, 0, (size_t)NE*NX*DM*sizeof(float));
        probe_experts(x_pool, NX, /*layer*/0, NE,
                      test_loader, &ctx, deq_identity,
                      DM, DF, nthreads, oref, ohat, resid);

        /* residual ≈ 0 everywhere */
        float zero[NE*NX*DM]; memset(zero, 0, sizeof zero);
        char nm[64]; snprintf(nm, sizeof nm, "probe residual~0 (nthreads=%d)", nthreads);
        check_close(nm, resid, zero, NE*NX*DM, 1e-5f);

        /* oref_out for (expert0, x0) matches a direct forward */
        float direct[DM];
        expert_forward_f32(x1, gate1, up1, down1, direct, DM, DF);
        snprintf(nm, sizeof nm, "probe oref vs direct (nthreads=%d)", nthreads);
        check_close(nm, oref, direct, DM, 1e-5f);
    }

    /* ---- Test 5 (informational): sign-quant gives a finite, non-zero residual */
    memset(resid, 0, (size_t)NE*NX*DM*sizeof(float));
    probe_experts(x_pool, NX, 0, NE, test_loader, &ctx, deq_sign,
                  DM, DF, 2, NULL, NULL, resid);
    double rss = 0.0; int finite = 1;
    for (int i = 0; i < NE*NX*DM; i++) {
        if (!isfinite(resid[i])) finite = 0;
        rss += (double)resid[i] * (double)resid[i];
    }
    printf("  INFO  sign-quant residual L2=%.6g finite=%s nonzero=%s\n",
           sqrt(rss), finite ? "yes" : "no", (rss > 0.0) ? "yes" : "no");
    if (!finite || !(rss > 0.0)) { printf("  FAIL  sign-quant residual sanity\n"); g_fail++; }

    free(oref); free(ohat); free(resid);

    printf(g_fail ? "RESULT: %d FAILURE(S)\n" : "RESULT: ALL TESTS PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
#endif /* PROBE_TEST */
