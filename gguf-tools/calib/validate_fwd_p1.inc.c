/* validate_fwd.c — FORWARD-VALIDATION + 1-bit cancellation + aggregate-correction
 * gate for the Go-domain quantization pipeline.
 *
 * PURPOSE (CONTRIBUTING/SPEC §5/§6 sanity gate): before any "this approach is
 * dead" verdict, prove the expert-forward pipeline is numerically CORRECT against
 * a trusted gold reference, then measure how 1-bit (go1b Theta_fix) errors behave
 * at the routed-MoE AGGREGATE, and whether a per-layer corrector recovers it.
 *
 * cap_m1 holds, per transformer layer L:
 *     ffn_in_L<L>.npy  [12288, 4096] f16  — MoE block input x (post ffn_norm)
 *     route_L<L>.npy   [12288, 6]    i16  — the token's top-6 routed expert ids
 *     routed_L<L>.npy  [12288, 4096] f16  — gold routed output Σ_{e∈top6} w_e·o_e
 *
 * STAGE 1 (forward validation). Recompute the routed output from ORIGINAL HF
 * weights (FP8 experts auto-dequant'd by hf_read_f32, router gate = BF16
 * layers.<L>.ffn.gate.weight), using the GIVEN route ids (we do NOT reimplement
 * top-k selection — only the per-expert WEIGHTS), and compare to routed_L. Three
 * variants (a) gate-weighted norm_topk+scaling×1.5, (b) unweighted mean, (c)
 * gate-weighted no-scaling; cosine is scale-invariant so (a)/(c) share a
 * direction. scoring_func="sqrtsoftplus" → score = sqrt(softplus(<gate_row_e,x>)).
 * topk_method="noaux_tc": exp_probs_b corrects SELECTION not the gate weight, so
 * it is NOT added here. norm_topk_prob=1, routed_scaling_factor=1.5.
 *
 * STAGE 2 (1-bit cancellation). For each routed expert, round-trip its w1/w3/w2
 * through go1b (strict ±mean|w| 1-bit) → ô_e, and ask whether the per-expert
 * error (cos(ô_e,o_e)≈0.55, lossy) CANCELS in the weighted sum ŷ=Σ w_e·ô_e vs
 * y=Σ w_e·o_e.
 *
 * STAGE 3 (aggregate correction). The decisive cheap test: treat the whole-layer
 * aggregate as one "expert" (n_exp=1) over MANY tokens and fit a per-layer
 * corrector ŷ→y on a TRAIN split, scoring HELD-OUT TEST tokens:
 *   - solve_denoise: ONE shared reduced-rank linear map o_ref ≈ M·ô + b
 *   - solve_genrf  : ONE shared random-feature NONLINEAR generator o_ref ≈ M_φ(ô,x)
 * Only the held-out cosine says whether the aggregate hidden variable is viable.
 *
 * DeepSeek-V4-Flash expert geometry: d_model=4096, d_ff=2048, SwiGLU MoE,
 * o = down·(SiLU(gate·x) ⊙ (up·x)). HF expert tensors (row-major, FP8):
 *   w1=gate [d_ff×d_model]  w3=up [d_ff×d_model]  w2=down [d_model×d_ff].
 *
 * The per-token collection is parallelized over tokens (each thread owns a
 * disjoint token range → disjoint output rows, race-free; one hf_db per thread,
 * the calib_run.c pattern). Pure C11 + pthreads, no C++. Links:
 *   npy.c hf_read.c layer_probe.c onebit_quant.c solve_denoise.c solve_genrf.c
 *   linalg_small.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <pthread.h>

#include "npy.h"           /* npy_read_f32 / npy_meta                        */
#include "hf_read.h"       /* hf_open / hf_has / hf_read_f32 (FP8/BF16->f32)  */
#include "layer_probe.h"   /* expert_forward_f32                             */
#include "onebit_quant.h"  /* go1b_quantize / go1b_dequantize_row (Theta_fix) */
#include "solve_denoise.h" /* solve_denoise (linear ŷ→y reduced-rank map)     */
#include "solve_genrf.h"   /* solve_genrf  (nonlinear random-feature M_φ)     */

/* DeepSeek-V4-Flash routed-MoE geometry (config.json). */
#define DM    4096   /* hidden_size / d_model            */
#define DF    2048   /* moe_intermediate_size / d_ff     */
#define NEXP  256    /* n_routed_experts                 */
#define TOPK  6      /* num_experts_per_tok              */
#define MAX_THREADS 32
#define MAX_LAYERS  32
static const float ROUTED_SCALING = 1.5f; /* routed_scaling_factor */

static const char *DEF_CAP = NULL;   /* 必传: 旧默认是某台机的 /tmp 取料目录, 漏传=静默读错机器 */
static const char *DEF_HF  =
  "/private/tmp/claude-501/-Users-fodelf-git-ds4-main/"
  "24503593-c406-4203-a34b-b2d8ea433b47/scratchpad/hf_all";

/* softplus(z) = log(1+e^z), numerically stable: max(z,0)+log1p(e^-|z|). */
static double softplus_d(double z) {
    double az = z < 0.0 ? -z : z;
    return (z > 0.0 ? z : 0.0) + log1p(exp(-az));
}
/* sqrtsoftplus score used by the V4 router. */
static double sqrtsoftplus_d(double z) { return sqrt(softplus_d(z)); }

#include "calib_shared.inc.c"   /* baseline_metrics/quant_dequant: 判决原语单一实现 */

/* full-precision f64 dot of a float row with x. */
static double dot_f32(const float *a, const float *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)a[i] * (double)b[i];
    return s;
}

/* cosine(y,ref) and rel-L2 ||y-ref||/||ref|| — float operands. */
static void cos_rel(const float *y, const float *ref, int n,
                    double *cos_out, double *rel_out) {
    double dot = 0.0, ny = 0.0, nr = 0.0, dl = 0.0;
    for (int i = 0; i < n; i++) {
        double a = (double)y[i], b = (double)ref[i], d = a - b;
        dot += a * b; ny += a * a; nr += b * b; dl += d * d;
    }
    double den = sqrt(ny) * sqrt(nr);
    *cos_out = (den > 0.0) ? dot / den : 0.0;
    *rel_out = (nr > 0.0) ? sqrt(dl) / sqrt(nr) : (dl > 0.0 ? INFINITY : 0.0);
}
/* same, double operands (used for the aggregate vectors). */
static void cos_rel_dd(const double *y, const double *ref, int n,
                       double *cos_out, double *rel_out) {
    double dot = 0.0, ny = 0.0, nr = 0.0, dl = 0.0;
    for (int i = 0; i < n; i++) {
        double a = y[i], b = ref[i], d = a - b;
        dot += a * b; ny += a * a; nr += b * b; dl += d * d;
    }
    double den = sqrt(ny) * sqrt(nr);
    *cos_out = (den > 0.0) ? dot / den : 0.0;
    *rel_out = (nr > 0.0) ? sqrt(dl) / sqrt(nr) : (dl > 0.0 ? INFINITY : 0.0);
}

/* ---- token-parallel collection worker --------------------------------------
 * Each thread owns a disjoint sample-index range [s0,s1); sample s maps to the
 * evenly-strided capture token t = s*stride. It writes ONLY rows [s0,s1) of the
 * shared output arrays (race-free) and keeps thread-local scalar accumulators. */
typedef struct {
    const char  *hf_dir;
    int          layer;
    const float *x_all, *ids_all, *ref_all, *gate_w; /* shared read-only */
    int64_t      stride, n_tok;
    int          s0, s1;                              /* sample range     */
    /* outputs: [nx][DM] row-major double (variant-a aggregates + inputs) */
    double      *y_a;     /* o_ref_agg = Σ w_e·o_e   (f32 path)          */
    double      *yhat_a;  /* o_hat_agg = Σ w_e·ô_e   (1-bit path)        */
    double      *x_d;     /* layer input x as double                    */
    /* thread-local accumulators (reduced after join) */
    double sum_cos[3], sum_rel[3];          /* variant a/b/c vs gold      */
    double sum_pe_cos; long pe_count;       /* per-expert 1-bit cos       */
    double sum_agg_cos, sum_agg_rel;        /* ŷ vs y (f32 aggregate)     */
    double sum_vsgold_cos;                  /* ŷ vs gold routed_L         */
    int    n_used;
    int    fail;
} wctx;

static void *agg_worker(void *arg) {
    wctx *c = (wctx *)arg;
    hf_db *db = hf_open(c->hf_dir);
    if (!db) { c->fail = 1; return NULL; }

    float *o_all    = (float *)malloc((size_t)TOPK * DM * sizeof(float));
    float *ohat_all = (float *)malloc((size_t)TOPK * DM * sizeof(float));
    size_t scr_n = (size_t)DF * go1b_row_bytes(DM);
    size_t scr_2 = (size_t)DM * go1b_row_bytes(DF);
    if (scr_2 > scr_n) scr_n = scr_2;
    unsigned char *scratch = (unsigned char *)malloc(scr_n);
    float  *gh = (float *)malloc((size_t)DF * DM * sizeof(float));
    float  *uh = (float *)malloc((size_t)DF * DM * sizeof(float));
    float  *dh = (float *)malloc((size_t)DM * DF * sizeof(float));
    double *refd = (double *)malloc((size_t)DM * sizeof(double)); /* ref as f64 */
    double *ysc  = (double *)malloc((size_t)DM * sizeof(double)); /* variant scratch */
    if (!o_all || !ohat_all || !scratch || !gh || !uh || !dh || !refd || !ysc) {
        c->fail = 1; goto done;
    }

    for (int s = c->s0; s < c->s1; s++) {
        int64_t t = (int64_t)s * c->stride;
        if (t >= c->n_tok) break;
        const float *x_t   = c->x_all   + (size_t)t * DM;
        const float *ref_t = c->ref_all + (size_t)t * DM;
        const float *idf   = c->ids_all + (size_t)t * TOPK;

        int ids[TOPK]; double score[TOPK]; int bad = 0;
        for (int k = 0; k < TOPK; k++) {
            long id = lround((double)idf[k]);
            if (id < 0 || id >= NEXP) { bad = 1; break; }
            ids[k] = (int)id;
        }
        if (bad) { c->fail = 1; goto done; }

        for (int k = 0; k < TOPK; k++) {
            int e = ids[k];
            char n1[256], n2[256], n3[256];
            snprintf(n1, sizeof n1, "layers.%d.ffn.experts.%d.w1.weight", c->layer, e); /* gate */
            snprintf(n3, sizeof n3, "layers.%d.ffn.experts.%d.w3.weight", c->layer, e); /* up   */
            snprintf(n2, sizeof n2, "layers.%d.ffn.experts.%d.w2.weight", c->layer, e); /* down */
            int64_t ng = 0, nu = 0, nd = 0;
            float *gate = hf_read_f32(db, n1, &ng);
            float *up   = hf_read_f32(db, n3, &nu);
            float *down = hf_read_f32(db, n2, &nd);
            if (!gate || !up || !down ||
                ng != (int64_t)DF * DM || nu != (int64_t)DF * DM || nd != (int64_t)DM * DF) {
                free(gate); free(up); free(down); c->fail = 1; goto done;
            }
            float *o_e = o_all + (size_t)k * DM;
            expert_forward_f32(x_t, gate, up, down, o_e, DM, DF);
            quant_dequant(gate, gh, DF, DM, scratch);
            quant_dequant(up,   uh, DF, DM, scratch);
            quant_dequant(down, dh, DM, DF, scratch);
            free(gate); free(up); free(down);
            float *oh_e = ohat_all + (size_t)k * DM;
            expert_forward_f32(x_t, gh, uh, dh, oh_e, DM, DF);

            double cpe, rpe; cos_rel(oh_e, o_e, DM, &cpe, &rpe);
            c->sum_pe_cos += cpe; c->pe_count++;

            double z = dot_f32(c->gate_w + (size_t)e * DM, x_t, DM);
            score[k] = sqrtsoftplus_d(z);
        }

        double ssum = 0.0;
        for (int k = 0; k < TOPK; k++) ssum += score[k];
        if (!(ssum > 0.0)) ssum = 1.0;
        double w[3][TOPK];
        for (int k = 0; k < TOPK; k++) {
            double nm = score[k] / ssum;
            w[0][k] = nm * (double)ROUTED_SCALING; /* (a) */
            w[1][k] = 1.0 / (double)TOPK;          /* (b) */
            w[2][k] = nm;                          /* (c) */
        }

        for (int d = 0; d < DM; d++) refd[d] = (double)ref_t[d];

        double *ya = c->y_a    + (size_t)s * DM;
        double *yh = c->yhat_a + (size_t)s * DM;
        double *xd = c->x_d    + (size_t)s * DM;
        for (int d = 0; d < DM; d++) {
            double afp = 0.0, ahat = 0.0;
            for (int k = 0; k < TOPK; k++) {
                afp  += w[0][k] * (double)o_all[(size_t)k * DM + d];
                ahat += w[0][k] * (double)ohat_all[(size_t)k * DM + d];
            }
            ya[d] = afp; yh[d] = ahat; xd[d] = (double)x_t[d];
        }

        /* variant a/b/c vs gold (a reuses ya) */
        double cc, rr;
        cos_rel_dd(ya, refd, DM, &cc, &rr); c->sum_cos[0] += cc; c->sum_rel[0] += rr;
        for (int v = 1; v < 3; v++) {
            for (int d = 0; d < DM; d++) {
                double acc = 0.0;
                for (int k = 0; k < TOPK; k++) acc += w[v][k] * (double)o_all[(size_t)k * DM + d];
                ysc[d] = acc;
            }
            cos_rel_dd(ysc, refd, DM, &cc, &rr); c->sum_cos[v] += cc; c->sum_rel[v] += rr;
        }

        /* aggregate 1-bit cancellation */
        double ac, ar; cos_rel_dd(yh, ya,   DM, &ac, &ar);
        double vc, vr; cos_rel_dd(yh, refd, DM, &vc, &vr);
        c->sum_agg_cos += ac; c->sum_agg_rel += ar; c->sum_vsgold_cos += vc;
        c->n_used++;
    }
done:
    free(o_all); free(ohat_all); free(scratch); free(gh); free(uh); free(dh);
    free(refd); free(ysc);
    hf_close(db);
    return NULL;
}

/* Forward + 1-bit cancellation + aggregate-correction for ONE layer. `db` is the
 * caller's shared handle (used single-threaded for the router gate); the parallel
 * workers each open their own. The headline one-liner is copied into summary_out
 * (if non-NULL). Returns 0 iff tokens were scored. */
static int process_layer(hf_db *db, const char *hf_dir, int layer, int nx,
                         int n_threads, int do_correct, const char *cap_dir,
                         char *summary_out, size_t summary_cap) {
    int rc = 1;
    if (summary_out && summary_cap) summary_out[0] = '\0';
    printf("\n######################## layer %d (nx=%d, threads=%d) ########################\n",
           layer, nx, n_threads);

    /* ---- load cap_m1 gold tensors for this layer (whole arrays, f32) ---- */
    char path[1024];
    npy_meta mx, mi, mr;
    float *x_all = NULL, *ids_all = NULL, *ref_all = NULL, *gate_w = NULL;
    double *y_a = NULL, *yhat_a = NULL, *x_d = NULL;

    snprintf(path, sizeof path, "%s/ffn_in_L%d.npy", cap_dir, layer);
    x_all = npy_read_f32(path, &mx);
    if (!x_all) { fprintf(stderr, "FATAL: cannot read %s\n", path); return 1; }
    snprintf(path, sizeof path, "%s/route_L%d.npy", cap_dir, layer);
    ids_all = npy_read_f32(path, &mi);
    if (!ids_all) { fprintf(stderr, "FATAL: cannot read %s\n", path); goto cleanup; }
    snprintf(path, sizeof path, "%s/routed_L%d.npy", cap_dir, layer);
    ref_all = npy_read_f32(path, &mr);
    if (!ref_all) { fprintf(stderr, "FATAL: cannot read %s\n", path); goto cleanup; }

    if (mx.ndim != 2 || mx.shape[1] != DM || mi.ndim != 2 || mi.shape[1] != TOPK ||
        mr.ndim != 2 || mr.shape[1] != DM ||
        mi.shape[0] != mx.shape[0] || mr.shape[0] != mx.shape[0]) {
        fprintf(stderr, "FATAL: cap tensor shapes invalid/mismatched\n");
        goto cleanup;
    }
    int64_t n_tok = mx.shape[0];

    /* ---- router gate weight [NEXP, DM] (BF16->f32), via the shared db ---- */
    char gate_name[256];
    snprintf(gate_name, sizeof gate_name, "layers.%d.ffn.gate.weight", layer);
    int64_t gate_n = 0;
    gate_w = hf_read_f32(db, gate_name, &gate_n);
    if (!gate_w || gate_n != (int64_t)NEXP * DM) {
        fprintf(stderr, "FATAL: router gate %s missing/wrong size\n", gate_name);
        goto cleanup;
    }
    char bias_name[256];
    snprintf(bias_name, sizeof bias_name, "layers.%d.ffn.gate.bias", layer);
    int bias_present = hf_has(db, bias_name);
    printf("router gate tensor: \"%s\"  [%d, %d] (BF16->f32)\n", gate_name, NEXP, DM);
    printf("exp_probs_b bias  : \"%s\" %s (noaux_tc: selection-only, NOT applied to weights)\n",
           bias_name, bias_present ? "PRESENT" : "absent");
    printf("scoring=sqrtsoftplus  norm_topk_prob=1  routed_scaling_factor=%.3f\n", ROUTED_SCALING);

    int64_t stride = n_tok / nx; if (stride < 1) stride = 1;

    /* sample-0 transparency (single-threaded; scores need only gate_w + x) */
    {
        const float *idf = ids_all;       /* token 0 */
        const float *x0  = x_all;
        printf("  [sample token 0] ids=");
        for (int k = 0; k < TOPK; k++) {
            int e = (int)lround((double)idf[k]);
            double z = dot_f32(gate_w + (size_t)e * DM, x0, DM);
            printf("%d(%.3f)%s", e, sqrtsoftplus_d(z), k < TOPK-1 ? "," : "");
        }
        printf("   [id(score)]\n");
    }

    /* ---- aggregate output arrays [nx][DM] double ---- */
    y_a    = (double *)malloc((size_t)nx * DM * sizeof(double));
    yhat_a = (double *)malloc((size_t)nx * DM * sizeof(double));
    x_d    = (double *)malloc((size_t)nx * DM * sizeof(double));
    if (!y_a || !yhat_a || !x_d) { fprintf(stderr, "FATAL: oom aggregate arrays\n"); goto cleanup; }

    /* ---- parallel collection over tokens ---- */
    if (n_threads < 1) n_threads = 1;
    if (n_threads > MAX_THREADS) n_threads = MAX_THREADS;
    if (n_threads > nx) n_threads = nx;
    wctx     ctx[MAX_THREADS];
    pthread_t th[MAX_THREADS];
    char      spawned[MAX_THREADS];
    memset(ctx, 0, sizeof ctx);
    memset(spawned, 0, sizeof spawned);
    int per = (nx + n_threads - 1) / n_threads;
    int nt = 0;
    for (int ti = 0; ti < n_threads; ti++) {
        int s0 = ti * per; if (s0 >= nx) break;
        int s1 = s0 + per; if (s1 > nx) s1 = nx;
        wctx *c = &ctx[nt];
        c->hf_dir = hf_dir; c->layer = layer;
        c->x_all = x_all; c->ids_all = ids_all; c->ref_all = ref_all; c->gate_w = gate_w;
        c->stride = stride; c->n_tok = n_tok; c->s0 = s0; c->s1 = s1;
        c->y_a = y_a; c->yhat_a = yhat_a; c->x_d = x_d;
        nt++;
    }
    for (int i = 0; i < nt; i++) {
        if (pthread_create(&th[i], NULL, agg_worker, &ctx[i]) == 0) spawned[i] = 1;
        else agg_worker(&ctx[i]);    /* inline on spawn failure */
    }
    for (int i = 0; i < nt; i++) if (spawned[i]) pthread_join(th[i], NULL);

    /* ---- reduce ---- */
    double sum_cos[3] = {0,0,0}, sum_rel[3] = {0,0,0};
    double sum_pe_cos = 0.0; long pe_count = 0;
    double sum_agg_cos = 0.0, sum_agg_rel = 0.0, sum_vsgold_cos = 0.0;
    int n_used = 0, any_fail = 0;
    for (int i = 0; i < nt; i++) {
        for (int v = 0; v < 3; v++) { sum_cos[v] += ctx[i].sum_cos[v]; sum_rel[v] += ctx[i].sum_rel[v]; }
        sum_pe_cos += ctx[i].sum_pe_cos; pe_count += ctx[i].pe_count;
        sum_agg_cos += ctx[i].sum_agg_cos; sum_agg_rel += ctx[i].sum_agg_rel;
        sum_vsgold_cos += ctx[i].sum_vsgold_cos;
        n_used += ctx[i].n_used; any_fail |= ctx[i].fail;
    }
    if (any_fail) fprintf(stderr, "WARN: a worker reported a failure (partial: %d tokens)\n", n_used);
    if (n_used <= 0) { fprintf(stderr, "FATAL: 0 tokens scored\n"); goto cleanup; }

    /* Collection done: the f32 captures + router gate are no longer needed (the
     * aggregates + x_d carry everything forward). Free them now so the heavy
     * d×d denoise covariances don't stack on top — keeps the watchdog happy. */
    free(x_all);   x_all   = NULL;
    free(ids_all); ids_all = NULL;
    free(ref_all); ref_all = NULL;
    free(gate_w);  gate_w  = NULL;

    /* ---- STAGE 1 report: forward validation ---- */
    const char *vname[3] = {
        "(a) gate-weighted (norm_topk + scaling x1.5)",
        "(b) unweighted mean of 6 experts          ",
        "(c) gate-weighted (norm_topk, NO scaling)  "
    };
    printf("\n== forward validation over %d tokens (layer %d) ==\n", n_used, layer);
    printf("%-46s   mean_cos     mean_relL2\n", "variant");
    int best = 0; double best_cos = -2.0;
    for (int v = 0; v < 3; v++) {
        double mc = sum_cos[v] / n_used, mr2 = sum_rel[v] / n_used;
        printf("%-46s   %8.5f     %8.5f\n", vname[v], mc, mr2);
        if (mc > best_cos) { best_cos = mc; best = v; }
    }
    printf("best variant: %s  (mean_cos=%.5f) -> %s\n", vname[best], best_cos,
           best_cos > 0.95 ? "FORWARD VALIDATED" :
           best_cos < 0.70 ? "FORWARD SUSPECT" : "FORWARD INCONCLUSIVE");

    /* ---- STAGE 2 report: 1-bit cancellation ---- */
    double pe   = (pe_count > 0) ? sum_pe_cos / (double)pe_count : 0.0;
    double agg  = sum_agg_cos / n_used;
    double aggr = sum_agg_rel / n_used;
    double vg   = sum_vsgold_cos / n_used;
    printf("\n== 1-bit (go1b Theta_fix) cancellation, layer %d ==\n", layer);
    printf("L%d: per-expert cos=%.3f | AGGREGATE cos=%.3f relL2=%.3f | vs-gold cos=%.3f\n",
           layer, pe, agg, aggr, vg);

    /* ---- STAGE 3: aggregate correction (held-out) ---- */
    rc = 0;
    if (do_correct && nx >= 8) {
        int n_train = 3 * nx / 4;
        if (n_train < 1) n_train = 1;
        if (n_train > nx - 1) n_train = nx - 1;

        /* baseline (uncorrected) aggregate cosine over the SAME held-out TEST
         * tokens [n_train,nx) — apples-to-apples with the correctors' held-out. */
        double base_cos = 0.0; int n_test = nx - n_train;
        for (int s = n_train; s < nx; s++) {
            double c1, r1;
            cos_rel_dd(yhat_a + (size_t)s * DM, y_a + (size_t)s * DM, DM, &c1, &r1);
            base_cos += c1;
        }
        base_cos /= n_test;

        printf("\n== STAGE 3 aggregate correction (n_train=%d, n_test=%d, d=%d) ==\n",
               n_train, n_test, DM);
        printf("  fitting solve_denoise (rank=64, lambda=1.0) ...\n"); fflush(stdout);
        dn_result dn = solve_denoise(y_a, yhat_a, /*n_exp*/1, nx, DM,
                                     /*rank*/64, /*lambda*/1.0, n_train);
        printf("  fitting solve_genrf  (n_feat=1024, lambda=1.0, seed=1234) ...\n"); fflush(stdout);
        gr_result gr = solve_genrf(y_a, yhat_a, x_d, /*n_exp*/1, nx, DM,
                                   /*n_feat*/1024, /*lambda*/1.0, n_train, 1234ULL);

        double D  = dn.ok ? dn.cos_mean : -1.0, Dr = dn.ok ? dn.rel_l2 : -1.0;
        double G  = gr.ok ? gr.cos_mean : -1.0, Gr = gr.ok ? gr.rel_l2 : -1.0;
        char line[256];
        snprintf(line, sizeof line,
                 "L%d AGG-CORRECT: baseline cos=%.3f | denoise held-out cos=%.3f rel=%.3f | "
                 "genrf held-out cos=%.3f rel=%.3f", layer, base_cos, D, Dr, G, Gr);
        printf("\n%s\n", line);
        if (!dn.ok) printf("  (denoise ok=0 — numerical failure / bad args)\n");
        if (!gr.ok) printf("  (genrf  ok=0 — numerical failure / bad args)\n");
        if (summary_out && summary_cap) snprintf(summary_out, summary_cap, "%s", line);

        double bestc = (D > G) ? D : G;
        if (bestc > 0.95)
            printf("VERDICT: AGGREGATE CORRECTION WORKS — hidden variable viable at aggregate level "
                   "(best held-out cos=%.3f vs baseline %.3f)\n", bestc, base_cos);
        else if (bestc > base_cos + 0.05)
            printf("VERDICT: aggregate PARTIALLY correctable — lift to %.3f from baseline %.3f, "
                   "still < 0.95 (not viable as-is)\n", bestc, base_cos);
        else
            printf("VERDICT: aggregate also not correctable (held-out best cos=%.3f ~ baseline %.3f)\n",
                   bestc, base_cos);
    } else {
        char line[256];
        snprintf(line, sizeof line,
                 "L%d: per-expert cos=%.3f | AGGREGATE cos=%.3f relL2=%.3f | vs-gold cos=%.3f",
                 layer, pe, agg, aggr, vg);
        if (summary_out && summary_cap) snprintf(summary_out, summary_cap, "%s", line);
        if (do_correct) printf("(correction skipped: nx=%d < 8)\n", nx);
    }

cleanup:
    free(y_a); free(yhat_a); free(x_d);
    free(gate_w); free(x_all); free(ids_all); free(ref_all);
    return rc;
}

