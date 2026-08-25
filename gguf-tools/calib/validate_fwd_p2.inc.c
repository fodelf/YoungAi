/* ============================================================================
 * GO-AWARE 1-bit scale test (coordinator follow-up).
 *
 * go1b's generic per-row scale is s_gen = mean(|w_i|) — weight-L2-optimal but
 * GO-BLIND. The Go-aware scale instead minimizes the Go-domain OUTPUT error of
 * the row over activation samples {a_t} (closed-form 1-D least squares, no
 * training): with b_i = sign(w_i),
 *     s_go_i = Σ_t (w_i·a_t)(b_i·a_t) / Σ_t (b_i·a_t)²
 * a_t = x (ffn_in) for gate(w1)/up(w3); a_t = h = SiLU(gate·x)⊙(up·x) (the
 * full-precision SwiGLU intermediate) for down(w2). ŵ_go_i = s_go_i·b_i.
 *
 * Structure: EXPERT-OUTER (each unique routed expert's per-row scale accumulates
 * over the tokens routed to it, per the formula's Σ_t). Threads own disjoint
 * expert ranges and accumulate into PER-THREAD token-aggregate arrays (race-free)
 * reduced after join.
 *
 * HONESTY CAVEAT (reported inline): at nx≈64 each expert is routed ~1-2 tokens,
 * so the per-expert scale is fit IN-SAMPLE on very few points — a 1-token expert
 * is interpolated exactly (gate/up rows reproduce that token → h exact → o exact
 * → cos 1.0), which inflates the headline. We therefore also report the average
 * samples/expert and split the per-expert go-aware cosine by 1-sample vs
 * ≥2-sample experts so the verdict is not fooled by interpolation.
 */
typedef struct {
    const char  *hf_dir; int layer, nx;
    int          n_train;   /* 0 => in-sample (fit & eval on all routed tokens);   */
                            /* >0 => fit s_go on s<n_train, eval on s>=n_train      */
    const float *x_all, *gate_w;
    int64_t      stride, n_tok;
    const int   *etok_s;    /* [256*nx] sample index of each (expert,occurrence) */
    const float *etok_w;    /* [256*nx] router weight w_e for that occurrence    */
    const int   *etok_n;    /* [256] occurrence count per expert                 */
    int          e0, e1;    /* expert range owned by this thread                 */
    /* per-thread token aggregates [nx][DM] (reduced after join) */
    double *yf, *yg, *yo;   /* full / generic-1bit / go-aware-1bit weighted sums */
    /* per-thread per-expert cosine accumulators */
    double sum_gen_pe, sum_go_pe; long pe_count;
    double sum_go_pe_1s;  long cnt_1s;     /* go cos on 1-sample experts        */
    double sum_go_pe_multi; long cnt_multi;/* go cos on >=2-sample experts      */
    long   experts_used, tokens_used;
    int    fail;
} gactx;

/* real and sign matvec in one pass: real[i]=Σ_c W[i,c]x[c], sgn[i]=Σ_c sign(W[i,c])x[c]. */
static void real_sign_matvec(const float *W, const float *x,
                             float *real, float *sgn, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        const float *wr = W + (size_t)i * cols;
        float r = 0.0f, g = 0.0f;
        for (int c = 0; c < cols; c++) {
            float xc = x[c];
            r += wr[c] * xc;
            g += (wr[c] >= 0.0f) ? xc : -xc;
        }
        real[i] = r; sgn[i] = g;
    }
}
static inline float silu_f(float z) { return z / (1.0f + expf(-z)); }

static void *go_worker(void *arg) {
    gactx *c = (gactx *)arg;
    hf_db *db = hf_open(c->hf_dir);
    if (!db) { c->fail = 1; return NULL; }
    const int nx = c->nx;

    float *gate = NULL, *up = NULL, *down = NULL;
    float *gh = (float *)malloc((size_t)DF * DM * sizeof(float)); /* recon: gen then go */
    float *uh = (float *)malloc((size_t)DF * DM * sizeof(float));
    float *dh = (float *)malloc((size_t)DM * DF * sizeof(float));
    size_t scrn = (size_t)DF * go1b_row_bytes(DM);
    size_t scr2 = (size_t)DM * go1b_row_bytes(DF); if (scr2 > scrn) scrn = scr2;
    unsigned char *scratch = (unsigned char *)malloc(scrn);
    float *g_real = malloc((size_t)DF*sizeof(float)), *g_sign = malloc((size_t)DF*sizeof(float));
    float *u_real = malloc((size_t)DF*sizeof(float)), *u_sign = malloc((size_t)DF*sizeof(float));
    float *h_real = malloc((size_t)DF*sizeof(float));
    float *d_real = malloc((size_t)DM*sizeof(float)), *d_sign = malloc((size_t)DM*sizeof(float));
    double *Ag = malloc((size_t)DF*sizeof(double)), *Bg = malloc((size_t)DF*sizeof(double));
    double *Au = malloc((size_t)DF*sizeof(double)), *Bu = malloc((size_t)DF*sizeof(double));
    double *Ad = malloc((size_t)DM*sizeof(double)), *Bd = malloc((size_t)DM*sizeof(double));
    float *sgen_g = malloc((size_t)DF*sizeof(float)), *sgen_u = malloc((size_t)DF*sizeof(float));
    float *sgen_d = malloc((size_t)DM*sizeof(float)); /* generic mean|w| row scales (fallback) */
    float *o_full = malloc((size_t)DM*sizeof(float));
    float *o_tmp  = malloc((size_t)DM*sizeof(float));
    float *ofull_store = malloc((size_t)nx*DM*sizeof(float)); /* o_full per token of expert */
    if (!gh||!uh||!dh||!scratch||!g_real||!g_sign||!u_real||!u_sign||!h_real||!d_real||!d_sign||
        !Ag||!Bg||!Au||!Bu||!Ad||!Bd||!sgen_g||!sgen_u||!sgen_d||!o_full||!o_tmp||!ofull_store) {
        c->fail = 1; goto done;
    }
    const int n_train = c->n_train;   /* 0 => in-sample */

    for (int e = c->e0; e < c->e1; e++) {
        int k = c->etok_n[e];
        if (k <= 0) continue;
        char n1[256], n2[256], n3[256];
        snprintf(n1, sizeof n1, "layers.%d.ffn.experts.%d.w1.weight", c->layer, e);
        snprintf(n3, sizeof n3, "layers.%d.ffn.experts.%d.w3.weight", c->layer, e);
        snprintf(n2, sizeof n2, "layers.%d.ffn.experts.%d.w2.weight", c->layer, e);
        int64_t ng=0,nu=0,nd=0;
        gate = hf_read_f32(db, n1, &ng); up = hf_read_f32(db, n3, &nu); down = hf_read_f32(db, n2, &nd);
        if (!gate||!up||!down||ng!=(int64_t)DF*DM||nu!=(int64_t)DF*DM||nd!=(int64_t)DM*DF) {
            free(gate);free(up);free(down); gate=up=down=NULL; c->fail=1; goto done;
        }
        /* generic 1-bit reconstruction (mean|w| scale) into gh/uh/dh; capture the
         * per-row generic scale = |first reconstructed entry| for the s_go fallback. */
        quant_dequant(gate, gh, DF, DM, scratch);
        quant_dequant(up,   uh, DF, DM, scratch);
        quant_dequant(down, dh, DM, DF, scratch);
        for (int i = 0; i < DF; i++) { sgen_g[i] = fabsf(gh[(size_t)i*DM]); sgen_u[i] = fabsf(uh[(size_t)i*DM]); }
        for (int j = 0; j < DM; j++) { sgen_d[j] = fabsf(dh[(size_t)j*DF]); }

        for (int i = 0; i < DF; i++) { Ag[i]=Bg[i]=Au[i]=Bu[i]=0.0; }
        for (int j = 0; j < DM; j++) { Ad[j]=Bd[j]=0.0; }

        /* SUBPASS A: per token compute full output o (ALL tokens → yf + store);
         * accumulate Go-aware stats on TRAIN tokens; generic eval on EVAL tokens. */
        for (int o = 0; o < k; o++) {
            int s = c->etok_s[(size_t)e*nx + o];
            float w = c->etok_w[(size_t)e*nx + o];
            const float *x_t = c->x_all + (size_t)((int64_t)s * c->stride) * DM;
            int is_train = (n_train <= 0) || (s <  n_train);
            int is_eval  = (n_train <= 0) || (s >= n_train);

            real_sign_matvec(gate, x_t, g_real, g_sign, DF, DM);
            real_sign_matvec(up,   x_t, u_real, u_sign, DF, DM);
            for (int i = 0; i < DF; i++) {
                if (is_train) {
                    Ag[i] += (double)g_real[i]*g_sign[i]; Bg[i] += (double)g_sign[i]*g_sign[i];
                    Au[i] += (double)u_real[i]*u_sign[i]; Bu[i] += (double)u_sign[i]*u_sign[i];
                }
                h_real[i] = silu_f(g_real[i]) * u_real[i];
            }
            real_sign_matvec(down, h_real, d_real, d_sign, DM, DF);
            for (int j = 0; j < DM; j++) {
                if (is_train) { Ad[j] += (double)d_real[j]*d_sign[j]; Bd[j] += (double)d_sign[j]*d_sign[j]; }
                o_full[j] = d_real[j];                /* o_full = down·h = d_real */
            }
            memcpy(ofull_store + (size_t)o*DM, o_full, (size_t)DM*sizeof(float));

            double *yf = c->yf + (size_t)s*DM;
            for (int j = 0; j < DM; j++) yf[j] += (double)w*o_full[j];   /* full aggregate, ALL tokens */

            if (is_eval) {
                expert_forward_f32(x_t, gh, uh, dh, o_tmp, DM, DF);   /* generic 1-bit ô */
                double cg, rg; cos_rel(o_tmp, o_full, DM, &cg, &rg);
                c->sum_gen_pe += cg;
                double *yg = c->yg + (size_t)s*DM;
                for (int j = 0; j < DM; j++) yg[j] += (double)w*o_tmp[j];
            }
        }

        /* Go-aware scales s_go = A/B from TRAIN stats (fallback mean|w| if a row
         * had no train signal), build ŵ_go into gh/uh/dh (overwriting generic). */
        for (int i = 0; i < DF; i++) {
            double sg = (Bg[i] > 1e-12) ? Ag[i]/Bg[i] : (double)sgen_g[i];
            double su = (Bu[i] > 1e-12) ? Au[i]/Bu[i] : (double)sgen_u[i];
            const float *gr = gate + (size_t)i*DM, *ur = up + (size_t)i*DM;
            float *ghr = gh + (size_t)i*DM, *uhr = uh + (size_t)i*DM;
            for (int cc = 0; cc < DM; cc++) {
                ghr[cc] = (float)sg * (gr[cc] >= 0.0f ? 1.0f : -1.0f);
                uhr[cc] = (float)su * (ur[cc] >= 0.0f ? 1.0f : -1.0f);
            }
        }
        for (int j = 0; j < DM; j++) {
            double sd = (Bd[j] > 1e-12) ? Ad[j]/Bd[j] : (double)sgen_d[j];
            const float *dr = down + (size_t)j*DF;
            float *dhr = dh + (size_t)j*DF;
            for (int cc = 0; cc < DF; cc++) dhr[cc] = (float)sd * (dr[cc] >= 0.0f ? 1.0f : -1.0f);
        }

        /* SUBPASS B: Go-aware output ô_go for ALL tokens (→ yo aggregate needed by
         * solve_denoise on train+test); per-expert go cosine on EVAL tokens only. */
        for (int o = 0; o < k; o++) {
            int s = c->etok_s[(size_t)e*nx + o];
            float w = c->etok_w[(size_t)e*nx + o];
            const float *x_t = c->x_all + (size_t)((int64_t)s * c->stride) * DM;
            const float *of  = ofull_store + (size_t)o*DM;
            int is_eval = (n_train <= 0) || (s >= n_train);
            expert_forward_f32(x_t, gh, uh, dh, o_tmp, DM, DF);   /* Go-aware 1-bit ô */
            double *yo = c->yo + (size_t)s*DM;
            for (int j = 0; j < DM; j++) yo[j] += (double)w*o_tmp[j];   /* go aggregate, ALL tokens */
            if (is_eval) {
                double cgo, rgo; cos_rel(o_tmp, of, DM, &cgo, &rgo);
                c->sum_go_pe += cgo; c->pe_count++;
                if (k == 1) { c->sum_go_pe_1s += cgo; c->cnt_1s++; }
                else        { c->sum_go_pe_multi += cgo; c->cnt_multi++; }
                c->tokens_used++;
            }
        }
        c->experts_used++;
        free(gate); free(up); free(down); gate=up=down=NULL;
    }
done:
    free(gate); free(up); free(down);
    free(gh); free(uh); free(dh); free(scratch);
    free(g_real); free(g_sign); free(u_real); free(u_sign); free(h_real);
    free(d_real); free(d_sign);
    free(Ag); free(Bg); free(Au); free(Bu); free(Ad); free(Bd);
    free(sgen_g); free(sgen_u); free(sgen_d);
    free(o_full); free(o_tmp); free(ofull_store);
    hf_close(db);
    return NULL;
}

static int go_aware_test(const char *hf_dir, int layer, int nx, int n_threads,
                         int heldout, const char *cap_dir,
                         char *summary_out, size_t summary_cap) {
    int rc = 1;
    if (summary_out && summary_cap) summary_out[0] = '\0';
    /* held-out split: fit s_go (and later the hidden-var map) on TRAIN tokens
     * [0,n_train), evaluate on TEST [n_train,nx). n_train=0 ⇒ in-sample. */
    int n_train = 0;
    if (heldout) { n_train = 3 * nx / 4; if (n_train < 1) n_train = 1; if (n_train > nx-1) n_train = nx-1; }
    printf("\n######################## %s layer %d (nx=%d, threads=%d%s) ########################\n",
           heldout ? "HELDOUT-STACK" : "GO-AWARE", layer, nx, n_threads,
           heldout ? ", train/test=3:1" : "");

    char path[1024]; npy_meta mx, mi;
    float *x_all = NULL, *ids_all = NULL, *gate_w = NULL;
    int *etok_s = NULL, *etok_n = NULL; float *etok_w = NULL;
    double *yf = NULL, *yg = NULL, *yo = NULL;

    snprintf(path, sizeof path, "%s/ffn_in_L%d.npy", cap_dir, layer);
    x_all = npy_read_f32(path, &mx);
    if (!x_all) { fprintf(stderr, "FATAL: cannot read %s\n", path); return 1; }
    snprintf(path, sizeof path, "%s/route_L%d.npy", cap_dir, layer);
    ids_all = npy_read_f32(path, &mi);
    if (!ids_all) { fprintf(stderr, "FATAL: cannot read %s\n", path); goto cleanup; }
    if (mx.ndim!=2 || mx.shape[1]!=DM || mi.ndim!=2 || mi.shape[1]!=TOPK || mi.shape[0]!=mx.shape[0]) {
        fprintf(stderr, "FATAL: cap shapes invalid\n"); goto cleanup;
    }
    int64_t n_tok = mx.shape[0];

    hf_db *db = hf_open(hf_dir);
    if (!db) { fprintf(stderr, "FATAL: hf_open failed\n"); goto cleanup; }
    char gate_name[256];
    snprintf(gate_name, sizeof gate_name, "layers.%d.ffn.gate.weight", layer);
    int64_t gate_n = 0; gate_w = hf_read_f32(db, gate_name, &gate_n);
    hf_close(db);
    if (!gate_w || gate_n != (int64_t)NEXP*DM) { fprintf(stderr, "FATAL: gate load\n"); goto cleanup; }

    int64_t stride = n_tok / nx; if (stride < 1) stride = 1;

    /* build per-expert occurrence lists (sample index + router weight) */
    etok_s = (int   *)malloc((size_t)NEXP*nx*sizeof(int));
    etok_w = (float *)malloc((size_t)NEXP*nx*sizeof(float));
    etok_n = (int   *)calloc((size_t)NEXP, sizeof(int));
    if (!etok_s || !etok_w || !etok_n) { fprintf(stderr, "FATAL: oom etok\n"); goto cleanup; }
    for (int s = 0; s < nx; s++) {
        int64_t t = (int64_t)s * stride; if (t >= n_tok) break;
        const float *x_t = x_all + (size_t)t * DM;
        const float *idf = ids_all + (size_t)t * TOPK;
        int ids[TOPK]; double score[TOPK], ssum = 0.0;
        for (int kk = 0; kk < TOPK; kk++) {
            int e = (int)lround((double)idf[kk]);
            if (e < 0 || e >= NEXP) { fprintf(stderr, "FATAL: bad id\n"); goto cleanup; }
            ids[kk] = e;
            score[kk] = sqrtsoftplus_d(dot_f32(gate_w + (size_t)e*DM, x_t, DM));
            ssum += score[kk];
        }
        if (!(ssum > 0.0)) ssum = 1.0;
        for (int kk = 0; kk < TOPK; kk++) {
            int e = ids[kk];
            int slot = etok_n[e];
            etok_s[(size_t)e*nx + slot] = s;
            etok_w[(size_t)e*nx + slot] = (float)(score[kk]/ssum * (double)ROUTED_SCALING);
            etok_n[e] = slot + 1;
        }
    }

    yf = (double *)calloc((size_t)nx*DM, sizeof(double));
    yg = (double *)calloc((size_t)nx*DM, sizeof(double));
    yo = (double *)calloc((size_t)nx*DM, sizeof(double));
    if (!yf || !yg || !yo) { fprintf(stderr, "FATAL: oom aggregates\n"); goto cleanup; }

    if (n_threads < 1) n_threads = 1;
    if (n_threads > MAX_THREADS) n_threads = MAX_THREADS;
    if (n_threads > NEXP) n_threads = NEXP;
    gactx     gc[MAX_THREADS]; pthread_t th[MAX_THREADS]; char spawned[MAX_THREADS];
    memset(gc, 0, sizeof gc); memset(spawned, 0, sizeof spawned);
    int per = (NEXP + n_threads - 1) / n_threads, nt = 0;
    for (int ti = 0; ti < n_threads; ti++) {
        int e0 = ti*per; if (e0 >= NEXP) break;
        int e1 = e0 + per; if (e1 > NEXP) e1 = NEXP;
        gactx *c = &gc[nt];
        c->hf_dir = hf_dir; c->layer = layer; c->nx = nx; c->n_train = n_train;
        c->x_all = x_all; c->gate_w = gate_w; c->stride = stride; c->n_tok = n_tok;
        c->etok_s = etok_s; c->etok_w = etok_w; c->etok_n = etok_n;
        c->e0 = e0; c->e1 = e1;
        /* per-thread aggregates */
        c->yf = (double *)calloc((size_t)nx*DM, sizeof(double));
        c->yg = (double *)calloc((size_t)nx*DM, sizeof(double));
        c->yo = (double *)calloc((size_t)nx*DM, sizeof(double));
        if (!c->yf || !c->yg || !c->yo) { fprintf(stderr, "FATAL: oom thread agg\n"); goto cleanup; }
        nt++;
    }
    for (int i = 0; i < nt; i++) {
        if (pthread_create(&th[i], NULL, go_worker, &gc[i]) == 0) spawned[i] = 1;
        else go_worker(&gc[i]);
    }
    for (int i = 0; i < nt; i++) if (spawned[i]) pthread_join(th[i], NULL);

    /* reduce */
    double sum_gen_pe=0, sum_go_pe=0, sum_go_1s=0, sum_go_multi=0;
    long pe_count=0, cnt_1s=0, cnt_multi=0, experts_used=0, tokens_used=0; int any_fail=0;
    for (int i = 0; i < nt; i++) {
        for (size_t q = 0; q < (size_t)nx*DM; q++) { yf[q]+=gc[i].yf[q]; yg[q]+=gc[i].yg[q]; yo[q]+=gc[i].yo[q]; }
        sum_gen_pe+=gc[i].sum_gen_pe; sum_go_pe+=gc[i].sum_go_pe; pe_count+=gc[i].pe_count;
        sum_go_1s+=gc[i].sum_go_pe_1s; cnt_1s+=gc[i].cnt_1s;
        sum_go_multi+=gc[i].sum_go_pe_multi; cnt_multi+=gc[i].cnt_multi;
        experts_used+=gc[i].experts_used; tokens_used+=gc[i].tokens_used; any_fail|=gc[i].fail;
        free(gc[i].yf); free(gc[i].yg); free(gc[i].yo);
    }
    if (any_fail) fprintf(stderr, "WARN: a go-worker reported failure\n");
    if (pe_count <= 0) { fprintf(stderr, "FATAL: no expert/token pairs\n"); goto cleanup; }

    double pe_gen = sum_gen_pe/pe_count, pe_go = sum_go_pe/pe_count;

    if (!heldout) {
        /* in-sample report (nx≈64 path): aggregate over ALL tokens + overfit split */
        double agg_gen=0, agg_go=0; int n_agg=0;
        for (int s = 0; s < nx; s++) {
            double c1,r1,c2,r2;
            cos_rel_dd(yg + (size_t)s*DM, yf + (size_t)s*DM, DM, &c1, &r1);
            cos_rel_dd(yo + (size_t)s*DM, yf + (size_t)s*DM, DM, &c2, &r2);
            agg_gen += c1; agg_go += c2; n_agg++;
        }
        double a_gen = agg_gen/n_agg, a_go = agg_go/n_agg;
        double go_1s  = cnt_1s   ? sum_go_1s/cnt_1s       : 0.0;
        double go_mlt = cnt_multi? sum_go_multi/cnt_multi : 0.0;
        char line[256];
        snprintf(line, sizeof line,
                 "L%d: per-expert cos gen=%.3f goaware=%.3f | aggregate cos gen=%.3f goaware=%.3f",
                 layer, pe_gen, pe_go, a_gen, a_go);
        printf("\n%s\n", line);
        printf("  diag: experts=%ld pairs=%ld samples/expert=%.2f | go per-expert split: "
               "1-sample=%.3f (n=%ld, IN-SAMPLE→trivially high) >=2-sample=%.3f (n=%ld)\n",
               experts_used, tokens_used, (double)tokens_used/experts_used,
               go_1s, cnt_1s, go_mlt, cnt_multi);
        if (summary_out && summary_cap) snprintf(summary_out, summary_cap, "%s", line);
        double honest_go = cnt_multi ? go_mlt : pe_go;
        if (honest_go > pe_gen + 0.10)
            printf("VERDICT: GO-AWARE HELPS (>=2-sample honest=%.3f vs generic %.3f)\n", honest_go, pe_gen);
        else if (honest_go > pe_gen + 0.03)
            printf("VERDICT: GO-AWARE marginal (honest %.3f vs generic %.3f; headline %.3f overfit)\n",
                   honest_go, pe_gen, pe_go);
        else
            printf("VERDICT: GO-AWARE does NOT beat generic (honest %.3f ~ generic %.3f)\n", honest_go, pe_gen);
        rc = 0;
        goto cleanup;
    }

    /* ---------- HELD-OUT STACK (nx large): the all-no-training ceiling ---------- */
    int n_test = nx - n_train;
    /* part (1): aggregate go-aware cosine on TEST tokens (cos ŷ_go vs y) */
    double agg_go_test = 0.0, agg_gen_test = 0.0;
    for (int s = n_train; s < nx; s++) {
        double c1,r1,c2,r2;
        cos_rel_dd(yo + (size_t)s*DM, yf + (size_t)s*DM, DM, &c1, &r1);  /* go-aware */
        cos_rel_dd(yg + (size_t)s*DM, yf + (size_t)s*DM, DM, &c2, &r2);  /* generic  */
        agg_go_test += c1; agg_gen_test += c2;
    }
    agg_go_test /= n_test; agg_gen_test /= n_test;

    /* part (2): closed-form rank-64 hidden-var correction ŷ_go→y, fit TRAIN, eval TEST */
    printf("  [held-out] per-expert gen=%.3f goaware=%.3f (TEST, %ld pairs); "
           "aggregate gen=%.3f goaware=%.3f (TEST)\n",
           pe_gen, pe_go, tokens_used, agg_gen_test, agg_go_test);
    printf("  fitting solve_denoise (rank=64, lambda=1.0) on the go-aware aggregate ŷ_go→y "
           "(n_train=%d, n_test=%d, d=%d) ...\n", n_train, n_test, DM); fflush(stdout);
    dn_result dn = solve_denoise(yf, yo, /*n_exp*/1, nx, DM, /*rank*/64, /*lambda*/1.0, n_train);
    double W = dn.ok ? dn.cos_mean : -1.0;

    char line[256];
    snprintf(line, sizeof line,
             "L%d HELDOUT: per-expert base gen=%.3f goaware=%.3f | aggregate goaware=%.3f | "
             "aggregate goaware+hiddenvar=%.3f", layer, pe_gen, pe_go, agg_go_test, W);
    printf("\n%s\n", line);
    if (!dn.ok) printf("  (solve_denoise ok=0 — numerical failure)\n");
    if (summary_out && summary_cap) snprintf(summary_out, summary_cap, "%s", line);

    double ceil_cos = (W > agg_go_test) ? W : agg_go_test;
    if (ceil_cos > 0.95)
        printf("VERDICT: no-training stack REACHES faithful-1bit territory (held-out cos=%.3f > 0.95)\n", ceil_cos);
    else
        printf("VERDICT: no-training stack CAPPED at held-out cos=%.3f (go-aware base %.3f, "
               "hidden-var %.3f) — still below the ~0.95 a faithful 1-bit needs\n",
               ceil_cos, agg_go_test, W);
    rc = 0;

cleanup:
    free(yf); free(yg); free(yo);
    free(etok_s); free(etok_w); free(etok_n);
    free(gate_w); free(x_all); free(ids_all);
    return rc;
}

/* ---- driver: parse args, open HF once, run each requested layer ---- */
int main(int argc, char **argv) {
    const char *layers_arg = "0";
    int nx = 48, n_threads = 6, do_correct = 1, goaware = 0, heldout = 0;
    const char *cap_dir = DEF_CAP, *hf_dir = DEF_HF;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers_arg = argv[++i];
        else if (!strcmp(argv[i], "--nx") && i + 1 < argc) nx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cap") && i + 1 < argc) cap_dir = argv[++i];
        else if (!strcmp(argv[i], "--hf") && i + 1 < argc) hf_dir = argv[++i];
        else if (!strcmp(argv[i], "--no-correct")) do_correct = 0;
        else if (!strcmp(argv[i], "--goaware")) goaware = 1;
        else if (!strcmp(argv[i], "--heldout")) { goaware = 1; heldout = 1; }
        else { fprintf(stderr, "unknown/incomplete arg: %s\n", argv[i]); return 2; }
    }
    if (nx < 1) nx = 1;

    printf("== validate_fwd: %s gate ==\n",
           heldout ? "HELD-OUT no-training stack (go-aware quant + closed-form hidden-var)" :
           goaware ? "GO-AWARE 1-bit scale" : "forward + 1-bit cancellation + aggregate-correction");
    printf("layers=%s  nx=%d  threads=%d  goaware=%d  heldout=%d  cap=%s\n  hf=%s\n",
           layers_arg, nx, n_threads, goaware, heldout, cap_dir, hf_dir);

    hf_db *db = NULL;
    if (!goaware) {
        db = hf_open(hf_dir);
        if (!db) { fprintf(stderr, "FATAL: hf_open(%s) failed\n", hf_dir); return 1; }
    }

    char buf[256];
    snprintf(buf, sizeof buf, "%s", layers_arg);
    char summary[MAX_LAYERS][256];
    int  nsum = 0, rc = 0;
    for (char *tok = strtok(buf, ","); tok && nsum < MAX_LAYERS; tok = strtok(NULL, ",")) {
        int L = atoi(tok);
        int r = goaware
              ? go_aware_test(hf_dir, L, nx, n_threads, heldout, cap_dir,
                              summary[nsum], sizeof summary[nsum])
              : process_layer(db, hf_dir, L, nx, n_threads, do_correct, cap_dir,
                              summary[nsum], sizeof summary[nsum]);
        if (r != 0) rc = 1;
        if (summary[nsum][0]) nsum++;
    }
    if (db) hf_close(db);

    printf("\n================ PER-LAYER SUMMARY ================\n");
    for (int i = 0; i < nsum; i++) printf("%s\n", summary[i]);
    return rc;
}
