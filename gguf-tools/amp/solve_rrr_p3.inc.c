/* ================================================================== */
/* -DRRR_TEST: synthetic recovery self-test                            */
/* ================================================================== */
#ifdef RRR_TEST
static double frand(uint64_t *s) {  /* uniform (-1,1) */
    return ((double)(sm64(s) >> 11) / 9007199254740992.0) * 2.0 - 1.0;
}
int main(void) {
    const int d = 96, ne = 24, topk = 4, ktrue = 6, kfit = 8;
    const int n = 3000, chunk = 100;
    uint64_t s = 42;

    double *Ut = malloc((size_t)d * ktrue * sizeof(double));
    double *Vt = malloc((size_t)ktrue * d * sizeof(double));
    double *Ct = malloc((size_t)ne * ktrue * sizeof(double));
    double *bt = malloc((size_t)d * sizeof(double));
    float *X = malloc((size_t)n * d * sizeof(float));
    float *Yhat = malloc((size_t)n * d * sizeof(float));
    float *Yref = malloc((size_t)n * d * sizeof(float));
    float *route = malloc((size_t)n * topk * sizeof(float));
    if (!Ut || !Vt || !Ct || !bt || !X || !Yhat || !Yref || !route) return 2;

    for (int i = 0; i < d * ktrue; i++) Ut[i] = frand(&s) * 0.8;
    for (int i = 0; i < ktrue * d; i++) Vt[i] = frand(&s) * 0.5;
    for (int e = 0; e < ne; e++)
        for (int kk = 0; kk < ktrue; kk++) Ct[(size_t)e * ktrue + kk] = 1.0 / topk + 0.3 * frand(&s);
    for (int j = 0; j < d; j++) bt[j] = 0.05 * frand(&s);

    for (int t = 0; t < n; t++) {
        float *x = X + (size_t)t * d;
        for (int j = 0; j < d; j++) x[j] = (float)frand(&s);
        /* topk distinct experts */
        float *rt = route + (size_t)t * topk;
        int used[64]; memset(used, 0, sizeof used);
        for (int e = 0; e < topk; e++) {
            int id;
            do { id = (int)(sm64(&s) % ne); } while (used[id]);
            used[id] = 1; rt[e] = (float)id;
        }
        double vt[16];
        for (int kk = 0; kk < ktrue; kk++) {
            double sv = 0; for (int j = 0; j < d; j++) sv += Vt[(size_t)kk * d + j] * x[j];
            vt[kk] = sv;
        }
        float *h = Yhat + (size_t)t * d, *r = Yref + (size_t)t * d;
        for (int j = 0; j < d; j++) {
            double base = 0.3 * frand(&s);          /* the "1-bit output" */
            double corr = bt[j] * topk;
            for (int kk = 0; kk < ktrue; kk++) {
                double sc = 0;
                for (int e = 0; e < topk; e++) sc += Ct[(size_t)(int)rt[e] * ktrue + kk];
                corr += Ut[(size_t)j * ktrue + kk] * sc * vt[kk];
            }
            double noise = 0.01 * frand(&s);
            h[j] = (float)base;
            r[j] = (float)(base + corr + noise);
        }
    }

    int ntr = 2400, nte = n - ntr;
    int *itr = malloc((size_t)ntr * sizeof(int));
    int *ite = malloc((size_t)nte * sizeof(int));
    for (int i = 0; i < ntr; i++) itr[i] = i;
    for (int i = 0; i < nte; i++) ite[i] = ntr + i;

    rrr_params p; rrr_params_default(&p);
    p.k = kfit; p.chunk = chunk; p.threads = 4; p.w_smooth = 0.5; p.procrustes = 1;
    z_layer z; rrr_report rep;
    int rc = rrr_solve(X, Yhat, Yref, route, topk, itr, ntr, ite, nte, d, ne, &p, &z, &rep);
    printf("rrr_test: rc=%d  energy@k=%.4f\n", rc, rep.energy_at_k);
    printf("  train: base_cos=%.4f corr_cos=%.4f  base_rel=%.4f corr_rel=%.4f\n",
           rep.base_cos_tr, rep.corr_cos_tr, rep.base_rel_tr, rep.corr_rel_tr);
    printf("  test : base_cos=%.4f corr_cos=%.4f  base_rel=%.4f corr_rel=%.4f\n",
           rep.base_cos_te, rep.corr_cos_te, rep.base_rel_te, rep.corr_rel_te);
    int ok = (rc == 0) && rep.corr_cos_te > 0.97 && rep.corr_rel_te < 0.25 &&
             rep.corr_cos_te > rep.base_cos_te + 0.2;
    /* b convention: stored b must be bfit/topk (kernel re-multiplies) */
    printf("  b[0]=%.5f  true b[0]=%.5f (stored should be ~true/topk=%.5f)\n",
           z.b[0], bt[0], bt[0] / topk);
    printf("%s\n", ok ? "RRR_TEST PASS" : "RRR_TEST FAIL");
    z_layer_free(&z);
    return ok ? 0 : 1;
}
#endif
