/* gate_route: raw=x@gate.T [Nt,NEXP]; scores=sqrt(log1p(exp(raw))); idx=tid2eid[ids] (hash);
 * w=take(scores,idx); w/=sum(w); w*=ROUTE_SCALE. tid2eid[vocab,NACT] int. 输出 idx[Nt,NACT], w[Nt,NACT]. */
void dq_gate_route_hash(const float *x, const float *gate, const int *tid2eid, const long *ids,
                        int *idx_out, float *w_out, int Nt, int DIM, int NEXP, int NACT,
                        float route_scale) {
    for (int t = 0; t < Nt; t++) {
        const float *xr = x + (size_t)t * DIM;
        const int *eids = tid2eid + (size_t)ids[t] * NACT;
        double wsum = 0.0; float ws[64];
        for (int a = 0; a < NACT; a++) {
            int e = eids[a];
            const float *gr = gate + (size_t)e * DIM;
            float raw = 0.0f; for (int k = 0; k < DIM; k++) raw += xr[k]*gr[k];
            float sc = sqrtf(log1pf(expf(raw)));   /* sqrt-softplus */
            ws[a] = sc; wsum += sc;
            idx_out[(size_t)t*NACT+a] = e;
        }
        float inv = (float)(route_scale / wsum);
        for (int a = 0; a < NACT; a++) w_out[(size_t)t*NACT+a] = ws[a] * inv;
    }
}

/* L3+ 路由(无 tid2eid): scores=sqrt-softplus(x@gate^T), 选 top-NACT of (scores+gbias),
 * 权重取无 bias 的 scores(归一化后 *route_scale). 与 numpy gate_route else 分支一致. */
void dq_gate_route_topk(const float *x, const float *gate, const float *gbias,
                        int *idx_out, float *w_out, int Nt, int DIM, int NEXP, int NACT,
                        float route_scale) {
    /* raw 分数=一个 [Nt,NEXP] GEMM, 走 dq_matmul(大尺寸上 GPU 显存暂存路)。32k 锚相位
     * 计时定罪: 原逐 token 点积单线程 18s/层(2026-08-26 用户令提速)。变换/top-k 原样。 */
    float *raw = malloc((size_t)Nt * NEXP * sizeof(float));
    dq_matmul(x, gate, raw, Nt, DIM, NEXP);
    float *sc = malloc((size_t)NEXP * sizeof(float));
    for (int t = 0; t < Nt; t++) {
        const float *rr = raw + (size_t)t * NEXP;
        for (int e = 0; e < NEXP; e++)
            sc[e] = sqrtf(log1pf(expf(rr[e])));    /* orig(无 bias) */
        int chosen[64]; double wsum = 0.0; float ws[64];
        for (int a = 0; a < NACT; a++) {           /* 选第 a 大的 (sc+gbias) */
            int best = -1; float bestv = -1e30f;
            for (int e = 0; e < NEXP; e++) {
                int used = 0; for (int b = 0; b < a; b++) if (chosen[b]==e){used=1;break;}
                if (used) continue;
                float v = sc[e] + (gbias ? gbias[e] : 0.0f);
                if (v > bestv){ bestv = v; best = e; }
            }
            chosen[a] = best; ws[a] = sc[best]; wsum += sc[best];
            idx_out[(size_t)t*NACT+a] = best;
        }
        float inv = (float)(route_scale / wsum);
        for (int a = 0; a < NACT; a++) w_out[(size_t)t*NACT+a] = ws[a] * inv;
    }
    free(sc); free(raw);
}

/* overlap_transform: t[sp,ratio,2d] → new[sp,2*ratio,d]. new[:,r:,:]=t[:,:,d:];
 * new[1:,:r,:]=t[:-1,:,:d]; 其余=val. (对应 numpy). */
void dq_overlap_transform(const float *t, float val, float *newt, int sp, int ratio, int d) {
    int r = ratio, twoR = 2*ratio, dd = 2*d;
    for (int i = 0; i < sp*twoR*d; i++) newt[i] = val;
    for (int s = 0; s < sp; s++)
        for (int a = 0; a < r; a++)
            for (int k = 0; k < d; k++)
                newt[((size_t)s*twoR + (r+a))*d + k] = t[((size_t)s*r + a)*dd + d + k];   /* [:,r:,:]=t[:,:,d:] */
    for (int s = 1; s < sp; s++)
        for (int a = 0; a < r; a++)
            for (int k = 0; k < d; k++)
                newt[((size_t)s*twoR + a)*d + k] = t[((size_t)(s-1)*r + a)*dd + k];        /* [1:,:r,:]=t[:-1,:,:d] */
}

/* compressor: x[S,DIM] → kvc[sp,coff*HD] (或 return 0 若 S<ratio). ratio=CR[L].
 * kv=x@cwkv.T; score=x@cwgate.T; reshape[sp,ratio,coff*HD]+cape; overlap(ratio==4)→2ratio;
 * softmax(axis=ratio)→加权和; rms(cnorm); rope last RD. 返回 sp(压缩后 token 数)。coff=1+overlap. */
int dq_compressor(const float *x, const float *cwkv, const float *cwgate, const float *cnorm,
                  const float *cape, const float *cos_t, const float *sin_t,
                  float *kvc_out, int S, int DIM, int HD, int RD, int ratio, float EPS) {
    if (S < ratio) return 0;
    int overlap = (ratio == 4), coff = 1 + overlap, wide = coff*HD;
    int remainder = S % ratio, cutoff = S - remainder, sp = cutoff / ratio;
    float *kv = (float*)malloc((size_t)cutoff*wide*sizeof(float));
    float *score = (float*)malloc((size_t)cutoff*wide*sizeof(float));
    dq_matmul(x, cwkv, kv, cutoff, DIM, wide);       /* only first cutoff rows需要, 但matmul算cutoff行 */
    dq_matmul(x, cwgate, score, cutoff, DIM, wide);
    /* reshape [sp,ratio,wide]; score += cape[ratio,wide] */
    for (int p = 0; p < sp; p++)
        for (int a = 0; a < ratio; a++)
            for (int w = 0; w < wide; w++)
                score[((size_t)p*ratio+a)*wide+w] += cape[(size_t)a*wide+w];
    int outr = overlap ? 2*ratio : ratio;
    float *kvw = kv, *scw = score;
    float *kvo = NULL, *sco = NULL;
    if (overlap) {
        kvo = (float*)malloc((size_t)sp*outr*HD*sizeof(float));
        sco = (float*)malloc((size_t)sp*outr*HD*sizeof(float));
        dq_overlap_transform(kv, 0.0f, kvo, sp, ratio, HD);
        dq_overlap_transform(score, -INFINITY, sco, sp, ratio, HD);
        kvw = kvo; scw = sco;
    }
    /* softmax over axis=outr (中间维), 逐 (p, HD-channel) 沿 outr; kv=(kv*sm).sum(outr) → [sp,HD] */
    /* 注意: overlap 后 wide=HD (2d→d); 非overlap wide=HD (coff=1). 结果每 p 一个 [HD]. */
    for (int p = 0; p < sp; p++) {
        for (int w = 0; w < HD; w++) {
            float col[512]; /* outr≤8 */
            for (int a = 0; a < outr; a++) col[a] = scw[((size_t)p*outr+a)*HD+w];
            float sm[512]; dq_softmax(col, sm, outr);
            float acc = 0.0f;
            for (int a = 0; a < outr; a++) acc += kvw[((size_t)p*outr+a)*HD+w]*sm[a];
            kvc_out[(size_t)p*HD+w] = acc;
        }
        /* rms(kvc[p], cnorm) 就地 */
        float tmp[512]; dq_rms(kvc_out+(size_t)p*HD, cnorm, tmp, HD, EPS);
        for (int w = 0; w < HD; w++) kvc_out[(size_t)p*HD+w] = tmp[w];
        /* rope last RD: fc[p] = cos_t/sin_t 采样 [:cutoff:ratio] 的第 p 个 = 行 p*ratio */
        dq_apply_rope(kvc_out+(size_t)p*HD + (HD-RD), cos_t+(size_t)(p*ratio)*(RD/2), sin_t+(size_t)(p*ratio)*(RD/2), RD, 0);
    }
    free(kv); free(score); if (kvo) free(kvo); if (sco) free(sco);
    return sp;
}

/* attention (MLA): x[S,DIM] → out[S,DIM]. kvc[Sc,HD] 由 compressor 传入(Sc=0 若无).
 * q=rms(x@wqa.T,qnorm)@wqb.T reshape[S,NH,HD], per-head rms, rope last RD;
 * kv=rms(x@wkv.T,kvnorm) rope last RD; kv_all=[kv;kvc][N,HD];
 * scores[S,NH,N]=q·kv_all*HD^-0.5; mask(sliding win + comp); softmax w/ sink;
 * o[S,NH,HD]=w·kv_all; rope inv last RD; o reshape[S,OG,NH*HD/OG]; o=einsum wo_a; @wo_b.T. */
/* mask+softmax 行并行: 各行独立, 逐位与串行版同式(每行内部顺序不变)。
 * 线程数取 DS4_THREADS(与专家循环同一预算), 只读一次。 */
double g_t_attn = 0;
typedef struct { float *SC; float sinkh; int S, N, WIN, ratio, s0, s1; } dq_smw;
static void *dq_sm_worker(void *arg) {
    dq_smw *w = (dq_smw *)arg;
    const int N = w->N, S = w->S, WIN = w->WIN, ratio = w->ratio;
    for (int s = w->s0; s < w->s1; s++) {
        float *scr = w->SC + (size_t)s * N;
        for (int n = 0; n < N; n++) {
            int ok;
            if (n < S) ok = (n <= s) && (n > s - WIN);        /* sliding window causal */
            else       ok = ((n - S) < (s + 1) / ratio);       /* comp_ok */
            if (!ok) scr[n] = -INFINITY;
        }
        float m = -INFINITY; for (int n = 0; n < N; n++) if (scr[n] > m) m = scr[n];
        double denom = exp((double)w->sinkh - m);
        for (int n = 0; n < N; n++) { if (scr[n] == -INFINITY) { scr[n] = 0; continue; }
                                      scr[n] = expf(scr[n] - m); denom += scr[n]; }
        float inv = (float)(1.0 / denom);
        for (int n = 0; n < N; n++) scr[n] *= inv;
    }
    return NULL;
}
static void dq_softmax_rows_par(float *SC, float sinkh, int S, int N, int WIN, int ratio) {
    static int NT = 0;   /* 线程数 = 在线核数, 不读 env(2026-08-22 铁律) */
    if (!NT) { long nc = sysconf(_SC_NPROCESSORS_ONLN); NT = (int)(nc < 1 ? 1 : (nc > 64 ? 64 : nc)); }
    if (NT == 1 || S < 64) { dq_smw w = {SC, sinkh, S, N, WIN, ratio, 0, S}; dq_sm_worker(&w); return; }
    pthread_t th[64]; dq_smw ws[64];
    int per = (S + NT - 1) / NT, nt = 0;
    for (int t = 0; t < NT; t++) {
        int a = t * per, b = a + per > S ? S : a + per;
        if (a >= b) break;
        ws[nt] = (dq_smw){SC, sinkh, S, N, WIN, ratio, a, b};
        if (pthread_create(&th[nt], NULL, dq_sm_worker, &ws[nt]) != 0) { dq_sm_worker(&ws[nt]); }
        else nt++;
    }
    for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
}

/* ---- s-切片并行(2026-08-26 用户令 GPU/并行化): 注意力里三段逐 token 标量循环
 * (q 逐头 rms+rope / kv rms+rope / 出口逆 rope)在 S=32768 时是单线程长尾;
 * 各 s 行完全独立, 按行切线程 = 数值逐位不变(与 dq_softmax_rows_par 同纪律)。 ---- */
typedef struct {
    float *q, *kva, *o; const float *qnorm2, *cos_t, *sin_t;
    int S, NH, HD, RD; float EPS;
} dq_arw;
static void dq_ar_qrms(dq_arw *w, int a, int b) {
    for (int s = a; s < b; s++) for (int h = 0; h < w->NH; h++) {
        float *qh = w->q + ((size_t)s * w->NH + h) * w->HD;
        double v = 0; for (int d = 0; d < w->HD; d++) v += (double)qh[d] * qh[d];
        float r = (float)(1.0 / sqrt(v / (double)w->HD + (double)w->EPS));
        for (int d = 0; d < w->HD; d++) qh[d] *= r;
        dq_apply_rope(qh + (w->HD - w->RD), w->cos_t + (size_t)s * (w->RD / 2),
                      w->sin_t + (size_t)s * (w->RD / 2), w->RD, 0);
    }
}
static void dq_ar_irope(dq_arw *w, int a, int b) {
    for (int s = a; s < b; s++) for (int h = 0; h < w->NH; h++)
        dq_apply_rope(w->o + ((size_t)s * w->NH + h) * w->HD + (w->HD - w->RD),
                      w->cos_t + (size_t)s * (w->RD / 2),
                      w->sin_t + (size_t)s * (w->RD / 2), w->RD, 1);
}
typedef struct { dq_arw *w; void (*fn)(dq_arw *, int, int); int a, b; } dq_arj;
static void *dq_ar_worker(void *arg) { dq_arj *j = (dq_arj *)arg; j->fn(j->w, j->a, j->b); return NULL; }
static void dq_ar_par(void (*fn)(dq_arw *, int, int), dq_arw *w, int S) {
    static int NT2 = 0;   /* 线程数 = 在线核数(dq_softmax_rows_par 同式, 不读 env) */
    if (!NT2) { long nc = sysconf(_SC_NPROCESSORS_ONLN); NT2 = (int)(nc < 1 ? 1 : (nc > 64 ? 64 : nc)); }
    int nt = NT2;
    if (nt <= 1 || S < 256) { fn(w, 0, S); return; }
    pthread_t th[64]; dq_arj js[64];
    int per = (S + nt - 1) / nt, n = 0;
    for (int t = 0; t < nt; t++) {
        int a = t * per, b = a + per > S ? S : a + per;
        if (a >= b) break;
        js[n] = (dq_arj){w, fn, a, b};
        if (pthread_create(&th[n], NULL, dq_ar_worker, &js[n]) != 0) fn(w, a, b);
        else n++;
    }
    for (int t = 0; t < n; t++) pthread_join(th[t], NULL);
}

/* ★q/o 跨调用缓冲池(2026-08-28 速度)★ 两块各 S·NH·HD·4 = S=8192 时 1024MB。原来每次
 * dq_attention 都新 malloc/free: 新页第一次被写(o 是 D2H 目的地)要逐页缺页+清零, 实测
 * 1GB D2H 只跑到 3.6 GB/s。跨层复用后页已在场, 拷贝走满带宽。仅在尺寸变化时重分配。
 * 数值零影响: 两块都是被完整写满后才读(q 由 gemm 写满, o 由 D2H/头循环写满)。
 * 全部调用点(ds4quant_layer.c / p7 / p2 自测)都是单线程串行, 不存在并发进入。 */
static float *ATT_Q=NULL,*ATT_O=NULL; static size_t ATT_N=0;
#ifdef DQ_BLAS
/* wo_a 的每组 gemm(见下方调用点注释): C[S,OLR] = o_g[S,GD](lda=NHHD) · wa_g[OLR,GD]^T */
static void *dq_wog_worker(void*a_){
    struct wog_s { const float*o,*wa; float*oo; int S,GD,OLR,NHHD,ldc; } *w=a_;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, w->S, w->OLR, w->GD,
                1.0f, w->o, w->NHHD, w->wa, w->GD, 0.0f, w->oo, w->ldc);
    return NULL;
}
#endif
static int ATT_PIN=0;   /* 1=cudaMallocHost 钉页(2026-08-30 BFLT: attn核桶里 q/o 各 67MB/调用往返) */
static void att_pool(size_t n){
    if(ATT_N>=n) return;   /* cap 语义: 够大就复用(粗筛512/复核行交替不毁建) */
#ifdef DS4QUANT_CUDA
    extern int vqg_host_alloc(void**,size_t); extern void vqg_host_free(void*);
    if(ATT_PIN){ vqg_host_free(ATT_Q); vqg_host_free(ATT_O); } else { free(ATT_Q); free(ATT_O); }
    ATT_Q=ATT_O=NULL;
    if(vqg_host_alloc((void**)&ATT_Q,n*sizeof(float))&&vqg_host_alloc((void**)&ATT_O,n*sizeof(float))) ATT_PIN=1;
    else { static int _w=0; if(!_w++) fprintf(stderr,"[att] ★钉页失败 → 普通 malloc★\n");
        if(ATT_Q){ vqg_host_free(ATT_Q); }
        ATT_Q=(float*)malloc(n*sizeof(float)); ATT_O=(float*)malloc(n*sizeof(float)); ATT_PIN=0; }
#else
    free(ATT_Q); free(ATT_O);
    ATT_Q=(float*)malloc(n*sizeof(float)); ATT_O=(float*)malloc(n*sizeof(float));
#endif
    ATT_N=n;
}
void dq_attention(const float *x, const float *wqa, const float *qnorm, const float *wqb,
                  const float *wkv, const float *kvnorm, const float *sink,
                  const float *wo_a, const float *wo_b, const float *kvc,
                  const float *cos_t, const float *sin_t,
                  float *out, int S, int DIM, int NH, int HD, int RD, int QLR, int OLR, int OG,
                  int WIN, int Sc, int ratio, float EPS) {
    int N = S + Sc;
    float *qr = (float*)malloc((size_t)S*QLR*sizeof(float));
    float *qra = (float*)malloc((size_t)S*QLR*sizeof(float));
    dq_matmul(x, wqa, qra, S, DIM, QLR);
    for (int s=0;s<S;s++) dq_rms(qra+(size_t)s*QLR, qnorm, qr+(size_t)s*QLR, QLR, EPS);
    att_pool((size_t)S*NH*HD);
    float *q = ATT_Q;
    dq_matmul(qr, wqb, q, S, QLR, NH*HD);
    /* per-head rms (mean over HD) + rope — s 切片并行(逐位同) */
    dq_arw arw = { q, NULL, NULL, NULL, cos_t, sin_t, S, NH, HD, RD, EPS };
    dq_ar_par(dq_ar_qrms, &arw, S);
    float *kv = (float*)malloc((size_t)S*HD*sizeof(float));
    { float *kvr=(float*)malloc((size_t)S*HD*sizeof(float));
      dq_matmul(x, wkv, kvr, S, DIM, HD);
      for(int s=0;s<S;s++){ dq_rms(kvr+(size_t)s*HD, kvnorm, kv+(size_t)s*HD, HD, EPS);
        dq_apply_rope(kv+(size_t)s*HD+(HD-RD), cos_t+(size_t)s*(RD/2), sin_t+(size_t)s*(RD/2), RD, 0); }
      free(kvr); }
    /* kv_all[N,HD] = [kv; kvc] */
    float *kva = (float*)malloc((size_t)N*HD*sizeof(float));
    memcpy(kva, kv, (size_t)S*HD*sizeof(float));
    if (Sc>0) memcpy(kva+(size_t)S*HD, kvc, (size_t)Sc*HD*sizeof(float));
    float scale = 1.0f/sqrtf((float)HD);
    float *o = ATT_O;
#ifdef DQ_BLAS
    /* per-head 两个 gemm: SC_h=Q_h·kva^T, O_h=P_h·kva. mask/softmax/sink 逻辑与标量路径逐字一致. */
#ifdef DS4QUANT_CUDA
    /* ★attention 整层驻留 GPU★ 成功则 o 已填好, 直接跳到 inverse rope。
     * 失败(显存不够/cuBLAS 出错)静默回落下方逐头路径, 数值语义相同。 */
    extern int vqg_attention(const float *q, const float *kva, const float *sink, float *o,
                             int S, int N, int NH, int HD, int WIN, int ratio, float scale);
    int _gpu_ok = vqg_attention(q, kva, sink, o, S, N, NH, HD, WIN, ratio, scale);
    if (!_gpu_ok) {
#endif
    float *SC = (float*)malloc((size_t)S*N*sizeof(float));
    for (int h=0;h<NH;h++) {
        dq_matmul_strided(q + (size_t)h*HD, NH*HD, kva, HD, SC, N, S, HD, N, scale);
        /* ★mask+softmax 按 s 切片并行(2026-08-22)★
         * 原来这段是整个 dq_attention 里唯一的 O(S·N·NH) 标量循环, 且完全单线程:
         * S=8192 时 = 8192×8192×64 ≈ 43 亿次迭代 + expf, 实测 ~60s/层, 而磁盘/GPU/其余
         * 19 个核全在等它(1/20 核跑满是实锤)。S=256 时只有 420 万次, 所以小尺寸看不出来。
         * 各 s 行之间完全独立(scr 是 SC 的行切片), 按行切给线程, 零额外内存、数值逐位不变。 */
        dq_softmax_rows_par(SC, sink[h], S, N, WIN, ratio);
        dq_matmul_nt_strided(SC, N, kva, HD, o + (size_t)h*HD, NH*HD, S, N, HD);
    }
    free(SC);
#ifdef DS4QUANT_CUDA
    }
#endif
    arw.o = o;
    dq_ar_par(dq_ar_irope, &arw, S);       /* inverse rope — s 切片并行(逐位同) */
#else
    float *scr = (float*)malloc((size_t)N*sizeof(float));
    for (int s=0;s<S;s++) for (int h=0;h<NH;h++) {
        const float *qh = q + ((size_t)s*NH+h)*HD;
        for (int n=0;n<N;n++) {
            int ok;
            if (n<S) ok = (n<=s) && (n> s-WIN);                 /* sliding window causal */
            else     ok = ((n-S) < (s+1)/ratio);                 /* comp_ok */
            if (!ok) { scr[n]=-INFINITY; continue; }
            const float *kn=kva+(size_t)n*HD; float a=0; for(int d=0;d<HD;d++) a+=qh[d]*kn[d];
            scr[n]=a*scale;
        }
        float m=-INFINITY; for(int n=0;n<N;n++) if(scr[n]>m) m=scr[n];
        double denom=exp((double)sink[h]-m);
        for(int n=0;n<N;n++){ if(scr[n]==-INFINITY){scr[n]=0;continue;} scr[n]=expf(scr[n]-m); denom+=scr[n]; }
        float inv=(float)(1.0/denom);
        float *oh=o+((size_t)s*NH+h)*HD;
        for(int d=0;d<HD;d++){ float a=0; for(int n=0;n<N;n++) a+=scr[n]*kva[(size_t)n*HD+d]; oh[d]=a*inv; }
        dq_apply_rope(oh+(HD-RD), cos_t+(size_t)s*(RD/2), sin_t+(size_t)s*(RD/2), RD, 1);  /* inverse */
    }
    free(scr);
#endif
    /* o[S,NH*HD] reshape [S,OG,NH*HD/OG]; woa=wo_a.reshape(OG,OLR,NH*HD/OG); oo[s,g,r]=Σ_d o[s,g,d]*woa[g,r,d] */
    int GD = (NH*HD)/OG;
    float *oo = (float*)malloc((size_t)S*OG*OLR*sizeof(float));
#ifdef DQ_BLAS
    /* ★OG 路 g 并行(2026-08-28)★ 本进程 OPENBLAS_NUM_THREADS=1(必须, 否则 bytes_moe 的
     * 20 个 worker 里再套 BLAS 线程=超订阅), 于是这 8 个 gemm 是单核串行磨, 实测 0.78s/层。
     * 试过换 dq_matmul_strided 走 cuBLAS —— 更慢(1.36s): 每个 g 都以 lda=NH*HD 跨步读遍
     * 整块 1GB 的 o, 8 次调用就是 8GB 的 HMM 读, GPU 这边一点也不划算。
     * 改为 8 条线程各跑一个原样的 cblas_sgemm: 各 g 只写 oo 的不相交列段 ⇒ 逐位不变。 */
    /* ★(g, 行块)二维切 20 路(2026-08-28)★ 这段是 550 GFLOP(OG·S·OLR·GD·2, S=8192/
     * OLR=1024/GD=4096)。只按 g 切最多 OG=8 路, 实测 0.70s=786 GFLOPS 已贴住 8 核峰值,
     * 剩下 12 个核全闲。gemm 的输出行彼此独立 ⇒ 再按行段切开, 每段仍是一句原样的
     * cblas_sgemm, 逐位不变。
     * ★GPU 两条路都试过, 都更慢★: 逐 g 调 cuBLAS 1.36s(每个 g 以 lda=NH*HD 跨步扫遍整块
     * 1GB 的 o, 8 次=8GB); 合成一次 SgemmStridedBatched 1.57s(只扫一遍, 但 o 在主机内存,
     * GPU 经 ATS 页粒度直访只有几 GB/s —— 本文件顶部 2026-08-22 那条注释记的就是这个坑)。
     * 要让 GPU 划算, 得先让 o 全程留在显存(irope 也得上 GPU), 那是另一件事。 */
    { typedef struct { const float*o,*wa; float*oo; int S,GD,OLR,NHHD,ldc; } wog_t;
      const int RB=3;                      /* 8 组 × 3 行块 = 24 任务, 铺满 20 核 */
      wog_t wg[OG*3]; pthread_t wt[OG*3]; int nt=0;
      for(int g=0;g<OG;g++) for(int b=0;b<RB;b++){
          int s0=(int)((long)S*b/RB), s1=(int)((long)S*(b+1)/RB);
          if(s1<=s0) continue;
          wg[nt++]=(wog_t){o+(size_t)s0*NH*HD+(size_t)g*GD, wo_a+(size_t)g*OLR*GD,
                           oo+(size_t)s0*OG*OLR+(size_t)g*OLR, s1-s0, GD, OLR, NH*HD, OG*OLR}; }
      for(int i=0;i<nt;i++) pthread_create(&wt[i],NULL,dq_wog_worker,&wg[i]);
      for(int i=0;i<nt;i++) pthread_join(wt[i],NULL); }
#else
    for(int s=0;s<S;s++) for(int g=0;g<OG;g++){
        const float *od = o + (size_t)s*NH*HD + (size_t)g*GD;
        for(int r=0;r<OLR;r++){ const float *wr=wo_a+((size_t)g*OLR+r)*GD; float a=0; for(int d=0;d<GD;d++) a+=od[d]*wr[d]; oo[((size_t)s*OG+g)*OLR+r]=a; }
    }
#endif
    dq_matmul(oo, wo_b, out, S, OG*OLR, DIM);   /* [S,OG*OLR]@wo_b[DIM,OG*OLR].T */
    free(qr);free(qra);free(kv);free(kva);free(oo);   /* q/o 属池, 不 free */
}

#ifdef DS4QUANT_SELFTEST
/* 自测: 已知输入打印输出, 与 numpy 参考对比 (脚本 selftest_fwd.py 生成参考并 diff). */
int main(void) {
    /* rms */
    float x4[4] = {1.0f, -2.0f, 3.0f, 0.5f}, w4[4] = {1.0f, 1.0f, 1.0f, 1.0f}, o4[4];
    dq_rms(x4, w4, o4, 4, 1e-6f);
    printf("RMS %.6f %.6f %.6f %.6f\n", o4[0], o4[1], o4[2], o4[3]);
    /* silu/sigmoid */
    printf("SILU %.6f %.6f\n", dq_silu(1.0f), dq_silu(-2.0f));
    printf("SIG %.6f %.6f\n", dq_sigmoid(0.5f), dq_sigmoid(-1.0f));
    /* softmax */
    float sz[3] = {1.0f, 2.0f, 3.0f}, so[3]; dq_softmax(sz, so, 3);
    printf("SOFTMAX %.6f %.6f %.6f\n", so[0], so[1], so[2]);
    /* freqs_cis: dim=8, seqlen=3, yarn */
    float cc[3 * 4], ss[3 * 4];
    dq_freqs_cis(8, 3, 4096.0, 10000.0, 40.0, 32.0, 1.0, cc, ss);
    printf("FREQS_COS"); for (int i = 0; i < 12; i++) printf(" %.6f", cc[i]); printf("\n");
    printf("FREQS_SIN"); for (int i = 0; i < 12; i++) printf(" %.6f", ss[i]); printf("\n");
    /* apply_rope: rd=4, token1 (非零角) cos/sin */
    float xp[4] = {1.0f, 0.0f, 0.5f, -0.5f};
    dq_apply_rope(xp, cc + 4, ss + 4, 4, 0);   /* token1 行 */
    printf("ROPE %.6f %.6f %.6f %.6f\n", xp[0], xp[1], xp[2], xp[3]);
    /* matmul: X[2,3] @ W[2,3].T → [2,2] */
    float X23[6] = {1,2,3, 4,5,6}, W23[6] = {1,0,1, 0,1,0}, MM[4];
    dq_matmul(X23, W23, MM, 2, 3, 2);
    printf("MATMUL %.6f %.6f %.6f %.6f\n", MM[0], MM[1], MM[2], MM[3]);
    /* expert_fp: S=1 DIM=2 MOEI=2, swlim=10 */
    float ex[2]={1,-1}, ew1[4]={1,0,0,1}, ew3[4]={0.5f,0.5f,1,0}, ew2[4]={1,1,0,1}, eacc[2]={0,0};
    dq_expert_fp(ex, ew1, ew3, ew2, NULL, eacc, 1, 2, 2, 10.0f);
    printf("EXPERT %.6f %.6f\n", eacc[0], eacc[1]);
    /* hc_sinkhorn: S=1 HCM=2 HCIT=3 */
    float mixes[8]={0.1f,0.2f, 0.3f,0.4f, 0.5f,0.6f,0.7f,0.8f}, hsc[3]={1,1,1}, hbase[8]={0,0,0,0,0,0,0,0};
    float pre2[2],post2[2],comb2[4];
    dq_hc_sinkhorn(mixes,hsc,hbase,pre2,post2,comb2,1,2,3,1e-6f);
    printf("HC_PRE %.6f %.6f\n", pre2[0],pre2[1]);
    printf("HC_POST %.6f %.6f\n", post2[0],post2[1]);
    printf("HC_COMB %.6f %.6f %.6f %.6f\n", comb2[0],comb2[1],comb2[2],comb2[3]);
    /* hc_pre: S=1 HCM=2 DIM=2 → HD=4, mixdim=2HCM+HCM²=8; fn[8,4] */
    float h_in[4]={0.5f,1.0f,-0.5f,0.2f};  /* h[1,2,2] flat */
    float fn8[32]; for(int i=0;i<32;i++) fn8[i]=0.01f*(i+1);
    float pre_sc[3]={1,1,1}, pre_base[8]={0,0,0,0,0,0,0,0};
    float yy[2],pp[2],ccb[4];
    dq_hc_pre(h_in,fn8,pre_sc,pre_base,yy,pp,ccb,1,2,2,8,3,1e-6f,1e-6f);
    printf("HC_PRE_Y %.6f %.6f\n", yy[0],yy[1]);
    /* hc_post: a[1,2], resid[1,2,2]=h_in, post=pp, comb=ccb */
    float aa[2]={2.0f,-1.0f}, opost[4];
    dq_hc_post(aa,h_in,pp,ccb,opost,1,2,2);
    printf("HC_POST_O %.6f %.6f %.6f %.6f\n", opost[0],opost[1],opost[2],opost[3]);
    /* gate_route hash: Nt=1 DIM=2 NEXP=3 NACT=2, tid2eid[token0]=[0,2] */
    float gx[2]={1.0f,0.5f}, gate3[6]={0.5f,0.5f, 1,0, 0,1}; int t2e[6]={0,2, 0,0, 0,0}; long gids[1]={0};
    int gidx[2]; float gw[2];
    dq_gate_route_hash(gx,gate3,t2e,gids,gidx,gw,1,2,3,2,1.5f);
    printf("GATE_IDX %d %d\n", gidx[0],gidx[1]);
    printf("GATE_W %.6f %.6f\n", gw[0],gw[1]);
    /* compressor: S=8 DIM=3 HD=2 RD=2 ratio=4(overlap). cwkv/cwgate[4,3], cape[4,4], cnorm[2] */
    float cx[24]; for(int i=0;i<24;i++) cx[i]=0.1f*(i+1)-1.0f;
    float cwkv[12],cwgate[12]; for(int i=0;i<12;i++){cwkv[i]=0.05f*(i+1);cwgate[i]=0.03f*(i+1)-0.1f;}
    float cnorm[2]={1.0f,1.0f}, cape[16]; for(int i=0;i<16;i++) cape[i]=0.01f*i;
    float ccos[16],csin[16]; dq_freqs_cis(2,8,4096.,10000.,40.,32.,1.,ccos,csin);
    float kvc[16];
    int sp=dq_compressor(cx,cwkv,cwgate,cnorm,cape,ccos,csin,kvc,8,3,2,2,4,1e-6f);
    printf("COMPRESSOR sp=%d kvc %.6f %.6f %.6f %.6f\n", sp, kvc[0],kvc[1],kvc[2],kvc[3]);
    /* attention: 读 /tmp/att_w.bin (numpy 存的相同权重) S=4 DIM=3 NH=2 HD=4 RD=2 QLR=3 OLR=3 OG=2 WIN=2 无comp */
    { int S=4,DIM=3,NH=2,HD=4,RD=2,QLR=3,OLR=3,OG=2,WIN=2;
      FILE *f=fopen("/tmp/att_w.bin","rb");
      if (f) {
        int nq=NH*HD*QLR, nwoa=OG*OLR*(NH*HD/OG), nwob=DIM*OG*OLR;
        float *buf=malloc((S*DIM+QLR*DIM+nq+HD*DIM+NH+nwoa+nwob)*sizeof(float));
        fread(buf,sizeof(float),(S*DIM+QLR*DIM+nq+HD*DIM+NH+nwoa+nwob),f); fclose(f);
        float *x=buf,*wqa=x+S*DIM,*wqb=wqa+QLR*DIM,*wkv=wqb+nq,*sink=wkv+HD*DIM,*woa=sink+NH,*wob=woa+nwoa;
        float qn[3]={1,1,1},kn[4]={1,1,1,1};
        float ac[4*1],as[4*1]; dq_freqs_cis(RD,S,4096.,10000.,40.,32.,1.,ac,as);
        float aout[4*3];
        dq_attention(x,wqa,qn,wqb,wkv,kn,sink,woa,wob,NULL,ac,as,aout,S,DIM,NH,HD,RD,QLR,OLR,OG,WIN,0,0,1e-6f);
        printf("ATTN %.5f %.5f %.5f %.5f %.5f %.5f\n",aout[0],aout[1],aout[2],aout[9],aout[10],aout[11]);
        free(buf);
      } else printf("ATTN (no /tmp/att_w.bin)\n");
    }
    return 0;
}
#endif
