/* ds4quant_fwd.c — DeepSeek V4 前向 (C 移植, 自 dsv4_fwd.py 300 行 numpy)。
 * 目的: 让动态量化方案(逐层动态 z + 4损失 + 向前向后 + 感知)全 C 快跑, 逐层量化测试。
 * 移植纪律: 每模块和 numpy 对拍 (bit-close) 才继续; 用 st_db 读 HF (deepseek4-quantize 里)。
 *
 * 增量 1 (本文件当前): 数值原语 rms/silu/sigmoid/softmax/freqs_cis(yarn rope)/apply_rope/sinkhorn
 *   + --selftest 打印已知输入的输出, 与 numpy 参考逐值对比。
 * 待续: attention(MLA/compressor) / hc_pre_post / moe(gate/expert) / 动态量化 / 逐层测试。
 *
 * 编译(自测): cc -O3 -DDS4QUANT_SELFTEST -lm ds4quant_fwd.c -o ds4quant_selftest
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* Apple Accelerate (AMX) sgemm: 大 matmul 走 BLAS, 标量路径留作参考/非苹果平台.
 * 累加同为 fp32, 与标量只差求和顺序(~1e-6 rel); anchor/quant 两遍同 kernel, 对比自洽. */
#if defined(__APPLE__) && !defined(DS4QUANT_NO_BLAS)
#define DQ_BLAS 1
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK   /* 新 CBLAS 头(避免 macOS13.3+ 弃用警告); 不开 ILP64, int 仍 32 位 */
#endif
#include <Accelerate/Accelerate.h>
#elif defined(DS4QUANT_OPENBLAS)
/* Linux/Spark(GB10 aarch64) 移植(2026-08-17): cblas 接口同名同义, 编译加
 * -DDS4QUANT_OPENBLAS -lopenblas。标量路径仍是无 BLAS 时的参考。 */
#define DQ_BLAS 1
#include <cblas.h>
#endif

/* ---- 数值原语 (逐一对应 dsv4_fwd.py) ---- */

/* rms(x,w): x*rsqrt(mean(x²)+eps)*w. x[n], w[n] (per-row, 这里单行). */
void dq_rms(const float *x, const float *w, float *out, int n, float eps) {
    double v = 0.0;
    for (int i = 0; i < n; i++) v += (double)x[i] * (double)x[i];
    v = v / (double)n;
    float r = (float)(1.0 / sqrt(v + (double)eps));
    for (int i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

static inline float dq_silu(float z) { return z / (1.0f + expf(-z)); }
static inline float dq_sigmoid(float z) { return 1.0f / (1.0f + expf(-z)); }

/* softmax(z, axis=last) over n, 数值稳定 (减 max). */
void dq_softmax(const float *z, float *out, int n) {
    float m = z[0];
    for (int i = 1; i < n; i++) if (z[i] > m) m = z[i];
    double s = 0.0;
    for (int i = 0; i < n; i++) { out[i] = expf(z[i] - m); s += out[i]; }
    float inv = (float)(1.0 / s);
    for (int i = 0; i < n; i++) out[i] *= inv;
}

/* freqs_cis(dim, seqlen, orig, base, factor, bfast, bslow) → cos/sin 表 [seqlen, dim/2].
 * yarn NTK-by-parts (对应 numpy). cos_out/sin_out 预分配 [seqlen*(dim/2)]. */
static double dq_cdim(double nr, int dim, double orig, double base) {
    return (double)dim * log(orig / (nr * 2.0 * M_PI)) / (2.0 * log(base));
}
void dq_freqs_cis(int dim, int seqlen, double orig, double base, double factor,
                  double bfast, double bslow, float *cos_out, float *sin_out) {
    int half = dim / 2;
    double *fr = (double *)malloc((size_t)half * sizeof(double));
    for (int i = 0; i < half; i++) fr[i] = 1.0 / pow(base, (double)(2 * i) / (double)dim);
    if (orig > 0) {
        int lo = (int)floor(dq_cdim(bfast, dim, orig, base));
        int hi = (int)ceil(dq_cdim(bslow, dim, orig, base));
        if (lo < 0) lo = 0; if (hi > dim - 1) hi = dim - 1;
        if (lo == hi) hi = lo; /* numpy: hi+=0.001, 整数化后相同 → 用浮点 hi */
        double hif = (lo == (int)ceil(dq_cdim(bslow, dim, orig, base))) ? (double)hi + 0.001 : (double)hi;
        for (int i = 0; i < half; i++) {
            double ramp = ((double)i - (double)lo) / (hif - (double)lo);
            if (ramp < 0) ramp = 0; if (ramp > 1) ramp = 1;
            double smooth = 1.0 - ramp;
            fr[i] = fr[i] / factor * (1.0 - smooth) + fr[i] * smooth;
        }
    }
    for (int t = 0; t < seqlen; t++)
        for (int i = 0; i < half; i++) {
            double ang = (double)t * fr[i];
            cos_out[t * half + i] = (float)cos(ang);
            sin_out[t * half + i] = (float)sin(ang);
        }
    free(fr);
}

/* apply_rope: xp[rd] (单 token, 单 head) 就地旋转. cos/sin[rd/2] (该 token 行).
 * numpy: z=xc[0]+i*xc[1] (交错对), z*=(cos+i*sin) [inverse→conj]. */
void dq_apply_rope(float *xp, const float *cosr, const float *sinr, int rd, int inverse) {
    int half = rd / 2;
    for (int i = 0; i < half; i++) {
        float re = xp[2 * i], im = xp[2 * i + 1];
        float c = cosr[i], s = inverse ? -sinr[i] : sinr[i];
        xp[2 * i]     = re * c - im * s;
        xp[2 * i + 1] = re * s + im * c;
    }
}

/* out[S,M] = X[S,K] @ W[M,K]^T  (W 行主序, 对应 numpy x@W.T). fp32 累加(近 numpy). */
#ifdef DS4QUANT_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif
/* strided 版(08-18 attention GPU 化): C[S,M]=A[S,K](lda)·B[M,K](ldb)^T, 行主序任意行距 */
void dq_matmul_strided(const float *A, int lda, const float *B, int ldb,
                       float *Cst, int ldc, int S, int K, int M, float alpha) {
#ifdef DS4QUANT_CUDA
    if ((double)S * K * (double)M * 2.0 >= 2.0e8) {
        static __thread cublasHandle_t h2 = NULL;
        static __thread cudaStream_t s2 = NULL;
        if (!h2) {
            if (cublasCreate(&h2) != CUBLAS_STATUS_SUCCESS) h2 = NULL;
            else { cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking); cublasSetStream(h2, s2); }
        }
        if (h2) {
            const float zero = 0.0f;
            if (cublasSgemm(h2, CUBLAS_OP_T, CUBLAS_OP_N, M, S, K,
                            &alpha, B, ldb, A, lda, &zero, Cst, ldc) == CUBLAS_STATUS_SUCCESS &&
                cudaStreamSynchronize(s2) == cudaSuccess)
                return;
        }
    }
#endif
#ifdef DQ_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, S, M, K,
                alpha, A, lda, B, ldb, 0.0f, Cst, ldc);
#else
    for (int s = 0; s < S; s++)
        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += A[(size_t)s * lda + k] * B[(size_t)m * ldb + k];
            Cst[(size_t)s * ldc + m] = acc * alpha;
        }
#endif
}
/* NT 版: C[S,M]=A[S,K](lda)·B[K,M](B 行主序[N,HD]视为 K=N 行 M=HD 列→NoTrans) */
void dq_matmul_nt_strided(const float *A, int lda, const float *B, int ldb,
                          float *Cst, int ldc, int S, int K, int M) {
#ifdef DS4QUANT_CUDA
    if ((double)S * K * (double)M * 2.0 >= 2.0e8) {
        static __thread cublasHandle_t h3 = NULL;
        static __thread cudaStream_t s3 = NULL;
        if (!h3) {
            if (cublasCreate(&h3) != CUBLAS_STATUS_SUCCESS) h3 = NULL;
            else { cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking); cublasSetStream(h3, s3); }
        }
        if (h3) {
            const float one = 1.0f, zero = 0.0f;
            /* RowMajor C=A·B ⇔ ColMajor C'=B'·A' */
            if (cublasSgemm(h3, CUBLAS_OP_N, CUBLAS_OP_N, M, S, K,
                            &one, B, ldb, A, lda, &zero, Cst, ldc) == CUBLAS_STATUS_SUCCESS &&
                cudaStreamSynchronize(s3) == cudaSuccess)
                return;
        }
    }
#endif
#ifdef DQ_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, S, M, K,
                1.0f, A, lda, B, ldb, 0.0f, Cst, ldc);
#else
    for (int s = 0; s < S; s++)
        for (int m = 0; m < M; m++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += A[(size_t)s * lda + k] * B[(size_t)k * ldb + m];
            Cst[(size_t)s * ldc + m] = acc;
        }
#endif
}
void dq_matmul(const float *X, const float *W, float *out, int S, int K, int M) {
#ifdef DS4QUANT_CUDA
    /* GB10 统一内存 cuBLAS 直传(2026-08-18 用户令"必须使用GPU"): malloc 指针 GPU 直访
     * (探针 relerr 1.6e-7, 热点尺寸 2906x4096x2048=3.67ms=13.3 TFLOPS vs CPU 单线程 ~3.5s)。
     * 小 GEMM 走 CPU(launch+sync ~100µs 不划算); per-thread handle+stream: 外层专家
     * pthread 并发提交 GPU 多流; 任一 CUDA 失败静默落 CPU 路径(数值语义同, fp32 同精度)。 */
    /* 阈=2e8(08-18 实测校准): 只让 g_r 级大 GEMM(24 GFLOP)上 GPU。5e6 低阈实测负收益
     * (259s vs 181s/层): 20 线程高频小 GEMM 并发提交, launch+sync 队列争用吃掉全部收益。
     * 高频小矩阵的正确姿势是批量结构改造(GPTQ 段 GPU 常驻), 不是逐调用换后端。 */
    if ((double)S * K * (double)M * 2.0 >= 2.0e8) {
        static __thread cublasHandle_t g_dqh = NULL;
        static __thread cudaStream_t g_dqs = NULL;
        if (!g_dqh) {
            if (cublasCreate(&g_dqh) != CUBLAS_STATUS_SUCCESS) g_dqh = NULL;
            else { cudaStreamCreateWithFlags(&g_dqs, cudaStreamNonBlocking); cublasSetStream(g_dqh, g_dqs); }
        }
        if (g_dqh) {
            const float one = 1.0f, zero = 0.0f;
            /* RowMajor C[S,M]=X·W^T ⇔ ColMajor C'[M,S]=W'^T·X' */
            if (cublasSgemm(g_dqh, CUBLAS_OP_T, CUBLAS_OP_N, M, S, K,
                            &one, W, K, X, K, &zero, out, M) == CUBLAS_STATUS_SUCCESS &&
                cudaStreamSynchronize(g_dqs) == cudaSuccess)
                return;
        }
    }
#endif
#ifdef DQ_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, S, M, K,
                1.0f, X, K, W, K, 0.0f, out, M);
#else
    for (int s = 0; s < S; s++) {
        const float *xr = X + (size_t)s * K;
        for (int m = 0; m < M; m++) {
            const float *wr = W + (size_t)m * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += xr[k] * wr[k];
            out[(size_t)s * M + m] = acc;
        }
    }
#endif
}

/* expert_fp: h = silu(clip(x@w1.T, ≤lim)) * clip(x@w3.T, [-lim,lim]); out = (weight*h) @ w2.T.
 * x[S,DIM], w1/w3[MOEI,DIM], w2[DIM,MOEI]. out[S,DIM] 累加进 acc (weight per-token 或 NULL). */
void dq_expert_fp(const float *x, const float *w1, const float *w3, const float *w2,
                  const float *weight, float *acc_out, int S, int DIM, int MOEI, float swlim) {
    float *g = (float *)malloc((size_t)S * MOEI * sizeof(float));
    float *u = (float *)malloc((size_t)S * MOEI * sizeof(float));
    dq_matmul(x, w1, g, S, DIM, MOEI);
    dq_matmul(x, w3, u, S, DIM, MOEI);
    for (size_t i = 0; i < (size_t)S * MOEI; i++) {
        float gg = g[i], uu = u[i];
        if (swlim > 0) { if (uu > swlim) uu = swlim; if (uu < -swlim) uu = -swlim; if (gg > swlim) gg = swlim; }
        g[i] = dq_silu(gg) * uu;   /* h */
    }
    if (weight) for (int s = 0; s < S; s++) for (int j = 0; j < MOEI; j++) g[(size_t)s*MOEI+j] *= weight[s];
    /* out += h @ w2.T  (w2[DIM,MOEI]) */
#ifdef DQ_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, S, DIM, MOEI,
                1.0f, g, MOEI, w2, MOEI, 1.0f, acc_out, DIM);
#else
    for (int s = 0; s < S; s++) {
        const float *hr = g + (size_t)s * MOEI;
        for (int d = 0; d < DIM; d++) {
            const float *wr = w2 + (size_t)d * MOEI;
            float a = 0.0f;
            for (int j = 0; j < MOEI; j++) a += hr[j] * wr[j];
            acc_out[(size_t)s * DIM + d] += a;
        }
    }
#endif
    free(g); free(u);
}

/* hc_sinkhorn: mixes[S, HCM+HCM+HCM*HCM] → pre[S,HCM], post[S,HCM], comb[S,HCM,HCM]. */
void dq_hc_sinkhorn(const float *mixes, const float *scale, const float *base,
                    float *pre, float *post, float *comb, int S, int HCM, int HCIT, float HCEPS) {
    for (int s = 0; s < S; s++) {
        const float *mx = mixes + (size_t)s * (2*HCM + HCM*HCM);
        for (int j = 0; j < HCM; j++) pre[(size_t)s*HCM+j]  = dq_sigmoid(mx[j]*scale[0]+base[j]) + HCEPS;
        for (int j = 0; j < HCM; j++) post[(size_t)s*HCM+j] = 2.0f*dq_sigmoid(mx[HCM+j]*scale[1]+base[HCM+j]);
        /* comb[HCM,HCM] = softmax over axis2 (last) of mixes[2HCM:].reshape(HCM,HCM)*scale2+base2, +eps */
        float *cb = comb + (size_t)s*HCM*HCM;
        for (int a = 0; a < HCM; a++) {
            float row[64];
            for (int b = 0; b < HCM; b++) row[b] = mx[2*HCM + a*HCM + b]*scale[2] + base[2*HCM + a*HCM + b];
            float sm[64]; dq_softmax(row, sm, HCM);
            for (int b = 0; b < HCM; b++) cb[a*HCM+b] = sm[b] + HCEPS;
        }
        /* comb /= comb.sum(axis1=行内跨a?, keepdims) — numpy: comb.sum(1)=沿HCM第一维(a) */
        /* numpy comb[S,HCM,HCM]; sum(1)=对a求和→[S,1,HCM(b)]; comb/=that. 然后迭代 HCIT-1 次 sum(2) then sum(1). */
        float colsum[64];
        for (int b = 0; b < HCM; b++) { double s2=0; for (int a=0;a<HCM;a++) s2+=cb[a*HCM+b]; colsum[b]=(float)s2+HCEPS; }
        for (int a=0;a<HCM;a++) for (int b=0;b<HCM;b++) cb[a*HCM+b] /= colsum[b];
        for (int it=0; it<HCIT-1; it++) {
            for (int a=0;a<HCM;a++){ double r=0; for(int b=0;b<HCM;b++) r+=cb[a*HCM+b]; float rr=(float)r+HCEPS; for(int b=0;b<HCM;b++) cb[a*HCM+b]/=rr; }
            for (int b=0;b<HCM;b++){ double c=0; for(int a=0;a<HCM;a++) c+=cb[a*HCM+b]; float cc=(float)c+HCEPS; for(int a=0;a<HCM;a++) cb[a*HCM+b]/=cc; }
        }
    }
}

/* hc_pre: h[S,HCM,DIM] → y[S,DIM], post[S,HCM], comb[S,HCM,HCM].
 * x=h.reshape(S,HCM*DIM); mixes=(x@fn.T)*rsqrt(mean(x²)); sinkhorn; y=Σ_j pre[j]*h[j]. */
void dq_hc_pre(const float *h, const float *fn, const float *scale, const float *base,
               float *y, float *post, float *comb, int S, int HCM, int DIM,
               int mixdim, int HCIT, float EPS, float HCEPS) {
    int HD = HCM * DIM;
    float *mixes = (float *)malloc((size_t)S * mixdim * sizeof(float));
    float *pre = (float *)malloc((size_t)S * HCM * sizeof(float));
    for (int s = 0; s < S; s++) {
        const float *x = h + (size_t)s * HD;
        double v = 0.0; for (int i = 0; i < HD; i++) v += (double)x[i]*(double)x[i];
        float rsq = (float)(1.0/sqrt(v/(double)HD + (double)EPS));
        float *mx = mixes + (size_t)s * mixdim;
        for (int m = 0; m < mixdim; m++) {
            const float *fr = fn + (size_t)m * HD;
            float a = 0.0f; for (int k = 0; k < HD; k++) a += x[k]*fr[k];
            mx[m] = a * rsq;
        }
    }
    dq_hc_sinkhorn(mixes, scale, base, pre, post, comb, S, HCM, HCIT, HCEPS);
    for (int s = 0; s < S; s++)
        for (int d = 0; d < DIM; d++) {
            float a = 0.0f;
            for (int j = 0; j < HCM; j++) a += pre[(size_t)s*HCM+j] * h[((size_t)s*HCM+j)*DIM+d];
            y[(size_t)s*DIM+d] = a;
        }
    free(mixes); free(pre);
}

/* hc_post: out[S,HCM,DIM] = post[:,:,None]*a[:,None,:] + einsum('sjk,skd->sjd', comb, resid). */
void dq_hc_post(const float *a, const float *resid, const float *post, const float *comb,
                float *out, int S, int HCM, int DIM) {
    for (int s = 0; s < S; s++)
        for (int j = 0; j < HCM; j++) {
            float pj = post[(size_t)s*HCM+j];
            const float *cj = comb + ((size_t)s*HCM+j)*HCM;
            float *o = out + ((size_t)s*HCM+j)*DIM;
            const float *ar = a + (size_t)s*DIM;
            for (int d = 0; d < DIM; d++) {
                float acc = pj * ar[d];
                for (int k = 0; k < HCM; k++) acc += cj[k] * resid[((size_t)s*HCM+k)*DIM+d];
                o[d] = acc;
            }
        }
}

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
    float *sc = malloc((size_t)NEXP * sizeof(float));
    for (int t = 0; t < Nt; t++) {
        const float *xr = x + (size_t)t * DIM;
        for (int e = 0; e < NEXP; e++) {
            const float *gr = gate + (size_t)e * DIM;
            float raw = 0.0f; for (int k = 0; k < DIM; k++) raw += xr[k]*gr[k];
            sc[e] = sqrtf(log1pf(expf(raw)));      /* orig(无 bias) */
        }
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
    free(sc);
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
    float *q = (float*)malloc((size_t)S*NH*HD*sizeof(float));
    dq_matmul(qr, wqb, q, S, QLR, NH*HD);
    /* per-head rms (mean over HD) + rope */
    for (int s=0;s<S;s++) for (int h=0;h<NH;h++) {
        float *qh = q + ((size_t)s*NH+h)*HD;
        double v=0; for(int d=0;d<HD;d++) v+=(double)qh[d]*qh[d];
        float r=(float)(1.0/sqrt(v/(double)HD+(double)EPS));
        for(int d=0;d<HD;d++) qh[d]*=r;
        dq_apply_rope(qh+(HD-RD), cos_t+(size_t)s*(RD/2), sin_t+(size_t)s*(RD/2), RD, 0);
    }
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
    float *o = (float*)malloc((size_t)S*NH*HD*sizeof(float));
#ifdef DQ_BLAS
    /* per-head 两个 gemm: SC_h=Q_h·kva^T, O_h=P_h·kva. mask/softmax/sink 逻辑与标量路径逐字一致. */
    float *SC = (float*)malloc((size_t)S*N*sizeof(float));
    for (int h=0;h<NH;h++) {
        dq_matmul_strided(q + (size_t)h*HD, NH*HD, kva, HD, SC, N, S, HD, N, scale);
        for (int s=0;s<S;s++) {
            float *scr = SC + (size_t)s*N;
            for (int n=0;n<N;n++) {
                int ok;
                if (n<S) ok = (n<=s) && (n> s-WIN);              /* sliding window causal */
                else     ok = ((n-S) < (s+1)/ratio);              /* comp_ok */
                if (!ok) scr[n]=-INFINITY;
            }
            float m=-INFINITY; for(int n=0;n<N;n++) if(scr[n]>m) m=scr[n];
            double denom=exp((double)sink[h]-m);
            for(int n=0;n<N;n++){ if(scr[n]==-INFINITY){scr[n]=0;continue;} scr[n]=expf(scr[n]-m); denom+=scr[n]; }
            float inv=(float)(1.0/denom);
            for(int n=0;n<N;n++) scr[n]*=inv;
        }
        dq_matmul_nt_strided(SC, N, kva, HD, o + (size_t)h*HD, NH*HD, S, N, HD);
    }
    free(SC);
    for (int s=0;s<S;s++) for (int h=0;h<NH;h++)
        dq_apply_rope(o+((size_t)s*NH+h)*HD+(HD-RD), cos_t+(size_t)s*(RD/2), sin_t+(size_t)s*(RD/2), RD, 1);  /* inverse */
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
    for(int g=0;g<OG;g++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, S, OLR, GD,
                    1.0f, o + (size_t)g*GD, NH*HD, wo_a + (size_t)g*OLR*GD, GD,
                    0.0f, oo + (size_t)g*OLR, OG*OLR);
#else
    for(int s=0;s<S;s++) for(int g=0;g<OG;g++){
        const float *od = o + (size_t)s*NH*HD + (size_t)g*GD;
        for(int r=0;r<OLR;r++){ const float *wr=wo_a+((size_t)g*OLR+r)*GD; float a=0; for(int d=0;d<GD;d++) a+=od[d]*wr[d]; oo[((size_t)s*OG+g)*OLR+r]=a; }
    }
#endif
    dq_matmul(oo, wo_b, out, S, OG*OLR, DIM);   /* [S,OG*OLR]@wo_b[DIM,OG*OLR].T */
    free(qr);free(qra);free(q);free(kv);free(kva);free(o);free(oo);
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
