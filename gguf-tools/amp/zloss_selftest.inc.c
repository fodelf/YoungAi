/* zloss_selftest.inc.c — zloss_solve 的合成金标(物理分片, 只被 zloss_solve.c
 * include)。种两模式各一张 rank-8 线性图, M=2 必须近零收回, M=1 必须收不动
 * —— 验的是 残差聚类→x 门→模式打包→eval 的全布线, 不是理论。
 *
 * 金标形态(两次翻案的教训都固化在此):
 * ①x = B·h 住 H=64 维流形 —— 真激活是低维各向异性的, 金标必须同构; 满维各向
 *   同性随机 x 下 1.5k 行撑不起 4096² 图的泛化, 谁来解都收不回(数据墙非布线针)。
 * ②模式种在"靶怎么依赖 x"里(sign(p·h) 选 U_c·G_c), x 的密度分布对两模式完全
 *   对称 —— x 侧无监督聚类原理上看不见这刀怎么切(任意对径切分密度等价),
 *   逼着解算器走 残差方向聚类+x 侧门 的路(gate(x) 支柱)。 */
static uint64_t st_s = 0x9E3779B97F4A7C15ULL;
static float st_u(void) {
    st_s = st_s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)((int64_t)(st_s >> 33) % 2000001 - 1000000) / 1e6f;
}
static void st_synth(int ntok, float **Xo, float **Ro, float **Yso) {
    const int RK = 8, H = 64;
    float *X = xmalloc((size_t)ntok * D * 4), *R = xmalloc((size_t)ntok * D * 4);
    float *Ys = xmalloc((size_t)ntok * D * 4);
    float *B = xmalloc((size_t)D * H * 4), *ph = xmalloc(H * 4), *h = xmalloc(H * 4);
    float *U = xmalloc((size_t)2 * RK * D * 4), *G = xmalloc((size_t)2 * RK * H * 4);
    for (size_t i = 0; i < (size_t)D * H; i++) B[i] = st_u();
    for (int t = 0; t < H; t++) ph[t] = st_u();
    for (size_t i = 0; i < (size_t)2 * RK * D; i++) U[i] = st_u();
    for (size_t i = 0; i < (size_t)2 * RK * H; i++) G[i] = st_u();
    memset(Ys, 0, (size_t)ntok * D * 4);
    for (int i = 0; i < ntok; i++) {
        float *x = X + (size_t)i * D, *r = R + (size_t)i * D;
        double dp = 0;
        for (int t = 0; t < H; t++) { h[t] = st_u(); dp += (double)h[t] * ph[t]; }
        int c = dp > 0;
        for (int j = 0; j < D; j++) {
            double a = 0; const float *b = B + (size_t)j * H;
            for (int t = 0; t < H; t++) a += (double)b[t] * h[t];
            x[j] = (float)a;
        }
        memset(r, 0, D * 4);
        for (int rr = 0; rr < RK; rr++) {
            const float *g = G + ((size_t)c * RK + rr) * H, *u = U + ((size_t)c * RK + rr) * D;
            double a = 0; for (int t = 0; t < H; t++) a += (double)g[t] * h[t];
            a /= H;
            for (int j = 0; j < D; j++) r[j] += (float)(a * u[j]);
        }
    }
    free(B); free(ph); free(h); free(U); free(G);
    *Xo = X; *Ro = R; *Yso = Ys;
}
