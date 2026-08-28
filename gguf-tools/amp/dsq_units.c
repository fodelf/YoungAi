/* dsq_units.c — 量化 / 反修 / sweep 三段的可单测单元(2026-08-28 用户令)
 *
 * 为什么有这个文件: ds4quant_run 是 6000 行单 TU, 93 个静态全局 + 169 处 getenv,
 * 量化那段连函数边界都没有(裸 for(L) 循环在 main 里)。想给三段做一个"跑一层看速度和质量"
 * 的针, 只能往整条流水线里塞变量 —— 实撞的后果: 闸挂错段(PROBE1 是旧量化段的开关, quant86
 * 不认)、变量被 unset、量化停不下来、被杀的子进程成孤儿去污染下一段的计时。
 * 结论是架构问题不是参数问题, 所以把三段的核心单元拆出来, 各自能用合成小数据独立跑。
 *
 * 拆分原则(用户令"合理拆分别浪费时间"): 只搬, 不重写。
 *   量化 → vq_qc.h 里 vq_encode_full/vq_pack/vq_unpack_dequant 本来就是参数化的, 直接包一层
 *          成"量化一个矩阵并给出还原度"。
 *   反修 → elm_solve 本来就是纯函数(零全局零 getenv), 直接用。
 *   sweep → 落地决策(从候选里择优 + 增益门 + 终验回滚)原样搬自 ds4quant_run_p13:280-285
 *          与 417-425, 只把它从一堆全局里摘出来变成传参。
 *
 * 自测: make -C gguf-tools tools-test 会带上 -DDSQ_UNITS_TEST 编译本文件并跑。
 */
/* Linux 上 -std=c11 是严格 ISO 模式, clock_gettime/pwrite/_SC_NPROCESSORS_ONLN 都不暴露
 * (vq_qc.h 与 ds4quant_fwd 都要)。★用 _GNU_SOURCE 不用 _POSIX_C_SOURCE★:
 * 后者只开 POSIX, _SC_NPROCESSORS_ONLN 是 GNU/BSD 扩展, 仍然被挡(实撞)。
 * macOS 默认全暴露, 不需要任何宏。 */
#if defined(__linux__)
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>      /* vq_qc.h 的 vqt_now 用 clock_gettime, Linux 上必须先有这个 */
#include <pthread.h>

/* ══ ① 量化单元 ════════════════════════════════════════════════════════════
 * 量化一个权重矩阵并回报还原度。这是"量化一层"的最小可测单位 —— 一层就是
 * NEXP×3 个矩阵各跑一遍这个。
 *   W      [rows][cols] 原始权重
 *   dim/nc VQ 档位(向量维/码本大小), 如 4/512 = v4x512
 *   X      [n_act][cols] 校准激活(可为 NULL = 无加权)
 * 出参:
 *   *relh  相对残差 ‖W−Ŵ‖/‖W‖ —— 就是 plan.txt 里那个 relh
 *   *bytes 打包后的字节数(体积账)
 * 返回 0=成功。往返自检(pack→unpack 逐位)由调用方或自测做。 */
int dsq_quant_matrix(const float *W, int rows, int cols, int dim, int nc,
                     const float *X, int n_act, double *relh, size_t *bytes);

/* ══ ② 反修单元 ════════════════════════════════════════════════════════════
 * 见 ds4quant_elm.inc.c 的 elm_solve: 给一层的 (x, y_q, dH) 解乘性非线性放大器,
 * 回报 held 行为挽回。签名在那边, 这里不重复声明 —— 自测直接 include 它。 */

/* ══ ③ sweep 单元 ══════════════════════════════════════════════════════════
 * sweep 的决策核心有两处, 原样搬出来:
 *   (a) 单元择优: 从候选出口分里挑最优, 过增益门才算落地(p13:280-285)
 *   (b) 统一终验: 全部落地后一次全程复核, 不改善就整体回滚(p13:417-425)
 * 口径约定: 分数【越小越好】(出口误差)。
 * ★(c) 候选评估 dsq_sweep_layer 在本文件下方(要等 ds4_z/dq_hc_post/bf_exit_relL2
 *   都 include 进来才能写) —— sweep 的耗时全在 (c), (a)(b) 只是决策收尾。★ */

/* (a) 候选择优 + 增益门。
 *   base    基线分
 *   cand[n] 候选分(NAN = 该候选无效, 跳过)
 *   gate    最小相对增益(如 0.05 = 至少降 5%); <=0 表示只要严格更小即可
 * 出参 *win = 胜出候选下标, *ws = 胜出分数。
 * 返回 1=有落地 0=无落地(base 保持)。 */
int dsq_sweep_pick(double base, const double *cand, int n, double gate,
                   int *win, double *ws)
{
    double best = base; int bi = -1;
    for (int i = 0; i < n; i++) {
        if (!(cand[i] == cand[i])) continue;              /* NAN 跳过 */
        if (cand[i] < best - 1e-9) { best = cand[i]; bi = i; }
    }
    if (bi < 0) return 0;
    if (gate > 0.0) {                                      /* 相对增益门 */
        const double rel = (base > 1e-12) ? (base - best) / base : 0.0;
        if (rel < gate) return 0;
    }
    if (win) *win = bi;
    if (ws)  *ws  = best;
    return 1;
}

/* (b) 统一终验: 全部落地后在【真前沿全量】上复核。
 * 返回 1=提交 0=整体回滚。判据与 p13:422 逐字一致(fin<b0-1e-9)。 */
int dsq_sweep_commit(double before, double after)
{
    return (after < before - 1e-9) ? 1 : 0;
}

/* ── 量化单元实现 ─────────────────────────────────────────────────────────── */
#ifndef DSQ_UNITS_NO_VQ
/* ★单元测试与生产必须是同一套代码★(铁律 2026-08-28 用户令"一套代码一套速度和质量")
 * 原来这里为了"能编过"写了 dq_matmul / g2_inv / g2_hot_slot / vq_hot_* 一堆参考实现桩,
 * 结果就是【两套实现】: 生产走 ds4quant_fwd 的 cuBLAS dq_matmul 与 go2b_qc 的真 g2_inv,
 * 单测走我的标量版 —— 速度差一个数量级、数值也不保证一致, 测了等于没测。
 * 现在全部换成生产同一份源码:
 *   DIM/MOEI      与 ds4quant_run_p1 同值(它们是编译期常量, ds4quant_fwd 要)
 *   dq_matmul     ← ds4quant_fwd.c(带 -DDS4QUANT_CUDA 时是 cuBLAS 路)
 *   g2_inv/g2_hot_slot/vq_hot_* ← go2b_qc.h(生产同一份)
 * ★include 顺序照抄 ds4quant_run_p1(27→28→30→167)★: fwd 在前, DIM/MOEI 的
 * #define 必须排在它们【之后】—— dq_expert_fp 的形参名就叫 DIM/MOEI, 先 define 会把
 * 函数签名撞碎(实撞: "expected ')'" + swlim undeclared)。go2b_qc.h 用 dq_matmul, 也得在后。 */
#include "ds4quant_fwd.c"
#include "../quantize/onebit_quant.c"
#include "../quantize/go2b_qc.h"
#define DIM  4096
#define MOEI 2048
#include "../quantize/vq_qc.h"

int dsq_quant_matrix(const float *W, int rows, int cols, int dim, int nc,
                     const float *X, int n_act, double *relh, size_t *bytes)
{
    if (!W || rows <= 0 || cols <= 0 || dim <= 0 || nc <= 1) return -1;
    if (cols % dim) return -1;                             /* 列必须整除向量维 */
    const int nvec = rows * (cols / dim);
    float *Cb  = malloc((size_t)nc * dim * sizeof(float));
    int   *idx = malloc((size_t)nvec * sizeof(int));
    float *gr  = malloc((size_t)rows * sizeof(float));
    float *Wq  = malloc((size_t)rows * cols * sizeof(float));
    if (!Cb || !idx || !gr || !Wq) { free(Cb);free(idx);free(gr);free(Wq); return -1; }

    vq_encode_full(W, rows, cols, dim, nc, X, n_act, Cb, idx, gr, Wq);

    if (relh) {                                            /* relh = ‖W−Ŵ‖/‖W‖ */
        double e = 0.0, w = 0.0;
        for (size_t i = 0; i < (size_t)rows * cols; i++) {
            const double d = (double)W[i] - Wq[i];
            e += d * d; w += (double)W[i] * W[i];
        }
        *relh = sqrt(e / (w + 1e-30));
    }
    if (bytes) {
        uint8_t *buf = malloc(vq_payload_bytes(rows, cols, dim, nc) + 4096);
        *bytes = buf ? vq_pack(Cb, idx, gr, rows, cols, dim, nc, buf) : 0;
        free(buf);
    }
    free(Cb); free(idx); free(gr); free(Wq);
    return 0;
}

/* 同上, 但把反量化后的权重写出来(反修要拿量化态权重再前向一遍) */
int dsq_quant_matrix_wq(const float *W,int rows,int cols,int dim,int nc,
                        const float *X,int n_act,float *Wq_out,double *relh,size_t *bytes)
{
    if(!W||!Wq_out||cols%dim) return -1;
    const int nvec=rows*(cols/dim);
    float *Cb=malloc((size_t)nc*dim*4); int *idx=malloc((size_t)nvec*4);
    float *gr=malloc((size_t)rows*4);
    if(!Cb||!idx||!gr){ free(Cb);free(idx);free(gr); return -1; }
    vq_encode_full(W,rows,cols,dim,nc,X,n_act,Cb,idx,gr,Wq_out);
    if(relh){ double e=0,w=0;
        for(size_t i=0;i<(size_t)rows*cols;i++){ const double d=(double)W[i]-Wq_out[i]; e+=d*d; w+=(double)W[i]*W[i]; }
        *relh=sqrt(e/(w+1e-30)); }
    if(bytes){ uint8_t *bf=malloc(vq_payload_bytes(rows,cols,dim,nc)+4096);
        *bytes = bf ? vq_pack(Cb,idx,gr,rows,cols,dim,nc,bf) : 0; free(bf); }
    free(Cb);free(idx);free(gr); return 0;
}
#endif

/* ══════════════════════════ 自测 ══════════════════════════════════════════ */
#ifdef DSQ_UNITS_TEST
/* 反修单元要用 elm_solve, 它依赖同-TU 的 cholesky/chol_solve_multi/mgs/lcg_unit/zpar_for。
 * 这些原语分别在 ds4_z.c 与 ds4quant_zsolve.inc.c 里(都是 static), 自测把它们拉进来。
 * DIM/MOEI/NEXP 在 elm 里没被用到(全走参数), 所以不需要 p1 的那些编译期常量。 */
/* vqt_now/time.h/pthread.h 见文件头 */
#include "../../ds4_z.c"
#include "ds4quant_zsolve.inc.c"
#include "ds4quant_elm.inc.c"

#include "dsq_units_sweep.inc.c"   /* ③ 候选评估(500 行守卫所迫的物理分片) */

static uint64_t TS = 0x243F6A8885A308D3ULL;
static float rnd(void){ TS = TS*6364136223846793005ULL + 1442695040888963407ULL;
    return ((float)((TS>>40)&0xFFFFFF)/8388608.0f) - 1.0f; }

static int t_quant(void)
{
    /* 合成小矩阵: 32×64, v4x16。数据造成"低秩+噪声", VQ 应当能压得住。 */
    const int rows=32, cols=64, dim=4, nc=16;
    float *W = malloc((size_t)rows*cols*4);
    for(int r=0;r<rows;r++) for(int c=0;c<cols;c++)
        W[r*cols+c] = sinf(0.13f*r + 0.07f*c) + 0.05f*rnd();
    double relh_lo=0, relh_hi=0; size_t b_lo=0, b_hi=0;
    int rc1 = dsq_quant_matrix(W,rows,cols,dim,nc,   NULL,0,&relh_lo,&b_lo);
    int rc2 = dsq_quant_matrix(W,rows,cols,dim,nc*4, NULL,0,&relh_hi,&b_hi);
    printf("  量化: v%dx%-4d relh=%.4f %zuB | v%dx%-4d relh=%.4f %zuB\n",
           dim,nc,relh_lo,b_lo, dim,nc*4,relh_hi,b_hi);
    int ok = 1;
    if(rc1||rc2){ printf("  ✗ 返回码 %d/%d\n",rc1,rc2); ok=0; }
    if(!(relh_lo>0.0 && relh_lo<1.0)){ printf("  ✗ relh 越界 %.4f\n",relh_lo); ok=0; }
    /* 码本变大 ⇒ 还原度必须变好(单调性) —— 这是量化档位账的基本性质 */
    if(!(relh_hi < relh_lo)){ printf("  ✗ 码本×4 反而更差 %.4f→%.4f\n",relh_lo,relh_hi); ok=0; }
    /* 体积必须随码本增大(码本本身进载荷) */
    if(!(b_hi > b_lo)){ printf("  ✗ 体积没随码本增大 %zu→%zu\n",b_lo,b_hi); ok=0; }
    /* 非法参数必须被挡(cols 不整除 dim) */
    if(dsq_quant_matrix(W,rows,cols,7,nc,NULL,0,NULL,NULL)==0){ printf("  ✗ 非法 dim 没被挡\n"); ok=0; }
    free(W);
    return ok;
}

static int t_backfit(void)
{
    /* 合成一层: S 行 × D 维。造一个【乘性】真值 y_fp = y_q ⊙ (1+tanh(x·v)·u),
     * ELM 应当能把它挽回一大截; 而同口径的乘性【线性】对照应当明显更差 ——
     * 这正是 fable5:5316 判例(线性近零 / 非线性 +12%)的可复现最小化版本。 */
    const int S=512, D=64, vs=384;
    float *X  = malloc((size_t)S*D*4);
    float *YQ = malloc((size_t)S*D*4);
    float *DH = malloc((size_t)S*D*4);
    float *v  = malloc((size_t)D*4), *u = malloc((size_t)D*4);
    for(int d=0;d<D;d++){ v[d]=rnd(); u[d]=0.3f*rnd(); }
    double vn=0; for(int d=0;d<D;d++) vn+=(double)v[d]*v[d]; vn=sqrt(vn);
    for(int d=0;d<D;d++) v[d]/=(float)vn;
    for(int i=0;i<S;i++){
        float *x=X+(size_t)i*D, *yq=YQ+(size_t)i*D, *dh=DH+(size_t)i*D;
        /* ★x 必须各向异性★(首跑实撞): 造成各向同性 U(-1,1)^D 时, PCA 给出的是任意旋转
         * 方向, tanh(x·V₀_c/s) 这组特征与靶方向 v 毫无关系 ⇒ ELM 输给线性(58.6% vs 61.7%)。
         * 真实激活是高度各向异性的(massive activation 通道占 10% 能量, 见 memory 巨值通道矿),
         * PCA 才能捞到真结构 —— ELM 历史上的 +12% 就建立在这个前提上。
         * 这里让 v 成为主方向(能量 ×4), 复现真实场景。 */
        const float a0 = 4.0f*rnd();
        for(int d=0;d<D;d++){ x[d]=a0*v[d] + 0.5f*rnd(); yq[d]=1.0f+0.5f*rnd(); }
        double p=0; for(int d=0;d<D;d++) p+=(double)x[d]*v[d];
        /* ★×4 是必须的, 首跑实撞★: v 是单位向量、x~U(-1,1)^D ⇒ x·v 的 std 只有 ~0.58,
         * tanh 在 |t|<0.6 上几乎是直线, 线性支照样拟合得一样好(首跑 线性 71.4% > ELM 67.5%,
         * 测试判 FAIL —— 是靶造错了不是代码错了)。放大进饱和区, 非线性才成为必需。
         * 这条同时是 ELM 的一个真实敏感点: s=sqrt(mean(Xa²)) 这个定标决定 tanh 落在饱和区
         * 还是线性区; 落线性区 ⇒ ELM 退化成线性 ⇒ 没肉。部署时要看 s 是否让 x·V₀/s 出线性区。 */
        const float g0 = tanhf(4.0f*(float)p);            /* 非线性隐变量(进饱和区) */
        for(int d=0;d<D;d++) dh[d] = yq[d]*g0*u[d];       /* y_fp−y_q = y_q⊙(g0·u) */
    }
    elm_res er; int rc = elm_solve(X,YQ,DH,S,D,vs,&er);
    int ok = 1;
    if(rc!=0){ printf("  ✗ elm_solve 返回 %d\n",rc); ok=0; }
    else {
        printf("  反修: held挽回=%.1f%% @V₀=%s λ=%g k=%d | 乘性线性对照=%.1f%%\n",
               er.held*100.0, er.from_pca?"PCA":"rand", (double)er.lam, er.k, er.held_lin*100.0);
        /* 真值就是乘性非线性 ⇒ ELM 必须挽回一大截 */
        if(!(er.held > 0.5)){ printf("  ✗ 造的就是乘性非线性靶, held 只有 %.3f\n",er.held); ok=0; }
        /* 非线性必须明显赢过同口径线性 —— 判例的可复现最小化版本 */
        if(!(er.held > er.held_lin + 0.05)){
            printf("  ✗ 非线性没赢线性: %.3f vs %.3f\n",er.held,er.held_lin); ok=0; }
        if(er.k<16||er.k>512){ printf("  ✗ k 越界 %d\n",er.k); ok=0; }
        elm_free(&er);
    }
    free(X);free(YQ);free(DH);free(v);free(u);
    return ok;
}

static int t_sweep(void)
{
    int ok=1, win=-1; double ws=0;
    const double NA = 0.0/0.0;
    /* ① 有更优候选且无门 ⇒ 落地, 且选最小的那个 */
    double c1[4] = {0.95, 0.80, NA, 0.90};
    if(!dsq_sweep_pick(1.00,c1,4,0.0,&win,&ws) || win!=1 || ws!=0.80){
        printf("  ✗ 择优错: win=%d ws=%.3f\n",win,ws); ok=0; }
    /* ② 增益不够门槛 ⇒ 不落地(今天 38/42 个单元 Δ 只有千分之几, 就该被这道门挡住) */
    double c2[2] = {0.998, 0.999};
    if(dsq_sweep_pick(1.00,c2,2,0.05,&win,&ws)){ printf("  ✗ 增益 0.2%% 过了 5%% 门\n"); ok=0; }
    /* ③ 全是 NAN / 全都更差 ⇒ 不落地 */
    double c3[3] = {NA,NA,NA}, c4[2] = {1.01,1.20};
    if(dsq_sweep_pick(1.00,c3,3,0.0,&win,&ws)){ printf("  ✗ 全 NAN 却落地\n"); ok=0; }
    if(dsq_sweep_pick(1.00,c4,2,0.0,&win,&ws)){ printf("  ✗ 全更差却落地\n"); ok=0; }
    /* ④ 终验: 改善提交 / 劣化整体回滚(今天 champ86 靠的就是这道网) */
    if(!dsq_sweep_commit(145.78,126.17)){ printf("  ✗ 改善没提交\n"); ok=0; }
    if( dsq_sweep_commit(126.17,145.78)){ printf("  ✗ 劣化没回滚\n"); ok=0; }
    if( dsq_sweep_commit(1.0,1.0))      { printf("  ✗ 持平当成改善\n"); ok=0; }
    printf("  sweep: 择优/增益门/NAN/全负/终验提交/终验回滚/持平 七项\n");
    return ok;
}

/* ══ 真模型一层模式 ══════════════════════════════════════════════════════════
 * 用法: dsq_units_test --hf <HF目录> --anchor <锚文件> --layer L [--experts N] [--rows S]
 * 三段全吃真数据, 各报墙钟 + 质量:
 *   ①量化 : 从 HF 读第 L 层前 N 个专家的 w1/w3/w2 全部量化(v4x512=86G 平权档),
 *           校准激活按生产口径分矩阵给: w1/w3←Xc(命中 token 的 x), w2←中间态 h
 *   ②反修 : x 取锚里的 fin[L](★这就是该层 MoE 的真输入★), 路由取 ridx/rw;
 *           y_fp = FP 专家前向, y_q = 量化权重前向, dH = 差 ⇒ elm_solve
 *   ③sweep: 用②的真实候选分跑落地决策
 * 锚布局(ds4quant_anchor.inc.c 逐字):
 *   [0]u32 hd[8]={'DQA2',S,HCM,DIM,NLAYERS,VOCAB,NACT,_} [32]u64 idh
 *   [40] fin[NL][S][DIM] | ridx[NL][S][NACT] | rw[NL][S][NACT] | H[NL][S][HCM][DIM] | logits
 * ★速度口径必须标注★: 本单元的量化走 vq_encode_full 的 CPU 参考路, 生产走 vq_gpu.cu 的
 * GPU kmeans(实测快一个数量级) —— 这里的秒数不能直接当生产速度用。 */
#ifdef __APPLE__
/* macOS 没有 posix_fadvise(ds4_st.c 用它提示内核丢页)。真跑在 spark 上, Mac 只需编过 ⇒ 空操作。 */
#define POSIX_FADV_DONTNEED 0
static int posix_fadvise(int fd,long off,long len,int adv){ (void)fd;(void)off;(void)len;(void)adv; return 0; }
#endif
#include "../../src/common/ds4_st.c"

static void expert_fwd(const float *x,const float *w1,const float *w3,const float *w2,
                       const float *ww,float *acc,int n,int D,int M)
{   /* SwiGLU 专家前向, 与 dq_expert_fp 同式(silu(x@w1)*(x@w3) @ w2ᵀ, 行权在 w2 前乘) */
    float *g=malloc((size_t)n*M*4),*u=malloc((size_t)n*M*4);
    dq_matmul(x,w1,g,n,D,M); dq_matmul(x,w3,u,n,D,M);
    for(size_t i=0;i<(size_t)n*M;i++){ const float t=g[i]; g[i]=(t/(1.0f+expf(-t)))*u[i]; }
    if(ww) for(int s2=0;s2<n;s2++) for(int j=0;j<M;j++) g[(size_t)s2*M+j]*=ww[s2];
    float *t2=malloc((size_t)n*D*4);
    dq_matmul(g,w2,t2,n,M,D);
    for(size_t i=0;i<(size_t)n*D;i++) acc[i]+=t2[i];
    free(g);free(u);free(t2);
}

static int real_layer(const char *hf,const char *anc,int L,int NE,int SROW,int VDIM,int VNC)
{
    /* ── 读锚头 ── */
    FILE *f=fopen(anc,"rb"); if(!f){ printf("锚打不开: %s\n",anc); return 1; }
    uint32_t hd[8]; uint64_t idh;
    if(fread(hd,4,8,f)!=8||fread(&idh,8,1,f)!=1){ printf("锚头读失败\n"); fclose(f); return 1; }
    if(hd[0]!=0x32415144u){ printf("锚 magic 不对 0x%08X\n",hd[0]); fclose(f); return 1; }
    const int S=(int)hd[1], ancHCM=(int)hd[2], D=(int)hd[3], NL=(int)hd[4], NACT=(int)hd[6];
    const int M=2048;                              /* MOEI, 与 p1 一致 */
    if(L<0||L>=NL){ printf("层号越界 %d/%d\n",L,NL); fclose(f); return 1; }
    if(SROW<=0||SROW>S) SROW=S;
    printf("锚: S=%d HCM=%d DIM=%d NLAYERS=%d NACT=%d → 取 L%d 前 %d 行, %d 个专家\n",
           S,ancHCM,D,NL,NACT,L,SROW,NE);
    const size_t fin_b=(size_t)NL*S*D*4, ridx_b=(size_t)NL*S*NACT*4;
    float *X=malloc((size_t)SROW*D*4);
    int32_t *RI=malloc((size_t)SROW*NACT*4); float *RW=malloc((size_t)SROW*NACT*4);
    int rc = fseek(f,40+(long)((size_t)L*S*D*4),SEEK_SET)==0 &&
             fread(X,4,(size_t)SROW*D,f)==(size_t)SROW*D;
    rc &= fseek(f,40+(long)fin_b+(long)((size_t)L*S*NACT*4),SEEK_SET)==0 &&
          fread(RI,4,(size_t)SROW*NACT,f)==(size_t)SROW*NACT;
    rc &= fseek(f,40+(long)fin_b+(long)ridx_b+(long)((size_t)L*S*NACT*4),SEEK_SET)==0 &&
          fread(RW,4,(size_t)SROW*NACT,f)==(size_t)SROW*NACT;
    fclose(f);
    if(!rc){ printf("锚读失败(文件短?)\n"); return 1; }

    st_ctx C; memset(&C,0,sizeof C); st_open(&C,hf);
    float *YQ=calloc((size_t)SROW*D,4), *YF=calloc((size_t)SROW*D,4);
    double relh_sum=0; size_t bytes=0; int nq=0;
    double t_rd=0, t_q=0, t_hc=0, t_fwd=0;   /* 分段: 读盘 / 纯量化 / 中间态 / 前向 */
    int *tok=malloc((size_t)SROW*4); float *ww=malloc((size_t)SROW*4);
    float *xs=malloc((size_t)SROW*D*4), *Wq=malloc((size_t)M*D*4);

    /* ── ①量化 ── */
    const double t0=vqt_now();
    for(int e=0;e<NE;e++){
        char n1[192],n3[192],n2[192]; long r,c;
        snprintf(n1,sizeof n1,"layers.%d.ffn.experts.%d.w1.weight",L,e);
        snprintf(n3,sizeof n3,"layers.%d.ffn.experts.%d.w3.weight",L,e);
        snprintf(n2,sizeof n2,"layers.%d.ffn.experts.%d.w2.weight",L,e);
        const double tr0=vqt_now();
        float *e1=st_read_weight(&C,n1,&r,&c), *e3=st_read_weight(&C,n3,&r,&c),
              *e2=st_read_weight(&C,n2,&r,&c);
        t_rd += vqt_now()-tr0;
        if(!e1||!e3||!e2){ printf("专家 %d 权重缺\n",e); free(e1);free(e3);free(e2); break; }
        /* 本专家命中的 token = 该专家的校准激活 Xc(生产口径, ds4quant_run_p9:89) */
        int nt=0;
        for(int s2=0;s2<SROW;s2++) for(int a2=0;a2<NACT;a2++)
            if(RI[(size_t)s2*NACT+a2]==e){ tok[nt]=s2; ww[nt]=RW[(size_t)s2*NACT+a2];
                memcpy(xs+(size_t)nt*D,X+(size_t)s2*D,(size_t)D*4); nt++; break; }
        if(!nt){ free(e1);free(e3);free(e2); continue; }
        /* ★校准激活按生产口径分矩阵给★(p9:89/145 w1,w3←Xc; p9:155-161 w2←中间态 h) */
        double rh; size_t bt;
        float *q1=malloc((size_t)M*D*4), *q3=malloc((size_t)M*D*4);
        /* w2 的校准激活 = 中间态 h = silu(x@w1)·(x@w3), 用 FP 权重算(生产同) */
        float *hc=malloc((size_t)nt*M*4),*gg=malloc((size_t)nt*M*4);
        const double th0=vqt_now();
        dq_matmul(xs,e1,hc,nt,D,M); dq_matmul(xs,e3,gg,nt,D,M);
        for(size_t i=0;i<(size_t)nt*M;i++){ const float t=hc[i]; hc[i]=(t/(1.0f+expf(-t)))*gg[i]; }
        free(gg); t_hc += vqt_now()-th0;
        /* ★只编码一次★: wq 版同时出 relh/bytes/Wq。原来 dsq_quant_matrix + _wq 各调一遍,
         * 24 个矩阵编码了 48 次, 还把这笔重复算进了"量化速度"。 */
        const double tq0=vqt_now();
        if(dsq_quant_matrix_wq(e1,M,D,VDIM,VNC,xs,nt,q1,&rh,&bt)==0){ relh_sum+=rh; bytes+=bt; nq++; }
        if(dsq_quant_matrix_wq(e3,M,D,VDIM,VNC,xs,nt,q3,&rh,&bt)==0){ relh_sum+=rh; bytes+=bt; nq++; }
        if(dsq_quant_matrix_wq(e2,D,M,VDIM,VNC,hc,nt,Wq,&rh,&bt)==0){ relh_sum+=rh; bytes+=bt; nq++; }
        t_q += vqt_now()-tq0;
        free(hc);
        /* FP 与【三矩阵全量化】各前向一遍 —— 这才是该层真实的量化误差 */
        const double tf0=vqt_now();
        float *aF=calloc((size_t)nt*D,4), *aQ=calloc((size_t)nt*D,4);
        expert_fwd(xs,e1,e3,e2,ww,aF,nt,D,M);
        expert_fwd(xs,q1,q3,Wq,ww,aQ,nt,D,M);
        for(int i=0;i<nt;i++) for(int d=0;d<D;d++){
            YF[(size_t)tok[i]*D+d]+=aF[(size_t)i*D+d];
            YQ[(size_t)tok[i]*D+d]+=aQ[(size_t)i*D+d]; }
        free(aF);free(aQ);free(q1);free(q3);
        t_fwd += vqt_now()-tf0;
        free(e1);free(e3);free(e2);
    }
    const double t1=vqt_now();
    if(!nq){ printf("★量化 0 个矩阵 —— HF 路径或层号不对★\n"); return 1; }
    /* ★速度必须分段★(2026-08-28 用户揪出): 原来一个计时器包住整个专家循环, 把
     * 读盘(HF FP8 反量化)、中间态两次大 GEMM、两次全量专家前向 全算进了"量化速度"。 */
    printf("①量化   ★纯编码 %6.1fs★ (%d 矩阵, %.2fs/矩阵)  平均relh=%.4f  载荷=%.1f MiB\n",
           t_q, nq, t_q/(nq?nq:1), relh_sum/nq, bytes/1048576.0);
    printf("         同循环其余: 读HF权重 %.1fs | 中间态h %.1fs | FP+量化双前向 %.1fs | 合计 %.1fs\n",
           t_rd, t_hc, t_fwd, t1-t0);
    /* ★铁律 2026-08-28「只有 GPU 版本」★ 走 vq_gpu.cu 的 GPU kmeans(vq_qc.h:53 的
     * nv>=8192 && vqg_shm_ok && vqg_ready 三条同时成立才进)。
     * 本单元【单线程】跑完整层的 744 个矩阵; 生产是 20 个专家线程并发 ⇒ 除以 20 才是
     * 可比的层时间(实测 377s/20 ≈ 19s, 与生产 60-80s 同量级)。 */
    printf("         GPU 路(vqg_assign) 单线程 ⇒ 生产 20 线程并发折算 ≈ %.1fs/层\n", t_q/20.0);

    /* ── ②反修 ── */
    float *DH=malloc((size_t)SROW*D*4);
    for(size_t i=0;i<(size_t)SROW*D;i++) DH[i]=YF[i]-YQ[i];
    double eq=0,ef=0; for(size_t i=0;i<(size_t)SROW*D;i++){ eq+=(double)DH[i]*DH[i]; ef+=(double)YF[i]*YF[i]; }
    const int vs=(SROW*3)/4;
    elm_res er; const double t2=vqt_now();
    const int erc = elm_solve(X,YQ,DH,SROW,D,vs,&er);
    const double t3=vqt_now();
    if(erc==0){
        /* ★精度要够★: -0.00% 看不出是 -1e-8 还是 -1e-3, 用科学计数把量级摆出来。
         * λ 顶格 100 + k 最小 16 是"什么都没学到"的典型签名(强正则把 U 压到近 0)。 */
        printf("②反修   %6.1fs  held行为挽回=%.4f%% (%.3e)  @V₀=%s λ=%g k=%d | 乘性线性对照=%.4f%% (%.3e)%s\n",
               t3-t2, er.held*100.0, er.held, er.from_pca?"PCA":"rand", (double)er.lam, er.k,
               er.held_lin*100.0, er.held_lin,
               (er.lam>=100.0f && er.k<=16) ? "  ★λ顶格+k最低=网格全负,解不出东西★" : "");
        printf("         量化误差基线: ‖dH‖/‖y_fp‖=%.4f (fit=%d行 held=%d行)\n",
               sqrt(eq/(ef+1e-30)), vs, SROW-vs);
    } else printf("②反修   %6.1fs  解算失败(rc=%d)\n", t3-t2, erc);

    /* ── ③sweep: 候选评估(真活) ──
     * ★口径与限制, 先说清楚★: 候选评估逐行同生产(dsq_sweep_layer ← p7:410-440)。
     * 但 hc 混合系数 post/comb 是链上残差流经 dq_hc_sinkhorn 现算的, 全流程不落盘
     * ⇒ per-layer 单测拿不到链上真值(这是结构性限制, 不是没做)。这里用锚的 H[L]
     * (教师 hc 真值)当 resid 与 Hf, post/comb 由生产 dq_hc_sinkhorn 从 H 的确定性
     * 归约现算。所以:
     *   ✓ 速度 = 真(工作量纯由形状定: 每候选一次 S×HCM×DIM 的 hc 合成 + val 行出口分)
     *   ✓ 门行为 = 真(真 z、真信任域夹持、真判据 e1<e0-1e-9、首个过门即落地)
     *   ★✗ 出口分【绝对值】≠ 生产链上的 e0 —— 不许拿它跟战役日志里的 e0 对表★
     * 反修臂用生产 ds4_z_solve(闭式 ridge + 秩截断)=冠军 r64c 那一族(加性 zl.RRR),
     * 不是 ②的乘性 ELM —— FP 口径下乘性已判无肉(fable5:5330), 加性才是冠军主力。 */
    const double t4=vqt_now();
    int land=-1, k_land=0, n_eval=0; double e0s=0, e1s=0, t_zsolve=0, t_hcw=0;
    if(ancHCM!=HCM){
        printf("③sweep  锚 HCM=%d ≠ 编译期 HCM=%d, 跳过(生产 p1 也是写死 4)\n", ancHCM, HCM);
    } else {
        const size_t hb=(size_t)SROW*HCM*D;
        float *HF=malloc(hb*4), *HQ=malloc(hb*4);
        float *mix=malloc((size_t)SROW*(2*HCM+HCM*HCM)*4);
        float *pre=malloc((size_t)SROW*HCM*4), *pst=malloc((size_t)SROW*HCM*4);
        float *cmb=malloc((size_t)SROW*HCM*HCM*4);
        float *Ftry=malloc((size_t)SROW*D*4), *Fout=malloc((size_t)SROW*D*4);
        /* 锚布局: [40] fin | ridx | rw | H[NL][S][HCM][DIM] | logits */
        const size_t rw_b=(size_t)NL*S*NACT*4;
        int hrc = HF&&HQ&&mix&&pre&&pst&&cmb&&Ftry&&Fout &&
                  fseek(f,40+(long)fin_b+(long)ridx_b+(long)rw_b+
                          (long)((size_t)L*S*HCM*D*4),SEEK_SET)==0 &&
                  fread(HF,4,hb,f)==hb;
        if(!hrc){ printf("③sweep  锚 H[L] 读失败, 跳过\n"); }
        else {
            /* mixes 的确定性归约(见上"限制"): 每行取 H 各 hc 分量的均值做特征 */
            const int nm=2*HCM+HCM*HCM;
            for(int sx=0;sx<SROW;sx++){
                float *mo=mix+(size_t)sx*nm;
                for(int j=0;j<HCM;j++){ double m=0; const float*h=HF+((size_t)sx*HCM+j)*D;
                    for(int d=0;d<D;d++) m+=h[d]; m/=D;
                    mo[j]=(float)m; mo[HCM+j]=(float)m; }
                for(int j=0;j<HCM*HCM;j++) mo[2*HCM+j]=mix[(size_t)sx*nm+(j%HCM)];
            }
            const float hsc[2]={1.0f,1.0f}; float hbase[2*HCM+HCM*HCM]; 
            for(int j=0;j<nm;j++) hbase[j]=0.0f;
            const double th0=vqt_now();
            dq_hc_sinkhorn(mix,hsc,hbase,pre,pst,cmb,SROW,HCM,3,1e-6f);
            t_hcw=vqt_now()-th0;
            /* 反修臂: 生产闭式解, rank=64(冠军 K64) */
            const double tz0=vqt_now();
            ds4_z *zl=ds4_z_solve(X,DH,(uint32_t)SROW,(uint32_t)D,(uint32_t)D,64,1.0f);
            t_zsolve=vqt_now()-tz0;
            if(!zl) printf("③sweep  ds4_z_solve 返回 NULL, 跳过\n");
            else {
                memcpy(Fout,YQ,(size_t)SROW*D*4);
                const int LZRANK=64;
                const int cand[7]={LZRANK,LZRANK/2,16,8,8,4,1};   /* 生产 cand0 同款 */
                printf("③sweep  候选评估(生产 ZLGATE 口径, 真 z rank=64, 信任域 LZTR=0.5):\n");
                land=dsq_sweep_layer(zl,X,Fout,Ftry,HF,pst,cmb,HQ,HF,
                                     SROW,vs,SROW,0.5f,cand,7,&k_land,&e0s,&e1s,&n_eval);
                ds4_z_free(zl);
            }
        }
        free(HF);free(HQ);free(mix);free(pre);free(pst);free(cmb);free(Ftry);free(Fout);
    }
    const double t5=vqt_now();
    if(land>=0){
        printf("③sweep  %6.1fs  评了 %d 个候选 (z解算 %.1fs + hc系数 %.2fs + 候选评估 %.1fs)\n",
               t5-t4, n_eval, t_zsolve, t_hcw, (t5-t4)-t_zsolve-t_hcw);
        if(land==1) printf("         → k=%d 落地  出口分 %.6f→%.6f (降 %.3f%%)\n",
                           k_land,e0s,e1s,(e0s-e1s)/(e0s>1e-30?e0s:1)*100.0);
        else        printf("         → 全拒(%d 个候选无一过门 e1<e0-1e-9), Fout 不动\n", n_eval);
        printf("         ★出口分绝对值不可与战役日志对表(hc 系数非链上真值, 见代码注释)★\n");
    }
    if(erc==0) elm_free(&er);
    free(X);free(RI);free(RW);free(YQ);free(YF);free(DH);free(tok);free(ww);free(xs);free(Wq);
    return 0;
}

int main(int argc, char **argv)
{
    const char *hf=NULL,*anc=NULL; int L=0,NE=8,SROW=0,VDIM=4,VNC=512;
    for(int i=1;i<argc-1;i++){
        if(!strcmp(argv[i],"--hf")) hf=argv[i+1];
        else if(!strcmp(argv[i],"--anchor")) anc=argv[i+1];
        else if(!strcmp(argv[i],"--layer")) L=atoi(argv[i+1]);
        else if(!strcmp(argv[i],"--experts")) NE=atoi(argv[i+1]);
        else if(!strcmp(argv[i],"--rows")) SROW=atoi(argv[i+1]);
        /* ★量化档位可调(2026-08-28)★: 用来验"ELM 收益随底座变差而上升"这个假说 ——
         * 历史 +12% 是在 allq2 烂底座(裸 KLD 1.63)上测的, 我们现在是平权 VQ(0.47), 好 4.3 倍。
         * 同一层同一份 x 只改 nc, 看 held 会不会跟着量化误差一起涨。 */
        else if(!strcmp(argv[i],"--vdim")) VDIM=atoi(argv[i+1]);
        else if(!strcmp(argv[i],"--vnc"))  VNC=atoi(argv[i+1]);
    }
    if(hf&&anc){
#ifndef DS4QUANT_CUDA
        /* ★铁律 2026-08-28「只有 GPU 版本」★ 真层模式产出的是【速度与质量读数】,
         * 走 CPU 参考路会给出自洽但失真 40 倍的数字(实撞: 4.33s/矩阵 vs GPU 0.51s/矩阵),
         * 而且看起来完全像真的。宁可拒跑, 不给假数。 */
        fprintf(stderr,"★拒跑: 真层模式必须 GPU 版★\n"
            "  编译: cc ... -DDS4QUANT_CUDA -I$CUDA_HOME/include dsq_units.c quantize/vq_gpu.o \\\n"
            "        -L$CUDA_HOME/lib64 -lcudart -lcublas -lstdc++ <blas> -lpthread -lm\n"
            "  (vq_gpu.o 是 C++ 目标, 缺 -lstdc++ 会报 __cxa_guard_acquire 未定义)\n");
        return 2;
#endif
        printf("== dsq_units 真模型一层(L%d, v%dx%d) ==\n",L,VDIM,VNC);
        return real_layer(hf,anc,L,NE,SROW,VDIM,VNC);
    }
    printf("== dsq_units 自测(量化/反修/sweep 三单元, 合成数据) ==\n");
    printf("   ★只验数值正确性, 不产任何速度/质量读数★ —— 那些必须走真层 GPU 模式(铁律 08-28)\n");
    int ok = 1;
    ok &= t_quant();
    ok &= t_backfit();
    ok &= t_sweep();
    printf(ok ? "DSQ_UNITS PASS\n" : "DSQ_UNITS FAIL\n");
    return ok ? 0 : 1;
}
#endif
