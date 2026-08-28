/* 反修层前向里的并行核(2026-08-28 从 ds4quant_run_p7 拆出, 守 500 行)。
 * 依赖 g_anc_rowmap/g_anc_rowstride(p6 声明)与 HCM/DIM(p1), 故必须在 p6 之后 include。 */
/* ★"其余"段的两处单线程热点并行化(2026-08-28)★ 实跑逐层账里 g_lt[4]("其余")稳定 28s,
 * 是仅次于 zsolve 的第二大头, 而它里面只有两件事在磨:
 *   ① DF 靶构造: S×DIM×HCM = 8192×4096×4 ≈ 1.34 亿次 double 乘加, 各 s 行完全独立;
 *   ② 出口 relL2: 每评一个候选就要扫 (n_fit-vs)×HCM×DIM ≈ 2500 万个 double, 一层要评 8 次。
 * ①按 s 切 ⇒ 逐位不变(每个输出元素的累加序没动)。
 * ②是个全局求和, 分块后求和顺序变了 ⇒ 不是逐位不变, 但确定(块序固定), 且相对误差 ~1e-15,
 *   而落地判据是 e1<e0-1e-9, 差了六个数量级, 不可能翻转任何一次 ✓/✗。 */
typedef struct { const float *post2,*Hf,*Hq; float *DF; } bfdf_ctx;
static void bfdf_worker(void *vc,int s0,int s1){ bfdf_ctx*c=(bfdf_ctx*)vc;
    const int ancS = g_anc_rowmap ? g_anc_rowstride : 0; (void)ancS;
    for(int s=s0;s<s1;s++){
        const float *ps=c->post2+(size_t)s*HCM; double pd=1e-12;
        for(int j=0;j<HCM;j++) pd+=(double)ps[j]*ps[j];
        const size_t sa = (size_t)(g_anc_rowmap ? g_anc_rowmap[s] : s);
        for(int d=0;d<DIM;d++){ double a2=0;
            for(int j=0;j<HCM;j++) a2+=(double)ps[j]*((double)c->Hf[(sa*HCM+j)*DIM+d]-(double)c->Hq[((size_t)s*HCM+j)*DIM+d]);
            c->DF[(size_t)s*DIM+d]=(float)(a2/pd); }
    } }
typedef struct { const float *Hq,*Hf; double *pe,*pa; int vs,per; } bfrl_ctx;
static void bfrl_worker(void *vc,int t0,int t1){ bfrl_ctx*c=(bfrl_ctx*)vc;
    const int slot=t0/c->per; double e2=0,a2=0;
    for(int t=t0;t<t1;t++){ const int s=c->vs+t;
        const float*hq=c->Hq+(size_t)s*HCM*DIM;
        const float*hf=c->Hf+(size_t)(g_anc_rowmap?g_anc_rowmap[s]:s)*HCM*DIM;
        for(size_t i=0;i<(size_t)HCM*DIM;i++){ double d=(double)hq[i]-hf[i]; e2+=d*d; a2+=(double)hf[i]*hf[i]; } }
    c->pe[slot]=e2; c->pa[slot]=a2; }
static double bf_exit_relL2(const float *Hq, const float *Hf, int vs, int n_fit){
    const int nrow=n_fit-vs; if(nrow<=0) return 0.0;
    int nth=20; if(nth>nrow) nth=nrow;
    const int per=(nrow+nth-1)/nth;
    double pe[32]={0},pa[32]={0};
    bfrl_ctx rc={Hq,Hf,pe,pa,vs,per};
    zpar_for(nrow,nth,bfrl_worker,&rc);
    double e2=0,a2=0; for(int i=0;i<32;i++){ e2+=pe[i]; a2+=pa[i]; }
    return sqrt(e2/(a2+1e-30));
}
