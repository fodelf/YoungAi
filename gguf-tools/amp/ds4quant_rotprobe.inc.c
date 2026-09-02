/* ds4quant_rotprobe.inc.c — B类底座旋转针(2026-09-01 用户令"b类底座能打针看下吗")
 *
 * 问题: 深带薄肉对 x 条件放大器两形态全零(fable5 09-01 终判), 剩余深带杠杆=量化器侧
 * B类底座(QuaRot/QTIP 族: 旋转不相干化+格码, 误差在出生地消灭)。全链重量化是小时级
 * 决策, 先按十分钟铁律打层级针: 同层同锚同生产量化路, 只加一个变量 —— 量化前右乘
 * 随机化 Hadamard 正交阵 Q(把巨值通道能量摊平, 见"巨值通道矿": 平权 VQ 抹零 massive
 * activation 通道 10% 能量, 旋转正是文献里治它的第一刀)。
 *
 * 口径:
 *   W' = W·Q, x' = Qᵀx, Q = D_s·H/√D(D_s=固定 seed ±1 对角, H=Sylvester Hadamard)。
 *   行变换与列变换同式: v → FWHT(s⊙v)/√D(H 对称 ⇒ W 行与 x 用同一个 worker)。
 *   两臂各自走 dq_quant_expert_vq(生产 v4x512 全家: kmeans+GPTQ 反馈+v2 通道权+g_r,
 *   Linux 下 -DDS4QUANT_CUDA = GPU 路), 判据 = held 行输出空间相对误差
 *   ‖(Ŵ−W)·Xhᵀ‖F/‖W·Xhᵀ‖F(能量加权聚合)。旋转不变性保证两臂分母同值(顺带当 sanity)。
 *   范围=w1/w3(巨值通道住残差流=它们的输入空间); w2 输入是 silu 后 h 空间, 本针不测。
 *   针只读不写: 不动层件不动模型, 纯读数(与 --elm-probe 同款)。
 * x 来源=本分支 Fin(部署口径量化链), 与量化战役的锚 FP-x 不同源 —— 针内两臂同 x
 *   单变量成立; 若针见肉, 全量重量化走战役路自然回到锚口径。
 */

static const char *g_rot_probe = NULL;   /* --rot-probe "28,36": 点名层, CLI 不进 env */
static int rot_probe_hit(int L){
    if(!g_rot_probe) return 0;
    char b[256]; snprintf(b,sizeof b,"%s",g_rot_probe);
    for(char *t=strtok(b,","); t; t=strtok(NULL,",")) if(atoi(t)==L) return 1;
    return 0;
}

/* 就地 FWHT(蝶形, n=2 的幂); 调用方负责 1/√n 定标 */
static void rp_fwht(float *v, int n){
    for(int len=1; len<n; len<<=1)
        for(int i=0;i<n;i+=len<<1)
            for(int j=i;j<i+len;j++){ float a=v[j], b=v[j+len]; v[j]=a+b; v[j+len]=a-b; }
}
/* 行并行: row → FWHT(s⊙row)/√D */
typedef struct { float *M; const float *sgn; int D; } rpr_ctx;
static void rpr_worker(void *vc, int r0, int r1){
    rpr_ctx *c=(rpr_ctx*)vc; const float inv=1.0f/sqrtf((float)c->D);
    for(int r=r0;r<r1;r++){ float *row=c->M+(size_t)r*c->D;
        for(int d=0;d<c->D;d++) row[d]*=c->sgn[d];
        rp_fwht(row,c->D);
        for(int d=0;d<c->D;d++) row[d]*=inv; }
}

/* 一臂: 量化 W([rows][D]) 并在 held X 上出能量比误差; 返回 num²/den² 走出参 */
static void rp_arm(const float *W,int rows,int D,const float *Xf,int nf,
                   const float *Xh,int nh,double *num2,double *den2){
    float *Wq = dq_quant_expert_vq(W,rows,D,Xf,nf,4,512);   /* 生产档 v4x512(v2 底座同款) */
    float *Dg = malloc((size_t)rows*D*4);
    for(size_t i=0;i<(size_t)rows*D;i++) Dg[i]=Wq[i]-W[i];
    float *E = malloc((size_t)nh*rows*4);
    dq_matmul(Xh,Dg,E,nh,D,rows);                  /* E = Xh·(Ŵ−W)ᵀ */
    double n2=0; for(size_t i=0;i<(size_t)nh*rows;i++) n2+=(double)E[i]*E[i];
    dq_matmul(Xh,W,E,nh,D,rows);                   /* 基线能量 Xh·Wᵀ */
    double d2=0; for(size_t i=0;i<(size_t)nh*rows;i++) d2+=(double)E[i]*E[i];
    *num2=n2; *den2=d2;
    free(Wq); free(Dg); free(E);
}

static void rot_probe_layer(int L,const float *Fin,int S,int n_fit,const int *idx){
    enum { NE = 8 };                               /* 采样专家数(容量上限, 显式记录不静默) */
    double t0 = vqt_now();
    /* ①固定 seed ±1 符号 + 整层旋转副本(一次, 专家共享) */
    float *sgn = malloc((size_t)DIM*4);
    { uint64_t sd=0xB57A11C0FFEEULL; for(int d=0;d<DIM;d++) sgn[d]=lcg_unit(&sd)>=0.f?1.f:-1.f; }
    float *FinR = malloc((size_t)S*DIM*4);
    memcpy(FinR,Fin,(size_t)S*DIM*4);
    { rpr_ctx rc={FinR,sgn,DIM}; zpar_for(S,64,rpr_worker,&rc); }
    /* ②巨值通道统计(fit 域列能量 max/mean, 旋转前后) —— 机理确认线 */
    double mc0=0,mm0=0,mc1=0,mm1=0;
    for(int d=0;d<DIM;d++){ double a=0,b=0;
        for(int s=0;s<n_fit;s++){ double v=Fin[(size_t)s*DIM+d]; a+=v*v;
                                  v=FinR[(size_t)s*DIM+d]; b+=v*v; }
        if(a>mc0)mc0=a; mm0+=a; if(b>mc1)mc1=b; mm1+=b; }
    mm0/=DIM; mm1/=DIM;
    /* ③专家使用计数 → top-NE */
    int *cnt = calloc((size_t)NEXP,sizeof(int));
    for(int s=0;s<S;s++)for(int a=0;a<NACT;a++) cnt[idx[(size_t)s*NACT+a]]++;
    int pick[NE]; for(int p=0;p<NE;p++){ int be=-1,bc=-1;
        for(int e=0;e<NEXP;e++){ int used=0; for(int q=0;q<p;q++) if(pick[q]==e)used=1;
            if(!used&&cnt[e]>bc){bc=cnt[e];be=e;} } pick[p]=be; }
    /* ④逐专家 w1/w3 双臂 */
    double a0n=0,a0d=0,a1n=0,a1d=0; int nmat=0;
    float *Xf=malloc((size_t)n_fit*DIM*4), *Xh=malloc((size_t)(S-n_fit)*DIM*4);
    float *XfR=malloc((size_t)n_fit*DIM*4), *XhR=malloc((size_t)(S-n_fit)*DIM*4);
    for(int p=0;p<NE;p++){ const int e=pick[p]; if(e<0) continue;
        int nf=0,nh=0;
        for(int s=0;s<S;s++)for(int a=0;a<NACT;a++) if(idx[(size_t)s*NACT+a]==e){
            if(s<n_fit){ memcpy(Xf +(size_t)nf*DIM,Fin +(size_t)s*DIM,(size_t)DIM*4);
                         memcpy(XfR+(size_t)nf*DIM,FinR+(size_t)s*DIM,(size_t)DIM*4); nf++; }
            else       { memcpy(Xh +(size_t)nh*DIM,Fin +(size_t)s*DIM,(size_t)DIM*4);
                         memcpy(XhR+(size_t)nh*DIM,FinR+(size_t)s*DIM,(size_t)DIM*4); nh++; }
            break; }
        if(nf<64||nh<32){ printf("  ROT L%d e%d 行不足(nf=%d nh=%d) 跳过\n",L,e,nf,nh); continue; }
        for(int m=0;m<2;m++){
            char nm[160]; snprintf(nm,sizeof nm,"layers.%d.ffn.experts.%d.w%d.weight",L,e,m?3:1);
            long rr,cc; float *W = st_read_weight(&C,nm,&rr,&cc);
            if(!W){ printf("  ROT L%d %s 读失败\n",L,nm); continue; }
            double n2,d2; rp_arm(W,(int)rr,DIM,Xf,nf,Xh,nh,&n2,&d2);
            a0n+=n2; a0d+=d2;
            float *Wr = malloc((size_t)rr*DIM*4);
            memcpy(Wr,W,(size_t)rr*DIM*4);
            { rpr_ctx rc={Wr,sgn,DIM}; zpar_for((int)rr,64,rpr_worker,&rc); }
            rp_arm(Wr,(int)rr,DIM,XfR,nf,XhR,nh,&n2,&d2);
            a1n+=n2; a1d+=d2; nmat++;
            free(Wr); free(W);
        }
    }
    const double e0=sqrt(a0n/(a0d+1e-30)), e1=sqrt(a1n/(a1d+1e-30));
    printf("★ROT L%02d held输出误差 基线 %.4f → 旋转 %.4f (Δ %+.2f%%) | 矩阵 %d(top%d 专家×w1w3, w2未测) "
           "| 巨值通道 max/mean %.1f → %.1f | 分母比 %.4f(sanity≈1) | %.0fs\n",
           L, e0, e1, (e1/(e0+1e-30)-1.0)*100.0, nmat, NE,
           mc0/(mm0+1e-30), mc1/(mm1+1e-30), a1d/(a0d+1e-30), vqt_now()-t0);
    fflush(stdout);
    free(sgn);free(FinR);free(cnt);free(Xf);free(Xh);free(XfR);free(XhR);
}
