/* ds4quant_qhelp.h — go1b 字节 dequant + 单专家 joint-LS 量化(供 layer/run 复用). */
#ifndef DS4QUANT_QHELP_H
#define DS4QUANT_QHELP_H
static void dq_go1b_bytes_dequant(const uint8_t *b, int nrows, int ncols, float *out) {
    int nblk = ncols / GO1B_BLK_QK;
    for (int r=0;r<nrows;r++) {
        const uint8_t *rb = b + (size_t)r*nblk*GO1B_BLK_BYTES;
        for (int bl=0;bl<nblk;bl++) {
            const uint8_t *bd = rb + (size_t)bl*GO1B_BLK_BYTES;
            uint16_t h; memcpy(&h,bd,2);
            uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,f;
            if(e==0){ if(m==0)f=s<<31; else {e=127-15+1; while(!(m&0x400)){m<<=1;e--;} m&=0x3ff; f=(s<<31)|(e<<23)|(m<<13);} }
            else if(e==0x1f) f=(s<<31)|(0xffu<<23)|(m<<13);
            else f=(s<<31)|((e-15+127)<<23)|(m<<13);
            float scale; memcpy(&scale,&f,4);
            const uint8_t *sg=bd+2;
            for(int k=0;k<GO1B_BLK_QK;k++){ int j=bl*GO1B_BLK_QK+k; if(j>=ncols)break;
                int bit=(sg[k/8]>>(k%8))&1; out[(size_t)r*ncols+j]=bit?scale:-scale; }
        }
    }
}
static float *dq_quant_expert(const float *W, int nrows, int ncols, const float *X, int n_act) {
    int nblk=ncols/GO1B_BLK_QK; size_t nb=(size_t)nrows*nblk*GO1B_BLK_BYTES;
    uint8_t *bytes=malloc(nb); go1b_blk_quantize_joint(W,bytes,nrows,ncols,X,n_act);
    float *wq=malloc((size_t)nrows*ncols*sizeof(float)); dq_go1b_bytes_dequant(bytes,nrows,ncols,wq);
    free(bytes); return wq;
}
/* 列正交化 (modified Gram-Schmidt), Q[d×k] 就地. */
static void dq_mgs_qh(float *Q, int d, int k){
    for(int c=0;c<k;c++){
        for(int p=0;p<c;p++){ double dp=0; for(int i=0;i<d;i++) dp+=(double)Q[(size_t)i*k+c]*Q[(size_t)i*k+p];
            for(int i=0;i<d;i++) Q[(size_t)i*k+c]-=(float)dp*Q[(size_t)i*k+p]; }
        double nr=0; for(int i=0;i<d;i++) nr+=(double)Q[(size_t)i*k+c]*Q[(size_t)i*k+c]; nr=sqrt(nr)+1e-20;
        for(int i=0;i<d;i++) Q[(size_t)i*k+c]=(float)(Q[(size_t)i*k+c]/nr); }
}
/* rank-k SVD 重构 out=V(V^T E) (E[nr×nc] 的秩-k投影), 子空间迭代. 自包含. */
static void dq_lowrank_recon(const float *E, int nr, int nc, int rank, float *out){
    float *V=malloc((size_t)nr*rank*4),*T=malloc((size_t)nc*rank*4),*M=malloc((size_t)rank*nc*4);
    uint64_t s=0x2545F4914F6CDD1DULL;
    for(size_t i=0;i<(size_t)nr*rank;i++){ s=s*6364136223846793005ULL+1442695040888963407ULL; V[i]=((float)((s>>40)&0xFFFFFF)/16777216.0f)-0.5f; }
    dq_mgs_qh(V,nr,rank);
    for(int it=0;it<5;it++){
        memset(T,0,(size_t)nc*rank*4);
        for(int i=0;i<nr;i++){ const float*Ei=E+(size_t)i*nc,*Vi=V+(size_t)i*rank;
            for(int j=0;j<nc;j++){ float e=Ei[j]; if(e==0)continue; float*Tj=T+(size_t)j*rank; for(int c=0;c<rank;c++) Tj[c]+=e*Vi[c]; } }
        for(int i=0;i<nr;i++){ const float*Ei=E+(size_t)i*nc; float*Vi=V+(size_t)i*rank; for(int c=0;c<rank;c++)Vi[c]=0;
            for(int j=0;j<nc;j++){ float e=Ei[j]; if(e==0)continue; const float*Tj=T+(size_t)j*rank; for(int c=0;c<rank;c++) Vi[c]+=e*Tj[c]; } }
        dq_mgs_qh(V,nr,rank);
    }
    memset(M,0,(size_t)rank*nc*4);
    for(int i=0;i<nr;i++){ const float*Ei=E+(size_t)i*nc,*Vi=V+(size_t)i*rank;
        for(int c=0;c<rank;c++){ float v=Vi[c]; if(v==0)continue; float*Mc=M+(size_t)c*nc; for(int j=0;j<nc;j++) Mc[j]+=v*Ei[j]; } }
    for(int i=0;i<nr;i++){ const float*Vi=V+(size_t)i*rank; float*Oi=out+(size_t)i*nc;
        for(int j=0;j<nc;j++){ double sm=0; for(int c=0;c<rank;c++) sm+=(double)Vi[c]*M[(size_t)c*nc+j]; Oi[j]=(float)sm; } }
    free(V);free(T);free(M);
}
/* weight-space 低秩 z(极小体积): Q1(W) + rank-k SVD(W-Q1). 只存 rank-k 因子(~1%体积), 可调秩. */
static float *dq_quant_expert_lowrank(const float *W, int nrows, int ncols, const float *X, int n_act, int rank){
    size_t N=(size_t)nrows*ncols;
    float *acc=dq_quant_expert(W,nrows,ncols,X,n_act);   /* Q1 */
    float *E=malloc(N*sizeof(float)); for(size_t i=0;i<N;i++) E[i]=W[i]-acc[i];
    float *rec=malloc(N*sizeof(float)); dq_lowrank_recon(E,nrows,ncols,rank,rec);
    for(size_t i=0;i<N;i++) acc[i]+=rec[i];
    free(E);free(rec); return acc;
}
/* per-row 单 scale 1-bit (真 1.004~1.008 bit/el, 纯1bit体积主线): ridge-to-prior 的
 * nblk=1 闭式特例, 无方程组. s* = (Σ_t p_t·y_t + λ·s0) / (Σ_t p_t² + λ),
 * p_t=Σ_j sign(w_j)·x_tj, y_t=w·x_t; λ=Σp²/(n_act+1) (下限 1e-3·Σp², 与 joint 同口径);
 * 负→回 s0, >4·s0→clip. X=NULL → s0=mean|w|. scale 过 fp16 圆整(与块格式公平). */
static float *dq_quant_expert_rowscale(const float *W,int nrows,int ncols,const float *X,int n_act){
    float *wq=malloc((size_t)nrows*ncols*sizeof(float));
    for(int r=0;r<nrows;r++){
        const float *w=W+(size_t)r*ncols;
        double s0=0; for(int j=0;j<ncols;j++) s0 += w[j]<0?-(double)w[j]:(double)w[j];
        s0/=(double)ncols;
        double s=s0;
        if(X&&n_act>=1){
            double sp2=0,spy=0;
            for(int t=0;t<n_act;t++){ const float *x=X+(size_t)t*ncols; double p=0,y=0;
                for(int j=0;j<ncols;j++){ double xj=x[j]; p += (w[j]>=0.0f)?xj:-xj; y += (double)w[j]*xj; }
                sp2+=p*p; spy+=p*y; }
            double lam=sp2/((double)n_act+1.0), lam0=1e-3*sp2+1e-9; if(lam<lam0)lam=lam0;
            s=(spy+lam*s0)/(sp2+lam);
            if(s<0.0){ s=s0; __sync_fetch_and_add(&go1b_joint_neg,1); }
            else if(s>4.0*s0+1e-12){ s=4.0*s0; __sync_fetch_and_add(&go1b_joint_clip,1); }
            __sync_fetch_and_add(&go1b_joint_rows,1);
        }
        float sf=go1b_fp16_to_fp32(go1b_fp32_to_fp16((float)s));
        float *o=wq+(size_t)r*ncols;
        for(int j=0;j<ncols;j++) o[j]=(w[j]>=0.0f)?sf:-sf;
    }
    return wq;
}
/* 符号精修 1-bit ('g'档, 零体积=1.0052b): 行scale 基础上做符号坐标下降(GPTQ 思想的
 * 1-bit 化): 符号不再钉死 sign(w), 允许翻转以最小化【输出误差 Σ_t(y_t−s·P_t)² +
 * μ·权重锚 Σ_j(w_j−q_j)²】。翻转 j 的增量闭式: ΔE_out = 4s·σ_j·Σ_t e_t x_tj + 4s²Σ_t x_tj²,
 * ΔE_w = 4s·σ_j·w_j·μ; ΔE<0 才翻。每轮后重解 s(rowscale 闭式)。X=NULL → 退化 rowscale。
 * μ 与 ridge-to-prior 同哲学: 样本少 → 权重锚压住输出项过拟合。 */
static double dq_signref_mu=1.0;   /* 权重锚强度(渐进调优逐层可变, main 从 env 初始化) */
static int    dq_signref_rounds=3;
/* _adj 核心: Yadj[n_act×nrows] 非 NULL 时, 行 r 在校准行 t 的目标从 w_r·x_t 变为
 * w_r·x_t + Yadj[t*nrows+r] —— 共适应(D3)用: base 的重解目标 = z 校正后残差
 * (Yadj = −该专家按路由权归因到的 z 份额)。μ 权重锚仍对原 FP w(先验不动)。
 * nflip 非 NULL 时累加最终符号 vs sign(w) 的翻转数(观测 base 是否真的在动)。 */
/* B0b(2026-07-25 G1a 判决落地): 符号固定后 per-256-block scale 联合 ridge LS。
 * go1b 块 f16 槽位本就逐块存在(生产=行值复制进每块=浪费); 两层探针 −6.9%/−12.5% relF。
 * DS4_SIGNREF_BLK=1 启用(默认关=现基线)。解 (PᵀP+λI)s = Pᵀy+λs0, clamp[0,4·s0_b],
 * p_b(t)=Σ_{j∈块b} sg_j·x_tj, y(t)=w·x_t(+Yadj)。sblk 输出未经 f16 往返。 */
static int dq_signref_blk_on(void){ static int v=-1; if(v<0) v=getenv("DS4_SIGNREF_BLK")?1:0; return v; }
/* DS4_TGT_ALPHA∈[0,1]: GPTAQ 非对称目标强度(0=现基线只条件于漂移, 1=全额纠正累积漂移) */
static float dq_tgt_alpha(void){ static float v=-1.0f;
    if(v<0.0f){ const char*e2=getenv("DS4_TGT_ALPHA"); double d=e2?atof(e2):0.0;
        if(d<0.0)d=0.0; if(d>1.0)d=1.0; v=(float)d; } return v; }
static void dq_blk_scales_solve(const float *w,const float *X,int n_act,int ncols,
                                const signed char *sg,const float *Yadj,int r,int nrows,
                                double *sblk,int nblk){
    double A[16*16],rhs[16],s0b[16],p[16];
    if(nblk>16){ nblk=16; }
    for(int i=0;i<nblk*nblk;i++)A[i]=0.0;
    for(int b=0;b<nblk;b++){ rhs[b]=0.0; double m=0; int j0=b*GO1B_BLK_QK, j1=j0+GO1B_BLK_QK;
        if(j1>ncols)j1=ncols;
        for(int j=j0;j<j1;j++) m+= w[j]<0?-(double)w[j]:(double)w[j];
        s0b[b]=m/(double)(j1-j0>0?j1-j0:1); }
    for(int t=0;t<n_act;t++){
        const float *x=X+(size_t)t*ncols; double y=0;
        for(int b=0;b<nblk;b++){ double pb=0; int j0=b*GO1B_BLK_QK, j1=j0+GO1B_BLK_QK; if(j1>ncols)j1=ncols;
            for(int j=j0;j<j1;j++){ double xj=x[j]; pb+= sg[j]>0?xj:-xj; y+=(double)w[j]*xj; }
            p[b]=pb; }
        if(Yadj) y+=(double)Yadj[(size_t)t*nrows+r];
        for(int b=0;b<nblk;b++){ rhs[b]+=p[b]*y;
            for(int c=b;c<nblk;c++) A[b*nblk+c]+=p[b]*p[c]; }
    }
    for(int b=0;b<nblk;b++)for(int c=0;c<b;c++) A[b*nblk+c]=A[c*nblk+b];
    double tr=0; for(int b=0;b<nblk;b++) tr+=A[b*nblk+b];
    double lam=tr/((double)n_act+1.0)/(double)nblk; double lam0=1e-9+1e-4*tr/(double)nblk; if(lam<lam0)lam=lam0;
    for(int b=0;b<nblk;b++){ A[b*nblk+b]+=lam; rhs[b]+=lam*s0b[b]; }
    /* Gauss-Jordan 消元(nblk≤16) */
    for(int c2=0;c2<nblk;c2++){
        int piv=c2; for(int rr2=c2+1;rr2<nblk;rr2++) if((A[rr2*nblk+c2]>0?A[rr2*nblk+c2]:-A[rr2*nblk+c2])>(A[piv*nblk+c2]>0?A[piv*nblk+c2]:-A[piv*nblk+c2])) piv=rr2;
        if(piv!=c2){ for(int k=0;k<nblk;k++){ double tt=A[c2*nblk+k];A[c2*nblk+k]=A[piv*nblk+k];A[piv*nblk+k]=tt; } double tt=rhs[c2];rhs[c2]=rhs[piv];rhs[piv]=tt; }
        double d=A[c2*nblk+c2]; if(d==0.0){ sblk[c2]=s0b[c2]; continue; }
        for(int rr2=0;rr2<nblk;rr2++){ if(rr2==c2)continue; double f=A[rr2*nblk+c2]/d;
            if(f!=0.0){ for(int k=c2;k<nblk;k++) A[rr2*nblk+k]-=f*A[c2*nblk+k]; rhs[rr2]-=f*rhs[c2]; } }
    }
    for(int b=0;b<nblk;b++){
        double d=A[b*nblk+b]; double s= d!=0.0? rhs[b]/d : s0b[b];
        if(s<0.0)s=s0b[b]; else if(s>4.0*s0b[b]+1e-12)s=4.0*s0b[b]+1e-12;
        sblk[b]=s;
    }
}
static float *dq_quant_expert_signref_adj(const float *W,int nrows,int ncols,const float *X,int n_act,
                                          const float *Yadj,long *nflip){
    float *wq=malloc((size_t)nrows*ncols*sizeof(float));
    if(!X||n_act<1){ free(wq); return dq_quant_expert_rowscale(W,nrows,ncols,NULL,0); }
    double MU_SCALE=dq_signref_mu;
    int ROUNDS=dq_signref_rounds;
    /* DS4_SIGNREF_NACT_CAP: 量化路同款行子采样(第四处同型实现, 2026-07-27 profile 复采
     * 实锤 cap 后仍 5449 采样在此) — 与 export 路完全同语义。 */
    float *Xsub2=NULL,*Yadjsub2=NULL;
    {   static int cap2=-2; if(cap2==-2){ const char*c=getenv("DS4_SIGNREF_NACT_CAP"); cap2=c?atoi(c):128; }
        if(cap2>0&&n_act>cap2){
            int stride=n_act/cap2;
            Xsub2=malloc((size_t)cap2*ncols*sizeof(float));
            if(Yadj) Yadjsub2=malloc((size_t)cap2*nrows*sizeof(float));
            for(int t=0;t<cap2;t++){
                memcpy(Xsub2+(size_t)t*ncols, X+(size_t)(t*stride)*ncols, (size_t)ncols*sizeof(float));
                if(Yadj) memcpy(Yadjsub2+(size_t)t*nrows, Yadj+(size_t)(t*stride)*nrows, (size_t)nrows*sizeof(float));
            }
            X=Xsub2; if(Yadj) Yadj=Yadjsub2; n_act=cap2;
        }
    }
    /* 列能量 cx[j]=Σ_t x_tj² 与 μ 基准(全矩阵一次) */
    double *cx=malloc((size_t)ncols*sizeof(double));
    for(int j=0;j<ncols;j++)cx[j]=0.0;
    double cxm=0;
    for(int t=0;t<n_act;t++){ const float*x=X+(size_t)t*ncols;
        for(int j=0;j<ncols;j++){ double v=x[j]; cx[j]+=v*v; } }
    for(int j=0;j<ncols;j++) cxm+=cx[j]; cxm/=(double)ncols;
    double mu=MU_SCALE*cxm*( (double)1.0 );   /* 锚强度: 列均能量 × MU_SCALE(默认1) */
    /* X 转置一份(Xt[j][t] 连续): 符号列循环从 16KB-stride 访存变顺序, ~5-10× */
    float *Xt=malloc((size_t)ncols*n_act*sizeof(float));
    for(int t=0;t<n_act;t++) for(int j=0;j<ncols;j++) Xt[(size_t)j*n_act+t]=X[(size_t)t*ncols+j];
    signed char *sg=malloc((size_t)ncols);
    double *e=malloc((size_t)n_act*sizeof(double)),*pp=malloc((size_t)n_act*sizeof(double));
    for(int r=0;r<nrows;r++){
        const float *w=W+(size_t)r*ncols;
        double s0=0; for(int j=0;j<ncols;j++) s0+= w[j]<0?-(double)w[j]:(double)w[j];
        s0/=(double)ncols;
        for(int j=0;j<ncols;j++) sg[j]= w[j]>=0.0f?1:-1;
        /* 初始 s: rowscale 闭式(ridge-to-prior nblk=1) */
        double s=s0;
        for(int round=0;round<=ROUNDS;round++){
            /* 解 s (符号固定): s=(Σp_t y_t+λs0)/(Σp_t²+λ); y 含 Yadj 目标平移 */
            double sp2=0,spy=0;
            for(int t=0;t<n_act;t++){ const float *x=X+(size_t)t*ncols; double p=0,y=0;
                for(int j=0;j<ncols;j++){ double xj=x[j]; p+=sg[j]>0?xj:-xj; y+=(double)w[j]*xj; }
                if(Yadj) y+=(double)Yadj[(size_t)t*nrows+r];
                sp2+=p*p; spy+=p*y; e[t]=y; pp[t]=p;
            }
            double lam=sp2/((double)n_act+1.0), lam0=1e-3*sp2+1e-9; if(lam<lam0)lam=lam0;
            s=(spy+lam*s0)/(sp2+lam);
            if(s<0.0)s=s0; else if(s>4.0*s0+1e-12)s=4.0*s0;
            for(int t=0;t<n_act;t++) e[t]-= s*pp[t];
            if(round==ROUNDS) break;
            /* 符号坐标下降一轮 */
            for(int j=0;j<ncols;j++){
                const float *xj=Xt+(size_t)j*n_act;
                double g1=0; for(int t=0;t<n_act;t++) g1+=e[t]*(double)xj[t];
                double sj=(double)sg[j];
                double dE = 4.0*s*sj*g1 + 4.0*s*s*cx[j] + mu*4.0*s*sj*(double)w[j];
                if(dE<0.0){ for(int t=0;t<n_act;t++) e[t]+= 2.0*s*sj*(double)xj[t];
                            sg[j]=(signed char)(-sg[j]); }
            }
        }
        float *o=wq+(size_t)r*ncols;
        if(dq_signref_blk_on()){
            int nblk2=(ncols+GO1B_BLK_QK-1)/GO1B_BLK_QK; double sb[16];
            dq_blk_scales_solve(w,X,n_act,ncols,sg,Yadj,r,nrows,sb,nblk2);
            for(int b=0;b<nblk2;b++){ float sf2=go1b_fp16_to_fp32(go1b_fp32_to_fp16((float)sb[b]));
                int j0=b*GO1B_BLK_QK,j1=j0+GO1B_BLK_QK; if(j1>ncols)j1=ncols;
                for(int j=j0;j<j1;j++) o[j]= sg[j]>0? sf2:-sf2; }
        } else {
            float sf=go1b_fp16_to_fp32(go1b_fp32_to_fp16((float)s));
            for(int j=0;j<ncols;j++) o[j]= sg[j]>0? sf:-sf;
        }
        if(nflip){ long f=0; for(int j=0;j<ncols;j++) if((sg[j]>0)!=(w[j]>=0.0f)) f++; *nflip+=f; }
    }
    free(cx);free(sg);free(e);free(pp);free(Xt);
    if(Xsub2)free(Xsub2); if(Yadjsub2)free(Yadjsub2);
    return wq;
}
static float *dq_quant_expert_signref(const float *W,int nrows,int ncols,const float *X,int n_act){
    return dq_quant_expert_signref_adj(W,nrows,ncols,X,n_act,NULL,NULL);
}
/* signref 的 GO1B_BLK 导出版(与 dq_quant_expert_signref 同解, 修改需同步):
 * 行scale+符号精修 → 34B/256el 块格式 bytes(行 scale fp16 复制进每块), 直接可写 GGUF。
 * out 需 nrows*go1b_blk_row_bytes(ncols) 字节。wq_opt 非 NULL 时同时输出 dequant float
 * (nrows×ncols, 供 w2 顺序补偿校准链复用, 免二次求解)。 */
static void dq_signref_export_adj(const float *W,int nrows,int ncols,const float *X,int n_act,
                              uint8_t *out,float *wq_opt,const float *Yadj);
static void dq_signref_export(const float *W,int nrows,int ncols,const float *X,int n_act,
                              uint8_t *out,float *wq_opt){ dq_signref_export_adj(W,nrows,ncols,X,n_act,out,wq_opt,NULL); }
static void dq_signref_export_adj(const float *W,int nrows,int ncols,const float *X,int n_act,
                              uint8_t *out,float *wq_opt,const float *Yadj){
    size_t rb=go1b_blk_row_bytes(ncols);
    int nblk=(ncols+GO1B_BLK_QK-1)/GO1B_BLK_QK;
    double MU_SCALE=dq_signref_mu; int ROUNDS=dq_signref_rounds;
    /* DS4_SIGNREF_NACT_CAP(2026-07-27 提速, 默认128): signref 求的是 256-block 标量
     * scale+符号翻转, ~128 行统计已饱和; fullset 把 n_act 抬到 612 后本函数的手写
     * 标量循环(∝n_act×ncols×rows×rounds)吃掉 ~70% 层时(sample 实测 14030/19k)。
     * 均匀 stride 子采样, X/Yadj 同步; 0=不 cap。 */
    float *Xsub=NULL,*Yadjsub=NULL;
    {   static int cap=-2; if(cap==-2){ const char*c=getenv("DS4_SIGNREF_NACT_CAP"); cap=c?atoi(c):128; }
        if(cap>0&&X&&n_act>cap){
            int stride=n_act/cap;
            Xsub=malloc((size_t)cap*ncols*sizeof(float));
            if(Yadj) Yadjsub=malloc((size_t)cap*nrows*sizeof(float));
            for(int t=0;t<cap;t++){
                memcpy(Xsub+(size_t)t*ncols, X+(size_t)(t*stride)*ncols, (size_t)ncols*sizeof(float));
                if(Yadj) memcpy(Yadjsub+(size_t)t*nrows, Yadj+(size_t)(t*stride)*nrows, (size_t)nrows*sizeof(float));
            }
            X=Xsub; if(Yadj) Yadj=Yadjsub; n_act=cap;
        }
    }
    float *Xt=NULL; double *cx=NULL,*e=NULL,*pp=NULL; double cxm=0,mu=0;
    if(X&&n_act>=1){
        Xt=malloc((size_t)ncols*n_act*sizeof(float));
        for(int t=0;t<n_act;t++) for(int j=0;j<ncols;j++) Xt[(size_t)j*n_act+t]=X[(size_t)t*ncols+j];
        cx=malloc((size_t)ncols*sizeof(double));
        for(int j=0;j<ncols;j++)cx[j]=0.0;
        for(int t=0;t<n_act;t++){ const float*x=X+(size_t)t*ncols;
            for(int j=0;j<ncols;j++){ double v=x[j]; cx[j]+=v*v; } }
        for(int j=0;j<ncols;j++) cxm+=cx[j]; cxm/=(double)ncols;
        mu=MU_SCALE*cxm;
        e=malloc((size_t)n_act*sizeof(double)); pp=malloc((size_t)n_act*sizeof(double));
    }
    signed char *sg=malloc((size_t)ncols);
    for(int r=0;r<nrows;r++){
        const float *w=W+(size_t)r*ncols;
        double s0=0; for(int j=0;j<ncols;j++) s0+= w[j]<0?-(double)w[j]:(double)w[j];
        s0/=(double)ncols;
        for(int j=0;j<ncols;j++) sg[j]= w[j]>=0.0f?1:-1;
        double s=s0;
        if(X&&n_act>=1){
            /* ★热循环向量化(2026-08-03 用户令"暂停找到问题": 全 signref 配方把本函数推成
             * 90% 热点, 18分/层 vs 冠军10分)★ 三处修复, 语义保持(同 double 累加域, 仅求和
             * 顺序 ε 差):
             *   ① y0[t]=w·x_t 与符号无关却每轮重算 → 提出轮外一次(cblas_dsdot: float 入
             *      double 累加, 与原手写同精度域);
             *   ② p_t=Σ_j sg_j·x_tj → 维护 sgf(±1 float) 向量, dsdot 批量;
             *   ③ g1_j=Σ_t e_t·xj_t → e 的 float 镜像 ef 与 Xt 列 dsdot(e 本体仍 double
             *      序贯更新, ef 同步; g1 误差 ~1e-7·|e| 远小于翻转判据量级)。 */
            float *sgf=malloc((size_t)ncols*sizeof(float));
            float *ef=malloc((size_t)n_act*sizeof(float));
            float *g1v=malloc((size_t)ncols*sizeof(float));
            double *y0=malloc((size_t)n_act*sizeof(double));
            for(int j=0;j<ncols;j++) sgf[j]= sg[j]>0?1.0f:-1.0f;
            for(int t=0;t<n_act;t++){
                y0[t]=cblas_dsdot(ncols,w,1,X+(size_t)t*ncols,1);
                if(Yadj) y0[t]+=(double)Yadj[(size_t)t*nrows+r];
            }
            for(int round=0;round<=ROUNDS;round++){
                double sp2=0,spy=0;
                for(int t=0;t<n_act;t++){
                    double p=cblas_dsdot(ncols,sgf,1,X+(size_t)t*ncols,1);
                    sp2+=p*p; spy+=p*y0[t]; e[t]=y0[t]; pp[t]=p;
                }
                double lam=sp2/((double)n_act+1.0), lam0=1e-3*sp2+1e-9; if(lam<lam0)lam=lam0;
                s=(spy+lam*s0)/(sp2+lam);
                if(s<0.0)s=s0; else if(s>4.0*s0+1e-12)s=4.0*s0;
                for(int t=0;t<n_act;t++){ e[t]-= s*pp[t]; ef[t]=(float)e[t]; }
                if(round==ROUNDS) break;
                /* ★g1 批量化(第2版, 2026-08-03)★: 第1版逐 j 短 dsdot(128 长)被调用开销吃平
                 * (840 万次/矩阵)。改一次 sgemv 算全部 g1[ncols] — Gauss-Seidel→Jacobi:
                 * 判据用轮开头的 e(翻转的 e 更新保留, 影响 s 重估与下一轮), 大部分 dE 远离 0,
                 * 边界位次轮自纠; L00 held 对拍 ±0.5% 是语义判决门。 */
                cblas_sgemv(CblasRowMajor,CblasNoTrans,ncols,n_act,1.0f,Xt,n_act,ef,1,0.0f,g1v,1);
                for(int j=0;j<ncols;j++){
                    const float *xj=Xt+(size_t)j*n_act;
                    double g1=(double)g1v[j];
                    double sj=(double)sg[j];
                    double dE = 4.0*s*sj*g1 + 4.0*s*s*cx[j] + mu*4.0*s*sj*(double)w[j];
                    if(dE<0.0){ double d2=2.0*s*sj;
                                for(int t=0;t<n_act;t++){ e[t]+= d2*(double)xj[t]; ef[t]=(float)e[t]; }
                                sg[j]=(signed char)(-sg[j]); sgf[j]=-sgf[j]; }
                }
            }
            free(sgf); free(ef); free(g1v); free(y0);
        }
        uint16_t shrow=go1b_fp32_to_fp16((float)s);
        double sbx[16]; int blkon=dq_signref_blk_on();
        if(blkon&&X&&n_act>=1) dq_blk_scales_solve(w,X,n_act,ncols,sg,Yadj,r,nrows,sbx,nblk);
        uint8_t *rd=out+(size_t)r*rb;
        for(int b=0;b<nblk;b++){
            uint8_t *bd=rd+(size_t)b*GO1B_BLK_BYTES;
            uint16_t sh=(blkon&&X&&n_act>=1)?go1b_fp32_to_fp16((float)sbx[b]):shrow;
            go1b_store_u16_le(bd,sh);
            uint8_t *sgb=bd+2;
            for(int k=0;k<GO1B_BLK_QK/8;k++){
                uint8_t byte=0;
                for(int bit=0;bit<8;bit++){ int j=b*GO1B_BLK_QK+k*8+bit;
                    if(j<ncols&&sg[j]>0) byte|=(uint8_t)(1u<<bit); }
                sgb[k]=byte;
            }
            if(wq_opt){ float sf=go1b_fp16_to_fp32(sh); float *o=wq_opt+(size_t)r*ncols;
                int j0=b*GO1B_BLK_QK,j1=j0+GO1B_BLK_QK; if(j1>ncols)j1=ncols;
                for(int j=j0;j<j1;j++) o[j]= sg[j]>0? sf:-sf; }
        }
    }
    free(sg); if(Xt)free(Xt); if(cx)free(cx); if(e)free(e); if(pp)free(pp);
    if(Xsub)free(Xsub); if(Yadjsub)free(Yadjsub);
}
/* weight-space z: 1-bit Q1(W) + Q1(残差 W-Q1). 权重空间确定性修正(对所有输入生效,
 * 无泛化 gap). nlvl=残差再量化遍数(1=Q1+Q2≈2bit, 2=Q1+Q2+Q3). 返回 dequant 求和权重. */
static float *dq_quant_expert_resid(const float *W, int nrows, int ncols, const float *X, int n_act, int nlvl) {
    size_t N=(size_t)nrows*ncols;
    float *acc=dq_quant_expert(W,nrows,ncols,X,n_act);      /* Q1 */
    float *res=malloc(N*sizeof(float));
    for(int lv=0; lv<nlvl; lv++){
        for(size_t i=0;i<N;i++) res[i]=W[i]-acc[i];         /* 当前残差 */
        float *qr=dq_quant_expert(res,nrows,ncols,X,n_act); /* 量化残差 */
        for(size_t i=0;i<N;i++) acc[i]+=qr[i]; free(qr);    /* 加回 */
    }
    free(res); return acc;
}
#endif
