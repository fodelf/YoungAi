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
static float *dq_quant_expert_signref_adj(const float *W,int nrows,int ncols,const float *X,int n_act,
                                          const float *Yadj,long *nflip){
    float *wq=malloc((size_t)nrows*ncols*sizeof(float));
    if(!X||n_act<1){ free(wq); return dq_quant_expert_rowscale(W,nrows,ncols,NULL,0); }
    double MU_SCALE=dq_signref_mu;
    int ROUNDS=dq_signref_rounds;
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
        float sf=go1b_fp16_to_fp32(go1b_fp32_to_fp16((float)s));
        float *o=wq+(size_t)r*ncols;
        for(int j=0;j<ncols;j++) o[j]= sg[j]>0? sf:-sf;
        if(nflip){ long f=0; for(int j=0;j<ncols;j++) if((sg[j]>0)!=(w[j]>=0.0f)) f++; *nflip+=f; }
    }
    free(cx);free(sg);free(e);free(pp);free(Xt);
    return wq;
}
static float *dq_quant_expert_signref(const float *W,int nrows,int ncols,const float *X,int n_act){
    return dq_quant_expert_signref_adj(W,nrows,ncols,X,n_act,NULL,NULL);
}
/* signref 的 GO1B_BLK 导出版(与 dq_quant_expert_signref 同解, 修改需同步):
 * 行scale+符号精修 → 34B/256el 块格式 bytes(行 scale fp16 复制进每块), 直接可写 GGUF。
 * out 需 nrows*go1b_blk_row_bytes(ncols) 字节。wq_opt 非 NULL 时同时输出 dequant float
 * (nrows×ncols, 供 w2 顺序补偿校准链复用, 免二次求解)。 */
static void dq_signref_export(const float *W,int nrows,int ncols,const float *X,int n_act,
                              uint8_t *out,float *wq_opt){
    size_t rb=go1b_blk_row_bytes(ncols);
    int nblk=(ncols+GO1B_BLK_QK-1)/GO1B_BLK_QK;
    double MU_SCALE=dq_signref_mu; int ROUNDS=dq_signref_rounds;
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
            for(int round=0;round<=ROUNDS;round++){
                double sp2=0,spy=0;
                for(int t=0;t<n_act;t++){ const float *x=X+(size_t)t*ncols; double p=0,y=0;
                    for(int j=0;j<ncols;j++){ double xj=x[j]; p += (sg[j]>0)?xj:-xj; y += (double)w[j]*xj; }
                    sp2+=p*p; spy+=p*y; e[t]=y; pp[t]=p;
                }
                double lam=sp2/((double)n_act+1.0), lam0=1e-3*sp2+1e-9; if(lam<lam0)lam=lam0;
                s=(spy+lam*s0)/(sp2+lam);
                if(s<0.0)s=s0; else if(s>4.0*s0+1e-12)s=4.0*s0;
                for(int t=0;t<n_act;t++) e[t]-= s*pp[t];
                if(round==ROUNDS) break;
                for(int j=0;j<ncols;j++){
                    const float *xj=Xt+(size_t)j*n_act;
                    double g1=0; for(int t=0;t<n_act;t++) g1+=e[t]*(double)xj[t];
                    double sj=(double)sg[j];
                    double dE = 4.0*s*sj*g1 + 4.0*s*s*cx[j] + mu*4.0*s*sj*(double)w[j];
                    if(dE<0.0){ for(int t=0;t<n_act;t++) e[t]+= 2.0*s*sj*(double)xj[t];
                                sg[j]=(signed char)(-sg[j]); }
                }
            }
        }
        uint16_t sh=go1b_fp32_to_fp16((float)s);
        uint8_t *rd=out+(size_t)r*rb;
        for(int b=0;b<nblk;b++){
            uint8_t *bd=rd+(size_t)b*GO1B_BLK_BYTES;
            go1b_store_u16_le(bd,sh);
            uint8_t *sgb=bd+2;
            for(int k=0;k<GO1B_BLK_QK/8;k++){
                uint8_t byte=0;
                for(int bit=0;bit<8;bit++){ int j=b*GO1B_BLK_QK+k*8+bit;
                    if(j<ncols&&sg[j]>0) byte|=(uint8_t)(1u<<bit); }
                sgb[k]=byte;
            }
        }
        if(wq_opt){ float sf=go1b_fp16_to_fp32(sh); float *o=wq_opt+(size_t)r*ncols;
            for(int j=0;j<ncols;j++) o[j]= sg[j]>0? sf:-sf; }
    }
    free(sg); if(Xt)free(Xt); if(cx)free(cx); if(e)free(e); if(pp)free(pp);
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
