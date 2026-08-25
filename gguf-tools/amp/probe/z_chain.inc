/* dual-form z-solve: n<<d_in 时 W = X^T (XX^T+λI)^-1 R, Cholesky 在 n×n(而非 d_in³).
 * 数学等价 ds4_z_solve(同 ridge 正规方程, dual 恒等式), 但 O(n³+n²·d) 而非 O(d³): 90s→<1s.
 * 复用 ds4_z.c 同-TU static 原语 cholesky/cholesky_solve/mgs/lcg_unit + 子空间迭代因子分解.
 * (遗留 z 工具链的落地实现, 本文件当前不调用, 保留供 z_explore 流复活。) */
__attribute__((unused))
static ds4_z *z_solve_dual(const float *X, const float *R, uint32_t n,
                           uint32_t d_in, uint32_t d_out, uint32_t rank, float lambda){
    if(!X||!R||!n||!d_in||!d_out||!rank) return NULL;
    if(rank>d_in)rank=d_in; if(rank>d_out)rank=d_out; if(rank>n)rank=n;
    double *G=calloc((size_t)n*n,sizeof(double)); if(!G) return NULL;
    for(uint32_t a=0;a<n;a++){ const float*xa=X+(size_t)a*d_in;
        for(uint32_t b=a;b<n;b++){ const float*xb=X+(size_t)b*d_in; double s=0;
            for(uint32_t i=0;i<d_in;i++) s+=(double)xa[i]*xb[i]; G[(size_t)a*n+b]=s; } }
    double tr=0; for(uint32_t a=0;a<n;a++) tr+=G[(size_t)a*n+a];       /* tr(XX^T)=tr(X^TX) */
    double ridge=(double)lambda*(tr/(double)d_in)+1e-10;              /* primal 口径 ridge */
    for(uint32_t a=0;a<n;a++){ G[(size_t)a*n+a]+=ridge;
        for(uint32_t b=a+1;b<n;b++) G[(size_t)b*n+a]=G[(size_t)a*n+b]; }
    if(cholesky(G,n)!=0){ free(G); return NULL; }
    double *al=malloc((size_t)n*d_out*sizeof(double)),*cc=malloc((size_t)n*sizeof(double));
    if(!al||!cc){free(G);free(al);free(cc);return NULL;}
    for(uint32_t j=0;j<d_out;j++){ for(uint32_t a=0;a<n;a++) cc[a]=R[(size_t)a*d_out+j];
        cholesky_solve(G,n,cc); for(uint32_t a=0;a<n;a++) al[(size_t)a*d_out+j]=cc[a]; }   /* α=G^-1 R */
    free(cc); free(G);
    float *W=malloc((size_t)d_in*d_out*sizeof(float)); if(!W){free(al);return NULL;}
    for(uint32_t i=0;i<d_in;i++){ float*Wi=W+(size_t)i*d_out; for(uint32_t j=0;j<d_out;j++)Wi[j]=0.0f;
        for(uint32_t a=0;a<n;a++){ double xai=X[(size_t)a*d_in+i]; if(xai==0.0)continue;
            const double*ala=al+(size_t)a*d_out; for(uint32_t j=0;j<d_out;j++) Wi[j]+=(float)(xai*ala[j]); } }  /* W=X^T α */
    free(al);
    /* 子空间迭代 rank-k 截断 (与 ds4_z_solve 尾部一致) */
    ds4_z *zl=calloc(1,sizeof(*zl));
    float *V=malloc((size_t)d_in*rank*4),*T=malloc((size_t)d_out*rank*4),*M=malloc((size_t)rank*d_out*4),*U=malloc((size_t)d_out*rank*4),*z=malloc((size_t)rank*4);
    if(!zl||!V||!T||!M||!U||!z){free(zl);free(V);free(T);free(M);free(U);free(z);free(W);return NULL;}
    uint64_t seed=0x5A5A1EEDULL;
    for(size_t i=0;i<(size_t)d_in*rank;i++) V[i]=lcg_unit(&seed);
    mgs(V,d_in,rank,&seed);
    for(int it=0;it<12;it++){
        memset(T,0,(size_t)d_out*rank*4);
        for(uint32_t i=0;i<d_in;i++){ const float*Wi=W+(size_t)i*d_out,*Vi=V+(size_t)i*rank;
            for(uint32_t j=0;j<d_out;j++){ float wij=Wi[j]; if(wij==0.0f)continue; float*Tj=T+(size_t)j*rank;
                for(uint32_t c=0;c<rank;c++) Tj[c]+=wij*Vi[c]; } }
        for(uint32_t i=0;i<d_in;i++){ const float*Wi=W+(size_t)i*d_out; float*Vi=V+(size_t)i*rank;
            for(uint32_t c=0;c<rank;c++)Vi[c]=0.0f;
            for(uint32_t j=0;j<d_out;j++){ float wij=Wi[j]; if(wij==0.0f)continue; const float*Tj=T+(size_t)j*rank;
                for(uint32_t c=0;c<rank;c++) Vi[c]+=wij*Tj[c]; } }
        mgs(V,d_in,rank,&seed);
    }
    memset(M,0,(size_t)rank*d_out*4);
    for(uint32_t i=0;i<d_in;i++){ const float*Wi=W+(size_t)i*d_out,*Vi=V+(size_t)i*rank;
        for(uint32_t c=0;c<rank;c++){ float v=Vi[c]; if(v==0.0f)continue; float*Mc=M+(size_t)c*d_out;
            for(uint32_t j=0;j<d_out;j++) Mc[j]+=v*Wi[j]; } }
    for(uint32_t c=0;c<rank;c++){ double nrm=0; const float*Mc=M+(size_t)c*d_out;
        for(uint32_t j=0;j<d_out;j++) nrm+=(double)Mc[j]*Mc[j]; nrm=sqrt(nrm); z[c]=(float)nrm;
        float inv=nrm>1e-20?(float)(1.0/nrm):0.0f; for(uint32_t j=0;j<d_out;j++) U[(size_t)j*rank+c]=Mc[j]*inv; }
    free(M);free(T);free(W);
    for(uint32_t a=0;a<rank;a++){ uint32_t best=a; for(uint32_t b=a+1;b<rank;b++) if(z[b]>z[best])best=b;
        if(best!=a){ float tz=z[a];z[a]=z[best];z[best]=tz;
            for(uint32_t i=0;i<d_in;i++){float tv=V[(size_t)i*rank+a];V[(size_t)i*rank+a]=V[(size_t)i*rank+best];V[(size_t)i*rank+best]=tv;}
            for(uint32_t j=0;j<d_out;j++){float tu=U[(size_t)j*rank+a];U[(size_t)j*rank+a]=U[(size_t)j*rank+best];U[(size_t)j*rank+best]=tu;} } }
    zl->U=U;zl->V=V;zl->z=z;zl->d_in=d_in;zl->d_out=d_out;zl->rank=rank;zl->k=rank; return zl;
}

/* 四损失选秩 k_L (产物③旋钮): 在【留出 val 行】(X[n,d]→R[n,d]) 上对候选 k(含 k=0=该层
 * z 关闭的 GO/NO-GO 门) 算 L_align/L_classify/L_smooth/L_fixed, 归一加权取最小。
 * gz 首跑教训: 在拟合行自评→过拟合链式反应(L26 拉出流形→L28 累积1.79); val 评估+k0 门是正解。 */
static int z_pick_rank(ds4_z*zl, const float*X, const float*R, int n, int d,
                       double*la,double*lc,double*ls,double*lx){
    int rank=(int)zl->rank; if(rank<1)return 0;
    float *yh=malloc((size_t)n*d*4);
    float *wv=malloc((size_t)d*4); ds4_loss_dim_variance(R,n,d,wv);
    float *dv=malloc((size_t)d*4);
    uint64_t sd=0x9E3779B97F4A7C15ULL;
    for(int i=0;i<d;i++){ sd=sd*6364136223846793005ULL+1442695040888963407ULL; dv[i]=((float)((sd>>40)&0xFFFFFF)/16777216.0f-0.5f)*0.02f; }
    float *yd=malloc((size_t)d*4);
    int ks[12],nk=0; ks[nk++]=0; for(int k=1;k<=rank;k*=2){ ks[nk++]=k; } if(ks[nk-1]!=rank)ks[nk++]=rank;
    double La[12],Lc[12],Ls[12],Lx[12];
    for(int ki=0;ki<nk;ki++){ int kk=ks[ki];
        memset(yh,0,(size_t)n*d*4);
        if(kk>0){ ds4_z_set_rank(zl,(uint32_t)kk);
            for(int s=0;s<n;s++) ds4_z_apply(zl,X+(size_t)s*d,yh+(size_t)s*d); }
        La[ki]=ds4_loss_align(yh,R,(uint32_t)n,(uint32_t)d);
        Lc[ki]=ds4_loss_classify(yh,R,wv,(uint32_t)n,(uint32_t)d);
        if(kk>0){ memset(yd,0,(size_t)d*4); ds4_z_apply(zl,dv,yd);
            double sm=0; for(int i=0;i<d;i++) sm+=(double)yd[i]*yd[i]; Ls[ki]=sm;
            double en=0; for(int c=0;c<kk;c++) en+=(double)zl->z[c]*zl->z[c]; Lx[ki]=en;
        } else { Ls[ki]=0.0; Lx[ki]=0.0; }
    }
    double mA=1e-30,mC=1e-30,mS=1e-30,mX=1e-30;
    for(int ki=0;ki<nk;ki++){ if(La[ki]>mA)mA=La[ki]; if(Lc[ki]>mC)mC=Lc[ki]; if(Ls[ki]>mS)mS=Ls[ki]; if(Lx[ki]>mX)mX=Lx[ki]; }
    int bk=0; double bL=1e300;
    for(int ki=0;ki<nk;ki++){ double t=1.0*(La[ki]/mA)+0.5*(Lc[ki]/mC)+0.15*(Ls[ki]/mS)+0.05*(Lx[ki]/mX);
        if(t<bL){bL=t;bk=ki;} }
    if(ks[bk]>0) ds4_z_set_rank(zl,(uint32_t)ks[bk]);
    if(la)*la=La[bk]; if(lc)*lc=Lc[bk]; if(ls)*ls=Ls[bk]; if(lx)*lx=Lx[bk];
    free(yh);free(wv);free(dv);free(yd); return ks[bk];
}

/* 四损失驱动的 z 求解(不只选秩, 四损失进求解本身; ALGORITHM.md §4 语义):
 *   L_align    = RRR/ridge 数据项本体(方向还原);
 *   L_classify = R 列按 per-dim 方差(重要性)加权 —— sqrt(v_d) 缩放进低秩截断(加权低秩),
 *                解完 U 行反缩放还原(逐 dim ridge 解本身不受列权影响, 影响的正是秩截断取舍);
 *   L_smooth   = 固定种子 dither 增广行 (x_t+δ_t → 同目标 R_t, 行权 augw): 线性 z 下
 *                即惩罚 ‖z(x+δ)−z(x)‖², 输入鲁棒/不跳变;
 *   L_fixed    = ridge λ(weight-decay, 外层按过拟合比动态) + 同一 dither 增广钉死解。
 * (加法 z^L 形态 val 门实测判 0 后退役; 保留给 legacy z 路径/后续复用。) */
__attribute__((unused))
static ds4_z *z_solve_fourloss(const float *X,const float *R,int n,int d_in,int d_out,int rank,float lambda,
                               const float *colw,int naug,float augw,uint64_t seed){
    /* ★2026-08-08 修: 原单一 d 同时当 d_in/d_out 用 — R 为矩形(如专家级 4096→2048)时
     * 行步长读越界 SIGSEGV。z_solve_dual 底层本就支持 d_in≠d_out, 包装层跟上。 */
    if(n<1||rank<1) return NULL;
    if(naug>n) naug=n;
    int N=n+naug;
    float *Xa=malloc((size_t)N*d_in*4), *Ra=malloc((size_t)N*d_out*4);
    if(!Xa||!Ra){ free(Xa);free(Ra); return NULL; }
    memcpy(Xa,X,(size_t)n*d_in*4);
    for(int i=0;i<n;i++) for(int j=0;j<d_out;j++)
        Ra[(size_t)i*d_out+j]=R[(size_t)i*d_out+j]*(colw?sqrtf(colw[j]):1.0f);
    uint64_t s=seed?seed:0x9E3779B97F4A7C15ULL;
    float sw=sqrtf(augw>0?augw:0.25f);
    for(int a=0;a<naug;a++){
        int src=a%n;
        const float*xs=X+(size_t)src*d_in; float*xd=Xa+(size_t)(n+a)*d_in;
        double rms=0; for(int j=0;j<d_in;j++) rms+=(double)xs[j]*xs[j]; rms=sqrt(rms/d_in+1e-30);
        for(int j=0;j<d_in;j++){ s=s*6364136223846793005ULL+1442695040888963407ULL;
            float dl=((float)((s>>40)&0xFFFFFF)/16777216.0f-0.5f)*0.04f*(float)rms;
            xd[j]=(xs[j]+dl)*sw; }
        const float*rs=Ra+(size_t)src*d_out; float*rd=Ra+(size_t)(n+a)*d_out;
        for(int j=0;j<d_out;j++) rd[j]=rs[j]*sw;
    }
    ds4_z *zl=z_solve_dual(Xa,Ra,(uint32_t)N,(uint32_t)d_in,(uint32_t)d_out,(uint32_t)rank,lambda);
    free(Xa);free(Ra);
    if(zl&&colw){ for(int j=0;j<d_out;j++){ float iw=colw[j]>1e-12f?1.0f/sqrtf(colw[j]):0.0f;
        for(uint32_t c=0;c<zl->rank;c++) zl->U[(size_t)j*zl->rank+c]*=iw; } }
    return zl;
}
