/* 对偶 ridge z 解算器 z_solve_dual 与它的并行原语(2026-08-28 从 ds4quant_run_p1 拆出)。
 * 依赖 p1 里的 cholesky/mgs/lcg_unit/ds4_z, 故必须紧跟 p1 之后 include。 */
/* ★z_solve_dual 并行化(2026-08-23 时长账: 单线程 600GFLOP=5min/层 → 20 线程)★
 * 数学不动: Gram 上三角/回代列/W 重建行 三段独立可并行; cholesky 分解保持单线程。 */
typedef struct { void (*fn)(void*,int,int); void *ctx; int i0,i1; } zpf_arg;
static void *zpf_tramp(void *a){ zpf_arg *p=(zpf_arg*)a; p->fn(p->ctx,p->i0,p->i1); return NULL; }
static void zpar_for(int n,int nth,void (*fn)(void*,int,int),void *ctx){
    if(nth>n)nth=n>0?n:1; if(nth>32)nth=32;
    pthread_t th[32]; zpf_arg pa[32]; int per=(n+nth-1)/nth,cnt=0;
    for(int t=0;t<nth;t++){ int i0=t*per,i1=i0+per>n?n:i0+per; if(i0>=i1)break;
        pa[cnt]=(zpf_arg){fn,ctx,i0,i1};
        if(pthread_create(&th[cnt],NULL,zpf_tramp,&pa[cnt])){ pa[cnt].fn(ctx,i0,i1); continue; }
        cnt++; }
    for(int t=0;t<cnt;t++) pthread_join(th[t],NULL);
}
typedef struct { const float*X; double*G; uint32_t n,d_in; } zg_ctx;
static void zg_worker(void *vc,int a0,int a1){ zg_ctx*c=(zg_ctx*)vc;
    for(int a=a0;a<a1;a++){ const float*xa=c->X+(size_t)a*c->d_in;
        for(uint32_t b=(uint32_t)a;b<c->n;b++){ const float*xb=c->X+(size_t)b*c->d_in; double s=0;
            for(uint32_t i=0;i<c->d_in;i++) s+=(double)xa[i]*xb[i]; c->G[(size_t)a*c->n+b]=s; } } }
typedef struct { const double*G; const float*R; double*al; uint32_t n,d_out; } zs_ctx;
static void cholesky_solve(const double *L, uint32_t d, double *b);   /* ds4_z.c 同-TU 原语(本就 const) */
#define cholesky_solve_nc cholesky_solve
static void zs_worker(void *vc,int j0,int j1){ zs_ctx*c=(zs_ctx*)vc;
    double *cc=malloc((size_t)c->n*sizeof(double));
    for(int j=j0;j<j1;j++){ for(uint32_t a=0;a<c->n;a++) cc[a]=c->R[(size_t)a*c->d_out+j];
        cholesky_solve_nc(c->G,c->n,cc);
        for(uint32_t a=0;a<c->n;a++) c->al[(size_t)a*c->d_out+j]=cc[a]; }
    free(cc); }
typedef struct { const float*X; const double*al; float*W; uint32_t n,d_in,d_out; } zw_ctx;
static void zw_worker(void *vc,int i0,int i1){ zw_ctx*c=(zw_ctx*)vc;
    for(int i=i0;i<i1;i++){ float*Wi=c->W+(size_t)i*c->d_out;
        for(uint32_t j=0;j<c->d_out;j++)Wi[j]=0.0f;
        for(uint32_t a=0;a<c->n;a++){ double xai=c->X[(size_t)a*c->d_in+i]; if(xai==0.0)continue;
            const double*ala=c->al+(size_t)a*c->d_out;
            for(uint32_t j=0;j<c->d_out;j++) Wi[j]+=(float)(xai*ala[j]); } } }
/* 子空间迭代的三个并行核(切法与逐位不变的理由见 z_solve_dual 里的调用点注释) */
typedef struct { const float*W; float*V,*T,*M; uint32_t d_in,d_out,rank; } zsub_ctx;
static void zsub_T(void *vc,int j0,int j1){ zsub_ctx*c=(zsub_ctx*)vc;
    for(uint32_t i=0;i<c->d_in;i++){ const float*Wi=c->W+(size_t)i*c->d_out,*Vi=c->V+(size_t)i*c->rank;
        for(int j=j0;j<j1;j++){ float wij=Wi[j]; if(wij==0.0f)continue; float*Tj=c->T+(size_t)j*c->rank;
            for(uint32_t k=0;k<c->rank;k++) Tj[k]+=wij*Vi[k]; } } }
static void zsub_V(void *vc,int i0,int i1){ zsub_ctx*c=(zsub_ctx*)vc;
    for(int i=i0;i<i1;i++){ const float*Wi=c->W+(size_t)i*c->d_out; float*Vi=c->V+(size_t)i*c->rank;
        for(uint32_t k=0;k<c->rank;k++)Vi[k]=0.0f;
        for(uint32_t j=0;j<c->d_out;j++){ float wij=Wi[j]; if(wij==0.0f)continue;
            const float*Tj=c->T+(size_t)j*c->rank;
            for(uint32_t k=0;k<c->rank;k++) Vi[k]+=wij*Tj[k]; } } }
static void zsub_M(void *vc,int c0,int c1){ zsub_ctx*c=(zsub_ctx*)vc;
    for(uint32_t i=0;i<c->d_in;i++){ const float*Wi=c->W+(size_t)i*c->d_out,*Vi=c->V+(size_t)i*c->rank;
        for(int k=c0;k<c1;k++){ float v=Vi[k]; if(v==0.0f)continue; float*Mc=c->M+(size_t)k*c->d_out;
            for(uint32_t j=0;j<c->d_out;j++) Mc[j]+=v*Wi[j]; } } }
static ds4_z *z_solve_dual(const float *X, const float *R, uint32_t n,
                           uint32_t d_in, uint32_t d_out, uint32_t rank, float lambda){
    if(!X||!R||!n||!d_in||!d_out||!rank) return NULL;
    if(rank>d_in)rank=d_in; if(rank>d_out)rank=d_out; if(rank>n)rank=n;
    double *G=calloc((size_t)n*n,sizeof(double)); if(!G) return NULL;
    { zg_ctx gc={X,G,n,d_in}; zpar_for((int)n,20,zg_worker,&gc); }
    double tr=0; for(uint32_t a=0;a<n;a++) tr+=G[(size_t)a*n+a];       /* tr(XX^T)=tr(X^TX) */
    double ridge=(double)lambda*(tr/(double)d_in)+1e-10;              /* primal 口径 ridge */
    for(uint32_t a=0;a<n;a++){ G[(size_t)a*n+a]+=ridge;
        for(uint32_t b=a+1;b<n;b++) G[(size_t)b*n+a]=G[(size_t)a*n+b]; }
    if(cholesky(G,n)!=0){ free(G); return NULL; }
    double *al=malloc((size_t)n*d_out*sizeof(double));
    if(!al){free(G);return NULL;}
    { zs_ctx sc={G,R,al,n,d_out}; zpar_for((int)d_out,20,zs_worker,&sc); }   /* α=G^-1 R, 列并行 */
    free(G);
    float *W=malloc((size_t)d_in*d_out*sizeof(float)); if(!W){free(al);return NULL;}
    { zw_ctx wc={X,al,W,n,d_in,d_out}; zpar_for((int)d_in,20,zw_worker,&wc); }  /* W=X^T α, 行并行 */
    free(al);
    /* 子空间迭代 rank-k 截断 (与 ds4_z_solve 尾部一致) */
    ds4_z *zl=calloc(1,sizeof(*zl));
    float *V=malloc((size_t)d_in*rank*4),*T=malloc((size_t)d_out*rank*4),*M=malloc((size_t)rank*d_out*4),*U=malloc((size_t)d_out*rank*4),*z=malloc((size_t)rank*4);
    if(!zl||!V||!T||!M||!U||!z){free(zl);free(V);free(T);free(M);free(U);free(z);free(W);return NULL;}
    uint64_t seed=0x5A5A1EEDULL;
    for(size_t i=0;i<(size_t)d_in*rank;i++) V[i]=lcg_unit(&seed);
    mgs(V,d_in,rank,&seed);
    /* ★子空间迭代三个循环并行化(2026-08-28)★ 上面 Gram/回代/W 三段 08-23 已经上了 20 线程,
     * 唯独这里(12 轮 × 两个 d_in×d_out×rank 的循环 + 收尾的 M)一直是单线程:
     * 12×2×4096×4096×64 ≈ 258 亿次乘加, 实测占 z_solve_dual(整层 32s)里的 5s 左右。
     * ★逐位不变★: 切法都选"输出元素的累加序不变"的那一维 ——
     *   T[j][c] 按 j 切(每线程仍按 i 升序累加, 且 Wi[j0..j1) 连续读);
     *   V[i][c] 按 i 切(本来就是每个 i 独立清零再累加);
     *   M[c][j] 按 c 切(每线程仍按 i 升序累加)。
     * 按 i 切会需要每线程私有 T 再归约 = 求和顺序变了, 所以没那么切。 */
    { zsub_ctx sc={W,V,T,M,d_in,d_out,rank};
      for(int it=0;it<12;it++){
        memset(T,0,(size_t)d_out*rank*4);
        zpar_for((int)d_out,20,zsub_T,&sc);
        zpar_for((int)d_in,20,zsub_V,&sc);
        mgs(V,d_in,rank,&seed);
      }
      memset(M,0,(size_t)rank*d_out*4);
      zpar_for((int)rank,20,zsub_M,&sc); }
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

