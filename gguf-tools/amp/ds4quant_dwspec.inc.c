/* ds4quant_dwspec.inc.c — 量化残差 ΔW 谱针(2026-09-02 用户令"专心解决 same-top 90%")
 *
 * 问: "大模型一定有低维表达/浅层放大规律"要成立, 量化误差本身得有低维结构。z^L(k=16/64, 四损失)
 *     在层出口只吃到 0.1~1.8% 误差能量(sweep3 ZLGATE), 非线性/SGD 版也零或过拟合。这些都是
 *     "数据条件"下的测量, 受 8192 token 限制。本针不靠数据: 直接看每个专家 ΔW=W−Ŵ 在权重空间
 *     的奇异谱 —— 前 k 个方向占多少能量。白噪声=k/r; 低维结构=远高于 k/r。
 * 法: 对点名层×固定 3 个专家×{w1,w3,w2}: W 从 HF 读(FP), Ŵ 从 dql_vq 反量化(生产同路),
 *     G=ΔW·ΔWᵀ(double), 块幂迭代取前 64 个特征值, 报 k=1/4/16/64 累计能量份额, 对照 W 自身谱
 *     与各向同性基线 k/r。只读, 打完即退。 */
static const char *g_dw_spec = NULL;   /* --dw-spectrum "0,12,24,36,41" 层号列表 */
typedef struct { const float *A; int r, c, r0, r1; double *G; } dws_gram_t;
static void *dws_gram_worker(void *a){
    dws_gram_t *g=(dws_gram_t*)a;
    for(int i=g->r0;i<g->r1;i++){ const float *ai=g->A+(size_t)i*g->c;
        for(int j=0;j<=i;j++){ const float *aj=g->A+(size_t)j*g->c; double s=0;
            for(int k=0;k<g->c;k++) s+=(double)ai[k]*aj[k];
            g->G[(size_t)i*g->r+j]=s; g->G[(size_t)j*g->r+i]=s; } }
    return NULL; }
/* 前 kk 个特征值(块幂迭代+Gram-Schmidt, 40 轮; 对称半正定 G) → 返回 λ 降序 */
static void dws_topk(const double *G,int r,int kk,int iters,double *lam){
    double *V=malloc((size_t)r*kk*8),*W=malloc((size_t)r*kk*8);
    unsigned s=12345u; for(size_t i=0;i<(size_t)r*kk;i++){ s=s*1103515245u+12345u; V[i]=((s>>8)&0xffff)/65536.0-0.5; }
    for(int it=0;it<iters;it++){
        for(int j=0;j<kk;j++) for(int i=0;i<r;i++){ double acc=0; const double *gi=G+(size_t)i*r;
            for(int t=0;t<r;t++) acc+=gi[t]*V[(size_t)t*kk+j]; W[(size_t)i*kk+j]=acc; }
        for(int j=0;j<kk;j++){   /* 正交化 */
            for(int p=0;p<j;p++){ double d=0; for(int i=0;i<r;i++) d+=W[(size_t)i*kk+j]*W[(size_t)i*kk+p];
                for(int i=0;i<r;i++) W[(size_t)i*kk+j]-=d*W[(size_t)i*kk+p]; }
            double n=0; for(int i=0;i<r;i++) n+=W[(size_t)i*kk+j]*W[(size_t)i*kk+j]; n=sqrt(n)+1e-300;
            for(int i=0;i<r;i++) W[(size_t)i*kk+j]/=n; }
        memcpy(V,W,(size_t)r*kk*8);
    }
    for(int j=0;j<kk;j++){ double acc=0;   /* Rayleigh 商 */
        for(int i=0;i<r;i++){ double gv=0; const double *gi=G+(size_t)i*r; for(int t=0;t<r;t++) gv+=gi[t]*V[(size_t)t*kk+j]; acc+=V[(size_t)i*kk+j]*gv; }
        lam[j]=acc; }
    for(int a=0;a<kk;a++) for(int b=a+1;b<kk;b++) if(lam[b]>lam[a]){ double t=lam[a]; lam[a]=lam[b]; lam[b]=t; }
    free(V); free(W);
}
static void dws_spectrum(const char*tag,const float*A,int r,int c,int nth){
    double *G=malloc((size_t)r*r*8); pthread_t th[32]; dws_gram_t gw[32]; if(nth>32)nth=32;
    /* 行块按三角面积均分(i 越大行越贵) */
    for(int t=0;t<nth;t++){ int r0=(int)(r*sqrt((double)t/nth)), r1=(int)(r*sqrt((double)(t+1)/nth)); if(t==nth-1)r1=r;
        gw[t]=(dws_gram_t){A,r,c,r0,r1,G}; pthread_create(&th[t],NULL,dws_gram_worker,&gw[t]); }
    for(int t=0;t<nth;t++) pthread_join(th[t],NULL);
    double tr=0; for(int i=0;i<r;i++) tr+=G[(size_t)i*r+i];
    enum { KK=64 }; double lam[KK]; dws_topk(G,r,KK,40,lam);
    double c1=lam[0],c4=0,c16=0,c64=0; for(int j=0;j<4;j++)c4+=lam[j]; for(int j=0;j<16;j++)c16+=lam[j]; for(int j=0;j<KK;j++)c64+=lam[j];
    printf("  %-3s r=%d 能量份额 k=1:%5.2f%% k=4:%5.2f%% k=16:%5.2f%% k=64:%5.2f%% | 各向同性基线 k=64:%5.2f%%\n",
           tag,r,100*c1/tr,100*c4/tr,100*c16/tr,100*c64/tr,100.0*KK/r);
    free(G);
}
static void dw_spectrum_probe(void){
    if(!g_dw_spec) return;
    if(!g_cli.layer_dir){ fprintf(stderr,"[DWSPEC]★需 --layer-dir★\n"); exit(2); }
    const int EX[3]={0,128,255}; int nth=NTHREADS>1?NTHREADS:8;
    char b[256]; snprintf(b,sizeof b,"%s",g_dw_spec);
    for(char *tok=strtok(b,","); tok; tok=strtok(NULL,",")){ int L=atoi(tok);
        char lp[512]; snprintf(lp,sizeof lp,"%s/dql_L%02d.bin",g_cli.layer_dir,L);
        lfile_t lf; if(lfile_load(lp,&lf)!=0||!lf.vqmap){ fprintf(stderr,"[DWSPEC]★L%d 层件/vqmap 缺★\n",L); continue; }
        const uint64_t *vtab=(const uint64_t*)(lf.vqmap+16);
        for(int xi=0;xi<3;xi++){ int e=EX[xi];
            const char *mn[3]={"w1","w3","w2"}; uint64_t of[3]={vtab[(size_t)e*3],vtab[(size_t)e*3+1],vtab[(size_t)e*3+2]};
            printf("[DWSPEC] L%02d e%03d\n",L,e);
            for(int m=0;m<3;m++){
                if(!of[m]){ printf("  %-3s 冷槽(w2 go1b)跳过\n",mn[m]); continue; }
                char n[160]; snprintf(n,sizeof n,"layers.%d.ffn.experts.%d.%s.weight",L,e,mn[m]);
                long r,c; float *W=st_read_weight(&C,n,&r,&c); if(!W){ printf("  %-3s HF 读失败\n",mn[m]); continue; }
                float *Q=malloc((size_t)r*c*4); int qr=0,qc=0;
                if(vq_unpack_dequant(lf.vqmap+of[m],lf.vqmsz-of[m],Q,&qr,&qc)!=0||qr!=r||qc!=c){   /* 0=成功(C 约定) */
                    printf("  %-3s 反量化失败/形状不齐(%d×%d vs %ld×%ld)\n",mn[m],qr,qc,r,c); free(W); free(Q); continue; }
                float *D=malloc((size_t)r*c*4); double eD=0,eW=0;
                for(size_t i=0;i<(size_t)r*c;i++){ D[i]=W[i]-Q[i]; eD+=(double)D[i]*D[i]; eW+=(double)W[i]*W[i]; }
                printf("  %-3s %ld×%ld ‖ΔW‖²/‖W‖²=%.4f\n",mn[m],r,c,eD/eW);
                dws_spectrum("ΔW",D,(int)r,(int)c,nth);
                dws_spectrum("W",W,(int)r,(int)c,nth);
                fflush(stdout); free(W); free(Q); free(D);
            }
        }
        lfile_free(&lf);
    }
    exit(0);
}

/* ═══ 巨值通道针(2026-09-02 用户批"打一针", 方案 C 第一刀的判决)═══
 * 权重空间 ΔW 是白的(上面 dwspec), 但模型看到的是 (W−Ŵ)·x: x 在残差流通道上极不均匀。
 * 问两个数: ①每层前 c 个输入通道占 Σx² 多少(巨值通道存不存在) ②把 w1/w3 的这 c 列换回原精度,
 * 专家输出误差能量 ‖(W−Ŵ)X‖² 剩多少(巨值列侧车这一刀有没有肉)。X=锚里该层 FP Fin 中实际路由到
 * 该专家的行(部署路由口径); 通道序=全层 Σx² 降序(可部署的固定集合), 只读不写。 */
static const char *g_mc_probe = NULL;   /* --mc-probe "0,12,24,36,41" */
static int mc_cmp_desc_idx(const void *a, const void *b, void *arg){ const double *E=arg; int i=*(const int*)a, j=*(const int*)b; return (E[j]>E[i])-(E[j]<E[i]); }
static void mc_probe(int S){
    if(!g_mc_probe) return;
    if(!ANC_OK||!ANC.fin||!ANC.ridx){ fprintf(stderr,"[MCPROBE]★需 --anchor(fin/ridx 在场)★\n"); exit(2); }
    if(!g_cli.layer_dir){ fprintf(stderr,"[MCPROBE]★需 --layer-dir★\n"); exit(2); }
    const int EX[3]={0,128,255}; const int CS[4]={8,32,128,512}; const int NEMAX=1024;
    char b[256]; snprintf(b,sizeof b,"%s",g_mc_probe);
    for(char *tok=strtok(b,","); tok; tok=strtok(NULL,",")){ int L=atoi(tok);
        const float *FIN=ANC.fin+(size_t)L*S*DIM; const int32_t *RIDX=ANC.ridx+(size_t)L*S*NACT;
        double *E=calloc(DIM,8); double Et=0;
        for(int s=0;s<S;s++){ const float*x=FIN+(size_t)s*DIM; for(int j=0;j<DIM;j++) E[j]+=(double)x[j]*x[j]; }
        for(int j=0;j<DIM;j++) Et+=E[j];
        int *ord=malloc(DIM*sizeof(int)); for(int j=0;j<DIM;j++) ord[j]=j; qsort_r(ord,DIM,sizeof(int),mc_cmp_desc_idx,E);
        double c8=0,c32=0,c128=0,c512=0; for(int k=0;k<512;k++){ double v=E[ord[k]]; if(k<8)c8+=v; if(k<32)c32+=v; if(k<128)c128+=v; c512+=v; }
        printf("[MCPROBE] L%02d 全层 Σx² 通道集中度: top8=%.1f%% top32=%.1f%% top128=%.1f%% top512=%.1f%% (各向同性 top32=%.1f%%) | top8 通道:",
               L,100*c8/Et,100*c32/Et,100*c128/Et,100*c512/Et,100.0*32/DIM);
        for(int k=0;k<8;k++) printf(" %d(%.1f%%)",ord[k],100*E[ord[k]]/Et); printf("\n"); fflush(stdout);
        char lp[512]; snprintf(lp,sizeof lp,"%s/dql_L%02d.bin",g_cli.layer_dir,L);
        lfile_t lf; if(lfile_load(lp,&lf)!=0||!lf.vqmap){ fprintf(stderr,"[MCPROBE]★L%d 层件缺★\n",L); free(E); free(ord); continue; }
        const uint64_t *vtab=(const uint64_t*)(lf.vqmap+16);
        for(int xi=0;xi<3;xi++){ int e=EX[xi];
            /* 该专家的行 */
            int *rows=malloc(S*sizeof(int)); int ne=0;
            for(int s=0;s<S&&ne<NEMAX;s++){ for(int a=0;a<NACT;a++) if(RIDX[(size_t)s*NACT+a]==e){ rows[ne++]=s; break; } }
            if(ne<8){ printf("  e%03d 路由行仅 %d, 跳过\n",e,ne); free(rows); continue; }
            /* 专家条件通道集中度 */
            double *Ee=calloc(DIM,8), Eet=0; for(int i=0;i<ne;i++){ const float*x=FIN+(size_t)rows[i]*DIM; for(int j=0;j<DIM;j++) Ee[j]+=(double)x[j]*x[j]; }
            for(int j=0;j<DIM;j++) Eet+=Ee[j]; double ce32=0; for(int k=0;k<32;k++) ce32+=Ee[ord[k]];
            printf("  e%03d 行=%d 专家条件 top32(全层序)占 Σx² %.1f%%\n",e,ne,100*ce32/Eet); fflush(stdout);
            const char *mn[2]={"w1","w3"}; uint64_t of[2]={vtab[(size_t)e*3],vtab[(size_t)e*3+1]};
            for(int m=0;m<2;m++){
                if(!of[m]){ printf("    %-2s 冷槽跳过\n",mn[m]); continue; }
                char n[160]; snprintf(n,sizeof n,"layers.%d.ffn.experts.%d.%s.weight",L,e,mn[m]);
                long r,c; float *W=st_read_weight(&C,n,&r,&c); if(!W){ printf("    %-2s HF 读失败\n",mn[m]); continue; }
                float *Q=malloc((size_t)r*c*4); int qr=0,qc=0;
                if(vq_unpack_dequant(lf.vqmap+of[m],lf.vqmsz-of[m],Q,&qr,&qc)!=0||qr!=r||qc!=c){ printf("    %-2s 反量化失败\n",mn[m]); free(W); free(Q); continue; }
                /* Y0 = D·Xᵀ, Ysig = W·Xᵀ  (r × ne) */
                float *D=malloc((size_t)r*c*4); for(size_t i=0;i<(size_t)r*c;i++) D[i]=W[i]-Q[i];
                double *Y=malloc((size_t)r*ne*8); double sig=0;
                for(int i=0;i<r;i++){ const float*di=D+(size_t)i*c,*wi=W+(size_t)i*c;
                    for(int t=0;t<ne;t++){ const float*x=FIN+(size_t)rows[t]*DIM; double yd=0,yw=0;
                        for(int j=0;j<c;j++){ yd+=(double)di[j]*x[j]; yw+=(double)wi[j]*x[j]; }
                        Y[(size_t)i*ne+t]=yd; sig+=yw*yw; } }
                double err0=0; for(size_t i=0;i<(size_t)r*ne;i++) err0+=Y[i]*Y[i];
                printf("    %-2s 输出误差 ‖ΔW·X‖²/‖W·X‖²=%.4f | 复原 top-c 列后剩余:",mn[m],err0/sig);
                /* 逐档: Y_c = Y0 − Σ_{j∈topc} D[:,j]·x_j (增量, 累积到 512) */
                int kdone=0; for(int ci=0;ci<4;ci++){ int cc=CS[ci];
                    for(;kdone<cc;kdone++){ int j=ord[kdone]; for(int i=0;i<r;i++){ double dij=D[(size_t)i*c+j]; if(dij==0)continue; double*yi=Y+(size_t)i*ne; for(int t=0;t<ne;t++) yi[t]-=dij*FIN[(size_t)rows[t]*DIM+j]; } }
                    double errc=0; for(size_t i=0;i<(size_t)r*ne;i++) errc+=Y[i]*Y[i];
                    printf(" c=%d:%.1f%%",cc,100*errc/err0); }
                printf("\n"); fflush(stdout);
                free(W); free(Q); free(D); free(Y);
            }
            free(rows); free(Ee);
        }
        lfile_free(&lf); free(E); free(ord);
    }
    exit(0);
}

/* ═══ 子空间针(2026-09-02 用户批"打一针", 方案 C 最后一条有物理依据的线)═══
 * 通道能量是对角视角(上面 mc-probe: 均匀); 这里看协方差: 专家输入 Fin 的能量是否集中在低维子空间
 * (头前 h 前 64 主方向占 78.5%)。若是, 等体积联合设计="W 对子空间的作用存高精度 + 补空间低比特",
 * 有效输出误差 ≈ 补空间能量份额 × 补空间误差。两个数: ①Fin 协方差前 k 主方向能量份额(k=64/128/256)
 * ②把子空间部分设为精确后, 专家输出误差剩余 ‖ΔW·(I−PPᵀ)X‖²/‖ΔW·X‖²(对照 1−份额)。只读。 */
static const char *g_sub_probe = NULL;   /* --sub-probe "0,12,24,36,41" */
typedef struct { const double *G; const double *V; double *W; int D, k, i0, i1; } sp_mv_t;
static void *sp_mv_worker(void *a){ sp_mv_t *m=(sp_mv_t*)a;   /* W[i][:] = G[i][:]·V */
    for(int i=m->i0;i<m->i1;i++){ const double *gi=m->G+(size_t)i*m->D; double *wi=m->W+(size_t)i*m->k;
        for(int j=0;j<m->k;j++) wi[j]=0;
        for(int t=0;t<m->D;t++){ double g=gi[t]; if(g==0) continue; const double *vt=m->V+(size_t)t*m->k; for(int j=0;j<m->k;j++) wi[j]+=g*vt[j]; } }
    return NULL; }
/* 前 k 特征向量(列存 V[D][k]) + 特征值降序; 块幂迭代(多线程 G·V)+Gram-Schmidt */
static void sp_top_eigvec(const double *G,int D,int k,int iters,double *V,double *lam,int nth){
    double *W=malloc((size_t)D*k*8); unsigned s=777u;
    for(size_t i=0;i<(size_t)D*k;i++){ s=s*1103515245u+12345u; V[i]=((s>>8)&0xffff)/65536.0-0.5; }
    pthread_t th[32]; sp_mv_t mv[32]; if(nth>32)nth=32;
    for(int it=0;it<iters;it++){
        for(int t=0;t<nth;t++){ mv[t]=(sp_mv_t){G,V,W,D,k,D*t/nth,D*(t+1)/nth}; pthread_create(&th[t],NULL,sp_mv_worker,&mv[t]); }
        for(int t=0;t<nth;t++) pthread_join(th[t],NULL);
        for(int j=0;j<k;j++){
            for(int p=0;p<j;p++){ double d=0; for(int i=0;i<D;i++) d+=W[(size_t)i*k+j]*W[(size_t)i*k+p]; for(int i=0;i<D;i++) W[(size_t)i*k+j]-=d*W[(size_t)i*k+p]; }
            double n=0; for(int i=0;i<D;i++) n+=W[(size_t)i*k+j]*W[(size_t)i*k+j]; n=sqrt(n)+1e-300; for(int i=0;i<D;i++) W[(size_t)i*k+j]/=n; }
        memcpy(V,W,(size_t)D*k*8);
    }
    for(int t=0;t<nth;t++){ mv[t]=(sp_mv_t){G,V,W,D,k,D*t/nth,D*(t+1)/nth}; pthread_create(&th[t],NULL,sp_mv_worker,&mv[t]); }
    for(int t=0;t<nth;t++) pthread_join(th[t],NULL);
    for(int j=0;j<k;j++){ double acc=0; for(int i=0;i<D;i++) acc+=V[(size_t)i*k+j]*W[(size_t)i*k+j]; lam[j]=acc; }
    /* 按 λ 降序重排列(选择排序, k≤256) */
    for(int a=0;a<k;a++){ int b=a; for(int c2=a+1;c2<k;c2++) if(lam[c2]>lam[b]) b=c2;
        if(b!=a){ double t=lam[a]; lam[a]=lam[b]; lam[b]=t; for(int i=0;i<D;i++){ double v=V[(size_t)i*k+a]; V[(size_t)i*k+a]=V[(size_t)i*k+b]; V[(size_t)i*k+b]=v; } } }
    free(W);
}
typedef struct { const float *D; const float *X; int r, c, n, i0, i1; double *err; } sp_err_t;
static void *sp_err_worker(void *a){ sp_err_t *e=(sp_err_t*)a; double s=0;
    for(int i=e->i0;i<e->i1;i++){ const float *di=e->D+(size_t)i*e->c; for(int t=0;t<e->n;t++){ const float *x=e->X+(size_t)t*e->c; double y=0; for(int j=0;j<e->c;j++) y+=(double)di[j]*x[j]; s+=y*y; } }
    *e->err=s; return NULL; }
static double sp_err(const float *D,const float *X,int r,int c,int n,int nth){   /* ‖D·Xᵀ‖² 多线程 */
    pthread_t th[32]; sp_err_t ea[32]; double part[32]; if(nth>32)nth=32; if(nth>r)nth=r;
    for(int t=0;t<nth;t++){ ea[t]=(sp_err_t){D,X,r,c,n,r*t/nth,r*(t+1)/nth,&part[t]}; pthread_create(&th[t],NULL,sp_err_worker,&ea[t]); }
    double s=0; for(int t=0;t<nth;t++){ pthread_join(th[t],NULL); s+=part[t]; } return s; }
static void sub_probe(int S){
    if(!g_sub_probe) return;
    if(!ANC_OK||!ANC.fin||!ANC.ridx){ fprintf(stderr,"[SUBPROBE]★需 --anchor★\n"); exit(2); }
    if(!g_cli.layer_dir){ fprintf(stderr,"[SUBPROBE]★需 --layer-dir★\n"); exit(2); }
    const int EX[3]={0,128,255}; const int KS[3]={64,128,256}; const int KM=256, NEMAX=768; int nth=NTHREADS>1?NTHREADS:8;
    char b[256]; snprintf(b,sizeof b,"%s",g_sub_probe);
    for(char *tok=strtok(b,","); tok; tok=strtok(NULL,",")){ int L=atoi(tok);
        const float *FIN=ANC.fin+(size_t)L*S*DIM; const int32_t *RIDX=ANC.ridx+(size_t)L*S*NACT;
        /* ①协方差 G=XᵀX: 先转置成 Xt[D][S] 复用行 Gram */
        float *Xt=malloc((size_t)DIM*S*4); for(int s=0;s<S;s++) for(int j=0;j<DIM;j++) Xt[(size_t)j*S+s]=FIN[(size_t)s*DIM+j];
        double *G=malloc((size_t)DIM*DIM*8); { pthread_t th[32]; dws_gram_t gw[32]; int n2=nth>32?32:nth;
          for(int t=0;t<n2;t++){ int r0=(int)(DIM*sqrt((double)t/n2)), r1=(int)(DIM*sqrt((double)(t+1)/n2)); if(t==n2-1)r1=DIM; gw[t]=(dws_gram_t){Xt,DIM,S,r0,r1,G}; pthread_create(&th[t],NULL,dws_gram_worker,&gw[t]); }
          for(int t=0;t<n2;t++) pthread_join(th[t],NULL); }
        free(Xt);
        double tr=0; for(int i=0;i<DIM;i++) tr+=G[(size_t)i*DIM+i];
        double *V=malloc((size_t)DIM*KM*8), lam[256]; sp_top_eigvec(G,DIM,KM,30,V,lam,nth); free(G);
        double c64=0,c128=0,c256=0; for(int j=0;j<KM;j++){ if(j<64)c64+=lam[j]; if(j<128)c128+=lam[j]; c256+=lam[j]; }
        printf("[SUBPROBE] L%02d Fin 协方差前 k 主方向能量份额: k=64:%.1f%% k=128:%.1f%% k=256:%.1f%% (各向同性 k=256:%.1f%%; λ1 占 %.1f%%)\n",
               L,100*c64/tr,100*c128/tr,100*c256/tr,100.0*KM/DIM,100*lam[0]/tr); fflush(stdout);
        char lp[512]; snprintf(lp,sizeof lp,"%s/dql_L%02d.bin",g_cli.layer_dir,L);
        lfile_t lf; if(lfile_load(lp,&lf)!=0||!lf.vqmap){ fprintf(stderr,"[SUBPROBE]★L%d 层件缺★\n",L); free(V); continue; }
        const uint64_t *vtab=(const uint64_t*)(lf.vqmap+16);
        float *Pf=malloc((size_t)DIM*KM*4); for(size_t i=0;i<(size_t)DIM*KM;i++) Pf[i]=(float)V[i];   /* P[D][k] 列=主方向 */
        for(int xi=0;xi<3;xi++){ int e=EX[xi];
            int *rows=malloc(S*sizeof(int)); int ne=0;
            for(int s=0;s<S&&ne<NEMAX;s++){ for(int a=0;a<NACT;a++) if(RIDX[(size_t)s*NACT+a]==e){ rows[ne++]=s; break; } }
            if(ne<8){ printf("  e%03d 路由行仅 %d, 跳过\n",e,ne); free(rows); continue; }
            /* X_e 与三档补空间投影 X⊥_k = X − P_k P_kᵀ X */
            float *X=malloc((size_t)ne*DIM*4); for(int t=0;t<ne;t++) memcpy(X+(size_t)t*DIM,FIN+(size_t)rows[t]*DIM,(size_t)DIM*4);
            float *Xp[3]; double xe=0, xp[3]={0,0,0};
            for(int t=0;t<ne;t++) for(int j=0;j<DIM;j++) xe+=(double)X[(size_t)t*DIM+j]*X[(size_t)t*DIM+j];
            for(int ki=0;ki<3;ki++){ int k=KS[ki]; Xp[ki]=malloc((size_t)ne*DIM*4);
                for(int t=0;t<ne;t++){ const float *x=X+(size_t)t*DIM; float *o=Xp[ki]+(size_t)t*DIM; double coef[256];
                    for(int j=0;j<k;j++){ double sacc=0; for(int i=0;i<DIM;i++) sacc+=(double)Pf[(size_t)i*KM+j]*x[i]; coef[j]=sacc; }
                    for(int i=0;i<DIM;i++){ double sacc=0; for(int j=0;j<k;j++) sacc+=coef[j]*Pf[(size_t)i*KM+j]; o[i]=(float)(x[i]-sacc); xp[ki]+=(double)o[i]*o[i]; } } }
            printf("  e%03d 行=%d 该专家 x 在补空间的能量份额: k=64:%.1f%% k=128:%.1f%% k=256:%.1f%%\n",e,ne,100*xp[0]/xe,100*xp[1]/xe,100*xp[2]/xe); fflush(stdout);
            const char *mn[2]={"w1","w3"}; uint64_t of[2]={vtab[(size_t)e*3],vtab[(size_t)e*3+1]};
            for(int m=0;m<2;m++){
                if(!of[m]){ printf("    %-2s 冷槽跳过\n",mn[m]); continue; }
                char n[160]; snprintf(n,sizeof n,"layers.%d.ffn.experts.%d.%s.weight",L,e,mn[m]);
                long r,c; float *W=st_read_weight(&C,n,&r,&c); if(!W){ printf("    %-2s HF 读失败\n",mn[m]); continue; }
                float *Q=malloc((size_t)r*c*4); int qr=0,qc=0;
                if(vq_unpack_dequant(lf.vqmap+of[m],lf.vqmsz-of[m],Q,&qr,&qc)!=0||qr!=r||qc!=c){ printf("    %-2s 反量化失败\n",mn[m]); free(W); free(Q); continue; }
                float *Dm=malloc((size_t)r*c*4); for(size_t i=0;i<(size_t)r*c;i++) Dm[i]=W[i]-Q[i];
                double err0=sp_err(Dm,X,(int)r,(int)c,ne,nth), sig=sp_err(W,X,(int)r,(int)c,ne,nth);
                printf("    %-2s ‖ΔW·X‖²/‖W·X‖²=%.4f | 子空间精确后剩余: ",mn[m],err0/sig);
                for(int ki=0;ki<3;ki++){ double ek=sp_err(Dm,Xp[ki],(int)r,(int)c,ne,nth); printf("k=%d:%.1f%%(x补空间%.1f%%) ",KS[ki],100*ek/err0,100*xp[ki]/xe); }
                printf("\n"); fflush(stdout);
                free(W); free(Q); free(Dm);
            }
            free(X); for(int ki=0;ki<3;ki++) free(Xp[ki]); free(rows);
        }
        lfile_free(&lf); free(V); free(Pf);
    }
    exit(0);
}

/* ═══ 子空间联合量化针(2026-09-02 用户令"哪怕一点点提升都应该尝试")═══
 * 把子空间针的估算换成生产量化器的真数, 矩阵级, held 行判:
 *   高部 H = W·P_k(k=256, P_k=该层 Fin 协方差主方向, 层内共享) → 逐行 absmax 定点(int4 / int8);
 *   余部 R = W − Ĥ·P_kᵀ → 生产 VQ(dq_quant_expert_vq: kmeans+GPTQ 反馈+通道权) 在等体积码本档量化,
 *   重要性权重喂 x⊥ = x − P P^T x(余部实际看到的输入); 重建 Ŵ' = Ĥ·P_kᵀ + R̂。
 * 等体积档(每矩阵 2048×4096, 现 v4x512 = 2.258 bpw): int4 高部 2.13 Mbit ⇒ 余部 v4x256(2.01 bpw);
 *   int8 高部 4.23 Mbit ⇒ 余部 v4x128(1.76 bpw)。
 * 判据 = held 行输出误差 ‖(W−Ŵ')X_h‖² 对 ①盘上 dql_vq ②同流程重量化 v4x512 两个基线的比值。只读。 */
static const char *g_subq_probe = NULL;   /* --sub-quant-probe "0,12,24,36,41" */
static void subq_rowquant(const float *H,int r,int k,int levels,float *Hq){   /* 逐行对称定点: levels=7(int4)/127(int8) */
    for(int i=0;i<r;i++){ const float *h=H+(size_t)i*k; float *q=Hq+(size_t)i*k; float am=0;
        for(int j=0;j<k;j++){ float a=fabsf(h[j]); if(a>am)am=a; }
        float sc=am>0?am/levels:1.0f;
        for(int j=0;j<k;j++){ float v=h[j]/sc; int iv=(int)floorf(v+0.5f); if(iv>levels)iv=levels; if(iv<-levels)iv=-levels; q[j]=iv*sc; } }
}
static double subq_err(const float *W,const float *Wq,const float *Xh,int nh,int r,int c){   /* ‖(W−Wq)·Xhᵀ‖² */
    float *Dm=malloc((size_t)r*c*4); for(size_t i=0;i<(size_t)r*c;i++) Dm[i]=W[i]-Wq[i];
    float *E=malloc((size_t)nh*r*4); dq_matmul(Xh,Dm,E,nh,c,r);
    double s=0; for(size_t i=0;i<(size_t)nh*r;i++) s+=(double)E[i]*E[i]; free(Dm); free(E); return s;
}
static void sub_quant_probe(int S){
    if(!g_subq_probe) return;
    if(!ANC_OK||!ANC.fin||!ANC.ridx){ fprintf(stderr,"[SUBQ]★需 --anchor★\n"); exit(2); }
    if(!g_cli.layer_dir){ fprintf(stderr,"[SUBQ]★需 --layer-dir★\n"); exit(2); }
    const int EX[3]={0,128,255}, K=256, NEMAX=768; int nth=NTHREADS>1?NTHREADS:8;
    printf("[SUBQ] 体积账(每矩阵 2048×4096 bit): 基线 v4x512=%.2fM | int4高部 %.2fM+v4x256 %.2fM=%.2fM | int8高部 %.2fM+v4x128 %.2fM=%.2fM\n",
           vq_payload_bytes(MOEI,DIM,4,512)*8/1e6, (2048.0*256*4+2048*16)/1e6, vq_payload_bytes(MOEI,DIM,4,256)*8/1e6, (2048.0*256*4+2048*16)/1e6+vq_payload_bytes(MOEI,DIM,4,256)*8/1e6,
           (2048.0*256*8+2048*16)/1e6, vq_payload_bytes(MOEI,DIM,4,128)*8/1e6, (2048.0*256*8+2048*16)/1e6+vq_payload_bytes(MOEI,DIM,4,128)*8/1e6);
    char b[256]; snprintf(b,sizeof b,"%s",g_subq_probe);
    double agg[4]={0,0,0,0}; int nagg=0;   /* 汇总: Σ err(dql, 512, V1, V2) 跨矩阵 */
    for(char *tok=strtok(b,","); tok; tok=strtok(NULL,",")){ int L=atoi(tok);
        const float *FIN=ANC.fin+(size_t)L*S*DIM; const int32_t *RIDX=ANC.ridx+(size_t)L*S*NACT;
        float *Xt=malloc((size_t)DIM*S*4); for(int s=0;s<S;s++) for(int j=0;j<DIM;j++) Xt[(size_t)j*S+s]=FIN[(size_t)s*DIM+j];
        double *G=malloc((size_t)DIM*DIM*8); { pthread_t th[32]; dws_gram_t gw[32]; int n2=nth>32?32:nth;
          for(int t=0;t<n2;t++){ int r0=(int)(DIM*sqrt((double)t/n2)), r1=(int)(DIM*sqrt((double)(t+1)/n2)); if(t==n2-1)r1=DIM; gw[t]=(dws_gram_t){Xt,DIM,S,r0,r1,G}; pthread_create(&th[t],NULL,dws_gram_worker,&gw[t]); }
          for(int t=0;t<n2;t++) pthread_join(th[t],NULL); }
        free(Xt);
        double *V=malloc((size_t)DIM*K*8), lam[256]; sp_top_eigvec(G,DIM,K,30,V,lam,nth); free(G);
        float *P=malloc((size_t)DIM*K*4), *Pt=malloc((size_t)K*DIM*4);   /* P[D][k] 行主; Pt[k][D] */
        for(int i=0;i<DIM;i++) for(int j=0;j<K;j++){ P[(size_t)i*K+j]=(float)V[(size_t)i*K+j]; Pt[(size_t)j*DIM+i]=(float)V[(size_t)i*K+j]; }
        free(V);
        char lp[512]; snprintf(lp,sizeof lp,"%s/dql_L%02d.bin",g_cli.layer_dir,L);
        lfile_t lf; if(lfile_load(lp,&lf)!=0||!lf.vqmap){ fprintf(stderr,"[SUBQ]★L%d 层件缺★\n",L); free(P); free(Pt); continue; }
        const uint64_t *vtab=(const uint64_t*)(lf.vqmap+16);
        for(int xi=0;xi<3;xi++){ int e=EX[xi];
            int *rows=malloc(S*sizeof(int)); int ne=0;
            for(int s=0;s<S&&ne<NEMAX;s++){ for(int a=0;a<NACT;a++) if(RIDX[(size_t)s*NACT+a]==e){ rows[ne++]=s; break; } }
            if(ne<16){ printf("  e%03d 路由行仅 %d, 跳过\n",e,ne); free(rows); continue; }
            int nf=ne*3/4, nh=ne-nf;   /* fit 行喂量化器, held 行判 */
            float *X=malloc((size_t)ne*DIM*4); for(int t=0;t<ne;t++) memcpy(X+(size_t)t*DIM,FIN+(size_t)rows[t]*DIM,(size_t)DIM*4);
            /* x⊥ (余部重要性): x − P Pᵀ x */
            float *Xp=malloc((size_t)ne*DIM*4); { float *cf=malloc((size_t)ne*K*4); dq_matmul(X,Pt,cf,ne,DIM,K);   /* cf = X·P */
              float *rec=malloc((size_t)ne*DIM*4); dq_matmul(cf,P,rec,ne,K,DIM);                                    /* rec = cf·Pᵀ */
              for(size_t i=0;i<(size_t)ne*DIM;i++) Xp[i]=X[i]-rec[i]; free(cf); free(rec); }
            const float *Xf=X, *Xh=X+(size_t)nf*DIM, *Xpf=Xp;
            const char *mn[2]={"w1","w3"}; uint64_t of[2]={vtab[(size_t)e*3],vtab[(size_t)e*3+1]};
            for(int m=0;m<2;m++){
                if(!of[m]){ printf("  L%02d e%03d %-2s 冷槽跳过\n",L,e,mn[m]); continue; }
                char n[160]; snprintf(n,sizeof n,"layers.%d.ffn.experts.%d.%s.weight",L,e,mn[m]);
                long r,c; float *W=st_read_weight(&C,n,&r,&c); if(!W){ printf("  %-2s HF 读失败\n",mn[m]); continue; }
                float *Qd=malloc((size_t)r*c*4); int qr=0,qc=0;
                if(vq_unpack_dequant(lf.vqmap+of[m],lf.vqmsz-of[m],Qd,&qr,&qc)!=0||qr!=r||qc!=c){ printf("  %-2s 反量化失败\n",mn[m]); free(W); free(Qd); continue; }
                double t0=vqt_now();
                float *Yref=malloc((size_t)nh*r*4); dq_matmul(Xh,W,Yref,nh,(int)c,(int)r); double sig=0; for(size_t i=0;i<(size_t)nh*r;i++) sig+=(double)Yref[i]*Yref[i];
                /* held 行的结构化输入: x_k = Pᵀx (nh×k), x⊥ = x − P x_k —— 设计里余部只吃 x⊥, 主方向只经 Ĥ */
                float *Xhk=malloc((size_t)nh*K*4); dq_matmul(Xh,Pt,Xhk,nh,(int)c,K);
                float *Xhp=malloc((size_t)nh*c*4); { float *rec=malloc((size_t)nh*c*4); dq_matmul(Xhk,P,rec,nh,K,(int)c); for(size_t i=0;i<(size_t)nh*c;i++) Xhp[i]=Xh[i]-rec[i]; free(rec); }
                double e_dql=subq_err(W,Qd,Xh,nh,(int)r,(int)c);
                float *Q512=dq_quant_expert_vq(W,(int)r,(int)c,Xf,nf,4,512); double e_512=subq_err(W,Q512,Xh,nh,(int)r,(int)c); free(Q512);
                /* 高部 H=W·P (r×k) */
                float *H=malloc((size_t)r*K*4); dq_matmul(W,Pt,H,(int)r,(int)c,K);
                double eV[2]; const int LV[2]={7,127}; const int NC[2]={256,128};
                for(int v=0;v<2;v++){
                    float *Hq=malloc((size_t)r*K*4); subq_rowquant(H,(int)r,K,LV[v],Hq);
                    float *HP=malloc((size_t)r*c*4); dq_matmul(Hq,P,HP,(int)r,K,(int)c);           /* Ĥ·Pᵀ */
                    float *R=malloc((size_t)r*c*4); for(size_t i=0;i<(size_t)r*c;i++) R[i]=W[i]-HP[i];
                    float *Rq=dq_quant_expert_vq(R,(int)r,(int)c,Xpf,nf,4,NC[v]);                  /* 余部: 重要性=x⊥ */
                    /* 结构化前向: y = Ĥ·x_k + R̂·x⊥ ; 误差 = Yref − Y1 − Y2 */
                    float *Y1=malloc((size_t)nh*r*4); dq_matmul(Xhk,Hq,Y1,nh,K,(int)r);
                    float *Y2=malloc((size_t)nh*r*4); dq_matmul(Xhp,Rq,Y2,nh,(int)c,(int)r);
                    double es=0; for(size_t i=0;i<(size_t)nh*r;i++){ double d=(double)Yref[i]-Y1[i]-Y2[i]; es+=d*d; } eV[v]=es;
                    free(Y1); free(Y2);
                    free(Hq); free(HP); free(R); free(Rq); }
                printf("  L%02d e%03d %-2s held=%d | 相对误差: dql %.4f | 同流程512 %.4f | int4+v256 %.4f (%.0f%% of 512) | int8+v128 %.4f (%.0f%% of 512) | %.0fs\n",
                       L,e,mn[m],nh,e_dql/sig,e_512/sig,eV[0]/sig,100*eV[0]/e_512,eV[1]/sig,100*eV[1]/e_512,vqt_now()-t0); fflush(stdout);
                agg[0]+=e_dql/sig; agg[1]+=e_512/sig; agg[2]+=eV[0]/sig; agg[3]+=eV[1]/sig; nagg++;
                free(W); free(Qd); free(H); free(Yref); free(Xhk); free(Xhp);
            }
            free(X); free(Xp); free(rows);
        }
        lfile_free(&lf); free(P); free(Pt);
    }
    if(nagg) printf("[SUBQ] 汇总(%d 阵, 相对误差均值): dql %.4f | 同流程512 %.4f | int4+v256 %.4f (%.0f%%) | int8+v128 %.4f (%.0f%%)\n",
                    nagg,agg[0]/nagg,agg[1]/nagg,agg[2]/nagg,100*agg[2]/agg[1],agg[3]/nagg,100*agg[3]/agg[1]);
    exit(0);
}
