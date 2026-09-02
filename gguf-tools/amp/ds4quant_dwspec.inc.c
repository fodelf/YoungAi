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
