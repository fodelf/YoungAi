/* ===================== 专家量化 + 应用 ===================== *
 * 量化三矩阵(档位 cfg)并把 routed 输出 scatter-add 进 dst[S,DIM]。
 * Xc/ncal = anchor FP 校准行(部署口径: 离线量化器只见 FP 激活);
 * w2 的校准 hidden 用【量化后 q1/q3】作用于校准行(顺序误差补偿, 与逐矩阵顺序量化一致)。 */
#define MV_HOTK 16   /* 冠军热档专家数(g10h16 档梯/影子链用) */

static void co_hc(const float*X,int n,const float*q1,const float*q3,float*hc);
/* ★不得 static 缓存(2026-08-01 实锤 bug): 计划表 vq_rplan(L) 每层 setenv 改 w2 档, 一次性
 * 缓存会把全 43 层锁死在 L00 的值 —— 实测 L18 起 w2 该 24×256 却仍用 16×256, 每层多 38 MiB,
 * 全模型超支 1.27 GiB(29.2 vs 目标 28), 且会在 L38 撞体积闸掐断整轮。冷档 vq_cold_dim()
 * 无此病(它读每层更新的全局 g_vq_dim)。每次 getenv 的开销相对一层 VQ 编码可忽略。 */
static int vq_w2_dim(void){ const char*e=getenv("DS4_VQ_W2_DIM"); int v=e?atoi(e):0; if(v&&v!=4&&v!=8&&v!=12&&v!=16&&v!=24&&v!=32) v=0; return v; }
/* ★R28 每层计划(DS4_VQ_RPLAN, 2026-07-31): "L=%d dim=%d nc=%d hot=%d w2dim=%d w2nc=%d"
 * 动态层大小(每层不同码本档) + 动态冷热(每层不同热专家数)。层序处理, 单全局安全。 */
static int g_vq_L=-1; int g_vq_dim=0, g_vq_nc=0; static int g_vq_hot=0;
/* 热档档位(2026-08-27): 用全局不用 env —— 本项目禁新增 env 配置(铁律 08-22)。
 * 缺省 4/512 = 历史行为; 计划表带 hotdim/hotnc 才改。 */
static int g_vq_hot_dim=4, g_vq_hot_nc=512;
static int vq_hot_dim(void){ return g_vq_hot_dim; }
static int vq_hot_nc(void){ return g_vq_hot_nc; }
static int vq_rplan(int L){
    const char*pp=getenv("DS4_VQ_RPLAN"); if(!pp) return 0;
    FILE*f=fopen(pp,"r"); if(!f){ fprintf(stderr,"[R28] 计划表打不开 %s\n",pp); exit(2); }
    char ln[256]; int l,d,n,h,wd,wn,ok=0;
    while(fgets(ln,sizeof(ln),f))
        if(sscanf(ln,"L=%d dim=%d nc=%d hot=%d w2dim=%d w2nc=%d",&l,&d,&n,&h,&wd,&wn)==6 && l==L){
            g_vq_L=L; g_vq_dim=d; g_vq_nc=n; g_vq_hot=h; ok=1;
            /* 可选热档字段(缺省 4/512 = 历史逐字节不变)。放在行尾追加而不是改列序,
             * 老计划表照读; 新表想让热专家更密就写 hotdim=4 hotnc=1024。 */
            { const char*hp=strstr(ln,"hotdim="); int hd,hn;
              if(hp && sscanf(hp,"hotdim=%d hotnc=%d",&hd,&hn)==2){ g_vq_hot_dim=hd; g_vq_hot_nc=hn; }
              else { g_vq_hot_dim=4; g_vq_hot_nc=512; } }
            char b[16]; snprintf(b,16,"%d",wd); setenv("DS4_VQ_W2_DIM",b,1);
            snprintf(b,16,"%d",wn); setenv("DS4_VQ_W2_NC",b,1);
            break; }
    fclose(f);
    if(!ok){ fprintf(stderr,"[R28] 计划表缺 L%d\n",L); exit(2); }
    fprintf(stderr,"[R28] L%02d 冷档 vq%dx%d 热%d(vq%dx%d) w2 vq%dx%d\n",L,d,n,h,g_vq_hot_dim,g_vq_hot_nc,wd,wn);
    return 1;
}
static int vq_w2_nc(void){ const char*e=getenv("DS4_VQ_W2_NC"); int v=e?atoi(e):256; if(v<16||v>4096) v=256; return v; }   /* 同上: 禁 static 缓存, 计划表逐层改档 */
static int hot_from_anchor(int L,int S,int K);   /* 前置声明: 回放自动 arm 用(定义在贪心段) */
static void quant_apply(char cfg,const float*e1,const float*e3,const float*e2,
                        const float*Xc,int ncal,const float*xs,const float*wwv,int nt,
                        const int*toks,float*dst,int L,int e){
    float *q1,*q3,*q2;
    const float *X=(cfg=='n')?NULL:Xc; int nx=(cfg=='n')?0:ncal;
    int g2hot=g2_hot_slot(L,e)>=0;   /* 热专家: go2b 合并2bit(残差+量化一体), 档位不适用 */
    if(g2hot){ q1=dq_quant_expert_go2b(e1,MOEI,DIM,X,nx); q3=dq_quant_expert_go2b(e3,MOEI,DIM,X,nx); }
    else if(cfg=='z'){ q1=dq_quant_expert_lowrank(e1,MOEI,DIM,X,nx,ZK); q3=dq_quant_expert_lowrank(e3,MOEI,DIM,X,nx,ZK); }
    else if(cfg=='r'){ q1=dq_quant_expert_rowscale(e1,MOEI,DIM,X,nx); q3=dq_quant_expert_rowscale(e3,MOEI,DIM,X,nx); }
    else if(cfg=='g'){ q1=dq_quant_expert_signref(e1,MOEI,DIM,X,nx); q3=dq_quant_expert_signref(e3,MOEI,DIM,X,nx); }
    else { int lv=(cfg=='2')?1:(cfg=='3')?2:0;
           q1=dq_quant_expert_resid(e1,MOEI,DIM,X,nx,lv); q3=dq_quant_expert_resid(e3,MOEI,DIM,X,nx,lv); }
    float *hc=NULL; int nhc=0;
    if(nx>0){
        hc=malloc((size_t)nx*MOEI*4);
        float *gg=malloc((size_t)nx*MOEI*4),*uu=malloc((size_t)nx*MOEI*4);
        dq_matmul(X,q1,gg,nx,DIM,MOEI); dq_matmul(X,q3,uu,nx,DIM,MOEI);
        for(size_t i=0;i<(size_t)nx*MOEI;i++){ float g2=gg[i],u2=uu[i];
            if(SWLIM>0){ if(u2>SWLIM)u2=SWLIM; if(u2<-SWLIM)u2=-SWLIM; if(g2>SWLIM)g2=SWLIM; }
            hc[i]=dq_silu(g2)*u2; }
        free(gg);free(uu); nhc=nx;
    }
    if(g2hot) q2=dq_quant_expert_go2b(e2,DIM,MOEI,hc,nhc);
    else if(cfg=='z') q2=dq_quant_expert_lowrank(e2,DIM,MOEI,hc,nhc,ZK);
    else if(cfg=='r') q2=dq_quant_expert_rowscale(e2,DIM,MOEI,hc,nhc);
    else if(cfg=='g') q2=dq_quant_expert_signref(e2,DIM,MOEI,hc,nhc);
    else { int lv=(cfg=='2')?1:(cfg=='3')?2:0; q2=dq_quant_expert_resid(e2,DIM,MOEI,hc,nhc,lv); }
    if(hc)free(hc);
    float *aq=calloc((size_t)nt*DIM,4);
    dq_expert_fp(xs,q1,q3,q2,wwv,aq,nt,DIM,MOEI,SWLIM);
    for(int i=0;i<nt;i++)for(int d=0;d<DIM;d++) dst[(size_t)toks[i]*DIM+d]+=aq[(size_t)i*DIM+d];
    free(q1);free(q3);free(q2);free(aq);
}

/* 并行专家 worker: 原子计数器分发 e, 各线程 scatter 进私有 partial buffer, 主线程归约。
 * st_read/量化核心无全局可变状态(已核), 线程安全。 */
typedef struct {
    int L,S,n_fit,do_quant; char cfg;
    const float *Fin; const int *idx; const float *rw_act;
    int *e_next;
    float *fout;        /* [S*DIM] 选定档 routed 累积 (私有) */
    float *ffp;         /* [S*DIM] FP routed (局部质量对照; NULL=不算) */
    float *abl[4];      /* 消融档 n/1/z/2 (DS4_ABLATE; NULL=不算) */
    double calib_rows; int calib_empty,nhit;
} ework_t;

static void *expert_worker(void*arg){
    ework_t *w=(ework_t*)arg;
    int S=w->S, L=w->L, n_fit=w->n_fit;
    int *toks=malloc((size_t)S*sizeof(int)); float *wwv=malloc((size_t)S*sizeof(float));
    float *xs=malloc((size_t)S*DIM*4);
    float *Xc=(w->do_quant&&ANC_OK)?malloc((size_t)(n_fit>0?n_fit:1)*DIM*4):NULL;
    for(;;){
        int e=__sync_fetch_and_add(w->e_next,1); if(e>=NEXP) break;
        int nt=0;
        for(int s=0;s<S;s++)for(int a=0;a<NACT_RT;a++)
            if(w->idx[(size_t)s*NACT_RT+a]==e){toks[nt]=s;wwv[nt]=w->rw_act[(size_t)s*NACT_RT+a];nt++;}
        if(!nt) continue;
        if((e&31)==0) fputc('.',stderr);
        char n1[160],n3[160],n2[160];
        snprintf(n1,sizeof(n1),"layers.%d.ffn.experts.%d.w1.weight",L,e);
        snprintf(n3,sizeof(n3),"layers.%d.ffn.experts.%d.w3.weight",L,e);
        snprintf(n2,sizeof(n2),"layers.%d.ffn.experts.%d.w2.weight",L,e);
        long rr,cc; float *e1=st_read_weight(&C,n1,&rr,&cc),*e3=st_read_weight(&C,n3,&rr,&cc),*e2=st_read_weight(&C,n2,&rr,&cc);
        if(!e1||!e3||!e2){ fprintf(stderr,"\n[!] L%d e%d 权重读失败\n",L,e); if(e1)free(e1); if(e3)free(e3); if(e2)free(e2); continue; }
        for(int i=0;i<nt;i++) memcpy(xs+(size_t)i*DIM, w->Fin+(size_t)toks[i]*DIM, (size_t)DIM*4);
        if(!w->do_quant || w->cfg=='F'){
            float *af=calloc((size_t)nt*DIM,4); dq_expert_fp(xs,e1,e3,e2,wwv,af,nt,DIM,MOEI,SWLIM);
            for(int i=0;i<nt;i++)for(int d=0;d<DIM;d++) w->fout[(size_t)toks[i]*DIM+d]+=af[(size_t)i*DIM+d];
            free(af);
        } else {
            if(w->ffp){
                float *af=calloc((size_t)nt*DIM,4); dq_expert_fp(xs,e1,e3,e2,wwv,af,nt,DIM,MOEI,SWLIM);
                for(int i=0;i<nt;i++)for(int d=0;d<DIM;d++) w->ffp[(size_t)toks[i]*DIM+d]+=af[(size_t)i*DIM+d];
                free(af);
            }
            /* anchor 校准行: FP 激活 + FP 路由 + 只 fit 行 (held 行不喂校准, 防泄漏) */
            int ncal=0;
            if(ANC_OK&&Xc){
                const float *afin=ANC.fin+(size_t)L*S*DIM;
                const int32_t *aidx=ANC.ridx+(size_t)L*S*NACT;
                /* DS4_CALIB_FULLSET(2026-07-27 g_r 饿死审计): 命中过滤在 n_fit~400 时每专家
                 * 仅~9行 → GPTQ-H 是 rank-9 残料, g_r 被 lam=sp2/(n+1) 钉死在 1(实测全模型
                 * g_r≈1.000)。全集喂入(ncal=n_fit)统计充分, lam 公式随 n 自愈。默认保持旧行为。 */
                static int fullset=-1; if(fullset<0) fullset=getenv("DS4_CALIB_FULLSET")?1:0;
                /* DS4_CALIB_CAP(2026-08-18 用户问"语料需要那么多吗"): FULLSET 每专家全集
                 * 2906 行是对"9行饿死"的矫枉过正 — H 只有 128 维, 512 行=4×过采样已统计
                 * 充分, 而 GPTQ-H/g_r 两个最重段都 ∝ 行数。激活行全保, 全集行均匀 stride
                 * 补到上限。0/未设=原全集行为。 */
                static int calcap=-1; if(calcap<0){ const char*cc=getenv("DS4_CALIB_CAP"); calcap=cc?atoi(cc):0; }
                if(fullset&&calcap>0&&n_fit>calcap){
                    int nact_e=0;   /* 先收激活行(全保) */
                    for(int s=0;s<n_fit;s++){ int hit=0;
                        for(int a=0;a<NACT;a++) if(aidx[(size_t)s*NACT+a]==e){hit=1;break;}
                        if(hit){ memcpy(Xc+(size_t)ncal*DIM, afin+(size_t)s*DIM, (size_t)DIM*4); ncal++; nact_e++; } }
                    int need=calcap-nact_e;
                    if(need>0){ int stride=n_fit/need; if(stride<1)stride=1;
                        for(int s=0;s<n_fit&&need>0;s+=stride){ int hit=0;
                            for(int a=0;a<NACT;a++) if(aidx[(size_t)s*NACT+a]==e){hit=1;break;}
                            if(!hit){ memcpy(Xc+(size_t)ncal*DIM, afin+(size_t)s*DIM, (size_t)DIM*4); ncal++; need--; } } }
                } else
                for(int s=0;s<n_fit;s++){ int hit=fullset;
                    if(!hit) for(int a=0;a<NACT;a++) if(aidx[(size_t)s*NACT+a]==e){hit=1;break;}
                    if(hit){ memcpy(Xc+(size_t)ncal*DIM, afin+(size_t)s*DIM, (size_t)DIM*4); ncal++; } }
            }
            w->nhit++; w->calib_rows+=ncal; if(!ncal) w->calib_empty++;
            quant_apply(w->cfg,e1,e3,e2,ncal?Xc:NULL,ncal,xs,wwv,nt,toks,w->fout,L,e);
            if(w->abl[0]){ const char AC[4]={'n','1','z','2'};
                for(int ci=0;ci<4;ci++) quant_apply(AC[ci],e1,e3,e2,ncal?Xc:NULL,ncal,xs,wwv,nt,toks,w->abl[ci],L,e); }
        }
        free(e1);free(e3);free(e2);
    }
    free(toks);free(wwv);free(xs); if(Xc)free(Xc);
    dq_gpu_thread_release();   /* 每层新建线程: __thread CUDA 资源不释放=逐层泄漏(32k 实锤) */
    return NULL;
}

/* ===================== 一层前向 ===================== *
 * 就地更新 H[S,HCM,DIM]。do_quant: routed 专家按 cfg 档量化(校准=anchor 部署口径), 误差随深度累积。
 * st(可NULL): 路由一致率/校准统计/局部质量。 */
typedef struct { double agree; double calib_rows; int calib_empty,nhit;
                 int have_loc; double loc_r2; double rs_ratio; int zk;
                 double abl_r2[4],abl_rel[2],abl_la,abl_lc; } lstat_t;

/* ===================== 每层修正算法搜索 + 合并动态侧车 (DS4_COADAPT) ===================== *
 * 用户设计(2026-07-10 定稿口径): 每层的任务是【找到那一层最好的算法】——不存在"没提升
 * 就丢弃/判0"的场景; 某候选无增益只说明还没找到, 换形态继续搜; 每层胜者可以不同;
 * 最后把所有层的动态修正【合并生成一个很小体积的文件】(z+四损失+向后+感知)叠在 1bit 底座上。
 *
 * 实现: base 量化一次(signref, anchor 校准)+过程数据缓存(hc_cal/hc_all/y2ref/yq_all/命中行),
 * 之后每个修正候选都在缓存上闭式求解(秒级), 逐层搜索菜单:
 *   GL   静态层标量 g_L(1 dof, 零方差, 必然非负基准)      —— 目标=累积漂移 DF
 *   GE   静态每专家增益 g_e(256 dof, 低方差)               —— 目标=自误差 / 累积 两版
 *   CE   动态每专家系数 c_e(x)=1+αᵀk(x) (kernel-ridge, λ 扫) —— 目标=自误差 / 累积 两版
 *   ZL   加法低秩 z^L(Fin→DF, 四损失求解+选秩, k/λ 扫)      —— 累积回拉
 *   组合  乘法胜者之上再解加法余量
 * 四损失=评审与正则(L_align+0.5·L_classify 评分, λ=L_fixed, clamp=L_smooth);
 * 感知=classify 维度方差加权; 向后=胜者落地后 H 传下层 + 1 轮 base 重解(D3)择优。
 * val(fit 内留出)=选择判据, held 只观测; 每层落【菜单内 val 最优】(GL 兜底, 永不空手)。
 * 胜者 payload 记入合并侧车(DQZ1): 逐层 {algo,params,数据}, 结束写盘并报体积。 */
typedef struct {
    int ncal,nhit;
    float *hc_cal;   /* [ncal×MOEI] anchor FP 校准行过量化 q1/q3 的 hidden(顺序补偿口径, 不缩放) */
    int   *calS;     /* [ncal] 校准行行号 */
    float *calW;     /* [ncal] anchor 路由权重 */
    float *hc_all;   /* [nhit×MOEI] 实际激活行(污染 Fin)过 q1/q3 */
    int   *hitS; float *hitW;
    float *y2ref;    /* [ncal×DIM] e2fp·hc_cal — w2 顺序补偿的 FP 目标(乘法系数的参照) */
    float *yq_all;   /* [nhit×DIM] 当前 w2q·hc_all — 应用行输出(系数在外面乘) */
    float *yq_cal;   /* [ncal×DIM] 当前 w2q·hc_cal — 校准行输出(累积目标候选用) */
    float *gcal;     /* [ncal] 每校准行自误差最优标量 ⟨y2ref,ŷq⟩/⟨ŷq,ŷq⟩ */
    float *gden;     /* [ncal] ⟨ŷq,ŷq⟩ (静态增益池化/累积目标的分母) */
    float *ccal;     /* [ncal] 胜者系数在校准行的值(向后重解目标用); NULL=c≡1 */
    float *alpha;    /* [ncal] z_e kernel 权(dual): c(x)=1+Σ_t α_t⟨x_t,x⟩ */
    float gstat;     /* 静态每专家增益候选(GE); 0=关, 非0 时优先于 alpha */
} coexp_t;
typedef struct {
    int L,S,n_fit,iter;
    const float *Fin; const int *idx; const float *rw;
    coexp_t *ex; int *e_next;
    float *fout;
    double calib_rows; int calib_empty,nhit_experts; long nflip;
} cowork_t;
static void co_hc(const float*X,int n,const float*q1,const float*q3,float*hc){
    float *gg=malloc((size_t)n*MOEI*4),*uu=malloc((size_t)n*MOEI*4);
    dq_matmul(X,q1,gg,n,DIM,MOEI); dq_matmul(X,q3,uu,n,DIM,MOEI);
    for(size_t i=0;i<(size_t)n*MOEI;i++){ float g2=gg[i],u2=uu[i];
        if(SWLIM>0){ if(u2>SWLIM)u2=SWLIM; if(u2<-SWLIM)u2=-SWLIM; if(g2>SWLIM)g2=SWLIM; }
        hc[i]=dq_silu(g2)*u2; }
    free(gg);free(uu);
}
/* w2 解完的收尾: 应用行输出 yq_all(存, 系数在主循环外乘)、c=1 scatter 进 fout(基准)、
 * 每校准行最优标量 gcal = ⟨y2ref,ŷq⟩/⟨ŷq,ŷq⟩ (z_e 的拟合目标; ŷq 用不缩放 hc_cal)。 */
static void co_w2_finish(coexp_t*ce,const float*w2q,float*fout){
    if(ce->nhit>0){
        if(!ce->yq_all) ce->yq_all=malloc((size_t)ce->nhit*DIM*4);
        dq_matmul(ce->hc_all,w2q,ce->yq_all,ce->nhit,MOEI,DIM);
        for(int i=0;i<ce->nhit;i++){ float wgt=ce->hitW[i]; float*dst=fout+(size_t)ce->hitS[i]*DIM;
            const float*yi=ce->yq_all+(size_t)i*DIM;
            for(int d2=0;d2<DIM;d2++) dst[d2]+=wgt*yi[d2]; }
    }
    if(ce->ncal>0&&ce->y2ref){
        if(!ce->gcal) ce->gcal=malloc((size_t)ce->ncal*4);
        if(!ce->gden) ce->gden=malloc((size_t)ce->ncal*4);
        if(!ce->yq_cal) ce->yq_cal=malloc((size_t)ce->ncal*DIM*4);
        dq_matmul(ce->hc_cal,w2q,ce->yq_cal,ce->ncal,MOEI,DIM);
        for(int t=0;t<ce->ncal;t++){
            const float*a=ce->y2ref+(size_t)t*DIM,*b=ce->yq_cal+(size_t)t*DIM;
            double num=0,den=0;
            for(int d2=0;d2<DIM;d2++){ num+=(double)a[d2]*b[d2]; den+=(double)b[d2]*b[d2]; }
            float g=(den>1e-20)?(float)(num/den):1.0f;
            if(g<0.25f)g=0.25f; if(g>4.0f)g=4.0f;   /* 系数信赖域(重设计文档实测 g 深层≤4.5) */
            ce->gcal[t]=g; ce->gden[t]=(float)den;
        }
    }
}
static void *coadapt_worker(void*a){
    cowork_t *w=a;
    /* ★累加前层★: 校准行取自当前累积激活 + 实际路由(部署真值), 非 FP 锚 */
    const float *afin=w->Fin;
    const int   *aidx=w->idx;
    const float *arw=w->rw;
    for(;;){ int e=__sync_fetch_and_add(w->e_next,1); if(e>=NEXP)break;
        if((e&31)==0){ char pg[24]; snprintf(pg,sizeof(pg),"%d/%d",e,NEXP);
            mlog(w->L,w->iter?"向后":"量化",w->iter?"w2重解(系数校正目标)":"signref",pg,
                 (uint64_t)NEXP*(2*(uint64_t)MOEI*go1b_blk_row_bytes(DIM)+(uint64_t)DIM*go1b_blk_row_bytes(MOEI)),
                 "待测","进行中"); }
        coexp_t *ce=&w->ex[e];
        char n2[160]; long rr,cc;
        snprintf(n2,sizeof(n2),"layers.%d.ffn.experts.%d.w2.weight",w->L,e);
        if(w->iter==0){
            char n1[160],n3[160];
            snprintf(n1,sizeof(n1),"layers.%d.ffn.experts.%d.w1.weight",w->L,e);
            snprintf(n3,sizeof(n3),"layers.%d.ffn.experts.%d.w3.weight",w->L,e);
            const uint64_t *bfvt0=(bf_from_bytes()&&BFB_VQMAP)?(const uint64_t*)(BFB_VQMAP+16):NULL;
            int bfsk13=(bfvt0&&g2_hot_slot(w->L,e)<0
                        &&bfvt0[(size_t)e*3]&&bfvt0[(size_t)e*3+1]);   /* ★字节起步冷专家: e1/e3 免读免转(~16GB/层);
                        * 盘上字节=量化段导出解, 已含同α GPTAQ 目标(export 侧 yadjE), 本处 yadj 块一并跳过 */
            float *e1=bfsk13?NULL:st_read_weight(&C,n1,&rr,&cc);
            float *e3=bfsk13?NULL:st_read_weight(&C,n3,&rr,&cc);
            float *e2=st_read_weight(&C,n2,&rr,&cc);
            if((!bfsk13&&(!e1||!e3))||!e2){ if(e1)free(e1);if(e3)free(e3);if(e2)free(e2);
                fprintf(stderr,"\n[!] L%d e%d 读失败\n",w->L,e); continue; }
            int ncal=0;
            for(int s2=0;s2<w->n_fit;s2++)for(int a2=0;a2<NACT;a2++)
                if(aidx[(size_t)s2*NACT+a2]==e){ncal++;break;}
            float *Xc=NULL;
            if(ncal){ Xc=malloc((size_t)ncal*DIM*4);
                ce->calS=malloc((size_t)ncal*sizeof(int)); ce->calW=malloc((size_t)ncal*4);
                int t=0;
                for(int s2=0;s2<w->n_fit;s2++)for(int a2=0;a2<NACT;a2++)
                    if(aidx[(size_t)s2*NACT+a2]==e){
                        memcpy(Xc+(size_t)t*DIM,afin+(size_t)s2*DIM,(size_t)DIM*4);
                        ce->calS[t]=s2; ce->calW[t]=arw[(size_t)s2*NACT+a2]; t++; break; }
            } else w->calib_empty++;
            ce->ncal=ncal; w->calib_rows+=ncal; w->nhit_experts++;
            int nhit=0;
            for(int s2=0;s2<w->S;s2++)for(int a2=0;a2<NACT_RT;a2++)
                if(w->idx[(size_t)s2*NACT_RT+a2]==e){nhit++;break;}
            ce->nhit=nhit;
            int g2hot=g2_hot_slot(w->L,e)>=0;   /* 热专家: go2b 合并态量化(残差+量化一体) */
            /* GPTAQ 非对称目标(2026-07-25 G1b): y_ref=W·x̂+α·W·(x̃−x̂); x̃=锚 FP 流(量化全程不被覆盖)。
             * 回归仍在漂移输入 x̂ 上(合并态), 只有目标含 FP 流 ⇒ 纠正累积上游漂移而非仅条件于它。 */
            float *yadj1=NULL,*yadj3=NULL; float tga=dq_tgt_alpha();
            if(!bfsk13&&tga>0.0f&&ncal&&ANC_OK){
                float *dx=malloc((size_t)ncal*DIM*4);
                const float *afp=ANC.fin+(size_t)w->L*w->S*DIM;
                for(int t=0;t<ncal;t++){ int s2=ce->calS[t];
                    const float *xf=afp+(size_t)s2*DIM,*xh=afin+(size_t)s2*DIM;
                    for(int d=0;d<DIM;d++) dx[(size_t)t*DIM+d]=xf[d]-xh[d]; }
                yadj1=malloc((size_t)ncal*MOEI*4); yadj3=malloc((size_t)ncal*MOEI*4);
                dq_matmul(dx,e1,yadj1,ncal,DIM,MOEI); dq_matmul(dx,e3,yadj3,ncal,DIM,MOEI);
                for(size_t i2=0;i2<(size_t)ncal*MOEI;i2++){ yadj1[i2]*=tga; yadj3[i2]*=tga; }
                free(dx);
            }
            float *q1,*q3;
            if(bfsk13){   /* ★字节起步: 冷 w1/w3=盘上 VQ 载荷 dequant */
                q1=malloc((size_t)MOEI*DIM*4); q3=malloc((size_t)MOEI*DIM*4);
                vq_unpack_dequant(BFB_VQMAP+bfvt0[(size_t)e*3],BFB_VQMSZ-bfvt0[(size_t)e*3],q1,NULL,NULL);
                vq_unpack_dequant(BFB_VQMAP+bfvt0[(size_t)e*3+1],BFB_VQMSZ-bfvt0[(size_t)e*3+1],q3,NULL,NULL);
            } else if(dq_vq_on()){   /* v2.2: 码本量化(α 目标暂不进 VQ 内环, 行乘子已含激活拟合) */
                q1=g2hot?dq_quant_expert_vq(e1,MOEI,DIM,Xc,ncal,vq_hot_dim(),vq_hot_nc()):dq_quant_expert_vq(e1,MOEI,DIM,Xc,ncal,vq_cold_dim(),vq_cold_nc());
                q3=g2hot?dq_quant_expert_vq(e3,MOEI,DIM,Xc,ncal,vq_hot_dim(),vq_hot_nc()):dq_quant_expert_vq(e3,MOEI,DIM,Xc,ncal,vq_cold_dim(),vq_cold_nc());
            } else {
                q1=g2hot?dq_quant_expert_go2b_adj(e1,MOEI,DIM,Xc,ncal,yadj1):dq_quant_expert_signref_adj(e1,MOEI,DIM,Xc,ncal,yadj1,NULL);
                q3=g2hot?dq_quant_expert_go2b_adj(e3,MOEI,DIM,Xc,ncal,yadj3):dq_quant_expert_signref_adj(e3,MOEI,DIM,Xc,ncal,yadj3,NULL);
            }
            if(yadj1){free(yadj1);yadj1=NULL;} if(yadj3){free(yadj3);yadj3=NULL;}
            if(ncal){ ce->hc_cal=malloc((size_t)ncal*MOEI*4); co_hc(Xc,ncal,q1,q3,ce->hc_cal); }
            if(nhit){
                ce->hitS=malloc((size_t)nhit*sizeof(int)); ce->hitW=malloc((size_t)nhit*4);
                float *Xa=malloc((size_t)nhit*DIM*4); int t=0;
                for(int s2=0;s2<w->S;s2++)for(int a2=0;a2<NACT_RT;a2++)
                    if(w->idx[(size_t)s2*NACT_RT+a2]==e){
                        memcpy(Xa+(size_t)t*DIM,w->Fin+(size_t)s2*DIM,(size_t)DIM*4);
                        ce->hitS[t]=s2; ce->hitW[t]=w->rw[(size_t)s2*NACT_RT+a2]; t++; break; }
                ce->hc_all=malloc((size_t)nhit*MOEI*4); co_hc(Xa,nhit,q1,q3,ce->hc_all);
                free(Xa);
            }
            free(q1);free(q3);free(e1);free(e3);
            if(ce->ncal){   /* w2 顺序补偿 FP 目标(乘法系数的参照, 只算一次) */
                ce->y2ref=malloc((size_t)ce->ncal*DIM*4);
                dq_matmul(ce->hc_cal,e2,ce->y2ref,ce->ncal,MOEI,DIM);
            }
            float *w2q;
            if(bf_from_bytes()&&BFB_W2&&!g2hot){   /* ★字节起步: 冷 w2 基座=盘上 D 段 go1b(同 μ10 FULLSET 解); D3 后续照常重解 */
                w2q=malloc((size_t)DIM*MOEI*4);
                dq_go1b_bytes_dequant(BFB_W2+(size_t)e*BFB_SZD,DIM,MOEI,w2q);
            }
            else if(dq_vq_on()&&g2hot) w2q=dq_quant_expert_vq(e2,DIM,MOEI,ce->hc_cal,ce->ncal,vq_hot_dim(),vq_hot_nc());
            /* ★冷 w2 评估路径带顺序补偿(2026-08-03): 与导出路径(vq_export_matrix_seq)同解,
             * 搜索/调优/z 全链看到的就是部署态 — 失配事故修 */
            else if(dq_vq_on()&&vq_w2_dim()>0)   /* R28: 冷 w2 码本(省 6.2G 给 w13; 冠军此处是 signref) */
                w2q=dq_quant_expert_vq_seq(e2,DIM,MOEI,ce->hc_cal,ce->ncal,vq_w2_dim(),vq_w2_nc(),1);
            else w2q=g2hot?dq_quant_expert_go2b(e2,DIM,MOEI,ce->hc_cal,ce->ncal)
                            :dq_quant_expert_signref_adj(e2,DIM,MOEI,ce->hc_cal,ce->ncal,NULL,&w->nflip);
            co_w2_finish(ce,w2q,w->fout);
            free(w2q);free(e2);free(Xc);
        } else {
            if(ce->nhit<1&&ce->ncal<1) continue;
            if(g2_hot_slot(w->L,e)>=0){
                /* 热专家 base=go2b 冻结: D3 不重解 w2(基座不随系数目标漂), 输出直接用缓存
                 * yq_all scatter(与重算逐字节同; gcal/yq_cal 也不变)。动态系数照常在外层拟合。 */
                if(ce->yq_all) for(int i=0;i<ce->nhit;i++){ float wgt=ce->hitW[i];
                    float*dst=w->fout+(size_t)ce->hitS[i]*DIM; const float*yi=ce->yq_all+(size_t)i*DIM;
                    for(int d2=0;d2<DIM;d2++) dst[d2]+=wgt*yi[d2]; }
                continue;
            }
            float *e2=st_read_weight(&C,n2,&rr,&cc); if(!e2)continue;
            /* D3 交替: sign/scale 重选目标 = 动态系数校正后残差。
             * 乘法系数 c_t 进模型侧: 缩放校准行 x'_t=c_t·hc_t (→ 模型项 c_t·s·P_t),
             * Yadj=(1−c_t)·y2ref 把内部目标 w·x'=c_t·y2ref 平移回 y2ref → 精确闭式。 */
            float *Xs=NULL,*Yadj=NULL;
            if(ce->ncal&&ce->ccal){
                Xs=malloc((size_t)ce->ncal*MOEI*4);
                Yadj=malloc((size_t)ce->ncal*DIM*4);
                for(int t=0;t<ce->ncal;t++){ float c=ce->ccal[t];
                    const float*hs=ce->hc_cal+(size_t)t*MOEI; float*xd=Xs+(size_t)t*MOEI;
                    for(int j=0;j<MOEI;j++) xd[j]=c*hs[j];
                    const float*yr=ce->y2ref+(size_t)t*DIM; float*ya=Yadj+(size_t)t*DIM;
                    float oc=1.0f-c;
                    for(int d2=0;d2<DIM;d2++) ya[d2]=oc*yr[d2];
                }
            }
            float *w2q;
            w2q=dq_quant_expert_signref_adj(e2,DIM,MOEI,Xs?Xs:ce->hc_cal,ce->ncal,Yadj,&w->nflip);
            co_w2_finish(ce,w2q,w->fout);
            free(w2q);free(e2);if(Xs)free(Xs);if(Yadj)free(Yadj);
        }
    }
    return NULL;
}
/* z_e 逐专家闭式条件系数(D2, 设计本义"全层动态系数"): 校准行标量目标 gt[t] → dual ridge
 * kernel 解 α=(K+λI)⁻¹(gt−1), K_ij=⟨x_i,x_j⟩(x=anchor Fin 行)。c(x)=1+Σα_t⟨x_t,x⟩,
 * 信赖域 clamp [0.25,4]。填 alpha + 校准行系数 ccal(向后重解目标用)。 */
static void co_fit_ze(coexp_t*ce,const float*afin,const float*gt,double lam,int vs){
    int n=ce->ncal; if(n<1||!gt) return;
    if(!ce->alpha) ce->alpha=malloc((size_t)n*4);
    if(!ce->ccal)  ce->ccal =malloc((size_t)n*4);
    for(int i=0;i<n;i++) ce->alpha[i]=0.0f;
    /* 防泄漏: 只用 calS<vs 的行拟合(val 行只当裁判) */
    int *sub=malloc((size_t)n*sizeof(int)),m=0;
    for(int i=0;i<n;i++) if(ce->calS[i]<vs) sub[m++]=i;
    if(m<1){ free(sub); if(ce->ccal) for(int i=0;i<n;i++) ce->ccal[i]=1.0f; return; }
    double *K=malloc((size_t)m*m*sizeof(double)),*b=malloc((size_t)m*sizeof(double));
    double tr=0;
    for(int i=0;i<m;i++){ const float*xi=afin+(size_t)ce->calS[sub[i]]*DIM;
        for(int j=i;j<m;j++){ const float*xj=afin+(size_t)ce->calS[sub[j]]*DIM;
            double s=0; for(int d2=0;d2<DIM;d2++) s+=(double)xi[d2]*xj[d2];
            K[(size_t)i*m+j]=s; K[(size_t)j*m+i]=s; }
        tr+=K[(size_t)i*m+i]; }
    double ridge=lam*(tr/(double)m)+1e-8;
    for(int i=0;i<m;i++) K[(size_t)i*m+i]+=ridge;
    for(int i=0;i<m;i++) b[i]=(double)gt[sub[i]]-1.0;
    if(cholesky(K,(uint32_t)m)==0){
        cholesky_solve(K,(uint32_t)m,b);
        for(int i=0;i<m;i++) ce->alpha[sub[i]]=(float)b[i];
    }
    /* 全部校准行的系数(向后重解目标): c=1+Σ_j α_j⟨x_j,x_i⟩ */
    for(int i=0;i<n;i++){ const float*xi=afin+(size_t)ce->calS[i]*DIM; double s=1.0;
        for(int j=0;j<m;j++){ const float*xj=afin+(size_t)ce->calS[sub[j]]*DIM;
            double d=0; for(int d2=0;d2<DIM;d2++) d+=(double)xi[d2]*xj[d2];
            s+=ce->alpha[sub[j]]*d; }
        if(s<0.25)s=0.25; if(s>4.0)s=4.0; ce->ccal[i]=(float)s; }
    free(K);free(b);free(sub);
}
/* 乘法应用: 各 ce 的 alpha(动态) 或 gstat(静态) → Ftest = shared + Σ w·c·ŷ */
