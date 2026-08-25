static void export_gguf(int S,int n_fit){
    const char*gp=getenv("DS4_EXPORT_GGUF"), *op=getenv("DS4_EXPORT_OFF");
    if(!op){ fprintf(stderr,"需 DS4_EXPORT_OFF=偏移表(gguf_offsets.py 输出)\n"); exit(1); }
    GOFF=calloc((size_t)NL*3,sizeof(goff_t)); NGOFF=0;
    FILE*f=fopen(op,"r"); if(!f){perror("off");exit(1);}
    { char nm[256]; int ty; long off,nel;
      while(fscanf(f,"%255s %d %ld %ld",nm,&ty,&off,&nel)==4){
        int L; char kind[16]; int rest=0;
        if(sscanf(nm,"blk.%d.ffn_%15[^_]_exps.weight%n",&L,kind,&rest)==2 && nm[rest]==0 && L<NL && NGOFF<NL*3){
            GOFF[NGOFF].L=L; snprintf(GOFF[NGOFF].kind,8,"%s",kind);
            GOFF[NGOFF].off=off; GOFF[NGOFF].nel=nel; GOFF[NGOFF].type=ty; NGOFF++; } } }
    fclose(f);
    fprintf(stderr,"[export] 偏移表专家 tensor %d 个 (期望 %d)\n",NGOFF,NLAYERS*3);
    FILE*gf=fopen(gp,"r+b"); if(!gf){perror("gguf");exit(1);}
    size_t rbG=go1b_blk_row_bytes(DIM), szG=(size_t)MOEI*rbG;   /* w1/w3 每专家 bytes */
    size_t rbD=go1b_blk_row_bytes(MOEI), szD=(size_t)DIM*rbD;   /* w2 */
    uint8_t *bg=malloc((size_t)NEXP*szG),*bu=malloc((size_t)NEXP*szG),*bd=malloc((size_t)NEXP*szD);
    for(int L=0;L<NLAYERS;L++){
        goff_t *tg=goff_find(L,"gate"),*tu=goff_find(L,"up"),*td=goff_find(L,"down");
        if(!tg||!tu||!td){ fprintf(stderr,"L%02d tensor 缺失, 跳过\n",L); continue; }
        if(tg->type!=40||tu->type!=40||td->type!=40){ fprintf(stderr,"L%02d 类型非 go1b(40): %d/%d/%d — 拒写\n",L,tg->type,tu->type,td->type); exit(1); }
        long need=(long)NEXP*MOEI*DIM;
        if(tg->nel!=need||tu->nel!=need||td->nel!=need){ fprintf(stderr,"L%02d 元素数不符 — 拒写\n",L); exit(1); }
        fprintf(stderr,"L%02d 导出 ",L);
        int e_next=0; int nth=NTHREADS>NEXP?NEXP:NTHREADS;
        expw_t *ws=calloc((size_t)nth,sizeof(expw_t)); pthread_t *th=malloc((size_t)nth*sizeof(pthread_t));
        for(int t=0;t<nth;t++){ ws[t]=(expw_t){L,S,n_fit,&e_next,bg,bu,bd,szG,szD,0,0,-1,0,0,{0},-1,{0}};
            pthread_create(&th[t],NULL,export_worker,&ws[t]); }
        for(int t=0;t<nth;t++) pthread_join(th[t],NULL);
        free(ws);free(th);
        int ok=1;
        ok &= fseeko(gf,(off_t)tg->off,SEEK_SET)==0 && fwrite(bg,1,(size_t)NEXP*szG,gf)==(size_t)NEXP*szG;
        ok &= fseeko(gf,(off_t)tu->off,SEEK_SET)==0 && fwrite(bu,1,(size_t)NEXP*szG,gf)==(size_t)NEXP*szG;
        ok &= fseeko(gf,(off_t)td->off,SEEK_SET)==0 && fwrite(bd,1,(size_t)NEXP*szD,gf)==(size_t)NEXP*szD;
        if(!ok){ fprintf(stderr,"L%02d 写失败!\n",L); exit(1); }
        fprintf(stderr," 已写 (%.0f MB)\n",(2.0*NEXP*szG+NEXP*szD)/1048576.0);
    }
    fflush(gf); fclose(gf);
    free(bg);free(bu);free(bd);
    fprintf(stderr,"[export] 完成: %s 全部专家已替换为 signref(μ=%.0f)+anchor校准 配方\n",gp,dq_signref_mu);
}

/* ===================== 逐层渐进调优 (DS4_TUNE=1) ===================== *
 * 用户架构(2026-07-10): 每层动态、渐进、成果永久复用 — 不做"一把跑43层最后作废"的长任务。
 *   每层: 候选集现场对比(层内优化) → held 泛化+感知加权评分(向 FP 锚偏离=向后代理) →
 *   选定即固化(plan 文件 + 激活 checkpoint 落盘) → 下一层在已固化成果上继续。
 *   重启时已调层秒级跳过(复用); 加新候选只重跑该层及以后(删对应 plan 行/ckpt 即可)。 */
typedef struct { const char*name; char cfg; double mu; int rounds; int lz; float lztr,lzlam; int hotk; } cand_t;
/* 候选集 v2(目标导向, 2026-07-10 晚): 同族微调(g3/g30/rz)每层只挽回 0.2-7% = 瞎跑;
 * 换入真杠杆 q2(+0.84GiB/层), 预算规则: 免费候选先选最优, q2 挽回≥DS4_TUNE_GAIN(默认15%)才升位。 */
static const cand_t CANDS[]={
    {"r",   'r',  0.0,0, 0,0,0, 0},       /* 行scale 保守锚(零体积差) */
    {"g10", 'g', 10.0,3, 0,0,0, 0},       /* 符号精修最优代表(零体积差) */
    {"q2",  '2',  0.0,0, 0,0,0, 0},       /* Q1+Q2 残差 2bit: 真杠杆, +0.84GiB/层, 预算门控 */
};
#define NCAND (int)(sizeof(CANDS)/sizeof(CANDS[0]))
/* ★最小体积模式(DS4_MINVOL=1)★ 计划表驱动逐层档位(DS4_MV_BASELINE+DS4_VQ_RPLAN),
 * 产物两文件: dql(1bit 基座+记录+z [+内嵌 g2hot 热载荷]) + opt。 */
static const cand_t MV_G10={"g10",'g',10.0,3,0,0,0,0};
/* 热集=FP 锚路由权重和 top-K(过程内口径, 每层现场算, 零静态表; X9 实测 K=16 覆盖~95%;
 * freq_to_norms 教训: 按路由权重和排序, 不按命中 count)。填 G2_* 全局 → quant_apply/
 * export_worker 既有 g2_hot_slot 门直接生效。 */
static int hot_from_anchor(int L,int S,int K){
    double wsum[256]={0};
    const int32_t*aidx=ANC.ridx+(size_t)L*S*NACT; const float*arw=ANC.rw+(size_t)L*S*NACT;
    for(int s=0;s<S;s++)for(int a=0;a<NACT;a++){ int e=aidx[(size_t)s*NACT+a];
        if(e>=0&&e<256) wsum[e]+=arw[(size_t)s*NACT+a]; }
    for(int e=0;e<256;e++) G2_SLOT[L][e]=-1;
    int k=0;
    for(;k<K&&k<256;k++){ int be=-1; double bw=0;
        for(int e=0;e<256;e++) if(G2_SLOT[L][e]<0&&wsum[e]>bw){bw=wsum[e];be=e;}
        if(be<0)break;
        G2_SLOT[L][be]=(int16_t)k; G2_IDS[L][k]=(uint16_t)be; }
    G2_K[L]=k; GO2B_HOT=(k>0);
    return k;
}
static const cand_t *TAB=CANDS; static int NTAB=NCAND;   /* plan_lookup 用活动表 */

/* held 泛化评分(越小越好): held 行 hc 状态 vs FP 锚, align + 0.5·classify(维度方差=感知加权) */
static double held_score(const float*H,int S,int n_fit,int L,double*relh){
    size_t row=(size_t)HCM*DIM; int nh=S-n_fit;
    const float*A=H+(size_t)n_fit*row;
    const float*B=ANC.H+(size_t)L*S*row+(size_t)n_fit*row;
    double e2=0,a2=0;
    for(size_t i=0;i<(size_t)nh*row;i++){ double d=(double)A[i]-B[i]; e2+=d*d; a2+=(double)B[i]*B[i]; }
    if(relh)*relh=sqrt(e2/(a2+1e-30));
    float *wv=malloc(row*4); ds4_loss_dim_variance(B,(uint32_t)nh,(uint32_t)row,wv);
    double la=ds4_loss_align(A,B,(uint32_t)nh,(uint32_t)row);
    double lc=ds4_loss_classify(A,B,wv,(uint32_t)nh,(uint32_t)row);
    free(wv);
    return la+0.5*lc;
}
static const char*ckpt_dir(void){ const char*p=getenv("DS4_CKPT_DIR"); return p?p:"/tmp/ds4q_ckpt"; }
static const char*plan_path(void){ const char*p=getenv("DS4_PLAN"); return p?p:"/tmp/ds4quant_plan.txt"; }
/* 读 plan 中第 L 层已锁链式 relh(增长率门用; 无记录=-1) */
static double plan_relh_of(int L){
    FILE*f=fopen(plan_path(),"r"); if(!f) return -1;
    char ln[256]; double out=-1;
    while(fgets(ln,sizeof(ln),f)){ int li; double rv;
        if(sscanf(ln,"L=%d cand=%*s relh=%lf",&li,&rv)==2 && li==L) out=rv; }
    fclose(f); return out;
}
static int ckpt_load(int L,float*H,size_t n,uint64_t idh){
    char p[512]; snprintf(p,sizeof(p),"%s/L%02d.bin",ckpt_dir(),L);
    FILE*f=fopen(p,"rb"); if(!f)return 0;
    uint64_t hd[2]; int ok = fread(hd,8,2,f)==2 && hd[0]==idh && hd[1]==(uint64_t)n
                          && fread(H,4,n,f)==n;
    fclose(f); return ok;
}
static void ckpt_save(int L,const float*H,size_t n,uint64_t idh){
    char p[512]; snprintf(p,sizeof(p),"%s/L%02d.bin",ckpt_dir(),L);
    FILE*f=fopen(p,"wb"); if(!f)return;
    uint64_t hd[2]={idh,(uint64_t)n};
    fwrite(hd,8,2,f); fwrite(H,4,n,f); fclose(f);
}
/* 影子冠军链 ckpt(断点续跑不重算影子) */
static int ckpt_load_s(int L,float*H,size_t n,uint64_t idh){
    char p[512]; snprintf(p,sizeof(p),"%s/L%02d.sh.bin",ckpt_dir(),L);
    FILE*f=fopen(p,"rb"); if(!f)return 0;
    uint64_t hd[2]; int ok=fread(hd,8,2,f)==2&&hd[0]==idh&&hd[1]==(uint64_t)n&&fread(H,4,n,f)==n;
    fclose(f); return ok;
}
static void ckpt_save_s(int L,const float*H,size_t n,uint64_t idh){
    char p[512]; snprintf(p,sizeof(p),"%s/L%02d.sh.bin",ckpt_dir(),L);
    FILE*f=fopen(p,"wb"); if(!f)return;
    uint64_t hd[2]={idh,(uint64_t)n};
    fwrite(hd,8,2,f); fwrite(H,4,n,f); fclose(f);
}
/* rr 判决链 H2 的 ckpt(断点续跑与主链同粒度; 头校验=rr 锚 ids 哈希) */
static int ckpt2_load(int L,float*H,size_t n){
    char p[512]; snprintf(p,sizeof(p),"%s/L%02d.rr.bin",ckpt_dir(),L);
    FILE*f=fopen(p,"rb"); if(!f)return 0;
    uint64_t hd[2]; int ok=fread(hd,8,2,f)==2&&hd[0]==ANC2.idh&&hd[1]==(uint64_t)n&&fread(H,4,n,f)==n;
    fclose(f); return ok;
}
static void ckpt2_save(int L,const float*H,size_t n){
    char p[512]; snprintf(p,sizeof(p),"%s/L%02d.rr.bin",ckpt_dir(),L);
    FILE*f=fopen(p,"wb"); if(!f)return;
    uint64_t hd[2]={ANC2.idh,(uint64_t)n};
    fwrite(hd,8,2,f); fwrite(H,4,n,f); fclose(f);
}
/* plan 文件: 每行 "L=%d cand=%s relh=..."; 同层多行取最后(允许追加覆盖) */
static int plan_lookup(int L){
    FILE*f=fopen(plan_path(),"r"); if(!f)return -1;
    char ln[256]; int found=-1;
    while(fgets(ln,sizeof(ln),f)){ int l; char nm[32];
        if(sscanf(ln,"L=%d cand=%31s",&l,nm)==2 && l==L)
            for(int c=0;c<NTAB;c++) if(!strcmp(nm,TAB[c].name)) found=c;
    }
    fclose(f); return found;
}
static void plan_append(int L,const char*nm,double relh){
    FILE*f=fopen(plan_path(),"a"); if(!f)return;
    fprintf(f,"L=%d cand=%s relh=%.4f\n",L,nm,relh); fclose(f);
}
/* minvol plan 行(rr 预算门版): 附 rr 里程碑值(审计+RESUME 参考; plan_lookup 前两字段不受影响) */
static void plan_append_rr(int L,const char*nm,double relh,double rrm){
    FILE*f=fopen(plan_path(),"a"); if(!f)return;
    fprintf(f,"L=%d cand=%s relh=%.4f gate=%.4f\n",L,nm,relh,rrm); fclose(f);
}
/* minvol plan 复用: 动态名("m448"/"m448h"/"g10"/"g10h")解析, 返回 cfg('m'/'g', 0=无记录) */
static char plan_lookup_mv(int L,char*nm_out){
    FILE*f=fopen(plan_path(),"r"); if(!f)return 0;
    char ln[256],got[32]={0};
    while(fgets(ln,sizeof(ln),f)){ int l; char nm[32];
        if(sscanf(ln,"L=%d cand=%31s",&l,nm)==2&&l==L) snprintf(got,sizeof(got),"%s",nm); }
    fclose(f);
    if(!got[0]) return 0;
    if(nm_out) snprintf(nm_out,16,"%s",got);
    return got[0]=='m'?'m':'g';
}
static void set_cand(const cand_t*c){
    dq_signref_mu=c->mu; dq_signref_rounds=c->rounds;
    /* ★候选未带 lz(=0)时回落 env(2026-08-23): pure 模式 set_cand(&MV_G10) 曾把
     * DS4_LZ=8 覆写清零 —— ZDIAG LZRANK=0 实锤。档位自带 lz 仍优先。 */
    LZRANK = c->lz ? c->lz : (getenv("DS4_LZ") ? atoi(getenv("DS4_LZ")) : 0);
    if(c->lztr) LZTR=c->lztr; if(c->lz)LZLAMBDA=c->lzlam;
}
/* verdict-lite: 只算 held Σmin/KL(里程碑探针用) */
static void verdict_lite(const float*lf,const float*lq,const long*ids,int S,int n_fit,double*smin,double*kl){
    int s0=n_fit,s1=S-1; if(s1-s0<1)s0=0;
    double sm=0,k=0; int n=0;
    for(int s=s0;s<s1;s++){
        const float*a=lf+(size_t)s*VOCAB,*b=lq+(size_t)s*VOCAB;
        float ma=a[0],mb=b[0];
        for(int v=1;v<VOCAB;v++){ if(a[v]>ma)ma=a[v]; if(b[v]>mb)mb=b[v]; }
        double sa=0,sb=0; for(int v=0;v<VOCAB;v++){ sa+=exp((double)a[v]-ma); sb+=exp((double)b[v]-mb); }
        double lsa=log(sa),lsb=log(sb);
        for(int v=0;v<VOCAB;v++){ double lpf=(double)a[v]-ma-lsa,lpq=(double)b[v]-mb-lsb;
            double pf=exp(lpf),pq=exp(lpq); sm+=pf<pq?pf:pq; if(pf>0)k+=pf*(lpf-lpq); }
        n++;
    }
    *smin=sm/n; *kl=k/n;
}
/* 还原率里程碑探针: 前 L 层用已调优结果 + 后缀纯 FP → 真实 Σmin(=若后面全无损的当前还原率上界)。
 * 对照目标线, 不达标趋势早停, 不瞎跑。 */
static double restore_probe(const float*H,int L,const long*ids,int S,int n_fit){
    size_t lstride=(size_t)S*HCM*DIM;
    float *Hp=malloc(lstride*4); memcpy(Hp,H,lstride*4);
    fprintf(stderr,"[里程碑探针 L%02d] 后缀 FP 前向...\n",L);
    for(int l=L+1;l<NLAYERS;l++){ LW W2=load_layer(l);
        fprintf(stderr,"  probe L%02d F\n",l);
        layer_fwd(l,&W2,Hp,ids,S,n_fit,0,'F',NULL); free_layer(&W2); }
    float *hcfn=st_read_weight(&C,"hc_head_fn",NULL,NULL),*hcb=st_read_weight(&C,"hc_head_base",NULL,NULL),*hcs=st_read_weight(&C,"hc_head_scale",NULL,NULL);
    float *norm=st_read_weight(&C,"norm.weight",NULL,NULL);
    float *lp=malloc((size_t)S*VOCAB*4); head_fwd_stream(Hp,S,hcfn,hcb,hcs,norm,lp);
    double sm,kl; verdict_lite(ANC.logits,lp,ids,S,n_fit,&sm,&kl);
    fprintf(stderr,"★[还原率里程碑] 前%d层已调优+后缀无损: Σmin=%.4f KL=%.3f  (目标线0.52/0.60; FP=1.0)\n",L+1,sm,kl);
    printf("MILESTONE L=%d smin=%.4f kl=%.4f\n",L,sm,kl); fflush(stdout);
    free(Hp);free(lp);free(hcfn);free(hcb);free(hcs);free(norm);
    return sm;
}
/* ---- rr_hard 判决锚初始化(统一标准): 载/建 ANC2, 全程一次 ---- */
static int rr_ids_load(const char*p){
    FILE*f=fopen(p,"r"); if(!f) return 0;
    long v; int n=0,cap=4096; g_ids2=malloc((size_t)cap*sizeof(long));
    while(fscanf(f,"%ld",&v)==1){ if(n>=cap){cap*=2;g_ids2=realloc(g_ids2,(size_t)cap*sizeof(long));} g_ids2[n++]=v; }
    fclose(f); g_S2=n; return n>0;
}
static void rr_anchor_init(void){
    const char*ip=getenv("DS4_RR_IDS"); if(!ip) return;
    const char*ap=getenv("DS4_ANCHOR2");
    if(!ap){ fprintf(stderr,"[rr锚] 需 DS4_ANCHOR2 路径 — 硬拒\n"); exit(2); }
    if(!rr_ids_load(ip)){ fprintf(stderr,"[rr锚] ids 读取失败 %s — 硬拒\n",ip); exit(2); }
    fprintf(stderr,"[rr锚] 判决语料 %s S=%d (统一标准: 判决口径=rr_hard, 冠军尺 0.7680/0.3602)\n",ip,g_S2);
    anchor_t sv=ANC; int svok=ANC_OK;
    char svpath[512]; snprintf(svpath,sizeof(svpath),"%s",anchor_path());
    setenv("DS4_ANCHOR",ap,1);
    memset(&ANC,0,sizeof(ANC)); ANC_OK=0;
    uint64_t idh2=dq_ids_hash(g_ids2,g_S2);
    if(anchor_load(g_S2,idh2)) fprintf(stderr,"[rr锚] 命中缓存 %s (FP 遍跳过)\n",ap);
    else {
        fprintf(stderr,"[rr锚] 建锚2: FP 遍 %d tok(一次性)...\n",g_S2);
        anchor_alloc(g_S2); ANC.idh=idh2; ANC_BUILD=1;
        ANC.logits=fwd_all(g_ids2,g_S2,g_S2,0,NULL);
        ANC_BUILD=0;
        if(anchor_save()) fprintf(stderr,"[rr锚] 已写 %s\n",ap);
    }
    ANC2=ANC; ANC2_OK=1;
    ANC=sv; ANC_OK=svok;
    setenv("DS4_ANCHOR",svpath,1);
}
/* rr 里程碑(判决口径): H2 前缀量化链 + 后缀 FP → 全段 Σmin/KL vs rr FP logits。
 * H2=裸胜者档链(不含 coadapt z 微增益) ⇒ 保守下界; 终验用完整回放取真值。 */
static double rr_probe(const float*H2,int L){
    size_t l2=(size_t)g_S2*HCM*DIM;
    float *Hp=malloc(l2*4); memcpy(Hp,H2,l2*4);
    for(int l=L+1;l<NLAYERS;l++){ LW W2=load_layer(l);
        layer_fwd(l,&W2,Hp,g_ids2,g_S2,g_S2,0,'F',NULL); free_layer(&W2); }
    float *hcfn=st_read_weight(&C,"hc_head_fn",NULL,NULL),*hcb=st_read_weight(&C,"hc_head_base",NULL,NULL),*hcs=st_read_weight(&C,"hc_head_scale",NULL,NULL);
    float *norm=st_read_weight(&C,"norm.weight",NULL,NULL);
    float *lp=malloc((size_t)g_S2*VOCAB*4); head_fwd_stream(Hp,g_S2,hcfn,hcb,hcs,norm,lp);
    double sm,kl; verdict_lite(ANC2.logits,lp,g_ids2,g_S2,0,&sm,&kl);
    fprintf(stderr,"★★[rr里程碑|判决口径|冠军尺0.7680/0.3602] 前%d层量化+后缀无损: Σmin=%.4f KL=%.3f\n",L+1,sm,kl);
    printf("RRMILESTONE L=%d smin=%.4f kl=%.4f\n",L,sm,kl); fflush(stdout);
    free(Hp);free(lp);free(hcfn);free(hcb);free(hcs);free(norm);
    return sm;
}
/* rr 链单层前向(候选/胜者态, 预算门与正式推进共用): H2s→H2d(可同址)。
 * 量化字节与定稿态同(确定性+EXP_ 覆盖=校准仍 v5mini fit 行, 判决语料零污染); COADAPT 关。 */
static void rr_step(int L,LW*W,const float*H2s,float*H2d,int S,int n_fit,char cfg){
    size_t l2=(size_t)g_S2*HCM*DIM;
    if(H2d!=H2s) memcpy(H2d,H2s,l2*4);
    anchor_t sv=ANC; int svok=ANC_OK;
    if(!EXP_FIN)EXP_FIN=malloc((size_t)S*DIM*4);
    if(!EXP_IDX)EXP_IDX=malloc((size_t)S*NACT*sizeof(int));
    memcpy(EXP_FIN,sv.fin+(size_t)L*S*DIM,(size_t)S*DIM*4);
    { const int32_t*ai=sv.ridx+(size_t)L*S*NACT;
      for(size_t i2=0;i2<(size_t)S*NACT;i2++)EXP_IDX[i2]=(int)ai[i2]; }
    EXP_L=L;
    ANC=ANC2; ANC_OK=1;
    int co_sv=COADAPT; COADAPT=0;
    lstat_t st2; memset(&st2,0,sizeof(st2));
    layer_fwd(L,W,H2d,g_ids2,g_S2,n_fit,1,cfg,&st2);
    COADAPT=co_sv;
    ANC=sv; ANC_OK=svok; EXP_L=-1;
}
static float *fwd_all_tune(const long*ids,int S,int n_fit,char*plan_out){
    size_t lstride=(size_t)S*HCM*DIM;
    uint64_t idh=ANC.idh;
    mkdir(ckpt_dir(),0755);
    double gain_th=0.15; if(getenv("DS4_TUNE_GAIN")) gain_th=atof(getenv("DS4_TUNE_GAIN"));
    float *H=malloc(lstride*4),*Hw=malloc(lstride*4),*Hb=malloc(lstride*4),*Hf2=malloc(lstride*4);
    { float *emb=st_read_weight(&C,"embed.weight",NULL,NULL);
      for(int s=0;s<S;s++)for(int j=0;j<HCM;j++)memcpy(H+((size_t)s*HCM+j)*DIM,emb+(size_t)ids[s]*DIM,(size_t)DIM*4);
      free(emb); }
    int minvol=getenv("DS4_MINVOL")?1:0;
    /* ★隔离探针(R24 D1, 2026-07-30 用户"先跑两个代表层试一试"): 上游=FP 锚定直通,
     * 只评 DS4_MV_PROBE_L 层 — 干净的单层判决(无上游漂移), R24 vs 冠军档同足对比。 */
    int g_probe_l=getenv("DS4_MV_PROBE_L")?atoi(getenv("DS4_MV_PROBE_L")):-1;
    double mv_alpha=getenv("DS4_MINVOL_ALPHA")?atof(getenv("DS4_MINVOL_ALPHA")):1.03;
    /* 活 α(2026-07-28 用户: "为啥α不能是活的"): 里程碑斜率外推终点 Σmin_proj,
     * proj<T−0.03 → α 收 0.03; proj>T+0.03 → α 放 0.02(顶 1.15)。
     * ★统一标准后: 外推/目标线一律 rr 口径(冠军尺), 默认终线 0.77≈v4bf 0.7680。 */
    double mv_target=getenv("DS4_MINVOL_TARGET")?atof(getenv("DS4_MINVOL_TARGET")):0.768;
    double mprev=-1;   /* rr 里程碑滚动值(预算门状态; -1=未初始化, RESUME 时现算恢复) */
    /* rr 判决链 H2(统一标准): 与主链同粒度推进, embed 起点 */
    size_t lstride2=ANC2_OK?(size_t)g_S2*HCM*DIM:0; float *H2=NULL;
    if(minvol&&ANC2_OK){
        H2=malloc(lstride2*4);
        float *emb2=st_read_weight(&C,"embed.weight",NULL,NULL);
        for(int s=0;s<g_S2;s++)for(int j=0;j<HCM;j++)memcpy(H2+((size_t)s*HCM+j)*DIM,emb2+(size_t)g_ids2[s]*DIM,(size_t)DIM*4);
        free(emb2);
    }
    if(minvol){
        RB_SEQ=getenv("DS4_ROUTE_SEQ")?1:0;
        if(RB_SEQ) fprintf(stderr,"[序贯路由] 开: 每层 FIT Δb + α{1.0,2.5,4.0} 扫(负收益自动关), 定稿/推进带修正, 收官落盘\n");
        fprintf(stderr,"\n[最小体积·计划表驱动] 每层档位由 DS4_VQ_RPLAN 计划表定; "
                       "地板门 α=%.2f; v5mini 里程碑每5层=护栏\n",mv_alpha); }
    else fprintf(stderr,"\n[渐进调优v2·目标导向] 免费候选(r/g)选最优; q2 挽回≥%.0f%% 才升位(+0.84GiB/层); 里程碑 L9/20/31\n",gain_th*100);
