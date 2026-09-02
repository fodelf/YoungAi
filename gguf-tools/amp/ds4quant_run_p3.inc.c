static void co_apply_mult(coexp_t*ex,const float*afin,const float*Fin,const float*shared,
                          int S,float*Ftest,double*cmean,double*cmx){
    memcpy(Ftest,shared,(size_t)S*DIM*4);
    double csum=0,cm=0; long cc=0;
    for(int e=0;e<NEXP;e++){ coexp_t*ce=&ex[e];
        if(ce->nhit<1||!ce->yq_all) continue;
        for(int i=0;i<ce->nhit;i++){
            double s=1.0;
            if(ce->gstat!=0.0f) s=ce->gstat;
            else if(ce->alpha&&ce->ncal){
                const float*xh=Fin+(size_t)ce->hitS[i]*DIM;
                for(int t=0;t<ce->ncal;t++){ if(ce->alpha[t]==0.0f)continue;
                    const float*xc=afin+(size_t)ce->calS[t]*DIM;
                    double d=0; for(int d2=0;d2<DIM;d2++) d+=(double)xc[d2]*xh[d2];
                    s+=ce->alpha[t]*d; }
            }
            if(s<DS4_AMP_LAM_MIN)s=DS4_AMP_LAM_MIN; if(s>DS4_AMP_LAM_MAX)s=DS4_AMP_LAM_MAX;
            double ac=fabs(s-1.0); csum+=ac; if(ac>cm)cm=ac; cc++;
            float wc=ce->hitW[i]*(float)s; float*dst=Ftest+(size_t)ce->hitS[i]*DIM;
            const float*yi=ce->yq_all+(size_t)i*DIM;
            for(int d2=0;d2<DIM;d2++) dst[d2]+=wc*yi[d2];
        }
    }
    if(cmean)*cmean=cc?csum/(double)cc:0; if(cmx)*cmx=cm;
}
static void co_rel(const float*Hq,const float*Hf,int a,int b,size_t rowsz,double*rel);
static double co_score(const float*Hq,const float*Hf,int a,int b,size_t rowsz);
/* 出口评分: fit/val/held 三段 relL2 + val 四损失评分(align+0.5·classify=感知加权) */
static void co_eval(const float*Ftest,const float*H2,const float*post2,const float*comb2,
                    const float*Hf,float*Hq,int S,int vs,int n_fit,size_t rowsz,
                    double*fit,double*val,double*held,double*sc){
    dq_hc_post(Ftest,H2,post2,comb2,Hq,S,HCM,DIM);
    co_rel(Hq,Hf,0,vs,rowsz,fit);
    co_rel(Hq,Hf,vs,n_fit,rowsz,val);
    co_rel(Hq,Hf,n_fit,S,rowsz,held);
    *sc = (n_fit-vs)>0 ? co_score(Hq,Hf,vs,n_fit,rowsz) : co_score(Hq,Hf,0,vs,rowsz);
}
/* 累积回拉目标 DF: hc_post 对 Fout 线性 → Fout 空间闭式反解到本层为止的漂移 */
static void co_df(const float*Fcur,const float*H2,const float*post2,const float*comb2,
                  const float*Hf,float*Hq,float*DF,int S){
    dq_hc_post(Fcur,H2,post2,comb2,Hq,S,HCM,DIM);
    for(int s=0;s<S;s++){
        const float *ps=post2+(size_t)s*HCM; double pd=1e-12;
        for(int j=0;j<HCM;j++) pd+=(double)ps[j]*ps[j];
        for(int d2=0;d2<DIM;d2++){ double a2=0;
            for(int j=0;j<HCM;j++) a2+=(double)ps[j]*((double)Hf[((size_t)s*HCM+j)*DIM+d2]-(double)Hq[((size_t)s*HCM+j)*DIM+d2]);
            DF[(size_t)s*DIM+d2]=(float)(a2/pd); }
    }
}
/* ===== 合并动态侧车(最终产物): 逐层胜者 {algo,params,payload}, 结束一次性落盘 ===== */
typedef struct { int L; char algo[12]; float lam,g; int k,backward;
                 float valb,vala,heldb,helda; uint8_t*pay; uint64_t paysz; } zrec_t;
static zrec_t ZREC[NL]; static int NZREC=0;
static void zfile_write(void){
    if(!NZREC) return;
    const char*p=g_cli.zfile;
    FILE*f=fopen(p,"wb"); if(!f){ fprintf(stderr,"[zfile] 写 %s 失败\n",p); return; }
    uint32_t magic=0x315A5144; uint32_t nr=(uint32_t)NZREC;   /* "DQZ1" */
    fwrite(&magic,4,1,f); fwrite(&nr,4,1,f);
    uint64_t tot=8;
    for(int i=0;i<NZREC;i++){ zrec_t*r=&ZREC[i];
        fwrite(&r->L,4,1,f); fwrite(r->algo,1,12,f); fwrite(&r->lam,4,1,f);
        fwrite(&r->g,4,1,f); fwrite(&r->k,4,1,f); fwrite(&r->backward,4,1,f);
        fwrite(&r->paysz,8,1,f);
        if(r->paysz&&r->pay) fwrite(r->pay,1,r->paysz,f);
        tot+=40+r->paysz;
    }
    fclose(f);
    printf("ZFILE %s layers=%d bytes=%llu (%.2f MB)\n",p,NZREC,(unsigned long long)tot,(double)tot/1048576.0);
    fflush(stdout);
}

/* base 量化遍(worker 并行): iter=0 全量+缓存, iter=1 向后重解(w2 目标=系数校正后残差) */
static long co_base_pass(int L,int S,int n_fit,int iter,const float*Fin,const int*idx,const float*rw,
                         coexp_t*ex,float*routed,lstat_t*st){
    memset(routed,0,(size_t)S*DIM*4);
    int e_next=0,nth=NTHREADS<1?1:(NTHREADS>NEXP?NEXP:NTHREADS);
    cowork_t *ws=calloc((size_t)nth,sizeof(cowork_t));
    pthread_t *th=malloc((size_t)nth*sizeof(pthread_t));
    long nflip=0;
    for(int t=0;t<nth;t++){
        ws[t]=(cowork_t){L,S,n_fit,iter,Fin,idx,rw,ex,&e_next,calloc((size_t)S*DIM,4),0,0,0,0};
        pthread_create(&th[t],NULL,coadapt_worker,&ws[t]);
    }
    for(int t=0;t<nth;t++){ pthread_join(th[t],NULL);
        for(size_t i=0;i<(size_t)S*DIM;i++) routed[i]+=ws[t].fout[i];
        free(ws[t].fout); nflip+=ws[t].nflip;
        if(iter==0&&st){ st->calib_rows+=ws[t].calib_rows; st->calib_empty+=ws[t].calib_empty; st->nhit+=ws[t].nhit_experts; }
    }
    free(ws);free(th);
    return nflip;
}
/* 胜者 payload 打包(合并侧车用, fp16) */
static uint8_t *co_pay_mult_stat(coexp_t*ex,uint64_t*sz){   /* GE: g_e ×NEXP */
    *sz=(uint64_t)NEXP*2; uint8_t*p=malloc((size_t)*sz); uint16_t*h=(uint16_t*)p;
    for(int e=0;e<NEXP;e++) h[e]=go1b_fp32_to_fp16(ex[e].gstat!=0.0f?ex[e].gstat:1.0f);
    return p;
}
static uint8_t *co_pay_mult_dyn(coexp_t*ex,const float*afin,uint64_t*sz){   /* CE: v_e=Σα·x fp16[DIM]×NEXP */
    *sz=(uint64_t)NEXP*DIM*2; uint8_t*p=malloc((size_t)*sz); uint16_t*h=(uint16_t*)p;
    for(int e=0;e<NEXP;e++){ coexp_t*ce=&ex[e];
        for(int d2=0;d2<DIM;d2++){ double v=0;
            if(ce->alpha&&ce->ncal)
                for(int t=0;t<ce->ncal;t++){ if(ce->alpha[t]==0.0f)continue;
                    v+=(double)ce->alpha[t]*afin[(size_t)ce->calS[t]*DIM+d2]; }
            h[(size_t)e*DIM+d2]=go1b_fp32_to_fp16((float)v); } }
    return p;
}
static uint8_t *co_pay_zl(const ds4_z*zl,int k,uint64_t*sz){   /* ZL: V[d,k],U[d,k],z[k] fp16 */
    int d=(int)zl->d_in,dout=(int)zl->d_out;
    *sz=(uint64_t)k*(d+dout+1)*2; uint8_t*p=malloc((size_t)*sz); uint16_t*h=(uint16_t*)p; size_t o=0;
    for(int c=0;c<k;c++) for(int i=0;i<d;i++)    h[o++]=go1b_fp32_to_fp16(zl->V[(size_t)i*zl->rank+c]);
    for(int c=0;c<k;c++) for(int j=0;j<dout;j++) h[o++]=go1b_fp32_to_fp16(zl->U[(size_t)j*zl->rank+c]);
    for(int c=0;c<k;c++) h[o++]=go1b_fp32_to_fp16(zl->z[c]);
    return p;
}
static void co_rel(const float*Hq,const float*Hf,int a,int b,size_t rowsz,double*rel){
    double e2=0,x2=0;
    for(size_t i=(size_t)a*rowsz;i<(size_t)b*rowsz;i++){
        double dd=(double)Hq[i]-Hf[i]; e2+=dd*dd; x2+=(double)Hf[i]*Hf[i]; }
    *rel=sqrt(e2/(x2+1e-30));
}
static double co_score(const float*Hq,const float*Hf,int a,int b,size_t rowsz){
    int n=b-a; if(n<1) return 0.0;
    const float *A=Hq+(size_t)a*rowsz,*B=Hf+(size_t)a*rowsz;
    float *wv=malloc(rowsz*4); ds4_loss_dim_variance(B,(uint32_t)n,(uint32_t)rowsz,wv);
    double la=ds4_loss_align(A,B,(uint32_t)n,(uint32_t)rowsz);
    double lc=ds4_loss_classify(A,B,wv,(uint32_t)n,(uint32_t)rowsz);
    free(wv);
    return la+0.5*lc;
}
/* ★行表版打分(2026-08-29)★ rows[nr]=【原始行号】, gather 成紧凑前缀后走【同一个】co_score。
 * 为什么必须有它: 旧的 (a,b) 连续区间在"域按块连续铺"的语料上只覆盖 1-2 个域 —— 实测
 * [4608,6144) = 西里尔后半+math, 而 sweep 的候选择优/落地/终验【全部】用这个区间打分,
 * 于是 42 层"层内正向落地 + 终验✓改善"在判决份 8 域尺上四项全负。行表由布局推导 ⇒ 跨全域。 */
/* ★BFUNIT 单元账(2026-08-30 用户"改成GPU了怎么还这么慢")★ 实测 L27 单元 234s 里 [bmwt]
 * 只占 28s, 其余在账外(主线程单核 99.9%, GPU 采样半数 0%)。sweep 单元臂内(g_bflt_on)
 * 逐段累计: 0..7=g_lt 同槽位 8=打分(co_score_rows+bf_rowdist) 9=layer_fwd 总墙钟
 * 10=前向次数 11=骨干fp16展开(lwh_expand/load_layer, gs_forward_exit 每层每候选翻炒)。
 * 首账(champ86amp L40): 层前向77s/单元81s, attn=26 moe=34 路由=16, 展开/打分/胶水≈0 →
 * 二级子账: 12=hc(pre/post/rms) 13=compressor 14=attn核 | 16=bdq 17=gemm+sync 18=gather
 * 19=scatter 20=池/memset 21=归约 22=修正链回放(含z) 23=其中z(type6双投影,单线程嫌疑#1)。
 * 胶水=单元墙钟−[9]−[8]−[11](malloc/gather/zrefit/pca/落地IO)。量化遍/终验不臂→零扰动。 */
double g_bflt[26]; int g_bflt_on=0, g_pos_off=0;   /* 24/25=attn投影/输出投影; g_pos_off=bkl真位置偏移(2026-08-31) */
double vqt_now_ref(void){ return vqt_now(); }   /* fwd_p2 埋点用(fwd 在 vq_qc.h 之前 include, 看不见 static inline vqt_now) */
static void bflt_print(int J,double uw){
    if(!g_bflt_on) return; g_bflt_on=0;
    double fw=g_bflt[9],sc=g_bflt[8],xp=g_bflt[11],gl=uw-fw-sc-xp; if(gl<0)gl=0;
    printf("[BFLT] L=%02d 层前向%d次=%.0fs(attn=%.0f[hc%.0f/comp%.0f/核%.0f=投%.0f+带%.0f+出%.0f+胶] 路由=%.0f 共享=%.0f moe=%.0f[bdq%.0f/gemm%.0f/取%.0f/散%.0f/池%.0f/归%.0f/链%.0f内z%.0f] lfload=%.0f 其余=%.0f zl=%.0f) 展开=%.0f 打分=%.0f 胶水=%.0f | 单元=%.0fs\n",
           J,(int)g_bflt[10],fw,g_bflt[0],g_bflt[12],g_bflt[13],g_bflt[14],g_bflt[24],g_bflt[15],g_bflt[25],g_bflt[1],g_bflt[2],g_bflt[3],
           g_bflt[16],g_bflt[17],g_bflt[18],g_bflt[19],g_bflt[20],g_bflt[21],g_bflt[22],g_bflt[23],
           g_bflt[6],g_bflt[4],g_bflt[5],xp,sc,gl,uw);
    fflush(stdout);
}
/* ★dirn=1 方向域(2026-08-30 判据病修)★ head 的 rms 抹掉每 token 模长 → 打分前逐行 rms
 * 归一, 只计 head 看得见的方向分量。42 单元实锤: L2 版被模长假肉喂饱(隐分↑/cos平/KL↑),
 * 真肉(L31/L19 型)反被打负分。sweep 侧全部传 1; 传 0=原语义(其他消费者不动)。 */
static void rows_rmsn(float*g,int nr,size_t rowsz){
    for(int i=0;i<nr;i++){ float*r=g+(size_t)i*rowsz; double n2=0;
        for(size_t k=0;k<rowsz;k++) n2+=(double)r[k]*r[k];
        float sc=(float)(1.0/sqrt(n2/rowsz+1e-12));
        for(size_t k=0;k<rowsz;k++) r[k]*=sc; }
}
static double co_score_rows(const float*Hq,const float*Hf,const int*rows,int nr,size_t rowsz,int dirn){
    if(nr<1||!rows) return 0.0;
    double _t0=g_bflt_on?vqt_now():0.0;
    float *ga=malloc((size_t)nr*rowsz*4), *gb=malloc((size_t)nr*rowsz*4);
    if(!ga||!gb){ free(ga); free(gb); return 0.0; }
    for(int i=0;i<nr;i++){
        memcpy(ga+(size_t)i*rowsz, Hq+(size_t)rows[i]*rowsz, rowsz*4);
        memcpy(gb+(size_t)i*rowsz, Hf+(size_t)rows[i]*rowsz, rowsz*4); }
    if(dirn){ rows_rmsn(ga,nr,rowsz); rows_rmsn(gb,nr,rowsz); }
    double v=co_score(ga,gb,0,nr,rowsz);
    free(ga); free(gb); if(g_bflt_on) g_bflt[8]+=vqt_now()-_t0; return v;
}
/* ★base 非有限闸(2026-08-30 L11 实案)★ champ86amp sweep 34 单元炸 1 次: base 遍前向内
 * 产生 NaN(zdiag |z|/|routed| 打 0.0000 是 nf>0 对 NaN 为假的掩码), 污染链 = NaN base →
 * bf_vertex→ms→zrefit 拟出 NaN 系数→形态E/F 两遍 NaN 前向全废, Δbest=+nan% 假"保持"。
 * 单发+只在 fresh-dequant 首遍 ⇒ 头号嫌疑 GB10 托管内存瞬态, 日志无法定谳。此闸不是兜底:
 * ①打印 NaN行/全零行 取证 ②复跑一次当场判"瞬态(复跑有限, 用有效值继续)/确定性(仍非有限,
 * 跳过单元防污染)" —— 每次触发都留完整证据链。 */
static float *gs_forward_exit(int J,int Lend,const float*Hin,const long*ids,int S,int n_fit,float*HQcache);
/* bkl 真尺(定义在 p12; p7 的 ZLGATE 链上落地闸要用, include 序在前 ⇒ 前置声明) */
static int bkl_init(const float*Htgt,const long*ids,int S,int n_fit,size_t rowsz);
static double bkl_exit_kl(const float*Hex,size_t rowsz);
static double bf_base_gate(int J,double base,float**Hb,int eF,const float*eHin,const long*eIds,
                           int eS,int eNf,const float*eTgt,const int*SEV,int nSEV,size_t rowsz){
    if(isfinite(base)) return base;
    size_t nanr=0,zr=0;
    for(int s=0;s<eS;s++){ const float*r=*Hb+(size_t)s*rowsz; int h=0; double e=0;
        for(size_t i=0;i<rowsz;i++){ if(!isfinite(r[i])) h=1; else e+=fabs(r[i]); }
        if(h) nanr++; else if(e==0.0) zr++; }
    printf("★[BF诊断] L=%02d base=%g 非有限: 出口H NaN行=%zu/%d 全零行=%zu — 复跑一次判瞬态/确定性\n",
           J,base,nanr,eS,zr); fflush(stdout);
    free(*Hb); *Hb=gs_forward_exit(J,eF,eHin,eIds,eS,eNf,NULL);
    double b2=co_score_rows(*Hb,eTgt,SEV,nSEV,rowsz,1);
    printf(isfinite(b2)?"★[BF诊断] L=%02d 复跑base=%.6g 有限 → ★瞬态实锤(嫌疑=GB10托管内存)★ 单元以复跑值继续\n"
                        :"★[BF诊断] L=%02d 复跑base=%g 仍非有限 → 确定性异常, 单元跳过\n",J,b2); fflush(stdout);
    return b2;
}

/* ===== 从层文件重前向(跨层反向的执行引擎): 文件即真相 =====
 * 读 dql_L<NN>.bin: 1bit 权重字节 + 落地修正记录(按时间序回放 GL/dyn2/dyn8/TREF)。
 * 不重量化 — 直接 dequant 字节 → 专家前向 → 修正链回放。 */
typedef struct { int type; float g,t; float w2p[4]; float w8[9]; float *V8; float *ge;
                 float *zlU,*zlV,*zlz; int zlk; float zltr;   /* 6=zl.RRR 冻结秩-k 方向修正(产物③) */
                 int zdin;   /* ★md86: 输入维 DIM=线性 | 3*DIM=ftA 特征提升(φ=[x,x²/rms,relu]) */
                 int erf_ne,erf_r; uint32_t *erf_eid; float *erf_tau; float *erf_UV;
                 /* 8=zl.ERF 死层部件(2026-08-12): 每专家 w2 低秩补丁 U[D,r](折S)·V[r,MOEI](折α)+token能量门τ */
                 size_t foff;     float ghc[2];   /* type7 GLhc: g_hot, g_cold(2026-08-06 冷热双通道) */
                 int ext;   /* 1=记录宿主是 zrec 外挂(foff 不在 dql, 禁 zfile_commit 原地改写) */
} lop_t;   /* 1=GL 2=dyn2 3=dyn8 4=TREF 5=GE(per-expert增益,累加时乘) 6=zl.RRR 8=zl.ERF; foff=载荷文件偏移(反修原地改写用) */
typedef struct { uint8_t *map; size_t msz; const uint8_t *w1,*w3,*w2; size_t szG,szD;
                 uint8_t *g2map; size_t g2msz;                                    /* go2b 侧车 mmap(可异机盘/NFS; 内嵌时 NULL, g2w* 指进主 map) */
                 const uint8_t *g2w1,*g2w3,*g2w2; int g2k; int16_t g2slot[256];   /* 热专家 2bit 覆盖 */
                 uint8_t *vqmap; size_t vqmsz;                                    /* v2.2 VQ 侧车 mmap(dql_vq_L%02d.bin) */
                 uint8_t *opsmap; size_t opsmsz; int has_ops;                     /* op 侧车 mmap(dql_ops_L%02d.bin, 平行架构权威源) */
#define LOPS_MAX 32   /* 单层 op 链容量; 超限=停车(修正静默消失过的判决不可信), 不静默丢 */
                 lop_t ops[LOPS_MAX]; int nops;
                 int zl_stub_at; } lfile_t;   /* zl.RRR 空壳占位序号(真载荷回填正位用; -1=无) */
/* go2b 侧车路径: --go2b-dir(默认=层文件同目录)/dql_go2b_L<NN>.bin — 分储设计:
 * 冷 go1b dql 在本机(热专家稀疏洞), 热 go2b 侧车可放对机 NFS(两机 16G 盘都装不下合体) */
static void g2_sidecar_path(const char*dql_path,int L,char*out,size_t outsz){
    const char*gd=g_cli.go2b_dir;
    if(gd){ snprintf(out,outsz,"%s/dql_go2b_L%02d.bin",gd,L); return; }
    char dir[512]; snprintf(dir,sizeof(dir),"%s",dql_path);
    char*sl=strrchr(dir,'/'); if(sl)*sl=0; else snprintf(dir,sizeof(dir),".");
    snprintf(out,outsz,"%s/dql_go2b_L%02d.bin",dir,L);
}
/* 侧车头: 'DQG2' u32 | ver u32 | L u32 | khot u32 | mean_cos f32 | rsv u32 | ids u16[khot] | pad8 → 载荷 */
#define G2SC_MAGIC 0x32475144u
static size_t g2_sidecar_hdr(int khot){ return (24+2*(size_t)khot+7)&~(size_t)7; }
/* ★op 侧车(2026-08-04 用户架构令: 量化模型与 z/动态平行, 反修不许动量化文件)★
 * dql_LXX.bin = 纯权重字节(1bit/g2hot), 量化后只读; 全部 op 记录(z 家族/zl.RRR/loss/bf/bwd)
 * 住 dql_ops_LXX.bin('DQO2' u32|L u32|nrec u32 + DQL2 同款记录), 应用序=侧车文件序(单一权威)。
 * 反修/sweep/回扫只写侧车 — 重量化在架构上不再可能被需要。旧混装 dql 兼容读(侧车缺席时)。 */
#define OPSC_MAGIC 0x324F5144u   /* 'DQO2' */
static void ops_sidecar_path(const char*dql_path,int L,char*out,size_t outsz){
    char dir[512]; snprintf(dir,sizeof(dir),"%s",dql_path);
    char*sl=strrchr(dir,'/'); if(sl)*sl=0; else snprintf(dir,sizeof(dir),".");
    snprintf(out,outsz,"%s/dql_ops_L%02d.bin",dir,L);
}
/* op 宿主文件 = dql 主文件(超冠 573b7f5 混装架构; 2026-08-08 用户令还原:
 * "侧车不是我要求加入的" — 平行架构 op 侧车 2026-08-04 系擅自引入, 已删)。 */
static void op_host_path(int L,char*out,size_t outsz){
    snprintf(out,outsz,"%s/dql_L%02d.bin",dsq_layer_dir_req(),L);
}
