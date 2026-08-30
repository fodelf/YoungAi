        /* 护栏碑已删除(2026-08-18 用户令"删除里程碑评估"; 08-04 已裁"没有意义"):
         * 质量判决由收官 VERDICT/五指标全权。 */
        /* 单层探针(2026-07-28 用户: "先跑一层看看, 不要蒙头就跑"): 锁满 MAXL 层即收工。
         * plan/ckpt/产物均已落盘; zfile/zchain 也刷终值(全程跑在 main 尾做, 探针早退补齐);
         * RESUME=1 从下一层无损续跑。 */

        if(minvol&&getenv("DS4_MINVOL_MAXL")&&L+1>=atoi(getenv("DS4_MINVOL_MAXL"))){
            rb_save(); zfile_write(); zchain_write();
            fprintf(stderr,"[贪心探针] 已锁 %d 层(DS4_MINVOL_MAXL) → 提前收工; RESUME=1 续跑\n",L+1);
            exit(0);
        }
    }
    plan_out[NLAYERS]=0;
    rb_save();   /* 序贯路由: 收官落盘 Δb+α(交付侧车) */
    free(Hw);free(Hb);free(Hf2); if(H2)free(H2);
    float *hcfn=st_read_weight(&C,"hc_head_fn",NULL,NULL),*hcb=st_read_weight(&C,"hc_head_base",NULL,NULL),*hcs=st_read_weight(&C,"hc_head_scale",NULL,NULL);
    float *norm=st_read_weight(&C,"norm.weight",NULL,NULL);
    float *logits=malloc((size_t)S*VOCAB*4); head_fwd_stream(H,S,hcfn,hcb,hcs,norm,logits);
    free(H);free(hcfn);free(hcb);free(hcs);free(norm);
    return logits;
}

/* ===================== 最终输出判决 ===================== *
 * held-out 行 [n_fit,S-1)。主判据(还原率铁律): 分布还原率 Σmin / KL(fp‖q) / PPL ratio;
 * top-1 已退役(teacher-forced 易文本虚高), 仅参考输出。VERDICT 单行供扫描脚本解析。 */
static void verdict(const float*lf,const float*lq,const long*ids,int S,int n_fit,const char*lcfg){
    int s0=n_fit,s1=S-1; if(s1-s0<1){ s0=0; fprintf(stderr,"⚠ held-out<1, 退回全序列\n"); }
    double nllf=0,nllq=0,smin=0,kl=0; int n=0,tf=0,tq=0,ag=0;
    for(int s=s0;s<s1;s++){ long tgt=ids[s+1];
        const float*a=lf+(size_t)s*VOCAB,*b=lq+(size_t)s*VOCAB;
        float ma=a[0],mb=b[0]; int amax=0,bmax=0;
        for(int v=1;v<VOCAB;v++){ if(a[v]>ma){ma=a[v];amax=v;} if(b[v]>mb){mb=b[v];bmax=v;} }
        double sa=0,sb=0; for(int v=0;v<VOCAB;v++){ sa+=exp((double)a[v]-ma); sb+=exp((double)b[v]-mb); }
        double lsa=log(sa),lsb=log(sb),srow=0,krow=0;
        for(int v=0;v<VOCAB;v++){
            double lpf=(double)a[v]-ma-lsa, lpq=(double)b[v]-mb-lsb;
            double pf=exp(lpf), pq=exp(lpq);
            srow += pf<pq?pf:pq;
            if(pf>0) krow += pf*(lpf-lpq);
        }
        smin+=srow; kl+=krow;
        nllf += -((double)a[tgt]-ma-lsa); nllq += -((double)b[tgt]-mb-lsb);
        tf+=(amax==(int)tgt); tq+=(bmax==(int)tgt); ag+=(amax==bmax); n++;
    }
    double pplf=exp(nllf/n),pplq=exp(nllq/n);
    printf("\n=== 最终输出判决 (held-out %d tok · 主判据=分布还原率Σmin/KL/PPL, top1 仅参考) ===\n",n);
    if(NLAYERS<NL) printf("  ⚠ 截断 NL=%d/%d: 绝对 PPL 无效; fp/quant 同深度 → 相对指标(ratio/Σmin/KL)仍可比\n",NLAYERS,NL);
    printf("  fp    : PPL=%12.4f  top1(vs真值)=%.1f%%\n",pplf,100.0*tf/n);
    printf("  quant : PPL=%12.4f  top1(vs真值)=%.1f%%\n",pplq,100.0*tq/n);
    double eg=lcfg_expert_gib(lcfg);
    double zmb=(double)LZ_TOTAL_K*(2.0*DIM+1.0)*2.0/1048576.0;   /* z^L 侧车: U/V/z fp16 */
    printf("  ★ PPL ratio=%.4fx   分布还原率Σmin=%.4f   KL(fp‖q)=%.4f nats   top1一致(vs fp)=%.1f%%\n",
           pplq/pplf,smin/n,kl/n,100.0*ag/n);
    printf("  体积: 专家=%.1f GiB + z^L侧车=%.2f MB (骨干Q8另+8.4)\n",eg,zmb);
    printf("VERDICT lcfg=%s S=%d held=%d pplf=%.4f pplq=%.4f ratio=%.4f smin=%.4f kl=%.4f agree=%.1f top1f=%.1f top1q=%.1f expgib=%.2f zmb=%.2f\n",
           lcfg,S,n,pplf,pplq,pplq/pplf,smin/n,kl/n,100.0*ag/n,100.0*tf/n,100.0*tq/n,eg,zmb);
}

/* ===== 全局联合回扫(★最终输出为判据, 非单层★): 43 层量化落地后, 多轮回扫,
 * 每层 GBL_G[L] 以【最终 logits KL vs FP 锚】判优, 循环到最终质量收敛。
 * 引擎: 全模型 'B' 字节前向(不重量化)+ head; 缓存每层入口 H, 改层 L 只重跑 L..42。 */
static float *gs_forward_from(int L0,const float *Hin,const long*ids,int S,int n_fit,
                              float *Hcache /*可选: 存每层入口 H, [NLAYERS+1][lstride]*/){
    size_t lstride=(size_t)S*HCM*DIM;
    float *H=malloc(lstride*4); memcpy(H,Hin,lstride*4);
    for(int L=L0;L<NLAYERS;L++){
        if(Hcache) memcpy(Hcache+(size_t)L*lstride,H,lstride*4);
        if(GS_LW&&GS_LW[L].loaded){ LW*_c32=lw32_get(L,&GS_LW[L]); LW T=_c32?*_c32:lwh_expand(&GS_LW[L]);
            if(getenv("DS4_GS_DIAG")){ fprintf(stderr,"[回扫诊断] L%d cache态 t2ei=%p 缺字段:",L,(void*)T.t2ei);
#define XN(f) if(!T.f) fprintf(stderr," %s",#f);
                LWF_LIST(XN)
#undef XN
                fprintf(stderr," <end>\n"); }
            layer_fwd(L,&T,H,ids,S,n_fit,1,'B',NULL); if(!_c32) free_layer(&T); }   /* ★fp16缓存展开(lw32 常驻优先)★ */
        else { LW W=load_layer(L);
            if(getenv("DS4_GS_DIAG")) fprintf(stderr,"[回扫诊断] L%d 重载态 t2ei=%p gate=%p ids=%p rowmap=%p\n",
                    L,(void*)W.t2ei,(void*)W.gate,(const void*)ids,(void*)g_anc_rowmap);
            layer_fwd(L,&W,H,ids,S,n_fit,1,'B',NULL); free_layer(&W); }   /* 未缓存/被驱逐: 临时重载 */
    }
    if(Hcache) memcpy(Hcache+(size_t)NLAYERS*lstride,H,lstride*4);
    float *lg=malloc((size_t)S*VOCAB*4);
    if(GS_HW){ head_fwd(H,S,GS_HCFN,GS_HCB,GS_HCS,GS_NORM,GS_HW,lg); }   /* ★缓存 head, 禁重读 2.1GB★ */
    else { float *hcfn=st_read_weight(&C,"hc_head_fn",NULL,NULL),*hcb=st_read_weight(&C,"hc_head_base",NULL,NULL),*hcs=st_read_weight(&C,"hc_head_scale",NULL,NULL);
        float *norm=st_read_weight(&C,"norm.weight",NULL,NULL),*hw=st_read_weight(&C,"head.weight",NULL,NULL);
        head_fwd(H,S,hcfn,hcb,hcs,norm,hw,lg);
        free(hcfn);free(hcb);free(hcs);free(norm);free(hw); }
    free(H);
    return lg;
}
/* ★真·反修前层(判据=最终输出 KL, 全局联合最优)★
 * 第 L 层的 z 系数用【最终 logits KL】重解, 不再只往文件尾巴粘标量:
 *   每 token 搜最优 routed 乘子 m_s* → 目标系数 new_c=m_s*·cur_c → 同特征最小二乘重拟合 →
 *   原地改写该层文件 z 载荷(内容真的变, mtime/字节变)。只有最终 val-KL 下降才提交, 否则回退(永不劣化)。
 * 返回 1=已改写该前层文件。 */
/* z 系数最小二乘重拟合(内存): 目标 new_c(x)=ms[s]·cur_c(x), 同特征(GL/GLdyn2/GLdyn8)拟合。
 * Fin=该层 MoE 输入(算特征+当前系数), NF=拟合行数。仅改内存, 落盘由 zfile_commit 做。*/
static void zrefit_w(lop_t*z,const float*ms,int NF,const float*Fin,const float*wrow);
static void zrefit(lop_t*z,const float*ms,int NF,const float*Fin){ zrefit_w(z,ms,NF,Fin,NULL); }
/* 行权版(2026-08-05 用户铁律: 分类/感知行权布线进 ONEPASS): wrow=每行权重(NULL=恒1),
 * 语义与 ALT 阶段2 G0+w*G1 同源 — 残差大的行在最小二乘中话语权大。 */
static void zrefit_w(lop_t*z,const float*ms,int NF,const float*Fin,const float*wrow){
    if(z->type==1){
        double num=0,den=0; for(int s=0;s<NF;s++){ double wr=wrow?wrow[s]:1.0;
            num+=wr*(double)ms[s]*z->g; den+=wr; }
        z->g=(float)(num/(den>0?den:1));
    } else if(z->type==2){
        double A[4]={0,0,0,0},b2[2]={0,0},x2[2];
        for(int s=0;s<NF;s++){ const float*fx=Fin+(size_t)s*DIM; double v=0;
            for(int d=0;d<DIM;d++) v+=(double)fx[d]*fx[d];
            double f=((sqrt(v)-(double)z->w2p[2])/(double)z->w2p[3]);
            double cur=(double)z->w2p[0]+(double)z->w2p[1]*f, tgt=(double)ms[s]*cur;
            double wr=wrow?wrow[s]:1.0;
            A[0]+=wr; A[1]+=wr*f; A[3]+=wr*f*f; b2[0]+=wr*tgt; b2[1]+=wr*f*tgt; }
        /* ★四损失接线(2026-08-05 用户铁律: 架构组件不许连坐砍除)★
         * lfix=ridge 总体倍率(固定损失), lsm=非截距(动态系数)额外光滑惩罚 — 与 ALT KGRID 同语义 */
        A[2]=A[1]; double lam=1e-3*(A[0]+A[3])/2.0*DYN_LFIX+1e-9;
        A[0]+=lam; A[3]+=lam*(DYN_LSM/0.1);   /* df(0.1) 锚定恒等旧解(2026-08-05 bug修: 旧公式 1+lsm*10 把默认组翻倍) */
        if(solve_sym(A,b2,2,x2)==0){ z->w2p[0]=(float)x2[0]; z->w2p[1]=(float)x2[1]; }
    } else if(z->type==3&&z->V8){
        double A[81],b9[9],x9[9]; for(int i=0;i<81;i++)A[i]=0; for(int i=0;i<9;i++)b9[i]=0;
        double phi[9];
        for(int s=0;s<NF;s++){ const float*fx=Fin+(size_t)s*DIM; phi[0]=1;
            for(int c=0;c<8;c++){ double a2=0; const float*vc=z->V8+(size_t)c*DIM;
                for(int d=0;d<DIM;d++) a2+=(double)fx[d]*vc[d]; phi[1+c]=a2; }
            double cur=(double)z->w8[0]; for(int c=0;c<8;c++) cur+=(double)z->w8[1+c]*phi[1+c];
            double tgt=(double)ms[s]*cur;
            double wr=wrow?wrow[s]:1.0;
            for(int r=0;r<9;r++){ b9[r]+=wr*phi[r]*tgt;
                for(int c=0;c<9;c++) A[(size_t)r*9+c]+=wr*phi[r]*phi[c]; } }
        double tr=0; for(int i=0;i<9;i++) tr+=A[(size_t)i*9+i];
        double lam=1e-3*tr/9.0*DYN_LFIX+1e-9;                 /* 固定损失: λ 总体倍率 */
        A[0]+=lam;
        for(int i=1;i<9;i++) A[(size_t)i*9+i]+=lam*(DYN_LSM/0.1);   /* 光滑: df(0.1) 锚定恒等旧解 */
        if(solve_sym(A,b9,9,x9)==0){ for(int i=0;i<9;i++) z->w8[i]=(float)x9[i]; }
    }
}
/* ★首写留档(2026-08-04 流程债)★: 反修 commit 原地覆盖量化段 op 原值 ⇒ 纯量化态从盘上消失,
 * 返修只能重量化(2.6h)。每个 (L,foff) 第一次改写前把原字节 append 进 layers/opbak_LXX.bin
 * (u64 foff|u32 len|bytes) — 恢复=按表回写, 反修从此可无损重来。 */
/* ONEPASS sweep 的进程内 undo 表(终验劣化=全回滚): op_backup 在 BFU_ARM 时旁录旧值 */
typedef struct { int L; size_t foff; uint32_t len; uint8_t*old; } bfu_t;
static bfu_t *BFU=NULL; static int NBFU=0,BFU_CAP=0,BFU_ARM=0;
static void op_backup(int L,int fd,size_t foff,size_t len){
    if(BFU_ARM&&len<=65536){
        if(NBFU>=BFU_CAP){ BFU_CAP=BFU_CAP?BFU_CAP*2:64; BFU=realloc(BFU,(size_t)BFU_CAP*sizeof(bfu_t)); }
        bfu_t*u=&BFU[NBFU]; u->L=L; u->foff=foff; u->len=(uint32_t)len; u->old=malloc(len);
        if(pread(fd,u->old,len,(off_t)foff)==(ssize_t)len) NBFU++; else free(u->old);
    }
    static uint64_t seen[4096]; static int nseen=0;   /* (L<<48|foff) 首写集(单进程规模足够) */
    uint64_t key=((uint64_t)L<<48)|(uint64_t)foff;
    for(int i=0;i<nseen;i++) if(seen[i]==key) return;
    if(nseen<4096) seen[nseen++]=key;
    uint8_t old[8192]; if(len>sizeof(old)) return;
    if(pread(fd,old,len,(off_t)foff)!=(ssize_t)len) return;
    char bp[512]; snprintf(bp,sizeof(bp),"%s/opbak_L%02d.bin",
        getenv("DS4_LAYER_DIR")?getenv("DS4_LAYER_DIR"):".",L);
    FILE*bf=fopen(bp,"ab"); if(!bf) return;
    uint64_t f64=(uint64_t)foff; uint32_t l32=(uint32_t)len;
    fwrite(&f64,8,1,bf); fwrite(&l32,4,1,bf); fwrite(old,1,len,bf); fclose(bf);
}
/* z 载荷原地改写(内容真的变, mtime/字节变) + 记录 mean[0]=新判据
 * ★平行架构: 宿主=op 侧车(在则), 量化 dql 永不被 commit 触碰 */
static void zfile_commit(int L,lop_t*z,float m1metric){
    char lp[512]; op_host_path(L,lp,sizeof(lp));
    int fd=open(lp,O_RDWR); if(fd<0){ perror("zcommit-open"); return; }
    { size_t blen=z->type==1?4:z->type==2?16:z->type==4?4:z->type==5?(size_t)NEXP*2:36;
      op_backup(L,fd,z->foff,blen); op_backup(L,fd,z->foff-20,4); }
    if(z->type==1){ if(pwrite(fd,&z->g,4,(off_t)z->foff)!=4) perror("bf-w1"); }
    else if(z->type==2){ if(pwrite(fd,z->w2p,16,(off_t)z->foff)!=16) perror("bf-w2"); }
    else if(z->type==4){ if(pwrite(fd,&z->t,4,(off_t)z->foff)!=4) perror("bf-t"); }
    else if(z->type==5&&z->ge){ uint16_t*h=malloc((size_t)NEXP*2);
        for(int e=0;e<NEXP;e++) h[e]=go1b_fp32_to_fp16(z->ge[e]);
        if(pwrite(fd,h,(size_t)NEXP*2,(off_t)z->foff)!=(ssize_t)((size_t)NEXP*2)) perror("bf-ge");
        free(h); }
    else { if(pwrite(fd,z->w8,36,(off_t)z->foff)!=36) perror("bf-w8"); }
    if(pwrite(fd,&m1metric,4,(off_t)(z->foff-20))!=4) perror("bf-m1");
    close(fd);
}
static int backfit_layer_z(int L,const long*ids,int S,int n_fit,float*Hc,size_t lstride,double*kl0){
    if(!GS_LF||L>=NLAYERS||!GS_LF[L].map) return 0;
    lfile_t*lf=&GS_LF[L]; int zi=-1;
    for(int i=0;i<lf->nops;i++) if(lf->ops[i].type>=1&&lf->ops[i].type<=3){ zi=i; break; }
    if(zi<0) return 0;                          /* 该层无 z 载荷可重解 */
    lop_t*z=&lf->ops[zi]; lop_t zbak=*z;        /* 备份系数(回退用; V8 指针共享不动) */
    int vs=(n_fit*3)/4; const float*Hin=Hc+(size_t)L*lstride;
    if(!GS_FIN) GS_FIN=malloc((size_t)S*DIM*4);
    /* (a) 基线: 捕获 Fin_L(算 z 特征) + per-token 基线 KL */
    GS_CAP_L=L; GS_GV=NULL; GS_GV_L=-1;
    float*lgb=gs_forward_from(L,Hin,ids,S,n_fit,NULL); GS_CAP_L=-1;
    float*ms=malloc((size_t)S*4); double*kb=malloc((size_t)S*sizeof(double));
    for(int s=0;s<S;s++){ ms[s]=1.0f; kb[s]=bwd_tok_kl(ANC.logits+(size_t)s*VOCAB,lgb+(size_t)s*VOCAB); }
    free(lgb);
    /* (b) per-token 最优乘子: 网格各前向一次(仅目标层 routed 缩放) */
    const float MG[4]={0.8f,0.9f,1.1f,1.25f};
    GS_GV=malloc((size_t)S*4); GS_GV_L=L;
    for(int gi=0;gi<4;gi++){ float m=MG[gi];
        for(int s=0;s<S;s++) GS_GV[s]=m;
        float*lgc=gs_forward_from(L,Hin,ids,S,n_fit,NULL);
        for(int s=0;s<S;s++){ double k=bwd_tok_kl(ANC.logits+(size_t)s*VOCAB,lgc+(size_t)s*VOCAB);
            if(k<kb[s]){ kb[s]=k; ms[s]=m; } }
        free(lgc);
    }
    GS_GV_L=-1; free(GS_GV); GS_GV=NULL; free(kb);
    /* (c) 目标 new_c=m_s*·cur_c, 同特征最小二乘重拟合(fit 行) → 写入内存 z */
    zrefit(z,ms,n_fit,GS_FIN); free(ms);
    /* (d) 验证: 新 z 已在内存, 前向到最终, val-KL 改善才提交 */
    float*lgv=gs_forward_from(L,Hin,ids,S,n_fit,NULL);
    double kln=bwd_val_kl(ANC.logits,lgv,vs,n_fit); free(lgv);
    if(kln<*kl0-1e-9){
        zfile_commit(L,z,(float)kln);          /* ★原地改写文件 z 载荷: 内容真的变★ */
        double imp=100.0*(*kl0-kln)/(*kl0>1e-9?*kl0:1);
        char rs[80]; snprintf(rs,80,"最终KL %.5f→%.5f 降%.1f%%",*kl0,kln,imp);
        mlog(L,"向后·反修",z->type==3?"z.GLdyn8 最终KL重解":z->type==2?"z.GLdyn2 最终KL重解":"z.GL 最终KL重解",
             "已改写文件",z->type==3?36+(uint64_t)8*DIM*2:z->type==2?16:4,rs,"✓正向落地");
        { float*lgx=gs_forward_from(L,Hin,ids,S,n_fit,Hc); free(lgx); }   /* 刷新下游 Hc[L+1..](用提交后 z) */
        *kl0=kln; return 1;
    } else {
        *z=zbak;                               /* 回退内存系数(文件未动) */
        char rs[48]; snprintf(rs,48,"最终KL=%.5f(未降)",*kl0);
        mlog(L,"向后·反修","z 最终KL重解","已最优保持",0,rs,"保持");
        return 0;
    }
}
/* ★联合批式回扫轮(2026-08-04 结构病重构)★
 * 旧逐层坐标下降 = 每层 6-7 次【全量行×全后缀】前向 ⇒ L00≈35min, 一轮 7-10h(实测掐停)。
 * 本轮结构: ①探测=每层 5 次【抽行×单层】前向(基线+4乘子网格), 判据=层出口 vs FP锚 ANC.H[L]
 * 的 per-token L2(局部, 零全程前向; 抽行/紧凑 zrefit 与 sweep 粗筛同先例) → 全部层闭式重解
 * ②终验=1 次【全量×全程】val-KL + β信赖域{1,0.5,0.25}(op_blend 向基线退) ③改善才逐层
 * zfile_commit, 否则全回退 — 判据不妥协(质量门)。一轮 ≈ 探测2-3min + 终验6min×β次。
 * DS4_GS_PERCOL=1 走旧逐层版(A/B 对照口径)。 */
static void op_blend(lop_t*dst,const lop_t*o,const lop_t*f,float b);   /* 定义在下方 β信赖域区 */
static int backfit_joint_round(const long*ids,int S,int n_fit,const float*H0,float*Hc,size_t lstride,double*kl0){
    int vs=(n_fit*3)/4;
    size_t rowsz=(size_t)HCM*DIM;
    int gdiv=getenv("DS4_GS_SCREEN_DIV")?atoi(getenv("DS4_GS_SCREEN_DIV")):12; if(gdiv<1)gdiv=1;
    int Sg=0,*gsx=malloc(sizeof(int)*(size_t)(S/gdiv+2));
    for(int s=0;s<n_fit;s+=gdiv) gsx[Sg++]=s;              /* 只抽 fit 行(zrefit 域) */
    if(!GS_FIN) GS_FIN=malloc((size_t)S*DIM*4);
    float *Hg=malloc((size_t)Sg*rowsz*4);                  /* 抽行紧凑层入口 */
    float *Hb=malloc((size_t)Sg*rowsz*4);                  /* 单层前向工作副本 */
    long  *idg=malloc(sizeof(long)*(size_t)Sg);
    float *msg=malloc((size_t)Sg*4);
    double*kbg=malloc(sizeof(double)*(size_t)Sg);
    float *fing=malloc((size_t)Sg*DIM*4);                  /* 基线捕获的 Fin(紧凑) */
    lop_t *zbaks=calloc((size_t)NLAYERS,sizeof(lop_t));
    lop_t *fits =calloc((size_t)NLAYERS,sizeof(lop_t));
    int   *zidx =malloc(sizeof(int)*(size_t)NLAYERS);
    const float MG[4]={0.8f,0.9f,1.1f,1.25f};
    for(int t=0;t<Sg;t++) idg[t]=ids[gsx[t]];
    for(int L=0;L<NLAYERS;L++){
        zidx[L]=-1;
        lfile_t*lf=&GS_LF[L]; if(!lf->map) continue;
        int zi=-1; for(int i=0;i<lf->nops;i++) if(lf->ops[i].type>=1&&lf->ops[i].type<=3){ zi=i; break; }
        if(zi<0) continue;
        zidx[L]=zi; zbaks[L]=lf->ops[zi];
        for(int t=0;t<Sg;t++) memcpy(Hg+(size_t)t*rowsz,Hc+(size_t)L*lstride+(size_t)gsx[t]*rowsz,rowsz*4);
        g_anc_rowmap=gsx; g_anc_rowstride=S;               /* 锚路由按原行号取 */
        /* 基线单层: 捕获 Fin(紧凑) + per-token 出口 L2 vs FP锚 ANC.H[L] */
        GS_CAP_L=L; GS_GV=NULL; GS_GV_L=-1;
        memcpy(Hb,Hg,(size_t)Sg*rowsz*4);
        { float*Ho=gs_forward_exit(L,L,Hb,idg,Sg,Sg,NULL);
          memcpy(fing,GS_FIN,(size_t)Sg*DIM*4); GS_CAP_L=-1;
          for(int t=0;t<Sg;t++){ double e=0; const float*a=Ho+(size_t)t*rowsz,
              *b=ANC.H+(size_t)L*lstride+(size_t)gsx[t]*rowsz;
              for(size_t i=0;i<rowsz;i++){ double d=(double)a[i]-b[i]; e+=d*d; }
              kbg[t]=e; msg[t]=1.0f; }
          free(Ho); }
        /* 4 乘子网格: 单层前向×抽行, per-token 最优 m */
        GS_GV=malloc((size_t)Sg*4); GS_GV_L=L;
        for(int gi=0;gi<4;gi++){ float m=MG[gi];
            for(int t=0;t<Sg;t++) GS_GV[t]=m;
            memcpy(Hb,Hg,(size_t)Sg*rowsz*4);
            float*Ho=gs_forward_exit(L,L,Hb,idg,Sg,Sg,NULL);
            for(int t=0;t<Sg;t++){ double e=0; const float*a=Ho+(size_t)t*rowsz,
                *b=ANC.H+(size_t)L*lstride+(size_t)gsx[t]*rowsz;
                for(size_t i=0;i<rowsz;i++){ double d=(double)a[i]-b[i]; e+=d*d; }
                if(e<kbg[t]){ kbg[t]=e; msg[t]=m; } }
            free(Ho);
        }
        GS_GV_L=-1; free(GS_GV); GS_GV=NULL;
        zrefit(&lf->ops[zi],msg,Sg,fing);                  /* 闭式重解(紧凑行, sweep 同先例) */
        fits[L]=lf->ops[zi];
        g_anc_rowmap=NULL;
    }
    /* 联合终验(全量×全程) + β信赖域: 过冲则向基线退半步再验 */
    const float BETAS[3]={1.0f,0.5f,0.25f};
    double kln=0; int ok=0;
    for(int bi=0;bi<3;bi++){
        if(bi>0) for(int L=0;L<NLAYERS;L++) if(zidx[L]>=0)
            op_blend(&GS_LF[L].ops[zidx[L]],&zbaks[L],&fits[L],BETAS[bi]);
        float*lg=gs_forward_from(0,H0,ids,S,n_fit,Hc);
        kln=bwd_val_kl(ANC.logits,lg,vs,n_fit); free(lg);
        printf("GSWEEP 联合终验 β=%.2f val-KL=%.5f (基线%.5f)%s\n",BETAS[bi],kln,*kl0,kln<*kl0-1e-9?" ✓":"");
        fflush(stdout);
        if(kln<*kl0-1e-9){ ok=1; break; }
    }
    if(ok){
        int nc=0;
        for(int L=0;L<NLAYERS;L++) if(zidx[L]>=0){ zfile_commit(L,&GS_LF[L].ops[zidx[L]],(float)kln); nc++; }
        char rs[80]; snprintf(rs,80,"联合轮 最终KL %.5f→%.5f 提交%d层",*kl0,kln,nc);
        mlog(0,"向后·反修","z 联合批式重解(抽行探测+全程终验)","已改写层文件",0,rs,"✓正向落地");
        *kl0=kln;
    } else {
        for(int L=0;L<NLAYERS;L++) if(zidx[L]>=0) GS_LF[L].ops[zidx[L]]=zbaks[L];
        { float*lg=gs_forward_from(0,H0,ids,S,n_fit,Hc); free(lg); }   /* 终验刷脏了 Hc → 回退后恢复 */
        mlog(0,"向后·反修","z 联合批式重解","已最优保持(β全拒→回退)",0,"信赖域三级未降","保持");
    }
    free(gsx); free(Hg); free(Hb); free(idg); free(msg); free(kbg); free(fing);
    free(zbaks); free(fits); free(zidx);
    return ok;
}
/* 字节模式前向 L=J..Lend(用 GS_LW 骨干缓存 + GS_LF 层文件缓存), 返回出口 H;
 * 可选把每层入口隐藏写进 HQcache[J..Lend+1](刷新量化态)。逐层反修专用。*/
static float *gs_forward_exit(int J,int Lend,const float*Hin,const long*ids,int S,int n_fit,float*HQcache){
    size_t lstride=(size_t)S*HCM*DIM;
    float *H=malloc(lstride*4); memcpy(H,Hin,lstride*4);
    for(int L=J;L<=Lend;L++){
        if(HQcache) memcpy(HQcache+(size_t)L*lstride,H,lstride*4);   /* 层 L 入口 */
        double _x0=g_bflt_on?vqt_now():0.0; LW*_c32=GS_LW?lw32_get(L,&GS_LW[L]):NULL;   /* 驱逐层也先问缓存 */
        LW T; if(_c32) T=*_c32; else T=GS_LW[L].loaded?lwh_expand(&GS_LW[L]):load_layer(L);
        if(g_bflt_on) g_bflt[11]+=vqt_now()-_x0;   /* fp16 展开(lw32 命中≈0); 被驱逐层临时从 HF 重载 */
        layer_fwd(L,&T,H,ids,S,n_fit,1,'B',NULL);
        if(!_c32) free_layer(&T);
    }
    if(HQcache) memcpy(HQcache+(size_t)(Lend+1)*lstride,H,lstride*4); /* Lend 出口 */
    return H;
}
/* 每行前沿出口 L2 距离(per-token 反修目标用) */
static void bf_rowdist(const float*H,const float*Htgt,int S,size_t rowsz,double*d){
    double _t0=g_bflt_on?vqt_now():0.0;
    for(int s=0;s<S;s++){ double e=0; const float*a=H+(size_t)s*rowsz,*b=Htgt+(size_t)s*rowsz;
        for(size_t i=0;i<rowsz;i++){ double dd=(double)a[i]-b[i]; e+=dd*dd; } d[s]=e; }
    if(g_bflt_on) g_bflt[8]+=vqt_now()-_t0;
}
/* β 信赖域混合: dst 系数 = (1−β)·old + β·fitted(特征参数 μ/σ/V8 不动, 只混系数) */
static void op_blend(lop_t*dst,const lop_t*o,const lop_t*f,float b){
    if(dst->type==1) dst->g=(1.0f-b)*o->g+b*f->g;
    else if(dst->type==2){ dst->w2p[0]=(1.0f-b)*o->w2p[0]+b*f->w2p[0];
                           dst->w2p[1]=(1.0f-b)*o->w2p[1]+b*f->w2p[1]; }
    else if(dst->type==3) for(int i=0;i<9;i++) dst->w8[i]=(1.0f-b)*o->w8[i]+b*f->w8[i];
}
/* 三点抛物线顶点(细搜: 粗网格漏 ±1% 级最优 → 顶点插值补上)。凸才给顶点, 否则回退三点 argmin。*/
static double bf_vertex(double x1,double y1,double x0,double y0,double x2,double y2){
    double s1=(y1-y0)/(x1-x0), s2=(y2-y0)/(x2-x0);
    double a2=(s1-s2)/(x1-x2);                     /* 二次项系数 */
    if(a2>1e-18){
        double num=(x0-x1)*(x0-x1)*(y0-y2)-(x0-x2)*(x0-x2)*(y0-y1);
        double den=(x0-x1)*(y0-y2)-(x0-x2)*(y0-y1);
        if(fabs(den)>1e-18) return x0-0.5*num/den;
    }
    double bx=x0,by=y0; if(y1<by){bx=x1;by=y1;} if(y2<by){bx=x2;by=y2;}
    return bx;
}
/* ★逐层前进即反修前面所有层★(用户设计): 前沿层 Lfront 完成后, 对 J=Lfront-1..0 逐个以
 * 【本前沿层 Lfront 出口 vs FP 锚 ANC.H[Lfront]】为判据, 多形态递进搜正向(铁律: 无提升=形态不对, 换形态):
 *   A 全局α(链末 bf.GL)  B 向后TREF t 重调(前沿判据)  C per-token 动态(argmin α_s → bf.GLdyn2 范数特征)
 * 择优原地改写该层文件 → 刷新下游量化态 HQE → 用反修后累积态继续。判据与落地都走字节回放(与合并同口径)。*/
/* ★形态F(2026-07-13)★ 反修新建 GLdyn8 用的 PCA8 方向: 与层搜索 STK8 同法
 * (确定性种子随机初始化 → 投影/重构交替×4 → MGS 正交), 返回行主序 [c*DIM+d] fp32;
 * 行数不足 3×8 返回 NULL(9-dof 会饥饿)。 */
static float*bf_pca8(const float*Fin,int n){
    const int KP=8;
    if(n<3*KP) return NULL;
    float *Vt=malloc((size_t)DIM*KP*4);
    uint64_t sd=0xC0FFEE123ULL;
    for(size_t i=0;i<(size_t)DIM*KP;i++){ sd=sd*6364136223846793005ULL+1442695040888963407ULL;
        Vt[i]=((float)((sd>>40)&0xFFFFFF)/16777216.0f)-0.5f; }
    dq_mgs_qh(Vt,DIM,KP);
    float *pj=malloc((size_t)n*KP*4);
    for(int it=0;it<4;it++){
        for(int s=0;s<n;s++){ const float*x=Fin+(size_t)s*DIM; float*pr=pj+(size_t)s*KP;
            for(int c=0;c<KP;c++){ double a=0; for(int d=0;d<DIM;d++) a+=(double)x[d]*Vt[(size_t)d*KP+c]; pr[c]=(float)a; } }
        memset(Vt,0,(size_t)DIM*KP*4);
        for(int s=0;s<n;s++){ const float*x=Fin+(size_t)s*DIM; const float*pr=pj+(size_t)s*KP;
            for(int d=0;d<DIM;d++){ float xv=x[d]; if(xv==0)continue; float*vt=Vt+(size_t)d*KP;
                for(int c=0;c<KP;c++) vt[c]+=xv*pr[c]; } }
        dq_mgs_qh(Vt,DIM,KP);
    }
    float*V=malloc((size_t)KP*DIM*4);
    for(int c=0;c<KP;c++) for(int d=0;d<DIM;d++) V[(size_t)c*DIM+d]=Vt[(size_t)d*KP+c];
    free(Vt); free(pj);
    return V;
}
