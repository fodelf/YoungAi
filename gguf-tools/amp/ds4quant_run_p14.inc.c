/* ===== ②冷专家修复(DS4_REPAIR_COLD=1, 2026-07-24 按序执行第2项) =====
 * 判据: 旧锚(DS4_ANCHOR_OLD)fit行零命中 ∧ 新锚(DS4_ANCHOR, 当前语料)fit行≥1 的冷专家 —
 * 这些专家原字节=无数据rowscale兜底(纯权重先验)=胡言触发面; 其对层内op链的拟合贡献≈0(零流量)
 * → 用新锚校准行重解 signref(含 w2 顺序补偿) 原地 pwrite 回 dql, 风险最小收益直接。热=go2b侧车不动。 */
typedef struct { int L,S,n_fit,S_old,n_fit_old; int *e_next; int fd; size_t off0,szG,szD;
                 const int32_t*aold; const float*afin; const int32_t*aidx; long nrep; } rcw_t;
static void *repair_worker(void*a){
    rcw_t*w=a;
    float *Xc=malloc((size_t)(w->n_fit>0?w->n_fit:1)*DIM*4);
    for(;;){ int e=__sync_fetch_and_add(w->e_next,1); if(e>=NEXP)break;
        if(g2_hot_slot(w->L,e)>=0) continue;                 /* 热=go2b 侧车, dql 洞不动 */
        int hit_old=0;
        for(int s=0;s<w->n_fit_old&&!hit_old;s++)for(int a2=0;a2<NACT;a2++)
            if(w->aold[(size_t)s*NACT+a2]==e){ hit_old=1; break; }
        if(hit_old) continue;                                /* 旧锚已有校准 → 字节不动 */
        int ncal=0;
        for(int s=0;s<w->n_fit;s++){ int hit=0;
            for(int a2=0;a2<NACT;a2++) if(w->aidx[(size_t)s*NACT+a2]==e){hit=1;break;}
            if(hit){ memcpy(Xc+(size_t)ncal*DIM,w->afin+(size_t)s*DIM,(size_t)DIM*4); ncal++; } }
        if(!ncal) continue;                                  /* 新语料也没打中 → 仍无据可修 */
        char n1[160],n3[160],n2[160]; long rr,cc;
        snprintf(n1,sizeof(n1),"layers.%d.ffn.experts.%d.w1.weight",w->L,e);
        snprintf(n3,sizeof(n3),"layers.%d.ffn.experts.%d.w3.weight",w->L,e);
        snprintf(n2,sizeof(n2),"layers.%d.ffn.experts.%d.w2.weight",w->L,e);
        float *e1=st_read_weight(&C,n1,&rr,&cc),*e3=st_read_weight(&C,n3,&rr,&cc),*e2=st_read_weight(&C,n2,&rr,&cc);
        if(!e1||!e3||!e2){ if(e1)free(e1);if(e3)free(e3);if(e2)free(e2); continue; }
        uint8_t *dG=malloc(w->szG),*dU=malloc(w->szG),*dDn=malloc(w->szD);
        float *q1=malloc((size_t)MOEI*DIM*4),*q3=malloc((size_t)MOEI*DIM*4);
        dq_signref_export(e1,MOEI,DIM,Xc,ncal,dG,q1);
        dq_signref_export(e3,MOEI,DIM,Xc,ncal,dU,q3);
        float *hc=malloc((size_t)ncal*MOEI*4);
        { float *gg=malloc((size_t)ncal*MOEI*4),*uu=malloc((size_t)ncal*MOEI*4);
          dq_matmul(Xc,q1,gg,ncal,DIM,MOEI); dq_matmul(Xc,q3,uu,ncal,DIM,MOEI);
          for(size_t i=0;i<(size_t)ncal*MOEI;i++){ float g2v=gg[i],u2=uu[i];
              if(SWLIM>0){ if(u2>SWLIM)u2=SWLIM; if(u2<-SWLIM)u2=-SWLIM; if(g2v>SWLIM)g2v=SWLIM; }
              hc[i]=dq_silu(g2v)*u2; }
          free(gg);free(uu); }
        dq_signref_export(e2,DIM,MOEI,hc,ncal,dDn,NULL);
        if(pwrite(w->fd,dG,w->szG,(off_t)(w->off0+(size_t)e*w->szG))!=(ssize_t)w->szG) perror("rep-g");
        if(pwrite(w->fd,dU,w->szG,(off_t)(w->off0+(size_t)NEXP*w->szG+(size_t)e*w->szG))!=(ssize_t)w->szG) perror("rep-u");
        if(pwrite(w->fd,dDn,w->szD,(off_t)(w->off0+2*(size_t)NEXP*w->szG+(size_t)e*w->szD))!=(ssize_t)w->szD) perror("rep-d");
        __sync_fetch_and_add(&w->nrep,1);
        free(dG);free(dU);free(dDn);free(q1);free(q3);free(hc);free(e1);free(e3);free(e2);
    }
    free(Xc); return NULL;
}
static void repair_cold(int S,int n_fit){
    const char*oldp=getenv("DS4_ANCHOR_OLD"); const char*ld=getenv("DS4_LAYER_DIR");
    if(!oldp||!ld){ fprintf(stderr,"[repair] 需 DS4_ANCHOR_OLD + DS4_LAYER_DIR\n"); exit(2); }
    FILE*fo=fopen(oldp,"rb"); if(!fo){ perror(oldp); exit(2); }
    uint32_t hd[8]; uint64_t idh0;
    if(fread(hd,4,8,fo)!=8||fread(&idh0,8,1,fo)!=1){ fprintf(stderr,"[repair] 旧锚读失败\n"); exit(2); }
    if(hd[0]!=0x32415144u||hd[4]!=(uint32_t)NLAYERS||hd[6]!=(uint32_t)NACT||hd[3]!=(uint32_t)DIM){
        fprintf(stderr,"[repair] 旧锚头不符(NL/NACT/DIM)\n"); exit(2); }
    int S_old=(int)hd[1], n_fit_old=(S_old*3)/4;
    fseeko(fo,(off_t)(40+(uint64_t)NLAYERS*(uint64_t)S_old*DIM*4),SEEK_SET);
    int32_t*ridx_old=malloc((size_t)NLAYERS*S_old*NACT*4);
    if(fread(ridx_old,4,(size_t)NLAYERS*S_old*NACT,fo)!=(size_t)NLAYERS*S_old*NACT){
        fprintf(stderr,"[repair] 旧锚ridx读失败\n"); exit(2); }
    fclose(fo);
    size_t szG=(size_t)MOEI*go1b_blk_row_bytes(DIM), szD=(size_t)DIM*go1b_blk_row_bytes(MOEI);
    fprintf(stderr,"[repair] 冷专家修复: 旧S=%d(fit%d) 新S=%d(fit%d) signref μ=%.0f\n",
            S_old,n_fit_old,S,n_fit,dq_signref_mu);
    long tot=0;
    for(int L=0;L<NLAYERS;L++){
        char lp[512]; snprintf(lp,sizeof(lp),"%s/dql_L%02d.bin",ld,L);
        lfile_t lf;
        if(lfile_load(lp,&lf)!=0){ fprintf(stderr,"[repair] L%02d 层文件缺/坏 — 停\n",L); exit(3); }
        size_t off0=(size_t)(lf.w1-lf.map);
        lfile_free(&lf);
        int fd=open(lp,O_WRONLY); if(fd<0){ perror(lp); exit(3); }
        int e_next=0,nth=NTHREADS<1?1:(NTHREADS>NEXP?NEXP:NTHREADS);
        rcw_t*ws=calloc((size_t)nth,sizeof(rcw_t)); pthread_t*th=malloc((size_t)nth*sizeof(pthread_t));
        for(int t=0;t<nth;t++){ ws[t]=(rcw_t){L,S,n_fit,S_old,n_fit_old,&e_next,fd,off0,szG,szD,
            ridx_old+(size_t)L*S_old*NACT, ANC.fin+(size_t)L*S*DIM, ANC.ridx+(size_t)L*S*NACT, 0};
            pthread_create(&th[t],NULL,repair_worker,&ws[t]); }
        long nrep=0;
        for(int t=0;t<nth;t++){ pthread_join(th[t],NULL); nrep+=ws[t].nrep; }
        free(ws);free(th); close(fd);
        printf("REPAIR L=%02d 修复=%ld\n",L,nrep); fflush(stdout);
        fprintf(stderr,"[repair][mem] L%02d footprint=%.2fGB\n",L,mem_gb());
        tot+=nrep;
    }
    free(ridx_old);
    printf("REPAIR_COLD 总修复=%ld 专家 (判据: 旧锚0校准∧新锚≥1; 热go2b/已校准不动)\n",tot);
    fflush(stdout);
}
static void global_sweep(const long*ids,int S,int n_fit){
    int MAXS=getenv("DS4_GSWEEP")?atoi(getenv("DS4_GSWEEP")):3;
    if(MAXS<1) return;
    size_t lstride=(size_t)S*HCM*DIM;
    for(int L=0;L<NLAYERS;L++) GBL_G[L]=1.0f;
    /* ★效率: 存档只加载一次★ — 释放回扫不用的大 anchor(fin/H/ridx/rw, 保 logits) 腾内存,
     * 再把 43 层骨干权重 + 层文件 mmap 各缓存一次(禁千次重读/重mmap/重反量化) */
    /* ★ridx/rw 必须保活(2026-08-03 回扫 SIGSEGV 终修): 全局回扫的 B 前向仍走锚路由覆盖
     * (layer_fwd 2218 读 ANC.ridx/rw), 二者合计 ~5MB(mmap=干净页零成本, malloc=小)。
     * 只释放大块 fin/H(GB 级)。
     * ★joint 模式(默认)必须保 ANC.H(2026-08-04 probe1 段错误实锤): backfit_joint_round 的
     * 单层探测判据=层出口 vs ANC.H[L]; mmap 锚=干净页可逐出零 RSS 代价 → 不释放。
     * 旧逐层版(DS4_GS_PERCOL=1)只用 ANC.logits, 保持原释放。 */
    { int percol0=getenv("DS4_GS_PERCOL")?atoi(getenv("DS4_GS_PERCOL")):0;
      if(percol0){
          if(ANC_MMAP){ ANC.fin=NULL; ANC.H=NULL; }   /* mmap 页可逐出, 禁 free */
          if(ANC.fin){ free(ANC.fin); ANC.fin=NULL; }
          if(ANC.H){ free(ANC.H); ANC.H=NULL; }
      } else if(!ANC_MMAP&&ANC.fin){ free(ANC.fin); ANC.fin=NULL; }   /* joint: 只放 fin(不用), H 必留 */
    }
    if(!GS_LW){   /* fwd_all 逐层反修未开(DS4_BACKFIT_INCR=0)时才现建 */
        fprintf(stderr,"[全局回扫] 缓存 43 层权重+文件(只一次)...\n");
        GS_LW=calloc((size_t)NLAYERS,sizeof(LWH));
        GS_LF=calloc((size_t)NLAYERS,sizeof(lfile_t));
        for(int L=0;L<NLAYERS;L++){
            { LW W=load_layer2(L,&GS_LW[L]); lwh_absorb(&GS_LW[L],&W); free_layer(&W); }
            char lp[512]; snprintf(lp,sizeof(lp),"%s/dql_L%02d.bin",getenv("DS4_LAYER_DIR"),L);
            if(lfile_load(lp,&GS_LF[L])!=0) fprintf(stderr,"[回扫] L%02d 层文件缓存失败\n",L);
            if((L&7)==0) fprintf(stderr,"  缓存 %d/%d\n",L,NLAYERS);
        }
    } else fprintf(stderr,"[全局回扫] 复用 fwd_all 逐层反修已建缓存(权重/文件在位, 免重读)\n");
    GS_HCFN=st_read_weight(&C,"hc_head_fn",NULL,NULL); GS_HCB=st_read_weight(&C,"hc_head_base",NULL,NULL);
    GS_HCS=st_read_weight(&C,"hc_head_scale",NULL,NULL); GS_NORM=st_read_weight(&C,"norm.weight",NULL,NULL);
    GS_HW=st_read_weight(&C,"head.weight",NULL,NULL);
    fprintf(stderr,"[全局回扫] 权重/文件/head 缓存完成 — 之后只算不读盘\n");
    float *emb=st_read_weight(&C,"embed.weight",NULL,NULL);
    float *H0=malloc(lstride*4);
    for(int s=0;s<S;s++)for(int j=0;j<HCM;j++)memcpy(H0+((size_t)s*HCM+j)*DIM,emb+(size_t)ids[s]*DIM,(size_t)DIM*4);
    free(emb);
    float *Hc=malloc((size_t)(NLAYERS+1)*lstride*4);
    int vs=(n_fit*3)/4;   /* 判据行 = fit 尾部 val, held 只观测 */
    fprintf(stderr,"\n[全局回扫] %d 轮 — 逐前层 z 系数以【最终输出 KL】重解并原地改写层文件(真·反修, 全局联合最优)\n",MAXS);
    float *lg=gs_forward_from(0,H0,ids,S,n_fit,Hc);
    double kl0=bwd_val_kl(ANC.logits,lg,vs,n_fit); free(lg);
    double kl_start=kl0;
    printf("GSWEEP 起点 最终val-KL=%.5f\n",kl0); fflush(stdout);
    /* ★轮间收敛判停(2026-08-04 用户裁决: 遍数可压)★: 布尔 improved 只挡"整轮零改写",
     * 数值尾巴轮(几层各降 1e-5)仍会把 MAXS 轮跑满 → 按轮改善量占起点比例判停,
     * 低于 DS4_GS_CONV_PCT(默认 0.1%)即数值收敛, 后续轮不跑。 */
    double conv_pct=getenv("DS4_GS_CONV_PCT")?atof(getenv("DS4_GS_CONV_PCT")):0.1;
    int percol=getenv("DS4_GS_PERCOL")?atoi(getenv("DS4_GS_PERCOL")):0;   /* 1=旧逐层坐标下降(20-30h, 对照口径) */
    for(int sw=1;sw<=MAXS;sw++){
        int improved=0; double kl_round0=kl0;
        if(!percol){
            fprintf(stderr,"[反修] 第%d轮 联合批式(抽行探测+全程终验)...\n",sw);
            improved=backfit_joint_round(ids,S,n_fit,H0,Hc,lstride,&kl0);
        } else for(int L=0;L<NLAYERS;L++){
            double before=kl0;
            fprintf(stderr,"[反修] 第%d轮 L%02d z重解(最终KL判优)... ",sw,L);
            int chg=backfit_layer_z(L,ids,S,n_fit,Hc,lstride,&kl0);
            if(chg){ improved=1;
                printf("GSWEEP sw%d L=%d z重解 最终val-KL %.5f→%.5f ✓改写层文件\n",sw,L,before,kl0); }
            else    printf("GSWEEP sw%d L=%d z保持(未降/无z) 最终val-KL=%.5f\n",sw,L,kl0);
            fflush(stdout);
        }
        double rimp=kl_start>1e-9?100.0*(kl_round0-kl0)/kl_start:0.0;
        printf("GSWEEP 第%d轮完 最终val-KL=%.5f 本轮改善=%.4f%% %s\n",sw,kl0,rimp,improved?"有改善":"收敛");
        fflush(stdout);
        if(!improved) break;
        if(rimp<conv_pct){ printf("GSWEEP 第%d轮改善%.4f%%<%.2f%%(DS4_GS_CONV_PCT) → 数值收敛提前止\n",sw,rimp,conv_pct);
            fflush(stdout); break; }
    }
    printf("GSWEEP_DONE 最终val-KL %.5f→%.5f 提升%.1f%%\n",
           kl_start,kl0,kl_start>1e-9?100.0*(kl_start-kl0)/kl_start:0.0);
    fflush(stdout);
    free(H0);free(Hc);
    for(int L=0;L<NLAYERS;L++){ lwh_free(&GS_LW[L]); lfile_free(&GS_LF[L]); }
    free(GS_LW); GS_LW=NULL; free(GS_LF); GS_LF=NULL;
    free(GS_HCFN);free(GS_HCB);free(GS_HCS);free(GS_NORM);free(GS_HW);
    GS_HCFN=GS_HCB=GS_HCS=GS_NORM=GS_HW=NULL;
    if(GS_FIN){ free(GS_FIN); GS_FIN=NULL; }
}
int main(int argc,char**argv){
    /* ★进程级分配器/BLAS 设置写进代码, 不走 env(2026-08-22 铁律: 本项目不得新增环境变量)★
     * 依据(实测, 全过程记在 fable5.md):
     *  · 专家循环每专家 malloc/free 三块 33.5MB。>128KB 默认走 mmap ⇒ 每次全新零页 + 逐 4KB 首触缺页;
     *    而 TRIM/MMAP 阈只管主 arena, **非主 arena 的堆增缩仍走 mprotect** —— 20 线程抢进程 mmap
     *    写锁, wchan 实锤 51/110 卡在 vm_mmap_pgoff/do_mprotect_pkey/__vm_munmap。
     *    放开 top_pad 后 mmap 风暴消失: S=256 从 164s→65s, 磁盘 118 MB/s→1939 MB/s。
     *  · attention 在专家 pthread 循环之外, 是单线程段; BLAS 只给 1 线程会把它锁死在 1 核。
     *    专家循环内部已有 20 路 pthread, 那段靠 GPU/单线程 BLAS, 不受此影响。 */
#ifdef __linux__
    mallopt(M_TOP_PAD,         256 * 1024 * 1024);
    mallopt(M_MMAP_THRESHOLD, 1024 * 1024 * 1024);
    mallopt(M_TRIM_THRESHOLD, 1024 * 1024 * 1024);
#endif
#ifdef DS4QUANT_OPENBLAS
    { void scipy_openblas_set_num_threads(int); long nc = sysconf(_SC_NPROCESSORS_ONLN);
      scipy_openblas_set_num_threads((int)(nc > 1 ? nc : 1)); }
#endif
    for(int i=0;i<NL;i++) GBL_G[i]=1.0f;   /* ★static 默认0=乘0清routed: 反修中途 'B' 回放被它抹平(实锤bug), 必须先置1★ */
    nact_rt_init(getenv("DS4_LAYER_DIR"));   /* 动态路由反修 Phase-B: 层目录 rroute.txt 门控 */
    const char*hf=getenv("DS4_HF");if(!hf)hf="/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base"; st_open(&C,hf);
    const char*idf=argc>1?argv[1]:"/tmp/rr_hard.ids"; int ntok=argc>2?atoi(argv[2]):64;
    if(getenv("DS4_Z_RANK")) ZRANK=atoi(getenv("DS4_Z_RANK"));
    if(getenv("DS4_Z_LAMBDA")) ZLAMBDA=atof(getenv("DS4_Z_LAMBDA"));
    if(getenv("DS4_NL")){ NLAYERS=atoi(getenv("DS4_NL")); if(NLAYERS<1)NLAYERS=1; if(NLAYERS>NL)NLAYERS=NL; }
    if(getenv("DS4_ZCHAIN_ONLY")){
        /* 独立模式: 从既有 dql 层文件重建 zchain_all.bin + 43 份 opt_LXX.bin, 不跑量化。
         * merge-only 恢复场景专用(量化被截停时 zchain_write 没到点) — 修"merge 出的
         * GGUF 缺优化张量"缺口: 脚本在建骨架前先跑本模式补齐侧车。 */
        if(!getenv("DS4_LAYER_DIR")||!getenv("DS4_ZCHAIN")){
            fprintf(stderr,"[zchain-only] 需 DS4_LAYER_DIR + DS4_ZCHAIN\n"); return 1; }
        zchain_write();
        return 0;
    }
    /* ★merge 独立早出口(在一切量化/锚逻辑之前): 只读层文件+偏移表, 秒级; 旧位置在量化遍之后=会重跑全量化(bug已修) */
    if(getenv("DS4_MERGE_GGUF")){
        /* ===== 合并: 每层 tuned 1bit 字节 → go1b 骨架 GGUF 专家槽(pwrite 偏移) =====
         * 需: DS4_MERGE_GGUF=骨架gguf(deepseek4-quantize --experts go1b 产, type40)
         *     DS4_MERGE_OFF=偏移表(gguf_offsets.py) DS4_LAYER_DIR=层文件目录 */
        const char*gp=getenv("DS4_MERGE_GGUF"),*op=getenv("DS4_MERGE_OFF"),*ld=getenv("DS4_LAYER_DIR");
        if(!op||!ld){ fprintf(stderr,"[merge] 需 DS4_MERGE_OFF + DS4_LAYER_DIR\n"); return 1; }
        /* 偏移表: name → (off,type,nel) */
        typedef struct { int L; char kind[8]; long off,nel; int ty; } mo_t;
        mo_t *MO=calloc((size_t)NL*3,sizeof(mo_t)); int nmo=0;
        FILE*f=fopen(op,"r"); if(!f){ perror("off"); return 1; }
        { char nm[256]; int ty; long off,nel;
          while(fscanf(f,"%255s %d %ld %ld",nm,&ty,&off,&nel)==4){
            int Lx; char kind[16]; int nfx=0;
            /* %n 全串守卫: ffn_gate_tid2eid 等名字也能让 %d+%[^_] 转换成功(返回2), 无守卫时伪匹配
             * 偷占 NL*3 容量槽, 把文件序靠后的真 exps(blk.42)挤出 → "偏移缺失"。nfx>0 且到串尾才算数。 */
            if(sscanf(nm,"blk.%d.ffn_%15[^_]_exps.weight%n",&Lx,kind,&nfx)==2&&nfx>0&&nm[nfx]==0&&Lx<NL&&nmo<NL*3){
                MO[nmo].L=Lx; snprintf(MO[nmo].kind,8,"%s",kind);
                MO[nmo].off=off; MO[nmo].nel=nel; MO[nmo].ty=ty; nmo++; } } }
        fclose(f);
        fprintf(stderr,"[merge] 偏移表专家 tensor %d 个(期望 %d)\n",nmo,NLAYERS*3);
        FILE*gf=fopen(gp,"r+b"); if(!gf){ perror("gguf骨架"); return 1; }
        size_t szG=(size_t)MOEI*go1b_blk_row_bytes(DIM), szD=(size_t)DIM*go1b_blk_row_bytes(MOEI);
        long done=0,skipped=0;
        for(int L=0;L<NLAYERS;L++){
            char lp[512]; snprintf(lp,sizeof(lp),"%s/dql_L%02d.bin",ld,L);
            int fd=open(lp,O_RDONLY);
            if(fd<0){
                /* 续并(DS4_MERGE_RESUME, 脚本核实 GGUF 在位后才给): consume 设计 = 注入成功才 unlink,
                 * 缺失 ⇒ 该层字节已在 GGUF ⇒ 跳过。非续并保持硬停 — 缺层=半成品, 不许静默出洞。 */
                if(getenv("DS4_MERGE_RESUME")){ skipped++;
                    fprintf(stderr,"[merge] L%02d 层文件缺席 → 已消费(此前注入), 跳过\n",L); continue; }
                fprintf(stderr,"[merge] 缺层文件 %s — 停\n",lp); return 1;
            }
            struct stat st2; fstat(fd,&st2); size_t msz=(size_t)st2.st_size;
            uint8_t*mp=mmap(NULL,msz,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
            if(mp==MAP_FAILED){ fprintf(stderr,"[merge] mmap %s 失败\n",lp); return 1; }
            /* 定位 1bit 记录 payload */
            const uint8_t*p=mp+12,*end=mp+msz,*w1=NULL,*w3=NULL,*w2=NULL;
            uint32_t nrec; memcpy(&nrec,mp+8,4);
            for(uint32_t i=0;i<nrec&&p+116<=end;i++){
                char nm[17]; memcpy(nm,p,16); nm[16]=0;
                uint64_t psz; memcpy(&psz,p+88,8); const uint8_t*pay=p+116; p=pay+psz;
                if(!strcmp(nm,"1bit")&&psz>=(uint64_t)NEXP*(2*szG+szD)){
                    w1=pay; w3=pay+(size_t)NEXP*szG; w2=pay+2*(size_t)NEXP*szG; break; }
            }
            if(!w1){ fprintf(stderr,"[merge] L%02d 无 1bit 记录\n",L); munmap(mp,msz); return 1; }
            mo_t *tg=NULL,*tu=NULL,*td=NULL;
            for(int k=0;k<nmo;k++) if(MO[k].L==L){
                if(!strcmp(MO[k].kind,"gate"))tg=&MO[k];
                else if(!strcmp(MO[k].kind,"up"))tu=&MO[k];
                else if(!strcmp(MO[k].kind,"down"))td=&MO[k]; }
            if(!tg||!tu||!td){ fprintf(stderr,"[merge] L%02d 偏移缺失\n",L); munmap(mp,msz); return 1; }
            if(tg->ty!=40||tu->ty!=40||td->ty!=40){ fprintf(stderr,"[merge] L%02d 骨架非go1b(40): %d/%d/%d — 拒写(需 --experts go1b 骨架)\n",L,tg->ty,tu->ty,td->ty); munmap(mp,msz); return 1; }
            int ok=1;
            ok&=fseeko(gf,tg->off,SEEK_SET)==0&&fwrite(w1,1,(size_t)NEXP*szG,gf)==(size_t)NEXP*szG;
            ok&=fseeko(gf,tu->off,SEEK_SET)==0&&fwrite(w3,1,(size_t)NEXP*szG,gf)==(size_t)NEXP*szG;
            ok&=fseeko(gf,td->off,SEEK_SET)==0&&fwrite(w2,1,(size_t)NEXP*szD,gf)==(size_t)NEXP*szD;
            munmap(mp,msz);
            if(!ok){ fprintf(stderr,"[merge] L%02d 写失败\n",L); return 1; }
            done++;
            fprintf(stderr,"[merge] L%02d 注入 (%.0f MiB)\n",L,(2.0*NEXP*szG+NEXP*szD)/1048576.0);
            if(getenv("DS4_MERGE_CONSUME")){ unlink(lp);   /* 边并边释放: 稀疏骨架回填+层文件删除, 峰值盘占恒定(挤盘刚需) */
                fprintf(stderr,"[merge] L%02d 已消费(层文件释放)\n",L); }
        }
        fflush(gf); fclose(gf);
        if(skipped) printf("MERGE_GGUF %s 注入层=%ld 已消费跳过=%ld 合计=%ld/%d 完成\n",gp,done,skipped,done+skipped,NLAYERS);
        else        printf("MERGE_GGUF %s 注入层=%ld/%d 完成\n",gp,done,NLAYERS);
        fflush(stdout);
        return 0;
    }
    if(getenv("DS4_ZK")) ZK=atoi(getenv("DS4_ZK"));
    if(getenv("DS4_ABLATE")) ABLATE=atoi(getenv("DS4_ABLATE"));
    if(getenv("DS4_LOCAL_Q")) LOCALQ=atoi(getenv("DS4_LOCAL_Q"));
    if(getenv("DS4_LZ")) LZRANK=atoi(getenv("DS4_LZ"));
    if(getenv("DS4_LZ_LAMBDA")) LZLAMBDA=atof(getenv("DS4_LZ_LAMBDA"));
    if(getenv("DS4_LZ_TR")) LZTR=atof(getenv("DS4_LZ_TR"));
    TUNE_T0=time(NULL);
    if(getenv("DS4_TUNE_MIN")) TUNE_MIN=atof(getenv("DS4_TUNE_MIN"));
    if(getenv("DS4_FAST")) FAST=atoi(getenv("DS4_FAST"));
    if(getenv("DS4_BACKFIT_INCR")) BACKFIT_INCR=atoi(getenv("DS4_BACKFIT_INCR"));
    if(getenv("DS4_BF_MEMGB")) BF_MEMGB=atof(getenv("DS4_BF_MEMGB"));
    if(getenv("DS4_COADAPT")){ COADAPT=atoi(getenv("DS4_COADAPT")); if(COADAPT<0)COADAPT=0;
        if(COADAPT>0&&LZRANK==0){ LZRANK=16; fprintf(stderr,"[共适应] DS4_LZ 未设 → z rank≤16\n"); } }
    if(getenv("DS4_GO2B_HOT")&&atoi(getenv("DS4_GO2B_HOT"))){
        /* 热专家 go2b 合并态量化(残差+量化一体, 消漂移): 热表加载失败=硬拒(禁静默退 go1b) */
        const char*hp=getenv("DS4_GO2B_HOT_TABLE");
        const char*cands[3]={hp,"../data/corpus/prog_active_top64.txt","gguf-tools/data/corpus/prog_active_top64.txt"};
        int okh=-1;
        for(int i=0;i<3;i++){ if(!cands[i])continue; if(go2b_hot_load(cands[i],NLAYERS)==0){ okh=i; break; } }
        if(okh<0){ fprintf(stderr,"[go2b] ★热表读失败(DS4_GO2B_HOT_TABLE/默认两处) — 硬拒★\n"); exit(8); }
        GO2B_HOT=1;
        int tot=0; for(int L2=0;L2<NLAYERS&&L2<64;L2++) tot+=G2_K[L2];
        fprintf(stderr,"[go2b] 热表 %s: Σ热=%d (热=合并2bit GPTQ+act2 激活最优; 冷=go1b; 反修/回放在合并态)\n",
                cands[okh],tot);
    }
    { long nc=sysconf(_SC_NPROCESSORS_ONLN); NTHREADS=(int)(nc>8?8:(nc<1?1:nc));
      if(getenv("DS4_THREADS")) NTHREADS=atoi(getenv("DS4_THREADS")); if(NTHREADS<1)NTHREADS=1; }
    /* ★2048帽拆除(2026-08-10 反修v4: 帽导致>2048语料全部静默截断, 锚头S=2048 与读取端
     * NTOK 错位 = "末层损坏"幻象/SCORE B 假口径/链锚案 三案同源)★ */
    int idcap=ntok>2048?ntok:2048;
    long *ids=malloc((size_t)idcap*sizeof(long));int S=0;FILE*f=fopen(idf,"r");
    if(!f){ fprintf(stderr,"ids 文件 %s 打不开\n",idf); return 1; }
    { char ln[64]; while(S<ntok&&fgets(ln,sizeof(ln),f))ids[S++]=atol(ln); fclose(f); }
    if(S<3){ fprintf(stderr,"token 太少 (S=%d)\n",S); return 1; }
    int n_fit=(int)(S*3/4);
    if(getenv("DS4_NFIT")) n_fit=atoi(getenv("DS4_NFIT"));
    if(n_fit<1)n_fit=1; if(n_fit>=S)n_fit=S>1?S-1:1;   /* 3/4 fit(校准), 1/4 held-out(判决) */
    /* 逐层档位: DS4_LCFG(NL字符或1字符广播) > DS4_INJECT > DS4_SPARE > 全'1' */
    for(int i=0;i<NLAYERS;i++)LCFG[i]='1'; LCFG[NLAYERS]=0;
    const char*lc=getenv("DS4_LCFG");
    if(lc){ size_t n=strlen(lc);
        if(n==1) for(int i=0;i<NLAYERS;i++)LCFG[i]=lc[0];
        else if(n>=(size_t)NLAYERS) memcpy(LCFG,lc,(size_t)NLAYERS);
        else { fprintf(stderr,"DS4_LCFG 长度 %zu < NL=%d (可 1 字符广播)\n",n,NLAYERS); return 1; } }
    else if(getenv("DS4_INJECT")){ int L=atoi(getenv("DS4_INJECT"));
        char q=getenv("DS4_QCHAR")?getenv("DS4_QCHAR")[0]:'1';
        for(int i=0;i<NLAYERS;i++)LCFG[i]='F'; if(L>=0&&L<NLAYERS)LCFG[L]=q; }
    else if(getenv("DS4_SPARE")){ int L=atoi(getenv("DS4_SPARE"));
        char q=getenv("DS4_QCHAR")?getenv("DS4_QCHAR")[0]:'1';
        for(int i=0;i<NLAYERS;i++)LCFG[i]=q; if(L>=0&&L<NLAYERS)LCFG[L]='F'; }
    for(int i=0;i<NLAYERS;i++) if(!strchr("Fn1z23rgmB",LCFG[i])){ fprintf(stderr,"LCFG[%d]='%c' 非法(F/n/1/r/g/z/2/3/m/B)\n",i,LCFG[i]); return 1; }
    if(TUNE_MIN>0) fprintf(stderr,"[预算] %.1f 分仅作选档/预估 — 收敛绝对优先: 每层(本层+累积)整轮零接管才进下一层, 不截断\n",TUNE_MIN);
    fprintf(stderr,"ntok=%d n_fit=%d held=%d NL=%d 线程=%d ZK=%d\nLCFG=%s\n",S,n_fit,S-n_fit,NLAYERS,NTHREADS,ZK,LCFG);
    if(getenv("DS4_GENPROBE")){
        /* ★生成侧验证器(2026-07-29 用户"先验证, 不要浪费时间"): 回放态自回归贪心 N token —
         * 教师闭环之外的第一个生成证据(词汤/复读直接检测)。prompt=ids 文件全体;
         * 层来源: DS4_LAYER_DIR 有 dql 的层='B' 字节回放(g10h 原生; 内嵌 g2hot 对 bytes_moe
         * 透明), 其余='F' FP。路由现算(锚不载, ANC_OK=0, 生成序列≠锚序列)。 */
        int N=atoi(getenv("DS4_GENPROBE")); if(N<1)N=24;
        int P=S;
        long *gg=malloc((size_t)(P+N)*sizeof(long)); memcpy(gg,ids,(size_t)P*sizeof(long));
        const char*ld2=getenv("DS4_LAYER_DIR");
        char lc2[64]; for(int i=0;i<NLAYERS;i++)lc2[i]='F'; lc2[NLAYERS]=0;
        if(ld2){ if(!GS_LF)GS_LF=calloc((size_t)NLAYERS,sizeof(lfile_t));
            for(int Lx=0;Lx<NLAYERS;Lx++){ char lp[512];
                snprintf(lp,sizeof(lp),"%s/dql_L%02d.bin",ld2,Lx);
                if(!GS_LF[Lx].map && lfile_load(lp,&GS_LF[Lx])==0) lc2[Lx]='B';
                else if(GS_LF[Lx].map) lc2[Lx]='B'; } }
        int nb=0; for(int i=0;i<NLAYERS;i++) if(lc2[i]=='B')nb++;
        fprintf(stderr,"[genprobe] prompt=%d tok 生成=%d 回放层=%d/%d(其余FP) LCFG=%s\n",P,N,nb,NLAYERS,lc2);
        for(int t=0;t<N;t++){
            int S3=P+t;
            float *lg=fwd_all(gg,S3,S3,1,lc2);
            const float*row=lg+(size_t)(S3-1)*VOCAB;
            int bi=0; float bv=row[0];
            for(int v=1;v<VOCAB;v++) if(row[v]>bv){bv=row[v];bi=v;}
            free(lg);
            gg[P+t]=bi;
            fprintf(stderr,"[genprobe] t=%d/%d id=%d\n",t+1,N,bi);
            printf("GEN t=%d id=%d\n",t,bi); fflush(stdout);
        }
        printf("GENIDS"); for(int t=0;t<N;t++) printf(" %ld",gg[P+t]); printf("\n"); fflush(stdout);
        return 0;
    }
    /* FP 锚定: 有缓存直接用, 无则跑一次并落盘 */
    uint64_t idh=dq_ids_hash(ids,S);
    if(anchor_load(S,idh)){ ANC_OK=1; fprintf(stderr,"[anchor] 命中缓存 %s (FP 遍跳过)\n",anchor_path()); }
    else{
        fprintf(stderr,"[anchor] 无缓存/不匹配 → 跑 FP 锚定遍(一次性)...\n");
        anchor_alloc(S); ANC.idh=idh; ANC_BUILD=1;
        ANC.logits=fwd_all(ids,S,n_fit,0,NULL);
        ANC_BUILD=0; ANC_OK=1;
        if(anchor_save()) fprintf(stderr,"[anchor] 已写 %s\n",anchor_path());
    }
    if(getenv("DS4_FP_ONLY")){ fprintf(stderr,"[anchor] DS4_FP_ONLY=1, 到此为止\n"); return 0; }
    if(getenv("DS4_BBQ4_AB")){
        /* backbone q4 真 A/B(2026-07-28): 锚=FP 参照, 同进程武装 q4 重跑完整前向(含 attention
         * 从 st 重读) → 同口径 verdict。旧 env-only 钩子在锚缓存模式测不到 backbone(学生回放
         * 不重读 backbone 权重), 本模式是唯一诚实口径。 */
        fprintf(stderr,"[bbq4-ab] 武装 q4 重跑完整前向...\n");
        g_bbq4=1;
        float *lq=fwd_all(ids,S,n_fit,0,NULL);
        g_bbq4=0;
        char lc2[64]; for(int i2=0;i2<NLAYERS;i2++)lc2[i2]='F'; lc2[NLAYERS]=0;
        verdict(ANC.logits,lq,ids,S,n_fit,lc2);
        return 0;
    }
    if(getenv("DS4_SIGNREF_MU")) dq_signref_mu=atof(getenv("DS4_SIGNREF_MU"));
    if(getenv("DS4_SIGNREF_ROUNDS")) dq_signref_rounds=atoi(getenv("DS4_SIGNREF_ROUNDS"));
    if(getenv("DS4_MINVOL")&&getenv("DS4_RR_IDS")) rr_anchor_init();   /* ★统一标准: rr 判决锚(载/建一次) */
    if(getenv("DS4_REPAIR_COLD")){ repair_cold(S,n_fit); return 0; }   /* ②冷专家修复(新锚=当前语料锚) */
    if(getenv("DS4_EXPORT_GGUF")){ export_gguf(S,n_fit); return 0; }
    if(getenv("DS4_TUNE")){
        char plan[NL+1];
        fprintf(stderr,"渐进调优模式 (plan=%s ckpt=%s)...\n",plan_path(),ckpt_dir());
        float *lq=fwd_all_tune(ids,S,n_fit,plan);
        verdict(ANC.logits,lq,ids,S,n_fit,plan);
        zfile_write();   /* 合并动态侧车(逐层搜索胜者)落盘 */
        zchain_write();  /* DQZ2 全链运行时侧车(层文件被 consume 前的唯一留存) */
        return 0;
    }
    fprintf(stderr,"量化遍 (LCFG=%s)...\n",LCFG);
    size_t g_chtot=0;
    { const char*chp=getenv("DS4_CHAIN_ANCHOR");
      if(chp){
        g_chfd=open(chp,O_RDWR|O_CREAT|O_TRUNC,0644);
        if(g_chfd<0){ perror("chain-anchor"); exit(1); }
        g_chS=S;
        g_chtot=40+(size_t)NLAYERS*S*DIM*4+2*(size_t)NLAYERS*S*NACT*4
               +(size_t)NLAYERS*S*HCM*DIM*4+(size_t)S*VOCAB*4;
        if(ftruncate(g_chfd,(off_t)g_chtot)!=0){ perror("chain-trunc"); exit(1); }
        uint32_t hd9[8]={0x32415144u,(uint32_t)S,(uint32_t)HCM,(uint32_t)DIM,(uint32_t)NLAYERS,(uint32_t)VOCAB,(uint32_t)NACT,0};
        uint64_t idh9=dq_ids_hash(ids,S);
        pwrite(g_chfd,hd9,32,0); pwrite(g_chfd,&idh9,8,32);
        fprintf(stderr,"[链态锚] 直写 %s (%.2f GiB)\n",chp,g_chtot/1073741824.0);
      } }
    float *lq=fwd_all(ids,S,n_fit,1,LCFG);
    if(g_chfd>=0){
        pwrite(g_chfd,lq,(size_t)S*VOCAB*4,(off_t)(g_chtot-(size_t)S*VOCAB*4));
        close(g_chfd); g_chfd=-1;
        fprintf(stderr,"[链态锚] 完成\n");
    }
    if(getenv("DS4_DUMP_LOGITS")){   /* 逐位置量化 logits 落盘(瑕疵归因: FP=锚内 logits, 量化=此) */
        FILE*df=fopen(getenv("DS4_DUMP_LOGITS"),"wb");
        if(df){ int hd2[2]={S,(int)VOCAB}; fwrite(hd2,4,2,df);
                fwrite(lq,4,(size_t)S*VOCAB,df); fclose(df);
                fprintf(stderr,"[dump] 量化 logits → %s (S=%d)\n",getenv("DS4_DUMP_LOGITS"),S); } }
    verdict(ANC.logits,lq,ids,S,n_fit,LCFG);
    rb_save();       /* ★BF_ONLY 评测遍也落盘 Δb(2026-08-13 路由归因: FIT 寄生回放前向, 收官必须存) */
    zfile_write();   /* 合并动态侧车(逐层搜索胜者)落盘 */
    zchain_write();  /* DQZ2 全链运行时侧车(层文件被 consume 前的唯一留存) */
    if(getenv("DS4_LAYER_DIR")&&getenv("DS4_GSWEEP")){
        global_sweep(ids,S,n_fit);   /* ★全局联合回扫: 最终输出判据, 反修所有层★ */
        zchain_write();              /* 回扫原地改写了层文件 op 载荷 → 刷新全链侧车 */
    }
    if(getenv("DS4_EXPORT_LAYER")){
        int L=atoi(getenv("DS4_EXPORT_LAYER"));
        const char*lf=getenv("DS4_LAYER_FILE"); if(!lf)lf="/tmp/dql_L00.bin";
        export_layer_file(L,S,n_fit,lf);
    }
    return 0;
}
