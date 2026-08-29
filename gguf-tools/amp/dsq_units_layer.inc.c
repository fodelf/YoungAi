/* dsq_units_layer.inc.c — 三段单元的【真模型一层】驱动(物理分片, 只被 dsq_units.c include)。
 * 拆出来的唯一原因是 500 行守卫; 逻辑上它就是 dsq_units.c 的真层模式, 不是另一份实现。
 * 必须在 vq_qc.h / ds4_z.c / dsq_units_sweep.inc.c 之后 include。 */
static int real_layer(const char *hf,const char *anc,int L,int NE,int SROW,int VDIM,int VNC)
{
    /* ── 读锚头 ── */
    FILE *f=fopen(anc,"rb"); if(!f){ printf("锚打不开: %s\n",anc); return 1; }
    uint32_t hd[8]; uint64_t idh;
    if(fread(hd,4,8,f)!=8||fread(&idh,8,1,f)!=1){ printf("锚头读失败\n"); fclose(f); return 1; }
    if(hd[0]!=0x32415144u){ printf("锚 magic 不对 0x%08X\n",hd[0]); fclose(f); return 1; }
    const int S=(int)hd[1], ancHCM=(int)hd[2], D=(int)hd[3], NL=(int)hd[4], NACT=(int)hd[6];
    const int M=2048;                              /* MOEI, 与 p1 一致 */
    if(L<0||L>=NL){ printf("层号越界 %d/%d\n",L,NL); fclose(f); return 1; }
    if(SROW<=0||SROW>S) SROW=S;
    printf("锚: S=%d HCM=%d DIM=%d NLAYERS=%d NACT=%d → 取 L%d 前 %d 行, %d 个专家\n",
           S,ancHCM,D,NL,NACT,L,SROW,NE);
    const size_t fin_b=(size_t)NL*S*D*4, ridx_b=(size_t)NL*S*NACT*4;
    float *X=malloc((size_t)SROW*D*4);
    int32_t *RI=malloc((size_t)SROW*NACT*4); float *RW=malloc((size_t)SROW*NACT*4);
    int rc = fseek(f,40+(long)((size_t)L*S*D*4),SEEK_SET)==0 &&
             fread(X,4,(size_t)SROW*D,f)==(size_t)SROW*D;
    rc &= fseek(f,40+(long)fin_b+(long)((size_t)L*S*NACT*4),SEEK_SET)==0 &&
          fread(RI,4,(size_t)SROW*NACT,f)==(size_t)SROW*NACT;
    rc &= fseek(f,40+(long)fin_b+(long)ridx_b+(long)((size_t)L*S*NACT*4),SEEK_SET)==0 &&
          fread(RW,4,(size_t)SROW*NACT,f)==(size_t)SROW*NACT;
    fclose(f);
    if(!rc){ printf("锚读失败(文件短?)\n"); return 1; }

    st_ctx C; memset(&C,0,sizeof C); st_open(&C,hf);
    float *YQ=calloc((size_t)SROW*D,4), *YF=calloc((size_t)SROW*D,4);
    double relh_sum=0; size_t bytes=0; int nq=0;
    double t_rd=0, t_q=0, t_hc=0, t_fwd=0;   /* 分段: 读盘 / 纯量化 / 中间态 / 前向 */
    int *tok=malloc((size_t)SROW*4); float *ww=malloc((size_t)SROW*4);
    float *xs=malloc((size_t)SROW*D*4), *Wq=malloc((size_t)M*D*4);

    /* ── ①量化 ── */
    const double t0=vqt_now();
    for(int e=0;e<NE;e++){
        char n1[192],n3[192],n2[192]; long r,c;
        snprintf(n1,sizeof n1,"layers.%d.ffn.experts.%d.w1.weight",L,e);
        snprintf(n3,sizeof n3,"layers.%d.ffn.experts.%d.w3.weight",L,e);
        snprintf(n2,sizeof n2,"layers.%d.ffn.experts.%d.w2.weight",L,e);
        const double tr0=vqt_now();
        float *e1=st_read_weight(&C,n1,&r,&c), *e3=st_read_weight(&C,n3,&r,&c),
              *e2=st_read_weight(&C,n2,&r,&c);
        t_rd += vqt_now()-tr0;
        if(!e1||!e3||!e2){ printf("专家 %d 权重缺\n",e); free(e1);free(e3);free(e2); break; }
        /* 本专家命中的 token = 该专家的校准激活 Xc(生产口径, ds4quant_run_p9:89) */
        int nt=0;
        for(int s2=0;s2<SROW;s2++) for(int a2=0;a2<NACT;a2++)
            if(RI[(size_t)s2*NACT+a2]==e){ tok[nt]=s2; ww[nt]=RW[(size_t)s2*NACT+a2];
                memcpy(xs+(size_t)nt*D,X+(size_t)s2*D,(size_t)D*4); nt++; break; }
        if(!nt){ free(e1);free(e3);free(e2); continue; }
        /* ★校准激活按生产口径分矩阵给★(p9:89/145 w1,w3←Xc; p9:155-161 w2←中间态 h) */
        double rh; size_t bt;
        float *q1=malloc((size_t)M*D*4), *q3=malloc((size_t)M*D*4);
        /* w2 的校准激活 = 中间态 h = silu(x@w1)·(x@w3), 用 FP 权重算(生产同) */
        float *hc=malloc((size_t)nt*M*4),*gg=malloc((size_t)nt*M*4);
        const double th0=vqt_now();
        dq_matmul(xs,e1,hc,nt,D,M); dq_matmul(xs,e3,gg,nt,D,M);
        for(size_t i=0;i<(size_t)nt*M;i++){ const float t=hc[i]; hc[i]=(t/(1.0f+expf(-t)))*gg[i]; }
        free(gg); t_hc += vqt_now()-th0;
        /* ★只编码一次★: wq 版同时出 relh/bytes/Wq。原来 dsq_quant_matrix + _wq 各调一遍,
         * 24 个矩阵编码了 48 次, 还把这笔重复算进了"量化速度"。 */
        const double tq0=vqt_now();
        if(dsq_quant_matrix_wq(e1,M,D,VDIM,VNC,xs,nt,q1,&rh,&bt)==0){ relh_sum+=rh; bytes+=bt; nq++; }
        if(dsq_quant_matrix_wq(e3,M,D,VDIM,VNC,xs,nt,q3,&rh,&bt)==0){ relh_sum+=rh; bytes+=bt; nq++; }
        if(dsq_quant_matrix_wq(e2,D,M,VDIM,VNC,hc,nt,Wq,&rh,&bt)==0){ relh_sum+=rh; bytes+=bt; nq++; }
        t_q += vqt_now()-tq0;
        free(hc);
        /* FP 与【三矩阵全量化】各前向一遍 —— 这才是该层真实的量化误差 */
        const double tf0=vqt_now();
        float *aF=calloc((size_t)nt*D,4), *aQ=calloc((size_t)nt*D,4);
        expert_fwd(xs,e1,e3,e2,ww,aF,nt,D,M);
        expert_fwd(xs,q1,q3,Wq,ww,aQ,nt,D,M);
        for(int i=0;i<nt;i++) for(int d=0;d<D;d++){
            YF[(size_t)tok[i]*D+d]+=aF[(size_t)i*D+d];
            YQ[(size_t)tok[i]*D+d]+=aQ[(size_t)i*D+d]; }
        free(aF);free(aQ);free(q1);free(q3);
        t_fwd += vqt_now()-tf0;
        free(e1);free(e3);free(e2);
    }
    const double t1=vqt_now();
    if(!nq){ printf("★量化 0 个矩阵 —— HF 路径或层号不对★\n"); return 1; }
    /* ★速度必须分段★(2026-08-28 用户揪出): 原来一个计时器包住整个专家循环, 把
     * 读盘(HF FP8 反量化)、中间态两次大 GEMM、两次全量专家前向 全算进了"量化速度"。 */
    printf("①量化   ★纯编码 %6.1fs★ (%d 矩阵, %.2fs/矩阵)  平均relh=%.4f  载荷=%.1f MiB\n",
           t_q, nq, t_q/(nq?nq:1), relh_sum/nq, bytes/1048576.0);
    printf("         同循环其余: 读HF权重 %.1fs | 中间态h %.1fs | FP+量化双前向 %.1fs | 合计 %.1fs\n",
           t_rd, t_hc, t_fwd, t1-t0);
    /* ★铁律 2026-08-28「只有 GPU 版本」★ 走 vq_gpu.cu 的 GPU kmeans(vq_qc.h:53 的
     * nv>=8192 && vqg_shm_ok && vqg_ready 三条同时成立才进)。
     * 本单元【单线程】跑完整层的 744 个矩阵; 生产是 20 个专家线程并发 ⇒ 除以 20 才是
     * 可比的层时间(实测 377s/20 ≈ 19s, 与生产 60-80s 同量级)。 */
    printf("         GPU 路(vqg_assign) 单线程 ⇒ 生产 20 线程并发折算 ≈ %.1fs/层\n", t_q/20.0);

    /* ── ②反修 ── */
    float *DH=malloc((size_t)SROW*D*4);
    for(size_t i=0;i<(size_t)SROW*D;i++) DH[i]=YF[i]-YQ[i];
    double eq=0,ef=0; for(size_t i=0;i<(size_t)SROW*D;i++){ eq+=(double)DH[i]*DH[i]; ef+=(double)YF[i]*YF[i]; }
    const int vs=(SROW*3)/4;
    elm_res er; const double t2=vqt_now();
    const int erc = elm_solve(X,YQ,DH,SROW,D,vs,&er);
    const double t3=vqt_now();
    if(erc==0){
        /* ★精度要够★: -0.00% 看不出是 -1e-8 还是 -1e-3, 用科学计数把量级摆出来。
         * λ 顶格 100 + k 最小 16 是"什么都没学到"的典型签名(强正则把 U 压到近 0)。 */
        printf("②反修   %6.1fs  held行为挽回=%.4f%% (%.3e)  @V₀=%s λ=%g k=%d | 乘性线性对照=%.4f%% (%.3e)%s\n",
               t3-t2, er.held*100.0, er.held, er.from_pca?"PCA":"rand", (double)er.lam, er.k,
               er.held_lin*100.0, er.held_lin,
               (er.lam>=100.0f && er.k<=16) ? "  ★λ顶格+k最低=网格全负,解不出东西★" : "");
        printf("         量化误差基线: ‖dH‖/‖y_fp‖=%.4f (fit=%d行 held=%d行)\n",
               sqrt(eq/(ef+1e-30)), vs, SROW-vs);
    } else printf("②反修   %6.1fs  解算失败(rc=%d)\n", t3-t2, erc);

    /* ── ③sweep: 候选评估(真活) ──
     * ★口径与限制, 先说清楚★: 候选评估逐行同生产(dsq_sweep_layer ← p7:410-440)。
     * 但 hc 混合系数 post/comb 是链上残差流经 dq_hc_sinkhorn 现算的, 全流程不落盘
     * ⇒ per-layer 单测拿不到链上真值(这是结构性限制, 不是没做)。这里用锚的 H[L]
     * (教师 hc 真值)当 resid 与 Hf, post/comb 由生产 dq_hc_sinkhorn 从 H 的确定性
     * 归约现算。所以:
     *   ✓ 速度 = 真(工作量纯由形状定: 每候选一次 S×HCM×DIM 的 hc 合成 + val 行出口分)
     *   ✓ 门行为 = 真(真 z、真信任域夹持、真判据 e1<e0-1e-9、首个过门即落地)
     *   ★✗ 出口分【绝对值】≠ 生产链上的 e0 —— 不许拿它跟战役日志里的 e0 对表★
     * 反修臂用生产 ds4_z_solve(闭式 ridge + 秩截断)=冠军 r64c 那一族(加性 zl.RRR),
     * 不是 ②的乘性 ELM —— FP 口径下乘性已判无肉(fable5:5330), 加性才是冠军主力。 */
    const double t4=vqt_now();
    int land=-1, k_land=0, n_eval=0; double e0s=0, e1s=0, t_zsolve=0, t_hcw=0;
    if(ancHCM!=HCM){
        printf("③sweep  锚 HCM=%d ≠ 编译期 HCM=%d, 跳过(生产 p1 也是写死 4)\n", ancHCM, HCM);
    } else {
        const size_t hb=(size_t)SROW*HCM*D;
        float *HF=malloc(hb*4), *HQ=malloc(hb*4);
        float *mix=malloc((size_t)SROW*(2*HCM+HCM*HCM)*4);
        float *pre=malloc((size_t)SROW*HCM*4), *pst=malloc((size_t)SROW*HCM*4);
        float *cmb=malloc((size_t)SROW*HCM*HCM*4);
        float *Ftry=malloc((size_t)SROW*D*4), *Fout=malloc((size_t)SROW*D*4);
        /* 锚布局: [40] fin | ridx | rw | H[NL][S][HCM][DIM] | logits */
        const size_t rw_b=(size_t)NL*S*NACT*4;
        int hrc = HF&&HQ&&mix&&pre&&pst&&cmb&&Ftry&&Fout &&
                  fseek(f,40+(long)fin_b+(long)ridx_b+(long)rw_b+
                          (long)((size_t)L*S*HCM*D*4),SEEK_SET)==0 &&
                  fread(HF,4,hb,f)==hb;
        if(!hrc){ printf("③sweep  锚 H[L] 读失败, 跳过\n"); }
        else {
            /* mixes 的确定性归约(见上"限制"): 每行取 H 各 hc 分量的均值做特征 */
            const int nm=2*HCM+HCM*HCM;
            for(int sx=0;sx<SROW;sx++){
                float *mo=mix+(size_t)sx*nm;
                for(int j=0;j<HCM;j++){ double m=0; const float*h=HF+((size_t)sx*HCM+j)*D;
                    for(int d=0;d<D;d++) m+=h[d]; m/=D;
                    mo[j]=(float)m; mo[HCM+j]=(float)m; }
                for(int j=0;j<HCM*HCM;j++) mo[2*HCM+j]=mix[(size_t)sx*nm+(j%HCM)];
            }
            const float hsc[2]={1.0f,1.0f}; float hbase[2*HCM+HCM*HCM]; 
            for(int j=0;j<nm;j++) hbase[j]=0.0f;
            const double th0=vqt_now();
            dq_hc_sinkhorn(mix,hsc,hbase,pre,pst,cmb,SROW,HCM,3,1e-6f);
            t_hcw=vqt_now()-th0;
            /* 反修臂: 生产闭式解, rank=64(冠军 K64) */
            const double tz0=vqt_now();
            ds4_z *zl=ds4_z_solve(X,DH,(uint32_t)SROW,(uint32_t)D,(uint32_t)D,64,1.0f);
            t_zsolve=vqt_now()-tz0;
            if(!zl) printf("③sweep  ds4_z_solve 返回 NULL, 跳过\n");
            else {
                memcpy(Fout,YQ,(size_t)SROW*D*4);
                const int LZRANK=64;
                const int cand[7]={LZRANK,LZRANK/2,16,8,8,4,1};   /* 生产 cand0 同款 */
                printf("③sweep  候选评估(生产 ZLGATE 口径, 真 z rank=64, 信任域 LZTR=0.5):\n");
                land=dsq_sweep_layer(zl,X,Fout,Ftry,HF,pst,cmb,HQ,HF,
                                     SROW,vs,SROW,0.5f,cand,7,&k_land,&e0s,&e1s,&n_eval);
                ds4_z_free(zl);
            }
        }
        free(HF);free(HQ);free(mix);free(pre);free(pst);free(cmb);free(Ftry);free(Fout);
    }
    const double t5=vqt_now();
    if(land>=0){
        printf("③sweep  %6.1fs  评了 %d 个候选 (z解算 %.1fs + hc系数 %.2fs + 候选评估 %.1fs)\n",
               t5-t4, n_eval, t_zsolve, t_hcw, (t5-t4)-t_zsolve-t_hcw);
        if(land==1) printf("         → k=%d 落地  出口分 %.6f→%.6f (降 %.3f%%)\n",
                           k_land,e0s,e1s,(e0s-e1s)/(e0s>1e-30?e0s:1)*100.0);
        else        printf("         → 全拒(%d 个候选无一过门 e1<e0-1e-9), Fout 不动\n", n_eval);
        printf("         ★出口分绝对值不可与战役日志对表(hc 系数非链上真值, 见代码注释)★\n");
    }
    if(erc==0) elm_free(&er);
    free(X);free(RI);free(RW);free(YQ);free(YF);free(DH);free(tok);free(ww);free(xs);free(Wq);
    return 0;
}

