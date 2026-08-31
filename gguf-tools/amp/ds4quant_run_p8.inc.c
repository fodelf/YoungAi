    dq_hc_post(Fout,H2,post2,comb2,H,S,HCM,DIM);   /* 就地更新 H(误差传下层=累积) */
    free(cosr);free(sinr);free(y);free(post);free(comb);free(xn);if(kvc)free(kvc);free(a);free(H2);free(y2);free(post2);free(comb2);free(Fin);free(idx);free(rw);free(Fout);if(z_shb)free(z_shb);
    if(g_bflt_on){ for(int i=0;i<8;i++) g_bflt[i]+=g_lt[i]; g_bflt[9]+=vqt_now()-lt_t0; g_bflt[10]+=1.0; extern double g_bmw_t[2]; g_bflt[16]+=g_bmw_t[0]; g_bflt[17]+=g_bmw_t[1]; }   /* BFUNIT 单元账(p3 bflt_print); 16/17=本层 bdq/gemm */
}

/* head: H[S,HCM,DIM] → logits[S,VOCAB] (vocab 投影批量 sgemm) */
/* ★yv 方向诊断(2026-08-27 用户判"引擎里有计算磨平收益"): logits 只吃 rms(yv) 的方向,
 * yv 的模长被 RMS 抹掉。反修解的是 L2(模长+方向), 若改善集中在模长 → relL2 大降而
 * KLD 不动(实测 L40 −41.6%/L42 −23.4% 对 KLD −0.27%)。本诊断直接量方向: 与 FP 的
 * yv 逐行 cos。ANC_OK 时启用, 只读不改数值。 */
static void yv_dir_diag(const float *yvq, const float *yvf, int S) {
    double sc = 0, sn = 0; int n = 0;
    for (int s = 0; s < S; s++) {
        const float *a = yvq + (size_t)s * DIM, *b = yvf + (size_t)s * DIM;
        double d = 0, na = 0, nb = 0, e2 = 0;
        for (int j = 0; j < DIM; j++) {
            d += (double)a[j] * b[j]; na += (double)a[j] * a[j]; nb += (double)b[j] * b[j];
            double df = (double)a[j] - b[j]; e2 += df * df;
        }
        if (na > 0 && nb > 0) { sc += d / (sqrt(na) * sqrt(nb)); sn += sqrt(e2 / nb); n++; }
    }
    if (n) fprintf(stderr, "[yvdiag] 最终 yv: 方向 cos=%.6f | relL2=%.6f (行=%d)\n",
                   sc / n, sn / n, n);
}

static void head_fwd(float*H,int S,float*hcfn,float*hcb,float*hcs,float*norm,float*hw,float*logits){
    int HD_=HCM*DIM;
    float *yn=malloc((size_t)S*DIM*4);
    float *yvall=malloc((size_t)S*DIM*4);
    for(int s=0;s<S;s++){ const float*x=H+(size_t)s*HD_; double v=0;for(int i=0;i<HD_;i++)v+=(double)x[i]*x[i];float rsq=(float)(1.0/sqrt(v/HD_+EPSF));
        float mix[8],pre[8]; for(int j=0;j<HCM;j++){const float*fr=hcfn+(size_t)j*HD_;float aa=0;for(int k=0;k<HD_;k++)aa+=x[k]*fr[k];mix[j]=aa*rsq;pre[j]=dq_sigmoid(mix[j]*hcs[0]+hcb[j])+EPSF;}
        float yv[DIM]; for(int d=0;d<DIM;d++){float aa=0;for(int j=0;j<HCM;j++)aa+=pre[j]*H[((size_t)s*HCM+j)*DIM+d];yv[d]=aa;}
        memcpy(yvall+(size_t)s*DIM,yv,(size_t)DIM*4);
        dq_rms(yv,norm,yn+(size_t)s*DIM,DIM,EPSF);
    }
    if(ANC_OK&&ANC.H){   /* FP 侧同式合成 yv, 比方向(RMS 前) */
        float *yvf=malloc((size_t)S*DIM*4);
        const float *Hf=ANC.H+(size_t)(NLAYERS-1)*(size_t)S*HCM*DIM;
        for(int s=0;s<S;s++){ const float*x=Hf+(size_t)s*HD_; double v=0;
            for(int i=0;i<HD_;i++)v+=(double)x[i]*x[i];
            float rsq=(float)(1.0/sqrt(v/HD_+EPSF));
            float mix[8],pre[8];
            for(int j=0;j<HCM;j++){const float*fr=hcfn+(size_t)j*HD_;float aa=0;
                for(int k=0;k<HD_;k++)aa+=x[k]*fr[k];mix[j]=aa*rsq;
                pre[j]=dq_sigmoid(mix[j]*hcs[0]+hcb[j])+EPSF;}
            for(int d=0;d<DIM;d++){float aa=0;for(int j=0;j<HCM;j++)aa+=pre[j]*Hf[((size_t)s*HCM+j)*DIM+d];
                yvf[(size_t)s*DIM+d]=aa;}
        }
        yv_dir_diag(yvall,yvf,S); free(yvf);
    }
    dq_matmul(yn,hw,logits,S,DIM,VOCAB);
    free(yn); free(yvall);
}

/* head_fwd 低内存版(2026-07-28 里程碑尖刺修): head.weight [VOCAB,DIM] 按行块流式
 * 直读(BF16/F32), 峰值 ~250MB vs 全量 fp32 2.1G(看门狗 12.3G 击杀根因)。 */
static void head_fwd_stream(float*H,int S,float*hcfn,float*hcb,float*hcs,float*norm,float*logits){
    int HD_=HCM*DIM;
    float *yn=malloc((size_t)S*DIM*4);
    for(int s=0;s<S;s++){ const float*x=H+(size_t)s*HD_; double v=0;for(int i=0;i<HD_;i++)v+=(double)x[i]*x[i];float rsq=(float)(1.0/sqrt(v/HD_+EPSF));
        float mix[8],pre[8]; for(int j=0;j<HCM;j++){const float*fr=hcfn+(size_t)j*HD_;float aa=0;for(int k=0;k<HD_;k++)aa+=x[k]*fr[k];mix[j]=aa*rsq;pre[j]=dq_sigmoid(mix[j]*hcs[0]+hcb[j])+EPSF;}
        float yv[DIM]; for(int d=0;d<DIM;d++){float aa=0;for(int j=0;j<HCM;j++)aa+=pre[j]*H[((size_t)s*HCM+j)*DIM+d];yv[d]=aa;}
        dq_rms(yv,norm,yn+(size_t)s*DIM,DIM,EPSF);
    }
    char shard[256]; long ds2; char dt[16]; long shp[2],off[2];
    if(!st_shard(&C,"head.weight",shard)){ fprintf(stderr,"[head-stream] shard 缺失\n"); exit(2); }
    char *hdr=st_shard_hdr(&C,shard,&ds2);
    if(!hdr||!st_find(hdr,"head.weight",dt,shp,off)){ fprintf(stderr,"[head-stream] 头解析失败\n"); exit(2); }
    int isf32=strcmp(dt,"F32")==0;
    char p[1300]; snprintf(p,sizeof(p),"%s/%s",C.hf,shard);
    FILE*f=fopen(p,"rb");
    const int BS=8192;
    float *blk=malloc((size_t)BS*DIM*4),*tmp=malloc((size_t)S*BS*4);
    uint16_t *b16=isf32?NULL:malloc((size_t)BS*DIM*2);
    for(long v0=0;v0<VOCAB;v0+=BS){
        long bs=VOCAB-v0<BS?VOCAB-v0:BS;
        fseek(f,ds2+off[0]+(long)((size_t)v0*DIM*(isf32?4:2)),SEEK_SET);
        if(isf32){ if(fread(blk,4,(size_t)bs*DIM,f)!=(size_t)bs*DIM){fprintf(stderr,"[head-stream] 读断\n");exit(2);} }
        else{ if(fread(b16,2,(size_t)bs*DIM,f)!=(size_t)bs*DIM){fprintf(stderr,"[head-stream] 读断\n");exit(2);}
              for(size_t i=0;i<(size_t)bs*DIM;i++){ uint32_t u=(uint32_t)b16[i]<<16; memcpy(&blk[i],&u,4);} }
        dq_matmul(yn,blk,tmp,S,DIM,(int)bs);
        for(int s=0;s<S;s++) memcpy(logits+(size_t)s*VOCAB+v0,tmp+(size_t)s*bs,(size_t)bs*4);
    }
    fclose(f); free(hdr); free(blk); free(tmp); if(b16)free(b16); free(yn);
}

/* 向后·终端反调的 val 行 KL(fp‖q): 选 t 只用 fit 尾部 val 行, held 永不参与 */
static double bwd_val_kl(const float*lf,const float*lq,int a,int b){
    double kl=0; int n=0;
    for(int s=a;s<b;s++){
        const float*x=lf+(size_t)s*VOCAB,*y=lq+(size_t)s*VOCAB;
        float ma=x[0],mb=y[0];
        for(int v=1;v<VOCAB;v++){ if(x[v]>ma)ma=x[v]; if(y[v]>mb)mb=y[v]; }
        double sa=0,sb=0;
        for(int v=0;v<VOCAB;v++){ sa+=exp((double)x[v]-ma); sb+=exp((double)y[v]-mb); }
        double lsa=log(sa),lsb=log(sb),krow=0;
        for(int v=0;v<VOCAB;v++){
            double lpf=(double)x[v]-ma-lsa, lpq=(double)y[v]-mb-lsb;
            double pf=exp(lpf); if(pf>0) krow+=pf*(lpf-lpq); }
        kl+=krow; n++;
    }
    return n?kl/n:0.0;
}
/* 单 token 行 KL(P_fp||P_q) — 反修按 token 挑最优乘子用 */
static double bwd_tok_kl(const float*x,const float*y){
    float ma=x[0],mb=y[0];
    for(int v=1;v<VOCAB;v++){ if(x[v]>ma)ma=x[v]; if(y[v]>mb)mb=y[v]; }
    double sa=0,sb=0;
    for(int v=0;v<VOCAB;v++){ sa+=exp((double)x[v]-ma); sb+=exp((double)y[v]-mb); }
    double lsa=log(sa),lsb=log(sb),k=0;
    for(int v=0;v<VOCAB;v++){ double lpf=(double)x[v]-ma-lsa,lpq=(double)y[v]-mb-lsb;
        double pf=exp(lpf); if(pf>0)k+=pf*(lpf-lpq); }
    return k;
}
/* 对称正定线性系统部分主元高斯消元(A n×n 行主序, 破坏性); 解入 x。ridge 已由调用方加对角。*/
static int solve_sym(double*A,double*b,int n,double*x){
    for(int c=0;c<n;c++){
        int piv=c; double best=fabs(A[(size_t)c*n+c]);
        for(int r=c+1;r<n;r++){ double v=fabs(A[(size_t)r*n+c]); if(v>best){best=v;piv=r;} }
        if(best<1e-18) return -1;
        if(piv!=c){ for(int j=0;j<n;j++){ double t=A[(size_t)c*n+j];A[(size_t)c*n+j]=A[(size_t)piv*n+j];A[(size_t)piv*n+j]=t; }
                    double t=b[c];b[c]=b[piv];b[piv]=t; }
        double d=A[(size_t)c*n+c];
        for(int r=c+1;r<n;r++){ double f=A[(size_t)r*n+c]/d; if(f==0)continue;
            for(int j=c;j<n;j++) A[(size_t)r*n+j]-=f*A[(size_t)c*n+j]; b[r]-=f*b[c]; }
    }
    for(int r=n-1;r>=0;r--){ double s=b[r];
        for(int j=r+1;j<n;j++) s-=A[(size_t)r*n+j]*x[j];
        x[r]=s/A[(size_t)r*n+r]; }
    return 0;
}
static void export_layer_file(int L,int S,int n_fit,const char*lf);
static int backfit_prev(int Lfront,const long*ids,int S,int n_fit);   /* 逐层反修(定义在 global_sweep 前) */
static float *gs_forward_exit(int J,int Lend,const float*Hin,const long*ids,int S,int n_fit,float*HQcache);   /* BF_ONLY 回放推进用 */
static double held_score(const float*Hq,int S,int n_fit,int L,double*relh_out);   /* 反修内动态α用(定义在贪心段) */
static float *fwd_all(const long*ids,int S,int n_fit,int do_quant,const char*lcfg){
    size_t lstride=(size_t)S*HCM*DIM;
    float *H=malloc(lstride*4);
    int L0=0;
    if(do_quant&&ANC_OK){   /* F 前缀 = FP 锚定原样, 直接恢复跳过 */
        int p=0; while(p<NLAYERS&&lcfg[p]=='F')p++;
        if(p>0){ memcpy(H,ANC.H+(size_t)(p-1)*lstride,lstride*4); L0=p;
            fprintf(stderr,"[前缀] L0..L%d 全 F → anchor 恢复, 从 L%d 开跑\n",p-1,L0<NLAYERS?L0:NLAYERS-1); }
    }
    if(L0==0){
        float *emb=st_read_weight(&C,"embed.weight",NULL,NULL);
        for(int s=0;s<S;s++)for(int j=0;j<HCM;j++)memcpy(H+((size_t)s*HCM+j)*DIM,emb+(size_t)ids[s]*DIM,(size_t)DIM*4);
        free(emb);
    }
    if(do_quant) fprintf(stderr,"\n判决=最终输出; 逐层为诊断: 累积=H_q vs FP锚定(误差爆炸在哪层) 局部=该层专家复现(旧口径)\n");
    /* ★逐层反修★缓存: 骨干权重(留存不 free)+层文件(export 后开)+量化态入口隐藏; 前沿层完成即反修 0..L-1 */
    int incr = do_quant && ANC_OK && COADAPT>0 && g_cli.layer_dir && lcfg;   /* 增量反修恒开(关闭开关 2026-08-31 清退) */
    /* ★只跑反修(2026-07-13, 用户裁决: "加个参数只跑返修")★: 推进段不重跑 SEARCH —
     * 逐层加载既有 dql 按 op 链字节回放推进累积态+建 HQE/GS_LF/GS_LW, 直达终局收敛 sweep。
     * 前提=层文件全齐(缺一层硬停); 已落地的 bf.* 修正随层文件一并回放(在其上继续叠加)。 */
    int BF_ONLY = incr && !FAST && g_cli.bf_only;
    if(BF_ONLY) fprintf(stderr,"[只跑反修] 开: 复用既有层文件(SEARCH 跳过), 回放推进 → 终局收敛 sweep\n");
    if(incr){
        if(!GS_LW) GS_LW=calloc((size_t)NLAYERS,sizeof(LWH));
        if(!GS_LF) GS_LF=calloc((size_t)NLAYERS,sizeof(lfile_t));
        if(!HQE){
            /* ★反修内存架构: HQE 快照(S=1716 时 ~2.5G)由 malloc(脏页)改为 /tmp 匿名文件后备
             * mmap(MAP_SHARED) — 有磁盘后备可换出, 内存压力下自回收, 与锚 mmap 合计砍反修
             * 基线脏内存 ~7G ⇒ 终局 sweep 进 12G 红线(2026-07-31 用户令核心价值必须反修)。 */
            size_t hb=(size_t)(NLAYERS+1)*lstride*4;
            /* ★回退匿名 mmap "优化"(2026-08-27 我引入, 08-28 实撞回退)★
             * 我当时按"物理内存 ≥ 快照 4 倍"就改走匿名 mmap(纯内存), 判据只看了快照本身
             * (22GiB×4=88 ≤ 121GiB 通过), ★没算这活儿自身还要 ~95GB★ ⇒ 95+22=117 把 121GB
             * 机器挤到 MemAvailable=2GB, 被看门狗停在 sweep 第 4 个单元。
             * 文件后备的价值恰恰在"可回收": MAP_SHARED 有磁盘后备, 内存压力下内核直接丢页;
             * 匿名页(无 swap)不可回收。这是原设计的安全性质, 不是小机器遗留。
             * 而且这个"优化"从头到尾没测过收益 —— 属于照注释猜的改动, 不该有。 */
            char hqp[]="/tmp/ds4_hqe_XXXXXX"; int hfd=mkstemp(hqp);
            if(hfd<0){ perror("hqe-tmp"); exit(1); }
            unlink(hqp);
            if(ftruncate(hfd,(off_t)hb)!=0){ perror("hqe-trunc"); exit(1); }
            HQE=mmap(NULL,hb,PROT_READ|PROT_WRITE,MAP_SHARED,hfd,0);
            close(hfd);
            if(HQE==MAP_FAILED){ perror("hqe-mmap"); exit(1); }
            fprintf(stderr,"[反修] HQE 快照 %.2f GiB → 文件后备 mmap(可换出)\n",hb/1073741824.0);
            g_hqe_lstride=lstride;
        }
        if(!BF_FINOP){ BF_FINOP=malloc((size_t)NLAYERS*sizeof(int)); for(int i=0;i<NLAYERS;i++) BF_FINOP[i]=-1; }
        if(!BF_DYN2OP){ BF_DYN2OP=malloc((size_t)NLAYERS*sizeof(int)); for(int i=0;i<NLAYERS;i++) BF_DYN2OP[i]=-1; }
        if(!BF_GEOP){ BF_GEOP=malloc((size_t)NLAYERS*sizeof(int)); for(int i=0;i<NLAYERS;i++) BF_GEOP[i]=-1;
            BF_HCOP=malloc((size_t)NLAYERS*sizeof(int)); for(int i9=0;i9<NLAYERS;i9++) BF_HCOP[i9]=-1; }
        if(!GS_IDXC){ GS_IDXC=malloc((size_t)S*NACT*sizeof(int)); GS_RWC=malloc((size_t)S*NACT*4); }
        fprintf(stderr,"[逐层反修] 开(终局收敛): 推进段不修(误差前向吸收=部署口径), 收尾以最终出口为判据全层 sweep(旧逐前沿模式已删, 2026-08-31 env逻辑清退)\n");
    }
    for(int L=L0;L<NLAYERS;L++){
        LW W=load_layer2(L,incr?&GS_LW[L]:NULL); lstat_t st; memset(&st,0,sizeof(st));   /* incr: 顺带录元素数(fp16缓存用) */
        fprintf(stderr,"L%02d %c ",L,do_quant?lcfg[L]:'@');
        if(incr&&HQE){ memcpy(HQE+(size_t)L*lstride,H,lstride*4);   /* 层 L 量化态入口(反修从此重前向; fast 无 HQE) */
            msync(HQE+(size_t)L*lstride,lstride*4,MS_ASYNC); }   /* 异步回写(内存卫生) */
        if(BF_ONLY&&do_quant&&lcfg[L]=='g'){
            /* 只跑反修: 加载既有 dql(含已落地 bf.* 修正)→ 绑定链末 op 槽位 → 字节回放本层 */
            lwh_absorb(&GS_LW[L],&W);
            char lp0[512]; snprintf(lp0,sizeof(lp0),"%s/dql_L%02d.bin",g_cli.layer_dir,L);
            if(GS_LF[L].map){ lfile_free(&GS_LF[L]); memset(&GS_LF[L],0,sizeof(lfile_t)); }
            if(lfile_load(lp0,&GS_LF[L])!=0){
                fprintf(stderr,"[只跑反修] L%02d 层文件缺失/损坏(%s) — 需 %d 层全齐, 硬停\n",L,lp0,NLAYERS);
                exit(9);
            }
            for(int i=GS_LF[L].nops-1;i>=0;i--){ int t=GS_LF[L].ops[i].type;   /* 链末槽位: 反修"已有则原地重解/累乘"口径 */
                if(t==1&&BF_FINOP[L]<0) BF_FINOP[L]=i;
                else if(t==2&&BF_DYN2OP[L]<0) BF_DYN2OP[L]=i;
                else if(t==5&&BF_GEOP[L]<0) BF_GEOP[L]=i; }
            float*He=gs_forward_exit(L,L,H,ids,S,n_fit,NULL);
            memcpy(H,He,lstride*4); free(He);
            memcpy(HQE+(size_t)(L+1)*lstride,H,lstride*4);
            msync(HQE+(size_t)(L+1)*lstride,lstride*4,MS_ASYNC);   /* 异步回写(内存卫生) */
            if(g_chfd>=0&&S==g_chS){   /* ★链态锚: 层出口 H 直写★ */
                size_t bH=40+(size_t)NLAYERS*S*DIM*4+2*(size_t)NLAYERS*S*NACT*4;
                pwrite(g_chfd,H,lstride*4,(off_t)(bH+(size_t)L*lstride*4));
                fdatasync(g_chfd); }   /* 内存卫生: 大 S 时脏页尽早落盘可回收(2026-08-24; 当日 OOM 真因=两条重任务并行, 非本处) */
            { const float *Hf=ANC.H+(size_t)L*lstride;   /* 累积诊断(与正常路径同口径) */
              size_t rowsz0=(size_t)HCM*DIM, fitsz=(size_t)n_fit*rowsz0;
              double e2=0,a2=0,e2h=0,a2h=0;
              for(size_t i=0;i<fitsz;i++){ double d=(double)H[i]-Hf[i]; e2+=d*d; a2+=(double)Hf[i]*Hf[i]; }
              for(size_t i=fitsz;i<lstride;i++){ double d=(double)H[i]-Hf[i]; e2h+=d*d; a2h+=(double)Hf[i]*Hf[i]; }
              fprintf(stderr," | 回放累积relL2 fit=%.4f held=%.4f | ops=%d\n",
                      sqrt(e2/(a2+1e-30)),sqrt(e2h/(a2h+1e-30)),GS_LF[L].nops); }
            fprintf(stderr,"[mem] L%02d footprint=%.2fGB 只跑反修回放\n",L,mem_gb());
            /* ★逐阶段墙钟(DS4_LT=1)★ 定位"单层 48-51 秒花在哪": p7 埋点, 这里汇总打印。
             * 三段和小于总时长的部分 = 本函数之外(层文件读写/HQE 快照/驱逐重载)。 */
            { extern double g_lt[8];   /* 恒打: 一层一行, 零成本, 不新增 env(铁律 08-22) */
                double tt=0; for(int i=0;i<8;i++) tt+=g_lt[i];
                fprintf(stderr,"[LT] L%02d attn=%.1f zsolve+路由=%.1f 共享=%.1f bmoe1=%.1f lfload=%.1f fp教师=%.1f 其余=%.1f zlgate=%.1f | 合计=%.1fs\n",
                        L,g_lt[0],g_lt[1],g_lt[2],g_lt[3],g_lt[6],g_lt[7],g_lt[4],g_lt[5],tt); }
            gs_lw_evict(L-6>0?L-6:0);
            free_layer(&W);
            continue;
        }
        layer_fwd(L,&W,H,ids,S,n_fit,do_quant,do_quant?lcfg[L]:'F',&st);
        if(ANC_BUILD) memcpy(ANC.H+(size_t)L*lstride,H,lstride*4);
        if(do_quant&&ANC_OK){
            /* 累积偏差拆 fit/held: z^L 的优化目标=压低 fit 行, held 列才是逐层可见的泛化真相 */
            const float *Hf=ANC.H+(size_t)L*lstride;
            size_t rowsz=(size_t)HCM*DIM, fitsz=(size_t)n_fit*rowsz;
            double e2=0,a2=0,e2h=0,a2h=0;
            for(size_t i=0;i<fitsz;i++){ double d=(double)H[i]-Hf[i]; e2+=d*d; a2+=(double)Hf[i]*Hf[i]; }
            for(size_t i=fitsz;i<lstride;i++){ double d=(double)H[i]-Hf[i]; e2h+=d*d; a2h+=(double)Hf[i]*Hf[i]; }
            double rel=sqrt(e2/(a2+1e-30)), relh=sqrt(e2h/(a2h+1e-30));
            fprintf(stderr," | 累积relL2 fit=%.4f held=%.4f | 路由一致=%5.1f%%",rel,relh,st.agree);
            if(lcfg[L]!='F'){
                fprintf(stderr," | 校准µ=%.1f行 空=%d/%d",st.nhit?st.calib_rows/st.nhit:0.0,st.calib_empty,st.nhit);
                if(st.have_loc) fprintf(stderr," | 局部R²=%5.1f%% R/S=%.2f",st.loc_r2*100.0,st.rs_ratio);
                if(LZRANK>0) fprintf(stderr," | z^L k=%d",st.zk);
                fprintf(stderr," | LS修[负%ld 爆%ld 败%ld /%ld]",go1b_joint_neg,go1b_joint_clip,go1b_joint_fail,go1b_joint_rows);
            }
            fputc('\n',stderr);
            go1b_joint_neg=go1b_joint_clip=go1b_joint_fail=go1b_joint_rows=0;   /* 计数按层清零 */
        } else fprintf(stderr," | anchor 捕获\n");
        if(incr){ lwh_absorb(&GS_LW[L],&W);          /* incr: 骨干 fp16 进缓存(fp32 即刻还, 防单调涨爆12G) */
            fprintf(stderr,"[mem] L%02d footprint=%.2fGB fp16缓存至本层\n",L,mem_gb());
            gs_lw_evict(L-6>0?L-6:0);                /* 超预算驱逐最远层(近6层是每条链的尾巴, 不驱逐) */
        }
        free_layer(&W);
        if(do_quant&&COADAPT>0&&lcfg&&lcfg[L]=='g'&&g_cli.layer_dir){
            char lp[512]; snprintf(lp,sizeof(lp),"%s/dql_L%02d.bin",g_cli.layer_dir,L);
            export_layer_file(L,S,n_fit,lp);   /* 每层产物当场留存(可单层重跑微调, 最后合并) */
            /* (z^L 落盘已并入 export_layer_file 正位直写 op 侧车 — 旧 append 块删除, 错序温床根除) */
            if(incr){                          /* ★逐层前进即反修前面所有层(用户设计)★ */
                if(GS_LF[L].map){ lfile_free(&GS_LF[L]); memset(&GS_LF[L],0,sizeof(lfile_t)); }
                lfile_load(lp,&GS_LF[L]);                        /* 本层文件进缓存 */
                zc_opt_emit(L,&GS_LF[L]);      /* 一层两份: dql(量化)+opt(优化)并排落盘; 终值由 zchain_write 刷新 */
                memcpy(HQE+(size_t)(L+1)*lstride,H,lstride*4);   /* 层 L 量化态出口(终局sweep重放起点; fast 也反修) */
                /* 旧·逐前沿反修分支已删(2026-08-31 env 逻辑清退): 只存终局收敛一种语义 */
            }
        }
    }
    if(do_quant&&incr&&HQE&&GS_LF&&NLAYERS>1&&n_fit>1){   /* n_fit<=1=判尺纯回放(caliper 既有约定 DS4_NFIT=1), 无 fit 行无可解, sweep 不触发 */
        /* ★终局收敛反修(2026-07-13)★: 判据=最终层出口 vs FP锚(真目标), 全层 sweep 循环到无落地。
         * 取代逐前沿 O(L³): 前沿判据是移动代理靶(同层随推进被反复翻修, 增量互相覆盖), 推进段的
         * 误差本就由下游各层自适应求解前向吸收(部署口径); 终局判据下每份修正只做一次、直指真目标。
         * 复杂度 O(K·L²), 实测类坐标下降 2-3 轮即干; DS4_BF_TERM_MAXP 护栏防不收敛。 */
        int maxp=g_cli.bf_term_maxp;   /* 默认=用户设计: 末层反修一遍(含复检)即止; 实测第2轮2.7h只换-0.7%, 多轮重扫默认不开 */
        for(int p=0;p<maxp;p++){
            int ch=backfit_prev(NLAYERS-1,ids,S,n_fit);
            printf("BACKFIT_TERM pass=%d 落地=%d%s\n",p,ch,ch?"":" → 收敛"); fflush(stdout);
            if(!ch) break;
        }
        memcpy(H,HQE+(size_t)NLAYERS*lstride,lstride*4);   /* 反修后最终出口 → 下方 logits/BWDFIN 同口径 */
    }
    /* (final hidden 捕获已删: DS4_DUMPH 诊断路 2026-08-31 env 清退) */
    float *hcfn=st_read_weight(&C,"hc_head_fn",NULL,NULL),*hcb=st_read_weight(&C,"hc_head_base",NULL,NULL),*hcs=st_read_weight(&C,"hc_head_scale",NULL,NULL);
    float *norm=st_read_weight(&C,"norm.weight",NULL,NULL),*hw=st_read_weight(&C,"head.weight",NULL,NULL);
    float *logits=malloc((size_t)S*VOCAB*4); head_fwd(H,S,hcfn,hcb,hcs,norm,hw,logits);
    if(do_quant&&ANC_OK&&BWD_L>=0&&g_cli.bwd_final){
        /* 向后·终端反调: 层出口 H 对修正量线性 → H(t)=lerp(H_base,H_corr,t);
         * t 网格重跑后缀+head 到最终 logits, val 行(fit 尾 1/4)选 t, held 行不参与选择 */
        int vsq=DS4_AMP_FIT_SPLIT(n_fit);
        double kl1=bwd_val_kl(ANC.logits,logits,vsq,n_fit);
        printf("BWDFIN L=%d t=1.00 val行KL=%.4f (基准)\n",BWD_L,kl1); fflush(stdout);
        double bestkl=kl1; float bestt=1.0f; float *bestlog=NULL;
        const float TT[3]={0.85f,1.15f,1.30f};
        for(int ti=0;ti<3;ti++){ float t=TT[ti];
            if(tune_over()){ printf("BWDFIN L=%d 时间预算耗尽 → 跳过剩余 t 候选\n",BWD_L); fflush(stdout); break; }
            float *Hb2=malloc(lstride*4);
            for(size_t i=0;i<lstride;i++) Hb2[i]=BWD_Hb[i]+t*(BWD_Hc[i]-BWD_Hb[i]);
            for(int l=BWD_L+1;l<NLAYERS;l++){ LW W2=load_layer(l);
                { char pg[32],ag[32]; snprintf(pg,sizeof(pg),"后缀%d/%d",l,NLAYERS-1);
                  snprintf(ag,sizeof(ag),"终端反调t=%.2f",t);
                  mlog(BWD_L,"向后",ag,pg,4,"待测","进行中"); }
                layer_fwd(l,&W2,Hb2,ids,S,n_fit,0,'F',NULL); free_layer(&W2); }
            float *lg=malloc((size_t)S*VOCAB*4); head_fwd(Hb2,S,hcfn,hcb,hcs,norm,hw,lg);
            double kl=bwd_val_kl(ANC.logits,lg,vsq,n_fit);
            printf("BWDFIN L=%d t=%.2f val行KL=%.4f%s\n",BWD_L,t,kl,kl<bestkl?" ✓":""); fflush(stdout);
            if(kl<bestkl){ bestkl=kl; bestt=t; free(bestlog); bestlog=lg; } else free(lg);
            free(Hb2);
        }
        BWDFIN_T=bestt;
        el_add("bwd.final","H(t)线性插值+后缀重前向, final-logits val行KL 选 t",4,
               kl1,bestkl,kl1>1e-12?100.0*(kl1-bestkl)/kl1:0.0,bestt,bestt!=1.0f?1:3);
        if(bestt!=1.0f&&g_cli.layer_dir){
            /* ★落地缺口修复★: 判决用了 H(t) 但此前只写台账行 → 文件回放/DQZ2/运行时全缺此项
             * (t=1 无实害, t≠1 判决虚高)。hc_post 对 F 线性 ⇒ H(t)=lerp ≡ 末层 TREF(锚 Fcur),
             * 名含 TREF → lfile_load/zchain/引擎三方零改动直接认。 */
            char lpB[512]; op_host_path(BWD_L,lpB,sizeof(lpB));   /* 平行架构: 落地进 op 侧车 */
            if(!append_rec(lpB,"bwd.TREF.fin","终端反调 H(t)=lerp ≡ 末层TREF(锚Fcur)",&bestt,4,(float)bestkl))
                fprintf(stderr,"[bwdfin] 落地追加失败: %s (文件与判决将不一致!)\n",lpB);
        }
        printf("BWDFIN_BEST L=%d t=%.2f val行KL %.4f→%.4f 提升%.1f%%\n",
               BWD_L,bestt,kl1,bestkl,kl1>1e-12?100.0*(kl1-bestkl)/kl1:0.0);
        fflush(stdout);
        if(bestlog){ free(logits); logits=bestlog; }
    }
    /* HQE 现为文件后备 mmap(2026-07-31 反修内存改造) — 必须 munmap 不能 free
     * (漏改这里 = Abort trap 6, 终局 sweep 收敛后清理阶段崩, 实测踩过一次) */
    if(HQE){ munmap(HQE,(size_t)(NLAYERS+1)*(size_t)g_hqe_lstride*4); HQE=NULL; }
    if(BF_FINOP){ free(BF_FINOP); BF_FINOP=NULL; }
    if(BF_DYN2OP){ free(BF_DYN2OP); BF_DYN2OP=NULL; }
    if(BF_GEOP){ free(BF_GEOP); BF_GEOP=NULL; }
    if(BF_HCOP){ free(BF_HCOP); BF_HCOP=NULL; }
    if(GS_IDXC){ free(GS_IDXC); GS_IDXC=NULL; } if(GS_RWC){ free(GS_RWC); GS_RWC=NULL; }
    /* ★这里原来有第二个 Δb 落盘器, 已删(2026-08-28 实锤 bug)★
     * 它读的是 DS4_ROUTE_BIAS_FIT 这个【开关】的值当路径 —— 那是【开关】不是路径, 战役脚本给的值是 "1",
     * 于是整个路由偏置侧车被写进了工作目录下一个名叫 `1` 的文件(实撞: gguf-tools/amp/1,
     * 88080 字节 = 16 头 + 43×256×4 均值 + 43×256×4 计数, margin 事件 659 万条)。
     * 更要命的是它写完就 free(RB_ACC) —— 而本函数 fwd_all 每跑完一遍完整前向就执行到这里,
     * 收官时真正的落盘函数 rb_save(用 DS4_ROUTE_BIAS_OUT, 还额外产 .alpha.txt)拿到的永远是
     * 空指针, 打印"Δb 无统计可落盘(哈希路由=选择零漂移)" —— 那句话把人往哈希路由上带,
     * 跟真因(变量名用错)毫无关系, 我第一遍就被它带偏了。
     * 落盘只留 rb_save 一处(铁律: 一份代码, 禁同功能重复); 它的 mincnt 门比这里更严。
     * RB_ACC/RB_CNT 各 44KB, 留到进程退出, 不再在这里 free。 */
    free(H);free(hcfn);free(hcb);free(hcs);free(norm);free(hw);return logits;
}

/* 专家体积账(GiB): 每层 routed 6.442B 参数(256×3×2048×4096), 按档 bit/el 累计。
 * F=8.0(FP8 原精度, 仅验证对照, 非量化产物); n/1/z=1.0625(块scale); r=1.0052(行scale,
 * 纯1bit体积主线); 2=2.125; 3=3.1875。'z' 另加低秩因子 ZK×6144×2B×3/专家。骨干 Q8 另计 ~8.4。 */
static double lcfg_expert_gib(const char*lcfg){
    const double PL=256.0*3*2048*4096, GI=1073741824.0;
    double tot=0;
    for(int L=0;L<NLAYERS;L++){
        double bits;
        switch(lcfg[L]){
            case 'n': case '1': case 'z': bits=1.0625; break;
            case 'r': case 'g': bits=1.0+16.0*(2048+2048+4096)/(3.0*2048*4096); break;  /* 1.00521 */
            case '2': bits=2.125; break;
            case '3': bits=3.1875; break;
            default:  bits=8.0; break;   /* F */
        }
        tot += PL*bits/8.0/GI;
        if(lcfg[L]=='z') tot += 256.0*(double)ZK*(2048+4096)*2.0*3.0/GI;
    }
    return tot;
}

