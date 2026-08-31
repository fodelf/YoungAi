/* ===================== GGUF 直写导出 (DS4_EXPORT_GGUF) ===================== *
 * 把调优配方(signref μ=DS4_SIGNREF_MU + anchor 校准)对全 43 层×256 专家量化,
 * 按 gguf_offsets.py 的偏移表直接 pwrite 进已生成的 go1b GGUF(零中间文件)。
 * 引擎 GO1B_BLK 块格式的 d 字段物理上就是行 scale 复制 → 行scale方案原生兼容。 */
typedef struct { int L; char kind[8]; long off; long nel; int type; } goff_t;
static goff_t *GOFF; static int NGOFF;
static goff_t *goff_find(int L,const char*kind){
    for(int i=0;i<NGOFF;i++) if(GOFF[i].L==L&&!strcmp(GOFF[i].kind,kind)) return &GOFF[i];
    return NULL;
}
typedef struct { int L,S,n_fit; int *e_next; uint8_t *bg,*bu,*bd; size_t szG,szD;
                 int fd; size_t off0;
                 int g2fd; size_t g2off; int g2k; double g2sum[5];   /* go2b 侧车 fd/载荷偏移/热数/门统计{n,Σrel1,Σcos1,Σrel2,Σcos2}(内嵌时 g2fd=fd) */
                 int vqfd; double vqsum[4];   /* VQ 侧车 fd + 门统计{n_hot,Σcos_hot,n_cold,Σcos_cold} */
               } expw_t;   /* bg==NULL → 流式: 按偏移 pwrite 进 fd(855MB buffer→3.4MB/worker) */
static long long EXP_US_READ=0,EXP_US_VQ13=0,EXP_US_HC=0,EXP_US_VQ2=0;
static long long exp_us_now(void){ struct timeval tv; gettimeofday(&tv,NULL); return (long long)tv.tv_sec*1000000+tv.tv_usec; }
static void *export_worker(void*a){
    expw_t*w=a;
    float *Xc=malloc((size_t)(w->n_fit>0?w->n_fit:1)*DIM*4);
    uint8_t *tG=NULL,*tU=NULL,*tD=NULL;
    if(!w->bg){ tG=malloc(w->szG); tU=malloc(w->szG); tD=malloc(w->szD); }
    for(;;){ int e=__sync_fetch_and_add(w->e_next,1); if(e>=NEXP)break;
        if((e&31)==0){ char pg[24]; snprintf(pg,sizeof(pg),"%d/%d",e,NEXP);
            mlog(w->L,"量化文件","DQL2·signref字节",pg,
                 (uint64_t)NEXP*(2*w->szG+w->szD),"—","导出中"); }
        char n1[160],n3[160],n2[160];
        snprintf(n1,sizeof(n1),"layers.%d.ffn.experts.%d.w1.weight",w->L,e);
        snprintf(n3,sizeof(n3),"layers.%d.ffn.experts.%d.w3.weight",w->L,e);
        snprintf(n2,sizeof(n2),"layers.%d.ffn.experts.%d.w2.weight",w->L,e);
        long rr,cc; long long _t0=exp_us_now();
        float *e1=st_read_weight(&C,n1,&rr,&cc),*e3=st_read_weight(&C,n3,&rr,&cc),*e2=st_read_weight(&C,n2,&rr,&cc);
        __sync_fetch_and_add(&EXP_US_READ,exp_us_now()-_t0);
        if(!e1||!e3||!e2){ fprintf(stderr,"\n[!] L%d e%d 读失败\n",w->L,e); if(e1)free(e1); if(e3)free(e3); if(e2)free(e2); continue; }
        int ncal=0;
        const float *afin=NULL; const int *aidx_i=NULL; const int32_t *aidx_a=NULL;
        int csel[4096];
        if(EXP_FIN&&EXP_IDX&&EXP_L==w->L){ afin=EXP_FIN; aidx_i=EXP_IDX; }   /* ★调优同口径 */
        else if(ANC_OK){ afin=ANC.fin+(size_t)w->L*w->S*DIM; aidx_a=ANC.ridx+(size_t)w->L*w->S*NACT; }
        if(afin){
            /* --calib-fullset: 导出段(blob g_r/GPTQ-H 的真正产地)同款全集喂入 —
             * hit 过滤在 n_fit~400 时每专家仅~9行, g_r 被钉死 1.000(全模型实测)。 */
            const int fullset2=g_cli.calib_fullset;
            /* ★导出侧行帽(2026-08-09 提速定罪: 导出相位 8:48/层, 全在 per-expert 校准行
             * 矩阵乘/kmeans/GPTQ-H, 与行数线性)★: FULLSET 语义保留(全集均匀 stride 采样,
             * 无 routed-hit 饿死); --calib-export-cap 0/未传=全集(行帽实测: 速度无效+质量-0.2pt)。 */
            const int expcap=g_cli.calib_export_cap;
            int estride=1;
            if(fullset2&&expcap>0&&w->n_fit>expcap) estride=(w->n_fit+expcap-1)/expcap;   /* ceil: 933/512→2 */
            for(int s=0;s<w->n_fit;s++){ int hit=fullset2&&(s%estride==0);
                if(!fullset2) for(int a2=0;a2<NACT;a2++){ int ee=aidx_i?aidx_i[(size_t)s*NACT+a2]:(int)aidx_a[(size_t)s*NACT+a2];
                    if(ee==e){hit=1;break;} }
                if(hit){ memcpy(Xc+(size_t)ncal*DIM,afin+(size_t)s*DIM,(size_t)DIM*4);
                         if(ncal<4096)csel[ncal]=s; ncal++; } }
        }
        /* GPTAQ 非对称目标(与 coadapt 同解, 导出=拟合一致性): dx=x̃−x̂ 仅当 afin=漂移流时非零 */
        float *yadjE1=NULL,*yadjE3=NULL; { float tgaE=dq_tgt_alpha();
        if(tgaE>0.0f&&ncal&&ncal<=4096&&ANC_OK&&aidx_i){
            const float *afp2=ANC.fin+(size_t)w->L*w->S*DIM;
            float *dx=malloc((size_t)ncal*DIM*4);
            for(int t=0;t<ncal;t++){ const float *xf=afp2+(size_t)csel[t]*DIM,*xh=Xc+(size_t)t*DIM;
                for(int d=0;d<DIM;d++) dx[(size_t)t*DIM+d]=xf[d]-xh[d]; }
            yadjE1=malloc((size_t)ncal*MOEI*4); yadjE3=malloc((size_t)ncal*MOEI*4);
            dq_matmul(dx,e1,yadjE1,ncal,DIM,MOEI); dq_matmul(dx,e3,yadjE3,ncal,DIM,MOEI);
            for(size_t i2=0;i2<(size_t)ncal*MOEI;i2++){ yadjE1[i2]*=tgaE; yadjE3[i2]*=tgaE; }
            free(dx);
        } }
        int sl2=(w->g2fd>0||dq_vq_on())?g2_hot_slot(w->L,e):-1;   /* VQ 模式热判定不依赖 g2fd(冒烟bug修) */
        float *q1=NULL,*q3=NULL,*hc=NULL;
        if(sl2<0){   /* 冷专家(或 go2b 关): signref go1b。热专家跳过 → dql 热槽位=稀疏洞(省盘+省算) */
        q1=malloc((size_t)MOEI*DIM*4); q3=malloc((size_t)MOEI*DIM*4);
        uint8_t *dG=w->bg?w->bg+(size_t)e*w->szG:tG, *dU=w->bu?w->bu+(size_t)e*w->szG:tU, *dDn=w->bd?w->bd+(size_t)e*w->szD:tD;
        int vqcold=dq_vq_on();
        if(vqcold){   /* v2.2: 冷 w1/w3 → vq8x256 侧车; dql G/U 段留稀疏洞 */
            double c1v=0,c3v=0;
            long long _tv=exp_us_now();
            float *t1=vq_export_matrix(e1,MOEI,DIM,ncal?Xc:NULL,ncal,vq_cold_dim(),vq_cold_nc(),w->vqfd,vq_slot_off(w->L,e,0,MOEI,DIM),&c1v);
            float *t3=vq_export_matrix(e3,MOEI,DIM,ncal?Xc:NULL,ncal,vq_cold_dim(),vq_cold_nc(),w->vqfd,vq_slot_off(w->L,e,1,MOEI,DIM),&c3v);
            __sync_fetch_and_add(&EXP_US_VQ13,exp_us_now()-_tv);
            memcpy(q1,t1,(size_t)MOEI*DIM*4); memcpy(q3,t3,(size_t)MOEI*DIM*4); free(t1); free(t3);
            w->vqsum[2]+=2; w->vqsum[3]+=c1v+c3v;
        } else {
        dq_signref_export_adj(e1,MOEI,DIM,ncal?Xc:NULL,ncal,dG,q1,yadjE1);
        dq_signref_export_adj(e3,MOEI,DIM,ncal?Xc:NULL,ncal,dU,q3,yadjE3);
        }
        int nhc=0;
        if(ncal){   /* w2 校准 = 量化 q1/q3 过校准行(顺序补偿, 与 harness 同口径) */
            hc=malloc((size_t)ncal*MOEI*4);
            long long _th=exp_us_now();
            float *gg=malloc((size_t)ncal*MOEI*4),*uu=malloc((size_t)ncal*MOEI*4);
            dq_matmul(Xc,q1,gg,ncal,DIM,MOEI); dq_matmul(Xc,q3,uu,ncal,DIM,MOEI);
            for(size_t i=0;i<(size_t)ncal*MOEI;i++){ float g2=gg[i],u2=uu[i];
                if(SWLIM>0){ if(u2>SWLIM)u2=SWLIM; if(u2<-SWLIM)u2=-SWLIM; if(g2>SWLIM)g2=SWLIM; }
                hc[i]=dq_silu(g2)*u2; }
            free(gg);free(uu); nhc=ncal;
            __sync_fetch_and_add(&EXP_US_HC,exp_us_now()-_th);
        }
        if(vqcold&&vq_w2_dim()>0){   /* R28: 冷 w2 码本导出(dql D 段留洞, 载荷进 VQ 侧车槽) */
            double cvw=0;
            /* ★w2 顺序补偿(2026-08-03 用户令)★: y2ref = FP w2 过【量化态】hidden —
             * 与 signref μ 精修/coadapt y2ref 完全同口径。码本重构后每行输出匹配闭式缩放
             * 烘进行 scale(vq_qc.h vq_export_matrix_seq), 载荷格式/引擎逐字节不变。 */
            long long _t2=exp_us_now();
            float *t2=vq_export_matrix_seq(e2,DIM,MOEI,nhc?hc:NULL,nhc,vq_w2_dim(),vq_w2_nc(),
                                       w->vqfd,vq_slot_off(w->L,e,2,MOEI,DIM),&cvw,1);
            __sync_fetch_and_add(&EXP_US_VQ2,exp_us_now()-_t2);
            free(t2); w->vqsum[2]+=1; w->vqsum[3]+=cvw;
        } else
        dq_signref_export(e2,DIM,MOEI,hc,nhc,dDn,NULL);
        if(!w->bg){   /* 流式: 三段各按偏移落盘(布局与整块拼接逐字节一致); VQ 模式 G/U=稀疏洞 */
            if(!vqcold){
            if(pwrite(w->fd,dG,w->szG,(off_t)(w->off0+(size_t)e*w->szG))!=(ssize_t)w->szG) perror("exp-pw-g");
            if(pwrite(w->fd,dU,w->szG,(off_t)(w->off0+(size_t)NEXP*w->szG+(size_t)e*w->szG))!=(ssize_t)w->szG) perror("exp-pw-u");
            }
            if(!(vqcold&&vq_w2_dim()>0))
            if(pwrite(w->fd,dDn,w->szD,(off_t)(w->off0+2*(size_t)NEXP*w->szG+(size_t)e*w->szD))!=(ssize_t)w->szD) perror("exp-pw-d");
        }
        } else {
            /* 热专家 go2b(合并2bit, 激活最优) → 独立侧车文件; hc2 走 go2b q1/q3 顺序补偿(合并态口径) */
            size_t szG2=(size_t)MOEI*go2b_row_bytes(DIM), szD2=(size_t)DIM*go2b_row_bytes(MOEI);
            int vqhot=dq_vq_on();
            /* ★b1/b3/bD 只有 go2b 路要★(2026-08-27 OOM 定位): VQ 热路径下这三块从头到尾没被
             * 碰过, 却每个热专家白分配 ~6.3MB 再释放。配上 MALLOC_TRIM_THRESHOLD_=1GiB
             * (base86p 配方为压 mmap 锁争用而设, glibc 因此不把大块还给系统), 150 热专家/层
             * ≈ 1GiB/层的空转churn 全留在 20 个线程 arena 里, 堆到 L28 起挤掉 page cache,
             * HF 读盘越来越慢, L35 单层 113s(L00-L27 只要 5-15s), L36 被 OOM killer 干掉。
             * 平权跑从没暴露: hot=0 时这条热分支根本不执行。 */
            uint8_t *b1=NULL,*b3=NULL,*bD=NULL;
            if(!vqhot){ b1=malloc(szG2); b3=malloc(szG2); bD=malloc(szD2); }
            float *g1=malloc((size_t)MOEI*DIM*4),*g3=malloc((size_t)MOEI*DIM*4);
            if(vqhot){   /* v2.2: 热全三矩阵 → vq4x512 侧车 */
                double cv1=0,cv3=0;
                float *t1=vq_export_matrix(e1,MOEI,DIM,ncal?Xc:NULL,ncal,vq_hot_dim(),vq_hot_nc(),w->vqfd,vq_slot_off(w->L,e,0,MOEI,DIM),&cv1);
                float *t3=vq_export_matrix(e3,MOEI,DIM,ncal?Xc:NULL,ncal,vq_hot_dim(),vq_hot_nc(),w->vqfd,vq_slot_off(w->L,e,1,MOEI,DIM),&cv3);
                memcpy(g1,t1,(size_t)MOEI*DIM*4); memcpy(g3,t3,(size_t)MOEI*DIM*4); free(t1); free(t3);
                w->vqsum[0]+=2; w->vqsum[1]+=cv1+cv3;
            } else {
            dq_go2b_encode_adj(e1,MOEI,DIM,ncal?Xc:NULL,ncal,b1,g1,yadjE1);
            dq_go2b_encode_adj(e3,MOEI,DIM,ncal?Xc:NULL,ncal,b3,g3,yadjE3);
            }
            float *hc2=NULL; int nh2=0;
            if(ncal){
                hc2=malloc((size_t)ncal*MOEI*4);
                float *gg=malloc((size_t)ncal*MOEI*4),*uu=malloc((size_t)ncal*MOEI*4);
                dq_matmul(Xc,g1,gg,ncal,DIM,MOEI); dq_matmul(Xc,g3,uu,ncal,DIM,MOEI);
                for(size_t i=0;i<(size_t)ncal*MOEI;i++){ float g2v=gg[i],u2=uu[i];
                    if(SWLIM>0){ if(u2>SWLIM)u2=SWLIM; if(u2<-SWLIM)u2=-SWLIM; if(g2v>SWLIM)g2v=SWLIM; }
                    hc2[i]=dq_silu(g2v)*u2; }
                free(gg);free(uu); nh2=ncal;
            }
            float *gD=malloc((size_t)DIM*MOEI*4);
            if(vqhot){   /* 热 w2=vq4x512(实测序: go2b>vq4x512>signref; signref 版 2026-08-03 已否决) */
                double cv2=0;
                float *t2=vq_export_matrix(e2,DIM,MOEI,hc2,nh2,vq_hot_dim(),vq_hot_nc(),w->vqfd,vq_slot_off(w->L,e,2,MOEI,DIM),&cv2);
                memcpy(gD,t2,(size_t)DIM*MOEI*4); free(t2);
                w->vqsum[0]+=1; w->vqsum[1]+=cv2;
            } else {
            dq_go2b_encode(e2,DIM,MOEI,hc2,nh2,bD,gD);
            if(pwrite(w->g2fd,b1,szG2,(off_t)(w->g2off+(size_t)sl2*szG2))!=(ssize_t)szG2) perror("g2-pw-g");
            if(pwrite(w->g2fd,b3,szG2,(off_t)(w->g2off+(size_t)w->g2k*szG2+(size_t)sl2*szG2))!=(ssize_t)szG2) perror("g2-pw-u");
            if(pwrite(w->g2fd,bD,szD2,(off_t)(w->g2off+2*(size_t)w->g2k*szG2+(size_t)sl2*szD2))!=(ssize_t)szD2) perror("g2-pw-d");
            }
            if(ncal){   /* 单层门统计: w1/w2 输出重建 vs FP(校准行) */
                float *Ot=malloc((size_t)ncal*MOEI*4),*Oq=malloc((size_t)ncal*MOEI*4);
                dq_matmul(Xc,e1,Ot,ncal,DIM,MOEI); dq_matmul(Xc,g1,Oq,ncal,DIM,MOEI);
                double e2s=0,a2s=0,nu=0,d1n=0,d2n=0;
                for(size_t i=0;i<(size_t)ncal*MOEI;i++){ double d=(double)Ot[i]-Oq[i]; e2s+=d*d; a2s+=(double)Ot[i]*Ot[i];
                    nu+=(double)Ot[i]*Oq[i]; d1n+=(double)Ot[i]*Ot[i]; d2n+=(double)Oq[i]*Oq[i]; }
                w->g2sum[1]+=sqrt(e2s/(a2s+1e-30)); w->g2sum[2]+=nu/(sqrt(d1n)*sqrt(d2n)+1e-30);
                free(Ot);free(Oq);
                float *Ot2=malloc((size_t)ncal*DIM*4),*Oq2=malloc((size_t)ncal*DIM*4);
                dq_matmul(hc2,e2,Ot2,ncal,MOEI,DIM); dq_matmul(hc2,gD,Oq2,ncal,MOEI,DIM);
                e2s=0;a2s=0;nu=0;d1n=0;d2n=0;
                for(size_t i=0;i<(size_t)ncal*DIM;i++){ double d=(double)Ot2[i]-Oq2[i]; e2s+=d*d; a2s+=(double)Ot2[i]*Ot2[i];
                    nu+=(double)Ot2[i]*Oq2[i]; d1n+=(double)Ot2[i]*Ot2[i]; d2n+=(double)Oq2[i]*Oq2[i]; }
                w->g2sum[3]+=sqrt(e2s/(a2s+1e-30)); w->g2sum[4]+=nu/(sqrt(d1n)*sqrt(d2n)+1e-30);
                free(Ot2);free(Oq2);
                w->g2sum[0]+=1.0;
            }
            free(b1);free(b3);free(bD);free(g1);free(g3);free(gD); if(hc2)free(hc2);
        }
        if(hc)free(hc);
        if(q1)free(q1); if(q3)free(q3); if(yadjE1)free(yadjE1); if(yadjE3)free(yadjE3); free(e1);free(e3);free(e2);
    }
    if(tG)free(tG); if(tU)free(tU); if(tD)free(tD);
    free(Xc); return NULL;
}
/* 记录 paysz(预扫与写循环同源, 防偏移漂移) */
static uint64_t rec_paysz(const elrec_t*r,size_t szG,size_t szD){
    if(!strcmp(r->name,"1bit")) return (uint64_t)NEXP*(2*szG+szD);
    /* ★dyn8 全家族带 V8(2026-08-03 实锤: pc./lf./ls.GLdyn8 只匹配 "z.GLdyn8" 前缀 ⇒ V8 从不落盘
     * ⇒ 回放/zchain/引擎全丢主力动态)。V8 为该层共享 PCA 方向, 逐记录冗余落盘换四方同源。 */
    if(strstr(r->name,"GLdyn8")&&r->paylen==36&&STK8_V8) return 36+(uint64_t)8*DIM*2;
    if(!strcmp(r->name,"loss.cls")&&STK_CLSV) return (uint64_t)HCM*DIM*2;
    if(strstr(r->name,"zl.RRR")&&r->paylen==0&&ZLP_K>0)   /* z^L 正位直写(平行架构): 表位即注入位 */
        return 16+2*((uint64_t)ZLP_K+2*(uint64_t)ZLP_K*DIM);
    return r->paylen;
}
/* 单层量化文件导出(DQL2, 与表格一一对应): 权重字节(signref 同解)+ 全部元素记录与载荷 */
static void export_layer_file(int L,int S,int n_fit,const char*lf){

        /* ===== 单层量化文件 DQL2 —— 与表格一一对应(第一原则) =====
         * 'DQL2' u32 | L u32 | nrec u32
         * 每记录(=表格一行): name[16] algo[64] vol u64(表格体积列) paysz u64(实际载荷)
         *                    m1..m4 f32(实测数值) verdict i32(1正向落地 2正向未落地 3探索中 4生效)
         *                    payload[paysz](落地元素才有: 1bit=全部权重字节, z.*=参数, bwd.*=t) */
        size_t rbG=go1b_blk_row_bytes(DIM), szG=(size_t)MOEI*rbG;
        size_t rbD=go1b_blk_row_bytes(MOEI), szD=(size_t)DIM*rbD;
        /* ★流式导出★: 旧版三块整拼 buffer=855MB 瞬时尖峰(实测在 L20 把 footprint 顶过 12G 看门狗);
         * 改为先写记录头到 1bit 载荷处 → 扩文件 → worker 按偏移并行 pwrite(3.4MB/worker), 字节布局不变。 */
        /* ★--export-bytes 0(反修段)★: dql 不可变 — 跳过全部字节写(1bit/g2hot/dql 本体),
         * 只重建 op 侧车(probe1 实锤: 反修段曾整写 dql 918M→816M, 平行架构破洞)。 */
        int wb=(g_cli.export_bytes!=0);
        FILE*f=wb?fopen(lf,"wb"):NULL;
        if(wb&&!f){ perror("layerfile"); return; }
        int g2k=(GO2B_HOT&&L<64)?G2_K[L]:0;   /* 本层热专家数 */
        /* ★热载荷内嵌(DS4_MINVOL): 追加 "g2hot"(热 go2b, DQG2 侧车布局原样内嵌)记录 →
         * dql 单文件自持, 不再产独立 g2 侧车。偏移预算好, worker 与 1bit 同批填。 */
        int embK=0, nextra=0;
        size_t off_scan=12, off_g2_rec=0,off_g2=0;
        uint64_t g2_psz=0;
        size_t szG2=(size_t)MOEI*go2b_row_bytes(DIM), szD2=(size_t)DIM*go2b_row_bytes(MOEI);
        /* ★平行架构分流(2026-08-04 用户令)★: dql=纯权重字节(1bit+g2hot 内嵌), 量化后只读;
         * 其余全部记录(z 家族/zl.RRR/loss/bwd/探索账)→ op 侧车 dql_ops_LXX.bin(应用序=侧车序)。 */
        uint32_t nbyte=0;
        for(int i=0;i<NELE;i++) if(!strcmp(ELE[i].name,"1bit")){ nbyte++;
            off_scan+=DS4_AMP_REC_HDR+rec_paysz(&ELE[i],szG,szD); }
        if(g_cli.minvol&&g2k>0){
            embK=g2k;
            g2_psz=(uint64_t)g2_sidecar_hdr(embK)+2*(uint64_t)embK*szG2+(uint64_t)embK*szD2;
            off_g2_rec=off_scan; off_g2=off_g2_rec+DS4_AMP_REC_HDR; off_scan=off_g2+(size_t)g2_psz; nextra++;
        }
        double g2_mc=0;   /* 热门统计 mean_cos(内嵌 g2hot 记录头用) */
        uint32_t mg=0x324C5144,Lu=(uint32_t)L,nr=(uint32_t)(nbyte+nextra);
        if(wb){ fwrite(&mg,4,1,f);fwrite(&Lu,4,1,f);fwrite(&nr,4,1,f); }
        char opspath[512]; ops_sidecar_path(lf,L,opspath,sizeof(opspath));
        FILE*fo=fopen(opspath,"wb");
        if(!fo){ perror("ops-sidecar"); fclose(f); return; }
        { uint32_t mgo=OPSC_MAGIC,nro=(uint32_t)(NELE-nbyte);
          fwrite(&mgo,4,1,fo); fwrite(&Lu,4,1,fo); fwrite(&nro,4,1,fo); }
        for(int i=0;i<NELE;i++){ elrec_t*r=&ELE[i];
            uint64_t paysz=rec_paysz(r,szG,szD);
            int isbyte=!strcmp(r->name,"1bit");
            if(isbyte&&!wb) continue;              /* 反修段: dql 字节记录整条跳过(不可变) */
            FILE*W=isbyte?f:fo;                    /* 字节→dql; 其余→op 侧车 */
            fwrite(r->name,1,16,W); fwrite(r->algo,1,64,W);
            fwrite(&r->vol,8,1,W); fwrite(&paysz,8,1,W);
            fwrite(&r->m1,4,1,W);fwrite(&r->m2,4,1,W);fwrite(&r->m3,4,1,W);fwrite(&r->m4,4,1,W);
            fwrite(&r->verdict,4,1,W);
            if(paysz){
                if(!strcmp(r->name,"1bit")){
                    fflush(f); size_t off0=(size_t)ftello(f);
                    if(fseeko(f,(off_t)(off0+paysz-1),SEEK_SET)!=0||fputc(0,f)==EOF){ perror("exp-extend"); fclose(f); return; }
                    fflush(f);
                    /* go2b 热载荷: minvol=内嵌 dql g2hot 记录区(g2fd=dql fd); 否则独立侧车文件 */
                    int g2fd=-1; size_t g2off=0;
                    char g2path[512]={0};
                    int vqfd=-1; char vqpath[512]={0};
                    /* ★反修不重写 VQ 载荷(2026-07-31 用户令"多大就是多大, 重新设计量化脚本")★
                     * 反修只调 z/四损失/感知这些【侧车参数】, 路由专家的 VQ 权重一个 bit 都不改。
                     * 旧行为每次导出都 O_TRUNC 重编码 256 专家×3 矩阵, 两个恶果:
                     *   ①纯浪费: 反修每层白烧一遍全层 VQ 编码(反修慢一倍的主因)
                     *   ②★体积漂移★: 重编码时热数取自全局 G2_K(反修的 DS4_GO2B_HOT_TABLE=top64),
                     *     覆盖掉量化时按计划表定的 hot ⇒ 每层 728.7→820 MiB, 43 层超预算 11.7%,
                     *     而且直到合并才发现。热档 4×512 比冷档密, 热数一涨体积就涨。
                     * 现在: 侧车已在且头合法(magic/L/nexp 对) ⇒ 保留其字节, 只更新 dql 记录区。
                     * 量化首次导出(侧车不存在)仍走完整编码路径, 语义不变。 */
                    int vq_keep=0;
                    if(dq_vq_on()){
                        vq_sidecar_path(lf,L,vqpath,sizeof(vqpath));
                        if(g_cli.bwd){
                            int pf=open(vqpath,O_RDONLY);
                            if(pf>=0){ uint32_t h4[4]={0};
                                if(read(pf,h4,16)==16&&h4[0]==VQSC_MAGIC&&h4[2]==(uint32_t)L&&h4[3]==256){
                                    struct stat vs;
                                    if(fstat(pf,&vs)==0&&(size_t)vs.st_size>=vq_hdr_bytes()){
                                        vq_keep=1;
                                        fprintf(stderr,"[层文件] L%02d 反修保留既有 VQ 载荷 %.1fMiB(权重未变, 不重编码)\n",
                                                L,vs.st_size/1048576.0);
                                    } }
                                close(pf); }
                        }
                    }
                    if(dq_vq_on()&&!vq_keep){
                        vqfd=open(vqpath,O_RDWR|O_CREAT|O_TRUNC,0644);
                        if(vqfd<0){ perror("vq-sidecar-open"); fclose(f); return; }
                        size_t hb2=vq_hdr_bytes(); uint8_t *hb=calloc(1,hb2);
                        uint32_t mgv=VQSC_MAGIC,verv=1,Lv=(uint32_t)L,nev=256;
                        memcpy(hb,&mgv,4); memcpy(hb+4,&verv,4); memcpy(hb+8,&Lv,4); memcpy(hb+12,&nev,4);
                        for(int e2=0;e2<256;e2++) for(int wh=0;wh<3;wh++){
                            uint64_t o64=(uint64_t)vq_slot_off(L,e2,wh,MOEI,DIM);   /* w2 槽随 vq_slot_off 的 w2 总闸(w2dim=0 ⇒ 0) */
                            memcpy(hb+16+((size_t)e2*3+wh)*8,&o64,8); }
                        if(pwrite(vqfd,hb,hb2,0)!=(ssize_t)hb2) perror("vq-hdr");
                        free(hb);
                        size_t vend=vq_total_bytes(L,MOEI,DIM); uint8_t z2=0;
                        if(pwrite(vqfd,&z2,1,(off_t)(vend-1))!=1){ perror("vq-extend"); close(vqfd); fclose(f); return; }
                        fprintf(stderr,"[层文件] L%02d VQ 侧车预留 %.1fMiB → %s\n",L,vend/1048576.0,vqpath);
                    }
                    if(g2k>0&&!dq_vq_on()){
                        if(embK>0){   /* 内嵌: 热载荷直写 dql 的 g2hot 记录区(DQG2 头由循环后主线程写) */
                            g2fd=fileno(f); g2off=off_g2+g2_sidecar_hdr(g2k);
                        } else {
                        g2_sidecar_path(lf,L,g2path,sizeof(g2path));
                        g2fd=open(g2path,O_RDWR|O_CREAT|O_TRUNC,0644);
                        if(g2fd<0){ perror("g2-sidecar-open"); fclose(f); return; }
                        size_t hdr2=g2_sidecar_hdr(g2k);
                        uint8_t*hb=calloc(1,hdr2);
                        uint32_t mg2=G2SC_MAGIC,ver=1,Lu2=(uint32_t)L,khu=(uint32_t)g2k;
                        memcpy(hb,&mg2,4); memcpy(hb+4,&ver,4); memcpy(hb+8,&Lu2,4); memcpy(hb+12,&khu,4);
                        for(int i2=0;i2<g2k;i2++){ uint16_t idv=G2_IDS[L][i2]; memcpy(hb+24+2*(size_t)i2,&idv,2); }
                        if(pwrite(g2fd,hb,hdr2,0)!=(ssize_t)hdr2) perror("g2-hdr");
                        free(hb);
                        g2off=hdr2;
                        size_t g2end=hdr2+2*(size_t)g2k*szG2+(size_t)g2k*szD2;
                        uint8_t z=0;
                        if(pwrite(g2fd,&z,1,(off_t)(g2end-1))!=1){ perror("g2-extend"); close(g2fd); fclose(f); return; }
                        }
                    }
                    fprintf(stderr,"[层文件] L%02d signref 权重字节生成(μ=%.0f, 流式%s) ",L,dq_signref_mu,
                            g2k>0?"+go2b热侧车":"");
                    int e_next=0,nth=NTHREADS>NEXP?NEXP:NTHREADS,fd=fileno(f);
                    expw_t *ws=calloc((size_t)nth,sizeof(expw_t)); pthread_t *th=malloc((size_t)nth*sizeof(pthread_t));
                    for(int t=0;t<nth;t++){ ws[t]=(expw_t){L,S,n_fit,&e_next,NULL,NULL,NULL,szG,szD,fd,off0,g2fd,g2off,g2k,{0},vqfd,{0}};
                        pthread_create(&th[t],NULL,export_worker,&ws[t]); }
                    for(int t=0;t<nth;t++) pthread_join(th[t],NULL);
                    fprintf(stderr,"[EXPORT_PROF] read=%.0fs vq13=%.0fs hc=%.0fs vq2=%.0fs (核秒累计)\n",
                            (double)EXP_US_READ/1e6,(double)EXP_US_VQ13/1e6,(double)EXP_US_HC/1e6,(double)EXP_US_VQ2/1e6);
                    EXP_US_READ=EXP_US_VQ13=EXP_US_HC=EXP_US_VQ2=0;
                    if(dq_vq_on()&&!vq_keep){   /* ★VQ_GATE 每层日志(用户令: 每层量化必须有日志)★ */
                        double nh=0,ch=0,ncd=0,cc=0;
                        for(int t=0;t<nth;t++){ nh+=ws[t].vqsum[0]; ch+=ws[t].vqsum[1];
                                                ncd+=ws[t].vqsum[2]; cc+=ws[t].vqsum[3]; }
                        double mh=nh>0?ch/nh:0, mc2=ncd>0?cc/ncd:0;
                        fprintf(stderr,"\nVQ_GATE L=%02d 热矩阵=%d 冷w1w3=%d 均值cos: hot=%.4f cold=%.4f %s → %s",
                                L,(int)nh,(int)ncd,mh,mc2,
                                (mh>=0.90&&mc2>=0.75)?"PASS":"★WARN(阈 hot0.90/cold0.75)★",vqpath);
                        close(vqfd); vqfd=-1;
                    }
                    if(g2k>0&&!dq_vq_on()){   /* 单层重建门(runbook §3.1): 均值 cos≥0.85; 内嵌时进 g2hot 记录头, 独立侧车回填 */
                        double n2=0,r1=0,c1=0,r2d=0,c2d=0;
                        for(int t=0;t<nth;t++){ n2+=ws[t].g2sum[0]; r1+=ws[t].g2sum[1]; c1+=ws[t].g2sum[2];
                            r2d+=ws[t].g2sum[3]; c2d+=ws[t].g2sum[4]; }
                        float mc=(n2>0)?(float)((c1+c2d)/(2.0*n2)):0.0f;
                        if(embK>0) g2_mc=mc;   /* 内嵌: fd 是 dql 本体, 不回填不 close */
                        else { if(pwrite(g2fd,&mc,4,16)!=4) perror("g2-m1"); close(g2fd); }
                        fprintf(stderr,"\nGO2B_GATE L=%02d 热=%d/%d w1[rel=%.4f cos=%.4f] w2[rel=%.4f cos=%.4f] 均值cos=%.4f %s → %s",
                                L,(int)n2,g2k,n2>0?r1/n2:0,n2>0?c1/n2:0,n2>0?r2d/n2:0,n2>0?c2d/n2:0,mc,
                                mc>=0.85f?"PASS":"★FAIL(<0.85)★",embK>0?"(内嵌dql)":g2path);
                    }
                    free(ws);free(th); fputc('\n',stderr);
                    if(fseeko(f,(off_t)(off0+paysz),SEEK_SET)!=0){ perror("exp-seek"); fclose(f); return; }
                } else if(strstr(r->name,"GLdyn8")&&paysz>36){
                    fwrite(r->pay,1,36,W);
                    uint16_t *h=malloc((size_t)8*DIM*2);
                    for(size_t j=0;j<(size_t)8*DIM;j++) h[j]=go1b_fp32_to_fp16(STK8_V8[j]);
                    fwrite(h,2,(size_t)8*DIM,W); free(h);
                } else if(strstr(r->name,"zl.RRR")&&r->paylen==0&&ZLP_K>0){
                    /* ★z^L 正位直写(取代旧 export后append)★: 表位=评估注入位, 错序温床根除 */
                    uint32_t zk=(uint32_t)ZLP_K,din=(uint32_t)DIM,dout=(uint32_t)DIM;
                    fwrite(&zk,4,1,W); fwrite(&ZLP_TR,4,1,W); fwrite(&din,4,1,W); fwrite(&dout,4,1,W);
                    size_t nh=(size_t)zk+(size_t)zk*din+(size_t)zk*dout;
                    uint16_t*h=malloc(nh*2); size_t o2=0;
                    for(uint32_t i2=0;i2<zk;i2++) h[o2++]=go1b_fp32_to_fp16(ZLP_Z[i2]);
                    for(size_t i2=0;i2<(size_t)dout*zk;i2++) h[o2++]=go1b_fp32_to_fp16(ZLP_U[i2]);
                    for(size_t i2=0;i2<(size_t)din*zk;i2++)  h[o2++]=go1b_fp32_to_fp16(ZLP_V[i2]);
                    fwrite(h,2,nh,W); free(h);
                } else if(!strcmp(r->name,"loss.cls")){
                    uint16_t *h=malloc((size_t)HCM*DIM*2);
                    for(size_t j=0;j<(size_t)HCM*DIM;j++) h[j]=go1b_fp32_to_fp16(STK_CLSV[j]);
                    fwrite(h,2,(size_t)HCM*DIM,W); free(h);
                } else fwrite(r->pay,1,(size_t)r->paylen,W);
            }
            printf("DQL2REC L=%d %-12s vol=%llu paysz=%llu m=[%.4f %.4f %.1f %.2f] 判定=%d | %s\n",
                   L,r->name,(unsigned long long)r->vol,(unsigned long long)paysz,
                   r->m1,r->m2,r->m3,r->m4,r->verdict,r->algo);
        }
        char nm16[16],al64[64]; uint64_t v8; float m1,m2,m3,m4=0; int vd=4;
        if(embK>0&&wb){   /* 追加内嵌记录 "g2hot"(DQG2 布局); 反修段 dql 不可变=跳过 */
            if(fseeko(f,(off_t)off_g2_rec,SEEK_SET)!=0) perror("g2-rec-seek");
            memset(nm16,0,16); memset(al64,0,64);
            snprintf(nm16,16,"g2hot"); snprintf(al64,64,"热%d go2b 内嵌(DQG2布局) mean_cos=%.4f",embK,g2_mc);
            v8=g2_psz; m1=(float)embK; m2=(float)g2_mc; m3=0;
            fwrite(nm16,1,16,f); fwrite(al64,1,64,f); fwrite(&v8,8,1,f); fwrite(&v8,8,1,f);
            fwrite(&m1,4,1,f);fwrite(&m2,4,1,f);fwrite(&m3,4,1,f);fwrite(&m4,4,1,f); fwrite(&vd,4,1,f);
            { size_t hdr2=g2_sidecar_hdr(embK); uint8_t*hb=calloc(1,hdr2);
              uint32_t mg2=G2SC_MAGIC,ver=1,Lu2=(uint32_t)L,khu=(uint32_t)embK; float mcf=(float)g2_mc;
              memcpy(hb,&mg2,4); memcpy(hb+4,&ver,4); memcpy(hb+8,&Lu2,4); memcpy(hb+12,&khu,4);
              memcpy(hb+16,&mcf,4);
              for(int i2=0;i2<embK;i2++){ uint16_t idv=G2_IDS[L][i2]; memcpy(hb+24+2*(size_t)i2,&idv,2); }
              fwrite(hb,1,hdr2,f); free(hb); }
            printf("DQL2REC L=%d %-12s vol=%llu paysz=%llu m=[%.4f %.4f %.1f %.2f] 判定=%d | %s\n",
                   L,"g2hot",(unsigned long long)v8,(unsigned long long)v8,m1,m2,m3,m4,vd,al64);
        }
        if(embK>0&&wb){
            fflush(f);
            if(ftruncate(fileno(f),(off_t)off_scan)!=0) perror("dql-trunc");   /* 尾洞(热槽/截断)补齐文件长 */
        }
        long fsz=0;
        if(wb){ fsz=(embK>0)?(long)off_scan:ftell(f); fclose(f); }
        else { struct stat dst2; fsz=stat(lf,&dst2)==0?(long)dst2.st_size:0; }   /* 反修段: dql 未动, 报现值 */
        long osz=ftell(fo); fclose(fo);
        if(ZLP_K>0&&ZLP_L==L){   /* z^L 已正位直写侧车 → 清暂存(接管旧 append 块职责) */
            mlog(L,"z^L","zl.RRR 正位直写(op 侧车)","已落地",
                 16+2*((uint64_t)ZLP_K+2*(uint64_t)ZLP_K*DIM),"平行架构: 表位=注入位","✓正向落地");
            free(ZLP_U); free(ZLP_V); free(ZLP_Z); ZLP_U=ZLP_V=ZLP_Z=NULL; ZLP_K=0; ZLP_L=-1;
        }
        printf("LAYERFILE %s L=%d 字节记录=%u bytes=%ld (%.2f MiB) | ops侧车 %s 记录=%u bytes=%ld\n",
               lf,L,nbyte+(uint32_t)nextra,fsz,(double)fsz/1048576.0,
               opspath,(uint32_t)(NELE-nbyte),osz);
        fflush(stdout);
}
