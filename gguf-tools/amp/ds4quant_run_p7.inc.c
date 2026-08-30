/* ★逐阶段计时(2026-08-27)★ 反修单层 48-51 秒墙钟, 而 bytes_moe 前向摊到墙钟只有 ~2 秒
 * (那个 [bmwt] 数是 20 线程累计, 除以 20 才是墙钟 —— 我误读过一次)。perf 被内核
 * perf_event_paranoid 挡、gdb 无符号, 只能自己埋点。四段: attn(含 hc_pre/post) /
 * 路由 / 专家前向 / ZLGATE 候选评估。恒开(一层一行, 不新增 env)。 */
double g_lt[8];   /* 0=attn 1=路由 2=共享专家 3=bytes_moe#1 4=段内其余 5=zlgate 6=lfile_load 7=bytes_moe#2 */
static void layer_fwd(int L, LW*W, float*H, const long*ids, int S, int n_fit,
                      int do_quant, char cfg, lstat_t*st){
    double lt_t0=vqt_now(), lt_mark=lt_t0;
    for(int i=0;i<8;i++) g_lt[i]=0;
    int mixd=2*HCM+HCM*HCM;
    /* ★zrec 存在性提前判定(2026-08-28 前移)★ 原在函数中段(z 解算块之前), 但 bf_fp_routed
     * 这个【生产者】比它更早执行, 拿不到 zrec_done ⇒ 守卫无法与消费者对齐, 见下。 */
    int zrec_done=0;
    {
        const char*ld2=getenv("DS4_LAYER_DIR");
        if(ld2){ char zp2[1024]; snprintf(zp2,sizeof zp2,"%s/zrec_L%02d.bin",ld2,L);
                 FILE*zf2=fopen(zp2,"rb"); if(zf2){ fclose(zf2); zrec_done=1; } }
    }
    float *cosr=malloc((size_t)S*(RD/2)*4),*sinr=malloc((size_t)S*(RD/2)*4);
    if(CR[L]>0) dq_freqs_cis(RD,S,65536.0,160000.0,16.0,32.0,1.0,cosr,sinr);
    else dq_freqs_cis(RD,S,0.0,10000.0,16.0,32.0,1.0,cosr,sinr);
    float *y=malloc((size_t)S*DIM*4),*post=malloc((size_t)S*HCM*4),*comb=malloc((size_t)S*HCM*HCM*4);
    double _am=vqt_now();   /* attn 子账(BFLT 二级): hc/comp/核 三桶, 残差=freqs/malloc/捕获 */
    dq_hc_pre(H,W->afn,W->asc,W->abase,y,post,comb,S,HCM,DIM,mixd,HCIT,EPSF,EPSF);
    float *xn=malloc((size_t)S*DIM*4); for(int s=0;s<S;s++)dq_rms(y+(size_t)s*DIM,W->an,xn+(size_t)s*DIM,DIM,EPSF);
    g_bflt[12]+=vqt_now()-_am; _am=vqt_now();
    int Sc=0; float *kvc=NULL;
    if(CR[L]>0){ kvc=malloc((size_t)((S/CR[L]+2)*2)*HD*4); Sc=dq_compressor(xn,W->cwkv,W->cwgate,W->cnorm,W->cape,cosr,sinr,kvc,S,DIM,HD,RD,CR[L],EPSF); }
    g_bflt[13]+=vqt_now()-_am; _am=vqt_now();
    float *a=malloc((size_t)S*DIM*4);
    dq_attention(xn,W->wqa,W->qn,W->wqb,W->wkv,W->kvn,W->sink,W->woa,W->wob,kvc,cosr,sinr,a,S,DIM,NH,HD,RD,QLR,OLR,OG,WIN,Sc,CR[L],EPSF);
    g_bflt[14]+=vqt_now()-_am; _am=vqt_now();
    float *H2=malloc((size_t)S*HCM*DIM*4); dq_hc_post(a,H,post,comb,H2,S,HCM,DIM);
    float *y2=malloc((size_t)S*DIM*4),*post2=malloc((size_t)S*HCM*4),*comb2=malloc((size_t)S*HCM*HCM*4);
    dq_hc_pre(H2,W->ffn,W->fsc,W->fbase,y2,post2,comb2,S,HCM,DIM,mixd,HCIT,EPSF,EPSF);
    float *Fin=malloc((size_t)S*DIM*4); for(int s=0;s<S;s++)dq_rms(y2+(size_t)s*DIM,W->fn,Fin+(size_t)s*DIM,DIM,EPSF);
    g_bflt[12]+=vqt_now()-_am;
    if(GS_CAP_L==L&&GS_FIN) memcpy(GS_FIN,Fin,(size_t)S*DIM*4);  if(g_xcap_out) xcap_dump_fin(L,Fin,S);   /* 反修取料 + 量化链 x 捕获 */
    g_lt[0]=vqt_now()-lt_mark; lt_mark=vqt_now();   /* ①attn+hc 段完 */
    /* moe 路由(实际激活: 量化遍即被污染激活 = 部署运行时口径) */
    int *idx=malloc((size_t)S*NACT_RT*sizeof(int)); float *rw=malloc((size_t)S*NACT_RT*4);
    /* ★锚 override 在场时 gate 分数是白算(2026-08-30 BFLT 实测路由 16s/单元)★: 下方
     * ANCHOR_ROUTE/BF_ANCROUTE 块会把 idx/rw【整个】覆盖, hash 路由的 gate GEMM+softmax
     * 输出无人消费。跳过条件与覆盖条件逐字相同 ⇒ 覆盖后字节不变。NACT_RT>NACT(rroute
     * 加宽)时尾槽无人覆盖, 不跳(保原行为)。只跳 hash 分支: score 分支带 RB Δb 采集寄生。 */
    const int rt_ov=do_quant&&ANC_OK&&(getenv("DS4_ANCHOR_ROUTE")||BF_ANCROUTE)&&NACT_RT==NACT;
    if(W->t2ei){ static int hbn=0;
        if(!hbn&&do_quant){ hbn=1;
            fprintf(stderr,"[路由] tid2eid 哈希路由生效(0731 原生): 专家选择=token 哈希表, 结构性零漂移;\n"
                           "[路由] Δb 选择偏置在此无对象(老 base 分数近似路由时代的机制), RB 族不武装。\n"); }
        if(!rt_ov) dq_gate_route_hash(Fin,W->gate,W->t2ei,ids,idx,rw,S,DIM,NEXP,NACT,ROUTE_SCALE);
    }
    else{
        if(!RB_TRIED&&getenv("DS4_ROUTE_BIAS")){   /* 懒加载路由偏置侧车(一次) */
            RB_TRIED=1; FILE*rf=fopen(getenv("DS4_ROUTE_BIAS"),"rb");
            if(rf){ uint32_t hd[4]={0,0,0,0};
                /* 文件层数≥当前 NL 即可(取前 NL 行): NL=1 探针消费 43 层侧车(2026-08-03) */
                if(fread(hd,4,4,rf)==4&&hd[0]==0x41494252u&&hd[1]>=(uint32_t)NLAYERS&&hd[2]==(uint32_t)NEXP){
                    uint32_t nlf=hd[1];
                    RB_APPLY=malloc((size_t)nlf*NEXP*4);
                    uint32_t*c=malloc((size_t)nlf*NEXP*4);
                    if(fread(RB_APPLY,4,(size_t)nlf*NEXP,rf)==(size_t)nlf*NEXP&&
                       fread(c,4,(size_t)nlf*NEXP,rf)==(size_t)nlf*NEXP){
                        if(getenv("DS4_ROUTE_BIAS_ALPHA")) RB_ALPHA=atof(getenv("DS4_ROUTE_BIAS_ALPHA"));
                        if(getenv("DS4_ROUTE_BIAS_MINCNT")) RB_MINCNT=atoi(getenv("DS4_ROUTE_BIAS_MINCNT"));
                        long armed=0;
                        for(size_t i=0;i<(size_t)NLAYERS*NEXP;i++){
                            if((int)c[i]<RB_MINCNT) RB_APPLY[i]=0.0f; else if(RB_APPLY[i]!=0.0f) armed++; }
                        printf("ROUTE_BIAS apply α=%.2f mincnt=%d 武装槽=%ld\n",RB_ALPHA,RB_MINCNT,armed);
                        if(getenv("DS4_ROUTE_SEQ")){   /* ★per-layer α 装载(2026-08-14): 读 <bias>.alpha.txt */
                            char ap2[512]; snprintf(ap2,sizeof(ap2),"%s.alpha.txt",getenv("DS4_ROUTE_BIAS"));
                            FILE*af=fopen(ap2,"r");
                            if(af){ int l2; float a2; int na=0;
                                while(fscanf(af,"L=%d a=%f\n",&l2,&a2)==2){ if(l2>=0&&l2<64){ RB_ALPHA_L[l2]=a2; if(a2>0)na++; } }
                                fclose(af); RB_SEQ=1;
                                printf("ROUTE_BIAS per-layer α 装载: 开启层=%d\n",na); } }
                    } else { free(RB_APPLY); RB_APPLY=NULL; }
                    free(c);
                }
                fclose(rf);
            }
        }
        const float *gb=W->gbias; float *gb2=NULL;
        if(RB_APPLY){ float al = RB_SEQ ? RB_ALPHA_L[L] : RB_ALPHA;   /* 序贯: per-layer 动态 α(0=层关) */
            if(al!=0.0f){ gb2=malloc((size_t)NEXP*4);
                for(int e=0;e<NEXP;e++) gb2[e]=(W->gbias?W->gbias[e]:0.0f)+al*RB_APPLY[(size_t)L*NEXP+e];
                gb=gb2; } }
        /* ★token门控路由偏置(2026-08-14 用户令"路由问题没有解决"·静态Δb中位KL2.4×判死后唯一剂型)★:
         * DS4_ROUTE_GATE_TAU>0 时 Δb 只施于路由不确定的 token — 无偏置选择的 max(rw)<τ
         * (权重平坦=选择摇摆=漂移高发区); 置信 token 保持原路由(保分布保真)。τ=0/未设=旧全量行为。 */
        { float gtau=getenv("DS4_ROUTE_GATE_TAU")?atof(getenv("DS4_ROUTE_GATE_TAU")):0.0f;
          if(gb2&&gtau>0.0f){
            dq_gate_route_topk(Fin,W->gate,W->gbias,idx,rw,S,DIM,NEXP,NACT_RT,ROUTE_SCALE);   /* 无偏置基线 */
            int *idxb=malloc((size_t)S*NACT_RT*sizeof(int)); float *rwb=malloc((size_t)S*NACT_RT*4);
            dq_gate_route_topk(Fin,W->gate,gb,idxb,rwb,S,DIM,NEXP,NACT_RT,ROUTE_SCALE);       /* 带偏置 */
            long ngate=0;
            for(int s2=0;s2<S;s2++){
                float mx=0.0f; for(int a2=0;a2<NACT_RT;a2++){ float v=rw[(size_t)s2*NACT_RT+a2]; if(v>mx)mx=v; }
                if(mx<gtau){ memcpy(idx+(size_t)s2*NACT_RT,idxb+(size_t)s2*NACT_RT,NACT_RT*sizeof(int));
                             memcpy(rw+(size_t)s2*NACT_RT,rwb+(size_t)s2*NACT_RT,NACT_RT*4); ngate++; }
            }
            free(idxb); free(rwb);
            static int gt_note=0;
            if(!gt_note){ gt_note=1; fprintf(stderr,"[路由门] τ=%.2f 首层门通过率=%.1f%%\n",gtau,100.0*ngate/S); }
          } else
        if(RR_EPS>0.0f && NACT_RT>NACT){
            /* ★加法式并集(2026-08-16): 前NACT名 id+权重=原协议逐位(6内归一不稀释);
             * 额外槽取第7/8名, 权重= eps × (∝sc 相对权) 加法进场 — m1乘法归一版
             * 已死于分数压缩变换(第7名抢走~10%质量), 本式把稀释归零。 */
            dq_gate_route_topk(Fin,W->gate,gb,idx,rw,S,DIM,NEXP,NACT_RT,ROUTE_SCALE);
            int *idx6=malloc((size_t)S*NACT*sizeof(int)); float *rw6=malloc((size_t)S*NACT*4);
            dq_gate_route_topk(Fin,W->gate,gb,idx6,rw6,S,DIM,NEXP,NACT,ROUTE_SCALE);
            for(int s2=0;s2<S;s2++){
                for(int a=NACT;a<NACT_RT;a++) rw[(size_t)s2*NACT_RT+a] *= RR_EPS;
                memcpy(idx+(size_t)s2*NACT_RT, idx6+(size_t)s2*NACT, NACT*sizeof(int));
                memcpy(rw +(size_t)s2*NACT_RT, rw6 +(size_t)s2*NACT, NACT*4);
            }
            free(idx6); free(rw6);
        } else
        dq_gate_route_topk(Fin,W->gate,gb,idx,rw,S,DIM,NEXP,NACT_RT,ROUTE_SCALE);
        }
        if(gb2) free(gb2);
        /* ★锚路由与 Δb 统计不互斥(2026-08-01 解除)★ 原条件带 !DS4_ANCHOR_ROUTE && !BF_ANCROUTE,
         * 但这是多余的自设限制: 本段统计只需要 ① 学生 gate 分数 scv(下面自己按 W->gate 算,
         * 与前向走哪条路由无关)② 学生 top-k idx(2628 行 dq_gate_route_topk 刚算出)③ FP 锚
         * top-k fpx。而锚路由的覆盖(下面 2652 段 idx=aidx)发生在本段【之后】—— 统计读到的
         * idx 仍是学生的。解除后可以同时拿到"前向 100% 走 FP 锚路由"(消除路由漂移这个误差源,
         * 量化误差纯净)和"Δb 仍被收集"(合并时烘焙, 让部署态学生路由逼近锚)。
         * R28 教训: 全程学生路由 ⇒ 路由一致率掉到 82.9%, 量化误差与路由漂移误差混在一起无法分离。*/
        if((RB_SEQ||getenv("DS4_ROUTE_BIAS_FIT"))&&do_quant&&ANC_OK
           &&(g_rb_fit_L<0||g_rb_fit_L==L)&&!(RB_SEQ&&g_rb_fit_L<0)){   /* 序贯: 只在指定层的 FIT 前向统计 */
            if(!RB_ACC){ RB_ACC=calloc((size_t)NLAYERS*NEXP,4); RB_CNT=calloc((size_t)NLAYERS*NEXP,4);
                fprintf(stderr,"[路由] Δb 统计首次武装 L=%d cfg前向\n",L); }
            const int32_t *fpx=ANC.ridx+(size_t)L*(g_anc_rowmap?g_anc_rowstride:S)*NACT;
            float *scv=malloc((size_t)NEXP*4);
            for(int s=0;s<S;s++){
                const float *xr=Fin+(size_t)s*DIM;
                for(int e=0;e<NEXP;e++){ const float*gr=W->gate+(size_t)e*DIM;
                    float raw=0.0f; for(int k=0;k<DIM;k++) raw+=xr[k]*gr[k];
                    scv[e]=sqrtf(log1pf(expf(raw)))+(W->gbias?W->gbias[e]:0.0f); }
                float thr=1e30f;
                for(int a=0;a<NACT;a++){ float v=scv[idx[(size_t)s*NACT+a]]; if(v<thr)thr=v; }
                const size_t frow=(size_t)(g_anc_rowmap?g_anc_rowmap[s]:s);
                for(int a=0;a<NACT;a++){ int e=fpx[frow*NACT+a]; int in=0;
                    for(int b=0;b<NACT;b++) if(idx[(size_t)s*NACT+b]==e){in=1;break;}
                    if(!in){ RB_ACC[(size_t)L*NEXP+e]+=thr-scv[e]; RB_CNT[(size_t)L*NEXP+e]++; } }
                for(int a=0;a<NACT;a++){ int e=idx[(size_t)s*NACT+a]; int in=0;
                    for(int b=0;b<NACT;b++) if(fpx[frow*NACT+b]==e){in=1;break;}
                    if(!in){ RB_ACC[(size_t)L*NEXP+e]-=scv[e]-thr; RB_CNT[(size_t)L*NEXP+e]++; } }
            }
            free(scv);
        }
    }
    if(do_quant&&ANC_OK&&(getenv("DS4_ANCHOR_ROUTE")||BF_ANCROUTE)){   /* 神谕路由归因: 强制 FP 路由(专家选择+权重),
        隔离"路由漂移"对最终质量的贡献; 反修判据也用它禁翻转噪声 */
        /* g_anc_rowmap 非空 = 反修粗筛的抽格前向: 锚按【原始行号】取, stride 用原始 S。
         * 为空则恒等映射, 与改动前逐字节一致。 */
        const int astride = g_anc_rowmap ? g_anc_rowstride : S;
        const int32_t *abase = ANC.ridx+(size_t)L*astride*NACT;
        const float   *rbase = ANC.rw  +(size_t)L*astride*NACT;
        for(int s=0;s<S;s++){
            const int src = g_anc_rowmap ? g_anc_rowmap[s] : s;
            for(int a=0;a<NACT;a++) idx[(size_t)s*NACT+a]=abase[(size_t)src*NACT+a];
            memcpy(rw+(size_t)s*NACT, rbase+(size_t)src*NACT, (size_t)NACT*4);
        }
    }
    if(GS_CAP_L==L&&GS_IDXC&&GS_RWC){   /* ★反修 GE 投影: 捕获目标层实际路由(命中+权重)★ */
        memcpy(GS_IDXC,idx,(size_t)S*NACT*sizeof(int));
        memcpy(GS_RWC,rw,(size_t)S*NACT*4);
    }
    if(g_chfd>=0&&S==g_chS){   /* ★链态锚捕获: 部署链 fin/路由 直写盘★ */
        size_t bf=40, br=bf+(size_t)NLAYERS*S*DIM*4, bw=br+(size_t)NLAYERS*S*NACT*4;
        pwrite(g_chfd,Fin,(size_t)S*DIM*4,(off_t)(bf+(size_t)L*S*DIM*4));
        int32_t*i32=malloc((size_t)S*NACT*4);
        for(size_t i=0;i<(size_t)S*NACT;i++) i32[i]=(int32_t)idx[i];
        pwrite(g_chfd,i32,(size_t)S*NACT*4,(off_t)(br+(size_t)L*S*NACT*4)); free(i32);
        pwrite(g_chfd,rw,(size_t)S*NACT*4,(off_t)(bw+(size_t)L*S*NACT*4));
        fdatasync(g_chfd);   /* 同内存卫生 */
    }
    if(ANC_BUILD){   /* 锚定捕获: FP 激活/路由 */
        memcpy(ANC.fin+(size_t)L*S*DIM, Fin, (size_t)S*DIM*4);
        for(size_t i=0;i<(size_t)S*NACT;i++) ANC.ridx[(size_t)L*S*NACT+i]=(int32_t)idx[i];
        memcpy(ANC.rw+(size_t)L*S*NACT, rw, (size_t)S*NACT*4);
        /* FP 锚路 X 捕获(2026-07-28 g4c 流形针): dql 层文件被 merge 消耗后学生态 B 回放
         * 不可得 → FP 态 ffn_in 口径(与 DS4_BF_DUMPXR 学生态口径区分, 探针内部自洽即可)。 */
        { const char*xe=getenv("DS4_BF_DUMPXR");
          if(xe){ const char*q=xe; int hit=0;
              while(*q){ if(atoi(q)==L){hit=1;break;} while(*q&&*q!=',')q++; if(*q==',')q++; }
              if(hit){ char px[64]; snprintf(px,sizeof(px),"/tmp/xr_x_L%02d.npy",L);
                  xr_npy(px,Fin,S,DIM);
                  fprintf(stderr,"[XR-FP] L%d X 已捕获: %s (S=%d)\n",L,px,S); } } }
    }
    if(st&&do_quant&&ANC_OK){   /* 路由一致率 vs FP 锚定 (路由漂移=MoE 误差放大器) */
        const int32_t *aidx=ANC.ridx+(size_t)L*S*NACT; long match=0;
        for(int s=0;s<S;s++)for(int a2=0;a2<NACT;a2++){ int e=idx[(size_t)s*NACT+a2];
            for(int b=0;b<NACT;b++) if(aidx[(size_t)s*NACT+b]==e){match++;break;} }
        st->agree=100.0*(double)match/((double)S*NACT);
    }
    /* shared 专家 FP 为公共基 */
    g_lt[1]=vqt_now()-lt_mark; lt_mark=vqt_now();   /* ①路由段完 */
    float *Fout=calloc((size_t)S*DIM,4); dq_expert_fp(Fin,W->s1,W->s3,W->s2,NULL,Fout,S,DIM,MOEI,SWLIM);
    g_lt[2]=vqt_now()-lt_mark; lt_mark=vqt_now();   /* ②共享专家(全 S 稠密前向)完 */
    if(do_quant&&cfg=='B'){
        /* 字节重前向: 从层文件(权重字节+落地修正链)执行, 不重量化 — 全局回扫的引擎 */
        float *shb=malloc((size_t)S*DIM*4); memcpy(shb,Fout,(size_t)S*DIM*4);   /* shared 基 */
        if(GS_LF&&L<NL&&GS_LF[L].map){   /* ★存档缓存: 不重 mmap/不重解析★ */
            g_replay_cur_L=L; { double _b=vqt_now(); bytes_moe(&GS_LF[L],S,Fin,idx,rw,Fout); g_lt[3]+=vqt_now()-_b; }
            if(GS_GV&&GS_GV_L==L){        /* ★反修: 目标层 per-token routed 增益(搜每 token 最优乘子)★ */
                for(int s=0;s<S;s++){ float g=GS_GV[s]; float*fw=Fout+(size_t)s*DIM,*sb=shb+(size_t)s*DIM;
                    for(int d=0;d<DIM;d++) fw[d]=sb[d]+g*(fw[d]-sb[d]); } }
            else if(GBL_G[L]!=1.0f) for(size_t i=0;i<(size_t)S*DIM;i++) Fout[i]=shb[i]+GBL_G[L]*(Fout[i]-shb[i]);
        /* ★corr 重建门物料(DS4_BF_DUMPXR=L, 一次性)★: X=学生态 ffn_in; R=FP专家(同X同路由)−学生routed。
         * shared FP 两侧同→抵消; 同 X 同 gate → 同路由 → R=纯专家量化残差(corr 契约口径, runbook §0.5)。 */
        { static uint64_t xr_done=0; const char*xe=getenv("DS4_BF_DUMPXR");   /* 支持逗号多层: "5,20,35"; 位图防重 */
          int xr_hit=0;
          if(xe&&L<64&&!((xr_done>>L)&1)){ const char*q=xe;
              while(*q){ if(atoi(q)==L){ xr_hit=1; break; } while(*q&&*q!=',')q++; if(*q==',')q++; } }
          if(xr_hit){ xr_done|=(1ULL<<L);
            float *fpout=calloc((size_t)S*DIM,4);
            int *toks=malloc((size_t)S*sizeof(int)); float *wwv2=malloc((size_t)S*4);
            float *xs2=malloc((size_t)S*DIM*4);
            for(int e=0;e<NEXP;e++){
                int nt=0;
                for(int s=0;s<S;s++)for(int a2=0;a2<NACT;a2++)
                    if(idx[(size_t)s*NACT+a2]==e){ toks[nt]=s; wwv2[nt]=rw[(size_t)s*NACT+a2];
                        memcpy(xs2+(size_t)nt*DIM,Fin+(size_t)s*DIM,(size_t)DIM*4); nt++; break; }
                if(!nt) continue;
                char n1[160],n3[160],n2[160]; long rr2,cc2;
                snprintf(n1,sizeof(n1),"layers.%d.ffn.experts.%d.w1.weight",L,e);
                snprintf(n3,sizeof(n3),"layers.%d.ffn.experts.%d.w3.weight",L,e);
                snprintf(n2,sizeof(n2),"layers.%d.ffn.experts.%d.w2.weight",L,e);
                float *e1=st_read_weight(&C,n1,&rr2,&cc2),*e3=st_read_weight(&C,n3,&rr2,&cc2),*e2=st_read_weight(&C,n2,&rr2,&cc2);
                if(!e1||!e3||!e2){ if(e1)free(e1);if(e3)free(e3);if(e2)free(e2); continue; }
                float *aq=calloc((size_t)nt*DIM,4);
                dq_expert_fp(xs2,e1,e3,e2,wwv2,aq,nt,DIM,MOEI,SWLIM);
                for(int i=0;i<nt;i++)for(int d2=0;d2<DIM;d2++) fpout[(size_t)toks[i]*DIM+d2]+=aq[(size_t)i*DIM+d2];
                free(aq); free(e1);free(e3);free(e2);
            }
            for(size_t i=0;i<(size_t)S*DIM;i++) fpout[i]-=(Fout[i]-shb[i]);   /* R = FP − 学生routed */
            char px[64],pr[64];
            snprintf(px,sizeof(px),"/tmp/xr_x_L%02d.npy",L); snprintf(pr,sizeof(pr),"/tmp/xr_r_L%02d.npy",L);
            xr_npy(px,Fin,S,DIM); xr_npy(pr,fpout,S,DIM);
            fprintf(stderr,"[XR] L%d 重建门物料已捕获: %s %s (S=%d)\n",L,px,pr,S);
            free(fpout);free(toks);free(wwv2);free(xs2);
          } }
        } else {
            lfile_t lf; char lp[512];
            snprintf(lp,sizeof(lp),"%s/dql_L%02d.bin",getenv("DS4_LAYER_DIR")?getenv("DS4_LAYER_DIR"):".",L);
            double _lf0=vqt_now();
            int _lfrc=lfile_load(lp,&lf);
            g_lt[6]+=vqt_now()-_lf0;
            if(_lfrc==0){
                /* ★反修遍硬闸(2026-08-23 审计)★: VQ 侧车挂载失败会静默 vqmap=NULL →
                 * bytes_moe 拿 1bit 基座当学生, z 全解错且无报错。禁静默假学生。 */
                if(LZRANK>0&&!lf.vqmap){
                    fprintf(stderr,"★反修遍 L%d: dql_vq 侧车缺/损(vqmap=NULL), 学生≠部署字节 — 拒跑★\n",L);
                    exit(1);
                }
                g_replay_cur_L=L; { double _b2=vqt_now(); bytes_moe(&lf,S,Fin,idx,rw,Fout); g_lt[7]+=vqt_now()-_b2; } lfile_free(&lf);
                if(L<NL&&GBL_G[L]!=1.0f) for(size_t i=0;i<(size_t)S*DIM;i++) Fout[i]=shb[i]+GBL_G[L]*(Fout[i]-shb[i]);
            } else fprintf(stderr,"[B] 层文件 %s 读失败 — Fout 只含 shared\n",lp);
        }
        /* ★守卫必须与唯一消费者(下方 z 解算块 + 343 行 memcpy)对齐★(2026-08-28 实锤 bug)
         * 原守卫只有 LZRANK>0&&do_quant, 而消费者还要 ANC_OK && cfg!='F' && !zrec_done。
         * 后果: 推进段跑完 43 层全有 zrec ⇒ sweep 里消费者一次都不执行, 生产者却每层每次
         * 前向照跑 —— bf_fp_routed 是【读 25.7GB FP 权重 + 全精度专家前向】, 且它不随抽格
         * 行数缩小(行少了权重还是要全读)。实测: 推进段 6.6s/层, sweep 每单元 7 次前向×5 层
         * ≈231s 纯浪费, 占 BFUNIT 460s 的一半。
         * (COADAPT 那支由外层 else-if 排除, 无需重复判。) */
        if(LZRANK>0&&do_quant&&ANC_OK&&cfg!='F'&&!zrec_done){   /* ★层局部靶物料: R=FP专家@Fin−回放routed */
            if(!BF_LT) BF_LT=malloc((size_t)S*DIM*4);
            { double _f0=vqt_now(); bf_fp_routed(L,Fin,idx,rw,S,BF_LT); g_lt[7]+=vqt_now()-_f0; }
            for(size_t i=0;i<(size_t)S*DIM;i++) BF_LT[i]-=(Fout[i]-shb[i]);
            BF_LT_L=L;
            /* ★ELM 针(2026-08-28 用户令"1、2 打一针再决策")★
             * 此刻三样物料正好齐: x=Fin(部署口径, 量化链的真 Fin, 不是锚 fin)、
             * y_q=Fout−shb(量化 routed)、dH=BF_LT(FP routed−量化 routed)。
             * 只在 --elm-probe 点名的层上跑, 不落盘不改模型, 纯读数。 */
            if(elm_probe_hit(L)){
                float *yq=malloc((size_t)S*DIM*4);
                for(size_t i=0;i<(size_t)S*DIM;i++) yq[i]=Fout[i]-shb[i];
                elm_res er; double t0=vqt_now();
                if(elm_solve(Fin,yq,BF_LT,S,DIM,n_fit,&er)==0){
                    printf("★ELM L%02d held行为挽回 %.2f%% @V₀=%s λ=%g k=%d "
                           "| 同口径乘性【线性】对照 %.2f%%(判例+0.6%%) | s=%.4g | %.0fs\n",
                           L, er.held*100.0, er.from_pca?"PCA":"rand", (double)er.lam, er.k,
                           er.held_lin*100.0, (double)er.s, vqt_now()-t0);
                    fflush(stdout); elm_free(&er);
                } else printf("★ELM L%02d 解算失败\n",L), fflush(stdout);
                free(yq);
            }
        }
        free(shb);
    } else if(do_quant&&cfg=='g'&&COADAPT>0&&ANC_OK){
        /* 共适应路径: base(w2)↔z 交替闭式收敛(取代下方一次性 worker + 一次性 z 块) */
        coadapt_moe(L,S,n_fit,Fin,idx,rw,Fout,H2,post2,comb2,st);
    } else {
    int want_local = do_quant && cfg!='F' && (LOCALQ||ABLATE);
    float *shared=NULL,*Ffp=NULL,*Fabl[4]={0,0,0,0};
    if(want_local){ shared=malloc((size_t)S*DIM*4); memcpy(shared,Fout,(size_t)S*DIM*4);
                    Ffp=malloc((size_t)S*DIM*4);    memcpy(Ffp,Fout,(size_t)S*DIM*4); }
    if(do_quant&&cfg!='F'&&ABLATE) for(int ci=0;ci<4;ci++){ Fabl[ci]=malloc((size_t)S*DIM*4); memcpy(Fabl[ci],shared,(size_t)S*DIM*4); }
    /* 并行专家循环 */
    int nth=NTHREADS; if(nth<1)nth=1; if(nth>NEXP)nth=NEXP;
    int e_next=0;
    ework_t *ws=calloc((size_t)nth,sizeof(ework_t));
    pthread_t *th=malloc((size_t)nth*sizeof(pthread_t));
    for(int t=0;t<nth;t++){
        ws[t].L=L;ws[t].S=S;ws[t].n_fit=n_fit;ws[t].do_quant=do_quant;ws[t].cfg=cfg;
        ws[t].Fin=Fin;ws[t].idx=idx;ws[t].rw_act=rw;ws[t].e_next=&e_next;
        ws[t].fout=calloc((size_t)S*DIM,4);
        ws[t].ffp = want_local? calloc((size_t)S*DIM,4) : NULL;
        if(do_quant&&cfg!='F'&&ABLATE) for(int ci=0;ci<4;ci++) ws[t].abl[ci]=calloc((size_t)S*DIM,4);
        pthread_create(&th[t],NULL,expert_worker,&ws[t]);
    }
    for(int t=0;t<nth;t++){
        pthread_join(th[t],NULL);
        for(size_t i=0;i<(size_t)S*DIM;i++) Fout[i]+=ws[t].fout[i];
        if(ws[t].ffp){ for(size_t i=0;i<(size_t)S*DIM;i++) Ffp[i]+=ws[t].ffp[i]; free(ws[t].ffp); }
        for(int ci=0;ci<4;ci++) if(ws[t].abl[ci]){ for(size_t i=0;i<(size_t)S*DIM;i++) Fabl[ci][i]+=ws[t].abl[ci][i]; free(ws[t].abl[ci]); }
        free(ws[t].fout);
        if(st){ st->calib_rows+=ws[t].calib_rows; st->calib_empty+=ws[t].calib_empty; st->nhit+=ws[t].nhit; }
    }
    free(ws);free(th);
    /* 局部质量 (held 行, routed-only R²=1-relL2², 减 FP shared): 旧口径诊断, 非判决 */
    if(want_local&&st){
        int s0=(n_fit>0&&n_fit<S)?n_fit:0; if(S-s0<1)s0=0;
        double sa=1e-12,se=1e-12; for(int s=s0;s<S;s++)for(int d=0;d<DIM;d++){ double A=Ffp[(size_t)s*DIM+d]-shared[(size_t)s*DIM+d]; sa+=A*A; se+=(double)shared[(size_t)s*DIM+d]*shared[(size_t)s*DIM+d]; }
        st->rs_ratio=sqrt(sa/se);   /* routed/shared 幅度比: 小 → 该层局部R²难看但对输出份额小(伪差层) */
        #define DQ_E2(F) ({ double e2=0; for(int s=s0;s<S;s++)for(int d=0;d<DIM;d++){ \
            double A=Ffp[(size_t)s*DIM+d]-shared[(size_t)s*DIM+d], B=(F)[(size_t)s*DIM+d]-shared[(size_t)s*DIM+d]; e2+=(B-A)*(B-A);} e2; })
        st->loc_r2=1.0-DQ_E2(Fout)/sa; st->have_loc=1;
        if(ABLATE&&Fabl[0]){
            double e[4]; for(int ci=0;ci<4;ci++){ e[ci]=DQ_E2(Fabl[ci]); st->abl_r2[ci]=1.0-e[ci]/sa; }
            st->abl_rel[0]=sqrt(e[0]/sa); st->abl_rel[1]=sqrt(e[2]/sa);
            int nh=S-s0;
            float *fh=malloc((size_t)nh*DIM*4),*gh=malloc((size_t)nh*DIM*4);
            for(int s=s0;s<S;s++)for(int d=0;d<DIM;d++){ fh[(size_t)(s-s0)*DIM+d]=Fabl[3][(size_t)s*DIM+d]-shared[(size_t)s*DIM+d];
                gh[(size_t)(s-s0)*DIM+d]=Ffp[(size_t)s*DIM+d]-shared[(size_t)s*DIM+d]; }
            float *wv=malloc((size_t)DIM*4); ds4_loss_dim_variance(gh,nh,DIM,wv);
            st->abl_la=ds4_loss_align(fh,gh,nh,DIM); st->abl_lc=ds4_loss_classify(fh,gh,wv,nh,DIM);
            free(wv);free(fh);free(gh);
        }
        #undef DQ_E2
    }
    if(shared)free(shared); if(Ffp)free(Ffp);
    for(int ci=0;ci<4;ci++) if(Fabl[ci])free(Fabl[ci]);
    }   /* end 非共适应路径 */
    /* ★纯VQ路径接入(2026-08-23)★: 纯 VQ 战役唯一的量化前向就是 B 回放(裸评撤),
     * cfg!='B' 禁入让序贯 z^L 在纯 VQ 下永不触发(ZLGATE=0 实锤)。B 禁入的两个理由
     * (活体每前向重解/修正不落盘假忠实)在"解一次+落盘 zrec+序贯"下都不成立:
     * 解一次由 zrec 幂等闸保证, 落盘走独立 zrec_L%02d.bin(引擎 type6/合并器直通),
     * dql 层文件不动。'F'(FP 锚遍)仍禁。 */
    /* ★搬到三分支合流点(2026-08-23): z 段原长在专家量化分支体内, B 回放分支(纯 VQ
     * 战役唯一前向)从不经过 → ZDIAG 无声实锤。coadapt 分支自带 z, 条件排除。 */
    { static int zc_diag=0;   /* 进入条件诊断(2026-08-29): 上轮进了这轮没进, 不猜, 打出来 */
      if(zc_diag++<2) fprintf(stderr,"[z块条件] do_quant=%d ANC_OK=%d LZRANK=%d cfg=%c zrec_done=%d COADAPT=%d\n",
                              do_quant,ANC_OK,LZRANK,cfg,zrec_done,COADAPT); }
    if(do_quant&&ANC_OK&&LZRANK>0&&cfg!='F'&&!zrec_done&&!(cfg=='g'&&COADAPT>0)){   /* ★'B'回放禁入已撤(见上) — 原注: 活体 z^L 每前向对锚重解会
        (a)把反修候选扰动拉回锚投影(橡皮筋, 候选逐位无效) (b)回放偷加不在文件的修正(合并模型没有→假忠实) */
        /* ★逐层动态 z^L(用户四支柱正确形态, 序贯锚定回拉)★: 输出端单点 z 要一口气补 43 层
         * 累积非线性误差(已证死路); z^L 每层只补【到本层为止的漂移】(小/局部/低秩可期), 分而治之。
         * hc_post 对 Fout 线性 → 目标在 Fout 空间闭式反解: ΔFout_td = Σ_j post_tj·(Hfp−Hq)_tjd / Σ_j post_tj²
         * (把本层出口整体拉回 FP 锚, 不论误差来自 MoE/attention/前层残差)。
         * z^L: Fin→ΔFout 的 dual-form ridge RRR + 四损失(align/classify/smooth感知/fixed)选秩 k_L;
         * 只用 fit 行拟合, 应用于全部行(held=泛化), 下层看到校正后激活(序贯, 同顺序量化哲学)。
         * 体积: Σk_L×2·DIM×fp16 → k=8 全层 ≈5.4MB(产物③可调秩侧车)。 */
        /* ★锚索引必须用建锚时的原始 S, 不是本次调用的 S★(2026-08-27 实锤 bug)
         * 粗筛前向(backfit sweep)调 layer_fwd 时传的是抽格后的 S(=Ss≈S/12), 但 ANC.H 是按
         * H_b=NLAYERS*S_full*HCM*DIM 分配的。用局部 S 算层偏移 ⇒ L*Ss 只有 L*S_full 的 1/12,
         * ★指到别的层的锚区去★。实撞: lyr86 sweep 评 L38(该层无 zrec 故进本块)基线
         * val出口 0.5637(推进段) → 34.8461, 链闸误判"链失稳"硬停, 三跑逐位复现。
         * 只在【没有 zrec 的层】暴露 —— 有 zrec 的层整块被 !zrec_done 跳过, 所以只炸一层, 极隐蔽。
         * p6 早就备好了这对全局(g_anc_rowstride=锚原始行 stride, g_anc_rowmap=抽格行→原始行),
         * p13:112 也设好了, 本块从来没用。行索引同理必须过映射。 */
        const int ancS = g_anc_rowmap ? g_anc_rowstride : S;
        #define ANCROW(sx) (g_anc_rowmap ? g_anc_rowmap[(sx)] : (sx))
        float *Hq=malloc((size_t)S*HCM*DIM*4);
        dq_hc_post(Fout,H2,post2,comb2,Hq,S,HCM,DIM);
        const float *Hf=ANC.H+(size_t)L*ancS*HCM*DIM;
        float *DF=malloc((size_t)S*DIM*4);
        if(BF_LT&&BF_LT_L==L){   /* ★B路序贯: 层局部靶(见 bf_fp_routed 注释); 非B路走原漂移靶 */
            memcpy(DF,BF_LT,(size_t)S*DIM*4);
        } else {
            bfdf_ctx dfc={post2,Hf,Hq,DF}; zpar_for(S,20,bfdf_worker,&dfc);
        }
        free(Hq);
        /* fit 内部再切 val: 前 vs 行拟合, [vs,n_fit) 选秩+GO门(k=0 可关本层 z); held 永不参与 */
        /* ★打分行域=布局分层, 不是行号区间(2026-08-29)★
         * 旧: vs=(n_fit*3)/4 ⇒ 打分区 [4608,6144)。抽样把 8 个域【连续】铺进 8192 行,
         * 那个区间只覆盖西里尔后半+math 两个域 —— z 闸(401/428)与 GE 闸(506/511)都靠它,
         * 于是"层内正向落地"是拿 2/8 个域判的。行表由 <锚>.layout 推导, 跨全域。
         * vs/nval 仍保留: 它们还管【解算用哪些行】(z_solve_dual 取前 vs 行), 那是拟合侧;
         * 这里换掉的只是【打分侧】。 */
        int nEV=0; const int *EVR=row_layout_ev(anchor_path(),&nEV);
        if(!EVR||nEV<2){ fprintf(stderr,"★反修停车: 行布局缺 %s.layout★\n",anchor_path());
            fprintf(stderr,"  补法: bash gguf-tools/scripts/amp_campaign.sh idshalf 然后 anchors3\n"); exit(9); }
        /* ★抽格语境换算(2026-08-29 SIGSEGV 实锤修)★ EVR 是【原始行号】(0..S_full)。
         * 推进段 S=S_full 行号=下标, 直接用没事; 但 sweep 的 backfit_prev_chunk 会用
         * 【抽格前向】(S=Ss≈512 紧凑数组)重跑无 zrec 层的本块 —— 拿 8000 级原始行号去
         * 索引 512 行的 Hq ⇒ 越界段错(gdb 栈: layer_fwd←gs_forward_exit←backfit_prev_chunk)。
         * p13 处理了这个换算(scr?EVc:LEV), 这里漏了。紧凑行 s 的原始行号=g_anc_rowmap[s],
         * 故紧凑打分行 = {s: g_anc_rowmap[s] ∈ EVR}。 */
        int *evr_loc=NULL;
        if(g_anc_rowmap){
            char *inev=calloc((size_t)g_anc_rowstride,1);
            for(int i=0;i<nEV;i++) if(EVR[i]<g_anc_rowstride) inev[EVR[i]]=1;
            evr_loc=malloc((size_t)S*sizeof(int)); int m=0;
            for(int sx=0;sx<S;sx++) if(inev[g_anc_rowmap[sx]]) evr_loc[m++]=sx;
            free(inev);
            if(m<2){ free(evr_loc); evr_loc=NULL; }   /* 抽格里 eval 行太少: 本层 z 块跳过打分侧 */
            else { EVR=evr_loc; nEV=m; }
        }
        int vs=(n_fit*3)/4; int nval=n_fit-vs;
        if(nval<8){ vs=n_fit; nval=0; }
        /* 行帽已撤(2026-08-23): 2048 行全梯拒(过拟合), 行数是拟合质量的硬需求;
         * 时长改从 z_solve_dual 并行化拿(600GFLOP 单线程→20 线程)。 */
        double _z0=vqt_now();
        ds4_z *zl=z_solve_dual(Fin,DF,(uint32_t)vs,DIM,DIM,(uint32_t)LZRANK,LZLAMBDA);
        g_lt[1]+=vqt_now()-_z0;   /* 借用槽1(路由实测仅 0.1s), 打印改名 zsolve */
        if(zl){
            int kL = nval>0 ? z_pick_rank(zl,Fin+(size_t)vs*DIM,DF+(size_t)vs*DIM,nval,DIM,NULL,NULL,NULL,NULL)
                            : (int)zl->rank;
            /* ★真判据闸(2026-07-14 用户裁决"必须正向落地")★: z_pick_rank 只做候选秩提议
             * (其抽象四损失分 k=0 正则恒0 → k>0 先付~20%罚 → 恒选0 的结构病, 与四损失阶段同源);
             * 落地判据换成全项目统一口径: val 行本层出口 relL2 净降才落地。梯子 {提议k, 8, 4, 2, 1}
             * 逐个过闸, 首个净降者冻结为 zl.RRR 记录(可回放/可合并/引擎可执行的产物③)。 */
            int kland=0;
            /* ★梯子顶格跟 LZRANK(2026-08-23)★: 旧梯 {kL?:8,8,4,2,1} 顶格 8 —— z_pick_rank
             * 恒提 0(结构病自认)时 DS4_LZ>8 完全进不了落地梯, k=64 白解。 */
            /* ★从大到小(2026-08-23 用户"落地没有肉"): 旧序从提议 k 起步"够用即止",
             * kL=8 过闸就停 → 64 永不被试。最大肉 = 顶格先试, 首个净降即最大可落 k。 */
            { int cand0[7]={LZRANK,LZRANK/2,16,kL>0?kL:8,8,4,1}, tried[7]={0,0,0,0,0,0,0};
              size_t lst2=(size_t)S*HCM*DIM;
              float *Hq2=malloc(lst2*4), *Ftry=malloc((size_t)S*DIM*4), *zd=malloc((size_t)DIM*4);
              const float *Hf2=ANC.H+(size_t)L*ancS*HCM*DIM;   /* 同上: 锚步长用 ancS 不是 lst2(=S*HCM*DIM) */
              /* 基线 val 出口 relL2 */
              double e0; { dq_hc_post(Fout,H2,post2,comb2,Hq2,S,HCM,DIM);
                e0=bf_exit_relL2_rows(Hq2,Hf2,EVR,nEV); }
              g_lt[4]=vqt_now()-lt_mark-g_lt[3]-g_lt[6]-g_lt[7]; lt_mark=vqt_now();   /* ④其余(扣三项) */
              { static double bf_e0_prev=-1.0;   /* ★链闸(2026-08-25)★: 逐层基线日志+失控守卫 */
                fprintf(stderr,"[链闸] L%02d 基线val出口=%.4f 前层=%.4f 靶=%s\n",
                        L,e0,bf_e0_prev,(BF_LT&&BF_LT_L==L)?"层局部":"漂移");
                if(bf_e0_prev>0.0&&e0>bf_e0_prev*1.5+0.05){
                    /* ★硬停降级为警告(2026-08-29 用户令"别浪费时间")★ 阈值 1.5x+0.05 是按
                     * 旧 2 域打分口径校的; 全域口径下 L41 实测 0.368→1.06 触发, 三刀二分
                     * (摘注入/换层件/查锚)已把数据损坏全排除, 剩下是口径尺度或链上漂移 ——
                     * 都不该由推进段自杀裁决。判决权交给五指标(项目铁律: 唯一裁判=在线五指标)。 */
                    fprintf(stderr,"★[链闸] L%02d 基线暴涨 %.4f→%.4f (>1.5x+0.05) — 警告继续(终判交五指标)★\n",L,bf_e0_prev,e0); }
                bf_e0_prev=e0; }
              for(int ci=0;ci<7&&!kland;ci++){ int kk=cand0[ci];
                if(kk<1||kk>(int)zl->rank) continue;
                int dup=0; for(int cj=0;cj<ci;cj++) if(cand0[cj]==kk&&tried[cj]) dup=1;
                if(dup) continue;
                tried[ci]=1;
                ds4_z_set_rank(zl,(uint32_t)kk);
                memcpy(Ftry,Fout,(size_t)S*DIM*4);
                for(int s=0;s<S;s++){
                    memset(zd,0,(size_t)DIM*4); ds4_z_apply(zl,Fin+(size_t)s*DIM,zd);
                    double nd=0,nf=0; const float*fo=Ftry+(size_t)s*DIM;
                    for(int d2=0;d2<DIM;d2++){ nd+=(double)zd[d2]*zd[d2]; nf+=(double)fo[d2]*fo[d2]; }
                    nd=sqrt(nd); nf=sqrt(nf);
                    double cap=LZTR*nf; float sc2=1.0f;
                    if(nd>cap&&nd>0) sc2=(float)(cap/nd);
                    float *fw=Ftry+(size_t)s*DIM;
                    for(int d2=0;d2<DIM;d2++) fw[d2]+=sc2*zd[d2];
                }
                double e1; { dq_hc_post(Ftry,H2,post2,comb2,Hq2,S,HCM,DIM);
                  e1=bf_exit_relL2_rows(Hq2,Hf2,EVR,nEV); }
                printf("ZLGATE L=%d k=%d val出口relL2 %.6f→%.6f %s\n",L,kk,e0,e1,e1<e0-1e-9?"✓落地":"✗拒");
                fflush(stdout);
                if(e1<e0-1e-9){
                    kland=kk;
                    memcpy(Fout,Ftry,(size_t)S*DIM*4);   /* 序贯: 下层看到校正后激活 */
                    /* 冻结紧凑因子(活跃 k 列)暂存 → fwd_all 导出层文件后 append zl.RRR */
                    if(ZLP_U){ free(ZLP_U); free(ZLP_V); free(ZLP_Z); ZLP_U=ZLP_V=ZLP_Z=NULL; }
                    ZLP_K=kk; ZLP_L=L; ZLP_TR=(float)LZTR;
                    ZLP_Z=malloc((size_t)kk*4);
                    for(int c=0;c<kk;c++) ZLP_Z[c]=zl->z[c];
                    ZLP_U=malloc((size_t)DIM*kk*4); ZLP_V=malloc((size_t)DIM*kk*4);
                    for(int d2=0;d2<DIM;d2++) for(int c=0;c<kk;c++){
                        ZLP_U[(size_t)d2*kk+c]=zl->U[(size_t)d2*zl->rank+c];
                        ZLP_V[(size_t)d2*kk+c]=zl->V[(size_t)d2*zl->rank+c]; }
                    /* ★纯VQ路径: 独立 zrec 直写(dql 不动; 合并器/引擎 type6 直通) */
                    { const char*ld3=getenv("DS4_LAYER_DIR");
                      if(ld3){ char zp3[1024]; snprintf(zp3,sizeof zp3,"%s/zrec_L%02d.bin",ld3,L);
                        FILE*zf3=fopen(zp3,"wb");
                        if(zf3){ unsigned char hdr3[116]; memset(hdr3,0,116);
                          memcpy(hdr3,"zl.RRR",6);
                          unsigned long long psz3=16ull+2ull*((unsigned long long)kk+2ull*(unsigned long long)kk*DIM);
                          memcpy(hdr3+88,&psz3,8); int one3=1; memcpy(hdr3+112,&one3,4);
                          fwrite(hdr3,1,116,zf3);
                          uint32_t zk3=(uint32_t)kk,di3=(uint32_t)DIM,do3=(uint32_t)DIM; float tr3=(float)LZTR;
                          fwrite(&zk3,4,1,zf3); fwrite(&tr3,4,1,zf3); fwrite(&di3,4,1,zf3); fwrite(&do3,4,1,zf3);
                          uint16_t*h3=malloc(((size_t)kk+2*(size_t)kk*DIM)*2); size_t o3=0;
                          for(int i3=0;i3<kk;i3++) h3[o3++]=go1b_fp32_to_fp16(ZLP_Z[i3]);
                          for(size_t i3=0;i3<(size_t)DIM*kk;i3++) h3[o3++]=go1b_fp32_to_fp16(ZLP_U[i3]);
                          for(size_t i3=0;i3<(size_t)DIM*kk;i3++) h3[o3++]=go1b_fp32_to_fp16(ZLP_V[i3]);
                          fwrite(h3,2,o3,zf3); free(h3); fclose(zf3);
                g_lt[5]=vqt_now()-lt_mark;   /* ⑤ZLGATE 候选评估段完 */
                free(evr_loc); evr_loc=NULL;
                          printf("ZREC L=%d k=%d → zrec_L%02d.bin\n",L,kk,L); fflush(stdout);
                        } } }
                }
              }
              free(Hq2); free(Ftry); free(zd);
            }
            /* ★拒层 GE 兜底(2026-08-23 用户令"抓不到肉是bug·先修bug")★
             * 哈希层(L1/L2 实锤)误差由离散专家身份决定, 连续 x 特征的 z^L 数学上抓不住
             * (L0 尚存 token 身份残留可修, L1+ 解耦后全梯拒)。对症=per-expert 门:
             * m*_s=<Fout+ΔF,Fout>/<Fout,Fout>(每 token 最优缩放) 按路由权重投到专家
             * (backfit form==4 同式), clamp[0.8,1.2]。评估用 ĝ_s(token 的 gate 加权平均门)
             * 近似 —— 专家间差异被平均抹平=低估改善, 过闸则真实改善≥评估(保守安全)。
             * 落地: Fout*=ĝ(序贯传链) + zrec 追加 bf.GE(引擎 type5 精确 per-expert 执行)。 */
            if(kland==0){
                int vs2=(n_fit*3)/4; int nval2=n_fit-vs2; if(nval2<8){ vs2=n_fit; nval2=0; }
                double *gnum=calloc((size_t)NEXP,sizeof(double)),*gden=calloc((size_t)NEXP,sizeof(double));
                for(int s2=0;s2<vs2;s2++){
                    const float*fo=Fout+(size_t)s2*DIM,*df=DF+(size_t)s2*DIM;
                    double num=0,den=1e-12;
                    for(int d2=0;d2<DIM;d2++){ num+=((double)fo[d2]+df[d2])*fo[d2]; den+=(double)fo[d2]*fo[d2]; }
                    double ms2=num/den;
                    for(int a2=0;a2<NACT;a2++){ int e2=idx[(size_t)s2*NACT+a2]; double w2=rw[(size_t)s2*NACT+a2];
                        if(e2>=0&&e2<NEXP&&w2>0){ gnum[e2]+=w2*ms2; gden[e2]+=w2; } }
                }
                float *gev=malloc((size_t)NEXP*4); int nge=0;
                for(int e2=0;e2<NEXP;e2++){ double g=gden[e2]>1e-12?gnum[e2]/gden[e2]:1.0;
                    /* clamp 放宽(2026-08-23 用户令"体积可以大一点·还原优先"):
                     * [0.8,1.2] 时代 L42 已修 8%, 夹断处即剩余肉 */
                    if(g<0.50)g=0.50; if(g>2.00)g=2.00; gev[e2]=(float)g; if(fabs(g-1.0)>1e-4)nge++; }
                free(gnum); free(gden);
                if(nge>=2&&nval2>0){
                    /* val 行评估: Ftry_s = ĝ_s·Fout_s → 出口 relL2 */
                    float *Ftry2=malloc((size_t)S*DIM*4),*Hq3=malloc((size_t)S*HCM*DIM*4);
                    memcpy(Ftry2,Fout,(size_t)S*DIM*4);
                    for(int s2=0;s2<S;s2++){
                        double gw=0,ww=1e-12;
                        for(int a2=0;a2<NACT;a2++){ int e2=idx[(size_t)s2*NACT+a2]; double w2=rw[(size_t)s2*NACT+a2];
                            if(e2>=0&&e2<NEXP&&w2>0){ gw+=w2*gev[e2]; ww+=w2; } }
                        float gh=(float)(gw/ww);
                        float*fw=Ftry2+(size_t)s2*DIM;
                        for(int d2=0;d2<DIM;d2++) fw[d2]*=gh;
                    }
                    const float *Hf3=ANC.H+(size_t)L*ancS*HCM*DIM;
                    double e0g,e1g;
                    { dq_hc_post(Fout,H2,post2,comb2,Hq3,S,HCM,DIM);
                      double e2s=0,a2s=0;
                      for(int ri=0;ri<nEV;ri++){ const int s2=EVR[ri];
                          const float*hq=Hq3+(size_t)s2*HCM*DIM,*hf=Hf3+(size_t)ANCROW(s2)*HCM*DIM;
                          for(size_t i2=0;i2<(size_t)HCM*DIM;i2++){ double d3=(double)hq[i2]-hf[i2]; e2s+=d3*d3; a2s+=(double)hf[i2]*hf[i2]; } }
                      e0g=sqrt(e2s/(a2s+1e-30)); }
                    { dq_hc_post(Ftry2,H2,post2,comb2,Hq3,S,HCM,DIM);
                      double e2s=0,a2s=0;
                      for(int ri=0;ri<nEV;ri++){ const int s2=EVR[ri];
                          const float*hq=Hq3+(size_t)s2*HCM*DIM,*hf=Hf3+(size_t)ANCROW(s2)*HCM*DIM;
                          for(size_t i2=0;i2<(size_t)HCM*DIM;i2++){ double d3=(double)hq[i2]-hf[i2]; e2s+=d3*d3; a2s+=(double)hf[i2]*hf[i2]; } }
                      e1g=sqrt(e2s/(a2s+1e-30)); }
                    printf("ZLGATE L=%d GE(路由投影) val出口relL2 %.6f→%.6f %s (活门=%d)\n",
                           L,e0g,e1g,e1g<e0g-1e-9?"✓落地":"✗拒",nge); fflush(stdout);
                    if(e1g<e0g-1e-9){
                        memcpy(Fout,Ftry2,(size_t)S*DIM*4);   /* 序贯: ĝ 近似传链 */
                        const char*ld4=getenv("DS4_LAYER_DIR");
                        if(ld4){ char zp4[1024]; snprintf(zp4,sizeof zp4,"%s/zrec_L%02d.bin",ld4,L);
                            FILE*zf4=fopen(zp4,"ab");   /* 追加(层可同时有 zl.RRR + bf.GE; 拒层=仅 GE) */
                            if(zf4){ unsigned char hdr4[116]; memset(hdr4,0,116);
                                memcpy(hdr4,"bf.GE",5);
                                unsigned long long psz4=(unsigned long long)NEXP*2;
                                memcpy(hdr4+88,&psz4,8); int one4=1; memcpy(hdr4+112,&one4,4);
                                fwrite(hdr4,1,116,zf4);
                                uint16_t geh4[NEXP];
                                for(int e2=0;e2<NEXP;e2++) geh4[e2]=go1b_fp32_to_fp16(gev[e2]);
                                fwrite(geh4,2,NEXP,zf4); fclose(zf4);
                                printf("ZREC L=%d GE → zrec_L%02d.bin(bf.GE)\n",L,L); fflush(stdout);
                            } }
                    }
                    free(Ftry2); free(Hq3);
                }
                free(gev);
            }
            if(st) st->zk=kland;
            if(kland>0) LZ_TOTAL_K+=kland;
            ds4_z_free(zl);
        }
        free(DF);
    }
