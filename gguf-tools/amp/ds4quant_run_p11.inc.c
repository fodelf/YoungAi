    for(int L=0;L<NLAYERS;L++){
        int pc=-1; char sel_nm[16]="?"; char sel_cfg=0;
        if(minvol&&g_probe_l>=0){
            if(L<g_probe_l) continue;   /* 探针: 前层不量化不推进(链在探针层入口一次性锚恢复) */
            if(L==g_probe_l&&L>0&&ANC_OK){
                memcpy(H,ANC.H+(size_t)(L-1)*lstride,lstride*4);
                fprintf(stderr,"[探针] L00..L%02d=FP 锚定直通(隔离口径), 评 L%02d\n",L-1,L);
            }
        }
        if(minvol){
            char mvnm[16]; char c=plan_lookup_mv(L,mvnm);
            if(c && ckpt_load(L,H,lstride,idh)){
                if(H2&&!ckpt2_load(L,H2,lstride2)){
                    fprintf(stderr,"L%02d rr链 ckpt 缺失(链断) — 硬拒: 删该层起 plan/ckpt 后 RESUME, 或 fresh\n",L); exit(2); }
                plan_out[L]=c; mv_prev_note(c,mvnm); mv_vol_note(L,mvnm,mv_name_bpw(mvnm));
                fprintf(stderr,"L%02d [复用 %s]\n",L,mvnm); continue; }
        } else if((pc=plan_lookup(L))>=0 && ckpt_load(L,H,lstride,idh)){   /* 已调层: 复用 */
            plan_out[L]=TAB[pc].cfg;
            fprintf(stderr,"L%02d [复用 %s]\n",L,TAB[pc].name);
            continue;
        }
        LW W=load_layer(L);
        int best=-1; double brel=0;
        if(minvol){
            /* ★FIX(2026-07-28 用户裁决"20-30分/层=bug"): coadapt 是定稿器不是评估器。
             * 评估用裸 signref g10 作 α 相对基线(分钟级); 四损失 coadapt 只对每层胜者定稿一次。 */
            int mv_base=getenv("DS4_MV_BASELINE")?1:0;   /* 基线模式: 全链冠军配方(g10h), 产 BT */
            int coad_save=COADAPT; COADAPT=0; GO2B_HOT=0;   /* 基线/纯m档必须无热污染 */
            /* ★热启动(2026-07-30): 上层锁 'g' 族 ⇒ m 档无望带, 跳裸基线+二分, 档-1 起试 */
            int warm = !mv_base && g_mv_pcfg=='g' && getenv("DS4_MINVOL_HIST")
                       && !getenv("DS4_MV_NOWARM");
            double rel0=0,s_base=0,relh_floor=-1;
            /* ★现场地板线(2026-07-30 L03 实锤: 静态 V4BF 线含 v4/v5mini 语料水位差 ~3%,
             * 冠军自家配方带 coadapt 也过不了自家线 0.0077 → 薄边距层被迫花体积补口径差 =
             * 本末倒置)。DS4_MV_FLOOR_LINE=1: 每层裸评 g10h16(冠军配方)于本链上文, 线=实测
             * +EPS — 链公平/零语料差/冠军档构造性恒过(真零债); 兼作二分 thr 基线。 */
            int floor_line = getenv("DS4_MV_FLOOR_LINE")?1:0;
            /* ★影子冠军链门(2026-07-30 终版, 用户"每层贪心最小反噬"方案性 bug 裁决):
             * 冠军配方(g10h16+coadapt)从影子自身上文前向一步, 链式 held=本层门线 —
             * 同语料/绝对轨迹(不随我们链漂移)/构造可达(封顶≈线)。兼作二分 thr 基线,
             * 替代 g10 裸基线。影子 ckpt 断点续; 缺档(旧跑)以我们链一次性播种并声明。 */
            static float *Hs=NULL; static int hs_L=-1;
            double shadow_line=-1;
            {int mvfl = getenv("DS4_MINVOL_FLOOR")?atoi(getenv("DS4_MINVOL_FLOOR")):1;
            int shadow_on = getenv("DS4_MINVOL_HIST")&&!mvfl&&!mv_base&&!floor_line;
            if(shadow_on){
                if(!Hs) Hs=malloc(lstride*4);
                if(hs_L!=L-1){
                    if(L>0&&ckpt_load_s(L-1,Hs,lstride,idh)){ hs_L=L-1;
                        fprintf(stderr,"L%02d [影子] ckpt 恢复(至 L%02d)\n",L,L-1); }
                    else { memcpy(Hs,H,lstride*4); hs_L=L-1;
                        fprintf(stderr,"L%02d [影子] %s播种(=当前链上文)\n",L,L>0?"缺 ckpt 一次性":"L0 "); }
                }
                time_t ts0=time(NULL);
                int coS=COADAPT; COADAPT=1;
                hot_from_anchor(L,S,MV_HOTK);
                memcpy(Hw,Hs,lstride*4); set_cand(&MV_G10);
                lstat_t sts; memset(&sts,0,sizeof(sts));
                fprintf(stderr,"L%02d 影子冠军链(g10h16定稿) ",L);
                layer_fwd(L,&W,Hw,ids,S,n_fit,1,'g',&sts);
                s_base=held_score(Hw,S,n_fit,L,&shadow_line);
                memcpy(Hs,Hw,lstride*4); hs_L=L; ckpt_save_s(L,Hs,lstride,idh);
                COADAPT=coS; GO2B_HOT=0;
                rel0=shadow_line;
                fprintf(stderr,"L%02d [影子线] held=%.4f [%lds]\n",L,shadow_line,(long)(time(NULL)-ts0));
            }
            if(!mv_base&&!shadow_on&&(floor_line||!warm)){
            time_t tc0=time(NULL);
            if(floor_line) hot_from_anchor(L,S,MV_HOTK); else GO2B_HOT=0;
            memcpy(Hw,H,lstride*4); set_cand(&MV_G10);
            lstat_t st0; memset(&st0,0,sizeof(st0));
            fprintf(stderr,"L%02d %s ",L,floor_line?"g10h16地板(裸)":"g10 ");
            layer_fwd(L,&W,Hw,ids,S,n_fit,1,'g',&st0);
            s_base=held_score(Hw,S,n_fit,L,&rel0);
            if(floor_line){ relh_floor=rel0; GO2B_HOT=0; }
            fprintf(stderr," held relL2=%.4f score=%.5g [%s %lds]\n",rel0,s_base,
                    floor_line?"地板":"裸基线",(long)(time(NULL)-tc0));
            memcpy(Hb,Hw,lstride*4);
            }}
            char win_nm[16]="g10"; char win_cfg='g'; int win_r=0,win_hot=0;
            double win_vol=1.0625; brel=rel0;
            if(mv_base){   /* 基线模式: 逐层直接冠军配方 g10h(不粗筛不建流形), 产同尺 BT */
                if(getenv("DS4_VQ_RPLAN")&&dq_vq_on()){
                    /* ★R28: 计划表驱动(每层档位+热数); 热集按本层 hot 数从锚重建 */
                    vq_rplan(L);
                    hot_from_anchor(L,S,g_vq_hot);
                    snprintf(win_nm,sizeof(win_nm),"v%dx%d h%d",g_vq_dim,g_vq_nc,g_vq_hot);
                    win_cfg='g'; win_r=0; win_hot=g_vq_hot;
                    win_vol=(double)(16+g_vq_nc*g_vq_dim*2+MOEI*2
                             +((size_t)MOEI*DIM/g_vq_dim*(g_vq_nc<=256?8:(g_vq_nc<=512?9:(g_vq_nc<=1024?10:12)))+7)/8+1)
                            *8.0/((double)MOEI*DIM);
                } else {
                snprintf(win_nm,sizeof(win_nm),"g10h"); win_hot=MV_HOTK;
                win_vol=(16*2.25+240*1.0625)/256.0;
                }
            } else {
            fprintf(stderr,"[minvol] 贪心 rank 二分已随流形档移除 — 用 DS4_MV_BASELINE=1 + DS4_VQ_RPLAN 计划表模式\n");
            exit(2);
            }   /* !mv_base 粗筛段结束 */
            COADAPT=coad_save;
            /* ④★逐层地板门(2026-07-29 用户"对标值用已有对应, 别浪费时间跑基线" → 零外推口径)★
             * 判决=用户 v5 语料 HELD 区(平均语言×平均场景, restore_probe 直measure)。
             * 地板=冠军配方(g10h)在本层同一链式上文下的碑值 M_champ(现场实测):候选档
             * M ≥ M_champ−EPS 即"该层不比冠军配方差" ⇒ 逐层支配 ⇒ 全链质量不降(构造性,
             * 零 BT/零换算/零基线全程)。fail 沿体积升序升档, 顶格=g10h 本身(恒过)。 */
            typedef struct { char nm[16]; char cfg; int r,hot; double vol; } up_t;
            up_t ups[12]; int nup=0;
            if(warm){
                /* ★热启动阶梯: 上层档-1(体积回收探针) → 上层档; 再欠由升档链往上翻 */
                up_t w0,w1; memset(&w0,0,sizeof(w0)); memset(&w1,0,sizeof(w1));
                if(g_mv_phot==0){        /* 上层=g10 → 探 g10, 回 g10h16 */
                    snprintf(w0.nm,16,"g10");   w0.cfg='g'; w0.vol=1.0625;
                    snprintf(w1.nm,16,"g10h%d",MV_HOTK); w1.cfg='g'; w1.hot=MV_HOTK;
                    w1.vol=((256.0-MV_HOTK)*1.0625+MV_HOTK*2.25)/256.0;
                } else if(g_mv_phot<=MV_HOTK){   /* 上层=g10h16 → 探 g10, 回 g10h16 */
                    snprintf(w0.nm,16,"g10");    w0.cfg='g'; w0.vol=1.0625;
                    snprintf(w1.nm,16,"g10h%d",MV_HOTK); w1.cfg='g'; w1.hot=MV_HOTK;
                    w1.vol=((256.0-MV_HOTK)*1.0625+MV_HOTK*2.25)/256.0;
                } else {                 /* 上层=g10hN → 探 g10h(N/2), 回 g10hN */
                    int nh=g_mv_phot/2;
                    snprintf(w0.nm,16,"g10h%d",nh); w0.cfg='g'; w0.hot=nh;
                    w0.vol=((256.0-nh)*1.0625+nh*2.25)/256.0;
                    snprintf(w1.nm,16,"g10h%d",g_mv_phot); w1.cfg='g'; w1.hot=g_mv_phot;
                    w1.vol=((256.0-g_mv_phot)*1.0625+g_mv_phot*2.25)/256.0;
                }
                ups[nup++]=w0; ups[nup++]=w1;
                fprintf(stderr,"L%02d [热启动] 上层锁 %c/h%d → 起试 %s→%s (跳裸基线+rank二分)\n",
                        L,g_mv_pcfg,g_mv_phot,ups[0].nm,ups[1].nm);
            } else {
              up_t w0; memset(&w0,0,sizeof(w0));
              snprintf(w0.nm,16,"%s",win_nm); w0.cfg=win_cfg; w0.r=win_r; w0.hot=win_hot; w0.vol=win_vol;
              ups[nup++]=w0;
              /* 升档梯: g10 → ★g10hN 往上翻(2026-07-30 用户令): 超标不卡死体积,
               * 活跃专家扩热处理, 债不再传链。 */
              up_t lad[1]={{"g10",'g',0,0,1.0625}};
              for(int u=0;u<1;u++) if(lad[u].vol>win_vol+1e-9&&nup<8) ups[nup++]=lad[u]; }
            static double MV_EPS=-1;
            if(MV_EPS<0) MV_EPS=getenv("DS4_MINVOL_EPS")?atof(getenv("DS4_MINVOL_EPS")):0.0005;
            /* ★两遍调度(2026-07-29 用户"快速确定最小体积, 精修选中层")★
             * DS4_MINVOL_FLOOR=0(快扫遍): 跳地板/碑/升档, 粗筛胜者直接定稿导出(~8min/层);
             * =1(精修遍, 默认): 逐层地板门(对 RESUME 选中的层生效)。 */
            static int MV_FLOOR=-1;
            if(MV_FLOOR<0) MV_FLOOR=getenv("DS4_MINVOL_FLOOR")?atoi(getenv("DS4_MINVOL_FLOOR")):1;
            double M_champ=1.0;
            if(!mv_base&&MV_FLOOR){   /* 地板先行: g10h(冠军配方)本层同上文碑值。
                                       * ★裸化(v10 488s→~210s): 门=裸对裸(公平, 免 coadapt);
                                       * 胜者过门后才 coadapt 定稿(产物质量不损)。 */
                int co1=COADAPT; COADAPT=0;
                hot_from_anchor(L,S,MV_HOTK);
                time_t tc1=time(NULL);
                memcpy(Hw,H,lstride*4); set_cand(&MV_G10);
                lstat_t stc; memset(&stc,0,sizeof(stc));
                fprintf(stderr,"L%02d g10h地板(裸) ",L);
                layer_fwd(L,&W,Hw,ids,S,n_fit,1,'g',&stc);
                double relc; held_score(Hw,S,n_fit,L,&relc);
                /* restore_probe 里程碑已删除(2026-08-18 用户令), M_champ 保持默认 1.0 */
                COADAPT=co1;
                fprintf(stderr,"L%02d [地板] g10h(裸) relh=%.4f M_champ=%.4f [%lds]\n",
                        L,relc,M_champ,(long)(time(NULL)-tc1));
            }
            /* ★单层快速优化(2026-07-29 用户终裁"单层快速优化, 不是整个流程"): 一遍流程,
             * 层内=粗筛快选 → 只精修选中档(coadapt 定稿+一次碑)。FLOOR=0(默认路)门=软门:
             * 仅防灾(边际掉幅 > SOFT 才升一档, nup≤2); FLOOR=1 保留逐层地板门(定向精修用)。 */
            static double MV_SOFT=-1;
            if(MV_SOFT<0) MV_SOFT=getenv("DS4_MINVOL_SOFT")?atof(getenv("DS4_MINVOL_SOFT")):0.05;   /* 纯防灾(v10 实测冠军配方浅层每层耗~0.046, 门不误咬正常层) */
            /* 里程碑评估已删除(2026-08-18 用户令): RESUME 首层不再跑后缀 FP 探针
             * (单线程 38 层前向 ~12 分钟)。mprev 保持 -1 = 无碑语义, 软门首层跳过。 */
            /* ★门语义 A(2026-07-29 用户裁决, 索引错位已纠): 逐层对齐冠军谱系 v1 历史链
             * (reports/v1_inherit_chain.txt, INHERIT=链式出口 relh, 同名同式)。
             * 索引验证(fable5 原文 L1=0.1129/L20=0.4526 ↔ 文件第2/第21值): **hist[i]=Li 层值,
             * 43 层一一对应, 门线=hist[L]**。hist[0]=0.0000(v1 链起点记录)→ L0 特判借 hist[1]
             * 并声明。判据=relh ≤ 门线; 语料偏差(历史530tok vs 现v5mini)由单层实测暴露。 */
            static double MV_HIST[64]; static int MV_NHIST=-1;
            if(MV_NHIST<0){ MV_NHIST=0;
                const char*hp=getenv("DS4_MINVOL_HIST");
                if(hp){ FILE*hf=fopen(hp,"r");
                    if(hf){ double v; while(MV_NHIST<64&&fscanf(hf,"%lf",&v)==1) MV_HIST[MV_NHIST++]=v; fclose(hf); }
                    const char*tf=getenv("DS4_MV_LINE_TIGHTEN");   /* 机制审计杠杆: 门线×f 强触发升档链 */
                    if(tf){ double f=atof(tf); for(int i=0;i<MV_NHIST;i++) MV_HIST[i]*=f;
                        fprintf(stderr,"[门A] ★审计: 门线×%.2f 收紧★\n",f); }
                    fprintf(stderr,"[门A] 历史通过线载入 %d 值 ← %s (门线=hist[L]; hist[0]=%.4f%s)\n",
                            MV_NHIST,hp?hp:"",MV_NHIST>0?MV_HIST[0]:0.0,
                            (MV_NHIST>0&&MV_HIST[0]<=0.0)?", ≤0→L0 借 hist[1]":"") ; } }
            int hist_on = (MV_NHIST>=NLAYERS) && !MV_FLOOR;
            double BT=mv_target;
            double hist_line = hist_on ? ((L==0&&MV_HIST[0]<=0.0)?MV_HIST[1]:MV_HIST[L<MV_NHIST?L:MV_NHIST-1]) : 0;
            if(hist_on&&shadow_line>0){
                /* ★影子线优先: 冠军配方同语料自链轨迹 — 绝对/可达/无口径差 */
                fprintf(stderr,"L%02d [门A·影子线] 线=%.4f(冠军配方同语料链) V4BF参考=%.4f (口径差 %+.1f%%)\n",
                        L,shadow_line,hist_line,hist_line>0?(shadow_line/hist_line-1.0)*100.0:0.0);
                hist_line=shadow_line;
            }
            else if(hist_on&&relh_floor>=0){
                /* ★现场地板覆盖静态线: 链公平, V4BF 静态线仅留作语料水位差遥测 */
                static double MV_FEPS=-1;
                if(MV_FEPS<0) MV_FEPS=getenv("DS4_MV_FLOOR_EPS")?atof(getenv("DS4_MV_FLOOR_EPS")):0.0005;
                double oldl=hist_line; hist_line=relh_floor+MV_FEPS;
                fprintf(stderr,"L%02d [门A·现场地板] 线=%.4f(g10h16裸+%.4g) V4BF参考=%.4f (水位差 %+.1f%%)\n",
                        L,hist_line,MV_FEPS,oldl,oldl>0?(relh_floor/oldl-1.0)*100.0:0.0);
            }
            else if(hist_on&&L>0&&L<MV_NHIST&&MV_HIST[L-1]>1e-9){
                /* ★增长率门(2026-07-30 用户"算法设计有问题重新优化"终版): 绝对线撞语料口径墙
                 * (L08 缺口>档梯回收=必硬停), 相对地板线无锚螺旋(+30%) — 门=冠军谱系逐层
                 * 增长率: req=ours(L-1)×hist[L]/hist[L-1]。比值内语料口径差消掉; 轨迹形状
                 * 钉死不漂; 终链 ≤ ours(0)/hist[0]×hist[42]=0.4616(L00 领先 11% 继承到终点)。 */
                double pv=plan_relh_of(L-1);
                if(pv>0){
                    double rt=MV_HIST[L]/MV_HIST[L-1], rq=pv*rt;
                    fprintf(stderr,"L%02d [门A·增长率] req=%.4f(上层链%.4f×谱系率%.4f) 绝对线参考=%.4f\n",
                            L,rq,pv,rt,hist_line);
                    hist_line=rq;
                }
            }
            double M_req = mv_base?-1e9:
                (MV_FLOOR? (M_champ-MV_EPS)
                 /* ★hist 门补 v16 标准容差(2026-07-30 L03 实锤: m768h coadapt 0.2244 距线
                  * 0.0003 被拒→逼向更差且贵 55% 的 g 族 = 本末倒置; v16 用户定义语义=
                  * "实测 ≥ 冠军同层−0.0005 即不比冠军差", FLOOR 支路一直有, hist 漏掉) */
                 : (hist_on? hist_line+MV_EPS
                    : (H2? ((L>=NLAYERS-1)?BT:(BT+((mprev<0?1.0:mprev)-BT)*(double)(NLAYERS-1-L)/(double)(NLAYERS-L)))
                         : ((mprev<0?1.0:mprev)-MV_SOFT))));
            if(!MV_FLOOR&&!hist_on&&nup>3) nup=3;   /* 软门: 至多三试; ★门A 零债(2026-07-30
                                                     * 用户令): 升档到过门为止, 债不传链 */
            float *H2t=H2?malloc(lstride2*4):NULL;
            int usel=-1; double Msel=mprev; int mv_pass=0;
            double vbare[12]; for(int vi=0;vi<12;vi++) vbare[vi]=-1;   /* 各档裸值(不可达层最小同质档选用) */
            /* ★搜索效率三件套(2026-07-30 L04 实锤, 门判据不动): 本层实测 coadapt 增益驱动
             * 复判窗(静态 0.006 窗把够不着的档也复判, 90s/档); m 带饱和截断; g10 免重评 */
            static double g_mv_gain=0.004;            /* coadapt 增益跨层 EMA */
            static double MV_CMARG=-1;
            if(MV_CMARG<0) MV_CMARG=getenv("DS4_MV_COAD_MARGIN")?atof(getenv("DS4_MV_COAD_MARGIN")):0.012;
            double lay_gain=-1, m_prev_bare=-1; (void)m_prev_bare;   /* -1=本层未实测(用跨层 EMA) */
#define MV_CWIN() ({ double _lg=lay_gain>=0?lay_gain:g_mv_gain; double _w=_lg*1.25+0.0005; _w>MV_CMARG?MV_CMARG:_w; })
            int pure=(mv_base&&getenv("DS4_PURE_VQ"))?1:0;
            if(pure){ usel=0; Msel=1.0; mv_pass=1;
                set_cand(&MV_G10); }   /* ★纯VQ·导出先行: 裸评撤, 链态由 B 回放出(⑤内)。
                                        * set_cand 必须补(μ10×3 冠军配方) — 否则 w2 导出退回 μ=1 弱锚解
                                        * (2026-08-03 实锤: μ=1 的 B 口径 held 0.2386 vs μ=10 应显著优) */
            for(int u=0;u<nup&&!pure;u++){ up_t*c=&ups[u];
                int is_champ = (c->cfg=='g'&&c->hot>0);
                if(c->hot) hot_from_anchor(L,S,c->hot); else GO2B_HOT=0;
                /* ★裸态门判(2026-07-29 用户"rr标准对齐冠军"): 门=裸对裸(与裸地板同待遇, 免每档
                 * coadapt); 胜者过门后才唯一一次 coadapt 定稿 — 产物质量不损, rrm=裸碑值(保守下界)。 */
                time_t tf0=time(NULL);
                int co2=COADAPT; COADAPT=0;
                memcpy(Hw,H,lstride*4); set_cand(&MV_G10);
                lstat_t stf; memset(&stf,0,sizeof(stf));
                fprintf(stderr,"L%02d %s(裸) ",L,c->nm);
                layer_fwd(L,&W,Hw,ids,S,n_fit,1,c->cfg,&stf);
                double relf,s_fin=held_score(Hw,S,n_fit,L,&relf); (void)s_fin;
                double M;
                if(mv_base) M=1.0;            /* 基线: 免碑(轨迹=每5层里程碑) */
                else if(hist_on) M=relf;      /* ★门A: 判据=链式 relh(与 v1 历史链同名同式, 方向小好), 免碑省时 */
                else if(MV_FLOOR&&is_champ) M=M_champ;  /* 地板门模式顶格免测 */
                else if(H2t){ rr_step(L,&W,H2,H2t,S,n_fit,c->cfg); M=rr_probe(H2t,L); }
                else M=restore_probe(Hw,L,ids,S,n_fit);   /* ★判决=v5 语料 HELD 区(裸态) */
                COADAPT=co2;
                int pass = mv_base || (hist_on ? (relf<=M_req) : (M>=M_req));
                vbare[u]=relf;
                if(!mv_base)
                fprintf(stderr,"L%02d [%s] %-6s 裸M=%.4f relh=%.4f 门=%.4f(%s) [%lds] → %s\n",
                        L,hist_on?"历史链门|relh":(MV_FLOOR?"地板门|v5held":(H2t?"预算门|rr":"软门|v5held")),c->nm,M,relf,M_req,
                        hist_on?"冠军历史held[L], relh≤":(MV_FLOOR?"地板−EPS":(H2t?"历史指标线":"上层−SOFT")),
                        (long)(time(NULL)-tf0),pass?"PASS":"升档");
                if(!pass&&hist_on&&!mv_base){
                    /* ★近失·coadapt 复判(2026-07-30 L03 实锤: 门线=冠军 coadapt 后过程值,
                     * 裸对线系统性吃亏 ~0.004-0.01≈一档, 逐层多锁档=本末倒置)。近失候选花
                     * 一次定稿态前向复判 — 状态(set_cand/热)沿用本候选裸评现场。 */
                    double cwin = MV_CWIN();
                    if(relf-M_req<=cwin){
                        time_t tn0=time(NULL);
                        memcpy(Hw,H,lstride*4); set_cand(&MV_G10);
                        lstat_t stn; memset(&stn,0,sizeof(stn));
                        fprintf(stderr,"L%02d %s(coad复判) ",L,c->nm);
                        layer_fwd(L,&W,Hw,ids,S,n_fit,1,'g',&stn);
                        double relc2; held_score(Hw,S,n_fit,L,&relc2);
                        fprintf(stderr,"L%02d [复判] %s coadapt relh=%.4f 门=%.4f [%lds] → %s\n",
                                L,c->nm,relc2,M_req,(long)(time(NULL)-tn0),relc2<=M_req?"PASS":"仍欠");
                        { double gg=relf-relc2; if(gg>lay_gain) lay_gain=gg;
                          g_mv_gain=0.5*g_mv_gain+0.5*(gg>0?gg:0); }
                        if(relc2<=M_req){ pass=1; M=relc2; relf=relc2; }
                    }
                }
                if(pass){ usel=u; Msel=M; brel=relf; mv_pass=1; break; }
                usel=u; Msel=M; brel=relf;
                if(!MV_FLOOR&&(u+1<nup||hist_on)&&u+1<12){
                    /* ★自适应升档: m 带按掉幅分级(微欠 r+128 / 中欠 m512h / 大欠 m768h),
                     * 'g' 带★往上翻(2026-07-30 用户令): 活跃专家按锚路由权重扩热
                     * (go2b 2.25), 16→32→64→128→256, 欠幅大跳级 — 体积不卡死, 债不传链。 */
                    double defi = hist_on ? (M-M_req) : (M_req-M);   /* 欠量(门A: relh 超出量) */
                    if(c->cfg=='g'){
                        if(!hist_on){ nup=u+1; fprintf(stderr,"L%02d [升档] g10 已顶格(软门) — 记债交 backfit\n",L); }
                        else if(c->hot>=256){ nup=u+1; /* 物理顶(全热): 环后零债裁决 */ }
                        else {
                            /* ★恒单步(2026-07-30 用户"别本末倒置"): g10 欠→必试 g10h16(=冠军
                             * 配方, 零债下构造性达线), 之上逐倍安全阀 — 禁跳级防越过最小可过档 */
                            int nh = c->hot? c->hot*2 : MV_HOTK;
                            if(nh>256) nh=256;
                            up_t nx; memset(&nx,0,sizeof(nx));
                            snprintf(nx.nm,sizeof(nx.nm),"g10h%d",nh); nx.cfg='g'; nx.hot=nh;
                            nx.vol=((256.0-nh)*1.0625+nh*2.25)/256.0;
                            ups[u+1]=nx; if(u+2>nup) nup=u+2;
                            fprintf(stderr,"L%02d [升档] 掉幅=%.4f → 往上翻 %s(bpw %.4f)\n",L,defi,nx.nm,nx.vol);
                        }
                    }
                }
            }
            if(hist_on&&!mv_pass&&!mv_base){
                /* ★不可达层终版(2026-07-30 L05 实锤: 全档梯+coadapt 距增长率 req 仍欠 0.0028
                 * = 该层物理不可达): 取与最佳裸值差≤0.001 的最小体积档(饱和区同质 — 如
                 * m768h 0.854bpw ≈ h256 2.25bpw), 链取实际值前进, 下层 req 自适应 —
                 * 不欠账不烧体积不硬停。DS4_MINVOL_STRICT=1 恢复硬停。 */
                double defi=brel-M_req;
                if(getenv("DS4_MINVOL_STRICT")){
                    fprintf(stderr,"L%02d ★门A档梯走完(%s)仍欠 %.4f — STRICT 硬停(rc=8)★\n",
                            L,ups[usel].nm,defi);
                    exit(8);
                }
                double vb=1e9; for(int vi=0;vi<nup&&vi<12;vi++) if(vbare[vi]>0&&vbare[vi]<vb) vb=vbare[vi];
                int bi=usel;
                for(int vi=0;vi<nup&&vi<12;vi++)
                    if(vbare[vi]>0&&vbare[vi]<=vb+0.001&&ups[vi].vol<ups[bi].vol) bi=vi;
                usel=bi; brel=vbare[bi]; Msel=vbare[bi];
                fprintf(stderr,"L%02d ★不可达层(req=%.4f 档梯最佳=%.4f 欠 %.4f) → 锁最小同质档 %s(%.4f bpw)★\n",
                        L,M_req,vb,vb-M_req,ups[usel].nm,ups[usel].vol);
                up_t*cs=&ups[usel];   /* 重臂选中档(走梯后续档覆盖过全局态) */
                if(cs->hot) hot_from_anchor(L,S,cs->hot); else GO2B_HOT=0;
            }
            static int rb_zstreak=0, rb_skip=0;
            if(RB_SEQ&&(!mv_base||getenv("DS4_ROUTE_SEQ_BASE"))&&rb_skip>0){
                /* ★自适应跳扫(2026-07-30 用户"每层太慢"): 深带 α 连选 0(v21 实测 L19+ 全 0,
                 * 每层白烧 460-530s) → 连 2 层 α=0 后跳 2 层再复测 */
                rb_skip--; RB_ALPHA_L[L]=0;
                fprintf(stderr,"L%02d [路由] 自适应跳过(前层连α=0, 余%d层后复测)\n",L,rb_skip);
            }
            else if(RB_SEQ&&(!mv_base||getenv("DS4_ROUTE_SEQ_BASE"))){   /* ★固定档也可序贯(2026-08-14 用户令"每层路由反修") */
                /* ★序贯路由: FIT 本层(胜者态一次前向) → α 三点扫(负收益自动关) →
                 *   之后的定稿/链推进即带路由修正, 下游层继承。 */
                up_t*cw=&ups[usel];
                if(W.t2ei){   /* ★哈希路由层(L0-L2)零漂移, α扫无对象直接跳(2026-08-14 自检针实锤48min/层白烧) */
                    RB_ALPHA_L[L]=0;
                    fprintf(stderr,"L%02d [路由] 哈希层跳过\n",L);
                } else {
                int co3=COADAPT; COADAPT=0;
                g_rb_fit_L=L; RB_ALPHA_L[L]=0;
                time_t trb=time(NULL);
                memcpy(Hw,H,lstride*4); set_cand(&MV_G10);
                lstat_t stq; memset(&stq,0,sizeof(stq));
                fprintf(stderr,"L%02d [路由FIT] ",L);
                layer_fwd(L,&W,Hw,ids,S,n_fit,1,cw->cfg,&stq);
                g_rb_fit_L=-1;
                rb_commit(L);
                /* ★α=0 并入扫点(2026-08-14): 复用模式 brel=0 判据失真, 基线由扫点自带, 取 argmin */
                double bestr=1e30; float besta=0.0f;
                const float ASCAN[3]={0.0f,2.5f,4.0f};
                for(int ai=0;ai<3;ai++){
                    RB_ALPHA_L[L]=ASCAN[ai];
                    memcpy(Hw,H,lstride*4);
                    lstat_t sta; memset(&sta,0,sizeof(sta));
                    layer_fwd(L,&W,Hw,ids,S,n_fit,1,cw->cfg,&sta);
                    double ra; held_score(Hw,S,n_fit,L,&ra);
                    fprintf(stderr,"L%02d [α扫] α=%.1f relh=%.4f%s\n",L,ASCAN[ai],ra,ra<bestr?" ↓":"");
                    if(ra<bestr){ bestr=ra; besta=ASCAN[ai]; }
                }
                RB_ALPHA_L[L]=besta; COADAPT=co3;
                }
                fprintf(stderr,"L%02d [路由] 选 α=%.1f bestr=%.4f\n",L,RB_ALPHA_L[L],RB_ALPHA_L[L]>0?0.0:0.0);
                if(RB_ALPHA_L[L]<=0.0f){ if(++rb_zstreak>=2) rb_skip=2; } else rb_zstreak=0;
            }
            int coadapted=(COADAPT>0&&(!mv_base||getenv("DS4_MV_COAD_BASE")));
            if(coadapted){   /* 胜者唯一一次 coadapt 定稿(全局态=胜者档已置位; DS4_MV_COAD_BASE=固定档也定稿(R24)) */
                up_t*c=&ups[usel];
                time_t tf1=time(NULL);
                memcpy(Hw,H,lstride*4); set_cand(&MV_G10);
                lstat_t stf2; memset(&stf2,0,sizeof(stf2));
                fprintf(stderr,"L%02d %s定稿(coadapt) ",L,c->nm);
                layer_fwd(L,&W,Hw,ids,S,n_fit,1,'g',&stf2);
                double relf2,s2=held_score(Hw,S,n_fit,L,&relf2); (void)s2;
                fprintf(stderr," held relL2=%.4f [定稿 %lds]\n",relf2,(long)(time(NULL)-tf1));
                brel=relf2;
            }
            { up_t*c=&ups[usel];
              snprintf(win_nm,sizeof(win_nm),"%s",c->nm); win_cfg=c->cfg; win_r=c->r; win_hot=c->hot; win_vol=c->vol; }
            if(!RB_SEQ&&getenv("DS4_ROUTE_BIAS_FIT")) rb_commit(L);   /* 固定档模式: 本层 Δb 折算入表(应用与否由 DS4_ROUTE_BIAS_ALPHA 定, 默认 1.0 即链上自带修正) */
            g_mv_pcfg=win_cfg; g_mv_phot=win_hot;   /* ★热启动记录: 下层档梯起点 */
            mv_vol_note(L,win_nm,win_vol);
            if(H2t){ memcpy(H2,H2t,lstride2*4); free(H2t); }
            mprev=Msel;
            if(!pure){
            memcpy(Hb,Hw,lstride*4);
            fprintf(stderr,"L%02d ★贪心选 %-6s held=%.4f w1w3bpw=%.4f rrM=%.4f\n",L,win_nm,brel,win_vol,mprev);
            }
            /* ⑤ 产物=两文件: dql(1bit w2+全记录+z + 热档:内嵌 g2hot) + opt */
            if(getenv("DS4_LAYER_DIR")){
                char dst[512];
                snprintf(dst,sizeof(dst),"%s/dql_L%02d.bin",getenv("DS4_LAYER_DIR"),L);
                if(!coadapted){   /* ★纯VQ量化(2026-08-03 用户令"只要vq量化"): coadapt 未跑 ⇒ 自建
                                   * 1bit 基座记录(否则 dql 零记录无载荷区), 并防上层 ELE 残留跨层泄漏 */
                    NELE=0;
                    el_add("1bit","纯VQ量化(计划表档位, coadapt未跑)",
                           (uint64_t)NEXP*(2*(uint64_t)MOEI*go1b_blk_row_bytes(DIM)+(uint64_t)DIM*go1b_blk_row_bytes(MOEI)),
                           brel,brel,0,0,1);
                }
                export_layer_file(L,S,n_fit,dst);
                lfile_t lfq; memset(&lfq,0,sizeof(lfq));
                if(lfile_load(dst,&lfq)==0){ zc_opt_emit(L,&lfq); lfile_free(&lfq); }
                if(pure){   /* ★纯VQ·导出先行(2026-08-03 用户令"只要vq量化"): 链态=盘上字节 B 回放
                             * (部署口径, 评估/导出失配根除; 省一遍全层 VQ 编码 — 裸评已撤) */
                    time_t tb0=time(NULL);
                    memcpy(Hw,H,lstride*4);
                    lstat_t stb; memset(&stb,0,sizeof(stb));
                    layer_fwd(L,&W,Hw,ids,S,n_fit,1,'B',&stb);
                    double relb; held_score(Hw,S,n_fit,L,&relb);
                    brel=relb;
                    memcpy(Hb,Hw,lstride*4);
                    /* ★pure 模式路由 FIT 补 commit(2026-08-03 实锤: 主流程 rb_commit 在本层任何前向
                     * 之前跑 ⇒ RB_ACC 空转 ⇒ Δb 侧车永不落盘, 反修 α2.5 静默失效): B 回放的
                     * 学生路由统计已累计, 此处 commit 本层真值。 */
                    if(!RB_SEQ&&getenv("DS4_ROUTE_BIAS_FIT")) rb_commit(L);
                    fprintf(stderr,"L%02d ★贪心选 %-6s held=%.4f w1w3bpw=%.4f rrM=%.4f [B回放·部署口径 %lds]\n",
                            L,win_nm,brel,win_vol,mprev,(long)(time(NULL)-tb0));
                }
                fprintf(stderr,"L%02d 产物落盘 %s(dql%s + opt) → %s\n",
                        L,win_nm,win_hot?"·内嵌g2hot":"",dst);
                /* ★体积账(2026-07-31 用户令"多大就是多大, 后面你能理得清哪些要哪些不要吗")★
                 * 进 GGUF 的字节 = VQ 侧车文件本身(dql 是 1bit 基座+记录, VQ 路线下不进模型)。
                 * 每层当场量、当场累加、当场落 manifest —— 合并端直接读这份账, 不再事后反解
                 * 段结构(vq_blob_truesize.py 那种反解器认不全新档位, 曾低估 833MiB → 截断载荷)。
                 * 超预算立刻停: 跑完才发现超 11.7% 的事不能再来一次。 */
                if(dq_vq_on()){
                    char vp[512]; vq_sidecar_path(dst,L,vp,sizeof(vp));
                    struct stat vs;
                    if(stat(vp,&vs)==0){
                        static double acc_gib=0; static FILE*mf=NULL;
                        acc_gib += vs.st_size/1073741824.0;
                        double bud = getenv("DS4_VOL_BUDGET_GIB")?atof(getenv("DS4_VOL_BUDGET_GIB")):0.0;
                        fprintf(stderr,"L%02d ★体积★ 本层 %.1f MiB | 累计 %.3f GiB%s%s\n",
                                L,vs.st_size/1048576.0,acc_gib,
                                bud>0?" / 预算 ":"", bud>0?({static char b[32];snprintf(b,32,"%.2f GiB",bud);b;}):"");
                        const char*md=getenv("DS4_LAYER_DIR");
                        if(md){ char mp[512]; snprintf(mp,sizeof(mp),"%s/manifest.txt",md);
                            if(!mf) mf=fopen(mp,L==0?"w":"a");
                            if(mf){ fprintf(mf,"%d %lld\n",L,(long long)vs.st_size); fflush(mf); } }
                        if(bud>0&&acc_gib>bud){
                            fprintf(stderr,"★★体积闸: 累计 %.3f GiB > 预算 %.2f GiB — 停止(计划表需降档)★★\n",
                                    acc_gib,bud);
                            exit(9); }
                    }
                }
            }
            if(H2) ckpt2_save(L,H2,lstride2);
            GO2B_HOT=0;   /* 层间卫生: 热态不跨层泄漏 */
            snprintf(sel_nm,sizeof(sel_nm),"%s",win_nm); sel_cfg=win_cfg;
        } else {
        int bfree=-1,b2=-1; double sfree=1e300,s2v=1e300,rfree=0,r2v=0;
        for(int c=0;c<NCAND;c++){
            memcpy(Hw,H,lstride*4);
            set_cand(&CANDS[c]);
            lstat_t st; memset(&st,0,sizeof(st));
            fprintf(stderr,"L%02d %-4s ",L,CANDS[c].name);
            layer_fwd(L,&W,Hw,ids,S,n_fit,1,CANDS[c].cfg,&st);
            double relh,sc=held_score(Hw,S,n_fit,L,&relh);
            fprintf(stderr," held relL2=%.4f score=%.5g\n",relh,sc);
            go1b_joint_neg=go1b_joint_clip=go1b_joint_fail=go1b_joint_rows=0;
            if(CANDS[c].cfg=='2'){ if(sc<s2v){s2v=sc;r2v=relh;b2=c;memcpy(Hf2,Hw,lstride*4);} }
            else if(sc<sfree){ sfree=sc; rfree=relh; bfree=c; memcpy(Hb,Hw,lstride*4); }
        }
        /* 预算门: q2 只有相对免费最优挽回 ≥ gain_th 才升位 */
        best=bfree; brel=rfree;
        double gain=(rfree>0)?(rfree-r2v)/rfree:0;
        if(b2>=0 && gain>=gain_th){ best=b2; brel=r2v; memcpy(Hb,Hf2,lstride*4); }
        fprintf(stderr,"L%02d ★选 %-4s held=%.4f (q2升位收益=%.1f%% 门槛%.0f%%)\n",
                L,TAB[best].name,brel,gain*100,gain_th*100);
        snprintf(sel_nm,sizeof(sel_nm),"%s",TAB[best].name); sel_cfg=TAB[best].cfg;
        }
        memcpy(H,Hb,lstride*4);
        plan_out[L]=sel_cfg;
        if(minvol) plan_append_rr(L,sel_nm,brel,mprev); else plan_append(L,sel_nm,brel);
        ckpt_save(L,H,lstride,idh);
        free_layer(&W);
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
