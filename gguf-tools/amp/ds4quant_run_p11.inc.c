    for(int L=0;L<NLAYERS;L++){
        int pc=-1; char sel_nm[16]="?"; char sel_cfg=0;
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
            int mv_base=g_cli.mv_baseline;   /* 基线模式: 全链冠军配方(g10h), 产 BT */
            int coad_save=COADAPT; COADAPT=0; GO2B_HOT=0;   /* 基线/纯m档必须无热污染 */
            /* (热启动 warm/影子冠军链/门A历史线三条实验路已删: DS4_MINVOL_HIST 族
             *  2026-08-31 env 清退, 含影子 ckpt 与门线派生; 地板线开关写死 DSQ_MV_FLOOR_LINE) */
            double rel0=0,s_base=0,relh_floor=-1; (void)relh_floor; (void)s_base;
            if(!mv_base){
            time_t tc0=time(NULL);
            if(DSQ_MV_FLOOR_LINE) hot_from_anchor(L,S,MV_HOTK); else GO2B_HOT=0;
            memcpy(Hw,H,lstride*4); set_cand(&MV_G10);
            lstat_t st0; memset(&st0,0,sizeof(st0));
            fprintf(stderr,"L%02d %s ",L,DSQ_MV_FLOOR_LINE?"g10h16地板(裸)":"g10 ");
            layer_fwd(L,&W,Hw,ids,S,n_fit,1,'g',&st0);
            s_base=held_score(Hw,S,n_fit,L,&rel0);
            if(DSQ_MV_FLOOR_LINE){ relh_floor=rel0; GO2B_HOT=0; }
            fprintf(stderr," held relL2=%.4f score=%.5g [%s %lds]\n",rel0,s_base,
                    DSQ_MV_FLOOR_LINE?"地板":"裸基线",(long)(time(NULL)-tc0));
            memcpy(Hb,Hw,lstride*4);
            }
            char win_nm[16]="g10"; char win_cfg='g'; int win_r=0,win_hot=0;
            double win_vol=1.0625; brel=rel0;
            if(mv_base){   /* 基线模式: 逐层直接冠军配方 g10h(不粗筛不建流形), 产同尺 BT */
                if(g_cli.vq_rplan&&dq_vq_on()){
                    /* ★R28: 计划表驱动(每层档位+热数); 热集按本层 hot 数从锚重建 */
                    vq_rplan(L);
                    hot_from_anchor(L,S,g_vq_hot);
                    snprintf(win_nm,sizeof(win_nm),"v%dx%d h%d",g_vq_dim,g_vq_nc,g_vq_hot);
                    win_cfg='g'; win_r=0; win_hot=g_vq_hot;
                    /* ★体积账必须混合热冷★(2026-08-27 修): 原式只算冷档 bpw, 热专家当不存在。
                     * 对照非计划表分支 (16*2.25+240*1.0625)/256 —— win_vol 的语义本就是
                     * 全 256 专家混合后的每权重 bpw。热档比冷档贵时低报, DS4_VOL_BUDGET_GIB
                     * 闸就失灵; r64 战役当年正是栽在同类口径错位(01:18 体积闸自绊 rc=9,
                     * 已量化 32 层作废重来)。用共享字节账函数算, 不再手抄公式。 */
                    { size_t hb=vq_payload_bytes(MOEI,DIM,vq_hot_dim(),vq_hot_nc());
                      size_t cb=vq_payload_bytes(MOEI,DIM,g_vq_dim,g_vq_nc);
                      win_vol=((double)g_vq_hot*hb+(double)(256-g_vq_hot)*cb)*8.0
                              /(256.0*(double)MOEI*DIM); }
                } else {
                snprintf(win_nm,sizeof(win_nm),"g10h"); win_hot=MV_HOTK;
                win_vol=(16*2.25+240*1.0625)/256.0;
                }
            } else {
            fprintf(stderr,"[minvol] 贪心 rank 二分已随流形档移除 — 用 --mv-baseline + --vq-rplan 计划表模式\n");
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
            {
              up_t w0; memset(&w0,0,sizeof(w0));
              snprintf(w0.nm,16,"%s",win_nm); w0.cfg=win_cfg; w0.r=win_r; w0.hot=win_hot; w0.vol=win_vol;
              ups[nup++]=w0;
              /* 升档梯: g10 → ★g10hN 往上翻(2026-07-30 用户令): 超标不卡死体积,
               * 活跃专家扩热处理, 债不再传链。 */
              up_t lad[1]={{"g10",'g',0,0,1.0625}};
              for(int u=0;u<1;u++) if(lad[u].vol>win_vol+1e-9&&nup<8) ups[nup++]=lad[u]; }
            const double MV_EPS=DSQ_MINVOL_EPS;
            /* ★两遍调度(2026-07-29 用户"快速确定最小体积, 精修选中层")★
             * DSQ_MINVOL_FLOOR=0(快扫遍): 跳地板/碑/升档, 粗筛胜者直接定稿导出(~8min/层);
             * =1(精修遍, 写死默认): 逐层地板门(对 RESUME 选中的层生效)。 */
            const int MV_FLOOR=DSQ_MINVOL_FLOOR;
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
            const double MV_SOFT=DSQ_MINVOL_SOFT;   /* 纯防灾(v10 实测冠军配方浅层每层耗~0.046, 门不误咬正常层) */
            /* 里程碑评估已删除(2026-08-18 用户令): RESUME 首层不再跑后缀 FP 探针
             * (单线程 38 层前向 ~12 分钟)。mprev 保持 -1 = 无碑语义, 软门首层跳过。 */
            /* (门A历史链/影子线/增长率门整族已删: DS4_MINVOL_HIST 族 2026-08-31 env 清退) */
            double BT=mv_target;
            double M_req = mv_base?-1e9:
                (MV_FLOOR? (M_champ-MV_EPS)
                    : (H2? ((L>=NLAYERS-1)?BT:(BT+((mprev<0?1.0:mprev)-BT)*(double)(NLAYERS-1-L)/(double)(NLAYERS-L)))
                         : ((mprev<0?1.0:mprev)-MV_SOFT)));
            if(!MV_FLOOR&&nup>3) nup=3;   /* 软门: 至多三试 */
            float *H2t=H2?malloc(lstride2*4):NULL;
            int usel=-1; double Msel=mprev; int mv_pass=0;
            /* (近失 coadapt 复判窗 MV_CWIN/g_mv_gain 已删: 门A历史链独占机制, 随 hist 清退) */
            int pure=(mv_base&&g_cli.pure_vq)?1:0;
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
                else if(MV_FLOOR&&is_champ) M=M_champ;  /* 地板门模式顶格免测 */
                else if(H2t){ rr_step(L,&W,H2,H2t,S,n_fit,c->cfg); M=rr_probe(H2t,L); }
                else M=restore_probe(Hw,L,ids,S,n_fit);   /* ★判决=v5 语料 HELD 区(裸态) */
                COADAPT=co2;
                int pass = mv_base || (M>=M_req);
                if(!mv_base)
                fprintf(stderr,"L%02d [%s] %-6s 裸M=%.4f relh=%.4f 门=%.4f(%s) [%lds] → %s\n",
                        L,MV_FLOOR?"地板门|v5held":(H2t?"预算门|rr":"软门|v5held"),c->nm,M,relf,M_req,
                        MV_FLOOR?"地板−EPS":(H2t?"历史指标线":"上层−SOFT"),
                        (long)(time(NULL)-tf0),pass?"PASS":"升档");
                if(pass){ usel=u; Msel=M; brel=relf; mv_pass=1; break; }
                usel=u; Msel=M; brel=relf;
                if(!MV_FLOOR&&u+1<nup&&u+1<12){
                    /* ★自适应升档: 'g' 带往上翻(2026-07-30 用户令): 活跃专家按锚路由权重扩热
                     * (go2b 2.25) — 体积不卡死, 债不传链。软门下 g10 顶格即记债。 */
                    if(c->cfg=='g'){
                        nup=u+1; fprintf(stderr,"L%02d [升档] g10 已顶格(软门) — 记债交 backfit\n",L);
                    }
                }
            }
            /* (不可达层锁档/序贯路由 α 扫两块已删: 门A历史链与 DS4_ROUTE_SEQ 独占机制,
             *  2026-08-31 env 清退; mv_pass 收官后只走地板门/软门语义) */
            (void)mv_pass;
            int coadapted=(COADAPT>0&&(!mv_base||DSQ_MV_COAD_BASE));
            if(coadapted){   /* 胜者唯一一次 coadapt 定稿(全局态=胜者档已置位; DSQ_MV_COAD_BASE=固定档也定稿(R24)) */
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
            if(g_cli.route_bias_fit) rb_commit(L);   /* 固定档模式: 本层 Δb 折算入表(应用与否由 --route-bias-alpha 定, 默认 1.0 即链上自带修正) */
            g_mv_pcfg=win_cfg; g_mv_phot=win_hot;   /* 档位记录: 体积账口径用 */
            mv_vol_note(L,win_nm,win_vol);
            if(H2t){ memcpy(H2,H2t,lstride2*4); free(H2t); }
            mprev=Msel;
            if(!pure){
            memcpy(Hb,Hw,lstride*4);
            fprintf(stderr,"L%02d ★贪心选 %-6s held=%.4f w1w3bpw=%.4f rrM=%.4f\n",L,win_nm,brel,win_vol,mprev);
            }
            /* ⑤ 产物=两文件: dql(1bit w2+全记录+z + 热档:内嵌 g2hot) + opt */
            if(g_cli.layer_dir){
                char dst[512];
                snprintf(dst,sizeof(dst),"%s/dql_L%02d.bin",g_cli.layer_dir,L);
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
                    if(g_cli.route_bias_fit) rb_commit(L);
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
                        /* (体积闸 DS4_VOL_BUDGET_GIB 已删 2026-08-31: 账照记照报, 停不停由人/脚本读 manifest 裁) */
                        fprintf(stderr,"L%02d ★体积★ 本层 %.1f MiB | 累计 %.3f GiB\n",
                                L,vs.st_size/1048576.0,acc_gib);
                        const char*md=g_cli.layer_dir;
                        if(md){ char mp[512]; snprintf(mp,sizeof(mp),"%s/manifest.txt",md);
                            if(!mf) mf=fopen(mp,L==0?"w":"a");
                            if(mf){ fprintf(mf,"%d %lld\n",L,(long long)vs.st_size); fflush(mf); } }
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
        /* ★层边界把空闲堆还给系统★(2026-08-27 两次 OOM 定位)
         * 症状: 逐层配置(每层不同 dim/nc)跑到第 26-30 层耗时开始翻倍(5s→38s), 最终被
         * 内存看门狗停车; 平权配置(43 层同档)跑 43 层从不复现。
         * 机制: mallopt(M_MMAP_THRESHOLD, 1GiB)(p14, 为压 mmap 写锁争用而设 —— 默认小阈会
         * 让每层重读 HF、线程全 D 态等 IO、9 分/层实测)副作用是 glibc 不把大块还给系统。
         * 层间档位一变, 工作缓冲尺寸就变(nc=256 与 nc=1024 差 4 倍), 旧尺寸的空闲块无法复用,
         * 在 20 个线程 arena 里越堆越多, 攒够六七层就挤爆 page cache → HF 读盘退化 → 停车。
         * 修法: 层内保持大阈值(性能要它), 只在层边界(此处已 free_layer, 无在飞分配)显式归还。
         * 零数值影响: malloc_trim 只动分配器簿记, 不碰任何权重/中间量。 */
#ifdef __linux__
        malloc_trim(0);
#endif
