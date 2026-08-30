#if defined(__linux__)
#include <malloc.h>   /* malloc_trim(arena 归还); Darwin 无此函数→空宏(mac 只编不跑) */
#else
#define malloc_trim(x) ((void)0)
#endif
/* ★分块 ONEPASS(2026-08-27 重新实现)★
 * 冠军 r64 用的是 DS4_BF_CHUNK=7, 但那份 C 实现【从未进过 git】(只活在当时的
 * ds4quant_run.dchunk 二进制里, 已佚), fable5 4985 留着完整设计, 照它重写。
 *
 * 【为什么必须有】纯 ONEPASS 是"冻结基线 Jacobi": 每层都在"别层不变"的假设下单独评估,
 * 各自都是正收益, composed 起来却互相打架。2026-08-27 实撞: L41(bf.GL α=1.2 −2.38%)
 * + L40(bf.GL α=0.8 −2.00%) + L39(bf.GLdyn8 −5.09%) 三个都是真收益, 到 L38 基线
 * 0.4302→34.8462 炸 80 倍, 链闸硬停。fable5 6497 记过同款(每层×2 正反馈 L4 0.48→L10 13.2)。
 *
 * 【怎么修】按 CH 层一块: 块内保持原 ONEPASS 语义(冻结基线选型), 块末做终验
 * ——★终验走真前沿全程出口, 不是块边界★, 判据不变; 过则提交并刷新 HQE(下块在新上下文
 * 里评估 = Gauss-Seidel), 败则只回滚本块。这样坏组合最多污染一块, 且后面的块看到真实状态。
 * CH 写死不走 env(铁律 08-22 禁新增 env; 冠军值 7)。 */
static int backfit_prev_chunk(int Jlo_in,int Jhi,int Lfront,const long*ids,int S,int n_fit){
    if(!GS_LW||!GS_LF||!HQE||Lfront<1||Jhi<=Jlo_in) return 0;
    size_t lstride=(size_t)S*HCM*DIM, rowsz=(size_t)HCM*DIM;
    const float*Htgt=ANC.H+(size_t)Lfront*lstride;
    int vs=(n_fit*3)/4, changed=0, evald=0; double simpr=0;
    double bf_bestdl=-1e300; int bf_bestJ=-1;   /* 可视化: 全部候选里最接近正向的Δ(没提交也看得见搜索) */
    char uc[300]; int ucl=0; uc[0]=0;           /* 未正向层清单(层号+最优候选Δ), 铁律: 不许隐身 */
    /* ★用户裁决2026-07-12★: 每前沿全量反修所有前层("末层最后一次全量"特例取消; fast 不进本函数) */
    int Jlo=Jlo_in;
    time_t bt0=time(NULL);
    if(!GS_FIN) GS_FIN=malloc((size_t)S*DIM*4);
    const int NA=FAST?2:4;
    const float AGf[2]={0.94f,1.06f}, AGu[4]={0.85f,0.94f,1.06f,1.15f};
    /* ★禁跨遍累乘(2026-08-07 用户裁决, 级联62倍定谳): GL 全部候选=绝对值 — 每遍以
     * g=1(量化基线)为原点在网格里重选优, 落地值天然在设计界 [0.85,1.15] 内, 语义与遍数
     * 无关(可复现), 历史越界值(旧 baseg×a 复利滚到 1.44)在重解时被自动清洗进界。 */   /* 2026-07-13: ±15% 探针(深层需要大动作, 原±10%把靶截断) */
    const float*AG=FAST?AGf:AGu;
    double *db=malloc((size_t)S*sizeof(double));
    double *dg=malloc((size_t)NA*S*sizeof(double));
    float  *ms=malloc((size_t)S*4);
    float  *kkrow=malloc((size_t)S*4), *effw=malloc((size_t)S*4);   /* 分类/感知行权(2026-08-05) */
    /* ★两级评分(2026-07-13 根因手术)★: 旧式=每候选 全深度(J→前沿)×全token 重放, 单元~14次重放
     * 且成本∝(F−J) → 每前沿 O(F²)、全程 O(L³)(实测 ≈0.40·F² min/前沿, 43层≈7天)。
     * 粗筛: 候选比价改在近视野出口 Fj=min(J+BK,Lfront) × token 抽格(fit区/打分区各按 SDIV 抽,
     * 保 train/val 切分)上进行 — 只定形态排序; 落地闸门不变: 胜者按真前沿×全token 复核必须净降。
     * 质量最坏情形=多几个"保持", 判据口径与旧全量一致, 永不反向。
     * DS4_BF_SCREEN_K=0 回旧全量行为; DS4_BF_SCREEN_DIV=1 只截深度不抽token。 */
    /* ★写死不走 env(2026-08-29 用户令"不要环境变量控制逻辑, 每次都这样不是丢了吗")★
     * 原来这五个都是 getenv+默认值。危害不是"可配", 是【C 默认与脚本实际值不一致】:
     * SCREEN_DIV 的 C 默认是 4 而 r30_campaign.sh 一直导出 12 —— 不经脚本直接跑二进制,
     * 抽格密度就悄悄变了。同款事故今天已撞两次(LZRANK 未设静默兜底 16 / 行掩码写死 32×256)。
     * 取值 = 脚本一直在用的那一组, 行为逐位不变。 */
    int BK=8;   /* 后面粗筛关闭时会置 0, 故不 const */
    const int SDIV=12;   /* r30_campaign.sh 一直导出的值(C 旧默认 4 与之不符) */
    /* ★2026-08-02: 原来这里有 "DS4_ANCHOR_ROUTE ⇒ BK=0 硬关粗筛"。已删 —— 不相容的根源
     * (锚按原始行号、抽格前向按紧凑行号)由 g_anc_rowmap 行映射解决了。关粗筛的代价是
     * 本文件自己注明的 "旧式全量 ≈0.40·F² min/前沿, 43 层≈7 天", 交付不了。 */
    int Ss=0,vss=0,vrej=0; int*sidx=NULL; long*ids_s=NULL; float*Hin_s=NULL,*Ht_s=NULL;
    /* ★行域 = 布局分层, 不是行号区间(2026-08-29)★
     * 旧写法 fit=[0,vs) / 打分=[vs,n_fit)。抽样把 8 个域【连续】铺进 8192 行(实测
     * [0,4096) 拉丁 / [4096,5120) 西里尔 / [6144,7168) 阿拉伯 / [7168,8192) 中日韩),
     * 于是打分区 [4608,6144) 只覆盖【西里尔后半+math】两个域 —— sweep 的六候选择优、
     * 逐层落地、连同最后的统一终验, 全部只看 2/8 个域。上一轮 42 层"✓正向落地"+终验
     * "✓改善→提交", 判决份 8 域尺上四项全负, 就是这么来的。
     * 布局由 <锚>.layout 给(amp_campaign.sh 抽样时落盘 → anchors3 随锚落一份),
     * 从【本进程已经拿到的锚路径】推导, 不经任何开关。读不到=硬停, 不许猜。 */
    int *LFIT=NULL,*LEV=NULL,nLFIT=0,nLEV=0;
    if(row_layout_split(anchor_path(),&LFIT,&nLFIT,&LEV,&nLEV)!=0){
        fprintf(stderr,"★sweep 停车: 行布局缺 %s.layout★\n",anchor_path());
        fprintf(stderr,"  补法: bash gguf-tools/scripts/amp_campaign.sh idshalf (只补布局, 逐字节校验, 不动 ids)\n");
        exit(9); }
    fprintf(stderr,"[sweep] 行域←%s.layout: fit %d 行 / 打分 %d 行(跨全域)\n",anchor_path(),nLFIT,nLEV);
    /* ★sidx 必须升序(2026-08-29 自审第 4 bug)★: 抽格前向的滑窗 attention 按【行下标】判
     * 因果(vq_gpu_attn:24 `n<=s && n>s-WIN`)。我上一版把 fit 抽格(0..8100)和 eval 抽格
     * (800..8191)首尾相接 —— 拼接处时序回跳, attention 因果全乱, 粗筛分数失真。
     * 修=两路各自升序、归并进 sidx; "哪些是打分行"不再靠"前段/后段"位置, 改由 EVc
     * (eval 行在紧凑前缀中的下标表)表达, 拟合侧同理用 FITc。 */
    int *FITc=NULL,*EVc=NULL,nFITc=0,nEVc=0;
    if(BK>0){
        sidx=malloc(sizeof(int)*(size_t)(nLFIT+nLEV+1));
        FITc=malloc(sizeof(int)*(size_t)(nLFIT+1)); EVc=malloc(sizeof(int)*(size_t)(nLEV+1));
        { int i=0,j=0;                                   /* 两路抽格归并(各自升序) */
          while(i<nLFIT||j<nLEV){
              const int a=i<nLFIT?LFIT[i]:0x7fffffff, b=j<nLEV?LEV[j]:0x7fffffff;
              if(a<=b){ FITc[nFITc++]=Ss; sidx[Ss++]=a; i+=SDIV; }
              else    { EVc[nEVc++]=Ss;  sidx[Ss++]=b; j+=SDIV; } } }
        vss=nFITc;   /* 旧变量只剩守卫在用 */
        if(Ss<8||nEVc<2){ free(sidx); sidx=NULL; free(FITc); free(EVc); FITc=EVc=NULL; BK=0; Ss=0; }   /* 校准太小: 粗筛无意义 */
        else{ ids_s=malloc(sizeof(long)*(size_t)Ss);
              for(int s=0;s<Ss;s++) ids_s[s]=ids[sidx[s]];
              Hin_s=malloc((size_t)Ss*rowsz*4); Ht_s=malloc((size_t)Ss*rowsz*4); }
    }
    /* ★复核专用前缀(2026-08-29 二修)★: 第一版复核打分用 EVc(抽格 128 行), 标准误 ~8.8%,
     * 而层级增益只有 0.02-1.03% —— 128 行上的"改善"全是噪声(4 层链实测: 复核全放行,
     * 终验 1536 行上仍劣化 20% 全回滚)。历史注释早警告过这一点, 我重启复核时没听。
     * 修 = 打分行用【LEV 全集 1536 行】(与终验同行集 ⇒ 判据方向必然对齐), 前向行数
     * 1536+FIT抽格≈1920(升序归并, moe GPU 照常接管), 全程复核 ~40s×2/单元, 可负担。 */
    /* ★复核前缀(eval 1536+fit 384, co_score 判)已废(2026-08-30)★: 三层口径错位实锤后,
     * 放行判据改 held-KL 闸(bkl_*, p12), 行少(512)且判的是 head 后真尺 → 又快又对。 */
    if(sidx&&!bkl_init(Htgt,ids,S,n_fit,rowsz)) fprintf(stderr,"[BKL]★init 失败/拒臂 → 本 sweep 全部拒落地(fail-closed)★\n");
    /* ★同前沿复检★: sweep 高层→低层, 低层落地会刷新高层输入上下文(HQE), 但高层已评过 →
     * 第一遍有落地则对"保持"层用新上下文再重解一遍(pass=1), 堵"评估时机"漏 */
    g_anc_rowmap = NULL;   /* 进函数先清干净 */
    char *done=calloc((size_t)Lfront,1);
    double *jd=malloc((size_t)Lfront*sizeof(double));
    for(int J=0;J<Lfront;J++) jd[J]=-1e300;
    /* ★二分调度(2026-08-04 用户设计)★ DS4_BF_SWEEP_ORDER=bisect: 首遍不再线性全扫 —
     * 段栈取中点评估, 中点 Δbest≥DS4_BF_BISECT_SKIP(%, 默认0.05)或落地才细分两侧;
     * 平坦段整段剪枝(done=1 ⇒ 复检遍与收尾日志都不再访问, 时间真省)。
     * 复检遍(pass=1)不变: 线性只扫已评估且保持的层("最后收尾一次")。 */
    /* 二分 sweep 已实现但【不武装】(见下方 08-04 单测实账: 剪枝与零漏检不可兼得,
     * 质量门铁律优先) —— 写死 0, 要开就改这一行, 不留 env。 */
    const int bis_on=0;
    double bskip=0.05;
    int (*bseg)[2]=bis_on?malloc(sizeof(int[2])*(size_t)(2*Lfront+8)):NULL; int bsp=0;
    /* ★ONEPASS(2026-08-04 用户裁决: "一遍就够, 每层落地反复矫正=后面跑偏")★
     * 冻结基线 Jacobi 式: 单元判据=近视野粗筛(不跑每单元全程复核), 落地不刷 HQE(所有层
     * 在同一反修完成态上选型互不污染), 单 pass, 收尾一次全程终验 — 劣化=全回滚
     * (truncate 掉 append + BFU 表回写原地改 + lfile 重读)。sweep 7.8h → ~1.5h。 */
    /* ONEPASS = 2026-08-04 用户终裁"一遍就够"; r30_campaign.sh 一直导出 1 ⇒ 写死。 */
    const int onep=1;
    size_t *ofl=onep?calloc((size_t)Lfront,sizeof(size_t)):NULL;   /* 每层落地前文件长度 */
    char *oland=onep?calloc((size_t)Lfront,1):NULL;                /* 本 pass 落地标记 */
    if(onep){ BFU_ARM=1; NBFU=0;
        fprintf(stderr,"[反修] ONEPASS: 冻结基线一遍选型+统一终验(近视野判据, 无逐单元全程复核/无复检遍)\n"); }
    for(int pass=0;pass<2;pass++){
    if(onep&&pass==1) break;   /* 单 pass: 复检遍的活交回扫 */
    if(pass==1&&changed==0) break;   /* 第一遍零落地 → 上下文没变, 复检无意义 */
    /* ★判决前置(用户裁决 2026-07-21)★: DS4_BF_NO_RECHECK=1 跳过全量复检遍 —
     * 先出 rr 终判"验证有没有问题", 有问题再手动 backfit 定向补(实测账: 复检遍 ~2h
     * 换 ~2% 出口分, 判决才是定谳)。默认 0 = 旧行为。 */
    if(0){   /* 复检遍从不禁用(脚本从未设过 DS4_BF_NO_RECHECK); 开关删除 */
        fprintf(stderr,"[反修] DS4_BF_NO_RECHECK=1: 跳过复检遍 → 直接判决\n"); break; }
    /* 访问驱动器: linear=Lfront-1..0 递减(默认, 原行为); bisect(仅首遍)=段栈中点序 */
    int bis=(bis_on&&pass==0);
    int blo=Jlo,bhi=Jhi-1;   /* ★只扫本块; Lfront 别处仍指真前沿(终验/近视野都要它) */
    if(bis){ bsp=0; bseg[0][0]=Jlo; bseg[0][1]=Jhi-1; bsp=1; }
    int J=Jhi;
    for(;;){
        if(bis){
            if(bsp<=0) break;
            bsp--; blo=bseg[bsp][0]; bhi=bseg[bsp][1];
            if(blo>bhi) continue;
            J=(blo+bhi)/2;
        } else if(--J<Jlo) break;
        if(done[J]) continue;
        if(!GS_LF[J].map){ done[J]=1; jd[J]=0;
            mlog(J,"向前反修","层文件缓存缺失","跳过",0,"lfile未加载","探索中"); continue; }
        time_t ut0=time(NULL); double ut0d=vqt_now(); memset(g_bflt,0,sizeof g_bflt); g_bflt_on=1;   /* 单元计时(BFUNIT 可观测行)+段账臂(p3 bflt_print) */
        lfile_t*lf=&GS_LF[J];
        char pj[512]; op_host_path(J,pj,sizeof(pj));   /* ★平行架构: 反修落地只写 op 侧车, 量化 dql 不可触 */
        if(onep&&!ofl[J]){ struct stat fst; ofl[J]=stat(pj,&fst)==0?(size_t)fst.st_size:0; }   /* 回滚锚点 */
        const float*Hin=HQE+(size_t)J*lstride;
        /* 粗筛视野: e* 别名 = 本单元候选比价用的(出口层/输入/ids/行数/切分); scr=0 时即旧全量口径 */
        int Fj=(BK>0&&J+BK<Lfront)?J+BK:Lfront;
        int scr=(BK>0)&&(Fj<Lfront||Ss<S);
        int eF=scr?Fj:Lfront, eS=scr?Ss:S, eNf=scr?Ss:n_fit, eVs=scr?vss:vs;
        /* 打分行表: 紧凑前缀里的 eval 行(scr=0 全量行时=LEV 原始行号)。拟合视图见 ms 之后。 */
        const int *SEV=scr?EVc:LEV; const int nSEV=scr?nEVc:nLEV;
        int eFlog=0;   /* 日志口径: 复核放行后 base/bestsc 是全程(Lfront)分, 标签必须跟着换 */
        /* ★锚路由行映射★ 抽格前向期间让锚按原始行号取; scr=0 时置空 = 恒等(原行为)。 */
        g_anc_rowmap = scr ? sidx : NULL; g_anc_rowstride = S;
        const float*eHin=Hin; const long*eIds=ids; const float*eTgt=Htgt;
        if(scr){   /* gather: 抽格行凑紧凑前缀(入口态 + Fj 出口 FP 锚), 行序=sidx */
            for(int s=0;s<Ss;s++){
                memcpy(Hin_s+(size_t)s*rowsz,Hin+(size_t)sidx[s]*rowsz,rowsz*4);
                memcpy(Ht_s +(size_t)s*rowsz,ANC.H+(size_t)Fj*lstride+(size_t)sidx[s]*rowsz,rowsz*4);
            }
            eHin=Hin_s; eIds=ids_s; eTgt=Ht_s;
        }
        /* base: 粗筛出口分 + per-token 距离 + 捕获 Fin_J(dyn 特征用; 抽格行序与 ms/dg 一致) */
        GS_CAP_L=J;
        float*Hb=gs_forward_exit(J,eF,eHin,eIds,eS,eNf,NULL);
        GS_CAP_L=-1;
        double base=co_score_rows(Hb,eTgt,SEV,nSEV,rowsz);
        { GS_CAP_L=J; base=bf_base_gate(J,base,&Hb,eF,eHin,eIds,eS,eNf,eTgt,SEV,nSEV,rowsz); GS_CAP_L=-1; if(!isfinite(base)){ free(Hb); mlog(J,"向前反修","base非有限(诊断已打)","跳过",0,"NaN闸","探索中"); done[J]=1; jd[J]=0; g_bflt_on=0; continue; } }   /* L11 NaN 实案闸(p3 bf_base_gate) */
        bf_rowdist(Hb,eTgt,eS,rowsz,db); free(Hb);
        /* 行权系数 kk_s(2026-08-05 分类/感知布线): 行残差能量/行目标能量, clamp[0,4] */
        for(int s=0;s<eS;s++){ const float*b=eTgt+(size_t)s*rowsz; double tn=0;
            for(size_t i2=0;i2<rowsz;i2++) tn+=(double)b[i2]*b[i2];
            double kv2=db[s]/(tn+1e-20); if(kv2>4.0)kv2=4.0; kkrow[s]=(float)kv2; }
        /* ★联合损失(2026-08-05 用户终裁: 四损失+感知同一目标并存, 非网格选一)★
         * 行权恒开: effw=1+W_pc·kk(W_pc 全局常数, →0 逐位退化基线解) */
        { static float JWPC=0.35f;
          if(0)JWPC=0.35f;
          for(int s=0;s<eS;s++) effw[s]=1.0f+JWPC*kkrow[s]; }
        evald++;
        /* --- 形态A: 全局α(链末 bf.GL, 已有则临时累乘) + 顶点细搜(粗网格漏±1%级最优) --- */
        int fo=BF_FINOP[J]; float baseg=(fo>=0)?lf->ops[fo].g:1.0f;
        double scA=base; float aA=1.0f; double scg[4]={base,base,base,base};
        for(int gi=0;gi<NA;gi++){ float a=AG[gi]; int tmp=fo,undo=0;
            if(tmp>=0) lf->ops[tmp].g=baseg*a;
            else if(lf->nops<32){ tmp=lf->nops; memset(&lf->ops[tmp],0,sizeof(lop_t));
                lf->ops[tmp].type=1; lf->ops[tmp].g=a; lf->nops++; undo=1; }
            else break;
            float*Hg=gs_forward_exit(J,eF,eHin,eIds,eS,eNf,NULL);
            double sc=co_score_rows(Hg,eTgt,SEV,nSEV,rowsz); scg[gi]=sc;
            bf_rowdist(Hg,eTgt,eS,rowsz,dg+(size_t)gi*S); free(Hg);
            if(sc<scA){ scA=sc; aA=a; }
            if(undo) lf->nops--; else lf->ops[tmp].g=baseg;
        }
        { char gb[128]; int go=0; for(int gi=0;gi<NA;gi++) go+=snprintf(gb+go,(size_t)(128-go),"%.2f:%.5g ",AG[gi],scg[gi]);
          printf("[BFA] L=%02d base=%.6g 网格 %s→选α=%.3f\n",J,base,gb,aA); fflush(stdout); }
        { double av=bf_vertex(AG[0],scg[0],1.0,base,AG[NA-1],scg[NA-1]);
          if(av<0.80)av=0.80; if(av>1.20)av=1.20;
          if(fabs(av-1.0)>2e-3&&fabs(av-(double)aA)>2e-3){   /* 顶点≠已测点才补一枪 */
              int tmp=fo,undo=0;
              if(tmp>=0) lf->ops[tmp].g=baseg*(float)av;
              else if(lf->nops<32){ tmp=lf->nops; memset(&lf->ops[tmp],0,sizeof(lop_t));
                  lf->ops[tmp].type=1; lf->ops[tmp].g=(float)av; lf->nops++; undo=1; }
              if(tmp>=0){
                  float*Hg=gs_forward_exit(J,eF,eHin,eIds,eS,eNf,NULL);
                  double sc=co_score_rows(Hg,eTgt,SEV,nSEV,rowsz); free(Hg);
                  if(sc<scA){ scA=sc; aA=(float)av; }
                  if(undo) lf->nops--; else lf->ops[tmp].g=baseg;
              } } }
        /* --- 形态B: 向后 TREF t 重调(层内落地的 type-4 op, 判据升级为前沿出口) + 顶点细搜 --- */
        int to=-1; for(int i=lf->nops-1;i>=0;i--) if(lf->ops[i].type==4){ to=i; break; }
        double scB=base; float tB=0;
        if(to>=0){ float t0v=lf->ops[to].t; const float TG[2]={0.94f,1.06f}; double scT[2];
            for(int ti=0;ti<2;ti++){ lf->ops[to].t=t0v*TG[ti];
                float*Hg=gs_forward_exit(J,eF,eHin,eIds,eS,eNf,NULL);
                double sc=co_score_rows(Hg,eTgt,SEV,nSEV,rowsz); free(Hg); scT[ti]=sc;
                if(sc<scB){ scB=sc; tB=t0v*TG[ti]; }
                lf->ops[to].t=t0v; }
            double rv=bf_vertex(TG[0],scT[0],1.0,base,TG[1],scT[1]);
            if(rv<0.80)rv=0.80; if(rv>1.20)rv=1.20;
            if(fabs(rv-1.0)>2e-3&&(tB==0||fabs(rv-(double)(tB/t0v))>2e-3)){
                lf->ops[to].t=t0v*(float)rv;
                float*Hg=gs_forward_exit(J,eF,eHin,eIds,eS,eNf,NULL);
                double sc=co_score_rows(Hg,eTgt,SEV,nSEV,rowsz); free(Hg);
                if(sc<scB){ scB=sc; tB=t0v*(float)rv; }
                lf->ops[to].t=t0v; } }
        /* per-token 连续目标 α*_s(抛物线顶点; 全部重解形态共用; 行域=粗筛行) */
        for(int s=0;s<eS;s++){
            double mv=bf_vertex(AG[0],dg[s],1.0,db[s],AG[NA-1],dg[(size_t)(NA-1)*S+s]);
            if(mv<0.85)mv=0.85; if(mv>1.15)mv=1.15; ms[s]=(float)mv;   /* clamp=探针包络(随探针加宽) */
        }
        /* ★拟合视图(与升序 sidx 配套)★: 原语义"前 eVs 行=fit 行"在归并后不成立。
         * zrefit_w/bf_pca8/nmix/dyn2 的 mu·sd 都是【fit 行】上的统计/最小二乘 —— 与行序
         * 无关, 只与行集合有关 ⇒ 把 fit 行 gather 成小前缀喂它们, 接口一个不动。
         * scr=0(全量行, 罕见路径)保持旧口径 [0,vs)。 */
        int nFV = scr?nFITc:eVs;
        float *msf=ms, *FINf=GS_FIN, *effwf=effw;   /* scr=0: 原地前缀 */
        float *_fv=NULL;
        if(scr){ _fv=malloc((size_t)nFV*(DIM+2)*4);
            msf=_fv; effwf=_fv+nFV; FINf=_fv+(size_t)2*nFV;
            for(int i=0;i<nFV;i++){ const int r=FITc[i];
                msf[i]=ms[r]; effwf[i]=effw[r];
                memcpy(FINf+(size_t)i*DIM,GS_FIN+(size_t)r*DIM,(size_t)DIM*4); } }
        int nmix=0;
        for(int s=0;s<nFV;s++) if(fabsf(msf[s]-1.0f)>2e-3f) nmix++;
        /* --- 形态E(★真·机制★): 重解本层自有 z 系数(最后 type3(带V8)>type2>type1 op) —
         * 闭式拟合 c_new(x)=α*_s·c_old(x)(zrefit, 非扰动试探), β 信赖域{1,0.5,0.25} 全步过冲退半步 --- */
        double scE=base; int ze=-1; lop_t opE; float bE=0;
        memset(&opE,0,sizeof(opE));
        if(nmix>=1){
            for(int i=lf->nops-1;i>=0;i--) if(lf->ops[i].type==3&&lf->ops[i].V8){ ze=i; break; }
            if(ze<0) for(int i=lf->nops-1;i>=0;i--) if(lf->ops[i].type==2){ ze=i; break; }
            if(ze<0) for(int i=lf->nops-1;i>=0;i--) if(lf->ops[i].type==1){ ze=i; break; }
            if(ze>=0){
                lop_t old=lf->ops[ze];
                const float BB[3]={1.0f,0.5f,0.25f};
                /* ★联合损失单次求解(2026-08-05 用户终裁: 网格选一=割裂+选择噪声, 已废)★
                 * 对齐=主项 / 感知·分类=effw 行权恒开 / 固定·光滑=λ 结构(df 锚定恒等基线) */
                lop_t fit=old;
                zrefit_w(&fit,msf,nFV,FINf,effwf);
                for(int bi=0;bi<3;bi++){
                    op_blend(&lf->ops[ze],&old,&fit,BB[bi]);
                    float*Hg=gs_forward_exit(J,eF,eHin,eIds,eS,eNf,NULL);
                    double sc=co_score_rows(Hg,eTgt,SEV,nSEV,rowsz); free(Hg);
                    if(sc<scE){ scE=sc; opE=lf->ops[ze]; bE=BB[bi]; }
                    lf->ops[ze]=old;
                    if(scE<base-1e-9) break;           /* 已改善即止(基线语义) */
                }
            }
        }
        /* --- 形态F(2026-07-13): bf.GLdyn8 新建 — 引擎词表内最强 9-dof 动态族。
         * 结构性盲区修复(用户裁决: "保持"≠层最优, 是候选族维度不足): 此前 dyn8 族只在
         * 层里恰好已有 type3 op 时被形态E重解, 没有的层从未试过。方向=本单元捕获 Fin 的
         * PCA8(与层搜索同法); V8 内联进记录载荷 → lfile/opt/merge/引擎全链路已支持。 --- */
        double scF=base; lop_t opF; memset(&opF,0,sizeof(opF)); float*V8F=NULL;
        {
            int hasd8=0; for(int i=0;i<lf->nops;i++) if(lf->ops[i].type==3){ hasd8=1; break; }
            if(!hasd8&&nmix>=3&&lf->nops<32&&(V8F=bf_pca8(FINf,nFV))!=NULL){
                /* 联合损失单次求解(同形态E) */
                opF.type=3; opF.V8=V8F; opF.w8[0]=1.0f;      /* 起点=恒等, zrefit 9-dof 岭回归拟 per-token 靶 */
                zrefit_w(&opF,msf,nFV,FINf,effwf);
                int tmp=lf->nops; lf->ops[tmp]=opF; lf->nops++;
                float*Hg=gs_forward_exit(J,eF,eHin,eIds,eS,eNf,NULL);
                scF=co_score_rows(Hg,eTgt,SEV,nSEV,rowsz); free(Hg);
                lf->nops--;
            }
        }
#include "ds4quant_sweep_arms.inc.c"   /* 形态C(dyn2)/形态D(GE投影) 候选臂(500 行守卫所迫的物理分片) */
        double jdl;   /* 本层最优候选Δ%(负=候选都更差) */
        { double cand=scE; if(scF<cand)cand=scF; if(scA<cand)cand=scA; if(scB<cand)cand=scB; if(scC<cand)cand=scC; if(scD<cand)cand=scD;
          double dl=(base-cand)/(base>1e-12?base:1); jdl=100.0*dl;
          if(dl>bf_bestdl){ bf_bestdl=dl; bf_bestJ=J; } }
        if(form){
            if(!BKL_FPLP){ vrej++; form=0; }   /* 闸材料缺失: fail-closed 拒落地 */
        }
        if(form){
            /* ★逐单元全程复核, ONEPASS 恒开(2026-08-29 终验判决后重启)★
             * 历史脉络: 08-27 我开过→08-28 用户令还原冠军原版(当时实测 0落地/4保持=全挡死,
             * 且"聚合终验功效更高"论成立)→08-29 全域行域修复后终验实测: 41 个近视野全正的
             * 落地在全程出口上净劣化 27% 被【整体】回滚 —— 聚合终验只能全灭全存, 好坏不分;
             * 逐单元近视野判据与全程出口不对齐是结构性的(en86 同构)。
             * 当年"全挡死"的两个前提都已变: ①坏行域(2域)+锚索引 bug 已修 ②复核成本 —— 旧版
             * 走【全量行】前向(43 层×8.5s), 现在走【抽格行】(moe GPU 接管, ~0.5s/层),
             * 单元 +2 次全程前向 ≈ 20-40s, 可负担。打分行=EVc(跨全域 eval 抽格), 判据=配对
             * 差分(同行集 basef vs scv), 行少的噪声在差分里大幅相消。
             * 统一终验保留(双保险, 应恒过 —— 每个落地都单独全程验证过)。 */
            double basef=0,scv=0,c0=0,m0=0,c1=0,m1=0;
            double basekl=bkl_gate(J,Lfront,Hin,rowsz,S,&basef,&c0,&m0);
            lop_t svE; float svg=0,svt=0,svw2[4]; float*gbak=NULL; int tmpop=-1;
            memset(&svE,0,sizeof(svE));
            if(form==5){ svE=lf->ops[ze]; lf->ops[ze]=opE; }
            else if(form==6){ tmpop=lf->nops; lf->ops[tmpop]=opF; lf->nops++; }
            else if(form==1){ if(fo>=0){ svg=lf->ops[fo].g; lf->ops[fo].g=baseg*aA; }
                else if(lf->nops<32){ tmpop=lf->nops; memset(&lf->ops[tmpop],0,sizeof(lop_t));
                    lf->ops[tmpop].type=1; lf->ops[tmpop].g=aA; lf->nops++; } }
            else if(form==2){ svt=lf->ops[to].t; lf->ops[to].t=tB; }
            else if(form==3){ int od=BF_DYN2OP[J];
                if(od>=0){ memcpy(svw2,lf->ops[od].w2p,16); memcpy(lf->ops[od].w2p,opC.w2p,16); }
                else if(lf->nops<32){ tmpop=lf->nops; lf->ops[tmpop]=opC; lf->nops++; } }
            else { int go=BF_GEOP[J];
                if(go>=0){ gbak=malloc((size_t)NEXP*4); memcpy(gbak,lf->ops[go].ge,(size_t)NEXP*4);
                    for(int e=0;e<NEXP;e++) lf->ops[go].ge[e]*=geD[e]; }
                else if(lf->nops<32){ tmpop=lf->nops; memset(&lf->ops[tmpop],0,sizeof(lop_t));
                    lf->ops[tmpop].type=5; lf->ops[tmpop].ge=geD; lf->nops++; } }   /* geD 所有权不转移 */
            double candkl=bkl_gate(J,Lfront,Hin,rowsz,S,&scv,&c1,&m1);
            g_anc_rowmap=scr?sidx:NULL;               /* 还原本单元的粗筛映射 */
            if(form==5) lf->ops[ze]=svE;
            else if(form==6) lf->nops--;
            else if(form==1){ if(tmpop>=0) lf->nops--; else lf->ops[fo].g=svg; }
            else if(form==2) lf->ops[to].t=svt;
            else if(form==3){ if(tmpop>=0) lf->nops--; else memcpy(lf->ops[BF_DYN2OP[J]].w2p,svw2,16); }
            else { if(tmpop>=0) lf->nops--; else memcpy(lf->ops[BF_GEOP[J]].ge,gbak,(size_t)NEXP*4); }
            if(gbak){ free(gbak); gbak=NULL; }
            printf("[BKL] L=%02d 复核: heldKL %.5f→%.5f(%s) 隐分 %.5g→%.5g | 方向cos %.5f→%.5f 模长误差 %.3f%%→%.3f%%\n",
                   J,basekl,candkl,candkl<basekl-1e-9?"降✓放行":"不降✗拒",basef,scv,c0,c1,100*m0,100*m1); fflush(stdout);
            if(candkl<basekl-1e-9){ base=basef; bestsc=scv; eFlog=Lfront; }   /* 放行: 真尺(held-KL)降 */
            else { vrej++; form=0; }                        /* KL 不降: 拒落地(隐分再好也不算肉) */
        }
        if(form){
            char al[64],rs[128]; uint64_t vol=4;
            if(form==5){   /* ★机制本体: 重解自有 z 落地(原地改写该 op 系数)★ */
                lf->ops[ze]=opE; zfile_commit(J,&lf->ops[ze],(float)bestsc);
                vol=opE.type==3?36:opE.type==2?16:4;
                snprintf(al,64,"own-z重解 type%d β=%.2f(前沿闭式)",opE.type,bE);
            } else if(form==1){
                if(fo>=0){ lf->ops[fo].g=baseg*aA; zfile_commit(J,&lf->ops[fo],(float)bestsc); }
                else { fo=lf->nops; memset(&lf->ops[fo],0,sizeof(lop_t));
                    lf->ops[fo].type=1; lf->ops[fo].g=aA; lf->nops++;
                    size_t off=append_rec(pj,"bf.GL","逐层反修最终缩放α(前沿判据)",&lf->ops[fo].g,4,aA);
                    if(off){ lf->ops[fo].foff=off; BF_FINOP[J]=fo; } else { lf->nops--; form=0; } }
                if(form) snprintf(al,64,"bf.GL α=%.3f(累计g=%.4f)",aA,lf->ops[fo].g);
            } else if(form==2){
                lf->ops[to].t=tB; zfile_commit(J,&lf->ops[to],(float)bestsc);
                snprintf(al,64,"bwd.TREF重调 t=%.4f(前沿判据)",tB);
            } else if(form==3){
                int od=BF_DYN2OP[J];
                if(od>=0){ memcpy(lf->ops[od].w2p,opC.w2p,16); zfile_commit(J,&lf->ops[od],(float)bestsc); }
                else { od=lf->nops; lf->ops[od]=opC; lf->nops++;
                    size_t off=append_rec(pj,"bf.GLdyn2","逐层反修per-token动态(范数2dof,前沿判据)",opC.w2p,16,opC.w2p[0]);
                    if(off){ lf->ops[od].foff=off; BF_DYN2OP[J]=od; } else { lf->nops--; form=0; } }
                vol=16; if(form) snprintf(al,64,"bf.GLdyn2 w=[%.3f,%.4f]",opC.w2p[0],opC.w2p[1]);
            } else if(form==6){   /* ★F: 新建 9-dof 动态 op(PCA8 方向, V8 内联)★ */
                int od=lf->nops;
                uint64_t plen=36+(uint64_t)8*DIM*2;
                uint8_t*payb=malloc((size_t)plen);
                memcpy(payb,opF.w8,36);
                { uint16_t*h=(uint16_t*)(payb+36);
                  for(size_t j=0;j<(size_t)8*DIM;j++) h[j]=go1b_fp32_to_fp16(opF.V8[j]); }
                size_t off=append_rec(pj,"bf.GLdyn8","逐层反修9dof动态新建(PCA8特征,前沿判据)",payb,plen,opF.w8[0]);
                free(payb);
                if(off){ lf->ops[od]=opF; lf->ops[od].foff=off; lf->nops++; V8F=NULL; /* 所有权交 op */ }
                else form=0;
                vol=plen; if(form) snprintf(al,64,"bf.GLdyn8新建 w0=%.3f(PCA8·9dof)",opF.w8[0]);
            } else {   /* form==4: per-expert 增益 GE(256-dof) */
                int go=BF_GEOP[J];
                if(go>=0){ for(int e=0;e<NEXP;e++) lf->ops[go].ge[e]*=geD[e];
                    zfile_commit(J,&lf->ops[go],(float)bestsc); free(geD); geD=NULL; }
                else { go=lf->nops; memset(&lf->ops[go],0,sizeof(lop_t));
                    lf->ops[go].type=5; lf->ops[go].ge=geD; lf->nops++;
                    uint16_t*h=malloc((size_t)NEXP*2);
                    for(int e=0;e<NEXP;e++) h[e]=go1b_fp32_to_fp16(geD[e]);
                    size_t off=append_rec(pj,"bf.GE","逐层反修per-expert增益(路由投影,前沿判据)",h,(uint64_t)NEXP*2,1.0f);
                    free(h);
                    if(off){ lf->ops[go].foff=off; BF_GEOP[J]=go; geD=NULL; /* 所有权交 op */ }
                    else { lf->nops--; free(geD); geD=NULL; form=0; } }
                vol=(uint64_t)NEXP*2; if(form) snprintf(al,64,"bf.GE 256专家增益(路由投影)");
            }
            if(form){
                if(!onep){ float*Hr=gs_forward_exit(J,Lfront,Hin,ids,S,n_fit,HQE); free(Hr); }
                else oland[J]=1;   /* ONEPASS: 基线冻结不刷 HQE, 终验统一做 */
                changed++; simpr+=(base-bestsc); done[J]=1;
                /* ★打印必须报真出口层(2026-08-28 用户揪出)★ 原来这里恒打 Lfront, 而 base/bestsc
                 * 是在【近视野出口 eF=min(J+BK,Lfront)】上算的(BK 由 DS4_BF_SCREEN_K 给, 战役脚本
                 * 设 4)。于是 J≥Lfront−BK 的单元报 L42, 再往前的单元实际在 L41/L40 上评分却照报
                 * L42 —— 日志里就出现"出口分 200.35 → 4.6968"这种 42 倍断崖, 看着像数值炸了。
                 * 后果不止是误读: 换了出口层, 各单元的 Δ% 根本不可比, 也不能拿去对冠军的 −58.4%。 */
                snprintf(rs,128,"出口L%d分 %.5g→%.5g 降%.2f%%%s%s",eFlog?eFlog:eF,base,bestsc,
                         100.0*(base-bestsc)/(base>1e-12?base:1),
                         eF!=Lfront?"(近视野, 与前沿L42单元不可比)":"",pass?"(复检)":"");
                mlog(J,"向前反修",al,"已改写层文件",vol,rs,"✓正向落地");
            }
        }
        if(geD){ free(geD); geD=NULL; }
        if(V8F){ free(V8F); V8F=NULL; }     /* F 候选没落地(落地时所有权已交 op) */
        if(!form) jd[J]=jdl;                /* 未正向: 记录候选Δ, 收尾统一上日志(复检后) */
        free(_fv); _fv=NULL;   /* 拟合视图: 每单元重建(ms 逐单元算) */
        /* ★arena 归还(2026-08-29 二修)★ 复核每单元~17次30MB malloc/free 在 1GB MMAP_THRESHOLD
         * (历史设定护 fp16 层缓存复用, 不能动)下滞留 arena→available 4G→managed 分配失败。
         * trim 只还已 free 碎片, 在用块不动, ms 级。 */
        malloc_trim(0);
        printf("BFUNIT L=%02d Δbest=%+.3f%% %s 用时=%lds\n",J,jdl,form?"落地":"保持",(long)(time(NULL)-ut0));
        fflush(stdout); bflt_print(J,vqt_now()-ut0d);   /* 实时可观测(用户裁决: 未落地层不许等到轮末才现身)+单元段账 */
        if(bis){
            /* ★一点采样不判整段生死(单测实锤: mid 平坦剪长段 ⇒ 深尾大增益整段漏检)★
             * 段长>DS4_BF_BISECT_MAXSEG(默认3)时 mid 平坦也必须细分; 只许剪 ≤MAXSEG 的小段
             * ⇒ 漏检窗口≤相邻2层, 增益孤岛必被四分位链命中。 */
            int seglen=bhi-blo+1;
            int mseg=3;
            if(form||jdl>=bskip||seglen>mseg){ /* 有增益或段还长 → 两侧细分(右段后压=先访问深层) */
                if(blo<=J-1){ bseg[bsp][0]=blo; bseg[bsp][1]=J-1; bsp++; }
                if(J+1<=bhi){ bseg[bsp][0]=J+1; bseg[bsp][1]=bhi; bsp++; }
            } else {                        /* 小段且 mid 平坦 → 剪两侧: done=1 复检遍/收尾日志都跳过 */
                int npr=0; for(int q=blo;q<=bhi;q++) if(!done[q]&&q!=J) npr++;
                if(npr){ printf("BFBISECT 剪枝 [L%02d..L%02d] mid=L%02d Δ=%+.3f%%<%.2f%% 剪%d层\n",
                                blo,bhi,J,jdl,bskip,npr); fflush(stdout);
                    for(int q=blo;q<=bhi;q++) if(!done[q]&&q!=J) done[q]=1; }
            }
        }
    }
    }   /* end pass 复检 */
    if(onep){ BFU_ARM=0;
        int nland=0; for(int J=Jlo;J<Lfront;J++) if(oland[J]) nland++;
        if(nland){
            /* 统一终验: 冻结基线上全部落地 → 一次全程出口分 vs 进入时基线(HQE[Lfront+1]=反修态出口) */
            g_anc_rowmap=NULL;
            double b0=co_score_rows(HQE+(size_t)(Lfront+1)*lstride,Htgt,LEV,nLEV,rowsz);
            double b0kl=BKL_FPLP?bkl_exit_kl(HQE+(size_t)(Lfront+1)*lstride,rowsz):1e300;
            float*Hf=gs_forward_exit(Jlo,Lfront,HQE+(size_t)Jlo*lstride,ids,S,n_fit,NULL);
            double fin=co_score_rows(Hf,Htgt,LEV,nLEV,rowsz);
            double finkl=BKL_FPLP?bkl_exit_kl(Hf,rowsz):1e300; free(Hf);
            int keep=BKL_FPLP?(finkl<b0kl-1e-9):0;   /* 判据=held-KL; 闸拒臂时无落地可判(fail-closed) */
            printf("BF_ONEPASS 终验 落地=%d heldKL %.5f→%.5f(真尺判据) 出口分 %.5g→%.5g(对照) %s\n",
                   nland,b0kl,finkl,b0,fin,keep?"✓改善→提交":"✗劣化→全回滚"); fflush(stdout);
            if(keep){
                float*Hr=gs_forward_exit(Jlo,Lfront,HQE+(size_t)Jlo*lstride,ids,S,n_fit,HQE); free(Hr);
            } else {
                for(int J=Lfront-1;J>=Jlo;J--) if(oland[J]&&ofl[J]){          /* ①截掉 append(宿主=op 侧车) */
                    char pj2[512]; op_host_path(J,pj2,sizeof(pj2));
                    if(truncate(pj2,(off_t)ofl[J])!=0) perror("onep-trunc"); }
                for(int i=NBFU-1;i>=0;i--){ bfu_t*u=&BFU[i];                   /* ②逆序回写原地改(宿主同上) */
                    char pj2[512]; op_host_path(u->L,pj2,sizeof(pj2));
                    int fd2=open(pj2,O_WRONLY);
                    if(fd2>=0){ if(pwrite(fd2,u->old,u->len,(off_t)u->foff)!=(ssize_t)u->len) perror("bfu-rb");
                        close(fd2); } }
                for(int J=Jlo;J<Lfront;J++) if(oland[J]){                     /* ③重读: 主 dql 路径(侧车自动挂) */
                    char pj2[512]; snprintf(pj2,sizeof(pj2),"%s/dql_L%02d.bin",
                        getenv("DS4_LAYER_DIR")?getenv("DS4_LAYER_DIR"):".",J);
                    lfile_free(&GS_LF[J]); memset(&GS_LF[J],0,sizeof(lfile_t)); lfile_load(pj2,&GS_LF[J]); }
                changed=0;
            }
        }
        for(int i=0;i<NBFU;i++) free(BFU[i].old);
        free(LFIT); free(LEV); LFIT=LEV=NULL; free(FITc); free(EVc); FITc=EVc=NULL;
        NBFU=0; free(ofl); free(oland);
    }
    for(int J=Lfront-1;J>=Jlo;J--) if(!done[J]){   /* 仍未正向: 逐层上日志(重解=原值, 非隐身) */
        if(ucl<(int)sizeof(uc)-24) ucl+=snprintf(uc+ucl,sizeof(uc)-ucl," L%d(%+.2f%%)",J,jd[J]);
        char rs[64]; snprintf(rs,64,"重解=原值(候选Δ=%+.2f%%), 前沿L%d 含复检",jd[J],Lfront);
        mlog(J,"向前反修","own-z重解+α/TREF/dyn2/GE 全形态","已最优保持",0,rs,"保持");
    }
    free(done); free(jd); if(bseg) free(bseg);
    free(db); free(dg); free(ms); free(kkrow); free(effw);
    if(sidx){ free(sidx); free(ids_s); free(Hin_s); free(Ht_s); }
    printf("BACKFIT_PREV Lfront=%d 评估=%d 提交=%d Δ改善=%.4g 最优候选Δ=%.3g%%@L%d 用时=%lds 粗筛=K%d÷%d(行%d)%s 全闸拒=%d 未正向:%s(判据=L%d出口回放; 形态=own-z/α/TREF/dyn2/GE+复检)\n",
           Lfront,evald,changed,simpr,100.0*bf_bestdl,bf_bestJ,(long)(time(NULL)-bt0),BK,SDIV,Ss,bis_on?" 序=bisect":"",vrej,ucl?uc:" 无",Lfront);
    fflush(stdout);
    return changed;
}

/* ★2026-08-28 用户令: 冠军 r64(08-05)没有分块, 还原单次全域调用★
 * DS4_BF_CHUNK 是 08-07 "增益蒸发"战役才加的后续修复, 不属于 r64 配方。
 * backfit_prev_chunk 的实现保留(参数化的边界), 这里只用 [0,Lfront) 一次调完 = 原语义。 */
static int backfit_prev(int Lfront,const long*ids,int S,int n_fit){
    return backfit_prev_chunk(0,Lfront,Lfront,ids,S,n_fit);
}
