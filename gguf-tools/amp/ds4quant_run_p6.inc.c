    /* ---- 胜者 payload(在 BWD 污染缓存前构建) ---- */
    uint8_t *pay=NULL; uint64_t paysz=0;
    if(!strncmp(win.algo,"GL",2)){ pay=malloc(4); memcpy(pay,&win.g,4); paysz=4; }
    else if(!strncmp(win.algo,"GE",2)){ /* 重建 gstat(确定性) 后打包 */
        int tgt=!strncmp(win.algo,"GEcum",5);
        for(int e=0;e<NEXP;e++){ coexp_t*ce=&ex[e]; ce->gstat=0.0f;
            double num=0,den=1e-30;
            if(!tgt){ if(!ce->gcal)continue;
                for(int t=0;t<ce->ncal;t++){ if(ce->calS[t]>=vs)continue;
                    num+=(double)ce->gcal[t]*ce->gden[t]; den+=ce->gden[t]; }
                if(den>1e-20){ double g=num/den; if(g<0.25)g=0.25; if(g>4.0)g=4.0; ce->gstat=(float)g; } }
            else { if(!ce->yq_all)continue;
                for(int i=0;i<ce->nhit;i++){ int s=ce->hitS[i]; if(s>=vs)continue;
                    double w=ce->hitW[i],a=w*w/(sw2act[s]>1e-9?sw2act[s]:1e-9);
                    const float*y=ce->yq_all+(size_t)i*DIM,*d0=DF+(size_t)s*DIM;
                    double yd=0,yy=0;
                    for(int d2=0;d2<DIM;d2++){ yd+=(double)y[d2]*d0[d2]; yy+=(double)y[d2]*y[d2]; }
                    num+=a*w*yd; den+=w*w*yy; }
                double g=1.0+num/den; if(g<0.25)g=0.25; if(g>4.0)g=4.0; ce->gstat=(float)g; } }
        pay=co_pay_mult_stat(ex,&paysz);
        for(int e=0;e<NEXP;e++){ if(ex[e].ccal)for(int t=0;t<ex[e].ncal;t++)ex[e].ccal[t]=ex[e].gstat?ex[e].gstat:1.0f; }
    }
    else if(!strncmp(win.algo,"CE",2)){ /* 重拟合胜者 λ/目标 → ccal 就位 + v_e 打包 */
        int tgt=!strncmp(win.algo,"CEcum",5);
        float *gt=malloc((size_t)n_fit*4);
        for(int e=0;e<NEXP;e++){ coexp_t*ce=&ex[e]; ce->gstat=0.0f;
            if(ce->ncal<1)continue;
            const float*g_use=ce->gcal;
            if(tgt){ for(int t=0;t<ce->ncal;t++){ int s=ce->calS[t];
                    const float*y=ce->yq_cal+(size_t)t*DIM,*d0=DF+(size_t)s*DIM;
                    double yd=0; for(int d2=0;d2<DIM;d2++) yd+=(double)y[d2]*d0[d2];
                    double den=ce->gden[t]>1e-20?ce->gden[t]:1e-20;
                    double g=1.0+((double)ce->calW[t]/(sw2anc[s]>1e-9?sw2anc[s]:1e-9))*yd/den;
                    if(g<0.25)g=0.25; if(g>4.0)g=4.0; gt[t]=(float)g; }
                g_use=gt; }
            co_fit_ze(ce,afin,g_use,win.lam,vs);
        }
        free(gt);
        pay=co_pay_mult_dyn(ex,afin,&paysz);
    }
    /* (ZL 胜者重建已删: 加法菜单 2026-08-31 随 DS4_ADD_FORMS 清退, "ZL" 不再可能胜出) */
    /* ---- BWD(向后, D3): 每专家乘法胜者之下 base 重解一轮, 胜了才接管(GL 无 per-expert 目标不参与) ---- */
    if(!strncmp(win.algo,"GE",2)||!strncmp(win.algo,"CE",2)){
        long nf2=co_base_pass(L,S,n_fit,1,Fin,idx,rw,ex,routed,NULL);
        (void)nf2;
        for(size_t i=0;i<(size_t)S*DIM;i++) Fcur[i]=shared[i]+routed[i];
        co_df(Fcur,H2,post2,comb2,Hf,Hq,DF,S);
        if(!strncmp(win.algo,"GE",2)){ /* 静态增益: 刷新后的 gcal/gden 重算 gstat */
            int tgt=!strncmp(win.algo,"GEcum",5);
            for(int e=0;e<NEXP;e++){ coexp_t*ce=&ex[e]; ce->gstat=0.0f;
                double num=0,den=1e-30;
                if(!tgt){ if(!ce->gcal)continue;
                    for(int t=0;t<ce->ncal;t++){ if(ce->calS[t]>=vs)continue;
                        num+=(double)ce->gcal[t]*ce->gden[t]; den+=ce->gden[t]; }
                    if(den>1e-20){ double g=num/den; if(g<0.25)g=0.25; if(g>4.0)g=4.0; ce->gstat=(float)g; } }
                else { if(!ce->yq_all)continue;
                    for(int i=0;i<ce->nhit;i++){ int s=ce->hitS[i]; if(s>=vs)continue;
                        double w=ce->hitW[i],a=w*w/(sw2act[s]>1e-9?sw2act[s]:1e-9);
                        const float*y=ce->yq_all+(size_t)i*DIM,*d0=DF+(size_t)s*DIM;
                        double yd=0,yy=0;
                        for(int d2=0;d2<DIM;d2++){ yd+=(double)y[d2]*d0[d2]; yy+=(double)y[d2]*y[d2]; }
                        num+=a*w*yd; den+=w*w*yy; }
                    double g=1.0+num/den; if(g<0.25)g=0.25; if(g>4.0)g=4.0; ce->gstat=(float)g; } }
        }
        if(!strncmp(win.algo,"CE",2)){ /* 重拟合胜者形态(gcal/gden/yq 已被 worker 刷新) */
            int tgt=!strncmp(win.algo,"CEcum",5);
            float *gt=malloc((size_t)n_fit*4);
            for(int e=0;e<NEXP;e++){ coexp_t*ce=&ex[e];
                if(ce->ncal<1)continue;
                const float*g_use=ce->gcal;
                if(tgt){ for(int t=0;t<ce->ncal;t++){ int s=ce->calS[t];
                        const float*y=ce->yq_cal+(size_t)t*DIM,*d0=DF+(size_t)s*DIM;
                        double yd=0; for(int d2=0;d2<DIM;d2++) yd+=(double)y[d2]*d0[d2];
                        double den=ce->gden[t]>1e-20?ce->gden[t]:1e-20;
                        double g=1.0+((double)ce->calW[t]/(sw2anc[s]>1e-9?sw2anc[s]:1e-9))*yd/den;
                        if(g<0.25)g=0.25; if(g>4.0)g=4.0; gt[t]=(float)g; }
                    g_use=gt; }
                co_fit_ze(ce,afin,g_use,win.lam,vs); }
            free(gt);
        }
        co_apply_mult(ex,afin,Fin,shared,S,Ftest,NULL,NULL);
        co_eval(Ftest,H2,post2,comb2,Hf,Hq,S,vs,n_fit,rowsz,&fitr,&valr,&heldr,&sc);
        printf("SEARCH L=%d BWD(%s)   val=%.4f(%+.1f%%) held=%.4f(%+.2f%%) fit=%.4f sc=%.5g\n",
               L,win.algo,valr,100*(valr-bval)/bval,heldr,100*(heldr-bheld)/bheld,fitr,sc);
        if(sc<win.sc){
            win.sc=sc; win.bwd=1; win.fit=fitr; win.val=valr; win.held=heldr;
            memcpy(FoutBest,Ftest,(size_t)S*DIM*4);
            free(pay); pay=NULL; paysz=0;
            if(win.algo[0]=='G') pay=co_pay_mult_stat(ex,&paysz);
            else                 pay=co_pay_mult_dyn(ex,afin,&paysz);
        }
    }
    /* ---- 元素叠加(只进不退: 每元素在当前余量上找正向形态, val 变好才接管) ---- *
     * 加法元素: BIAS 平均漂移向量(逐维统计量, 零回归方差, 与 GL 同理必泛化);
     * 动态元素: GLdyn 低维动态层标量 g(x)=w·[1,PCA8(Fin)] (9 dof vs 171 行, 不饥饿);
     * 向后元素: TREF 修正量 t 精调(从落地点出发只向 val 更好走, 构造上非负)。 */
    /* ==== 每轮迭代累计最优: 循环 {GL重拟合→GLdyn2→GLdyn8→TREF}, val 变好才接管,
     * 一轮零接管即收敛; 每次接管一条记录(#轮次)带参数载荷入文件 —— 第一原则 ==== */
    /* ★根因修复 'B'!='g'(字节回放≠调优): sweep 胜者(如 GL g=1.16)经 el_add【无 payload】记录,
     * 且 Fstate=FoutBest 把这个【未记录】修正设为起点 → co_rounds 只在其上找≈1 余量 → 主增益不进层文件 →
     * bytes_moe 回放丢主增益(实测 Fout 169 vs 82)。改为 co_rounds 从 base 重推全部修正(每条 el_addp 带
     * payload 可回放, 与 bytes_moe/合并同口径)。sweep 只保留选形态诊断; zfile 侧车 win/pay 不受影响。*/
    memcpy(FoutBest,Fcur,(size_t)S*DIM*4);
    /* ★落地门用的基座快照(2026-08-01): 增益不足时 Fout 回退到这里, 等于本层空手 */
    float *FoutBase=malloc((size_t)S*DIM*4); memcpy(FoutBase,Fcur,(size_t)S*DIM*4);
    double v_mult=bval, sc_state=scb;
    double v_cur=bval, h_cur=bheld, v_dyn=bval, v_bwd=bval, h_bwd=bheld;
    float *Fstate=FoutBest;
    float *routedC=malloc((size_t)S*DIM*4);
    /* ==== 阶段2: z动态(默认旋钮) → 收敛 ==== */
    costx_t CX={L,S,n_fit,vs,rowsz,bval,bheld,Fin,Hf,H2,post2,comb2,shared,Fcur,
                Fstate,Ftest,routedC,DF,Hq,&sc_state,&v_cur,&h_cur};
    double LTUNE_val[5],LTUNE_knob[5]; int LTUNE_hit[5];   /* 记账在联合循环外只记终态 */
    /* ★同层联合(2026-08-02 用户令"z/四损失/感知/向后应同一层一起生效, 不是顺序执行")★
     * {阶段2 旋钮网格(GL/dyn/TREF×四损失/感知)}→{阶段7 向后}→{阶段9 z^L} 原为单向链:
     * z 最后解且不回环, GL/dyn 永远没机会在"z 在场"的残差上重解自己(症状=侧车增益频繁
     * 过不了落地门)。改块坐标下降: 整链循环 DS4_BF_ALT 轮(默认 2), 第 2 轮各机制在彼此
     * 已落地的 Fstate 上原地重解(BF_*OP 下标复用不重复追加; ZLP 暂存整体替换; loss.*
     * 记账在循环外只记终态)。质量保底: 全链统一 sc 闸只进不退 ⇒ 最坏零接管纯耗时。
     * 提前收敛: 某轮里阶段7+9 都零接管 ⇒ 下一轮无新增在场机制, 必零改善, 直接跳出。 */
    { const int BF_ALT=DSQ_BF_ALT;   /* 写死 2(原 DS4_BF_ALT env, 无脚本设置) */
    for(int alt=0;alt<BF_ALT;alt++){
    double alt_entry=v_cur, alt_s2exit=v_cur;
    /* ★一次解决(2026-08-02 用户令"两轮时间投入不一定收益高")★: 完全 one-shot 无闭式
     * (GL×z×t 双线性耦合), 但顺序可换 — R29 实证 z(RRR)是最强机制却排最后捡剩饭。
     * r1 只解 z(闭式便宜, 最强机制先定调); r2 全链一遍(网格/四损失/感知在 z 已在场的
     * 残差上解 → 向后 → z 重解)。总耗时 ≈1.1×单遍(原两全链轮 1.6-1.8×), 联合语义更完整。 */
    int zfirst=(alt==0&&BF_ALT>1);
    if(!zfirst){
    /* ==== 阶段1.5(2026-08-06 用户令"冷热双通道"): GLhc 链首 2×2 闭式 ====
     * routed=R_hot+R_cold(bytes_moe 分桶快照); 解 min‖δh·RH+δc·RC−DF‖²(fit 行),
     * DF=co_df 目标修正方向。统一 sc 闸只进不退; 记录 bf.GLhc(gh,gc)=覆盖式链首语义。 */
    if(BM_RH&&BM_S==S){
        co_df(Fstate,H2,post2,comb2,Hf,Hq,DF,S);
        double a11=0,a12=0,a22=0,d1=0,d2v=0;
        for(int s2=0;s2<n_fit;s2++){ const float*rh=BM_RH+(size_t)s2*DIM,*rc=BM_RC+(size_t)s2*DIM,*df=DF+(size_t)s2*DIM;
            for(int d3=0;d3<DIM;d3++){ a11+=(double)rh[d3]*rh[d3]; a12+=(double)rh[d3]*rc[d3];
                a22+=(double)rc[d3]*rc[d3]; d1+=(double)rh[d3]*df[d3]; d2v+=(double)rc[d3]*df[d3]; } }
        double det=a11*a22-a12*a12;
        if(det>1e-9*(a11+a22+1e-30)){
            double dh=( a22*d1-a12*d2v)/det, dc=(-a12*d1+a11*d2v)/det;
            if(dh>1.0)dh=1.0; if(dh<-0.75)dh=-0.75; if(dc>1.0)dc=1.0; if(dc<-0.75)dc=-0.75;   /* 信赖域 */
            for(int s2=0;s2<S;s2++){ const float*rh=BM_RH+(size_t)s2*DIM,*rc=BM_RC+(size_t)s2*DIM;
                float*ft=Ftest+(size_t)s2*DIM; const float*fs=Fstate+(size_t)s2*DIM;
                for(int d3=0;d3<DIM;d3++) ft[d3]=fs[d3]+(float)dh*rh[d3]+(float)dc*rc[d3]; }
            co_eval(Ftest,H2,post2,comb2,Hf,Hq,S,vs,n_fit,rowsz,&fitr,&valr,&heldr,&sc);
            printf("STAGE L=%d 冷热GLhc δh=%+.4f δc=%+.4f val=%.4f held=%.4f %s\n",
                   L,dh,dc,valr,heldr,sc<sc_state?"✓接管":"✗");
            fflush(stdout);
            if(sc<sc_state){ sc_state=sc; v_cur=valr; h_cur=heldr;
                memcpy(Fstate,Ftest,(size_t)S*DIM*4);
                float g2p[2]={(float)(1.0+dh),(float)(1.0+dc)};
                el_addp("bf.GLhc","冷热双通道链首(gh,gc 2x2闭式)",2,v_cur,h_cur,0,dh,1,g2p,8);
                mlog(L,"冷热双通道","GLhc","接管",8,"-","✓正向"); }
        }
    }
    DYN_LSM=0.1; DYN_LFIX=1.0; SOLVE_WCLS=0.0;
    memset(LTW_hit,0,sizeof(LTW_hit)); memset(LTW_knob,0,sizeof(LTW_knob));
    mlog(L,"调优阶段","z动态×旋钮网格(GL/dyn2/dyn8/TREF × 默认+四损失+感知)","阶段2(3-6/8已并入)",0,"-","进行中");
    co_rounds(&CX,2,"z动态×旋钮网格","z");   /* 网格只在 r2 跑一次(z 已在场), 完整 2 轮 */
    alt_s2exit=v_cur;   /* 本轮阶段2出口 — 联合收敛判据基准 */
    v_dyn=v_cur;
    /* ==== 阶段3..6+8(2026-07-13 用户裁决重构): 四损失/感知旋钮并进阶段2的 dyn2/dyn8 求解
     * 网格(KGRID) — 每轮每形态全网格独立解+评, 统一 val argmax, 胜者按家族入账 LTW_*。
     * 旧"先默认收敛→逐损失重跑同族求解"结构性必输(去打已收敛局部最优, 43层全零接管实证), 已删。 ==== */
    for(int li=0;li<5;li++){ LTUNE_val[li]=v_cur; LTUNE_knob[li]=LTW_knob[li]; LTUNE_hit[li]=LTW_hit[li]; }
    printf("STAGE_BEST L=%d 四损失+感知(网格内落地) al=%d cl=%d fx=%d sm=%d pc=%d val=%.4f\n",
           L,LTUNE_hit[0],LTUNE_hit[1],LTUNE_hit[2],LTUNE_hit[3],LTUNE_hit[4],v_cur);
    fflush(stdout);
    DYN_LSM=0.1; DYN_LFIX=1.0; SOLVE_WCLS=0.0;
    /* ==== 阶段7: 向后调优(动态: TREF t 全网格候选扫描, 循环到收敛, 只进不退) ==== */
    mlog(L,"调优阶段","向后(TREF t网格动态)","阶段7",0,"-","进行中");
    { const double TS[7]={0.75,0.80,0.85,0.90,0.95,1.05,1.15};
      double bw_entry=v_cur; float bw_bestk=1.0f;
      int rd=1;
      for(;;){
        int improved=0; float *bestT=NULL; double t_b=1.0;
        for(int ti=0;ti<7;ti++){ double t=TS[ti];
            for(size_t i=0;i<(size_t)S*DIM;i++) Ftest[i]=Fcur[i]+(float)t*(Fstate[i]-Fcur[i]);
            co_eval(Ftest,H2,post2,comb2,Hf,Hq,S,vs,n_fit,rowsz,&fitr,&valr,&heldr,&sc);
            printf("STAGE L=%d 向后 r%d 候选t=%.2f val=%.4f held=%.4f %s\n",
                   L,rd,t,valr,heldr,sc<sc_state?"✓暂最优":"✗");
            if(sc<sc_state){ sc_state=sc; v_cur=valr; h_cur=heldr; t_b=t;
                if(!bestT) bestT=malloc((size_t)S*DIM*4);
                memcpy(bestT,Ftest,(size_t)S*DIM*4); }
        }
        if(bestT){ improved=1; TREF_T=(float)t_b; bw_bestk=(float)t_b;
            float tf=(float)t_b; char nm[16]; snprintf(nm,16,"bwd.TREF#7.%d",rd);
            el_addp(nm,"向后动态: 修正量整体t网格精调",4,v_cur,h_cur,100*(bw_entry-v_cur)/bval,t_b,1,&tf,4);
            memcpy(Fstate,bestT,(size_t)S*DIM*4); free(bestT);
            { char ag[40],rs[48]; snprintf(ag,40,"向后 t=%.2f",t_b); snprintf(rs,48,"val→%.4f",v_cur);
              mlog(L,"向后调优",ag,"接管",4,rs,"✓正向"); } }
        printf("ROUND  L=%d 阶段=向后 r%d %s val=%.4f held=%.4f\n",L,rd,improved?"有接管":"零接管→收敛",v_cur,h_cur);
        fflush(stdout);
        if(!improved) break;
        if(++rd>16) break;
      }
      printf("STAGE_BEST L=%d 向后 %s 最优t=%.2f 阶段后val=%.4f(入口%.4f)\n",
             L,v_cur<bw_entry-1e-12?"✓有增益":"零接管",bw_bestk,v_cur,bw_entry);
      fflush(stdout);
    }
    v_dyn=v_cur;
    /* 阶段8(感知)已并入阶段2旋钮网格(pc 家族, KGRID w_cls∈{0.5,1,2,4}); TREF 后刷新记账 val */
    LTUNE_val[4]=v_cur;
    }   /* !zfirst — r1(z先行)跳过网格与向后, 直取阶段9 */
    /* ==== 阶段9(2026-07-14): 冻结 z^L(产物③) — 修三重缺席: ①活体块只在非共适应老路径
     * (COADAPT 跑从未执行) ②抽象四损失门恒选0 ③回放禁入不可导出。此处: 最终态解一次 →
     * 秩梯子 {提议,8,4,2,1} 过统一 sc 闸(与全部阶段同"只进不退"口径) → 冻结暂存 ZLP,
     * fwd_all 导出层文件后 append zl.RRR(可回放/可合并/引擎可执行)。 ==== */
    if(LZRANK>0){
        co_df(Fstate,H2,post2,comb2,Hf,Hq,DF,S);
        ds4_z *zl=z_solve_dual(afin,DF,(uint32_t)vs,DIM,DIM,(uint32_t)LZRANK,LZLAMBDA);
        if(zl){
            int kL=(nval>0)?z_pick_rank(zl,afin+(size_t)vs*DIM,DF+(size_t)vs*DIM,nval,DIM,NULL,NULL,NULL,NULL):8;
            int cand0[5]={kL>0?kL:8,8,4,2,1}; int kland=0;
            float *zd=malloc((size_t)DIM*4);
            for(int ci=0;ci<5&&!kland;ci++){ int kk=cand0[ci];
                if(kk<1||kk>(int)zl->rank) continue;
                int dup=0; for(int cj=0;cj<ci;cj++) if(cand0[cj]==kk) dup=1;
                if(dup) continue;
                ds4_z_set_rank(zl,(uint32_t)kk);
                memcpy(Ftest,Fstate,(size_t)S*DIM*4);
                for(int s2=0;s2<S;s2++){
                    memset(zd,0,(size_t)DIM*4); ds4_z_apply(zl,afin+(size_t)s2*DIM,zd);
                    double nd=0,nf=0; const float*fo=Ftest+(size_t)s2*DIM,*sh=shared+(size_t)s2*DIM;
                    for(int d2=0;d2<DIM;d2++){ nd+=(double)zd[d2]*zd[d2];
                        double rt=(double)fo[d2]-sh[d2]; nf+=rt*rt; }   /* 信赖域基准=routed(与引擎 λ 点同口径) */
                    nd=sqrt(nd); nf=sqrt(nf);
                    double cap=LZTR*nf; float sc2=1.0f;
                    if(nd>cap&&nd>0) sc2=(float)(cap/nd);
                    float *fw=Ftest+(size_t)s2*DIM;
                    for(int d2=0;d2<DIM;d2++) fw[d2]+=sc2*zd[d2];
                }
                co_eval(Ftest,H2,post2,comb2,Hf,Hq,S,vs,n_fit,rowsz,&fitr,&valr,&heldr,&sc);
                printf("ZLGATE L=%d k=%d sc %.6g→%.6g val=%.4f held=%.4f %s\n",
                       L,kk,sc_state,sc,valr,heldr,sc<sc_state?"✓落地":"✗拒");
                fflush(stdout);
                if(sc<sc_state){
                    kland=kk; sc_state=sc; v_cur=valr; h_cur=heldr;
                    memcpy(Fstate,Ftest,(size_t)S*DIM*4);
                    if(ZLP_U){ free(ZLP_U); free(ZLP_V); free(ZLP_Z); ZLP_U=ZLP_V=ZLP_Z=NULL; }
                    ZLP_K=kk; ZLP_L=L; ZLP_TR=(float)LZTR;
                    ZLP_Z=malloc((size_t)kk*4);
                    for(int c=0;c<kk;c++) ZLP_Z[c]=zl->z[c];
                    ZLP_U=malloc((size_t)DIM*kk*4); ZLP_V=malloc((size_t)DIM*kk*4);
                    for(int d2=0;d2<DIM;d2++) for(int c=0;c<kk;c++){
                        ZLP_U[(size_t)d2*kk+c]=zl->U[(size_t)d2*zl->rank+c];
                        ZLP_V[(size_t)d2*kk+c]=zl->V[(size_t)d2*zl->rank+c]; }
                    { char rs[64]; snprintf(rs,64,"k=%d val=%.4f held=%.4f",kk,valr,heldr);
                      mlog(L,"z^L","zl.RRR 秩梯子·统一sc闸",\
                           "✓过闸",(uint64_t)(16+2*((size_t)kk+2*(size_t)kk*DIM)),rs,"✓正向落地"); }
                    el_add("zl.RRR","冻结秩k方向修正(统一sc闸)",(uint64_t)(16+2*((size_t)kk+2*(size_t)kk*DIM)),
                           valr,heldr,0,kk,1);
                }
            }
            free(zd);
            if(st) st->zk=kland;
            if(kland>0) LZ_TOTAL_K+=kland;
            ds4_z_free(zl);
        }
    }
    printf("ALT L=%d r%d%s val %.4f→%.4f held=%.4f %s\n",L,alt+1,zfirst?"(z先行)":"",
           alt_entry,v_cur,h_cur,
           zfirst?(v_cur<alt_entry-1e-9?"z落地":"z零落地"):
           ((v_cur<alt_s2exit-1e-9)?"z/向后有接管":"z/向后零接管→联合收敛"));
    fflush(stdout);
    /* zfirst 轮永不 break(网格还没跑过, 必须进 r2); r2 起按"阶段7+9 零接管"收敛 */
    if(!zfirst && v_cur>=alt_s2exit-1e-9) break;
    } }
    free(routedC);
    v_bwd=v_cur; h_bwd=h_cur;
    /* 四损失+感知记录: 携带各自调优结果(最优旋钮/调后val), 参数载荷入文件 */
    if(!STK_CLSV) STK_CLSV=malloc(rowsz*4);
    ds4_loss_dim_variance(Hf+(size_t)vs*rowsz,(uint32_t)(nval>0?nval:1),(uint32_t)rowsz,STK_CLSV);
    {
        char ag[96];
        snprintf(ag,96,"对齐调优(λfix弱正则纯数据项) 最优=%.2f",LTUNE_knob[0]);
        { float pw[2]={1.0f,0.5f};
          el_addp("loss.align",ag,8,LTUNE_val[0],h_cur,LTUNE_hit[0]?1:0,LTUNE_knob[0],LTUNE_hit[0]?1:3,pw,8); }
        snprintf(ag,96,"分类调优(漂移行加权 w_cls) 最优=%.2f",LTUNE_knob[1]);
        el_add("loss.cls",ag,(uint64_t)rowsz*2,LTUNE_val[1],h_cur,LTUNE_hit[1]?1:0,LTUNE_knob[1],LTUNE_hit[1]?1:3);
        snprintf(ag,96,"固定调优(强锚 λfix) 最优=%.2f",LTUNE_knob[2]);
        { float pf[3]={(float)LTUNE_knob[2],0.25f,4.0f};
          el_addp("loss.fix",ag,12,LTUNE_val[2],h_cur,LTUNE_hit[2]?1:0,LTUNE_knob[2],LTUNE_hit[2]?1:3,pf,12); }
        snprintf(ag,96,"光滑调优(非截距ridge λsm) 最优=%.2f",LTUNE_knob[3]);
        { float ps[3]={(float)LTUNE_knob[3],0.25f,0.5f};
          el_addp("loss.smooth",ag,12,LTUNE_val[3],h_cur,LTUNE_hit[3]?1:0,LTUNE_knob[3],LTUNE_hit[3]?1:3,ps,12); }
        snprintf(ag,96,"感知调优(w_cls=1.5 重解+终端探针) %s",LTUNE_hit[4]?"有增益":"零接管");
        { float pc[2]={(float)LTUNE_val[4],(float)h_cur};
          el_addp("percept",ag,8,LTUNE_val[4],h_cur,LTUNE_hit[4]?1:0,LTUNE_knob[4],LTUNE_hit[4]?1:3,pc,8); }
    }
    printf("ELEMENTS L=%d base=%.4f | 乘法+%.1f%% | 动态+%.1f%% | 向后+%.1f%% | 终 val=%.4f held=%.4f(%+.2f%%)\n",
           L,bval,100*(bval-v_mult)/bval,100*(v_mult-v_dyn)/bval,100*(v_dyn-v_bwd)/bval,
           v_bwd,h_bwd,100*(h_bwd-bheld)/bheld);
    fflush(stdout);
    win.val=v_bwd; win.held=h_bwd;
    /* ★落地门(2026-08-01 用户令"负收益没有落地吧 / 这是一个致命的问题")★
     * 原逻辑注释是"每层必落菜单最优, 不存在空手" —— 没有任何增益门槛, 零增益甚至负增益的
     * 胜者照样改权重+写侧车。实测: L06/L07/L08/L10 各花 2048KB 换 +0.00%; L01(+0.01%)、
     * L05(+0.08%) 是负增益也落了。
     * ★零增益侧车不是中性的★: 它在校准集上恰好是 0, 但确实改了权重, 在未见语料上可能是负。
     * L08 带 Δb 后 -2.23% 修复能力凭空消失, 很可能就是前面几层这类"零增益改动"累积的结果。
     * 现在: 相对增益 < DS4_BF_GAIN_GATE(默认 0.05%) ⇒ 空手回退 —— Fout 回基座、侧车置 0、
     * win 记回基座值, 该层什么都不改。 */
    {
        double gain_pct = 100.0*(bheld - win.held)/bheld;
        double gate = g_cli.bf_gain_gate;
        if(gain_pct < gate){
            printf("SEARCH_GATE L=%d 增益 %+.3f%% < 门 %.2f%% -> 空手回退(侧车 %.1fKB->0)\n",
                   L,gain_pct,gate,(double)paysz/1024.0);
            fflush(stdout);
            memcpy(Fout,FoutBase,(size_t)S*DIM*4);
            if(pay){ free(pay); pay=NULL; }
            paysz=0; win.val=bval; win.held=bheld; win.k=0;
        } else {
            memcpy(Fout,FoutBest,(size_t)S*DIM*4);
        }
    }
    free(FoutBase);
    if(st) st->zk=win.k;
    printf("SEARCH_BEST L=%d algo=%s lam=%g g=%.4f k=%d bwd=%d val %.4f→%.4f(%+.1f%%) held %.4f→%.4f(%+.2f%%) payload=%.1fKB\n",
           L,win.algo,win.lam,win.g,win.k,win.bwd,bval,win.val,100*(win.val-bval)/bval,
           bheld,win.held,100*(win.held-bheld)/bheld,(double)paysz/1024.0);
    fflush(stdout);
    if(NZREC<NL){ zrec_t*r=&ZREC[NZREC++];
        r->L=L; snprintf(r->algo,12,"%s",win.algo); r->lam=win.lam; r->g=win.g; r->k=win.k;
        r->backward=win.bwd; r->valb=(float)bval; r->vala=(float)win.val;
        r->heldb=(float)bheld; r->helda=(float)win.held; r->pay=pay; r->paysz=paysz; }
    else free(pay);
    #undef CK
    for(int i=0;i<NELE;i++){ elrec_t*r=&ELE[i];
        printf("TABREC L=%d %-12s vol=%llu m=[%.4f %.4f %.1f %.2f] 判定=%d | %s\n",
               L,r->name,(unsigned long long)r->vol,r->m1,r->m2,r->m3,r->m4,r->verdict,r->algo); }
    fflush(stdout);
    if(g_cli.bwd_final){   /* 终端反调物料: base/校正 两份出口 H(hc_post 线性 → t 插值合法) */
        if(!BWD_Hb){ BWD_Hb=malloc(lstride*4); BWD_Hc=malloc(lstride*4); }
        dq_hc_post(Fcur,H2,post2,comb2,BWD_Hb,S,HCM,DIM);
        dq_hc_post(Fout,H2,post2,comb2,BWD_Hc,S,HCM,DIM);
        BWD_L=L;
    }
    for(int e=0;e<NEXP;e++){ coexp_t*ce=&ex[e];
        free(ce->hc_cal);free(ce->calS);free(ce->calW);free(ce->hc_all);free(ce->hitS);free(ce->hitW);
        free(ce->y2ref);free(ce->yq_all);free(ce->yq_cal);free(ce->gcal);free(ce->gden);free(ce->ccal);free(ce->alpha); }
    free(ex);free(shared);free(routed);free(Fcur);free(Ftest);free(FoutBest);free(Hq);
    free(DF);free(colw);free(sw2act);free(sw2anc);
}

/* ★锚路由 × 反修粗筛的行映射(2026-08-02)★
 * 反修粗筛(gs_backfit BACKFIT_SCREEN)把 token 按 SDIV 抽格, gather 成紧凑前缀后前向,
 * 于是前向看到的 S 是抽格行数、行号是 0..Ss-1; 而 ANC.ridx/ANC.rw 是按【原始全量行】
 * 排布的(stride = 原始 S)。两者直接对齐会双重错位(stride 错 + 行号错), 这正是原来
 * "DS4_ANCHOR_ROUTE ⇒ 硬关粗筛"的理由。
 * 但关掉粗筛的代价是灾难性的 —— 代码自己的注释: 旧式全量 ≈0.40·F² min/前沿, 43 层≈7 天。
 * 所以改成给锚路由一张行映射: rowmap[s] = 该抽格行对应的原始行号, rowstride = 原始 S。
 * rowmap==NULL 时退化为恒等(全量前向/量化阶段), 逐字节等价于原行为。 */
static const int *g_anc_rowmap;   /* 抽格行 → 原始行; NULL = 恒等 */
static int        g_anc_rowstride;/* 锚数据的原始行 stride */

/* ★层局部靶(2026-08-25 靶换血)★: 教师=FP专家@链态Fin(逐专家st_read, 与XR捕获同式)。
 * 旧漂移靶(把出口拉回FP锚)在强漂移区自激失控: 08-25凌晨实测 L4基线0.48→L10 13.2(每层×2),
 * z被迫打信赖域上限大修正硬拽FP→状态更离流形→下层放大→正反馈。层局部靶无漂移项。 */
static float *BF_LT=NULL; static int BF_LT_L=-1;
static void bf_fp_routed(int L,const float*Fin,const int*idx,const float*rw,int S,float*out){
    memset(out,0,(size_t)S*DIM*4);
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
        for(int i=0;i<nt;i++)for(int d2=0;d2<DIM;d2++) out[(size_t)toks[i]*DIM+d2]+=aq[(size_t)i*DIM+d2];
        free(aq); free(e1);free(e3);free(e2);
    }
    free(toks);free(wwv2);free(xs2);
}
