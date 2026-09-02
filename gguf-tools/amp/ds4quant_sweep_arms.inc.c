/* ds4quant_sweep_arms.inc.c — sweep 的形态C(bf.GLdyn2 范数特征)与形态D(bf.GE per-expert
 * 投影)候选臂。物理分片, 只被 ds4quant_run_p13.inc.c 在单元循环【体内】include —— 它就是
 * 那段代码, 不是函数; 依赖当场变量(nFV/FITc/scr/ms/effw/SEV/nSEV/GS_*)。 */
        /* --- 形态C: per-token 动态 → bf.GLdyn2(范数特征拟合) --- */
        double scC=base; lop_t opC; memset(&opC,0,sizeof(opC));
        if(nmix>=2&&lf->nops<32){
            double mu=0,sd=0; float*fn=malloc((size_t)S*4);
            for(int s=0;s<eS;s++){ const float*fx=GS_FIN+(size_t)s*DIM; double v=0;
                for(int d=0;d<DIM;d++) v+=(double)fx[d]*fx[d]; fn[s]=(float)sqrt(v); }
            /* fit 行统计走视图下标(升序 sidx 后前段不再纯 fit) */
            for(int i=0;i<nFV;i++) mu+=fn[scr?FITc[i]:i]; mu/=(nFV>0?nFV:1);
            for(int i=0;i<nFV;i++){ double d=fn[scr?FITc[i]:i]-mu; sd+=d*d; } sd=sqrt(sd/(nFV>0?nFV:1))+1e-9;
            /* 联合损失单次求解(dyn2 新建: 行权恒开+λ df 锚定) */
            { double A[4]={0,0,0,0},b2[2]={0,0},x2[2];
              for(int i=0;i<nFV;i++){ const int s=scr?FITc[i]:i;
                  double f=((double)fn[s]-mu)/sd, wr=(double)effw[s];
                  A[0]+=wr; A[1]+=wr*f; A[3]+=wr*f*f; b2[0]+=wr*ms[s]; b2[1]+=wr*f*ms[s]; }
              A[2]=A[1];
              double lam=1e-3*(A[0]+A[3])/2.0+1e-9;
              A[0]+=lam; A[3]+=lam;
              if(solve_sym(A,b2,2,x2)==0){
                opC.type=2; opC.w2p[0]=(float)x2[0]; opC.w2p[1]=(float)x2[1];
                opC.w2p[2]=(float)mu; opC.w2p[3]=(float)sd;
                int tmp=lf->nops; lf->ops[tmp]=opC; lf->nops++;
                float*Hg=gs_forward_exit(J,eF,eHin,eIds,eS,eNf,NULL);
                scC=co_score_rows(Hg,eTgt,SEV,nSEV,rowsz,1); free(Hg);
                lf->nops--;
              } }
            free(fn);
        }
        /* --- 形态D: per-expert 增益投影(256-dof; per-token 目标 m*_s 按路由权重投到专家) --- */
        double scD=base; float *geD=NULL;
        if(nmix>=1&&lf->nops<32){
            geD=malloc((size_t)NEXP*4);
            double *gnum=calloc((size_t)NEXP,sizeof(double)),*gden=calloc((size_t)NEXP,sizeof(double));
            for(int i=0;i<nFV;i++){ const int s=scr?FITc[i]:i;
                for(int a2=0;a2<NACT;a2++){
                    int e=GS_IDXC[(size_t)s*NACT+a2]; double w=GS_RWC[(size_t)s*NACT+a2];
                    if(e>=0&&e<NEXP&&w>0){ gnum[e]+=w*ms[s]; gden[e]+=w; } } }
            int nge=0;
            for(int e=0;e<NEXP;e++){ double g=gden[e]>1e-12?gnum[e]/gden[e]:1.0;
                if(g<0.80)g=0.80; if(g>1.20)g=1.20; geD[e]=(float)g; if(g!=1.0)nge++; }
            free(gnum); free(gden);
            if(nge>=2){
                int go=BF_GEOP[J]; float *gbak=NULL;
                if(go>=0){ gbak=malloc((size_t)NEXP*4); memcpy(gbak,lf->ops[go].ge,(size_t)NEXP*4);
                    for(int e=0;e<NEXP;e++) lf->ops[go].ge[e]*=geD[e]; }   /* 已有: 临时累乘 */
                else { go=lf->nops; memset(&lf->ops[go],0,sizeof(lop_t));
                    lf->ops[go].type=5; lf->ops[go].ge=geD; lf->nops++; }
                float*Hg=gs_forward_exit(J,eF,eHin,eIds,eS,eNf,NULL);
                scD=co_score_rows(Hg,eTgt,SEV,nSEV,rowsz,1); free(Hg);
                if(gbak){ memcpy(lf->ops[go].ge,gbak,(size_t)NEXP*4); free(gbak); }   /* 复原 */
                else { lf->nops--; }   /* 临时 op 移除(geD 保留待提交) */
            } else { free(geD); geD=NULL; }
        }
        double jdl;   /* 本层最优候选Δ%(隐分口径, 仅诊断/日志; 负=候选都更差)。必须在真尺块
                       * 之前算: 放行会把 base 换成 bkl 行隐分, 之后再算就是跨口径假数。 */
        { double cand=scE; if(scF<cand)cand=scF; if(scA<cand)cand=scA; if(scB<cand)cand=scB; if(scC<cand)cand=scC; if(scD<cand)cand=scD;
          double dl=(base-cand)/(base>1e-12?base:1); jdl=100.0*dl;
          if(dl>bf_bestdl){ bf_bestdl=dl; bf_bestJ=J; } }
        /* --- ★真尺择优(2026-09-01 用户令"指标换成真尺不要假肉")★ ---
         * 旧口径 = 臂间按隐分(co_score 前沿出口)取最小, 再对唯一胜者打一枪 bkl 复核 ——
         * 隐分与部署KL不对齐时(chain9 实锤"层内优/端到端劣"), 真尺上更优的臂可能根本
         * 没进复核。新口径: 隐分只当【臂内拟合器 + 成臂条件】; 臂间裁决与放行一律部署KL
         * (bkl_gate 2026-09-01 二修: 全序列真值回放+LEV 八域全位置打分行, 与终验/caliper
         * 同工作点 —— 尾块口径"补晚伤早"已定罪, 见 p12 bkl_init 注): 每个成臂候选
         * apply→bkl→revert 各打一枪, 取 KL 最低的臂, 且必须低于本单元基线 KL 才落地
         * (不降=拒)。成本账: 每枪=(Lfront−J+1)层×全 S 行前向+head, 单元 1+臂数≤7 枪。 */
        double bestsc=base; int form=0;
        int cforms[6],ncf=0;
        if(ze>=0&&opE.type&&scE<base-1e-9) cforms[ncf++]=5;
        if(opF.V8&&scF<base-1e-9)          cforms[ncf++]=6;
        if(scA<base-1e-9)                  cforms[ncf++]=1;
        if(to>=0&&tB!=0&&scB<base-1e-9)    cforms[ncf++]=2;
        if(opC.type&&scC<base-1e-9)        cforms[ncf++]=3;
        if(geD&&scD<base-1e-9)             cforms[ncf++]=4;
        if(ncf&&!BKL_FPLP){ vrej++; ncf=0; }   /* 闸材料缺失: fail-closed 拒落地 */
        if(ncf){
            double basef=0,c0=0,m0=0;
            double basekl=bkl_gate(J,Lfront,Hin,rowsz,S,ids,n_fit,&basef,&c0,&m0);
            double bkl_best=basekl, sc_best=basef;
            for(int ci=0;ci<ncf;ci++){ int f=cforms[ci];
                double scv=0,c1=0,m1=0;
                lop_t svE; float svg=0,svt=0,svw2[4]; float*gbak=NULL; int tmpop=-1;
                memset(&svE,0,sizeof(svE));
                if(f==5){ svE=lf->ops[ze]; lf->ops[ze]=opE; }
                else if(f==6){ tmpop=lf->nops; lf->ops[tmpop]=opF; lf->nops++; }
                else if(f==1){ if(fo>=0){ svg=lf->ops[fo].g; lf->ops[fo].g=baseg*aA; }
                    else if(lf->nops<32){ tmpop=lf->nops; memset(&lf->ops[tmpop],0,sizeof(lop_t));
                        lf->ops[tmpop].type=1; lf->ops[tmpop].g=aA; lf->nops++; }
                    else continue; }
                else if(f==2){ svt=lf->ops[to].t; lf->ops[to].t=tB; }
                else if(f==3){ int od=BF_DYN2OP[J];
                    if(od>=0){ memcpy(svw2,lf->ops[od].w2p,16); memcpy(lf->ops[od].w2p,opC.w2p,16); }
                    else if(lf->nops<32){ tmpop=lf->nops; lf->ops[tmpop]=opC; lf->nops++; }
                    else continue; }
                else { int go=BF_GEOP[J];
                    if(go>=0){ gbak=malloc((size_t)NEXP*4); memcpy(gbak,lf->ops[go].ge,(size_t)NEXP*4);
                        for(int e=0;e<NEXP;e++) lf->ops[go].ge[e]*=geD[e]; }
                    else if(lf->nops<32){ tmpop=lf->nops; memset(&lf->ops[tmpop],0,sizeof(lop_t));
                        lf->ops[tmpop].type=5; lf->ops[tmpop].ge=geD; lf->nops++; }   /* geD 所有权不转移 */
                    else continue; }
                double candkl=bkl_gate(J,Lfront,Hin,rowsz,S,ids,n_fit,&scv,&c1,&m1);
                if(f==5) lf->ops[ze]=svE;
                else if(f==6) lf->nops--;
                else if(f==1){ if(tmpop>=0) lf->nops--; else lf->ops[fo].g=svg; }
                else if(f==2) lf->ops[to].t=svt;
                else if(f==3){ if(tmpop>=0) lf->nops--; else memcpy(lf->ops[BF_DYN2OP[J]].w2p,svw2,16); }
                else { if(tmpop>=0) lf->nops--; else memcpy(lf->ops[BF_GEOP[J]].ge,gbak,(size_t)NEXP*4); }
                if(gbak){ free(gbak); gbak=NULL; }
                printf("[BKL] L=%02d 臂%d: 部署KL %.5f→%.5f%s 隐分%.5g cos%.5f\n",
                       J,f,basekl,candkl,candkl<bkl_best-1e-9?"(新优)":"",scv,c1); fflush(stdout);
                if(candkl<bkl_best-1e-9){ bkl_best=candkl; form=f; sc_best=scv; }
            }
            g_anc_rowmap=scr?sidx:NULL;               /* 还原本单元的粗筛映射 */
            if(form){ base=basef; bestsc=sc_best; eFlog=Lfront;
                printf("[BKL] L=%02d 真尺裁决: 臂%d 部署KL %.5f→%.5f 降✓放行\n",J,form,basekl,bkl_best); }
            else { vrej++;
                printf("[BKL] L=%02d 真尺裁决: %d 臂全拒(基线KL %.5f 无一真降)\n",J,ncf,basekl); }
            fflush(stdout);
        }
