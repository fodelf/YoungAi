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
                scC=co_score_rows(Hg,eTgt,SEV,nSEV,rowsz); free(Hg);
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
                scD=co_score_rows(Hg,eTgt,SEV,nSEV,rowsz); free(Hg);
                if(gbak){ memcpy(lf->ops[go].ge,gbak,(size_t)NEXP*4); free(gbak); }   /* 复原 */
                else { lf->nops--; }   /* 临时 op 移除(geD 保留待提交) */
            } else { free(geD); geD=NULL; }
        }
        /* --- 择优提交(字节回放分, 与合并同口径; 永不劣化; 同改善优先自有 z 重解=机制本体) --- */
        double bestsc=base; int form=0;
        if(scE<bestsc-1e-9){ bestsc=scE; form=5; }
        if(scF<bestsc-1e-9){ bestsc=scF; form=6; }
        if(scA<bestsc-1e-9){ bestsc=scA; form=1; }
        if(scB<bestsc-1e-9){ bestsc=scB; form=2; }
        if(scC<bestsc-1e-9){ bestsc=scC; form=3; }
        if(scD<bestsc-1e-9){ bestsc=scD; form=4; }
