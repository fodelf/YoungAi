static int co_rounds(costx_t*cx,int stgno,const char*stgnm,const char*pfx){
    int L=cx->L,S=cx->S,n_fit=cx->n_fit,vs=cx->vs; size_t rowsz=cx->rowsz;
    double bval=cx->bval,bheld=cx->bheld;
    const float *Fin=cx->Fin,*Hf=cx->Hf,*H2=cx->H2,*post2=cx->post2,*comb2=cx->comb2,*shared=cx->shared,*Fcur=cx->Fcur;
    float *Fstate=cx->Fstate,*Ftest=cx->Ftest,*routedC=cx->routedC,*DF=cx->DF,*Hq=cx->Hq;
    double sc_state=*cx->sc_state,v_cur=*cx->v_cur,h_cur=*cx->h_cur;
    double fitr,valr,heldr,sc;
    (void)bheld;(void)Fcur;
    int any=0;
    int RCAP=FAST?2:512;            /* FAST: 2 轮走流程; 常规 512 仅病态保险(真收敛判据在轮尾) */
    for(int rd=1; rd<=RCAP; rd++){
        double sc_round0=sc_state;
        int improved=0;
        if(tune_over()){ printf("ROUND  L=%d 时间预算(%.0f分)耗尽 → 落当前累计最优\n",L,TUNE_MIN); fflush(stdout); break; }
        char tg[12]=""; if(rd>1||stgno>2) snprintf(tg,12,"#%d.%d",stgno,rd);
        /* -- GL 重拟合(当前余量上的层标量) -- */
        {
            co_df(Fstate,H2,post2,comb2,Hf,Hq,DF,S);
            for(size_t i=0;i<(size_t)S*DIM;i++) routedC[i]=Fstate[i]-shared[i];
            double num=0,den=1e-30;
            for(int s2=0;s2<vs;s2++){ const float*r0=routedC+(size_t)s2*DIM,*d0=DF+(size_t)s2*DIM;
                for(int d2=0;d2<DIM;d2++){ num+=(double)r0[d2]*d0[d2]; den+=(double)r0[d2]*r0[d2]; } }
            double g2=1.0+num/den; if(g2<0.25)g2=0.25; if(g2>4.0)g2=4.0;
            for(size_t i=0;i<(size_t)S*DIM;i++) Ftest[i]=shared[i]+(float)g2*routedC[i];
            co_eval(Ftest,H2,post2,comb2,Hf,Hq,S,vs,n_fit,rowsz,&fitr,&valr,&heldr,&sc);
            printf("STACK  L=%d r%d GL%s g=%.4f val=%.4f(%+.1f%%) held=%.4f(%+.2f%%) %s\n",
                   L,rd,tg,g2,valr,100*(valr-v_cur)/bval,heldr,100*(heldr-h_cur)/bheld,sc<sc_state?"✓":"✗");
            if(sc<sc_state){ sc_state=sc; v_cur=valr; h_cur=heldr; improved=1;
                float gf=(float)g2; char nm[16]; snprintf(nm,16,"%s.GL%s",pfx,tg);
                el_addp(nm,"层标量重拟合(当前余量)",4,valr,heldr,100*(bval-valr)/bval,g2,1,&gf,4);
                memcpy(Fstate,Ftest,(size_t)S*DIM*4); }
        }
        /* -- GLdyn2 范数动态(当前余量) -- */
        {
            co_df(Fstate,H2,post2,comb2,Hf,Hq,DF,S);
            for(size_t i=0;i<(size_t)S*DIM;i++) routedC[i]=Fstate[i]-shared[i];
            double nm2=0,nsd=0;
            float *xn2=malloc((size_t)S*4);
            for(int s2=0;s2<S;s2++){ const float*x=Fin+(size_t)s2*DIM; double v=0;
                for(int d2=0;d2<DIM;d2++) v+=(double)x[d2]*x[d2]; xn2[s2]=(float)sqrt(v); }
            for(int s2=0;s2<vs;s2++) nm2+=xn2[s2]; nm2/=(vs>0?vs:1);
            for(int s2=0;s2<vs;s2++){ double d=xn2[s2]-nm2; nsd+=d*d; } nsd=sqrt(nsd/(vs>0?vs:1))+1e-9;
            double G2[4]={0,0,0,0},rh2[2]={0,0};
            for(int s2=0;s2<vs;s2++){
                const float*rc=routedC+(size_t)s2*DIM,*d0=DF+(size_t)s2*DIM;
                double rd2=0,rr=0;
                for(int d2=0;d2<DIM;d2++){ rd2+=(double)rc[d2]*d0[d2]; rr+=(double)rc[d2]*rc[d2]; }
                double f1=(xn2[s2]-nm2)/nsd;
                G2[0]+=rr; G2[1]+=rr*f1; G2[3]+=rr*f1*f1;
                rh2[0]+=rr+rd2; rh2[1]+=(rr+rd2)*f1;
            }
            G2[2]=G2[1];
            double trg=G2[0]+G2[3];
            /* ★旋钮网格★: G2/rh2 与旋钮无关只累加一次; 每变体加各自 ridge 独立解+评,
             * 全变体对同一入口态 argmax, 赢基线才落地(判据与旧口径一致, 家族入账)。 */
            { double bsc=1e300,bv2=0,bh2=0,bw0=0,bw1=0; int bkv=-1;
              for(int kv=0;kv<NKG_D2;kv++){
                double Gk0=G2[0]+1e-6*trg*KGRID[kv].lfix,
                       Gk3=G2[3]+KGRID[kv].lsm*KGRID[kv].lfix*trg/2;
                double det=Gk0*Gk3-G2[1]*G2[2];
                if(fabs(det)<=1e-12) continue;
                double w0=(rh2[0]*Gk3-rh2[1]*G2[1])/det, w1=(Gk0*rh2[1]-G2[2]*rh2[0])/det;
                for(int s2=0;s2<S;s2++){ double c=w0+w1*((double)xn2[s2]-nm2)/nsd;
                    if(c<0.25)c=0.25; if(c>4.0)c=4.0;
                    float*fw=Ftest+(size_t)s2*DIM; const float*sh=shared+(size_t)s2*DIM,*rc=routedC+(size_t)s2*DIM;
                    for(int d2=0;d2<DIM;d2++) fw[d2]=sh[d2]+(float)c*rc[d2]; }
                co_eval(Ftest,H2,post2,comb2,Hf,Hq,S,vs,n_fit,rowsz,&fitr,&valr,&heldr,&sc);
                printf("STACK  L=%d r%d GLdyn2%s[%s] val=%.4f(%+.1f%%) held=%.4f(%+.2f%%) w=[%.3f,%.4f] %s\n",
                       L,rd,tg,KGRID[kv].fam,valr,100*(valr-v_cur)/bval,heldr,100*(heldr-h_cur)/bheld,w0,w1,sc<bsc?"✓best":"✗");
                if(sc<bsc){ bsc=sc; bv2=valr; bh2=heldr; bw0=w0; bw1=w1; bkv=kv; }
              }
              if(bkv>=0&&bsc<sc_state){
                double w0=bw0,w1=bw1;
                for(int s2=0;s2<S;s2++){ double c=w0+w1*((double)xn2[s2]-nm2)/nsd;
                    if(c<0.25)c=0.25; if(c>4.0)c=4.0;
                    float*fw=Ftest+(size_t)s2*DIM; const float*sh=shared+(size_t)s2*DIM,*rc=routedC+(size_t)s2*DIM;
                    for(int d2=0;d2<DIM;d2++) fw[d2]=sh[d2]+(float)c*rc[d2]; }
                sc_state=bsc; v_cur=bv2; h_cur=bh2; improved=1;
                float pp[4]={(float)w0,(float)w1,(float)nm2,(float)nsd}; char nm[16];
                snprintf(nm,16,"%s.GLdyn2%s",KGRID[bkv].lti<0?pfx:KGRID[bkv].fam,tg);
                el_addp(nm,"范数动态 c=w0+w1·zn(‖Fin‖) 加权LS+光滑(旋钮网格)",16,bv2,bh2,100*(bval-bv2)/bval,0,1,pp,16);
                if(KGRID[bkv].lti>=0){ LTW_hit[KGRID[bkv].lti]=1;
                    LTW_knob[KGRID[bkv].lti]=KGRID[bkv].lti==3?KGRID[bkv].lsm:KGRID[bkv].lti==1||KGRID[bkv].lti==4?KGRID[bkv].wcls:KGRID[bkv].lfix; }
                STK_DYN=1; STK_W0=(float)w0; STK_W1=(float)w1; STK_NM=(float)nm2; STK_SD=(float)nsd;
                memcpy(Fstate,Ftest,(size_t)S*DIM*4); }
            }
            free(xn2);
        }
        /* -- GLdyn8 PCA 动态(当前余量, 光滑 ridge 非截距) -- */
        {
            int KP=8,DP=KP+1;
            if(!STK8_V8){   /* PCA 方向只算一次(Fin 不变) */
                STK8_V8=malloc((size_t)KP*DIM*4);
                uint64_t sd=0xC0FFEE123ULL;
                float *Vt=malloc((size_t)DIM*KP*4);
                for(size_t i=0;i<(size_t)DIM*KP;i++){ sd=sd*6364136223846793005ULL+1442695040888963407ULL;
                    Vt[i]=((float)((sd>>40)&0xFFFFFF)/16777216.0f)-0.5f; }
                dq_mgs_qh(Vt,DIM,KP);
                float *pj=malloc((size_t)vs*KP*4);
                for(int it2=0;it2<4;it2++){
                    for(int s2=0;s2<vs;s2++){ const float*x=Fin+(size_t)s2*DIM; float*pr=pj+(size_t)s2*KP;
                        for(int c=0;c<KP;c++){ double a2=0; for(int d2=0;d2<DIM;d2++) a2+=(double)x[d2]*Vt[(size_t)d2*KP+c]; pr[c]=(float)a2; } }
                    memset(Vt,0,(size_t)DIM*KP*4);
                    for(int s2=0;s2<vs;s2++){ const float*x=Fin+(size_t)s2*DIM; const float*pr=pj+(size_t)s2*KP;
                        for(int d2=0;d2<DIM;d2++){ float xv=x[d2]; if(xv==0)continue; float*vt=Vt+(size_t)d2*KP;
                            for(int c=0;c<KP;c++) vt[c]+=xv*pr[c]; } }
                    dq_mgs_qh(Vt,DIM,KP);
                }
                for(int c=0;c<KP;c++) for(int d2=0;d2<DIM;d2++) STK8_V8[(size_t)c*DIM+d2]=Vt[(size_t)d2*KP+c];
                free(Vt); free(pj);
            }
            co_df(Fstate,H2,post2,comb2,Hf,Hq,DF,S);
            for(size_t i=0;i<(size_t)S*DIM;i++) routedC[i]=Fstate[i]-shared[i];
            float *phi=malloc((size_t)S*DP*4);
            dq_matmul(Fin,STK8_V8,Ftest,S,DIM,KP);
            for(int s2=0;s2<S;s2++){ phi[(size_t)s2*DP]=1.0f;
                for(int c=0;c<KP;c++) phi[(size_t)s2*DP+1+c]=Ftest[(size_t)s2*KP+c]; }
            /* ★旋钮网格★: wrow=1+w_cls·k 对 w_cls 线性 → G/rh 拆基底+线性两份只累加一次;
             * 每变体 G=G0+w·G1(+ridge) 独立解+评, 同一入口态 argmax, 赢基线才落地。 */
            double *G0=calloc((size_t)DP*DP,sizeof(double)),*G1=calloc((size_t)DP*DP,sizeof(double));
            double *rh0=calloc((size_t)DP,sizeof(double)),*rh1=calloc((size_t)DP,sizeof(double));
            double *G=malloc((size_t)DP*DP*sizeof(double)),*rh=malloc((size_t)DP*sizeof(double));
            for(int s2=0;s2<vs;s2++){
                const float*rc=routedC+(size_t)s2*DIM,*d0=DF+(size_t)s2*DIM;
                double rd2=0,rr=0,dd=0;
                for(int d2=0;d2<DIM;d2++){ rd2+=(double)rc[d2]*d0[d2]; rr+=(double)rc[d2]*rc[d2]; dd+=(double)d0[d2]*d0[d2]; }
                double kk=(rr>1e-20?dd/rr:0.0);   /* 分类/感知加权的行系数 */
                const float*ph=phi+(size_t)s2*DP;
                for(int a2=0;a2<DP;a2++){ double pa=(double)ph[a2];
                    rh0[a2]+=pa*(rr+rd2); rh1[a2]+=kk*pa*(rr+rd2);
                    for(int b2=a2;b2<DP;b2++){ double gg=rr*pa*ph[b2];
                        G0[(size_t)a2*DP+b2]+=gg; G1[(size_t)a2*DP+b2]+=kk*gg; } }
            }
            { double bsc=1e300,bv8=0,bh8=0,brh[9]; int bkv=-1;
              for(int kv=0;kv<NKGRID;kv++){
                double w=KGRID[kv].wcls;
                for(int a2=0;a2<DP;a2++){ rh[a2]=rh0[a2]+w*rh1[a2];
                    for(int b2=a2;b2<DP;b2++){ double g=G0[(size_t)a2*DP+b2]+w*G1[(size_t)a2*DP+b2];
                        G[(size_t)a2*DP+b2]=g; G[(size_t)b2*DP+a2]=g; } }
                double trg=0; for(int a2=0;a2<DP;a2++) trg+=G[(size_t)a2*DP+a2];
                G[0]+=1e-6*trg*KGRID[kv].lfix;
                for(int a2=1;a2<DP;a2++) G[(size_t)a2*DP+a2]+=KGRID[kv].lsm*KGRID[kv].lfix*trg/DP+1e-9;
                if(cholesky(G,(uint32_t)DP)!=0) continue;
                cholesky_solve(G,(uint32_t)DP,rh);
                for(int s2=0;s2<S;s2++){ const float*ph=phi+(size_t)s2*DP;
                    double c=0; for(int a2=0;a2<DP;a2++) c+=rh[a2]*(double)ph[a2];
                    if(c<0.25)c=0.25; if(c>4.0)c=4.0;
                    float*fw=Ftest+(size_t)s2*DIM; const float*sh=shared+(size_t)s2*DIM,*rc=routedC+(size_t)s2*DIM;
                    for(int d2=0;d2<DIM;d2++) fw[d2]=sh[d2]+(float)c*rc[d2]; }
                co_eval(Ftest,H2,post2,comb2,Hf,Hq,S,vs,n_fit,rowsz,&fitr,&valr,&heldr,&sc);
                printf("STACK  L=%d r%d GLdyn8%s[%s] val=%.4f(%+.1f%%) held=%.4f(%+.2f%%) %s\n",
                       L,rd,tg,KGRID[kv].fam,valr,100*(valr-v_cur)/bval,heldr,100*(heldr-h_cur)/bheld,sc<bsc?"✓best":"✗");
                if(sc<bsc){ bsc=sc; bv8=valr; bh8=heldr; bkv=kv;
                    for(int a2=0;a2<DP;a2++) brh[a2]=rh[a2]; }
              }
              if(bkv>=0&&bsc<sc_state){
                for(int s2=0;s2<S;s2++){ const float*ph=phi+(size_t)s2*DP;
                    double c=0; for(int a2=0;a2<DP;a2++) c+=brh[a2]*(double)ph[a2];
                    if(c<0.25)c=0.25; if(c>4.0)c=4.0;
                    float*fw=Ftest+(size_t)s2*DIM; const float*sh=shared+(size_t)s2*DIM,*rc=routedC+(size_t)s2*DIM;
                    for(int d2=0;d2<DIM;d2++) fw[d2]=sh[d2]+(float)c*rc[d2]; }
                sc_state=bsc; v_cur=bv8; h_cur=bh8; improved=1;
                float pw[9]; for(int a2=0;a2<DP;a2++) pw[a2]=(float)brh[a2];
                char nm[16]; snprintf(nm,16,"%s.GLdyn8%s",KGRID[bkv].lti<0?pfx:KGRID[bkv].fam,tg);
                el_addp(nm,"PCA8 动态 9-dof 加权LS+光滑ridge(旋钮网格)",
                        STK8_ON?36:(uint64_t)(36+8*DIM*2),bv8,bh8,100*(bval-bv8)/bval,0,1,pw,36);
                if(KGRID[bkv].lti>=0){ LTW_hit[KGRID[bkv].lti]=1;
                    LTW_knob[KGRID[bkv].lti]=KGRID[bkv].lti==3?KGRID[bkv].lsm:KGRID[bkv].lti==1||KGRID[bkv].lti==4?KGRID[bkv].wcls:KGRID[bkv].lfix; }
                STK8_ON=1;
                memcpy(Fstate,Ftest,(size_t)S*DIM*4); }
            }
            free(G0);free(G1);free(rh0);free(rh1);free(G);free(rh);free(phi);
        }
        double dsc=sc_round0>1e-30?(sc_round0-sc_state)/sc_round0:0.0;
        printf("ROUND  L=%d 阶段=%s r%d %s val=%.4f held=%.4f 本轮改善=%.3f%%\n",
               L,stgnm,rd,improved?(dsc<5e-4?"接管但<0.05%→数值收敛":"有接管"):"零接管→收敛",v_cur,h_cur,100.0*dsc);
        fflush(stdout);
        if(improved) any=1;
        if(!improved||dsc<5e-4) break;   /* 收敛=零接管 或 单轮改善<0.05% */
        if(rd==RCAP&&!FAST) fprintf(stderr,"[!] L%d %s 撞 512 轮保险帽仍在改善 — 请报告\n",L,stgnm);
    }
    *cx->sc_state=sc_state; *cx->v_cur=v_cur; *cx->h_cur=h_cur;
    return any;
}
static void co_ex_reset(coexp_t*ex){
    for(int e=0;e<NEXP;e++){ coexp_t*ce=&ex[e];
        free(ce->hc_cal);free(ce->calS);free(ce->calW);free(ce->hc_all);free(ce->hitS);free(ce->hitW);
        free(ce->y2ref);free(ce->yq_all);free(ce->yq_cal);free(ce->gcal);free(ce->gden);free(ce->ccal);free(ce->alpha);
        memset(ce,0,sizeof(*ce)); }
}
static void coadapt_moe(int L,int S,int n_fit,const float*Fin,const int*idx,const float*rw,
                        float*Fout,const float*H2,const float*post2,const float*comb2,lstat_t*st){
    size_t rowsz=(size_t)HCM*DIM, lstride=(size_t)S*rowsz;
    const float *Hf=ANC.H+(size_t)L*lstride;   /* 目标仍是 FP 输出(要还原原模型) */
    const float *afin=Fin;   /* ★累加前层★: 校准/锚基底=当前累积激活(前层量化+z 已注入), 非 FP 锚 */
    int vs=(n_fit*3)/4, nval=n_fit-vs; if(nval<8){ vs=n_fit; nval=0; }
    float *shared=malloc((size_t)S*DIM*4); memcpy(shared,Fout,(size_t)S*DIM*4);
    coexp_t *ex=calloc((size_t)NEXP,sizeof(coexp_t));
    float *routed=malloc((size_t)S*DIM*4),*Fcur=malloc((size_t)S*DIM*4),*Ftest=malloc((size_t)S*DIM*4);
    float *FoutBest=malloc((size_t)S*DIM*4);
    float *Hq=malloc(lstride*4);
    float *DF=malloc((size_t)S*DIM*4),*colw=malloc((size_t)DIM*4);
    CUR_L=L; CO_LT0=time(NULL);
    STK_DYN=0; STK8_ON=0;                    /* 层间清零(修 L1+ 乱序: PCA方向/动态参数不跨层复用) */
    if(STK8_V8){ free(STK8_V8); STK8_V8=NULL; }
    if(STK_CLSV){ free(STK_CLSV); STK_CLSV=NULL; }
    if(g_cli.layer_dir){   /* 导出口径=调优口径: 存本层累积激活+实际路由 */
        if(!EXP_FIN) EXP_FIN=malloc((size_t)S*DIM*4);
        if(!EXP_IDX) EXP_IDX=malloc((size_t)S*NACT_RT*sizeof(int));
        memcpy(EXP_FIN,Fin,(size_t)S*DIM*4);
        memcpy(EXP_IDX,idx,(size_t)S*NACT_RT*sizeof(int)); EXP_L=L;
    }
    /* ★向前叠加可观测★: 本层输入=上游累积(前层量化+z 注入)。量化上游漂移 vs FP 输入锚 */
    { const float *fpin=ANC.fin+(size_t)L*S*DIM; double e2=0,a2=0;
      for(size_t i=0;i<(size_t)S*DIM;i++){ double d=(double)Fin[i]-fpin[i]; e2+=d*d; a2+=(double)fpin[i]*fpin[i]; }
      double up=sqrt(e2/(a2+1e-30));
      printf("INHERIT L=%d 上游累积漂移(输入 vs FP)=%.4f %s\n",L,up,L==0?"(L0 无上游, 应≈0)":"(继承前层量化)");
      { char rs[48]; snprintf(rs,48,"上游漂移=%.4f",up);
        mlog(L,"向前叠加",L==0?"L0起点":"继承前层累积",up<1e-6?"无上游":"已叠加",0,rs,"基线"); }
      fflush(stdout); }
    fprintf(stderr,"\n[逐层搜索 L%02d] base(signref μ=%.0f×%d) + 修正菜单{GL,GE,CE(λ扫),ZL(k/λ扫),BWD} fit=%d val=%d held=%d\n",
            L,dq_signref_mu,dq_signref_rounds,vs,nval,S-n_fit);
    /* ==== 阶段1: 量化调优(体积不变, 同 1bit 不同算法/参数, 统一 val 判优) ==== */
    NELE=0;
    struct { const char*nm; double mu; int rounds; } QC[]={
        {"符号精修μ10×3轮",10,3},{"符号精修μ3×3轮(弱锚)",3,3},{"符号精修μ30×3轮(强锚)",30,3},
        {"符号精修μ10×6轮(深精修)",10,6},{"行scale闭式(0轮翻转)",0,0},
    };
    /* FAST: 只跑首个变体走流程; VQ 模式: μ 只剩冷 w2 一处消费, 5 配置全层重量化=4/5 空转
     * (VQ 编码单价又是 signref 3-5×), 裁到单配置 μ10×3(生产惯用中档) */
    const int nqc=(FAST||dq_vq_on()||g_cli.minvol)?1:(int)(sizeof(QC)/sizeof(QC[0]));   /* minvol: g10 定稿单配置(g10=历史最优代表, 13.4→~3min/层) */
    const uint64_t QVOL=(uint64_t)NEXP*(2*(uint64_t)MOEI*go1b_blk_row_bytes(DIM)+(uint64_t)DIM*go1b_blk_row_bytes(MOEI));
    int qbest=-1; double qsc=1e300,qv=0,qh=0,qf=0; long nflip=0;
    double bfit,bval,bheld,scb;
    mlog(L,"量化","阶段1·量化调优(5候选,体积不变)","开始",QVOL,"-","进行中");
    for(int qi=0;qi<nqc;qi++){
        dq_signref_mu=QC[qi].mu; dq_signref_rounds=QC[qi].rounds;
        co_ex_reset(ex);
        long nf=co_base_pass(L,S,n_fit,0,Fin,idx,rw,ex,routed,qi==0?st:NULL);
        for(size_t i=0;i<(size_t)S*DIM;i++) Fcur[i]=shared[i]+routed[i];
        co_eval(Fcur,H2,post2,comb2,Hf,Hq,S,vs,n_fit,rowsz,&bfit,&bval,&bheld,&scb);
        printf("SEARCH L=%d QT[%s] val=%.4f held=%.4f fit=%.4f sc=%.5g\n",L,QC[qi].nm,bval,bheld,bfit,scb);
        fflush(stdout);
        { char rs[64]; snprintf(rs,64,"val=%.4f held=%.4f",bval,bheld);
          mlog(L,"量化",QC[qi].nm,"候选完成",QVOL,rs,scb<qsc?"暂最优":"落选"); }
        if(scb<qsc){ qsc=scb; qbest=qi; nflip=nf; qv=bval; qh=bheld; qf=bfit; }
    }
    dq_signref_mu=QC[qbest].mu; dq_signref_rounds=QC[qbest].rounds;
    if(qbest!=nqc-1){   /* 胜者非最后评估者 → 确定性重跑胜者恢复缓存状态 */
        co_ex_reset(ex);
        nflip=co_base_pass(L,S,n_fit,0,Fin,idx,rw,ex,routed,NULL);
        for(size_t i=0;i<(size_t)S*DIM;i++) Fcur[i]=shared[i]+routed[i];
    }
    bval=qv; bheld=qh; bfit=qf; scb=qsc;
    printf("SEARCH L=%d base       val=%.4f held=%.4f fit=%.4f sc=%.5g w2flip=%.3f%%\n",
           L,bval,bheld,bfit,scb,100.0*(double)nflip/((double)DIM*MOEI*NEXP));
    { char qalgo[64];
      snprintf(qalgo,sizeof(qalgo),"量化调优胜者:%s(锚校准+w2顺序补偿)",QC[qbest].nm);
      el_add("1bit",qalgo,QVOL,bval,bheld,0,0,1); }
    mlog(L,"量化",QC[qbest].nm,"阶段1收敛",QVOL,"见base行","✓胜者落地");
    int fastlane=0;   /* 收敛优先: 无快车道, 每层全菜单 */
    fprintf(stderr,"\n[菜单] GL/GE×2/CE×6 逐个评估(秒级)...\n");
    co_df(Fcur,H2,post2,comb2,Hf,Hq,DF,S);   /* 累积回拉目标 */
    /* 归因分母(实际路由/锚路由) */
    float *sw2act=calloc((size_t)S,4),*sw2anc=calloc((size_t)S,4);
    for(int s=0;s<S;s++){ double ta=0;   /* 累加口径: 归因分母用实际路由权 */
        for(int a=0;a<NACT_RT;a++){ double v=rw[(size_t)s*NACT_RT+a]; ta+=v*v; }
        sw2act[s]=(float)ta; sw2anc[s]=(float)ta; }
    /* ---- 搜索状态: 永不空手 —— 无提升≠丢弃, 换形态继续; GL(1 dof)兜底 ---- */
    typedef struct { char algo[12]; float lam,g; int k,bwd; double fit,val,held,sc; } win_t;
    win_t win; memset(&win,0,sizeof(win)); win.sc=1e300;
    double fitr,valr,heldr,sc;
    #define CK(NM,LM,GG,KK) do{ if(sc<win.sc){ win.sc=sc; snprintf(win.algo,12,"%s",NM); \
        win.lam=(float)(LM); win.g=(float)(GG); win.k=(KK); win.bwd=0; \
        win.fit=fitr; win.val=valr; win.held=heldr; memcpy(FoutBest,Ftest,(size_t)S*DIM*4); } }while(0)
    /* GL: 静态层标量(零方差兜底) */
    { double num=0,den=1e-30;
      for(int s=0;s<vs;s++){ const float*r0=routed+(size_t)s*DIM,*d0=DF+(size_t)s*DIM;
          for(int d2=0;d2<DIM;d2++){ num+=(double)r0[d2]*d0[d2]; den+=(double)r0[d2]*r0[d2]; } }
      double g=1.0+num/den; if(g<0.25)g=0.25; if(g>4.0)g=4.0;
      for(size_t i=0;i<(size_t)S*DIM;i++) Ftest[i]=shared[i]+(float)g*routed[i];
      co_eval(Ftest,H2,post2,comb2,Hf,Hq,S,vs,n_fit,rowsz,&fitr,&valr,&heldr,&sc);
      printf("SEARCH L=%d GL         val=%.4f(%+.1f%%) held=%.4f(%+.2f%%) fit=%.4f sc=%.5g g=%.4f\n",
             L,valr,100*(valr-bval)/bval,heldr,100*(heldr-bheld)/bheld,fitr,sc,g);
      el_add("z.GL","层标量 g=1+⟨routed,DF⟩/⟨routed,routed⟩",4,valr,heldr,100*(bval-valr)/bval,g,valr<bval?1:3);
      CK("GL",0,g,0);
    }
    /* GE: 静态每专家增益, 两种目标(自误差/累积) */
    if(!fastlane) for(int tgt=0;tgt<2;tgt++){
        for(int e=0;e<NEXP;e++){ coexp_t*ce=&ex[e]; ce->gstat=0.0f;
            double num=0,den=1e-30;
            if(tgt==0){ if(!ce->gcal)continue;
                for(int t=0;t<ce->ncal;t++){ if(ce->calS[t]>=vs)continue;
                    num+=(double)ce->gcal[t]*ce->gden[t]; den+=ce->gden[t]; }
                if(den>1e-20){ double g=num/den; if(g<0.25)g=0.25; if(g>4.0)g=4.0; ce->gstat=(float)g; }
            } else { if(!ce->yq_all)continue;
                for(int i=0;i<ce->nhit;i++){ int s=ce->hitS[i]; if(s>=vs)continue;
                    double w=ce->hitW[i],a=w*w/(sw2act[s]>1e-9?sw2act[s]:1e-9);
                    const float*y=ce->yq_all+(size_t)i*DIM,*d0=DF+(size_t)s*DIM;
                    double yd=0,yy=0;
                    for(int d2=0;d2<DIM;d2++){ yd+=(double)y[d2]*d0[d2]; yy+=(double)y[d2]*y[d2]; }
                    num+=a*w*yd; den+=w*w*yy; }
                double g=1.0+num/den; if(g<0.25)g=0.25; if(g>4.0)g=4.0; ce->gstat=(float)g;
            } }
        co_apply_mult(ex,afin,Fin,shared,S,Ftest,NULL,NULL);
        co_eval(Ftest,H2,post2,comb2,Hf,Hq,S,vs,n_fit,rowsz,&fitr,&valr,&heldr,&sc);
        printf("SEARCH L=%d %-10s val=%.4f(%+.1f%%) held=%.4f(%+.2f%%) fit=%.4f sc=%.5g\n",
               L,tgt?"GEcum":"GEself",valr,100*(valr-bval)/bval,heldr,100*(heldr-bheld)/bheld,fitr,sc);
        el_add(tgt?"z.GEcum":"z.GEself",tgt?"每专家静态增益·累积目标":"每专家静态增益·自误差",
               (uint64_t)NEXP*2,valr,heldr,100*(bval-valr)/bval,0,valr<bval?2:3);
        CK(tgt?"GEcum":"GEself",0,0,0);
        for(int e=0;e<NEXP;e++) ex[e].gstat=0.0f;
    }
    /* CE: 动态每专家系数 c_e(x), 自误差目标 λ 扫 + 累积目标 λ 扫 */
    if(!fastlane){ const double LSW[4]={0.3,1,3,10};
      float *gt=malloc((size_t)n_fit*4);
      for(int tgt=0;tgt<2;tgt++) for(int li=0;li<(tgt?2:4);li++){
        double lam=tgt?(li?10.0:1.0):LSW[li];
        double cmean,cmx;
        for(int e=0;e<NEXP;e++){ coexp_t*ce=&ex[e]; ce->gstat=0.0f;
            if(ce->ncal<1){ continue; }
            const float *g_use=ce->gcal;
            if(tgt){ /* 累积目标: c=1+(w/sw2)·⟨DF,ŷ⟩/⟨ŷ,ŷ⟩ (锚路由归因) */
                for(int t=0;t<ce->ncal;t++){ int s=ce->calS[t];
                    const float*y=ce->yq_cal+(size_t)t*DIM,*d0=DF+(size_t)s*DIM;
                    double yd=0;
                    for(int d2=0;d2<DIM;d2++) yd+=(double)y[d2]*d0[d2];
                    double den=ce->gden[t]>1e-20?ce->gden[t]:1e-20;
                    double g=1.0+((double)ce->calW[t]/(sw2anc[s]>1e-9?sw2anc[s]:1e-9))*yd/den;
                    if(g<0.25)g=0.25; if(g>4.0)g=4.0; gt[t]=(float)g; }
                g_use=gt;
            }
            co_fit_ze(ce,afin,g_use,lam,vs);
        }
        co_apply_mult(ex,afin,Fin,shared,S,Ftest,&cmean,&cmx);
        co_eval(Ftest,H2,post2,comb2,Hf,Hq,S,vs,n_fit,rowsz,&fitr,&valr,&heldr,&sc);
        printf("SEARCH L=%d %s λ=%-4g val=%.4f(%+.1f%%) held=%.4f(%+.2f%%) fit=%.4f sc=%.5g c[|Δ|=%.4f max=%.2f]\n",
               L,tgt?"CEcum ":"CEself",lam,valr,100*(valr-bval)/bval,heldr,100*(heldr-bheld)/bheld,fitr,sc,cmean,cmx);
        { char al2[64]; snprintf(al2,64,"%s kernel-ridge λ=%g",tgt?"每专家动态·累积":"每专家动态·自误差",lam);
          el_add(tgt?"z.CEcum":"z.CEself",al2,(uint64_t)NEXP*DIM*2,valr,heldr,100*(bval-valr)/bval,lam,valr<bval?2:3); }
        CK(tgt?"CEcum":"CEself",lam,0,0);
      }
      free(gt);
    }
    /* (ZL 加法低秩菜单已删: 用户裁决删除加法元素(2026-07-11)后仅存 DS4_ADD_FORMS 复活口,
     * 无脚本设置, 2026-08-31 env 清退连同独占代码一并移除) */
