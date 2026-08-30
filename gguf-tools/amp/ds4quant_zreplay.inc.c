/* ds4quant_zreplay.inc.c — 冻结 z^L 回放(type6)的行并行版。物理分片(p4 500 行守卫),
 * 只被 ds4quant_run_p4.inc.c include; 逻辑上是 bytes_moe 修正链的 type6 分支, 不是另一份实现。
 * ★为什么(2026-08-30 BFLT 二级账)★: 双投影 zk=64(zdin 可为 3·DIM 的 φ 特征)每行 ~50 万乘,
 * 单线程 38s/深单元 —— 链桶几乎全是它。行独立(每行只写自己的 fw 段), 按行切线程 ⇒
 * 模型数值逐位不变; zdiag 聚合按线程序求和, 只动诊断打印的浮点末位, 不进模型。 */
typedef struct { const lop_t*o; const float*Fin,*Fbase; float*Fout; int s0,s1,zk,zdin;
                 double nd,nf,sc; long long n,clip; } zrw_t;
static void *zrep_worker(void*a){
    zrw_t*w=a; const lop_t*o=w->o; const int zk=w->zk,zdin=w->zdin;
    float *zd=malloc((size_t)DIM*4); double *pv=malloc((size_t)zk*8);
    float *phi=(zdin==3*DIM)?malloc((size_t)zdin*4):NULL;
    for(int s2=w->s0;s2<w->s1;s2++){
        const float*x=w->Fin+(size_t)s2*DIM; float*fw=w->Fout+(size_t)s2*DIM;
        const float*fb=w->Fbase+(size_t)s2*DIM;   /* 信赖域基准=routed: 与引擎 λ 点同口径 */
        const float*xin=x;
        if(phi){   /* φ=[x, x⊙x/rms, relu(x)] — 与 zlayer zl_phi 逐式一致 */
            double ss=0; for(int d2=0;d2<DIM;d2++) ss+=(double)x[d2]*x[d2];
            float nrm=(float)sqrt(ss/DIM)+1e-6f;
            for(int d2=0;d2<DIM;d2++){ phi[d2]=x[d2]; phi[DIM+d2]=x[d2]*x[d2]/nrm; phi[2*DIM+d2]=x[d2]>0?x[d2]:0; }
            xin=phi; }
        for(int c=0;c<zk;c++){ double a2=0; const float*vv=o->zlV;
            for(int d2=0;d2<zdin;d2++) a2+=(double)xin[d2]*vv[(size_t)d2*zk+c];
            pv[c]=a2*(double)o->zlz[c]; }
        double nd=0,nf=0;
        for(int d2=0;d2<DIM;d2++){ double a2=0; const float*uu=o->zlU+(size_t)d2*zk;
            for(int c=0;c<zk;c++) a2+=pv[c]*(double)uu[c];
            zd[d2]=(float)a2; nd+=a2*a2;
            double rt=(double)fw[d2]-fb[d2]; nf+=rt*rt; }
        nd=sqrt(nd); nf=sqrt(nf);
        double cap=(double)o->zltr*nf; float sc2=1.0f;
        if(nd>cap&&nd>0) sc2=(float)(cap/nd);
        for(int d2=0;d2<DIM;d2++) fw[d2]+=sc2*zd[d2];
        w->nd+=nd; w->nf+=nf; w->sc+=sc2; if(sc2<1.0f)w->clip++; w->n++;
    }
    free(zd); free(pv); if(phi)free(phi); return NULL;
}
static void zrep_par(const lop_t*o,const float*Fin,const float*Fbase,float*Fout,int S){
    const int zk=o->zlk, zdin=o->zdin>0?o->zdin:DIM;   /* ★md86: 3*DIM=ftA */
    int nth=NTHREADS>1?NTHREADS:1; if(nth>S)nth=S; if(nth>32)nth=32;
    zrw_t w[32]; pthread_t th[32];
    for(int t=0;t<nth;t++){
        w[t]=(zrw_t){o,Fin,Fbase,Fout,S*t/nth,S*(t+1)/nth,zk,zdin,0,0,0,0,0};
        pthread_create(&th[t],NULL,zrep_worker,&w[t]); }
    double nd=0,nf=0,sc=0; long long n=0,clip=0;
    for(int t=0;t<nth;t++){ pthread_join(th[t],NULL);
        nd+=w[t].nd; nf+=w[t].nf; sc+=w[t].sc; n+=w[t].n; clip+=w[t].clip; }
    /* z 回放插桩(2026-08-26 用户令"打日志找"): 与解算侧同口径, 两边对不上即回放路 bug */
    if(n) fprintf(stderr,"[zdiag]L%02d k=%d tr=%.2f 行=%lld |z|/|routed|=%.4f "
        "夹持率=%.1f%% 平均缩放=%.3f\n", g_replay_cur_L, zk, (double)o->zltr, n,
        nf>0?nd/nf:0.0, 100.0*clip/n, sc/n);
}
