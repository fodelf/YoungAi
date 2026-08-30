/* ds4quant_moe_gpu.inc.c — GPU 批量专家前向的宿主侧(gather/提交/scatter)。
 * 物理分片, 只被 ds4quant_run_p4.inc.c include; 逻辑上是 bytes_moe 的 GPU 分支, 不是另一份实现。
 * 设备侧核: fused=vq_gpu_moefused.inc.cu(首选, dequant 融进 GEMM), 旧路=vq_gpu_moe.inc.cu。 */
#ifdef DS4QUANT_CUDA
static const int *g_anc_rowmap;   /* 前向声明: 定义在 p6:360(本分片被 p4 include, 在 p6 之前);
                                   * 同 TU 双 tentative definition 合并, 合法 */
static int bmw_batch_dequant(lfile_t*,int,int);   /* 定义在 p4 本 include 之后(同 TU) */
extern int vqg_moe_batch(const float*,const float*,const float*,float*,int,int,int,int,float);
/* ★与 vq_gpu_moefused.inc.cu 的 vqg_fj 必须逐字段同步(双侧 POD, 漂移=错位读)★ */
typedef struct { uint64_t o1,o3,o2; int nc13,nb13,nc2,nb2,nt; } vqg_fj;
extern int vqg_moe_batch_fused(const uint8_t*,const vqg_fj*,const float*,const float*,float*,int,int,int,int,float);
/* ★一块专家一次提交 GPU(2026-08-29)★ 抽格行(nt≈12)的小 GEMM 全落 CPU 是原始病; 批量化后
 * BFLT 又实锤 bdq(26GB fp32 物化)+瘦 GEMM 合计 120s/深单元 → 2026-08-30 融合版首选:
 * dequant 融进 kernel, 权重流量=索引 ~3MB/矩阵。层级: ①fused(免 bdq) ②bdq+cublas
 * ③CPU 20 线程逐专家 —— 每级失败大声打印, 不静默; 首 2 块 fused 与旧路全量对拍,
 * relerr>1e-4 永久回退(质量门优先)。
 * 返回 1=整块吃掉; 0=不适用(调用方 CPU 路); g_bmw_batched: 2=fused 1=物化可别名 0=无。 */
static int bmw_gpu_chunk(bmw_t *w, int e0, int e1)
{
    lfile_t *lf=w->lf; const int S=w->S, nE=e1-e0;
    extern float *g_bmw_buf; extern int g_bmw_e0;
    g_bmw_batched=0;
    if(nE<1||!lf->vqmap) return 0;                       /* 非 VQ 层: CPU 逐专家 */
    /* ★全量行(非抽格)保持原两级(2026-08-29 实测: 大 nt 时 dq_matmul 原路本就 GPU, 批量版
     * 补齐+单线程 gather 反而慢) — bdq 后 worker 直接别名物化缓冲。 */
    if(!g_anc_rowmap){ g_bmw_batched=bmw_batch_dequant(lf,e0,e1); return 0; }
    const double _g0=vqt_now();   /* BFLT 子账[18]: gather/补齐(单线程宿主) */
    /* ★gather 语义必须与原 worker 完全一致(2026-08-29 数值 bug 实案)★: 同 token 多槽命中
     * 本块不同专家=每个都算; 同行同专家查重跳过。计数与 gather 两循环必须同语义。 */
    int *nt=(int*)calloc((size_t)nE,sizeof(int)); if(!nt) return 0;
    for(int s=0;s<S;s++) for(int a=0;a<NACT_RT;a++){
        const int e=w->idx[(size_t)s*NACT_RT+a];
        if(e<e0||e>=e1) continue;
        int dup=0; for(int b=0;b<a;b++) if(w->idx[(size_t)s*NACT_RT+b]==e){ dup=1; break; }
        if(!dup) nt[e-e0]++; }
    int ntmax=0; for(int i=0;i<nE;i++) if(nt[i]>ntmax) ntmax=nt[i];
    if(ntmax<1){ free(nt); return 1; }
    const size_t npad=(size_t)nE*ntmax;
    if((double)npad*DIM*4.0*2.0 > 8.0e9){ free(nt); return 0; }   /* 补齐太胖: 让 CPU 路接 */
    /* ★钉页常驻暂存(2026-08-30 BFLT)★ 每 chunk calloc/free 10MB×2=页表毁建+pageable 慢拷。
     * cap 只增不缩; Xp 补齐行内容任意(垃圾行不被 scatter), Wt 补齐槽必须 0。失败大声回退。 */
    static float *Xp=NULL,*Yp=NULL,*Wt=NULL; static int *tk=NULL; static size_t _xc=0,_wc=0; static int _pin=-1;
    if(_xc<npad*(size_t)DIM){
        if(_pin==1){ cudaFreeHost(Xp); cudaFreeHost(Yp); } else { free(Xp); free(Yp); }
        Xp=Yp=NULL; _xc=0;
        if(cudaMallocHost((void**)&Xp,npad*(size_t)DIM*4)==cudaSuccess&&
           cudaMallocHost((void**)&Yp,npad*(size_t)DIM*4)==cudaSuccess) _pin=1;
        else { if(Xp){cudaFreeHost(Xp);Xp=NULL;} static int _w9=0; if(!_w9++) fprintf(stderr,"[moe-gpu] ★钉页分配失败 → 普通 malloc 暂存★\n");
               Xp=(float*)malloc(npad*(size_t)DIM*4); Yp=(float*)malloc(npad*(size_t)DIM*4); _pin=0; }
        if(Xp&&Yp) _xc=npad*(size_t)DIM;
    }
    if(_wc<npad){ if(_pin==1&&Wt){cudaFreeHost(Wt);} else free(Wt); free(tk); Wt=NULL; tk=NULL; _wc=0;
        if(_pin!=1||cudaMallocHost((void**)&Wt,npad*4)!=cudaSuccess) Wt=(float*)malloc(npad*4);
        tk=(int*)malloc(npad*sizeof(int)); if(Wt&&tk) _wc=npad; }
    if(!Xp||!Yp||!Wt||!tk){ free(nt); return 0; }
    memset(Wt,0,npad*4);
    int *fill=(int*)calloc((size_t)nE,sizeof(int));
    for(int s=0;s<S;s++) for(int a=0;a<NACT_RT;a++){
        const int e=w->idx[(size_t)s*NACT_RT+a];
        if(e<e0||e>=e1) continue;
        { int dup=0; for(int b=0;b<a;b++) if(w->idx[(size_t)s*NACT_RT+b]==e){ dup=1; break; }
          if(dup) continue; }
        const int j=e-e0, i=fill[j]++;
        const float gee=w->ge?w->ge[e]:1.0f;
        tk[(size_t)j*ntmax+i]=s;
        Wt[(size_t)j*ntmax+i]=gee*w->rw[(size_t)s*NACT_RT+a];
        memcpy(Xp+((size_t)j*ntmax+i)*DIM, w->Fin+(size_t)s*DIM, (size_t)DIM*4); }
    if(g_bflt_on) g_bflt[18]+=vqt_now()-_g0;
    /* --- ①VQ-fused: 载荷头逐专家校验, 任一不合(冷槽/d16≠4/形状不符)→整块回退, 不是近似 --- */
    static int fdead=0, fchk=2;
    int ok=0; const double gt0=vqt_now();
    if(!fdead&&nE<=64&&ntmax<=512){
        vqg_fj fj[64]; int jok=1;
        const uint64_t *vtab=(const uint64_t*)(lf->vqmap+16);
        for(int e=e0;e<e1&&jok;e++){
            uint64_t of[3]={vtab[(size_t)e*3],vtab[(size_t)e*3+1],vtab[(size_t)e*3+2]};
            int nc[3],nb[3];
            for(int m2=0;m2<3&&jok;m2++){
                if(!of[m2]){ jok=0; break; }             /* 冷槽混合层: 回退 */
                const uint8_t*pay=lf->vqmap+of[m2];
                uint16_t d16,n16; uint32_t rr,cc;
                memcpy(&d16,pay+4,2); memcpy(&n16,pay+6,2); memcpy(&rr,pay+8,4); memcpy(&cc,pay+12,4);
                const uint32_t er=m2<2?(uint32_t)MOEI:(uint32_t)DIM, ec2=m2<2?(uint32_t)DIM:(uint32_t)MOEI;
                if(d16!=4||n16>1024||rr!=er||cc!=ec2){ jok=0; break; }
                int b2=1; while((1<<b2)<n16) b2++; nc[m2]=n16; nb[m2]=b2;
            }
            if(jok&&(nc[0]!=nc[1]||nb[0]!=nb[1])) jok=0;   /* w1/w3 同码本规格(kernel 单参) */
            if(jok) fj[e-e0]=(vqg_fj){of[0],of[1],of[2],nc[0],nb[0],nc[2],nb[2],nt[e-e0]};
        }
        if(jok){
            ok=vqg_moe_batch_fused(lf->vqmap,fj,Xp,Wt,Yp,nE,ntmax,DIM,MOEI,SWLIM);
            if(ok&&fchk>0){   /* ★对拍自检: 与旧路(bdq+cublas)全量比, 只比有效槽★ */
                float *Y2=(float*)malloc(npad*(size_t)DIM*4);
                if(Y2&&bmw_batch_dequant(lf,e0,e1)&&vqg_moe_batch(g_bmw_buf,Xp,Wt,Y2,nE,ntmax,DIM,MOEI,SWLIM)){
                    double e2=0,r2=0;
                    for(int j=0;j<nE;j++) for(int i=0;i<nt[j];i++){
                        const float*a=Yp+((size_t)j*ntmax+i)*DIM,*b=Y2+((size_t)j*ntmax+i)*DIM;
                        for(int d=0;d<DIM;d++){ double dd=(double)a[d]-b[d]; e2+=dd*dd; r2+=(double)b[d]*b[d]; } }
                    double rel=sqrt(e2/(r2+1e-30));
                    fprintf(stderr,"[moe-fused] 对拍#%d e[%d,%d) relerr=%.3g %s\n",3-fchk,e0,e1,rel,
                            rel<1e-4?"✓":"★超限→永久回退旧路★");
                    if(rel>=1e-4){ fdead=1; memcpy(Yp,Y2,npad*(size_t)DIM*4); }
                    fchk--;
                } else { static int _w8=0; if(!_w8++) fprintf(stderr,"[moe-fused] 对拍旧路不可用, 跳过自检\n"); fchk=0; }
                free(Y2);
            }
            if(ok&&!fdead) g_bmw_batched=2;
        }
    }
    /* --- ②旧路: 物化 dequant + cublas 批量 --- */
    if(!ok||fdead){
        if(g_bmw_batched!=1) g_bmw_batched=bmw_batch_dequant(lf,e0,e1);
        if(!g_bmw_batched){ free(nt); free(fill); return 0; }   /* ③CPU worker 接 */
        if(!ok||fdead) ok=vqg_moe_batch(g_bmw_buf,Xp,Wt,Yp,nE,ntmax,DIM,MOEI,SWLIM);
    }
    { extern double g_bmw_t[2]; if(ok) g_bmw_t[1]+=vqt_now()-gt0; }
    { static int d2=0; if(d2++<4) fprintf(stderr,"[moe-gpu] e[%d,%d) S=%d ntmax=%d 补齐率=%.1fx %s %.2fs\n",
        e0,e1,S,ntmax,(double)nE*ntmax/((double)S*NACT_RT/NEXP*nE),
        ok?(g_bmw_batched==2?"GPU-fused":"GPU"):"★失败→CPU★",vqt_now()-gt0); }
    const double _s0=vqt_now();   /* BFLT 子账[19]: scatter(单线程宿主) */
    if(ok){   /* scatter: 冷热分桶与原路同判据(vtab 的 w2 槽非零=热) */
        const uint64_t *vtab=(const uint64_t*)(lf->vqmap+16);
        for(int j=0;j<nE;j++){
            const int e_hot=vtab[(size_t)(e0+j)*3+2]!=0;
            float *bucket=e_hot?w->partial:w->partial_c;
            for(int i=0;i<nt[j];i++){ float *dst=bucket+(size_t)tk[(size_t)j*ntmax+i]*DIM;
                const float *yi=Yp+((size_t)j*ntmax+i)*DIM;
                for(int d=0;d<DIM;d++) dst[d]+=yi[d]; } } }
    if(ok&&g_bflt_on) g_bflt[19]+=vqt_now()-_s0;
    free(nt);free(fill);   /* Xp/Yp/Wt/tk 常驻(见上) */
    return ok;
}
#endif
