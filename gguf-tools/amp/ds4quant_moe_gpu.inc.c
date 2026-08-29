/* ds4quant_moe_gpu.inc.c — GPU 批量专家前向的宿主侧(gather/提交/scatter)。
 * 物理分片, 只被 ds4quant_run_p4.inc.c include; 逻辑上是 bytes_moe 的 GPU 分支, 不是另一份实现。
 * 设备侧核在 gguf-tools/quantize/vq_gpu_moe.inc.cu。 */
#ifdef DS4QUANT_CUDA
extern int vqg_moe_batch(const float*,const float*,const float*,float*,int,int,int,int,float);
/* ★一块专家一次提交 GPU(2026-08-29 用户令"先把 sweep 改成 gpu")★
 * 原路: 20 线程各跑【单专家】3 次 GEMM。sweep 抽格到 ~512 行后每专家只有 nt≈12 个 token,
 * 单次 GEMM 正好卡在 dq_matmul 的 GPU 门槛(2e8 FLOP)上 ⇒ 全落 CPU, 20 线程满载 17.7s/前向。
 * dq_matmul 注释里已实测过"降门槛"是负收益(259s vs 181s/层, 小 GEMM 队列争用), 并指明正解
 * 是【批量结构改造】—— 就是这里: 专家权重批 dequant 后本就在 managed 的 g_bmw_buf 里,
 * 排布 [nE][3][DIM*MOEI] 等步长连续, 正好喂 cublasSgemmStridedBatched。
 * 补齐到 ntmax 有浪费(sweep 场景约 3×), 相对 GPU/CPU 的差距可忽略。
 * 返回 1=整块吃掉; 0=不适用, 调用方走原 CPU 逐专家路(那条路逐字节不变, 不是近似兜底)。 */
static int bmw_gpu_chunk(bmw_t *w, int e0, int e1)
{
    lfile_t *lf=w->lf; const int S=w->S, nE=e1-e0;
    extern float *g_bmw_buf; extern int g_bmw_e0;
    if(nE<1||!lf->vqmap||!g_bmw_buf||g_bmw_e0!=e0) return 0;
    int *nt=(int*)calloc((size_t)nE,sizeof(int)); if(!nt) return 0;
    for(int s=0;s<S;s++) for(int a=0;a<NACT_RT;a++){
        const int e=w->idx[(size_t)s*NACT_RT+a];
        if(e>=e0&&e<e1){ nt[e-e0]++; break; } }
    int ntmax=0; for(int i=0;i<nE;i++) if(nt[i]>ntmax) ntmax=nt[i];
    if(ntmax<1){ free(nt); return 1; }
    const size_t npad=(size_t)nE*ntmax;
    if((double)npad*DIM*4.0*2.0 > 8.0e9){ free(nt); return 0; }   /* 补齐太胖: 让 CPU 路接 */
    float *Xp=(float*)calloc(npad*DIM,4), *Yp=(float*)malloc(npad*(size_t)DIM*4);
    float *Wt=(float*)calloc(npad,4);  int *tk=(int*)malloc(npad*sizeof(int));
    if(!Xp||!Yp||!Wt||!tk){ free(nt);free(Xp);free(Yp);free(Wt);free(tk); return 0; }
    /* gather: 与原 worker 同序(按 s 升序, 每 token 取首个命中的槽) */
    int *fill=(int*)calloc((size_t)nE,sizeof(int));
    for(int s=0;s<S;s++) for(int a=0;a<NACT_RT;a++){
        const int e=w->idx[(size_t)s*NACT_RT+a];
        if(e<e0||e>=e1) continue;
        const int j=e-e0, i=fill[j]++;
        const float gee=w->ge?w->ge[e]:1.0f;
        tk[(size_t)j*ntmax+i]=s;
        Wt[(size_t)j*ntmax+i]=gee*w->rw[(size_t)s*NACT_RT+a];
        memcpy(Xp+((size_t)j*ntmax+i)*DIM, w->Fin+(size_t)s*DIM, (size_t)DIM*4);
        break; }
    const int ok=vqg_moe_batch(g_bmw_buf,Xp,Wt,Yp,nE,ntmax,DIM,MOEI,SWLIM);
    if(ok){   /* scatter: 冷热分桶与原路同判据(vtab 的 w2 槽非零=热) */
        const uint64_t *vtab=(const uint64_t*)(lf->vqmap+16);
        for(int j=0;j<nE;j++){
            const int e_hot=vtab[(size_t)(e0+j)*3+2]!=0;
            float *bucket=e_hot?w->partial:w->partial_c;
            for(int i=0;i<nt[j];i++){ float *dst=bucket+(size_t)tk[(size_t)j*ntmax+i]*DIM;
                const float *yi=Yp+((size_t)j*ntmax+i)*DIM;
                for(int d=0;d<DIM;d++) dst[d]+=yi[d]; } } }
    free(nt);free(fill);free(Xp);free(Yp);free(Wt);free(tk);
    return ok;
}
#endif

