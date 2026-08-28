#include <sys/mman.h>
/* ★跨层线程缓冲池(2026-08-28 速度)★ 原来 bytes_moe 每次调用给每个 worker 新 malloc:
 * q1/q3/q2 各 MOEI·DIM·4 = 33MB×3, xs/partial/partial_c/aq 各 S·DIM·4(S=8192 时 134MB)。
 * 20 线程合计约 10GB/层, 全是 mmap 新页 ⇒ 首触零页故障是纯开销。改成按 (nth,S) 缓存复用,
 * 只在形状变化时重分配。数值零影响: 唯一依赖初值的是 partial/partial_c 与 aq(都是 += 累加),
 * 已在 worker 内显式 memset。 */
/* 自造屏障(macOS 没有 pthread_barrier_t)。用途见下面 bytes_moe 的分块循环: 20 条 worker
 * 线程跨 4 个专家块复用, 每块前后各会合一次。 */
volatile int ws_e_end=0;   /* 见 ds4quant_run_p3 的声明 */
typedef struct { pthread_mutex_t m; pthread_cond_t c; int need, cnt, gen; } bmbar_t;
static void bmbar_init(bmbar_t*b,int need){ pthread_mutex_init(&b->m,NULL);
    pthread_cond_init(&b->c,NULL); b->need=need; b->cnt=0; b->gen=0; }
static void bmbar_wait(bmbar_t*b){ pthread_mutex_lock(&b->m);
    int g=b->gen;
    if(++b->cnt==b->need){ b->cnt=0; b->gen++; pthread_cond_broadcast(&b->c); }
    else while(g==b->gen) pthread_cond_wait(&b->c,&b->m);
    pthread_mutex_unlock(&b->m); }
static void bmbar_destroy(bmbar_t*b){ pthread_mutex_destroy(&b->m); pthread_cond_destroy(&b->c); }
/* ★常驻 worker 线程池(2026-08-28)★
 * 原设计是每次 bytes_moe 新建 20 条线程、干完就退。两条路都是死路:
 *   不收 __thread CUDA 资源 ⇒ 每条线程留下 cuBLAS 句柄/流 + dX/dW/dO 显存暂存(dW 就是
 *     DIM×MOEI=33.5MB), 43 层几千条线程全泄在驱动侧。实撞: /proc/meminfo 只认得
 *     57GB/121GB, 另外 65GB 在 GPU 驱动手里, MemAvailable 逐层单调降, 两跑都被看门狗停;
 *   收(调 dq_gpu_thread_release) ⇒ cublasDestroy+cudaFree 每条线程几百毫秒, 实测 bmoe1
 *     从 5.5s 涨到 20.5s。
 * 线程建一次、全进程复用就都没有了: 句柄跟着线程活到进程结束, 不泄也不用反复建销。
 * 主线程与 worker 用屏障对齐: 每个专家块前后各会合一次。 */
static pthread_t *BMW_TH=NULL; static int BMW_NTH=0;
static bmbar_t BMW_BAR; static volatile int BMW_STOP=0, BMW_ZERO=0;
static int BMW_ENEXT=0;   /* 抢专家的原子计数器(常驻池共享) */
static struct { lfile_t*lf; int S; const float*Fin; const int*idx; const float*rw; const float*ge; } BMW_JOB;
typedef struct { float *q1,*q3,*q2,*xs,*wwv,*partial,*partial_c,*aq; int *tok; } bmwbuf_t;
static bmwbuf_t *BMW_BUF=NULL; static int BMW_BUF_N=0, BMW_BUF_S=0;
static void bmw_pool(int nth,int S){
    if(BMW_BUF_N==nth&&BMW_BUF_S==S) return;
    for(int t=0;t<BMW_BUF_N;t++){ bmwbuf_t*b=&BMW_BUF[t];
        free(b->q1);free(b->q3);free(b->q2);free(b->xs);free(b->wwv);
        free(b->partial);free(b->partial_c);free(b->aq);free(b->tok); }
    free(BMW_BUF); BMW_BUF=calloc((size_t)nth,sizeof(bmwbuf_t)); BMW_BUF_N=nth; BMW_BUF_S=S;
    for(int t=0;t<nth;t++){ bmwbuf_t*b=&BMW_BUF[t];
        b->q1=malloc((size_t)MOEI*DIM*4); b->q3=malloc((size_t)MOEI*DIM*4); b->q2=malloc((size_t)DIM*MOEI*4);
        b->xs=malloc((size_t)S*DIM*4);      b->aq=malloc((size_t)S*DIM*4);
        b->partial=malloc((size_t)S*DIM*4); b->partial_c=malloc((size_t)S*DIM*4);
        b->wwv=malloc((size_t)S*4);         b->tok=malloc((size_t)S*sizeof(int)); }
}
/* 归约 worker: 元素区间 [i0,i1) 上按 t 升序求和(与原串行同序=逐位同值) */
typedef struct { size_t i0,i1; int nth; float *rh,*rc,*fout; } bmred_t;
static void *bmw_reduce_worker(void*a){
    bmred_t*r=a;
    for(size_t i=r->i0;i<r->i1;i++){
        float h=0,c=0;
        for(int t=0;t<r->nth;t++){ h+=BMW_BUF[t].partial[i]; c+=BMW_BUF[t].partial_c[i]; }
        r->rh[i]=h; r->rc[i]=c; r->fout[i]+=h+c;
    }
    return NULL;
}
static void *bytes_moe_worker(void*a){
    const int _ti=(int)(intptr_t)a;
  for(;;){
    bmbar_wait(&BMW_BAR);                 /* 等主线程把本块的专家反量化好 */
    if(BMW_STOP) break;
    bmw_t _w={BMW_JOB.lf,BMW_JOB.S,BMW_JOB.Fin,BMW_JOB.idx,BMW_JOB.rw,&BMW_ENEXT,
              BMW_BUF[_ti].partial,BMW_JOB.ge,BMW_BUF[_ti].partial_c,_ti,0,NULL,0};
    bmw_t*w=&_w; lfile_t*lf=w->lf; int S=w->S;
    /* ★缓冲取自跨层池(2026-08-28)★ 见 bmw_pool 注释: 原来这里每层每线程新 malloc 约 500MB,
     * 20 线程 = 每层 10GB 首触零页, 实测吃掉 bmoe1 的 4.6s。partial/partial_c 由本线程自己
     * memset(20 路并行, 替代主线程串行 calloc)。 */
    bmwbuf_t *B=&BMW_BUF[w->ti];
    float *q1=B->q1,*q3=B->q3,*q2=B->q2;
    int *tok=B->tok; float *xs=B->xs,*wwv=B->wwv;
    if(BMW_ZERO){ memset(w->partial,0,(size_t)S*DIM*4); memset(w->partial_c,0,(size_t)S*DIM*4); }
    float *w1p=NULL,*w3p=NULL,*w2p=NULL;   /* 本迭代实际权重指针(本地缓冲或批 dequant 切片) */
    for(;;){ int e=__sync_fetch_and_add(w->e_next,1); if(e>=ws_e_end)break;
        int nt=0; float gee=w->ge?w->ge[e]:1.0f;   /* bf.GE: per-expert 增益, 累加时乘 */
        for(int s2=0;s2<S;s2++)for(int a2=0;a2<NACT_RT;a2++)
            if(w->idx[(size_t)s2*NACT_RT+a2]==e){ tok[nt]=s2; wwv[nt]=gee*w->rw[(size_t)s2*NACT_RT+a2];
                memcpy(xs+(size_t)nt*DIM,w->Fin+(size_t)s2*DIM,(size_t)DIM*4); nt++; break; }
        if(!nt) continue;
        int e_hot=0;   /* 冷热判定(合并态口径), 分桶累计用 */
        if(lf->vqmap){   /* v2.2 VQ 回放: 表内 w2 槽非零=热(全三矩阵 VQ), 否则冷(w1/w3 VQ + w2 go1b) */
            const uint64_t *vtab=(const uint64_t*)(lf->vqmap+16);
            uint64_t o1=vtab[(size_t)e*3],o3=vtab[(size_t)e*3+1],o2=vtab[(size_t)e*3+2];
            e_hot=o2!=0;
            extern double g_bmw_t[2];
            w1p=q1; w3p=q3; w2p=q2;   /* 默认走本地 dequant 缓冲 */
#ifdef DS4QUANT_CUDA
            extern float *g_bmw_buf; extern int g_bmw_batched; extern int g_bmw_e0;
            if(g_bmw_batched){   /* 批 dequant 已就位: 指针别名切片, 不动 q1..q2 生命周期 */
                /* ★下标必须减去本块首专家号★(2026-08-28 分块后 SIGSEGV 实撞): 缓冲从
                 * 全部 256 专家(25.8GB)改成一块 64 个(6.4GB)后, 这里的绝对 e 一过 64
                 * 就指到缓冲外面去了。写 job 表那侧走的是 bmw_slot_off(已折算), 这侧是
                 * 手写公式 —— 两处必须同口径。 */
                const size_t sl0=(size_t)(e-g_bmw_e0)*3;
                w1p=g_bmw_buf+(sl0+0)*((size_t)DIM*MOEI);
                w3p=g_bmw_buf+(sl0+1)*((size_t)DIM*MOEI);
                w2p=g_bmw_buf+(sl0+2)*((size_t)DIM*MOEI);
            } else {
#endif
            double bt0=vqt_now();
            vq_unpack_dequant(lf->vqmap+o1,lf->vqmsz-o1,q1,NULL,NULL);
            vq_unpack_dequant(lf->vqmap+o3,lf->vqmsz-o3,q3,NULL,NULL);
            if(o2) vq_unpack_dequant(lf->vqmap+o2,lf->vqmsz-o2,q2,NULL,NULL);
            else   dq_go1b_bytes_dequant(lf->w2+(size_t)e*lf->szD,DIM,MOEI,q2);
            g_bmw_t[0]+=vqt_now()-bt0;
#ifdef DS4QUANT_CUDA
            }
#endif
        } else {
        int sl=(lf->g2k>0&&g2_replay_en())?lf->g2slot[e]:-1;
        e_hot=sl>=0;
        if(sl>=0){   /* 热专家: go2b 合并2bit 回放(反修/rr_verdict 在合并态前向上进行) */
            size_t szG2=(size_t)MOEI*go2b_row_bytes(DIM), szD2=(size_t)DIM*go2b_row_bytes(MOEI);
            dq_go2b_bytes_dequant(lf->g2w1+(size_t)sl*szG2,MOEI,DIM,q1);
            dq_go2b_bytes_dequant(lf->g2w3+(size_t)sl*szG2,MOEI,DIM,q3);
            dq_go2b_bytes_dequant(lf->g2w2+(size_t)sl*szD2,DIM,MOEI,q2);
        } else {
            dq_go1b_bytes_dequant(lf->w1+(size_t)e*lf->szG,MOEI,DIM,q1);
            dq_go1b_bytes_dequant(lf->w3+(size_t)e*lf->szG,MOEI,DIM,q3);
            dq_go1b_bytes_dequant(lf->w2+(size_t)e*lf->szD,DIM,MOEI,q2);
        } }
        float *aq=B->aq; memset(aq,0,(size_t)nt*DIM*4);   /* 池化: dq_expert_fp 是 += 累加, 必须先清零 */
        { extern double g_bmw_t[2]; double bt1=vqt_now();
          dq_expert_fp(xs,w1p?w1p:q1,w3p?w3p:q3,w2p?w2p:q2,wwv,aq,nt,DIM,MOEI,SWLIM);
          g_bmw_t[1]+=vqt_now()-bt1; }
        for(int i=0;i<nt;i++){ float*dst=(e_hot?w->partial:w->partial_c)+(size_t)tok[i]*DIM;
            const float*yi=aq+(size_t)i*DIM;
            for(int d2=0;d2<DIM;d2++) dst[d2]+=yi[d2]; }
        /* ★zl.ERF 死层补丁回放(2026-08-12)★: c=w·(h@Vᵀ)@Uᵀ(U折S/V折α), token门 |c|²≥τ。
         * h 与 dq_expert_fp 完全同式(clip+silu); 只在带 type8 记录且本专家在册的层花算力。
         * DS4_TYPE8_OFF=1 消融开关(同态 A/B 归因用)。 */
        if(getenv("DS4_TYPE8_OFF")) goto erf_skip;
        for(int oi2=0;oi2<lf->nops;oi2++){
            lop_t*op=&lf->ops[oi2];
            if(op->type!=8) continue;
            int ei=-1;
            for(int i2=0;i2<op->erf_ne;i2++) if(op->erf_eid[i2]==(uint32_t)e){ei=i2;break;}
            if(ei<0) break;
            int r2=op->erf_r;
            size_t blkf=(size_t)DIM*r2+(size_t)r2*MOEI;
            const float*Ue=op->erf_UV+(size_t)ei*blkf;      /* [DIM,r] */
            const float*Ve=Ue+(size_t)DIM*r2;               /* [r,MOEI] */
            float tau=op->erf_tau[ei];
            float *hh=malloc((size_t)nt*MOEI*4),*uu2=malloc((size_t)nt*MOEI*4);
            dq_matmul(xs,w1p?w1p:q1,hh,nt,DIM,MOEI); dq_matmul(xs,w3p?w3p:q3,uu2,nt,DIM,MOEI);
            for(size_t ii=0;ii<(size_t)nt*MOEI;ii++){ float gg=hh[ii],vv=uu2[ii];
                if(SWLIM>0){ if(vv>SWLIM)vv=SWLIM; if(vv<-SWLIM)vv=-SWLIM; if(gg>SWLIM)gg=SWLIM; }
                hh[ii]=dq_silu(gg)*vv; }
            for(int i2=0;i2<nt;i2++)for(int j=0;j<MOEI;j++) hh[(size_t)i2*MOEI+j]*=wwv[i2];
            float *gvb=malloc((size_t)nt*r2*4),*cb=malloc((size_t)nt*DIM*4);
            dq_matmul(hh,Ve,gvb,nt,MOEI,r2);
            dq_matmul(gvb,Ue,cb,nt,r2,DIM);
            for(int i2=0;i2<nt;i2++){
                const float*ci=cb+(size_t)i2*DIM;
                float en2=0; for(int d2=0;d2<DIM;d2++) en2+=ci[d2]*ci[d2];
                if(en2<tau) continue;
                float*dst2=(e_hot?w->partial:w->partial_c)+(size_t)tok[i2]*DIM;
                for(int d2=0;d2<DIM;d2++) dst2[d2]+=ci[d2];
            }
            free(hh);free(uu2);free(gvb);free(cb);
            break;
        }
        erf_skip: ;
    }
    bmbar_wait(&BMW_BAR);                 /* 告诉主线程本块做完 */
  }
    /* ★必须收 __thread CUDA 资源(2026-08-28 实撞 OOM)★ dq_matmul 给每条线程留了 cuBLAS
     * 句柄/流 + dX/dW/dO 三块显存暂存(dW 就是 DIM×MOEI=33.5MB), 线程退出不自动释放。
     * p1 里早就写好了 dq_gpu_thread_release(注释还记着"32k 锚实锤 ~2-3GB/层泄漏"), 但
     * 只有量化遍的 worker(p2:156)调了它 —— 反修全程走的是本函数, 一次没调过。
     * 实撞: /proc/meminfo 只认得 57GB/121GB, 另外 65GB 在 GPU 驱动手里(nvidia-smi 该进程
     * 56.7GB, 而本路显式分配只有 ~9GB), MemAvailable 逐层单调降, 两跑都被看门狗停在半路。 */
    dq_gpu_thread_release();
    return NULL;   /* 缓冲属池, 不 free */
}
static lfile_t *GS_LF=NULL;    /* 回扫: 全43层文件 mmap 只开一次(禁重复 mmap+解析) */
/* ★真·反修前层★ 状态: 目标层 z 系数以【最终输出 KL】重解时用 */
static int BF_ANCROUTE=0;   /* 反修前向强制锚路由(禁稀疏路由翻转); 备用, 默认关 */
/* ★路由偏置侧车(2026-07-28 部署侧路由修正)★: 神谕探针实证反修字节全部亏损住在路由漂移
 * (agree 77.6→82.9=top1f天花板)。FIT: 学生路由回放时按层累计 FP锚集合 vs 学生集合的选择分
 * margin 缺口(漏选+=thr−v, 多选−=v−thr; thr=学生第6名选择分) → 均值+计数落盘。
 * APPLY: Δb 只加进 dq_gate_route_topk 的选择分(gbias 侧), 权重分不动(与引擎语义同构)。
 * 铁律: fit 只跑校准语料(判决锚不混入); 锚路由(神谕)下 fit 无意义, 自动跳过。 */
static float *RB_ACC=NULL; static uint32_t *RB_CNT=NULL;  /* fit 累计 [NLAYERS][NEXP] */
static float *RB_APPLY=NULL; static int RB_TRIED=0;       /* apply Δb_raw [NLAYERS][NEXP] */
static float RB_ALPHA=1.0f; static int RB_MINCNT=8;
/* ★序贯路由(2026-07-29 用户终令"添加路由和动态α扫"): 贪心内每层锁定后 FIT 本层 Δb →
 * α 三点扫选层优(负收益自动关) → 定稿/链推进带路由修正, 下游继承; 收官 Δb+α 落盘。 */
static int RB_SEQ=0;                 /* DS4_ROUTE_SEQ=1 */
static float RB_ALPHA_L[64]={0};     /* per-layer 动态 α(0=该层关) */
/* ★热启动记录(2026-07-30 用户"每层太慢"): 上层锁定档族/热数, 深带同带惯性 —
 * 上层 'g' 族 ⇒ 本层跳 g10 裸基线+rank 二分(g10 曾被每层裸评两次), 直接从上层档-1 起试 */
static char g_mv_pcfg=0; static int g_mv_phot=0;
/* ★体积账(2026-07-30 用户"别本末倒置"): 累计 bpw vs 冠军口径(g10h16=1.1367/层),
 * h32+ 频繁出现=门线/语料错位告警, 超支即刻可见 */
static double g_mv_accv=0, g_mv_accc=0;
#define MV_CHAMP_BPW ((240.0*1.0625+16.0*2.25)/256.0)
static double mv_name_bpw(const char*nm){
    if(nm&&nm[0]=='v'){   /* vq 名字族(2026-08-09 记账修): "v4x1024[ h0]" → 索引位/dim + gr + 码本 */
        int dim=0,nc=0;
        if(sscanf(nm,"v%dx%d",&dim,&nc)==2&&dim>0&&nc>1){
            int b=0; while((1<<b)<nc) b++;
            return (double)b/dim + 16.0/4096.0 + ((double)nc*dim*16.0)/(2048.0*4096.0);
        }
    }
    if(!strncmp(nm,"g10h",4)){ int n=atoi(nm+4); if(n<=0)n=16; return ((256.0-n)*1.0625+n*2.25)/256.0; }
    return 1.0625;   /* g10/未知 */
}
static void mv_vol_note(int L,const char*nm,double vol){
    g_mv_accv+=vol; g_mv_accc+=MV_CHAMP_BPW;
    fprintf(stderr,"L%02d [体积账] 锁 %s(%.4f bpw) 累计 %.2f vs 冠军口径 %.2f (%+.1f%%)\n",
            L,nm,vol,g_mv_accv,g_mv_accc,(g_mv_accv/g_mv_accc-1.0)*100.0);
}
static void mv_prev_note(char cfg,const char*nm){
    g_mv_pcfg=cfg;
    const char*h=nm?strrchr(nm,'h'):NULL;
    g_mv_phot = h? atoi(h+1) : 0;
    if(cfg=='g'&&h&&g_mv_phot==0) g_mv_phot=16;   /* 旧名 "g10h"(基线) = 热16 */
}
static int g_rb_fit_L=-1;            /* 序贯: FIT 只统计该层(-1=全层旧行为) */
static void rb_commit(int L){        /* 本层 FIT 累计折算进 RB_APPLY(内存直通, mincnt 门) */
    if(!RB_ACC) return;
    if(!RB_APPLY) RB_APPLY=calloc((size_t)NL*NEXP,4);
    long armed=0;
    for(int e=0;e<NEXP;e++){ size_t i=(size_t)L*NEXP+e;
        RB_APPLY[i]=(RB_CNT[i]>=(uint32_t)RB_MINCNT)?RB_ACC[i]/(float)RB_CNT[i]:0.0f;
        if(RB_APPLY[i]!=0.0f) armed++; }
    fprintf(stderr,"L%02d [路由] Δb 就位(武装槽=%ld)\n",L,armed);
}
static void rb_save(void){           /* 收官/探针早退: Δb+cnt(0x41494252 同格式)+α 表落盘 */
    /* ★固定规则模式补路由 FIT(2026-07-31 实锤: RB_SEQ 两处入口都带 !mv_base, R36 战役
     * (mv_base 固定档)从未跑过路由 FIT ⇒ 裸路由 ⇒ 自回归活路由漂移无补偿 → 退化循环。
     * DS4_ROUTE_BIAS_FIT 走非序贯统计(在既有量化前向里顺带累计, 零额外前向), 落盘同格式,
     * α 由烘焙侧给(冠军同款 2.5)。teacher-forced 指标不受影响, 修的是自由生成稳定性。 */
    { long nz=0; if(RB_CNT) for(size_t i=0;i<(size_t)NL*NEXP;i++) if(RB_CNT[i]) nz++;
      fprintf(stderr,"[路由][diag] rb_save入口 ACC=%p APPLY=%p 非零CNT槽=%ld\n",(void*)RB_ACC,(void*)RB_APPLY,nz); }
    if(RB_ACC&&!RB_APPLY){   /* ★评测遍FIT(2026-08-13): BF_ONLY 无逐层 rb_commit, 收官一次性折算 */
        for(int l=0;l<NL;l++) rb_commit(l); }
    if(!(RB_SEQ||getenv("DS4_ROUTE_BIAS_FIT"))||!RB_APPLY){
        if(RB_SEQ||getenv("DS4_ROUTE_BIAS_FIT"))
            fprintf(stderr,"[路由] Δb 无统计可落盘(哈希路由=选择零漂移, RB 不适用)\n");
        return; }
    const char*rp=getenv("DS4_ROUTE_BIAS_OUT"); if(!rp) return;
    FILE*f=fopen(rp,"wb");
    if(f){ uint32_t hd[4]={0x41494252u,(uint32_t)NL,(uint32_t)NEXP,0};
        fwrite(hd,4,4,f); fwrite(RB_APPLY,4,(size_t)NL*NEXP,f);
        if(RB_CNT) fwrite(RB_CNT,4,(size_t)NL*NEXP,f); fclose(f); }
    char ap[512]; snprintf(ap,sizeof(ap),"%s.alpha.txt",rp);
    FILE*g=fopen(ap,"w");
    if(g){ for(int l=0;l<NL;l++) fprintf(g,"L=%d a=%.1f\n",l,RB_ALPHA_L[l]); fclose(g); }
    fprintf(stderr,"[路由] Δb+α 落盘 → %s(+.alpha.txt)\n",rp);
}
static float *GS_GV=NULL; static int GS_GV_L=-1;   /* 目标层 per-token routed 增益向量(搜每 token 最优乘子) */
static float *GS_FIN=NULL; static int GS_CAP_L=-1; /* 目标层 MoE 输入 Fin_L 捕获(重算 z 特征/当前系数) */
static int *GS_IDXC=NULL; static float *GS_RWC=NULL; /* 目标层路由捕获(GE 投影: token→专家 命中+权重) */
static float *HQE=NULL;    /* ★逐层反修★: 每层量化态入口隐藏 [NLAYERS+1][lstride](反修前层时从此重前向) */
static size_t g_hqe_lstride=0;   /* HQE mmap 尺寸记账(munmap 用; 文件后备映射禁 free) */
static int BACKFIT_INCR=1; /* DS4_BACKFIT_INCR: 逐层前进即反修(1=开, 默认); 0=只末尾全局回扫。
                              裁决2026-07-12: 非fast=每前沿全量反修所有前层(取消"末层最后一次全量"特例, 原DS4_BF_SWEEP已删);
                              fast=不向前修复(反修整个跳过, dql/opt 层文件照常落盘) */
static int *BF_FINOP=NULL;  /* 逐层反修: 每层"最终缩放α op"(bf.GL) 在 ops[] 的下标(-1=未加) */
static int *BF_DYN2OP=NULL; /* 逐层反修: 每层"per-token 动态 op"(bf.GLdyn2) 下标(-1=未加); 更新原地不重复追加 */
static int *BF_GEOP=NULL;   /* 逐层反修: 每层"per-expert 增益 op"(bf.GE, type-5) 下标(-1=未加) */
static int *BF_HCOP=NULL;   /* 冷热双通道: 每层"GLhc op"(bf.GLhc, type-7) 下标(-1=未加)(2026-08-06) */
/* 追加一条落地记录(vd=1, 链末回放; bf.GL/bf.GLdyn2 等), 返回其载荷文件偏移(原地更新用) */
static size_t append_rec(const char*path,const char*nm0,const char*al0,const void*pay,uint64_t paysz,float m1){
    FILE*f=fopen(path,"r+b"); if(!f) return 0;
    uint32_t nrec; fseek(f,8,SEEK_SET);
    if(fread(&nrec,4,1,f)!=1){ fclose(f); return 0; }
    fseek(f,0,SEEK_END); long eof=ftell(f); if(eof<0){ fclose(f); return 0; }
    char nm[16]={0},al[64]={0}; snprintf(nm,16,"%s",nm0); snprintf(al,64,"%s",al0);
    uint64_t vol=paysz; float m[4]={m1,0,0,m1}; int vd=1;
    fwrite(nm,1,16,f); fwrite(al,1,64,f); fwrite(&vol,8,1,f); fwrite(&paysz,8,1,f);
    fwrite(m,4,4,f); fwrite(&vd,4,1,f); fwrite(pay,1,(size_t)paysz,f);
    fseek(f,8,SEEK_SET); nrec++; fwrite(&nrec,4,1,f);
    fclose(f);
    return (size_t)eof + 116;   /* 载荷偏移 = 记录起点 + (name16+algo64+vol8+psz8+mean16+vd4) */
}
double g_bmw_t[2]={0,0};
/* 分块反量化的两个量必须在 #ifdef 外: CPU 参考路(-DDS4_NO_GPU / 无 CUDA)同样按块跑循环, 只是
 * 块内走逐矩阵 dequant。理由见下面 bmw_batch_dequant 的注释。 */
#define BMW_CHUNK 64
int g_bmw_e0=0;          /* 本块首个专家号(把绝对 e 折成块内下标) */
#ifdef DS4QUANT_CUDA
/* 第三刀(08-18): 整层 768 矩阵一次批 dequant(vq_gpu.cu), 消 99k 次 per-矩阵 sync。
 * 26GB fp32 缓冲静态复用(裸判期内存空闲); worker 前向直接吃切片指针零拷贝。 */
typedef struct { uint64_t pay_off, dst_off; int rows, cols, nc, nbit; } vqg_deq_job;
extern int vqg_dequant_batch(const uint8_t*, float*, const vqg_deq_job*, int, int, int);
extern int vqg_ready(void);
/* ★分块 dequant(2026-08-28 内存回归修)★
 * 原来一次把 256 个专家 ×3 个矩阵全反量化进一块 fp32 缓冲 = 256×3×4096×2048×4 = 25.8GB,
 * 且 08-28 把它 cudaMemAdvise 钉在 GPU 常驻(那是 dequant 6.8× 提速的来源, 必须保留)。
 * 后果: 常驻 25.8GB + 锚 23.1GB + HQE 23.6GB + 线程池 5.4GB ⇒ 121GB 机器的 MemAvailable
 * 在 12 分钟里从 12GB 单调掉到 3GB, 眼看要 OOM(实撞, 已停车)。
 * 专家前向本来就是 20 个线程从原子计数器抢活, 任一时刻在飞的只有 20 个 —— 256 个同时
 * 在场纯属浪费。改成一批 64: 缓冲 6.4GB, 省 19.4GB。代价是每层多 3 次 join 屏障(µs 级),
 * 块内仍是 20 路动态调度。★数值零影响★: 同样的专家同样的权重, 只是分批喂。 */
float *g_bmw_buf=NULL;   /* [BMW_CHUNK][3][8.4M] 切片: (e-g_bmw_e0)*3+w */
int g_bmw_batched=0;     /* 本块批 dequant 成功旗标 */
static size_t bmw_slot_off(int e,int w){ return ((size_t)(e-g_bmw_e0)*3+w)*( (size_t)DIM*MOEI ); }
static int bmw_batch_dequant(lfile_t*lf,int e0,int e1){
    if(!lf->vqmap||!vqg_ready()) return 0;
    const uint64_t *vtab=(const uint64_t*)(lf->vqmap+16);
    if(!g_bmw_buf){   /* managed: GPU 写零页故障(malloc 26GB 首触=百万级 HMM fault, 实测比逐矩阵还慢) */
        extern int vqg_alloc_managed(void**,size_t);
        if(!vqg_alloc_managed((void**)&g_bmw_buf,(size_t)BMW_CHUNK*3*DIM*MOEI*4)) return 0; }
    g_bmw_e0=e0;
    static vqg_deq_job jobs[BMW_CHUNK*3]; int nj=0, nc_max=0;
    /* ★批量预读(2026-08-28)★ 下面这个 768 次的表构建循环要在 855MB 的 mmap 层件里随机点
     * 768 个矩阵头(每个 16 字节, 分散在整个文件里)。实测这一段 3.1s, 而真正的 GPU dequant
     * 内核只有 0.90s —— 全是逐 4KB 按需缺页的代价。MADV_WILLNEED 让内核一次性顺序预读整段,
     * 后面的 kernel 也要读同一片payload, 一并受益。异步返回, 不阻塞。 */
    madvise((void*)lf->vqmap, lf->vqmsz, MADV_WILLNEED);
    for(int e=e0;e<e1;e++) for(int w=0;w<3;w++){
        const uint64_t off=vtab[(size_t)e*3+w];
        if(!off) return 0;                       /* 冷槽混合层: 退回逐矩阵路径 */
        const uint8_t *pay=lf->vqmap+off;
        uint16_t d16,n16; uint32_t rr,cc;
        memcpy(&d16,pay+4,2); memcpy(&n16,pay+6,2); memcpy(&rr,pay+8,4); memcpy(&cc,pay+12,4);
        if(d16!=4||n16>1024) return 0;
        int nb=1; while((1<<nb)<n16) nb++;
        jobs[nj].pay_off=off; jobs[nj].dst_off=bmw_slot_off(e,w);
        jobs[nj].rows=(int)rr; jobs[nj].cols=(int)cc; jobs[nj].nc=(int)n16; jobs[nj].nbit=nb;
        if((int)n16>nc_max) nc_max=n16;
        nj++;
    }
    double bt0=vqt_now();
    int ok=vqg_dequant_batch(lf->vqmap,g_bmw_buf,jobs,nj,4,nc_max);
    g_bmw_t[0]+=vqt_now()-bt0;
    return ok;
}
#endif
static void bytes_moe(lfile_t*lf,int S,const float*Fin,const int*idx,const float*rw,float*Fout){
    /* ★逐层口径(2026-08-28 改)★ 原为全程累计, 我曾把它当墙钟误读一次(1731s 实为 20 线程
     * 累计 ÷20 = 87s)。改为每次进本函数清零 + 每层打印, 并显式标注"20线程累计/墙钟"两栏。 */
    { extern double g_bmw_t[2]; g_bmw_t[0]=0; g_bmw_t[1]=0; }
    float *Fbase=malloc((size_t)S*DIM*4); memcpy(Fbase,Fout,(size_t)S*DIM*4);   /* shared 基 */
    const float *ge=NULL;   /* bf.GE(type-5, 取最后一条): per-expert 增益, 专家累加时乘(链 op 之前) */
    const int lay_skip=replay_layer_skipped();
    if(!lay_skip) for(int i=lf->nops-1;i>=0;i--) if(lf->ops[i].type==5&&lf->ops[i].ge){ ge=lf->ops[i].ge; break; }
    int nth=NTHREADS<1?1:(NTHREADS>NEXP?NEXP:NTHREADS);
    bmw_pool(nth,S);
    /* 冷热分桶缓存重建(hot=partial / cold=partial_c 归约) */
    if(BM_S!=S){ free(BM_RH); free(BM_RC); BM_RH=malloc((size_t)S*DIM*4); BM_RC=malloc((size_t)S*DIM*4); BM_S=S; }
    /* ★线程只建一次, 跨 4 个块复用(2026-08-28)★ 每条线程退出时要 dq_gpu_thread_release
     * (见 worker 尾注释), 那是 cublasDestroy + 三次 cudaFree, 几十毫秒起。若每块都重建
     * 20 条线程, 分块本身就把这笔开销翻 4 倍。用屏障把主线程的"准备下一块"和 worker 的
     * "干完本块"串起来: 建 20 条, 每块前后各会合一次。 */
    if(BMW_NTH!=nth){   /* 首次(或线程数变了): 起常驻池 */
        if(BMW_NTH){ BMW_STOP=1; bmbar_wait(&BMW_BAR);
            for(int t=0;t<BMW_NTH;t++) pthread_join(BMW_TH[t],NULL);
            bmbar_destroy(&BMW_BAR); free(BMW_TH); BMW_STOP=0; }
        bmbar_init(&BMW_BAR,nth+1); BMW_TH=malloc((size_t)nth*sizeof(pthread_t));
        for(int t=0;t<nth;t++) pthread_create(&BMW_TH[t],NULL,bytes_moe_worker,(void*)(intptr_t)t);
        BMW_NTH=nth; }
    BMW_JOB.lf=lf; BMW_JOB.S=S; BMW_JOB.Fin=Fin; BMW_JOB.idx=idx; BMW_JOB.rw=rw; BMW_JOB.ge=ge;
    for(int ec=0;ec<NEXP;ec+=BMW_CHUNK){
        const int ec1=(ec+BMW_CHUNK<NEXP)?ec+BMW_CHUNK:NEXP;
#ifdef DS4QUANT_CUDA
        g_bmw_batched=bmw_batch_dequant(lf,ec,ec1);
#else
        g_bmw_e0=ec;
#endif
        BMW_ENEXT=ec; ws_e_end=ec1; BMW_ZERO=(ec==0);
        bmbar_wait(&BMW_BAR);   /* 放行本块 */
        bmbar_wait(&BMW_BAR);   /* 等本块做完 */
    }
    /* ★归约并行化(2026-08-28)★ 原为主线程串行 20×2×S·DIM 次加(S=8192 时 13.4 亿次 +
     * 5.4GB 读)。按元素区间切给线程, 每元素内仍按 t 升序累加 ⇒ 浮点求和顺序不变, 逐位同值。 */
    { bmred_t *rs=malloc((size_t)nth*sizeof(bmred_t)); pthread_t *th=malloc((size_t)nth*sizeof(pthread_t));
      size_t N=(size_t)S*DIM, chunk=(N+nth-1)/nth;
      for(int t=0;t<nth;t++){ size_t a=chunk*(size_t)t, b=a+chunk>N?N:a+chunk; if(a>N)a=N;
          rs[t]=(bmred_t){a,b,nth,BM_RH,BM_RC,Fout};
          pthread_create(&th[t],NULL,bmw_reduce_worker,&rs[t]); }
      for(int t=0;t<nth;t++) pthread_join(th[t],NULL);
      free(rs); free(th); }
    /* ★Fcur=专家后·ops前的态(shared+base_routed)=coadapt 的 base态 Fcur★: TREF(type4)按 coadapt 口径
     * 从 Fcur 插值(Fout=Fcur+t·(Fout−Fcur)), 而非从 shared 缩放(否则 t 把 routed 整体放大, 抵消临界点处 Fout 翻倍误差)。*/
    /* (worker 内 bf.GE 已乘; Fcur 捕获在专家累加+GE 之后 = 调优的 base态口径) */
    float *Fcur=malloc((size_t)S*DIM*4); memcpy(Fcur,Fout,(size_t)S*DIM*4);
    /* 修正链回放(时间序): 缩放型(GL/dyn)以 Fbase=shared 为基; TREF 以 Fcur=base态 为基(与调优一致) */
    float *xn=NULL,*pj=NULL; const float*pjV8=NULL;
    /* ★op 族消融门(2026-08-04 诊断)★ DS4_REPLAY_SKIP_TYPES="4,6": 回放时跳过指定 type 的
     * 修正 op — 定位"过程态1.853 vs 回放2.001"分叉的元凶族。生产不设=全应用(原行为)。 */
    static int skip_t[8]={0}, skip_init=0;
    if(!skip_init){ skip_init=1; const char*sv=getenv("DS4_REPLAY_SKIP_TYPES");
        if(sv&&*sv){ char b2[64]; snprintf(b2,64,"%s",sv);
            for(char*tk=strtok(b2,",");tk;tk=strtok(NULL,",")){ int t2=atoi(tk);
                if(t2>=1&&t2<=7) skip_t[t2]=1; }
            fprintf(stderr,"[replay] 消融: 跳过 op type {%s}\n",sv); } }
    /* ★冷热基座先行(2026-08-06)★: type7 无论侧车序恒为链首 — 先应用基座分桶, 其余缩放
     * op 在其结果上链式作用(尾置=毁链: 覆盖式会丢掉已调 op 效果, 10 连全负实锤)。 */
    for(int oi=0;oi<lf->nops&&!lay_skip;oi++){ lop_t*o=&lf->ops[oi];
        if(o->type!=7||skip_t[7]) continue;
        for(int s2=0;s2<S;s2++){ float*fw=Fout+(size_t)s2*DIM; const float*fb=Fbase+(size_t)s2*DIM;
            const float*rh=BM_RH+(size_t)s2*DIM,*rc=BM_RC+(size_t)s2*DIM;
            for(int d2=0;d2<DIM;d2++) fw[d2]=fb[d2]+o->ghc[0]*rh[d2]+o->ghc[1]*rc[d2]; }
        memcpy(Fcur,Fout,(size_t)S*DIM*4);   /* TREF 基准=基座后的 base态 */
    }
    for(int oi=0;oi<lf->nops&&!lay_skip;oi++){ lop_t*o=&lf->ops[oi];
        if(o->type==7) continue;
        if(o->type>=1&&o->type<=6&&skip_t[o->type]) continue;
        if(o->type==1){ for(int s2=0;s2<S;s2++){ float*fw=Fout+(size_t)s2*DIM; const float*fb=Fbase+(size_t)s2*DIM;
                for(int d2=0;d2<DIM;d2++) fw[d2]=fb[d2]+o->g*(fw[d2]-fb[d2]); } }
        else if(o->type==2){
            if(!xn){ xn=malloc((size_t)S*4);
                for(int s2=0;s2<S;s2++){ const float*x=Fin+(size_t)s2*DIM; double v=0;
                    for(int d2=0;d2<DIM;d2++) v+=(double)x[d2]*x[d2]; xn[s2]=(float)sqrt(v); } }
            for(int s2=0;s2<S;s2++){ double c=o->w2p[0]+o->w2p[1]*((double)xn[s2]-o->w2p[2])/o->w2p[3];
                if(c<0.25)c=0.25; if(c>4.0)c=4.0;
                float*fw=Fout+(size_t)s2*DIM; const float*fb=Fbase+(size_t)s2*DIM;
                for(int d2=0;d2<DIM;d2++) fw[d2]=fb[d2]+(float)c*(fw[d2]-fb[d2]); } }
        else if(o->type==3&&o->V8){
            /* ★pj 按本条 op 的 V8 投影(2026-08-04 错配实锤修)★: 旧版跨 op 复用首条投影 ⇒
             * 反修追加的 bf.V8F(自带新方向)拿旧方向配自己的系数 → 部署回放 KL 2.00 vs 过程态 1.85。
             * V8 指针变化即重算 — 单条 op 层零额外成本。 */
            if(!pj||pjV8!=o->V8){ if(!pj) pj=malloc((size_t)S*8*4);
                pjV8=o->V8;
                for(int s2=0;s2<S;s2++){ const float*x=Fin+(size_t)s2*DIM; float*pr=pj+(size_t)s2*8;
                    for(int c=0;c<8;c++){ double a2=0; const float*vc=o->V8+(size_t)c*DIM;
                        for(int d2=0;d2<DIM;d2++) a2+=(double)x[d2]*vc[d2]; pr[c]=(float)a2; } } }
            for(int s2=0;s2<S;s2++){ const float*pr=pj+(size_t)s2*8;
                double c=o->w8[0]; for(int k=0;k<8;k++) c+=o->w8[1+k]*pr[k];
                if(c<0.25)c=0.25; if(c>4.0)c=4.0;
                float*fw=Fout+(size_t)s2*DIM; const float*fb=Fbase+(size_t)s2*DIM;
                for(int d2=0;d2<DIM;d2++) fw[d2]=fb[d2]+(float)c*(fw[d2]-fb[d2]); } }
        else if(o->type==4){ for(size_t i=0;i<(size_t)S*DIM;i++) Fout[i]=Fcur[i]+o->t*(Fout[i]-Fcur[i]); }   /* TREF: 从 base态 Fcur 插值(coadapt 口径) */
        else if(o->type==6&&o->zlk>0){
            double zdiag_nd=0,zdiag_nf=0,zdiag_sc=0; long long zdiag_n=0,zdiag_clip=0;
            /* ★冻结 z^L 回放(2026-07-14, 产物③)★: Fout += clip·U diag(z) Vᵀ Fin,
             * clip=min(1, tr·‖Fout‖/‖zd‖) 逐 token(与活体/引擎同口径的信赖域) */
            int zk=o->zlk;
            int zdin=o->zdin>0?o->zdin:DIM;   /* ★md86: 3*DIM=ftA */
            float *zd=malloc((size_t)DIM*4); double *pv=malloc((size_t)zk*8);
            float *phi=(zdin==3*DIM)?malloc((size_t)zdin*4):NULL;
            for(int s2=0;s2<S;s2++){
                const float*x=Fin+(size_t)s2*DIM; float*fw=Fout+(size_t)s2*DIM;
                const float*fb=Fbase+(size_t)s2*DIM;   /* 信赖域基准=routed(=Fout−shared基): 与引擎 λ 点同口径 */
                const float*xin=x;
                if(phi){   /* φ=[x, x⊙x/rms, relu(x)], rms=sqrt(mean(x²))+1e-6 — 与 zlayer zl_phi 逐式一致 */
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
                zdiag_nd+=nd; zdiag_nf+=nf; zdiag_sc+=sc2; if(sc2<1.0f) zdiag_clip++; zdiag_n++;
            }
            /* ★z 回放插桩(2026-08-26 用户令"打日志找"): 修正/基 幅度比 + 夹持触发率
             * —— 解算侧同口径打印(zloss_solve eval_apply), 两边对不上即回放路 bug。 */
            if(zdiag_n) fprintf(stderr,"[zdiag]L%02d k=%d tr=%.2f 行=%lld |z|/|routed|=%.4f "
                "夹持率=%.1f%% 平均缩放=%.3f\n", g_replay_cur_L, zk, (double)o->zltr, zdiag_n,
                zdiag_nf>0?zdiag_nd/zdiag_nf:0.0, 100.0*zdiag_clip/zdiag_n, zdiag_sc/zdiag_n);
            free(zd); free(pv); if(phi)free(phi);
        }
    }
    if(xn)free(xn); if(pj)free(pj);
    free(Fbase); free(Fcur);
    { int nth2=NTHREADS>0?NTHREADS:1;
      /* ★口径必须显式标注★: dequant 计数器同时接收两条路 —— 批量路(单线程调一次, 值即墙钟)
       * 与逐矩阵回退路(20 线程累计, 要 ÷线程数)。不标 batched 旗标就无法判读, 我曾误读过一次。 */
      fprintf(stderr,"[bmwt] 本层 batched=%d dequant=%.2fs 前向=%.2fs(20线程累计,墙钟约%.2f)\n",
#ifdef DS4QUANT_CUDA
              g_bmw_batched,
#else
              0,
#endif
              g_bmw_t[0],g_bmw_t[1],g_bmw_t[1]/nth2); }
}
/* 可复用调优轮(GL重拟合→GLdyn2→GLdyn8→TREF), 循环至整轮零接管; 返回是否有过接管 */
typedef struct { int L,S,n_fit,vs; size_t rowsz; double bval,bheld;
    const float *Fin,*Hf,*H2,*post2,*comb2,*shared,*Fcur;
    float *Fstate,*Ftest,*routedC,*DF,*Hq;
    double *sc_state,*v_cur,*h_cur; } costx_t;
