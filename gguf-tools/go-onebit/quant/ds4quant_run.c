/* ds4quant_run.c — 锚定-累积-快验: 43层量化前向以【最终输出质量】为判决, 逐层只是诊断。
 *
 * 方法学修正(2026-07-10, 取代旧"两遍前向+每层4档消融"):
 *   ① 逐层局部 relL2 高 ≠ 层本质差 —— 可能只是校准"没找到"(样本 1~3 行的 joint-LS 是噪声拟合)。
 *      → 每层报告校准样本数; 校准改为部署口径(anchor FP 激活 + FP 路由 + 只用 fit 行, 不再泄漏 held-out)。
 *   ② 每层独立最优 ≠ 43 层堆叠最优 —— 误差层间累积/抵消, 唯一判决 = 最终 logits 质量。
 *      → 量化遍逐层打印【累积】偏差(H_q vs anchor H_fp)+路由一致率, 结束打 VERDICT(Σmin/KL/PPL 主判据)。
 *
 * 机制:
 *   - FP 锚定遍只跑一次, 缓存到 DS4_ANCHOR(默认 /tmp/ds4quant_anchor.bin): 每层 MoE 输入 Fin_fp、
 *     FP 路由、每层出口 H_fp、最终 logits。之后所有配置验证只跑量化遍。
 *   - DS4_LCFG 逐层档位(NL 字符或 1 字符广播): F=不量化 n=朴素1bit 1=joint1bit(块scale,1.0625b)
 *     r=行scale1bit(1.0052b, 纯1bit体积主线) z=1bit+低秩 2=1bit+Q2 3=1bit+Q2+Q3。
 *     便捷: DS4_INJECT=L(只 L 层量化, 其余 F; F 前缀直接从 anchor 恢复跳过) / DS4_SPARE=L(只 L 层保 F)。
 *     全 F 配置是框架自检: VERDICT 必须 ratio=1.0000/Σmin≈1。
 *   - 专家循环 pthread 并行(DS4_THREADS) + Accelerate sgemm(ds4quant_fwd.c) → 单配置分钟级。
 *   - DS4_ABLATE=1 恢复旧 4 档局部消融表(慢, 诊断用); DS4_LOCAL_Q=0 可关每层局部 R²。
 *
 * 编译: cc -O3 -lm -framework Accelerate -I<repo根> ds4quant_run.c -o ds4quant_run -lpthread
 * 用法: DS4_HF=... ./ds4quant_run [ids_file=/tmp/rr_hard.ids] [ntok=64]
 *       (还原率铁律: 判决用硬多样文本 rr_hard.ids; Σmin/KL/PPL 为主, top1 仅参考)
 */
#ifdef __linux__
#include <malloc.h>   /* mallopt: 分配器设置写进代码, 不走 env(2026-08-22 铁律); glibc 专属, macOS 无此接口 */
#endif
#include "st_read.c"
#include "ds4quant_fwd.c"
#include "onebit_quant.c"
#include "ds4quant_qhelp.h"   /* dq_go1b_bytes_dequant, dq_quant_expert (从 layer.c 抽出) */
#include "go2b_qc.h"          /* GO2B 编码/解码+热表: 热专家合并2bit(残差+量化一体, 消漂移) */
#include "vq_qc.h"            /* v2.2 VQ 码本(DS4_VQ=1): 热 vq4x512 全三矩阵 / 冷 w1w3 vq8x256 / 冷 w2 signref */
#include "ds4_z.c"            /* 闭式秩-k RRR 隐变量 z (产物③) */
#include "ds4_loss.c"        /* 四损失: align/classify/smooth/fixed */
#include <pthread.h>
#include <time.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#ifdef __APPLE__
#include <mach/mach.h>   /* 内存自报: task_info phys_footprint(看门狗同口径) */
#endif
static int   ZRANK=16;        /* z 最大秩 (DS4_Z_RANK; 遗留 z 工具用) */
static float ZLAMBDA=0.1f;    /* ridge (无量纲, DS4_Z_LAMBDA) = L_fixed 锚 */
static int   NLAYERS=43;      /* 前向层数上限 (DS4_NL; <43 绝对PPL无效, 相对指标同深度仍可比) */
static int   ZK=16;           /* 低秩档 'z' 的秩 (DS4_ZK) */
static int   NTHREADS=8;      /* 专家循环并行度 (DS4_THREADS) */
static int   ABLATE=0;        /* DS4_ABLATE: 每量化层附带旧 4 档局部消融(慢) */
static int   LOCALQ=1;        /* DS4_LOCAL_Q: 每量化层局部 R²(选定档, held 行) */
static int   LZRANK=0;        /* DS4_LZ: 逐层动态 z^L 最大秩(0=关): 序贯锚定回拉+四损失选秩 */
static float LZLAMBDA=1.0f;   /* DS4_LZ_LAMBDA: z^L ridge(比全局 ZLAMBDA 紧; gz首跑过拟合链式反应教训) */
static float LZTR=0.5f;       /* DS4_LZ_TR: 信赖域 — 每 token ‖z(Fin)‖≤TR·‖Fout‖, 防把激活拉出流形 */
static long  LZ_TOTAL_K=0;    /* Σk_L (侧车体积账) */
static int   COADAPT=0;       /* DS4_COADAPT: 每层 base(w2)↔z 交替闭式共适应轮数(0=关=原路径不变) */
/* ★反修字节起步(2026-08-03 用户令"提速"): 冷专家基座=量化段盘上字节 dequant(部署口径,
 * 免重编码 480 冷 VQ + 240 冷 w2 重解); 热/D3/菜单/z/GSWEEP 反修本职照常。层入口 arm, 导出前 free。 */
static const uint8_t *BFB_VQMAP=NULL; static size_t BFB_VQMSZ=0;
static const uint8_t *BFB_W2=NULL;    static size_t BFB_SZD=0;
static uint8_t *BFB_W2COPY=NULL;      /* 冷 w2 字节快照(导出直拷; mmap 先释放故需持有副本) */
static int bf_from_bytes(void){ static int v=-1; if(v<0) v=getenv("DS4_BF_FROM_BYTES")?1:0; return v; }
static time_t TUNE_T0=0;      /* 全程计时起点 */
static double TUNE_MIN=0;     /* DS4_TUNE_MIN: 总时间预算(分钟, 0=不限) */
static int FAST=0;            /* DS4_FAST: 快速走流程(QC 5→1 变体, co_rounds 512→2 轮封顶, 不向前修复=逐层反修跳过); 验流程非出质量 */
static time_t CO_LT0=0;       /* 当前层调优起点 */
static double TUNE_LSEC=0;    /* 每层调优死线(秒)=总预算/量化层数; 超时→该层落当前累计最优 */
static int tune_over(void){
    /* 用户裁决(2026-07-11): 收敛绝对优先 — 时间参数只选档不截断;
     * 每层必须整轮零接管(本层+累积最优)才进下一层。恒 0 = 永不截断。 */
    (void)TUNE_LSEC;(void)CO_LT0;
    return 0;
}
static float *BWD_Hb=NULL,*BWD_Hc=NULL; static int BWD_L=-1;   /* 向后·终端反调: 搜索层的 base/校正 出口 H (DS4_BWD_FINAL) */
static float *EXP_FIN=NULL; static int *EXP_IDX=NULL; static int EXP_L=-1;   /* 导出=调优同口径: 累积激活+实际路由 */
static float GBL_G[43];        /* ★全局联合最优★: 每层 routed 增量的全局增益(回扫以最终输出判优, 默认1) */
static float BWDFIN_T=1.0f;                                      /* 终端反调选中的 t */
static int   STK_DYN=0; static float STK_W0=1,STK_W1=0,STK_NM=0,STK_SD=1;   /* GLdyn2 落地参数 */
static int   STK8_ON=0; static float *STK8_V8=NULL;   /* GLdyn8: PCA8 方向(只算一次, 首条记录入文件) */
/* 阶段化调优旋钮(四损失逐个阶段扫这些, 判优永远用统一 val 评分) */
static double DYN_LSM=0.1;    /* 光滑损失: 动态系数非截距 ridge 强度 */
static double DYN_LFIX=1.0;   /* 固定损失: ridge 总体倍率 */
/* ★四损失/感知一等公民(2026-07-13 用户裁决)★: 旋钮网格并进 dyn2/dyn8 求解本身 —
 * 旧设计"阶段2默认旋钮收敛→四损失/感知逐个重跑同族求解"结构性必输(去打已收敛局部最优,
 * 43层全零接管实证)。新设计: 每轮每形态在完整旋钮网格上各解一次, 统一 val 只进不退取最优,
 * 胜者记账到对应损失家族(fam: la=对齐 lc=分类 lf=固定 ls=光滑 pc=感知; df=默认)。 */
typedef struct { const char*fam; int lti; double lfix,lsm,wcls; } kvar_t;
static const kvar_t KGRID[]={
    {"df",-1, 1.0, 0.1, 0.0},
    {"la", 0, 0.3, 0.1, 0.0},
    {"lf", 2, 3.0, 0.1, 0.0},
    {"lf", 2,10.0, 0.1, 0.0},
    {"ls", 3, 1.0, 0.03,0.0},
    {"ls", 3, 1.0, 0.5, 0.0},
    {"lc", 1, 1.0, 0.1, 0.35},
    {"lc", 1, 1.0, 0.1, 0.7},
    {"lc", 1, 1.0, 0.1, 1.5},
    {"pc", 4, 1.0, 0.1, 0.5},
    {"pc", 4, 1.0, 0.1, 1.0},
    {"pc", 4, 1.0, 0.1, 2.0},
    {"pc", 4, 1.0, 0.1, 4.0},
};
#define NKGRID ((int)(sizeof(KGRID)/sizeof(KGRID[0])))
#define NKG_D2 6   /* dyn2 无 wrow 布线: 只扫前 6 个(默认+λfix+λsm) */
static double LTW_knob[5]; static int LTW_hit[5];   /* 家族记账(层内, 阶段2重置) */
static float *ZLP_U=NULL,*ZLP_V=NULL,*ZLP_Z=NULL;   /* 冻结 z^L 暂存(过闸→导出层文件后 append zl.RRR) */
static int ZLP_K=0,ZLP_L=-1; static float ZLP_TR=0.25f;
static double SOLVE_WCLS=0.0; /* 分类损失: 动态求解行权中的感知加权(0=关) */
static float TREF_T=1.0f;
/* 跨层反向已升级为 backfit_prev(逐层前进即反修前面所有层, 判据=前沿出口 vs FP锚), 见 fwd_all/backfit_prev */
/* ===== 统一模块日志: 【L层】元素 算法=… 进度=… 体积=… 还原度=… 研判=… ===== */
static int CUR_L=-1;
static void hb64(uint64_t n,char*b,size_t bs){
    if(n>=(1ULL<<30)) snprintf(b,bs,"%.2fGiB",(double)n/(1ULL<<30));
    else if(n>=(1ULL<<20)) snprintf(b,bs,"%.2fMiB",(double)n/(1ULL<<20));
    else if(n>=1024) snprintf(b,bs,"%.1fKB",(double)n/1024.0);
    else snprintf(b,bs,"%lluB",(unsigned long long)n);
}
/* 最小 npy v1 写出(f32 C-order, 64 对齐) — corr 重建门 X/R 捕获用 */
static void xr_npy(const char*p,const float*A,int n,int d){
    FILE*f=fopen(p,"wb"); if(!f){ perror(p); return; }
    char hdr[128]; int hl=snprintf(hdr,sizeof(hdr),"{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }",n,d);
    int total=10+hl, pad=((total+63)/64)*64-total;
    unsigned short hlen=(unsigned short)(hl+pad);
    fwrite("\x93NUMPY\x01\x00",1,8,f); fwrite(&hlen,2,1,f);
    fwrite(hdr,1,(size_t)hl,f);
    for(int i=0;i<pad-1;i++) fputc(' ',f);
    fputc('\n',f);
    fwrite(A,4,(size_t)n*d,f); fclose(f);
}
static void mlog(int L,const char*elem,const char*algo,const char*prog,uint64_t vol,
                 const char*restore,const char*verdict){
    char vb[24]; hb64(vol,vb,sizeof(vb));
    fprintf(stderr,"【L%d】%s 算法=%s 进度=%s 体积=%s 还原度=%s 研判=%s\n",
            L,elem,algo,prog,vb,restore,verdict);
}
/* ===== 表格记录(第一原则): 文件与表格一一对应, 每元素一条记录 ===== */
typedef struct { char name[16],algo[64]; uint64_t vol; float m1,m2,m3,m4; int verdict;
                 uint16_t paylen; uint8_t pay[96]; } elrec_t;
static elrec_t ELE[64]; static int NELE=0;
/* verdict: 1=正向落地 2=正向未落地 3=探索中 4=生效(结构性) */
static void el_addp(const char*nm,const char*al,uint64_t vol,double m1,double m2,double m3,double m4,int vd,
                    const void*pay,int paylen){
    if(NELE>=64) return; elrec_t*r=&ELE[NELE++];
    snprintf(r->name,16,"%s",nm); snprintf(r->algo,64,"%s",al);
    r->vol=vol; r->m1=(float)m1; r->m2=(float)m2; r->m3=(float)m3; r->m4=(float)m4; r->verdict=vd;
    r->paylen=0; if(pay&&paylen>0&&paylen<=96){ memcpy(r->pay,pay,(size_t)paylen); r->paylen=(uint16_t)paylen; }
    { const char*VT[5]={"?","✓正向落地","✓正向未落地","探索中","✓生效"};
      char rs[96];
      if(vd==4) snprintf(rs,sizeof(rs),"结构性");
      else if(!strcmp(nm,"1bit")) snprintf(rs,sizeof(rs),"val=%.4f held=%.4f",m1,m2);
      else snprintf(rs,sizeof(rs),"val=%.4f held=%.4f 提升%.1f%%",m1,m2,m3);
      const char*el = nm[0]=='z'?"z变量":
          (!strncmp(nm,"la.",3)?"损失·对齐": (!strncmp(nm,"lc.",3)?"损失·分类":
          (!strncmp(nm,"lf.",3)?"损失·固定": (!strncmp(nm,"ls.",3)?"损失·光滑":
          (!strncmp(nm,"pc.",3)?"感知": (!strncmp(nm,"bwd",3)?"向后":
          (!strncmp(nm,"loss",4)?"损失": (!strcmp(nm,"percept")?"感知":"量化"))))))));
      char an[96]; snprintf(an,sizeof(an),"%s(%s)",nm,al);
      mlog(CUR_L,el,an,"完成",vol,rs,VT[vd>=0&&vd<5?vd:0]);
    }
}
static void el_add(const char*nm,const char*al,uint64_t vol,double m1,double m2,double m3,double m4,int vd){
    el_addp(nm,al,vol,m1,m2,m3,m4,vd,NULL,0);
}
static float *STK_CLSV=NULL;   /* 分类损失的层感知加权向量(出口维方差, 入文件) */

#define DIM 4096
#define NH 64
#define HD 512
#define RD 64
#define QLR 1024
#define OLR 1024
#define OG 8
#define HCM 4
#define HCIT 20
#define MOEI 2048
#define WIN 128
#define NEXP 256
#define NACT 6
/* ★动态路由反修 Phase-B(2026-08-16 用户令"真正的路由反修")★:
 * NACT_RT=自路由运行时槽数(默认=NACT=6, 逐位不变)。层目录 rroute.txt(单整数 m)存在时
 * NACT_RT=6+m: 每 token 多请 m 个专家入场——权重按分数归一自动完成 margin 条件化
 * (平票处第7/8位分数≈第6位→实质对冲; 尖峰处分数小→权重趋零近似无稀释)。
 * 锚格式/t2ei 恒用 NACT=6, 不受影响。文件门控, 非 env(2026-08-15 用户铁律)。 */
static int NACT_RT = NACT;
static float RR_EPS = 0.0f;   /* >0 = 加法式并集: 前NACT名权重按原协议(6内归一)逐位不变,
                                 额外槽以 eps*sc 加法进场(不稀释前6质量; m1乘法归一版已判死于分数压缩) */
static void nact_rt_init(const char *layer_dir){
    if(!layer_dir) return;
    char p[512]; snprintf(p,sizeof p,"%s/rroute.txt",layer_dir);
    FILE *f=fopen(p,"r"); if(!f) return;
    int m=0; float ep=0.0f;
    if(fscanf(f,"%d %f",&m,&ep)>=1 && m>0 && m<=8){ NACT_RT=NACT+m; RR_EPS=ep;
        fprintf(stderr,"[路由反修] rroute.txt m=%d eps=%.3f → NACT_RT=%d %s\n",
                m,ep,NACT_RT,ep>0?"加法式":"归一式"); }
    fclose(f);
}
#define ROUTE_SCALE 1.5f
#define SWLIM 10.0f
#define EPSF 1e-6f
#define NL 43
#define VOCAB 129280

static int CR[44]={0,0,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,128,4,0};
static st_ctx C;
static char LCFG[NL+1];

typedef struct { float *afn,*asc,*abase,*ffn,*fsc,*fbase,*an,*fn,*wqa,*qn,*wqb,*wkv,*kvn,*sink,*woa,*wob;
                 float *cwkv,*cwgate,*cnorm,*cape; float *gate,*gbias,*t2e; int *t2ei; float *s1,*s3,*s2; } LW;
/* GS_LW(fp16 骨干缓存)声明移至 LWH 定义后 */
static float *GS_HCFN=NULL,*GS_HCB=NULL,*GS_HCS=NULL,*GS_NORM=NULL,*GS_HW=NULL;  /* 回扫: head 权重(head.weight 2.1GB)只读一次 */

static long G_N=0;   /* G 副产物: 最近一次读取的元素数(fp16 缓存转换用) */
static float *G(const char*fmt,int L){char n[160];snprintf(n,sizeof(n),fmt,L);long r,c;float*w=st_read_weight(&C,n,&r,&c);
    G_N=w?(r>0?r:1)*(c>0?c:1):0; return w;}
/* ★内存红线★: 骨干单层 fp32≈435MB(wqb151+woa134+shared100+…), 43层全缓存=18.7GB 物理不可能
 * (实测 L20 撞 12G 看门狗)。缓存改 fp16(≈9.3GB), 前向按层临时展开 fp32(~0.1s/层访, 换活命)。*/
#define LWF_LIST(X) X(afn) X(asc) X(abase) X(ffn) X(fsc) X(fbase) X(an) X(fn) \
    X(wqa) X(qn) X(wqb) X(wkv) X(kvn) X(sink) X(woa) X(wob) \
    X(cwkv) X(cwgate) X(cnorm) X(cape) X(gate) X(gbias) X(t2e) X(s1) X(s3) X(s2)
typedef struct {
#define XF(f) uint16_t *f; long n_##f;
    LWF_LIST(XF)
#undef XF
    int *t2ei; int loaded;
} LWH;

static LW load_layer2(int L, LWH*H){ LW W={0};
#define GC(dst,fmt) do{ W.dst=G(fmt,L); if(H) H->n_##dst=G_N; }while(0)
    GC(afn,"layers.%d.hc_attn_fn");GC(asc,"layers.%d.hc_attn_scale");GC(abase,"layers.%d.hc_attn_base");
    GC(ffn,"layers.%d.hc_ffn_fn");GC(fsc,"layers.%d.hc_ffn_scale");GC(fbase,"layers.%d.hc_ffn_base");
    GC(an,"layers.%d.attn_norm.weight");GC(fn,"layers.%d.ffn_norm.weight");
    GC(wqa,"layers.%d.attn.wq_a.weight");GC(qn,"layers.%d.attn.q_norm.weight");GC(wqb,"layers.%d.attn.wq_b.weight");
    GC(wkv,"layers.%d.attn.wkv.weight");GC(kvn,"layers.%d.attn.kv_norm.weight");GC(sink,"layers.%d.attn.attn_sink");
    GC(woa,"layers.%d.attn.wo_a.weight");GC(wob,"layers.%d.attn.wo_b.weight");
    if(CR[L]>0){GC(cwkv,"layers.%d.attn.compressor.wkv.weight");GC(cwgate,"layers.%d.attn.compressor.wgate.weight");
        GC(cnorm,"layers.%d.attn.compressor.norm.weight");GC(cape,"layers.%d.attn.compressor.ape");}
    GC(gate,"layers.%d.ffn.gate.weight");
    if(L<=2){ GC(t2e,"layers.%d.ffn.gate.tid2eid"); }   /* 仅 L0-L2 有 hash 路由表, L3+ 不探测(消 st:no 噪声) */
    if(W.t2e){ W.t2ei=malloc((size_t)VOCAB*NACT*sizeof(int));for(size_t i=0;i<(size_t)VOCAB*NACT;i++)W.t2ei[i]=(int)W.t2e[i]; }
    else { GC(gbias,"layers.%d.ffn.gate.bias"); }  /* L3+(top-k 路由) */
    GC(s1,"layers.%d.ffn.shared_experts.w1.weight");GC(s3,"layers.%d.ffn.shared_experts.w3.weight");GC(s2,"layers.%d.ffn.shared_experts.w2.weight");
#undef GC
    return W; }
static LW load_layer(int L){ return load_layer2(L,NULL); }
static void free_layer(LW*W){
    float*ff[]={W->afn,W->asc,W->abase,W->ffn,W->fsc,W->fbase,W->an,W->fn,W->wqa,W->qn,W->wqb,W->wkv,W->kvn,W->sink,W->woa,W->wob,W->cwkv,W->cwgate,W->cnorm,W->cape,W->gate,W->gbias,W->t2e,W->s1,W->s3,W->s2};
    for(size_t i=0;i<sizeof(ff)/sizeof(ff[0]);i++) if(ff[i]) free(ff[i]);
    if(W->t2ei) free(W->t2ei);
}
/* fp32 LW(load_layer2 已录计数) → fp16 吸收进缓存(W 字段仍归调用方 free_layer) */
static void lwh_absorb(LWH*H,const LW*W){
#define XA(f) if(W->f&&H->n_##f>0){ H->f=malloc((size_t)H->n_##f*2); \
        for(long i=0;i<H->n_##f;i++) H->f[i]=go1b_fp32_to_fp16(W->f[i]); } else { H->f=NULL; H->n_##f=W->f?H->n_##f:0; }
    LWF_LIST(XA)
#undef XA
    if(W->t2ei){ H->t2ei=malloc((size_t)VOCAB*NACT*sizeof(int));
        memcpy(H->t2ei,W->t2ei,(size_t)VOCAB*NACT*sizeof(int)); }
    H->loaded=1;
}
/* fp16 缓存 → 临时 fp32 LW(用完 free_layer; ~0.1s/层, 换 43 层缓存活在 16G 里) */
static LW lwh_expand(const LWH*H){
    LW W={0};
#define XE(f) if(H->f){ W.f=malloc((size_t)H->n_##f*4); for(long i=0;i<H->n_##f;i++) W.f[i]=go1b_fp16_to_fp32(H->f[i]); }
    LWF_LIST(XE)
#undef XE
    if(H->t2ei){ W.t2ei=malloc((size_t)VOCAB*NACT*sizeof(int));
        memcpy(W.t2ei,H->t2ei,(size_t)VOCAB*NACT*sizeof(int)); }
    return W;
}
static void lwh_free(LWH*H){
#define XG(f) if(H->f) free(H->f);
    LWF_LIST(XG)
#undef XG
    if(H->t2ei) free(H->t2ei);
    memset(H,0,sizeof(*H));
}
static LWH *GS_LW=NULL;        /* 回扫/反修: 全43层骨干 fp16 缓存(只加载一次, 前向按层临时展开) */
/* ★内存卫兵★: 实测斜率 0.281GB/层(fp16 0.217 + 杂项), 外推 L38 必撞 11.5G 看门狗 →
 * 超预算(DS4_BF_MEMGB, 默认9.5)驱逐最低层号的 fp16 缓存; 被驱逐层前向时临时从 HF 重载(慢~0.6s/访, 换到头)。*/
#ifdef __APPLE__
static double mem_gb(void){ task_vm_info_data_t vi; mach_msg_type_number_t vc=TASK_VM_INFO_COUNT;
    if(task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&vi,&vc)!=KERN_SUCCESS) return 0;
    return (double)vi.phys_footprint/1073741824.0; }
#else
static double mem_gb(void){   /* Linux: VmRSS(看门狗近似口径) */
    FILE *f = fopen("/proc/self/status", "r"); if (!f) return 0;
    char ln[128]; double gb = 0;
    while (fgets(ln, sizeof(ln), f)) {
        long kb;
        if (sscanf(ln, "VmRSS: %ld kB", &kb) == 1) { gb = (double)kb / 1048576.0; break; }
    }
    fclose(f); return gb; }
#endif
static double BF_MEMGB=9.5;
static void lwh_free(LWH*H);
static void gs_lw_evict(int upto){   /* 驱逐 [0,upto) 里已缓存的最低层, 直到回预算 */
    if(!GS_LW) return;
    while(mem_gb()>BF_MEMGB){
        int v=-1; for(int i=0;i<upto;i++) if(GS_LW[i].loaded){ v=i; break; }
        if(v<0) break;
        lwh_free(&GS_LW[v]);
        fprintf(stderr,"[mem] 驱逐 L%02d fp16缓存(footprint=%.2fGB>%.1f预算; 该层链访转临时重载)\n",v,mem_gb(),BF_MEMGB);
    }
}

/* dual-form z-solve: n<<d_in 时 W = X^T (XX^T+λI)^-1 R, Cholesky 在 n×n(而非 d_in³).
 * 数学等价 ds4_z_solve(同 ridge 正规方程, dual 恒等式), 但 O(n³+n²·d) 而非 O(d³): 90s→<1s.
 * 复用 ds4_z.c 同-TU static 原语 cholesky/cholesky_solve/mgs/lcg_unit + 子空间迭代因子分解.
 * (遗留 z 工具链的落地实现, 本文件当前不调用, 保留供 z_explore 流复活。) */
__attribute__((unused))
/* ★z_solve_dual 并行化(2026-08-23 时长账: 单线程 600GFLOP=5min/层 → 20 线程)★
 * 数学不动: Gram 上三角/回代列/W 重建行 三段独立可并行; cholesky 分解保持单线程。 */
typedef struct { void (*fn)(void*,int,int); void *ctx; int i0,i1; } zpf_arg;
static void *zpf_tramp(void *a){ zpf_arg *p=(zpf_arg*)a; p->fn(p->ctx,p->i0,p->i1); return NULL; }
static void zpar_for(int n,int nth,void (*fn)(void*,int,int),void *ctx){
    if(nth>n)nth=n>0?n:1; if(nth>32)nth=32;
    pthread_t th[32]; zpf_arg pa[32]; int per=(n+nth-1)/nth,cnt=0;
    for(int t=0;t<nth;t++){ int i0=t*per,i1=i0+per>n?n:i0+per; if(i0>=i1)break;
        pa[cnt]=(zpf_arg){fn,ctx,i0,i1};
        if(pthread_create(&th[cnt],NULL,zpf_tramp,&pa[cnt])){ pa[cnt].fn(ctx,i0,i1); continue; }
        cnt++; }
    for(int t=0;t<cnt;t++) pthread_join(th[t],NULL);
}
typedef struct { const float*X; double*G; uint32_t n,d_in; } zg_ctx;
static void zg_worker(void *vc,int a0,int a1){ zg_ctx*c=(zg_ctx*)vc;
    for(int a=a0;a<a1;a++){ const float*xa=c->X+(size_t)a*c->d_in;
        for(uint32_t b=(uint32_t)a;b<c->n;b++){ const float*xb=c->X+(size_t)b*c->d_in; double s=0;
            for(uint32_t i=0;i<c->d_in;i++) s+=(double)xa[i]*xb[i]; c->G[(size_t)a*c->n+b]=s; } } }
typedef struct { const double*G; const float*R; double*al; uint32_t n,d_out; } zs_ctx;
static void cholesky_solve(const double *L, uint32_t d, double *b);   /* ds4_z.c 同-TU 原语(本就 const) */
#define cholesky_solve_nc cholesky_solve
static void zs_worker(void *vc,int j0,int j1){ zs_ctx*c=(zs_ctx*)vc;
    double *cc=malloc((size_t)c->n*sizeof(double));
    for(int j=j0;j<j1;j++){ for(uint32_t a=0;a<c->n;a++) cc[a]=c->R[(size_t)a*c->d_out+j];
        cholesky_solve_nc(c->G,c->n,cc);
        for(uint32_t a=0;a<c->n;a++) c->al[(size_t)a*c->d_out+j]=cc[a]; }
    free(cc); }
typedef struct { const float*X; const double*al; float*W; uint32_t n,d_in,d_out; } zw_ctx;
static void zw_worker(void *vc,int i0,int i1){ zw_ctx*c=(zw_ctx*)vc;
    for(int i=i0;i<i1;i++){ float*Wi=c->W+(size_t)i*c->d_out;
        for(uint32_t j=0;j<c->d_out;j++)Wi[j]=0.0f;
        for(uint32_t a=0;a<c->n;a++){ double xai=c->X[(size_t)a*c->d_in+i]; if(xai==0.0)continue;
            const double*ala=c->al+(size_t)a*c->d_out;
            for(uint32_t j=0;j<c->d_out;j++) Wi[j]+=(float)(xai*ala[j]); } } }
static ds4_z *z_solve_dual(const float *X, const float *R, uint32_t n,
                           uint32_t d_in, uint32_t d_out, uint32_t rank, float lambda){
    if(!X||!R||!n||!d_in||!d_out||!rank) return NULL;
    if(rank>d_in)rank=d_in; if(rank>d_out)rank=d_out; if(rank>n)rank=n;
    double *G=calloc((size_t)n*n,sizeof(double)); if(!G) return NULL;
    { zg_ctx gc={X,G,n,d_in}; zpar_for((int)n,20,zg_worker,&gc); }
    double tr=0; for(uint32_t a=0;a<n;a++) tr+=G[(size_t)a*n+a];       /* tr(XX^T)=tr(X^TX) */
    double ridge=(double)lambda*(tr/(double)d_in)+1e-10;              /* primal 口径 ridge */
    for(uint32_t a=0;a<n;a++){ G[(size_t)a*n+a]+=ridge;
        for(uint32_t b=a+1;b<n;b++) G[(size_t)b*n+a]=G[(size_t)a*n+b]; }
    if(cholesky(G,n)!=0){ free(G); return NULL; }
    double *al=malloc((size_t)n*d_out*sizeof(double));
    if(!al){free(G);return NULL;}
    { zs_ctx sc={G,R,al,n,d_out}; zpar_for((int)d_out,20,zs_worker,&sc); }   /* α=G^-1 R, 列并行 */
    free(G);
    float *W=malloc((size_t)d_in*d_out*sizeof(float)); if(!W){free(al);return NULL;}
    { zw_ctx wc={X,al,W,n,d_in,d_out}; zpar_for((int)d_in,20,zw_worker,&wc); }  /* W=X^T α, 行并行 */
    free(al);
    /* 子空间迭代 rank-k 截断 (与 ds4_z_solve 尾部一致) */
    ds4_z *zl=calloc(1,sizeof(*zl));
    float *V=malloc((size_t)d_in*rank*4),*T=malloc((size_t)d_out*rank*4),*M=malloc((size_t)rank*d_out*4),*U=malloc((size_t)d_out*rank*4),*z=malloc((size_t)rank*4);
    if(!zl||!V||!T||!M||!U||!z){free(zl);free(V);free(T);free(M);free(U);free(z);free(W);return NULL;}
    uint64_t seed=0x5A5A1EEDULL;
    for(size_t i=0;i<(size_t)d_in*rank;i++) V[i]=lcg_unit(&seed);
    mgs(V,d_in,rank,&seed);
    for(int it=0;it<12;it++){
        memset(T,0,(size_t)d_out*rank*4);
        for(uint32_t i=0;i<d_in;i++){ const float*Wi=W+(size_t)i*d_out,*Vi=V+(size_t)i*rank;
            for(uint32_t j=0;j<d_out;j++){ float wij=Wi[j]; if(wij==0.0f)continue; float*Tj=T+(size_t)j*rank;
                for(uint32_t c=0;c<rank;c++) Tj[c]+=wij*Vi[c]; } }
        for(uint32_t i=0;i<d_in;i++){ const float*Wi=W+(size_t)i*d_out; float*Vi=V+(size_t)i*rank;
            for(uint32_t c=0;c<rank;c++)Vi[c]=0.0f;
            for(uint32_t j=0;j<d_out;j++){ float wij=Wi[j]; if(wij==0.0f)continue; const float*Tj=T+(size_t)j*rank;
                for(uint32_t c=0;c<rank;c++) Vi[c]+=wij*Tj[c]; } }
        mgs(V,d_in,rank,&seed);
    }
    memset(M,0,(size_t)rank*d_out*4);
    for(uint32_t i=0;i<d_in;i++){ const float*Wi=W+(size_t)i*d_out,*Vi=V+(size_t)i*rank;
        for(uint32_t c=0;c<rank;c++){ float v=Vi[c]; if(v==0.0f)continue; float*Mc=M+(size_t)c*d_out;
            for(uint32_t j=0;j<d_out;j++) Mc[j]+=v*Wi[j]; } }
    for(uint32_t c=0;c<rank;c++){ double nrm=0; const float*Mc=M+(size_t)c*d_out;
        for(uint32_t j=0;j<d_out;j++) nrm+=(double)Mc[j]*Mc[j]; nrm=sqrt(nrm); z[c]=(float)nrm;
        float inv=nrm>1e-20?(float)(1.0/nrm):0.0f; for(uint32_t j=0;j<d_out;j++) U[(size_t)j*rank+c]=Mc[j]*inv; }
    free(M);free(T);free(W);
    for(uint32_t a=0;a<rank;a++){ uint32_t best=a; for(uint32_t b=a+1;b<rank;b++) if(z[b]>z[best])best=b;
        if(best!=a){ float tz=z[a];z[a]=z[best];z[best]=tz;
            for(uint32_t i=0;i<d_in;i++){float tv=V[(size_t)i*rank+a];V[(size_t)i*rank+a]=V[(size_t)i*rank+best];V[(size_t)i*rank+best]=tv;}
            for(uint32_t j=0;j<d_out;j++){float tu=U[(size_t)j*rank+a];U[(size_t)j*rank+a]=U[(size_t)j*rank+best];U[(size_t)j*rank+best]=tu;} } }
    zl->U=U;zl->V=V;zl->z=z;zl->d_in=d_in;zl->d_out=d_out;zl->rank=rank;zl->k=rank; return zl;
}

/* 四损失选秩 k_L (产物③旋钮): 在【留出 val 行】(X[n,d]→R[n,d]) 上对候选 k(含 k=0=该层
 * z 关闭的 GO/NO-GO 门) 算 L_align/L_classify/L_smooth/L_fixed, 归一加权取最小。
 * gz 首跑教训: 在拟合行自评→过拟合链式反应(L26 拉出流形→L28 累积1.79); val 评估+k0 门是正解。 */
static int z_pick_rank(ds4_z*zl, const float*X, const float*R, int n, int d,
                       double*la,double*lc,double*ls,double*lx){
    int rank=(int)zl->rank; if(rank<1)return 0;
    float *yh=malloc((size_t)n*d*4);
    float *wv=malloc((size_t)d*4); ds4_loss_dim_variance(R,n,d,wv);
    float *dv=malloc((size_t)d*4);
    uint64_t sd=0x9E3779B97F4A7C15ULL;
    for(int i=0;i<d;i++){ sd=sd*6364136223846793005ULL+1442695040888963407ULL; dv[i]=((float)((sd>>40)&0xFFFFFF)/16777216.0f-0.5f)*0.02f; }
    float *yd=malloc((size_t)d*4);
    int ks[12],nk=0; ks[nk++]=0; for(int k=1;k<=rank;k*=2){ ks[nk++]=k; } if(ks[nk-1]!=rank)ks[nk++]=rank;
    double La[12],Lc[12],Ls[12],Lx[12];
    for(int ki=0;ki<nk;ki++){ int kk=ks[ki];
        memset(yh,0,(size_t)n*d*4);
        if(kk>0){ ds4_z_set_rank(zl,(uint32_t)kk);
            for(int s=0;s<n;s++) ds4_z_apply(zl,X+(size_t)s*d,yh+(size_t)s*d); }
        La[ki]=ds4_loss_align(yh,R,(uint32_t)n,(uint32_t)d);
        Lc[ki]=ds4_loss_classify(yh,R,wv,(uint32_t)n,(uint32_t)d);
        if(kk>0){ memset(yd,0,(size_t)d*4); ds4_z_apply(zl,dv,yd);
            double sm=0; for(int i=0;i<d;i++) sm+=(double)yd[i]*yd[i]; Ls[ki]=sm;
            double en=0; for(int c=0;c<kk;c++) en+=(double)zl->z[c]*zl->z[c]; Lx[ki]=en;
        } else { Ls[ki]=0.0; Lx[ki]=0.0; }
    }
    double mA=1e-30,mC=1e-30,mS=1e-30,mX=1e-30;
    for(int ki=0;ki<nk;ki++){ if(La[ki]>mA)mA=La[ki]; if(Lc[ki]>mC)mC=Lc[ki]; if(Ls[ki]>mS)mS=Ls[ki]; if(Lx[ki]>mX)mX=Lx[ki]; }
    int bk=0; double bL=1e300;
    for(int ki=0;ki<nk;ki++){ double t=1.0*(La[ki]/mA)+0.5*(Lc[ki]/mC)+0.15*(Ls[ki]/mS)+0.05*(Lx[ki]/mX);
        if(t<bL){bL=t;bk=ki;} }
    if(ks[bk]>0) ds4_z_set_rank(zl,(uint32_t)ks[bk]);
    if(la)*la=La[bk]; if(lc)*lc=Lc[bk]; if(ls)*ls=Ls[bk]; if(lx)*lx=Lx[bk];
    free(yh);free(wv);free(dv);free(yd); return ks[bk];
}

/* 四损失驱动的 z 求解(不只选秩, 四损失进求解本身; ALGORITHM.md §4 语义):
 *   L_align    = RRR/ridge 数据项本体(方向还原);
 *   L_classify = R 列按 per-dim 方差(重要性)加权 —— sqrt(v_d) 缩放进低秩截断(加权低秩),
 *                解完 U 行反缩放还原(逐 dim ridge 解本身不受列权影响, 影响的正是秩截断取舍);
 *   L_smooth   = 固定种子 dither 增广行 (x_t+δ_t → 同目标 R_t, 行权 augw): 线性 z 下
 *                即惩罚 ‖z(x+δ)−z(x)‖², 输入鲁棒/不跳变;
 *   L_fixed    = ridge λ(weight-decay, 外层按过拟合比动态) + 同一 dither 增广钉死解。
 * (加法 z^L 形态 val 门实测判 0 后退役; 保留给 legacy z 路径/后续复用。) */
__attribute__((unused))
static ds4_z *z_solve_fourloss(const float *X,const float *R,int n,int d_in,int d_out,int rank,float lambda,
                               const float *colw,int naug,float augw,uint64_t seed){
    /* ★2026-08-08 修: 原单一 d 同时当 d_in/d_out 用 — R 为矩形(如专家级 4096→2048)时
     * 行步长读越界 SIGSEGV。z_solve_dual 底层本就支持 d_in≠d_out, 包装层跟上。 */
    if(n<1||rank<1) return NULL;
    if(naug>n) naug=n;
    int N=n+naug;
    float *Xa=malloc((size_t)N*d_in*4), *Ra=malloc((size_t)N*d_out*4);
    if(!Xa||!Ra){ free(Xa);free(Ra); return NULL; }
    memcpy(Xa,X,(size_t)n*d_in*4);
    for(int i=0;i<n;i++) for(int j=0;j<d_out;j++)
        Ra[(size_t)i*d_out+j]=R[(size_t)i*d_out+j]*(colw?sqrtf(colw[j]):1.0f);
    uint64_t s=seed?seed:0x9E3779B97F4A7C15ULL;
    float sw=sqrtf(augw>0?augw:0.25f);
    for(int a=0;a<naug;a++){
        int src=a%n;
        const float*xs=X+(size_t)src*d_in; float*xd=Xa+(size_t)(n+a)*d_in;
        double rms=0; for(int j=0;j<d_in;j++) rms+=(double)xs[j]*xs[j]; rms=sqrt(rms/d_in+1e-30);
        for(int j=0;j<d_in;j++){ s=s*6364136223846793005ULL+1442695040888963407ULL;
            float dl=((float)((s>>40)&0xFFFFFF)/16777216.0f-0.5f)*0.04f*(float)rms;
            xd[j]=(xs[j]+dl)*sw; }
        const float*rs=Ra+(size_t)src*d_out; float*rd=Ra+(size_t)(n+a)*d_out;
        for(int j=0;j<d_out;j++) rd[j]=rs[j]*sw;
    }
    ds4_z *zl=z_solve_dual(Xa,Ra,(uint32_t)N,(uint32_t)d_in,(uint32_t)d_out,(uint32_t)rank,lambda);
    free(Xa);free(Ra);
    if(zl&&colw){ for(int j=0;j<d_out;j++){ float iw=colw[j]>1e-12f?1.0f/sqrtf(colw[j]):0.0f;
        for(uint32_t c=0;c<zl->rank;c++) zl->U[(size_t)j*zl->rank+c]*=iw; } }
    return zl;
}

/* ===================== FP 锚定 (anchor) ===================== *
 * FP 遍的完整快照, 只算一次: 每层 MoE 输入(校准=部署口径)/FP 路由/层出口 H(累积对比+前缀恢复)
 * + 最终 logits(判决基线)。header 校验 (S, NLAYERS, ids 哈希) 不符自动重建。 */
typedef struct { uint32_t S; uint64_t idh;
    float *fin;      /* [NLAYERS][S][DIM]      每层 MoE 输入 (rms 后) */
    int32_t *ridx;   /* [NLAYERS][S][NACT]     FP 路由专家 id */
    float *rw;       /* [NLAYERS][S][NACT]     FP 路由权重 */
    float *H;        /* [NLAYERS][S][HCM][DIM] 每层出口 hc 状态 */
    float *logits;   /* [S][VOCAB] */
} anchor_t;
static anchor_t ANC; static int ANC_OK=0, ANC_BUILD=0;
/* ★链态锚直写(2026-08-10 反修v4: 量化链 fin/路由/H 逐层 pwrite, 零 RAM; DS4_CHAIN_ANCHOR)★ */
static int g_chfd=-1; static int g_chS=0;
/* ★统一标准(2026-07-29 用户令): 判决语料=rr_hard(与冠军 v4bf 0.7680 同尺)。ANC2=rr FP 锚,
 * H2=贪心逐层推进的 rr 判决链; v5mini 降级为校准+层门护栏(判决数字一律 rr 口径)。 */
static anchor_t ANC2; static int ANC2_OK=0;
static long *g_ids2=NULL; static int g_S2=0;

static const char *anchor_path(void){ const char*p=getenv("DS4_ANCHOR"); return p?p:"/tmp/ds4quant_anchor.bin"; }
static uint64_t dq_ids_hash(const long*ids,int S){
    uint64_t h=1469598103934665603ULL;
    for(int i=0;i<S;i++){ uint64_t v=(uint64_t)ids[i];
        for(int b=0;b<8;b++){ h^=(v>>(8*b))&0xFF; h*=1099511628211ULL; } }
    return h;
}
static void anchor_alloc(int S){
    ANC.S=(uint32_t)S;
    ANC.fin =malloc((size_t)NLAYERS*S*DIM*4);
    ANC.ridx=malloc((size_t)NLAYERS*S*NACT*sizeof(int32_t));
    ANC.rw  =malloc((size_t)NLAYERS*S*NACT*4);
    ANC.H   =malloc((size_t)NLAYERS*S*HCM*DIM*4);
    if(!ANC.fin||!ANC.ridx||!ANC.rw||!ANC.H){ fprintf(stderr,"anchor: 内存分配失败\n"); exit(1); }
}
/* 大块落盘必须分块+逐块 fsync: 19G 锚一把 fwrite 的脏文件页叠在 ~19G 换页捕获缓冲上,
 * 16G 机上 memorystatus 直接 SIGKILL(08-11 FP_ONLY 与 quant 内捕获两连杀实锤)。 */
static int wr_fs(FILE*f,const void*p,size_t nbytes){
    const uint8_t*b=p; const size_t CH=268435456;
    for(size_t o=0;o<nbytes;o+=CH){ size_t n=nbytes-o<CH?nbytes-o:CH;
        if(fwrite(b+o,1,n,f)!=n) return 0;
        fflush(f); fsync(fileno(f)); }
    return 1;
}
static int anchor_save(void){
    /* ★NL<43 禁覆盖(2026-08-03 事故: DS4_NL=1 探针反修现场重建 1 层锚并覆盖 6.46G 全量锚,
     * 全链被迫重跑 FP 锚定遍): 截断跑的锚只在内存用, 不落盘。 */
    if(NLAYERS<43){ fprintf(stderr,"anchor: NL=%d<43 截断锚不落盘(防覆盖全量锚)\n",NLAYERS); return 0; }
    FILE*f=fopen(anchor_path(),"wb"); if(!f){ fprintf(stderr,"anchor: 写 %s 失败\n",anchor_path()); return 0; }
    uint32_t hd[8]={0x32415144u,ANC.S,HCM,DIM,(uint32_t)NLAYERS,VOCAB,NACT,0};  /* "DQA2" LE */
    int S=(int)ANC.S, ok=1;
    ok &= fwrite(hd,4,8,f)==8 && fwrite(&ANC.idh,8,1,f)==1;
    ok &= wr_fs(f,ANC.fin ,4ul*(size_t)NLAYERS*S*DIM);
    ok &= wr_fs(f,ANC.ridx,4ul*(size_t)NLAYERS*S*NACT);
    ok &= wr_fs(f,ANC.rw  ,4ul*(size_t)NLAYERS*S*NACT);
    ok &= wr_fs(f,ANC.H   ,4ul*(size_t)NLAYERS*S*HCM*DIM);
    ok &= wr_fs(f,ANC.logits,4ul*(size_t)S*VOCAB);
    fclose(f); if(!ok) fprintf(stderr,"anchor: 写不完整\n");
    return ok;
}
static int ANC_MMAP=0;   /* 1=ANC 数组指向 mmap(禁 free) */
static int anchor_load(int S,uint64_t idh){
    /* ★反修内存架构(2026-07-31 用户令"反修必须跑, 内存大改代码"): 锚 ~4.5G 由 malloc+fread
     * (脏页不可回收)改为整文件 mmap 零拷贝 — MAP_PRIVATE 写时复制(捕获路偶发写只脏所触页),
     * 读页=干净文件页, 内存压力下可逐出重读 ⇒ 反修基线脏内存 -4.5G, 终局 sweep 进 12G 红线。 */
    int fd=open(anchor_path(),O_RDONLY); if(fd<0) return 0;
    uint32_t hd[8]; uint64_t h;
    if(read(fd,hd,32)!=32||read(fd,&h,8)!=8){close(fd);return 0;}
    if(hd[0]!=0x32415144u||hd[1]!=(uint32_t)S||hd[2]!=HCM||hd[3]!=DIM||
       hd[4]!=(uint32_t)NLAYERS||hd[5]!=VOCAB||hd[6]!=NACT||h!=idh){close(fd);return 0;}
    size_t fin_b=(size_t)NLAYERS*S*DIM*4, ridx_b=(size_t)NLAYERS*S*NACT*4, rw_b=ridx_b,
           H_b=(size_t)NLAYERS*S*HCM*DIM*4, lg_b=(size_t)S*VOCAB*4;
    size_t need=40+fin_b+ridx_b+rw_b+H_b+lg_b;
    struct stat stt;
    if(fstat(fd,&stt)!=0||(size_t)stt.st_size<need){ close(fd); return 0; }
    void*mp=mmap(NULL,need,PROT_READ|PROT_WRITE,MAP_PRIVATE,fd,0);
    close(fd);
    if(mp==MAP_FAILED) return 0;
    uint8_t*p=(uint8_t*)mp+40;
    ANC.S=(uint32_t)S; ANC.idh=idh;
    ANC.fin=(float*)p;        p+=fin_b;
    ANC.ridx=(int32_t*)p;     p+=ridx_b;
    ANC.rw=(float*)p;         p+=rw_b;
    ANC.H=(float*)p;          p+=H_b;
    ANC.logits=(float*)p;
    ANC_MMAP=1;
    fprintf(stderr,"anchor: mmap 零拷贝装载 %.2f GiB(干净页可逐出)\n",need/1073741824.0);
    return 1;
}

/* ===================== 专家量化 + 应用 ===================== *
 * 量化三矩阵(档位 cfg)并把 routed 输出 scatter-add 进 dst[S,DIM]。
 * Xc/ncal = anchor FP 校准行(部署口径: 离线量化器只见 FP 激活);
 * w2 的校准 hidden 用【量化后 q1/q3】作用于校准行(顺序误差补偿, 与逐矩阵顺序量化一致)。 */
#define MV_HOTK 16   /* 冠军热档专家数(g10h16 档梯/影子链用) */

static void co_hc(const float*X,int n,const float*q1,const float*q3,float*hc);
/* ★不得 static 缓存(2026-08-01 实锤 bug): 计划表 vq_rplan(L) 每层 setenv 改 w2 档, 一次性
 * 缓存会把全 43 层锁死在 L00 的值 —— 实测 L18 起 w2 该 24×256 却仍用 16×256, 每层多 38 MiB,
 * 全模型超支 1.27 GiB(29.2 vs 目标 28), 且会在 L38 撞体积闸掐断整轮。冷档 vq_cold_dim()
 * 无此病(它读每层更新的全局 g_vq_dim)。每次 getenv 的开销相对一层 VQ 编码可忽略。 */
static int vq_w2_dim(void){ const char*e=getenv("DS4_VQ_W2_DIM"); int v=e?atoi(e):0; if(v&&v!=4&&v!=8&&v!=12&&v!=16&&v!=24&&v!=32) v=0; return v; }
/* ★R28 每层计划(DS4_VQ_RPLAN, 2026-07-31): "L=%d dim=%d nc=%d hot=%d w2dim=%d w2nc=%d"
 * 动态层大小(每层不同码本档) + 动态冷热(每层不同热专家数)。层序处理, 单全局安全。 */
static int g_vq_L=-1; int g_vq_dim=0, g_vq_nc=0; static int g_vq_hot=0;
static int vq_rplan(int L){
    const char*pp=getenv("DS4_VQ_RPLAN"); if(!pp) return 0;
    FILE*f=fopen(pp,"r"); if(!f){ fprintf(stderr,"[R28] 计划表打不开 %s\n",pp); exit(2); }
    char ln[256]; int l,d,n,h,wd,wn,ok=0;
    while(fgets(ln,sizeof(ln),f))
        if(sscanf(ln,"L=%d dim=%d nc=%d hot=%d w2dim=%d w2nc=%d",&l,&d,&n,&h,&wd,&wn)==6 && l==L){
            g_vq_L=L; g_vq_dim=d; g_vq_nc=n; g_vq_hot=h; ok=1;
            char b[16]; snprintf(b,16,"%d",wd); setenv("DS4_VQ_W2_DIM",b,1);
            snprintf(b,16,"%d",wn); setenv("DS4_VQ_W2_NC",b,1);
            break; }
    fclose(f);
    if(!ok){ fprintf(stderr,"[R28] 计划表缺 L%d\n",L); exit(2); }
    fprintf(stderr,"[R28] L%02d 档位 vq%dx%d 热%d w2 vq%dx%d\n",L,d,n,h,wd,wn);
    return 1;
}
static int vq_w2_nc(void){ const char*e=getenv("DS4_VQ_W2_NC"); int v=e?atoi(e):256; if(v<16||v>4096) v=256; return v; }   /* 同上: 禁 static 缓存, 计划表逐层改档 */
static int hot_from_anchor(int L,int S,int K);   /* 前置声明: 回放自动 arm 用(定义在贪心段) */
static void quant_apply(char cfg,const float*e1,const float*e3,const float*e2,
                        const float*Xc,int ncal,const float*xs,const float*wwv,int nt,
                        const int*toks,float*dst,int L,int e){
    float *q1,*q3,*q2;
    const float *X=(cfg=='n')?NULL:Xc; int nx=(cfg=='n')?0:ncal;
    int g2hot=g2_hot_slot(L,e)>=0;   /* 热专家: go2b 合并2bit(残差+量化一体), 档位不适用 */
    if(g2hot){ q1=dq_quant_expert_go2b(e1,MOEI,DIM,X,nx); q3=dq_quant_expert_go2b(e3,MOEI,DIM,X,nx); }
    else if(cfg=='z'){ q1=dq_quant_expert_lowrank(e1,MOEI,DIM,X,nx,ZK); q3=dq_quant_expert_lowrank(e3,MOEI,DIM,X,nx,ZK); }
    else if(cfg=='r'){ q1=dq_quant_expert_rowscale(e1,MOEI,DIM,X,nx); q3=dq_quant_expert_rowscale(e3,MOEI,DIM,X,nx); }
    else if(cfg=='g'){ q1=dq_quant_expert_signref(e1,MOEI,DIM,X,nx); q3=dq_quant_expert_signref(e3,MOEI,DIM,X,nx); }
    else { int lv=(cfg=='2')?1:(cfg=='3')?2:0;
           q1=dq_quant_expert_resid(e1,MOEI,DIM,X,nx,lv); q3=dq_quant_expert_resid(e3,MOEI,DIM,X,nx,lv); }
    float *hc=NULL; int nhc=0;
    if(nx>0){
        hc=malloc((size_t)nx*MOEI*4);
        float *gg=malloc((size_t)nx*MOEI*4),*uu=malloc((size_t)nx*MOEI*4);
        dq_matmul(X,q1,gg,nx,DIM,MOEI); dq_matmul(X,q3,uu,nx,DIM,MOEI);
        for(size_t i=0;i<(size_t)nx*MOEI;i++){ float g2=gg[i],u2=uu[i];
            if(SWLIM>0){ if(u2>SWLIM)u2=SWLIM; if(u2<-SWLIM)u2=-SWLIM; if(g2>SWLIM)g2=SWLIM; }
            hc[i]=dq_silu(g2)*u2; }
        free(gg);free(uu); nhc=nx;
    }
    if(g2hot) q2=dq_quant_expert_go2b(e2,DIM,MOEI,hc,nhc);
    else if(cfg=='z') q2=dq_quant_expert_lowrank(e2,DIM,MOEI,hc,nhc,ZK);
    else if(cfg=='r') q2=dq_quant_expert_rowscale(e2,DIM,MOEI,hc,nhc);
    else if(cfg=='g') q2=dq_quant_expert_signref(e2,DIM,MOEI,hc,nhc);
    else { int lv=(cfg=='2')?1:(cfg=='3')?2:0; q2=dq_quant_expert_resid(e2,DIM,MOEI,hc,nhc,lv); }
    if(hc)free(hc);
    float *aq=calloc((size_t)nt*DIM,4);
    dq_expert_fp(xs,q1,q3,q2,wwv,aq,nt,DIM,MOEI,SWLIM);
    for(int i=0;i<nt;i++)for(int d=0;d<DIM;d++) dst[(size_t)toks[i]*DIM+d]+=aq[(size_t)i*DIM+d];
    free(q1);free(q3);free(q2);free(aq);
}

/* 并行专家 worker: 原子计数器分发 e, 各线程 scatter 进私有 partial buffer, 主线程归约。
 * st_read/量化核心无全局可变状态(已核), 线程安全。 */
typedef struct {
    int L,S,n_fit,do_quant; char cfg;
    const float *Fin; const int *idx; const float *rw_act;
    int *e_next;
    float *fout;        /* [S*DIM] 选定档 routed 累积 (私有) */
    float *ffp;         /* [S*DIM] FP routed (局部质量对照; NULL=不算) */
    float *abl[4];      /* 消融档 n/1/z/2 (DS4_ABLATE; NULL=不算) */
    double calib_rows; int calib_empty,nhit;
} ework_t;

static void *expert_worker(void*arg){
    ework_t *w=(ework_t*)arg;
    int S=w->S, L=w->L, n_fit=w->n_fit;
    int *toks=malloc((size_t)S*sizeof(int)); float *wwv=malloc((size_t)S*sizeof(float));
    float *xs=malloc((size_t)S*DIM*4);
    float *Xc=(w->do_quant&&ANC_OK)?malloc((size_t)(n_fit>0?n_fit:1)*DIM*4):NULL;
    for(;;){
        int e=__sync_fetch_and_add(w->e_next,1); if(e>=NEXP) break;
        int nt=0;
        for(int s=0;s<S;s++)for(int a=0;a<NACT_RT;a++)
            if(w->idx[(size_t)s*NACT_RT+a]==e){toks[nt]=s;wwv[nt]=w->rw_act[(size_t)s*NACT_RT+a];nt++;}
        if(!nt) continue;
        if((e&31)==0) fputc('.',stderr);
        char n1[160],n3[160],n2[160];
        snprintf(n1,sizeof(n1),"layers.%d.ffn.experts.%d.w1.weight",L,e);
        snprintf(n3,sizeof(n3),"layers.%d.ffn.experts.%d.w3.weight",L,e);
        snprintf(n2,sizeof(n2),"layers.%d.ffn.experts.%d.w2.weight",L,e);
        long rr,cc; float *e1=st_read_weight(&C,n1,&rr,&cc),*e3=st_read_weight(&C,n3,&rr,&cc),*e2=st_read_weight(&C,n2,&rr,&cc);
        if(!e1||!e3||!e2){ fprintf(stderr,"\n[!] L%d e%d 权重读失败\n",L,e); if(e1)free(e1); if(e3)free(e3); if(e2)free(e2); continue; }
        for(int i=0;i<nt;i++) memcpy(xs+(size_t)i*DIM, w->Fin+(size_t)toks[i]*DIM, (size_t)DIM*4);
        if(!w->do_quant || w->cfg=='F'){
            float *af=calloc((size_t)nt*DIM,4); dq_expert_fp(xs,e1,e3,e2,wwv,af,nt,DIM,MOEI,SWLIM);
            for(int i=0;i<nt;i++)for(int d=0;d<DIM;d++) w->fout[(size_t)toks[i]*DIM+d]+=af[(size_t)i*DIM+d];
            free(af);
        } else {
            if(w->ffp){
                float *af=calloc((size_t)nt*DIM,4); dq_expert_fp(xs,e1,e3,e2,wwv,af,nt,DIM,MOEI,SWLIM);
                for(int i=0;i<nt;i++)for(int d=0;d<DIM;d++) w->ffp[(size_t)toks[i]*DIM+d]+=af[(size_t)i*DIM+d];
                free(af);
            }
            /* anchor 校准行: FP 激活 + FP 路由 + 只 fit 行 (held 行不喂校准, 防泄漏) */
            int ncal=0;
            if(ANC_OK&&Xc){
                const float *afin=ANC.fin+(size_t)L*S*DIM;
                const int32_t *aidx=ANC.ridx+(size_t)L*S*NACT;
                /* DS4_CALIB_FULLSET(2026-07-27 g_r 饿死审计): 命中过滤在 n_fit~400 时每专家
                 * 仅~9行 → GPTQ-H 是 rank-9 残料, g_r 被 lam=sp2/(n+1) 钉死在 1(实测全模型
                 * g_r≈1.000)。全集喂入(ncal=n_fit)统计充分, lam 公式随 n 自愈。默认保持旧行为。 */
                static int fullset=-1; if(fullset<0) fullset=getenv("DS4_CALIB_FULLSET")?1:0;
                /* DS4_CALIB_CAP(2026-08-18 用户问"语料需要那么多吗"): FULLSET 每专家全集
                 * 2906 行是对"9行饿死"的矫枉过正 — H 只有 128 维, 512 行=4×过采样已统计
                 * 充分, 而 GPTQ-H/g_r 两个最重段都 ∝ 行数。激活行全保, 全集行均匀 stride
                 * 补到上限。0/未设=原全集行为。 */
                static int calcap=-1; if(calcap<0){ const char*cc=getenv("DS4_CALIB_CAP"); calcap=cc?atoi(cc):0; }
                if(fullset&&calcap>0&&n_fit>calcap){
                    int nact_e=0;   /* 先收激活行(全保) */
                    for(int s=0;s<n_fit;s++){ int hit=0;
                        for(int a=0;a<NACT;a++) if(aidx[(size_t)s*NACT+a]==e){hit=1;break;}
                        if(hit){ memcpy(Xc+(size_t)ncal*DIM, afin+(size_t)s*DIM, (size_t)DIM*4); ncal++; nact_e++; } }
                    int need=calcap-nact_e;
                    if(need>0){ int stride=n_fit/need; if(stride<1)stride=1;
                        for(int s=0;s<n_fit&&need>0;s+=stride){ int hit=0;
                            for(int a=0;a<NACT;a++) if(aidx[(size_t)s*NACT+a]==e){hit=1;break;}
                            if(!hit){ memcpy(Xc+(size_t)ncal*DIM, afin+(size_t)s*DIM, (size_t)DIM*4); ncal++; need--; } } }
                } else
                for(int s=0;s<n_fit;s++){ int hit=fullset;
                    if(!hit) for(int a=0;a<NACT;a++) if(aidx[(size_t)s*NACT+a]==e){hit=1;break;}
                    if(hit){ memcpy(Xc+(size_t)ncal*DIM, afin+(size_t)s*DIM, (size_t)DIM*4); ncal++; } }
            }
            w->nhit++; w->calib_rows+=ncal; if(!ncal) w->calib_empty++;
            quant_apply(w->cfg,e1,e3,e2,ncal?Xc:NULL,ncal,xs,wwv,nt,toks,w->fout,L,e);
            if(w->abl[0]){ const char AC[4]={'n','1','z','2'};
                for(int ci=0;ci<4;ci++) quant_apply(AC[ci],e1,e3,e2,ncal?Xc:NULL,ncal,xs,wwv,nt,toks,w->abl[ci],L,e); }
        }
        free(e1);free(e3);free(e2);
    }
    free(toks);free(wwv);free(xs); if(Xc)free(Xc);
    return NULL;
}

/* ===================== 一层前向 ===================== *
 * 就地更新 H[S,HCM,DIM]。do_quant: routed 专家按 cfg 档量化(校准=anchor 部署口径), 误差随深度累积。
 * st(可NULL): 路由一致率/校准统计/局部质量。 */
typedef struct { double agree; double calib_rows; int calib_empty,nhit;
                 int have_loc; double loc_r2; double rs_ratio; int zk;
                 double abl_r2[4],abl_rel[2],abl_la,abl_lc; } lstat_t;

/* ===================== 每层修正算法搜索 + 合并动态侧车 (DS4_COADAPT) ===================== *
 * 用户设计(2026-07-10 定稿口径): 每层的任务是【找到那一层最好的算法】——不存在"没提升
 * 就丢弃/判0"的场景; 某候选无增益只说明还没找到, 换形态继续搜; 每层胜者可以不同;
 * 最后把所有层的动态修正【合并生成一个很小体积的文件】(z+四损失+向后+感知)叠在 1bit 底座上。
 *
 * 实现: base 量化一次(signref, anchor 校准)+过程数据缓存(hc_cal/hc_all/y2ref/yq_all/命中行),
 * 之后每个修正候选都在缓存上闭式求解(秒级), 逐层搜索菜单:
 *   GL   静态层标量 g_L(1 dof, 零方差, 必然非负基准)      —— 目标=累积漂移 DF
 *   GE   静态每专家增益 g_e(256 dof, 低方差)               —— 目标=自误差 / 累积 两版
 *   CE   动态每专家系数 c_e(x)=1+αᵀk(x) (kernel-ridge, λ 扫) —— 目标=自误差 / 累积 两版
 *   ZL   加法低秩 z^L(Fin→DF, 四损失求解+选秩, k/λ 扫)      —— 累积回拉
 *   组合  乘法胜者之上再解加法余量
 * 四损失=评审与正则(L_align+0.5·L_classify 评分, λ=L_fixed, clamp=L_smooth);
 * 感知=classify 维度方差加权; 向后=胜者落地后 H 传下层 + 1 轮 base 重解(D3)择优。
 * val(fit 内留出)=选择判据, held 只观测; 每层落【菜单内 val 最优】(GL 兜底, 永不空手)。
 * 胜者 payload 记入合并侧车(DQZ1): 逐层 {algo,params,数据}, 结束写盘并报体积。 */
typedef struct {
    int ncal,nhit;
    float *hc_cal;   /* [ncal×MOEI] anchor FP 校准行过量化 q1/q3 的 hidden(顺序补偿口径, 不缩放) */
    int   *calS;     /* [ncal] 校准行行号 */
    float *calW;     /* [ncal] anchor 路由权重 */
    float *hc_all;   /* [nhit×MOEI] 实际激活行(污染 Fin)过 q1/q3 */
    int   *hitS; float *hitW;
    float *y2ref;    /* [ncal×DIM] e2fp·hc_cal — w2 顺序补偿的 FP 目标(乘法系数的参照) */
    float *yq_all;   /* [nhit×DIM] 当前 w2q·hc_all — 应用行输出(系数在外面乘) */
    float *yq_cal;   /* [ncal×DIM] 当前 w2q·hc_cal — 校准行输出(累积目标候选用) */
    float *gcal;     /* [ncal] 每校准行自误差最优标量 ⟨y2ref,ŷq⟩/⟨ŷq,ŷq⟩ */
    float *gden;     /* [ncal] ⟨ŷq,ŷq⟩ (静态增益池化/累积目标的分母) */
    float *ccal;     /* [ncal] 胜者系数在校准行的值(向后重解目标用); NULL=c≡1 */
    float *alpha;    /* [ncal] z_e kernel 权(dual): c(x)=1+Σ_t α_t⟨x_t,x⟩ */
    float gstat;     /* 静态每专家增益候选(GE); 0=关, 非0 时优先于 alpha */
} coexp_t;
typedef struct {
    int L,S,n_fit,iter;
    const float *Fin; const int *idx; const float *rw;
    coexp_t *ex; int *e_next;
    float *fout;
    double calib_rows; int calib_empty,nhit_experts; long nflip;
} cowork_t;
static void co_hc(const float*X,int n,const float*q1,const float*q3,float*hc){
    float *gg=malloc((size_t)n*MOEI*4),*uu=malloc((size_t)n*MOEI*4);
    dq_matmul(X,q1,gg,n,DIM,MOEI); dq_matmul(X,q3,uu,n,DIM,MOEI);
    for(size_t i=0;i<(size_t)n*MOEI;i++){ float g2=gg[i],u2=uu[i];
        if(SWLIM>0){ if(u2>SWLIM)u2=SWLIM; if(u2<-SWLIM)u2=-SWLIM; if(g2>SWLIM)g2=SWLIM; }
        hc[i]=dq_silu(g2)*u2; }
    free(gg);free(uu);
}
/* w2 解完的收尾: 应用行输出 yq_all(存, 系数在主循环外乘)、c=1 scatter 进 fout(基准)、
 * 每校准行最优标量 gcal = ⟨y2ref,ŷq⟩/⟨ŷq,ŷq⟩ (z_e 的拟合目标; ŷq 用不缩放 hc_cal)。 */
static void co_w2_finish(coexp_t*ce,const float*w2q,float*fout){
    if(ce->nhit>0){
        if(!ce->yq_all) ce->yq_all=malloc((size_t)ce->nhit*DIM*4);
        dq_matmul(ce->hc_all,w2q,ce->yq_all,ce->nhit,MOEI,DIM);
        for(int i=0;i<ce->nhit;i++){ float wgt=ce->hitW[i]; float*dst=fout+(size_t)ce->hitS[i]*DIM;
            const float*yi=ce->yq_all+(size_t)i*DIM;
            for(int d2=0;d2<DIM;d2++) dst[d2]+=wgt*yi[d2]; }
    }
    if(ce->ncal>0&&ce->y2ref){
        if(!ce->gcal) ce->gcal=malloc((size_t)ce->ncal*4);
        if(!ce->gden) ce->gden=malloc((size_t)ce->ncal*4);
        if(!ce->yq_cal) ce->yq_cal=malloc((size_t)ce->ncal*DIM*4);
        dq_matmul(ce->hc_cal,w2q,ce->yq_cal,ce->ncal,MOEI,DIM);
        for(int t=0;t<ce->ncal;t++){
            const float*a=ce->y2ref+(size_t)t*DIM,*b=ce->yq_cal+(size_t)t*DIM;
            double num=0,den=0;
            for(int d2=0;d2<DIM;d2++){ num+=(double)a[d2]*b[d2]; den+=(double)b[d2]*b[d2]; }
            float g=(den>1e-20)?(float)(num/den):1.0f;
            if(g<0.25f)g=0.25f; if(g>4.0f)g=4.0f;   /* 系数信赖域(重设计文档实测 g 深层≤4.5) */
            ce->gcal[t]=g; ce->gden[t]=(float)den;
        }
    }
}
static void *coadapt_worker(void*a){
    cowork_t *w=a;
    /* ★累加前层★: 校准行取自当前累积激活 + 实际路由(部署真值), 非 FP 锚 */
    const float *afin=w->Fin;
    const int   *aidx=w->idx;
    const float *arw=w->rw;
    for(;;){ int e=__sync_fetch_and_add(w->e_next,1); if(e>=NEXP)break;
        if((e&31)==0){ char pg[24]; snprintf(pg,sizeof(pg),"%d/%d",e,NEXP);
            mlog(w->L,w->iter?"向后":"量化",w->iter?"w2重解(系数校正目标)":"signref",pg,
                 (uint64_t)NEXP*(2*(uint64_t)MOEI*go1b_blk_row_bytes(DIM)+(uint64_t)DIM*go1b_blk_row_bytes(MOEI)),
                 "待测","进行中"); }
        coexp_t *ce=&w->ex[e];
        char n2[160]; long rr,cc;
        snprintf(n2,sizeof(n2),"layers.%d.ffn.experts.%d.w2.weight",w->L,e);
        if(w->iter==0){
            char n1[160],n3[160];
            snprintf(n1,sizeof(n1),"layers.%d.ffn.experts.%d.w1.weight",w->L,e);
            snprintf(n3,sizeof(n3),"layers.%d.ffn.experts.%d.w3.weight",w->L,e);
            const uint64_t *bfvt0=(bf_from_bytes()&&BFB_VQMAP)?(const uint64_t*)(BFB_VQMAP+16):NULL;
            int bfsk13=(bfvt0&&g2_hot_slot(w->L,e)<0
                        &&bfvt0[(size_t)e*3]&&bfvt0[(size_t)e*3+1]);   /* ★字节起步冷专家: e1/e3 免读免转(~16GB/层);
                        * 盘上字节=量化段导出解, 已含同α GPTAQ 目标(export 侧 yadjE), 本处 yadj 块一并跳过 */
            float *e1=bfsk13?NULL:st_read_weight(&C,n1,&rr,&cc);
            float *e3=bfsk13?NULL:st_read_weight(&C,n3,&rr,&cc);
            float *e2=st_read_weight(&C,n2,&rr,&cc);
            if((!bfsk13&&(!e1||!e3))||!e2){ if(e1)free(e1);if(e3)free(e3);if(e2)free(e2);
                fprintf(stderr,"\n[!] L%d e%d 读失败\n",w->L,e); continue; }
            int ncal=0;
            for(int s2=0;s2<w->n_fit;s2++)for(int a2=0;a2<NACT;a2++)
                if(aidx[(size_t)s2*NACT+a2]==e){ncal++;break;}
            float *Xc=NULL;
            if(ncal){ Xc=malloc((size_t)ncal*DIM*4);
                ce->calS=malloc((size_t)ncal*sizeof(int)); ce->calW=malloc((size_t)ncal*4);
                int t=0;
                for(int s2=0;s2<w->n_fit;s2++)for(int a2=0;a2<NACT;a2++)
                    if(aidx[(size_t)s2*NACT+a2]==e){
                        memcpy(Xc+(size_t)t*DIM,afin+(size_t)s2*DIM,(size_t)DIM*4);
                        ce->calS[t]=s2; ce->calW[t]=arw[(size_t)s2*NACT+a2]; t++; break; }
            } else w->calib_empty++;
            ce->ncal=ncal; w->calib_rows+=ncal; w->nhit_experts++;
            int nhit=0;
            for(int s2=0;s2<w->S;s2++)for(int a2=0;a2<NACT_RT;a2++)
                if(w->idx[(size_t)s2*NACT_RT+a2]==e){nhit++;break;}
            ce->nhit=nhit;
            int g2hot=g2_hot_slot(w->L,e)>=0;   /* 热专家: go2b 合并态量化(残差+量化一体) */
            /* GPTAQ 非对称目标(2026-07-25 G1b): y_ref=W·x̂+α·W·(x̃−x̂); x̃=锚 FP 流(量化全程不被覆盖)。
             * 回归仍在漂移输入 x̂ 上(合并态), 只有目标含 FP 流 ⇒ 纠正累积上游漂移而非仅条件于它。 */
            float *yadj1=NULL,*yadj3=NULL; float tga=dq_tgt_alpha();
            if(!bfsk13&&tga>0.0f&&ncal&&ANC_OK){
                float *dx=malloc((size_t)ncal*DIM*4);
                const float *afp=ANC.fin+(size_t)w->L*w->S*DIM;
                for(int t=0;t<ncal;t++){ int s2=ce->calS[t];
                    const float *xf=afp+(size_t)s2*DIM,*xh=afin+(size_t)s2*DIM;
                    for(int d=0;d<DIM;d++) dx[(size_t)t*DIM+d]=xf[d]-xh[d]; }
                yadj1=malloc((size_t)ncal*MOEI*4); yadj3=malloc((size_t)ncal*MOEI*4);
                dq_matmul(dx,e1,yadj1,ncal,DIM,MOEI); dq_matmul(dx,e3,yadj3,ncal,DIM,MOEI);
                for(size_t i2=0;i2<(size_t)ncal*MOEI;i2++){ yadj1[i2]*=tga; yadj3[i2]*=tga; }
                free(dx);
            }
            float *q1,*q3;
            if(bfsk13){   /* ★字节起步: 冷 w1/w3=盘上 VQ 载荷 dequant */
                q1=malloc((size_t)MOEI*DIM*4); q3=malloc((size_t)MOEI*DIM*4);
                vq_unpack_dequant(BFB_VQMAP+bfvt0[(size_t)e*3],BFB_VQMSZ-bfvt0[(size_t)e*3],q1,NULL,NULL);
                vq_unpack_dequant(BFB_VQMAP+bfvt0[(size_t)e*3+1],BFB_VQMSZ-bfvt0[(size_t)e*3+1],q3,NULL,NULL);
            } else if(dq_vq_on()){   /* v2.2: 码本量化(α 目标暂不进 VQ 内环, 行乘子已含激活拟合) */
                q1=g2hot?dq_quant_expert_vq(e1,MOEI,DIM,Xc,ncal,4,512):dq_quant_expert_vq(e1,MOEI,DIM,Xc,ncal,vq_cold_dim(),vq_cold_nc());
                q3=g2hot?dq_quant_expert_vq(e3,MOEI,DIM,Xc,ncal,4,512):dq_quant_expert_vq(e3,MOEI,DIM,Xc,ncal,vq_cold_dim(),vq_cold_nc());
            } else {
                q1=g2hot?dq_quant_expert_go2b_adj(e1,MOEI,DIM,Xc,ncal,yadj1):dq_quant_expert_signref_adj(e1,MOEI,DIM,Xc,ncal,yadj1,NULL);
                q3=g2hot?dq_quant_expert_go2b_adj(e3,MOEI,DIM,Xc,ncal,yadj3):dq_quant_expert_signref_adj(e3,MOEI,DIM,Xc,ncal,yadj3,NULL);
            }
            if(yadj1){free(yadj1);yadj1=NULL;} if(yadj3){free(yadj3);yadj3=NULL;}
            if(ncal){ ce->hc_cal=malloc((size_t)ncal*MOEI*4); co_hc(Xc,ncal,q1,q3,ce->hc_cal); }
            if(nhit){
                ce->hitS=malloc((size_t)nhit*sizeof(int)); ce->hitW=malloc((size_t)nhit*4);
                float *Xa=malloc((size_t)nhit*DIM*4); int t=0;
                for(int s2=0;s2<w->S;s2++)for(int a2=0;a2<NACT_RT;a2++)
                    if(w->idx[(size_t)s2*NACT_RT+a2]==e){
                        memcpy(Xa+(size_t)t*DIM,w->Fin+(size_t)s2*DIM,(size_t)DIM*4);
                        ce->hitS[t]=s2; ce->hitW[t]=w->rw[(size_t)s2*NACT_RT+a2]; t++; break; }
                ce->hc_all=malloc((size_t)nhit*MOEI*4); co_hc(Xa,nhit,q1,q3,ce->hc_all);
                free(Xa);
            }
            free(q1);free(q3);free(e1);free(e3);
            if(ce->ncal){   /* w2 顺序补偿 FP 目标(乘法系数的参照, 只算一次) */
                ce->y2ref=malloc((size_t)ce->ncal*DIM*4);
                dq_matmul(ce->hc_cal,e2,ce->y2ref,ce->ncal,MOEI,DIM);
            }
            float *w2q;
            if(bf_from_bytes()&&BFB_W2&&!g2hot){   /* ★字节起步: 冷 w2 基座=盘上 D 段 go1b(同 μ10 FULLSET 解); D3 后续照常重解 */
                w2q=malloc((size_t)DIM*MOEI*4);
                dq_go1b_bytes_dequant(BFB_W2+(size_t)e*BFB_SZD,DIM,MOEI,w2q);
            }
            else if(dq_vq_on()&&g2hot) w2q=dq_quant_expert_vq(e2,DIM,MOEI,ce->hc_cal,ce->ncal,4,512);
            /* ★冷 w2 评估路径带顺序补偿(2026-08-03): 与导出路径(vq_export_matrix_seq)同解,
             * 搜索/调优/z 全链看到的就是部署态 — 失配事故修 */
            else if(dq_vq_on()&&vq_w2_dim()>0)   /* R28: 冷 w2 码本(省 6.2G 给 w13; 冠军此处是 signref) */
                w2q=dq_quant_expert_vq_seq(e2,DIM,MOEI,ce->hc_cal,ce->ncal,vq_w2_dim(),vq_w2_nc(),1);
            else w2q=g2hot?dq_quant_expert_go2b(e2,DIM,MOEI,ce->hc_cal,ce->ncal)
                            :dq_quant_expert_signref_adj(e2,DIM,MOEI,ce->hc_cal,ce->ncal,NULL,&w->nflip);
            co_w2_finish(ce,w2q,w->fout);
            free(w2q);free(e2);free(Xc);
        } else {
            if(ce->nhit<1&&ce->ncal<1) continue;
            if(g2_hot_slot(w->L,e)>=0){
                /* 热专家 base=go2b 冻结: D3 不重解 w2(基座不随系数目标漂), 输出直接用缓存
                 * yq_all scatter(与重算逐字节同; gcal/yq_cal 也不变)。动态系数照常在外层拟合。 */
                if(ce->yq_all) for(int i=0;i<ce->nhit;i++){ float wgt=ce->hitW[i];
                    float*dst=w->fout+(size_t)ce->hitS[i]*DIM; const float*yi=ce->yq_all+(size_t)i*DIM;
                    for(int d2=0;d2<DIM;d2++) dst[d2]+=wgt*yi[d2]; }
                continue;
            }
            float *e2=st_read_weight(&C,n2,&rr,&cc); if(!e2)continue;
            /* D3 交替: sign/scale 重选目标 = 动态系数校正后残差。
             * 乘法系数 c_t 进模型侧: 缩放校准行 x'_t=c_t·hc_t (→ 模型项 c_t·s·P_t),
             * Yadj=(1−c_t)·y2ref 把内部目标 w·x'=c_t·y2ref 平移回 y2ref → 精确闭式。 */
            float *Xs=NULL,*Yadj=NULL;
            if(ce->ncal&&ce->ccal){
                Xs=malloc((size_t)ce->ncal*MOEI*4);
                Yadj=malloc((size_t)ce->ncal*DIM*4);
                for(int t=0;t<ce->ncal;t++){ float c=ce->ccal[t];
                    const float*hs=ce->hc_cal+(size_t)t*MOEI; float*xd=Xs+(size_t)t*MOEI;
                    for(int j=0;j<MOEI;j++) xd[j]=c*hs[j];
                    const float*yr=ce->y2ref+(size_t)t*DIM; float*ya=Yadj+(size_t)t*DIM;
                    float oc=1.0f-c;
                    for(int d2=0;d2<DIM;d2++) ya[d2]=oc*yr[d2];
                }
            }
            float *w2q;
            w2q=dq_quant_expert_signref_adj(e2,DIM,MOEI,Xs?Xs:ce->hc_cal,ce->ncal,Yadj,&w->nflip);
            co_w2_finish(ce,w2q,w->fout);
            free(w2q);free(e2);if(Xs)free(Xs);if(Yadj)free(Yadj);
        }
    }
    return NULL;
}
/* z_e 逐专家闭式条件系数(D2, 设计本义"全层动态系数"): 校准行标量目标 gt[t] → dual ridge
 * kernel 解 α=(K+λI)⁻¹(gt−1), K_ij=⟨x_i,x_j⟩(x=anchor Fin 行)。c(x)=1+Σα_t⟨x_t,x⟩,
 * 信赖域 clamp [0.25,4]。填 alpha + 校准行系数 ccal(向后重解目标用)。 */
static void co_fit_ze(coexp_t*ce,const float*afin,const float*gt,double lam,int vs){
    int n=ce->ncal; if(n<1||!gt) return;
    if(!ce->alpha) ce->alpha=malloc((size_t)n*4);
    if(!ce->ccal)  ce->ccal =malloc((size_t)n*4);
    for(int i=0;i<n;i++) ce->alpha[i]=0.0f;
    /* 防泄漏: 只用 calS<vs 的行拟合(val 行只当裁判) */
    int *sub=malloc((size_t)n*sizeof(int)),m=0;
    for(int i=0;i<n;i++) if(ce->calS[i]<vs) sub[m++]=i;
    if(m<1){ free(sub); if(ce->ccal) for(int i=0;i<n;i++) ce->ccal[i]=1.0f; return; }
    double *K=malloc((size_t)m*m*sizeof(double)),*b=malloc((size_t)m*sizeof(double));
    double tr=0;
    for(int i=0;i<m;i++){ const float*xi=afin+(size_t)ce->calS[sub[i]]*DIM;
        for(int j=i;j<m;j++){ const float*xj=afin+(size_t)ce->calS[sub[j]]*DIM;
            double s=0; for(int d2=0;d2<DIM;d2++) s+=(double)xi[d2]*xj[d2];
            K[(size_t)i*m+j]=s; K[(size_t)j*m+i]=s; }
        tr+=K[(size_t)i*m+i]; }
    double ridge=lam*(tr/(double)m)+1e-8;
    for(int i=0;i<m;i++) K[(size_t)i*m+i]+=ridge;
    for(int i=0;i<m;i++) b[i]=(double)gt[sub[i]]-1.0;
    if(cholesky(K,(uint32_t)m)==0){
        cholesky_solve(K,(uint32_t)m,b);
        for(int i=0;i<m;i++) ce->alpha[sub[i]]=(float)b[i];
    }
    /* 全部校准行的系数(向后重解目标): c=1+Σ_j α_j⟨x_j,x_i⟩ */
    for(int i=0;i<n;i++){ const float*xi=afin+(size_t)ce->calS[i]*DIM; double s=1.0;
        for(int j=0;j<m;j++){ const float*xj=afin+(size_t)ce->calS[sub[j]]*DIM;
            double d=0; for(int d2=0;d2<DIM;d2++) d+=(double)xi[d2]*xj[d2];
            s+=ce->alpha[sub[j]]*d; }
        if(s<0.25)s=0.25; if(s>4.0)s=4.0; ce->ccal[i]=(float)s; }
    free(K);free(b);free(sub);
}
/* 乘法应用: 各 ce 的 alpha(动态) 或 gstat(静态) → Ftest = shared + Σ w·c·ŷ */
static void co_apply_mult(coexp_t*ex,const float*afin,const float*Fin,const float*shared,
                          int S,float*Ftest,double*cmean,double*cmx){
    memcpy(Ftest,shared,(size_t)S*DIM*4);
    double csum=0,cm=0; long cc=0;
    for(int e=0;e<NEXP;e++){ coexp_t*ce=&ex[e];
        if(ce->nhit<1||!ce->yq_all) continue;
        for(int i=0;i<ce->nhit;i++){
            double s=1.0;
            if(ce->gstat!=0.0f) s=ce->gstat;
            else if(ce->alpha&&ce->ncal){
                const float*xh=Fin+(size_t)ce->hitS[i]*DIM;
                for(int t=0;t<ce->ncal;t++){ if(ce->alpha[t]==0.0f)continue;
                    const float*xc=afin+(size_t)ce->calS[t]*DIM;
                    double d=0; for(int d2=0;d2<DIM;d2++) d+=(double)xc[d2]*xh[d2];
                    s+=ce->alpha[t]*d; }
            }
            if(s<0.25)s=0.25; if(s>4.0)s=4.0;
            double ac=fabs(s-1.0); csum+=ac; if(ac>cm)cm=ac; cc++;
            float wc=ce->hitW[i]*(float)s; float*dst=Ftest+(size_t)ce->hitS[i]*DIM;
            const float*yi=ce->yq_all+(size_t)i*DIM;
            for(int d2=0;d2<DIM;d2++) dst[d2]+=wc*yi[d2];
        }
    }
    if(cmean)*cmean=cc?csum/(double)cc:0; if(cmx)*cmx=cm;
}
static void co_rel(const float*Hq,const float*Hf,int a,int b,size_t rowsz,double*rel);
static double co_score(const float*Hq,const float*Hf,int a,int b,size_t rowsz);
/* 出口评分: fit/val/held 三段 relL2 + val 四损失评分(align+0.5·classify=感知加权) */
static void co_eval(const float*Ftest,const float*H2,const float*post2,const float*comb2,
                    const float*Hf,float*Hq,int S,int vs,int n_fit,size_t rowsz,
                    double*fit,double*val,double*held,double*sc){
    dq_hc_post(Ftest,H2,post2,comb2,Hq,S,HCM,DIM);
    co_rel(Hq,Hf,0,vs,rowsz,fit);
    co_rel(Hq,Hf,vs,n_fit,rowsz,val);
    co_rel(Hq,Hf,n_fit,S,rowsz,held);
    *sc = (n_fit-vs)>0 ? co_score(Hq,Hf,vs,n_fit,rowsz) : co_score(Hq,Hf,0,vs,rowsz);
}
/* 累积回拉目标 DF: hc_post 对 Fout 线性 → Fout 空间闭式反解到本层为止的漂移 */
static void co_df(const float*Fcur,const float*H2,const float*post2,const float*comb2,
                  const float*Hf,float*Hq,float*DF,int S){
    dq_hc_post(Fcur,H2,post2,comb2,Hq,S,HCM,DIM);
    for(int s=0;s<S;s++){
        const float *ps=post2+(size_t)s*HCM; double pd=1e-12;
        for(int j=0;j<HCM;j++) pd+=(double)ps[j]*ps[j];
        for(int d2=0;d2<DIM;d2++){ double a2=0;
            for(int j=0;j<HCM;j++) a2+=(double)ps[j]*((double)Hf[((size_t)s*HCM+j)*DIM+d2]-(double)Hq[((size_t)s*HCM+j)*DIM+d2]);
            DF[(size_t)s*DIM+d2]=(float)(a2/pd); }
    }
}
/* ===== 合并动态侧车(最终产物): 逐层胜者 {algo,params,payload}, 结束一次性落盘 ===== */
typedef struct { int L; char algo[12]; float lam,g; int k,backward;
                 float valb,vala,heldb,helda; uint8_t*pay; uint64_t paysz; } zrec_t;
static zrec_t ZREC[NL]; static int NZREC=0;
static void zfile_write(void){
    if(!NZREC) return;
    const char*p=getenv("DS4_ZFILE"); if(!p)p="/tmp/ds4quant_zfile.bin";
    FILE*f=fopen(p,"wb"); if(!f){ fprintf(stderr,"[zfile] 写 %s 失败\n",p); return; }
    uint32_t magic=0x315A5144; uint32_t nr=(uint32_t)NZREC;   /* "DQZ1" */
    fwrite(&magic,4,1,f); fwrite(&nr,4,1,f);
    uint64_t tot=8;
    for(int i=0;i<NZREC;i++){ zrec_t*r=&ZREC[i];
        fwrite(&r->L,4,1,f); fwrite(r->algo,1,12,f); fwrite(&r->lam,4,1,f);
        fwrite(&r->g,4,1,f); fwrite(&r->k,4,1,f); fwrite(&r->backward,4,1,f);
        fwrite(&r->paysz,8,1,f);
        if(r->paysz&&r->pay) fwrite(r->pay,1,r->paysz,f);
        tot+=40+r->paysz;
    }
    fclose(f);
    printf("ZFILE %s layers=%d bytes=%llu (%.2f MB)\n",p,NZREC,(unsigned long long)tot,(double)tot/1048576.0);
    fflush(stdout);
}

/* base 量化遍(worker 并行): iter=0 全量+缓存, iter=1 向后重解(w2 目标=系数校正后残差) */
static long co_base_pass(int L,int S,int n_fit,int iter,const float*Fin,const int*idx,const float*rw,
                         coexp_t*ex,float*routed,lstat_t*st){
    memset(routed,0,(size_t)S*DIM*4);
    int e_next=0,nth=NTHREADS<1?1:(NTHREADS>NEXP?NEXP:NTHREADS);
    cowork_t *ws=calloc((size_t)nth,sizeof(cowork_t));
    pthread_t *th=malloc((size_t)nth*sizeof(pthread_t));
    long nflip=0;
    for(int t=0;t<nth;t++){
        ws[t]=(cowork_t){L,S,n_fit,iter,Fin,idx,rw,ex,&e_next,calloc((size_t)S*DIM,4),0,0,0,0};
        pthread_create(&th[t],NULL,coadapt_worker,&ws[t]);
    }
    for(int t=0;t<nth;t++){ pthread_join(th[t],NULL);
        for(size_t i=0;i<(size_t)S*DIM;i++) routed[i]+=ws[t].fout[i];
        free(ws[t].fout); nflip+=ws[t].nflip;
        if(iter==0&&st){ st->calib_rows+=ws[t].calib_rows; st->calib_empty+=ws[t].calib_empty; st->nhit+=ws[t].nhit_experts; }
    }
    free(ws);free(th);
    return nflip;
}
/* 胜者 payload 打包(合并侧车用, fp16) */
static uint8_t *co_pay_mult_stat(coexp_t*ex,uint64_t*sz){   /* GE: g_e ×NEXP */
    *sz=(uint64_t)NEXP*2; uint8_t*p=malloc((size_t)*sz); uint16_t*h=(uint16_t*)p;
    for(int e=0;e<NEXP;e++) h[e]=go1b_fp32_to_fp16(ex[e].gstat!=0.0f?ex[e].gstat:1.0f);
    return p;
}
static uint8_t *co_pay_mult_dyn(coexp_t*ex,const float*afin,uint64_t*sz){   /* CE: v_e=Σα·x fp16[DIM]×NEXP */
    *sz=(uint64_t)NEXP*DIM*2; uint8_t*p=malloc((size_t)*sz); uint16_t*h=(uint16_t*)p;
    for(int e=0;e<NEXP;e++){ coexp_t*ce=&ex[e];
        for(int d2=0;d2<DIM;d2++){ double v=0;
            if(ce->alpha&&ce->ncal)
                for(int t=0;t<ce->ncal;t++){ if(ce->alpha[t]==0.0f)continue;
                    v+=(double)ce->alpha[t]*afin[(size_t)ce->calS[t]*DIM+d2]; }
            h[(size_t)e*DIM+d2]=go1b_fp32_to_fp16((float)v); } }
    return p;
}
static uint8_t *co_pay_zl(const ds4_z*zl,int k,uint64_t*sz){   /* ZL: V[d,k],U[d,k],z[k] fp16 */
    int d=(int)zl->d_in,dout=(int)zl->d_out;
    *sz=(uint64_t)k*(d+dout+1)*2; uint8_t*p=malloc((size_t)*sz); uint16_t*h=(uint16_t*)p; size_t o=0;
    for(int c=0;c<k;c++) for(int i=0;i<d;i++)    h[o++]=go1b_fp32_to_fp16(zl->V[(size_t)i*zl->rank+c]);
    for(int c=0;c<k;c++) for(int j=0;j<dout;j++) h[o++]=go1b_fp32_to_fp16(zl->U[(size_t)j*zl->rank+c]);
    for(int c=0;c<k;c++) h[o++]=go1b_fp32_to_fp16(zl->z[c]);
    return p;
}
static void co_rel(const float*Hq,const float*Hf,int a,int b,size_t rowsz,double*rel){
    double e2=0,x2=0;
    for(size_t i=(size_t)a*rowsz;i<(size_t)b*rowsz;i++){
        double dd=(double)Hq[i]-Hf[i]; e2+=dd*dd; x2+=(double)Hf[i]*Hf[i]; }
    *rel=sqrt(e2/(x2+1e-30));
}
static double co_score(const float*Hq,const float*Hf,int a,int b,size_t rowsz){
    int n=b-a; if(n<1) return 0.0;
    const float *A=Hq+(size_t)a*rowsz,*B=Hf+(size_t)a*rowsz;
    float *wv=malloc(rowsz*4); ds4_loss_dim_variance(B,(uint32_t)n,(uint32_t)rowsz,wv);
    double la=ds4_loss_align(A,B,(uint32_t)n,(uint32_t)rowsz);
    double lc=ds4_loss_classify(A,B,wv,(uint32_t)n,(uint32_t)rowsz);
    free(wv);
    return la+0.5*lc;
}
/* ===== 从层文件重前向(跨层反向的执行引擎): 文件即真相 =====
 * 读 dql_L<NN>.bin: 1bit 权重字节 + 落地修正记录(按时间序回放 GL/dyn2/dyn8/TREF)。
 * 不重量化 — 直接 dequant 字节 → 专家前向 → 修正链回放。 */
typedef struct { int type; float g,t; float w2p[4]; float w8[9]; float *V8; float *ge;
                 float *zlU,*zlV,*zlz; int zlk; float zltr;   /* 6=zl.RRR 冻结秩-k 方向修正(产物③) */
                 int zdin;   /* ★md86: 输入维 DIM=线性 | 3*DIM=ftA 特征提升(φ=[x,x²/rms,relu]) */
                 int erf_ne,erf_r; uint32_t *erf_eid; float *erf_tau; float *erf_UV;
                 /* 8=zl.ERF 死层部件(2026-08-12): 每专家 w2 低秩补丁 U[D,r](折S)·V[r,MOEI](折α)+token能量门τ */
                 size_t foff;     float ghc[2];   /* type7 GLhc: g_hot, g_cold(2026-08-06 冷热双通道) */
} lop_t;   /* 1=GL 2=dyn2 3=dyn8 4=TREF 5=GE(per-expert增益,累加时乘) 6=zl.RRR 8=zl.ERF; foff=载荷文件偏移(反修原地改写用) */
typedef struct { uint8_t *map; size_t msz; const uint8_t *w1,*w3,*w2; size_t szG,szD;
                 uint8_t *g2map; size_t g2msz;                                    /* go2b 侧车 mmap(可异机盘/NFS; 内嵌时 NULL, g2w* 指进主 map) */
                 const uint8_t *g2w1,*g2w3,*g2w2; int g2k; int16_t g2slot[256];   /* 热专家 2bit 覆盖 */
                 uint8_t *vqmap; size_t vqmsz;                                    /* v2.2 VQ 侧车 mmap(dql_vq_L%02d.bin) */
                 uint8_t *opsmap; size_t opsmsz; int has_ops;                     /* op 侧车 mmap(dql_ops_L%02d.bin, 平行架构权威源) */
                 lop_t ops[32]; int nops;
                 int zl_stub_at; } lfile_t;   /* zl.RRR 空壳占位序号(真载荷回填正位用; -1=无) */
/* go2b 侧车路径: DS4_GO2B_DIR(默认=层文件同目录)/dql_go2b_L<NN>.bin — 分储设计:
 * 冷 go1b dql 在本机(热专家稀疏洞), 热 go2b 侧车可放对机 NFS(两机 16G 盘都装不下合体) */
static void g2_sidecar_path(const char*dql_path,int L,char*out,size_t outsz){
    const char*gd=getenv("DS4_GO2B_DIR");
    if(gd){ snprintf(out,outsz,"%s/dql_go2b_L%02d.bin",gd,L); return; }
    char dir[512]; snprintf(dir,sizeof(dir),"%s",dql_path);
    char*sl=strrchr(dir,'/'); if(sl)*sl=0; else snprintf(dir,sizeof(dir),".");
    snprintf(out,outsz,"%s/dql_go2b_L%02d.bin",dir,L);
}
/* 侧车头: 'DQG2' u32 | ver u32 | L u32 | khot u32 | mean_cos f32 | rsv u32 | ids u16[khot] | pad8 → 载荷 */
#define G2SC_MAGIC 0x32475144u
static size_t g2_sidecar_hdr(int khot){ return (24+2*(size_t)khot+7)&~(size_t)7; }
/* ★op 侧车(2026-08-04 用户架构令: 量化模型与 z/动态平行, 反修不许动量化文件)★
 * dql_LXX.bin = 纯权重字节(1bit/g2hot), 量化后只读; 全部 op 记录(z 家族/zl.RRR/loss/bf/bwd)
 * 住 dql_ops_LXX.bin('DQO2' u32|L u32|nrec u32 + DQL2 同款记录), 应用序=侧车文件序(单一权威)。
 * 反修/sweep/回扫只写侧车 — 重量化在架构上不再可能被需要。旧混装 dql 兼容读(侧车缺席时)。 */
#define OPSC_MAGIC 0x324F5144u   /* 'DQO2' */
static void ops_sidecar_path(const char*dql_path,int L,char*out,size_t outsz){
    char dir[512]; snprintf(dir,sizeof(dir),"%s",dql_path);
    char*sl=strrchr(dir,'/'); if(sl)*sl=0; else snprintf(dir,sizeof(dir),".");
    snprintf(out,outsz,"%s/dql_ops_L%02d.bin",dir,L);
}
/* op 宿主文件 = dql 主文件(超冠 573b7f5 混装架构; 2026-08-08 用户令还原:
 * "侧车不是我要求加入的" — 平行架构 op 侧车 2026-08-04 系擅自引入, 已删)。 */
static void op_host_path(int L,char*out,size_t outsz){
    snprintf(out,outsz,"%s/dql_L%02d.bin",
        getenv("DS4_LAYER_DIR")?getenv("DS4_LAYER_DIR"):".",L);
}
static lfile_t BF_LFF; static int BF_LFF_ON=0;   /* 反修字节起步的层 mmap 持有者 */
/* op 记录解析(主文件旧混装 与 op 侧车 共用): curfoff=载荷在其宿主文件内的偏移 */
static void parse_op_rec(lfile_t*lf,const char*nm,const uint8_t*pay,uint64_t psz,size_t curfoff){
    if(lf->nops>=32) return;
    lop_t*o=&lf->ops[lf->nops];
    if(strstr(nm,"GLhc")&&psz>=8){ o->type=7; memcpy(o->ghc,pay,8); o->foff=curfoff; lf->nops++; }
    else if(strstr(nm,"GLdyn2")&&psz>=16){ o->type=2; memcpy(o->w2p,pay,16); o->foff=curfoff; lf->nops++; }
    else if(strstr(nm,"GLdyn8")&&psz>=36){ o->type=3; memcpy(o->w8,pay,36); o->foff=curfoff;
        if(psz>=36+(uint64_t)8*DIM*2){ o->V8=malloc((size_t)8*DIM*4);
            const uint16_t*h=(const uint16_t*)(pay+36);
            for(size_t j=0;j<(size_t)8*DIM;j++) o->V8[j]=go1b_fp16_to_fp32(h[j]); }
        lf->nops++; }
    else if(strstr(nm,"bf.GE")&&psz>=(uint64_t)NEXP*2){   /* per-expert 增益(fp16×NEXP), 累加时乘 */
        o->type=5; o->ge=malloc((size_t)NEXP*4);
        const uint16_t*h=(const uint16_t*)pay;
        for(int e=0;e<NEXP;e++) o->ge[e]=go1b_fp16_to_fp32(h[e]);
        o->foff=curfoff; lf->nops++; }
    else if(strstr(nm,"zl.RRR")&&psz>=16){   /* 冻结 z^L: u32 k|f32 tr|u32 din|u32 dout|fp16 z[k],U[dout·k],V[din·k] */
        uint32_t zk,din,dout; float ztr;
        memcpy(&zk,pay,4); memcpy(&ztr,pay+4,4); memcpy(&din,pay+8,4); memcpy(&dout,pay+12,4);
        if(zk>0&&zk<=1024&&(din==(uint32_t)DIM||din==3u*(uint32_t)DIM)&&dout==(uint32_t)DIM
           &&psz>=16+(uint64_t)2*(zk+(uint64_t)zk*din+(uint64_t)zk*dout)){
            o->type=6; o->zlk=(int)zk; o->zltr=ztr; o->zdin=(int)din;
            const uint16_t*h=(const uint16_t*)(pay+16);
            o->zlz=malloc((size_t)zk*4);
            for(uint32_t i=0;i<zk;i++) o->zlz[i]=go1b_fp16_to_fp32(h[i]);
            o->zlU=malloc((size_t)dout*zk*4);
            for(size_t i=0;i<(size_t)dout*zk;i++) o->zlU[i]=go1b_fp16_to_fp32(h[zk+i]);
            o->zlV=malloc((size_t)din*zk*4);
            for(size_t i=0;i<(size_t)din*zk;i++) o->zlV[i]=go1b_fp16_to_fp32(h[zk+(size_t)dout*zk+i]);
            o->foff=curfoff;
            /* 错序回填(旧混装兼容; 侧车模式 zl 在自然表位带载荷, 空壳不出现, 此段不触发):
             * 真载荷曾是 export 后 append 的 ⇒ 文件序在全部 op 之后, 而评估注入点在缩放族之前
             * (空壳占位序); 修正链非交换 → 回填到空壳位。 */
            { int at=-1;
              if(lf->zl_stub_at>=0&&lf->zl_stub_at<=lf->nops) at=lf->zl_stub_at;
              if(at>=0&&at<lf->nops){
                  lop_t tmp=*o;
                  memmove(&lf->ops[at+1],&lf->ops[at],(size_t)(lf->nops-at)*sizeof(lop_t));
                  lf->ops[at]=tmp;
              } }
            lf->nops++; } }
    else if(strstr(nm,"zl.ERF")&&psz>=8){   /* ★死层部件(2026-08-12 用户令"修死层并入反修"): 每专家低秩补丁 */
        uint32_t ne=0; uint16_t r16=0;
        memcpy(&ne,pay,4); memcpy(&r16,pay+4,2);
        size_t blkf=(size_t)DIM*r16+(size_t)r16*MOEI;
        size_t per=8+2*blkf;
        if(ne>0&&ne<=256&&r16>0&&r16<=64&&psz>=8+(uint64_t)ne*per){
            o->type=8; o->erf_ne=(int)ne; o->erf_r=(int)r16;
            o->erf_eid=malloc((size_t)ne*4); o->erf_tau=malloc((size_t)ne*4);
            o->erf_UV=malloc((size_t)ne*blkf*4);
            const uint8_t*pp=pay+8;
            for(uint32_t i=0;i<ne;i++){
                memcpy(&o->erf_eid[i],pp,4); memcpy(&o->erf_tau[i],pp+4,4); pp+=8;
                const uint16_t*hh=(const uint16_t*)pp;
                float*df=o->erf_UV+(size_t)i*blkf;
                for(size_t j=0;j<blkf;j++) df[j]=go1b_fp16_to_fp32(hh[j]);
                pp+=2*blkf;
            }
            o->foff=curfoff; lf->nops++; } }
    else if(strstr(nm,"zl.RRR")){ lf->zl_stub_at=lf->nops; }   /* 空壳占位(psz<16): 记正位 */
    else if(strstr(nm,".GL")&&psz>=4){ o->type=1; memcpy(&o->g,pay,4); o->foff=curfoff; lf->nops++; }
    else if((strstr(nm,"TREF")||strstr(nm,"xlayer"))&&psz>=4){
        o->type=4; memcpy(&o->t,pay,4); o->foff=curfoff; lf->nops++; }
}
static int lfile_load(const char*path,lfile_t*lf){
    memset(lf,0,sizeof(*lf));
    lf->zl_stub_at=-1;
    int fd=open(path,O_RDONLY); if(fd<0) return -1;
    struct stat st2; if(fstat(fd,&st2)!=0){ close(fd); return -1; }
    lf->msz=(size_t)st2.st_size;
    lf->map=mmap(NULL,lf->msz,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    if(lf->map==MAP_FAILED){ lf->map=NULL; return -1; }
    const uint8_t*p=lf->map,*end=lf->map+lf->msz;
    if(lf->msz<12||memcmp(p,"DQL2",4)!=0){ munmap(lf->map,lf->msz); lf->map=NULL; return -1; }
    uint32_t nrec; memcpy(&nrec,p+8,4); p+=12;
    lf->szG=(size_t)MOEI*go1b_blk_row_bytes(DIM);
    lf->szD=(size_t)DIM*go1b_blk_row_bytes(MOEI);
    for(uint32_t i=0;i<nrec&&p+112<=end;i++){
        char nm[17]; memcpy(nm,p,16); nm[16]=0;
        uint64_t vol,psz; memcpy(&vol,p+80,8); memcpy(&psz,p+88,8);
        float m4; memcpy(&m4,p+108,4);
        int vd; memcpy(&vd,p+112,4);
        const uint8_t*pay=p+116;
        p=pay+psz; if(p>end) break;
        if(!strcmp(nm,"1bit")&&psz>=(uint64_t)NEXP*(2*lf->szG+lf->szD)){
            lf->w1=pay; lf->w3=pay+(size_t)NEXP*lf->szG; lf->w2=pay+2*(size_t)NEXP*lf->szG;
        } else if(!strcmp(nm,"g2hot")&&psz>=24){ /* 热 go2b 内嵌(DQG2 布局原样): 指针直指主 map, 无独立 mmap */
            uint32_t mg2,kh; memcpy(&mg2,pay,4); memcpy(&kh,pay+12,4);
            size_t szG2=(size_t)MOEI*go2b_row_bytes(DIM), szD2=(size_t)DIM*go2b_row_bytes(MOEI);
            size_t hdr2=g2_sidecar_hdr((int)kh);
            if(mg2==G2SC_MAGIC&&kh>0&&kh<=256&&psz>=hdr2+2*(uint64_t)kh*szG2+(uint64_t)kh*szD2){
                lf->g2k=(int)kh;
                for(int e2=0;e2<256;e2++) lf->g2slot[e2]=-1;
                const uint16_t*idsp=(const uint16_t*)(pay+24);
                for(uint32_t i2=0;i2<kh;i2++){ uint16_t ee=idsp[i2]; if(ee<256) lf->g2slot[ee]=(int16_t)i2; }
                lf->g2w1=pay+hdr2; lf->g2w3=lf->g2w1+(size_t)kh*szG2; lf->g2w2=lf->g2w3+(size_t)kh*szG2;
            }
        } else if(vd==1){   /* 超冠混装: 主文件 op 记录即权威 */
            parse_op_rec(lf,nm,pay,psz,(size_t)(pay-lf->map));
        }
    }
    if(!lf->w1){ munmap(lf->map,lf->msz); lf->map=NULL; return -1; }
    /* go2b 侧车(有则挂): 热专家 2bit 覆盖 — 回放/反修在合并态前向。冷 dql 热槽位是稀疏洞,
     * 侧车缺失时热专家会 dequant 全零 → 硬拒加载(禁静默错) */
    { char gp[512];
      uint32_t Lh; memcpy(&Lh,lf->map+4,4);   /* L 从主文件头取(侧车路径推导需层号) */
      g2_sidecar_path(path,(int)Lh,gp,sizeof(gp));
      int g2fd=lf->g2k>0?-1:open(gp,O_RDONLY);   /* 内嵌 g2hot 已认领 → 独立侧车不再挂 */
      if(g2fd>=0){
          struct stat gst; fstat(g2fd,&gst); lf->g2msz=(size_t)gst.st_size;
          lf->g2map=mmap(NULL,lf->g2msz,PROT_READ,MAP_PRIVATE,g2fd,0); close(g2fd);
          if(lf->g2map==MAP_FAILED){ lf->g2map=NULL; }
          else {
              uint32_t mg2,kh; memcpy(&mg2,lf->g2map,4); memcpy(&kh,lf->g2map+12,4);
              size_t szG2=(size_t)MOEI*go2b_row_bytes(DIM), szD2=(size_t)DIM*go2b_row_bytes(MOEI);
              size_t hdr2=g2_sidecar_hdr((int)kh);
              if(mg2==G2SC_MAGIC&&kh>0&&kh<=256&&lf->g2msz>=hdr2+2*(size_t)kh*szG2+(size_t)kh*szD2){
                  lf->g2k=(int)kh;
                  for(int e2=0;e2<256;e2++) lf->g2slot[e2]=-1;
                  const uint16_t*idsp=(const uint16_t*)(lf->g2map+24);
                  for(uint32_t i2=0;i2<kh;i2++){ uint16_t ee=idsp[i2]; if(ee<256) lf->g2slot[ee]=(int16_t)i2; }
                  lf->g2w1=lf->g2map+hdr2; lf->g2w3=lf->g2w1+(size_t)kh*szG2; lf->g2w2=lf->g2w3+(size_t)kh*szG2;
              } else { munmap(lf->g2map,lf->g2msz); lf->g2map=NULL; }
          }
      }
      /* v2.2 VQ 侧车(有则挂): 冷 w1/w3 + 热全三矩阵字节 */
      { char vqp[512]; vq_sidecar_path(path,(int)Lh,vqp,sizeof(vqp));
        int vfd=open(vqp,O_RDONLY);
        if(vfd>=0){
            struct stat vst; fstat(vfd,&vst); lf->vqmsz=(size_t)vst.st_size;
            lf->vqmap=mmap(NULL,lf->vqmsz,PROT_READ,MAP_PRIVATE,vfd,0); close(vfd);
            if(lf->vqmap==MAP_FAILED){ lf->vqmap=NULL; }
            else { uint32_t mgv; memcpy(&mgv,lf->vqmap,4);
                   if(mgv!=VQSC_MAGIC||lf->vqmsz<vq_hdr_bytes()){ munmap(lf->vqmap,lf->vqmsz); lf->vqmap=NULL; } }
        }
      }
      if(GO2B_HOT&&(int)Lh<64&&G2_K[Lh]>0&&lf->g2k<=0&&!lf->vqmap){
          fprintf(stderr,"[go2b] ★L%u 侧车 %s 缺失/损坏(无 VQ 侧车) — 热槽位是稀疏洞, 拒加载★\n",Lh,gp);
          munmap(lf->map,lf->msz); lf->map=NULL; return -1;
      }
    }
    return 0;
}
/* (旧 lfile_append_xlayer "追加标量伪反修" 已删 — 反修改为 zfile_commit 原地重解 z, 见 backfit_prev) */
static void lfile_free(lfile_t*lf){
    for(int i=0;i<lf->nops;i++){
        if(lf->ops[i].type==3&&lf->ops[i].V8) free(lf->ops[i].V8);
        if(lf->ops[i].type==5&&lf->ops[i].ge) free(lf->ops[i].ge);
        if(lf->ops[i].type==6){ free(lf->ops[i].zlU); free(lf->ops[i].zlV); free(lf->ops[i].zlz); }
        if(lf->ops[i].type==8){ free(lf->ops[i].erf_eid); free(lf->ops[i].erf_tau); free(lf->ops[i].erf_UV); }
    }
    if(lf->vqmap) munmap(lf->vqmap,lf->vqmsz);
    if(lf->g2map) munmap(lf->g2map,lf->g2msz);
    if(lf->opsmap) munmap(lf->opsmap,lf->opsmsz);
    if(lf->map) munmap(lf->map,lf->msz);
}
/* ===== DQZ2 运行时全链侧车: 最终落地 op 链 1:1 序列化(引擎回放的唯一权威载体) =====
 * DQZ1(zfile_all.bin) 只存 SEARCH 阶段每层单条胜者; 反修/堆叠后的最终链只活在层文件的
 * vd=1 记录里, 而层文件会被 merge consume 释放 → 必须独立落盘全链(教训: 2026-07-12 首版
 * GGUF 出炉后链数据随层文件消失, 只能重跑量化找回)。
 * 格式: "DQZ2" u32 | nlayers u32 | 每层{ L u32, nops u32, 每op{ type u32, paysz u32, payload } }
 * payload 与 lfile_load/bytes_moe 口径一致: 1=GL g f32 | 2=GLdyn2 w2p f32[4](w0,w1,特征均值,SD;
 * 特征=‖Fin_s‖₂, clamp[0.25,4]) | 3=GLdyn8 w8 f32[9]+V8 fp16[8*DIM](特征=V8·Fin_s, 同 clamp) |
 * 4=TREF t f32(锚 Fcur=专家累加后) | 5=GE fp16[NEXP](乘 gate 权重, 回放取最后一条)。
 * 1/2/3 锚 shared 基 → 引擎侧整链塌缩为 routed 贡献逐 token 标量: λ←g·λ | λ←c_s·λ | λ←1+t·(λ−1)。 */
/* 单层 op 链发射(DQZ2 层块: {L,nops,每op{type,paysz,payload}}); 返回写入字节数 */
static uint64_t zc_emit_layer(FILE*f, uint32_t Lw, lfile_t*lf){
    uint32_t nops = lf ? (uint32_t)lf->nops : 0;
    fwrite(&Lw,4,1,f); fwrite(&nops,4,1,f);
    uint64_t tot=8;
    for(int i=0;lf&&i<lf->nops;i++){ lop_t*o=&lf->ops[i];
        uint32_t ty=(uint32_t)o->type, psz=0;
        if(o->type==1){ psz=4; fwrite(&ty,4,1,f); fwrite(&psz,4,1,f); fwrite(&o->g,4,1,f); }
        else if(o->type==2){ psz=16; fwrite(&ty,4,1,f); fwrite(&psz,4,1,f); fwrite(o->w2p,4,4,f); }
        else if(o->type==3){ psz=36+(o->V8?(uint32_t)(8*DIM*2):0);
            fwrite(&ty,4,1,f); fwrite(&psz,4,1,f); fwrite(o->w8,4,9,f);
            if(o->V8){ uint16_t*h=malloc((size_t)8*DIM*2);
                for(size_t j=0;j<(size_t)8*DIM;j++) h[j]=go1b_fp32_to_fp16(o->V8[j]);
                fwrite(h,2,(size_t)8*DIM,f); free(h); } }
        else if(o->type==4){ psz=4; fwrite(&ty,4,1,f); fwrite(&psz,4,1,f); fwrite(&o->t,4,1,f); }
        else if(o->type==5&&o->ge){ psz=(uint32_t)NEXP*2;
            fwrite(&ty,4,1,f); fwrite(&psz,4,1,f);
            uint16_t*h=malloc((size_t)NEXP*2);
            for(int e=0;e<NEXP;e++) h[e]=go1b_fp32_to_fp16(o->ge[e]);
            fwrite(h,2,(size_t)NEXP,f); free(h); }
        else if(o->type==6&&o->zlk>0&&o->zlU&&o->zlV&&o->zlz){   /* 冻结 z^L: 头16B + fp16{z,U[dout·k],V[din·k]} */
            uint32_t zk=(uint32_t)o->zlk, din=(uint32_t)DIM, dout=(uint32_t)DIM;
            size_t nh=(size_t)zk+(size_t)zk*din+(size_t)zk*dout;
            psz=16+(uint32_t)(2*nh);
            fwrite(&ty,4,1,f); fwrite(&psz,4,1,f);
            fwrite(&zk,4,1,f); fwrite(&o->zltr,4,1,f); fwrite(&din,4,1,f); fwrite(&dout,4,1,f);
            uint16_t*h=malloc(nh*2); size_t off2=0;
            for(uint32_t i=0;i<zk;i++) h[off2++]=go1b_fp32_to_fp16(o->zlz[i]);
            for(size_t i=0;i<(size_t)dout*zk;i++) h[off2++]=go1b_fp32_to_fp16(o->zlU[i]);
            for(size_t i=0;i<(size_t)din*zk;i++)  h[off2++]=go1b_fp32_to_fp16(o->zlV[i]);
            fwrite(h,2,nh,f); free(h); }
        else { uint32_t z=0; fwrite(&ty,4,1,f); fwrite(&z,4,1,f); }   /* 未知型: 空载荷占位 */
        tot+=8+psz;
    }
    return tot;
}
/* 每层"优化文件"(用户产品形态: 一层两份 = dql_LXX.bin 量化 + opt_LXX.bin 优化):
 * 单层 DQZ2(nlayers=1), 任何 DQZ2 读者可直接消费。导出时写初版, zchain_write 刷终值。 */
static void zc_opt_emit(int L, lfile_t*lf){
    const char*ld=getenv("DS4_LAYER_DIR"); if(!ld) return;
    char op2[512]; snprintf(op2,sizeof(op2),"%s/opt_L%02d.bin",ld,L);
    FILE*f=fopen(op2,"wb"); if(!f) return;
    uint32_t magic=0x325A5144, one=1;
    fwrite(&magic,4,1,f); fwrite(&one,4,1,f);
    zc_emit_layer(f,(uint32_t)L,lf);
    fclose(f);
}
static void zchain_write(void){
    const char*p=getenv("DS4_ZCHAIN"); if(!p) return;
    if(getenv("DS4_MINVOL_MAXL")){   /* ★探针/部分层跑禁写(2026-08-03 事故: 7层探针把 43 层终值 zchain 覆盖成空链) */
        fprintf(stderr,"[zchain] 探针模式(MAXL)跳过落盘, 防覆盖全量终值\n"); return; }
    const char*ld=getenv("DS4_LAYER_DIR"); if(!ld) return;
    FILE*f=fopen(p,"wb"); if(!f){ fprintf(stderr,"[zchain] 写 %s 失败\n",p); return; }
    uint32_t magic=0x325A5144, nlay=(uint32_t)NL;
    fwrite(&magic,4,1,f); fwrite(&nlay,4,1,f);
    uint64_t tot=8; int lay_ok=0, ops_tot=0;
    for(int L=0;L<NL;L++){
        char lp[512]; snprintf(lp,sizeof(lp),"%s/dql_L%02d.bin",ld,L);
        lfile_t lf;
        if(lfile_load(lp,&lf)!=0){ tot+=zc_emit_layer(f,(uint32_t)L,NULL); continue; }
        tot+=zc_emit_layer(f,(uint32_t)L,&lf);
        zc_opt_emit(L,&lf);   /* 刷新每层优化文件为终值(反修/回扫后的链) */
        lay_ok++; ops_tot+=lf.nops; lfile_free(&lf);
    }
    fclose(f);
    printf("ZCHAIN %s layers=%d/%d ops=%d bytes=%llu (%.2f MB) (+opt_LXX.bin×%d 终值)\n",
           p,lay_ok,NL,ops_tot,(unsigned long long)tot,(double)tot/1048576.0,lay_ok);
    fflush(stdout);
}
/* bytes MoE: 量化字节前向 + 修正链回放(shared FP 已在 Fout 里) */
/* 并行专家 worker(bytes_moe): 原子计数器分发 e, 私有 partial 累加, 主线程归约 */
typedef struct { lfile_t*lf; int S; const float*Fin; const int*idx; const float*rw;
                 int *e_next; float *partial; const float*ge; float *partial_c; } bmw_t;
/* ★冷热分桶缓存(2026-08-06 用户令"冷热双通道")★: bytes_moe 按专家冷热分离累计
 * routed = R_hot + R_cold(贡献项分桶, 非按 token)。热判定=合并态口径(vq w2 槽非零 /
 * g2slot>=0)。供 GLhc(type7) 求解与回放; 每次 bytes_moe 重写。 */
static float *BM_RH=NULL,*BM_RC=NULL; static int BM_S=0;
/* ★层级消融门(2026-08-19 五指标诊断)★ DS4_REPLAY_SKIP_LAYERS="24,25,..": 回放时
 * 整层跳过修正链(权重字节前向保留) — 与 DS4_REPLAY_SKIP_TYPES 正交组合。 */
static int g_replay_cur_L=-1;
static int replay_layer_skipped(void){
    static int init=0, skip[64]={0};
    if(!init){ init=1; const char*sv=getenv("DS4_REPLAY_SKIP_LAYERS");
        if(sv&&*sv){ char b2[256]; snprintf(b2,256,"%s",sv);
            for(char*tk=strtok(b2,",");tk;tk=strtok(NULL,",")){ int t2=atoi(tk);
                if(t2>=0&&t2<64) skip[t2]=1; }
            fprintf(stderr,"[replay] 层消融: 跳过修正链 层{%s}\n",sv); } }
    return g_replay_cur_L>=0&&g_replay_cur_L<64&&skip[g_replay_cur_L];
}
static void *bytes_moe_worker(void*a){
    bmw_t*w=a; lfile_t*lf=w->lf; int S=w->S;
    float *q1=malloc((size_t)MOEI*DIM*4),*q3=malloc((size_t)MOEI*DIM*4),*q2=malloc((size_t)DIM*MOEI*4);
    int *tok=malloc((size_t)S*sizeof(int)); float *xs=malloc((size_t)S*DIM*4),*wwv=malloc((size_t)S*4);
    float *w1p=NULL,*w3p=NULL,*w2p=NULL;   /* 本迭代实际权重指针(本地缓冲或批 dequant 切片) */
    for(;;){ int e=__sync_fetch_and_add(w->e_next,1); if(e>=NEXP)break;
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
            extern float *g_bmw_buf; extern int g_bmw_batched;
            if(g_bmw_batched){   /* 批 dequant 已就位: 指针别名切片, 不动 q1..q2 生命周期 */
                w1p=g_bmw_buf+((size_t)e*3+0)*((size_t)DIM*MOEI);
                w3p=g_bmw_buf+((size_t)e*3+1)*((size_t)DIM*MOEI);
                w2p=g_bmw_buf+((size_t)e*3+2)*((size_t)DIM*MOEI);
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
        float *aq=calloc((size_t)nt*DIM,4);
        { extern double g_bmw_t[2]; double bt1=vqt_now();
          dq_expert_fp(xs,w1p?w1p:q1,w3p?w3p:q3,w2p?w2p:q2,wwv,aq,nt,DIM,MOEI,SWLIM);
          g_bmw_t[1]+=vqt_now()-bt1; }
        for(int i=0;i<nt;i++){ float*dst=(e_hot?w->partial:w->partial_c)+(size_t)tok[i]*DIM;
            const float*yi=aq+(size_t)i*DIM;
            for(int d2=0;d2<DIM;d2++) dst[d2]+=yi[d2]; }
        free(aq);
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
    free(q1);free(q3);free(q2);free(tok);free(xs);free(wwv); return NULL;
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
#ifdef DS4QUANT_CUDA
/* 第三刀(08-18): 整层 768 矩阵一次批 dequant(vq_gpu.cu), 消 99k 次 per-矩阵 sync。
 * 26GB fp32 缓冲静态复用(裸判期内存空闲); worker 前向直接吃切片指针零拷贝。 */
typedef struct { uint64_t pay_off, dst_off; int rows, cols, nc, nbit; } vqg_deq_job;
extern int vqg_dequant_batch(const uint8_t*, float*, const vqg_deq_job*, int, int, int);
extern int vqg_ready(void);
float *g_bmw_buf=NULL;   /* [256][3][8.4M] 切片: e*3+w */
int g_bmw_batched=0;     /* 本层批 dequant 成功旗标 */
static size_t bmw_slot_off(int e,int w){ return ((size_t)e*3+w)*( (size_t)DIM*MOEI ); }
static int bmw_batch_dequant(lfile_t*lf){
    if(!lf->vqmap||!vqg_ready()) return 0;
    const uint64_t *vtab=(const uint64_t*)(lf->vqmap+16);
    if(!g_bmw_buf){   /* managed: GPU 写零页故障(malloc 26GB 首触=百万级 HMM fault, 实测比逐矩阵还慢) */
        extern int vqg_alloc_managed(void**,size_t);
        if(!vqg_alloc_managed((void**)&g_bmw_buf,(size_t)NEXP*3*DIM*MOEI*4)) return 0; }
    static vqg_deq_job jobs[NEXP*3]; int nj=0, nc_max=0;
    for(int e=0;e<NEXP;e++) for(int w=0;w<3;w++){
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
#ifdef DS4QUANT_CUDA
    g_bmw_batched=bmw_batch_dequant(lf);
#endif
    float *Fbase=malloc((size_t)S*DIM*4); memcpy(Fbase,Fout,(size_t)S*DIM*4);   /* shared 基 */
    const float *ge=NULL;   /* bf.GE(type-5, 取最后一条): per-expert 增益, 专家累加时乘(链 op 之前) */
    const int lay_skip=replay_layer_skipped();
    if(!lay_skip) for(int i=lf->nops-1;i>=0;i--) if(lf->ops[i].type==5&&lf->ops[i].ge){ ge=lf->ops[i].ge; break; }
    int nth=NTHREADS<1?1:(NTHREADS>NEXP?NEXP:NTHREADS); int e_next=0;
    bmw_t *ws=calloc((size_t)nth,sizeof(bmw_t)); pthread_t *th=malloc((size_t)nth*sizeof(pthread_t));
    for(int t=0;t<nth;t++){ ws[t]=(bmw_t){lf,S,Fin,idx,rw,&e_next,calloc((size_t)S*DIM,4),ge,
                                          calloc((size_t)S*DIM,4)};
        pthread_create(&th[t],NULL,bytes_moe_worker,&ws[t]); }
    /* 冷热分桶缓存重建(hot=partial / cold=partial_c 归约) */
    if(BM_S!=S){ free(BM_RH); free(BM_RC); BM_RH=malloc((size_t)S*DIM*4); BM_RC=malloc((size_t)S*DIM*4); BM_S=S; }
    memset(BM_RH,0,(size_t)S*DIM*4); memset(BM_RC,0,(size_t)S*DIM*4);
    for(int t=0;t<nth;t++){ pthread_join(th[t],NULL);
        for(size_t i=0;i<(size_t)S*DIM;i++){ BM_RH[i]+=ws[t].partial[i]; BM_RC[i]+=ws[t].partial_c[i]; }
        free(ws[t].partial); free(ws[t].partial_c); }
    for(size_t i=0;i<(size_t)S*DIM;i++) Fout[i]+=BM_RH[i]+BM_RC[i];
    free(ws);free(th);
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
            }
            free(zd); free(pv); if(phi)free(phi);
        }
    }
    if(xn)free(xn); if(pj)free(pj);
    free(Fbase); free(Fcur);
    { static int bn=0; if((++bn%43)==0||getenv("DS4_BMW_TIMING"))
        fprintf(stderr,"[bmwt] dequant=%.1fs 前向=%.1fs (bytes_moe 累计)\n",g_bmw_t[0],g_bmw_t[1]); }
}
/* 可复用调优轮(GL重拟合→GLdyn2→GLdyn8→TREF), 循环至整轮零接管; 返回是否有过接管 */
typedef struct { int L,S,n_fit,vs; size_t rowsz; double bval,bheld;
    const float *Fin,*Hf,*H2,*post2,*comb2,*shared,*Fcur;
    float *Fstate,*Ftest,*routedC,*DF,*Hq;
    double *sc_state,*v_cur,*h_cur; } costx_t;
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
    if(getenv("DS4_LAYER_DIR")){   /* 导出口径=调优口径: 存本层累积激活+实际路由 */
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
    const int nqc=(FAST||dq_vq_on()||getenv("DS4_MINVOL"))?1:(int)(sizeof(QC)/sizeof(QC[0]));   /* minvol: g10 定稿单配置(g10=历史最优代表, 13.4→~3min/层) */
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
    /* ZL: 加法低秩 — 用户裁决删除加法元素(2026-07-11), 默认关, DS4_ADD_FORMS=1 可复活 */
    if(getenv("DS4_ADD_FORMS")){ ds4_loss_dim_variance(DF,(uint32_t)(vs>0?vs:1),(uint32_t)DIM,colw);
      double m=0; for(int j=0;j<DIM;j++) m+=colw[j]; m/=DIM; if(m<1e-30)m=1e-30;
      for(int j=0;j<DIM;j++){ colw[j]=(float)(colw[j]/m); if(colw[j]<0.05f)colw[j]=0.05f; }
      const double ZLAM[2]={1,10};
      for(int li=0;li<2;li++){
        ds4_z *zl=z_solve_fourloss(Fin,DF,vs,DIM,DIM,16,(float)ZLAM[li],colw,vs,0.25f,0x51F0F00DULL+li);
        if(!zl) continue;
        int kp = nval>0 ? z_pick_rank(zl,Fin+(size_t)vs*DIM,DF+(size_t)vs*DIM,nval,DIM,NULL,NULL,NULL,NULL) : 4;
        int ks[2]={kp,4}; float *zd=malloc((size_t)DIM*4);
        for(int ki=0;ki<2;ki++){ int k=ks[ki];
            if(k<1||(ki==1&&ks[0]==4)) continue;
            ds4_z_set_rank(zl,(uint32_t)k);
            memcpy(Ftest,Fcur,(size_t)S*DIM*4);
            for(int s=0;s<S;s++){
                memset(zd,0,(size_t)DIM*4); ds4_z_apply(zl,Fin+(size_t)s*DIM,zd);
                double nd=0,nf=0; const float*fo=Fcur+(size_t)s*DIM;
                for(int d2=0;d2<DIM;d2++){ nd+=(double)zd[d2]*zd[d2]; nf+=(double)fo[d2]*fo[d2]; }
                nd=sqrt(nd); nf=sqrt(nf); double cap=LZTR*nf; float s2=1.0f;
                if(nd>cap&&nd>0) s2=(float)(cap/nd);
                float*fw=Ftest+(size_t)s*DIM;
                for(int d2=0;d2<DIM;d2++) fw[d2]+=s2*zd[d2];
            }
            co_eval(Ftest,H2,post2,comb2,Hf,Hq,S,vs,n_fit,rowsz,&fitr,&valr,&heldr,&sc);
            printf("SEARCH L=%d ZL λ=%-3g k=%-2d val=%.4f(%+.1f%%) held=%.4f(%+.2f%%) fit=%.4f sc=%.5g\n",
                   L,ZLAM[li],k,valr,100*(valr-bval)/bval,heldr,100*(heldr-bheld)/bheld,fitr,sc);
            CK("ZL",ZLAM[li],0,k);
        }
        free(zd); ds4_z_free(zl);
      }
    }
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
    else if(!strncmp(win.algo,"ZL",2)){ /* 确定性重解并截断打包 */
        ds4_z *zl=z_solve_fourloss(Fin,DF,vs,DIM,DIM,16,win.lam,colw,vs,0.25f,
                                   0x51F0F00DULL+(win.lam>5?1:0));
        if(zl){ pay=co_pay_zl(zl,win.k,&paysz); ds4_z_free(zl); }
    }
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
    { int BF_ALT=getenv("DS4_BF_ALT")?atoi(getenv("DS4_BF_ALT")):2; if(BF_ALT<1)BF_ALT=1;
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
        double gate = getenv("DS4_BF_GAIN_GATE")?atof(getenv("DS4_BF_GAIN_GATE")):0.05;
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
    if(getenv("DS4_BWD_FINAL")){   /* 终端反调物料: base/校正 两份出口 H(hc_post 线性 → t 插值合法) */
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
static void layer_fwd(int L, LW*W, float*H, const long*ids, int S, int n_fit,
                      int do_quant, char cfg, lstat_t*st){
    int mixd=2*HCM+HCM*HCM;
    float *cosr=malloc((size_t)S*(RD/2)*4),*sinr=malloc((size_t)S*(RD/2)*4);
    if(CR[L]>0) dq_freqs_cis(RD,S,65536.0,160000.0,16.0,32.0,1.0,cosr,sinr);
    else dq_freqs_cis(RD,S,0.0,10000.0,16.0,32.0,1.0,cosr,sinr);
    float *y=malloc((size_t)S*DIM*4),*post=malloc((size_t)S*HCM*4),*comb=malloc((size_t)S*HCM*HCM*4);
    dq_hc_pre(H,W->afn,W->asc,W->abase,y,post,comb,S,HCM,DIM,mixd,HCIT,EPSF,EPSF);
    float *xn=malloc((size_t)S*DIM*4); for(int s=0;s<S;s++)dq_rms(y+(size_t)s*DIM,W->an,xn+(size_t)s*DIM,DIM,EPSF);
    int Sc=0; float *kvc=NULL;
    if(CR[L]>0){ kvc=malloc((size_t)((S/CR[L]+2)*2)*HD*4); Sc=dq_compressor(xn,W->cwkv,W->cwgate,W->cnorm,W->cape,cosr,sinr,kvc,S,DIM,HD,RD,CR[L],EPSF); }
    float *a=malloc((size_t)S*DIM*4);
    dq_attention(xn,W->wqa,W->qn,W->wqb,W->wkv,W->kvn,W->sink,W->woa,W->wob,kvc,cosr,sinr,a,S,DIM,NH,HD,RD,QLR,OLR,OG,WIN,Sc,CR[L],EPSF);
    float *H2=malloc((size_t)S*HCM*DIM*4); dq_hc_post(a,H,post,comb,H2,S,HCM,DIM);
    float *y2=malloc((size_t)S*DIM*4),*post2=malloc((size_t)S*HCM*4),*comb2=malloc((size_t)S*HCM*HCM*4);
    dq_hc_pre(H2,W->ffn,W->fsc,W->fbase,y2,post2,comb2,S,HCM,DIM,mixd,HCIT,EPSF,EPSF);
    float *Fin=malloc((size_t)S*DIM*4); for(int s=0;s<S;s++)dq_rms(y2+(size_t)s*DIM,W->fn,Fin+(size_t)s*DIM,DIM,EPSF);
    if(GS_CAP_L==L&&GS_FIN) memcpy(GS_FIN,Fin,(size_t)S*DIM*4);   /* ★反修: 捕获目标层 MoE 输入(算 z 特征用)★ */
    /* moe 路由(实际激活: 量化遍即被污染激活 = 部署运行时口径) */
    int *idx=malloc((size_t)S*NACT_RT*sizeof(int)); float *rw=malloc((size_t)S*NACT_RT*4);
    if(W->t2ei){ static int hbn=0;
        if(!hbn&&do_quant){ hbn=1;
            fprintf(stderr,"[路由] tid2eid 哈希路由生效(0731 原生): 专家选择=token 哈希表, 结构性零漂移;\n"
                           "[路由] Δb 选择偏置在此无对象(老 base 分数近似路由时代的机制), RB 族不武装。\n"); }
        dq_gate_route_hash(Fin,W->gate,W->t2ei,ids,idx,rw,S,DIM,NEXP,NACT,ROUTE_SCALE);
    }
    else{
        if(!RB_TRIED&&getenv("DS4_ROUTE_BIAS")){   /* 懒加载路由偏置侧车(一次) */
            RB_TRIED=1; FILE*rf=fopen(getenv("DS4_ROUTE_BIAS"),"rb");
            if(rf){ uint32_t hd[4]={0,0,0,0};
                /* 文件层数≥当前 NL 即可(取前 NL 行): NL=1 探针消费 43 层侧车(2026-08-03) */
                if(fread(hd,4,4,rf)==4&&hd[0]==0x41494252u&&hd[1]>=(uint32_t)NLAYERS&&hd[2]==(uint32_t)NEXP){
                    uint32_t nlf=hd[1];
                    RB_APPLY=malloc((size_t)nlf*NEXP*4);
                    uint32_t*c=malloc((size_t)nlf*NEXP*4);
                    if(fread(RB_APPLY,4,(size_t)nlf*NEXP,rf)==(size_t)nlf*NEXP&&
                       fread(c,4,(size_t)nlf*NEXP,rf)==(size_t)nlf*NEXP){
                        if(getenv("DS4_ROUTE_BIAS_ALPHA")) RB_ALPHA=atof(getenv("DS4_ROUTE_BIAS_ALPHA"));
                        if(getenv("DS4_ROUTE_BIAS_MINCNT")) RB_MINCNT=atoi(getenv("DS4_ROUTE_BIAS_MINCNT"));
                        long armed=0;
                        for(size_t i=0;i<(size_t)NLAYERS*NEXP;i++){
                            if((int)c[i]<RB_MINCNT) RB_APPLY[i]=0.0f; else if(RB_APPLY[i]!=0.0f) armed++; }
                        printf("ROUTE_BIAS apply α=%.2f mincnt=%d 武装槽=%ld\n",RB_ALPHA,RB_MINCNT,armed);
                        if(getenv("DS4_ROUTE_SEQ")){   /* ★per-layer α 装载(2026-08-14): 读 <bias>.alpha.txt */
                            char ap2[512]; snprintf(ap2,sizeof(ap2),"%s.alpha.txt",getenv("DS4_ROUTE_BIAS"));
                            FILE*af=fopen(ap2,"r");
                            if(af){ int l2; float a2; int na=0;
                                while(fscanf(af,"L=%d a=%f\n",&l2,&a2)==2){ if(l2>=0&&l2<64){ RB_ALPHA_L[l2]=a2; if(a2>0)na++; } }
                                fclose(af); RB_SEQ=1;
                                printf("ROUTE_BIAS per-layer α 装载: 开启层=%d\n",na); } }
                    } else { free(RB_APPLY); RB_APPLY=NULL; }
                    free(c);
                }
                fclose(rf);
            }
        }
        const float *gb=W->gbias; float *gb2=NULL;
        if(RB_APPLY){ float al = RB_SEQ ? RB_ALPHA_L[L] : RB_ALPHA;   /* 序贯: per-layer 动态 α(0=层关) */
            if(al!=0.0f){ gb2=malloc((size_t)NEXP*4);
                for(int e=0;e<NEXP;e++) gb2[e]=(W->gbias?W->gbias[e]:0.0f)+al*RB_APPLY[(size_t)L*NEXP+e];
                gb=gb2; } }
        /* ★token门控路由偏置(2026-08-14 用户令"路由问题没有解决"·静态Δb中位KL2.4×判死后唯一剂型)★:
         * DS4_ROUTE_GATE_TAU>0 时 Δb 只施于路由不确定的 token — 无偏置选择的 max(rw)<τ
         * (权重平坦=选择摇摆=漂移高发区); 置信 token 保持原路由(保分布保真)。τ=0/未设=旧全量行为。 */
        { float gtau=getenv("DS4_ROUTE_GATE_TAU")?atof(getenv("DS4_ROUTE_GATE_TAU")):0.0f;
          if(gb2&&gtau>0.0f){
            dq_gate_route_topk(Fin,W->gate,W->gbias,idx,rw,S,DIM,NEXP,NACT_RT,ROUTE_SCALE);   /* 无偏置基线 */
            int *idxb=malloc((size_t)S*NACT_RT*sizeof(int)); float *rwb=malloc((size_t)S*NACT_RT*4);
            dq_gate_route_topk(Fin,W->gate,gb,idxb,rwb,S,DIM,NEXP,NACT_RT,ROUTE_SCALE);       /* 带偏置 */
            long ngate=0;
            for(int s2=0;s2<S;s2++){
                float mx=0.0f; for(int a2=0;a2<NACT_RT;a2++){ float v=rw[(size_t)s2*NACT_RT+a2]; if(v>mx)mx=v; }
                if(mx<gtau){ memcpy(idx+(size_t)s2*NACT_RT,idxb+(size_t)s2*NACT_RT,NACT_RT*sizeof(int));
                             memcpy(rw+(size_t)s2*NACT_RT,rwb+(size_t)s2*NACT_RT,NACT_RT*4); ngate++; }
            }
            free(idxb); free(rwb);
            static int gt_note=0;
            if(!gt_note){ gt_note=1; fprintf(stderr,"[路由门] τ=%.2f 首层门通过率=%.1f%%\n",gtau,100.0*ngate/S); }
          } else
        if(RR_EPS>0.0f && NACT_RT>NACT){
            /* ★加法式并集(2026-08-16): 前NACT名 id+权重=原协议逐位(6内归一不稀释);
             * 额外槽取第7/8名, 权重= eps × (∝sc 相对权) 加法进场 — m1乘法归一版
             * 已死于分数压缩变换(第7名抢走~10%质量), 本式把稀释归零。 */
            dq_gate_route_topk(Fin,W->gate,gb,idx,rw,S,DIM,NEXP,NACT_RT,ROUTE_SCALE);
            int *idx6=malloc((size_t)S*NACT*sizeof(int)); float *rw6=malloc((size_t)S*NACT*4);
            dq_gate_route_topk(Fin,W->gate,gb,idx6,rw6,S,DIM,NEXP,NACT,ROUTE_SCALE);
            for(int s2=0;s2<S;s2++){
                for(int a=NACT;a<NACT_RT;a++) rw[(size_t)s2*NACT_RT+a] *= RR_EPS;
                memcpy(idx+(size_t)s2*NACT_RT, idx6+(size_t)s2*NACT, NACT*sizeof(int));
                memcpy(rw +(size_t)s2*NACT_RT, rw6 +(size_t)s2*NACT, NACT*4);
            }
            free(idx6); free(rw6);
        } else
        dq_gate_route_topk(Fin,W->gate,gb,idx,rw,S,DIM,NEXP,NACT_RT,ROUTE_SCALE);
        }
        if(gb2) free(gb2);
        /* ★锚路由与 Δb 统计不互斥(2026-08-01 解除)★ 原条件带 !DS4_ANCHOR_ROUTE && !BF_ANCROUTE,
         * 但这是多余的自设限制: 本段统计只需要 ① 学生 gate 分数 scv(下面自己按 W->gate 算,
         * 与前向走哪条路由无关)② 学生 top-k idx(2628 行 dq_gate_route_topk 刚算出)③ FP 锚
         * top-k fpx。而锚路由的覆盖(下面 2652 段 idx=aidx)发生在本段【之后】—— 统计读到的
         * idx 仍是学生的。解除后可以同时拿到"前向 100% 走 FP 锚路由"(消除路由漂移这个误差源,
         * 量化误差纯净)和"Δb 仍被收集"(合并时烘焙, 让部署态学生路由逼近锚)。
         * R28 教训: 全程学生路由 ⇒ 路由一致率掉到 82.9%, 量化误差与路由漂移误差混在一起无法分离。*/
        if((RB_SEQ||getenv("DS4_ROUTE_BIAS_FIT"))&&do_quant&&ANC_OK
           &&(g_rb_fit_L<0||g_rb_fit_L==L)&&!(RB_SEQ&&g_rb_fit_L<0)){   /* 序贯: 只在指定层的 FIT 前向统计 */
            if(!RB_ACC){ RB_ACC=calloc((size_t)NLAYERS*NEXP,4); RB_CNT=calloc((size_t)NLAYERS*NEXP,4);
                fprintf(stderr,"[路由] Δb 统计首次武装 L=%d cfg前向\n",L); }
            const int32_t *fpx=ANC.ridx+(size_t)L*(g_anc_rowmap?g_anc_rowstride:S)*NACT;
            float *scv=malloc((size_t)NEXP*4);
            for(int s=0;s<S;s++){
                const float *xr=Fin+(size_t)s*DIM;
                for(int e=0;e<NEXP;e++){ const float*gr=W->gate+(size_t)e*DIM;
                    float raw=0.0f; for(int k=0;k<DIM;k++) raw+=xr[k]*gr[k];
                    scv[e]=sqrtf(log1pf(expf(raw)))+(W->gbias?W->gbias[e]:0.0f); }
                float thr=1e30f;
                for(int a=0;a<NACT;a++){ float v=scv[idx[(size_t)s*NACT+a]]; if(v<thr)thr=v; }
                const size_t frow=(size_t)(g_anc_rowmap?g_anc_rowmap[s]:s);
                for(int a=0;a<NACT;a++){ int e=fpx[frow*NACT+a]; int in=0;
                    for(int b=0;b<NACT;b++) if(idx[(size_t)s*NACT+b]==e){in=1;break;}
                    if(!in){ RB_ACC[(size_t)L*NEXP+e]+=thr-scv[e]; RB_CNT[(size_t)L*NEXP+e]++; } }
                for(int a=0;a<NACT;a++){ int e=idx[(size_t)s*NACT+a]; int in=0;
                    for(int b=0;b<NACT;b++) if(fpx[frow*NACT+b]==e){in=1;break;}
                    if(!in){ RB_ACC[(size_t)L*NEXP+e]-=scv[e]-thr; RB_CNT[(size_t)L*NEXP+e]++; } }
            }
            free(scv);
        }
    }
    if(do_quant&&ANC_OK&&(getenv("DS4_ANCHOR_ROUTE")||BF_ANCROUTE)){   /* 神谕路由归因: 强制 FP 路由(专家选择+权重),
        隔离"路由漂移"对最终质量的贡献; 反修判据也用它禁翻转噪声 */
        /* g_anc_rowmap 非空 = 反修粗筛的抽格前向: 锚按【原始行号】取, stride 用原始 S。
         * 为空则恒等映射, 与改动前逐字节一致。 */
        const int astride = g_anc_rowmap ? g_anc_rowstride : S;
        const int32_t *abase = ANC.ridx+(size_t)L*astride*NACT;
        const float   *rbase = ANC.rw  +(size_t)L*astride*NACT;
        for(int s=0;s<S;s++){
            const int src = g_anc_rowmap ? g_anc_rowmap[s] : s;
            for(int a=0;a<NACT;a++) idx[(size_t)s*NACT+a]=abase[(size_t)src*NACT+a];
            memcpy(rw+(size_t)s*NACT, rbase+(size_t)src*NACT, (size_t)NACT*4);
        }
    }
    if(GS_CAP_L==L&&GS_IDXC&&GS_RWC){   /* ★反修 GE 投影: 捕获目标层实际路由(命中+权重)★ */
        memcpy(GS_IDXC,idx,(size_t)S*NACT*sizeof(int));
        memcpy(GS_RWC,rw,(size_t)S*NACT*4);
    }
    if(g_chfd>=0&&S==g_chS){   /* ★链态锚捕获: 部署链 fin/路由 直写盘★ */
        size_t bf=40, br=bf+(size_t)NLAYERS*S*DIM*4, bw=br+(size_t)NLAYERS*S*NACT*4;
        pwrite(g_chfd,Fin,(size_t)S*DIM*4,(off_t)(bf+(size_t)L*S*DIM*4));
        int32_t*i32=malloc((size_t)S*NACT*4);
        for(size_t i=0;i<(size_t)S*NACT;i++) i32[i]=(int32_t)idx[i];
        pwrite(g_chfd,i32,(size_t)S*NACT*4,(off_t)(br+(size_t)L*S*NACT*4)); free(i32);
        pwrite(g_chfd,rw,(size_t)S*NACT*4,(off_t)(bw+(size_t)L*S*NACT*4));
        fdatasync(g_chfd);   /* 同内存卫生 */
    }
    if(ANC_BUILD){   /* 锚定捕获: FP 激活/路由 */
        memcpy(ANC.fin+(size_t)L*S*DIM, Fin, (size_t)S*DIM*4);
        for(size_t i=0;i<(size_t)S*NACT;i++) ANC.ridx[(size_t)L*S*NACT+i]=(int32_t)idx[i];
        memcpy(ANC.rw+(size_t)L*S*NACT, rw, (size_t)S*NACT*4);
        /* FP 锚路 X 捕获(2026-07-28 g4c 流形针): dql 层文件被 merge 消耗后学生态 B 回放
         * 不可得 → FP 态 ffn_in 口径(与 DS4_BF_DUMPXR 学生态口径区分, 探针内部自洽即可)。 */
        { const char*xe=getenv("DS4_BF_DUMPXR");
          if(xe){ const char*q=xe; int hit=0;
              while(*q){ if(atoi(q)==L){hit=1;break;} while(*q&&*q!=',')q++; if(*q==',')q++; }
              if(hit){ char px[64]; snprintf(px,sizeof(px),"/tmp/xr_x_L%02d.npy",L);
                  xr_npy(px,Fin,S,DIM);
                  fprintf(stderr,"[XR-FP] L%d X 已捕获: %s (S=%d)\n",L,px,S); } } }
    }
    if(st&&do_quant&&ANC_OK){   /* 路由一致率 vs FP 锚定 (路由漂移=MoE 误差放大器) */
        const int32_t *aidx=ANC.ridx+(size_t)L*S*NACT; long match=0;
        for(int s=0;s<S;s++)for(int a2=0;a2<NACT;a2++){ int e=idx[(size_t)s*NACT+a2];
            for(int b=0;b<NACT;b++) if(aidx[(size_t)s*NACT+b]==e){match++;break;} }
        st->agree=100.0*(double)match/((double)S*NACT);
    }
    /* shared 专家 FP 为公共基 */
    float *Fout=calloc((size_t)S*DIM,4); dq_expert_fp(Fin,W->s1,W->s3,W->s2,NULL,Fout,S,DIM,MOEI,SWLIM);
    if(do_quant&&cfg=='B'){
        /* 字节重前向: 从层文件(权重字节+落地修正链)执行, 不重量化 — 全局回扫的引擎 */
        float *shb=malloc((size_t)S*DIM*4); memcpy(shb,Fout,(size_t)S*DIM*4);   /* shared 基 */
        if(GS_LF&&L<NL&&GS_LF[L].map){   /* ★存档缓存: 不重 mmap/不重解析★ */
            g_replay_cur_L=L; bytes_moe(&GS_LF[L],S,Fin,idx,rw,Fout);
            if(GS_GV&&GS_GV_L==L){        /* ★反修: 目标层 per-token routed 增益(搜每 token 最优乘子)★ */
                for(int s=0;s<S;s++){ float g=GS_GV[s]; float*fw=Fout+(size_t)s*DIM,*sb=shb+(size_t)s*DIM;
                    for(int d=0;d<DIM;d++) fw[d]=sb[d]+g*(fw[d]-sb[d]); } }
            else if(GBL_G[L]!=1.0f) for(size_t i=0;i<(size_t)S*DIM;i++) Fout[i]=shb[i]+GBL_G[L]*(Fout[i]-shb[i]);
        /* ★corr 重建门物料(DS4_BF_DUMPXR=L, 一次性)★: X=学生态 ffn_in; R=FP专家(同X同路由)−学生routed。
         * shared FP 两侧同→抵消; 同 X 同 gate → 同路由 → R=纯专家量化残差(corr 契约口径, runbook §0.5)。 */
        { static uint64_t xr_done=0; const char*xe=getenv("DS4_BF_DUMPXR");   /* 支持逗号多层: "5,20,35"; 位图防重 */
          int xr_hit=0;
          if(xe&&L<64&&!((xr_done>>L)&1)){ const char*q=xe;
              while(*q){ if(atoi(q)==L){ xr_hit=1; break; } while(*q&&*q!=',')q++; if(*q==',')q++; } }
          if(xr_hit){ xr_done|=(1ULL<<L);
            float *fpout=calloc((size_t)S*DIM,4);
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
                for(int i=0;i<nt;i++)for(int d2=0;d2<DIM;d2++) fpout[(size_t)toks[i]*DIM+d2]+=aq[(size_t)i*DIM+d2];
                free(aq); free(e1);free(e3);free(e2);
            }
            for(size_t i=0;i<(size_t)S*DIM;i++) fpout[i]-=(Fout[i]-shb[i]);   /* R = FP − 学生routed */
            char px[64],pr[64];
            snprintf(px,sizeof(px),"/tmp/xr_x_L%02d.npy",L); snprintf(pr,sizeof(pr),"/tmp/xr_r_L%02d.npy",L);
            xr_npy(px,Fin,S,DIM); xr_npy(pr,fpout,S,DIM);
            fprintf(stderr,"[XR] L%d 重建门物料已捕获: %s %s (S=%d)\n",L,px,pr,S);
            free(fpout);free(toks);free(wwv2);free(xs2);
          } }
        } else {
            lfile_t lf; char lp[512];
            snprintf(lp,sizeof(lp),"%s/dql_L%02d.bin",getenv("DS4_LAYER_DIR")?getenv("DS4_LAYER_DIR"):".",L);
            if(lfile_load(lp,&lf)==0){
                /* ★反修遍硬闸(2026-08-23 审计)★: VQ 侧车挂载失败会静默 vqmap=NULL →
                 * bytes_moe 拿 1bit 基座当学生, z 全解错且无报错。禁静默假学生。 */
                if(LZRANK>0&&!lf.vqmap){
                    fprintf(stderr,"★反修遍 L%d: dql_vq 侧车缺/损(vqmap=NULL), 学生≠部署字节 — 拒跑★\n",L);
                    exit(1);
                }
                g_replay_cur_L=L; bytes_moe(&lf,S,Fin,idx,rw,Fout); lfile_free(&lf);
                if(L<NL&&GBL_G[L]!=1.0f) for(size_t i=0;i<(size_t)S*DIM;i++) Fout[i]=shb[i]+GBL_G[L]*(Fout[i]-shb[i]);
            } else fprintf(stderr,"[B] 层文件 %s 读失败 — Fout 只含 shared\n",lp);
        }
        if(LZRANK>0&&do_quant){   /* ★层局部靶物料(两分支汇合点, shb 仍活): R=FP专家@Fin−回放routed */
            if(!BF_LT) BF_LT=malloc((size_t)S*DIM*4);
            bf_fp_routed(L,Fin,idx,rw,S,BF_LT);
            for(size_t i=0;i<(size_t)S*DIM;i++) BF_LT[i]-=(Fout[i]-shb[i]);
            BF_LT_L=L;
        }
        free(shb);
    } else if(do_quant&&cfg=='g'&&COADAPT>0&&ANC_OK){
        /* 共适应路径: base(w2)↔z 交替闭式收敛(取代下方一次性 worker + 一次性 z 块) */
        coadapt_moe(L,S,n_fit,Fin,idx,rw,Fout,H2,post2,comb2,st);
    } else {
    int want_local = do_quant && cfg!='F' && (LOCALQ||ABLATE);
    float *shared=NULL,*Ffp=NULL,*Fabl[4]={0,0,0,0};
    if(want_local){ shared=malloc((size_t)S*DIM*4); memcpy(shared,Fout,(size_t)S*DIM*4);
                    Ffp=malloc((size_t)S*DIM*4);    memcpy(Ffp,Fout,(size_t)S*DIM*4); }
    if(do_quant&&cfg!='F'&&ABLATE) for(int ci=0;ci<4;ci++){ Fabl[ci]=malloc((size_t)S*DIM*4); memcpy(Fabl[ci],shared,(size_t)S*DIM*4); }
    /* 并行专家循环 */
    int nth=NTHREADS; if(nth<1)nth=1; if(nth>NEXP)nth=NEXP;
    int e_next=0;
    ework_t *ws=calloc((size_t)nth,sizeof(ework_t));
    pthread_t *th=malloc((size_t)nth*sizeof(pthread_t));
    for(int t=0;t<nth;t++){
        ws[t].L=L;ws[t].S=S;ws[t].n_fit=n_fit;ws[t].do_quant=do_quant;ws[t].cfg=cfg;
        ws[t].Fin=Fin;ws[t].idx=idx;ws[t].rw_act=rw;ws[t].e_next=&e_next;
        ws[t].fout=calloc((size_t)S*DIM,4);
        ws[t].ffp = want_local? calloc((size_t)S*DIM,4) : NULL;
        if(do_quant&&cfg!='F'&&ABLATE) for(int ci=0;ci<4;ci++) ws[t].abl[ci]=calloc((size_t)S*DIM,4);
        pthread_create(&th[t],NULL,expert_worker,&ws[t]);
    }
    for(int t=0;t<nth;t++){
        pthread_join(th[t],NULL);
        for(size_t i=0;i<(size_t)S*DIM;i++) Fout[i]+=ws[t].fout[i];
        if(ws[t].ffp){ for(size_t i=0;i<(size_t)S*DIM;i++) Ffp[i]+=ws[t].ffp[i]; free(ws[t].ffp); }
        for(int ci=0;ci<4;ci++) if(ws[t].abl[ci]){ for(size_t i=0;i<(size_t)S*DIM;i++) Fabl[ci][i]+=ws[t].abl[ci][i]; free(ws[t].abl[ci]); }
        free(ws[t].fout);
        if(st){ st->calib_rows+=ws[t].calib_rows; st->calib_empty+=ws[t].calib_empty; st->nhit+=ws[t].nhit; }
    }
    free(ws);free(th);
    /* 局部质量 (held 行, routed-only R²=1-relL2², 减 FP shared): 旧口径诊断, 非判决 */
    if(want_local&&st){
        int s0=(n_fit>0&&n_fit<S)?n_fit:0; if(S-s0<1)s0=0;
        double sa=1e-12,se=1e-12; for(int s=s0;s<S;s++)for(int d=0;d<DIM;d++){ double A=Ffp[(size_t)s*DIM+d]-shared[(size_t)s*DIM+d]; sa+=A*A; se+=(double)shared[(size_t)s*DIM+d]*shared[(size_t)s*DIM+d]; }
        st->rs_ratio=sqrt(sa/se);   /* routed/shared 幅度比: 小 → 该层局部R²难看但对输出份额小(伪差层) */
        #define DQ_E2(F) ({ double e2=0; for(int s=s0;s<S;s++)for(int d=0;d<DIM;d++){ \
            double A=Ffp[(size_t)s*DIM+d]-shared[(size_t)s*DIM+d], B=(F)[(size_t)s*DIM+d]-shared[(size_t)s*DIM+d]; e2+=(B-A)*(B-A);} e2; })
        st->loc_r2=1.0-DQ_E2(Fout)/sa; st->have_loc=1;
        if(ABLATE&&Fabl[0]){
            double e[4]; for(int ci=0;ci<4;ci++){ e[ci]=DQ_E2(Fabl[ci]); st->abl_r2[ci]=1.0-e[ci]/sa; }
            st->abl_rel[0]=sqrt(e[0]/sa); st->abl_rel[1]=sqrt(e[2]/sa);
            int nh=S-s0;
            float *fh=malloc((size_t)nh*DIM*4),*gh=malloc((size_t)nh*DIM*4);
            for(int s=s0;s<S;s++)for(int d=0;d<DIM;d++){ fh[(size_t)(s-s0)*DIM+d]=Fabl[3][(size_t)s*DIM+d]-shared[(size_t)s*DIM+d];
                gh[(size_t)(s-s0)*DIM+d]=Ffp[(size_t)s*DIM+d]-shared[(size_t)s*DIM+d]; }
            float *wv=malloc((size_t)DIM*4); ds4_loss_dim_variance(gh,nh,DIM,wv);
            st->abl_la=ds4_loss_align(fh,gh,nh,DIM); st->abl_lc=ds4_loss_classify(fh,gh,wv,nh,DIM);
            free(wv);free(fh);free(gh);
        }
        #undef DQ_E2
    }
    if(shared)free(shared); if(Ffp)free(Ffp);
    for(int ci=0;ci<4;ci++) if(Fabl[ci])free(Fabl[ci]);
    }   /* end 非共适应路径 */
    /* ★纯VQ路径接入(2026-08-23)★: 纯 VQ 战役唯一的量化前向就是 B 回放(裸评撤),
     * cfg!='B' 禁入让序贯 z^L 在纯 VQ 下永不触发(ZLGATE=0 实锤)。B 禁入的两个理由
     * (活体每前向重解/修正不落盘假忠实)在"解一次+落盘 zrec+序贯"下都不成立:
     * 解一次由 zrec 幂等闸保证, 落盘走独立 zrec_L%02d.bin(引擎 type6/合并器直通),
     * dql 层文件不动。'F'(FP 锚遍)仍禁。 */
    /* ★搬到三分支合流点(2026-08-23): z 段原长在专家量化分支体内, B 回放分支(纯 VQ
     * 战役唯一前向)从不经过 → ZDIAG 无声实锤。coadapt 分支自带 z, 条件排除。 */
    int zrec_done=0;
    {
        const char*ld2=getenv("DS4_LAYER_DIR");
        if(ld2){ char zp2[1024]; snprintf(zp2,sizeof zp2,"%s/zrec_L%02d.bin",ld2,L);
                 FILE*zf2=fopen(zp2,"rb"); if(zf2){ fclose(zf2); zrec_done=1; } }
    }
    if(do_quant&&ANC_OK&&LZRANK>0&&cfg!='F'&&!zrec_done&&!(cfg=='g'&&COADAPT>0)){   /* ★'B'回放禁入已撤(见上) — 原注: 活体 z^L 每前向对锚重解会
        (a)把反修候选扰动拉回锚投影(橡皮筋, 候选逐位无效) (b)回放偷加不在文件的修正(合并模型没有→假忠实) */
        /* ★逐层动态 z^L(用户四支柱正确形态, 序贯锚定回拉)★: 输出端单点 z 要一口气补 43 层
         * 累积非线性误差(已证死路); z^L 每层只补【到本层为止的漂移】(小/局部/低秩可期), 分而治之。
         * hc_post 对 Fout 线性 → 目标在 Fout 空间闭式反解: ΔFout_td = Σ_j post_tj·(Hfp−Hq)_tjd / Σ_j post_tj²
         * (把本层出口整体拉回 FP 锚, 不论误差来自 MoE/attention/前层残差)。
         * z^L: Fin→ΔFout 的 dual-form ridge RRR + 四损失(align/classify/smooth感知/fixed)选秩 k_L;
         * 只用 fit 行拟合, 应用于全部行(held=泛化), 下层看到校正后激活(序贯, 同顺序量化哲学)。
         * 体积: Σk_L×2·DIM×fp16 → k=8 全层 ≈5.4MB(产物③可调秩侧车)。 */
        float *Hq=malloc((size_t)S*HCM*DIM*4);
        dq_hc_post(Fout,H2,post2,comb2,Hq,S,HCM,DIM);
        const float *Hf=ANC.H+(size_t)L*S*HCM*DIM;
        float *DF=malloc((size_t)S*DIM*4);
        if(BF_LT&&BF_LT_L==L){   /* ★B路序贯: 层局部靶(见 bf_fp_routed 注释); 非B路走原漂移靶 */
            memcpy(DF,BF_LT,(size_t)S*DIM*4);
        } else {
        for(int s=0;s<S;s++){
            const float *ps=post2+(size_t)s*HCM; double pd=1e-12;
            for(int j=0;j<HCM;j++) pd+=(double)ps[j]*ps[j];
            for(int d=0;d<DIM;d++){ double a2=0;
                for(int j=0;j<HCM;j++) a2+=(double)ps[j]*((double)Hf[((size_t)s*HCM+j)*DIM+d]-(double)Hq[((size_t)s*HCM+j)*DIM+d]);
                DF[(size_t)s*DIM+d]=(float)(a2/pd); }
        }
        }
        free(Hq);
        /* fit 内部再切 val: 前 vs 行拟合, [vs,n_fit) 选秩+GO门(k=0 可关本层 z); held 永不参与 */
        int vs=(n_fit*3)/4; int nval=n_fit-vs;
        if(nval<8){ vs=n_fit; nval=0; }
        /* 行帽已撤(2026-08-23): 2048 行全梯拒(过拟合), 行数是拟合质量的硬需求;
         * 时长改从 z_solve_dual 并行化拿(600GFLOP 单线程→20 线程)。 */
        ds4_z *zl=z_solve_dual(Fin,DF,(uint32_t)vs,DIM,DIM,(uint32_t)LZRANK,LZLAMBDA);
        if(zl){
            int kL = nval>0 ? z_pick_rank(zl,Fin+(size_t)vs*DIM,DF+(size_t)vs*DIM,nval,DIM,NULL,NULL,NULL,NULL)
                            : (int)zl->rank;
            /* ★真判据闸(2026-07-14 用户裁决"必须正向落地")★: z_pick_rank 只做候选秩提议
             * (其抽象四损失分 k=0 正则恒0 → k>0 先付~20%罚 → 恒选0 的结构病, 与四损失阶段同源);
             * 落地判据换成全项目统一口径: val 行本层出口 relL2 净降才落地。梯子 {提议k, 8, 4, 2, 1}
             * 逐个过闸, 首个净降者冻结为 zl.RRR 记录(可回放/可合并/引擎可执行的产物③)。 */
            int kland=0;
            /* ★梯子顶格跟 LZRANK(2026-08-23)★: 旧梯 {kL?:8,8,4,2,1} 顶格 8 —— z_pick_rank
             * 恒提 0(结构病自认)时 DS4_LZ>8 完全进不了落地梯, k=64 白解。 */
            /* ★从大到小(2026-08-23 用户"落地没有肉"): 旧序从提议 k 起步"够用即止",
             * kL=8 过闸就停 → 64 永不被试。最大肉 = 顶格先试, 首个净降即最大可落 k。 */
            { int cand0[7]={LZRANK,LZRANK/2,16,kL>0?kL:8,8,4,1}, tried[7]={0,0,0,0,0,0,0};
              size_t lst2=(size_t)S*HCM*DIM;
              float *Hq2=malloc(lst2*4), *Ftry=malloc((size_t)S*DIM*4), *zd=malloc((size_t)DIM*4);
              const float *Hf2=ANC.H+(size_t)L*lst2;
              /* 基线 val 出口 relL2 */
              double e0; { dq_hc_post(Fout,H2,post2,comb2,Hq2,S,HCM,DIM);
                double e2=0,a2=0;
                for(int s=vs;s<n_fit;s++){ const float*hq=Hq2+(size_t)s*HCM*DIM,*hf=Hf2+(size_t)s*HCM*DIM;
                    for(size_t i=0;i<(size_t)HCM*DIM;i++){ double d=(double)hq[i]-hf[i]; e2+=d*d; a2+=(double)hf[i]*hf[i]; } }
                e0=sqrt(e2/(a2+1e-30)); }
              { static double bf_e0_prev=-1.0;   /* ★链闸(2026-08-25)★: 逐层基线日志+失控守卫 */
                fprintf(stderr,"[链闸] L%02d 基线val出口=%.4f 前层=%.4f 靶=%s\n",
                        L,e0,bf_e0_prev,(BF_LT&&BF_LT_L==L)?"层局部":"漂移");
                if(bf_e0_prev>0.0&&e0>bf_e0_prev*1.5+0.05){
                    fprintf(stderr,"★[链闸] L%02d 基线暴涨 %.4f→%.4f (>1.5x+0.05) 链失稳 — 硬停★\n",L,bf_e0_prev,e0);
                    exit(7); }
                bf_e0_prev=e0; }
              for(int ci=0;ci<7&&!kland;ci++){ int kk=cand0[ci];
                if(kk<1||kk>(int)zl->rank) continue;
                int dup=0; for(int cj=0;cj<ci;cj++) if(cand0[cj]==kk&&tried[cj]) dup=1;
                if(dup) continue;
                tried[ci]=1;
                ds4_z_set_rank(zl,(uint32_t)kk);
                memcpy(Ftry,Fout,(size_t)S*DIM*4);
                for(int s=0;s<S;s++){
                    memset(zd,0,(size_t)DIM*4); ds4_z_apply(zl,Fin+(size_t)s*DIM,zd);
                    double nd=0,nf=0; const float*fo=Ftry+(size_t)s*DIM;
                    for(int d2=0;d2<DIM;d2++){ nd+=(double)zd[d2]*zd[d2]; nf+=(double)fo[d2]*fo[d2]; }
                    nd=sqrt(nd); nf=sqrt(nf);
                    double cap=LZTR*nf; float sc2=1.0f;
                    if(nd>cap&&nd>0) sc2=(float)(cap/nd);
                    float *fw=Ftry+(size_t)s*DIM;
                    for(int d2=0;d2<DIM;d2++) fw[d2]+=sc2*zd[d2];
                }
                double e1; { dq_hc_post(Ftry,H2,post2,comb2,Hq2,S,HCM,DIM);
                  double e2=0,a2=0;
                  for(int s=vs;s<n_fit;s++){ const float*hq=Hq2+(size_t)s*HCM*DIM,*hf=Hf2+(size_t)s*HCM*DIM;
                      for(size_t i=0;i<(size_t)HCM*DIM;i++){ double d=(double)hq[i]-hf[i]; e2+=d*d; a2+=(double)hf[i]*hf[i]; } }
                  e1=sqrt(e2/(a2+1e-30)); }
                printf("ZLGATE L=%d k=%d val出口relL2 %.6f→%.6f %s\n",L,kk,e0,e1,e1<e0-1e-9?"✓落地":"✗拒");
                fflush(stdout);
                if(e1<e0-1e-9){
                    kland=kk;
                    memcpy(Fout,Ftry,(size_t)S*DIM*4);   /* 序贯: 下层看到校正后激活 */
                    /* 冻结紧凑因子(活跃 k 列)暂存 → fwd_all 导出层文件后 append zl.RRR */
                    if(ZLP_U){ free(ZLP_U); free(ZLP_V); free(ZLP_Z); ZLP_U=ZLP_V=ZLP_Z=NULL; }
                    ZLP_K=kk; ZLP_L=L; ZLP_TR=(float)LZTR;
                    ZLP_Z=malloc((size_t)kk*4);
                    for(int c=0;c<kk;c++) ZLP_Z[c]=zl->z[c];
                    ZLP_U=malloc((size_t)DIM*kk*4); ZLP_V=malloc((size_t)DIM*kk*4);
                    for(int d2=0;d2<DIM;d2++) for(int c=0;c<kk;c++){
                        ZLP_U[(size_t)d2*kk+c]=zl->U[(size_t)d2*zl->rank+c];
                        ZLP_V[(size_t)d2*kk+c]=zl->V[(size_t)d2*zl->rank+c]; }
                    /* ★纯VQ路径: 独立 zrec 直写(dql 不动; 合并器/引擎 type6 直通) */
                    { const char*ld3=getenv("DS4_LAYER_DIR");
                      if(ld3){ char zp3[1024]; snprintf(zp3,sizeof zp3,"%s/zrec_L%02d.bin",ld3,L);
                        FILE*zf3=fopen(zp3,"wb");
                        if(zf3){ unsigned char hdr3[116]; memset(hdr3,0,116);
                          memcpy(hdr3,"zl.RRR",6);
                          unsigned long long psz3=16ull+2ull*((unsigned long long)kk+2ull*(unsigned long long)kk*DIM);
                          memcpy(hdr3+88,&psz3,8); int one3=1; memcpy(hdr3+112,&one3,4);
                          fwrite(hdr3,1,116,zf3);
                          uint32_t zk3=(uint32_t)kk,di3=(uint32_t)DIM,do3=(uint32_t)DIM; float tr3=(float)LZTR;
                          fwrite(&zk3,4,1,zf3); fwrite(&tr3,4,1,zf3); fwrite(&di3,4,1,zf3); fwrite(&do3,4,1,zf3);
                          uint16_t*h3=malloc(((size_t)kk+2*(size_t)kk*DIM)*2); size_t o3=0;
                          for(int i3=0;i3<kk;i3++) h3[o3++]=go1b_fp32_to_fp16(ZLP_Z[i3]);
                          for(size_t i3=0;i3<(size_t)DIM*kk;i3++) h3[o3++]=go1b_fp32_to_fp16(ZLP_U[i3]);
                          for(size_t i3=0;i3<(size_t)DIM*kk;i3++) h3[o3++]=go1b_fp32_to_fp16(ZLP_V[i3]);
                          fwrite(h3,2,o3,zf3); free(h3); fclose(zf3);
                          printf("ZREC L=%d k=%d → zrec_L%02d.bin\n",L,kk,L); fflush(stdout);
                        } } }
                }
              }
              free(Hq2); free(Ftry); free(zd);
            }
            /* ★拒层 GE 兜底(2026-08-23 用户令"抓不到肉是bug·先修bug")★
             * 哈希层(L1/L2 实锤)误差由离散专家身份决定, 连续 x 特征的 z^L 数学上抓不住
             * (L0 尚存 token 身份残留可修, L1+ 解耦后全梯拒)。对症=per-expert 门:
             * m*_s=<Fout+ΔF,Fout>/<Fout,Fout>(每 token 最优缩放) 按路由权重投到专家
             * (backfit form==4 同式), clamp[0.8,1.2]。评估用 ĝ_s(token 的 gate 加权平均门)
             * 近似 —— 专家间差异被平均抹平=低估改善, 过闸则真实改善≥评估(保守安全)。
             * 落地: Fout*=ĝ(序贯传链) + zrec 追加 bf.GE(引擎 type5 精确 per-expert 执行)。 */
            if(kland==0){
                int vs2=(n_fit*3)/4; int nval2=n_fit-vs2; if(nval2<8){ vs2=n_fit; nval2=0; }
                double *gnum=calloc((size_t)NEXP,sizeof(double)),*gden=calloc((size_t)NEXP,sizeof(double));
                for(int s2=0;s2<vs2;s2++){
                    const float*fo=Fout+(size_t)s2*DIM,*df=DF+(size_t)s2*DIM;
                    double num=0,den=1e-12;
                    for(int d2=0;d2<DIM;d2++){ num+=((double)fo[d2]+df[d2])*fo[d2]; den+=(double)fo[d2]*fo[d2]; }
                    double ms2=num/den;
                    for(int a2=0;a2<NACT;a2++){ int e2=idx[(size_t)s2*NACT+a2]; double w2=rw[(size_t)s2*NACT+a2];
                        if(e2>=0&&e2<NEXP&&w2>0){ gnum[e2]+=w2*ms2; gden[e2]+=w2; } }
                }
                float *gev=malloc((size_t)NEXP*4); int nge=0;
                for(int e2=0;e2<NEXP;e2++){ double g=gden[e2]>1e-12?gnum[e2]/gden[e2]:1.0;
                    /* clamp 放宽(2026-08-23 用户令"体积可以大一点·还原优先"):
                     * [0.8,1.2] 时代 L42 已修 8%, 夹断处即剩余肉 */
                    if(g<0.50)g=0.50; if(g>2.00)g=2.00; gev[e2]=(float)g; if(fabs(g-1.0)>1e-4)nge++; }
                free(gnum); free(gden);
                if(nge>=2&&nval2>0){
                    /* val 行评估: Ftry_s = ĝ_s·Fout_s → 出口 relL2 */
                    float *Ftry2=malloc((size_t)S*DIM*4),*Hq3=malloc((size_t)S*HCM*DIM*4);
                    memcpy(Ftry2,Fout,(size_t)S*DIM*4);
                    for(int s2=0;s2<S;s2++){
                        double gw=0,ww=1e-12;
                        for(int a2=0;a2<NACT;a2++){ int e2=idx[(size_t)s2*NACT+a2]; double w2=rw[(size_t)s2*NACT+a2];
                            if(e2>=0&&e2<NEXP&&w2>0){ gw+=w2*gev[e2]; ww+=w2; } }
                        float gh=(float)(gw/ww);
                        float*fw=Ftry2+(size_t)s2*DIM;
                        for(int d2=0;d2<DIM;d2++) fw[d2]*=gh;
                    }
                    const float *Hf3=ANC.H+(size_t)L*S*HCM*DIM;
                    double e0g,e1g;
                    { dq_hc_post(Fout,H2,post2,comb2,Hq3,S,HCM,DIM);
                      double e2s=0,a2s=0;
                      for(int s2=vs2;s2<n_fit;s2++){ const float*hq=Hq3+(size_t)s2*HCM*DIM,*hf=Hf3+(size_t)s2*HCM*DIM;
                          for(size_t i2=0;i2<(size_t)HCM*DIM;i2++){ double d3=(double)hq[i2]-hf[i2]; e2s+=d3*d3; a2s+=(double)hf[i2]*hf[i2]; } }
                      e0g=sqrt(e2s/(a2s+1e-30)); }
                    { dq_hc_post(Ftry2,H2,post2,comb2,Hq3,S,HCM,DIM);
                      double e2s=0,a2s=0;
                      for(int s2=vs2;s2<n_fit;s2++){ const float*hq=Hq3+(size_t)s2*HCM*DIM,*hf=Hf3+(size_t)s2*HCM*DIM;
                          for(size_t i2=0;i2<(size_t)HCM*DIM;i2++){ double d3=(double)hq[i2]-hf[i2]; e2s+=d3*d3; a2s+=(double)hf[i2]*hf[i2]; } }
                      e1g=sqrt(e2s/(a2s+1e-30)); }
                    printf("ZLGATE L=%d GE(路由投影) val出口relL2 %.6f→%.6f %s (活门=%d)\n",
                           L,e0g,e1g,e1g<e0g-1e-9?"✓落地":"✗拒",nge); fflush(stdout);
                    if(e1g<e0g-1e-9){
                        memcpy(Fout,Ftry2,(size_t)S*DIM*4);   /* 序贯: ĝ 近似传链 */
                        const char*ld4=getenv("DS4_LAYER_DIR");
                        if(ld4){ char zp4[1024]; snprintf(zp4,sizeof zp4,"%s/zrec_L%02d.bin",ld4,L);
                            FILE*zf4=fopen(zp4,"ab");   /* 追加(层可同时有 zl.RRR + bf.GE; 拒层=仅 GE) */
                            if(zf4){ unsigned char hdr4[116]; memset(hdr4,0,116);
                                memcpy(hdr4,"bf.GE",5);
                                unsigned long long psz4=(unsigned long long)NEXP*2;
                                memcpy(hdr4+88,&psz4,8); int one4=1; memcpy(hdr4+112,&one4,4);
                                fwrite(hdr4,1,116,zf4);
                                uint16_t geh4[NEXP];
                                for(int e2=0;e2<NEXP;e2++) geh4[e2]=go1b_fp32_to_fp16(gev[e2]);
                                fwrite(geh4,2,NEXP,zf4); fclose(zf4);
                                printf("ZREC L=%d GE → zrec_L%02d.bin(bf.GE)\n",L,L); fflush(stdout);
                            } }
                    }
                    free(Ftry2); free(Hq3);
                }
                free(gev);
            }
            if(st) st->zk=kland;
            if(kland>0) LZ_TOTAL_K+=kland;
            ds4_z_free(zl);
        }
        free(DF);
    }
    dq_hc_post(Fout,H2,post2,comb2,H,S,HCM,DIM);   /* 就地更新 H(误差传下层=累积) */
    free(cosr);free(sinr);free(y);free(post);free(comb);free(xn);if(kvc)free(kvc);free(a);free(H2);free(y2);free(post2);free(comb2);free(Fin);free(idx);free(rw);free(Fout);
}

/* head: H[S,HCM,DIM] → logits[S,VOCAB] (vocab 投影批量 sgemm) */
static void head_fwd(float*H,int S,float*hcfn,float*hcb,float*hcs,float*norm,float*hw,float*logits){
    int HD_=HCM*DIM;
    float *yn=malloc((size_t)S*DIM*4);
    for(int s=0;s<S;s++){ const float*x=H+(size_t)s*HD_; double v=0;for(int i=0;i<HD_;i++)v+=(double)x[i]*x[i];float rsq=(float)(1.0/sqrt(v/HD_+EPSF));
        float mix[8],pre[8]; for(int j=0;j<HCM;j++){const float*fr=hcfn+(size_t)j*HD_;float aa=0;for(int k=0;k<HD_;k++)aa+=x[k]*fr[k];mix[j]=aa*rsq;pre[j]=dq_sigmoid(mix[j]*hcs[0]+hcb[j])+EPSF;}
        float yv[DIM]; for(int d=0;d<DIM;d++){float aa=0;for(int j=0;j<HCM;j++)aa+=pre[j]*H[((size_t)s*HCM+j)*DIM+d];yv[d]=aa;}
        dq_rms(yv,norm,yn+(size_t)s*DIM,DIM,EPSF);
    }
    dq_matmul(yn,hw,logits,S,DIM,VOCAB);
    free(yn);
}

/* head_fwd 低内存版(2026-07-28 里程碑尖刺修): head.weight [VOCAB,DIM] 按行块流式
 * 直读(BF16/F32), 峰值 ~250MB vs 全量 fp32 2.1G(看门狗 12.3G 击杀根因)。 */
static void head_fwd_stream(float*H,int S,float*hcfn,float*hcb,float*hcs,float*norm,float*logits){
    int HD_=HCM*DIM;
    float *yn=malloc((size_t)S*DIM*4);
    for(int s=0;s<S;s++){ const float*x=H+(size_t)s*HD_; double v=0;for(int i=0;i<HD_;i++)v+=(double)x[i]*x[i];float rsq=(float)(1.0/sqrt(v/HD_+EPSF));
        float mix[8],pre[8]; for(int j=0;j<HCM;j++){const float*fr=hcfn+(size_t)j*HD_;float aa=0;for(int k=0;k<HD_;k++)aa+=x[k]*fr[k];mix[j]=aa*rsq;pre[j]=dq_sigmoid(mix[j]*hcs[0]+hcb[j])+EPSF;}
        float yv[DIM]; for(int d=0;d<DIM;d++){float aa=0;for(int j=0;j<HCM;j++)aa+=pre[j]*H[((size_t)s*HCM+j)*DIM+d];yv[d]=aa;}
        dq_rms(yv,norm,yn+(size_t)s*DIM,DIM,EPSF);
    }
    char shard[256]; long ds2; char dt[16]; long shp[2],off[2];
    if(!st_shard(&C,"head.weight",shard)){ fprintf(stderr,"[head-stream] shard 缺失\n"); exit(2); }
    char *hdr=st_shard_hdr(&C,shard,&ds2);
    if(!hdr||!st_find(hdr,"head.weight",dt,shp,off)){ fprintf(stderr,"[head-stream] 头解析失败\n"); exit(2); }
    int isf32=strcmp(dt,"F32")==0;
    char p[1300]; snprintf(p,sizeof(p),"%s/%s",C.hf,shard);
    FILE*f=fopen(p,"rb");
    const int BS=8192;
    float *blk=malloc((size_t)BS*DIM*4),*tmp=malloc((size_t)S*BS*4);
    uint16_t *b16=isf32?NULL:malloc((size_t)BS*DIM*2);
    for(long v0=0;v0<VOCAB;v0+=BS){
        long bs=VOCAB-v0<BS?VOCAB-v0:BS;
        fseek(f,ds2+off[0]+(long)((size_t)v0*DIM*(isf32?4:2)),SEEK_SET);
        if(isf32){ if(fread(blk,4,(size_t)bs*DIM,f)!=(size_t)bs*DIM){fprintf(stderr,"[head-stream] 读断\n");exit(2);} }
        else{ if(fread(b16,2,(size_t)bs*DIM,f)!=(size_t)bs*DIM){fprintf(stderr,"[head-stream] 读断\n");exit(2);}
              for(size_t i=0;i<(size_t)bs*DIM;i++){ uint32_t u=(uint32_t)b16[i]<<16; memcpy(&blk[i],&u,4);} }
        dq_matmul(yn,blk,tmp,S,DIM,(int)bs);
        for(int s=0;s<S;s++) memcpy(logits+(size_t)s*VOCAB+v0,tmp+(size_t)s*bs,(size_t)bs*4);
    }
    fclose(f); free(hdr); free(blk); free(tmp); if(b16)free(b16); free(yn);
}

/* 向后·终端反调的 val 行 KL(fp‖q): 选 t 只用 fit 尾部 val 行, held 永不参与 */
static double bwd_val_kl(const float*lf,const float*lq,int a,int b){
    double kl=0; int n=0;
    for(int s=a;s<b;s++){
        const float*x=lf+(size_t)s*VOCAB,*y=lq+(size_t)s*VOCAB;
        float ma=x[0],mb=y[0];
        for(int v=1;v<VOCAB;v++){ if(x[v]>ma)ma=x[v]; if(y[v]>mb)mb=y[v]; }
        double sa=0,sb=0;
        for(int v=0;v<VOCAB;v++){ sa+=exp((double)x[v]-ma); sb+=exp((double)y[v]-mb); }
        double lsa=log(sa),lsb=log(sb),krow=0;
        for(int v=0;v<VOCAB;v++){
            double lpf=(double)x[v]-ma-lsa, lpq=(double)y[v]-mb-lsb;
            double pf=exp(lpf); if(pf>0) krow+=pf*(lpf-lpq); }
        kl+=krow; n++;
    }
    return n?kl/n:0.0;
}
/* 单 token 行 KL(P_fp||P_q) — 反修按 token 挑最优乘子用 */
static double bwd_tok_kl(const float*x,const float*y){
    float ma=x[0],mb=y[0];
    for(int v=1;v<VOCAB;v++){ if(x[v]>ma)ma=x[v]; if(y[v]>mb)mb=y[v]; }
    double sa=0,sb=0;
    for(int v=0;v<VOCAB;v++){ sa+=exp((double)x[v]-ma); sb+=exp((double)y[v]-mb); }
    double lsa=log(sa),lsb=log(sb),k=0;
    for(int v=0;v<VOCAB;v++){ double lpf=(double)x[v]-ma-lsa,lpq=(double)y[v]-mb-lsb;
        double pf=exp(lpf); if(pf>0)k+=pf*(lpf-lpq); }
    return k;
}
/* 对称正定线性系统部分主元高斯消元(A n×n 行主序, 破坏性); 解入 x。ridge 已由调用方加对角。*/
static int solve_sym(double*A,double*b,int n,double*x){
    for(int c=0;c<n;c++){
        int piv=c; double best=fabs(A[(size_t)c*n+c]);
        for(int r=c+1;r<n;r++){ double v=fabs(A[(size_t)r*n+c]); if(v>best){best=v;piv=r;} }
        if(best<1e-18) return -1;
        if(piv!=c){ for(int j=0;j<n;j++){ double t=A[(size_t)c*n+j];A[(size_t)c*n+j]=A[(size_t)piv*n+j];A[(size_t)piv*n+j]=t; }
                    double t=b[c];b[c]=b[piv];b[piv]=t; }
        double d=A[(size_t)c*n+c];
        for(int r=c+1;r<n;r++){ double f=A[(size_t)r*n+c]/d; if(f==0)continue;
            for(int j=c;j<n;j++) A[(size_t)r*n+j]-=f*A[(size_t)c*n+j]; b[r]-=f*b[c]; }
    }
    for(int r=n-1;r>=0;r--){ double s=b[r];
        for(int j=r+1;j<n;j++) s-=A[(size_t)r*n+j]*x[j];
        x[r]=s/A[(size_t)r*n+r]; }
    return 0;
}
static void export_layer_file(int L,int S,int n_fit,const char*lf);
static int backfit_prev(int Lfront,const long*ids,int S,int n_fit);   /* 逐层反修(定义在 global_sweep 前) */
static float *gs_forward_exit(int J,int Lend,const float*Hin,const long*ids,int S,int n_fit,float*HQcache);   /* BF_ONLY 回放推进用 */
static double held_score(const float*Hq,int S,int n_fit,int L,double*relh_out);   /* 反修内动态α用(定义在贪心段) */
static float *fwd_all(const long*ids,int S,int n_fit,int do_quant,const char*lcfg){
    size_t lstride=(size_t)S*HCM*DIM;
    if(!RB_SEQ&&getenv("DS4_ROUTE_SEQ")) RB_SEQ=1;   /* 反修路径也需 per-layer α(minvol 分支之外) */
    float *H=malloc(lstride*4);
    int L0=0;
    if(do_quant&&ANC_OK){   /* F 前缀 = FP 锚定原样, 直接恢复跳过 */
        int p=0; while(p<NLAYERS&&lcfg[p]=='F')p++;
        if(p>0){ memcpy(H,ANC.H+(size_t)(p-1)*lstride,lstride*4); L0=p;
            fprintf(stderr,"[前缀] L0..L%d 全 F → anchor 恢复, 从 L%d 开跑\n",p-1,L0<NLAYERS?L0:NLAYERS-1); }
    }
    if(L0==0){
        float *emb=st_read_weight(&C,"embed.weight",NULL,NULL);
        for(int s=0;s<S;s++)for(int j=0;j<HCM;j++)memcpy(H+((size_t)s*HCM+j)*DIM,emb+(size_t)ids[s]*DIM,(size_t)DIM*4);
        free(emb);
    }
    if(do_quant) fprintf(stderr,"\n判决=最终输出; 逐层为诊断: 累积=H_q vs FP锚定(误差爆炸在哪层) 局部=该层专家复现(旧口径)\n");
    /* ★逐层反修★缓存: 骨干权重(留存不 free)+层文件(export 后开)+量化态入口隐藏; 前沿层完成即反修 0..L-1 */
    int incr = do_quant && ANC_OK && COADAPT>0 && BACKFIT_INCR && getenv("DS4_LAYER_DIR") && lcfg;
    /* ★只跑反修(2026-07-13, 用户裁决: "加个参数只跑返修")★: 推进段不重跑 SEARCH —
     * 逐层加载既有 dql 按 op 链字节回放推进累积态+建 HQE/GS_LF/GS_LW, 直达终局收敛 sweep。
     * 前提=层文件全齐(缺一层硬停); 已落地的 bf.* 修正随层文件一并回放(在其上继续叠加)。 */
    int BF_ONLY = incr && !FAST && getenv("DS4_BF_ONLY") && atoi(getenv("DS4_BF_ONLY"));
    if(BF_ONLY) fprintf(stderr,"[只跑反修] 开: 复用既有层文件(SEARCH 跳过), 回放推进 → 终局收敛 sweep\n");
    if(incr){
        if(!GS_LW) GS_LW=calloc((size_t)NLAYERS,sizeof(LWH));
        if(!GS_LF) GS_LF=calloc((size_t)NLAYERS,sizeof(lfile_t));
        if(!HQE){
            /* ★反修内存架构: HQE 快照(S=1716 时 ~2.5G)由 malloc(脏页)改为 /tmp 匿名文件后备
             * mmap(MAP_SHARED) — 有磁盘后备可换出, 内存压力下自回收, 与锚 mmap 合计砍反修
             * 基线脏内存 ~7G ⇒ 终局 sweep 进 12G 红线(2026-07-31 用户令核心价值必须反修)。 */
            char hqp[]="/tmp/ds4_hqe_XXXXXX"; int hfd=mkstemp(hqp);
            if(hfd<0){ perror("hqe-tmp"); exit(1); }
            unlink(hqp);
            size_t hb=(size_t)(NLAYERS+1)*lstride*4;
            if(ftruncate(hfd,(off_t)hb)!=0){ perror("hqe-trunc"); exit(1); }
            HQE=mmap(NULL,hb,PROT_READ|PROT_WRITE,MAP_SHARED,hfd,0);
            close(hfd);
            if(HQE==MAP_FAILED){ perror("hqe-mmap"); exit(1); }
            g_hqe_lstride=lstride;
            fprintf(stderr,"[反修] HQE 快照 %.2f GiB → 文件后备 mmap(可换出)\n",hb/1073741824.0);
        }
        if(!BF_FINOP){ BF_FINOP=malloc((size_t)NLAYERS*sizeof(int)); for(int i=0;i<NLAYERS;i++) BF_FINOP[i]=-1; }
        if(!BF_DYN2OP){ BF_DYN2OP=malloc((size_t)NLAYERS*sizeof(int)); for(int i=0;i<NLAYERS;i++) BF_DYN2OP[i]=-1; }
        if(!BF_GEOP){ BF_GEOP=malloc((size_t)NLAYERS*sizeof(int)); for(int i=0;i<NLAYERS;i++) BF_GEOP[i]=-1;
            BF_HCOP=malloc((size_t)NLAYERS*sizeof(int)); for(int i9=0;i9<NLAYERS;i9++) BF_HCOP[i9]=-1; }
        if(!GS_IDXC){ GS_IDXC=malloc((size_t)S*NACT*sizeof(int)); GS_RWC=malloc((size_t)S*NACT*4); }
        if(getenv("DS4_BF_TERMINAL")&&atoi(getenv("DS4_BF_TERMINAL"))==0)
                 fprintf(stderr,"[逐层反修] 开(逐前沿·旧): 每前沿全量反修所有前层; 判据=前沿出口 vs FP锚(移动代理, O(L³))\n");
        else     fprintf(stderr,"[逐层反修] 开(终局收敛): 推进段不修(误差前向吸收=部署口径), 收尾以最终出口为判据全层 sweep 到无落地(O(K·L²)); 旧逐前沿语义用 DS4_BF_TERMINAL=0\n");
    }
    for(int L=L0;L<NLAYERS;L++){
        LW W=load_layer2(L,incr?&GS_LW[L]:NULL); lstat_t st; memset(&st,0,sizeof(st));   /* incr: 顺带录元素数(fp16缓存用) */
        fprintf(stderr,"L%02d %c ",L,do_quant?lcfg[L]:'@');
        if(incr&&HQE){ memcpy(HQE+(size_t)L*lstride,H,lstride*4);   /* 层 L 量化态入口(反修从此重前向; fast 无 HQE) */
            msync(HQE+(size_t)L*lstride,lstride*4,MS_ASYNC); }   /* 异步回写(内存卫生) */
        if(BF_ONLY&&do_quant&&lcfg[L]=='g'){
            /* 只跑反修: 加载既有 dql(含已落地 bf.* 修正)→ 绑定链末 op 槽位 → 字节回放本层 */
            lwh_absorb(&GS_LW[L],&W);
            char lp0[512]; snprintf(lp0,sizeof(lp0),"%s/dql_L%02d.bin",getenv("DS4_LAYER_DIR"),L);
            if(GS_LF[L].map){ lfile_free(&GS_LF[L]); memset(&GS_LF[L],0,sizeof(lfile_t)); }
            if(lfile_load(lp0,&GS_LF[L])!=0){
                fprintf(stderr,"[只跑反修] L%02d 层文件缺失/损坏(%s) — 需 %d 层全齐, 硬停\n",L,lp0,NLAYERS);
                exit(9);
            }
            for(int i=GS_LF[L].nops-1;i>=0;i--){ int t=GS_LF[L].ops[i].type;   /* 链末槽位: 反修"已有则原地重解/累乘"口径 */
                if(t==1&&BF_FINOP[L]<0) BF_FINOP[L]=i;
                else if(t==2&&BF_DYN2OP[L]<0) BF_DYN2OP[L]=i;
                else if(t==5&&BF_GEOP[L]<0) BF_GEOP[L]=i; }
            float*He=gs_forward_exit(L,L,H,ids,S,n_fit,NULL);
            memcpy(H,He,lstride*4); free(He);
            memcpy(HQE+(size_t)(L+1)*lstride,H,lstride*4);
            msync(HQE+(size_t)(L+1)*lstride,lstride*4,MS_ASYNC);   /* 异步回写(内存卫生) */
            if(g_chfd>=0&&S==g_chS){   /* ★链态锚: 层出口 H 直写★ */
                size_t bH=40+(size_t)NLAYERS*S*DIM*4+2*(size_t)NLAYERS*S*NACT*4;
                pwrite(g_chfd,H,lstride*4,(off_t)(bH+(size_t)L*lstride*4));
                fdatasync(g_chfd); }   /* 内存卫生: 大 S 时脏页尽早落盘可回收(2026-08-24; 当日 OOM 真因=两条重任务并行, 非本处) */
            { const float *Hf=ANC.H+(size_t)L*lstride;   /* 累积诊断(与正常路径同口径) */
              size_t rowsz0=(size_t)HCM*DIM, fitsz=(size_t)n_fit*rowsz0;
              double e2=0,a2=0,e2h=0,a2h=0;
              for(size_t i=0;i<fitsz;i++){ double d=(double)H[i]-Hf[i]; e2+=d*d; a2+=(double)Hf[i]*Hf[i]; }
              for(size_t i=fitsz;i<lstride;i++){ double d=(double)H[i]-Hf[i]; e2h+=d*d; a2h+=(double)Hf[i]*Hf[i]; }
              fprintf(stderr," | 回放累积relL2 fit=%.4f held=%.4f | ops=%d\n",
                      sqrt(e2/(a2+1e-30)),sqrt(e2h/(a2h+1e-30)),GS_LF[L].nops); }
            fprintf(stderr,"[mem] L%02d footprint=%.2fGB 只跑反修回放\n",L,mem_gb());
            gs_lw_evict(L-6>0?L-6:0);
            free_layer(&W);
            continue;
        }
        if(bf_from_bytes()&&do_quant&&lcfg&&lcfg[L]=='g'&&getenv("DS4_LAYER_DIR")){
            char bfp[512]; snprintf(bfp,sizeof(bfp),"%s/dql_L%02d.bin",getenv("DS4_LAYER_DIR"),L);
            if(!BF_LFF_ON&&lfile_load(bfp,&BF_LFF)==0&&BF_LFF.vqmap&&BF_LFF.w2){
                BFB_VQMAP=BF_LFF.vqmap; BFB_VQMSZ=BF_LFF.vqmsz; BFB_W2=BF_LFF.w2; BFB_SZD=BF_LFF.szD; BF_LFF_ON=1;
                if(BFB_W2COPY){ free(BFB_W2COPY); BFB_W2COPY=NULL; }
                BFB_W2COPY=malloc((size_t)NEXP*BF_LFF.szD);
                if(BFB_W2COPY) memcpy(BFB_W2COPY,BF_LFF.w2,(size_t)NEXP*BF_LFF.szD);
                fprintf(stderr,"[反修字节起步] L%02d 冷基座=盘上字节(vq blob + dql D 段, w2快照%.0fMB)\n",
                        L,(double)NEXP*BF_LFF.szD/1048576.0);
            }
        }
        layer_fwd(L,&W,H,ids,S,n_fit,do_quant,do_quant?lcfg[L]:'F',&st);
        if(BF_LFF_ON){ BFB_VQMAP=NULL; BFB_W2=NULL; lfile_free(&BF_LFF);
            memset(&BF_LFF,0,sizeof(BF_LFF)); BF_LFF_ON=0; }   /* 导出重写(截断)前必须解除 mmap */
        if(ANC_BUILD) memcpy(ANC.H+(size_t)L*lstride,H,lstride*4);
        if(do_quant&&ANC_OK){
            /* 累积偏差拆 fit/held: z^L 的优化目标=压低 fit 行, held 列才是逐层可见的泛化真相 */
            const float *Hf=ANC.H+(size_t)L*lstride;
            size_t rowsz=(size_t)HCM*DIM, fitsz=(size_t)n_fit*rowsz;
            double e2=0,a2=0,e2h=0,a2h=0;
            for(size_t i=0;i<fitsz;i++){ double d=(double)H[i]-Hf[i]; e2+=d*d; a2+=(double)Hf[i]*Hf[i]; }
            for(size_t i=fitsz;i<lstride;i++){ double d=(double)H[i]-Hf[i]; e2h+=d*d; a2h+=(double)Hf[i]*Hf[i]; }
            double rel=sqrt(e2/(a2+1e-30)), relh=sqrt(e2h/(a2h+1e-30));
            fprintf(stderr," | 累积relL2 fit=%.4f held=%.4f | 路由一致=%5.1f%%",rel,relh,st.agree);
            if(lcfg[L]!='F'){
                fprintf(stderr," | 校准µ=%.1f行 空=%d/%d",st.nhit?st.calib_rows/st.nhit:0.0,st.calib_empty,st.nhit);
                if(st.have_loc) fprintf(stderr," | 局部R²=%5.1f%% R/S=%.2f",st.loc_r2*100.0,st.rs_ratio);
                if(LZRANK>0) fprintf(stderr," | z^L k=%d",st.zk);
                fprintf(stderr," | LS修[负%ld 爆%ld 败%ld /%ld]",go1b_joint_neg,go1b_joint_clip,go1b_joint_fail,go1b_joint_rows);
            }
            fputc('\n',stderr);
            go1b_joint_neg=go1b_joint_clip=go1b_joint_fail=go1b_joint_rows=0;   /* 计数按层清零 */
            if(ABLATE&&lcfg[L]!='F'&&st.have_loc)
                fprintf(stderr,"      消融R²(局部): 朴素%5.1f%% joint%5.1f%% z%5.1f%% Q2残差%5.1f%% | relL2 朴素%.3f z%.3f | 损[al %.3f cl %.3f]\n",
                        st.abl_r2[0]*100,st.abl_r2[1]*100,st.abl_r2[2]*100,st.abl_r2[3]*100,st.abl_rel[0],st.abl_rel[1],st.abl_la,st.abl_lc);
        } else fprintf(stderr," | anchor 捕获\n");
        if(incr){ lwh_absorb(&GS_LW[L],&W);          /* incr: 骨干 fp16 进缓存(fp32 即刻还, 防单调涨爆12G) */
            fprintf(stderr,"[mem] L%02d footprint=%.2fGB fp16缓存至本层\n",L,mem_gb());
            gs_lw_evict(L-6>0?L-6:0);                /* 超预算驱逐最远层(近6层是每条链的尾巴, 不驱逐) */
        }
        free_layer(&W);
        if(do_quant&&COADAPT>0&&lcfg&&lcfg[L]=='g'&&getenv("DS4_LAYER_DIR")){
            char lp[512]; snprintf(lp,sizeof(lp),"%s/dql_L%02d.bin",getenv("DS4_LAYER_DIR"),L);
            export_layer_file(L,S,n_fit,lp);   /* 每层产物当场留存(可单层重跑微调, 最后合并) */
            if(BFB_W2COPY){ free(BFB_W2COPY); BFB_W2COPY=NULL; }   /* 反修复用快照层内即弃 */
            /* (z^L 落盘已并入 export_layer_file 正位直写 op 侧车 — 旧 append 块删除, 错序温床根除) */
            if(incr){                          /* ★逐层前进即反修前面所有层(用户设计)★ */
                if(GS_LF[L].map){ lfile_free(&GS_LF[L]); memset(&GS_LF[L],0,sizeof(lfile_t)); }
                lfile_load(lp,&GS_LF[L]);                        /* 本层文件进缓存 */
                zc_opt_emit(L,&GS_LF[L]);      /* 一层两份: dql(量化)+opt(优化)并排落盘; 终值由 zchain_write 刷新 */
                memcpy(HQE+(size_t)(L+1)*lstride,H,lstride*4);   /* 层 L 量化态出口(终局sweep重放起点; fast 也反修) */
                if(!FAST&&getenv("DS4_BF_TERMINAL")&&atoi(getenv("DS4_BF_TERMINAL"))==0){
                    /* 旧·逐前沿反修: 判据=前沿出口(移动代理, 同层随前沿推进被反复翻修 → O(L³)) */
                    int chg=backfit_prev(L,ids,S,n_fit);
                    if(chg) memcpy(H,HQE+(size_t)(L+1)*lstride,lstride*4);  /* 用反修后累积态继续前进 */
                }
            }
        }
    }
    if(do_quant&&incr&&HQE&&GS_LF&&NLAYERS>1
       &&!(getenv("DS4_BF_TERMINAL")&&atoi(getenv("DS4_BF_TERMINAL"))==0)){   /* 2026-07-14: fast 也跑终局反修(用户裁决) */
        /* ★终局收敛反修(2026-07-13)★: 判据=最终层出口 vs FP锚(真目标), 全层 sweep 循环到无落地。
         * 取代逐前沿 O(L³): 前沿判据是移动代理靶(同层随推进被反复翻修, 增量互相覆盖), 推进段的
         * 误差本就由下游各层自适应求解前向吸收(部署口径); 终局判据下每份修正只做一次、直指真目标。
         * 复杂度 O(K·L²), 实测类坐标下降 2-3 轮即干; DS4_BF_TERM_MAXP 护栏防不收敛。 */
        int maxp=getenv("DS4_BF_TERM_MAXP")?atoi(getenv("DS4_BF_TERM_MAXP")):1;   /* 默认=用户设计: 末层反修一遍(含复检)即止; 实测第2轮2.7h只换-0.7%, 多轮重扫默认不开 */
        for(int p=0;p<maxp;p++){
            int ch=backfit_prev(NLAYERS-1,ids,S,n_fit);
            printf("BACKFIT_TERM pass=%d 落地=%d%s\n",p,ch,ch?"":" → 收敛"); fflush(stdout);
            if(!ch) break;
        }
        memcpy(H,HQE+(size_t)NLAYERS*lstride,lstride*4);   /* 反修后最终出口 → 下方 logits/BWDFIN 同口径 */
    }
    if(getenv("DS4_DUMPH")){ const char*p=do_quant?"/tmp/hq.bin":"/tmp/hfp.bin"; FILE*fp=fopen(p,"wb");
        if(fp){ int hd[3]={S,HCM,DIM}; fwrite(hd,4,3,fp); fwrite(H,4,lstride,fp); fclose(fp); } }  /* final hidden 捕获 */
    float *hcfn=st_read_weight(&C,"hc_head_fn",NULL,NULL),*hcb=st_read_weight(&C,"hc_head_base",NULL,NULL),*hcs=st_read_weight(&C,"hc_head_scale",NULL,NULL);
    float *norm=st_read_weight(&C,"norm.weight",NULL,NULL),*hw=st_read_weight(&C,"head.weight",NULL,NULL);
    float *logits=malloc((size_t)S*VOCAB*4); head_fwd(H,S,hcfn,hcb,hcs,norm,hw,logits);
    if(do_quant&&ANC_OK&&BWD_L>=0&&getenv("DS4_BWD_FINAL")){
        /* 向后·终端反调: 层出口 H 对修正量线性 → H(t)=lerp(H_base,H_corr,t);
         * t 网格重跑后缀+head 到最终 logits, val 行(fit 尾 1/4)选 t, held 行不参与选择 */
        int vsq=(n_fit*3)/4;
        double kl1=bwd_val_kl(ANC.logits,logits,vsq,n_fit);
        printf("BWDFIN L=%d t=1.00 val行KL=%.4f (基准)\n",BWD_L,kl1); fflush(stdout);
        double bestkl=kl1; float bestt=1.0f; float *bestlog=NULL;
        const float TT[3]={0.85f,1.15f,1.30f};
        for(int ti=0;ti<3;ti++){ float t=TT[ti];
            if(tune_over()){ printf("BWDFIN L=%d 时间预算耗尽 → 跳过剩余 t 候选\n",BWD_L); fflush(stdout); break; }
            float *Hb2=malloc(lstride*4);
            for(size_t i=0;i<lstride;i++) Hb2[i]=BWD_Hb[i]+t*(BWD_Hc[i]-BWD_Hb[i]);
            for(int l=BWD_L+1;l<NLAYERS;l++){ LW W2=load_layer(l);
                { char pg[32],ag[32]; snprintf(pg,sizeof(pg),"后缀%d/%d",l,NLAYERS-1);
                  snprintf(ag,sizeof(ag),"终端反调t=%.2f",t);
                  mlog(BWD_L,"向后",ag,pg,4,"待测","进行中"); }
                layer_fwd(l,&W2,Hb2,ids,S,n_fit,0,'F',NULL); free_layer(&W2); }
            float *lg=malloc((size_t)S*VOCAB*4); head_fwd(Hb2,S,hcfn,hcb,hcs,norm,hw,lg);
            double kl=bwd_val_kl(ANC.logits,lg,vsq,n_fit);
            printf("BWDFIN L=%d t=%.2f val行KL=%.4f%s\n",BWD_L,t,kl,kl<bestkl?" ✓":""); fflush(stdout);
            if(kl<bestkl){ bestkl=kl; bestt=t; free(bestlog); bestlog=lg; } else free(lg);
            free(Hb2);
        }
        BWDFIN_T=bestt;
        el_add("bwd.final","H(t)线性插值+后缀重前向, final-logits val行KL 选 t",4,
               kl1,bestkl,kl1>1e-12?100.0*(kl1-bestkl)/kl1:0.0,bestt,bestt!=1.0f?1:3);
        if(bestt!=1.0f&&getenv("DS4_LAYER_DIR")){
            /* ★落地缺口修复★: 判决用了 H(t) 但此前只写台账行 → 文件回放/DQZ2/运行时全缺此项
             * (t=1 无实害, t≠1 判决虚高)。hc_post 对 F 线性 ⇒ H(t)=lerp ≡ 末层 TREF(锚 Fcur),
             * 名含 TREF → lfile_load/zchain/引擎三方零改动直接认。 */
            char lpB[512]; op_host_path(BWD_L,lpB,sizeof(lpB));   /* 平行架构: 落地进 op 侧车 */
            if(!append_rec(lpB,"bwd.TREF.fin","终端反调 H(t)=lerp ≡ 末层TREF(锚Fcur)",&bestt,4,(float)bestkl))
                fprintf(stderr,"[bwdfin] 落地追加失败: %s (文件与判决将不一致!)\n",lpB);
        }
        printf("BWDFIN_BEST L=%d t=%.2f val行KL %.4f→%.4f 提升%.1f%%\n",
               BWD_L,bestt,kl1,bestkl,kl1>1e-12?100.0*(kl1-bestkl)/kl1:0.0);
        fflush(stdout);
        if(bestlog){ free(logits); logits=bestlog; }
    }
    /* HQE 现为文件后备 mmap(2026-07-31 反修内存改造) — 必须 munmap 不能 free
     * (漏改这里 = Abort trap 6, 终局 sweep 收敛后清理阶段崩, 实测踩过一次) */
    if(HQE){ munmap(HQE,(size_t)(NLAYERS+1)*(size_t)g_hqe_lstride*4); HQE=NULL; }
    if(BF_FINOP){ free(BF_FINOP); BF_FINOP=NULL; }
    if(BF_DYN2OP){ free(BF_DYN2OP); BF_DYN2OP=NULL; }
    if(BF_GEOP){ free(BF_GEOP); BF_GEOP=NULL; }
    if(BF_HCOP){ free(BF_HCOP); BF_HCOP=NULL; }
    if(GS_IDXC){ free(GS_IDXC); GS_IDXC=NULL; } if(GS_RWC){ free(GS_RWC); GS_RWC=NULL; }
    if(RB_ACC&&getenv("DS4_ROUTE_BIAS_FIT")){   /* ★路由偏置侧车: 均值化+落盘★ */
        const char*rp=getenv("DS4_ROUTE_BIAS_FIT");
        float *rbo=malloc((size_t)NLAYERS*NEXP*4); long rbtot=0;
        for(size_t i=0;i<(size_t)NLAYERS*NEXP;i++){ rbo[i]=RB_CNT[i]?RB_ACC[i]/(float)RB_CNT[i]:0.0f; rbtot+=RB_CNT[i]; }
        FILE*rbf=fopen(rp,"wb");
        if(rbf){ uint32_t hd[4]={0x41494252u,(uint32_t)NLAYERS,(uint32_t)NEXP,0};
            fwrite(hd,4,4,rbf); fwrite(rbo,4,(size_t)NLAYERS*NEXP,rbf);
            fwrite(RB_CNT,4,(size_t)NLAYERS*NEXP,rbf); fclose(rbf);
            printf("ROUTE_BIAS fit → %s (margin事件=%ld)\n",rp,rbtot); }
        free(rbo); free(RB_ACC); RB_ACC=NULL; free(RB_CNT); RB_CNT=NULL;
    }
    free(H);free(hcfn);free(hcb);free(hcs);free(norm);free(hw);return logits;
}

/* 专家体积账(GiB): 每层 routed 6.442B 参数(256×3×2048×4096), 按档 bit/el 累计。
 * F=8.0(FP8 原精度, 仅验证对照, 非量化产物); n/1/z=1.0625(块scale); r=1.0052(行scale,
 * 纯1bit体积主线); 2=2.125; 3=3.1875。'z' 另加低秩因子 ZK×6144×2B×3/专家。骨干 Q8 另计 ~8.4。 */
static double lcfg_expert_gib(const char*lcfg){
    const double PL=256.0*3*2048*4096, GI=1073741824.0;
    double tot=0;
    for(int L=0;L<NLAYERS;L++){
        double bits;
        switch(lcfg[L]){
            case 'n': case '1': case 'z': bits=1.0625; break;
            case 'r': case 'g': bits=1.0+16.0*(2048+2048+4096)/(3.0*2048*4096); break;  /* 1.00521 */
            case '2': bits=2.125; break;
            case '3': bits=3.1875; break;
            default:  bits=8.0; break;   /* F */
        }
        tot += PL*bits/8.0/GI;
        if(lcfg[L]=='z') tot += 256.0*(double)ZK*(2048+4096)*2.0*3.0/GI;
    }
    return tot;
}

/* ===================== GGUF 直写导出 (DS4_EXPORT_GGUF) ===================== *
 * 把调优配方(signref μ=DS4_SIGNREF_MU + anchor 校准)对全 43 层×256 专家量化,
 * 按 gguf_offsets.py 的偏移表直接 pwrite 进已生成的 go1b GGUF(零中间文件)。
 * 引擎 GO1B_BLK 块格式的 d 字段物理上就是行 scale 复制 → 行scale方案原生兼容。 */
typedef struct { int L; char kind[8]; long off; long nel; int type; } goff_t;
static goff_t *GOFF; static int NGOFF;
static goff_t *goff_find(int L,const char*kind){
    for(int i=0;i<NGOFF;i++) if(GOFF[i].L==L&&!strcmp(GOFF[i].kind,kind)) return &GOFF[i];
    return NULL;
}
typedef struct { int L,S,n_fit; int *e_next; uint8_t *bg,*bu,*bd; size_t szG,szD;
                 int fd; size_t off0;
                 int g2fd; size_t g2off; int g2k; double g2sum[5];   /* go2b 侧车 fd/载荷偏移/热数/门统计{n,Σrel1,Σcos1,Σrel2,Σcos2}(内嵌时 g2fd=fd) */
                 int vqfd; double vqsum[4];   /* VQ 侧车 fd + 门统计{n_hot,Σcos_hot,n_cold,Σcos_cold} */
               } expw_t;   /* bg==NULL → 流式: 按偏移 pwrite 进 fd(855MB buffer→3.4MB/worker) */
static long long EXP_US_READ=0,EXP_US_VQ13=0,EXP_US_HC=0,EXP_US_VQ2=0;
static long long exp_us_now(void){ struct timeval tv; gettimeofday(&tv,NULL); return (long long)tv.tv_sec*1000000+tv.tv_usec; }
static void *export_worker(void*a){
    expw_t*w=a;
    float *Xc=malloc((size_t)(w->n_fit>0?w->n_fit:1)*DIM*4);
    uint8_t *tG=NULL,*tU=NULL,*tD=NULL;
    if(!w->bg){ tG=malloc(w->szG); tU=malloc(w->szG); tD=malloc(w->szD); }
    for(;;){ int e=__sync_fetch_and_add(w->e_next,1); if(e>=NEXP)break;
        if((e&31)==0){ char pg[24]; snprintf(pg,sizeof(pg),"%d/%d",e,NEXP);
            mlog(w->L,"量化文件","DQL2·signref字节",pg,
                 (uint64_t)NEXP*(2*w->szG+w->szD),"—","导出中"); }
        if(bf_from_bytes()&&BFB_W2COPY){   /* ★反修导出复用(2026-08-03 用户令"反修速度"): 冷 w2=量化段字节
                                             * 直拷(确定性同源: 同FP+同Xc+同μ10 ⇒ 同解, memcpy 使其构造性成立);
                                             * 热=洞(VQ 载荷 vq_keep 已保), G/U=VQ 洞 ⇒ 导出塌缩为纯 IO。 */
            int slr=(w->g2fd>0||dq_vq_on())?g2_hot_slot(w->L,e):-1;
            if(slr<0){
                uint8_t *dDr=w->bd?w->bd+(size_t)e*w->szD:tD;
                memcpy(dDr,BFB_W2COPY+(size_t)e*w->szD,w->szD);
                if(!w->bg){ if(pwrite(w->fd,dDr,w->szD,(off_t)(w->off0+2*(size_t)NEXP*w->szG+(size_t)e*w->szD))!=(ssize_t)w->szD) perror("exp-pw-d"); }
            }
            continue;
        }
        char n1[160],n3[160],n2[160];
        snprintf(n1,sizeof(n1),"layers.%d.ffn.experts.%d.w1.weight",w->L,e);
        snprintf(n3,sizeof(n3),"layers.%d.ffn.experts.%d.w3.weight",w->L,e);
        snprintf(n2,sizeof(n2),"layers.%d.ffn.experts.%d.w2.weight",w->L,e);
        long rr,cc; long long _t0=exp_us_now();
        float *e1=st_read_weight(&C,n1,&rr,&cc),*e3=st_read_weight(&C,n3,&rr,&cc),*e2=st_read_weight(&C,n2,&rr,&cc);
        __sync_fetch_and_add(&EXP_US_READ,exp_us_now()-_t0);
        if(!e1||!e3||!e2){ fprintf(stderr,"\n[!] L%d e%d 读失败\n",w->L,e); if(e1)free(e1); if(e3)free(e3); if(e2)free(e2); continue; }
        int ncal=0;
        const float *afin=NULL; const int *aidx_i=NULL; const int32_t *aidx_a=NULL;
        int csel[4096];
        if(EXP_FIN&&EXP_IDX&&EXP_L==w->L){ afin=EXP_FIN; aidx_i=EXP_IDX; }   /* ★调优同口径 */
        else if(ANC_OK){ afin=ANC.fin+(size_t)w->L*w->S*DIM; aidx_a=ANC.ridx+(size_t)w->L*w->S*NACT; }
        if(afin){
            /* DS4_CALIB_FULLSET: 导出段(blob g_r/GPTQ-H 的真正产地)同款全集喂入 —
             * hit 过滤在 n_fit~400 时每专家仅~9行, g_r 被钉死 1.000(全模型实测)。 */
            static int fullset2=-1; if(fullset2<0) fullset2=getenv("DS4_CALIB_FULLSET")?1:0;
            /* ★导出侧行帽(2026-08-09 提速定罪: 导出相位 8:48/层, 全在 per-expert 校准行
             * 矩阵乘/kmeans/GPTQ-H, 与行数线性)★: FULLSET 语义保留(全集均匀 stride 采样,
             * 无 routed-hit 饿死), 默认帽 512; DS4_CALIB_EXPORT_CAP=0 回全集。 */
            static int expcap=-2; if(expcap==-2){ const char*ecv=getenv("DS4_CALIB_EXPORT_CAP");
                expcap=ecv?atoi(ecv):0; }   /* 默认0=全集(行帽实测: 速度无效+质量-0.2pt, 2026-08-09) */
            int estride=1;
            if(fullset2&&expcap>0&&w->n_fit>expcap) estride=(w->n_fit+expcap-1)/expcap;   /* ceil: 933/512→2 */
            for(int s=0;s<w->n_fit;s++){ int hit=fullset2&&(s%estride==0);
                if(!fullset2) for(int a2=0;a2<NACT;a2++){ int ee=aidx_i?aidx_i[(size_t)s*NACT+a2]:(int)aidx_a[(size_t)s*NACT+a2];
                    if(ee==e){hit=1;break;} }
                if(hit){ memcpy(Xc+(size_t)ncal*DIM,afin+(size_t)s*DIM,(size_t)DIM*4);
                         if(ncal<4096)csel[ncal]=s; ncal++; } }
        }
        /* GPTAQ 非对称目标(与 coadapt 同解, 导出=拟合一致性): dx=x̃−x̂ 仅当 afin=漂移流时非零 */
        float *yadjE1=NULL,*yadjE3=NULL; { float tgaE=dq_tgt_alpha();
        if(tgaE>0.0f&&ncal&&ncal<=4096&&ANC_OK&&aidx_i){
            const float *afp2=ANC.fin+(size_t)w->L*w->S*DIM;
            float *dx=malloc((size_t)ncal*DIM*4);
            for(int t=0;t<ncal;t++){ const float *xf=afp2+(size_t)csel[t]*DIM,*xh=Xc+(size_t)t*DIM;
                for(int d=0;d<DIM;d++) dx[(size_t)t*DIM+d]=xf[d]-xh[d]; }
            yadjE1=malloc((size_t)ncal*MOEI*4); yadjE3=malloc((size_t)ncal*MOEI*4);
            dq_matmul(dx,e1,yadjE1,ncal,DIM,MOEI); dq_matmul(dx,e3,yadjE3,ncal,DIM,MOEI);
            for(size_t i2=0;i2<(size_t)ncal*MOEI;i2++){ yadjE1[i2]*=tgaE; yadjE3[i2]*=tgaE; }
            free(dx);
        } }
        int sl2=(w->g2fd>0||dq_vq_on())?g2_hot_slot(w->L,e):-1;   /* VQ 模式热判定不依赖 g2fd(冒烟bug修) */
        float *q1=NULL,*q3=NULL,*hc=NULL;
        if(sl2<0){   /* 冷专家(或 go2b 关): signref go1b。热专家跳过 → dql 热槽位=稀疏洞(省盘+省算) */
        q1=malloc((size_t)MOEI*DIM*4); q3=malloc((size_t)MOEI*DIM*4);
        uint8_t *dG=w->bg?w->bg+(size_t)e*w->szG:tG, *dU=w->bu?w->bu+(size_t)e*w->szG:tU, *dDn=w->bd?w->bd+(size_t)e*w->szD:tD;
        int vqcold=dq_vq_on();
        if(vqcold){   /* v2.2: 冷 w1/w3 → vq8x256 侧车; dql G/U 段留稀疏洞 */
            double c1v=0,c3v=0;
            long long _tv=exp_us_now();
            float *t1=vq_export_matrix(e1,MOEI,DIM,ncal?Xc:NULL,ncal,vq_cold_dim(),vq_cold_nc(),w->vqfd,vq_slot_off(w->L,e,0,MOEI,DIM),&c1v);
            float *t3=vq_export_matrix(e3,MOEI,DIM,ncal?Xc:NULL,ncal,vq_cold_dim(),vq_cold_nc(),w->vqfd,vq_slot_off(w->L,e,1,MOEI,DIM),&c3v);
            __sync_fetch_and_add(&EXP_US_VQ13,exp_us_now()-_tv);
            memcpy(q1,t1,(size_t)MOEI*DIM*4); memcpy(q3,t3,(size_t)MOEI*DIM*4); free(t1); free(t3);
            w->vqsum[2]+=2; w->vqsum[3]+=c1v+c3v;
        } else {
        dq_signref_export_adj(e1,MOEI,DIM,ncal?Xc:NULL,ncal,dG,q1,yadjE1);
        dq_signref_export_adj(e3,MOEI,DIM,ncal?Xc:NULL,ncal,dU,q3,yadjE3);
        }
        int nhc=0;
        if(ncal){   /* w2 校准 = 量化 q1/q3 过校准行(顺序补偿, 与 harness 同口径) */
            hc=malloc((size_t)ncal*MOEI*4);
            long long _th=exp_us_now();
            float *gg=malloc((size_t)ncal*MOEI*4),*uu=malloc((size_t)ncal*MOEI*4);
            dq_matmul(Xc,q1,gg,ncal,DIM,MOEI); dq_matmul(Xc,q3,uu,ncal,DIM,MOEI);
            for(size_t i=0;i<(size_t)ncal*MOEI;i++){ float g2=gg[i],u2=uu[i];
                if(SWLIM>0){ if(u2>SWLIM)u2=SWLIM; if(u2<-SWLIM)u2=-SWLIM; if(g2>SWLIM)g2=SWLIM; }
                hc[i]=dq_silu(g2)*u2; }
            free(gg);free(uu); nhc=ncal;
            __sync_fetch_and_add(&EXP_US_HC,exp_us_now()-_th);
        }
        if(vqcold&&vq_w2_dim()>0){   /* R28: 冷 w2 码本导出(dql D 段留洞, 载荷进 VQ 侧车槽) */
            double cvw=0;
            /* ★w2 顺序补偿(2026-08-03 用户令)★: y2ref = FP w2 过【量化态】hidden —
             * 与 signref μ 精修/coadapt y2ref 完全同口径。码本重构后每行输出匹配闭式缩放
             * 烘进行 scale(vq_qc.h vq_export_matrix_seq), 载荷格式/引擎逐字节不变。 */
            long long _t2=exp_us_now();
            float *t2=vq_export_matrix_seq(e2,DIM,MOEI,nhc?hc:NULL,nhc,vq_w2_dim(),vq_w2_nc(),
                                       w->vqfd,vq_slot_off(w->L,e,2,MOEI,DIM),&cvw,1);
            __sync_fetch_and_add(&EXP_US_VQ2,exp_us_now()-_t2);
            free(t2); w->vqsum[2]+=1; w->vqsum[3]+=cvw;
        } else
        dq_signref_export(e2,DIM,MOEI,hc,nhc,dDn,NULL);
        if(!w->bg){   /* 流式: 三段各按偏移落盘(布局与整块拼接逐字节一致); VQ 模式 G/U=稀疏洞 */
            if(!vqcold){
            if(pwrite(w->fd,dG,w->szG,(off_t)(w->off0+(size_t)e*w->szG))!=(ssize_t)w->szG) perror("exp-pw-g");
            if(pwrite(w->fd,dU,w->szG,(off_t)(w->off0+(size_t)NEXP*w->szG+(size_t)e*w->szG))!=(ssize_t)w->szG) perror("exp-pw-u");
            }
            if(!(vqcold&&vq_w2_dim()>0))
            if(pwrite(w->fd,dDn,w->szD,(off_t)(w->off0+2*(size_t)NEXP*w->szG+(size_t)e*w->szD))!=(ssize_t)w->szD) perror("exp-pw-d");
        }
        } else {
            /* 热专家 go2b(合并2bit, 激活最优) → 独立侧车文件; hc2 走 go2b q1/q3 顺序补偿(合并态口径) */
            size_t szG2=(size_t)MOEI*go2b_row_bytes(DIM), szD2=(size_t)DIM*go2b_row_bytes(MOEI);
            uint8_t *b1=malloc(szG2),*b3=malloc(szG2),*bD=malloc(szD2);
            float *g1=malloc((size_t)MOEI*DIM*4),*g3=malloc((size_t)MOEI*DIM*4);
            int vqhot=dq_vq_on();
            if(vqhot){   /* v2.2: 热全三矩阵 → vq4x512 侧车 */
                double cv1=0,cv3=0;
                float *t1=vq_export_matrix(e1,MOEI,DIM,ncal?Xc:NULL,ncal,4,512,w->vqfd,vq_slot_off(w->L,e,0,MOEI,DIM),&cv1);
                float *t3=vq_export_matrix(e3,MOEI,DIM,ncal?Xc:NULL,ncal,4,512,w->vqfd,vq_slot_off(w->L,e,1,MOEI,DIM),&cv3);
                memcpy(g1,t1,(size_t)MOEI*DIM*4); memcpy(g3,t3,(size_t)MOEI*DIM*4); free(t1); free(t3);
                w->vqsum[0]+=2; w->vqsum[1]+=cv1+cv3;
            } else {
            dq_go2b_encode_adj(e1,MOEI,DIM,ncal?Xc:NULL,ncal,b1,g1,yadjE1);
            dq_go2b_encode_adj(e3,MOEI,DIM,ncal?Xc:NULL,ncal,b3,g3,yadjE3);
            }
            float *hc2=NULL; int nh2=0;
            if(ncal){
                hc2=malloc((size_t)ncal*MOEI*4);
                float *gg=malloc((size_t)ncal*MOEI*4),*uu=malloc((size_t)ncal*MOEI*4);
                dq_matmul(Xc,g1,gg,ncal,DIM,MOEI); dq_matmul(Xc,g3,uu,ncal,DIM,MOEI);
                for(size_t i=0;i<(size_t)ncal*MOEI;i++){ float g2v=gg[i],u2=uu[i];
                    if(SWLIM>0){ if(u2>SWLIM)u2=SWLIM; if(u2<-SWLIM)u2=-SWLIM; if(g2v>SWLIM)g2v=SWLIM; }
                    hc2[i]=dq_silu(g2v)*u2; }
                free(gg);free(uu); nh2=ncal;
            }
            float *gD=malloc((size_t)DIM*MOEI*4);
            if(vqhot){   /* 热 w2=vq4x512(实测序: go2b>vq4x512>signref; signref 版 2026-08-03 已否决) */
                double cv2=0;
                float *t2=vq_export_matrix(e2,DIM,MOEI,hc2,nh2,4,512,w->vqfd,vq_slot_off(w->L,e,2,MOEI,DIM),&cv2);
                memcpy(gD,t2,(size_t)DIM*MOEI*4); free(t2);
                w->vqsum[0]+=1; w->vqsum[1]+=cv2;
            } else {
            dq_go2b_encode(e2,DIM,MOEI,hc2,nh2,bD,gD);
            if(pwrite(w->g2fd,b1,szG2,(off_t)(w->g2off+(size_t)sl2*szG2))!=(ssize_t)szG2) perror("g2-pw-g");
            if(pwrite(w->g2fd,b3,szG2,(off_t)(w->g2off+(size_t)w->g2k*szG2+(size_t)sl2*szG2))!=(ssize_t)szG2) perror("g2-pw-u");
            if(pwrite(w->g2fd,bD,szD2,(off_t)(w->g2off+2*(size_t)w->g2k*szG2+(size_t)sl2*szD2))!=(ssize_t)szD2) perror("g2-pw-d");
            }
            if(ncal){   /* 单层门统计: w1/w2 输出重建 vs FP(校准行) */
                float *Ot=malloc((size_t)ncal*MOEI*4),*Oq=malloc((size_t)ncal*MOEI*4);
                dq_matmul(Xc,e1,Ot,ncal,DIM,MOEI); dq_matmul(Xc,g1,Oq,ncal,DIM,MOEI);
                double e2s=0,a2s=0,nu=0,d1n=0,d2n=0;
                for(size_t i=0;i<(size_t)ncal*MOEI;i++){ double d=(double)Ot[i]-Oq[i]; e2s+=d*d; a2s+=(double)Ot[i]*Ot[i];
                    nu+=(double)Ot[i]*Oq[i]; d1n+=(double)Ot[i]*Ot[i]; d2n+=(double)Oq[i]*Oq[i]; }
                w->g2sum[1]+=sqrt(e2s/(a2s+1e-30)); w->g2sum[2]+=nu/(sqrt(d1n)*sqrt(d2n)+1e-30);
                free(Ot);free(Oq);
                float *Ot2=malloc((size_t)ncal*DIM*4),*Oq2=malloc((size_t)ncal*DIM*4);
                dq_matmul(hc2,e2,Ot2,ncal,MOEI,DIM); dq_matmul(hc2,gD,Oq2,ncal,MOEI,DIM);
                e2s=0;a2s=0;nu=0;d1n=0;d2n=0;
                for(size_t i=0;i<(size_t)ncal*DIM;i++){ double d=(double)Ot2[i]-Oq2[i]; e2s+=d*d; a2s+=(double)Ot2[i]*Ot2[i];
                    nu+=(double)Ot2[i]*Oq2[i]; d1n+=(double)Ot2[i]*Ot2[i]; d2n+=(double)Oq2[i]*Oq2[i]; }
                w->g2sum[3]+=sqrt(e2s/(a2s+1e-30)); w->g2sum[4]+=nu/(sqrt(d1n)*sqrt(d2n)+1e-30);
                free(Ot2);free(Oq2);
                w->g2sum[0]+=1.0;
            }
            free(b1);free(b3);free(bD);free(g1);free(g3);free(gD); if(hc2)free(hc2);
        }
        if(hc)free(hc);
        if(q1)free(q1); if(q3)free(q3); if(yadjE1)free(yadjE1); if(yadjE3)free(yadjE3); free(e1);free(e3);free(e2);
    }
    if(tG)free(tG); if(tU)free(tU); if(tD)free(tD);
    free(Xc); return NULL;
}
/* 记录 paysz(预扫与写循环同源, 防偏移漂移) */
static uint64_t rec_paysz(const elrec_t*r,size_t szG,size_t szD){
    if(!strcmp(r->name,"1bit")) return (uint64_t)NEXP*(2*szG+szD);
    /* ★dyn8 全家族带 V8(2026-08-03 实锤: pc./lf./ls.GLdyn8 只匹配 "z.GLdyn8" 前缀 ⇒ V8 从不落盘
     * ⇒ 回放/zchain/引擎全丢主力动态)。V8 为该层共享 PCA 方向, 逐记录冗余落盘换四方同源。 */
    if(strstr(r->name,"GLdyn8")&&r->paylen==36&&STK8_V8) return 36+(uint64_t)8*DIM*2;
    if(!strcmp(r->name,"loss.cls")&&STK_CLSV) return (uint64_t)HCM*DIM*2;
    if(strstr(r->name,"zl.RRR")&&r->paylen==0&&ZLP_K>0)   /* z^L 正位直写(平行架构): 表位即注入位 */
        return 16+2*((uint64_t)ZLP_K+2*(uint64_t)ZLP_K*DIM);
    return r->paylen;
}
/* 单层量化文件导出(DQL2, 与表格一一对应): 权重字节(signref 同解)+ 全部元素记录与载荷 */
static void export_layer_file(int L,int S,int n_fit,const char*lf){

        /* ===== 单层量化文件 DQL2 —— 与表格一一对应(第一原则) =====
         * 'DQL2' u32 | L u32 | nrec u32
         * 每记录(=表格一行): name[16] algo[64] vol u64(表格体积列) paysz u64(实际载荷)
         *                    m1..m4 f32(实测数值) verdict i32(1正向落地 2正向未落地 3探索中 4生效)
         *                    payload[paysz](落地元素才有: 1bit=全部权重字节, z.*=参数, bwd.*=t) */
        size_t rbG=go1b_blk_row_bytes(DIM), szG=(size_t)MOEI*rbG;
        size_t rbD=go1b_blk_row_bytes(MOEI), szD=(size_t)DIM*rbD;
        /* ★流式导出★: 旧版三块整拼 buffer=855MB 瞬时尖峰(实测在 L20 把 footprint 顶过 12G 看门狗);
         * 改为先写记录头到 1bit 载荷处 → 扩文件 → worker 按偏移并行 pwrite(3.4MB/worker), 字节布局不变。 */
        /* ★DS4_EXPORT_BYTES=0(反修段)★: dql 不可变 — 跳过全部字节写(1bit/g2hot/dql 本体),
         * 只重建 op 侧车(probe1 实锤: 反修段曾整写 dql 918M→816M, 平行架构破洞)。 */
        int wb=!(getenv("DS4_EXPORT_BYTES")&&atoi(getenv("DS4_EXPORT_BYTES"))==0);
        FILE*f=wb?fopen(lf,"wb"):NULL;
        if(wb&&!f){ perror("layerfile"); return; }
        int g2k=(GO2B_HOT&&L<64)?G2_K[L]:0;   /* 本层热专家数 */
        /* ★热载荷内嵌(DS4_MINVOL): 追加 "g2hot"(热 go2b, DQG2 侧车布局原样内嵌)记录 →
         * dql 单文件自持, 不再产独立 g2 侧车。偏移预算好, worker 与 1bit 同批填。 */
        int embK=0, nextra=0;
        size_t off_scan=12, off_g2_rec=0,off_g2=0;
        uint64_t g2_psz=0;
        size_t szG2=(size_t)MOEI*go2b_row_bytes(DIM), szD2=(size_t)DIM*go2b_row_bytes(MOEI);
        /* ★平行架构分流(2026-08-04 用户令)★: dql=纯权重字节(1bit+g2hot 内嵌), 量化后只读;
         * 其余全部记录(z 家族/zl.RRR/loss/bwd/探索账)→ op 侧车 dql_ops_LXX.bin(应用序=侧车序)。 */
        uint32_t nbyte=0;
        for(int i=0;i<NELE;i++) if(!strcmp(ELE[i].name,"1bit")){ nbyte++;
            off_scan+=116+rec_paysz(&ELE[i],szG,szD); }
        if(getenv("DS4_MINVOL")&&g2k>0){
            embK=g2k;
            g2_psz=(uint64_t)g2_sidecar_hdr(embK)+2*(uint64_t)embK*szG2+(uint64_t)embK*szD2;
            off_g2_rec=off_scan; off_g2=off_g2_rec+116; off_scan=off_g2+(size_t)g2_psz; nextra++;
        }
        double g2_mc=0;   /* 热门统计 mean_cos(内嵌 g2hot 记录头用) */
        uint32_t mg=0x324C5144,Lu=(uint32_t)L,nr=(uint32_t)(nbyte+nextra);
        if(wb){ fwrite(&mg,4,1,f);fwrite(&Lu,4,1,f);fwrite(&nr,4,1,f); }
        char opspath[512]; ops_sidecar_path(lf,L,opspath,sizeof(opspath));
        FILE*fo=fopen(opspath,"wb");
        if(!fo){ perror("ops-sidecar"); fclose(f); return; }
        { uint32_t mgo=OPSC_MAGIC,nro=(uint32_t)(NELE-nbyte);
          fwrite(&mgo,4,1,fo); fwrite(&Lu,4,1,fo); fwrite(&nro,4,1,fo); }
        for(int i=0;i<NELE;i++){ elrec_t*r=&ELE[i];
            uint64_t paysz=rec_paysz(r,szG,szD);
            int isbyte=!strcmp(r->name,"1bit");
            if(isbyte&&!wb) continue;              /* 反修段: dql 字节记录整条跳过(不可变) */
            FILE*W=isbyte?f:fo;                    /* 字节→dql; 其余→op 侧车 */
            fwrite(r->name,1,16,W); fwrite(r->algo,1,64,W);
            fwrite(&r->vol,8,1,W); fwrite(&paysz,8,1,W);
            fwrite(&r->m1,4,1,W);fwrite(&r->m2,4,1,W);fwrite(&r->m3,4,1,W);fwrite(&r->m4,4,1,W);
            fwrite(&r->verdict,4,1,W);
            if(paysz){
                if(!strcmp(r->name,"1bit")){
                    fflush(f); size_t off0=(size_t)ftello(f);
                    if(fseeko(f,(off_t)(off0+paysz-1),SEEK_SET)!=0||fputc(0,f)==EOF){ perror("exp-extend"); fclose(f); return; }
                    fflush(f);
                    /* go2b 热载荷: minvol=内嵌 dql g2hot 记录区(g2fd=dql fd); 否则独立侧车文件 */
                    int g2fd=-1; size_t g2off=0;
                    char g2path[512]={0};
                    int vqfd=-1; char vqpath[512]={0};
                    /* ★反修不重写 VQ 载荷(2026-07-31 用户令"多大就是多大, 重新设计量化脚本")★
                     * 反修只调 z/四损失/感知这些【侧车参数】, 路由专家的 VQ 权重一个 bit 都不改。
                     * 旧行为每次导出都 O_TRUNC 重编码 256 专家×3 矩阵, 两个恶果:
                     *   ①纯浪费: 反修每层白烧一遍全层 VQ 编码(反修慢一倍的主因)
                     *   ②★体积漂移★: 重编码时热数取自全局 G2_K(反修的 DS4_GO2B_HOT_TABLE=top64),
                     *     覆盖掉量化时按计划表定的 hot ⇒ 每层 728.7→820 MiB, 43 层超预算 11.7%,
                     *     而且直到合并才发现。热档 4×512 比冷档密, 热数一涨体积就涨。
                     * 现在: 侧车已在且头合法(magic/L/nexp 对) ⇒ 保留其字节, 只更新 dql 记录区。
                     * 量化首次导出(侧车不存在)仍走完整编码路径, 语义不变。 */
                    int vq_keep=0;
                    if(dq_vq_on()){
                        vq_sidecar_path(lf,L,vqpath,sizeof(vqpath));
                        if(getenv("DS4_BWD")){
                            int pf=open(vqpath,O_RDONLY);
                            if(pf>=0){ uint32_t h4[4]={0};
                                if(read(pf,h4,16)==16&&h4[0]==VQSC_MAGIC&&h4[2]==(uint32_t)L&&h4[3]==256){
                                    struct stat vs;
                                    if(fstat(pf,&vs)==0&&(size_t)vs.st_size>=vq_hdr_bytes()){
                                        vq_keep=1;
                                        fprintf(stderr,"[层文件] L%02d 反修保留既有 VQ 载荷 %.1fMiB(权重未变, 不重编码)\n",
                                                L,vs.st_size/1048576.0);
                                    } }
                                close(pf); }
                        }
                    }
                    if(dq_vq_on()&&!vq_keep){
                        vqfd=open(vqpath,O_RDWR|O_CREAT|O_TRUNC,0644);
                        if(vqfd<0){ perror("vq-sidecar-open"); fclose(f); return; }
                        size_t hb2=vq_hdr_bytes(); uint8_t *hb=calloc(1,hb2);
                        uint32_t mgv=VQSC_MAGIC,verv=1,Lv=(uint32_t)L,nev=256;
                        memcpy(hb,&mgv,4); memcpy(hb+4,&verv,4); memcpy(hb+8,&Lv,4); memcpy(hb+12,&nev,4);
                        for(int e2=0;e2<256;e2++) for(int wh=0;wh<3;wh++){
                            uint64_t o64=(uint64_t)vq_slot_off(L,e2,wh,MOEI,DIM);   /* w2 槽随 vq_slot_off 的 w2 总闸(w2dim=0 ⇒ 0) */
                            memcpy(hb+16+((size_t)e2*3+wh)*8,&o64,8); }
                        if(pwrite(vqfd,hb,hb2,0)!=(ssize_t)hb2) perror("vq-hdr");
                        free(hb);
                        size_t vend=vq_total_bytes(L,MOEI,DIM); uint8_t z2=0;
                        if(pwrite(vqfd,&z2,1,(off_t)(vend-1))!=1){ perror("vq-extend"); close(vqfd); fclose(f); return; }
                        fprintf(stderr,"[层文件] L%02d VQ 侧车预留 %.1fMiB → %s\n",L,vend/1048576.0,vqpath);
                    }
                    if(g2k>0&&!dq_vq_on()){
                        if(embK>0){   /* 内嵌: 热载荷直写 dql 的 g2hot 记录区(DQG2 头由循环后主线程写) */
                            g2fd=fileno(f); g2off=off_g2+g2_sidecar_hdr(g2k);
                        } else {
                        g2_sidecar_path(lf,L,g2path,sizeof(g2path));
                        g2fd=open(g2path,O_RDWR|O_CREAT|O_TRUNC,0644);
                        if(g2fd<0){ perror("g2-sidecar-open"); fclose(f); return; }
                        size_t hdr2=g2_sidecar_hdr(g2k);
                        uint8_t*hb=calloc(1,hdr2);
                        uint32_t mg2=G2SC_MAGIC,ver=1,Lu2=(uint32_t)L,khu=(uint32_t)g2k;
                        memcpy(hb,&mg2,4); memcpy(hb+4,&ver,4); memcpy(hb+8,&Lu2,4); memcpy(hb+12,&khu,4);
                        for(int i2=0;i2<g2k;i2++){ uint16_t idv=G2_IDS[L][i2]; memcpy(hb+24+2*(size_t)i2,&idv,2); }
                        if(pwrite(g2fd,hb,hdr2,0)!=(ssize_t)hdr2) perror("g2-hdr");
                        free(hb);
                        g2off=hdr2;
                        size_t g2end=hdr2+2*(size_t)g2k*szG2+(size_t)g2k*szD2;
                        uint8_t z=0;
                        if(pwrite(g2fd,&z,1,(off_t)(g2end-1))!=1){ perror("g2-extend"); close(g2fd); fclose(f); return; }
                        }
                    }
                    fprintf(stderr,"[层文件] L%02d signref 权重字节生成(μ=%.0f, 流式%s) ",L,dq_signref_mu,
                            g2k>0?"+go2b热侧车":"");
                    int e_next=0,nth=NTHREADS>NEXP?NEXP:NTHREADS,fd=fileno(f);
                    expw_t *ws=calloc((size_t)nth,sizeof(expw_t)); pthread_t *th=malloc((size_t)nth*sizeof(pthread_t));
                    for(int t=0;t<nth;t++){ ws[t]=(expw_t){L,S,n_fit,&e_next,NULL,NULL,NULL,szG,szD,fd,off0,g2fd,g2off,g2k,{0},vqfd,{0}};
                        pthread_create(&th[t],NULL,export_worker,&ws[t]); }
                    for(int t=0;t<nth;t++) pthread_join(th[t],NULL);
                    fprintf(stderr,"[EXPORT_PROF] read=%.0fs vq13=%.0fs hc=%.0fs vq2=%.0fs (核秒累计)\n",
                            (double)EXP_US_READ/1e6,(double)EXP_US_VQ13/1e6,(double)EXP_US_HC/1e6,(double)EXP_US_VQ2/1e6);
                    EXP_US_READ=EXP_US_VQ13=EXP_US_HC=EXP_US_VQ2=0;
                    if(dq_vq_on()&&!vq_keep){   /* ★VQ_GATE 每层日志(用户令: 每层量化必须有日志)★ */
                        double nh=0,ch=0,ncd=0,cc=0;
                        for(int t=0;t<nth;t++){ nh+=ws[t].vqsum[0]; ch+=ws[t].vqsum[1];
                                                ncd+=ws[t].vqsum[2]; cc+=ws[t].vqsum[3]; }
                        double mh=nh>0?ch/nh:0, mc2=ncd>0?cc/ncd:0;
                        fprintf(stderr,"\nVQ_GATE L=%02d 热矩阵=%d 冷w1w3=%d 均值cos: hot=%.4f cold=%.4f %s → %s",
                                L,(int)nh,(int)ncd,mh,mc2,
                                (mh>=0.90&&mc2>=0.75)?"PASS":"★WARN(阈 hot0.90/cold0.75)★",vqpath);
                        close(vqfd); vqfd=-1;
                    }
                    if(g2k>0&&!dq_vq_on()){   /* 单层重建门(runbook §3.1): 均值 cos≥0.85; 内嵌时进 g2hot 记录头, 独立侧车回填 */
                        double n2=0,r1=0,c1=0,r2d=0,c2d=0;
                        for(int t=0;t<nth;t++){ n2+=ws[t].g2sum[0]; r1+=ws[t].g2sum[1]; c1+=ws[t].g2sum[2];
                            r2d+=ws[t].g2sum[3]; c2d+=ws[t].g2sum[4]; }
                        float mc=(n2>0)?(float)((c1+c2d)/(2.0*n2)):0.0f;
                        if(embK>0) g2_mc=mc;   /* 内嵌: fd 是 dql 本体, 不回填不 close */
                        else { if(pwrite(g2fd,&mc,4,16)!=4) perror("g2-m1"); close(g2fd); }
                        fprintf(stderr,"\nGO2B_GATE L=%02d 热=%d/%d w1[rel=%.4f cos=%.4f] w2[rel=%.4f cos=%.4f] 均值cos=%.4f %s → %s",
                                L,(int)n2,g2k,n2>0?r1/n2:0,n2>0?c1/n2:0,n2>0?r2d/n2:0,n2>0?c2d/n2:0,mc,
                                mc>=0.85f?"PASS":"★FAIL(<0.85)★",embK>0?"(内嵌dql)":g2path);
                    }
                    free(ws);free(th); fputc('\n',stderr);
                    if(fseeko(f,(off_t)(off0+paysz),SEEK_SET)!=0){ perror("exp-seek"); fclose(f); return; }
                } else if(strstr(r->name,"GLdyn8")&&paysz>36){
                    fwrite(r->pay,1,36,W);
                    uint16_t *h=malloc((size_t)8*DIM*2);
                    for(size_t j=0;j<(size_t)8*DIM;j++) h[j]=go1b_fp32_to_fp16(STK8_V8[j]);
                    fwrite(h,2,(size_t)8*DIM,W); free(h);
                } else if(strstr(r->name,"zl.RRR")&&r->paylen==0&&ZLP_K>0){
                    /* ★z^L 正位直写(取代旧 export后append)★: 表位=评估注入位, 错序温床根除 */
                    uint32_t zk=(uint32_t)ZLP_K,din=(uint32_t)DIM,dout=(uint32_t)DIM;
                    fwrite(&zk,4,1,W); fwrite(&ZLP_TR,4,1,W); fwrite(&din,4,1,W); fwrite(&dout,4,1,W);
                    size_t nh=(size_t)zk+(size_t)zk*din+(size_t)zk*dout;
                    uint16_t*h=malloc(nh*2); size_t o2=0;
                    for(uint32_t i2=0;i2<zk;i2++) h[o2++]=go1b_fp32_to_fp16(ZLP_Z[i2]);
                    for(size_t i2=0;i2<(size_t)dout*zk;i2++) h[o2++]=go1b_fp32_to_fp16(ZLP_U[i2]);
                    for(size_t i2=0;i2<(size_t)din*zk;i2++)  h[o2++]=go1b_fp32_to_fp16(ZLP_V[i2]);
                    fwrite(h,2,nh,W); free(h);
                } else if(!strcmp(r->name,"loss.cls")){
                    uint16_t *h=malloc((size_t)HCM*DIM*2);
                    for(size_t j=0;j<(size_t)HCM*DIM;j++) h[j]=go1b_fp32_to_fp16(STK_CLSV[j]);
                    fwrite(h,2,(size_t)HCM*DIM,W); free(h);
                } else fwrite(r->pay,1,(size_t)r->paylen,W);
            }
            printf("DQL2REC L=%d %-12s vol=%llu paysz=%llu m=[%.4f %.4f %.1f %.2f] 判定=%d | %s\n",
                   L,r->name,(unsigned long long)r->vol,(unsigned long long)paysz,
                   r->m1,r->m2,r->m3,r->m4,r->verdict,r->algo);
        }
        char nm16[16],al64[64]; uint64_t v8; float m1,m2,m3,m4=0; int vd=4;
        if(embK>0&&wb){   /* 追加内嵌记录 "g2hot"(DQG2 布局); 反修段 dql 不可变=跳过 */
            if(fseeko(f,(off_t)off_g2_rec,SEEK_SET)!=0) perror("g2-rec-seek");
            memset(nm16,0,16); memset(al64,0,64);
            snprintf(nm16,16,"g2hot"); snprintf(al64,64,"热%d go2b 内嵌(DQG2布局) mean_cos=%.4f",embK,g2_mc);
            v8=g2_psz; m1=(float)embK; m2=(float)g2_mc; m3=0;
            fwrite(nm16,1,16,f); fwrite(al64,1,64,f); fwrite(&v8,8,1,f); fwrite(&v8,8,1,f);
            fwrite(&m1,4,1,f);fwrite(&m2,4,1,f);fwrite(&m3,4,1,f);fwrite(&m4,4,1,f); fwrite(&vd,4,1,f);
            { size_t hdr2=g2_sidecar_hdr(embK); uint8_t*hb=calloc(1,hdr2);
              uint32_t mg2=G2SC_MAGIC,ver=1,Lu2=(uint32_t)L,khu=(uint32_t)embK; float mcf=(float)g2_mc;
              memcpy(hb,&mg2,4); memcpy(hb+4,&ver,4); memcpy(hb+8,&Lu2,4); memcpy(hb+12,&khu,4);
              memcpy(hb+16,&mcf,4);
              for(int i2=0;i2<embK;i2++){ uint16_t idv=G2_IDS[L][i2]; memcpy(hb+24+2*(size_t)i2,&idv,2); }
              fwrite(hb,1,hdr2,f); free(hb); }
            printf("DQL2REC L=%d %-12s vol=%llu paysz=%llu m=[%.4f %.4f %.1f %.2f] 判定=%d | %s\n",
                   L,"g2hot",(unsigned long long)v8,(unsigned long long)v8,m1,m2,m3,m4,vd,al64);
        }
        if(embK>0&&wb){
            fflush(f);
            if(ftruncate(fileno(f),(off_t)off_scan)!=0) perror("dql-trunc");   /* 尾洞(热槽/截断)补齐文件长 */
        }
        long fsz=0;
        if(wb){ fsz=(embK>0)?(long)off_scan:ftell(f); fclose(f); }
        else { struct stat dst2; fsz=stat(lf,&dst2)==0?(long)dst2.st_size:0; }   /* 反修段: dql 未动, 报现值 */
        long osz=ftell(fo); fclose(fo);
        if(ZLP_K>0&&ZLP_L==L){   /* z^L 已正位直写侧车 → 清暂存(接管旧 append 块职责) */
            mlog(L,"z^L","zl.RRR 正位直写(op 侧车)","已落地",
                 16+2*((uint64_t)ZLP_K+2*(uint64_t)ZLP_K*DIM),"平行架构: 表位=注入位","✓正向落地");
            free(ZLP_U); free(ZLP_V); free(ZLP_Z); ZLP_U=ZLP_V=ZLP_Z=NULL; ZLP_K=0; ZLP_L=-1;
        }
        printf("LAYERFILE %s L=%d 字节记录=%u bytes=%ld (%.2f MiB) | ops侧车 %s 记录=%u bytes=%ld\n",
               lf,L,nbyte+(uint32_t)nextra,fsz,(double)fsz/1048576.0,
               opspath,(uint32_t)(NELE-nbyte),osz);
        fflush(stdout);
}
static void export_gguf(int S,int n_fit){
    const char*gp=getenv("DS4_EXPORT_GGUF"), *op=getenv("DS4_EXPORT_OFF");
    if(!op){ fprintf(stderr,"需 DS4_EXPORT_OFF=偏移表(gguf_offsets.py 输出)\n"); exit(1); }
    GOFF=calloc((size_t)NL*3,sizeof(goff_t)); NGOFF=0;
    FILE*f=fopen(op,"r"); if(!f){perror("off");exit(1);}
    { char nm[256]; int ty; long off,nel;
      while(fscanf(f,"%255s %d %ld %ld",nm,&ty,&off,&nel)==4){
        int L; char kind[16]; int rest=0;
        if(sscanf(nm,"blk.%d.ffn_%15[^_]_exps.weight%n",&L,kind,&rest)==2 && nm[rest]==0 && L<NL && NGOFF<NL*3){
            GOFF[NGOFF].L=L; snprintf(GOFF[NGOFF].kind,8,"%s",kind);
            GOFF[NGOFF].off=off; GOFF[NGOFF].nel=nel; GOFF[NGOFF].type=ty; NGOFF++; } } }
    fclose(f);
    fprintf(stderr,"[export] 偏移表专家 tensor %d 个 (期望 %d)\n",NGOFF,NLAYERS*3);
    FILE*gf=fopen(gp,"r+b"); if(!gf){perror("gguf");exit(1);}
    size_t rbG=go1b_blk_row_bytes(DIM), szG=(size_t)MOEI*rbG;   /* w1/w3 每专家 bytes */
    size_t rbD=go1b_blk_row_bytes(MOEI), szD=(size_t)DIM*rbD;   /* w2 */
    uint8_t *bg=malloc((size_t)NEXP*szG),*bu=malloc((size_t)NEXP*szG),*bd=malloc((size_t)NEXP*szD);
    for(int L=0;L<NLAYERS;L++){
        goff_t *tg=goff_find(L,"gate"),*tu=goff_find(L,"up"),*td=goff_find(L,"down");
        if(!tg||!tu||!td){ fprintf(stderr,"L%02d tensor 缺失, 跳过\n",L); continue; }
        if(tg->type!=40||tu->type!=40||td->type!=40){ fprintf(stderr,"L%02d 类型非 go1b(40): %d/%d/%d — 拒写\n",L,tg->type,tu->type,td->type); exit(1); }
        long need=(long)NEXP*MOEI*DIM;
        if(tg->nel!=need||tu->nel!=need||td->nel!=need){ fprintf(stderr,"L%02d 元素数不符 — 拒写\n",L); exit(1); }
        fprintf(stderr,"L%02d 导出 ",L);
        int e_next=0; int nth=NTHREADS>NEXP?NEXP:NTHREADS;
        expw_t *ws=calloc((size_t)nth,sizeof(expw_t)); pthread_t *th=malloc((size_t)nth*sizeof(pthread_t));
        for(int t=0;t<nth;t++){ ws[t]=(expw_t){L,S,n_fit,&e_next,bg,bu,bd,szG,szD,0,0,-1,0,0,{0},-1,{0}};
            pthread_create(&th[t],NULL,export_worker,&ws[t]); }
        for(int t=0;t<nth;t++) pthread_join(th[t],NULL);
        free(ws);free(th);
        int ok=1;
        ok &= fseeko(gf,(off_t)tg->off,SEEK_SET)==0 && fwrite(bg,1,(size_t)NEXP*szG,gf)==(size_t)NEXP*szG;
        ok &= fseeko(gf,(off_t)tu->off,SEEK_SET)==0 && fwrite(bu,1,(size_t)NEXP*szG,gf)==(size_t)NEXP*szG;
        ok &= fseeko(gf,(off_t)td->off,SEEK_SET)==0 && fwrite(bd,1,(size_t)NEXP*szD,gf)==(size_t)NEXP*szD;
        if(!ok){ fprintf(stderr,"L%02d 写失败!\n",L); exit(1); }
        fprintf(stderr," 已写 (%.0f MB)\n",(2.0*NEXP*szG+NEXP*szD)/1048576.0);
    }
    fflush(gf); fclose(gf);
    free(bg);free(bu);free(bd);
    fprintf(stderr,"[export] 完成: %s 全部专家已替换为 signref(μ=%.0f)+anchor校准 配方\n",gp,dq_signref_mu);
}

/* ===================== 逐层渐进调优 (DS4_TUNE=1) ===================== *
 * 用户架构(2026-07-10): 每层动态、渐进、成果永久复用 — 不做"一把跑43层最后作废"的长任务。
 *   每层: 候选集现场对比(层内优化) → held 泛化+感知加权评分(向 FP 锚偏离=向后代理) →
 *   选定即固化(plan 文件 + 激活 checkpoint 落盘) → 下一层在已固化成果上继续。
 *   重启时已调层秒级跳过(复用); 加新候选只重跑该层及以后(删对应 plan 行/ckpt 即可)。 */
typedef struct { const char*name; char cfg; double mu; int rounds; int lz; float lztr,lzlam; int hotk; } cand_t;
/* 候选集 v2(目标导向, 2026-07-10 晚): 同族微调(g3/g30/rz)每层只挽回 0.2-7% = 瞎跑;
 * 换入真杠杆 q2(+0.84GiB/层), 预算规则: 免费候选先选最优, q2 挽回≥DS4_TUNE_GAIN(默认15%)才升位。 */
static const cand_t CANDS[]={
    {"r",   'r',  0.0,0, 0,0,0, 0},       /* 行scale 保守锚(零体积差) */
    {"g10", 'g', 10.0,3, 0,0,0, 0},       /* 符号精修最优代表(零体积差) */
    {"q2",  '2',  0.0,0, 0,0,0, 0},       /* Q1+Q2 残差 2bit: 真杠杆, +0.84GiB/层, 预算门控 */
};
#define NCAND (int)(sizeof(CANDS)/sizeof(CANDS[0]))
/* ★最小体积模式(DS4_MINVOL=1)★ 计划表驱动逐层档位(DS4_MV_BASELINE+DS4_VQ_RPLAN),
 * 产物两文件: dql(1bit 基座+记录+z [+内嵌 g2hot 热载荷]) + opt。 */
static const cand_t MV_G10={"g10",'g',10.0,3,0,0,0,0};
/* 热集=FP 锚路由权重和 top-K(过程内口径, 每层现场算, 零静态表; X9 实测 K=16 覆盖~95%;
 * freq_to_norms 教训: 按路由权重和排序, 不按命中 count)。填 G2_* 全局 → quant_apply/
 * export_worker 既有 g2_hot_slot 门直接生效。 */
static int hot_from_anchor(int L,int S,int K){
    double wsum[256]={0};
    const int32_t*aidx=ANC.ridx+(size_t)L*S*NACT; const float*arw=ANC.rw+(size_t)L*S*NACT;
    for(int s=0;s<S;s++)for(int a=0;a<NACT;a++){ int e=aidx[(size_t)s*NACT+a];
        if(e>=0&&e<256) wsum[e]+=arw[(size_t)s*NACT+a]; }
    for(int e=0;e<256;e++) G2_SLOT[L][e]=-1;
    int k=0;
    for(;k<K&&k<256;k++){ int be=-1; double bw=0;
        for(int e=0;e<256;e++) if(G2_SLOT[L][e]<0&&wsum[e]>bw){bw=wsum[e];be=e;}
        if(be<0)break;
        G2_SLOT[L][be]=(int16_t)k; G2_IDS[L][k]=(uint16_t)be; }
    G2_K[L]=k; GO2B_HOT=(k>0);
    return k;
}
static const cand_t *TAB=CANDS; static int NTAB=NCAND;   /* plan_lookup 用活动表 */

/* held 泛化评分(越小越好): held 行 hc 状态 vs FP 锚, align + 0.5·classify(维度方差=感知加权) */
static double held_score(const float*H,int S,int n_fit,int L,double*relh){
    size_t row=(size_t)HCM*DIM; int nh=S-n_fit;
    const float*A=H+(size_t)n_fit*row;
    const float*B=ANC.H+(size_t)L*S*row+(size_t)n_fit*row;
    double e2=0,a2=0;
    for(size_t i=0;i<(size_t)nh*row;i++){ double d=(double)A[i]-B[i]; e2+=d*d; a2+=(double)B[i]*B[i]; }
    if(relh)*relh=sqrt(e2/(a2+1e-30));
    float *wv=malloc(row*4); ds4_loss_dim_variance(B,(uint32_t)nh,(uint32_t)row,wv);
    double la=ds4_loss_align(A,B,(uint32_t)nh,(uint32_t)row);
    double lc=ds4_loss_classify(A,B,wv,(uint32_t)nh,(uint32_t)row);
    free(wv);
    return la+0.5*lc;
}
static const char*ckpt_dir(void){ const char*p=getenv("DS4_CKPT_DIR"); return p?p:"/tmp/ds4q_ckpt"; }
static const char*plan_path(void){ const char*p=getenv("DS4_PLAN"); return p?p:"/tmp/ds4quant_plan.txt"; }
/* 读 plan 中第 L 层已锁链式 relh(增长率门用; 无记录=-1) */
static double plan_relh_of(int L){
    FILE*f=fopen(plan_path(),"r"); if(!f) return -1;
    char ln[256]; double out=-1;
    while(fgets(ln,sizeof(ln),f)){ int li; double rv;
        if(sscanf(ln,"L=%d cand=%*s relh=%lf",&li,&rv)==2 && li==L) out=rv; }
    fclose(f); return out;
}
static int ckpt_load(int L,float*H,size_t n,uint64_t idh){
    char p[512]; snprintf(p,sizeof(p),"%s/L%02d.bin",ckpt_dir(),L);
    FILE*f=fopen(p,"rb"); if(!f)return 0;
    uint64_t hd[2]; int ok = fread(hd,8,2,f)==2 && hd[0]==idh && hd[1]==(uint64_t)n
                          && fread(H,4,n,f)==n;
    fclose(f); return ok;
}
static void ckpt_save(int L,const float*H,size_t n,uint64_t idh){
    char p[512]; snprintf(p,sizeof(p),"%s/L%02d.bin",ckpt_dir(),L);
    FILE*f=fopen(p,"wb"); if(!f)return;
    uint64_t hd[2]={idh,(uint64_t)n};
    fwrite(hd,8,2,f); fwrite(H,4,n,f); fclose(f);
}
/* 影子冠军链 ckpt(断点续跑不重算影子) */
static int ckpt_load_s(int L,float*H,size_t n,uint64_t idh){
    char p[512]; snprintf(p,sizeof(p),"%s/L%02d.sh.bin",ckpt_dir(),L);
    FILE*f=fopen(p,"rb"); if(!f)return 0;
    uint64_t hd[2]; int ok=fread(hd,8,2,f)==2&&hd[0]==idh&&hd[1]==(uint64_t)n&&fread(H,4,n,f)==n;
    fclose(f); return ok;
}
static void ckpt_save_s(int L,const float*H,size_t n,uint64_t idh){
    char p[512]; snprintf(p,sizeof(p),"%s/L%02d.sh.bin",ckpt_dir(),L);
    FILE*f=fopen(p,"wb"); if(!f)return;
    uint64_t hd[2]={idh,(uint64_t)n};
    fwrite(hd,8,2,f); fwrite(H,4,n,f); fclose(f);
}
/* rr 判决链 H2 的 ckpt(断点续跑与主链同粒度; 头校验=rr 锚 ids 哈希) */
static int ckpt2_load(int L,float*H,size_t n){
    char p[512]; snprintf(p,sizeof(p),"%s/L%02d.rr.bin",ckpt_dir(),L);
    FILE*f=fopen(p,"rb"); if(!f)return 0;
    uint64_t hd[2]; int ok=fread(hd,8,2,f)==2&&hd[0]==ANC2.idh&&hd[1]==(uint64_t)n&&fread(H,4,n,f)==n;
    fclose(f); return ok;
}
static void ckpt2_save(int L,const float*H,size_t n){
    char p[512]; snprintf(p,sizeof(p),"%s/L%02d.rr.bin",ckpt_dir(),L);
    FILE*f=fopen(p,"wb"); if(!f)return;
    uint64_t hd[2]={ANC2.idh,(uint64_t)n};
    fwrite(hd,8,2,f); fwrite(H,4,n,f); fclose(f);
}
/* plan 文件: 每行 "L=%d cand=%s relh=..."; 同层多行取最后(允许追加覆盖) */
static int plan_lookup(int L){
    FILE*f=fopen(plan_path(),"r"); if(!f)return -1;
    char ln[256]; int found=-1;
    while(fgets(ln,sizeof(ln),f)){ int l; char nm[32];
        if(sscanf(ln,"L=%d cand=%31s",&l,nm)==2 && l==L)
            for(int c=0;c<NTAB;c++) if(!strcmp(nm,TAB[c].name)) found=c;
    }
    fclose(f); return found;
}
static void plan_append(int L,const char*nm,double relh){
    FILE*f=fopen(plan_path(),"a"); if(!f)return;
    fprintf(f,"L=%d cand=%s relh=%.4f\n",L,nm,relh); fclose(f);
}
/* minvol plan 行(rr 预算门版): 附 rr 里程碑值(审计+RESUME 参考; plan_lookup 前两字段不受影响) */
static void plan_append_rr(int L,const char*nm,double relh,double rrm){
    FILE*f=fopen(plan_path(),"a"); if(!f)return;
    fprintf(f,"L=%d cand=%s relh=%.4f gate=%.4f\n",L,nm,relh,rrm); fclose(f);
}
/* minvol plan 复用: 动态名("m448"/"m448h"/"g10"/"g10h")解析, 返回 cfg('m'/'g', 0=无记录) */
static char plan_lookup_mv(int L,char*nm_out){
    FILE*f=fopen(plan_path(),"r"); if(!f)return 0;
    char ln[256],got[32]={0};
    while(fgets(ln,sizeof(ln),f)){ int l; char nm[32];
        if(sscanf(ln,"L=%d cand=%31s",&l,nm)==2&&l==L) snprintf(got,sizeof(got),"%s",nm); }
    fclose(f);
    if(!got[0]) return 0;
    if(nm_out) snprintf(nm_out,16,"%s",got);
    return got[0]=='m'?'m':'g';
}
static void set_cand(const cand_t*c){
    dq_signref_mu=c->mu; dq_signref_rounds=c->rounds;
    /* ★候选未带 lz(=0)时回落 env(2026-08-23): pure 模式 set_cand(&MV_G10) 曾把
     * DS4_LZ=8 覆写清零 —— ZDIAG LZRANK=0 实锤。档位自带 lz 仍优先。 */
    LZRANK = c->lz ? c->lz : (getenv("DS4_LZ") ? atoi(getenv("DS4_LZ")) : 0);
    if(c->lztr) LZTR=c->lztr; if(c->lz)LZLAMBDA=c->lzlam;
}
/* verdict-lite: 只算 held Σmin/KL(里程碑探针用) */
static void verdict_lite(const float*lf,const float*lq,const long*ids,int S,int n_fit,double*smin,double*kl){
    int s0=n_fit,s1=S-1; if(s1-s0<1)s0=0;
    double sm=0,k=0; int n=0;
    for(int s=s0;s<s1;s++){
        const float*a=lf+(size_t)s*VOCAB,*b=lq+(size_t)s*VOCAB;
        float ma=a[0],mb=b[0];
        for(int v=1;v<VOCAB;v++){ if(a[v]>ma)ma=a[v]; if(b[v]>mb)mb=b[v]; }
        double sa=0,sb=0; for(int v=0;v<VOCAB;v++){ sa+=exp((double)a[v]-ma); sb+=exp((double)b[v]-mb); }
        double lsa=log(sa),lsb=log(sb);
        for(int v=0;v<VOCAB;v++){ double lpf=(double)a[v]-ma-lsa,lpq=(double)b[v]-mb-lsb;
            double pf=exp(lpf),pq=exp(lpq); sm+=pf<pq?pf:pq; if(pf>0)k+=pf*(lpf-lpq); }
        n++;
    }
    *smin=sm/n; *kl=k/n;
}
/* 还原率里程碑探针: 前 L 层用已调优结果 + 后缀纯 FP → 真实 Σmin(=若后面全无损的当前还原率上界)。
 * 对照目标线, 不达标趋势早停, 不瞎跑。 */
static double restore_probe(const float*H,int L,const long*ids,int S,int n_fit){
    size_t lstride=(size_t)S*HCM*DIM;
    float *Hp=malloc(lstride*4); memcpy(Hp,H,lstride*4);
    fprintf(stderr,"[里程碑探针 L%02d] 后缀 FP 前向...\n",L);
    for(int l=L+1;l<NLAYERS;l++){ LW W2=load_layer(l);
        fprintf(stderr,"  probe L%02d F\n",l);
        layer_fwd(l,&W2,Hp,ids,S,n_fit,0,'F',NULL); free_layer(&W2); }
    float *hcfn=st_read_weight(&C,"hc_head_fn",NULL,NULL),*hcb=st_read_weight(&C,"hc_head_base",NULL,NULL),*hcs=st_read_weight(&C,"hc_head_scale",NULL,NULL);
    float *norm=st_read_weight(&C,"norm.weight",NULL,NULL);
    float *lp=malloc((size_t)S*VOCAB*4); head_fwd_stream(Hp,S,hcfn,hcb,hcs,norm,lp);
    double sm,kl; verdict_lite(ANC.logits,lp,ids,S,n_fit,&sm,&kl);
    fprintf(stderr,"★[还原率里程碑] 前%d层已调优+后缀无损: Σmin=%.4f KL=%.3f  (目标线0.52/0.60; FP=1.0)\n",L+1,sm,kl);
    printf("MILESTONE L=%d smin=%.4f kl=%.4f\n",L,sm,kl); fflush(stdout);
    free(Hp);free(lp);free(hcfn);free(hcb);free(hcs);free(norm);
    return sm;
}
/* ---- rr_hard 判决锚初始化(统一标准): 载/建 ANC2, 全程一次 ---- */
static int rr_ids_load(const char*p){
    FILE*f=fopen(p,"r"); if(!f) return 0;
    long v; int n=0,cap=4096; g_ids2=malloc((size_t)cap*sizeof(long));
    while(fscanf(f,"%ld",&v)==1){ if(n>=cap){cap*=2;g_ids2=realloc(g_ids2,(size_t)cap*sizeof(long));} g_ids2[n++]=v; }
    fclose(f); g_S2=n; return n>0;
}
static void rr_anchor_init(void){
    const char*ip=getenv("DS4_RR_IDS"); if(!ip) return;
    const char*ap=getenv("DS4_ANCHOR2");
    if(!ap){ fprintf(stderr,"[rr锚] 需 DS4_ANCHOR2 路径 — 硬拒\n"); exit(2); }
    if(!rr_ids_load(ip)){ fprintf(stderr,"[rr锚] ids 读取失败 %s — 硬拒\n",ip); exit(2); }
    fprintf(stderr,"[rr锚] 判决语料 %s S=%d (统一标准: 判决口径=rr_hard, 冠军尺 0.7680/0.3602)\n",ip,g_S2);
    anchor_t sv=ANC; int svok=ANC_OK;
    char svpath[512]; snprintf(svpath,sizeof(svpath),"%s",anchor_path());
    setenv("DS4_ANCHOR",ap,1);
    memset(&ANC,0,sizeof(ANC)); ANC_OK=0;
    uint64_t idh2=dq_ids_hash(g_ids2,g_S2);
    if(anchor_load(g_S2,idh2)) fprintf(stderr,"[rr锚] 命中缓存 %s (FP 遍跳过)\n",ap);
    else {
        fprintf(stderr,"[rr锚] 建锚2: FP 遍 %d tok(一次性)...\n",g_S2);
        anchor_alloc(g_S2); ANC.idh=idh2; ANC_BUILD=1;
        ANC.logits=fwd_all(g_ids2,g_S2,g_S2,0,NULL);
        ANC_BUILD=0;
        if(anchor_save()) fprintf(stderr,"[rr锚] 已写 %s\n",ap);
    }
    ANC2=ANC; ANC2_OK=1;
    ANC=sv; ANC_OK=svok;
    setenv("DS4_ANCHOR",svpath,1);
}
/* rr 里程碑(判决口径): H2 前缀量化链 + 后缀 FP → 全段 Σmin/KL vs rr FP logits。
 * H2=裸胜者档链(不含 coadapt z 微增益) ⇒ 保守下界; 终验用完整回放取真值。 */
static double rr_probe(const float*H2,int L){
    size_t l2=(size_t)g_S2*HCM*DIM;
    float *Hp=malloc(l2*4); memcpy(Hp,H2,l2*4);
    for(int l=L+1;l<NLAYERS;l++){ LW W2=load_layer(l);
        layer_fwd(l,&W2,Hp,g_ids2,g_S2,g_S2,0,'F',NULL); free_layer(&W2); }
    float *hcfn=st_read_weight(&C,"hc_head_fn",NULL,NULL),*hcb=st_read_weight(&C,"hc_head_base",NULL,NULL),*hcs=st_read_weight(&C,"hc_head_scale",NULL,NULL);
    float *norm=st_read_weight(&C,"norm.weight",NULL,NULL);
    float *lp=malloc((size_t)g_S2*VOCAB*4); head_fwd_stream(Hp,g_S2,hcfn,hcb,hcs,norm,lp);
    double sm,kl; verdict_lite(ANC2.logits,lp,g_ids2,g_S2,0,&sm,&kl);
    fprintf(stderr,"★★[rr里程碑|判决口径|冠军尺0.7680/0.3602] 前%d层量化+后缀无损: Σmin=%.4f KL=%.3f\n",L+1,sm,kl);
    printf("RRMILESTONE L=%d smin=%.4f kl=%.4f\n",L,sm,kl); fflush(stdout);
    free(Hp);free(lp);free(hcfn);free(hcb);free(hcs);free(norm);
    return sm;
}
/* rr 链单层前向(候选/胜者态, 预算门与正式推进共用): H2s→H2d(可同址)。
 * 量化字节与定稿态同(确定性+EXP_ 覆盖=校准仍 v5mini fit 行, 判决语料零污染); COADAPT 关。 */
static void rr_step(int L,LW*W,const float*H2s,float*H2d,int S,int n_fit,char cfg){
    size_t l2=(size_t)g_S2*HCM*DIM;
    if(H2d!=H2s) memcpy(H2d,H2s,l2*4);
    anchor_t sv=ANC; int svok=ANC_OK;
    if(!EXP_FIN)EXP_FIN=malloc((size_t)S*DIM*4);
    if(!EXP_IDX)EXP_IDX=malloc((size_t)S*NACT*sizeof(int));
    memcpy(EXP_FIN,sv.fin+(size_t)L*S*DIM,(size_t)S*DIM*4);
    { const int32_t*ai=sv.ridx+(size_t)L*S*NACT;
      for(size_t i2=0;i2<(size_t)S*NACT;i2++)EXP_IDX[i2]=(int)ai[i2]; }
    EXP_L=L;
    ANC=ANC2; ANC_OK=1;
    int co_sv=COADAPT; COADAPT=0;
    lstat_t st2; memset(&st2,0,sizeof(st2));
    layer_fwd(L,W,H2d,g_ids2,g_S2,n_fit,1,cfg,&st2);
    COADAPT=co_sv;
    ANC=sv; ANC_OK=svok; EXP_L=-1;
}
static float *fwd_all_tune(const long*ids,int S,int n_fit,char*plan_out){
    size_t lstride=(size_t)S*HCM*DIM;
    uint64_t idh=ANC.idh;
    mkdir(ckpt_dir(),0755);
    double gain_th=0.15; if(getenv("DS4_TUNE_GAIN")) gain_th=atof(getenv("DS4_TUNE_GAIN"));
    float *H=malloc(lstride*4),*Hw=malloc(lstride*4),*Hb=malloc(lstride*4),*Hf2=malloc(lstride*4);
    { float *emb=st_read_weight(&C,"embed.weight",NULL,NULL);
      for(int s=0;s<S;s++)for(int j=0;j<HCM;j++)memcpy(H+((size_t)s*HCM+j)*DIM,emb+(size_t)ids[s]*DIM,(size_t)DIM*4);
      free(emb); }
    int minvol=getenv("DS4_MINVOL")?1:0;
    /* ★隔离探针(R24 D1, 2026-07-30 用户"先跑两个代表层试一试"): 上游=FP 锚定直通,
     * 只评 DS4_MV_PROBE_L 层 — 干净的单层判决(无上游漂移), R24 vs 冠军档同足对比。 */
    int g_probe_l=getenv("DS4_MV_PROBE_L")?atoi(getenv("DS4_MV_PROBE_L")):-1;
    double mv_alpha=getenv("DS4_MINVOL_ALPHA")?atof(getenv("DS4_MINVOL_ALPHA")):1.03;
    /* 活 α(2026-07-28 用户: "为啥α不能是活的"): 里程碑斜率外推终点 Σmin_proj,
     * proj<T−0.03 → α 收 0.03; proj>T+0.03 → α 放 0.02(顶 1.15)。
     * ★统一标准后: 外推/目标线一律 rr 口径(冠军尺), 默认终线 0.77≈v4bf 0.7680。 */
    double mv_target=getenv("DS4_MINVOL_TARGET")?atof(getenv("DS4_MINVOL_TARGET")):0.768;
    double mprev=-1;   /* rr 里程碑滚动值(预算门状态; -1=未初始化, RESUME 时现算恢复) */
    /* rr 判决链 H2(统一标准): 与主链同粒度推进, embed 起点 */
    size_t lstride2=ANC2_OK?(size_t)g_S2*HCM*DIM:0; float *H2=NULL;
    if(minvol&&ANC2_OK){
        H2=malloc(lstride2*4);
        float *emb2=st_read_weight(&C,"embed.weight",NULL,NULL);
        for(int s=0;s<g_S2;s++)for(int j=0;j<HCM;j++)memcpy(H2+((size_t)s*HCM+j)*DIM,emb2+(size_t)g_ids2[s]*DIM,(size_t)DIM*4);
        free(emb2);
    }
    if(minvol){
        RB_SEQ=getenv("DS4_ROUTE_SEQ")?1:0;
        if(RB_SEQ) fprintf(stderr,"[序贯路由] 开: 每层 FIT Δb + α{1.0,2.5,4.0} 扫(负收益自动关), 定稿/推进带修正, 收官落盘\n");
        fprintf(stderr,"\n[最小体积·计划表驱动] 每层档位由 DS4_VQ_RPLAN 计划表定; "
                       "地板门 α=%.2f; v5mini 里程碑每5层=护栏\n",mv_alpha); }
    else fprintf(stderr,"\n[渐进调优v2·目标导向] 免费候选(r/g)选最优; q2 挽回≥%.0f%% 才升位(+0.84GiB/层); 里程碑 L9/20/31\n",gain_th*100);
    for(int L=0;L<NLAYERS;L++){
        int pc=-1; char sel_nm[16]="?"; char sel_cfg=0;
        if(minvol&&g_probe_l>=0){
            if(L<g_probe_l) continue;   /* 探针: 前层不量化不推进(链在探针层入口一次性锚恢复) */
            if(L==g_probe_l&&L>0&&ANC_OK){
                memcpy(H,ANC.H+(size_t)(L-1)*lstride,lstride*4);
                fprintf(stderr,"[探针] L00..L%02d=FP 锚定直通(隔离口径), 评 L%02d\n",L-1,L);
            }
        }
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
            int mv_base=getenv("DS4_MV_BASELINE")?1:0;   /* 基线模式: 全链冠军配方(g10h), 产 BT */
            int coad_save=COADAPT; COADAPT=0; GO2B_HOT=0;   /* 基线/纯m档必须无热污染 */
            /* ★热启动(2026-07-30): 上层锁 'g' 族 ⇒ m 档无望带, 跳裸基线+二分, 档-1 起试 */
            int warm = !mv_base && g_mv_pcfg=='g' && getenv("DS4_MINVOL_HIST")
                       && !getenv("DS4_MV_NOWARM");
            double rel0=0,s_base=0,relh_floor=-1;
            /* ★现场地板线(2026-07-30 L03 实锤: 静态 V4BF 线含 v4/v5mini 语料水位差 ~3%,
             * 冠军自家配方带 coadapt 也过不了自家线 0.0077 → 薄边距层被迫花体积补口径差 =
             * 本末倒置)。DS4_MV_FLOOR_LINE=1: 每层裸评 g10h16(冠军配方)于本链上文, 线=实测
             * +EPS — 链公平/零语料差/冠军档构造性恒过(真零债); 兼作二分 thr 基线。 */
            int floor_line = getenv("DS4_MV_FLOOR_LINE")?1:0;
            /* ★影子冠军链门(2026-07-30 终版, 用户"每层贪心最小反噬"方案性 bug 裁决):
             * 冠军配方(g10h16+coadapt)从影子自身上文前向一步, 链式 held=本层门线 —
             * 同语料/绝对轨迹(不随我们链漂移)/构造可达(封顶≈线)。兼作二分 thr 基线,
             * 替代 g10 裸基线。影子 ckpt 断点续; 缺档(旧跑)以我们链一次性播种并声明。 */
            static float *Hs=NULL; static int hs_L=-1;
            double shadow_line=-1;
            {int mvfl = getenv("DS4_MINVOL_FLOOR")?atoi(getenv("DS4_MINVOL_FLOOR")):1;
            int shadow_on = getenv("DS4_MINVOL_HIST")&&!mvfl&&!mv_base&&!floor_line;
            if(shadow_on){
                if(!Hs) Hs=malloc(lstride*4);
                if(hs_L!=L-1){
                    if(L>0&&ckpt_load_s(L-1,Hs,lstride,idh)){ hs_L=L-1;
                        fprintf(stderr,"L%02d [影子] ckpt 恢复(至 L%02d)\n",L,L-1); }
                    else { memcpy(Hs,H,lstride*4); hs_L=L-1;
                        fprintf(stderr,"L%02d [影子] %s播种(=当前链上文)\n",L,L>0?"缺 ckpt 一次性":"L0 "); }
                }
                time_t ts0=time(NULL);
                int coS=COADAPT; COADAPT=1;
                hot_from_anchor(L,S,MV_HOTK);
                memcpy(Hw,Hs,lstride*4); set_cand(&MV_G10);
                lstat_t sts; memset(&sts,0,sizeof(sts));
                fprintf(stderr,"L%02d 影子冠军链(g10h16定稿) ",L);
                layer_fwd(L,&W,Hw,ids,S,n_fit,1,'g',&sts);
                s_base=held_score(Hw,S,n_fit,L,&shadow_line);
                memcpy(Hs,Hw,lstride*4); hs_L=L; ckpt_save_s(L,Hs,lstride,idh);
                COADAPT=coS; GO2B_HOT=0;
                rel0=shadow_line;
                fprintf(stderr,"L%02d [影子线] held=%.4f [%lds]\n",L,shadow_line,(long)(time(NULL)-ts0));
            }
            if(!mv_base&&!shadow_on&&(floor_line||!warm)){
            time_t tc0=time(NULL);
            if(floor_line) hot_from_anchor(L,S,MV_HOTK); else GO2B_HOT=0;
            memcpy(Hw,H,lstride*4); set_cand(&MV_G10);
            lstat_t st0; memset(&st0,0,sizeof(st0));
            fprintf(stderr,"L%02d %s ",L,floor_line?"g10h16地板(裸)":"g10 ");
            layer_fwd(L,&W,Hw,ids,S,n_fit,1,'g',&st0);
            s_base=held_score(Hw,S,n_fit,L,&rel0);
            if(floor_line){ relh_floor=rel0; GO2B_HOT=0; }
            fprintf(stderr," held relL2=%.4f score=%.5g [%s %lds]\n",rel0,s_base,
                    floor_line?"地板":"裸基线",(long)(time(NULL)-tc0));
            memcpy(Hb,Hw,lstride*4);
            }}
            char win_nm[16]="g10"; char win_cfg='g'; int win_r=0,win_hot=0;
            double win_vol=1.0625; brel=rel0;
            if(mv_base){   /* 基线模式: 逐层直接冠军配方 g10h(不粗筛不建流形), 产同尺 BT */
                if(getenv("DS4_VQ_RPLAN")&&dq_vq_on()){
                    /* ★R28: 计划表驱动(每层档位+热数); 热集按本层 hot 数从锚重建 */
                    vq_rplan(L);
                    hot_from_anchor(L,S,g_vq_hot);
                    snprintf(win_nm,sizeof(win_nm),"v%dx%d h%d",g_vq_dim,g_vq_nc,g_vq_hot);
                    win_cfg='g'; win_r=0; win_hot=g_vq_hot;
                    win_vol=(double)(16+g_vq_nc*g_vq_dim*2+MOEI*2
                             +((size_t)MOEI*DIM/g_vq_dim*(g_vq_nc<=256?8:(g_vq_nc<=512?9:(g_vq_nc<=1024?10:12)))+7)/8+1)
                            *8.0/((double)MOEI*DIM);
                } else {
                snprintf(win_nm,sizeof(win_nm),"g10h"); win_hot=MV_HOTK;
                win_vol=(16*2.25+240*1.0625)/256.0;
                }
            } else {
            fprintf(stderr,"[minvol] 贪心 rank 二分已随流形档移除 — 用 DS4_MV_BASELINE=1 + DS4_VQ_RPLAN 计划表模式\n");
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
            if(warm){
                /* ★热启动阶梯: 上层档-1(体积回收探针) → 上层档; 再欠由升档链往上翻 */
                up_t w0,w1; memset(&w0,0,sizeof(w0)); memset(&w1,0,sizeof(w1));
                if(g_mv_phot==0){        /* 上层=g10 → 探 g10, 回 g10h16 */
                    snprintf(w0.nm,16,"g10");   w0.cfg='g'; w0.vol=1.0625;
                    snprintf(w1.nm,16,"g10h%d",MV_HOTK); w1.cfg='g'; w1.hot=MV_HOTK;
                    w1.vol=((256.0-MV_HOTK)*1.0625+MV_HOTK*2.25)/256.0;
                } else if(g_mv_phot<=MV_HOTK){   /* 上层=g10h16 → 探 g10, 回 g10h16 */
                    snprintf(w0.nm,16,"g10");    w0.cfg='g'; w0.vol=1.0625;
                    snprintf(w1.nm,16,"g10h%d",MV_HOTK); w1.cfg='g'; w1.hot=MV_HOTK;
                    w1.vol=((256.0-MV_HOTK)*1.0625+MV_HOTK*2.25)/256.0;
                } else {                 /* 上层=g10hN → 探 g10h(N/2), 回 g10hN */
                    int nh=g_mv_phot/2;
                    snprintf(w0.nm,16,"g10h%d",nh); w0.cfg='g'; w0.hot=nh;
                    w0.vol=((256.0-nh)*1.0625+nh*2.25)/256.0;
                    snprintf(w1.nm,16,"g10h%d",g_mv_phot); w1.cfg='g'; w1.hot=g_mv_phot;
                    w1.vol=((256.0-g_mv_phot)*1.0625+g_mv_phot*2.25)/256.0;
                }
                ups[nup++]=w0; ups[nup++]=w1;
                fprintf(stderr,"L%02d [热启动] 上层锁 %c/h%d → 起试 %s→%s (跳裸基线+rank二分)\n",
                        L,g_mv_pcfg,g_mv_phot,ups[0].nm,ups[1].nm);
            } else {
              up_t w0; memset(&w0,0,sizeof(w0));
              snprintf(w0.nm,16,"%s",win_nm); w0.cfg=win_cfg; w0.r=win_r; w0.hot=win_hot; w0.vol=win_vol;
              ups[nup++]=w0;
              /* 升档梯: g10 → ★g10hN 往上翻(2026-07-30 用户令): 超标不卡死体积,
               * 活跃专家扩热处理, 债不再传链。 */
              up_t lad[1]={{"g10",'g',0,0,1.0625}};
              for(int u=0;u<1;u++) if(lad[u].vol>win_vol+1e-9&&nup<8) ups[nup++]=lad[u]; }
            static double MV_EPS=-1;
            if(MV_EPS<0) MV_EPS=getenv("DS4_MINVOL_EPS")?atof(getenv("DS4_MINVOL_EPS")):0.0005;
            /* ★两遍调度(2026-07-29 用户"快速确定最小体积, 精修选中层")★
             * DS4_MINVOL_FLOOR=0(快扫遍): 跳地板/碑/升档, 粗筛胜者直接定稿导出(~8min/层);
             * =1(精修遍, 默认): 逐层地板门(对 RESUME 选中的层生效)。 */
            static int MV_FLOOR=-1;
            if(MV_FLOOR<0) MV_FLOOR=getenv("DS4_MINVOL_FLOOR")?atoi(getenv("DS4_MINVOL_FLOOR")):1;
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
            static double MV_SOFT=-1;
            if(MV_SOFT<0) MV_SOFT=getenv("DS4_MINVOL_SOFT")?atof(getenv("DS4_MINVOL_SOFT")):0.05;   /* 纯防灾(v10 实测冠军配方浅层每层耗~0.046, 门不误咬正常层) */
            /* 里程碑评估已删除(2026-08-18 用户令): RESUME 首层不再跑后缀 FP 探针
             * (单线程 38 层前向 ~12 分钟)。mprev 保持 -1 = 无碑语义, 软门首层跳过。 */
            /* ★门语义 A(2026-07-29 用户裁决, 索引错位已纠): 逐层对齐冠军谱系 v1 历史链
             * (reports/v1_inherit_chain.txt, INHERIT=链式出口 relh, 同名同式)。
             * 索引验证(fable5 原文 L1=0.1129/L20=0.4526 ↔ 文件第2/第21值): **hist[i]=Li 层值,
             * 43 层一一对应, 门线=hist[L]**。hist[0]=0.0000(v1 链起点记录)→ L0 特判借 hist[1]
             * 并声明。判据=relh ≤ 门线; 语料偏差(历史530tok vs 现v5mini)由单层实测暴露。 */
            static double MV_HIST[64]; static int MV_NHIST=-1;
            if(MV_NHIST<0){ MV_NHIST=0;
                const char*hp=getenv("DS4_MINVOL_HIST");
                if(hp){ FILE*hf=fopen(hp,"r");
                    if(hf){ double v; while(MV_NHIST<64&&fscanf(hf,"%lf",&v)==1) MV_HIST[MV_NHIST++]=v; fclose(hf); }
                    const char*tf=getenv("DS4_MV_LINE_TIGHTEN");   /* 机制审计杠杆: 门线×f 强触发升档链 */
                    if(tf){ double f=atof(tf); for(int i=0;i<MV_NHIST;i++) MV_HIST[i]*=f;
                        fprintf(stderr,"[门A] ★审计: 门线×%.2f 收紧★\n",f); }
                    fprintf(stderr,"[门A] 历史通过线载入 %d 值 ← %s (门线=hist[L]; hist[0]=%.4f%s)\n",
                            MV_NHIST,hp?hp:"",MV_NHIST>0?MV_HIST[0]:0.0,
                            (MV_NHIST>0&&MV_HIST[0]<=0.0)?", ≤0→L0 借 hist[1]":"") ; } }
            int hist_on = (MV_NHIST>=NLAYERS) && !MV_FLOOR;
            double BT=mv_target;
            double hist_line = hist_on ? ((L==0&&MV_HIST[0]<=0.0)?MV_HIST[1]:MV_HIST[L<MV_NHIST?L:MV_NHIST-1]) : 0;
            if(hist_on&&shadow_line>0){
                /* ★影子线优先: 冠军配方同语料自链轨迹 — 绝对/可达/无口径差 */
                fprintf(stderr,"L%02d [门A·影子线] 线=%.4f(冠军配方同语料链) V4BF参考=%.4f (口径差 %+.1f%%)\n",
                        L,shadow_line,hist_line,hist_line>0?(shadow_line/hist_line-1.0)*100.0:0.0);
                hist_line=shadow_line;
            }
            else if(hist_on&&relh_floor>=0){
                /* ★现场地板覆盖静态线: 链公平, V4BF 静态线仅留作语料水位差遥测 */
                static double MV_FEPS=-1;
                if(MV_FEPS<0) MV_FEPS=getenv("DS4_MV_FLOOR_EPS")?atof(getenv("DS4_MV_FLOOR_EPS")):0.0005;
                double oldl=hist_line; hist_line=relh_floor+MV_FEPS;
                fprintf(stderr,"L%02d [门A·现场地板] 线=%.4f(g10h16裸+%.4g) V4BF参考=%.4f (水位差 %+.1f%%)\n",
                        L,hist_line,MV_FEPS,oldl,oldl>0?(relh_floor/oldl-1.0)*100.0:0.0);
            }
            else if(hist_on&&L>0&&L<MV_NHIST&&MV_HIST[L-1]>1e-9){
                /* ★增长率门(2026-07-30 用户"算法设计有问题重新优化"终版): 绝对线撞语料口径墙
                 * (L08 缺口>档梯回收=必硬停), 相对地板线无锚螺旋(+30%) — 门=冠军谱系逐层
                 * 增长率: req=ours(L-1)×hist[L]/hist[L-1]。比值内语料口径差消掉; 轨迹形状
                 * 钉死不漂; 终链 ≤ ours(0)/hist[0]×hist[42]=0.4616(L00 领先 11% 继承到终点)。 */
                double pv=plan_relh_of(L-1);
                if(pv>0){
                    double rt=MV_HIST[L]/MV_HIST[L-1], rq=pv*rt;
                    fprintf(stderr,"L%02d [门A·增长率] req=%.4f(上层链%.4f×谱系率%.4f) 绝对线参考=%.4f\n",
                            L,rq,pv,rt,hist_line);
                    hist_line=rq;
                }
            }
            double M_req = mv_base?-1e9:
                (MV_FLOOR? (M_champ-MV_EPS)
                 /* ★hist 门补 v16 标准容差(2026-07-30 L03 实锤: m768h coadapt 0.2244 距线
                  * 0.0003 被拒→逼向更差且贵 55% 的 g 族 = 本末倒置; v16 用户定义语义=
                  * "实测 ≥ 冠军同层−0.0005 即不比冠军差", FLOOR 支路一直有, hist 漏掉) */
                 : (hist_on? hist_line+MV_EPS
                    : (H2? ((L>=NLAYERS-1)?BT:(BT+((mprev<0?1.0:mprev)-BT)*(double)(NLAYERS-1-L)/(double)(NLAYERS-L)))
                         : ((mprev<0?1.0:mprev)-MV_SOFT))));
            if(!MV_FLOOR&&!hist_on&&nup>3) nup=3;   /* 软门: 至多三试; ★门A 零债(2026-07-30
                                                     * 用户令): 升档到过门为止, 债不传链 */
            float *H2t=H2?malloc(lstride2*4):NULL;
            int usel=-1; double Msel=mprev; int mv_pass=0;
            double vbare[12]; for(int vi=0;vi<12;vi++) vbare[vi]=-1;   /* 各档裸值(不可达层最小同质档选用) */
            /* ★搜索效率三件套(2026-07-30 L04 实锤, 门判据不动): 本层实测 coadapt 增益驱动
             * 复判窗(静态 0.006 窗把够不着的档也复判, 90s/档); m 带饱和截断; g10 免重评 */
            static double g_mv_gain=0.004;            /* coadapt 增益跨层 EMA */
            static double MV_CMARG=-1;
            if(MV_CMARG<0) MV_CMARG=getenv("DS4_MV_COAD_MARGIN")?atof(getenv("DS4_MV_COAD_MARGIN")):0.012;
            double lay_gain=-1, m_prev_bare=-1; (void)m_prev_bare;   /* -1=本层未实测(用跨层 EMA) */
#define MV_CWIN() ({ double _lg=lay_gain>=0?lay_gain:g_mv_gain; double _w=_lg*1.25+0.0005; _w>MV_CMARG?MV_CMARG:_w; })
            int pure=(mv_base&&getenv("DS4_PURE_VQ"))?1:0;
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
                else if(hist_on) M=relf;      /* ★门A: 判据=链式 relh(与 v1 历史链同名同式, 方向小好), 免碑省时 */
                else if(MV_FLOOR&&is_champ) M=M_champ;  /* 地板门模式顶格免测 */
                else if(H2t){ rr_step(L,&W,H2,H2t,S,n_fit,c->cfg); M=rr_probe(H2t,L); }
                else M=restore_probe(Hw,L,ids,S,n_fit);   /* ★判决=v5 语料 HELD 区(裸态) */
                COADAPT=co2;
                int pass = mv_base || (hist_on ? (relf<=M_req) : (M>=M_req));
                vbare[u]=relf;
                if(!mv_base)
                fprintf(stderr,"L%02d [%s] %-6s 裸M=%.4f relh=%.4f 门=%.4f(%s) [%lds] → %s\n",
                        L,hist_on?"历史链门|relh":(MV_FLOOR?"地板门|v5held":(H2t?"预算门|rr":"软门|v5held")),c->nm,M,relf,M_req,
                        hist_on?"冠军历史held[L], relh≤":(MV_FLOOR?"地板−EPS":(H2t?"历史指标线":"上层−SOFT")),
                        (long)(time(NULL)-tf0),pass?"PASS":"升档");
                if(!pass&&hist_on&&!mv_base){
                    /* ★近失·coadapt 复判(2026-07-30 L03 实锤: 门线=冠军 coadapt 后过程值,
                     * 裸对线系统性吃亏 ~0.004-0.01≈一档, 逐层多锁档=本末倒置)。近失候选花
                     * 一次定稿态前向复判 — 状态(set_cand/热)沿用本候选裸评现场。 */
                    double cwin = MV_CWIN();
                    if(relf-M_req<=cwin){
                        time_t tn0=time(NULL);
                        memcpy(Hw,H,lstride*4); set_cand(&MV_G10);
                        lstat_t stn; memset(&stn,0,sizeof(stn));
                        fprintf(stderr,"L%02d %s(coad复判) ",L,c->nm);
                        layer_fwd(L,&W,Hw,ids,S,n_fit,1,'g',&stn);
                        double relc2; held_score(Hw,S,n_fit,L,&relc2);
                        fprintf(stderr,"L%02d [复判] %s coadapt relh=%.4f 门=%.4f [%lds] → %s\n",
                                L,c->nm,relc2,M_req,(long)(time(NULL)-tn0),relc2<=M_req?"PASS":"仍欠");
                        { double gg=relf-relc2; if(gg>lay_gain) lay_gain=gg;
                          g_mv_gain=0.5*g_mv_gain+0.5*(gg>0?gg:0); }
                        if(relc2<=M_req){ pass=1; M=relc2; relf=relc2; }
                    }
                }
                if(pass){ usel=u; Msel=M; brel=relf; mv_pass=1; break; }
                usel=u; Msel=M; brel=relf;
                if(!MV_FLOOR&&(u+1<nup||hist_on)&&u+1<12){
                    /* ★自适应升档: m 带按掉幅分级(微欠 r+128 / 中欠 m512h / 大欠 m768h),
                     * 'g' 带★往上翻(2026-07-30 用户令): 活跃专家按锚路由权重扩热
                     * (go2b 2.25), 16→32→64→128→256, 欠幅大跳级 — 体积不卡死, 债不传链。 */
                    double defi = hist_on ? (M-M_req) : (M_req-M);   /* 欠量(门A: relh 超出量) */
                    if(c->cfg=='g'){
                        if(!hist_on){ nup=u+1; fprintf(stderr,"L%02d [升档] g10 已顶格(软门) — 记债交 backfit\n",L); }
                        else if(c->hot>=256){ nup=u+1; /* 物理顶(全热): 环后零债裁决 */ }
                        else {
                            /* ★恒单步(2026-07-30 用户"别本末倒置"): g10 欠→必试 g10h16(=冠军
                             * 配方, 零债下构造性达线), 之上逐倍安全阀 — 禁跳级防越过最小可过档 */
                            int nh = c->hot? c->hot*2 : MV_HOTK;
                            if(nh>256) nh=256;
                            up_t nx; memset(&nx,0,sizeof(nx));
                            snprintf(nx.nm,sizeof(nx.nm),"g10h%d",nh); nx.cfg='g'; nx.hot=nh;
                            nx.vol=((256.0-nh)*1.0625+nh*2.25)/256.0;
                            ups[u+1]=nx; if(u+2>nup) nup=u+2;
                            fprintf(stderr,"L%02d [升档] 掉幅=%.4f → 往上翻 %s(bpw %.4f)\n",L,defi,nx.nm,nx.vol);
                        }
                    }
                }
            }
            if(hist_on&&!mv_pass&&!mv_base){
                /* ★不可达层终版(2026-07-30 L05 实锤: 全档梯+coadapt 距增长率 req 仍欠 0.0028
                 * = 该层物理不可达): 取与最佳裸值差≤0.001 的最小体积档(饱和区同质 — 如
                 * m768h 0.854bpw ≈ h256 2.25bpw), 链取实际值前进, 下层 req 自适应 —
                 * 不欠账不烧体积不硬停。DS4_MINVOL_STRICT=1 恢复硬停。 */
                double defi=brel-M_req;
                if(getenv("DS4_MINVOL_STRICT")){
                    fprintf(stderr,"L%02d ★门A档梯走完(%s)仍欠 %.4f — STRICT 硬停(rc=8)★\n",
                            L,ups[usel].nm,defi);
                    exit(8);
                }
                double vb=1e9; for(int vi=0;vi<nup&&vi<12;vi++) if(vbare[vi]>0&&vbare[vi]<vb) vb=vbare[vi];
                int bi=usel;
                for(int vi=0;vi<nup&&vi<12;vi++)
                    if(vbare[vi]>0&&vbare[vi]<=vb+0.001&&ups[vi].vol<ups[bi].vol) bi=vi;
                usel=bi; brel=vbare[bi]; Msel=vbare[bi];
                fprintf(stderr,"L%02d ★不可达层(req=%.4f 档梯最佳=%.4f 欠 %.4f) → 锁最小同质档 %s(%.4f bpw)★\n",
                        L,M_req,vb,vb-M_req,ups[usel].nm,ups[usel].vol);
                up_t*cs=&ups[usel];   /* 重臂选中档(走梯后续档覆盖过全局态) */
                if(cs->hot) hot_from_anchor(L,S,cs->hot); else GO2B_HOT=0;
            }
            static int rb_zstreak=0, rb_skip=0;
            if(RB_SEQ&&(!mv_base||getenv("DS4_ROUTE_SEQ_BASE"))&&rb_skip>0){
                /* ★自适应跳扫(2026-07-30 用户"每层太慢"): 深带 α 连选 0(v21 实测 L19+ 全 0,
                 * 每层白烧 460-530s) → 连 2 层 α=0 后跳 2 层再复测 */
                rb_skip--; RB_ALPHA_L[L]=0;
                fprintf(stderr,"L%02d [路由] 自适应跳过(前层连α=0, 余%d层后复测)\n",L,rb_skip);
            }
            else if(RB_SEQ&&(!mv_base||getenv("DS4_ROUTE_SEQ_BASE"))){   /* ★固定档也可序贯(2026-08-14 用户令"每层路由反修") */
                /* ★序贯路由: FIT 本层(胜者态一次前向) → α 三点扫(负收益自动关) →
                 *   之后的定稿/链推进即带路由修正, 下游层继承。 */
                up_t*cw=&ups[usel];
                if(W.t2ei){   /* ★哈希路由层(L0-L2)零漂移, α扫无对象直接跳(2026-08-14 自检针实锤48min/层白烧) */
                    RB_ALPHA_L[L]=0;
                    fprintf(stderr,"L%02d [路由] 哈希层跳过\n",L);
                } else {
                int co3=COADAPT; COADAPT=0;
                g_rb_fit_L=L; RB_ALPHA_L[L]=0;
                time_t trb=time(NULL);
                memcpy(Hw,H,lstride*4); set_cand(&MV_G10);
                lstat_t stq; memset(&stq,0,sizeof(stq));
                fprintf(stderr,"L%02d [路由FIT] ",L);
                layer_fwd(L,&W,Hw,ids,S,n_fit,1,cw->cfg,&stq);
                g_rb_fit_L=-1;
                rb_commit(L);
                /* ★α=0 并入扫点(2026-08-14): 复用模式 brel=0 判据失真, 基线由扫点自带, 取 argmin */
                double bestr=1e30; float besta=0.0f;
                const float ASCAN[3]={0.0f,2.5f,4.0f};
                for(int ai=0;ai<3;ai++){
                    RB_ALPHA_L[L]=ASCAN[ai];
                    memcpy(Hw,H,lstride*4);
                    lstat_t sta; memset(&sta,0,sizeof(sta));
                    layer_fwd(L,&W,Hw,ids,S,n_fit,1,cw->cfg,&sta);
                    double ra; held_score(Hw,S,n_fit,L,&ra);
                    fprintf(stderr,"L%02d [α扫] α=%.1f relh=%.4f%s\n",L,ASCAN[ai],ra,ra<bestr?" ↓":"");
                    if(ra<bestr){ bestr=ra; besta=ASCAN[ai]; }
                }
                RB_ALPHA_L[L]=besta; COADAPT=co3;
                }
                fprintf(stderr,"L%02d [路由] 选 α=%.1f bestr=%.4f\n",L,RB_ALPHA_L[L],RB_ALPHA_L[L]>0?0.0:0.0);
                if(RB_ALPHA_L[L]<=0.0f){ if(++rb_zstreak>=2) rb_skip=2; } else rb_zstreak=0;
            }
            int coadapted=(COADAPT>0&&(!mv_base||getenv("DS4_MV_COAD_BASE")));
            if(coadapted){   /* 胜者唯一一次 coadapt 定稿(全局态=胜者档已置位; DS4_MV_COAD_BASE=固定档也定稿(R24)) */
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
            if(!RB_SEQ&&getenv("DS4_ROUTE_BIAS_FIT")) rb_commit(L);   /* 固定档模式: 本层 Δb 折算入表(应用与否由 DS4_ROUTE_BIAS_ALPHA 定, 默认 1.0 即链上自带修正) */
            g_mv_pcfg=win_cfg; g_mv_phot=win_hot;   /* ★热启动记录: 下层档梯起点 */
            mv_vol_note(L,win_nm,win_vol);
            if(H2t){ memcpy(H2,H2t,lstride2*4); free(H2t); }
            mprev=Msel;
            if(!pure){
            memcpy(Hb,Hw,lstride*4);
            fprintf(stderr,"L%02d ★贪心选 %-6s held=%.4f w1w3bpw=%.4f rrM=%.4f\n",L,win_nm,brel,win_vol,mprev);
            }
            /* ⑤ 产物=两文件: dql(1bit w2+全记录+z + 热档:内嵌 g2hot) + opt */
            if(getenv("DS4_LAYER_DIR")){
                char dst[512];
                snprintf(dst,sizeof(dst),"%s/dql_L%02d.bin",getenv("DS4_LAYER_DIR"),L);
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
                    if(!RB_SEQ&&getenv("DS4_ROUTE_BIAS_FIT")) rb_commit(L);
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
                        double bud = getenv("DS4_VOL_BUDGET_GIB")?atof(getenv("DS4_VOL_BUDGET_GIB")):0.0;
                        fprintf(stderr,"L%02d ★体积★ 本层 %.1f MiB | 累计 %.3f GiB%s%s\n",
                                L,vs.st_size/1048576.0,acc_gib,
                                bud>0?" / 预算 ":"", bud>0?({static char b[32];snprintf(b,32,"%.2f GiB",bud);b;}):"");
                        const char*md=getenv("DS4_LAYER_DIR");
                        if(md){ char mp[512]; snprintf(mp,sizeof(mp),"%s/manifest.txt",md);
                            if(!mf) mf=fopen(mp,L==0?"w":"a");
                            if(mf){ fprintf(mf,"%d %lld\n",L,(long long)vs.st_size); fflush(mf); } }
                        if(bud>0&&acc_gib>bud){
                            fprintf(stderr,"★★体积闸: 累计 %.3f GiB > 预算 %.2f GiB — 停止(计划表需降档)★★\n",
                                    acc_gib,bud);
                            exit(9); }
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
        /* 护栏碑已删除(2026-08-18 用户令"删除里程碑评估"; 08-04 已裁"没有意义"):
         * 质量判决由收官 VERDICT/五指标全权。 */
        /* 单层探针(2026-07-28 用户: "先跑一层看看, 不要蒙头就跑"): 锁满 MAXL 层即收工。
         * plan/ckpt/产物均已落盘; zfile/zchain 也刷终值(全程跑在 main 尾做, 探针早退补齐);
         * RESUME=1 从下一层无损续跑。 */
        if(minvol&&getenv("DS4_MINVOL_MAXL")&&L+1>=atoi(getenv("DS4_MINVOL_MAXL"))){
            rb_save(); zfile_write(); zchain_write();
            fprintf(stderr,"[贪心探针] 已锁 %d 层(DS4_MINVOL_MAXL) → 提前收工; RESUME=1 续跑\n",L+1);
            exit(0);
        }
    }
    plan_out[NLAYERS]=0;
    rb_save();   /* 序贯路由: 收官落盘 Δb+α(交付侧车) */
    free(Hw);free(Hb);free(Hf2); if(H2)free(H2);
    float *hcfn=st_read_weight(&C,"hc_head_fn",NULL,NULL),*hcb=st_read_weight(&C,"hc_head_base",NULL,NULL),*hcs=st_read_weight(&C,"hc_head_scale",NULL,NULL);
    float *norm=st_read_weight(&C,"norm.weight",NULL,NULL);
    float *logits=malloc((size_t)S*VOCAB*4); head_fwd_stream(H,S,hcfn,hcb,hcs,norm,logits);
    free(H);free(hcfn);free(hcb);free(hcs);free(norm);
    return logits;
}

/* ===================== 最终输出判决 ===================== *
 * held-out 行 [n_fit,S-1)。主判据(还原率铁律): 分布还原率 Σmin / KL(fp‖q) / PPL ratio;
 * top-1 已退役(teacher-forced 易文本虚高), 仅参考输出。VERDICT 单行供扫描脚本解析。 */
static void verdict(const float*lf,const float*lq,const long*ids,int S,int n_fit,const char*lcfg){
    int s0=n_fit,s1=S-1; if(s1-s0<1){ s0=0; fprintf(stderr,"⚠ held-out<1, 退回全序列\n"); }
    double nllf=0,nllq=0,smin=0,kl=0; int n=0,tf=0,tq=0,ag=0;
    for(int s=s0;s<s1;s++){ long tgt=ids[s+1];
        const float*a=lf+(size_t)s*VOCAB,*b=lq+(size_t)s*VOCAB;
        float ma=a[0],mb=b[0]; int amax=0,bmax=0;
        for(int v=1;v<VOCAB;v++){ if(a[v]>ma){ma=a[v];amax=v;} if(b[v]>mb){mb=b[v];bmax=v;} }
        double sa=0,sb=0; for(int v=0;v<VOCAB;v++){ sa+=exp((double)a[v]-ma); sb+=exp((double)b[v]-mb); }
        double lsa=log(sa),lsb=log(sb),srow=0,krow=0;
        for(int v=0;v<VOCAB;v++){
            double lpf=(double)a[v]-ma-lsa, lpq=(double)b[v]-mb-lsb;
            double pf=exp(lpf), pq=exp(lpq);
            srow += pf<pq?pf:pq;
            if(pf>0) krow += pf*(lpf-lpq);
        }
        smin+=srow; kl+=krow;
        nllf += -((double)a[tgt]-ma-lsa); nllq += -((double)b[tgt]-mb-lsb);
        tf+=(amax==(int)tgt); tq+=(bmax==(int)tgt); ag+=(amax==bmax); n++;
    }
    double pplf=exp(nllf/n),pplq=exp(nllq/n);
    printf("\n=== 最终输出判决 (held-out %d tok · 主判据=分布还原率Σmin/KL/PPL, top1 仅参考) ===\n",n);
    if(NLAYERS<NL) printf("  ⚠ 截断 NL=%d/%d: 绝对 PPL 无效; fp/quant 同深度 → 相对指标(ratio/Σmin/KL)仍可比\n",NLAYERS,NL);
    printf("  fp    : PPL=%12.4f  top1(vs真值)=%.1f%%\n",pplf,100.0*tf/n);
    printf("  quant : PPL=%12.4f  top1(vs真值)=%.1f%%\n",pplq,100.0*tq/n);
    double eg=lcfg_expert_gib(lcfg);
    double zmb=(double)LZ_TOTAL_K*(2.0*DIM+1.0)*2.0/1048576.0;   /* z^L 侧车: U/V/z fp16 */
    printf("  ★ PPL ratio=%.4fx   分布还原率Σmin=%.4f   KL(fp‖q)=%.4f nats   top1一致(vs fp)=%.1f%%\n",
           pplq/pplf,smin/n,kl/n,100.0*ag/n);
    printf("  体积: 专家=%.1f GiB + z^L侧车=%.2f MB (骨干Q8另+8.4)\n",eg,zmb);
    printf("VERDICT lcfg=%s S=%d held=%d pplf=%.4f pplq=%.4f ratio=%.4f smin=%.4f kl=%.4f agree=%.1f top1f=%.1f top1q=%.1f expgib=%.2f zmb=%.2f\n",
           lcfg,S,n,pplf,pplq,pplq/pplf,smin/n,kl/n,100.0*ag/n,100.0*tf/n,100.0*tq/n,eg,zmb);
}

/* ===== 全局联合回扫(★最终输出为判据, 非单层★): 43 层量化落地后, 多轮回扫,
 * 每层 GBL_G[L] 以【最终 logits KL vs FP 锚】判优, 循环到最终质量收敛。
 * 引擎: 全模型 'B' 字节前向(不重量化)+ head; 缓存每层入口 H, 改层 L 只重跑 L..42。 */
static float *gs_forward_from(int L0,const float *Hin,const long*ids,int S,int n_fit,
                              float *Hcache /*可选: 存每层入口 H, [NLAYERS+1][lstride]*/){
    size_t lstride=(size_t)S*HCM*DIM;
    float *H=malloc(lstride*4); memcpy(H,Hin,lstride*4);
    for(int L=L0;L<NLAYERS;L++){
        if(Hcache) memcpy(Hcache+(size_t)L*lstride,H,lstride*4);
        if(GS_LW&&GS_LW[L].loaded){ LW T=lwh_expand(&GS_LW[L]);
            if(getenv("DS4_GS_DIAG")){ fprintf(stderr,"[回扫诊断] L%d cache态 t2ei=%p 缺字段:",L,(void*)T.t2ei);
#define XN(f) if(!T.f) fprintf(stderr," %s",#f);
                LWF_LIST(XN)
#undef XN
                fprintf(stderr," <end>\n"); }
            layer_fwd(L,&T,H,ids,S,n_fit,1,'B',NULL); free_layer(&T); }   /* ★fp16缓存展开★ */
        else { LW W=load_layer(L);
            if(getenv("DS4_GS_DIAG")) fprintf(stderr,"[回扫诊断] L%d 重载态 t2ei=%p gate=%p ids=%p rowmap=%p\n",
                    L,(void*)W.t2ei,(void*)W.gate,(const void*)ids,(void*)g_anc_rowmap);
            layer_fwd(L,&W,H,ids,S,n_fit,1,'B',NULL); free_layer(&W); }   /* 未缓存/被驱逐: 临时重载 */
    }
    if(Hcache) memcpy(Hcache+(size_t)NLAYERS*lstride,H,lstride*4);
    float *lg=malloc((size_t)S*VOCAB*4);
    if(GS_HW){ head_fwd(H,S,GS_HCFN,GS_HCB,GS_HCS,GS_NORM,GS_HW,lg); }   /* ★缓存 head, 禁重读 2.1GB★ */
    else { float *hcfn=st_read_weight(&C,"hc_head_fn",NULL,NULL),*hcb=st_read_weight(&C,"hc_head_base",NULL,NULL),*hcs=st_read_weight(&C,"hc_head_scale",NULL,NULL);
        float *norm=st_read_weight(&C,"norm.weight",NULL,NULL),*hw=st_read_weight(&C,"head.weight",NULL,NULL);
        head_fwd(H,S,hcfn,hcb,hcs,norm,hw,lg);
        free(hcfn);free(hcb);free(hcs);free(norm);free(hw); }
    free(H);
    return lg;
}
/* ★真·反修前层(判据=最终输出 KL, 全局联合最优)★
 * 第 L 层的 z 系数用【最终 logits KL】重解, 不再只往文件尾巴粘标量:
 *   每 token 搜最优 routed 乘子 m_s* → 目标系数 new_c=m_s*·cur_c → 同特征最小二乘重拟合 →
 *   原地改写该层文件 z 载荷(内容真的变, mtime/字节变)。只有最终 val-KL 下降才提交, 否则回退(永不劣化)。
 * 返回 1=已改写该前层文件。 */
/* z 系数最小二乘重拟合(内存): 目标 new_c(x)=ms[s]·cur_c(x), 同特征(GL/GLdyn2/GLdyn8)拟合。
 * Fin=该层 MoE 输入(算特征+当前系数), NF=拟合行数。仅改内存, 落盘由 zfile_commit 做。*/
static void zrefit_w(lop_t*z,const float*ms,int NF,const float*Fin,const float*wrow);
static void zrefit(lop_t*z,const float*ms,int NF,const float*Fin){ zrefit_w(z,ms,NF,Fin,NULL); }
/* 行权版(2026-08-05 用户铁律: 分类/感知行权布线进 ONEPASS): wrow=每行权重(NULL=恒1),
 * 语义与 ALT 阶段2 G0+w*G1 同源 — 残差大的行在最小二乘中话语权大。 */
static void zrefit_w(lop_t*z,const float*ms,int NF,const float*Fin,const float*wrow){
    if(z->type==1){
        double num=0,den=0; for(int s=0;s<NF;s++){ double wr=wrow?wrow[s]:1.0;
            num+=wr*(double)ms[s]*z->g; den+=wr; }
        z->g=(float)(num/(den>0?den:1));
    } else if(z->type==2){
        double A[4]={0,0,0,0},b2[2]={0,0},x2[2];
        for(int s=0;s<NF;s++){ const float*fx=Fin+(size_t)s*DIM; double v=0;
            for(int d=0;d<DIM;d++) v+=(double)fx[d]*fx[d];
            double f=((sqrt(v)-(double)z->w2p[2])/(double)z->w2p[3]);
            double cur=(double)z->w2p[0]+(double)z->w2p[1]*f, tgt=(double)ms[s]*cur;
            double wr=wrow?wrow[s]:1.0;
            A[0]+=wr; A[1]+=wr*f; A[3]+=wr*f*f; b2[0]+=wr*tgt; b2[1]+=wr*f*tgt; }
        /* ★四损失接线(2026-08-05 用户铁律: 架构组件不许连坐砍除)★
         * lfix=ridge 总体倍率(固定损失), lsm=非截距(动态系数)额外光滑惩罚 — 与 ALT KGRID 同语义 */
        A[2]=A[1]; double lam=1e-3*(A[0]+A[3])/2.0*DYN_LFIX+1e-9;
        A[0]+=lam; A[3]+=lam*(DYN_LSM/0.1);   /* df(0.1) 锚定恒等旧解(2026-08-05 bug修: 旧公式 1+lsm*10 把默认组翻倍) */
        if(solve_sym(A,b2,2,x2)==0){ z->w2p[0]=(float)x2[0]; z->w2p[1]=(float)x2[1]; }
    } else if(z->type==3&&z->V8){
        double A[81],b9[9],x9[9]; for(int i=0;i<81;i++)A[i]=0; for(int i=0;i<9;i++)b9[i]=0;
        double phi[9];
        for(int s=0;s<NF;s++){ const float*fx=Fin+(size_t)s*DIM; phi[0]=1;
            for(int c=0;c<8;c++){ double a2=0; const float*vc=z->V8+(size_t)c*DIM;
                for(int d=0;d<DIM;d++) a2+=(double)fx[d]*vc[d]; phi[1+c]=a2; }
            double cur=(double)z->w8[0]; for(int c=0;c<8;c++) cur+=(double)z->w8[1+c]*phi[1+c];
            double tgt=(double)ms[s]*cur;
            double wr=wrow?wrow[s]:1.0;
            for(int r=0;r<9;r++){ b9[r]+=wr*phi[r]*tgt;
                for(int c=0;c<9;c++) A[(size_t)r*9+c]+=wr*phi[r]*phi[c]; } }
        double tr=0; for(int i=0;i<9;i++) tr+=A[(size_t)i*9+i];
        double lam=1e-3*tr/9.0*DYN_LFIX+1e-9;                 /* 固定损失: λ 总体倍率 */
        A[0]+=lam;
        for(int i=1;i<9;i++) A[(size_t)i*9+i]+=lam*(DYN_LSM/0.1);   /* 光滑: df(0.1) 锚定恒等旧解 */
        if(solve_sym(A,b9,9,x9)==0){ for(int i=0;i<9;i++) z->w8[i]=(float)x9[i]; }
    }
}
/* ★首写留档(2026-08-04 流程债)★: 反修 commit 原地覆盖量化段 op 原值 ⇒ 纯量化态从盘上消失,
 * 返修只能重量化(2.6h)。每个 (L,foff) 第一次改写前把原字节 append 进 layers/opbak_LXX.bin
 * (u64 foff|u32 len|bytes) — 恢复=按表回写, 反修从此可无损重来。 */
/* ONEPASS sweep 的进程内 undo 表(终验劣化=全回滚): op_backup 在 BFU_ARM 时旁录旧值 */
typedef struct { int L; size_t foff; uint32_t len; uint8_t*old; } bfu_t;
static bfu_t *BFU=NULL; static int NBFU=0,BFU_CAP=0,BFU_ARM=0;
static void op_backup(int L,int fd,size_t foff,size_t len){
    if(BFU_ARM&&len<=65536){
        if(NBFU>=BFU_CAP){ BFU_CAP=BFU_CAP?BFU_CAP*2:64; BFU=realloc(BFU,(size_t)BFU_CAP*sizeof(bfu_t)); }
        bfu_t*u=&BFU[NBFU]; u->L=L; u->foff=foff; u->len=(uint32_t)len; u->old=malloc(len);
        if(pread(fd,u->old,len,(off_t)foff)==(ssize_t)len) NBFU++; else free(u->old);
    }
    static uint64_t seen[4096]; static int nseen=0;   /* (L<<48|foff) 首写集(单进程规模足够) */
    uint64_t key=((uint64_t)L<<48)|(uint64_t)foff;
    for(int i=0;i<nseen;i++) if(seen[i]==key) return;
    if(nseen<4096) seen[nseen++]=key;
    uint8_t old[8192]; if(len>sizeof(old)) return;
    if(pread(fd,old,len,(off_t)foff)!=(ssize_t)len) return;
    char bp[512]; snprintf(bp,sizeof(bp),"%s/opbak_L%02d.bin",
        getenv("DS4_LAYER_DIR")?getenv("DS4_LAYER_DIR"):".",L);
    FILE*bf=fopen(bp,"ab"); if(!bf) return;
    uint64_t f64=(uint64_t)foff; uint32_t l32=(uint32_t)len;
    fwrite(&f64,8,1,bf); fwrite(&l32,4,1,bf); fwrite(old,1,len,bf); fclose(bf);
}
/* z 载荷原地改写(内容真的变, mtime/字节变) + 记录 mean[0]=新判据
 * ★平行架构: 宿主=op 侧车(在则), 量化 dql 永不被 commit 触碰 */
static void zfile_commit(int L,lop_t*z,float m1metric){
    char lp[512]; op_host_path(L,lp,sizeof(lp));
    int fd=open(lp,O_RDWR); if(fd<0){ perror("zcommit-open"); return; }
    { size_t blen=z->type==1?4:z->type==2?16:z->type==4?4:z->type==5?(size_t)NEXP*2:36;
      op_backup(L,fd,z->foff,blen); op_backup(L,fd,z->foff-20,4); }
    if(z->type==1){ if(pwrite(fd,&z->g,4,(off_t)z->foff)!=4) perror("bf-w1"); }
    else if(z->type==2){ if(pwrite(fd,z->w2p,16,(off_t)z->foff)!=16) perror("bf-w2"); }
    else if(z->type==4){ if(pwrite(fd,&z->t,4,(off_t)z->foff)!=4) perror("bf-t"); }
    else if(z->type==5&&z->ge){ uint16_t*h=malloc((size_t)NEXP*2);
        for(int e=0;e<NEXP;e++) h[e]=go1b_fp32_to_fp16(z->ge[e]);
        if(pwrite(fd,h,(size_t)NEXP*2,(off_t)z->foff)!=(ssize_t)((size_t)NEXP*2)) perror("bf-ge");
        free(h); }
    else { if(pwrite(fd,z->w8,36,(off_t)z->foff)!=36) perror("bf-w8"); }
    if(pwrite(fd,&m1metric,4,(off_t)(z->foff-20))!=4) perror("bf-m1");
    close(fd);
}
static int backfit_layer_z(int L,const long*ids,int S,int n_fit,float*Hc,size_t lstride,double*kl0){
    if(!GS_LF||L>=NLAYERS||!GS_LF[L].map) return 0;
    lfile_t*lf=&GS_LF[L]; int zi=-1;
    for(int i=0;i<lf->nops;i++) if(lf->ops[i].type>=1&&lf->ops[i].type<=3){ zi=i; break; }
    if(zi<0) return 0;                          /* 该层无 z 载荷可重解 */
    lop_t*z=&lf->ops[zi]; lop_t zbak=*z;        /* 备份系数(回退用; V8 指针共享不动) */
    int vs=(n_fit*3)/4; const float*Hin=Hc+(size_t)L*lstride;
    if(!GS_FIN) GS_FIN=malloc((size_t)S*DIM*4);
    /* (a) 基线: 捕获 Fin_L(算 z 特征) + per-token 基线 KL */
    GS_CAP_L=L; GS_GV=NULL; GS_GV_L=-1;
    float*lgb=gs_forward_from(L,Hin,ids,S,n_fit,NULL); GS_CAP_L=-1;
    float*ms=malloc((size_t)S*4); double*kb=malloc((size_t)S*sizeof(double));
    for(int s=0;s<S;s++){ ms[s]=1.0f; kb[s]=bwd_tok_kl(ANC.logits+(size_t)s*VOCAB,lgb+(size_t)s*VOCAB); }
    free(lgb);
    /* (b) per-token 最优乘子: 网格各前向一次(仅目标层 routed 缩放) */
    const float MG[4]={0.8f,0.9f,1.1f,1.25f};
    GS_GV=malloc((size_t)S*4); GS_GV_L=L;
    for(int gi=0;gi<4;gi++){ float m=MG[gi];
        for(int s=0;s<S;s++) GS_GV[s]=m;
        float*lgc=gs_forward_from(L,Hin,ids,S,n_fit,NULL);
        for(int s=0;s<S;s++){ double k=bwd_tok_kl(ANC.logits+(size_t)s*VOCAB,lgc+(size_t)s*VOCAB);
            if(k<kb[s]){ kb[s]=k; ms[s]=m; } }
        free(lgc);
    }
    GS_GV_L=-1; free(GS_GV); GS_GV=NULL; free(kb);
    /* (c) 目标 new_c=m_s*·cur_c, 同特征最小二乘重拟合(fit 行) → 写入内存 z */
    zrefit(z,ms,n_fit,GS_FIN); free(ms);
    /* (d) 验证: 新 z 已在内存, 前向到最终, val-KL 改善才提交 */
    float*lgv=gs_forward_from(L,Hin,ids,S,n_fit,NULL);
    double kln=bwd_val_kl(ANC.logits,lgv,vs,n_fit); free(lgv);
    if(kln<*kl0-1e-9){
        zfile_commit(L,z,(float)kln);          /* ★原地改写文件 z 载荷: 内容真的变★ */
        double imp=100.0*(*kl0-kln)/(*kl0>1e-9?*kl0:1);
        char rs[80]; snprintf(rs,80,"最终KL %.5f→%.5f 降%.1f%%",*kl0,kln,imp);
        mlog(L,"向后·反修",z->type==3?"z.GLdyn8 最终KL重解":z->type==2?"z.GLdyn2 最终KL重解":"z.GL 最终KL重解",
             "已改写文件",z->type==3?36+(uint64_t)8*DIM*2:z->type==2?16:4,rs,"✓正向落地");
        { float*lgx=gs_forward_from(L,Hin,ids,S,n_fit,Hc); free(lgx); }   /* 刷新下游 Hc[L+1..](用提交后 z) */
        *kl0=kln; return 1;
    } else {
        *z=zbak;                               /* 回退内存系数(文件未动) */
        char rs[48]; snprintf(rs,48,"最终KL=%.5f(未降)",*kl0);
        mlog(L,"向后·反修","z 最终KL重解","已最优保持",0,rs,"保持");
        return 0;
    }
}
/* ★联合批式回扫轮(2026-08-04 结构病重构)★
 * 旧逐层坐标下降 = 每层 6-7 次【全量行×全后缀】前向 ⇒ L00≈35min, 一轮 7-10h(实测掐停)。
 * 本轮结构: ①探测=每层 5 次【抽行×单层】前向(基线+4乘子网格), 判据=层出口 vs FP锚 ANC.H[L]
 * 的 per-token L2(局部, 零全程前向; 抽行/紧凑 zrefit 与 sweep 粗筛同先例) → 全部层闭式重解
 * ②终验=1 次【全量×全程】val-KL + β信赖域{1,0.5,0.25}(op_blend 向基线退) ③改善才逐层
 * zfile_commit, 否则全回退 — 判据不妥协(质量门)。一轮 ≈ 探测2-3min + 终验6min×β次。
 * DS4_GS_PERCOL=1 走旧逐层版(A/B 对照口径)。 */
static void op_blend(lop_t*dst,const lop_t*o,const lop_t*f,float b);   /* 定义在下方 β信赖域区 */
static int backfit_joint_round(const long*ids,int S,int n_fit,const float*H0,float*Hc,size_t lstride,double*kl0){
    int vs=(n_fit*3)/4;
    size_t rowsz=(size_t)HCM*DIM;
    int gdiv=getenv("DS4_GS_SCREEN_DIV")?atoi(getenv("DS4_GS_SCREEN_DIV")):12; if(gdiv<1)gdiv=1;
    int Sg=0,*gsx=malloc(sizeof(int)*(size_t)(S/gdiv+2));
    for(int s=0;s<n_fit;s+=gdiv) gsx[Sg++]=s;              /* 只抽 fit 行(zrefit 域) */
    if(!GS_FIN) GS_FIN=malloc((size_t)S*DIM*4);
    float *Hg=malloc((size_t)Sg*rowsz*4);                  /* 抽行紧凑层入口 */
    float *Hb=malloc((size_t)Sg*rowsz*4);                  /* 单层前向工作副本 */
    long  *idg=malloc(sizeof(long)*(size_t)Sg);
    float *msg=malloc((size_t)Sg*4);
    double*kbg=malloc(sizeof(double)*(size_t)Sg);
    float *fing=malloc((size_t)Sg*DIM*4);                  /* 基线捕获的 Fin(紧凑) */
    lop_t *zbaks=calloc((size_t)NLAYERS,sizeof(lop_t));
    lop_t *fits =calloc((size_t)NLAYERS,sizeof(lop_t));
    int   *zidx =malloc(sizeof(int)*(size_t)NLAYERS);
    const float MG[4]={0.8f,0.9f,1.1f,1.25f};
    for(int t=0;t<Sg;t++) idg[t]=ids[gsx[t]];
    for(int L=0;L<NLAYERS;L++){
        zidx[L]=-1;
        lfile_t*lf=&GS_LF[L]; if(!lf->map) continue;
        int zi=-1; for(int i=0;i<lf->nops;i++) if(lf->ops[i].type>=1&&lf->ops[i].type<=3){ zi=i; break; }
        if(zi<0) continue;
        zidx[L]=zi; zbaks[L]=lf->ops[zi];
        for(int t=0;t<Sg;t++) memcpy(Hg+(size_t)t*rowsz,Hc+(size_t)L*lstride+(size_t)gsx[t]*rowsz,rowsz*4);
        g_anc_rowmap=gsx; g_anc_rowstride=S;               /* 锚路由按原行号取 */
        /* 基线单层: 捕获 Fin(紧凑) + per-token 出口 L2 vs FP锚 ANC.H[L] */
        GS_CAP_L=L; GS_GV=NULL; GS_GV_L=-1;
        memcpy(Hb,Hg,(size_t)Sg*rowsz*4);
        { float*Ho=gs_forward_exit(L,L,Hb,idg,Sg,Sg,NULL);
          memcpy(fing,GS_FIN,(size_t)Sg*DIM*4); GS_CAP_L=-1;
          for(int t=0;t<Sg;t++){ double e=0; const float*a=Ho+(size_t)t*rowsz,
              *b=ANC.H+(size_t)L*lstride+(size_t)gsx[t]*rowsz;
              for(size_t i=0;i<rowsz;i++){ double d=(double)a[i]-b[i]; e+=d*d; }
              kbg[t]=e; msg[t]=1.0f; }
          free(Ho); }
        /* 4 乘子网格: 单层前向×抽行, per-token 最优 m */
        GS_GV=malloc((size_t)Sg*4); GS_GV_L=L;
        for(int gi=0;gi<4;gi++){ float m=MG[gi];
            for(int t=0;t<Sg;t++) GS_GV[t]=m;
            memcpy(Hb,Hg,(size_t)Sg*rowsz*4);
            float*Ho=gs_forward_exit(L,L,Hb,idg,Sg,Sg,NULL);
            for(int t=0;t<Sg;t++){ double e=0; const float*a=Ho+(size_t)t*rowsz,
                *b=ANC.H+(size_t)L*lstride+(size_t)gsx[t]*rowsz;
                for(size_t i=0;i<rowsz;i++){ double d=(double)a[i]-b[i]; e+=d*d; }
                if(e<kbg[t]){ kbg[t]=e; msg[t]=m; } }
            free(Ho);
        }
        GS_GV_L=-1; free(GS_GV); GS_GV=NULL;
        zrefit(&lf->ops[zi],msg,Sg,fing);                  /* 闭式重解(紧凑行, sweep 同先例) */
        fits[L]=lf->ops[zi];
        g_anc_rowmap=NULL;
    }
    /* 联合终验(全量×全程) + β信赖域: 过冲则向基线退半步再验 */
    const float BETAS[3]={1.0f,0.5f,0.25f};
    double kln=0; int ok=0;
    for(int bi=0;bi<3;bi++){
        if(bi>0) for(int L=0;L<NLAYERS;L++) if(zidx[L]>=0)
            op_blend(&GS_LF[L].ops[zidx[L]],&zbaks[L],&fits[L],BETAS[bi]);
        float*lg=gs_forward_from(0,H0,ids,S,n_fit,Hc);
        kln=bwd_val_kl(ANC.logits,lg,vs,n_fit); free(lg);
        printf("GSWEEP 联合终验 β=%.2f val-KL=%.5f (基线%.5f)%s\n",BETAS[bi],kln,*kl0,kln<*kl0-1e-9?" ✓":"");
        fflush(stdout);
        if(kln<*kl0-1e-9){ ok=1; break; }
    }
    if(ok){
        int nc=0;
        for(int L=0;L<NLAYERS;L++) if(zidx[L]>=0){ zfile_commit(L,&GS_LF[L].ops[zidx[L]],(float)kln); nc++; }
        char rs[80]; snprintf(rs,80,"联合轮 最终KL %.5f→%.5f 提交%d层",*kl0,kln,nc);
        mlog(0,"向后·反修","z 联合批式重解(抽行探测+全程终验)","已改写层文件",0,rs,"✓正向落地");
        *kl0=kln;
    } else {
        for(int L=0;L<NLAYERS;L++) if(zidx[L]>=0) GS_LF[L].ops[zidx[L]]=zbaks[L];
        { float*lg=gs_forward_from(0,H0,ids,S,n_fit,Hc); free(lg); }   /* 终验刷脏了 Hc → 回退后恢复 */
        mlog(0,"向后·反修","z 联合批式重解","已最优保持(β全拒→回退)",0,"信赖域三级未降","保持");
    }
    free(gsx); free(Hg); free(Hb); free(idg); free(msg); free(kbg); free(fing);
    free(zbaks); free(fits); free(zidx);
    return ok;
}
/* 字节模式前向 L=J..Lend(用 GS_LW 骨干缓存 + GS_LF 层文件缓存), 返回出口 H;
 * 可选把每层入口隐藏写进 HQcache[J..Lend+1](刷新量化态)。逐层反修专用。*/
static float *gs_forward_exit(int J,int Lend,const float*Hin,const long*ids,int S,int n_fit,float*HQcache){
    size_t lstride=(size_t)S*HCM*DIM;
    float *H=malloc(lstride*4); memcpy(H,Hin,lstride*4);
    for(int L=J;L<=Lend;L++){
        if(HQcache) memcpy(HQcache+(size_t)L*lstride,H,lstride*4);   /* 层 L 入口 */
        LW T=GS_LW[L].loaded?lwh_expand(&GS_LW[L]):load_layer(L);   /* fp16 展开; 被驱逐层临时从 HF 重载 */
        layer_fwd(L,&T,H,ids,S,n_fit,1,'B',NULL);
        free_layer(&T);
    }
    if(HQcache) memcpy(HQcache+(size_t)(Lend+1)*lstride,H,lstride*4); /* Lend 出口 */
    return H;
}
/* 每行前沿出口 L2 距离(per-token 反修目标用) */
static void bf_rowdist(const float*H,const float*Htgt,int S,size_t rowsz,double*d){
    for(int s=0;s<S;s++){ double e=0; const float*a=H+(size_t)s*rowsz,*b=Htgt+(size_t)s*rowsz;
        for(size_t i=0;i<rowsz;i++){ double dd=(double)a[i]-b[i]; e+=dd*dd; } d[s]=e; }
}
/* β 信赖域混合: dst 系数 = (1−β)·old + β·fitted(特征参数 μ/σ/V8 不动, 只混系数) */
static void op_blend(lop_t*dst,const lop_t*o,const lop_t*f,float b){
    if(dst->type==1) dst->g=(1.0f-b)*o->g+b*f->g;
    else if(dst->type==2){ dst->w2p[0]=(1.0f-b)*o->w2p[0]+b*f->w2p[0];
                           dst->w2p[1]=(1.0f-b)*o->w2p[1]+b*f->w2p[1]; }
    else if(dst->type==3) for(int i=0;i<9;i++) dst->w8[i]=(1.0f-b)*o->w8[i]+b*f->w8[i];
}
/* 三点抛物线顶点(细搜: 粗网格漏 ±1% 级最优 → 顶点插值补上)。凸才给顶点, 否则回退三点 argmin。*/
static double bf_vertex(double x1,double y1,double x0,double y0,double x2,double y2){
    double s1=(y1-y0)/(x1-x0), s2=(y2-y0)/(x2-x0);
    double a2=(s1-s2)/(x1-x2);                     /* 二次项系数 */
    if(a2>1e-18){
        double num=(x0-x1)*(x0-x1)*(y0-y2)-(x0-x2)*(x0-x2)*(y0-y1);
        double den=(x0-x1)*(y0-y2)-(x0-x2)*(y0-y1);
        if(fabs(den)>1e-18) return x0-0.5*num/den;
    }
    double bx=x0,by=y0; if(y1<by){bx=x1;by=y1;} if(y2<by){bx=x2;by=y2;}
    return bx;
}
/* ★逐层前进即反修前面所有层★(用户设计): 前沿层 Lfront 完成后, 对 J=Lfront-1..0 逐个以
 * 【本前沿层 Lfront 出口 vs FP 锚 ANC.H[Lfront]】为判据, 多形态递进搜正向(铁律: 无提升=形态不对, 换形态):
 *   A 全局α(链末 bf.GL)  B 向后TREF t 重调(前沿判据)  C per-token 动态(argmin α_s → bf.GLdyn2 范数特征)
 * 择优原地改写该层文件 → 刷新下游量化态 HQE → 用反修后累积态继续。判据与落地都走字节回放(与合并同口径)。*/
/* ★形态F(2026-07-13)★ 反修新建 GLdyn8 用的 PCA8 方向: 与层搜索 STK8 同法
 * (确定性种子随机初始化 → 投影/重构交替×4 → MGS 正交), 返回行主序 [c*DIM+d] fp32;
 * 行数不足 3×8 返回 NULL(9-dof 会饥饿)。 */
static float*bf_pca8(const float*Fin,int n){
    const int KP=8;
    if(n<3*KP) return NULL;
    float *Vt=malloc((size_t)DIM*KP*4);
    uint64_t sd=0xC0FFEE123ULL;
    for(size_t i=0;i<(size_t)DIM*KP;i++){ sd=sd*6364136223846793005ULL+1442695040888963407ULL;
        Vt[i]=((float)((sd>>40)&0xFFFFFF)/16777216.0f)-0.5f; }
    dq_mgs_qh(Vt,DIM,KP);
    float *pj=malloc((size_t)n*KP*4);
    for(int it=0;it<4;it++){
        for(int s=0;s<n;s++){ const float*x=Fin+(size_t)s*DIM; float*pr=pj+(size_t)s*KP;
            for(int c=0;c<KP;c++){ double a=0; for(int d=0;d<DIM;d++) a+=(double)x[d]*Vt[(size_t)d*KP+c]; pr[c]=(float)a; } }
        memset(Vt,0,(size_t)DIM*KP*4);
        for(int s=0;s<n;s++){ const float*x=Fin+(size_t)s*DIM; const float*pr=pj+(size_t)s*KP;
            for(int d=0;d<DIM;d++){ float xv=x[d]; if(xv==0)continue; float*vt=Vt+(size_t)d*KP;
                for(int c=0;c<KP;c++) vt[c]+=xv*pr[c]; } }
        dq_mgs_qh(Vt,DIM,KP);
    }
    float*V=malloc((size_t)KP*DIM*4);
    for(int c=0;c<KP;c++) for(int d=0;d<DIM;d++) V[(size_t)c*DIM+d]=Vt[(size_t)d*KP+c];
    free(Vt); free(pj);
    return V;
}
static int backfit_prev(int Lfront,const long*ids,int S,int n_fit){
    if(!GS_LW||!GS_LF||!HQE||Lfront<1) return 0;
    size_t lstride=(size_t)S*HCM*DIM, rowsz=(size_t)HCM*DIM;
    const float*Htgt=ANC.H+(size_t)Lfront*lstride;
    int vs=(n_fit*3)/4, changed=0, evald=0; double simpr=0;
    double bf_bestdl=-1e300; int bf_bestJ=-1;   /* 可视化: 全部候选里最接近正向的Δ(没提交也看得见搜索) */
    char uc[300]; int ucl=0; uc[0]=0;           /* 未正向层清单(层号+最优候选Δ), 铁律: 不许隐身 */
    /* ★用户裁决2026-07-12★: 每前沿全量反修所有前层("末层最后一次全量"特例取消; fast 不进本函数) */
    int Jlo=0;
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
    int BK=getenv("DS4_BF_SCREEN_K")?atoi(getenv("DS4_BF_SCREEN_K")):8;
    int SDIV=getenv("DS4_BF_SCREEN_DIV")?atoi(getenv("DS4_BF_SCREEN_DIV")):4; if(SDIV<1)SDIV=1;
    /* ★2026-08-02: 原来这里有 "DS4_ANCHOR_ROUTE ⇒ BK=0 硬关粗筛"。已删 —— 不相容的根源
     * (锚按原始行号、抽格前向按紧凑行号)由 g_anc_rowmap 行映射解决了。关粗筛的代价是
     * 本文件自己注明的 "旧式全量 ≈0.40·F² min/前沿, 43 层≈7 天", 交付不了。 */
    int Ss=0,vss=0,vrej=0; int*sidx=NULL; long*ids_s=NULL; float*Hin_s=NULL,*Ht_s=NULL;
    if(BK>0){
        sidx=malloc(sizeof(int)*(size_t)(n_fit+1));
        for(int s=0;s<vs;s+=SDIV) sidx[Ss++]=s;
        vss=Ss;
        for(int s=vs;s<n_fit;s+=SDIV) sidx[Ss++]=s;
        if(Ss<8||Ss-vss<2){ free(sidx); sidx=NULL; BK=0; Ss=0; }   /* 校准太小: 粗筛无意义 */
        else{ ids_s=malloc(sizeof(long)*(size_t)Ss);
              for(int s=0;s<Ss;s++) ids_s[s]=ids[sidx[s]];
              Hin_s=malloc((size_t)Ss*rowsz*4); Ht_s=malloc((size_t)Ss*rowsz*4); }
    }
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
    int bis_on=(getenv("DS4_BF_SWEEP_ORDER")&&!strcmp(getenv("DS4_BF_SWEEP_ORDER"),"bisect"));
    double bskip=getenv("DS4_BF_BISECT_SKIP")?atof(getenv("DS4_BF_BISECT_SKIP")):0.05;
    int (*bseg)[2]=bis_on?malloc(sizeof(int[2])*(size_t)(2*Lfront+8)):NULL; int bsp=0;
    /* ★ONEPASS(2026-08-04 用户裁决: "一遍就够, 每层落地反复矫正=后面跑偏")★
     * 冻结基线 Jacobi 式: 单元判据=近视野粗筛(不跑每单元全程复核), 落地不刷 HQE(所有层
     * 在同一反修完成态上选型互不污染), 单 pass, 收尾一次全程终验 — 劣化=全回滚
     * (truncate 掉 append + BFU 表回写原地改 + lfile 重读)。sweep 7.8h → ~1.5h。 */
    int onep=getenv("DS4_BF_ONEPASS")&&atoi(getenv("DS4_BF_ONEPASS"));
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
    if(pass==1&&getenv("DS4_BF_NO_RECHECK")){
        fprintf(stderr,"[反修] DS4_BF_NO_RECHECK=1: 跳过复检遍 → 直接判决\n"); break; }
    /* 访问驱动器: linear=Lfront-1..0 递减(默认, 原行为); bisect(仅首遍)=段栈中点序 */
    int bis=(bis_on&&pass==0);
    int blo=Jlo,bhi=Lfront-1;
    if(bis){ bsp=0; bseg[0][0]=Jlo; bseg[0][1]=Lfront-1; bsp=1; }
    int J=Lfront;
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
        time_t ut0=time(NULL);   /* 单元计时(BFUNIT 可观测行) */
        lfile_t*lf=&GS_LF[J];
        char pj[512]; op_host_path(J,pj,sizeof(pj));   /* ★平行架构: 反修落地只写 op 侧车, 量化 dql 不可触 */
        if(onep&&!ofl[J]){ struct stat fst; ofl[J]=stat(pj,&fst)==0?(size_t)fst.st_size:0; }   /* 回滚锚点 */
        const float*Hin=HQE+(size_t)J*lstride;
        /* 粗筛视野: e* 别名 = 本单元候选比价用的(出口层/输入/ids/行数/切分); scr=0 时即旧全量口径 */
        int Fj=(BK>0&&J+BK<Lfront)?J+BK:Lfront;
        int scr=(BK>0)&&(Fj<Lfront||Ss<S);
        int eF=scr?Fj:Lfront, eS=scr?Ss:S, eNf=scr?Ss:n_fit, eVs=scr?vss:vs;
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
        double base=co_score(Hb,eTgt,eVs,eNf,rowsz);
        bf_rowdist(Hb,eTgt,eS,rowsz,db); free(Hb);
        /* 行权系数 kk_s(2026-08-05 分类/感知布线): 行残差能量/行目标能量, clamp[0,4] */
        for(int s=0;s<eS;s++){ const float*b=eTgt+(size_t)s*rowsz; double tn=0;
            for(size_t i2=0;i2<rowsz;i2++) tn+=(double)b[i2]*b[i2];
            double kv2=db[s]/(tn+1e-20); if(kv2>4.0)kv2=4.0; kkrow[s]=(float)kv2; }
        /* ★联合损失(2026-08-05 用户终裁: 四损失+感知同一目标并存, 非网格选一)★
         * 行权恒开: effw=1+W_pc·kk(W_pc 全局常数, →0 逐位退化基线解) */
        { static float JWPC=-1.0f;
          if(JWPC<0){ const char*e=getenv("DS4_JOINT_WPC"); JWPC=e?(float)atof(e):0.35f; }
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
            double sc=co_score(Hg,eTgt,eVs,eNf,rowsz); scg[gi]=sc;
            bf_rowdist(Hg,eTgt,eS,rowsz,dg+(size_t)gi*S); free(Hg);
            if(sc<scA){ scA=sc; aA=a; }
            if(undo) lf->nops--; else lf->ops[tmp].g=baseg;
        }
        { double av=bf_vertex(AG[0],scg[0],1.0,base,AG[NA-1],scg[NA-1]);
          if(av<0.80)av=0.80; if(av>1.20)av=1.20;
          if(fabs(av-1.0)>2e-3&&fabs(av-(double)aA)>2e-3){   /* 顶点≠已测点才补一枪 */
              int tmp=fo,undo=0;
              if(tmp>=0) lf->ops[tmp].g=baseg*(float)av;
              else if(lf->nops<32){ tmp=lf->nops; memset(&lf->ops[tmp],0,sizeof(lop_t));
                  lf->ops[tmp].type=1; lf->ops[tmp].g=(float)av; lf->nops++; undo=1; }
              if(tmp>=0){
                  float*Hg=gs_forward_exit(J,eF,eHin,eIds,eS,eNf,NULL);
                  double sc=co_score(Hg,eTgt,eVs,eNf,rowsz); free(Hg);
                  if(sc<scA){ scA=sc; aA=(float)av; }
                  if(undo) lf->nops--; else lf->ops[tmp].g=baseg;
              } } }
        /* --- 形态B: 向后 TREF t 重调(层内落地的 type-4 op, 判据升级为前沿出口) + 顶点细搜 --- */
        int to=-1; for(int i=lf->nops-1;i>=0;i--) if(lf->ops[i].type==4){ to=i; break; }
        double scB=base; float tB=0;
        if(to>=0){ float t0v=lf->ops[to].t; const float TG[2]={0.94f,1.06f}; double scT[2];
            for(int ti=0;ti<2;ti++){ lf->ops[to].t=t0v*TG[ti];
                float*Hg=gs_forward_exit(J,eF,eHin,eIds,eS,eNf,NULL);
                double sc=co_score(Hg,eTgt,eVs,eNf,rowsz); free(Hg); scT[ti]=sc;
                if(sc<scB){ scB=sc; tB=t0v*TG[ti]; }
                lf->ops[to].t=t0v; }
            double rv=bf_vertex(TG[0],scT[0],1.0,base,TG[1],scT[1]);
            if(rv<0.80)rv=0.80; if(rv>1.20)rv=1.20;
            if(fabs(rv-1.0)>2e-3&&(tB==0||fabs(rv-(double)(tB/t0v))>2e-3)){
                lf->ops[to].t=t0v*(float)rv;
                float*Hg=gs_forward_exit(J,eF,eHin,eIds,eS,eNf,NULL);
                double sc=co_score(Hg,eTgt,eVs,eNf,rowsz); free(Hg);
                if(sc<scB){ scB=sc; tB=t0v*(float)rv; }
                lf->ops[to].t=t0v; } }
        /* per-token 连续目标 α*_s(抛物线顶点; 全部重解形态共用; 行域=粗筛行) */
        for(int s=0;s<eS;s++){
            double mv=bf_vertex(AG[0],dg[s],1.0,db[s],AG[NA-1],dg[(size_t)(NA-1)*S+s]);
            if(mv<0.85)mv=0.85; if(mv>1.15)mv=1.15; ms[s]=(float)mv;   /* clamp=探针包络(随探针加宽) */
        }
        int nmix=0;
        for(int s=0;s<eVs;s++) if(fabsf(ms[s]-1.0f)>2e-3f) nmix++;
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
                zrefit_w(&fit,ms,eVs,GS_FIN,effw);
                for(int bi=0;bi<3;bi++){
                    op_blend(&lf->ops[ze],&old,&fit,BB[bi]);
                    float*Hg=gs_forward_exit(J,eF,eHin,eIds,eS,eNf,NULL);
                    double sc=co_score(Hg,eTgt,eVs,eNf,rowsz); free(Hg);
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
            if(!hasd8&&nmix>=3&&lf->nops<32&&(V8F=bf_pca8(GS_FIN,eVs))!=NULL){
                /* 联合损失单次求解(同形态E) */
                opF.type=3; opF.V8=V8F; opF.w8[0]=1.0f;      /* 起点=恒等, zrefit 9-dof 岭回归拟 per-token 靶 */
                zrefit_w(&opF,ms,eVs,GS_FIN,effw);
                int tmp=lf->nops; lf->ops[tmp]=opF; lf->nops++;
                float*Hg=gs_forward_exit(J,eF,eHin,eIds,eS,eNf,NULL);
                scF=co_score(Hg,eTgt,eVs,eNf,rowsz); free(Hg);
                lf->nops--;
            }
        }
        /* --- 形态C: per-token 动态 → bf.GLdyn2(范数特征拟合) --- */
        double scC=base; lop_t opC; memset(&opC,0,sizeof(opC));
        if(nmix>=2&&lf->nops<32){
            double mu=0,sd=0; float*fn=malloc((size_t)S*4);
            for(int s=0;s<eS;s++){ const float*fx=GS_FIN+(size_t)s*DIM; double v=0;
                for(int d=0;d<DIM;d++) v+=(double)fx[d]*fx[d]; fn[s]=(float)sqrt(v); }
            for(int s=0;s<eVs;s++) mu+=fn[s]; mu/=(eVs>0?eVs:1);
            for(int s=0;s<eVs;s++){ double d=fn[s]-mu; sd+=d*d; } sd=sqrt(sd/(eVs>0?eVs:1))+1e-9;
            /* 联合损失单次求解(dyn2 新建: 行权恒开+λ df 锚定) */
            { double A[4]={0,0,0,0},b2[2]={0,0},x2[2];
              for(int s=0;s<eVs;s++){ double f=((double)fn[s]-mu)/sd, wr=(double)effw[s];
                  A[0]+=wr; A[1]+=wr*f; A[3]+=wr*f*f; b2[0]+=wr*ms[s]; b2[1]+=wr*f*ms[s]; }
              A[2]=A[1];
              double lam=1e-3*(A[0]+A[3])/2.0+1e-9;
              A[0]+=lam; A[3]+=lam;
              if(solve_sym(A,b2,2,x2)==0){
                opC.type=2; opC.w2p[0]=(float)x2[0]; opC.w2p[1]=(float)x2[1];
                opC.w2p[2]=(float)mu; opC.w2p[3]=(float)sd;
                int tmp=lf->nops; lf->ops[tmp]=opC; lf->nops++;
                float*Hg=gs_forward_exit(J,eF,eHin,eIds,eS,eNf,NULL);
                scC=co_score(Hg,eTgt,eVs,eNf,rowsz); free(Hg);
                lf->nops--;
              } }
            free(fn);
        }
        /* --- 形态D: per-expert 增益投影(256-dof; per-token 目标 m*_s 按路由权重投到专家) --- */
        double scD=base; float *geD=NULL;
        if(nmix>=1&&lf->nops<32){
            geD=malloc((size_t)NEXP*4);
            double *gnum=calloc((size_t)NEXP,sizeof(double)),*gden=calloc((size_t)NEXP,sizeof(double));
            for(int s=0;s<eVs;s++) for(int a2=0;a2<NACT;a2++){
                int e=GS_IDXC[(size_t)s*NACT+a2]; double w=GS_RWC[(size_t)s*NACT+a2];
                if(e>=0&&e<NEXP&&w>0){ gnum[e]+=w*ms[s]; gden[e]+=w; } }
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
                scD=co_score(Hg,eTgt,eVs,eNf,rowsz); free(Hg);
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
        double jdl;   /* 本层最优候选Δ%(负=候选都更差) */
        { double cand=scE; if(scF<cand)cand=scF; if(scA<cand)cand=scA; if(scB<cand)cand=scB; if(scC<cand)cand=scC; if(scD<cand)cand=scD;
          double dl=(base-cand)/(base>1e-12?base:1); jdl=100.0*dl;
          if(dl>bf_bestdl){ bf_bestdl=dl; bf_bestJ=J; } }
        if(form&&scr&&!onep){
            /* ★全闸复核★: 粗筛只定排序; 胜者内存试装 → 真前沿×全token 重打, 必须净降才放行
             * (落地判据与旧全量逐字节同口径; 未过=还原+保持, 复检 pass 仍可再试)
             * ONEPASS: 跳过 — 逐单元全程复核被收尾统一终验+全回滚取代 */
            g_anc_rowmap = NULL;   /* 复核走全量行 ⇒ 锚回恒等映射 */
            float*Hf0=gs_forward_exit(J,Lfront,Hin,ids,S,n_fit,NULL);
            double basef=co_score(Hf0,Htgt,vs,n_fit,rowsz); free(Hf0);
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
            float*Hv=gs_forward_exit(J,Lfront,Hin,ids,S,n_fit,NULL);
            double scv=co_score(Hv,Htgt,vs,n_fit,rowsz); free(Hv);
            if(form==5) lf->ops[ze]=svE;
            else if(form==6) lf->nops--;
            else if(form==1){ if(tmpop>=0) lf->nops--; else lf->ops[fo].g=svg; }
            else if(form==2) lf->ops[to].t=svt;
            else if(form==3){ if(tmpop>=0) lf->nops--; else memcpy(lf->ops[BF_DYN2OP[J]].w2p,svw2,16); }
            else { if(tmpop>=0) lf->nops--; else memcpy(lf->ops[BF_GEOP[J]].ge,gbak,(size_t)NEXP*4); }
            if(gbak){ free(gbak); gbak=NULL; }
            if(scv<basef-1e-9){ base=basef; bestsc=scv; }   /* 放行: 日志/账目切换到全口径分 */
            else { vrej++; form=0; }                        /* 全闸拒: 不落地(计入汇总) */
        }
        if(form){
            char al[64],rs[80]; uint64_t vol=4;
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
                snprintf(rs,80,"前沿L%d出口分 %.5g→%.5g 降%.2f%%%s",Lfront,base,bestsc,
                         100.0*(base-bestsc)/(base>1e-12?base:1),pass?"(复检)":"");
                mlog(J,"向前反修",al,"已改写层文件",vol,rs,"✓正向落地");
            }
        }
        if(geD){ free(geD); geD=NULL; }
        if(V8F){ free(V8F); V8F=NULL; }     /* F 候选没落地(落地时所有权已交 op) */
        if(!form) jd[J]=jdl;                /* 未正向: 记录候选Δ, 收尾统一上日志(复检后) */
        printf("BFUNIT L=%02d Δbest=%+.3f%% %s 用时=%lds\n",J,jdl,form?"落地":"保持",(long)(time(NULL)-ut0));
        fflush(stdout);                     /* 实时可观测(用户裁决: 未落地层不许等到轮末才现身) */
        if(bis){
            /* ★一点采样不判整段生死(单测实锤: mid 平坦剪长段 ⇒ 深尾大增益整段漏检)★
             * 段长>DS4_BF_BISECT_MAXSEG(默认3)时 mid 平坦也必须细分; 只许剪 ≤MAXSEG 的小段
             * ⇒ 漏检窗口≤相邻2层, 增益孤岛必被四分位链命中。 */
            int seglen=bhi-blo+1;
            int mseg=getenv("DS4_BF_BISECT_MAXSEG")?atoi(getenv("DS4_BF_BISECT_MAXSEG")):3;
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
            double b0=co_score(HQE+(size_t)(Lfront+1)*lstride,Htgt,vs,n_fit,rowsz);
            float*Hf=gs_forward_exit(Jlo,Lfront,HQE+(size_t)Jlo*lstride,ids,S,n_fit,NULL);
            double fin=co_score(Hf,Htgt,vs,n_fit,rowsz); free(Hf);
            printf("BF_ONEPASS 终验 落地=%d 出口分 %.5g→%.5g %s\n",nland,b0,fin,
                   fin<b0-1e-9?"✓改善→提交":"✗劣化→全回滚"); fflush(stdout);
            if(fin<b0-1e-9){
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
/* ===== ②冷专家修复(DS4_REPAIR_COLD=1, 2026-07-24 按序执行第2项) =====
 * 判据: 旧锚(DS4_ANCHOR_OLD)fit行零命中 ∧ 新锚(DS4_ANCHOR, 当前语料)fit行≥1 的冷专家 —
 * 这些专家原字节=无数据rowscale兜底(纯权重先验)=胡言触发面; 其对层内op链的拟合贡献≈0(零流量)
 * → 用新锚校准行重解 signref(含 w2 顺序补偿) 原地 pwrite 回 dql, 风险最小收益直接。热=go2b侧车不动。 */
typedef struct { int L,S,n_fit,S_old,n_fit_old; int *e_next; int fd; size_t off0,szG,szD;
                 const int32_t*aold; const float*afin; const int32_t*aidx; long nrep; } rcw_t;
static void *repair_worker(void*a){
    rcw_t*w=a;
    float *Xc=malloc((size_t)(w->n_fit>0?w->n_fit:1)*DIM*4);
    for(;;){ int e=__sync_fetch_and_add(w->e_next,1); if(e>=NEXP)break;
        if(g2_hot_slot(w->L,e)>=0) continue;                 /* 热=go2b 侧车, dql 洞不动 */
        int hit_old=0;
        for(int s=0;s<w->n_fit_old&&!hit_old;s++)for(int a2=0;a2<NACT;a2++)
            if(w->aold[(size_t)s*NACT+a2]==e){ hit_old=1; break; }
        if(hit_old) continue;                                /* 旧锚已有校准 → 字节不动 */
        int ncal=0;
        for(int s=0;s<w->n_fit;s++){ int hit=0;
            for(int a2=0;a2<NACT;a2++) if(w->aidx[(size_t)s*NACT+a2]==e){hit=1;break;}
            if(hit){ memcpy(Xc+(size_t)ncal*DIM,w->afin+(size_t)s*DIM,(size_t)DIM*4); ncal++; } }
        if(!ncal) continue;                                  /* 新语料也没打中 → 仍无据可修 */
        char n1[160],n3[160],n2[160]; long rr,cc;
        snprintf(n1,sizeof(n1),"layers.%d.ffn.experts.%d.w1.weight",w->L,e);
        snprintf(n3,sizeof(n3),"layers.%d.ffn.experts.%d.w3.weight",w->L,e);
        snprintf(n2,sizeof(n2),"layers.%d.ffn.experts.%d.w2.weight",w->L,e);
        float *e1=st_read_weight(&C,n1,&rr,&cc),*e3=st_read_weight(&C,n3,&rr,&cc),*e2=st_read_weight(&C,n2,&rr,&cc);
        if(!e1||!e3||!e2){ if(e1)free(e1);if(e3)free(e3);if(e2)free(e2); continue; }
        uint8_t *dG=malloc(w->szG),*dU=malloc(w->szG),*dDn=malloc(w->szD);
        float *q1=malloc((size_t)MOEI*DIM*4),*q3=malloc((size_t)MOEI*DIM*4);
        dq_signref_export(e1,MOEI,DIM,Xc,ncal,dG,q1);
        dq_signref_export(e3,MOEI,DIM,Xc,ncal,dU,q3);
        float *hc=malloc((size_t)ncal*MOEI*4);
        { float *gg=malloc((size_t)ncal*MOEI*4),*uu=malloc((size_t)ncal*MOEI*4);
          dq_matmul(Xc,q1,gg,ncal,DIM,MOEI); dq_matmul(Xc,q3,uu,ncal,DIM,MOEI);
          for(size_t i=0;i<(size_t)ncal*MOEI;i++){ float g2v=gg[i],u2=uu[i];
              if(SWLIM>0){ if(u2>SWLIM)u2=SWLIM; if(u2<-SWLIM)u2=-SWLIM; if(g2v>SWLIM)g2v=SWLIM; }
              hc[i]=dq_silu(g2v)*u2; }
          free(gg);free(uu); }
        dq_signref_export(e2,DIM,MOEI,hc,ncal,dDn,NULL);
        if(pwrite(w->fd,dG,w->szG,(off_t)(w->off0+(size_t)e*w->szG))!=(ssize_t)w->szG) perror("rep-g");
        if(pwrite(w->fd,dU,w->szG,(off_t)(w->off0+(size_t)NEXP*w->szG+(size_t)e*w->szG))!=(ssize_t)w->szG) perror("rep-u");
        if(pwrite(w->fd,dDn,w->szD,(off_t)(w->off0+2*(size_t)NEXP*w->szG+(size_t)e*w->szD))!=(ssize_t)w->szD) perror("rep-d");
        __sync_fetch_and_add(&w->nrep,1);
        free(dG);free(dU);free(dDn);free(q1);free(q3);free(hc);free(e1);free(e3);free(e2);
    }
    free(Xc); return NULL;
}
static void repair_cold(int S,int n_fit){
    const char*oldp=getenv("DS4_ANCHOR_OLD"); const char*ld=getenv("DS4_LAYER_DIR");
    if(!oldp||!ld){ fprintf(stderr,"[repair] 需 DS4_ANCHOR_OLD + DS4_LAYER_DIR\n"); exit(2); }
    FILE*fo=fopen(oldp,"rb"); if(!fo){ perror(oldp); exit(2); }
    uint32_t hd[8]; uint64_t idh0;
    if(fread(hd,4,8,fo)!=8||fread(&idh0,8,1,fo)!=1){ fprintf(stderr,"[repair] 旧锚读失败\n"); exit(2); }
    if(hd[0]!=0x32415144u||hd[4]!=(uint32_t)NLAYERS||hd[6]!=(uint32_t)NACT||hd[3]!=(uint32_t)DIM){
        fprintf(stderr,"[repair] 旧锚头不符(NL/NACT/DIM)\n"); exit(2); }
    int S_old=(int)hd[1], n_fit_old=(S_old*3)/4;
    fseeko(fo,(off_t)(40+(uint64_t)NLAYERS*(uint64_t)S_old*DIM*4),SEEK_SET);
    int32_t*ridx_old=malloc((size_t)NLAYERS*S_old*NACT*4);
    if(fread(ridx_old,4,(size_t)NLAYERS*S_old*NACT,fo)!=(size_t)NLAYERS*S_old*NACT){
        fprintf(stderr,"[repair] 旧锚ridx读失败\n"); exit(2); }
    fclose(fo);
    size_t szG=(size_t)MOEI*go1b_blk_row_bytes(DIM), szD=(size_t)DIM*go1b_blk_row_bytes(MOEI);
    fprintf(stderr,"[repair] 冷专家修复: 旧S=%d(fit%d) 新S=%d(fit%d) signref μ=%.0f\n",
            S_old,n_fit_old,S,n_fit,dq_signref_mu);
    long tot=0;
    for(int L=0;L<NLAYERS;L++){
        char lp[512]; snprintf(lp,sizeof(lp),"%s/dql_L%02d.bin",ld,L);
        lfile_t lf;
        if(lfile_load(lp,&lf)!=0){ fprintf(stderr,"[repair] L%02d 层文件缺/坏 — 停\n",L); exit(3); }
        size_t off0=(size_t)(lf.w1-lf.map);
        lfile_free(&lf);
        int fd=open(lp,O_WRONLY); if(fd<0){ perror(lp); exit(3); }
        int e_next=0,nth=NTHREADS<1?1:(NTHREADS>NEXP?NEXP:NTHREADS);
        rcw_t*ws=calloc((size_t)nth,sizeof(rcw_t)); pthread_t*th=malloc((size_t)nth*sizeof(pthread_t));
        for(int t=0;t<nth;t++){ ws[t]=(rcw_t){L,S,n_fit,S_old,n_fit_old,&e_next,fd,off0,szG,szD,
            ridx_old+(size_t)L*S_old*NACT, ANC.fin+(size_t)L*S*DIM, ANC.ridx+(size_t)L*S*NACT, 0};
            pthread_create(&th[t],NULL,repair_worker,&ws[t]); }
        long nrep=0;
        for(int t=0;t<nth;t++){ pthread_join(th[t],NULL); nrep+=ws[t].nrep; }
        free(ws);free(th); close(fd);
        printf("REPAIR L=%02d 修复=%ld\n",L,nrep); fflush(stdout);
        fprintf(stderr,"[repair][mem] L%02d footprint=%.2fGB\n",L,mem_gb());
        tot+=nrep;
    }
    free(ridx_old);
    printf("REPAIR_COLD 总修复=%ld 专家 (判据: 旧锚0校准∧新锚≥1; 热go2b/已校准不动)\n",tot);
    fflush(stdout);
}
static void global_sweep(const long*ids,int S,int n_fit){
    int MAXS=getenv("DS4_GSWEEP")?atoi(getenv("DS4_GSWEEP")):3;
    if(MAXS<1) return;
    size_t lstride=(size_t)S*HCM*DIM;
    for(int L=0;L<NLAYERS;L++) GBL_G[L]=1.0f;
    /* ★效率: 存档只加载一次★ — 释放回扫不用的大 anchor(fin/H/ridx/rw, 保 logits) 腾内存,
     * 再把 43 层骨干权重 + 层文件 mmap 各缓存一次(禁千次重读/重mmap/重反量化) */
    /* ★ridx/rw 必须保活(2026-08-03 回扫 SIGSEGV 终修): 全局回扫的 B 前向仍走锚路由覆盖
     * (layer_fwd 2218 读 ANC.ridx/rw), 二者合计 ~5MB(mmap=干净页零成本, malloc=小)。
     * 只释放大块 fin/H(GB 级)。
     * ★joint 模式(默认)必须保 ANC.H(2026-08-04 probe1 段错误实锤): backfit_joint_round 的
     * 单层探测判据=层出口 vs ANC.H[L]; mmap 锚=干净页可逐出零 RSS 代价 → 不释放。
     * 旧逐层版(DS4_GS_PERCOL=1)只用 ANC.logits, 保持原释放。 */
    { int percol0=getenv("DS4_GS_PERCOL")?atoi(getenv("DS4_GS_PERCOL")):0;
      if(percol0){
          if(ANC_MMAP){ ANC.fin=NULL; ANC.H=NULL; }   /* mmap 页可逐出, 禁 free */
          if(ANC.fin){ free(ANC.fin); ANC.fin=NULL; }
          if(ANC.H){ free(ANC.H); ANC.H=NULL; }
      } else if(!ANC_MMAP&&ANC.fin){ free(ANC.fin); ANC.fin=NULL; }   /* joint: 只放 fin(不用), H 必留 */
    }
    if(!GS_LW){   /* fwd_all 逐层反修未开(DS4_BACKFIT_INCR=0)时才现建 */
        fprintf(stderr,"[全局回扫] 缓存 43 层权重+文件(只一次)...\n");
        GS_LW=calloc((size_t)NLAYERS,sizeof(LWH));
        GS_LF=calloc((size_t)NLAYERS,sizeof(lfile_t));
        for(int L=0;L<NLAYERS;L++){
            { LW W=load_layer2(L,&GS_LW[L]); lwh_absorb(&GS_LW[L],&W); free_layer(&W); }
            char lp[512]; snprintf(lp,sizeof(lp),"%s/dql_L%02d.bin",getenv("DS4_LAYER_DIR"),L);
            if(lfile_load(lp,&GS_LF[L])!=0) fprintf(stderr,"[回扫] L%02d 层文件缓存失败\n",L);
            if((L&7)==0) fprintf(stderr,"  缓存 %d/%d\n",L,NLAYERS);
        }
    } else fprintf(stderr,"[全局回扫] 复用 fwd_all 逐层反修已建缓存(权重/文件在位, 免重读)\n");
    GS_HCFN=st_read_weight(&C,"hc_head_fn",NULL,NULL); GS_HCB=st_read_weight(&C,"hc_head_base",NULL,NULL);
    GS_HCS=st_read_weight(&C,"hc_head_scale",NULL,NULL); GS_NORM=st_read_weight(&C,"norm.weight",NULL,NULL);
    GS_HW=st_read_weight(&C,"head.weight",NULL,NULL);
    fprintf(stderr,"[全局回扫] 权重/文件/head 缓存完成 — 之后只算不读盘\n");
    float *emb=st_read_weight(&C,"embed.weight",NULL,NULL);
    float *H0=malloc(lstride*4);
    for(int s=0;s<S;s++)for(int j=0;j<HCM;j++)memcpy(H0+((size_t)s*HCM+j)*DIM,emb+(size_t)ids[s]*DIM,(size_t)DIM*4);
    free(emb);
    float *Hc=malloc((size_t)(NLAYERS+1)*lstride*4);
    int vs=(n_fit*3)/4;   /* 判据行 = fit 尾部 val, held 只观测 */
    fprintf(stderr,"\n[全局回扫] %d 轮 — 逐前层 z 系数以【最终输出 KL】重解并原地改写层文件(真·反修, 全局联合最优)\n",MAXS);
    float *lg=gs_forward_from(0,H0,ids,S,n_fit,Hc);
    double kl0=bwd_val_kl(ANC.logits,lg,vs,n_fit); free(lg);
    double kl_start=kl0;
    printf("GSWEEP 起点 最终val-KL=%.5f\n",kl0); fflush(stdout);
    /* ★轮间收敛判停(2026-08-04 用户裁决: 遍数可压)★: 布尔 improved 只挡"整轮零改写",
     * 数值尾巴轮(几层各降 1e-5)仍会把 MAXS 轮跑满 → 按轮改善量占起点比例判停,
     * 低于 DS4_GS_CONV_PCT(默认 0.1%)即数值收敛, 后续轮不跑。 */
    double conv_pct=getenv("DS4_GS_CONV_PCT")?atof(getenv("DS4_GS_CONV_PCT")):0.1;
    int percol=getenv("DS4_GS_PERCOL")?atoi(getenv("DS4_GS_PERCOL")):0;   /* 1=旧逐层坐标下降(20-30h, 对照口径) */
    for(int sw=1;sw<=MAXS;sw++){
        int improved=0; double kl_round0=kl0;
        if(!percol){
            fprintf(stderr,"[反修] 第%d轮 联合批式(抽行探测+全程终验)...\n",sw);
            improved=backfit_joint_round(ids,S,n_fit,H0,Hc,lstride,&kl0);
        } else for(int L=0;L<NLAYERS;L++){
            double before=kl0;
            fprintf(stderr,"[反修] 第%d轮 L%02d z重解(最终KL判优)... ",sw,L);
            int chg=backfit_layer_z(L,ids,S,n_fit,Hc,lstride,&kl0);
            if(chg){ improved=1;
                printf("GSWEEP sw%d L=%d z重解 最终val-KL %.5f→%.5f ✓改写层文件\n",sw,L,before,kl0); }
            else    printf("GSWEEP sw%d L=%d z保持(未降/无z) 最终val-KL=%.5f\n",sw,L,kl0);
            fflush(stdout);
        }
        double rimp=kl_start>1e-9?100.0*(kl_round0-kl0)/kl_start:0.0;
        printf("GSWEEP 第%d轮完 最终val-KL=%.5f 本轮改善=%.4f%% %s\n",sw,kl0,rimp,improved?"有改善":"收敛");
        fflush(stdout);
        if(!improved) break;
        if(rimp<conv_pct){ printf("GSWEEP 第%d轮改善%.4f%%<%.2f%%(DS4_GS_CONV_PCT) → 数值收敛提前止\n",sw,rimp,conv_pct);
            fflush(stdout); break; }
    }
    printf("GSWEEP_DONE 最终val-KL %.5f→%.5f 提升%.1f%%\n",
           kl_start,kl0,kl_start>1e-9?100.0*(kl_start-kl0)/kl_start:0.0);
    fflush(stdout);
    free(H0);free(Hc);
    for(int L=0;L<NLAYERS;L++){ lwh_free(&GS_LW[L]); lfile_free(&GS_LF[L]); }
    free(GS_LW); GS_LW=NULL; free(GS_LF); GS_LF=NULL;
    free(GS_HCFN);free(GS_HCB);free(GS_HCS);free(GS_NORM);free(GS_HW);
    GS_HCFN=GS_HCB=GS_HCS=GS_NORM=GS_HW=NULL;
    if(GS_FIN){ free(GS_FIN); GS_FIN=NULL; }
}
int main(int argc,char**argv){
    /* ★进程级分配器/BLAS 设置写进代码, 不走 env(2026-08-22 铁律: 本项目不得新增环境变量)★
     * 依据(实测, 全过程记在 fable5.md):
     *  · 专家循环每专家 malloc/free 三块 33.5MB。>128KB 默认走 mmap ⇒ 每次全新零页 + 逐 4KB 首触缺页;
     *    而 TRIM/MMAP 阈只管主 arena, **非主 arena 的堆增缩仍走 mprotect** —— 20 线程抢进程 mmap
     *    写锁, wchan 实锤 51/110 卡在 vm_mmap_pgoff/do_mprotect_pkey/__vm_munmap。
     *    放开 top_pad 后 mmap 风暴消失: S=256 从 164s→65s, 磁盘 118 MB/s→1939 MB/s。
     *  · attention 在专家 pthread 循环之外, 是单线程段; BLAS 只给 1 线程会把它锁死在 1 核。
     *    专家循环内部已有 20 路 pthread, 那段靠 GPU/单线程 BLAS, 不受此影响。 */
#ifdef __linux__
    mallopt(M_TOP_PAD,         256 * 1024 * 1024);
    mallopt(M_MMAP_THRESHOLD, 1024 * 1024 * 1024);
    mallopt(M_TRIM_THRESHOLD, 1024 * 1024 * 1024);
#endif
#ifdef DS4QUANT_OPENBLAS
    { void scipy_openblas_set_num_threads(int); long nc = sysconf(_SC_NPROCESSORS_ONLN);
      scipy_openblas_set_num_threads((int)(nc > 1 ? nc : 1)); }
#endif
    for(int i=0;i<NL;i++) GBL_G[i]=1.0f;   /* ★static 默认0=乘0清routed: 反修中途 'B' 回放被它抹平(实锤bug), 必须先置1★ */
    nact_rt_init(getenv("DS4_LAYER_DIR"));   /* 动态路由反修 Phase-B: 层目录 rroute.txt 门控 */
    const char*hf=getenv("DS4_HF");if(!hf)hf="/Users/fodelf/ds4-main/hf/DeepSeek-V4-Flash-Base"; st_open(&C,hf);
    const char*idf=argc>1?argv[1]:"/tmp/rr_hard.ids"; int ntok=argc>2?atoi(argv[2]):64;
    if(getenv("DS4_Z_RANK")) ZRANK=atoi(getenv("DS4_Z_RANK"));
    if(getenv("DS4_Z_LAMBDA")) ZLAMBDA=atof(getenv("DS4_Z_LAMBDA"));
    if(getenv("DS4_NL")){ NLAYERS=atoi(getenv("DS4_NL")); if(NLAYERS<1)NLAYERS=1; if(NLAYERS>NL)NLAYERS=NL; }
    if(getenv("DS4_ZCHAIN_ONLY")){
        /* 独立模式: 从既有 dql 层文件重建 zchain_all.bin + 43 份 opt_LXX.bin, 不跑量化。
         * merge-only 恢复场景专用(量化被截停时 zchain_write 没到点) — 修"merge 出的
         * GGUF 缺优化张量"缺口: 脚本在建骨架前先跑本模式补齐侧车。 */
        if(!getenv("DS4_LAYER_DIR")||!getenv("DS4_ZCHAIN")){
            fprintf(stderr,"[zchain-only] 需 DS4_LAYER_DIR + DS4_ZCHAIN\n"); return 1; }
        zchain_write();
        return 0;
    }
    /* ★merge 独立早出口(在一切量化/锚逻辑之前): 只读层文件+偏移表, 秒级; 旧位置在量化遍之后=会重跑全量化(bug已修) */
    if(getenv("DS4_MERGE_GGUF")){
        /* ===== 合并: 每层 tuned 1bit 字节 → go1b 骨架 GGUF 专家槽(pwrite 偏移) =====
         * 需: DS4_MERGE_GGUF=骨架gguf(deepseek4-quantize --experts go1b 产, type40)
         *     DS4_MERGE_OFF=偏移表(gguf_offsets.py) DS4_LAYER_DIR=层文件目录 */
        const char*gp=getenv("DS4_MERGE_GGUF"),*op=getenv("DS4_MERGE_OFF"),*ld=getenv("DS4_LAYER_DIR");
        if(!op||!ld){ fprintf(stderr,"[merge] 需 DS4_MERGE_OFF + DS4_LAYER_DIR\n"); return 1; }
        /* 偏移表: name → (off,type,nel) */
        typedef struct { int L; char kind[8]; long off,nel; int ty; } mo_t;
        mo_t *MO=calloc((size_t)NL*3,sizeof(mo_t)); int nmo=0;
        FILE*f=fopen(op,"r"); if(!f){ perror("off"); return 1; }
        { char nm[256]; int ty; long off,nel;
          while(fscanf(f,"%255s %d %ld %ld",nm,&ty,&off,&nel)==4){
            int Lx; char kind[16]; int nfx=0;
            /* %n 全串守卫: ffn_gate_tid2eid 等名字也能让 %d+%[^_] 转换成功(返回2), 无守卫时伪匹配
             * 偷占 NL*3 容量槽, 把文件序靠后的真 exps(blk.42)挤出 → "偏移缺失"。nfx>0 且到串尾才算数。 */
            if(sscanf(nm,"blk.%d.ffn_%15[^_]_exps.weight%n",&Lx,kind,&nfx)==2&&nfx>0&&nm[nfx]==0&&Lx<NL&&nmo<NL*3){
                MO[nmo].L=Lx; snprintf(MO[nmo].kind,8,"%s",kind);
                MO[nmo].off=off; MO[nmo].nel=nel; MO[nmo].ty=ty; nmo++; } } }
        fclose(f);
        fprintf(stderr,"[merge] 偏移表专家 tensor %d 个(期望 %d)\n",nmo,NLAYERS*3);
        FILE*gf=fopen(gp,"r+b"); if(!gf){ perror("gguf骨架"); return 1; }
        size_t szG=(size_t)MOEI*go1b_blk_row_bytes(DIM), szD=(size_t)DIM*go1b_blk_row_bytes(MOEI);
        long done=0,skipped=0;
        for(int L=0;L<NLAYERS;L++){
            char lp[512]; snprintf(lp,sizeof(lp),"%s/dql_L%02d.bin",ld,L);
            int fd=open(lp,O_RDONLY);
            if(fd<0){
                /* 续并(DS4_MERGE_RESUME, 脚本核实 GGUF 在位后才给): consume 设计 = 注入成功才 unlink,
                 * 缺失 ⇒ 该层字节已在 GGUF ⇒ 跳过。非续并保持硬停 — 缺层=半成品, 不许静默出洞。 */
                if(getenv("DS4_MERGE_RESUME")){ skipped++;
                    fprintf(stderr,"[merge] L%02d 层文件缺席 → 已消费(此前注入), 跳过\n",L); continue; }
                fprintf(stderr,"[merge] 缺层文件 %s — 停\n",lp); return 1;
            }
            struct stat st2; fstat(fd,&st2); size_t msz=(size_t)st2.st_size;
            uint8_t*mp=mmap(NULL,msz,PROT_READ,MAP_PRIVATE,fd,0); close(fd);
            if(mp==MAP_FAILED){ fprintf(stderr,"[merge] mmap %s 失败\n",lp); return 1; }
            /* 定位 1bit 记录 payload */
            const uint8_t*p=mp+12,*end=mp+msz,*w1=NULL,*w3=NULL,*w2=NULL;
            uint32_t nrec; memcpy(&nrec,mp+8,4);
            for(uint32_t i=0;i<nrec&&p+116<=end;i++){
                char nm[17]; memcpy(nm,p,16); nm[16]=0;
                uint64_t psz; memcpy(&psz,p+88,8); const uint8_t*pay=p+116; p=pay+psz;
                if(!strcmp(nm,"1bit")&&psz>=(uint64_t)NEXP*(2*szG+szD)){
                    w1=pay; w3=pay+(size_t)NEXP*szG; w2=pay+2*(size_t)NEXP*szG; break; }
            }
            if(!w1){ fprintf(stderr,"[merge] L%02d 无 1bit 记录\n",L); munmap(mp,msz); return 1; }
            mo_t *tg=NULL,*tu=NULL,*td=NULL;
            for(int k=0;k<nmo;k++) if(MO[k].L==L){
                if(!strcmp(MO[k].kind,"gate"))tg=&MO[k];
                else if(!strcmp(MO[k].kind,"up"))tu=&MO[k];
                else if(!strcmp(MO[k].kind,"down"))td=&MO[k]; }
            if(!tg||!tu||!td){ fprintf(stderr,"[merge] L%02d 偏移缺失\n",L); munmap(mp,msz); return 1; }
            if(tg->ty!=40||tu->ty!=40||td->ty!=40){ fprintf(stderr,"[merge] L%02d 骨架非go1b(40): %d/%d/%d — 拒写(需 --experts go1b 骨架)\n",L,tg->ty,tu->ty,td->ty); munmap(mp,msz); return 1; }
            int ok=1;
            ok&=fseeko(gf,tg->off,SEEK_SET)==0&&fwrite(w1,1,(size_t)NEXP*szG,gf)==(size_t)NEXP*szG;
            ok&=fseeko(gf,tu->off,SEEK_SET)==0&&fwrite(w3,1,(size_t)NEXP*szG,gf)==(size_t)NEXP*szG;
            ok&=fseeko(gf,td->off,SEEK_SET)==0&&fwrite(w2,1,(size_t)NEXP*szD,gf)==(size_t)NEXP*szD;
            munmap(mp,msz);
            if(!ok){ fprintf(stderr,"[merge] L%02d 写失败\n",L); return 1; }
            done++;
            fprintf(stderr,"[merge] L%02d 注入 (%.0f MiB)\n",L,(2.0*NEXP*szG+NEXP*szD)/1048576.0);
            if(getenv("DS4_MERGE_CONSUME")){ unlink(lp);   /* 边并边释放: 稀疏骨架回填+层文件删除, 峰值盘占恒定(挤盘刚需) */
                fprintf(stderr,"[merge] L%02d 已消费(层文件释放)\n",L); }
        }
        fflush(gf); fclose(gf);
        if(skipped) printf("MERGE_GGUF %s 注入层=%ld 已消费跳过=%ld 合计=%ld/%d 完成\n",gp,done,skipped,done+skipped,NLAYERS);
        else        printf("MERGE_GGUF %s 注入层=%ld/%d 完成\n",gp,done,NLAYERS);
        fflush(stdout);
        return 0;
    }
    if(getenv("DS4_ZK")) ZK=atoi(getenv("DS4_ZK"));
    if(getenv("DS4_ABLATE")) ABLATE=atoi(getenv("DS4_ABLATE"));
    if(getenv("DS4_LOCAL_Q")) LOCALQ=atoi(getenv("DS4_LOCAL_Q"));
    if(getenv("DS4_LZ")) LZRANK=atoi(getenv("DS4_LZ"));
    if(getenv("DS4_LZ_LAMBDA")) LZLAMBDA=atof(getenv("DS4_LZ_LAMBDA"));
    if(getenv("DS4_LZ_TR")) LZTR=atof(getenv("DS4_LZ_TR"));
    TUNE_T0=time(NULL);
    if(getenv("DS4_TUNE_MIN")) TUNE_MIN=atof(getenv("DS4_TUNE_MIN"));
    if(getenv("DS4_FAST")) FAST=atoi(getenv("DS4_FAST"));
    if(getenv("DS4_BACKFIT_INCR")) BACKFIT_INCR=atoi(getenv("DS4_BACKFIT_INCR"));
    if(getenv("DS4_BF_MEMGB")) BF_MEMGB=atof(getenv("DS4_BF_MEMGB"));
    if(getenv("DS4_COADAPT")){ COADAPT=atoi(getenv("DS4_COADAPT")); if(COADAPT<0)COADAPT=0;
        if(COADAPT>0&&LZRANK==0){ LZRANK=16; fprintf(stderr,"[共适应] DS4_LZ 未设 → z rank≤16\n"); } }
    if(getenv("DS4_GO2B_HOT")&&atoi(getenv("DS4_GO2B_HOT"))){
        /* 热专家 go2b 合并态量化(残差+量化一体, 消漂移): 热表加载失败=硬拒(禁静默退 go1b) */
        const char*hp=getenv("DS4_GO2B_HOT_TABLE");
        const char*cands[3]={hp,"../corpus/prog_active_top64.txt","gguf-tools/go-onebit/corpus/prog_active_top64.txt"};
        int okh=-1;
        for(int i=0;i<3;i++){ if(!cands[i])continue; if(go2b_hot_load(cands[i],NLAYERS)==0){ okh=i; break; } }
        if(okh<0){ fprintf(stderr,"[go2b] ★热表读失败(DS4_GO2B_HOT_TABLE/默认两处) — 硬拒★\n"); exit(8); }
        GO2B_HOT=1;
        int tot=0; for(int L2=0;L2<NLAYERS&&L2<64;L2++) tot+=G2_K[L2];
        fprintf(stderr,"[go2b] 热表 %s: Σ热=%d (热=合并2bit GPTQ+act2 激活最优; 冷=go1b; 反修/回放在合并态)\n",
                cands[okh],tot);
    }
    { long nc=sysconf(_SC_NPROCESSORS_ONLN); NTHREADS=(int)(nc>8?8:(nc<1?1:nc));
      if(getenv("DS4_THREADS")) NTHREADS=atoi(getenv("DS4_THREADS")); if(NTHREADS<1)NTHREADS=1; }
    /* ★2048帽拆除(2026-08-10 反修v4: 帽导致>2048语料全部静默截断, 锚头S=2048 与读取端
     * NTOK 错位 = "末层损坏"幻象/SCORE B 假口径/链锚案 三案同源)★ */
    int idcap=ntok>2048?ntok:2048;
    long *ids=malloc((size_t)idcap*sizeof(long));int S=0;FILE*f=fopen(idf,"r");
    if(!f){ fprintf(stderr,"ids 文件 %s 打不开\n",idf); return 1; }
    { char ln[64]; while(S<ntok&&fgets(ln,sizeof(ln),f))ids[S++]=atol(ln); fclose(f); }
    if(S<3){ fprintf(stderr,"token 太少 (S=%d)\n",S); return 1; }
    int n_fit=(int)(S*3/4);
    if(getenv("DS4_NFIT")) n_fit=atoi(getenv("DS4_NFIT"));
    if(n_fit<1)n_fit=1; if(n_fit>=S)n_fit=S>1?S-1:1;   /* 3/4 fit(校准), 1/4 held-out(判决) */
    /* 逐层档位: DS4_LCFG(NL字符或1字符广播) > DS4_INJECT > DS4_SPARE > 全'1' */
    for(int i=0;i<NLAYERS;i++)LCFG[i]='1'; LCFG[NLAYERS]=0;
    const char*lc=getenv("DS4_LCFG");
    if(lc){ size_t n=strlen(lc);
        if(n==1) for(int i=0;i<NLAYERS;i++)LCFG[i]=lc[0];
        else if(n>=(size_t)NLAYERS) memcpy(LCFG,lc,(size_t)NLAYERS);
        else { fprintf(stderr,"DS4_LCFG 长度 %zu < NL=%d (可 1 字符广播)\n",n,NLAYERS); return 1; } }
    else if(getenv("DS4_INJECT")){ int L=atoi(getenv("DS4_INJECT"));
        char q=getenv("DS4_QCHAR")?getenv("DS4_QCHAR")[0]:'1';
        for(int i=0;i<NLAYERS;i++)LCFG[i]='F'; if(L>=0&&L<NLAYERS)LCFG[L]=q; }
    else if(getenv("DS4_SPARE")){ int L=atoi(getenv("DS4_SPARE"));
        char q=getenv("DS4_QCHAR")?getenv("DS4_QCHAR")[0]:'1';
        for(int i=0;i<NLAYERS;i++)LCFG[i]=q; if(L>=0&&L<NLAYERS)LCFG[L]='F'; }
    for(int i=0;i<NLAYERS;i++) if(!strchr("Fn1z23rgmB",LCFG[i])){ fprintf(stderr,"LCFG[%d]='%c' 非法(F/n/1/r/g/z/2/3/m/B)\n",i,LCFG[i]); return 1; }
    if(TUNE_MIN>0) fprintf(stderr,"[预算] %.1f 分仅作选档/预估 — 收敛绝对优先: 每层(本层+累积)整轮零接管才进下一层, 不截断\n",TUNE_MIN);
    fprintf(stderr,"ntok=%d n_fit=%d held=%d NL=%d 线程=%d ZK=%d\nLCFG=%s\n",S,n_fit,S-n_fit,NLAYERS,NTHREADS,ZK,LCFG);
    if(getenv("DS4_GENPROBE")){
        /* ★生成侧验证器(2026-07-29 用户"先验证, 不要浪费时间"): 回放态自回归贪心 N token —
         * 教师闭环之外的第一个生成证据(词汤/复读直接检测)。prompt=ids 文件全体;
         * 层来源: DS4_LAYER_DIR 有 dql 的层='B' 字节回放(g10h 原生; 内嵌 g2hot 对 bytes_moe
         * 透明), 其余='F' FP。路由现算(锚不载, ANC_OK=0, 生成序列≠锚序列)。 */
        int N=atoi(getenv("DS4_GENPROBE")); if(N<1)N=24;
        int P=S;
        long *gg=malloc((size_t)(P+N)*sizeof(long)); memcpy(gg,ids,(size_t)P*sizeof(long));
        const char*ld2=getenv("DS4_LAYER_DIR");
        char lc2[64]; for(int i=0;i<NLAYERS;i++)lc2[i]='F'; lc2[NLAYERS]=0;
        if(ld2){ if(!GS_LF)GS_LF=calloc((size_t)NLAYERS,sizeof(lfile_t));
            for(int Lx=0;Lx<NLAYERS;Lx++){ char lp[512];
                snprintf(lp,sizeof(lp),"%s/dql_L%02d.bin",ld2,Lx);
                if(!GS_LF[Lx].map && lfile_load(lp,&GS_LF[Lx])==0) lc2[Lx]='B';
                else if(GS_LF[Lx].map) lc2[Lx]='B'; } }
        int nb=0; for(int i=0;i<NLAYERS;i++) if(lc2[i]=='B')nb++;
        fprintf(stderr,"[genprobe] prompt=%d tok 生成=%d 回放层=%d/%d(其余FP) LCFG=%s\n",P,N,nb,NLAYERS,lc2);
        for(int t=0;t<N;t++){
            int S3=P+t;
            float *lg=fwd_all(gg,S3,S3,1,lc2);
            const float*row=lg+(size_t)(S3-1)*VOCAB;
            int bi=0; float bv=row[0];
            for(int v=1;v<VOCAB;v++) if(row[v]>bv){bv=row[v];bi=v;}
            free(lg);
            gg[P+t]=bi;
            fprintf(stderr,"[genprobe] t=%d/%d id=%d\n",t+1,N,bi);
            printf("GEN t=%d id=%d\n",t,bi); fflush(stdout);
        }
        printf("GENIDS"); for(int t=0;t<N;t++) printf(" %ld",gg[P+t]); printf("\n"); fflush(stdout);
        return 0;
    }
    /* FP 锚定: 有缓存直接用, 无则跑一次并落盘 */
    uint64_t idh=dq_ids_hash(ids,S);
    if(anchor_load(S,idh)){ ANC_OK=1; fprintf(stderr,"[anchor] 命中缓存 %s (FP 遍跳过)\n",anchor_path()); }
    else{
        fprintf(stderr,"[anchor] 无缓存/不匹配 → 跑 FP 锚定遍(一次性)...\n");
        anchor_alloc(S); ANC.idh=idh; ANC_BUILD=1;
        ANC.logits=fwd_all(ids,S,n_fit,0,NULL);
        ANC_BUILD=0; ANC_OK=1;
        if(anchor_save()) fprintf(stderr,"[anchor] 已写 %s\n",anchor_path());
    }
    if(getenv("DS4_FP_ONLY")){ fprintf(stderr,"[anchor] DS4_FP_ONLY=1, 到此为止\n"); return 0; }
    if(getenv("DS4_BBQ4_AB")){
        /* backbone q4 真 A/B(2026-07-28): 锚=FP 参照, 同进程武装 q4 重跑完整前向(含 attention
         * 从 st 重读) → 同口径 verdict。旧 env-only 钩子在锚缓存模式测不到 backbone(学生回放
         * 不重读 backbone 权重), 本模式是唯一诚实口径。 */
        fprintf(stderr,"[bbq4-ab] 武装 q4 重跑完整前向...\n");
        g_bbq4=1;
        float *lq=fwd_all(ids,S,n_fit,0,NULL);
        g_bbq4=0;
        char lc2[64]; for(int i2=0;i2<NLAYERS;i2++)lc2[i2]='F'; lc2[NLAYERS]=0;
        verdict(ANC.logits,lq,ids,S,n_fit,lc2);
        return 0;
    }
    if(getenv("DS4_SIGNREF_MU")) dq_signref_mu=atof(getenv("DS4_SIGNREF_MU"));
    if(getenv("DS4_SIGNREF_ROUNDS")) dq_signref_rounds=atoi(getenv("DS4_SIGNREF_ROUNDS"));
    if(getenv("DS4_MINVOL")&&getenv("DS4_RR_IDS")) rr_anchor_init();   /* ★统一标准: rr 判决锚(载/建一次) */
    if(getenv("DS4_REPAIR_COLD")){ repair_cold(S,n_fit); return 0; }   /* ②冷专家修复(新锚=当前语料锚) */
    if(getenv("DS4_EXPORT_GGUF")){ export_gguf(S,n_fit); return 0; }
    if(getenv("DS4_TUNE")){
        char plan[NL+1];
        fprintf(stderr,"渐进调优模式 (plan=%s ckpt=%s)...\n",plan_path(),ckpt_dir());
        float *lq=fwd_all_tune(ids,S,n_fit,plan);
        verdict(ANC.logits,lq,ids,S,n_fit,plan);
        zfile_write();   /* 合并动态侧车(逐层搜索胜者)落盘 */
        zchain_write();  /* DQZ2 全链运行时侧车(层文件被 consume 前的唯一留存) */
        return 0;
    }
    fprintf(stderr,"量化遍 (LCFG=%s)...\n",LCFG);
    size_t g_chtot=0;
    { const char*chp=getenv("DS4_CHAIN_ANCHOR");
      if(chp){
        g_chfd=open(chp,O_RDWR|O_CREAT|O_TRUNC,0644);
        if(g_chfd<0){ perror("chain-anchor"); exit(1); }
        g_chS=S;
        g_chtot=40+(size_t)NLAYERS*S*DIM*4+2*(size_t)NLAYERS*S*NACT*4
               +(size_t)NLAYERS*S*HCM*DIM*4+(size_t)S*VOCAB*4;
        if(ftruncate(g_chfd,(off_t)g_chtot)!=0){ perror("chain-trunc"); exit(1); }
        uint32_t hd9[8]={0x32415144u,(uint32_t)S,(uint32_t)HCM,(uint32_t)DIM,(uint32_t)NLAYERS,(uint32_t)VOCAB,(uint32_t)NACT,0};
        uint64_t idh9=dq_ids_hash(ids,S);
        pwrite(g_chfd,hd9,32,0); pwrite(g_chfd,&idh9,8,32);
        fprintf(stderr,"[链态锚] 直写 %s (%.2f GiB)\n",chp,g_chtot/1073741824.0);
      } }
    float *lq=fwd_all(ids,S,n_fit,1,LCFG);
    if(g_chfd>=0){
        pwrite(g_chfd,lq,(size_t)S*VOCAB*4,(off_t)(g_chtot-(size_t)S*VOCAB*4));
        close(g_chfd); g_chfd=-1;
        fprintf(stderr,"[链态锚] 完成\n");
    }
    if(getenv("DS4_DUMP_LOGITS")){   /* 逐位置量化 logits 落盘(瑕疵归因: FP=锚内 logits, 量化=此) */
        FILE*df=fopen(getenv("DS4_DUMP_LOGITS"),"wb");
        if(df){ int hd2[2]={S,(int)VOCAB}; fwrite(hd2,4,2,df);
                fwrite(lq,4,(size_t)S*VOCAB,df); fclose(df);
                fprintf(stderr,"[dump] 量化 logits → %s (S=%d)\n",getenv("DS4_DUMP_LOGITS"),S); } }
    verdict(ANC.logits,lq,ids,S,n_fit,LCFG);
    rb_save();       /* ★BF_ONLY 评测遍也落盘 Δb(2026-08-13 路由归因: FIT 寄生回放前向, 收官必须存) */
    zfile_write();   /* 合并动态侧车(逐层搜索胜者)落盘 */
    zchain_write();  /* DQZ2 全链运行时侧车(层文件被 consume 前的唯一留存) */
    if(getenv("DS4_LAYER_DIR")&&getenv("DS4_GSWEEP")){
        global_sweep(ids,S,n_fit);   /* ★全局联合回扫: 最终输出判据, 反修所有层★ */
        zchain_write();              /* 回扫原地改写了层文件 op 载荷 → 刷新全链侧车 */
    }
    if(getenv("DS4_EXPORT_LAYER")){
        int L=atoi(getenv("DS4_EXPORT_LAYER"));
        const char*lf=getenv("DS4_LAYER_FILE"); if(!lf)lf="/tmp/dql_L00.bin";
        export_layer_file(L,S,n_fit,lf);
    }
    return 0;
}
