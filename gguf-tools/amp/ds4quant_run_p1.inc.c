/* ds4quant_run.c — 锚定-累积-快验: 43层量化前向以【最终输出质量】为判决, 逐层只是诊断。
 *
 * 方法学修正(2026-07-10, 取代旧"两遍前向+每层4档消融"):
 *   ① 逐层局部 relL2 高 ≠ 层本质差 —— 可能只是校准"没找到"(样本 1~3 行的 joint-LS 是噪声拟合)。
 *      → 每层报告校准样本数; 校准改为部署口径(anchor FP 激活 + FP 路由 + 只用 fit 行, 不再泄漏 held-out)。
 *   ② 每层独立最优 ≠ 43 层堆叠最优 —— 误差层间累积/抵消, 唯一判决 = 最终 logits 质量。
 *      → 量化遍逐层打印【累积】偏差(H_q vs anchor H_fp)+路由一致率, 结束打 VERDICT(Σmin/KL/PPL 主判据)。
 *
 * 机制:
 *   - FP 锚定遍只跑一次, 缓存到 --anchor(默认 /tmp/ds4quant_anchor.bin): 每层 MoE 输入 Fin_fp、
 *     FP 路由、每层出口 H_fp、最终 logits。之后所有配置验证只跑量化遍。
 *   - --lcfg 逐层档位(NL 字符或 1 字符广播): F=不量化 n=朴素1bit 1=joint1bit(块scale,1.0625b)
 *     r=行scale1bit(1.0052b, 纯1bit体积主线) z=1bit+低秩 2=1bit+Q2 3=1bit+Q2+Q3。
 *     全 F 配置是框架自检: VERDICT 必须 ratio=1.0000/Σmin≈1。
 *   - 专家循环 pthread 并行(--threads) + Accelerate sgemm(ds4quant_fwd.c) → 单配置分钟级。
 *
 * 编译: cc -O3 -lm -framework Accelerate -I<repo根> ds4quant_run.c -o ds4quant_run -lpthread
 * 用法: ./ds4quant_run [ids_file=/tmp/rr_hard.ids] [ntok=64] --hf <dir> [--flag ...]
 *       (全部旋钮见 ds4quant_cli.inc.c 的表; 还原率铁律: 判决用硬多样文本 rr_hard.ids)
 */
#ifdef __linux__
#include <malloc.h>   /* mallopt: 分配器设置写进代码, 不走 env(2026-08-22 铁律); glibc 专属, macOS 无此接口 */
#endif
#include "../calib/st_read.c"
#include "ds4quant_fwd.c"
#include "../quantize/onebit_quant.c"
#include "ds4quant_qhelp.h"   /* dq_go1b_bytes_dequant, dq_quant_expert (从 layer.c 抽出) */
#include "../quantize/go2b_qc.h"          /* GO2B 编码/解码+热表: 热专家合并2bit(残差+量化一体, 消漂移) */
#include "../quantize/vq_qc.h"            /* v2.2 VQ 码本(DS4_VQ=1): 热 vq4x512 全三矩阵 / 冷 w1w3 vq8x256 / 冷 w2 signref */
#include "ds4_z.c"            /* 闭式秩-k RRR 隐变量 z (产物③) */
#include "ds4_loss.c"        /* 四损失: align/classify/smooth/fixed */
#include "../../src/common/ds4_amp_fmt.h"   /* 记录头/λ clamp/秩上限/fit切分: 与引擎同一契约 */
#include <pthread.h>
#include <time.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>


#include <sys/stat.h>
#ifdef __APPLE__
#include <mach/mach.h>   /* 内存自报: task_info phys_footprint(看门狗同口径) */
#endif
static int   ZRANK=16;        /* z 最大秩(遗留 z 工具用; 2026-08-31 起写死, env 旋钮已清退) */
static float ZLAMBDA=0.1f;    /* ridge (无量纲) = L_fixed 锚 */
static int   NLAYERS=43;      /* 前向层数上限 (--nl; <43 绝对PPL无效, 相对指标同深度仍可比) */
static int   ZK=16;           /* 低秩档 'z' 的秩(写死) */
static int   NTHREADS=8;      /* 专家循环并行度 (--threads) */
static int   LZRANK=0;        /* 逐层动态 z^L 最大秩(0=关): --coadapt>0 时升 DSQ_COADAPT_LZRANK */
static float LZLAMBDA=1.0f;   /* z^L ridge(比全局 ZLAMBDA 紧; gz首跑过拟合链式反应教训) */
static float LZTR=0.5f;       /* 信赖域 — 每 token ‖z(Fin)‖≤TR·‖Fout‖, 防把激活拉出流形 */
static long  LZ_TOTAL_K=0;    /* Σk_L (侧车体积账) */
static int   COADAPT=0;       /* --coadapt: 每层 base(w2)↔z 交替闭式共适应轮数(0=关=原路径不变) */
static time_t TUNE_T0=0;      /* 全程计时起点 */
static double TUNE_MIN=0;     /* --tune-min: 总时间预算(分钟, 0=不限) */
static int FAST=0;            /* --fast: 快速走流程(QC 5→1 变体, co_rounds 512→2 轮封顶, 不向前修复=逐层反修跳过); 验流程非出质量 */
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
/* (xr_npy 捕获器已删: DS4_BF_DUMPXR 诊断路 2026-08-31 env 清退) */
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
/* ★fp32 展开缓存(2026-08-30 BFLT 定谳)★ sweep 单元 attn核 24s 的真因: lwh_expand 每次
 * 前向展开完就 free ⇒ dq_matmul 对普通 malloc 权重(!wdev)每调用 H2D 重传 ~0.3GB。
 * 改成每层展开一次进 managed 单板(vqg_alloc_managed: preferred-GPU+prefetch), cuBLAS
 * 直读零拷贝, 页首触后常驻。0.43GB/层×43=18.7GB, 分配失败则大声回退逐调用展开(不静默)。
 * 前提已核: layer_fwd 对 W-> 零写(权重只读), fp16 母本不变 ⇒ 缓存永不过期。 */
static LW *lw32_get(int L,const LWH*H){
    static LW c[64]; static char ok[64], fail=0;
    if(L>=64) return NULL;
    if(ok[L]) return &c[L];   /* 命中判定先于 loaded 守卫: fp16 母本被预算驱逐后缓存仍有效(展开自不变 fp16, 免 HF 重载) */
    if(fail||!H->loaded) return NULL;
    size_t tot=0;
#define XS(f) tot+=(size_t)H->n_##f;
    LWF_LIST(XS)
#undef XS
    float *slab=NULL;
    size_t bytes=tot*4+(H->t2ei?(size_t)VOCAB*NACT*sizeof(int):0);
#ifdef DS4QUANT_CUDA
    { extern int vqg_alloc_managed(void**,size_t);
      if(!vqg_alloc_managed((void**)&slab,bytes)) slab=NULL; }
#else
    slab=malloc(bytes);
#endif
    if(!slab){ if(!fail){ fail=1; fprintf(stderr,"[lw32] ★managed 分配失败(L%02d, %.2fGB/层) → 回退逐调用展开★\n",L,bytes/1073741824.0); } return NULL; }
    float *p=slab; LW W={0};
#define XE(f) if(H->f){ W.f=p; for(long i=0;i<H->n_##f;i++) p[i]=go1b_fp16_to_fp32(H->f[i]); p+=H->n_##f; }
    LWF_LIST(XE)
#undef XE
    if(H->t2ei){ W.t2ei=(int*)p; memcpy(W.t2ei,H->t2ei,(size_t)VOCAB*NACT*sizeof(int)); }
    c[L]=W; ok[L]=1; return &c[L];
}
static LWH *GS_LW=NULL;        /* 回扫/反修: 全43层骨干 fp16 缓存(只加载一次, 前向按层临时展开) */
/* ★内存卫兵★: 实测斜率 0.281GB/层(fp16 0.217 + 杂项), 外推 L38 必撞 11.5G 看门狗 →
 * 超预算(--bf-memgb, 默认=物理内存一半)驱逐最低层号的 fp16 缓存; 被驱逐层前向时临时从 HF 重载(慢~0.6s/访, 换到头)。*/
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
/* 机器物理内存(GiB); 读不到回退 16(旧默认机型)。分配预算类的默认值一律由它导出,
 * 不再写死 —— 死数在换机器时会静默变成性能杀手(2026-08-27 用户: "之前代码根本不适配
 * 当前的设备了, 当前是 128 的 GPU 场景")。 */
static double dq_host_ram_gb(void) {
    static double g = -1.0;
    if (g < 0) {
        g = 16.0;
        FILE *f = fopen("/proc/meminfo", "r");
        if (f) { char ln[256]; unsigned long kb;
            while (fgets(ln, sizeof ln, f))
                if (sscanf(ln, "MemTotal: %lu kB", &kb) == 1) { g = (double)kb / 1048576.0; break; }
            fclose(f); }
    }
    return g;
}
/* ★别写死 9.5★(2026-08-27): 这是 16GiB Mac 时代的预算。超预算就驱逐 fp16 层缓存, 该层
 * 下次被访问时从 HF 盘重载(~0.6s/访) —— 在 121GiB 机器上等于"内存明明够, 却坚持走盘"。
 * 改为物理内存的一半(spark≈60GiB, 旧 16G 机上≈8GiB 与原值同量级), --bf-memgb 仍可覆盖。 */
static double BF_MEMGB_init(void){ return dq_host_ram_gb() * 0.5; }
static double BF_MEMGB=0;   /* 0=未初始化, 首次 mem_evict 时由 BF_MEMGB_init 填(见 p14 flag 覆盖) */
static void lwh_free(LWH*H);
static void gs_lw_evict(int upto){   /* 驱逐 [0,upto) 里已缓存的最低层, 直到回预算 */
    if(!GS_LW) return;
    if(BF_MEMGB<=0) BF_MEMGB=BF_MEMGB_init();
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
/* z_solve_dual 拆去了 ds4quant_zsolve.inc.c(紧跟本文件 include), 但本文件后面还有调用者,
 * 故留一条 static 前向声明。 */
static ds4_z *z_solve_dual(const float *X, const float *R, uint32_t n,
                           uint32_t d_in, uint32_t d_out, uint32_t rank, float lambda);
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
