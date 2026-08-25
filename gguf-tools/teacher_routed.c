/* teacher_routed.c — 教师(原始 HF FP)的 routed-MoE 输出, 供四损失求解器当 y*。
 * C 实现(2026-08-22 铁律: 反修与引擎复用同一份口径, 禁 Python 另写一套)。
 *
 *   y*_t = Σ_{e∈S_t} w_{t,e} · Expert_FP_e(x_t)
 *
 * x_t / S_t / w_{t,e} 全部取自**引擎捕获**(解码路 = 部署同路), 只有专家权重来自 HF ——
 * 于是 y* − ŷ 是纯粹的"权重被量化"那部分, 路由与输入两边逐位同源, 不掺任何口径差。
 * (早先 Python 版在这里混用了 FP 锚的输入/路由, 导致靶子变成一个不存在的混合模型。)
 *
 * 输入(npy, cap 契约): ffn_in_L{L} [n×4096] f16, route_L{L} [n×6] i16, route_w_L{L} [n×6] f16
 * 输出: routed_L{L}.npy [n×4096] <f4
 *
 * 专家权重按需读入并缓存到本层用完; 每层最多常驻 256×3 个矩阵 ⇒ 按专家流式, 峰值几 GiB。
 * 用法: teacher_routed --hf DIR --cap DIR --layers a-b [--ntok N] [--threads T] [--swlim F]
 *       [--anchor FILE]  ★FP 锚口径(2026-08-23 用户裁决"教师必须是要还原的那个模型")★:
 *       x/路由/权重改读锚(FP 模型自己的轨迹: fin[NL][S][D] + ridx/rw[NL][S][NACT], off=40),
 *       替代引擎轨迹 —— 引擎轨迹教师把路由漂移/上游量化效应从靶里剔除, 靶变纯噪声(全层
 *       闸的根因, zlayer.py 里 2026-08-22 已裁决过同一错误)。输出仍写 cap 目录 routed_L。
 * 纯 C99 + pthread。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include "go-onebit/calib/npy.h"
/* 0731 的 routed 专家是 MXFP4(I8 容器), hf_read.c 不认 —— 改用 quant/st_read.c
 * (量化器同款读器; 今天刚给它做过 shard 句柄+头缓存、pread、页缓存旁路)。 */
#include "go-onebit/quant/st_read.c"
#include "go-onebit/calib/layer_probe.h"

#define DM 4096
#define DFF 2048
#define TOPK 6
#define NEXP 256

static int write_npy_f32(const char *path, const float *d, int64_t n, int64_t w) {
    char dict[256];
    int k = snprintf(dict, sizeof dict,
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%lld, %lld, ), }",
        (long long)n, (long long)w);
    int base = 10 + k + 1, pad = (64 - (base % 64)) % 64;
    for (int i = 0; i < pad && k < (int)sizeof dict - 1; i++) dict[k++] = ' ';
    dict[k++] = '\n';
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    unsigned char h[8] = {0x93,'N','U','M','P','Y',1,0};
    unsigned char hl[2] = {(unsigned char)(k & 0xff), (unsigned char)((k >> 8) & 0xff)};
    size_t nb = (size_t)n * w * sizeof(float);
    int ok = fwrite(h,1,8,f)==8 && fwrite(hl,1,2,f)==2 &&
             fwrite(dict,1,(size_t)k,f)==(size_t)k && fwrite(d,1,nb,f)==nb;
    fclose(f); return ok ? 0 : -1;
}

typedef struct {
    const float *x, *rw; const float *ids;
    const float *gate, *up, *down;      /* 当前专家的三矩阵 */
    float *acc;                          /* [n×DM] 累加 */
    const int *rows; int n_rows;         /* 命中该专家的 token 行 */
    int slot_of_row_base; const int *slot;
    float swlim;
    int lo, hi;                          /* 本线程负责的 rows 区间 */
} work_t;

static void *worker(void *p) {
    work_t *w = (work_t *)p;
    float o[DM];
    for (int i = w->lo; i < w->hi; i++) {
        int t = w->rows[i], s = w->slot[i];
        expert_forward_f32_lim(w->x + (size_t)t * DM, w->gate, w->up, w->down,
                               o, DM, DFF, w->swlim);
        float g = w->rw[(size_t)t * TOPK + s];
        float *a = w->acc + (size_t)t * DM;
        for (int d = 0; d < DM; d++) a[d] += g * o[d];
    }
    return NULL;
}

int main(int argc, char **argv) {
    const char *hf_dir = NULL, *cap = NULL;
    int lo = 0, hi = 42, ntok_want = 0, nth = 8;
    float swlim = 0.0f; const char *anchor = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i],"--hf") && i+1<argc) hf_dir = argv[++i];
        else if (!strcmp(argv[i],"--cap") && i+1<argc) cap = argv[++i];
        else if (!strcmp(argv[i],"--ntok") && i+1<argc) ntok_want = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--threads") && i+1<argc) nth = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--swlim") && i+1<argc) swlim = (float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--anchor") && i+1<argc) anchor = argv[++i];
        else if (!strcmp(argv[i],"--layers") && i+1<argc) {
            if (sscanf(argv[++i], "%d-%d", &lo, &hi) != 2) { fprintf(stderr,"--layers a-b\n"); return 2; }
        } else { fprintf(stderr,"未知参数 %s\n", argv[i]); return 2; }
    }
    if (!hf_dir || !cap) { fprintf(stderr,
        "用法: teacher_routed --hf DIR --cap DIR --layers a-b [--ntok N] [--threads T] [--swlim F]\n"); return 2; }
    if (nth < 1) nth = 1; if (nth > 64) nth = 64;

    st_ctx C; st_open(&C, hf_dir);   /* 失败会自行 exit */

    for (int L = lo; L <= hi; L++) {
        char p[1024]; npy_meta mx, mi, mw;
        float *x = NULL, *id = NULL, *rw = NULL;
        if (anchor) {   /* FP 锚口径: fin/ridx/rw @ off 40, [NL][S][*] 布局 */
            FILE *af = fopen(anchor, "rb");
            if (!af) { fprintf(stderr, "锚 %s 打不开\n", anchor); return 3; }
            uint32_t hd[8];
            if (fread(hd,4,8,af)!=8 || hd[0]!=0x32415144u) { fprintf(stderr,"%s: 不是 DQA2 锚\n",anchor); return 3; }
            const long long S=hd[1], DIMa=hd[3], NLa=hd[4], NACT=hd[6];
            if (DIMa!=DM || NACT!=TOPK) { fprintf(stderr,"锚形状不符 DIM=%lld NACT=%lld\n",DIMa,NACT); return 3; }
            long long n_use = ntok_want>0 && ntok_want<S ? ntok_want : S;
            const long long fin_off=40, ridx_off=fin_off+NLa*S*DM*4, rw_off=ridx_off+NLa*S*TOPK*4;
            x  = malloc((size_t)n_use*DM*sizeof(float));
            id = malloc((size_t)n_use*TOPK*sizeof(float));
            rw = malloc((size_t)n_use*TOPK*sizeof(float));
            int32_t *idi = malloc((size_t)n_use*TOPK*sizeof(int32_t));
            fseek(af, (long)(fin_off + (long long)L*S*DM*4), SEEK_SET);
            if (fread(x,4,(size_t)n_use*DM,af)!=(size_t)n_use*DM) { fprintf(stderr,"锚 fin 截断\n"); return 3; }
            fseek(af, (long)(ridx_off + (long long)L*S*TOPK*4), SEEK_SET);
            if (fread(idi,4,(size_t)n_use*TOPK,af)!=(size_t)n_use*TOPK) { fprintf(stderr,"锚 ridx 截断\n"); return 3; }
            fseek(af, (long)(rw_off + (long long)L*S*TOPK*4), SEEK_SET);
            if (fread(rw,4,(size_t)n_use*TOPK,af)!=(size_t)n_use*TOPK) { fprintf(stderr,"锚 rw 截断\n"); return 3; }
            fclose(af);
            for (long long q=0;q<n_use*TOPK;q++) id[q]=(float)idi[q];
            free(idi);
            /* ★锚口径自检★ 锚 x(FP 轨迹) vs 引擎 x(链态) 同行 cos 中位 — 两条轨迹的同
             * 一 token 输入应中高相关(~0.5-0.9); ≈0 = 锚读取错位/布局错, 停车。 */
            {
                npy_meta mc; char pc[1024];
                snprintf(pc,sizeof pc,"%s/ffn_in_L%d.npy",cap,L);
                float *xc = npy_read_f32(pc,&mc);
                if (xc && mc.shape[0] >= n_use) {
                    double cs[4096]; long long nn = n_use < 4096 ? n_use : 4096;
                    for (long long t=0;t<nn;t++) {
                        const float *a=x+(size_t)t*DM, *b=xc+(size_t)t*DM;
                        double d=0,na=0,nb=0;
                        for (int j=0;j<DM;j++){d+=(double)a[j]*b[j];na+=(double)a[j]*a[j];nb+=(double)b[j]*b[j];}
                        cs[t]=(na>0&&nb>0)?d/(sqrt(na)*sqrt(nb)):0;
                    }
                    /* 中位: 简易排序 */
                    for (long long a1=0;a1<nn;a1++) for (long long b1=a1+1;b1<nn;b1++)
                        if (cs[b1]<cs[a1]) { double tt=cs[a1];cs[a1]=cs[b1];cs[b1]=tt; }
                    /* 错位对照: cap 前插一行(BOS)会让对角跌、+1 错位跳高 */
                    double cs2[4096];
                    for (long long t=0;t<nn-1;t++) {
                        const float *a=x+(size_t)t*DM, *b=xc+(size_t)(t+1)*DM;
                        double d=0,na=0,nb=0;
                        for (int j=0;j<DM;j++){d+=(double)a[j]*b[j];na+=(double)a[j]*a[j];nb+=(double)b[j]*b[j];}
                        cs2[t]=(na>0&&nb>0)?d/(sqrt(na)*sqrt(nb)):0;
                    }
                    for (long long a1=0;a1<nn-1;a1++) for (long long b1=a1+1;b1<nn-1;b1++)
                        if (cs2[b1]<cs2[a1]) { double tt=cs2[a1];cs2[a1]=cs2[b1];cs2[b1]=tt; }
                    fprintf(stderr,"L%d 锚自检: 对角=%.4f 错位+1=%.4f (锚[t] vs cap[t+1])\n",L,cs[nn/2],cs2[(nn-1)/2]);
                    /* 路由重合率: 锚 FP 路由 vs 引擎路由, 历史参照 ~3.17/6; ~0.14/6=随机=不同文本 */
                    {
                        npy_meta mr2; char pr2[1024];
                        snprintf(pr2,sizeof pr2,"%s/route_L%d.npy",cap,L);
                        float *rc = npy_read_f32(pr2,&mr2);
                        if (rc) {
                            double ov = 0; long long nt2 = n_use < mr2.shape[0] ? n_use : mr2.shape[0];
                            for (long long t=0;t<nt2;t++)
                                for (int a2=0;a2<TOPK;a2++)
                                    for (int b2=0;b2<TOPK;b2++)
                                        if ((int)id[t*TOPK+a2]==(int)rc[t*TOPK+b2]) { ov++; break; }
                            fprintf(stderr,"L%d 路由重合: %.2f/6 (参照~3.2; ~0.14=不同文本)\n",L,ov/nt2);
                            free(rc);
                        }
                    }
                    if (cs[nn/2] < 0.15) { fprintf(stderr,"★锚口径错位, 拒跑★\n"); return 5; }
                    free(xc);
                }
            }
            mx.ndim=2; mx.shape[0]=n_use; mx.shape[1]=DM;
            mi.ndim=2; mi.shape[0]=n_use; mi.shape[1]=TOPK;
            mw.ndim=2; mw.shape[0]=n_use; mw.shape[1]=TOPK;
        } else {
            snprintf(p,sizeof p,"%s/ffn_in_L%d.npy",cap,L);   x  = npy_read_f32(p,&mx);
            snprintf(p,sizeof p,"%s/route_L%d.npy",cap,L);    id = npy_read_f32(p,&mi);
            snprintf(p,sizeof p,"%s/route_w_L%d.npy",cap,L);  rw = npy_read_f32(p,&mw);
        }
        if (!x || !id || !rw || mx.shape[1]!=DM || mi.shape[1]!=TOPK || mw.shape[1]!=TOPK) {
            fprintf(stderr,"L%d: cap 张量缺或形状不符\n",L); free(x);free(id);free(rw); continue;
        }
        int64_t n = mx.shape[0];
        if (mi.shape[0]!=n || mw.shape[0]!=n) { fprintf(stderr,"L%d: 行数不一致\n",L); return 3; }
        if (ntok_want > 0 && ntok_want < n) n = ntok_want;

        float *acc = calloc((size_t)n*DM, sizeof(float));
        if (!acc) { fprintf(stderr,"L%d OOM\n",L); return 3; }

        /* 按专家分桶: 每个专家只读一次权重 */
        int *cnt = calloc(NEXP,sizeof(int));
        for (int64_t t=0;t<n;t++) for (int s=0;s<TOPK;s++) { int e=(int)id[t*TOPK+s]; if(e>=0&&e<NEXP) cnt[e]++; }
        int **rows = calloc(NEXP,sizeof(int*)), **slot = calloc(NEXP,sizeof(int*)), *fill = calloc(NEXP,sizeof(int));
        for (int e=0;e<NEXP;e++) if (cnt[e]) { rows[e]=malloc(cnt[e]*sizeof(int)); slot[e]=malloc(cnt[e]*sizeof(int)); }
        for (int64_t t=0;t<n;t++) for (int s=0;s<TOPK;s++) {
            int e=(int)id[t*TOPK+s]; if(e<0||e>=NEXP) continue;
            rows[e][fill[e]]=(int)t; slot[e][fill[e]]=s; fill[e]++;
        }
        int used=0;
        for (int e=0;e<NEXP;e++) {
            if (!cnt[e]) continue;
            char n1[192],n2[192],n3[192]; long r1,cc1,r2,cc2,r3,cc3;
            snprintf(n1,sizeof n1,"layers.%d.ffn.experts.%d.w1.weight",L,e);
            snprintf(n2,sizeof n2,"layers.%d.ffn.experts.%d.w3.weight",L,e);
            snprintf(n3,sizeof n3,"layers.%d.ffn.experts.%d.w2.weight",L,e);
            float *g1=st_read_weight(&C,n1,&r1,&cc1), *u1=st_read_weight(&C,n2,&r2,&cc2),
                  *d1=st_read_weight(&C,n3,&r3,&cc3);
            if (!g1||!u1||!d1) { fprintf(stderr,"L%d e%d: HF 权重读不到\n",L,e); return 3; }
            pthread_t th[64]; work_t wk[64]; int per=(cnt[e]+nth-1)/nth, nt=0;
            for (int t2=0;t2<nth;t2++) {
                int a=t2*per, b=a+per>cnt[e]?cnt[e]:a+per; if(a>=b) break;
                wk[nt]=(work_t){x,rw,id,g1,u1,d1,acc,rows[e],cnt[e],0,slot[e],swlim,a,b};
                if (pthread_create(&th[nt],NULL,worker,&wk[nt])!=0) worker(&wk[nt]); else nt++;
            }
            for (int t2=0;t2<nt;t2++) pthread_join(th[t2],NULL);
            free(g1);free(u1);free(d1); used++;
        }
        snprintf(p,sizeof p,"%s/routed_L%d.npy",cap,L);
        if (write_npy_f32(p,acc,n,DM)!=0) { fprintf(stderr,"L%d 写失败\n",L); return 3; }
        printf("L%-3d n=%lld 专家%d/%d → routed_L%d.npy\n",L,(long long)n,used,NEXP,L);
        fflush(stdout);
        for (int e=0;e<NEXP;e++){ free(rows[e]); free(slot[e]); }
        free(rows);free(slot);free(fill);free(cnt);free(acc);free(x);free(id);free(rw);
    }
    return 0;
}
