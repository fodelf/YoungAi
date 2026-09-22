/* v41_amp_run.c — V4.1 反修驱动: 在 ds4 引擎的真前向上逐层【取 → 扫 → 选 → 解】(2026-09-13), 零 Python。
 *
 * 【为什么要这个】解算循环必须跑在部署那条前向上: 引擎钩子 ds4_engine_v41_set_moe_hook 在每层 MoE 出口回调, 本程序拿到
 * 部署态 x_q / y_q / 路由, 用 HF 出厂权重算 y_fp(x_q)(v41_fp_moe.cu), 靶 = y_fp − y_q, 解算走 v41_amp_solve.cu。
 * (09-12 那版挂在 torch 前向上, 解算看到的 x 是 torch 模拟量化传播的, 与部署不是同一笔账: 同档对拍 KL 0.007。)
 *
 * 【多遍序贯】第 k 遍: 引擎挂着本目录里已解的 L0..L(k−1), 按 512 分块正常前向, 每块到第 k 层取料即停; 收齐全部行后解第 k 层
 * 落盘; 末遍全挂跑完整前向打拟合料 PPL。与"一次前向逐层解+挂"数学等价。不整批一次前向的原因: 8192 整批的引擎缓冲把
 * spark avail 压到 6 GB(看门狗线以下)且 L02 前向失败; 多遍用判决同一套 512 分块, 内存账已验证。
 *
 * 【择优规则】held-out 按 <ids>.layout 窗分层(每域窗号 k%4==3 作 val), K∈{64,128,256,512} × λ∈{1,3,10,30,100} 只认 val;
 * val ≥ 0.9×最优 的格里取最小 K、同 K 最大 λ; 层最优 val < 0.5% 不挂。产物 amp_Lnn.bin + manifest.txt。
 *
 * 用法(三条路, 见 main 的 usage): 反修逐层 / 反修-蒸馏靶(--target kl:) / 后训练第三件(--sft-list, 见 v41_sft_run.inc.c)。
 * 编译: make -C gguf-tools v41_amp_run(Linux+CUDA; 要先在仓库根 make cuda-spark 出引擎对象) */
#define _GNU_SOURCE
#include "../../src/common/ds4_st41.h"
#include "../../src/common/ds4_quantfmt.h"
#include "../../src/common/ds4_float.h"
#include "../../ds4.h"
#include "../../ds4_gpu.h"               /* ds4_gpu_set_model_cache_limit_mb(设备权重缓存封顶, --weight-cache-mb) */
#include "v41_fp_moe.h"
#include "v41_kl_target.h"
#include "v41_route_solve.h"             /* 路由偏置侧车的 GPU 数值件(v41_rb_run.inc.c 驱动) */
#include "../../src/common/ds4_gguf.h"    /* 路由侧车要从 GGUF 取路由张量(与引擎同一份字节) */
#include "../../src/common/ds4_etgd.h"   /* --score-topk 产物的唯一读取实现(引擎写/解算器读/判决器读) */
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>

/* v41_amp_solve.cu 的三个入口(Python 路经 libv41amp.so 调的也是这三个) */
int v41_amp_scan_k_gpu(const void *dX, const void *dYfp, const void *dYq, int N, int D, int NFIT, const int *Ks, int nk,
                       const float *lams, int nl, float *out_train, float *out_val, float *out_tgt_rel, int whiten,
                       float *out_train_w, float *out_val_w);
int v41_amp_solve_layer_gpu(const void *dX, const void *dYfp, const void *dYq, int N, int D, int K, float lam,
                            void *dA, void *dB, float *out_ratio, int whiten);
int v41_amp_apply_gpu(const void *dX, void *dY, const void *dA, const void *dB, int N, int D, int K, float *out_dz);
/* v41_gr_solve.cu: 权重侧逐专家逐通道增益(down 行增益)重解 */
int v41_gr_solve_layer_gpu(const float *dye, const float *drw, const int *dsel, const float *dyfp, const float *dysh,
                           int n, int nfit, int nu, int D, int n_expert, float lam, int niter, float *ds, uint8_t *dpk,
                           float *out_train, float *out_val);

#define USAGE "用法: v41_amp_run <model.gguf> <hf-dir> <ids> <ntok上限> <出目录> [--layers N] [--whiten 0|1|2]\n" \
              "      [--mem-budget-mb M] [--weight-cache-mb M] [--no-engram]\n" \
              "  路由侧车: --rb <教师 --dump-moe 目录> [--base-amp <现役侧车>](一趟统计全部层的 Δb, 落 rb_Lnn.bin; 见 v41_rb_run.inc.c)\n" \
              "  合并序贯: --rb <dump> --capture-ye  (每层先解 Δb 挂上, 再取料解增益; rb_Lnn + gr_Lnn 同目录)\n" \
              "  蒸馏靶:   --only-layer N --target kl:<教师锚> --eta-rel r\n" \
              "  后训练:   --sft-list <清单> --only-layer 39 --capture-ye --base-amp <②> [--share]\n" \
              "            [--reuse-tables] [--tau-list 1] [--rho-list 0.1,1,10] [--lam-list 0.3,1,3,10]\n" \
              "            [--kappa-list 1e9](专家频率岭) [--nu-list 0](词对重复度权重) [--gates R](x 键控, 1=旧形态)\n" \
              "  验证表:   --sft-list <清单> --verify-tabs g_ --base-pt <候选>(一次加载跑完所有样本)\n" \
              "  校准取料: --dump-calib  (裸 ① 上一趟, 40 层同趟落 <出目录>/calib_Lnn.bin, 给 v41_quantize --calib; 见 v41_calib_dump.inc.c)\n"

#define NK 4
#define NL 5
static const int SEL_KS[NK] = {64, 128, 256, 512};
static const float SEL_LAMS[NL] = {1.f, 3.f, 10.f, 30.f, 100.f};
#define GATE_PCT 0.5f        /* 层最优 val 低于此(百分点)不挂 */
#define KEEP 0.9f            /* 择优: val ≥ KEEP×最优 的格里挑最便宜的 */
#define MAXK 512
#define MAXU 16              /* n_used 上限(V4.1 = 6) */

typedef struct {
    v41_st S;                              /* HF 出厂权重索引 */
    int *perm, nfit, n, D, MID, n_expert, n_used;   /* held-out 行置换(拟合行在前) / 形状 */
    FILE *mf; const char *out_dir;
    int layer, whiten;                     /* 本遍取第几层 */
    float clamp;
    float *rx, *ry, *rrw, *ralpha; int *rsel; int got;   /* 本遍原序取料 [n][D] / [n][n_used] / [n], got = 已收到的行数 */
    float *rye, *rysh; int want_ye;        /* --capture-ye: 逐专家 down 输出 [n][n_used][D](未乘 rw) + shared 输出 [n][D] */
    float *hye, *hysh, *hgr;               /* 置换后副本 + 增益缩放因子下载 [n_expert][D] */
    float *dye, *dysh, *drw, *dgr; int *dsel;   /* 权重侧解算的设备缓冲 */
    uint8_t *dgrpk, *hgrpk;                /* ★解算器直接产出的 fp4x32 码字★ [n_expert][D/32][17]: 落盘写的就是它 */
    float *dX, *dYq, *dYfp, *dYtmp, *dA, *dB;     /* 设备 */
    float *hA, *hB, *hx, *hy, *hrw, *halpha; int *hsel;   /* 主机: 放大器下载 / 置换后副本 */
    uint8_t *pkA, *pkB;                    /* fp4x32 码字(落盘的就是它) */
    float *hvf, *hvt;                      /* val 行的 y_fp / 工作副本: fp4 往返后重算 val 用 */
    v41_fp_ffn *ex, sh;
    /* 蒸馏靶(--target kl): 靶不再是"这一层像教师", 而是"让最终 logits 像教师该往哪挪"。见 v41_kl_target.h */
    v41_klt *klt; const char *kl_ref, *kl_stu; float eta_rel;
    /* 后训练第三件(2026-09-13 夜第二版): 靶 = 决策点上两个 token 的 logit 差(第一版拟合损失梯度已判死) */
    const char *sft_list, *base_amp, *base_pt;
    const char *tau_s, *rho_s, *lam_s;     /* 决策差网格: τ=目标 logit 差 / ρ=约束行权重 / λ=岭 */
    const char *kap_s, *nu_s; float kappa;  /* 泛化先验(back.md §4.7): κ=专家频率岭 / ν=词对重复度权重 */
    int nsmp;                              /* 样本条数: 尺 L 按条留一, 折数 = 它 */
    int reuse_tab; const char *verify_pref;   /* 榜单已在盘上就取完料停车 / 挂候选只出表不解算(gatea 用) */
    int share; float *dgrf;                /* --share: 全体专家共用一组通道增益(未知数 D 个); dgrf = 广播+量化后的落地态 */
    const char *gate_s; int gates; float *dgate;   /* --gates 列表: x 键控(R=1 = 旧形态; 一次取料跑多个 R);
                                                    * gates = 当前这一档 R; dgate[n][R] = 每行每门的 σ 标量 */
    int nsel;                              /* 解算真正用的行数(fit+val); 取料仍是全部 n 行 */
    const char *rb_dump; ds4_gguf gq;      /* 路由偏置侧车(v41_rb_run.inc.c): 教师 dump 目录 / GGUF(取路由张量) */
    int ok_layers, skip_layers, kcount[NK]; double val_sum, dz_sum, t_fp, t_scan, t_solve, t_fwd;
} ctx_t;

/* RNE 舍到 bf16 再回 f32 —— 与引擎 v41_bf16r(cuda_v41_1.inc.cu)逐位同式。取料自检要重建引擎的
 * y = round_bf16(routed + shared), 舍入口径差一位就对不上。 */
static float bf16r(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    if ((u & 0x7F800000u) == 0x7F800000u) return x;
    u += 0x7FFFu + ((u >> 16) & 1u); u &= 0xFFFF0000u;
    float y; memcpy(&y, &u, 4); return y;
}

static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "★CUDA %s @%d: %s★\n", #x, __LINE__, cudaGetErrorString(e_)); return -1; } } while (0)

/* <ids>.layout 的窗分层 held-out(= v41_amp_split.layout_split): 每域内窗号 k%4==3 的窗作 val, 其余拟合; 不满一窗的尾巴归拟合。
 * 为什么不前缀切: idshalf 把各域连续铺进 8192 行, 前缀切的 val 全是最后一个域而拟合段没见过它 —— 量的是跨域泛化, 假账。 */
static int layout_split(const char *ids_path, int ntok, ctx_t *c) {
    char p[4300]; snprintf(p, sizeof p, "%s.layout", ids_path);
    FILE *f = fopen(p, "r");
    if (!f) { fprintf(stderr, "★%s 缺: 择优路要 layout 分层 held-out★\n", p); return -1; }
    int win = 128, nf = 0, nv = 0;
    int *fit = malloc(sizeof(int) * (size_t)ntok), *val = malloc(sizeof(int) * (size_t)ntok);
    char line[512], desc[2048] = "";
    while (fgets(line, sizeof line, f)) {
        char dom[128]; int off, cnt;
        if (line[0] == '#' || line[0] == '\n') continue;
        if (sscanf(line, "win %d", &win) == 1) continue;
        if (sscanf(line, "%127s %d %d", dom, &off, &cnt) != 3) continue;
        const int nw = cnt / win; int dv = 0;
        for (int k = 0; k < nw; k++) {
            const int isval = (k % 4 == 3); if (isval) dv++;
            for (int r = off + k * win; r < off + (k + 1) * win; r++) { if (r >= ntok) continue; if (isval) val[nv++] = r; else fit[nf++] = r; }
        }
        for (int r = off + nw * win; r < off + cnt; r++) if (r < ntok) fit[nf++] = r;
        char d1[192]; snprintf(d1, sizeof d1, "%s%s %d+%d窗", desc[0] ? ", " : "", dom, nw - dv, dv); strncat(desc, d1, sizeof desc - strlen(desc) - 1);
    }
    fclose(f);
    if (nf + nv != ntok) { fprintf(stderr, "★layout 覆盖 %d 行 ≠ ntok %d★\n", nf + nv, ntok); return -1; }
    c->perm = malloc(sizeof(int) * (size_t)ntok);
    memcpy(c->perm, fit, sizeof(int) * (size_t)nf); memcpy(c->perm + nf, val, sizeof(int) * (size_t)nv);
    c->nfit = nf; c->n = ntok;
    free(fit); free(val);
    printf("[held-out·分层] 窗宽 %d: 拟合 %d / val %d 行 (%s)\n", win, nf, nv, desc);
    return 0;
}

static int resolve_layer(ctx_t *c, int il);   /* v41_fp_bind.inc.c: HF 出厂权重索引(反修 FP 靶用) */

/* 择优(= v41_amp_hooks._choose): 百分比表 pct[NL][NK], val ≥ KEEP×最优 的格里取最小 K, 同 K 取最大 λ。返回 0 = 没有可选格 */
static int choose(const float pct[NL][NK], float *best, int *ki, int *lj) {
    *best = pct[0][0];
    for (int l = 0; l < NL; l++) for (int k = 0; k < NK; k++) if (pct[l][k] > *best) *best = pct[l][k];
    if (*best <= 0.f) return 0;
    *ki = -1; *lj = -1;
    for (int k = 0; k < NK && *ki < 0; k++) for (int l = 0; l < NL; l++) if (pct[l][k] >= KEEP * *best) { *ki = k; break; }
    for (int l = NL - 1; l >= 0; l--) if (pct[l][*ki] >= KEEP * *best) { *lj = l; break; }
    return 1;
}

/* 取料钩子: 只认本遍的层, 原序收进 rx/ry/rsel/rrw, 收完本块即让引擎停下(返回 1) */
static int hook(void *ud, int il, int pos0, int n, int D, int n_used, float clamp, const float *x, const float *y, const int *sel, const float *rw, const float *alpha, const float *ye, const float *ysh) {
    ctx_t *c = ud;
    if (il != c->layer) return 0;
    if (D != c->D || n_used > MAXU || pos0 < 0 || pos0 + n > c->n) { fprintf(stderr, "★钩子 pos0=%d n=%d D=%d n_used=%d 与拟合 %d/%d 不符★\n", pos0, n, D, n_used, c->n, c->D); return -1; }
    c->n_used = n_used; c->clamp = clamp;
    memcpy(c->rx + (size_t)pos0 * D, x, (size_t)n * D * 4); memcpy(c->ry + (size_t)pos0 * D, y, (size_t)n * D * 4);
    memcpy(c->rsel + (size_t)pos0 * n_used, sel, (size_t)n * n_used * 4); memcpy(c->rrw + (size_t)pos0 * n_used, rw, (size_t)n * n_used * 4);
    memcpy(c->ralpha + (size_t)pos0, alpha, (size_t)n * 4);
    if (c->want_ye) {
        /* 取不到就硬停: 逐专家输出只有 prefill GEMM 路物化, 拿不到说明走了解码路或形状对不上 ——
         * 这时候继续跑会解出一个没有料支撑的东西, 比报错难查得多。 */
        if (!ye || !ysh) { fprintf(stderr, "★L%02d 拿不到逐专家 down 输出/shared(ye=%p ysh=%p): 块 n=%d 可能走了解码 gemv 路★\n", il, (const void *)ye, (const void *)ysh, n); return -1; }
        memcpy(c->rye + (size_t)pos0 * n_used * D, ye, (size_t)n * n_used * D * 4);
        memcpy(c->rysh + (size_t)pos0 * D, ysh, (size_t)n * D * 4);
    }
    c->got += n;
    /* ★取完本层要不要继续跑到出口★
     * 蒸馏靶要同一趟的学生 logits, 必须跑完。后训练也要(榜单 top-K 与 rms 就在出口那一步算),
     * ★两样本来就在同一趟里★ —— 第一版让它半路停车、另跑一趟专门打分, 于是同一批 token 前向了
     * 两遍, 8 条样本白烧 22 分钟, 外加 8 次独立进程各加载一遍模型(107s×8 ≈ 14 分钟)。
     * 当时停车的理由是"跑到出口会写一份全词表 logits"(517 KB/位置, 统一内存上写它=掏 GPU 内存),
     * 但现在有 --score-no-logits + --score-topk/--score-nll/--score-rms 三个小出口, 出口那一段
     * 只落几十 MB。所以只在【表已经在盘上】(c->reuse_tab)时才提前停车。 */
    return (c->klt && !c->reuse_tab) ? 0 : 1;
}

static int solve_layer(ctx_t *c, int il);                            /* v41_amp_lowrank.inc.c(低秩放大器路) */
static int solve_layer_gr(ctx_t *c, int il, double t0, double t1);   /* v41_gr_run.inc.c(权重侧逐专家增益) */
static int rb_fit(ctx_t *c, ds4_engine *e, const int *ids, int ntok, const char *lp, int no_engram,
                  const char *dump, const char *base_amp, int nl);   /* v41_rb_run.inc.c(路由偏置侧车, 一趟模式) */
static int rb_seq_init(ctx_t *c, int ntok, int nl);                  /* 同上, 合并序贯: 统计表分配 / 本层解 Δb / 双职钩子 */
static int rb_seq_solve(ctx_t *c, int il, int *nw);
static void rb_seq_free(void);
static int rbgr_hook(void *ud, int il, int pos0, int n, int D, int n_used, float clamp, const float *x, const float *y, const int *sel,
                     const float *rw, const float *alpha, const float *ye, const float *ysh);
static int sft_run(ctx_t *c, ds4_engine *e, const char *list_path, const char *out, int il, int ntok_cap, int no_engram);  /* v41_sft_run.inc.c */
static int *rows_spec_parse(const char *spec, int ntok, int *out_n); /* v41_sft_run.inc.c(行子集) */
static int calib_dump(ds4_engine *e, const int *ids, int ntok, const char *dir, int nl, int no_engram, int D, int n_expert,
                      const char *gguf, const char *ids_path);   /* v41_calib_dump.inc.c(量化校准取料, --dump-calib) */


int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "%s", USAGE); return 2; }
    const char *gguf = argv[1], *hf = argv[2], *ids_path = argv[3], *out = argv[5];
    const int ntok = atoi(argv[4]);
    ctx_t c; memset(&c, 0, sizeof c);
    c.out_dir = out;
    int mem_mb = 0, cache_mb = 0, no_engram = 0, nlayers = 40, only_layer = -1, dump_calib = 0;
    const char *fit_spec = NULL, *val_spec = NULL;
    c.eta_rel = 0.10f;
    /* 网格默认值: τ=1 nat(p_对/p_错 ≥ e, 给 bf16/rms 的二阶误差留余量), ρ 三档看"别处别动"多贵,
     * λ 四档同反修量纲。12 个候选, 秒级一个 —— 贵的是取料那 21 分钟, 不是解算。 */
    c.tau_s = "1"; c.rho_s = "0.1,1,10"; c.lam_s = "0.3,1,3,10";
    /* κ=1e9 等于"不加频率岭"(退化成第二版), 留着当对照组; ν=0 同理 = 不看词对重复度。 */
    c.kap_s = "1e9"; c.nu_s = "0"; c.gate_s = "1";
    for (int i = 6; i < argc; i++) {
        if (!strcmp(argv[i], "--layers") && i + 1 < argc) nlayers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--whiten") && i + 1 < argc) c.whiten = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mem-budget-mb") && i + 1 < argc) mem_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--weight-cache-mb") && i + 1 < argc) cache_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-engram")) no_engram = 1;
        else if (!strcmp(argv[i], "--only-layer") && i + 1 < argc) only_layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--capture-ye")) c.want_ye = 1;
        else if (!strcmp(argv[i], "--dump-calib")) dump_calib = 1;
        else if (!strcmp(argv[i], "--rb") && i + 1 < argc) c.rb_dump = argv[++i];
        else if (!strcmp(argv[i], "--eta-rel") && i + 1 < argc) c.eta_rel = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--sft-list") && i + 1 < argc) c.sft_list = argv[++i];
        else if (!strcmp(argv[i], "--reuse-tables")) c.reuse_tab = 1;
        else if (!strcmp(argv[i], "--share")) c.share = 1;
        else if (!strcmp(argv[i], "--verify-tabs") && i + 1 < argc) c.verify_pref = argv[++i];
        else if (!strcmp(argv[i], "--tau-list") && i + 1 < argc) c.tau_s = argv[++i];
        else if (!strcmp(argv[i], "--rho-list") && i + 1 < argc) c.rho_s = argv[++i];
        else if (!strcmp(argv[i], "--lam-list") && i + 1 < argc) c.lam_s = argv[++i];
        else if (!strcmp(argv[i], "--kappa-list") && i + 1 < argc) c.kap_s = argv[++i];
        else if (!strcmp(argv[i], "--nu-list") && i + 1 < argc) c.nu_s = argv[++i];
        else if (!strcmp(argv[i], "--gates") && i + 1 < argc) c.gate_s = argv[++i];
        else if (!strcmp(argv[i], "--base-amp") && i + 1 < argc) c.base_amp = argv[++i];
        else if (!strcmp(argv[i], "--base-pt") && i + 1 < argc) c.base_pt = argv[++i];
        else if (!strcmp(argv[i], "--fit-rows") && i + 1 < argc) fit_spec = argv[++i];
        else if (!strcmp(argv[i], "--val-rows") && i + 1 < argc) val_spec = argv[++i];
        else if (!strcmp(argv[i], "--target") && i + 1 < argc) {   /* --target kl:<教师锚> */
            const char *t = argv[++i];
            if (strncmp(t, "kl:", 3)) { fprintf(stderr, "★--target 只认 kl:<教师锚.bin>★\n"); return 2; }
            c.kl_ref = t + 3;
        } else { fprintf(stderr, "★不认识的参数 %s★\n", argv[i]); return 2; }
    }
    if (c.kl_ref && only_layer < 0) { fprintf(stderr, "★蒸馏靶只对末层成立(梯度链止于出口), 必须配 --only-layer <末层号>★\n"); return 2; }
    if (c.sft_list) {
        /* 后训练靶与蒸馏靶同一条闭式链, 同样只对末层精确 —— 往前的层要穿过后面各层的雅可比,
         * 那是 back.md §3.3 的级 2/级 3, 不在这条路上偷偷做。 */
        /* 验证趟只跑前向出表, 不取料不解算 ⇒ 不要求 --only-layer/--capture-ye, 也不用出口头 */
        if (!c.verify_pref) {
            if (only_layer < 0) { fprintf(stderr, "★后训练靶只对末层精确, 必须配 --only-layer <末层号>★\n"); return 2; }
            if (!c.want_ye) { fprintf(stderr, "★后训练解的是逐专家增益, 必须配 --capture-ye★\n"); return 2; }
        }
        if (c.kl_ref) { fprintf(stderr, "★--target kl 与 --sft-list 是两个靶, 只能给一个★\n"); return 2; }
    }
    if (c.eta_rel <= 0.f || c.eta_rel > 2.f) { fprintf(stderr, "★--eta-rel %g 不合理★\n", c.eta_rel); return 2; }
    if (c.rb_dump && (c.sft_list || c.kl_ref)) { fprintf(stderr, "★--rb 不与 --sft-list/--target 同给(配 --capture-ye = 路由+增益合并序贯)★\n"); return 2; }
    const int merged = c.rb_dump && c.want_ye;   /* 合并序贯: 每层 Δb → 挂上 → 增益 */
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (ntok < 2 || nlayers < 1 || nlayers > 64) { fprintf(stderr, "★ntok %d / layers %d★\n", ntok, nlayers); return 2; }
    /* 清单模式(后训练)自己按条读 ids, 这里的 <ids> 只是个占位; 行分层也由清单的 fit/val 列给,
     * 不走 .layout 窗分层 —— 两套分层同时生效只会打架, 而打架的结果在读数里看不出来。 */
    int *ids = NULL, n = 0, t;
    if (!c.sft_list) {
        ids = malloc(sizeof(int) * (size_t)ntok);
        FILE *fi = fopen(ids_path, "r");
        if (!fi) { fprintf(stderr, "★打不开 %s★\n", ids_path); return 1; }
        while (n < ntok && fscanf(fi, "%d", &t) == 1) ids[n++] = t;
        fclose(fi);
        if (n != ntok) { fprintf(stderr, "★%s 只有 %d 个 id < ntok %d★\n", ids_path, n, ntok); return 1; }
    }
    /* ★行子集(后训练用)★: 报告段之外的行(当日材料)不该进解算 —— 它们不是"该写出来的东西",
     * 梯度在那里没有意义。--fit-rows/--val-rows 直接给绝对行号区间(与判决器 --rows 同语法),
     * 不给就退回 layout 窗分层(反修那条路的老规矩)。两者只能二选一, 免得两套分层打架。 */
    if (fit_spec || val_spec) {
        if (!fit_spec || !val_spec) { fprintf(stderr, "★--fit-rows 与 --val-rows 必须成对给★\n"); return 2; }
        int nf = 0, nv = 0;
        int *fit = rows_spec_parse(fit_spec, ntok, &nf), *val = rows_spec_parse(val_spec, ntok, &nv);
        if (!fit || !val) { fprintf(stderr, "★--fit-rows/--val-rows 要 a:b 或 a:b,c:d,...★\n"); return 2; }
        c.perm = malloc(sizeof(int) * (size_t)(nf + nv));
        memcpy(c.perm, fit, sizeof(int) * (size_t)nf);
        memcpy(c.perm + nf, val, sizeof(int) * (size_t)nv);
        c.nfit = nf; c.n = ntok; c.nsel = nf + nv;
        free(fit); free(val);
        printf("[行子集] 取料 %d 行, 解算用 %d 行(拟合 %d / val %d)\n", ntok, c.nsel, nf, nv);
    } else if (c.sft_list) {
        c.n = ntok; c.nfit = ntok;   /* 真正的行数与分层在 sft_run 里按清单定 */
    } else if (layout_split(ids_path, ntok, &c)) return 1;
    if (mkdir(out, 0775) && errno != EEXIST) { fprintf(stderr, "★建不了 %s★\n", out); return 1; }
    /* HF 出厂索引 + 形状(专家数/MID/D 从张量名与形状读, 不写死) */
    if (v41_st_open(&c.S, hf)) return 1;
    for (int i = 0; i < c.S.ne; i++) { int L, e; if (sscanf(c.S.e[i].name, "layers.%d.ffn.experts.%d.w1.weight", &L, &e) == 2 && L == 0 && e + 1 > c.n_expert) c.n_expert = e + 1; }
    const v41_st_ent *w1 = v41_st_find(&c.S, "layers.0.ffn.experts.0.w1.weight");
    if (!c.n_expert || !w1 || w1->nd != 2) { fprintf(stderr, "★%s 不像 V4.1 HF 目录(没有 layers.0.ffn.experts.*)★\n", hf); return 1; }
    c.MID = (int)w1->shape[0]; c.D = (int)w1->shape[1] * 2;
    if (only_layer >= 0) printf("[反修·引擎路] HF %d 专家 × [%d×%d], 拟合料 %s n=%d, ★只解 L%02d★, 白化 %d → %s\n", c.n_expert, c.MID, c.D, ids_path, ntok, only_layer, c.whiten, out);
    else printf("[反修·引擎路] HF %d 专家 × [%d×%d], 拟合料 %s n=%d, 前 %d 层, 白化 %d → %s\n", c.n_expert, c.MID, c.D, ids_path, ntok, nlayers, c.whiten, out);
    c.ex = calloc((size_t)c.n_expert, sizeof(v41_fp_ffn));
    const size_t nD = (size_t)ntok * c.D, nU = (size_t)ntok * MAXU;
    c.rx = malloc(nD * 4); c.ry = malloc(nD * 4); c.rrw = malloc(nU * 4); c.rsel = malloc(nU * 4); c.ralpha = malloc((size_t)ntok * 4);
    c.hx = malloc(nD * 4); c.hy = malloc(nD * 4); c.hrw = malloc(nU * 4); c.hsel = malloc(nU * 4); c.halpha = malloc((size_t)ntok * 4);
    c.hA = malloc((size_t)MAXK * c.D * 4); c.hB = malloc((size_t)MAXK * c.D * 4);
    c.pkA = malloc((size_t)MAXK * c.D / 32u * 17u); c.pkB = malloc((size_t)MAXK * c.D / 32u * 17u);
    c.hvf = malloc((size_t)(ntok - c.nfit + 1) * c.D * 4); c.hvt = malloc((size_t)(ntok - c.nfit + 1) * c.D * 4);
    if (!c.pkA || !c.pkB || !c.hvf || !c.hvt) { fprintf(stderr, "★fp4/val 缓冲分配失败★\n"); return 1; }
    if (c.want_ye) {   /* [n][n_used][D] f32: 8192×6×5120×4 = 1.0 GB */
        const size_t nyD = nU * (size_t)c.D, nsD = (size_t)c.n_expert * c.D;
        c.rye = malloc(nyD * 4); c.rysh = malloc(nD * 4);
        c.hye = malloc(nyD * 4); c.hysh = malloc(nD * 4); c.hgr = malloc(nsD * 4);
        c.hgrpk = (uint8_t *)malloc(nsD / 32u * 17u);
        if (!c.rye || !c.rysh || !c.hye || !c.hysh || !c.hgr || !c.hgrpk) {
            fprintf(stderr, "★逐专家输出缓冲(2×%.1f GB)分配失败★\n", (double)nyD * 4 / 1e9); return 1; }
        if (cudaMalloc((void **)&c.dye, nyD * 4) || cudaMalloc((void **)&c.dysh, nD * 4) ||
            cudaMalloc((void **)&c.drw, nU * 4) || cudaMalloc((void **)&c.dsel, nU * 4) ||
            cudaMalloc((void **)&c.dgr, nsD * 4) ||
            cudaMalloc((void **)&c.dgrpk, nsD / 32u * 17u)) { fprintf(stderr, "★权重侧解算设备缓冲分配失败★\n"); return 1; }
    }
    if (cudaMalloc((void **)&c.dX, nD * 4) || cudaMalloc((void **)&c.dYq, nD * 4) || cudaMalloc((void **)&c.dYfp, nD * 4) ||
        cudaMalloc((void **)&c.dYtmp, nD * 4) || cudaMalloc((void **)&c.dA, (size_t)MAXK * c.D * 4) || cudaMalloc((void **)&c.dB, (size_t)MAXK * c.D * 4)) {
        fprintf(stderr, "★设备缓冲分配失败★\n"); return 1;
    }
    char mp[4300]; snprintf(mp, sizeof mp, "%s/manifest.txt", out);
    c.mf = fopen(mp, "w");
    if (!c.mf) { fprintf(stderr, "★写不了 %s★\n", mp); return 1; }
    fprintf(c.mf, "# 层 λ K 靶/‖y_q‖ train%% val%% ‖Δ‖/‖y_q‖ [val_w%%]  (val<gate 的层记 skip; 数值来自 C 库; 百分比=1−‖残差‖/‖靶‖ 幅值比)\n"
                  "# 候选 K=[64, 128, 256, 512] λ=[1, 3, 10, 30, 100] gate=%.1f%% keep=%.1f whiten=%d%s\n"
                  "# 解算前向 = ds4 引擎(部署同路, v41_amp_run 多遍序贯) gguf=%s 拟合=%s n=%d\n",
            GATE_PCT, KEEP, c.whiten, c.whiten ? "(择优看白化 val)" : "", gguf, ids_path, ntok);
    fflush(c.mf);
    /* 引擎: 与 CLI 同款打开(CUDA 后端) */
    if (mem_mb > 0) ds4_set_mem_budget_mb(mem_mb);
    /* 拟合工作区 ~17 GiB(09-20 实测反推)叠在整模型设备缓存之上, 105.87 GiB 的模型在 121 GiB 机器上装不下
     * ⇒ 封顶缓存, 装不下的层走逐层流式(见 ds4_gpu_core.h)。必须在引擎打开(注册映射)之前设。 */
    if (cache_mb > 0) ds4_gpu_set_model_cache_limit_mb((uint64_t)cache_mb);
    ds4_engine_options opt; memset(&opt, 0, sizeof opt);
    opt.model_path = gguf; opt.backend = DS4_BACKEND_CUDA;
    ds4_engine *e = NULL;
    const double t0 = now_s();
    if (ds4_engine_open(&e, &opt) != 0 || !e) { fprintf(stderr, "★引擎打不开 %s★\n", gguf); return 1; }
    if (!ds4_engine_is_v41(e)) { fprintf(stderr, "★%s 不是 V4.1 模型★\n", gguf); return 1; }
    printf("[引擎] 打开 %.0fs; 多遍序贯反修开始(每遍取一层)\n", now_s() - t0);
    char lp[4300]; snprintf(lp, sizeof lp, "%s/fit_logits.bin", out);
    int rc = 0;
    if (dump_calib) {   /* 量化校准取料: 40 层同趟落 x/路由, 不解算(v41_calib_dump.inc.c) */
        const int drc = calib_dump(e, ids, ntok, out, nlayers, no_engram, c.D, c.n_expert, gguf, ids_path);
        fclose(c.mf); unlink(mp);   /* 取料目录不该有 manifest(那是反修产物的账) */
        ds4_engine_close(e); v41_st_close(&c.S);
        return drc == 0 ? 0 : 1;
    }
    if (c.rb_dump) {   /* 路由侧车(两种模式都要): 路由张量从 GGUF 取, 与引擎同一份字节 */
        char err[256];
        if (ds4_gguf_open(&c.gq, gguf, err, sizeof err)) { fprintf(stderr, "★路由侧车要读 GGUF 的路由张量, 打不开 %s: %s★\n", gguf, err); return 1; }
    }
    if (c.rb_dump && !merged) {   /* 路由偏置侧车一趟模式: 一趟前向统计全部层的 Δb, 落 rb_Lnn.bin(与 gr 同目录部署) */
        const int prc = rb_fit(&c, e, ids, ntok, lp, no_engram, c.rb_dump, c.base_amp, nlayers);
        if (prc != 0) fprintf(c.mf, "# 失败 路由侧车\n");
        fclose(c.mf);
        ds4_gguf_close(&c.gq); ds4_engine_close(e); v41_st_close(&c.S);
        return prc == 0 ? 0 : 1;
    }
    if ((c.kl_ref || c.sft_list) && !c.verify_pref) {   /* 出口头 + output_norm 从同一个 GGUF 取(学生自己的出口) */
        if (v41_klt_open(&c.klt, gguf, c.D)) { fprintf(stderr, "★出口靶初始化失败★\n"); return 1; }
        c.kl_stu = lp;
        if (c.sft_list) printf("[后训练] 样本清单 %s; 网格 τ=%s ρ=%s λ=%s; 底座 ②=%s ③=%s\n",
                               c.sft_list, c.tau_s, c.rho_s, c.lam_s, c.base_amp ? c.base_amp : "(无)", c.base_pt ? c.base_pt : "(无)");
        else printf("[蒸馏靶] 教师锚 %s; 学生 logits = 本趟前向落盘 %s; --eta-rel %.3f\n", c.kl_ref, lp, c.eta_rel);
    }
    if (c.sft_list) {   /* 后训练: 多条样本各取一趟料, 累积后一次解(v41_sft_run.inc.c) */
        const int src = sft_run(&c, e, c.sft_list, out, only_layer, ntok, no_engram);
        fprintf(c.mf, "# 后训练 %s\n", src == 0 ? "完成" : "失败");
        fclose(c.mf);
        if (c.klt) v41_klt_close(c.klt);
        ds4_engine_close(e);
        v41_st_close(&c.S);
        return src == 0 ? 0 : 1;
    }
    const int il_lo = only_layer >= 0 ? only_layer : 0, il_hi = only_layer >= 0 ? only_layer + 1 : nlayers;
    int nrb = 0;   /* 合并序贯落盘的 rb 层数: 目录里有它也算"有插件", 下一遍必须挂 */
    if (merged) {
        if (rb_seq_init(&c, ntok, nlayers)) { fprintf(stderr, "★路由侧车统计表分配失败★\n"); return 1; }
        printf("[合并序贯] 每层: 路由统计+增益取料 同一遍 → 解 Δb(严格涨才落盘) → 落了盘就挂上重取一遍 → 解增益; FP 路由 ← %s\n", c.rb_dump);
    }
    for (int il = il_lo; il < il_hi && rc == 0; il++) {   /* 第 il 遍: 挂 L0..L(il−1) 已解的放大器, 到 L(il) 取料即停 */
        /* ★取料必须在"要部署的那个态"上★: 后训练解的是 ①+②(+③上一版)之上的增量, 所以取料遍挂的是
         * --base-amp/--base-pt, 不是本次的输出目录(输出目录里此刻还什么都没有)。反修序贯那条路仍
         * 挂自己的输出目录(第 k 遍挂 L0..L(k−1))。目录里一个插件都没有时引擎会拒开状态, 所以没解出过就不挂。
         * 合并序贯时第二趟(pass=1)只在本层 Δb 刚落盘后才跑: 路由变了, 增益必须解在挂着 Δb 的取料上, 上一趟的 ye 是旧路由的。 */
        for (int pass = 0; pass < 2 && rc == 0; pass++) {
            c.layer = il; c.got = 0;
            ds4_engine_v41_set_amp_dir((c.ok_layers || nrb) ? out : NULL);
            ds4_engine_v41_set_moe_hook((merged && pass == 0) ? rbgr_hook : hook, &c);
            ds4_engine_v41_set_moe_hook_layer(il);   /* 只要这一层, 省掉另外 39 层的白拷贝 */
            const double tf = now_s();
            rc = ds4_engine_v41_score_ids(e, ids, ntok, lp, no_engram, 0);
            ds4_engine_v41_set_moe_hook(NULL, NULL);
            c.t_fwd += now_s() - tf;
            if (rc != 0) { fprintf(stderr, "★第 %d 遍前向失败 rc=%d★\n", il, rc); break; }
            if (c.got != ntok) { fprintf(stderr, "★第 %d 遍只收到 %d/%d 行★\n", il, c.got, ntok); rc = 1; break; }
            if (!merged || pass == 1) break;
            const int w = rb_seq_solve(&c, il, &nrb);
            if (w < 0) { rc = 1; break; }
            if (w == 0) break;   /* 本层路由不动: 这一趟的取料就是部署态, 直接解增益 */
            printf("  [L%02d] Δb 已落盘 → 挂上重取本层(增益解在新路由上)\n", il);
        }
        if (rc == 0 && solve_layer(&c, il) < 0) rc = 1;
    }
    if (rc == 0) {   /* 末遍: 全部放大器挂上跑完整前向, 拟合料 PPL(与 Python 解算趟末尾的 PPL 同口径) */
        ds4_engine_v41_set_amp_dir((c.ok_layers || nrb) ? out : NULL);
        const double tf = now_s();
        rc = ds4_engine_v41_score_ids(e, ids, ntok, lp, no_engram, 0);
        c.t_fwd += now_s() - tf;
        unlink(lp);   /* 拟合料 logits(4 GB)只为打 PPL, 文件不留 */
        if (rc != 0) fprintf(stderr, "★末遍前向失败 rc=%d★\n", rc);
    }
    if (rc != 0) { fprintf(c.mf, "# 失败 rc=%d\n", rc); fclose(c.mf); fprintf(stderr, "★反修失败, 放大器目录是半成品★\n"); return 1; }
    /* 总账(= v41_amp_hooks.report_select) */
    double mb = 0.0; char kd[128] = "";
    /* 体积 = 盘上真实字节: A、B 两块各 K×D 元素, fp4x32 是 17 B/32 元素 */
    for (int k = 0; k < NK; k++) { mb += 2.0 * ((double)c.D * SEL_KS[k] / 32.0 * 17.0) / 1e6 * c.kcount[k]; if (c.kcount[k]) { char b[32]; snprintf(b, sizeof b, "%s%d×%d", kd[0] ? " " : "", SEL_KS[k], c.kcount[k]); strncat(kd, b, sizeof kd - strlen(kd) - 1); } }
    if (c.ok_layers)
        printf("[择优] 挂 %d 层 / 闸掉 %d 层; fp4 后 val 平均 %+.2f%%; ‖Δ‖/‖y_q‖ 平均 %.4f; K 分布 %s; fp4x32 落地 %.0f MB = %.2f%% 解码带宽; 用时 前向 %.0fs fp %.0fs 扫 %.0fs 解 %.0fs 总 %.0fs\n",
               c.ok_layers, c.skip_layers, c.val_sum / c.ok_layers, c.dz_sum / c.ok_layers, kd, mb, mb / 10050 * 100, c.t_fwd, c.t_fp, c.t_scan, c.t_solve, now_s() - t0);
    else printf("[择优] 零层挂上(闸掉 %d) —— 检查靶/输入\n", c.skip_layers);
    if (merged) { printf("[合并序贯] 路由偏置落盘 %d 层 rb_Lnn.bin(与 gr 同目录)\n", nrb); rb_seq_free(); ds4_gguf_close(&c.gq); }
    if (merged) fprintf(c.mf, "# 完成 挂%d 闸%d 路由%d\n", c.ok_layers, c.skip_layers, nrb);
    else fprintf(c.mf, "# 完成 挂%d 闸%d\n", c.ok_layers, c.skip_layers);
    fclose(c.mf);
    if (c.klt) v41_klt_close(c.klt);
    ds4_engine_close(e);
    v41_st_close(&c.S);
    return 0;
}

#include "v41_fp_bind.inc.c"  /* HF 张量 → v41_fp_mat 的绑定(同上) */
#include "v41_amp_lowrank.inc.c" /* 低秩放大器那条路的逐层解算(同上) */
#include "v41_gr_run.inc.c"   /* 权重侧逐专家增益的驱动编排(单 TU; 拆文件只为守 500 行) */
#include "v41_rb_run.inc.c"   /* 路由偏置侧车的拟合驱动(同上) */
#include "v41_sft_run.inc.c"  /* 后训练第三件: 取料、清单、行表(同上) */
#include "v41_margin_solve.inc.c" /* 后训练第三件: 决策差解算 + 主动集(用上面的 mg_acc) */
#include "v41_margin_gen.inc.c"   /* 后训练第三件: 留一泛化尺 + 诊断(用上面的 mg_candidate) */
#include "v41_calib_dump.inc.c"   /* 量化校准取料(--dump-calib; 单 TU 同上) */
