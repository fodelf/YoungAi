/* v41_amp_run.c — V4.1 反修驱动: 在 ds4 引擎的真前向上逐层【取 → 扫 → 选 → 解】(2026-09-13), 零 Python。
 *
 * 【为什么要这个】09-12 的放大器是在 v41_teacher.py 的 torch 前向里解的 —— 零件(VQ 解码/解算/应用/判决)早已 C 化, 但序贯
 * 循环挂在 Python+torch 上, 违反"算法必须 C"与"捕获必须与部署同路"两条铁律: 解算时看到的 x 是 torch 模拟量化传播出来的,
 * 部署时喂进放大器的是引擎 VQ kernel 传播出来的, 不是同一笔账(同档逐位对拍 KL 0.007, 小但不是零)。
 * 这里把循环搬到引擎上: ds4_engine_v41_set_moe_hook 在每层 MoE 出口回调, 本程序拿到部署态 x_q / y_q / 路由, 用 HF 出厂权重
 * 算 y_fp(x_q)(v41_fp_moe.cu), 靶 = y_fp − y_q, 解算走 v41_amp_solve.cu(与 Python 路 libv41amp.so 同一份 CUDA 源)。
 *
 * 【多遍序贯】第 k 遍: 引擎挂着本目录里已解的 L0..L(k−1)(--zchain 同一条加载/应用路), 按 512 分块正常前向, 每块到第 k 层
 * 取料即停; 收齐全部行后解第 k 层落盘; 末遍不挂钩子跑完整前向, 打出拟合料 PPL(与 Python 解算趟末尾的 PPL 同口径)。
 * 与"一次前向里逐层解+挂"数学等价(第 k 层看到的 x 都是 L0..L(k−1) 修正后传播的)。为什么不整批一次前向: 8192 整批的引擎
 * 缓冲(logits 4.2 GB / q,o 各 1 GB / hc 1.3 GB …)把 spark avail 压到 6 GB(看门狗线 8 GB 以下), 且 L02 前向失败(未查);
 * 多遍用的是判决同一套 512 分块配置, 内存账已验证。代价 = 第 k 遍只跑 k+1 层, 40 遍合计约 20 趟全前向。
 *
 * 【择优规则】与 v41_amp_hooks.install_online_select 同(09-12 金融反修方案): held-out 按 <ids>.layout 的窗分层(每域窗号
 * k%4==3 作 val), K∈{64,128,256,512} × λ∈{1,3,10,30,100} 只认 val; val ≥ 0.9×最优 的格里取最小 K、同 K 最大 λ; 层最优
 * val < 0.5% 不挂。产物 amp_Lnn.bin + manifest.txt 与 Python 路同格式(judge 脚本认 "# 完成")。
 *
 * 用法: v41_amp_run <model.gguf> <hf-dir> <拟合ids> <ntok> <出目录> [--layers N] [--whiten 0|1|2] [--mem-budget-mb M] [--no-engram] [--only-layer N --target kl:<教师锚> --eta-rel r]
 * 编译: make -C gguf-tools v41_amp_run(Linux+CUDA; 要先在仓库根 make cuda-spark 出引擎对象) */
#define _GNU_SOURCE
#include "../../src/common/ds4_st41.h"
#include "../../src/common/ds4_quantfmt.h"
#include "../../src/common/ds4_float.h"
#include "../../ds4.h"
#include "v41_fp_moe.h"
#include "v41_kl_target.h"
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
                           int n, int nfit, int nu, int D, int n_expert, float lam, int niter, float *ds,
                           float *out_train, float *out_val);

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
    float *dX, *dYq, *dYfp, *dYtmp, *dA, *dB;     /* 设备 */
    float *hA, *hB, *hx, *hy, *hrw, *halpha; int *hsel;   /* 主机: 放大器下载 / 置换后副本 */
    uint8_t *pkA, *pkB;                    /* fp4x32 码字(落盘的就是它) */
    float *hvf, *hvt;                      /* val 行的 y_fp / 工作副本: fp4 往返后重算 val 用 */
    v41_fp_ffn *ex, sh;
    /* 蒸馏靶(--target kl): 靶不再是"这一层像教师", 而是"让最终 logits 像教师该往哪挪"。见 v41_kl_target.h */
    v41_klt *klt; const char *kl_ref, *kl_stu; float eta_rel;
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

/* HF 张量 → v41_fp_mat(指针指向 mmap, 分片按需映射) */
static int fill_mat(v41_st *S, const char *base, v41_fp_mat *M, int fp8) {
    char wn[256], sn[256]; snprintf(wn, sizeof wn, "%s.weight", base); snprintf(sn, sizeof sn, "%s.scale", base);
    const v41_st_ent *W = v41_st_find(S, wn), *Sc = v41_st_find(S, sn);
    if (!W || !Sc) { fprintf(stderr, "★HF 缺 %s / .scale★\n", wn); return -1; }
    if (strcmp(W->dtype, fp8 ? "F8_E4M3" : "I8")) { fprintf(stderr, "★%s dtype %s, 期 %s★\n", wn, W->dtype, fp8 ? "F8_E4M3" : "I8"); return -1; }
    if (W->nd != 2 || Sc->nd != 2) { fprintf(stderr, "★%s 不是二维★\n", wn); return -1; }
    M->w = v41_st_data(S, W); M->s = v41_st_data(S, Sc);
    if (!M->w || !M->s) return -1;
    M->rows = (int)W->shape[0]; M->cols = fp8 ? (int)W->shape[1] : (int)W->shape[1] * 2; M->fp8 = fp8;
    if (fp8) { M->sbr = M->rows / (int)Sc->shape[0]; M->sbc = M->cols / (int)Sc->shape[1]; }
    else {
        M->sbr = 1; M->sbc = 32;
        if (Sc->shape[0] != M->rows || Sc->shape[1] * 32 != M->cols) { fprintf(stderr, "★%s scale 形状 %lld×%lld 对不上 %d×%d★\n", wn, (long long)Sc->shape[0], (long long)Sc->shape[1], M->rows, M->cols); return -1; }
    }
    return 0;
}
static int resolve_layer(ctx_t *c, int il) {
    char nm[256];
    for (int e = 0; e < c->n_expert; e++) {
        snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w1", il, e); if (fill_mat(&c->S, nm, &c->ex[e].w1, 0)) return -1;
        snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w3", il, e); if (fill_mat(&c->S, nm, &c->ex[e].w3, 0)) return -1;
        snprintf(nm, sizeof nm, "layers.%d.ffn.experts.%d.w2", il, e); if (fill_mat(&c->S, nm, &c->ex[e].w2, 0)) return -1;
    }
    snprintf(nm, sizeof nm, "layers.%d.ffn.shared_experts.w1", il); if (fill_mat(&c->S, nm, &c->sh.w1, 1)) return -1;
    snprintf(nm, sizeof nm, "layers.%d.ffn.shared_experts.w3", il); if (fill_mat(&c->S, nm, &c->sh.w3, 1)) return -1;
    snprintf(nm, sizeof nm, "layers.%d.ffn.shared_experts.w2", il); if (fill_mat(&c->S, nm, &c->sh.w2, 1)) return -1;
    return 0;
}

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
    /* 蒸馏靶要同一趟的学生 logits ⇒ 不能半路停车, 让前向跑完出口(取料只多一层拷贝的代价)。 */
    return c->klt ? 0 : 1;
}

static int solve_layer_gr(ctx_t *c, int il, double t0, double t1);   /* v41_gr_run.inc.c(权重侧逐专家增益) */

/* 解第 il 层: 置换 → 上传 → FP 靶 → K×λ 扫描 → 择优 → 解 → 落盘 + manifest。返回 <0 失败, 0 跳过, 1 挂上 */
static int solve_layer(ctx_t *c, int il) {
    const int n = c->n, D = c->D, nu = c->n_used;
    const double t0 = now_s();
    if (!c->klt && resolve_layer(c, il)) return -1;   /* 蒸馏靶不碰 HF 出厂权重: 靶只来自两份 logits */
    if (c->want_ye) {
        /* ★机制审计★ 逐专家输出必须能逐位重建引擎这一层的输出: y[i][d] == Σ_k rw[i][k]·ye[i][k][d]。
         * 对不上就说明取的不是本层的 ys、或 inv 配对错位 —— 那样解出来的 g 全是假账, 且不会报错。
         * 阈值 1e-5: f32 累加序不同(kernel 定序 vs 这里顺序)只该差到 1e-7 量级, 留两个数量级余量。 */
        double num = 0, den = 0; long same = 0, tot = 0;
        for (int i = 0; i < n; i++) for (int d = 0; d < D; d++) {
            float s = c->rysh[(size_t)i * D + d];
            for (int k = 0; k < nu; k++) s += c->rrw[(size_t)i * nu + k] * c->rye[((size_t)i * nu + k) * D + d];
            const float sb = bf16r(s), yv = c->ry[(size_t)i * D + d];
            num += (double)(sb - yv) * (sb - yv); den += (double)yv * yv;
            tot++; if (sb == yv) same++;
        }
        const double rel = den > 0 ? sqrt(num / den) : 0;
        printf("[L%02d] 取料自检: round_bf16(Σ_k rw·ye + ysh) vs y — 逐位同 %.4f%%, 相对差 %.3e %s\n",
               il, 100.0 * same / tot, rel, rel < 1e-4 ? "✓" : "★不符★");
        fflush(stdout);
        if (!(rel < 1e-4)) return -1;
    }
    for (int i = 0; i < n; i++) {   /* 行置换: 拟合行在前、val 行在后 —— 解算只看前 nfit 行(连续切片) */
        const int r = c->perm[i];
        memcpy(c->hx + (size_t)i * D, c->rx + (size_t)r * D, (size_t)D * 4); memcpy(c->hy + (size_t)i * D, c->ry + (size_t)r * D, (size_t)D * 4);
        memcpy(c->hsel + (size_t)i * nu, c->rsel + (size_t)r * nu, (size_t)nu * 4); memcpy(c->hrw + (size_t)i * nu, c->rrw + (size_t)r * nu, (size_t)nu * 4);
        c->halpha[i] = c->ralpha[r];
    }
    if (c->want_ye) {   /* ye/ysh 也按同一个置换搬 —— 行序配错不会报错, 只会出假账 */
        for (int i = 0; i < n; i++) {
            const int r = c->perm[i];
            memcpy(c->hye + (size_t)i * nu * D, c->rye + (size_t)r * nu * D, (size_t)nu * D * 4);
            memcpy(c->hysh + (size_t)i * D, c->rysh + (size_t)r * D, (size_t)D * 4);
        }
        CK(cudaMemcpy(c->dye, c->hye, (size_t)n * nu * D * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(c->dysh, c->hysh, (size_t)n * D * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(c->drw, c->hrw, (size_t)n * nu * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(c->dsel, c->hsel, (size_t)n * nu * 4, cudaMemcpyHostToDevice));
    }
    const size_t nD = (size_t)n * D;
    CK(cudaMemcpy(c->dX, c->hx, nD * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c->dYq, c->hy, nD * 4, cudaMemcpyHostToDevice));
    double qn2 = 0.0; for (size_t i = 0; i < nD; i++) qn2 += (double)c->hy[i] * c->hy[i];
    const double qnorm = sqrt(qn2);
    if (c->klt) {
        float st4[4];
        if (v41_klt_target(c->klt, c->kl_ref, c->kl_stu, c->ralpha, c->perm, n, D, qnorm, c->eta_rel, c->dYq, c->dYfp, st4)) {
            fprintf(stderr, "  [L%02d] ★蒸馏靶计算失败, 停车★\n", il); return -1;
        }
        printf("  [L%02d] 蒸馏靶: ‖α·g‖/‖y_q‖=%.4f → 按 --eta-rel 缩到 %.4f; α 范围 [%.4f, %.4f] 均值 %.4f\n",
               il, st4[0], c->eta_rel, st4[1], st4[2], st4[3]);
        if (st4[1] <= 0.f) printf("  [L%02d] ★α 跨零: 有行的靶方向是反的, 记账不停车(判决时看端到端)★\n", il);
    } else if (v41_fp_moe_layer(c->ex, c->n_expert, &c->sh, c->dX, c->hsel, c->hrw, n, D, c->MID, nu, c->clamp, c->dYfp)) {
        fprintf(stderr, "  [L%02d] ★FP 靶计算失败, 停车★\n", il); return -1;
    }
    const double t1 = now_s(); c->t_fp += t1 - t0;
    if (c->want_ye) return solve_layer_gr(c, il, t0, t1);
    float tr[NL * NK], va[NL * NK], trw[NL * NK], vaw[NL * NK], rel = 0.f;
    if (v41_amp_scan_k_gpu(c->dX, c->dYfp, c->dYq, n, D, c->nfit, SEL_KS, NK, SEL_LAMS, NL, tr, va, &rel, c->whiten, trw, vaw)) { fprintf(stderr, "  [L%02d] ★K×λ 扫描失败, 停车★\n", il); return -1; }
    const double t2 = now_s(); c->t_scan += t2 - t1;
    float pct[NL][NK], pct_w[NL][NK], pct_tr[NL][NK];
    for (int l = 0; l < NL; l++) for (int k = 0; k < NK; k++) {
        pct[l][k] = 100.f * (1.f - va[l * NK + k]); pct_tr[l][k] = 100.f * (1.f - tr[l * NK + k]);
        pct_w[l][k] = c->whiten ? 100.f * (1.f - vaw[l * NK + k]) : pct[l][k];
    }
    float best; int ki, lj;
    const int has = choose(c->whiten ? pct_w : pct, &best, &ki, &lj);   /* 白化时按白化 val 择优, 原始 val 照记 */
    if (!has || best < GATE_PCT) {
        c->skip_layers++;
        fprintf(c->mf, "L%02d skip - %.4f - %+.2f -\n", il, rel, best); fflush(c->mf);
        printf("  [L%02d] 靶/‖y_q‖=%.3f | val 最优 %+.2f%% < 闸 %.1f%%, 不挂  (fp %.1fs 扫 %.1fs)\n", il, rel, best, GATE_PCT, t1 - t0, t2 - t1);
        v41_st_release_idle(&c->S);
        return 0;
    }
    const int Kc = SEL_KS[ki]; const float lam = SEL_LAMS[lj];
    float ratio = 1.f;
    const int rc = v41_amp_solve_layer_gpu(c->dX, c->dYfp, c->dYq, c->nfit, D, Kc, lam, c->dA, c->dB, &ratio, c->whiten);
    const double t3 = now_s(); c->t_solve += t3 - t2;
    if (rc != 0) {
        c->skip_layers++;
        fprintf(c->mf, "L%02d skip - %.4f - %+.2f - (解算自检不过 残差比 %.3f)\n", il, rel, best, ratio); fflush(c->mf);
        printf("  [L%02d] ★解算自检不过(残差比 %.3f), 不挂★\n", il, ratio);
        v41_st_release_idle(&c->S);
        return 0;
    }
    const size_t KD = (size_t)Kc * D;
    CK(cudaMemcpy(c->hA, c->dA, KD * 4, cudaMemcpyDeviceToHost));   /* A 行主序 [K][D] */
    CK(cudaMemcpy(c->hB, c->dB, KD * 4, cudaMemcpyDeviceToHost));   /* B 列主序 D×K = 内存 [K][D](与离线格式/引擎加载同) */
    /* ★fp4x32 往返必须在判决之前★(2026-09-13): 落盘格式是 4.25 bpw 的 fp4x32, 部署时引擎解出来的
     * 是【量化过的】A/B。所以这里编码完立刻解回来覆盖 hA/hB 并推回设备 —— 下面的 val 重算与
     * ‖Δ‖ 信任域量的都是部署那一份。少了这一步, 报的是 f32 解的成绩而挂上去的是 fp4 的东西,
     * 又是一笔"内部好看、落地劣化"的账(本仓吃过亏)。 */
    if (KD % 32u) { fprintf(stderr, "  [L%02d] ★K×D=%zu 不是 32 的整数倍, fp4x32 装不下★\n", il, KD); return -1; }
    const size_t nblk = KD / 32u, nby = nblk * 17u;
    ds4_quant_fp4x32(c->hA, nblk, c->pkA); ds4_deq_fp4x32(c->pkA, nblk, c->hA);
    ds4_quant_fp4x32(c->hB, nblk, c->pkB); ds4_deq_fp4x32(c->pkB, nblk, c->hB);
    CK(cudaMemcpy(c->dA, c->hA, KD * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(c->dB, c->hB, KD * 4, cudaMemcpyHostToDevice));
    /* ‖Δ‖/‖y_q‖ 信任域 + fp4 态 val 重算: 在 y_q 副本上应用一次, 再在 val 行(置换后是连续的尾段)
     * 上量 ‖(y_fp−y_q−Δ)‖/‖y_fp−y_q‖ —— 与扫描的 val 同式, 差别只在 A/B 过了 fp4。 */
    float dz = 0.f;
    CK(cudaMemcpy(c->dYtmp, c->dYq, nD * 4, cudaMemcpyDeviceToDevice));
    if (v41_amp_apply_gpu(c->dX, c->dYtmp, c->dA, c->dB, n, D, Kc, &dz)) return -1;
    const int nval = n - c->nfit;
    const size_t voff = (size_t)c->nfit * D, vlen = (size_t)nval * D;
    CK(cudaMemcpy(c->hvf, c->dYfp + voff, vlen * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(c->hvt, c->dYtmp + voff, vlen * 4, cudaMemcpyDeviceToHost));
    double rn2 = 0.0;
    for (size_t i = 0; i < vlen; i++) { const double d0 = (double)c->hvf[i] - c->hvt[i]; rn2 += d0 * d0; }
    CK(cudaMemcpy(c->hvt, c->dYq + voff, vlen * 4, cudaMemcpyDeviceToHost));
    double bn2 = 0.0;
    for (size_t i = 0; i < vlen; i++) { const double d0 = (double)c->hvf[i] - c->hvt[i]; bn2 += d0 * d0; }
    const float val4 = bn2 > 0.0 ? (float)(100.0 * (1.0 - sqrt(rn2 / bn2))) : 0.f;
    if (val4 < GATE_PCT) {   /* ★闸判在 fp4 态★: f32 解过闸但量化后不过, 说明这一层的收益全在 fp4 装不下的精度里 */
        c->skip_layers++;
        fprintf(c->mf, "L%02d skip - %.4f - %+.2f - (fp4 后 val %+.2f < 闸)\n", il, rel, best, val4);
        printf("  [L%02d] 选 λ=%g K=%d val %+.1f%% → ★fp4 往返后掉到 %+.2f%% < 闸 %.1f%%, 不挂★\n", il, lam, Kc, pct[lj][ki], val4, GATE_PCT);
        fflush(c->mf); v41_st_release_idle(&c->S);
        return 0;
    }
    char po[4300], pt[4300]; snprintf(po, sizeof po, "%s/amp_L%02d.bin", c->out_dir, il); snprintf(pt, sizeof pt, "%s.part", po);
    FILE *f = fopen(pt, "wb");
    if (!f) { fprintf(stderr, "★写不了 %s★\n", pt); return -1; }
    const int32_t hd[3] = { D, Kc, DS4_GGT_FP4X32 };   /* 头第三字段 = 存储类型, 引擎按它分派(43=fp4x32) */
    fwrite(hd, 4, 3, f); fwrite(c->pkA, 1, nby, f); fwrite(c->pkB, 1, nby, f);
    if (fclose(f) || rename(pt, po)) { fprintf(stderr, "★落盘 %s 失败★\n", po); return -1; }   /* 写完才改名: 半成品永不被下一遍挂上 */
    const double relz = qnorm > 0 ? dz / qnorm : 0.0, tr_pct = 100.0 * (1.0 - ratio);
    c->ok_layers++; c->kcount[ki]++; c->val_sum += val4; c->dz_sum += relz;
    fprintf(c->mf, "L%02d %g %d %.4f %+.2f %+.2f %.4f", il, lam, Kc, rel, tr_pct, val4, relz);
    if (c->whiten) fprintf(c->mf, " %+.2f", pct_w[lj][ki]);
    fprintf(c->mf, "\n"); fflush(c->mf);
    printf("  [L%02d] 靶/‖y_q‖=%.3f | 选 λ=%g K=%d: train %+.1f%% val %+.1f%% → ★fp4 后 %+.1f%%★%s [最优 %+.1f%%]  ‖Δ‖/‖y_q‖=%.4f  (fp %.1fs 扫 %.1fs 解 %.1fs)\n",
           il, rel, lam, Kc, tr_pct, pct[lj][ki], val4, c->whiten ? " (白化择优)" : "", best, relz, t1 - t0, t2 - t1, t3 - t2);
    v41_st_release_idle(&c->S);
    return 1;
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "用法: v41_amp_run <model.gguf> <hf-dir> <拟合ids> <ntok> <出目录> [--layers N] [--whiten 0|1|2] [--mem-budget-mb M] [--no-engram] [--only-layer N --target kl:<教师锚> --eta-rel r]\n"); return 2; }
    const char *gguf = argv[1], *hf = argv[2], *ids_path = argv[3], *out = argv[5];
    const int ntok = atoi(argv[4]);
    ctx_t c; memset(&c, 0, sizeof c);
    c.out_dir = out;
    int mem_mb = 0, no_engram = 0, nlayers = 40, only_layer = -1;
    c.eta_rel = 0.10f;
    for (int i = 6; i < argc; i++) {
        if (!strcmp(argv[i], "--layers") && i + 1 < argc) nlayers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--whiten") && i + 1 < argc) c.whiten = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mem-budget-mb") && i + 1 < argc) mem_mb = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-engram")) no_engram = 1;
        else if (!strcmp(argv[i], "--only-layer") && i + 1 < argc) only_layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--capture-ye")) c.want_ye = 1;
        else if (!strcmp(argv[i], "--eta-rel") && i + 1 < argc) c.eta_rel = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--target") && i + 1 < argc) {   /* --target kl:<教师锚> */
            const char *t = argv[++i];
            if (strncmp(t, "kl:", 3)) { fprintf(stderr, "★--target 只认 kl:<教师锚.bin>★\n"); return 2; }
            c.kl_ref = t + 3;
        } else { fprintf(stderr, "★不认识的参数 %s★\n", argv[i]); return 2; }
    }
    if (c.kl_ref && only_layer < 0) { fprintf(stderr, "★蒸馏靶只对末层成立(梯度链止于出口), 必须配 --only-layer <末层号>★\n"); return 2; }
    if (c.eta_rel <= 0.f || c.eta_rel > 2.f) { fprintf(stderr, "★--eta-rel %g 不合理★\n", c.eta_rel); return 2; }
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (ntok < 2 || nlayers < 1 || nlayers > 64) { fprintf(stderr, "★ntok %d / layers %d★\n", ntok, nlayers); return 2; }
    int *ids = malloc(sizeof(int) * (size_t)ntok), n = 0, t;
    FILE *fi = fopen(ids_path, "r");
    if (!fi) { fprintf(stderr, "★打不开 %s★\n", ids_path); return 1; }
    while (n < ntok && fscanf(fi, "%d", &t) == 1) ids[n++] = t;
    fclose(fi);
    if (n != ntok) { fprintf(stderr, "★%s 只有 %d 个 id < ntok %d★\n", ids_path, n, ntok); return 1; }
    if (layout_split(ids_path, ntok, &c)) return 1;
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
    c.hvf = malloc((size_t)(ntok - c.nfit) * c.D * 4); c.hvt = malloc((size_t)(ntok - c.nfit) * c.D * 4);
    if (!c.pkA || !c.pkB || !c.hvf || !c.hvt) { fprintf(stderr, "★fp4/val 缓冲分配失败★\n"); return 1; }
    if (c.want_ye) {   /* [n][n_used][D] f32: 8192×6×5120×4 = 1.0 GB */
        const size_t nyD = nU * (size_t)c.D, nsD = (size_t)c.n_expert * c.D;
        c.rye = malloc(nyD * 4); c.rysh = malloc(nD * 4);
        c.hye = malloc(nyD * 4); c.hysh = malloc(nD * 4); c.hgr = malloc(nsD * 4);
        if (!c.rye || !c.rysh || !c.hye || !c.hysh || !c.hgr) {
            fprintf(stderr, "★逐专家输出缓冲(2×%.1f GB)分配失败★\n", (double)nyD * 4 / 1e9); return 1; }
        if (cudaMalloc((void **)&c.dye, nyD * 4) || cudaMalloc((void **)&c.dysh, nD * 4) ||
            cudaMalloc((void **)&c.drw, nU * 4) || cudaMalloc((void **)&c.dsel, nU * 4) ||
            cudaMalloc((void **)&c.dgr, nsD * 4)) { fprintf(stderr, "★权重侧解算设备缓冲分配失败★\n"); return 1; }
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
    ds4_engine_options opt; memset(&opt, 0, sizeof opt);
    opt.model_path = gguf; opt.backend = DS4_BACKEND_CUDA;
    ds4_engine *e = NULL;
    const double t0 = now_s();
    if (ds4_engine_open(&e, &opt) != 0 || !e) { fprintf(stderr, "★引擎打不开 %s★\n", gguf); return 1; }
    if (!ds4_engine_is_v41(e)) { fprintf(stderr, "★%s 不是 V4.1 模型★\n", gguf); return 1; }
    printf("[引擎] 打开 %.0fs; 多遍序贯反修开始(每遍取一层)\n", now_s() - t0);
    char lp[4300]; snprintf(lp, sizeof lp, "%s/fit_logits.bin", out);
    int rc = 0;
    if (c.kl_ref) {   /* 蒸馏靶: 出口头 + output_norm 从同一个 GGUF 取(学生自己的出口, 不是 FP 教师的) */
        if (v41_klt_open(&c.klt, gguf, c.D)) { fprintf(stderr, "★蒸馏靶初始化失败★\n"); return 1; }
        c.kl_stu = lp;
        printf("[蒸馏靶] 教师锚 %s; 学生 logits = 本趟前向落盘 %s; --eta-rel %.3f\n", c.kl_ref, lp, c.eta_rel);
    }
    const int il_lo = only_layer >= 0 ? only_layer : 0, il_hi = only_layer >= 0 ? only_layer + 1 : nlayers;
    for (int il = il_lo; il < il_hi && rc == 0; il++) {   /* 第 il 遍: 挂 L0..L(il−1) 已解的放大器, 到 L(il) 取料即停 */
        c.layer = il; c.got = 0;
        ds4_engine_v41_set_amp_dir(c.ok_layers ? out : NULL);   /* 目录里一个放大器都没有时引擎会拒开状态, 所以没解出过就不挂 */
        ds4_engine_v41_set_moe_hook(hook, &c);
        const double tf = now_s();
        rc = ds4_engine_v41_score_ids(e, ids, ntok, lp, no_engram, 0);
        ds4_engine_v41_set_moe_hook(NULL, NULL);
        c.t_fwd += now_s() - tf;
        if (rc != 0) { fprintf(stderr, "★第 %d 遍前向失败 rc=%d★\n", il, rc); break; }
        if (c.got != ntok) { fprintf(stderr, "★第 %d 遍只收到 %d/%d 行★\n", il, c.got, ntok); rc = 1; break; }
        if (solve_layer(&c, il) < 0) rc = 1;
    }
    if (rc == 0) {   /* 末遍: 全部放大器挂上跑完整前向, 拟合料 PPL(与 Python 解算趟末尾的 PPL 同口径) */
        ds4_engine_v41_set_amp_dir(c.ok_layers ? out : NULL);
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
    fprintf(c.mf, "# 完成 挂%d 闸%d\n", c.ok_layers, c.skip_layers); fclose(c.mf);
    if (c.klt) v41_klt_close(c.klt);
    ds4_engine_close(e);
    v41_st_close(&c.S);
    return 0;
}

#include "v41_gr_run.inc.c"   /* 权重侧逐专家增益的驱动编排(单 TU; 拆文件只为守 500 行) */
