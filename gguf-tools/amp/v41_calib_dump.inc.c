/* v41_calib_dump.inc.c — 量化校准取料(2026-09-20 深夜, 用户令"按金融域量化"): 引擎一趟跑金融量化份, 40 层同趟把每层
 * MoE 入口 x(bf16 格点 ⇒ 拷成 bf16 零损)与路由(sel/rw)落盘, 给 v41_quantize --calib 当列权/Gram 的料。
 * v41_amp_run.c 单 TU include, 拆出来只为守 500 行。
 *
 * 【为什么取引擎的 x 而不是教师的 x】部署时每层吃的是量化前缀传播来的 x; 拿教师 x 解出来的修正到部署态全是错位
 * (09-11 x 错位税: 端到端 PPL 崩到 10311)。这里 x 来自现役 q4k-12 裸模型: 老量化前缀与新量化前缀的差是二阶。
 * 【文件】<dir>/calib_Lnn.bin, 布局见 quantize/v41_calib.h(读写只此一份口径)。引擎 512 分块顺序推进、40 层轮着回调,
 * 每块按 pos0 pwrite 到位; 头(含 clamp)在收齐后补写。
 * 【自检】收工后按盘上文件重读: 每层 E[x²] 前 12 通道能量占比(08-24 巨值通道先验: 深层 ~10%)与名下行数(最少/中位/最多
 * 与 <8 行的专家数) —— <8 行的那批就是级 1 里 w2 列权退回平权的专家, 先看见再量化。
 * 出错会怎样: 行数收不齐 = 钩子错位, rc≠0 且文件全删, 不留半成品(半份校准料喂进量化器只会出假数)。 */
#include "../quantize/v41_calib.h"
#include <fcntl.h>
#include <unistd.h>

typedef struct { int fd[64]; int nl, ntok, D, NU; long long got[64]; float clamp; } cd_ctx;

static int cd_pwrite(int fd, const void *p, size_t n, off_t off) {
    const char *b = p;
    while (n) { ssize_t w = pwrite(fd, b, n, off); if (w <= 0) return -1; b += w; n -= (size_t)w; off += w; }
    return 0;
}

static int cd_hook(void *ud, int il, int pos0, int n, int D, int n_used, float clamp, const float *x, const float *y, const int *sel,
                   const float *rw, const float *alpha, const float *ye, const float *ysh) {
    cd_ctx *c = ud; (void)y; (void)alpha; (void)ye; (void)ysh;
    if (il < 0 || il >= c->nl) return 0;
    if (c->NU < 0) c->NU = n_used;
    if (D != c->D || n_used != c->NU || n_used > MAXU || pos0 < 0 || pos0 + n > c->ntok) {
        fprintf(stderr, "★校准钩子 L%02d pos0=%d n=%d D=%d n_used=%d 与 ntok %d / D %d / NU %d 不符★\n", il, pos0, n, D, n_used, c->ntok, c->D, c->NU);
        return -1;
    }
    c->clamp = clamp;
    v41_calib_hdr h; memset(&h, 0, sizeof h); h.ntok = c->ntok; h.D = D; h.n_used = n_used;
    uint16_t *xb = malloc((size_t)n * D * 2);
    if (!xb) return -1;
    for (size_t i = 0; i < (size_t)n * D; i++) { float v = bf16r(x[i]); uint32_t u; memcpy(&u, &v, 4); xb[i] = (uint16_t)(u >> 16); }
    int rc = 0;
    if (cd_pwrite(c->fd[il], xb, (size_t)n * D * 2, (off_t)(v41_calib_off_x(&h) + (size_t)pos0 * D * 2)) ||
        cd_pwrite(c->fd[il], sel, (size_t)n * n_used * 4, (off_t)(v41_calib_off_sel(&h) + (size_t)pos0 * n_used * 4)) ||
        cd_pwrite(c->fd[il], rw, (size_t)n * n_used * 4, (off_t)(v41_calib_off_rw(&h) + (size_t)pos0 * n_used * 4))) {
        fprintf(stderr, "★校准落盘 L%02d 写失败: %s★\n", il, strerror(errno)); rc = -1;
    }
    free(xb);
    if (rc == 0) c->got[il] += n;
    return rc;
}

/* 自检: 按盘上文件重读一层, 报 E[x²] 前 12 通道占比 + 名下行数分布 */
static int cd_selfcheck(const char *dir, int il, int n_expert) {
    char p[4300]; snprintf(p, sizeof p, "%s/calib_L%02d.bin", dir, il);
    FILE *f = fopen(p, "rb"); if (!f) return -1;
    v41_calib_hdr h;
    if (fread(&h, sizeof h, 1, f) != 1 || memcmp(h.magic, V41_CALIB_MAGIC, 4)) { fclose(f); return -1; }
    const size_t nD = (size_t)h.ntok * h.D, nU = (size_t)h.ntok * h.n_used;
    uint16_t *xb = malloc(nD * 2); int *sel = malloc(nU * 4);
    double *e2 = calloc((size_t)h.D, sizeof(double)); int *cnt = calloc((size_t)n_expert, sizeof(int));
    int ok = xb && sel && e2 && cnt && fread(xb, 2, nD, f) == nD && fseek(f, (long)v41_calib_off_sel(&h), SEEK_SET) == 0 && fread(sel, 4, nU, f) == nU;
    fclose(f);
    if (ok) {
        for (size_t i = 0; i < nD; i++) { uint32_t u = (uint32_t)xb[i] << 16; float v; memcpy(&v, &u, 4); e2[i % h.D] += (double)v * v; }
        double tot = 0, top = 0;
        for (int d = 0; d < h.D; d++) tot += e2[d];
        for (int t = 0; t < 12; t++) { int bi = 0; for (int d = 1; d < h.D; d++) if (e2[d] > e2[bi]) bi = d; top += e2[bi]; e2[bi] = -1; }
        for (size_t i = 0; i < nU; i++) if (sel[i] >= 0 && sel[i] < n_expert) cnt[sel[i]]++;
        int mn = cnt[0], mx = cnt[0], lt8 = 0, med = 0;
        for (int e = 0; e < n_expert; e++) { if (cnt[e] < mn) mn = cnt[e]; if (cnt[e] > mx) mx = cnt[e]; if (cnt[e] < 8) lt8++; }
        int *hist = calloc((size_t)h.ntok * h.n_used + 1, sizeof(int));
        if (hist) { for (int e = 0; e < n_expert; e++) hist[cnt[e]]++; int acc = 0; for (int v = 0; v <= h.ntok * h.n_used; v++) { acc += hist[v]; if (acc * 2 >= n_expert) { med = v; break; } } free(hist); }
        printf("  [校准] L%02d 行 %d  E[x²] top12 通道占 %.1f%%  名下行数 最少/中位/最多 %d/%d/%d, <8 行 %d 个专家\n",
               il, h.ntok, 100.0 * top / (tot > 0 ? tot : 1), mn, med, mx, lt8);
    }
    free(xb); free(sel); free(e2); free(cnt);
    return ok ? 0 : -1;
}

static int calib_dump(ds4_engine *e, const int *ids, int ntok, const char *dir, int nl, int no_engram, int D, int n_expert,
                      const char *gguf, const char *ids_path) {
    cd_ctx c; memset(&c, 0, sizeof c); c.nl = nl; c.ntok = ntok; c.D = D; c.NU = -1;
    char p[4300];
    for (int il = 0; il < nl; il++) {
        snprintf(p, sizeof p, "%s/calib_L%02d.bin", dir, il);
        c.fd[il] = open(p, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (c.fd[il] < 0) { fprintf(stderr, "★建不了 %s: %s★\n", p, strerror(errno)); return -1; }
    }
    ds4_engine_v41_set_amp_dir(NULL);   /* 裸 ①: 校准料不依赖 ②(新 ① 定案后 ② 另解) */
    ds4_engine_v41_set_moe_hook(cd_hook, &c);
    ds4_engine_v41_set_moe_hook_layer(-1);
    snprintf(p, sizeof p, "%s/fit_logits.bin", dir);
    const double t0 = now_s();
    int rc = ds4_engine_v41_score_ids(e, ids, ntok, p, no_engram, 0);
    ds4_engine_v41_set_moe_hook(NULL, NULL);
    unlink(p);   /* 出口 logits 只是前向的副产物, 校准不用 */
    for (int il = 0; il < nl && rc == 0; il++) if (c.got[il] != ntok) { fprintf(stderr, "★L%02d 只收到 %lld/%d 行★\n", il, c.got[il], ntok); rc = 1; }
    for (int il = 0; il < nl && rc == 0; il++) {
        v41_calib_hdr h; memset(&h, 0, sizeof h); memcpy(h.magic, V41_CALIB_MAGIC, 4);
        h.ntok = ntok; h.D = D; h.n_used = c.NU; h.clamp = c.clamp;
        if (cd_pwrite(c.fd[il], &h, sizeof h, 0)) { fprintf(stderr, "★L%02d 头写失败★\n", il); rc = 1; }
    }
    for (int il = 0; il < nl; il++) close(c.fd[il]);
    if (rc != 0) {
        for (int il = 0; il < nl; il++) { snprintf(p, sizeof p, "%s/calib_L%02d.bin", dir, il); unlink(p); }
        return -1;
    }
    printf("[校准取料] %d 层 × %d 行, D=%d n_used=%d clamp=%g, 前向 %.0fs → %s/calib_Lnn.bin(每层 %.1f MB)\n",
           nl, ntok, D, c.NU, c.clamp, now_s() - t0, dir, ((double)ntok * D * 2 + (double)ntok * c.NU * 8) / 1e6);
    for (int il = 0; il < nl; il++) if (cd_selfcheck(dir, il, n_expert)) { fprintf(stderr, "★L%02d 自检读回失败★\n", il); return -1; }
    /* <ids>.layout 抄一份进取料目录: 级 2 的 β 二选一按 val 窗判, 量化器只认目录里这份(与料同源, 不再指回语料路径) */
    { char lp[4300], dp[4300]; snprintf(lp, sizeof lp, "%s.layout", ids_path); snprintf(dp, sizeof dp, "%s/calib.layout", dir);
      FILE *fi = fopen(lp, "rb"), *fo = fi ? fopen(dp, "wb") : NULL; char buf[4096]; size_t nb;
      if (fi && fo) while ((nb = fread(buf, 1, sizeof buf, fi)) > 0) fwrite(buf, 1, nb, fo);
      if (fi) fclose(fi); if (fo) fclose(fo); }
    snprintf(p, sizeof p, "%s/calib.txt", dir);
    FILE *f = fopen(p, "w");
    if (f) { fprintf(f, "gguf=%s ids=%s ntok=%d layers=%d D=%d n_used=%d clamp=%g amp=none\n", gguf, ids_path, ntok, nl, D, c.NU, c.clamp); fclose(f); }
    return 0;
}
