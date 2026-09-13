/* v41_quantize.c — DeepSeek-V4.1-Flash HF → 量化模型目录, 纯权重量化不需要前向(2026-09-12)。
 *
 * 【配方(用户 09-12 定)】
 *   routed 专家(40 层 × 384)  VQ dim8×nc4096: 索引 1.5 bpw + f16 码本(每专家) + f16 行增益
 *                              ≈ 1.518 bpw。无语料校准(最近邻平权)。
 *   骨架(attn/shared/indexer 投影, embed/head, vision 大矩阵)  FP4 E2M1 + 每 32 列 ue8m0 = 4.25 bpw
 *   MTP 三塔自带专家            出厂就是 FP4, 原样透传(草稿质量决定投机接受率, 只占专家 2.4%)
 *   engram 查表(203 GB)        不动: 留在 HF 第 47/48 分片, 输出目录软链过去(量化器不写它)
 *   其余(norm/gate/hc/sink/小 BF16 投影)  原样透传
 *
 * 【输出】<out>/model-layerNN.safetensors × 40 + model-common.safetensors + model.safetensors.index.json。
 * 每层一个分片: 断点续跑按层跳过已完成的(最终名存在才算完成, 半成品是 .part)。
 * 【读它】v41_hf_io.py(教师端)认 index.json: 骨架 FP4 与出厂 routed 专家同格式零改动;
 * VQ 三件(idx/cb/gain)走 libv41vq.so 的 v41_vq_decode_gpu —— 与量化器内部算残差的是同一个核。
 *
 * 用法: v41_quantize <hf-dir> <out-dir> [--vq-dim 8] [--vq-nc 4096] [--vq-iters 8] [--vq-stride 8]
 *                    [--layers a:b] [--no-common] [--force]
 * 编译: make -C gguf-tools v41_quantize (Linux+CUDA) */
#include "../../src/common/ds4_st41.h"
#include "v41_st_write.h"
#include <math.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>

extern int v41_vq_expert_from_fp4(const uint8_t *const *w, const uint8_t *const *s, const int *rows, const int *cols,
                                  int nmat, int dim, int nc, int iters, int stride,
                                  uint8_t **idx_out, uint16_t *cb_out, uint16_t **gain_out, double *stats);
extern int v41_fp4_requant_host(const uint8_t *w, const uint8_t *s, const char *dtype, int rows, int cols,
                                int sbr, int sbc, uint8_t *packed_out, uint8_t *scale_out, double *stats);

typedef struct { int dim, nc, iters, stride, l0, l1, common, force; } cfg_t;
enum { J_PASS, J_FP4, J_VQ };
typedef struct {
    int kind;
    const v41_st_ent *w, *s;              /* PASS/FP4 */
    const v41_st_ent *ew[3], *es[3];      /* VQ: w1, w3, w2 */
    int t[8];                             /* 输出张量序号 */
} job_t;
typedef struct { double sse, en; uint64_t bytes; long long params; } acc_t;
static acc_t g_vq, g_fp4, g_pass;        /* 全程账 */
static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static int bits_of(int nc) { int b = 1; while ((1 << b) < nc) b++; return b; }
static int ends_with(const char *s, const char *suf) { size_t a = strlen(s), b = strlen(suf); return a >= b && !strcmp(s + a - b, suf); }
static int is_routed_expert(const char *n) { return !strncmp(n, "layers.", 7) && strstr(n, ".ffn.experts.") != NULL; }

/* 哪些张量重量化到 FP4(见文件头"配方") */
static int wants_fp4(const v41_st_ent *E) {
    if (E->nd != 2 || !ends_with(E->name, ".weight")) return 0;
    if (strstr(E->name, ".gate.") || strstr(E->name, ".compressor.") || strstr(E->name, ".indexer.w") ||
        strstr(E->name, ".indexer.k_norm") || strstr(E->name, ".indexer.weights_proj")) {
        if (strcmp(E->dtype, "F8_E4M3")) return 0;        /* 官方留 BF16 的小投影不动; F8 的照常 */
    }
    if (!strcmp(E->dtype, "F8_E4M3")) return E->shape[1] % 32 == 0;
    if (!strcmp(E->dtype, "BF16")) {
        int big = (!strcmp(E->name, "embed.weight") || !strcmp(E->name, "head.weight") || !strncmp(E->name, "vision.", 7));
        return big && E->shape[0] * E->shape[1] >= 1000000 && E->shape[1] % 32 == 0;
    }
    return 0;
}

static const v41_st_ent *sibling_scale(const v41_st *S, const v41_st_ent *W) {
    char n[256]; size_t L = strlen(W->name);
    if (L < 7 || L + 1 > sizeof n) return NULL;
    memcpy(n, W->name, L - 6); strcpy(n + L - 6, "scale");
    return v41_st_find(S, n);
}

/* ---- 计划: 给一个分片登记全部输出张量 ---- */
static int plan_pass(v41_stw *W, job_t *J, const v41_st *S, const v41_st_ent *E) {
    J->kind = J_PASS; J->w = E; J->s = ends_with(E->name, ".weight") ? sibling_scale(S, E) : NULL;
    J->t[0] = v41_stw_plan(W, E->name, E->dtype, E->nd, E->shape, E->nbytes);
    if (J->s) J->t[1] = v41_stw_plan(W, J->s->name, J->s->dtype, J->s->nd, J->s->shape, J->s->nbytes);
    return 0;
}
static int plan_fp4(v41_stw *W, job_t *J, const v41_st *S, const v41_st_ent *E) {
    J->kind = J_FP4; J->w = E; J->s = sibling_scale(S, E);
    if (!strcmp(E->dtype, "F8_E4M3") && !J->s) { fprintf(stderr, "★%s 是 F8 却没有 scale★\n", E->name); return -1; }
    int64_t rows = E->shape[0], cols = E->shape[1];
    int64_t shw[2] = {rows, cols / 2}, shs[2] = {rows, cols / 32};
    char sn[256]; snprintf(sn, sizeof sn, "%.*sscale", (int)strlen(E->name) - 6, E->name);
    J->t[0] = v41_stw_plan(W, E->name, "I8", 2, shw, (uint64_t)rows * cols / 2);
    J->t[1] = v41_stw_plan(W, sn, "F8_E8M0", 2, shs, (uint64_t)rows * (cols / 32));
    return 0;
}
static int plan_vq(v41_stw *W, job_t *J, const v41_st *S, int L, int e, const cfg_t *C) {
    static const char *mat[3] = {"w1", "w3", "w2"};
    J->kind = J_VQ;
    char n[256];
    snprintf(n, sizeof n, "layers.%d.ffn.experts.%d.vq.cb", L, e);
    int64_t shc[2] = {C->nc, C->dim};
    J->t[0] = v41_stw_plan(W, n, "F16", 2, shc, (uint64_t)C->nc * C->dim * 2);
    int bits = bits_of(C->nc);
    for (int m = 0; m < 3; m++) {
        snprintf(n, sizeof n, "layers.%d.ffn.experts.%d.%s.weight", L, e, mat[m]);
        J->ew[m] = v41_st_find(S, n);
        snprintf(n, sizeof n, "layers.%d.ffn.experts.%d.%s.scale", L, e, mat[m]);
        J->es[m] = v41_st_find(S, n);
        if (!J->ew[m] || !J->es[m] || strcmp(J->ew[m]->dtype, "I8")) { fprintf(stderr, "★缺专家张量或非 FP4: %s★\n", n); return -1; }
        int64_t rows = J->ew[m]->shape[0], cols = J->ew[m]->shape[1] * 2;
        if (cols % C->dim) { fprintf(stderr, "★%s 列数 %lld 不是 dim 倍数★\n", n, (long long)cols); return -1; }
        int64_t nidx_row = cols / C->dim, bytes_row = (nidx_row * bits + 7) / 8;
        int64_t shi[2] = {rows, bytes_row}, shg[1] = {rows};
        snprintf(n, sizeof n, "layers.%d.ffn.experts.%d.%s.vq.idx", L, e, mat[m]);
        J->t[1 + m * 2] = v41_stw_plan(W, n, "U8", 2, shi, (uint64_t)rows * bytes_row);
        snprintf(n, sizeof n, "layers.%d.ffn.experts.%d.%s.vq.gain", L, e, mat[m]);
        J->t[2 + m * 2] = v41_stw_plan(W, n, "F16", 1, shg, (uint64_t)rows * 2);
    }
    return 0;
}

/* ---- 执行一个 job: 算 + 写 ---- */
static int run_job(v41_stw *W, v41_st *S, const job_t *J, const cfg_t *C, acc_t *avq, acc_t *afp4) {
    double st[2] = {0, 0};
    if (J->kind == J_PASS) {
        const uint8_t *d = v41_st_data(S, J->w); if (!d) return -1;
        if (v41_stw_write(W, J->t[0], d, J->w->nbytes)) return -1;
        g_pass.bytes += J->w->nbytes; g_pass.params += v41_st_numel(J->w) * (!strcmp(J->w->dtype, "I8") ? 2 : 1);
        if (J->s) { d = v41_st_data(S, J->s); if (!d || v41_stw_write(W, J->t[1], d, J->s->nbytes)) return -1; g_pass.bytes += J->s->nbytes; }
        return 0;
    }
    if (J->kind == J_FP4) {
        int rows = (int)J->w->shape[0], cols = (int)J->w->shape[1];
        const uint8_t *w = v41_st_data(S, J->w), *s = J->s ? v41_st_data(S, J->s) : NULL;
        if (!w || (J->s && !s)) return -1;
        int sbr = J->s ? (int)(rows / J->s->shape[0]) : 1, sbc = J->s ? (int)(cols / J->s->shape[1]) : 1;
        uint8_t *pk = (uint8_t *)malloc((size_t)rows * cols / 2), *sc = (uint8_t *)malloc((size_t)rows * (cols / 32));
        int rc = v41_fp4_requant_host(w, s, J->w->dtype, rows, cols, sbr, sbc, pk, sc, st);
        if (rc) { free(pk); free(sc); fprintf(stderr, "★FP4 失败 %s rc=%d★\n", J->w->name, rc); return -1; }
        if (v41_stw_write(W, J->t[0], pk, (uint64_t)rows * cols / 2) || v41_stw_write(W, J->t[1], sc, (uint64_t)rows * (cols / 32))) return -1;
        free(pk); free(sc);
        afp4->sse += st[0]; afp4->en += st[1]; afp4->bytes += (uint64_t)rows * cols / 2 + (uint64_t)rows * (cols / 32); afp4->params += (long long)rows * cols;
        return 0;
    }
    /* VQ: 三矩阵一份码本 */
    const uint8_t *w[3], *s[3]; int rows[3], cols[3]; uint8_t *idx[3]; uint16_t *gain[3];
    int bits = bits_of(C->nc);
    for (int m = 0; m < 3; m++) {
        w[m] = v41_st_data(S, J->ew[m]); s[m] = v41_st_data(S, J->es[m]);
        if (!w[m] || !s[m]) return -1;
        rows[m] = (int)J->ew[m]->shape[0]; cols[m] = (int)J->ew[m]->shape[1] * 2;
        idx[m] = (uint8_t *)malloc((size_t)rows[m] * ((cols[m] / C->dim * bits + 7) / 8));
        gain[m] = (uint16_t *)malloc(sizeof(uint16_t) * rows[m]);
    }
    uint16_t *cb = (uint16_t *)malloc(sizeof(uint16_t) * C->nc * C->dim);
    int rc = v41_vq_expert_from_fp4(w, s, rows, cols, 3, C->dim, C->nc, C->iters, C->stride, idx, cb, gain, st);
    if (rc) { fprintf(stderr, "★VQ 失败 %s rc=%d★\n", J->ew[0]->name, rc); return -1; }
    if (v41_stw_write(W, J->t[0], cb, (uint64_t)C->nc * C->dim * 2)) return -1;
    for (int m = 0; m < 3; m++) {
        uint64_t nb = (uint64_t)rows[m] * ((cols[m] / C->dim * bits + 7) / 8);
        if (v41_stw_write(W, J->t[1 + m * 2], idx[m], nb) || v41_stw_write(W, J->t[2 + m * 2], gain[m], (uint64_t)rows[m] * 2)) return -1;
        avq->bytes += nb + (uint64_t)rows[m] * 2; avq->params += (long long)rows[m] * cols[m];
        free(idx[m]); free(gain[m]);
    }
    avq->bytes += (uint64_t)C->nc * C->dim * 2;
    avq->sse += st[0]; avq->en += st[1];
    free(cb);
    return 0;
}

static const char *meta_json(const cfg_t *C) {
    static char m[512];
    snprintf(m, sizeof m, "\"format\":\"pt\",\"ds4_v41_recipe\":\"routed=vq%dx%d skeleton=fp4_1x32 mtp=fp4_passthrough engram=hf_sidecar nocal\","
             "\"vq_dim\":\"%d\",\"vq_nc\":\"%d\",\"vq_iters\":\"%d\",\"vq_stride\":\"%d\"", C->dim, C->nc, C->dim, C->nc, C->iters, C->stride);
    return m;
}
static double cosv(const acc_t *a) { return a->en > 0 ? sqrt(fmax(0.0, 1.0 - a->sse / a->en)) : 1.0; }

/* 一个分片(某一层, 或 L=-1 的 common) */
static int do_shard(v41_st *S, const cfg_t *C, const char *out, int L, int nexp) {
    char path[4200];
    if (L >= 0) snprintf(path, sizeof path, "%s/model-layer%02d.safetensors", out, L);
    else snprintf(path, sizeof path, "%s/model-common.safetensors", out);
    struct stat sb;
    if (!C->force && !stat(path, &sb)) { printf("[跳过] %s 已完成 (%.2f GB)\n", path, sb.st_size / 1e9); return 0; }
    double t0 = now_s();
    v41_stw W; if (v41_stw_begin(&W, path)) return -1;
    job_t *jobs = (job_t *)calloc((size_t)nexp + 8192, sizeof(job_t)); int nj = 0;
    char pre[32]; int plen = 0;
    if (L >= 0) { plen = snprintf(pre, sizeof pre, "layers.%d.", L); for (int e = 0; e < nexp; e++) if (plan_vq(&W, &jobs[nj++], S, L, e, C)) return -1; }
    for (int i = 0; i < S->ne; i++) {
        const v41_st_ent *E = &S->e[i];
        if (L >= 0 ? strncmp(E->name, pre, (size_t)plen) != 0 : !strncmp(E->name, "layers.", 7)) continue;
        if (strstr(E->name, ".engram.") || ends_with(E->name, ".scale")) continue;
        if (L >= 0 && is_routed_expert(E->name)) continue;           /* 已在 VQ 计划里 */
        if (nj >= nexp + 8192) { fprintf(stderr, "★job 表满★\n"); return -1; }
        if (wants_fp4(E) && !(!strncmp(E->name, "mtp.", 4) && strstr(E->name, ".ffn.experts."))) { if (plan_fp4(&W, &jobs[nj++], S, E)) return -1; }
        else plan_pass(&W, &jobs[nj++], S, E);
    }
    if (v41_stw_write_header(&W, meta_json(C))) return -1;
    acc_t avq = {0, 0, 0, 0}, afp4 = {0, 0, 0, 0};
    int nvq = 0;
    for (int j = 0; j < nj; j++) {
        if (run_job(&W, S, &jobs[j], C, &avq, &afp4)) { v41_stw_abort(&W); return -1; }
        if (jobs[j].kind == J_VQ && (++nvq % 64 == 0 || nvq == nexp))
            printf("  L%02d 专家 %3d/%d  %6.0fs  VQ 残差 %.2f%% (cos %.4f)\n", L, nvq, nexp, now_s() - t0, 100.0 * avq.sse / fmax(avq.en, 1e-30), cosv(&avq));
    }
    if (v41_stw_end(&W)) return -1;
    g_vq.sse += avq.sse; g_vq.en += avq.en; g_vq.bytes += avq.bytes; g_vq.params += avq.params;
    g_fp4.sse += afp4.sse; g_fp4.en += afp4.en; g_fp4.bytes += afp4.bytes; g_fp4.params += afp4.params;
    printf("[分片] %s  %.3f GB  %.0fs  | VQ %.3f B 参数 cos %.4f %.4f bpw | FP4 %.3f B 参数 cos %.4f\n",
           path + strlen(out) + 1, W.total / 1e9, now_s() - t0, avq.params / 1e9, cosv(&avq),
           avq.params ? avq.bytes * 8.0 / avq.params : 0.0, afp4.params / 1e9, cosv(&afp4));
    fflush(stdout);
    free(jobs);
    v41_st_release_idle(S);
    return 0;
}

/* index.json 按【盘上实际文件】生成(含软链进来的 engram 分片), 续跑/重跑都对得上 */
static int write_index(const char *out) {
    v41_st O; if (v41_st_open(&O, out)) return -1;
    char p[4200]; snprintf(p, sizeof p, "%s/model.safetensors.index.json", out);
    FILE *f = fopen(p, "w"); if (!f) return -1;
    uint64_t tot = 0; for (int i = 0; i < O.ne; i++) tot += O.e[i].nbytes;
    fprintf(f, "{\"metadata\":{\"total_size\":%llu},\"weight_map\":{", (unsigned long long)tot);
    for (int i = 0; i < O.ne; i++) fprintf(f, "%s\"%s\":\"%s\"", i ? "," : "", O.e[i].name, O.sh[O.e[i].shard].file);
    fprintf(f, "}}\n"); fclose(f);
    printf("[index] %d 个张量 / %d 个文件 / 总字节 %.3f GB → %s\n", O.ne, O.nsh, tot / 1e9, p);
    v41_st_close(&O);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "用法: v41_quantize <hf-dir> <out-dir> [--vq-dim 8] [--vq-nc 4096] [--vq-iters 8] [--vq-stride 8] [--layers a:b] [--no-common] [--force]\n"); return 2; }
    cfg_t C = {8, 4096, 8, 8, 0, -1, 1, 0};
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--vq-dim") && i + 1 < argc) C.dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vq-nc") && i + 1 < argc) C.nc = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vq-iters") && i + 1 < argc) C.iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--vq-stride") && i + 1 < argc) C.stride = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layers") && i + 1 < argc) { if (sscanf(argv[++i], "%d:%d", &C.l0, &C.l1) != 2) { fprintf(stderr, "★--layers 要 a:b★\n"); return 2; } }
        else if (!strcmp(argv[i], "--no-common")) C.common = 0;
        else if (!strcmp(argv[i], "--force")) C.force = 1;
        else { fprintf(stderr, "★不认识的参数 %s★\n", argv[i]); return 2; }
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    v41_st S; double t0 = now_s();
    if (v41_st_open(&S, argv[1])) return 1;
    int nlayers = 0, nexp = 0;
    for (int i = 0; i < S.ne; i++) {
        int L, e;
        if (sscanf(S.e[i].name, "layers.%d.ffn.experts.%d.w1.weight", &L, &e) == 2) { if (L + 1 > nlayers) nlayers = L + 1; if (L == 0 && e + 1 > nexp) nexp = e + 1; }
        else if (sscanf(S.e[i].name, "layers.%d.", &L) == 1 && L + 1 > nlayers) nlayers = L + 1;
    }
    if (C.l1 < 0) C.l1 = nlayers;
    printf("[索引] %d 个张量 / %d 分片, %.1fs; %d 层 × %d 专家; 配方 VQ dim%d×nc%d iters%d stride%d, 层 [%d,%d)%s\n",
           S.ne, S.nsh, now_s() - t0, nlayers, nexp, C.dim, C.nc, C.iters, C.stride, C.l0, C.l1, C.common ? " + common" : "");
    if (mkdir(argv[2], 0775) && errno != EEXIST) { fprintf(stderr, "★建不了 %s★\n", argv[2]); return 1; }
    for (int L = C.l0; L < C.l1; L++) if (do_shard(&S, &C, argv[2], L, nexp)) { fprintf(stderr, "★第 %d 层失败, 停车★\n", L); return 1; }
    if (C.common && do_shard(&S, &C, argv[2], -1, nexp)) return 1;
    v41_st_close(&S);
    if (write_index(argv[2])) return 1;
    printf("[总账] routed VQ: %.3f B 参数 → %.3f GB (%.4f bpw) cos %.4f\n", g_vq.params / 1e9, g_vq.bytes / 1e9,
           g_vq.params ? g_vq.bytes * 8.0 / g_vq.params : 0.0, cosv(&g_vq));
    printf("[总账] 骨架 FP4: %.3f B 参数 → %.3f GB (4.25 bpw) cos %.4f\n", g_fp4.params / 1e9, g_fp4.bytes / 1e9, cosv(&g_fp4));
    printf("[总账] 透传: %.3f B 参数 → %.3f GB\n", g_pass.params / 1e9, g_pass.bytes / 1e9);
    printf("[总账] 本程序写出 %.3f GB (engram 203 GB 软链不计), 用时 %.0fs\n", (g_vq.bytes + g_fp4.bytes + g_pass.bytes) / 1e9, now_s() - t0);
    return 0;
}
