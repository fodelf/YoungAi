/* zsolve — P2 管线步骤 4/6: 激活对 (X, R) → 闭式 ds4_z_solve → corr 侧车 GGUF。
 *
 * 输入: 每层一对 float32 C-order .npy —— X (n × d_model = ffn_in), R (n × d_model
 *       = o_ref − o_base 残差目标)。
 * 输出: 引擎 corr_load 可直接挂载的 GGUF (ds4.corr.present=true +
 *       blk.N.corr_{U,V,C,b,beta,delta})。corr 语义 out += U@(C[e]⊙(V@x))+b+beta[e];
 *       v0 全专家共享一个 z: C 的每行(专家)都写同一 z 向量, b/beta/delta 置零。
 *
 * 用法: zsolve --out corr.gguf --rank 32 --lambda 1e-3 \
 *              --layer 23 X23.npy R23.npy [--layer 24 X24.npy R24.npy ...]
 *
 * 布局对接 (ds4_corr.c corr_tensor 校验):
 *   corr_U: ne=(d_l, d_model)  行主 d_model 行 × d_l 连续  = ds4_z 的 U[j*rank+c] 原样
 *   corr_V: ne=(d_model, d_l)  行主 d_l 行 × d_model 连续  = ds4_z 的 V[i*rank+c] 转置
 *   corr_C: ne=(d_l, 256)      行主 256 行 × d_l 连续      = 每行 z[0..d_l)
 */
#include "../ds4_z.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define D_MODEL 4096u
#define N_EXPERT 256u
#define N_EXPERT_USED 6u   /* 运行时每 token 选中数; C 共享 z 需按此折算 */
#define GGUF_ALIGN 32u

/* ---- 极简 npy reader: float32/'<f4' C-order 2 维 ---- */
static float *npy_load_2d(const char *path, uint32_t *rows, uint32_t *cols) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "zsolve: cannot open %s\n", path); return NULL; }
    unsigned char magic[8];
    if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "zsolve: %s is not npy\n", path); fclose(fp); return NULL;
    }
    uint32_t hlen = 0;
    if (magic[6] == 1) { unsigned char b[2]; if (fread(b,1,2,fp)!=2){fclose(fp);return NULL;} hlen = b[0] | (b[1]<<8); }
    else { unsigned char b[4]; if (fread(b,1,4,fp)!=4){fclose(fp);return NULL;} hlen = b[0]|(b[1]<<8)|(b[2]<<16)|((uint32_t)b[3]<<24); }
    char *hdr = malloc(hlen + 1);
    if (!hdr || fread(hdr, 1, hlen, fp) != hlen) { free(hdr); fclose(fp); return NULL; }
    hdr[hlen] = 0;
    if (!strstr(hdr, "'<f4'") || strstr(hdr, "'fortran_order': True")) {
        fprintf(stderr, "zsolve: %s must be little-endian float32 C-order (header: %s)\n", path, hdr);
        free(hdr); fclose(fp); return NULL;
    }
    const char *sh = strstr(hdr, "'shape':");
    unsigned long r = 0, c = 0;
    if (!sh || sscanf(sh, "'shape': (%lu, %lu", &r, &c) != 2 || !r || !c) {
        fprintf(stderr, "zsolve: %s: cannot parse 2-D shape\n", path);
        free(hdr); fclose(fp); return NULL;
    }
    free(hdr);
    float *data = malloc((size_t)r * c * sizeof(float));
    if (!data || fread(data, sizeof(float), (size_t)r * c, fp) != (size_t)r * c) {
        fprintf(stderr, "zsolve: %s: truncated data\n", path);
        free(data); fclose(fp); return NULL;
    }
    fclose(fp);
    *rows = (uint32_t)r; *cols = (uint32_t)c;
    return data;
}

/* ---- 极简 GGUF v3 writer ---- */
typedef struct { char name[96]; uint32_t ndim; uint64_t ne[2]; uint64_t off; const float *data; uint64_t count; } wtensor;
static void wstr(FILE *fp, const char *s) { uint64_t n = strlen(s); fwrite(&n, 8, 1, fp); fwrite(s, 1, n, fp); }

int main(int argc, char **argv) {
    const char *out_path = NULL;
    uint32_t rank = 32; float lambda = 1e-3f;
    struct { uint32_t layer; const char *xp, *rp; } jobs[64]; int njobs = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--rank") && i + 1 < argc) rank = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lambda") && i + 1 < argc) lambda = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--layer") && i + 3 < argc && njobs < 64) {
            jobs[njobs].layer = (uint32_t)atoi(argv[++i]);
            jobs[njobs].xp = argv[++i];
            jobs[njobs].rp = argv[++i];
            njobs++;
        } else { fprintf(stderr, "用法: zsolve --out corr.gguf [--rank K] [--lambda F] --layer N X.npy R.npy ...\n"); return 2; }
    }
    if (!out_path || !njobs) { fprintf(stderr, "zsolve: 需要 --out 和至少一组 --layer\n"); return 2; }

    wtensor tens[64 * 6]; int nt = 0;
    for (int jb = 0; jb < njobs; jb++) {
        uint32_t xn, xd, rn, rd;
        float *X = npy_load_2d(jobs[jb].xp, &xn, &xd);
        float *R = npy_load_2d(jobs[jb].rp, &rn, &rd);
        if (!X || !R) return 1;
        if (xd != D_MODEL || rd != D_MODEL || xn != rn) {
            fprintf(stderr, "zsolve: L%u 形状不符 X(%u,%u) R(%u,%u), 需 (n,%u) 对齐\n",
                    jobs[jb].layer, xn, xd, rn, rd, D_MODEL);
            return 1;
        }
        fprintf(stderr, "zsolve: L%u n=%u rank=%u 求解中 (O(n·d²+d³), 数分钟级)…\n", jobs[jb].layer, xn, rank);
        ds4_z *z = ds4_z_solve(X, R, xn, D_MODEL, D_MODEL, rank, lambda);
        free(X); free(R);
        if (!z) { fprintf(stderr, "zsolve: L%u 求解失败\n", jobs[jb].layer); return 1; }
        const uint32_t d_l = z->rank;
        /* corr_U (d_l, d_model): ds4_z U 行主 (d_model × rank) 原样 */
        float *U = malloc((size_t)D_MODEL * d_l * sizeof(float));
        memcpy(U, z->U, (size_t)D_MODEL * d_l * sizeof(float));
        /* corr_V (d_model, d_l): ds4_z V (d_in × rank) 转置成 (rank 行 × d_model 连续) */
        float *V = malloc((size_t)d_l * D_MODEL * sizeof(float));
        for (uint32_t c = 0; c < d_l; c++)
            for (uint32_t i = 0; i < D_MODEL; i++)
                V[(size_t)c * D_MODEL + i] = z->V[(size_t)i * z->rank + c];
        /* corr_C (d_l, 256): 每个专家一行, v0 全写同一 z。
         * 运行时 kernel (kernel_dsv4_corr_apply) 对每个选中专家各累加一次
         * (Σ_{s<n_used} U@(C[e]⊙Vx)), 而闭式解按"每 token 一次"拟合 —— 共享 z
         * 必须除以 n_used(6), 否则 6 倍过冲 (实测直接打崩生成, 乱码)。 */
        float *C = malloc((size_t)N_EXPERT * d_l * sizeof(float));
        for (uint32_t i = 0; i < d_l; i++) z->z[i] /= (float)N_EXPERT_USED;
        for (uint32_t e = 0; e < N_EXPERT; e++)
            memcpy(C + (size_t)e * d_l, z->z, d_l * sizeof(float));
        float *b = calloc(D_MODEL, sizeof(float));
        float *beta = calloc(N_EXPERT, sizeof(float));
        float *delta = calloc(N_EXPERT, sizeof(float));
        const uint32_t L = jobs[jb].layer;
        #define PUSH(suffix, nd, e0, e1, ptr, cnt) do { \
            wtensor *t = &tens[nt++]; \
            snprintf(t->name, sizeof(t->name), "blk.%u.corr_%s", L, suffix); \
            t->ndim = (nd); t->ne[0] = (e0); t->ne[1] = (e1); t->data = (ptr); t->count = (cnt); \
        } while (0)
        PUSH("U",     2, d_l,     D_MODEL,  U,     (uint64_t)d_l * D_MODEL);
        PUSH("V",     2, D_MODEL, d_l,      V,     (uint64_t)d_l * D_MODEL);
        PUSH("C",     2, d_l,     N_EXPERT, C,     (uint64_t)d_l * N_EXPERT);
        PUSH("b",     1, D_MODEL, 0,        b,     D_MODEL);
        PUSH("beta",  1, N_EXPERT, 0,       beta,  N_EXPERT);
        PUSH("delta", 1, N_EXPERT, 0,       delta, N_EXPERT);
        #undef PUSH
        uint32_t zr = 0; const float *zv = ds4_z_values(z, &zr);
        fprintf(stderr, "zsolve: L%u 完成, z 谱头: %.4f %.4f %.4f …\n", L, zv[0],
                zr > 1 ? zv[1] : 0.0f, zr > 2 ? zv[2] : 0.0f);
        /* U/V/C/b/beta/delta 的内存交给 writer 阶段, 进程退出时回收 */
        ds4_z_free(z);
    }

    /* 布局: offset 按 GGUF_ALIGN 对齐 */
    uint64_t off = 0;
    for (int i = 0; i < nt; i++) {
        tens[i].off = off;
        off += tens[i].count * sizeof(float);
        off = (off + GGUF_ALIGN - 1) / GGUF_ALIGN * GGUF_ALIGN;
    }
    FILE *fp = fopen(out_path, "wb");
    if (!fp) { fprintf(stderr, "zsolve: cannot write %s\n", out_path); return 1; }
    const uint32_t magic = 0x46554747; /* GGUF */
    const uint32_t version = 3;
    const uint64_t n_tensors = (uint64_t)nt, n_kv = 1;
    fwrite(&magic, 4, 1, fp); fwrite(&version, 4, 1, fp);
    fwrite(&n_tensors, 8, 1, fp); fwrite(&n_kv, 8, 1, fp);
    wstr(fp, "ds4.corr.present");
    { uint32_t t = 7; unsigned char v = 1; fwrite(&t, 4, 1, fp); fwrite(&v, 1, 1, fp); }  /* bool true */
    for (int i = 0; i < nt; i++) {
        wstr(fp, tens[i].name);
        fwrite(&tens[i].ndim, 4, 1, fp);
        for (uint32_t d = 0; d < tens[i].ndim; d++) fwrite(&tens[i].ne[d], 8, 1, fp);
        uint32_t ty = 0;   /* GGML F32 */
        fwrite(&ty, 4, 1, fp);
        fwrite(&tens[i].off, 8, 1, fp);
    }
    /* data 区从对齐边界开始 */
    long pos = ftell(fp);
    long pad = (long)((pos + GGUF_ALIGN - 1) / GGUF_ALIGN * GGUF_ALIGN) - pos;
    for (long p = 0; p < pad; p++) fputc(0, fp);
    for (int i = 0; i < nt; i++) {
        long want = (long)tens[i].off - (long)(ftell(fp) - (pos + pad));
        for (long p = 0; p < want; p++) fputc(0, fp);
        fwrite(tens[i].data, sizeof(float), tens[i].count, fp);
    }
    fclose(fp);
    fprintf(stderr, "zsolve: 写出 %s (%d 张量, %d 层)\n", out_path, nt, njobs);
    return 0;
}
