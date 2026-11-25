/* core_v41_exit.c — Stage B 出口头 FP4x32 残差(head_fp4.bin) + 平滑出口补偿(exit_ctrl.bin)(2026-11-25, Round 126)。
 *
 * 原理(fable5 11-25 Round 124~126): 1017 底座 output.weight 是 q4_K(相对误差 7.15%, 独占 −3.11 pp Same), 且位于全部 40 层之后 ——
 * 修出口零 MoE 路由翻转。head_fp4.bin 存原厂 BF16 head.weight 减 q4_K 解码的 FP4x32 残差(351.64 MB, 恢复 98.83% 方差);
 * exit_ctrl.bin 存缩放系数 alpha_head 与 f32 平滑补偿 dh = (dg ? xn ⊙ dg : 0) + (xn·B)·A, 独立 f32 GEMM 分支规避 q4_K 出口的 bf16 激活截断。
 * 暂存缓冲(CH=256 行)在 v41_exit_load 时一次分配到位, 前向/解码 CUDA Graph 捕获期间绝不重分配。 */
#include "core_v41_exit.h"
#include "../common/ds4_quantfmt.h"
#ifndef DS4_NO_GPU
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static bool f32_finite(float x) { uint32_t u; memcpy(&u, &x, 4); return (u & 0x7f800000u) != 0x7f800000u; }

#define V41_EXIT_CHUNK 256u

static struct {
    uint8_t *buf;
    uint64_t bytes, fnv;
    float file_alpha;
    int loaded;
    char src_path[4200];
} g_head;

static struct {
    float alpha_head;
    uint32_t rank, flags;
    int has_ctrl, on;
    ds4_gpu_tensor *dg, *A, *B;
} g_ctrl;

static ds4_gpu_tensor *g_tmp_log, *g_tmp_dh, *g_tmp_T, *g_tmp_bwd;
static uint64_t g_cap_bwd;

static void free_tmp_all(void) {
    if (g_tmp_log) { ds4_gpu_tensor_free(g_tmp_log); g_tmp_log = NULL; }
    if (g_tmp_dh)  { ds4_gpu_tensor_free(g_tmp_dh);  g_tmp_dh  = NULL; }
    if (g_tmp_T)   { ds4_gpu_tensor_free(g_tmp_T);   g_tmp_T   = NULL; }
    if (g_tmp_bwd) { ds4_gpu_tensor_free(g_tmp_bwd); g_tmp_bwd = NULL; }
    g_cap_bwd = 0;
}

void v41_exit_clear(void) {
    if (g_ctrl.dg) { ds4_gpu_tensor_free(g_ctrl.dg); g_ctrl.dg = NULL; }
    if (g_ctrl.A)  { ds4_gpu_tensor_free(g_ctrl.A);  g_ctrl.A  = NULL; }
    if (g_ctrl.B)  { ds4_gpu_tensor_free(g_ctrl.B);  g_ctrl.B  = NULL; }
    memset(&g_ctrl, 0, sizeof g_ctrl);
    g_head.loaded = 0;
    free_tmp_all();
}

static bool mount_head_file(const char *p) {
    FILE *f = fopen(p, "rb");
    if (!f) return false;
    pt124_head_hdr H;
    const uint64_t want = 64ull + (uint64_t)DS4_N_VOCAB * (DS4_N_EMBD / 32ull) * 17ull;
    if (fread(&H, 1, sizeof H, f) != sizeof H || H.magic != PT124_HEAD_MAGIC || H.version != PT124_VER ||
        H.rows != DS4_N_VOCAB || H.cols != DS4_N_EMBD || H.total_bytes != want || !f32_finite(H.alpha)) {
        fprintf(stderr, "ds4: ★[pt124] head_fp4.bin 头或尺寸不符: %s★\n", p);
        fclose(f); return false;
    }
    if (g_head.buf && g_head.bytes == want && g_head.fnv == H.fnv64) {
        fclose(f);
        g_head.file_alpha = H.alpha;
        g_head.loaded = 1;
        snprintf(g_head.src_path, sizeof g_head.src_path, "%s", p);
        return true;
    }
    uint8_t *raw = NULL;
    if (posix_memalign((void **)&raw, 64, (size_t)want) != 0 || !raw) { fclose(f); return false; }
    memcpy(raw, &H, sizeof H);
    if (fread(raw + sizeof H, 1, (size_t)(want - sizeof H), f) != (size_t)(want - sizeof H)) {
        fprintf(stderr, "ds4: ★[pt124] head_fp4.bin 截断: %s★\n", p);
        free(raw); fclose(f); return false;
    }
    fclose(f);
    if (!ds4_gpu_register_aux_model_map(raw, want)) {
        fprintf(stderr, "ds4: ★[pt124] register_aux_model_map 失败: %s★\n", p);
        free(raw); return false;
    }
    free(g_head.buf);
    g_head.buf = raw; g_head.bytes = want; g_head.fnv = H.fnv64;
    g_head.file_alpha = H.alpha; g_head.loaded = 1;
    snprintf(g_head.src_path, sizeof g_head.src_path, "%s", p);
    return true;
}

int v41_exit_load(const char *dir, double *mb_io) {
    v41_exit_clear();
    if (!dir || !dir[0]) return 0;
    int n_loaded = 0;
    bool has_head = false;
    char hp[4200]; snprintf(hp, sizeof hp, "%s/head_fp4.bin", dir);
    FILE *tf = fopen(hp, "rb");
    if (tf) {
        fclose(tf);
        if (!mount_head_file(hp)) return -1;
        has_head = true;
        n_loaded++;
        if (mb_io) *mb_io += (double)g_head.bytes / 1e6;
    }
    float alpha = has_head ? g_head.file_alpha : 0.f;
    char cp[4200]; snprintf(cp, sizeof cp, "%s/exit_ctrl.bin", dir);
    FILE *fc = fopen(cp, "rb");
    if (fc) {
        pt124_exit_hdr C;
        if (fread(&C, 1, sizeof C, fc) != sizeof C || C.magic != PT124_EXIT_MAGIC || C.version != PT124_VER ||
            C.rank > PT124_MAX_RANK || !f32_finite(C.alpha_head)) {
            fprintf(stderr, "ds4: ★[pt124] exit_ctrl.bin 头不符: %s★\n", cp);
            fclose(fc); v41_exit_clear(); return -1;
        }
        const uint32_t E = DS4_N_EMBD, R = C.rank, has_dg = (C.flags & 1u) ? 1u : 0u;
        const uint64_t want = 64ull + ((uint64_t)(has_dg ? E : 0u) + 2ull * R * E) * 4ull;
        if (C.total_bytes != want) { fclose(fc); v41_exit_clear(); return -1; }
        const size_t max_el = R > 0u ? (size_t)R * E : (size_t)E;
        float *buf = xmalloc(max_el * 4u);
        bool ok = true;
        if (has_dg) {
            g_ctrl.dg = ds4_gpu_tensor_alloc((uint64_t)E * 4u);
            ok = g_ctrl.dg && fread(buf, 4, E, fc) == E && ds4_gpu_tensor_write(g_ctrl.dg, 0, buf, (uint64_t)E * 4u) != 0;
        }
        if (ok && R > 0u) {
            const uint64_t re = (uint64_t)R * E;
            g_ctrl.A = ds4_gpu_tensor_alloc(re * 4u);
            g_ctrl.B = ds4_gpu_tensor_alloc(re * 4u);
            ok = g_ctrl.A && g_ctrl.B &&
                 fread(buf, 4, (size_t)re, fc) == (size_t)re && ds4_gpu_tensor_write(g_ctrl.A, 0, buf, re * 4u) != 0 &&
                 fread(buf, 4, (size_t)re, fc) == (size_t)re && ds4_gpu_tensor_write(g_ctrl.B, 0, buf, re * 4u) != 0;
        }
        free(buf); fclose(fc);
        if (!ok) { v41_exit_clear(); return -1; }
        alpha = C.alpha_head;
        g_ctrl.rank = R; g_ctrl.flags = C.flags;
        g_ctrl.has_ctrl = (has_dg || R > 0u) ? 1 : 0;
        n_loaded++;
        if (mb_io) *mb_io += (double)want / 1e6;
    }
    if (!n_loaded) return 0;
    if (alpha != 0.f && !has_head) {
        fprintf(stderr, "ds4: ★[pt124] exit_ctrl 要求 alpha_head=%.4f 但缺 head_fp4.bin★\n", (double)alpha);
        v41_exit_clear(); return -1;
    }
    g_ctrl.alpha_head = alpha;
    g_ctrl.on = ((g_head.loaded && alpha != 0.f) || g_ctrl.has_ctrl) ? 1 : 0;
    if (g_ctrl.on) {
        g_tmp_log = ds4_gpu_tensor_alloc((uint64_t)V41_EXIT_CHUNK * DS4_N_VOCAB * 4u);
        if (!g_tmp_log) { v41_exit_clear(); return -1; }
        if (g_ctrl.has_ctrl) {
            g_tmp_dh = ds4_gpu_tensor_alloc((uint64_t)V41_EXIT_CHUNK * DS4_N_EMBD * 4u);
            if (!g_tmp_dh) { v41_exit_clear(); return -1; }
            if (g_ctrl.rank > 0u) {
                g_tmp_T = ds4_gpu_tensor_alloc((uint64_t)V41_EXIT_CHUNK * g_ctrl.rank * 4u);
                if (!g_tmp_T) { v41_exit_clear(); return -1; }
            }
        }
    }
    fprintf(stderr, "ds4: [pt124] loaded dir=%s head=%d alpha=%.4f ctrl=%d rank=%u\n",
            dir, has_head ? 1 : 0, (double)alpha, g_ctrl.has_ctrl, g_ctrl.rank);
    return n_loaded;
}

int v41_exit_save(const char *dir) {
    if (!dir || !dir[0]) return -1;
    int n = 0;
    if (g_head.loaded && g_head.buf) {
        char hp[4200]; snprintf(hp, sizeof hp, "%s/head_fp4.bin", dir);
        (void)unlink(hp);
        if (!g_head.src_path[0] || link(g_head.src_path, hp) != 0) {
            FILE *f = fopen(hp, "wb");
            if (!f || fwrite(g_head.buf, 1, (size_t)g_head.bytes, f) != (size_t)g_head.bytes || fclose(f) != 0) {
                if (f) fclose(f);
                return -1;
            }
        }
        n++;
    }
    if (g_ctrl.has_ctrl || (g_head.loaded && g_ctrl.alpha_head != g_head.file_alpha)) {
        const uint32_t E = DS4_N_EMBD, R = g_ctrl.rank, has_dg = g_ctrl.dg ? 1u : 0u;
        pt124_exit_hdr C; memset(&C, 0, sizeof C);
        C.magic = PT124_EXIT_MAGIC; C.version = PT124_VER; C.rank = R; C.flags = has_dg;
        C.alpha_head = g_ctrl.alpha_head;
        C.total_bytes = 64ull + ((uint64_t)(has_dg ? E : 0u) + 2ull * R * E) * 4ull;
        char cp[4200]; snprintf(cp, sizeof cp, "%s/exit_ctrl.bin", dir);
        FILE *f = fopen(cp, "wb");
        if (!f) return -1;
        const size_t max_el = R > 0u ? (size_t)R * E : (size_t)E;
        float *buf = xmalloc(max_el * 4u);
        bool ok = fwrite(&C, 1, sizeof C, f) == sizeof C;
        if (ok && has_dg) ok = ds4_gpu_tensor_read(g_ctrl.dg, 0, buf, (uint64_t)E * 4u) && fwrite(buf, 4, E, f) == E;
        if (ok && R > 0u) {
            const uint64_t re = (uint64_t)R * E;
            ok = ds4_gpu_tensor_read(g_ctrl.A, 0, buf, re * 4u) && fwrite(buf, 4, (size_t)re, f) == (size_t)re &&
                 ds4_gpu_tensor_read(g_ctrl.B, 0, buf, re * 4u) && fwrite(buf, 4, (size_t)re, f) == (size_t)re;
        }
        free(buf);
        if (fclose(f) != 0) ok = false;
        if (!ok) return -1;
        n++;
    }
    return n;
}

bool v41_exit_active(void) { return g_ctrl.on != 0; }

bool v41_exit_apply(const ds4_model *mdl, const ds4_tensor *w_out,
                    ds4_gpu_tensor *logits, const ds4_gpu_tensor *xn, uint32_t n) {
    if (!g_ctrl.on || n == 0u) return true;
    const uint32_t E = DS4_N_EMBD, V = DS4_N_VOCAB, CH = V41_EXIT_CHUNK;
    for (uint32_t r0 = 0; r0 < n; r0 += CH) {
        const uint32_t nr = (n - r0 < CH) ? (n - r0) : CH;
        ds4_gpu_tensor *vx = ds4_gpu_tensor_view(xn, (uint64_t)r0 * E * 4u, (uint64_t)nr * E * 4u);
        ds4_gpu_tensor *vz = ds4_gpu_tensor_view(logits, (uint64_t)r0 * V * 4u, (uint64_t)nr * V * 4u);
        bool ok = vx && vz;
        if (ok && g_head.loaded && g_ctrl.alpha_head != 0.f) {
            ok = ds4_gpu_v41_matmul_fp4x32_tensor(g_tmp_log, g_head.buf, g_head.bytes, 64ull, E, V, vx, nr, 0) &&
                 ds4_gpu_bwd_axpy_tensor(vz, g_tmp_log, g_ctrl.alpha_head, (uint64_t)nr * V);
        }
        if (ok && g_ctrl.has_ctrl) {
            ds4_gpu_tensor *vdh = ds4_gpu_tensor_view(g_tmp_dh, 0, (uint64_t)nr * E * 4u);
            ok = vdh &&
                 ds4_gpu_v41_exit_delta_h_tensor(vdh, g_tmp_T, vx, g_ctrl.dg, g_ctrl.A, g_ctrl.B, nr, E, g_ctrl.rank) &&
                 v41_tproj(mdl, g_tmp_log, w_out, E, V, vdh, nr, 0) &&
                 ds4_gpu_bwd_axpy_tensor(vz, g_tmp_log, 1.0f, (uint64_t)nr * V);
            if (ok && g_head.loaded && g_ctrl.alpha_head != 0.f) {
                ok = ds4_gpu_v41_matmul_fp4x32_tensor(g_tmp_log, g_head.buf, g_head.bytes, 64ull, E, V, vdh, nr, 0) &&
                     ds4_gpu_bwd_axpy_tensor(vz, g_tmp_log, g_ctrl.alpha_head, (uint64_t)nr * V);
            }
            if (vdh) ds4_gpu_tensor_free(vdh);
        }
        if (vx) ds4_gpu_tensor_free(vx);
        if (vz) ds4_gpu_tensor_free(vz);
        if (!ok) return false;
    }
    return true;
}

bool v41_exit_bwd_proj(const ds4_model *mdl, const ds4_tensor *w_out,
                       ds4_gpu_tensor *g_xn, const ds4_gpu_tensor *glog, uint32_t m) {
    const uint32_t E = DS4_N_EMBD, V = DS4_N_VOCAB;
    if (!ds4_gpu_bwd_matmul_t_tensor(g_xn, mdl->map, mdl->size, w_out->type, w_out->abs_offset, E, V, glog, m, 0))
        return false;
    if (!g_ctrl.on) return true;
    const uint64_t need = (uint64_t)m * (E + (g_ctrl.rank > 0u ? g_ctrl.rank : 0u)) * 4u;
    if (need > g_cap_bwd) {
        if (g_tmp_bwd) ds4_gpu_tensor_free(g_tmp_bwd);
        g_tmp_bwd = ds4_gpu_tensor_alloc(need);
        g_cap_bwd = g_tmp_bwd ? need : 0;
        if (!g_tmp_bwd) return false;
    }
    ds4_gpu_tensor *gtmp = ds4_gpu_tensor_view(g_tmp_bwd, 0, (uint64_t)m * E * 4u);
    if (!gtmp) return false;
    bool ok = true;
    if (g_head.loaded && g_ctrl.alpha_head != 0.f) {
        ok = ds4_gpu_bwd_matmul_t_tensor(gtmp, g_head.buf, g_head.bytes, DS4_GGT_FP4X32, 64ull, E, V, glog, m, 0) &&
             ds4_gpu_bwd_axpy_tensor(g_xn, gtmp, g_ctrl.alpha_head, (uint64_t)m * E);
    }
    if (ok && g_ctrl.has_ctrl) {
        ds4_gpu_tensor *vt = g_ctrl.rank > 0u ? ds4_gpu_tensor_view(g_tmp_bwd, (uint64_t)m * E * 4u, (uint64_t)m * g_ctrl.rank * 4u) : NULL;
        ok = (!g_ctrl.rank || vt != NULL) &&
             ds4_gpu_v41_exit_delta_h_tensor(gtmp, vt, g_xn, g_ctrl.dg, g_ctrl.B, g_ctrl.A, m, E, g_ctrl.rank) &&
             ds4_gpu_bwd_axpy_tensor(g_xn, gtmp, 1.0f, (uint64_t)m * E);
        if (vt) ds4_gpu_tensor_free(vt);
    }
    ds4_gpu_tensor_free(gtmp);
    return ok;
}
#endif
typedef int ds4_core_v41_exit_nonempty_tu;
