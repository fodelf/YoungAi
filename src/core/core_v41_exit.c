/* core_v41_exit.c — Stage B 出口头 FP4x32 残差(head_fp4.bin) + 双项叠加出口补偿(exit_ctrl.bin)(2026-11-26, Round 126/144)。
 *
 * 原理(fable5 Round 124~126 & Round 142~144):
 *   1) 1017 底座 output.weight 是 q4_K(相对误差 7.15%), 且位于全部 40 层之后 —— 修出口零下游 MoE 路由翻转。
 *      head_fp4.bin 存原厂 BF16 head.weight 减 q4_K 解码的 FP4x32 残差(351.64 MB, 恢复 98.83% 方差)。
 *   2) exit_ctrl.bin 同时叠加保边际补偿 dh(x) = (dg ? x⊙dg : 0) + (x·Bᵀ)·A 与主子空间解耦项(flags & 16u):
 *      z = x·W_q4kᵀ + dh(x)·(W_q4k + alpha_head·dW)ᵀ + (alpha_perp·x + (x·B2ᵀ)·A2)·dWᵀ,
 *      其中默认 A2 = (alpha_head - alpha_perp)·B2(亦支持 flags & 32u 逐模态谱权重 w2 或 flags & 64u 显式 f16 A2)。
 *   3) 暂存缓冲(CH=256 行)在 v41_exit_load 时一次分配到位, 前向/解码 CUDA Graph 捕获期间绝不重分配。 */
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
    float alpha_head, alpha_perp;
    uint32_t rank, rank2, flags;
    int has_ctrl, has_sub, sub_nz, on;
    ds4_gpu_tensor *dg, *A, *B, *zero_dg, *A2, *B2;
    char src_path[4200];
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
    ds4_gpu_tensor *ts[] = { g_ctrl.dg, g_ctrl.A, g_ctrl.B, g_ctrl.zero_dg, g_ctrl.A2, g_ctrl.B2 };
    for (size_t i = 0; i < 6u; i++) if (ts[i]) ds4_gpu_tensor_free(ts[i]);
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

static bool read_f16_mat_to_gpu(FILE *fc, ds4_gpu_tensor *dst, float *buf, size_t n_el) {
    uint16_t *u = xmalloc(n_el * 2u);
    bool ok = dst && fread(u, 2, n_el, fc) == n_el;
    if (ok) {
        for (size_t i = 0; i < n_el; i++) buf[i] = f16_to_f32(u[i]);
        ok = ds4_gpu_tensor_write(dst, 0, buf, (uint64_t)n_el * 4u) != 0;
    }
    free(u);
    return ok;
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
        const uint32_t is_f16 = (C.flags & 4u) ? 1u : 0u, is_sub = (C.flags & 16u) ? 1u : 0u;
        const uint32_t R2 = is_sub ? C.pad[0] : 0u;
        const uint32_t has_a2 = (is_sub && (C.flags & 64u)) ? 1u : 0u;
        const uint32_t has_w2 = (is_sub && !has_a2 && (C.flags & 32u)) ? 1u : 0u;
        float a_perp = C.alpha_head;
        if (is_sub) memcpy(&a_perp, &C.pad[1], 4);
        if (R2 > PT124_MAX_RANK || !f32_finite(a_perp)) { fclose(fc); v41_exit_clear(); return -1; }
        const uint64_t re = (uint64_t)R * E, re2 = (uint64_t)R2 * E;
        const uint64_t ab_b = is_f16 ? re * 2ull : re * 4ull;
        const uint64_t sub_b = is_sub ? ((has_a2 ? 2ull : 1ull) * re2 * 2ull + (has_w2 ? (uint64_t)R2 * 4ull : 0ull)) : 0ull;
        const uint64_t want = 64ull + (has_dg ? (uint64_t)E * 4ull : 0ull) + 2ull * ab_b + sub_b;
        if (C.total_bytes != want) { fclose(fc); v41_exit_clear(); return -1; }
        const uint32_t rmax_f = R > R2 ? R : R2;
        const size_t max_el = rmax_f > 0u ? (size_t)rmax_f * E : (size_t)E;
        float *buf = xmalloc(max_el * 4u);
        bool ok = true;
        if (has_dg) {
            g_ctrl.dg = ds4_gpu_tensor_alloc((uint64_t)E * 4u);
            ok = g_ctrl.dg && fread(buf, 4, E, fc) == E && ds4_gpu_tensor_write(g_ctrl.dg, 0, buf, (uint64_t)E * 4u) != 0;
        }
        if (ok && R > 0u) {
            g_ctrl.A = ds4_gpu_tensor_alloc(re * 4u);
            g_ctrl.B = ds4_gpu_tensor_alloc(re * 4u);
            if (is_f16) {
                ok = read_f16_mat_to_gpu(fc, g_ctrl.A, buf, (size_t)re) &&
                     read_f16_mat_to_gpu(fc, g_ctrl.B, buf, (size_t)re);
            } else {
                ok = g_ctrl.A && g_ctrl.B &&
                     fread(buf, 4, (size_t)re, fc) == (size_t)re && ds4_gpu_tensor_write(g_ctrl.A, 0, buf, re * 4u) != 0 &&
                     fread(buf, 4, (size_t)re, fc) == (size_t)re && ds4_gpu_tensor_write(g_ctrl.B, 0, buf, re * 4u) != 0;
            }
        }
        if (ok && is_sub && R2 > 0u) {
            memset(buf, 0, (size_t)E * 4u);
            g_ctrl.zero_dg = ds4_gpu_tensor_alloc((uint64_t)E * 4u);
            g_ctrl.A2 = ds4_gpu_tensor_alloc(re2 * 4u);
            g_ctrl.B2 = ds4_gpu_tensor_alloc(re2 * 4u);
            ok = g_ctrl.zero_dg && g_ctrl.A2 && g_ctrl.B2 &&
                 ds4_gpu_tensor_write(g_ctrl.zero_dg, 0, buf, (uint64_t)E * 4u) != 0;
            if (ok && has_a2) {
                ok = read_f16_mat_to_gpu(fc, g_ctrl.A2, buf, (size_t)re2) &&
                     read_f16_mat_to_gpu(fc, g_ctrl.B2, buf, (size_t)re2);
            } else if (ok) {
                ok = read_f16_mat_to_gpu(fc, g_ctrl.B2, buf, (size_t)re2);
                float *w2 = xmalloc((size_t)R2 * 4u);
                float *a2 = xmalloc((size_t)re2 * 4u);
                if (ok && has_w2) {
                    ok = fread(w2, 4, R2, fc) == R2;
                    for (uint32_t r = 0; ok && r < R2; r++) ok = f32_finite(w2[r]);
                } else {
                    const float dlt = C.alpha_head - a_perp;
                    for (uint32_t r = 0; r < R2; r++) w2[r] = dlt;
                }
                if (ok) {
                    for (uint32_t r = 0; r < R2; r++) {
                        const float wr = w2[r];
                        const float *br = buf + (size_t)r * E;
                        float *ar = a2 + (size_t)r * E;
                        for (uint32_t j = 0; j < E; j++) ar[j] = wr * br[j];
                    }
                    ok = ds4_gpu_tensor_write(g_ctrl.A2, 0, a2, re2 * 4u) != 0;
                }
                free(w2); free(a2);
            }
        }
        free(buf); fclose(fc);
        if (!ok) { v41_exit_clear(); return -1; }
        alpha = C.alpha_head;
        g_ctrl.rank = R; g_ctrl.rank2 = R2; g_ctrl.flags = C.flags;
        g_ctrl.alpha_perp = a_perp;
        g_ctrl.has_ctrl = (has_dg || R > 0u) ? 1 : 0;
        g_ctrl.has_sub = is_sub ? 1 : 0;
        g_ctrl.sub_nz = (is_sub && R2 > 0u && (has_a2 || has_w2 || C.alpha_head != a_perp)) ? 1 : 0;
        snprintf(g_ctrl.src_path, sizeof g_ctrl.src_path, "%s", cp);
        n_loaded++;
        if (mb_io) *mb_io += (double)want / 1e6;
    }
    if (!n_loaded) return 0;
    if ((alpha != 0.f || g_ctrl.has_sub) && !has_head) {
        fprintf(stderr, "ds4: ★[pt124] exit_ctrl 要求 alpha_head=%.4f 但缺 head_fp4.bin★\n", (double)alpha);
        v41_exit_clear(); return -1;
    }
    g_ctrl.alpha_head = alpha;
    g_ctrl.on = ((g_head.loaded && (alpha != 0.f || g_ctrl.has_sub)) || g_ctrl.has_ctrl) ? 1 : 0;
    const uint32_t rmax = g_ctrl.rank > g_ctrl.rank2 ? g_ctrl.rank : g_ctrl.rank2;
    if (g_ctrl.on) {
        g_tmp_log = ds4_gpu_tensor_alloc((uint64_t)V41_EXIT_CHUNK * DS4_N_VOCAB * 4u);
        if (!g_tmp_log) { v41_exit_clear(); return -1; }
        if (g_ctrl.has_ctrl || g_ctrl.sub_nz) {
            g_tmp_dh = ds4_gpu_tensor_alloc((uint64_t)V41_EXIT_CHUNK * DS4_N_EMBD * 4u);
            if (!g_tmp_dh) { v41_exit_clear(); return -1; }
            if (rmax > 0u) {
                g_tmp_T = ds4_gpu_tensor_alloc((uint64_t)V41_EXIT_CHUNK * rmax * 4u);
                if (!g_tmp_T) { v41_exit_clear(); return -1; }
            }
        }
    }
    fprintf(stderr, "ds4: [pt124] loaded dir=%s head=%d alpha=%.4f aperp=%.4f ctrl=%d rank=%u sub=%d rank2=%u\n",
            dir, has_head ? 1 : 0, (double)alpha, (double)(g_ctrl.has_sub ? g_ctrl.alpha_perp : alpha),
            g_ctrl.has_ctrl, g_ctrl.rank, g_ctrl.has_sub, g_ctrl.rank2);
    return n_loaded;
}

static bool copy_or_link_file(const char *src, const char *dst) {
    (void)unlink(dst);
    if (link(src, dst) == 0) return true;
    FILE *fi = fopen(src, "rb");
    if (!fi) return false;
    FILE *fo = fopen(dst, "wb");
    if (!fo) { fclose(fi); return false; }
    uint8_t chunk[65536];
    size_t nr;
    bool ok = true;
    while ((nr = fread(chunk, 1, sizeof chunk, fi)) > 0) {
        if (fwrite(chunk, 1, nr, fo) != nr) { ok = false; break; }
    }
    if (ferror(fi)) ok = false;
    fclose(fi);
    if (fclose(fo) != 0) ok = false;
    return ok;
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
    if (g_ctrl.has_ctrl || g_ctrl.has_sub || (g_head.loaded && g_ctrl.alpha_head != g_head.file_alpha)) {
        char cp[4200]; snprintf(cp, sizeof cp, "%s/exit_ctrl.bin", dir);
        if (g_ctrl.src_path[0] && (g_ctrl.flags & (4u | 16u | 32u | 64u))) {
            if (!copy_or_link_file(g_ctrl.src_path, cp)) return -1;
            return n + 1;
        }
        const uint32_t E = DS4_N_EMBD, R = g_ctrl.rank, has_dg = g_ctrl.dg ? 1u : 0u;
        pt124_exit_hdr C; memset(&C, 0, sizeof C);
        C.magic = PT124_EXIT_MAGIC; C.version = PT124_VER; C.rank = R; C.flags = has_dg;
        C.alpha_head = g_ctrl.alpha_head;
        C.total_bytes = 64ull + ((uint64_t)(has_dg ? E : 0u) + 2ull * R * E) * 4ull;
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
    const float a_main = g_ctrl.has_sub ? g_ctrl.alpha_perp : g_ctrl.alpha_head;
    for (uint32_t r0 = 0; r0 < n; r0 += CH) {
        const uint32_t nr = (n - r0 < CH) ? (n - r0) : CH;
        ds4_gpu_tensor *vx = ds4_gpu_tensor_view(xn, (uint64_t)r0 * E * 4u, (uint64_t)nr * E * 4u);
        ds4_gpu_tensor *vz = ds4_gpu_tensor_view(logits, (uint64_t)r0 * V * 4u, (uint64_t)nr * V * 4u);
        bool ok = vx && vz;
        if (ok && g_head.loaded && a_main != 0.f) {
            ok = ds4_gpu_v41_matmul_fp4x32_tensor(g_tmp_log, g_head.buf, g_head.bytes, 64ull, E, V, vx, nr, 0) &&
                 ds4_gpu_bwd_axpy_tensor(vz, g_tmp_log, a_main, (uint64_t)nr * V);
        }
        if (ok && g_head.loaded && g_ctrl.sub_nz) {
            ds4_gpu_tensor *vdh2 = ds4_gpu_tensor_view(g_tmp_dh, 0, (uint64_t)nr * E * 4u);
            ok = vdh2 &&
                 ds4_gpu_v41_exit_delta_h_tensor(vdh2, g_tmp_T, vx, g_ctrl.zero_dg, g_ctrl.A2, g_ctrl.B2, nr, E, g_ctrl.rank2) &&
                 ds4_gpu_v41_matmul_fp4x32_tensor(g_tmp_log, g_head.buf, g_head.bytes, 64ull, E, V, vdh2, nr, 0) &&
                 ds4_gpu_bwd_axpy_tensor(vz, g_tmp_log, 1.0f, (uint64_t)nr * V);
            if (vdh2) ds4_gpu_tensor_free(vdh2);
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
    const uint32_t rmax = g_ctrl.rank > g_ctrl.rank2 ? g_ctrl.rank : g_ctrl.rank2;
    const uint64_t need = (uint64_t)m * (2u * E + rmax) * 4u;
    if (need > g_cap_bwd) {
        if (g_tmp_bwd) ds4_gpu_tensor_free(g_tmp_bwd);
        g_tmp_bwd = ds4_gpu_tensor_alloc(need);
        g_cap_bwd = g_tmp_bwd ? need : 0;
        if (!g_tmp_bwd) return false;
    }
    ds4_gpu_tensor *gd = ds4_gpu_tensor_view(g_tmp_bwd, 0, (uint64_t)m * E * 4u);
    ds4_gpu_tensor *gtmp = ds4_gpu_tensor_view(g_tmp_bwd, (uint64_t)m * E * 4u, (uint64_t)m * E * 4u);
    ds4_gpu_tensor *vt = rmax > 0u ? ds4_gpu_tensor_view(g_tmp_bwd, (uint64_t)m * 2u * E * 4u, (uint64_t)m * rmax * 4u) : NULL;
    bool ok = gd && gtmp && (!rmax || vt != NULL);
    const bool need_gd = g_head.loaded && (g_ctrl.alpha_head != 0.f || (g_ctrl.has_sub && g_ctrl.alpha_perp != 0.f) || g_ctrl.sub_nz);
    if (ok && need_gd) {
        ok = ds4_gpu_bwd_matmul_t_tensor(gd, g_head.buf, g_head.bytes, DS4_GGT_FP4X32, 64ull, E, V, glog, m, 0);
        if (ok && g_ctrl.alpha_head != 0.f)
            ok = ds4_gpu_bwd_axpy_tensor(g_xn, gd, g_ctrl.alpha_head, (uint64_t)m * E);
    }
    if (ok && g_ctrl.has_ctrl) {
        ok = ds4_gpu_v41_exit_delta_h_tensor(gtmp, vt, g_xn, g_ctrl.dg, g_ctrl.B, g_ctrl.A, m, E, g_ctrl.rank) &&
             ds4_gpu_bwd_axpy_tensor(g_xn, gtmp, 1.0f, (uint64_t)m * E);
    }
    if (ok && g_head.loaded && g_ctrl.has_sub) {
        const float da = g_ctrl.alpha_perp - g_ctrl.alpha_head;
        if (da != 0.f) ok = ds4_gpu_bwd_axpy_tensor(g_xn, gd, da, (uint64_t)m * E);
        if (ok && g_ctrl.sub_nz) {
            ok = ds4_gpu_v41_exit_delta_h_tensor(gtmp, vt, gd, g_ctrl.zero_dg, g_ctrl.B2, g_ctrl.A2, m, E, g_ctrl.rank2) &&
                 ds4_gpu_bwd_axpy_tensor(g_xn, gtmp, 1.0f, (uint64_t)m * E);
        }
    }
    if (vt) ds4_gpu_tensor_free(vt);
    if (gtmp) ds4_gpu_tensor_free(gtmp);
    if (gd) ds4_gpu_tensor_free(gd);
    return ok;
}
#endif
typedef int ds4_core_v41_exit_nonempty_tu;
