/* core_v41_amp.c — DeepSeek V4.1 反修放大器(zchain 的 V4.1 形态)在引擎里的加载与应用(2026-09-12)。
 *
 * 产物: <dir>/amp_Lnn.bin, 头 <i32 D><i32 K><i32 存储类型>, 接 A[K][D], 再 B[K][D](B 内存 [K][D] 行主序 = 数学上的
 * D×K 列主序)。语义与 Python 判决态(v41_amp_hooks.install_apply → libv41amp v41_amp_apply_gpu)逐式同:
 * 每层 MoE 输出 y(bf16 格点) 做 y += x·(B·A) (f32), 再舍回 bf16; x = 该层 MoE 输入(ffn_norm 出口)。
 * 引擎侧算法: T = x·B (n×K), y += T·A —— 两发 cuBLAS Sgemm, 与 C 库先乘 B·A 再乘 x 只差累加序。
 * 没有的层就不挂(逐层 K 可不同)。
 *
 * 【存储类型(2026-09-13)】头第三字段: 43 = fp4x32(4.25 bpw, 骨架/出口头/HF 出厂专家同一种块格式, spark 原生),
 * 1 = f32(09-12 的第一版产物, 8 倍体积)。★必须按这个字段分派★ —— 两种格式的字节数差 7.5 倍, 拿 f32 的读法去读
 * fp4 文件不会报错(长度校验之外), 只会把码字当浮点位型解出一堆垃圾, 挂上去照跑, 出的是"一眼合理"的假账。
 * 解出来一律是 f32 上设备: 省的是磁盘与解码带宽, GEMM 仍走 f32(K≤512 的两发小 GEMM 不是瓶颈)。 */
#include "core_internal.h"
#include "../common/ds4_quantfmt.h"
#include "../common/ds4_float.h"
#ifndef DS4_NO_GPU

/* 读一块 [K][D] 权重到 buf(f32)。ty 43 = fp4x32 按块解码, 1 = 直接 f32。返回 false = 截断/类型不认。 */
static bool amp_read_mat(FILE *f, int32_t ty, size_t nel, float *buf, uint8_t *pk) {
    if (ty == (int32_t)DS4_GGT_FP4X32) {
        const size_t nb = nel / 32, nby = nb * 17;
        if (fread(pk, 1, nby, f) != nby) return false;
        ds4_deq_fp4x32(pk, nb, buf);
        return true;
    }
    return fread(buf, 4, nel, f) == nel;
}

/* 权重侧反修(gr_Lnn.bin): 逐专家逐输出通道的增益【缩放因子】, 头 <i32 n_expert><i32 D><i32 1(f32)>。
 * 装进 GPU 后 VQ 解码时 g_eff = 载荷 g_r × s —— 盘上权重不动。返回挂上的层数, <0 = 有文件但读坏了。 */
static int v41_gr_load(const char *dir) {
    int n = 0; double mb = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        char p[4200]; snprintf(p, sizeof p, "%s/gr_L%02u.bin", dir, il);
        FILE *f = fopen(p, "rb");
        if (!f) continue;
        int32_t hd[3];
        if (fread(hd, 4, 3, f) != 3 || hd[0] != (int32_t)DS4_N_EXPERT || hd[1] != (int32_t)DS4_N_EMBD ||
            (hd[2] != 1 && hd[2] != 2)) {
            fprintf(stderr, "ds4: 增益覆盖 %s 头不对(专家 %d 通道 %d 类型 %d; 要 %u/%u/1或2)\n",
                    p, hd[0], hd[1], hd[2], (unsigned)DS4_N_EXPERT, (unsigned)DS4_N_EMBD);
            fclose(f); return -1;
        }
        const size_t nel = (size_t)hd[0] * hd[1];
        const int f16 = hd[2] == 2;
        float *buf = xmalloc(nel * 4);
        bool ok;
        if (f16) {   /* 盘上 f16(缩放因子恒在 1 附近, f16 相对精度 0.1%), 解成 f32 上设备 */
            uint16_t *raw = xmalloc(nel * 2);
            ok = fread(raw, 2, nel, f) == nel;
            if (ok) for (size_t t = 0; t < nel; t++) buf[t] = ds4_f16_to_f32(raw[t]);
            free(raw);
        } else ok = fread(buf, 4, nel, f) == nel;
        ok = ok && ds4_gpu_v41_set_gr_override(il, buf, (uint32_t)hd[0], (uint32_t)hd[1]);
        free(buf); fclose(f);
        if (!ok) { fprintf(stderr, "ds4: 增益覆盖 %s 读/上传失败\n", p); return -1; }
        n++; mb += (double)nel * (f16 ? 2 : 4) / 1e6;
    }
    if (n) fprintf(stderr, "ds4: [反修·权重侧] %d 层挂上逐专家增益覆盖 (盘上 %.1f MB)\n", n, mb);
    return n;
}

bool v41_amp_load(ds4_v41_state *st, const char *dir) {
    uint32_t n_arm = 0, kmin = 0, kmax = 0; double mb = 0.0;
    const char *tyname = "";
    const int n_gr = v41_gr_load(dir);
    if (n_gr < 0) return false;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        char p[4200]; snprintf(p, sizeof p, "%s/amp_L%02u.bin", dir, il);
        FILE *f = fopen(p, "rb");
        if (!f) continue;
        int32_t hd[3];
        if (fread(hd, 4, 3, f) != 3 || hd[0] != (int32_t)DS4_N_EMBD || hd[1] <= 0 || hd[1] > 8192) {
            fprintf(stderr, "ds4: 放大器 %s 头不对(D %d K %d)\n", p, hd[0], hd[1]); fclose(f); return false;
        }
        const uint32_t D = (uint32_t)hd[0], K = (uint32_t)hd[1];
        const int32_t ty = hd[2];
        if (ty != 1 && ty != (int32_t)DS4_GGT_FP4X32) {
            fprintf(stderr, "ds4: 放大器 %s 存储类型 %d 不认(1=f32 / 43=fp4x32)\n", p, ty); fclose(f); return false;
        }
        const size_t nel = (size_t)K * D;
        if (ty == (int32_t)DS4_GGT_FP4X32 && nel % 32u) {
            fprintf(stderr, "ds4: 放大器 %s K×D=%zu 不是 32 的整数倍, fp4x32 装不下\n", p, nel); fclose(f); return false;
        }
        float *buf = xmalloc(nel * 4);
        uint8_t *pk = ty == (int32_t)DS4_GGT_FP4X32 ? xmalloc(nel / 32u * 17u) : NULL;
        bool ok = true;
        st->ampA[il] = ds4_gpu_tensor_alloc(nel * 4); st->ampB[il] = ds4_gpu_tensor_alloc(nel * 4);
        if (!st->ampA[il] || !st->ampB[il]) ok = false;
        /* A 乘 β(--zchain-scale): y += x·(B·(βA)) = β·(x·B·A) ⇒ 整层修正缩到 β 倍。乘在 A 这一边而不是 B,
         * 是因为两发 GEMM 里 A 是第二发的权重, 缩它不改第一发 T = x·B 的数值范围。 */
        if (ok && (!amp_read_mat(f, ty, nel, buf, pk))) ok = false;
        if (ok && g_ds4_v41_amp_scale != 1.0f) for (size_t t = 0; t < nel; t++) buf[t] *= g_ds4_v41_amp_scale;
        if (ok && !ds4_gpu_tensor_write(st->ampA[il], 0, buf, nel * 4)) ok = false;
        if (ok && (!amp_read_mat(f, ty, nel, buf, pk) || !ds4_gpu_tensor_write(st->ampB[il], 0, buf, nel * 4))) ok = false;
        free(buf); free(pk); fclose(f);
        if (!ok) { fprintf(stderr, "ds4: 放大器 %s 读/上传失败\n", p); return false; }
        st->ampK[il] = K;
        if (!n_arm || K < kmin) kmin = K;
        if (K > kmax) kmax = K;
        n_arm++;
        /* 体积按【盘上真实字节】算, 不按"部署时打算存成什么"估 —— 旧版这里固定按 f16 估, 而盘上落的是 f32, 报了一半的假账 */
        mb += ty == (int32_t)DS4_GGT_FP4X32 ? 2.0 * nel / 32.0 * 17.0 / 1e6 : 2.0 * nel * 4.0 / 1e6;
        tyname = ty == (int32_t)DS4_GGT_FP4X32 ? "fp4x32" : "f32";
    }
    if (!n_arm) {
        /* 只有增益覆盖、没有低秩放大器也是合法的一种插件(权重侧反修的产物就长这样) */
        if (n_gr > 0) return true;
        fprintf(stderr, "ds4: 反修目录 %s 里既没有 amp_Lnn.bin 也没有 gr_Lnn.bin\n", dir); return false;
    }
    st->ampT = ds4_gpu_tensor_alloc((uint64_t)st->cap_tok * kmax * 4);
    if (!st->ampT) return false;
    fprintf(stderr, "ds4: [反修] 挂上 %u 层放大器 (K %u~%u, %s 盘上合计 %.1f MB, 步长 β=%.4g) ← %s\n",
            n_arm, kmin, kmax, tyname, mb, (double)g_ds4_v41_amp_scale, dir);
    return true;
}

/* 反修取料钩子: 放大器应用前把这一层的 MoE 入/出与路由交给解算方(主机内存)。先 flush+synchronize —— y 刚由 GPU 写完,
 * 不同步就 D2H 读到的是半成品。回调说 1 = 取完了 ⇒ 置 stop_early, 前向驱动跳过余下层与出口。 */
int v41_amp_hook(ds4_v41_state *st, uint32_t il) {
    if (!g_ds4_v41_hook) return 0;
    const uint32_t n = st->n, D = DS4_N_EMBD, NU = DS4_N_EXPERT_USED;
    if (!ds4_gpu_flush_commands() || !ds4_gpu_synchronize()) return -1;
    const uint32_t HC = DS4_N_HC;
    float *x = xmalloc((size_t)n * D * 4), *y = xmalloc((size_t)n * D * 4), *rw = xmalloc((size_t)n * NU * 4);
    int32_t *sel = xmalloc((size_t)n * NU * 4);
    /* α[i] = Σ_k pre[i][k]·post[i][k]: y 经 hc_post 进各路、再由 hc_pre 用本层 ffn_pre 合成回来的系数。
     * 此刻 st->pre 正是本层 ffn_pre(层末才与 pre_mix 交换), st->post 是 ffn 的 post —— 末层出口用的就是这两件。 */
    float *pre = xmalloc((size_t)n * HC * 4), *post = xmalloc((size_t)n * HC * 4), *alpha = xmalloc((size_t)n * 4);
    /* 逐专家 down 输出(未乘路由权重): 只有 prefill GEMM 路物化, 取不到就递 NULL —— 不猜、不另算一遍。
     * 63 MB(n=512) 一层一块, 解码路(n≤8)与取不到时都不分配。 */
    float *ye = NULL, *ysh = NULL;
    if (n > 8u) {
        ye = xmalloc((size_t)n * NU * D * 4);
        ysh = xmalloc((size_t)n * D * 4);
        if (!ds4_gpu_v41_vq_capture_expert_out(ye, n, NU, D) ||
            !ds4_gpu_tensor_read(st->so, 0, ysh, (uint64_t)n * D * 4)) { free(ye); free(ysh); ye = ysh = NULL; }
    }
    int rc = -1;
    if (ds4_gpu_tensor_read(st->xn, 0, x, (uint64_t)n * D * 4) && ds4_gpu_tensor_read(st->y, 0, y, (uint64_t)n * D * 4) &&
        ds4_gpu_tensor_read(st->sel, 0, sel, (uint64_t)n * NU * 4) && ds4_gpu_tensor_read(st->rw, 0, rw, (uint64_t)n * NU * 4) &&
        ds4_gpu_tensor_read(st->pre, 0, pre, (uint64_t)n * HC * 4) && ds4_gpu_tensor_read(st->post, 0, post, (uint64_t)n * HC * 4)) {
        for (uint32_t i = 0; i < n; i++) {
            float a = 0.f;
            for (uint32_t k = 0; k < HC; k++) a += pre[i * HC + k] * post[i * HC + k];
            alpha[i] = a;
        }
        rc = g_ds4_v41_hook(g_ds4_v41_hook_ud, (int)il, (int)st->pos0, (int)n, (int)D, (int)NU, DS4_SWIGLU_CLAMP_EXP, x, y, sel, rw, alpha, ye, ysh);
        if (rc < 0) fprintf(stderr, "ds4: [反修钩子] L%02u 回调要求停车(rc %d)\n", il, rc);
        else if (rc > 0) { st->stop_early = 1; rc = 1; }
    } else fprintf(stderr, "ds4: [反修钩子] L%02u 取 x/y/路由/hc 系数 下主机失败\n", il);
    free(x); free(y); free(sel); free(rw); free(pre); free(post); free(alpha); free(ye); free(ysh);
    return rc;
}

void v41_amp_free(ds4_v41_state *st) {
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        if (st->ampA[il]) ds4_gpu_tensor_free(st->ampA[il]);
        if (st->ampB[il]) ds4_gpu_tensor_free(st->ampB[il]);
        st->ampA[il] = st->ampB[il] = NULL; st->ampK[il] = 0;
    }
    if (st->ampT) { ds4_gpu_tensor_free(st->ampT); st->ampT = NULL; }
}

/* y[n][D](bf16 格点) += x[n][D]·(B·A) → 舍 bf16 */
bool v41_amp_apply(ds4_v41_state *st, uint32_t il) {
    if (!st->ampA[il]) return true;
    if (!ds4_gpu_v41_amp_apply_tensor(st->y, st->xn, st->ampA[il], st->ampB[il], st->ampT, st->n, DS4_N_EMBD, st->ampK[il])) return false;
    return ds4_gpu_v41_round_bf16_tensor(st->y, (uint64_t)st->n * DS4_N_EMBD) != 0;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_v41_amp_nonempty_tu;
