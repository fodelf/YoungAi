/* ds4_zchain.c -- DQZ2 sidecar loader + λ fold. See ds4_zchain.h for the math
 * contract; the byte format is defined by zchain_write() in
 * gguf-tools/go-onebit/quant/ds4quant_run.c:
 *   "DQZ2" u32 | n_layer u32 | per layer { L u32, n_ops u32,
 *       per op { type u32, paysz u32, payload } }
 *   payload: 1=GL f32 g | 2=GLdyn2 f32 w2p[4] | 3=GLdyn8 f32 w8[9] (+ fp16
 *   V8[8][d_model] when carried) | 4=TREF f32 t | 5=GE fp16[n_expert]. */
#include "ds4_zchain.h"
#include "ds4_loss.h"   /* posttrain 钩子: 四损失(dither/权)同一份实现 */

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static float zc_fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t man  = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) { bits = sign; }
        else {                       /* subnormal: renormalize */
            uint32_t e = 127 - 15 + 1, m = man;
            while (!(m & 0x400u)) { m <<= 1; e--; }
            bits = sign | (e << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

ds4_zchain *ds4_zchain_load(const char *path, uint32_t n_layer, uint32_t n_expert, uint32_t d_model) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "ds4: zchain %s: cannot open\n", path); return NULL; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 8) { close(fd); fprintf(stderr, "ds4: zchain %s: bad size\n", path); return NULL; }
    size_t sz = (size_t)st.st_size;
    uint8_t *map = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { fprintf(stderr, "ds4: zchain %s: mmap failed\n", path); return NULL; }

    const uint8_t *p = map, *end = map + sz;
    uint32_t magic, nlay;
    memcpy(&magic, p, 4); memcpy(&nlay, p + 4, 4); p += 8;
    if (magic != 0x325A5144u) {         /* "DQZ2" */
        fprintf(stderr, "ds4: zchain %s: bad magic 0x%08x\n", path, magic);
        munmap(map, sz); return NULL;
    }
    if (nlay != n_layer) {
        fprintf(stderr, "ds4: zchain %s: %u layers, model has %u -- refusing\n", path, nlay, n_layer);
        munmap(map, sz); return NULL;
    }

    ds4_zchain *z = calloc(1, sizeof(*z));
    z->n_layer = n_layer; z->n_expert = n_expert; z->d_model = d_model;
    z->layer = calloc(n_layer, sizeof(*z->layer));
    z->map = map; z->map_size = sz;

    for (uint32_t li = 0; li < nlay && p + 8 <= end; li++) {
        uint32_t L, n_ops;
        memcpy(&L, p, 4); memcpy(&n_ops, p + 4, 4); p += 8;
        ds4_zchain_layer *zl = (L < n_layer) ? &z->layer[L] : NULL;
        if (zl && n_ops) zl->ops = calloc(n_ops, sizeof(*zl->ops));
        for (uint32_t oi = 0; oi < n_ops && p + 8 <= end; oi++) {
            uint32_t ty, psz;
            memcpy(&ty, p, 4); memcpy(&psz, p + 4, 4); p += 8;
            const uint8_t *pay = p;
            if (pay + psz > end) { p = end; break; }
            p += psz;
            if (!zl) continue;
            if (ty == 5u && psz >= (uint64_t)n_expert * 2u) {
                /* effective GE = LAST type-5 record (quantizer bytes_moe parity) */
                if (!zl->ge) zl->ge = malloc((size_t)n_expert * sizeof(float));
                const uint16_t *h = (const uint16_t *)pay;
                for (uint32_t e = 0; e < n_expert; e++) zl->ge[e] = zc_fp16_to_fp32(h[e]);
                continue;
            }
            if (ty == 8u && psz >= 16) {
                /* type8 路由闭式侧车(zl.RTE, 2026-08-19): δlogits=U·tanh(Vᵀx/s),
                 * 载荷 u32 k | f32 scale | u32 din=d_model | u32 dout=n_expert
                 * | fp16 z[k],U[n_expert*k],V[din*k] */
                uint32_t zk, din, dout; float tr;
                memcpy(&zk, pay, 4); memcpy(&tr, pay + 4, 4);
                memcpy(&din, pay + 8, 4); memcpy(&dout, pay + 12, 4);
                size_t nh = (size_t)zk + (size_t)zk * din + (size_t)zk * dout;
                if (zk > 0 && zk <= 1024 && din == d_model && dout == n_expert &&
                    psz >= 16 + nh * 2) {
                    zl->rte.zlk = zk; zl->rte.zltr = tr; zl->rte.zdin = din;
                    zl->rte.zmul = 2u;                       /* 标记: 路由偏置形态 */
                    zl->rte.zlm = (const uint16_t *)(pay + 16);
                }
                continue;
            }
            if (ty == 10u && psz >= 32) {
                /* type10 zl.4L 四损失参数(2026-08-26 用户架构): f32 w[4] | u64 seed |
                 * f32 dscale | u32 d | fp16 wcls[d]。wnorm=均值归一 f32(前向加权范数用)。 */
                ds4_zchain_4l *q = &zl->l4;
                memcpy(q->w, pay, 16);
                memcpy(&q->seed, pay + 16, 8);
                memcpy(&q->dscale, pay + 24, 4);
                memcpy(&q->d, pay + 28, 4);
                if (q->d == d_model && psz >= 32u + (uint64_t)q->d * 2u) {
                    q->wcls = (const uint16_t *)(pay + 32);
                    float *wn = malloc((size_t)q->d * sizeof(float));
                    if (wn) {
                        double m = 0.0;
                        for (uint32_t j2 = 0; j2 < q->d; j2++) {
                            wn[j2] = zc_fp16_to_fp32(q->wcls[j2]);
                            m += wn[j2];
                        }
                        m = m / q->d + 1e-30;
                        for (uint32_t j2 = 0; j2 < q->d; j2++) wn[j2] = (float)(wn[j2] / m);
                        q->wnorm = wn;
                    }
                } else { memset(q, 0, sizeof(*q)); }
                continue;
            }
            /* type7(乘性AMP)/type9(动态z AMPD) 已删(2026-08-26 清仓): 生产链非现役、
             * 现役产物零消费者。载荷遇到即拒(响亮跳过), 不再静默进 apply。 */
            if ((ty == 7u || ty == 9u) && psz >= 16) {
                fprintf(stderr, "ds4: zchain L%u type%u 已废弃(2026-08-26), 记录跳过\n", L, ty);
                continue;
            }
            if (ty == 6u && psz >= 16) {
                /* type6 frozen z^L: 载荷 u32 k | f32 tr | u32 din | u32 dout | fp16 z,U,V */
                uint32_t zk, din, dout; float tr;
                memcpy(&zk, pay, 4); memcpy(&tr, pay + 4, 4);
                memcpy(&din, pay + 8, 4); memcpy(&dout, pay + 12, 4);
                size_t nh = (size_t)zk + (size_t)zk * din + (size_t)zk * dout;
                if (zk > 0 && zk <= 1024 && (din == d_model || din == 3u * d_model)
                    && dout == d_model && psz >= 16 + nh * 2) {
                    zl->zl.zlk = zk; zl->zl.zltr = tr; zl->zl.zdin = din;
                    zl->zl.zmul = 0u;
                    zl->zl.zlm = (const uint16_t *)(pay + 16);   /* aliases the mmap */
                    if (din == d_model) {
                        /* 线性 z: 载入时转 f32, apply 走 ds4_z 模块(反修解算同一份实现)。
                         * ftA(din=3d)是 φ 特征提升形态, 模块无 φ, 留 fp16 旧路。 */
                        ds4_z *zm = calloc(1, sizeof(*zm));
                        if (zm) {
                            zm->d_in = din; zm->d_out = dout; zm->rank = zk; zm->k = zk;
                            zm->z = malloc((size_t)zk * sizeof(float));
                            zm->U = malloc((size_t)dout * zk * sizeof(float));
                            zm->V = malloc((size_t)din * zk * sizeof(float));
                            if (zm->z && zm->U && zm->V) {
                                const uint16_t *hz = zl->zl.zlm, *hU = hz + zk,
                                               *hV = hU + (size_t)dout * zk;
                                for (uint32_t c = 0; c < zk; c++) zm->z[c] = zc_fp16_to_fp32(hz[c]);
                                for (size_t i = 0; i < (size_t)dout * zk; i++) zm->U[i] = zc_fp16_to_fp32(hU[i]);
                                for (size_t i = 0; i < (size_t)din * zk; i++) zm->V[i] = zc_fp16_to_fp32(hV[i]);
                                zl->zl.zmod = zm;
                            } else { free(zm->z); free(zm->U); free(zm->V); free(zm); }
                        }
                    }
                }
                continue;
            }
            ds4_zchain_op *o = &zl->ops[zl->n_ops];
            memset(o, 0, sizeof(*o));
            if (ty == 1u && psz >= 4) { o->type = 1; memcpy(&o->g, pay, 4); }
            else if (ty == 2u && psz >= 16) { o->type = 2; memcpy(o->w2p, pay, 16); }
            else if (ty == 3u && psz >= 36) {
                o->type = 3; memcpy(o->w8, pay, 36);
                if (psz >= 36u + 8u * d_model * 2u) o->v8 = (const uint16_t *)(pay + 36);
                else continue;          /* V8-less dyn8 is a no-op in the quantizer replay: drop it */
            }
            else if (ty == 4u && psz >= 4) { o->type = 4; memcpy(&o->g, pay, 4); }
            else continue;              /* unknown / short op: skip */
            zl->n_ops++;
            z->n_ops_total++;
        }
    }
    uint32_t n_zl = 0, n_rte = 0;
    for (uint32_t il = 0; il < n_layer; il++) {
        z->layer[il].zl.w4norm = z->layer[il].l4.wnorm;   /* 4L→z 信任域加权链接 */
        if (z->layer[il].ge) z->n_ge_layers++;
        if (z->layer[il].zl.zlk) n_zl++;
        if (z->layer[il].rte.zlk) n_rte++;
    }
    fprintf(stderr, "ds4: zchain loaded %s: %u chain ops + %u GE layers + %u z^L layers + %u route layers over %u layers\n",
            path, z->n_ops_total, z->n_ge_layers, n_zl, n_rte, n_layer);
    if (z->n_ops_total == 0 && z->n_ge_layers == 0 && n_zl == 0 && n_rte == 0) {
        fprintf(stderr, "ds4: zchain %s carries no ops; ignoring\n", path);
        ds4_zchain_free(z);
        return NULL;
    }
    return z;
}

void ds4_zchain_free(ds4_zchain *z) {
    if (!z) return;
    if (z->layer) {
        for (uint32_t il = 0; il < z->n_layer; il++) {
            free(z->layer[il].ops);
            free(z->layer[il].ge);
            free(z->layer[il].l4.wnorm);
        }
        free(z->layer);
    }
    if (z->map) munmap(z->map, z->map_size);
    free(z);
}

float ds4_zchain_lambda(const ds4_zchain *z, uint32_t il, const float *x) {
    if (!z || il >= z->n_layer || z->layer[il].n_ops == 0) return 1.0f;
    const ds4_zchain_layer *zl = &z->layer[il];
    const uint32_t d = z->d_model;
    float xnorm = -1.0f;                /* lazy: only when a dyn2 op needs it */
    float lam = 1.0f;
    for (uint32_t i = 0; i < zl->n_ops; i++) {
        const ds4_zchain_op *o = &zl->ops[i];
        if (o->type == 1u) lam = o->g * lam;
        else if (o->type == 2u) {
            if (xnorm < 0.0f) {
                double v = 0.0;
                for (uint32_t j = 0; j < d; j++) v += (double)x[j] * (double)x[j];
                xnorm = (float)sqrt(v);
            }
            double c = (double)o->w2p[0] + (double)o->w2p[1] * (((double)xnorm - o->w2p[2]) / o->w2p[3]);
            if (c < 0.25) c = 0.25;
            if (c > 4.0)  c = 4.0;
            lam = (float)c * lam;
        } else if (o->type == 3u && o->v8) {
            double c = o->w8[0];
            for (uint32_t k = 0; k < 8; k++) {
                const uint16_t *vr = o->v8 + (size_t)k * d;
                double a = 0.0;
                for (uint32_t j = 0; j < d; j++) a += (double)x[j] * (double)zc_fp16_to_fp32(vr[j]);
                c += (double)o->w8[1 + k] * a;
            }
            if (c < 0.25) c = 0.25;
            if (c > 4.0)  c = 4.0;
            lam = (float)c * lam;
        } else if (o->type == 4u) {
            lam = 1.0f + o->g * (lam - 1.0f);
        }
    }
    return lam;
}

void ds4_zchain_zl_apply(const ds4_zchain_zl *zl, uint32_t d_model, const float *x, float *routed) {
    if (!zl || !zl->zlk || !zl->zlm) return;
    const uint32_t k = zl->zlk, d = d_model;
    const uint32_t din = zl->zdin ? zl->zdin : d;   /* md86: 3d = ftA feature lift */

    /* 线性 z(din==d): 修正走 ds4_z 模块 —— 与反修解算器同一份实现(2026-08-26 复用定案)。
     * 信任域夹持(‖Δ‖ ≤ tr·‖routed‖)是运行时语义, 留在模块外面(判决尺 tr=0.5 同契约)。
     * 精度注: 模块 f32 累加 vs 旧手抄 f64, 尾位差(单测 ut_zmod_parity 盯 1e-4 相对)。 */
    if (zl->zmod) {
        float *delta = calloc(d, sizeof(float));
        if (!delta) return;
        ds4_z_apply(zl->zmod, x, delta);
        double nd = 0.0, nr = 0.0;
        const float *w4 = zl->w4norm;    /* zl.4L classify 权: 重要维度说了算的夹持 */
        for (uint32_t j = 0; j < d; j++) {
            const double wj = w4 ? (double)w4[j] : 1.0;
            nd += wj * (double)delta[j] * delta[j];
            nr += wj * (double)routed[j] * routed[j];
        }
        nd = sqrt(nd); nr = sqrt(nr);
        const double cap = (double)zl->zltr * nr;
        const float s = (nd > cap && nd > 0.0) ? (float)(cap / nd) : 1.0f;
        for (uint32_t j = 0; j < d; j++) routed[j] += s * delta[j];
        free(delta);
        return;
    }

    /* ftA(din=3d) 旧路: φ 特征提升形态, 模块无 φ —— fp16 直读(type7/9 分支已删)。 */
    const uint16_t *hz = zl->zlm;
    const uint16_t *hU = hz + k;
    const uint16_t *hV = hU + (size_t)d * k;
    double pvs[16];   /* k<=16 走栈(旧路径零开销); 大 k 堆分配 */
    double *pv = k <= 16 ? pvs : malloc((size_t)k * sizeof(double));
    if (!pv) return;
    float *phi = NULL;
    const float *xin = x;
    if (din == 3u * d) {   /* φ=[x, x⊙x/rms, relu(x)], rms=sqrt(mean(x²))+1e-6 — zlayer zl_phi 逐式一致 */
        phi = malloc((size_t)din * sizeof(float));
        if (!phi) { if (pv != pvs) free(pv); return; }
        double ss = 0.0;
        for (uint32_t j = 0; j < d; j++) ss += (double)x[j] * (double)x[j];
        const float nrm = (float)sqrt(ss / d) + 1e-6f;
        for (uint32_t j = 0; j < d; j++) {
            phi[j] = x[j]; phi[d + j] = x[j] * x[j] / nrm; phi[2 * d + j] = x[j] > 0.0f ? x[j] : 0.0f;
        }
        xin = phi;
    }
    for (uint32_t c = 0; c < k; c++) {
        double a = 0.0;
        for (uint32_t j = 0; j < din; j++)
            a += (double)xin[j] * (double)zc_fp16_to_fp32(hV[(size_t)j * k + c]);
        pv[c] = a * (double)zc_fp16_to_fp32(hz[c]);
    }
    double nd = 0.0, nr = 0.0;
    for (uint32_t j = 0; j < d; j++) {
        double a = 0.0;
        const uint16_t *ur = hU + (size_t)j * k;
        for (uint32_t c = 0; c < k; c++) a += pv[c] * (double)zc_fp16_to_fp32(ur[c]);
        const double wj = zl->w4norm ? (double)zl->w4norm[j] : 1.0;
        nd += wj * a * a;
        nr += wj * (double)routed[j] * (double)routed[j];
    }
    nd = sqrt(nd); nr = sqrt(nr);
    double cap = (double)zl->zltr * nr;
    const float s = (nd > cap && nd > 0.0) ? (float)(cap / nd) : 1.0f;
    for (uint32_t j = 0; j < d; j++) {
        double a = 0.0;
        const uint16_t *ur = hU + (size_t)j * k;
        for (uint32_t c = 0; c < k; c++) a += pv[c] * (double)zc_fp16_to_fp32(ur[c]);
        routed[j] += s * (float)a;
    }
    if (pv != pvs) free(pv);
    if (phi) free(phi);
}


/* ★引擎内调 z(用户架构②)★: U/V 冻结, 按 zl.4L 四权闭式重解 z 对角。
 * 目标 min_z Σ_t a_t ‖diag(√w)(R_t − U diag(z) Vᵀ x_t)‖² + λ‖z‖², 其中
 * a_t=1/‖Yt_t‖²(align 行权=方向平权), w=classify 权(缺则平权), smooth=固定种子
 * dither 增广行, λ=w_fixed·Gram 迹(fixed)。k×k 正规方程, 高斯消元(k≤1024)。 */
int ds4_zchain_posttrain_z(struct ds4_zchain *z, uint32_t layer,
                           const float *X, const float *R, const float *Yt, uint32_t n) {
    if (!z || layer >= z->n_layer || !X || !R || !n) return -1;
    ds4_zchain_layer *zl = &z->layer[layer];
    ds4_z *zm = zl->zl.zmod;
    if (!zm || !zl->l4.wnorm) return -1;              /* 只支持线性 z + 有 4L 参数的层 */
    const uint32_t k = zm->k, d = zm->d_out, din = zm->d_in;
    if (din != d) return -1;
    const float *w4 = zl->l4.wnorm;
    const float wa = zl->l4.w[0], wc = zl->l4.w[1], ws = zl->l4.w[2], wf = zl->l4.w[3];
    double *G = calloc((size_t)k * k, sizeof(double));
    double *b = calloc(k, sizeof(double));
    float *pv = malloc((size_t)k * sizeof(float));
    float *ub = malloc((size_t)k * sizeof(float));
    float *xp = malloc((size_t)d * sizeof(float));
    float *dl = malloc((size_t)d * sizeof(float));
    if (!G || !b || !pv || !ub || !xp || !dl) {
        free(G); free(b); free(pv); free(ub); free(xp); free(dl); return -1;
    }
    const int naug = (ws > 0.0f && zl->l4.dscale > 0.0f) ? 2 : 1;   /* smooth: 扰动增广遍 */
    for (uint32_t t = 0; t < n; t++) {
        const float *yt = Yt ? Yt + (size_t)t * d : NULL;
        double an = 0.0;
        if (yt) { for (uint32_t j = 0; j < d; j++) an += (double)yt[j] * yt[j]; }
        const double at = (wa > 0.0f && an > 0.0) ? 1.0 / an : 1.0;    /* align 行权 */
        for (int aug = 0; aug < naug; aug++) {
            const float *xr = X + (size_t)t * d;
            if (aug) {
                double ss = 0.0;
                for (uint32_t j = 0; j < d; j++) ss += (double)xr[j] * xr[j];
                ds4_loss_dither(zl->l4.seed, t, (float)(zl->l4.dscale * sqrt(ss / d)), dl, d);
                for (uint32_t j = 0; j < d; j++) xp[j] = xr[j] + dl[j];
                xr = xp;
            }
            const double roww = at * (aug ? (double)ws : 1.0);
            for (uint32_t c = 0; c < k; c++) {         /* p = Vᵀx */
                double a = 0.0;
                for (uint32_t j = 0; j < din; j++) a += (double)xr[j] * zm->V[(size_t)j * k + c];
                pv[c] = (float)a;
            }
            const float *rt = R + (size_t)t * d;
            for (uint32_t c = 0; c < k; c++) {         /* 列 c 的加权基向量 = U[:,c]·p_c */
                double bc = 0.0;
                for (uint32_t j = 0; j < d; j++) {
                    const double wj = 1.0 + (double)wc * ((double)w4[j] - 1.0);   /* classify 混权 */
                    bc += wj * (double)zm->U[(size_t)j * k + c] * pv[c] * (aug ? 0.0 : (double)rt[j]);
                }
                b[c] += roww * bc;
                for (uint32_t c2 = c; c2 < k; c2++) {
                    double g2 = 0.0;
                    for (uint32_t j = 0; j < d; j++) {
                        const double wj = 1.0 + (double)wc * ((double)w4[j] - 1.0);
                        g2 += wj * (double)zm->U[(size_t)j * k + c] * (double)zm->U[(size_t)j * k + c2];
                    }
                    g2 *= (double)pv[c] * pv[c2] * roww;
                    G[(size_t)c * k + c2] += g2;
                    if (c2 != c) G[(size_t)c2 * k + c] += g2;
                }
            }
        }
    }
    double tr = 0.0;
    for (uint32_t c = 0; c < k; c++) tr += G[(size_t)c * k + c];
    tr = tr / k + 1e-30;
    for (uint32_t c = 0; c < k; c++) G[(size_t)c * k + c] += (double)(wf > 0.0f ? wf : 1e-3f) * tr;
    for (uint32_t c = 0; c < k; c++) {                 /* 高斯消元(部分主元) */
        uint32_t piv = c;
        for (uint32_t r2 = c + 1; r2 < k; r2++)
            if (fabs(G[(size_t)r2 * k + c]) > fabs(G[(size_t)piv * k + c])) piv = r2;
        if (piv != c) {
            for (uint32_t j = 0; j < k; j++) {
                double tmp = G[(size_t)c * k + j];
                G[(size_t)c * k + j] = G[(size_t)piv * k + j];
                G[(size_t)piv * k + j] = tmp;
            }
            double tb = b[c]; b[c] = b[piv]; b[piv] = tb;
        }
        const double dg = G[(size_t)c * k + c];
        if (fabs(dg) < 1e-300) { free(G); free(b); free(pv); free(ub); free(xp); free(dl); return -1; }
        for (uint32_t r2 = c + 1; r2 < k; r2++) {
            const double f2 = G[(size_t)r2 * k + c] / dg;
            if (f2 == 0.0) continue;
            for (uint32_t j = c; j < k; j++) G[(size_t)r2 * k + j] -= f2 * G[(size_t)c * k + j];
            b[r2] -= f2 * b[c];
        }
    }
    for (int c = (int)k - 1; c >= 0; c--) {
        double a = b[c];
        for (uint32_t j = (uint32_t)c + 1; j < k; j++) a -= G[(size_t)c * k + j] * (double)ub[j];
        ub[c] = (float)(a / G[(size_t)c * k + c]);
    }
    for (uint32_t c = 0; c < k; c++) zm->z[c] = ub[c];   /* z 就地更新(天然 LoRA 位) */
    free(G); free(b); free(pv); free(ub); free(xp); free(dl);
    return 0;
}
