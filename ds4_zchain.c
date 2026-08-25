/* ds4_zchain.c -- DQZ2 sidecar loader + λ fold. See ds4_zchain.h for the math
 * contract; the byte format is defined by zchain_write() in
 * gguf-tools/go-onebit/quant/ds4quant_run.c:
 *   "DQZ2" u32 | n_layer u32 | per layer { L u32, n_ops u32,
 *       per op { type u32, paysz u32, payload } }
 *   payload: 1=GL f32 g | 2=GLdyn2 f32 w2p[4] | 3=GLdyn8 f32 w8[9] (+ fp16
 *   V8[8][d_model] when carried) | 4=TREF f32 t | 5=GE fp16[n_expert]. */
#include "ds4_zchain.h"

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
        for (uint32_t j = 0; j < d; j++) {
            nd += (double)delta[j] * delta[j];
            nr += (double)routed[j] * routed[j];
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
        nd += a * a;
        nr += (double)routed[j] * (double)routed[j];
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
