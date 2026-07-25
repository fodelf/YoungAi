/* ds4_corr.c — go1b z-sidecar (三件套②③) runtime module.
 *
 * Loads the per-layer four-loss correction sidecar (blk.{L}.corr_*),
 * publishes host pointers + resident GPU copies, and provides the CPU
 * reference mirror of the corr kernel. Dispatch call-sites live in the
 * decode/prefill graph (ds4.c); the plugin registry / hot-swap for the
 * multi-domain sidecar feature (R3-h) lands here. Moved verbatim from
 * ds4.c (R3-g P1a); no behavior change.
 */
#include <stdlib.h>
#include <string.h>
#include "ds4_internal.h"

void corr_free(struct ds4_corr *corr) {
    if (!corr) return;
#ifndef DS4_NO_GPU
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_corr_layer *cl = &corr->layer[il];
        ds4_gpu_tensor_free(cl->gU);
        ds4_gpu_tensor_free(cl->gV);
        ds4_gpu_tensor_free(cl->gC);
        ds4_gpu_tensor_free(cl->gb);
        ds4_gpu_tensor_free(cl->gbeta);
        ds4_gpu_tensor_free(cl->gdelta);
    }
#endif
    model_close(&corr->sidecar);
    free(corr);
}

/* Look up a required corr tensor for layer il, validate it is F32 with the given
 * GGUF ne dims (inner dim first, 0 = unused), and return its host payload. */
static const float *corr_tensor(const ds4_model *m, uint32_t il, const char *suffix,
                                uint32_t ndim, uint64_t d0, uint64_t d1) {
    char name[128];
    snprintf(name, sizeof(name), "blk.%u.corr_%s", il, suffix);
    ds4_tensor *t = model_find_tensor(m, name);
    if (!t) { fprintf(stderr, "ds4: corr sidecar missing tensor %s\n", name); exit(1); }
    if (t->type != DS4_TENSOR_F32 || t->ndim != ndim ||
        t->dim[0] != d0 || (ndim >= 2 && t->dim[1] != d1)) {
        fprintf(stderr, "ds4: corr tensor %s has unexpected type/shape\n", name);
        exit(1);
    }
    return (const float *)tensor_data(m, t);
}

#ifndef DS4_NO_GPU
/* Allocate a resident GPU buffer and copy the F32 sidecar payload into it. */
static ds4_gpu_tensor *corr_upload(const float *host, uint64_t count) {
    ds4_gpu_tensor *g = ds4_gpu_tensor_alloc(count * sizeof(float));
    if (!g) ds4_die("corr: GPU buffer allocation failed");
    if (!ds4_gpu_tensor_write(g, 0, host, count * sizeof(float))) {
        ds4_gpu_tensor_free(g);
        ds4_die("corr: GPU buffer upload failed");
    }
    return g;
}
#endif

/* Open the correction sidecar GGUF and bind per-layer corr tensors (resident).
 * Returns NULL (and leaves the engine on the pure 1-bit path) when the file has
 * no usable correction (ds4.corr.present absent/false). d_model/n_expert are the
 * fixed model shape; d_l is recovered per layer from corr_C ne[0]. */
struct ds4_corr *corr_load(const char *path, bool metal_mapping) {
    struct ds4_corr *corr = xcalloc(1, sizeof(*corr));
    model_open(&corr->sidecar, path, metal_mapping, false);

    bool present = false;
    if (!model_get_bool(&corr->sidecar, "ds4.corr.present", &present) || !present) {
        fprintf(stderr, "ds4: corr sidecar %s lacks ds4.corr.present=true; ignoring\n", path);
        model_close(&corr->sidecar);
        free(corr);
        return NULL;
    }
    bool phi_yhat = false;
    model_get_bool(&corr->sidecar, "ds4.corr.phi_yhat", &phi_yhat);   /* absent → x (legacy) */
    corr->phi_yhat = phi_yhat;

    const uint64_t d_model = DS4_N_EMBD;
    const uint64_t n_exp = DS4_N_EXPERT;
    uint32_t loaded = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        char cname[128];
        snprintf(cname, sizeof(cname), "blk.%u.corr_C", il);
        ds4_tensor *tc = model_find_tensor(&corr->sidecar, cname);
        if (!tc) continue;                    /* layer not corrected (e.g. absent slice) */
        if (tc->ndim != 2 || tc->dim[1] != n_exp || tc->dim[0] == 0) {
            ds4_die("corr: corr_C has unexpected shape");
        }
        const uint64_t d_l = tc->dim[0];      /* low-rank width, recovered per layer */

        ds4_corr_layer *cl = &corr->layer[il];
        cl->d_l   = (uint32_t)d_l;
        cl->U     = corr_tensor(&corr->sidecar, il, "U",     2, d_l,     d_model);
        cl->V     = corr_tensor(&corr->sidecar, il, "V",     2, d_model, d_l);
        cl->C     = corr_tensor(&corr->sidecar, il, "C",     2, d_l,     n_exp);
        cl->b     = corr_tensor(&corr->sidecar, il, "b",     1, d_model, 0);
        cl->beta  = corr_tensor(&corr->sidecar, il, "beta",  1, n_exp,   0);
        cl->delta = corr_tensor(&corr->sidecar, il, "delta", 1, n_exp,   0);
        cl->has_delta = false;
        for (uint64_t e = 0; e < n_exp; e++)
            if (cl->delta[e] != 0.0f) { cl->has_delta = true; break; }
#ifndef DS4_NO_GPU
        cl->gU     = corr_upload(cl->U,     d_model * d_l);
        cl->gV     = corr_upload(cl->V,     d_l * d_model);
        /* DS4_CORR_SCALE=α (默认1): 逐层修正阻尼, 压 23 层激活空间 corr 的联合复利
         * 爆炸(2026-07-23 找到根因: 单层可辨识/23层乱码)。C 是 mmap 只读→缩放副本上传。 */
        const char *cs = getenv("DS4_CORR_SCALE");
        float corr_scale = cs ? (float)atof(cs) : 1.0f;
        if (corr_scale != 1.0f) {
            float *Cs = xmalloc((size_t)n_exp * d_l * sizeof(float));
            for (size_t i = 0; i < (size_t)n_exp * d_l; i++) Cs[i] = cl->C[i] * corr_scale;
            cl->gC = corr_upload(Cs, n_exp * d_l);
            free(Cs);
        } else
        cl->gC     = corr_upload(cl->C,     n_exp * d_l);
        cl->gb     = corr_upload(cl->b,     d_model);
        cl->gbeta  = corr_upload(cl->beta,  n_exp);
        cl->gdelta = corr_upload(cl->delta, n_exp);
#endif
        cl->present = true;
        loaded++;
    }

    if (loaded == 0) {
        fprintf(stderr, "ds4: corr sidecar %s carried no per-layer corr tensors; ignoring\n", path);
        corr_free(corr);
        return NULL;
    }
    corr->present = true;
    fprintf(stderr, "ds4: go1b correction loaded from %s (%u/%u layers)\n",
            path, loaded, (uint32_t)DS4_N_LAYER);
    return corr;
}

const float *corr_layer_delta(const ds4_model *m, uint32_t il) {
    if (!m || !m->corr || il >= DS4_MAX_LAYER || !m->corr->layer[il].present) return NULL;
    return m->corr->layer[il].delta;
}

/* Host-side correction term: out[d] += sum over selected e of
 *   ( U @ ( C[e] (.*) (V @ x) ) )[d] + b[d] + beta[e].
 * Mirrors metal/cuda kernel_dsv4_corr_apply for the CPU reference path. */
void corr_apply_moe_host(float *out, const ds4_model *m, uint32_t il,
                                const float *x, const int *selected, uint32_t n_sel) {
    if (!m || !m->corr || il >= DS4_MAX_LAYER || !m->corr->layer[il].present) return;
    const ds4_corr_layer *cl = &m->corr->layer[il];
    const uint32_t d_model = DS4_N_EMBD;
    const uint32_t d_l = cl->d_l;
    const uint32_t n_exp = DS4_N_EXPERT;

    /* φ selector mirrors the GPU dispatch: yhat sidecars project the routed
     * output itself. vx is fully computed before out is written below. */
    const float *feat = m->corr->phi_yhat ? out : x;
    float *vx = xmalloc((size_t)d_l * sizeof(float));
    for (uint32_t i = 0; i < d_l; i++) {
        const float *Vr = cl->V + (size_t)i * d_model;
        float acc = 0.0f;
        for (uint32_t j = 0; j < d_model; j++) acc += Vr[j] * feat[j];
        vx[i] = acc;
    }
    for (uint32_t s = 0; s < n_sel; s++) {
        const int e = selected[s];
        if (e < 0 || (uint32_t)e >= n_exp) continue;
        const float *Cr = cl->C + (size_t)e * d_l;
        const float be = cl->beta[e];
        for (uint32_t d = 0; d < d_model; d++) {
            const float *Ur = cl->U + (size_t)d * d_l;
            float p = 0.0f;
            for (uint32_t i = 0; i < d_l; i++) p += Ur[i] * Cr[i] * vx[i];
            out[d] += p + cl->b[d] + be;
        }
    }
    free(vx);
}
