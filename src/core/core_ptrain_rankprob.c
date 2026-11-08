/* 排序项在训练/方向检查期间开启，统一评估入口关闭，评分仍是原纯 KL。 */
#include "core_ptrain.h"
#include "ds4_gpu_rankprob.h"
#include "ds4_gpu_keepmargin.h"
#include "core_ptrain_split.h"
#ifndef DS4_NO_GPU
static float pt_rankprob_lambda;
static bool pt_keep_training;

bool pt_loss_config(const pt_cfg *c) {
    if (!isfinite(c->rankprob) || c->rankprob < 0.f || (c->rankprob > 0.f && (c->margin > 0.f || c->vq_nl || c->sample_n))) {
        fprintf(stderr, "ds4: rankprob要有限非负，不能与margin、V步或在线采样同时开\n"); return false;
    }
    return true;
}

bool pt_loss_mode(const pt_cfg *c, bool training) {
#ifdef __APPLE__
    if (c->rankprob > 0.f) { fprintf(stderr, "ds4: 概率排序暂只支持 CUDA 后端\n"); return false; }
#endif
    pt_rankprob_lambda = training ? c->rankprob : 0.f;
    pt_keep_training = training;
    return ds4_gpu_bwd_kl_margin(training ? c->margin : 0.f, c->margin_p1, c->margin_gap);
}

bool pt_loss_kl(const pt_cfg *c, pt_run *r, ds4_gpu_tensor *g, ds4_gpu_tensor *logits,
                 uint32_t row, uint32_t m, float scale) {
#ifndef __APPLE__
    if(c->diag==16 && pt_keep_training && pt_protect_mask())
        return ds4_gpu_bwd_kl_keepmargin_tensor(g,r->loss,logits,row,m,DS4_N_VOCAB,r->tid,r->tp,r->trest,
            c->topk,scale,pt_protect_mask(),c->margin,c->margin_p1,c->margin_gap);
    return ds4_gpu_bwd_kl_rankprob_tensor(g, r->loss, logits, row, m, DS4_N_VOCAB,
        r->tid, r->tp, r->trest, NULL, c->topk, scale, pt_rankprob_lambda);
#else
    return ds4_gpu_bwd_kl_topk_tensor(g, r->loss, logits, row, m, DS4_N_VOCAB,
        r->tid, r->tp, r->trest, NULL, c->topk, scale);
#endif
}
#endif
typedef int ds4_core_ptrain_rankprob_nonempty_tu;
