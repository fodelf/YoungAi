/* 排序项在训练/方向检查期间开启，统一评估入口关闭，评分仍是原纯 KL。 */
#include "core_ptrain.h"
#include "ds4_gpu_rankprob.h"
#include "ds4_gpu_keepmargin.h"
#include "core_ptrain_split.h"
#ifndef PT_TAIL_ALLKEEP
#define PT_TAIL_ALLKEEP 0
#endif
#ifndef PT_JSD_BETA
#define PT_JSD_BETA 0.f
#endif
#ifndef DS4_NO_GPU
static float pt_rankprob_lambda;
static bool pt_keep_training;
static bool pt_jsd_checked;

#ifndef __APPLE__
static bool jsd_fixture(float beta) {
    const uint32_t M = 3, V = 16, K = 3;
    const float sc = 0.37f, eps = 3e-3f;
    float z[48] = {
        3.0f, 0.5f, -0.5f, -1.0f, -1.2f, -1.4f, -1.5f, -1.6f, -1.7f, -1.8f, -1.9f, -2.0f, -2.1f, -2.2f, -2.3f, -2.4f,
        4.0f, -3.5f, 0.1f, -0.8f, -1.0f, -1.2f, -1.4f, -1.6f, -1.8f, -2.0f, -2.2f, -2.4f, -2.6f, -2.8f, -3.0f, -3.2f,
        2.2f, 1.5f, 0.2f, -8.0f, -8.5f, -9.0f, -9.5f, -10.f, -8.2f, -8.7f, -9.2f, -9.7f, -8.4f, -8.9f, -9.4f, -9.9f
    };
    const int32_t id[9] = { 0, 1, 2,  0, 1, 2,  0, 1, 2 };
    const float tp[9] = { 0.10f, 0.65f, 0.15f,  0.50f, 0.20f, 0.10f,  0.60f, 0.30f, 0.10f };
    const float rest[3] = { 0.10f, 0.20f, 0.0f };
    const size_t sz[6] = { sizeof z, sizeof id, sizeof tp, sizeof rest, sizeof z, M * sizeof(float) };
    ds4_gpu_tensor *p[6] = { 0 }; bool ok = true;
    float g[48], gip[48], lp[3], lm[3], gfkl[48];
    for (unsigned i = 0; i < 6; i++) p[i] = v41_alloc(sz[i], &ok);
    ok = ok && ds4_gpu_tensor_write(p[0], 0, z, sizeof z) && ds4_gpu_tensor_write(p[1], 0, id, sizeof id) &&
         ds4_gpu_tensor_write(p[2], 0, tp, sizeof tp) && ds4_gpu_tensor_write(p[3], 0, rest, sizeof rest) &&
         ds4_gpu_bwd_kl_margin(0.f, 0.f, 0.f) && ds4_gpu_bwd_kl_jsd(beta) &&
         ds4_gpu_bwd_kl_topk_tensor(p[4], p[5], p[0], 0, M, V, p[1], p[2], p[3], NULL, K, sc) &&
         ds4_gpu_tensor_read(p[4], 0, g, sizeof g) &&
         ds4_gpu_bwd_kl_topk_tensor(p[0], p[5], p[0], 0, M, V, p[1], p[2], p[3], NULL, K, sc) &&
         ds4_gpu_tensor_read(p[0], 0, gip, sizeof gip);
    double max_fd = 0.0, max_sum = 0.0, max_ip = 0.0, max_fkl = 0.0, max_ref = 0.0;
    for (unsigned i = 0; ok && i < M * V; i++) max_ip = fmax(max_ip, fabs((double)g[i] - gip[i]));
    for (unsigned r = 0; ok && r < M; r++) {
        double s = 0.0, mx = -1e30, qs = 0.0, rs = 0.0, ubar = 0.0;
        for (unsigned j = 0; j < V; j++) { s += g[r * V + j]; if (z[r * V + j] > mx) mx = z[r * V + j]; }
        max_sum = fmax(max_sum, fabs(s));
        double q[16];
        for (unsigned j = 0; j < V; j++) qs += (q[j] = exp(z[r * V + j] - mx));
        for (unsigned j = 0; j < V; j++) q[j] /= qs;
        for (unsigned j = K; j < V; j++) rs += q[j];
        const double b = (double)beta, om = 1.0 - b, mr = b * rest[r] + om * rs;
        const double ur = rs > 1e-30 ? log(rs / mr) : 0.0;
        for (unsigned k = 0; k < K; k++) {
            const double qk = q[id[r * K + k]], mk = b * tp[r * K + k] + om * qk;
            if (qk > 1e-30) ubar += qk * log(qk / mk);
        }
        ubar += rs * ur;
        for (unsigned j = 0; j < V; j++) {
            double uj = ur;
            for (unsigned k = 0; k < K; k++) if ((unsigned)id[r * K + k] == j) {
                const double mk = b * tp[r * K + k] + om * q[j];
                uj = q[j] > 1e-30 ? log(q[j] / mk) : 0.0;
            }
            const double refg = (double)sc * q[j] * (uj - ubar) / b;
            max_ref = fmax(max_ref, fabs((double)g[r * V + j] - refg));
        }
    }
    for (unsigned j = 0; ok && j < V; j++) {
        float zp[48], zm[48]; memcpy(zp, z, sizeof z); memcpy(zm, z, sizeof z);
        for (unsigned r = 0; r < M; r++) { zp[r * V + j] += eps; zm[r * V + j] -= eps; }
        ok = ds4_gpu_tensor_write(p[0], 0, zp, sizeof zp) &&
             ds4_gpu_bwd_kl_topk_tensor(p[4], p[5], p[0], 0, M, V, p[1], p[2], p[3], NULL, K, sc) &&
             ds4_gpu_tensor_read(p[5], 0, lp, sizeof lp) &&
             ds4_gpu_tensor_write(p[0], 0, zm, sizeof zm) &&
             ds4_gpu_bwd_kl_topk_tensor(p[4], p[5], p[0], 0, M, V, p[1], p[2], p[3], NULL, K, sc) &&
             ds4_gpu_tensor_read(p[5], 0, lm, sizeof lm);
        for (unsigned r = 0; ok && r < M; r++) {
            const double fd = ((double)lp[r] - lm[r]) / (2.0 * eps) * sc;
            max_fd = fmax(max_fd, fabs((double)g[r * V + j] - fd));
        }
    }
    ok = ok && ds4_gpu_bwd_kl_jsd(0.f) && ds4_gpu_tensor_write(p[0], 0, z, sizeof z) &&
         ds4_gpu_bwd_kl_topk_tensor(p[4], p[5], p[0], 0, M, V, p[1], p[2], p[3], NULL, K, sc) &&
         ds4_gpu_tensor_read(p[4], 0, gfkl, sizeof gfkl);
    for (unsigned r = 0; ok && r < M; r++) {
        double mx = -1e30, s = 0.0;
        for (unsigned j = 0; j < V; j++) if (z[r * V + j] > mx) mx = z[r * V + j];
        for (unsigned j = 0; j < V; j++) s += exp(z[r * V + j] - mx);
        for (unsigned k = 0; k < K; k++) {
            const double expg = sc * (exp(z[r * V + id[r * K + k]] - mx) / s - tp[r * K + k]);
            max_fkl = fmax(max_fkl, fabs((double)gfkl[r * V + id[r * K + k]] - expg));
        }
    }
    for (unsigned i = 0; i < 6; i++) if (p[i]) ds4_gpu_tensor_free(p[i]);
    ok = ok && isfinite(max_fd) && max_ref < 2e-6 && max_fd < 2e-4 && max_sum < 1e-6 && max_ip == 0.0 && max_fkl < 1e-6;
    fprintf(stderr, "[jsd] known_answer beta=%.6g max_ref_err=%.6g max_fd_err=%.6g max_sum=%.6g inplace_err=%.6g fkl0_err=%.6g %s\n",
            (double)beta, max_ref, max_fd, max_sum, max_ip, max_fkl, ok ? "PASS" : "FAIL");
    return ok;
}
#endif

bool pt_loss_config(const pt_cfg *c) {
    if (!isfinite(c->rankprob) || c->rankprob < 0.f || (c->rankprob > 0.f && (c->margin > 0.f || c->vq_nl || c->sample_n))) {
        fprintf(stderr, "ds4: rankprob要有限非负，不能与margin、V步或在线采样同时开\n"); return false;
    }
    return true;
}

bool pt_loss_mode(const pt_cfg *c, bool training) {
#ifdef __APPLE__
    if (c->rankprob > 0.f || PT_JSD_BETA > 0.f) { fprintf(stderr, "ds4: 概率排序/JSD 暂只支持 CUDA 后端\n"); return false; }
#else
    if (training && PT_JSD_BETA > 0.f && !pt_jsd_checked) {
        if (!jsd_fixture((float)PT_JSD_BETA)) return false;
        pt_jsd_checked = true;
    }
#endif
    pt_rankprob_lambda = training ? c->rankprob : 0.f;
    pt_keep_training = training;
    if (!ds4_gpu_bwd_kl_jsd(training ? (float)PT_JSD_BETA : 0.f)) {
        fprintf(stderr, "ds4: JSD(beta=%.6g) 参数无效或后端不支持\n", (double)PT_JSD_BETA);
        return false;
    }
    return ds4_gpu_bwd_kl_margin(training ? c->margin : 0.f, c->margin_p1, c->margin_gap);
}

bool pt_loss_kl(const pt_cfg *c, pt_run *r, ds4_gpu_tensor *g, ds4_gpu_tensor *logits,
                 uint32_t row, uint32_t m, float scale) {
#ifndef __APPLE__
    if((c->diag==16 || c->diag==18) && pt_keep_training && pt_protect_mask())
        return ds4_gpu_bwd_kl_keepmargin_tensor(g,r->loss,logits,row,m,DS4_N_VOCAB,r->tid,r->tp,r->trest,
            c->topk,scale,pt_protect_mask(),c->margin,
            c->diag==18 && PT_TAIL_ALLKEEP?1.01f:c->margin_p1,c->margin_gap);
    return ds4_gpu_bwd_kl_rankprob_tensor(g, r->loss, logits, row, m, DS4_N_VOCAB,
        r->tid, r->tp, r->trest, NULL, c->topk, scale, pt_rankprob_lambda);
#else
    return ds4_gpu_bwd_kl_topk_tensor(g, r->loss, logits, row, m, DS4_N_VOCAB,
        r->tid, r->tp, r->trest, NULL, c->topk, scale);
#endif
}
#endif
typedef int ds4_core_ptrain_rankprob_nonempty_tu;
