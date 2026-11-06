/* loss_norm=token: 每次参数更新按实际目标 token 数归一。
 * 长流最后可能只剩 7 位；按段平均会让它与 1022 位长段同权。
 * 分母在装包前合计，拆成几个 GPU 包、末批不足 batch 都不改每位权重。 */
#include "core_ptrain.h"
#ifndef DS4_NO_GPU

uint32_t pt_loss_offset(const pt_run *r, const pt_sample *s, uint32_t *pfx_out) {
    const uint32_t cap = r->st.cap_tok, f0 = s->sa0 + s->l0;
    uint32_t pfx = 0;
    if (s->sn > cap) {
        const uint32_t a = pt_pfx_align(), lo = (s->sn - cap + a - 1u) / a * a, hi = (f0 - 1u) / a * a;
        pfx = hi > lo ? hi : lo;
    }
    if (pfx_out) *pfx_out = pfx;
    const uint32_t a0 = f0 > pfx + 1u ? f0 : pfx + 1u;
    return a0 - s->sa0;
}

uint64_t pt_batch_loss_tokens(const pt_run *r, const pt_sample *const *ss, uint32_t n) {
    uint64_t total = 0;
    for (uint32_t k = 0; k < n; k++) {
        if (!pt_runnable(r, ss[k])) continue;
        const uint32_t off = pt_loss_offset(r, ss[k], NULL);
        if (off < ss[k]->m) total += ss[k]->m - off;
    }
    return total;
}

float pt_loss_scale(const pt_cfg *c, const pt_run *r, uint32_t m) {
    if (!m) return 0.f;
    if (!c->token_mean) return 1.0f / ((float)m * (float)c->batch);
    return 1.0f / (float)(r->grad_tokens ? r->grad_tokens : m);
}

#endif
typedef int ds4_core_ptrain_norm_nonempty_tu;
