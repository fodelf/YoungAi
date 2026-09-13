/* ds4_zfinetune.c -- 微调侧车与 zchain 的按秩拼接。契约见 ds4_zfinetune.h。 */
#include "ds4_zfinetune.h"
#include "ds4_z.h"
#include "src/common/ds4_amp_fmt.h"   /* 秩上限 DS4_AMP_ZK_MAX: 与 kernel/工具同一契约 */
#include "src/common/ds4_float.h"     /* f16→f32 唯一实现(重建 zmod 用) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 载荷布局(type6/7/8 同): z[k] | U[dout·k] | V[din·k], U/V 行主序 [dim][k]。
 * 按秩拼接 = 每一行把两段的 k 个数首尾接上 —— 全是 fp16 原值搬运, 不做任何
 * 数值转换, 所以合并后的产物与两个源文件逐位可追溯。 */
static uint16_t *zf_concat_payload(const uint16_t *a, uint32_t ka,
                                   const uint16_t *b, uint32_t kb,
                                   uint32_t dout, uint32_t din) {
    const uint32_t k = ka + kb;
    const size_t n = (size_t)k * (1u + dout + din);
    uint16_t *o = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!o) return NULL;
    memcpy(o, a, (size_t)ka * sizeof(uint16_t));                    /* z 段 */
    memcpy(o + ka, b, (size_t)kb * sizeof(uint16_t));
    uint16_t *oU = o + k;
    const uint16_t *aU = a + ka, *bU = b + kb;
    for (uint32_t r = 0; r < dout; r++) {
        memcpy(oU + (size_t)r * k, aU + (size_t)r * ka, (size_t)ka * sizeof(uint16_t));
        memcpy(oU + (size_t)r * k + ka, bU + (size_t)r * kb, (size_t)kb * sizeof(uint16_t));
    }
    uint16_t *oV = oU + (size_t)dout * k;
    const uint16_t *aV = aU + (size_t)dout * ka, *bV = bU + (size_t)dout * kb;
    for (uint32_t r = 0; r < din; r++) {
        memcpy(oV + (size_t)r * k, aV + (size_t)r * ka, (size_t)ka * sizeof(uint16_t));
        memcpy(oV + (size_t)r * k + ka, bV + (size_t)r * kb, (size_t)kb * sizeof(uint16_t));
    }
    return o;
}

/* 从 fp16 载荷重建 f32 的 ds4_z(线性 z 的 host apply 走它, 与反修解算同一份实现)。 */
static ds4_z *zf_zmod_from_payload(const uint16_t *p, uint32_t k, uint32_t dout, uint32_t din) {
    ds4_z *zm = (ds4_z *)calloc(1, sizeof(*zm));
    if (!zm) return NULL;
    zm->d_in = din; zm->d_out = dout; zm->rank = k; zm->k = k;
    zm->z = (float *)malloc((size_t)k * sizeof(float));
    zm->U = (float *)malloc((size_t)dout * k * sizeof(float));
    zm->V = (float *)malloc((size_t)din * k * sizeof(float));
    if (!zm->z || !zm->U || !zm->V) {
        free(zm->z); free(zm->U); free(zm->V); free(zm);
        return NULL;
    }
    const uint16_t *hU = p + k, *hV = hU + (size_t)dout * k;
    for (uint32_t c = 0; c < k; c++) zm->z[c] = ds4_f16_to_f32(p[c]);
    for (size_t i = 0; i < (size_t)dout * k; i++) zm->U[i] = ds4_f16_to_f32(hU[i]);
    for (size_t i = 0; i < (size_t)din * k; i++) zm->V[i] = ds4_f16_to_f32(hV[i]);
    return zm;
}

static void zf_zmod_free(ds4_z *zm) {
    if (!zm) return;
    free(zm->z); free(zm->U); free(zm->V); free(zm);
}

/* type8 路由侧车(RTE): δlogits = U·(z ⊙ tanh(Vᵀx/s)), U 是 [n_expert][k]。
 * 与 z^L 不同, 它【不能】按秩拼接 —— s 是每层一个标量, 两段各有各的 s, 拼起来算不出原式。
 * 所以规则是"只占空槽": 底座该层没有 RTE 才搬进去, 已有就拒绝(而不是悄悄覆盖掉底座的)。 */
static int zf_take_rte(ds4_zchain *base, const ds4_zchain *add, uint32_t il, uint32_t ne) {
    const ds4_zchain_zl *a = &add->layer[il].rte;
    ds4_zchain_zl *b = &base->layer[il].rte;
    const size_t n = (size_t)a->zlk * (1u + ne + a->zdin);
    uint16_t *pay = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!pay) { fprintf(stderr, "ds4: L%u RTE 载荷分配失败\n", il); return -1; }
    memcpy(pay, a->zlm, n * sizeof(uint16_t));
    free(base->layer[il].rtem_own);
    base->layer[il].rtem_own = pay;
    b->zlk = a->zlk; b->zdin = a->zdin; b->zltr = a->zltr; b->zmul = 2u; b->zlm = pay;
    return 0;
}

int ds4_zfinetune_merge(ds4_zchain *base, const ds4_zchain *add) {
    if (!base || !add) return -1;
    if (base->n_layer != add->n_layer) {
        fprintf(stderr, "ds4: finetune 层数 %u ≠ zchain 层数 %u\n", add->n_layer, base->n_layer);
        return -1;
    }
    const uint32_t dm = base->d_model, ne = base->n_expert;
    /* 先全量体检再动手: 任何一层不合规就整体拒绝, 不留"合并了一半"的模型 —— 半成品
     * 模型出的分没人能解释, 比不合并更坏。 */
    for (uint32_t il = 0; il < add->n_layer; il++) {
        const ds4_zchain_zl *r = &add->layer[il].rte;
        if (r->zlk) {                                            /* 路由形态: 只占空槽 */
            if (r->zdin != dm || !r->zlm) {
                fprintf(stderr, "ds4: finetune L%u RTE din=%u 非法\n", il, r->zdin); return -1; }
            if (base->layer[il].rte.zlk) {
                fprintf(stderr, "ds4: zchain L%u 已有 RTE, 两段各有各的 s 无法合并 — 拒绝\n", il);
                return -1;
            }
        }
        const ds4_zchain_zl *a = &add->layer[il].zl;
        const ds4_zchain_zl *b = &base->layer[il].zl;
        if (!a->zlk) continue;                                   /* 该层不微调 */
        if (a->zmul != 0u || a->zdin != dm || !a->zlm) {
            fprintf(stderr, "ds4: finetune L%u 不是加性线性 z(zmul=%u din=%u) — 拒绝合并\n",
                    il, a->zmul, a->zdin);
            return -1;
        }
        if (b->zlk && (b->zmul != 0u || b->zdin != dm || !b->zlm)) {
            fprintf(stderr, "ds4: zchain L%u 的 z^L 是乘性/ftA 形态(zmul=%u din=%u), 无法与加性微调合并\n",
                    il, b->zmul, b->zdin);
            return -1;
        }
        if ((uint64_t)b->zlk + a->zlk > DS4_AMP_ZK_MAX) {
            fprintf(stderr, "ds4: L%u 合并秩 %u+%u 超上限 %d\n",
                    il, b->zlk, a->zlk, (int)DS4_AMP_ZK_MAX);
            return -1;
        }
    }
    int merged = 0;
    for (uint32_t il = 0; il < add->n_layer; il++) {
        if (add->layer[il].rte.zlk) {
            if (zf_take_rte(base, add, il, ne) != 0) return -1;
            merged++;
        }
        const ds4_zchain_zl *a = &add->layer[il].zl;
        ds4_zchain_zl *b = &base->layer[il].zl;
        if (!a->zlk) continue;
        uint16_t *pay;
        uint32_t k;
        if (!b->zlk) {                                           /* 该层原本没有 z: 直接搬过来 */
            k = a->zlk;
            const size_t n = (size_t)k * (1u + dm + dm);
            pay = (uint16_t *)malloc(n * sizeof(uint16_t));
            if (pay) memcpy(pay, a->zlm, n * sizeof(uint16_t));
        } else {
            k = b->zlk + a->zlk;
            pay = zf_concat_payload(b->zlm, b->zlk, a->zlm, a->zlk, dm, dm);
        }
        if (!pay) { fprintf(stderr, "ds4: L%u 合并载荷分配失败\n", il); return -1; }
        ds4_z *zm = zf_zmod_from_payload(pay, k, dm, dm);
        if (!zm) { free(pay); fprintf(stderr, "ds4: L%u zmod 重建失败\n", il); return -1; }
        /* 夹持取两者较小(更保守): 合并后一个 clip 管两段, 见 .h 的"夹持语义变化"。 */
        const float tr = (b->zlk && b->zltr < a->zltr) ? b->zltr : a->zltr;
        zf_zmod_free(b->zmod);
        free(base->layer[il].zlm_own);
        base->layer[il].zlm_own = pay;
        b->zlm = pay; b->zlk = k; b->zdin = dm; b->zltr = tr; b->zmul = 0u; b->zmod = zm;
        merged++;
    }
    if (!merged) {
        fprintf(stderr, "ds4: finetune 里没有任何层带 z^L / RTE — 空侧车, 拒绝\n");
        return -1;
    }
    return merged;
}
