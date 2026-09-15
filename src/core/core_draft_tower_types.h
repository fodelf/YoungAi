/* core_draft_tower_types.h — DSpark 三塔的权重表(speed.md 段 6, 2026-09-15)。
 *
 * 为什么单起一个头: core_types.h 已经 500 行顶格(make linecount 硬规矩), 而三塔这一坨是一件完整的事。
 * 为什么不叫 ds4_mtp_weights: 那个名字被 V4 时代的草稿器占了(core_types.h 里还留着)。按机制命名 = draft tower。
 *
 * 官方 model.py 的 `self.mtp` 是三个 SWA(128) 草稿块。**每塔的结构与一个普通层一模一样**
 * (hc 六件 + attn 五投影 + ffn gate/专家/shared), 所以直接复用 ds4_layer_weights —— 唯一的差别是
 * 专家不走 VQ blob 而是逐专家 FP4 张量(speed.md §2: "草稿质量 = 接受率 = 速度, 不省这里")。
 * 塔数与每塔专家数一律从 GGUF 元数据读(`deepseek4.mtp.tower_count/expert_count`), 不写死 ——
 * 守用户 2026-09-15 定的"引擎禁版本名/魔数硬编码"铁律。 */
#ifndef DS4_CORE_DRAFT_TOWER_TYPES_H
#define DS4_CORE_DRAFT_TOWER_TYPES_H

typedef struct {
    ds4_layer_weights tower[DS4_MTP_MAX_TOWERS];
    ds4_tensor *exp_gate[DS4_MTP_MAX_TOWERS][DS4_MTP_MAX_EXPERTS];
    ds4_tensor *exp_up[DS4_MTP_MAX_TOWERS][DS4_MTP_MAX_EXPERTS];
    ds4_tensor *exp_down[DS4_MTP_MAX_TOWERS][DS4_MTP_MAX_EXPERTS];
    /* 五个只此一份的头(官方 mtp.0 带前两个, mtp.2 带后三个) */
    ds4_tensor *main_proj;    /* [5120][15360]: 主模型 L37/38/39 的注意力输入(hc 四路均值)拼接后投回 5120 = main_x,
                               * 它是块注意力的主 KV 来源。★取错位置不报错, 只是接受率掉到 1 附近★ */
    ds4_tensor *main_norm;
    ds4_tensor *markov_embd;  /* 秩 256: 按块内前一个草稿 token 给 logit 加偏置, 逐位贪心 */
    ds4_tensor *markov_head;
    ds4_tensor *confidence;   /* 出每一位的条件接受概率; 调度器据此 × 实测吞吐曲线定每轮验证几个 */
    ds4_tensor *out_norm;     /* 三塔共用的出口 norm(官方 self.mtp[-1] 借主模型的 embed/head, GGUF 不再存一份) */
} ds4_draft_tower_weights;

#endif /* DS4_CORE_DRAFT_TOWER_TYPES_H */
