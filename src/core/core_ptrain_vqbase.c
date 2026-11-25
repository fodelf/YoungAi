/* 低秩续训也要加载部署时已有的索引补丁。只在判决时拼上补丁，
 * 会让梯度针对另一个模型；因此训练时冻结补丁，落盘时一起保存。 */
#include "core_ptrain.h"
#include "core_v41_exit.h"
#ifndef DS4_NO_GPU

bool pt_frozen_vq_load(const pt_cfg *c) {
    const char *ed = c->init[0] ? c->init : c->vq_init;
    if (!c->vq_nl && ed[0] && v41_exit_load(ed, NULL) < 0) return false;
    if (c->vq_nl || !c->vq_init[0]) return true;
    if (!c->teacher_fp[0] || c->sample_n) {
        fprintf(stderr, "ds4: [ptrain] 固定索引补丁续训要求 teacher=fp: 且 sample_n=0；在线教师会切换插件状态，当前不支持\n");
        return false;
    }
    /* 补丁在插件缓存之外切换；清掉旧缓存，随后探针才会按自己的目录重装。训练张量已独占，不受影响。 */
    v41_amp_cache_drop();
    const int n = v41_vqpatch_load(c->vq_init);
    if (n <= 0) fprintf(stderr, "ds4: [ptrain] 固定补丁 %s 未加载到有效层，停止\n", c->vq_init);
    return n > 0 && (!ed[0] || v41_exit_load(ed, NULL) >= 0);
}

bool pt_frozen_vq_save(const pt_cfg *c, const char *dir) {
    if (!c->vq_nl && v41_exit_active() && v41_exit_save(dir) < 0) return false;
    if (c->vq_nl || !c->vq_init[0]) return true;
    const int n = v41_vqpatch_save(dir);
    if (n <= 0) fprintf(stderr, "ds4: [ptrain] 固定补丁未能随 %s 保存，产物不可部署\n", dir);
    return n > 0;
}

#endif
typedef int ds4_core_ptrain_vqbase_nonempty_tu;
