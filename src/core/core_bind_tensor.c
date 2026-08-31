/* core_bind_tensor.c — required 族/tensor_expect 族绑定校验原语 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
/* =========================================================================
 * Fixed Weight Binding and Model Validation.
 * =========================================================================
 *
 * The GGUF tensor directory is converted into a DS4-specific pointer table.
 * After this section, the rest of the program addresses tensors by semantic
 * fields such as layer->attn_q_a or layer->ffn_gate_exps rather than by string
 * lookup.  Shape validation is intentionally strict.
 */

uint32_t required_u32(const ds4_model *m, const char *key) {
    uint32_t v = 0;
    if (!model_get_u32(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

float required_f32(const ds4_model *m, const char *key) {
    float v = 0.0f;
    if (!model_get_f32_compat(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

bool required_bool(const ds4_model *m, const char *key) {
    bool v = false;
    if (!model_get_bool(m, key, &v)) {
        fprintf(stderr, "ds4: required metadata key is missing: %s\n", key);
        exit(1);
    }
    return v;
}

static ds4_tensor *required_tensor(const ds4_model *m, const char *name) {
    ds4_tensor *t = model_find_tensor(m, name);
    if (!t) {
        fprintf(stderr, "ds4: required tensor is missing: %s\n", name);
        exit(1);
    }
    return t;
}

ds4_tensor *tensor_by_namef(const ds4_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) ds4_die("tensor name is too long");
    return model_find_tensor(m, name);
}

ds4_tensor *required_tensorf(const ds4_model *m, const char *fmt, uint32_t layer) {
    char name[128];
    int n = snprintf(name, sizeof(name), fmt, layer);
    if (n < 0 || (size_t)n >= sizeof(name)) ds4_die("tensor name is too long");
    return required_tensor(m, name);
}

static void tensor_expect_layout(
        const ds4_tensor *t,
        uint32_t          type,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) return;  /* sharded per-machine slice: tensors for the other machine's
                      * layers (and the head half not held here) are absent -> skip.
                      * Present layers are still validated; weights_bind's
                      * required_tensorf guarantees a present layer is complete. */
    if (t->type != type) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %s, expected %s\n",
                (int)t->name.len,
                t->name.ptr,
                tensor_type_name(t->type),
                tensor_type_name(type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr,
                "ds4: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len,
                t->name.ptr,
                t->ndim,
                ndim);
        exit(1);
    }

    const uint64_t want[3] = { d0, d1, d2 };
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr,
                "ds4: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len,
                t->name.ptr,
                i,
                t->dim[i],
                want[i]);
        exit(1);
    }
}

static void tensor_expect_optional(
        const ds4_tensor *t,
        uint32_t          type,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (t) tensor_expect_layout(t, type, ndim, d0, d1, d2);
}

static bool tensor_is_routed_expert_type(uint32_t type) {
    return type == DS4_TENSOR_IQ2_XXS ||
           type == DS4_TENSOR_Q2_K ||
           type == DS4_TENSOR_Q4_K ||
           type == DS4_TENSOR_GO1B ||
           type == DS4_TENSOR_GO2B;
}

static DS4_MAYBE_UNUSED uint64_t routed_expert_block_bytes(uint32_t type) {
    switch (type) {
    case DS4_TENSOR_IQ2_XXS: return sizeof(block_iq2_xxs);
    case DS4_TENSOR_Q2_K:    return sizeof(block_q2_K);
    case DS4_TENSOR_Q4_K:    return sizeof(block_q4_K);
    case DS4_TENSOR_GO1B:    return sizeof(block_go1b);
    case DS4_TENSOR_GO2B:    return sizeof(block_go2b);
    default:                 ds4_die("unsupported routed expert tensor type");
    }
    return 0;
}

DS4_MAYBE_UNUSED uint64_t routed_expert_row_bytes(const ds4_tensor *t) {
    if ((t->dim[0] % QK_K) != 0) ds4_die("routed expert row is not QK_K aligned");
    return (t->dim[0] / QK_K) * routed_expert_block_bytes(t->type);
}

static void tensor_expect_routed_expert(
        const ds4_tensor *t,
        uint32_t          ndim,
        uint64_t          d0,
        uint64_t          d1,
        uint64_t          d2) {
    if (!t) return;  /* sharded per-machine slice: routed experts of the other
                      * machine's layers are absent -> skip (see tensor_expect_layout). */
    if (!tensor_is_routed_expert_type(t->type)) {
        fprintf(stderr,
                "ds4: tensor %.*s has type %u (%s), expected a routed expert quant type\n",
                (int)t->name.len,
                t->name.ptr,
                t->type,
                tensor_type_name(t->type));
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr,
                "ds4: tensor %.*s has %u dimensions, expected %u\n",
                (int)t->name.len,
                t->name.ptr,
                t->ndim,
                ndim);
        exit(1);
    }

    const uint64_t want[3] = { d0, d1, d2 };
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] == want[i]) continue;
        fprintf(stderr,
                "ds4: tensor %.*s has dim[%u]=%" PRIu64 ", expected %" PRIu64 "\n",
                (int)t->name.len,
                t->name.ptr,
                i,
                t->dim[i],
                want[i]);
        exit(1);
    }
}

/* Verify every tensor type and dimension used by the specialized pipeline.
 * After this succeeds, inference code can rely on fixed DS4 constants. */
/* 合一 VQ GGUF(2026-07-27): base 的 ffn_{gate,up}_exps(go1b, 被 VQ 覆盖的死重 22.85 GiB)
 * 不再入文件, 专家维度/量化类型/偏移改由这些 helper 供给 — gate 在场走原值(所有旧文件
 * 字节不变), 缺席(内嵌 VQ)时维度取自 down 张量(in=down.dim[1], mid=down.dim[0]),
 * 类型按 GO1B 报(dispatch 走 batch mm_id 路 = VQ 消费端所在), 偏移 0(do_vq 不读)。 */
uint64_t routed_expert_in_dim(const ds4_layer_weights *l) {
    return l->ffn_gate_exps ? l->ffn_gate_exps->dim[0] : l->ffn_down_exps->dim[1];
}
uint64_t routed_expert_mid_dim(const ds4_layer_weights *l) {
    return l->ffn_gate_exps ? l->ffn_gate_exps->dim[1] : l->ffn_down_exps->dim[0];
}
uint32_t routed_expert_quant_type(const ds4_layer_weights *l) {
    return l->ffn_gate_exps ? l->ffn_gate_exps->type : (uint32_t)DS4_TENSOR_GO1B;
}
uint64_t routed_expert_gate_off(const ds4_layer_weights *l) {
    return l->ffn_gate_exps ? l->ffn_gate_exps->abs_offset : 0;
}
uint64_t routed_expert_up_off(const ds4_layer_weights *l) {
    return l->ffn_up_exps ? l->ffn_up_exps->abs_offset : 0;
}

/* R28 VQ(2026-07-31): 计划表 w2dim>0 时冷 w2 的 VQ 载荷也在层 blob 里(槽 which=2),
 * 于是 base 的 ffn_down_exps(go1b 死重, 43 层 11.42 GiB)不再入合一文件 —— 这是 28 GiB
 * 总量能成立的前提。缺席时合成一个"影子张量": 形状/类型按 DS4 常量填, bytes=0 且
 * abs_offset=0。span 构建见 bytes==0 即跳过(model_map_span_include_tensor), 所以一个
 * 字节都不会被 mmap、也不进 Metal residency; 而图/校验/分布式里三十余处 ->dim/->type
 * 解引用语义完全不变, 不必逐点加 NULL 判(那种改法漏一处就是运行期空指针)。
 * 唯一真去读 down 字节的是 VQ gather 的冷-w2-回退分支(blob 槽为 0 时从 base go1b 展开
 * ±d), 那里按 down_expert_bytes==0 硬失败, 不静默降级成垃圾权重。 */
ds4_tensor *routed_down_shadow(uint32_t il) {
    static ds4_tensor *shadow;      /* [DS4_N_LAYER]; DS4_N_LAYER 是运行期形状, 故堆分配 */
    static char (*names)[40];
    if (!shadow) {
        shadow = xcalloc(DS4_N_LAYER, sizeof(*shadow));
        names  = xcalloc(DS4_N_LAYER, sizeof(*names));
    }
    ds4_tensor *t = &shadow[il];
    if (t->ndim == 0) {
        int n = snprintf(names[il], sizeof(names[il]), "blk.%u.ffn_down_exps.weight", il);
        t->name.ptr = names[il];
        t->name.len = (uint64_t)(n < 0 ? 0 : n);
        t->ndim = 3;
        t->dim[0] = DS4_N_FF_EXP;   /* down_in_dim  = 专家中间维 */
        t->dim[1] = DS4_N_EMBD;     /* routed_out_dim = 残差流维 */
        t->dim[2] = DS4_N_EXPERT;
        t->type = DS4_TENSOR_GO1B;  /* dispatch 与 gate/up 缺席时同口径 */
        t->elements = t->dim[0] * t->dim[1] * t->dim[2];
        t->rel_offset = t->abs_offset = t->bytes = 0;
    }
    return t;
}

/* 全q2(2026-08-19): f16 专线家族允许 Q2_K, 装载即注册 f16 影子(引擎适配, 文件全 q2)。 */
static void tensor_expect_f16_q2(const ds4_model *m, ds4_tensor *t, uint64_t d0, uint64_t d1) {
    if (t && t->type == DS4_TENSOR_Q2_K) {
        tensor_expect_layout(t, DS4_TENSOR_Q2_K, 2, d0, d1, 0);
#ifndef DS4_NO_GPU
        if (!ds4_gpu_register_q2k_f16_shadow(m->map, m->size, t->abs_offset, d1, d0))
            ds4_die("全q2: f16 影子注册失败(行长须为256倍数且显存足)");
        /* 数据面已由影子替换成 f16(range_ptr 按 offset 命中优先于一切界检), 类型面必须
         * 跟着翻——图编码把 t->type 一路传给 GPU 分发, 留着 q2_k 会被 f16 专线 kernel
         * 拒收(L2 首个压缩层 attention_batch 静默失败即此)。t->bytes 保持文件真值,
         * span/mmap 账仍按 q2 字节算。 */
        t->type = DS4_TENSOR_F16;
#else
        ds4_die("全q2 模型的 f16 专线家族需 GPU 后端(CPU 参考未适配)");
#endif
    } else {
        tensor_expect_layout(t, DS4_TENSOR_F16, 2, d0, d1, 0);
    }
}

void weights_validate_layout(const ds4_model *m, const ds4_weights *w) {
    const uint64_t hc_dim = (uint64_t)DS4_N_EMBD * DS4_N_HC;
    const uint64_t hc_mix_dim = 2u * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t out_low_dim = (uint64_t)DS4_N_OUT_GROUP * DS4_N_LORA_O;

    tensor_expect_f16_q2(m, w->token_embd, DS4_N_EMBD, DS4_N_VOCAB);
    tensor_expect_layout(w->output_hc_base,  DS4_TENSOR_F32,  1, DS4_N_HC, 0, 0);
    tensor_expect_f16_q2(m, w->output_hc_fn, hc_dim, DS4_N_HC);
    tensor_expect_layout(w->output_hc_scale, DS4_TENSOR_F32,  1, 1, 0, 0);
    tensor_expect_layout(w->output_norm,     DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
    tensor_expect_layout(w->output,          w->output && (w->output->type == DS4_TENSOR_Q4_K || w->output->type == DS4_TENSOR_Q2_K) ? w->output->type : DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_VOCAB, 0);

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &w->layer[il];
        const uint32_t ratio = ds4_layer_compress_ratio(il);

        /* Sharded per-machine slice: a layer owned by the other machine is absent
         * (weights_bind left it zeroed) -> skip its validation entirely. Some
         * checks here dereference tensor dims directly, so the per-call NULL
         * guards in tensor_expect_* are not enough on their own. */
        if (!l->attn_norm) continue;

        tensor_expect_f16_q2(m, l->hc_attn_fn, hc_dim, hc_mix_dim);
        tensor_expect_layout(l->hc_attn_scale,  DS4_TENSOR_F32,  1, 3, 0, 0);
        tensor_expect_layout(l->hc_attn_base,   DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
        tensor_expect_layout(l->attn_norm,      DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
        tensor_expect_layout(l->attn_q_a,       l->attn_q_a && (l->attn_q_a->type == DS4_TENSOR_Q4_K || l->attn_q_a->type == DS4_TENSOR_Q2_K) ? l->attn_q_a->type : DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_LORA_Q, 0);
        tensor_expect_layout(l->attn_q_a_norm,  DS4_TENSOR_F32,  1, DS4_N_LORA_Q, 0, 0);
        tensor_expect_layout(l->attn_q_b,       l->attn_q_b && (l->attn_q_b->type == DS4_TENSOR_Q4_K || l->attn_q_b->type == DS4_TENSOR_Q2_K) ? l->attn_q_b->type : DS4_TENSOR_Q8_0, 2, DS4_N_LORA_Q, q_dim, 0);
        tensor_expect_layout(l->attn_kv,        l->attn_kv && (l->attn_kv->type == DS4_TENSOR_Q4_K || l->attn_kv->type == DS4_TENSOR_Q2_K) ? l->attn_kv->type : DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_HEAD_DIM, 0);
        tensor_expect_layout(l->attn_kv_a_norm, DS4_TENSOR_F32,  1, DS4_N_HEAD_DIM, 0, 0);
        tensor_expect_layout(l->attn_sinks,     DS4_TENSOR_F32,  1, DS4_N_HEAD, 0, 0);
        tensor_expect_layout(l->attn_output_a,  l->attn_output_a && (l->attn_output_a->type == DS4_TENSOR_Q4_K || l->attn_output_a->type == DS4_TENSOR_Q2_K) ? l->attn_output_a->type : DS4_TENSOR_Q8_0, 2, DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP), out_low_dim, 0);
        tensor_expect_layout(l->attn_output_b,  l->attn_output_b && (l->attn_output_b->type == DS4_TENSOR_Q4_K || l->attn_output_b->type == DS4_TENSOR_Q2_K) ? l->attn_output_b->type : DS4_TENSOR_Q8_0, 2, out_low_dim, DS4_N_EMBD, 0);

        if (ratio != 0) {
            const uint64_t comp_width = ds4_comp_row_width(ratio, DS4_N_HEAD_DIM);
            tensor_expect_f16_q2(m, l->attn_compressor_ape, comp_width, ratio);
            tensor_expect_f16_q2(m, l->attn_compressor_kv, DS4_N_EMBD, comp_width);
            tensor_expect_f16_q2(m, l->attn_compressor_gate, DS4_N_EMBD, comp_width);
            tensor_expect_layout(l->attn_compressor_norm, DS4_TENSOR_F32, 1, DS4_N_HEAD_DIM, 0, 0);
        }
        if (ratio == 4) {
            const uint64_t index_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
            const uint64_t index_width = 2u * DS4_N_INDEXER_HEAD_DIM;
            tensor_expect_f16_q2(m, l->indexer_attn_q_b, DS4_N_LORA_Q, index_q_dim);
            tensor_expect_f16_q2(m, l->indexer_proj, DS4_N_EMBD, DS4_N_INDEXER_HEAD);
            tensor_expect_f16_q2(m, l->indexer_compressor_ape, index_width, ratio);
            tensor_expect_f16_q2(m, l->indexer_compressor_kv, DS4_N_EMBD, index_width);
            tensor_expect_f16_q2(m, l->indexer_compressor_gate, DS4_N_EMBD, index_width);
            tensor_expect_layout(l->indexer_compressor_norm,   DS4_TENSOR_F32, 1, DS4_N_INDEXER_HEAD_DIM, 0, 0);
        }

        tensor_expect_f16_q2(m, l->hc_ffn_fn, hc_dim, hc_mix_dim);
        tensor_expect_layout(l->hc_ffn_scale,   DS4_TENSOR_F32,  1, 3, 0, 0);
        tensor_expect_layout(l->hc_ffn_base,    DS4_TENSOR_F32,  1, hc_mix_dim, 0, 0);
        tensor_expect_layout(l->ffn_norm,       DS4_TENSOR_F32,  1, DS4_N_EMBD, 0, 0);
        tensor_expect_f16_q2(m, l->ffn_gate_inp, DS4_N_EMBD, DS4_N_EXPERT);
        tensor_expect_optional(l->ffn_exp_probs_b, DS4_TENSOR_F32, 1, DS4_N_EXPERT, 0, 0);
        /* A shrunken (keep-map) model carries only the kept routed experts; the
         * router + ffn_gate_inp + ffn_exp_probs_b stay 256-wide above. */
        const uint64_t exp_dim = model_expert_kept_count(m, il);
        /* 内嵌 VQ 合一文件: gate/up 死重不入文件(blob 张量替代), 仅 down(冷 w2 源)必在。 */
        if (l->ffn_gate_exps)
            tensor_expect_routed_expert(l->ffn_gate_exps, 3, DS4_N_EMBD, DS4_N_FF_EXP, exp_dim);
        if (l->ffn_up_exps)
            tensor_expect_routed_expert(l->ffn_up_exps,   3, DS4_N_EMBD, DS4_N_FF_EXP, exp_dim);
        tensor_expect_routed_expert(l->ffn_down_exps, 3, DS4_N_FF_EXP, DS4_N_EMBD, exp_dim);
        if (l->ffn_gate_exps && l->ffn_up_exps &&
            l->ffn_gate_exps->type != l->ffn_up_exps->type) {
            fprintf(stderr, "ds4: routed gate/up experts use different quant types in layer %u\n", il);
            exit(1);
        }
        tensor_expect_layout(l->ffn_gate_shexp, l->ffn_gate_shexp && (l->ffn_gate_shexp->type == DS4_TENSOR_Q4_K || l->ffn_gate_shexp->type == DS4_TENSOR_Q2_K) ? l->ffn_gate_shexp->type : DS4_TENSOR_Q8_0,    2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
        tensor_expect_layout(l->ffn_up_shexp,   l->ffn_up_shexp && (l->ffn_up_shexp->type == DS4_TENSOR_Q4_K || l->ffn_up_shexp->type == DS4_TENSOR_Q2_K) ? l->ffn_up_shexp->type : DS4_TENSOR_Q8_0,    2, DS4_N_EMBD, DS4_N_FF_EXP, 0);
        tensor_expect_layout(l->ffn_down_shexp, l->ffn_down_shexp && (l->ffn_down_shexp->type == DS4_TENSOR_Q4_K || l->ffn_down_shexp->type == DS4_TENSOR_Q2_K) ? l->ffn_down_shexp->type : DS4_TENSOR_Q8_0,    2, DS4_N_FF_EXP, DS4_N_EMBD, 0);
        if (il < DS4_N_HASH_LAYER) {
            tensor_expect_layout(l->ffn_gate_tid2eid, DS4_TENSOR_I32, 2, DS4_N_EXPERT_USED, DS4_N_VOCAB, 0);
        }
    }
}

