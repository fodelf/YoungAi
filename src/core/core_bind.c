/* core_bind.c — weights_bind + model map span (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
void weights_bind(ds4_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));
    if (DS4_MODEL_VARIANT == DS4_VARIANT_V41) { weights_bind_v41(w, m); return; }   /* V4.1 张量名/类型另一套 */
    /* Head tensors are optional so a sharded per-machine slice can hold only the
     * half it needs: the coordinator (first slice) carries token_embd, the last
     * slice carries output*. A whole-model GGUF has them all -> model_find_tensor
     * binds every one exactly as before. */
    w->token_embd       = model_find_tensor(m, "token_embd.weight");
    w->output_hc_base   = model_find_tensor(m, "output_hc_base.weight");
    w->output_hc_fn     = model_find_tensor(m, "output_hc_fn.weight");
    w->output_hc_scale  = model_find_tensor(m, "output_hc_scale.weight");
    w->output_norm      = model_find_tensor(m, "output_norm.weight");
    w->output           = model_find_tensor(m, "output.weight");

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_layer_weights *l = &w->layer[il];
        const uint32_t compress_ratio = ds4_layer_compress_ratio(il);

        /* Sharded per-machine slice: a layer owned by the other machine has no
         * tensors in this GGUF -> leave w->layer[il] zeroed and skip. The graph
         * only evaluates this machine's --layers range, so the gap is never read.
         * A whole-model GGUF has every layer present -> nothing is skipped. */
        if (!tensor_by_namef(m, "blk.%u.attn_norm.weight", il)) continue;

        l->hc_attn_fn      = required_tensorf(m, "blk.%u.hc_attn_fn.weight", il);
        l->hc_attn_scale   = required_tensorf(m, "blk.%u.hc_attn_scale.weight", il);
        l->hc_attn_base    = required_tensorf(m, "blk.%u.hc_attn_base.weight", il);
        l->attn_norm       = required_tensorf(m, "blk.%u.attn_norm.weight", il);
        l->attn_q_a        = required_tensorf(m, "blk.%u.attn_q_a.weight", il);
        l->attn_q_a_norm   = required_tensorf(m, "blk.%u.attn_q_a_norm.weight", il);
        l->attn_q_b        = required_tensorf(m, "blk.%u.attn_q_b.weight", il);
        l->attn_kv         = required_tensorf(m, "blk.%u.attn_kv.weight", il);
        l->attn_kv_a_norm  = required_tensorf(m, "blk.%u.attn_kv_a_norm.weight", il);
        l->attn_sinks      = required_tensorf(m, "blk.%u.attn_sinks.weight", il);
        l->attn_output_a   = required_tensorf(m, "blk.%u.attn_output_a.weight", il);
        l->attn_output_b   = required_tensorf(m, "blk.%u.attn_output_b.weight", il);
        if (compress_ratio != 0) {
            l->attn_compressor_ape  = required_tensorf(m, "blk.%u.attn_compressor_ape.weight", il);
            l->attn_compressor_kv   = required_tensorf(m, "blk.%u.attn_compressor_kv.weight", il);
            l->attn_compressor_gate = required_tensorf(m, "blk.%u.attn_compressor_gate.weight", il);
            l->attn_compressor_norm = required_tensorf(m, "blk.%u.attn_compressor_norm.weight", il);
        }
        if (compress_ratio == 4) {
            l->indexer_attn_q_b = required_tensorf(m, "blk.%u.indexer.attn_q_b.weight", il);
            l->indexer_proj     = required_tensorf(m, "blk.%u.indexer.proj.weight", il);
            l->indexer_compressor_ape  = required_tensorf(m, "blk.%u.indexer_compressor_ape.weight", il);
            l->indexer_compressor_kv   = required_tensorf(m, "blk.%u.indexer_compressor_kv.weight", il);
            l->indexer_compressor_gate = required_tensorf(m, "blk.%u.indexer_compressor_gate.weight", il);
            l->indexer_compressor_norm = required_tensorf(m, "blk.%u.indexer_compressor_norm.weight", il);
        }
        l->hc_ffn_fn       = required_tensorf(m, "blk.%u.hc_ffn_fn.weight", il);
        l->hc_ffn_scale    = required_tensorf(m, "blk.%u.hc_ffn_scale.weight", il);
        l->hc_ffn_base     = required_tensorf(m, "blk.%u.hc_ffn_base.weight", il);
        l->ffn_norm        = required_tensorf(m, "blk.%u.ffn_norm.weight", il);
        l->ffn_gate_inp    = required_tensorf(m, "blk.%u.ffn_gate_inp.weight", il);
        l->ffn_exp_probs_b = tensor_by_namef(m, "blk.%u.exp_probs_b.bias", il);
        /* 合一 VQ 文件携带 blk.L.ffn_exps_vq.blob 时 gate/up 死重不入文件 → 降为可选。
         * down 同样可选(R28: 冷 w2 也编在 blob 的 which=2 槽里), 缺席走影子张量;
         * 冠军 vq4bf 那种"冷 w2 从 base go1b 读"的文件仍带着真 down, 语义不变。
         * 无 blob 的旧文件语义不变(三者仍必需)。 */
        if (tensor_by_namef(m, "blk.%u.ffn_exps_vq.blob", il)) {
            l->ffn_gate_exps = tensor_by_namef(m, "blk.%u.ffn_gate_exps.weight", il);
            l->ffn_up_exps   = tensor_by_namef(m, "blk.%u.ffn_up_exps.weight", il);
            l->ffn_down_exps = tensor_by_namef(m, "blk.%u.ffn_down_exps.weight", il);
            if (!l->ffn_down_exps) l->ffn_down_exps = routed_down_shadow(il);
        } else {
            l->ffn_gate_exps = required_tensorf(m, "blk.%u.ffn_gate_exps.weight", il);
            l->ffn_up_exps   = required_tensorf(m, "blk.%u.ffn_up_exps.weight", il);
            l->ffn_down_exps = required_tensorf(m, "blk.%u.ffn_down_exps.weight", il);
        }
        l->ffn_gate_shexp  = required_tensorf(m, "blk.%u.ffn_gate_shexp.weight", il);
        l->ffn_up_shexp    = required_tensorf(m, "blk.%u.ffn_up_shexp.weight", il);
        l->ffn_down_shexp  = required_tensorf(m, "blk.%u.ffn_down_shexp.weight", il);

        if (il < DS4_N_HASH_LAYER) {
            l->ffn_gate_tid2eid = required_tensorf(m, "blk.%u.ffn_gate_tid2eid.weight", il);
        }
    }

    weights_validate_layout(m, w);
}


static void model_map_span_include_tensor(
        const ds4_tensor *t,
        uint64_t *lo,
        uint64_t *hi,
        uint64_t *max_tensor_bytes) {
    if (!t || t->bytes == 0) return;
    const uint64_t end = t->abs_offset + t->bytes;
    if (*lo == UINT64_MAX || t->abs_offset < *lo) *lo = t->abs_offset;
    if (end > *hi) *hi = end;
    if (t->bytes > *max_tensor_bytes) *max_tensor_bytes = t->bytes;
}

static void model_map_span_vec_append(ds4_model_map_span_vec *spans, uint64_t lo, uint64_t hi) {
    if (!spans || lo == UINT64_MAX || hi <= lo) return;
    if (spans->len == spans->cap) {
        uint32_t new_cap = spans->cap ? spans->cap * 2u : 16u;
        spans->v = xrealloc(spans->v, (size_t)new_cap * sizeof(spans->v[0]));
        spans->cap = new_cap;
    }
    spans->v[spans->len++] = (ds4_model_map_span){lo, hi};
}

static void model_map_span_vec_include_one(ds4_model_map_span_vec *spans, const ds4_tensor *t) {
    if (!t) return;   /* 内嵌 VQ 合一文件: gate/up_exps 缺席即跳过(blob 走 CPU mmap 非 Metal span) */
    uint64_t lo = UINT64_MAX, hi = 0;
    model_map_span_include_tensor(t, &lo, &hi, &spans->max_tensor_bytes);
    model_map_span_vec_append(spans, lo, hi);
}

static void model_map_span_vec_include_layer(ds4_model_map_span_vec *spans, const ds4_layer_weights *l) {
#define DS4_INCLUDE_TENSOR(t_) model_map_span_vec_include_one(spans, (t_))
    DS4_INCLUDE_TENSOR(l->hc_attn_fn);
    DS4_INCLUDE_TENSOR(l->hc_attn_scale);
    DS4_INCLUDE_TENSOR(l->hc_attn_base);
    DS4_INCLUDE_TENSOR(l->attn_norm);
    DS4_INCLUDE_TENSOR(l->attn_q_a);
    DS4_INCLUDE_TENSOR(l->attn_q_a_norm);
    DS4_INCLUDE_TENSOR(l->attn_q_b);
    DS4_INCLUDE_TENSOR(l->attn_kv);
    DS4_INCLUDE_TENSOR(l->attn_kv_a_norm);
    DS4_INCLUDE_TENSOR(l->attn_sinks);
    DS4_INCLUDE_TENSOR(l->attn_output_a);
    DS4_INCLUDE_TENSOR(l->attn_output_b);
    DS4_INCLUDE_TENSOR(l->attn_compressor_ape);
    DS4_INCLUDE_TENSOR(l->attn_compressor_kv);
    DS4_INCLUDE_TENSOR(l->attn_compressor_gate);
    DS4_INCLUDE_TENSOR(l->attn_compressor_norm);
    DS4_INCLUDE_TENSOR(l->indexer_attn_q_b);
    DS4_INCLUDE_TENSOR(l->indexer_proj);
    DS4_INCLUDE_TENSOR(l->indexer_compressor_ape);
    DS4_INCLUDE_TENSOR(l->indexer_compressor_kv);
    DS4_INCLUDE_TENSOR(l->indexer_compressor_gate);
    DS4_INCLUDE_TENSOR(l->indexer_compressor_norm);
    DS4_INCLUDE_TENSOR(l->hc_ffn_fn);
    DS4_INCLUDE_TENSOR(l->hc_ffn_scale);
    DS4_INCLUDE_TENSOR(l->hc_ffn_base);
    DS4_INCLUDE_TENSOR(l->ffn_norm);
    DS4_INCLUDE_TENSOR(l->ffn_gate_tid2eid);
    DS4_INCLUDE_TENSOR(l->ffn_gate_inp);
    DS4_INCLUDE_TENSOR(l->ffn_exp_probs_b);
    DS4_INCLUDE_TENSOR(l->ffn_gate_exps);
    DS4_INCLUDE_TENSOR(l->ffn_up_exps);
    DS4_INCLUDE_TENSOR(l->ffn_down_exps);
    DS4_INCLUDE_TENSOR(l->ffn_gate_shexp);
    DS4_INCLUDE_TENSOR(l->ffn_up_shexp);
    DS4_INCLUDE_TENSOR(l->ffn_down_shexp);
#undef DS4_INCLUDE_TENSOR
}

static void model_map_span_vec_include_output(ds4_model_map_span_vec *spans, const ds4_weights *w) {
    model_map_span_vec_include_one(spans, w->output_hc_base);
    model_map_span_vec_include_one(spans, w->output_hc_fn);
    model_map_span_vec_include_one(spans, w->output_hc_scale);
    model_map_span_vec_include_one(spans, w->output_norm);
    model_map_span_vec_include_one(spans, w->output);
}

static int model_map_span_cmp(const void *a, const void *b) {
    const ds4_model_map_span *sa = a;
    const ds4_model_map_span *sb = b;
    if (sa->off < sb->off) return -1;
    if (sa->off > sb->off) return 1;
    if (sa->end < sb->end) return -1;
    if (sa->end > sb->end) return 1;
    return 0;
}

DS4_MAYBE_UNUSED bool weights_model_map_spans(
        const ds4_weights *w,
        uint32_t layer_start,
        uint32_t layer_end,
        bool include_output,
        bool include_token_embd,
        ds4_model_map_span_vec *spans) {
    if (!w || !spans) return false;
    if (layer_start >= DS4_N_LAYER) return false;
    if (layer_end == UINT32_MAX) layer_end = DS4_N_LAYER - 1u;
    if (layer_end >= DS4_N_LAYER || layer_end < layer_start) return false;

    memset(spans, 0, sizeof(*spans));
    /* Layer-0 workers always need token_embd to embed the prompt; a distributed
     * MTP drafter on a nonzero-start worker needs it too (docs/archive/mtp.md Phase 1). */
    if (layer_start == 0 || include_token_embd) {
        model_map_span_vec_include_one(spans, w->token_embd);
    }
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        model_map_span_vec_include_layer(spans, &w->layer[il]);
    }
    if (include_output) model_map_span_vec_include_output(spans, w);
    if (spans->len == 0 || spans->max_tensor_bytes == 0) return false;

    qsort(spans->v, spans->len, sizeof(spans->v[0]), model_map_span_cmp);
    uint32_t out = 0;
    for (uint32_t i = 0; i < spans->len; i++) {
        if (out == 0 || spans->v[i].off > spans->v[out - 1u].end) {
            spans->v[out++] = spans->v[i];
        } else if (spans->v[i].end > spans->v[out - 1u].end) {
            spans->v[out - 1u].end = spans->v[i].end;
        }
    }
    spans->len = out;
    return spans->len != 0;
}

/* Collect this layer's routed-expert weight tensors (ffn_gate/up/down_exps) into
 * `experts` and every other layer tensor into `backbone`. Mirrors
 * model_map_span_vec_include_layer but splits by tensor role so a reduced-memory
 * loader can keep the backbone resident and let cold experts stay reclaimable. */
static void model_map_span_vec_split_layer(
        ds4_model_map_span_vec *backbone,
        ds4_model_map_span_vec *experts,
        const ds4_layer_weights *l) {
#define DS4_BACKBONE(t_) model_map_span_vec_include_one(backbone, (t_))
#define DS4_EXPERT(t_)   model_map_span_vec_include_one(experts, (t_))
    DS4_BACKBONE(l->hc_attn_fn);
    DS4_BACKBONE(l->hc_attn_scale);
    DS4_BACKBONE(l->hc_attn_base);
    DS4_BACKBONE(l->attn_norm);
    DS4_BACKBONE(l->attn_q_a);
    DS4_BACKBONE(l->attn_q_a_norm);
    DS4_BACKBONE(l->attn_q_b);
    DS4_BACKBONE(l->attn_kv);
    DS4_BACKBONE(l->attn_kv_a_norm);
    DS4_BACKBONE(l->attn_sinks);
    DS4_BACKBONE(l->attn_output_a);
    DS4_BACKBONE(l->attn_output_b);
    DS4_BACKBONE(l->attn_compressor_ape);
    DS4_BACKBONE(l->attn_compressor_kv);
    DS4_BACKBONE(l->attn_compressor_gate);
    DS4_BACKBONE(l->attn_compressor_norm);
    DS4_BACKBONE(l->indexer_attn_q_b);
    DS4_BACKBONE(l->indexer_proj);
    DS4_BACKBONE(l->indexer_compressor_ape);
    DS4_BACKBONE(l->indexer_compressor_kv);
    DS4_BACKBONE(l->indexer_compressor_gate);
    DS4_BACKBONE(l->indexer_compressor_norm);
    DS4_BACKBONE(l->hc_ffn_fn);
    DS4_BACKBONE(l->hc_ffn_scale);
    DS4_BACKBONE(l->hc_ffn_base);
    DS4_BACKBONE(l->ffn_norm);
    DS4_BACKBONE(l->ffn_gate_tid2eid);
    DS4_BACKBONE(l->ffn_gate_inp);
    DS4_BACKBONE(l->ffn_exp_probs_b);
    /* Routed experts: the bulk of the model, only top-K fire per token. */
    DS4_EXPERT(l->ffn_gate_exps);
    DS4_EXPERT(l->ffn_up_exps);
    DS4_EXPERT(l->ffn_down_exps);
    /* Shared expert fires every token: it is backbone, not routed. */
    DS4_BACKBONE(l->ffn_gate_shexp);
    DS4_BACKBONE(l->ffn_up_shexp);
    DS4_BACKBONE(l->ffn_down_shexp);
#undef DS4_BACKBONE
#undef DS4_EXPERT
}

static void model_map_span_vec_finalize(ds4_model_map_span_vec *spans) {
    if (spans->len == 0) return;
    qsort(spans->v, spans->len, sizeof(spans->v[0]), model_map_span_cmp);
    uint32_t out = 0;
    for (uint32_t i = 0; i < spans->len; i++) {
        if (out == 0 || spans->v[i].off > spans->v[out - 1u].end) {
            spans->v[out++] = spans->v[i];
        } else if (spans->v[i].end > spans->v[out - 1u].end) {
            spans->v[out - 1u].end = spans->v[i].end;
        }
    }
    spans->len = out;
}

static void model_map_span_vec_sort_dedupe_exact(ds4_model_map_span_vec *spans) {
    if (spans->len == 0) return;
    qsort(spans->v, spans->len, sizeof(spans->v[0]), model_map_span_cmp);
    uint32_t out = 0;
    for (uint32_t i = 0; i < spans->len; i++) {
        if (out == 0 ||
            spans->v[i].off != spans->v[out - 1u].off ||
            spans->v[i].end != spans->v[out - 1u].end) {
            spans->v[out++] = spans->v[i];
        }
    }
    spans->len = out;
}

/* Build the backbone (resident) and routed-expert (reclaimable) span lists for a
 * reduced-memory Metal load. Returns false if either list is empty or sizes look
 * wrong; the caller then falls back to the whole-tensor-data range loader. */
bool weights_model_map_spans_split(
        const ds4_weights *w,
        ds4_model_map_span_vec *backbone,
        ds4_model_map_span_vec *experts) {
    if (!w || !backbone || !experts) return false;
    memset(backbone, 0, sizeof(*backbone));
    memset(experts, 0, sizeof(*experts));

    model_map_span_vec_include_one(backbone, w->token_embd);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        model_map_span_vec_split_layer(backbone, experts, &w->layer[il]);
    }
    model_map_span_vec_include_output(backbone, w);

    model_map_span_vec_finalize(backbone);
    model_map_span_vec_sort_dedupe_exact(experts);

    /* experts 为空在 VQ blob 下是正确结果(专家字节在 blob 里), 不是切分失败 —— 若当成
     * 失败返回, 调用方会退回整条 tensor-data 全驻留包装(35 GiB), 正好是要避免的事。 */
    if (backbone->len == 0) return false;
    if (experts->len == 0 && !g_vq_experts_blob) return false;
    if (backbone->max_tensor_bytes == 0) return false;
    return true;
}

bool weights_model_map_spans_split_slice(
        const ds4_weights *w,
        uint32_t layer_start,
        uint32_t layer_end,
        bool include_output,
        bool include_token_embd,
        ds4_model_map_span_vec *backbone,
        ds4_model_map_span_vec *experts) {
    if (!w || !backbone || !experts) return false;
    if (layer_start >= DS4_N_LAYER) return false;
    if (layer_end == UINT32_MAX) layer_end = DS4_N_LAYER - 1u;
    if (layer_end >= DS4_N_LAYER || layer_end < layer_start) return false;

    memset(backbone, 0, sizeof(*backbone));
    memset(experts, 0, sizeof(*experts));
    if (layer_start == 0 || include_token_embd) {
        model_map_span_vec_include_one(backbone, w->token_embd);
    }
    for (uint32_t il = layer_start; il <= layer_end; il++) {
        model_map_span_vec_split_layer(backbone, experts, &w->layer[il]);
    }
    if (include_output) model_map_span_vec_include_output(backbone, w);

    model_map_span_vec_finalize(backbone);
    model_map_span_vec_sort_dedupe_exact(experts);

    /* experts 为空在 VQ blob 下是正确结果(专家字节在 blob 里), 不是切分失败 —— 若当成
     * 失败返回, 调用方会退回整条 tensor-data 全驻留包装(35 GiB), 正好是要避免的事。 */
    if (backbone->len == 0) return false;
    if (experts->len == 0 && !g_vq_experts_blob) return false;
    if (backbone->max_tensor_bytes == 0) return false;
    return true;
}

void weights_free(ds4_weights *w) {
    memset(w, 0, sizeof(*w));
}

/* Load one token embedding row and expand it to float activations. */
/* 全q2 影子模型宿主行解码: 装载时 type 已翻成 F16 供图分发, 但 bytes 保持文件真值,
 * mmap 里躺的仍是 q2_K 块 —— 宿主读行必须按真实编码解码(与 CUDA 侧
 * host_deq_q2k_block 同式, 已对拍验证)。 */
