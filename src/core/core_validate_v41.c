/* core_validate_v41.c — V4.1 接线元数据装填(2026-09-12 战役 P1)。
 *
 * V4.1 与 V4 Flash 的结构差异不在维度而在"接线": 只有 kv_source_layers 压缩并持有压缩 KV,
 * 其余压缩层读最近一个源层的缓存; 只有 index_source_layers 跑 indexer, 其余层复用最近源层的
 * topk; candidate_source_layer 之后各层的 topk 限制在它筛出的候选块内; engram 只在两层。
 * 这些全部来自转换器写进 GGUF 的 deepseek4.* 键(与官方 config.json 一一对应), 这里装进
 * g_ds4_v41 供热路径按层查表。缺键 = 硬停, 不给默认值。 */
#include "core_internal.h"

static uint32_t v41_arr_i32(const ds4_model *m, const char *key, int32_t *out, uint32_t cap) {
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr) || (arr.type != GGUF_VALUE_INT32 && arr.type != GGUF_VALUE_UINT32)) {
        fprintf(stderr, "ds4: V4.1 required int32 array key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len > cap) ds4_die("V4.1 metadata array longer than capacity");
    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint64_t i = 0; i < arr.len; i++) if (!cursor_read(&c, &out[i], 4)) ds4_die(c.error);
    return (uint32_t)arr.len;
}

static uint32_t v41_arr_u64(const ds4_model *m, const char *key, uint64_t *out, uint32_t cap) {
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr) || arr.type != GGUF_VALUE_UINT64) {
        fprintf(stderr, "ds4: V4.1 required uint64 array key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len > cap) ds4_die("V4.1 metadata array longer than capacity");
    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint64_t i = 0; i < arr.len; i++) if (!cursor_u64(&c, &out[i])) ds4_die(c.error);
    return (uint32_t)arr.len;
}

static uint64_t v41_req_u64(const ds4_model *m, const char *key) {
    uint64_t v = 0;
    if (!model_get_u64_compat(m, key, &v)) { fprintf(stderr, "ds4: V4.1 required key is missing: %s\n", key); exit(1); }
    return v;
}

void v41_load_metadata(const ds4_model *m) {
    ds4_v41_cfg *v = &g_ds4_v41;
    memset(v, 0, sizeof *v);
    v->active = 1;
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        v->kv_source_of[il] = -1; v->index_source_of[il] = -1; v->engram_index_of[il] = -1;
    }
    int32_t ids[DS4_MAX_LAYER];
    uint32_t n = v41_arr_i32(m, "deepseek4.attention.kv_source_layers", ids, DS4_MAX_LAYER);
    for (uint32_t i = 0; i < n; i++) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= DS4_N_LAYER) ds4_die("kv_source_layers out of range");
        v->is_kv_source[ids[i]] = 1;
    }
    n = v41_arr_i32(m, "deepseek4.attention.index_source_layers", ids, DS4_MAX_LAYER);
    for (uint32_t i = 0; i < n; i++) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= DS4_N_LAYER) ds4_die("index_source_layers out of range");
        v->is_index_source[ids[i]] = 1;
    }
    /* 接线: 压缩层读"最近一个 ≤ 本层的源层"。官方 SharedAttentionRuntime 的语义就是
     * "源层写一槽, 后面的层读同一槽", 层按序执行, 所以等价于最近源层。 */
    int16_t last_kv = -1, last_idx = -1;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (v->is_kv_source[il]) last_kv = (int16_t)il;
        if (v->is_index_source[il]) last_idx = (int16_t)il;
        if (ds4_layer_compress_ratio(il) == 0) continue;
        if (last_kv < 0 || last_idx < 0) {
            fprintf(stderr, "ds4: V4.1 layer %u compresses but has no kv/index source before it\n", il);
            exit(1);
        }
        v->kv_source_of[il] = last_kv;
        v->index_source_of[il] = last_idx;
        if (v->is_kv_source[il] && !v->is_index_source[il])
            ds4_die("V4.1: a kv source layer must also be an index source (indexer keys come from its latent)");
    }
    v->candidate_source_layer = (int32_t)required_u32(m, "deepseek4.attention.candidate.source_layer");
    v->candidate_topk_blocks = (int32_t)required_u32(m, "deepseek4.attention.candidate.topk_blocks");
    v->candidate_block_size = (int32_t)required_u32(m, "deepseek4.attention.candidate.block_size");

    n = v41_arr_i32(m, "deepseek4.engram.layer_ids", ids, DS4_V41_MAX_ENGRAM);
    v->n_engram = n;
    uint64_t rows[DS4_V41_MAX_ENGRAM];
    if (v41_arr_u64(m, "deepseek4.engram.num_embeddings", rows, DS4_V41_MAX_ENGRAM) != n)
        ds4_die("engram.num_embeddings count != engram.layer_ids count");
    for (uint32_t i = 0; i < n; i++) {
        if (ids[i] < 0 || (uint32_t)ids[i] >= DS4_N_LAYER) ds4_die("engram layer id out of range");
        v->engram_layer[i] = ids[i];
        v->engram_index_of[ids[i]] = (int16_t)i;
        v->engram_rows[i] = rows[i];
        char key[96]; ds4_str s;
        snprintf(key, sizeof key, "deepseek4.engram.%u.table_path", i);
        if (!model_get_string(m, key, &s) || s.len == 0 || s.len >= sizeof v->engram_table_path[i]) {
            fprintf(stderr, "ds4: V4.1 required key is missing: %s\n", key); exit(1);
        }
        memcpy(v->engram_table_path[i], s.ptr, s.len); v->engram_table_path[i][s.len] = 0;
        snprintf(key, sizeof key, "deepseek4.engram.%u.weight_offset", i); v->engram_weight_off[i] = v41_req_u64(m, key);
        snprintf(key, sizeof key, "deepseek4.engram.%u.scale_offset", i);  v->engram_scale_off[i]  = v41_req_u64(m, key);
    }
    v->engram_max_ngram = required_u32(m, "deepseek4.engram.max_ngram_size");
    v->engram_heads     = required_u32(m, "deepseek4.engram.head_count");
    v->engram_head_dim  = required_u32(m, "deepseek4.engram.head_dim");
    v->engram_vocab     = required_u32(m, "deepseek4.engram.vocab_size");
    v->engram_cvocab    = required_u32(m, "deepseek4.engram.compressed_vocab_size");
    v->engram_pad       = required_u32(m, "deepseek4.engram.pad_id_compressed");
    v->swiglu_limit     = DS4_SWIGLU_CLAMP_EXP;

    int nk = 0, ni = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) { nk += v->is_kv_source[il]; ni += v->is_index_source[il]; }
    fprintf(stderr, "ds4: V4.1 接线: kv 源 %d 层 / indexer 源 %d 层 / 候选源 L%d(%d 块×%d) / engram %u 层 (表 %llu+%llu 行, 在盘)\n",
            nk, ni, v->candidate_source_layer, v->candidate_topk_blocks, v->candidate_block_size, v->n_engram,
            (unsigned long long)(n > 0 ? v->engram_rows[0] : 0), (unsigned long long)(n > 1 ? v->engram_rows[1] : 0));
}
