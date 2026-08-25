/* core_validate.c — shape 选择/config 校验/dspark 绑定 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
static const ds4_shape DS4_SHAPE_FLASH = {
    .name = "DeepSeek V4 Flash",
    .variant = DS4_VARIANT_FLASH,
    .n_layer = 43,
    .n_embd = 4096,
    .n_vocab = 129280,
    .n_head = 64,
    .n_head_kv = 1,
    .n_head_dim = 512,
    .n_value_dim = 512,
    .n_rot = 64,
    .n_out_group = 8,
    .n_lora_q = 1024,
    .n_lora_o = 1024,
    .n_expert = 256,
    .n_expert_used = 6,
    .n_expert_shared = 1,
    .n_ff_exp = 2048,
    .n_hash_layer = 3,
    .n_swa = 128,
    .n_indexer_head = 64,
    .n_indexer_head_dim = 128,
    .n_indexer_top_k = 512,
    .n_hc = 4,
    .n_hc_sinkhorn_iter = 20,
    .rms_eps = DS4_DEFAULT_RMS_EPS,
    .hc_eps = DS4_DEFAULT_HC_EPS,
    .expert_weight_scale = 1.5f,
    .swiglu_clamp_exp = DS4_DEFAULT_SWIGLU_CLAMP_EXP,
    .rope_freq_base = DS4_DEFAULT_ROPE_FREQ_BASE,
    .rope_scale_factor = DS4_DEFAULT_ROPE_SCALE_FACTOR,
    .rope_yarn_beta_fast = DS4_DEFAULT_ROPE_YARN_BETA_FAST,
    .rope_yarn_beta_slow = DS4_DEFAULT_ROPE_YARN_BETA_SLOW,
    .compress_rope_freq_base = DS4_DEFAULT_COMPRESS_ROPE_FREQ_BASE,
    .rope_orig_ctx = DS4_DEFAULT_ROPE_ORIG_CTX,
};

static const ds4_shape DS4_SHAPE_PRO = {
    .name = "DeepSeek V4 Pro",
    .variant = DS4_VARIANT_PRO,
    .n_layer = 61,
    .n_embd = 7168,
    .n_vocab = 129280,
    .n_head = 128,
    .n_head_kv = 1,
    .n_head_dim = 512,
    .n_value_dim = 512,
    .n_rot = 64,
    .n_out_group = 16,
    .n_lora_q = 1536,
    .n_lora_o = 1024,
    .n_expert = 384,
    .n_expert_used = 6,
    .n_expert_shared = 1,
    .n_ff_exp = 3072,
    .n_hash_layer = 3,
    .n_swa = 128,
    .n_indexer_head = 64,
    .n_indexer_head_dim = 128,
    .n_indexer_top_k = 1024,
    .n_hc = 4,
    .n_hc_sinkhorn_iter = 20,
    .rms_eps = DS4_DEFAULT_RMS_EPS,
    .hc_eps = DS4_DEFAULT_HC_EPS,
    .expert_weight_scale = 2.5f,
    .swiglu_clamp_exp = DS4_DEFAULT_SWIGLU_CLAMP_EXP,
    .rope_freq_base = DS4_DEFAULT_ROPE_FREQ_BASE,
    .rope_scale_factor = DS4_DEFAULT_ROPE_SCALE_FACTOR,
    .rope_yarn_beta_fast = DS4_DEFAULT_ROPE_YARN_BETA_FAST,
    .rope_yarn_beta_slow = DS4_DEFAULT_ROPE_YARN_BETA_SLOW,
    .compress_rope_freq_base = DS4_DEFAULT_COMPRESS_ROPE_FREQ_BASE,
    .rope_orig_ctx = DS4_DEFAULT_ROPE_ORIG_CTX,
};

static bool ds4_shape_matches_metadata(
        const ds4_shape *s,
        uint32_t n_layer,
        uint32_t n_embd,
        uint32_t n_vocab,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_head_dim,
        uint32_t n_value_dim,
        uint32_t n_rot,
        uint32_t n_lora_q,
        uint32_t n_lora_o,
        uint32_t n_out_group,
        uint32_t n_expert,
        uint32_t n_expert_used,
        uint32_t n_ff_exp,
        uint32_t n_expert_shared,
        uint32_t n_hash_layer,
        uint32_t n_swa,
        uint32_t n_indexer_head,
        uint32_t n_indexer_head_dim,
        uint32_t n_indexer_top_k,
        uint32_t n_hc,
        uint32_t n_hc_sinkhorn_iter) {
    return s->n_layer == n_layer &&
           s->n_embd == n_embd &&
           s->n_vocab == n_vocab &&
           s->n_head == n_head &&
           s->n_head_kv == n_head_kv &&
           s->n_head_dim == n_head_dim &&
           s->n_value_dim == n_value_dim &&
           s->n_rot == n_rot &&
           s->n_lora_q == n_lora_q &&
           s->n_lora_o == n_lora_o &&
           s->n_out_group == n_out_group &&
           s->n_expert == n_expert &&
           s->n_expert_used == n_expert_used &&
           s->n_ff_exp == n_ff_exp &&
           s->n_expert_shared == n_expert_shared &&
           s->n_hash_layer == n_hash_layer &&
           s->n_swa == n_swa &&
           s->n_indexer_head == n_indexer_head &&
           s->n_indexer_head_dim == n_indexer_head_dim &&
           s->n_indexer_top_k == n_indexer_top_k &&
           s->n_hc == n_hc &&
           s->n_hc_sinkhorn_iter == n_hc_sinkhorn_iter;
}

static void ds4_select_shape_from_metadata(
        uint32_t n_layer,
        uint32_t n_embd,
        uint32_t n_vocab,
        uint32_t n_head,
        uint32_t n_head_kv,
        uint32_t n_head_dim,
        uint32_t n_value_dim,
        uint32_t n_rot,
        uint32_t n_lora_q,
        uint32_t n_lora_o,
        uint32_t n_out_group,
        uint32_t n_expert,
        uint32_t n_expert_used,
        uint32_t n_ff_exp,
        uint32_t n_expert_shared,
        uint32_t n_hash_layer,
        uint32_t n_swa,
        uint32_t n_indexer_head,
        uint32_t n_indexer_head_dim,
        uint32_t n_indexer_top_k,
        uint32_t n_hc,
        uint32_t n_hc_sinkhorn_iter) {
    if (ds4_shape_matches_metadata(&DS4_SHAPE_FLASH,
                                   n_layer, n_embd, n_vocab, n_head, n_head_kv,
                                   n_head_dim, n_value_dim, n_rot, n_lora_q,
                                   n_lora_o, n_out_group, n_expert,
                                   n_expert_used, n_ff_exp, n_expert_shared,
                                   n_hash_layer, n_swa, n_indexer_head,
                                   n_indexer_head_dim, n_indexer_top_k, n_hc,
                                   n_hc_sinkhorn_iter)) {
        g_ds4_shape = DS4_SHAPE_FLASH;
        return;
    }
    if (ds4_shape_matches_metadata(&DS4_SHAPE_PRO,
                                   n_layer, n_embd, n_vocab, n_head, n_head_kv,
                                   n_head_dim, n_value_dim, n_rot, n_lora_q,
                                   n_lora_o, n_out_group, n_expert,
                                   n_expert_used, n_ff_exp, n_expert_shared,
                                   n_hash_layer, n_swa, n_indexer_head,
                                   n_indexer_head_dim, n_indexer_top_k, n_hc,
                                   n_hc_sinkhorn_iter)) {
        g_ds4_shape = DS4_SHAPE_PRO;
        return;
    }

    fprintf(stderr,
            "ds4: unsupported DeepSeek4 shape: layers=%u embd=%u heads=%u "
            "q_lora=%u out_groups=%u experts=%u ff_exp=%u indexer_top_k=%u\n",
            n_layer,
            n_embd,
            n_head,
            n_lora_q,
            n_out_group,
            n_expert,
            n_ff_exp,
            n_indexer_top_k);
    exit(1);
}

static void validate_compress_ratio_metadata(const ds4_model *m) {
    const char *key = "deepseek4.attention.compress_ratios";
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != GGUF_VALUE_UINT32 && arr.type != GGUF_VALUE_INT32)) {
        fprintf(stderr, "ds4: required int32/uint32 array metadata key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len < DS4_N_LAYER) {
        ds4_die("deepseek4.attention.compress_ratios is shorter than the layer count");
    }

    memset(g_ds4_compress_ratios, 0, sizeof(g_ds4_compress_ratios));
    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        uint32_t got = 0;
        if (arr.type == GGUF_VALUE_UINT32) {
            if (!cursor_u32(&c, &got)) ds4_die(c.error);
        } else {
            int32_t v = 0;
            if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
            if (v < 0) ds4_die("metadata array contains a negative value");
            got = (uint32_t)v;
        }

        const uint32_t expected = ds4_expected_layer_compress_ratio(il);
        if (got != expected) {
            fprintf(stderr,
                    "ds4: unexpected DeepSeek4 compression ratio at layer %u for %s: got %u, expected %u\n",
                    il, DS4_MODEL_SHAPE_NAME, got, expected);
            exit(1);
        }
        g_ds4_compress_ratios[il] = got;
    }
}

static void config_expect_f32(const char *name, float got, float expected);

static void validate_swiglu_clamp_metadata(const ds4_model *m) {
    const char *key = "deepseek4.swiglu_clamp_exp";
    ds4_array_ref arr;
    if (!model_get_array(m, key, &arr) ||
        (arr.type != GGUF_VALUE_FLOAT32 && arr.type != GGUF_VALUE_FLOAT64)) {
        fprintf(stderr, "ds4: required float array metadata key is missing: %s\n", key);
        exit(1);
    }
    if (arr.len < DS4_N_LAYER) {
        ds4_die("deepseek4.swiglu_clamp_exp is shorter than the layer count");
    }

    ds4_cursor c = cursor_at(m, arr.data_pos);
    for (uint32_t i = 0; i < DS4_N_LAYER; i++) {
        float got = 0.0f;
        if (arr.type == GGUF_VALUE_FLOAT32) {
            if (!cursor_read(&c, &got, sizeof(got))) ds4_die(c.error);
        } else {
            double v = 0.0;
            if (!cursor_read(&c, &v, sizeof(v))) ds4_die(c.error);
            got = (float)v;
        }
        config_expect_f32("swiglu_clamp_exp", got, DS4_SWIGLU_CLAMP_EXP);
    }
}

static void config_expect_u32(const char *name, uint32_t got, uint32_t expected) {
    if (got == expected) return;
    fprintf(stderr, "ds4: expected %s=%u for %s, got %u\n",
            name, expected, DS4_MODEL_SHAPE_NAME, got);
    exit(1);
}

static void config_expect_f32(const char *name, float got, float expected) {
    const float scale = fabsf(expected) > 1.0f ? fabsf(expected) : 1.0f;
    if (fabsf(got - expected) <= scale * 1.0e-6f) return;
    fprintf(stderr, "ds4: expected %s=%.9g for %s, got %.9g\n",
            name, (double)expected, DS4_MODEL_SHAPE_NAME, (double)got);
    exit(1);
}

static void config_expect_bool(const char *name, bool got, bool expected) {
    if (got == expected) return;
    fprintf(stderr, "ds4: expected %s=%s for %s, got %s\n",
            name, expected ? "true" : "false", DS4_MODEL_SHAPE_NAME, got ? "true" : "false");
    exit(1);
}

static void config_validate_fixed_shape(uint32_t n_layer) {
    config_expect_u32("block_count",                  n_layer,                 DS4_N_LAYER);
}

/* Validate metadata values that affect semantics: attention shape, HC count,
 * expert routing, RoPE scaling, compression ratios, and SwiGLU clamp. */
void config_validate_model(const ds4_model *m) {
    const uint32_t n_layer = required_u32(m, "deepseek4.block_count");
    const uint32_t n_embd = required_u32(m, "deepseek4.embedding_length");
    const uint32_t n_vocab = required_u32(m, "deepseek4.vocab_size");
    const uint32_t n_head = required_u32(m, "deepseek4.attention.head_count");
    const uint32_t n_head_kv = required_u32(m, "deepseek4.attention.head_count_kv");
    const uint32_t n_head_dim = required_u32(m, "deepseek4.attention.key_length");
    const uint32_t n_value_dim = required_u32(m, "deepseek4.attention.value_length");
    const uint32_t n_rot = required_u32(m, "deepseek4.rope.dimension_count");
    const uint32_t n_lora_q = required_u32(m, "deepseek4.attention.q_lora_rank");
    const uint32_t n_lora_o = required_u32(m, "deepseek4.attention.output_lora_rank");
    const uint32_t n_out_group = required_u32(m, "deepseek4.attention.output_group_count");
    const uint32_t n_expert = required_u32(m, "deepseek4.expert_count");
    const uint32_t n_expert_used = required_u32(m, "deepseek4.expert_used_count");
    const uint32_t n_ff_exp = required_u32(m, "deepseek4.expert_feed_forward_length");
    const uint32_t n_expert_shared = required_u32(m, "deepseek4.expert_shared_count");
    const uint32_t n_hash_layer = required_u32(m, "deepseek4.hash_layer_count");
    uint32_t n_expert_groups = 0;
    uint32_t n_group_used = 0;
    model_get_u32(m, "deepseek4.expert_group_count", &n_expert_groups);
    model_get_u32(m, "deepseek4.expert_group_used_count", &n_group_used);
    const uint32_t n_swa = required_u32(m, "deepseek4.attention.sliding_window");
    const uint32_t n_indexer_head = required_u32(m, "deepseek4.attention.indexer.head_count");
    const uint32_t n_indexer_head_dim = required_u32(m, "deepseek4.attention.indexer.key_length");
    const uint32_t n_indexer_top_k = required_u32(m, "deepseek4.attention.indexer.top_k");
    const uint32_t n_hc = required_u32(m, "deepseek4.hyper_connection.count");
    const uint32_t n_hc_sinkhorn_iter = required_u32(m, "deepseek4.hyper_connection.sinkhorn_iterations");

    ds4_select_shape_from_metadata(n_layer,
                                   n_embd,
                                   n_vocab,
                                   n_head,
                                   n_head_kv,
                                   n_head_dim,
                                   n_value_dim,
                                   n_rot,
                                   n_lora_q,
                                   n_lora_o,
                                   n_out_group,
                                   n_expert,
                                   n_expert_used,
                                   n_ff_exp,
                                   n_expert_shared,
                                   n_hash_layer,
                                   n_swa,
                                   n_indexer_head,
                                   n_indexer_head_dim,
                                   n_indexer_top_k,
                                   n_hc,
                                   n_hc_sinkhorn_iter);

    config_expect_u32("embedding_length",            n_embd,         DS4_N_EMBD);
    config_expect_u32("vocab_size",                  n_vocab,        DS4_N_VOCAB);
    config_expect_u32("attention.head_count",        n_head,         DS4_N_HEAD);
    config_expect_u32("attention.key_length",        n_head_dim,     DS4_N_HEAD_DIM);
    config_expect_u32("attention.head_count_kv",     n_head_kv,      DS4_N_HEAD_KV);
    config_expect_u32("attention.value_length",      n_value_dim,    DS4_N_VALUE_DIM);
    config_expect_u32("rope.dimension_count",        n_rot,          DS4_N_ROT);
    config_expect_u32("attention.output_group_count", n_out_group,    DS4_N_OUT_GROUP);
    config_expect_u32("attention.q_lora_rank",       n_lora_q,        DS4_N_LORA_Q);
    config_expect_u32("attention.output_lora_rank",  n_lora_o,        DS4_N_LORA_O);
    config_expect_u32("expert_count",               n_expert,        DS4_N_EXPERT);
    config_expect_u32("expert_used_count",          n_expert_used,   DS4_N_EXPERT_USED);
    config_expect_u32("expert_feed_forward_length", n_ff_exp,        DS4_N_FF_EXP);
    config_expect_u32("expert_shared_count",         n_expert_shared, DS4_N_EXPERT_SHARED);
    config_expect_u32("hash_layer_count",            n_hash_layer,    DS4_N_HASH_LAYER);
    config_expect_u32("expert_group_count",         n_expert_groups, 0);
    config_expect_u32("expert_group_used_count",    n_group_used,    0);

    config_expect_u32("attention.sliding_window",     n_swa,                   DS4_N_SWA);
    config_expect_u32("attention.indexer.head_count", n_indexer_head,     DS4_N_INDEXER_HEAD);
    config_expect_u32("attention.indexer.key_length", n_indexer_head_dim, DS4_N_INDEXER_HEAD_DIM);
    config_expect_u32("attention.indexer.top_k",      n_indexer_top_k,    DS4_N_INDEXER_TOP_K);
    config_expect_u32("hyper_connection.count", n_hc, DS4_N_HC);
    config_expect_u32("hyper_connection.sinkhorn_iterations", n_hc_sinkhorn_iter, DS4_N_HC_SINKHORN_ITER);

    config_validate_fixed_shape(n_layer);
    validate_compress_ratio_metadata(m);

    validate_swiglu_clamp_metadata(m);

    uint64_t rope_orig_ctx = DS4_ROPE_ORIG_CTX;
    model_get_u64_compat(m, "deepseek4.rope.scaling.original_context_length", &rope_orig_ctx);
    if (rope_orig_ctx != DS4_ROPE_ORIG_CTX) {
        fprintf(stderr, "ds4: expected rope.scaling.original_context_length=%" PRIu64
                " for %s, got %" PRIu64 "\n",
                (uint64_t)DS4_ROPE_ORIG_CTX, DS4_MODEL_SHAPE_NAME, rope_orig_ctx);
        exit(1);
    }
    const float rope_freq_base = required_f32(m, "deepseek4.rope.freq_base");
    config_expect_f32("rope.freq_base", rope_freq_base, DS4_ROPE_FREQ_BASE);
    float rope_scale_factor = DS4_ROPE_SCALE_FACTOR;
    model_get_f32_compat(m, "deepseek4.rope.scaling.factor", &rope_scale_factor);
    config_expect_f32("rope.scaling.factor", rope_scale_factor, DS4_ROPE_SCALE_FACTOR);
    float rope_yarn_beta_fast = DS4_ROPE_YARN_BETA_FAST;
    model_get_f32_compat(m, "deepseek4.rope.scaling.yarn_beta_fast", &rope_yarn_beta_fast);
    config_expect_f32("rope.scaling.yarn_beta_fast", rope_yarn_beta_fast, DS4_ROPE_YARN_BETA_FAST);
    float rope_yarn_beta_slow = DS4_ROPE_YARN_BETA_SLOW;
    model_get_f32_compat(m, "deepseek4.rope.scaling.yarn_beta_slow", &rope_yarn_beta_slow);
    config_expect_f32("rope.scaling.yarn_beta_slow", rope_yarn_beta_slow, DS4_ROPE_YARN_BETA_SLOW);
    const float compress_rope_freq_base = required_f32(m, "deepseek4.attention.compress_rope_freq_base");
    config_expect_f32("attention.compress_rope_freq_base", compress_rope_freq_base, DS4_COMPRESS_ROPE_FREQ_BASE);
    const float expert_weight_scale = required_f32(m, "deepseek4.expert_weights_scale");
    config_expect_f32("expert_weights_scale", expert_weight_scale, DS4_EXPERT_WEIGHT_SCALE);
    const float rms_eps = required_f32(m, "deepseek4.attention.layer_norm_rms_epsilon");
    config_expect_f32("attention.layer_norm_rms_epsilon", rms_eps, DS4_RMS_EPS);
    const float hc_eps = required_f32(m, "deepseek4.hyper_connection.epsilon");
    config_expect_f32("hyper_connection.epsilon", hc_eps, DS4_HC_EPS);
    const bool expert_weight_norm = required_bool(m, "deepseek4.expert_weights_norm");
    config_expect_bool("expert_weights_norm", expert_weight_norm, true);
}

/* Bind tensor names once into the fixed DS4 layer layout.  This is the point
 * where stringly GGUF metadata becomes direct model-specific pointers. */
/* DSpark drafter 绑定(全 optional): 首块 attn_q_a 缺 ⇒ 模型没带 drafter, 静默关闭。
 * 层内字段与主层同构(indexer/compressor 字段留 NULL — drafter 层没有这两族)。 */
static ds4_tensor *dspark_tensorf(const ds4_model *m, const char *fmt, uint32_t blk) {
    return tensor_by_namef(m, fmt, blk);
}

int g_dspark_ready_global = 0;   /* engine open 后置; graph alloc 统一消费 */
const ds4_dspark_weights *g_dspark_bound_for_prefill = NULL;   /* prefill 建窗弱引用 */

static void dspark_weights_bind(ds4_dspark_weights *w, const ds4_model *m) {
    memset(w, 0, sizeof(*w));
    if (!model_find_tensor(m, "mtp.0.attn_q_a.weight")) return;   /* 没带 drafter */
    for (uint32_t b = 0; b < DS4_DSPARK_MAX_BLOCKS; b++) {
        if (!dspark_tensorf(m, "mtp.%u.attn_q_a.weight", b)) break;
        ds4_layer_weights *l = &w->block[b];
        l->hc_attn_fn      = dspark_tensorf(m, "mtp.%u.hc_attn_fn.weight", b);
        l->hc_attn_scale   = dspark_tensorf(m, "mtp.%u.hc_attn_scale.weight", b);
        l->hc_attn_base    = dspark_tensorf(m, "mtp.%u.hc_attn_base.weight", b);
        l->attn_norm       = dspark_tensorf(m, "mtp.%u.attn_norm.weight", b);
        l->attn_q_a        = dspark_tensorf(m, "mtp.%u.attn_q_a.weight", b);
        l->attn_q_a_norm   = dspark_tensorf(m, "mtp.%u.attn_q_a_norm.weight", b);
        l->attn_q_b        = dspark_tensorf(m, "mtp.%u.attn_q_b.weight", b);
        l->attn_kv         = dspark_tensorf(m, "mtp.%u.attn_kv.weight", b);
        l->attn_kv_a_norm  = dspark_tensorf(m, "mtp.%u.attn_kv_a_norm.weight", b);
        l->attn_sinks      = dspark_tensorf(m, "mtp.%u.attn_sinks.weight", b);
        l->attn_output_a   = dspark_tensorf(m, "mtp.%u.attn_output_a.weight", b);
        l->attn_output_b   = dspark_tensorf(m, "mtp.%u.attn_output_b.weight", b);
        l->hc_ffn_fn       = dspark_tensorf(m, "mtp.%u.hc_ffn_fn.weight", b);
        l->hc_ffn_scale    = dspark_tensorf(m, "mtp.%u.hc_ffn_scale.weight", b);
        l->hc_ffn_base     = dspark_tensorf(m, "mtp.%u.hc_ffn_base.weight", b);
        l->ffn_norm        = dspark_tensorf(m, "mtp.%u.ffn_norm.weight", b);
        l->ffn_gate_inp    = dspark_tensorf(m, "mtp.%u.ffn_gate_inp.weight", b);
        l->ffn_exp_probs_b = dspark_tensorf(m, "mtp.%u.exp_probs_b.bias", b);
        l->ffn_gate_exps   = dspark_tensorf(m, "mtp.%u.ffn_gate_exps.weight", b);
        l->ffn_up_exps     = dspark_tensorf(m, "mtp.%u.ffn_up_exps.weight", b);
        l->ffn_down_exps   = dspark_tensorf(m, "mtp.%u.ffn_down_exps.weight", b);
        l->ffn_gate_shexp  = dspark_tensorf(m, "mtp.%u.ffn_gate_shexp.weight", b);
        l->ffn_up_shexp    = dspark_tensorf(m, "mtp.%u.ffn_up_shexp.weight", b);
        l->ffn_down_shexp  = dspark_tensorf(m, "mtp.%u.ffn_down_shexp.weight", b);
        w->n_blocks = (int)b + 1;
    }
    /* 特殊头挂点分散(0731: main_proj 在 mtp.0, markov/confidence/norm/hc_head 在
     * mtp.2) —— 逐模块探测取首个存在者 */
    for (uint32_t b = 0; b < DS4_DSPARK_MAX_BLOCKS; b++) {
        if (!w->main_proj)       w->main_proj       = dspark_tensorf(m, "mtp.%u.main_proj.weight", b);
        if (!w->main_norm)       w->main_norm       = dspark_tensorf(m, "mtp.%u.main_norm.weight", b);
        if (!w->confidence_proj) w->confidence_proj = dspark_tensorf(m, "mtp.%u.confidence_proj.weight", b);
        if (!w->markov_w1)       w->markov_w1       = dspark_tensorf(m, "mtp.%u.markov_w1.weight", b);
        if (!w->markov_w2)       w->markov_w2       = dspark_tensorf(m, "mtp.%u.markov_w2.weight", b);
        if (!w->norm)            w->norm            = dspark_tensorf(m, "mtp.%u.norm.weight", b);
        if (!w->hc_head_base)    w->hc_head_base    = dspark_tensorf(m, "mtp.%u.hc_head_base.weight", b);
        if (!w->hc_head_fn)      w->hc_head_fn      = dspark_tensorf(m, "mtp.%u.hc_head_fn.weight", b);
        if (!w->hc_head_scale)   w->hc_head_scale   = dspark_tensorf(m, "mtp.%u.hc_head_scale.weight", b);
    }
    /* 最小可用集: 至少一个块 + 融合投影 + 出口 norm */
    w->ready = w->n_blocks > 0 && w->main_proj && w->norm;
    if (w->ready) { g_dspark_ready_global = 1; g_dspark_bound_for_prefill = w; }
    if (w->ready)
        fprintf(stderr, "ds4: DSpark drafter armed: %d block(s)%s%s\n",
                w->n_blocks,
                w->confidence_proj ? " +confidence" : "",
                w->markov_w1 ? " +markov" : "");
}

/* DS4_DRAFT_GGUF(2026-08-20): 独立 drafter gguf(官方开源 DSpark 量化版形态, 仅 mtp.*)
 * 挂在主模型旁。主模型自带 mtp.* 时优先; 副 model 懒加载单例, 进程生命周期持有。
 * CUDA (map,offset)→device 解析按 host_base 区分文件, 第二 mmap 天然可用。 */
ds4_model *g_draft_model = NULL;
void dspark_bind_with_draft(ds4_dspark_weights *w, const ds4_model *m, bool graph_backend) {
    dspark_weights_bind(w, m);
    if (w->ready) { w->src = m; w->head_src = m; return; }
    const char *p = getenv("DS4_DRAFT_GGUF");
    if (!p || !p[0]) return;
    if (!g_draft_model) {
        g_draft_model = xmalloc(sizeof(*g_draft_model));
        model_open(g_draft_model, p, graph_backend, true);   /* prefetch: 副 map 页表预热(nsys: 冷页长尾 30x) */

        fprintf(stderr, "ds4: draft gguf mapped: %s\n", p);
    }
    dspark_weights_bind(w, g_draft_model);
    w->src = g_draft_model;
    w->head_src = g_draft_model;
    /* 出口侧借主模型(官方: drafter 复用主 head)。norm 是 ready 硬条件, hc_head 是
     * step ④ 的硬解引用 —— 三者齐借, head_src 指向主模型供 step 选 map。 */
    if (!w->ready && w->n_blocks > 0 && w->main_proj) {
        if (!w->norm)          w->norm          = model_find_tensor(m, "output_norm.weight");
        if (!w->hc_head_base)  w->hc_head_base  = model_find_tensor(m, "output_hc_base.weight");
        if (!w->hc_head_fn)    w->hc_head_fn    = model_find_tensor(m, "output_hc_fn.weight");
        if (!w->hc_head_scale) w->hc_head_scale = model_find_tensor(m, "output_hc_scale.weight");
        if (w->norm && w->hc_head_base && w->hc_head_fn && w->hc_head_scale) {
            w->head_src = m;
            w->ready = true;
            g_dspark_ready_global = 1;
            g_dspark_bound_for_prefill = w;
            fprintf(stderr, "ds4: DSpark drafter armed from draft gguf: %d block(s), head borrowed from base\n",
                    w->n_blocks);
        }
    }
    if (!w->ready) fprintf(stderr, "ds4: draft gguf has no usable mtp.* tensors: %s\n", p);
}

