/* core_globals.c — 全局形状/压缩比与后端判定 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"

#ifdef DS4_NO_GPU
/* ds4_distributed.o is compiled once (no DS4_NO_GPU variant) and calls this
 * GPU-side helper unconditionally; the CPU build supplies the no-op here. */
void ds4_gpu_expert_remote_fetch_kick(void) {}
#endif

/* ★V4.1 官方 encoding.py 的 reasoning effort 前缀★(REASONING_EFFORT_TEMPLATE; 数字档 low 50 / high 75 / max 100, 默认 high)。
 * 位置: BOS → <｜System｜> → 这一行 → system 正文 → <｜User｜>…, 只在 thinking 模式出现, 每条对话一次。
 * 2026-09-21 之前这里是 V4 时代的一段英文长段落, 且只在 max 档写、high 档什么都不写、前面也没有 <｜System｜> ——
 * 三处都与 V4.1 官方不同(bug.md §1.4)。模板字符串只在这一处, 服务端/CLI/agent 都引它。 */
const char DS4_REASONING_EFFORT_HIGH_PREFIX[] =
    "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
const char DS4_REASONING_EFFORT_MAX_PREFIX[] =
    "Reasoning Effort: 100 (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
bool g_ds4_chat_system_token = false;

bool ds4_backend_uses_graph(ds4_backend backend) {
    return backend == DS4_BACKEND_METAL || backend == DS4_BACKEND_CUDA;
}

/* =========================================================================
 * DeepSeek V4 Shape Profiles.
 * =========================================================================
 *
 * The weight binder and metadata validator select one of the known model
 * profiles below.  Arrays reserve the maximum Pro dimensions; hot loops read
 * the active profile after GGUF validation.
 */


ds4_shape g_ds4_shape = {
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

uint32_t g_ds4_compress_ratios[DS4_MAX_LAYER] = {0};
ds4_v41_cfg g_ds4_v41;   /* V4.1 接线表; V4 模型下 active=0, 由 core_validate_v41.c 装填 */
/* Attention compression is read from GGUF metadata after validating that it
 * matches the exact layout expected for the loaded model shape. */
uint32_t ds4_layer_compress_ratio(uint32_t il) {
    if (il >= DS4_N_LAYER) ds4_die("DeepSeek4 layer index is outside the loaded model layout");
    return g_ds4_compress_ratios[il];
}

uint32_t ds4_expected_layer_compress_ratio(uint32_t il) {
    if (il >= DS4_N_LAYER) ds4_die("DeepSeek4 layer index is outside the loaded model layout");

    switch (DS4_MODEL_VARIANT) {
    case DS4_VARIANT_FLASH:
        if (il < 2) return 0;
        return (il & 1u) == 0 ? 4u : 128u;
    case DS4_VARIANT_PRO:
        if (il < 2) return 128u;
        return (il & 1u) == 0 ? 4u : 128u;
    case DS4_VARIANT_V41:
        /* V4.1 的压缩比是元数据的真值(0/1/2 按官方 config 逐层给), 没有"预期公式";
         * 校验在 core_validate_v41.c 只做取值范围检查。 */
        return g_ds4_compress_ratios[il];
    default:
        ds4_die("unsupported DeepSeek4 model variant");
    }
    return 0;
}
