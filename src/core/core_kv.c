/* core_kv.c — KV cache/压缩器状态 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"

/* =========================================================================
 * KV Cache, Compressors, and CPU Layer Execution.
 * =========================================================================
 *
 * The CPU path is the correctness reference.  It maintains raw SWA KV rows,
 * optional compressed KV rows, the indexer mask for ratio-4 layers, and a
 * reusable decode scratch arena so token generation does not allocate in the
 * hot loop.
 */

uint32_t ds4_default_raw_cap(uint32_t ctx_size) {
    uint32_t raw_cap = DS4_N_SWA;
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;
    return raw_cap;
}

/* CUDA 侧默认分块(2026-08-21): 原 dense/grouped 批 kernel 每个 warp 为自己那一行把整批
 * token 的激活重读一遍(340 token 的块 ≈ 12.9GB 流量, 权重才 11MB), 于是"一次灌完"是最慢
 * 的配置(42 t/s), 小块反而快。加了 token 分片 GEMM(激活按 tile 进 shared, 块内共用)之后
 * 大块重新变优: 实测 3800 token prompt chunk=128 119 / 256 **123** / 512 116 t/s。
 * 分片 kernel 与原路逐位一致(NLL 与 max|Δlogit| 均为 0 差)。Metal 侧保持原行为。 */
/* 默认 prefill 分块上限: 影响批 prefill 的切分边界, 而批路路由本就不确定 —— 改它
 * 会改捕获轨迹的可复现性(见 memory 捕获铁律), 不是单纯的性能旋钮。 */
#define DS4_PREFILL_CHUNK_DEFAULT 4096u
uint32_t ds4_default_prefill_cap_for_prompt(int prompt_len) {
    if (prompt_len <= 0) return 1;
    uint32_t cap = (uint32_t)prompt_len;

    const int req = ds4_tool_prefill_chunk();   /* --prefill-chunk (0=整段一批) */
    if (req >= 0) {
        if (req == 0) return cap;
        cap = (uint32_t)req;
    } else if (g_prefill_chunk_cuda > 0) {
        cap = (uint32_t)g_prefill_chunk_cuda;
    } else if (prompt_len > DS4_PREFILL_CHUNK_DEFAULT) {
        cap = DS4_PREFILL_CHUNK_DEFAULT;
    }

    if (cap == 0) cap = 1;
    if (cap > (uint32_t)prompt_len) cap = (uint32_t)prompt_len;
    return cap;
}

/* Allocate all CPU decode temporaries once.  This keeps generation deterministic
 * from the VM's point of view and makes accidental hot-loop malloc visible. */
void cpu_decode_scratch_init(ds4_cpu_decode_scratch *scratch, uint32_t ctx_size) {
    memset(scratch, 0, sizeof(*scratch));
    if (ctx_size == 0) ctx_size = 1;
    const uint32_t raw_cap = ds4_default_raw_cap(ctx_size);
    const uint32_t comp_cap = ds4_comp_cap_for(ctx_size, 4);   /* 4 = 全仓最小压缩比(上界口径) */
    const uint32_t attn_score_cap = raw_cap + comp_cap;
    const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
    const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t q8_cap = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
    const uint64_t q8_blocks = (q8_cap + 31u) / 32u;

    /*
     * The CPU decode path used to malloc/free dozens of medium-sized buffers
     * for every layer of every generated token. On macOS this can drive the VM
     * system through repeated map/unmap bookkeeping while the huge model mmap is
     * also being streamed, and we have observed kernel panics in VM accounting.
     * Keep decode scratch resident for the whole generation instead.
     */
    scratch->ctx_size = ctx_size;
    scratch->comp_cap = comp_cap;
    scratch->attn_score_cap = attn_score_cap;
    scratch->q8_cap = (uint32_t)q8_cap;

    scratch->plain = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->cur = xmalloc((size_t)hc_dim * sizeof(float));
    scratch->next = xmalloc((size_t)hc_dim * sizeof(float));

    scratch->attn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->attn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->attn_residual = xmalloc((size_t)hc_dim * sizeof(float));
    scratch->q = xmalloc((size_t)q_dim * sizeof(float));
    scratch->qr = xmalloc((size_t)DS4_N_LORA_Q * sizeof(float));
    scratch->qr_norm = xmalloc((size_t)DS4_N_LORA_Q * sizeof(float));
    scratch->kv_raw = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    scratch->kv = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    scratch->heads = xmalloc((size_t)q_dim * sizeof(float));
    scratch->attn_low = xmalloc((size_t)DS4_N_OUT_GROUP * DS4_N_LORA_O * sizeof(float));
    scratch->attn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->after_attn_hc = xmalloc((size_t)hc_dim * sizeof(float));
    scratch->attn_score = xmalloc((size_t)attn_score_cap * sizeof(float));

    scratch->comp = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));
    scratch->index_comp = xmalloc((size_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float));
    scratch->comp_kv_cur = xmalloc((size_t)2u * DS4_N_HEAD_DIM * sizeof(float));
    scratch->comp_sc_cur = xmalloc((size_t)2u * DS4_N_HEAD_DIM * sizeof(float));
    scratch->comp_pooled = xmalloc((size_t)DS4_N_HEAD_DIM * sizeof(float));

    scratch->index_allowed = xmalloc((size_t)comp_cap * sizeof(bool));
    scratch->index_q = xmalloc((size_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM * sizeof(float));
    scratch->index_weights = xmalloc((size_t)DS4_N_INDEXER_HEAD * sizeof(float));
    scratch->index_scores = xmalloc((size_t)comp_cap * sizeof(float));

    scratch->ffn_cur = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->ffn_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->ffn_moe = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->ffn_shared = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->ffn_out = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->shared_gate = xmalloc((size_t)DS4_N_FF_EXP * sizeof(float));
    scratch->shared_up = xmalloc((size_t)DS4_N_FF_EXP * sizeof(float));
    scratch->shared_mid = xmalloc((size_t)DS4_N_FF_EXP * sizeof(float));
    scratch->routed_mid_all = xmalloc((size_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(float));
    scratch->routed_xq = xmalloc((size_t)(DS4_N_EMBD / QK_K) * sizeof(block_q8_K));
    scratch->routed_midq = xmalloc((size_t)DS4_N_EXPERT_USED * (DS4_N_FF_EXP / QK_K) * sizeof(block_q8_K));

    scratch->q8_xq = xmalloc((size_t)q8_blocks * 32u);
    scratch->q8_xscale = xmalloc((size_t)q8_blocks * sizeof(float));

    scratch->hc_flat = xmalloc((size_t)hc_dim * sizeof(float));
    scratch->output_flat = xmalloc((size_t)hc_dim * sizeof(float));
    scratch->output_pre = xmalloc((size_t)DS4_N_HC * sizeof(float));
    scratch->output_weights = xmalloc((size_t)DS4_N_HC * sizeof(float));
    scratch->output_embd = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
    scratch->output_norm = xmalloc((size_t)DS4_N_EMBD * sizeof(float));
}

void cpu_decode_scratch_free(ds4_cpu_decode_scratch *scratch) {
    if (!scratch) return;
    free(scratch->output_norm);
    free(scratch->output_embd);
    free(scratch->output_weights);
    free(scratch->output_pre);
    free(scratch->output_flat);
    free(scratch->hc_flat);
    free(scratch->q8_xscale);
    free(scratch->q8_xq);
    free(scratch->routed_midq);
    free(scratch->routed_xq);
    free(scratch->routed_mid_all);
    free(scratch->shared_mid);
    free(scratch->shared_up);
    free(scratch->shared_gate);
    free(scratch->ffn_out);
    free(scratch->ffn_shared);
    free(scratch->ffn_moe);
    free(scratch->ffn_norm);
    free(scratch->ffn_cur);
    free(scratch->index_scores);
    free(scratch->index_weights);
    free(scratch->index_q);
    free(scratch->index_allowed);
    free(scratch->comp_pooled);
    free(scratch->comp_sc_cur);
    free(scratch->comp_kv_cur);
    free(scratch->index_comp);
    free(scratch->comp);
    free(scratch->attn_score);
    free(scratch->after_attn_hc);
    free(scratch->attn_out);
    free(scratch->attn_low);
    free(scratch->heads);
    free(scratch->kv);
    free(scratch->kv_raw);
    free(scratch->qr_norm);
    free(scratch->qr);
    free(scratch->q);
    free(scratch->attn_residual);
    free(scratch->attn_norm);
    free(scratch->attn_cur);
    free(scratch->next);
    free(scratch->cur);
    free(scratch->plain);
    memset(scratch, 0, sizeof(*scratch));
}

/* Allocate per-layer KV state: a raw sliding window for all layers, plus
 * compressed attention/indexer caches for layers whose ratio is nonzero. */
void kv_cache_init(ds4_kv_cache *cache, uint32_t ctx_size, uint32_t raw_cap) {
    memset(cache, 0, sizeof(*cache));
    if (raw_cap == 0) raw_cap = ds4_default_raw_cap(ctx_size);
    if (raw_cap > ctx_size) raw_cap = ctx_size;
    if (raw_cap == 0) raw_cap = 1;

    cache->head_dim = DS4_N_HEAD_DIM;

    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        cache->layer[il].cap_raw = raw_cap;
        cache->layer[il].raw_kv = xmalloc_zeroed((size_t)raw_cap * DS4_N_HEAD_DIM, sizeof(float));
        cache->layer[il].compress_ratio = ratio;

        if (ratio != 0) {
            const uint32_t comp_cap = ds4_comp_cap_for(ctx_size, ratio);
            const uint32_t attn_width = ds4_comp_row_width(ratio, DS4_N_HEAD_DIM);
            const uint32_t attn_rows = ds4_comp_row_slots(ratio) * ratio;

            cache->layer[il].comp_cap = comp_cap;
            cache->layer[il].attn_comp_kv = xmalloc_zeroed((size_t)comp_cap * DS4_N_HEAD_DIM, sizeof(float));
            cache->layer[il].attn_state_kv = xmalloc_zeroed((size_t)attn_width * attn_rows, sizeof(float));
            cache->layer[il].attn_state_score = xmalloc((size_t)attn_width * attn_rows * sizeof(float));
            for (uint64_t i = 0; i < (uint64_t)attn_width * attn_rows; i++) {
                cache->layer[il].attn_state_score[i] = DS4_NEG_INF;
            }

            if (ratio == 4) {
                const uint32_t index_width = ds4_comp_row_width(ratio, DS4_N_INDEXER_HEAD_DIM);
                const uint32_t index_rows = ds4_comp_row_slots(ratio) * ratio;
                cache->layer[il].index_comp_kv = xmalloc_zeroed((size_t)comp_cap * DS4_N_INDEXER_HEAD_DIM, sizeof(float));
                cache->layer[il].index_state_kv = xmalloc_zeroed((size_t)index_width * index_rows, sizeof(float));
                cache->layer[il].index_state_score = xmalloc((size_t)index_width * index_rows * sizeof(float));
                for (uint64_t i = 0; i < (uint64_t)index_width * index_rows; i++) {
                    cache->layer[il].index_state_score[i] = DS4_NEG_INF;
                }
            }
        }
    }
}

void kv_cache_free(ds4_kv_cache *cache) {
    if (!cache) return;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        free(cache->layer[il].raw_kv);
        free(cache->layer[il].attn_comp_kv);
        free(cache->layer[il].attn_state_kv);
        free(cache->layer[il].attn_state_score);
        free(cache->layer[il].index_comp_kv);
        free(cache->layer[il].index_state_kv);
        free(cache->layer[il].index_state_score);
    }
    memset(cache, 0, sizeof(*cache));
}

/* Append to the raw SWA cache.  Once full, it slides by one row. */
void kv_cache_push_raw(ds4_layer_cache *cache, const float *kv) {
    if (cache->n_raw < cache->cap_raw) {
        float *dst = cache->raw_kv + (uint64_t)cache->n_raw * DS4_N_HEAD_DIM;
        for (uint32_t i = 0; i < DS4_N_HEAD_DIM; i++) dst[i] = f16_to_f32(f32_to_f16(kv[i]));
        cache->n_raw++;
        return;
    }

    memmove(cache->raw_kv,
            cache->raw_kv + DS4_N_HEAD_DIM,
            (size_t)(cache->cap_raw - 1) * DS4_N_HEAD_DIM * sizeof(cache->raw_kv[0]));
    float *dst = cache->raw_kv + (uint64_t)(cache->cap_raw - 1) * DS4_N_HEAD_DIM;
    for (uint32_t i = 0; i < DS4_N_HEAD_DIM; i++) dst[i] = f16_to_f32(f32_to_f16(kv[i]));
}

void kv_cache_push_comp(float *rows, uint32_t *n_rows, uint32_t cap_rows, uint32_t row_dim, const float *kv) {
    if (*n_rows >= cap_rows) ds4_die("compressed KV cache capacity exceeded");
    float *dst = rows + (uint64_t)(*n_rows) * row_dim;
    for (uint32_t i = 0; i < row_dim; i++) dst[i] = f16_to_f32(f32_to_f16(kv[i]));
    (*n_rows)++;
}

/* After prefill, clear unused compressor state rows so decode starts from the
 * same partial-window state the streaming path would have produced. */
static void compressor_finish_prefill_state_cpu(
        float    * state_kv,
        float    * state_score,
        uint32_t   head_dim,
        uint32_t   compress_ratio,
        uint32_t   n_tokens) {
    if (!state_kv || !state_score || head_dim == 0 || compress_ratio == 0) return;

    const uint32_t width = ds4_comp_row_width(compress_ratio, head_dim);
    const uint32_t rem = n_tokens % compress_ratio;
    const uint32_t clear_start = ds4_comp_stage_base(compress_ratio) + rem;
    const uint32_t clear_end = ds4_comp_stage_base(compress_ratio) + compress_ratio;

    for (uint32_t row = clear_start; row < clear_end; row++) {
        float *kv = state_kv + (uint64_t)row * width;
        float *score = state_score + (uint64_t)row * width;
        memset(kv, 0, (size_t)width * sizeof(kv[0]));
        for (uint32_t i = 0; i < width; i++) score[i] = DS4_NEG_INF;
    }
}

void kv_cache_finish_prefill_states(ds4_kv_cache *cache, uint32_t n_tokens) {
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        ds4_layer_cache *layer = &cache->layer[il];
        const uint32_t ratio = layer->compress_ratio;
        if (ratio == 0) continue;

        compressor_finish_prefill_state_cpu(layer->attn_state_kv,
                                            layer->attn_state_score,
                                            DS4_N_HEAD_DIM,
                                            ratio,
                                            n_tokens);
        if (ratio == 4) {
            compressor_finish_prefill_state_cpu(layer->index_state_kv,
                                                layer->index_state_score,
                                                DS4_N_INDEXER_HEAD_DIM,
                                                ratio,
                                                n_tokens);
        }
    }
}

/* Pool the current compression window with a softmax over per-dimension scores.
 * Ratio-4 layers keep two lanes: attention compression and indexer compression. */
void compressor_pool_decode_state(
        float    * out,
        float    * state_kv,
        float    * state_score,
        uint32_t   head_dim,
        uint32_t   compress_ratio) {
    const uint32_t width = ds4_comp_row_width(compress_ratio, head_dim);

    for (uint32_t j = 0; j < head_dim; j++) {
        float max_score = DS4_NEG_INF;

        if (compress_ratio == 4) {
            for (uint32_t r = 0; r < compress_ratio; r++) {
                const float sp = state_score[(uint64_t)r * width + j];
                const float sc = state_score[(uint64_t)(compress_ratio + r) * width + head_dim + j];
                if (sp > max_score) max_score = sp;
                if (sc > max_score) max_score = sc;
            }
        } else {
            for (uint32_t r = 0; r < compress_ratio; r++) {
                const float s = state_score[(uint64_t)r * width + j];
                if (s > max_score) max_score = s;
            }
        }

        if (max_score <= DS4_NEG_INF * 0.5f) {
            out[j] = 0.0f;
            continue;
        }

        float denom = 0.0f;
        float sum = 0.0f;
        if (compress_ratio == 4) {
            for (uint32_t r = 0; r < compress_ratio; r++) {
                const float wp = expf(state_score[(uint64_t)r * width + j] - max_score);
                const float wc = expf(state_score[(uint64_t)(compress_ratio + r) * width + head_dim + j] - max_score);
                denom += wp + wc;
                sum += wp * state_kv[(uint64_t)r * width + j];
                sum += wc * state_kv[(uint64_t)(compress_ratio + r) * width + head_dim + j];
            }
        } else {
            for (uint32_t r = 0; r < compress_ratio; r++) {
                const float w = expf(state_score[(uint64_t)r * width + j] - max_score);
                denom += w;
                sum += w * state_kv[(uint64_t)r * width + j];
            }
        }

        out[j] = denom > 0.0f ? sum / denom : 0.0f;
    }
}

/* Streaming compressor update for one token.  It projects kv/score rows,
 * updates the rolling state, and emits a compressed KV row on ratio boundaries. */
bool compressor_decode_one(
        float                   * out_comp,
        const ds4_model         * model,
        const ds4_tensor        * wkv,
        const ds4_tensor        * wgate,
        const ds4_tensor        * ape,
        const ds4_tensor        * norm,
        const float             * x,
        float                   * state_kv,
        float                   * state_score,
        uint32_t                  head_dim,
        uint32_t                  compress_ratio,
        uint32_t                  il,
        uint32_t                  pos) {
    const uint32_t coff = ds4_comp_row_slots(compress_ratio);
    const uint32_t width = coff * head_dim;
    const uint32_t pos_mod = pos % compress_ratio;
    const uint32_t row = compress_ratio == 4 ? compress_ratio + pos_mod : pos_mod;
    const bool should_compress = ((pos + 1) % compress_ratio) == 0;

    float *kv_cur = xmalloc((size_t)width * sizeof(kv_cur[0]));
    float *sc_cur = xmalloc((size_t)width * sizeof(sc_cur[0]));
    if (wkv->type == 8 &&
        wgate->type == 8 &&
        wkv->ndim == 2 &&
        wgate->ndim == 2 &&
        wkv->dim[0] == wgate->dim[0]) {
        const uint64_t in_dim = wkv->dim[0];
        const uint64_t blocks = (in_dim + 31) / 32;
        int8_t *xq = xmalloc((size_t)blocks * 32);
        float *xscale = xmalloc((size_t)blocks * sizeof(xscale[0]));

        quantize_q8_0_activation(x, xq, xscale, in_dim);
        matvec_q8_0_pair_prequant(kv_cur, sc_cur, model, wkv, wgate, xq, xscale);

        free(xscale);
        free(xq);
    } else {
        matvec_any(kv_cur, model, wkv, x);
        matvec_any(sc_cur, model, wgate, x);
    }

    for (uint32_t j = 0; j < width; j++) {
        sc_cur[j] += tensor_2d_value(model, ape, j, pos_mod);
    }

    memcpy(state_kv + (uint64_t)row * width, kv_cur, (size_t)width * sizeof(kv_cur[0]));
    memcpy(state_score + (uint64_t)row * width, sc_cur, (size_t)width * sizeof(sc_cur[0]));

    free(sc_cur);
    free(kv_cur);

    if (!should_compress) {
        return false;
    }

    float *pooled = xmalloc((size_t)head_dim * sizeof(pooled[0]));
    compressor_pool_decode_state(pooled, state_kv, state_score, head_dim, compress_ratio);

    double ss = 0.0;
    for (uint32_t i = 0; i < head_dim; i++) ss += (double)pooled[i] * pooled[i];
    const float rms = 1.0f / sqrtf((float)(ss / (double)head_dim) + DS4_RMS_EPS);
    for (uint32_t i = 0; i < head_dim; i++) {
        out_comp[i] = pooled[i] * rms * tensor_1d_value(model, norm, i);
    }

    const uint32_t comp_pos = pos + 1 - compress_ratio;
    rope_tail_layer_inplace(out_comp, 1, head_dim, DS4_N_ROT, comp_pos, il, false);
    if (head_dim == DS4_N_HEAD_DIM) {
        dsv4_fp8_kv_quantize_row_inplace_cpu(out_comp, head_dim, DS4_N_ROT);
    } else if (head_dim == DS4_N_INDEXER_HEAD_DIM) {
        dsv4_indexer_qat_row_inplace_cpu(out_comp, head_dim);
    }

    if (compress_ratio == 4) {
        for (uint32_t r = 0; r < compress_ratio; r++) {
            memcpy(state_kv + (uint64_t)r * width,
                   state_kv + (uint64_t)(compress_ratio + r) * width,
                   (size_t)width * sizeof(state_kv[0]));
            memcpy(state_score + (uint64_t)r * width,
                   state_score + (uint64_t)(compress_ratio + r) * width,
                   (size_t)width * sizeof(state_score[0]));
        }
        for (uint32_t r = 0; r < compress_ratio; r++) {
            memcpy(state_kv + (uint64_t)(compress_ratio + r) * width,
                   state_kv + (uint64_t)r * width,
                   (size_t)width * sizeof(state_kv[0]));
            memcpy(state_score + (uint64_t)(compress_ratio + r) * width,
                   state_score + (uint64_t)r * width,
                   (size_t)width * sizeof(state_score[0]));
        }
    }

    free(pooled);
    return true;
}

