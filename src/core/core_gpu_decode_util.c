/* core_gpu_decode_util.c — decode 杂项/attn comp 视图 (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU
bool metal_graph_capture_prefix1_attn_state(ds4_gpu_graph *g, uint32_t il) {
    if (!g->spec_capture_prefix1 || !g->spec_prefix1_attn_state_kv[il]) return true;
    const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_attn_state_kv[il]);
    g->spec_prefix1_n_comp[il] = g->layer_n_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_kv[il], 0,
                                 g->layer_attn_state_kv[il], 0, bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix1_attn_state_score[il], 0,
                                 g->layer_attn_state_score[il], 0, bytes) != 0;
}

bool metal_graph_capture_prefix1_index_state(ds4_gpu_graph *g, uint32_t il) {
    if (!g->spec_capture_prefix1 || !g->spec_prefix1_index_state_kv[il]) return true;
    const uint64_t bytes = ds4_gpu_tensor_bytes(g->layer_index_state_kv[il]);
    g->spec_prefix1_n_index_comp[il] = g->layer_n_index_comp[il];
    return ds4_gpu_tensor_copy(g->spec_prefix1_index_state_kv[il], 0,
                                 g->layer_index_state_kv[il], 0, bytes) != 0 &&
           ds4_gpu_tensor_copy(g->spec_prefix1_index_state_score[il], 0,
                                 g->layer_index_state_score[il], 0, bytes) != 0;
}

/* Keep dense attention longer than the legacy 512-row window.
 * Around the 2K frontier the sparse path's score/top-k setup dominates
 * the smaller attention scan, while larger contexts benefit from sparse
 * indexed attention.  This threshold changes only the implementation used
 * to consume the compressed rows; it must not lower the 512-row indexer
 * selection defined by DS4_N_INDEXER_TOP_K. */
#define DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD_ROWS 1024u

uint32_t metal_graph_decode_indexer_sparse_threshold(const ds4_gpu_graph *g) {
    (void)g;
    return DS4_METAL_DECODE_INDEXER_SPARSE_THRESHOLD_ROWS;
}

/* =========================================================================
 * Metal Decode Release Helpers.
 * ========================================================================= */

bool metal_graph_decode_hc_pre(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const ds4_model        *model,
        uint64_t                scale_offset,
        uint64_t                base_offset) {
    return ds4_gpu_hc_split_weighted_sum_tensor(out,
                                                  split,
                                                  mix,
                                                  residual_hc,
                                                  model->map,
                                                  model->size,
                                                  scale_offset,
                                                  base_offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_HC,
                                                  DS4_N_HC_SINKHORN_ITER,
                                                  DS4_HC_EPS) != 0;
}

bool metal_graph_decode_kv_store(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          raw_row) {
    return ds4_gpu_kv_fp8_store_raw_tensor(kv,
                                             raw_cache,
                                             raw_cap,
                                             raw_row,
                                             DS4_N_HEAD_DIM,
                                             DS4_N_ROT) != 0;
}

/* ---- 压缩缓存(两后端唯一行格式 [448 f16][64 f32], 见 ds4_gpu_core.h; 09-07 拔掉 Apple/CUDA 开关): 写入方(压缩器
 * prefill/replay/update)照旧写 f32, 目标是 g->attn_comp_stage 行 0 起, FP8 KV 量化也在暂存上做, 然后由这里提交:
 * 后端把 f32 行转成缓存行格式抄进 layer_attn_comp_cache 的 first_row 行起。 */
bool metal_graph_commit_attn_comp_stage(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    if (!g || il >= DS4_N_LAYER) return false;
    if (rows == 0) return true;
    if (!g->layer_attn_comp_cache[il] || !g->attn_comp_stage) return false;
    if (rows > g->attn_comp_stage_cap || first_row > g->layer_comp_cap[il] ||
        rows > g->layer_comp_cap[il] - first_row) {
        return false;
    }
    return ds4_gpu_comp_rows_commit(g->layer_attn_comp_cache[il], first_row, g->attn_comp_stage, rows) != 0;
}

/* ---- indexer 缓存(2026-09-06, 两后端恒 f16): 写入方(压缩器 prefill/replay/update)照旧写 f32, 但目标是
 * g->attn_comp_stage 的行 0 起(其 attn_comp_stage_cap 行 × 512 宽足够放同样行数的 128 宽 indexer 行),
 * QAT 在暂存上做, 然后由这里提交: f32→f16 抄进 layer_index_comp_cache 的 first_row 行起。 */
bool metal_graph_commit_index_comp_stage(ds4_gpu_graph *g, uint32_t il, uint32_t first_row, uint32_t rows) {
    if (!g || il >= DS4_N_LAYER) return false;
    if (rows == 0) return true;
    if (!g->layer_index_comp_cache[il] || !g->attn_comp_stage) return false;
    /* 暂存按 attn_comp_stage_cap 行 × DS4_N_HEAD_DIM 宽分配, 128 宽的 indexer 行能放 4 倍行数 */
    if (first_row > g->layer_comp_cap[il] || rows > g->layer_comp_cap[il] - first_row ||
        (uint64_t)rows * DS4_N_INDEXER_HEAD_DIM > (uint64_t)g->attn_comp_stage_cap * DS4_N_HEAD_DIM) {
        return false;
    }
    return ds4_gpu_tensor_copy_f32_to_f16(g->layer_index_comp_cache[il],
                                           (uint64_t)first_row * DS4_N_INDEXER_HEAD_DIM * sizeof(uint16_t),
                                           g->attn_comp_stage, 0,
                                           (uint64_t)rows * DS4_N_INDEXER_HEAD_DIM) != 0;
}

/* Encode one DS4 decode layer on Metal.  This is the release single-token
 * layer path; diagnostics reuse it so they compare exactly what generation
 * runs. */

#endif /* !DS4_NO_GPU */
typedef int ds4_core_gpu_decode_util_nonempty_tu; /* 空TU防御(CPU构建) */
