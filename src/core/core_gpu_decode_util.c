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

static uint64_t metal_graph_attn_comp_cache_row_bytes(void) {
    return (uint64_t)DS4_N_HEAD_DIM *
           (DS4_GPU_ATTN_COMP_CACHE_F16 ? sizeof(uint16_t) : sizeof(float));
}

uint32_t metal_graph_attn_comp_cache_is_f16(void) {
    return DS4_GPU_ATTN_COMP_CACHE_F16 ? 1u : 0u;
}

static bool metal_graph_store_attn_comp_stage(
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

    const uint64_t count = (uint64_t)rows * DS4_N_HEAD_DIM;
    const uint64_t dst_offset = (uint64_t)first_row *
                                metal_graph_attn_comp_cache_row_bytes();
    if (DS4_GPU_ATTN_COMP_CACHE_F16) {
        return ds4_gpu_tensor_copy_f32_to_f16(g->layer_attn_comp_cache[il],
                                               dst_offset,
                                               g->attn_comp_stage,
                                               0,
                                               count) != 0;
    }

    return ds4_gpu_tensor_copy(g->layer_attn_comp_cache[il],
                               dst_offset,
                               g->attn_comp_stage,
                               0,
                               count * sizeof(float)) != 0;
}

ds4_gpu_tensor *metal_graph_attn_comp_update_target(
        ds4_gpu_graph *g,
        uint32_t       il) {
    return DS4_GPU_ATTN_COMP_CACHE_F16
        ? g->attn_comp_stage
        : g->layer_attn_comp_cache[il];
}

uint32_t metal_graph_attn_comp_update_row(uint32_t row) {
    return DS4_GPU_ATTN_COMP_CACHE_F16 ? 0u : row;
}

bool metal_graph_commit_attn_comp_stage(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    if (!DS4_GPU_ATTN_COMP_CACHE_F16) return true;
    return metal_graph_store_attn_comp_stage(g, il, first_row, rows);
}

ds4_gpu_tensor *metal_graph_attn_comp_row_view(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       row) {
    if (DS4_GPU_ATTN_COMP_CACHE_F16) {
        return ds4_gpu_tensor_view(g->attn_comp_stage,
                                   0,
                                   (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
    }
    return ds4_gpu_tensor_view(g->layer_attn_comp_cache[il],
                               (uint64_t)row * DS4_N_HEAD_DIM * sizeof(float),
                               (uint64_t)DS4_N_HEAD_DIM * sizeof(float));
}

ds4_gpu_tensor *metal_graph_attn_comp_prefill_target(
        ds4_gpu_graph *g,
        uint32_t       il,
        uint32_t       first_row,
        uint32_t       rows) {
    if (DS4_GPU_ATTN_COMP_CACHE_F16) return g->attn_comp_stage;
    const uint32_t view_rows = rows ? rows : 1u;
    return ds4_gpu_tensor_view(g->layer_attn_comp_cache[il],
                               (uint64_t)first_row * DS4_N_HEAD_DIM * sizeof(float),
                               (uint64_t)view_rows * DS4_N_HEAD_DIM * sizeof(float));
}

void metal_graph_attn_comp_prefill_target_free(ds4_gpu_tensor *t) {
    if (!DS4_GPU_ATTN_COMP_CACHE_F16) ds4_gpu_tensor_free(t);
}

/* Encode one DS4 decode layer on Metal.  This is the release single-token
 * layer path; diagnostics reuse it so they compare exactly what generation
 * runs. */

#endif /* !DS4_NO_GPU */
typedef int ds4_core_gpu_decode_util_nonempty_tu; /* 空TU防御(CPU构建) */
