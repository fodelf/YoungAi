/* metal_expert_io.m — ds4_metal.m 机械拆分产物(不改名/不改逻辑/不改字符串)。 */
#import "metal_internal.h"

int ds4_gpu_expert_pread_fd_nocache(void) {
    static int cached_fd = -2;
    if (cached_fd == -2) {
        cached_fd = -1;
        if (g_model_fd >= 0) {
            char path[MAXPATHLEN];
            memset(path, 0, sizeof(path));
            if (fcntl(g_model_fd, F_GETPATH, path) == 0) {
                int nfd = open(path, O_RDONLY);
                if (nfd >= 0) {
                    if (fcntl(nfd, F_NOCACHE, 1) != -1) {
                        cached_fd = nfd;
                        fprintf(stderr,
                                "ds4: batch expert gather cold reads use a separate "
                                "F_NOCACHE descriptor\n");
                    } else {
                        close(nfd);
                    }
                }
            }
        }
        if (cached_fd < 0) {
            fprintf(stderr,
                    "ds4: batch F_NOCACHE descriptor unavailable; "
                    "batch gathers fall back to the shared cached fd\n");
        }
    }
    return cached_fd;
}

int ds4_gpu_expert_pread_fd(void) {
    return g_model_fd;
}

/* P2.1 prediction-accuracy counters (maintained by the prefetch section; the
 * armed prediction marks in metal_expert_stage.m score against them). */
uint64_t g_pf_pred_hits, g_pf_pred_total;

ds4_metal_layer_router g_layer_router[DS4_METAL_EXPERT_PROFILE_MAX_LAYERS];

int ds4_gpu_register_layer_router(
        const void *model_map,
        uint32_t layer,
        uint64_t gate_inp_offset,
        int gate_inp_is_f32,
        uint64_t probs_bias_offset,
        uint64_t gate_exps_offset,
        uint64_t up_exps_offset,
        uint64_t down_exps_offset,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        uint32_t n_embd,
        uint32_t n_expert,
        uint64_t hash_table_offset,
        uint32_t hash_k,
        uint32_t hash_rows) {
    if (!model_map || layer >= DS4_METAL_EXPERT_PROFILE_MAX_LAYERS ||
        n_embd == 0 || n_embd > DS4_METAL_PF_MAX_EMBD ||
        n_expert == 0 || n_expert > DS4_METAL_EXPERT_PROFILE_MAX_EXPERTS ||
        gate_expert_bytes == 0 || down_expert_bytes == 0) {
        return 0;
    }
    if (hash_table_offset != UINT64_MAX && (hash_k == 0 || hash_k > 16u || hash_rows == 0)) {
        return 0;
    }
    ds4_metal_layer_router *r = &g_layer_router[layer];
    r->model_map = model_map;
    r->gate_inp_off = gate_inp_offset;
    r->gate_inp_is_f32 = gate_inp_is_f32;
    r->hash_off = hash_table_offset;
    r->hash_k = hash_k;
    r->hash_rows = hash_rows;
    r->probs_bias_off = probs_bias_offset;
    r->gate_exps_off = gate_exps_offset;
    r->up_exps_off = up_exps_offset;
    r->down_exps_off = down_exps_offset;
    r->gate_expert_bytes = gate_expert_bytes;
    r->down_expert_bytes = down_expert_bytes;
    r->n_embd = n_embd;
    r->n_expert = n_expert;
    r->valid = 1;
    return 1;
}
