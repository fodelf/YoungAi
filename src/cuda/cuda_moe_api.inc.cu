/* cuda_moe_api.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * routed_moe one/batch 契约包装。
 */
int ds4_gpu_routed_moe_one_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const ds4_gpu_residual_set *residual, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index) {
    if (residual && residual->vq)
        return cuda_vq_moe_forward(out, mid, residual, model_map, down_offset, down_expert_bytes,
                                   expert_in_dim, expert_mid_dim, out_dim, selected, weights,
                                   n_total_expert, n_expert, clamp, x, layer_index, 1u);
    (void)layer_index; (void)residual;  /* go1b 残差(非 VQ): CUDA 未实现, 见 ds4_gpu.h */
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x, 1);
}
/* ★参数表必须与 ds4_gpu.h 逐字一致★ —— 本文件不 include 那个头(GPU 句柄类型两边
 * 各自实现), 编译器无从校验。此处曾少了 slot_start/slot_count 两个参数: ds4.c 按
 * 头文件压 25 个实参, 这里按 23 个取, mid_is_f16 收到的其实是 slot_start 的值 ⇒
 * 解引用垃圾指针段错误。以前没暴露只是因为本文件根本编译不过(缺 residual_set 定义)。 */
int ds4_gpu_routed_moe_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down, const ds4_gpu_residual_set *residual, const void *model_map, uint64_t model_size, uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset, uint32_t gate_type, uint32_t down_type, uint64_t gate_expert_bytes, uint64_t gate_row_bytes, uint64_t down_expert_bytes, uint64_t down_row_bytes, uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim, const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights, uint32_t n_total_expert, uint32_t n_expert, float clamp, const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens, uint32_t slot_start, uint32_t slot_count, bool *mid_is_f16) {
    if (mid_is_f16) *mid_is_f16 = false;
    /* TP Phase-3 专家切分是双机拆模型用的; CUDA 走单机整模型, 没有对端可 all-reduce。
     * 非平凡切分直接拒绝, 不能只算半边专家却当成完整输出。 */
    if (slot_count != 0u && slot_count != n_expert) {
        fprintf(stderr, "ds4: CUDA 不支持 TP 专家切分 (slot_start=%u slot_count=%u n_expert=%u)\n",
                slot_start, slot_count, n_expert);
        return 0;
    }
    (void)slot_start;
    if (residual && residual->vq)
        return cuda_vq_moe_forward(out, mid, residual, model_map, down_offset, down_expert_bytes,
                                   expert_in_dim, expert_mid_dim, out_dim, selected, weights,
                                   n_total_expert, n_expert, clamp, x, layer_index, n_tokens);
    (void)layer_index; (void)residual;  /* go1b 残差(非 VQ): CUDA 未实现, 见 ds4_gpu.h */
    return routed_moe_launch(out, gate, up, mid, down, model_map, model_size,
                             gate_offset, up_offset, down_offset,
                             gate_type, down_type,
                             gate_expert_bytes, gate_row_bytes,
                             down_expert_bytes, down_row_bytes,
                             expert_in_dim, expert_mid_dim, out_dim,
                             selected, weights, n_total_expert, n_expert, clamp, x, n_tokens);
}
