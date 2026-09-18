/* cuda_kv_ring.inc.cu — ds4_cuda.cu 分片: SWA 窗口的环维护(decode.md D1, 2026-09-16)。
 *
 * 说人话: 每层都要记住"最近 128 个 token 的 KV"。原来的存法是一条排好序的线性缓冲, 每算完一步
 * 就把整段往前挪一行 —— 每层两发 256 KB 的设备内拷贝, 40 层一步就是 80 发、20 MB, 纯白搬。
 * 官方的存法是**环**(`window_kv_cache[start_pos % win]`): 位置 a 恒住在 a % 128 那一格, 新的一行
 * 直接盖掉 128 步之前那一行, 一个字节都不用挪。这一片就是环的三件事:
 *
 *   commit  本批算完 → 把这 n 行写进环(只有被"提交"的位置才进环)
 *   save    投机验证批之前 → 把 commit 将要盖掉的那几格存一份
 *   restore 部分接受之后 → 把没被接受的那几格还原回去
 *
 * ★为什么本批的行不直接写进环★: 投机验证一次算 1+k 个位置, 其中没被接受的那几位不算数。
 * 它们要是进了环, 就盖掉了 128 步之前的真行, 而那几行正好还在下一步的可见窗口里 ——
 * 不报错, 只是下一步读到"未来的草稿"当历史, 输出慢慢走偏。所以本批的行先待在缓冲的
 * [window, window+n) 段(注意力按 v41_win_row 去那里读), 确认接受几位之后才 commit 几行。
 *
 * ★出错会怎样★: commit 的行数给多了(把没接受的也提交了)= 上面那个病; 给少了 = 下一步在环里
 * 读到 128 步前的陈旧行, 症状同样是输出悄悄变样, 温 0 逐字节门会抓。 */

/* 本批第 i 行(缓冲第 window+i 行) → 环的第 (pos0+i) % window 格。
 * grid.x = 要提交的行数, 每 block 搬一行 hd 个 float。 */
__global__ static void v41_win_commit_kernel(float *win, uint32_t pos0, uint32_t i0, uint32_t window, uint32_t hd,
                                             const int32_t *posd) {
    const uint32_t j = blockIdx.x, i = i0 + j;
    if (posd) pos0 = (uint32_t)posd[0];   /* graph 路: 位置在设备槽(ds4_gpu_v41.h "设备位置"口径) */
    const float *src = win + (uint64_t)(window + i) * hd;
    float *dst = win + (uint64_t)((pos0 + i) % window) * hd;
    for (uint32_t d = threadIdx.x; d < hd; d += blockDim.x) dst[d] = src[d];
}

/* save: snap[j] ← 环的第 (pos0+i0+j) % window 格(commit 将要盖掉的那些)。
 * restore 是反过来写回去 —— 两件事共用一个核, back≠0 就是回写。 */
__global__ static void v41_win_ring_snap_kernel(float *win, float *snap, uint32_t pos0, uint32_t i0,
                                                uint32_t window, uint32_t hd, uint32_t back) {
    const uint32_t j = blockIdx.x, i = i0 + j;
    float *ring = win + (uint64_t)((pos0 + i) % window) * hd;
    float *sp = snap + (uint64_t)i * hd;   /* 按批内行号存, 回滚时按"接受几位"直接取区间 */
    if (back) { for (uint32_t d = threadIdx.x; d < hd; d += blockDim.x) ring[d] = sp[d]; }
    else      { for (uint32_t d = threadIdx.x; d < hd; d += blockDim.x) sp[d] = ring[d]; }
}

/* 提交本批的 n 行。n > window 时(预填一块 512 个 token)只有最后 window 行能留在环里,
 * 前面的行写进去也会被后面的盖掉 —— 直接只提交最后 window 行, 省掉白写, 结果完全一样。
 * ★只提交最后 window 行是等价的, 不是近似★: 环只有 window 格, 第 i 行与第 i+window 行落同一格,
 * 按 i 升序写的最终结果就是"最后 window 行各就各位"。 */
int ds4_gpu_v41_win_commit_tensor(ds4_gpu_tensor *win, uint32_t pos0, uint32_t n, uint32_t window, uint32_t head_dim,
                                  const ds4_gpu_tensor *posd) {
    if (!win || !window || !n) return 0;
    if (win->bytes < (uint64_t)(window + n) * head_dim * 4) return 0;
    if (posd && n != 1u) return 0;
    const uint32_t i0 = n > window ? n - window : 0u, rows = n - i0;
    v41_win_commit_kernel<<<rows, 256, 0, g_cur_stream>>>((float *)win->ptr, pos0, i0, window, head_dim,
                                                          posd ? (const int32_t *)posd->ptr : NULL);
    return cuda_ok(cudaGetLastError(), "v41 win commit");
}

/* 投机: 存/还原环里会被本批 commit 盖掉的那几格。rows 给 n(存)或"从第 keep 行起到 n"(还原)。 */
int ds4_gpu_v41_win_ring_snap_tensor(ds4_gpu_tensor *win, ds4_gpu_tensor *snap, uint32_t pos0,
                                     uint32_t i0, uint32_t n, uint32_t window, uint32_t head_dim, int back) {
    if (!win || !snap || !window || n <= i0) return 1;   /* 没有要动的行 = 成功 */
    if (snap->bytes < (uint64_t)n * head_dim * 4) return 0;
    v41_win_ring_snap_kernel<<<n - i0, 256, 0, g_cur_stream>>>((float *)win->ptr, (float *)snap->ptr,
                                                               pos0, i0, window, head_dim, back ? 1u : 0u);
    return cuda_ok(cudaGetLastError(), back ? "v41 win ring restore" : "v41 win ring save");
}
