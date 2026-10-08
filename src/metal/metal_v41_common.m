/* metal_v41_common.m — V4.1 Metal 一族的公共件(2026-10-08): 暂存槽 / 核发射器 / 权重范围解析 / 主机同步 / 填充。
 *
 * 发射口径: 全部走 ds4_gpu_command_buffer + ds4_gpu_compute_encoder —— core 的 V4.1 前向在 begin_commands/end_commands 之间,
 * 这里的核就排进同一个批编码器(同一编码器里的 dispatch 对同一缓冲的读写由 Metal 的 hazard tracking 保序); 批没开时各发自己的
 * 命令缓冲并异步提交(ds4_gpu_submit_command_buffer_async), 与 zchain 那组核同一套路。
 * 主机要读设备结果(预填的路由排序 / 反修取料 / Σg²)时用 v41_host_sync: 把已排的命令提交并等完, 批保持打开 —— 不能调 ds4_gpu_synchronize,
 * 它在 Metal 上会把批关掉, core 之后的 end_commands 就报失败(fable5 10-08 Metal 落地记录)。 */
#import "metal_v41.h"

#define V41_SCRATCH_REG_MAX 512u
static v41_scratch *g_v41_reg[V41_SCRATCH_REG_MAX];
static uint32_t g_v41_nreg = 0;
static uint64_t g_v41_gen = 0;
static id<MTLBuffer> g_v41_dummy;   /* NULL 张量的占位绑定(16 B) */

id<MTLBuffer> v41_grow(v41_scratch *s, uint64_t bytes, const char *what) {
    if (bytes <= s->cap && s->buf) return s->buf;
    if (!g_initialized && !ds4_gpu_init()) return nil;
    /* 旧缓冲可能还被在飞的命令缓冲引用: Metal 按引用计数保活到命令完成, 这里直接放手即可 */
    id<MTLBuffer> nb = [g_device newBufferWithLength:(NSUInteger)(bytes ? bytes : 16) options:MTLResourceStorageModeShared];
    if (!nb) { fprintf(stderr, "ds4: [v41-metal] %s 暂存分配失败 (%.1f MB)\n", what, (double)bytes / 1048576.0); return nil; }
    s->buf = nb; s->cap = bytes; s->what = what;
    g_v41_gen++;
    uint32_t i = 0;
    while (i < g_v41_nreg && g_v41_reg[i] != s) i++;
    if (i == g_v41_nreg && g_v41_nreg < V41_SCRATCH_REG_MAX) g_v41_reg[g_v41_nreg++] = s;
    return nb;
}
uint64_t ds4_gpu_v41_scratch_generation(void) { return g_v41_gen; }
uint64_t ds4_gpu_v41_scratch_bytes(void) {
    uint64_t b = 0;
    for (uint32_t i = 0; i < g_v41_nreg; i++) b += g_v41_reg[i]->cap;
    return b;
}
uint64_t ds4_gpu_v41_scratch_release(void) {
    (void)v41_host_sync();
    uint64_t freed = 0;
    for (uint32_t i = 0; i < g_v41_nreg; i++) {
        v41_scratch *s = g_v41_reg[i];
        if (s->buf) freed += s->cap;
        s->buf = nil; s->cap = 0;
    }
    g_v41_gen++;
    return freed;
}

v41_bind v41_bind_buf(id<MTLBuffer> b, NSUInteger off) { v41_bind r; r.buf = b; r.off = off; r.bytes = NULL; r.len = 0; return r; }
v41_bind v41_bind_bytes(const void *p, NSUInteger len) { v41_bind r; r.buf = nil; r.off = 0; r.bytes = p; r.len = len; return r; }
v41_bind v41_bind_tensor(const ds4_gpu_tensor *t) {
    if (!t) {
        if (!g_v41_dummy) g_v41_dummy = [g_device newBufferWithLength:64 options:MTLResourceStorageModeShared];
        return v41_bind_buf(g_v41_dummy, 0);
    }
    return v41_bind_buf(ds4_gpu_tensor_buffer(t), ds4_gpu_tensor_offset(t));
}
v41_bind v41_bind_tensor_off(const ds4_gpu_tensor *t, uint64_t byte_off) {
    v41_bind r = v41_bind_tensor(t);
    if (t) r.off += (NSUInteger)byte_off;
    return r;
}

int v41_launch(const char *kernel, const v41_bind *binds, uint32_t nbind, MTLSize tgs, MTLSize tg) {
    if (!g_initialized && !ds4_gpu_init()) return 0;
    if (tgs.width == 0 || tgs.height == 0 || tgs.depth == 0) return 1;   /* 空网格 = 没活, 成功 */
    @autoreleasepool {
        id<MTLComputePipelineState> pso = ds4_gpu_get_pipeline(kernel);
        if (!pso) { fprintf(stderr, "ds4: [v41-metal] 核 %s 不在库里\n", kernel); return 0; }
        const NSUInteger maxt = pso.maxTotalThreadsPerThreadgroup;
        if (tg.width * tg.height * tg.depth > maxt) {
            fprintf(stderr, "ds4: [v41-metal] 核 %s 要 %lu 线程/threadgroup, 设备上限 %lu\n", kernel,
                    (unsigned long)(tg.width * tg.height * tg.depth), (unsigned long)maxt);
            return 0;
        }
        int owned = 0;
        id<MTLCommandBuffer> cb = ds4_gpu_command_buffer(&owned);
        if (!cb) return 0;
        id<MTLComputeCommandEncoder> enc = ds4_gpu_compute_encoder(cb);
        [enc setComputePipelineState:pso];
        for (uint32_t i = 0; i < nbind; i++) {
            if (binds[i].buf) [enc setBuffer:binds[i].buf offset:binds[i].off atIndex:i];
            else [enc setBytes:binds[i].bytes length:binds[i].len atIndex:i];
        }
        [enc dispatchThreadgroups:tgs threadsPerThreadgroup:tg];
        ds4_gpu_end_compute_encoder(cb, enc);
        return ds4_gpu_submit_command_buffer_async(cb, owned);
    }
}
int v41_launch_1d(const char *kernel, const v41_bind *binds, uint32_t nbind, uint64_t n) {
    if (n == 0) return 1;
    return v41_launch(kernel, binds, nbind, MTLSizeMake((NSUInteger)((n + 255u) / 256u), 1, 1), MTLSizeMake(256, 1, 1));
}

id<MTLBuffer> v41_model_buf(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t len, uint64_t *inner_off, const char *what) {
    if (offset > model_size || len > model_size - offset) {
        fprintf(stderr, "ds4: [v41-metal] %s: 权重范围 [%llu, +%llu) 超出映射 %llu\n", what, (unsigned long long)offset, (unsigned long long)len, (unsigned long long)model_size);
        return nil;
    }
    id<MTLBuffer> b = ds4_gpu_wrap_model_range(model_map, model_size, offset, len, inner_off);
    if (!b) fprintf(stderr, "ds4: [v41-metal] %s: 权重范围没有被映射视图覆盖\n", what);
    return b;
}

int v41_host_sync(void) {
    if (!g_initialized) return 1;
    if (g_batch_cb && !ds4_gpu_flush_commands()) return 0;
    return ds4_gpu_wait_pending_command_buffers("v41 host sync");
}

int v41_fill_buf_u32(id<MTLBuffer> b, NSUInteger off, uint64_t n_u32, uint32_t v) {
    if (!b) return 0;
    if (n_u32 == 0) return 1;
    v41_bind bd[] = { V41_B(b, off), V41_A(n_u32), V41_A(v) };
    return v41_launch_1d("kernel_v41_fill_u32", bd, 3, n_u32);
}
int v41_fill_u32(ds4_gpu_tensor *t, uint64_t byte_off, uint64_t n_u32, uint32_t v) {
    if (!t) return 0;
    return v41_fill_buf_u32(ds4_gpu_tensor_buffer(t), ds4_gpu_tensor_offset(t) + (NSUInteger)byte_off, n_u32, v);
}
