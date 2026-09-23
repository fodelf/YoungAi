/* cuda_decode_graph.inc.cu — ds4_cuda.cu 分片: 解码整步 CUDA graph 的原语(2026-09-18, fable5 09-18 立案)。
 *
 * 说人话: 解码一步要发 ~1500 个核, 每发的主机开销 + GPU 侧核间排空实测吃掉 4.5 ms/步(12k 尺 45.7 ms 的一成)。
 * CUDA graph 把整步录成一张图, 之后每步一发 cudaGraphLaunch 重放。录的办法是"流捕获": 开捕获之后照常发核,
 * 核**不执行**只记节点, 结束时实例化成 exec。位置/token 不烤进图 —— 那是 core_decode_graph.c 的事(设备槽)。
 *
 * ★捕获模式用 ThreadLocal, 不用 Relaxed★: 捕获期间任何"不安全"的调用(同步 memcpy / cudaMalloc /
 * cudaDeviceSynchronize)在 ThreadLocal 下直接让捕获作废、capture_end 报 NULL —— 这正是想要的: 这类调用要是被
 * Relaxed 放过去, 它会**立刻执行**而不进图, 重放时那一步就少了(同步 memcpy 尤其致命: 图里没有它, 每步都读旧值)。
 * 出错会怎样: capture_end 返回 NULL 时这一步的核一个都没跑, 调用方必须直发重来, 否则状态少推进一步。
 * 全部走 cudaStreamPerThread: 引擎按 -default-stream per-thread 编译, 无流参数的发射都落在它上面(捕获态照样进图)。 */
static void v41_pdl_register_small(void);   /* 定义在聚合根 ds4_cuda.cu 末尾(那里所有核都已定义) */
int ds4_gpu_decode_graph_capture_begin(void) {
    v41_pdl_register_small();
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &cs) != cudaSuccess || cs != cudaStreamCaptureStatusNone) {
        (void)cudaGetLastError();
        return 0;   /* 已经在别的捕获里(V4 token 图那族), 不嵌套 */
    }
    if (cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        fprintf(stderr, "ds4: [graph] 开捕获失败: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 0;
    }
    return 1;
}

/* 把"核 → 已登记核(v41_pdl_register)"的普通边改成程序化边(见 cuda_internal.cuh 的 PDL 段)。
 * 出口用 LaunchCompletion(上游所有 block 都已开跑就放下游发射): 上游核一行不用改(不必调 trigger);
 * 上游那时已全部驻留, 下游 block 只占空位, 不会把上游饿死。下游完成与内存可见由它自己的 v41_pdl_wait 保证。
 * 返回改了几条; -1 = API 失败(调用方丢掉这张图, 不带着半改的边去实例化)。 */
static int decode_graph_pdl_edges(cudaGraph_t graph) {
    if (g_v41_pdl_n == 0) return 0;
    size_t ne = 0;
    if (cudaGraphGetEdges(graph, NULL, NULL, NULL, &ne) != cudaSuccess) { (void)cudaGetLastError(); return -1; }
    if (ne == 0) return 0;
    cudaGraphNode_t *from = (cudaGraphNode_t *)malloc(ne * sizeof *from), *to = (cudaGraphNode_t *)malloc(ne * sizeof *to);
    cudaGraphEdgeData *ed = (cudaGraphEdgeData *)malloc(ne * sizeof *ed);
    int n = 0, bad = 0;
    if (!from || !to || !ed || cudaGraphGetEdges(graph, from, to, ed, &ne) != cudaSuccess) bad = 1;
    for (size_t i = 0; !bad && i < ne; i++) {
        if (ed[i].type != cudaGraphDependencyTypeDefault || ed[i].from_port != cudaGraphKernelNodePortDefault) continue;
        cudaGraphNodeType tf, tt;
        if (cudaGraphNodeGetType(from[i], &tf) != cudaSuccess || cudaGraphNodeGetType(to[i], &tt) != cudaSuccess) { bad = 1; break; }
        if (tf != cudaGraphNodeTypeKernel || tt != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams kp; memset(&kp, 0, sizeof kp);
        if (cudaGraphKernelNodeGetParams(to[i], &kp) != cudaSuccess) { bad = 1; break; }
        if (!v41_pdl_is_ready(kp.func)) continue;
        cudaGraphEdgeData pe; memset(&pe, 0, sizeof pe);
        pe.from_port = cudaGraphKernelNodePortLaunchCompletion; pe.type = cudaGraphDependencyTypeProgrammatic;
        if (cudaGraphRemoveDependencies(graph, &from[i], &to[i], &ed[i], 1) != cudaSuccess ||
            cudaGraphAddDependencies(graph, &from[i], &to[i], &pe, 1) != cudaSuccess) { bad = 1; break; }
        n++;
    }
    free(from); free(to); free(ed);
    if (bad) { fprintf(stderr, "ds4: [graph] PDL 改边失败: %s\n", cudaGetErrorString(cudaGetLastError())); return -1; }
    return n;
}

void *ds4_gpu_decode_graph_capture_end(void) {
    cudaGraph_t graph = NULL;
    const cudaError_t e = cudaStreamEndCapture(cudaStreamPerThread, &graph);
    if (e != cudaSuccess || !graph) {
        fprintf(stderr, "ds4: [graph] 捕获作废: %s\n", cudaGetErrorString(e));
        (void)cudaGetLastError();
        if (graph) (void)cudaGraphDestroy(graph);
        return NULL;
    }
    size_t n_nodes = 0;
    (void)cudaGraphGetNodes(graph, NULL, &n_nodes);
    const int n_pdl = decode_graph_pdl_edges(graph);
    if (n_pdl < 0) { (void)cudaGraphDestroy(graph); return NULL; }
    cudaGraphExec_t exec = NULL;
    const cudaError_t ei = cudaGraphInstantiate(&exec, graph, 0);
    (void)cudaGraphDestroy(graph);
    if (ei != cudaSuccess || !exec) {
        fprintf(stderr, "ds4: [graph] 实例化失败(%zu 节点): %s\n", n_nodes, cudaGetErrorString(ei));
        (void)cudaGetLastError();
        return NULL;
    }
    fprintf(stderr, "ds4: [graph] 解码整步已捕获: %zu 个节点(其中 %d 条边改成 PDL 程序化边)\n", n_nodes, n_pdl);
    return (void *)exec;
}

int ds4_gpu_decode_graph_launch(void *exec) {
    if (!exec) return 0;
    return cuda_ok(cudaGraphLaunch((cudaGraphExec_t)exec, cudaStreamPerThread), "decode graph launch");
}

void ds4_gpu_decode_graph_free(void *exec) {
    if (exec) (void)cudaGraphExecDestroy((cudaGraphExec_t)exec);
}

/* ★GPU 自旋等主机标志(2026-09-18, 取代 host 节点)★
 * 病(nsys 逐节点实撞): 图里的 host 节点(cudaLaunchHostFunc)在 GB10 上每个留 **0.6~0.9 ms** 的洞 —— 回调线程唤醒 + 回调返回后
 * 驱动再把后续节点推上去的延迟, 与回调里等没等东西无关(engram 第二层真等 0, 洞照样 0.94 ms)。
 * 改成一个 1 线程的小核自旋读映射的 pinned 标志: 主机把行读齐、写完 raw 之后 `flag = seq`, GPU 在 1 µs 内继续。
 * 语义: 等到 flag ≥ want(want 也在 pinned 里, 每步由主机写成本步序号, 图捕一次即可; 序号只增不减, 没有 ABA)。
 * 出错会怎样: 主机那边取行失败没置位 ⇒ GPU 会一直等 —— 所以有超时: 约 5 s 后写 err 并放行, 主机 sync 之后查 err 停车,
 * 不会挂死。__ldcv = ld.global.cv, 每次到一致点取, 不吃 L2 里的旧值(V4 时代实撞过"每个 token 慢一拍")。 */
__global__ static void decode_flagwait_kernel(const int32_t *flag, const int32_t *want, int32_t *err) {
    const int32_t w = __ldcv(want);
    for (uint32_t i = 0; ; i++) {
        if (__ldcv(flag) >= w) return;
        if (i >= 5000000u) { *err = 1; return; }
        __nanosleep(1000);
    }
}
int ds4_gpu_host_flag_wait(const void *flag_pinned, const void *want_pinned, void *err_pinned) {
    void *f = NULL, *w = NULL, *e = NULL;
    if (cudaHostGetDevicePointer(&f, (void *)flag_pinned, 0) != cudaSuccess || cudaHostGetDevicePointer(&w, (void *)want_pinned, 0) != cudaSuccess ||
        cudaHostGetDevicePointer(&e, err_pinned, 0) != cudaSuccess || !f || !w || !e) { (void)cudaGetLastError(); return 0; }
    decode_flagwait_kernel<<<1, 1, 0, cudaStreamPerThread>>>((const int32_t *)f, (const int32_t *)w, (int32_t *)e);
    return cuda_ok(cudaGetLastError(), "host flag wait");
}

/* pinned ↔ 设备的异步拷贝(捕获态 = memcpy 节点; 节点记的是**地址**, 每次重放读那一刻 pinned 里的内容)。
 * 主机侧必须是 ds4_gpu_host_alloc 给的 pinned 内存: 分页内存的 cudaMemcpyAsync 会退化成同步拷贝, 捕获态下更是
 * 直接作废捕获(ThreadLocal 模式把它算作不安全调用)。 */
int ds4_gpu_tensor_write_async(ds4_gpu_tensor *t, uint64_t offset, const void *pinned, uint64_t bytes) {
    if (!t || !pinned || offset > t->bytes || bytes > t->bytes - offset) return 0;
    if (bytes == 0) return 1;
    return cuda_ok(cudaMemcpyAsync((char *)t->ptr + offset, pinned, (size_t)bytes, cudaMemcpyHostToDevice, cudaStreamPerThread),
                   "tensor write async");
}
int ds4_gpu_tensor_read_async(void *pinned, const ds4_gpu_tensor *t, uint64_t offset, uint64_t bytes) {
    if (!t || !pinned || offset > t->bytes || bytes > t->bytes - offset) return 0;
    if (bytes == 0) return 1;
    return cuda_ok(cudaMemcpyAsync(pinned, (const char *)t->ptr + offset, (size_t)bytes, cudaMemcpyDeviceToHost, cudaStreamPerThread),
                   "tensor read async");
}

/* ★零拷贝上传(2026-09-18 实撞)★: 图里的 memcpy 节点在 GB10 上每个要 **~170 µs**(nsys 逐节点: {tok, pos, one-hot} 三个 H2D +
 * 一个 4 B D2H 每步 0.69 ms, engram 两层的 6 KB 上传更是排在 host 节点后面), 而一个 256 线程的小核只要 2~3 µs。
 * 统一内存下 pinned 内存 GPU 可以直接寻址(cudaHostGetDevicePointer), 所以改成核直接读主机内存再写设备。
 * ★必须用 ld.global.cv(__ldcv)★: V4 时代实撞过 —— GB10 的 L2 会缓存主机内存的行, 普通读拿到的是上一步的旧值
 * (输出"用户的用户的请求请求", 每个 token 慢一拍)。.cv = 每次都到一致点取, 绕过缓存。 */
__global__ static void decode_hostcopy_kernel(uint32_t *dst, const uint32_t *src_host, uint32_t nwords) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nwords) dst[i] = __ldcv(src_host + i);
}
int ds4_gpu_tensor_write_zerocopy(ds4_gpu_tensor *t, uint64_t offset, const void *pinned, uint64_t bytes) {
    if (!t || !pinned || offset > t->bytes || bytes > t->bytes - offset || (bytes & 3u) || (offset & 3u)) return 0;
    if (bytes == 0) return 1;
    void *dev = NULL;
    if (cudaHostGetDevicePointer(&dev, (void *)pinned, 0) != cudaSuccess || !dev) {
        (void)cudaGetLastError();
        fprintf(stderr, "ds4: [graph] pinned 内存拿不到设备地址(要 cudaHostAllocMapped)\n");
        return 0;
    }
    const uint32_t nw = (uint32_t)(bytes / 4u);
    decode_hostcopy_kernel<<<(nw + 255u) / 256u, 256, 0, cudaStreamPerThread>>>((uint32_t *)((char *)t->ptr + offset),
                                                                                (const uint32_t *)dev, nw);
    return cuda_ok(cudaGetLastError(), "tensor write zerocopy");
}
/* 反向: 核把设备张量的几个字写进映射的主机内存(argmax 落点), 省掉那个 170 µs 的 D2H 节点; synchronize 之后主机就能读 */
__global__ static void decode_hoststore_kernel(uint32_t *dst_host, const uint32_t *src, uint32_t nwords) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nwords) dst_host[i] = src[i];
}
int ds4_gpu_tensor_read_zerocopy(void *pinned, const ds4_gpu_tensor *t, uint64_t offset, uint64_t bytes) {
    if (!t || !pinned || offset > t->bytes || bytes > t->bytes - offset || (bytes & 3u) || (offset & 3u)) return 0;
    if (bytes == 0) return 1;
    void *dev = NULL;
    if (cudaHostGetDevicePointer(&dev, pinned, 0) != cudaSuccess || !dev) { (void)cudaGetLastError(); return 0; }
    const uint32_t nw = (uint32_t)(bytes / 4u);
    decode_hoststore_kernel<<<(nw + 255u) / 256u, 256, 0, cudaStreamPerThread>>>((uint32_t *)dev, (const uint32_t *)((const char *)t->ptr + offset), nw);
    return cuda_ok(cudaGetLastError(), "tensor read zerocopy");
}
void *ds4_gpu_host_device_ptr(void *pinned) {
    void *dev = NULL;
    if (!pinned || cudaHostGetDevicePointer(&dev, pinned, 0) != cudaSuccess) { (void)cudaGetLastError(); return NULL; }
    return dev;
}
