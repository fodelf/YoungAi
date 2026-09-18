/* cuda_graphcap.inc.cu — ds4_cuda.cu 机械拆分分片(聚合根按序 #include, 单 TU 语义不变)。
 * GPU 跨度计时 + verify 批/token CUDA Graph 捕获。
 */
/* GPU 跨度计时(2026-08-21 verify 归因): 事件对夹住一段, end 返回 GPU 侧毫秒。
 * 与 host 计时对照即可分离"GPU 真忙" vs "CPU 编码/背压阻塞"。 */
static cudaEvent_t g_span_a = NULL, g_span_b = NULL;
void ds4_gpu_span_begin(void) {
    if (!g_span_a) { if (cudaEventCreate(&g_span_a) != cudaSuccess) { g_span_a = NULL; return; } }
    if (!g_span_b) { if (cudaEventCreate(&g_span_b) != cudaSuccess) { g_span_b = NULL; return; } }
    (void)cudaEventRecord(g_span_a, cudaStreamPerThread);
}
float ds4_gpu_span_end(void) {
    if (!g_span_a || !g_span_b) return -1.0f;
    if (cudaEventRecord(g_span_b, cudaStreamPerThread) != cudaSuccess) return -1.0f;
    if (cudaEventSynchronize(g_span_b) != cudaSuccess) return -1.0f;
    float ms = -1.0f;
    (void)cudaEventElapsedTime(&ms, g_span_a, g_span_b);
    return ms;
}

/* verify 批 CUDA 图(2026-08-21): 实测每轮发 ~3494 个 kernel, 平均相邻间隙 5.4us
 * => 每轮 ~19ms GPU 空转(利用率仅 69%)。decode 早已用 token graph 消除这一项, 批路径
 * 一直没有。这里按 (pos0 & 3) 四相建槽(ratio-4 压缩器周期决定拓扑), 同相拓扑同构 =>
 * cudaGraphExecUpdate 只改 kernel 参数; 失败则回退直发(调用方重编码)。 */
static cudaGraphExec_t g_bat_execs[4] = {NULL, NULL, NULL, NULL};
static int g_bat_graph_on = -1;
static int g_bat_slot = -1;

int ds4_gpu_batch_graph_begin(int slot) {
    /* 08-21 关(verify 64.9 vs 65.6 无肉: 那时批核又大又慢, 发射税占比小)。09-07 重开: 批核已提到 decode 级, 剖面按
     * "本核结束 − 上一核结束"算真实增量, 五个大核只占 ~23 ms/轮, 其余 ~60 ms 摊在 3198 个小核的发射/排空上 ——
     * 与 decode token 图 09-05 的故事同构。候选数钉死 4 ⇒ 同相位拓扑同构, ExecUpdate 只改参数。 */
    /* 09-07 实测(p15): 每轮重捕获 + ExecUpdate 的主机开销让轮空隙 3.8 → 12.3 ms, GPU 忙只省 3 ms ⇒ 净亏 6 ms/轮。
     * 批图要赚钱得像 decode token 图那样一次捕获、参数走设备槽重放, 不是每轮重捕获; 先关。 */
    if (g_bat_graph_on < 0) g_bat_graph_on = 0;
    if (!g_bat_graph_on || slot < 0 || slot > 3) return 0;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &cs) == cudaSuccess &&
        cs != cudaStreamCaptureStatusNone) return 0;   /* 已在别的 capture 里 */
    /* Relaxed 模式(09-07): 固定工作区后 cuBLAS GemmEx(drafter 建窗的 q8 f16 影子 GEMM)在 ThreadLocal 捕获态仍报
     * status 14 —— cuBLAS 内部有 cudaStreamQuery 一类"可能不安全"的调用, 严格模式一律作废捕获。Relaxed 只是不拦这些
     * 调用(它们不进图), kernel/memcpy 节点照常捕获; 批路里没有主机同步(有则 verify 结果早就错了)。token 图仍 ThreadLocal。 */
    if (cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeRelaxed) != cudaSuccess) {
        (void)cudaGetLastError();
        g_bat_graph_on = 0;
        return 0;
    }
    g_bat_slot = slot;
    return 1;
}

int ds4_gpu_batch_graph_end_launch(int encode_ok) {
    if (g_bat_slot < 0) return 0;
    const int slot = g_bat_slot;
    g_bat_slot = -1;
    cudaGraph_t graph = NULL;
    const cudaError_t e = cudaStreamEndCapture(cudaStreamPerThread, &graph);
    if (e != cudaSuccess || !graph || !encode_ok) {
        (void)cudaGetLastError();
        if (graph) (void)cudaGraphDestroy(graph);
        g_bat_graph_on = 0;   /* 一次失败就永久回退, 不反复试探 */
        fprintf(stderr, "ds4: CUDA batch graph capture failed; falling back to direct launch\n");
        return -1;
    }
    int ok = 1;
    if (!g_bat_execs[slot]) {
        if (cudaGraphInstantiate(&g_bat_execs[slot], graph, NULL, NULL, 0) != cudaSuccess) {
            (void)cudaGetLastError(); g_bat_execs[slot] = NULL; ok = 0;
        }
    } else {
        cudaGraphExecUpdateResult ur;
        cudaGraphNode_t err_node = NULL;
        if (cudaGraphExecUpdate(g_bat_execs[slot], graph, &err_node, &ur) != cudaSuccess) {
            (void)cudaGetLastError();
            (void)cudaGraphExecDestroy(g_bat_execs[slot]);
            g_bat_execs[slot] = NULL;
            if (cudaGraphInstantiate(&g_bat_execs[slot], graph, NULL, NULL, 0) != cudaSuccess) {
                (void)cudaGetLastError(); g_bat_execs[slot] = NULL; ok = 0;
            }
        }
    }
    (void)cudaGraphDestroy(graph);
    if (!ok) { g_bat_graph_on = 0; return -1; }
    if (cudaGraphLaunch(g_bat_execs[slot], cudaStreamPerThread) != cudaSuccess) {
        (void)cudaGetLastError();
        g_bat_graph_on = 0;
        return -1;
    }
    return 1;
}

/* 路由空槽消毒(2026-08-21 抓到 drafter 不可复现的真因): drafter 约 1.1% 的路由槽是 -1
 * (router 在退化行上选不满 top-k)。引擎只在"使用点"clamp(<0 → 0), 但 sorted-pairs 记账
 * 走 counts[selected[pair]] —— **counts[-1] 是越界原子写**, 同时该 pair 拿不到 tile ⇒ 它的
 * mid 槽从未被写 ⇒ down 读到未初始化内存。症状: 同输入两跑草稿完全不同(logits 差 5%),
 * 偶发 NaN, acc 大幅波动。这里在记账前把 -1 归零并把其权重清零(语义等价: 空槽无贡献)。 */
__global__ static void sanitize_router_kernel(int32_t *sel, float *w, uint32_t n, uint32_t n_expert) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t e = sel[i];
    if (e < 0 || (uint32_t)e >= n_expert) { sel[i] = 0; if (w) w[i] = 0.0f; }
}

int ds4_gpu_sanitize_router_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
                                              uint32_t n_pairs, uint32_t n_total_expert) {
    if (!selected || n_pairs == 0) return 0;
    sanitize_router_kernel<<<(n_pairs + 255u) / 256u, 256, 0, g_cur_stream>>>(
        (int32_t *)selected->ptr, weights ? (float *)weights->ptr : NULL, n_pairs, n_total_expert);
    return cuda_ok(cudaGetLastError(), "sanitize router");
}

__global__ static void sanitize_finite_kernel(float *x, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    if (!isfinite(v)) x[i] = 0.0f;
}

/* drafter 数值护栏(2026-08-21): DSpark drafter 约 1-4% 的行会产出 NaN(实测锚里
 * blk0 干净输入也能出 10 个 NaN 维, 新旧两条 MoE kernel 同现 ⇒ 既有问题不是本轮改动)。
 * NaN 一旦落到锚位就整轮草稿作废(acc 塌到 1.00 的间歇故障)。这里在每块出口清洗,
 * 代价 = B×hc_dim 一次写。 */
int ds4_gpu_sanitize_finite_tensor(ds4_gpu_tensor *t, uint64_t n_float) {
    if (!t || n_float == 0 || t->bytes < n_float * sizeof(float)) return 0;
    sanitize_finite_kernel<<<(unsigned)((n_float + 255u) / 256u), 256, 0, g_cur_stream>>>(
        (float *)t->ptr, n_float);
    return cuda_ok(cudaGetLastError(), "sanitize finite");
}

/* ===== 预发射(2026-09-07): 图开头取 token / 图末尾 argmax / 主机侧发射与回传 =====
 * 契约见 ds4_gpu_core.h。数值: 图内所有核不变, 只多了取槽核与两级 argmax; argmax 的并列语义
 * 与主机 sample_argmax / ds4_session_argmax_excluding 同一条规则(严格大于, 首个最大胜, NaN 永不选)
 * ⇒ 主机事后算出的 token 与设备槽逐位一致, core 用这一点做预发射对账。 */
__global__ static void tok_id_load_kernel(int32_t *dev_id, const int32_t *slot, const int32_t *next_dev) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    /* slot 是主机 pinned 内存: GB10 统一内存下 GPU L2 会缓存它, 普通读拿到的是上一图的旧行(实撞:
     * 输出"用户的用户的请求请求", 每个 token 慢一拍)。volatile = ld.volatile(系统作用域强读), 绕过缓存到
     * 一致点取主机刚写的值; 主机侧写完做 fence(tok_slot_write)。 */
    const volatile int32_t *vs = slot;
    const int32_t mode = vs[0], id = vs[1];
    *dev_id = mode ? *next_dev : id;
}
/* 规则(与主机 sample_argmax / ds4_session_argmax_excluding 同): 严格大于才替换 ⇒ 首个最大胜, NaN 永不
 * 入选(x > m 对 NaN 恒假); 候选 (m, i), i<0 = 无候选。两级: 64 block 各扫一段连续下标, 再 1 block 归约。
 * 首版单 block 串扫 129280 个 logit 要 653 µs/token(剖面实测), 把边界省下的时间又吃回去; 两级 ≈ 10 µs。 */
#define DS4_ARGMAX_BLOCKS 64u
__device__ __forceinline__ static void argmax_block_reduce(float *bm, int32_t *bi, float m, int32_t mi) {
    bm[threadIdx.x] = m; bi[threadIdx.x] = mi;
    __syncthreads();
    for (uint32_t s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float lm = bm[threadIdx.x], rm = bm[threadIdx.x + s];
            const int32_t li = bi[threadIdx.x], ri = bi[threadIdx.x + s];
            const bool take = (ri >= 0) && (li < 0 || rm > lm || (rm == lm && ri < li));
            if (take) { bm[threadIdx.x] = rm; bi[threadIdx.x] = ri; }
        }
        __syncthreads();
    }
}
__global__ static void decode_argmax_part_kernel(float *pm, int32_t *pi, const float *logits,
                                                 uint32_t vocab, int32_t exclude) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    __shared__ float bm[256];
    __shared__ int32_t bi[256];
    const uint32_t per = (vocab + gridDim.x - 1u) / gridDim.x;
    const uint32_t beg = blockIdx.x * per, end = (beg + per < vocab) ? beg + per : vocab;
    float m = -INFINITY; int32_t mi = -1;
    for (uint32_t v = beg + threadIdx.x; v < end; v += blockDim.x) {
        if ((int32_t)v == exclude) continue;
        const float x = logits[v];
        if (x > m) { m = x; mi = (int32_t)v; }
    }
    argmax_block_reduce(bm, bi, m, mi);
    if (threadIdx.x == 0) { pm[blockIdx.x] = bm[0]; pi[blockIdx.x] = bi[0]; }
}
__global__ static void decode_argmax_final_kernel(int32_t *out_id, const float *pm, const int32_t *pi,
                                                  uint32_t n, int32_t exclude) {
    DS4_PDL_WAIT(); DS4_PDL_TRIGGER();
    __shared__ float bm[256];
    __shared__ int32_t bi[256];
    const float m = (threadIdx.x < n) ? pm[threadIdx.x] : -INFINITY;
    const int32_t mi = (threadIdx.x < n) ? pi[threadIdx.x] : -1;
    argmax_block_reduce(bm, bi, m, mi);
    if (threadIdx.x == 0) *out_id = (bi[0] < 0) ? ((exclude == 0) ? 1 : 0) : bi[0];   /* 全 -inf/NaN 兜底同主机 */
}
static void tok_slot_write(int slot, int32_t mode, int32_t id) {
    if (!g_tok_slots_host) return;
    ((volatile int32_t *)g_tok_slots_host)[2 * slot] = mode;
    ((volatile int32_t *)g_tok_slots_host)[2 * slot + 1] = id;
    __sync_synchronize();   /* 先落内存再发射, GPU 的系统作用域读才看得到 */
}
int ds4_gpu_decode_prelaunch_capable(void) { return 1; }
int ds4_gpu_decode_argmax_tensor(const ds4_gpu_tensor *logits, uint32_t n_vocab, int exclude_id) {
    if (!logits || logits->bytes < (uint64_t)n_vocab * sizeof(float)) return 0;
    tok_graph_ensure_id_slot();   /* 首次解码 encode 在 capture 之前已建槽(token_graph_begin), 这里只是保险 */
    if (!g_tok_next_dev || !g_argmax_pm) return 0;
    /* 分段核故意不用 PDL 发射: 前序是 1.27 ms 的输出头矩阵, PDL 会让这 64 个 block 提前上 SM 干等
     * 整段时间(剖面把等待算进核时长: 476 µs/token), 还占着输出头的驻留槽。普通发射只多 ~2 µs 边界。 */
    decode_argmax_part_kernel<<<DS4_ARGMAX_BLOCKS, 256>>>(
        g_argmax_pm, g_argmax_pi, (const float *)logits->ptr, n_vocab, (int32_t)exclude_id);
    ds4_launch_pdl(decode_argmax_final_kernel, 1, 256, 0, 0,
                   g_tok_next_dev, (const float *)g_argmax_pm, (const int32_t *)g_argmax_pi,
                   DS4_ARGMAX_BLOCKS, (int32_t)exclude_id);
    return cuda_ok(cudaGetLastError(), "decode argmax launch");
}
int ds4_gpu_token_graph_prelaunch(uint32_t pos, int need_logits) {
    if (g_tok_graph_on <= 0 || g_tok_pending < 0 || g_tok_pending_pos != pos ||
        g_tok_pending_logits != need_logits || !g_tok_execs[g_tok_pending] ||
        !g_tok_slots_host || !g_tok_next_dev)
        return 0;
    const int slot = g_tok_pending;
    g_tok_pending = -1;
    tok_slot_write(slot, 1, 0);   /* 取设备槽 */
    if (cudaGraphLaunch(g_tok_execs[slot], cudaStreamPerThread) != cudaSuccess) {
        (void)cudaGetLastError();
        g_tok_graph_on = 0;
        return 0;
    }
    g_tok_cur = slot;
    g_tok_prelaunched_pos = (int64_t)pos;
    return 1;
}
/* core 在 eval(pos) 开头认领"pos 已预发射"(只有发起预发射的那个 graph 会来认领: 它自己记着
 * prelaunched_pos, 别的会话不会误领), 认领即清; token 对账由 core 做。 */
int ds4_gpu_token_graph_prelaunch_claim(uint32_t pos) {
    if (g_tok_prelaunched_pos != (int64_t)pos) return 0;
    g_tok_prelaunched_pos = -1;
    return 1;
}
int ds4_gpu_decode_readback_async(const ds4_gpu_tensor *logits, uint64_t bytes,
                                  float *pinned_logits, int32_t *pinned_next_tok) {
    if (!logits || !pinned_logits || bytes > logits->bytes) return 0;
    tok_graph_ensure_id_slot();
    if (!g_readback_ev) return 0;
    if (cudaMemcpyAsync(pinned_logits, logits->ptr, (size_t)bytes, cudaMemcpyDeviceToHost,
                        cudaStreamPerThread) != cudaSuccess)
        return cuda_ok(cudaGetLastError(), "logits readback");
    if (pinned_next_tok) {
        if (!g_tok_next_dev) return 0;
        if (cudaMemcpyAsync(pinned_next_tok, g_tok_next_dev, sizeof(int32_t), cudaMemcpyDeviceToHost,
                            cudaStreamPerThread) != cudaSuccess)
            return cuda_ok(cudaGetLastError(), "next token readback");
        /* [1] = 本图实际消费的 token(取槽核写的 g_tok_id_dev), core 对账/诊断用 */
        if (g_tok_id_dev && cudaMemcpyAsync(pinned_next_tok + 1, g_tok_id_dev, sizeof(int32_t),
                                            cudaMemcpyDeviceToHost, cudaStreamPerThread) != cudaSuccess)
            return cuda_ok(cudaGetLastError(), "consumed token readback");
    }
    return cuda_ok(cudaEventRecord(g_readback_ev, cudaStreamPerThread), "readback event");
}
int ds4_gpu_decode_readback_wait(void) {
    if (!g_readback_ev) return 0;
    return cuda_ok(cudaEventSynchronize(g_readback_ev), "readback wait");
}
void *ds4_gpu_host_alloc(uint64_t bytes) {
    void *p = NULL;
    if (cudaHostAlloc(&p, (size_t)bytes, cudaHostAllocMapped) != cudaSuccess) { (void)cudaGetLastError(); return NULL; }
    return p;
}
void ds4_gpu_host_free(void *p) { if (p) (void)cudaFreeHost(p); }

void ds4_gpu_token_graph_set_pos(uint32_t pos) { g_tok_launch_pos = pos; }
int ds4_gpu_token_graph_begin(void) {
    if (g_tok_graph_on < 0)
        /* ★decode token graph 整族关闭(2026-08-22 实测判定)★
         * 图在建图那次算对, 之后重放时逐 token 变化的注意力范围参数(KV 行数/raw 窗口起点/
         * 可见压缩条目数)没有被正确打补丁 —— 每个解码 token 都在用错误的历史范围。
         * 症状: 位置 0 正确, 之后系统性变差。实测同一序列同一模型:
         *     批量 prefill 路      PPL = 44.10
         *     解码路 token图【开】 PPL = 55.30   ← 差 25%
         *     解码路 token图【关】 PPL = 45.72   ← 回到批量路水平
         * 影响面: decode 是生成的唯一路径, 所有生成质量都被它压着; 本项目此前全部判决
         * 数字都是在这个 bug 上量的。代价: generation 39.18 → 36.05 t/s(−8%), 换 25% PPL。
         * 按"删除而非默认关闭"处理, 不留开关; 要恢复须先修好参数补丁并逐位对拍。
         * ★09-05 重开并定罪(用户令"利用好 CUDA 架构")★: 08-22 "PPL 差 25%"的真身是三个
         * 捕获态 bug, 全部机理级修掉(fable5 09-05 22:20 记录): ①fused2 探针在捕获态不跑 ⇒ 图内
         * 走老 VQ 核(慢 2.9×, 这就是"开图反而掉"); ②splitk 部分和缓冲捕获态不分配 ⇒ 图内走另一
         * 条归约序(现 cuda_decode_scratch_prepare 进捕获前预建); ③hash 路由 token id 以 host 标量
         * 烤进预捕获的图 ⇒ 重放用上一个 token 的 id(现读设备槽 g_tok_id_dev)。验收尺 =
         * tokgraph_ab_spark.sh 对直发二进制 --dump-logprobs 贪心逐字节比: 短提示 256 token +
         * 长提示(4300 token, 稀疏 indexer 路)64 token 全同。任何触碰捕获态的改动都要复跑它。 */
        g_tok_graph_on = 1;
    if (!g_tok_graph_on) return 0;
    tok_graph_ensure_id_slot();
    cuda_decode_scratch_prepare();   /* capture 内禁分配的 decode scratch 先建好(见 cuda_internal.cuh) */
    if (cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        (void)cudaGetLastError();
        g_tok_graph_on = 0;
        return 0;
    }
    return 1;
}

int ds4_gpu_token_graph_end_launch(void) {
    if (!g_tok_graph_on) return 0;
    cudaGraph_t graph = NULL;
    if (cudaStreamEndCapture(cudaStreamPerThread, &graph) != cudaSuccess || !graph) {
        (void)cudaGetLastError();
        g_tok_graph_on = 0;
        if (graph) (void)cudaGraphDestroy(graph);
        fprintf(stderr, "ds4: CUDA token graph capture failed; falling back to direct launch\n");
        return -1;   /* 捕获期 kernel 未执行, 调用方必须重编码 */
    }
    const int slot_el = (int)(g_tok_launch_pos & 3u);   /* 四相: 同相拓扑同构 update 恒过 */
    const int ok = tok_graph_finalize_slot(graph, slot_el);
    (void)cudaGraphDestroy(graph);
    if (!ok) {
        g_tok_graph_on = 0;
        fprintf(stderr, "ds4: CUDA token graph instantiate failed; falling back\n");
        return -1;
    }
    g_tok_cur = slot_el;
    tok_slot_write(slot_el, 0, g_tok_id_want);   /* 发射前落本相位参数槽(此刻流已静) */
    if (cudaGraphLaunch(g_tok_execs[slot_el], cudaStreamPerThread) != cudaSuccess) {
        (void)cudaGetLastError();
        g_tok_graph_on = 0;
        return -1;
    }
    return 1;
}

/* 命中预编码: 写参数槽后直接发射, encode 零成本。返回 1=已发射。 */
int ds4_gpu_token_graph_try_pending(int token, uint32_t pos, int need_logits) {
    if (((const char *)0) /* DS4_TOK_GRAPH_DEBUG: 诊断开关已删(2026-08-22) */ && (g_tok_graph_on <= 0 || g_tok_pending < 0 ||
        g_tok_pending_pos != pos || g_tok_pending_logits != need_logits ||
        !g_tok_execs[g_tok_pending < 0 ? 0 : g_tok_pending]))
        fprintf(stderr, "[tokdbg] miss pos=%u: on=%d pending=%d ppos=%u plog=%d\n",
                pos, g_tok_graph_on, g_tok_pending, g_tok_pending_pos, g_tok_pending_logits);
    if (g_tok_graph_on <= 0 || g_tok_pending < 0 ||
        g_tok_pending_pos != pos || g_tok_pending_logits != need_logits ||
        !g_tok_execs[g_tok_pending])
        return 0;
    const int slot = g_tok_pending;
    g_tok_pending = -1;
    tok_slot_write(slot, 0, (int32_t)token);
    if (cudaGraphLaunch(g_tok_execs[slot], cudaStreamPerThread) != cudaSuccess) {
        (void)cudaGetLastError();
        g_tok_graph_on = 0;
        return 0;
    }
    g_tok_cur = slot;
    return 1;
}

/* 待发射图作废: exec 槽留着(下次同相位 ExecUpdate 复用), 只清"待发射"标记, 免得一次位置错位后
 * 又回到同一 pos 时把陈旧参数的图发出去。 */
int ds4_gpu_token_graph_pending_discard(void) { g_tok_pending = -1; return 1; }

int ds4_gpu_token_graph_precapture_begin(void) {
    if (g_tok_graph_on <= 0 || !g_tok_id_dev || !g_tok_id_host) {
        if (((const char *)0) /* DS4_TOK_GRAPH_DEBUG: 诊断开关已删(2026-08-22) */)
            fprintf(stderr, "[tokdbg] precap-skip: on=%d id_dev=%d id_host=%d\n",
                    g_tok_graph_on, g_tok_id_dev != NULL, g_tok_id_host != NULL);
        return 0;
    }
    if (cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        if (((const char *)0) /* DS4_TOK_GRAPH_DEBUG: 诊断开关已删(2026-08-22) */)
            fprintf(stderr, "[tokdbg] precap BeginCapture FAIL\n");
        (void)cudaGetLastError();
        return 0;
    }
    return 1;
}

static double tokdbg_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}
int ds4_gpu_token_graph_precapture_end(uint32_t pos, int need_logits, int encode_ok) {
    const int dbg = ((const char *)0) /* DS4_TOK_GRAPH_DEBUG: 诊断开关已删(2026-08-22) */ != NULL;
    const double td0 = dbg ? tokdbg_now() : 0.0;
    cudaGraph_t graph = NULL;
    if (cudaStreamEndCapture(cudaStreamPerThread, &graph) != cudaSuccess) {
        (void)cudaGetLastError();
        if (graph) (void)cudaGraphDestroy(graph);
        g_tok_pending = -1;
        return 0;
    }
    if (!encode_ok || !graph) {
        if (((const char *)0) /* DS4_TOK_GRAPH_DEBUG: 诊断开关已删(2026-08-22) */)
            fprintf(stderr, "[tokdbg] precap-end reject: encode_ok=%d graph=%d\n", encode_ok, graph != NULL);
        if (graph) (void)cudaGraphDestroy(graph);
        g_tok_pending = -1;
        return 0;
    }
    const double td1 = dbg ? tokdbg_now() : 0.0;
    if (((const char *)0) /* DS4_TOK_GRAPH_DOT: 路径开关已删(2026-08-22 隐形炸弹清理) */) {   /* 奇偶图导出 diff 用: pos 8/9 各一张 */
        if (pos == 7u || pos == 9u) {
            char fp[64]; snprintf(fp, sizeof fp, "/tmp/tokgraph_pos%u.dot", pos);
            (void)cudaGraphDebugDotPrint(graph, fp, 0);
        }
    }
    const int slot = (int)(pos & 3u);    /* 四相分槽; 目标槽上次发射≥1个全同步前=已静默 */
    const int ok = tok_graph_finalize_slot(graph, slot);
    const double td2 = dbg ? tokdbg_now() : 0.0;
    (void)cudaGraphDestroy(graph);
    if (dbg) fprintf(stderr, "[tokdbg] precap-end pos=%u endcap=%.1fms finalize=%.1fms destroy=%.1fms\n",
                     pos, td1 - td0, td2 - td1, tokdbg_now() - td2);
    if (!ok) { g_tok_pending = -1; return 0; }
    g_tok_pending = slot;
    g_tok_pending_pos = pos;
    g_tok_pending_logits = need_logits;
    /* 预上传(09-05): ExecUpdate 改过参数的 exec 在下次 cudaGraphLaunch 时才把节点参数推到设备,
     * nsys 见 bench 4096 ctx 每 token cudaGraphLaunch 平均 2.9~3.7 ms(短 ctx 0.48), 这段落在
     * token 边界的关键路径上。cudaGraphUpload 在独立非阻塞流上做同一件事: 只排在本 exec 上一次
     * 发射(4 token 前, 早完成)之后, 不排在主流当前 token 之后, 于是与 GPU 跑当前 token 重叠。
     * 失败只是没预热, 发射时照常上传, 不影响正确性。 */
    static cudaStream_t upload_stream = NULL;
    if (!upload_stream &&
        cudaStreamCreateWithFlags(&upload_stream, cudaStreamNonBlocking) != cudaSuccess) {
        upload_stream = NULL; (void)cudaGetLastError();
    }
    if (upload_stream && cudaGraphUpload(g_tok_execs[slot], upload_stream) != cudaSuccess)
        (void)cudaGetLastError();
    return 1;
}

int ds4_gpu_begin_commands(void) { return 1; }
int ds4_gpu_flush_commands(void) {
    /* 在 token-graph capture 期间, flush(全设备同步)既非法也无意义 —— 图作为整体
     * 执行, 中途没有 host 可见状态。capture 态直接成功返回。 */
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &cs) == cudaSuccess &&
        cs == cudaStreamCaptureStatusActive) {
        return 1;
    }
    (void)cudaGetLastError();
    return cuda_ok(cudaDeviceSynchronize(), "flush");
}
int ds4_gpu_end_commands(void) {
    /* capture 期间全设备同步既非法也无意义(图整体执行), 直接成功返回。 */
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &cs) == cudaSuccess &&
        cs == cudaStreamCaptureStatusActive) return 1;
    return cuda_ok(cudaDeviceSynchronize(), "end commands");
}
int ds4_gpu_synchronize(void) { return cuda_ok(cudaDeviceSynchronize(), "synchronize"); }
/* TP rendezvous: CUDA has no MTLSharedEvent fast path, but ds4_gpu_flush_commands
 * already does a full device sync, so by the time the host waits the results are
 * visible. signal returns a nonzero token; host_wait is a no-op success. */
uint64_t ds4_gpu_tp_signal_after_batch(void) { return 1; }
int ds4_gpu_tp_host_wait(uint64_t value) { (void)value; return 1; }

