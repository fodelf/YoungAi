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
    /* 默认关(2026-08-21 A/B): verify 64.9 vs 65.6 = 噪声内。批 kernel 比 decode 大 3x,
     * launch 开销占比小 => 图收益消失; 31% 空转是 kernel 间排空(ramp/drain)不是发射延迟。 */
    if (g_bat_graph_on < 0) g_bat_graph_on = ((const char *)0) /* DS4_CUDA_BATCH_GRAPH: 路径开关已删(2026-08-22 隐形炸弹清理) */ ? 1 : 0;
    if (!g_bat_graph_on || slot < 0 || slot > 3) return 0;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &cs) == cudaSuccess &&
        cs != cudaStreamCaptureStatusNone) return 0;   /* 已在别的 capture 里 */
    if (cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
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
         * 按"删除而非默认关闭"处理, 不留开关; 要恢复须先修好参数补丁并逐位对拍。 */
        g_tok_graph_on = 0;
    if (!g_tok_graph_on) return 0;
    tok_graph_ensure_id_slot();
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
    if (g_tok_id_host) *g_tok_id_host = g_tok_id_want;   /* 发射前落参数槽(此刻流已静) */
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
    if (g_tok_id_host) *g_tok_id_host = (int32_t)token;
    if (cudaGraphLaunch(g_tok_execs[slot], cudaStreamPerThread) != cudaSuccess) {
        (void)cudaGetLastError();
        g_tok_graph_on = 0;
        return 0;
    }
    g_tok_cur = slot;
    return 1;
}

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

