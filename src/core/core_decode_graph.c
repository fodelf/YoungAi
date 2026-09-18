/* core_decode_graph.c — 解码整步 CUDA graph 的编排(2026-09-18, fable5 09-18 立案)。
 *
 * 说人话: 解码一步 = 40 层 ~1500 发核。直发时每发都要主机敲一次门、GPU 核间也各排一次空, 12k 尺实测吃 4.5 ms/步
 * (整步 45.7 ms 的一成), 步边界还有三次同步 cudaMemcpy。这里把一整步录成一张 CUDA graph, 之后每步只做三件事:
 * 往 pinned 槽写 {token, 位置} → 一发 cudaGraphLaunch → 等完读回设备 argmax 出的下一个 token。
 *
 * 图为什么能只捕一次: 所有随位置变的东西都不烤进图 ——
 *   ①位置: 核从 st->pos(图开头由 memcpy 节点从 pinned 槽灌进去)读, 见 ds4_gpu_v41.h "设备位置"口径;
 *   ②grid/shared: 按位置桶(DGRAPH_BUCKET 个位置一桶)的上限开, 跨桶重捕获;
 *   ③engram 的盘读: 图里 engram 层前放 host 节点等线程池 pread 完 + memcpy 节点从常驻 pinned 行缓冲上传;
 *   ④压缩源层"凑满一组才池化": 核里按位置判, 没凑满写垃圾槽(core_v41_attn.c)。
 * 门 = 温 0 输出与直发路逐字节同(speed-bench/d1_kv_ring_gate.sh 那套 cmp)。
 *
 * 出错会怎样: 捕获期间任何同步调用都让捕获作废(ThreadLocal 模式), capture_end 报 NULL —— 那一步的核一个都没跑。
 * 这里把 graph 关掉、按直发重来这一步(捕获不推进任何主机状态, 重来是干净的), 之后整个会话直发,
 * 日志里有一行"[graph] 捕获作废"。它不是兜底: 直发就是引擎的原路, 图只是同一条路的另一种发法。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU

#define DGRAPH_BUCKET 1024u   /* 位置桶宽: 桶内各核的 grid 上限不变; 跨桶重捕获(几十 ms, 每 1024 token 摊一次) */

typedef struct {
    void *exec;                /* 当前桶的图实例; NULL = 还没捕获 */
    uint32_t lo, cap;          /* 图对位置 [lo, cap] 有效 */
    int32_t *slot;             /* pinned [2] = {token, 位置}: 图开头两个 memcpy 节点的源 */
    float *onehot;             /* pinned [HC]: pre_mix 的 one-hot(第 0 路), 图开头灌进 st->pre_mix */
    int32_t *next;             /* pinned [1]: 图末尾 argmax 的落点 */
    ds4_gpu_tensor *am;        /* 设备 argmax 落点 */
    uint32_t steps, captures;  /* 走图解了几步 / 捕获了几次(日志) */
    double t_prep, t_launch, t_sync, t_gap;   /* 步边界的账(秒, 累计): 起手+取行提交 / cudaGraphLaunch 主机耗时 / 等图 / 上一步 sync 返回→本步进来 */
    double t_last_sync, t_launched;   /* 上一步 sync 返回的时刻 / 本步 launch 返回的时刻 */
    int direct_pending; int32_t pending_tok;   /* 捕获失败那一步: launch 没发出去, wait 里按直发补跑 */
} decode_graph;

int g_ds4_v41_graph = 1;
void ds4_engine_v41_set_graph(int on) { g_ds4_v41_graph = on; }

void v41_graph_free(ds4_v41_state *st) {
    decode_graph *g = (decode_graph *)st->dgraph;
    if (!g) return;
    if (g->exec) ds4_gpu_decode_graph_free(g->exec);
    ds4_gpu_host_free(g->slot); ds4_gpu_host_free(g->onehot); ds4_gpu_host_free(g->next);
    if (g->am) ds4_gpu_tensor_free(g->am);
    if (g->steps) {
        const double wall = (g->t_gap + g->t_prep + g->t_launch + g->t_sync) / g->steps;   /* 稳态每步壁钟(不含首步直发与捕获) */
        fprintf(stderr, "ds4: [graph] 走图解了 %u 步, 捕获 %u 次; 每步主机: 上步 sync→本步进来 %.0f us, 起手+取行提交 %.0f us, "
                        "cudaGraphLaunch %.0f us, 等图 %.2f ms ⇒ ★稳态 %.2f ms/步 = %.2f t/s★\n", g->steps, g->captures,
                g->t_gap / g->steps * 1e6, g->t_prep / g->steps * 1e6, g->t_launch / g->steps * 1e6, g->t_sync / g->steps * 1e3,
                wall * 1e3, 1.0 / wall);
    }
    free(g); st->dgraph = NULL;
}

static bool dg_alloc(ds4_v41_state *st) {
    if (st->dgraph) return true;
    decode_graph *g = xmalloc(sizeof *g); memset(g, 0, sizeof *g);
    g->slot = ds4_gpu_host_alloc(2u * sizeof(int32_t));
    g->onehot = ds4_gpu_host_alloc((uint64_t)DS4_N_HC * sizeof(float));
    g->next = ds4_gpu_host_alloc(sizeof(int32_t));
    g->am = ds4_gpu_tensor_alloc(16);
    st->dgraph = g;
    if (!g->slot || !g->onehot || !g->next || !g->am) { v41_graph_free(st); return false; }
    for (uint32_t c = 0; c < DS4_N_HC; c++) g->onehot[c] = c == 0 ? 1.0f : 0.0f;
    return true;
}

/* 能走图的条件: 标志开、主路(不是草稿塔)、暖过一步直发(懒分配全建好了)、没有要读回主机的探针/钩子/夹具。
 * 投机路不进图(它一轮两个 step, 形状随接受数变), 调用方自己判。 */
bool v41_graph_ready(const ds4_v41_state *st) {
    return g_ds4_v41_graph && !st->draft && st->n_direct1 > 0 && !g_ds4_v41_prof && !g_ds4_v41_hook && !st->dump_prefix;
}

/* 主机计数按闭式推进一步。直发路里 ng_src/cpend 是 v41_compress_source 逐层算的(g0 + ng_new / rem), n=1 时
 * 恒等于 n_past/ratio 与 n_past%ratio —— graph 路核里按位置算, 主机只记同一个数。 */
static void dg_advance(ds4_v41_state *st) {
    st->n_past += 1u;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!g_ds4_v41.is_kv_source[il]) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (!ratio) continue;
        st->ng_src[il] = st->n_past / ratio;
        st->cpend[il] = ratio > 1u ? st->n_past % ratio : 0u;
    }
}

/* 一步的主机侧起手(直发与捕获共用): 状态字段照 v41_forward 的写法, hist 先落(engram 哈希要回看它) */
static void dg_begin_step(ds4_v41_state *st, int32_t tok) {
    st->n = 1; st->pos0 = st->n_past; st->idx_owner = -1; st->cand_owner = -1; st->idx_topk = 0; st->idx_ratio = 0;
    st->stop_early = 0; st->ced_skip = 0; st->egraph_err = 0;
    st->hist[st->pos0] = tok;
}

/* 捕获当前桶的图。捕获期间核不执行, 只记节点; 主机状态一个都不推进(n_past/计数都在 step 里按步推)。 */
static bool dg_capture(ds4_engine *e, ds4_v41_state *st) {
    decode_graph *g = (decode_graph *)st->dgraph;
    if (g->exec) { ds4_gpu_decode_graph_free(g->exec); g->exec = NULL; }
    g->lo = st->n_past;
    uint32_t cap = (st->n_past / DGRAPH_BUCKET + 1u) * DGRAPH_BUCKET - 1u;
    if (cap > st->ctx - 1u) cap = st->ctx - 1u;
    g->cap = cap;
    st->graph = 1; st->graph_pos_lo = g->lo; st->graph_pos_cap = cap; st->egraph_uploaded = 0;
    /* 捕获态下不许分配: 注意力的局部件暂存先按段数上限长够(其余暂存在暖身那一步已按 n=1 的尺寸建好) */
    if (!ds4_gpu_v41_attn_scratch_prepare(DS4_N_HEAD, DS4_N_HEAD_DIM)) { st->graph = 0; return false; }
    if (!ds4_gpu_decode_graph_capture_begin()) { st->graph = 0; return false; }
    /* ★槽走零拷贝小核, 不走 memcpy 节点★(nsys 实撞: GB10 上图里每个 memcpy 节点 ~170 µs, 四个就是 0.69 ms/步) */
    bool ok = ds4_gpu_tensor_write_zerocopy(st->tok, 0, g->slot, sizeof(int32_t)) &&
              ds4_gpu_tensor_write_zerocopy(st->pos, 0, g->slot + 1, sizeof(int32_t)) &&
              ds4_gpu_tensor_write_zerocopy(st->pre_mix, 0, g->onehot, (uint64_t)DS4_N_HC * sizeof(float));
    if (ok) ok = v41_forward_body(e, st);
    if (ok) ok = ds4_gpu_v41_argmax_tensor(g->am, st->logits, 0, DS4_N_VOCAB) &&
                 ds4_gpu_tensor_read_zerocopy(g->next, g->am, 0, sizeof(int32_t));
    st->graph = 0;
    void *exec = ds4_gpu_decode_graph_capture_end();   /* 不管 ok 与否都要收捕获, 否则流一直停在捕获态 */
    if (!ok || !exec) { if (exec) ds4_gpu_decode_graph_free(exec); return false; }
    g->exec = exec;
    g->captures++;
    fprintf(stderr, "ds4: [graph] 位置桶 [%u, %u]\n", g->lo, g->cap);
    return true;
}

/* 直发解一步(捕获作废时的重来路, 与 core_v41_api.c 原来的单 token 环同一套调用) */
static bool dg_direct_step(ds4_engine *e, ds4_v41_state *st, int32_t tok, int32_t *next_tok) {
    decode_graph *g = (decode_graph *)st->dgraph;
    if (!v41_forward(e, st, &tok, 1u)) return false;
    return ds4_gpu_v41_argmax_tensor(g->am, st->logits, 0, DS4_N_VOCAB) && ds4_gpu_synchronize() &&
           ds4_gpu_tensor_read(g->am, 0, next_tok, sizeof(int32_t));
}

/* 一步拆成"发"与"等"两半(2026-09-18 主机侧分账): 调用方在 launch 与 wait 之间去 emit 当前 token ——
 * emit(文本 fwrite+fflush 到文件)实测 300 µs, 夹在两步之间就是 GPU 干等 300 µs; 放到图跑着的时候做, 白赚。
 * launch 之后、wait 之前**不许**碰 st 的位置状态(图在读槽); 取下一个 token 只能在 wait 之后。 */
bool v41_graph_launch(ds4_engine *e, ds4_v41_state *st, int32_t tok) {
    if (!dg_alloc(st)) return false;
    decode_graph *g = (decode_graph *)st->dgraph;
    if (st->n_past + 1u > st->ctx) { fprintf(stderr, "ds4: V4.1 上下文满(%u+1 > %u)\n", st->n_past, st->ctx); return false; }
    const double t0 = now_sec();
    if (g->t_last_sync > 0.0) g->t_gap += t0 - g->t_last_sync;
    dg_begin_step(st, tok);
    g->slot[0] = tok; g->slot[1] = (int32_t)st->pos0;
    __sync_synchronize();   /* 槽先落内存再发图: 图开头的零拷贝小核读的是内存里的值 */
    /* 取行任务先提交: 图里 engram 层前的 host 节点等的就是这一轮 */
    if (!st->no_engram && !v41_engram_prefetch(e, st)) return false;
    if (!g->exec || st->pos0 < g->lo || st->pos0 > g->cap) {
        if (!dg_capture(e, st)) {
            fprintf(stderr, "ds4: ★[graph] 捕获失败, 这一步与之后全部改走直发★\n");
            g_ds4_v41_graph = 0;
            g->direct_pending = 1;   /* wait 里按直发把这一步跑完 */
            g->pending_tok = tok;
            return true;
        }
    }
    const double t1 = now_sec();
    st->eg_t_launch = t1;
    if (!st->no_engram && !v41_engram_graph_arm(st)) return false;   /* 本步序号进 want 槽, 图里的自旋核等它 */
    if (!ds4_gpu_decode_graph_launch(g->exec)) return false;
    g->t_launched = now_sec();
    g->t_prep += t1 - t0; g->t_launch += g->t_launched - t1;
    return true;
}

bool v41_graph_wait(ds4_engine *e, ds4_v41_state *st, int32_t *next_tok) {
    decode_graph *g = (decode_graph *)st->dgraph;
    if (!g) return false;
    if (g->direct_pending) { g->direct_pending = 0; return dg_direct_step(e, st, g->pending_tok, next_tok); }
    /* 图跑着的时候主机在这儿收 engram 的 pread、逐层置位(GPU 到 engram 层前自旋等它); 收不到就不置位, GPU 超时放行后按 err 停车 */
    if (!st->no_engram && !v41_engram_graph_serve(st)) st->egraph_err = 1;
    if (!ds4_gpu_synchronize()) return false;
    const double t3 = now_sec();
    g->t_sync += t3 - g->t_launched; g->t_last_sync = t3;
    __sync_synchronize();
    if (st->egraph_err || (!st->no_engram && v41_engram_graph_err(st))) { fprintf(stderr, "ds4: [graph] engram 取行失败(位置 %u)\n", st->pos0); return false; }
    *next_tok = g->next[0];
    dg_advance(st);
    g->steps++;
    return true;
}

bool v41_graph_step(ds4_engine *e, ds4_v41_state *st, int32_t tok, int32_t *next_tok) {
    return v41_graph_launch(e, st, tok) && v41_graph_wait(e, st, next_tok);
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_decode_graph_nonempty_tu;
