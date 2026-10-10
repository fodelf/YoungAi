/* core_decode_graph.c — 解码整步 CUDA graph 的编排(2026-09-18, fable5 09-18 立案; 2026-09-22 扩到投机验证批)。
 *
 * 说人话: 解码一步 = 40 层 ~1500 发核。直发时每发都要主机敲一次门、GPU 核间也各排一次空, 12k 尺实测吃 4.5 ms/步
 * (整步 45.7 ms 的一成), 步边界还有三次同步 cudaMemcpy。这里把一整步录成一张 CUDA graph, 之后每步只做三件事:
 * 往 pinned 槽写 {token, 位置} → 一发 cudaGraphLaunch → 等完读回设备 argmax 出的下一个 token。
 *
 * ★投机验证批也进图(2026-09-22)★: 一轮验证 n = 1+k 行(k ≤ 块长 5), 以前恒走直发 —— 一轮里验证批那一发比走图贵 4 ms
 * (09-19 实测直发验 1 行 42.5 ms 对走图一步 38.5), 加上一轮两次同步。现在按批大小 n 各捕一张图(形状随 n 变, 每个 n 一次),
 * 图里: 零拷贝灌 n 个 token/位置 → 整步前向(核按"设备位置"口径逐行算位置 pos0+i) → n 行各一发 argmax(采样开: 一发采样核, 2026-09-28) → 零拷贝读回。
 * 回滚要的快照也在图里(窗口环: 每层 commit 前存那 n 格; 压缩器余行: 追加核顺手存), 主机只在发图时记账(snap_n/snap_past/snap_cpend),
 * 部分接受之后的还原仍是 v41_spec_rollback 那几发直发(形状随接受数变, 不进图)。
 *
 * 图为什么能只捕一次: 所有随位置变的东西都不烤进图 ——
 *   ①位置: 核从 st->pos(图开头由零拷贝小核从 pinned 槽灌进去)读, 见 ds4_gpu_v41.h "设备位置"口径;
 *   ②grid/shared: 按位置桶(DGRAPH_BUCKET 个位置一桶)的上限开, 跨桶重捕获;
 *   ③engram 的盘读: 图里 engram 层前放自旋小核等线程池 pread 完 + 零拷贝小核从常驻 pinned 行缓冲上传;
 *   ④压缩源层"凑满一组才池化": 核里按位置判, 没凑满写垃圾槽(core_v41_attn.c)。
 * 门 = 温 0 输出与直发路逐字节同(speed-bench/d1_kv_ring_gate.sh 那套 cmp; 投机路 = 投机 == 纯解码 逐字节同)。
 *
 * 出错会怎样: 捕获期间任何同步调用都让捕获作废(ThreadLocal 模式), capture_end 报 NULL —— 那一步的核一个都没跑。
 * n=1 时把 graph 关掉、按直发重来这一步(捕获不推进任何主机状态, 重来是干净的), 之后这条请求(本状态)直发 —— 不关进程级开关, 否则服务端一次失败会拖累之后所有请求; 验证批捕获失败则
 * 这一批与之后的批都走直发(纯解码图照走)。日志里有一行"[graph] 捕获失败/作废"。它不是兜底: 直发就是引擎的原路, 图只是同一条路的另一种发法。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU

#define DGRAPH_BUCKET 1024u   /* 位置桶宽: 桶内各核的 grid 上限不变; 跨桶重捕获(几十 ms, 每 1024 token 摊一次) */
#define DGRAPH_NMAX (DS4_MTP_MAX_BLOCK + 2u)   /* 图按批大小 n 索引: [1] 纯解码; [2..1+块长] 投机验证批 */

typedef struct {
    void *exec;                /* 这个 n 的图实例; NULL = 还没捕获 */
    uint32_t lo, cap;          /* 图对本批首行位置 [lo, cap] 有效(末行到 cap+n−1) */
    uint64_t gen;              /* 捕获时的后端暂存代号(见 ds4_gpu_v41_scratch_generation) */
    uint32_t idx_gen;          /* 捕获时的索引草稿代号(st->iscap_gen) */
} dg_inst;
typedef struct {
    dg_inst g[DGRAPH_NMAX];
    int32_t *tokv, *posv;      /* pinned [NMAX] = 本批的 token / 位置: 图开头零拷贝小核的源 */
    float *onehot;             /* pinned [NMAX][HC]: pre_mix 的 one-hot(第 0 路), 图开头灌进 st->pre_mix */
    int32_t *next;             /* pinned [NMAX·4]: 第 i 行 argmax 的落点在 next[4i](设备落点每行 16 B, 整块零拷贝读回) */
    ds4_gpu_tensor *am;        /* 设备 argmax 落点 [NMAX][16 B] */
    ds4_gpu_tensor *amv[DGRAPH_NMAX];   /* 逐行视图(argmax 核只写自己那块的第 0 个 int) */
    uint32_t steps, bsteps, captures, regrow;   /* 走图解了几步(n=1 纯解码步) / 验证批走图几轮 / 捕获几次 / 因暂存换指针而重捕获几次 */
    uint32_t k0steps;          /* 草稿白跑(k=0)后走 n=1 图的步(2026-09-28): 不进"稳态"账 —— 它前面挂着一轮草稿, 不是纯解码步 */
    int cur_pure;              /* 本步是纯解码步(进账)还是 k=0 轮的步(只计数) */
    double t_prep, t_launch, t_sync, t_gap;   /* 步边界的账(秒, 累计; 只记 n=1): 起手+取行提交 / cudaGraphLaunch 主机耗时 / 等图 / 上一步 sync 返回→本步进来 */
    double b_prep, b_launch; uint32_t b_n;    /* 同两项, 验证批(n≥2)与 k=0 轮的步(2026-10-07 主机空隙分账: nsys 量到草稿完→验证首核之间 GPU 空 1.2 ms) */
    double b_begin, b_engram, b_check, b_arm;  /* 起手拆四段: 写槽+hist / engram 取行提交 / 图代号校验(+重捕获) / 置位 */
    double t_capture; uint32_t n_capture;      /* 捕获(含 n=1)的主机耗时累计 / 次数 */
    double t_last_sync, t_launched;   /* 上一步 sync 返回的时刻 / 本步 launch 返回的时刻 */
    int direct_pending; int32_t pending_tok;   /* n=1 捕获失败那一步: launch 没发出去, wait 里按直发补跑 */
    uint32_t cur_n;            /* 本步发的是几行(wait 按它读回/推进) */
    int batch_off;             /* 验证批的图捕获失败过: 之后的批一律直发(只打一次日志) */
    int n1_off;                /* n=1 的图捕获失败过: 本状态(= 这一条请求)之后直发。★只关本状态★, 不碰进程级 g_ds4_v41_graph ——
                                * 服务端是常驻进程, 09-22 夜一次失败写了全局开关, 之后整夜所有请求都直发 */
} decode_graph;

int g_ds4_v41_graph = 1;
void ds4_engine_v41_set_graph(int on) { g_ds4_v41_graph = on; }

void v41_graph_free(ds4_v41_state *st) {
    decode_graph *g = (decode_graph *)st->dgraph;
    if (!g) return;
    for (uint32_t n = 0; n < DGRAPH_NMAX; n++) {
        if (g->g[n].exec) ds4_gpu_decode_graph_free(g->g[n].exec);
        if (g->amv[n]) ds4_gpu_tensor_free(g->amv[n]);
    }
    ds4_gpu_host_free(g->tokv); ds4_gpu_host_free(g->posv); ds4_gpu_host_free(g->onehot); ds4_gpu_host_free(g->next);
    if (g->am) ds4_gpu_tensor_free(g->am);
    if (g->steps) {
        const double wall = (g->t_gap + g->t_prep + g->t_launch + g->t_sync) / g->steps;   /* 稳态每步壁钟(不含首步直发与捕获) */
        fprintf(stderr, "ds4: [graph] 走图解了 %u 步, 捕获 %u 次(其中暂存换指针重捕 %u 次); 每步主机: 上步 sync→本步进来 %.0f us, 起手+取行提交 %.0f us, "
                        "cudaGraphLaunch %.0f us, 等图 %.2f ms ⇒ ★稳态 %.2f ms/步 = %.2f t/s★\n", g->steps, g->captures, g->regrow,
                g->t_gap / g->steps * 1e6, g->t_prep / g->steps * 1e6, g->t_launch / g->steps * 1e6, g->t_sync / g->steps * 1e3,
                wall * 1e3, 1.0 / wall);
    }
    if (g->bsteps || g->k0steps) fprintf(stderr, "ds4: [graph] 投机验证批走图 %u 轮, 草稿白跑(k=0)后走 n=1 图 %u 步(不进稳态账)\n", g->bsteps, g->k0steps);
    free(g); st->dgraph = NULL;
}

static bool dg_alloc(ds4_v41_state *st) {
    if (st->dgraph) return true;
    decode_graph *g = xmalloc(sizeof *g); memset(g, 0, sizeof *g);
    g->tokv = ds4_gpu_host_alloc((uint64_t)DGRAPH_NMAX * sizeof(int32_t));
    g->posv = ds4_gpu_host_alloc((uint64_t)DGRAPH_NMAX * sizeof(int32_t));
    g->onehot = ds4_gpu_host_alloc((uint64_t)DGRAPH_NMAX * DS4_N_HC * sizeof(float));
    g->next = ds4_gpu_host_alloc((uint64_t)DGRAPH_NMAX * 4u * sizeof(int32_t));
    g->am = ds4_gpu_tensor_alloc((uint64_t)DGRAPH_NMAX * 16u);
    st->dgraph = g;
    if (!g->tokv || !g->posv || !g->onehot || !g->next || !g->am) { v41_graph_free(st); return false; }
    for (uint32_t n = 0; n < DGRAPH_NMAX; n++) {
        g->amv[n] = ds4_gpu_tensor_view(g->am, (uint64_t)n * 16u, 16u);
        if (!g->amv[n]) { v41_graph_free(st); return false; }
    }
    for (uint32_t i = 0; i < DGRAPH_NMAX * DS4_N_HC; i++) g->onehot[i] = (i % DS4_N_HC) == 0 ? 1.0f : 0.0f;
    return true;
}

/* 能走图的条件: 标志开、主路(不是草稿塔)、暖过一步直发(懒分配全建好了)、没有要读回主机的探针/钩子/夹具。 */
bool v41_graph_ready(const ds4_v41_state *st) {
    const decode_graph *g = (const decode_graph *)st->dgraph;
    return g_ds4_v41_graph && !st->draft && st->n_direct1 > 0 && !g_ds4_v41_prof && !g_ds4_v41_hook && !st->dump_prefix &&
           !(g && g->n1_off);
}
/* 草稿图用的"主路允许走图"(2026-10-07 实撞): 以前草稿图拿 v41_graph_ready 判, 它要求 n=1 直发暖过一步 —— 投机每轮 k≥1 时 n=1 一步都不跑,
 * 草稿图就永远开不了, 每轮 ~200 发核直发(nsys: 每发 3~4 µs 空隙 ≈ 0.8 ms/轮), 日志里一行"草稿图已捕获"都没有。验证批图 09-22 已按同样理由
 * 去掉了 n=1 暖过的要求(v41_graph_batch_ready), 草稿图漏了。草稿图自己的暖身由 dr->gwarm[rows] 守。 */
bool v41_graph_allowed(const ds4_v41_state *st) {
    const decode_graph *g = (const decode_graph *)st->dgraph;
    return g_ds4_v41_graph && !st->draft && !g_ds4_v41_prof && !g_ds4_v41_hook && !st->dump_prefix && !(g && g->n1_off);
}
/* 验证批同上, 且这个 n 直发暖过(每个 n 的暂存/核属性各自懒建)、这个批的快照缓冲建过(直发那一轮 v41_spec_snapshot 建) */
bool v41_graph_batch_ready(const ds4_v41_state *st, uint32_t n) {
    const decode_graph *g = (const decode_graph *)st->dgraph;
    /* 不要求 n=1 暖过: 钉死 k 的趟(--dspark-verify)每一步都是投机轮, n=1 一次都不跑, 按 v41_graph_ready 判就永远开不了批图(09-22 实撞) */
    return g_ds4_v41_graph && !st->draft && !g_ds4_v41_prof && !g_ds4_v41_hook && !st->dump_prefix &&
           n >= 2u && n < DGRAPH_NMAX && st->n_direct_n[n] > 0 && st->snap_win[0] && !(g && g->batch_off);
}

/* 主机计数按闭式推进 n 步。直发路里 ng_src/cpend 是 v41_compress_source 逐层算的(g0 + ng_new / rem), 闭式就是
 * n_past/ratio 与 n_past%ratio —— graph 路核里按位置算, 主机只记同一个数。mainh 环的账与 v41_forward 收尾同式。 */
static void dg_advance(ds4_v41_state *st, uint32_t n) {
    const uint32_t pos0 = st->n_past;
    st->n_past += n;
    if (st->mainh) {   /* 图里 hc_mean 核按设备位置把这 n 行的 main_hidden 落进了环, 主机只记账(同 v41_forward 的规则) */
        const uint32_t rows = n < st->mainh_cap ? n : st->mainh_cap;
        const int64_t first = (int64_t)pos0 + (int64_t)(n - rows);
        const bool contig = st->mainh_n > 0 && st->mainh_end == first - 1;
        const uint32_t nn = contig ? st->mainh_n + rows : rows;
        st->mainh_n = nn > st->mainh_cap ? st->mainh_cap : nn;
        st->mainh_end = (int64_t)pos0 + (int64_t)n - 1;
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!g_ds4_v41.is_kv_source[il]) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (!ratio) continue;
        st->ng_src[il] = st->n_past / ratio;
        st->cpend[il] = ratio > 1u ? st->n_past % ratio : 0u;
    }
}

/* 一步的主机侧起手(直发与捕获共用): 状态字段照 v41_forward 的写法, hist 先落(engram 哈希要回看它) */
static void dg_begin_step(ds4_v41_state *st, const int32_t *ids, uint32_t n) {
    st->n = n; st->pos0 = st->n_past; st->idx_owner = -1; st->cand_owner = -1; st->idx_topk = 0; st->idx_ratio = 0;
    st->stop_early = 0; st->ced_skip = 0; st->egraph_err = 0;
    memcpy(st->hist + st->pos0, ids, (size_t)n * 4);
}

/* 捕获这个 n 的图(当前桶)。捕获期间核不执行, 只记节点; 主机状态一个都不推进(n_past/计数都在 wait 里按步推)。 */
static bool dg_capture(ds4_engine *e, ds4_v41_state *st, uint32_t n) {
    decode_graph *g = (decode_graph *)st->dgraph;
    dg_inst *in = &g->g[n];
    if (in->exec) { ds4_gpu_decode_graph_free(in->exec); in->exec = NULL; }
    in->lo = st->n_past;
    uint32_t cap = (st->n_past / DGRAPH_BUCKET + 1u) * DGRAPH_BUCKET - 1u;   /* 桶上限 = 本批首行位置的上限, 末行到 cap+n−1 */
    if (cap > st->ctx - n) cap = st->ctx - n;
    in->cap = cap;
    /* ★三个暂存先长够, 之后才置 st->graph★(2026-09-23 实撞): 以前先置 st->graph=1 再长索引草稿, 而
     * v41_index_scratch_prepare 见 st->graph 就当"捕获态分配"拒掉 ⇒ 只要新桶要扩草稿捕获必失败。服务端第一条 22 token
     * 冒烟就撞上, 整夜直发(每步 +3 ms)。CLI 门的长提示预填翻倍长出的余量碰巧够, 所以没暴露。 */
    /* 注意力的局部件暂存按 段数上限 × n 行 长够(其余暂存在暖身那一步已按这个 n 的尺寸建好) */
    if (!ds4_gpu_v41_attn_scratch_prepare(n, DS4_N_HEAD, DS4_N_HEAD_DIM)) return false;
    /* 索引打分草稿: 桶里最靠后那一批、压缩比最小那个 indexer 源层的组数最多, 按它长够。
     * 遍历口径必须与消费方 v41_index_source(core_v41_attn.c, 按 is_index_source 层自己的压缩比)同源 ——
     * 只看 kv 源层会漏掉压缩比更小的纯 indexer 源层, 捕获时照样要扩容。 */
    uint32_t ng_max = 0;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!g_ds4_v41.is_index_source[il]) continue;
        const uint32_t r = ds4_layer_compress_ratio(il);
        if (r && (cap + n) / r > ng_max) ng_max = (cap + n) / r;
    }
    if (ng_max && !v41_index_scratch_prepare(st, ng_max, n, n)) return false;
    /* 候选块暂存: 按**桶上限**那一批的组数算块数(桶里位置越靠后组越多, 捕获时就得按最大的开), n 行各一份 */
    if (g_ds4_v41.candidate_source_layer >= 0 && g_ds4_v41.candidate_block_size > 0) {
        const uint32_t cr = ds4_layer_compress_ratio((uint32_t)g_ds4_v41.candidate_source_layer);
        const uint32_t cbs = (uint32_t)g_ds4_v41.candidate_block_size;
        const uint32_t cng = cr ? (cap + n) / cr : 0u;
        if (cng && !ds4_gpu_v41_candidate_scratch_prepare(n, (cng + cbs - 1u) / cbs)) return false;
    }
    if (!ds4_gpu_v41_indexer_scratch_prepare(n, DS4_N_INDEXER_HEAD)) return false;   /* --idx-mma 的 q 整数尾数暂存(开关关着 = no-op) */
    st->graph = 1; st->graph_pos_lo = in->lo; st->graph_pos_cap = cap; st->egraph_uploaded = 0;
    if (!ds4_gpu_decode_graph_capture_begin()) { st->graph = 0; return false; }
    /* ★槽走零拷贝小核, 不走 memcpy 节点★(nsys 实撞: GB10 上图里每个 memcpy 节点 ~170 µs, 四个就是 0.69 ms/步) */
    bool ok = ds4_gpu_tensor_write_zerocopy(st->tok, 0, g->tokv, (uint64_t)n * sizeof(int32_t)) &&
              ds4_gpu_tensor_write_zerocopy(st->pos, 0, g->posv, (uint64_t)n * sizeof(int32_t)) &&
              ds4_gpu_tensor_write_zerocopy(st->pre_mix, 0, g->onehot, (uint64_t)n * DS4_N_HC * sizeof(float));
    if (ok) ok = v41_forward_body(e, st);
    /* 末尾: 采样开 ⇒ 一发采样核出 n 行(全分布样本 / 草稿接受位 / 残差样本, 每行 16 B; 参数烤进图 = 每请求常量);
     * 否则逐行 argmax(只写槽的第 0 个 int)。两种都落同一块 g->am, 零拷贝整块读回。 */
    if (ok && st->dev_sample) ok = ds4_gpu_v41_sample_tensor(g->am, st->logits, 0u, n, DS4_N_VOCAB, st->pos, st->tok, &st->samp,
                                                             n > 1u ? st->spec_q : NULL) != 0;
    else for (uint32_t i = 0; ok && i < n; i++) ok = ds4_gpu_v41_argmax_tensor(g->amv[i], st->logits, i, DS4_N_VOCAB) != 0;
    if (ok) ok = ds4_gpu_tensor_read_zerocopy(g->next, g->am, 0, (uint64_t)n * 16u) != 0;
    st->graph = 0;
    void *exec = ds4_gpu_decode_graph_capture_end();   /* 不管 ok 与否都要收捕获, 否则流一直停在捕获态 */
    if (!ok || !exec) { if (exec) ds4_gpu_decode_graph_free(exec); return false; }
    in->exec = exec;
    g->captures++;
    in->gen = ds4_gpu_v41_scratch_generation();   /* 捕获收完再记: 捕获前的 attn_scratch_prepare 自己就可能长一次 */
    in->idx_gen = st->iscap_gen;                  /* 同上: 上面那次 index_scratch_prepare 可能换了 iscore/cand 的指针 */
    fprintf(stderr, "ds4: [graph] %u 行的图: 位置桶 [%u, %u]\n", n, in->lo, in->cap);
    return true;
}

/* 直发解一步(n=1 捕获作废时的重来路, 与 core_v41_api.c 原来的单 token 环同一套调用) */
static bool dg_direct_step(ds4_engine *e, ds4_v41_state *st, int32_t tok, int32_t *next_tok) {
    decode_graph *g = (decode_graph *)st->dgraph;
    if (!v41_forward(e, st, &tok, 1u)) return false;
    return v41_device_next(st, g->am, 0u, 1u, NULL, next_tok);
}

/* 暂存换过指针 ⇒ 所有图作废(它们烤死的都是捕获那一刻的指针) */
static void dg_invalidate(decode_graph *g, const char *why) {
    int any = 0;
    for (uint32_t n = 0; n < DGRAPH_NMAX; n++) if (g->g[n].exec) { ds4_gpu_decode_graph_free(g->g[n].exec); g->g[n].exec = NULL; any = 1; }
    if (any) { fprintf(stderr, "ds4: [graph] %s, 图作废, 重捕获\n", why); g->regrow++; }
}

/* 一步拆成"发"与"等"两半(2026-09-18 主机侧分账): 调用方在 launch 与 wait 之间去 emit 当前 token ——
 * emit(文本 fwrite+fflush 到文件)实测 300 µs, 夹在两步之间就是 GPU 干等 300 µs; 放到图跑着的时候做, 白赚。
 * launch 之后、wait 之前**不许**碰 st 的位置状态(图在读槽); 取下一个 token 只能在 wait 之后。 */
static bool dg_launch(ds4_engine *e, ds4_v41_state *st, const int32_t *ids, uint32_t n, int pure) {
    if (!dg_alloc(st)) return false;
    decode_graph *g = (decode_graph *)st->dgraph;
    dg_inst *in = &g->g[n];
    if (st->n_past + n > st->ctx) { fprintf(stderr, "ds4: V4.1 上下文满(%u+%u > %u)\n", st->n_past, n, st->ctx); return false; }
    const double t0 = now_sec();
    g->cur_pure = n == 1u && pure;
    /* 步边界的账只记"上一步也是纯解码步"的间隔: 中间隔着验证批或草稿的, 上一步 sync 时已把 t_last_sync 清零 */
    if (g->cur_pure && g->t_last_sync > 0.0) g->t_gap += t0 - g->t_last_sync;
    dg_begin_step(st, ids, n);
    for (uint32_t i = 0; i < n; i++) { g->tokv[i] = ids[i]; g->posv[i] = (int32_t)(st->pos0 + i); }
    __sync_synchronize();   /* 槽先落内存再发图: 图开头的零拷贝小核读的是内存里的值 */
    const double tp0 = now_sec();
    /* 取行任务先提交: 图里 engram 层前的自旋核等的就是这一轮 */
    if (!st->no_engram && !v41_engram_prefetch(e, st)) return false;
    const double tp1 = now_sec();
    /* ★暂存换过指针 ⇒ 图作废, 重捕获★(2026-09-19): 投机验证批(直发, n≤6)让 attn/hc/VQ 的暂存扩容, 图里烤死的旧指针
     * 指向已释放页 —— 这就是 09-18 第三版"歇轮走图"2K 崩 illegal memory access 的真因(定罪见 cuda_v41_1.inc.cu v41_grow)。
     * 代号是全局的: 变了所有 n 的图一起作废。 */
    if (in->exec && in->gen != ds4_gpu_v41_scratch_generation()) dg_invalidate(g, "后端暂存换过指针");
    /* 同一个坑的另一半: 直发路(投机验证批/捕获失败重来)可能让索引草稿翻倍, iscore/cand 换了指针 */
    if (in->exec && in->idx_gen != st->iscap_gen) dg_invalidate(g, "索引草稿长过");
    if (!in->exec || st->pos0 < in->lo || st->pos0 + n - 1u > in->cap) {
        const double tc0 = now_sec();
        const bool cap_ok = dg_capture(e, st, n);
        g->t_capture += now_sec() - tc0; g->n_capture++;   /* 捕获的主机耗时单记: 短跑里它把"起手"均值撑大(10-07 实撞: 4 次捕获摊成 1 ms/轮) */
        if (!cap_ok) {
            if (n == 1u) {
                fprintf(stderr, "ds4: ★[graph] 捕获失败, 本条请求这一步与之后改走直发★\n");
                g->n1_off = 1;
                g->direct_pending = 1;   /* wait 里按直发把这一步跑完 */
                g->pending_tok = ids[0]; g->cur_n = 1u;
                return true;
            }
            fprintf(stderr, "ds4: ★[graph] %u 行验证批捕获失败, 这一批与之后的验证批走直发(纯解码图照走)★\n", n);
            g->batch_off = 1;
            return false;   /* 没发出去, 主机状态由调用方的直发路重新起手 */
        }
    }
    const double t1 = now_sec();
    st->eg_t_launch = t1;
    if (!st->no_engram && !v41_engram_graph_arm(st)) return false;   /* 本步序号进 want 槽, 图里的自旋核等它 */
    const double tp2 = now_sec();
    if (!ds4_gpu_decode_graph_launch(in->exec)) return false;
    g->t_launched = now_sec();
    if (g->cur_pure) { g->t_prep += t1 - t0; g->t_launch += g->t_launched - t1; }
    else { g->b_prep += tp2 - t0; g->b_launch += g->t_launched - tp2; g->b_n++;
           g->b_begin += tp0 - t0; g->b_engram += tp1 - tp0; g->b_check += t1 - tp1; g->b_arm += tp2 - t1; }
    g->cur_n = n;
    return true;
}
/* 验证批/k=0 步的发图主机账(累计秒): 起手(写槽 + 取行提交 + 图校验) / cudaGraphLaunch 本身 / 几发。给 core_v41_api 的 --v41-prof 分账行 */
void v41_graph_batch_host(const ds4_v41_state *st, double *prep, double *launch, uint32_t *n) {
    const decode_graph *g = (const decode_graph *)st->dgraph;
    *prep = g ? g->b_prep : 0.0; *launch = g ? g->b_launch : 0.0; *n = g ? g->b_n : 0u;
}
void v41_graph_batch_prep(const ds4_v41_state *st, double *t_begin, double *t_engram, double *t_check, double *t_arm) {
    const decode_graph *g = (const decode_graph *)st->dgraph;
    *t_begin = g ? g->b_begin : 0.0; *t_engram = g ? g->b_engram : 0.0; *t_check = g ? g->b_check : 0.0; *t_arm = g ? g->b_arm : 0.0;
}
void v41_graph_capture_cost(const ds4_v41_state *st, double *t_capture, uint32_t *n_capture) {
    const decode_graph *g = (const decode_graph *)st->dgraph;
    *t_capture = g ? g->t_capture : 0.0; *n_capture = g ? g->n_capture : 0u;
}

static bool dg_wait(ds4_engine *e, ds4_v41_state *st, int32_t *next) {
    decode_graph *g = (decode_graph *)st->dgraph;
    if (!g) return false;
    if (g->direct_pending) { g->direct_pending = 0; return dg_direct_step(e, st, g->pending_tok, next); }
    const uint32_t n = g->cur_n;
    /* 图跑着的时候主机在这儿收 engram 的 pread、逐层置位(GPU 到 engram 层前自旋等它); 收不到就不置位, GPU 超时放行后按 err 停车 */
    if (!st->no_engram && !v41_engram_graph_serve(st)) st->egraph_err = 1;
    if (!ds4_gpu_synchronize()) return false;
    const double t3 = now_sec();
    if (g->cur_pure) { g->t_sync += t3 - g->t_launched; g->t_last_sync = t3; }
    else g->t_last_sync = 0.0;   /* 验证批 / k=0 轮的步之后, 下一个纯解码步的"上步→本步"间隔不算(中间不是纯解码) */
    __sync_synchronize();
    if (st->egraph_err || (!st->no_engram && v41_engram_graph_err(st))) { fprintf(stderr, "ds4: [graph] engram 取行失败(位置 %u)\n", st->pos0); return false; }
    /* 每行 4 个 int: 采样路按"接受 ⇒ 草稿 / 拒绝 ⇒ 残差"拼, argmax 路取第 0 个(core_v41_sample.c) */
    v41_sample_pick(st, g->next, g->tokv, n, next);
    dg_advance(st, n);
    if (n != 1u) g->bsteps++; else if (g->cur_pure) g->steps++; else g->k0steps++;
    return true;
}

bool v41_graph_launch(ds4_engine *e, ds4_v41_state *st, int32_t tok, int after_draft) { return dg_launch(e, st, &tok, 1u, !after_draft); }
bool v41_graph_wait(ds4_engine *e, ds4_v41_state *st, int32_t *next_tok) { return dg_wait(e, st, next_tok); }
bool v41_graph_step(ds4_engine *e, ds4_v41_state *st, int32_t tok, int32_t *next_tok) {
    return v41_graph_launch(e, st, tok, 0) && v41_graph_wait(e, st, next_tok);
}

/* 验证批: 发图 + 快照记账。快照的字节在图里(窗口环按层 / 压缩器余行在追加核里), 这里只记"这一批从哪起、几行、各源层当时的余行数",
 * 与直发路 v41_spec_snapshot + v41_compress_source 记的同一套字段 ⇒ v41_spec_rollback 一个字不改。 */
bool v41_graph_batch_launch(ds4_engine *e, ds4_v41_state *st, const int32_t *ids, uint32_t n) {
    if (n < 2u || n >= DGRAPH_NMAX) return false;
    if (!dg_launch(e, st, ids, n, 0)) return false;
    st->snap_n = n; st->snap_past = st->pos0; st->snap_on = 1u;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!g_ds4_v41.is_kv_source[il]) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        if (ratio > 1u) st->snap_cpend[il] = st->pos0 % ratio;
    }
    return true;
}
bool v41_graph_batch_wait(ds4_engine *e, ds4_v41_state *st, int32_t *next) { return dg_wait(e, st, next); }
#endif /* !DS4_NO_GPU */
typedef int ds4_core_decode_graph_nonempty_tu;
