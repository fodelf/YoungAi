/* core_v41_engram.c — DeepSeek V4.1 engram 层(2026-09-12 战役 P2b), 逐式对照官方 engram.py NgramHashState.forward
 * 与 model.py Engram.forward。
 *
 * 链路: 压缩 id(token_map) → 每位置 4-gram 滚动 XOR 哈希(乘子/素数/偏移来自 GGUF 常量张量, 转换器从 tokenizer
 * 算好的) → 24 行(3 种 n-gram × 8 头) → 从 203 GB 表(原 HF 分片, 在盘)并行 pread 原始行字节(256 B fp8 + 8 B ue8m0)
 * → 上 GPU dequant 成 bf16 格点 → wkv(f16 [25600][6144]) → key(4 路)|value → 门(CUDA 核) → hc 就地更新。
 * 取行(P4): 两个 engram 层的行在前向一开始就由 48 个线程一次性 pread(一线程一行, NVMe 吃并发), 与前面几层的 GPU 算重叠;
 * 到 engram 层只收结果。第一版 mmap 逐行页错误串行: 512 token 两层吃 16.6 s/38 s; 第二版每层各自 16 线程: 解码 16 ms/层。
 * 回看的 3 个 token 可能在上一块 —— 取自 st->hist(整段 token 历史), 按绝对位置索引。 */
#include <fcntl.h>   /* posix_fadvise: 关 engram 表的内核预读, 见 v41_engram_open_shard */
#include "core_internal.h"
#ifndef DS4_NO_GPU

/* 池里的线程数与 O_DIRECT 对齐粒度的正本在 core_v41.h(epool 也要用) */

static const void *v41_tensor_host(const ds4_model *m, const ds4_tensor *t) { return tensor_data(m, t); }


/* 从 O_DIRECT 的 fd 读任意 [off, off+len) —— 对齐到块读进 bounce 再拷出来。
 * len 最大 = HD(256) ⇒ 跨块最多 2 块, bounce 给 2×ALIGN 就够。 */
static int v41_edio_pread(int fd, uint8_t *bounce, void *dst, uint64_t off, uint32_t len) {
    const uint64_t a = off & ~(uint64_t)(V41_EDIO_ALIGN - 1u);
    const uint64_t span = off + len - a;
    const uint32_t nb = (uint32_t)((span + V41_EDIO_ALIGN - 1u) / V41_EDIO_ALIGN) * V41_EDIO_ALIGN;
    if (pread(fd, bounce, nb, (off_t)a) != (ssize_t)nb) return 0;
    memcpy(dst, bounce + (off - a), len);
    return 1;
}

static bool v41_engram_open_shard(ds4_v41_state *st, uint32_t ei) {
    if (st->eshard[ei].fd >= 0) return true;
    const char *path = g_ds4_v41.engram_table_path[ei];
    /* ★O_DIRECT(2026-09-15)★: 表 203 GB, 每块 512 token 要 ~49k 次 264 B 随机读 = 200 MB 灌进页缓存,
     * 把引擎那 103 GiB 注册映射的模型页挤出去再回填 —— 09-12 定罪过的"回收压力下瞬时脏读"就是这么来的。
     * 实撞后果不是变慢而是**温度 0 下两次输出不同**(09-15: 开 engram 两跑不同, --v41-no-engram 两跑逐字同)。
     * 绕开页缓存 = 既不污染别人, 自己也不需要缓存(行号随机, 命中率本来就≈0)。
     * 拿不到 O_DIRECT(文件系统不支持)就退普通读 + FADV_RANDOM, 并把话说在日志里, 不装作没事。 */
    int fd = -1, dio = 1;
#ifdef O_DIRECT
    fd = open(path, O_RDONLY | O_DIRECT);
#endif
    if (fd < 0) { dio = 0; fd = open(path, O_RDONLY); }
    if (fd < 0) {   /* 最常见的原因: GGUF 里记的是转换那台机器的绝对路径, 换了机器就不在 */
        fprintf(stderr, "ds4: engram 表打不开 %s —— 用 --engram-dir 指向放官方分片(model-0004{7,8}-of-00048.safetensors)的目录\n", path);
        return false;
    }
    struct stat sb; if (fstat(fd, &sb) != 0) { close(fd); return false; }
    /* ★关预读(2026-09-15)★: 表是 203 GB 的 HF 分片, 每行只读 264 B 且行号随机, 而内核默认预读会把每次
     * 读放大成 128 KB 灌进页缓存 —— 512 token 一块、两层、24 行就是 GB 级的churn, 把模型的 103 GiB 映射页
     * 挤出去再回填。实撞的后果不是变慢而是**温度 0 下两次输出不同**(09-15: 开 engram 两跑不同, --v41-no-engram
     * 两跑逐字同; 与 09-12 定罪的"映射页回收压力下瞬时脏读"同一条链)。FADV_RANDOM 让内核按请求大小读。 */
    if (!dio) {
#ifdef POSIX_FADV_RANDOM
        (void)posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
#endif
        fprintf(stderr, "ds4: ★engram 表拿不到 O_DIRECT, 退普通读 —— 页缓存会被搅动, 输出可能不可复现★\n");
    }
    st->eshard[ei].dio = dio;
    st->eshard[ei].fd = fd; st->eshard[ei].size = (uint64_t)sb.st_size;
    return true;
}

/* 官方 NgramHashState.forward(无 image mask): tokens[shift] = compressed[p-shift](p-shift<0 → pad);
 * products[k] = tokens[k]·mult[layer][k]; rolling = products[0]; for i=1..G-1: rolling ^= products[i];
 * hash_i = rolling % primes[layer][i-1][head] + offsets[layer][(i-1)·heads+head]。p 是绝对位置。 */
static void v41_engram_hash(const ds4_engine *e, const ds4_v41_state *st, uint32_t ei, uint32_t p, int64_t *rows /*[cols]*/) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    const ds4_weights *W = &e->weights;
    const int32_t *tmap = (const int32_t *)v41_tensor_host(&e->model, W->engram_token_map);
    const int64_t *mult = (const int64_t *)v41_tensor_host(&e->model, W->engram_multipliers);
    const int64_t *prim = (const int64_t *)v41_tensor_host(&e->model, W->engram_primes);
    const int64_t *offs = (const int64_t *)v41_tensor_host(&e->model, W->engram_offsets);
    const uint32_t G = v->engram_max_ngram, H = v->engram_heads;
    int64_t prod[8];
    for (uint32_t k = 0; k < G; k++) {
        int64_t cid;
        if ((int64_t)p - (int64_t)k < 0) cid = (int64_t)v->engram_pad;
        else { int32_t tok = st->hist[p - k]; cid = (tok >= 0 && (uint32_t)tok < DS4_N_VOCAB) ? tmap[tok] : (int64_t)v->engram_pad; }
        prod[k] = (int64_t)((uint64_t)cid * (uint64_t)mult[(uint64_t)ei * G + k]);   /* 官方 int64 乘(乘子有界不溢出) */
    }
    int64_t rolling = prod[0];
    for (uint32_t i = 1; i < G; i++) {
        rolling ^= prod[i];
        for (uint32_t h = 0; h < H; h++) {
            const int64_t pr = prim[((uint64_t)ei * (G - 1) + (i - 1)) * H + h];
            int64_t r = rolling % pr; if (r < 0) r += pr;          /* torch 对非负数 %, 这里保险 */
            rows[(i - 1) * H + h] = r + offs[(uint64_t)ei * (G - 1) * H + (i - 1) * H + h];
        }
    }
}

/* 取行任务的工作单元 = (engram 层 ei, 位置 p, 列 c) 一行; 线程按单元区间切, 同 (ei,p) 的哈希只算一次。
 * v41_eworker 的定义在 core_v41.h(常驻线程池 core_v41_epool.c 也要用)。 */
/* ★任务对象与行缓冲常驻(2026-09-18, 解码整步 graph)★: 以前每次前向都 free + malloc 一份。graph 里的 memcpy 节点
 * 记的是 raw[ei] 的**地址**(每步重放读那一刻的内容), 所以这块内存的地址在整个会话里不许变 —— 一次按 cap_tok 行
 * 分配到底, 每步只改 n。raw 用 pinned(ds4_gpu_host_alloc): 分页内存的 cudaMemcpyAsync 会退化成同步拷贝, 进不了图。 */
/* ★取行按 engram 层分轮, 单元按"一次 pread"切(2026-09-18, 盘的尺定的)★
 * 病: 原来一轮读齐两层 48 行 = 每线程 2 次串行 O_DIRECT 读, 实测一轮 2.6 ms; 而解码整步 graph 只能把它藏在 L0 的 1.2 ms 后面,
 * 于是 L1 前 GPU 干等 1.2~1.4 ms/步。盘的地板(gguf-tools/bench/pread_lat): 单次 4 KB 随机读 200 µs, 96 次并发一轮 1.1~1.5 ms ——
 * 一轮读两层不可能藏进 1.2 ms。
 * 修: 第 0 轮只读 L1 那一层(24 行 = 48 次单读, 48 线程各一次 ≈ 一个盘时延), 收完立刻提交下一层的轮 —— 它到 L14 前有 13 ms 余量。
 * 单元 = (位置, 列, 半边): 半边 0 读权重 256 B、半边 1 读缩放 8 B, 一线程一次 pread, 并发拉满。 */
typedef struct {
    uint32_t n_eng, n, cols, HD, nsc, cap;
    uint8_t *raw[DS4_V41_MAX_ENGRAM];      /* [cap][cols][HD+nsc], pinned; 本轮只用前 n 行 */
    int64_t *rows[DS4_V41_MAX_ENGRAM];     /* [cap][cols] */
    v41_eworker w[V41_EGATHER_THREADS]; uint32_t nth;
    uint8_t *fallback_bounce;   /* 没线程池时本线程用的 O_DIRECT 落脚点 */
    const ds4_engine *e;        /* 提交下一层的轮要它(回调里没有引擎指针) */
    uint32_t cur;               /* 池里正在跑/刚跑完的轮 = 第几个 engram 层 */
    int started[DS4_V41_MAX_ENGRAM], joined[DS4_V41_MAX_ENGRAM], err;
    double t_submit[DS4_V41_MAX_ENGRAM];   /* 各轮提交时刻(账: 提交→完成 = 这一轮 pread 的真实时长) */
    /* graph 路的标志(pinned, GPU 直接读): [ei] = 第 ei 层的行已就绪(值 = 步序号), [n_eng] = 本步序号 want, [n_eng+1] = GPU 自旋超时 */
    int32_t *flags; uint32_t seq;
    /* io_uring 通道(core_v41_ering.c): 有它就不走线程池 —— 主线程算哈希、一次提交整轮的读, 没有惊群(一轮 2.1 → ~0.3 ms) */
    v41_ering *ring; v41_ering_req *reqs;
} v41_ejob;

void *v41_eworker_run(void *arg) {
    v41_eworker *wk = (v41_eworker *)arg;
    const ds4_v41_state *st = wk->st; const v41_ejob *J = (const v41_ejob *)st->ejob; const ds4_v41_cfg *v = &g_ds4_v41;
    const uint32_t ei = J->cur, stride = J->HD + J->nsc;
    int64_t rows[64]; uint32_t last_p = UINT32_MAX;
    for (uint64_t u = wk->u0; u < wk->u1; u++) {
        const uint32_t part = (uint32_t)(u & 1u), rem = (uint32_t)(u >> 1);   /* 半边: 0 权重 / 1 缩放 */
        const uint32_t p = rem / J->cols, c = rem % J->cols;
        if (p != last_p) {
            v41_engram_hash(wk->e, st, ei, st->pos0 + p, rows);
            /* 行号表只给对拍夹具/指纹看; 两个线程可能算同一个 p, 写的是同一组值 */
            if (part == 0) memcpy(J->rows[ei] + (size_t)p * J->cols, rows, (size_t)J->cols * sizeof(int64_t));
            last_p = p;
        }
        const int64_t r = rows[c];
        if (r < 0 || (uint64_t)r >= v->engram_rows[ei]) { wk->err = 1; return NULL; }
        uint8_t *dst = J->raw[ei] + ((size_t)p * J->cols + c) * stride;
        const int fd = st->eshard[ei].fd;
        const uint64_t off = part ? v->engram_scale_off[ei] + (uint64_t)r * J->nsc : v->engram_weight_off[ei] + (uint64_t)r * J->HD;
        const uint32_t len = part ? J->nsc : J->HD;
        if (part) dst += J->HD;
        if (st->eshard[ei].dio) {
            if (!v41_edio_pread(fd, wk->bounce, dst, off, len)) { wk->err = 2; return NULL; }
        } else if (pread(fd, dst, len, (off_t)off) != (ssize_t)len) { wk->err = 2; return NULL; }
    }
    return NULL;
}

static void v41_ejob_free(ds4_v41_state *st) {
    v41_ejob *J = (v41_ejob *)st->ejob;
    if (!J) return;
    for (uint32_t ei = 0; ei < J->n_eng; ei++)
        if (J->started[ei] && !J->joined[ei]) { if (J->ring) (void)v41_ering_wait(J->ring); else v41_epool_wait(); }   /* 收完在飞的那轮再走 */
    v41_ering_close(J->ring); free(J->reqs);
    free(J->fallback_bounce);
    for (uint32_t i = 0; i < DS4_V41_MAX_ENGRAM; i++) { ds4_gpu_host_free(J->raw[i]); free(J->rows[i]); }
    ds4_gpu_host_free(J->flags);
    free(J); st->ejob = NULL;
}

/* 建一次: 行缓冲按 cap_tok 行开(地址在会话里不变, graph 的 memcpy 节点靠它) */
static v41_ejob *v41_ejob_create(ds4_v41_state *st) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    v41_ejob *J = xmalloc(sizeof *J); memset(J, 0, sizeof *J);
    if (!v41_epool_threads() && posix_memalign((void **)&J->fallback_bounce, V41_EDIO_ALIGN, 2u * V41_EDIO_ALIGN) != 0) {
        fprintf(stderr, "ds4: engram O_DIRECT 落脚点分配失败\n"); free(J); return NULL;
    }
    J->n_eng = v->n_engram; J->cap = st->cap_tok; J->cols = (v->engram_max_ngram - 1) * v->engram_heads;
    J->HD = v->engram_head_dim; J->nsc = J->HD / 32u;
    st->ejob = J;
    for (uint32_t ei = 0; ei < J->n_eng; ei++) {
        if (!v41_engram_open_shard(st, ei)) return NULL;
        if (v->engram_weight_off[ei] + v->engram_rows[ei] * J->HD > st->eshard[ei].size ||
            v->engram_scale_off[ei] + v->engram_rows[ei] * J->nsc > st->eshard[ei].size) { fprintf(stderr, "ds4: engram 表偏移越界\n"); return NULL; }
        J->raw[ei] = ds4_gpu_host_alloc((uint64_t)J->cap * J->cols * (J->HD + J->nsc));
        J->rows[ei] = xmalloc((size_t)J->cap * J->cols * sizeof(int64_t));
        if (!J->raw[ei]) { fprintf(stderr, "ds4: engram 行缓冲(pinned)分配失败\n"); return NULL; }
    }
    J->flags = ds4_gpu_host_alloc((uint64_t)(DS4_V41_MAX_ENGRAM + 2) * sizeof(int32_t));
    if (!J->flags) { fprintf(stderr, "ds4: engram 标志槽(pinned)分配失败\n"); return NULL; }
    memset(J->flags, 0, (DS4_V41_MAX_ENGRAM + 2) * sizeof(int32_t));
    /* io_uring: 队列 512(预填一轮 24576 次读分批续发; 解码一轮 48 次一批就走)。开不了就退线程池, 话说在日志里。 */
    J->ring = v41_ering_open(512u);
    if (J->ring) J->reqs = xmalloc((size_t)J->cap * J->cols * 2u * sizeof(v41_ering_req));
    fprintf(stderr, "ds4: [engram] 取行走 %s\n", J->ring ? "io_uring(主线程一次提交整轮)" : "线程池(io_uring 不可用)");
    return J;
}

/* io_uring 版的一轮: 主线程算哈希、填请求(一行两条: 权重 256 B、缩放 8 B)、一次提交 */
static bool v41_ejob_submit_ring(ds4_v41_state *st, uint32_t ei) {
    v41_ejob *J = (v41_ejob *)st->ejob; const ds4_v41_cfg *v = &g_ds4_v41;
    const uint32_t stride = J->HD + J->nsc;
    uint32_t nreq = 0;
    for (uint32_t p = 0; p < J->n; p++) {
        int64_t *rows = J->rows[ei] + (size_t)p * J->cols;
        v41_engram_hash(J->e, st, ei, st->pos0 + p, rows);
        for (uint32_t c = 0; c < J->cols; c++) {
            const int64_t r = rows[c];
            if (r < 0 || (uint64_t)r >= v->engram_rows[ei]) { J->err = 1; return false; }
            uint8_t *dst = J->raw[ei] + ((size_t)p * J->cols + c) * stride;
            const int fd = st->eshard[ei].fd, dio = st->eshard[ei].dio;
            J->reqs[nreq++] = (v41_ering_req){ fd, dio, J->HD, v->engram_weight_off[ei] + (uint64_t)r * J->HD, dst };
            J->reqs[nreq++] = (v41_ering_req){ fd, dio, J->nsc, v->engram_scale_off[ei] + (uint64_t)r * J->nsc, dst + J->HD };
        }
    }
    return v41_ering_submit(J->ring, J->reqs, nreq);
}

/* ---- graph 路的三步(core_decode_graph.c 调; 直发路不用) ----
 * arm: 发图前把本步序号写进 want 槽(图里每个 engram 层前的自旋核等 flag[ei] ≥ want);
 * serve: 发图后、sync 前, 主机按层收 pread(收完一层就置位一层, GPU 那边 1 µs 内继续);
 * err: sync 后查 GPU 自旋有没有超时(主机这边取行失败没置位就会超时放行, 这一步的输出不能用)。 */
bool v41_engram_graph_arm(ds4_v41_state *st) {
    v41_ejob *J = (v41_ejob *)st->ejob;
    if (!J) return g_ds4_v41.n_engram == 0;
    J->seq++;
    J->flags[J->n_eng + 1] = 0;
    J->flags[J->n_eng] = (int32_t)J->seq;
    __sync_synchronize();
    return true;
}
static bool v41_ejob_wait(ds4_v41_state *st, uint32_t ei);
bool v41_engram_graph_serve(ds4_v41_state *st) {
    v41_ejob *J = (v41_ejob *)st->ejob;
    if (!J) return g_ds4_v41.n_engram == 0;
    for (uint32_t ei = 0; ei < J->n_eng; ei++) {
        if (!v41_ejob_wait(st, ei)) return false;   /* 取行失败: 不置位 ⇒ GPU 超时放行, 调用方按 err 停车 */
        __sync_synchronize();                       /* 行先落内存, 标志后置位 */
        J->flags[ei] = (int32_t)J->seq;
        __sync_synchronize();
    }
    return true;
}
int v41_engram_graph_err(const ds4_v41_state *st) {
    const v41_ejob *J = (const v41_ejob *)st->ejob;
    return J ? J->flags[J->n_eng + 1] : 0;
}

/* 提交第 ei 层那一轮(池一次只跑一轮; 调用方保证上一轮已收)。单元数 = 行数 × 列 × 2 个半边。 */
static bool v41_ejob_submit_round(ds4_v41_state *st, uint32_t ei) {
    v41_ejob *J = (v41_ejob *)st->ejob;
    J->cur = ei; J->t_submit[ei] = now_sec();
    if (J->ring) { J->started[ei] = 1; J->joined[ei] = 0; return v41_ejob_submit_ring(st, ei); }
    const uint64_t U = (uint64_t)J->n * J->cols * 2u;
    /* ★交给常驻线程池(core_v41_epool.c)★: 以前这里每步 pthread_create 48 个线程 + 48 次
     * posix_memalign, nsys 量到每步 2.59 ms 的 GPU 空转就是这段 —— 而真正的 IO 只有几十次 4 KB 读。
     * 池起不来就本线程同步做完(慢但正确)。 */
    const uint32_t pool = v41_epool_threads();
    J->nth = pool ? (U < pool ? (uint32_t)U : pool) : 1u;
    for (uint32_t t = 0; t < J->nth; t++)
        J->w[t] = (v41_eworker){ J->e, st, U * t / J->nth, U * (t + 1) / J->nth, 0, NULL };
    J->started[ei] = 1; J->joined[ei] = 0;
    if (!pool) {   /* 没池: 本线程一口气做完 */
        J->w[0].bounce = J->fallback_bounce;
        v41_eworker_run(&J->w[0]);
        if (J->w[0].err) J->err = J->w[0].err;
        J->joined[ei] = 1;
        return J->err == 0;
    }
    return v41_epool_submit(J->w, J->nth);
}

bool v41_engram_prefetch(ds4_engine *e, ds4_v41_state *st) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    if (v->n_engram == 0) return true;
    v41_ejob *J = (v41_ejob *)st->ejob;
    if (J) for (uint32_t ei = 0; ei < J->n_eng; ei++)   /* 上一步的轮还在跑(不该发生: 每层都收过) */
        if (J->started[ei] && !J->joined[ei]) { if (J->ring) (void)v41_ering_wait(J->ring); else v41_epool_wait(); J->joined[ei] = 1; }
    if (!J && !(J = v41_ejob_create(st))) return false;
    if (st->n > J->cap) { fprintf(stderr, "ds4: engram 行缓冲 %u 行装不下本块 %u 行\n", J->cap, st->n); return false; }
    J->n = st->n; J->e = e; J->err = 0;
    for (uint32_t ei = 0; ei < J->n_eng; ei++) { J->started[ei] = 0; J->joined[ei] = 0; }
    return v41_ejob_submit_round(st, 0);   /* 只发第一层的; 后面的轮由收上一轮的人接着发 */
}

/* 收第 ei 层那一轮(没发过就现发), 收完立刻把下一层的轮发出去 —— 它到那一层之前有十几毫秒可以慢慢读。 */
static bool v41_ejob_wait(ds4_v41_state *st, uint32_t ei) {
    v41_ejob *J = (v41_ejob *)st->ejob;
    if (!J || ei >= J->n_eng) return false;
    if (!J->started[ei]) {
        for (uint32_t k = 0; k < ei; k++)
            if (J->started[k] && !J->joined[k]) { if (J->ring) (void)v41_ering_wait(J->ring); else v41_epool_wait(); J->joined[k] = 1; }
        if (!v41_ejob_submit_round(st, ei)) return false;
    }
    if (!J->joined[ei]) {
        const double t0 = now_sec();
        if (J->ring) { if (!v41_ering_wait(J->ring)) J->err = 2; }
        else v41_epool_wait();
        const double t1 = now_sec();
        if (J->n == 1u) {   /* 只记解码步(预填一轮几百 ms); 提交→完成只记第 0 轮(它才在关键路径上) */
            st->eg_wait_s += t1 - t0;
            if (ei == 0) { st->eg_job_s += t1 - J->t_submit[0]; st->eg_n++; if (st->eg_t_launch > 0.0) st->eg_enter_s += t0 - st->eg_t_launch; }
        }
        for (uint32_t t = 0; t < J->nth; t++) if (J->w[t].err) J->err = J->w[t].err;
        J->joined[ei] = 1;
        if (ei + 1u < J->n_eng && !J->started[ei + 1u] && !v41_ejob_submit_round(st, ei + 1u)) return false;
    }
    if (J->err) fprintf(stderr, "ds4: engram 取行失败(%s)\n", J->err == 1 ? "行号越界" : "pread 短读");
    return J->err == 0;
}

void v41_engram_close(ds4_v41_state *st) {
    if (st->eg_n) fprintf(stderr, "ds4: [engram] 解码步取行 %u 轮: 提交→完成均 %.2f ms, 到 engram 层时真等均 %.2f ms, launch→host 节点开跑均 %.2f ms\n",
                          st->eg_n, st->eg_job_s / st->eg_n * 1e3, st->eg_wait_s / st->eg_n * 1e3, st->eg_enter_s / st->eg_n * 1e3);
    v41_ejob_free(st);
    for (uint32_t i = 0; i < DS4_V41_MAX_ENGRAM; i++) {
        if (st->eshard[i].fd >= 0) close(st->eshard[i].fd);
        st->eshard[i].fd = -1;
        if (st->eraw[i]) { ds4_gpu_tensor_free(st->eraw[i]); st->eraw[i] = NULL; }
    }
    if (st->erows) { ds4_gpu_tensor_free(st->erows); st->erows = NULL; }
    if (st->ekv) { ds4_gpu_tensor_free(st->ekv); st->ekv = NULL; }
}

/* 温 0 不确定定位(2026-09-15 段 0): 同一份输入跑两遍, 比这三个 64 位指纹就知道病在哪一段 ——
 * rows 变 = 哈希/hist 路(主机纯计算, 不该变); raw 变 = 盘上读回的字节(pread/O_DIRECT 路);
 * 前两个都不变而 hc 变 = GPU 路(dequant 核 / wkv mmap 脏读 / 门核)。FNV-1a 64。 */
static uint64_t v41_fnv1a(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p; uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

static void v41_engram_fingerprint(const ds4_v41_state *st, const v41_ejob *J, uint32_t il, uint32_t ei) {
    const uint32_t n = st->n, cols = J->cols, stride = J->HD + J->nsc;
    const uint64_t cnt = (uint64_t)n * DS4_N_HC * DS4_N_EMBD;
    float *buf = xmalloc((size_t)cnt * 4);
    ds4_gpu_synchronize();
    const uint64_t hhc = ds4_gpu_tensor_read(st->hc, 0, buf, cnt * 4) ? v41_fnv1a(buf, (size_t)cnt * 4) : 0;
    free(buf);
    fprintf(stderr, "[v41-prof] L%02u engram 指纹 rows %016llx raw %016llx hc %016llx\n", il,
            (unsigned long long)v41_fnv1a(J->rows[ei], (size_t)n * cols * sizeof(int64_t)),
            (unsigned long long)v41_fnv1a(J->raw[ei], (size_t)n * cols * stride),
            (unsigned long long)hhc);
}

bool v41_engram(ds4_engine *e, ds4_v41_state *st, uint32_t il) {
    const ds4_v41_cfg *v = &g_ds4_v41;
    const int16_t ei = v->engram_index_of[il];
    if (ei < 0) return true;
    if (!st->ejob && !v41_engram_prefetch(e, st)) return false;   /* 没预取(不该发生)就现取 */
    const v41_ejob *J = (const v41_ejob *)st->ejob;
    const ds4_model *m = &e->model; const ds4_layer_weights *l = &e->weights.layer[il];
    const uint32_t n = st->n, E = DS4_N_EMBD, HC = DS4_N_HC, HD = J->HD, cols = J->cols, stride = J->HD + J->nsc;
    const uint64_t in_dim = (uint64_t)cols * HD, out_dim = (uint64_t)(HC + 1) * E;
    if (!st->erows) {
        for (uint32_t k = 0; k < J->n_eng; k++) st->eraw[k] = ds4_gpu_tensor_alloc((uint64_t)st->cap_tok * cols * stride);
        st->erows = ds4_gpu_tensor_alloc((uint64_t)st->cap_tok * in_dim * 4);
        st->ekv = ds4_gpu_tensor_alloc((uint64_t)st->cap_tok * out_dim * 4);
    }
    for (uint32_t k = 0; k < J->n_eng; k++) if (!st->eraw[k]) return false;
    if (!st->erows || !st->ekv) return false;
    if (st->graph) {
        /* ★捕获中★: 不在主机上等、不做同步拷贝(二者都会作废捕获)。每个 engram 层前放一个自旋小核等主机置位本层的标志
         * (主机在 v41_engram_graph_serve 里收完这一层的 pread 就置位) + 零拷贝小核把本层的行搬进设备。
         * ★不用 host 节点★: GB10 上每个 host 节点留 0.6~0.9 ms 的洞(驱动回调线程的唤醒/恢复), 自旋核 1 µs 恢复。 */
        if (!(st->egraph_uploaded & (1 << ei))) {
            if (!ds4_gpu_host_flag_wait(&J->flags[ei], &J->flags[J->n_eng], &J->flags[J->n_eng + 1])) return false;
            if (!ds4_gpu_tensor_write_zerocopy(st->eraw[ei], 0, J->raw[ei], (uint64_t)n * cols * stride)) return false;
            st->egraph_uploaded |= 1 << ei;
        }
    } else {
        if (!v41_ejob_wait(st, (uint32_t)ei)) return false;
        if (!ds4_gpu_tensor_write(st->eraw[ei], 0, J->raw[ei], (uint64_t)n * cols * stride)) return false;
    }
    if (st->dump_prefix) {   /* 对拍夹具: 行号落 <prefix>.erows_Lnn.txt(每行一个位置, 24 个全局行号), 与 Python hash_ids 直接 diff */
        char p[4400]; snprintf(p, sizeof p, "%s.erows_L%02u.txt", st->dump_prefix, il);
        FILE *df = fopen(p, "w");
        if (df) { for (uint32_t q = 0; q < n; q++) for (uint32_t c = 0; c < cols; c++) fprintf(df, "%lld%c", (long long)J->rows[ei][(size_t)q * cols + c], c + 1 == cols ? '\n' : ' '); fclose(df); }
    }
    if (!ds4_gpu_v41_engram_rows_tensor(st->erows, st->eraw[ei], n * cols, HD)) return false;
    /* wkv(盘上就是官方 FP8: e4m3 + 32×32 块缩放, clear.md C1) → 门 → hc 就地 */
    if (!ds4_gpu_v41_matmul_fp8blk_tensor(st->ekv, m->map, m->size, l->engram_wkv->abs_offset, in_dim, out_dim, st->erows, n)) return false;
    if (!ds4_gpu_v41_round_bf16_tensor(st->ekv, (uint64_t)n * out_dim)) return false;
    if (!ds4_gpu_v41_engram_gate_tensor(st->hc, st->ekv, m->map, m->size, l->engram_q->abs_offset, l->engram_k->abs_offset, E, HC, n, DS4_RMS_EPS)) return false;
    if (st->dump_prefix) {   /* 对拍夹具: engram 后的 hc [n][HC][E] 落 <prefix>.hce_Lnn.bin(对 Python engram 模块输出) */
        char p[4400]; snprintf(p, sizeof p, "%s.hce_L%02u.bin", st->dump_prefix, il);
        float *buf = xmalloc((size_t)n * HC * E * 4); ds4_gpu_synchronize();
        if (ds4_gpu_tensor_read(st->hc, 0, buf, (uint64_t)n * HC * E * 4)) { FILE *f = fopen(p, "wb"); if (f) { fwrite(buf, 4, (size_t)n * HC * E, f); fclose(f); } }
        free(buf);
    }
    if (g_ds4_v41_prof) v41_engram_fingerprint(st, J, il, (uint32_t)ei);
    return true;
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_v41_engram_nonempty_tu;
