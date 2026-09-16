/* core_v41_epool.c — engram 取行的常驻线程池(single.md, 2026-09-16)。
 *
 * 病(nsys 时间线定的): 每步解码 GPU 有 5 ms 空转, 其中最大的一笔 **2.59 ms 在"每步开头的 H2D 之后、
 * embed 核之前"** —— 那一段主机正在 `v41_engram_prefetch` 里, 而它每步都要
 * `pthread_create` **48 个线程** + 48 次 `posix_memalign` + 3 次 malloc/free。
 * 解码 n=1 时取行任务正好是 2 层 × 24 行 = 48 个单元, 于是线程数拉满 48;
 * 而真正的 IO 只有 48 次 4 KB 对齐读(NVMe 并行, 几百微秒)。**开销全在起线程上。**
 *
 * 修: 线程起一次就不散, 每步只"填参数 + 广播一个条件变量"; bounce 落脚点与行缓冲同样只分配一次。
 * 为什么单独一个文件: core_v41_engram.c 已经 252 行, 而池是一件独立的事(生命周期与前向无关)。
 *
 * 并发结构一个没动(还是一线程一行、各写自己那段 J->raw) —— 所以它既不引入也不修复
 * E0 那个"温 0 四跑四个 PPL"的不确定性(speed.md 段 0), 那是另一条线。
 *
 * 出错会怎样: 池起不来(线程创建失败)就退回"本线程同步做完全部单元", 慢但正确, 不停车。 */
#include "core_internal.h"
#ifndef DS4_NO_GPU

typedef struct {
    pthread_t th[V41_EGATHER_THREADS];
    pthread_mutex_t mu;
    pthread_cond_t cv_work, cv_done;
    uint8_t *bounce[V41_EGATHER_THREADS];
    v41_eworker *job;        /* 本轮的任务数组(指向调用方的 J->w), NULL = 没活 */
    uint32_t nth;            /* 池里线程数 */
    uint32_t njob;           /* 本轮要做的单元组数(= 调用方填了几个 w) */
    uint32_t seq;            /* 轮次: worker 用它判断"这是新活" */
    uint32_t pending;        /* 还没干完的 */
    int stop, ready;
} v41_epool;

static v41_epool g_epool;
/* 一个进程同时只有一条解码流(实例锁保证), 所以池是全局唯一的; 谁先用谁初始化。 */
static pthread_once_t g_epool_once = PTHREAD_ONCE_INIT;

static void *v41_epool_worker(void *arg) {
    const uint32_t id = (uint32_t)(uintptr_t)arg;
    uint32_t seen = 0;
    for (;;) {
        pthread_mutex_lock(&g_epool.mu);
        while (!g_epool.stop && g_epool.seq == seen) pthread_cond_wait(&g_epool.cv_work, &g_epool.mu);
        if (g_epool.stop) { pthread_mutex_unlock(&g_epool.mu); return NULL; }
        seen = g_epool.seq;
        v41_eworker *job = g_epool.job;
        const uint32_t njob = g_epool.njob;
        pthread_mutex_unlock(&g_epool.mu);
        if (job && id < njob) { job[id].bounce = g_epool.bounce[id]; v41_eworker_run(&job[id]); }
        pthread_mutex_lock(&g_epool.mu);
        if (g_epool.pending) g_epool.pending--;
        if (g_epool.pending == 0) pthread_cond_broadcast(&g_epool.cv_done);
        pthread_mutex_unlock(&g_epool.mu);
    }
}

static void v41_epool_init(void) {
    pthread_mutex_init(&g_epool.mu, NULL);
    pthread_cond_init(&g_epool.cv_work, NULL);
    pthread_cond_init(&g_epool.cv_done, NULL);
    for (uint32_t t = 0; t < V41_EGATHER_THREADS; t++) {
        /* bounce: O_DIRECT 的落脚点, 一行最多跨 2 个 4 KB 块 */
        if (posix_memalign((void **)&g_epool.bounce[t], V41_EDIO_ALIGN, 2u * V41_EDIO_ALIGN) != 0) break;
        if (pthread_create(&g_epool.th[t], NULL, v41_epool_worker, (void *)(uintptr_t)t) != 0) { free(g_epool.bounce[t]); g_epool.bounce[t] = NULL; break; }
        g_epool.nth = t + 1u;
    }
    g_epool.ready = g_epool.nth > 0;
    if (!g_epool.ready) fprintf(stderr, "ds4: [v41] engram 线程池起不来, 退回单线程同步取行(慢但正确)\n");
}

uint32_t v41_epool_threads(void) {
    (void)pthread_once(&g_epool_once, v41_epool_init);
    return g_epool.ready ? g_epool.nth : 0u;
}

/* 提交 njob 个单元组(w[0..njob) 已由调用方填好 u0/u1/e/st)。不阻塞。 */
bool v41_epool_submit(v41_eworker *w, uint32_t njob) {
    if (!v41_epool_threads() || njob == 0 || njob > g_epool.nth) return false;
    pthread_mutex_lock(&g_epool.mu);
    g_epool.job = w;
    g_epool.njob = njob;
    g_epool.pending = g_epool.nth;   /* 全部线程都要醒一次(多出来的空转一圈), 这样 pending 归零就是"这轮完了" */
    g_epool.seq++;
    pthread_cond_broadcast(&g_epool.cv_work);
    pthread_mutex_unlock(&g_epool.mu);
    return true;
}

void v41_epool_wait(void) {
    if (!g_epool.ready) return;
    pthread_mutex_lock(&g_epool.mu);
    while (g_epool.pending) pthread_cond_wait(&g_epool.cv_done, &g_epool.mu);
    g_epool.job = NULL;
    pthread_mutex_unlock(&g_epool.mu);
}
#endif /* !DS4_NO_GPU */
typedef int ds4_core_v41_epool_nonempty_tu;
