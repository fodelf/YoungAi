/* core_v41_ering.c — engram 取行的 io_uring 通道(2026-09-18, fable5 09-18 解码整步 graph 一节)。
 *
 * 说人话: 每步解码要从 203 GB 的 engram 表随机读 48 个 264 B 的行(每层 24 行, 一行 = 权重 256 B + 缩放 8 B, 两处各读一次)。
 * 盘的尺(gguf-tools/bench/pread_lat)量到: 单次 4 KB O_DIRECT 随机读 200 µs, 但 48 路并发一轮只要 ~0.25 ms ——
 * 而线程池版(48 线程一起被条件变量惊醒、再一起抢一把锁)一轮实测 2.1 ms, 藏不进 graph 里 L0 的 1.2 ms。
 * io_uring: 主线程把 48 个读请求一次塞进提交队列、一次系统调用交给内核, 内核并发下盘, 完成了再一次系统调用收 ——
 * 没有线程, 没有锁, 一轮 ≈ 一个盘时延。
 *
 * 只有 Linux 有 io_uring; 非 Linux / 内核不支持时 v41_ering_open 返回 NULL, 调用方退回线程池(core_v41_epool.c, 慢但正确)。
 * 裸系统调用, 不依赖 liburing(引擎零第三方依赖)。
 *
 * O_DIRECT 的读必须 4 KB 对齐(缓冲/偏移/长度三者), 而一行 264 B 落在任意偏移 ⇒ 读对齐超集进落脚点再 memcpy 出来
 * (一行最多跨 2 个块, 落脚点 8 KB); 非 O_DIRECT 的 fd 直接按原偏移读进目的地。
 * 出错会怎样: 哪一次读短了/错了, 整轮判失败(err 记住), 调用方停车 —— 不会拿半截行去算。 */
#include "core_internal.h"
#if !defined(DS4_NO_GPU) && defined(__linux__)
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>

#define V41_ERING_BLK 4096u

struct v41_ering {
    int fd; uint32_t cap;
    void *sq_map, *cq_map; size_t sq_map_sz, cq_map_sz; int single_map;
    struct io_uring_sqe *sqes; size_t sqes_sz;
    uint32_t *sq_head, *sq_tail, *sq_mask, *sq_array, *cq_head, *cq_tail, *cq_mask;
    struct io_uring_cqe *cqes;
    uint8_t *bounce;                 /* cap × 2 块 */
    const v41_ering_req *reqs; uint32_t n, submitted, done, inflight, batch_base; int err;
};

static int v41_uring_setup(uint32_t entries, struct io_uring_params *p) { return (int)syscall(__NR_io_uring_setup, entries, p); }
static int v41_uring_enter(int fd, uint32_t to_submit, uint32_t min_complete, uint32_t flags) {
    return (int)syscall(__NR_io_uring_enter, fd, to_submit, min_complete, flags, NULL, 0);
}

v41_ering *v41_ering_open(uint32_t cap) {
    struct io_uring_params p; memset(&p, 0, sizeof p);
    const int fd = v41_uring_setup(cap, &p);
    if (fd < 0) return NULL;
    v41_ering *r = xmalloc(sizeof *r); memset(r, 0, sizeof *r);
    r->fd = fd; r->cap = p.sq_entries;
    r->sq_map_sz = p.sq_off.array + (size_t)p.sq_entries * sizeof(uint32_t);
    r->cq_map_sz = p.cq_off.cqes + (size_t)p.cq_entries * sizeof(struct io_uring_cqe);
    r->single_map = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
    if (r->single_map && r->cq_map_sz > r->sq_map_sz) r->sq_map_sz = r->cq_map_sz;
    r->sq_map = mmap(NULL, r->sq_map_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
    if (r->sq_map == MAP_FAILED) { close(fd); free(r); return NULL; }
    if (r->single_map) r->cq_map = r->sq_map;
    else {
        r->cq_map = mmap(NULL, r->cq_map_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
        if (r->cq_map == MAP_FAILED) { munmap(r->sq_map, r->sq_map_sz); close(fd); free(r); return NULL; }
    }
    r->sqes_sz = (size_t)p.sq_entries * sizeof(struct io_uring_sqe);
    r->sqes = mmap(NULL, r->sqes_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
    if (r->sqes == MAP_FAILED) { v41_ering_close(r); return NULL; }
    uint8_t *sq = (uint8_t *)r->sq_map, *cq = (uint8_t *)r->cq_map;
    r->sq_head = (uint32_t *)(sq + p.sq_off.head); r->sq_tail = (uint32_t *)(sq + p.sq_off.tail);
    r->sq_mask = (uint32_t *)(sq + p.sq_off.ring_mask); r->sq_array = (uint32_t *)(sq + p.sq_off.array);
    r->cq_head = (uint32_t *)(cq + p.cq_off.head); r->cq_tail = (uint32_t *)(cq + p.cq_off.tail);
    r->cq_mask = (uint32_t *)(cq + p.cq_off.ring_mask); r->cqes = (struct io_uring_cqe *)(cq + p.cq_off.cqes);
    if (posix_memalign((void **)&r->bounce, V41_ERING_BLK, (size_t)r->cap * 2u * V41_ERING_BLK) != 0) { r->bounce = NULL; v41_ering_close(r); return NULL; }
    return r;
}

void v41_ering_close(v41_ering *r) {
    if (!r) return;
    if (r->sqes && r->sqes != MAP_FAILED) munmap(r->sqes, r->sqes_sz);
    if (r->cq_map && !r->single_map && r->cq_map != MAP_FAILED) munmap(r->cq_map, r->cq_map_sz);
    if (r->sq_map && r->sq_map != MAP_FAILED) munmap(r->sq_map, r->sq_map_sz);
    if (r->fd >= 0) close(r->fd);
    free(r->bounce); free(r);
}

/* 发一批: 把 [submitted, min(n, submitted+cap)) 填进提交队列交给内核(不等完成)。
 * ★严格分批★(2026-09-18 实撞): 第一版按 i % cap 选落脚点、只要在飞的少于 cap 就续发 —— 但完成是**乱序**的, 请求 i+cap 会在
 * 请求 i 还没落地时覆盖同一块落脚点(预填一轮 24576 次读远超队列 512), 行被污染: 12k 直发与基线第 1 字节就分叉、走图两跑不一致。
 * 现在只有上一批全部收齐(inflight == 0)才发下一批, 落脚点 = 批内序号, 绝不复用在飞的槽。解码一轮 48 次 = 一批, 不多付。 */
static bool v41_ering_push(v41_ering *r) {
    if (r->inflight) return false;
    uint32_t tail = *r->sq_tail, pushed = 0;
    const uint32_t mask = *r->sq_mask;
    r->batch_base = r->submitted;
    while (r->submitted < r->n && r->inflight < r->cap) {
        const uint32_t i = r->submitted, slot = i - r->batch_base;
        const v41_ering_req *q = &r->reqs[i];
        struct io_uring_sqe *s = &r->sqes[tail & mask];
        memset(s, 0, sizeof *s);
        s->opcode = IORING_OP_READ; s->fd = q->fd; s->user_data = i;
        if (q->direct) {
            const uint64_t a = q->off & ~(uint64_t)(V41_ERING_BLK - 1u);
            const uint64_t span = q->off + q->len - a;
            s->addr = (uint64_t)(uintptr_t)(r->bounce + (size_t)slot * 2u * V41_ERING_BLK);
            s->len = (uint32_t)((span + V41_ERING_BLK - 1u) / V41_ERING_BLK) * V41_ERING_BLK;
            s->off = a;
        } else { s->addr = (uint64_t)(uintptr_t)q->dst; s->len = q->len; s->off = q->off; }
        r->sq_array[tail & mask] = tail & mask;
        tail++; pushed++; r->submitted++; r->inflight++;
    }
    if (!pushed) return true;
    __atomic_store_n(r->sq_tail, tail, __ATOMIC_RELEASE);
    return v41_uring_enter(r->fd, pushed, 0, 0) >= 0;
}

bool v41_ering_submit(v41_ering *r, const v41_ering_req *reqs, uint32_t n) {
    if (!r || r->inflight) return false;   /* 上一轮还没收完就来一轮 = 调用方的顺序错了 */
    r->reqs = reqs; r->n = n; r->submitted = 0; r->done = 0; r->err = 0;
    return v41_ering_push(r);
}

/* 等这一轮全部完成(队列装不下的分批续发, 上一批收齐才发下一批); 完成的请求把行从落脚点拷到目的地 */
bool v41_ering_wait(v41_ering *r) {
    if (!r) return false;
    while (r->done < r->n) {
        if (r->inflight == 0) { if (!v41_ering_push(r)) { r->err = 1; break; } continue; }
        if (v41_uring_enter(r->fd, 0, 1, IORING_ENTER_GETEVENTS) < 0) { if (errno == EINTR) continue; r->err = 1; break; }
        uint32_t head = *r->cq_head;
        const uint32_t tail = __atomic_load_n(r->cq_tail, __ATOMIC_ACQUIRE), mask = *r->cq_mask;
        while (head != tail) {
            const struct io_uring_cqe *c = &r->cqes[head & mask];
            const uint32_t i = (uint32_t)c->user_data;
            const v41_ering_req *q = &r->reqs[i];
            if (i < r->batch_base || i - r->batch_base >= r->cap) r->err = 1;   /* 不该发生: 完成的不是本批的请求 */
            else if (q->direct) {
                const uint64_t a = q->off & ~(uint64_t)(V41_ERING_BLK - 1u);
                if (c->res < 0 || (uint64_t)c->res < q->off + q->len - a) r->err = 1;
                else memcpy(q->dst, r->bounce + (size_t)(i - r->batch_base) * 2u * V41_ERING_BLK + (q->off - a), q->len);
            } else if (c->res != (int32_t)q->len) r->err = 1;
            head++; r->done++; r->inflight--;
        }
        __atomic_store_n(r->cq_head, head, __ATOMIC_RELEASE);
    }
    return r->err == 0;
}
#else
v41_ering *v41_ering_open(uint32_t cap) { (void)cap; return NULL; }
void v41_ering_close(v41_ering *r) { (void)r; }
bool v41_ering_submit(v41_ering *r, const v41_ering_req *reqs, uint32_t n) { (void)r; (void)reqs; (void)n; return false; }
bool v41_ering_wait(v41_ering *r) { (void)r; return false; }
#endif
typedef int ds4_core_v41_ering_nonempty_tu;
