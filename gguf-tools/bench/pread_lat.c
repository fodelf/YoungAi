/* pread_lat.c — 盘的尺(2026-09-18): engram 取行 = 每步 96 次 4 KB 对齐的 O_DIRECT 随机小读, 实测 48 线程一轮 2.6 ms,
 * 而解码整步 graph 只能把它藏在 L0 后面 1.2 ms 里 —— 剩下 1.4 ms/步 GPU 干等。这把尺回答两个问题:
 *   ① 单次 4 KB O_DIRECT 随机读的时延是多少(盘本身的地板);
 *   ② N 个线程各发 K 次, 一轮壁钟多少 —— 是并发不够(线程唤醒/排队)还是盘本身慢。
 * 用法: pread_lat <文件> [线程数=48] [每线程几次=2] [轮数=50]
 * 编译: cc -O2 -pthread -o pread_lat pread_lat.c   (Linux; O_DIRECT)
 * 读法: 单次时延 ~100 µs 而 48 线程一轮 2.6 ms ⇒ 病在线程/排队, 换 io_uring 一次提交; 单次就 1 ms ⇒ 盘慢, 换读法也救不了。 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

typedef struct { int fd; uint64_t size; uint32_t k; uint8_t *buf; uint64_t *offs; int err; } worker;
static void *run(void *arg) {
    worker *w = (worker *)arg;
    for (uint32_t i = 0; i < w->k; i++)
        if (pread(w->fd, w->buf, 4096, (off_t)w->offs[i]) != 4096) { w->err = 1; return NULL; }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "用法: pread_lat <文件> [线程数] [每线程几次] [轮数]\n"); return 1; }
    const uint32_t nth = argc > 2 ? (uint32_t)atoi(argv[2]) : 48u, k = argc > 3 ? (uint32_t)atoi(argv[3]) : 2u;
    const uint32_t rounds = argc > 4 ? (uint32_t)atoi(argv[4]) : 50u;
    int fd = open(argv[1], O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open O_DIRECT"); return 1; }
    struct stat sb; if (fstat(fd, &sb) != 0) { perror("fstat"); return 1; }
    const uint64_t nblk = (uint64_t)sb.st_size / 4096u;
    /* ① 单线程: 200 次随机 4 KB 读, 报中位/均值 */
    uint8_t *b1; if (posix_memalign((void **)&b1, 4096, 4096) != 0) return 1;
    double lat[200];
    for (int i = 0; i < 200; i++) {
        const uint64_t off = (rnd() % nblk) * 4096u;
        const double t0 = now();
        if (pread(fd, b1, 4096, (off_t)off) != 4096) { perror("pread"); return 1; }
        lat[i] = now() - t0;
    }
    double sum = 0; for (int i = 0; i < 200; i++) sum += lat[i];
    for (int i = 0; i < 200; i++) for (int j = i + 1; j < 200; j++) if (lat[j] < lat[i]) { double t = lat[i]; lat[i] = lat[j]; lat[j] = t; }
    printf("单线程 4 KB O_DIRECT 随机读: 中位 %.0f us, 均 %.0f us, p90 %.0f us, 最大 %.0f us\n", lat[100] * 1e6, sum / 200 * 1e6, lat[180] * 1e6, lat[199] * 1e6);
    /* ② 每轮起 nth 个线程各读 k 次(与引擎线程池"一轮"同形; 线程创建开销也算进去, 常驻池只会更快) */
    worker *w = calloc(nth, sizeof *w); pthread_t *th = calloc(nth, sizeof *th);
    for (uint32_t t = 0; t < nth; t++) {
        w[t].fd = fd; w[t].size = sb.st_size; w[t].k = k; w[t].offs = calloc(k, sizeof(uint64_t));
        if (posix_memalign((void **)&w[t].buf, 4096, 4096) != 0) return 1;
    }
    double best = 1e9, tot = 0;
    for (uint32_t r = 0; r < rounds; r++) {
        for (uint32_t t = 0; t < nth; t++) for (uint32_t i = 0; i < k; i++) w[t].offs[i] = (rnd() % nblk) * 4096u;
        const double t0 = now();
        for (uint32_t t = 0; t < nth; t++) pthread_create(&th[t], NULL, run, &w[t]);
        for (uint32_t t = 0; t < nth; t++) pthread_join(th[t], NULL);
        const double dt = now() - t0;
        tot += dt; if (dt < best) best = dt;
    }
    printf("%u 线程 × %u 次(含起线程): 一轮均 %.2f ms, 最好 %.2f ms  (= %u 次读, 每次摊 %.0f us)\n",
           nth, k, tot / rounds * 1e3, best * 1e3, nth * k, tot / rounds / (nth * k) * 1e6);
    return 0;
}
