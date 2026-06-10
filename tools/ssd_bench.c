/* P0.3 (project.md): SSD read-bandwidth baseline on a big file (the GGUF),
 * bypassing the page cache via F_NOCACHE.  Decides whether IO-pattern work
 * (expert bundling, larger reads, deeper queues) can pay on this disk.
 *
 *   cc -O2 -o ssd_bench tools/ssd_bench.c -pthread
 *   ./ssd_bench gguf/DeepSeek-V4-Flash-...gguf
 *
 * Measured 2026-06-10:
 *   M4 mini (coordinator): ~2.4-2.6 GB/s for EVERYTHING (seq 16MiB, rand
 *     2MiB qd8, rand 6.75MiB qd8) => bundling/repack cannot pay there (P1.3
 *     rejected); the gather is already at the disk ceiling.
 *   M1 MacBook (worker): seq 6.7, rand 2MiB qd4-8 ~5.5, rand 6.75MiB ~6.5
 *     GB/s => the worker disk is 2.3-2.6x faster; layer rebalance toward the
 *     worker is profitable until its RAM budget binds. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <sys/stat.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

typedef struct {
    int fd;
    uint64_t fsize, rsize, n, next, seed, base;
    int seq;
} job;

static void *worker(void *a) {
    job *j = (job *)a;
    uint8_t *buf = malloc(j->rsize);
    uint64_t s = j->seed;
    if (!buf) return NULL;
    for (;;) {
        uint64_t i = __sync_fetch_and_add(&j->next, 1);
        if (i >= j->n) break;
        uint64_t off;
        if (j->seq) {
            off = j->base + i * j->rsize;
        } else {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            off = (s >> 17) % (j->fsize - j->rsize);
            off &= ~16383ULL;
        }
        if (pread(j->fd, buf, j->rsize, (off_t)off) < 0) {
            perror("pread");
            break;
        }
    }
    free(buf);
    return NULL;
}

static void run(const char *name, int fd, uint64_t fsize, uint64_t rsize,
                uint64_t total, int nth, int seq, uint64_t base) {
    job j = { fd, fsize, rsize, total / rsize, 0, (uint64_t)(now_ms() * 1e6), base, seq };
    pthread_t th[16];
    double t0 = now_ms();
    for (int i = 0; i < nth; i++) pthread_create(&th[i], NULL, worker, &j);
    for (int i = 0; i < nth; i++) pthread_join(th[i], NULL);
    double ms = now_ms() - t0;
    printf("%-22s rsize=%6.2fMiB qd=%2d total=%4lluMiB time=%7.1fms bw=%.2f GB/s\n",
           name, rsize / 1048576.0, nth, (unsigned long long)(total >> 20), ms,
           total / (ms * 1e-3) / 1e9);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <big-file>\n", argv[0]);
        return 1;
    }
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    (void)fcntl(fd, F_NOCACHE, 1);
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (40ll << 30)) {
        fprintf(stderr, "want a file >=40GiB so reads are uncached\n");
    }
    uint64_t fsize = (uint64_t)st.st_size;
    const uint64_t MB = 1048576ULL;
    run("seq 16MiB qd1", fd, fsize, 16 * MB, 2048 * MB, 1, 1, 8ULL << 30);
    run("seq 16MiB qd4", fd, fsize, 16 * MB, 2048 * MB, 4, 1, 20ULL << 30);
    run("rand 2MiB qd1", fd, fsize, 2 * MB, 1024 * MB, 1, 0, 0);
    run("rand 2MiB qd4", fd, fsize, 2 * MB, 1024 * MB, 4, 0, 0);
    run("rand 2MiB qd8", fd, fsize, 2 * MB, 2048 * MB, 8, 0, 0);
    run("rand 2MiB qd16", fd, fsize, 2 * MB, 2048 * MB, 16, 0, 0);
    run("rand 6.75MiB qd4", fd, fsize, 6912 * 1024ULL, 2048 * MB, 4, 0, 0);
    run("rand 6.75MiB qd8", fd, fsize, 6912 * 1024ULL, 2048 * MB, 8, 0, 0);
    run("rand 16MiB qd8", fd, fsize, 16 * MB, 2048 * MB, 8, 0, 0);
    close(fd);
    return 0;
}
