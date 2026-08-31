/* core_parallel.c — CPU 线程池 + parallel_for + f32 文件IO (机械拆分自 ds4.c, 重构阶段4)。 */
#include "core_internal.h"
bool write_f32_binary_file(const char *path, const float *data, uint64_t n) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open %s for writing: %s\n", path, strerror(errno));
        return false;
    }
    const size_t nw = fwrite(data, sizeof(float), (size_t)n, fp);
    const bool ok = nw == (size_t)n && fclose(fp) == 0;
    if (!ok) {
        fprintf(stderr, "ds4: failed to write %s\n", path);
        return false;
    }
    return true;
}

bool read_f32_binary_file(const char *path, float *data, uint64_t n) {
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "ds4: failed to stat %s: %s\n", path, strerror(errno));
        return false;
    }
    if (st.st_size < 0 || (uint64_t)st.st_size != n * sizeof(float)) {
        fprintf(stderr,
                "ds4: %s has size %llu bytes, expected %llu bytes\n",
                path,
                (unsigned long long)st.st_size,
                (unsigned long long)(n * sizeof(float)));
        return false;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4: failed to open %s for reading: %s\n", path, strerror(errno));
        return false;
    }
    const size_t nr = fread(data, sizeof(float), (size_t)n, fp);
    const bool ok = nr == (size_t)n && fclose(fp) == 0;
    if (!ok) {
        fprintf(stderr, "ds4: failed to read %s\n", path);
        return false;
    }
    return true;
}

#define DS4_MAX_THREADS 32

typedef struct {
    pthread_t threads[DS4_MAX_THREADS];
    pthread_mutex_t mutex;
    pthread_cond_t work_cond;
    pthread_cond_t done_cond;
    uint32_t n_threads;
    uint32_t n_workers;
    uint32_t generation;
    uint32_t done;
    bool initialized;
    bool shutdown;
    ds4_parallel_fn fn;
    void *ctx;
    uint64_t n_rows;
} ds4_thread_pool;

static ds4_thread_pool g_pool;
static __thread int g_parallel_depth;
uint32_t g_requested_threads;

static void *ds4_worker_main(void *arg) {
    const uint32_t tid = (uint32_t)(uintptr_t)arg;
    uint32_t seen_generation = 0;

    for (;;) {
        pthread_mutex_lock(&g_pool.mutex);
        while (seen_generation == g_pool.generation && !g_pool.shutdown) {
            pthread_cond_wait(&g_pool.work_cond, &g_pool.mutex);
        }
        if (g_pool.shutdown) {
            pthread_mutex_unlock(&g_pool.mutex);
            return NULL;
        }

        seen_generation = g_pool.generation;
        ds4_parallel_fn fn = g_pool.fn;
        void *ctx = g_pool.ctx;
        const uint64_t n_rows = g_pool.n_rows;
        const uint32_t n_threads = g_pool.n_threads;
        pthread_mutex_unlock(&g_pool.mutex);

        const uint64_t rows_per_thread = (n_rows + n_threads - 1) / n_threads;
        const uint64_t row0 = (uint64_t)tid * rows_per_thread;
        uint64_t row1 = row0 + rows_per_thread;
        if (row1 > n_rows) row1 = n_rows;
        if (row0 < row1) {
            g_parallel_depth++;
            fn(ctx, row0, row1);
            g_parallel_depth--;
        }

        pthread_mutex_lock(&g_pool.mutex);
        g_pool.done++;
        if (g_pool.done == g_pool.n_workers) {
            pthread_cond_signal(&g_pool.done_cond);
        }
        pthread_mutex_unlock(&g_pool.mutex);
    }
}

/* Create the persistent CPU worker pool.  Decode reuses these threads instead
 * of creating pthreads in the token loop. */
static void ds4_threads_init(void) {
    if (g_pool.initialized) return;

    pthread_once(&iq2xxs_signed_grid_once, iq2xxs_signed_grid_init);

    uint32_t n_threads = 12;
    const long online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (online_cpus > 0) {
        n_threads = online_cpus < 12 ? (uint32_t)online_cpus : 12;
    }

    if (g_requested_threads > 0) n_threads = g_requested_threads;
    if (n_threads > DS4_MAX_THREADS) n_threads = DS4_MAX_THREADS;
    if (n_threads == 0) n_threads = 1;

    pthread_mutex_init(&g_pool.mutex, NULL);
    pthread_cond_init(&g_pool.work_cond, NULL);
    pthread_cond_init(&g_pool.done_cond, NULL);
    g_pool.n_threads = n_threads;
    g_pool.n_workers = n_threads > 0 ? n_threads - 1 : 0;
    g_pool.generation = 0;
    g_pool.done = 0;
    g_pool.shutdown = false;
    g_pool.initialized = true;

    for (uint32_t i = 1; i < n_threads; i++) {
        if (pthread_create(&g_pool.threads[i], NULL, ds4_worker_main, (void *)(uintptr_t)i) != 0) {
            ds4_die("failed to create worker thread");
        }
    }
}

void ds4_threads_shutdown(void) {
    if (!g_pool.initialized) return;

    pthread_mutex_lock(&g_pool.mutex);
    g_pool.shutdown = true;
    g_pool.generation++;
    pthread_cond_broadcast(&g_pool.work_cond);
    pthread_mutex_unlock(&g_pool.mutex);

    for (uint32_t i = 1; i < g_pool.n_threads; i++) {
        pthread_join(g_pool.threads[i], NULL);
    }

    pthread_cond_destroy(&g_pool.done_cond);
    pthread_cond_destroy(&g_pool.work_cond);
    pthread_mutex_destroy(&g_pool.mutex);
    memset(&g_pool, 0, sizeof(g_pool));
}

/* Run a row-parallel CPU kernel, falling back to serial execution for small
 * jobs or nested calls where spawning more work would only add latency. */
void ds4_parallel_for_min_rows(uint64_t n_rows, ds4_parallel_fn fn, void *ctx, uint64_t min_parallel_rows) {
    ds4_threads_init();

    if (g_parallel_depth > 0 || g_pool.n_threads <= 1 || n_rows < min_parallel_rows) {
        fn(ctx, 0, n_rows);
        return;
    }

    pthread_mutex_lock(&g_pool.mutex);
    g_pool.fn = fn;
    g_pool.ctx = ctx;
    g_pool.n_rows = n_rows;
    g_pool.done = 0;
    g_pool.generation++;
    pthread_cond_broadcast(&g_pool.work_cond);

    const uint64_t rows_per_thread = (n_rows + g_pool.n_threads - 1) / g_pool.n_threads;
    uint64_t main_row1 = rows_per_thread;
    if (main_row1 > n_rows) main_row1 = n_rows;
    pthread_mutex_unlock(&g_pool.mutex);

    if (main_row1 > 0) {
        g_parallel_depth++;
        fn(ctx, 0, main_row1);
        g_parallel_depth--;
    }

    pthread_mutex_lock(&g_pool.mutex);
    while (g_pool.done < g_pool.n_workers) {
        pthread_cond_wait(&g_pool.done_cond, &g_pool.mutex);
    }
    pthread_mutex_unlock(&g_pool.mutex);
}

void ds4_parallel_for(uint64_t n_rows, ds4_parallel_fn fn, void *ctx) {
    ds4_parallel_for_min_rows(n_rows, fn, ctx, 512);
}

